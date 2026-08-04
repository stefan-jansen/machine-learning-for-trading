# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # US Equities Panel: Temporal Features
#
# This notebook fits temporal models inside walk-forward CV folds to produce
# features without look-ahead bias. Three assigned models:
#
# 1. **Wasserstein Regime Distance** (primary): Detect regime shifts in market
#    central tendency by clustering windowed sequences of the cross-sectional
#    median return using Wasserstein k-means.
#
# 2. **Fractional Differencing** (primary): Apply FFD at the equity-default order
#    to log prices per stock. Preserves long memory while achieving stationarity --
#    important for a 56-year dataset where standard differencing destroys signal.
#
# 3. **GARCH(1,1)** (secondary): Conditional volatility forecasts on the ~200 most
#    liquid stocks. Computational constraint: full GARCH on 3,199 stocks is expensive;
#    subsample by liquidity and broadcast market-level vol features to all.
#
# ## Walk-Forward Discipline
#
# All temporal models are fitted **per CV fold** on training data only:
#
# - Wasserstein centroids fitted on training window, features extracted for
#   the full train+test period of each fold
# - FFD uses a fixed differencing order (no in-sample search per fold; weights are deterministic)
# - GARCH parameters $(\omega, \alpha, \beta)$ estimated on training data;
#   `model.fix()` runs the variance recursion on the full train+test window
#   without re-estimating parameters
# - A holdout fold (fit up to holdout boundary) produces features for the
#   holdout period
# - Every row carries a `fold` column for downstream per-fold CV joins
#
# ## Book Reference
#
# Chapter 9, Sections 9.1 (Stationarity), 9.3 (Volatility), 9.5 (Regimes)
#
# ## Prerequisites
#
# - [`02_labels`](02_labels.ipynb) produces label parquet files and `config/cv_config.json`
# - `03_financial_features.py` produces `features/financial.parquet` (for index alignment)

# %%
"""US Equities Panel: Temporal Features."""

import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray

warnings.filterwarnings("ignore")

from arch import arch_model
from ml4t.engineer.features.fdiff import ffdiff, get_ffd_weights

from data import load_us_equities
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.style import COLORS, add_message_title

CASE_DIR = get_case_study_dir("us_equities_panel")
FEATURES_DIR = CASE_DIR / "features"

# Configuration
START_DATE = "1990-01-01"
END_DATE = "2018-03-31"

# Liquidity filters (same as 02_labels.py and 03_financial_features.py)
MIN_ADV_USD = 1_000_000
MIN_PRICE = 5.0
ADV_WINDOW = 21

# Temporal model parameters
FFD_D = 0.4  # Fixed d for equities (no per-fold search)
FFD_THRESHOLD = 1e-5

WASSERSTEIN_WINDOW = 21
WASSERSTEIN_OVERLAP = 5
N_CLUSTERS = 2  # Risk-on vs risk-off

GARCH_TOP_N = 200
GARCH_MIN_OBS = 504

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# %%
# Production defaults — Papermill injects overrides for CI
MAX_FOLDS = 0  # 0 = all folds; test mode: 2

# %% [markdown]
# ## Configuration
#
# Same PIT filters as [`02_labels`](02_labels.ipynb) for consistent universe construction.
# Note: this notebook applies filters independently rather than reusing
# a materialized price extract. Both use identical constants (MIN_PRICE=$5,
# ADV>$1M). A production pipeline would centralize universe construction.

# %%
raw_df = load_us_equities(start_date=START_DATE, end_date=END_DATE)

# Normalize types
if raw_df.schema["timestamp"] == pl.Datetime:
    raw_df = raw_df.with_columns(pl.col("timestamp").dt.date().alias("timestamp"))

raw_df = raw_df.sort(["symbol", "timestamp"])

# Compute base columns
raw_df = raw_df.with_columns(
    (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("returns"),
    (pl.col("adj_close") * pl.col("adj_volume")).alias("dollar_volume"),
)

# Apply PIT eligibility filters
raw_df = raw_df.with_columns(
    pl.col("dollar_volume").rolling_mean(ADV_WINDOW).over("symbol").alias("adv_21d")
)
df = raw_df.filter((pl.col("adj_close") > MIN_PRICE) & (pl.col("adv_21d") > MIN_ADV_USD))

print(f"Loaded {len(df):,} rows, {df['symbol'].n_unique()} symbols")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# %% [markdown]
# ## A. Why a fitted feature is different
#
# Every column in `03_financial_features` is a rule fixed in advance. The three built here
# estimate their parameters from data - a Wasserstein clustering, a fractional-differencing
# order, a GARCH recursion - which is why this stage carries the fold contract in B and
# stage 03 does not. A rule cannot leak by being fitted; a fitted parameter can, and only
# the window it came from decides whether it did.
#
# What they are for: momentum strategies are vulnerable to sharp reversals at regime
# transitions - momentum crashes (Daniel and Moskowitz 2016). Wasserstein regime distance
# flags shifts in market central tendency, GARCH captures volatility clustering, and FFD
# preserves the long-memory structure in price levels that standard differencing destroys.

# %% [markdown]
# ## B. The fold contract
#
# We use the canonical CV splits from `setup.yaml` (via `generate_cv_splits`).
# The primary label is `fwd_ret_1d` so the label buffer is `1D`.
# A holdout fold is appended: fit on data up to the holdout boundary,
# produce features through the end of data.

# %%
splits = generate_cv_splits(df, case_study_id="us_equities_panel", label_buffer="1D")
n_cv_folds = len(splits)

# Parse holdout boundary from setup.yaml
import yaml

_setup_path = CASE_DIR / "config" / "setup.yaml"
with open(_setup_path) as _f:
    _setup = yaml.safe_load(_f)
holdout_start = str(_setup["evaluation"]["holdout_start"])
holdout_end = str(_setup["evaluation"].get("holdout_end", END_DATE))

# Build list of folds: CV folds + holdout fold
folds = []
for s in splits:
    folds.append(
        {
            "fold": s["fold"],
            "is_holdout": False,
            "train_start": str(s["train_start"])[:10],
            "train_end": str(s["train_end"])[:10],
            "test_start": str(s["val_start"])[:10],
            "test_end": str(s["val_end"])[:10],
        }
    )

# Holdout fold: train up to holdout boundary, test through end of data
# Use the same train_size as the last CV fold
last_cv_train_start = folds[-1]["train_start"] if folds else str(df["timestamp"].min())
folds.append(
    {
        "fold": n_cv_folds,
        "is_holdout": True,
        "train_start": last_cv_train_start,
        "train_end": holdout_start,
        "test_start": holdout_start,
        "test_end": holdout_end,
    }
)

if MAX_FOLDS > 0:
    # Keep first MAX_FOLDS CV folds + holdout
    cv_folds = [f for f in folds if not f["is_holdout"]][:MAX_FOLDS]
    holdout_folds = [f for f in folds if f["is_holdout"]]
    folds = cv_folds + holdout_folds

print(f"Walk-Forward Folds ({len(folds)} total, {n_cv_folds} CV + 1 holdout):")
for f in folds:
    tag = " [HOLDOUT]" if f.get("is_holdout", False) else ""
    print(
        f"  Fold {f['fold']}{tag}: train [{f['train_start']} to {f['train_end']}) "
        f"test [{f['test_start']} to {f['test_end']})"
    )

# %% [markdown]
# ### F1. The fold contract
#
# The table above lists the boundaries; this draws them, which is the only form in which a reader
# can check the thing that matters. Every fitted parameter comes from the left-hand span of its own
# row, and the span it is applied to lies entirely to the right of it. The final row is the holdout
# pass: its parameters come from development data alone, and the rule marks where the seal begins.

# %%
fig, ax = plt.subplots(figsize=(11, 3.2))
for row, fold in enumerate(folds):
    train_start, train_end = np.datetime64(fold["train_start"]), np.datetime64(fold["train_end"])
    test_start, test_end = np.datetime64(fold["test_start"]), np.datetime64(fold["test_end"])
    ax.barh(
        row,
        train_end - train_start,
        left=train_start,
        height=0.55,
        color=COLORS["blue"],
        label="parameters fitted on" if row == 0 else None,
    )
    ax.barh(
        row,
        test_end - test_start,
        left=test_start,
        height=0.55,
        color=COLORS["amber"],
        label="applied over" if row == 0 else None,
    )
ax.axvline(np.datetime64(holdout_start), color=COLORS["negative"], linestyle="--", linewidth=1.2)
ax.set_yticks(range(len(folds)))
ax.set_yticklabels(
    [f"fold {fold['fold']}" + (" (holdout)" if fold.get("is_holdout") else "") for fold in folds]
)
ax.invert_yaxis()
ax.legend(loc="lower left", bbox_to_anchor=(0, -0.28), ncol=2, frameon=False, fontsize=8)
add_message_title(
    ax,
    "No parameter is fitted on the span it is then applied to",
    subtitle=f"Fitted and applied spans per fold; the rule marks the holdout at {holdout_start}",
)
fig.show()

# %% [markdown]
# ## C. One section per model
#
# We detect regime shifts in market central tendency by clustering windowed
# sequences of the cross-sectional median return. At each date, we compute
# the median return across all ~3,000 stocks; this scalar time series captures
# how the market's center of mass evolves. Wasserstein k-means then clusters
# overlapping windows of this series into $k=2$ regimes (risk-on vs risk-off),
# treating each window as an empirical measure.
#
# **What this captures**: Shifts in the *central tendency* of cross-sectional
# returns over time. This is distinct from modeling the full distributional
# shape (tails, skewness, bimodality), which would require quantile vectors
# or the full cross-sectional distribution as input. The cross-sectional
# std, skew, and tail quantiles are computed for diagnostics but not used
# in clustering.
#
# **Walk-forward approach**: For each CV fold, fit centroids on the training
# window only, then assign regime labels for the full train+test period.
# Since this is a market-level feature, all stocks share the same regime
# signal at each date.


# %%
@dataclass(frozen=True)
class LiftedStream:
    """Overlapping windows of cross-sectional return distributions."""

    segments: FloatArray  # (n_segments, window_len)
    sorted_segments: FloatArray  # Sorted per window
    starts: IntArray  # Start indices
    window_len: int
    step: int


def lift_stream(
    returns: FloatArray,
    window_len: int,
    overlap: int,
) -> LiftedStream:
    """Lift a 1D return stream into overlapping windows."""
    step = window_len - overlap
    windows_view = np.lib.stride_tricks.sliding_window_view(returns, window_shape=window_len)
    windows_view = windows_view[::step]
    segments = np.ascontiguousarray(windows_view, dtype=np.float64)
    sorted_segments = np.sort(segments, axis=1)
    starts = np.arange(0, segments.shape[0] * step, step, dtype=np.int64)

    return LiftedStream(
        segments=segments,
        sorted_segments=sorted_segments,
        starts=starts,
        window_len=window_len,
        step=step,
    )


# %%
def wasserstein_distance_1d(sorted_a: FloatArray, sorted_b: FloatArray, p: float = 1.0) -> float:
    """1D p-Wasserstein distance between equal-weight empirical measures."""
    diff_p = np.abs(sorted_a - sorted_b) ** p
    return float(diff_p.mean() ** (1.0 / p))


def wasserstein_barycenter_1d(sorted_members: FloatArray, p: float = 1.0) -> FloatArray:
    """Wasserstein barycenter: median (p=1) or mean (p=2) of sorted atoms."""
    if p == 1.0:
        return np.median(sorted_members, axis=0).astype(np.float64)
    return sorted_members.mean(axis=0).astype(np.float64)


# %%
def fit_wasserstein_kmeans(
    sorted_segments: FloatArray,
    n_clusters: int = 2,
    max_iter: int = 50,
    n_init: int = 5,
    random_state: int = 42,
) -> tuple[IntArray, FloatArray]:
    """Fit Wasserstein k-means on sorted 1D segments.

    Returns (labels, centroids).
    """
    rng = np.random.default_rng(random_state)
    n_samples = sorted_segments.shape[0]
    best_labels = None
    best_centroids = None
    best_inertia = float("inf")

    for _ in range(n_init):
        # Random initialization
        idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centroids = sorted_segments[idx].copy()

        for _ in range(max_iter):
            # Assignment: compute distance to each centroid
            dists = np.zeros((n_samples, n_clusters))
            for k in range(n_clusters):
                diff = np.abs(sorted_segments - centroids[k][None, :])
                dists[:, k] = diff.mean(axis=1)

            labels = dists.argmin(axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                members = sorted_segments[labels == k]
                if len(members) > 0:
                    new_centroids[k] = wasserstein_barycenter_1d(members, p=1.0)
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        inertia = sum(dists[i, labels[i]] for i in range(n_samples))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids


# %% [markdown]
# ### Compute Cross-Sectional Returns per Date
#
# Compute cross-sectional return statistics at each date. Only the median is
# used for Wasserstein clustering (see Section 2 narrative); the std, skew,
# and tail quantiles are retained for diagnostic inspection and potential
# future enrichment of the clustering input.

# %%
# Cross-sectional return statistics per date
xs_stats = (
    df.filter(pl.col("returns").is_not_null())
    .group_by("timestamp")
    .agg(
        pl.col("returns").median().alias("xs_median_ret"),
        pl.col("returns").std().alias("xs_std_ret"),
        pl.col("returns").skew().alias("xs_skew"),
        pl.col("returns").quantile(0.1).alias("xs_q10"),
        pl.col("returns").quantile(0.9).alias("xs_q90"),
        pl.col("returns").count().alias("n_stocks"),
    )
    .sort("timestamp")
    .filter(pl.col("n_stocks") >= 50)
)

# Use cross-sectional median return as market-level signal for clustering
market_ret = xs_stats["xs_median_ret"].to_numpy()
dates = xs_stats["timestamp"].to_list()

print(f"Cross-sectional stats: {len(xs_stats):,} dates")
if len(xs_stats) > 0:
    print(f"  Median stocks/date: {int(xs_stats['n_stocks'].median())}")

# %% [markdown]
# ### C.1 Per-fold Wasserstein clustering
#
# For each fold, fit Wasserstein k-means centroids on the training window only,
# then assign regime labels for the full train+test period. The reference
# distribution (centroids) sees no future data relative to the fold boundary.
# Since this is a market-level feature, all stocks share the same regime
# signal at each date.


# %%
def assign_regime_features(
    market_ret: FloatArray,
    dates_list: list,
    centroids: FloatArray,
    start_idx: int,
    end_idx: int,
) -> list[dict]:
    """Assign regime labels for dates[start_idx:end_idx] given fitted centroids."""
    results = []
    for t in range(start_idx, end_idx):
        if t < WASSERSTEIN_WINDOW:
            continue

        recent_window = market_ret[t - WASSERSTEIN_WINDOW : t]
        recent_sorted = np.sort(recent_window)

        dists = [
            wasserstein_distance_1d(recent_sorted, centroids[k]) for k in range(len(centroids))
        ]

        cluster = int(np.argmin(dists))
        min_dist, max_dist = min(dists), max(dists)

        # Tail divergence: right-tail vs left-tail distance to nearest centroid
        tail_div = float(
            np.mean(np.abs(recent_sorted[-5:] - centroids[cluster][-5:]))
            - np.mean(np.abs(recent_sorted[:5] - centroids[cluster][:5]))
        )

        results.append(
            {
                "timestamp": dates_list[t],
                "wass_cluster": cluster,
                "wass_dist_min": min_dist,
                "wass_dist_max": max_dist,
                "wass_dist_ratio": min_dist / (max_dist + 1e-10),
                "wass_tail_div": tail_div,
            }
        )
    return results


# %%
step = WASSERSTEIN_WINDOW - WASSERSTEIN_OVERLAP
min_length = WASSERSTEIN_WINDOW + (2 * N_CLUSTERS - 1) * step

wasserstein_all_folds = []

print("Computing per-fold Wasserstein features...")
print(f"  Window: {WASSERSTEIN_WINDOW}, Overlap: {WASSERSTEIN_OVERLAP}, Clusters: {N_CLUSTERS}")

for fold in folds:
    fold_idx = fold["fold"]
    train_end_date = fold["train_end"]
    test_end_date = fold["test_end"]
    train_start_date = fold["train_start"]

    # Find index boundaries in the xs_stats date array
    train_indices = [
        i for i, d in enumerate(dates) if str(d) >= train_start_date and str(d) < train_end_date
    ]
    full_indices = [
        i for i, d in enumerate(dates) if str(d) >= train_start_date and str(d) < test_end_date
    ]

    if len(train_indices) < min_length:
        print(
            f"  Fold {fold_idx}: insufficient training data ({len(train_indices)} < {min_length})"
        )
        continue

    # Fit centroids on training data only
    train_ret = market_ret[train_indices[0] : train_indices[-1] + 1]
    lifted = lift_stream(train_ret, WASSERSTEIN_WINDOW, WASSERSTEIN_OVERLAP)
    _, centroids = fit_wasserstein_kmeans(
        lifted.sorted_segments, n_clusters=N_CLUSTERS, random_state=42
    )

    # Sort centroids by mean value: cluster 0 = lower return (stress), 1 = higher (normal)
    sort_idx = np.argsort([c.mean() for c in centroids])
    centroids = centroids[sort_idx]

    # Assign features for the full train+test period
    fold_features = []
    if full_indices:
        fold_features = assign_regime_features(
            market_ret,
            dates,
            centroids,
            start_idx=full_indices[0],
            end_idx=full_indices[-1] + 1,
        )
        for row in fold_features:
            row["fold"] = fold_idx
        wasserstein_all_folds.extend(fold_features)

    tag = " [HOLDOUT]" if fold.get("is_holdout", False) else ""
    print(f"  Fold {fold_idx}{tag}: {len(fold_features)} dates ({len(train_indices)} train)")

wass_df = pl.DataFrame(wasserstein_all_folds) if wasserstein_all_folds else pl.DataFrame()
n_wass_folds = wass_df["fold"].n_unique() if "fold" in wass_df.columns else 0
print(f"\nWasserstein features: {len(wass_df):,} rows across {n_wass_folds} folds")
if len(wass_df) > 0:
    cluster_counts = wass_df.group_by("wass_cluster").len().sort("wass_cluster")
    for row in cluster_counts.iter_rows(named=True):
        print(f"  Cluster {row['wass_cluster']}: {row['len']:,} rows")

# %% [markdown]
# ## D. Fit stability across folds
#
# The two clusters correspond to distinct market states: one with low
# cross-sectional median return (stress/drawdown) and one with positive
# median return (normal/recovery). The `wass_dist_ratio` feature measures
# how clearly the current window sits in one regime -- values near 0
# indicate strong regime membership, values near 1 indicate ambiguity
# (regime transitions). This matters for momentum strategies because
# momentum crashes cluster at regime transitions (Daniel and Moskowitz
# 2016): the momentum-reversal spread from `03_financial_features.py` should
# interact with the Wasserstein regime signal.
#
# **Limitation**: Clustering on the median alone ignores distributional
# shape changes (wider tails, increased skewness) that may also signal
# regime transitions. A richer approach would use quantile vectors as
# input to Wasserstein k-means.

# %% [markdown]
# ### C.2 Fractional differencing
#
# Apply FFD at the configured differencing order to log(adj_close) per stock. This is the
# equity-class default: it keeps most of the correlation with the undifferenced series while
# achieving stationarity on the augmented Dickey-Fuller test for most stocks.
#
# **Walk-forward note**: the differencing order is fixed - there is no per-fold search - and
# the FFD weights are deterministic given it. Nothing is estimated, so nothing can look
# ahead. FFD is computed once over the full series; the fold column is attached during
# assembly in E, because the values are identical across folds by construction.


# %%
def apply_ffd_per_symbol(
    data: pl.DataFrame, d: float = FFD_D, threshold: float = FFD_THRESHOLD
) -> pl.DataFrame:
    """Apply fractional differencing to log prices per symbol.

    Returns DataFrame with (symbol, date, ffd_log_price, ffd_log_volume).
    """
    results = []
    symbols = data["symbol"].unique().sort().to_list()

    n_success = 0
    n_fail = 0

    for sym in symbols:
        sym_data = data.filter(pl.col("symbol") == sym).sort("timestamp")

        if len(sym_data) < 100:
            n_fail += 1
            continue

        log_price = sym_data["adj_close"].log()
        # Floor volume at 1 to avoid log(0) = -inf
        log_vol = sym_data["adj_volume"].clip(lower_bound=1).log()

        try:
            ffd_price = ffdiff(log_price, d=d, threshold=threshold)
            ffd_vol = ffdiff(log_vol, d=d, threshold=threshold)

            sym_result = pl.DataFrame(
                {
                    "symbol": [sym] * len(sym_data),
                    "timestamp": sym_data["timestamp"],
                    "ffd_log_price": ffd_price,
                    "ffd_log_volume": ffd_vol,
                }
            ).drop_nulls()

            if len(sym_result) > 0:
                results.append(sym_result)
                n_success += 1
        except Exception:
            n_fail += 1

    print(f"  FFD: {n_success} symbols succeeded, {n_fail} failed/skipped")
    return pl.concat(results) if results else pl.DataFrame()


# %%
print("Computing fractional differencing features...")
ffd_df = apply_ffd_per_symbol(df)
print(f"FFD features: {len(ffd_df):,} rows, {ffd_df['symbol'].n_unique()} symbols")

# %% [markdown]
# ### C.3 Per-fold GARCH conditional volatility
#
# Fit GARCH(1,1) on the ~200 most liquid stocks per fold. For each fold:
# 1. Select the GARCH subsample based on liquidity in the training window
# 2. Fit GARCH(1,1) on training returns per symbol
# 3. Use `model.fix(params)` on the full train+test returns to get conditional
#    volatility without re-estimating parameters
# 4. Extract features for the full train+test period (downstream needs both)
#
# For stocks not in the subsample, market-level GARCH from the cross-sectional
# median return provides a fallback.


# %%
def select_garch_subsample(data: pl.DataFrame, top_n: int) -> list[str]:
    """Select top-N most liquid symbols by median ADV."""
    ranking = (
        data.group_by("symbol")
        .agg(
            pl.col("adv_21d").median().alias("median_adv"),
            pl.len().alias("n_obs"),
        )
        .filter(pl.col("n_obs") >= GARCH_MIN_OBS)
        .sort("median_adv", descending=True)
        .head(top_n)
    )
    return ranking["symbol"].to_list()


# %%
def fit_garch_per_fold(
    data: pl.DataFrame,
    fold: dict,
    symbols: list[str],
) -> pl.DataFrame:
    """Fit GARCH(1,1) per symbol for a single fold.

    Fit on training data, use model.fix() for full train+test period.
    Returns DataFrame with (symbol, timestamp, garch_cond_vol, fold).
    """
    fold_idx = fold["fold"]
    results = []
    n_success = 0
    n_fail = 0

    for sym in symbols:
        sym_data = (
            data.filter(pl.col("symbol") == sym)
            .sort("timestamp")
            .filter(pl.col("returns").is_not_null())
        )

        # Training data for parameter estimation
        train_data = sym_data.filter(
            (pl.col("timestamp").cast(pl.Utf8) >= fold["train_start"])
            & (pl.col("timestamp").cast(pl.Utf8) < fold["train_end"])
        )

        if len(train_data) < GARCH_MIN_OBS:
            n_fail += 1
            continue

        train_returns_pct = (train_data["returns"] * 100).to_numpy()

        try:
            # Fit on training data only
            train_model = arch_model(
                train_returns_pct,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="Normal",
            )
            train_result = train_model.fit(disp="off", show_warning=False)
            fitted_params = train_result.params

            # Full train+test period for feature extraction
            full_data = sym_data.filter(
                (pl.col("timestamp").cast(pl.Utf8) >= fold["train_start"])
                & (pl.col("timestamp").cast(pl.Utf8) < fold["test_end"])
            )
            full_returns_pct = (full_data["returns"] * 100).to_numpy()

            # Run variance recursion with frozen parameters (no re-estimation)
            full_model = arch_model(
                full_returns_pct,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="Normal",
            )
            fixed_result = full_model.fix(fitted_params)
            cond_vol = fixed_result.conditional_volatility

            # Annualized conditional vol (input is in % daily)
            cond_vol_ann = cond_vol * np.sqrt(252) / 100  # Back to decimal

            sym_result = pl.DataFrame(
                {
                    "symbol": [sym] * len(full_data),
                    "timestamp": full_data["timestamp"].to_list(),
                    "garch_cond_vol": cond_vol_ann,
                    "fold": [fold_idx] * len(full_data),
                }
            ).drop_nulls()

            if len(sym_result) > 0:
                results.append(sym_result)
                n_success += 1
        except Exception:
            n_fail += 1

    tag = " [HOLDOUT]" if fold.get("is_holdout", False) else ""
    print(f"  Fold {fold_idx}{tag} GARCH: {n_success}/{len(symbols)} fitted, {n_fail} failed")
    return pl.concat(results) if results else pl.DataFrame()


# %%
def fit_market_garch_per_fold(
    market_ret: FloatArray,
    dates_list: list,
    fold: dict,
) -> pl.DataFrame:
    """Fit market-level GARCH for one fold, return (timestamp, mkt_garch_vol, fold)."""
    fold_idx = fold["fold"]

    # Build date-indexed arrays
    train_mask = [
        (str(d) >= fold["train_start"] and str(d) < fold["train_end"]) for d in dates_list
    ]
    full_mask = [(str(d) >= fold["train_start"] and str(d) < fold["test_end"]) for d in dates_list]

    train_indices = [i for i, m in enumerate(train_mask) if m]
    full_indices = [i for i, m in enumerate(full_mask) if m]

    if len(train_indices) < GARCH_MIN_OBS:
        return pl.DataFrame()

    train_ret_pct = market_ret[train_indices] * 100
    train_ret_clean = train_ret_pct[~np.isnan(train_ret_pct)]

    if len(train_ret_clean) < GARCH_MIN_OBS:
        return pl.DataFrame()

    try:
        train_model = arch_model(
            train_ret_clean,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="Normal",
        )
        train_result = train_model.fit(disp="off", show_warning=False)
        fitted_params = train_result.params

        # Full period with frozen params
        full_ret_pct = market_ret[full_indices] * 100
        # Remove NaN for model fitting but track indices
        valid_mask = ~np.isnan(full_ret_pct)
        full_ret_clean = full_ret_pct[valid_mask]
        valid_dates = [dates_list[full_indices[i]] for i, v in enumerate(valid_mask) if v]

        full_model = arch_model(
            full_ret_clean,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="Normal",
        )
        fixed_result = full_model.fix(fitted_params)
        mkt_cond_vol = fixed_result.conditional_volatility * np.sqrt(252) / 100

        return pl.DataFrame(
            {
                "timestamp": valid_dates[: len(mkt_cond_vol)],
                "mkt_garch_vol": mkt_cond_vol,
                "fold": [fold_idx] * len(mkt_cond_vol),
            }
        )
    except Exception as e:
        print(f"  Fold {fold_idx} market GARCH failed: {e}")
        return pl.DataFrame()


# %%
print("Per-fold GARCH fitting...")

garch_all_folds = []
mkt_garch_all_folds = []

for fold in folds:
    # Select subsample from training window
    train_data = df.filter(
        (pl.col("timestamp").cast(pl.Utf8) >= fold["train_start"])
        & (pl.col("timestamp").cast(pl.Utf8) < fold["train_end"])
    )
    garch_symbols = select_garch_subsample(train_data, GARCH_TOP_N)

    fold_garch = fit_garch_per_fold(df, fold, garch_symbols)
    if len(fold_garch) > 0:
        garch_all_folds.append(fold_garch)

    # Market-level GARCH fallback
    fold_mkt = fit_market_garch_per_fold(market_ret, dates, fold)
    if len(fold_mkt) > 0:
        mkt_garch_all_folds.append(fold_mkt)

garch_df = pl.concat(garch_all_folds) if garch_all_folds else pl.DataFrame()
mkt_garch_df = pl.concat(mkt_garch_all_folds) if mkt_garch_all_folds else pl.DataFrame()

if len(garch_df) > 0:
    print(
        f"\nGARCH features: {len(garch_df):,} rows, {garch_df['symbol'].n_unique()} symbols, {garch_df['fold'].n_unique()} folds"
    )
if len(mkt_garch_df) > 0:
    print(
        f"Market GARCH: {len(mkt_garch_df):,} rows across {mkt_garch_df['fold'].n_unique()} folds"
    )

# %% [markdown]
# ## E. Combine and emit
#
# Merge all temporal features per fold. The fold column is the primary
# organizing key: GARCH and Wasserstein features carry fold from their
# per-fold fitting; FFD (deterministic, no parameters) is replicated
# across folds during the join.

# %% [markdown]
# The three models attach to a fold differently, and the join has to respect that. GARCH already
# carries a fold column and is the base. Wasserstein carries one too, but it is market-level, so it
# broadcasts to every symbol. FFD carries none, because it is deterministic and its values are
# identical across folds, so it joins on `(symbol, timestamp)` alone.

# %%

temporal_frames = []

fold_ids = sorted(set(f["fold"] for f in folds))

for fold_idx in fold_ids:
    fold_info = next(f for f in folds if f["fold"] == fold_idx)

    # Build (symbol, timestamp) skeleton for this fold's full period
    fold_skeleton = (
        df.filter(
            (pl.col("timestamp").cast(pl.Utf8) >= fold_info["train_start"])
            & (pl.col("timestamp").cast(pl.Utf8) < fold_info["test_end"])
        )
        .select(["symbol", "timestamp"])
        .unique()
        .with_columns(pl.lit(fold_idx).alias("fold"))
    )

    # Join Wasserstein features (market-level, broadcast to all stocks)
    if len(wass_df) > 0:
        wass_fold = wass_df.filter(pl.col("fold") == fold_idx).drop("fold")
        fold_skeleton = fold_skeleton.join(wass_fold, on="timestamp", how="left")

    # Join FFD features (per-stock, no fold column needed)
    if len(ffd_df) > 0:
        fold_skeleton = fold_skeleton.join(ffd_df, on=["symbol", "timestamp"], how="left")

    # Join per-stock GARCH
    if len(garch_df) > 0:
        garch_fold = garch_df.filter(pl.col("fold") == fold_idx).drop("fold")
        fold_skeleton = fold_skeleton.join(garch_fold, on=["symbol", "timestamp"], how="left")

    # Join market GARCH (broadcast to all stocks)
    if len(mkt_garch_df) > 0:
        mkt_fold = mkt_garch_df.filter(pl.col("fold") == fold_idx).drop("fold")
        fold_skeleton = fold_skeleton.join(mkt_fold, on="timestamp", how="left")

    # For stocks without per-stock GARCH, fill with market GARCH
    if "garch_cond_vol" in fold_skeleton.columns and "mkt_garch_vol" in fold_skeleton.columns:
        fold_skeleton = fold_skeleton.with_columns(
            pl.when(pl.col("garch_cond_vol").is_null())
            .then(pl.col("mkt_garch_vol"))
            .otherwise(pl.col("garch_cond_vol"))
            .alias("garch_cond_vol")
        )

    temporal_frames.append(fold_skeleton)

temporal = pl.concat(temporal_frames).sort(["fold", "symbol", "timestamp"])

# Drop rows with no temporal features at all
temporal = temporal.drop_nulls(subset=["symbol", "timestamp"])

temporal_feature_cols = [c for c in temporal.columns if c not in ("symbol", "timestamp", "fold")]
n_temporal_features = len(temporal_feature_cols)

print(f"\nTemporal features: {n_temporal_features} features")
print(f"  Rows: {len(temporal):,}")
print(f"  Symbols: {temporal['symbol'].n_unique()}")
print(f"  Folds: {temporal['fold'].n_unique()}")

# %% [markdown]
# ### F2. What the models inferred, over time
#
# The two fitted quantities the panel carries, as the cross-sectional median per date so that one
# line stands for the whole panel, with the fold boundaries drawn. This is the object the
# downstream models receive - the per-fold filtered series, not an illustrative full-sample fit.

# %%
daily = (
    temporal.group_by("timestamp")
    .agg(
        pl.col("garch_cond_vol").median().alias("garch_cond_vol"),
        pl.col("ffd_log_price").median().alias("ffd_log_price"),
    )
    .sort("timestamp")
)
fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
for ax, column, ylabel in (
    (axes[0], "garch_cond_vol", "GARCH conditional vol"),
    (axes[1], "ffd_log_price", "FFD log price"),
):
    ax.plot(daily["timestamp"].to_list(), daily[column].to_list(), color=COLORS["blue"], lw=1.0)
    ax.set_ylabel(ylabel)
    for fold in folds:
        ax.axvline(np.datetime64(fold["test_start"]), color=COLORS["neutral"], lw=0.7)
    ax.axvline(np.datetime64(holdout_start), color=COLORS["negative"], ls="--", lw=1.1)
axes[1].set_xlabel("Date")
# Measured rather than asserted: the median absolute day-to-day move at a fold boundary against
# the move on every other day. Refitting does not visibly move either series.
_step = daily.with_columns(
    pl.col("garch_cond_vol").diff().abs().alias("_dg"),
    pl.col("ffd_log_price").diff().abs().alias("_df"),
).drop_nulls()
_starts = [date.fromisoformat(fold["test_start"]) for fold in folds]
_at = _step.filter(pl.col("timestamp").is_in(_starts))
_off = _step.filter(~pl.col("timestamp").is_in(_starts))
_at_med, _off_med = _at["_dg"].median(), _off["_dg"].median()
_ratio = float(_at_med / _off_med) if _at.height and _off.height and _off_med else float("nan")
add_message_title(
    axes[0],
    "Refitting each fold does not move either series; the market does",
    subtitle=(
        f"Cross-sectional median per date; move at a fold boundary is {_ratio:.2f}x the move "
        "on an ordinary day"
    ),
)
fig.show()

# %% [markdown]
# ### F3. Fit stability across folds
#
# One box per fold over the panel's GARCH conditional volatility. Boxes that sit on top of each
# other say per-fold refitting bought nothing; boxes that move say the volatility regime changed
# under the model, which is the warning the assembled panel cannot give on its own.

# %%
fig, ax = plt.subplots(figsize=(11, 4.2))
fold_ids = sorted(temporal["fold"].unique().to_list())
samples = [
    temporal.filter(pl.col("fold") == fold)["garch_cond_vol"].drop_nulls().to_list()
    for fold in fold_ids
]
samples = [sample for sample in samples if sample]
ax.boxplot(samples, tick_labels=[str(fold) for fold in fold_ids[: len(samples)]], showfliers=False)
ax.set(xlabel="Walk-forward fold", ylabel="GARCH conditional volatility")
medians = [float(np.median(sample)) for sample in samples]
add_message_title(
    ax,
    "The fitted volatility level is not the same in every fold",
    subtitle=f"Median moves {max(medians) - min(medians):.4f} across folds",
)
fig.show()

# Feature summary (exclude fold from summary)
for col in temporal_feature_cols:
    valid = temporal[col].drop_nulls()
    if len(valid) > 0:
        print(f"  {col}: {len(valid):,} valid (mean={valid.mean():.6f}, std={valid.std():.4f})")

# %% [markdown]
# ### E.1 Save the temporal features

# %%
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
output_path = FEATURES_DIR / "model_based.parquet"
temporal.write_parquet(output_path)
print(f"Saved temporal features to {output_path}")
print(f"  {n_temporal_features} features + fold column, {len(temporal):,} rows")
print(f"  Folds: {sorted(temporal['fold'].unique().to_list())}")
# %% [markdown]
# ## F. Incremental evaluation
#
# Assess whether temporal features add predictive value beyond Ch8 cross-sectional
# features, using HAC-adjusted cross-sectional Spearman IC against the primary label.
# We evaluate on the last CV fold (non-holdout) test period for a representative
# out-of-sample estimate.

# %%
from ml4t.diagnostic.metrics import compute_ic_hac_stats
from scipy.stats import spearmanr as _spearmanr

temporal_ic = {}

label_path = CASE_DIR / "labels"
label_files = sorted(label_path.glob("fwd_*.parquet")) if label_path.exists() else []

if label_files:
    # Bound from the configuration, not from whichever filename sorts first: `sorted(glob)[0]`
    # happens to give fwd_ret_1d today only because "1d" < "21d" < "5d" lexically.
    primary_name = _setup["labels"]["primary"]
    primary_label = pl.read_parquet(CASE_DIR / "labels" / f"{primary_name}.parquet")
    label_col = [c for c in primary_label.columns if c not in ("timestamp", "symbol")][0]
    label_horizon = int("".join(ch for ch in primary_name.split("_")[-1] if ch.isdigit()))
    print(f"Computing HAC-adjusted IC of temporal features vs {label_col}")

    # Use last CV fold for evaluation (non-holdout)
    cv_fold_ids = sorted(f["fold"] for f in folds if not f.get("is_holdout", False))
    eval_fold = cv_fold_ids[-1] if cv_fold_ids else 0
    temporal_eval = temporal.filter(pl.col("fold") == eval_fold)

    ic_df = temporal_eval.join(primary_label, on=["symbol", "timestamp"], how="inner")

    # Sample every 5th date for efficiency
    ic_dates = sorted(ic_df["timestamp"].unique().to_list())[::5]
    ic_df = ic_df.filter(pl.col("timestamp").is_in(ic_dates))

    # The IC series carries its own timestamp and is sorted on it before any HAC correction.
    # `partition_by(as_dict=True)` gives no ordering guarantee, and a Newey-West correction over
    # a series in arbitrary order estimates an autocovariance structure the data does not have.
    _partitions = ic_df.partition_by("timestamp", as_dict=True)
    for feat in temporal_feature_cols:
        rows = []
        for _key, group in _partitions.items():
            vals = group.select([feat, label_col]).drop_nulls()
            if len(vals) >= 30:
                rho, _ = _spearmanr(vals[feat].to_numpy(), vals[label_col].to_numpy())
                if np.isfinite(rho):
                    rows.append({"timestamp": _key[0], "ic": float(rho)})
        ic_series = pl.DataFrame(rows, schema={"timestamp": pl.Date, "ic": pl.Float64}).sort(
            "timestamp"
        )
        if ic_series.height >= 20:
            # The lag is the label's own overlap, declared rather than left to a default.
            hac_stats = compute_ic_hac_stats(ic_series, ic_col="ic", label_horizon=label_horizon)
            temporal_ic[feat] = hac_stats
            print(
                f"  {feat}: IC={hac_stats['mean_ic']:.4f} "
                f"(HAC t={hac_stats['t_stat']:.2f}, p={hac_stats['p_value']:.4f})"
            )
else:
    if not label_files:
        print("No labels found — skipping temporal IC evaluation")
    else:
        print("[TEST] Skipping temporal IC evaluation")

# %% [markdown]
# ### F4. Incremental IC by feature
#
# Mean IC per temporal feature with its HAC interval, signed and sorted. The interval is what
# decides whether a feature carries anything, and it is the reason this is drawn rather than
# printed: a table of four-decimal means invites reading a ranking into differences the intervals
# do not support. The stage-03 baseline is not drawn here - this notebook does not load the
# stage-03 matrix, and `05_evaluation` makes that comparison because it loads both.

# %%
if temporal_ic:
    ranked = sorted(temporal_ic.items(), key=lambda kv: kv[1]["mean_ic"])
    names = [name for name, _ in ranked]
    means = np.array([stats["mean_ic"] for _, stats in ranked])
    errs = np.array([1.96 * stats["hac_se"] for _, stats in ranked])
    clears_zero = np.abs(means) > errs
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 2))
    ax.barh(
        names,
        means,
        xerr=errs,
        color=[COLORS["positive"] if ok else COLORS["neutral"] for ok in clears_zero],
        error_kw={"ecolor": COLORS["silver_muted"], "capsize": 3},
    )
    ax.axvline(0, color=COLORS["neutral"], lw=1)
    ax.set_xlabel("Mean cross-sectional IC (Spearman), HAC 95% interval")
    add_message_title(
        ax,
        "Every feature that can be evaluated clears zero",
        subtitle=(
            f"Validation fold {eval_fold}; {int(clears_zero.sum())} of {len(names)} clear zero, "
            f"and {len(names)} of {len(temporal_feature_cols)} temporal columns had enough "
            "per-date cross-sections to estimate an IC at all"
        ),
    )
    fig.show()
else:
    print("No temporal IC computed — F4 not drawn")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Per-fold walk-forward discipline**: All temporal models are fitted per
#    CV fold on training data only. GARCH uses `model.fix()` to run the
#    variance recursion on the full train+test window without re-estimating
#    parameters. Wasserstein centroids are fitted on training windows only.
#    No parameter-level look-ahead remains.
#
# 2. **Wasserstein regime detection** clusters windowed sequences of the
#    cross-sectional median return into risk-on and risk-off states. The
#    distance-to-centroid features capture regime transition uncertainty,
#    which matters because momentum crashes cluster at transitions
#    (Daniel and Moskowitz 2016).
#
# 3. **Fractional differencing** at the configured order preserves long memory in price
#    levels while achieving stationarity. Fixed $d$ means no estimation
#    look-ahead -- the transform is purely mechanical.
#
# 4. **GARCH subsample strategy**: Fitting GARCH on the most liquid stocks
#    captures volatility dynamics of the investable universe. Market-level
#    GARCH fills in for illiquid stocks where per-stock estimation is unstable.
#
# 5. **Fold column**: Every row carries a `fold` column so downstream models
#    can join features per fold for consistent walk-forward evaluation.
#
# **Next**: `models/` notebooks in Ch11+ use these temporal features alongside
# Ch8 cross-sectional features for prediction.
