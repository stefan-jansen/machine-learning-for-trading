# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
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

# %% [markdown] tags=[]
# # Mechanism Plausibility Checks for Feature Triage
#
# **Docker image**: `ml4t`
#
# **Chapter 7: Defining the Learning Task**
# **Section Reference**: 7.5 - From Correlation to Causality
#
# ## Purpose
#
# This notebook implements **lightweight falsification tests** for feature evaluation.
# These tests complement the correlation-based IC analysis from Section 7.3 by checking
# whether a feature-outcome association is *consistent* with a proposed mechanism.
#
# **Important**: These checks do not identify causal effects. They are robustness
# diagnostics guided by mechanism reasoning, testing one feature-outcome pair at
# a time. They cannot detect multivariate confounding. Formal causal
# identification is deferred to Chapter 15.
#
# We begin with a **feature × horizon scan** that expands on the ETF results from
# [`07_multiple_testing`](07_multiple_testing.ipynb), where no feature survived BH-FDR at a 5-day horizon.
# Widening the search to 10 features × 3 horizons and correcting across all 30 tests,
# the long-lookback momentum terms (252d and 12-1) are the only ones that survive, and
# they survive at the 5-day horizon. Their largest ICs are at 21 days but do not clear
# the corrected threshold there - a weaker result than the textbook finding
# (Jegadeesh and Titman 1993, Asness et al. 2013) would lead you to expect from 92 ETFs
# over 14 years, and the honest one. We then apply the three diagnostic checks to two
# features selected from the scan:
#
# - **12-1 Momentum** (12-month return skipping the most recent month): survives
#   triage with actionable caveats - it carries genuine cross-sectional information
#   (strongest at lag 0), does not predict Treasury returns, but concentrates in
#   low-volatility regimes.
# - **Short-term reversal** (negated 1-day return): fails triage - no significant
#   IC at the 21-day horizon, and its one significant cell does not survive the
#   shifted-label check in Section 4.1.
#
# ## Learning Objectives
#
# 1. Run a multi-feature, multi-horizon IC scan to identify where signal exists
# 2. State a proposed mechanism using DAG vocabulary (confounder, mediator, collider)
# 3. Implement timing placebos (shift features, examine IC half-life)
# 4. Run shared-driver checks (unrelated outcomes should show no effect)
# 5. Assess regime heterogeneity (IC stability across VIX regime partitions)
# 6. Interpret results as triage decisions (proceed / revise / stop)
#
# ## Book Reference
#
# Section 7.5 applies the three core checks to two features on the `etfs` universe
# with a 21-day forward return label. This notebook produces the feature scan
# heatmap and dual-feature diagnostics referenced in the worked example.
#
# **Prerequisites**: Notebooks [`05_signal_evaluation`](05_signal_evaluation.ipynb) and [`06_ic_inference`](06_ic_inference.ipynb)
# introduce IC analysis; [`07_multiple_testing`](07_multiple_testing.ipynb) motivates the horizon expansion.

# %% tags=[]
"""Mechanism Plausibility Checks for Feature Triage."""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from ml4t.diagnostic.metrics import compute_ic_hac_stats
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.multitest import multipletests

from data import load_etfs, load_macro
from utils.reproducibility import set_global_seeds
from utils.style import COLORS  # importing utils.style activates the ml4t Plotly template

warnings.filterwarnings("ignore")


# %% tags=["parameters"]
SEED = 42
START_DATE = "2010-01-01"
# A permutation test cannot resolve a p-value finer than 1/(B+1). At B=200 that
# floor is 0.005, which is coarser than the significance this notebook reports.
N_PERMUTATIONS = 1000
OUTPUT_DIR = Path("07_defining_the_learning_task/output")

# %% tags=[]
set_global_seeds(SEED)


# %% [markdown] tags=[]
# ## 1. Motivation: Why Expand the Search?
#
# The multiple-testing scan in [`07_multiple_testing`](07_multiple_testing.ipynb) found that 0 out of 13
# short-lookback features survived BH-FDR correction at a 5-day horizon on ETFs.
# Two explanations are possible: the features are genuinely uninformative, or the
# 5-day horizon is too short for cross-sectional predictability in a 100-asset
# universe. Cross-asset momentum is well-established at monthly+ horizons
# (Asness, Moskowitz, and Pedersen 2013) but largely absent at weekly frequencies.
#
# We resolve this by scanning 10 features across three horizons (5d, 21d, 63d).
# This empirical scan determines which feature–horizon combinations carry signal,
# and two are selected for the mechanism plausibility checks that follow.

# %% [markdown] tags=[]
# ## 2. Load Data
#
# We use the full ETF universe (~92 non-bond assets) and load VIX from FRED macro
# data for regime conditioning. Treasury ETFs are excluded from the analysis
# universe but IEF is retained for the shared-driver control.

# %% tags=[]
etfs = load_etfs()

END_DATE = date(2024, 1, 1)

etfs = etfs.filter(
    (pl.col("timestamp") >= date.fromisoformat(START_DATE)) & (pl.col("timestamp") < END_DATE)
).sort(["symbol", "timestamp"])

print(f"ETF universe: {etfs['symbol'].n_unique()} symbols, {len(etfs):,} rows")
print(f"Date range: {etfs['timestamp'].min()} to {etfs['timestamp'].max()}")

# %% tags=[]
# Load VIX for regime conditioning
vix_raw = load_macro(series=["vixcls"])
vix = vix_raw.filter(
    (pl.col("timestamp") >= date.fromisoformat(START_DATE)) & (pl.col("timestamp") < END_DATE)
).drop_nulls(subset=["vixcls"])

print(f"VIX: {len(vix):,} observations")
print(f"VIX range: {vix['vixcls'].min():.1f} to {vix['vixcls'].max():.1f}")

# %% tags=[]
# Compute Treasury 21d forward return (shared-driver control)
TREASURY_SYMBOL = "IEF"  # 7-10 Year Treasury Bond ETF
LABEL_HORIZON = 21  # days

treasury_fwd = (
    etfs.filter(pl.col("symbol") == TREASURY_SYMBOL)
    .sort("timestamp")
    .with_columns(
        (pl.col("close").shift(-LABEL_HORIZON) / pl.col("close")).log().alias("treasury_fwd_return")
    )
    .select(["timestamp", "treasury_fwd_return"])
    .drop_nulls()
)

# The shared-driver control is joined below with an inner semantics
# (`drop_nulls` on the joined column), so an absent Treasury series does not
# weaken the analysis - it empties it, and every statistic downstream is then
# computed on nothing. Say so here rather than let a later cell fail on whichever
# column happens to be missing first.
if treasury_fwd.is_empty():
    raise ValueError(
        f"No {TREASURY_SYMBOL} rows between {START_DATE} and {END_DATE}, so the "
        f"shared-driver control cannot be built and the analysis frame would be empty. "
        f"Universe contains: {sorted(etfs['symbol'].unique().to_list())}"
    )

# %% [markdown] tags=[]
# ### Compute features and labels
#
# We compute 10 features spanning four families (momentum, reversal, trend,
# volatility) and forward returns at three horizons (5d, 21d, 63d). All features
# use data available at time $t$ (no lookahead). The 12-1 momentum follows the
# academic convention of skipping the most recent month to avoid short-term
# reversal contamination.

# %% tags=[]
# Exclude Treasury/Bond ETFs from the analysis universe
TREASURY_SYMBOLS = ["IEF", "TLT", "SHY", "AGG", "BND", "TIP", "GOVT", "BNDX", "VGSH"]

analysis = (
    etfs.filter(~pl.col("symbol").is_in(TREASURY_SYMBOLS))
    .with_columns(
        # Daily return for volatility computation
        pl.col("close").pct_change().over("symbol").alias("daily_ret"),
    )
    .with_columns(
        # ── Momentum features ──
        (pl.col("close") / pl.col("close").shift(21).over("symbol")).log().alias("mom_21d"),
        (pl.col("close") / pl.col("close").shift(63).over("symbol")).log().alias("mom_63d"),
        (pl.col("close") / pl.col("close").shift(126).over("symbol")).log().alias("mom_126d"),
        (pl.col("close") / pl.col("close").shift(252).over("symbol")).log().alias("mom_252d"),
        # 12-1: skip most recent month (Jegadeesh-Titman convention)
        (pl.col("close").shift(21).over("symbol") / pl.col("close").shift(252).over("symbol"))
        .log()
        .alias("mom_12_1"),
        # Trailing volatility (126d, annualized)
        (pl.col("daily_ret").rolling_std(126).over("symbol") * np.sqrt(252)).alias("vol_126d"),
        # ── Reversal features ──
        (pl.col("close").shift(1).over("symbol") / pl.col("close")).log().alias("rev_1d"),
        (pl.col("close").shift(5).over("symbol") / pl.col("close")).log().alias("rev_5d"),
        # ── Trend ──
        (
            (pl.col("close") - pl.col("close").rolling_mean(200).over("symbol"))
            / pl.col("close").rolling_mean(200).over("symbol")
        ).alias("dist_200ma"),
        # ── Volatility ──
        (
            pl.col("daily_ret").rolling_std(20).over("symbol")
            / pl.col("daily_ret").rolling_std(60).over("symbol")
        ).alias("rvol_ratio"),
        # ── Forward returns (labels) ──
        (pl.col("close").shift(-5).over("symbol") / pl.col("close")).log().alias("fwd_5d"),
        (pl.col("close").shift(-21).over("symbol") / pl.col("close")).log().alias("fwd_21d"),
        (pl.col("close").shift(-63).over("symbol") / pl.col("close")).log().alias("fwd_63d"),
    )
)

# %% tags=[]
# Risk-adjusted features and label alias (requires vol_126d from previous step)
analysis = analysis.with_columns(
    (pl.col("mom_126d") / pl.col("vol_126d")).alias("adj_mom_126d"),
    pl.col("fwd_21d").alias("forward_return"),
)

# %% tags=[]
# Join VIX for regime conditioning
analysis = analysis.join(vix, on="timestamp", how="left").drop_nulls(subset=["vixcls"])

# Join Treasury forward return for shared-driver check
analysis = analysis.join(treasury_fwd, on="timestamp", how="left").drop_nulls(
    subset=["treasury_fwd_return"]
)

n_symbols = analysis["symbol"].n_unique()
n_dates = analysis["timestamp"].n_unique()
print(f"Analysis panel: {len(analysis):,} rows, {n_symbols} symbols, {n_dates:,} dates")
if n_dates > 0:
    print(f"Cross-section size: ~{len(analysis) // n_dates} assets per date")
else:
    print("Cross-section size: N/A (no dates after join)")

# %% [markdown] tags=[]
# ## 3. Cross-Sectional IC Function
#
# We compute Spearman rank correlation between each feature and forward returns at
# each date, producing a time series of ICs. This is the same IC framework from
# Section 7.3.


# %% tags=[]
def _contiguous_groups(sorted_keys: np.ndarray) -> list[np.ndarray]:
    """Row indices of each run of equal keys, for an array already sorted by key."""
    if len(sorted_keys) == 0:
        return []
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    return np.split(np.arange(len(sorted_keys)), starts[1:])


def block_permutation_null(
    df: pl.DataFrame,
    feature_col: str,
    baseline_ic: float,
    seed: int,
    n_permutations: int,
) -> tuple[np.ndarray, float, float]:
    """Permutation null for a cross-sectional IC, blocked at the label horizon.

    Shuffling assets independently on each date would imply the per-date ICs are
    independent, so the null mean's spread would shrink like sigma/sqrt(n_dates).
    The labels are LABEL_HORIZON-day forward returns sampled daily - consecutive
    dates share all but one day of return - so both the returns and the per-date
    ICs are strongly autocorrelated and the true spread is far larger. One asset
    relabeling is therefore drawn per block of LABEL_HORIZON sessions and held
    fixed across the block, exactly as ``05_signal_evaluation`` does.

    Returns ``(null_ics, perm_p, perm_resolution)``. The p-value is
    ``(r + 1) / (B + 1)``: a permutation p-value can never be exactly zero, since
    the observed assignment is itself one of the arrangements under the null, and
    with B permutations the finest resolvable value is ``1 / (B + 1)``.
    """
    perm_df = df.drop_nulls([feature_col, "forward_return"]).sort(["timestamp", "symbol"])
    dates_arr = perm_df["timestamp"].to_numpy()
    symbol_codes = perm_df["symbol"].cast(pl.Categorical).to_physical().to_numpy()
    n_symbols = int(symbol_codes.max()) + 1
    feat_arr_p = perm_df[feature_col].to_numpy()
    ret_arr_p = perm_df["forward_return"].to_numpy()

    # Ranks are invariant to relabeling, so pre-compute both sides once per date.
    # Standardize them here too: a permutation changes neither a vector's mean nor its
    # standard deviation, so the Spearman IC of a permuted pair is just the dot product
    # of the two standardized rank vectors divided by n. That is identical to
    # np.corrcoef to floating-point noise and about 17x faster, which matters because
    # this loop runs n_permutations x n_dates times.
    def _z(v: np.ndarray) -> np.ndarray | None:
        sd = v.std()
        return (v - v.mean()) / sd if sd > 0 else None

    date_groups = [idx for idx in _contiguous_groups(dates_arr) if len(idx) >= 20]
    feat_ranks, ret_ranks, syms_by_date = [], [], []
    for idx in date_groups:
        fz = _z(stats.rankdata(feat_arr_p[idx]))
        rz = _z(stats.rankdata(ret_arr_p[idx]))
        if fz is None or rz is None:
            continue
        feat_ranks.append(fz)
        ret_ranks.append(rz)
        syms_by_date.append(symbol_codes[idx])

    rng = np.random.default_rng(seed)
    null_ics = []
    for _ in range(n_permutations):
        ic_per_date = []
        block_keys = None
        for i, f_ranks in enumerate(feat_ranks):
            # New relabeling only when a block boundary is crossed
            if i % LABEL_HORIZON == 0:
                block_keys = rng.permutation(n_symbols)
            # Same key vector across the block => same asset->asset map across the block
            order = np.argsort(block_keys[syms_by_date[i]], kind="stable")
            ic_per_date.append(float(f_ranks @ ret_ranks[i][order]) / len(f_ranks))
        if ic_per_date:
            null_ics.append(np.mean(ic_per_date))

    null_ics = np.array(null_ics)
    n_at_least = int(np.sum(np.abs(null_ics) >= abs(baseline_ic)))
    return null_ics, (n_at_least + 1) / (len(null_ics) + 1), 1.0 / (len(null_ics) + 1)


def compute_cross_sectional_ic(
    df: pl.DataFrame, feature_col: str, outcome_col: str, min_obs: int = 20
) -> tuple[float, float, list[float]]:
    """Compute cross-sectional IC (Spearman) at each timestamp.

    Returns: (mean_ic, t_stat, ic_series)
    """
    groups = df.select("timestamp", feature_col, outcome_col).partition_by(
        "timestamp", as_dict=True
    )

    ic_values = []
    for _key, group in groups.items():
        if len(group) < min_obs:
            continue
        feature = group[feature_col].to_numpy()
        outcome = group[outcome_col].to_numpy()
        if np.std(feature) < 1e-10 or np.std(outcome) < 1e-10:
            continue
        ic, _ = stats.spearmanr(feature, outcome)
        if not np.isnan(ic):
            ic_values.append(ic)

    if not ic_values:
        return np.nan, np.nan, []

    ic_array = np.array(ic_values)
    mean_ic = np.mean(ic_array)
    std_ic = np.std(ic_array, ddof=1)
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic_array))) if std_ic > 0 else np.nan

    return mean_ic, t_stat, ic_values


# %% [markdown] tags=[]
# ## 4. Feature × Horizon Scan
#
# We scan 10 features across three forward-return horizons (5d, 21d, 63d) using
# HAC-adjusted inference. The heatmap reveals a clear pattern: long-lookback
# momentum features (126d+) carry significant cross-sectional information,
# while short-term features remain noise regardless of horizon.

# %% tags=[]
SCAN_FEATURES = [
    ("mom_21d", "21d Momentum"),
    ("mom_63d", "63d Momentum"),
    ("mom_126d", "126d Momentum"),
    ("mom_252d", "252d Momentum"),
    ("mom_12_1", "12-1 Momentum"),
    ("adj_mom_126d", "Risk-Adj. Mom."),
    ("rev_1d", "1d Reversal"),
    ("rev_5d", "5d Reversal"),
    ("dist_200ma", "Dist. 200d MA"),
    ("rvol_ratio", "Vol Ratio 20/60"),
]

SCAN_HORIZONS = [("fwd_5d", "5d"), ("fwd_21d", "21d"), ("fwd_63d", "63d")]

# %% tags=[]
# Run scan: IC + HAC t-stat for each feature × horizon pair
scan_rows = []
for feat_col, feat_label in SCAN_FEATURES:
    for hz_col, hz_label in SCAN_HORIZONS:
        sub = analysis.drop_nulls(subset=[feat_col, hz_col])
        _, _, ic_series = compute_cross_sectional_ic(sub, feat_col, hz_col)
        if len(ic_series) < 50:
            # Carry the key even when there is nothing to put in it: if every pair
            # took this branch the column would not exist at all, and the
            # multiple-testing correction below would fail on a missing column
            # rather than on an empty scan.
            scan_rows.append(
                {
                    "feature_col": feat_col,
                    "feature": feat_label,
                    "horizon": hz_label,
                    "ic": np.nan,
                    "t_hac": np.nan,
                    "p_hac": None,
                }
            )
            continue
        hac = compute_ic_hac_stats(ic_series, label_horizon=int(hz_label[:-1]))
        scan_rows.append(
            {
                "feature_col": feat_col,
                "feature": feat_label,
                "horizon": hz_label,
                "ic": round(hac["mean_ic"], 4),
                "t_hac": round(hac["t_stat"], 2),
                "p_hac": hac["p_value"],
            }
        )

scan_df = pl.DataFrame(scan_rows, schema_overrides={"p_hac": pl.Float64})

# %% [markdown] tags=[]
# ### Correcting the scan for multiple testing
#
# The scan above is not one test, it is
# $10 \times 3 = 30$. At $\alpha = 0.05$ we expect 1.5 false positives from noise
# alone, so reading `|t| > 2` off 30 cells is precisely the error
# [`07_multiple_testing`](07_multiple_testing.ipynb) exists to prevent. We control
# the false discovery rate across the whole grid with Benjamini-Hochberg, and mark
# significance with the corrected decision rather than the raw threshold.

# %% tags=[]
scan_valid = scan_df.drop_nulls("p_hac")
bh_reject, bh_qvalues = multipletests(scan_valid["p_hac"].to_numpy(), alpha=0.05, method="fdr_bh")[
    :2
]
scan_df = scan_df.join(
    scan_valid.select("feature", "horizon").with_columns(
        q_bh=pl.Series(bh_qvalues), sig_bh=pl.Series(bh_reject)
    ),
    on=["feature", "horizon"],
    how="left",
)

n_raw = int((scan_df["p_hac"] < 0.05).sum())
n_bh = int(scan_df["sig_bh"].fill_null(False).sum())
print(f"Tests in the scan: {len(scan_valid)}")
print(f"Significant at raw p < 0.05:        {n_raw}")
print(f"Significant after BH FDR control:   {n_bh}")
print(f"Expected false positives if all 30 were null: {0.05 * len(scan_valid):.1f}")

# %% tags=[]
# Heatmap: HAC t-statistics by feature × horizon
feat_labels = [f[1] for f in SCAN_FEATURES]
hz_labels = [h[1] for h in SCAN_HORIZONS]

# Build matrices for the heatmap
z_matrix = []
text_matrix = []
for feat_label in feat_labels:
    row_z = []
    row_text = []
    for hz_label in hz_labels:
        match = scan_df.filter((pl.col("feature") == feat_label) & (pl.col("horizon") == hz_label))
        t_val = match["t_hac"][0] if len(match) > 0 else np.nan
        ic_val = match["ic"][0] if len(match) > 0 else np.nan
        sig_bh = (
            bool(match["sig_bh"][0]) if len(match) > 0 and match["sig_bh"][0] is not None else False
        )
        row_z.append(t_val if not np.isnan(t_val) else 0)
        # The star is the BH-corrected decision across all 30 cells, not |t| > 2
        sig = "*" if sig_bh else ""
        row_text.append(f"IC={ic_val:.3f}<br>t={t_val:.1f}{sig}" if not np.isnan(t_val) else " - ")
    z_matrix.append(row_z)
    text_matrix.append(row_text)

# %% tags=[]
fig = go.Figure(
    data=go.Heatmap(
        z=z_matrix,
        x=hz_labels,
        y=feat_labels,
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorscale="RdBu",
        zmid=0,
        zmin=-5,
        zmax=5,
        colorbar=dict(title="HAC t-stat"),
    )
)
fig.update_layout(
    title="Three of 30 cells survive FDR correction, all at the 5-day horizon",
    xaxis_title="Forward Return Horizon",
    height=450,
    width=760,
    # The feature names are long; without the left margin they render clipped
    margin=dict(l=150),
    yaxis=dict(autorange="reversed"),
)
fig.show()

# %% [markdown] tags=[]
# **Findings from the scan.** Read the starred cells, not the $|t| > 2$ ones: six cells
# clear raw $p < 0.05$ and **three survive BH across the grid**, which is close to the
# 1.5 false positives 30 null tests would produce on their own.
#
# 1. **The three survivors all sit at the 5-day horizon**: 12-1 momentum ($t = 3.7$),
#    252d momentum ($t = 3.4$), and 1-day reversal ($t = 3.0$). Section 4.1 shows the
#    third does not survive a five-day label measured one session later, which
#    leaves two to carry forward.
# 2. **The 21-day cells do not survive the correction.** 12-1 momentum reaches
#    $t = 2.6$ and 252d momentum $t = 2.3$ - the largest ICs anywhere in the grid
#    (0.053 and 0.045) - but against 30 tests that is not enough. This is the honest
#    reading, and it is *weaker* than the chapter's later worked example needs; the
#    deep-dive below proceeds on 12-1 momentum at 21 days anyway, with the caveat that
#    its evidence is a large effect at borderline significance rather than a clean one.
# 3. **63-day horizon**: signal fades. Distance from the 200-day MA is the largest
#    remaining $t$ at 2.1, and the momentum terms slip to $t = 1.7$ (12-1 and 252d)
#    and $t = 1.4$ (126d) - consistent with momentum being a monthly rather than a
#    quarterly effect.
# 4. **Short-lookback and non-momentum features are noise throughout.** 21d and 63d
#    momentum, risk-adjusted momentum, 5d reversal and the vol ratio never exceed
#    $|t| = 1.8$ at any horizon.
#
# Note what the correction changed. Reading $|t| > 2$ off the grid would have credited
# six discoveries, including two at the horizon the rest of the chapter uses. The
# correction is not a formality here - it removes half of them.
#
# The surviving pattern is consistent with well-established findings on cross-asset
# momentum:
# the effect requires long lookbacks (6–12 months) and manifests at monthly+
# horizons.

# %% [markdown] tags=[]
# ### 4.1 A Significant Cell That Does Not Survive a Shift of Its Label
#
# One cell in the scan deserves a second look before we trust it: 1-day reversal at
# the 5-day horizon. Write the two quantities out in logs, with $p_t = \log P_t$:
#
# $$\text{rev\_1d}_t = p_{t-1} - p_t \qquad \text{fwd\_5d}_t = p_{t+5} - p_t$$
#
# Both contain $-p_t$. Whatever noise sits in the close on day $t$ - a wide bid-ask
# spread, a stale print, a bad tick - enters the feature and the label with the
# *same* sign, so it induces positive covariance between them whether or not any
# reversal exists. The feature and the label share an endpoint.
#
# This is not a hypothetical. The check is to move the label one day forward, so it
# spans $p_{t+6} - p_{t+1}$: the same five-day holding period and no shared endpoint.
#
# **What this check can and cannot settle.** It is a *shift* of the return window,
# not a removal of the shared price alone. Going from $p_{t+5} - p_t$ to
# $p_{t+6} - p_{t+1}$ drops the $t \to t+1$ session **and adds a $t+5 \to t+6$
# session**, on top of dropping the shared $p_t$. At least four things could
# produce the decline we are about to see:
#
# 1. shared-endpoint noise inflated the original statistic;
# 2. a real effect concentrated in the $t \to t+1$ session, now excluded;
# 3. the added $t+5 \to t+6$ session contributes returns that offset the rest;
# 4. sampling variation, on a $t$ that was not large to begin with.
#
# **This test distinguishes none of them.** One comparison with several
# simultaneous changes cannot attribute its own result, and the only honest reading
# is that the statistic moved when the window did.
#
# Nor does the drop say anything about tradeability. A signal computed from the
# day-$t$ close executes at the $t+1$ *open*, which is this repository's label
# convention, so neither of these close-to-close labels is what a strategy would
# earn. Separating the four needs measurements this notebook does not make: an
# independent price for day $t$ - a quote midpoint rather than a last trade - to
# isolate endpoint noise, one-session-at-a-time attribution across the window, and
# an open-to-open label for the tradeable quantity. All belong with the execution
# assumptions in Chapter 16.
#
# What the check does establish is enough for the decision at hand: **the cell does
# not survive a five-day label measured one session later.** A result that depends
# on which five sessions are used is not one to build on, whichever of the four
# explanations holds. That is a reason to withhold belief until it is re-measured -
# not a demonstration that no reversal effect exists.

# %% tags=[]
shared_endpoint = analysis.with_columns(
    # Same 5-day span, measured from t+1: shares no price with rev_1d
    (pl.col("close").shift(-6).over("symbol") / pl.col("close").shift(-1).over("symbol"))
    .log()
    .alias("fwd_5d_from_t1"),
)

endpoint_rows = []
for label_col, label_desc in [
    ("fwd_5d", "p(t+5) - p(t)   [shares p(t) with the feature]"),
    ("fwd_5d_from_t1", "p(t+6) - p(t+1) [no shared endpoint]"),
]:
    sub = shared_endpoint.drop_nulls(subset=["rev_1d", label_col])
    _, _, ic_series_ep = compute_cross_sectional_ic(sub, "rev_1d", label_col)
    hac_ep = compute_ic_hac_stats(ic_series_ep, label_horizon=5)
    endpoint_rows.append(
        {
            "label": label_desc,
            "mean_ic": f"{hac_ep['mean_ic']:+.4f}",
            "t_hac": f"{hac_ep['t_stat']:+.2f}",
            "p_hac": f"{hac_ep['p_value']:.4f}",
            "n_dates": len(ic_series_ep),
        }
    )

display(pl.DataFrame(endpoint_rows))

# %% [markdown] tags=[]
# Removing the shared endpoint takes the 1-day reversal hit from HAC $t \approx 2.9$
# to $t \approx 1.6$ - from "significant" to not - on the same panel and the same
# five-day holding period, moved forward by one day.
#
# So the scan's short-horizon reversal cell does not survive its own robustness
# check, and the deep-dive below therefore treats 1-day reversal as a near-null
# baseline rather than as a signal to explain. Note what that is *not*: it is not a
# finding that one-day reversal is absent from this panel, and it is not a
# tradeability result. Both would need the measurements named above.
# **The general rule: a feature ending at $t$ and a label beginning at $t$ share a
# price, and shared noise in that price induces covariance between them whether or
# not any effect exists. That is a property of the construction, provable without
# looking at data - which is exactly why it cannot be read backwards. Seeing a cell
# fail this check does not establish that the mechanism fired; it establishes that
# the cell has to be re-measured before it is believed.**

# %% [markdown] tags=[]
# ### Feature selection for diagnostic deep-dive
#
# We select two features for the mechanism plausibility checks, chosen to
# illustrate contrasting triage outcomes:
#
# - **12-1 Momentum** (IC = 0.053, HAC $t$ = 2.6 at 21d): The strongest signal
#   in the scan. Follows the Jegadeesh–Titman convention of skipping the most
#   recent month to separate momentum from short-term reversal. The question:
#   does it survive mechanism checks, or is the signal driven by a confound?
# - **1-day Reversal** (IC = 0.001, HAC $t$ = 0.4 at 21d): a near-null baseline that
#   also flips sign across VIX regimes. Its one significant cell, $t = 3.0$ at the
#   5-day horizon, is the one Section 4.1 shows collapsing under a shifted label. We
#   expect the mechanism checks to return STOP - and the useful part is *which* check
#   catches it, since neither a timing placebo nor a shared-driver control was
#   designed to detect how a feature and its label are constructed.

# %% tags=[]
# Selected features for deep diagnostics
FEATURES = {
    "mom_12_1": "12-1 Momentum",
    "rev_1d": "1-day Reversal",
}

# %% [markdown] tags=[]
# ## 5. Mechanism Hypotheses
#
# Before running diagnostics, we state the assumed causal mechanism for each
# feature. This structures the interpretation of the checks.
#
# ### Feature A: 12-1 Momentum
#
# The 12-1 momentum signal (return from $t-252$ to $t-21$) is attributed to
# behavioral underreaction: investors are slow to incorporate information, so
# past winners continue to outperform. The alternative: momentum proxies a
# risk-on/risk-off regime rather than encoding timely cross-sectional information.
#
# ```
#        Volatility Regime (VIX)
#             /        \
#            v          v
#     12-1 Momentum -->?  Forward Return
# ```
#
# ### Feature B: 1-day Reversal
#
# Negated 1-day return. The hypothesized mechanism is microstructure-driven mean
# reversion: temporary price dislocations reverse within days. At a 21-day
# horizon, this mechanism should have no power - the effect decays too quickly.
#
# ```
#       Microstructure Friction
#             |
#             v
#       1-day Return  -->?  21-day Forward Return
# ```

# %% [markdown] tags=[]
# ### Why run these checks?
#
# These checks cost minutes; building a full model pipeline costs chapters.
# Their primary value is **catching dead features early** (reversal correctly
# gets STOP) and **generating actionable diagnostics** (momentum's regime
# concentration informs Ch12+ modeling choices).
#
# **REVISE is the expected outcome, not failure.** In efficient markets,
# cross-sectional ICs are small ($\sim$0.03–0.05) and HAC-adjusted significance is
# conservative. Few single features will cleanly pass all three bivariate
# checks. The diagnostic information - *where* and *when* a feature works -
# matters more than the triage label.

# %% [markdown] tags=[]
# ## 6. Baseline IC (HAC-Adjusted)
#
# We report the HAC-adjusted IC for both features at the 21-day horizon. This
# establishes the baseline that the diagnostic checks will probe.

# %% tags=[]
baseline = {}
for feat_col, feat_label in FEATURES.items():
    sub = analysis.drop_nulls(subset=[feat_col, "forward_return"])
    ic, t, series = compute_cross_sectional_ic(sub, feat_col, "forward_return")
    hac = (
        compute_ic_hac_stats(series, label_horizon=LABEL_HORIZON)
        if series
        else {"t_stat": np.nan, "p_value": np.nan}
    )
    baseline[feat_col] = {"ic": ic, "t": t, "series": series, "hac": hac}


# The BH decision for these same two cells is carried alongside the raw p-value.
# Reporting a raw p here without it is what let this section call a cell
# "significant" that Section 4's own grid-wide correction rejects.
def _bh_for(feat_col: str) -> tuple[str, str]:
    # Join on the feature column, not the display label. SCAN_FEATURES and FEATURES
    # spell the same feature differently ("1d Reversal" vs "1-day Reversal"), so a
    # label join silently returns "n/a" for one of the two features here - which is
    # exactly the kind of quiet miss this table exists to prevent.
    row = scan_df.filter(
        (pl.col("feature_col") == feat_col) & (pl.col("horizon") == f"{LABEL_HORIZON}d")
    )
    if len(row) == 0:
        raise KeyError(f"{feat_col} at {LABEL_HORIZON}d is not in the scan grid")
    if row["q_bh"][0] is None:
        return "n/a", "n/a"
    return f"{row['q_bh'][0]:.4f}", "yes" if row["sig_bh"][0] else "no"


_bh = {f: _bh_for(f) for f in FEATURES}

print(
    pl.DataFrame(
        {
            "feature": list(FEATURES.values()),
            "mean_ic": [f"{baseline[f]['ic']:.4f}" for f in FEATURES],
            "t_naive": [f"{baseline[f]['t']:.2f}" for f in FEATURES],
            "t_hac": [f"{baseline[f]['hac']['t_stat']:.2f}" for f in FEATURES],
            "p_hac_raw": [f"{baseline[f]['hac']['p_value']:.4f}" for f in FEATURES],
            "q_bh_grid": [_bh[f][0] for f in FEATURES],
            "survives_bh": [_bh[f][1] for f in FEATURES],
        }
    )
)

# %% [markdown] tags=[]
# 12-1 momentum has the largest IC in the grid and the smallest raw p-value of the
# two features here, and it **does not survive the grid-wide BH correction** - the
# `survives_bh` column says so, and Section 4 said so about the same cell. The raw
# $p$ is a post-selection number: this feature and this horizon were chosen by
# looking at the scan, so the 30 comparisons that produced the choice have to be
# paid for, and BH is the bill. Reversal IC is indistinguishable from zero on any
# reading.
#
# The naive t-stat for momentum is higher than the HAC one because it ignores
# autocorrelation in the IC series. That correction and the multiplicity correction
# are separate, and both apply: HAC widens the interval for one test, BH sets the
# threshold that test has to clear given the other 29.
#
# The deep-dive continues on 12-1 momentum regardless, because the chapter needs a
# worked example and this is the strongest candidate the scan produced. What it is
# an example of is a **large effect at borderline significance** - which is the
# ordinary situation in cross-sectional equity work, and a more useful thing to
# demonstrate the diagnostics on than a clean winner would be.

# %% [markdown] tags=[]
# ## 7. Reusable Diagnostic Functions
#
# We extract each falsification test into a reusable function so we can apply
# the same checks to both features systematically.


# %% tags=[]
def run_timing_placebo(
    df: pl.DataFrame,
    feature_col: str,
    outcome_col: str = "forward_return",
    lags: list[int] | None = None,
) -> tuple[pl.DataFrame, str, str]:
    """Shift the feature backward by increasing lags and recompute IC.

    Uses HAC-adjusted inference at each lag. Reports IC half-life as a
    diagnostic rather than applying a hard decay threshold.

    Returns: (lag_results_df, result_label, interpretation_msg)
    """
    if lags is None:
        lags = [0, 5, 21, 42, 63, 126, 252]

    lag_rows = []
    for lag in lags:
        col_name = f"{feature_col}_lag{lag}"
        lagged = df.with_columns(
            pl.col(feature_col).shift(lag).over("symbol").alias(col_name)
        ).drop_nulls(subset=[col_name])
        ic, _, ic_series = compute_cross_sectional_ic(lagged, col_name, outcome_col)
        hac = (
            compute_ic_hac_stats(ic_series, label_horizon=LABEL_HORIZON)
            if ic_series
            else {"t_stat": np.nan}
        )
        lag_rows.append({"lag": lag, "ic": ic, "t_hac": hac["t_stat"]})

    lag_df = pl.DataFrame(lag_rows)

    # IC half-life: smallest lag where |IC| drops to 50% of |IC at lag 0|
    ic_0 = lag_df.filter(pl.col("lag") == 0)["ic"][0]
    half_target = abs(ic_0) * 0.5
    half_life = None
    ics = lag_df["ic"].to_list()
    lag_vals = lag_df["lag"].to_list()
    for i in range(1, len(lag_vals)):
        if abs(ics[i]) <= half_target and abs(ics[i - 1]) > half_target:
            frac = (abs(ics[i - 1]) - half_target) / (abs(ics[i - 1]) - abs(ics[i]))
            half_life = lag_vals[i - 1] + frac * (lag_vals[i] - lag_vals[i - 1])
            break

    # Assess: does IC at lag 0 have HAC significance AND meaningful decay?
    t_0 = lag_df.filter(pl.col("lag") == 0)["t_hac"][0]
    sig_0 = abs(t_0) > 2.0 if not np.isnan(t_0) else False

    # Check for IC increasing at DISTANT lags (>= 42d, well beyond any lookback).
    # Only flag if the increase is substantial - noise at nearby lags doesn't count.
    distant = [(lag_vals[i], abs(ics[i])) for i in range(1, len(ics)) if lag_vals[i] >= 42]
    ic_increases = any(abs_ic > abs(ic_0) * 1.5 for _, abs_ic in distant)

    if sig_0 and half_life is not None:
        result = "PASS"
        msg = f"IC half-life ≈ {half_life:.0f}d (HAC t₀={t_0:.1f})"
    elif not sig_0 and abs(ic_0) < 0.005:
        result = "STOP"
        msg = f"No timely signal (IC₀={ic_0:.4f}, HAC t₀={t_0:.1f})"
    elif ic_increases:
        result = "STOP"
        msg = f"IC increases at distant lags (HAC t₀={t_0:.1f})"
    elif half_life is not None:
        result = "CAUTION"
        msg = (
            f"IC decays (half-life ≈ {half_life:.0f}d) but marginal significance (HAC t₀={t_0:.1f})"
        )
    else:
        ic_range = [abs(ic) for ic in ics]
        result = "CAUTION"
        msg = (
            f"IC persists without clear decay; |IC| range "
            f"[{min(ic_range):.4f}, {max(ic_range):.4f}] (HAC t₀={t_0:.1f})"
        )

    return lag_df, result, msg


# %% [markdown] tags=[]
# ### Shared-Driver Check
# Test whether a common exogenous factor (Treasury returns) or permutation
# control explains the observed feature-outcome correlation.


# %% tags=[]
def run_shared_driver_check(
    df: pl.DataFrame,
    feature_col: str,
    baseline_ic: float,
    n_permutations: int = 200,
    seed: int = 42,
) -> tuple[dict, str]:
    """Run shared-driver check (Treasury) and permutation control.

    The Treasury check uses a cross-sectional mean approach with HAC inference:
    at each date, compute the cross-sectional mean of the feature, then correlate
    the daily mean-feature series with Treasury forward returns. This avoids
    per-asset time-series regressions and produces a single IC series suitable
    for HAC adjustment.

    Returns: (metrics_dict, result_label)
    """
    # Shared-driver check: cross-sectional mean of feature vs Treasury fwd return
    daily_mean_feat = (
        df.group_by("timestamp")
        .agg(pl.col(feature_col).mean().alias("mean_feature"))
        .sort("timestamp")
    )
    tsy_series = (
        df.select("timestamp", "treasury_fwd_return").unique(subset=["timestamp"]).sort("timestamp")
    )
    shared = daily_mean_feat.join(tsy_series, on="timestamp", how="inner").drop_nulls()

    # Compute rolling rank correlation as an IC series for HAC.
    # The 63-day window is a quarterly convention; consecutive windows share
    # 62/63 of their data, so the effective sample size is much smaller than
    # the number of windows.  The HAC adjustment on the resulting series
    # partially accounts for this, but interpret borderline t-stats with care.
    feat_arr = shared["mean_feature"].to_numpy()
    tsy_arr = shared["treasury_fwd_return"].to_numpy()
    WINDOW = 63
    tsy_ic_series = []
    for i in range(WINDOW, len(feat_arr)):
        f_w = feat_arr[i - WINDOW : i]
        t_w = tsy_arr[i - WINDOW : i]
        if np.std(f_w) > 1e-10 and np.std(t_w) > 1e-10:
            rho, _ = stats.spearmanr(f_w, t_w)
            if not np.isnan(rho):
                tsy_ic_series.append(rho)

    if tsy_ic_series:
        hac_tsy = compute_ic_hac_stats(tsy_ic_series, label_horizon=LABEL_HORIZON)
        ic_tsy = hac_tsy["mean_ic"]
        t_tsy = hac_tsy["t_stat"]
    else:
        ic_tsy, t_tsy = np.nan, np.nan
    tsy_pass = abs(t_tsy) < 2.0 if not np.isnan(t_tsy) else True

    # Permutation control: block permutation at the label horizon.
    null_ics, perm_p, perm_resolution = block_permutation_null(
        df, feature_col, baseline_ic, seed=seed, n_permutations=n_permutations
    )
    perm_signal = perm_p < 0.05

    metrics = {
        "ic_treasury": ic_tsy,
        "t_treasury": t_tsy,
        "tsy_pass": tsy_pass,
        "perm_p": perm_p,
        "perm_resolution": perm_resolution,
        "perm_signal": perm_signal,
        "null_std": float(np.std(null_ics)),
        "n_rolling_windows": len(tsy_ic_series),
        "n_permutations": len(null_ics),
    }

    # The verdict is about the shared driver, and only the Treasury arm speaks to
    # that. The permutation control answers a different question - "is there any
    # signal at all?" - so it is reported beside the verdict and does not enter it.
    #
    # An earlier version said exactly that and then wrote `elif not perm_signal:
    # result = "CAUTION"`, which is the gate the paragraph disclaims. It marked a
    # feature CAUTION for a *shared-driver confound* when its real problem is
    # having no signal to confound - a different finding, already reported by the
    # permutation row and by the timing check.
    #
    # So PASS here means "not explained by the Treasury driver", which for a
    # feature with no signal is true and uninformative. Read it with the
    # permutation p beside it; neither number means anything alone.
    result = "PASS" if tsy_pass else "STOP"

    return metrics, result


# %% [markdown] tags=[]
# ### Regime Heterogeneity
# Partition by VIX regime and test whether signal effectiveness varies
# across market states.


# %% tags=[]
def run_regime_heterogeneity(
    df: pl.DataFrame,
    feature_col: str,
    outcome_col: str = "forward_return",
    vix_low: float = 15,
    vix_high: float = 22,
) -> tuple[list[dict], str, str]:
    """Partition by VIX regime and assess IC heterogeneity.

    This is a heterogeneity diagnostic, not a confounding test. It checks
    whether the feature-outcome association varies across market states. It
    cannot distinguish confounding from genuine effect modification.

    Uses HAC-adjusted inference at each partition. Decision logic:
    - PASS: IC maintains sign, magnitude varies < 2x
    - CAUTION: sign stable but magnitude varies >2x; OR sign flips but
      opposite-sign partition not significant (HAC |t| < 2)
    - STOP: sign flips with significant opposite AND unconditional IC ≈ 0

    Returns: (regime_results, result_label, interpretation_msg)
    """
    partitions = [
        ("Low VIX", df.filter(pl.col("vixcls") < vix_low)),
        ("Mid VIX", df.filter((pl.col("vixcls") >= vix_low) & (pl.col("vixcls") <= vix_high))),
        ("High VIX", df.filter(pl.col("vixcls") > vix_high)),
    ]

    # Unconditional IC with HAC
    ic_unc, _, ic_series_unc = compute_cross_sectional_ic(df, feature_col, outcome_col)
    hac_unc = (
        compute_ic_hac_stats(ic_series_unc, label_horizon=LABEL_HORIZON)
        if ic_series_unc
        else {"t_stat": np.nan}
    )
    unc_sign = np.sign(ic_unc) if ic_unc != 0 else 0
    unc_sig = abs(hac_unc["t_stat"]) > 2.0 if not np.isnan(hac_unc["t_stat"]) else False

    regime_results = []
    for name, partition_df in partitions:
        ic, _, ic_series = compute_cross_sectional_ic(partition_df, feature_col, outcome_col)
        hac = (
            compute_ic_hac_stats(ic_series, label_horizon=LABEL_HORIZON)
            if ic_series
            else {"t_stat": np.nan}
        )
        regime_results.append(
            {"regime": name, "ic": ic, "t_hac": hac["t_stat"], "n": len(partition_df)}
        )

    # Assess sign stability with HAC significance
    valid = [r for r in regime_results if not np.isnan(r["ic"])]
    has_significant_opposite = False
    has_nonsig_opposite = False

    for r in valid:
        if r["ic"] != 0 and np.sign(r["ic"]) != unc_sign:
            if not np.isnan(r["t_hac"]) and abs(r["t_hac"]) > 2.0:
                has_significant_opposite = True
            else:
                has_nonsig_opposite = True

    signs = [np.sign(r["ic"]) for r in valid if r["ic"] != 0]
    all_same_sign = len(set(signs)) <= 1 if signs else False

    if all_same_sign:
        ic_vals = [abs(r["ic"]) for r in valid if r["ic"] != 0]
        magnitude_ratio = (
            max(ic_vals) / min(ic_vals) if ic_vals and min(ic_vals) > 1e-6 else float("inf")
        )
        if magnitude_ratio < 2.0:
            result = "PASS"
            msg = "IC maintains sign and magnitude across VIX regimes"
        else:
            result = "CAUTION"
            msg = f"IC sign stable but magnitude varies {magnitude_ratio:.1f}x"
    elif has_significant_opposite and not unc_sig:
        result = "STOP"
        msg = "IC flips sign (HAC sig.) and unconditional IC ≈ 0 - aggregation artifact"
    elif has_significant_opposite and unc_sig:
        result = "CAUTION"
        msg = "IC flips sign (HAC sig.) but unconditional IC is itself significant"
    else:
        result = "CAUTION"
        msg = "IC changes sign in one regime but not significantly (HAC)"

    return regime_results, result, msg


# %% [markdown] tags=[]
# ## 8. Run All Checks on Both Features
#
# We apply each plausibility check to 12-1 momentum and 1-day reversal, collecting
# results for the comparison scorecard. All t-statistics use Newey-West (HAC)
# standard errors to account for serial dependence in overlapping IC series.

# %% [markdown] tags=[]
# ### 8.1 Timing Placebo
#
# For a rolling-window feature with lookback $L$, a $\Delta$-shifted version
# shares roughly $(L - \Delta)/L$ of its inputs with the original. This creates
# a mechanical floor on IC persistence. We therefore treat the IC-lag profile as
# a **diagnostic** (how does IC behave as information ages?) rather than applying
# a hard decay threshold. More weight should be placed on lags beyond the lookback
# window.
#
# For 12-1 momentum (lookback ~231 trading days), we extend the lag grid to
# 252d so the furthest lag reaches beyond the lookback window entirely.

# %% tags=[]
timing = {}

# 12-1 Momentum: extended lag grid for long lookback (~231d)
lag_df, result, msg = run_timing_placebo(
    analysis.drop_nulls(subset=["mom_12_1", "forward_return"]),
    "mom_12_1",
    lags=[0, 5, 21, 42, 63, 126, 252],
)
timing["mom_12_1"] = {"lag_df": lag_df, "result": result, "msg": msg}
print(f"{FEATURES['mom_12_1']}: {result} - {msg}")

# 1-day Reversal: fine-grained grid for short lookback (1d)
lag_df, result, msg = run_timing_placebo(
    analysis.drop_nulls(subset=["rev_1d", "forward_return"]),
    "rev_1d",
    lags=[0, 1, 2, 5, 10, 21],
)
timing["rev_1d"] = {"lag_df": lag_df, "result": result, "msg": msg}
print(f"{FEATURES['rev_1d']}: {result} - {msg}")

# %% tags=[]
# Side-by-side timing placebo visualization
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["(a) 12-1 Momentum", "(b) 1-day Reversal"],
    shared_yaxes=True,
)

for i, feat_col in enumerate(FEATURES):
    lag_df = timing[feat_col]["lag_df"]
    fig.add_trace(
        go.Bar(
            x=[f"+{lag}d" for lag in lag_df["lag"].to_list()],
            y=lag_df["ic"].to_list(),
            marker_color=COLORS["blue"],
            showlegend=False,
        ),
        row=1,
        col=i + 1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=i + 1)

fig.update_layout(
    height=350,
    title="12-1 momentum keeps most of its IC even when the feature is stale",
)
fig.update_yaxes(title_text="Mean IC", row=1, col=1)
fig.update_xaxes(title_text="Feature Lag", row=1, col=1)
fig.update_xaxes(title_text="Feature Lag", row=1, col=2)
fig.show()

# %% [markdown] tags=[]
# 12-1 momentum IC is strongest at lag 0 (IC = 0.053) and decays gradually to
# IC ≈ 0.034 at lag 252d. The persistence is partly mechanical: with a 231-day
# lookback, shifted features share most of their input data with the original.
# The key diagnostic is that IC peaks at lag 0 - the most recent version of the
# feature is the most informative - and even the fully stale version (lag 252d,
# beyond the lookback) retains some residual predictability. Reversal IC is near
# zero at all lags - there is no timely information to decay.

# %% [markdown] tags=[]
# ### 8.2 Shared-Driver Check
#
# We test whether each feature's cross-sectional mean predicts Treasury (IEF)
# 21-day forward returns. For 12-1 momentum, Treasury co-movement through
# risk-on/risk-off dynamics is plausible, so a non-zero Treasury IC would flag
# a shared-driver concern rather than strictly falsify the mechanism.

# %% tags=[]
n_perms = N_PERMUTATIONS

nc = {}
for feat_col in FEATURES:
    sub = analysis.drop_nulls(subset=[feat_col, "forward_return", "treasury_fwd_return"])
    metrics, result = run_shared_driver_check(
        sub, feat_col, baseline[feat_col]["ic"], n_permutations=n_perms
    )
    nc[feat_col] = {"metrics": metrics, "result": result}
    m = metrics
    print(
        f"{FEATURES[feat_col]}: {result}\n"
        f"  Treasury IC={m['ic_treasury']:.4f} (HAC t={m['t_treasury']:.2f})\n"
        f"  Block-permutation p={m['perm_p']:.4f} "
        f"(B={m['n_permutations']}, finest resolvable p={m['perm_resolution']:.4f}, "
        f"null sd={m['null_std']:.4f})"
    )

# %% [markdown] tags=[]
# The verdict on this row is the **Treasury** column, and only that column. Neither
# feature's cross-sectional mean predicts Treasury forward returns, so neither shows
# the shared-driver confound this check exists to detect.
#
# The permutation column is reported beside it but does not set the verdict, because
# it answers a different question - *is there any signal here at all?* - and the
# timing check already reports that. Gating the shared-driver verdict on it would
# mark a feature as having a confound when its actual problem is having nothing to
# confound.
#
# Two things to read off the permutation column. First, the p-value is computed as
# $(r+1)/(B+1)$ and printed beside the finest value $B$ permutations can resolve; a
# permutation p-value is never exactly zero, because the observed assignment is
# itself one of the arrangements under the null. Second, the null is **block**
# permuted at the 21-day label horizon, so its spread reflects how persistent this
# data actually is. An independent within-date shuffle would produce a null roughly
# an order of magnitude too narrow and would reject nearly anything.

# %% [markdown] tags=[]
# ### 8.3 Regime Heterogeneity (VIX Regimes)
#
# A sign flip is only flagged as STOP if the opposite-sign partition has
# HAC $|t| > 2$ **and** the unconditional IC is itself not significant. In
# low-power subsamples, small ICs with the wrong sign may reflect noise rather
# than genuine heterogeneity. This criterion is stated here as an a priori
# design choice.

# %% tags=[]
# Tercile cutoffs; results are qualitatively similar with median splits or
# quartiles - the key diagnostic is sign stability, not exact boundaries.
VIX_LOW_THRESHOLD = 15
VIX_HIGH_THRESHOLD = 22

cond = {}
for feat_col in FEATURES:
    sub = analysis.drop_nulls(subset=[feat_col, "forward_return"])
    regime_results, result, msg = run_regime_heterogeneity(
        sub, feat_col, vix_low=VIX_LOW_THRESHOLD, vix_high=VIX_HIGH_THRESHOLD
    )
    cond[feat_col] = {"regimes": regime_results, "result": result, "msg": msg}
    print(f"\n{FEATURES[feat_col]}: {result} - {msg}")
    for r in regime_results:
        print(f"  {r['regime']:10s}: IC={r['ic']:+.4f} (HAC t={r['t_hac']:+.2f}, n={r['n']:,})")

# %% tags=[]
# Side-by-side conditioning visualization
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["(a) 12-1 Momentum", "(b) 1-day Reversal"],
    shared_yaxes=True,
)

for i, feat_col in enumerate(FEATURES):
    regimes = cond[feat_col]["regimes"]
    ic_unc = baseline[feat_col]["ic"]
    x_labels = [r["regime"] for r in regimes] + ["Unconditional"]
    y_values = [r["ic"] for r in regimes] + [ic_unc]
    colors = [COLORS["blue"]] * len(regimes) + [COLORS["amber"]]

    fig.add_trace(
        go.Bar(x=x_labels, y=y_values, marker_color=colors, showlegend=False),
        row=1,
        col=i + 1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=i + 1)

fig.update_layout(
    height=350,
    title="Momentum concentrates in calm regimes; reversal flips sign across them",
)
fig.update_yaxes(title_text="Mean IC", row=1, col=1)
fig.show()

# %% [markdown] tags=[]
# ### Publication Figure Artifact
#
# The book scorecard figure reads this compact artifact so formatting changes do
# not rerun ETF/VIX preparation or permutation checks.

# %% tags=[]
FIGURE_7_10_LAGS = [0, 1, 5, 21, 63, 126, 252]
FIGURE_7_10_FEATURES = {"mom_12_1": "12-1 Momentum", "rev_1d": "1-day Reversal"}


def _figure_7_10_timing(df: pl.DataFrame, feature_col: str) -> np.ndarray:
    # shift().over("symbol") is order-sensitive - sort chronologically within
    # symbol first so the lagged value aligns with the correct earlier row.
    df = df.sort(["symbol", "timestamp"])
    lag_ics = []
    for lag in FIGURE_7_10_LAGS:
        shifted = df.with_columns(
            pl.col(feature_col).shift(lag).over("symbol").alias("_shifted")
        ).drop_nulls(subset=["_shifted", "forward_return"])
        lag_ics.append(compute_cross_sectional_ic(shifted, "_shifted", "forward_return")[0])
    return np.array(lag_ics)


def _figure_7_10_regime(df: pl.DataFrame, feature_col: str) -> np.ndarray:
    regimes = [
        df.filter(pl.col("vixcls") < VIX_LOW_THRESHOLD),
        df.filter(
            (pl.col("vixcls") >= VIX_LOW_THRESHOLD) & (pl.col("vixcls") <= VIX_HIGH_THRESHOLD)
        ),
        df.filter(pl.col("vixcls") > VIX_HIGH_THRESHOLD),
    ]
    return np.array(
        [compute_cross_sectional_ic(regime, feature_col, "forward_return")[0] for regime in regimes]
    )


def _figure_7_10_permutation(
    df: pl.DataFrame, feature_col: str, baseline_ic: float
) -> tuple[np.ndarray, float, float]:
    # Same null as the shared-driver check above, and for the same two reasons.
    #
    # This path used to draw its own: an independent within-date shuffle, scored
    # with `np.mean(|null| >= |observed|)`. Both are the defects §4 corrects - the
    # iid null is roughly an order of magnitude too narrow against overlapping
    # labels, and the plain proportion can return exactly 0, which no permutation
    # test can resolve. It mattered more here than anywhere else in the notebook,
    # because this p-value is what the published figure reports.
    return block_permutation_null(
        df, feature_col, baseline_ic, seed=SEED, n_permutations=N_PERMUTATIONS
    )


def write_figure_7_10_artifact() -> Path:
    artifact_data: dict[str, np.ndarray | float] = {"common_lags": np.array(FIGURE_7_10_LAGS)}
    for feature_col in FIGURE_7_10_FEATURES:
        sub = analysis.drop_nulls(subset=[feature_col, "forward_return", "vixcls"])
        baseline_ic = compute_cross_sectional_ic(sub, feature_col, "forward_return")[0]
        perm_ics, perm_p, perm_resolution = _figure_7_10_permutation(sub, feature_col, baseline_ic)
        artifact_data[f"baseline__{feature_col}"] = baseline_ic
        artifact_data[f"timing__{feature_col}"] = _figure_7_10_timing(sub, feature_col)
        artifact_data[f"regime__{feature_col}"] = _figure_7_10_regime(sub, feature_col)
        artifact_data[f"perm__{feature_col}"] = perm_ics
        artifact_data[f"perm_p__{feature_col}"] = perm_p
        # The floor travels with the value: a figure caption quoting p without it
        # cannot tell a measurement from the smallest number B can express.
        artifact_data[f"perm_resolution__{feature_col}"] = perm_resolution
        print(
            f"  {feature_col}: permutation p={perm_p:.4f} "
            f"(B={N_PERMUTATIONS}, finest resolvable p={perm_resolution:.4f})"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT_DIR / "figure_7_10_inputs.npz"
    np.savez(artifact, **artifact_data)
    return artifact


figure_7_10_artifact = write_figure_7_10_artifact()
print(f"Wrote publication figure artifact: {figure_7_10_artifact}")

# %% [markdown] tags=[]
# 12-1 momentum maintains positive IC across all VIX regimes, though the
# magnitude varies roughly 5x - strongest in low VIX (IC = 0.086, HAC $t$ = 3.0)
# and attenuated in high VIX (IC = 0.017, $t$ = 0.4). This is consistent with
# the well-documented "momentum crash" phenomenon: momentum strategies suffer in
# high-volatility environments (Daniel and Moskowitz 2016). The sign stability
# across all regimes earns CAUTION (magnitude variation) rather than STOP.
#
# Reversal shows a significant sign flip: negative IC in low-VIX (HAC $t$ = −2.6)
# and positive IC in high-VIX (HAC $t$ = 2.5), producing a near-zero unconditional
# IC. This is a textbook aggregation artifact - the feature encodes opposite
# information depending on volatility state - and earns STOP.
#
# **Important**: this check cannot distinguish confounding from genuine effect
# modification. A feature whose IC varies by regime may be confounded *or* may
# have a mechanism that operates differently across states. Chapter 15 provides
# multivariate sensitivity analysis to separate these cases.

# %% [markdown] tags=[]
# ## 9. Collider Bias: A Synthetic Demonstration
#
# We place this simulation here - after the three main checks - because its
# purpose is different: it illustrates a DAG concept from Section 2 rather
# than diagnosing a specific feature. Readers who want to reinforce the DAG
# vocabulary before running the checks can read this section first.
#
# The text defines **collider bias**: conditioning on a common *effect* of feature
# and label creates a spurious association between them. This short simulation
# makes the concept concrete.
#
# **Setup**: Momentum ($X$) and forward returns ($Y$) are independent. Fund flows
# ($Z$) are a function of both: $Z = \beta_1 X + \beta_2 Y + \varepsilon$. Because
# $Z$ is a collider, conditioning on it opens a path between $X$ and $Y$.

# %% tags=[]
# Collider-bias simulation
rng_collider = np.random.default_rng(123)
n_obs = 5000

# X (momentum) and Y (forward return) are independent
X = rng_collider.standard_normal(n_obs)
Y = rng_collider.standard_normal(n_obs)

# Z (fund flows) is a common effect of both
Z = 0.5 * X + 0.5 * Y + rng_collider.standard_normal(n_obs) * 0.5

# Unconditional correlation: should be ~0
rho_uncond, _ = stats.spearmanr(X, Y)

# Conditional on Z > median (selecting funds with high flows): spurious correlation
high_Z = np.median(Z) < Z
rho_cond, _ = stats.spearmanr(X[high_Z], Y[high_Z])

print(
    pl.DataFrame(
        {
            "condition": ["Unconditional", "Conditional on Z > median"],
            "spearman_rho": [f"{rho_uncond:.4f}", f"{rho_cond:.4f}"],
            "interpretation": [
                "X and Y are independent (as generated)",
                "Spurious negative correlation - collider bias",
            ],
        }
    )
)

# %% [markdown] tags=[]
# Conditioning on high fund flows (the collider) induces a negative correlation
# between momentum and returns: among high-flow funds, strong momentum implies
# weaker returns and vice versa. This is an artifact of the conditioning, not a
# real effect. The practical lesson: never condition on a variable that is
# *caused by* both the feature and the label.

# %% [markdown] tags=[]
# ## 10. Plausibility Scorecard
#
# Aggregate results into a dual-feature scorecard aligned with the
# proceed / revise / stop framework from Section 7.3.

# %% tags=[]
# Build scorecard for both features
scorecard_rows = []
for feat_col, feat_label in FEATURES.items():
    nc_m = nc[feat_col]["metrics"]
    cond_regimes = cond[feat_col]["regimes"]
    t_lag = timing[feat_col]["lag_df"]
    ic_0 = t_lag.filter(pl.col("lag") == 0)["ic"][0]

    # IC decay: find the last lag in the grid and compute % decay
    last_lag = t_lag["lag"].max()
    ic_last = t_lag.filter(pl.col("lag") == last_lag)["ic"][0]
    decay = ((ic_0 - ic_last) / abs(ic_0) * 100) if ic_0 != 0 else 0

    scorecard_rows.append(
        {
            "feature": feat_label,
            "timing": timing[feat_col]["result"],
            "timing_detail": f"IC decay {decay:.0f}% at {last_lag}d",
            "shared_driver": nc[feat_col]["result"],
            "sd_detail": f"Tsy HAC t={nc_m['t_treasury']:.1f}, perm p={nc_m['perm_p']:.3f}",
            "regime": cond[feat_col]["result"],
            "regime_detail": ", ".join(f"{r['regime']}: {r['ic']:+.4f}" for r in cond_regimes),
        }
    )

scorecard = pl.DataFrame(scorecard_rows)
print(f"Label: 21-day forward return | Universe: ETFs | N={len(analysis):,}")
display(scorecard.select("feature", "timing", "shared_driver", "regime"))

# %% tags=[]
# Triage decisions
for feat_col, feat_label in FEATURES.items():
    results = [timing[feat_col]["result"], nc[feat_col]["result"], cond[feat_col]["result"]]
    n_pass = sum(r == "PASS" for r in results)
    n_stop = sum(r == "STOP" for r in results)

    if n_stop > 0:
        decision = "STOP"
    elif n_pass == 3:
        decision = "PROCEED"
    else:
        decision = "REVISE"

    print(f"{feat_label}: {decision} ({n_pass}/3 passed, checks={results})")

# %% [markdown] tags=[]
# **Interpretation**: 12-1 momentum earns REVISE or PROCEED - it carries timely
# cross-sectional information (timing check) that does not predict unrelated
# Treasury returns (shared-driver check). The regime heterogeneity check reveals
# that the effect attenuates in high-VIX markets, an actionable finding that
# motivates regime-conditional modeling in later chapters.
#
# 1-day reversal earns STOP - no signal at the 21-day horizon, no regime
# drives a significant IC, and its IC is indistinguishable from the permutation
# null. The framework correctly rejects a non-signal.
#
# The contrast illustrates the triage value: without these checks, a researcher
# might spend chapters modeling reversal on the wrong horizon. The scan in
# Section 4 and the mechanism checks here redirect effort toward features with
# genuine cross-sectional predictive power.

# %% [markdown] tags=[]
# ## 11. Event-Time Alignment (Concept Only)
#
# This fourth check from Section 7.5 requires event-specific features (e.g.,
# earnings surprises, FOMC announcements). Neither momentum nor reversal is
# event-driven, so the check does not apply here. Chapter 8 demonstrates
# event-time alignment with fundamental features, and Chapter 9 with regime
# transitions.
#
# The concept: for event-motivated features, IC should concentrate in the window
# where the mechanism predicts an effect. Red flags:
# - IC peaking *before* the event (anticipation or leakage)
# - Symmetric pre- and post-event IC (confound, not causal timing)

# %% [markdown] tags=[]
# ## Key Takeaways
#
# 1. **Match feature lookback to label horizon**: The scan in Section 4 shows that
#    long-lookback momentum features (126d+) carry significant cross-sectional
#    information at monthly horizons, while short-term features (1d, 5d) are mostly
#    noise. A 5-day horizon yields 3/10 significant features (252d and 12-1
#    momentum plus a 1d-reversal hit that the mechanism checks reject as
#    regime-driven); at 21d, 12-1 momentum reaches HAC $t = 2.6$. Horizon selection
#    is a modeling decision that changes which features appear informative.
#
# 2. **State the mechanism first**: Before testing, draw the assumed DAG and identify
#    potential confounders, mediators, and colliders.
#
# 3. **Timing placebos** are diagnostics, not gates. We report IC half-life and note
#    that rolling-window features create mechanical IC persistence through input
#    overlap. Focus on decay at lags beyond the lookback window. 12-1 momentum IC
#    was strongest at lag 0 and decayed gradually (0.053 → 0.034 at lag 252d);
#    reversal had no signal to decay.
#
# 4. **Shared-driver checks** test whether a feature predicts an outcome it shouldn't.
#    We used Treasury forward returns (economic control) and permutation (statistical
#    control). The choice of control outcome is feature-dependent: for momentum,
#    Treasury co-movement through risk-on/risk-off dynamics is plausible, making this
#    a shared-driver diagnostic rather than a strict falsification.
#
# 5. **Regime heterogeneity** reveals whether IC varies across market states but
#    cannot distinguish confounding from genuine effect modification. Momentum's
#    IC attenuated in high-VIX markets (5x magnitude variation) - consistent with
#    the "momentum crash" phenomenon - while maintaining sign across regimes.
#    Reversal showed a significant sign flip: an aggregation artifact where
#    opposite-sign regime ICs cancel to near-zero unconditional IC.
#
# 6. **No feature is guaranteed to pass all checks**. The triage framework produces
#    nuanced decisions: STOP (abandon), REVISE (investigate), or PROCEED (continue).
#    12-1 momentum earned REVISE, not PROCEED - a realistic outcome that motivates
#    regime-aware modeling in later chapters. Reversal earned STOP, correctly
#    redirecting effort away from an uninformative feature.
#
# 7. **Bivariate limitation**: These checks test one feature at a time. The most
#    dangerous confounders in feature engineering are often *other features* in the
#    model. Chapter 15 provides multivariate sensitivity analysis (partial IC,
#    double/debiased ML) for the features that survive this initial screen.
#
# **Next**: Chapter 8 applies event-time alignment with fundamental features.
# Chapter 15 introduces the formal causal inference toolkit (identification, DML,
# sensitivity analysis).
