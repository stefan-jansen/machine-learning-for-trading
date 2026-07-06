# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ETFs: Model-Based Features (Per-Fold Temporal Models)
#
# This notebook fits temporal models and extracts features that capture
# latent market dynamics. All models are fit **per CV fold** on training
# data only, eliminating parameter-level look-ahead bias. It produces
# features for three model families:
#
# 1. **HMM Regime Detection**: 2-state Gaussian HMM on aggregate market (SPY)
#    with filtered (causal) probabilities, regime transition indicators, and
#    regime duration features.
# 2. **Fractional Differencing**: Memory-preserving stationarity transforms
#    on 10 reference ETFs spanning all major asset classes.
# 3. **GARCH(1,1) Conditional Volatility**: Per-ETF volatility forecasts that
#    provide each asset's own risk dynamics.
#
# Each row in the output carries a `fold` column identifying which fold's
# model produced it. This enables downstream CV to use the correct
# (non-leaked) features for each fold.
#
# ## Learning Objectives
#
# - Fit temporal models per CV fold to avoid parameter-level look-ahead
# - Fit a 2-state HMM with k-means initialization and multiple restarts
# - Compute filtered (not smoothed) probabilities for production use
# - Derive regime transition and duration features from filtered probs
# - Apply fractional differencing with fixed $d$ values by asset class
# - Fit per-asset GARCH(1,1) with frozen-parameter filtering
# - Combine date-level and per-asset features into a per-fold panel
#
# ## Book Reference
#
# Chapter 9, Sections 9.1 (Fractional Differencing), 9.3 (GARCH), and
# 9.5 (Regime Features)
#
# ## Prerequisites
#
# - [`02_labels`](02_labels.ipynb) (produces label parquet files)
# - `03_financial_features.py` (produces `features/financial.parquet`)

# %%
"""ETFs: Model-Based Features (per-fold HMM + FFD + GARCH)."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from ml4t.diagnostic.evaluation.stats import robust_ic
from ml4t.engineer.features.fdiff import ffdiff
from sklearn.cluster import KMeans

from data import load_etfs
from utils.cv_splits import generate_cv_splits, load_evaluation_config
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")


# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
START_DATE = None  # None = use full dataset
N_RESTARTS = 10
GARCH_MIN_OBS = 504  # Minimum observations for GARCH fit (~2 years)
MAX_SYMBOLS = 0  # 0 = all symbols (production)

# %%
CASE_DIR = get_case_study_dir("etfs")

prices = load_etfs()
print(f"Prices: {len(prices):,} rows, {prices['symbol'].n_unique()} assets")

# %% [markdown]
# ## CV Fold Setup
#
# We load the walk-forward CV splits from `setup.yaml` and add a holdout
# fold. All temporal models are fit per fold on training data only.

# %%
# Generate CV splits
cv_splits = generate_cv_splits(prices, case_study_id="etfs", label_buffer="21D")
eval_config = load_evaluation_config("etfs")

# Add holdout fold: fit on all pre-holdout data, extract for holdout period
holdout_start = str(eval_config["holdout_start"])
holdout_end = str(eval_config.get("holdout_end", prices["timestamp"].max()))

# The last CV fold's val_end is the boundary before holdout
# For holdout, train on everything up to holdout_start
holdout_fold = {
    "fold": len(cv_splits),
    "train_start": cv_splits[0]["train_start"],
    "train_end": holdout_start,
    "val_start": holdout_start,
    "val_end": str(holdout_end),
}
all_folds = cv_splits + [holdout_fold]

n_cv = len(cv_splits)
n_total = len(all_folds)
print(f"CV folds: {n_cv}, plus holdout fold (fold {n_cv})")
for fold in all_folds:
    label = "HOLDOUT" if fold["fold"] == n_cv else f"Fold {fold['fold']}"
    print(
        f"  {label}: train {fold['train_start']}..{fold['train_end']}, "
        f"val {fold['val_start']}..{fold['val_end']}"
    )

# %% [markdown]
# ## Part 1: HMM Regime Detection
#
# We fit a 2-state Gaussian HMM on SPY returns + volatility **per fold**.
# The aggregate market drives regime classification; individual ETFs
# inherit it.
#
# ### Why SPY Only
#
# Using a single aggregate proxy (SPY) rather than per-asset HMMs:
# - Avoids overfitting 100 independent HMMs
# - Regime is a market-level phenomenon (risk-on/risk-off)
# - All ETFs inherit the same regime state, ensuring cross-sectional consistency

# %%
spy_full = (
    prices.filter(pl.col("symbol") == "SPY")
    .sort("timestamp")
    .with_columns(
        log_ret=(pl.col("close").log().diff() * 100),
        vol_21d=(pl.col("close").log().diff().rolling_std(window_size=21) * 100 * np.sqrt(252)),
    )
    .drop_nulls()
)

print(
    f"SPY: {len(spy_full):,} observations ({spy_full['timestamp'].min()} to {spy_full['timestamp'].max()})"
)

# %% [markdown]
# ### K-Means-Seeded HMM Fitting
#
# K-means clustering provides better initial emission parameters than random
# initialization, reducing sensitivity to local optima.


# %%
def fit_hmm_kmeans_init(X: np.ndarray, n_states: int, random_state: int = 42) -> GaussianHMM:
    """Fit HMM with k-means-seeded initialization."""
    kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
    kmeans.fit(X)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
        init_params="st",  # Only init startprob and transmat
    )

    model.means_ = kmeans.cluster_centers_
    model.covars_ = np.array(
        [np.cov(X[kmeans.labels_ == k].T) + np.eye(X.shape[1]) * 1e-6 for k in range(n_states)]
    )

    model.fit(X)
    return model


# %% [markdown]
# ### Label Switching Prevention
#
# Sort states by variance (ascending) so State 0 is always "low volatility"
# (calm) and State 1 is always "high volatility" (stressed).


# %%
def sort_states_by_variance(model: GaussianHMM) -> np.ndarray:
    """Sort HMM states by variance (ascending) for consistent labeling."""
    variances = np.array([np.trace(model.covars_[k]) for k in range(model.n_components)])
    return np.argsort(variances)  # Low vol first


def relabel_states(states: np.ndarray, probs: np.ndarray, order: np.ndarray) -> tuple:
    """Relabel states according to the given order."""
    inv_order = np.argsort(order)
    new_states = inv_order[states]
    new_probs = probs[:, order]
    return new_states, new_probs


# %% [markdown]
# ### Filtered Probabilities (No Look-Ahead)
#
# In production, we must use **filtered** probabilities $P(z_t | x_{1:t})$
# which condition only on past and present observations. hmmlearn's
# `predict_proba()` returns **smoothed** probabilities $P(z_t | x_{1:T})$
# which use future data and would introduce look-ahead bias.


# %%
def compute_filtered_probs(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Compute filtered probabilities P(state_t | obs_{1:t}).

    Uses the forward algorithm, then normalizes.
    """
    framelogprob = model._compute_log_likelihood(X)

    n_samples = X.shape[0]
    n_components = model.n_components

    log_startprob = np.log(model.startprob_ + 1e-300)
    log_transmat = np.log(model.transmat_ + 1e-300)

    # Forward pass (log-domain for numerical stability)
    fwdlattice = np.zeros((n_samples, n_components))
    fwdlattice[0] = log_startprob + framelogprob[0]

    for t in range(1, n_samples):
        for j in range(n_components):
            fwdlattice[t, j] = framelogprob[t, j] + np.logaddexp.reduce(
                fwdlattice[t - 1] + log_transmat[:, j]
            )

    # Normalize to get probabilities
    log_normalizer = np.logaddexp.reduce(fwdlattice, axis=1, keepdims=True)
    filtered = np.exp(fwdlattice - log_normalizer)

    return filtered


# %% [markdown]
# ### Derive Regime Features from Filtered Probabilities
#
# From filtered probabilities, derive three feature types:
# 1. **regime_prob_stress**: Filtered probability of being in the high-vol state
# 2. **regime_transition**: Absolute change in stress probability (detects regime shifts)
# 3. **regime_duration**: Days since last regime change (persistence indicator)


# %%
def derive_regime_features(
    timestamps: pl.Series,
    filtered_probs: np.ndarray,
    states: np.ndarray,
    order: np.ndarray,
) -> pl.DataFrame:
    """Derive regime features from HMM output for a single fold window."""
    states_sorted, filtered_sorted = relabel_states(states, filtered_probs, order)

    regime_prob_stress = filtered_sorted[:, 1]  # P(high-vol state)

    # Transition: absolute 1-day change in stress probability
    regime_transition = np.abs(np.diff(regime_prob_stress, prepend=regime_prob_stress[0]))

    # Duration: days since last regime change
    regime_duration = np.zeros(len(states_sorted))
    current_run = 0
    for i in range(len(states_sorted)):
        if i == 0 or states_sorted[i] != states_sorted[i - 1]:
            current_run = 1
        else:
            current_run += 1
        regime_duration[i] = current_run

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "regime_prob_stress": regime_prob_stress,
            "regime_transition": regime_transition,
            "regime_log_duration": np.log1p(regime_duration),
        }
    )


# %% [markdown]
# ### Illustrative Full-Sample HMM Fit
#
# Before the per-fold loop, we fit one full-sample HMM for visualization
# purposes only. This shows readers the regime overlay on SPY and validates
# the methodology. **These features are NOT used in the saved output.**

# %%
N_STATES = 2
X_full = spy_full.select(["log_ret", "vol_21d"]).to_numpy()

best_model_illustrative = None
best_ll = -np.inf

for seed in range(N_RESTARTS):
    try:
        model = fit_hmm_kmeans_init(X_full, n_states=N_STATES, random_state=seed)
        ll = model.score(X_full)
        if ll > best_ll:
            best_ll = ll
            best_model_illustrative = model
    except Exception:
        continue

print(f"Illustrative full-sample HMM: best log-likelihood = {best_ll:.1f}")

order_illustrative = sort_states_by_variance(best_model_illustrative)
filtered_illustrative = compute_filtered_probs(best_model_illustrative, X_full)
states_illustrative = best_model_illustrative.predict(X_full)
states_sorted_ill, _ = relabel_states(
    states_illustrative, filtered_illustrative, order_illustrative
)

for k in range(N_STATES):
    mask = states_sorted_ill == k
    label = "Low-Vol" if k == 0 else "High-Vol"
    mean_ret = X_full[mask, 0].mean()
    mean_vol = X_full[mask, 1].mean()
    pct = mask.mean()
    print(f"State {k} ({label}): {pct:.1%} of time, mean ret={mean_ret:.3f}%, vol={mean_vol:.1f}%")

# %% [markdown]
# ### Regime Overlay on SPY (Illustrative)
#
# Shading stressed periods on SPY cumulative returns shows whether the HMM
# captures known market episodes (GFC, COVID, 2022 rate shock).

# %%
spy_cum = spy_full.with_columns(cum_ret=(pl.col("close") / pl.col("close").first() - 1) * 100)

fig_regime, ax = plt.subplots(figsize=(12, 5))
dates = spy_cum["timestamp"].to_numpy()
cum_ret = spy_cum["cum_ret"].to_numpy()
ax.plot(dates, cum_ret, linewidth=0.8, color="0.3")

# Shade stressed periods
stressed = states_sorted_ill == 1
in_stress = False
start = None
for i in range(len(stressed)):
    if stressed[i] and not in_stress:
        start = dates[i]
        in_stress = True
    elif not stressed[i] and in_stress:
        ax.axvspan(start, dates[i], alpha=0.15, color="red", linewidth=0)
        in_stress = False
if in_stress:
    ax.axvspan(start, dates[-1], alpha=0.15, color="red", linewidth=0)

ax.set_ylabel("Cumulative Return (%)")
ax.set_title("SPY with HMM Stress Regimes (full-sample illustrative)")
fig_regime.tight_layout()
plt.show()

# %% [markdown]
# ### Per-Fold HMM Fitting
#
# For each fold, we fit the HMM on **training data only**, then apply the
# forward algorithm to the full fold window (train_start through val_end)
# for filtered probabilities. The model parameters $\theta$ are estimated
# exclusively from training observations, eliminating parameter-level
# look-ahead.

# %%
hmm_fold_results = []

for fold in all_folds:
    fold_idx = fold["fold"]
    train_start, train_end = fold["train_start"], fold["train_end"]
    val_end = fold["val_end"]

    # Training data: fit HMM parameters
    spy_train = spy_full.filter(
        (pl.col("timestamp") >= pl.lit(train_start).cast(pl.Date))
        & (pl.col("timestamp") < pl.lit(train_end).cast(pl.Date))
    )
    X_train = spy_train.select(["log_ret", "vol_21d"]).to_numpy()

    if len(X_train) < 252:
        print(
            f"  Fold {fold_idx}: insufficient SPY training data ({len(X_train)} obs), skipping HMM"
        )
        continue

    # Fit HMM with multiple restarts on training data
    best_fold_model = None
    best_fold_ll = -np.inf
    for seed in range(N_RESTARTS):
        try:
            m = fit_hmm_kmeans_init(X_train, n_states=N_STATES, random_state=seed)
            ll = m.score(X_train)
            if ll > best_fold_ll:
                best_fold_ll = ll
                best_fold_model = m
        except Exception:
            continue

    if best_fold_model is None:
        print(f"  Fold {fold_idx}: HMM fitting failed")
        continue

    order = sort_states_by_variance(best_fold_model)

    # Full fold window (train_start through val_end) for filtered probs
    spy_fold = spy_full.filter(
        (pl.col("timestamp") >= pl.lit(train_start).cast(pl.Date))
        & (pl.col("timestamp") <= pl.lit(val_end).cast(pl.Date))
    )
    X_fold = spy_fold.select(["log_ret", "vol_21d"]).to_numpy()

    filtered = compute_filtered_probs(best_fold_model, X_fold)
    states = best_fold_model.predict(X_fold)

    regime_df = derive_regime_features(spy_fold["timestamp"], filtered, states, order)
    regime_df = regime_df.with_columns(pl.lit(fold_idx).alias("fold"))

    hmm_fold_results.append(regime_df)
    print(
        f"  Fold {fold_idx}: HMM LL={best_fold_ll:.1f}, {len(regime_df)} dates, "
        f"stress={regime_df['regime_prob_stress'].mean():.3f}"
    )

hmm_features = (
    pl.concat(hmm_fold_results)
    if hmm_fold_results
    else pl.DataFrame(schema={"timestamp": pl.Date, "fold": pl.Int64})
)
n_hmm_folds = hmm_features["fold"].n_unique() if len(hmm_features) > 0 else 0
print(f"\nHMM features: {len(hmm_features):,} rows across {n_hmm_folds} folds")

# %% [markdown]
# ## Part 2: Fractional Differencing (Per Fold)
#
# Fractional differencing preserves long-range memory while achieving
# stationarity. We apply fixed $d$ values by asset class to 10 reference
# ETFs spanning all major asset classes in our universe.
#
# **Why fixed $d$?** Using pre-specified $d$ values avoids parameter estimation
# lookahead entirely -- no data-dependent optimization, so the transform is
# purely mechanical. We still compute per fold so each fold window gets a
# clean series starting from its own `train_start`.
#
# | Asset Class     | Reference ETFs | $d$ |
# |-----------------|---------------|-----|
# | US Equities     | SPY, QQQ, IWM | 0.4 |
# | Int'l Equities  | EFA, EEM      | 0.4 |
# | Fixed Income    | TLT           | 0.5 |
# | Gold            | GLD           | 0.4 |
# | Real Estate     | VNQ           | 0.4 |
# | High Yield      | HYG           | 0.5 |
# | Inv. Grade      | LQD           | 0.5 |

# %%
REFERENCE_ETFS = {
    "SPY": 0.4,  # US large-cap equities
    "QQQ": 0.4,  # US tech equities
    "IWM": 0.4,  # US small-cap equities
    "EFA": 0.4,  # International developed
    "EEM": 0.4,  # Emerging markets
    "TLT": 0.5,  # Long-term treasuries
    "GLD": 0.4,  # Gold
    "VNQ": 0.4,  # Real estate
    "HYG": 0.5,  # High yield bonds
    "LQD": 0.5,  # Investment grade bonds
}

# %% [markdown]
# ### Per-Fold FFD Application
#
# For each fold, apply fractional differencing to the fold window
# (train_start through val_end). The FFD filter uses a fixed-width window
# of historical weights, so it only needs a warmup period at the start --
# no fitting is involved. Computing per fold ensures the warmup loss
# does not leak information across fold boundaries.

# %%
ffd_fold_results = []

for fold in all_folds:
    fold_idx = fold["fold"]
    train_start = fold["train_start"]
    val_end = fold["val_end"]

    fold_frames = []
    for symbol, d in REFERENCE_ETFS.items():
        etf = (
            prices.filter(
                (pl.col("symbol") == symbol)
                & (pl.col("timestamp") >= pl.lit(train_start).cast(pl.Date))
                & (pl.col("timestamp") <= pl.lit(val_end).cast(pl.Date))
            )
            .sort("timestamp")
            .select(["timestamp", "close"])
        )

        if len(etf) == 0:
            continue

        log_close = etf["close"].log()
        ffd_series = ffdiff(log_close, d=d)

        ffd_df = pl.DataFrame(
            {
                "timestamp": etf["timestamp"],
                f"ffd_{symbol.lower()}": ffd_series,
            }
        ).drop_nulls()

        fold_frames.append(ffd_df)

    if fold_frames:
        ffd_fold = fold_frames[0]
        for df in fold_frames[1:]:
            ffd_fold = ffd_fold.join(df, on="timestamp", how="outer_coalesce")
        ffd_fold = ffd_fold.sort("timestamp").with_columns(pl.lit(fold_idx).alias("fold"))
        ffd_fold_results.append(ffd_fold)
        ffd_cols = [c for c in ffd_fold.columns if c.startswith("ffd_")]
        print(f"  Fold {fold_idx}: FFD {len(ffd_cols)} series, {len(ffd_fold):,} dates")
    else:
        print(f"  Fold {fold_idx}: no FFD series produced (no qualifying ETFs)")

ffd_features = pl.concat(ffd_fold_results) if ffd_fold_results else pl.DataFrame()
ffd_col_names = [c for c in ffd_features.columns if c.startswith("ffd_")]
print(f"\nFFD features: {len(ffd_col_names)} series, {len(ffd_features):,} rows across folds")

# %% [markdown]
# ## Part 3: Per-ETF GARCH(1,1) (Per Fold)
#
# For each fold, fit GARCH(1,1) on each ETF's **training returns**, then
# use `model.fix(params)` to run the variance recursion on the full fold
# window (train through val_end) without re-estimating parameters. This
# is the **fit-then-filter** paradigm: parameters come from training only,
# but conditional volatility is computed for every date in the window.
#
# The `fix()` method applies frozen parameters to a new data series,
# producing a causal conditional volatility path $\sigma_t$ that depends
# only on past returns given the frozen parameters.

# %%
all_symbols = sorted(prices["symbol"].unique().to_list())
if MAX_SYMBOLS > 0:
    all_symbols = all_symbols[:MAX_SYMBOLS]

print(f"Fitting GARCH(1,1) on {len(all_symbols)} ETFs across {len(all_folds)} folds...")


# %%
def fit_garch_fold(
    prices_df: pl.DataFrame,
    symbols: list[str],
    fold: dict,
    min_obs: int,
) -> pl.DataFrame:
    """Fit GARCH(1,1) per symbol for one fold.

    Parameters
    ----------
    prices_df : pl.DataFrame
        Full prices panel with timestamp, symbol, close columns.
    symbols : list[str]
        Symbols to fit.
    fold : dict
        Fold dict with train_start, train_end, val_end keys.
    min_obs : int
        Minimum training observations for GARCH fit.

    Returns
    -------
    pl.DataFrame
        Conditional volatility for the full fold window with fold column.
    """
    fold_idx = fold["fold"]
    train_start = fold["train_start"]
    train_end = fold["train_end"]
    val_end = fold["val_end"]

    results = []
    n_success = 0
    n_fail = 0

    for sym in symbols:
        # Get full fold window data (train_start through val_end)
        sym_data = (
            prices_df.filter(
                (pl.col("symbol") == sym)
                & (pl.col("timestamp") >= pl.lit(train_start).cast(pl.Date))
                & (pl.col("timestamp") <= pl.lit(val_end).cast(pl.Date))
            )
            .sort("timestamp")
            .with_columns(ret=pl.col("close").pct_change())
            .drop_nulls(subset=["ret"])
        )

        # Training returns only
        train_data = sym_data.filter(pl.col("timestamp") < pl.lit(train_end).cast(pl.Date))

        if len(train_data) < min_obs:
            n_fail += 1
            continue

        train_returns_pct = (train_data["ret"] * 100).to_numpy()

        try:
            # Fit on training data
            train_model = arch_model(
                train_returns_pct,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="Normal",
            )
            train_result = train_model.fit(disp="off", show_warning=False)

            # Apply frozen parameters to full fold window
            full_returns_pct = (sym_data["ret"] * 100).to_numpy()
            full_model = arch_model(
                full_returns_pct,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="Normal",
            )
            filtered = full_model.fix(train_result.params)

            # Annualized conditional vol (input is in % daily)
            cond_vol_ann = filtered.conditional_volatility * np.sqrt(252) / 100

            sym_result = pl.DataFrame(
                {
                    "timestamp": sym_data["timestamp"].to_list(),
                    "symbol": [sym] * len(sym_data),
                    "garch_cond_vol": cond_vol_ann,
                    "fold": [fold_idx] * len(sym_data),
                }
            ).drop_nulls()

            if len(sym_result) > 0:
                results.append(sym_result)
                n_success += 1
        except Exception:
            n_fail += 1

    print(f"  Fold {fold_idx} GARCH: {n_success}/{len(symbols)} fitted, {n_fail} failed/skipped")
    return pl.concat(results) if results else pl.DataFrame()


# %%
garch_fold_results = []

for fold in all_folds:
    garch_fold = fit_garch_fold(prices, all_symbols, fold, GARCH_MIN_OBS)
    if len(garch_fold) > 0:
        garch_fold_results.append(garch_fold)

garch_df = pl.concat(garch_fold_results) if garch_fold_results else pl.DataFrame()
garch_cols = ["garch_cond_vol"]

if len(garch_df) > 0:
    n_syms = garch_df["symbol"].n_unique()
    print(
        f"\nGARCH features: {len(garch_df):,} rows, {n_syms} assets, {garch_df['fold'].n_unique()} folds"
    )
    print(
        f"  Conditional vol: mean={garch_df['garch_cond_vol'].mean():.3f}, "
        f"std={garch_df['garch_cond_vol'].std():.3f}"
    )

# %% [markdown]
# ## Part 4: Combine and Broadcast to Per-Fold Panel
#
# HMM regime features and FFD features are date-level (one value per day
# shared by all ETFs). GARCH features are per-asset. For each fold, we
# broadcast date-level features to all symbols and join with per-asset
# GARCH features, producing a `(fold, timestamp, symbol)` panel.

# %%
fold_panels = []

for fold in all_folds:
    fold_idx = fold["fold"]
    train_start = fold["train_start"]
    val_end = fold["val_end"]

    # Get panel skeleton for this fold's date range
    fold_skeleton = (
        prices.filter(
            (pl.col("timestamp") >= pl.lit(train_start).cast(pl.Date))
            & (pl.col("timestamp") <= pl.lit(val_end).cast(pl.Date))
        )
        .select(["timestamp", "symbol"])
        .unique()
        .sort(["timestamp", "symbol"])
        .with_columns(pl.lit(fold_idx).alias("fold"))
    )

    # Get date-level features for this fold
    fold_hmm = (
        hmm_features.filter(pl.col("fold") == fold_idx).drop("fold")
        if len(hmm_features) > 0
        else pl.DataFrame()
    )
    fold_ffd = (
        ffd_features.filter(pl.col("fold") == fold_idx).drop("fold")
        if len(ffd_features) > 0
        else pl.DataFrame()
    )

    # Combine date-level features
    if len(fold_hmm) > 0 and len(fold_ffd) > 0:
        date_level = fold_hmm.join(fold_ffd, on="timestamp", how="outer_coalesce")
    elif len(fold_hmm) > 0:
        date_level = fold_hmm
    elif len(fold_ffd) > 0:
        date_level = fold_ffd
    else:
        date_level = pl.DataFrame()

    # Broadcast date-level features to all assets
    if len(date_level) > 0:
        panel = fold_skeleton.join(date_level, on="timestamp", how="left")
    else:
        panel = fold_skeleton

    # Join per-asset GARCH features
    if len(garch_df) > 0:
        fold_garch = garch_df.filter(pl.col("fold") == fold_idx).drop("fold")
        panel = panel.join(fold_garch, on=["timestamp", "symbol"], how="left")

    fold_panels.append(panel)

temporal = pl.concat(fold_panels).sort(["fold", "timestamp", "symbol"])

temporal_cols = [c for c in temporal.columns if c not in ("timestamp", "symbol", "fold")]
print(f"Combined model-based features: {len(temporal_cols)} columns, {len(temporal):,} rows")
print(f"  Assets: {temporal['symbol'].n_unique()}, Folds: {temporal['fold'].n_unique()}")

# %% [markdown]
# ### Quality Check
#
# Verify that temporal features have reasonable coverage and distributions.

# %%
for col in temporal_cols:
    valid = temporal[col].drop_nulls().len()
    total = len(temporal)
    mean = temporal[col].drop_nulls().mean()
    std = temporal[col].drop_nulls().std()
    print(f"  {col:30s}: {valid:,}/{total:,} valid, mean={mean:.4f}, std={std:.4f}")

# %% [markdown]
# ### Per-Fold Feature Stability
#
# Check that HMM and GARCH features are stable across folds (model
# parameters should not drift drastically with overlapping training windows).

# %%
print("\nPer-fold feature means:")
fold_summary = (
    temporal.group_by("fold")
    .agg([pl.col(c).mean().alias(f"mean_{c}") for c in temporal_cols])
    .sort("fold")
)
print(fold_summary)

# %% [markdown]
# ## Save Artifacts

# %%
FEATURES_DIR = CASE_DIR / "features"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
temporal.write_parquet(FEATURES_DIR / "model_based.parquet")
print(
    f"Saved: {FEATURES_DIR / 'model_based.parquet'} "
    f"({len(temporal):,} rows, {len(temporal_cols)} features + fold column)"
)

# %% [markdown]
# ## Incremental Evaluation
#
# Evaluate feature quality using **validation-period data only** from each
# fold. We compute cross-sectional Spearman IC for per-asset features and
# time-series IC for date-level features.

# %%
labels = pl.read_parquet(CASE_DIR / "labels" / "fwd_ret_21d.parquet")
label_col = "fwd_ret_21d"

# Only evaluate on validation periods (not training periods)
val_rows = []
for fold in all_folds:
    fold_idx = fold["fold"]
    val_start = fold["val_start"]
    val_end = fold["val_end"]
    fold_val = temporal.filter(
        (pl.col("fold") == fold_idx)
        & (pl.col("timestamp") >= pl.lit(val_start).cast(pl.Date))
        & (pl.col("timestamp") <= pl.lit(val_end).cast(pl.Date))
    )
    val_rows.append(fold_val)

val_temporal = pl.concat(val_rows) if val_rows else temporal

eval_df = val_temporal.join(labels, on=["timestamp", "symbol"], how="inner").drop_nulls(
    subset=[label_col]
)
print(
    f"Evaluation panel (val periods only): {len(eval_df):,} rows, {eval_df['symbol'].n_unique()} assets"
)

# %% [markdown]
# ### Cross-Sectional IC for Per-Asset Features

# %%
temporal_ic = {}

# Per-asset features: cross-sectional IC (rank correlation per date)
per_asset_features = [c for c in temporal_cols if c in garch_cols]
date_level_features = [c for c in temporal_cols if c not in garch_cols]

for feat in per_asset_features:
    ic_series = (
        eval_df.filter(pl.col(feat).is_not_null())
        .with_columns(
            pl.col(feat).rank(method="average").over("timestamp").alias("_feat_rank"),
            pl.col(label_col).rank(method="average").over("timestamp").alias("_label_rank"),
        )
        .group_by("timestamp")
        .agg(pl.corr("_feat_rank", "_label_rank").alias("ic"))
        .sort("timestamp")
    )
    ics = ic_series["ic"].drop_nulls().to_numpy()
    if len(ics) > 50:
        temporal_ic[feat] = {
            "ic": float(np.nanmean(ics)),
            "t_stat": float(np.nanmean(ics) / (np.nanstd(ics) / np.sqrt(len(ics)))),
            "p_value": None,
            "bootstrap_std": float(np.nanstd(ics)),
        }

# %% [markdown]
# ### Time-Series IC for Date-Level Features
#
# Date-level features (HMM, FFD) are identical across symbols on each date,
# so cross-sectional IC is zero by construction. We evaluate them via
# time-series correlation with the cross-sectional average return.

# %%
avg_ret = (
    eval_df.group_by("timestamp")
    .agg(pl.col(label_col).mean().alias("avg_fwd_ret"))
    .sort("timestamp")
)

date_features = (
    val_temporal.select(["timestamp"] + date_level_features)
    .unique(subset=["timestamp"])
    .sort("timestamp")
)
eval_ts = date_features.join(avg_ret, on="timestamp", how="inner").drop_nulls()

for feat in date_level_features:
    x = eval_ts[feat].to_numpy()
    y = eval_ts["avg_fwd_ret"].to_numpy()
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 50:
        continue
    result = robust_ic(x[valid], y[valid], return_details=True)
    temporal_ic[feat] = result

# %%
temporal_eval = pl.DataFrame(
    [
        {
            "feature": feat,
            "ic": stats["ic"],
            "t_stat": stats.get("t_stat", 0.0),
            "p_value": stats.get("p_value"),
            "bootstrap_se": stats.get("bootstrap_std", stats.get("bootstrap_se", 0.0)),
        }
        for feat, stats in temporal_ic.items()
    ]
).sort("ic", descending=True)

print("\nModel-Based Feature Evaluation (validation periods only):")
print(temporal_eval)

# %% [markdown]
# **Interpretation**: GARCH conditional volatility is evaluated via
# cross-sectional IC (does higher predicted vol predict lower returns?).
# Date-level features (HMM, FFD) are evaluated via time-series IC against
# the average market return. Both types add value through different channels:
# GARCH enables cross-sectional differentiation, while regime features
# enable conditional strategies (e.g., reduce exposure in stress regimes).
#
# Because all features are now computed per fold with training-only
# parameters, the IC values reflect genuine out-of-sample signal strength
# rather than in-sample fit quality.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Per-fold fitting eliminates parameter look-ahead**: HMM and GARCH
#    parameters are estimated on each fold's training window only.
#    The `fold` column in the output lets downstream models use the
#    correct features for each CV fold.
# 2. **HMM on aggregate market**: 2-state Gaussian HMM on SPY captures
#    market-wide risk-on/risk-off regimes. All ETFs inherit the same state.
# 3. **Filtered probabilities**: Forward algorithm only -- smoothed probs
#    use future data and would introduce observation-level look-ahead bias.
# 4. **Label switching prevention**: States sorted by variance ensures
#    State 0 = calm, State 1 = stressed across all folds.
# 5. **GARCH fit-then-filter**: `model.fix(params)` applies frozen training
#    parameters to the full fold window, producing causal conditional
#    volatility without re-estimation.
# 6. **Fractional differencing**: Fixed $d$ by asset class (equities=0.4,
#    fixed income=0.5) requires no fitting, but is computed per fold
#    window for clean warmup handling.
# 7. **Per-ETF GARCH**: Conditional volatility provides asset-specific risk
#    dynamics, enabling cross-sectional differentiation (unlike date-level
#    features which are shared across all ETFs).
# 8. **Holdout fold**: A dedicated holdout fold (fit on all pre-holdout
#    data) provides features for the final out-of-sample evaluation.
#
# **Next**: Ch11 models will join financial features (Ch8) with
# model-based features (Ch9) and use the `fold` column to ensure each
# CV fold uses only training-fitted temporal features.
