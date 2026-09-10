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

# %% [markdown]
# # Double Machine Learning on Crypto Premium Index with Regime Conditioning
#
# **Chapter 15: Causal Estimation with ML**
# **Docker image**: `ml4t`
#
# **Section Reference**: Section 15.4 for DML, and Section 15.3 for the refutation tests
#
# ## Purpose
# Double machine learning applied to the perpetual-futures premium: does a stretched premium
# cause the next 8-hour return, and does the answer differ between calm and volatile markets?
# The panel is nineteen perpetuals sharing every timestamp, which is what makes the second
# half of the notebook about how a correction knows what "nearby" means.
#
# The estimand is the marginal effect of a one-unit change in the 14-day
# premium z-score on the next-bar return, after adjustment for observed
# pre-treatment controls:
#
# $$\theta = \frac{\partial}{\partial t}\, E[Y(t)] \bigm|_{t = \text{premium z-score}}$$
#
# Regime-specific estimates apply this same estimand within volatility strata
# (subgroup ATEs), not Conditional Average Treatment Effects (CATE), which
# would require a single heterogeneity model rather than separate
# regime-stratified fits.
#
# ## Causal Design Contract
#
# | Element                   | This notebook                                                                                          |
# |---------------------------|--------------------------------------------------------------------------------------------------------|
# | Unit                      | Crypto-symbol × 8-hour-bar from the crypto perps panel (19 symbols)                                    |
# | Treatment                 | `premium_zscore_14d` (continuous)                                                                       |
# | Outcome                   | `fwd_ret_8h` (next-bar return)                                                                          |
# | Controls (W in EconML)    | `price_vol_14d`, `funding_rate`, `premium_dev_mean_14d`, `premium_vol_72h`, `vol_ratio_short`, `premium_persistence_7d` |
# | Effect modifiers (X)      | Volatility regime (high vs low), entered via regime-stratified DML and a single-model interaction |
# | Identification assumption | Selection on observables given the six controls; controls are constructed strictly pre-treatment       |
# | Main failure modes        | Bad-control bias from premium-derived controls; mistimed treatment relative to control horizons; cross-symbol contagion that the controls don't capture |
# | Estimand                  | Marginal effect of a one-unit z-score change; ATE within each volatility regime; interaction coefficient on T × regime |
#
# **Learning Outcomes**:
# - Fit DML on a multi-symbol panel and report a standard error that counts in bars
# - Estimate a treatment effect within volatility regimes and test the difference in one model
# - Build a block permutation whose blocks are the durations their labels claim
#
# **Methodological Notes** (per Chernozhukov et al. 2017):
# - **WalkForwardCV**: cross-fitting with purging and embargo over decision times
# - **Driscoll-Kraay standard errors**: the score is aggregated by timestamp before the
#   Newey-West kernel is applied, so nineteen perpetuals in one bar are one period
# - **Block permutation within symbol**: the placebo treatment keeps its persistence
#
# **Timing Protocol**: confounders are pre-computed features known before `t`, the treatment
# is `premium_zscore_14d` measured at `t`, and the outcome is the return over the bar that
# follows.
#
# **Cross-References**:
# - Chapter 15: [`03_econml_dml`](03_econml_dml.ipynb) (ETF momentum DML)
# - Chapter 8: Crypto premium features
# - Chapter 14: Latent factor regime detection
#
# **Prerequisites**: [`03_econml_dml`](03_econml_dml.ipynb) for DML methodology;
# crypto premium data from Ch2 data pipeline

# %% [markdown]
# ## 1. Setup and Imports

# %%
"""Double Machine Learning on Crypto Premium Index with Regime Conditioning: estimate regime-conditional causal effects."""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from ml4t.diagnostic.splitters import WalkForwardCV
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor

import utils.style  # noqa: F401  # registers + activates the ml4t Plotly template
from case_studies.utils.causal import (
    block_permute,
    empirical_permutation_p,
    manual_dml_timeseries,
)
from utils.modeling import load_modeling_dataset
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

# scikit-learn repeats a notice, once per nuisance fit, that a frame carrying feature names
# was fitted and a bare array predicted; EconML does that internally. Convergence and
# numerical warnings stay visible.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

# Statsmodels for HAC standard errors
import statsmodels.api as sm
from econml.dml import LinearDML
from statsmodels.regression.linear_model import OLS

# %% [markdown]
# ### Settings, and What the Bandwidth Is Derived From
#
# The serial-correlation bandwidth is the one setting here that is not free. Consecutive
# observations of one symbol share up to 14 days of input to the premium z-score, so that
# window is what the correction has to cover, and `HAC_LAGS` is it expressed in 8-hour bars.
# The forward return spans a single bar and does not overlap, so unlike a 21-day label it
# contributes nothing to the bandwidth.

# %% tags=["parameters"]
# Configuration - readers can modify these
CASE_STUDY_ID = "crypto_perps_funding"
PRIMARY_LABEL = "fwd_ret_8h"
MAX_SYMBOLS = 0
SEED = 42
CV_FOLDS = 5
MAX_SAMPLES = 30000  # Temporal subsample if dataset too large

# Cross-validation parameters
EMBARGO_PERIODS = 3  # 3 bars = 24h embargo for 8h bars
EMBARGO_PCT = 0.01

# Refutation parameters
N_PLACEBO_PERMUTATIONS = 100
# Block-permutation sizes (in 8h bars). Sweep over 1-day / 7-day / 14-day
# blocks so the placebo preserves the autocorrelation induced by the
# 14-day premium z-score treatment rather than only short-range structure.
BLOCK_SIZES = [3, 21, 42]
BLOCK_SIZE_HEADLINE = 21  # headline test uses 7-day blocks (one treatment half-life)

# Bandwidth: the treatment's own window, in bars (see the markdown above).
BARS_PER_DAY = 3  # 8-hour bars
TREATMENT_WINDOW_DAYS = 14  # premium_zscore_14d
HAC_LAGS = TREATMENT_WINDOW_DAYS * BARS_PER_DAY

# %%
set_global_seeds(SEED)

print("DML Crypto Premium Index Configuration:")
print(f"  Case study: {CASE_STUDY_ID}")
print(f"  Label: {PRIMARY_LABEL}")
print(f"  CV Folds: {CV_FOLDS}")
print(f"  Embargo: {EMBARGO_PERIODS} bars (24h)")

# %% [markdown]
# ## 2. Load Pre-Computed Features from Modeling Pipeline
#
# We use `load_modeling_dataset()` to load pre-computed premium features (Ch8)
# and labels, joined and ready for analysis. This replaces manual OHLCV loading
# and confounder engineering. Real-data only; no synthetic fallback. If the
# modeling dataset is missing, the notebook fails loudly rather than silently
# switching to a synthetic substitute.

# %%
# Real-data only; load failure is a fatal error.
mds = load_modeling_dataset(CASE_STUDY_ID, PRIMARY_LABEL, max_symbols=MAX_SYMBOLS)

treatment_col = "premium_zscore_14d"
confounder_cols = [
    "price_vol_14d",
    "funding_rate",
    "premium_dev_mean_14d",
    "premium_vol_72h",
    "vol_ratio_short",
    "premium_persistence_7d",
]

available = set(mds.dataset.columns)
missing = [c for c in [treatment_col] + confounder_cols if c not in available]
if missing:
    raise RuntimeError(
        f"Required columns missing from modeling dataset "
        f"{CASE_STUDY_ID}/{PRIMARY_LABEL}: {missing}. "
        f"Available features: {mds.feature_names[:20]}... "
        f"Rebuild the Ch8 features pipeline for case study '{CASE_STUDY_ID}'."
    )

# %%
# Build analysis DataFrame from loaded features
outcome_col = mds.label_col
date_col = mds.date_col

analysis_cols = [date_col] + mds.entity_cols + [treatment_col, outcome_col] + confounder_cols
df = (
    mds.dataset.select([c for c in analysis_cols if c in available])
    .drop_nulls()
    .sort(date_col)
    .to_pandas()
)

if len(df) > MAX_SAMPLES:
    # Subsample to MAX_SAMPLES most recent unique timestamps, never cutting
    # through a cross-section. On a stacked multi-symbol panel the row-level
    # iloc tail would leave a fragmented final timestamp.
    rows_per_ts = df.groupby(date_col).size().median()
    n_ts = int(np.ceil(MAX_SAMPLES / max(rows_per_ts, 1)))
    keep_ts = df[date_col].drop_duplicates().iloc[-n_ts:]
    df = df[df[date_col].isin(keep_ts)].reset_index(drop=True)
    print(
        f"Taking most recent {n_ts} timestamps ({len(df):,} rows) "
        f"from {len(keep_ts):,} unique timestamps"
    )

# Market-wide regime per timestamp, threshold shifted one bar (see the markdown above).
market_vol = (
    df.groupby(date_col)["price_vol_14d"].median().sort_index().rename("market_vol").to_frame()
)
market_vol["vol_threshold"] = (
    market_vol["market_vol"].shift(1).rolling(168, min_periods=50).median()
)
market_vol["high_vol_regime"] = (market_vol["market_vol"] > market_vol["vol_threshold"]).astype(
    "Int64"
)

df = df.merge(
    market_vol[["high_vol_regime"]],
    left_on=date_col,
    right_index=True,
    how="left",
)
df = df.dropna(subset=["high_vol_regime"]).reset_index(drop=True)
df["high_vol_regime"] = df["high_vol_regime"].astype(int)

print(f"Analysis data: {df.shape[0]:,} rows")
print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
print(f"Treatment: {treatment_col}, Outcome: {outcome_col}")
print(f"Confounders: {confounder_cols}")
if "symbol" in df.columns:
    print(f"Assets: {df['symbol'].nunique()}")

# %% [markdown]
# ### The Panel Keys
#
# A row is a symbol and an 8-hour bar, and the frame is sorted by timestamp, so nineteen
# consecutive rows are nineteen perpetuals in the same bar rather than nineteen bars. Every
# correction below that has a notion of "nearby" - the standard errors, the cross-fitting
# folds and embargo, the permutation blocks - takes the two arrays below and counts in
# decision times, not in rows. Counted in rows, a `maxlags` of twelve would spend its whole
# bandwidth inside a single bar.

# %%
entity_col = mds.entity_cols[0]
decision_times = df[date_col].to_numpy()
entities = df[entity_col].to_numpy()
time_codes = pd.factorize(decision_times, sort=False)[0]

print(
    f"{len(df):,} rows over {len(np.unique(decision_times)):,} decision times "
    f"and {df[entity_col].nunique()} entities "
    f"({len(df) / len(np.unique(decision_times)):.0f} rows per bar)"
)


def driscoll_kraay(endog, exog, times, maxlags=HAC_LAGS):
    """Fit OLS and return it with Driscoll-Kraay standard errors.

    `cov_type="hac-groupsum"` aggregates the regression score by decision time before
    applying the Newey-West kernel, so the bandwidth counts bars rather than rows and
    whatever the perpetuals share within one bar is absorbed instead of being counted as
    extra periods. Same call as `case_studies/utils/causal.py` makes for the case studies.
    """
    return (
        OLS(endog, exog)
        .fit()
        .get_robustcov_results(
            cov_type="hac-groupsum",
            time=pd.factorize(times, sort=False)[0],
            maxlags=maxlags,
            use_correction="hac",
            df_correction=False,
        )
    )


# %% [markdown]
# ## 3. Define Treatment, Outcome, and Confounders

# %%
T = df[treatment_col].values
Y = df[outcome_col].values
X = df[confounder_cols].values
regime = df["high_vol_regime"].values

print("\nCausal Setup:")
print(f"  Treatment (T): {treatment_col}, range [{T.min():.6f}, {T.max():.6f}]")
print(f"  Outcome (Y): {outcome_col}, range [{Y.min():.4f}, {Y.max():.4f}]")
print(f"  Confounders (X): {len(confounder_cols)} variables")
print(f"  Regime: {regime.sum():,} high-vol, {(1 - regime).sum():,} low-vol")

# %% [markdown]
# ## 4. Naive Estimate (Biased)

# %%
print("\nNaive Estimate (OLS, ignoring confounders)...")

T_with_const = np.column_stack([np.ones(len(T)), T])

ols_naive_iid = OLS(Y, T_with_const).fit()
naive_effect = float(ols_naive_iid.params[1])
se_naive_iid = float(ols_naive_iid.bse[1])

ols_naive = driscoll_kraay(Y, T_with_const, decision_times)
se_naive_hac = float(np.sqrt(np.asarray(ols_naive.cov_params())[1, 1]))
t_stat_naive = naive_effect / se_naive_hac

print(f"  Naive effect: {naive_effect:.4f}")
print(f"  SE (IID): {se_naive_iid:.4f}")
print(f"  SE (Driscoll-Kraay, {HAC_LAGS} bars): {se_naive_hac:.4f}")
print(f"  SE inflation (DK/IID): {se_naive_hac / se_naive_iid:.2f}x")
print(f"  t-stat (Driscoll-Kraay): {t_stat_naive:.2f}")

# %% [markdown]
# ## 5. DML with Walk-Forward CV + Embargo
#
# Uses shared `manual_dml_timeseries` from `utils.causal`.

# %%
print("\nDouble Machine Learning with walk-forward CV + embargo...")

dml_result = manual_dml_timeseries(
    Y,
    T,
    X,
    n_folds=CV_FOLDS,
    embargo=EMBARGO_PERIODS,
    return_residuals=True,
    groups=decision_times,
    hac_maxlags=HAC_LAGS,
)

dml_effect = dml_result["theta"]
dml_se_iid = dml_result["se_iid"]
dml_se_hac = dml_result["se_hac"]
dml_t_stat = dml_result["t_stat_hac"]
Y_res = dml_result.get("Y_res", np.zeros_like(Y))
T_res = dml_result.get("T_res", np.zeros_like(T))

print(f"\nDML Results (walk-forward CV with {EMBARGO_PERIODS}-bar embargo):")
print(f"  DML effect: {dml_effect:.4f}")
print(f"  SE (IID): {dml_se_iid:.4f}")
print(f"  SE (Driscoll-Kraay): {dml_se_hac:.4f}")
print(f"  SE inflation: {dml_se_hac / dml_se_iid:.2f}x")
print(f"  t-stat (Driscoll-Kraay): {dml_t_stat:.2f}")

# %% [markdown]
# ## 6. Compare Naive vs DML

# %%
bias = naive_effect - dml_effect
bias_pct = bias / abs(dml_effect) * 100 if dml_effect != 0 else 0

comparison_df = pd.DataFrame(
    {
        "Naive": [naive_effect, se_naive_hac, t_stat_naive],
        "DML": [dml_effect, dml_se_hac, dml_t_stat],
    },
    index=["Effect", "SE (Driscoll-Kraay)", "t-stat"],
)
display(comparison_df)

print(f"Confounding bias (naive - DML): {bias:.4f} ({bias_pct:+.1f}%)")

# %% [markdown]
# ## 7. Regime-Conditional Effects
#
# **Estimand and terminology**. The target quantity in this notebook is the
# marginal effect of a one-unit change in the 14-day premium z-score on the
# next-bar return, after adjustment for observed pre-treatment controls:
#
# $$
# \theta = \frac{\partial}{\partial t}\, E[Y(t)] \bigm|_{t = \text{premium z-score}}
# $$
#
# Regime-specific estimates apply this same estimand within volatility
# strata; these are **subgroup ATEs**, not Conditional Average Treatment
# Effects (CATE). CATE refers to individual-level or covariate-conditional
# heterogeneity estimated by a single model (Causal Forest, X-learner). We
# repeat the estimand inside each regime rather than fitting a single
# heterogeneity model.

# %%
print("\nRegime-Conditional Treatment Effects (Regime-Stratified ATE)...")

low_vol_mask = regime == 0
high_vol_mask = regime == 1

print("\n  Low Volatility Regime:")
if low_vol_mask.sum() > 100:
    result_low = manual_dml_timeseries(
        Y[low_vol_mask],
        T[low_vol_mask],
        X[low_vol_mask],
        n_folds=3,
        embargo=EMBARGO_PERIODS,
        groups=decision_times[low_vol_mask],
        hac_maxlags=HAC_LAGS,
    )
    effect_low = result_low["theta"]
    se_low_hac = result_low["se_hac"]
    t_low = result_low["t_stat_hac"]
    print(f"    Effect: {effect_low:.4f} (t={t_low:.2f}, SE={se_low_hac:.4f})")
else:
    effect_low, se_low_hac, t_low = 0, 1, 0
    print("    Insufficient data")

print("\n  High Volatility Regime:")
if high_vol_mask.sum() > 100:
    result_high = manual_dml_timeseries(
        Y[high_vol_mask],
        T[high_vol_mask],
        X[high_vol_mask],
        n_folds=3,
        embargo=EMBARGO_PERIODS,
        groups=decision_times[high_vol_mask],
        hac_maxlags=HAC_LAGS,
    )
    effect_high = result_high["theta"]
    se_high_hac = result_high["se_hac"]
    t_high = result_high["t_stat_hac"]
    print(f"    Effect: {effect_high:.4f} (t={t_high:.2f}, SE={se_high_hac:.4f})")
else:
    effect_high, se_high_hac, t_high = 0, 1, 0
    print("    Insufficient data")

# %%
# Independence-of-subsets approximation; the interaction model below is the better answer.
effect_diff = effect_high - effect_low
se_diff_independent = np.sqrt(se_low_hac**2 + se_high_hac**2)
t_diff_independent = effect_diff / se_diff_independent if se_diff_independent > 0 else 0

print(f"\n  Regime Difference (independence approximation): {effect_diff:.4f}")
print(f"    SE: {se_diff_independent:.4f}, t={t_diff_independent:.2f}")

# %% [markdown]
# The difference above adds the two subgroup variances as if the estimates were independent
# draws. Disjoint subsets are not independent estimates: both come from the same market and
# the residuals they are built from share the nuisance models that produced them.
#
# The interaction model is the better answer. It fits one regression on the residualized
# full sample with the treatment, the regime and their product, so the coefficient on the
# product is the regime difference and its standard error comes from the joint fit. The
# regime main effect is in the design so the interaction is identified against
# regime-specific intercepts rather than absorbing a level shift between regimes.

# %%
keep = ~np.isnan(T_res) & ~np.isnan(Y_res)
T_res_int = T_res[keep]
Y_res_int = Y_res[keep]
regime_int = regime[keep]

if len(T_res_int) > 100:
    interaction_design = np.column_stack(
        [
            np.ones(len(T_res_int)),
            regime_int,
            T_res_int,
            T_res_int * regime_int,
        ]
    )
    interaction_model = driscoll_kraay(Y_res_int, interaction_design, decision_times[keep])
    interaction_effect = float(interaction_model.params[3])
    interaction_se = float(interaction_model.bse[3])
    interaction_t = float(interaction_model.tvalues[3])
    interaction_p = float(interaction_model.pvalues[3])

    print(f"\n  Regime Interaction (Driscoll-Kraay, single model): {interaction_effect:.4f}")
    print(f"    SE: {interaction_se:.4f}, t={interaction_t:.2f}, p={interaction_p:.3f}")
    t_diff = interaction_t
    se_diff = interaction_se
else:
    interaction_effect = interaction_se = interaction_t = interaction_p = float("nan")
    t_diff = t_diff_independent
    se_diff = se_diff_independent

print(f"  -> |t| on the interaction: {abs(t_diff):.2f}")

# %% [markdown]
# ## 8. EconML Comparison (if available)

# %%
print("\nEconML LinearDML Comparison...")

T_reshaped = T.reshape(-1, 1)

cv = WalkForwardCV(
    n_splits=CV_FOLDS,
    label_horizon=EMBARGO_PERIODS,
    embargo_pct=EMBARGO_PCT,
    expanding=True,
)

linear_dml = LinearDML(
    model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
    model_t=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
    cv=cv,
    random_state=42,
)
# Confounders enter via W; the regime split stays stratified (see the markdown above).
linear_dml.fit(Y, T_reshaped, W=X)

econml_effect = float(linear_dml.ate())
ci_lower, ci_upper = (float(v) for v in linear_dml.ate_interval(alpha=0.05))

print(f"  EconML effect: {econml_effect:.4f}")
print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  Manual DML effect: {dml_effect:.4f}")
print(f"  Difference: {abs(econml_effect - dml_effect):.6f}")

# %% [markdown]
# ## 9. Refutation Tests
#
# **Block permutation** breaks the alignment between treatment and outcome while leaving the
# treatment's own persistence intact. A plain random shuffle removes the persistence too,
# which makes the placebo distribution too narrow and the effect too easy to distinguish
# from it.
#
# The blocks are permuted **within symbol**, along that symbol's own ordered bars. Passed a
# bare array, `block_permute` would count `block_size` in panel rows, and nineteen
# consecutive rows here are one 8-hour bar - so a 21-row "seven-day block" would be a
# handful of perpetuals in a single bar, which is the random shuffle the test exists to
# avoid. With `groups` and `units` the sweep below spans one, seven and fourteen days of a
# symbol's own history, which is what makes the comparison across block lengths mean
# anything.
#
# The treatment is a 14-day premium z-score and several controls use multi-day windows, so
# the fourteen-day block is the one that covers the treatment's whole input window.

# %%
block_sweep_rows = []

for block_size in BLOCK_SIZES:
    print(
        f"\n  Block size {block_size} bars "
        f"({block_size * 8 // 24}d): {N_PLACEBO_PERMUTATIONS} permutations..."
    )
    placebo_effects_block = []
    rng = np.random.default_rng(SEED)

    for _ in range(N_PLACEBO_PERMUTATIONS):
        T_placebo = block_permute(
            T,
            block_size,
            rng=rng,
            groups=decision_times,
            units=entities,
            expected_step="8h",
        )
        perm_result = manual_dml_timeseries(
            Y,
            T_placebo,
            X,
            n_folds=3,
            embargo=EMBARGO_PERIODS,
            groups=decision_times,
            hac_maxlags=HAC_LAGS,
        )
        if not np.isnan(perm_result["theta"]):
            placebo_effects_block.append(perm_result["theta"])

    if len(placebo_effects_block) >= 10:
        p_mean = float(np.mean(placebo_effects_block))
        p_std = float(np.std(placebo_effects_block))
        z = (dml_effect - p_mean) / p_std if p_std > 0 else np.inf
        # Plus-one corrected, so the floor is 1 / (n + 1); it is not a false discovery rate.
        block_p = empirical_permutation_p(np.asarray(placebo_effects_block), dml_effect)
    else:
        p_mean = p_std = z = block_p = float("nan")

    block_sweep_rows.append(
        {
            "block_size": block_size,
            "n_successful": len(placebo_effects_block),
            "placebo_mean": p_mean,
            "placebo_std": p_std,
            "block_days": block_size / BARS_PER_DAY,
            "z_score": z,
            "permutation_p": block_p,
        }
    )

block_sweep_df = pd.DataFrame(block_sweep_rows).set_index("block_size")
print("\nBlock-permutation sweep:")
display(block_sweep_df)

# Pull headline figures from the 7-day-block (21-bar) row
headline = block_sweep_df.loc[BLOCK_SIZE_HEADLINE]
placebo_mean = headline["placebo_mean"]
placebo_std = headline["placebo_std"]
z_score = headline["z_score"]
permutation_p = headline["permutation_p"]

print(
    f"\nHeadline block size: {BLOCK_SIZE_HEADLINE} bars "
    f"({BLOCK_SIZE_HEADLINE / BARS_PER_DAY:.0f}d of one symbol's bars). "
    f"z={z_score:.2f}, permutation p={permutation_p:.4f} "
    f"(floor {1 / (N_PLACEBO_PERMUTATIONS + 1):.4f})"
)

# %% [markdown]
# ## 10. Visualization

# %%
# Residualized scatter: raw vs DML-adjusted relationship
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Raw: Premium Index vs Forward Return", "Residualized (DML)"),
)

plot_rng = np.random.default_rng(SEED)
sample_idx = plot_rng.choice(len(T), size=min(2000, len(T)), replace=False)
fig.add_trace(
    go.Scatter(
        x=T[sample_idx],
        y=Y[sample_idx],
        mode="markers",
        marker=dict(size=3, opacity=0.3, color=COLORS["blue"]),
        name="Raw",
    ),
    row=1,
    col=1,
)

valid_res = ~np.isnan(Y_res) & ~np.isnan(T_res) & (Y_res != 0) & (T_res != 0)
if valid_res.sum() > 100:
    sample_res_idx = plot_rng.choice(
        np.where(valid_res)[0], size=min(2000, valid_res.sum()), replace=False
    )
    fig.add_trace(
        go.Scatter(
            x=T_res[sample_res_idx],
            y=Y_res[sample_res_idx],
            mode="markers",
            marker=dict(size=3, opacity=0.3, color=COLORS["amber"]),
            name="Residualized",
        ),
        row=1,
        col=2,
    )

# %%
# Overlay regression lines showing naive vs DML slopes
x_range = np.linspace(T.min(), T.max(), 100)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=naive_effect * x_range,
        mode="lines",
        line=dict(color=COLORS["negative"], width=2),
        name="Naive slope",
    ),
    row=1,
    col=1,
)

if valid_res.sum() > 100:
    x_range_res = np.linspace(T_res[valid_res].min(), T_res[valid_res].max(), 100)
    fig.add_trace(
        go.Scatter(
            x=x_range_res,
            y=dml_effect * x_range_res,
            mode="lines",
            line=dict(color=COLORS["negative"], width=2),
            name="DML slope",
        ),
        row=1,
        col=2,
    )

fig.update_xaxes(title_text="Premium z-score (treatment)", row=1, col=1)
fig.update_yaxes(title_text="Forward 8h return", row=1, col=1)
fig.update_xaxes(title_text="Residualized premium (T_res)", row=1, col=2)
fig.update_yaxes(title_text="Residualized return (Y_res)", row=1, col=2)
fig.update_layout(
    height=440,
    title_text="Premium against forward return, raw and after residualization",
    margin=dict(t=90, b=70, l=70, r=40),
)
show_plotly_with_alt(
    fig,
    "Two scatter panels of the same sample. The left panel plots the raw premium z-score "
    "against the forward 8-hour return; the right plots the residualized treatment against "
    "the residualized outcome after the confounders are partialled out. Each carries a "
    "straight fitted line through the origin, the naive slope on the left and the DML slope "
    "on the right. Both point clouds are wide relative to the slope of either line.",
)

# %%
# Regime comparison
fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=["Low Vol", "High Vol", "Overall"],
        y=[effect_low, effect_high, dml_effect],
        error_y=dict(type="data", array=[1.96 * se_low_hac, 1.96 * se_high_hac, 1.96 * dml_se_hac]),
        marker_color=[COLORS["blue"], COLORS["amber"], COLORS["neutral"]],
    )
)
fig2.update_layout(
    title="Adjusted premium effect by volatility regime",
    xaxis_title="Volatility regime",
    yaxis_title="Causal effect (unit return per unit premium-index)",
    height=440,
    margin=dict(t=90, b=60, l=80, r=40),
)
show_plotly_with_alt(
    fig2,
    "Bar chart of the adjusted premium effect in the low-volatility regime, the "
    "high-volatility regime and the full sample, each with a 95 percent error bar. The bars "
    "are small relative to their error bars, and every error bar spans zero.",
)

# %%
# Quantitative summary for takeaways
se_inflation_naive = se_naive_hac / se_naive_iid
se_inflation_dml = dml_se_hac / dml_se_iid
regime_ratio = effect_high / effect_low if effect_low != 0 else np.nan

print(f"Driscoll-Kraay over iid SE, naive: {se_inflation_naive:.1f}x, DML: {se_inflation_dml:.1f}x")
print(f"Confounding bias (naive vs DML): {bias_pct:+.1f}%")
print(f"Regime effect ratio (high/low vol): {regime_ratio:.2f}x")

# %% [markdown]
# ## Key Takeaways
#
# 1. **The treatment has a mechanism.** A stretched premium makes the long side of the
#    perpetual expensive through funding, which is a reason to expect the premium to
#    compress and, less directly, a reason to expect something of the return. The first of
#    those is what `02_dowhy_causal_graph` estimates; this notebook takes the second.
#
# 2. **The standard error decides what the estimate can support**, more than the estimator
#    does. The inflation printed above is the ratio between the iid standard error and one
#    that treats each 8-hour bar as a single period rather than nineteen, and the interval
#    it produces is what the effect has to clear.
#
# 3. **A regime difference needs one model, not two.** Adding the two subgroup variances
#    assumes the estimates are independent draws, which disjoint subsets of one market are
#    not. The interaction on the residualized full sample gives the difference and its
#    standard error from a single fit.
#
# 4. **A block permutation is only a block permutation if the blocks run along time.** On
#    this panel a bare `block_permute` would shuffle within a bar. The sweep over one, seven
#    and fourteen days is worth reading precisely because the blocks now differ.
#
# **Next**: [`05_momentum_causal_trading`](05_momentum_causal_trading.ipynb) turns
# regime-conditional effects into position sizes.

# %% [markdown]
# ## 11. Summary

# %%
# Consolidated results table
summary_rows = [
    ("Naive OLS", naive_effect, se_naive_hac, t_stat_naive, "n/a"),
    ("DML (overall)", dml_effect, dml_se_hac, dml_t_stat, f"{bias_pct:+.1f}%"),
    ("DML (low vol)", effect_low, se_low_hac, t_low, "n/a"),
    ("DML (high vol)", effect_high, se_high_hac, t_high, "n/a"),
]

summary_df = pl.DataFrame(
    {
        "Estimator": [r[0] for r in summary_rows],
        "Effect": [r[1] for r in summary_rows],
        "SE (Driscoll-Kraay)": [r[2] for r in summary_rows],
        "t-stat": [r[3] for r in summary_rows],
        "Bias vs DML": [r[4] for r in summary_rows],
    }
)
summary_df

# %%
# Refutation and regime difference
refutation_str = (
    f"z={z_score:.2f}, permutation p={permutation_p:.4f}"
    if np.isfinite(z_score)
    else "insufficient successful permutations"
)
print(f"Regime difference: {effect_diff:.4f} (t={t_diff:.2f})")
print(f"Block permutation refutation: {refutation_str}")

# %% [markdown]
# **Reading the table.** The "Bias vs DML" column is how far the unadjusted slope sits from
# the adjusted one, as a share of the adjusted estimate; it measures what the six controls
# were carrying, not how much of the remaining estimate is causal. The regime rows and the
# interaction below them answer whether the effect differs between calm and volatile
# markets, and the interaction is the one with a standard error that accounts for both rows
# coming from the same market.
#
# The block permutation builds the null the estimate is compared against. Permuting within
# symbol along each symbol's own bars keeps the treatment as persistent as it really is,
# which widens that null; a shuffle that destroys the persistence would narrow it and make
# the estimate easier to distinguish from a placebo than the data warrants.
