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
# # Double Machine Learning for Momentum Causal Effect
#
# **Chapter 15: Causal Estimation**
# **Docker image**: `ml4t`
# **Section Reference**: See Section 15.4 for DML theory and the ETF factor application
#
# ## Purpose
# This notebook implements **Double Machine Learning (DML)** to estimate the causal effect
# of momentum signals on forward returns, controlling for complex confounders. We demonstrate
# how to move beyond correlation to establish whether momentum has genuine predictive power.
#
# ## Learning Objectives
# - LO1: Understand confounding in factor research and why correlation ≠ causation
# - LO2: Implement DML using EconML library for continuous treatment effects
# - LO3: Compare naive vs DML estimates to quantify confounding bias
# - LO4: Apply correct temporal cross-validation and HAC standard errors
# - LO5: Validate causal effects with refutation tests
#
# ## Cross-References
# - **Upstream**: Chapter 8 (ETF momentum features)
# - **Downstream**: Chapter 16 (strategy simulation), Chapter 19 (risk management)
# - **Related**: [`02_dowhy_causal_graph`](02_dowhy_causal_graph.ipynb) (graphical approach), [`04_dml_crypto_regime`](04_dml_crypto_regime.ipynb) (regime effects)
#
# ## Methodological Notes
# This notebook follows Chernozhukov et al. (2017) and de Prado (2018):
# - **WalkForwardCV**: Cross-fitting with purging and embargo (ml4t-diagnostics)
#   - Purging: Removes training samples whose labels overlap with test period
#   - Embargo: Adds buffer after test to prevent autocorrelation leakage
# - **HAC Standard Errors**: Newey-West correction for autocorrelated residuals
# - **Refutation Tests**: Block permutation to validate causal claims
#
# **Prerequisites**: [`02_dowhy_causal_graph`](02_dowhy_causal_graph.ipynb) for DoWhy concepts;
# ETF feature data from Ch8 pipeline
#
# ## Causal Design Contract
#
# | Element                   | This notebook                                                                                |
# |---------------------------|----------------------------------------------------------------------------------------------|
# | Unit                      | ETF-date row from the ETF modeling panel                                                     |
# | Treatment                 | `skip_recent_6_1` - 6/1 momentum factor (continuous)                                          |
# | Outcome                   | `fwd_ret_21d` - 21-day forward return                                                         |
# | Controls (W in EconML)    | `vol_21d`, `vol_126d`, `regime`, `yield_curve_slope` - all backward-looking, pre-treatment    |
# | Effect modifiers (X)      | None in this notebook; constant ATE target. See `04_dml_crypto_regime` for the X-slot example |
# | Identification assumption | Selection on observables given the four controls; sufficient pre-treatment information       |
# | Main failure mode         | Unobserved confounding (sentiment / macro shocks), nuisance-model misspecification, panel-time leakage if CV is not date-grouped |
# | Estimand                  | Marginal effect of a one-unit change in 6/1 momentum on 21-day forward return after adjustment |

# %% [markdown]
# ## Setup

# %%
"""Double Machine Learning for Momentum Causal Effect - estimate causal effect of momentum on forward returns."""

import warnings

import numpy as np
import pandas as pd
from ml4t.diagnostic.splitters import WalkForwardCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

from case_studies.utils.causal import (
    block_permute,
    manual_dml_timeseries,
)
from utils.modeling import load_modeling_dataset
from utils.reproducibility import set_global_seeds

warnings.filterwarnings("ignore")

# Statsmodels for HAC standard errors
import statsmodels.api as sm
from econml.dml import LinearDML
from statsmodels.regression.linear_model import OLS

# %% [markdown]
# ## Configuration

# %% tags=["parameters"]
# Configuration - readers can modify these
CASE_STUDY_ID = "etfs"
PRIMARY_LABEL = "fwd_ret_21d"
MAX_SYMBOLS = 0
CV_FOLDS = 5
MAX_SAMPLES = 50000  # Temporal subsample if dataset too large
SEED = 42

# Cross-validation parameters for WalkForwardCV
FORWARD_HORIZON = 21  # 21-day forward returns
LABEL_HORIZON = FORWARD_HORIZON  # Purge overlapping samples
EMBARGO_PCT = 0.01  # 1% embargo after test set

# Refutation test parameters
N_PLACEBO_PERMUTATIONS = 100
BLOCK_SIZE = 21  # Block size for permutation (match forward horizon)

# %%
set_global_seeds(SEED)
print(f"Seed: {SEED}")

# %% [markdown]
# ## 1. Load ETF Features from Modeling Pipeline
#
# We use `load_modeling_dataset()` to load pre-computed features (Ch8),
# temporal features (Ch9), and labels, joined and ready for analysis.
# Real-data only - no synthetic fallback. If the modeling dataset is missing,
# the notebook fails loudly with a clear error rather than silently switching
# to a synthetic substitute that would publish indistinguishable numbers.

# %%
# Real-data only - load failure is a fatal error so CI / a fresh reader
# environment without ML4T_DATA_PATH cannot silently publish synthetic numbers.
mds = load_modeling_dataset(CASE_STUDY_ID, PRIMARY_LABEL, max_symbols=MAX_SYMBOLS)

treatment_col = "skip_recent_6_1"
outcome_col = mds.label_col
confounder_cols = ["vol_21d", "vol_126d", "regime", "yield_curve_slope"]

available = set(mds.dataset.columns)
missing = [c for c in [treatment_col, outcome_col] + confounder_cols if c not in available]
if missing:
    raise RuntimeError(
        f"Required columns missing from modeling dataset "
        f"{CASE_STUDY_ID}/{PRIMARY_LABEL}: {missing}. "
        f"Available features: {mds.feature_names[:20]}... "
        f"Set ML4T_DATA_PATH and rebuild the Ch8 features pipeline for case "
        f"study '{CASE_STUDY_ID}'."
    )

# Convert to pandas for sklearn/econml, sorted by date
analysis_cols = [mds.date_col] + mds.entity_cols + [treatment_col, outcome_col] + confounder_cols
df = (
    mds.dataset.select([c for c in analysis_cols if c in available])
    .drop_nulls()
    .sort(mds.date_col)
    .to_pandas()
)
# Temporal subsample if too large: take the most recent N unique dates so the
# subsample never cuts through a cross-section. `df.iloc[-MAX_SAMPLES:]` would
# slice at row level on a stacked panel and leave a fragmented final date.
if len(df) > MAX_SAMPLES:
    rows_per_date = df.groupby(mds.date_col).size().median()
    n_dates = int(np.ceil(MAX_SAMPLES / max(rows_per_date, 1)))
    keep_dates = df[mds.date_col].drop_duplicates().iloc[-n_dates:]
    df = df[df[mds.date_col].isin(keep_dates)].reset_index(drop=True)
    print(f"Taking most recent {n_dates} dates ({len(df):,} rows) from {len(keep_dates):,} dates")

print(f"Analysis data: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Date range: {df[mds.date_col].min()} to {df[mds.date_col].max()}")
print(f"Treatment: {treatment_col}, Outcome: {outcome_col}")

# %% [markdown]
# ## 2. Naive Correlation Estimate (Biased)
#
# Simple OLS regression of returns on momentum ignores confounders.
# This estimate is **biased** because:
# - High volatility reduces both momentum and returns
# - Risk-on regime increases both momentum and returns

# %%
# Naive OLS estimate
X_naive = df[[treatment_col]].values
y = df[outcome_col].values

naive_model = LinearRegression()
naive_model.fit(X_naive, y)
naive_estimate = naive_model.coef_[0]

# Standard error via OLS formula (assumes IID - incorrect for time series!)
n = len(y)
y_pred = naive_model.predict(X_naive)
residuals = y - y_pred
rss = np.sum(residuals**2)
mse = rss / (n - 2)
se_iid = np.sqrt(mse / np.sum((X_naive - X_naive.mean()) ** 2))

# HAC (Newey-West) standard error for autocorrelation-robust inference.
# Bandwidth matches the 21-day forward outcome - shorter bandwidths
# underestimate the variance of overlapping 21-day returns.
HAC_LAGS = FORWARD_HORIZON
X_with_const = sm.add_constant(df[[treatment_col]])
ols_model = OLS(y, X_with_const).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
se_hac = np.sqrt(ols_model.cov_params().iloc[1, 1])
t_stat_hac = naive_estimate / se_hac

naive_ci = (naive_estimate - 1.96 * se_hac, naive_estimate + 1.96 * se_hac)

print("=" * 60)
print("NAIVE ESTIMATE (ignoring confounders)")
print("=" * 60)
print(f"Coefficient: {naive_estimate:.6f}")
print(f"Standard Error (IID): {se_iid:.6f}")
print(f"Standard Error (HAC): {se_hac:.6f}")
print(f"SE Inflation (HAC/IID): {se_hac / se_iid:.2f}x")
print(f"95% CI (HAC): [{naive_ci[0]:.6f}, {naive_ci[1]:.6f}]")
print(f"t-statistic (HAC): {t_stat_hac:.2f}")

# %% [markdown]
# ## 3. Double Machine Learning Estimate under Observed-Confounder Adjustment
#
# DML uses a three-step orthogonalization process:
#
# 1. **Predict outcome from confounders**: $\hat{Y} = g(X)$ → residual $\tilde{Y} = Y - \hat{Y}$
# 2. **Predict treatment from confounders**: $\hat{T} = m(X)$ → residual $\tilde{T} = T - \hat{T}$
# 3. **Regress residual outcome on residual treatment**: $\tilde{Y} \sim \theta \tilde{T}$
#
# The coefficient $\theta$ is an *orthogonalized* estimate of the treatment
# effect under the maintained assumption that the specified controls capture
# the relevant pre-treatment confounding variation. DML reduces sensitivity
# to nuisance-model errors but does not, by itself, solve unobserved
# confounding, simultaneity, interference, or bad-control bias.
#
# **Critical for Time Series**: We use `WalkForwardCV` from ml4t-diagnostics
# (not random KFold or plain TimeSeriesSplit) to prevent temporal leakage:
# - **Purging**: Removes training samples whose labels overlap with test period
# - **Embargo**: Adds buffer after test set to prevent autocorrelation leakage
# Per Chernozhukov et al. (2017) and de Prado (2018).

# %%
# Prepare data for EconML
# `W` is the EconML slot for controls used for residualization (the role our
# confounders play here). EconML's `X` slot is reserved for effect modifiers
# along which the treatment effect is allowed to vary; using `X` for plain
# confounders works for a single ATE but blurs the role for readers and
# transfers the wrong habit to the heterogeneity-modeling notebook
# (`04_dml_crypto_regime`).
Y = df[outcome_col].values.reshape(-1, 1)
T = df[treatment_col].values.reshape(-1, 1)
W = df[confounder_cols].values  # controls used for residualization

# Use canonical walk-forward CV with purging/embargo (no sklearn fallback).
cv = WalkForwardCV(
    n_splits=CV_FOLDS,
    label_horizon=LABEL_HORIZON,
    embargo_pct=EMBARGO_PCT,
    expanding=True,
)
print(f"Using WalkForwardCV (label_horizon={LABEL_HORIZON}, embargo={EMBARGO_PCT:.1%})")

dml = LinearDML(
    model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=SEED),
    model_t=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=SEED),
    cv=cv,
    random_state=SEED,
)

dml.fit(Y, T, W=W)

dml_estimate = float(dml.ate())
dml_ci_lower, dml_ci_upper = (float(v) for v in dml.ate_interval(alpha=0.05))
bias = naive_estimate - dml_estimate
bias_pct = 100 * bias / abs(dml_estimate) if dml_estimate != 0 else float("inf")

# %%
print("\n" + "=" * 60)
print("DOUBLE ML ESTIMATE (controlling for confounders)")
print("=" * 60)
print(f"Average Treatment Effect: {dml_estimate:.6f}")
if not np.isnan(dml_ci_lower):
    print(f"95% CI: [{dml_ci_lower:.6f}, {dml_ci_upper:.6f}]")
else:
    print(
        "95% CI: unavailable (EconML inference returned NaN - "
        "likely numerically degenerate first-stage residualization)"
    )

print("\n" + "=" * 60)
print("COMPARISON: Naive vs DML")
print("=" * 60)
print(f"Naive estimate: {naive_estimate:.6f}")
print(f"DML estimate:   {dml_estimate:.6f}")
print(f"Bias (Naive - DML): {bias:.6f}")
print(f"Bias percentage: {bias_pct:.1f}%")

if abs(naive_estimate) > abs(dml_estimate):
    print("\n-> Confounders INFLATED the apparent effect of momentum")
else:
    print("\n-> Confounders MASKED the true effect of momentum")

# %% [markdown]
# **Interpretation**: The naive OLS estimate captures both the causal momentum effect and
# spurious correlation induced by shared confounders (volatility, regime). DML's
# orthogonalization removes this confounding by residualizing both treatment and outcome
# against the confounders before estimating the final coefficient.
#
# The direction of bias reveals the confounding structure. If the naive effect is
# smaller in magnitude than the DML estimate, confounders *mask* the true effect -
# for example, high volatility reduces both momentum and returns simultaneously.
# If the naive effect is larger, confounders *inflate* the apparent predictive power.
#
# **Trading implication**: The DML-adjusted effect size provides a more honest
# estimate of factor efficacy for position sizing, particularly when volatility
# and regime are not explicitly hedged.

# %% [markdown]
# ## 4. Manual DML Implementation with WalkForwardCV (Educational)
#
# To understand DML, let's implement it manually using temporal cross-fitting.
# We use the shared `manual_dml_timeseries` from `utils.causal` which includes:
# - Walk-forward temporal splitting with embargo
# - HAC (Newey-West) standard errors

# %%
# Use shared manual DML implementation
dml_result = manual_dml_timeseries(
    df[outcome_col].values,
    df[treatment_col].values,
    df[confounder_cols].values,
    n_folds=CV_FOLDS,
    embargo=LABEL_HORIZON,
)

manual_ate = dml_result["theta"]
manual_se_iid = dml_result["se_iid"]
manual_se_hac = dml_result["se_hac"]
manual_ci = (manual_ate - 1.96 * manual_se_hac, manual_ate + 1.96 * manual_se_hac)

print("\n" + "=" * 60)
print("MANUAL DML WITH WALK-FORWARD CV + EMBARGO")
print("=" * 60)
print(f"ATE estimate: {manual_ate:.6f}")
print(f"Standard Error (IID): {manual_se_iid:.6f}")
print(f"Standard Error (HAC): {manual_se_hac:.6f}")
print(f"SE Inflation (HAC/IID): {manual_se_hac / manual_se_iid:.2f}x")
print(f"95% CI (HAC): [{manual_ci[0]:.6f}, {manual_ci[1]:.6f}]")

# %% [markdown]
# ## 5. Refutation Tests
#
# **Critical for causal validity**: We validate the DML estimate using refutation tests.
#
# 1. **Temporal Placebo**: Regress Y on *lead* of T (should be ~0 if no reverse causality)
# 2. **Block Permutation**: Shuffle treatment in blocks to preserve autocorrelation
# 3. **Subset Stability**: Check if effect is stable across temporal subsets

# %%
print("\n" + "=" * 60)
print("REFUTATION TESTS")
print("=" * 60)

# Test 1: Temporal Placebo - lead of treatment should have ~0 effect.
# On a stacked ETF panel the shift must be *within symbol*; a row-level
# shift mixes ETFs at the boundary and would leak treatment values from
# one ETF into another's outcome row.
print("\n1. TEMPORAL PLACEBO TEST (lead of treatment)")
entity_col = mds.entity_cols[0]
df_placebo = df.sort_values([entity_col, mds.date_col]).copy()
df_placebo["treatment_lead"] = df_placebo.groupby(entity_col)[treatment_col].shift(-FORWARD_HORIZON)
df_placebo = df_placebo.dropna(subset=["treatment_lead", outcome_col])

if len(df_placebo) > 100:
    placebo_result = manual_dml_timeseries(
        df_placebo[outcome_col].values,
        df_placebo["treatment_lead"].values,
        df_placebo[confounder_cols].values,
        n_folds=CV_FOLDS,
        embargo=LABEL_HORIZON,
    )
    placebo_effect = placebo_result["theta"]

    print(f"   Lead treatment effect (DML): {placebo_effect:.6f}")
    print(f"   Original DML effect:         {manual_ate:.6f}")

    if abs(placebo_effect) < abs(manual_ate) * 0.5:
        print("   PASS: Lead effect is smaller than actual effect")
    else:
        print(
            "   FAIL: Lead effect is comparable to the actual effect "
            "(expected: the treatment is highly autocorrelated)"
        )
else:
    placebo_effect = None
    print("   Insufficient data for placebo test")

# %% [markdown]
# **Reading the temporal placebo.** The lead test fails here, but not because
# of reverse causality. The treatment is a momentum signal that is highly
# autocorrelated over the label horizon, so the horizon-shifted "placebo"
# treatment is nearly the same variable as the real treatment and reproduces a
# similar effect. For an autocorrelated treatment the shifted-signal placebo is
# a weak refutation by construction. The block-permutation test below is the
# reliable refuter here: it shuffles treatment in blocks that preserve the
# autocorrelation structure, so a low placebo effect cannot be manufactured by
# persistence alone.

# %%
# Test 2: Block Permutation Test (uses shared block_permute)
print(f"\n2. BLOCK PERMUTATION TEST ({N_PLACEBO_PERMUTATIONS} permutations)")
placebo_effects = []
permutation_failures = 0
T_original = df[treatment_col].values
rng = np.random.default_rng(SEED)

for i in range(N_PLACEBO_PERMUTATIONS):
    T_permuted = block_permute(T_original, BLOCK_SIZE, rng=rng)

    df_perm = df.copy()
    df_perm[treatment_col] = T_permuted

    try:
        perm_result = manual_dml_timeseries(
            df_perm[outcome_col].values,
            df_perm[treatment_col].values,
            df_perm[confounder_cols].values,
            n_folds=3,
        )
        if not np.isnan(perm_result["theta"]):
            placebo_effects.append(perm_result["theta"])
        else:
            permutation_failures += 1
    except Exception as exc:
        permutation_failures += 1
        print(f"   Permutation {i} failed: {type(exc).__name__}: {exc}")

PERMUTATION_MIN_SUCCESS = max(10, int(0.5 * N_PLACEBO_PERMUTATIONS))
print(f"   Permutations: {len(placebo_effects)} successful, {permutation_failures} failed")
if len(placebo_effects) < PERMUTATION_MIN_SUCCESS:
    raise RuntimeError(
        f"Block-permutation placebo: only {len(placebo_effects)} successful "
        f"runs (need ≥{PERMUTATION_MIN_SUCCESS}); refuter cannot be trusted."
    )

# %%
if len(placebo_effects) > 10:
    placebo_mean = np.mean(placebo_effects)
    placebo_std = np.std(placebo_effects)
    z_score = (manual_ate - placebo_mean) / placebo_std if placebo_std > 0 else np.inf
    fdr = np.mean(np.abs(placebo_effects) >= np.abs(manual_ate))

    print(f"   Placebo mean: {placebo_mean:.6f}")
    print(f"   Placebo std:  {placebo_std:.6f}")
    print(f"   Original effect: {manual_ate:.6f}")
    print(f"   Z-score vs placebo: {z_score:.2f}")
    print(f"   False discovery rate: {fdr:.1%}")

    if abs(z_score) > 2:
        print("   PASS: Effect distinguishable from placebo (z > 2)")
    else:
        print("   FAIL: Effect not distinguishable from placebo")
else:
    print("   Insufficient successful permutations")
    z_score = None
    fdr = None

# %%
# Test 3: Subset Stability
print("\n3. SUBSET STABILITY TEST (temporal halves)")
n_obs = len(df)
df_first_half = df.iloc[: n_obs // 2]
df_second_half = df.iloc[n_obs // 2 :]

if len(df_first_half) > 100 and len(df_second_half) > 100:
    result_first = manual_dml_timeseries(
        df_first_half[outcome_col].values,
        df_first_half[treatment_col].values,
        df_first_half[confounder_cols].values,
        n_folds=3,
    )
    result_second = manual_dml_timeseries(
        df_second_half[outcome_col].values,
        df_second_half[treatment_col].values,
        df_second_half[confounder_cols].values,
        n_folds=3,
    )

    effect_first = result_first["theta"]
    se_first = result_first["se_hac"]
    effect_second = result_second["theta"]
    se_second = result_second["se_hac"]

    print(f"   First half effect:  {effect_first:.6f} (SE: {se_first:.6f})")
    print(f"   Second half effect: {effect_second:.6f} (SE: {se_second:.6f})")

    diff = abs(effect_first - effect_second)
    diff_se = np.sqrt(se_first**2 + se_second**2)
    diff_z = diff / diff_se if diff_se > 0 else 0

    print(f"   Difference: {diff:.6f} (z = {diff_z:.2f})")

    if diff_z < 2:
        print("   PASS: Effect stable across temporal halves")
    else:
        print("   CAUTION: Effect differs across halves (possible regime change)")
else:
    print("   Insufficient data for subset test")

# %% [markdown]
# ## 6. Nuisance Model Sensitivity
#
# DML results depend on the quality of nuisance models. Let's compare different choices.

# %%
results = []

nuisance_models = [
    ("Linear", Ridge(alpha=1.0), Ridge(alpha=1.0)),
    (
        "GBM (shallow)",
        GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=SEED),
        GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=SEED),
    ),
    (
        "GBM (deep)",
        GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=SEED),
        GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=SEED),
    ),
]

cv_sensitivity = WalkForwardCV(
    n_splits=3, label_horizon=LABEL_HORIZON, embargo_pct=EMBARGO_PCT, expanding=True
)

for name, model_y, model_t in nuisance_models:
    dml_test = LinearDML(model_y=model_y, model_t=model_t, cv=cv_sensitivity, random_state=SEED)
    dml_test.fit(Y, T, W=W)
    ate = float(dml_test.ate())
    results.append({"Nuisance Model": name, "ATE Estimate": ate})

sensitivity_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("NUISANCE MODEL SENSITIVITY")
print("=" * 60)
print(sensitivity_df.to_string(index=False))
print("\n-> Results vary with model choice - interpret with caution!")

# %% [markdown]
# ## 7. Results Summary

# %%
print("\n" + "=" * 60)
print("CHAPTER 15 NOTEBOOK RESULTS: 03_econml_dml.py")
print("=" * 60)

results_summary = {
    "naive_estimate": naive_estimate,
    "naive_se_iid": se_iid,
    "naive_se_hac": se_hac,
    "naive_ci_lower": naive_ci[0],
    "naive_ci_upper": naive_ci[1],
    "manual_dml_estimate": manual_ate,
    "manual_dml_se_iid": manual_se_iid,
    "manual_dml_se_hac": manual_se_hac,
}

if dml_estimate is not None:
    results_summary["econml_dml_estimate"] = dml_estimate
    results_summary["econml_dml_ci_lower"] = dml_ci_lower
    results_summary["econml_dml_ci_upper"] = dml_ci_upper
    results_summary["bias_from_confounding"] = bias
    results_summary["bias_pct"] = bias_pct

if z_score is not None:
    results_summary["placebo_z_score"] = z_score
if fdr is not None:
    results_summary["false_discovery_rate"] = fdr

for key, value in results_summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")

# %% [markdown]
# ## Key Methodological Points
#
# ### What This Notebook Does Correctly (per Chernozhukov et al. 2017, de Prado 2018)
# 1. **WalkForwardCV**: Cross-fitting with purging and embargo prevents leakage
#    - Purging: Removes training samples whose labels overlap with test period
#    - Embargo: Adds buffer after test to prevent autocorrelation leakage
# 2. **HAC Standard Errors**: Newey-West correction for autocorrelated residuals
# 3. **Block Permutation**: Refutation preserves autocorrelation structure
# 4. **Sensitivity Analysis**: Multiple nuisance model specifications tested

# %%
# Quantitative findings (computed, not hardcoded)
se_inflation = se_hac / se_iid

print("Quantitative Findings")
print("-" * 40)
print(
    f"SE Inflation (HAC/IID): {se_inflation:.1%} - HAC standard errors are {se_inflation:.2f}x larger than IID"
)
if dml_estimate is not None:
    direction = "overstates" if abs(naive_estimate) > abs(dml_estimate) else "understates"
    print(f"Confounding Bias: {abs(bias_pct):.1f}% - naive estimate {direction} the DML effect")
    if not np.isnan(dml_ci_lower):
        print(
            f"DML Effect Size: {dml_estimate:.6f} (95% CI: [{dml_ci_lower:.6f}, {dml_ci_upper:.6f}])"
        )
        ci_width = dml_ci_upper - dml_ci_lower
        ci_includes_zero = dml_ci_lower <= 0 <= dml_ci_upper
        print(
            f"CI Width: {ci_width:.6f} - {'includes zero (not significant at 5%)' if ci_includes_zero else 'excludes zero (significant at 5%)'}"
        )
    else:
        print(
            f"DML Effect Size: {dml_estimate:.6f} (CI unavailable - EconML inference failed with custom CV)"
        )
print(f"Manual DML Effect: {manual_ate:.6f} (HAC SE: {manual_se_hac:.6f})")
if z_score is not None:
    print(
        f"Placebo Z-Score: {z_score:.2f} - {'distinguishable from noise' if abs(z_score) > 2 else 'not distinguishable from noise'}"
    )
if fdr is not None:
    print(f"False Discovery Rate: {fdr:.1%}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **DML is an adjusted estimate, not a free lunch**. The orthogonalized
#    coefficient is interpretable as a causal effect only under the maintained
#    assumptions: pre-treatment controls are sufficient, positivity holds, no
#    interference, and the specified DAG is correct.
#
# 2. **HAC inference is the binding inference**. IID standard errors understate
#    uncertainty for overlapping 21-day labels; HAC widens the CI by a factor
#    of roughly 1.5-2.2× on this dataset.
#
# 3. **Manual DML matches EconML conceptually but differs numerically**.
#    The point estimate is sensitive to nuisance-model flexibility - the DML
#    guarantee is on Neyman orthogonality, not on numerical stability across
#    nuisance choices.
#
# 4. **Block permutation is the appropriate placebo for autocorrelated
#    series**. Random permutation would destroy the temporal structure that
#    makes the inference task hard in the first place.
#
# 5. **Naive factor research understates this effect, it does not overstate it**.
#    Here the naive momentum slope (-0.038) is smaller in magnitude than the
#    orthogonalized DML estimate (-0.054): volatility and the yield-curve regime
#    *mask* part of the true effect rather than inflating a spurious one. Once
#    those confounders are controlled for, the effect becomes roughly 28% more
#    negative, so naive factor research misses part of the effect rather than
#    overstating it. The sign of the confounding bias is an empirical result,
#    not a rule, other treatments (for example the crypto funding premium) show
#    the reverse pattern, where confounders inflate the apparent effect.
