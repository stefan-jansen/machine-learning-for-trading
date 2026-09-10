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
# **Double machine learning** (DML) estimates the effect of a momentum signal on forward
# returns while adjusting for volatility and regime, on a panel of ETFs. The estimate is one
# of two things this notebook is about. The other is that a panel breaks every correction
# that has a notion of "nearby": the standard error, the cross-validation folds, the
# permutation blocks and the temporal subsets all count in rows unless they are told not to,
# and on a stacked panel a row is a different ETF rather than the next day.
#
# ## Learning Objectives
# - LO1: Explain what confounding does to a factor regression, and what adjustment can and
#   cannot recover
# - LO2: Fit a DML estimator with EconML for a continuous treatment
# - LO3: Compare the raw and adjusted estimates and read the direction of the bias
# - LO4: Build folds, standard errors and permutations that count in decision times
# - LO5: Read a refutation test, including one that cannot refute anything
#
# ## Cross-References
# - **Upstream**: Chapter 8 (ETF momentum features)
# - **Downstream**: Chapter 16 (strategy simulation), Chapter 19 (risk management)
# - **Related**: [`02_dowhy_causal_graph`](02_dowhy_causal_graph.ipynb) (graphical approach), [`04_dml_crypto_regime`](04_dml_crypto_regime.ipynb) (regime effects)
#
# ## Methodological Notes
# Following Chernozhukov et al. (2017) and de Prado (2018):
# - **WalkForwardCV** from ml4t-diagnostics, with purging and embargo, built over the
#   panel's decision times
# - **Driscoll-Kraay standard errors**, which aggregate by decision time before applying the
#   Newey-West kernel
# - **Block permutation within entity**, so the placebo treatment keeps its persistence
#
# **Prerequisites**: [`02_dowhy_causal_graph`](02_dowhy_causal_graph.ipynb) for the DoWhy
# workflow, and an ETF modeling dataset built by the features pipeline
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
from sklearn.linear_model import Ridge

from case_studies.utils.causal import (
    block_permute,
    empirical_permutation_p,
    manual_dml_timeseries,
)
from utils.modeling import load_modeling_dataset
from utils.reproducibility import set_global_seeds

# scikit-learn repeats a notice, once per nuisance fit, that a frame carrying feature names
# was fitted and a bare array predicted; EconML does that internally. Convergence and
# numerical warnings stay visible.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

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
# A load failure is fatal rather than a fallback to synthetic data.
mds = load_modeling_dataset(CASE_STUDY_ID, PRIMARY_LABEL, max_symbols=MAX_SYMBOLS)

treatment_col = "skip_recent_6_1"
outcome_col = mds.label_col
confounder_cols = ["vol_21d", "vol_126d", "regime", "yield_curve_slope"]
entity_col = mds.entity_cols[0]

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
# ### The Panel Keys, and Why Everything Below Needs Them
#
# A row here is an ETF *and* a date, and the frame is sorted by date, so consecutive rows
# are usually different ETFs on the same day rather than the same ETF on consecutive days.
# Every correction in this notebook that has a notion of "nearby" - the standard error, the
# cross-validation folds, the permutation blocks, the temporal subsets - has to count in
# decision times, not in rows. Counting in rows on this frame measures a slice of one day's
# cross-section and calls it three weeks.
#
# The two arrays below carry that information, and each of those four places takes them.

# %%
decision_times = df[mds.date_col].to_numpy()
entities = df[entity_col].to_numpy()

unique_dates = np.sort(pd.unique(decision_times))
date_position = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
row_date_position = date_position.reindex(decision_times).to_numpy()

print(
    f"{len(df):,} rows over {len(unique_dates):,} decision times "
    f"and {df[entity_col].nunique()} entities "
    f"({len(df) / len(unique_dates):.0f} rows per decision time)"
)


def panel_folds(n_splits):
    """Walk-forward folds built over decision times, then expanded to panel rows.

    WalkForwardCV counts `label_horizon` and the embargo in the positions it is handed. Fed
    the panel's rows it would purge a fraction of one date; fed the ordered unique dates it
    purges the 21 trading days the forward return actually spans. The row indices come back
    by membership, so no fold boundary cuts through a cross-section.
    """
    splitter = WalkForwardCV(
        n_splits=n_splits,
        label_horizon=LABEL_HORIZON,
        embargo_pct=EMBARGO_PCT,
        expanding=True,
    )
    folds = []
    for train_dates, test_dates in splitter.split(np.arange(len(unique_dates)).reshape(-1, 1)):
        folds.append(
            (
                np.flatnonzero(np.isin(row_date_position, train_dates)),
                np.flatnonzero(np.isin(row_date_position, test_dates)),
            )
        )
    return folds


# %% [markdown]
# ## 2. The Unadjusted Slope
#
# An OLS regression of forward returns on momentum, with nothing else in it. It is the
# benchmark the adjusted estimates are read against, and it is confounded by construction:
# volatility and the yield-curve regime both move the momentum signal and move forward
# returns, so its slope carries their contribution as well as momentum's. Which way that
# pushes the slope is an empirical question the comparison below answers.

# %% [markdown]
# ### Which Robust Standard Error a Panel Takes
#
# Overlapping 21-day returns make consecutive observations of one ETF correlated, and the
# usual answer is a Newey-West standard error with the bandwidth set to the label horizon.
# On this frame that answer is applied to the wrong axis. `cov_type="HAC"` runs its kernel
# down the rows, and 21 rows here are a fifth of one day's cross-section, so the correction
# treats different ETFs on the same day as if they were successive days.
#
# **Driscoll-Kraay** is the version that fits a panel. It aggregates the regression score by
# decision time first and applies the Newey-West kernel to that time series, which makes it
# robust both to the serial correlation the overlap creates and to whatever the ETFs share
# on a given day. statsmodels reaches it through `cov_type="hac-groupsum"` with a `time`
# argument, which is the same call `case_studies/utils/causal.py` makes for every case study.

# %%
y = df[outcome_col].values
X_with_const = sm.add_constant(df[[treatment_col]])

ols_iid = OLS(y, X_with_const).fit()
naive_estimate = float(ols_iid.params.iloc[1])
se_iid = float(ols_iid.bse.iloc[1])

# Driscoll-Kraay: the kernel runs over decision times, with the bandwidth at the label
# horizon because a 21-day forward return overlaps for 20 of every 21 days.
HAC_LAGS = FORWARD_HORIZON
time_codes = pd.factorize(decision_times, sort=False)[0]
ols_dk = ols_iid.get_robustcov_results(
    cov_type="hac-groupsum",
    time=time_codes,
    maxlags=HAC_LAGS,
    use_correction="hac",
    df_correction=False,
)
se_hac = float(np.sqrt(np.asarray(ols_dk.cov_params())[1, 1]))
t_stat_hac = naive_estimate / se_hac

naive_ci = (naive_estimate - 1.96 * se_hac, naive_estimate + 1.96 * se_hac)

print("=" * 60)
print("NAIVE ESTIMATE (ignoring confounders)")
print("=" * 60)
print(f"Coefficient: {naive_estimate:.6f}")
print(f"Standard Error (IID): {se_iid:.6f}")
print(f"Standard Error (Driscoll-Kraay): {se_hac:.6f}")
print(f"SE Inflation (DK/IID): {se_hac / se_iid:.2f}x")
print(f"95% CI (Driscoll-Kraay): [{naive_ci[0]:.6f}, {naive_ci[1]:.6f}]")
print(f"t-statistic (Driscoll-Kraay): {t_stat_hac:.2f}")

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
# The folds come from `WalkForwardCV` rather than `KFold` or `TimeSeriesSplit`, because
# cross-fitting a forward-looking label needs two things neither of those provides. **Purging**
# drops training rows whose 21-day label window overlaps the test window; the **embargo**
# leaves a gap after the test window so autocorrelation does not carry back into the next
# training set (Chernozhukov et al. 2017, de Prado 2018).

# %% [markdown]
# The confounders go into EconML's `W` slot, which holds controls used for residualization.
# The `X` slot is for effect modifiers, the variables along which the treatment effect is
# allowed to vary. Putting plain confounders in `X` still produces a single ATE, so nothing
# visibly breaks, but it teaches the wrong habit for `04_dml_crypto_regime`, where the two
# slots carry different variables and the distinction decides what the model estimates.

# %%
Y = df[outcome_col].to_numpy()
T = df[treatment_col].to_numpy()
W = df[confounder_cols].to_numpy()  # controls used for residualization

# Walk-forward folds over decision times, expanded to rows (see `panel_folds` above).
cv = panel_folds(CV_FOLDS)
print(
    f"Using WalkForwardCV over decision times (label_horizon={LABEL_HORIZON} trading days, "
    f"embargo={EMBARGO_PCT:.1%})"
)
for i, (train_idx, test_idx) in enumerate(cv):
    print(f"  fold {i}: train {len(train_idx):,} rows, test {len(test_idx):,} rows")

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
    print("\n-> The adjusted effect is smaller: the controls were inflating the raw slope")
else:
    print("\n-> The adjusted effect is larger: the controls were masking part of the slope")

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
# **What it changes downstream**: a position size derived from the raw slope inherits
# whatever the confounders contributed to it. The adjusted estimate is the one to size on
# when volatility and regime are not separately hedged, and the gap between the two is how
# much the raw slope was carrying.

# %% [markdown]
# ## 4. The Same Estimate, Assembled by Hand
#
# `manual_dml_timeseries` from `case_studies/utils/causal.py` runs the three steps in the
# open: cross-fit the two nuisance models over walk-forward folds with an embargo, take the
# residuals, regress one on the other. Handed `groups`, it builds those folds over decision
# times and reports a Driscoll-Kraay standard error; without them it would do both by row.

# %%
# Use shared manual DML implementation
dml_result = manual_dml_timeseries(
    df[outcome_col].values,
    df[treatment_col].values,
    df[confounder_cols].values,
    n_folds=CV_FOLDS,
    embargo=LABEL_HORIZON,
    groups=decision_times,
    horizon=FORWARD_HORIZON,
)

manual_ate = dml_result["theta"]
manual_se_iid = dml_result["se_iid"]
manual_se_hac = dml_result["se_hac"]
manual_t_hac = dml_result["t_stat_hac"]
manual_ci = (manual_ate - 1.96 * manual_se_hac, manual_ate + 1.96 * manual_se_hac)

print("\n" + "=" * 60)
print("MANUAL DML WITH WALK-FORWARD CV + EMBARGO")
print("=" * 60)
print(f"ATE estimate: {manual_ate:.6f}")
print(f"Standard Error (IID): {manual_se_iid:.6f}")
print(f"Standard Error (HAC): {manual_se_hac:.6f}")
print(f"SE Inflation (HAC/IID): {manual_se_hac / manual_se_iid:.2f}x")
print(f"95% CI (HAC): [{manual_ci[0]:.6f}, {manual_ci[1]:.6f}]")
print(f"t-statistic (Driscoll-Kraay): {manual_t_hac:.2f}")

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

# The lead is taken within symbol; a row-level shift would mix ETFs at the boundary.
print("\n1. TEMPORAL PLACEBO TEST (lead of treatment)")
df_placebo = df.sort_values([entity_col, mds.date_col]).copy()
df_placebo["treatment_lead"] = df_placebo.groupby(entity_col)[treatment_col].shift(-FORWARD_HORIZON)
# Back to date order before the fit. The shift needed entity-major rows; the folds need
# every row of a decision time adjacent, and `manual_dml_timeseries` rejects groups that
# are not sorted and contiguous rather than silently splitting one date across two folds.
df_placebo = (
    df_placebo.dropna(subset=["treatment_lead", outcome_col])
    .sort_values(mds.date_col, kind="stable")
    .reset_index(drop=True)
)

if len(df_placebo) > 100:
    placebo_result = manual_dml_timeseries(
        df_placebo[outcome_col].values,
        df_placebo["treatment_lead"].values,
        df_placebo[confounder_cols].values,
        n_folds=CV_FOLDS,
        embargo=LABEL_HORIZON,
        groups=df_placebo[mds.date_col].to_numpy(),
        horizon=FORWARD_HORIZON,
    )
    placebo_effect = placebo_result["theta"]

    print(f"   Lead treatment effect (DML): {placebo_effect:.6f}")
    print(f"   Original DML effect:         {manual_ate:.6f}")
    print(f"   Ratio |lead / original|:     {abs(placebo_effect) / abs(manual_ate):.3f}")
else:
    placebo_effect = None
    print("   Insufficient data for placebo test")

# %% [markdown]
# **Reading the temporal placebo.** A ratio near one is what this test returns here, and it
# is not evidence of reverse causality. The treatment is a momentum signal that barely moves
# over 21 trading days, so the horizon-shifted placebo is nearly the same variable as the
# real treatment and reproduces a similar effect. For a treatment this persistent, the
# shifted-signal placebo cannot refute anything.
#
# The block permutation below is the refuter that can, because it breaks the alignment
# between treatment and outcome while keeping the treatment's own persistence intact - but
# only if the blocks are built along time.

# %% [markdown]
# ### Blocks Along Time, Not Along Rows
#
# `block_permute` given a bare array counts `BLOCK_SIZE` in the positions it is handed. On
# this frame that is about a fifth of one day's cross-section, so a "block" is a handful of
# ETFs on a single day. Permuting those destroys the serial dependence the test is supposed
# to preserve, which is the iid shuffle `block_permute`'s own docstring warns makes placebo
# tests too easy to pass.
#
# Passing `groups` and `units` makes it permute within each ETF, where `BLOCK_SIZE` counts
# that ETF's own ordered trading days and a block is the three weeks the name promises.

# %%
# Test 2: Block Permutation Test (uses shared block_permute)
print(f"\n2. BLOCK PERMUTATION TEST ({N_PLACEBO_PERMUTATIONS} permutations)")
placebo_effects = []
placebo_t_stats = []
permutation_failures = 0
T_original = df[treatment_col].values
rng = np.random.default_rng(SEED)

for i in range(N_PLACEBO_PERMUTATIONS):
    T_permuted = block_permute(
        T_original,
        BLOCK_SIZE,
        rng=rng,
        groups=decision_times,
        units=entities,
    )

    df_perm = df.copy()
    df_perm[treatment_col] = T_permuted

    try:
        perm_result = manual_dml_timeseries(
            df_perm[outcome_col].values,
            df_perm[treatment_col].values,
            df_perm[confounder_cols].values,
            n_folds=3,
            groups=decision_times,
            horizon=FORWARD_HORIZON,
        )
        if np.isfinite(perm_result["t_stat_hac"]):
            placebo_t_stats.append(perm_result["t_stat_hac"])
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

# %% [markdown]
# ### Compared on t-Statistics, Not on Effect Sizes
#
# Permuting the treatment also frees it from the controls. The second stage regresses the
# residualized outcome on the residualized treatment, so whatever the controls explain leaves
# the denominator; a permuted treatment is no longer explained by them, its residual variance
# is larger, and a placebo effect is mechanically smaller than the observed one whether or not
# there is anything to find. Comparing raw effects against that distribution reports
# significance the standard error does not support. Each permutation's t-statistic divides by
# its own standard error, so the scale cancels and only the alignment between treatment and
# outcome is left.
#
# The p-value is the fraction of the placebo distribution at least as extreme as the observed
# statistic, and it carries a plus-one correction because the observed statistic is itself one
# draw that distribution could produce. Without the correction, a run in which no placebo
# reaches it reports zero, which no finite number of permutations can establish. With n draws
# the smallest reportable value is 1 / (n + 1), printed beside it as the floor. It is not a
# false discovery rate, which is what this quantity used to be called.

# %%
if len(placebo_t_stats) > 10:
    placebo_mean = np.mean(placebo_t_stats)
    placebo_std = np.std(placebo_t_stats)
    z_score = (manual_t_hac - placebo_mean) / placebo_std if placebo_std > 0 else np.inf
    # The plus-one correction is why the floor below is 1 / (n + 1); see the markdown above.
    permutation_p = empirical_permutation_p(np.asarray(placebo_t_stats), manual_t_hac)

    print(f"   Placebo t mean: {placebo_mean:.4f}")
    print(f"   Placebo t std:  {placebo_std:.4f}")
    print(f"   Observed t (Driscoll-Kraay): {manual_t_hac:.4f}")
    print(f"   Z-score vs placebo: {z_score:.2f}")
    print(
        f"   Permutation p-value: {permutation_p:.4f} (floor {1 / (len(placebo_t_stats) + 1):.4f})"
    )

    print(
        f"   Placebo draws at least as extreme: "
        f"{int(round(permutation_p * (len(placebo_t_stats) + 1))) - 1} of {len(placebo_t_stats)}"
    )
    print(
        f"   Placebo effect spread {np.std(placebo_effects):.6f} against a Driscoll-Kraay "
        f"standard error of {manual_se_hac:.6f}; the comparison above is on t-statistics, "
        f"which holds whatever the ratio of those two turns out to be"
    )
else:
    print("   Insufficient successful permutations")
    z_score = None
    permutation_p = None

# %% [markdown]
# **The z-score and the p-value can disagree, and the count is the one that holds.** The
# z-score measures how far the observed t-statistic sits from the placebo *mean* in placebo
# standard deviations, which is a statement about a normal distribution centred where the
# placebos are. The permutation p-value counts how many placebo draws reach its magnitude.
# When the placebo distribution is not centred near zero the two answer different questions,
# and only the count is a statement about the null the test actually built. Read the count
# printed above, and the mean and standard deviation beside it, before reading the z-score.
#
# **And this null is not centred at zero.** Permuting blocks within a symbol reorders when
# that symbol's momentum was high; it leaves untouched *which* symbols had high momentum on
# average and which had high average forward returns. The between-symbol part of the
# association is therefore intact in every draw, and the null the test builds is the narrower
# one of "no within-symbol timing relation", not "no relation". A placebo mean well away from
# zero is that between-symbol component showing up, and it is the reason the count and the
# z-score part company here.

# %% [markdown]
# The two halves are cut at a decision time rather than at a row, so neither holds a
# fragment of a cross-section. That is the same reason the subsample near the top keeps
# whole dates.

# %%
print("\n3. SUBSET STABILITY TEST (temporal halves)")
midpoint_date = unique_dates[len(unique_dates) // 2]
df_first_half = df[decision_times < midpoint_date]
df_second_half = df[decision_times >= midpoint_date]

if len(df_first_half) > 100 and len(df_second_half) > 100:
    result_first = manual_dml_timeseries(
        df_first_half[outcome_col].values,
        df_first_half[treatment_col].values,
        df_first_half[confounder_cols].values,
        n_folds=3,
        groups=df_first_half[mds.date_col].to_numpy(),
        horizon=FORWARD_HORIZON,
    )
    result_second = manual_dml_timeseries(
        df_second_half[outcome_col].values,
        df_second_half[treatment_col].values,
        df_second_half[confounder_cols].values,
        n_folds=3,
        groups=df_second_half[mds.date_col].to_numpy(),
        horizon=FORWARD_HORIZON,
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

    print(f"   First half: {len(df_first_half):,} rows before {midpoint_date}")
    print(f"   Second half: {len(df_second_half):,} rows from {midpoint_date}")
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

cv_sensitivity = panel_folds(3)

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
spread = sensitivity_df["ATE Estimate"].abs()
print(f"\nSpread across nuisance specifications: {spread.max() / spread.min():.2f}x")

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
if permutation_p is not None:
    results_summary["permutation_p"] = permutation_p

for key, value in results_summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")

# %% [markdown]
# ## Key Methodological Points
#
# ### The four choices that decide whether the estimate means anything
# 1. **Cross-fitting with purging and embargo**, over decision times rather than rows.
#    Purging drops training rows whose labels overlap the test window; the embargo adds a
#    buffer after it (Chernozhukov et al. 2017, de Prado 2018).
# 2. **Driscoll-Kraay standard errors**, so the serial-correlation correction is applied
#    along time and the cross-sectional dependence within a date is absorbed rather than
#    counted as extra observations.
# 3. **Block permutation within entity**, so the placebo distribution is built from a
#    treatment that keeps its persistence.
# 4. **A nuisance-model sweep**, because the point estimate depends on the first stage and
#    the spread across specifications is the honest width of the finding.

# %% [markdown]
# ### Three Intervals on One Effect
#
# The notebook produces three statements of uncertainty about the same quantity, and they do
# not agree. EconML's `ate_interval` treats the residualized observations as independent
# draws. The manual DML path reports a Driscoll-Kraay standard error, which aggregates by
# decision time and absorbs whatever the ETFs share on a day. The permutation test compares
# the estimate against a null built by shuffling the treatment within each ETF.
#
# The spread between them is not a defect in any one of them. It is the price of the panel:
# an interval is only as good as its account of what is independent, and 52,000 ETF-days are
# not 52,000 independent observations.

# %%
se_inflation = se_hac / se_iid

print("Quantitative Findings")
print("-" * 40)
print(f"SE inflation, Driscoll-Kraay over iid: {se_inflation:.2f}x")
if dml_estimate is not None:
    direction = "larger" if abs(naive_estimate) > abs(dml_estimate) else "smaller"
    print(
        f"Adjustment moves the slope by {abs(bias_pct):.1f}% - the unadjusted estimate is "
        f"{direction} in magnitude"
    )
    if not np.isnan(dml_ci_lower):
        print(
            f"EconML DML: {dml_estimate:.6f}, iid 95% CI "
            f"[{dml_ci_lower:.6f}, {dml_ci_upper:.6f}], width {dml_ci_upper - dml_ci_lower:.6f}"
        )
    else:
        print(f"EconML DML: {dml_estimate:.6f} (interval unavailable)")
print(
    f"Manual DML:  {manual_ate:.6f}, Driscoll-Kraay 95% CI "
    f"[{manual_ci[0]:.6f}, {manual_ci[1]:.6f}], width {manual_ci[1] - manual_ci[0]:.6f}"
)
print(
    f"Naive OLS:   {naive_estimate:.6f}, Driscoll-Kraay 95% CI [{naive_ci[0]:.6f}, {naive_ci[1]:.6f}]"
)
if z_score is not None:
    print(f"Placebo z-score (on t-statistics): {z_score:.2f}")
if permutation_p is not None:
    print(f"Permutation p-value: {permutation_p:.4f}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **DML is an adjusted estimate, not a free lunch**. The orthogonalized
#    coefficient is interpretable as a causal effect only under the maintained
#    assumptions: pre-treatment controls are sufficient, positivity holds, no
#    interference, and the specified DAG is correct.
#
# 2. **The robust standard error has to match the data's axis.** Overlapping 21-day labels
#    make the iid standard error too small, and on a stacked panel the ordinary Newey-West
#    correction fixes that along the wrong axis: its lags run across the cross-section.
#    Driscoll-Kraay aggregates by decision time first, which is what the SE inflation
#    printed above measures.
#
# 3. **Manual DML matches EconML conceptually and differs numerically.** The point estimate
#    moves with nuisance-model flexibility, and the sweep in section 6 shows by how much.
#    Neyman orthogonality is a guarantee about first-order sensitivity to nuisance error,
#    not about agreement across nuisance choices.
#
# 4. **A refutation counts in the units its name claims.** Block permutation preserves
#    autocorrelation only when the blocks run along an entity's own trading days; on a
#    flattened panel the same call permutes a slice of one day's cross-section, which is
#    the iid shuffle the test exists to avoid. The same is true of the walk-forward folds
#    and of the temporal halves.
#
# 5. **The sign of the confounding bias is a result, not a rule.** Here the controls change
#    the momentum slope in one direction; on the crypto funding premium in
#    `02_dowhy_causal_graph` they change it in the other. The comparison printed above says
#    which happened on this data, and it is the printed numbers, not this sentence, that
#    settle it.
