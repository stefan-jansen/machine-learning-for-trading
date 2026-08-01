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
# # Causal Inference on Crypto Premiums: A Tale of Two Outcomes
#
# **Chapter 15: Causal Estimation with ML**
# **Docker image**: `ml4t`
# **Section Reference**: See Section 15.2 for outcome choice and mechanism credibility
#
# ## Purpose
# This notebook demonstrates **why outcome choice matters for causal credibility**.
# We analyze the same treatment (an extreme positive crypto-premium state) with
# two different outcomes, showing that the robustness of causal claims depends
# critically on the tightness of the mechanism connecting treatment to outcome.
# The treatment is not the funding payment itself; it is the premium state
# (perp price above spot by more than two standard deviations), which creates
# funding pressure and arbitrage incentives.
#
# ## Learning Objectives
# After completing this notebook, you will be able to:
# - LO1: Apply DoWhy for DAG specification and sensitivity analysis
# - LO2: Understand why mechanism tightness determines causal credibility
# - LO3: Choose outcomes that support credible causal claims
# - LO4: Interpret sensitivity analysis to assess claim fragility
#
# ## Cross-References
# - **Upstream**: Crypto Premium Index data (Chapter 2), causal foundations (Chapter 15)
# - **Downstream**: Strategy validation (Chapter 16), risk assessment (Chapter 19)
# - **Related**: [`04_dml_crypto_regime`](04_dml_crypto_regime.ipynb) (DML on same data)
#
# ## Data Requirements
# - Crypto premium index at 8h frequency - loaded via `load_crypto_premium(frequency="8h")`
# - Crypto perpetual futures OHLCV at 1h frequency - loaded via `load_crypto_perps(frequency="1h")`
# Both are produced by the Chapter 2 data pipeline (``data/crypto/download.py``).
#
# ## Causal Design
#
# | Outcome              | Mechanism                              | Main validation tests                                                              |
# |----------------------|----------------------------------------|------------------------------------------------------------------------------------|
# | Forward 24h returns  | Indirect sentiment and risk channel    | OOS stability, placebo-date shift, return-scale and reversion-scale negative controls |
# | Forward 24h premium  | Direct arbitrage-pressure channel      | OOS stability, placebo-date shift, return-scale and reversion-scale negative controls |
#
# The computed comparison table appears at the end of the notebook. The
# difference between a credible and questionable causal claim often is not
# the method - it is the outcome you choose to study.
#
# ## Causal Design Contract
#
# | Element                   | Definition                                                                            |
# |---------------------------|---------------------------------------------------------------------------------------|
# | Unit                      | One BTC 8-hour bar                                                                    |
# | Treatment                 | `extreme_high_premium` = `premium_zscore > 2` (binary indicator of premium state)     |
# | Outcomes                  | `fwd_return_24h` (3 bars) and `fwd_premium_change` (3 bars)                           |
# | Controls                  | `return_24h`, `volatility_24h` (both backward-looking, strictly pre-treatment)        |
# | Effect modifiers          | None in this notebook                                                                 |
# | Identification assumption | Selection on observables given the specified DAG; no contemporaneous unobserved cause |
# | Main failure modes        | Unobserved confounding (sentiment shocks), bad controls, mistimed treatment           |
# | Estimand                  | ATE of entering the extreme-high-premium state - not the marginal effect of a one-unit change in premium z-score |
#
# **Prerequisites**: [`01_library_overview`](01_library_overview.ipynb) for library context;
# crypto premium index data from Ch8 features pipeline

# %% [markdown]
# ## The Crypto Funding Rate Mechanism
#
# In perpetual futures markets:
# - **Premium** = (perp price - spot price) / spot price
# - **Funding rate** = periodic payment to close the gap
# - **High premium** $\rightarrow$ longs pay shorts $\rightarrow$ pressure to close longs
#
# **Two Causal Questions**:
# 1. Does extreme *high* premium $\rightarrow$ future **returns**? (What traders want to know)
# 2. Does extreme *high* premium $\rightarrow$ premium **reversion**? (What arbitrageurs exploit)
#
# These seem similar but have very different causal structures.

# %% [markdown]
# ## Timing Protocol (CRITICAL for Causal Validity)
#
# ```
# Time:        t-168h         t-24h           t            t+24h
#              |               |              |              |
#              +-- premium_ma -+              |              |
#              +-- premium_std +              |              |
#                              +- return_24h -+              |
#                              +- volatility_24h -+          |
#                                              |              |
#                                    premium_zscore (treatment)
#                                    extreme_high_premium (treatment: z > 2)
#                                              |              |
#                                              +- fwd_return_24h --+
#                                              +- fwd_premium_change -+
#
# Key: All confounders computed BEFORE treatment decision time.
# Outcome is strictly FORWARD-looking from treatment time.
# ```

# %%
"""Causal Inference on Crypto Premiums - demonstrate why outcome choice matters for causal credibility."""

import warnings
from datetime import datetime

import dowhy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from dowhy import CausalModel
from IPython.display import display
from scipy import stats

from data import load_crypto_perps, load_crypto_premium
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

warnings.filterwarnings("ignore")

# networkx 3.x compatibility (d_separated moved under d_separation)
if not hasattr(nx.algorithms, "d_separated") and hasattr(
    nx.algorithms.d_separation, "is_d_separator"
):
    nx.algorithms.d_separated = nx.algorithms.d_separation.is_d_separator


# %% tags=["parameters"]
MAX_SYMBOLS = 0
START_DATE = "2019-01-01"
SEED = 42

# %%
set_global_seeds(SEED)

print(f"DoWhy version: {dowhy.__version__}")

# %% [markdown]
# ## Configuration

# %%
# Configuration - readers can modify these
N_SAMPLES = 20000
EXTREME_THRESHOLD = 2.0  # Z-score threshold
REFUTATION_SIMULATIONS = 50

# Train/test split date (temporal, not random)
TRAIN_END_DATE = "2023-06-30"

rng = np.random.default_rng(SEED)

# %% [markdown]
# ## 1. Load and Prepare Data

# %%
# Load premium index (8h frequency) and OHLCV (1h, resampled to 8h) via the canonical loaders
premium = load_crypto_premium(frequency="8h")
ohlcv_1h = load_crypto_perps(frequency="1h")

print(f"Premium data (8h): {premium.shape}")
print(f"OHLCV data (1h): {ohlcv_1h.shape}")

# %%
# Focus on BTC for clean single-asset analysis
btc_prem = premium.filter(pl.col("symbol") == "BTCUSDT").sort("timestamp")

# Resample 1h OHLCV to 8h to match premium frequency
btc_ohlcv_1h = ohlcv_1h.filter(pl.col("symbol") == "BTCUSDT").sort("timestamp")
btc_ohlcv = (
    btc_ohlcv_1h.group_by_dynamic("timestamp", every="8h")
    .agg(
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    )
    .sort("timestamp")
)

# Merge premium and OHLCV on 8h timestamps
btc = btc_prem.join(
    btc_ohlcv.select(["timestamp", "close", "volume"]),
    on="timestamp",
    how="inner",
)

print(f"BTC data (8h): {btc.shape}")
print(f"Date range: {btc['timestamp'].min()} to {btc['timestamp'].max()}")

# %%
# Feature engineering with STRICT timing discipline
# All confounders must be computed from data BEFORE treatment time

# Step 1: Base features (8h frequency -- 3 bars = 24h, 21 bars = 7 days)
btc = btc.with_columns(
    pl.col("premium_index_close").alias("premium"),
    pl.col("close").pct_change().alias("return_8h"),
)

# Step 2: Rolling features (backward-looking only)
# At 8h frequency: 21 bars = 7 days, 3 bars = 24h
btc = btc.with_columns(
    # Rolling stats: 7-day window (21 bars at 8h), shifted by 1 bar
    pl.col("premium").shift(1).rolling_mean(window_size=21).alias("premium_ma"),
    pl.col("premium").shift(1).rolling_std(window_size=21).alias("premium_std"),
    # 24h return and volatility: 3 bars at 8h, shifted by 1 bar
    pl.col("return_8h").shift(1).rolling_sum(window_size=3).alias("return_24h"),
    pl.col("return_8h").shift(1).rolling_std(window_size=3).alias("volatility_24h"),
)

# Step 3: Treatment (measured at time t) and outcomes (forward-looking)
btc = btc.with_columns(
    # Premium z-score at time t (treatment)
    ((pl.col("premium") - pl.col("premium_ma")) / pl.col("premium_std")).alias("premium_zscore"),
    # Forward premium change: t to t+24h = 3 bars (OUTCOME 2: reversion)
    (pl.col("premium").shift(-3) - pl.col("premium")).alias("fwd_premium_change"),
    # Forward return: t to t+24h = 3 bars (OUTCOME 1: returns)
    pl.col("close").pct_change(n=3).shift(-3).alias("fwd_return_24h"),
    # Negative-control outcomes: 24h windows ending 48h BEFORE treatment, so the
    # treatment at time t cannot causally affect them. Two scales - one for the
    # returns outcome, one for the reversion outcome - so each headline ATE has
    # a same-scale negative control.
    pl.col("close").pct_change(n=3).shift(6).alias("past_return_48h"),
    (pl.col("premium").shift(6) - pl.col("premium").shift(9)).alias("past_premium_change_48h"),
)

# Step 4: Binary treatment
btc = btc.with_columns(
    (pl.col("premium_zscore") > EXTREME_THRESHOLD).cast(pl.Int32).alias("extreme_high_premium"),
)

# Drop nulls
btc = btc.drop_nulls()

print(f"After feature engineering: {btc.shape}")

# %% [markdown]
# ## 2. Train/Test Split
#
# We use a temporal split consistent with the pre-treatment timing discipline
# discussed in Section 15.2.
#
# Because the outcome is 24-hour forward (3 bars at 8h frequency), training
# rows within the last `HORIZON_BARS` of the cutoff would realize outcomes
# *inside* the test window. We purge that boundary band so training outcomes
# are fully observed before the test period begins.

# %%
# Cast the split boundary to match the data's timestamp dtype to avoid
# resolution/timezone mismatches (e.g. Datetime('ms','UTC') vs Datetime('μs',None)).
HORIZON_BARS = 3  # 24h forward outcome at 8h frequency
PURGE_HOURS = 8 * HORIZON_BARS

train_end = datetime.fromisoformat(TRAIN_END_DATE)
train_end_lit = pl.lit(train_end).cast(btc["timestamp"].dtype)
purge_boundary_lit = pl.lit(train_end - pd.Timedelta(hours=PURGE_HOURS)).cast(
    btc["timestamp"].dtype
)

# Temporal split with horizon purge: training rows must have outcomes that
# realize before the test period starts.
train_data = btc.filter(pl.col("timestamp") <= purge_boundary_lit)
test_data = btc.filter(pl.col("timestamp") > train_end_lit)

# Subsample to N_SAMPLES using MOST RECENT observations (not random)
if len(test_data) > N_SAMPLES:
    test_data = test_data.tail(N_SAMPLES)

train_sample_size = min(len(train_data), N_SAMPLES)
train_data = train_data.tail(train_sample_size)

# Convert to pandas for DoWhy
df_train = train_data.to_pandas()
df_test = test_data.to_pandas()

print(
    f"Train: {len(df_train):,} obs (up to {TRAIN_END_DATE}), "
    f"extreme rate: {df_train['extreme_high_premium'].mean():.1%}"
)
print(
    f"Test:  {len(df_test):,} obs (after {TRAIN_END_DATE}), "
    f"extreme rate: {df_test['extreme_high_premium'].mean():.1%}"
)

# %% [markdown]
# ## 3. Descriptive Statistics: Mean Reversion Is Real
#
# Before causal analysis, we verify the basic phenomenon using an AR(1)
# regression of premium z-score on forward premium change in the training set.

# %%
slope, intercept, r, p, se = stats.linregress(
    df_train["premium_zscore"], df_train["fwd_premium_change"]
)

print("AR(1): premium_zscore -> fwd_premium_change (train)")
print(f"  Slope: {slope:.6f}, t={slope / se:.1f}, p={p:.2e}, R2={r**2:.3f}")

# Reversion rates
extreme_high = df_train["premium_zscore"] > EXTREME_THRESHOLD
extreme_low = df_train["premium_zscore"] < -EXTREME_THRESHOLD

if extreme_high.sum() > 0:
    high_reverts = (df_train.loc[extreme_high, "fwd_premium_change"] < 0).mean()
    print(f"  Extreme HIGH -> drops in 24h: {high_reverts:.1%}")

if extreme_low.sum() > 0:
    low_reverts = (df_train.loc[extreme_low, "fwd_premium_change"] > 0).mean()
    print(f"  Extreme LOW  -> rises in 24h: {low_reverts:.1%}")

# %% [markdown]
# Mean reversion is real and strong. The question is whether this association
# reflects a *causal* effect of extreme premium on future outcomes -- and
# whether the answer depends on which outcome we choose.

# %% [markdown]
# ## 4. Causal Graph Specification
#
# We specify a DAG encoding our domain knowledge. The treatment
# `extreme_high_premium` is a directional indicator: premium z-score
# above the threshold. Using a directional treatment avoids the
# cancellation that arises from combining extreme high and low premiums.
#
# ```
#     return_24h ---------> extreme_high_premium
#          |                        |
#          v                        v
#     volatility_24h -----> OUTCOME
# ```
#
# Both confounders (recent returns and volatility) affect the treatment
# and the outcome, creating backdoor paths that the adjustment set must
# block. The sensitivity analysis below quantifies how much *additional*
# unobserved confounding would change our conclusions.

# %%
# DAG for returns outcome (more confounded)
graph_returns = """
digraph {
    return_24h -> extreme_high_premium;
    return_24h -> fwd_return_24h;

    volatility_24h -> extreme_high_premium;
    volatility_24h -> fwd_return_24h;

    extreme_high_premium -> fwd_return_24h;
}
"""

# DAG for reversion outcome (tighter mechanism)
graph_reversion = """
digraph {
    return_24h -> extreme_high_premium;
    return_24h -> fwd_premium_change;

    volatility_24h -> extreme_high_premium;
    volatility_24h -> fwd_premium_change;

    extreme_high_premium -> fwd_premium_change;
}
"""

# The two negative-control outcomes (`past_return_48h`, `past_premium_change_48h`)
# are both pre-treatment and strictly unreachable from `extreme_high_premium`
# by temporal ordering. A DAG for them would (correctly) omit any
# `extreme_high_premium -> past_*` edge, so DoWhy's backdoor identification
# returns zero by construction and gives no numeric diagnostic of residual
# association. Section 9 therefore measures that residual association directly
# with an OLS+HAC regression on the same adjustment set - a placebo
# diagnostic, not an identifiable causal effect.

# %% [markdown]
# ### Identifying the Adjustment Set
#
# DoWhy's `identify_effect` applies the backdoor criterion to our DAG and
# determines which variables must be conditioned on to block confounding paths.
# The estimand below shows the identified adjustment set.

# %%
# Use the returns graph to demonstrate identification
model_demo = CausalModel(
    data=df_train,
    treatment="extreme_high_premium",
    outcome="fwd_return_24h",
    graph=graph_returns,
)
estimand_demo = model_demo.identify_effect(proceed_when_unidentifiable=True)
print(estimand_demo)

# %% [markdown]
# The backdoor adjustment set consists of `{return_24h, volatility_24h}` --
# these are the observed confounders that lie on non-causal paths between
# treatment and outcome. By conditioning on them, we block spurious
# associations while leaving the causal path
# `extreme_high_premium` $\rightarrow$ `outcome` open.
# Both graphs share the same adjustment set because the confounding structure
# is identical; only the outcome variable differs.

# %% [markdown]
# ## 5. Causal Analysis Helper Functions
#
# We split the analysis pipeline into three reusable stages:
# estimation, refutation, and sensitivity analysis.

# %% [markdown]
# ### Fit and Estimate
#
# Build a `CausalModel`, identify the estimand via the backdoor criterion,
# and estimate the average treatment effect (ATE) using linear regression.


# %%
def fit_and_estimate(df, outcome_col, graph):
    """Identify and estimate causal effect using DoWhy backdoor criterion.

    Returns (estimate, estimand, model) tuple.
    """
    model = CausalModel(
        data=df,
        treatment="extreme_high_premium",
        outcome=outcome_col,
        graph=graph,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
    )
    return estimate, estimand, model


# %% [markdown]
# ### HAC Standard Errors
#
# DoWhy's `linear_regression` estimator reports iid standard errors. For
# 8-hour crypto series the residuals are autocorrelated by construction -
# rolling-window confounders and overlapping forward outcomes both carry
# persistence. We complement each DoWhy estimate with an OLS regression
# that uses the same adjustment set and Newey-West HAC standard errors,
# keeping the pedagogy of DoWhy while giving readers inference that matches
# the data structure.


# %%
def estimate_backdoor_ols_hac(
    df,
    outcome_col,
    treatment_col="extreme_high_premium",
    controls=("return_24h", "volatility_24h"),
    maxlags=3,
):
    """Adjusted treatment effect with HAC (Newey-West) standard errors.

    `maxlags=3` matches the 24-hour outcome horizon at 8h frequency. The
    point estimate matches `backdoor.linear_regression` up to numerical
    precision; the standard error is the HAC-corrected version.
    """
    cols = [treatment_col, *controls]
    X = sm.add_constant(df[cols])
    y = df[outcome_col]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "effect": float(model.params[treatment_col]),
        "se_hac": float(model.bse[treatment_col]),
        "t_hac": float(model.tvalues[treatment_col]),
        "p_hac": float(model.pvalues[treatment_col]),
    }


# %% [markdown]
# ### Refutation Tests
#
# Run placebo treatment and random common cause refutations. A credible
# causal claim should survive both: the placebo effect should be near zero,
# and adding a random confounder should not materially change the estimate.


# %%
def run_refutations(model, estimand, estimate, n_sims=20):
    """Run placebo and random-cause refutation tests.

    Returns dict mapping test name to (passed: bool, detail: str).
    """
    results = {}

    # Placebo treatment
    try:
        refute_placebo = model.refute_estimate(
            estimand,
            estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=n_sims,
        )
        passed = abs(refute_placebo.new_effect) < abs(estimate.value) * 0.5
        results["Placebo"] = (passed, f"effect={refute_placebo.new_effect:.6f}")
    except Exception as e:
        results["Placebo"] = (False, str(e))

    # Random common cause
    try:
        refute_random = model.refute_estimate(
            estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=n_sims,
        )
        change = abs(refute_random.new_effect - estimate.value) / max(abs(estimate.value), 1e-8)
        passed = change < 0.3
        results["Random Cause"] = (passed, f"change={change:.1%}")
    except Exception as e:
        results["Random Cause"] = (False, str(e))

    return results


# %% [markdown]
# ### Sensitivity Analysis
#
# Progressively stronger unobserved confounders are injected to test whether
# the estimated effect flips sign. A robust effect survives strong confounding;
# a fragile one flips at low confounder strength.


# %%
def run_sensitivity(model, estimand, estimate):
    """Test robustness to unobserved confounding at increasing strengths.

    Returns (sensitivity_results, flip_strength) where sensitivity_results
    is a list of {strength, effect} dicts and flip_strength is the first
    strength at which the effect sign flips (or None if it never flips).
    """
    effect_strengths = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    sensitivity_results = []

    for strength in effect_strengths:
        try:
            refute = model.refute_estimate(
                estimand,
                estimate,
                method_name="add_unobserved_common_cause",
                confounders_effect_on_treatment="binary_flip",
                confounders_effect_on_outcome="linear",
                effect_strength_on_treatment=strength,
                effect_strength_on_outcome=strength,
            )
            sensitivity_results.append({"strength": strength, "effect": refute.new_effect})
        except Exception:
            pass

    # Find flip point
    original_sign = np.sign(estimate.value)
    flip_strength = None
    for r in sensitivity_results:
        if np.sign(r["effect"]) != original_sign and flip_strength is None:
            flip_strength = r["strength"]

    return sensitivity_results, flip_strength


# %% [markdown]
# ## 6. Analysis A: Returns as Outcome
#
# We first test whether extreme high premium *causes* forward BTC returns.
# The mechanism here is indirect: high premium reflects bullish sentiment,
# which may also drive returns through other channels.

# %%
n_sims = REFUTATION_SIMULATIONS

# Estimate on train
est_ret_train, estd_ret_train, model_ret_train = fit_and_estimate(
    df_train, "fwd_return_24h", graph_returns
)
# Estimate on test (out-of-sample)
est_ret_test, _, _ = fit_and_estimate(df_test, "fwd_return_24h", graph_returns)

ate_diff_ret = abs(est_ret_train.value - est_ret_test.value) / max(abs(est_ret_train.value), 1e-6)
stable_ret = ate_diff_ret < 0.5

print(f"ATE (train): {est_ret_train.value:.6f}")
print(f"ATE (test):  {est_ret_test.value:.6f}")
print(f"Train/test difference: {ate_diff_ret:.1%} {'(stable)' if stable_ret else '(unstable)'}")

# %%
# HAC-corrected inference on the same adjustment set
hac_ret = estimate_backdoor_ols_hac(df_train, "fwd_return_24h")
print(
    f"OLS+HAC effect: {hac_ret['effect']:.6f}, "
    f"SE: {hac_ret['se_hac']:.6f}, "
    f"t: {hac_ret['t_hac']:.2f}, "
    f"p: {hac_ret['p_hac']:.3f}"
)

# %%
refut_ret = run_refutations(model_ret_train, estd_ret_train, est_ret_train, n_sims)
for test_name, (passed, detail) in refut_ret.items():
    print(f"  {test_name}: {'PASS' if passed else 'FAIL'} ({detail})")

# %%
np.random.seed(SEED)
sens_ret, flip_ret = run_sensitivity(model_ret_train, estd_ret_train, est_ret_train)
max_tested = max(r["strength"] for r in sens_ret) if sens_ret else 0

if flip_ret is not None:
    sensitivity_ret = f"effect flips at sensitivity setting {flip_ret:.2f}"
    print(f"Effect FLIPS at sensitivity setting {flip_ret:.2f}")
else:
    sensitivity_ret = f"effect survives through sensitivity setting {max_tested:.2f}"
    print(f"Effect survives through sensitivity setting {max_tested:.2f}")
print(f"Sensitivity-test result: {sensitivity_ret}")

# %% [markdown]
# DoWhy's `effect_strength_on_treatment` / `effect_strength_on_outcome`
# settings are diagnostic perturbations, not calibrated economic effect
# sizes - they say nothing about percent-of-variance or any market-level
# interpretation. We report them as raw settings ("flips at 0.5") rather
# than percentages ("flips at 50% confounder strength") to avoid
# overinterpretation.
#
# The sensitivity analysis cannot flip the returns effect - but this does
# not mean the claim is credible. The OOS drift and same-scale negative
# control reported below are the binding diagnostics; sensitivity alone
# gives a false sense of security.

# %% [markdown]
# ## 7. Analysis B: Premium Reversion as Outcome
#
# Now we test whether extreme high premium causes premium *reversion*. The
# mechanism here is direct: high premium triggers arbitrage trades
# (sell perp, buy spot) that mechanically push the premium back toward zero.

# %%
# Estimate on train
est_rev_train, estd_rev_train, model_rev_train = fit_and_estimate(
    df_train, "fwd_premium_change", graph_reversion
)
# Estimate on test (out-of-sample)
est_rev_test, _, _ = fit_and_estimate(df_test, "fwd_premium_change", graph_reversion)

ate_diff_rev = abs(est_rev_train.value - est_rev_test.value) / max(abs(est_rev_train.value), 1e-6)
stable_rev = ate_diff_rev < 0.5

print(f"ATE (train): {est_rev_train.value:.6f}")
print(f"ATE (test):  {est_rev_test.value:.6f}")
print(f"Train/test difference: {ate_diff_rev:.1%} {'(stable)' if stable_rev else '(unstable)'}")

# %%
# HAC-corrected inference on the same adjustment set
hac_rev = estimate_backdoor_ols_hac(df_train, "fwd_premium_change")
print(
    f"OLS+HAC effect: {hac_rev['effect']:.6f}, "
    f"SE: {hac_rev['se_hac']:.6f}, "
    f"t: {hac_rev['t_hac']:.2f}, "
    f"p: {hac_rev['p_hac']:.3f}"
)

# %%
refut_rev = run_refutations(model_rev_train, estd_rev_train, est_rev_train, n_sims)
for test_name, (passed, detail) in refut_rev.items():
    print(f"  {test_name}: {'PASS' if passed else 'FAIL'} ({detail})")

# %%
np.random.seed(SEED)
sens_rev, flip_rev = run_sensitivity(model_rev_train, estd_rev_train, est_rev_train)
max_tested_rev = max(r["strength"] for r in sens_rev) if sens_rev else 0

if flip_rev is not None:
    sensitivity_rev = f"effect flips at sensitivity setting {flip_rev:.2f}"
    print(f"Effect FLIPS at sensitivity setting {flip_rev:.2f}")
else:
    sensitivity_rev = f"effect survives through sensitivity setting {max_tested_rev:.2f}"
    print(f"Effect survives through sensitivity setting {max_tested_rev:.2f}")
print(f"Sensitivity-test result: {sensitivity_rev}")

# %% [markdown]
# The reversion outcome also survives sensitivity analysis, but the
# placebo-date, same-scale negative-control, and OOS tests below tell
# the full story. The direct arbitrage mechanism gives the reversion
# outcome a more credible causal interpretation than the indirect
# sentiment channel - but only after every diagnostic supports the
# claim. Liquidity, funding congestion, exchange risk, and market-wide
# leverage can in principle affect both extreme-premium states and
# reversion speed, so the reversion claim is *less likely* to be
# confounded, not immune.

# %% [markdown]
# ## 8. Side-by-Side Comparison

# %%
flip_ret_str = f"{flip_ret:.0%}" if flip_ret else "Never"
flip_rev_str = f"{flip_rev:.0%}" if flip_rev else "Never"

comparison = pd.DataFrame(
    {
        "Metric": [
            "ATE (train)",
            "ATE (test)",
            "OOS drift",
            "Sensitivity flip",
        ],
        "Returns": [
            f"{est_ret_train.value:.6f}",
            f"{est_ret_test.value:.6f}",
            f"{ate_diff_ret:.1%}",
            flip_ret_str,
        ],
        "Reversion": [
            f"{est_rev_train.value:.6f}",
            f"{est_rev_test.value:.6f}",
            f"{ate_diff_rev:.1%}",
            flip_rev_str,
        ],
    }
).set_index("Metric")

display(comparison)

# %% [markdown]
# ### Sensitivity Curve: ATE vs Confounder Strength
#
# The plot below shows how the estimated ATE changes as we inject
# progressively stronger unobserved confounders. A credible causal
# claim keeps the same sign across all confounder strengths.

# %%
if sens_ret or sens_rev:
    fig, ax = plt.subplots(figsize=(8, 5))

    # Baseline points at strength=0
    ret_strengths = [0.0] + [r["strength"] for r in sens_ret]
    ret_effects = [est_ret_train.value] + [r["effect"] for r in sens_ret]
    rev_strengths = [0.0] + [r["strength"] for r in sens_rev]
    rev_effects = [est_rev_train.value] + [r["effect"] for r in sens_rev]

    # Explicit ML4T palette: the inline backend does not always honor the
    # repo matplotlibrc color cycle, so set the series colors directly.
    ax.plot(
        ret_strengths,
        ret_effects,
        marker="o",
        color=COLORS["blue"],
        label="Forward Returns (indirect)",
    )
    ax.plot(
        rev_strengths,
        rev_effects,
        marker="s",
        color=COLORS["copper"],
        linestyle="--",
        label="Premium Reversion (direct)",
    )

    ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax.set_xlabel("Unobserved Confounder Strength")
    ax.set_ylabel("Estimated ATE")
    ax.set_title("Sensitivity alone fails to separate the returns and reversion claims")
    ax.legend()

    fig.tight_layout()
    fig.show()

# %% [markdown]
# ## 9. Additional Validation: Placebo Date and Negative Control
#
# Section 15.2.4 describes two validation tests beyond the sensitivity
# analysis above. A **placebo-date test** shifts the treatment assignment by
# several days -- a credible effect should vanish when the treatment timing is
# wrong. A **negative-control outcome** uses a pre-treatment variable that
# the treatment cannot cause -- a nonzero effect signals confounding leakage.

# %% [markdown]
# ### Placebo-Date Test
#
# We shift the treatment (`extreme_high_premium`) forward by 21 bars
# (7 days at 8h frequency). At each row, the treatment value now comes
# from a week earlier -- well outside the 24h mechanism window. If the
# original effect is causal, the placebo effect should be near zero.

# %%
shift_periods = 21  # 7 days at 8h frequency

# Shift treatment variable and re-estimate for both outcomes
df_placebo = df_train.copy()
df_placebo["extreme_high_premium"] = df_placebo["extreme_high_premium"].shift(shift_periods)
df_placebo = df_placebo.dropna()

est_placebo_rev, _, _ = fit_and_estimate(df_placebo, "fwd_premium_change", graph_reversion)
est_placebo_ret, _, _ = fit_and_estimate(df_placebo, "fwd_return_24h", graph_returns)

placebo_df = pd.DataFrame(
    {
        "Outcome": ["Forward Returns", "Premium Reversion"],
        "Original ATE": [
            f"{est_ret_train.value:.6f}",
            f"{est_rev_train.value:.6f}",
        ],
        "Placebo ATE (7d shift)": [
            f"{est_placebo_ret.value:.6f}",
            f"{est_placebo_rev.value:.6f}",
        ],
        "Ratio |placebo/original|": [
            f"{abs(est_placebo_ret.value) / max(abs(est_ret_train.value), 1e-8):.1%}",
            f"{abs(est_placebo_rev.value) / max(abs(est_rev_train.value), 1e-8):.1%}",
        ],
    }
).set_index("Outcome")

display(placebo_df)

# %% [markdown]
# A credible causal effect should shrink substantially under the placebo --
# ideally to less than 20% of the original magnitude. If the placebo ATE
# remains large, the association reflects persistent confounding rather
# than a genuine treatment effect at the specified timing.

# %% [markdown]
# ### Negative-Control Outcomes
#
# Each headline outcome gets its own same-scale negative control:
# `past_return_48h` (24h return ending 48h before treatment) for the
# returns ATE, and `past_premium_change_48h` (24h premium change ending
# 48h before treatment) for the reversion ATE.
#
# Because the negative-control DAGs (correctly) omit any
# `extreme_high_premium → past_*` edge, DoWhy's backdoor identification
# refuses to attribute a causal effect - the *graph structure itself*
# rules the path out, and `fit_and_estimate` returns zero by construction.
# That is the right behavior for an identifiability check, but it gives
# us no numeric diagnostic of *residual association* in the data.
#
# To recover the diagnostic, we run an OLS regression of each past
# outcome on the treatment and confounders, with HAC standard errors. If
# the coefficient on treatment is non-trivial, the backdoor adjustment is
# not blocking all confounding paths - even though no real causal effect
# can exist on a pre-treatment outcome.

# %%
neg_ret_assoc = estimate_backdoor_ols_hac(df_train, "past_return_48h")
neg_rev_assoc = estimate_backdoor_ols_hac(df_train, "past_premium_change_48h")

neg_ratio_ret = abs(neg_ret_assoc["effect"]) / max(abs(hac_ret["effect"]), 1e-8)
neg_ratio_rev = abs(neg_rev_assoc["effect"]) / max(abs(hac_rev["effect"]), 1e-8)

print("Returns outcome (negative-control association via OLS+HAC):")
print(
    f"  past_return_48h ~ treatment: {neg_ret_assoc['effect']:.6f} "
    f"(SE {neg_ret_assoc['se_hac']:.6f}, t {neg_ret_assoc['t_hac']:.2f}, "
    f"p {neg_ret_assoc['p_hac']:.3f})"
)
print(f"  Relative to returns OLS+HAC effect: {neg_ratio_ret:.1%}")
print("Reversion outcome (negative-control association via OLS+HAC):")
print(
    f"  past_premium_change_48h ~ treatment: {neg_rev_assoc['effect']:.6f} "
    f"(SE {neg_rev_assoc['se_hac']:.6f}, t {neg_rev_assoc['t_hac']:.2f}, "
    f"p {neg_rev_assoc['p_hac']:.3f})"
)
print(f"  Relative to reversion OLS+HAC effect: {neg_ratio_rev:.1%}")

# %% [markdown]
# Each residual association should be negligible relative to its
# corresponding headline effect. A large ratio on one outcome but not the
# other signals that the headline claim on the affected outcome inherits
# residual confounding the backdoor adjustment did not block. Treat this
# as a placebo diagnostic - not an identifiable causal effect - because
# the DAG structurally forbids the path.

# %% [markdown]
# ## 10. Why the Difference? Mechanism Analysis
#
# The key insight: **mechanism tightness determines causal credibility**.
#
# ### Returns Outcome (Questionable)
# ```
# market_sentiment -> { premium, returns }  (CONFOUNDING PATH)
#           premium -> returns              (CAUSAL PATH - indirect)
# ```
# - Market sentiment affects BOTH premium AND returns
# - The causal path (premium $\rightarrow$ returns) is indirect
# - OOS drift and the same-scale negative control are the binding tests;
#   exact magnitudes appear in the run output above
#
# ### Reversion Outcome (More credible)
# ```
# premium -> arbitrage_pressure -> premium_reversion  (CAUSAL PATH - tight)
# ```
# - The mechanism is DIRECT: high premium $\rightarrow$ arbs sell perp,
#   buy spot $\rightarrow$ premium drops
# - The same-scale negative control (`past_premium_change_48h`) confirms
#   the pipeline is not manufacturing reversion-scale signal from
#   pre-treatment data
# - Placebo-date shift eliminates the effect, confirming the 24h
#   mechanism window

# %% [markdown]
# ## 11. Practical Implications
#
# - **Outcome design matters more than method sophistication**: A well-chosen
#   outcome with a tight mechanism (reversion) yields a more credible claim
#   that passes the negative-control and placebo-date checks; a confounded
#   outcome (returns) fails them under scrutiny.
# - **Use multiple validation tests**: Sensitivity analysis alone can miss
#   problems that the negative control, placebo-date test, or out-of-sample
#   stability reveal. No single test is sufficient.
# - **Always validate out-of-sample**: If the ATE differs by more than 50%
#   between train and test, the identification strategy is suspect.

# %% [markdown]
# ## 12. Results Summary

# %%
print(f"Train samples: {len(df_train):,}  |  Test samples: {len(df_test):,}")
print(f"Treatment: extreme high premium (z > {EXTREME_THRESHOLD})\n")
print(
    f"Returns  -- ATE train: {est_ret_train.value:.6f}, "
    f"test: {est_ret_test.value:.6f}, "
    f"sensitivity: {sensitivity_ret}"
)
print(
    f"Reversion -- ATE train: {est_rev_train.value:.6f}, "
    f"test: {est_rev_test.value:.6f}, "
    f"sensitivity: {sensitivity_rev}"
)

# %% [markdown]
# The same treatment yields different causal credibility depending on the
# outcome variable. The decisive discriminator is the same-scale negative
# control: for the reversion outcome the pre-treatment placebo association is
# a few percent of the headline effect and statistically insignificant, while
# for the returns outcome it is several times the headline effect and highly
# significant - proof that the returns backdoor adjustment leaves large
# residual confounding. The reversion outcome also passes the placebo-date
# shift at a tighter ratio than returns, and neither estimate flips sign under
# the sensitivity-analysis setting range tested. Out-of-sample stability
# separates the two only by degree: both train/test drifts exceed the 50%
# guideline, so neither outcome is OOS-stable in absolute terms, but the
# returns drift is far larger than the reversion drift. The returns claim
# therefore fails the two binding checks (negative control and OOS drift)
# while surviving sensitivity analysis on confounder strength alone - a
# reminder that sensitivity alone is not the binding test. Outcome choice,
# not estimator sophistication, dominates the credibility of the claim.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Mechanism tightness matters**: Arbitrage (direct) > sentiment (indirect)
# 2. **Multiple validation tests are essential**: Sensitivity analysis, placebo-date,
#    negative control, and OOS stability each catch different failure modes
# 3. **Outcome choice is a design decision**: Frame questions for credible answers
# 4. **Honest reporting**: State exactly which validation tests an effect passes
#    or fails, with the confounder-strength threshold that flips it
# 5. **Train/test validation**: Always check out-of-sample consistency
# 6. **Timing discipline**: Ensure confounders precede treatment precede outcome
