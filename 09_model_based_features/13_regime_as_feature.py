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
# # Regime as Feature
#
# **Docker image**: `ml4t`
#
# This notebook demonstrates the **regime-as-feature** methodology: using
# regime probabilities as input features to ML models, rather than switching
# between specialized models based on detected regime.
#
# **Learning Objectives**:
# - Implement regime probabilities as ML features (vs hard switching)
# - Compare regime-as-feature vs mixture-of-experts approaches
# - Evaluate regime-aware vs baseline model performance
# - Handle transition period instability in regime features
#
# **Book Reference**: Chapter 9, Section 9.5 (Regime Features)
#
# **Prerequisites**: `11_hmm_regimes` for HMM regime detection and
# `12_wasserstein_regimes` for distribution-based regime clustering.
#
# ### Scope: methodology demonstration, not lookahead-safe production
#
# The HMM in this notebook is fit on the **full sample** before the downstream
# regressors are evaluated via `TimeSeriesSplit`. The full-sample HMM fit
# means the regime probabilities used as features carry information from
# future folds into the training set; the downstream CV metrics are therefore
# biased upward. The notebook serves as a **methodology demonstration**: it
# shows how to wire regime probabilities into a regression and contrasts the
# regime-as-feature pattern with mixture of experts. The reported RMSE and R²
# illustrate the *mechanics*, not lookahead-safe out-of-sample performance.
#
# Lookahead-safe regime features (HMM refit inside each walk-forward fold,
# filtered probabilities for the test period only) are demonstrated in the
# case studies under `case_studies/` from Chapter 16 onward, where regime
# features enter the per-case-study feature pipeline with point-in-time
# discipline. The walk-forward caveat is repeated inline at the HMM-fit cell
# below.

# %%
"""Regime as Feature — use regime probabilities as ML input features for regime-aware prediction."""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from hmmlearn.hmm import GaussianHMM
from IPython.display import display
from ml4t.engineer.features.ml import regime_conditional_features, rolling_entropy
from ml4t.engineer.features.regime import (
    choppiness_index,
    hurst_exponent,
    market_regime_classifier,
)
from ml4t.engineer.features.volatility import (
    garch_forecast,
    realized_volatility,
    volatility_regime_probability,
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from data import load_etfs, load_macro
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
START_DATE = "2005-01-01"
END_DATE = "2024-06-30"
N_SPLITS = 5
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## Load Data and Create Features

# %%
# Load SPY from ETF universe
etfs = load_etfs(symbols=["SPY"])
spy = etfs.select(["timestamp", "close"]).rename({"close": "SP500"}).sort("timestamp")

# Load VIX from FRED macro
macro = load_macro()
vix = macro.select(["timestamp", "vixcls"]).rename({"vixcls": "VIXCLS"}).sort("timestamp")

# Join SPY and VIX
data = spy.join(vix, on="timestamp", how="inner").drop_nulls().sort("timestamp")

# Filter date range
data = data.filter(
    (pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    & (pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
)

# Feature engineering in Polars (before pandas conversion)
data = (
    data.with_columns(
        returns=pl.col("SP500").log().diff() * 100,
    )
    .with_columns(
        volatility=pl.col("returns").rolling_std(window_size=21) * np.sqrt(252),
        momentum_21=pl.col("returns").rolling_sum(window_size=21),
        momentum_63=pl.col("returns").rolling_sum(window_size=63),
        vix_ma=pl.col("VIXCLS").rolling_mean(window_size=21),
    )
    .with_columns(
        vix_zscore=(pl.col("VIXCLS") - pl.col("vix_ma"))
        / pl.col("VIXCLS").rolling_std(window_size=63),
    )
    .drop_nulls()
)

# Convert to pandas for sklearn and hmmlearn
# NOTE: sklearn, hmmlearn require numpy/pandas input
df = data.to_pandas().set_index("timestamp")
df.index = pd.DatetimeIndex(df.index)

# Target: 5-day forward return (pandas for shift(-5) which is look-ahead)
df["target"] = df["returns"].shift(-5).rolling(5).sum()
df = df.dropna()

print(f"Data: {len(df)} observations from {df.index.min()} to {df.index.max()}")

# %% [markdown]
# ## Regime Detection with HMM
#
# We refit a simple 2-state HMM here for self-containedness. See
# `11_hmm_regimes` for a thorough treatment of initialization, model
# selection, and label switching.
#
# **Critical**: We use **filtered** probabilities (forward algorithm only)
# as features. Smoothed probabilities use future data and would introduce
# look-ahead bias into the downstream ML models.


# %%
def compute_filtered_probs(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Compute filtered probabilities P(state_t | obs_{1:t}).

    Uses the forward algorithm internally, then normalizes.
    Note: uses hmmlearn's private _compute_log_likelihood API.
    """
    framelogprob = model._compute_log_likelihood(X)
    n_samples = X.shape[0]
    n_components = model.n_components

    log_startprob = np.log(model.startprob_ + 1e-300)
    log_transmat = np.log(model.transmat_ + 1e-300)

    fwdlattice = np.zeros((n_samples, n_components))
    fwdlattice[0] = log_startprob + framelogprob[0]

    for t in range(1, n_samples):
        for j in range(n_components):
            fwdlattice[t, j] = framelogprob[t, j] + np.logaddexp.reduce(
                fwdlattice[t - 1] + log_transmat[:, j]
            )

    log_normalizer = np.logaddexp.reduce(fwdlattice, axis=1, keepdims=True)
    return np.exp(fwdlattice - log_normalizer)


# %%
# Prepare data for HMM
X_hmm = df[["returns", "volatility"]].values

# Fit HMM with 2 states
n_states = 2
hmm_model = GaussianHMM(
    n_components=n_states,
    covariance_type="full",
    n_iter=100,
    random_state=42,
)
hmm_model.fit(X_hmm)

# Filtered (causal) probabilities; hard states are their argmax so both are
# consistent and use only past/current observations. We deliberately do NOT
# use `hmm_model.predict(X_hmm)`: that is Viterbi global decoding, which
# conditions on the entire sample (including future observations) and must
# not feed test-time expert routing (see 11_hmm_regimes, which saves
# argmax-of-filtered states for the same reason).
regime_probs = compute_filtered_probs(hmm_model, X_hmm)
df["regime"] = regime_probs.argmax(axis=1)

# Label states by volatility (ensure consistent labeling)
regime_vols = df.groupby("regime")["volatility"].mean()
low_vol_state = regime_vols.idxmin()
high_vol_state = regime_vols.idxmax()

# Rename to ensure state 0 = low vol, state 1 = high vol
# Simplified label switching — see 11_hmm_regimes for variance-based sorting
if low_vol_state == 1:
    df["regime"] = 1 - df["regime"]
    regime_probs = regime_probs[:, ::-1]

df["prob_high_vol"] = regime_probs[:, 1]
df["prob_low_vol"] = regime_probs[:, 0]

print("=== Regime Characteristics ===")
for regime in [0, 1]:
    mask = df["regime"] == regime
    label = "Low Vol" if regime == 0 else "High Vol"
    pct = mask.mean() * 100
    mean_ret = df.loc[mask, "returns"].mean()
    mean_vol = df.loc[mask, "volatility"].mean()
    print(f"{label}: {pct:.1f}% of time, mean ret: {mean_ret:.3f}%, vol: {mean_vol:.1f}%")

# %% [markdown]
# **Walk-forward caveat**: The HMM above is fit on the full dataset for
# demonstration simplicity. In production, the HMM must be refit inside
# each walk-forward fold — fit on training data, then extract filtered
# probabilities only for the test period. The full-sample fit here means
# regime features carry information from future folds into the training
# set, biasing the downstream CV results upward. The comparison below
# still illustrates the *relative* benefit of regime-as-feature vs
# mixture-of-experts, but absolute performance numbers should not be
# taken at face value.

# %% [markdown]
# ## Approach 1: Regime-as-Feature
#
# Include the regime probability as a feature in a single model.
#
# **Properties relative to mixture of experts**:
# - Single model to maintain rather than one per regime
# - Smooth transitions at regime boundaries (the feature varies continuously)
# - Model learns the weighting of regime information from the data
# - Misclassified regimes degrade the feature, not the model assignment

# %%
# Define feature sets
base_features = ["returns", "volatility", "momentum_21", "momentum_63", "vix_zscore"]
regime_features = base_features + ["prob_high_vol"]

# Prepare data
X_base = df[base_features].values
X_regime = df[regime_features].values
y = df["target"].values

# Time-series cross-validation. gap=5 purges the 5 bars whose forward-return
# target (shift(-5).rolling(5)) overlaps the test fold — see the label-buffer
# (purge) discussion in 06_strategy_definition/02_cv_foundations.
tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=5)

# Scale features
scaler_base = StandardScaler()
scaler_regime = StandardScaler()


def evaluate_model(model_class, X, y, scaler, **kwargs):
    """Evaluate model with time-series cross-validation."""
    scores_rmse = []
    scores_r2 = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit and predict
        model = model_class(**kwargs)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        scores_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        scores_r2.append(r2_score(y_test, y_pred))

    return np.mean(scores_rmse), np.mean(scores_r2)


# Evaluate models with and without regime features
print("=== Approach 1: Regime-as-Feature ===\n")
print("Ridge Regression:")
rmse_base, r2_base = evaluate_model(Ridge, X_base, y, scaler_base, alpha=1.0)
rmse_regime, r2_regime = evaluate_model(Ridge, X_regime, y, scaler_regime, alpha=1.0)
print(f"  Without regime: RMSE={rmse_base:.4f}, R²={r2_base:.4f}")
print(f"  With regime:    RMSE={rmse_regime:.4f}, R²={r2_regime:.4f}")
improvement_ridge = (rmse_base - rmse_regime) / rmse_base * 100
print(f"  Improvement:    {improvement_ridge:.2f}% RMSE reduction")

print("\nGradient Boosting:")
rmse_base_gb, r2_base_gb = evaluate_model(
    GradientBoostingRegressor,
    X_base,
    y,
    scaler_base,
    n_estimators=100,
    max_depth=3,
    random_state=42,
)
rmse_regime_gb, r2_regime_gb = evaluate_model(
    GradientBoostingRegressor,
    X_regime,
    y,
    scaler_regime,
    n_estimators=100,
    max_depth=3,
    random_state=42,
)
print(f"  Without regime: RMSE={rmse_base_gb:.4f}, R²={r2_base_gb:.4f}")
print(f"  With regime:    RMSE={rmse_regime_gb:.4f}, R²={r2_regime_gb:.4f}")
improvement_gb = (rmse_base_gb - rmse_regime_gb) / rmse_base_gb * 100
print(f"  Improvement:    {improvement_gb:.2f}% RMSE reduction")

# %% [markdown]
# ## Approach 2: Mixture of Experts
#
# **Alternative approach**: Train separate models for each regime and switch between them.
#
# **Advantages**:
# - Models can capture fundamentally different dynamics
# - More interpretable (separate coefficients per regime)
#
# **Disadvantages**:
# - Less data per model (training split by regime)
# - Sharp transitions at regime boundaries
# - Sensitive to regime misclassification
# - Multiple models to maintain


# %%
def mixture_of_experts_cv(X, y, regimes, model_class, tscv, **kwargs):
    """Evaluate mixture of experts with time-series CV."""
    scores_rmse = []
    scores_r2 = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        regime_train = regimes[train_idx]
        regime_test = regimes[test_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train separate models per regime
        models = {}
        for r in [0, 1]:
            mask = regime_train == r
            if mask.sum() < 20:  # Not enough data
                continue
            model = model_class(**kwargs)
            model.fit(X_train_scaled[mask], y_train[mask])
            models[r] = model

        # Predict using regime-specific models
        y_pred = np.zeros(len(y_test))
        for r in [0, 1]:
            mask = regime_test == r
            if r in models and mask.sum() > 0:
                y_pred[mask] = models[r].predict(X_test_scaled[mask])
            elif mask.sum() > 0:
                # Fallback if no model for this regime
                fallback_model = list(models.values())[0]
                y_pred[mask] = fallback_model.predict(X_test_scaled[mask])

        scores_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        scores_r2.append(r2_score(y_test, y_pred))

    return np.mean(scores_rmse), np.mean(scores_r2)


print("=== Approach 2: Mixture of Experts ===\n")
regimes = df["regime"].values

print("Ridge Regression (separate models per regime):")
rmse_moe, r2_moe = mixture_of_experts_cv(X_base, y, regimes, Ridge, tscv, alpha=1.0)
print(f"  RMSE={rmse_moe:.4f}, R²={r2_moe:.4f}")
print(f"  vs Single model: {'Better' if rmse_moe < rmse_base else 'Worse'}")
print(f"  vs Regime-as-feature: {'Better' if rmse_moe < rmse_regime else 'Worse'}")

print("\nGradient Boosting (separate models per regime):")
rmse_moe_gb, r2_moe_gb = mixture_of_experts_cv(
    X_base,
    y,
    regimes,
    GradientBoostingRegressor,
    tscv,
    n_estimators=100,
    max_depth=3,
    random_state=42,
)
print(f"  RMSE={rmse_moe_gb:.4f}, R²={r2_moe_gb:.4f}")
print(f"  vs Single model: {'Better' if rmse_moe_gb < rmse_base_gb else 'Worse'}")
print(f"  vs Regime-as-feature: {'Better' if rmse_moe_gb < rmse_regime_gb else 'Worse'}")

# %% [markdown]
# ## Comparison Summary

# %%
# Collect results
results = pd.DataFrame(
    {
        "Approach": [
            "Baseline (no regime)",
            "Regime-as-Feature",
            "Mixture of Experts",
        ],
        "Ridge RMSE": [rmse_base, rmse_regime, rmse_moe],
        "Ridge R²": [r2_base, r2_regime, r2_moe],
        "GB RMSE": [rmse_base_gb, rmse_regime_gb, rmse_moe_gb],
        "GB R²": [r2_base_gb, r2_regime_gb, r2_moe_gb],
    }
)

display(results)

# Best approach
best_rmse = results["GB RMSE"].min()
best_approach = results.loc[results["GB RMSE"].idxmin(), "Approach"]
print(f"\nBest approach (GB by RMSE): {best_approach}")

# %% [markdown]
# ## Visualizing Regime-Aware Predictions

# %%
# Train final model with regime features on last 80% of data
split_idx = int(len(df) * 0.8)
X_train_final = X_regime[:split_idx]
X_test_final = X_regime[split_idx:]
y_train_final = y[:split_idx]
y_test_final = y[split_idx:]

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_final)
X_test_scaled = scaler_final.transform(X_test_final)

model_final = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
model_final.fit(X_train_scaled, y_train_final)
y_pred_final = model_final.predict(X_test_scaled)

# Get test period data
test_dates = df.index[split_idx:]
test_df = df.iloc[split_idx:].copy()
test_df["predicted"] = y_pred_final
test_df["actual"] = y_test_final

# %%
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Price with regime coloring
ax = axes[0]
colors = [COLORS["blue"], COLORS["copper"]]  # Low vol (calm) vs high vol (stress)
regime_names = ["Low Vol", "High Vol"]
for regime in range(2):
    mask = test_df["regime"] == regime
    ax.scatter(
        test_df.index[mask],
        test_df.loc[mask, "SP500"],
        c=colors[regime],
        s=1,
        alpha=0.5,
        label=regime_names[regime],
    )
ax.set_ylabel("S&P 500 (index level)")
ax.set_title("Test-Period S&P 500 Colored by Detected Regime")
ax.legend(loc="upper left")

# Regime probabilities
ax = axes[1]
ax.fill_between(
    test_df.index,
    0,
    test_df["prob_high_vol"],
    alpha=0.7,
    color=COLORS["copper"],
    label="P(High Vol)",
)
ax.set_ylabel("Probability")
ax.set_title("Filtered High-Volatility Regime Probability")

# Predictions vs Actual
ax = axes[2]
ax.plot(
    test_df.index,
    test_df["actual"],
    label="Actual",
    linewidth=0.8,
    alpha=0.7,
    color=COLORS["neutral"],
)
ax.plot(
    test_df.index,
    test_df["predicted"],
    label="Predicted",
    linewidth=0.8,
    alpha=0.9,
    color=COLORS["blue"],
)
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_ylabel("5-Day Return (%)")
ax.set_title("Regime-Aware Model: Predicted vs Actual 5-Day Returns")
ax.legend()

# Prediction error by regime
ax = axes[3]
test_df["error"] = test_df["predicted"] - test_df["actual"]
for regime in range(2):
    mask = test_df["regime"] == regime
    ax.scatter(
        test_df.index[mask],
        test_df.loc[mask, "error"],
        c=colors[regime],
        s=5,
        alpha=0.5,
        label=f"{regime_names[regime]} error",
    )
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_ylabel("Prediction Error (%)")
ax.set_title("Prediction Errors Are Larger in the High-Vol Regime")
ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Feature Importance Analysis

# %%
# Feature importance from the regime-aware model
feature_importance = pd.DataFrame(
    {"Feature": regime_features, "Importance": model_final.feature_importances_}
).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(feature_importance["Feature"], feature_importance["Importance"], color=COLORS["blue"])
ax.set_xlabel("Impurity-based importance (Gradient Boosting)")
ax.set_title("Regime Probability Contributes Little; Base Features Dominate")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation**: If `prob_high_vol` ranks among the top features, the model
# is learning regime-conditional relationships — the regime-as-feature approach
# is adding value beyond what the base features provide. If it ranks low, the
# base features may already capture the regime information implicitly (e.g.,
# volatility correlates with regime state).

# %% [markdown]
# ## Error Analysis by Regime

# %%
print("=== Error Analysis by Regime ===\n")
for regime in [0, 1]:
    mask = test_df["regime"] == regime
    label = "Low Vol" if regime == 0 else "High Vol"
    rmse = np.sqrt(mean_squared_error(test_df.loc[mask, "actual"], test_df.loc[mask, "predicted"]))
    r2 = r2_score(test_df.loc[mask, "actual"], test_df.loc[mask, "predicted"])
    n = mask.sum()
    print(f"{label} Regime (n={n}):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print()

# %% [markdown]
# ## Technical Pitfalls
#
# See Section 9.5 and `11_hmm_regimes` for a thorough treatment of HMM
# estimation pitfalls (EM sensitivity, label switching, number-of-states
# selection). The key practical concern for regime-as-feature is
# **walk-forward discipline**: refit the HMM inside each CV fold to avoid
# leaking future regime information into training features.

# %% [markdown]
# ## Summary Statistics for Chapter

# %%
summary = {
    "Data observations": len(df),
    "Date range": f"{df.index.min().date()} to {df.index.max().date()}",
    "Low vol regime %": f"{(df['regime'] == 0).mean():.1%}",
    "High vol regime %": f"{(df['regime'] == 1).mean():.1%}",
    "Baseline RMSE (GB)": f"{rmse_base_gb:.4f}",
    "Regime-as-Feature RMSE (GB)": f"{rmse_regime_gb:.4f}",
    "Mixture of Experts RMSE (GB)": f"{rmse_moe_gb:.4f}",
    "Improvement from regime feature": f"{improvement_gb:.2f}%",
    "Regime feature importance": f"{feature_importance[feature_importance['Feature'] == 'prob_high_vol']['Importance'].values[0]:.3f}",
}

print("\n=== Summary for Chapter ===")
for key, value in summary.items():
    print(f"{key}: {value}")

# %% [markdown]
# ## ml4t-engineer: Temporal Feature Generation
#
# So far we used HMM probabilities as regime features. `ml4t-engineer`
# provides a richer feature toolkit combining regime indicators, volatility
# features, and entropy — all as Polars expressions composable in
# `with_columns()`. This demonstrates the feature catalog that would feed
# into downstream ML models (Ch11-12).

# %%
# Reload data as Polars for ml4t-engineer feature generation
etfs = load_etfs(symbols=["SPY"])
spy_pl = (
    etfs.select(["timestamp", "open", "high", "low", "close", "volume"])
    .sort("timestamp")
    .with_columns(returns=pl.col("close").pct_change())
    .drop_nulls()
)

# Generate a comprehensive feature set in one pipeline
spy_features = spy_pl.with_columns(
    hurst=hurst_exponent("close", period=100),
    chop=choppiness_index("high", "low", "close", period=14),
    regime=market_regime_classifier("high", "low", "close", "volume"),
    rv_20=realized_volatility("returns", period=20),
    garch_vol=garch_forecast("returns", horizon=1, alpha=0.1, beta=0.85),
    entropy=rolling_entropy("returns", window=50, n_bins=10),
)

# Add volatility regime probabilities (returns dict of expressions)
vol_regime_exprs = volatility_regime_probability("returns")
spy_features = spy_features.with_columns(**vol_regime_exprs)

print(f"Feature matrix: {spy_features.shape}")
spy_features.select(["timestamp", "hurst", "chop", "regime", "rv_20", "garch_vol", "entropy"]).tail(
    5
)

# %% [markdown]
# ### Regime-Conditional Features
#
# `regime_conditional_features()` creates interaction terms between a base
# feature and detected regime — e.g., "momentum only in trending regime."
# This avoids manual if/else logic and produces sparse features that tree
# models can split on efficiently.

# %%
# Create regime-conditional momentum features
cond_features = regime_conditional_features("returns", "regime", regime_values=[-1, 0, 1])
spy_cond = spy_features.with_columns(**cond_features)

print("=== Regime-Conditional Features ===")
for col_name in sorted(cond_features.keys()):
    vals = spy_cond[col_name].drop_nulls()
    non_zero = (vals != 0).sum()
    print(f"  {col_name}: {non_zero} non-zero values ({100 * non_zero / len(vals):.1f}%)")

# %% [markdown]
# **Interpretation**: Each conditional feature is non-zero only during its
# assigned regime, creating a natural interaction effect. For example,
# `returns_bear` (regime=-1) captures momentum only during range-bound
# periods, while `returns_bull` (regime=1) captures it during strong trends.
# Tree-based models can learn regime-specific coefficients from these features.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Regime-as-Feature** uses a single model with the regime probability
#    as an additional feature; on this single-symbol test the GB
#    Regime-as-Feature configuration achieves the lowest RMSE among the
#    three configurations reported in the comparison table above
# 2. **Mixture of Experts** trains a separate model per regime, which reduces
#    the training sample per model and produces sharper transitions at
#    regime boundaries
# 3. **Regime probability importance varies with the base feature set** — on
#    this single-symbol test `prob_high_vol` ranks low (~0.01 impurity-based
#    importance) because the base features (notably `volatility`) already encode
#    most of the regime signal; it can rank higher when the base features are
#    less regime-informative
# 4. **Error patterns differ by regime** — high-vol periods harder to predict
# 5. **Watch for pitfalls** — EM sensitivity, label switching, overfitting
# 6. **Cross-validate properly** — always use time-series CV for financial data
# 7. **ml4t-engineer composes features declaratively** — regime indicators,
#    volatility estimators, entropy, and conditional features in a single
#    `with_columns()` pipeline
# 8. **Regime-conditional features** create natural interaction terms that
#    tree models exploit without manual feature engineering
#
# **Previous**: `11_hmm_regimes` for HMM estimation and filtered probability
# extraction; `12_wasserstein_regimes` for distribution-based regime clustering.
# **Next**: `14_panel_features` for cross-sectional and panel feature construction.
