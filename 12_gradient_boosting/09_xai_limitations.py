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
# # XAI Limitations: Explanation Instability
#
# **Chapter 12, Section 12.5**: Model Explainability with SHAP
#
# ## Purpose
# This notebook demonstrates critical limitations of model explanations
# that practitioners must understand before relying on SHAP for
# downstream decisions (feature pruning, risk allocation, reporting).
#
# ## Learning Objectives
# - Identify explanation instability: similar predictions with different SHAP profiles
# - Demonstrate the Rashomon effect across model families (GBM vs Random Forest)
# - Understand when SHAP attributions reflect model-specific fitting, not data truth
#
# ## Cross-References
# - **Section 12.5**: Rashomon effect, explanation multiplicity
# - **Related**: `08_shap_analysis` (SHAP fundamentals), `11_conformal_gbm` (UQ)

# %% [markdown]
# ## 1. Setup

# %%
"""XAI Limitations — demonstrate explanation instability across seeds and model families."""

import warnings

# lightgbm must be imported before anything that pulls in scikit-learn, which
# ml4t.diagnostic does transitively: the first OpenMP runtime loaded wins the
# process, and getting this order wrong segfaults on macOS ARM64. Keeping it in
# this block is what makes the order stable - isort sorts plain `import x` above
# `from x import y`, so lightgbm cannot drift back below the ml4t import.
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from ml4t.diagnostic.metrics import cross_sectional_ic_series

warnings.filterwarnings("ignore")

import shap
from sklearn.ensemble import RandomForestRegressor

from utils.modeling import load_modeling_dataset
from utils.reproducibility import set_global_seeds
from utils.style import COLOR_CYCLER

# %% tags=["parameters"]
MAX_SYMBOLS = 0  # 0 = full universe (production); test override reduces for a fast smoke run
SEED = 42


# %%
set_global_seeds(SEED)

# %% [markdown]
# ## 2. Load Data

# %%
mds = load_modeling_dataset("etfs", "fwd_ret_21d", max_symbols=MAX_SYMBOLS)
df = mds.dataset.to_pandas()
date_col = mds.date_col
FEATURE_COLS = mds.feature_names

# Use first walk-forward fold
split = mds.splits[0]
train_mask = (df[date_col] >= split["train_start"]) & (df[date_col] <= split["train_end"])
test_mask = (df[date_col] >= split["val_start"]) & (df[date_col] <= split["val_end"])

X_train = df.loc[train_mask, FEATURE_COLS].values
y_train = df.loc[train_mask, mds.label_col].values
X_test = df.loc[test_mask, FEATURE_COLS].values
y_test = df.loc[test_mask, mds.label_col].values

# Drop NaN labels
train_valid = np.isfinite(y_train)
test_valid = np.isfinite(y_test)
X_train, y_train = X_train[train_valid], y_train[train_valid]
X_test, y_test = X_test[test_valid], y_test[test_valid]

# Test dates and symbols for cross-sectional IC
test_entity_col = mds.entity_cols[0]
dates_test = df.loc[test_mask, date_col].values[test_valid]
symbols_test = df.loc[test_mask, test_entity_col].values[test_valid]


def cross_sectional_ic_mean(y_true, y_pred, dates, symbols):
    pred_df = pl.DataFrame({"timestamp": dates, "symbol": symbols, "prediction": y_pred})
    ret_df = pl.DataFrame({"timestamp": dates, "symbol": symbols, "forward_return": y_true})
    ic_per_date = cross_sectional_ic_series(
        pred_df,
        ret_df,
        pred_col="prediction",
        ret_col="forward_return",
        date_col="timestamp",
        entity_col="symbol",
    )
    ic_clean = ic_per_date.drop_nulls("ic")
    return float(ic_clean["ic"].mean()) if ic_clean.height else float("nan")


print(f"ETFs: {len(X_train):,} train / {len(X_test):,} test ({len(FEATURE_COLS)} features)")

# %% [markdown]
# ## 3. Demonstration 1: Similar Predictions, Different Explanations
#
# Two samples with nearly identical predictions can have entirely
# different SHAP profiles — the model arrived at the same answer
# through different reasoning paths.

# %%
model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, random_state=SEED, verbose=-1)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
predictions = model.predict(X_test)

# %%
# Find pairs with similar predictions but different SHAP
n_samples = min(500, len(predictions))
prediction_diffs = np.abs(predictions[:n_samples, None] - predictions[None, :n_samples])
shap_diffs = np.sqrt(
    ((shap_values[:n_samples, None, :] - shap_values[None, :n_samples, :]) ** 2).sum(axis=2)
)

# Use generous thresholds: prediction diff in bottom 25%, SHAP diff in top 75%
pred_threshold = np.percentile(prediction_diffs[prediction_diffs > 0], 25)
shap_threshold = np.percentile(shap_diffs[shap_diffs > 0], 75)

mask = (prediction_diffs < pred_threshold) & (shap_diffs > shap_threshold) & (prediction_diffs > 0)
pairs = np.argwhere(mask)

# %%
print(f"Searched {n_samples} samples: {len(pairs)} instability pairs found")
print(f"  Prediction threshold: {pred_threshold:.6f}")
print(f"  SHAP distance threshold: {shap_threshold:.4f}")

if len(pairs) > 0:
    i, j = pairs[0]
    instability_df = pl.DataFrame(
        {
            "feature": FEATURE_COLS,
            f"SHAP (sample {i})": shap_values[i].round(4),
            f"SHAP (sample {j})": shap_values[j].round(4),
            "abs_diff": np.abs(shap_values[i] - shap_values[j]).round(4),
        }
    ).sort("abs_diff", descending=True)

    print(f"\nExample: samples {i} and {j}")
    print(f"  Prediction diff: {abs(predictions[i] - predictions[j]):.6f}")
    print(f"  SHAP Euclidean distance: {shap_diffs[i, j]:.4f}")
else:
    instability_df = pl.DataFrame(
        schema={
            "feature": str,
            "SHAP (sample a)": float,
            "SHAP (sample b)": float,
            "abs_diff": float,
        }
    )
    print("\nNo pairs met both thresholds — predictions and SHAP are tightly coupled")
    print("in this dataset. This is informative: it means the model's explanation")
    print("space is relatively stable for the ETF feature set.")
instability_df.head(10)

# %% [markdown]
# **Interpretation**: SHAP explanations are not unique — the same prediction
# can be reached through different feature contribution paths. When instability
# pairs are found, the top-three SHAP contributors can differ completely even
# when predictions agree to within fractions of a percent. For production
# deployment, report confidence intervals on feature importance rather than
# point estimates.

# %% [markdown]
# ## 4. Demonstration 2: Rashomon Effect Across Model Families
#
# Models with similar predictive performance can attribute predictions
# to different features. We compare three LightGBM seeds plus a
# Random Forest to demonstrate cross-architecture disagreement.

# %%
# Enable stochastic training (subsample < 1) so different seeds produce
# different models — without this, LightGBM is fully deterministic.
_lgb_kw = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
)
model_configs = [
    ("LightGBM (seed=42)", lgb.LGBMRegressor(**_lgb_kw, random_state=42)),
    ("LightGBM (seed=123)", lgb.LGBMRegressor(**_lgb_kw, random_state=123)),
    ("LightGBM (seed=456)", lgb.LGBMRegressor(**_lgb_kw, random_state=456)),
    (
        "RandomForest",
        RandomForestRegressor(n_estimators=200, max_depth=6, random_state=SEED, n_jobs=-1),
    ),
]

# %%
model_results = []
importance_arrays = []

for name, m in model_configs:
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    ic = cross_sectional_ic_mean(y_test, pred, dates_test, symbols_test)

    exp = shap.TreeExplainer(m)
    sv = exp.shap_values(X_test)
    mean_abs = np.abs(sv).mean(axis=0)

    model_results.append({"model": name, "ic": round(ic, 4)})
    importance_arrays.append(mean_abs)

# %%
rashomon_df = pl.DataFrame(model_results)
rashomon_df

# %% [markdown]
# Read the table by the spread rather than by any single value. The three
# LightGBM seeds differ only in their random seed, and they span a wider range
# than separates any of them from the Random Forest, which lands inside that
# span. Changing the seed therefore moves validation IC by more than changing
# the model family does here. The Rashomon point holds in either direction:
# similar predictive performance, different feature attributions.

# %%
# Show top-5 features per model
for i, (name, _) in enumerate(model_configs):
    order = np.argsort(importance_arrays[i])[::-1][:5]
    top5 = [(FEATURE_COLS[idx], round(importance_arrays[i][idx], 4)) for idx in order]
    print(f"{name}: {', '.join(f'{f} ({v})' for f, v in top5)}")

# %%
# Grouped bar chart comparing SHAP importance profiles
fig, ax = plt.subplots(figsize=(12, 5))
n_feat = min(10, len(FEATURE_COLS))
top_feats_idx = np.argsort(importance_arrays[0])[::-1][:n_feat]
feat_names = [FEATURE_COLS[idx] for idx in top_feats_idx]

x = np.arange(n_feat)
width = 0.2
colors = COLOR_CYCLER[:4]  # four distinct categorical hues (blue, amber, copper, green)

for i, (name, _) in enumerate(model_configs):
    vals = [importance_arrays[i][idx] for idx in top_feats_idx]
    ax.bar(x + i * width, vals, width, label=name, color=colors[i])

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Mean |SHAP|")
ax.set_title("Feature-importance magnitudes shift across seeds and model family (Rashomon effect)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation**: Features that rank highly across all four models
# (both GBM and Random Forest) are more likely to reflect genuine data
# structure. Features that only one family highlights may reflect
# architecture-specific fitting patterns. When SHAP attributions
# drive downstream decisions, validate across model specifications.

# %% [markdown]
# ## 5. Stakeholder Guidance
#
# | Audience | Recommended Explanation | Key Caveat |
# |----------|------------------------|------------|
# | Risk Manager | SHAP summary plot | Correlation $\neq$ causation |
# | Trader | Feature importance rank | May vary with retraining |
# | Regulator | Model documentation | Include confidence intervals |
# | Quant Researcher | Full SHAP + interactions | Check stability across seeds |

# %% [markdown]
# ## 6. Key Takeaways
#
# 1. **Explanation instability**: Samples with similar predictions can have
#    different top SHAP contributors — the same output is reachable through
#    different feature-contribution paths.
#
# 2. **Rashomon effect**: Models with different architectures or random seeds
#    attribute predictions to different features — SHAP explanations reflect
#    model-specific fitting patterns, not ground truth about the data.
#
# 3. **Best practices**: Report confidence intervals on feature importance,
#    check stability across random seeds and model architectures, and never
#    claim SHAP proves causation.
#
# **Next**: See `08_shap_analysis` for SHAP fundamentals and drift detection,
# or Section 12.5 in the chapter text for the theoretical framework.
