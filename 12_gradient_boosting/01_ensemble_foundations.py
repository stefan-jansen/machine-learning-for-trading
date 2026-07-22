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
# # Ensemble Foundations: Random Forests vs Gradient Boosting
#
# **Docker image**: `ml4t`
#
# This notebook benchmarks bagging (Random Forests) against boosting (XGBoost, LightGBM,
# CatBoost) on the Chen-Pelger-Zhu (2020) firm characteristics dataset — 1.2M
# stock-month observations with 46 characteristics and predefined temporal splits.
#
# ## Learning Objectives
# - Compare Random Forest (bagging) against XGBoost, LightGBM, and CatBoost (boosting)
#   on a large cross-sectional financial dataset
# - Evaluate models using rank IC on temporal hold-out splits
# - Analyze feature importance differences across ensemble methods
# - Measure the IC gap between Random Forest and the three GBM libraries on this benchmark
#
# **Book reference**: Section 12.1 motivates the progression from averaging (RF) to
# sequential error correction (boosting). This notebook makes that comparison empirical.
#
# **Prerequisites**: Chen-Pelger-Zhu firm characteristics dataset (via `load_firm_characteristics`)
#
# **Cross-chapter**: Results feed into Ch14 (latent factor models) and Ch20 (model
# synthesis) for method comparison on the same benchmark.

# %%
"""Ensemble Foundations — benchmark bagging vs boosting on financial return prediction."""

import warnings

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from data import load_firm_characteristics
from utils.paths import display_path, get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLOR_CYCLER, COLORS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

OUTPUT_DIR = get_output_dir(12, "us_firm_characteristics")

# %% tags=["parameters"]
SEED = 42


# %%
set_global_seeds(SEED)
# %% [markdown]
# ## 1. Load Firm Characteristics Dataset
#
# The Chen-Pelger-Zhu dataset provides a clean benchmark: 46 anonymized firm
# characteristics with predefined temporal splits that avoid any lookahead.

# %%
train_df = load_firm_characteristics(split="train")
valid_df = load_firm_characteristics(split="valid")
test_df = load_firm_characteristics(split="test")

feature_cols = [c for c in train_df.columns if c not in ["timestamp", "ret", "split", "stock_id"]]

X_train = train_df.select(feature_cols).to_numpy()
y_train = train_df["ret"].to_numpy()
X_valid = valid_df.select(feature_cols).to_numpy()
y_valid = valid_df["ret"].to_numpy()
X_test = test_df.select(feature_cols).to_numpy()
y_test = test_df["ret"].to_numpy()

print(
    f"Train: {len(X_train):,} obs (1967-1989) | Valid: {len(X_valid):,} (1990-1999) | Test: {len(X_test):,} (2000-2016)"
)
print(f"Features: {len(feature_cols)} characteristics")

# %% [markdown]
# ## 2. Evaluation Framework
#
# We use Spearman rank IC as the primary metric (measures cross-sectional ranking
# quality), supplemented by $R^2$ and MSE.


# %%
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute IC (Spearman), R², and MSE."""
    ic, ic_pvalue = stats.spearmanr(y_true, y_pred)
    return {
        "ic": ic,
        "ic_pvalue": ic_pvalue,
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
    }


# %% [markdown]
# ## 3. Model Comparison
#
# We train four models with hyperparameters following Gu, Kelly, and Xiu (2020)
# guidelines for asset pricing. The Random Forest serves as the bagging baseline
# that §12.1 argues GBMs must beat to justify their sequential complexity.

# %%
results = []
predictions = {}

# %% [markdown]
# ### 3.1 Random Forest (Bagging Baseline)
#
# Random Forests average independent trees — effective for variance reduction
# but unable to correct systematic bias (§12.1).

# %%
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=100,
    max_features=0.3,
    n_jobs=-1,
    random_state=42,
    verbose=0,
)
rf_model.fit(X_train, y_train)

rf_metrics = {
    "valid": evaluate_model(y_valid, rf_model.predict(X_valid)),
    "test": evaluate_model(y_test, rf_model.predict(X_test)),
}
predictions["RandomForest"] = rf_model.predict(X_test)
results.append(
    {
        "model": "Random Forest",
        **{f"{s}_{k}": v for s, m in rf_metrics.items() for k, v in m.items()},
    }
)

print(
    f"Random Forest — Valid IC: {rf_metrics['valid']['ic']:.4f}, Test IC: {rf_metrics['test']['ic']:.4f}"
)

# %% [markdown]
# ### 3.2 XGBoost
#
# XGBoost adds L1/L2 regularization on leaf weights and uses second-order
# gradient approximations. We enable early stopping to demonstrate automatic
# iteration selection — the model stops when validation loss plateaus.

# %%
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,  # high ceiling — early stopping selects actual count
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=100,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    early_stopping_rounds=50,
    random_state=42,
    verbosity=0,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

xgb_metrics = {
    "valid": evaluate_model(y_valid, xgb_model.predict(X_valid)),
    "test": evaluate_model(y_test, xgb_model.predict(X_test)),
}
predictions["XGBoost"] = xgb_model.predict(X_test)
results.append(
    {"model": "XGBoost", **{f"{s}_{k}": v for s, m in xgb_metrics.items() for k, v in m.items()}}
)

print(
    f"XGBoost — Valid IC: {xgb_metrics['valid']['ic']:.4f}, Test IC: {xgb_metrics['test']['ic']:.4f}"
)
print(f"  Early stopping at {xgb_model.best_iteration} / 1000 rounds")

# %% [markdown]
# ### 3.3 LightGBM
#
# LightGBM's leaf-wise growth and histogram binning make it the fastest library
# on large datasets. The `num_leaves` parameter (not `max_depth`) is the primary
# complexity control.

# %%
lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=100,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)
lgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

lgb_metrics = {
    "valid": evaluate_model(y_valid, lgb_model.predict(X_valid)),
    "test": evaluate_model(y_test, lgb_model.predict(X_test)),
}
predictions["LightGBM"] = lgb_model.predict(X_test)
results.append(
    {"model": "LightGBM", **{f"{s}_{k}": v for s, m in lgb_metrics.items() for k, v in m.items()}}
)

print(
    f"LightGBM — Valid IC: {lgb_metrics['valid']['ic']:.4f}, Test IC: {lgb_metrics['test']['ic']:.4f}"
)

# %% [markdown]
# ### 3.4 CatBoost
#
# CatBoost uses symmetric (oblivious) trees where all nodes at a given depth share
# the same split — enabling fast bitwise inference. Note the API differences:
# `iterations` (not `n_estimators`), `depth` (not `max_depth`), `l2_leaf_reg`
# (not `reg_lambda`), and `colsample_bylevel` (per-level sampling, vs per-tree
# in XGBoost/LightGBM).

# %%
cb_model = cb.CatBoostRegressor(
    iterations=300,
    depth=4,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    subsample=0.8,
    colsample_bylevel=0.8,
    min_data_in_leaf=100,
    random_seed=42,
    verbose=False,
    train_dir="/tmp/catboost_info",
)
cb_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)

cb_metrics = {
    "valid": evaluate_model(y_valid, cb_model.predict(X_valid)),
    "test": evaluate_model(y_test, cb_model.predict(X_test)),
}
predictions["CatBoost"] = cb_model.predict(X_test)
results.append(
    {"model": "CatBoost", **{f"{s}_{k}": v for s, m in cb_metrics.items() for k, v in m.items()}}
)

print(
    f"CatBoost — Valid IC: {cb_metrics['valid']['ic']:.4f}, Test IC: {cb_metrics['test']['ic']:.4f}"
)

# %% [markdown]
# ## 4. Results Comparison

# %%
results_df = pl.DataFrame(results).sort("test_ic", descending=True)
results_df.select("model", "valid_ic", "test_ic", "valid_r2", "test_r2")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models = results_df["model"].to_list()
x = np.arange(len(models))

# Validation is the selection set (neutral); test is the sealed holdout (emphasis).
valid_color, test_color = COLORS["neutral"], COLORS["blue"]

# IC comparison
axes[0].bar(x - 0.15, results_df["valid_ic"].to_list(), 0.3, label="Validation", color=valid_color)
axes[0].bar(x + 0.15, results_df["test_ic"].to_list(), 0.3, label="Test", color=test_color)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=15)
axes[0].set_ylabel("Spearman rank IC")
axes[0].set_title("Information coefficient")
axes[0].legend()

# R² comparison
axes[1].bar(x - 0.15, results_df["valid_r2"].to_list(), 0.3, label="Validation", color=valid_color)
axes[1].bar(x + 0.15, results_df["test_r2"].to_list(), 0.3, label="Test", color=test_color)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=15)
axes[1].set_ylabel("Out-of-sample $R^2$")
axes[1].set_title("Out-of-sample $R^2$")
axes[1].legend()

fig.suptitle("All three GBMs edge out the Random Forest baseline on test IC")
plt.show()

# %% [markdown]
# **Interpretation**: All three GBM libraries outperform the Random Forest baseline
# (test IC 0.058-0.060 vs 0.056), confirming that sequential error correction adds
# value beyond variance reduction alone. The 0.004 IC gap between the best GBM and
# RF is modest but consistent across validation and test. The gap between GBM
# variants (0.002) is even smaller — consistent with §12.2's observation that
# hyperparameter quality matters more than library choice.

# %% [markdown]
# ## 5. Feature Importance
#
# Each library reports importance on its own native scale — sklearn RF and XGBoost
# return normalized gain (summing to 1), CatBoost returns prediction-value change
# (summing to 100), and LightGBM defaults to raw split counts (summing to the total
# number of splits). Averaging those raw vectors would let LightGBM's counts dominate,
# so we first rescale each library to a **share of its own total importance** before
# comparing. These native rankings are fast but biased toward high-cardinality features
# (§12.2); for robust attributions, see the SHAP analysis in §12.5.

# %%
_libs = ["rf", "xgb", "lgb", "cb"]
importances = (
    pl.DataFrame(
        {
            "feature": feature_cols,
            "rf": rf_model.feature_importances_,
            "xgb": xgb_model.feature_importances_,
            "lgb": lgb_model.feature_importances_,
            "cb": cb_model.feature_importances_,
        }
    )
    .with_columns(
        # Rescale each library to a share of total importance so the scales are comparable.
        [(pl.col(c) / pl.col(c).sum()).alias(c) for c in _libs]
    )
    .with_columns(avg=(pl.col("rf") + pl.col("xgb") + pl.col("lgb") + pl.col("cb")) / 4)
)

top10 = importances.sort("avg", descending=True).head(10)

# %%
fig, ax = plt.subplots(figsize=(10, 5))

features = top10["feature"].to_list()[::-1]
y_pos = np.arange(len(features))
width = 0.2

lib_colors = COLOR_CYCLER[:4]  # blue, amber, copper, green - four distinct categorical hues
for i, (lib, label) in enumerate([("rf", "RF"), ("xgb", "XGB"), ("lgb", "LGB"), ("cb", "Cat")]):
    vals = top10[lib].to_list()[::-1]
    ax.barh(y_pos + i * width, vals, width, label=label, color=lib_colors[i])

ax.set_yticks(y_pos + 1.5 * width)
ax.set_yticklabels(features)
ax.set_xlabel("Share of total importance (each library rescaled to sum to 1)")
ax.set_ylabel("Firm characteristic")
ax.set_title("Importance rankings diverge across ensemble methods")
ax.legend(loc="lower right")
plt.show()

# %% [markdown]
# **Interpretation**: The four methods disagree on which characteristics matter and how
# sharply. Random Forest and CatBoost concentrate importance in a handful of features —
# short-term reversal (`ST_REV`) alone carries a fifth to a quarter of each model's total,
# and their top five characteristics account for more than half — while XGBoost and
# LightGBM spread importance far more evenly across the cross-section. Part of this gap is
# an artifact: each library measures importance on a different native scale (impurity
# reduction, gain, split frequency, prediction-value change), so a feature that tops one
# ranking can sit mid-table in another. Only characteristics that rank highly across all
# four are robust signals; the disagreement is exactly why §12.5 turns to SHAP for
# model-agnostic attribution.

# %% [markdown]
# ## 6. Save Outputs

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

test_dates = test_df["timestamp"].to_list()

# Vectorized construction — one DataFrame per model, then concat
pred_frames = []
for model_name, preds in predictions.items():
    pred_frames.append(
        pl.DataFrame(
            {
                "timestamp": test_dates,
                "y_true": y_test.astype(np.float64),
                "y_pred": np.asarray(preds, dtype=np.float64),
                "model": model_name,
            }
        )
    )

predictions_df = pl.concat(pred_frames)
predictions_df.write_parquet(OUTPUT_DIR / "gbm_predictions.parquet")

results_df.write_csv(OUTPUT_DIR / "gbm_results.csv")
importances.write_parquet(OUTPUT_DIR / "gbm_feature_importances.parquet")

print(
    f"Saved {len(predictions_df):,} predictions, results summary, and feature importances to {display_path(OUTPUT_DIR)}"
)

# %% [markdown]
# ## Key Takeaways
#
# 1. **GBMs achieve higher test IC than Random Forest on this benchmark**: all
#    three GBM libraries land at test IC 0.058–0.060 versus 0.056 for Random
#    Forest — a 0.004 gap that is small but consistent across validation and
#    test. The gap is the empirical anchor for §12.1's argument that sequential
#    error correction adds signal beyond variance reduction.
#
# 2. **Library differences are small**: XGBoost, LightGBM, and CatBoost achieve
#    similar IC on this dataset. Hyperparameter configuration matters more than
#    library choice (explored in §12.4).
#
# 3. **Early stopping selects complexity automatically**: XGBoost stopped well below
#    the 1,000-round ceiling, showing that validation-based stopping is an effective
#    regularizer for noisy financial targets.
#
# 4. **Feature importance varies across libraries**: The top-ranked characteristics
#    shift between methods, and so does concentration — Random Forest and CatBoost
#    load heavily on a few features (their top five carry more than half the total)
#    while XGBoost and LightGBM spread importance more evenly. Because each library
#    uses a different native importance metric, these rankings are unstable — a known
#    limitation discussed in §12.2 that motivates SHAP-based attribution in §12.5.
#
# **Next**: See Ch14 for latent factor models (IPCA, RP-PCA, CAE, SDF-GAN)
# on this same benchmark dataset.
