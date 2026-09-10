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
# CatBoost) on the Chen-Pelger-Zhu (2020) firm characteristics dataset: a panel of
# stock-month observations with anonymized characteristics and predefined temporal
# splits. The load below prints how many rows and characteristics each split carries.
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
"""Ensemble Foundations - benchmark bagging against boosting on financial return prediction."""

import warnings

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb
from IPython.display import Markdown, display
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from data import load_firm_characteristics
from utils.paths import display_path, get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLOR_CYCLER, COLORS, show_with_alt

# LightGBM records synthetic feature names when fitted on an array with an eval_set,
# and sklearn then warns at every predict on an array that has none to compare. One
# message, not the category: the fit and the predictions are unaffected.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn.utils.validation",
)

OUTPUT_DIR = get_output_dir(12, "us_firm_characteristics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=["parameters"]
SEED = 42


# %%
set_global_seeds(SEED)
# %% [markdown]
# ## 1. Load Firm Characteristics Dataset
#
# The Chen-Pelger-Zhu dataset provides a clean benchmark: anonymized firm
# characteristics with predefined temporal splits, so the split boundaries are the
# dataset's own rather than a choice made here.

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


def _span(frame: pl.DataFrame) -> str:
    """Report a split's first and last year, so the boundaries come from the data."""
    return f"{frame['timestamp'].min():%Y}-{frame['timestamp'].max():%Y}"


for _name, _frame, _X in [
    ("Train", train_df, X_train),
    ("Valid", valid_df, X_valid),
    ("Test", test_df, X_test),
]:
    print(f"{_name}: {len(_X):,} observations, {_span(_frame)}")
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
# ### Random Forest, the bagging baseline
#
# Random Forests average independent trees, which reduces variance and leaves
# systematic bias where it is (§12.1).

# %%
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=100,
    max_features=0.3,
    n_jobs=-1,
    random_state=SEED,
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
    f"Random Forest, valid IC: {rf_metrics['valid']['ic']:.4f}, test IC: {rf_metrics['test']['ic']:.4f}"
)

# %% [markdown]
# ### XGBoost
#
# XGBoost adds L1/L2 regularization on leaf weights and uses second-order
# gradient approximations. Early stopping selects the iteration count: the model
# stops when validation loss plateaus.

# %%
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,  # a ceiling; early stopping picks the count that runs
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=100,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    early_stopping_rounds=50,
    random_state=SEED,
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
    f"XGBoost, valid IC: {xgb_metrics['valid']['ic']:.4f}, test IC: {xgb_metrics['test']['ic']:.4f}"
)
print(f"  Early stopping at {xgb_model.best_iteration} / 1000 rounds")

# %% [markdown]
# ### LightGBM
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
    random_state=SEED,
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
    f"LightGBM, valid IC: {lgb_metrics['valid']['ic']:.4f}, test IC: {lgb_metrics['test']['ic']:.4f}"
)

# %% [markdown]
# ### CatBoost
#
# CatBoost uses symmetric (oblivious) trees where all nodes at a given depth share
# the same split, which is what lets inference be a bitwise operation. Note the API
# differences:
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
    random_seed=SEED,
    verbose=False,
    train_dir=str(OUTPUT_DIR / "catboost_info"),
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
    f"CatBoost, valid IC: {cb_metrics['valid']['ic']:.4f}, test IC: {cb_metrics['test']['ic']:.4f}"
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

# Validation is where the models were selected, so it is drawn in the neutral colour
# and the untouched test split takes the emphasis.
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

fig.suptitle("Rank IC and out-of-sample $R^2$ by model, validation and test")
show_with_alt(
    fig,
    "Two panels of grouped bars, one bar pair per model. Left: Spearman rank IC on the "
    "validation split beside the test split. Right: out-of-sample $R^2$ for the same "
    "splits. The models are ordered by test IC, and the validation bar is taller than "
    "the test bar for every model in both panels.",
)

# %% tags=["results"]
_rf_ic = results_df.filter(pl.col("model") == "Random Forest")["test_ic"].item()
_gbm = results_df.filter(pl.col("model") != "Random Forest")
_drop = (results_df["valid_ic"] - results_df["test_ic"]).max()
display(
    Markdown(
        "- Test IC, high to low: "
        + ", ".join(
            f"{row['model']} {row['test_ic']:.4f}"
            for row in results_df.sort("test_ic", descending=True).iter_rows(named=True)
        )
        + "\n- Test $R^2$, high to low: "
        + ", ".join(
            f"{row['model']} {row['test_r2']:.4f}"
            for row in results_df.sort("test_r2", descending=True).iter_rows(named=True)
        )
        + f"\n- Widest test-IC gap between a boosted model and the Random Forest: "
        f"{_gbm['test_ic'].max() - _rf_ic:+.4f}. Spread across the three boosted "
        f"models: {_gbm['test_ic'].max() - _gbm['test_ic'].min():.4f}. Largest "
        f"validation-to-test drop within one model: {_drop:.4f}."
    )
)

# %% [markdown]
# **What to read off it.** These bars are one split with no interval attached, so the
# ordering is what this run produced rather than a measurement of which method is
# better on data of this kind. The results cell prints two numbers to hold against
# each other: how far apart the models are on test, and how far a single model moves
# between validation and test. Where the second is the larger, a between-model
# ordering is smaller than the movement one model shows across splits, and reading a
# ranking off it is reading that movement.
#
# The two panels also need not order the models the same way. Rank IC scores how well
# a model orders the cross-section, $R^2$ scores how close its predictions sit to the
# realized return, and a model can do better on one and worse on the other. A
# conclusion that holds in one panel and not the other is a conclusion about the
# panel.

# %% [markdown]
# ## 5. Feature Importance
#
# Each library reports importance on its own native scale: sklearn RF and XGBoost
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
ax.set_title("Top firm characteristics by importance share, four libraries")
ax.legend(loc="lower right")
show_with_alt(
    fig,
    "Grouped horizontal bars, one group per firm characteristic and one bar per "
    "library, showing each library's share of its own total importance. The "
    "characteristics are ordered by the average share across the four libraries.",
)

# %% tags=["results"]
_shares = {
    label: importances.select(pl.col(lib).sort(descending=True))[lib]
    for lib, label in [("rf", "RF"), ("xgb", "XGB"), ("lgb", "LGB"), ("cb", "Cat")]
}
_largest = {label: float(share[0]) for label, share in _shares.items()}
_top_five = {label: float(share.head(5).sum()) for label, share in _shares.items()}
display(
    Markdown(
        "- Share of a library's own importance carried by its single largest "
        "characteristic: "
        + ", ".join(f"{label} {value:.0%}" for label, value in _largest.items())
        + "\n- Share carried by its top five: "
        + ", ".join(f"{label} {value:.0%}" for label, value in _top_five.items())
    )
)

# %% [markdown]
# **What to read off it.** Two questions live in this chart and only one of them is
# answerable here. How concentrated each library's importance is, and which
# characteristics it concentrates on, are both visible. Whether a disagreement between
# two libraries is about the data or about the metric is not: each library measures
# importance on its own native scale (impurity reduction, gain, split frequency,
# prediction-value change), and the rescaling above makes the shares comparable in size
# without making the definitions the same. So a characteristic that tops one ranking and
# sits mid-table in another has told you nothing yet about its predictive role.
#
# §12.2 sets out why these native rankings are unstable in the first place: gain is
# biased toward features with more candidate splits, split counts toward continuous
# features, and a different seed can reorder correlated features that substitute for one
# another. That is the argument for the SHAP attributions in §12.5, which are defined the
# same way for every model.

# %% [markdown]
# ## 6. Save Outputs

# %%
test_dates = test_df["timestamp"].to_list()

# One frame per model, then concat.
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
# 1. **The boosted models and the bagged baseline are separated by less than the
#    validation-to-test drop.** The results cell above prints both numbers. A gap that
#    small, measured on one split with no interval, is a fact about this run; it is the
#    kind of evidence §12.1's argument predicts, not a test of it.
#
# 2. **Rank IC and $R^2$ can order the same models differently.** One scores the
#    ordering of predictions across the cross-section, the other their distance from
#    the realized return. Deciding which matters is a decision about the strategy the
#    predictions feed, and it has to be made before the comparison, not after seeing it.
#
# 3. **Early stopping selects complexity from data rather than from a guess.** XGBoost
#    stopped well below its ceiling; the printed round count says where. On a target
#    with this little signal, that is the difference between a regularizer and a
#    hyperparameter someone had to pick.
#
# 4. **Native importance rankings are not comparable across libraries, even rescaled.**
#    The shares above are on one axis because each was divided by its own total, not
#    because the four libraries measure the same thing. §12.5's SHAP attributions are
#    the version of this question that has one definition for every model.
#
# **Next**: Ch14 fits latent factor models (IPCA, RP-PCA, CAE, SDF-GAN) on this same
# benchmark, which is what makes the comparison across chapters a comparison.
