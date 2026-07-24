# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     formats: py:percent,ipynb
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
# # SHAP Interpretability for Linear Models
#
# **Docker image**: `ml4t`
#
# **Purpose**: use SHAP (SHapley Additive exPlanations) to interpret a Ridge
# regression model trained on the ETF panel. For linear models, SHAP values
# decompose exactly into $\phi_j = \beta_j \cdot (x_j - \bar{x}_j)$, making
# attributions transparent and verifiable.
#
# We demonstrate four layers of interpretation:
#
# 1. **Global feature importance** — which features matter most across all predictions.
# 2. **Individual forecast explanation** — how features combine for a single prediction.
# 3. **Decision-relevant analysis** — SHAP profiles for *right* vs *wrong*
#    high-magnitude predictions.
# 4. **Stability** — cross-fold importance trajectories + within-fold bootstrap
#    confidence bands on the top features' mean $|\phi_j|$.
#
# **Learning objectives**
#
# - Compute and verify exact linear SHAP values for Ridge
# - Explain a single forecast via SHAP waterfall decomposition
# - Compare feature drivers for correct vs incorrect high-conviction predictions
# - Build a SHAP stability chart and bootstrap confidence bands within a fold
#
# **Book reference**: Section 11.4 — Inside the Black Box: Model
# Interpretability with SHAP.
#
# **Prerequisites**
#
# - Ch7 21-day forward return labels at `case_studies/etfs/labels/fwd_ret_21d.parquet`
# - Ch8 ETF features at `case_studies/etfs/features/financial.parquet`
# - `02_regularization_paths` (Ridge / LASSO context)
#
# **Downstream**: Ch12 extends to tree-based models via `TreeExplainer`; Ch13
# applies KernelSHAP to deep learning.

# %% [markdown] tags=[]
# ## Setup

# %% tags=[]
"""SHAP Interpretability for Linear Models — decompose Ridge predictions into per-feature attributions."""

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils.cv_splits import generate_cv_splits
from utils.paths import display_path, get_case_study_dir, get_chapter_dir, get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
SEED = 42
MAX_SYMBOLS = 0
MAX_CV_FOLDS = 0
RETRAIN = False
ARTIFACT_TAG = ""

# %% tags=[]
RANDOM_SEED = SEED
set_global_seeds(SEED)

# %% [markdown] tags=[]
# ## Load Features and Labels
#
# We use the same ETF features (Ch8) and 21-day forward returns (Ch7) as
# `02_regularization_paths`. The features matrix has 57 columns spanning
# momentum, volatility, volume, and cross-asset families.

# %% tags=[]
CASE_DIR = get_case_study_dir("etfs")
FEATURES_PATH = CASE_DIR / "features" / "financial.parquet"
LABELS_PATH = CASE_DIR / "labels" / "fwd_ret_21d.parquet"

assert FEATURES_PATH.exists(), (
    f"Features not found: {FEATURES_PATH}\nRun the Ch8 ETF features notebook first."
)
assert LABELS_PATH.exists(), (
    f"Labels not found: {LABELS_PATH}\nRun the Ch7 ETF labels notebook first."
)

features_df = pl.read_parquet(FEATURES_PATH).with_columns(pl.col("timestamp").cast(pl.Date))
labels_df = pl.read_parquet(LABELS_PATH).with_columns(pl.col("timestamp").cast(pl.Date))

# %% tags=[]
TARGET_COL = "fwd_ret_21d"
ASSET_COL = "symbol"

df = features_df.join(labels_df, on=["timestamp", ASSET_COL], how="inner")

META_COLS = {"timestamp", ASSET_COL, TARGET_COL}
FEATURE_COLS = sorted(c for c in df.columns if c not in META_COLS)

# Drop features that are entirely null (can happen with reduced test universes)
all_null = [c for c in FEATURE_COLS if df[c].null_count() == df.height]
if all_null:
    print(f"Dropping {len(all_null)} all-null features: {all_null}")
    df = df.drop(all_null)
    FEATURE_COLS = [c for c in FEATURE_COLS if c not in all_null]

# Replace inf/NaN with null, then drop all nulls
df = df.with_columns(
    [
        pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in FEATURE_COLS
    ]
)
df = df.drop_nulls(subset=FEATURE_COLS + [TARGET_COL]).sort(["timestamp", ASSET_COL])

if MAX_SYMBOLS > 0:
    assets = df[ASSET_COL].unique().sort().head(MAX_SYMBOLS).to_list()
    df = df.filter(pl.col(ASSET_COL).is_in(assets))

print(f"Shape: {df.height:,} rows x {len(FEATURE_COLS)} features")
print(f"Assets: {df[ASSET_COL].n_unique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# %% [markdown] tags=[]
# ## Walk-Forward CV: Select Last Fold
#
# SHAP analysis requires a single trained model and its corresponding test set.
# We use the **last fold** of the walk-forward protocol (most recent test period),
# which gives us the model trained on the most data and the most realistic
# out-of-sample evaluation.

# %% tags=[]
splits = generate_cv_splits(df, case_study_id="etfs", label_buffer="21D", date_col="timestamp")

features_array = df.select(FEATURE_COLS).to_numpy()
target_array = df[TARGET_COL].to_numpy()
dates_np = df["timestamp"].to_numpy()
assets_np = df[ASSET_COL].to_numpy()

cv_splits = []
for s in splits:
    tr_start, tr_end = np.datetime64(s["train_start"]), np.datetime64(s["train_end"])
    te_start, te_end = np.datetime64(s["val_start"]), np.datetime64(s["val_end"])
    train_idx = np.where((dates_np >= tr_start) & (dates_np <= tr_end))[0]
    test_idx = np.where((dates_np >= te_start) & (dates_np <= te_end))[0]
    cv_splits.append((train_idx, test_idx))

if MAX_CV_FOLDS > 0 and len(cv_splits) > MAX_CV_FOLDS:
    cv_splits = cv_splits[:MAX_CV_FOLDS]
    splits = splits[:MAX_CV_FOLDS]

# Use last fold
train_idx, test_idx = cv_splits[-1]
last_split = splits[-1]

print(f"Last fold (fold {len(cv_splits)}):")
print(
    f"  Train: {last_split['train_start']} to {last_split['train_end']} ({len(train_idx):,} samples)"
)
print(f"  Test:  {last_split['val_start']} to {last_split['val_end']} ({len(test_idx):,} samples)")

# %% [markdown] tags=[]
# ## Train Ridge Model and Compute SHAP Values
#
# We fit Ridge regression ($\alpha = 1.0$, a moderate default after standardization)
# on the training fold and compute SHAP values on the test set.
# `shap.LinearExplainer` gives **exact** SHAP values for linear models — no
# approximation, no sampling.
#
# **SHAP API pattern** (v0.50+): Create a masker to describe the background
# distribution, pass it to the explainer, then call the explainer on the test
# data to get an `Explanation` object containing `.values`, `.base_values`,
# and `.data` in a single structure.

# %% tags=[]
MODELS_DIR = get_chapter_dir(11) / "models" / "05_shap_analysis"
is_reduced_run = MAX_SYMBOLS > 0 or MAX_CV_FOLDS > 0
if is_reduced_run and not ARTIFACT_TAG:
    ARTIFACT_TAG = "_fast"

RESULTS_PATH = MODELS_DIR / f"shap_results{ARTIFACT_TAG}.joblib"
NEED_TRAINING = RETRAIN or not RESULTS_PATH.exists()

# %% tags=[]
if NEED_TRAINING:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features_array[train_idx])
    X_test = scaler.transform(features_array[test_idx])
    y_train = target_array[train_idx]
    y_test = target_array[test_idx]

    model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Ridge trained on {len(y_train):,} samples, predicting {len(y_test):,}.")

# %% [markdown] tags=[]
# ### SHAP Computation (Last Fold)

# %% tags=[]
if NEED_TRAINING:
    # Independent masker assumes feature independence → SHAP = coef * (x - mean)
    masker = shap.maskers.Independent(X_train)
    explainer = shap.LinearExplainer(model, masker)
    explanation = explainer(X_test)
    shap_values = explanation.values
    expected_value = float(explanation.base_values[0])
    print(f"SHAP values computed: {shap_values.shape}")

# %% [markdown] tags=[]
# ### SHAP Stability: All Folds
#
# Train Ridge on every fold and compute mean |SHAP| per feature. This reveals
# whether the model learns consistent feature importance across time periods.

# %% tags=[]
if NEED_TRAINING:
    fold_importance = {}
    for fold_i, (tr_idx, te_idx) in enumerate(cv_splits):
        sc_f = StandardScaler()
        X_tr_f = sc_f.fit_transform(features_array[tr_idx])
        X_te_f = sc_f.transform(features_array[te_idx])
        m_f = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        m_f.fit(X_tr_f, target_array[tr_idx])
        exp_f = shap.LinearExplainer(m_f, shap.maskers.Independent(X_tr_f))
        fold_importance[fold_i + 1] = np.mean(np.abs(exp_f(X_te_f).values), axis=0)
    print(f"Computed SHAP stability across {len(fold_importance)} folds.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "shap_values": shap_values,
            "expected_value": expected_value,
            "fold_importance": fold_importance,
        },
        RESULTS_PATH,
    )
    print("Cached results saved.")

# %% [markdown] tags=[]
# ### Load Cached Results
#
# When pre-computed artifacts exist, restore all arrays and reconstruct the
# SHAP `Explanation` object (joblib caches raw arrays, not SHAP objects).

# %% tags=[]
if not NEED_TRAINING:
    _cached = joblib.load(RESULTS_PATH)
    model = _cached["model"]
    scaler = _cached["scaler"]
    X_train = _cached["X_train"]
    X_test = _cached["X_test"]
    y_train = _cached["y_train"]
    y_test = _cached["y_test"]
    y_pred = _cached["y_pred"]
    shap_values = _cached["shap_values"]
    expected_value = _cached["expected_value"]
    fold_importance = _cached.get("fold_importance")
    del _cached

    # Reconstruct Explanation — joblib caches arrays, not SHAP objects
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(shap_values), expected_value),
        data=X_test,
        feature_names=FEATURE_COLS,
    )

# %% [markdown] tags=[]
# ### Recompute Stale Artifacts
#
# The cache is stale if it was produced with fewer folds (e.g. a TEST run)
# or against a different data vintage — when the feature/label tables are
# regenerated the test fold's row count changes, so a cached `y_pred` no
# longer aligns with the current `test_idx`. Either condition triggers a
# refit on the full split set and a recompute of per-fold SHAP importance.

# %% tags=[]
_cache_fold_stale = fold_importance is not None and len(fold_importance) < len(cv_splits)
_cache_data_stale = len(y_pred) != len(test_idx) or X_test.shape[0] != len(test_idx)
if not NEED_TRAINING and (_cache_fold_stale or _cache_data_stale):
    print(
        "Cached SHAP artifacts are stale (fold count or data vintage changed); recomputing full artifacts."
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features_array[train_idx])
    X_test = scaler.transform(features_array[test_idx])
    y_train = target_array[train_idx]
    y_test = target_array[test_idx]

    model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    masker = shap.maskers.Independent(X_train)
    explainer = shap.LinearExplainer(model, masker)
    explanation = explainer(X_test)
    shap_values = explanation.values
    expected_value = float(explanation.base_values[0])

    fold_importance = {}
    for fold_i, (tr_idx, te_idx) in enumerate(cv_splits):
        sc_f = StandardScaler()
        X_tr_f = sc_f.fit_transform(features_array[tr_idx])
        X_te_f = sc_f.transform(features_array[te_idx])
        m_f = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        m_f.fit(X_tr_f, target_array[tr_idx])
        exp_f = shap.LinearExplainer(m_f, shap.maskers.Independent(X_tr_f))
        fold_importance[fold_i + 1] = np.mean(np.abs(exp_f(X_te_f).values), axis=0)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "shap_values": shap_values,
            "expected_value": expected_value,
            "fold_importance": fold_importance,
        },
        RESULTS_PATH,
    )
    print("Cached results updated.")

# %% [markdown] tags=[]
# ### Persist SHAP Arrays for Downstream Figures
#
# Writes the raw arrays needed to reproduce the SHAP figures
# (`shap_values`, `X_test`, `y_pred`, `y_test`, `expected_value`,
# `feature_names`, and the per-fold importance matrix) to
# `output/05_shap_analysis/shap_arrays.npz`. Downstream consumers can
# load these with `np.load` without depending on the SHAP toolchain.

# %% tags=[]
SHAP_ARRAYS_PATH = get_output_dir(11, "05_shap_analysis") / "shap_arrays.npz"
fold_ids_sorted = sorted(fold_importance) if fold_importance is not None else []
fold_importance_matrix = (
    np.array([fold_importance[k] for k in fold_ids_sorted], dtype=np.float32)
    if fold_ids_sorted
    else np.zeros((0, len(FEATURE_COLS)), dtype=np.float32)
)
np.savez(
    SHAP_ARRAYS_PATH,
    shap_values=shap_values.astype(np.float32),
    X_test=X_test.astype(np.float32),
    y_pred=y_pred.astype(np.float32),
    y_test=y_test.astype(np.float32),
    expected_value=np.float32(expected_value),
    feature_names=np.asarray(FEATURE_COLS, dtype=np.str_),
    fold_ids=np.array(fold_ids_sorted, dtype=np.int32),
    fold_importance=fold_importance_matrix,
)
print(f"Persisted SHAP arrays to {display_path(SHAP_ARRAYS_PATH)}")

# %% tags=[]
dates_test = dates_np[test_idx]
symbols_test = assets_np[test_idx]

pred_df = pl.DataFrame({"timestamp": dates_test, "symbol": symbols_test, "prediction": y_pred})
ret_df = pl.DataFrame({"timestamp": dates_test, "symbol": symbols_test, "forward_return": y_test})
ic_per_date = cross_sectional_ic_series(
    pred_df,
    ret_df,
    pred_col="prediction",
    ret_col="forward_return",
    date_col="timestamp",
    entity_col="symbol",
)
ic_clean = ic_per_date.drop_nulls("ic")
ic = float(ic_clean["ic"].mean())
ic_std = float(ic_clean["ic"].std())
n_periods = ic_clean.height
ic_t = ic / (ic_std / np.sqrt(n_periods)) if ic_std > 0 else float("nan")

print(f"Test set: {len(y_test):,} samples ({n_periods} dates)")
print(f"IC (cross-sectional Spearman, mean across dates): {ic:.4f}")
print(f"IC t-stat: {ic_t:.2f}")
print(f"SHAP values shape: {shap_values.shape}")

# %% [markdown] tags=[]
# ## Verify: SHAP Matches Coefficient Attribution
#
# For a linear model, each SHAP value should closely match the coefficient
# times the feature's deviation from its background mean:
#
# $$\phi_j^{(i)} = \beta_j \cdot \bigl(x_j^{(i)} - \bar{x}_j\bigr)$$
#
# This makes linear SHAP transparent — every attribution can be checked
# by hand. The SHAP library's internal handling of background distributions
# may introduce tiny numerical differences, but the correspondence is
# near-perfect. Ch12 contrasts this with tree-based models where SHAP
# requires algorithmic computation.

# %% tags=[]
coef_attribution = model.coef_ * (X_test - X_train.mean(axis=0))

corr = np.corrcoef(shap_values.ravel(), coef_attribution.ravel())[0, 1]
max_diff = np.max(np.abs(shap_values - coef_attribution))
mean_abs_diff = np.mean(np.abs(shap_values - coef_attribution))

print(f"Correlation (SHAP vs coef*deviation): {corr:.6f}")
print(f"Mean |difference|: {mean_abs_diff:.2e}")
print(f"Max |difference|:  {max_diff:.2e}")
print("\nLinear SHAP is transparent: attributions match the closed-form solution.")

# %% [markdown] tags=[]
# ## Global Feature Importance
#
# The beeswarm plot ranks features by mean $|\phi_j|$ across all test samples.
# Each dot is one prediction: its x-position shows the SHAP value (impact on
# predicted return), and color shows the feature value (red = high, blue = low).
# This reveals both *which* features matter and *how* they affect predictions.

# %% tags=[]
# Ensure feature names are set (numpy arrays don't carry column names)
explanation.feature_names = FEATURE_COLS

# %% tags=[]
shap.plots.beeswarm(explanation, max_display=20, show=False)
plt.title("Global Feature Importance (SHAP)")
plt.show()

# %% [markdown] tags=[]
# ### Mean |SHAP| by Feature
#
# The bar chart provides a simpler summary — mean absolute SHAP value per
# feature, equivalent to the average *magnitude* of each feature's contribution
# to predictions.

# %% tags=[]
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
importance = pl.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs_shap}).sort(
    "mean_abs_shap", descending=True
)
importance.head(15)

# %% [markdown] tags=[]
# ### Sign Consistency Check
#
# The first layer of economic narrative building: do learned coefficients have
# the expected sign? A model where momentum has a negative coefficient or
# volatility a positive one is likely fitting noise.

# %% tags=[]
# Expected signs for key features (based on factor theory)
EXPECTED_SIGNS = {
    "ret_21d": "+",  # short-term momentum
    "ret_63d": "+",  # medium-term momentum
    "ret_126d": "+",  # long-term momentum
    "vol_21d": "-",  # short-term volatility (risk)
    "vol_63d": "-",  # medium-term volatility
    "obv_zscore_63d": "+",  # volume confirms trend
}

coefs = dict(zip(FEATURE_COLS, model.coef_, strict=False))
sign_check = pl.DataFrame(
    [
        {
            "feature": feat,
            "expected": sign,
            "actual": "+" if coefs.get(feat, 0) >= 0 else "-",
            "coefficient": coefs.get(feat, 0),
            "consistent": (sign == "+") == (coefs.get(feat, 0) >= 0),
        }
        for feat, sign in EXPECTED_SIGNS.items()
        if feat in coefs
    ]
)
sign_check

# %% [markdown] tags=[]
# Sign violations are not necessarily wrong — Ridge splits credit across
# correlated features, so a feature may flip sign if a correlated partner
# absorbs its effect. But persistent violations across folds warrant investigation.

# %% [markdown] tags=[]
# ### Feature Dependence
#
# For a linear model, the SHAP dependence plot is linear by construction
# ($\phi_j = \beta_j \cdot (x_j - \bar{x}_j)$). We show it here to establish
# the pattern — Ch12 extends this to tree models where dependence reveals
# threshold effects and non-linear interactions.

# %% tags=[]
# Skip degenerate features (near-constant on the test fold), which can collapse
# dependence plots into a single point.
dep_feature = None
for feat in importance["feature"].to_list():
    idx = FEATURE_COLS.index(feat)
    if np.std(X_test[:, idx]) > 1e-8 and np.std(shap_values[:, idx]) > 1e-12:
        dep_feature = feat
        break

if dep_feature is None:
    dep_feature = importance["feature"][0]

top_idx_dep = FEATURE_COLS.index(dep_feature)
color_feature = importance["feature"][1] if len(importance) > 1 else dep_feature
color_idx = FEATURE_COLS.index(color_feature)

plt.figure(figsize=(8, 5))
shap.plots.scatter(
    explanation[:, top_idx_dep],
    color=explanation[:, color_idx],
    show=False,
)
ax = plt.gca()
ax.set_title(f"SHAP Dependence: {dep_feature} (color: {color_feature})")
plt.show()

# %% [markdown] tags=[]
# ## Individual Forecast Explanation
#
# SHAP waterfall plots decompose a *single* prediction into feature contributions.
# Starting from the expected value (population mean prediction), each feature
# pushes the prediction up (red) or down (blue) by its SHAP value.
#
# We show two examples: one where the model made a large positive prediction
# that turned out correct, and one where a large prediction was wrong.

# %% tags=[]
# Find a high-conviction correct prediction
signed_error = y_pred - y_test
correct_sign = np.sign(y_pred) == np.sign(y_test)

# High positive prediction that was correct
candidates_right = np.where(correct_sign & (y_pred > np.percentile(y_pred, 95)))[0]
example_right = candidates_right[0]

# High positive prediction that was wrong
candidates_wrong = np.where(~correct_sign & (y_pred > np.percentile(y_pred, 95)))[0]
example_wrong = candidates_wrong[0] if len(candidates_wrong) > 0 else np.where(~correct_sign)[0][0]

# %% tags=[]
print("--- Correct high-conviction prediction ---")
print(f"  Predicted: {y_pred[example_right]:+.4f}  Actual: {y_test[example_right]:+.4f}")
print(f"  Date: {dates_np[test_idx[example_right]]}  Asset: {assets_np[test_idx[example_right]]}")

fig, ax = plt.subplots(figsize=(8, 6))
shap.plots.waterfall(explanation[example_right], max_display=12, show=False)
plt.title("Correct High-Conviction Prediction")
plt.show()

# %% tags=[]
print("--- Incorrect high-conviction prediction ---")
print(f"  Predicted: {y_pred[example_wrong]:+.4f}  Actual: {y_test[example_wrong]:+.4f}")
print(f"  Date: {dates_np[test_idx[example_wrong]]}  Asset: {assets_np[test_idx[example_wrong]]}")

fig, ax = plt.subplots(figsize=(8, 6))
shap.plots.waterfall(explanation[example_wrong], max_display=12, show=False)
plt.title("Incorrect High-Conviction Prediction")
plt.show()

# %% [markdown] tags=[]
# The waterfall plots show the same model making confident calls for different
# reasons. Comparing the feature attributions gives a first hint at what
# distinguishes correct from incorrect predictions — the systematic analysis
# below makes this rigorous.

# %% [markdown] tags=[]
# ## Decision-Relevant Predictions: Right vs Wrong
#
# Not all predictions are equally important. In a portfolio context, predicted
# return magnitude maps directly to position size ($w_i \propto \hat{y}_i$), so
# predictions with large $|\hat{y}|$ are **decision-relevant** — they drive the
# biggest bets and have the greatest impact on PnL.
#
# We isolate the top 20% of predictions by $|\hat{y}|$ and split them into:
#
# - **Right**: $\text{sign}(\hat{y}) = \text{sign}(y)$ — the model was
#   confidently correct
# - **Wrong**: $\text{sign}(\hat{y}) \neq \text{sign}(y)$ — the model was
#   confidently incorrect
#
# Comparing their SHAP profiles reveals which features *systematically* mislead
# the model at the moments that matter most.

# %% tags=[]
conviction = np.abs(y_pred)
threshold = np.percentile(conviction, 80)
high_mag_mask = conviction >= threshold

correct_mask = np.sign(y_pred) == np.sign(y_test)
right_mask = high_mag_mask & correct_mask
wrong_mask = high_mag_mask & ~correct_mask

n_high = high_mag_mask.sum()
n_right = right_mask.sum()
n_wrong = wrong_mask.sum()

print(f"High-magnitude predictions (top 20%): {n_high:,}")
print(f"  Right (correct sign): {n_right:,} ({n_right / n_high:.0%})")
print(f"  Wrong (incorrect sign): {n_wrong:,} ({n_wrong / n_high:.0%})")
print("\nMean |predicted return|:")
print(f"  Right: {np.mean(np.abs(y_pred[right_mask])):.4f}")
print(f"  Wrong: {np.mean(np.abs(y_pred[wrong_mask])):.4f}")
print("Mean actual return:")
print(f"  Right: {np.mean(y_test[right_mask]):+.4f}")
print(f"  Wrong: {np.mean(y_test[wrong_mask]):+.4f}")

# %% [markdown] tags=[]
# **Frame the result honestly**: at the 80th-percentile conviction band the
# model is wrong $58\%$ of the time — its biggest bets are systematically
# *worse* than a coin flip. The right-vs-wrong SHAP comparison below diagnoses
# *why*; it is not a sign that the model has skill at high conviction. The
# Ridge model on this fold has IC $\approx 0.018$ ($t \approx 1.17$), which
# is barely above zero and consistent with the high-conviction breakdown.
# Treat the analysis as a model-improvement diagnostic, not a validation of
# the current model.

# %% [markdown] tags=[]
# ### SHAP Profiles: Right vs Wrong
#
# We compute the mean |SHAP| per feature for each group. Features where the
# "wrong" group has disproportionately high SHAP are candidates for improvement:
# the model relied on them heavily, but they led it astray.

# %% tags=[]
shap_right = shap_values[right_mask]
shap_wrong = shap_values[wrong_mask]

mean_shap_right = np.mean(np.abs(shap_right), axis=0)
mean_shap_wrong = np.mean(np.abs(shap_wrong), axis=0)

comparison = pl.DataFrame(
    {
        "feature": FEATURE_COLS,
        "shap_all": mean_abs_shap,  # Reuse from global importance above
        "shap_right": mean_shap_right,
        "shap_wrong": mean_shap_wrong,
        "wrong_vs_right": mean_shap_wrong / np.maximum(mean_shap_right, 1e-10),
    }
).sort("wrong_vs_right", descending=True)

print(
    "Features ranked by wrong/right SHAP ratio (high ratio = feature drives more of the wrong predictions):"
)
comparison.head(15)

# %% [markdown] tags=[]
# Features with wrong/right ratios well above 1.0 are the model's weak points
# at high conviction — it relied on them heavily when making its biggest
# mistakes. If volume or cross-asset features dominate the top of this list,
# they may be unreliable as standalone predictors and could benefit from
# interaction terms or non-linear modeling in Ch12.

# %% [markdown] tags=[]
# ### Visualizing the Difference
#
# The bar chart compares mean |SHAP| for the top features, side by side for
# right (correct) and wrong (incorrect) high-magnitude predictions.

# %% tags=[]
top_features = comparison.head(15)["feature"].to_list()
top_idx = [FEATURE_COLS.index(f) for f in top_features]

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(top_features))
width = 0.35

right_vals = [mean_shap_right[i] for i in top_idx]
wrong_vals = [mean_shap_wrong[i] for i in top_idx]

ax.barh(x + width / 2, right_vals, width, label="Right (correct)", alpha=0.8)
ax.barh(x - width / 2, wrong_vals, width, label="Wrong (incorrect)", alpha=0.8)

ax.set_yticks(x)
ax.set_yticklabels(top_features, fontsize=9)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Feature Importance: Right vs Wrong High-Magnitude Predictions")
ax.legend(loc="lower right")
ax.invert_yaxis()
plt.show()

# %% [markdown] tags=[]
# **Interpretation**: Features where the "wrong" bar substantially exceeds the "right"
# bar are the model's weak points at high conviction — it relied on them heavily when
# making its biggest mistakes. These features are candidates for re-engineering
# (e.g., adding interaction terms) or non-linear modeling in Ch12.

# %% [markdown] tags=[]
# ### Signed SHAP: Direction of Influence
#
# Mean |SHAP| shows *magnitude*, but the *sign* matters too. A feature that
# consistently pushes predictions in the wrong direction is more problematic
# than one that adds noise symmetrically.
#
# We compute mean signed SHAP for each group to see directional bias.

# %% tags=[]
mean_signed_right = np.mean(shap_right, axis=0)
mean_signed_wrong = np.mean(shap_wrong, axis=0)

signed_comparison = pl.DataFrame(
    {
        "feature": FEATURE_COLS,
        "signed_shap_right": mean_signed_right,
        "signed_shap_wrong": mean_signed_wrong,
        "direction_diff": mean_signed_wrong - mean_signed_right,
    }
).sort("direction_diff", descending=True)

print("Signed SHAP: features with largest direction difference (wrong - right):")
print(
    "Positive = feature pushed wrong predictions UP more than right ones; negative = the reverse."
)
signed_comparison.head(10)

# %% tags=[]
signed_comparison.tail(10)

# %% [markdown] tags=[]
# **Interpretation**: Large positive direction differences mean the feature pushed wrong
# predictions upward more than right ones — a systematic directional bias at the moments
# that matter most. Features with large negative differences show the opposite pattern.
# Both directions suggest the feature's signal is unreliable under high conviction.

# %% [markdown] tags=[]
# ### Concentration Risk
#
# When a single feature dominates the total SHAP attribution, the prediction
# hinges on one input — a fragile basis for trading. We flag predictions where
# any feature accounts for more than 60% of the total |SHAP|.

# %% tags=[]
total_abs = np.abs(shap_values).sum(axis=1)
max_abs = np.abs(shap_values).max(axis=1)
max_frac = max_abs / np.maximum(total_abs, 1e-10)
concentrated = max_frac > 0.60

print(
    f"Predictions with >60% concentration: {concentrated.sum():,} / {len(max_frac):,} ({concentrated.mean():.1%})"
)
print(f"Mean max-feature fraction: {max_frac.mean():.1%}")
print(f"95th percentile: {np.percentile(max_frac, 95):.1%}")

# %% [markdown] tags=[]
# On this fold the 60% concentration threshold flags zero predictions —
# the typical max-feature share is $\approx 20\%$ and the 95th percentile sits
# at $\approx 33\%$. The model is not relying on a single feature to make
# any given call; attribution is spread across the feature set. A higher
# threshold (or a different aggregation, such as the share of the top-three
# features) would be needed to surface concentration risk on this dataset.
# In a fold dominated by a regime change, the same diagnostic can flip and
# flag many predictions; checking it per fold is part of the standing
# pipeline rather than a one-time exercise.

# %% [markdown] tags=[]
# ### Example Waterfall: Typical Right vs Typical Wrong
#
# To make the comparison concrete, we pick the *median* high-magnitude
# prediction from each group and show its SHAP waterfall.

# %% tags=[]
# Median by |predicted return| within each group
right_indices = np.where(right_mask)[0]
wrong_indices = np.where(wrong_mask)[0]

right_by_mag = right_indices[np.argsort(np.abs(y_pred[right_indices]))]
wrong_by_mag = wrong_indices[np.argsort(np.abs(y_pred[wrong_indices]))]

median_right = right_by_mag[len(right_by_mag) // 2]
median_wrong = wrong_by_mag[len(wrong_by_mag) // 2]

# %% tags=[]
print("--- Median high-magnitude RIGHT prediction ---")
print(f"  Predicted: {y_pred[median_right]:+.4f}  Actual: {y_test[median_right]:+.4f}")
print(f"  Asset: {assets_np[test_idx[median_right]]}")

fig, ax = plt.subplots(figsize=(8, 6))
shap.plots.waterfall(explanation[median_right], max_display=12, show=False)
plt.title("Median Right: High-Magnitude Correct Prediction")
plt.show()

# %% tags=[]
print("--- Median high-magnitude WRONG prediction ---")
print(f"  Predicted: {y_pred[median_wrong]:+.4f}  Actual: {y_test[median_wrong]:+.4f}")
print(f"  Asset: {assets_np[test_idx[median_wrong]]}")

fig, ax = plt.subplots(figsize=(8, 6))
shap.plots.waterfall(explanation[median_wrong], max_display=12, show=False)
plt.title("Median Wrong: High-Magnitude Incorrect Prediction")
plt.show()

# %% [markdown] tags=[]
# ## SHAP Stability Across Folds
#
# A model that learns different feature rankings in each fold is less trustworthy
# than one with stable importance. We train Ridge on all 8 walk-forward folds
# and track mean |SHAP| for the top features — a key validation diagnostic.

# %% tags=[]
if fold_importance is not None:
    top_stab = importance["feature"].head(10).to_list()
    top_stab_idx = [FEATURE_COLS.index(f) for f in top_stab]
    # 5 distinguishable hues × 2 linestyles → 10 unique combinations,
    # greyscale-readable via the linestyle dimension.
    stab_palette = [
        COLORS["blue"],
        COLORS["amber"],
        COLORS["copper"],
        COLORS["positive"],
        COLORS["slate"],
    ]
    linestyles = ["-", "--"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (feat, fi) in enumerate(zip(top_stab, top_stab_idx, strict=True)):
        vals = [fold_importance[k][fi] for k in sorted(fold_importance)]
        ax.plot(
            range(1, len(vals) + 1),
            vals,
            marker="o",
            color=stab_palette[i % len(stab_palette)],
            linestyle=linestyles[i // len(stab_palette)],
            label=feat,
            markersize=4,
        )

    ax.set_xlabel("Fold")
    ax.set_ylabel("Mean |SHAP value| on test fold")
    ax.set_title("SHAP Feature Importance Stability Across Folds")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.show()
else:
    print("Fold importance not available (run with RETRAIN=True)")

# %% [markdown] tags=[]
# **Interpretation**: Features that maintain their importance across folds represent
# stable signals. Features that spike in one fold and vanish in the next may be
# learning fold-specific noise rather than a persistent alpha source.
#
# > **Library note**: In production, `ml4t.diagnostic.evaluation.compute_shap_importance()`
# > automates fold-level SHAP aggregation.
#
# > **Extension**: Regime-conditional SHAP analysis — partitioning test data by
# > volatility tercile and computing SHAP summaries within each partition — is
# > demonstrated in the cross-dataset evaluation notebooks where regime labels
# > from Ch7 are available. This adds the fourth layer of economic narrative
# > building described in the chapter text.

# %% [markdown] tags=[]
# ## Bootstrap Confidence Bands (Within-Fold)
#
# Cross-fold variation conflates *temporal regime shifts* (interesting) with
# *finite-sample estimation noise* (uninteresting). Bootstrapping SHAP values
# *within* a fold isolates the second source: we resample the test rows with
# replacement, recompute mean $|\phi_j|$ for each top feature, and quote the
# percentile interval. A wide band on a feature with a high cross-fold rank
# means its mean $|\phi_j|$ for *this* fold is itself uncertain.

# %% tags=[]
N_BOOT = 200
TOP_K_BOOT = 10
top_boot = importance["feature"].head(TOP_K_BOOT).to_list()
top_boot_idx = [FEATURE_COLS.index(f) for f in top_boot]
n_test = shap_values.shape[0]

rng = np.random.default_rng(RANDOM_SEED)
boot_means = np.empty((N_BOOT, TOP_K_BOOT), dtype=float)
for b in range(N_BOOT):
    sample = rng.integers(0, n_test, n_test)
    boot_means[b] = np.mean(np.abs(shap_values[sample][:, top_boot_idx]), axis=0)

boot_lo = np.percentile(boot_means, 2.5, axis=0)
boot_hi = np.percentile(boot_means, 97.5, axis=0)
boot_med = np.median(boot_means, axis=0)
point_estimate = np.mean(np.abs(shap_values[:, top_boot_idx]), axis=0)

bootstrap_ci = pl.DataFrame(
    {
        "feature": top_boot,
        "mean_abs_shap": point_estimate,
        "boot_p2.5": boot_lo,
        "boot_median": boot_med,
        "boot_p97.5": boot_hi,
        "ci_width": boot_hi - boot_lo,
    }
)
bootstrap_ci

# %% tags=[]
fig, ax = plt.subplots(figsize=(9, 5))
ypos = np.arange(TOP_K_BOOT)
ax.errorbar(
    point_estimate,
    ypos,
    xerr=[point_estimate - boot_lo, boot_hi - point_estimate],
    fmt="o",
    color=COLORS["blue"],
    ecolor=COLORS["amber"],
    elinewidth=2,
    capsize=4,
)
ax.set_yticks(ypos)
ax.set_yticklabels(top_boot)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP value| (with 95% bootstrap CI)")
ax.set_title("Within-Fold Bootstrap Confidence Bands on Top-10 Features")
plt.show()

# %% [markdown] tags=[]
# **Interpretation**: a top feature whose 95% bootstrap interval excludes the
# point estimate of a lower-ranked feature is genuinely *more important on this
# fold* — finite-sample noise alone cannot explain the gap. Features whose
# intervals overlap with several neighbours have brittle rankings; they may
# trade rank order across bootstrap replicates and should not be treated as
# uniquely "the most important" feature. This is the within-fold complement
# to the across-fold stability chart above.

# %% [markdown] tags=[]
# ## Key Takeaways
#
# 1. **Linear SHAP is exact.** For Ridge regression, SHAP values equal
#    $\beta_j \cdot (x_j - \bar{x}_j)$ — no approximation, fully verifiable.
#    This makes linear models the ideal starting point for interpretability;
#    the closed-form match (correlation $\approx 0.99$, max difference
#    $< 10^{-2}$) is a sanity check for the whole SHAP toolchain.
#
# 2. **Volatility horizons dominate global importance.** $vol_{126d}$
#    (positive coefficient) and $vol_{63d}$ (negative coefficient) lead
#    mean $|\phi_j|$, with $sharpe_{63d}$, $vol\_ratio_{medium}$, and
#    $yield\_curve\_slope$ rounding out the top five. Three of six prior-belief
#    momentum features (`ret_63d`, `ret_126d`, `obv_zscore_63d`) flip sign
#    versus expectation; the standalone factor signals are not strong on this
#    panel.
#
# 3. **High-conviction predictions are wrong more often than right** ($58\%$
#    wrong vs $42\%$ right at the top-20% conviction band). The right-vs-wrong
#    SHAP comparison is therefore a model-improvement diagnostic, not a
#    validation of the current model. Features with high *wrong / right*
#    SHAP ratios — most notably `hurst_100`, `yield_curve_zscore`, and
#    `chop_14` on this fold — are the first candidates for re-engineering or
#    interaction terms.
#
# 4. **Concentration and stability checks belong in the standing pipeline,
#    not as one-off diagnostics.** The 60% concentration threshold flags zero
#    predictions on this fold (median max-feature share $\approx 20\%$); the
#    cross-fold importance trajectory plus within-fold bootstrap CIs together
#    distinguish "feature is important everywhere" from "feature looks
#    important on this fold only."
#
# **Next**: `06_conformal_prediction` quantifies *how uncertain* each
# prediction is — the natural complement to "what drives this prediction."
# *Chapter 12* extends SHAP to gradient boosting via `TreeExplainer`, where
# attributions capture non-linear interactions invisible to coefficient
# tables.
