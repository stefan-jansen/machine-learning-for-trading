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

# %% [markdown] tags=[]
# # Hyperparameter Selection and Validation Bias
#
# **Docker image**: `ml4t`
#
# **Purpose**: develop a systematic approach to hyperparameter selection for
# regularized models. The notebook answers two questions:
#
# 1. How does regularization affect Ridge performance on the ETF panel? (alpha
#    grid analysis)
# 2. How much does HPO selection bias inflate single-loop CV estimates relative
#    to nested CV? (Cawley and Talbot 2010)
#
# **Learning objectives**
#
# - Map performance stability across a wide regularization range
# - Distinguish a robust alpha plateau from a noisy landscape
# - Implement nested walk-forward CV with proper inner/outer purging and embargo
# - Quantify the inflation from optimizing hyperparameters and reporting on the
#   same data
# - Apply the 3σ overfitting diagnostic to a hyperparameter search
#
# **Book reference**: Section 11.2 - Regularized Regression (nested CV +
# selection bias).
#
# **Prerequisites**
#
# - Ch7 21-day forward return labels at `case_studies/etfs/labels/fwd_ret_21d.parquet`
# - Ch8 ETF features at `case_studies/etfs/features/financial.parquet`
# - `02_regularization_paths` (coefficient paths context for Ridge)
#
# **Downstream**: Ch12 extends the same nested-CV machinery to gradient boosting's
# higher-dimensional hyperparameter space.

# %% [markdown] tags=[]
# ## 1. Setup and Imports

# %% tags=[]
"""Hyperparameter Selection and Validation Bias - nested CV for unbiased performance evaluation."""

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from ml4t.diagnostic.splitters import WalkForwardCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils.modeling import cross_sectional_ic_mean
from utils.paths import get_case_study_dir, get_chapter_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, ml4t_diverging, show_with_alt

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% tags=["parameters"]
SEED = 42
RETRAIN = False
MAX_SYMBOLS = 0
N_SPLITS = 5
N_TRIALS = 20
CV_MAX_SPLITS = 0
CV_MAX_TRIALS = 0
ALPHA_GRID_POINTS = 23
TEST_SIZE = 200
ARTIFACT_TAG = ""

# %% tags=[]
# Configuration
RANDOM_SEED = SEED
set_global_seeds(SEED)
N_SPLITS_EFFECTIVE = N_SPLITS if CV_MAX_SPLITS <= 0 else min(N_SPLITS, CV_MAX_SPLITS)
N_TRIALS_EFFECTIVE = N_TRIALS if CV_MAX_TRIALS <= 0 else min(N_TRIALS, CV_MAX_TRIALS)

# %% [markdown] tags=[]
# ## 2. Load ETF Features
#
# We use the pre-computed ETF features from Chapter 8 and labels from Chapter 7.

# %% tags=[]
CASE_DIR = get_case_study_dir("etfs")
FEATURES_PATH = CASE_DIR / "features" / "financial.parquet"
LABELS_PATH = CASE_DIR / "labels" / "fwd_ret_21d.parquet"

assert FEATURES_PATH.exists(), (
    f"Features not found: {FEATURES_PATH}\nRun Ch8 feature engineering first."
)
assert LABELS_PATH.exists(), f"Labels not found: {LABELS_PATH}\nRun Ch7 label engineering first."

features_df = pl.read_parquet(FEATURES_PATH).with_columns(pl.col("timestamp").cast(pl.Date))
labels_df = pl.read_parquet(LABELS_PATH).with_columns(pl.col("timestamp").cast(pl.Date))

# Join features and labels
TARGET_COL = "fwd_ret_21d"
ASSET_COL = "symbol"
dataset = features_df.join(labels_df, on=["timestamp", ASSET_COL], how="inner").drop_nulls(
    subset=[TARGET_COL]
)
dataset = dataset.sort(["timestamp", ASSET_COL])

if MAX_SYMBOLS > 0:
    assets = sorted(dataset[ASSET_COL].unique().to_list())[:MAX_SYMBOLS]
    dataset = dataset.filter(pl.col(ASSET_COL).is_in(assets))

print(f"Dataset: {len(dataset):,} rows")
print(f"Assets: {dataset[ASSET_COL].n_unique()}")
print(f"Date range: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}")

# %% [markdown] tags=[]
# ## 3. Prepare Features and Target
#
# A feature that could not be computed arrives as a null, and every row carrying
# one is dropped. Zero is not a neutral filler for any of these columns: it is a
# flat 252-day return, a zero volatility, and a mid-range oscillator reading, so
# filling with it would put invented observations into the training set. The rows
# this removes are the early history of a handful of symbols, where a long-window
# feature has not had its warm-up period yet.

# %% tags=[]
meta_cols = {"timestamp", ASSET_COL}
label_cols = {c for c in dataset.columns if c.startswith("fwd_ret")}
feature_cols = [c for c in dataset.columns if c not in meta_cols and c not in label_cols]

df = dataset.select(feature_cols + [TARGET_COL, "timestamp", ASSET_COL]).with_columns(
    [
        pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in feature_cols
    ]
)
rows_before = df.height
df = df.drop_nulls(subset=feature_cols)
print(f"Dropped {rows_before - df.height:,} rows with an incomplete feature vector")

X = df.select(feature_cols).to_numpy()
y = df[TARGET_COL].to_numpy()

# Create DataFrame with datetime index for WalkForwardCV
dates = df["timestamp"].to_list()
dates_arr = df["timestamp"].to_numpy()
symbols_arr = df[ASSET_COL].to_numpy()
df_for_cv = pd.DataFrame(X, columns=feature_cols)
df_for_cv.index = pd.DatetimeIndex(dates).tz_localize("UTC")


print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target: {TARGET_COL}")

# %% [markdown] tags=[]
# ---
# # Part A: Alpha Grid Analysis
#
# Before optimizing hyperparameters, we should understand the **landscape** of
# performance across regularization levels. This reveals:
#
# - Is there a stable plateau where performance is robust?
# - Or is performance noisy/sensitive to alpha choice?
# - Should we use a single alpha or an ensemble?
#
# This analysis informs our HPO strategy before we even run Optuna.

# %% [markdown] tags=[]
# ## 4. Define Alpha Grid and Evaluation Function

# %% tags=[]
ALPHAS = np.logspace(-2, 9, ALPHA_GRID_POINTS)
LABEL_HORIZON = 21  # sessions the label looks forward, and the purge each split needs

# %% [markdown] tags=[]
# ### Evaluate Alpha Grid
#
# Walk-forward CV across the full alpha range, returning per-fold IC for each alpha.


# %% tags=[]
def evaluate_alpha_grid(
    X: np.ndarray,
    y: np.ndarray,
    df_cv: pd.DataFrame,
    alphas: np.ndarray,
    n_splits: int,
    test_size: int,
    label_horizon: int,
) -> pd.DataFrame:
    """
    Evaluate Ridge regression across a grid of alphas for each CV fold.

    Returns DataFrame with columns: fold, alpha, ic
    """
    results = []

    cv = WalkForwardCV(
        n_splits=n_splits,
        test_size=test_size,
        label_horizon=label_horizon,
        embargo_size=10,
        expanding=True,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df_cv)):
        if len(train_idx) < 10:
            continue

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale inside fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        dates_test = dates_arr[test_idx]
        symbols_test = symbols_arr[test_idx]

        # Evaluate each alpha
        for alpha in alphas:
            model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            ic = cross_sectional_ic_mean(y_test, pred, dates_test, symbols_test)
            results.append({"fold": fold_idx + 1, "alpha": alpha, "ic": ic})

    return pd.DataFrame(results)


# %% [markdown] tags=[]
# ## 5. Run Alpha Grid Analysis

# %% tags=[]
MODELS_DIR = get_chapter_dir(11) / "models" / "04_nested_cv_hpo"
is_reduced_run = CV_MAX_SPLITS > 0 or CV_MAX_TRIALS > 0 or ALPHA_GRID_POINTS < 23 or MAX_SYMBOLS > 0
if is_reduced_run and not ARTIFACT_TAG:
    ARTIFACT_TAG = "_fast"

GRID_PATH = MODELS_DIR / f"grid_results{ARTIFACT_TAG}.joblib"
CV_PATH = MODELS_DIR / f"cv_results{ARTIFACT_TAG}.joblib"

NEED_GRID = RETRAIN or not GRID_PATH.exists()

if not NEED_GRID:
    grid_results = joblib.load(GRID_PATH)
    cached_alphas = np.sort(grid_results["alpha"].unique())
    if (
        cached_alphas.min() > ALPHAS.min()
        or cached_alphas.max() < ALPHAS.max()
        or len(cached_alphas) < len(ALPHAS)
    ):
        print("Cached grid uses a narrower alpha range; recomputing with expanded sweep.")
        NEED_GRID = True

if NEED_GRID:
    print("Evaluating alpha grid across CV folds...")
    grid_results = evaluate_alpha_grid(
        X, y, df_for_cv, ALPHAS, N_SPLITS_EFFECTIVE, TEST_SIZE, LABEL_HORIZON
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid_results, GRID_PATH)

# Pivot to get fold × alpha matrix
ic_matrix = grid_results.pivot(index="fold", columns="alpha", values="ic")
print(f"\nGrid shape: {ic_matrix.shape[0]} folds × {ic_matrix.shape[1]} alphas")

# %% [markdown] tags=[]
# ## 6. Visualize Alpha Landscape

# %% [markdown] tags=[]
# The colour scale diverges around zero because zero is the meaningful point for
# an IC: it separates a ranking that helped from one that hurt. The scale is made
# symmetric so that equal distances either side of zero read as equal.

# %% tags=[]
ml4t_div = LinearSegmentedColormap.from_list("ml4t_div", ml4t_diverging())
fig, ax = plt.subplots(figsize=(12, 4))
ic_limit = float(np.nanmax(np.abs(ic_matrix.values)))
im = ax.imshow(
    ic_matrix.values,
    aspect="auto",
    cmap=ml4t_div,
    vmin=-ic_limit,
    vmax=ic_limit,
)
# Compact scientific labels (every other tick) keep the alpha grid legible.
tick_idx = list(range(0, len(ic_matrix.columns), 2))
ax.set_xticks(tick_idx)
ax.set_xticklabels(
    [f"{ic_matrix.columns[i]:.0e}" for i in tick_idx], rotation=45, ha="right", fontsize=8
)
ax.set_yticks(range(len(ic_matrix.index)))
ax.set_yticklabels([f"Fold {i}" for i in ic_matrix.index])
ax.set_xlabel(r"Alpha (log spacing, $10^{-2}$ to $10^{9}$)")
ax.set_title("The same alpha does not score alike on every fold")
fig.colorbar(im, ax=ax, label="IC", shrink=0.8)
show_with_alt(
    fig,
    "Heatmap of information coefficient with one row per fold and one column per "
    "alpha, on a scale diverging around zero.",
)

# %% tags=[]
alpha_stats = (
    grid_results.groupby("alpha")
    .agg(
        mean_ic=("ic", "mean"),
        std_ic=("ic", "std"),
        min_ic=("ic", "min"),
        max_ic=("ic", "max"),
    )
    .reset_index()
)

# Plot mean IC with error bands
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(
    alpha_stats["alpha"],
    alpha_stats["mean_ic"] - alpha_stats["std_ic"],
    alpha_stats["mean_ic"] + alpha_stats["std_ic"],
    alpha=0.2,
    color=COLORS["amber"],
)
ax.semilogx(
    alpha_stats["alpha"],
    alpha_stats["mean_ic"],
    "o-",
    color=COLORS["amber"],
    lw=2,
    markersize=6,
)
ax.set_xlabel("Alpha (log scale)")
ax.set_ylabel("Information Coefficient")
ax.set_title("Mean IC moves smoothly with alpha, inside a wide error band")
show_with_alt(
    fig,
    "Mean information coefficient against alpha on a log axis, with a shaded band "
    "one standard deviation wide either side.",
)

# %% [markdown] tags=[]
# ## 7. Stability Analysis

# %% tags=[]
alpha_stats["cv"] = alpha_stats["std_ic"] / alpha_stats["mean_ic"].abs()

# Find the most stable region
best_alpha_idx = alpha_stats["mean_ic"].idxmax()
best_alpha = alpha_stats.loc[best_alpha_idx, "alpha"]
best_ic = alpha_stats.loc[best_alpha_idx, "mean_ic"]
best_std = alpha_stats.loc[best_alpha_idx, "std_ic"]

# Find stable plateau (alphas within 90% of best performance)
threshold = best_ic * 0.9 if best_ic > 0 else best_ic * 1.1
stable_alphas = alpha_stats[alpha_stats["mean_ic"] >= threshold]["alpha"]

# Assess stability
overall_cv = alpha_stats["cv"].median()
if overall_cv < 0.5:
    stability_label = "stable"
elif overall_cv < 1.0:
    stability_label = "moderate"
else:
    stability_label = "unstable"

# %% tags=[]
stability_df = pl.DataFrame(
    {
        "Metric": [
            "Best alpha",
            "Mean IC",
            "Std IC",
            "Plateau size",
            "Plateau range",
            "Median CV",
            "Stability",
        ],
        "Value": [
            f"{best_alpha:.3f}",
            f"{best_ic:.4f}",
            f"{best_std:.4f}",
            f"{len(stable_alphas)} alphas",
            f"[{stable_alphas.min():.3f}, {stable_alphas.max():.3f}]",
            f"{overall_cv:.2f}",
            stability_label,
        ],
    }
)
stability_df

# %% [markdown] tags=[]
# **How to read the table.** Three of its rows describe the landscape and one
# describes how much you should trust it.
#
# The plateau size and range say whether the alpha that scores highest is
# meaningfully better than its neighbours or merely first past the post. A
# plateau spanning orders of magnitude means the choice within it hardly
# matters, and picking the exact maximum is picking noise.
#
# The median coefficient of variation is the ratio of fold-to-fold dispersion to
# the size of the mean, taken across alphas. Above one, the IC at a given alpha
# varies between folds by more than the mean IC itself is worth, and no ranking
# of alphas computed from these folds is reliable.
#
# That ratio is what makes Part B necessary. Where fold-level noise dwarfs the
# mean, a search that takes the highest-scoring alpha per fold is taking whichever alpha
# suited that fold's noise, and reporting its score as performance measures the
# search rather than the model.

# %% [markdown] tags=[]
# ---
# # Part B: Nested vs Single-Loop Cross-Validation
#
# Now that we understand the alpha landscape, we address the **measurement problem**:
# when we select hyperparameters and evaluate on the same data, we get optimistic
# estimates due to selection bias.
#
# **Key insight**: Single-loop CV (select best alpha, report that performance)
# overestimates because we're measuring "how well did we fit the test set during HPO"
# rather than "how well will we generalize."

# %% [markdown] tags=[]
# ## 8. Single-Loop CV (BIASED)
#
# This approach performs HPO and evaluation on the same data splits.
# The selected hyperparameters are optimized for the test set, leading to
# overly optimistic results.


# %% tags=[]
def _make_single_loop_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dates_test: np.ndarray,
    symbols_test: np.ndarray,
):
    """Create objective function that optimizes on test set (biased!)."""

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.01, 1e9, log=True)
        model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        ic = cross_sectional_ic_mean(y_test, pred, dates_test, symbols_test)
        if np.isnan(ic):
            raise optuna.TrialPruned("no validation date carried a defined IC")
        return ic

    return objective


# %% [markdown] tags=[]
# ### Single-Loop CV Runner
#
# Runs HPO and evaluation on the same data splits - the biased baseline.


# %% tags=[]
def single_loop_cv(
    X: np.ndarray,
    y: np.ndarray,
    df_cv: pd.DataFrame,
    n_splits: int = 5,
    n_trials: int = 15,
    label_horizon: int = 21,
    test_size: int = 200,
) -> dict:
    """
    Single-loop CV: HPO and evaluation on same splits (BIASED).

    This is the WRONG way - hyperparameters are selected based on
    the same test set used for final evaluation.
    """
    results = {"ic": [], "best_alpha": []}

    cv = WalkForwardCV(
        n_splits=n_splits,
        test_size=test_size,
        label_horizon=label_horizon,
        embargo_size=10,
        expanding=True,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df_cv)):
        if len(train_idx) < 10:
            continue

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        dates_test = dates_arr[test_idx]
        symbols_test = symbols_arr[test_idx]

        # Scale inside fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # HPO on test set (this is the bias!)
        objective = _make_single_loop_objective(
            X_train, y_train, X_test, y_test, dates_test, symbols_test
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Evaluate with best params on SAME test set
        best_alpha = study.best_params["alpha"]
        model = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        ic = cross_sectional_ic_mean(y_test, pred, dates_test, symbols_test)

        results["ic"].append(ic)
        results["best_alpha"].append(best_alpha)

    return results


# %% [markdown] tags=[]
# ## 9. Nested CV (UNBIASED)
#
# This approach uses an **inner loop** for HPO and **outer loop** for evaluation.
# The test set is never seen during hyperparameter selection.
#
# **Both loops use WalkForwardCV** to maintain proper temporal structure
# with purging and embargo.


# %% tags=[]
def _make_nested_objective(
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    dates_outer_train: np.ndarray,
    symbols_outer_train: np.ndarray,
    df_inner: pd.DataFrame,
    n_inner: int,
    label_horizon: int,
):
    """
    Create objective function for nested CV inner loop.

    Uses WalkForwardCV for inner loop to maintain proper temporal
    structure with purging and embargo.
    """
    # WalkForwardCV sizes its windows in sessions; this is a panel with many rows
    # per session, so the inner test size is counted in sessions too.
    inner_sessions = int(pd.DatetimeIndex(df_inner.index).nunique())
    inner_cv = WalkForwardCV(
        n_splits=n_inner,
        test_size=max(label_horizon, inner_sessions // (n_inner + 2)),
        label_horizon=label_horizon,
        embargo_size=label_horizon,  # Inner embargo must also respect label horizon
        expanding=True,
    )

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.01, 1e9, log=True)
        inner_scores = []

        for inner_train_idx, inner_val_idx in inner_cv.split(df_inner):
            if len(inner_train_idx) < 10:
                continue

            X_inner_train = X_outer_train[inner_train_idx]
            X_inner_val = X_outer_train[inner_val_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            y_inner_val = y_outer_train[inner_val_idx]
            dates_inner_val = dates_outer_train[inner_val_idx]
            symbols_inner_val = symbols_outer_train[inner_val_idx]

            # Scale inside inner fold
            inner_scaler = StandardScaler()
            X_inner_train_scaled = inner_scaler.fit_transform(X_inner_train)
            X_inner_val_scaled = inner_scaler.transform(X_inner_val)

            model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
            model.fit(X_inner_train_scaled, y_inner_train)
            pred = model.predict(X_inner_val_scaled)
            ic = cross_sectional_ic_mean(y_inner_val, pred, dates_inner_val, symbols_inner_val)
            if not np.isnan(ic):
                inner_scores.append(ic)

        if not inner_scores:
            raise optuna.TrialPruned("no inner fold carried a defined IC")
        return float(np.mean(inner_scores))

    return objective


# %% [markdown] tags=[]
# ### Nested CV Runner
#
# Uses an inner WalkForwardCV loop for HPO, keeping the outer test set
# completely hidden from hyperparameter selection.


# %% tags=[]
def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    df_cv: pd.DataFrame,
    n_outer: int = 5,
    n_inner: int = 3,
    n_trials: int = 15,
    label_horizon: int = 21,
    test_size: int = 200,
) -> dict:
    """
    Nested CV: HPO isolated from evaluation (UNBIASED).

    Outer loop: WalkForwardCV for temporal validation
    Inner loop: WalkForwardCV for HPO (within training data only)

    This is the CORRECT approach - hyperparameters are selected using only
    training data, and final evaluation is on truly held-out test data.
    """
    results = {"ic": [], "best_alpha": []}

    outer_cv = WalkForwardCV(
        n_splits=n_outer,
        test_size=test_size,
        label_horizon=label_horizon,
        embargo_size=10,
        expanding=True,
    )

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(df_cv)):
        if len(outer_train_idx) < 10:
            continue

        X_outer_train_raw = X[outer_train_idx]
        X_outer_test_raw = X[outer_test_idx]
        y_outer_train = y[outer_train_idx]
        y_outer_test = y[outer_test_idx]
        dates_outer_train = dates_arr[outer_train_idx]
        symbols_outer_train = symbols_arr[outer_train_idx]
        dates_outer_test = dates_arr[outer_test_idx]
        symbols_outer_test = symbols_arr[outer_test_idx]

        # Create inner DataFrame for WalkForwardCV
        df_inner = df_cv.iloc[outer_train_idx].copy()

        # Inner loop: HPO on outer training data ONLY
        objective = _make_nested_objective(
            X_outer_train_raw,
            y_outer_train,
            dates_outer_train,
            symbols_outer_train,
            df_inner,
            n_inner,
            label_horizon,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Outer evaluation with best params (UNBIASED)
        best_alpha = study.best_params["alpha"]

        # Scale outer fold for final evaluation
        outer_scaler = StandardScaler()
        X_outer_train = outer_scaler.fit_transform(X_outer_train_raw)
        X_outer_test = outer_scaler.transform(X_outer_test_raw)

        model = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
        model.fit(X_outer_train, y_outer_train)
        pred = model.predict(X_outer_test)
        ic = cross_sectional_ic_mean(y_outer_test, pred, dates_outer_test, symbols_outer_test)

        results["ic"].append(ic)
        results["best_alpha"].append(best_alpha)

    return results


# %% [markdown] tags=[]
# ## 10. Run Comparison

# %% tags=[]
N_INNER = 3

NEED_CV = RETRAIN or not CV_PATH.exists()

if NEED_CV:
    print("=" * 60)
    print("Running Single-Loop CV (BIASED)...")
    print("=" * 60)
    single_results = single_loop_cv(
        X,
        y,
        df_for_cv,
        n_splits=N_SPLITS_EFFECTIVE,
        n_trials=N_TRIALS_EFFECTIVE,
        label_horizon=LABEL_HORIZON,
        test_size=TEST_SIZE,
    )

    print("\n" + "=" * 60)
    print("Running Nested CV (UNBIASED)...")
    print("=" * 60)
    nested_results = nested_cv(
        X,
        y,
        df_for_cv,
        n_outer=N_SPLITS_EFFECTIVE,
        n_inner=N_INNER,
        n_trials=N_TRIALS_EFFECTIVE,
        label_horizon=LABEL_HORIZON,
        test_size=TEST_SIZE,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"single_results": single_results, "nested_results": nested_results}, CV_PATH)
    print(f"Saved results to {CV_PATH}")

# %% [markdown] tags=[]
# ### Load Cached Results
#
# If CV results are already cached, load them and verify completeness.

# %% tags=[]
if not NEED_CV:
    _cached = joblib.load(CV_PATH)
    single_results = _cached["single_results"]
    nested_results = _cached["nested_results"]
    del _cached
    if (
        len(single_results["ic"]) < N_SPLITS_EFFECTIVE
        or len(nested_results["ic"]) < N_SPLITS_EFFECTIVE
    ):
        print("Cached CV results appear incomplete; recomputing.")
        single_results = single_loop_cv(
            X,
            y,
            df_for_cv,
            n_splits=N_SPLITS_EFFECTIVE,
            n_trials=N_TRIALS_EFFECTIVE,
            label_horizon=LABEL_HORIZON,
            test_size=TEST_SIZE,
        )
        nested_results = nested_cv(
            X,
            y,
            df_for_cv,
            n_outer=N_SPLITS_EFFECTIVE,
            n_inner=N_INNER,
            n_trials=N_TRIALS_EFFECTIVE,
            label_horizon=LABEL_HORIZON,
            test_size=TEST_SIZE,
        )
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"single_results": single_results, "nested_results": nested_results}, CV_PATH)

# %% [markdown] tags=[]
# ## 11. Results Analysis

# %% tags=[]
single_mean_ic = np.nanmean(single_results["ic"])
single_std_ic = np.nanstd(single_results["ic"])
nested_mean_ic = np.nanmean(nested_results["ic"])
nested_std_ic = np.nanstd(nested_results["ic"])

# Calculate inflation
if nested_mean_ic != 0:
    inflation_pct = (single_mean_ic - nested_mean_ic) / abs(nested_mean_ic) * 100
else:
    inflation_pct = 0.0

comparison_df = pl.DataFrame(
    {
        "Method": ["Single-Loop (Biased)", "Nested (Unbiased)", "Inflation"],
        "Mean IC": [f"{single_mean_ic:.4f}", f"{nested_mean_ic:.4f}", f"{inflation_pct:+.1f}%"],
        "Std IC": [f"{single_std_ic:.4f}", f"{nested_std_ic:.4f}", ""],
        "Alpha Range": [
            f"[{min(single_results['best_alpha']):.2f}, {max(single_results['best_alpha']):.2f}]",
            f"[{min(nested_results['best_alpha']):.2f}, {max(nested_results['best_alpha']):.2f}]",
            "",
        ],
    }
)
comparison_df

# %% [markdown] tags=[]
# **How to read the table.** The two protocols ran on identical outer splits and
# differ in one respect: where the alpha came from. Single-loop CV chose it by
# maximizing IC on the very rows it then reports, so its number answers "how
# well can this model be made to fit this test fold". Nested CV chose it inside
# the training window and never showed the outer test fold to the search, so its
# number answers "how well does the procedure generalize". The gap between them
# is the selection bias, and it is a property of the search, not of Ridge.
#
# The alpha column is the more direct evidence. Selections that jump across
# orders of magnitude from fold to fold mean the search is tracking each fold's
# noise rather than a stable property of the data. Selections that cluster mean
# the inner-loop average is finding something the folds agree on. Compare the
# spread of the two columns before reading either mean IC.

# %% [markdown] tags=[]
# ### Validation Overfitting Diagnostic (3σ Heuristic)
#
# The text proposes a concrete diagnostic: if the top configuration's IC exceeds the
# median by more than 3× the inter-configuration standard deviation, the search is
# likely overfitting to validation noise. We apply this to the alpha grid from Part A,
# where each alpha's mean IC across folds serves as a "trial" value.

# %% tags=[]
trial_ics = alpha_stats["mean_ic"].values
best_trial_ic = trial_ics.max()
median_trial_ic = np.median(trial_ics)
std_trial_ic = np.std(trial_ics)

gap_sigma = (best_trial_ic - median_trial_ic) / std_trial_ic if std_trial_ic > 0 else 0.0
overfitting_flag = "WARNING: likely overfitting" if gap_sigma > 3.0 else "OK"

diagnostic_df = pl.DataFrame(
    {
        "Metric": ["Best IC", "Median IC", "Std IC", "Gap (σ)", "Overfit flag"],
        "Value": [
            f"{best_trial_ic:.4f}",
            f"{median_trial_ic:.4f}",
            f"{std_trial_ic:.4f}",
            f"{gap_sigma:.1f}σ",
            overfitting_flag,
        ],
    }
)
diagnostic_df

# %% [markdown] tags=[]
# This diagnostic and the stability classification above measure dispersion in
# two different directions, and they can disagree without either being wrong.
# Stability looks *within* an alpha, across folds: it asks whether the same
# setting scores consistently on different periods. The three-sigma gap looks
# *across* alphas, at the fold-averaged scores: it asks whether the top setting
# stands out from the field.
#
# A landscape can be smooth across alphas and noisy across folds at the same
# time, and that combination is the one to watch for. It means the alpha-by-alpha
# curve looks orderly enough to invite a confident choice, while the evidence
# behind each point on it is thin.

# %% [markdown] tags=[]
# ## 12. Visualization

# %% tags=[]
folds = list(range(1, len(single_results["ic"]) + 1))
x = np.arange(len(folds))
bar_w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) IC comparison
axes[0].bar(
    x - bar_w / 2, single_results["ic"], bar_w, label="Single-Loop (Biased)", color=COLORS["blue"]
)
axes[0].bar(
    x + bar_w / 2, nested_results["ic"], bar_w, label="Nested (Unbiased)", color=COLORS["amber"]
)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"Fold {f}" for f in folds])
axes[0].set_ylabel("Information Coefficient")
axes[0].set_title("(a) IC by Fold")
axes[0].legend(frameon=False)

# (b) Alpha selection
axes[1].semilogy(
    folds, single_results["best_alpha"], "o-", color=COLORS["blue"], label="Single-Loop"
)
axes[1].semilogy(folds, nested_results["best_alpha"], "s-", color=COLORS["amber"], label="Nested")
axes[1].set_xlabel("Fold")
axes[1].set_ylabel("Best Alpha (log)")
axes[1].set_title("(b) Alpha Selection by Fold")
axes[1].legend(frameon=False)

fig.suptitle("Isolating the search from the evaluation changes both", fontsize=13)
show_with_alt(
    fig,
    "Two panels: paired bars of information coefficient per fold for the single-loop "
    "and nested protocols, and the alpha each protocol selected per fold on a log axis.",
)

# %% [markdown] tags=[]
# ## 13. Key Takeaways
#
# 1. **Report the number the protocol earns.** A hyperparameter chosen on a set
#    of rows makes any score computed on those same rows a statement about the
#    search. Nested CV separates the two by keeping the outer test fold out of
#    the selection entirely, and the difference between the two mean ICs above
#    is what the separation costs on paper and buys in honesty.
#
# 2. **The spread of the selected values is the tell.** Where a search chases
#    noise, its choice moves from fold to fold with nothing in the data driving
#    it. Plotting the selected alpha per fold, as panel (b) does, diagnoses that
#    faster than comparing scores.
#
# 3. **Two dispersion diagnostics, in different directions.** Fold-to-fold
#    variation at a fixed alpha and alpha-to-alpha variation of the fold average
#    answer different questions, and a landscape that is smooth in one can be
#    dominated by noise in the other.
#
# 4. **A wide grid before an expensive search is cheap insurance.** Sweeping
#    alpha across many orders of magnitude costs one pass and tells you whether
#    the landscape has a plateau worth searching in at all.
#
# 5. **Both loops need the purge.** The inner split has the same overlapping
#    label as the outer one, so it needs the same gap; an inner loop that
#    silently produces no folds leaves the search with nothing to optimize and
#    hands back whatever its sampler tried first.
#
# **Reference**: Cawley and Talbot (2010) document the same bias mechanism
# across multiple datasets and model classes.

# %% tags=[]
print("Hyperparameter selection analysis complete")
print(
    f"  Alpha grid: {len(ALPHAS)} levels evaluated, mean IC range "
    f"{alpha_stats['mean_ic'].min():.4f} to {alpha_stats['mean_ic'].max():.4f}"
)
print(f"  Single-loop mean IC: {single_mean_ic:+.4f}  |  Nested mean IC: {nested_mean_ic:+.4f}")
print(f"  HPO inflation:       {inflation_pct:+.1f}%")
print(f"  Stability label:     {stability_label}   |  Cross-alpha gap: {gap_sigma:.1f}σ")
