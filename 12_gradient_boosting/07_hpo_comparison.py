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
# # Grid Search vs Optuna: HPO Method Comparison
#
# **Docker image**: `ml4t`
#
# **Chapter 12, Section 12.4**: Advanced Hyperparameter Tuning with Optuna
#
# ## Purpose
# This notebook compares grid search against Optuna's Bayesian optimization on
# the same parameter budget, first on a space grid can enumerate and then on a
# continuous one it cannot. It also measures validation overfitting: how
# test IC diverges from validation IC as trial count increases.
#
# ## Key Insight
# The comparison is decided by the space, not by the sampler. On a grid small
# enough to exhaust, search has nothing to exploit and exhaustion wins by
# construction; a continuous space is where sampling reaches configurations a
# discrete grid cannot represent at all. The wall times reported along the way
# are single uncontrolled runs, not a cost comparison.
#
# ## Cross-References
# - **Section 12.4**: TPE, pruning, and its box on validation overfitting
# - **Related**: `04_optuna_tuning` (full workflow), `06_optuna_multi_asset` (multi-objective)
#
# ## References
# - Bergstra & Bengio (2012). "Random Search for Hyper-Parameter Optimization"
# - Akiba et al. (2019). "Optuna: A Next-generation HPO Framework"

# %% [markdown]
# ## 1. Setup

# %%
"""Grid Search against Optuna - compare HPO methods on convergence and final score."""

import time
import warnings

# lightgbm loads before anything that pulls in scikit-learn, ml4t.diagnostic included:
# the first OpenMP runtime loaded wins the process, and sklearn's first segfaults
# LightGBM's next threaded fit on macOS ARM64. This notebook fits with n_jobs=-1.
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import polars as pl
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from optuna.samplers import TPESampler
from sklearn.model_selection import ParameterGrid

# LightGBM records synthetic feature names when fitted on an array with an eval_set,
# and sklearn then warns at every predict on an array that has none to compare. One
# message, not the category: the fit and the predictions are unaffected.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn.utils.validation",
)


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
    ic_clean = ic_per_date.drop_nans("ic").drop_nulls("ic")
    return float(ic_clean["ic"].mean()) if ic_clean.height else float("nan")


from utils.cv_splits import load_evaluation_config
from utils.modeling import load_modeling_dataset
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_with_alt

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% [markdown]
# ## Settings
#
# `PARAM_GRID` is the searched space and the only thing that sets what either search costs.
# The grid is exhaustive, so its size is `len(ParameterGrid(PARAM_GRID))`, and section 5 gives
# Optuna that same number of trials because equal budget is what the comparison holds fixed.
# There is deliberately no separate trial count beside it: one would let a caller set a budget
# that is not the grid's and break the comparison without saying so. Shrink the space and both
# searches shrink with it.
#
# `TRIAL_BUDGETS` is the ladder for the overfitting sweep in section 8, which searches the
# continuous space. The grid does not bound that space, so the ladder is declared rather than
# derived from the grid's size.
#
# `MAX_SYMBOLS` caps the universe each fit trains on, and is what a reduced run sets. Zero is
# the full universe.

# %% tags=["parameters"]
MAX_SYMBOLS = 0
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "num_leaves": [15, 31],
}
TRIAL_BUDGETS = [10, 25, 50, 75]
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

# The test set is the case study's holdout, declared in `setup.yaml`. Every
# walk-forward fold ends before it starts, so no search here can reach it.
eval_cfg = load_evaluation_config("etfs")
holdout_start = pd.Timestamp(eval_cfg["holdout_start"])
holdout_end = pd.Timestamp(eval_cfg["holdout_end"])
LABEL_HORIZON = 21  # trading days, matching fwd_ret_21d

# Embargo: a label at date d resolves LABEL_HORIZON trading days later, so the last
# admissible validation date is that many days before the holdout starts. Index
# -(LABEL_HORIZON + 1) is it; -LABEL_HORIZON would land on the holdout's first day.
pre_holdout_dates = np.sort(df.loc[df[date_col] < holdout_start, date_col].unique())
val_embargo_cutoff = pd.Timestamp(pre_holdout_dates[-(LABEL_HORIZON + 1)])

# Fold 0 supplies train and validation, the latter trimmed by the embargo cutoff.
split0 = mds.splits[0]

train_mask = (df[date_col] >= split0["train_start"]) & (df[date_col] <= split0["train_end"])
val_end = min(pd.Timestamp(split0["val_end"]), val_embargo_cutoff)
val_mask = (df[date_col] >= split0["val_start"]) & (df[date_col] <= val_end)
test_mask = (df[date_col] >= holdout_start) & (df[date_col] <= holdout_end)

X_train = df.loc[train_mask, FEATURE_COLS].values
y_train = df.loc[train_mask, mds.label_col].values
X_val = df.loc[val_mask, FEATURE_COLS].values
y_val = df.loc[val_mask, mds.label_col].values
X_test = df.loc[test_mask, FEATURE_COLS].values
y_test = df.loc[test_mask, mds.label_col].values

primary_entity_col = mds.entity_cols[0]
dates_val = df.loc[val_mask, date_col].values
symbols_val = df.loc[val_mask, primary_entity_col].values
dates_test = df.loc[test_mask, date_col].values
symbols_test = df.loc[test_mask, primary_entity_col].values

# Drop NaN labels
valid = np.isfinite(y_train)
X_train, y_train = X_train[valid], y_train[valid]
valid = np.isfinite(y_val)
X_val, y_val = X_val[valid], y_val[valid]
dates_val, symbols_val = dates_val[valid], symbols_val[valid]
valid = np.isfinite(y_test)
X_test, y_test = X_test[valid], y_test[valid]
dates_test, symbols_test = dates_test[valid], symbols_test[valid]

print(f"ETFs: {len(FEATURE_COLS)} features")
print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test (holdout): {len(X_test):,}")
print(f"Holdout window: {holdout_start.date()} → {holdout_end.date()}")


# %% [markdown]
# ## 3. Evaluation Functions


# %%
def evaluate_params(params):
    """Train model and return validation IC."""
    model = lgb.LGBMRegressor(**params, random_state=SEED, verbose=-1, n_jobs=-1)
    model.fit(X_train, y_train)
    return cross_sectional_ic_mean(y_val, model.predict(X_val), dates_val, symbols_val)


# %% [markdown]
# ## 4. Grid Search
#
# `PARAM_GRID` in the settings cell is four parameters with a handful of values each, small
# enough that enumerating it is tractable. Its size is what both searches spend.

# %%
grid = list(ParameterGrid(PARAM_GRID))
N_GRID = len(grid)
print(f"Grid Search: {N_GRID} combinations")

grid_start = time.time()
grid_results = []
for i, params in enumerate(grid):
    ic = evaluate_params(params)
    grid_results.append({"params": params, "ic": ic})
    if (i + 1) % 18 == 0:
        print(f"  {i + 1}/{N_GRID} completed...")

grid_time = time.time() - grid_start
# Select over finite ICs only. `max()` compares with `<`, and every comparison
# against NaN is false, so a NaN anywhere in the list makes it return whichever
# element came first rather than the best one - silently, and with no warning.
scored = [r for r in grid_results if np.isfinite(r["ic"])]
if not scored:
    raise RuntimeError("No grid configuration produced a finite IC.")
best_grid = max(scored, key=lambda x: x["ic"])
print(f"Done in {grid_time:.1f}s, best IC {best_grid['ic']:.4f}")

# %% [markdown]
# ## 5. Optuna (Same Budget, Same Space)
#
# Run Optuna with the same number of trials as grid combinations,
# searching the same categorical parameter space.


# %%
def optuna_objective(trial):
    # Suggested from `PARAM_GRID` itself rather than from a second copy of the values, so
    # "same space" holds by construction instead of by two lists agreeing.
    params = {name: trial.suggest_categorical(name, values) for name, values in PARAM_GRID.items()}
    return evaluate_params(params)


optuna_start = time.time()
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
study.optimize(optuna_objective, n_trials=N_GRID, show_progress_bar=True)
optuna_time = time.time() - optuna_start

print(f"Done in {optuna_time:.1f}s, best IC {study.best_value:.4f}")

# %% [markdown]
# ### Same-Budget Comparison

# %%
comparison_df = pl.DataFrame(
    {
        "method": ["Grid Search", "Optuna (Grid Space)"],
        "time_s": [round(grid_time, 1), round(optuna_time, 1)],
        "best_val_ic": [round(best_grid["ic"], 4), round(study.best_value, 4)],
    }
)
comparison_df

# %% [markdown]
# **How to read this comparison.** Grid evaluates every combination in `PARAM_GRID`, so
# whatever it returns is that space's maximum by construction. Optuna is given the
# same number of trials on the same categorical space, and TPE samples with replacement, so
# those trials do not cover that many distinct points. It therefore cannot beat the
# enumeration here and can only tie it, by happening to sample the maximum - which is not a
# fact about the two samplers. On a space small enough to enumerate there is nothing for a
# sampler to exploit in exchange for the coverage it gives up; search pays where the space is
# too big to enumerate, which is what section 6 tests instead.
#
# The wall-time column is one uncontrolled measurement on a shared machine, and the two
# searches fit different sets of configurations, which cost different amounts to train. It is
# neither a benchmark nor evidence that either method is cheaper.
#
# The chapter's recommendation stands either way: grid is fine for ≤4 parameters × ≤3 values
# each. What the continuous space adds is reach, to configurations a discrete grid cannot
# represent at all - a property of the space, not a measurement of the sampler.

# %% [markdown]
# ## 6. Optuna with Continuous Space
#
# The real power of Bayesian optimization: searching continuous hyperparameter
# spaces that grid search cannot represent.


# %%
def optuna_continuous_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    return evaluate_params(params)


cont_start = time.time()
study_cont = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
study_cont.optimize(optuna_continuous_objective, n_trials=N_GRID, show_progress_bar=True)
cont_time = time.time() - cont_start

print(f"Done in {cont_time:.1f}s, best IC {study_cont.best_value:.4f}")

# %% [markdown]
# ## 7. Efficiency: how many trials reach most of the search's final value

# %%
trials_df = study_cont.trials_dataframe()
best_ic = trials_df["value"].max()
threshold = 0.95 * best_ic if best_ic > 0 else best_ic * 1.05

cummax = trials_df["value"].cummax()
trials_to_95 = int((cummax >= threshold).idxmax()) + 1

print(f"Best IC: {best_ic:.4f}")
print(f"95% threshold: {threshold:.4f}")
print(f"Trials to reach 95% of best: {trials_to_95}")

# %% [markdown]
# ## 8. Validation Overfitting: IC Divergence
#
# Section 12.4's box on validation overfitting warns that a long search fits the
# validation window's noise. This section measures it: for increasing trial budgets,
# the highest validation IC the search reached, against the holdout IC of the
# configuration it selected at that budget.

# %%
overfit_results = []

for budget in TRIAL_BUDGETS:
    sub_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    sub_study.optimize(optuna_continuous_objective, n_trials=budget, show_progress_bar=False)

    # Best validation IC at this budget
    best_val_ic = sub_study.best_value

    # Train best model and evaluate on test
    best_p = {**sub_study.best_params, "random_state": SEED, "verbose": -1, "n_jobs": -1}
    model = lgb.LGBMRegressor(**best_p)
    model.fit(X_train, y_train)
    test_ic = cross_sectional_ic_mean(y_test, model.predict(X_test), dates_test, symbols_test)

    overfit_results.append(
        {"trials": budget, "val_ic": round(best_val_ic, 4), "test_ic": round(test_ic, 4)}
    )
    print(f"  Budget={budget:>3d}: Val IC={best_val_ic:.4f}, Test IC={test_ic:.4f}")

overfit_df = pl.DataFrame(overfit_results)
overfit_df

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    overfit_df["trials"],
    overfit_df["val_ic"],
    "o-",
    color=COLORS["slate"],
    label="Validation IC",
    linewidth=2,
)
ax.plot(
    overfit_df["trials"],
    overfit_df["test_ic"],
    "s--",
    color=COLORS["amber"],
    label="Test IC",
    linewidth=2,
)
ax.set_xlabel("Number of Optuna Trials")
ax.set_ylabel("Cross-sectional IC")
ax.set_title("Validation and holdout IC against the trial budget")
ax.legend()
show_with_alt(
    fig,
    "Two lines against the number of trials in the search: the validation IC the search "
    "reached at each budget, and the holdout IC of the configuration it selected there.",
)

# %% [markdown]
# **How to read the two columns, and why only one of them can say anything.** The
# validation column cannot fall. Each budget builds a fresh study from the same seed, so a
# longer budget repeats the shorter one's trials in the same order before adding any of its
# own, and the value it records is a running maximum over them. Whatever that column does is
# arithmetic. The holdout column has no such constraint - it is the out-of-sample IC of
# whichever configuration the search selected at that budget, and nothing pins it in either
# direction - so it is the only one of the two that carries information about the budget.
#
# What the mechanism predicts: past some budget the extra trials buy fit to the validation
# window's noise and pay for it out of sample. That is the effect Section 12.4's box on
# validation overfitting describes, and the reason a trial budget is a parameter to choose
# rather than to maximize.
#
# The ladder is a diagnostic and not a selection procedure, and it is bounded on one side
# only: it starts at its shortest budget and cannot see below it. It also varies neither the
# seed nor the fold, so it gives a direction without a width. Where the two series sit
# relative to each other in level is a property of the holdout window, not a sign that tuning
# helped. Section 10 asks a different
# question - how the two selected configurations compare after refitting on train plus
# validation - and a budget sweep inside one space is not the same measurement as a
# comparison of two picks from different spaces.

# %% [markdown] tags=["results"]
# Holdout IC is already highest at the shortest budget swept, so the turn this section exists
# to show sits at or below the ladder rather than inside it. Read the figure as evidence that
# the two criteria diverge, not as a way to locate where.

# %% [markdown]
# ## 9. Visualization

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: method comparison
methods = ["Grid Search", "Optuna\n(Grid Space)", "Optuna\n(Continuous)"]
ics = [best_grid["ic"], study.best_value, study_cont.best_value]
colors = [COLORS["slate"], COLORS["copper"], COLORS["amber"]]

ax1 = axes[0]
bars = ax1.bar(methods, ics, color=colors)
for bar, ic in zip(bars, ics, strict=False):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f"{ic:.4f}",
        ha="center",
        fontsize=10,
    )
ax1.set_ylabel("Validation IC")
ax1.set_title("Validation IC of each search's selected configuration")

# Right: convergence
ax2 = axes[1]
ax2.scatter(trials_df["number"], trials_df["value"], s=15, alpha=0.5, color=COLORS["slate"])
ax2.plot(trials_df["number"], cummax, color=COLORS["amber"], linewidth=2, label="Best so far")
ax2.axhline(threshold, linestyle="--", color="gray", linewidth=0.8, label="95% threshold")
ax2.set_xlabel("Trial Number")
ax2.set_ylabel("Validation IC")
ax2.set_title("Validation IC by trial, against the threshold")
ax2.legend(fontsize=9)

show_with_alt(
    fig,
    "Two panels. Left: one bar per search method, the validation IC of the "
    "configuration it selected, each labelled with its value. Right: every trial's "
    "validation IC against its trial number, with a line tracing the best so far and a "
    "dashed line at the threshold used to count trials.",
)

# %% [markdown]
# ## 10. Final Test Set Evaluation

# %%
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

# Grid best
grid_model = lgb.LGBMRegressor(**best_grid["params"], random_state=SEED, verbose=-1)
grid_model.fit(X_trainval, y_trainval)
grid_test_ic = cross_sectional_ic_mean(y_test, grid_model.predict(X_test), dates_test, symbols_test)

# Optuna continuous best
cont_params = {**study_cont.best_params, "random_state": SEED, "verbose": -1}
cont_model = lgb.LGBMRegressor(**cont_params)
cont_model.fit(X_trainval, y_trainval)
cont_test_ic = cross_sectional_ic_mean(y_test, cont_model.predict(X_test), dates_test, symbols_test)

# %%
final_df = pl.DataFrame(
    {
        "method": ["Grid Search", "Optuna (Continuous)"],
        "test_ic": [round(grid_test_ic, 4), round(cont_test_ic, 4)],
    }
)
final_df

# %% [markdown]
# **Interpretation**: This is the load-bearing comparison of the notebook, and it
# is measured on the case study's holdout, which no search here could see. Two
# configurations are compared and two only: the one each search selected on
# validation. Whichever scores higher out of sample, that is a statement about those
# two configurations and not about the spaces they came from, because the rest of the
# grid was never scored on the holdout.
#
# Read it against the budget sweep above rather than on its own. That sweep is drawn to
# separate a different question - *how long* you search within one space - from this one,
# which is which of two spaces the selected configuration came from. Neither licenses
# trusting a validation number on its own: on this ETF target the ICs are thin.

# %% [markdown] tags=["results"]
# **One run of this notebook does not decide grid against Optuna.** An earlier execution of
# this same code, on an earlier vintage of the ETF artifacts, ranked the two in the opposite
# order. That instability is the finding: when validation IC is low and noisy, the search
# method is not what decides the outcome, and the only reliable defence is a holdout the
# search cannot touch, plus walk-forward HPO, demonstrated in `04_optuna_tuning`.

# %% [markdown]
# ## Key Takeaways
#
# ### Grid Search
#
# | Pros | Cons |
# |------|------|
# | Simple, reproducible | Exponential scaling with dimensions |
# | Exhaustive coverage | Fixed, discrete values only |
# | Easy to parallelize | Wastes evaluations in poor regions |
#
# ### Optuna (Bayesian)
#
# | Pros | Cons |
# |------|------|
# | Intelligent sampling (TPE) | More complex setup |
# | Continuous search spaces | Results vary with seed |
# | Adapts during search | Requires more trials for stability |
#
# ### Practical Recommendations
#
# 1. **Grid Search**: Fine for <=4 parameters with <=3 values each
# 2. **Optuna**: Preferred for >4 parameters or continuous ranges
# 3. **Budget**: 50–100 trials is the common convention for GBM tuning, but it is
#    a parameter, not a default. Choose it with nested or walk-forward
#    validation, then score the holdout once. Section 8 reads the holdout
#    across budgets to *show* the two criteria diverging; that is a diagnostic
#    run after the fact, not a way to select the budget
# 4. **Always**: Hold out a test set untouched during optimization
#
# **Next**: See `04_optuna_tuning` for the full Optuna workflow with pruning
# and walk-forward HPO, or `06_optuna_multi_asset` for multi-objective optimization.
