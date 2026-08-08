# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
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
# continuous one it cannot. It also investigates validation overfitting — how
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
# - **Section 12.4**: TPE, pruning, validation overfitting (Box 12.3)
# - **Related**: `04_optuna_tuning` (full workflow), `06_optuna_multi_asset` (multi-objective)
#
# ## References
# - Bergstra & Bengio (2012). "Random Search for Hyper-Parameter Optimization"
# - Akiba et al. (2019). "Optuna: A Next-generation HPO Framework"

# %% [markdown]
# ## 1. Setup

# %%
"""Grid Search vs Optuna — compare HPO methods on convergence speed and final performance."""

import time
import warnings

# lightgbm must be imported before anything that loads scikit-learn, which
# includes ml4t.diagnostic transitively. Both ship their own OpenMP runtime and
# the first one loaded wins for the whole process; on macOS ARM64, getting
# scikit-learn's libomp first makes LightGBM's next multithreaded fit segfault
# in __kmp_suspend_initialize_thread, killing the kernel with no traceback.
# This notebook fits with n_jobs=-1, so it is squarely exposed. Plain `import`
# statements sort ahead of `from ... import` ones, so one canonical block keeps
# this order and isort will not undo it.
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import polars as pl
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from optuna.samplers import TPESampler
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore")


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
from utils.style import COLORS

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% tags=["parameters"]
N_GRID_POINTS = 50
N_OPTUNA_TRIALS = 50
SEED = 42


# %%
set_global_seeds(SEED)

# %% [markdown]
# ## 2. Load Data

# %%
mds = load_modeling_dataset("etfs", "fwd_ret_21d")
df = mds.dataset.to_pandas()
date_col = mds.date_col
FEATURE_COLS = mds.feature_names

# The TEST set is the case study's sealed holdout (setup.yaml `holdout_start` /
# `holdout_end`). All eight walk-forward folds live strictly BEFORE holdout_start,
# so the entire HPO search — grid, Optuna-on-grid, and Optuna-continuous — is
# confined to fold 0's validation window and never touches the holdout. Reporting
# "test IC" on a fold's own validation window (the earlier design here) leaked: the
# fold-1 val window sat ~92% inside fold 0's training window, so the "test" was
# mostly in-sample.
eval_cfg = load_evaluation_config("etfs")
holdout_start = pd.Timestamp(eval_cfg["holdout_start"])
holdout_end = pd.Timestamp(eval_cfg["holdout_end"])
LABEL_HORIZON = 21  # trading days — fwd_ret_21d

# Embargo: trim any validation date whose 21-trading-day label horizon would reach
# into the holdout, so selection never scores on a return realized on or after the
# first holdout day. A fwd_ret_21d label at date d is realized 21 trading days
# later, so the last admissible val date must have at least 21 trading days BEFORE
# holdout_start — index -(LABEL_HORIZON + 1) into the pre-holdout calendar (the
# -LABEL_HORIZON date lands on the first holdout day). Only fold 0, adjacent to
# holdout_start, is affected; earlier folds end well before the cutoff.
pre_holdout_dates = np.sort(df.loc[df[date_col] < holdout_start, date_col].unique())
val_embargo_cutoff = pd.Timestamp(pre_holdout_dates[-(LABEL_HORIZON + 1)])

# Fold 0 (most recent walk-forward fold) supplies train + validation; validation is
# fold 0's val window trimmed by the embargo cutoff above. Test is the sealed holdout.
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
print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test (sealed holdout): {len(X_test):,}")
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
# A small grid (3 x 3 x 3 x 2 = 54 combinations) to keep exhaustive
# search tractable.

# %%
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "num_leaves": [15, 31],
}

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
print(f"Done in {grid_time:.1f}s — Best IC: {best_grid['ic']:.4f}")

# %% [markdown]
# ## 5. Optuna (Same Budget, Same Space)
#
# Run Optuna with the same number of trials as grid combinations,
# searching the same categorical parameter space.


# %%
def optuna_objective(trial):
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
        "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7]),
        "num_leaves": trial.suggest_categorical("num_leaves", [15, 31]),
    }
    return evaluate_params(params)


optuna_start = time.time()
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
study.optimize(optuna_objective, n_trials=N_GRID, show_progress_bar=True)
optuna_time = time.time() - optuna_start

print(f"Done in {optuna_time:.1f}s — Best IC: {study.best_value:.4f}")

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
# **Interpretation**: On this small discrete grid, exhaustion is the safer method.
# Grid ran every one of the 54 combinations, so it found the global best by
# construction. Optuna, given the same 54-trial budget on the same categorical
# space, came in below it: TPE samples with replacement, so 54 trials do not cover
# 54 points, and on a space this small there is nothing for the sampler to exploit
# in exchange. Search only pays where the space is too big to enumerate. The
# wall-time column is one uncontrolled measurement on a shared machine, and the
# two searches fit different sets of configurations, which cost different amounts
# to train — so it is neither a benchmark nor evidence that either method is
# cheaper. The lesson is not that one method wins but that on a space this small
# there is nothing to optimize: the chapter's recommendation stands — grid is
# fine for ≤4 parameters × ≤3 values each. What the continuous space tested next
# adds is reach, to configurations a discrete grid cannot represent at all; that
# is a property of the space, and it is not a measurement of the sampler.

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

print(f"Done in {cont_time:.1f}s — Best IC: {study_cont.best_value:.4f}")

# %% [markdown]
# ## 7. Efficiency Analysis: Trials to 95% of Best

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
# Box 12.3 warns that excessive tuning overfits validation noise. We test
# this empirically: for increasing trial budgets, compare the best validation
# IC against the test IC of the corresponding model.

# %%
trial_budgets = [10, 25, 50, 75, N_GRID]
overfit_results = []

for budget in trial_budgets:
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
ax.set_title("Extra trials inflate validation IC far more than held-out test IC")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation**: The gap between the two curves is the point. Read the table
# above by column: best *validation* IC rises with every increase in budget and
# then plateaus, while *test* IC on the sealed 2024–2025 holdout peaks partway
# through and then falls. Past that peak the extra trials are buying fit to the
# validation window's noise and paying for it out of sample — the signature Box
# 12.3 warns about, and the reason a budget is a parameter to choose rather than
# maximize. (The test IC sits *above* the val IC in
# absolute terms only because the 2024–2025 holdout was an unusually strong,
# trending regime for this ETF universe — a level artifact of that window, not
# evidence that tuning helped.) This single-fold, fixed-seed setup shows the
# *direction* of overfitting but is a blunt instrument for quantifying it, because
# it lacks trial-level variance. Section 10 asks a different question — how the two
# selected configurations compare after refitting on `train + val` — and does not
# reproduce this curve's shape, which is the honest state of the evidence rather
# than a contradiction: a budget sweep within one space and a comparison of two
# picks from different spaces are not the same experiment. For a higher-resolution
# view of the effect this section does show, `04_optuna_tuning` shows how
# single-fold HPO collapses to a one-tree model on a different fold.

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
ax1.set_title("The continuous space reaches a higher validation IC than either discrete search")

# Right: convergence
ax2 = axes[1]
ax2.scatter(trials_df["number"], trials_df["value"], s=15, alpha=0.5, color=COLORS["slate"])
ax2.plot(trials_df["number"], cummax, color=COLORS["amber"], linewidth=2, label="Best so far")
ax2.axhline(threshold, linestyle="--", color="gray", linewidth=0.8, label="95% threshold")
ax2.set_xlabel("Trial Number")
ax2.set_ylabel("Validation IC")
ax2.set_title(f"TPE reaches 95% of its best validation IC by trial {trials_to_95}")
ax2.legend(fontsize=9)

plt.tight_layout()
plt.show()

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
# is measured on the case study's **sealed 2024–2025 holdout** — never seen during
# any search. Two configurations are compared, and two only: the one each search
# selected on validation. The continuous selection beats the discrete one out of
# sample here. That says the configuration Optuna picked was better, not that the
# continuous space is better than the grid — the rest of the grid was never scored
# on the holdout, so this measures the picks, not the spaces.
#
# Read it against the budget sweep above rather than on its own. *How long* you
# search within a space stops helping and starts hurting, which is what the test
# curve turning down shows. Neither result licenses trusting the validation number
# itself — on this ETF target every IC here is thin.
#
# **Do not read a single run of this notebook as a verdict on grid versus Optuna.**
# An earlier execution of this same code, on an earlier vintage of the ETF
# artifacts, ranked them the other way round. That instability is the finding: when
# validation IC is low and noisy, the search method is not what decides the outcome,
# and the only reliable defence is a holdout the search cannot touch — plus
# walk-forward HPO, demonstrated in `04_optuna_tuning`.

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
#    validation, then score the holdout once. Section 8 reads the sealed holdout
#    across budgets to *show* that the turn exists; that is a diagnostic run
#    after the fact, not a way to select the budget
# 4. **Always**: Hold out a test set untouched during optimization
#
# **Next**: See `04_optuna_tuning` for the full Optuna workflow with pruning
# and walk-forward HPO, or `06_optuna_multi_asset` for multi-objective optimization.
