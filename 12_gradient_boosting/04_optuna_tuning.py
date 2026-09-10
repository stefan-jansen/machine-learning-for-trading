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
# # Hyperparameter Tuning with Optuna
#
# **Docker image**: `ml4t`
#
# **Chapter 12, Section 12.4**: Advanced Hyperparameter Tuning with Optuna
#
# ## Purpose
# This notebook demonstrates efficient hyperparameter optimization using Optuna's
# Bayesian optimization framework with TPE on the ETF case study. It covers
# single-fold tuning with pruning and early stopping, then extends to averaged
# walk-forward HPO, which is the approach Section 12.4 recommends for financial data.
#
# ## Learning Objectives
# After completing this notebook, you will be able to:
# - Apply Optuna's define-by-run API to tune LightGBM hyperparameters
# - Use early stopping and MedianPruner to reduce wasted computation
# - Implement time-series-aware tuning with averaged walk-forward evaluation
# - Interpret hyperparameter importance rankings
# - Compare default vs tuned model performance on held-out data
#
# ## Cross-References
# - **Section 12.4**: TPE, pruning, GBM tuning strategy, time-series-aware tuning
# - **Related**: `07_hpo_comparison` (grid vs Optuna), `06_optuna_multi_asset` (multi-objective)

# %% [markdown]
# ## 1. Setup

# %%
"""Hyperparameter Tuning with Optuna - TPE-based optimization with pruning for GBMs."""

import time
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

# LightGBM records synthetic feature names when fitted on an array with an eval_set,
# and sklearn then warns at every predict on an array that has none to compare. One
# message, not the category: the fit and the predictions are unaffected.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn.utils.validation",
)

import lightgbm as lgb
import optuna
from lightgbm import LGBMRegressor
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


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


from utils.cv_splits import load_evaluation_config
from utils.modeling import load_modeling_dataset
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_with_alt

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% tags=["parameters"]
N_TRIALS = 50
# 0 = all folds for walk-forward HPO
MAX_FOLDS = 0
SEED = 42


# %%
set_global_seeds(SEED)
# %% [markdown]
# ## 2. Load ETF Features

# %%
mds = load_modeling_dataset("etfs", "fwd_ret_21d")
df = mds.dataset.to_pandas()
date_col = mds.date_col
FEATURE_COLS = mds.feature_names

n_folds = len(mds.splits)
if MAX_FOLDS > 0:
    n_folds = min(n_folds, MAX_FOLDS)

# The test set is the case study's holdout, declared in `setup.yaml`; every
# walk-forward fold ends before it starts, so no part of the search sees it.
eval_cfg = load_evaluation_config("etfs")
holdout_start = pd.Timestamp(eval_cfg["holdout_start"])
holdout_end = pd.Timestamp(eval_cfg["holdout_end"])
LABEL_HORIZON = 21  # trading days, matching fwd_ret_21d

# Embargo: a label at date d resolves LABEL_HORIZON trading days later, so the last
# admissible validation date is that many days before the holdout starts. Index
# -(LABEL_HORIZON + 1) is it; -LABEL_HORIZON would land on the holdout's first day.
pre_holdout_dates = np.sort(df.loc[df[date_col] < holdout_start, date_col].unique())
val_embargo_cutoff = pd.Timestamp(pre_holdout_dates[-(LABEL_HORIZON + 1)])

# Single-fold demo uses the most recent walk-forward fold (fold 0); validation is
# fold 0's val window trimmed by the embargo cutoff above.
split0 = mds.splits[0]

train_mask = (df[date_col] >= split0["train_start"]) & (df[date_col] <= split0["train_end"])
val_end = min(pd.Timestamp(split0["val_end"]), val_embargo_cutoff)
val_mask = (df[date_col] >= split0["val_start"]) & (df[date_col] <= val_end)
test_mask = (df[date_col] >= holdout_start) & (df[date_col] <= holdout_end)

primary_entity_col = mds.entity_cols[0]

X_train = df.loc[train_mask, FEATURE_COLS].values
y_train = df.loc[train_mask, mds.label_col].values
X_val = df.loc[val_mask, FEATURE_COLS].values
y_val = df.loc[val_mask, mds.label_col].values
X_test = df.loc[test_mask, FEATURE_COLS].values
y_test = df.loc[test_mask, mds.label_col].values
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

print(f"ETFs: {len(FEATURE_COLS)} features, N_TRIALS: {N_TRIALS}")
print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test (holdout): {len(X_test):,}")
print(f"Holdout window: {holdout_start.date()} → {holdout_end.date()}")
print(f"Walk-forward folds available: {len(mds.splits)} (using {n_folds})")


# %% [markdown]
# ## 3. LightGBM Hyperparameters: What to Tune
#
# See Section 12.4 for a detailed discussion of parameter families and their
# effects. The key insight: **regularization parameters often have the largest
# impact** on out-of-sample performance in low signal-to-noise regimes.
#
# The ranges are the `suggest_*` calls in the objective below, so this table says what
# each parameter does rather than repeating a bound that can drift away from the code.
#
# ### Structure
#
# | Parameter | Effect |
# |-----------|--------|
# | `num_leaves` | Tree complexity: more leaves fit more and overfit sooner |
# | `learning_rate` | Step size; keep it low and let early stopping find the rounds |
# | `max_depth` | A second constraint on leaf-wise growth |
# | `min_child_samples` | Minimum rows behind a leaf; higher smooths the fit |
#
# ### Regularization
#
# | Parameter | Effect |
# |-----------|--------|
# | `reg_alpha` (L1) | Lasso penalty on leaf weights |
# | `reg_lambda` (L2) | Ridge penalty on leaf weights |
# | `subsample` | Row sampling per tree |
# | `colsample_bytree` | Column sampling per tree |

# %% [markdown]
# ## 4. Define Objective with Early Stopping and Pruning
#
# Optuna's **define-by-run** API defines the search space dynamically within the
# objective function. We add two efficiency mechanisms:
#
# - **Early stopping**: LightGBM monitors validation loss and stops adding trees
#   when performance plateaus, so we set `n_estimators` high and let the callback
#   determine the actual count.
# - **Pruning**: Optuna's `MedianPruner` terminates trials that fall below the
#   median validation IC at the same boosting step. Because the off-the-shelf
#   `optuna_integration.LightGBMPruningCallback` only supports loss-style metrics
#   (i.e., minimization), we use a small custom callback that reports
#   cross-sectional IC every `report_every` boosting rounds.


# %%
class ICPruningCallback:
    """LightGBM callback that reports validation IC to an Optuna trial.

    Reports every `report_every` rounds (predicting after every round is
    expensive). Honors the study's MAXIMIZE direction: `should_prune()` fires
    when the trial's reported IC is below the running median.
    """

    def __init__(self, trial, X_eval, y_eval, dates_eval, symbols_eval, report_every=20):
        self.trial = trial
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.dates_eval = dates_eval
        self.symbols_eval = symbols_eval
        self.report_every = report_every

    def __call__(self, env):
        if (env.iteration + 1) % self.report_every != 0:
            return
        y_pred = env.model.predict(self.X_eval)
        ic = cross_sectional_ic_mean(self.y_eval, y_pred, self.dates_eval, self.symbols_eval)
        if not np.isfinite(ic):
            return
        self.trial.report(ic, step=env.iteration)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# %%
def objective(trial: optuna.Trial) -> float:
    """Optuna objective with early stopping and IC-based pruning."""
    params: dict[str, Any] = {
        "n_estimators": 500,  # a ceiling; early stopping finds the count
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": -1,
    }

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=0),
        ICPruningCallback(trial, X_val, y_val, dates_val, symbols_val, report_every=20),
    ]

    model = LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )

    y_pred = model.predict(X_val)
    ic = cross_sectional_ic_mean(y_val, y_pred, dates_val, symbols_val)
    if not np.isfinite(ic):
        # An undefined IC is not a score of minus one, which would be a perfect inverse
        # ranking and a strong signal. It means no validation date had enough distinct
        # predictions to rank, so the trial produced no value: that is a pruned trial.
        raise optuna.TrialPruned
    return ic


# %% [markdown]
# ## 5. Run Optimization Study
#
# TPE (Tree-structured Parzen Estimator) maintains density estimators for good
# and poor hyperparameters, concentrating evaluations in promising regions.
# The `MedianPruner` terminates trials that underperform the median at each
# boosting step.

# %%
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=SEED),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)

start_time = time.time()
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
study_time = time.time() - start_time

n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

print(f"Best trial: #{study.best_trial.number}, IC: {study.best_value:.4f}")
print(f"Completed: {n_complete}, Pruned: {n_pruned} ({100 * n_pruned / N_TRIALS:.0f}%)")
print(f"Wall time: {study_time:.1f}s")

# %% [markdown]
# **Pruning effectiveness**: Trials pruned early free compute budget for more
# promising configurations. The pruning rate and wall-time savings depend on
# the signal-to-noise ratio: noisier objectives prune more aggressively.

# %% [markdown]
# ## 6. Best Hyperparameters

# %%
best_params_df = pl.DataFrame(
    [
        {"parameter": k, "value": f"{v:.4f}" if isinstance(v, float) else str(v)}
        for k, v in study.best_params.items()
    ]
)
best_params_df

# %% [markdown]
# ## 7. Compare Default vs Tuned Model

# %%
# Default LightGBM
default_model = LGBMRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=SEED, verbose=-1
)
default_model.fit(X_train, y_train)

# Tuned model (re-train with early stopping on val set)
tuned_params: dict[str, Any] = {
    **study.best_params,
    "n_estimators": 500,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
}
tuned_model = LGBMRegressor(**tuned_params)
tuned_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
)

# %%
comparison = pl.DataFrame(
    {
        "model": ["Default", "Tuned (Optuna)"],
        "val_ic": [
            round(
                cross_sectional_ic_mean(
                    y_val, default_model.predict(X_val), dates_val, symbols_val
                ),
                4,
            ),
            round(
                cross_sectional_ic_mean(y_val, tuned_model.predict(X_val), dates_val, symbols_val),
                4,
            ),
        ],
        "test_ic": [
            round(
                cross_sectional_ic_mean(
                    y_test, default_model.predict(X_test), dates_test, symbols_test
                ),
                4,
            ),
            round(
                cross_sectional_ic_mean(
                    y_test, tuned_model.predict(X_test), dates_test, symbols_test
                ),
                4,
            ),
        ],
        "n_trees": [default_model.n_estimators, tuned_model.best_iteration_],
    }
)
comparison

# %% [markdown]
# **Interpretation**: the table above holds the comparison, and the column that matters
# is `n_trees`. Where the tuned configuration early-stops after a single boosting round,
# its predictions are nearly flat and whatever validation IC it earned rests on rank
# differences too small to mean much; a holdout number from such a model is not evidence
# that tuning worked, whichever way it lands.
#
# That is the case Section 12.4's box on validation overfitting makes. One validation window barely
# constrains a search over eight hyperparameters, so the configuration it selects is
# partly a fit to that window's noise, and the holdout can flatter or punish it at
# random. Section 10 repeats the search with the objective averaged across walk-forward
# folds, which is the cheapest thing that changes the answer.

# %% [markdown]
# ## 8. Optimization History

# %%
trials_df = study.trials_dataframe()
completed = trials_df[trials_df["state"] == "COMPLETE"].copy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: trial scatter + best-so-far
ax1 = axes[0]
ax1.scatter(
    completed["number"], completed["value"], s=20, alpha=0.5, color=COLORS["slate"], label="Trials"
)
best_so_far = completed["value"].cummax()
ax1.plot(completed["number"], best_so_far, color=COLORS["amber"], linewidth=2, label="Best so far")
ax1.set_xlabel("Trial Number")
ax1.set_ylabel("Validation IC")
ax1.set_title("Validation IC by trial, with the best so far")
ax1.legend()

# Right: pruned vs completed
ax2 = axes[1]
states = ["Completed", "Pruned"]
counts = [n_complete, n_pruned]
colors = [COLORS["slate"], COLORS["silver_muted"]]
ax2.bar(states, counts, color=colors)
ax2.set_ylabel("Count")
ax2.set_title("Trials completed and pruned")
for i, c in enumerate(counts):
    ax2.text(i, c + 0.5, str(c), ha="center", fontweight="bold")

show_with_alt(
    fig,
    "Two panels. Left: each trial's validation IC against its trial number, with a line "
    "tracing the best value reached so far. Right: two bars, the number of trials that "
    "completed and the number pruned, each labelled with its count.",
)

# %% [markdown]
# ## 9. Hyperparameter Importance

# %%
importance = optuna.importance.get_param_importances(study)

fig, ax = plt.subplots(figsize=(8, 5))
params_sorted = list(importance.keys())
values_sorted = list(importance.values())
ax.barh(params_sorted, values_sorted, color=COLORS["slate"])
ax.set_xlabel("Importance (fANOVA)")
ax.set_title("Hyperparameter importance for the validation objective")
ax.invert_yaxis()
show_with_alt(
    fig,
    "Horizontal bars of fANOVA importance, one per tuned hyperparameter, ordered from "
    "the largest share of the objective's variance down.",
)

# %% [markdown]
# **Interpretation**: fANOVA importance decomposes the variance of the validation
# objective across the hyperparameters, and the chart above shows how concentrated that
# decomposition is here. Where one parameter takes almost all of it, the others' order
# among themselves is noise, and the reading is that most configurations early-stop
# before the leaf-weight penalties or the sampling fractions get to matter. That is a
# statement about this study, not a law: with a signal
# this weak the importance surface is itself noisy, and Section 12.4's general
# guidance still holds. Fix the learning rate low and let Optuna trade off tree
# structure against regularization.

# %% [markdown]
# ## 10. Time-Series-Aware Tuning: Averaged Walk-Forward HPO
#
# The single-fold study above may overfit to one market period. **Averaged
# walk-forward HPO** evaluates each trial across multiple temporal windows,
# returning the mean IC as the objective. This is the approach Section 12.4
# recommends as the default for financial data.
#
# The computational cost is proportional to the number of folds, but Optuna's
# pruning partially offsets this by terminating weak trials early.


# %%
def prepare_fold_data(fold_idx):
    """Prepare train/val arrays for a walk-forward fold."""
    split = mds.splits[fold_idx]
    v_end = min(pd.Timestamp(split["val_end"]), val_embargo_cutoff)
    train_m = (df[date_col] >= split["train_start"]) & (df[date_col] <= split["train_end"])
    val_m = (df[date_col] >= split["val_start"]) & (df[date_col] <= v_end)

    X_tr = df.loc[train_m, FEATURE_COLS].values
    y_tr = df.loc[train_m, mds.label_col].values
    X_va = df.loc[val_m, FEATURE_COLS].values
    y_va = df.loc[val_m, mds.label_col].values
    dates_va = df.loc[val_m, date_col].values
    symbols_va = df.loc[val_m, primary_entity_col].values

    v = np.isfinite(y_tr)
    X_tr, y_tr = X_tr[v], y_tr[v]
    v = np.isfinite(y_va)
    X_va, y_va = X_va[v], y_va[v]
    dates_va, symbols_va = dates_va[v], symbols_va[v]
    return X_tr, y_tr, X_va, y_va, dates_va, symbols_va


# Pre-load all fold data to avoid repeated I/O
# %% [markdown]
# A date is scored only when at least `IC_MIN_OBS` names are priced on it, so a fold
# whose validation window never reaches that width cannot be scored by any
# configuration at all. Those folds come out here, before the search, which keeps the
# set of scored folds a property of the data. Every trial is then scored on the same
# folds, which is what makes two trial values comparable, and a configuration that
# cannot rank one of them has no value rather than a partial one.

# %%
IC_MIN_OBS = 10


def fold_is_scorable(dates_va, min_obs=IC_MIN_OBS):
    """Whether any validation date in this fold carries enough names to rank."""
    per_date = pl.DataFrame({"timestamp": dates_va}).group_by("timestamp").len()
    return bool((per_date["len"] >= min_obs).any())


all_folds = [prepare_fold_data(i) for i in range(n_folds)]
fold_data = [fold for fold in all_folds if fold_is_scorable(fold[4])]
if len(fold_data) < len(all_folds):
    print(
        f"Dropped {len(all_folds) - len(fold_data)} of {len(all_folds)} folds: no "
        f"validation date carries {IC_MIN_OBS} names, so no configuration could be "
        "scored on them."
    )
print(f"Scoring the averaged objective on {len(fold_data)} walk-forward folds")


# %%
def walkforward_objective(trial: optuna.Trial) -> float:
    """Averaged walk-forward objective: mean IC across temporal folds."""
    params: dict[str, Any] = {
        "n_estimators": 500,
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": -1,
    }

    ics = []
    for X_tr, y_tr, X_va, y_va, dates_va, symbols_va in fold_data:
        model = LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
        )
        ic = cross_sectional_ic_mean(y_va, model.predict(X_va), dates_va, symbols_va)
        if not np.isfinite(ic):
            # Same folds for every trial, or no value at all: averaging whichever folds
            # a configuration managed to rank would score each trial on its own set.
            raise optuna.TrialPruned
        ics.append(ic)

    return float(np.mean(ics))


# %%
wf_study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=SEED),
)

start_time = time.time()
wf_study.optimize(walkforward_objective, n_trials=N_TRIALS, show_progress_bar=True)
wf_time = time.time() - start_time

print(f"Walk-forward HPO: best mean IC = {wf_study.best_value:.4f}")
print(f"Wall time: {wf_time:.1f}s ({wf_time / study_time:.1f}x single-fold)")

# %% [markdown]
# ### Compare Single-Fold vs Walk-Forward Tuning
#
# The acid test: evaluate both sets of tuned hyperparameters on the held-out
# test fold to see which generalizes better.

# %%
# Single-fold tuned model (already trained above)
single_test_ic = cross_sectional_ic_mean(
    y_test, tuned_model.predict(X_test), dates_test, symbols_test
)

# Walk-forward tuned model
wf_params: dict[str, Any] = {
    **wf_study.best_params,
    "n_estimators": 500,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
}
wf_model = LGBMRegressor(**wf_params)
wf_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
)
wf_test_ic = cross_sectional_ic_mean(y_test, wf_model.predict(X_test), dates_test, symbols_test)

tuning_comparison = pl.DataFrame(
    {
        "method": ["Single-fold HPO", "Walk-forward HPO"],
        "best_val_ic": [round(study.best_value, 4), round(wf_study.best_value, 4)],
        "test_ic": [round(single_test_ic, 4), round(wf_test_ic, 4)],
        "wall_time_s": [round(study_time, 1), round(wf_time, 1)],
    }
)
tuning_comparison

# %% [markdown]
# **Interpretation**: the two objectives and the two holdout numbers do not tell the
# same story, and why they differ is worth more than either number. A fold whose
# cross-sectional IC is undefined carries no information about the hyperparameters that
# produced it: a near-constant prediction has no ranking to correlate. Two different
# things follow, and the notebook keeps them apart. A fold too narrow for any
# configuration to rank is a property of the data, so it is removed before the search
# and every trial then faces the same folds. A fold that this particular configuration
# could not rank is a property of the trial, so the trial has no value and is pruned.
# Scoring either case at minus one would enter a perfect inverse ranking into the
# average, which on four folds moves the mean by a quarter.
#
# What is left is the shape Section 12.4 warns about. The margins between tuned and
# untuned on the holdout are small, and the walk-forward search costs many times the
# single-fold one; the timing line above says how many. Averaging across folds is the
# right instinct because it stops one window's noise from choosing the configuration,
# not because it can manufacture signal in a target this weak.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Let the data set the tree count, and let the pruner spend the budget.** A high
#    `n_estimators` with `lgb.early_stopping` picks the count per configuration, and
#    `MedianPruner`, driven here by a custom IC callback, stops trials that are behind
#    at an intermediate checkpoint. The figure above says how many trials that was.
#
# 2. **Score the search and score the result on different data.** The test set is the
#    case study's declared holdout and every walk-forward fold ends before it starts, so
#    no trial could see it. That is what makes the holdout column readable at all; the
#    margins it shows are thin, which is what makes it worth reading carefully.
#
# 3. **A single validation window rewards degenerate models.** The single-fold search
#    selected a configuration that early-stops after one tree, whose predictions are
#    nearly flat. Its objective value was real and its meaning was not. Across folds the
#    same family produces undefined ICs on some folds, which is the same fact from the
#    other side.
#
# 4. **An undefined score is not a bad score, and it is not a smaller sample either.**
#    How a search treats a fold it cannot score decides what it selects. Scoring the
#    fold at the worst possible value teaches the sampler to avoid a region for a reason
#    nothing measured. Averaging over the folds a configuration did manage scores every
#    trial on its own set, which rewards ranking one easy fold and predicting a constant
#    everywhere else. What is left is to score every trial on the same folds, and to
#    treat a trial that cannot as having no value.
#
# 5. **The trial budget is a parameter, and Section 12.4 gives its range.** The book
#    suggests starting in the low hundreds and warns that beyond that the marginal gain
#    shrinks while the validation-overfitting risk grows. `N_TRIALS` in the parameters
#    cell is what this run used, which is smaller so the notebook stays runnable.
#
# **Next**: See `07_hpo_comparison` for grid search vs Optuna efficiency,
# or `06_optuna_multi_asset` for multi-objective IC vs turnover optimization.
