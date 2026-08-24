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
# # NASDAQ-100 microstructure: what trees find in order flow that a linear map cannot
#
# [`06_linear`](06_linear.ipynb) fitted a penalty grid to the microstructure features and found a
# few thousandths of rank correlation at the fifteen-minute horizon. A linear model can only
# represent a relationship it is handed in the right shape: it sees an interaction between spread
# and imbalance only because someone multiplied those columns together and named the product.
# Gradient boosting has no such restriction. It builds a sequence of shallow trees, each fitted to
# what the ones before it got wrong, and a tree splits on one feature inside a region defined by
# others - so an interaction is something it discovers rather than something it must be given.
#
# There is a specific interaction to expect here, and it is the reason this notebook is worth
# running rather than assumed. Order-flow imbalance is not equally informative at every spread:
# when the book is tight, a lopsided flow moves the price; when it is wide, the same imbalance may
# be liquidity provision that reverts. A linear coefficient on imbalance has to be one number for
# both states. A tree can split on spread first and read imbalance differently on each side.
#
# Against that, the same collinearity that made a dense penalty the right answer in `06_linear` is
# hostile to a greedy splitter. Faced with several near-identical columns, a tree picks one of them
# at each split and the choice is close to arbitrary; ridge's advantage came precisely from *not*
# choosing. Whether the interaction is worth more than the instability costs is what the grid
# measures.
#
# Three dials control how far a boosted fit goes, and this notebook varies all three:
#
# - **Capacity**, set by `num_leaves`: how many regions one tree may carve the feature space into.
#   Seven leaves express a handful of conditions; 63 can express a partition fine enough to
#   describe the training window and nothing beyond it.
# - **The loss function**, which decides what "got wrong" means. `mse` minimizes squared error,
#   `mae` absolute error, and `huber` behaves like squared error for small residuals and like
#   absolute error past a threshold derived from each fold's own label spread. At a fifteen-minute
#   horizon almost all of the return is noise with occasional large jumps, so how much the tails
#   are allowed to steer the fit is a live question rather than a detail.
# - **When to stop**, set by the number of trees. Unlike a linear fit, a boosted model has a
#   meaningful state at every iteration, so each configuration is scored at ten points along its
#   own training run rather than only at the end.
#
# The third dial changes how the results must be read. **A checkpoint is part of a configuration,
# not a detail of how it was fitted.** Scoring a grid at ten checkpoints each multiplies the
# candidate count by ten, and treating that as one candidate per configuration while quietly
# keeping each one's best iteration would be reporting the maximum of ten numbers as though it
# were one.
#
# **This notebook fits every declared label in one run**, at three horizons and in the
# classification form of the primary one. The horizon comparison is the part that generalizes: a
# grid that ranks one way at five minutes and another way at sixty is telling you something about
# the horizon, not about the models.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Read a declared gradient boosting grid and say what each configuration varies.
# - Explain why a boosted model produces one result per checkpoint while a linear model produces
#   one result in total, and what that implies for counting candidates.
# - Read a learning curve of out-of-sample information coefficient against tree count, and tell
#   apart a model still learning from one that has begun fitting the training window.
# - Say why the choice of loss function is a statement about the label's tails, and relate that to
#   what a rank-based metric rewards.
# - Compare the same grid across three prediction horizons and say what a reversal between them
#   rules out.
# - Recognise that picking each configuration's best checkpoint after seeing the results is a
#   selection decision, and locate where selection is actually made.
#
# **Docker image**: `ml4t`
#
# **Book reference**: Chapter 12, Section 12.2 (GBM libraries) and Section 12.3 (how to tune a
# boosted model). Chapter 6, Section 6.7 (Search accounting and run logging) introduces the run
# log this notebook writes to.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices,
# [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds, and
# [`06_linear`](06_linear.ipynb) fitted the linear population this one is compared against.
#
# **What it writes**: one training run per configuration and one complete validation prediction
# set per configuration and checkpoint, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a named population.
# [`14_backtest`](14_backtest.ipynb) reads that population and selects on validation backtest
# Sharpe. **Selection happens there, not here.**

# %%
"""Fit the declared NASDAQ-100 microstructure gradient boosting population on the folds."""

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from case_studies.research import (
    declared_labels,
    load_model_configs,
    model_requests,
    narrows_declared_catalog,
    open_study,
    plan_models,
    planned_model_plan,
    primary_label,
    run_model_population,
)
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
LABELS: list[str] = []
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
PREVIEW_REDUCTIONS: dict = {}
CONFIG_NAMES: list[str] = []
POPULATION_NAME = ""
SUPERSEDES_POPULATION: str = ""

# %%
study = open_study(
    "nasdaq100_microstructure", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None
)

# %% [markdown]
# ## 1. Which labels, and which models
#
# A label is the thing being predicted. This case study declares four in `config/setup.yaml`.
# `fwd_ret_15m`, the return over the fifteen minutes after the decision minute, is the primary one
# - the horizon the strategy chapters trade. `fwd_ret_5m` and `fwd_ret_60m` are the same
# construction at shorter and longer horizons, and `fwd_dir_15m` is the classification form of the
# primary label.
#
# **`LABELS = []` fits all of them in one run**, which is what the empty list means: not "no
# labels" but "every label whose training menu declares this family". Each label carries its own
# menu at `config/training/{label}.yaml`, so the grid is not necessarily the same on each - the
# classification label declares multiclass objectives and nothing else, because squared error on
# a categorical target is not a thing to fit.

# %%
declared_labels(study, "gbm")

# %% [markdown]
# Each name in a menu resolves to a preset in `case_studies/config/lgb/`, which holds that
# configuration's complete LightGBM parameter set. The regression grids are a product of two axes:
#
# - **Five profiles.** Four of them fix `num_leaves` at 7, 15, 31 and 63 and otherwise share one
#   regularized setting: learning rate 0.05, 80% of rows and 70% of columns sampled per tree, L1
#   0.5, L2 5.0, and at least 50 observations behind a leaf. The fifth, `default`, is LightGBM out
#   of the box: 31 leaves, learning rate 0.1, no sampling and no penalty.
# - **Three objectives**, as described above.
#
# Only the four `leaves_*` profiles vary one thing at a time. `default` carries the same leaf count
# as `leaves_31` and differs from it on the learning rate and on every regularization setting, so
# that pair measures nothing about capacity; read `default` as the unconfigured baseline and the
# `leaves_*` row as the capacity axis. All fifteen run 500 boosting iterations and publish a
# checkpoint every 50. To change what runs, edit the menu or the presets rather than this notebook.

# %%
configs = load_model_configs(
    study,
    "gbm",
    labels=LABELS or None,
    config_names=CONFIG_NAMES or None,
)
configs

# %% [markdown]
# `LABELS` and `CONFIG_NAMES` both narrow what is fitted, and a narrowed run declares a different
# population from the canonical one. Publishing it under the canonical name would leave that name
# meaning two different member sets at two different times, so the guard below requires a run that
# narrows anything to say what to call its population.

# %%
if narrows_declared_catalog(study, "gbm", configs) and not POPULATION_NAME:
    raise ValueError(
        f"this run fits {configs.height} of the declared configurations, so it cannot publish "
        "the canonical population; pass POPULATION_NAME to give it its own"
    )

# %% [markdown]
# ## 2. Binding the declarations to the data
#
# A menu entry says which estimator to fit. It does not say which feature columns exist today,
# where the walk-forward folds fall, or which symbol-timestamp pairs carry both a feature row and
# a label. **Planning** works all of that out - every training and prediction identity, derived
# from the declarations and the fold boundaries in `config/setup.yaml` - without holding the data
# it was derived from.
#
# The distinction matters here more than anywhere else in the book. A *resolved* request carries
# its prepared folds, and this case study is a minute panel: holding one resolved request per
# configuration means holding the same design matrix once per configuration. Planning prices the
# whole panel before a single fit and keeps nothing but the plan. Execution then walks folds on
# the outside and configurations on the inside, so one prepared fold is live at a time however
# many configurations were declared.
#
# The plan is the population, written down before anything is fitted. Four things to check in the
# table below:
#
# - **`feature_count` and `eligible_rows` agree within a label.** A row that differs is a
#   configuration measured on a different sample from its neighbours.
# - **`folds` is the same everywhere**, and equals the number of walk-forward splits.
# - **`validation_start` and `validation_end` bracket the development sample**, with none of the
#   held-out tail visible.
# - **`checkpoints` is where this differs from the linear plan.** It is the number of training
#   states each configuration publishes predictions for. Multiply it by the number of rows to get
#   the number of candidate models this notebook is about to create.
#
# `eligible_rows` differs *between* labels and should: a sixty-minute forward window runs out
# earlier than a five-minute one, so the longer horizon has fewer scorable decision times to begin
# with.

# %%
requests = model_requests(
    study,
    configs,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
)
plan = plan_models(study, requests=requests)

planned = planned_model_plan(plan)
planned.select(
    "label",
    "config_name",
    "task",
    "feature_count",
    "eligible_rows",
    "folds",
    "checkpoints",
    "validation_start",
    "validation_end",
)

# %% [markdown]
# ## 3. Fitting the population
#
# `run_model_population` fits every planned request. For one request it walks the folds, and on
# each one:
#
# 1. takes the rows inside that fold's training window,
# 2. casts the design matrix to the precision LightGBM works in and leaves missing values in
#    place - a tree routes a missing value down its own branch, so imputing a median here would
#    hand the model an observation nobody made,
# 3. fits the declared number of boosting iterations,
# 4. predicts the fold's validation rows at each checkpoint, using only the trees built up to that
#    iteration.
#
# Step 4 is what makes one fit produce many results. The fold predictions are concatenated into
# one series per checkpoint covering the whole validation period, and each becomes its own
# registered prediction set with its own identity.
#
# **How many bins the features are quantized into is part of the model, not a property of the
# machine.** LightGBM buckets each feature's values before considering any split, and the same
# data at 63 bins and at 255 produces different trees. `config/setup.yaml` declares `max_bin`
# explicitly for that reason. Reading it off whichever device happened to be visible - the GPU
# default is 63, the CPU default 255 - would let two runs of the same named configuration fit
# different models and register them under the same name.
#
# **What the call publishes is a population**: a named, immutable list of the prediction sets it
# will produce, written down before the first fit. Afterwards every member must exist and be
# complete, which is what makes the downstream comparison well defined - `14_backtest` backtests
# this population, not whatever predictions happen to be in the registry.
#
# **One population covers all four labels**, because one run fits them all and the population is
# what that run declares. A population is immutable once written, so a notebook that fitted one
# label per run under a single name would publish the first label and be refused for the second.
#
# `SUPERSEDES_POPULATION` names the population hash this run replaces. A changed estimator
# parameter moves every training identity as surely as a changed menu does, so the refit is a
# different population under the same name and the registry refuses to write it without being told
# which snapshot it supersedes. That lineage is the only record of which generation is which.

# %%
population_name = POPULATION_NAME or "nasdaq100_microstructure-gbm-validation-v1"
execution, population = run_model_population(
    study,
    plan,
    population_name=population_name,
    supersedes=SUPERSEDES_POPULATION or None,
)

fitted = sum(len(item["fitted_folds"]) for item in execution.diagnostics)
reused = sum(len(item["reused_folds"]) for item in execution.diagnostics)
print(f"{len(execution.runs)} configurations: {fitted} folds fitted, {reused} reused")
print(f"population {population.name}: {len(population.members)} prediction sets")

# %% [markdown]
# `reused` is not zero on a second run. Every identity is re-derived from the inputs, the registry
# already holds the matching rows, and the runner returns the stored result rather than fitting
# again - so re-running this notebook unchanged costs the time it takes to read the data.
#
# ### Running configurations of your own
#
# The published run log is read-only. To add runs, open the study against a workspace, which holds
# its own registry and artifacts and reads the same labels and features:
#
# ```python
# study = open_study("nasdaq100_microstructure", workspace="~/ml4t-experiments")
# configs = load_model_configs(
#     study, "gbm", labels=["fwd_ret_15m"], config_names=["leaves_15_huber", "leaves_31_huber"]
# )
# requests = model_requests(study, configs)
# plan = plan_models(study, requests=requests)
# execution, population = run_model_population(study, plan, population_name="my-gbm-v1")
# ```
#
# `CONFIG_NAMES` fits a subset of what the menu declares. To fit something new, add a preset at
# `case_studies/config/lgb/leaves_127_huber.yaml` and list `leaves_127_huber` under `gbm:` in the
# label's menu. Editing an existing preset changes that configuration's identity, so its result
# registers as a new row beside the old one rather than replacing it.
# [`RUN_LOG.md`](../RUN_LOG.md#running-your-own-configurations) covers the rest.

# %% [markdown]
# ## 4. What came out
#
# One row per configuration and checkpoint. `ic_mean` is the **information coefficient**: at each
# decision time, rank the constituents by the model's prediction, rank them by the return they
# went on to earn, correlate the two rankings, and average that correlation over the validation
# period. It measures whether the model ranks names correctly, on a scale where zero is no
# relationship. Intraday values are smaller than the daily-horizon ICs elsewhere in the book: a
# return measured over minutes is mostly noise, and a few thousandths of consistent rank
# correlation is a real effect at these horizons. How much of that noise a horizon carries is
# what separates the five-minute label from the sixty-minute one, and reading them side by side
# is what the multi-label run is for.
#
# **`ic_n_days` does not count days here.** The column is named for the daily case studies, where
# one decision date produces one cross-section. This case study decides every minute, so the
# stored count is a count of *decision times* - tens of thousands of them across the validation
# period. It is still the right thing to compare configurations on, because every configuration is
# counted the same way, but do not read it as a number of trading days.
#
# Coverage is judged against **each label's own** maximum. The horizons do not offer the same
# number of scorable decision times to begin with, so one global maximum would mark a whole label
# incomplete for a reason unrelated to any model. Within a label, a configuration measured on
# fewer of them has an `ic_mean` averaged over a sample it selected itself, and is not comparable
# to its neighbours.

# %% tags=["results"]
catalog = execution.catalog_rows.select(
    "config_name",
    "label",
    "task",
    # The sibling a regression prediction was scored as a direction signal against. `by_label`
    # reports it beside the AUC so a null AUC is readable as "no sibling declared at this horizon"
    # rather than as a missing measurement.
    "direction_label",
    "complete",
    "checkpoint_value",
    "ic_mean",
    "ic_std",
    "ic_n_days",
    "auc_mean_daily",
    "n_folds",
    "training_hash",
    "prediction_hash",
).sort(["label", "ic_mean"], descending=[False, True])

if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("gbm execution returned a partial prediction set")

catalog = catalog.with_columns(
    full_coverage=pl.col("ic_n_days") == pl.col("ic_n_days").max().over("label")
)

primary = primary_label(study)
present = sorted(set(catalog.get_column("label")))
# The primary label leads when it was fitted. A subset run that leaves it out orders the panels by
# whichever label it did fit rather than by one that is not there.
panel_labels = [label for label in [primary] if label in present] + [
    label for label in present if label != primary
]
order_label = panel_labels[0]
print(f"{catalog.height} candidate models: {catalog.n_unique('config_name')} configurations")
print(f"at {catalog.n_unique('checkpoint_value')} checkpoints each, on {len(panel_labels)} labels")
catalog.select(
    "label",
    "config_name",
    "checkpoint_value",
    "ic_mean",
    "ic_std",
    "ic_n_days",
    "full_coverage",
).head(15)

# %% [markdown]
# `ic_mean` is what puts all four labels on one axis. On a regression label it ranks constituents
# by predicted return; on `fwd_dir_15m` it ranks them by predicted probability and correlates that
# with the continuous return the label was cut from. Same measurement either way, so the classifier
# and the three regressions are read side by side.
#
# `auc_mean_daily` is a second reading, and which rows carry it is the opposite of what the column
# name suggests. An AUC needs a two-state outcome, and `fwd_dir_15m` has three - down, flat, up -
# so the classification rows carry no AUC at all. The rows that do are regressions: a predicted
# return is scored as a ranking signal against a declared direction sibling, binarized to whether
# the constituent rose or did not, and `auc_scored_against` names the sibling it was scored
# against. `config/setup.yaml` declares exactly one such cut, `fwd_dir_15m` from `fwd_ret_15m`, so
# `fwd_ret_15m` is the only label here with an AUC and the other three have none. Null in that
# column means not computed, not zero.

# %% tags=["results"]
by_label = (
    catalog.group_by("label")
    .agg(
        task=pl.col("task").first(),
        configurations=pl.col("config_name").n_unique(),
        candidates=pl.len(),
        scored_times=pl.col("ic_n_days").max(),
        full_coverage=pl.col("full_coverage").sum(),
        best_ic=pl.col("ic_mean").max(),
        best_auc_daily=pl.col("auc_mean_daily").max(),
        auc_scored_against=pl.col("direction_label").drop_nulls().first(),
    )
    .sort("label")
)
by_label

# %% [markdown]
# ### What more trees do
#
# Each line traces one configuration's out-of-sample IC as trees are added to it. This is the
# figure the checkpoint dimension exists to produce, and it separates two things a single
# end-of-training number cannot.
#
# A line that rises and then falls has an interior optimum: the model was still learning, then
# began fitting the training window at the expense of the validation folds. A line that wanders
# without trend around zero never had anything to learn in the first place, and its highest point
# is wherever the noise happened to peak. The difference matters, because both produce a
# respectable-looking maximum.
#
# Only full-coverage configurations are drawn, and the panels share an x-axis so the horizons are
# compared at the same training length rather than at each one's own best point.

# %%
curves = catalog.filter("full_coverage").sort("label", "config_name", "checkpoint_value")
objectives = {
    "mse": COLORS["blue"],
    "mae": COLORS["amber"],
    "huber": COLORS["copper"],
    # Not `slate`: at #1a2d4a it is a shade off the `blue` used for squared error, and the two
    # would be one swatch in a legend the reader consults for all four panels at once.
    "multiclass": COLORS["recede"],
}


def objective_of(name: str) -> str:
    """Read the loss function out of a declared configuration name."""
    match = next((key for key in objectives if name.endswith(key)), None)
    if match is None:
        raise ValueError(f"{name!r} does not end in a declared objective: {sorted(objectives)}")
    return match


fig_curves = make_subplots(
    rows=len(panel_labels),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=[
        f"{label} ({'primary' if label == primary else 'variant'})" for label in panel_labels
    ],
)
drawn_objectives: set[str] = set()
for row, label in enumerate(panel_labels, start=1):
    panel = curves.filter(pl.col("label") == label)
    for objective, color in objectives.items():
        members = [
            name
            for name in panel.get_column("config_name").unique(maintain_order=True)
            if objective_of(name) == objective
        ]
        for config_name in members:
            series = panel.filter(pl.col("config_name") == config_name)
            fig_curves.add_trace(
                go.Scatter(
                    x=series.get_column("checkpoint_value").to_list(),
                    y=series.get_column("ic_mean").to_list(),
                    mode="lines",
                    name=objective,
                    legendgroup=objective,
                    # One legend entry per loss function, not per configuration: the colour is the
                    # claim, and fifty named lines would bury it.
                    showlegend=objective not in drawn_objectives,
                    line=dict(color=color, width=1.5),
                    opacity=0.75,
                ),
                row=row,
                col=1,
            )
            drawn_objectives.add(objective)
    fig_curves.add_hline(
        y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"], row=row, col=1
    )
    fig_curves.update_yaxes(title_text="Mean IC (validation)", row=row, col=1)
fig_curves.update_xaxes(title_text="Boosting iterations (trees kept)", row=len(panel_labels), col=1)
fig_curves.update_layout(
    title="Validation IC against boosting iteration, by loss function and label",
    height=280 * len(panel_labels),
    width=1000,
    margin=dict(t=90),
    legend=dict(title_text="Loss function"),
)
# How many lines end below zero is a fact about the frame, so the alt text reads it rather than
# asserting it: a panel described as uniformly positive when it is not is a claim the data refutes.
dip_text = " and ".join(
    f"{row['below']} of {row['total']} in the {row['label']} panel"
    for row in curves.group_by("label")
    .agg(
        total=pl.col("config_name").n_unique(),
        below=pl.col("config_name").filter(pl.col("ic_mean") < 0).n_unique(),
    )
    .sort("label")
    .iter_rows(named=True)
)
show_plotly_with_alt(
    fig_curves,
    "Line charts of mean validation information coefficient against boosting iteration, one line "
    "per configuration, coloured by loss function: dark navy for squared error, gold for absolute "
    "error, copper for Huber, muted grey-blue for multiclass. One panel per declared label, "
    "sharing the iteration axis, each carrying a dashed zero line. Counted from the underlying frame, the "
    f"lines that dip below zero at some checkpoint are {dip_text}. The vertical scale is "
    "thousandths throughout, which is what a fifteen-minute horizon supports.",
)

# %% [markdown]
# Whether the lines trend, and which way, is not something to read off a page of overlapping
# curves. The frame below measures it: for each configuration, its IC at the last checkpoint minus
# its IC at the first, and whether its best checkpoint is an interior one rather than either end.
# A family still learning would show positive changes and interior peaks; a family overfitting
# would show negative changes; a family with nothing to learn would show changes centred on zero
# with peaks scattered anywhere.

# %% tags=["results"]
drift = (
    curves.group_by("label", "config_name")
    .agg(
        first_ic=pl.col("ic_mean").sort_by("checkpoint_value").first(),
        last_ic=pl.col("ic_mean").sort_by("checkpoint_value").last(),
        peak_checkpoint=pl.col("checkpoint_value").sort_by("ic_mean", descending=True).first(),
        first_checkpoint=pl.col("checkpoint_value").min(),
        last_checkpoint=pl.col("checkpoint_value").max(),
    )
    .with_columns(
        change=pl.col("last_ic") - pl.col("first_ic"),
        interior_peak=pl.col("peak_checkpoint").is_between(
            pl.col("first_checkpoint"), pl.col("last_checkpoint"), closed="none"
        ),
    )
)
trees_effect = (
    drift.group_by("label")
    .agg(
        configurations=pl.len(),
        median_change=pl.col("change").median(),
        ended_lower=(pl.col("change") < 0).sum(),
        interior_peaks=pl.col("interior_peak").sum(),
    )
    .sort("label")
)
trees_effect

# %% [markdown]
# ### Comparing every configuration at the same training length
#
# The chart below drops the checkpoint dimension by taking each configuration's final state, so
# every configuration is compared at the same amount of training. That is the comparison that does
# not require choosing anything after the fact.
#
# The configurations are held in one order across the panels - their ranking on the primary label
# - so a panel that does not descend is a horizon that orders the grid differently. That is the
# result worth having from a three-horizon sweep: an ordering that holds across the horizon
# change is a property of the models, and one that does not is a property of the horizon.

# %%
final = (
    catalog.filter(pl.col("checkpoint_value") == pl.col("checkpoint_value").max().over("label"))
    .filter("full_coverage")
    .sort(["label", "ic_mean"], descending=[False, True])
)
final_iteration = int(final.get_column("checkpoint_value").max())
config_order = (
    final.filter(pl.col("label") == order_label)
    .sort("ic_mean", descending=True)
    .get_column("config_name")
    .to_list()
)

fig_final = make_subplots(
    rows=len(panel_labels),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=[
        f"{label} ({'primary' if label == primary else 'variant'})" for label in panel_labels
    ],
)
for row, label in enumerate(panel_labels, start=1):
    panel = final.filter(pl.col("label") == label)
    # A label whose menu declares different configurations - the classification one does - keeps
    # its own members and appends them after the shared order rather than being dropped.
    order = [name for name in config_order if name in set(panel.get_column("config_name"))]
    order += [name for name in panel.get_column("config_name").to_list() if name not in order]
    panel = panel.with_columns(
        rank=pl.col("config_name").replace_strict(
            {name: index for index, name in enumerate(order)}, return_dtype=pl.Int32
        )
    ).sort("rank")
    fig_final.add_trace(
        go.Bar(
            x=panel.get_column("config_name").to_list(),
            y=panel.get_column("ic_mean").to_list(),
            marker_color=[
                objectives[objective_of(name)] for name in panel.get_column("config_name")
            ],
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig_final.add_hline(
        y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"], row=row, col=1
    )
    fig_final.update_yaxes(title_text="Mean IC (validation)", row=row, col=1)
fig_final.update_xaxes(
    title_text=f"Configuration, ordered by rank on {order_label}",
    tickangle=-45,
    row=len(panel_labels),
    col=1,
)
fig_final.update_layout(
    title="The grid does not keep one order across the three horizons",
    height=280 * len(panel_labels),
    width=1000,
    margin=dict(t=90),
)
show_plotly_with_alt(
    fig_final,
    "Bar charts of mean validation information coefficient at the final boosting iteration, one "
    "panel per declared label, bars coloured by loss function and held in the primary label's "
    "ranking order across every panel. The primary panel descends by construction; whether the "
    "others do is the comparison the figure exists for. Every panel carries a dashed zero line "
    "and the vertical scale is thousandths of rank correlation.",
)

# %% [markdown]
# The frame below puts a number on what the chart shows: for each label, the rank correlation
# between that label's ordering of the grid and the primary label's. A value near one means the
# horizon reorders nothing; a value near zero means the grid that wins at one horizon carries no
# information about which wins at another.

# %% tags=["results"]
shared = final.filter(pl.col("config_name").is_in(config_order))
order_ic = shared.filter(pl.col("label") == order_label).select("config_name", "ic_mean")
agreement = (
    shared.join(order_ic, on="config_name", suffix="_primary")
    .group_by("label")
    .agg(
        shared_configurations=pl.len(),
        rank_agreement=pl.corr(
            pl.col("ic_mean").rank(), pl.col("ic_mean_primary").rank(), method="pearson"
        ),
        best_ic=pl.col("ic_mean").max(),
    )
    .sort("label")
)
print(f"compared at {final_iteration} boosting iterations")
agreement

# %% [markdown]
# ## 5. What to notice
#
# **The learning curves say whether the checkpoint dimension is buying anything.** If every
# configuration's highest-IC checkpoint is its last, the ten checkpoints cost storage and bought
# nothing, and the grid could have been scored at the end. If those checkpoints are interior
# and the curves turn over, then stopping early is a real dial and the `trees_effect` frame says
# on which labels. Read that frame before reading the ranking: it is the difference between a grid
# where capacity binds and one where it does not.
#
# **The horizon comparison is the finding that generalizes.** Boosting a fifteen-minute return and
# boosting a sixty-minute return are the same code on the same features, and the only thing that
# changed is how far ahead the label looks. Where the orderings agree, the grid is measuring
# something about the models; where they disagree, it is measuring something about the horizon,
# and a configuration chosen on one horizon should not be carried to another on the strength of
# that choice.
#
# **Read the ranking with the coverage column or it will mislead you.** A configuration whose
# predictions collapse to near-constant at some decision times contributes no rank correlation
# there, so its `ic_mean` is an average over a sample it selected itself. `full_coverage` marks the
# ones measured on all of them, and the figures draw only those.
#
# **The absolute level is small at every horizon here, and that is the reading an intraday
# forecast supports.** A daily-horizon equity signal reports IC in the hundredths; over minutes a
# few thousandths is what a working signal looks like, because almost all of a five- or
# fifteen-minute return is noise. The `by_label` frame is where to check whether the level rises
# with the horizon, which is the direction that argument predicts. Whether any of it is tradable
# is not answerable from IC at all - at these horizons the position turns over constantly, so
# costs decide it, and costs are not in this notebook.
#
# **None of this selects anything.** The leading row of the catalog is the maximum of the whole
# grid at ten checkpoints, and reading it as the result of one experiment would attribute to the
# model whatever the stopping point contributed. Selection is on validation backtest Sharpe over
# the population this notebook just published, and it happens in
# [`14_backtest`](14_backtest.ipynb).
#
# **Known limitations.** The IC here is an average of per-timestamp rank correlations with no
# adjustment for the serial dependence that overlapping forward returns create, so it is a ranking
# diagnostic rather than a test. The grid varies capacity and loss at a fixed learning rate and
# fixed features, so it says nothing about interactions with either. Every number is measured on
# the validation folds, which have been read many times over by the time a case study reaches this
# notebook.
#
# **Next**: [`08_dl_nlinear`](08_dl_nlinear.ipynb) is the first of four sequence models, and it
# changes what the model is shown rather than how flexible it is. Everything fitted so far reads one
# row of summary features per decision minute; NLinear reads the last several minutes of every
# feature and maps that window to a forecast. Whether the recent path carries anything the summary
# row does not is a separate question from the one this notebook answered.
