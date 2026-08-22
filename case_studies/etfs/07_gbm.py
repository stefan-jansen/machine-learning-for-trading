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
# # ETFs: what trees can find that a penalty cannot, and what they give up to look
#
# [`06_linear`](06_linear.ipynb) is the one place in these case studies where the linear grid
# worked. Ridge at a very large penalty ranked the 99 funds well ahead of zero across the full
# validation period, and the shape of that sweep said why: the feature set is **collinear**,
# several columns carry nearly the same information, and a dense penalty that spreads weight
# across a group of near-duplicates recovers signal that an unregularized fit buries in
# offsetting coefficients.
#
# That result frames the question for gradient boosting differently than it would be framed on a
# case study where the linear model found nothing. There is a working baseline here, boosting has
# to beat it, and the two properties that decide whether it can pull in opposite directions.
#
# **In its favour, trees represent interactions a linear model cannot.** A linear model sees an
# interaction only if someone multiplied two columns together and named the product. A tree splits
# on one feature inside a region already defined by others, so an interaction is something it
# discovers rather than something it is handed. The one this case study has a reason to expect is
# the market-stress regime probability from
# [`04_model_based_features`](04_model_based_features.ipynb): momentum worth following in a calm
# regime may be worth fading in a stressed one, and no single coefficient on momentum expresses
# both.
#
# **Against it, collinearity is hostile to trees in a way it is not to ridge.** Faced with several
# near-identical columns, a tree picks one of them at each split, and which one is close to
# arbitrary. Ridge's advantage on this data came precisely from *not* choosing - from spreading
# weight over the correlated group and averaging their noise down. A greedy splitter makes that
# choice at every split and remakes it independently on every fold. The same property of the
# feature set that made a dense penalty the right answer makes boosting an awkward fit for it.
#
# Three dials control how far the fit goes, and this notebook varies all three:
#
# - **Capacity**, set by `num_leaves`: how many regions one tree may carve the feature space into.
#   Seven leaves can express a handful of conditions; 63 can express a partition fine enough to
#   describe the training window and nothing beyond it. On a cross-section of 99 funds the top of
#   this axis is in the grid as a failure case as much as a candidate.
# - **The loss function**, which decides what "got wrong" means. `mse` minimizes squared error,
#   `mae` absolute error, and `huber` behaves like squared error for small residuals and like
#   absolute error past a threshold derived from each fold's own label spread.
# - **When to stop**, set by the number of trees. Unlike a linear fit, a boosted model has a
#   meaningful state at every iteration, so each configuration is scored at ten points along its
#   own training run rather than only at the end.
#
# The third dial changes how the results must be read. **A checkpoint is part of a configuration,
# not a detail of how it was fitted.** Scoring 15 declared configurations at ten checkpoints each
# produces 150 candidate models, and treating that as 15 candidates while quietly keeping each
# one's best iteration would be reporting the maximum of ten numbers as though it were one.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Say what a tree ensemble can represent that a penalized linear model cannot, and what a
#   collinear feature set costs a greedy splitter.
# - Explain why a boosted model produces one result per checkpoint while a linear model produces
#   one result in total, and what that implies for counting candidates.
# - Read a learning curve of out-of-sample information coefficient against tree count, and tell
#   apart a model still learning from one that has begun fitting the training window.
# - Say why the choice of loss function is a statement about the label's tails, and relate that to
#   what a rank-based metric rewards.
# - Recognise that picking each configuration's best checkpoint after seeing the results is a
#   selection decision, and locate where selection is actually made.
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
"""Fit the declared ETF gradient boosting population on the walk-forward folds."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from case_studies.research import (
    declared_labels,
    load_model_configs,
    model_requests,
    narrows_declared_catalog,
    open_study,
    primary_label,
    resolved_model_plan,
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
study = open_study("etfs", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)

# %% [markdown]
# ## 1. Which labels, and which models
#
# The labels are the ones the linear notebook fitted: `fwd_ret_21d`, the total return over the 21
# trading days after the decision date, and the five-day variant `fwd_ret_5d`. Fitting the same
# two here is what makes the two populations comparable - the families differ, the targets do
# not - and it lets the question `06_linear` raised about the horizon be asked again of a tree
# ensemble. Each label carries its own training menu at `config/training/{label}.yaml` and this
# notebook fits the union of them; `LABELS` restricts the run to a subset when you want one.

# %%
declared_labels(study, "gbm")

# %% [markdown]
# The menu lists 15 named configurations under `gbm:`, and each resolves to a preset in
# `case_studies/config/lgb/`. The grid is a product of two axes:
#
# - **Five capacity profiles.** `default` uses the library's own leaf count; the rest fix it at 7,
#   15, 31 and 63.
# - **Three objectives**, as described above.
#
# Every configuration runs the same number of boosting iterations at the same learning rate, so
# the grid isolates capacity and loss rather than confounding them with training length.

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
# set of members than the canonical population does. A population is immutable once written, so
# such a run must publish under its own name. Comparing the loaded rows against the complete
# declared catalog catches either knob, and says so here rather than several cells later in a
# message about hashes.

# %%
if narrows_declared_catalog(study, "gbm", configs) and not POPULATION_NAME:
    raise ValueError(
        f"this run declares {configs.height} label-configuration pairs, which is not the "
        f"complete declared catalog, so it cannot publish the canonical population; pass "
        f"POPULATION_NAME to give it its own"
    )

# %% [markdown]
# ## 2. Binding the declarations to the data
#
# Resolving reads the label and feature files, computes the fold boundaries, works out the exact
# rows each fit must predict, and turns any data-dependent parameter into the number it will use.
# Huber's threshold is one of those: it is a fraction of the training labels' standard deviation,
# so it is a different number on every fold and is resolved from that fold's own data.
#
# Nothing is fitted here, so the plan can be inspected first. Four things to check:
#
# - **`feature_count`, `eligible_entities` and `eligible_rows` agree across every row.** A row
#   that differs is a configuration measured on a different sample from its neighbours.
# - **`folds` is the same everywhere**, and equals the number of walk-forward splits.
# - **`validation_start` and `validation_end` bracket the development sample**, with none of the
#   held-out tail visible.
# - **`checkpoints` is where this differs from the linear plan.** It is the number of training
#   states each configuration will publish predictions for. Multiply it by the number of rows to
#   get the number of candidate models this notebook is about to create.

# %%
requests = model_requests(
    study,
    configs,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
)
resolved = tuple(request.resolve() for request in requests)

plan = resolved_model_plan(resolved)
plan.select(
    "config_name",
    "feature_count",
    "eligible_entities",
    "eligible_rows",
    "folds",
    "checkpoints",
    "validation_start",
    "validation_end",
)

# %% [markdown]
# ## 3. Fitting the population
#
# `run_model_population` fits every resolved request. For one request it walks the folds, and on
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
# Preparation happens once per fold and is shared by every configuration, because slicing the
# window and cleaning the rows depends on the data and not on the model. The run walks folds on
# the outside and configurations on the inside for the same reason: one prepared fold is held at a
# time rather than the whole set.
#
# **What the call publishes is a population**: a named, immutable list of the prediction sets it
# will produce, written down before the first fit. Afterwards every member must exist and be
# complete, which is what makes the downstream comparison well defined.
#
# `SUPERSEDES_POPULATION` names the population hash this run replaces. A population is the set
# of prediction identities, so anything that moves a training identity - a changed estimator
# parameter as much as a changed configuration menu - produces a different population under the
# same name, and the registry refuses to write it without being told which snapshot it
# supersedes. That lineage is the only record of which generation is which.
#
# The default name is the contract with the notebooks downstream - `11_model_analysis` and
# `12_backtest` resolve this population by name - rather than a label of convenience, which is
# why a run that narrows the member set has to pass its own.

# %%
population_name = POPULATION_NAME or "etfs-gbm-validation-v1"
execution, population = run_model_population(
    study,
    resolved,
    population_name=population_name,
    supersedes=SUPERSEDES_POPULATION or None,
)

print(f"{len(execution.runs)} configurations fitted")
print(f"population {population.name}: {len(population.members)} prediction sets")

# %% [markdown]
# Re-running this notebook unchanged costs the time it takes to read the data. Every identity is
# re-derived from the inputs, the registry already holds the matching rows, and the runner returns
# the stored result rather than fitting again.
#
# ### Running configurations of your own
#
# The published run log is read-only. To add runs, open the study against a workspace, which holds
# its own registry and artifacts and reads the same labels and features:
#
# ```python
# study = open_study("etfs", workspace="~/ml4t-experiments")
# configs = load_model_configs(
#     study, "gbm", labels=["fwd_ret_21d"], config_names=["leaves_15_huber", "leaves_31_huber"]
# )
# requests = model_requests(study, configs)
# resolved = tuple(request.resolve() for request in requests)
# execution, population = run_model_population(study, resolved, population_name="my-gbm-v1")
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
# One row per configuration and checkpoint. `ic_mean` is the **information coefficient**: on each
# validation date, rank the funds by the model's prediction, rank them by the return they went on
# to earn, correlate the two rankings, and average that daily correlation over the validation
# period.
#
# The table is sorted by label and then by IC, and the top of each label's block is the trap this
# notebook exists to describe. The leading row for a label is the maximum of 150 numbers - fifteen
# configurations at ten checkpoints each. Reading it as the result of one experiment would
# attribute to the model whatever the stopping point contributed, and the section below measures
# how large that contribution is before anything is concluded from the ranking.
#
# `ic_n_days` carries the second warning, and it is the one `06_linear` turned on: a configuration
# that scored fewer dates than its neighbours is not comparable to them, because its IC is an
# average over the dates where it stayed non-degenerate. Every comparison below is restricted to
# full-coverage members for that reason.
#
# Coverage is judged against each label's own maximum number of scorable validation dates. The
# horizons do not offer the same number to begin with, since a longer forward window runs out
# earlier, so one global maximum would mark a whole label incomplete for a reason unrelated to any
# model.

# %% tags=["results"]
catalog = execution.catalog_rows.select(
    "config_name",
    "label",
    "complete",
    "checkpoint_value",
    "ic_mean",
    "ic_std",
    "ic_n_days",
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
# The primary label leads when it was fitted. A subset run that leaves it out orders the panels
# by whichever label it did fit rather than by one that is not there.
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
# One panel per label, because a horizon that has something to learn and one that does not would
# be averaged into a single indistinct band if they shared axes.

# %%
curves = catalog.filter("full_coverage").sort("label", "config_name", "checkpoint_value")
objectives = {"mse": COLORS["blue"], "mae": COLORS["amber"], "huber": COLORS["copper"]}


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
    vertical_spacing=0.05,
    subplot_titles=[
        f"{label} ({'primary' if label == primary else 'variant'})" for label in panel_labels
    ],
)
for row, label in enumerate(panel_labels, start=1):
    panel = curves.filter(pl.col("label") == label)
    for objective, color in objectives.items():
        members = [
            name
            for name in panel.get_column("config_name").unique(maintain_order=True)
            if objective_of(name) == objective
        ]
        for index, config_name in enumerate(members):
            series = panel.filter(pl.col("config_name") == config_name)
            fig_curves.add_trace(
                go.Scatter(
                    x=series.get_column("checkpoint_value").to_list(),
                    y=series.get_column("ic_mean").to_list(),
                    mode="lines",
                    name=objective,
                    legendgroup=objective,
                    # One legend entry per loss function, not per configuration: the colour is
                    # the claim, and thirty named lines would bury it.
                    showlegend=row == 1 and index == 0,
                    line=dict(color=color, width=1.5),
                    opacity=0.75,
                ),
                row=row,
                col=1,
            )
    fig_curves.add_hline(
        y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"], row=row, col=1
    )
    fig_curves.update_yaxes(title_text="Mean IC (validation)", row=row, col=1)
fig_curves.update_xaxes(title_text="Boosting iterations (trees kept)", row=len(panel_labels), col=1)
fig_curves.update_layout(
    title="Validation IC against boosting iteration, by loss function and label",
    height=330 * len(panel_labels),
    width=1000,
    margin=dict(t=90),
    legend=dict(title_text="Loss function"),
)
# Where each panel sits and how many of its lines dip below zero are facts about the frame, so
# the alt text reads them rather than asserting them. A description written once and left alone
# would go stale the first time the fits move.
panel_facts = (
    curves.group_by("label")
    .agg(
        lowest=pl.col("ic_mean").min(),
        highest=pl.col("ic_mean").max(),
        total=pl.col("config_name").n_unique(),
        below=pl.col("config_name").filter(pl.col("ic_mean") < 0).n_unique(),
    )
    .to_dicts()
)
panel_facts = {row["label"]: row for row in panel_facts}
panel_text = ". ".join(
    "The {} panel spans {:+.3f} to {:+.3f}, with {} of its {} lines dipping below zero at some "
    "checkpoint".format(
        label,
        panel_facts[label]["lowest"],
        panel_facts[label]["highest"],
        panel_facts[label]["below"],
        panel_facts[label]["total"],
    )
    for label in panel_labels
)
show_plotly_with_alt(
    fig_curves,
    "Line charts of mean validation information coefficient against boosting iteration, one line "
    "per configuration, coloured by loss function: dark navy for squared error, gold for absolute "
    "error, copper for Huber. One panel per label, each with its own vertical scale and a dashed "
    f"zero line. {panel_text}.",
)

# %% [markdown]
# Whether the lines trend, and which way, is not something to read off a page of overlapping
# curves. The frame below measures it: for each configuration, its IC at the last checkpoint
# minus its IC at the first, and whether its best checkpoint is an interior one rather than
# either end. A family that is still learning would show positive changes and interior peaks; a
# family that is overfitting would show negative changes; a family with nothing to learn would
# show changes centred on zero with peaks scattered anywhere.

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
# The curves are coloured by objective because that is the axis with a mechanism behind it. If
# heavy tails are steering the squared-error fits, the three colours should separate, and they
# should separate more as trees are added, since each additional tree is fitted to the residuals
# the previous ones left.
#
# The chart below drops the checkpoint dimension by taking each configuration's final state, so
# every configuration is compared at the same amount of training. That is the comparison that does
# not require choosing anything after the fact. The configurations are held in one order across
# the panels - their ranking on the primary label - so a panel that does not descend is a horizon
# that orders the grid differently.

# %%
final = (
    catalog.filter(pl.col("checkpoint_value") == pl.col("checkpoint_value").max().over("label"))
    .filter("full_coverage")
    .with_columns(objective=pl.col("config_name").map_elements(objective_of, return_dtype=pl.Utf8))
    .sort(["label", "ic_mean"], descending=[False, True])
)
final_iteration = int(final.get_column("checkpoint_value").max())
config_order = (
    final.filter(pl.col("label") == order_label)
    .sort("ic_mean", descending=True)
    .get_column("config_name")
    .to_list()
)

fig_obj = make_subplots(
    rows=len(panel_labels),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.09,
    subplot_titles=[
        f"{label} ({'primary' if label == primary else 'variant'})" for label in panel_labels
    ],
)
for row, label in enumerate(panel_labels, start=1):
    panel = final.filter(pl.col("label") == label)
    fig_obj.add_trace(
        go.Bar(
            x=panel.get_column("config_name").to_list(),
            y=panel.get_column("ic_mean").to_list(),
            marker_color=[objectives[value] for value in panel.get_column("objective")],
        ),
        row=row,
        col=1,
    )
    fig_obj.add_hline(
        y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"], row=row, col=1
    )
    fig_obj.update_yaxes(title_text="Mean IC (validation)", row=row, col=1)
fig_obj.update_xaxes(
    categoryorder="array",
    categoryarray=config_order,
    tickangle=-45,
    title_text=f"Configuration (ordered by validation IC on {order_label})",
    row=len(panel_labels),
    col=1,
)
fig_obj.update_layout(
    title="Validation IC at the final iteration, by loss function and label",
    height=320 * len(panel_labels),
    width=1000,
    showlegend=False,
    margin=dict(t=90),
)
final_text = ". ".join(
    "The {} panel runs from {:+.3f} to {:+.3f} with {} of {} bars above zero, led by {}".format(
        label,
        final.filter(pl.col("label") == label).get_column("ic_mean").min(),
        final.filter(pl.col("label") == label).get_column("ic_mean").max(),
        final.filter((pl.col("label") == label) & (pl.col("ic_mean") > 0)).height,
        final.filter(pl.col("label") == label).height,
        final.filter(pl.col("label") == label)
        .sort("ic_mean", descending=True)
        .row(0, named=True)["config_name"],
    )
    for label in panel_labels
)
show_plotly_with_alt(
    fig_obj,
    "Stacked bar charts of mean validation information coefficient at the final boosting "
    "iteration, coloured by loss function: dark navy for squared error, gold for absolute error, "
    "copper for Huber. The configurations are in the same order in each panel, that order being "
    f"their ranking on {order_label}, and each panel carries a dashed zero line. {final_text}.",
)

# %% [markdown]
# ### Whether the horizon does here what it did to the linear grid
#
# The question this notebook took from `06_linear`, answered at the same training length for
# every configuration: whether the horizon that ranked the penalty grid one way ranks a tree
# ensemble the same way. `rank_correlation` is between the two labels' orderings of the
# configurations they both charted, so one would mean the horizons agree on the whole grid.

# %% tags=["results"]
horizons = (
    final.group_by("label")
    .agg(
        configurations=pl.len(),
        best_config=pl.col("config_name").sort_by("ic_mean", descending=True).first(),
        best_ic=pl.col("ic_mean").max(),
        worst_ic=pl.col("ic_mean").min(),
        above_zero=(pl.col("ic_mean") > 0).sum(),
    )
    .sort("label")
)
horizons

# %% tags=["results"]
pair = (
    final.select("label", "config_name", "ic_mean")
    .pivot(on="label", index="config_name", values="ic_mean")
    .drop_nulls()
)
if len(panel_labels) > 1:
    print(f"{pair.height} configurations charted at both horizons")
    for index, label_a in enumerate(panel_labels):
        for label_b in panel_labels[index + 1 :]:
            correlation = pair.select(
                pl.corr(pl.col(label_a).rank(), pl.col(label_b).rank())
            ).item()
            print(f"rank correlation {label_a} against {label_b}: {correlation:+.3f}")
else:
    print("one label charted, so there is no horizon comparison to make")

# %% [markdown]
# ### Whether the loss function is what separates them
#
# The colours in the chart above are the claim, so here is the claim as numbers: at the final
# iteration, each loss function's mean IC and how many of its configurations finished above zero,
# per label. The ordering a heavy-tailed label produces is Huber highest, absolute error next,
# squared error lowest, because a squared-error fit weights a residual by its square and one
# extreme observation therefore moves it more than many ordinary ones, while a rank measure gives
# nothing back for chasing that extreme.

# %% tags=["results"]
objective_summary = (
    final.group_by("label", "objective")
    .agg(
        configurations=pl.len(),
        mean_ic=pl.col("ic_mean").mean(),
        best_ic=pl.col("ic_mean").max(),
        above_zero=(pl.col("ic_mean") > 0).sum(),
    )
    .sort(["label", "mean_ic"], descending=[False, True])
)
objective_summary

# %% [markdown]
# ### What the tails of each label look like
#
# The mechanism above is a claim about the label, not the model, so it is checkable without
# fitting anything. Excess kurtosis measures how much of a distribution's variance comes from
# rare large moves; where the loss functions separate, the heavier-tailed label should separate
# them further. The rows are cut at the development boundary the fits were resolved against, so
# nothing held back for the holdout is described here.

# %% tags=["results"]
development_end = plan.get_column("validation_end").max()
tails = pl.DataFrame(
    [
        {
            "label": label,
            "observations": len(values),
            "std": float(values.std(ddof=1)),
            "excess_kurtosis": float(
                ((values - values.mean()) ** 4).mean() / values.var(ddof=0) ** 2 - 3.0
            ),
            "share_beyond_4_sd": float(
                (np.abs(values - values.mean()) > 4 * values.std(ddof=1)).mean()
            ),
        }
        for label, values in (
            (
                label,
                study.labels.get(label, execution_tier=EXECUTION_TIER)
                .load()
                .filter(pl.col("timestamp") <= development_end)
                .get_column(label)
                .drop_nulls()
                .to_numpy(),
            )
            for label in panel_labels
        )
    ]
).sort("label")
tails

# %% [markdown]
# ### How much the checkpoint moves a configuration
#
# One number per configuration: the range its IC covers across its own ten checkpoints. This is
# the quantity that decides whether choosing a stopping point is a decision worth making carefully
# or one being made by noise. A configuration whose IC varies more across its own training run
# than the configurations vary among themselves is one where the checkpoint, not the model, is
# doing the ranking. The comparison is within a label, since the two horizons do not produce ICs
# of the same size.

# %% tags=["results"]
spread = (
    curves.group_by("label", "config_name")
    .agg(
        ic_min=pl.col("ic_mean").min(),
        ic_max=pl.col("ic_mean").max(),
        ic_final=pl.col("ic_mean").sort_by("checkpoint_value").last(),
    )
    .with_columns(checkpoint_range=pl.col("ic_max") - pl.col("ic_min"))
    .sort(["label", "checkpoint_range"], descending=[False, True])
)
checkpoint_vs_config = (
    spread.group_by("label")
    .agg(
        across_configs=pl.col("ic_final").max() - pl.col("ic_final").min(),
        median_within_config=pl.col("checkpoint_range").median(),
    )
    .with_columns(
        checkpoint_dominates=pl.col("median_within_config") > pl.col("across_configs"),
    )
    .sort("label")
)
checkpoint_vs_config

# %% tags=["results"]
spread

# %% [markdown]
# ## 5. What to notice
#
# **Every configuration ranks the cross-section the right way, and the loss function orders them.**
# All fifteen are positive at the end of training. Sorted by IC, the four fixed-capacity Huber
# configurations take the top four places, absolute error follows, and squared error occupies the
# bottom - the weakest Huber setting still ranks above the strongest absolute-error one. That is
# the order the label's tails predict. Squared error weights an observation by the square of its
# error, so the largest 21-day moves dominate what each successive tree is fitted to, while the
# information coefficient is a rank correlation that cares about order rather than magnitude:
# effort spent getting the extremes right buys nothing on this metric. **An objective is a claim
# about which errors matter, and it is worth choosing to match the metric the result will be
# judged on.**
#
# **The boosted population does not beat the linear one.** The strongest full-coverage
# configuration here ranks below the strongest full-coverage configuration in
# [`06_linear`](06_linear.ipynb), on the same label, the same features and the same folds. This is
# the outcome the collinearity argument at the top of this notebook implies, and it is worth
# taking seriously rather than treating as a tuning failure. Ridge earned its result by refusing
# to choose among near-duplicate columns - spreading weight across a correlated group averages
# their noise down. A tree chooses one column at every split, and remakes that choice
# independently on every fold, which is the opposite operation on the same feature set. The
# interactions the trees can represent and a linear model cannot are real, but on this data they
# do not pay for what greedy splitting gives up.
#
# **Every configuration is past its peak by the time it is reported.** All fifteen reach their
# highest validation IC at the first or the second checkpoint and decline from there, without
# exception. So the declared training length is longer than this data supports, and the
# fixed-iteration comparison above is a comparison of fifteen models in their overfitted regime.
# That comparison is still the honest one, because each configuration is measured at the same
# amount of training and nothing is chosen after the fact - but the ranking it produces is not the
# ranking their best states would produce, and neither is a result until something selects on a
# criterion that was fixed in advance.
#
# **The checkpoint is nearly as large a dial as the model.** Across the fifteen configurations at
# fixed training length the IC spans a range; within a single configuration, across its own ten
# checkpoints, the median range is about four fifths of that. A stopping point chosen after seeing
# the curves would therefore be doing almost as much work as the choice of model, which is why
# reporting the leading row of the results table would be reporting the maximum of 150 numbers as
# though it were one.
#
# **None of this selects anything.** IC measures whether predictions rank funds correctly, not
# whether a strategy trading them makes money after costs and turnover. A monthly-horizon signal
# that reorders the portfolio at every rebalance is exactly where a small ranking advantage is
# lost. Selection is on validation backtest Sharpe over the population just published, and it
# happens in [`14_backtest`](14_backtest.ipynb), where the checkpoint is part of what is selected.
#
# **Known limitations.** The IC is an average of daily rank correlations with no adjustment for
# the serial dependence overlapping 21-day returns create, so it is a diagnostic rather than a
# test, and it carries no interval that would say whether these configurations differ from each
# other or from the linear ones. The grid varies capacity and loss at a fixed learning rate and a
# fixed training length, so it says nothing about trading one against another - and the previous
# point is the reason that matters here. And every number is measured on validation folds that
# have been read many times over by the time a case study reaches this notebook.

# %% [markdown]
# **Next**: [`08_tabular_dl`](08_tabular_dl.ipynb) asks the same question of a neural network
# built for tabular data, which represents interactions in a third way again. The useful thing to
# watch there is whether it recovers whatever structure the trees found here, and whether a
# collinear feature set costs it what it costs a greedy splitter.
