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
# # Firm characteristics: dropping the factor structure and keeping the bottleneck
#
# The three notebooks before this one all impose structure on how a prediction is formed.
# [`08a_ipca`](08a_ipca.ipynb) and [`08b_conditional_autoencoder`](08b_conditional_autoencoder.ipynb)
# require the prediction to be an exposure times a factor return;
# [`08c_stochastic_discount_factor`](08c_stochastic_discount_factor.ipynb) requires it to fall out
# of a no-arbitrage condition. The **supervised autoencoder** imposes neither. It maps the
# characteristics to the forward return end to end, and the only thing left of the family is the
# **bottleneck**: the network is forced to compress 57 characteristics through a small number of
# units before predicting, and that width is the same `n_factors` the other three declare.
#
# So the question this notebook puts is whether the compression is doing the work, or the factor
# interpretation is. If a bottleneck alone gets you what the conditional-factor structure gets you,
# then the structure was a constraint the data did not need; if it does not, the structure was
# buying something.
#
# **It is also the closest thing in this family to the earlier notebooks.** A network mapping
# characteristics to returns is what [`07_tabular_dl`](07_tabular_dl.ipynb) fits, without a
# bottleneck and with a different architecture, so those two are worth reading together as well.
#
# **Learning objectives**
#
# - Separate the two things a latent-factor model asserts: that a low-dimensional representation
#   exists, and that it should be interpreted as factor exposures.
# - Read a checkpoint path where the objective and the metric are the same kind of quantity, unlike
#   the pricing model before it.
# - See why the least-structured member of a family is worth publishing rather than assuming the
#   structured ones dominate.
#
# **Book reference**: Chapter 14, Section 14.7 (The stochastic discount factor and the supervised
# autoencoder models). Chapter 6, Section 6.7 (Search accounting and run logging) introduces the
# run log this notebook writes into.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb),
# [`04_evaluation`](04_evaluation.ipynb), [`08_latent_factors`](08_latent_factors.ipynb) and
# [`08b_conditional_autoencoder`](08b_conditional_autoencoder.ipynb), whose bottleneck this one
# keeps and whose factor structure it drops.
#
# **What it writes**: one training run per label and one complete validation prediction set per
# label and epoch checkpoint, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a population named for this model. The family splits across
# four notebooks, so each publishes its own population rather than one shared one.
# [`11_backtest`](11_backtest.ipynb) reads them and selects on validation backtest Sharpe.
# **Selection happens there, not here**, and the checkpoint is part of what is selected.

# %%
"""Fit the declared firm-characteristics supervised-autoencoder population."""

import plotly.graph_objects as go
import polars as pl

from case_studies.research import (
    declared_labels,
    load_model_configs,
    model_requests,
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
POPULATION_NAME = ""
SUPERSEDES_POPULATION: str = ""

MODEL_NAME = "sae"

# %%
study = open_study(
    "us_firm_characteristics", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None
)

# %% [markdown]
# ## 1. Which labels, and what the configuration says
#
# Every label whose training menu declares `latent_factors:` is fitted, and all three do:
# `fwd_ret_1m`, `fwd_ret_1m_win` - the same return with each month's cross-section clipped at its
# own tails - and `fwd_class_1m`, which turns that return into a class. The target matters more
# here than anywhere else in the family: this is the one member fitted directly against it, so a
# change of label changes what the whole network is optimizing rather than only what the last stage
# is scored on.

# %%
declared_labels(study, "latent_factors")

# %% [markdown]
# `case_studies/config/sae/sae.yaml` declares the bottleneck width as `n_factors`, the training
# budget and the checkpoint interval. The budget and the interval together decide the checkpoint
# surface: a state is saved every `checkpoint_interval` epochs up to `n_epochs`, and each saved
# state becomes a registered prediction set covering the whole validation period. The runtime comes
# from `config/setup.yaml` under `modeling.latent_factors`.

# %%
configs = load_model_configs(
    study,
    "latent_factors",
    labels=LABELS or None,
    config_names=[MODEL_NAME],
)
configs

# %% [markdown]
# `LABELS` narrows what is fitted, and a narrowed run declares a different set of members than the
# canonical population does. A population is immutable once written, so such a run must publish
# under its own name: on a fresh workspace it would otherwise register an incomplete snapshot under
# the canonical one, and where the full population already exists the registry refuses it.
#
# The comparison is against this model's own declared rows rather than the whole `latent_factors`
# catalog. The family is split across four notebooks and each publishes one model, so the complete
# catalog is four times what this one publishes, and comparing against it would report every
# canonical run as narrowed.

# %%
declared = load_model_configs(study, "latent_factors", config_names=[MODEL_NAME])
if set(configs.get_column("label")) != set(declared.get_column("label")) and not POPULATION_NAME:
    raise ValueError(
        f"this run fits {configs.height} of the {declared.height} declared labels, so it cannot "
        "publish the canonical population; pass POPULATION_NAME to give it its own"
    )

# %% [markdown]
# ## 2. Binding the declarations to the data
#
# Planning reads the label and feature files, computes the fold boundaries from the walk-forward
# parameters in `config/setup.yaml`, works out the exact rows each fit must predict, and derives
# every training and prediction identity - without fitting anything. Four things to check:
#
# - **`feature_count` and `eligible_rows` agree across the rows.** A row that differs is a label
#   measured on a different sample from its neighbours.
# - **`folds` is the same everywhere**, and equals the number of walk-forward splits
#   [`04_evaluation`](04_evaluation.ipynb) established.
# - **`validation_start` and `validation_end` bracket the development sample**, with none of the
#   held-out 2016 tail visible.
# - **`checkpoints` is how many training states each label will publish predictions for.**
#   Multiply it by the number of rows to get the number of candidate models this notebook creates.

# %%
requests = model_requests(
    study,
    configs,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
)
plan = plan_models(study, requests=requests)

planned_model_plan(plan).select(
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
# The plan is also where a short population is visible. A run that fitted fewer labels than the
# menu declares, or published fewer checkpoints than the configuration does, would still print an
# IC table and still register its rows; what it would not do is announce the gap.

# %%
planned_pairs = {(member.label, member.config_name) for member in plan.members}
requested_pairs = set(zip(configs["label"], configs["config_name"], strict=True))
if planned_pairs != requested_pairs:
    raise RuntimeError(
        "the plan does not match the loaded supervised-autoencoder menu; "
        f"missing {sorted(requested_pairs - planned_pairs)}, "
        f"unexpected {sorted(planned_pairs - requested_pairs)}"
    )
print(f"{len(requested_pairs)} label-configuration pairs")
print(f"{len(plan.expected_training_hashes)} training identities")
print(f"{len(plan.expected_prediction_hashes)} validation prediction sets")

# %% [markdown]
# ## 3. Fitting the population
#
# `run_model_population` runs every planned request. For one request it walks the folds, and on
# each one:
#
# 1. takes the firm-months inside that fold's training window,
# 2. trains the network - characteristics in, through the bottleneck, forward return out - for the
#    declared number of epochs, saving the fitted state at each checkpoint on the way,
# 3. applies each saved state to the validation months' characteristics to get a predicted return.
#
# Step 3 is what makes one fit produce many results. Fold predictions are concatenated into one
# series per checkpoint covering the whole validation period, and each becomes its own registered
# prediction set with its own identity. Nothing here chooses among them.
#
# **This notebook passes the plan rather than resolved requests, and on this family that changes
# what the plan table could show rather than how the fitting is ordered.** The linear, boosted and
# TabM adapters each provide a batch runner that prepares one fold set for several configurations;
# the latent-factor adapter provides none, so every request is fitted on its own whichever path is
# taken. There is nothing here for a batch runner to share in any case: this notebook publishes one
# model against three labels, and each label is a different set of rows. What passing the plan does
# cost is `eligible_entities`, which needs the eligibility keys themselves; `eligible_rows` above
# moves whenever the universe does, so the check that column existed for is still made.
#
# **What the call publishes is a population**: a named, immutable list of the prediction sets it
# will produce, written down before the first fit. Afterwards every member must exist and be
# complete, which is what makes the downstream comparison well defined.

# %%
population_name = POPULATION_NAME or f"us_firm_characteristics-{MODEL_NAME}-validation-v1"
execution, population = run_model_population(
    study, plan, population_name=population_name, supersedes=SUPERSEDES_POPULATION or None
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
# study = open_study("us_firm_characteristics", workspace="~/ml4t-experiments")
# configs = load_model_configs(study, "latent_factors", labels=["fwd_ret_1m"], config_names=["sae"])
# requests = model_requests(study, configs)
# plan = plan_models(study, requests=requests)
# execution, population = run_model_population(study, plan, population_name="my-sae-v1")
# ```
#
# The bottleneck width, the training budget and the checkpoint interval live in
# `case_studies/config/sae/sae.yaml`. Each changes this configuration's identity - the interval as
# much as the budget, because it changes which states exist - so the result registers as a new row
# beside the old one rather than replacing it.
# [`RUN_LOG.md`](../RUN_LOG.md#running-your-own-configurations) covers the rest.

# %% [markdown]
# ## 4. What came out
#
# One row per label and epoch checkpoint. `ic_mean` is the **information coefficient**: in each
# validation month, rank the firms by the model's prediction, rank them by the return they went on
# to earn, correlate the two rankings, and average that monthly correlation over the validation
# period.
#
# **Every count and aggregate below is keyed on `(label, checkpoint_value)`.** Grouping on the
# checkpoint alone would average across targets; grouping on the label alone would collapse the
# training path this notebook exists to show.
#
# The published catalog is checked against the population planned before fitting rather than
# against its own row count, because a run that lost a member would otherwise report a shorter
# table and nothing else.

# %% tags=["results"]
catalog = execution.catalog_rows.select(
    "config_name",
    "label",
    "task",
    "complete",
    "checkpoint_kind",
    "checkpoint_value",
    "ic_mean",
    "ic_std",
    "ic_t",
    "ic_n_days",
    "auc_mean_daily",
    pl.col("direction_label").alias("auc_scored_against"),
    "n_folds",
    "training_hash",
    "prediction_hash",
).sort(["label", "checkpoint_value"])

if set(catalog.get_column("prediction_hash")) != set(plan.expected_prediction_hashes):
    raise RuntimeError("the published catalog differs from the population planned before fitting")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("a partial supervised-autoencoder checkpoint cannot pass to backtesting")
if catalog.select("label", "config_name", "checkpoint_value").n_unique() != catalog.height:
    raise RuntimeError("each label and checkpoint must identify one prediction set")
if catalog.get_column("checkpoint_value").null_count():
    raise RuntimeError("every checkpoint must name the epoch it was taken at")

catalog = catalog.with_columns(
    full_coverage=pl.col("ic_n_days") == pl.col("ic_n_days").max().over("label")
)

primary = primary_label(study)
present = sorted(set(catalog.get_column("label")))
# The primary label leads when it was fitted. A subset run that leaves it out orders by whichever
# label it did fit rather than by one that is not there.
panel_labels = [label for label in [primary] if label in present] + [
    label for label in present if label != primary
]
print(f"{catalog.height} candidate models across {len(panel_labels)} labels")
print(f"at {catalog.n_unique('checkpoint_value')} checkpoints each")
catalog.select(
    "label",
    "checkpoint_value",
    "ic_mean",
    "ic_std",
    "ic_t",
    "ic_n_days",
    "full_coverage",
).head(15)

# %% [markdown]
# ### What more training does
#
# Each line traces one label's out-of-sample IC as epochs are added, and the shape to read is where
# it reaches its highest point.
#
# This is the one member of the family where the objective and this axis measure the same kind of
# thing - the network is fitted against the return and scored on how it ranks the return - so a
# line that turns down is the ordinary story of a model beginning to fit its training window rather
# than the objective-versus-metric gap that
# [`08c_stochastic_discount_factor`](08c_stochastic_discount_factor.ipynb) has by construction.
# That makes this the cleanest checkpoint path in the family to read, and the right one to compare
# the others against.

# %%
curves = catalog.filter("full_coverage").sort("label", "checkpoint_value")
fig = go.Figure()
for label in panel_labels:
    series = curves.filter(pl.col("label") == label).sort("checkpoint_value")
    fig.add_trace(
        go.Scatter(
            x=series.get_column("checkpoint_value").to_list(),
            y=series.get_column("ic_mean").to_list(),
            mode="lines+markers",
            name=f"{label} ({'primary' if label == primary else 'variant'})",
            line=dict(
                color=COLORS["blue"] if label == primary else COLORS["slate"],
                width=2 if label == primary else 1.5,
            ),
            marker=dict(size=6),
        )
    )
fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
fig.update_xaxes(title_text="Training epochs completed")
fig.update_yaxes(title_text="Mean IC (validation)")
fig.update_layout(
    title="Fitting the return directly makes the training path readable",
    height=460,
    width=900,
    margin=dict(t=90),
    legend=dict(title_text="Label"),
)
# Where each line peaks is a fact about the frame, so the alt text counts it rather than asserting
# a shape the next run may not reproduce.
peaks = (
    curves.group_by("label")
    .agg(
        peak_checkpoint=pl.col("checkpoint_value").sort_by("ic_mean", descending=True).first(),
        first_checkpoint=pl.col("checkpoint_value").min(),
        last_checkpoint=pl.col("checkpoint_value").max(),
    )
    .sort("label")
)
peak_text = "; ".join(
    f"{row['label']} peaks at epoch {row['peak_checkpoint']} of "
    f"{row['first_checkpoint']} to {row['last_checkpoint']}"
    for row in peaks.iter_rows(named=True)
)
show_plotly_with_alt(
    fig,
    "Line chart of mean validation information coefficient against training epochs completed, one "
    "line per label with a marker at each declared checkpoint: the primary target in dark navy and "
    f"the two variants in slate, over a dashed zero line. Counted from the frame: {peak_text}.",
)

# %% [markdown]
# ### How much the stopping point is worth
#
# The frame below puts the range a label's IC covers across its own checkpoints against the spread
# across labels at a fixed amount of training. That is the quantity deciding whether the stopping
# point is a decision worth making carefully or one being made by noise, and both sides are
# computed within this model so the comparison is not smuggling in a difference between models.

# %% tags=["results"]
final_epoch = int(curves.get_column("checkpoint_value").max())
checkpoint_vs_label = (
    curves.group_by("label")
    .agg(
        ic_min=pl.col("ic_mean").min(),
        ic_max=pl.col("ic_mean").max(),
        ic_final=pl.col("ic_mean").filter(pl.col("checkpoint_value") == final_epoch).first(),
        scored_months=pl.col("ic_n_days").max(),
    )
    .with_columns(checkpoint_range=pl.col("ic_max") - pl.col("ic_min"))
    .sort("label")
)
across_labels = float(
    checkpoint_vs_label.get_column("ic_final").max()
    - checkpoint_vs_label.get_column("ic_final").min()
)
print(f"compared at {final_epoch} epochs; spread across labels there: {across_labels:.4f}")
checkpoint_vs_label

# %% [markdown]
# ## 5. What to notice
#
# **This model is the family's control, and that is why it is published rather than assumed to
# lose.** It keeps the bottleneck and drops everything else - no conditional-factor structure, no
# pricing condition. So reading it against
# [`08b_conditional_autoencoder`](08b_conditional_autoencoder.ipynb), which has the same kind of
# network and the same bottleneck width, isolates what the factor structure is worth on this data.
# A structured model that does not beat its own unstructured control has not earned the structure.
#
# **The four members are two pairs, not four points on a line.**
# [`08a_ipca`](08a_ipca.ipynb) and `08b` are the two-stage models and differ in whether the loading
# map is linear. `08c` and this one both break the two stages, from opposite ends - one because it
# prices directly and one because it predicts directly. Ranking all four on one number would hide
# that, which is part of why ranking happens in
# [`10_model_analysis`](10_model_analysis.ipynb) with the whole population rather than here.
#
# **The bottleneck is the only latent-factor assumption left, and it is a real one.** Forcing 57
# characteristics through a small number of units asserts that the useful information in them is
# low-dimensional. That is testable - widen the bottleneck and see - and it is not tested here,
# because the width is declared and a search over validation IC is what this arrangement exists to
# avoid.
#
# **None of this selects anything.** IC measures whether predictions rank firms correctly, not
# whether a strategy trading them makes money after costs and turnover. Selection is on validation
# backtest Sharpe in [`11_backtest`](11_backtest.ipynb), where the checkpoint is part of what is
# selected.
#
# **Known limitations.** The bottleneck width, the training budget, the checkpoint interval and the
# network's shape are declared rather than chosen. Fitting directly against the return means this
# model inherits the target's shape problem that [`06_gbm`](06_gbm.ipynb) diagnosed - a monthly
# cross-section of individual firms is heavy-tailed, and a squared-error objective is steered by
# exactly the observations a rank correlation is indifferent to - which is one reason to read its
# `fwd_ret_1m` and `fwd_ret_1m_win` rows against each other. The IC is an average of monthly rank
# correlations with no adjustment for serial dependence, so `ic_t` is a diagnostic rather than a
# test. And every number is measured on validation folds that have been read many times over by the
# time a case study reaches this notebook.

# %% [markdown]
# **Next**: [`09_causal_dml`](09_causal_dml.ipynb) stops predicting altogether and asks a different
# question of the same panel - what the effect of one characteristic on the forward return is once
# the others are controlled for, which is not what any model in this family estimates.
