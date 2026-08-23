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
# # Firm characteristics: pricing the cross-section instead of forecasting it
#
# [`08a_ipca`](08a_ipca.ipynb) and [`08b_conditional_autoencoder`](08b_conditional_autoencoder.ipynb)
# share a two-stage shape: estimate a factor representation, then apply it. The **stochastic
# discount factor** model has no such split. It learns one object - a weight on each firm-month that
# makes a portfolio of the cross-section price everything else correctly - from a no-arbitrage
# condition rather than from a forecasting objective, so there is no factor-return history to hand
# to a second stage at all.
#
# The condition it minimizes is that every asset's return, discounted by the SDF, should have zero
# expected value. Written out, that is a set of moment conditions, and the model has three moving
# parts fitted in phases: an unconditional stage that fixes the SDF weights, a conditional stage
# that lets the weights depend on each firm's characteristics and on the state of the market, and a
# moment network that searches for the test assets on which the current SDF prices worst - which is
# what makes the fit adversarial rather than a plain least-squares problem.
#
# **The forecast is a by-product, and that is the point of reading this notebook against the two
# before it.** A firm's predicted return here is what the fitted pricing object implies it should
# earn, not something the fit was asked to get right. So a model that prices well and ranks badly
# is not a contradiction, and the IC below measures the second while the objective optimized the
# first.
#
# **Macro context enters here and nowhere else in the family.** `config/setup.yaml` declares eleven
# unrevised daily rate and volatility series under `modeling.latent_factors.macro_series`, with a
# one-day availability lag. Only unrevised series are eligible: a finalized quarterly figure is not
# what anyone could have read at the decision date, and the lag is what stops a same-close
# observation reaching a forecast made at that close.
#
# **Learning objectives**
#
# - Separate a pricing objective from a forecasting one, and say what each is optimizing.
# - Read a checkpoint surface whose epoch numbers run across training phases rather than within one.
# - See why a point-in-time constraint on macro inputs is a data decision rather than a modelling
#   preference.
#
# **Book reference**: Chapter 14, Section 14.7 (The stochastic discount factor and the supervised
# autoencoder models). Chapter 6, Section 6.7 (Search accounting and run logging) introduces the
# run log this notebook writes into.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb),
# [`04_evaluation`](04_evaluation.ipynb), [`08_latent_factors`](08_latent_factors.ipynb) and
# [`08b_conditional_autoencoder`](08b_conditional_autoencoder.ipynb), the two-stage model this one
# departs from.
#
# **What it writes**: one training run per label and one complete validation prediction set per
# label and epoch checkpoint, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a population named for this model. The family splits across
# four notebooks, so each publishes its own population rather than one shared one.
# [`11_backtest`](11_backtest.ipynb) reads them and selects on validation backtest Sharpe.
# **Selection happens there, not here**, and the checkpoint is part of what is selected.

# %%
"""Fit the declared firm-characteristics stochastic discount-factor population."""

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

MODEL_NAME = "sdf"

# %%
study = open_study(
    "us_firm_characteristics", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None
)

# %% [markdown]
# ## 1. Which labels, and what the configuration says
#
# Every label whose training menu declares `latent_factors:` is fitted, and all three do:
# `fwd_ret_1m`, `fwd_ret_1m_win` - the same return with each month's cross-section clipped at its
# own tails - and `fwd_class_1m`, which turns that return into a class.

# %%
declared_labels(study, "latent_factors")

# %% [markdown]
# `case_studies/config/sdf/sdf.yaml` declares one epoch budget per phase rather than one for the
# model, and the checkpoint surface is built from them. The unconditional stage runs
# `n_epochs_unc` epochs and the conditional stage `n_epochs_cond` after it, so a checkpoint's
# number is **cumulative across phases**: the first is the end of the unconditional stage, and each
# one after it is that plus a point inside the conditional stage. Reading a checkpoint value as an
# epoch within one training run would put them in the wrong places relative to each other.
#
# `checkpoint_epochs` lists the points to save at rather than an interval, because the phases are
# of different lengths and an interval would land differently in each. `model_kwargs.sdf` in
# `config/setup.yaml` narrows that list for this case study.

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
# Planning reads the label and feature files, loads and lags the declared macro series, computes
# the fold boundaries from the walk-forward parameters in `config/setup.yaml`, works out the exact
# rows each fit must predict, and derives every training and prediction identity - without fitting
# anything. Four things to check:
#
# - **`feature_count` and `eligible_rows` agree across the rows.** A row that differs is a label
#   measured on a different sample from its neighbours.
# - **`folds` is the same everywhere**, and equals the number of walk-forward splits
#   [`04_evaluation`](04_evaluation.ipynb) established.
# - **`validation_start` and `validation_end` bracket the development sample**, with none of the
#   held-out 2016 tail visible.
# - **`checkpoints` is how many training states each label will publish predictions for**, at the
#   cumulative epoch numbers described above.
#
# The macro panel does not appear in `feature_count`. It is not a per-firm characteristic: it
# enters the conditional stage as the state the SDF weights are allowed to depend on, and its
# digest is recorded in the training identity so a different vintage is a different computation.

# %%
requests = model_requests(
    study,
    configs,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
    notebook="08c_stochastic_discount_factor",
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
        "the plan does not match the loaded stochastic discount-factor menu; "
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
# 1. takes the firm-months inside that fold's training window, and the macro observations dated at
#    or before each of those months,
# 2. fits the unconditional stage: SDF weights that do not vary with the state, trained to make the
#    discounted returns price to zero,
# 3. fits the conditional stage on top, letting the weights depend on each firm's characteristics
#    and on the macro state, alternating with a moment network that looks for the portfolios the
#    current SDF prices worst,
# 4. saves the fitted state at each declared checkpoint and, from each one, produces the return
#    each validation firm-month is implied to earn.
#
# Step 4 is what makes one fit produce many results. Fold predictions are concatenated into one
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

print(f"{len(execution.runs)} configurations fitted")
print(f"population {population.name}: {len(population.members)} prediction sets")

# %% [markdown]
# Re-running this notebook unchanged costs the time it takes to read the data. Every identity is
# re-derived from the inputs, the registry already holds the matching rows, and the runner returns
# the stored result rather than fitting again.
#
# **There are no fold counts above, and their absence is the honest reading rather than an
# omission.** [`05_linear`](05_linear.ipynb), [`06_gbm`](06_gbm.ipynb) and
# [`07_tabular_dl`](07_tabular_dl.ipynb) each print folds fitted against folds served from the
# registry, because their runners record `fitted_folds` and `reused_folds` per run. The
# latent-factor runner records neither - its result carries no diagnostics at all, so the only
# per-run keys are the status and the training hash. Printing two zeros here would say every fold
# came from cache, which is the opposite of what nothing-recorded means.
#
# ### Running configurations of your own
#
# The published run log is read-only. To add runs, open the study against a workspace, which holds
# its own registry and artifacts and reads the same labels and features:
#
# ```python
# study = open_study("us_firm_characteristics", workspace="~/ml4t-experiments")
# configs = load_model_configs(study, "latent_factors", labels=["fwd_ret_1m"], config_names=["sdf"])
# requests = model_requests(study, configs)
# plan = plan_models(study, requests=requests)
# execution, population = run_model_population(study, plan, population_name="my-sdf-v1")
# ```
#
# The phase budgets and the checkpoint list live in `case_studies/config/sdf/sdf.yaml`, narrowed
# for this case study by `model_kwargs.sdf` in `config/setup.yaml`; the macro series and their
# availability lag are declared beside it. Each changes this configuration's identity, so the
# result registers as a new row beside the old one rather than replacing it.
# [`RUN_LOG.md`](../RUN_LOG.md#running-your-own-configurations) covers the rest.

# %% [markdown]
# ## 4. What came out
#
# One row per label and epoch checkpoint. `ic_mean` is the **information coefficient**: in each
# validation month, rank the firms by the model's prediction, rank them by the return they went on
# to earn, correlate the two rankings, and average that monthly correlation over the validation
# period. It measures ranking, which is not what this model was fitted to do.
#
# **Every count and aggregate below is keyed on `(label, checkpoint_value)`.** Grouping on the
# checkpoint alone would average across targets; grouping on the label alone would collapse the
# phase structure the checkpoint numbers carry.
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
    raise RuntimeError("a partial stochastic discount-factor checkpoint cannot pass to backtesting")
if catalog.select("label", "config_name", "checkpoint_value").n_unique() != catalog.height:
    raise RuntimeError("each label and checkpoint must identify one prediction set")
if catalog.get_column("checkpoint_value").null_count():
    raise RuntimeError("every checkpoint must name the cumulative epoch it was taken at")

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
# ### The ranking across the two training phases
#
# Each line traces one label's out-of-sample IC against the cumulative epoch count, so the leftmost
# point is the end of the unconditional stage and everything to its right is the conditional stage
# refining it. That boundary is the one place on this chart where the model changes shape rather
# than merely trains longer, and it is marked.
#
# What the chart cannot say is whether the fit is pricing better, because the vertical axis is a
# ranking measure and the objective is a pricing one. A line that falls after the phase boundary is
# a model whose ranking got worse while it was being asked to price better, which is a real thing
# for this family rather than a failure to converge.

# %%
curves = catalog.filter("full_coverage").sort("label", "checkpoint_value")
phase_boundary = int(curves.get_column("checkpoint_value").min())
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
fig.add_vline(
    x=phase_boundary,
    line_width=1,
    line_dash="dot",
    line_color=COLORS["copper"],
    annotation_text="conditional stage begins",
    annotation_position="top right",
)
fig.update_xaxes(title_text="Cumulative training epochs across both phases")
fig.update_yaxes(title_text="Mean IC (validation)")
fig.update_layout(
    title="A pricing objective is not asked to rank, and the path shows it",
    height=460,
    width=900,
    margin=dict(t=90),
    legend=dict(title_text="Label"),
)
# Which direction each line moves after the phase boundary is a fact about the frame, so the alt
# text reads it rather than asserting a shape the next run may not reproduce.
moves = (
    curves.group_by("label")
    .agg(
        at_boundary=pl.col("ic_mean").sort_by("checkpoint_value").first(),
        at_end=pl.col("ic_mean").sort_by("checkpoint_value").last(),
    )
    .sort("label")
)
move_text = "; ".join(
    f"{row['label']} {'rises' if row['at_end'] > row['at_boundary'] else 'falls'} across the "
    "conditional stage"
    for row in moves.iter_rows(named=True)
)
show_plotly_with_alt(
    fig,
    "Line chart of mean validation information coefficient against cumulative training epochs, one "
    "line per label with a marker at each declared checkpoint: the primary target in dark navy and "
    "the two variants in slate, over a dashed zero line, with a dotted copper vertical line where "
    f"the conditional stage begins. Counted from the frame: {move_text}.",
)

# %% [markdown]
# ### How much the stopping point is worth
#
# The frame below puts the range a label's IC covers across its own checkpoints against the spread
# across labels at the final one. That is the quantity deciding whether the stopping point is a
# decision worth making carefully or one being made by noise, and both sides are computed within
# this model so the comparison is not smuggling in a difference between models.

# %% tags=["results"]
final_epoch = int(curves.get_column("checkpoint_value").max())
checkpoint_vs_label = (
    curves.group_by("label")
    .agg(
        ic_min=pl.col("ic_mean").min(),
        ic_max=pl.col("ic_mean").max(),
        ic_at_boundary=pl.col("ic_mean")
        .filter(pl.col("checkpoint_value") == phase_boundary)
        .first(),
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
print(f"unconditional stage ends at epoch {phase_boundary}; last checkpoint is {final_epoch}")
print(f"spread across labels at the last checkpoint: {across_labels:.4f}")
checkpoint_vs_label

# %% [markdown]
# ## 5. What to notice
#
# **The objective here is not the metric here, and that gap is the lesson rather than a defect.**
# The fit minimizes pricing error under a no-arbitrage condition; `ic_mean` measures whether the
# implied returns rank the cross-section. Those can move together and they need not. Reading a low
# IC as the model failing would be reading it against a target it was never given - and reading a
# high one as vindication of the pricing would be the same error in the other direction.
#
# **Where this sits relative to the two notebooks before it.**
# [`08a_ipca`](08a_ipca.ipynb) and [`08b_conditional_autoencoder`](08b_conditional_autoencoder.ipynb)
# both estimate a factor representation and then apply it, and differ only in whether the map from
# characteristics to exposures is linear. This model has no such two-stage split to make: the
# pricing object is what is learned, and the forecast falls out of it. So the three of them are not
# three points on one axis, and the family's four members are better read as two pairs.
#
# **The macro constraint is a data decision and it costs coverage.** Only unrevised daily series
# are eligible, which rules out most published macro aggregates, and the one-day availability lag
# rules out the same-close reading. Both are what make the conditional stage's state variable
# something a reader could actually have observed at the decision date. A version of this model
# conditioned on revised quarterly data would look better and would not be a forecast.
#
# **None of this selects anything.** IC measures whether predictions rank firms correctly, not
# whether a strategy trading them makes money after costs and turnover. Selection is on validation
# backtest Sharpe in [`11_backtest`](11_backtest.ipynb), where the checkpoint is part of what is
# selected.
#
# **Known limitations.** The phase budgets, the checkpoint list, the factor count and the network
# widths are declared rather than chosen, and nothing here searches over them. The adversarial
# moment network makes the fit sensitive to its own optimization path in a way a least-squares
# problem is not, so two runs at different thread counts are not guaranteed to agree to full
# precision even at a fixed seed - the runtime is inside the training identity for that reason. The
# IC is an average of monthly rank correlations with no adjustment for serial dependence, so `ic_t`
# is a diagnostic rather than a test. And every number is measured on validation folds that have
# been read many times over by the time a case study reaches this notebook.

# %% [markdown]
# **Next**: [`08d_supervised_autoencoder`](08d_supervised_autoencoder.ipynb) is the other member
# that breaks the two-stage shape, and it breaks it from the opposite end: no pricing condition and
# no factor intermediate, just characteristics mapped to forward returns directly.
