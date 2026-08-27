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
# # Option analytics: the factor model that is allowed to see the target
#
# Every model in this family so far has been denied one thing. PCA, IPCA and the conditional
# autoencoder are all fitted by reproducing returns the panel already realized;
# [`11d_stochastic_discount_factor`](11d_stochastic_discount_factor.ipynb) is fitted against a
# no-arbitrage moment condition. None of those objectives mentions the forward return the notebooks
# are scored on, which is why
# [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb) can reconstruct the
# cross-section better and better without its IC following.
#
# **The supervised autoencoder adds the forward return to the autoencoder's loss.** The bottleneck
# still has to describe the cross-section, and it now also has to predict. That is the idea the
# notebook is for.
#
# It is not, however, a controlled comparison against
# [`11c`](11c_conditional_autoencoder.ipynb), and the presets are worth reading before the results
# are. The two share the characteristics, the folds, the labels and the epoch schedule, and they
# differ in three things besides the loss: the bottleneck is 96 factors here against IPCA-scale 5
# there, the encoder is `(896, 448, 448, 256)` against `(32,)`, and the learning rate is 1e-4
# against 1e-3. A gap between their rows in
# [`13_model_analysis`](13_model_analysis.ipynb) is therefore not attributable to supervision
# alone. The training log's `sae (K=5)` is misleading on the same point: `run_sae_fold` discards
# `n_factors` outright, and `bottleneck_dim` is what sets the width.
#
# **It is the model in this family most able to overfit, for the same reason it is the one most able
# to fit.** A reconstruction objective is a constraint - it forces the bottleneck to explain
# co-movement whether or not that helps the forecast. Adding the target relaxes the constraint in
# the direction of the thing being scored, on two folds of a cross-section that has been read many
# times. Whether the relaxation buys anything on the validation split is what the table below
# measures.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - State the one difference between this model and the conditional autoencoder.
# - Explain why supervising a factor model is a weaker constraint rather than more information.
# - Read a pair of notebooks as a controlled comparison, and say what is being controlled.
# - Say why a higher validation IC here would not settle whether supervision helped.
#
# **Book reference**: Chapter 14, Section 14.4 (autoencoder factor models) and Section 14.5 (what a
# supervised bottleneck changes). Chapter 6, Section 6.7 (Search accounting and run logging)
# introduces the run log this notebook writes into.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices,
# [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds,
# [`11_latent_factors`](11_latent_factors.ipynb) introduces the family, and
# [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb) is the unsupervised model this
# one is read against.
#
# **What it writes**: one training run per label and ten complete validation prediction sets per
# label - one per checkpoint - in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a population named for this model. The family splits across
# five notebooks, so each publishes its own population rather than one shared one.
# [`13_model_analysis`](13_model_analysis.ipynb) compares them against the other families and
# [`14_backtest`](14_backtest.ipynb) selects on validation backtest Sharpe. **Selection happens
# there, not here.**

# %%
"""Fit the declared option-analytics supervised-autoencoder population on the walk-forward folds."""

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
DEVICE: str = ""

MODEL_NAME = "sae"

# %%
study = open_study(
    "sp500_equity_option_analytics",
    execution_tier=EXECUTION_TIER,
    workspace=WORKSPACE or None,
    entry_point="11e_supervised_autoencoder",
)

# %% [markdown]
# ## 1. Which labels, and what the configuration says
#
# Every label whose training menu declares `latent_factors:` is fitted, and the cell below reads
# which those are rather than this sentence asserting a count that would go stale the moment a
# sixth label declared the family. What the names mean is the part prose has to supply:
# `fwd_ret_5d` is the stock's total return over the five trading days after the decision date,
# `fwd_ret_10d` the same over ten, and `fwd_ret_risk_adj_5d` the five-day return divided by a
# measure of its own dispersion. A `fwd_dir_*` classification label declaring only linear and
# gradient boosting is absent here rather than dropped.
#
# The label matters more here than anywhere else in this family. For the other four models the label
# decides only what the predictions are scored against; the fit itself is identical across the three
# rows. Here the label is *in the objective*, so these are three different models rather than one
# model read three ways.

# %%
declared_labels(study, "latent_factors")

# %% [markdown]
# `case_studies/config/sae/sae.yaml` declares `n_factors: 5`, `n_epochs: 50` and
# `checkpoint_interval: 5`. The CAE and SAE presets are not otherwise controlled: their bottleneck
# widths, hidden layers, learning rates, and losses differ, as Section 5 shows explicitly.
#
# The case study may override any of these under `modeling.latent_factors.model_kwargs` in
# `config/setup.yaml`, which wins where it is given; this one declares entries for `ipca`, `sdf`,
# and `sae`. The SAE override fixes the production batch size at 10,000 rows, and that value enters
# the training identity with the other model arguments. A reduced run may then override the values
# again through `PREVIEW_REDUCTIONS`, where the reduction becomes part of the preview identity.

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
# catalog. The family is split across five notebooks and each publishes one model, so the complete
# catalog is five times what this one publishes, and comparing against it would report every
# canonical run as narrowed.

# %%
declared = load_model_configs(study, "latent_factors", config_names=[MODEL_NAME])
if set(configs.get_column("label")) != set(declared.get_column("label")) and not POPULATION_NAME:
    raise ValueError(
        f"this run fits {configs.height} of the {declared.height} declared labels, so it cannot "
        "publish the canonical population; pass POPULATION_NAME to give it its own"
    )

# %% [markdown]
# ### Where this one runs
#
# `config/setup.yaml` declares the latent-factor family on CUDA. That declaration exists because the
# device sits inside the hashed computation rather than beside it as provenance: with no value
# declared, the library falls back to whichever device the machine happens to have, and the training
# identity becomes a property of the host. [`11a_pca`](11a_pca.ipynb) and
# [`11b_ipca`](11b_ipca.ipynb) override it to `cpu` because their runners are numpy and scipy and
# take no device at all. This one is a torch model trained on the declared device, so it takes the
# family's value unchanged and passes no override.
#
# A machine without a card cannot run the declared device at all, and the library refuses to
# substitute one (`case_studies/utils/latent_factors/library_bridge.py:53-54`). `DEVICE` names the
# device such a run fits on. It travels with `POPULATION_NAME`, because the device it names is
# inside the training identity: the run computes a different member set and has to publish under
# its own name rather than into the population the declaration describes.

# %%
overrides: dict = {"device": DEVICE} if DEVICE else {}
print(f"training device: {DEVICE or 'the family declaration in config/setup.yaml (cuda)'}")

# %% [markdown]
# ## 2. Binding the declarations to the data
#
# A menu entry names an estimator, a factor count and an epoch schedule. It does not say which
# feature columns exist today, where the walk-forward folds fall, or which symbol-date pairs have
# both a return and a label. **Planning** works all of that out - and every training and prediction
# identity with it - without fitting anything. Four things to check:
#
# - **`feature_count` and `eligible_rows` agree across the rows.** A row that differs is a label
#   measured on a different sample from its neighbours. They differ *between* labels, because a
#   ten-day forward window runs out earlier than a five-day one.
# - **`folds` is the same everywhere**, and equals the number of walk-forward splits
#   [`05_evaluation`](05_evaluation.ipynb) established.
# - **`validation_start` and `validation_end` bracket the development sample**, with none of the
#   held-out 2021 tail visible.
# - **`checkpoints` is 10**, matching [`11c`](11c_conditional_autoencoder.ipynb) exactly, so the two
#   are read at the same points on their schedules. That is one axis held level, not a controlled
#   comparison: section 5 lists the three the presets do not hold.

# %%
requests = model_requests(
    study,
    configs,
    execution_tier=EXECUTION_TIER,
    overrides=overrides,
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
# menu declares would still print an IC table and still register its rows; what it would not do is
# announce the gap.

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
# 1. passes each stock's characteristics through a network that outputs `n_factors` exposures,
# 2. solves for the factor returns that, applied to those exposures, best reproduce the training
#    window's realized returns,
# 3. scores the reconstruction error *and* the error against the fold's forward-return label, and
#    backpropagates both into the network,
# 4. forecasts each validation date by applying the fitted network to that date's characteristics.
#
# Step 3 is the entire difference from [`11c`](11c_conditional_autoencoder.ipynb), whose step 3
# carries the first of those two terms and not the second.
#
# **This notebook passes the plan rather than resolved requests, and on this family that changes
# what the plan table can show rather than how the fitting is ordered.** The linear, boosted and
# TabM adapters each provide a batch runner that prepares one fold set for several configurations;
# the latent-factor adapter provides none, so every request is fitted on its own whichever path is
# taken. There would be nothing to share in any case: this notebook publishes one model against
# three labels, and each label is a different set of rows - and here, a different objective too.
# What passing the plan costs is `eligible_entities`, which needs the eligibility keys themselves;
# `eligible_rows` above moves whenever the universe does, so the check that column existed for is
# still made.
#
# **What the call publishes is a population**: a named, immutable list of the prediction sets it
# will produce, written down before the first fit. Afterwards every member must exist and be
# complete, which is what makes the downstream comparison well defined.


# %%
def catalog_labels(execution) -> int:
    """Number of distinct labels the execution published, read from its catalog rows."""
    return execution.catalog_rows.get_column("label").n_unique()


population_name = POPULATION_NAME or f"sp500_equity_option_analytics-{MODEL_NAME}-validation-v1"
execution, population = run_model_population(
    study, plan, population_name=population_name, supersedes=SUPERSEDES_POPULATION or None
)

# No per-fold counts: this family's runner reports none, and two zeros would read as a failed fit.
print(f"{len(execution.runs)} training runs across {catalog_labels(execution)} labels")
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
# study = open_study("sp500_equity_option_analytics", workspace="~/ml4t-experiments")
# configs = load_model_configs(
#     study, "latent_factors", labels=["fwd_ret_5d"], config_names=["sae"]
# )
# requests = model_requests(study, configs)
# plan = plan_models(study, requests=requests)
# execution, population = run_model_population(study, plan, population_name="my-sae-v1")
# ```
#
# To change the bottleneck width or the epoch schedule for your own run, edit
# `case_studies/config/sae/sae.yaml`, or declare the value under
# `modeling.latent_factors.model_kwargs.sae` in `config/setup.yaml` to override the preset for this
# case study alone - and change `cae` to match, or the pair stops being a controlled comparison.
# Note which of those you want: the preset is shared by every case study that declares `sae`, so
# editing it moves the training identity of all of them. Either way the result registers as a new
# row beside the old one rather than replacing it.
# [`RUN_LOG.md`](../RUN_LOG.md#running-your-own-configurations) covers the rest.

# %% [markdown]
# ## 4. What came out
#
# One row per label and checkpoint. The **information coefficient** is the rank correlation, on one
# validation date, between the stocks ordered by the model's prediction and the stocks ordered by
# the return they went on to earn.
#
# `ic_mean` aggregates that **over folds, not over days**: each fold's own mean IC is computed and
# those are averaged with equal weight (`latent_factors/cv.py`, and the convention is stated in
# `registry/metrics.py`). With folds of unequal length the fold mean and the pooled daily mean are
# different numbers, and this column is the first.
#
# `ic_n_days` is how many validation dates produced a defined correlation, and it decides which rows
# are comparable with each other. `auc_scored_against` says what the AUC column was scored against:
# a regression row has no classes of its own and is scored as a ranking signal against a declared
# direction sibling, so `fwd_ret_5d` is scored against `fwd_dir_5d` and `fwd_ret_10d` against
# `fwd_dir_10d`. `fwd_ret_risk_adj_5d` declares no sibling and carries no AUC; null there means not
# computed, not zero.
#
# The published catalog is checked against the population planned before fitting rather than against
# its own row count, because a run that lost a member would otherwise report a shorter table and
# nothing else.

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
).sort("label", "checkpoint_value")

if set(catalog.get_column("prediction_hash")) != set(plan.expected_prediction_hashes):
    raise RuntimeError("the published catalog differs from the population planned before fitting")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("a partial supervised-autoencoder prediction set cannot pass to backtesting")
if catalog.select("label", "config_name", "checkpoint_value").n_unique() != catalog.height:
    raise RuntimeError("each label and checkpoint must identify one prediction set")

primary = primary_label(study)
present = sorted(set(catalog.get_column("label")))
# The primary label leads when it was fitted. A subset run that leaves it out orders by whichever
# label it did fit rather than by one that is not there.
ordered_labels = [label for label in [primary] if label in present] + [
    label for label in present if label != primary
]
print(f"{catalog.height} candidate models across {len(ordered_labels)} labels")
catalog.select(
    "label",
    "checkpoint_value",
    "ic_mean",
    "ic_std",
    "ic_t",
    "ic_n_days",
    "auc_mean_daily",
    "auc_scored_against",
)

# %% [markdown]
# ### Where each label's schedule reached its highest IC
#
# Ten checkpoints per label is too many rows to read as a table, so this reduces each label to the
# epoch with its highest validation IC. `epochs_above_zero` counts how many of the ten put the IC on
# the positive side at all, which separates a model that is consistently weak from one that crossed
# zero once. `ic_t` is the t-statistic across those fold means rather than a
# Newey-West statistic on the daily series - the registry keeps that separately as `ic_t_hac`, and
# it is the inferential one. Either way `ic_t` is a diagnostic and not a selection rule - the series is short, overlapping multi-day returns make successive days
# dependent, and the folds have been read many times over by the time a case study reaches this
# notebook.
#
# **Reading the highest of ten checkpoints is itself a selection**, and it matters more here than in
# [`11c`](11c_conditional_autoencoder.ipynb). There the objective never saw the target, so a high
# checkpoint was at least measuring something the fit was not aiming at. Here the fit is aiming at
# the target, on data adjacent to the split the IC is computed on, so the highest of ten is the
# number most likely to flatter the model. Nothing downstream inherits it.

# %% tags=["results"]
by_label = (
    catalog.sort("ic_mean", descending=True)
    .group_by("label", maintain_order=True)
    .agg(
        peak_epoch=pl.col("checkpoint_value").first(),
        peak_ic=pl.col("ic_mean").first(),
        ic_t=pl.col("ic_t").first(),
        epochs_above_zero=(pl.col("ic_mean") > 0).sum(),
        checkpoints=pl.len(),
        scored_dates=pl.col("ic_n_days").first(),
    )
    .sort("peak_ic", descending=True)
)
by_label

# %% [markdown]
# ### What supervision does to the schedule
#
# One line per label across the published schedule, on one axis, drawn exactly as
# [`11c`](11c_conditional_autoencoder.ipynb) draws its own so the two charts can be laid side by
# side. Same axes, same schedule, same labels: the difference between the two figures is the
# difference between the two objectives, which is the only comparison in this family that is
# controlled to a single term in a loss function.

# %%
fig = go.Figure()
for label in ordered_labels:
    rows = catalog.filter(pl.col("label") == label).sort("checkpoint_value")
    fig.add_trace(
        go.Scatter(
            x=rows.get_column("checkpoint_value").to_list(),
            y=rows.get_column("ic_mean").to_list(),
            mode="lines+markers",
            name=label,
            line=dict(color=COLORS["blue"] if label == primary else COLORS["recede"]),
            marker=dict(color=COLORS["blue"] if label == primary else COLORS["recede"]),
        )
    )
fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
fig.update_yaxes(title_text="Mean IC (validation)")
fig.update_xaxes(title_text="Training epoch at which the checkpoint was published")
fig.update_layout(
    title="The same schedule, fitted against a loss that includes the target",
    height=460,
    width=860,
    margin=dict(t=90),
)
# The direction of each line is a fact about the frame, so the alt text counts it rather than
# asserting a shape the next run may not reproduce.
spans = "; ".join(
    f"{row['label']} spans {row['lo']:+.4f} to {row['hi']:+.4f}"
    for row in catalog.group_by("label")
    .agg(lo=pl.col("ic_mean").min(), hi=pl.col("ic_mean").max())
    .sort("label")
    .iter_rows(named=True)
)
show_plotly_with_alt(
    fig,
    "Line chart of mean validation information coefficient against the training epoch at which "
    "each checkpoint was published, one line per label on one axis, the primary target in dark "
    f"navy and the two variants in light slate, with a dashed zero line. Counted from the frame: "
    f"{spans}.",
)

# %% [markdown]
# ## 5. What to notice
#
# **This notebook and [`11c`](11c_conditional_autoencoder.ipynb) are not a controlled pair, and the
# presets say so.** They share the characteristics, the folds, the labels, the epoch schedule and
# the device, and the panel they fit is the same down to its raggedness - both report
# `ragged train=493/475, max_N=503`. They differ in four things, not one:
#
# | | `11c` cae | `11e` sae |
# |---|---|---|
# | bottleneck / factors | 5 | 96 |
# | hidden units | `(32,)` | `(896, 448, 448, 256)` |
# | learning rate | 1e-3 | 1e-4 |
# | loss | reconstruction | reconstruction + forward return |
#
# So a gap between their rows in [`13_model_analysis`](13_model_analysis.ipynb) is not
# attributable to supervision. Reading it that way would credit a loss term for what a
# nineteen-fold wider bottleneck and a four-layer encoder may well have done on their own. Making
# the pair controlled means matching the three architectural values, which is a change to the
# presets and to both populations' identities rather than a change to this prose - so it is stated
# here rather than assumed away.
#
# **Supervising a factor model is a weaker constraint, not more information.** Both models see the
# same characteristics; neither sees anything the other does not. What changes is what the
# bottleneck is required to be good at. Reconstruction forces it to explain co-movement whether that
# helps or not, and that requirement is precisely what stops it chasing the target - so removing it
# can only help where the co-movement constraint was the thing in the way.
#
# **A higher IC here would not settle whether supervision helped.** Two folds, one target, one
# factor count, one architecture, no repetition across seeds. The difference between two numbers
# whose intervals overlap is not evidence about the objective, and this family's intervals are wide
# for the reason every notebook in it repeats: overlapping multi-day returns on a short validation
# window. The comparison is set up so the question is answerable, not so this run answers it.
#
# **Ten checkpoints are ten observations of one fit, not ten candidates.** They are published so
# that [`14_backtest`](14_backtest.ipynb) can carry the entire schedule into its own sweep and
# select on validation backtest Sharpe. Reading the highest IC off this notebook and stopping there
# would be selecting on a quantity that measures ranking rather than money, before costs and
# turnover have been applied to any of it.
#
# **Known limitations.** Two folds, one of them validating on 2020, a year in which the
# cross-section co-moved unlike any other in the sample - which bears on a factor model more
# directly than on a supervised one, and this model is both. The weight between the reconstruction
# and prediction terms is the library's default and is not declared, searched, or reported here, so
# "supervised" names a family of models of which exactly one was fitted. The factor count and the
# network's shape are declared rather than chosen. The IC carries no adjustment for the serial
# dependence overlapping multi-day returns create, so it is a ranking diagnostic rather than a test.
# And training is stochastic, so a rerun reproduces the identity but not necessarily the third
# decimal place.
#
# **Next**: [`11_latent_factors`](11_latent_factors.ipynb) reads the five populations this family
# published as one table, and [`12_causal_dml`](12_causal_dml.ipynb) leaves prediction behind for a
# different question - what a change in one feature does to the return, rather than what the
# features jointly predict.
