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
# # TabM on the funding panel, and what a network adds that a tree does not
#
# [`06_linear`](06_linear.ipynb) and [`07_gbm`](07_gbm.ipynb) read the same design matrix this
# notebook does: one row per perpetual per settlement, one column per feature, with nothing in
# the table saying the rows are ordered in time. They differ in what they can represent. A
# penalized linear model gives each feature one coefficient and can spread weight across a group
# of near-duplicate columns. A tree ensemble can express an interaction - a condition on one
# feature evaluated inside a region defined by others - but it reaches one by choosing a single
# column at each split, and several columns here carry almost the same information, so which one
# is chosen is close to arbitrary.
#
# A neural network on the same table answers the same question a third way. Its first layer is a
# weighted sum of every feature, so like the linear model it never has to choose among correlated
# columns; the nonlinearity after it lets those sums combine into interactions the linear model
# cannot write down. That is the reason to fit one here, rather than a general preference for
# neural networks: the two properties that pulled against each other in the previous two
# notebooks are not obviously in conflict in this architecture.
#
# **TabM is an ensemble, and the ensemble is the point.** Averaging several independently
# initialized networks is a standard way to make a neural fit on a table less erratic, and the
# cost is normally that you train several networks. TabM trains most of one. A backbone of two
# layers is shared by every member; each member owns only a vector carrying one number per hidden
# unit, which scales the backbone's output element by element, and its own final linear layer.
# The members' predictions are averaged. So `n_members: 4` at `hidden_dim: 64` costs four small
# vectors and four output layers on top of one backbone, not four networks - which is why the
# member count can be raised much further than the width can.
#
# **This notebook fits three of the four declared labels, and two of them are not returns.**
# `fwd_ret_8h` is a regression target. `fwd_dir_8h` is its sign, a binary classification, and
# `fwd_dir_8h_3c` adds a flat class for moves too small to trade - three-way, and deliberately
# unbalanced, because most settlements are small. A classification request therefore resolves
# more than a regression one: the class weights that correct the imbalance are fitted per fold,
# because the balance of a fold is a property of its own training window and not of the panel.
# It also resolves a *continuous* evaluation target, so a classifier's ranking can be scored
# against the return it was trying to sign rather than against its own discrete labels.
#
# **A neural fit has a meaningful state at every epoch**, in the way a boosted model has one at
# every iteration and a linear fit does not. An epoch is one pass over the training rows. These
# configurations train for 200 and save the weights every 25, so each produces eight scoreable
# models rather than one and each is registered separately. What counts downstream is
# configurations times checkpoints, not configurations.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Say what a weight-sharing ensemble holds in common between its members and what it keeps
#   separate, and why that makes *k* members cost far less than *k* networks.
# - Tell apart a regression, a binary and a multiclass request, and say what each additionally
#   resolves before anything is fitted.
# - Explain why class weights are fitted per fold rather than once for the panel, and what would
#   go wrong if a single weighting were carried across folds.
# - Read the epoch schedule out of a declared configuration and say how many scoreable models a
#   run will publish for it.
# - Say why a catalog identity has to bind the device policy as well as the model and seed.
#
# **Book reference:** Chapter 18, deep learning for tabular data.
#
# **Prerequisites:** [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices, and
# [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds. The canonical run
# uses CUDA; the reduced run in CI does not.
#
# **What it writes:** one training run per configuration and one complete validation prediction set
# per checkpoint, grouped under a named population that [`13_backtest`](13_backtest.ipynb) reads
# and selects from on validation backtest Sharpe. **Nothing here ranks anything**, and no number
# printed below decides which model the case study goes on to use.

# %%
import os

import polars as pl

from case_studies.crypto_perps_funding.research_workflow import (
    ALL_LABELS,
    declared_contracts,
    freeze_official_model_population,
    model_request_catalog,
    open_study,
    plan_model_catalog,
    plan_specs,
    run_model_plan,
)

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
SUPERSEDES_POPULATION: str = ""
# The generation of this notebook's own checkpoint population that this run replaces, if any.
# Distinct from SUPERSEDES_POPULATION above, which is the case-wide official model population:
# the two are separate declarations and a refit can move either without moving the other.
SUPERSEDES_MODEL_POPULATION: str = ""
WORKSPACE = os.environ.get("ML4T_OUTPUT_DIR", "")
LABELS = ALL_LABELS
PREVIEW_REDUCTIONS = {}
OVERRIDES = {"class_weight": "balanced", "device": "cuda"}

# %% [markdown]
# ## 1. Resolve targets, imbalance policy, and checkpoints
#
# Nothing is fitted below. The catalog resolves each declared configuration against each label into
# a request with an identity, and the table that follows prints what those requests will actually
# do. Three fields on it repay attention.
#
# `task` is where the three labels stop being interchangeable. A regression request minimizes
# squared error against `fwd_ret_8h`; a binary request fits the sign; the three-class request fits
# a sign with a flat band in the middle. They are different objectives on the same features, and a
# comparison across them is a comparison of what each was asked to do, not of which is better.
#
# `class_weights` is empty for the regression request and populated for the other two, and it is
# resolved **per fold**. The proportion of flat settlements is a property of a particular training
# window, not of the panel: crypto funding regimes are long-lived, and a fold covering a quiet
# stretch has a different balance from one covering a volatile stretch. A single weighting computed
# once over the whole panel would carry each fold a correction fitted partly on the others.
#
# `checkpoint_schedule` is what turns each configuration into several scoreable models. Read the
# epoch count and interval off this table rather than from the configuration file, because it is
# the frozen specification that the run will follow.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
official_population = (
    freeze_official_model_population(study, supersedes=SUPERSEDES_POPULATION or None)
    if EXECUTION_TIER == "canonical"
    else None
)
requests = model_request_catalog("tabular_dl", labels=LABELS, config_prefix="tabm")
requests

# %% tags=["results"]
plan = plan_model_catalog(
    study,
    requests,
    execution_tier=EXECUTION_TIER,
    overrides=OVERRIDES,
    preview_reductions=PREVIEW_REDUCTIONS,
)
# Task semantics and imbalance treatment are resolved inputs, so read them from the frozen
# specification rather than restating the configuration file here.
resolved_tasks = [spec["computation"]["task"] for spec in plan_specs(plan)]
resolved_contracts = declared_contracts(plan).with_columns(
    pl.Series("metrics", [task.get("metrics", []) for task in resolved_tasks]),
    pl.Series("imbalance", [task.get("imbalance") for task in resolved_tasks]),
)
resolved_contracts.select(
    "label",
    "config_name",
    "task",
    "continuous_eval_label",
    "imbalance",
    "metrics",
    "checkpoint_value",
    "eligible_rows",
    "training_hash",
)

# %% [markdown]
# The complete case-wide population is recorded before the first fit, so a member that later
# fails to train cannot quietly disappear from the population it was declared in. This notebook
# produces one slice of it, and that slice must lie inside the declaration.

# %% tags=["results"]
if official_population is not None:
    outside = set(plan.expected_prediction_hashes) - set(official_population.members)
    if outside:
        raise RuntimeError(
            f"{len(outside)} declared checkpoints lie outside the official model population"
        )

# %% [markdown]
# ## 2. Execute and validate the fitted-state population
#
# Each configuration is fitted on each fold; the weights are persisted at every checkpoint epoch
# with a digest, and one complete validation prediction set is registered per checkpoint. A cached
# fitted state is reused only when its digest matches, so a resumed run cannot silently continue
# from weights that a code change has invalidated.
#
# The completeness check is the substantive one. A prediction set is complete when it covers every
# validation key its fold declares. A set covering most of them is not a slightly worse result - it
# is a different sample, and putting it beside a complete one in the backtest would compare two
# models measured on different data. The run raises rather than publishing an incomplete
# population, which is the behaviour to want: a loud failure here costs a re-run, and a quiet one
# costs a wrong comparison that nothing downstream can detect.

# %% tags=["results"]
execution = run_model_plan(
    plan,
    supersedes=SUPERSEDES_MODEL_POPULATION or None,
    population_name="crypto-tabm-validation-predictions-v1"
    if EXECUTION_TIER == "canonical"
    else None,
)
catalog = execution.catalog_rows.sort("label", "config_name", "checkpoint_value")
if (
    catalog.height != len(plan.expected_prediction_hashes)
    or catalog.filter(~pl.col("complete")).height
):
    raise RuntimeError("TabM fitted-state or prediction population is incomplete")
catalog.select(
    "label",
    "config_name",
    "task",
    "checkpoint_kind",
    "checkpoint_value",
    "training_hash",
    "prediction_hash",
    "complete",
)

# %% [markdown]
# ## Key takeaways and limitations
#
# - **The ensemble is nearly free, and that is the design.** Four members share one two-layer
#   backbone and own only a per-unit scaling vector and a final linear layer each. The averaging
#   that steadies a neural fit on a table costs four small tensors here rather than four networks,
#   which is why the member count is the cheap dial and the hidden width is not.
# - **Task semantics and imbalance treatment are resolved inputs, not notebook conventions.** What
#   objective is minimized, and how a fold's class imbalance is corrected, are read back out of the
#   frozen specification. If they were decided in notebook code, two runs of the same declared
#   configuration could differ without their identities differing.
# - **Class weights belong to a fold, not to the panel.** Fitting them once over the whole history
#   would carry every fold a correction estimated partly on windows it must not see.
# - **Configurations times checkpoints is the count that matters.** Eight scoreable models per
#   configuration, each registered separately. Reporting the best of them as a single model's score
#   would be reporting a maximum over eight draws, and the selection that handles this correctly
#   happens in [`13_backtest`](13_backtest.ipynb), not here.
# - **The identity binds the device policy, not only the model and seed.** GPU kernels reorder
#   floating-point reductions, so the same weights and the same data can produce slightly different
#   numbers on a different device. Binding the device policy into the identity means a result is
#   never compared against one produced under a different arithmetic.
# - **Two folds is the binding constraint, not the architecture.** As with every model family in
#   this case study, the usable perpetual funding history is short, and no amount of capacity
#   compensates for that.
