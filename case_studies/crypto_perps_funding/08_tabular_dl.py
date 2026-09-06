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
# **Prerequisites:** finalized crypto labels, features, and purged walk-forward folds; CUDA for
# the canonical run.

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
# ## Resolve targets, imbalance policy, and checkpoints

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
# ## Execute and validate the fitted-state population

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
# - Task semantics and imbalance treatment are resolved inputs, not notebook-side conventions.
# - Every reported checkpoint has a persisted fitted state and exact prediction coverage.
# - GPU kernels can introduce small numerical differences; catalog identity still binds the model,
#   seed, device policy, and checkpoint schedule used by the run.
