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
# # TabM models for an unbalanced perpetual-futures panel
#
# The shared TabM request resolves regression or classification explicitly. Classification requests
# also resolve their continuous return target, task metrics, fold-specific class weights, and every
# epoch checkpoint before fitting. The complete checkpoint catalog is the notebook output.
#
# **Learning objectives**
#
# - distinguish regression, binary classification, and multiclass requests;
# - inspect the continuous return target and fold-specific imbalance treatment; and
# - verify fitted-state and checkpoint lineage after GPU execution.
#
# **Book reference:** Chapter 18, deep learning for tabular data.
#
# **Prerequisites:** finalized crypto labels, features, and purged walk-forward folds; CUDA for the
# canonical run.

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
WORKSPACE = os.environ.get("ML4T_OUTPUT_DIR", "")
LABELS = ALL_LABELS
PREVIEW_REDUCTIONS = {}
OVERRIDES = {"class_weight": "balanced", "device": "cuda"}

# %% [markdown]
# ## Resolve targets, imbalance policy, and checkpoints

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
official_population = (
    freeze_official_model_population(study) if EXECUTION_TIER == "canonical" else None
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
resolved_tasks = [spec.get("computation", spec)["task"] for spec in plan_specs(plan)]
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
