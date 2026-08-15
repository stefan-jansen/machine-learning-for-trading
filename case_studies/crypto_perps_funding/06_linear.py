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
# # Linear models for funding-aligned returns and direction
#
# This execution notebook submits the complete declared linear-model population to the shared
# research boundary. The request resolves the label artifact, task, continuous evaluation target,
# fold geometry, estimator parameters, and exact eligible prediction keys before fitting. Model
# interpretation is developed in `12_model_analysis`.
#
# **Learning objectives**
#
# - construct linear regression and classification requests from the published menu;
# - inspect target, fold, and prediction-coverage identity before fitting; and
# - verify that every declared result enters the prediction catalog with complete lineage.
#
# **Book reference:** Chapter 11, linear models for trading signals.
#
# **Prerequisites:** finalized crypto labels, features, and purged walk-forward folds.

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
    run_model_plan,
)

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE = os.environ.get("ML4T_OUTPUT_DIR", "")
LABELS = ALL_LABELS
PREVIEW_REDUCTIONS = {}
OVERRIDES = {}

# %% [markdown]
# ## Declared requests
#
# Every label and configured estimator is visible before execution. Preview reductions are part of
# the resolved identity and cannot enter the canonical catalog.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
official_population = (
    freeze_official_model_population(study) if EXECUTION_TIER == "canonical" else None
)
requests = model_request_catalog("linear", labels=LABELS)
requests

# %% tags=["results"]
plan = plan_model_catalog(
    study,
    requests,
    execution_tier=EXECUTION_TIER,
    overrides=OVERRIDES,
    preview_reductions=PREVIEW_REDUCTIONS,
)
resolved_contracts = declared_contracts(plan).select(
    "label",
    "config_name",
    "task",
    "continuous_eval_label",
    "eligible_rows",
    "training_hash",
)
resolved_contracts

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
# ## Execute and expose the catalog rows
#
# The shared runner validates exact keys, fold completeness, finite predictions, and fitted-state
# lineage before a row is marked complete.

# %% tags=["results"]
execution = run_model_plan(
    plan,
    population_name="crypto-linear-validation-predictions-v1"
    if EXECUTION_TIER == "canonical"
    else None,
)
catalog = execution.catalog_rows.sort("label", "config_name", "checkpoint_value")
if (
    catalog.height != len(plan.expected_prediction_hashes)
    or catalog.filter(~pl.col("complete")).height
):
    raise RuntimeError("linear execution returned an incomplete catalog row")
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
# - The request is the reproducible unit: label, task, configuration, folds, and eligible keys are
#   resolved before fitting.
# - Classification diagnostics retain the continuous return target used for trading evaluation.
# - Linear models provide an interpretable reference class but do not represent nonlinear feature
#   interactions or time-dependent hidden state.
