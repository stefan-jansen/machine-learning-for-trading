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
# # Temporal convolutional models on the 8-hour grid
#
# The shared sequence request fixes the TCN architecture, feature order, chronological folds,
# missing-observation policy, and full checkpoint schedule before training. Every complete
# checkpoint remains a distinct prediction identity for later validation backtests.
#
# **Learning objectives**
#
# - construct a temporal-convolution request on an explicit observation cadence;
# - inspect receptive-field, gap-policy, and checkpoint identity; and
# - verify exact validation coverage and fitted-state persistence.
#
# **Book reference:** Chapter 19, convolutional sequence models.
#
# **Prerequisites:** finalized crypto labels, features, and purged walk-forward folds; CUDA for the
# canonical run.

# %%
import os

import polars as pl

from case_studies.crypto_perps_funding.research_workflow import (
    REGRESSION_LABELS,
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
LABELS = REGRESSION_LABELS
PREVIEW_REDUCTIONS = {}
OVERRIDES = {"device": "cuda"}

# %% [markdown]
# ## Resolve sequence and checkpoint identities

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
official_population = (
    freeze_official_model_population(study) if EXECUTION_TIER == "canonical" else None
)
requests = model_request_catalog("deep_learning", labels=LABELS, config_prefix="tcn")
requests

# %% tags=["results"]
plan = plan_model_catalog(
    study,
    requests,
    execution_tier=EXECUTION_TIER,
    overrides=OVERRIDES,
    preview_reductions=PREVIEW_REDUCTIONS,
)
# Sequence eligibility follows from the resolved gap policy and lookback, so read both from the
# frozen specification instead of restating the configuration file here.
# `preprocessing` sits at the top level for the current sequence spec and under `computation`
# for every family already migrated to the shared resolved specification; read either shape.
resolved_preprocessing = [
    spec.get("computation", spec)["preprocessing"] for spec in plan_specs(plan)
]
contracts = declared_contracts(plan).with_columns(
    pl.Series("gap_policy", [step["gap_policy"] for step in resolved_preprocessing]),
    pl.Series("lookback", [step["lookback"] for step in resolved_preprocessing]),
)
contracts.select(
    "label",
    "config_name",
    "gap_policy",
    "lookback",
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
# ## Execute the declared population

# %% tags=["results"]
execution = run_model_plan(
    plan,
    population_name="crypto-tcn-validation-predictions-v1"
    if EXECUTION_TIER == "canonical"
    else None,
)
catalog = execution.catalog_rows.sort("label", "config_name", "checkpoint_value")
if (
    catalog.height != len(plan.expected_prediction_hashes)
    or catalog.filter(~pl.col("complete")).height
):
    raise RuntimeError("TCN checkpoint population is incomplete")
catalog.select(
    "label",
    "config_name",
    "checkpoint_value",
    "training_hash",
    "prediction_hash",
    "complete",
)

# %% [markdown]
# ## Key takeaways and limitations
#
# - Dilated convolutions use a fixed chronological input window whose eligible keys are known before
#   fitting.
# - Gap handling and checkpoint membership remain visible in the resolved request.
# - The configured receptive field limits the temporal dependencies the model can represent.
