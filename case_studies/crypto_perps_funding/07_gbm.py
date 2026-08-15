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
# # Gradient-boosted models for perpetual-funding signals
#
# This execution notebook runs every declared gradient-boosted configuration and every declared
# checkpoint through the shared request boundary. Checkpoint identity is retained in the prediction
# catalog. Comparative interpretation is deferred to `12_model_analysis`.
#
# **Learning objectives**
#
# - construct a complete gradient-boosting request grid from the published menu;
# - inspect fold-scaled loss settings and checkpoint membership before fitting; and
# - verify that fitted checkpoints remain distinct catalog identities.
#
# **Book reference:** Chapter 12, gradient boosting for trading.
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
# ## Resolve the complete grid

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
official_population = (
    freeze_official_model_population(study) if EXECUTION_TIER == "canonical" else None
)
requests = model_request_catalog("gbm", labels=LABELS)
requests

# %% tags=["results"]
plan = plan_model_catalog(
    study,
    requests,
    execution_tier=EXECUTION_TIER,
    overrides=OVERRIDES,
    preview_reductions=PREVIEW_REDUCTIONS,
)
# One planned row per boosting checkpoint; group them back to the configuration that declares them.
checkpoint_contracts = (
    declared_contracts(plan)
    .group_by("label", "config_name", "training_hash", "eligible_rows", maintain_order=True)
    .agg(pl.col("checkpoint_value").alias("checkpoints"))
)
checkpoint_contracts

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
# ## Execute and validate catalog membership

# %% tags=["results"]
execution = run_model_plan(
    plan,
    population_name="crypto-gbm-validation-predictions-v1"
    if EXECUTION_TIER == "canonical"
    else None,
)
catalog = execution.catalog_rows.sort("label", "config_name", "checkpoint_value")
if (
    catalog.height != len(plan.expected_prediction_hashes)
    or catalog.filter(~pl.col("complete")).height
):
    raise RuntimeError("GBM checkpoint population is incomplete")
catalog.select(
    "label",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "training_hash",
    "prediction_hash",
    "complete",
)

# %% [markdown]
# ## Key takeaways and limitations
#
# - A checkpoint is part of a model configuration and remains available for validation backtests.
# - Robust-loss thresholds are resolved at the scale of each training fold.
# - The notebook establishes execution and lineage, not which model should be traded.
