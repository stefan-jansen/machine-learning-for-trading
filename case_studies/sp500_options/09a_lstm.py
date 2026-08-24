# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # S&P 500 Options: LSTM
#
# This notebook fits the declared LSTM member of the sequence population snapshotted by
# `09_deep_learning`. Chronological windows, validation gaps, checkpoints, and prediction
# eligibility are resolved through the shared sequence boundary.
#
# Prerequisite: `09_deep_learning` must create the complete official sequence population.

# %%
"""Fit the declared S&P 500 options LSTM request."""

import polars as pl

from case_studies.sp500_options.research_workflow import (
    ALL_LABELS,
    model_request_catalog,
    open_study,
    resolve_model_requests,
    resolved_model_plan,
    run_official_model_subset,
    run_resolved_model_requests,
)

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
PREVIEW_REDUCTIONS: dict = {}

POPULATION_NAME = "sp500-options-sequence-validation-v1"

# %% [markdown]
# ## Declared request

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
requests = model_request_catalog(
    "deep_learning",
    labels=ALL_LABELS,
    config_names=("lstm_h64",),
)
resolved = resolve_model_requests(
    study,
    requests,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
)
resolved_model_plan(resolved)

# %% [markdown]
# ## Execute and validate
#
# The shared sequence runner owns chronological window construction, fold fitting, fitted-state
# reload, checkpoint publication, restart, and exact eligible-key validation.

# %%
if EXECUTION_TIER == "canonical":
    execution, population = run_official_model_subset(
        study,
        resolved,
        population=POPULATION_NAME,
    )
else:
    if not WORKSPACE or not PREVIEW_REDUCTIONS:
        raise ValueError("preview execution requires WORKSPACE and PREVIEW_REDUCTIONS")
    execution = run_resolved_model_requests(study, resolved)
    population = None

# %% tags=["results"]
catalog = execution.catalog_rows.select(
    "family",
    "label",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "execution_tier",
    "complete",
    "training_hash",
    "prediction_hash",
).sort("checkpoint_value")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("LSTM execution returned a partial checkpoint")
catalog

# %% [markdown]
# The complete LSTM checkpoint population is ready for model analysis and backtesting. This
# notebook does not compare it with another family or choose a checkpoint.
