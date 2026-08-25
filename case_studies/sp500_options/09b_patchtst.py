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
# # S&P 500 Options: PatchTST
#
# This notebook fits the declared PatchTST member of the sequence population snapshotted by
# `09_deep_learning`. After publishing every PatchTST checkpoint, it verifies that the complete
# NLinear, LSTM, and PatchTST population is present.
#
# Prerequisites: `09_deep_learning` and `09a_lstm`.

# %%
"""Fit the declared S&P 500 options PatchTST request."""

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
    config_names=("patchtst",),
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
# The shared sequence runner owns gap-safe window construction, fold fitting, fitted-state reload,
# checkpoint publication, restart, and exact eligible-key validation.

# %%
if EXECUTION_TIER == "canonical":
    execution, population = run_official_model_subset(
        study,
        resolved,
        population=POPULATION_NAME,
        require_population_complete=True,
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
    raise RuntimeError("PatchTST execution returned a partial checkpoint")
catalog

# %% [markdown]
# The official sequence population is complete and ready for model analysis and backtesting. This
# notebook does not compare configurations or choose a checkpoint.
