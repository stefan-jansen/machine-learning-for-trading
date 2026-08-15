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
# # CME Futures: Linear Models
#
# This notebook fits the declared linear-model population for both return horizons. Each model
# configuration uses the same walk-forward folds and eligibility rules established by
# `05_evaluation`. Ridge, Lasso, and ElasticNet differ only in their regularization parameters.
#
# The notebook executes models and publishes complete validation predictions. `13_backtest`
# evaluates every configuration and checkpoint with the equal-weight signal baseline. Validation
# backtest Sharpe performs selection.
#
# Prerequisites: `03_financial_features`, `04_model_based_features`, and `05_evaluation`.

# %%
"""Fit the declared CME futures linear-model population."""

import polars as pl

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    model_request_catalog,
    open_study,
    product_universe_table,
    resolve_model_requests,
    resolved_model_plan,
    run_official_model_catalog,
    run_resolved_model_requests,
)

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None
PREVIEW_REDUCTIONS: dict = {}
# The population hash this run replaces, read from the registry and set by a person. A
# first population takes None; a re-run whose membership has changed is refused without
# the hash it supersedes, and the refusal names the value required.
SUPERSEDES_POPULATION: str | None = None

# %% [markdown]
# ## Declared requests
#
# The request table is the model population. A canonical run snapshots every expected prediction
# identity before fitting, including every declared checkpoint. A preview must name its reductions
# and writes to a separate workspace.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
requests = model_request_catalog("linear", labels=ALL_LABELS)
resolved = resolve_model_requests(
    study,
    requests,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
)
universe = product_universe_table()
universe

# %%
resolved_model_plan(resolved)

# %% [markdown]
# ## Execute and validate
#
# The shared linear runner owns preprocessing, fold fitting, fitted-state digests, restart, metric
# computation, and exact eligible-key checks. Any missing or partial member fails the cell.

# %%
if EXECUTION_TIER == "canonical":
    execution, population = run_official_model_catalog(
        study,
        requests,
        population_name="cme-linear-validation-v1",
        resolved_requests=resolved,
        supersedes=SUPERSEDES_POPULATION,
    )
else:
    if WORKSPACE is None or not PREVIEW_REDUCTIONS:
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
).sort("label", "config_name", "checkpoint_value")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("linear execution returned a partial prediction")
catalog
