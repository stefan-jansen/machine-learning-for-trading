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
# # CME Futures: Tabular Deep Learning
#
# TabM applies a parameter-efficient neural ensemble to the same point-in-time feature rows used by
# the linear and gradient-boosting families. The declared configurations vary model capacity while
# retaining the walk-forward fold and label contracts from `05_evaluation`.
#
# The shared runner publishes every declared epoch checkpoint with its fitted weights and exact
# validation coverage. The equal-weight validation backtest in `13_backtest` evaluates all
# checkpoints and selects by Sharpe.
#
# Prerequisites: `03_financial_features`, `04_model_based_features`, and `05_evaluation`.

# %%
"""Fit the declared CME futures TabM population."""

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
# Both configured return horizons enter the same visible request table. Preview epoch or fold limits
# must be passed through `PREVIEW_REDUCTIONS`, which changes identity and excludes the output from
# the canonical catalog.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
requests = model_request_catalog("tabular_dl", labels=ALL_LABELS)
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
# Fold-scoped preprocessing, seeded training, fitted-state persistence, checkpoint membership, and
# prediction eligibility are enforced by the shared TabM adapter.

# %%
if EXECUTION_TIER == "canonical":
    execution, population = run_official_model_catalog(
        study,
        requests,
        population_name="cme-tabular-dl-validation-v1",
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
    raise RuntimeError("TabM execution returned a partial prediction")
catalog
