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
# # CME Futures: Principal-Component Factors
#
# PCA compresses the point-in-time feature panel into fold-scoped components. The transformer fits
# on training rows only, and the saved fitted state is reused to transform that fold's validation
# rows. Both return horizons are declared explicitly.
#
# This notebook publishes predictions and fitted-state lineage. `13_backtest` applies the common
# validation-Sharpe selection rule.
#
# Prerequisites: `03_financial_features`, `04_model_based_features`, and `05_evaluation`.

# %%
"""Fit the declared CME futures PCA factor population."""

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
# Both return horizons use the named PCA configuration. The resolved plan shows the eligible rows,
# folds, feature count, checkpoint schedule, and identity before fitting begins.
#
# **PCA publishes on the CPU, and says so rather than inheriting it.** `setup.yaml` declares
# `cuda` for the latent-factor family because the stochastic discount factor in
# [`10b`](10b_stochastic_discount_factor.ipynb) is a neural model. This one is not: `PCAModel` is
# numpy and scipy linear algebra and its `fit` takes no device argument, so a run recording `cuda`
# would name hardware the computation never touched. The device is part of the hashed computation -
# it enters both `runtime` and `numerical_runtime` - so this is an identity the notebook is
# choosing, not a comment about it.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
requests = model_request_catalog(
    "latent_factors",
    labels=ALL_LABELS,
    config_names=("pca",),
)
resolved = resolve_model_requests(
    study,
    requests,
    execution_tier=EXECUTION_TIER,
    overrides={"device": "cpu"},
    preview_reductions=PREVIEW_REDUCTIONS,
)
universe = product_universe_table()
universe

# %%
resolved_model_plan(resolved)

# %% [markdown]
# ## Execute and validate
#
# The shared latent-factor runner fits PCA inside each training fold, persists the transformer, and
# requires the complete validation key set before publication.

# %%
if EXECUTION_TIER == "canonical":
    execution, population = run_official_model_catalog(
        study,
        requests,
        population_name="cme_futures-pca-validation-v1",
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
).sort("label", "checkpoint_value")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("PCA execution returned a partial prediction")
catalog
