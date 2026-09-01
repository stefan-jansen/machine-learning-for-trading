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
# # CME Futures: Stochastic Discount-Factor Features
#
# The stochastic discount-factor model learns fold-scoped latent factors from the product panel and
# maps them to each declared forward-return horizon. Training rows determine the representation;
# validation rows are transformed without refitting. The fitted model, fold identity, prediction
# shard, and eligible validation keys are persisted together.
#
# The notebook executes the declared SDF configurations and publishes their catalog rows. IC remains
# diagnostic. The equal-weight validation backtest in `13_backtest` selects configurations.
#
# Prerequisites: `03_financial_features`, `04_model_based_features`, and `05_evaluation`.

# %%
"""Fit the declared CME futures stochastic discount-factor population."""

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
# Both return horizons use the named stochastic discount-factor configuration. The resolved plan
# shows the eligible rows, folds, feature count, checkpoint schedule, and identity before fitting.
#
# **This model publishes on `cuda`, declared in `setup.yaml` rather than detected.** The device
# enters both `runtime` and `numerical_runtime` inside the hashed computation, so leaving it to
# `preferred_latent_device()` would resolve one training identity on a GPU host and a different one
# on a CPU host, both publishing under this population's name. `10a_pca` overrides the same
# declaration to `cpu`, because PCA has no GPU implementation to record.
#
# **Nothing here fails on a fit that did not converge.** The shared runner enforces convergence for
# IPCA only, and this case study does not declare IPCA; for the stochastic discount factor it
# records the training history and the terminal Sharpe as fold extras and checks neither. That is a
# gap in shared code rather than in this notebook, so what is claimed below is that every declared
# checkpoint was produced and registered - not that the objective had settled when it was.

# %%
study = open_study(
    execution_tier=EXECUTION_TIER, workspace=WORKSPACE, entry_point="10b_stochastic_discount_factor"
)
requests = model_request_catalog(
    "latent_factors",
    labels=ALL_LABELS,
    config_names=("sdf",),
)
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
# The shared latent-factor runner fits each representation inside its training fold, persists the
# fitted state, and requires the complete validation key set before publication.

# %%
if EXECUTION_TIER == "canonical":
    execution, population = run_official_model_catalog(
        study,
        requests,
        population_name="cme_futures-sdf-validation-v1",
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
    raise RuntimeError("stochastic discount-factor execution returned a partial prediction")
catalog
