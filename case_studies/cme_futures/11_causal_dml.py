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
# # CME Futures: Double Machine Learning
#
# This notebook estimates the configured treatment effect for each return horizon. The request
# resolves the outcome, treatment, confounders, timing, embargo, nuisance models, and placebo design
# before fitting. Missing confounders are an error rather than an invitation to change the estimand.
#
# Nuisance models train on complete timestamp panels before the holdout. HAC uncertainty uses the
# declared temporal ordering, and contiguous within-product blocks define the placebo refutation.
# `12_model_analysis` reports the causal diagnostics. Trading configuration selection uses
# validation backtest Sharpe.
#
# Prerequisites: `03_financial_features`, `04_model_based_features`, and `05_evaluation`.

# %%
"""Fit the declared CME futures double-machine-learning requests."""

import polars as pl

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    open_study,
    product_universe_table,
)

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None
PREVIEW_REDUCTIONS: dict = {}

# %% [markdown]
# ## Resolve the estimands
#
# Preview row and fold limits must be named in `PREVIEW_REDUCTIONS`. They enter the computation
# identity and cannot be mistaken for canonical causal estimates.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
if EXECUTION_TIER == "preview" and (WORKSPACE is None or not PREVIEW_REDUCTIONS):
    raise ValueError("preview execution requires WORKSPACE and PREVIEW_REDUCTIONS")
universe = product_universe_table()
universe

# %%
requests = tuple(
    study.causal(
        method="dml",
        label=label,
        execution_tier=EXECUTION_TIER,
        preview_reductions=PREVIEW_REDUCTIONS,
    ).resolve()
    for label in ALL_LABELS
)
request_rows = []
for request in requests:
    computation = request.spec["computation"]
    estimand = computation["estimand"]
    population = computation["analysis_population"]
    request_rows.append(
        {
            "label": request.spec["label"],
            "treatment": estimand["treatment"],
            "outcome": estimand["outcome"],
            "outcome_horizon": estimand["outcome_horizon"],
            "confounders": ", ".join(estimand["confounders"]),
            "analysis_rows": population["n_rows"],
            "analysis_timestamps": population["n_timestamps"],
            "request_hash": request.identity,
        }
    )
request_table = pl.DataFrame(request_rows)
request_table

# %% [markdown]
# ## Execute and verify restart
#
# The shared causal runner uses deterministic seeds and persists one immutable result per resolved
# request. Reopening the same request must return the same identity and a complete result.

# %%
results = []
for request in requests:
    label = request.spec["label"]
    result = request.run()
    restarted = request.run()
    if not result.complete or restarted.hash != result.hash:
        raise RuntimeError(f"causal request for {label} did not persist completely")
    results.append(
        {
            "label": label,
            "execution_tier": result.execution_tier,
            "complete": result.complete,
            "causal_hash": result.hash,
        }
    )

# %% tags=["results"]
pl.DataFrame(results).sort("label")
