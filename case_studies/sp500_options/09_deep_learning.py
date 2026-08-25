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
# # S&P 500 Options: NLinear
#
# This notebook snapshots the complete three-model sequence population before fitting its NLinear
# member. `09a_lstm` and `09b_patchtst` execute the other declared members against the same
# immutable population. Every configured checkpoint remains eligible for model analysis and
# backtesting.
#
# Prerequisites: `03_financial_features`, `04_model_based_features`, and `05_evaluation`.

# %%
"""Fit NLinear within the declared S&P 500 options sequence population."""

import polars as pl

from case_studies.sp500_options.research_workflow import (
    ALL_LABELS,
    model_request_catalog,
    open_study,
    resolve_model_requests,
    resolved_model_plan,
    run_official_model_subset,
    run_resolved_model_requests,
    snapshot_official_model_catalog,
)

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
PREVIEW_REDUCTIONS: dict = {}
DEVICE: str = ""

SEQUENCE_CONFIGS = ("nlinear", "lstm_h64", "patchtst")
POPULATION_NAME: str = ""

# %% [markdown]
# ### The device the population was fitted on
#
# A network trained on a GPU and the same network trained on a CPU accumulate their sums in a
# different order and reach different weights, so the device is part of what the fitted model is
# and sits inside the training identity rather than beside it. `PUBLISHED_DEVICE` is the device
# this population was fitted on, and pinning it is what makes the notebook fit the same thing
# wherever it runs. On a machine with no NVIDIA card the run stops here rather than quietly
# training something else: set `DEVICE="cpu"` and pass a `POPULATION_NAME` to fit the same
# requests there, under a name of their own. `08_tabular_dl` pins its device the same way.

# %%
PUBLISHED_DEVICE = "cuda"
CANONICAL_POPULATION_NAME = "sp500-options-sequence-validation-v1"

device = DEVICE or PUBLISHED_DEVICE
population_name = POPULATION_NAME or CANONICAL_POPULATION_NAME
if device != PUBLISHED_DEVICE and population_name == CANONICAL_POPULATION_NAME:
    raise ValueError(
        f"this run fits on {device!r}, not the published {PUBLISHED_DEVICE!r}, so its "
        f"identities are not the ones {CANONICAL_POPULATION_NAME!r} holds; pass "
        f"POPULATION_NAME to give them a population of their own"
    )
print(f"training device: {device}")

# %% [markdown]
# ## Complete sequence request population
#
# The case-wide table is resolved before the first member executes. Canonical execution snapshots
# all configuration-checkpoint identities so a failed member cannot disappear from later analysis.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
all_requests = model_request_catalog(
    "deep_learning",
    labels=ALL_LABELS,
    config_names=SEQUENCE_CONFIGS,
)
all_resolved = resolve_model_requests(
    study,
    all_requests,
    execution_tier=EXECUTION_TIER,
    overrides={"device": device},
    preview_reductions=PREVIEW_REDUCTIONS,
)
resolved_model_plan(all_resolved)

# %% [markdown]
# ## Execute NLinear
#
# NLinear shares the gap-safe sequence construction, fold boundaries, fitted-state persistence,
# restart, and exact eligible-key checks used by the other sequence configurations.

# %%
nlinear_resolved = tuple(
    request for request in all_resolved if request.spec["config_name"] == "nlinear"
)
if len(nlinear_resolved) != 1:
    raise ValueError("the sequence population must contain exactly one NLinear request")

if EXECUTION_TIER == "canonical":
    population = snapshot_official_model_catalog(
        study,
        all_requests,
        population_name=population_name,
        resolved_requests=all_resolved,
    )
    execution, population = run_official_model_subset(
        study,
        nlinear_resolved,
        population=population,
    )
else:
    if not WORKSPACE or not PREVIEW_REDUCTIONS:
        raise ValueError("preview execution requires WORKSPACE and PREVIEW_REDUCTIONS")
    execution = run_resolved_model_requests(study, nlinear_resolved)
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
    raise RuntimeError("NLinear execution returned a partial checkpoint")
catalog

# %% [markdown]
# The NLinear checkpoint artifacts are complete. The official sequence population remains open
# until `09a_lstm` and `09b_patchtst` publish their declared members.
