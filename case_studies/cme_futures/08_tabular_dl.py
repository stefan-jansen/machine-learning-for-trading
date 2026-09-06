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

# %% [markdown]
# ## Why a neural network on a feature table at all
#
# Deep learning earned its reputation on images, audio and text, where the input has structure a
# network can exploit: neighbouring pixels are related, words have order, and the architecture is
# built to reflect that. A table of engineered features has none of it. Columns can be permuted
# with no loss of meaning, and there is no locality for a convolution to use or a sequence for a
# recurrence to traverse.
#
# On that kind of input, gradient-boosted trees remain the standard to beat, and they beat neural
# networks often enough that "tabular deep learning" is a live research area rather than a
# settled one. Trees handle mixed scales without preprocessing, ignore irrelevant columns almost
# for free, and split on thresholds - which is exactly the shape of many real relationships in a
# feature table, where an effect appears above some level of a variable and not below it.
#
# So this stage runs with a specific question rather than an assumption: on this panel, with
# these features, does a network find anything the trees in `07_gbm` do not? It reads the same
# point-in-time feature rows under the same fold and label contracts, so the comparison isolates
# the model family.
#
# ### What TabM is doing differently
#
# The obvious way to improve a neural network's reliability is to train several and average
# them, which reduces the variance that comes from initialization and from the optimizer's path.
# The obvious cost is that k models take k times the compute and k times the memory.
#
# TabM is a parameter-efficient ensemble: it trains what behaves like several models while
# sharing most of the weights between them, so the averaging is available at close to the cost of
# one. That matters here more than it would on a large dataset, because the thing most likely to
# go wrong on a panel this size is not bias but variance - a single network on a small, noisy
# feature table can land in a very different place depending on where it started, and the spread
# between those places can exceed whatever edge is being measured.
#
# `varies model capacity` in the declared configurations is the other half of the same concern.
# Capacity is the dial that trades fitting the training rows against generalizing off them, and
# on a noisy panel the best setting is usually much smaller than intuition suggests. Declaring
# several and letting the backtest choose is what keeps that from being a guess.
#
# One consequence worth stating for the comparison with `07_gbm`: a network needs its features
# standardized and trees do not. So the two families do not read quite the same inputs even
# though they read the same columns, and a difference in their results is partly a difference in
# preprocessing rather than purely in model family. That is unavoidable - an unstandardized
# network on mixed-scale features does not train - but it is worth knowing before the gap
# between the two is attributed entirely to what the models can represent.
#
# ### Why every checkpoint is published
#
# A neural fit is a trajectory rather than a model: it passes through a sequence of states, and
# which one is kept is a choice with the same standing as the architecture. The runner publishes
# every declared epoch checkpoint with its own fitted weights and validation coverage, and
# `13_backtest` selects among them on Sharpe like any other configuration.
#
# Publishing them rather than picking one is the honest arrangement. Choosing the best epoch by
# looking at validation performance and then reporting that model's validation performance is
# selection inside the number being reported, and it does not stop being that because the choice
# was made by hand rather than by a search. Declaring the checkpoints puts the choice into the
# same funnel and the same trial count as everything else - which the deflated Sharpe downstream
# then has to divide by, and which is why the count is not free.

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
#
# Routing reductions through identity rather than through a flag is what keeps a reduced run
# from being mistaken for a real one later. A preview that trained for two epochs instead of the
# declared schedule produces a genuine prediction set with genuine metrics, and nothing about
# the numbers announces that they came from a fraction of the work. Because the reduction enters
# the hash, the reduced rows cannot resolve to the same identity as canonical ones, cannot be
# served back in place of them, and are excluded from the catalog the backtest reads.
#
# The alternative - a boolean that says "this was a preview" - fails the moment anyone queries
# the registry without checking it, which is the failure mode that makes a leaderboard quietly
# wrong rather than visibly broken.
#
# **TabM runs on the GPU, and the request says so rather than inheriting it.** With no override the
# shared adapter falls back to a literal `"cuda"` written in `case_studies/utils/tabular_dl.py`, and
# `resolve_torch_device` raises `CUDA was requested but is unavailable` rather than quietly moving
# the fit to the CPU. A CUDA device is therefore a hard requirement of this population, declared two
# layers below the notebook: without one these configurations cannot be reproduced at all. Naming it
# in the request puts that requirement where a reader meets it. The resolved specification hash is
# the same with the override as without, so this states what the published run already did.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
requests = model_request_catalog("tabular_dl", labels=ALL_LABELS)
resolved = resolve_model_requests(
    study,
    requests,
    execution_tier=EXECUTION_TIER,
    overrides={"device": "cuda"},
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
#
# **Fold-scoped preprocessing is the item on that list most easily got wrong.** A network needs
# its inputs standardized, and standardizing means subtracting a mean and dividing by a scale -
# both of which are estimated quantities. Estimating them over the whole panel and then applying
# them inside each fold leaks: the training rows are centred using a mean that already reflects
# the validation period, and the resulting predictions are built from a summary of data the model
# was not supposed to have. It is a small leak and an invisible one - no assertion over the
# prediction frame can see it, because the leaked quantity is two numbers that never appear in
# the output. The adapter refits the scaler inside each training fold for that reason.
#
# **Seeded training is what makes a result a result rather than a draw.** Two runs of the same
# configuration with different seeds land in different places, and on a panel this size the gap
# between them can be comparable to the differences the backtest is trying to measure. Fixing the
# seed does not make the model better; it makes the number attributable to the configuration
# rather than to the draw, which is the precondition for comparing configurations at all.
#
# That is also why the seed lives in the resolved specification rather than in a notebook
# constant. A seed that only reached the training call would be a dial that turned without
# moving the identity - change it, and the registry serves back the result fitted under the old
# one while the notebook claims the new.

# %%
if EXECUTION_TIER == "canonical":
    execution, population = run_official_model_catalog(
        study,
        requests,
        population_name="cme_futures-tabular_dl-validation-v1",
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
