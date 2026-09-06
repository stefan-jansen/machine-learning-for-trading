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
# PCA is fitted on the training **return** panel and produces fold-scoped components. It fits on
# training rows only, and the saved fitted state is reused to transform that fold's validation
# rows. Both return horizons are declared explicitly.
#
# This notebook publishes predictions and fitted-state lineage. `13_backtest` applies the common
# validation-Sharpe selection rule.
#
# Prerequisites: `03_financial_features`, `04_model_based_features`, and `05_evaluation`.

# %% [markdown]
# ## What is factored, and what is not
#
# The characteristics panel is passed to this stage and then deliberately discarded:
# `case_studies/utils/latent_factors/pca.py::run_pca_fold` takes `chars_train` and `chars_val`
# and drops them with `del` before fitting. What PCA sees is the matrix of product returns, one
# column per contract, and nothing else. Its module docstring says so - "Return-panel PCA
# baseline" - and until this notebook was reviewed its own header said the opposite.
#
# That distinction is the whole difference between this stage and every stage before it, and it
# is easy to read past. This is **not** a dimensionality reduction of the engineered features -
# it is not compressing carry, momentum and volatility into fewer columns. It is a factor model
# on returns, which asks a different question: given only how the thirty products moved
# together, what small set of directions accounts for most of that movement?
#
# A reader who takes it as "PCA over the feature panel" will expect components that mean
# something in terms of carry or momentum, and will be looking for an interpretation the method
# never produced.
#
# ## What the components are, and what they are not
#
# The first component on a futures return panel is typically close to "everything moves
# together" - a level factor. The next few usually separate the sectors, because energy
# contracts co-move with each other more than with the metals. None of that is imposed: PCA is
# given no sector labels and no economic structure, and finds the directions of greatest
# variance whatever they turn out to be.
#
# This is also why "how many factors" is the only real dial. Everything else is determined once
# the panel and the count are fixed - there is no loss function to choose, no optimizer, no
# stopping rule, and no seed that changes the answer. That is unusual in this case study, and it
# is the source of both the method's robustness and its ceiling.
#
# The components are **directions in return space with no names**, ordered by how much variance
# they explain. That ordering is not a ranking of usefulness for prediction; it is a ranking of
# how much the panel moved along each. A component explaining a large share of variance can be
# entirely unrewarded, and it will still come first.
#
# The forecast published here comes from projecting onto those components and carrying the
# factor returns forward with an expanding mean. So a product's prediction is built from how the
# panel as a whole has behaved along a few directions, rather than from anything about that
# product's own carry - which is what makes it a different kind of candidate for `13_backtest`
# to compare against the feature-based families.
#
# ## Why the fit is per fold, and what is reused
#
# The components are estimated on each fold's training rows and applied unchanged to that fold's
# validation rows, with the fitted state saved rather than refitted. Refitting on the validation
# rows would let the components be shaped by the returns they are about to be scored on, and the
# failure would be invisible: the predictions look ordinary, the backtest runs, and it reports a
# strategy built from knowing how those contracts would go on to co-move. There is no output
# frame in which that is detectable, which is why it is handled by construction rather than by a
# check afterwards.
#
# The factor count is declared rather than chosen per fold. Choosing it per fold on validation
# performance would select the number that best suited each window's outcomes, and an earlier
# training window supports fewer well-estimated components than a later one anyway - so a fixed
# count is a compromise made deliberately and recorded, not an optimum found.

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
# **Resolving the identity before fitting is what makes the run answerable afterwards.** The
# identity hashes the whole specification - configuration, fold geometry, input artifact
# digests, the code's declared behaviour version - and it is computed from the declaration
# rather than from the result. A row already in the registry under that identity is served back
# instead of refitted, so two runs of one specification cannot record two different answers.
# Seeing it here, before anything is fitted, is also what lets a reader tell a re-run that
# recomputed nothing from one that quietly fitted something new.
#
# The checkpoint schedule in the plan is worth reading as a contrast rather than a setting.
# `10b` publishes several checkpoints because a neural fit passes through a sequence of states
# and which one to keep is a real choice that enters selection. PCA has no such sequence: the
# solution is closed-form, so there is one fitted state per fold and nothing to checkpoint.
#
# **Both horizons are declared explicitly rather than derived.** A latent-factor stage has no
# per-label parameter to vary: the components come from the return panel, which is the same
# panel whatever horizon is being predicted, so the two requests differ only in the outcome the
# factor forecasts are mapped onto. Declaring them separately keeps each horizon's predictions a
# distinct population with its own identity, rather than one fit reused under two labels - which
# is what would make the downstream count of configurations wrong.
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
#
# Requiring the complete key set matters more for a factor model than for a per-product one. A
# per-product model that failed on one contract leaves that contract without a prediction and
# the gap is local. A factor model's output for every product depends on the panel it was fitted
# across, so a missing column does not leave one prediction missing - it changes the components
# themselves, and therefore every prediction the fold publishes.
#
# ### Why this is the baseline the other configuration has to beat
#
# PCA estimates very little: a covariance matrix and its leading eigenvectors, closed-form, with
# no tuning beyond the factor count. That parsimony is why it is here. The neural stochastic
# discount factor in `10b` has far more freedom, and freedom on a panel this size is as likely
# to fit noise as structure - so the comparison is informative only because one side of it is
# this simple. If the elaborate method does not beat the eigenvectors of a covariance matrix,
# that is the finding.

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
