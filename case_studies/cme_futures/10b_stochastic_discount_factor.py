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
# One framing to carry through the rest: this is the most theoretically motivated model in the
# case study and it is given no special standing because of that. Its rows enter the same funnel
# as the linear model's, are selected on the same statistic, and can lose to a gradient-boosting
# configuration that rests on no asset-pricing argument at all. A better story about why a model
# should work is not evidence that it does.
#
# Prerequisites: `03_financial_features`, `04_model_based_features`, and `05_evaluation`.

# %% [markdown]
# ## What a stochastic discount factor is
#
# Asset pricing has one central result worth stating plainly, because everything this notebook
# does follows from it. If prices admit no arbitrage, then there exists a single random variable
# - call it `m` - such that for every asset, the expected product of `m` and that asset's return
# is the same constant. One object prices everything: equities, bonds, corn futures, all of it.
# That object is the **stochastic discount factor**.
#
# Its usefulness is that it converts "which assets earn more, and why" into a question about one
# quantity. An asset earns a premium when its return covaries negatively with `m` - when it pays
# badly in the states `m` says are expensive, which are the states investors care most about
# being protected in. Under that view a risk premium is not compensation for variance; it is
# compensation for failing to pay off when payment matters.
#
# `m` is not observable. But it is a well-defined thing to estimate, and estimating it with a
# neural network means not having to assume in advance what functional form it takes - which
# matters, because the classical models that assume one (a market factor, a small set of
# characteristics entering linearly) have a long record of being rejected on the data.
#
# ## How this differs from `10a`, on two axes rather than one
#
# The index for this stage frames the two configurations as differing in objective. That is true
# and it is not the whole difference - they also see different data, which is worth being exact
# about because it changes what each is capable of finding.
#
# **PCA sees returns only.** `run_pca_fold` takes the characteristics panel and discards it with
# `del` before fitting; it is handed a matrix of product returns and nothing else.
#
# **The SDF sees returns and characteristics.** `run_sdf_fold` passes `chars_train` through to
# `_cross_section_batch(chars_train, returns=returns_train)`, and the number of instruments it
# builds is derived from the characteristics' width. So carry, momentum and volatility are
# inputs here in a way they are not in `10a`.
#
# The two differences compound. PCA asks which directions explain the most variation in returns,
# using returns alone. The SDF asks which combination of assets, weighted by their observable
# characteristics, best explains the cross-section of returns - so it can express "products with
# high carry and low volatility load on this factor" in a way PCA structurally cannot, because
# PCA never sees carry.
#
# That is why the comparison between them is informative rather than decorative. A win for the
# SDF is evidence that the characteristics carry pricing information beyond what the return
# covariance already encodes. A win for PCA is evidence that they do not, and that the extra
# freedom bought overfitting.

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
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
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
#
# ### Why checkpoints are part of what gets selected
#
# A neural fit is not one model, it is a trajectory. It passes through a sequence of states as
# training proceeds, and which one is kept is a decision with the same standing as the choice of
# architecture. `checkpoint_epochs` declares the epochs this configuration publishes, so each
# becomes its own candidate row, and `13_backtest` selects among them on validation backtest
# Sharpe like any other configuration.
#
# Publishing them rather than picking one is the honest arrangement. Choosing the best epoch by
# looking at validation performance and then reporting that model's validation performance is
# selection inside the number being reported, and it does not stop being that because the choice
# was made by hand. Declaring the checkpoints up front puts the choice into the same funnel and
# the same trial count as everything else - which the deflated Sharpe downstream then divides by.
#
# This is the concrete contrast with `10a`, which has one fitted state per fold and nothing to
# checkpoint: its solution is closed-form, so there is no trajectory to choose a point on.
#
# ### What the fold discipline protects
#
# Training rows determine the representation and validation rows are transformed without
# refitting. For a model that reads characteristics as well as returns, that discipline covers
# two channels rather than one: the factors must not be shaped by validation returns, and the
# instrument weights must not be shaped by validation characteristics either. Both would produce
# predictions that look ordinary and a backtest that runs.
#
# ### Why the complete key set is required
#
# The same reason it is required in `10a`, and for the same structural cause: this is a
# cross-sectional model, so a missing product does not leave one prediction absent - it changes
# the estimated factors and therefore every prediction the fold publishes. An incomplete key set
# is refused rather than published short.
#
# ### Where the freedom cuts against it
#
# The SDF has far more capacity than the eigenvectors of a covariance matrix, and this panel is
# thirty products. Capacity on a panel that size is as likely to fit noise as structure, and
# nothing in a training loss distinguishes the two. That is what makes `10a` the baseline worth
# beating rather than a formality - and what makes a narrow SDF win, if one appears, weaker
# evidence than its margin suggests.

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
