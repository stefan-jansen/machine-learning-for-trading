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
# # CME Futures: Latent-Factor Requests
#
# The latent-factor stage contains two declared configurations. `10a_pca` fits principal components
# within each training fold. `10b_stochastic_discount_factor` estimates the neural stochastic
# discount factor within the same fold contract. Neither notebook selects by IC.
#
# This index exposes the complete request population without launching either computation. The two
# execution notebooks publish disjoint official populations that `13_backtest` later combines with
# the other predictive families.

# %% [markdown]
# ## What a latent factor is, and how this stage differs from the ones before it
#
# Every model up to this point was handed named predictors. Carry, momentum, the volatility
# estimate, the regime probability - each is a quantity somebody decided to compute, and the
# model's job was to weigh them. The choice of what to compute came from the researcher, and a
# driver nobody thought to name was a driver no model could use.
#
# A latent factor is inferred instead of specified. The starting observation is that futures
# returns move together far more than thirty independent series would: energy contracts rise and
# fall as a group, the metals do, the equity indices do, and there are days on which nearly
# everything moves the same way. That co-movement is evidence of a small number of underlying
# drivers acting on many contracts at once. A latent-factor method estimates those drivers from
# the covariance of returns themselves, without being told in advance what they are or how many
# there should be.
#
# The appeal is that it can find structure nobody encoded. The cost is that what it finds has no
# name and no economic interpretation attached - a factor is a direction in return space that
# explains variance, and whether it corresponds to anything a reader would recognise is a
# separate question the method does not answer.
#
# ## Why two configurations, and what separates them
#
# The two are not variations on one method. They disagree about what a factor is *for*, and
# that disagreement is the reason both are here.
#
# **`10a_pca` maximizes explained variance.** Principal components find the directions along
# which returns vary most, in order, each uncorrelated with the ones before it. It is linear,
# it has a closed-form solution, and it makes no reference to returns being predictable at all.
# Its first component on a futures panel is typically close to "everything moves together"; the
# next few usually separate the sectors. It is the standard baseline for exactly the reasons
# equal weight is one in the backtest stage: it is well understood, it estimates little, and
# anything more elaborate has to beat it to justify itself.
#
# The weakness is that variance and return are different quantities. The direction along which
# a panel varies most is not necessarily the direction that pays, and PCA has no mechanism for
# preferring one that does - a factor capturing a large, entirely unrewarded common movement is
# exactly what it is built to find first.
#
# **`10b_stochastic_discount_factor` starts from what prices assets.** Asset pricing theory says
# that if markets are free of arbitrage there exists a single random variable - the stochastic
# discount factor - whose covariance with any asset's return explains that asset's expected
# return. Everything that is priced is priced by the same object. The SDF is not observable, but
# it is a well-defined thing to estimate, and estimating it with a neural network means not
# having to assume in advance which functional form it takes.
#
# The difference from PCA is the objective rather than the architecture. PCA asks which
# directions explain the most variation; the SDF asks which combination best explains the
# cross-section of *returns*. A factor that moves a lot but earns nothing is a success for the
# first and a failure for the second.
#
# So the comparison between the two is not "which fits better". It is a question about this
# panel: whether the directions along which futures returns vary most are also the directions
# along which they are compensated. The two configurations are run under the same fold contract
# and the same universe precisely so the comparison isolates that.
#
# ## Why the factors are fitted inside each fold, and why that matters more here
#
# Both configurations estimate their factors within the training portion of each fold, never
# once over the whole panel. That is the same discipline every other family follows, but the
# consequence of breaking it is worse here and easier to miss.
#
# A supervised model that saw future data would be caught by its own validation score looking
# implausible. A latent-factor model fitted on the full sample fails more quietly: the factors
# are estimated from the covariance of returns, so a factor fitted over 2011 to 2025 encodes
# which contracts moved together across the entire period. Using it to form a position in 2014
# means holding a portfolio constructed from the knowledge that those contracts would go on
# co-moving. Nothing about the resulting prediction looks impossible. The returns are real, the
# weights are finite, and the backtest runs - it just reports a strategy that could not have
# been held.
#
# The cost of doing it correctly is visible in what the early folds can support. A covariance
# matrix over thirty products needs a meaningful amount of history before its estimate means
# anything, so the earliest training window supports fewer reliable factors than the latest,
# and a factor count fixed across folds is a compromise rather than a free choice. That is the
# tradeoff the declared configuration is making, and it is the reason the count is declared in
# `setup.yaml` rather than selected per fold - selecting it per fold on validation performance
# would choose the number that best suited each window's outcomes.
#
# ## What the two tables below show
#
# The universe table is the set of products the factors are estimated across. It is worth
# reading before the request catalog, because a latent factor is a property of the panel rather
# than of any one contract: adding or removing products changes what the factors are, in a way
# that changing the universe for a per-product model does not. Two runs over different universes
# do not produce comparable factors even under identical settings.
#
# The request catalog is the complete declared population - one row per label and configuration,
# resolved but unfitted. Reading it here is what makes the count the execution notebooks produce
# checkable against a declaration.
#
# ## Why neither selects by IC, and why this page launches nothing
#
# Both fit within each training fold, and both publish predictions like any other family. They
# are not privileged by being unsupervised: their rows enter `13_backtest` alongside the linear,
# gradient-boosting and sequence families and are selected on validation backtest Sharpe like
# everything else. A high IC here decides nothing, which is the same rule the whole case study
# runs under.
#
# This notebook computes neither. It exists so the declared request population can be read
# before anything is fitted - the two execution notebooks publish disjoint official populations,
# and seeing what they *will* contain is what makes a later count checkable against a
# declaration rather than against whatever finished.
#
# Disjoint is the part worth noticing. The two populations share no members, so `13_backtest`
# combines rather than reconciles them, and a configuration missing from one is not covered by
# the other being complete.

# %%
"""Show the declared CME futures latent-factor requests."""

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    model_request_catalog,
    product_universe_table,
)

# %%
requests = model_request_catalog("latent_factors", labels=ALL_LABELS)
universe = product_universe_table()
universe

# %%
requests.sort("label", "config_name")
