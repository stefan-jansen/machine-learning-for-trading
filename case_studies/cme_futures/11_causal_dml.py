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
# Nothing estimated here competes for that selection. A causal result is not a candidate
# configuration: it produces no prediction set, is not eligible for a backtest, and never enters
# the funnel. A reader looking for it on the leaderboard will not find it, and that absence is
# the design rather than an omission - the two stages answer questions that do not compare.
#
# One consequence for how the result is read: a small or insignificant effect here does not
# invalidate the trading strategy, and a large one does not endorse it. The strategy stands or
# falls on out-of-sample Sharpe either way. What the estimate changes is what the chapter may
# claim about *why* it works.
#
# Prerequisites: `03_financial_features`, `04_model_based_features`, and `05_evaluation`.

# %% [markdown]
# ## The question, and why this case study in particular has to ask it
#
# The treatment here is `carry_pct` - the term-structure spread between the front contract and
# the next one. That is not an arbitrary choice of variable to interrogate. **Carry is the
# signal this entire case study trades.** Every model in stages 06 through 10 is, in one form or
# another, learning to predict returns from carry and things built out of carry.
#
# So the question this notebook asks is whether the premise underneath all of that holds: does
# carry itself move subsequent returns, or does carry merely travel alongside something else
# that does? A predictive model does not care - it profits either way, as long as the
# association persists. The chapter's explanation of *why* the strategy works cares a great
# deal, because an explanation built on the wrong mechanism is wrong even when the strategy is
# profitable.
#
# This is also why the answer cannot come from the backtest. A backtest that earns a positive
# Sharpe from carry demonstrates that carry predicted returns over that history. It says nothing
# about whether carry was the cause, and no amount of out-of-sample performance converts one
# into the other.
#
# ### What is being adjusted for, and why each one is a candidate explanation
#
# Double machine learning estimates the effect of the treatment after flexibly removing what the
# confounders explain of both the treatment and the outcome. It does this by predicting each
# from the confounders alone, using models that need not be linear in anything, and then
# estimating the effect from what is left over in each. The predictions are cross-fitted, so the
# model that residualizes an observation was never fitted on it.
#
# The three confounders are the three rival explanations worth taking seriously here:
#
# - **`vol_21d`.** Contracts in backwardation are frequently contracts under supply stress, and
#   stress raises volatility. If volatile contracts earn more simply for being volatile, carry
#   would look rewarded without being the reason.
# - **`momentum_composite`.** Carry and momentum are correlated in futures - a contract in
#   sustained backwardation has often been rising. An unadjusted carry effect could be a
#   momentum effect wearing carry's label.
# - **`carry_rank`.** This is the subtle one, and it is the reason the confounder list is not
#   just "the other features". `carry_rank` is a product's position in the cross-section, while
#   `carry_pct` is its level. Those come apart: a contract can have a high carry level in a
#   period when every contract does, and rank in the middle of its peers. Adjusting for the rank
#   asks whether the carry *level* carries information beyond where the product sits relative to
#   the others - which is precisely what a cross-sectional strategy is already exploiting.
#
# ### Why the standard error is HAC
#
# The uncertainty around the effect is computed with a heteroskedasticity- and
# autocorrelation-consistent estimator, because the ordinary one assumes the residuals are
# independent and identically scattered and neither holds on a futures panel. Volatility
# clusters, so the scatter is larger in some periods than others; and overlapping outcomes tie
# neighbouring residuals together, so they carry information about each other. Both inflate the
# effective information in the sample relative to what an ordinary standard error assumes, and
# the result is a confidence interval that is too narrow and a t-statistic that is too large.
# The HAC estimator does not fix the estimate - it corrects what may be claimed about it.
#
# ### Why the placebo block is one bar here, and not 252
#
# The refutation permutes the treatment in contiguous blocks and re-runs the whole estimation,
# building a distribution of effects under the hypothesis that the treatment does nothing. The
# block has to be long enough to preserve whatever serial dependence the real treatment has, or
# the placebo is a weaker opponent than the truth and every p-value looks significant.
#
# `causal.treatment_window` is 1 for this case study, and the reason is a property of the
# construction rather than a judgement. `carry_pct` is
# `(c0_price - c1_price) / c0_price * 12`, computed from two prices at the same timestamp.
# Nothing rolls, nothing averages, no window is spanned - so one value carries no dependence of
# its own and a one-bar block destroys nothing the refutation needs.
#
# The contrast with a rolling treatment is worth holding onto, because it is where this is
# usually got wrong. A treatment built from a 252-session window overlaps its neighbours in 251
# of them, and permuting it in short blocks shreds that overlap; the placebo distribution
# narrows, and the p-value collapses toward zero whether or not the effect is real. The number
# is declared in `setup.yaml` and derived from how the column is built, because inferring it
# from a window list would put a wrong number behind a right-looking one.

# %%
"""Fit the declared CME futures double-machine-learning requests."""

import polars as pl

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    open_study,
    product_universe_table,
)
from case_studies.research import supersedes_for

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None
PREVIEW_REDUCTIONS: dict = {}
SUPERSEDES_CAUSAL: str = ""

# %% [markdown]
# ## Resolve the estimands
#
# Preview row and fold limits must be named in `PREVIEW_REDUCTIONS`. They enter the computation
# identity and cannot be mistaken for canonical causal estimates.
#
# **Missing confounders are an error rather than an invitation to change the estimand**, which
# is worth stating as a rule because the alternative is so tempting. Dropping a confounder that
# failed to resolve leaves a request that still runs and still returns a number - a number
# answering a different question, with no adjustment for the thing that went missing, and
# nothing in the result marking it. An estimand is the whole specification: outcome, treatment,
# confounder set, timing and embargo together. Changing any part of it produces a different
# quantity, not a degraded version of the same one.

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
        supersedes=supersedes_for(SUPERSEDES_CAUSAL, label, labels=list(ALL_LABELS)),
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
#
# The restart check runs the request twice and requires the same hash. That is cheap here
# because the second call is served from the registry, and it is worth doing because a causal
# estimate has no external reference to be checked against. A predictive model can be caught by
# out-of-sample performance; an effect estimate cannot, so reproducibility is most of what is
# available. An estimate that moved between two runs of the same specification would mean the
# seed does not control everything the fit depends on, and the number would be one draw from a
# distribution nobody characterised.
#
# `SUPERSEDES_CAUSAL` above is how a re-run names the identity it retires, per label. A refit
# under a changed specification produces a second current identity for the same label, and
# `CausalResult.one` resolves a label to exactly one - so the run has to say which it replaces,
# and the retired estimate stays in the registry rather than being deleted.

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
