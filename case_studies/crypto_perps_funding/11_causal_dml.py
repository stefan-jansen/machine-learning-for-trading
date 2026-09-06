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
# # A different question from the one every other model here asks
#
# Every notebook from [`06_linear`](06_linear.ipynb) through [`10_dl_tcn`](10_dl_tcn.ipynb) asks
# the same question in different ways: **given what I can see now, what is the best guess for the
# next 8-hour return?** A model that answers it well is useful whether or not any of its features
# cause anything. Collinear features, proxies, coincidences - all fine, as long as the association
# holds out of sample.
#
# This notebook asks something else: **if the premium's z-score were higher, holding the other
# declared drivers fixed, would the subsequent return be different?** That is a question about an
# intervention, and no amount of predictive accuracy answers it. The two live in the same case
# study and must not be reported as though they were the same finding, which is why the causal
# result is registered under its own identity and never enters the population that
# [`13_backtest`](13_backtest.ipynb) selects from.
#
# ## The estimand, stated before anything is fitted
#
# - **Treatment:** `premium_zscore_14d`, the premium's standing relative to its own recent range.
#   Continuous, not binary - so the estimate is a slope, the change in expected return per unit of
#   treatment, not a difference between two groups.
# - **Outcome:** `fwd_ret_8h`, the return over the settlement interval after the decision time.
# - **Confounders:** `price_vol_14d`, `funding_rate`, `premium_dev_mean_14d`. These are the
#   variables declared to drive both the treatment and the outcome, and adjusting for them is the
#   entire identification claim.
#
# That list is short, and its shortness is the honest part of the exercise. **Double machine
# learning removes confounding by variables you name.** It does nothing about one you did not, and
# nothing about the possibility that the relationship runs the other way. Naming three confounders
# is a claim that those three are the relevant ones; the estimate is only as good as that claim.
#
# ## What double machine learning actually does
#
# The naive approach - regress the outcome on the treatment and the confounders together - biases
# the treatment coefficient whenever the confounders enter nonlinearly, because whatever the linear
# term fails to absorb leaks into the treatment. DML avoids that by splitting the problem in two:
#
# 1. Predict the **outcome** from the confounders alone, and take the residual.
# 2. Predict the **treatment** from the confounders alone, and take the residual.
# 3. Regress residual on residual. What remains is the part of the treatment the confounders do not
#   explain, against the part of the outcome they do not explain.
#
# The two nuisance predictions can be any flexible learner, because the residual-on-residual step
# is what carries the causal interpretation. **Cross-fitting** is what keeps that step honest: each
# observation's residual is computed by a model that did not see it, so an overfitted nuisance
# model cannot manufacture a residual correlation. Here the folds are chronological with an
# embargo, so the nuisance models are also never fitted on data that comes after what they predict.
#
# ## Why the placebo is the part to read carefully
#
# A causal estimate on financial data will produce a number whether or not there is anything there,
# so the refutation matters more than the point estimate. The placebo permutes the treatment and
# re-estimates, and the estimate should collapse. What makes it a real test rather than a formality
# is the **block size**, discussed at the table below: permuting one bar at a time destroys exactly
# the serial dependence that makes the original estimate hard to get right, and a placebo that
# destroys the difficulty is a test the estimate passes for free.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - State an estimand - treatment, outcome, confounders, horizon - precisely enough that someone
#   else could disagree with it.
# - Explain what cross-fitting protects against, and why chronological folds with an embargo are
#   the right shape for it on overlapping financial labels.
# - Say why a block-permutation placebo needs a block long enough to preserve serial dependence,
#   and how the block size is derived here rather than chosen.
# - Keep a causal estimate and a predictive result in separate reports, and say in one sentence
#   why combining them would misdescribe both.
#
# **Book reference:** Chapter 15, causal inference for trading research.
#
# **Prerequisites:** [`03_financial_features`](03_financial_features.ipynb),
# [`02_labels`](02_labels.ipynb) and [`05_evaluation`](05_evaluation.ipynb) - the features, the
# return labels and the purged walk-forward folds.
#
# **What it writes:** one causal result under its own identity in `run_log/registry.db`. It is read
# by [`12_model_analysis`](12_model_analysis.ipynb), which reports it **beside** the predictive
# results rather than among them.

# %%
import os

import polars as pl

from case_studies.crypto_perps_funding.research_workflow import open_study
from case_studies.research import supersedes_for

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE = os.environ.get("ML4T_OUTPUT_DIR", "")
LABEL = "fwd_ret_8h"
CONFIG_NAME = "dml"
PREVIEW_REDUCTIONS = {}
OVERRIDES = {}
# The causal identity this run retires, empty by default because a reader has nothing to
# retire. `run_log/` is not shipped, so a first run meets an empty registry, and a
# predecessor named here that is not in it is rejected at the registering write - after the
# DML fit and every placebo refit have been paid for.
#
# It is still needed on the machine that holds the chain. A causal identity hashes the whole
# of `case_studies/utils/causal.py`, so any edit to that file moves it, and `CausalResult.one`
# resolves a label to exactly one canonical identity: a re-run that does not name what it
# replaces leaves two live and the next notebook fails with "resolved to 2 identities". Pass
# it for that one-time repair instead of committing it:
#
#   papermill 11_causal_dml.ipynb out.ipynb -p SUPERSEDES_CAUSAL <the hash being retired>
SUPERSEDES_CAUSAL: str = ""

# %% [markdown]
# ## 1. Resolve the estimand and the refutation contract
#
# Nothing is fitted below. The request resolves to a specification and an identity, and the table
# prints the fields that decide what the estimate means: what is being intervened on, what is being
# measured, over what horizon, how the nuisance folds are cut, and how the placebo will be built.
# Reading them before the fit is the only point at which disagreeing with the estimand is cheap.
#
# `eligible_rows` is the analysis population - the rows that survive having a treatment, an
# outcome and every confounder present. Quote that, not the panel height, when describing what the
# estimate rests on.

# %% tags=["results"]
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
request = study.causal(
    method="dml",
    label=LABEL,
    config_name=CONFIG_NAME,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
    overrides=OVERRIDES,
    supersedes=supersedes_for(SUPERSEDES_CAUSAL, LABEL, labels=[LABEL]),
)
resolved = request.resolve()
computation = resolved.spec["computation"]
pl.DataFrame(
    {
        "causal_hash": [resolved.identity],
        "outcome": [computation["estimand"]["outcome"]],
        "treatment": [computation["estimand"]["treatment"]],
        "outcome_horizon": [computation["estimand"]["outcome_horizon"]],
        "n_folds": [computation["cv"]["n_folds"]],
        "embargo_periods": [computation["cv"]["embargo_periods"]],
        "block_size": [computation["refutation"]["block_size"]],
        "block_size_basis": [computation["refutation"]["block_size_basis"]],
        "gap_policy": [computation["refutation"]["temporal_gap_policy"]],
        "eligible_rows": [computation["analysis_population"]["n_rows"]],
    }
)

# %% [markdown]
# `block_size` is the parameter the refutation lives or dies on, so it is on the
# table rather than buried in the spec. The placebo permutes contiguous blocks
# within each symbol; a block of one bar is an iid shuffle, which destroys the
# serial dependence the placebo is meant to keep and makes the test trivially
# easy to pass. Two things create that dependence and the block spans the longer
# of them: the overlapping labels span the outcome horizon, and the treatment
# spans its own construction window. Here the horizon is a single 8-hour bar
# while `premium_zscore_14d` is a 42-bar rolling statistic, so the treatment
# window sets the block and `block_size_basis` says so.

# %% [markdown]
# ## 2. Execute, and register the result separately
#
# The fit runs the nuisance models across the chronological folds, forms the residual-on-residual
# estimate, and then pays for the placebo refits - which is where most of the cost is, since the
# whole procedure is repeated once per placebo draw.
#
# The check below refuses a result whose specification is not the one that was resolved. That is
# not defensive coding: a causal identity hashes the whole of `case_studies/utils/causal.py`, so an
# edit to that file mid-run would produce a result describing a contract that no longer exists, and
# the notebook downstream would resolve the label to two live identities and stop.

# %% tags=["results"]
result = resolved.run()
if not result.complete or result.spec != resolved.spec:
    raise RuntimeError("causal execution is incomplete or has conflicting identity")
pl.DataFrame(
    {
        "causal_hash": [result.hash],
        "n_obs": [result.metrics["n_obs"]],
        "complete": [result.complete],
        "execution_tier": [result.execution_tier],
    }
)

# %% [markdown]
# ## Key takeaways and limitations
#
# - **This is not a trading signal and must not be reported as one.** The estimate answers what
#   would happen under an intervention on the premium z-score. The backtest in
#   [`13_backtest`](13_backtest.ipynb) answers what a strategy would have earned. A number from
#   here quoted as evidence for a strategy is a category error, and the separate registration
#   exists to make that mistake require deliberate effort.
# - **Adjustment covers three named confounders and nothing else.** DML removes bias from variables
#   it is given. An omitted common cause, reverse causation from returns to the premium, or a
#   confounder measured with error all survive it untouched, and no diagnostic in this notebook
#   would reveal them. The identifying assumption is an argument, not an output.
# - **The refutation is the load-bearing part.** A point estimate arrives whether or not there is
#   anything to estimate. What distinguishes the two is whether the placebo collapses, and whether
#   the placebo was built to be hard - which is what the block size decides.
# - **The identity hashes the estimator's source.** Any edit to `case_studies/utils/causal.py`
#   produces a different causal identity, whether or not it changes a number. That is deliberate: a
#   causal claim is a claim about a procedure, and a changed procedure is a new claim until someone
#   shows the estimate did not move.
# - **A preview run is not a canonical result.** The reduced sample and fold counts a preview
#   applies are recorded in its identity and it is barred from the canonical population, so a cheap
#   run can never be mistaken for the published estimate.
