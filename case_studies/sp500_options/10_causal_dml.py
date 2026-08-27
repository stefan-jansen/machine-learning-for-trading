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
# # S&P 500 Options: Causal DML Execution
#
# This notebook estimates the effect of the variance-risk-premium treatment on the
# return-to-expiry outcome. It declares the request through the shared causal boundary and exposes
# the resolved estimand, timing, confounders, nuisance model, covariance design, and refutation
# protocol before execution.
#
# `11_model_analysis` interprets the causal estimates. This notebook validates the computation
# and publishes its artifact only.
#
# Prerequisites: `03_financial_features`, `04_model_based_features`, and `05_evaluation`.

# %%
"""Execute the declared S&P 500 options causal DML request."""

import polars as pl

from case_studies.research import supersedes_for
from case_studies.sp500_options.research_workflow import open_study

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
PREVIEW_REDUCTIONS: dict = {}
SUPERSEDES_CAUSAL: str = "4e310dbab236"

# %% [markdown]
# ## Declared and resolved request
#
# A preview must declare all sample, symbol, fold, or placebo reductions. Canonical execution uses
# the complete pre-holdout analysis population.
#
# ### What `SUPERSEDES_CAUSAL` retires here
#
# `CausalResult.one` resolves a label to exactly one canonical identity, so a refit has to name the
# identity it replaces or the registry is left with two and refuses. The retired identity is
# `4e310dbab236`.
#
# It is worth saying what did and did not change, because the two are usually the same and here
# they are not. The block-permutation refutation now records *how* its block size was derived, not
# only what it was: the resolver takes the larger of the label buffer and the treatment's own
# construction window, and it writes down both along with which one governed. For this case study
# the label buffer is 35 steps and the `vrp_21d` treatment window is 21, so the block size is
# unchanged at 35 and the basis is the label buffer. Method, seed, cadence, gap policy and the 100
# placebo draws are all unchanged too.
#
# So the estimate and the refutation this run publishes are the same numbers the retired identity
# held. What was wrong with the old identity was not its arithmetic but its silence: it recorded a
# block size of 35 with nothing saying where 35 came from, which is indistinguishable from a block
# size that happened to be 35 for the wrong reason. That is the defect the retirement closes.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
request_table = pl.DataFrame(
    {
        "method": ["dml"],
        "label": ["ret_to_expiry"],
        "config_name": ["dml"],
        "execution_tier": [EXECUTION_TIER],
    }
)
request_table

# %%
request = study.causal(
    **request_table.row(0, named=True),
    preview_reductions=PREVIEW_REDUCTIONS,
    supersedes=supersedes_for(SUPERSEDES_CAUSAL, "ret_to_expiry", labels=["ret_to_expiry"]),
)
resolved = request.resolve()
computation = resolved.spec["computation"]
estimand = computation["estimand"]
causal_plan = pl.DataFrame(
    {
        "treatment": [estimand["treatment"]],
        "outcome": [estimand["outcome"]],
        "confounders": [", ".join(estimand["confounders"])],
        "treatment_observed_at": [estimand["treatment_observed_at"]],
        "outcome_horizon": [estimand["outcome_horizon"]],
        "folds": [computation["cv"]["n_folds"]],
        "embargo_periods": [computation["cv"]["embargo_periods"]],
        "nuisance_model": [computation["model"]["class"]],
        "covariance": ["HAC with the outcome horizon"],
        "placebo_method": [computation["refutation"]["method"]],
        "placebo_block": [computation["refutation"]["block_size"]],
        "placebo_block_basis": [computation["refutation"]["block_size_basis"]],
        "analysis_rows": [computation["analysis_population"]["n_rows"]],
        "training_hash": [resolved.identity],
    }
)
causal_plan

# %% [markdown]
# ## Execute and validate
#
# The shared DML runner fails on missing confounders, invalid temporal folds, incomplete nuisance
# fits, or a non-finite HAC standard error. A cached result must match the complete resolved
# identity before it can be reused.

# %%
if EXECUTION_TIER == "preview" and (not WORKSPACE or not PREVIEW_REDUCTIONS):
    raise ValueError("preview execution requires WORKSPACE and PREVIEW_REDUCTIONS")
result = resolved.run()
if not result.complete or result.hash != resolved.identity:
    raise RuntimeError("causal execution did not publish the complete resolved request")

# %% tags=["results"]
artifact = pl.DataFrame(
    {
        "causal_hash": [result.hash],
        "label": [resolved.spec["label"]],
        "execution_tier": [result.execution_tier],
        "complete": [result.complete],
    }
)
artifact

# %% [markdown]
# The registered causal artifact is the handoff to `11_model_analysis`. No estimate or empirical
# conclusion is interpreted here.
