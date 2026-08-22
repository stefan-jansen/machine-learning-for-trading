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
# # Causal estimate of funding and subsequent returns
#
# This notebook keeps causal estimation separate from predictive model results. The request resolves
# the funding treatment, continuous return outcome, confounders, chronological nuisance folds,
# embargo, nuisance estimator parameters, and cadence-aware placebo policy before fitting.
#
# **Learning objectives**
#
# - state the treatment, outcome, and confounders that define the causal estimand;
# - inspect chronological nuisance folds and temporal refutation policy; and
# - distinguish a causal result from predictive model diagnostics.
#
# **Book reference:** Chapter 15, causal inference for trading research.
#
# **Prerequisites:** finalized funding features, return labels, and purged walk-forward folds.

# %%
import os

import polars as pl

from case_studies.crypto_perps_funding.research_workflow import open_study

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE = os.environ.get("ML4T_OUTPUT_DIR", "")
LABEL = "fwd_ret_8h"
CONFIG_NAME = "dml"
PREVIEW_REDUCTIONS = {}
OVERRIDES = {}

# %% [markdown]
# ## Resolve the estimand and refutation contract

# %% tags=["results"]
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
request = study.causal(
    method="dml",
    label=LABEL,
    config_name=CONFIG_NAME,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
    overrides=OVERRIDES,
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
        "gap_policy": [computation["refutation"]["temporal_gap_policy"]],
        "eligible_rows": [computation["analysis_population"]["n_rows"]],
    }
)

# %% [markdown]
# ## Execute the separate causal result

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
# - The causal identity includes the estimand, nuisance models, sample population, folds, and
#   refutation settings.
# - Preview sample limits remain outside canonical causal results.
# - Double machine learning adjusts for declared observed confounders; it cannot remove bias from an
#   omitted cause or establish that the identifying assumptions hold.
