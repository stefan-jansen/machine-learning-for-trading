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
# # CME Futures: Sequence Models
#
# This notebook evaluates the declared NLinear and LSTM sequence configurations. Each input window
# contains observations from one product and ends before its prediction timestamp. Purge gaps and
# fold boundaries prevent a sequence from crossing into another validation interval, and hidden
# state does not pass between products or folds.
#
# Every declared epoch checkpoint is published with fitted weights and exact chronological
# eligibility. MC dropout is not an undeclared side experiment. Configuration selection remains the
# validation backtest decision in `13_backtest`.
#
# Prerequisites: `03_financial_features`, `04_model_based_features`, and `05_evaluation`.

# %% [markdown]
# ## What a sequence model reads that the other families do not
#
# Every family up to this point saw one row per product per decision: a vector of features
# describing that product at that moment. Anything about how it got there had to be engineered
# into a column - a 21-session volatility, a momentum composite, a carry z-score against a
# rolling window. The model saw the summary, never the path.
#
# A sequence model reads the path. Its input is a window of consecutive observations for one
# product, and the architecture is built to make use of their order. The claim being tested is
# that the shape of recent history carries information that no fixed set of summary statistics
# captured - that a product whose carry rose steadily to its current level differs from one that
# spiked and fell back, even where both end at the same value with the same 21-session
# volatility.
#
# For thirty futures products the windows are also the scarcest data in the case study. A
# feature-row model gets one training example per product per session; a sequence model needs a
# whole window per example, so the same history yields fewer independent examples and they
# overlap heavily with each other. That is the structural reason to expect these models to
# struggle here relative to a benchmark with millions of series, and it is worth holding
# alongside whatever the backtest reports.
#
# ### Two architectures, and why both
#
# **LSTM** processes the window one step at a time, carrying a hidden state forward and learning
# what to keep and what to forget. It is the general answer, and its generality is the cost: it
# has many parameters, it trains slowly, and on a short noisy series it has ample capacity to
# memorize.
#
# **NLinear** is close to the opposite. It is a linear map from the window to the forecast, with
# a normalization step that subtracts the window's last value before the map and adds it back
# afterwards. That subtraction is the whole idea: it makes the model predict the *change* from
# where the series currently sits rather than the level, which removes the drift that otherwise
# dominates a naive fit.
#
# It is here because a body of recent work found that simple linear baselines matched or beat
# elaborate sequence architectures on many forecasting benchmarks once evaluated carefully - a
# finding that survived enough scrutiny to be worth designing around. Running both is what turns
# "the sophisticated model should win" into something this case study measures rather than
# assumes.
#
# ## Where a sequence model can leak, and what stops it
#
# A window is a span of time rather than a point, which gives leakage more places to enter than
# the other families have.
#
# - **Each window ends before its prediction timestamp.** The last observation a window contains
#   is strictly earlier than the moment being predicted, so a forecast never reads the bar it is
#   forecasting.
# - **Purge gaps and fold boundaries stop a window crossing into another interval.** Without them
#   a window ending just after a fold boundary would extend back across it, and validation rows
#   would be predicted from a window overlapping the training period. The failure would be
#   invisible in the output: the prediction is dated correctly and the returns are real.
# - **Hidden state does not pass between products or folds.** An LSTM's state accumulates
#   whatever it has seen, so carrying it across a boundary carries information across that
#   boundary too - and unlike a feature column, the state never appears in any frame, so nothing
#   downstream could detect it.
#
# The three are separate mechanisms rather than one guarantee stated three times, which is why
# they are enforced separately rather than by a single check on the output.

# %%
"""Fit the declared CME futures sequence-model population."""

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
# The request rows identify architecture, label, and published configuration. Sequence length,
# checkpoint schedule, seed, gap policy, and device enter the resolved computation identity.
#
# **The device is declared here rather than inherited.** With no override the shared sequence
# adapter falls back to a literal `"cuda"` written in `case_studies/utils/deep_learning.py`, and
# resolving the request raises `CUDA was requested for sequence training, but CUDA is unavailable`
# rather than quietly moving the fit to the CPU. That refusal comes from resolving the request, so
# it arrives before any fitting starts. A CUDA device is therefore a hard requirement of this
# population, and stating it in the request puts that requirement where a reader meets it instead
# of two layers below. The resolved specification hash is the same with the override as without,
# so this names what the published run already did.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
requests = model_request_catalog("deep_learning", labels=ALL_LABELS)
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
# The shared sequence adapter owns window construction, checkpoint reload, prediction coverage, and
# restart. A failed configuration cannot remove itself from the population snapshot.
#
# **"Cannot remove itself" is the load-bearing clause.** The natural way to write a sweep is to
# catch a failure, log it, and carry on with what worked - which produces a population defined
# by what happened to train rather than by what was declared. The leaderboard still looks
# sensible, and the configuration that failed is indistinguishable from one that was never
# requested. Sequence models make this more likely than the other families do, because they are
# the ones that run out of memory or fail to converge on a thin product.
#
# ### What MC dropout is, and why it is declared rather than switched on
#
# Dropout during training randomly disables units so the network cannot rely on any one path.
# **MC dropout** leaves it enabled at prediction time and runs the forward pass several times, so
# each pass gives a slightly different answer and their spread estimates the model's uncertainty
# about that prediction.
#
# That is a useful quantity - it is what an allocator sizing inversely to uncertainty would want
# - but it changes what the model outputs. A prediction averaged over stochastic passes is not
# the same number as the deterministic one, and a run that quietly enabled it would publish
# different values under the same configuration name. So it is part of the declared
# configuration and enters the identity, which is what the header means by "not an undeclared
# side experiment": either the population says these predictions are MC-dropout predictions, or
# they are not, and no run gets to decide that on its own.
#
# ### Why checkpoints are published rather than chosen
#
# As in `08_tabular_dl`: a neural fit is a trajectory, and choosing the best epoch by validation
# performance before reporting that model's validation performance is selection inside the
# number being reported. Every declared checkpoint becomes a candidate row and `13_backtest`
# selects among them on Sharpe, so the choice sits in the same funnel and the same trial count
# as everything else.

# %%
if EXECUTION_TIER == "canonical":
    execution, population = run_official_model_catalog(
        study,
        requests,
        population_name="cme_futures-deep_learning-validation-v1",
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
    raise RuntimeError("sequence execution returned a partial prediction")
catalog
