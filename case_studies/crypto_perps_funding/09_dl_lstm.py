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
# # Does the premium's path carry more than its summaries?
#
# [`06_linear`](06_linear.ipynb) fitted a design matrix that is mostly one economic quantity - the
# **premium**, the gap between the perpetual price and spot that the funding payment is computed
# from - measured many ways. Among those ways are hand-built summaries of the premium's recent
# *path*: its change over six horizons, its volatility over four, its z-score over two windows, its
# quantile position over three. Each of those columns compresses a stretch of history into one
# number, and a human chose the compression.
#
# A sequence model does not take that compression as given. It reads the last 60 settlements of
# every feature as an ordered window and learns its own summary. So the question this notebook and
# [`10_dl_tcn`](10_dl_tcn.ipynb) put to the data is narrow and answerable: **on this case study,
# does a learned representation of the path beat the hand-built one?** Not "are neural networks
# useful" - the design matrix already contains a considerable amount of path information, and the
# sequence family has to earn its keep against that, not against a naive baseline.
#
# Two architectures are fitted here against the same request:
#
# - **NLinear** is the baseline, and it is deliberately almost nothing. It subtracts the last value
#   of each window from the window, applies a single linear map to what remains, and adds the
#   subtracted value back. It has no recurrence, no gating and no nonlinearity. It exists so that
#   "the LSTM did better" has to mean better than the simplest thing that reads the same window in
#   the same order - which, on financial series, is a bar a great many published architectures do
#   not clear.
# - **LSTM** is the recurrent model: two layers, a hidden state of 64, dropout 0.1. It processes
#   the window one settlement at a time and carries a state forward, so unlike NLinear it can in
#   principle represent an interaction between what happened early in the window and what happened
#   late.
#
# Both go through the same request contract - the same feature order, the same folds, the same
# missing-observation policy, the same checkpoint schedule. **That is the point of running them
# from one notebook.** When the two differ in a later backtest, the difference is the architecture,
# because nothing else was allowed to vary.
#
# ## The grid is 8-hourly, and gaps in it are real
#
# A perpetual's funding is settled every 8 hours, and this case study's observation grid is that
# settlement cadence. A lookback of 60 is therefore **60 settlements, about 20 days** - not 60 days
# and not 60 rows of whatever happened to be adjacent in the file.
#
# That distinction has teeth here. A perpetual can be delisted, halted, or newly listed, and the
# exchange's history has holes. If a 60-bar window were built by taking 60 adjacent *rows*, a
# window spanning a two-day outage would silently splice across it and present the model with a
# discontinuity as though it were a normal step. The resolved policy on every request below is
# `exclude_windows_crossing_missing_expected_periods`: a window that would cross a settlement the
# grid expects and the data does not have is **dropped, not imputed**. The eligible-row count in
# the contracts table is what survives that rule, and it is smaller than the row count of the
# panel.
#
# ## A checkpoint is a model, not a progress marker
#
# Each configuration trains for 100 epochs and persists its state every 5, so each produces 20
# checkpoints, and **each checkpoint is a distinct prediction identity** that a later backtest can
# select. Early stopping is not implemented as a rule that halts training; it is implemented as a
# population of checkpoints from which selection picks. That is why the population is frozen before
# the first fit: a checkpoint that trains and then turns out to be poor stays in the population it
# was declared in, and cannot quietly disappear from the count it is judged against.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Explain why a sequence model on an irregular observation grid needs a declared cadence, and
#   what goes wrong when window construction uses row adjacency instead.
# - Read a resolved sequence request and say what lookback, gap policy and eligible row count the
#   run will actually use, before anything is fitted.
# - Say what a checkpoint schedule buys, and why every checkpoint is registered as its own
#   prediction set rather than only the last or the best.
# - Recognise that a linear baseline sharing the sequence contract is the correct comparison for a
#   recurrent model, and that beating a cross-sectional model is not the same claim.
#
# **Book reference:** Chapter 19, recurrent neural networks for time series.
#
# **Prerequisites:** [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices, and
# [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds. The canonical run
# uses CUDA; the reduced run in CI does not.
#
# **What it writes:** one training run per configuration and one complete validation prediction set
# per checkpoint, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a named population.
# [`13_backtest`](13_backtest.ipynb) reads that population and selects on validation backtest
# Sharpe. **Selection happens there, not here.** Nothing in this notebook ranks anything.

# %%
import os

import polars as pl

from case_studies.crypto_perps_funding.research_workflow import (
    REGRESSION_LABELS,
    declared_contracts,
    freeze_official_model_population,
    model_request_catalog,
    open_study,
    plan_model_catalog,
    plan_specs,
    run_model_plan,
)

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
SUPERSEDES_POPULATION: str = ""
# The generation of this notebook's own checkpoint population that this run replaces, if any.
# Distinct from SUPERSEDES_POPULATION above, which is the case-wide official model population:
# the two are separate declarations and a refit can move either without moving the other.
SUPERSEDES_MODEL_POPULATION: str = ""
WORKSPACE = os.environ.get("ML4T_OUTPUT_DIR", "")
LABELS = REGRESSION_LABELS
PREVIEW_REDUCTIONS = {}
OVERRIDES = {"device": "cuda"}

# %% [markdown]
# ## 1. Resolve the sequence and checkpoint identities
#
# Nothing is fitted in this cell. `model_request_catalog` reads the configurations this case study
# declares for the regression labels and returns the requests they resolve to; the plan that
# follows binds those requests to the data on disk and computes an identity for each. Reading the
# resolved plan before training is what makes the run auditable: if the lookback, the gap policy or
# the eligible row count is not what you expected, you find out here rather than after the fits.
#
# `config_prefix=("nlinear", "lstm")` is what restricts this notebook to the two architectures
# discussed above. The TCN declared alongside them in `config/training/fwd_ret_8h.yaml` is fitted
# by [`10_dl_tcn`](10_dl_tcn.ipynb) against the same contract.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
official_population = (
    freeze_official_model_population(study, supersedes=SUPERSEDES_POPULATION or None)
    if EXECUTION_TIER == "canonical"
    else None
)
requests = model_request_catalog("deep_learning", labels=LABELS, config_prefix=("nlinear", "lstm"))
requests

# %% [markdown]
# The table below is the run's declaration of what it is about to do. `gap_policy` and `lookback`
# are read back out of the frozen specification rather than restated from the configuration file,
# so the table cannot drift from what the fit will use. `eligible_rows` is the count of window
# end-points that survive the gap rule - the effective sample the model is fitted on, which is
# always smaller than the panel and is the number to quote when describing how much data a
# sequence model here actually saw.

# %% tags=["results"]
plan = plan_model_catalog(
    study,
    requests,
    execution_tier=EXECUTION_TIER,
    overrides=OVERRIDES,
    preview_reductions=PREVIEW_REDUCTIONS,
)
# Sequence eligibility follows from the resolved gap policy and lookback, so read both from the
# frozen specification instead of restating the configuration file here.
resolved_preprocessing = [spec["computation"]["preprocessing"] for spec in plan_specs(plan)]
contracts = declared_contracts(plan).with_columns(
    pl.Series("gap_policy", [step["gap_policy"] for step in resolved_preprocessing]),
    pl.Series("lookback", [step["lookback"] for step in resolved_preprocessing]),
)
contracts.select(
    "label",
    "config_name",
    "gap_policy",
    "lookback",
    "checkpoint_value",
    "eligible_rows",
    "training_hash",
)

# %% [markdown]
# The complete case-wide population is recorded before the first fit, so a member that later
# fails to train cannot quietly disappear from the population it was declared in. This notebook
# produces one slice of it, and that slice must lie inside the declaration.

# %% tags=["results"]
if official_population is not None:
    outside = set(plan.expected_prediction_hashes) - set(official_population.members)
    if outside:
        raise RuntimeError(
            f"{len(outside)} declared checkpoints lie outside the official model population"
        )

# %% [markdown]
# ## 2. Execute the declared population
#
# The adapter fits each configuration on each fold, writes a checkpoint every fifth epoch, and
# registers one complete validation prediction set per checkpoint. A fitted state is persisted with
# a digest, and a cached state is reused only when the digest matches, so a resumed run cannot
# quietly continue from a state that a code change has invalidated.
#
# The completeness check below is the one that matters. A prediction set is `complete` when it
# covers every eligible validation key for its fold; a set that covers most of them is not a
# slightly worse result, it is a different sample, and comparing it against a full one would be
# comparing two things measured on different data. The run raises rather than publishing a
# population containing one.

# %% tags=["results"]
execution = run_model_plan(
    plan,
    supersedes=SUPERSEDES_MODEL_POPULATION or None,
    population_name="crypto-lstm-validation-predictions-v1"
    if EXECUTION_TIER == "canonical"
    else None,
)
catalog = execution.catalog_rows.sort("label", "config_name", "checkpoint_value")
if (
    catalog.height != len(plan.expected_prediction_hashes)
    or catalog.filter(~pl.col("complete")).height
):
    raise RuntimeError("sequence baseline and LSTM checkpoint population is incomplete")
catalog.select(
    "label",
    "config_name",
    "checkpoint_value",
    "training_hash",
    "prediction_hash",
    "complete",
)

# %% [markdown]
# ## Key takeaways and limitations
#
# - **Eligibility follows the declared cadence, not row adjacency.** A 60-bar window is 60 expected
#   8-hour settlements. A window that would cross a settlement missing from the data is dropped,
#   which is why `eligible_rows` is smaller than the panel and why that count, not the panel
#   height, is the sample size to quote.
# - **The linear baseline is the comparison that means something.** NLinear reads the same window,
#   in the same order, under the same contract, and has no recurrence at all. An LSTM that does not
#   beat it has not shown that recurrence bought anything on this data.
# - **Every checkpoint is a model.** Twenty per configuration, each registered as its own
#   prediction identity, and selection among them happens in [`13_backtest`](13_backtest.ipynb) on
#   validation backtest Sharpe. Reporting the best checkpoint's score as though one model had
#   achieved it would be reporting a maximum over twenty draws as a single measurement.
# - **The history is short and the folds are few.** This case study's usable perpetual funding
#   history supports two validation folds, and a two-layer LSTM with a 64-unit hidden state has far
#   more capacity than two folds of an 8-hourly panel can identify. Dropout and the checkpoint
#   population are doing the regularization that a longer history would not need as badly.
# - **A fixed lookback is a modelling assumption, not a neutral default.** Sixty settlements is
#   about twenty days. Any dependence on something that happened before that window is invisible to
#   these models by construction, however long the funding cycle they are meant to capture.
