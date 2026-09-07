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
# # A third way to read the same window
#
# [`09_dl_lstm`](09_dl_lstm.ipynb) put two readings of the 60-settlement window against each other:
# NLinear, which applies one linear map to the whole window at once, and an LSTM, which walks the
# window one settlement at a time and carries a state. This notebook adds a third, fitted against
# the identical request contract so that the comparison is architecture and nothing else.
#
# A **temporal convolutional network** slides a small filter along the window instead of stepping
# through it. The filter here is `kernel_size: 3`, so one convolution sees three consecutive
# settlements. Stacking convolutions with growing **dilations** - `1, 2, 4, 8`, meaning each
# successive block skips one, three, then seven settlements between the positions it combines -
# lets a shallow stack reach far back without one filter per lag. Every convolution is **causal**:
# it is padded on the left and the padding is trimmed from the right, so the value at a position is
# computed only from that position and earlier ones. A model that reads its own future within the
# window would score well and mean nothing.
#
# The arithmetic is worth doing once, because it is what the dilation schedule is chosen for. Each
# of the four blocks applies two convolutions at its dilation, so a block extends the reach by
# `2 x (3 - 1) x d`. Summed over `d` in `1, 2, 4, 8`, the receptive field is
# `1 + 4 x (1 + 2 + 4 + 8) = 61` settlements against a declared lookback of 60. **The stack is
# sized so the last position sees the entire window**, with one settlement to spare - and a
# shorter dilation schedule would leave the earliest part of the window unreachable no matter how
# long the lookback said it was.
#
# ## Where this differs from the LSTM, and why it might matter here
#
# The two architectures aggregate over time in genuinely different ways, and on this data that is
# not a detail.
#
# - The LSTM's prediction is read off the state after the **last** settlement, so information from
#   early in the window has to survive being carried through sixty updates to be used.
# - This TCN pools its representation by **averaging over all positions** before the output layer.
#   Nothing has to survive a recurrence, and a pattern that occurred early in the window
#   contributes on the same footing as one that occurred late.
#
# For a premium that mean-reverts on a timescale of days, the two are different hypotheses about
# where the signal sits: at the end of the window, or spread across it. Neither is obviously right,
# which is the reason to fit both rather than to pick one.
#
# ## Same contract, same gaps, same checkpoints
#
# Everything [`09_dl_lstm`](09_dl_lstm.ipynb) establishes about the observation grid applies here
# unchanged, because it is the same request contract. The grid is the 8-hour funding settlement
# cadence, so a lookback of 60 is about 20 days. A window that would cross a settlement the grid
# expects and the data does not have is dropped rather than imputed
# (`exclude_windows_crossing_missing_expected_periods`), so `eligible_rows` in the contracts table
# below, not the panel height, is the sample the model is fitted on. Training runs 100 epochs with
# a checkpoint every 5, and each of the resulting 20 checkpoints is registered as its own
# prediction identity.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Explain what causal padding is for, and what a convolutional sequence model would be measuring
#   without it.
# - Compute the receptive field of a dilated stack and check it against the declared lookback,
#   rather than assuming the two agree.
# - State how a convolutional model's time aggregation differs from a recurrent model's, and why
#   that is a hypothesis about the data rather than an implementation choice.
# - Read a resolved request and say what will be fitted, on how many eligible rows, before any
#   fitting happens.
#
# **Book reference:** Chapter 19, convolutional sequence models.
#
# **Prerequisites:** [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices, and
# [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds. The canonical run
# uses CUDA; the reduced run in CI does not.
#
# **What it writes:** one training run per configuration and one complete validation prediction set
# per checkpoint, grouped under a named population that [`13_backtest`](13_backtest.ipynb) reads.
# **Selection happens there, on validation backtest Sharpe.** Nothing here ranks anything.

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
# Nothing is fitted below. The catalog is filtered to `config_prefix="tcn"`, which is what confines
# this notebook to the convolutional configurations declared in
# `config/training/fwd_ret_8h.yaml` alongside the two that
# [`09_dl_lstm`](09_dl_lstm.ipynb) fits.
#
# The contracts table reads `gap_policy` and `lookback` back out of the frozen specification rather
# than restating the configuration file, so it cannot describe something other than what the fit
# will use. Check the lookback against the receptive field computed in the header before running
# anything: if a future edit shortens the dilation schedule, the two stop agreeing and the window
# grows a region the model cannot see.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
official_population = (
    freeze_official_model_population(study, supersedes=SUPERSEDES_POPULATION or None)
    if EXECUTION_TIER == "canonical"
    else None
)
requests = model_request_catalog("deep_learning", labels=LABELS, config_prefix="tcn")
requests

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
# Each configuration is fitted on each fold, a checkpoint is persisted every fifth epoch, and one
# complete validation prediction set is registered per checkpoint. The completeness check is not a
# formality: a prediction set covering most of its fold's eligible keys is a different sample, not
# a slightly worse result, and comparing it against a complete one in the backtest would be
# comparing two models measured on different data. The run raises rather than publishing one.

# %% tags=["results"]
execution = run_model_plan(
    plan,
    supersedes=SUPERSEDES_MODEL_POPULATION or None,
    population_name="crypto-tcn-validation-predictions-v1"
    if EXECUTION_TIER == "canonical"
    else None,
)
catalog = execution.catalog_rows.sort("label", "config_name", "checkpoint_value")
if (
    catalog.height != len(plan.expected_prediction_hashes)
    or catalog.filter(~pl.col("complete")).height
):
    raise RuntimeError("TCN checkpoint population is incomplete")
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
# - **The receptive field is a property of the architecture, not of the lookback.** Four blocks at
#   dilations 1, 2, 4, 8 with kernel 3 reach 61 settlements; the lookback is 60. Change either
#   without checking the other and the model quietly stops seeing part of the window it is handed.
# - **Causal padding is what makes the number honest.** Without trimming the right-hand padding,
#   each position would be computed partly from later ones, and the validation score would be
#   measuring a model that had seen the answer.
# - **Averaging over positions is a hypothesis.** This TCN pools its representation across the whole
#   window, so it treats a pattern early in the window as no less usable than one at the end. The
#   LSTM in [`09_dl_lstm`](09_dl_lstm.ipynb) does the opposite. Which is right is an empirical
#   question about where in the window the premium's information sits, and the backtest is where it
#   gets answered.
# - **Batch normalization pools across windows, not across time within one.** The statistics used to
#   normalize a training window come from the other windows in its batch, which may be
#   chronologically later within the same fold. Fold boundaries are respected, so no validation
#   information reaches training - but the training objective is not a pure per-window causal
#   function, and that is worth knowing before attributing a result entirely to the convolutions.
# - **Two folds is what the history supports.** The reach of the dilation schedule is not the
#   binding constraint on what this model can learn here; the length of the usable perpetual
#   funding record is.
