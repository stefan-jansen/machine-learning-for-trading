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
# # S&P 500 Options: LSTM
#
# This notebook fits the declared LSTM member of the sequence population snapshotted by
# `09_deep_learning`. Chronological windows, validation gaps, checkpoints, and prediction
# eligibility are resolved through the shared sequence boundary.
#
# Prerequisite: `09_deep_learning` must create the complete official sequence population.
#
# **Why the population is declared in one notebook and filled by several.** The set of members is
# a claim made once, before any of them is fitted, so that no family can be added or dropped after
# its results are visible. This notebook fits one declared member into a population it did not
# define and cannot extend; running it alone leaves the population incomplete rather than smaller.
#
# ## What this model is, and what it is being asked to do here
#
# An LSTM reads a symbol's history one session at a time and carries a state forward, updating it
# at each step through gates that decide how much of the new observation to admit and how much of
# the existing state to keep. The gates are what separate it from a plain recurrent network: they
# give the model a route by which information from many steps back can reach the output without
# being multiplied away at every step, which is what makes a long lookback usable at all.
#
# **What that buys on this data, and what it costs.** The cross-sectional families in this case
# study see one row per symbol per decision time: whatever history matters has to have been
# compressed into a feature first. This model is handed the window instead and left to decide what
# in it matters, so a pattern nobody wrote a feature for is reachable. The cost is that it has far
# more freedom to fit noise, and options data on a few hundred names is not abundant, so the
# comparison against the cross-sectional families is the point of running it rather than a
# formality.
#
# **It is not expected to win, and that is worth saying before the numbers.** A sequence model
# earns its keep where the ordering of observations carries information the features do not. If
# it does not beat a gradient-boosted model on engineered features here, that is a result about
# this data, not a failed run, and the chapter reports it either way.

# %%
"""Fit the declared S&P 500 options LSTM request."""

import polars as pl

from case_studies.sp500_options.research_workflow import (
    ALL_LABELS,
    declared_dl_device,
    model_request_catalog,
    open_study,
    published_dl_device,
    resolve_model_requests,
    resolved_model_plan,
    run_official_model_subset,
    run_resolved_model_requests,
)

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
PREVIEW_REDUCTIONS: dict = {}
DEVICE: str = ""

POPULATION_NAME: str = ""

# %% [markdown]
# ### The device the population was fitted on
#
# A network trained on a GPU and the same network trained on a CPU accumulate their sums in a
# different order and reach different weights, so the device is part of what the fitted model is
# and sits inside the training identity rather than beside it. The device this population was
# fitted on is declared once, in `modeling.dl.device` in `config/setup.yaml`, and read from there
# by all four deep-learning notebooks rather than retyped in each. On a machine with no NVIDIA
# card the run stops here rather than quietly training something else: set `DEVICE="cpu"` and pass
# a `POPULATION_NAME` to fit the same requests there, under a name of their own.
#
# **Why a second name rather than a second run under the first.** The published population is a
# claim about a specific set of fitted models. A CPU fit of the same request is a different set,
# close but not identical, and letting it join the published name would make the population mean
# "these requests, fitted somewhere" instead of "these models". The check above refuses that
# combination outright rather than warning about it, because a warning in a long run is read once
# and then not read.
#
# **This is why the gradient-boosted families run on CPU and these run on GPU.** A reader without
# a card can reproduce everything the book compares on trees; the sequence families are the part
# that needs hardware, and they are separated so that the absence of a GPU costs a chapter's
# comparison rather than the whole case study.

# %%
CANONICAL_POPULATION_NAME = "sp500-options-sequence-validation-v1"

published_device = published_dl_device()
device = declared_dl_device(DEVICE)
population_name = POPULATION_NAME or CANONICAL_POPULATION_NAME
if device != published_device and population_name == CANONICAL_POPULATION_NAME:
    raise ValueError(
        f"this run fits on {device!r}, not the published {published_device!r}, so its "
        f"identities are not the ones {CANONICAL_POPULATION_NAME!r} holds; pass "
        f"POPULATION_NAME to give them a population of their own"
    )
print(f"training device: {device} (declared: {published_device})")

# %% [markdown]
# ## Declared request
#
# **What the settings decide.** `lookback: 60` is the window handed to the model: sixty sessions,
# about a quarter, so a fitted state can span an earnings cycle without reaching back to a regime
# the symbol has left. `hidden_size: 64` and `n_layers: 2` set how much the state can hold and how
# many times it is re-read before the output; larger values fit more and generalize less, and on a
# panel this size they are the first place overfitting shows. `dropout: 0.1` drops a tenth of the
# connections on each training pass, which stops the network leaning on any single one.
#
# `batch_size: 2048` is a throughput choice rather than a modelling one, but it is not neutral:
# gradient noise falls as the batch grows, so a large batch trains more smoothly and explores
# less. It is declared rather than tuned because tuning it would change what was fitted while
# looking like an infrastructure decision.
#
# **The configuration is read from a preset, not written here.** `lstm_h64` names a file under
# `case_studies/config/`, so this notebook cannot quietly differ from the same architecture in
# another chapter, and a reader comparing the two is comparing declarations rather than code.
#
# **Every label is fitted, not just the primary one.** The request spans `ALL_LABELS`, because
# selection downstream ranks across labels as well as across configurations, and a label with no
# candidates cannot be chosen or ruled out.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
requests = model_request_catalog(
    "deep_learning",
    labels=ALL_LABELS,
    config_names=("lstm_h64",),
)
resolved = resolve_model_requests(
    study,
    requests,
    execution_tier=EXECUTION_TIER,
    overrides={"device": device},
    preview_reductions=PREVIEW_REDUCTIONS,
)
resolved_model_plan(resolved)

# %% [markdown]
# ## Execute and validate
#
# The shared sequence runner owns chronological window construction, fold fitting, fitted-state
# reload, checkpoint publication, restart, and exact eligible-key validation.
#
# **A checkpoint is part of a configuration, not a detail of how it was fitted.** Training runs for
# 100 epochs and publishes every fifth, so this one request becomes twenty scored candidates rather
# than one. That is deliberate: a network's validation performance is not monotone in training
# time, and the epoch at which it peaks is a property of the fit that a reader is entitled to see
# rather than a number chosen after the fact. Each published checkpoint therefore carries its own
# identity and competes on its own downstream, and picking the best epoch after seeing the results
# is selection, which happens once, downstream, on backtests.
#
# **Restart is a correctness property, not a convenience.** Fold fits are written as they finish
# and reloaded rather than refitted, so a run interrupted after eight of ten folds resumes at the
# ninth. What matters is not the time saved: it is that the alternative - starting over - invites
# quietly reducing the job to make it fit, and a population assembled from a reduced re-run and a
# full first attempt is not one population. Reloading a fitted state means the checkpoint that
# reaches the registry is the one the schedule asked for, whatever happened to the process.
#
# **Windows are built chronologically and never span a fold boundary.** A sequence handed to the
# model has to end before the fold's validation window opens, or the state carries information
# from the period being scored. The runner owns that construction for the same reason the fold
# geometry is shared: it is the kind of rule that is easy to restate slightly differently in each
# notebook and impossible to notice when someone does.

# %%
if EXECUTION_TIER == "canonical":
    execution, population = run_official_model_subset(
        study,
        resolved,
        population=population_name,
    )
else:
    if not WORKSPACE or not PREVIEW_REDUCTIONS:
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
).sort("checkpoint_value")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("LSTM execution returned a partial checkpoint")
catalog

# %% [markdown]
# The complete LSTM checkpoint population is ready for model analysis and backtesting. This
# notebook does not compare it with another family or choose a checkpoint.
#
# **What completeness means here and why it is checked before anything leaves.** Every requested
# checkpoint produced predictions on exactly the rows its eligibility contract declared - not
# more, and not fewer. A partial checkpoint is refused rather than published, because a downstream
# comparison against a model scored on a subset of the panel is not a comparison, and the subset
# is invisible by the time anyone reads the result.
#
# **The eligible rows are fewer than the cross-sectional families see, and that is structural.**
# A symbol cannot be scored until sixty sessions of it exist, so this family is eligible on
# strictly fewer rows than a model reading one row at a time. `11_model_analysis` groups by
# eligibility for exactly this reason: comparing an IC from this population against one from a
# cross-sectional population mixes the models with the rows they were scored on.
