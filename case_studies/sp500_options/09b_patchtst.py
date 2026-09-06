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
# # S&P 500 Options: PatchTST
#
# This notebook fits the declared PatchTST member of the sequence population snapshotted by
# `09_deep_learning`. After publishing every PatchTST checkpoint, it verifies that the complete
# NLinear, LSTM, and PatchTST population is present.
#
# Prerequisites: `09_deep_learning` and `09a_lstm`.
#
# **Why the population is declared in one notebook and filled by several.** The set of members is
# a claim made once, before any of them is fitted, so that no family can be added or dropped after
# its results are visible. This notebook fits the last declared member and then checks that all
# three are present, which is the point at which the population becomes readable downstream.
#
# ## What this model is, and how it differs from the LSTM beside it
#
# PatchTST cuts the lookback window into fixed-length patches, embeds each patch as a token, and
# lets attention weigh every token against every other. Where the LSTM reads the window one
# session at a time and carries a state forward, this model sees the whole window at once and
# learns which parts of it to look at.
#
# **What the difference buys.** Recurrence reaches a distant session only by carrying information
# through every session between; attention reaches it directly, so a pattern that depends on two
# separated stretches of the window is easier to represent. It also parallelizes across the
# window, where recurrence is sequential by construction.
#
# **What it costs, and why both are run.** Attention has no built-in notion that yesterday is
# nearer than last month - the ordering has to be learned from position embeddings rather than
# being structural - and it has more parameters to fit from the same data. Running both against
# the same population and the same folds is what turns "sequence models on options data" from an
# assertion into a comparison, and either can lose.

# %%
"""Fit the declared S&P 500 options PatchTST request."""

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
# **What the settings decide.** `lookback: 60` and `patch_size: 16` together set the tokens: a
# sixty-session window becomes a handful of patches rather than sixty steps, which is what makes
# attention affordable here and also what limits its resolution, since nothing inside a patch is
# distinguished. `d_model: 64` is the width each patch is embedded into and `n_heads: 4` the
# number of attention patterns learned in parallel, so the model can attend to several
# relationships at once instead of averaging them into one. `n_layers: 2` stacks that twice.
# `dropout: 0.1` is the same regularizer the LSTM uses, and deliberately so: two families whose
# regularization differs are not being compared on architecture.
#
# **The configuration is read from a preset, not written here**, so this architecture cannot
# quietly differ from the same architecture in another chapter.
#
# **Every label is fitted**, because selection downstream ranks across labels as well as across
# configurations, and a label with no candidates cannot be chosen or ruled out.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
requests = model_request_catalog(
    "deep_learning",
    labels=ALL_LABELS,
    config_names=("patchtst",),
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
# The shared sequence runner owns gap-safe window construction, fold fitting, fitted-state reload,
# checkpoint publication, restart, and exact eligible-key validation.
#
# **A checkpoint is part of a configuration, not a detail of how it was fitted.** Training runs for
# 100 epochs and publishes every fifth, so this one request becomes twenty scored candidates. A
# network's validation performance is not monotone in training time, and the epoch at which it
# peaks is a property of the fit a reader is entitled to see rather than a number chosen after the
# fact. Picking the best epoch after seeing the results is selection, and selection happens once,
# downstream, on backtests.
#
# **Gap-safe means a window never reaches across a fold boundary.** A sequence handed to the model
# has to end before the validation window opens, or its state carries information from the period
# being scored. The runner owns that construction because it is exactly the kind of rule that gets
# restated slightly differently in each notebook and is impossible to notice when it is.

# %%
if EXECUTION_TIER == "canonical":
    execution, population = run_official_model_subset(
        study,
        resolved,
        population=population_name,
        require_population_complete=True,
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
    raise RuntimeError("PatchTST execution returned a partial checkpoint")
catalog

# %% [markdown]
# The official sequence population is complete and ready for model analysis and backtesting. This
# notebook does not compare configurations or choose a checkpoint.
#
# **What completeness means here, and why it is checked rather than assumed.** Every requested
# checkpoint of all three families produced predictions on exactly the rows its eligibility
# contract declared, not more and not fewer. A partial checkpoint is refused rather than
# published: a downstream comparison against a model scored on part of the panel is not a
# comparison, and by the time anyone reads the result the missing part is invisible.
#
# **The three families share one eligibility group, and that is what makes them comparable.** All
# of them need the same sixty-session window before a symbol can be scored, so they are eligible
# on the same rows and the difference between their numbers is the models. That does not extend to
# the cross-sectional families, which score a symbol from its first row; `11_model_analysis` groups
# by eligibility for exactly that reason.
#
# **What has and has not been established at this point.** Three architectures have been fitted on
# identical folds, identical windows and identical labels, and every checkpoint each of them
# published is registered and complete. Nothing has been ranked. A reader who wants to know which
# sequence family works better on options data has the material for that question, and the answer
# comes from backtests rather than from anything on this page.
