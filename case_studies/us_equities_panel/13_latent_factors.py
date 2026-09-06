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
# # US equities panel: finding the few things three thousand stocks have in common
#
# Every model so far has predicted each stock from that stock's own features. But stocks do not
# move independently - most of what a broad panel does on any day is one thing happening to all of
# it, and a handful of further things happening to overlapping groups of it. A **latent factor**
# is one of those common movements: not a column anybody computed, but a pattern extracted from
# how the returns move together, with each stock carrying a **loading** saying how much of that
# pattern it takes.
#
# Two ways of extracting them are fitted here, and the difference between them is the whole
# lesson:
#
# - [`13a_pca`](13a_pca.ipynb) takes the factors from the return panel alone. Principal component
#   analysis asks which combinations of stocks account for the most common variation, and answers
#   without being told anything about the stocks. A loading is then a number attached to a stock,
#   fitted over the training window and carried forward.
# - [`13b_ipca`](13b_ipca.ipynb) conditions the loadings on what the stocks *are*. Instrumented
#   principal components makes a stock's loading a function of its observable characteristics, so
#   two stocks with the same characteristics load the same way and a stock whose characteristics
#   change has its loading change with them.
#
# **Why the second exists.** A loading attached to a stock says nothing about a stock that has not
# been seen, and cannot move when the stock does. On a panel where names enter and leave and a
# company's size and value change over a decade, that is a real limitation rather than a technical
# one, and conditioning on characteristics is what removes it. What it costs is a stronger
# assumption: that the relation between characteristics and loadings is stable, and is the same
# for every stock.
#
# **This notebook runs nothing.** It is the index over the two that do: it names them, opens what
# they published, and shows that both are complete. Which of the two is worth more is a predictive
# question, and it is answered in [`15_model_analysis`](15_model_analysis.ipynb).
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Say what a latent factor and a loading are, in terms of a panel of returns rather than of an
#   algorithm.
# - State the difference between a loading attached to a stock and a loading conditioned on the
#   stock's characteristics, and name a situation in which only the second can answer.
# - Say what the conditioned version assumes in exchange, and when that assumption would be
#   uncomfortable.
# - Read a table of published latent-factor results and tell a complete one from an incomplete one.
#
# **Book reference**: Chapter 13.
#
# **Prerequisites**: [`13a_pca`](13a_pca.ipynb) and [`13b_ipca`](13b_ipca.ipynb) have published the
# results this index reads.
#
# **What it writes**: nothing. It reads.

# %%
"""Reference index for the latent-factor execution notebooks."""

import os
from pathlib import Path

import polars as pl

from case_studies.research import Study, open_study

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"

# %% [markdown]
# ## What the two notebooks published
#
# One row per label per factor model. Read it for two things.
#
# **Both models present at every label.** A label carrying a PCA row and no IPCA row means the
# second notebook did not finish there, and the comparison in
# [`15_model_analysis`](15_model_analysis.ipynb) would then be measuring a difference between
# labels rather than between factor models.
#
# **`cv_identity` the same across the rows being compared.** It records which walk-forward design
# a result was fitted and scored under. Two rows with different values measured themselves over
# different windows, and ranking them is not a comparison.
#
# Canonical execution reads the released study; preview execution reads an isolated workspace.

# %%
if EXECUTION_TIER == "canonical":
    # `Study.open` with no workspace, not `open_study`: this notebook writes nothing, and that is
    # the call that opens the released study read-only. `open_study` opens it for regeneration.
    study = Study.open(CASE_STUDY_ID)
elif EXECUTION_TIER == "preview":
    study = open_study(
        CASE_STUDY_ID,
        execution_tier=EXECUTION_TIER,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
    )
else:
    raise ValueError(f"Unsupported execution tier: {EXECUTION_TIER!r}")

latent_results = (
    study.predictions.table(include_preview=EXECUTION_TIER == "preview")
    .filter(
        (pl.col("family") == "latent_factors")
        & (pl.col("split") == "validation")
        & (pl.col("execution_tier") == EXECUTION_TIER)
        & pl.col("complete")
    )
    .select(
        "label",
        "config_name",
        "checkpoint_kind",
        "checkpoint_value",
        "cv_identity",
        "training_hash",
        "prediction_hash",
    )
    .sort("label", "config_name", "checkpoint_kind", "checkpoint_value")
)
latent_results

# %% [markdown]
# ## What happens next
#
# If a row is missing above, run the notebook that produces it - the execution notebooks reuse an
# identity that already exists rather than refitting it, so re-running is cheap and safe.
#
# [`15_model_analysis`](15_model_analysis.ipynb) reads these results alongside the other model
# families and asks which ranks the cross-section better.
# [`16_backtest`](16_backtest.ipynb) backtests every one of them. This index chooses nothing, and
# the number of factors is not tuned anywhere in this case study: each model's preset declares one
# and a sweep over that count would be a different experiment.

# %% [markdown]
# ## What to notice
#
# **The two models answer the same question with different information.** PCA sees only how the
# returns moved together; IPCA is additionally told what each stock is. Any difference between
# them is what the characteristics were worth, on this panel, under the assumption that the
# relation between characteristics and loadings holds across stocks and over time.
#
# **A factor model is a compression, and a compression discards.** A handful of factors summarise
# a three-thousand-name panel, so whatever is specific to one stock is by construction not in the
# prediction. That is the trade being made rather than a defect: the models before this one are
# where stock-specific information lives.
#
# **Known limitations.** The factor count is declared, not searched, so nothing here says the
# declared one is right - only what it gives. Both models are fitted on training windows only and
# scored on validation folds that have been read many times over by the time a case study reaches
# this notebook. And a latent factor has no name: it is a direction in the returns, and reading an
# economic story into it is an interpretation this notebook does not support.
