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
# # CME Futures: Model Analysis
#
# This notebook reads the complete canonical model population produced by `06_linear` through
# `10b_stochastic_discount_factor`. Each row retains family, configuration, label, checkpoint, fold
# contract, training identity, and prediction identity. No null label is assigned to another
# horizon, and checkpoints are not collapsed into a single model label.
#
# IC measures whether a configuration ranks the cross-section correctly on a decision date. It
# selects nothing: every row here proceeds to the equal-weight validation backtest in
# `13_backtest`, where Sharpe performs selection and the checkpoint is part of what is selected.
#
# What `ic_mean` and `ic_t` are, precisely, because the two readings are easy to confuse. Both are
# computed **across folds**: `ic_mean` averages each fold's cross-sectional IC, and `ic_t` divides
# that by the *standard error* of the same five numbers, which is their dispersion divided by the
# square root of how many are defined - not the dispersion itself. `ic_std` in the table below is
# the dispersion, so reproducing `ic_t` from the two columns needs the `sqrt(n_folds_ic)` factor.
#
# The registry also computes the daily-series reading with its HAC standard error, which is the
# inferential statistic, but the predictions reader does not surface it, so it is not in the table
# below. `ic_n_days` is carried instead: it
# counts the validation dates that produced a defined IC, and a configuration whose predictions
# collapse to near-constant on some dates has its `ic_mean` measured over fewer of them. Reading
# `ic_mean` without `ic_n_days` is how a partial-coverage artifact reads as a leader.

# %%
"""Analyze complete CME futures model and causal result catalogs."""

import json

import polars as pl

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    CASE_STUDY,
    MODEL_POPULATION_NAMES,
    official_prediction_catalog,
    open_study,
    product_universe_table,
)
from case_studies.research import CausalResult, require_declared_menu_coverage

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None

# %% [markdown]
# ## Complete prediction catalog
#
# The six official population snapshots were created before their model runs. Opening all six and
# calling `require_complete` means a failed configuration or checkpoint cannot disappear from this
# analysis because another row happened to finish.

# %%
if EXECUTION_TIER == "preview" and WORKSPACE is None:
    raise ValueError("preview execution requires WORKSPACE")
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
universe = product_universe_table()
universe

# %% [markdown]
# Canonical analysis reads the six published population snapshots, which is the whole point of
# freezing them before their runs. A preview has no published population to read: it is a reduced
# re-execution whose rows exist only in its own workspace and which is deliberately excluded from
# every official population. It reads its own complete validation predictions instead, and the
# comparison against the declared menus below is skipped with them, because a preview fits a named
# subset by design and would fail that comparison on every configuration it left out.

# %%
if EXECUTION_TIER == "canonical":
    catalog = official_prediction_catalog(study, MODEL_POPULATION_NAMES)
else:
    catalog = (
        study.predictions.table(include_preview=True)
        .filter(
            (pl.col("execution_tier") == "preview")
            & (pl.col("split") == "validation")
            & pl.col("complete")
        )
        .sort("label", "family", "config_name", "checkpoint_kind", "checkpoint_value")
    )
    if catalog.is_empty():
        raise RuntimeError("preview execution registered no complete validation predictions")


def _feature_count(spec_json: str) -> int:
    spec = json.loads(spec_json)
    computation = spec.get("computation", spec)
    return len(computation.get("feature_names") or [])


analysis = catalog.with_columns(
    pl.col("spec_json").map_elements(_feature_count, return_dtype=pl.Int64).alias("feature_count")
).select(
    "family",
    "config_name",
    "label",
    "checkpoint_kind",
    "checkpoint_value",
    "feature_count",
    "ic_mean",
    "ic_t",
    "ic_n_days",
    "n_folds",
    "training_hash",
    "prediction_hash",
)

# %% [markdown]
# ### Every declared model is here
#
# Each execution notebook checks that it produced everything **it** requested, so none of them can
# see a configuration that no notebook requests at all - a menu entry nobody claimed publishes
# nothing and every completeness check still passes. This is the one place the families reassemble,
# so it is the only place that check can be made. `require_declared_menu_coverage` compares
# `(family, label, config_name)` against the training menus and raises on either direction: a
# declared model the population omits, or a model in the population that no menu declares.
#
# It returns the rows knowingly excluded, so what this notebook is missing is displayed rather than
# taken on trust. `causal_dml` is not in the comparison - it is not a predictive family and the
# adapter registry, not a list here, is what decides that.

# %%
if EXECUTION_TIER == "canonical":
    excluded = require_declared_menu_coverage(analysis, case_study=CASE_STUDY)
else:
    excluded = analysis.clear()
excluded

# %% tags=["results"]
analysis.sort("label", "family", "config_name", "checkpoint_value")

# %% [markdown]
# ## Interpretation boundaries
#
# The table compares ranking diagnostics under the declared walk-forward protocol. The backtest
# engine supplies portfolio returns, transaction costs, contract sizing, and roll execution before
# selection.
#
# Conformal weighting, when used later as an allocator, calibrates chronologically from prior
# validation observations. It uses the calibration-window scale and the finite-sample higher order
# statistic. It does not fit a scale on the evaluation fold or pool all folds before calibration.

# %% [markdown]
# ## Causal diagnostics
#
# Double machine learning answers a different question from prediction. The treatment effect is
# conditioned on the configured confounders, and HAC uncertainty follows the decision-time order.
# A covariance-estimator failure is not relabeled as HAC. The shared runner must return a finite HAC
# standard error for a result to be complete.
#
# **The `refutation_p` column below is not evidence that these effects survived a placebo test.**
# The refutation permutes contiguous blocks within each product, and the shared runner sizes those
# blocks from the label buffer rather than from the treatment: `block_size` is set equal to
# `embargo`, so the registered rows carry a 21-period block for `fwd_ret_21d` and a 5-period block
# for `fwd_ret_5d`. Neither length is a property of `carry_pct`. Measured on this case study's own
# feature panel, `carry_pct` has a lag-1 autocorrelation of 0.943, an AR(1) half-life of 11.8
# trading days, and autocorrelation still at 0.44 by lag 21 and 0.17 by lag 63. Blocks of 5 and 21
# periods therefore destroy serial dependence that the real treatment has. That narrows the placebo
# distribution relative to the true null and pushes the empirical p-value toward zero whether or not
# the effect is real, which is what both labels report. Read the DML point estimate and its HAC
# standard error. The refutation column is recorded for completeness and carries no evidence here.

# %%
causal_rows = []
for label in ALL_LABELS:
    result = CausalResult.one(study, label=label, execution_tier=EXECUTION_TIER)
    if not result.complete:
        raise RuntimeError(f"causal result for {label} is incomplete")
    causal_rows.append({"label": label, "causal_hash": result.hash, **result.metrics})
causal = pl.DataFrame(causal_rows).sort("label")

# %% tags=["results"]
causal

# %% [markdown]
# ## What proceeds to backtesting
#
# All complete prediction rows proceed. The next notebook passes the selected Polars rows directly
# to the shared backtest call, publishes product-keyed decisions, and records contract, roll, price,
# and prediction lineage. Validation backtest Sharpe, with the prediction checkpoint included in the
# configuration identity, is the selection statistic.
