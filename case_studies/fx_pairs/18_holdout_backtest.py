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
# # Holdout Backtest - FX Pairs
#
# This notebook replays the selected validation strategy against the holdout predictions
# `17_holdout_predictions` registered. It changes nothing about the strategy: signal, allocation,
# risk controls, rebalance rule, costs and account configuration are the resolved specification of
# the selected validation backtest, and the replay refuses if reconstructing them changes any
# field the comparison depends on.
#
# Keeping this separate from the refit is what makes the holdout diagnosable. When the two ran as
# one transaction, a failure in the backtest half discarded the model half with it, and the
# retrain needed to investigate was the exact thing the transaction forbade.
#
# **Learning objectives**
#
# - Replay one resolved strategy specification against a new prediction set.
# - Distinguish reproducing a strategy from re-choosing one.
# - Read a holdout result without letting it revise anything upstream.
#
# **Book reference**: Chapters 16-20
#
# **Prerequisite**: `17_holdout_predictions`.

# %%
"""Replay the selected FX strategy against its holdout predictions."""

import polars as pl

from case_studies.research import open_study
from case_studies.research.holdout import resolve_holdout_selection
from case_studies.utils.registry import load_backtest_metrics

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
CANDIDATE_SET_NAME = "fx_pairs:holdout-candidates"

# %% [markdown]
# ## Resolve the same selection, and require its holdout predictions
#
# The selection is re-derived here rather than handed over from the previous notebook. The
# derivation is pure and the candidate set is immutable, so it resolves the same configuration;
# passing it through a file or a parameter would add a way for the two notebooks to disagree
# without either of them being wrong about anything it did itself.
#
# What this notebook cannot derive is whether the refit has run. When it has not, the missing
# result is named rather than reported as an empty query.

# %% tags=["results"]
study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
selection = resolve_holdout_selection(study, candidate_set_name=CANDIDATE_SET_NAME)

holdout_prediction = selection.holdout_prediction
if holdout_prediction is None:
    raise RuntimeError(
        f"no holdout prediction is registered for training {selection.holdout_training_hash} "
        f"at checkpoint {selection.checkpoint_kind}={selection.checkpoint_value}; "
        "run 17_holdout_predictions first"
    )
if not holdout_prediction.complete:
    raise RuntimeError(f"holdout prediction {holdout_prediction.hash} is incomplete")

pl.DataFrame(
    {
        "field": [
            "candidate set",
            "selected validation backtest",
            "validation prediction",
            "holdout training",
            "holdout prediction",
            "label",
        ],
        "value": [
            selection.candidate_set.hash,
            selection.validation_backtest.hash,
            selection.validation_prediction.hash,
            selection.holdout_training_hash,
            holdout_prediction.hash,
            selection.label,
        ],
    }
)

# %% [markdown]
# ## Replay the strategy on the holdout
#
# The replay reconstructs the strategy from the selected validation backtest's own specification
# and refuses if the reconstruction changes any field the comparison rests on. That check is the
# point of the notebook: a holdout number is only comparable to its validation number when the
# only difference between the two runs is the interval they were run over.
#
# Re-running is safe. The backtest identity is a function of the prediction and the resolved
# specification, so a second run resolves the registered result and returns it rather than
# writing a second one.

# %% tags=["results"]
existing = selection.holdout_backtest
if existing is not None:
    holdout_backtest = existing
    print(f"Holdout backtest {holdout_backtest.hash} already registered; replay not re-run.")
else:
    holdout_backtest = selection.strategy_replay().run(holdout_prediction)
    print(f"Holdout backtest {holdout_backtest.hash} produced.")

record = holdout_backtest.registry_record()
if record["prediction_hash"] != holdout_prediction.hash:
    raise RuntimeError("the holdout backtest does not reference the holdout prediction")
if not holdout_backtest.complete:
    raise RuntimeError("the holdout backtest is incomplete")
# The split is a property of the prediction the backtest ran on, so it is read from the catalog
# view rather than from the `backtest_runs` row, which has no split column of its own.
_catalog = study.backtests.table().filter(pl.col("backtest_hash") == holdout_backtest.hash)
if _catalog.height != 1:
    raise RuntimeError(f"backtest {holdout_backtest.hash} has {_catalog.height} catalog rows")
_split = _catalog.get_column("split").item()
if _split != "holdout":
    raise RuntimeError(f"the replay published a {_split!r} backtest")

# %% [markdown]
# ## Validation and holdout, side by side
#
# Both rows are printed here without interpretation. Whether the difference between them is
# evidence of anything is `19_strategy_analysis`'s question, and it needs the candidate
# distribution and the interval evidence to answer it - neither of which belongs in a notebook
# whose job is to produce one result.

# %% tags=["results"]


# The registry's own names. "annual_return" and "annual_volatility" are what the quantities are
# called in prose and neither exists as a column, so asking for them returns nulls that read as a
# strategy with no return rather than as a query that missed.
REPORTED = ("sharpe", "cagr", "volatility", "max_drawdown")


def _metrics(backtest_hash: str, split: str) -> dict[str, object]:
    metrics = load_backtest_metrics(CASE_STUDY_ID, backtest_hash=backtest_hash, case_dir=study.root)
    if metrics.height != 1:
        raise ValueError(f"backtest {backtest_hash} has {metrics.height} metric rows")
    row = metrics.row(0, named=True)
    return {
        "split": split,
        "backtest_hash": backtest_hash,
        **{name: row.get(name) for name in REPORTED},
    }


pl.DataFrame(
    [
        _metrics(selection.validation_backtest.hash, "validation"),
        _metrics(holdout_backtest.hash, "holdout"),
    ]
)

# %% [markdown]
# - The strategy replayed here is the resolved specification of the selected validation backtest,
#   not a strategy rebuilt from this notebook's own defaults.
# - The holdout result is produced once and read many times; the identity, not a lock, is what
#   stops a second one from appearing.
# - Nothing here revises the selection, which was made on validation and is already frozen.
