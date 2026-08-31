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
# This notebook runs the carrier's own backtest configuration against the holdout predictions
# `17_holdout_predictions` registered. It changes nothing about the strategy: signal, allocation,
# risk controls, rebalance rule, costs and account configuration are the carrier's registered
# specification with only the prediction set re-pointed.
#
# Keeping this separate from the refit is what makes the holdout diagnosable. While the two ran
# as one transaction, a failure in the backtest half discarded the model half with it, and the
# retrain needed to investigate was the exact thing the transaction forbade.
#
# **Learning objectives**
#
# - Replay one registered strategy specification against a new prediction set.
# - Distinguish reproducing a strategy from re-choosing one.
# - Read a holdout result without letting it revise anything upstream.
#
# **Book reference**: Chapters 16-20
#
# **Prerequisite**: `17_holdout_predictions`.

# %%
"""Run the carrier's registered strategy against its holdout predictions."""

import json
import sqlite3

import polars as pl

from case_studies.research import open_study
from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_presets import ensure_backtest_spec
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.registry import load_backtest_metrics, read_predictions
from case_studies.utils.strategy_analysis import resolve_solvent_carrier
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""

# %% [markdown]
# ## Resolve the same carrier, and require its holdout predictions
#
# The carrier is re-resolved here rather than handed over from the previous notebook. The
# resolution is a query against registered validation backtests, so it returns the same
# configuration; passing it through a file or a parameter would add a way for the two notebooks
# to disagree without either being wrong about anything it did itself.
#
# What this notebook cannot derive is whether the refit has run. When it has not, the missing
# result is named rather than reported as an empty query.

# %% tags=["results"]
study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)

carrier = resolve_solvent_carrier(CASE_STUDY_ID)
LABEL = str(carrier["label"])
validation_record = study.results.open(carrier["val_prediction_hash"]).registry_record()

# The holdout prediction is matched on the carrier's CONFIGURATION rather than on its training
# hash, because a genuine retrain never shares the validation model's training identity - that
# is what makes it a retrain. Family, configuration, label and checkpoint are what carry across.
with sqlite3.connect(str(CASE_DIR / "run_log" / "registry.db")) as _conn:
    _rows = _conn.execute(
        """
        SELECT p.prediction_hash
        FROM prediction_sets p
        JOIN training_runs t ON t.training_hash = p.training_hash
        WHERE p.split = 'holdout'
          AND p.checkpoint_kind IS ?
          AND p.checkpoint_value IS ?
          AND t.family = ?
          AND t.config_name = ?
          AND t.label = ?
        ORDER BY p.prediction_hash
        """,
        (
            validation_record["checkpoint_kind"],
            validation_record["checkpoint_value"],
            carrier["family"],
            carrier["config_name"],
            LABEL,
        ),
    ).fetchall()

if not _rows:
    raise RuntimeError(
        f"no holdout prediction set is registered for {carrier['family']}/"
        f"{carrier['config_name']} on {LABEL} at checkpoint "
        f"{validation_record['checkpoint_kind']}={validation_record['checkpoint_value']}; "
        "run 17_holdout_predictions first"
    )
if len(_rows) > 1:
    raise RuntimeError(
        f"the carrier's configuration resolves {len(_rows)} holdout prediction sets: "
        f"{sorted(row[0] for row in _rows)}. One configuration has one holdout generation; "
        "delete the superseded one rather than choosing between them here."
    )
HOLDOUT_PREDICTION_HASH = _rows[0][0]

pl.DataFrame(
    {
        "field": [
            "carrier backtest",
            "carrier stage",
            "family",
            "configuration",
            "label",
            "validation prediction",
            "holdout prediction",
        ],
        "value": [
            str(carrier["val_backtest_hash"]),
            str(carrier["val_stage"]),
            str(carrier["family"]),
            str(carrier["config_name"]),
            LABEL,
            str(carrier["val_prediction_hash"]),
            HOLDOUT_PREDICTION_HASH,
        ],
    }
)

# %% [markdown]
# ## Run the carrier's configuration on the holdout
#
# The specification is the carrier's own registered one with the prediction set re-pointed and
# the chapter re-tagged. Nothing else is rebuilt from this notebook's defaults, because a
# holdout number is only comparable to its validation number when the only difference between
# the two runs is the interval they cover.

# %% tags=["results"]
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="holdout")
predictions = read_predictions(CASE_STUDY_ID, HOLDOUT_PREDICTION_HASH)
print(f"Prices: {len(prices):,} rows, {prices['symbol'].n_unique()} pairs")
print(f"Predictions: {predictions.height:,} rows, {predictions['timestamp'].n_unique()} sessions")

spec = ensure_backtest_spec(
    CASE_STUDY_ID,
    bt_config,
    json.loads(carrier["spec_json"]),
    prices=prices,
    prediction_hash=HOLDOUT_PREDICTION_HASH,
    initial_cash=bt_config.initial_cash,
)
spec["chapter"] = "ch20"

result = run_backtest(
    CASE_STUDY_ID,
    HOLDOUT_PREDICTION_HASH,
    spec,
    prices=prices,
    predictions=predictions,
    label=LABEL,
    register=True,
    initial_cash=bt_config.initial_cash,
    calendar=bt_config.calendar,
)
print(f"Holdout backtest: {result.backtest_hash}")

# %% [markdown]
# ## Validation and holdout, side by side
#
# Both rows are printed without interpretation. Whether the difference between them is evidence
# of anything is `19_strategy_analysis`'s question, and answering it needs the candidate
# distribution and the interval evidence - neither of which belongs in a notebook whose job is
# to produce one result.

# %% tags=["results"]
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
        _metrics(str(carrier["val_backtest_hash"]), "validation"),
        _metrics(result.backtest_hash, "holdout"),
    ]
)

# %% [markdown]
# ## Key takeaways
#
# - The configuration run here is the carrier's registered specification, not a strategy rebuilt
#   from this notebook's own defaults.
# - The holdout prediction set is matched by configuration, because a genuine retrain does not
#   share the validation model's training identity.
# - Nothing here revises the selection, which was made on validation and is already registered.
