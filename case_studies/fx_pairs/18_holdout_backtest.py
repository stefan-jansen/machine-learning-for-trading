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
# specification. What does change is what the specification points at - the prediction set, and
# the price frame identity, which names the holdout frame the run actually reads.
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
from case_studies.research.comparison import CandidateSet
from case_studies.research.holdout import build_holdout_training_spec
from case_studies.research.strategy import strategy_warmup_periods
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_presets import ensure_backtest_spec
from case_studies.utils.backtest_runner import resolved_allow_short_selling, run_backtest
from case_studies.utils.conformal import (
    compute_holdout_conformal_widths,
    ensure_conformal_calibration_identity,
    holdout_conformal_embargo_steps,
)
from case_studies.utils.registry import (
    backtest_run_status,
    load_backtest_metrics,
    read_predictions,
)
from case_studies.utils.registry.specs import training_hash_from_spec
from case_studies.utils.strategy_analysis import resolve_solvent_carrier
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
CANDIDATE_SET_NAME = "fx_pairs:holdout-candidates"

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

# Same admission contract `17_holdout_predictions` enforces, checked again rather than inherited.
# This notebook re-resolves the carrier, so it can reach a different one, and the frozen set is
# what says which carriers the holdout may run at all.
holdout_candidates = CandidateSet.one(study, name=CANDIDATE_SET_NAME)
if carrier["val_backtest_hash"] not in holdout_candidates.members:
    raise RuntimeError(
        f"the resolved carrier {carrier['val_backtest_hash']} is not a member of "
        f"{CANDIDATE_SET_NAME} ({holdout_candidates.hash}); the holdout may only replay a "
        "configuration the frozen set admitted"
    )

validation_prediction = study.results.open(carrier["val_prediction_hash"])
validation_record = validation_prediction.registry_record()

# The holdout prediction is resolved by DERIVING the holdout training identity here and querying
# for it exactly, not by matching the carrier's family, configuration name and label. Those names
# do not identify a training specification: an earlier refit against different folds, features or
# CV geometry carries the same three names, and when it is the only row present it is selected
# silently. Deriving the specification the same way `17` does and hashing it means this notebook
# accepts the run `17` would produce and nothing else.
observation_timeline = (
    pl.read_parquet(study.root / "labels" / f"{LABEL}.parquet")
    .get_column("timestamp")
    .unique()
    .sort()
    .to_list()
)
holdout_training_spec = build_holdout_training_spec(
    study,
    study.results.open(carrier["training_hash"]).spec(),
    timeline=observation_timeline,
    case_study=CASE_STUDY_ID,
)
EXPECTED_TRAINING_HASH = training_hash_from_spec(holdout_training_spec)

with sqlite3.connect(str(CASE_DIR / "run_log" / "registry.db")) as _conn:
    _rows = _conn.execute(
        """
        SELECT prediction_hash
        FROM prediction_sets
        WHERE split = 'holdout'
          AND training_hash = ?
          AND checkpoint_kind IS ?
          AND checkpoint_value IS ?
        ORDER BY prediction_hash
        """,
        (
            EXPECTED_TRAINING_HASH,
            validation_record["checkpoint_kind"],
            validation_record["checkpoint_value"],
        ),
    ).fetchall()

if not _rows:
    raise RuntimeError(
        f"no holdout prediction set is registered for training run {EXPECTED_TRAINING_HASH} "
        f"at checkpoint {validation_record['checkpoint_kind']}="
        f"{validation_record['checkpoint_value']}; run 17_holdout_predictions first"
    )
if len(_rows) > 1:
    raise RuntimeError(
        f"training run {EXPECTED_TRAINING_HASH} registered {len(_rows)} holdout prediction sets "
        f"at one checkpoint: {sorted(row[0] for row in _rows)}. One checkpoint has one "
        "prediction generation; delete the superseded one rather than choosing between them here."
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
#
# The price frame is loaded with the strategy's own warmup, read from the specification rather
# than assumed. `run_backtest` does not trim the lower bound for exactly this reason - it names
# `mvo_ledoit_wolf`, which is this carrier's allocator, among the rolling estimators whose
# callers are expected to load a prefix - and it slices the returns frame to the window
# afterwards, so the warmup rows extend what the allocator can see without extending what is
# measured.
#
# On this carrier they change nothing: the holdout return series is bit-identical with and
# without the 63-observation prefix, so the number below does not depend on this. It is loaded
# anyway because the specification asks for it, and a run that silently ignores a declared
# warmup is only correct by accident.

# %% tags=["results"]
carrier_spec = json.loads(carrier["spec_json"])
WARMUP_PERIODS = strategy_warmup_periods(carrier_spec)
prices = load_backtest_prices_for(
    CASE_STUDY_ID, LABEL, split="holdout", warmup_periods=WARMUP_PERIODS
)
predictions = read_predictions(CASE_STUDY_ID, HOLDOUT_PREDICTION_HASH)
print(f"Prices: {len(prices):,} rows, {prices['symbol'].n_unique()} pairs")
print(f"  warmup: {WARMUP_PERIODS} observation(s) before the holdout opens")
print(f"Predictions: {predictions.height:,} rows, {predictions['timestamp'].n_unique()} sessions")

spec = ensure_backtest_spec(
    CASE_STUDY_ID,
    bt_config,
    carrier_spec,
    prices=prices,
    prediction_hash=HOLDOUT_PREDICTION_HASH,
    initial_cash=bt_config.initial_cash,
)
spec["chapter"] = "ch20"
# `ensure_backtest_spec` returns an already-canonical specification untouched apart from the
# prediction hash, so the carrier's `input_identity.prices` survives into the holdout run and
# the registered lineage then names the validation price frame for a run that never read it.
# The digest is of the frame this notebook actually passes to the engine.
spec.setdefault("input_identity", {})["prices"] = value_digest(prices)

# fx's carrier allocates by `mvo_ledoit_wolf`, so this does not fire today. It is here because
# selection can move: a `conformal_weighted` carrier is sized by widths calibrated from the
# validation residuals, and with no widths beside the holdout prediction the runner generates
# them from the holdout's own outcomes - which is the holdout deciding its own position sizes.
allocation = spec.get("strategy", {}).get("allocation", {})
NEEDS_CALIBRATION = allocation.get("method") == "conformal_weighted"
if NEEDS_CALIBRATION:
    embargo_steps = holdout_conformal_embargo_steps(CASE_STUDY_ID, LABEL)
    spec = ensure_conformal_calibration_identity(spec, holdout_embargo_steps=embargo_steps)

# The guard is on the backtest hash rather than a field comparison, because every input that
# changes the result is in that hash by construction and a guard naming fields has to be right
# about all of them. It resolves before anything runs, so nothing is evaluated on the holdout
# before the question of what is already registered is answered.
spec["backtest_config"]["account"]["allow_short_selling"] = resolved_allow_short_selling(spec, None)
prospective_hash = backtest_run_status(CASE_STUDY_ID, HOLDOUT_PREDICTION_HASH, spec).backtest_hash
with sqlite3.connect(str(CASE_DIR / "run_log" / "registry.db")) as _conn:
    _registered = {
        row[0]
        for row in _conn.execute(
            "SELECT backtest_hash FROM backtest_runs WHERE prediction_hash = ?",
            (HOLDOUT_PREDICTION_HASH,),
        ).fetchall()
    }
_superseded = sorted(_registered - {prospective_hash})
if _superseded:
    raise RuntimeError(
        "the holdout window already carries a backtest of a different configuration: "
        + ", ".join(_superseded)
        + f". This run would register {prospective_hash} and has not run. Delete the earlier "
        "one rather than leaving the window carrying two answers."
    )

if NEEDS_CALIBRATION:
    widths = compute_holdout_conformal_widths(
        CASE_STUDY_ID,
        carrier["val_prediction_hash"],
        HOLDOUT_PREDICTION_HASH,
        alpha=float(allocation.get("alpha", 0.2)),
        min_calibration_n=int(allocation["min_calibration_n"]),
        embargo_steps=embargo_steps,
        write=True,
    )
    print(f"Conformal widths: {widths.height:,} rows, embargo {embargo_steps} observation(s)")

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
if result.backtest_hash != prospective_hash:
    raise RuntimeError(
        f"the guard predicted {prospective_hash} and the runner registered "
        f"{result.backtest_hash}; a guard that has stopped predicting the hash lets "
        "everything through while looking correct"
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
# - The holdout prediction set is matched by the training identity derived here, not by the
#   carrier's family and configuration name, which several training specifications share.
# - The price frame carries the strategy's declared warmup. It makes no difference to this
#   carrier's result, which is a measurement rather than an assumption.
# - Nothing here revises the selection, which was made on validation and is already registered.
