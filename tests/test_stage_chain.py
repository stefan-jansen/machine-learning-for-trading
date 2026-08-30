"""The backtest stage chain: risk controls are applied before costs are measured.

Each stage is benchmarked against the leader of the stage before it. These tests pin
the behaviour that follows from that - which stage a benchmark resolves to, and what
happens when a case study has not run one of the stages - rather than the contents of
the constant, which any edit to the constant would trivially satisfy.
"""

from __future__ import annotations

import sqlite3

import pytest

from case_studies.utils.cohort_metrics import _resolve_baseline_hash
from case_studies.utils.uncertainty import (
    STAGE_BASELINE,
    STAGE_CARRIER_BLOCK,
    STAGE_SEQUENCE,
    descends_from,
)

LABEL = "fwd_ret_5d"


def _db(stages: dict[str, float], with_equal_weight: bool = True) -> sqlite3.Connection:
    """Registry holding one backtest per named stage, with the given Sharpe."""
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE training_runs (training_hash TEXT, family TEXT, label TEXT);
        CREATE TABLE prediction_sets (prediction_hash TEXT, training_hash TEXT, split TEXT);
        CREATE TABLE prediction_metrics (prediction_hash TEXT, ic_n_days INTEGER);
        CREATE TABLE backtest_runs (backtest_hash TEXT, prediction_hash TEXT, stage TEXT);
        CREATE TABLE backtest_metrics (backtest_hash TEXT, sharpe REAL);
        """
    )
    db.execute("INSERT INTO training_runs VALUES ('t1', 'gbm', ?)", (LABEL,))
    db.execute("INSERT INTO prediction_sets VALUES ('p1', 't1', 'validation')")
    db.execute("INSERT INTO prediction_metrics VALUES ('p1', 500)")
    for stage, sharpe in stages.items():
        db.execute("INSERT INTO backtest_runs VALUES (?, 'p1', ?)", (f"bt_{stage}", stage))
        db.execute("INSERT INTO backtest_metrics VALUES (?, ?)", (f"bt_{stage}", sharpe))
    if with_equal_weight:
        db.execute("INSERT INTO training_runs VALUES ('t0', 'benchmark', ?)", (LABEL,))
        db.execute("INSERT INTO prediction_sets VALUES ('p0', 't0', 'validation')")
        db.execute("INSERT INTO prediction_metrics VALUES ('p0', 500)")
        db.execute("INSERT INTO backtest_runs VALUES ('bt_eqw', 'p0', 'signal')")
        db.execute("INSERT INTO backtest_metrics VALUES ('bt_eqw', 0.1)")
    return db


def test_risk_is_applied_before_costs_are_measured():
    """The sequence the backtests run: size, overlay controls, then price the winner."""
    assert STAGE_SEQUENCE.index("risk_overlay") < STAGE_SEQUENCE.index("cost_sensitivity")


def test_cost_sensitivity_is_benchmarked_against_the_risk_leader():
    db = _db({"signal": 0.5, "allocation": 1.0, "risk_overlay": 1.2, "cost_sensitivity": 0.9})
    assert _resolve_baseline_hash(db, "cost_sensitivity", LABEL) == "bt_risk_overlay"


def test_risk_overlay_is_benchmarked_against_the_allocation_leader():
    db = _db({"signal": 0.5, "allocation": 1.0, "risk_overlay": 1.2})
    assert _resolve_baseline_hash(db, "risk_overlay", LABEL) == "bt_allocation"


def test_no_stage_is_its_own_descendants_benchmark():
    """A stage is never benchmarked against a stage that comes after it."""
    for stage, kind in STAGE_BASELINE.items():
        if not kind.endswith("_leader"):
            continue
        parent = kind[: -len("_leader")]
        assert STAGE_SEQUENCE.index(parent) < STAGE_SEQUENCE.index(stage), (
            f"{stage} is benchmarked against {parent}, which runs after it"
        )


def test_a_skipped_stage_falls_back_to_the_nearest_earlier_one():
    """A case study that has not run the risk stage still gets a real benchmark.

    Before the chain was made a sequence this returned None silently, and the cost
    surface was reported against no baseline at all.
    """
    db = _db({"signal": 0.5, "allocation": 1.0, "cost_sensitivity": 0.9})
    assert _resolve_baseline_hash(db, "cost_sensitivity", LABEL) == "bt_allocation"


def test_the_fallback_ends_at_the_equal_weight_benchmark():
    db = _db({"cost_sensitivity": 0.9})
    assert _resolve_baseline_hash(db, "cost_sensitivity", LABEL) == "bt_eqw"


@pytest.mark.parametrize("stage", [s for s in STAGE_SEQUENCE if s != "signal"])
def test_every_stage_resolves_to_a_benchmark_rather_than_none(stage):
    """No stage may resolve to None: a missing baseline is silent, not loud."""
    db = _db({s: 1.0 for s in STAGE_SEQUENCE})
    assert _resolve_baseline_hash(db, stage, LABEL) is not None


# ---------------------------------------------------------------------------
# A later stage is only comparable to an earlier one it was actually built on.
# ---------------------------------------------------------------------------

_ALLOC = {"method": "risk_parity"}
_RISK = {"name": "trailing_5pct"}


def test_a_cost_run_built_on_the_risk_carrier_descends_from_it():
    risk_leader = {"allocation": _ALLOC, "risk": _RISK}
    cost_run = {"allocation": _ALLOC, "risk": _RISK, "costs": {"commission_bps": 5}}
    assert descends_from(cost_run, risk_leader, "risk_overlay")


def test_a_cost_run_that_cloned_the_allocation_carrier_does_not():
    """The etfs shape: costs and risk branch off allocation independently.

    Pairing these two would book the entire difference between two unrelated
    strategies - one overlaid, one not - as the cost of trading.
    """
    risk_leader = {"allocation": _ALLOC, "risk": _RISK}
    cost_run = {"allocation": _ALLOC, "costs": {"commission_bps": 5}}
    assert not descends_from(cost_run, risk_leader, "risk_overlay")


def test_a_risk_overlay_on_a_different_allocator_does_not_descend():
    alloc_leader = {"allocation": _ALLOC}
    other = {"allocation": {"method": "equal_weight"}, "risk": _RISK}
    assert not descends_from(other, alloc_leader, "allocation")


def test_every_stage_descends_from_signal():
    """Every stage carries a signal, so the signal stage constrains nothing."""
    for spec in ({"allocation": _ALLOC}, {"risk": _RISK}, {"costs": {}}):
        assert descends_from(spec, {"signal": {"method": "top_k"}}, "signal")


def test_the_carrier_blocks_name_real_stages():
    for stage in STAGE_CARRIER_BLOCK:
        assert stage in STAGE_SEQUENCE
