"""The holdout anchor is never chosen by its own holdout result."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils import paired_metrics


def _registry(tmp_path: Path, rows) -> Path:
    """``rows`` are (prediction_hash, training_hash, config_name, backtest_hash, sharpe)."""
    case_dir = tmp_path / "probe"
    (case_dir / "run_log").mkdir(parents=True)
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.execute(
        "CREATE TABLE prediction_sets (prediction_hash TEXT, training_hash TEXT, split TEXT, "
        "checkpoint_kind TEXT, checkpoint_value INTEGER)"
    )
    db.execute(
        "CREATE TABLE training_runs (training_hash TEXT, family TEXT, config_name TEXT, label TEXT)"
    )
    db.execute(
        "CREATE TABLE backtest_runs (backtest_hash TEXT, prediction_hash TEXT, stage TEXT, "
        "spec_json TEXT)"
    )
    db.execute("CREATE TABLE backtest_metrics (backtest_hash TEXT, sharpe REAL)")
    db.execute(
        "CREATE TABLE research_locks (lock_hash TEXT, lock_json TEXT, state TEXT, created_at TEXT)"
    )
    for prediction_hash, training_hash, config_name, backtest_hash, sharpe in rows:
        db.execute(
            "INSERT INTO prediction_sets VALUES (?,?,?,?,?)",
            (prediction_hash, training_hash, "holdout", "epoch", 50),
        )
        db.execute(
            "INSERT INTO training_runs VALUES (?,?,?,?)",
            (training_hash, "gbm", config_name, "fwd_ret_21d"),
        )
        db.execute(
            "INSERT INTO backtest_runs VALUES (?,?,?,?)",
            (backtest_hash, prediction_hash, "holdout", "{}"),
        )
        db.execute("INSERT INTO backtest_metrics VALUES (?,?)", (backtest_hash, sharpe))
    db.commit()
    db.close()
    return case_dir


@pytest.fixture(autouse=True)
def _clear_cache():
    paired_metrics._retired_prediction_hashes.cache_clear()
    yield
    paired_metrics._retired_prediction_hashes.cache_clear()


def _install(monkeypatch, case_dir: Path) -> None:
    monkeypatch.setattr(paired_metrics, "get_case_study_dir", lambda cs: case_dir)
    monkeypatch.setattr(
        "case_studies.research.population.superseded_members_at",
        lambda _dir, member_kind="prediction": frozenset(),
    )


def _lineage(cs="probe"):
    return paired_metrics._holdout_lineage_for(
        cs, "fwd_ret_21d", None, label_restriction=None, rung=None
    )


def test_one_lineage_resolves(monkeypatch, tmp_path) -> None:
    case_dir = _registry(tmp_path, [("p1", "t1", "cfg", "b1", 0.4)])
    _install(monkeypatch, case_dir)

    assert _lineage()["backtest_hash"] == "b1"


def test_several_trained_models_refuse_rather_than_rank(monkeypatch, tmp_path) -> None:
    """Nothing records which validation carrier each retrain came from, so the higher
    Sharpe is not evidence - it is how a holdout from a retired carrier would win."""
    case_dir = _registry(
        tmp_path,
        [
            ("p1", "t1", "cfg_a", "b1", 0.4),
            ("p2", "t2", "cfg_b", "b2", 9.9),
        ],
    )
    _install(monkeypatch, case_dir)

    with pytest.raises(ValueError, match="rank the holdout on its own result"):
        _lineage()


def _take_lock(case_dir: Path, holdout_training_hash: str, carrier: str = "carrier") -> None:
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.execute(
        "INSERT INTO research_locks VALUES (?,?,?,?)",
        (
            "lk",
            json.dumps(
                {"holdout_training_hash": holdout_training_hash, "prediction_hash": carrier}
            ),
            "LOCKED",
            "2026-01-01",
        ),
    )
    db.commit()
    db.close()


def test_the_lock_names_the_holdout_even_when_another_scores_higher(monkeypatch, tmp_path) -> None:
    """The lock records the sealed carrier and the run made from it, so the choice is
    determinate and never falls to the holdout's own Sharpe."""
    case_dir = _registry(
        tmp_path,
        [
            ("p1", "t1", "cfg_a", "b1", 0.4),
            ("p2", "t2", "cfg_b", "b2", 9.9),
        ],
    )
    _take_lock(case_dir, "t1")
    _install(monkeypatch, case_dir)

    assert _lineage()["backtest_hash"] == "b1"


def test_no_candidates_is_not_an_error(monkeypatch, tmp_path) -> None:
    case_dir = _registry(tmp_path, [])
    _install(monkeypatch, case_dir)

    assert _lineage() is None


def test_a_lock_whose_carrier_was_superseded_yields_no_holdout(monkeypatch, tmp_path) -> None:
    """A lock is immutable; the carrier it sealed can be refit past afterwards. Its holdout
    then evaluates a generation the study no longer publishes, so there is no holdout pair."""
    case_dir = _registry(tmp_path, [("p1", "t1", "cfg_a", "b1", 0.4)])
    _take_lock(case_dir, "t1", carrier="stale_carrier")
    monkeypatch.setattr(paired_metrics, "get_case_study_dir", lambda cs: case_dir)
    monkeypatch.setattr(
        paired_metrics, "_retired_prediction_hashes", lambda cs: frozenset({"stale_carrier"})
    )

    assert _lineage() is None
