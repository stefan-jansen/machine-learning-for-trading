"""Tests for case_studies/utils/registry/completeness.py.

The skip-if-complete invariant is load-bearing: wrong answers either
waste hours of compute (should have skipped) or silently reuse stale
partial artifacts (should have retrained). Tests:

- missing training_run → exists=False, not complete
- present training_run but no prediction_sets → partial, missing list
- complete run → exists=True, complete=True
- partial backtest_run (no daily_returns.parquet) → partial
- require_metrics=False relaxes the completeness rule as advertised
- skip_* wrappers return the same status (behavior pin for callers)
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from case_studies.utils.registry.completeness import (
    BacktestRunStatus,
    TrainingRunStatus,
    backtest_run_status,
    skip_backtest_if_complete,
    skip_training_if_complete,
    training_run_status,
)
from case_studies.utils.registry.specs import (
    backtest_hash_from_parts,
    training_hash_from_spec,
)
from case_studies.utils.registry.store import (
    REGISTRY_SCHEMA_SQL,
    _backtest_dir,
    _prediction_dir,
    _registry_db_path,
)


@pytest.fixture
def case_dir(tmp_path) -> Path:
    """Create a minimal case study dir with an empty registry.db."""
    case = tmp_path / "etfs"
    case.mkdir()
    db_path = _registry_db_path(case)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path))
    db.executescript(REGISTRY_SCHEMA_SQL)
    db.commit()
    db.close()
    return case


@pytest.fixture
def canonical_spec() -> dict:
    return {"family": "linear", "label": "fwd_ret_21d", "seed": 42, "n_folds": 5}


def _insert_training_run(case_dir: Path, spec: dict) -> str:
    """Insert a training_runs row. Returns the training_hash."""
    t_hash = training_hash_from_spec(spec)
    db = sqlite3.connect(str(_registry_db_path(case_dir)))
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    db.execute(
        "INSERT INTO training_runs (training_hash, family, label, config_name, spec_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (t_hash, spec["family"], spec["label"], "test", "{}", now),
    )
    db.commit()
    db.close()
    return t_hash


def _insert_prediction_set(case_dir: Path, t_hash: str, split: str = "val") -> str:
    """Insert a prediction_sets row. Returns the prediction_hash."""
    from case_studies.utils.registry.specs import prediction_hash_from_parts

    p_hash = prediction_hash_from_parts(t_hash, None, split)
    db = sqlite3.connect(str(_registry_db_path(case_dir)))
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    db.execute(
        "INSERT INTO prediction_sets (prediction_hash, training_hash, split, created_at) "
        "VALUES (?, ?, ?, ?)",
        (p_hash, t_hash, split, now),
    )
    db.commit()
    db.close()
    return p_hash


def _insert_prediction_metric(case_dir: Path, p_hash: str, ic_mean: float = 0.01) -> None:
    db = sqlite3.connect(str(_registry_db_path(case_dir)))
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    db.execute(
        "INSERT INTO prediction_metrics (prediction_hash, computed_at, ic_mean) VALUES (?, ?, ?)",
        (p_hash, now, ic_mean),
    )
    db.commit()
    db.close()


def _touch_predictions_file(case_dir: Path, p_hash: str) -> None:
    d = _prediction_dir(case_dir, p_hash)
    d.mkdir(parents=True, exist_ok=True)
    (d / "predictions.parquet").write_bytes(b"fake")


# -----------------------------------------------------------------------------
# training_run_status
# -----------------------------------------------------------------------------


def test_training_status_missing_run_is_not_complete(case_dir, canonical_spec) -> None:
    status = training_run_status("etfs", canonical_spec, case_dir=case_dir)

    assert not status.exists
    assert not status.complete
    assert not status.partial  # Neither complete nor partial when nothing exists
    assert "training_run" in status.missing
    assert status.training_hash == training_hash_from_spec(canonical_spec)


def test_training_status_run_without_predictions_is_partial(case_dir, canonical_spec) -> None:
    _insert_training_run(case_dir, canonical_spec)

    status = training_run_status("etfs", canonical_spec, case_dir=case_dir)
    assert status.exists
    assert status.partial
    assert not status.complete
    assert "prediction_sets" in status.missing


def test_training_status_run_without_metrics_is_partial(case_dir, canonical_spec) -> None:
    t_hash = _insert_training_run(case_dir, canonical_spec)
    p_hash = _insert_prediction_set(case_dir, t_hash)
    _touch_predictions_file(case_dir, p_hash)
    # Deliberately skip metric insertion

    status = training_run_status("etfs", canonical_spec, case_dir=case_dir)
    assert status.partial
    assert not status.complete
    assert "ic_mean" in status.missing


def test_training_status_complete_when_all_artifacts_present(case_dir, canonical_spec) -> None:
    t_hash = _insert_training_run(case_dir, canonical_spec)
    p_hash = _insert_prediction_set(case_dir, t_hash)
    _insert_prediction_metric(case_dir, p_hash)
    _touch_predictions_file(case_dir, p_hash)

    status = training_run_status("etfs", canonical_spec, case_dir=case_dir)
    assert status.complete
    assert not status.partial
    assert status.missing == ()


def test_training_status_require_metrics_false_relaxes_completeness(
    case_dir, canonical_spec
) -> None:
    """With require_metrics=False, a run is complete even without ic_mean."""
    t_hash = _insert_training_run(case_dir, canonical_spec)
    p_hash = _insert_prediction_set(case_dir, t_hash)
    _touch_predictions_file(case_dir, p_hash)
    # No metric insertion

    status = training_run_status("etfs", canonical_spec, case_dir=case_dir, require_metrics=False)
    assert status.complete
    assert "ic_mean" not in status.missing


def test_training_status_require_predictions_file_false_relaxes(case_dir, canonical_spec) -> None:
    t_hash = _insert_training_run(case_dir, canonical_spec)
    p_hash = _insert_prediction_set(case_dir, t_hash)
    _insert_prediction_metric(case_dir, p_hash)
    # Deliberately skip writing predictions.parquet

    status = training_run_status(
        "etfs", canonical_spec, case_dir=case_dir, require_predictions_file=False
    )
    assert status.complete


def test_training_status_summary_formats() -> None:
    missing = TrainingRunStatus(
        training_hash="abcdef1234567890",
        exists=False,
        has_predictions=False,
        has_predictions_file=False,
        has_metrics=False,
        missing=("training_run",),
    )
    assert "no training_run" in missing.summary()
    assert "abcdef123456" in missing.summary()

    partial = TrainingRunStatus(
        training_hash="abcdef1234567890",
        exists=True,
        has_predictions=True,
        has_predictions_file=False,
        has_metrics=False,
        missing=("predictions.parquet", "ic_mean"),
    )
    assert "partial" in partial.summary()
    assert "predictions.parquet" in partial.summary()

    complete = TrainingRunStatus(
        training_hash="abcdef1234567890",
        exists=True,
        has_predictions=True,
        has_predictions_file=True,
        has_metrics=True,
    )
    assert "complete" in complete.summary()


# -----------------------------------------------------------------------------
# skip_training_if_complete (thin wrapper)
# -----------------------------------------------------------------------------


def test_skip_training_returns_same_status_as_direct_call(case_dir, canonical_spec) -> None:
    direct = training_run_status("etfs", canonical_spec, case_dir=case_dir)
    wrapped = skip_training_if_complete("etfs", canonical_spec, case_dir=case_dir, verbose=False)
    assert direct.training_hash == wrapped.training_hash
    assert direct.complete == wrapped.complete


# -----------------------------------------------------------------------------
# backtest_run_status
# -----------------------------------------------------------------------------


def test_backtest_status_missing_is_not_complete(case_dir) -> None:
    strategy = {"signal": {"method": "equal_weight_top_k", "top_k": 10}}

    status = backtest_run_status("etfs", "pred123", strategy, case_dir=case_dir)
    assert not status.exists
    assert not status.complete
    assert "backtest_run" in status.missing


def test_backtest_status_partial_when_returns_missing(case_dir) -> None:
    strategy = {"signal": {"method": "equal_weight_top_k", "top_k": 10}}
    b_hash = backtest_hash_from_parts("pred123", strategy)

    # Insert backtest_runs row but no returns file
    db = sqlite3.connect(str(_registry_db_path(case_dir)))
    # Need to satisfy FK: insert a synthetic prediction first.
    # The schema is ON CASCADE default, but FK references must exist.
    db.execute("PRAGMA foreign_keys=OFF")  # Tests: skip FK check to simplify fixture
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    db.execute(
        "INSERT INTO backtest_runs (backtest_hash, prediction_hash, spec_json, created_at) "
        "VALUES (?, ?, ?, ?)",
        (b_hash, "pred123", "{}", now),
    )
    db.execute(
        "INSERT INTO backtest_metrics (backtest_hash, computed_at, sharpe) VALUES (?, ?, ?)",
        (b_hash, now, 1.5),
    )
    db.commit()
    db.close()

    status = backtest_run_status("etfs", "pred123", strategy, case_dir=case_dir)
    assert status.exists
    assert status.partial
    assert not status.complete
    assert "daily_returns.parquet" in status.missing


def test_backtest_status_complete_when_all_present(case_dir) -> None:
    strategy = {"signal": {"method": "equal_weight_top_k", "top_k": 10}}
    b_hash = backtest_hash_from_parts("pred123", strategy)

    db = sqlite3.connect(str(_registry_db_path(case_dir)))
    db.execute("PRAGMA foreign_keys=OFF")
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    db.execute(
        "INSERT INTO backtest_runs (backtest_hash, prediction_hash, spec_json, created_at) "
        "VALUES (?, ?, ?, ?)",
        (b_hash, "pred123", "{}", now),
    )
    db.execute(
        "INSERT INTO backtest_metrics (backtest_hash, computed_at, sharpe) VALUES (?, ?, ?)",
        (b_hash, now, 1.5),
    )
    db.commit()
    db.close()

    d = _backtest_dir(case_dir, b_hash)
    d.mkdir(parents=True, exist_ok=True)
    (d / "daily_returns.parquet").write_bytes(b"fake")

    status = backtest_run_status("etfs", "pred123", strategy, case_dir=case_dir)
    assert status.complete


def test_backtest_status_hash_is_deterministic(case_dir) -> None:
    strategy = {"signal": {"method": "x", "top_k": 10}, "allocation": {"method": "eq"}}
    s1 = backtest_run_status("etfs", "pred123", strategy, case_dir=case_dir)
    s2 = backtest_run_status("etfs", "pred123", dict(strategy), case_dir=case_dir)
    assert s1.backtest_hash == s2.backtest_hash


def test_backtest_status_summary_formats() -> None:
    missing = BacktestRunStatus(
        backtest_hash="abc123def456000",
        exists=False,
        has_returns=False,
        has_metrics=False,
        missing=("backtest_run",),
    )
    assert "no backtest_run" in missing.summary()


# -----------------------------------------------------------------------------
# skip_backtest_if_complete (thin wrapper)
# -----------------------------------------------------------------------------


def test_skip_backtest_wraps_backtest_run_status(case_dir) -> None:
    strategy = {"signal": {"method": "x"}}
    direct = backtest_run_status("etfs", "pred123", strategy, case_dir=case_dir)
    wrapped = skip_backtest_if_complete(
        "etfs", "pred123", strategy, case_dir=case_dir, verbose=False
    )
    assert direct.backtest_hash == wrapped.backtest_hash
    assert direct.complete == wrapped.complete
