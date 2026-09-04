"""`generate_holdout(force=True)` deletes the existing holdout, stale or not.

The delete used to be gated on `has_holdout_predictions`, which asks whether a holdout is
tied to one of the *current* validation top-N. Its own docstring says it returns False
when a holdout exists but no top-N candidate matches it - a holdout whose training fell
out of the top-N after a sweep reshuffle. So the delete was skipped exactly when the
existing holdout was most stale, the new one was written beside it, and the case study
carried two holdouts against the one-holdout-per-case-study rule.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "strategy_synthesis_holdout_force",
    Path(__file__).resolve().parents[1] / "20_strategy_synthesis" / "holdout.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_HOLDOUT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HOLDOUT
_SPEC.loader.exec_module(_HOLDOUT)


class _ReachedSelection(Exception):
    """Raised in place of the retrain, which is what this test is not exercising."""


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    def _delete(cs_id: str) -> int:
        seen.append("delete")
        return 1

    def _select(*args, **kwargs):
        raise _ReachedSelection

    monkeypatch.setattr(_HOLDOUT, "delete_holdout_predictions", _delete)
    monkeypatch.setattr(_HOLDOUT, "select_best_models", _select)
    monkeypatch.setattr(_HOLDOUT, "load_existing_holdout", lambda cs_id: seen.append("load") or {})
    return seen


def test_force_deletes_a_holdout_no_top_n_candidate_matches(calls, monkeypatch) -> None:
    """The stale case: `has_holdout_predictions` is False and a holdout is still on disk."""
    monkeypatch.setattr(_HOLDOUT, "has_holdout_predictions", lambda cs_id: False)

    with pytest.raises(_ReachedSelection):
        _HOLDOUT.generate_holdout("nasdaq100_microstructure", force=True, verbose=False)

    assert calls == ["delete"]


def test_force_deletes_a_holdout_the_top_n_still_matches(calls, monkeypatch) -> None:
    monkeypatch.setattr(_HOLDOUT, "has_holdout_predictions", lambda cs_id: True)

    with pytest.raises(_ReachedSelection):
        _HOLDOUT.generate_holdout("nasdaq100_microstructure", force=True, verbose=False)

    assert calls == ["delete"]


def test_without_force_a_matching_holdout_is_still_loaded(calls, monkeypatch) -> None:
    monkeypatch.setattr(_HOLDOUT, "has_holdout_predictions", lambda cs_id: True)

    _HOLDOUT.generate_holdout("nasdaq100_microstructure", force=False, verbose=False)

    assert calls == ["load"]


def _registry_with_a_holdout(tmp_path):
    """The parent/child shape the delete has to walk, with the two FKs it used to miss."""
    import sqlite3

    case_dir = tmp_path / "cs"
    (case_dir / "run_log").mkdir(parents=True)
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.executescript(
        """
        CREATE TABLE prediction_sets (prediction_hash TEXT PRIMARY KEY, split TEXT);
        CREATE TABLE prediction_coverage (
            prediction_hash TEXT PRIMARY KEY
                REFERENCES prediction_sets(prediction_hash));
        CREATE TABLE prediction_metrics (
            prediction_hash TEXT REFERENCES prediction_sets(prediction_hash));
        CREATE TABLE fold_metrics (
            prediction_hash TEXT REFERENCES prediction_sets(prediction_hash));
        CREATE TABLE backtest_runs (
            backtest_hash TEXT PRIMARY KEY,
            prediction_hash TEXT REFERENCES prediction_sets(prediction_hash));
        CREATE TABLE backtest_metrics (
            backtest_hash TEXT REFERENCES backtest_runs(backtest_hash));
        CREATE TABLE backtest_fold_metrics (
            backtest_hash TEXT REFERENCES backtest_runs(backtest_hash));
        CREATE TABLE backtest_paired_metrics (
            challenger_hash TEXT REFERENCES backtest_runs(backtest_hash),
            benchmark_hash TEXT);
        CREATE TABLE cohort_metrics (
            leader_hash TEXT REFERENCES backtest_runs(backtest_hash));

        INSERT INTO prediction_sets VALUES ('p_hold', 'holdout');
        INSERT INTO prediction_sets VALUES ('p_val', 'validation');
        INSERT INTO prediction_coverage VALUES ('p_hold');
        INSERT INTO prediction_coverage VALUES ('p_val');
        INSERT INTO prediction_metrics VALUES ('p_hold');
        INSERT INTO fold_metrics VALUES ('p_hold');
        INSERT INTO backtest_runs VALUES ('b_hold', 'p_hold');
        INSERT INTO backtest_metrics VALUES ('b_hold');
        INSERT INTO backtest_fold_metrics VALUES ('b_hold');
        INSERT INTO backtest_paired_metrics VALUES ('b_hold', 'other');
        INSERT INTO backtest_paired_metrics VALUES ('other', 'b_hold');
        INSERT INTO cohort_metrics VALUES ('b_hold');
        """
    )
    db.commit()
    db.close()
    return case_dir


def test_deleting_a_holdout_clears_every_row_the_schema_points_at(tmp_path, monkeypatch) -> None:
    """`prediction_coverage` and `cohort_metrics` were missing from the hand-written list.

    Both are declared foreign keys into the rows being deleted, so with
    `PRAGMA foreign_keys=ON` every registered holdout raised
    `sqlite3.IntegrityError: FOREIGN KEY constraint failed`. Reproduced on a copy of
    cme_futures' production registry (holdout 18d48c3b9cc2) before the fix.
    """
    import sqlite3

    case_dir = _registry_with_a_holdout(tmp_path)
    monkeypatch.setattr(_HOLDOUT, "get_case_study_dir", lambda cs_id, **kwargs: case_dir)

    assert _HOLDOUT.delete_holdout_predictions("cs") == 1

    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    counts = {
        table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "prediction_sets",
            "prediction_coverage",
            "prediction_metrics",
            "fold_metrics",
            "backtest_runs",
            "backtest_metrics",
            "backtest_fold_metrics",
            "backtest_paired_metrics",
            "cohort_metrics",
        )
    }
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    db.close()

    # The validation prediction and its coverage row are untouched.
    assert counts["prediction_sets"] == 1
    assert counts["prediction_coverage"] == 1
    assert counts["prediction_metrics"] == 0
    assert counts["fold_metrics"] == 0
    assert counts["backtest_runs"] == 0
    assert counts["backtest_metrics"] == 0
    assert counts["backtest_fold_metrics"] == 0
    # Both the challenger row and the benchmark row referencing the holdout backtest.
    assert counts["backtest_paired_metrics"] == 0
    assert counts["cohort_metrics"] == 0
