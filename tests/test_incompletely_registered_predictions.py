"""The check `13_model_analysis` makes before it reads a leaderboard off declared members.

Filtering the metrics to the members of the populations in force answers "which rows belong
here", not "did those rows finish". Coverage, the headline metrics and the per-fold metrics are
separate writes, so a run interrupted between them leaves a member that every metrics query
returns and that `PredictionResult.complete` rejects. Scoring it anyway averages over the folds
it managed, and a shorter window is an easier window - the direction that flatters it into the
leaderboard rather than out of it.

These exercise the registry states that distinction turns on, so a check that only asks whether
a headline row exists fails them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.notebook_contracts import incompletely_registered_predictions

FINISHED = "aaaa11112222"
INTERRUPTED = "bbbb33334444"


def _registry(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case"
    (case_dir / "run_log").mkdir(parents=True)
    db_path = case_dir / "run_log" / "registry.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE prediction_coverage "
            "(prediction_hash TEXT PRIMARY KEY, status TEXT, n_folds_expected INTEGER)"
        )
        db.execute("CREATE TABLE fold_metrics (prediction_hash TEXT, fold_id INTEGER, ic REAL)")
    return case_dir


def _register(case_dir: Path, member: str, *, status: str, expected: int, scored: int) -> None:
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute("INSERT INTO prediction_coverage VALUES (?, ?, ?)", (member, status, expected))
        db.executemany(
            "INSERT INTO fold_metrics VALUES (?, ?, ?)",
            [(member, fold, 0.01) for fold in range(scored)],
        )


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    case_dir = _registry(tmp_path)
    _register(case_dir, FINISHED, status="complete", expected=5, scored=5)
    return case_dir


class TestAMemberThatFinished:
    def test_is_not_reported(self, case_dir: Path) -> None:
        assert incompletely_registered_predictions(case_dir, [FINISHED]) == {}

    def test_a_null_fold_ic_is_not_incompleteness(self, case_dir: Path) -> None:
        """A degenerate fold is a scored fold. It is excluded from the leaderboard on its own
        rule (`degenerate_prediction_hashes`), which says something different about the member
        than "it never finished registering"."""
        _register(case_dir, INTERRUPTED, status="complete", expected=3, scored=0)
        with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
            db.executemany(
                "INSERT INTO fold_metrics VALUES (?, ?, ?)",
                [(INTERRUPTED, fold, None) for fold in range(3)],
            )
        assert incompletely_registered_predictions(case_dir, [INTERRUPTED]) == {}


class TestAMemberInterruptedPartWay:
    def test_fewer_folds_than_expected_is_reported_with_both_counts(self, case_dir: Path) -> None:
        _register(case_dir, INTERRUPTED, status="complete", expected=5, scored=2)
        assert incompletely_registered_predictions(case_dir, [FINISHED, INTERRUPTED]) == {
            INTERRUPTED: "2 of 5 folds scored"
        }

    def test_no_fold_metrics_at_all_is_reported(self, case_dir: Path) -> None:
        _register(case_dir, INTERRUPTED, status="complete", expected=5, scored=0)
        assert incompletely_registered_predictions(case_dir, [INTERRUPTED]) == {
            INTERRUPTED: "0 of 5 folds scored"
        }

    def test_coverage_that_does_not_say_complete_is_reported(self, case_dir: Path) -> None:
        _register(case_dir, INTERRUPTED, status="missing_rows", expected=5, scored=5)
        assert incompletely_registered_predictions(case_dir, [INTERRUPTED]) == {
            INTERRUPTED: "coverage missing_rows"
        }

    def test_a_member_with_no_coverage_row_is_reported(self, case_dir: Path) -> None:
        assert incompletely_registered_predictions(case_dir, [INTERRUPTED]) == {
            INTERRUPTED: "no coverage row"
        }


class TestARegistryThatCannotAnswer:
    """A reader's clean clone has no registry, and a fixture may have no coverage table. Neither
    is evidence that a member is unfinished, so neither may raise a notebook that is about to
    fall back to comparing whatever it does hold."""

    def test_a_missing_registry_reports_nothing(self, tmp_path: Path) -> None:
        assert incompletely_registered_predictions(tmp_path / "absent", [FINISHED]) == {}

    def test_a_registry_without_the_coverage_table_reports_nothing(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case"
        (case_dir / "run_log").mkdir(parents=True)
        with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
            db.execute("CREATE TABLE prediction_metrics (prediction_hash TEXT)")
        assert incompletely_registered_predictions(case_dir, [FINISHED]) == {}

    def test_no_members_reports_nothing(self, case_dir: Path) -> None:
        assert incompletely_registered_predictions(case_dir, []) == {}
