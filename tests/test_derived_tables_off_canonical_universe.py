"""A derived table populated under a different selection is populated, and wrong.

`18_strategy_analysis` fills `cohort_metrics` and `backtest_paired_metrics` when they are
empty. Emptiness is the wrong question on a rerun: a table written by an earlier run that
selected across the full universe is not empty, and rebuilding is skipped precisely when it is
needed. Neither table records the selection that produced it, so it is recovered from what its
rows point at.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.notebook_contracts import derived_tables_off_canonical_universe


def _registry(case_dir: Path) -> sqlite3.Connection:
    (case_dir / "run_log").mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.execute("CREATE TABLE IF NOT EXISTS backtest_runs (backtest_hash TEXT, spec_json TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS cohort_metrics (leader_hash TEXT)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS backtest_paired_metrics "
        "(challenger_hash TEXT, benchmark_hash TEXT)"
    )
    return db


def _run(
    db: sqlite3.Connection, backtest_hash: str, universe: str | None, *, canonical: bool = True
) -> str:
    """A backtest row carrying `universe`, in either spec shape the registry holds.

    `strategy_view` exists because both shapes are in the registries: the canonical form is
    `version: 2` with `strategy` and `backtest_config` present, and anything else is read as
    already being the strategy view. Getting this wrong is silent - the lookup falls through to
    the default and every row reads as `full` - so both shapes are exercised.
    """
    if canonical:
        spec = {
            "version": 2,
            "strategy": {"signal": {"universe_filter": universe}},
            "backtest_config": {},
        }
    else:
        spec = {"signal": {"universe_filter": universe}}
    db.execute(
        "INSERT INTO backtest_runs (backtest_hash, spec_json) VALUES (?, ?)",
        (backtest_hash, json.dumps(spec)),
    )
    return backtest_hash


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    return tmp_path / "nasdaq100_microstructure"


def test_a_cohort_leader_outside_the_universe_marks_the_table(case_dir: Path) -> None:
    """The reported case: cohorts computed before the notebook passed a universe filter."""
    with _registry(case_dir) as db:
        _run(db, "aaaa", "cost_feasible")
        _run(db, "bbbb", "full")
        db.executemany("INSERT INTO cohort_metrics VALUES (?)", [("aaaa",), ("bbbb",)])

    assert derived_tables_off_canonical_universe(case_dir, "cost_feasible") == {"cohort_metrics"}


def test_a_table_entirely_inside_the_universe_is_left_alone(case_dir: Path) -> None:
    """Rebuilding a correct table is not free - `compute_and_register` replaces all rows."""
    with _registry(case_dir) as db:
        _run(db, "aaaa", "cost_feasible")
        db.execute("INSERT INTO cohort_metrics VALUES (?)", ("aaaa",))

    assert derived_tables_off_canonical_universe(case_dir, "cost_feasible") == set()


def test_either_side_of_a_pair_is_enough(case_dir: Path) -> None:
    """A comparison is off-canon if either leg is, so both columns are checked."""
    with _registry(case_dir) as db:
        _run(db, "aaaa", "cost_feasible")
        _run(db, "bbbb", "full")
        db.execute("INSERT INTO backtest_paired_metrics VALUES (?, ?)", ("aaaa", "bbbb"))

    assert derived_tables_off_canonical_universe(case_dir, "cost_feasible") == {
        "backtest_paired_metrics"
    }


def test_a_missing_universe_in_the_spec_reads_as_full(case_dir: Path) -> None:
    """Rows predating the universe axis carry no filter, and `full` is what they were.

    Reading a null as "matches whatever is canonical" would make exactly the oldest rows - the
    ones most likely to have been selected differently - invisible to this check.
    """
    with _registry(case_dir) as db:
        _run(db, "aaaa", None)
        db.execute("INSERT INTO cohort_metrics VALUES (?)", ("aaaa",))

    assert derived_tables_off_canonical_universe(case_dir, "cost_feasible") == {"cohort_metrics"}


def test_an_unpinned_case_study_has_nothing_to_be_outside_of(case_dir: Path) -> None:
    with _registry(case_dir) as db:
        _run(db, "bbbb", "full")
        db.execute("INSERT INTO cohort_metrics VALUES (?)", ("bbbb",))

    assert derived_tables_off_canonical_universe(case_dir, None) == set()


def test_a_directory_with_no_registry_reports_nothing(tmp_path: Path) -> None:
    assert derived_tables_off_canonical_universe(tmp_path / "absent", "cost_feasible") == set()


def test_the_legacy_spec_shape_is_read_too(case_dir: Path) -> None:
    """A pre-`version: 2` spec is already the strategy view, and must not read as `full`.

    If it did, every legacy row would look off-canon and every table holding one would be
    rebuilt on every run - the opposite failure, and just as invisible.
    """
    with _registry(case_dir) as db:
        _run(db, "aaaa", "cost_feasible", canonical=False)
        db.execute("INSERT INTO cohort_metrics VALUES (?)", ("aaaa",))

    assert derived_tables_off_canonical_universe(case_dir, "cost_feasible") == set()
