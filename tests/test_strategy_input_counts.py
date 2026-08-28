"""What a strategy-analysis notebook can tell about its own inputs before it reads them.

`cohort_metrics` and `backtest_paired_metrics` are derived from `backtest_runs`, and until
`cme_futures/17` adopted the case-study-scoped producers they existed only because
`20_strategy_synthesis/01_aggregate_synthesis.py` had been run - a case study depending upward
on the chapter that aggregates it. Fifteen notebooks across seven case studies read those two
tables; one derives them (#943).

The distinction this helper draws is the one that decides what the notebook does next: no runs
at all is a refusal, because every figure downstream is computed from runs and an empty report
reads exactly like a finished one; runs present with nothing derived is work to do.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from case_studies.utils.notebook_contracts import strategy_input_counts


def _registry(case_dir: Path, tables: dict[str, int]) -> Path:
    db_path = case_dir / "run_log" / "registry.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as db:
        for table, rows in tables.items():
            db.execute(f"CREATE TABLE {table} (hash TEXT)")
            for i in range(rows):
                db.execute(f"INSERT INTO {table} VALUES (?)", (f"h{i}",))
    return db_path


def test_a_case_with_no_registry_reports_zeros_rather_than_raising(tmp_path: Path) -> None:
    """A clean clone has no registry, and that is an ordinary state, not an error.

    Reported rather than raised so the caller decides what it means: for the notebook it is a
    refusal, but the same reading is how a test or a status sweep asks the question harmlessly.
    """
    assert strategy_input_counts(tmp_path) == {
        "backtest_runs": 0,
        "cohort_metrics": 0,
        "backtest_paired_metrics": 0,
    }


def test_a_missing_table_counts_as_zero_not_as_a_failure(tmp_path: Path) -> None:
    """A registry can predate the derived tables entirely.

    Querying a table that does not exist raises in sqlite, so a helper that did not check
    `sqlite_master` first would turn "nothing has derived these yet" - the exact state this
    exists to detect - into an error the notebook could not distinguish from a broken registry.
    """
    _registry(tmp_path, {"backtest_runs": 7})
    assert strategy_input_counts(tmp_path) == {
        "backtest_runs": 7,
        "cohort_metrics": 0,
        "backtest_paired_metrics": 0,
    }


def test_it_separates_no_runs_from_runs_with_nothing_derived(tmp_path: Path) -> None:
    """The two states a caller must not confuse.

    Both leave the derived tables empty, and only one of them is work the notebook can do. With
    no runs there is nothing to derive from, and deriving would succeed while producing nothing.
    """
    nothing = tmp_path / "nothing"
    _registry(nothing, {"backtest_runs": 0, "cohort_metrics": 0, "backtest_paired_metrics": 0})
    undrived = tmp_path / "underived"
    _registry(undrived, {"backtest_runs": 12, "cohort_metrics": 0, "backtest_paired_metrics": 0})

    assert strategy_input_counts(nothing)["backtest_runs"] == 0
    assert strategy_input_counts(undrived)["backtest_runs"] == 12
    for counts in (strategy_input_counts(nothing), strategy_input_counts(undrived)):
        assert counts["cohort_metrics"] == 0
        assert counts["backtest_paired_metrics"] == 0


def test_a_fully_populated_registry_reports_every_count(tmp_path: Path) -> None:
    _registry(tmp_path, {"backtest_runs": 698, "cohort_metrics": 23, "backtest_paired_metrics": 5})
    assert strategy_input_counts(tmp_path) == {
        "backtest_runs": 698,
        "cohort_metrics": 23,
        "backtest_paired_metrics": 5,
    }


def test_it_opens_the_registry_read_only(tmp_path: Path) -> None:
    """It must not create or migrate anything it is only asking about.

    A helper that opened the registry for writing would create the file where there is none,
    turning "this case study has not been run" into "this case study has an empty registry" -
    and the notebook would then refuse for a reason one step removed from the truth.
    """
    strategy_input_counts(tmp_path)
    assert not (tmp_path / "run_log" / "registry.db").exists()
