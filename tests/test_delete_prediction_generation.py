"""The holdout replacement cascade removes the whole subtree, or removes nothing.

`REPLACE_HOLDOUT` in a holdout-predictions notebook deletes a superseded generation. The first
version of that deletion listed the child tables by hand, missed three of them, and left four
rows in the live sp500_options registry keyed to a prediction set nothing resolves. Enabling
foreign keys makes the same omission worse rather than better: `cohort_metrics.leader_hash`
references `backtest_runs`, so deleting the parent with a `cohort_metrics` row still pointing at
it raises `IntegrityError` and aborts the replacement half-done.

These tests build a registry with the real schema, put a row in every table that references the
two parents, and require the delete to leave none of them - and to leave a second generation
completely alone.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.registry.maintenance import delete_prediction_generation

SCHEMA = """
CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY);
CREATE TABLE prediction_sets (
    prediction_hash TEXT PRIMARY KEY,
    training_hash   TEXT NOT NULL REFERENCES training_runs(training_hash),
    split           TEXT NOT NULL
);
CREATE TABLE prediction_coverage (
    prediction_hash TEXT PRIMARY KEY REFERENCES prediction_sets(prediction_hash),
    status          TEXT
);
CREATE TABLE prediction_metrics (
    prediction_hash TEXT PRIMARY KEY REFERENCES prediction_sets(prediction_hash),
    ic_mean         REAL
);
CREATE TABLE fold_metrics (
    prediction_hash TEXT NOT NULL REFERENCES prediction_sets(prediction_hash),
    fold_id         INTEGER NOT NULL,
    ic              REAL
);
CREATE TABLE backtest_runs (
    backtest_hash   TEXT PRIMARY KEY,
    prediction_hash TEXT NOT NULL REFERENCES prediction_sets(prediction_hash),
    stage           TEXT
);
CREATE TABLE backtest_metrics (
    backtest_hash TEXT PRIMARY KEY REFERENCES backtest_runs(backtest_hash),
    sharpe        REAL
);
CREATE TABLE backtest_fold_metrics (
    backtest_hash TEXT NOT NULL REFERENCES backtest_runs(backtest_hash),
    fold_id       INTEGER NOT NULL
);
CREATE TABLE backtest_paired_metrics (
    challenger_hash TEXT NOT NULL REFERENCES backtest_runs(backtest_hash),
    benchmark_hash  TEXT NOT NULL REFERENCES backtest_runs(backtest_hash)
);
CREATE TABLE cohort_metrics (
    cohort_type TEXT NOT NULL,
    leader_hash TEXT NOT NULL REFERENCES backtest_runs(backtest_hash),
    k_variants  INTEGER NOT NULL
);
"""


def _registry(tmp_path: Path) -> Path:
    """Two complete generations, `doomed` and `keeper`, with every child row populated."""
    run_log = tmp_path / "run_log"
    run_log.mkdir()
    db_path = run_log / "registry.db"
    with sqlite3.connect(db_path) as db:
        db.executescript(SCHEMA)
        db.execute("INSERT INTO training_runs VALUES ('t1')")
        for pred, bt in (("doomed", "bt_doomed"), ("keeper", "bt_keeper")):
            db.execute("INSERT INTO prediction_sets VALUES (?, 't1', 'holdout')", (pred,))
            db.execute("INSERT INTO prediction_coverage VALUES (?, 'ok')", (pred,))
            db.execute("INSERT INTO prediction_metrics VALUES (?, 0.01)", (pred,))
            db.execute("INSERT INTO fold_metrics VALUES (?, 0, 0.01)", (pred,))
            db.execute("INSERT INTO fold_metrics VALUES (?, 1, 0.02)", (pred,))
            db.execute("INSERT INTO backtest_runs VALUES (?, ?, 'holdout')", (bt, pred))
            db.execute("INSERT INTO backtest_metrics VALUES (?, 0.5)", (bt,))
            db.execute("INSERT INTO backtest_fold_metrics VALUES (?, 0)", (bt,))
            db.execute("INSERT INTO backtest_paired_metrics VALUES (?, ?)", (bt, bt))
            db.execute("INSERT INTO cohort_metrics VALUES ('stagelabel', ?, 60)", (bt,))
        db.commit()
    for name in ("doomed", "keeper"):
        (run_log / "predictions" / name).mkdir(parents=True)
        (run_log / "predictions" / name / "predictions.parquet").write_bytes(b"x")
    (run_log / "backtest" / "bt_doomed").mkdir(parents=True)
    (run_log / "backtest" / "bt_doomed" / "daily_returns.parquet").write_bytes(b"x")
    return db_path


CHILD_TABLES = (
    "prediction_coverage",
    "prediction_metrics",
    "fold_metrics",
    "backtest_metrics",
    "backtest_fold_metrics",
    "cohort_metrics",
)


def test_the_delete_leaves_no_row_referencing_the_removed_generation(tmp_path: Path) -> None:
    db_path = _registry(tmp_path)

    delete_prediction_generation(db_path, "doomed")

    with sqlite3.connect(db_path) as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM prediction_sets WHERE prediction_hash = 'doomed'"
            ).fetchone()[0]
            == 0
        )
        assert (
            db.execute(
                "SELECT COUNT(*) FROM backtest_runs WHERE backtest_hash = 'bt_doomed'"
            ).fetchone()[0]
            == 0
        )
        orphans = {
            table: db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE "
                + (
                    "prediction_hash = 'doomed'"
                    if table.startswith("prediction") or table == "fold_metrics"
                    else "leader_hash = 'bt_doomed'"
                    if table == "cohort_metrics"
                    else "backtest_hash = 'bt_doomed'"
                )
            ).fetchone()[0]
            for table in CHILD_TABLES
        }
    assert orphans == dict.fromkeys(CHILD_TABLES, 0)


def test_the_delete_survives_foreign_key_enforcement(tmp_path: Path) -> None:
    """`cohort_metrics.leader_hash` is the one a hand-written list missed.

    With foreign keys on, leaving it behind does not orphan a row - it raises, and the
    replacement aborts with the parent gone and the children still there.
    """
    db_path = _registry(tmp_path)

    delete_prediction_generation(db_path, "doomed")

    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA foreign_keys = ON")
        violations = db.execute("PRAGMA foreign_key_check").fetchall()
    assert violations == []


def test_the_other_generation_is_untouched(tmp_path: Path) -> None:
    db_path = _registry(tmp_path)

    delete_prediction_generation(db_path, "doomed")

    with sqlite3.connect(db_path) as db:
        for table, column, value in (
            ("prediction_sets", "prediction_hash", "keeper"),
            ("prediction_coverage", "prediction_hash", "keeper"),
            ("prediction_metrics", "prediction_hash", "keeper"),
            ("backtest_runs", "backtest_hash", "bt_keeper"),
            ("backtest_metrics", "backtest_hash", "bt_keeper"),
            ("cohort_metrics", "leader_hash", "bt_keeper"),
        ):
            assert (
                db.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (value,)).fetchone()[
                    0
                ]
                == 1
            )
        assert (
            db.execute(
                "SELECT COUNT(*) FROM fold_metrics WHERE prediction_hash = 'keeper'"
            ).fetchone()[0]
            == 2
        )
    assert (db_path.parent / "predictions" / "keeper").is_dir()


def test_the_artifact_directories_go_with_the_rows(tmp_path: Path) -> None:
    """An unregistered artifact directory blocks its own re-run, so it cannot be left."""
    db_path = _registry(tmp_path)

    delete_prediction_generation(db_path, "doomed")

    assert not (db_path.parent / "predictions" / "doomed").exists()
    assert not (db_path.parent / "backtest" / "bt_doomed").exists()


def test_the_counts_report_what_was_removed(tmp_path: Path) -> None:
    db_path = _registry(tmp_path)

    deleted = delete_prediction_generation(db_path, "doomed")

    assert deleted["prediction_sets"] == 1
    assert deleted["backtest_runs"] == 1
    assert deleted["fold_metrics"] == 2
    assert deleted["cohort_metrics"] == 1
    assert 0 not in deleted.values()


def test_an_unknown_generation_deletes_nothing(tmp_path: Path) -> None:
    db_path = _registry(tmp_path)

    assert delete_prediction_generation(db_path, "never_registered") == {}

    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM prediction_sets").fetchone()[0] == 2


@pytest.mark.parametrize("parent", ["backtest_runs", "prediction_sets"])
def test_every_referencing_table_in_the_schema_is_reached(tmp_path: Path, parent: str) -> None:
    """The list is derived, so a table added later is covered without an edit here."""
    from case_studies.utils.registry.maintenance import _referencing_tables

    db_path = _registry(tmp_path)
    key = "backtest_hash" if parent == "backtest_runs" else "prediction_hash"
    with sqlite3.connect(db_path) as db:
        found = {table for table, _ in _referencing_tables(db, parent, key)}

    expected = {
        "backtest_runs": {
            "backtest_metrics",
            "backtest_fold_metrics",
            "backtest_paired_metrics",
            "cohort_metrics",
        },
        "prediction_sets": {
            "prediction_coverage",
            "prediction_metrics",
            "fold_metrics",
            "backtest_runs",
        },
    }[parent]
    assert found == expected
