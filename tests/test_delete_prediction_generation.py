"""The holdout replacement cascade removes the whole subtree, or removes nothing.

`REPLACE_HOLDOUT` in a holdout-predictions notebook deletes a superseded generation. The first
version of that deletion listed the child tables by hand, missed three of them, and left four
rows in the live sp500_options registry keyed to a prediction set nothing resolves.

Two things then have to hold at once, and they pull in opposite directions:

- Enabling foreign keys makes a missed table fail loudly rather than quietly:
  `cohort_metrics.leader_hash` references `backtest_runs`, so deleting the parent while a
  `cohort_metrics` row still points at it raises `IntegrityError` and aborts the replacement
  half-done. Deriving the child tables from `PRAGMA foreign_key_list` is what closes that.
- Deriving them is not sufficient. `backtest_paired_metrics.benchmark_hash` carries a synthetic
  benchmark as often as a registered one - the equal-weight universe is not a `backtest_runs`
  row - so it deliberately has no foreign key, and the pragma cannot see it.

Every registry here is built from `REGISTRY_SCHEMA_SQL`, the DDL production uses. A fixture that
declares its own schema is free to give `benchmark_hash` a foreign key production does not have,
and would then pass while the real registry kept the row.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.registry.maintenance import (
    _UNENFORCED_BACKTEST_REFERENCES,
    _UNENFORCED_PREDICTION_REFERENCES,
    _referencing_tables,
    delete_prediction_generation,
)
from case_studies.utils.registry.store import REGISTRY_SCHEMA_SQL

TS = "2026-01-01T00:00:00+00:00"


def _generation(db: sqlite3.Connection, pred: str, bt: str, label: str) -> None:
    """One prediction set and one backtest, with a row in every table that references either."""
    db.execute(
        "INSERT INTO prediction_sets (prediction_hash, training_hash, split, created_at) "
        "VALUES (?, 't1', 'holdout', ?)",
        (pred, TS),
    )
    db.execute(
        "INSERT INTO prediction_coverage (prediction_hash, expected_key_digest, "
        "actual_key_digest, n_expected, n_actual, n_duplicates, n_missing, n_extra, n_null, "
        "n_non_finite, n_folds_expected, n_folds_actual, schema_json, artifact_digest, status) "
        "VALUES (?, 'd', 'd', 1, 1, 0, 0, 0, 0, 0, 1, 1, '{}', 'a', 'ok')",
        (pred,),
    )
    db.execute(
        "INSERT INTO prediction_metrics (prediction_hash, computed_at) VALUES (?, ?)", (pred, TS)
    )
    for fold_id in (0, 1):
        db.execute(
            "INSERT INTO fold_metrics (prediction_hash, fold_id, computed_at) VALUES (?, ?, ?)",
            (pred, fold_id, TS),
        )
    db.execute(
        "INSERT INTO backtest_runs (backtest_hash, prediction_hash, stage, created_at) "
        "VALUES (?, ?, 'holdout', ?)",
        (bt, pred, TS),
    )
    db.execute("INSERT INTO backtest_metrics (backtest_hash, computed_at) VALUES (?, ?)", (bt, TS))
    db.execute(
        "INSERT INTO backtest_fold_metrics (backtest_hash, fold_id, computed_at) VALUES (?, 0, ?)",
        (bt, TS),
    )
    db.execute(
        "INSERT INTO cohort_metrics (cohort_type, label, leader_hash, k_variants, "
        "periods_per_year, computed_at) VALUES ('stagelabel', ?, ?, 60, 252.0, ?)",
        # A distinct label per generation: idx_cohort_unique is unique on
        # (cohort_type, stage, label, family), so two generations cannot share one.
        (label, bt, TS),
    )


def _registry(tmp_path: Path) -> Path:
    """Two generations, `doomed` and `keeper`, on the production schema."""
    run_log = tmp_path / "run_log"
    run_log.mkdir()
    db_path = run_log / "registry.db"
    with sqlite3.connect(db_path) as db:
        db.executescript(REGISTRY_SCHEMA_SQL)
        db.execute(
            "INSERT INTO training_runs (training_hash, family, label, created_at) "
            "VALUES ('t1', 'linear', 'lab', ?)",
            (TS,),
        )
        _generation(db, "doomed", "bt_doomed", "lab_doomed")
        _generation(db, "keeper", "bt_keeper", "lab_keeper")
        # The deleted backtest as CHALLENGER against a synthetic benchmark, and - the case the
        # pragma cannot reach - as the BENCHMARK a surviving backtest was compared against.
        db.execute(
            "INSERT INTO backtest_paired_metrics (challenger_hash, benchmark_hash, "
            "benchmark_kind, computed_at) VALUES ('bt_doomed', 'ew_universe', 'synthetic', ?)",
            (TS,),
        )
        db.execute(
            "INSERT INTO backtest_paired_metrics (challenger_hash, benchmark_hash, "
            "benchmark_kind, computed_at) VALUES ('bt_keeper', 'bt_doomed', 'registered', ?)",
            (TS,),
        )
        db.commit()
    for name in ("doomed", "keeper"):
        (run_log / "predictions" / name).mkdir(parents=True)
        (run_log / "predictions" / name / "predictions.parquet").write_bytes(b"x")
    (run_log / "backtest" / "bt_doomed").mkdir(parents=True)
    (run_log / "backtest" / "bt_doomed" / "daily_returns.parquet").write_bytes(b"x")
    return db_path


def test_no_row_referencing_the_removed_generation_survives(tmp_path: Path) -> None:
    db_path = _registry(tmp_path)

    delete_prediction_generation(db_path, "doomed")

    with sqlite3.connect(db_path) as db:
        remaining = {
            table: db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (value,)
            ).fetchone()[0]
            for table, column, value in (
                ("prediction_sets", "prediction_hash", "doomed"),
                ("prediction_coverage", "prediction_hash", "doomed"),
                ("prediction_metrics", "prediction_hash", "doomed"),
                ("fold_metrics", "prediction_hash", "doomed"),
                ("backtest_runs", "backtest_hash", "bt_doomed"),
                ("backtest_metrics", "backtest_hash", "bt_doomed"),
                ("backtest_fold_metrics", "backtest_hash", "bt_doomed"),
                ("cohort_metrics", "leader_hash", "bt_doomed"),
                ("backtest_paired_metrics", "challenger_hash", "bt_doomed"),
            )
        }
    assert remaining == dict.fromkeys(remaining, 0)


def test_the_paired_row_that_names_it_only_as_the_benchmark_goes_too(tmp_path: Path) -> None:
    """`benchmark_hash` has no foreign key, so nothing about the schema announces this row.

    A surviving backtest compared AGAINST the deleted one leaves a paired row whose benchmark
    resolves to nothing. `PRAGMA foreign_key_check` stays silent about it - which is exactly why
    a purely schema-derived cascade is not enough.
    """
    db_path = _registry(tmp_path)

    delete_prediction_generation(db_path, "doomed")

    with sqlite3.connect(db_path) as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM backtest_paired_metrics WHERE benchmark_hash = 'bt_doomed'"
            ).fetchone()[0]
            == 0
        )


def test_benchmark_hash_really_has_no_foreign_key_in_the_production_schema() -> None:
    """Pins the premise. If a key is added later, the declared entry becomes redundant."""
    with sqlite3.connect(":memory:") as db:
        db.executescript(REGISTRY_SCHEMA_SQL)
        keyed = {fk[3] for fk in db.execute("PRAGMA foreign_key_list(backtest_paired_metrics)")}
    assert "challenger_hash" in keyed
    assert "benchmark_hash" not in keyed


def test_every_declared_unenforced_reference_is_really_unenforced() -> None:
    """A declared entry that the schema does in fact key is a duplicate, not a safety net."""
    with sqlite3.connect(":memory:") as db:
        db.executescript(REGISTRY_SCHEMA_SQL)
        present = {
            row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for parent, key, declared in (
            ("backtest_runs", "backtest_hash", _UNENFORCED_BACKTEST_REFERENCES),
            ("prediction_sets", "prediction_hash", _UNENFORCED_PREDICTION_REFERENCES),
        ):
            derived = set(_referencing_tables(db, parent, key))
            for table, column in declared:
                if table in present:
                    assert (table, column) not in derived


def test_the_delete_survives_foreign_key_enforcement(tmp_path: Path) -> None:
    db_path = _registry(tmp_path)

    delete_prediction_generation(db_path, "doomed")

    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA foreign_keys = ON")
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_the_other_generation_is_untouched(tmp_path: Path) -> None:
    db_path = _registry(tmp_path)

    delete_prediction_generation(db_path, "doomed")

    with sqlite3.connect(db_path) as db:
        for table, column, value, expected in (
            ("prediction_sets", "prediction_hash", "keeper", 1),
            ("prediction_coverage", "prediction_hash", "keeper", 1),
            ("prediction_metrics", "prediction_hash", "keeper", 1),
            ("fold_metrics", "prediction_hash", "keeper", 2),
            ("backtest_runs", "backtest_hash", "bt_keeper", 1),
            ("backtest_metrics", "backtest_hash", "bt_keeper", 1),
            ("cohort_metrics", "leader_hash", "bt_keeper", 1),
        ):
            assert (
                db.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (value,)).fetchone()[
                    0
                ]
                == expected
            ), f"{table}.{column}"
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
    # Both paired rows: one where it is the challenger, one where it is the benchmark.
    assert deleted["backtest_paired_metrics"] == 2
    assert 0 not in deleted.values()


def test_an_unknown_generation_deletes_nothing(tmp_path: Path) -> None:
    db_path = _registry(tmp_path)

    assert delete_prediction_generation(db_path, "never_registered") == {}

    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM prediction_sets").fetchone()[0] == 2


@pytest.mark.parametrize(
    ("parent", "key", "expected"),
    [
        (
            "backtest_runs",
            "backtest_hash",
            {
                "backtest_metrics",
                "backtest_fold_metrics",
                "backtest_paired_metrics",
                "cohort_metrics",
            },
        ),
        (
            "prediction_sets",
            "prediction_hash",
            {"prediction_coverage", "prediction_metrics", "fold_metrics", "backtest_runs"},
        ),
    ],
)
def test_the_derived_tables_match_the_production_schema(
    parent: str, key: str, expected: set[str]
) -> None:
    """Derived, not written down, so a table added later is covered without an edit here."""
    with sqlite3.connect(":memory:") as db:
        db.executescript(REGISTRY_SCHEMA_SQL)
        found = {table for table, _ in _referencing_tables(db, parent, key)}
    assert found == expected


# The three tables a retired holdout-lock mechanism left behind. No production code writes them
# and `REGISTRY_SCHEMA_SQL` no longer creates them, but they are still present with rows in
# registries that predate the change - sp500_options' among them - which is why the cascade
# declares them. Their DDL is copied from a live registry, so the declared table and column
# names are checked against the shape they actually have rather than against a restatement.
LEGACY_HOLDOUT_SCHEMA = """
CREATE TABLE research_locks (
    lock_hash  TEXT PRIMARY KEY,
    lock_json  TEXT NOT NULL,
    state      TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE holdout_evaluations (
    lock_hash               TEXT PRIMARY KEY REFERENCES research_locks(lock_hash),
    holdout_training_hash   TEXT NOT NULL,
    holdout_prediction_hash TEXT NOT NULL,
    holdout_backtest_hash   TEXT NOT NULL,
    fitted_state_digest     TEXT,
    evaluated_at            TEXT NOT NULL
);
CREATE TABLE holdout_staging (
    lock_hash               TEXT PRIMARY KEY REFERENCES research_locks(lock_hash),
    holdout_training_hash   TEXT NOT NULL,
    holdout_prediction_hash TEXT NOT NULL,
    holdout_backtest_hash   TEXT NOT NULL,
    fitted_state_digest     TEXT,
    lineage_digest          TEXT NOT NULL,
    staged_at               TEXT NOT NULL
);
"""


def _legacy_registry(tmp_path: Path) -> Path:
    db_path = _registry(tmp_path)
    with sqlite3.connect(db_path) as db:
        db.executescript(LEGACY_HOLDOUT_SCHEMA)
        db.execute("INSERT INTO research_locks VALUES ('lock1', '{}', 'evaluated', ?)", (TS,))
        db.execute(
            "INSERT INTO holdout_evaluations VALUES ('lock1', 't1', 'doomed', 'bt_doomed', "
            "'digest', ?)",
            (TS,),
        )
        db.execute(
            "INSERT INTO holdout_staging VALUES ('lock1', 't1', 'doomed', 'bt_doomed', "
            "'digest', 'lineage', ?)",
            (TS,),
        )
        db.commit()
    return db_path


def test_a_legacy_registrys_holdout_tables_are_cleaned_too(tmp_path: Path) -> None:
    """Without this the two declared legacy entries are unverifiable.

    Neither table is in `REGISTRY_SCHEMA_SQL` any more, so every other test here skips them by
    table presence, and a typo in either the table or the column name would pass while a
    retained registry kept the rows.
    """
    db_path = _legacy_registry(tmp_path)

    deleted = delete_prediction_generation(db_path, "doomed")

    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM holdout_staging").fetchone()[0] == 0
    assert deleted["holdout_evaluations"] == 1
    assert deleted["holdout_staging"] == 1


def test_the_legacy_tables_are_reached_by_the_declared_list_not_the_pragma(
    tmp_path: Path,
) -> None:
    """They carry no foreign key to either parent, so nothing derives them."""
    db_path = _legacy_registry(tmp_path)

    with sqlite3.connect(db_path) as db:
        derived = {table for table, _ in _referencing_tables(db, "backtest_runs", "backtest_hash")}
        derived |= {
            table for table, _ in _referencing_tables(db, "prediction_sets", "prediction_hash")
        }
    assert "holdout_evaluations" not in derived
    assert "holdout_staging" not in derived


def test_a_registry_without_the_legacy_tables_is_unaffected(tmp_path: Path) -> None:
    """The declared entries are guarded by table presence, not assumed to exist."""
    db_path = _registry(tmp_path)

    deleted = delete_prediction_generation(db_path, "doomed")

    assert "holdout_evaluations" not in deleted
    assert "holdout_staging" not in deleted
