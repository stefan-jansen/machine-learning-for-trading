"""A NULL direction_label says which kind of NULL it is.

The column carried two opposite claims behind one face. `fwd_ret_24h` declares no direction
sibling, so its NULL means "there is nothing to score against" - correct, in every family.
Every `deep_learning` run in `crypto_perps_funding` also wrote NULL, and there it meant "a
sibling is declared and scoring against it raised". 120 of 120 sets, for months, with the
only trace a `logger.warning` into a papermill log the harness deletes when the run succeeds.

The ambiguity was not cosmetic: once the join was repaired, 20 of the 21 significant sets
turned out to sit *below* 0.5 with confidence intervals excluding it. A blank told a reader
nothing was there when the answer was "this model ranks down-moves higher".

Three states, and the tests below pin each: computed, declined, raised.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from case_studies.utils.registry.metrics import compute_prediction_fold_metrics

_SYMBOLS = ("BTC", "ETH", "SOL", "ADA", "DOT", "AVAX", "LINK", "XRP")


def _panel(n: int) -> tuple[list[dt.datetime], list[str]]:
    start = dt.datetime(2024, 1, 1)
    stamps = [start + dt.timedelta(hours=8 * i) for i in range(n)]
    return (
        [s for s in stamps for _ in _SYMBOLS],
        [sym for _ in stamps for sym in _SYMBOLS],
    )


def _predictions(n: int = 200, unit: str = "ms") -> pl.DataFrame:
    stamps, symbols = _panel(n)
    width = len(_SYMBOLS)
    return pl.DataFrame(
        {
            "timestamp": stamps,
            "symbol": symbols,
            "fold_id": [0] * (n * width),
            "y_true": [(j - width / 2) / width for _ in range(n) for j in range(width)],
            "y_score": [j / width for _ in range(n) for j in range(width)],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime(unit, "UTC")))


def _direction(n: int = 200, unit: str = "ms") -> pl.DataFrame:
    stamps, symbols = _panel(n)
    width = len(_SYMBOLS)
    return pl.DataFrame(
        {
            "timestamp": stamps,
            "symbol": symbols,
            "fwd_dir_8h": [1 if j >= width // 2 else 0 for _ in range(n) for j in range(width)],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime(unit, "UTC")))


def _headline(predictions, direction_labels, direction_col):
    headline, _ = compute_prediction_fold_metrics(
        predictions,
        direction_labels=direction_labels,
        direction_col=direction_col,
        task_type="regression",
    )
    return headline


def test_a_computed_direction_auc_records_no_error():
    headline = _headline(_predictions(), _direction(), "fwd_dir_8h")

    assert headline["direction_label"] == "fwd_dir_8h"
    assert headline["direction_label_error"] is None


def test_a_raising_join_records_why():
    """The shape that hid 120 runs: a sibling declared, and scoring against it raised."""
    unjoinable = _direction().with_columns(pl.col("timestamp").dt.strftime("%Y-%m-%dT%H:%M:%S"))

    headline = _headline(_predictions(), unjoinable, "fwd_dir_8h")

    assert "direction_label" not in headline, "a failed run must not claim a direction label"
    assert "cannot align join key" in headline["direction_label_error"]
    assert headline["direction_label_error"].startswith("TypeError: ")


def test_a_declined_computation_is_distinguishable_from_a_raise():
    """The third state: declared, joinable, and not computable. Also formerly silent."""
    headline = _headline(_predictions(n=4), _direction(n=4), "fwd_dir_8h")

    assert "direction_label" not in headline
    assert "not computable" in headline["direction_label_error"]


def test_no_declared_sibling_writes_no_error_at_all():
    """`fwd_ret_24h`'s NULL. Nothing was asked for, so nothing failed."""
    headline = _headline(_predictions(), None, None)

    assert "direction_label" not in headline
    assert "direction_label_error" not in headline


def test_the_error_is_bounded():
    """A metrics row is not a log; a polars schema error runs to hundreds of characters."""
    unjoinable = _direction().with_columns(pl.col("timestamp").dt.strftime("%Y-%m-%dT%H:%M:%S"))

    message = _headline(_predictions(), unjoinable, "fwd_dir_8h")["direction_label_error"]

    assert len(message) <= 300
    assert "\n" not in message, "a newline in a metrics cell reads as a broken row"


def test_the_column_is_declared_text_not_inferred():
    """A fresh registry must type it from the DDL, not from whichever value lands first.

    `_upsert_wide_metrics` auto-adds an unknown metric column and infers REAL from a None -
    which is exactly what a healthy run writes - so leaving it undeclared would put every
    later message in a numeric column.
    """
    import sqlite3

    from case_studies.utils.registry.store import REGISTRY_SCHEMA_SQL

    db = sqlite3.connect(":memory:")
    db.executescript(REGISTRY_SCHEMA_SQL)
    types = {
        row[1]: (row[2] or "").upper()
        for row in db.execute("PRAGMA table_info(prediction_metrics)")
    }

    assert types.get("direction_label_error") == "TEXT"


def test_an_existing_registry_gains_the_column():
    """The migration half, for the registries that already exist."""
    import sqlite3

    from case_studies.utils.registry.store import REGISTRY_SCHEMA_SQL, _migrate_registry

    db = sqlite3.connect(":memory:")
    db.executescript(REGISTRY_SCHEMA_SQL)
    # A registry created before the column existed, built the way it was then rather than
    # by dropping it now: SQLite re-parses the stored CREATE text on DROP COLUMN, so a
    # drop is not a faithful stand-in for never having had it.
    db.execute("DROP TABLE prediction_metrics")
    db.execute(
        "CREATE TABLE prediction_metrics ("
        "  prediction_hash TEXT PRIMARY KEY, computed_at TEXT NOT NULL,"
        "  ic_mean REAL, task_type TEXT, direction_label TEXT)"
    )
    assert "direction_label_error" not in {
        r[1] for r in db.execute("PRAGMA table_info(prediction_metrics)")
    }

    _migrate_registry(db)

    types = {
        row[1]: (row[2] or "").upper()
        for row in db.execute("PRAGMA table_info(prediction_metrics)")
    }
    assert types.get("direction_label_error") == "TEXT"


@pytest.mark.parametrize("unit", ["ms", "us"])
def test_both_clocks_reach_a_computed_value(unit: str):
    """The repaired join, from both sides, so this cannot silently regress into the error."""
    headline = _headline(_predictions(unit=unit), _direction(), "fwd_dir_8h")

    assert headline["direction_label"] == "fwd_dir_8h"
    assert headline["direction_label_error"] is None
