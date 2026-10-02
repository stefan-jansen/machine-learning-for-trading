"""`contiguous_run` splits an entity's rows wherever a step is missing.

A shift or a rolling window over an entity's rows counts rows. It counts sessions (or bars)
only while the entity has a row on every one, so a panel with a gap - a halt, a vendor outage,
a ticker the exchange handed to a different company - silently widens every window that spans
it. Taking the window over the entity *and* its run makes a window that would span a gap
restart after it instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from case_studies.utils.feature_engineering import contiguous_run


def _sessions() -> pl.DataFrame:
    # A trades sessions 0-5; B trades 0-1, misses 2-3, trades 4-6 and misses nothing after.
    return pl.DataFrame(
        {
            "symbol": ["A"] * 6 + ["B"] * 5,
            "session": [0, 1, 2, 3, 4, 5, 0, 1, 4, 5, 6],
            "price": [10.0, 11, 12, 13, 14, 15, 20, 21, 40, 42, 44],
        }
    )


def test_an_unbroken_entity_is_one_run() -> None:
    runs = _sessions().with_columns(contiguous_run("session", "symbol"))
    assert runs.filter(pl.col("symbol") == "A")["run"].unique().to_list() == [0]


def test_a_missing_step_starts_a_new_run() -> None:
    runs = _sessions().with_columns(contiguous_run("session", "symbol"))
    assert runs.filter(pl.col("symbol") == "B")["run"].to_list() == [0, 0, 1, 1, 1]


def test_the_run_does_not_depend_on_row_order() -> None:
    frame = _sessions()
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=3)
    expected = frame.with_columns(contiguous_run("session", "symbol"))
    got = shuffled.with_columns(contiguous_run("session", "symbol")).sort("symbol", "session")
    assert got.equals(expected)


def test_a_return_over_the_run_is_null_across_the_gap() -> None:
    frame = _sessions().with_columns(contiguous_run("session", "symbol"))
    ret = frame.with_columns(
        (pl.col("price") / pl.col("price").shift(1).over("symbol", "run") - 1).alias("ret")
    )
    b = ret.filter(pl.col("symbol") == "B")
    # Session 4 follows session 1: the 90% jump is two missing sessions, not one day's move.
    assert b.filter(pl.col("session") == 4)["ret"].item() is None
    assert b.filter(pl.col("session") == 5)["ret"].item() == 42 / 40 - 1


def test_a_timestamp_step_is_a_duration() -> None:
    start = datetime(2022, 3, 1)
    stamps = [start + timedelta(hours=8 * k) for k in (0, 1, 2, 9, 10)]
    frame = pl.DataFrame({"symbol": ["X"] * 5, "timestamp": stamps})
    runs = frame.with_columns(contiguous_run("timestamp", "symbol", size=timedelta(hours=8)))
    assert runs["run"].to_list() == [0, 0, 0, 1, 1]
