"""`load_us_equities(lazy=True)` has to hand back the same panel, not merely a plan.

`04_model_based_features` reads six of the archive's fourteen columns and asserts its own
digest of them against the one `02_labels` recorded. Deferring the collect so the projection
reaches the parquet scan is only safe if the deferred plan applies the rename, the date cast
and the universe reduction the eager path applied - a lazy frame that skipped any of them
would still collect, and would collect something else.
"""

from __future__ import annotations

import polars as pl
import pytest

from data.equities import loader

ARCHIVE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ex-dividend",
    "split_ratio",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
]
# The six `04_model_based_features` reads, and the only ones its digest covers.
READ_COLS = ["symbol", "timestamp", "close", "volume", "adj_close", "adj_volume"]


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A panel in the archive's own schema: `ticker`/`date`, Datetime, unequal histories.

    The histories are unequal on purpose - `max_symbols` reduces by observation count, so a
    panel of equal-length symbols cannot tell a working reduction from one that returns
    whatever order the group-by produced.
    """
    rows = {"AAA": 40, "BBB": 25, "CCC": 10}
    frames = []
    for i, (ticker, n) in enumerate(rows.items()):
        frames.append(
            pl.DataFrame(
                {
                    "ticker": [ticker] * n,
                    "date": pl.datetime_range(
                        pl.datetime(2020, 1, 1),
                        pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                        "1d",
                        eager=True,
                    ),
                    **{c: [float(i * 100 + j) for j in range(n)] for c in ARCHIVE_COLUMNS},
                }
            )
        )
    path = tmp_path / "equities" / "market" / "us_equities"
    path.mkdir(parents=True)
    pl.concat(frames).write_parquet(path / "us_equities.parquet")
    monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)
    return rows


def test_the_deferred_plan_collects_the_eager_panel(archive):
    """Same rows, same columns, same values - the rename and the date cast included."""
    eager = loader.load_us_equities()
    deferred = loader.load_us_equities(lazy=True)

    assert isinstance(deferred, pl.LazyFrame)
    assert isinstance(eager, pl.DataFrame)
    assert deferred.collect().equals(eager)
    # The rename and the cast are what a caller's `select` depends on: it names `symbol` and
    # `timestamp`, neither of which the archive carries.
    assert eager["timestamp"].dtype == pl.Date
    assert {"symbol", "timestamp"} <= set(eager.columns)


def test_a_projection_through_the_plan_is_the_projection_of_the_panel(archive):
    """The claim 04's digest rests on: pushing the projection into the scan moves no value."""
    pushed = loader.load_us_equities(lazy=True).select(READ_COLS).collect()
    materialized = loader.load_us_equities().select(READ_COLS)

    assert pushed.equals(materialized)
    assert pushed.width == 6
    # And the eight columns it never reads really were on the panel it did not read them from.
    assert loader.load_us_equities().width == 14


def test_the_reduction_picks_the_same_universe_on_either_path(archive):
    """`max_symbols` moved ahead of the collect, so it now filters a plan rather than a frame."""
    eager = loader.load_us_equities(max_symbols=2)
    deferred = loader.load_us_equities(max_symbols=2, lazy=True).collect()

    assert sorted(eager["symbol"].unique()) == ["AAA", "BBB"]
    assert deferred.equals(eager)
    # CCC is the shortest history, so a reduction that ignored observation counts would be
    # free to keep it.
    assert "CCC" not in deferred["symbol"].unique()


def test_a_projected_plan_still_reduces_by_the_full_panels_counts(archive):
    """A caller projects before reducing; the counts must not depend on which columns survive."""
    projected = (
        loader.load_us_equities(lazy=True)
        .select(READ_COLS)
        .collect()
        .filter(pl.col("symbol") != "")
    )
    reduced = loader.load_us_equities(max_symbols=2, lazy=True).select(READ_COLS).collect()

    assert sorted(reduced["symbol"].unique()) == ["AAA", "BBB"]
    assert reduced.equals(projected.filter(pl.col("symbol").is_in(["AAA", "BBB"])))
