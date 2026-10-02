"""Roll factors compare both contracts at one moment (data/futures/roll.py)."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from data.futures.roll import (
    DIFFERENT_TIMES,
    SAME_HOUR_AT_FIRST_NEW_BAR,
    SAME_HOUR_AT_LAST_OLD_BAR,
    roll_factors,
)
from utils import ML4T_DATA_PATH

FRI = datetime(2023, 9, 15, 13, tzinfo=UTC)
SUN = datetime(2023, 9, 17, 22, tzinfo=UTC)
MON = datetime(2023, 9, 18, 13, tzinfo=UTC)


def _bars(rows: list[tuple[int, datetime, int, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["tenor", "timestamp", "instrument_id", "open", "close"],
        orient="row",
    ).with_columns(pl.lit("ES").alias("product"))


def test_a_weekend_roll_takes_both_contracts_on_the_old_contracts_last_hour() -> None:
    # U trades 100 on Friday; Z trades 101 in the same hour and opens 99 on Sunday, so Z
    # lost 2% over the weekend. The factor must be 101 / 100, not 99 / 100.
    hourly = _bars(
        [
            (0, FRI, 1, 100.0, 100.0),
            (1, FRI, 2, 101.0, 101.0),
            (0, SUN, 2, 99.0, 98.98),
        ]
    )
    roll = roll_factors(hourly).filter(pl.col("tenor") == 0).row(0, named=True)

    assert roll["ratio"] == pytest.approx(101.0 / 100.0)
    assert roll["basis"] == SAME_HOUR_AT_LAST_OLD_BAR
    assert roll["pair_timestamp"] == FRI
    # The adjusted return across the roll is the loss Z's holder actually took.
    adjusted_friday_close = 100.0 * roll["ratio"]
    assert 98.98 / adjusted_friday_close - 1 == pytest.approx(98.98 / 101.0 - 1)


def test_the_new_contracts_first_hour_is_used_when_it_alone_has_both_quotes() -> None:
    hourly = _bars(
        [
            (0, FRI, 1, 100.0, 100.0),
            (0, SUN, 2, 99.0, 99.5),
            (1, SUN, 1, 98.4, 98.5),
        ]
    )
    roll = roll_factors(hourly).filter(pl.col("tenor") == 0).row(0, named=True)

    assert roll["ratio"] == pytest.approx(99.5 / 98.5)
    assert roll["basis"] == SAME_HOUR_AT_FIRST_NEW_BAR
    # The return across the roll is then the old contract's own: 98.5 / 100.
    assert 99.5 / (100.0 * roll["ratio"]) - 1 == pytest.approx(98.5 / 100.0 - 1)


def test_a_roll_without_a_same_hour_pair_is_labelled_rather_than_passed_off() -> None:
    hourly = _bars([(0, FRI, 1, 100.0, 100.0), (0, MON, 2, 99.0, 99.5)])
    roll = roll_factors(hourly).row(0, named=True)

    assert roll["basis"] == DIFFERENT_TIMES
    assert roll["pair_timestamp"] is None
    assert roll["gap_hours"] == pytest.approx(72.0)


def test_a_series_that_never_switches_has_no_roll() -> None:
    hourly = _bars([(0, FRI, 1, 100.0, 100.0), (0, SUN, 1, 99.0, 99.5)])
    assert roll_factors(hourly).is_empty()


HOURLY = ML4T_DATA_PATH / "futures" / "market" / "continuous" / "hourly" / "product=ES"


@pytest.mark.skipif(not HOURLY.exists(), reason="ES hourly continuous data not present")
def test_es_september_2023_roll_uses_both_contracts_at_friday_0800_ct() -> None:
    hourly = pl.read_parquet(HOURLY / "**/*.parquet", hive_partitioning=True).with_columns(
        pl.lit("ES").alias("product")
    )
    roll = (
        roll_factors(hourly)
        .filter(
            (pl.col("tenor") == 0) & (pl.col("timestamp").dt.date() == datetime(2023, 9, 17).date())
        )
        .row(0, named=True)
    )
    assert roll["pair_timestamp"] == datetime(2023, 9, 15, 13, tzinfo=UTC)
    assert roll["ratio"] == pytest.approx(4529.00 / 4482.75)
