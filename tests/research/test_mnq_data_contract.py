from datetime import datetime, timezone

import pytest

import polars as pl

from research.mnq_strategy.data_contract import (
    assign_new_york_sessions,
    normalize_mnq_bars,
)


def make_session_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc),
                datetime(2026, 1, 5, 23, 0, tzinfo=timezone.utc),
            ],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1200, 800],
        }
    )


def test_normalize_uses_new_york_and_rejects_intrabar_rows():
    frame = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc),
                datetime(2026, 1, 5, 14, 35, tzinfo=timezone.utc),
            ],
            "open": [100.0, 100.5],
            "high": [101.0, 101.5],
            "low": [99.0, 100.0],
            "close": [100.5, 101.0],
            "volume": [1200, 1000],
            "bar_closed": [True, False],
        }
    )
    result = normalize_mnq_bars(frame)
    assert result["timestamp_ny"][0].strftime("%Y-%m-%d %H:%M") == "2026-01-05 09:30"
    assert result["bar_closed"].to_list() == [True]


def test_assigns_rth_and_overnight_sessions():
    result = assign_new_york_sessions(make_session_fixture())
    assert result["session_type"].to_list() == ["rth", "overnight"]
    assert result["session_date"].to_list() == [datetime(2026, 1, 5).date(), datetime(2026, 1, 6).date()]


@pytest.mark.parametrize(
    ("timestamp", "session_type"),
    [
        (datetime(2026, 1, 5, 14, 29, tzinfo=timezone.utc), "overnight"),
        (datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc), "rth"),
        (datetime(2026, 1, 5, 20, 59, tzinfo=timezone.utc), "rth"),
        (datetime(2026, 1, 5, 21, 0, tzinfo=timezone.utc), "closed"),
        (datetime(2026, 1, 5, 23, 0, tzinfo=timezone.utc), "overnight"),
    ],
)
def test_session_boundaries(timestamp, session_type):
    result = assign_new_york_sessions(pl.DataFrame({"timestamp": [timestamp]}))
    assert result["session_type"].to_list() == [session_type]


def test_maintenance_interval_is_closed_and_overnight_date_rolls_forward():
    bars = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 5, 20, 55, tzinfo=timezone.utc),
                datetime(2026, 1, 5, 21, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 5, 22, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 6, 4, 29, tzinfo=timezone.utc),
                datetime(2026, 1, 6, 4, 30, tzinfo=timezone.utc),
            ]
        }
    )
    result = assign_new_york_sessions(bars)
    assert result["session_type"].to_list() == ["rth", "closed", "closed", "overnight", "overnight"]
    assert result["session_date"].to_list()[3:] == [datetime(2026, 1, 6).date()] * 2


def test_dst_conversion_uses_new_york_rules():
    bars = pl.DataFrame({"timestamp": [datetime(2026, 3, 8, 13, 30, tzinfo=timezone.utc)]})
    result = assign_new_york_sessions(bars)
    assert result["timestamp_ny"][0].strftime("%Y-%m-%d %H:%M %z") == "2026-03-08 09:30 -0400"


@pytest.mark.parametrize(
    "bad_timestamps",
    [
        [datetime(2026, 1, 5, 14, 30, 1, tzinfo=timezone.utc)],
        [datetime(2026, 1, 5, 14, 31, tzinfo=timezone.utc)],
        [
            datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc),
            datetime(2026, 1, 5, 14, 37, tzinfo=timezone.utc),
        ],
    ],
)
def test_rejects_non_five_minute_input(bad_timestamps):
    frame = pl.DataFrame(
        {
            "timestamp": bad_timestamps,
            "open": [100.0] * len(bad_timestamps),
            "high": [101.0] * len(bad_timestamps),
            "low": [99.0] * len(bad_timestamps),
            "close": [100.5] * len(bad_timestamps),
            "volume": [1200] * len(bad_timestamps),
            "bar_closed": [True] * len(bad_timestamps),
        }
    )
    with pytest.raises(ValueError, match="5-minute|cadence"):
        normalize_mnq_bars(frame)


def test_rejects_missing_columns_closed_status_and_invalid_timestamp():
    frame = make_session_fixture()
    with pytest.raises(ValueError, match="bar_closed"):
        normalize_mnq_bars(frame)
    with pytest.raises(ValueError, match="required input columns missing"):
        normalize_mnq_bars(frame.drop("volume").with_columns(pl.lit(True).alias("bar_closed")))
    invalid = frame.with_columns(pl.Series("timestamp", ["not-a-date", "still-not-a-date"]))
    invalid = invalid.with_columns(pl.lit(True).alias("bar_closed"))
    with pytest.raises(ValueError, match="invalid datetime"):
        normalize_mnq_bars(invalid)
