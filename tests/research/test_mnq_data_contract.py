from datetime import UTC, datetime, timezone

import polars as pl
import pytest

from research.mnq_strategy.data_contract import (
    assign_new_york_sessions,
    normalize_mnq_bars,
    normalize_timestamp_columns,
)


def make_session_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 5, 14, 30, tzinfo=UTC),
                datetime(2026, 1, 5, 23, 0, tzinfo=UTC),
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
                datetime(2026, 1, 5, 14, 30, tzinfo=UTC),
                datetime(2026, 1, 5, 14, 35, tzinfo=UTC),
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


def test_normalize_interprets_naive_raw_timestamp_as_utc():
    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 5, 14, 30)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1200],
            "bar_closed": [True],
        }
    )

    result = normalize_mnq_bars(frame)

    assert result["timestamp"][0].strftime("%H:%M %z") == "14:30 +0000"
    assert result["timestamp_ny"][0].strftime("%H:%M %z") == "09:30 -0500"


def test_rejects_naive_timestamp_ny_as_ambiguous():
    with pytest.raises(ValueError, match="timestamp_ny.*timezone-aware|ambiguous"):
        normalize_timestamp_columns(pl.DataFrame({"timestamp_ny": [datetime(2026, 1, 5, 9, 30)]}))


def test_assigns_rth_and_overnight_sessions():
    result = assign_new_york_sessions(make_session_fixture())
    assert result["session_type"].to_list() == ["rth", "overnight"]
    assert result["session_date"].to_list() == [
        datetime(2026, 1, 5).date(),
        datetime(2026, 1, 6).date(),
    ]


@pytest.mark.parametrize(
    ("timestamp", "session_type"),
    [
        (datetime(2026, 1, 5, 14, 29, tzinfo=UTC), "overnight"),
        (datetime(2026, 1, 5, 14, 30, tzinfo=UTC), "rth"),
        (datetime(2026, 1, 5, 20, 59, tzinfo=UTC), "rth"),
        (datetime(2026, 1, 5, 21, 0, tzinfo=UTC), "closed"),
        (datetime(2026, 1, 5, 23, 0, tzinfo=UTC), "overnight"),
    ],
)
def test_session_boundaries(timestamp, session_type):
    result = assign_new_york_sessions(pl.DataFrame({"timestamp": [timestamp]}))
    assert result["session_type"].to_list() == [session_type]


def test_maintenance_interval_is_closed_and_overnight_date_rolls_forward():
    bars = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 5, 20, 55, tzinfo=UTC),
                datetime(2026, 1, 5, 21, 0, tzinfo=UTC),
                datetime(2026, 1, 5, 22, 0, tzinfo=UTC),
                datetime(2026, 1, 6, 4, 29, tzinfo=UTC),
                datetime(2026, 1, 6, 4, 30, tzinfo=UTC),
            ]
        }
    )
    result = assign_new_york_sessions(bars)
    assert result["session_type"].to_list() == ["rth", "closed", "closed", "overnight", "overnight"]
    assert result["session_date"].to_list()[3:] == [datetime(2026, 1, 6).date()] * 2


def test_1800_new_york_bar_rolls_to_next_cme_session_date():
    result = assign_new_york_sessions(
        pl.DataFrame({"timestamp": [datetime(2026, 1, 5, 23, 0, tzinfo=UTC)]})
    )

    assert result["timestamp_ny"][0].strftime("%Y-%m-%d %H:%M") == "2026-01-05 18:00"
    assert result["session_date"][0] == datetime(2026, 1, 6).date()


def test_dst_conversion_uses_new_york_rules():
    bars = pl.DataFrame({"timestamp": [datetime(2026, 3, 8, 13, 30, tzinfo=UTC)]})
    result = assign_new_york_sessions(bars)
    assert result["timestamp_ny"][0].strftime("%Y-%m-%d %H:%M %z") == "2026-03-08 09:30 -0400"


@pytest.mark.parametrize(
    "bad_timestamps",
    [
        [datetime(2026, 1, 5, 14, 30, 1, tzinfo=UTC)],
        [datetime(2026, 1, 5, 14, 31, tzinfo=UTC)],
        [
            datetime(2026, 1, 5, 14, 30, tzinfo=UTC),
            datetime(2026, 1, 5, 14, 37, tzinfo=UTC),
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


def test_rejects_valid_aligned_ten_minute_gap():
    frame = make_session_fixture().with_columns(
        pl.Series(
            "timestamp",
            [
                datetime(2026, 1, 5, 14, 30, tzinfo=UTC),
                datetime(2026, 1, 5, 14, 40, tzinfo=UTC),
            ],
        ),
        pl.lit(True).alias("bar_closed"),
    )
    with pytest.raises(ValueError, match="cadence"):
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


@pytest.mark.parametrize("bar_closed", [[2, 0], ["true", "false"]])
def test_rejects_non_boolean_closed_status(bar_closed):
    frame = make_session_fixture().with_columns(pl.Series("bar_closed", bar_closed))
    with pytest.raises(ValueError, match="explicit boolean"):
        normalize_mnq_bars(frame)
