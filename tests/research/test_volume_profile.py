from datetime import datetime, timezone

import polars as pl
import pytest

from research.mnq_strategy.volume_profile import (
    attach_previous_rth_profile,
    build_rth_profile,
)


def test_value_area_is_40_percent_and_contains_poc():
    bars = pl.DataFrame(
        {
            "price": [100.0, 100.25, 100.5, 100.75, 101.0],
            "volume": [10, 40, 100, 30, 10],
        }
    )

    profile = build_rth_profile(bars, value_area_fraction=0.40)

    assert profile["value_area_fraction"] == 0.40
    assert profile["poc"] == 100.5
    assert profile["val"] <= profile["poc"] <= profile["vah"]
    assert sum(profile["volume_by_price"].values()) == 190


def test_value_area_expands_one_adjacent_bin_at_a_time():
    bars = pl.DataFrame(
        {
            "price": [99.5, 99.75, 100.0, 100.25, 100.5],
            "volume": [1, 40, 100, 30, 1],
        }
    )

    profile = build_rth_profile(bars, value_area_fraction=0.80)

    assert profile["val"] == 99.75
    assert profile["vah"] == 100.0


def test_lvn_zones_are_contiguous_low_volume_bins():
    bars = pl.DataFrame(
        {
            "price": [100.0, 100.25, 100.5, 100.75, 101.0, 101.25, 101.5, 101.75, 102.0, 102.25],
            "volume": [100.0, 1.0, 1.1, 1.2, 1.3, 100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )

    profile = build_rth_profile(bars)

    assert profile["lvn_zones"] == [
        {"low": 100.25, "high": 100.5},
    ]


def test_attach_uses_only_previous_fully_closed_rth_session():
    bars = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc),
                datetime(2026, 1, 5, 14, 35, tzinfo=timezone.utc),
                datetime(2026, 1, 6, 14, 30, tzinfo=timezone.utc),
                datetime(2026, 1, 6, 14, 35, tzinfo=timezone.utc),
            ],
            "timestamp_ny": [
                datetime(2026, 1, 5, 9, 30),
                datetime(2026, 1, 5, 9, 35),
                datetime(2026, 1, 6, 9, 30),
                datetime(2026, 1, 6, 9, 35),
            ],
            "session_date": [
                datetime(2026, 1, 5).date(),
                datetime(2026, 1, 5).date(),
                datetime(2026, 1, 6).date(),
                datetime(2026, 1, 6).date(),
            ],
            "session_type": ["rth"] * 4,
            "open": [100.0, 100.0, 200.0, 200.0],
            "high": [100.25, 100.25, 200.25, 200.25],
            "low": [99.75, 99.75, 199.75, 199.75],
            "close": [100.0, 100.25, 200.0, 200.25],
            "volume": [100, 10, 1000, 1000],
            "bar_closed": [True] * 4,
        }
    )

    result = attach_previous_rth_profile(bars)

    assert result["previous_profile_date"].to_list() == [None, None, datetime(2026, 1, 5).date(), datetime(2026, 1, 5).date()]
    assert result["previous_poc"].to_list() == [None, None, 100.0, 100.0]


def test_current_day_later_rth_bars_cannot_change_attached_profile():
    bars = pl.DataFrame(
        {
            "session_date": [
                datetime(2026, 1, 5).date(),
                datetime(2026, 1, 6).date(),
                datetime(2026, 1, 6).date(),
            ],
            "session_type": ["rth"] * 3,
            "close": [100.0, 200.0, 300.0],
            "volume": [100, 1, 10000],
            "bar_closed": [True] * 3,
        }
    )

    result = attach_previous_rth_profile(bars)

    assert result["previous_poc"].to_list() == [None, 100.0, 100.0]


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_rejects_invalid_value_area_fraction(fraction):
    bars = pl.DataFrame({"price": [100.0], "volume": [1]})
    with pytest.raises(ValueError, match="value_area_fraction"):
        build_rth_profile(bars, fraction)
