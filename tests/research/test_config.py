"""Tests for the reproducible MNQ strategy configuration and fixtures."""

import hashlib
import json
from dataclasses import FrozenInstanceError
from datetime import date

import polars as pl
import pytest

from research.mnq_strategy.config import StrategyConfig
from research.mnq_strategy.fixtures import (
    make_10am_confirmation_fixture,
    make_confirmed_signal_fixture,
    make_cost_fixture,
    make_daily_guard_fixture,
    make_lvn_retest_fixture,
    make_multi_month_fixture,
    make_overlapping_signals_fixture,
    make_profile_attachment_fixture,
    make_rejection_fixture,
    make_stop_target_fixture,
)
from research.mnq_strategy.signals import (
    detect_10am_confirmation,
    detect_lvn_break_retest,
    detect_midnight_rejection,
)
from research.mnq_strategy.volume_profile import attach_previous_rth_profile

CANONICAL_COLUMNS = {
    "timestamp",
    "timestamp_ny",
    "session_date",
    "session_type",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bar_closed",
}


def test_default_config_matches_approved_spec():
    config = StrategyConfig()

    assert config.instrument == "MNQ"
    assert config.timezone == "America/New_York"
    assert config.bar_minutes == 5
    assert config.value_area_fraction == 0.40
    assert config.min_contracts == 4
    assert config.max_contracts == 10
    assert config.max_trade_risk == 250.0
    assert config.daily_stop == 400.0
    assert config.max_consecutive_losses == 2
    assert config.point_value == 2.0
    assert config.cost_model.commission_per_contract == 1.50
    assert config.cost_model.slippage_points == 0.50


def test_config_contains_explicit_signal_backtest_and_session_defaults():
    config = StrategyConfig()

    assert {
        "rejection_wick_ratio": config.rejection_wick_ratio,
        "rejection_close_pct": config.rejection_close_pct,
        "momentum_body_range": config.momentum_body_range,
        "momentum_close_pct": config.momentum_close_pct,
        "momentum_lookback": config.momentum_lookback,
        "momentum_multiplier": config.momentum_multiplier,
        "lvn_percentile": config.lvn_percentile,
    } == {
        "rejection_wick_ratio": 2.0,
        "rejection_close_pct": 0.30,
        "momentum_body_range": 0.60,
        "momentum_close_pct": 0.80,
        "momentum_lookback": 6,
        "momentum_multiplier": 1.5,
        "lvn_percentile": 0.25,
    }
    assert config.stop_points == 10.0
    assert config.target_points == 20.0
    assert config.session_boundaries == {
        "rth_start": "09:30",
        "rth_end": "16:00",
        "maintenance_start": "16:00",
        "maintenance_end": "18:00",
        "overnight_start": "18:00",
    }


def test_config_serialization_is_json_compatible_and_has_nested_costs():
    serialized = StrategyConfig().to_dict()

    assert json.loads(json.dumps(serialized)) == serialized
    assert serialized["cost_model"] == {
        "commission_per_contract": 1.50,
        "slippage_points": 0.50,
    }
    assert serialized["session_boundaries"]["rth_start"] == "09:30"
    assert serialized["stop_points"] == 10.0
    assert serialized["target_points"] == 20.0


def test_config_hash_uses_canonical_sorted_json():
    config = StrategyConfig()
    canonical = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert config.config_hash == expected
    assert config.config_hash == StrategyConfig().config_hash
    assert config.config_hash != StrategyConfig(target_points=21.0).config_hash


def test_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        StrategyConfig().bar_minutes = 1


@pytest.mark.parametrize(
    "fixture_factory",
    [
        make_confirmed_signal_fixture,
        make_overlapping_signals_fixture,
        make_multi_month_fixture,
    ],
)
def test_fixtures_are_fresh_polars_frames_with_canonical_columns(fixture_factory):
    first = fixture_factory()
    second = fixture_factory()

    assert isinstance(first, pl.DataFrame)
    assert first is not second
    assert first.equals(second)
    assert CANONICAL_COLUMNS.issubset(first.columns)
    assert first.schema["bar_closed"] == pl.Boolean
    assert first.schema["signal"] == pl.Boolean
    assert first.schema["direction"] == pl.Utf8
    assert first.schema["signal_type"] == pl.Utf8


def test_required_fixtures_are_chronological_and_entry_times_are_present():
    for fixture_factory in (
        make_confirmed_signal_fixture,
        make_overlapping_signals_fixture,
        make_multi_month_fixture,
    ):
        frame = fixture_factory()
        timestamps = frame["timestamp_ny"].to_list()

        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == frame.height
        signal_rows = frame.filter(pl.col("signal"))
        assert signal_rows["entry_time"].null_count() == 0
        assert set(signal_rows["entry_time"].to_list()).issubset(set(timestamps))


def test_market_fixtures_use_explicit_new_york_and_utc_timestamps():
    frame = make_confirmed_signal_fixture()

    assert frame.schema["timestamp"].time_zone == "UTC"
    assert frame.schema["timestamp_ny"].time_zone == "America/New_York"
    assert frame["timestamp_ny"][0].strftime("%H:%M") == "10:00"
    assert frame["timestamp"][0].strftime("%H:%M") == "15:00"


def test_confirmed_signal_fixture_places_entry_on_next_bar_open():
    frame = make_confirmed_signal_fixture()
    signal = frame.filter(pl.col("signal"))

    assert signal.height == 1
    assert signal["bar_closed"][0] is True
    assert signal["direction"][0] == "long"
    assert signal["signal_type"][0] == "10am"

    next_bar = frame.filter(pl.col("timestamp_ny") == signal["entry_time"][0])
    assert next_bar.height == 1
    assert next_bar["open"][0] == 100.25


def test_overlapping_fixture_exposes_overlapping_eligible_entry_windows():
    signals = make_overlapping_signals_fixture().filter(pl.col("signal")).sort("timestamp_ny")

    assert signals.height >= 2
    assert signals["entry_window_start"][1] < signals["entry_window_end"][0]


def test_multi_month_fixture_is_chronological_and_has_a_test_window_signal():
    frame = make_multi_month_fixture()
    dates = frame["session_date"].to_list()

    assert dates == sorted(dates)
    assert min(dates) >= date(2024, 1, 1)
    assert max(dates) <= date(2024, 12, 31)
    assert frame.filter(pl.col("signal") & (pl.col("session_date") >= date(2024, 7, 1))).height >= 1


def test_supplementary_fixtures_feed_existing_signal_and_profile_contracts():
    assert detect_midnight_rejection(make_rejection_fixture()).filter(pl.col("signal")).height == 1
    assert detect_lvn_break_retest(make_lvn_retest_fixture()).filter(pl.col("signal")).height == 1
    assert (
        detect_10am_confirmation(make_10am_confirmation_fixture()).filter(pl.col("signal")).height
        == 1
    )
    assert (
        attach_previous_rth_profile(make_profile_attachment_fixture())["previous_poc"][1] == 100.25
    )


def test_supplementary_execution_fixtures_include_explicit_trade_fields():
    stop_target = make_stop_target_fixture().filter(pl.col("signal"))
    costs = make_cost_fixture().filter(pl.col("signal"))

    assert stop_target.height == 1
    assert stop_target["direction"][0] == "long"
    assert stop_target["signal_type"][0] == "10am"
    assert stop_target["entry_time"][0] is not None
    assert costs.height == 1
    assert make_daily_guard_fixture()["net_pnl"].to_list() == [-200.0, -200.0, 100.0]
