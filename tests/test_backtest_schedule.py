"""Tests for calendar-aware backtest schedule resolution and execution delay mapping.

These tests validate the fixes for:
- Finding 1: Calendar-named cadences must use actual calendar dates, not elapsed time
- Finding 2: Execution delay mapping must be explicit, not substring-based
- Finding 3: Session enforcement must be enabled for CME
- Finding 4: Vectorized path must use resolved schedule, not gather_every
"""

from datetime import datetime

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# resolve_rebalance_timestamps tests
# ---------------------------------------------------------------------------


def _make_weekday_series(start: str, end: str) -> pl.Series:
    """Create a daily timestamp series (weekdays only, simulating trading days)."""
    dates = pl.date_range(
        pl.lit(datetime.strptime(start, "%Y-%m-%d")),
        pl.lit(datetime.strptime(end, "%Y-%m-%d")),
        interval="1d",
        eager=True,
    )
    # Filter to weekdays (Mon=1..Fri=5 in Polars)
    df = pl.DataFrame({"ts": dates}).filter(pl.col("ts").dt.weekday() <= 5)
    return df["ts"].sort()


class TestResolveRebalanceTimestamps:
    """Test calendar-aware schedule resolution."""

    def test_monthly_month_end_returns_actual_month_ends(self):
        """ETF scenario: monthly_month_end must return last session of each month."""
        from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

        ts = _make_weekday_series("2016-01-01", "2016-06-30")
        result = resolve_rebalance_timestamps(ts, "monthly_month_end")
        all_dates = ts.to_list()

        for dt in result.to_list():
            month, year = dt.month, dt.year
            later = [d for d in all_dates if d.year == year and d.month == month and d > dt]
            assert len(later) == 0, f"Month-end {dt} is not the last session in {year}-{month:02d}"

    def test_monthly_month_end_not_every_21_days(self):
        """Regression: month-end should NOT produce evenly-spaced 21-day intervals."""
        from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

        ts = _make_weekday_series("2016-01-01", "2016-12-31")
        result = resolve_rebalance_timestamps(ts, "monthly_month_end")

        assert 11 <= len(result) <= 12

        gaps = result.diff().drop_nulls().dt.total_seconds() / 86400
        gap_values = set(int(g) for g in gaps.to_list())
        assert len(gap_values) > 1, "Month-end gaps should not all be identical"

    def test_weekly_friday_returns_end_of_week(self):
        """CME/SP500 scenario: weekly_friday_close returns last session per ISO week."""
        from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

        ts = _make_weekday_series("2018-08-01", "2018-10-31")
        result = resolve_rebalance_timestamps(ts, "weekly_friday_close")

        # Exclude last week (may be incomplete if date range ends mid-week)
        dates = result.to_list()
        # All complete weeks should end on Friday
        for dt in dates[:-1]:
            assert dt.weekday() == 4, f"Expected Friday, got {dt} (weekday={dt.weekday()})"

    def test_weekly_friday_holiday_fallback(self):
        """If Friday is missing (holiday), should take Thursday of that week."""
        from datetime import date

        from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

        ts = _make_weekday_series("2018-08-27", "2018-09-14")
        # Remove Friday 2018-09-07 (simulate holiday)
        # ts contains date objects, so compare with date
        friday_to_remove = date(2018, 9, 7)
        ts_list = [d for d in ts.to_list() if d != friday_to_remove]
        ts_filtered = pl.Series("ts", ts_list)

        result = resolve_rebalance_timestamps(ts_filtered, "weekly_friday_close")

        # The week of Sep 3-7 should still have a rebalance, but on Thursday Sep 6
        week_36_dates = [dt for dt in result.to_list() if dt.isocalendar()[1] == 36]
        assert len(week_36_dates) == 1
        assert week_36_dates[0] == date(2018, 9, 6), (
            f"Expected Thursday fallback, got {week_36_dates[0]}"
        )

    def test_daily_returns_all_timestamps(self):
        """Daily cadence should return every available timestamp."""
        from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

        ts = _make_weekday_series("2020-01-01", "2020-01-31")
        for cadence in ("daily", "daily_close", "daily_ny_close"):
            result = resolve_rebalance_timestamps(ts, cadence)
            assert len(result) == len(ts), f"{cadence}: expected all {len(ts)} dates"

    def test_eight_hour_returns_all_timestamps(self):
        """8-hour funding cadence should return all timestamps."""
        from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

        dates = pl.datetime_range(
            datetime(2020, 1, 1),
            datetime(2020, 1, 31),
            interval="8h",
            eager=True,
        )
        result = resolve_rebalance_timestamps(dates, "8_hour_funding_aligned")
        assert len(result) == len(dates.unique())

    def test_empty_series(self):
        """Empty input should return empty output."""
        from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

        ts = pl.Series("ts", [], dtype=pl.Datetime("us"))
        result = resolve_rebalance_timestamps(ts, "monthly_month_end")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Execution delay mapping tests
# ---------------------------------------------------------------------------


class TestExecutionDelayMapping:
    """Test explicit execution delay -> ExecutionMode mapping."""

    def test_next_bar_open_maps_to_next_bar(self):
        from ml4t.backtest import ExecutionMode

        from case_studies.utils.backtest_presets import (
            resolve_execution_mode as _resolve_execution_mode,
        )

        assert _resolve_execution_mode("NEXT_BAR_OPEN") == ExecutionMode.NEXT_BAR
        assert _resolve_execution_mode("next_bar_open") == ExecutionMode.NEXT_BAR

    def test_monday_open_maps_to_next_bar(self):
        """Regression: monday_open must NOT fall through to SAME_BAR."""
        from ml4t.backtest import ExecutionMode

        from case_studies.utils.backtest_presets import (
            resolve_execution_mode as _resolve_execution_mode,
        )

        assert _resolve_execution_mode("MONDAY_OPEN") == ExecutionMode.NEXT_BAR
        assert _resolve_execution_mode("monday_open") == ExecutionMode.NEXT_BAR

    def test_1_bar_maps_to_next_bar(self):
        from ml4t.backtest import ExecutionMode

        from case_studies.utils.backtest_presets import (
            resolve_execution_mode as _resolve_execution_mode,
        )

        assert _resolve_execution_mode("1_BAR") == ExecutionMode.NEXT_BAR
        assert _resolve_execution_mode("1_bar") == ExecutionMode.NEXT_BAR

    def test_at_funding_timestamp_maps_to_same_bar(self):
        from ml4t.backtest import ExecutionMode

        from case_studies.utils.backtest_presets import (
            resolve_execution_mode as _resolve_execution_mode,
        )

        assert _resolve_execution_mode("AT_FUNDING_TIMESTAMP") == ExecutionMode.SAME_BAR

    def test_unknown_token_raises_value_error(self):
        """Unknown execution delay must raise, not silently degrade."""
        from case_studies.utils.backtest_presets import (
            resolve_execution_mode as _resolve_execution_mode,
        )

        with pytest.raises(ValueError, match="Unknown execution delay"):
            _resolve_execution_mode("SOME_RANDOM_TOKEN")


# ---------------------------------------------------------------------------
# Vectorized thinning tests
# ---------------------------------------------------------------------------


class TestThinToRebalanceDates:
    """Test that vectorized thinning uses calendar-aware schedule."""

    def test_weekly_friday_keeps_fridays_not_every_5th(self):
        """Regression: sp500_options weekly_friday must keep actual Fridays."""
        from case_studies.utils.backtest_loaders import thin_to_rebalance_dates

        dates = _make_weekday_series("2020-01-01", "2020-02-28")

        preds = pl.DataFrame(
            {
                "timestamp": dates,
                "symbol": ["SPY"] * len(dates),
                "y_score": [0.1] * len(dates),
                "y_true": [0.01] * len(dates),
            }
        )

        # fwd_ret_5d on weekly_friday schedule: step=1 (5d horizon <= 7d gap)
        result = thin_to_rebalance_dates(preds, cadence="weekly_friday", step=1)

        unique_dates = result["timestamp"].unique().sort()
        # Exclude last date (may be incomplete week)
        for dt in unique_dates.to_list()[:-1]:
            assert dt.weekday() in (3, 4), f"Expected Thu/Fri, got {dt} (weekday={dt.weekday()})"

    def test_monthly_month_end_keeps_month_ends(self):
        """Regression: ETFs monthly_month_end must keep actual month-end sessions."""
        from case_studies.utils.backtest_loaders import thin_to_rebalance_dates

        dates = _make_weekday_series("2016-01-01", "2016-06-30")
        all_dates = dates.to_list()

        preds = pl.DataFrame(
            {
                "timestamp": dates,
                "symbol": ["SPY"] * len(dates),
                "y_score": [0.1] * len(dates),
                "y_true": [0.01] * len(dates),
            }
        )

        # fwd_ret_21d on monthly_month_end schedule: step=1 (21d horizon <= 30d gap)
        result = thin_to_rebalance_dates(preds, cadence="monthly_month_end", step=1)

        unique_dates = result["timestamp"].unique().sort()
        assert 5 <= len(unique_dates) <= 6

        for dt in unique_dates.to_list():
            month, year = dt.month, dt.year
            later = [d for d in all_dates if d.year == year and d.month == month and d > dt]
            assert len(later) == 0, f"{dt} is not the last trading day of {year}-{month:02d}"


# ---------------------------------------------------------------------------
# Rebalance-step lookup (declared in each case study's setup.yaml)
# ---------------------------------------------------------------------------


class TestGetRebalanceStep:
    """Verify that per-label thinning steps are read from setup.yaml.

    Replaces the legacy regex-based implementation which silently returned
    step=1 for any label without a digit-unit token (e.g., ``ret_to_expiry``),
    producing 4-5× inflated Sharpe for overlapping-cohort strategies.

    The step is now a design-time constant declared under
    ``labels.rebalance_step`` in each case study's setup.yaml.
    """

    def test_sp500_options_ret_to_expiry_is_5(self):
        """Regression: ret_to_expiry must thin weekly_friday cohorts by 5.

        30-day DTE on a 7-day schedule -> ceil(30/7) = 5. Pre-fix this
        silently returned 1 (overlapping 5-cohort double-counting).
        """
        from case_studies.utils.backtest_loaders import get_rebalance_step

        assert get_rebalance_step("sp500_options", "ret_to_expiry") == 5

    def test_cme_fwd_ret_21d_is_3(self):
        """cme_futures fwd_ret_21d on weekly_friday -> ceil(21/7) = 3."""
        from case_studies.utils.backtest_loaders import get_rebalance_step

        assert get_rebalance_step("cme_futures", "fwd_ret_21d") == 3

    def test_us_firm_fwd_ret_1m_is_1(self):
        """Monthly label on monthly schedule -> step=1."""
        from case_studies.utils.backtest_loaders import get_rebalance_step

        assert get_rebalance_step("us_firm_characteristics", "fwd_ret_1m") == 1

    def test_nasdaq100_fwd_ret_60m_is_4(self):
        """nasdaq100 fwd_ret_60m on 15-minute schedule -> ceil(60/15) = 4."""
        from case_studies.utils.backtest_loaders import get_rebalance_step

        assert get_rebalance_step("nasdaq100_microstructure", "fwd_ret_60m") == 4

    def test_nasdaq100_fwd_ret_5m_is_1(self):
        """Regression: fwd_ret_5m on 15-minute schedule must stay at 1.

        Pre-fix, the regex matched `(5, m)` and the old `n <= 12` branch
        mis-read it as 5 MONTHS, computing step ~10,000 and collapsing
        backtests to a handful of points.
        """
        from case_studies.utils.backtest_loaders import get_rebalance_step

        assert get_rebalance_step("nasdaq100_microstructure", "fwd_ret_5m") == 1

    def test_crypto_fwd_ret_24h_is_3(self):
        """crypto 24h label on 8h schedule -> ceil(24/8) = 3."""
        from case_studies.utils.backtest_loaders import get_rebalance_step

        assert get_rebalance_step("crypto_perps_funding", "fwd_ret_24h") == 3

    def test_unknown_label_raises(self):
        """Unknown label must raise KeyError pointing at setup.yaml."""
        from case_studies.utils.backtest_loaders import get_rebalance_step

        with pytest.raises(KeyError, match="rebalance_step"):
            get_rebalance_step("sp500_options", "fwd_ret_unknown_label")


# ---------------------------------------------------------------------------
# Integration: engine schedule set membership
# ---------------------------------------------------------------------------


class TestEngineScheduleIntegration:
    """Verify the engine path builds correct schedule sets."""

    def test_etf_monthly_schedule_matches_month_ends(self):
        """ETF backtest should rebalance on actual month-end sessions."""
        from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

        dates = _make_weekday_series("2015-12-01", "2016-04-30")
        schedule = resolve_rebalance_timestamps(dates, "monthly_month_end")
        all_dates = dates.to_list()

        for dt in schedule.to_list():
            month, year = dt.month, dt.year
            later = [d for d in all_dates if d.month == month and d.year == year and d > dt]
            assert len(later) == 0, f"Rebalance {dt} is not month-end: later dates {later[:3]}"

    def test_cme_weekly_schedule_matches_fridays(self):
        """CME backtest should rebalance on Friday sessions."""
        from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

        dates = _make_weekday_series("2018-08-01", "2018-10-26")  # End on a Friday
        schedule = resolve_rebalance_timestamps(dates, "weekly_friday_close")

        for dt in schedule.to_list():
            assert dt.weekday() == 4, (
                f"CME rebalance {dt} should be Friday, got weekday={dt.weekday()}"
            )
