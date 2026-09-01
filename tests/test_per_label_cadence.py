"""Per-label decision cadence: a case study whose labels span horizons trades each on its own grid.

A single ``decision.cadence`` forced every label of a case study onto one rebalance schedule, so a
5-day forecast in a monthly case study was held for 21 sessions - the label measured one thing and
the backtest traded another. ``decision.cadence_by_label`` overrides the default per label; a label
with no entry is unchanged, which is what keeps every already-registered spec identical.
"""

from datetime import date, datetime

import polars as pl
import pytest

from case_studies.utils.backtest_loaders import BacktestConfig, resolve_rebalance_timestamps


def _weekdays(start: str, end: str) -> pl.Series:
    dates = pl.date_range(
        pl.lit(datetime.strptime(start, "%Y-%m-%d")),
        pl.lit(datetime.strptime(end, "%Y-%m-%d")),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame({"ts": dates}).filter(pl.col("ts").dt.weekday() <= 5)["ts"].sort()


def _config(**over) -> BacktestConfig:
    base = dict(
        case_study_id="etfs",
        primary_label="fwd_ret_21d",
        label_buffer="",
        calendar="NYSE",
        cadence="monthly_month_end",
        execution_delay="next_bar_open",
        commission_bps=1.0,
        slippage_bps=1.0,
        costs_class="material",
        long_short=False,
        holdout_start="2024-01-01",
        holdout_end="2024-12-31",
        n_splits=8,
        raw_costs={},
    )
    base.update(over)
    return BacktestConfig(**base)


class TestCadenceFor:
    def test_no_overrides_every_label_gets_the_case_study_cadence(self):
        cfg = _config()
        assert cfg.cadence_for("fwd_ret_21d") == "monthly_month_end"
        assert cfg.cadence_for("fwd_ret_5d") == "monthly_month_end"
        assert cfg.cadence_for(None) == "monthly_month_end"

    def test_declared_label_gets_its_own_cadence(self):
        cfg = _config(cadence_by_label={"fwd_ret_5d": "weekly_friday_close"})
        assert cfg.cadence_for("fwd_ret_5d") == "weekly_friday_close"
        assert cfg.cadence_for("fwd_ret_21d") == "monthly_month_end"

    def test_unlabelled_caller_gets_the_default_not_an_override(self):
        """An override must never leak into a call that did not name a label."""
        cfg = _config(cadence_by_label={"fwd_ret_5d": "weekly_friday_close"})
        assert cfg.cadence_for() == "monthly_month_end"
        assert cfg.cadence_for("") == "monthly_month_end"


class TestBiweeklySchedule:
    def test_biweekly_is_half_of_weekly(self):
        ts = _weekdays("2020-01-01", "2020-12-31")
        weekly = resolve_rebalance_timestamps(ts, "weekly_friday_close")
        biweekly = resolve_rebalance_timestamps(ts, "biweekly")
        assert 0 < len(biweekly) < len(weekly)
        assert abs(len(biweekly) - len(weekly) / 2) <= 1

    def test_biweekly_dates_are_a_subset_of_weekly_dates(self):
        ts = _weekdays("2020-01-01", "2020-12-31")
        weekly = set(resolve_rebalance_timestamps(ts, "weekly_friday_close").to_list())
        biweekly = set(resolve_rebalance_timestamps(ts, "biweekly").to_list())
        assert biweekly <= weekly

    def test_biweekly_phase_does_not_depend_on_the_window_loaded(self):
        """Two callers loading different windows must land on the same calendar dates.

        Parity taken on position in the resolved list would interleave the two: a split starting
        one week later would rebalance on exactly the weeks the other skipped, and the same
        strategy would register two different schedules depending on how much history was read.
        """
        full = resolve_rebalance_timestamps(_weekdays("2020-01-01", "2020-12-31"), "biweekly")
        late = resolve_rebalance_timestamps(_weekdays("2020-04-06", "2020-12-31"), "biweekly")
        overlap = [t for t in full.to_list() if t >= date(2020, 4, 6)]
        assert late.to_list() == overlap

    def test_gaps_stay_fourteen_days_across_a_52_week_year_boundary(self):
        """2020 is a 53-week ISO year and 2021 a 52-week one.

        A counter of the form ``iso_year * 53 + iso_week`` jumps by two at every 52-week
        boundary, which flips the parity and lands a 7-day or 21-day gap there. Elapsed weeks
        since a fixed Monday cannot do that.
        """
        ts = _weekdays("2020-10-01", "2021-04-01")
        dates = resolve_rebalance_timestamps(ts, "biweekly").to_list()
        gaps = [(b - a).days for a, b in zip(dates, dates[1:], strict=False)]
        assert gaps, "no rebalance dates resolved"
        assert max(gaps) <= 21, gaps
        assert min(gaps) >= 10, gaps

    def test_every_gap_is_two_calendar_weeks_apart_in_monday_terms(self):
        """The invariant: consecutive rebalances sit exactly 2 ISO weeks apart."""
        from datetime import timedelta

        ts = _weekdays("2019-01-01", "2022-12-31")
        dates = resolve_rebalance_timestamps(ts, "biweekly").to_list()
        mondays = [d - timedelta(days=d.weekday()) for d in dates]
        deltas = {(b - a).days for a, b in zip(mondays, mondays[1:], strict=False)}
        assert deltas == {14}, sorted(deltas)

    def test_gaps_are_about_fourteen_days(self):
        """Two weeks apart in Mondays, but the sessions inside them move.

        The rebalance is the last available session of its week, so a holiday-shortened week
        ends on Thursday and the gap to the next Friday is 13 days rather than 14. The exact
        invariant is the Monday one above; this bounds how far a real calendar can stretch it.
        """
        ts = _weekdays("2020-01-01", "2020-12-31")
        dates = resolve_rebalance_timestamps(ts, "biweekly").to_list()
        gaps = {(b - a).days for a, b in zip(dates, dates[1:], strict=False)}
        assert min(gaps) >= 10 and max(gaps) <= 18, sorted(gaps)


class TestBuildSpecRefusesAnUnlabelledCall:
    def test_declaring_overrides_makes_label_required(self):
        """Silently using the default cadence would register a spec on the wrong grid."""
        from case_studies.utils.backtest_presets import build_backtest_spec

        cfg = _config(cadence_by_label={"fwd_ret_5d": "weekly_friday_close"})
        with pytest.raises(ValueError, match="cadence_by_label"):
            build_backtest_spec(
                "etfs",
                cfg,
                prices=pl.DataFrame({"symbol": ["A"], "timestamp": [datetime(2020, 1, 2)]}),
                prediction_hash="h",
                initial_cash=100_000.0,
                signal={"method": "equal_weight_top_k", "top_k": 2},
            )

    def test_no_overrides_means_no_new_requirement(self):
        """Every case study that declares nothing keeps calling exactly as it did."""
        from case_studies.utils.backtest_presets import build_backtest_spec

        cfg = _config()
        spec = build_backtest_spec(
            "etfs",
            cfg,
            prices=pl.DataFrame(
                {
                    "symbol": ["A", "A"],
                    "timestamp": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
                    "close": [1.0, 1.1],
                }
            ),
            prediction_hash="h",
            initial_cash=100_000.0,
            signal={"method": "equal_weight_top_k", "top_k": 2},
        )
        assert spec["strategy"]["rebalance"]["cadence"] == "monthly_month_end"


class TestTwoLabelsTradeDifferentGrids:
    def test_the_cadence_reaches_the_spec_and_separates_the_two(self):
        """The point of the feature: same signal, two labels, two schedules, two identities."""
        from case_studies.utils.backtest_presets import build_backtest_spec

        cfg = _config(cadence_by_label={"fwd_ret_5d": "weekly_friday_close"})
        prices = pl.DataFrame(
            {
                "symbol": ["A", "A"],
                "timestamp": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
                "close": [1.0, 1.1],
            }
        )
        kw = dict(
            prices=prices,
            prediction_hash="h",
            initial_cash=100_000.0,
            signal={"method": "equal_weight_top_k", "top_k": 2},
        )
        long_spec = build_backtest_spec("etfs", cfg, label="fwd_ret_21d", **kw)
        short_spec = build_backtest_spec("etfs", cfg, label="fwd_ret_5d", **kw)
        assert long_spec["strategy"]["rebalance"]["cadence"] == "monthly_month_end"
        assert short_spec["strategy"]["rebalance"]["cadence"] == "weekly_friday_close"
        assert long_spec["strategy"]["rebalance"] != short_spec["strategy"]["rebalance"]


class TestEffectiveScheduleAfterThinning:
    """cadence and rebalance_step compose; the pair is what decides the holding period.

    rebalance_step exists because the cadence used to be one grid for every label: a 10-session
    label on a weekly grid had to skip every other decision or its holding periods overlapped.
    Now that the label has its own grid the step is ceil(horizon / cadence) against THAT grid,
    and leaving the old value in place thins an already-thinned schedule. This is the check that
    catches it, on the real setup.yaml rather than a fixture.
    """

    def test_seoa_ten_day_labels_end_up_ten_sessions_apart(self):
        from case_studies.utils.backtest_loaders import (
            get_backtest_config,
            get_rebalance_step,
            resolve_rebalance_timestamps,
        )

        cfg = get_backtest_config("sp500_equity_option_analytics")
        ts = _weekdays("2020-01-01", "2020-12-31")
        for label, horizon_sessions in [("fwd_ret_10d", 10), ("fwd_dir_10d", 10)]:
            schedule = resolve_rebalance_timestamps(ts, cfg.cadence_for(label))
            step = get_rebalance_step("sp500_equity_option_analytics", label)
            traded = schedule.to_list()[::step]
            gaps = [(b - a).days for a, b in zip(traded, traded[1:], strict=False)]
            # 10 sessions is about 14 calendar days; a doubled step would put this near 28.
            assert max(gaps) <= 18, (label, step, sorted(set(gaps)))
            assert horizon_sessions <= max(gaps) <= 18

    def test_seoa_five_day_labels_are_unchanged_at_weekly(self):
        from case_studies.utils.backtest_loaders import get_backtest_config, get_rebalance_step

        cfg = get_backtest_config("sp500_equity_option_analytics")
        for label in ["fwd_ret_5d", "fwd_ret_risk_adj_5d", "fwd_dir_5d"]:
            assert cfg.cadence_for(label) == "weekly_friday_close"
            assert get_rebalance_step("sp500_equity_option_analytics", label) == 1

    def test_etfs_five_day_label_trades_weekly_with_step_one(self):
        from case_studies.utils.backtest_loaders import get_backtest_config, get_rebalance_step

        cfg = get_backtest_config("etfs")
        assert cfg.cadence_for("fwd_ret_5d") == "weekly_friday_close"
        assert cfg.cadence_for("fwd_ret_21d") == "monthly_month_end"
        assert get_rebalance_step("etfs", "fwd_ret_5d") == 1
