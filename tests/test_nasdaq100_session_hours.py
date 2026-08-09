"""The regular-hours filter in ``data/equities/loader.py`` follows the exchange calendar.

A fixed 09:30-16:00 clock bound is right on 250 sessions a year and wrong on two:
NYSE and NASDAQ close at 13:00 ET the day after Thanksgiving and on Christmas Eve, and
the afternoon prints on those dates entered every downstream notebook as ordinary bars.
`nasdaq100_microstructure` reads this loader at stages 01 through 04.

The bars here are constructed rather than sampled, so the test states the shape it is
about - a half day, a full day, a holiday - instead of depending on the archive.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from data.equities.loader import _exchange_sessions, _filter_to_exchange_sessions

HALF_DAYS = [dt.date(2020, 11, 27), dt.date(2020, 12, 24)]
FULL_DAY = dt.date(2020, 11, 25)
HOLIDAY = dt.date(2020, 11, 26)  # Thanksgiving


def _bars(dates: list[dt.date]) -> pl.LazyFrame:
    """One bar every 15 minutes from 09:00 to 16:45 on each date."""
    rows = []
    for date in dates:
        stamp = dt.datetime.combine(date, dt.time(9, 0))
        while stamp.time() <= dt.time(16, 45):
            rows.append({"timestamp": stamp, "symbol": "AAPL", "close": 1.0})
            stamp += dt.timedelta(minutes=15)
    return pl.DataFrame(rows).lazy()


def _kept(dates: list[dt.date]) -> pl.DataFrame:
    return _filter_to_exchange_sessions(_bars(dates)).collect()


class TestSessionBounds:
    def test_a_full_session_runs_to_the_close(self):
        kept = _kept([FULL_DAY])

        assert kept["timestamp"].min().time() == dt.time(9, 30)
        assert kept["timestamp"].max().time() == dt.time(15, 45)

    @pytest.mark.parametrize("half_day", HALF_DAYS)
    def test_an_early_close_ends_the_session(self, half_day: dt.date):
        kept = _kept([half_day])

        assert kept["timestamp"].min().time() == dt.time(9, 30)
        # 13:00 is the close, so the 12:45 bar is the last one inside the session.
        assert kept["timestamp"].max().time() == dt.time(12, 45)
        assert kept.filter(pl.col("timestamp").dt.hour() >= 13).height == 0

    @pytest.mark.parametrize("half_day", HALF_DAYS)
    def test_a_clock_bound_would_keep_the_afternoon(self, half_day: dt.date):
        """What the filter used to do, stated so a revert cannot pass quietly."""
        clock_bound = (
            _bars([half_day])
            .filter(
                (pl.col("timestamp").dt.hour() >= 10)
                | ((pl.col("timestamp").dt.hour() == 9) & (pl.col("timestamp").dt.minute() >= 30))
            )
            .filter(pl.col("timestamp").dt.hour() < 16)
            .collect()
        )

        after_close = clock_bound.filter(pl.col("timestamp").dt.hour() >= 13)
        assert after_close.height == 12, "the market was shut for these twelve bars"
        assert _kept([half_day]).height == clock_bound.height - 12

    def test_a_holiday_keeps_nothing(self):
        assert _kept([HOLIDAY]).height == 0

    def test_the_schema_is_unchanged(self):
        assert _kept([FULL_DAY]).columns == ["timestamp", "symbol", "close"]

    def test_only_the_early_closes_are_shortened(self):
        """The fix must not touch the 250 sessions a year that close at 16:00."""
        kept = _kept([FULL_DAY, *HALF_DAYS])
        last = (
            kept.group_by(pl.col("timestamp").dt.date().alias("date"))
            .agg(pl.col("timestamp").max().dt.time().alias("last"))
            .sort("date")
        )

        assert dict(zip(last["date"], last["last"], strict=True)) == {
            FULL_DAY: dt.time(15, 45),
            HALF_DAYS[0]: dt.time(12, 45),
            HALF_DAYS[1]: dt.time(12, 45),
        }


class TestThroughTheLoader:
    """The helper is only worth anything if `load_nasdaq100_bars` still calls it."""

    @pytest.fixture
    def archive(self, tmp_path, monkeypatch):
        """A minute-bar archive in the layout the loader scans."""
        rows = []
        for date in [FULL_DAY, HOLIDAY, *HALF_DAYS]:
            stamp = dt.datetime.combine(date, dt.time(9, 0))
            while stamp.time() <= dt.time(16, 45):
                rows.append(
                    {
                        "timestamp": stamp,
                        "symbol": "AAPL",
                        "date": date,
                        "first_trade_price": 100.0,
                        "high_trade_price": 101.0,
                        "low_trade_price": 99.0,
                        "last_trade_price": 100.5,
                        "volume": 1_000.0,
                    }
                )
                stamp += dt.timedelta(minutes=15)

        partition = tmp_path / "equities" / "market" / "nasdaq100" / "minute_bars" / "year=2020"
        partition.mkdir(parents=True)
        pl.DataFrame(rows).write_parquet(partition / "part.parquet")

        from data.equities import loader

        monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)
        return loader

    def test_regular_hours_stops_at_the_early_close(self, archive):
        bars = archive.load_nasdaq100_bars(regular_hours=True)
        last = (
            bars.group_by(pl.col("timestamp").dt.date().alias("date"))
            .agg(pl.col("timestamp").max().dt.time().alias("last"))
            .sort("date")
        )

        assert dict(zip(last["date"], last["last"], strict=True)) == {
            FULL_DAY: dt.time(15, 45),
            HALF_DAYS[0]: dt.time(12, 45),
            HALF_DAYS[1]: dt.time(12, 45),
        }
        assert HOLIDAY not in last["date"].to_list()

    def test_regular_hours_false_keeps_everything(self, archive):
        bars = archive.load_nasdaq100_bars(regular_hours=False)

        assert bars["timestamp"].min().time() == dt.time(9, 0)
        assert bars["timestamp"].max().time() == dt.time(16, 45)
        assert bars["timestamp"].dt.date().n_unique() == 4


class TestTheSessionTable:
    def test_the_two_early_closes_are_the_only_ones_in_the_window(self):
        sessions = _exchange_sessions().filter(
            pl.col("session_date").is_between(dt.date(2020, 1, 1), dt.date(2021, 7, 1))
        )

        early = sessions.filter(pl.col("session_close").dt.hour() < 16)
        assert early["session_date"].to_list() == HALF_DAYS
        assert sessions.height == 378

    def test_open_and_close_are_naive_eastern(self):
        """The archive's timestamps are naive Eastern; the bounds have to match."""
        sessions = _exchange_sessions()
        full = sessions.filter(pl.col("session_date") == FULL_DAY)

        assert sessions.schema["session_open"] == pl.Datetime("us")
        assert full["session_open"][0].time() == dt.time(9, 30)
        assert full["session_close"][0].time() == dt.time(16, 0)
