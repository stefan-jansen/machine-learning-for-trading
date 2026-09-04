"""The bound that keeps a daily fetch off the session in progress.

`02_financial_data_universe/18_data_management` and `19_incremental_updates`
both called `DataManager.update(..., provider="yahoo")`, which fetches from the
last stored bar to `datetime.now(UTC)`. Yahoo returns the current exchange date
as a row with accumulating volume and null open/high/low/close until it
consolidates hours after the close, and ml4t-data's provider raises on it:

    DataValidationError: yahoo: Column 'open' contains 1 null values

That is not vendor flakiness on one day and it is not a property of CI. The row
appears at the 16:00 New York close and stays until the daily bar consolidates
some six hours later: across 2026-09-02 and 2026-09-03 the `ch02-03` job was
green on every run started between 02:53Z and 20:29Z and red on every run
started between 23:55Z and 02:21Z. A reader in Europe meets it every morning.

The fix is the bound in `utils.downloading.last_complete_daily_bar`, exercised
here without a network so the rule can fail in a test rather than only against a
live vendor.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from utils.downloading import last_complete_daily_bar, update_through_last_complete_bar

UTC = dt.UTC


def _bars(dates: list[str], close: float = 100.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [dt.datetime.fromisoformat(d).replace(tzinfo=UTC) for d in dates],
            "symbol": ["AAPL"] * len(dates),
            "open": [close] * len(dates),
            "high": [close] * len(dates),
            "low": [close] * len(dates),
            "close": [close] * len(dates),
            "volume": [1_000.0] * len(dates),
        }
    )


class FakeStorage:
    """Just enough of HiveStorage: one key, read back what was written."""

    def __init__(self, frames: dict[str, pl.DataFrame]) -> None:
        self.frames = frames
        self.writes: list[tuple[str, dict | None]] = []

    def read(self, key: str) -> pl.LazyFrame:
        return self.frames.get(key, pl.DataFrame()).lazy()

    def write(self, data, key, metadata=None, *, preserve_metadata=False):
        self.frames[key] = data
        self.writes.append((key, metadata))


class FakeManager:
    """Records the window it was asked for and returns bars covering it."""

    def __init__(self, response: pl.DataFrame | None = None) -> None:
        self.response = response if response is not None else _bars([])
        self.calls: list[tuple[str, str, str]] = []

    def fetch(self, symbol, start, end, provider=None):
        self.calls.append((symbol, start, end))
        return self.response


# --- the bound ---------------------------------------------------------------


def test_the_bound_is_the_date_before_the_current_exchange_date() -> None:
    now = dt.datetime(2026, 9, 3, 14, 0, tzinfo=UTC)  # 10:00 in New York, mid-session
    assert last_complete_daily_bar(now) == dt.date(2026, 9, 2)


def test_the_bound_is_taken_in_exchange_time_not_utc() -> None:
    """The timestamp of the CI run that reported the failure.

    2026-09-04T02:02Z is 22:02 on 2026-09-03 in New York. A bound derived from
    the UTC date would be 2026-09-03 - the session Yahoo had not consolidated,
    and the one that raised.
    """
    now = dt.datetime(2026, 9, 4, 2, 2, tzinfo=UTC)
    assert now.date() - dt.timedelta(days=1) == dt.date(2026, 9, 3)
    assert last_complete_daily_bar(now) == dt.date(2026, 9, 2)


def test_the_bound_never_reaches_the_current_exchange_date() -> None:
    from zoneinfo import ZoneInfo

    for hour in range(24):
        now = dt.datetime(2026, 9, 3, hour, 30, tzinfo=UTC)
        exchange_today = now.astimezone(ZoneInfo("America/New_York")).date()
        assert last_complete_daily_bar(now) < exchange_today


# --- the update --------------------------------------------------------------


def test_the_fetch_window_stops_at_the_bound() -> None:
    storage = FakeStorage({"equities/daily/AAPL": _bars(["2026-08-20", "2026-08-21"])})
    manager = FakeManager(_bars(["2026-08-24"]))

    update_through_last_complete_bar(
        manager, storage, "AAPL", provider="yahoo", through=dt.date(2026, 8, 25)
    )

    (_, start, end) = manager.calls[0]
    assert end == "2026-08-25"
    assert start == "2026-08-14", "the lookback overlap is refetched, not skipped"


def test_new_bars_are_appended_and_the_row_count_returned() -> None:
    storage = FakeStorage({"equities/daily/AAPL": _bars(["2026-08-20", "2026-08-21"])})
    manager = FakeManager(_bars(["2026-08-24", "2026-08-25"]))

    rows = update_through_last_complete_bar(
        manager, storage, "AAPL", provider="yahoo", through=dt.date(2026, 8, 25)
    )

    assert rows == 4
    stored = storage.frames["equities/daily/AAPL"]
    assert stored["timestamp"].is_sorted()
    assert stored.height == 4


def test_a_bar_the_vendor_revised_replaces_the_stored_one() -> None:
    """The lookback overlap exists for this; a duplicate timestamp would be a bug."""
    storage = FakeStorage({"equities/daily/AAPL": _bars(["2026-08-20", "2026-08-21"], close=100.0)})
    manager = FakeManager(_bars(["2026-08-21", "2026-08-24"], close=222.0))

    rows = update_through_last_complete_bar(
        manager, storage, "AAPL", provider="yahoo", through=dt.date(2026, 8, 25)
    )

    stored = storage.frames["equities/daily/AAPL"]
    assert rows == 3
    revised = stored.filter(pl.col("timestamp") == dt.datetime(2026, 8, 21, tzinfo=UTC))
    assert revised["close"].item() == 222.0


def test_the_written_metadata_carries_the_new_end_date() -> None:
    storage = FakeStorage({"equities/daily/AAPL": _bars(["2026-08-20"])})
    manager = FakeManager(_bars(["2026-08-24"]))

    update_through_last_complete_bar(
        manager, storage, "AAPL", provider="yahoo", through=dt.date(2026, 8, 25)
    )

    (_, metadata) = storage.writes[0]
    assert metadata["end_date"].startswith("2026-08-24")
    assert metadata["start_date"].startswith("2026-08-20")


def test_a_store_already_past_the_bound_fetches_nothing() -> None:
    storage = FakeStorage({"equities/daily/AAPL": _bars(["2026-08-20", "2026-08-21"])})
    manager = FakeManager()

    rows = update_through_last_complete_bar(
        manager, storage, "AAPL", provider="yahoo", lookback_days=0, through=dt.date(2026, 8, 20)
    )

    assert rows == 2
    assert manager.calls == [], "no window is left to ask for"


def test_an_empty_store_says_to_load_first() -> None:
    storage = FakeStorage({})
    with pytest.raises(ValueError, match="load it before updating"):
        update_through_last_complete_bar(FakeManager(), storage, "AAPL", provider="yahoo")
