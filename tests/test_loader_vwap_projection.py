"""A backtest that fills at VWAP needs the VWAP on the bar it fills, and a real one when resampled.

`load_nasdaq100_bars` projected seven trade columns and dropped `vwap`, so the column existed on
disk and never reached a caller. Adding it to the projection is what makes the resample question
live: there was no wrong aggregation before, because there was no column to aggregate.
"""

from __future__ import annotations

import polars as pl

from data.equities.loader import _TRADE_OHLCV_AGGS


def _window(volumes: list[int], vwaps: list[float | None]) -> pl.DataFrame:
    n = len(volumes)
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2021, 3, 1, 10, 0),
                pl.datetime(2021, 3, 1, 10, 0) + pl.duration(minutes=n - 1),
                "1m",
                eager=True,
            ),
            "symbol": ["AAPL"] * n,
            "open": [10.0] * n,
            "high": [10.0] * n,
            "low": [10.0] * n,
            "close": [10.0] * n,
            "volume": volumes,
            "vwap": vwaps,
        }
    )


def _resample(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.group_by_dynamic("timestamp", every="15m", group_by="symbol").agg(
        _TRADE_OHLCV_AGGS
    )


def test_a_resampled_vwap_is_volume_weighted_not_averaged():
    """The distinction the aggregation exists for, on volumes lopsided enough to show it."""
    out = _resample(_window([100, 300], [10.0, 12.0]))
    assert out["vwap"][0] == (10.0 * 100 + 12.0 * 300) / 400 == 11.5
    # The reading that looks right and is not: the unweighted mean of the same two minutes.
    assert out["vwap"][0] != (10.0 + 12.0) / 2


def test_a_minute_with_no_trade_contributes_no_weight():
    """`vwap` is null exactly when `volume` is 0, and a null must not count as a zero price.

    Counting it would drag the window's price toward zero in proportion to how much of the
    window did not trade, which is the opposite of ignoring it.
    """
    with_gap = _resample(_window([100, 0, 300], [10.0, None, 12.0]))
    without_gap = _resample(_window([100, 300], [10.0, 12.0]))
    assert with_gap["vwap"][0] == without_gap["vwap"][0] == 11.5


def test_a_window_that_never_traded_has_no_vwap():
    """Null rather than a division by zero, and never a substituted price."""
    out = _resample(_window([0, 0], [None, None]))
    assert out["vwap"][0] is None


def test_volume_still_sums_across_a_window_that_did_not_trade():
    """The weight exclusion is scoped to the vwap; `volume` keeps its own aggregation."""
    out = _resample(_window([100, 0, 300], [10.0, None, 12.0]))
    assert out["volume"][0] == 400
