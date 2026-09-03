"""The interval the label predicts is the interval the strategy holds, and both are VWAP.

ml4t/agent-workspace#187: the labels were built as `mid_close[t+15] / mid_close[t+1]` on the
minute grid while the backtest filled on a fifteen-minute clock and held `[t+15, t+30]`. The two
intervals shared a single instant. What hid it for months is that both were fifteen minutes long
and nothing printed distinguished them, so this pins the arithmetic rather than the description.

Two properties, and the second is the one the old code failed:

* the legs are **traded** prices - the volume-weighted price of the fill minute, not a quote
  midpoint, because a midpoint is not a price anything transacted at; and
* the span from entry fill to exit fill is the **declared horizon**, not the horizon minus a
  bar. `HORIZONS[name] - BAR` is what produced a fourteen-minute `fwd_ret_15m`, and it read as
  fifteen to anyone who saw the subtraction and rounded it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest
from ml4t.engineer.labeling import fixed_time_horizon_labels


def _minutes(vwaps: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2021, 3, 1, 10, 0),
                pl.datetime(2021, 3, 1, 10, 0) + pl.duration(minutes=len(vwaps) - 1),
                "1m",
                eager=True,
            ),
            "symbol": ["AAPL"] * len(vwaps),
            "vwap": vwaps,
        }
    )


def _label(frame: pl.DataFrame, horizon: str) -> pl.DataFrame:
    """The notebook's construction: entry one bar forward, exit `horizon` past the entry."""
    priced = frame.with_columns(
        pl.col("vwap").shift(-1).over("symbol").alias("entry_vwap")
    ).drop_nulls("entry_vwap")
    return fixed_time_horizon_labels(
        priced,
        horizon=horizon,
        method="returns",
        price_col="entry_vwap",
        group_col=["symbol"],
        timestamp_col="timestamp",
        tolerance="0s",
    ).rename({f"label_return_{horizon}": "label"})


@pytest.mark.filterwarnings("ignore:Sortedness of columns")
def test_the_label_spans_entry_fill_to_exit_fill():
    """`fwd_ret_15m[t]` is `vwap[t+16] / vwap[t+1] - 1` - fifteen minutes of exposure."""
    prices = [100.0 + i for i in range(20)]
    out = _label(_minutes(prices), "15m").drop_nulls("label")
    first = out.row(0, named=True)
    assert first["timestamp"] == datetime(2021, 3, 1, 10, 0)
    assert first["label"] == pytest.approx(prices[16] / prices[1] - 1)
    # The fourteen-minute reading the previous convention produced, explicitly excluded.
    assert first["label"] != pytest.approx(prices[15] / prices[1] - 1)


@pytest.mark.filterwarnings("ignore:Sortedness of columns")
def test_the_exit_of_one_decision_is_the_entry_of_the_next():
    """Consecutive fifteen-minute decisions abut without overlapping, which is what makes the
    exposure exactly the horizon rather than a bar more or less."""
    prices = [100.0 + i for i in range(40)]
    out = _label(_minutes(prices), "15m").drop_nulls("label")
    rows = {r["timestamp"]: r for r in out.iter_rows(named=True)}
    t0 = datetime(2021, 3, 1, 10, 0)
    first, second = rows[t0], rows[t0 + timedelta(minutes=15)]
    assert first["label"] == pytest.approx(prices[16] / prices[1] - 1)
    assert second["label"] == pytest.approx(prices[31] / prices[16] - 1)


@pytest.mark.filterwarnings("ignore:Sortedness of columns")
def test_a_minute_that_did_not_trade_carries_no_label():
    """No VWAP is no fill, on either leg, rather than a substituted price.

    At the close of `t` nothing knows whether `t+1` will print, so filling the gap would put
    information in the label that the decision could not have had.
    """
    prices: list[float | None] = [100.0 + i for i in range(20)]
    prices[1] = None  # the entry minute for the decision at t=0 does not trade
    out = _label(_minutes(prices), "15m")
    labelled = set(out.filter(pl.col("label").is_not_null())["timestamp"].to_list())
    # The decision at 10:00 would have filled at 10:01, which did not trade.
    assert datetime(2021, 3, 1, 10, 0) not in labelled
    # Its neighbours are unaffected: the hole drops one decision, not the series.
    assert datetime(2021, 3, 1, 10, 1) in labelled


class TestTheDecisionGridIsBuiltFromTheDeclaration:
    """The declared cadence builds the decision schedule; the panel's own spacing does not.

    These pinned the opposite until stefan-jansen/machine-learning-for-trading#736. The
    resolver returned an intraday panel's timestamps unchanged, on the comment that the data
    was already at the correct granularity, so `rebalance_step` thinned whatever it was
    handed. A minute-grid panel therefore made step 4 four minutes rather than sixty, and
    "predictions are emitted on the decision grid" was a precondition nothing enforced.

    It is enforced now, at the source: the schedule is constructed from the token, so a panel
    at any resolution produces the same decisions. What these keep is the shape of the check -
    the reading the old code produced is still asserted to be gone, not merely unasserted.
    """

    @staticmethod
    def _panel(minutes: int, count: int) -> pl.Series:
        start = datetime(2021, 3, 1, 9, 30)
        return pl.Series(
            "ts", [start + timedelta(minutes=minutes * i) for i in range(count)]
        ).sort()

    def test_a_fifteen_minute_cadence_decides_every_fifteen_minutes(self):
        from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

        schedule = resolve_rebalance_timestamps(self._panel(15, 16), "15_minute")
        assert schedule.diff().drop_nulls().unique().to_list() == [timedelta(minutes=15)]

    def test_a_minute_panel_gives_the_same_schedule_as_a_fifteen_minute_one(self):
        """The panel's resolution is no longer decisive, which is the whole of the fix."""
        from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

        coarse = resolve_rebalance_timestamps(self._panel(15, 16), "15_minute")
        fine = resolve_rebalance_timestamps(self._panel(1, 16 * 15), "15_minute")
        assert fine.to_list() == coarse.to_list()

    def test_a_minute_panel_no_longer_trades_every_four_minutes(self):
        """The reading the old code produced, asserted to be gone rather than left unchecked."""
        from case_studies.utils.backtest_loaders import resolve_decision_schedule

        schedule = resolve_decision_schedule(self._panel(1, 64 * 15), "15_minute", 4)
        gaps = schedule.diff().drop_nulls().unique().to_list()
        assert gaps == [timedelta(hours=1)]
        assert gaps != [timedelta(minutes=4)]
