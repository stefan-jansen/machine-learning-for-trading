"""The equity-option backtest panel marks positions on the series the labels use.

`load_sp500_daily_bars` prints the close as it traded and carries the cumulative
price factor separately, so a panel built on the printed close books every split,
reverse split and cash dividend as P&L. `02_labels` builds every label from
`close * adj_factor` inside a `sec_id`, which left the label and the backtest
measuring returns on two different price series.
"""

from __future__ import annotations

import polars as pl
import pytest

from case_studies.utils.backtest_loaders import load_backtest_prices
from case_studies.utils.sp500_price_lineage import adjustment_scale, continuous_adjusted_panel
from data import load_sp500_daily_bars
from data.exceptions import DataNotFoundError

CASE_STUDY = "sp500_equity_option_analytics"


def _bars(rows: list[tuple[str, int, str, float, float, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "sec_id": [r[1] for r in rows],
            "timestamp": [pl.Series([r[2]]).str.to_date().item() for r in rows],
            "close": [r[3] for r in rows],
            "adj_factor": [r[4] for r in rows],
            "volume": [float(r[5]) for r in rows],
        }
    )


def _adjusted(bars: pl.DataFrame) -> pl.DataFrame:
    """Adjust a frame that IS the complete history for the tickers it holds."""
    return continuous_adjusted_panel(bars, scale=adjustment_scale(bars))


def _returns(frame: pl.DataFrame, symbol: str) -> list[float]:
    series = frame.filter(pl.col("symbol") == symbol).sort("timestamp")["close"]
    return (series / series.shift(1) - 1).drop_nulls().to_list()


def test_a_subdivision_is_not_a_price_move():
    # AAPL's 4-for-1 on 2020-08-31: the printed close falls 499.23 -> 129.04
    # while adj_factor rises by the same 4x, so the adjusted move is +3.4%.
    panel = _adjusted(
        _bars(
            [
                ("AAPL", 33449, "2020-08-28", 499.23, 8.100549, 44_260_263),
                ("AAPL", 33449, "2020-08-31", 129.04, 32.402197, 210_249_674),
            ]
        )
    )
    (change,) = _returns(panel, "AAPL")
    assert change == pytest.approx(0.03383, abs=1e-4)


def test_a_reverse_split_is_not_a_price_move():
    # GE's 1-for-8 on 2021-08-02: the printed close rises 12.96 -> 100.60.
    panel = _adjusted(
        _bars(
            [
                ("GE", 33645, "2021-07-30", 12.96, 8.0, 80_000_000),
                ("GE", 33645, "2021-08-02", 100.60, 1.0, 10_000_000),
            ]
        )
    )
    (change,) = _returns(panel, "GE")
    assert change == pytest.approx(-0.02970, abs=1e-4)


def test_a_ticker_reassignment_contributes_no_return():
    # The factor restarts at 1.0 with the new security, so multiplying without
    # splicing would fabricate a jump where the ticker changed hands.
    panel = _adjusted(
        _bars(
            [
                ("XRX", 44733, "2019-07-30", 30.00, 1.853437, 1_000_000),
                ("XRX", 44733, "2019-07-31", 31.00, 1.853437, 1_000_000),
                ("XRX", 5888027, "2019-08-01", 31.64, 1.0, 1_000_000),
                ("XRX", 5888027, "2019-08-02", 32.00, 1.0, 1_000_000),
            ]
        )
    )
    first, boundary, second = _returns(panel, "XRX")
    assert first == pytest.approx(1 / 30, abs=1e-9)
    assert boundary == pytest.approx(0.0, abs=1e-12)
    assert second == pytest.approx(32.00 / 31.64 - 1, abs=1e-9)


def test_each_security_keeps_its_own_returns_across_a_reassignment():
    # Splicing rescales a segment; it must not change any return inside one.
    rows = [
        ("IR", 5122670, "2020-02-28", 129.04, 1.578133, 1_000_000),
        ("IR", 5122670, "2020-03-02", 130.00, 1.578133, 1_000_000),
        ("IR", 6285863, "2020-03-03", 32.80, 1.0, 1_000_000),
        ("IR", 6285863, "2020-03-04", 33.00, 1.0, 1_000_000),
    ]
    panel = _adjusted(_bars(rows))
    within = _returns(panel, "IR")
    assert within[0] == pytest.approx(130.00 / 129.04 - 1, abs=1e-9)
    assert within[2] == pytest.approx(33.00 / 32.80 - 1, abs=1e-9)


def test_dollar_volume_survives_the_adjustment():
    rows = [
        ("AAPL", 33449, "2020-08-28", 499.23, 8.100549, 44_260_263),
        ("AAPL", 33449, "2020-08-31", 129.04, 32.402197, 210_249_674),
    ]
    panel = _adjusted(_bars(rows)).sort("timestamp")
    dollars = (panel["close"] * panel["volume"]).to_list()
    assert dollars == [
        pytest.approx(499.23 * 44_260_263),
        pytest.approx(129.04 * 210_249_674),
    ]


def test_the_anchor_is_the_close_that_printed():
    # The series is back-adjusted, not an index: the last row a ticker has keeps
    # its quoted level, so whole-share sizing sees the price the market showed.
    rows = [
        ("AAPL", 33449, "2020-08-28", 499.23, 8.100549, 44_260_263),
        ("AAPL", 33449, "2020-08-31", 129.04, 32.402197, 210_249_674),
    ]
    panel = _adjusted(_bars(rows)).sort("timestamp")
    assert panel["close"].to_list()[-1] == pytest.approx(129.04)
    # ...and the session before it is 499.23 carried through the ratio of the two
    # factors, which is the 4-for-1 plus the dividend that accrued with it.
    assert panel["close"].to_list()[0] == pytest.approx(499.23 * 8.100549 / 32.402197, abs=1e-12)
    assert panel["close"].to_list()[0] == pytest.approx(499.23 / 4.0, rel=1e-4)


def test_a_restarting_factor_without_a_lineage_column_is_refused():
    with pytest.raises(ValueError, match="missing columns"):
        _adjusted(_bars([("AAPL", 33449, "2020-08-31", 129.04, 32.402197, 1)]).drop("sec_id"))


def test_a_scale_that_does_not_cover_the_panel_is_refused():
    # The failure mode this guards is a scale resolved over a narrower frame than
    # the panel it is applied to, which is how the window dependence got in.
    bars = _bars(
        [
            ("GE", 33645, "2021-07-30", 12.96, 8.0, 80_000_000),
            ("GE", 33645, "2021-08-02", 100.60, 1.0, 10_000_000),
        ]
    )
    partial = adjustment_scale(bars.head(1))
    with pytest.raises(ValueError, match="complete bar history"):
        continuous_adjusted_panel(bars, scale=partial)


def test_a_null_close_at_a_segment_boundary_is_refused():
    """The splice ratio reads the previous segment's last close, which may be null.

    Filling that null the way a first segment's is filled would drop the splice and
    carry a wrong offset into every later segment, visible only as P&L. The scale is
    resolved over the whole history, so this must fail loudly rather than quietly.
    """
    bars = _bars(
        [
            ("XRX", 44733, "2019-07-30", 30.00, 1.853437, 1_000_000),
            ("XRX", 44733, "2019-07-31", 31.00, 1.853437, 1_000_000),
            ("XRX", 5888027, "2019-08-01", 31.64, 1.0, 1_000_000),
        ]
    ).with_columns(
        pl.when(pl.col("timestamp") == pl.date(2019, 7, 31))
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    with pytest.raises(ValueError, match="cannot splice"):
        adjustment_scale(bars)


def test_a_null_close_that_ends_a_ticker_is_not_refused():
    """The 10 in the shipped extract are all of this shape and must keep working.

    Measured on `daily_bars.parquet`: 10 segments carry a null close, every one the
    terminal row of a delisted ticker that has a single `sec_id`, and none at a
    segment boundary. Nothing splices onto them, so the guard above must not fire.
    """
    bars = _bars(
        [
            ("WFM", 40001, "2017-08-25", 41.00, 1.0, 1_000_000),
            ("WFM", 40001, "2017-08-28", 30.89, 1.0, 1_000_000),
        ]
    ).with_columns(
        pl.when(pl.col("timestamp") == pl.date(2017, 8, 28))
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    scale = adjustment_scale(bars)
    assert scale.height == 2
    assert scale["price_scale"].null_count() == 0


def test_two_windows_of_one_series_agree_about_the_dates_they_share():
    """The regression guard for a scale anchored on the frame it was handed.

    The holdout path concatenates a validation load and a holdout load to give the
    rolling-volatility allocators their burn-in. If the anchor is the last row of
    each *window*, the two halves are on different scales and the seam carries a
    fabricated return. GE is the case that makes it unmissable: its 1-for-8 falls
    inside the holdout year, so the two windows would disagree by a factor of 8.
    """
    history = _bars(
        [
            ("GE", 33645, "2020-11-30", 10.00, 8.0, 1_000_000),
            ("GE", 33645, "2020-12-31", 11.00, 8.0, 1_000_000),
            ("GE", 33645, "2021-06-30", 12.96, 8.0, 1_000_000),
            ("GE", 33645, "2021-08-02", 100.60, 1.0, 1_000_000),
            ("GE", 33645, "2021-12-31", 104.00, 1.0, 1_000_000),
        ]
    )
    scale = adjustment_scale(history)
    validation = continuous_adjusted_panel(
        history.filter(pl.col("timestamp") <= pl.date(2020, 12, 31)), scale=scale
    )
    holdout = continuous_adjusted_panel(
        history.filter(pl.col("timestamp") >= pl.date(2020, 12, 31)), scale=scale
    )
    shared = validation.join(holdout, on=["symbol", "timestamp"], suffix="_holdout")
    assert shared.height == 1
    assert shared["close"][0] == pytest.approx(shared["close_holdout"][0], abs=1e-12)

    # And the seam itself: concatenating the two halves must reproduce the return
    # the whole series gives, not an 8x step.
    seam = pl.concat([validation, holdout]).unique(subset=["symbol", "timestamp"]).sort("timestamp")
    whole = continuous_adjusted_panel(history, scale=scale).sort("timestamp")
    assert seam["close"].to_list() == pytest.approx(whole["close"].to_list(), abs=1e-12)
    assert max(abs(r) for r in _returns(seam, "GE")) < 0.20


def test_two_real_loads_agree_about_the_dates_they_share():
    """The same guard through `load_backtest_prices`, on the windows holdout actually uses.

    `20_strategy_synthesis/holdout.py` concatenates a validation load and a
    holdout load, so these are the two frames whose seam has to hold. GE's
    1-for-8 on 2021-08-02 sits inside the holdout window.
    """
    try:
        load_sp500_daily_bars(symbols=["GE"])
    except DataNotFoundError:
        pytest.skip("Licensed S&P 500 data is unavailable")

    validation = load_backtest_prices(CASE_STUDY, start_date="2019-01-01", end_date="2020-12-31")
    holdout = load_backtest_prices(CASE_STUDY, start_date="2020-06-01", end_date="2021-12-31")
    shared = validation.join(holdout, on=["symbol", "timestamp"], suffix="_holdout")
    assert shared.height > 50_000
    drift = (shared["close"] - shared["close_holdout"]).abs().max()
    assert drift == pytest.approx(0.0, abs=1e-9)

    seam = (
        pl.concat([validation, holdout])
        .unique(subset=["symbol", "timestamp"])
        .sort("symbol", "timestamp")
        .with_columns((pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("r"))
        .drop_nulls("r")
    )
    assert seam.filter(pl.col("r").abs() > 0.60).is_empty()


def test_the_real_panel_carries_no_corporate_action_as_a_price_move():
    """The shipped 2017-2021 extract, through the loader the backtest calls."""
    try:
        load_sp500_daily_bars(symbols=["AAPL"])
    except DataNotFoundError:
        pytest.skip("Licensed S&P 500 data is unavailable")

    panel = load_backtest_prices(CASE_STUDY, start_date="2017-01-01", end_date="2021-12-31")
    assert {"open", "high", "low", "close", "volume"} <= set(panel.columns)
    assert not {"adj_factor", "sec_id", "adjustment_reason"} & set(panel.columns)

    moves = (
        panel.sort("symbol", "timestamp")
        .with_columns((pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("r"))
        .drop_nulls("r")
    )
    # On the printed close, 92 sessions move more than 30% on a corporate action
    # alone. What survives here is the market, not the adjustment: the largest
    # single-session move in the extract is a real one.
    extreme = moves.filter(pl.col("r").abs() > 0.60)
    assert extreme.height == 0, extreme.sort(pl.col("r").abs(), descending=True).head(10)

    # The 15 ticker reassignments contribute no return rather than a spliced-in jump.
    boundaries = (
        load_sp500_daily_bars(start_date="2017-01-01", end_date="2021-12-31")
        .sort("symbol", "timestamp")
        .with_columns(
            (pl.col("sec_id") != pl.col("sec_id").shift(1).over("symbol")).alias("boundary")
        )
        .filter(pl.col("boundary").fill_null(False))
        .select("symbol", "timestamp")
    )
    assert boundaries.height == 15
    crossed = moves.join(
        boundaries.with_columns(pl.col("timestamp").cast(moves.schema["timestamp"])),
        on=["symbol", "timestamp"],
    )
    assert crossed.height == 15
    assert crossed["r"].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_the_real_panel_keeps_prices_at_the_level_positions_are_sized_against():
    """`execution.share_type` is `integer`, so the level is not cosmetic."""
    try:
        bars = load_sp500_daily_bars(start_date="2017-01-01", end_date="2021-12-31")
    except DataNotFoundError:
        pytest.skip("Licensed S&P 500 data is unavailable")

    panel = load_backtest_prices(CASE_STUDY, start_date="2017-01-01", end_date="2021-12-31")
    last_printed = (
        bars.sort("timestamp").group_by("symbol").last().select("symbol", "close").sort("symbol")
    )
    last_panel = (
        panel.sort("timestamp").group_by("symbol").last().select("symbol", "close").sort("symbol")
    )
    joined = last_printed.join(last_panel, on="symbol", suffix="_panel")
    assert joined.height == last_printed.height
    drift = (joined["close_panel"] - joined["close"]).abs().max()
    assert drift == pytest.approx(0.0, abs=1e-9)


def test_the_real_panel_matches_the_price_basis_the_labels_use():
    """`02_labels` builds every label from `close * adj_factor` within a `sec_id`."""
    try:
        bars = load_sp500_daily_bars(start_date="2019-01-01", end_date="2019-12-31")
    except DataNotFoundError:
        pytest.skip("Licensed S&P 500 data is unavailable")

    panel = load_backtest_prices(CASE_STUDY, start_date="2019-01-01", end_date="2019-12-31")
    label_basis = (
        bars.sort("symbol", "timestamp")
        .with_columns((pl.col("close") * pl.col("adj_factor")).alias("p"))
        .with_columns((pl.col("p").log().diff().over("symbol", "sec_id")).alias("label_r"))
        .drop_nulls("label_r")
        .select("symbol", "timestamp", "label_r")
    )
    backtest_basis = (
        panel.sort("symbol", "timestamp")
        .with_columns(pl.col("close").log().diff().over("symbol").alias("backtest_r"))
        .drop_nulls("backtest_r")
        .with_columns(pl.col("timestamp").dt.date())
        .select("symbol", "timestamp", "backtest_r")
    )
    joined = label_basis.join(backtest_basis, on=["symbol", "timestamp"], how="inner")
    assert joined.height > 100_000
    disagreement = joined.filter((pl.col("label_r") - pl.col("backtest_r")).abs() > 1e-10)
    assert disagreement.is_empty(), disagreement.head(10)
