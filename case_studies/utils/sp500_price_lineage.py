"""Point-in-time S&P 500 prices that respect security identity boundaries.

``load_sp500_daily_bars`` returns the close as it printed plus ``adj_factor``, a
cumulative price factor. ``close * adj_factor`` is the series in which a split, a
reverse split or a cash dividend is no longer a price move, and it is what
``sp500_equity_option_analytics/02_labels`` builds every label from.

The factor restarts at 1.0 when the exchange reassigns a ticker to a new
security, so a series taken across that boundary carries a jump that is neither a
price move nor a corporate action on the security being held. Both consumers of
these bars need that boundary, and they need it in different shapes: a return
series can leave it null, a backtest price panel cannot. Both shapes are here so
the boundary is defined once.
"""

from __future__ import annotations

import polars as pl

PRICE_COLS = ("open", "high", "low", "close")


def validate_reconciled_returns(frame: pl.DataFrame) -> None:
    """Fail if a return crosses a security identity or violates adjusted-price arithmetic."""
    required = {
        "sec_id",
        "adjusted_close",
        "clean_log_return",
        "identity_boundary",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Reconciled returns missing columns: {sorted(missing)}")
    if frame.filter(pl.col("identity_boundary") & pl.col("clean_log_return").is_not_null()).height:
        raise ValueError("A return crosses a security identity boundary")

    expected = pl.col("adjusted_close").log().diff().over(["symbol", "sec_id"])
    checked = frame.with_columns(expected.alias("expected_log_return"))
    violations = checked.filter(
        ~(
            pl.col("clean_log_return").eq_missing(pl.col("expected_log_return"))
            | ((pl.col("clean_log_return") - pl.col("expected_log_return")).abs() <= 1e-12)
        )
    )
    if not violations.is_empty():
        raise ValueError(f"Adjusted-return identity violations: {violations.height}")


def _validated(prices: pl.DataFrame) -> pl.DataFrame:
    """Check the bar columns both shapes depend on, and mark each identity boundary."""
    required = {"timestamp", "symbol", "sec_id", "close", "adj_factor"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Underlying bars missing columns: {sorted(missing)}")
    if prices.select(pl.struct("timestamp", "symbol").is_duplicated().any()).item():
        raise ValueError("Underlying bars contain duplicate timestamp-symbol keys")
    if prices["sec_id"].null_count():
        raise ValueError("Underlying bars contain null sec_id values")
    invalid_levels = prices.filter(
        ((pl.col("close").is_not_null()) & (pl.col("close") <= 0))
        | ((pl.col("adj_factor").is_not_null()) & (pl.col("adj_factor") <= 0))
    )
    if not invalid_levels.is_empty():
        raise ValueError(
            f"Underlying bars contain nonpositive price/factor rows: {invalid_levels.height}"
        )
    return prices.sort(["symbol", "timestamp"]).with_columns(
        (pl.col("close") * pl.col("adj_factor")).alias("adjusted_close"),
        (
            pl.col("sec_id").shift(1).over("symbol").is_not_null()
            & (pl.col("sec_id") != pl.col("sec_id").shift(1).over("symbol"))
        ).alias("identity_boundary"),
    )


def reconcile_underlying_log_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute adjusted daily log returns within stable ``sec_id`` segments."""
    frame = _validated(prices).with_columns(
        pl.col("adjusted_close").log().diff().over(["symbol", "sec_id"]).alias("clean_log_return")
    )
    validate_reconciled_returns(frame)
    return frame


def continuous_adjusted_panel(
    prices: pl.DataFrame,
    *,
    price_cols: tuple[str, ...] = PRICE_COLS,
    volume_col: str | None = "volume",
) -> pl.DataFrame:
    """Back-adjust every price column and splice at each security identity boundary.

    A backtest feed reads a level, so it has no way to say that a return does not
    exist. Each ``sec_id`` segment after a ticker's first is rescaled to meet the
    level the previous segment closed at: the two securities keep their own
    returns, and the changeover contributes zero instead of the jump the
    restarting factor would otherwise produce. It is the treatment ``cme_futures``
    gives a contract roll, applied to a ticker reassignment.

    The series is then anchored so each ticker's **last** row equals the close
    that printed, which is what makes it a back-adjusted price rather than an
    index. The level matters: this case study sizes positions in whole shares
    against a fixed cash budget, so a series carried on the factor's own scale
    would put AAPL near 4000 instead of near 130 and turn integer rounding into
    a material allocation error. History before the anchor is the split- and
    dividend-adjusted equivalent, exactly as ``us_equities_panel``'s stored
    ``adj_close`` is.

    ``volume_col`` is divided by the same total factor each row's price is
    multiplied by, so ``price * volume`` stays the dollar volume that printed.
    """
    frame = _validated(prices)
    columns = [column for column in price_cols if column in frame.columns]
    if "close" not in columns:
        raise ValueError("the adjusted panel requires a close column")

    frame = frame.with_columns(pl.col("identity_boundary").cum_sum().over("symbol").alias("_seg"))
    splice = (
        frame.group_by("symbol", "_seg")
        .agg(
            pl.col("adjusted_close").first().alias("_open"),
            pl.col("adjusted_close").last().alias("_close"),
        )
        .sort("symbol", "_seg")
        .with_columns(
            (pl.col("_close").shift(1) / pl.col("_open")).over("symbol").fill_null(1.0).alias("_st")
        )
        .with_columns(pl.col("_st").cum_prod().over("symbol").alias("_splice"))
        .select("symbol", "_seg", "_splice")
    )
    frame = frame.join(splice, on=["symbol", "_seg"], how="left").with_columns(
        (pl.col("adj_factor") * pl.col("_splice")).alias("_scale")
    )
    # Anchor: each ticker's last row keeps the close that printed, so the level a
    # position is sized against is the one the market quoted.
    frame = frame.with_columns(
        (pl.col("_scale") / pl.col("_scale").last().over("symbol", order_by="timestamp")).alias(
            "_scale"
        )
    )
    volume = (
        [(pl.col(volume_col) / pl.col("_scale")).alias(volume_col)]
        if _has(frame, volume_col)
        else []
    )
    return frame.with_columns(
        [(pl.col(column) * pl.col("_scale")).alias(column) for column in columns] + volume
    ).drop("_seg", "_splice", "_scale", "adjusted_close", "identity_boundary")


def _has(frame: pl.DataFrame, column: str | None) -> bool:
    return bool(column) and column in frame.columns
