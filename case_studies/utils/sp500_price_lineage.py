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


def adjustment_scale(bars: pl.DataFrame) -> pl.DataFrame:
    """Return the per ``(symbol, timestamp)`` multiplier that back-adjusts a printed price.

    Two things go into it. ``adj_factor`` removes the corporate action. Then each
    ``sec_id`` segment after a ticker's first is rescaled to meet the level the
    previous segment closed at, so the two securities keep their own returns and
    the changeover contributes zero rather than the jump a factor restarting at
    1.0 would produce - the treatment ``cme_futures`` gives a contract roll,
    applied to a ticker reassignment. A backtest feed reads a level, so unlike
    :func:`reconcile_underlying_log_returns` it cannot say that a return does not
    exist.

    Finally the series is anchored so each ticker's **last** row equals the close
    that printed, which is what makes it a back-adjusted price rather than an
    index. The level is not cosmetic: this case study sizes positions in whole
    shares against a fixed cash budget, and on the factor's own scale AAPL sits
    near 4000 instead of near 130, which turns integer rounding into a material
    allocation error.

    **Pass the complete bar history.** Both halves depend on every segment a
    ticker has and on which row is its last, so a scale derived from a
    date-filtered frame is a function of the window as well as the session, and
    two windows would disagree about the same date. That is not hypothetical:
    the holdout path concatenates a validation load and a holdout load to give
    the rolling-volatility allocators their burn-in, and a window-dependent
    scale puts a fabricated return on the seam - 8x for GE, whose 1-for-8 falls
    inside the holdout year.
    """
    frame = _validated(bars).with_columns(
        pl.col("identity_boundary").cum_sum().over("symbol").alias("_seg")
    )
    splice = (
        frame.group_by("symbol", "_seg")
        .agg(
            pl.col("adjusted_close").first().alias("_open"),
            pl.col("adjusted_close").last().alias("_close"),
        )
        .sort("symbol", "_seg")
        .with_columns((pl.col("_close").shift(1) / pl.col("_open")).over("symbol").alias("_step"))
    )
    # The null `_step` belongs to a ticker's first segment, where there is nothing
    # to splice onto. A null anywhere else means a null close reached the ratio -
    # `_validated` tolerates those - and filling it would silently drop the splice
    # and carry a wrong offset into every later segment, visible only as P&L.
    unspliceable = splice.filter((pl.col("_seg") > 0) & pl.col("_step").is_null())
    if not unspliceable.is_empty():
        raise ValueError(
            "cannot splice a security identity boundary whose adjusted close is null: "
            f"{unspliceable.select('symbol', '_seg').rows()}"
        )
    splice = (
        splice.with_columns(pl.col("_step").fill_null(1.0))
        .with_columns(pl.col("_step").cum_prod().over("symbol").alias("_splice"))
        .select("symbol", "_seg", "_splice")
    )
    return (
        frame.join(splice, on=["symbol", "_seg"], how="inner")
        .with_columns((pl.col("adj_factor") * pl.col("_splice")).alias("price_scale"))
        .with_columns(
            (
                pl.col("price_scale")
                / pl.col("price_scale").last().over("symbol", order_by="timestamp")
            ).alias("price_scale")
        )
        .select("symbol", "timestamp", "price_scale")
    )


def continuous_adjusted_panel(
    prices: pl.DataFrame,
    *,
    scale: pl.DataFrame,
    price_cols: tuple[str, ...] = PRICE_COLS,
    volume_col: str | None = "volume",
) -> pl.DataFrame:
    """Apply a full-history :func:`adjustment_scale` to a panel that may be one window of it.

    Keeping the scale a separate argument is what makes the result a function of
    ``(symbol, timestamp)`` alone: the caller resolves it once over the whole
    series, and every window of that series then agrees about every date it
    contains.

    ``volume_col`` is divided by the same factor its row's price is multiplied
    by, so ``price * volume`` stays the dollar volume that printed. Note that the
    share **count** moves with the level, so a per-share cost schedule evaluated
    on this panel is charged on adjusted share counts and is not comparable to a
    live per-share commission.
    """
    columns = [column for column in price_cols if column in prices.columns]
    if "close" not in columns:
        raise ValueError("the adjusted panel requires a close column")
    if "price_scale" in prices.columns:
        raise ValueError("the panel already carries a price_scale column")

    frame = prices.join(scale, on=["symbol", "timestamp"], how="left")
    unscaled = frame.filter(pl.col("price_scale").is_null())
    if not unscaled.is_empty():
        raise ValueError(
            "the adjustment scale does not cover every panel row; it must be resolved "
            f"over the complete bar history. First uncovered: {unscaled.head(3).rows()}"
        )
    volume = (
        [(pl.col(volume_col) / pl.col("price_scale")).alias(volume_col)]
        if _has(frame, volume_col)
        else []
    )
    return frame.with_columns(
        [(pl.col(column) * pl.col("price_scale")).alias(column) for column in columns] + volume
    ).drop("price_scale")


def _has(frame: pl.DataFrame, column: str | None) -> bool:
    return bool(column) and column in frame.columns
