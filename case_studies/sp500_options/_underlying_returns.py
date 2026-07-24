"""Point-in-time S&P 500 returns that respect security identity boundaries."""

from __future__ import annotations

import polars as pl


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


def reconcile_underlying_log_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute adjusted daily log returns within stable ``sec_id`` segments."""
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

    frame = prices.sort(["symbol", "timestamp"]).with_columns(
        (pl.col("close") * pl.col("adj_factor")).alias("adjusted_close"),
        (
            pl.col("sec_id").shift(1).over("symbol").is_not_null()
            & (pl.col("sec_id") != pl.col("sec_id").shift(1).over("symbol"))
        ).alias("identity_boundary"),
    )
    frame = frame.with_columns(
        pl.col("adjusted_close").log().diff().over(["symbol", "sec_id"]).alias("clean_log_return")
    )
    validate_reconciled_returns(frame)
    return frame
