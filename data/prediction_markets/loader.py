"""Prediction markets data loaders (Kalshi + Polymarket)."""

from pathlib import Path

import polars as pl

from data.exceptions import DataNotFoundError
from utils import ML4T_DATA_PATH


def load_kalshi(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load Kalshi prediction market OHLCV data.

    Args:
        symbols: Optional list of market tickers to filter (e.g., ["KXFED", "KXINFL"])
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    path = ML4T_DATA_PATH / "prediction_markets" / "kalshi_events.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="Kalshi Prediction Markets",
            path=path,
            download_script="data/prediction_markets/download.py",
            readme="data/prediction_markets/README.md",
        )

    df = pl.read_parquet(path)

    # Normalize to canonical schema
    if "timestamp" in df.columns and df["timestamp"].dtype != pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Date))

    if symbols:
        df = df.filter(pl.col("symbol").is_in(symbols))
    if start_date:
        df = df.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
    if end_date:
        df = df.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())

    return df


def load_polymarket(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load Polymarket prediction market OHLCV data.

    Non-political markets only (political content filtered at download time).

    Args:
        symbols: Optional list of market slugs to filter
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    path = ML4T_DATA_PATH / "prediction_markets" / "polymarket_events.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="Polymarket Prediction Markets",
            path=path,
            download_script="data/prediction_markets/download.py",
            readme="data/prediction_markets/README.md",
        )

    df = pl.read_parquet(path)

    # Normalize to canonical schema
    if "timestamp" in df.columns and df["timestamp"].dtype != pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Date))

    if symbols:
        df = df.filter(pl.col("symbol").is_in(symbols))
    if start_date:
        df = df.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
    if end_date:
        df = df.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())

    return df
