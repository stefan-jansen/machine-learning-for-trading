"""FRED macroeconomic data loader."""

from pathlib import Path

import polars as pl

from data.exceptions import DataNotFoundError
from utils import ML4T_DATA_PATH


def load_macro(
    series: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load FRED macro data including treasury yields and economic indicators.

    Args:
        series: Optional list of series to include (column names, e.g., ["DGS10", "FEDFUNDS"])
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)

    Returns:
        DataFrame with columns: date, series columns (wide format)
    """
    path = ML4T_DATA_PATH / "macro" / "fred_macro.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="FRED Macro Indicators",
            path=path,
            download_script="data/macro/download.py",
            readme="data/macro/README.md",
            requires_api_key="FRED_API_KEY",
        )

    df = pl.read_parquet(path)

    # Normalize to canonical schema
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename({"date": "timestamp"})
    if df["timestamp"].dtype != pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Date))

    # Apply filters
    if start_date:
        df = df.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
    if end_date:
        df = df.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())

    # Select specific series if requested
    if series:
        cols = ["timestamp"] + [s for s in series if s in df.columns]
        df = df.select(cols)

    return df


def load_macro_metadata() -> pl.DataFrame:
    """Load the FRED macro series metadata (series name, source, frequency, group, description).

    Companion to `load_macro()`. Useful when a notebook needs to describe or
    group the series columns returned by the main loader.

    Returns:
        DataFrame with columns: series, source_id, native_frequency, group,
        description, kind, formula.
    """
    path = ML4T_DATA_PATH / "macro" / "fred_macro_metadata.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="FRED Macro Metadata",
            path=path,
            download_script="data/macro/download.py",
            readme="data/macro/README.md",
            requires_api_key="FRED_API_KEY",
        )

    return pl.read_parquet(path)
