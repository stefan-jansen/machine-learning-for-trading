"""FRED macroeconomic data loader."""

from pathlib import Path

import polars as pl

from data.exceptions import DataNotFoundError
from utils import ML4T_DATA_PATH


def _load_macro_file(
    path: Path,
    *,
    dataset_name: str,
    download_script: str,
    series: list[str] | None,
    start_date: str | None,
    end_date: str | None,
) -> pl.DataFrame:
    if not path.exists():
        raise DataNotFoundError(
            dataset_name=dataset_name,
            path=path,
            download_script=download_script,
            readme="data/macro/README.md",
            requires_api_key="FRED_API_KEY",
        )

    df = pl.read_parquet(path)
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename({"date": "timestamp"})
    if df["timestamp"].dtype != pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Date))

    if start_date:
        df = df.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
    if end_date:
        df = df.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())
    if series:
        cols = ["timestamp"] + [name for name in series if name in df.columns]
        df = df.select(cols)
    return df


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
    return _load_macro_file(
        ML4T_DATA_PATH / "macro" / "fred_macro.parquet",
        dataset_name="FRED Macro Indicators",
        download_script="data/macro/download.py",
        series=series,
        start_date=start_date,
        end_date=end_date,
    )


def load_macro_initial_release(
    series: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load locally materialized ALFRED initial-release market-state data."""
    return _load_macro_file(
        ML4T_DATA_PATH / "macro" / "fred_macro_initial_release.parquet",
        dataset_name="ALFRED Initial-Release Market State",
        download_script="data/macro/download_alfred.py",
        series=series,
        start_date=start_date,
        end_date=end_date,
    )


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
