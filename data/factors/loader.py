"""Fama-French and AQR factor loaders."""

from pathlib import Path
from typing import Literal

import polars as pl

from data.exceptions import DataNotFoundError
from utils import ML4T_DATA_PATH


def load_ff_factors(
    dataset: Literal["ff3", "ff5", "mom"] = "ff5",
    frequency: Literal["daily", "monthly"] = "monthly",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load Fama-French factor returns.

    Args:
        dataset: "ff3" (3-factor), "ff5" (5-factor model), or "mom" (momentum factor)
        frequency: "daily" or "monthly"
        start_date: Optional start date filter (YYYY-MM-DD format)
        end_date: Optional end date filter (YYYY-MM-DD format)

    Returns:
        DataFrame with factor returns and a timestamp/date column.
    """
    filename = f"{dataset}_{frequency}.parquet"
    path = ML4T_DATA_PATH / "factors" / "fama-french" / filename
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="Fama-French Factors",
            path=path,
            download_script="data/factors/ff_download.py",
            readme="data/factors/README.md",
        )

    df = pl.read_parquet(path)

    # Normalize to canonical schema
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename({"date": "timestamp"})
    if "timestamp" in df.columns and df["timestamp"].dtype != pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Date))

    if "timestamp" in df.columns:
        if start_date:
            df = df.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
        if end_date:
            df = df.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())

    return df


def load_aqr_factors(
    dataset: Literal["qmj", "bab", "hml_devil", "vme"] = "qmj",
) -> pl.DataFrame:
    """Load AQR factor returns.

    Args:
        dataset: Factor dataset
            - "qmj": Quality Minus Junk
            - "bab": Betting Against Beta
            - "hml_devil": HML Devil (quality-adjusted value)
            - "vme": Value-Momentum Everywhere

    Returns:
        DataFrame with factor returns across geographies
    """
    # Files have inconsistent naming - use lookup
    file_map = {
        "qmj": "qmj_factors.parquet",
        "bab": "bab_factors.parquet",
        "hml_devil": "hml_devil.parquet",
        "vme": "vme_factors.parquet",
    }
    filename = file_map[dataset]
    path = ML4T_DATA_PATH / "factors" / "aqr" / filename
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="AQR Research Factors",
            path=path,
            download_script="data/factors/aqr_download.py",
            readme="data/factors/README.md",
        )

    return pl.read_parquet(path)
