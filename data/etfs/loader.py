"""ETF universe loader."""

import polars as pl

from data.exceptions import DataNotFoundError
from utils import ML4T_DATA_PATH
from utils.data_quality import apply_max_symbols


def list_etfs() -> list[str]:
    """List ETF symbols available in the local data store.

    Returns:
        Sorted list of ETF tickers (e.g., ``["ACWI", "AGG", ..., "XLK"]``).

    Raises:
        DataNotFoundError: If ``etfs/etf_universe.parquet`` is missing.

    Example:
        >>> list_etfs()[:3]
        ['ACWI', 'ACWX', 'AGG']
    """
    path = ML4T_DATA_PATH / "etfs" / "market" / "etf_universe.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="ETF Universe",
            path=path,
            download_script="data/etfs/market/download.py",
            readme="data/etfs/README.md",
        )
    return pl.scan_parquet(path).select("symbol").unique().collect().to_series().sort().to_list()


def load_etfs(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_symbols: int = 0,
) -> pl.DataFrame:
    """Load ETF universe for momentum case study.

    Args:
        symbols: Optional list of symbols to filter (e.g., ["SPY", "QQQ"])
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)
        max_symbols: Limit to N random symbols (0 = all). Seed-deterministic.

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    path = ML4T_DATA_PATH / "etfs" / "market" / "etf_universe.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="ETF Universe",
            path=path,
            download_script="data/etfs/market/download.py",
            readme="data/etfs/README.md",
        )

    lf = pl.scan_parquet(path)
    ts_type = lf.collect_schema()["timestamp"]

    # Apply filters lazily (parquet pushdown / row-group pruning)
    if symbols:
        lf = lf.filter(pl.col("symbol").is_in(symbols))
    if start_date:
        lit = (
            pl.lit(start_date).str.to_date()
            if ts_type == pl.Date
            else pl.lit(start_date).str.to_datetime()
        )
        lf = lf.filter(pl.col("timestamp") >= lit)
    if end_date:
        lit = (
            pl.lit(end_date).str.to_date()
            if ts_type == pl.Date
            else pl.lit(end_date).str.to_datetime()
        )
        lf = lf.filter(pl.col("timestamp") <= lit)

    # Normalize daily data to Date type (post-filter so pushdown works on raw type)
    if ts_type != pl.Date:
        lf = lf.with_columns(pl.col("timestamp").cast(pl.Date))

    return apply_max_symbols(lf.collect(), max_symbols)


def load_etfs_unadjusted(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load the traded close and share count, unadjusted for splits and distributions.

    :func:`load_etfs` returns an adjusted panel, which is what a return needs: the
    adjustment divides out distributions, and dividing both ends of a ratio by the same
    factor leaves the ratio alone. A dollar amount has no such cancellation. An adjusted
    close on an early session sits well below what the fund traded at - $87.23 against
    $126.70 for SPY on 2006-01-03 - so a turnover screen or a per-share commission read
    off it is wrong by the cumulative adjustment, and wrong by more the further back it
    looks.

    Use this series for anything denominated in dollars or in shares, and
    :func:`load_etfs` for anything denominated in returns.

    Args:
        symbols: Optional list of symbols to filter (e.g., ["SPY", "QQQ"])
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)

    Returns:
        DataFrame with columns: timestamp, symbol, close, volume

    Raises:
        DataNotFoundError: If ``etfs/etf_universe_unadjusted.parquet`` is missing.
    """
    path = ML4T_DATA_PATH / "etfs" / "market" / "etf_universe_unadjusted.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="ETF Universe (unadjusted)",
            path=path,
            download_script="data/etfs/market/download.py",
            readme="data/etfs/README.md",
        )

    lf = pl.scan_parquet(path)
    ts_type = lf.collect_schema()["timestamp"]

    if symbols:
        lf = lf.filter(pl.col("symbol").is_in(symbols))
    if start_date:
        lit = (
            pl.lit(start_date).str.to_date()
            if ts_type == pl.Date
            else pl.lit(start_date).str.to_datetime()
        )
        lf = lf.filter(pl.col("timestamp") >= lit)
    if end_date:
        lit = (
            pl.lit(end_date).str.to_date()
            if ts_type == pl.Date
            else pl.lit(end_date).str.to_datetime()
        )
        lf = lf.filter(pl.col("timestamp") <= lit)

    if ts_type != pl.Date:
        lf = lf.with_columns(pl.col("timestamp").cast(pl.Date))

    return lf.collect().sort(["symbol", "timestamp"])
