"""FX pairs loader."""

from typing import Literal

import polars as pl

from data.exceptions import DataNotFoundError
from utils import ML4T_DATA_PATH
from utils.data_quality import apply_max_symbols


def list_fx_pairs(frequency: Literal["daily", "4h"] = "daily") -> list[str]:
    """List currency pairs available in the local data store.

    Args:
        frequency: Which parquet to probe (``"daily"`` or ``"4h"``). Defaults
            to ``"daily"`` — the universe is identical across frequencies.

    Returns:
        Sorted list of OANDA pair codes (e.g., ``["AUD_JPY", ..., "USD_JPY"]``).

    Raises:
        DataNotFoundError: If the relevant parquet is missing.
        ValueError: If ``frequency`` is not one of the supported values.

    Example:
        >>> list_fx_pairs()[:3]
        ['AUD_JPY', 'AUD_NZD', 'AUD_USD']
    """
    if frequency not in ("daily", "4h"):
        msg = f"frequency must be 'daily' or '4h', got {frequency!r}"
        raise ValueError(msg)
    path = ML4T_DATA_PATH / "fx" / "market" / f"{frequency}.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="FX Pairs",
            path=path,
            download_script="data/fx/market/download.py",
            requires_api_key="OANDA_API_KEY",
        )
    return pl.scan_parquet(path).select("symbol").unique().collect().to_series().sort().to_list()


def load_fx_pairs(
    frequency: Literal["daily", "4h"] = "4h",
    pairs: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_symbols: int = 0,
) -> pl.DataFrame:
    """Load FX OHLCV data from OANDA.

    Args:
        frequency: Data frequency - "daily" or "4h" (default: "4h")
        pairs: Optional list of currency pairs (e.g., ["EUR_USD", "GBP_USD"])
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)
        max_symbols: Limit to N random pairs (0 = all). Seed-deterministic.

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume

    Available pairs (20):
        Majors: EUR_USD, GBP_USD, USD_JPY, USD_CHF
        Commodity: AUD_USD, USD_CAD, NZD_USD
        Crosses: EUR_GBP, EUR_JPY, EUR_CHF, EUR_CAD, EUR_AUD,
                 GBP_JPY, GBP_CHF, GBP_AUD, AUD_JPY, CHF_JPY,
                 CAD_JPY, NZD_JPY, AUD_NZD

    Note:
        Requires OANDA_API_KEY environment variable for download.
    """
    path = ML4T_DATA_PATH / "fx" / "market" / f"{frequency}.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="FX Pairs",
            path=path,
            download_script="data/fx/market/download.py",
            requires_api_key="OANDA_API_KEY",
        )

    lf = pl.scan_parquet(path)
    ts_type = lf.collect_schema()["timestamp"]
    tz = ts_type.time_zone if isinstance(ts_type, pl.Datetime) else None

    def _ts_literal(date_str: str) -> pl.Expr:
        """Build a literal matching the column's dtype (preserves timezone if present)."""
        if ts_type == pl.Date:
            return pl.lit(date_str).str.to_date()
        lit = pl.lit(date_str).str.to_datetime()
        return lit.dt.replace_time_zone(tz) if tz else lit

    if pairs:
        lf = lf.filter(pl.col("symbol").is_in(pairs))
    if start_date:
        lf = lf.filter(pl.col("timestamp") >= _ts_literal(start_date))
    if end_date:
        if ts_type == pl.Date:
            lf = lf.filter(pl.col("timestamp") <= _ts_literal(end_date))
        else:
            # Include the entire end_date for intraday: timestamp < end_date+1day
            lf = lf.filter(pl.col("timestamp") < _ts_literal(end_date) + pl.duration(days=1))

    # Normalize daily data to Date type (post-filter so parquet pushdown works on raw type)
    if frequency == "daily" and ts_type != pl.Date:
        lf = lf.with_columns(pl.col("timestamp").cast(pl.Date))

    return apply_max_symbols(lf.collect(), max_symbols)
