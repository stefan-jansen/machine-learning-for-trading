"""Equities loaders: market (OHLCV, options, microstructure), fundamentals (SEC filings, XBRL), and positioning (13F)."""

from pathlib import Path
from typing import Literal

import polars as pl

from data.exceptions import DataNotFoundError
from utils import ML4T_DATA_PATH
from utils.data_quality import apply_max_symbols


def load_sp500_index() -> pl.DataFrame:
    """Load S&P 500 index OHLCV data (bundled with repository).

    This dataset is shipped with the ML4T repository and does not require
    any download or API keys. It provides daily S&P 500 index data from 1980.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close

    Example:
        >>> sp500 = load_sp500_index()
        >>> sp500.head()
    """
    path = ML4T_DATA_PATH / "equities" / "market" / "sp500" / "sp500.csv"
    if not path.exists():
        msg = f"S&P 500 index data not found at {path}."
        raise FileNotFoundError(msg)

    df = pl.read_csv(path, try_parse_dates=True)
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename({"date": "timestamp"})
    return df


def load_us_equities(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_symbols: int = 0,
) -> pl.DataFrame:
    """Load US equities dataset (NASDAQ Data Link, 1962-2018).

    Survivorship-bias free dataset with 3,199 US companies including delisted stocks.

    Args:
        symbols: Optional list of symbols to filter (e.g., ["AAPL", "MSFT"])
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)
        max_symbols: Limit to N random symbols (0 = all). Seed-deterministic.

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume, adj_close, etc.
    """
    path = ML4T_DATA_PATH / "equities" / "market" / "us_equities" / "us_equities.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="US Equities Dataset",
            path=path,
            download_script="data/equities/market/us_equities/download.py",
            requires_api_key="QUANDL_API_KEY",
        )

    # Lazy scan with filter pushdown into parquet (row-group pruning).
    lf = pl.scan_parquet(path)
    schema_names = lf.collect_schema().names()

    symbol_col = "ticker" if "ticker" in schema_names else "symbol"
    # Prefer canonical `timestamp` if both are present (older snapshots carry
    # both `date` and `timestamp`); rename-after-filter would otherwise collide
    # with the existing `timestamp` column. Mirrors load_sp500_daily_bars.
    if "timestamp" in schema_names:
        time_col = "timestamp"
        drop_date = "date" in schema_names
    else:
        time_col = "date"
        drop_date = False
    time_type = lf.collect_schema()[time_col]

    if symbols:
        lf = lf.filter(pl.col(symbol_col).is_in(symbols))
    if start_date:
        lit = (
            pl.lit(start_date).str.to_date()
            if time_type == pl.Date
            else pl.lit(start_date).str.to_datetime()
        )
        lf = lf.filter(pl.col(time_col) >= lit)
    if end_date:
        if time_type == pl.Date:
            lf = lf.filter(pl.col(time_col) <= pl.lit(end_date).str.to_date())
        else:
            # Use half-open interval for Datetime so non-midnight ticks on
            # `end_date` are still included (matches the intraday loaders).
            lf = lf.filter(
                pl.col(time_col) < pl.lit(end_date).str.to_datetime() + pl.duration(days=1)
            )

    if drop_date:
        lf = lf.drop("date")

    renames = {}
    if symbol_col != "symbol":
        renames[symbol_col] = "symbol"
    if time_col != "timestamp":
        renames[time_col] = "timestamp"
    if renames:
        lf = lf.rename(renames)

    if time_type != pl.Date:
        lf = lf.with_columns(pl.col("timestamp").cast(pl.Date))

    df = lf.collect()
    return apply_max_symbols(df, max_symbols)


# Resampling aggregation specs for group_by_dynamic
_TRADE_OHLCV_AGGS = [
    pl.col("open").first().alias("open"),
    pl.col("high").max().alias("high"),
    pl.col("low").min().alias("low"),
    pl.col("close").last().alias("close"),
    pl.col("volume").sum().alias("volume"),
]

_QUOTE_OHLCV_AGGS = [
    pl.col("bid_open").first().alias("bid_open"),
    pl.col("bid_high").max().alias("bid_high"),
    pl.col("bid_low").min().alias("bid_low"),
    pl.col("bid_close").last().alias("bid_close"),
    pl.col("ask_open").first().alias("ask_open"),
    pl.col("ask_high").max().alias("ask_high"),
    pl.col("ask_low").min().alias("ask_low"),
    pl.col("ask_close").last().alias("ask_close"),
]

_RESAMPLE_FREQUENCIES = {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h"}


def load_nasdaq100_bars(
    frequency: str = "1m",
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    include_quotes: bool = False,
    include_microstructure: bool = False,
    regular_hours: bool = True,
    lazy: bool = False,
    max_symbols: int = 0,
) -> pl.DataFrame | pl.LazyFrame:
    """Load AlgoSeek NASDAQ-100 bar data.

    Default: minute-frequency trade OHLCV, filtered to regular trading hours
    (09:30-16:00 ET). Supports resampling to coarser frequencies, optional
    bid/ask quote OHLCV, and a raw 60-column microstructure mode.

    Args:
        frequency: Bar frequency. ``"1m"`` returns raw minute bars (no
            resampling); ``"5m"``/``"15m"``/``"30m"``/``"1h"``/``"4h"``
            resample via ``group_by_dynamic``. Ignored when
            ``include_microstructure=True``.
        symbols: Optional list of symbols to filter.
        start_date: Optional start date (YYYY-MM-DD).
        end_date: Optional end date (YYYY-MM-DD).
        include_quotes: If True, include bid/ask OHLCV columns alongside
            trade OHLCV. Required for bid/ask-aware execution in the risk
            layer. Ignored when ``include_microstructure=True``.
        include_microstructure: If True, return all 60 raw AlgoSeek columns
            without projection, regular-hours filtering, or resampling.
            Mutually exclusive with ``frequency != "1m"`` and
            ``include_quotes``.
        regular_hours: If True (default), filter to 09:30-16:00 ET. Ignored
            when ``include_microstructure=True``.
        lazy: If True, return a LazyFrame for deferred execution.
        max_symbols: Limit to N random symbols (0 = all). Seed-deterministic.

    Returns:
        DataFrame (or LazyFrame if ``lazy=True``) with columns:

        - Default: ``[timestamp, symbol, open, high, low, close, volume]``.
        - With ``include_quotes``: adds ``bid_open, bid_high, bid_low,
          bid_close, ask_open, ask_high, ask_low, ask_close``.
        - With ``include_microstructure``: all 60 raw AlgoSeek columns.
    """
    if include_microstructure and (frequency != "1m" or include_quotes):
        msg = (
            "include_microstructure=True returns the raw schema and cannot be "
            "combined with resampling (frequency!='1m') or include_quotes."
        )
        raise ValueError(msg)

    hive_path = ML4T_DATA_PATH / "equities" / "market" / "nasdaq100" / "minute_bars"
    if not hive_path.exists() or not list(hive_path.glob("year=*")):
        raise DataNotFoundError(
            dataset_name="NASDAQ-100 Minute Bars",
            path=hive_path,
            instructions="""This dataset requires a commercial license from AlgoSeek.

To obtain the data:
  1. Contact AlgoSeek: https://www.algoseek.com/
  2. Request NASDAQ-100 Minute Bar data (2020-2021)
  3. Download data to: $ML4T_DATA_PATH/algoseek/minute_nq100/
  4. Run extraction: python data/_licensed/algoseek/nasdaq100_minute_bars.py

Note: Academic pricing may be available for educational use.""",
        )

    lf = pl.scan_parquet(hive_path / "**/*.parquet", hive_partitioning=True)

    if start_date:
        lf = lf.filter(pl.col("date") >= pl.lit(start_date).str.to_date())
    if end_date:
        lf = lf.filter(pl.col("date") <= pl.lit(end_date).str.to_date())
    if symbols:
        lf = lf.filter(pl.col("symbol").is_in(symbols))

    if include_microstructure:
        if max_symbols > 0:
            lf = apply_max_symbols(lf, max_symbols)
        return lf if lazy else lf.collect()

    trade_cols = [
        pl.col("timestamp"),
        pl.col("symbol"),
        pl.col("first_trade_price").alias("open"),
        pl.col("high_trade_price").alias("high"),
        pl.col("low_trade_price").alias("low"),
        pl.col("last_trade_price").alias("close"),
        pl.col("volume"),
    ]
    quote_cols = [
        pl.col("open_bid_price").alias("bid_open"),
        pl.col("high_bid_price").alias("bid_high"),
        pl.col("low_bid_price").alias("bid_low"),
        pl.col("close_bid_price").alias("bid_close"),
        pl.col("open_ask_price").alias("ask_open"),
        pl.col("high_ask_price").alias("ask_high"),
        pl.col("low_ask_price").alias("ask_low"),
        pl.col("close_ask_price").alias("ask_close"),
    ]
    select_cols = trade_cols + quote_cols if include_quotes else trade_cols

    lf = lf.select(select_cols).filter(
        pl.col("open").is_not_null()
        & pl.col("high").is_not_null()
        & pl.col("low").is_not_null()
        & pl.col("close").is_not_null()
    )

    if regular_hours:
        lf = lf.filter(
            (pl.col("timestamp").dt.hour() >= 10)
            | ((pl.col("timestamp").dt.hour() == 9) & (pl.col("timestamp").dt.minute() >= 30))
        ).filter(pl.col("timestamp").dt.hour() < 16)

    lf = lf.sort("symbol", "timestamp")

    if frequency != "1m":
        every = _RESAMPLE_FREQUENCIES.get(frequency)
        if every is None:
            msg = (
                f"Unsupported frequency {frequency!r}. Use: {list(_RESAMPLE_FREQUENCIES)} or '1m'."
            )
            raise ValueError(msg)
        aggs = list(_TRADE_OHLCV_AGGS)
        if include_quotes:
            aggs.extend(_QUOTE_OHLCV_AGGS)
        lf = (
            lf.sort("timestamp")
            .group_by_dynamic("timestamp", every=every, group_by="symbol")
            .agg(aggs)
            .filter(pl.col("open").is_not_null())
            .sort("timestamp", "symbol")
        )

    if max_symbols > 0:
        lf = apply_max_symbols(lf, max_symbols)

    return lf if lazy else lf.collect()


def load_sp500_daily_bars(
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list[str] | None = None,
    max_symbols: int = 0,
) -> pl.DataFrame:
    """Load AlgoSeek daily OHLCV bars for S&P 500 constituents.

    Args:
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)
        symbols: Optional list of symbols to filter
        max_symbols: Limit to N random symbols (0 = all). Seed-deterministic.

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume,
        adj_factor (cumulative price factor for split adjustment)

    Coverage: 2017-2021, ~638 symbols (S&P 500 + some changes)
    """
    path = ML4T_DATA_PATH / "equities" / "market" / "sp500" / "daily_bars.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="S&P 500 Daily Bars",
            path=path,
            instructions="""This dataset requires a commercial license from AlgoSeek.

To obtain the data:
  1. Contact AlgoSeek: https://www.algoseek.com/
  2. Request S&P 500 Daily OHLCV data (2017-2021)
  3. Download data to: $ML4T_DATA_PATH/algoseek/sp500_daily/
  4. Run extraction: python data/_licensed/algoseek/sp500_daily_bars.py

Note: Academic pricing may be available for educational use.""",
        )

    lf = pl.scan_parquet(path)
    schema_names = lf.collect_schema().names()

    time_col = "date" if "date" in schema_names else "timestamp"
    time_type = lf.collect_schema()[time_col]

    if symbols:
        lf = lf.filter(pl.col("symbol").is_in(symbols))
    if start_date:
        lit = (
            pl.lit(start_date).str.to_date()
            if time_type == pl.Date
            else pl.lit(start_date).str.to_datetime()
        )
        lf = lf.filter(pl.col(time_col) >= lit)
    if end_date:
        lit = (
            pl.lit(end_date).str.to_date()
            if time_type == pl.Date
            else pl.lit(end_date).str.to_datetime()
        )
        lf = lf.filter(pl.col(time_col) <= lit)

    # Canonical schema: rename date → timestamp (handle "date and timestamp both present")
    if "date" in schema_names and "timestamp" not in schema_names:
        lf = lf.rename({"date": "timestamp"})
    elif "date" in schema_names and "timestamp" in schema_names:
        lf = lf.drop("date")

    # Cast surviving timestamp column to Date (post-filter; checks final dtype)
    if lf.collect_schema()["timestamp"] != pl.Date:
        lf = lf.with_columns(pl.col("timestamp").cast(pl.Date))

    data = lf.collect()
    return apply_max_symbols(data, max_symbols)


def load_sp500_options(
    symbols: list[str] | None = None,
    option_type: Literal["C", "P", "all"] = "all",
    start_date: str | None = None,
    end_date: str | None = None,
    include_greeks: bool = True,
    lazy: bool = False,
    max_symbols: int = 0,
) -> pl.DataFrame | pl.LazyFrame:
    """Load RAW AlgoSeek options Greeks data for S&P 500 constituents.

    WARNING: Raw data is 347M rows / 11GB on disk / ~30GB in RAM.
    Do NOT collect the full dataset eagerly. For pipeline work, use:
      - load_sp500_options_surface() — daily IV surface summary
      - load_sp500_options_straddles() — daily ATM straddle data
    This loader is for EDA and deep-dive analysis only.

    Uses lazy scanning with Hive partition pruning for efficient loading.
    Only reads data matching the requested date range and symbols.

    Args:
        symbols: Optional list of underlying symbols to filter (e.g., ["AAPL", "MSFT"])
        option_type: "C" for calls only, "P" for puts only, "all" for both
        start_date: Optional start date (YYYY-MM-DD format). Default: earliest available
        end_date: Optional end date (YYYY-MM-DD format). Default: latest available
        include_greeks: If True, include delta, gamma, theta, vega, rho columns
        lazy: If True, return LazyFrame for deferred execution. Default: False
        max_symbols: Limit to N random symbols (0 = all). Seed-deterministic.

    Returns:
        DataFrame (or LazyFrame if lazy=True) with columns:
            date, symbol, call_put, option_style, strike, expiration,
            years_to_maturity, days_to_maturity, underlying_price,
            bid, ask, mid_price, implied_vol, theo_price,
            delta, gamma, theta, vega, rho (if include_greeks=True),
            iv_convergence

    Coverage: 2017-2021, ~500 S&P 500 constituents, all listed options

    Example:
        >>> # Load AAPL calls for 2020
        >>> calls = load_sp500_options(symbols=["AAPL"], option_type="C",
        ...                            start_date="2020-01-01", end_date="2020-12-31")
        >>> # Lazy load for large queries
        >>> lf = load_sp500_options(lazy=True)
        >>> filtered = lf.filter(pl.col("implied_vol") > 0.5).collect()
    """
    # Canonical path: Hive-partitioned options data
    base_path = ML4T_DATA_PATH / "equities" / "market" / "sp500" / "options"
    if not base_path.exists() or not list(base_path.glob("year=*")):
        raise DataNotFoundError(
            dataset_name="S&P 500 Options Greeks",
            path=base_path,
            instructions="""This dataset requires a commercial license from AlgoSeek.

To obtain the data:
  1. Contact AlgoSeek: https://www.algoseek.com/
  2. Request S&P 500 Options Greeks data (2017-2021)
  3. Download data to: $ML4T_DATA_PATH/algoseek/options_sp500/
  4. Run extraction: python data/_licensed/algoseek/sp500_options.py

Note: Academic pricing may be available for educational use.""",
        )

    # Use lazy scan with Hive partitioning
    lf = pl.scan_parquet(
        base_path / "**/*.parquet",
        hive_partitioning=True,
    )

    # Normalize: rename date→timestamp for canonical schema, cast to Date
    if "date" in lf.collect_schema().names():
        lf = lf.with_columns(pl.col("date").cast(pl.Date).alias("timestamp")).drop("date")
    elif lf.collect_schema()["timestamp"] != pl.Date:
        lf = lf.with_columns(pl.col("timestamp").cast(pl.Date))

    # Apply date filters using timestamp
    if start_date:
        lf = lf.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
    if end_date:
        lf = lf.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())

    # Apply symbol filter (predicate pushdown)
    if symbols:
        lf = lf.filter(pl.col("symbol").is_in(symbols))

    # Apply option type filter
    if option_type != "all":
        lf = lf.filter(pl.col("call_put") == option_type)

    # Remove Greeks if not requested
    if not include_greeks:
        greek_cols = ["delta", "gamma", "theta", "vega", "rho"]
        existing = [c for c in greek_cols if c in lf.collect_schema().names()]
        if existing:
            lf = lf.drop(existing)

    # Sort by timestamp, symbol, expiration, strike
    lf = lf.sort(["timestamp", "symbol", "expiration", "strike"])

    if max_symbols > 0:
        lf = apply_max_symbols(lf, max_symbols)

    return lf if lazy else lf.collect()


_SP500_OPTIONS_S3_BASE = "https://algoseek-public.s3.amazonaws.com/ml4t/sp500_options"


def load_sp500_options_eda(
    symbols: list[str] | None = None,
    option_type: Literal["C", "P", "all"] = "all",
    start_date: str | None = None,
    end_date: str | None = None,
    include_greeks: bool = True,
    max_symbols: int = 0,
) -> pl.DataFrame:
    """Load the S&P 500 options EDA dataset.

    Daily option chains for AAPL, MSFT, GOOGL, AMZN, JPM, BA, XOM, KO over
    2019-2020, partitioned by year so the files can be downloaded
    independently. Full AlgoSeek schema (Greeks, implied vol, diagnostics).

    Args:
        symbols: Optional subset of the eight available symbols.
        option_type: "C" for calls only, "P" for puts only, "all" for both.
        start_date: Optional start date (YYYY-MM-DD).
        end_date: Optional end date (YYYY-MM-DD).
        include_greeks: If False, drop delta, gamma, theta, vega, rho.
        max_symbols: Limit to N random symbols (0 = all).

    Returns:
        DataFrame with columns: timestamp, symbol, call_put, option_style,
        strike, expiration, years_to_maturity, days_to_maturity,
        underlying_price, bid, ask, mid_price, implied_vol, theo_price,
        delta, gamma, theta, vega, rho, iv_convergence.
    """
    base_path = ML4T_DATA_PATH / "equities" / "market" / "sp500" / "options_eda"
    if not base_path.exists() or not list(base_path.glob("year=*.parquet")):
        raise DataNotFoundError(
            dataset_name="S&P 500 Options — EDA subset",
            path=base_path,
            download_url=f"{_SP500_OPTIONS_S3_BASE}/options_eda/",
        )

    lf = pl.scan_parquet(base_path / "year=*.parquet", hive_partitioning=True)

    if "date" in lf.collect_schema().names():
        lf = lf.with_columns(pl.col("date").cast(pl.Date).alias("timestamp")).drop("date")
    elif lf.collect_schema()["timestamp"] != pl.Date:
        lf = lf.with_columns(pl.col("timestamp").cast(pl.Date))

    if start_date:
        lf = lf.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
    if end_date:
        lf = lf.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())
    if symbols:
        lf = lf.filter(pl.col("symbol").is_in(symbols))
    if option_type != "all":
        lf = lf.filter(pl.col("call_put") == option_type)
    if not include_greeks:
        greek_cols = [
            c
            for c in ("delta", "gamma", "theta", "vega", "rho")
            if c in lf.collect_schema().names()
        ]
        if greek_cols:
            lf = lf.drop(greek_cols)

    df = lf.sort(["timestamp", "symbol", "expiration", "call_put", "strike"]).collect()
    return apply_max_symbols(df, max_symbols)


def load_sp500_options_straddles_raw(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lazy: bool = False,
    max_symbols: int = 0,
) -> pl.DataFrame | pl.LazyFrame:
    """Load the lifecycle-preserving ATM-band raw option chains used by the
    sp500_options straddle case study.

    Contains every daily observation (both legs) of each (symbol, strike,
    expiration) contract that enters the 30D ATM straddle candidate window
    (DTE ∈ [25, 35], |delta| ∈ [0.35, 0.65], converged IV, tight spread) at
    any point during 2017-2021 — from first listing through expiration.

    Args:
        symbols: Optional list of underlying symbols to filter.
        start_date: Optional start date (YYYY-MM-DD).
        end_date: Optional end date (YYYY-MM-DD).
        lazy: If True, return a LazyFrame for deferred execution.
        max_symbols: Limit to N random symbols (0 = all).

    Returns:
        DataFrame (or LazyFrame) with the full AlgoSeek option-chain schema.
    """
    base_path = ML4T_DATA_PATH / "equities" / "market" / "sp500" / "options_straddles_raw"
    if not base_path.exists() or not list(base_path.glob("year=*.parquet")):
        raise DataNotFoundError(
            dataset_name="S&P 500 Options — ATM-band straddle raw chains",
            path=base_path,
            download_url=f"{_SP500_OPTIONS_S3_BASE}/options_straddles_raw.tar.zst",
        )

    lf = pl.scan_parquet(base_path / "year=*.parquet", hive_partitioning=True)

    if "date" in lf.collect_schema().names():
        lf = lf.with_columns(pl.col("date").cast(pl.Date).alias("timestamp")).drop("date")
    elif lf.collect_schema()["timestamp"] != pl.Date:
        lf = lf.with_columns(pl.col("timestamp").cast(pl.Date))

    if start_date:
        lf = lf.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
    if end_date:
        lf = lf.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())
    if symbols:
        lf = lf.filter(pl.col("symbol").is_in(symbols))

    lf = lf.sort(["timestamp", "symbol", "expiration", "call_put", "strike"])

    if max_symbols > 0:
        lf = apply_max_symbols(lf, max_symbols)

    return lf if lazy else lf.collect()


def load_sp500_options_surface(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_symbols: int = 0,
) -> pl.DataFrame:
    """Load the daily IV surface summary for S&P 500 options.

    One row per (symbol, date) with ATM implied vol at 7d/30d/90d tenors,
    25-delta risk reversal, term structure slope/ratio, bid-ask spread, and
    IV convergence quality.

    Columns: timestamp, symbol, iv_30_atm, iv_7_atm, iv_90_atm,
    iv_30_put_25d, iv_30_call_25d, skew_rr_30_25d, spread_atm_30,
    qc_converged_share, term_slope_near_atm, term_slope_far_atm,
    term_ratio_atm, term_convexity, skew_to_atm_ratio.

    Used by sp500_equity_option_analytics/03_financial_features.py, which
    also shows how this summary is computed from raw option chains.
    """
    path = ML4T_DATA_PATH / "equities" / "market" / "sp500" / "options_surface_daily.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="S&P 500 Options — Daily IV Surface",
            path=path,
            download_url=f"{_SP500_OPTIONS_S3_BASE}/options_surface_daily.parquet",
            derivation_notebook=(
                "case_studies/sp500_equity_option_analytics/03_financial_features.py"
            ),
        )
    df = pl.read_parquet(path)
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename({"date": "timestamp"})
    if start_date:
        df = df.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
    if end_date:
        df = df.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())
    if symbols:
        df = df.filter(pl.col("symbol").is_in(symbols))
    df = apply_max_symbols(df, max_symbols)
    return df.sort(["timestamp", "symbol"])


def load_sp500_options_straddles(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_symbols: int = 0,
) -> pl.DataFrame:
    """Load the daily 30D ATM straddle dataset for S&P 500 options.

    One row per (symbol, date) with call and put leg details at matched
    strike and expiration, plus straddle-level aggregates (mid, bid/ask,
    spread, delta, gamma, theta, vega, IV, DTE). Used by
    sp500_options/04_financial_features.py.
    """
    path = ML4T_DATA_PATH / "equities" / "market" / "sp500" / "options_straddles_daily.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="S&P 500 Options — Daily 30D ATM Straddles",
            path=path,
            download_url=f"{_SP500_OPTIONS_S3_BASE}/options_straddles_daily.parquet",
            derivation_notebook="data/equities/market/sp500/materialize_options.py",
        )
    lf = pl.scan_parquet(path)
    schema_names = lf.collect_schema().names()
    time_col = "date" if "date" in schema_names else "timestamp"
    time_type = lf.collect_schema()[time_col]

    if symbols:
        lf = lf.filter(pl.col("symbol").is_in(symbols))
    if start_date:
        lit = (
            pl.lit(start_date).str.to_date()
            if time_type == pl.Date
            else pl.lit(start_date).str.to_datetime()
        )
        lf = lf.filter(pl.col(time_col) >= lit)
    if end_date:
        lit = (
            pl.lit(end_date).str.to_date()
            if time_type == pl.Date
            else pl.lit(end_date).str.to_datetime()
        )
        lf = lf.filter(pl.col(time_col) <= lit)

    if time_col == "date" and "timestamp" not in schema_names:
        lf = lf.rename({"date": "timestamp"})

    df = lf.collect()
    df = apply_max_symbols(df, max_symbols)
    return df.sort(["symbol", "timestamp"])


def load_nasdaq100_taq(
    symbols: list[str] | None = None,
    event_types: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load AlgoSeek TAQ tick data for March 2020 (COVID crash period).

    High-frequency tick data including trades and quotes with nanosecond
    precision timestamps. Data covers AAPL, AMZN, MSFT during the
    March 2020 market crash - ideal for studying market microstructure
    during extreme volatility.

    Args:
        symbols: Optional list of symbols to filter (e.g., ["AAPL"])
                Available: AAPL, AMZN, MSFT
        event_types: Optional list of event types to filter. Available:
            - "TRADE": Executed trades
            - "TRADE NB": Non-binding trades
            - "TRADE CANCELLED": Trade cancellations
            - "QUOTE BID" / "QUOTE ASK": NBBO quotes
            - "QUOTE BID NB" / "QUOTE ASK NB": Non-binding quotes
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)

    Returns:
        DataFrame with columns:
            timestamp (microsecond precision), symbol, event_type,
            price, quantity, exchange, conditions

    Coverage: 21 trading days in March 2020, 3 tickers (~500M rows total)
    """
    base_path = ML4T_DATA_PATH / "equities" / "market" / "microstructure" / "trade_and_quotes"

    algoseek_instructions = """This dataset requires a commercial license from AlgoSeek.

To obtain the data:
  1. Contact AlgoSeek: https://www.algoseek.com/
  2. Request Trade and Quote (TAQ) data for March 2020
  3. Download data to: $ML4T_DATA_PATH/algoseek/taq/
  4. Run extraction: python data/_licensed/algoseek/trade_and_quotes.py

Note: Academic pricing may be available for educational use."""

    if not base_path.exists() or not list(base_path.glob("symbol=*")):
        raise DataNotFoundError(
            dataset_name="Trade and Quotes Tick Data",
            path=base_path,
            instructions=algoseek_instructions,
        )

    # Load data based on symbol filter
    if symbols:
        # Read specific symbol partitions
        dfs = []
        for s in symbols:
            pattern = base_path / f"symbol={s}" / "*.parquet"
            files = list(pattern.parent.glob(pattern.name))
            if files:
                df = pl.read_parquet(files).with_columns(pl.lit(s).alias("symbol"))
                dfs.append(df)
        if not dfs:
            raise DataNotFoundError(
                dataset_name="Trade and Quotes Tick Data",
                path=base_path,
                instructions=algoseek_instructions,
            )
        data = pl.concat(dfs, how="diagonal_relaxed")
    else:
        # Read all with hive partitioning (symbol only)
        data = pl.read_parquet(
            base_path / "**/*.parquet",
            hive_partitioning=True,
        )

    # Apply filters
    if event_types:
        data = data.filter(pl.col("event_type").is_in(event_types))
    if start_date:
        data = data.filter(pl.col("timestamp").dt.date() >= pl.lit(start_date).str.to_date())
    if end_date:
        data = data.filter(pl.col("timestamp").dt.date() <= pl.lit(end_date).str.to_date())

    return data.sort(["symbol", "timestamp"])


def load_mbo_data(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    list_files: bool = False,
) -> pl.DataFrame | list[Path]:
    """Load DataBento MBO (Market-By-Order) tick data.

    High-frequency order book data from NASDAQ ITCH via DataBento API.
    Includes individual order messages (add, cancel, fill, modify, trade).

    Args:
        symbols: Optional list of symbols to filter (e.g., ["NVDA"])
                Available: NVDA (Nov 2024, 10 trading days)
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)
        list_files: If True, return list of file paths instead of loading data.
            Useful for notebooks that need to iterate files day-by-day.

    Returns:
        If list_files=False (default): DataFrame with columns:
            ts_event, symbol, action, side, price, size, order_id, flags
        If list_files=True: List of Path objects to parquet files

    Coverage: 10 trading days in November 2024, NVDA only

    Example:
        >>> # Load all data
        >>> df = load_mbo_data(symbols=["NVDA"])
        >>> # Load date range
        >>> df = load_mbo_data(symbols=["NVDA"], start_date="2024-11-04", end_date="2024-11-08")
        >>> # Get file paths for iteration
        >>> files = load_mbo_data(symbols=["NVDA"], list_files=True)
        >>> for f in files:
        ...     day_df = pl.read_parquet(f)
    """
    base_path = ML4T_DATA_PATH / "equities" / "market" / "microstructure" / "market_by_order"
    if not base_path.exists():
        raise DataNotFoundError(
            dataset_name="DataBento MBO Data",
            path=base_path,
            instructions=(
                "Recommended: manual download from the Databento Download Center.\n"
                "  See data/equities/market/microstructure/MBO_DOWNLOAD.md for step-by-step\n"
                "  instructions (XNAS.ITCH, schema mbo, NVDA, 2024-11-04 to 2024-11-15,\n"
                "  Parquet output, ~$5).\n"
                "\n"
                "Alternative: API script (requires DATABENTO_API_KEY in .env):\n"
                "  uv run python data/equities/market/microstructure/mbo_download.py --estimate-only\n"
                "  uv run python data/equities/market/microstructure/mbo_download.py"
            ),
        )

    # Find available symbols
    available_symbols = [d.name for d in base_path.iterdir() if d.is_dir()]
    if not available_symbols:
        raise DataNotFoundError(
            dataset_name="DataBento MBO Data",
            path=base_path,
            instructions=(
                "Recommended: manual download from the Databento Download Center.\n"
                "  See data/equities/market/microstructure/MBO_DOWNLOAD.md for step-by-step\n"
                "  instructions (XNAS.ITCH, schema mbo, NVDA, 2024-11-04 to 2024-11-15,\n"
                "  Parquet output, ~$5).\n"
                "\n"
                "Alternative: API script (requires DATABENTO_API_KEY in .env):\n"
                "  uv run python data/equities/market/microstructure/mbo_download.py --estimate-only\n"
                "  uv run python data/equities/market/microstructure/mbo_download.py"
            ),
        )

    # Filter symbols
    load_symbols = symbols if symbols else available_symbols
    load_symbols = [s for s in load_symbols if s in available_symbols]

    if not load_symbols:
        available = ", ".join(available_symbols)
        msg = f"No matching symbols found. Available: {available}"
        raise ValueError(msg)

    # Collect files
    all_files = []
    for symbol in load_symbols:
        symbol_dir = base_path / symbol
        all_files.extend(sorted(symbol_dir.glob("*.parquet")))

    if not all_files:
        raise DataNotFoundError(
            dataset_name="DataBento MBO Data",
            path=base_path,
            instructions=(
                "Recommended: manual download from the Databento Download Center.\n"
                "  See data/equities/market/microstructure/MBO_DOWNLOAD.md for step-by-step\n"
                "  instructions (XNAS.ITCH, schema mbo, NVDA, 2024-11-04 to 2024-11-15,\n"
                "  Parquet output, ~$5).\n"
                "\n"
                "Alternative: API script (requires DATABENTO_API_KEY in .env):\n"
                "  uv run python data/equities/market/microstructure/mbo_download.py --estimate-only\n"
                "  uv run python data/equities/market/microstructure/mbo_download.py"
            ),
        )

    # Return file list if requested
    if list_files:
        return sorted(all_files)

    # Load data
    dfs = []
    for f in all_files:
        df = pl.read_parquet(f)
        # Infer symbol from directory structure
        symbol = f.parent.name
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit(symbol).alias("symbol"))
        dfs.append(df)

    data = pl.concat(dfs, how="diagonal_relaxed")

    # Normalize: ts_event → timestamp for canonical schema
    if "ts_event" in data.columns and "timestamp" not in data.columns:
        data = data.rename({"ts_event": "timestamp"})

    # Apply date filters
    if start_date:
        data = data.filter(pl.col("timestamp").dt.date() >= pl.lit(start_date).str.to_date())
    if end_date:
        data = data.filter(pl.col("timestamp").dt.date() <= pl.lit(end_date).str.to_date())

    return data.sort(["symbol", "timestamp"])


def load_nasdaq_itch(
    message_types: list[str] | None = None,
    symbols: list[str] | None = None,
    get_base_path: bool = False,
) -> pl.DataFrame | Path:
    """Load parsed NASDAQ ITCH message data.

    Pre-parsed ITCH protocol messages from NASDAQ TotalView-ITCH feed.
    Messages are organized by type in separate partitions.

    Args:
        message_types: Optional list of message types to filter. Available:
            - "A": Add Order (no attribution)
            - "F": Add Order (with attribution)
            - "E": Order Executed
            - "C": Order Executed with Price
            - "X": Order Cancel
            - "D": Order Delete
            - "U": Order Replace
            - "P": Trade (non-cross)
            - "Q": Cross Trade
            - "I": Imbalance
            - "S": System Event
            - "R": Stock Directory
            - "H": Stock Trading Action
        symbols: Optional list of stock symbols to filter (e.g., ["AAPL", "MSFT"])
        get_base_path: If True, return the resolved base path instead of loading data.
            Useful for notebooks that need direct access to message type directories.

    Returns:
        If get_base_path=False (default): DataFrame with message-type-specific columns.
            Common columns include: timestamp, stock, order_reference_number, shares, price
        If get_base_path=True: Path to ITCH messages directory

    Note:
        Raw ITCH files (~5GB each) must first be parsed using the Rust parser
        or Python parser in Chapter 3 notebooks.

    Example:
        >>> # Load all add orders
        >>> df = load_nasdaq_itch(message_types=["A", "F"])
        >>> # Get base path for custom access
        >>> itch_path = load_nasdaq_itch(get_base_path=True)
        >>> add_orders = pl.read_parquet(itch_path / "A")
    """
    base_path = (
        ML4T_DATA_PATH / "equities" / "market" / "microstructure" / "nasdaq_itch" / "messages"
    )
    if not base_path.exists():
        raise DataNotFoundError(
            dataset_name="NASDAQ ITCH Parsed Messages",
            path=base_path,
            download_script="data/equities/market/microstructure/nasdaq_itch_download.py",
        )

    # Return base path if requested
    if get_base_path:
        return base_path

    # Find available message types (single uppercase letter directories only)
    available_types = [
        d.name for d in base_path.iterdir() if d.is_dir() and len(d.name) == 1 and d.name.isupper()
    ]
    if not available_types:
        raise DataNotFoundError(
            dataset_name="NASDAQ ITCH Parsed Messages",
            path=base_path,
            download_script="data/equities/market/microstructure/nasdaq_itch_download.py",
        )

    # Filter message types
    load_types = message_types if message_types else available_types
    load_types = [t for t in load_types if t in available_types]

    if not load_types:
        available = ", ".join(sorted(available_types))
        msg = f"No matching message types found. Available: {available}"
        raise ValueError(msg)

    # Load data
    dfs = []
    for msg_type in load_types:
        type_dir = base_path / msg_type
        files = sorted(type_dir.glob("*.parquet"))
        if files:
            df = pl.read_parquet(files)
            if "message_type" not in df.columns:
                df = df.with_columns(pl.lit(msg_type).alias("message_type"))
            dfs.append(df)

    if not dfs:
        raise DataNotFoundError(
            dataset_name="NASDAQ ITCH Parsed Messages",
            path=base_path,
            download_script="data/equities/market/microstructure/nasdaq_itch_download.py",
        )

    data = pl.concat(dfs, how="diagonal_relaxed")

    # Filter by symbol if requested
    if symbols:
        # ITCH uses 'stock' column
        stock_col = "stock" if "stock" in data.columns else "symbol"
        if stock_col in data.columns:
            data = data.filter(pl.col(stock_col).is_in(symbols))

    return data


def load_firm_characteristics(
    split: Literal["all", "train", "valid", "test"] = "all",
    include_macro: bool = False,
) -> pl.DataFrame:
    """Load firm characteristics dataset for ML-based asset pricing.

    Chen-Pelger-Zhu (2020) anonymized dataset with ~180 firm characteristics
    (accounting ratios, price-based measures) and monthly returns for US equities.
    Standard benchmark for Chapters 10-16.

    Args:
        split: Which split to load ("all", "train", "valid", "test")
               - train: 1967-1989 (~70%)
               - valid: 1990-1999 (~15%)
               - test: 2000-2016 (~15%)
        include_macro: Whether to include macro columns

    Returns:
        DataFrame with ~180 firm characteristics and returns
    """
    # Use canonical firm_characteristics_*.parquet naming
    if split == "valid":
        # Valid split derived from full dataset
        filename = "firm_characteristics_all.parquet"
    else:
        filename = f"firm_characteristics_{split}.parquet"

    path = ML4T_DATA_PATH / "equities" / "firm_characteristics" / filename

    if not path.exists():
        raise DataNotFoundError(
            dataset_name="Firm Characteristics Dataset (Chen-Pelger-Zhu 2020)",
            path=path,
            download_script="data/equities/firm_characteristics/download.py",
        )

    df = pl.read_parquet(path)

    # Normalize: date → timestamp for canonical schema
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename({"date": "timestamp"})
    if df["timestamp"].dtype != pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Date))

    # Add split column based on year (matches Chen-Pelger-Zhu methodology)
    if "split" not in df.columns and "timestamp" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("timestamp").dt.year() < 1990)
            .then(pl.lit("train"))
            .when(pl.col("timestamp").dt.year() < 2000)
            .then(pl.lit("valid"))
            .otherwise(pl.lit("test"))
            .alias("split")
        )

    # Filter to requested split if not "all"
    if split != "all" and "split" in df.columns:
        df = df.filter(pl.col("split") == split)

    if not include_macro:
        # Filter out macro columns if present
        macro_cols = [c for c in df.columns if c.startswith("macro_")]
        if macro_cols:
            df = df.drop(macro_cols)

    return df


def load_iex_hist(
    feed: Literal["tops", "deep"] = "deep",
    data_type: Literal["quotes", "trades", "price_levels", "all"] = "all",
    symbols: list[str] | None = None,
    dates: list[str] | None = None,
    get_raw_files: bool = False,
) -> pl.DataFrame | list[Path]:
    """Load IEX HIST market data (free public data from IEX Exchange).

    IEX provides free historical market data with 12 months rolling history.
    Data must be downloaded first using iex_hist.py, then parsed by the
    iex_lob_reconstruction notebook.

    Args:
        feed: Which feed to load:
            - "deep": Full depth of book (price level updates) - required for LOB
            - "tops": Top of book only (BBO quotes and trades)
        data_type: Which data type to load:
            - "quotes": BBO quote updates (bid/ask prices and sizes)
            - "trades": Trade executions
            - "price_levels": Price level updates (DEEP only, for LOB reconstruction)
            - "all": All available data types
        symbols: Optional list of symbols to filter (e.g., ["AAPL", "SPY"])
        dates: Optional list of dates to filter (YYYYMMDD format)
        get_raw_files: If True, return list of raw pcap file paths instead of
            loading parsed data. Useful for notebooks that need to parse data.

    Returns:
        If get_raw_files=False (default): DataFrame with requested data
        If get_raw_files=True: List of Path objects to raw pcap.gz files

    Note:
        Raw pcap files must be parsed before use. The iex_lob_reconstruction
        notebook handles parsing and saves results to the canonical location.

    Attribution:
        Data provided for free by IEX. By accessing or using IEX Historical Data,
        you agree to the IEX Historical Data Terms of Use:
        https://www.iexexchange.io/legal/hist-data-terms

    Example:
        >>> # Load all DEEP data
        >>> df = load_iex_hist(feed="deep")
        >>> # Load only trades for specific symbols
        >>> trades = load_iex_hist(feed="tops", data_type="trades", symbols=["AAPL"])
        >>> # Get raw files for custom parsing
        >>> raw_files = load_iex_hist(feed="deep", get_raw_files=True)
    """
    feed = feed.lower()
    if feed not in ["tops", "deep"]:
        raise ValueError("feed must be 'tops' or 'deep'")

    base_path = ML4T_DATA_PATH / "equities" / "market" / "microstructure" / "iex" / feed

    if not base_path.exists():
        raise DataNotFoundError(
            dataset_name=f"IEX HIST {feed.upper()} Data",
            path=base_path,
            download_script=f"data/equities/market/microstructure/iex_download.py --{'deep' if feed == 'deep' else 'smallest'}",
        )

    # Return raw pcap files if requested
    if get_raw_files:
        raw_files = sorted(base_path.glob("*.pcap.gz"))
        if not raw_files:
            raise DataNotFoundError(
                dataset_name=f"IEX HIST {feed.upper()} Raw Files",
                path=base_path,
                download_script="data/equities/market/microstructure/iex_download.py",
            )
        return raw_files

    # Look for parsed parquet files
    parsed_path = base_path / "parsed"
    if not parsed_path.exists() or not list(parsed_path.glob("**/*.parquet")):
        # Check if raw files exist but aren't parsed
        raw_files = list(base_path.glob("*.pcap.gz"))
        if raw_files:
            raise DataNotFoundError(
                dataset_name=f"IEX HIST {feed.upper()} Parsed Data",
                path=parsed_path,
                download_script="See notebook: 16_iex_lob_reconstruction.py (run to parse raw files)",
            )
        raise DataNotFoundError(
            dataset_name=f"IEX HIST {feed.upper()} Data",
            path=base_path,
            download_script="data/equities/market/microstructure/iex_download.py",
        )

    # Determine which data types to load
    if data_type == "all":
        if feed == "deep":
            load_types = ["quotes", "trades", "price_levels"]
        else:
            load_types = ["quotes", "trades"]
    else:
        if data_type == "price_levels" and feed == "tops":
            raise ValueError("price_levels only available with DEEP feed")
        load_types = [data_type]

    # Load data
    dfs = []
    for dtype in load_types:
        type_path = parsed_path / dtype
        if not type_path.exists():
            continue

        # Find parquet files, optionally filtering by date
        if dates:
            files = []
            for d in dates:
                files.extend(type_path.glob(f"{d}*.parquet"))
        else:
            files = sorted(type_path.glob("*.parquet"))

        if files:
            df = pl.read_parquet(files)
            if "data_type" not in df.columns:
                df = df.with_columns(pl.lit(dtype).alias("data_type"))
            dfs.append(df)

    if not dfs:
        raise DataNotFoundError(
            dataset_name=f"IEX HIST {feed.upper()} {data_type} Data",
            path=parsed_path,
            download_script="See notebook: 16_iex_lob_reconstruction.py",
        )

    data = pl.concat(dfs, how="diagonal_relaxed")

    # Filter by symbol if requested
    if symbols and "symbol" in data.columns:
        data = data.filter(pl.col("symbol").is_in(symbols))

    return data.sort("timestamp")


# --- Fundamentals: SEC filings (10-K, 10-Q, 8-K) + XBRL ---


def load_sec_filings(
    form_type: str = "10-K",
    universe: str = "sp100",
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load SEC filing text from the consolidated aggregate parquet.

    The same canonical schema is returned for every form type:

        symbol, cik, form, filing_date, period_end, accession_no,
        company_name, year, text, text_length

    Args:
        form_type: SEC form type — ``"10-K"``, ``"10-Q"``, or ``"8-K"``.
        universe: Symbol universe — ``"sp100"`` (full-text 10-K/8-K) or
            ``"sp500"`` (10-Q MD&A).
        symbols: Optional symbol filter.
        start_date: Optional filing_date start (``YYYY-MM-DD``).
        end_date: Optional filing_date end (``YYYY-MM-DD``).

    Download first:
        python data/equities/fundamentals/filings_download.py \\
            --form {form_type} --universe {universe}
    """
    form_slug = form_type.lower().replace("-", "")
    path = (
        ML4T_DATA_PATH
        / "equities"
        / "fundamentals"
        / form_slug
        / universe
        / "reference"
        / f"all_{form_slug}_filings.parquet"
    )

    if not path.exists():
        raise DataNotFoundError(
            dataset_name=f"SEC {form_type} Filings ({universe})",
            path=path,
            download_script=f"data/equities/fundamentals/filings_download.py --form {form_type} --universe {universe}",
            readme="data/equities/fundamentals/README.md",
        )

    data = pl.read_parquet(path)

    if symbols:
        data = data.filter(pl.col("symbol").is_in(symbols))
    if start_date:
        data = data.filter(pl.col("filing_date") >= pl.lit(start_date).str.to_date())
    if end_date:
        data = data.filter(pl.col("filing_date") <= pl.lit(end_date).str.to_date())

    return data


def load_sp500_10q_mda(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Convenience wrapper for ``load_sec_filings("10-Q", "sp500")``.

    The MD&A text lives in the canonical ``text`` column; pre-canonical
    consumers that referenced ``mda_text`` should rename to ``text``.

    Source: SEC EDGAR (public domain, redistributable).
    Coverage: 2017-2021, ~600 S&P 500 constituents, ~9,000 quarterly filings.
    """
    return load_sec_filings(
        form_type="10-Q",
        universe="sp500",
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )


def iter_sec_filings(
    form_type: str = "10-K",
    universe: str = "sp100",
    symbols: list[str] | None = None,
):
    """Yield per-filing dict records for filesystem-style iteration.

    Replaces the legacy per-ticker parquet directory layout. Each yielded
    record has the canonical schema — readers that previously walked the
    ``{form}/{universe}/{TICKER}/{YEAR}.parquet`` tree should iterate this
    generator instead.

    Args:
        form_type: SEC form type — ``"10-K"``, ``"10-Q"``, or ``"8-K"``.
        universe: Symbol universe (``"sp100"`` or ``"sp500"``).
        symbols: Optional symbol filter.

    Yields:
        dict with keys: symbol, cik, form, filing_date, period_end,
        accession_no, company_name, year, text, text_length
    """
    df = load_sec_filings(form_type=form_type, universe=universe, symbols=symbols)
    yield from df.iter_rows(named=True)


def load_sec_xbrl_fundamentals(
    concepts: list[str] | None = None,
    years: list[int] | None = None,
    symbols: list[str] | None = None,
    ciks: list[int] | None = None,
) -> pl.DataFrame:
    """Load SEC XBRL fundamentals panel (CIK × quarter × concept).

    Produced by `data/equities/fundamentals/xbrl_download.py`. Sourced from
    the SEC XBRL Frames API (cross-sectional bulk snapshots) with
    filing dates joined from the Submissions API for point-in-time
    correctness.

    Args:
        concepts: Optional list of us-gaap concept names to keep as
            columns (case-insensitive, e.g. ["Assets", "Revenues"]).
            Default returns all fetched concepts.
        years: Optional list of calendar years to filter on
            `fiscal_quarter_end`.
        symbols: Optional ticker filter.
        ciks: Optional CIK filter.

    Returns:
        DataFrame with columns: symbol, cik, entity_name,
        fiscal_quarter_end (Date), announcement_date (Date), accession,
        and one lowercase column per requested concept.

    Note:
        Use `announcement_date` for point-in-time backtesting — not
        `fiscal_quarter_end`, which introduces lookahead bias.
    """
    path = ML4T_DATA_PATH / "equities" / "fundamentals" / "xbrl" / "fundamentals.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="SEC XBRL Fundamentals",
            path=path,
            download_script="data/equities/fundamentals/xbrl_download.py",
            readme="data/equities/fundamentals/README.md",
        )

    data = pl.read_parquet(path)

    if symbols:
        data = data.filter(pl.col("symbol").is_in(symbols))
    if ciks:
        data = data.filter(pl.col("cik").is_in(ciks))
    if years:
        data = data.filter(pl.col("fiscal_quarter_end").dt.year().is_in(years))

    if concepts:
        wanted = {c.lower() for c in concepts}
        leading = [
            c
            for c in (
                "symbol",
                "cik",
                "entity_name",
                "fiscal_quarter_end",
                "announcement_date",
                "accession",
            )
            if c in data.columns
        ]
        concept_cols = [c for c in data.columns if c not in leading and c in wanted]
        data = data.select(leading + concept_cols)

    return data


# --- Positioning: 13F institutional holdings ---


def load_institutional_holdings_13f(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load 13F institutional holdings for Chapter 10 / Chapter 22-23 notebooks.

    Produced by `data/equities/positioning/13f_download.py`.
    """
    path = ML4T_DATA_PATH / "equities" / "positioning" / "13f" / "institutional_holdings.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="13F Institutional Holdings",
            path=path,
            download_script="data/equities/positioning/13f_download.py",
            readme="data/equities/positioning/README.md",
        )

    data = pl.read_parquet(path)
    if "filing_date" in data.columns and data["filing_date"].dtype == pl.String:
        data = data.with_columns(pl.col("filing_date").str.to_date(strict=False))

    if start_date and "filing_date" in data.columns:
        data = data.filter(pl.col("filing_date") >= pl.lit(start_date).str.to_date())
    if end_date and "filing_date" in data.columns:
        data = data.filter(pl.col("filing_date") <= pl.lit(end_date).str.to_date())

    return data


def load_13f_stock_features() -> pl.DataFrame:
    """Load stock-level features derived from 13F holdings."""
    path = ML4T_DATA_PATH / "equities" / "positioning" / "13f" / "stock_features.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="13F Stock Features",
            path=path,
            download_script="data/equities/positioning/13f_download.py",
            readme="data/equities/positioning/README.md",
        )

    return pl.read_parquet(path)


def load_13f_edges() -> pl.DataFrame:
    """Load the institution-to-stock edge list for graph construction."""
    path = ML4T_DATA_PATH / "equities" / "positioning" / "13f" / "institution_stock_edges.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="13F Institution-Stock Edges",
            path=path,
            download_script="data/equities/positioning/13f_download.py",
            readme="data/equities/positioning/README.md",
        )

    return pl.read_parquet(path)


def load_13f_bulk_holdings(quarter: str) -> pl.DataFrame:
    """Load one quarter of SEC bulk 13F holdings (full universe, ~3M rows).

    Produced by `data/equities/positioning/13f_download.py --mode bulk`.
    Same canonical schema as `load_institutional_holdings_13f` (cik,
    accession_no, issuer, cusip, value_thousands, shares, filing_date,
    company_name) but covers all ~5-7K filers in the window instead of the
    curated institution list.

    Args:
        quarter: SEC filing-window label, e.g. '2024Q3'. SEC labels by
            filing date: Q1=Mar-May, Q2=Jun-Aug, Q3=Sep-Nov, Q4=Dec-Feb.
    """
    path = (
        ML4T_DATA_PATH
        / "equities"
        / "positioning"
        / "13f"
        / "bulk"
        / quarter
        / "institutional_holdings.parquet"
    )
    if not path.exists():
        raise DataNotFoundError(
            dataset_name=f"13F Bulk Holdings ({quarter})",
            path=path,
            download_script=f"data/equities/positioning/13f_download.py --mode bulk --quarters {quarter}",
            readme="data/equities/positioning/README.md",
        )
    return pl.read_parquet(path)
