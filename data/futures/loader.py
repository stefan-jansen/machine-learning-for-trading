"""CME futures loaders."""

from typing import Literal

import polars as pl

from data.exceptions import DataNotFoundError
from utils import ML4T_DATA_PATH
from utils.data_quality import apply_max_symbols


def load_cme_futures(
    products: list[str] | None = None,
    tenors: list[int] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    frequency: str = "daily",
    continuous: bool = True,
    lazy: bool = False,
    max_symbols: int = 0,
) -> pl.DataFrame | pl.LazyFrame:
    """Load CME futures data.

    Default: session-aligned **daily** bars (used by Ch6-Ch21).
    Use ``frequency="hourly"`` for raw Databento hourly bars (Ch2 only).

    Data pipeline::

        DataBento API → hourly OHLCV → session-aligned daily OHLCV
        (download)      frequency="hourly"   frequency="daily" (default)

    The daily data is generated once in Ch2
    (``05_futures_session_aggregation.py``) and consumed by all later chapters.

    Args:
        products: Product codes (e.g., ``["ES", "NQ", "GC"]``).
                 If None, loads all available products.
        tenors: Contract tenors (0=front month, 1=second, 2=third).
        start_date: Filter start date (YYYY-MM-DD).
        end_date: Filter end date (YYYY-MM-DD).
        frequency: ``"daily"`` (default) for session-aligned bars, or
                  ``"hourly"`` for raw Databento bars.
        continuous: If True (default), load volume-rolled continuous contracts.
                   If False, load individual contract data.
                   Only applies when ``frequency="hourly"``.
        lazy: If True, return LazyFrame for deferred execution.
              Only applies when ``frequency="hourly"``.
        max_symbols: Limit to N random products (0 = all). Seed-deterministic.

    Returns:
        DataFrame with futures prices.

        Daily columns: session_date, product, tenor, adj_open, adj_high,
        adj_low, adj_close, raw_open, raw_high, raw_low, raw_close, cum_ratio,
        volume, bar_count, session_start, session_end. There is no bare
        ``close``: use ``adj_close`` for returns / momentum / volatility /
        labels (roll-continuous) and ``raw_close`` for carry / term structure
        / roll yield / notional / costs (contemporaneous traded levels);
        ``adj_close == raw_close * cum_ratio``.

        Hourly columns: ts_event, product, tenor, open, high, low,
        close, volume. (Hourly bars are per-contract and not roll-adjusted,
        so they keep bare OHLC.)

    Example:
        >>> # Daily data (default) — most notebooks use this
        >>> df = load_cme_futures(products=["ES", "NQ"], tenors=[0])
        >>>
        >>> # Hourly data — Ch2 session aggregation notebook
        >>> df = load_cme_futures(products=["ES"], frequency="hourly")
    """
    if frequency == "daily":
        return _load_cme_futures_daily(products, tenors, start_date, end_date, max_symbols)
    elif frequency == "hourly":
        return _load_cme_futures_hourly(
            products, tenors, start_date, end_date, continuous, lazy, max_symbols
        )
    else:
        msg = f"frequency must be 'daily' or 'hourly', got {frequency!r}"
        raise ValueError(msg)


def list_cme_products(frequency: str = "hourly") -> list[str]:
    """List CME product codes available in the local data store.

    Args:
        frequency: ``"hourly"`` (default) lists products under the Hive-partitioned
            continuous hourly store (``futures/continuous/hourly/product={P}/``);
            ``"individual"`` lists products with raw per-contract files under
            ``futures/individual/{P}/data.parquet``.

    Returns:
        Sorted list of product codes (e.g., ``["6A", "6B", ..., "ZW"]``).

    Raises:
        DataNotFoundError: If the relevant directory is missing.

    Example:
        >>> list_cme_products()[:5]
        ['6A', '6B', '6C', '6E', '6J']
    """
    if frequency == "hourly":
        root = ML4T_DATA_PATH / "futures" / "market" / "continuous" / "hourly"
        if not root.exists():
            raise DataNotFoundError(
                dataset_name="CME Futures Hourly (Continuous Contracts)",
                path=root,
                download_script="data/futures/market/download.py",
                requires_api_key="DATABENTO_API_KEY",
            )
        return sorted(
            p.name.split("=", 1)[1]
            for p in root.iterdir()
            if p.is_dir() and p.name.startswith("product=")
        )
    elif frequency == "individual":
        root = ML4T_DATA_PATH / "futures" / "market" / "individual"
        if not root.exists():
            raise DataNotFoundError(
                dataset_name="CME Futures Individual Contracts",
                path=root,
                download_script="data/futures/market/download.py",
                requires_api_key="DATABENTO_API_KEY",
            )
        return sorted(
            p.name for p in root.iterdir() if p.is_dir() and (p / "data.parquet").exists()
        )
    else:
        msg = f"frequency must be 'hourly' or 'individual', got {frequency!r}"
        raise ValueError(msg)


def _load_cme_futures_daily(
    products: list[str] | None,
    tenors: list[int] | None,
    start_date: str | None,
    end_date: str | None,
    max_symbols: int = 0,
) -> pl.DataFrame:
    """Load session-aligned daily CME futures (internal)."""
    daily_path = (
        ML4T_DATA_PATH / "futures" / "market" / "continuous" / "daily" / "continuous_daily.parquet"
    )

    if not daily_path.exists():
        raise DataNotFoundError(
            dataset_name="CME Futures Daily (Session-Aligned)",
            path=daily_path,
            instructions=(
                "This dataset is generated from hourly futures data.\n"
                "\n"
                "Step 1 — Generate daily bars from hourly data:\n"
                "  python 02_financial_data_universe/code/05_futures_session_aggregation.py\n"
                "\n"
                "Step 2 — If hourly data is also missing, download it first:\n"
                "  python data/futures/download.py --estimate-only\n"
                "  python data/futures/download.py\n"
                "  (requires DATABENTO_API_KEY in .env)"
            ),
            readme="data/futures/README.md",
        )

    lf = pl.scan_parquet(daily_path)

    if products:
        lf = lf.filter(pl.col("product").is_in(products))
    if tenors is not None:
        lf = lf.filter(pl.col("tenor").is_in(tenors))
    if start_date:
        lf = lf.filter(pl.col("session_date") >= pl.lit(start_date).str.to_date())
    if end_date:
        lf = lf.filter(pl.col("session_date") <= pl.lit(end_date).str.to_date())

    df = lf.collect()
    df = apply_max_symbols(df, max_symbols, symbol_col="product")

    return df.sort(["product", "tenor", "session_date"])


def _load_cme_futures_hourly(
    products: list[str] | None,
    tenors: list[int] | None,
    start_date: str | None,
    end_date: str | None,
    continuous: bool,
    lazy: bool,
    max_symbols: int = 0,
) -> pl.DataFrame | pl.LazyFrame:
    """Load hourly CME futures from Hive-partitioned storage (internal)."""
    if continuous:
        hive_path = ML4T_DATA_PATH / "futures" / "market" / "continuous" / "hourly"
        if not hive_path.exists():
            raise DataNotFoundError(
                dataset_name="CME Futures Hourly (Continuous Contracts)",
                path=hive_path,
                download_script="data/futures/market/download.py",
                requires_api_key="DATABENTO_API_KEY",
            )

        scan_opts = {
            "hive_partitioning": True,
            "missing_columns": "insert",
            "extra_columns": "ignore",
        }

        if products:
            product_dirs = [
                hive_path / f"product={p}"
                for p in products
                if (hive_path / f"product={p}").exists()
            ]
            if not product_dirs:
                raise DataNotFoundError(
                    dataset_name=f"CME Futures ({', '.join(products)})",
                    path=hive_path,
                    download_script="data/futures/market/download.py",
                    requires_api_key="DATABENTO_API_KEY",
                )
            lf = pl.concat(
                [pl.scan_parquet(str(d / "year=*/data.parquet"), **scan_opts) for d in product_dirs]
            )
        else:
            lf = pl.scan_parquet(str(hive_path / "product=*/year=*/data.parquet"), **scan_opts)

        schema_names = lf.collect_schema().names()
        if "asset" in schema_names and "symbol" not in schema_names:
            lf = lf.with_columns(pl.col("asset").alias("symbol"))
        # Normalize time column: ensure both ts_event (for internal filtering)
        # and timestamp (canonical schema) exist
        if "timestamp" in schema_names and "ts_event" not in schema_names:
            lf = lf.with_columns(pl.col("timestamp").alias("ts_event"))
        elif "ts_event" in schema_names and "timestamp" not in schema_names:
            lf = lf.with_columns(pl.col("ts_event").alias("timestamp"))

        if tenors is not None:
            lf = lf.filter(pl.col("tenor").is_in(tenors))

        if start_date:
            lf = lf.filter(
                pl.col("ts_event").dt.date() >= pl.lit(start_date).str.to_date("%Y-%m-%d")
            )

        if end_date:
            lf = lf.filter(pl.col("ts_event").dt.date() <= pl.lit(end_date).str.to_date("%Y-%m-%d"))

        if max_symbols > 0:
            lf = apply_max_symbols(lf, max_symbols, symbol_col="product")

        if lazy:
            return lf
        return lf.collect()

    else:
        individual_dir = ML4T_DATA_PATH / "futures" / "market" / "individual"
        if not individual_dir.exists():
            raise DataNotFoundError(
                dataset_name="CME Futures Individual Contracts",
                path=individual_dir,
                download_script="data/futures/market/download.py",
                requires_api_key="DATABENTO_API_KEY",
            )

        if products is None:
            products = [
                p.name
                for p in individual_dir.iterdir()
                if p.is_dir() and (p / "data.parquet").exists()
            ]

        if not products:
            raise DataNotFoundError(
                dataset_name="CME Futures Individual Contracts",
                path=individual_dir,
                download_script="data/futures/market/download.py",
                requires_api_key="DATABENTO_API_KEY",
            )

        dfs = []
        missing = []
        for product in products:
            path = individual_dir / product / "data.parquet"
            if path.exists():
                df = pl.read_parquet(path)
                if "product" not in df.columns:
                    df = df.with_columns(pl.lit(product).alias("product"))
                dfs.append(df)
            else:
                missing.append(product)

        if missing:
            raise DataNotFoundError(
                dataset_name=f"CME Futures Individual Contracts ({', '.join(missing)})",
                path=individual_dir / missing[0] / "data.parquet",
                download_script="data/futures/market/download.py",
                requires_api_key="DATABENTO_API_KEY",
            )

        result = pl.concat(dfs).sort(["timestamp", "product"])

        if start_date:
            start_dt = pl.lit(start_date).str.to_datetime("%Y-%m-%d")
            result = result.filter(pl.col("timestamp") >= start_dt)
        if end_date:
            end_dt = pl.lit(end_date).str.to_datetime("%Y-%m-%d")
            result = result.filter(pl.col("timestamp") <= end_dt)

        return apply_max_symbols(result, max_symbols, symbol_col="product")


def load_cot(
    products: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load CFTC Commitment of Traders (COT) data.

    Reads per-product parquets written by ``data/futures/positioning/cot_download.py``
    from ``$ML4T_DATA_PATH/futures/positioning/cot/{PRODUCT}.parquet`` and concatenates
    them into a single DataFrame using ``diagonal_relaxed`` to reconcile
    schema differences between financial (TFF) and commodity (disaggregated)
    report types.

    Args:
        products: Product codes (e.g., ``["ES", "NQ", "CL"]``). If None,
                 loads all products available under ``futures/positioning/cot/``.
        start_date: Filter ``report_date`` >= this date (YYYY-MM-DD).
        end_date: Filter ``report_date`` <= this date (YYYY-MM-DD).

    Returns:
        DataFrame with columns including ``product``, ``report_type``,
        ``report_date``, ``open_interest``, plus per-trader long/short/net
        columns that depend on the report type. Financial futures rows carry
        ``dealer_*``, ``asset_mgr_*``, ``lev_money_*``; commodity rows carry
        ``commercial_*``, ``managed_money_*``, ``swap_*``.

    Raises:
        DataNotFoundError: If the CoT directory is missing or the requested
            products are not present.

    Example:
        >>> df = load_cot(products=["ES", "NQ"])
    """
    root = ML4T_DATA_PATH / "futures" / "positioning" / "cot"
    if not root.exists():
        raise DataNotFoundError(
            dataset_name="CFTC Commitment of Traders",
            path=root,
            download_script="data/futures/positioning/cot_download.py",
        )

    available = sorted(p.stem for p in root.glob("*.parquet"))
    if not available:
        raise DataNotFoundError(
            dataset_name="CFTC Commitment of Traders",
            path=root,
            download_script="data/futures/positioning/cot_download.py",
        )

    if products is None:
        products = available
    else:
        missing = [p for p in products if p not in available]
        if missing:
            raise DataNotFoundError(
                dataset_name=f"CFTC COT ({', '.join(missing)})",
                path=root / f"{missing[0]}.parquet",
                download_script="data/futures/positioning/cot_download.py",
            )

    dfs = [pl.read_parquet(root / f"{p}.parquet") for p in products]
    df = pl.concat(dfs, how="diagonal_relaxed")

    if start_date:
        df = df.filter(pl.col("report_date") >= pl.lit(start_date).str.to_date())
    if end_date:
        df = df.filter(pl.col("report_date") <= pl.lit(end_date).str.to_date())

    return df.sort(["product", "report_date"])


def list_cot_products() -> list[str]:
    """List CoT product codes available in the local data store.

    Returns:
        Sorted list of product codes with per-product parquets under
        ``$ML4T_DATA_PATH/futures/positioning/cot/``.

    Raises:
        DataNotFoundError: If the CoT directory is missing.
    """
    root = ML4T_DATA_PATH / "futures" / "positioning" / "cot"
    if not root.exists():
        raise DataNotFoundError(
            dataset_name="CFTC Commitment of Traders",
            path=root,
            download_script="data/futures/positioning/cot_download.py",
        )
    return sorted(p.stem for p in root.glob("*.parquet"))
