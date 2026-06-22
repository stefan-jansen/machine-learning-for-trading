"""Crypto loaders: market (OHLCV, premium index) and on-chain (DefiLlama TVL, CoinGecko)."""

from typing import Literal

import polars as pl

from data.exceptions import DataNotFoundError
from utils import ML4T_DATA_PATH
from utils.data_quality import apply_max_symbols


def list_crypto_perps() -> list[str]:
    """List perpetual-futures symbols available in the local data store.

    Returns:
        Sorted list of Binance USDT perps (e.g., ``["AAVEUSDT", ..., "XRPUSDT"]``).

    Raises:
        DataNotFoundError: If ``crypto/perps_1h.parquet`` is missing.

    Example:
        >>> list_crypto_perps()[:3]
        ['AAVEUSDT', 'ADAUSDT', 'APTUSDT']
    """
    path = ML4T_DATA_PATH / "crypto" / "market" / "perps_1h.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="Crypto Perpetuals OHLCV",
            path=path,
            download_script="data/crypto/market/download.py",
            readme="data/crypto/README.md",
        )
    return pl.scan_parquet(path).select("symbol").unique().collect().to_series().sort().to_list()


def load_crypto_premium(
    frequency: Literal["1h", "8h"] = "8h",
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_symbols: int = 0,
) -> pl.DataFrame:
    """Load crypto premium index for funding rate arbitrage case study.

    Args:
        frequency: Data frequency. "8h" aligns with Binance funding settlement times
            (00:00, 08:00, 16:00 UTC). Default is "8h".
        symbols: Optional list of symbols to filter (e.g., ["BTCUSDT", "ETHUSDT"])
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)
        max_symbols: Limit to N random symbols (0 = all). Seed-deterministic.

    Returns:
        DataFrame with columns: timestamp, symbol, premium_index_open/high/low/close
    """
    filename = f"premium_index_{frequency}.parquet"
    path = ML4T_DATA_PATH / "crypto" / "market" / filename
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="Crypto Premium Index",
            path=path,
            download_script="data/crypto/market/download.py --premium",
            readme="data/crypto/README.md",
        )

    df = pl.read_parquet(path)

    # Apply filters
    if symbols:
        df = df.filter(pl.col("symbol").is_in(symbols))
    if start_date:
        df = df.filter(pl.col("timestamp").dt.date() >= pl.lit(start_date).str.to_date())
    if end_date:
        df = df.filter(pl.col("timestamp").dt.date() <= pl.lit(end_date).str.to_date())

    return apply_max_symbols(df, max_symbols)


def load_crypto_perps(
    frequency: Literal["1h", "8h"] = "1h",
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_symbols: int = 0,
) -> pl.DataFrame:
    """Load crypto perpetual futures OHLCV data.

    Args:
        frequency: Data frequency. "1h" for raw hourly data, "8h" for funding-aligned
            8-hour bars (00:00, 08:00, 16:00 UTC - standard funding settlement times).
        symbols: Optional list of symbols to filter (e.g., ["BTCUSDT", "ETHUSDT"])
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)
        max_symbols: Limit to N random symbols (0 = all). Seed-deterministic.

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    # Always load from 1h source
    filename = "perps_1h.parquet"
    path = ML4T_DATA_PATH / "crypto" / "market" / filename
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="Crypto Perpetuals OHLCV",
            path=path,
            download_script="data/crypto/market/download.py",
            readme="data/crypto/README.md",
        )

    lf = pl.scan_parquet(path)
    ts_type = lf.collect_schema()["timestamp"]
    tz = getattr(ts_type, "time_zone", None)

    def _ts_lit(d: str) -> pl.Expr:
        e = pl.lit(d).str.to_datetime()
        return e.dt.replace_time_zone(tz) if tz else e

    # Apply filters before resampling (parquet pushdown via row-group pruning)
    if symbols:
        lf = lf.filter(pl.col("symbol").is_in(symbols))
    if start_date:
        lf = lf.filter(pl.col("timestamp") >= _ts_lit(start_date))
    if end_date:
        # Include the entire end_date for intraday
        lf = lf.filter(pl.col("timestamp") < _ts_lit(end_date) + pl.duration(days=1))

    df = lf.collect()
    # Apply max_symbols before resampling
    df = apply_max_symbols(df, max_symbols)

    if frequency == "8h":
        # Resample to 8H aligned with funding settlement times (00:00, 08:00, 16:00 UTC)
        df = (
            df.sort(["symbol", "timestamp"])
            .group_by_dynamic(
                "timestamp",
                every="8h",
                period="8h",
                by="symbol",
                closed="left",
                label="left",
            )
            .agg(
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
            )
            .sort(["symbol", "timestamp"])
        )

        # Join premium index data (8H aligned, same schedule as funding settlements)
        premium_path = ML4T_DATA_PATH / "crypto" / "market" / "premium_index_8h.parquet"
        if premium_path.exists():
            premium = pl.read_parquet(premium_path)
            if symbols:
                premium = premium.filter(pl.col("symbol").is_in(df["symbol"].unique()))
            df = df.join(premium, on=["symbol", "timestamp"], how="left")

    return df


# --- On-chain / DeFi metrics ---


def load_defillama_chain_tvl(chain: str = "total") -> pl.DataFrame:
    """Load historical Total Value Locked (TVL) from DefiLlama.

    Produced by `data/crypto/onchain/download.py`.

    Args:
        chain: "total" for aggregate DeFi TVL across all chains (default),
            or a chain name like "Ethereum", "Solana", "BSC", "Arbitrum".
            Matches the filename suffix (lowercased).

    Returns:
        DataFrame with `timestamp` (Date) and `tvl_usd` (float) columns.
    """
    suffix = chain.lower()
    path = ML4T_DATA_PATH / "crypto" / "onchain" / f"defillama_tvl_{suffix}.parquet"
    if not path.exists():
        chains_flag = "" if suffix == "total" else f" --chains {chain}"
        raise DataNotFoundError(
            dataset_name=f"DefiLlama TVL ({chain})",
            path=path,
            download_script=f"data/crypto/onchain/download.py --dataset defillama{chains_flag}",
            readme="data/crypto/onchain/README.md",
        )
    return pl.read_parquet(path)


def load_coingecko_ohlcv(coin: str = "ethereum") -> pl.DataFrame:
    """Load daily prices/volume for one coin from CoinGecko.

    Produced by `data/crypto/onchain/download.py --dataset coingecko`.
    Free-tier window is 365 days; re-run the downloader to refresh.

    Args:
        coin: CoinGecko coin id (lowercase). Defaults to "ethereum".

    Returns:
        DataFrame with `timestamp` (Date), `price_usd`, `volume_usd`.
    """
    path = ML4T_DATA_PATH / "crypto" / "onchain" / f"coingecko_{coin.lower()}.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name=f"CoinGecko OHLCV ({coin})",
            path=path,
            download_script=f"data/crypto/onchain/download.py --dataset coingecko --coins {coin}",
            readme="data/crypto/onchain/README.md",
        )
    return pl.read_parquet(path)
