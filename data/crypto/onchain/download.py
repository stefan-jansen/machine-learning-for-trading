#!/usr/bin/env python3
"""Download on-chain DeFi metrics from DefiLlama and CoinGecko.

Two public, no-key endpoints power the Chapter 4 on-chain / DeFi TVL
notebooks:

  - DefiLlama historicalChainTvl        (total + per-chain TVL)
  - CoinGecko /coins/{id}/market_chart  (daily OHLCV-lite, 365-day
                                         free-tier window)

The downloader normalizes both feeds to single-frequency parquet files
so notebooks can consume them via `load_defillama_chain_tvl()` and
`load_coingecko_ohlcv()` without per-NB HTTP code.

Output layout under `$ML4T_DATA_PATH/crypto/onchain/`:

    defillama_tvl_total.parquet       total DeFi TVL (all chains)
    defillama_tvl_<chain>.parquet     per-chain TVL (Ethereum, Solana, …)
    coingecko_<coin>.parquet          daily prices/volume, one coin

Usage:
    # Default: total + 4 major chains + ETH prices
    python data/crypto/onchain/download.py

    # Only TVL (both total and chains), skip prices
    python data/crypto/onchain/download.py --dataset defillama

    # Just ETH prices, 365-day free-tier window
    python data/crypto/onchain/download.py --dataset coingecko

    # Custom chain set
    python data/crypto/onchain/download.py --chains Ethereum,Solana,Arbitrum

Rate limits: DefiLlama is generous (no documented cap); CoinGecko free
tier is 10-50 req/min, so we sleep between coins.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl
import requests

from utils.downloading import resolve_data_dir

DEFILLAMA_TOTAL_URL = "https://api.llama.fi/v2/historicalChainTvl"
DEFILLAMA_CHAIN_URL = "https://api.llama.fi/v2/historicalChainTvl/{chain}"
COINGECKO_MARKET_URL = "https://api.coingecko.com/api/v3/coins/{coin}/market_chart"

DEFAULT_CHAINS = ["Ethereum", "Solana", "BSC", "Arbitrum"]
DEFAULT_COINS = ["ethereum"]  # CoinGecko ids (lowercase)

REQUEST_TIMEOUT = 60
COINGECKO_SLEEP = 2.0  # seconds between coins — respects free-tier limits


def _fetch_defillama_total() -> pl.DataFrame:
    resp = requests.get(DEFILLAMA_TOTAL_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return (
        pl.DataFrame(
            {
                "timestamp": [d["date"] for d in data],
                "tvl_usd": [d["tvl"] for d in data],
            }
        )
        .with_columns(pl.from_epoch("timestamp", time_unit="s").cast(pl.Date))
        .sort("timestamp")
    )


def _fetch_defillama_chain(chain: str) -> pl.DataFrame:
    resp = requests.get(DEFILLAMA_CHAIN_URL.format(chain=chain), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return (
        pl.DataFrame(
            {
                "timestamp": [d["date"] for d in data],
                "tvl_usd": [d["tvl"] for d in data],
            }
        )
        .with_columns(pl.from_epoch("timestamp", time_unit="s").cast(pl.Date))
        .sort("timestamp")
    )


def _fetch_coingecko_ohlcv(coin: str, days: int) -> pl.DataFrame:
    # CoinGecko returns [[ts_ms, value], ...] arrays for prices and
    # total_volumes. Daily interval only on free tier.
    params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
    resp = requests.get(
        COINGECKO_MARKET_URL.format(coin=coin), params=params, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()
    return (
        pl.DataFrame(
            {
                "timestamp": [p[0] for p in data["prices"]],
                "price_usd": [p[1] for p in data["prices"]],
                "volume_usd": [v[1] for v in data["total_volumes"]],
            }
        )
        .with_columns(pl.from_epoch("timestamp", time_unit="ms").cast(pl.Date))
        .sort("timestamp")
    )


def run_defillama(output_dir: Path, chains: list[str], skip_total: bool) -> None:
    if not skip_total:
        print("  DefiLlama total TVL…")
        total = _fetch_defillama_total()
        path = output_dir / "defillama_tvl_total.parquet"
        total.write_parquet(path)
        print(
            f"    → {path.name}  ({len(total):,} rows, "
            f"{total['timestamp'].min()} → {total['timestamp'].max()})"
        )
    for chain in chains:
        print(f"  DefiLlama {chain} TVL…")
        try:
            df = _fetch_defillama_chain(chain)
        except requests.HTTPError as e:
            print(f"    skip {chain}: {e}")
            continue
        path = output_dir / f"defillama_tvl_{chain.lower()}.parquet"
        df.write_parquet(path)
        print(f"    → {path.name}  ({len(df):,} rows)")


def run_coingecko(output_dir: Path, coins: list[str], days: int) -> None:
    for i, coin in enumerate(coins):
        if i > 0:
            time.sleep(COINGECKO_SLEEP)
        print(f"  CoinGecko {coin} (days={days})…")
        try:
            df = _fetch_coingecko_ohlcv(coin, days)
        except requests.HTTPError as e:
            print(f"    skip {coin}: {e}")
            continue
        path = output_dir / f"coingecko_{coin}.parquet"
        df.write_parquet(path)
        print(
            f"    → {path.name}  ({len(df):,} rows, "
            f"{df['timestamp'].min()} → {df['timestamp'].max()})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download on-chain DeFi metrics")
    parser.add_argument(
        "--dataset",
        choices=["defillama", "coingecko", "both"],
        default="both",
        help="Which feed to pull (default: both)",
    )
    parser.add_argument(
        "--chains",
        type=str,
        default=",".join(DEFAULT_CHAINS),
        help="[defillama] Comma-separated chain names (default: "
        f"{','.join(DEFAULT_CHAINS)}). Use empty string to skip per-chain.",
    )
    parser.add_argument(
        "--skip-total",
        action="store_true",
        help="[defillama] Skip the aggregate (all-chain) TVL series",
    )
    parser.add_argument(
        "--coins",
        type=str,
        default=",".join(DEFAULT_COINS),
        help=f"[coingecko] Comma-separated coin ids (default: {','.join(DEFAULT_COINS)})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="[coingecko] Lookback window in days (free tier max 365)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override output root (default: $ML4T_DATA_PATH)",
    )
    args = parser.parse_args()

    data_path = resolve_data_dir(args.data_path)
    output_dir = data_path / "crypto" / "onchain"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    if args.dataset in ("defillama", "both"):
        chains = [c.strip() for c in args.chains.split(",") if c.strip()]
        run_defillama(output_dir, chains, args.skip_total)

    if args.dataset in ("coingecko", "both"):
        coins = [c.strip().lower() for c in args.coins.split(",") if c.strip()]
        run_coingecko(output_dir, coins, args.days)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
