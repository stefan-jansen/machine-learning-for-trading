# On-Chain / DeFi Metrics

Public-good on-chain metrics used by Chapter 4 to show how to evaluate
"alternative" data for trading alpha. Two feeds, both key-free:

- **DefiLlama** — aggregate and per-chain Total Value Locked (TVL),
  daily granularity, several-year history.
- **CoinGecko** — daily coin prices + volumes (OHLCV-lite), bounded to
  the free-tier 365-day window.

## License / Cost

- **DefiLlama**: public API, no key, no rate limits advertised. Data
  is released under the Open Database License (`ODbL-1.0`), per
  https://defillama.com/docs/api — attribution "DefiLlama" is required
  when using the data in published work.
- **CoinGecko**: free-tier Public API. Free-tier terms cap history to
  365 days and throughput to ~10-50 req/min. See
  https://www.coingecko.com/en/api/terms for attribution requirements.

Disk footprint: under 5 MB total for the default pull (1 TVL total +
4 chain TVLs + 1 ETH OHLCV file). Runtime: under 1 minute.

## Download

```bash
# Default: total TVL + 4 major chains + ETH prices
uv run python data/crypto/onchain/download.py

# Just DefiLlama (total + chains)
uv run python data/crypto/onchain/download.py --dataset defillama

# Just ETH prices
uv run python data/crypto/onchain/download.py --dataset coingecko

# Custom chain set
uv run python data/crypto/onchain/download.py --chains Ethereum,Solana,Arbitrum

# Different coin (CoinGecko id, lowercase)
uv run python data/crypto/onchain/download.py --dataset coingecko --coins bitcoin
```

Output under `$ML4T_DATA_PATH/crypto/onchain/`:

| File | Source | Contents |
| --- | --- | --- |
| `defillama_tvl_total.parquet` | DefiLlama | Aggregate TVL across all chains |
| `defillama_tvl_<chain>.parquet` | DefiLlama | Per-chain TVL (e.g. `ethereum`, `solana`) |
| `coingecko_<coin>.parquet` | CoinGecko | Daily `price_usd`, `volume_usd` (365-day window) |

Canonical schema for all files: `timestamp` (Date), plus one or two
value columns.

## Loading

```python
from data import load_defillama_chain_tvl, load_coingecko_ohlcv

total_tvl = load_defillama_chain_tvl()                # total across chains
eth_tvl   = load_defillama_chain_tvl("Ethereum")
eth_px    = load_coingecko_ohlcv()                    # ETH, default
btc_px    = load_coingecko_ohlcv("bitcoin")
```

If a requested parquet is missing, each loader raises
`DataNotFoundError` with the exact command (including the right
`--chains` / `--coins` flag) to produce it.

## Rate limits

- **DefiLlama**: no documented cap; we download serially with no
  artificial delay.
- **CoinGecko**: free tier caps at ~10-50 requests/minute. The
  downloader sleeps 2s between coins by default — fine for the handful
  used by Ch4. Raise the limit (paid plan) or supply a demo API key if
  you need more.

## Consumers

- Chapter 4 NB 09 — `09_onchain_fundamentals.py`
- Chapter 4 NB 11 — `11_defi_tvl_evaluation.py` (evaluation case study)
