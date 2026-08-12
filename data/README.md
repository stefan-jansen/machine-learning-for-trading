# ML4T Data Infrastructure

Central data management for *Machine Learning for Trading, 3rd Edition*.

Each dataset has its own directory with a download script, loader, config, and exploration notebook. All loaders return Polars DataFrames with a consistent API.

---

## Quick Start

```bash
# 1. Set data path in repository root .env file
ML4T_DATA_PATH=/path/to/your/data

# 2. Download free datasets (no API keys needed)
uv run python data/download_all.py --free-only

# 3. Use in notebooks
from data import load_etfs
df = load_etfs()
```

---

## Dataset Catalog

Organized by asset class and data type. "Type" column maps each dataset
to its place in the Ch2/Ch4 taxonomy: market (OHLCV / microstructure /
options), fundamentals (accounting + regulatory filings), positioning
(positions / insider activity), onchain (crypto-native fundamentals), or
cross-asset (factors, macro, prediction markets, news, text).

| Dataset                  | Asset Class  | Type          | Frequency    | Symbols | Coverage  | Source           | Access |
| ------------------------ | ------------ | ------------- | ------------ | ------- | --------- | ---------------- | ------ |
| ETF Universe             | Equity       | Market        | Daily        | 100     | 2006-2025 | Yahoo Finance    | No     |
| US Equities              | Equity       | Market        | Daily        | 3,199   | 1962-2018 | NASDAQ DL        | Free   |
| S&P 500 Bars             | Equity       | Market        | Daily        | 638     | 2017-2021 | AlgoSeek         | No     |
| S&P 500 Options          | Equity       | Market        | Daily        | 634     | 2017-2021 | AlgoSeek         | Convert|
| NASDAQ-100 Bars          | Equity       | Market        | Minute       | ~100    | 2020-2021 | AlgoSeek         | Convert|
| TAQ Tick                 | Equity       | Market        | Tick         | 1       | Mar 2020  | AlgoSeek         | Unzip  |
| MBO Tick                 | Equity       | Market        | Tick         | 1       | Nov 2024  | Databento        | Manual |
| NASDAQ ITCH              | Equity       | Market        | Tick         | all     | varies    | NASDAQ FTP       | No     |
| IEX DEEP/TOPS            | Equity       | Market        | Tick         | all     | varies    | IEX public       | No     |
| SEC XBRL Fundamentals    | Equity       | Fundamentals  | Quarterly    | 20      | 2022-2024 | SEC EDGAR        | No     |
| SEC 10-K (SP100)         | Equity       | Fundamentals  | Annual       | ~100    | 2020-2025 | SEC EDGAR        | No     |
| SEC 10-Q MD&A (SP500)    | Equity       | Fundamentals  | Quarterly    | ~600    | 2017-2021 | SEC EDGAR        | No     |
| SEC 8-K (SP100)          | Equity       | Fundamentals  | Event        | ~100    | 2024-2025 | SEC EDGAR        | No     |
| 13F Institutional        | Equity       | Positioning   | Quarterly    | 10 inst | rolling   | SEC EDGAR        | No     |
| Form 4 Insider           | Equity       | Positioning   | Event        | varies  | varies    | SEC EDGAR        | No     |
| Firm Characteristics     | Equity       | Packaged      | Monthly      | anon    | 1967-2016 | GitHub           | No     |
| CME Futures              | Futures      | Market        | Daily/Hourly | 30      | 2011-2025 | Databento        | Paid   |
| CFTC Commitment of Traders | Futures    | Positioning   | Weekly       | 25+     | 2020-2025 | CFTC public      | No     |
| Crypto Perps             | Crypto       | Market        | 1h           | 19      | 2020-2025 | Binance Public   | No     |
| Crypto Premium           | Crypto       | Market        | 8h           | 19      | 2020-2025 | Binance Public   | No     |
| DefiLlama TVL            | Crypto       | Onchain       | Daily        | chains  | varies    | DefiLlama        | No     |
| CoinGecko OHLCV          | Crypto       | Onchain       | Daily        | varies  | 365 days  | CoinGecko        | No     |
| FX Pairs                 | Currency     | Market        | 4h/Daily     | 20      | 2011-2025 | OANDA            | Free   |
| FF Factors               | Cross-asset  | Factors       | Monthly      | 5       | 1926-now  | Ken French       | No     |
| AQR Factors              | Cross-asset  | Factors       | Monthly      | 8       | varies    | AQR              | No     |
| FRED Macro               | Cross-asset  | Macro         | Various      | 40      | 2000-2025 | FRED             | Free   |
| Kalshi events            | Cross-asset  | Prediction    | Daily        | varies  | 2021-2025 | Kalshi public    | No     |
| Polymarket events        | Cross-asset  | Prediction    | Daily        | varies  | 2020-2025 | Polymarket public| No     |
| FNSPID news              | Cross-asset  | News          | Daily        | 4,775   | 1999-2023 | HuggingFace      | No     |
| Bloomberg news archive   | Cross-asset  | News          | Daily        | mixed   | 2006-2013 | HuggingFace      | No     |
| Financial Phrasebank     | Cross-asset  | Text          | Static       | —       | n/a       | HuggingFace      | No     |

**Access legend.** `No` — included with the repo or fetched by an
unauthenticated script. `Free` — script-download, free API key required.
`Paid` — script-download, billed API (see per-dataset estimates).
`Manual` — reader downloads from a hosted URL or provider portal and
places the files under `$ML4T_DATA_PATH` (no script); the DataBento MBO
one-off has step-by-step instructions below. `Convert` — AlgoSeek hosts
the archive openly, and one script turns it into what the loaders read.
`Unzip` — AlgoSeek hosts it openly as parquet already in the loader's layout,
so unpacking it is the whole of the work. Both: see
[AlgoSeek datasets](#algoseek-datasets).

---

## Download Commands

> **Times below are rough, indicative estimates only.** Actual duration
> depends on your bandwidth, the providers' current rate limits, and disk
> speed — treat them as ballpark, not guarantees.

### Free Datasets (No API Keys)

```bash
# All free datasets at once: about 4.1 GB and 12 minutes, of which the
# firm-characteristics dataset is 4.0 GB. Without it the whole set is ~75 MB:
uv run python data/download_all.py --free-only
uv run python data/download_all.py --free-only --skip-firm-characteristics

# Individual datasets (from repo root)
uv run python data/etfs/market/download.py                           # ~30s
uv run python data/crypto/market/download.py                         # ~10-15 min (see note)
uv run python data/factors/ff_download.py                     # ~5s
uv run python data/factors/aqr_download.py                    # ~5s
uv run python data/equities/firm_characteristics/download.py  # ~4.0 GB on disk, largest free dataset by far; downloads + converts (minutes, bandwidth-dependent)
uv run python data/futures/positioning/cot_download.py                    # ~2-3 min (CFTC CoT)
```

**Note on crypto download time**: The Binance public API returns max 1,500 rows per request with ~1s server response time. Downloading 5 years of hourly data for 19 symbols requires ~700 API calls. Downloads run in parallel (5 concurrent), but the total still takes 10-15 minutes. This is a Binance server-side rate limit, not a bug.

If a symbol is missing from the result, the script names it and exits non-zero. Re-running fetches every symbol again and merges into what is already on disk, so one that failed can arrive on the second run; the status is then computed from the merged data, not from the second run's failures. Pass `--allow-partial` to keep what arrived and exit 0.

### Free API Key Required

```bash
# FRED macro indicators
uv run python data/macro/download.py

# US Equities (NASDAQ Data Link — frozen, ends 2018)
uv run python data/equities/market/us_equities/download.py

# FX pairs (OANDA)
uv run python data/fx/market/download.py              # 4-hourly (default)
uv run python data/fx/market/download.py --daily      # Daily
```

### Paid API Key (Databento)

```bash
# CME Futures — ALWAYS estimate cost first!
uv run python data/futures/market/download.py --estimate-only
uv run python data/futures/market/download.py
```

### Manual Download (Databento Download Center)

The Chapter 3 MBO slice (NVDA, 10 trading days in November 2024) is best
obtained as a one-off download from the Databento Download Center —
total cost is under $10 and the files stay available for 30 days.

See `data/equities/market/microstructure/MBO_DOWNLOAD.md` for step-by-step
instructions. An API-based alternative (`mbo_download.py`) is available
for users who already have a `DATABENTO_API_KEY`.

### AlgoSeek datasets

The book uses four AlgoSeek datasets. **Three are published** at
<https://algoseek.com/ml-for-trading/> — plain download links, no account, no
API key, no license request. **The fourth, the S&P 500 daily bars, ships inside
this repository**, so there is nothing to fetch for it and nothing to configure.

| Dataset | Archive | Size | What it is |
| --- | --- | --- | --- |
| NASDAQ-100 minute bars | `nasdaq-100-constituents-taq-ext.zip` | 5.9 GB | Extended-hours minute bars, up to 90 fields, 505 days 2020-01-02 to 2021-12-31 |
| S&P 500 options | `options_daily_greeks_sp500.zip` | 14.1 GB | Daily option chains with Greeks, 1,259 days 2017-2021, 634 symbols |
| NASDAQ-100 TAQ ticks | `symbol=AAPL.zip` | 67 MB | Trade and NBBO quote events, AAPL on 2020-03-13 and 2020-03-16, 21,284,141 rows |
| S&P 500 daily bars | *(in this repo)* | 8.6 MB | Daily OHLCV with split factors, 638 symbols, 2017-01-03 to 2021-12-31, 635,703 rows |

The two large archives are CSV and every loader reads parquet, so one conversion
step sits between the download and the notebooks:

```bash
# NASDAQ-100 minute bars -> equities/market/nasdaq100/minute_bars/
uv run python data/equities/market/algoseek_convert.py \
    --dataset nasdaq100-minute-bars --source ~/Downloads/nasdaq-100-constituents-taq-ext.zip

# S&P 500 raw option chains -> equities/market/sp500/options/
uv run python data/equities/market/algoseek_convert.py \
    --dataset sp500-options --source ~/Downloads/options_daily_greeks_sp500.zip

# then build what the options notebooks actually load
uv run python data/equities/market/sp500/build_options_eda.py
uv run python data/equities/market/sp500/build_options_straddles_raw.py
uv run python data/equities/market/sp500/materialize_options.py
```

`--source` also takes a directory you have already extracted, which is
considerably faster for the options archive: it holds 1,275,314 gzipped files,
and reading them out of the zip pays the archive's index on every open. Both
conversions resume, so an interrupted run continues where it stopped.

The TAQ ticks are already parquet in the layout the loader scans, so unpacking
them is the whole of the work. Name the members — Dropbox writes a stray root
entry into that archive, and unzipping without `"*.parquet"` warns and exits 2:

```bash
unzip -q "symbol=AAPL.zip" "*.parquet" \
    -d "$ML4T_DATA_PATH/equities/market/microstructure/trade_and_quotes/symbol=AAPL"
```

The minute-bar archive cannot stand in for the ticks. Despite the `taq-ext` in
its name it is quote-aware bar aggregates, and reconstructing an order book needs
the individual events.

The S&P 500 daily bars need no step at all. `ML4T_DATA_PATH` defaults to this
`data/` directory, and the loader falls back to the repository copy even when you
have pointed `ML4T_DATA_PATH` somewhere else:

```python
from data import load_sp500_daily_bars
bars = load_sp500_daily_bars(symbols=["AAPL"], start_date="2020-01-01")
```

#### Attribution

The S&P 500 daily bars in `data/equities/market/sp500/daily_bars.parquet` are
© AlgoSeek LLC, redistributed here with AlgoSeek's permission for readers of
*Machine Learning for Trading*. AlgoSeek retains all rights to the data. Cite
AlgoSeek (<https://algoseek.com>) as the source in anything you publish from it.
The same applies to the three datasets you download from the page above.

### Update Existing Data

Extend datasets beyond the default end date:

```bash
uv run python data/download_all.py --update
```

---

## Using Loaders

All loaders are importable from `data` and return Polars DataFrames:

```python
from data import (
    load_etfs,
    load_crypto_perps,
    load_crypto_premium,
    load_cme_futures,
    load_cot,
    load_fx_pairs,
    load_macro,
    load_us_equities,
    load_ff_factors,
    load_aqr_factors,
    load_firm_characteristics,
    load_nasdaq100_bars,
    load_sp500_daily_bars,
    load_sp500_options,
    load_sp500_options_eda,
    load_sp500_options_straddles_raw,
    load_sp500_options_surface,
    load_sp500_options_straddles,
    load_nasdaq100_taq,
    load_mbo_data,
    load_nasdaq_itch,
    load_iex_hist,
)

# All loaders support filtering
df = load_etfs(symbols=["SPY", "QQQ"], start_date="2020-01-01")

# Futures use 'products' instead of 'symbols'
futures = load_cme_futures(products=["ES", "NQ"], start_date="2020-01-01")

# Test mode: limit to N random symbols (seed-deterministic)
df = load_etfs(max_symbols=15)
```

When data is missing, loaders raise `DataNotFoundError` with download instructions.

---

## API Keys

### Free API Keys

| Provider         | Variable         | Sign Up                                           |
| ---------------- | ---------------- | ------------------------------------------------- |
| FRED             | `FRED_API_KEY`   | https://fred.stlouisfed.org/docs/api/api_key.html |
| NASDAQ Data Link | `QUANDL_API_KEY` | https://data.nasdaq.com/sign-up                   |
| OANDA            | `OANDA_API_KEY`  | https://www.oanda.com/                            |

### Paid API Keys

| Provider  | Variable            | Cost             |
| --------- | ------------------- | ---------------- |
| Databento | `DATABENTO_API_KEY` | $125 free credit |

### Configuration

Create `.env` in repository root:

```bash
ML4T_DATA_PATH=/path/to/your/data

# Free API keys
FRED_API_KEY=your-fred-key
QUANDL_API_KEY=your-nasdaq-key
OANDA_API_KEY=your-oanda-key

# Paid
DATABENTO_API_KEY=db-your-key
```

---

## Directory Structure

Every dataset directory is self-contained: a download script, a loader
(or re-export from a parent loader), a README with the full instructions
that `DataNotFoundError` points readers to, and optionally a config
and exploration notebook.

Data is organized by **asset class × data type**, matching the Ch2 /
Ch4 taxonomy. Each asset class has `market/` for OHLCV-style data, and
optionally `fundamentals/`, `positioning/`, or other type-specific
subdirectories. Cross-asset datasets (`factors/`, `macro/`,
`prediction_markets/`, `alternative/`) sit at the top level.

```
data/
├── __init__.py              # Single import point for all loaders
├── exceptions.py            # DataNotFoundError, DownloadError, MissingDependencyError
├── download_all.py          # Download orchestrator
├── README.md                # (this file)
│
├── equities/                # US equities
│   ├── market/              # us_equities, sp500 (daily + options), nasdaq100, microstructure
│   ├── fundamentals/        # 10-K / 10-Q / 8-K filings, XBRL financials
│   ├── positioning/         # 13F institutional holdings, Form 4 insider
│   ├── firm_characteristics/  # Chen-Pelger-Zhu panel (standalone packaged dataset)
│   └── loader.py            # All equities loaders in one module
│
├── futures/                 # CME futures
│   ├── market/              # Databento continuous + individual contracts
│   ├── positioning/         # CFTC Commitment of Traders (CoT)
│   └── loader.py
│
├── crypto/                  # Crypto
│   ├── market/              # Binance perps OHLCV + premium index
│   ├── onchain/             # DefiLlama TVL + CoinGecko OHLCV
│   └── loader.py
│
├── fx/market/               # FX pairs (OANDA)
├── etfs/market/             # ETF universe (Yahoo)
│
├── factors/                 # Fama-French, AQR (cross-asset, academic)
├── macro/                   # FRED macro indicators (cross-asset)
├── prediction_markets/      # Kalshi + Polymarket events
│
└── alternative/             # Cross-asset third-party alt data
    ├── news/                # Bloomberg, FNSPID
    └── text/                # Financial Phrasebook sentiment benchmark
```

Every subdirectory owns its data's lifecycle — a reader can open any
leaf README and find the download command and file layout without
consulting the top-level doc.

### Equities Loaders (all in `equities/loader.py`)

**Market (OHLCV, microstructure, options):**

| Loader | Dataset | Source |
| ------ | ------- | ------ |
| `load_sp500_index()` | S&P 500 index OHLCV (1980-2025) | Bundled |
| `load_us_equities()` | 3,199 US stocks (1962-2018) | NASDAQ DL |
| `load_sp500_daily_bars()` | S&P 500 daily OHLCV (638 symbols, 2017-2021) | AlgoSeek, bundled |
| `load_sp500_options()` | Raw options chains (legacy) | AlgoSeek |
| `load_sp500_options_eda()` | Options EDA slice (8 symbols, 2019-2020) | AlgoSeek (slim) |
| `load_sp500_options_straddles_raw()` | ATM-band raw chains, lifecycle-preserving (2017-2021) | AlgoSeek (slim) |
| `load_sp500_options_surface()` | Daily IV surface summary | Materialized |
| `load_sp500_options_straddles()` | Daily ATM straddles | Materialized |
| `load_nasdaq100_bars()` | NASDAQ-100 bars (minute default; resampling, quotes, full microstructure) | AlgoSeek |
| `load_nasdaq100_taq()` | TAQ tick data (AAPL, 2020-03-13 / 2020-03-16) | AlgoSeek |
| `load_mbo_data()` | MBO order book data | Databento |
| `load_nasdaq_itch()` | NASDAQ ITCH messages | NASDAQ FTP |
| `load_iex_hist()` | IEX DEEP/TOPS data | IEX (free) |

**Fundamentals (SEC filings + XBRL):**

| Loader | Dataset | Source |
| ------ | ------- | ------ |
| `load_sp500_10q_mda()` | S&P 500 10-Q MD&A text (2017-2021) | SEC EDGAR |
| `load_sec_filings(form_type)` | 10-K / 10-Q / 8-K aggregate text | SEC EDGAR |
| `resolve_sec_filings_dir()` | Per-ticker filings directory (Ch22 RAG) | SEC EDGAR |
| `load_sec_xbrl_fundamentals()` | XBRL financial facts (CIK × quarter × concept) | SEC XBRL Frames |

**Positioning (13F):**

| Loader | Dataset | Source |
| ------ | ------- | ------ |
| `load_institutional_holdings_13f()` | 13F holdings (per-cik, 10 curated managers) | SEC EDGAR |
| `load_13f_bulk_holdings(quarter)` | 13F full universe (~3M rows per quarter) | SEC bulk |
| `load_13f_stock_features()` | Stock-level features (breadth, concentration) | Derived |
| `load_13f_edges()` | Institution → stock edge list (graph) | Derived |

**Firm characteristics (packaged dataset):**

| Loader | Dataset | Source |
| ------ | ------- | ------ |
| `load_firm_characteristics()` | Chen-Pelger-Zhu panel (~180 features, returns + accounting) | GitHub |

---

## Storage Requirements

Each row is a complete profile, not an increment on the row above. The first three were measured
on a clean install; the additions below them are the individual dataset sizes to add on top.

| Profile | Contents | Size | How you get it |
|---|---|---|---|
| Minimum | ETFs, Crypto, Factors (Fama-French + AQR) | ~70 MB | the individual download scripts above |
| Free, without firm characteristics | Minimum + CFTC CoT, prediction markets | ~75 MB | `--free-only --skip-firm-characteristics` |
| **Free** | **the row above + firm characteristics** | **~4.1 GB** | **`--free-only`** |

`--free-only` covers seven datasets and does **not** include Macro or FX; fetch those with
their own scripts when a chapter needs them.

| Add on top of any profile | Size |
|---|---|
| US Equities | ~670 MB |
| CME Futures | ~85 MB |
| AlgoSeek NASDAQ-100 minute bars | 5.9 GB archive, 3.5 GB converted |
| AlgoSeek TAQ ticks | 67 MB, no conversion |
| AlgoSeek S&P 500 options | 14.1 GB archive; the raw conversion is the peak, and the slices the notebooks read are 1.7 GB |
| ITCH, MBO | ~1 GB plus 4-6 GB per ITCH date |

The S&P 500 daily bars are not in this table: they ship with the repository.
Delete each AlgoSeek archive once its conversion has finished — nothing reads it
again, and the converter resumes from what it has already written.

The **Free** row is where most readers land, because `--free-only` is what the README tells you
to run. It is 4.1 GB rather than 75 MB entirely because of firm characteristics, which is 4.0 GB
on its own, of which 3.5 GB is the `dl_asset_pricing` source archive.

These are data sizes only. The `uv` environment is a further ~11 GB on Linux and ~3 GB on macOS,
and the cloned git history is ~0.9 GB.

---

## Canonical Schema

All loaders return data with consistent column names:

- **Entity column**: `symbol` (exception: CME futures use `product`)
- **Time column**: `timestamp` (for all frequencies — daily, hourly, minute, tick)

Notebooks should always use these canonical names. If older data files use legacy names like `asset`, `date`, `ticker`, or `pair`, the loaders normalize them automatically.
