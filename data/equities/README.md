# Equity Data

Equity datasets organised by **type**: market data (OHLCV, options,
microstructure), fundamentals (SEC filings + XBRL), positioning (13F,
Form 4), and firm characteristics (standalone packaged dataset).

## Directory

```
equities/
├── loader.py                # All equities loaders
├── README.md                # (this file)
│
├── market/                  # Market data (OHLCV, options, microstructure)
│   ├── us_equities/           # NASDAQ Data Link Wiki Prices (1962-2018)
│   ├── sp500/                 # AlgoSeek daily bars + options slices
│   ├── nasdaq100/             # AlgoSeek minute bars
│   ├── microstructure/        # TAQ / MBO / ITCH / IEX tick data
│   ├── sp500.csv              # Index reference file
│   └── us_equities_profile.json
│
├── fundamentals/            # SEC regulatory filings (text + XBRL)
│   ├── 10k/sp100/             # Per-ticker 10-K text (S&P 100)
│   ├── 10q/sp500/mda.parquet  # 10-Q MD&A aggregate (S&P 500)
│   ├── 8k/sp100/              # 8-K event filings (S&P 100)
│   ├── xbrl/                  # XBRL financial facts (CIK × quarter × concept)
│   ├── filings_download.py    # Unified 10-K / 10-Q / 8-K downloader
│   ├── xbrl_download.py
│   └── README.md
│
├── positioning/             # SEC regulatory filings (positions / insider activity)
│   ├── 13f/                   # 13F institutional holdings (per-cik + bulk)
│   ├── form4/                 # Form 4 insider transactions (raw XML)
│   ├── 13f_download.py
│   ├── form4_download.py
│   └── README.md
│
└── firm_characteristics/    # Standalone packaged dataset (Chen-Pelger-Zhu)
    ├── firm_characteristics_all.parquet
    ├── download.py
    └── README.md
```

## Loaders (all in `equities/loader.py`)

### Market

| Loader                             | Dataset                                                    |
| ---------------------------------- | ---------------------------------------------------------- |
| `load_sp500_index()`               | S&P 500 index OHLCV (bundled)                              |
| `load_us_equities()`               | 3,199 US stocks (1962-2018, frozen) — NASDAQ DL            |
| `load_sp500_daily_bars()`          | S&P 500 daily OHLCV — AlgoSeek                             |
| `load_sp500_options()`             | Raw options chains (legacy) — AlgoSeek                     |
| `load_sp500_options_eda()`         | Options EDA slice (8 symbols, 2019-2020) — AlgoSeek slim   |
| `load_sp500_options_straddles_raw()` | ATM-band raw chains, lifecycle-preserving (2017-2021)    |
| `load_sp500_options_surface()`     | Daily IV surface summary                                   |
| `load_sp500_options_straddles()`   | Daily ATM straddles                                        |
| `load_nasdaq100_bars()`            | NASDAQ-100 minute bars (resampling, quotes, microstructure)|
| `load_nasdaq100_taq()`             | TAQ tick data (AAPL, 2020-03-13 / 2020-03-16)              |
| `load_mbo_data()`                  | MBO order book data — Databento                            |
| `load_nasdaq_itch()`               | NASDAQ ITCH messages                                       |
| `load_iex_hist()`                  | IEX DEEP/TOPS data (free)                                  |

### Fundamentals

| Loader                             | Dataset                                                    |
| ---------------------------------- | ---------------------------------------------------------- |
| `load_sp500_10q_mda()`             | S&P 500 10-Q MD&A text (2017-2021)                         |
| `load_sec_filings(form_type, universe)` | 10-K / 10-Q / 8-K aggregate text                      |
| `resolve_sec_filings_dir()`        | Per-ticker filings directory (Ch22 RAG)                    |
| `load_sec_xbrl_fundamentals()`     | XBRL financial facts (CIK × quarter × concept)             |

### Positioning

| Loader                             | Dataset                                                    |
| ---------------------------------- | ---------------------------------------------------------- |
| `load_institutional_holdings_13f()` | 13F holdings (per-cik, 10 curated managers)               |
| `load_13f_bulk_holdings(quarter)`  | 13F full universe (~3M rows per quarter)                   |
| `load_13f_stock_features()`        | Stock-level features (breadth, concentration, value)       |
| `load_13f_edges()`                 | Institution → stock edge list (graph construction)         |

### Firm Characteristics

| Loader                             | Dataset                                                    |
| ---------------------------------- | ---------------------------------------------------------- |
| `load_firm_characteristics()`      | Chen-Pelger-Zhu panel (~180 features; returns + accounting)|

## Quick Start

```bash
# Market
uv run python data/equities/market/us_equities/download.py

# Fundamentals
uv run python data/equities/fundamentals/filings_download.py --form 10-Q --universe sp500
uv run python data/equities/fundamentals/xbrl_download.py

# Positioning
uv run python data/equities/positioning/13f_download.py
uv run python data/equities/positioning/form4_download.py --ticker TSLA --count 20

# Firm characteristics
uv run python data/equities/firm_characteristics/download.py
```

## Book Usage

- **Ch2** — market datasets appear in `02_financial_data_universe/*_eda.py`.
- **Ch3** — microstructure (TAQ / MBO / ITCH / IEX).
- **Ch4** — fundamentals (XBRL, SEC filings), positioning (13F, Form 4).
- **Ch8** — fundamentals + macro cross-sectional features.
- **Ch10** — filing-text signals from 10-Q MD&A; asset embeddings from 13F.
- **Ch11-14** — firm characteristics as the standard ML asset-pricing benchmark.
- **Ch16-20** — NASDAQ-100 + S&P 500 options case studies.
- **Ch22-23** — RAG over SP100 10-Ks; institutional ownership graph.

See the sub-READMEs in each type directory for license, disk footprint,
download commands, and on-disk layout details.
