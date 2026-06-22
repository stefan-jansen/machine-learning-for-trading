# Equity Fundamentals: SEC filings + XBRL

10-K / 10-Q / 8-K filing text and XBRL financial facts from SEC EDGAR.
All data is public domain (17 C.F.R. § 203.30-1); SEC asks that
downloaders respect the **10 req/sec** rate limit via a descriptive
`User-Agent` — downloaders here handle this automatically.

13F institutional holdings and Form 4 insider transactions (also SEC
regulatory filings) live under `../positioning/` — they capture
positioning rather than firm fundamentals, so they have their own home.

## License / Cost

- **License**: public domain; redistribution permitted. Cite SEC EDGAR
  as the source (https://www.sec.gov/os/accessing-edgar-data).
- **Disk**: ~240 MB for a full pull (SP500 10-Q MD&A ~190 MB; SP100 10-K
  full text ~40 MB; 8-K ~1-5 MB per ticker; XBRL fundamentals ~5 MB).
- **Runtime**: dominated by SEC's 10 req/sec cap. SP500 10-Q MD&A for
  2017-2021 runs ~4-5 hours.

## Datasets

| Form | Universe | Script | Loader(s) |
| --- | --- | --- | --- |
| 10-K (annual) | S&P 100 | `filings_download.py --form 10-K --universe sp100` | `load_sec_filings(form_type="10-K")`, `resolve_sec_filings_dir` |
| 10-Q (quarterly MD&A) | S&P 500 | `filings_download.py --form 10-Q --universe sp500` | `load_sp500_10q_mda`, `load_sec_filings(form_type="10-Q", universe="sp500")` |
| 8-K (event) | S&P 100 | `filings_download.py --form 8-K --universe sp100` | `load_sec_filings(form_type="8-K")` |
| XBRL fundamentals | 20 large-cap US equities (default) | `xbrl_download.py` | `load_sec_xbrl_fundamentals` |

## Download Commands

```bash
# S&P 500 10-Q MD&A (~4-5 hours, rate-limited; writes 190 MB parquet)
uv run python data/equities/fundamentals/filings_download.py --form 10-Q --universe sp500 --years 2017-2021

# S&P 100 10-K full text (~1 hour for 2020-2025; writes ~40 MB)
uv run python data/equities/fundamentals/filings_download.py --form 10-K --universe sp100 --years 2020-2025

# S&P 100 8-K (event filings, lighter)
uv run python data/equities/fundamentals/filings_download.py --form 8-K --universe sp100 --years 2024-2025

# Smoke tests
uv run python data/equities/fundamentals/filings_download.py --form 10-Q --universe sp500 --sample 20
uv run python data/equities/fundamentals/filings_download.py --form 10-K --universe sp100 --sample 5

# XBRL fundamentals (20 large-cap CIKs × 2022-2024, ~2-3 min)
uv run python data/equities/fundamentals/xbrl_download.py
uv run python data/equities/fundamentals/xbrl_download.py --years 2020,2021,2022,2023,2024 \
    --concepts Assets,Revenues,NetIncomeLoss
uv run python data/equities/fundamentals/xbrl_download.py --ciks 320193
```

`filings_download.py` supports `--resume` for interrupted runs.

## Directory Layout

Form-first: every dataset lives under `fundamentals/{form}/`.

```
$ML4T_DATA_PATH/equities/fundamentals/
├── 10k/
│   └── sp100/
│       ├── {TICKER}/{YEAR}.parquet          # Per-filing text
│       └── reference/
│           └── all_10k_filings.parquet      # Aggregate for fast bulk loading
├── 10q/
│   └── sp500/
│       └── mda.parquet                      # 10-Q MD&A aggregate (all tickers)
├── 8k/
│   └── sp100/
│       ├── {TICKER}/{YEAR}.parquet
│       └── reference/
│           └── all_8k_filings.parquet
└── xbrl/
    ├── fundamentals.parquet                 # CIK × quarter × concept panel
    └── filing_dates/CIK{cik}.parquet        # Per-CIK accession → filing_date cache
```

## Loading

```python
from data import (
    load_sec_filings,
    load_sp500_10q_mda,
    load_sec_xbrl_fundamentals,
)

# Aggregate 10-K full text for S&P 100
tenk = load_sec_filings(form_type="10-K", universe="sp100",
                        symbols=["AAPL", "MSFT"], start_date="2022-01-01")

# 10-Q MD&A for S&P 500
mda = load_sp500_10q_mda(symbols=["AAPL"], start_date="2020-01-01")

# XBRL fundamentals panel (use announcement_date for PIT backtesting)
fundamentals = load_sec_xbrl_fundamentals(
    symbols=["AAPL", "MSFT"], years=[2023, 2024],
    concepts=["Assets", "Revenues"],
)
```

If a required artifact is missing, loaders raise `DataNotFoundError` with
the exact download command needed to produce it.

## Consumers

- **Ch4 NB 02** — SEC filings explorer (10-K text)
- **Ch4 NB 04** — XBRL fundamentals + bitemporal (PIT) query patterns
- **Ch8 NB 04** — Fundamentals + macro cross-sectional features
- **Ch10 NB 09** — Filing-text signals from 10-Q MD&A
- **Ch22 NB 01-05, 08** — RAG pipeline over SP100 10-Ks
- **Ch23** — 8-K extraction, supply chain KG, related RAG benchmarks
