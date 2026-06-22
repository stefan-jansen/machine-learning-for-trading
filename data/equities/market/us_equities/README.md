# US Equities (NASDAQ Data Link WIKI Prices)

Historical US equity OHLCV for ~3,200 companies from 1962-01-02 to
2018-03-27. Survivorship-bias free — includes delisted issuers. The
dataset is frozen (NASDAQ Data Link stopped updating it in March 2018),
so the downloader writes once and you never re-pull.

## Dataset

- **Source**: NASDAQ Data Link (formerly Quandl) `WIKI/PRICES`
- **Coverage**: 1962-01-02 → 2018-03-27 daily OHLCV, ~3,199 US tickers
- **Size on disk**: ~650 MB parquet
- **Access**: Free with a NASDAQ Data Link API key
  (https://data.nasdaq.com/sign-up)
- **Canonical schema**: `symbol`, `timestamp` (Date), `open`, `high`,
  `low`, `close`, `volume`, `adj_close`, …

## Download

```bash
# One-off: ~several minutes, 650 MB
uv run python data/equities/market/us_equities/download.py

# Preview without hitting the API
uv run python data/equities/market/us_equities/download.py --dry-run

# Re-download even if the parquet is already on disk
uv run python data/equities/market/us_equities/download.py --force
```

API key resolution order (first match wins):

1. `--api-key <key>` CLI flag
2. `QUANDL_API_KEY` environment variable
3. `NASDAQ_DATA_LINK_API_KEY` environment variable
4. Same keys in a `.env` file at the repo root

Output: `$ML4T_DATA_PATH/equities/us_equities/us_equities.parquet`.

## Loading

```python
from data import load_us_equities

df = load_us_equities()                        # full universe
df = load_us_equities(symbols=["AAPL", "MSFT"])
df = load_us_equities(start_date="2000-01-01", end_date="2010-12-31")
df = load_us_equities(max_symbols=50)          # random 50 for prototyping
```

If the parquet is missing, the loader raises `DataNotFoundError` with
the exact download command.

## Consumers

- `case_studies/us_equities_panel/` — the dedicated case study (Ch7 →
  Ch20 pipeline) operates on this universe.
- Chapter 2 EDA notebook `01_us_equities_eda.py` — coverage survey.
- Any chapter that needs a long, survivor-free US equities history.
