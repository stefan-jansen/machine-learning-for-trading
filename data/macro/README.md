# Macroeconomic Indicators (FRED)

~40 macroeconomic and financial-condition time series from the Federal
Reserve Economic Data (FRED) service. Used across the book for regime
detection, macro factor engineering, and control variables in causal
analyses. All series are stored in a single wide-format parquet.

## Dataset

- **Source**: FRED (Federal Reserve Bank of St. Louis) via the public
  FRED REST API.
- **Coverage**: 2000-01-01 → present; mixed frequency (daily, weekly,
  monthly — daily-aligned in the loader output).
- **Series**: ~40 (treasury yields, credit spreads, volatility indices,
  real-economy indicators, financial-conditions indices).
- **Size on disk**: under 1 MB (primary parquet + metadata + dictionary).
- **Runtime**: under 2 minutes for a full refresh (FRED API is fast).
- **API key**: `FRED_API_KEY` required — free at https://fred.stlouisfed.org/docs/api/api_key.html.
- **License / attribution**: FRED data is public domain; the service
  requests attribution when series are redisplayed. See
  https://fred.stlouisfed.org/legal/.

## Key Series

| Group                | Series                                                  |
| -------------------- | ------------------------------------------------------- |
| Treasury yields      | DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS10, DGS30  |
| Spreads              | T10Y2Y, T10Y3M, BAMLH0A0HYM2, TEDRATE                   |
| Volatility           | VIXCLS, MOVE                                            |
| Economic             | UNRATE, INDPRO, CPIAUCSL, PCEPI, GDP                    |
| Financial conditions | NFCI, STLFSI4                                           |

Full list + descriptions: `config.yaml` and `fred_macro_metadata.parquet`
(load via `load_macro_metadata()`).

## Download

```bash
uv run python data/macro/download.py           # full refresh
uv run python data/macro/download.py --dry-run # plan only
```

Output layout under `$ML4T_DATA_PATH/macro/`:

```
fred_macro.parquet                # primary wide-format series (loader target)
fred_macro_metadata.parquet       # per-series metadata (name, group, description, source id)
fred_macro_dictionary.parquet     # code-book for series names
fred_macro_profile.json           # summary statistics
fred_macro_raw.parquet            # raw unprocessed pull (reserve)
fred_macro_raw_dictionary.parquet
fred_macro_raw_profile.json
```

## Loading

```python
from data import load_macro, load_macro_metadata

df = load_macro()                              # all series, wide format
df = load_macro(series=["DGS10", "DGS2", "VIXCLS"])
df = load_macro(start_date="2015-01-01", end_date="2023-12-31")

meta = load_macro_metadata()                   # series metadata table
```

Schema (loader output, canonical):

| Column       | Type     | Description                     |
| ------------ | -------- | ------------------------------- |
| `timestamp`  | Date     | Observation date                |
| `{SERIES}`   | Float    | Series value (one col per code) |

The on-disk parquet uses `date`; `load_macro()` renames it to `timestamp`
to match the book's canonical schema.

## Consumers

- **Ch4**: `06_fred_macro_eda.py`, `07_macro_data_alignment.py`.
- **Ch7**: `08_causal_sanity_checks.py`.
- **Ch8**: `04_fundamentals_macro_calendar.py`.
- **Ch16**: `01_backtest_first_principles.py`, `06_framework_parity.py`.
- **Ch18**: `02_spread_estimation.py`, `03_market_impact_calibration.py`.
