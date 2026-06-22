# FX Pairs (OANDA)

20 major FX pairs sampled daily and at 4-hour bars. Used for currency
momentum, carry analyses, and cost-model calibration (spread / impact
across majors vs crosses).

## Dataset

- **Source**: OANDA REST v20 API (production endpoint).
- **Coverage**: 2011-01-01 → present (daily back-fill); 4-hour bars from
  2015 onward.
- **Pairs**: 20 (4 majors, 3 commodity-linked, 13 crosses).
- **Size on disk**: ~22 MB total (~1.5 MB daily parquet, ~8.6 MB 4h
  parquet, plus per-symbol hive partitions).
- **Runtime**: ~3-5 minutes for a daily refresh; ~10-15 minutes for a
  4-hour refresh (OANDA rate-limits at 120 req/min).
- **API key**: `OANDA_API_KEY` required — free practice account at
  https://developer.oanda.com/. Production credentials also work.
- **License / attribution**: Data is provided for personal and
  educational use under the OANDA API terms
  (https://www.oanda.com/legal/api-terms-of-service). Redistribution of
  the raw time series is not permitted; derived analytics (returns,
  features, model outputs) are fine.

## Pairs

| Group     | Pairs                                                                                                |
| --------- | ---------------------------------------------------------------------------------------------------- |
| Majors    | EUR_USD, GBP_USD, USD_JPY, USD_CHF                                                                   |
| Commodity | AUD_USD, USD_CAD, NZD_USD                                                                            |
| Crosses   | EUR_GBP, EUR_JPY, EUR_CHF, EUR_CAD, EUR_AUD, GBP_JPY, GBP_CHF, GBP_AUD, AUD_JPY, CHF_JPY, CAD_JPY, NZD_JPY, AUD_NZD |

## Download

```bash
uv run python data/fx/market/download.py              # full refresh (daily + 4h)
uv run python data/fx/market/download.py --dry-run    # plan only
```

Output layout under `$ML4T_DATA_PATH/fx/`:

```
daily.parquet                 # consolidated daily bars (loader target for frequency="daily")
4h.parquet                    # consolidated 4-hour bars (loader target for frequency="4h")
fx_dictionary.parquet         # pair metadata (base/quote, group, OANDA instrument code)
ohlcv_daily/symbol=<PAIR>/data.parquet     # hive-partitioned per-symbol daily bars (provider-native)
ohlcv_4h/symbol=<PAIR>/data.parquet        # hive-partitioned per-symbol 4h bars
config.yaml                   # pair list + date range + API endpoint config
```

The consolidated parquets at the root are what `load_fx_pairs()` reads.
The per-symbol hive partitions are kept for incremental-refresh work and
for per-symbol exploratory access.

## Loading

```python
from data import load_fx_pairs

df = load_fx_pairs()                              # 4-hour bars (default)
df = load_fx_pairs(frequency="daily")
df = load_fx_pairs(pairs=["EUR_USD", "GBP_USD"])
df = load_fx_pairs(start_date="2020-01-01", end_date="2023-12-31")
```

Schema (canonical):

| Column      | Type     | Description                 |
| ----------- | -------- | --------------------------- |
| `symbol`    | String   | FX pair (e.g., `EUR_USD`)   |
| `timestamp` | Datetime | Bar timestamp (UTC)         |
| `open`      | Float    | Opening price (mid)         |
| `high`      | Float    | High price                  |
| `low`       | Float    | Low price                   |
| `close`     | Float    | Closing price               |
| `volume`    | Int      | Tick volume                 |

## Consumers

- **Ch2**: `12_fx_pairs_eda.py`.
- **Ch7**: `01_data_quality_diagnostics.py`.
- **Ch16 validation**: `validation/weights.py`.
- **Ch18**: `01_cost_taxonomy.py`, `02_spread_estimation.py`, `03_market_impact_calibration.py`.
- **`case_studies/fx_pairs/`**: full pipeline from `01_feasibility_analysis.py` through
  `17_strategy_analysis.py`.
