# Crypto Perpetuals & Premium Index (Binance)

20 major crypto perpetual futures with 1-hour OHLCV and the 8-hourly
premium index. Spot-perpetual basis (the premium index) drives funding
payments; this dataset anchors the Ch12 / Ch17 funding-arbitrage case
study.

## Dataset

- **Source**: Binance USDT-margined futures — public market-data REST
  endpoints (`/fapi/v1/klines`, `/fapi/v1/premiumIndexKlines`).
- **Coverage**: 2021-01-01 → present.
- **Symbols**: 20 perpetuals grouped by theme (majors, DeFi, layer-1).
- **Size on disk**: ~37 MB (perps + premium, hive partitions + consolidated
  parquets).
- **Runtime**: ~8-12 minutes for a full refresh (1-hour bars for 20
  symbols, rate-limited by Binance at 2400 weight/min).
- **API key**: not required (public endpoints; see
  https://developers.binance.com/docs/derivatives/usds-margined-futures).
- **License / attribution**: Binance public market data is free to use
  for personal research under the Binance API Terms
  (https://www.binance.com/en/support/announcement/binance-api-terms-and-conditions-edff9a85aff24eb6afbe15fbe7cc5c47).
  Redistribution of the raw OHLCV as a product is not permitted; derived
  analytics are fine.

## Symbols

| Group    | Symbols (20)                                                                 |
| -------- | ---------------------------------------------------------------------------- |
| Majors   | BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, DOTUSDT, MATICUSDT |
| DeFi     | LINKUSDT, UNIUSDT, AAVEUSDT, MKRUSDT, COMPUSDT                               |
| Layer-1  | ATOMUSDT, NEARUSDT, APTUSDT, SUIUSDT, INJUSDT                                |

## Download

```bash
uv run python data/crypto/market/download.py              # perps + premium index
uv run python data/crypto/market/download.py --dry-run    # plan only
```

Output layout under `$ML4T_DATA_PATH/crypto/`:

```
perps_1h.parquet                           # consolidated 1-hour OHLCV (loader target)
premium_index_8h.parquet                   # consolidated 8-hour premium index (loader target)
ohlcv_1h/symbol=<SYM>/data.parquet         # hive-partitioned per-symbol 1h OHLCV
premium_index/symbol=<SYM>/data.parquet    # hive-partitioned per-symbol premium index
crypto_dictionary.parquet                  # symbol metadata (base, group, listing date)
perps_1h_profile.json                      # summary statistics
premium_index_8h_profile.json
config.yaml                                # symbol list + date range + API endpoint config
```

## Loading

```python
from data import load_crypto_perps, load_crypto_premium

ohlcv = load_crypto_perps()                                    # all 20, 1-hour bars
ohlcv = load_crypto_perps(symbols=["BTCUSDT", "ETHUSDT"])
ohlcv = load_crypto_perps(start_date="2024-01-01")

premium = load_crypto_premium()                                # all 20, 8-hour premium
premium = load_crypto_premium(symbols=["BTCUSDT"])
```

Schemas (canonical):

### Perpetual OHLCV
| Column      | Type     | Description             |
| ----------- | -------- | ----------------------- |
| `symbol`    | String   | Perpetual symbol        |
| `timestamp` | Datetime | Bar timestamp (UTC)     |
| `open`      | Float    | Opening price           |
| `high`      | Float    | High price              |
| `low`       | Float    | Low price               |
| `close`     | Float    | Closing price           |
| `volume`    | Float    | Trading volume          |

### Premium Index
| Column                 | Type     | Description                     |
| ---------------------- | -------- | ------------------------------- |
| `symbol`               | String   | Perpetual symbol                |
| `timestamp`            | Datetime | Funding timestamp (UTC)         |
| `premium_index_open`   | Float    | Premium index at funding open   |
| `premium_index_high`   | Float    | Premium index high              |
| `premium_index_low`    | Float    | Premium index low               |
| `premium_index_close`  | Float    | Premium index at funding close  |

## Consumers

- **Ch2**: `10_crypto_premium_eda.py`.
- **Ch6**: `02_crypto_premium_setup.py` (strategy definition).
- **Ch8**: `04_fundamentals_macro_calendar.py`, `07_event_studies.py`.
- **Ch12**: `01_funding_rate_alpha.py`.
- **Ch16**: `04_single_asset_ml4t_backtest.py`, `09_performance_reporting.py`.
- **`case_studies/crypto_perps_funding/`**: full funding-arbitrage pipeline
  (`01_feasibility_analysis.py` → `17_strategy_analysis.py`).
