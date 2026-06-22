# CME Futures (Databento)

30 CME futures products — continuous front-month contracts plus the next
two tenors — for term-structure analysis, carry strategies, and the
`cme_futures` case study. Daily and hourly bars available.

## Dataset

- **Source**: Databento via the `GLBX.MDP3` CME dataset.
- **Coverage**: 2011-01-01 → present, hourly OHLCV with daily aggregation.
- **Products**: 30 (equity index, energy, metals, grains, softs, rates,
  currencies).
- **Tenors**: V0 (front month), V1, V2 for each product.
- **Size on disk**: ~400 MB total (hourly hive partitions + daily
  aggregate).
- **Runtime**: ~20-40 minutes for a full refresh (Databento API is fast;
  the bottleneck is download volume, not rate limiting).
- **API key**: `DATABENTO_API_KEY` required.
- **Cost**: ~$0.05-0.10 per product per year. A full 30-product × 15-year
  refresh runs ~$20-50. **Always** run `--estimate-only` first — new
  Databento accounts receive $125 free credit which is enough for the
  default ES + NQ + CL demo slice but not for a full fetch.
- **License / attribution**: Databento's standard license permits
  personal research and analytics. Redistribution of the raw time
  series as a product is prohibited; derived analytics are fine. See
  https://databento.com/terms.

## Products

| Group         | Symbols (30)                                              |
| ------------- | --------------------------------------------------------- |
| Equity Index  | ES, NQ, RTY, YM                                           |
| Energy        | CL, NG, RB, HO, BZ                                        |
| Metals        | GC, SI, HG, PL                                            |
| Grains        | ZC, ZW, ZS, ZM, ZL, ZO                                    |
| Softs         | KC, CT, SB, CC, OJ                                        |
| Interest Rates| ZB, ZN, ZF, ZT                                            |
| Currencies    | 6E, 6J                                                    |

## Download

```bash
# === Market (Databento — paid, always estimate cost first) ===

uv run python data/futures/market/download.py --estimate-only

uv run python data/futures/market/download.py                        # full
uv run python data/futures/market/download.py --product ES --product NQ
uv run python data/futures/market/download.py --start-date 2020-01-01 --end-date 2023-12-31

# === Positioning (CFTC CoT — free, weekly) ===

uv run python data/futures/positioning/cot_download.py               # all products, 2020-current
uv run python data/futures/positioning/cot_download.py --products ES,NQ,CL,GC --start-year 2010
```

Output layout under `$ML4T_DATA_PATH/futures/`:

```
market/
├── continuous/
│   ├── hourly/product=<PROD>/year=<YYYY>/data.parquet   # raw from Databento
│   └── daily/continuous_daily.parquet                   # session-aligned daily
├── individual/{PRODUCT}/data.parquet                    # individual contract roll demo
└── config.yaml                                          # product list, tenors, Databento codes
positioning/
└── cot/{PRODUCT}.parquet                                # CFTC Commitment of Traders
```

## CFTC Commitment of Traders (free)

Weekly positioning snapshots (Tuesday; released Friday) broken down by
trader category. Used in Ch4 NB 10 for sentiment/positioning features.

```python
from data.futures.loader import load_cot

df = load_cot(products=["ES"], start_date="2020-01-01", end_date="2024-12-31")
df = load_cot()  # everything available locally
```

Schema includes `product`, `report_type`, `report_date`, `open_interest`,
and per-trader long/short/net columns (financial: `dealer_*`, `asset_mgr_*`,
`lev_money_*`; commodity: `commercial_*`, `managed_money_*`, `swap_*`).

## Loading

```python
from data import load_cme_futures

df = load_cme_futures()                              # daily, all products, front + 2 tenors
df = load_cme_futures(frequency="hourly")            # hourly panel
df = load_cme_futures(products=["ES", "NQ", "CL"])
df = load_cme_futures(tenors=[0])                    # front month only
```

Schema (canonical — note `product` instead of `symbol` for CME per the
book's naming convention):

| Column      | Type     | Description                           |
| ----------- | -------- | ------------------------------------- |
| `product`   | String   | CME product code (e.g., ES)           |
| `tenor`     | Int      | 0 = front month, 1 = next, 2 = after  |
| `timestamp` | Datetime | Bar timestamp (daily or hourly)       |
| `open`      | Float    | Opening price                         |
| `high`      | Float    | High price                            |
| `low`       | Float    | Low price                             |
| `close`     | Float    | Closing price                         |
| `volume`    | Int      | Trading volume                        |

## Consumers

- **Ch2**: `05_futures_session_aggregation.py`, `06_cme_futures_eda.py`.
- **Ch6**: `03_cme_futures_setup.py` (carry strategy definition).
- **Ch12-17**: modelling and backtesting.
- **`case_studies/cme_futures/`**: full pipeline from `01_feasibility_analysis.py`
  through `17_strategy_analysis.py`.
