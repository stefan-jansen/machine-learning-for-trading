# Futures Positioning: CFTC Commitment of Traders

Weekly positioning snapshots (Tuesday; released Friday at 3:30 PM ET) for
CME / ICE / CBOT futures, broken down by trader category. Free and public
domain. Used in Ch4 NB 10 for sentiment/positioning features and in
Ch8 / Ch16 for futures strategy inputs.

## Dataset

| Report type | Trader categories | Products |
| --- | --- | --- |
| TFF (Traders in Financial Futures) | Dealers, Asset Managers, Leveraged Money | Financial futures (ES, NQ, 6E, ZN, …) |
| Disaggregated | Commercials, Managed Money, Swap Dealers | Commodity futures (CL, GC, ZC, …) |

Product mapping + report-type dispatch live in the `ml4t.data.cot` library
(`ml4t.data.cot.PRODUCT_MAPPINGS`). The downloader here wraps that library
and persists one parquet per product to the local data store.

## Download

```bash
# Default: all products in PRODUCT_MAPPINGS, 2020 through current year
uv run python data/futures/positioning/cot_download.py

# Subset of products + longer history
uv run python data/futures/positioning/cot_download.py --products ES,NQ,CL,GC --start-year 2010

# Override output root
uv run python data/futures/positioning/cot_download.py --data-path /tmp/ml4t-data
```

## Directory Layout

```
$ML4T_DATA_PATH/futures/positioning/cot/
└── {PRODUCT}.parquet    # one parquet per product code (e.g., ES.parquet)
```

## Schema

Columns vary by report type but always include:

| Column | Notes |
| --- | --- |
| `product` | Exchange product code (ES, CL, GC, …) |
| `report_type` | CFTC report that produced the row |
| `report_date` | Tuesday snapshot date |
| `open_interest` | Total open interest |

Per-trader long/short/net columns by report type:

- **Financial (TFF)**: `dealer_long/short/net`, `asset_mgr_long/short/net`, `lev_money_long/short/net`
- **Commodity (disaggregated)**: `commercial_long/short/net`, `managed_money_long/short/net`, `swap_long/short/net`

## Loading

```python
from data import load_cot, list_cot_products

# Everything available locally
df = load_cot()

# Subset + date filter
df = load_cot(products=["ES", "NQ"], start_date="2020-01-01", end_date="2024-12-31")

# Enumerate what's been downloaded
list_cot_products()  # -> ['CL', 'ES', 'GC', 'NQ', ...]
```

`load_cot()` uses `diagonal_relaxed` concat so financial and commodity
products can be combined in one frame despite their different schemas.

## Release Lag

CFTC publishes reports Friday at 3:30 PM ET; the snapshot is as-of
**Tuesday** of the same week (3-day lag). For daily-bar backtests a
conservative +6 calendar-day availability lag from Tuesday is standard.

## Consumers

- **Ch4 NB 10** — positioning analysis, z-scores, contrarian signals
- **Ch8** — futures_features.py (positioning feature family)
- **Ch16** — futures strategies using CoT signals
