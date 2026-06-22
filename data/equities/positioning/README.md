# Equity Positioning: 13F + Form 4

SEC regulatory filings that capture **positions** and **insider activity**
at the equity level. All data is public domain (SEC EDGAR). Respect the
10 req/sec rate limit via a descriptive `User-Agent` — downloaders here
handle this automatically.

## Datasets

| Dataset | Filing | Universe | Script | Loader(s) |
| --- | --- | --- | --- | --- |
| Institutional holdings | 13F | Curated 10 managers (per-cik) or full universe (bulk) | `13f_download.py` | `load_institutional_holdings_13f`, `load_13f_stock_features`, `load_13f_edges`, `load_13f_bulk_holdings` |
| Insider transactions | Form 4 | User-chosen tickers | `form4_download.py` | Raw XML — read directly via `pathlib` (see Ch4 NB 03) |

## Download Commands

```bash
# === 13F (institutional holdings) ===

# per-cik (default): 10 curated managers, 4 most recent 13F-HR filings each
uv run python data/equities/positioning/13f_download.py
uv run python data/equities/positioning/13f_download.py --num-filings 8
uv run python data/equities/positioning/13f_download.py --max-institutions 3

# bulk: one or more quarterly filing windows (SEC's own labels)
uv run python data/equities/positioning/13f_download.py --mode bulk --quarters 2024Q3
uv run python data/equities/positioning/13f_download.py --mode bulk --quarters 2024Q2,2024Q3

# === Form 4 (insider transactions) ===

uv run python data/equities/positioning/form4_download.py --ticker TSLA --count 20
uv run python data/equities/positioning/form4_download.py --ticker TSLA,AAPL,MSFT --count 10
```

## Directory Layout

```
$ML4T_DATA_PATH/equities/positioning/
├── 13f/
│   ├── institutional_holdings.parquet       # per-cik: raw holdings (10 curated managers)
│   ├── institution_stock_edges.parquet      # per-cik: institution → stock edges
│   ├── stock_features.parquet               # per-cik: stock-level features
│   ├── coownership_matrix.npy               # per-cik: stock × stock similarity
│   ├── coownership_stocks.txt               # per-cik: row/col CUSIPs
│   └── bulk/
│       └── {YYYYQN}/
│           ├── institutional_holdings.parquet   # bulk: full-window universe (~3M rows)
│           └── bulk_13f.zip                     # cached raw SEC zip
└── form4/
    └── {TICKER}/{accession}.xml             # Raw Form 4 insider filings
```

## Loading

```python
from data import (
    load_institutional_holdings_13f,
    load_13f_stock_features,
    load_13f_edges,
    load_13f_bulk_holdings,
)

# 13F per-cik (curated managers)
holdings = load_institutional_holdings_13f(start_date="2024-01-01")
features = load_13f_stock_features()
edges = load_13f_edges()

# 13F bulk (full universe, one quarter)
q3_2024 = load_13f_bulk_holdings("2024Q3")
```

## Consumers

- **Ch4 NB 03** — Form 4 insider-transaction parsing
- **Ch4 NB 05** — Full-universe 13F analysis via `--mode bulk`
- **Ch10 NB 02** — Asset embeddings from 13F holdings
- **Ch22 NB 07** — Institutional ownership graph + crowding signals
- **Ch23** — `09_knowledge_graph_features`, `05_institutional_holdings_kg`, `04_rag_comparison_benchmark`
