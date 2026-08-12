# Equity Microstructure Data

Tick-level datasets used in Chapter 3 (Market Microstructure) and related
chapters. Four independent sources at different granularities and cost
points.

| Dataset | Granularity | Source | Access | Disk |
|---------|-------------|--------|--------|------|
| [Trade & Quotes (TAQ)](#trade--quotes-taq) | Tick (trades + NBBO quotes) | AlgoSeek slim | Unzip (no account) | 67 MB |
| [Market by Order (MBO)](#market-by-order-mbo) | Per-order | Databento `XNAS.ITCH` | Paid (~$5, free credit covers) | ~1 GB |
| [NASDAQ ITCH](#nasdaq-itch) | Raw binary (all messages) | NASDAQ public FTP | Free | 4-6 GB/day |
| [IEX HIST](#iex-hist) | Tick (TOPS / DEEP) | IEX public | Free | 150 MB - 10 GB/day |

Every loader lives in `data/equities/loader.py` and raises
`DataNotFoundError` with a runnable download command when data is missing.

## Trade & Quotes (TAQ)

AlgoSeek TAQ slim slice — AAPL on 2020-03-13 (pre-stress) and 2020-03-16
(COVID crash). Two days preserve the original Hive layout so the loader
is identical to the full commercial feed.

| Property | Value |
|----------|-------|
| **Source** | AlgoSeek — <https://algoseek.com/ml-for-trading/>, no account or API key |
| **Frequency** | Tick (trades + NBBO quote events) |
| **Dates** | 2020-03-13, 2020-03-16 |
| **Symbols** | AAPL |
| **Rows** | 21,284,141 events — 13,651,726 and 7,632,415 |
| **Schema** | `timestamp` (µs), `symbol`, `event_type`, `price`, `quantity`, `exchange`, `conditions` |
| **License** | Commercial — slim slice redistributed under reader license |

Downloaded as `symbol=AAPL.zip` (67 MB). It is already parquet in the layout the
loader scans, so there is nothing to convert. Name the members when you unpack it
— Dropbox writes a stray root entry into the archive, and unzipping without
`"*.parquet"` warns and exits 2 having extracted them anyway:

```bash
unzip -q "symbol=AAPL.zip" "*.parquet" \
    -d "$ML4T_DATA_PATH/equities/market/microstructure/trade_and_quotes/symbol=AAPL"
```

The NASDAQ-100 minute-bar archive cannot stand in. Despite the "taq-ext" in its
name it is quote-aware minute-bar aggregates — `OpenBidPrice`, `TradeAtBid`,
`NBBOQuoteCount` and so on — not individual events, and an order book cannot be
reconstructed from bars. It converts to the minute-bar dataset instead; see
[AlgoSeek datasets](../../../README.md#algoseek-datasets).

```python
from data import load_nasdaq100_taq
df = load_nasdaq100_taq(symbols=["AAPL"])
```

The re-encoder that produced the slim slice lives at
[`build_taq_slim.py`](build_taq_slim.py) (zstd level 22, same schema).

**Notebooks**: `03_market_microstructure/11_algoseek_taq_eda.py`,
`03_market_microstructure/12_algoseek_taq_lob_reconstruction.py`.

## Market by Order (MBO)

Databento `XNAS.ITCH` MBO schema — NVDA across November 2024 (10 trading
days). Full order-level messages (add / cancel / modify / fill / trade)
for order-book reconstruction.

| Property | Value |
|----------|-------|
| **Source** | Databento Download Center or API |
| **Frequency** | Tick (per-order) |
| **Dates** | 2024-11-04 to 2024-11-15 |
| **Symbols** | NVDA |
| **Disk** | ~1 GB |
| **Cost** | ~$5 (under $10; new accounts get $125 free credit) |
| **Schema** | `ts_event`, `symbol`, `action`, `side`, `price`, `size`, `order_id`, `flags` |
| **License** | Paid (per-job cost); redistribution prohibited |

**Manual download is preferred** — see
[`MBO_DOWNLOAD.md`](MBO_DOWNLOAD.md) for click-through Databento Download
Center steps.

API-driven alternative (requires `DATABENTO_API_KEY`):

```bash
# Always estimate first to avoid surprise charges
uv run python data/equities/market/microstructure/mbo_download.py --estimate-only
uv run python data/equities/market/microstructure/mbo_download.py
```

```python
from data import load_mbo_data
df = load_mbo_data(symbols=["NVDA"])
files = load_mbo_data(symbols=["NVDA"], list_files=True)  # lazy iteration
```

**Notebooks**: `03_market_microstructure/08_databento_lob_reconstruction.py`,
`09_databento_mbo_analysis.py`, `17_databento_bar_sampling.py`.

## NASDAQ ITCH

Raw TotalView-ITCH message stream from NASDAQ's public FTP mirror.
Includes all order-book messages (add, cancel, delete, execute, trade,
imbalance, status changes).

| Property | Value |
|----------|-------|
| **Source** | NASDAQ public FTP (`emi.nasdaq.com`) |
| **Frequency** | Tick (all message types) |
| **Dates** | Various sample dates (default: 2020-01-30) |
| **Disk** | 4-6 GB per date (compressed binary) |
| **Cost** | Free |
| **License** | NASDAQ ITCH Specification (no restriction on educational use) |

```bash
uv run python data/equities/market/microstructure/nasdaq_itch_download.py --list
uv run python data/equities/market/microstructure/nasdaq_itch_download.py --date 01302020
```

```python
from data import load_nasdaq_itch
messages = load_nasdaq_itch(date="20200130", msg_type="trade")
```

Files are raw binary — parsing happens in the download script; parsed
output lives under `equities/market/microstructure/nasdaq_itch/messages/`.

**Notebooks**: `03_market_microstructure/01_itch_parser.py` through
`07_itch_stylized_facts.py`, plus `14_itch_bar_sampling.py`,
`15_itch_lee_ready.py` and `16_itch_information_bars.py`.

## IEX HIST

IEX exchange historical data, updated T+1 with a rolling 12-month window.
Two feed types available — TOPS (top of book) is small; DEEP (full depth)
is required for limit-order-book reconstruction.

| Property | Value |
|----------|-------|
| **Source** | IEX public (iextrading.com/trading/market-data) |
| **Frequency** | Tick (TOPS: BBO + trades; DEEP: full depth updates) |
| **Retention** | 12 months rolling |
| **Disk** | TOPS ~150-500 MB/day; DEEP ~5-10 GB/day |
| **Cost** | Free |
| **License** | [IEX Historical Data Terms of Use](https://www.iexexchange.io/legal/hist-data-terms) — attribution required |

```bash
uv run python data/equities/market/microstructure/iex_download.py --list
uv run python data/equities/market/microstructure/iex_download.py --smallest     # tiny TOPS sample
uv run python data/equities/market/microstructure/iex_download.py --date 20241220 --deep
```

```python
from data import load_iex_hist
df = load_iex_hist(feed="tops", data_type="trades", symbols=["AAPL"])
raw = load_iex_hist(feed="deep", get_raw_files=True)  # pcap paths for custom parsing
```

Raw pcap files must be parsed before use — the IEX LOB reconstruction
notebook handles this and writes results back under the canonical
`iex/{feed}/parsed/` location.

**Notebooks**: `03_market_microstructure/10_iex_lob_reconstruction.py`.

## Expected On-Disk Layout

```text
equities/market/microstructure/
├── trade_and_quotes/                   # AlgoSeek TAQ — what the loader scans.
│   └── symbol={SYMBOL}/date={YYYYMMDD}.parquet    # the published slice is symbol=AAPL
├── market_by_order/
│   └── {SYMBOL}/xnas-itch-{YYYYMMDD}.mbo.dbn.parquet
├── nasdaq_itch/
│   ├── raw/{date}.bin.gz               # binary downloads
│   └── messages/{msg_type}/{date}.parquet    # parsed
└── iex/
    ├── tops/{YYYYMMDD}.pcap.gz
    ├── tops/parsed/                    # populated by 10_iex_lob_reconstruction.py
    ├── deep/{YYYYMMDD}.pcap.gz
    └── deep/parsed/
```

## Dataset Card

Run the executable dataset card for a side-by-side view:

```bash
uv run python data/equities/market/microstructure/dataset_card.py
```

## Loader Surface

| Loader | Returns | DataNotFoundError prints |
|--------|---------|--------------------------|
| `load_nasdaq100_taq(symbols=...)` | DataFrame (tick events) | the download link and the `unzip` line |
| `load_mbo_data(symbols=..., list_files=...)` | DataFrame or list[Path] | `mbo_download.py --estimate-only` |
| `load_nasdaq_itch(date=..., msg_type=...)` | DataFrame | `nasdaq_itch_download.py --date ...` |
| `load_iex_hist(feed=..., data_type=..., symbols=..., get_raw_files=...)` | DataFrame or list[Path] | `iex_download.py --smallest` or `--deep` |
