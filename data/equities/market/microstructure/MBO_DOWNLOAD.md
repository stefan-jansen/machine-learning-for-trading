# DataBento MBO Download (Manual)

Chapter 3 uses one trading week of **Market-By-Order (MBO)** tick data for **NVDA**
from the NASDAQ TotalView-ITCH feed (Databento dataset `XNAS.ITCH`).

The recommended way to obtain this slice is a **one-off manual download from the
[Databento Download Center](https://databento.com/portal/downloads)** rather than
an API script. It is simpler, costs at most a few dollars, and your files stay
available in the Download Center for 30 days in case you need to re-download.

If you prefer an API-driven workflow, a script is available at
[`mbo_download.py`](mbo_download.py) — see the end of this document.

---

## What you need

- A Databento account. New accounts receive a **$125 free credit**, which is
  more than enough for this slice (expected cost: **under $10**).
- Roughly **~1 GB** of disk space under `$ML4T_DATA_PATH/equities/market/microstructure/market_by_order/NVDA/`.

## Dataset selection

| Field                | Value                       |
| -------------------- | --------------------------- |
| Dataset              | `XNAS.ITCH`                 |
| Schema               | `mbo` (Market by Order)     |
| Symbols              | `NVDA`                      |
| Symbology / `stype`  | `raw_symbol`                |
| Start date           | `2024-11-04`                |
| End date             | `2024-11-15` (inclusive)    |
| Output format        | **Parquet** (`dbn.parquet`) |
| Compression          | None / default              |

This produces 10 files, one per trading day (Nov 4-8 and Nov 11-15).

## Steps

1. Sign in to Databento and open **[Download Center](https://databento.com/portal/downloads)**.
2. Click **Get data** (or **New job**) and select:
   - Dataset: `XNAS.ITCH`
   - Schema: `mbo`
   - Symbol: `NVDA` (stype `raw_symbol`)
   - Date range: `2024-11-04` to `2024-11-15`
   - Output format: **Parquet**
3. Review the cost estimate before submitting. It should be **under $10**.
4. Submit the job. Files appear in the Download Center once processing
   completes (usually a few minutes).
5. Download the ten `xnas-itch-YYYYMMDD.mbo.dbn.parquet` files.

## Where to put the files

Place the downloaded files directly into:

```text
$ML4T_DATA_PATH/equities/market/microstructure/market_by_order/NVDA/
```

The final layout should look like:

```text
$ML4T_DATA_PATH/equities/market/microstructure/market_by_order/NVDA/
├── xnas-itch-20241104.mbo.dbn.parquet
├── xnas-itch-20241105.mbo.dbn.parquet
├── xnas-itch-20241106.mbo.dbn.parquet
├── xnas-itch-20241107.mbo.dbn.parquet
├── xnas-itch-20241108.mbo.dbn.parquet
├── xnas-itch-20241111.mbo.dbn.parquet
├── xnas-itch-20241112.mbo.dbn.parquet
├── xnas-itch-20241113.mbo.dbn.parquet
├── xnas-itch-20241114.mbo.dbn.parquet
└── xnas-itch-20241115.mbo.dbn.parquet
```

No renaming or post-processing is required. The loader globs `*.parquet`, so
Databento's native file names work as-is.

## Verify

From the repo root:

```bash
uv run python -c "
from data import load_mbo_data
files = load_mbo_data(symbols=['NVDA'], list_files=True)
print(f'{len(files)} file(s) found')
for f in files:
    print(' ', f.name)
"
```

You should see 10 files listed. A first-row sanity check:

```bash
uv run python -c "
import polars as pl
from data import load_mbo_data
files = load_mbo_data(symbols=['NVDA'], list_files=True)
df = pl.read_parquet(files[0])
print(df.shape, list(df.columns))
"
```

Expected schema includes `ts_event`, `action`, `side`, `price`, `size`, `order_id`,
`flags`, `sequence` — a single day is typically 8-10 million rows.

If the Download Center window lapses (30 days), simply re-run the same job; the
cost will be identical.

## Why not an API script by default?

- One-off slice: ten days, one symbol. A Download Center job is faster than
  setting up an API key and running a script.
- Files stay available for 30 days, so partial failures are easy to recover
  from without re-billing.
- No `DATABENTO_API_KEY` needed in your environment.

## API alternative

If you already have a Databento API key and prefer automation, the script at
[`mbo_download.py`](mbo_download.py) performs the same download programmatically:

```bash
# Always estimate cost first
uv run python data/equities/market/microstructure/mbo_download.py --estimate-only

# Download (NVDA, Nov 2024 defaults)
uv run python data/equities/market/microstructure/mbo_download.py
```

The script writes to the same target directory and produces files the loader
accepts. Requires `DATABENTO_API_KEY` set in `.env`.
