# Prediction Markets

Event-outcome probability data from Kalshi (CFTC-regulated) and Polymarket
(on-chain). Used as an alternative asset class alongside equities, crypto,
and futures.

## Datasets

| Venue | Frequency | Coverage | Loader |
| --- | --- | --- | --- |
| Kalshi | Daily OHLCV | 2021–present | `load_kalshi` |
| Polymarket | Daily OHLCV | 2020–present | `load_polymarket` |

## Download

```bash
# Both venues (~2 minutes, no API key)
uv run python data/prediction_markets/download.py

# Individual
uv run python data/prediction_markets/download.py --provider kalshi
uv run python data/prediction_markets/download.py --provider polymarket
```

### Coverage levers

The defaults live in [`config.yaml`](config.yaml). Override on a per-run
basis without touching the file:

| Flag | Applies to | Effect |
| --- | --- | --- |
| `--start-date YYYY-MM-DD` | both | Override `start` window |
| `--end-date YYYY-MM-DD` | both | Override `end` window |
| `--max-markets-per-category N` | Polymarket | Cap on markets retained per category |
| `--search-results-per-query N` | Polymarket | Search-result depth per query before the per-category cap |
| `--categories crypto,economics` | Polymarket | Replace the category list (each must have a `search_terms` entry) |

```bash
# Wider window, deeper search, more markets per bucket
uv run python data/prediction_markets/download.py \
    --provider polymarket \
    --start-date 2025-01-01 --end-date 2026-12-31 \
    --max-markets-per-category 10 \
    --search-results-per-query 25
```

The resolved values are printed at the top of every run. `--dry-run`
shows what would be downloaded without hitting the API.

## Files Produced

```
$ML4T_DATA_PATH/prediction_markets/
├── kalshi_events.parquet                  # Event-level OHLCV
├── kalshi_events_profile.json             # Schema + summary stats
├── polymarket_events.parquet
├── polymarket_events_profile.json
└── prediction_markets_metadata.parquet    # Combined event metadata
```

## Loading

```python
from data import load_kalshi, load_polymarket

kalshi = load_kalshi(start_date="2025-10-01")
polymarket = load_polymarket(symbols=["WILL-BITCOIN-HIT-150K-BY-JUNE-2026:YES"])
```

Both loaders accept `symbols`, `start_date`, and `end_date` filters and
return canonical `(timestamp, symbol, open, high, low, close, volume)`
schema. Polymarket also surfaces a `category` column (provider label).

## Consumers

- Chapter 2 — cross-asset EDA (demonstrates events-as-assets framing)
- Chapter 6 — prediction-market case study (funding-rate-style alpha on event resolution)

## License / Cost / Size

- **Kalshi**: public REST endpoint; no API key required for historical
  candles. Kalshi's developer terms permit personal research use
  (https://kalshi.com/developers/terms). Cite "Kalshi" when publishing
  results.
- **Polymarket**: public Gamma/CLOB endpoints; no API key for historical
  candles. Polymarket's ToS permit read-only research use
  (https://polymarket.com/terms-of-service). Cite "Polymarket" when
  publishing.
- **Disk**: under 2 MB total for both venues (historical OHLCV only —
  no order-book snapshots).
- **Runtime**: ~2 minutes for a full refresh.

## Notes

Kalshi is CFTC-regulated and cash-settled; Polymarket settles on-chain
in stablecoin and is blocked from US users. The downloader does not
authenticate as a trader — it only pulls historical candles.
