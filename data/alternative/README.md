# Alternative Data (cross-asset third-party)

Narrowed scope — datasets that are genuinely cross-asset or not tied to
a single asset class live here. SEC filings and 13F now live under
`data/equities/{fundamentals,positioning}/`; CFTC Commitment of Traders
under `data/futures/positioning/`; on-chain (DefiLlama, CoinGecko) under
`data/crypto/onchain/`.

## Directory

| Subdir | Dataset | Download script | Loader(s) |
| --- | --- | --- | --- |
| `news/` | Bloomberg news archive + FNSPID financial headlines | `news/bloomberg_download.py`, `news/fnspid_download.py` | `load_bloomberg_news`, `load_fnspid` |
| `text/` | Reference / benchmark corpora (Financial Phrasebank) | — (HuggingFace cache) | `load_financial_phrasebank` |

All loaders return Polars DataFrames. Each subdirectory has its own
README with download instructions and storage footprint.

## Quick Start

```bash
# FNSPID news (HuggingFace, ~1 GB)
uv run python data/alternative/news/fnspid_download.py

# Bloomberg news archive (HuggingFace mirror)
uv run python data/alternative/news/bloomberg_download.py
```

## From Python

```python
from data import load_bloomberg_news, load_financial_phrasebank, load_fnspid

news = load_fnspid(symbols=["AAPL", "MSFT"], start_date="2020-01-01")
phrasebook = load_financial_phrasebank(agreement="100")
```

When data is missing, loaders raise `DataNotFoundError` with a pointer
to the relevant subdirectory README and the exact download command.
