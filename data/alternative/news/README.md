# Financial News Archives

Two news corpora used for sentiment analysis, text-signal engineering,
and news-return experiments. Both distributed via HuggingFace Hub.

| Dataset                 | Records                                      | Coverage   | Disk   | Source                                              |
| ----------------------- | -------------------------------------------- | ---------- | ------ | --------------------------------------------------- |
| FNSPID                  | 15.7M headlines, 4,775 S&P 500 companies     | 1999-2023  | ~50 MB (1M sample); ~3 GB full | HuggingFace `Zihan1004/FNSPID` |
| Bloomberg news archive  | ~470k news records + structured fin-data     | 2006-2013  | ~940 MB | HuggingFace (mirrored archive)                      |

Both downloads are HuggingFace public datasets — **no API key required**,
but a free HuggingFace account (`huggingface-cli login`) makes downloads
faster and more reliable.

## FNSPID

Large-scale financial news dataset linking headlines to stock tickers.
Used in Ch10 for news-sentiment features and return-attribution
experiments.

- **Source / citation**: Zhao et al., "FNSPID: A Comprehensive Financial
  News Dataset in Time Series," *arXiv:2402.06698* (2024).
  GitHub: https://github.com/Zdong104/FNSPID_Financial_News_Dataset.
- **License**: the HuggingFace dataset page lists `cc-by-4.0` — free for
  any use (including commercial) with attribution to the FNSPID paper.
- **Size on disk**: 1M sample ~50 MB (default); full ~3 GB.
- **Runtime**: ~30 seconds for 1M sample; ~10 minutes for the full pull.

### Download

```bash
# 1M sample (default, recommended — ~50 MB)
uv run python data/alternative/news/fnspid_download.py

# Larger samples
uv run python data/alternative/news/fnspid_download.py --sample 2000000
uv run python data/alternative/news/fnspid_download.py --sample 0   # full ~15.7M rows

# Preview
uv run python data/alternative/news/fnspid_download.py --dry-run
```

Output under `$ML4T_DATA_PATH/alternative/news/fnspid/`:

```
fnspid_1000k.parquet      # 1M sample (default)
fnspid_2000k.parquet      # when --sample 2000000 is used
fnspid_full.parquet       # --sample 0
```

### Loading

```python
from data import load_fnspid

news = load_fnspid()                                               # newest sample on disk
aapl = load_fnspid(symbols=["AAPL", "MSFT"],
                   start_date="2020-01-01", end_date="2023-12-31")
```

Schema: `symbol`, `timestamp`, `title`, `body`, `source`, `url`.

### Consumers

- **Ch10**: `07_news_return_signals.py`, `08_text_feature_evaluation.py`.

## Bloomberg News Archive

Bloomberg news headlines and bodies combined with structured financial
data series, distributed via a HuggingFace-mirrored archive. Used as a
secondary corpus for sentiment / topic experiments and cross-dataset
robustness checks.

- **Source**: HuggingFace mirrored Bloomberg archive.
- **License**: Bloomberg owns the underlying text; the mirrored archive
  is distributed under HuggingFace terms **for research use only**.
  Commercial redistribution is not permitted. See the dataset card on
  HuggingFace before using for anything beyond personal study.
- **Size on disk**: ~940 MB total (news ~460 MB, structured ~480 MB).
- **Runtime**: ~3-5 minutes.

### Download

```bash
uv run python data/alternative/news/bloomberg_download.py
```

Output under `$ML4T_DATA_PATH/alternative/news/bloomberg/`:

```
bloomberg_news.parquet                   # headline + body corpus (~460 MB)
bloomberg_financial_data.parquet.gzip    # structured financial fields (~480 MB)
.cache/huggingface/download/             # HF download staging (safe to wipe)
```

### Loading

```python
from data import load_bloomberg_news
df = load_bloomberg_news(start_date="2010-01-01", end_date="2013-12-31")
```

The loader covers ``bloomberg_news.parquet`` (headline + body corpus).
The structured-financials file is still read directly when needed —
Bloomberg corpora are secondary to FNSPID; prefer ``load_fnspid()``
when building new pipelines.

### Consumers

- **Ch22**: ESG RAG-vs-fine-tune comparison (``06_esg_rag_vs_finetune.py``).
