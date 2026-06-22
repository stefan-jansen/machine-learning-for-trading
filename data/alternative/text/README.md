# Text Reference Corpora

Labeled text corpora used as training or evaluation data for NLP models.
Unlike `sec/` (filings we produce) or `news/` (news archives we mirror),
these are published academic datasets.

## Datasets

| Corpus | Size | Use case | Loader |
| --- | --- | --- | --- |
| Financial Phrasebank (Malo et al. 2014) | ~2,300–4,800 sentences depending on agreement level | Sentiment classification benchmark | `load_financial_phrasebank` |

## On-disk Layout

```
$ML4T_DATA_PATH/alternative/text/financial_phrasebank/
├── sentences_allagree.parquet    # 100% agreement, ~2,264 rows (default)
├── sentences_75agree.parquet     # ~3,453 rows
├── sentences_66agree.parquet     # ~4,217 rows
└── sentences_50agree.parquet     # ~4,846 rows
```

Total disk footprint: under 500 KB. First load triggers a one-time
HuggingFace download (~1-2 seconds).

## Financial Phrasebank

Academic sentiment benchmark: sentences from financial news labeled
positive/neutral/negative by human annotators. Four agreement levels
are published (100%, 75%, 66%, 50%); the `allagree` subset is the most
reliable but smallest.

**License**: Creative Commons Attribution-NonCommercial-ShareAlike 3.0
(`CC BY-NC-SA 3.0`). Free for academic and non-commercial research;
attribution to Malo et al. (2014) required.

### Download

```bash
# The loader downloads from HuggingFace on first call; no manual step required.
uv run python -c "from data import load_financial_phrasebank; df = load_financial_phrasebank(); print(df.shape)"
```

To pre-populate the cache manually:

```python
from huggingface_hub import hf_hub_download
import zipfile, polars as pl
from pathlib import Path

DATA = Path("$ML4T_DATA_PATH/alternative/text/financial_phrasebank")
zip_path = hf_hub_download("takala/financial_phrasebank",
                           "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
label_map = {"negative": 0, "neutral": 1, "positive": 2}
rows = []
with zipfile.ZipFile(zip_path) as z, z.open(
    "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
) as f:
    for line in f.read().decode("latin-1").strip().splitlines():
        sentence, label = line.rsplit("@", 1)
        rows.append({"sentence": sentence.strip(), "label": label_map[label.strip()]})
DATA.mkdir(parents=True, exist_ok=True)
pl.DataFrame(rows).write_parquet(DATA / "sentences_allagree.parquet")
```

### Loading

```python
from data import load_financial_phrasebank

# Default: 100% agreement subset (most reliable, ~2,264 sentences)
df = load_financial_phrasebank(agreement="100")

# Or lower agreement levels for more training data
df = load_financial_phrasebank(agreement="50")
```

**Reference**: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P.
(2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.*
Journal of the Association for Information Science and Technology, 65(4).

## Consumers

- Chapter 10 NB 01 (word2vec training)
- Chapter 10 NB 03 (sentiment evolution)
- Chapter 10 NB 04 (transformer fine-tuning)
