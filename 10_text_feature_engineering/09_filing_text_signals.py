# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SEC Filing Signals: From 10-Q Text to Alpha Factors
#
# **Chapter 10: Text Feature Engineering**
#
# **Docker image**: `ml4t-gpu`
#
# **Section Reference**: See Section 10.5 for practitioner workflow and signal validation protocol
#
# > **GPU recommended**: this notebook runs FinBERT and sentence-transformer
# > inference over thousands of MD&A passages (no training). On a GPU the
# > MAX_SYMBOLS=50 default lands in ~5 minutes; CPU is 5–10× slower. For GPU:
# > ```bash
# > docker compose run --rm ml4t-gpu python 10_text_feature_engineering/09_filing_text_signals.py
# > ```
#
#
# ## Purpose
#
# This notebook demonstrates how to construct alpha factors from SEC 10-Q filings.
# Unlike headline-based signals (NB07), corporate filings provide dense, structured text
# that reflects management's assessment of financial condition. We extract two complementary
# signal types from MD&A sections:
#
# 1. **Sentiment signals** via FinBERT (directional bias in management language)
# 2. **Semantic change signals** via sentence-transformer embeddings (quarter-over-quarter narrative shifts)
#
# The filing date provides a natural point-in-time anchor: the signal becomes available
# when the SEC accepts the filing, not when the quarter ends.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Load and explore SEC 10-Q MD&A text at scale
# - Apply FinBERT sentiment scoring to long-form corporate text
# - Compute document embeddings using sentence-transformers
# - Construct a "narrative change" signal from sequential filing embeddings
# - Join text signals to market data with point-in-time correctness
# - Evaluate signal quality using IC, ICIR, and quintile analysis
#
# ## Prerequisites
# - Section 10.5 of the chapter (alpha-factor evaluation, point-in-time joins).
# - SEC 10-Q MD&A panel produced by `data/equities/fundamentals/filings_download.py`.
#
# ## Related Notebooks
# - `04_bert_finetuning.py` / `06_finbert_cross_dataset.py` — FinBERT model details.
# - `07_news_return_signals.py` — analogous workflow on headlines instead of filings.

# %%
"""SEC Filing Text Signals - FinBERT sentiment and embedding-based alpha factors."""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from utils.paths import get_chapter_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, show_with_alt

# The tokenizer's Rust parallelism warns on every batch once this process has forked, and the
# inference passes below do not need it.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# %% [markdown]
# Zero means every symbol. FinBERT scores one filing at a time, so runtime is close to linear
# in the number of filings and the default trades coverage for a notebook that finishes.


# %% tags=["parameters"]
SEED = 42
MAX_SYMBOLS = 50
MAX_FILINGS = 0
BATCH_SIZE = 8
MAX_TOKENS = 512
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTIMENT_MODEL = "yiyanghkust/finbert-tone"

# %%
OUTPUT_DIR = get_chapter_dir(10) / "output" / "filing_signals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility — set_global_seeds covers Python random / NumPy / Torch.
set_global_seeds(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

# %% [markdown]
# ## 1. Load SEC 10-Q MD&A Data
#
# The MD&A (Management's Discussion and Analysis) section is the most valuable narrative
# section of quarterly filings. Unlike boilerplate Risk Factors that change slowly,
# MD&A discusses current quarter performance and forward-looking outlook.
#
# Data comes from our SEC EDGAR download script
# (`data/equities/fundamentals/filings_download.py --form 10-Q --universe sp500`),
# which extracts MD&A sections from S&P 500 10-Q filings (2017-2021).

# %%
from data import load_sp500_10q_mda

filings = load_sp500_10q_mda()

print(f"Loaded {len(filings):,} MD&A sections from {filings['symbol'].n_unique()} companies")
print(f"Date range: {filings['filing_date'].min()} to {filings['filing_date'].max()}")

if MAX_SYMBOLS > 0:
    # Ties broken by symbol: `group_by` returns rows in no fixed order, so sorting on the
    # count alone picks different symbols at the cutoff on each run.
    top_symbols = (
        filings.group_by("symbol")
        .len()
        .sort(["len", "symbol"], descending=[True, False])
        .head(MAX_SYMBOLS)["symbol"]
        .to_list()
    )
    filings = filings.filter(pl.col("symbol").is_in(top_symbols))
    print(f"Filtered to {MAX_SYMBOLS} symbols: {len(filings):,} filings")

if MAX_FILINGS > 0 and len(filings) > MAX_FILINGS:
    filings = filings.sort(["filing_date", "symbol"]).head(MAX_FILINGS)
    print(f"Reduced to first {MAX_FILINGS} filings for test run")

# %% [markdown]
# ### One signal per filing date
#
# A catch-up filer can submit several quarters' 10-Qs on one date. The investable event is
# the date, so only the most recent period is kept. Left in, the duplicates give the sentiment
# and narrative tables repeated keys, the later join fans out on both sides, and those firms
# are counted several times in every information coefficient below.


# %%
n_before = len(filings)
filings = filings.sort(["symbol", "filing_date", "period_end"]).unique(
    subset=["symbol", "filing_date"], keep="last", maintain_order=True
)
if len(filings) < n_before:
    print(
        f"Collapsed {n_before - len(filings)} same-date multi-quarter filings -> {len(filings):,}"
    )

# Compute MD&A word count from the canonical `text` column.
filings = filings.with_columns(pl.col("text").str.split(" ").list.len().alias("word_count"))

filings.head(5).select(["symbol", "filing_date", "period_end", "word_count"])

# %%
# Word count distribution
print("MD&A word count statistics:")
print(filings["word_count"].describe())

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["dual_h_tall"])

axes[0].hist(filings["word_count"].to_numpy(), bins=50, color=COLORS["blue"])
axes[0].set_xlabel("Words in the MD&A section")
axes[0].set_ylabel("Filings")
axes[0].set_title("Length of the extracted MD&A text")
axes[0].axvline(
    filings["word_count"].median(), color=COLORS["amber"], linestyle="--", label="Median"
)
axes[0].legend(fontsize=6)

quarterly = (
    filings.with_columns(
        quarter=pl.col("filing_date").dt.year().cast(pl.String)
        + "-Q"
        + pl.col("filing_date").dt.quarter().cast(pl.String)
    )
    .group_by("quarter")
    .len()
    .sort("quarter")
)
axes[1].bar(range(len(quarterly)), quarterly["len"].to_numpy(), color=COLORS["blue"])
axes[1].set_xticks(range(0, len(quarterly), 4))
axes[1].set_xticklabels(quarterly["quarter"].to_list()[::4], rotation=45, fontsize=6)
axes[1].set_ylabel("Filings")
axes[1].set_xlabel("Calendar quarter the filing was accepted")
axes[1].set_title("Filings accepted per calendar quarter")

show_with_alt(
    fig,
    "Two panels. The left is a histogram of MD&A length in words, sharply peaked a few "
    "thousand words in with a long thin tail reaching several times further right, and a "
    "dashed median line just right of the peak. The right is a bar per calendar quarter "
    "across the sample: the bars repeat a four-quarter pattern in which one quarter of each "
    "year carries roughly a fifth as many filings as the other three.",
)

# %% [markdown]
# The gap in the right panel is a property of the forms, not of the sample. A company files
# three 10-Qs a year and a 10-K for the fourth quarter, and the 10-K is the form that lands
# in the first calendar quarter for most filers. So the quarter that looks empty is the one
# whose disclosure went into an annual report this notebook does not read.
#
# Anything read seasonally from these signals has to account for that: one quarter in four is
# a different and much smaller sample of companies, not a quiet period.

# %% [markdown]
# ## 2. FinBERT Sentiment Scoring
#
# FinBERT processes text at the sentence level (max 512 tokens). For long MD&A sections
# (median ~7,800 words), we use a chunking strategy:
#
# 1. Split MD&A into sentences
# 2. Score each sentence with FinBERT (positive/negative/neutral probabilities)
# 3. Aggregate sentence scores to document-level sentiment
#
# This mirrors how analysts read filings: extracting overall tone from many paragraphs.
# The aggregation captures both the **average sentiment** (management tone) and
# **sentiment dispersion** (mixed signals within the same filing).

# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, set_seed

set_seed(SEED)
print(f"Loading FinBERT: {SENTIMENT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
model = model.to(device)
model.eval()

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device,
    truncation=True,
    max_length=MAX_TOKENS,
    batch_size=BATCH_SIZE,
)
print("FinBERT loaded")


# %% [markdown]
# ### Chunking Strategy
#
# MD&A sections average ~9,500 words but FinBERT accepts only 512 tokens (~380 words).
# We split into paragraphs and score each one, then aggregate.


# %%
def chunk_text(text: str, max_chars: int = 1500) -> list[str]:
    """Split text into chunks suitable for FinBERT (roughly 512 tokens each)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    if not paragraphs:
        # Fall back to sentence splitting
        paragraphs = [s.strip() + "." for s in text.split(".") if len(s.strip()) > 30]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text[:max_chars]]


# %%
def score_document_sentiment(text: str) -> dict:
    """Score full MD&A document by aggregating chunk-level FinBERT predictions."""
    chunks = chunk_text(text)
    if not chunks:
        return {"sentiment_mean": 0.0, "sentiment_std": 0.0, "n_chunks": 0}

    # Score all chunks
    results = sentiment_pipeline(chunks)

    # Convert labels to numeric: positive=+1, neutral=0, negative=-1
    label_map = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
    scores = []
    for r in results:
        label = r["label"]
        confidence = r["score"]
        numeric = label_map.get(label, 0.0) * confidence
        scores.append(numeric)

    scores_arr = np.array(scores)
    return {
        "sentiment_mean": float(scores_arr.mean()),
        "sentiment_std": float(scores_arr.std()) if len(scores_arr) > 1 else 0.0,
        "sentiment_pos_pct": float((scores_arr > 0).mean()),
        "sentiment_neg_pct": float((scores_arr < 0).mean()),
        "n_chunks": len(chunks),
    }


# %%
# Score all filings
print(f"Scoring {len(filings):,} MD&A sections with FinBERT...")
print("(This may take several minutes depending on GPU/CPU)")

sentiment_records = []
for i, row in enumerate(filings.iter_rows(named=True)):
    scores = score_document_sentiment(row["text"])
    scores["symbol"] = row["symbol"]
    scores["filing_date"] = row["filing_date"]
    sentiment_records.append(scores)

    if (i + 1) % 100 == 0 or (i + 1) == len(filings):
        print(f"  Scored {i + 1:,}/{len(filings):,} filings")

sentiment_df = pl.DataFrame(sentiment_records)
print(f"\nSentiment scoring complete: {len(sentiment_df):,} filings scored")
sentiment_df.head(5)

# %%
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE["triple_h_tall"])

axes[0].hist(sentiment_df["sentiment_mean"].to_numpy(), bins=50, color=COLORS["blue"])
axes[0].set_xlabel("Mean chunk sentiment")
axes[0].set_ylabel("Filings")
axes[0].set_title("Sentiment averaged over a filing")
axes[0].axvline(0, color=COLORS["amber"], linestyle="--")

axes[1].hist(sentiment_df["sentiment_std"].to_numpy(), bins=50, color=COLORS["blue"])
axes[1].set_xlabel("Standard deviation across chunks")
axes[1].set_ylabel("Filings")
axes[1].set_title("Spread of sentiment within a filing")

axes[2].hist(
    sentiment_df["sentiment_pos_pct"].to_numpy(),
    bins=30,
    alpha=0.7,
    color=COLORS["blue"],
    label="Chunks scored positive",
)
axes[2].hist(
    sentiment_df["sentiment_neg_pct"].to_numpy(),
    bins=30,
    alpha=0.7,
    color=COLORS["amber"],
    label="Chunks scored negative",
)
axes[2].set_xlabel("Share of a filing's chunks")
axes[2].set_ylabel("Filings")
axes[2].set_title("Positive and negative shares, overlaid")
axes[2].legend(fontsize=6)

for ax in axes:
    ax.tick_params(labelsize=7)

show_with_alt(
    fig,
    "Three histograms over filings. The first is the mean sentiment of a filing's chunks, a "
    "single hump whose bulk sits to the right of the dashed zero line. The second is the "
    "spread of sentiment within a filing, a symmetric hump centered well above zero, so a "
    "typical filing contains chunks scored both ways rather than a uniform tone. The third "
    "overlays the share of chunks scored positive against the share scored negative: the "
    "negative distribution is concentrated near zero and the positive one sits to its right, "
    "with the two overlapping in the middle.",
)

# %% [markdown]
# The middle panel is the one to read before using the left one. Sentiment averaged over a
# filing is a mean of chunk scores whose spread is substantial, so two filings with the same
# mean can differ in whether the document was uniformly mild or a mix of strongly positive
# and strongly negative passages. That is why `sentiment_std` is carried as its own signal
# rather than discarded as noise around the mean.

# %% [markdown]
# ## 3. Document Embeddings and Narrative Change
#
# Beyond sentiment polarity, we capture **semantic content** using sentence-transformer
# embeddings. The key signal is **narrative change**: how much the MD&A text shifts
# from one quarter to the next.
#
# Intuition: a large semantic shift between consecutive filings suggests material
# new information that the market may not have fully priced. This is analogous to
# the "news surprise" factor in NB07, but applied to corporate disclosures.

# %%
from sentence_transformers import SentenceTransformer

print(f"Loading embedding model: {EMBEDDING_MODEL}")
embed_model = SentenceTransformer(EMBEDDING_MODEL, device=str(device))
print(f"Embedding dimension: {embed_model.get_sentence_embedding_dimension()}")


# %% [markdown]
# ### Document Embedding Strategy
#
# Full MD&A texts are too long for a single embedding pass. We use **mean pooling
# over chunk embeddings**: embed each paragraph/chunk, then average. This captures
# the overall semantic content while respecting model token limits.


# %%
def embed_document(text: str, model: SentenceTransformer) -> np.ndarray:
    """Compute document embedding by mean-pooling chunk embeddings."""
    chunks = chunk_text(text, max_chars=1200)
    if not chunks:
        return np.zeros(model.get_sentence_embedding_dimension())

    chunk_embeddings = model.encode(chunks, show_progress_bar=False, batch_size=32)
    return chunk_embeddings.mean(axis=0)


# %%
# Compute embeddings for all filings
print(f"Computing embeddings for {len(filings):,} filings...")

embeddings = []
for i, row in enumerate(filings.iter_rows(named=True)):
    emb = embed_document(row["text"], embed_model)
    embeddings.append(emb)

    if (i + 1) % 100 == 0 or (i + 1) == len(filings):
        print(f"  Embedded {i + 1:,}/{len(filings):,} filings")

embeddings_array = np.stack(embeddings)
print(f"Embedding matrix: {embeddings_array.shape}")

# %%
filing_order = (
    filings.select(["symbol", "filing_date"]).with_row_index("idx").sort(["symbol", "filing_date"])
)

narrative_changes = []
prev_emb_by_symbol = {}

for row in filing_order.iter_rows(named=True):
    idx = row["idx"]
    symbol = row["symbol"]
    emb = embeddings_array[idx]

    if symbol in prev_emb_by_symbol:
        prev_emb = prev_emb_by_symbol[symbol]
        # Cosine distance (0 = identical, 2 = opposite)
        cos_sim = np.dot(emb, prev_emb) / (np.linalg.norm(emb) * np.linalg.norm(prev_emb) + 1e-8)
        cos_dist = 1.0 - cos_sim
    else:
        cos_dist = None  # No previous quarter

    narrative_changes.append(
        {
            "symbol": symbol,
            "filing_date": row["filing_date"],
            "narrative_change": cos_dist,
        }
    )
    prev_emb_by_symbol[symbol] = emb

narrative_df = pl.DataFrame(narrative_changes)
print(
    f"Narrative change computed for {narrative_df.drop_nulls('narrative_change').height:,} filings"
)
print("  (first filing per company has no prior quarter for comparison)")

narrative_df.drop_nulls("narrative_change")["narrative_change"].describe()

# %%
valid_changes = narrative_df.drop_nulls("narrative_change")["narrative_change"].to_numpy()

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.hist(valid_changes, bins=50, color=COLORS["blue"])
ax.set_xlabel("Cosine distance between consecutive filings by the same company")
ax.set_ylabel("Filings")
ax.set_title("Quarter-over-quarter change in MD&A text")
ax.axvline(np.median(valid_changes), color=COLORS["amber"], linestyle="--", label="Median")
ax.legend(fontsize=6)

show_with_alt(
    fig,
    "A histogram of the cosine distance between each filing's MD&A embedding and the same "
    "company's previous one. The distribution is a single hump concentrated at small "
    "distances, with a dashed median line inside it and a tail extending to larger distances "
    "that thins out well before the axis ends.",
)

# %% [markdown]
# Most consecutive filings are close together, which is what boilerplate does: an MD&A is
# largely carried forward and edited, so the base rate of change is low and the tail is where
# something was rewritten. The signal is the distance relative to that base rate, not the
# distance itself.

# %% [markdown]
# ## 4. Combine Signals and Join to Market Data
#
# Two signal families are available:
# - **Sentiment signals**: mean, std, positive/negative fractions
# - **Narrative change**: cosine distance between consecutive filings
#
# We join these to AlgoSeek S&P 500 daily prices using the **filing_date** as the
# point-in-time anchor. The signal becomes investable on the filing date itself
# (SEC filings are public immediately upon acceptance).

# %%
# Merge sentiment and narrative change
signals = sentiment_df.join(
    narrative_df.select(["symbol", "filing_date", "narrative_change"]),
    on=["symbol", "filing_date"],
    how="left",
)
print(f"Combined signals: {signals.shape}")
signals.head(5)

# %%
# Load S&P 500 daily prices
from data import load_sp500_daily_bars

prices = load_sp500_daily_bars()
print(f"Loaded {len(prices):,} price observations for {prices['symbol'].n_unique()} symbols")

# Compute forward returns: return from day t to day t+N
price_returns = (
    prices.sort(["symbol", "timestamp"])
    .with_columns(
        fwd_1d=(pl.col("close").shift(-1) / pl.col("close") - 1).over("symbol"),
        fwd_5d=(pl.col("close").shift(-5) / pl.col("close") - 1).over("symbol"),
        fwd_20d=(pl.col("close").shift(-20) / pl.col("close") - 1).over("symbol"),
    )
    .select(["symbol", "timestamp", "fwd_1d", "fwd_5d", "fwd_20d"])
)

# %% [markdown]
# ### The point-in-time join
#
# Each filing date is matched forward to the first trading day on or after it. Forward, not
# backward: a backward match would attach the signal to a session that closed before the
# filing existed, which is the look-ahead this whole construction is arranged to avoid. An
# SEC filing is public on acceptance, so the filing date itself is investable when it is a
# trading day.


# %%
prices_with_trade_date = price_returns.with_columns(trade_date=pl.col("timestamp")).sort(
    ["symbol", "timestamp"]
)

# Prepare signals: rename filing_date -> timestamp for the asof join key
signals_for_join = signals.rename({"filing_date": "timestamp"}).sort(["symbol", "timestamp"])

# Asof join: for each signal date, find the nearest price date >= signal date
# strategy="forward" means: match the next trading day on or after the signal date
eval_df = (
    signals_for_join.join_asof(
        prices_with_trade_date,
        on="timestamp",
        by="symbol",
        strategy="forward",
    )
    .rename({"timestamp": "filing_date"})
    .drop_nulls(["fwd_5d"])
)

print(f"Evaluation dataset: {len(eval_df):,} observations ({eval_df['symbol'].n_unique()} symbols)")
print(f"Date range: {eval_df['filing_date'].min()} to {eval_df['filing_date'].max()}")

# %% [markdown]
# ## 5. Signal Evaluation: Information Coefficients
#
# We evaluate each signal using rank Information Coefficients (IC): the Spearman
# correlation between signal values and subsequent returns. A good alpha factor
# should show consistent, positive IC over time.
#
# The **IC** measures predictive power per cross-section (one date), while
# **ICIR** (IC / std(IC)) measures consistency across dates.

# %%
from scipy.stats import spearmanr

signal_cols = [
    "sentiment_mean",
    "sentiment_std",
    "sentiment_pos_pct",
    "sentiment_neg_pct",
    "narrative_change",
]
return_cols = ["fwd_1d", "fwd_5d", "fwd_20d"]

# %% [markdown]
# The IC is pooled across all filing and forward-return pairs, and each one is paired with a
# cluster bootstrap: resample whole symbols with replacement, recompute the pooled IC on each
# replicate, and report the percentile interval and a two-sided bootstrap p-value against the
# null that the IC is zero.
#
# The bootstrap needs a cluster id, so filings with no symbol are dropped before both the
# point estimate and the resampling. That shrinks the observation count per row slightly and
# is deliberate: the alternative is a point estimate computed on rows the interval could not
# be computed on.
#
# Why a cluster bootstrap on symbols? The pooled sample places the same firm
# at multiple quarterly filings into one correlation. Returns are also
# overlapping for the 5-day and 20-day horizons. The i.i.d. t-stat formula
# `t = r·sqrt((n-2)/(1-r²))` would therefore overstate significance. Cluster
# bootstrap on symbols preserves the within-firm dependence between filings
# and overlapping returns. Treat the headline ICs as **screening** values;
# the chapter's headline inference framework uses HAC on cross-sectional IC
# series with adequate breadth (see NB07, NB08).

# %% [markdown]
# A replicate that drops too many observations to be meaningful is discarded, and if fewer
# than `MIN_VALID_BOOT` of the replicates survive, the interval and p-value for that pair are
# reported as missing rather than computed from what remains. A draw dominated by one cluster
# still produces a number, and a missing entry says that where a computed one would not.


# %%
print("Signal Evaluation: Pooled ICs with cluster bootstrap (cluster=symbol)")
print("=" * 70)

N_BOOT = 1000
# One RNG across every (signal, horizon) pair, so the table reproduces only for a fixed
# iteration order: reorder either list and every later pair draws differently.
MIN_VALID_BOOT = 200
_boot_rng = np.random.default_rng(SEED)


def _pooled_spearman(sig_vals: np.ndarray, ret_vals: np.ndarray) -> float:
    """Pooled Spearman correlation, robust to constant-signal slices."""
    if len(sig_vals) < 5:
        return np.nan
    if np.std(sig_vals) == 0 or np.std(ret_vals) == 0:
        return np.nan
    return float(spearmanr(sig_vals, ret_vals)[0])


ic_results = []
for sig in signal_cols:
    for ret in return_cols:
        valid = eval_df.select(["symbol", sig, ret]).drop_nulls()
        if valid.height < 20:
            continue
        symbols = valid["symbol"].to_numpy()
        sig_vals = valid[sig].to_numpy()
        ret_vals = valid[ret].to_numpy()
        n = len(sig_vals)

        ic_point = _pooled_spearman(sig_vals, ret_vals)

        # Cluster bootstrap: resample whole symbols with replacement.
        unique_symbols = np.unique(symbols)
        idx_by_symbol = {s: np.where(symbols == s)[0] for s in unique_symbols}
        boot_ics = np.empty(N_BOOT)
        for b in range(N_BOOT):
            drawn = _boot_rng.choice(unique_symbols, size=len(unique_symbols), replace=True)
            indices = np.concatenate([idx_by_symbol[s] for s in drawn])
            boot_ics[b] = _pooled_spearman(sig_vals[indices], ret_vals[indices])
        boot_ics = boot_ics[~np.isnan(boot_ics)]

        if boot_ics.size >= MIN_VALID_BOOT:
            ci_lo = float(np.percentile(boot_ics, 2.5))
            ci_hi = float(np.percentile(boot_ics, 97.5))
            # Percentile-method test inversion: the smallest alpha whose interval excludes
            # zero. Not a recentered reflection test, which would be
            # `mean(|boot - boot.mean()| >= |ic_point|)`.
            p_boot = 2.0 * min(
                float(np.mean(boot_ics <= 0.0)),
                float(np.mean(boot_ics >= 0.0)),
            )
        else:
            ci_lo = ci_hi = p_boot = np.nan

        ic_results.append(
            {
                "signal": sig,
                "horizon": ret,
                "ic": round(ic_point, 4) if not np.isnan(ic_point) else np.nan,
                "ci95_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
                "ci95_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
                "p_cluster_boot": round(p_boot, 4) if not np.isnan(p_boot) else np.nan,
                "n_obs": n,
                "n_symbols": int(len(unique_symbols)),
            }
        )

if ic_results:
    ic_summary = pl.DataFrame(ic_results).sort(["signal", "horizon"])
    print(ic_summary)
    print(
        "\nInference: cluster bootstrap with N_BOOT=1000 replicates; cluster = symbol.\n"
        "Bootstrap CIs and p-values supersede the i.i.d. t-stat formula because\n"
        "the pooled sample contains multiple filings per firm and overlapping\n"
        "forward-return windows. Treat ICs as screening; HAC on cross-sectional\n"
        "IC series (NB07, NB08) is the chapter's headline inference framework."
    )
else:
    print("Insufficient data for IC computation (need more symbols/filings)")

# %% [markdown]
# The point estimates with their cluster-bootstrap intervals, charted so the signal-horizon
# pairs separated from zero stand out rather than having to be read off the table above.
#
# The interval lengths carry as much as the bar heights here. Where an interval is long
# relative to the bar it sits on, the point estimate is a draw from a wide distribution and
# its sign is not information.

# %%
if len(ic_summary) > 0:
    signals = ic_summary["signal"].unique(maintain_order=True).to_list()
    # Sorted by the horizon they name rather than alphabetically, which orders a set like
    # 1d / 5d / 20d as 1, 20, 5 and puts the legend out of sequence with the axis.
    horizons = sorted(
        ic_summary["horizon"].unique().to_list(),
        key=lambda label: int("".join(ch for ch in label if ch.isdigit()) or 0),
    )
    x = np.arange(len(signals))
    width = 0.8 / max(len(horizons), 1)
    fig, ax = plt.subplots(figsize=FIGSIZE["single_tall"])
    for i, h in enumerate(horizons):
        by_sig = {
            r["signal"]: r for r in ic_summary.filter(pl.col("horizon") == h).iter_rows(named=True)
        }
        ic = np.array([by_sig.get(s, {}).get("ic", np.nan) for s in signals], dtype=float)
        lo = np.array([by_sig.get(s, {}).get("ci95_lo", np.nan) for s in signals], dtype=float)
        hi = np.array([by_sig.get(s, {}).get("ci95_hi", np.nan) for s in signals], dtype=float)
        yerr = np.nan_to_num(np.vstack([ic - lo, hi - ic]), nan=0.0)
        ax.bar(x + i * width, np.nan_to_num(ic), width, yerr=yerr, capsize=3, label=h)
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.8)
    ax.set_xticks(x + width * (len(horizons) - 1) / 2)
    ax.set_xticklabels(signals, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("Pooled rank IC")
    ax.set_title("Signal ICs by horizon, with cluster-bootstrap intervals")
    ax.legend(title="Forward horizon", fontsize=6, title_fontsize=6)
    ax.tick_params(axis="y", labelsize=7)

    show_with_alt(
        fig,
        "A grouped bar chart with one group per signal and one bar per forward horizon "
        "inside it, each bar carrying a vertical interval line. The bars are small against "
        "the axis and fall on both sides of the zero line, and almost every interval line "
        "crosses zero, so few of the signal-horizon pairs are separated from it. The "
        "intervals are long relative to the bars they sit on.",
    )
    ic_summary = pl.DataFrame()

# %% [markdown]
# ## 6. Quintile Analysis
#
# Beyond IC, we examine whether signals produce economically meaningful return spreads
# by sorting stocks into quintiles based on each signal and comparing average returns.
# A strong signal should show a monotonic relationship between quintile rank and
# subsequent returns.


# %%
def quintile_analysis(df: pl.DataFrame, signal_col: str, return_col: str) -> pl.DataFrame:
    """Sort into quintiles by signal, compute average return per quintile."""
    valid = df.drop_nulls([signal_col, return_col])
    if valid.height < 25:  # Need at least 5 per quintile
        return pl.DataFrame()

    # Assign quintiles using qcut on the signal column
    result = valid.with_columns(
        quintile=pl.col(signal_col).qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    )

    # Average return per quintile
    summary = (
        result.group_by("quintile")
        .agg(
            avg_return=pl.col(return_col).mean(),
            std_return=pl.col(return_col).std(),
            n_obs=pl.col(return_col).len(),
        )
        .sort("quintile")
    )

    return summary


# %%
# Quintile analysis for key signals
key_signals = ["sentiment_mean", "narrative_change"]
fig, axes = plt.subplots(1, len(key_signals), figsize=FIGSIZE["dual_h_tall"])
if len(key_signals) == 1:
    axes = [axes]

for i, sig in enumerate(key_signals):
    ax = axes[i]
    q_df = quintile_analysis(eval_df, sig, "fwd_20d")

    if q_df.height > 0:
        quintiles = q_df["quintile"].to_list()
        returns = q_df["avg_return"].to_numpy()

        # One color for every bar: coloring by sign encodes the outcome twice, once in the
        # height and once in the hue, and makes a difference of a basis point either side of
        # zero look categorical.
        ax.bar(range(len(quintiles)), returns * 100, color=COLORS["blue"])
        ax.set_xticks(range(len(quintiles)))
        ax.set_xticklabels(quintiles, fontsize=7)
        ax.set_ylabel("Mean 20-day forward return, percent")
        ax.set_xlabel("Bucket, lowest signal at the left")
        ax.set_title(f"Forward return by {sig} bucket", fontsize=8)
        ax.axhline(0, color=COLORS["neutral"], linewidth=0.5)
        ax.tick_params(axis="y", labelsize=7)
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
        ax.set_title(f"Forward return by {sig} bucket", fontsize=8)

show_with_alt(
    fig,
    "Two panels, one per signal, each with five bars for the mean forward return of a bucket, "
    "ordered from the lowest signal values at the left to the highest at the right. Neither "
    "panel's bars rise or fall across the buckets: in the left panel the tallest bar is the "
    "leftmost, and in the right panel the tallest is the rightmost with the second tallest "
    "the leftmost, so in both the two extreme buckets are among the highest.",
)

# %% [markdown]
# The spread between the extreme buckets is not annotated on either panel. It is a difference
# between two pooled means with no interval attached, and the cluster bootstrap above is the
# inference this notebook actually supports - putting a bare number in the corner of a chart
# invites it to be read as a result when the chart beside it shows the ordering is not there.

# %% [markdown]
# ## 7. Save Signals
#
# Save the computed signals for potential downstream use.

# %%
# Save evaluation dataset
eval_df.write_parquet(OUTPUT_DIR / "filing_signals.parquet")
print(f"Saved {len(eval_df):,} signal observations to {OUTPUT_DIR / 'filing_signals.parquet'}")

# Save IC summary
if len(ic_summary) > 0:
    ic_summary.write_parquet(OUTPUT_DIR / "ic_summary.parquet")
    print(f"Saved IC summary to {OUTPUT_DIR / 'ic_summary.parquet'}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **SEC filings provide dense, structured text** with natural PIT anchoring via filing dates.
#    MD&A sections average ~9,500 words per quarter — far richer than news headlines.
#
# 2. **Chunking is essential for transformer models** that have 512-token limits.
#    Mean-pooled paragraph-level scores approximate full-document analysis.
#
# 3. **Two complementary signal types emerge from the same text**:
#    - *Sentiment* captures directional management tone (optimistic vs cautious)
#    - *Narrative change* captures information novelty (quarter-over-quarter semantic shift)
#
# 4. **Signal evaluation is screening-grade**. The pooled Spearman ICs above
#    are reported with cluster bootstrap (cluster=symbol) CIs and p-values
#    rather than the i.i.d. t-stat formula, because the pooled sample
#    contains multiple filings per firm and overlapping forward-return
#    windows. Treat the magnitudes as a screen; chapter-headline inference
#    uses HAC on per-date cross-sectional IC series (NB07, NB08), which
#    requires adequate breadth per date.
#
# 5. **Filing signals complement headline signals** (NB07). News captures market reaction
#    speed; filings capture management's own assessment of financial condition.
#
# **Next**: See NB07/08 for news-based signal construction and evaluation.
# **Book**: Section 10.5 discusses the full pre-train → adapt → fine-tune cascade
# and signal validation protocol for production deployment.

# %%
print("\n" + "=" * 70)
print("NOTEBOOK COMPLETE: SEC Filing Text Signals")
print("=" * 70)
print(f"""
Signals computed:
  - sentiment_mean: FinBERT paragraph-level sentiment (mean across chunks)
  - sentiment_std: Within-filing sentiment dispersion
  - sentiment_pos_pct / sentiment_neg_pct: Fraction of positive/negative chunks
  - narrative_change: Cosine distance to prior quarter's MD&A embedding

Evaluation dataset: {len(eval_df):,} filing-date observations
Symbols: {eval_df["symbol"].n_unique()}
Date range: {eval_df["filing_date"].min()} to {eval_df["filing_date"].max()}
Output: {OUTPUT_DIR}
""")
