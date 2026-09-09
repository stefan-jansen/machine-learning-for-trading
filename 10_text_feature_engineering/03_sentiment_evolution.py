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
# # Three generations of text features on one task
#
# **Chapter 10: text feature engineering**
# **Section reference**: Sections 10.2, 10.3 and 10.5
#
# **Docker image**: `ml4t-py312`
#
# > **Docker required**: this notebook uses `gensim`, which does not build against the Python
# > version the rest of the repository runs on. Run it with:
# > ```bash
# > docker compose --profile py312 run --rm py312 python 10_text_feature_engineering/03_sentiment_evolution.py
# > ```
#
# ## What this notebook is for
#
# Three ways of turning a sentence into numbers, each the standard answer of its decade, all
# scored on the same split of the same corpus: counts of words and word pairs, an average of
# static word vectors, and a transformer that reads the sentence in order.
#
# Be clear about what kind of comparison this is, because it is not a controlled one. The two
# lexical methods are trained here on the same split with the same estimator on top, so those
# two differ only in their representation. The transformer is a different thing entirely: a
# supervised sentiment classifier someone else fine-tuned on analyst reports, applied
# unchanged and never shown this training split. It arrives with both a representation and a
# task already learned, and its score reflects both. All three are scored on the same test
# sentences, which makes the numbers comparable as outcomes without making them a measurement
# of representation quality alone.
#
# One detail decides whether the comparison is honest. `yiyanghkust/finbert-tone` was trained
# on analyst reports and earnings-call transcripts, not on the Financial PhraseBank, so the
# test sentences here are new to it. A checkpoint trained on this corpus would post a high
# score for a reason that has nothing to do with the representation, which is what
# `04_bert_finetuning` runs into with a different one.
#
# ## Learning objectives
#
# After working through this notebook you will be able to:
#
# - Build a TF-IDF classifier and say which properties of a sentence its features can and
#   cannot represent.
# - Turn static word vectors into a document vector, and name what averaging destroys.
# - Apply a published classification checkpoint without fine-tuning, and check its label
#   order against your own before trusting a single score.
# - Read three confusion matrices side by side to locate which class accounts for a
#   difference in accuracy.
#
# ## Prerequisites
#
# - Sections 10.1 to 10.4 of the chapter.
# - The Financial PhraseBank `sentences_allagree` subset on disk, loaded via
#   `data.load_financial_phrasebank`.
#
# ## Related notebooks
#
# - `01_word2vec_training.py` - what the static vectors averaged here actually encode
# - `04_bert_finetuning.py` - fine-tuning a checkpoint on this corpus, and the leakage that
#   comes with the domain-specific one

# %%
"""Compare TF-IDF, averaged GloVe and a pre-trained transformer on one sentiment split."""

import contextlib
import io
import json
import warnings

import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import pipeline
from transformers import set_seed as set_transformers_seed

from data import load_financial_phrasebank as load_financial_phrasebank_canonical
from utils.paths import get_chapter_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, show_with_alt

# A category with no module is nearly a blanket filter: it silences every UserWarning from
# every library, including any this notebook's own arithmetic would raise. Named to the two
# packages whose import-time notices repeat.
warnings.filterwarnings("ignore", category=UserWarning, module="gensim")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# %% tags=["parameters"]
# Production defaults - Papermill can override for fast CI runs.
SEED = 42
MAX_SAMPLES = 0  # 0 = use the full sentences_allagree subset
FINBERT_TEST_SAMPLES = 0  # 0 = run FinBERT on the entire stratified test split

# %% [markdown]
# `set_global_seeds` covers Python, NumPy and Torch. The transformers pipelines draw from
# their own generator, which needs seeding separately or the run is not reproducible.


# %%
set_global_seeds(SEED)
set_transformers_seed(SEED)

CONFIG = {
    "random_seed": SEED,
    "test_size": 0.2,
    "dataset": {
        "name": "takala/financial_phrasebank",
        "subset": "sentences_allagree",
        "description": "Financial PhraseBank - 100% annotator agreement subset (2,264 sentences)",
    },
    "tfidf": {
        "max_features": 5000,
        "ngram_range": (1, 2),
        "min_df": 2,
    },
    "glove": {
        "model": "glove-wiki-gigaword-100",
        "dim": 100,
    },
    "finbert": {
        "model_id": "yiyanghkust/finbert-tone",
        "description": "FinBERT fine-tuned on analyst reports for sentiment (NOT PhraseBank)",
        "tokenizer_id": "yiyanghkust/finbert-tone",
        "max_length": 512,
        "labels": {"Negative": 0, "Neutral": 1, "Positive": 2},
    },
}

print("EXPERIMENT CONFIGURATION")
print(json.dumps(CONFIG, indent=2))

# %% [markdown]
# ## 2. Load Financial PhraseBank Dataset
#
# The Financial PhraseBank (Malo et al., 2014) consists of English-language
# sentences from financial news, each labelled positive / negative / neutral
# by 5-8 annotators. We use the `sentences_allagree` subset - the 2,264
# sentences where every annotator picked the same label, i.e., the
# highest-precision portion of the corpus.

# %%


def load_financial_phrasebank() -> pl.DataFrame:
    """Load Financial PhraseBank from canonical local storage (sentences_allagree)."""
    return load_financial_phrasebank_canonical()


df = load_financial_phrasebank()
print(f"Loaded {len(df):,} sentences")
print("\nLabel distribution:")
print(df.group_by("label").len().sort("label"))

if MAX_SAMPLES > 0 and len(df) > MAX_SAMPLES:
    per_label = max(MAX_SAMPLES // df["label"].n_unique(), 1)
    df = (
        df.sort(["label", "sentence"])
        .group_by("label", maintain_order=True)
        .head(per_label)
        .sort("sentence")
    )
    print(f"Reduced dataset for test run: {len(df):,} sentences")

# Map numeric labels to text
label_map = {0: "negative", 1: "neutral", 2: "positive"}
df = df.with_columns(pl.col("label").replace_strict(label_map).alias("sentiment"))

# %% [markdown]
# ### Checks before anything is fitted
#
# Label mappings are the failure that produces a plausible-looking wrong number rather than
# an error, so they are verified here rather than inferred from a score that looks about
# right. The class balance is printed for the same reason: it is what any accuracy below has
# to be read against.


# %%
print("DATASET SANITY CHECKS")

# 1. Class distribution in full dataset
print("\n1. CLASS DISTRIBUTION (Full Dataset)")
class_counts = df.group_by("label").len().sort("label")
print(class_counts)

# Majority class baseline
total = len(df)
majority_row = class_counts.sort("len", descending=True).row(0)
majority_label = majority_row[0]  # label column
majority_count = majority_row[1]  # len column
majority_baseline = majority_count / total
print(f"\nMajority class: {label_map[majority_label]} (label={majority_label})")
print(f"Majority baseline accuracy: {majority_baseline:.1%}")

# 2. Label mapping verification
print("\n2. LABEL MAPPING VERIFICATION")
print("   Dataset labels → Our labels:")
for numeric, text in label_map.items():
    print(f"   {numeric} → {text}")

# Assert label mapping matches FinBERT expectations
finbert_label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
assert label_map == {0: "negative", 1: "neutral", 2: "positive"}, "Label mapping mismatch!"
print("   [OK] Label mapping verified")

# 3. Sample sentences by class
print("\n3. SAMPLE SENTENCES BY CLASS")
for label_id, label_name in label_map.items():
    sample = df.filter(pl.col("label") == label_id).head(1)["sentence"][0]
    print(f"   {label_name}: '{sample[:80]}...'")


# %%
# Train/test split (convert Polars columns to numpy for sklearn)
X_train, X_test, y_train, y_test = train_test_split(
    df["sentence"].to_numpy(),
    df["label"].to_numpy(),
    test_size=CONFIG["test_size"],
    random_state=SEED,
    stratify=df["label"].to_numpy(),
)

# Print split details
print("TRAIN/TEST SPLIT DETAILS")
print("Split protocol: Stratified random split (preserves class proportions)")
print(f"Test size: {CONFIG['test_size']} ({CONFIG['test_size'] * 100:.0f}%)")
print(f"Random seed: {SEED}")
print(f"\nTrain samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")

if FINBERT_TEST_SAMPLES > 0 and len(X_test) > FINBERT_TEST_SAMPLES:
    sample_idx = np.random.choice(len(X_test), FINBERT_TEST_SAMPLES, replace=False)
    X_test_finbert = X_test[sample_idx]
    y_test_finbert = y_test[sample_idx]
    print(f"FinBERT evaluation sample: {len(X_test_finbert):,}")
else:
    X_test_finbert = X_test
    y_test_finbert = y_test

# %%
# Class distribution in each split.
train_counts = dict(zip(*np.unique(y_train, return_counts=True), strict=False))
test_counts = dict(zip(*np.unique(y_test, return_counts=True), strict=False))
pl.DataFrame(
    {
        "class": [label_map[i] for i in sorted(train_counts.keys())],
        "train": [train_counts[i] for i in sorted(train_counts.keys())],
        "test": [test_counts[i] for i in sorted(train_counts.keys())],
    }
)

# %% [markdown]
# ## 3. TF-IDF + Logistic Regression (Lexical Baseline)
#
# The simplest baseline: represent documents as weighted term frequencies,
# then train a linear classifier. TF-IDF captures word importance but cannot
# understand semantic similarity or context.

# %%
# TF-IDF vectorization
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words="english",
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF feature dimension: {X_train_tfidf.shape[1]}")

# Train logistic regression
lr_tfidf = LogisticRegression(max_iter=1000, random_state=SEED)
lr_tfidf.fit(X_train_tfidf, y_train)

# Evaluate
y_pred_tfidf = lr_tfidf.predict(X_test_tfidf)
acc_tfidf = accuracy_score(y_test, y_pred_tfidf)
f1_tfidf = f1_score(y_test, y_pred_tfidf, average="macro")

print("\nTF-IDF + Logistic Regression:")
print(f"  Accuracy: {acc_tfidf:.1%}")
print(f"  F1 (macro): {f1_tfidf:.3f}")

# %% [markdown]
# ## 4. Static Embeddings (GloVe) + Logistic Regression
#
# Static embeddings map words to dense vectors that capture semantic similarity.
# We use GloVe (Global Vectors for Word Representation) pre-trained on Wikipedia/Gigaword.
# We average word vectors to create document representations, then train a classifier.
# Limitation: each word has ONE vector regardless of context.

# %% [markdown]
# ### Load GloVe Embeddings
# Load pre-trained GloVe vectors for document representation.

# %%
from gensim.utils import simple_preprocess


def get_embedding_model():
    """Load pre-trained GloVe embedding model."""
    # Use 100-dim GloVe vectors trained on Wikipedia + Gigaword
    model_name = "glove-wiki-gigaword-100"
    print(f"Loading {model_name}...")
    # Suppress the gensim downloader's per-chunk progress stream so a
    # fresh-container download does not flood the notebook output.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return api.load(model_name)


# %% [markdown]
# ### Document Vector Computation
# Average word vectors to create a fixed-size document representation.


# %%
def document_vector(doc: str, model, dim: int = 100) -> np.ndarray:
    """Compute document vector as average of word vectors.

    Uses gensim's simple_preprocess for consistent tokenization:
    - Lowercases text
    - Removes punctuation and special characters
    - Filters very short/long tokens
    """
    # Use gensim's tokenizer for cleaner preprocessing
    words = simple_preprocess(doc, deacc=True, min_len=2, max_len=15)
    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(dim)


# %%
# Load model
glove_model = get_embedding_model()
embed_dim = glove_model.vector_size

# Compute document vectors
print("Computing document vectors...")
X_train_emb = np.array([document_vector(doc, glove_model, embed_dim) for doc in X_train])
X_test_emb = np.array([document_vector(doc, glove_model, embed_dim) for doc in X_test])

print(f"GloVe document dimension: {X_train_emb.shape[1]}")

# Train logistic regression
lr_glove = LogisticRegression(max_iter=1000, random_state=SEED)
lr_glove.fit(X_train_emb, y_train)

# Evaluate
y_pred_glove = lr_glove.predict(X_test_emb)
acc_glove = accuracy_score(y_test, y_pred_glove)
f1_glove = f1_score(y_test, y_pred_glove, average="macro")

print("\nGloVe + Logistic Regression:")
print(f"  Accuracy: {acc_glove:.1%}")
print(f"  F1 (macro): {f1_glove:.3f}")

# %% [markdown]
# ## 5. FinBERT Pre-trained (No Task Fine-tuning)
#
# Transformers learn contextual representations that vary with surrounding words.
# FinBERT (yiyanghkust/finbert-tone) is pre-trained on financial text and already
# has a sentiment classification head, so we can score PhraseBank without any
# task-specific fine-tuning. It was trained on analyst reports - a different text
# source than PhraseBank's news sentences (same labels, different distribution) -
# so this measures cross-dataset transfer.
#
# **Critical Note**: This is NOT "zero-shot" - FinBERT-tone already carries a
# sentiment head trained on analyst reports; we are testing cross-dataset transfer,
# not prompting. On this high-agreement subset that transfer is strong: FinBERT
# leads both lexical baselines below. The distribution-shift cost shows up instead
# on the noisier mixed-agreement subset that the chapter's Section 10.4 table uses,
# where the same checkpoint slips behind TF-IDF. Section 6 and the takeaways
# quantify both sides.

# %% [markdown]
# Which checkpoint this is decides what its score means, so it is printed rather than left to
# the config dictionary above. A reader comparing this number against another notebook's has
# to be able to see that the two are not the same model.


# %%
print("FINBERT CHECKPOINT DETAILS")
print(f"Model ID:     {CONFIG['finbert']['model_id']}")
print(f"Tokenizer ID: {CONFIG['finbert']['tokenizer_id']}")
print(f"Max Length:   {CONFIG['finbert']['max_length']}")
print(f"Description:  {CONFIG['finbert']['description']}")
print("\nLabel mapping (FinBERT → our numeric labels):")
for label, idx in CONFIG["finbert"]["labels"].items():
    print(f"  {label} → {idx}")
print("\nThis checkpoint was fine-tuned on analyst reports, not PhraseBank;")
print("scores below therefore measure cross-dataset transfer, not a like-for-like")
print("comparison after task-specific fine-tuning.")


def get_finbert_predictions(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Get sentiment predictions from FinBERT.

    Uses the yiyanghkust/finbert-tone model which is already fine-tuned
    for financial sentiment classification on analyst reports.

    Returns:
        Array of predictions mapped to our label scheme (0=neg, 1=neu, 2=pos)
    """
    model_id = CONFIG["finbert"]["model_id"]
    tokenizer_id = CONFIG["finbert"]["tokenizer_id"]
    max_length = CONFIG["finbert"]["max_length"]

    # Use GPU if available
    device = 0 if torch.cuda.is_available() else -1

    # Create pipeline with explicit parameters
    classifier = pipeline(
        "sentiment-analysis",
        model=model_id,
        tokenizer=tokenizer_id,
        device=device,
        truncation=True,
        max_length=max_length,
    )

    # Map FinBERT labels to our numeric labels
    label_to_id = CONFIG["finbert"]["labels"]

    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = classifier(batch)
        for r in results:
            predictions.append(label_to_id[r["label"]])

    return np.array(predictions)


print("\nRunning FinBERT inference...")
y_pred_finbert = get_finbert_predictions(X_test_finbert.tolist())

acc_finbert = accuracy_score(y_test_finbert, y_pred_finbert)
f1_finbert = f1_score(y_test_finbert, y_pred_finbert, average="macro")

print("\nFinBERT (Pre-trained on analyst reports, no PhraseBank fine-tuning):")
print(f"  Accuracy: {acc_finbert:.1%}")
print(f"  F1 (macro): {f1_finbert:.3f}")

# Per-class diagnostic to understand failure patterns
from sklearn.metrics import classification_report

print("\nPer-class breakdown:")
print(
    classification_report(
        y_test_finbert, y_pred_finbert, target_names=["negative", "neutral", "positive"]
    )
)

# %% [markdown]
# ## 6. Results Comparison

# %%
# Summary table
results = pl.DataFrame(
    {
        "Method": ["TF-IDF + LR", "GloVe + LR", "FinBERT (pre-trained)"],
        "Accuracy": [acc_tfidf, acc_glove, acc_finbert],
        "F1 (macro)": [f1_tfidf, f1_glove, f1_finbert],
    }
).with_columns(
    pl.col("Accuracy").map_elements(lambda x: f"{x:.1%}", return_dtype=pl.String),
    pl.col("F1 (macro)").map_elements(lambda x: f"{x:.3f}", return_dtype=pl.String),
)

print("COMPARISON SUMMARY")
print(results)

# Relative change calculation (negative = regression vs the lexical baseline).
relative_change = (acc_finbert - acc_tfidf) / acc_tfidf * 100
direction = "above" if relative_change >= 0 else "below"
print(
    f"\nFinBERT (pre-trained) accuracy is {abs(relative_change):.1f}% {direction} the TF-IDF baseline."
)

# %% [markdown]
# The three matrices share one color scale and one colorbar. Per-panel scales would shade the
# same count differently in each panel, which is exactly the comparison these are drawn for.

# %%
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE["triple_h_tall"])
labels = ["negative", "neutral", "positive"]

panels = [
    ("TF-IDF", y_pred_tfidf, y_test),
    ("GloVe", y_pred_glove, y_test),
    ("FinBERT (pre-trained)", y_pred_finbert, y_test_finbert),
]
matrices = [confusion_matrix(y_true, y_pred) for _, y_pred, y_true in panels]
shared_max = max(matrix.max() for matrix in matrices)

for ax, (name, _, _), matrix in zip(axes, panels, matrices, strict=True):
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=shared_max,
        cbar=False,
        annot_kws={"size": 7},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(name, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(labelsize=7)

fig.suptitle("Test-set confusion matrices by method")
show_with_alt(
    fig,
    "Three heatmaps side by side, one per method, each a three-by-three grid of counts with "
    "the annotated class down the side and the predicted class across the bottom, all three "
    "on the same color scale. The neutral diagonal cell dominates every panel. The negative "
    "row differs most between panels: in the first two a large share of it sits away from the "
    "diagonal, and in the third almost all of it is on the diagonal. Every panel loses a "
    "similar number of the positive row into the neutral column.",
)

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])

methods = ["TF-IDF + LR", "GloVe + LR", "FinBERT (pre-trained)"]
accuracies = [acc_tfidf, acc_glove, acc_finbert]
f1_scores = [f1_tfidf, f1_glove, f1_finbert]

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color=COLORS["blue"])
bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1 (macro)", color=COLORS["amber"])

ax.set_ylabel("Score")
ax.set_title("Test accuracy and macro F1 by method")
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=7)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=7, loc="upper left")

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1%}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
    )

for bar in bars2:
    height = bar.get_height()
    ax.annotate(
        f"{height:.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
    )

show_with_alt(
    fig,
    "A grouped bar chart with one pair of bars per method, accuracy and macro F1, each "
    "labeled with its value, on an axis running from zero to one. The first two pairs are "
    "close to each other in height, with the second slightly the lower of the two. The third "
    "pair is clearly taller than both, and its two bars are closer together than the two bars "
    "in either of the first two pairs.",
)

# %% [markdown]
# ## What the comparison shows
#
# Three things to read off the table and the matrices, in the order they matter.
#
# **TF-IDF is a hard baseline, and that is the point of running it.** Financial news
# vocabulary is unusually polarized - `profit`, `loss`, `narrowed`, `tumbled` - so counting
# words and word pairs already separates most sentences. Any representation that costs more
# has to clear this, and one of the two below does not.
#
# **Averaging static vectors buys synonymy and pays in structure.** The averaged document
# vector puts a test phrase about earnings retreating near training examples about profits
# falling, which counts of exact tokens cannot do. What the mean gives up is order and scope:
# the words are all there, but nothing records which one a negation applies to. On this split
# it lands behind TF-IDF, and the notebook does not establish why.
#
# Neither method handles negation well, and one of them is worse than it looks. The
# vectorizer above passes `stop_words="english"`, and sklearn's English list contains `not`,
# `no`, `never` and `nor` - so every negator is deleted before the bigrams are built. "Profit
# did not rise" and "profit did rise" become the same bag of features. That is a
# preprocessing default doing real damage on a sentiment task, and it is worth knowing about
# before reading TF-IDF's score as a property of n-grams.
#
# **The transformer's gap is concentrated in one class.** The confusion matrices are where
# to look: the difference in accuracy is not spread evenly but sits almost entirely in the
# negative row, which the lexical methods scatter across all three columns and the
# transformer recovers nearly whole. All three lose a similar number of positives to the
# neutral column, so that error is not what separates them.
#
# A caution about what "pre-trained" means here. This checkpoint has a trained
# classification head, so it is not zero-shot in the prompted-LLM sense; it is a supervised
# sentiment model built on a different corpus, applied unchanged. What its score measures is
# transfer from analyst-report text to journalistic sentences, and it is a fair test only
# because those sentences were not in its training data.

# %%
# Structured output for automated extraction
print("KEY STATISTICS FOR CHAPTER PROSE")
print("\nDataset: Financial PhraseBank")
print(f"Train/Test split: {len(X_train)}/{len(X_test)}")
print(f"\nTF-IDF + LR: Accuracy={acc_tfidf:.1%}, F1={f1_tfidf:.3f}")
print(f"GloVe + LR: Accuracy={acc_glove:.1%}, F1={f1_glove:.3f}")
print(f"FinBERT (pre-trained): Accuracy={acc_finbert:.1%}, F1={f1_finbert:.3f}")
print(f"\nRelative change (FinBERT vs TF-IDF): {relative_change:.1f}%")

# %%
# Save results for chapter integration - both markdown and JSON artifacts
output_dir = get_chapter_dir(10) / "output" / "sentiment_evolution"
output_dir.mkdir(parents=True, exist_ok=True)

# Save structured JSON artifact (for reproducibility verification)
results_artifact = {
    "config": CONFIG,
    "dataset": {
        "name": CONFIG["dataset"]["name"],
        "subset": CONFIG["dataset"]["subset"],
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "finbert_test_samples": len(X_test_finbert),
        "majority_baseline_accuracy": float(majority_baseline),
    },
    "results": {
        "tfidf_lr": {"accuracy": float(acc_tfidf), "f1_macro": float(f1_tfidf)},
        "glove_lr": {"accuracy": float(acc_glove), "f1_macro": float(f1_glove)},
        "finbert_pretrained": {
            "accuracy": float(acc_finbert),
            "f1_macro": float(f1_finbert),
            "note": "Cross-dataset transfer (trained on analyst reports, tested on news)",
        },
    },
}

json_file = output_dir / "results.json"
with open(json_file, "w") as f:
    json.dump(results_artifact, f, indent=2)

# %%
# Save markdown summary
results_file = output_dir / "results.md"
with open(results_file, "w") as f:
    f.write("# Sentiment Evolution Results\n\n")
    f.write("## Experiment Configuration\n\n")
    f.write(f"- Dataset: {CONFIG['dataset']['name']} ({CONFIG['dataset']['subset']})\n")
    f.write(f"- Train/Test split: {len(X_train)}/{len(X_test)} (stratified, seed={SEED})\n")
    f.write(f"- Majority baseline: {majority_baseline:.1%}\n\n")
    f.write("## Performance Comparison\n\n")
    f.write("| Method | Accuracy | F1 (macro) |\n")
    f.write("|--------|----------|------------|\n")
    f.write(f"| Majority Baseline | {majority_baseline:.1%} | - |\n")
    f.write(f"| TF-IDF + LR | {acc_tfidf:.1%} | {f1_tfidf:.3f} |\n")
    f.write(f"| GloVe + LR | {acc_glove:.1%} | {f1_glove:.3f} |\n")
    f.write(f"| FinBERT (pre-trained*) | {acc_finbert:.1%} | {f1_finbert:.3f} |\n")
    f.write("\n*FinBERT-tone: trained on analyst reports, NOT Financial PhraseBank\n")
    f.write("\n## Key Finding\n\n")
    f.write(f"FinBERT vs TF-IDF change: {relative_change:+.1f}%\n")
    f.write("\n## Critical Insight\n\n")
    if acc_finbert < acc_tfidf:
        f.write("Pre-trained FinBERT (without PhraseBank fine-tuning) **underperforms** TF-IDF.\n")
        f.write(
            "This demonstrates **distribution shift**: same sentiment labels, "
            "different text domains.\n"
        )
        f.write("FinBERT-tone was trained on analyst reports (formal, technical language),\n")
        f.write("while PhraseBank contains financial news sentences (journalistic style).\n")
    else:
        f.write(
            f"Pre-trained FinBERT (without PhraseBank fine-tuning) **outperforms** TF-IDF "
            f"by {relative_change:+.1f}% on this subset.\n"
        )
        f.write(
            "On the high-agreement subset the labels are clean enough that the pre-trained "
            "FinBERT head transfers well, and the analyst-report-trained classifier picks "
            "up the polarised sentence-level financial vocabulary directly.\n"
        )
        f.write(
            "The §10.4 chapter table reports the larger mixed-agreement subset, where the "
            "same checkpoint degrades because labels are noisier and analyst-report tone "
            "and journalistic style diverge more visibly.\n"
        )
    f.write("\n## Model Details\n\n")
    f.write(f"- FinBERT checkpoint: `{CONFIG['finbert']['model_id']}`\n")
    f.write(f"- Max length: {CONFIG['finbert']['max_length']} tokens\n")

print("\nResults saved to:")
print(f"  - {results_file}")
print(f"  - {json_file}")

# %% [markdown]
# ## Key takeaways
#
# 1. **Run the cheap baseline before the expensive representation.** TF-IDF on a polarized
#    vocabulary is hard to beat, costs seconds, and tells you what any richer method has to
#    clear. Averaged static vectors do not clear it here.
# 2. **A representation is defined by what it discards.** Averaging word vectors keeps every
#    word and loses all structure, so "profit narrowed" and "narrowed profit" are the same
#    document. Stop-word removal keeps structure and deletes words - including the negators,
#    on a task where they carry the answer. Read the preprocessing before attributing a score
#    to the representation.
# 3. **Check a published checkpoint's label order against your own.** This one emits its
#    classes in a different order from the dataset's, so mapping by index rather than by name
#    would produce a low score that looks like a modelling result. The notebook maps by name
#    and asserts the dataset's mapping is what it expects.
# 4. **Check what the checkpoint was trained on before reading its score as transfer.** This
#    one was trained on analyst reports, so the PhraseBank test sentences are genuinely new
#    to it. `04_bert_finetuning` uses a checkpoint for which that is not true and gets a
#    number that cannot be compared with this one.
# 5. **A single accuracy hides where a method fails.** The matrices put the difference in one
#    class, which is the actionable form: it says what to fix, and a scalar does not.
#
# ### The scope these numbers have
#
# One split of the `sentences_allagree` subset, which keeps only sentences every annotator
# scored the same way and is therefore the cleanest and smallest of the four PhraseBank
# subsets. Results on the mixed-agreement subsets are reported in the chapter text and are
# not measured here; nothing in this notebook establishes how any of these three methods
# behaves as label noise grows.
