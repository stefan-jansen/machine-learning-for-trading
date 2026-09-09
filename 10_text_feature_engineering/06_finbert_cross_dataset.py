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
# # A model scored against labels that mean something else
#
# **Chapter 10: text feature engineering**
# **Section reference**: Section 10.4
#
# **Docker image**: `ml4t-gpu`
#
# > **GPU recommended**: this runs FinBERT over roughly eight thousand headlines. A GPU takes
# > about a minute end to end; a CPU takes several times that. For GPU acceleration:
# > ```bash
# > docker compose run --rm ml4t-gpu python 10_text_feature_engineering/06_finbert_cross_dataset.py
# > ```
#
# ## What this notebook is for
#
# A model that scores well on its own test split and badly on someone else's data is the
# ordinary situation, and the useful skill is diagnosing which of several very different
# causes is responsible. This notebook works one case through to the end.
#
# `ProsusAI/finbert` is applied to FinMarBa, a corpus of financial headlines. Both carry
# three labels called negative, neutral and positive, both are financial text, and the
# accuracy that comes out is far below what the model reports on its own domain. The obvious
# reading is that the model does not transfer.
#
# The obvious reading is wrong, and the dataset says so in a column the notebook loads and
# then ignores. FinMarBa's label is not a judgment about the headline; it is the sign of what
# the mentioned tickers did afterwards. The two corpora agree on three label *names* and
# disagree about what the labels are *for*. So the number here measures how well a sentiment
# reading anticipates a price move, which is a different question from whether the model
# reads sentiment in a new domain - and a much harder one, taken up in
# `07_news_return_signals`.
#
# ## Learning objectives
#
# After working through this notebook you will be able to:
#
# - Read a dataset's label definition from how the labels were produced, rather than from
#   what they are called, and say when two datasets sharing label names share a task.
# - Compare an accuracy against the majority-class rate before reading it as skill.
# - Separate three causes of a cross-dataset drop that are routinely conflated: the text
#   looks different, the labeling standard differs, or the target is a different quantity.
# - Say what a confusion matrix shows about which way a model's errors run when its notion of
#   the classes does not match the data's.
#
# ## Prerequisites
#
# - Section 10.4 of the chapter.
# - A Hugging Face `datasets` cache able to fetch `baptle/financial_headlines_market_based`,
#   which downloads on first run.
#
# ## Related notebooks
#
# - `04_bert_finetuning.py` - fine-tuning the same checkpoint on its own domain
# - `07_news_return_signals.py` - measuring what news sentiment is worth against returns

# %%
"""Score FinBERT against market-derived labels and diagnose what the gap measures."""

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import pipeline

from utils.paths import get_chapter_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, show_with_alt

# The tokenizer's Rust parallelism forks after this process has already used threads, which
# it warns about on every batch. One inference pass over eight thousand short headlines does
# not need it.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %% tags=["parameters"]
SEED = 42
SAMPLE_SIZE = 10000

# %%
# Reproducibility - set_global_seeds covers Python random / NumPy / Torch.
set_global_seeds(SEED)

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")


# %% [markdown]
# ## The corpus, and where its labels come from
#
# FinMarBa pairs financial headlines with the subsequent percentage move of the tickers each
# one mentions. Its `Global Sentiment` column, which this notebook uses as the label, is the
# sign of that move aggregated across the tickers - so a headline is "positive" when the
# things it named went up, whatever the headline says.
#
# That is a different labeling standard from Financial PhraseBank, where annotators read each
# sentence and judged what it expressed. Both produce three classes with the same names. Only
# one of them is a statement about the text.
#
# The upstream dataset has been resized since first release, so the loader asks for up to
# `SAMPLE_SIZE` rows and takes what is available.


# %%
def load_finmarba_dataset(sample_size: int = 10000) -> pl.DataFrame:
    """Load FinMarBa dataset from Hugging Face.

    Args:
        sample_size: Number of samples to use (dataset has 60K+ samples).
            Set to 0 for all samples.

    Raises:
        RuntimeError: If dataset cannot be loaded (network error, etc.)
    """
    try:
        # Try 'test' split first, fall back to 'train' (dataset structure changed)
        try:
            ds = load_dataset(
                "baptle/financial_headlines_market_based",
                split="test",
            )
            print("Loaded FinMarBa test split")
        except ValueError:
            # Dataset may only have 'train' split now
            split_str = f"train[:{sample_size}]" if sample_size > 0 else "train"
            ds = load_dataset(
                "baptle/financial_headlines_market_based",
                split=split_str,
            )
            print(f"Loaded FinMarBa train split (sampled {sample_size:,} samples)")

        df = pl.DataFrame(ds.to_pandas())

        # Handle schema changes: newer versions have different column names
        # Map 'Title' -> 'text' and 'Global Sentiment' -> 'label'
        if "Title" in df.columns and "text" not in df.columns:
            df = df.rename({"Title": "text"})
        if "Global Sentiment" in df.columns and "label" not in df.columns:
            # Global Sentiment is -1/0/1, map to 0/1/2 for consistency
            # -1 (negative) -> 0, 0 (neutral) -> 1, 1 (positive) -> 2
            df = df.with_columns((pl.col("Global Sentiment") + 1).cast(pl.Int64).alias("label"))

        print(f"Loaded FinMarBa dataset: {len(df):,} samples")
        return df
    except Exception as e:
        raise RuntimeError(
            f"\n"
            f"{'=' * 70}\n"
            f"DATASET NOT AVAILABLE: FinMarBa\n"
            f"{'=' * 70}\n"
            f"\n"
            f"Error: {e}\n"
            f"\n"
            f"This notebook requires the FinMarBa dataset from Hugging Face:\n"
            f"  baptle/financial_headlines_market_based\n"
            f"\n"
            f"Possible causes:\n"
            f"  - No internet connection\n"
            f"  - HuggingFace servers unavailable\n"
            f"  - Dataset has been moved or renamed\n"
            f"\n"
            f"To run this notebook, ensure you have internet access.\n"
            f"{'=' * 70}\n"
        ) from e


df = load_finmarba_dataset(sample_size=SAMPLE_SIZE)
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

print("Label distribution:")
print(df.group_by("label").len().sort("label"))

majority_rate = df.group_by("label").len()["len"].max() / len(df)
print(f"Majority-class rate: {majority_rate:.1%}")

# %% [markdown]
# The majority rate is the number every accuracy below has to be read against. A classifier
# that ignores the headline and always answers with the most common label scores that much,
# so it is the zero point for skill on this data rather than the 33 percent that three
# balanced classes would give.

# %% [markdown]
# ### What the label is made of
#
# The dataset ships the move it derived each label from. Printing a few rows beside their
# labels is the whole argument of this notebook, and it takes one cell.

# %%
if "Pct_Change" in df.columns:
    evidence = df.select(["text", "Pct_Change", "label"]).head(4)
    for row in evidence.iter_rows(named=True):
        print(f"{row['text'][:70]}")
        print(f"    moves: {row['Pct_Change']}")
        print(f"    label: {LABEL_MAP[row['label']]}\n")
else:
    print("This copy of the dataset does not carry Pct_Change; the label is still its sign.")

# %% [markdown]
# Read the second row if it is the one about the dollar slumping: a headline whose sentiment
# any reader would call negative, labeled by two tickers that moved in opposite directions.
# The label is arithmetic on those moves. Nothing about it is a claim about the sentence.

# %% [markdown]
# ## The model
#
# `ProsusAI/finbert` is a BERT fine-tuned on Financial PhraseBank for sentiment. It is
# applied here with no further training, which is what makes the comparison clean: whatever
# it produces is what it learned from annotator judgments, evaluated against price moves.

# %%
model_name = "ProsusAI/finbert"
print(f"Loading {model_name}...")

classifier = pipeline(
    "sentiment-analysis",
    model=model_name,
    tokenizer=model_name,
    device=device,
    truncation=True,
    max_length=512,
)

# ProsusAI/finbert label mapping (lowercase labels)
FINBERT_LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


# %%
def get_finbert_predictions(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Get predictions from FinBERT."""
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = classifier(batch)
        for r in results:
            predictions.append(FINBERT_LABEL_MAP[r["label"]])
    return np.array(predictions)


# %% [markdown]
# ## 4. Evaluate FinBERT on FinMarBa

# %%
# Get the text column (may be 'headline' or 'text' depending on dataset)
text_col = "headline" if "headline" in df.columns else "text"
texts = df[text_col].to_list()
true_labels = df["label"].to_numpy()

print(f"Running FinBERT on {len(texts)} samples...")
predictions = get_finbert_predictions(texts)

# Compute metrics
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average="macro")

print(f"Accuracy: {accuracy:.1%}  (majority-class rate: {majority_rate:.1%})")
print(f"F1 (macro): {f1:.3f}")

# %% [markdown]
# ## What that number is and is not
#
# The accuracy sits a little above the majority-class rate. Published figures for this
# checkpoint on its own held-out PhraseBank split are far higher; this notebook does not
# reproduce that measurement, so treat it as context from the literature rather than as an
# in-notebook comparison.
#
# There are three different things a drop like this can mean, and they call for different
# responses:
#
# 1. **The text is different.** Headlines are shorter and blunter than the analyst sentences
#    the model was trained on. Real, and the fix is more training data from the new domain.
# 2. **The labeling standard is different.** Two annotators can disagree about what counts as
#    neutral. Real, and the fix is a shared annotation guide or a calibration step.
# 3. **The target is a different quantity.** The labels answer another question entirely, and
#    no amount of adaptation on the text side closes the gap because the model is not being
#    asked what it was built to answer.
#
# The third is what is happening here, and the `Pct_Change` column above is the evidence. A
# model that read the sentiment of every headline perfectly would still be wrong whenever a
# gloomy headline preceded a rally. What this notebook measures is closer to the predictive
# value of news sentiment for returns, which is `07_news_return_signals`' subject and which
# nobody expects to be high.

# %%
print(f"FinMarBa, zero-shot: accuracy {accuracy:.1%}, macro F1 {f1:.3f}, n={len(true_labels)}")
print(f"Always answering the majority class: accuracy {majority_rate:.1%}")

# %% [markdown]
# The matrix says which way the disagreements run, which a single accuracy cannot. The row is
# what the market did after the headline and the column is the model's reading of it, so each
# off-diagonal cell counts one kind of disagreement and the grid shows whether they are
# spread evenly or concentrated in particular classes.
#
# It cannot say why they disagree. The same counts would arise from a model that is unsure
# and from one answering a different question, and nothing in a table of counts separates
# those. What settles it here is the label definition printed earlier, not this figure.
#
# One column is worth reading closely. The model predicts neutral far less often than the
# labels call for it, and that is the two definitions of "neutral" coming apart. A market
# neutral is a small move, which is common. A sentiment neutral is a sentence expressing no
# view, which a headline written to be read almost never is. Two classes with one name, doing
# different jobs, in the same three-way problem.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
labels = ["negative", "neutral", "positive"]

cm = confusion_matrix(true_labels, predictions)
ax.imshow(cm, cmap="Blues")

# Light text on the dark high-count cells; the threshold is half the largest count.
threshold = cm.max() / 2.0
for i in range(3):
    for j in range(3):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=8,
            color=COLORS["silver"] if cm[i, j] > threshold else COLORS["blue"],
        )

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted by FinBERT from the headline")
ax.set_ylabel("Labeled by the subsequent move")
ax.set_title("Headline sentiment against the direction the market moved")

show_with_alt(
    fig,
    "A three-by-three grid of counts, with the label derived from the market move down the "
    "side and FinBERT's reading of the headline across the bottom. The counts are spread "
    "widely rather than concentrated on the diagonal, and each row puts substantial weight in "
    "more than one column. The middle column, where the model predicts neutral, is much "
    "lighter than the two beside it in all three rows, so the model rarely calls a headline "
    "neutral even on the row whose label says the market barely moved.",
)

# %% [markdown]
# ## Key takeaways
#
# 1. **Read a label's definition from how it was produced.** Two datasets can agree on three
#    class names and disagree about what the classes are for. Here one was produced by
#    annotators reading sentences and the other by taking the sign of a price move, and the
#    column that produced it ships with the data.
# 2. **Compare an accuracy to the majority-class rate before calling it skill.** On three
#    imbalanced classes the zero point is not a third; it is whatever always answering the
#    most common label would score.
# 3. **A cross-dataset drop has at least three causes and they need different responses.**
#    Different text, a different labeling standard, or a different target quantity. Only the
#    first two are addressed by adapting the model to the new domain; the third means the
#    question changed and no amount of adaptation is the answer.
# 4. **Naming the cause changes what you would do next.** Read as domain shift, this result
#    argues for fine-tuning on headlines. Read correctly, it argues for deciding whether you
#    want a model of what text says or a model of what prices do next, because they are
#    different models and only one of them is trained here.
# 5. **A confusion matrix shows where the disagreement sits, not what causes it.** It is
#    worth reading for which classes agree and which do not, and it cannot tell you whether a
#    model is unsure or is answering a different question. Only the labels' provenance does
#    that, and measuring this model's sentiment performance would need sentiment annotations
#    on these headlines, which no one has made.

# %%
output_dir = get_chapter_dir(10) / "output" / "finbert_cross_dataset"
output_dir.mkdir(parents=True, exist_ok=True)
results_file = output_dir / "results.md"
with open(results_file, "w") as f:
    f.write("# FinBERT scored against market-derived labels\n\n")
    f.write(f"- Headlines: {len(df):,}\n")
    f.write(f"- Accuracy: {accuracy:.1%}\n")
    f.write(f"- Macro F1: {f1:.3f}\n")
    f.write(f"- Majority-class rate: {majority_rate:.1%}\n\n")
    f.write("## What this measures\n\n")
    f.write(
        "FinMarBa's label is the sign of the subsequent move of the tickers a headline\n"
        "names, not a judgment about the headline. So this is the agreement between a\n"
        "sentiment reading and a price move, and not a measurement of how well the model\n"
        "reads sentiment in a new text domain.\n"
    )

# The path is printed relative to the repository so the render does not carry the absolute
# path of whichever checkout produced it.
print(f"Results saved to: {results_file.relative_to(get_chapter_dir(10).parent)}")
