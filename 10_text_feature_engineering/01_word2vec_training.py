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
# # Word2Vec: what a word's neighbors can and cannot tell you
#
# **Chapter 10: Text feature engineering**
# **Section reference**: Section 10.2, on the distributional hypothesis and Word2Vec
#
# **Docker image**: `ml4t-py312`
#
# > **Docker required**: this notebook uses `gensim`, which does not build against the
# > Python version the rest of the repository runs on. Run it with:
# > ```bash
# > docker compose --profile py312 run --rm py312 python 10_text_feature_engineering/01_word2vec_training.py
# > ```
#
# ## What this notebook is for
#
# Word2Vec turns one idea into a number: a word is described by the words it appears next
# to. The idea is worth understanding twice over, because Chapter 10 applies exactly the
# same machinery to portfolios, where a stock is described by the stocks held alongside it.
#
# This notebook trains the model on a small corpus of financial sentences and then asks
# what the resulting vectors know. The answer is more interesting than a demonstration
# that it works: co-occurrence puts words used in the same sentences close together, and
# the words a financial writer uses in the same sentence as `profit` include `loss`. What
# the model learns is topic, and polarity is not topic.
#
# ## Learning objectives
#
# After working through this notebook you will be able to:
#
# - Train a Word2Vec model on your own sentences and say what each of its five main
#   settings changes about the result.
# - Read a nearest-neighbor table as evidence about what the model measures, rather than
#   as a list of synonyms.
# - Show why a model built from co-occurrence places opposites next to each other, and
#   name the cases where that makes it the wrong feature.
# - Compare a model trained on a few thousand domain sentences against one trained on
#   billions of general ones, and say what each has that the other does not.
#
# ## Prerequisites
#
# - Section 10.2 of the chapter.
# - A vocabulary is the set of distinct tokens a model keeps; a context window is the
#   number of words on each side that count as a word's neighbors. Nothing else is assumed.
#
# ## Related notebooks
#
# - `02_asset_embeddings.py` - the same Skip-gram machinery applied to 13F portfolios
# - `03_sentiment_evolution.py` - compares static embeddings to TF-IDF and Transformers

# %%
"""Train Word2Vec on financial sentences and read what the vectors encode."""

import contextlib
import io
import json
import re
import warnings
from collections import Counter

import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

from data import load_financial_phrasebank as load_financial_phrasebank_canonical
from utils.paths import get_chapter_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, show_with_alt

# gensim builds its Cython extensions against an older NumPy ABI and imports `scipy.linalg`
# through a path scipy has deprecated. Both are import-time and neither can report a problem
# with this notebook's own arithmetic. Everything else stays visible.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gensim")
warnings.filterwarnings("ignore", category=UserWarning, module="gensim")

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides for CI
SEED = 42

# %%
# Reproducibility - single source of seeds for Python random, NumPy, and (if installed) Torch.
set_global_seeds(SEED)

CONFIG = {
    "random_seed": SEED,
    "word2vec": {
        "vector_size": 100,
        "window": 5,
        "min_count": 3,
        "sg": 1,  # 1 = Skip-gram, 0 = CBOW
        "negative": 10,  # Negative sampling
        "workers": 1,  # Use 1 for reproducibility (workers > 1 is non-deterministic)
        "epochs": 20,
    },
    "dataset": {
        "name": "takala/financial_phrasebank",
        "subset": "sentences_allagree",
    },
    "pretrained_comparison": "glove-wiki-gigaword-100",
}

print(json.dumps(CONFIG, indent=2))


# %% [markdown]
# ## The one assumption
#
# "You shall know a word by the company it keeps," in J. R. Firth's 1957 phrasing. Word2Vec
# turns that into an optimization: two words that appear beside the same other words are
# pushed toward the same vector. There are two ways to set up the prediction and they differ
# only in direction. **Skip-gram** is given a word and predicts its neighbors; **CBOW** is
# given the neighbors and predicts the word.
#
# The assumption is doing more work than it looks like. It defines similar to mean *used in
# the same places*, and that is not the same as *means the same thing*. Where a corpus uses
# two opposite words in the same sentence frames, the model has been told they are similar,
# and it will say so. The rest of this notebook is largely about that gap.

# %% [markdown]
# ## The corpus
#
# The Financial PhraseBank is a few thousand sentences from financial news, each labeled for
# sentiment by human annotators. The `sentences_allagree` subset keeps only the sentences
# every annotator scored the same way, which is the cleanest and the smallest of the four.
# Its size is the point of the comparison later on: this is a domain corpus, not a large one.


# %%
# Load Financial PhraseBank
def load_financial_phrasebank() -> pl.DataFrame:
    """Load Financial PhraseBank dataset from canonical local storage."""
    return load_financial_phrasebank_canonical()


print("Loading Financial PhraseBank...")
df = load_financial_phrasebank()
print(f"Loaded {len(df):,} sentences")
df.head()


# %% [markdown]
# Tokenization here is deliberately the simplest thing that works: lowercase, drop
# punctuation, split on whitespace, and discard single characters. Note what it does *not*
# do, because it shapes every result below: it keeps numbers. A corpus of earnings sentences
# is full of them, and they will compete with words for the model's attention.


# %%
def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace, drop single characters."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


# Tokenize all sentences
sentences = [tokenize(s) for s in df["sentence"].to_list()]
print(f"Total sentences: {len(sentences):,}")
print(f"Total tokens: {sum(len(s) for s in sentences):,}")
print("\nSample tokenized sentence:")
print(sentences[0])

# %% [markdown]
# The fifteen most frequent tokens, before `min_count` removes anything. Function words lead,
# as they do in any corpus. What is worth noticing is how high `eur` and `mn` sit: currency
# and magnitude markers are among the most common tokens in this corpus, and the numbers they
# introduce are in the vocabulary too.


# %%
all_tokens = [t for s in sentences for t in s]
token_counts = Counter(all_tokens)
print(f"Distinct tokens before min_count filter: {len(token_counts):,}")

vocab_freq = pl.DataFrame(
    {
        "token": [t for t, _ in token_counts.most_common(15)],
        "count": [c for _, c in token_counts.most_common(15)],
    }
)
vocab_freq

# %% [markdown]
# ## Training
#
# Five settings decide what comes out, and each answers a different question:
#
# - `vector_size` is how many numbers describe a word. More capacity needs more text to fill
#   it; 100 to 300 is the usual range and this corpus does not justify the top of it.
# - `window` is how many words on each side count as neighbors. A small window learns which
#   words are interchangeable in a phrase; a large one learns which words share a topic.
# - `min_count` drops words seen fewer than this many times, because a handful of occurrences
#   cannot locate a vector.
# - `sg` picks the direction of the prediction: 1 for Skip-gram, 0 for CBOW.
# - `negative` is how many random words each update pushes *away*, which is what keeps the
#   optimization from collapsing every vector onto one point.
#
# `workers=1` is not a performance choice. More than one worker means the OS decides the
# order updates are applied in, and the result stops being reproducible from the seed.


# %%
print("Training Word2Vec (Skip-gram)...")

w2v_config = CONFIG["word2vec"]
model = Word2Vec(
    sentences=sentences,
    vector_size=w2v_config["vector_size"],
    window=w2v_config["window"],
    min_count=w2v_config["min_count"],
    sg=w2v_config["sg"],
    negative=w2v_config["negative"],
    workers=w2v_config["workers"],
    epochs=w2v_config["epochs"],
    seed=CONFIG["random_seed"],
)

print(f"Vocabulary after min_count={w2v_config['min_count']}: {len(model.wv):,} tokens")
print(f"Embedding shape: ({len(model.wv)}, {model.wv.vector_size})")


# %% [markdown]
# ## What the neighbors actually are
#
# The most direct question to ask a set of embeddings is which words each one sits next to.
# Read the table below as evidence about what the model measures, not as a thesaurus.


# %%
# Build a similarity table for a fixed set of financial probe words.
def similar_words_frame(words: list[str], top_n: int = 5) -> pl.DataFrame:
    """Return top-n similar tokens (and cosine similarity) for each probe word."""
    rows = []
    for word in words:
        if word not in model.wv:
            rows.append({"probe": word, "rank": 0, "neighbor": "<OOV>", "similarity": float("nan")})
            continue
        for rank, (neighbor, score) in enumerate(model.wv.most_similar(word, topn=top_n), start=1):
            rows.append({"probe": word, "rank": rank, "neighbor": neighbor, "similarity": score})
    return pl.DataFrame(rows)


probe_words = ["profit", "loss", "revenue", "growth", "shares", "market"]
similar_words_frame(probe_words, top_n=5).pivot(values="neighbor", index="rank", on="probe")

# %% [markdown]
# Two things in that table are the notebook's whole argument.
#
# **The nearest neighbor of `profit` is `loss`, and the nearest neighbor of `loss` is
# `profit`.** They are not similar in meaning; they are maximally opposite. But a sentence
# reporting one is written almost identically to a sentence reporting the other - "operating
# profit rose to EUR ... mn" against "operating loss narrowed to EUR ... mn" - so their
# neighbors are the same words and co-occurrence has no way to tell them apart. This is not
# a defect in the training run; it is what the distributional hypothesis says, applied
# honestly.
#
# **Several neighbors are numbers or currency fragments.** The tokenizer kept digits and
# `eur4`, `eur7` and bare figures are frequent enough to earn vectors of their own. They
# attach to `profit` because they appear in the same frames it does. A production pipeline
# either strips numeric tokens or normalizes them to a placeholder before training; this one
# leaves them in so the effect is visible rather than hidden.

# %% [markdown]
# ## Analogies, and what a small corpus does to them
#
# The property Word2Vec is famous for is that some relationships behave like vector
# addition: the vector for `king` minus `man` plus `woman` lands near `queen`. It is a
# genuine property of the geometry, and it is also the part of the technique that scales
# worst with corpus size, because it needs each of the four words to be well located.
#
# On a few thousand sentences they are not. Read the completions below as a check on whether
# the arithmetic is even defined here, not as an answer to the analogy.


# %%
def analogy_frame(triplets: list[tuple[str, str, str]], top_n: int = 3) -> pl.DataFrame:
    """For each (a, b, c), return top completions of a - b + c."""
    rows = []
    for a, b, c in triplets:
        try:
            results = model.wv.most_similar(positive=[a, c], negative=[b], topn=top_n)
        except KeyError as e:
            rows.append(
                {
                    "analogy": f"{a} - {b} + {c}",
                    "rank": 0,
                    "completion": f"<OOV: {e.args[0]}>",
                    "similarity": float("nan"),
                }
            )
            continue
        for rank, (word, score) in enumerate(results, start=1):
            rows.append(
                {
                    "analogy": f"{a} - {b} + {c}",
                    "rank": rank,
                    "completion": word,
                    "similarity": score,
                }
            )
    return pl.DataFrame(rows)


analogy_triplets = [
    ("profit", "increase", "decrease"),
    ("revenue", "growth", "decline"),
    ("shares", "rose", "fell"),
    ("strong", "profit", "loss"),
]
analogy_frame(analogy_triplets, top_n=3)

# %% [markdown]
# ## The same words, trained on six billion others
#
# GloVe on Wikipedia plus Gigaword sees roughly six billion tokens against this corpus's
# forty thousand. The interesting comparison is not which is better but what each has: the
# large model has enough text to separate words that this one confuses, and the small model
# has read sentences about earnings that the large one has barely seen.
#
# The download is about 128 MB and is cached after the first run. Its progress bar streams
# tens of thousands of lines, so it is captured rather than printed.


# %%
print("Loading pre-trained GloVe embeddings (glove-wiki-gigaword-100)...")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    glove = api.load("glove-wiki-gigaword-100")
print(f"GloVe vocabulary size: {len(glove):,}")


# %%
def neighbors(word: str, vectors, top_n: int = 5) -> list[str]:
    if word not in vectors:
        return ["<OOV>"] + [""] * (top_n - 1)
    return [f"{w} ({s:.2f})" for w, s in vectors.most_similar(word, topn=top_n)]


probe_terms = ["profit", "dividend", "shares"]
comparison_rows = []
for term in probe_terms:
    custom_n = neighbors(term, model.wv)
    glove_n = neighbors(term, glove)
    for rank, (a, b) in enumerate(zip(custom_n, glove_n), start=1):
        comparison_rows.append(
            {
                "probe": term,
                "rank": rank,
                "custom (Financial PhraseBank)": a,
                "GloVe (Wikipedia)": b,
            }
        )

pl.DataFrame(comparison_rows)

# %% [markdown]
# ## The same finding, drawn
#
# The neighbor table showed the effect one word at a time. Projecting a set of words to two
# dimensions shows whether it holds across the set. Three groups go in - words a reader would
# call positive, words they would call negative, and topical financial nouns - and the
# question is whether the model's geometry recovers that grouping.
#
# t-SNE preserves which points are near which; it does not preserve distance, direction or
# scale, so the axes carry no units and only adjacency is readable.

# %%
categories = {
    "positive": ["profit", "growth", "increase", "gain", "strong", "positive", "improved"],
    "negative": ["loss", "decline", "decrease", "weak", "negative", "dropped", "fell"],
    "financial": ["revenue", "earnings", "dividend", "shares", "market", "stock", "company"],
}

words_to_plot = []
embeddings_to_plot = []
colors = []

color_map = {
    "positive": COLORS["positive"],
    "negative": COLORS["negative"],
    "financial": COLORS["blue"],
}

for category, word_list in categories.items():
    for word in word_list:
        if word in model.wv:
            words_to_plot.append(word)
            embeddings_to_plot.append(model.wv[word])
            colors.append(color_map[category])

embeddings_array = np.array(embeddings_to_plot)
print(f"Visualizing {len(words_to_plot)} words")

# %% [markdown]
# t-SNE needs more points than its perplexity setting, so a vocabulary too small to supply
# them skips the figure rather than drawing a misleading one.


# %%
if len(words_to_plot) < 5:
    print(f"Too few words for t-SNE visualization (need at least 5, have {len(words_to_plot)})")
    print("Skipping visualization - train with more data or check vocabulary coverage")
    embeddings_2d = None
else:
    tsne = TSNE(
        n_components=2, perplexity=min(5, len(words_to_plot) - 1), random_state=SEED, max_iter=1000
    )
    embeddings_2d = tsne.fit_transform(embeddings_array)

if embeddings_2d is not None:
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=FIGSIZE["single_tall"])
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=60, alpha=0.8)

    for i, word in enumerate(words_to_plot):
        ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=7, alpha=0.9)

    legend_elements = [
        Patch(facecolor=COLORS["positive"], label="Reader calls positive"),
        Patch(facecolor=COLORS["negative"], label="Reader calls negative"),
        Patch(facecolor=COLORS["blue"], label="Topical noun"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

    ax.set_xlabel("t-SNE axis 1, no units")
    ax.set_ylabel("t-SNE axis 2, no units")
    ax.set_title("Opposites land together, so this geometry is not polarity")
    show_with_alt(
        fig,
        "A scatter of about twenty labeled words in two t-SNE dimensions, colored by whether "
        "a reader would call them positive, negative or topical. The three colors are "
        "interleaved rather than grouped: the closest pair in the plot is a green word and a "
        "red one at the bottom right, a red word sits between two green ones at the top, and "
        "the dark topical words are spread from one corner to the other with no region of "
        "their own.",
    )

# %% [markdown]
# The three colors do not separate, and the figure is worth more for that than it would be
# if they did.
#
# The closest pair in the plot is `profit` and `loss`, which is the neighbor table's finding
# holding across the whole set rather than at one probe word. `weak` sits among `strong` and
# `positive`. The topical nouns are scattered rather than grouped: `revenue` is at one edge
# and `earnings` and `dividend` at the opposite one, which says the model has placed
# `revenue` by the sentences it appears in rather than by what it denotes.
#
# What the geometry does encode is which words are used in the same frames. That is a real
# and useful signal - it is what makes a nearest-neighbor lookup a good way to expand a query
# or find a substitutable term - and it is the wrong feature for a sentiment model. A
# classifier handed these vectors would be asked to separate two words the representation has
# placed on top of each other.
#
# Two caveats on the picture itself. t-SNE run on about twenty points with a perplexity of
# five is a crude summary of a hundred-dimensional space, and a different seed moves the
# layout. Neither changes the finding, which is already visible in the neighbor table at full
# dimensionality; the figure only shows that it is general rather than anecdotal.

# %% [markdown]
# ## The same machinery, on portfolios
#
# `02_asset_embeddings` does not adapt this technique to assets; it runs the same one on a
# different corpus. The correspondence is exact term for term:
#
# | Here | There |
# |---|---|
# | A sentence | A filer's portfolio |
# | A word in it | A stock held in it |
# | Words sharing a sentence are pushed together | Stocks held by the same institution are pushed together |
# | Nearest neighbors give substitutable words | Nearest neighbors give substitutable stocks |
#
# The transfer carries this notebook's finding with it. Co-holding is not similar
# performance any more than co-occurrence is similar meaning, and two stocks a manager holds
# as a pair trade are held together precisely because they are expected to move apart. The
# question to bring to the next notebook is which of those two things its embedding measures.
#
# ## What static embeddings do not encode
#
# Four limits, in the order they cost you something. Each is about what the geometry carries,
# not about what can eventually be built on it.
#
# 1. **Polarity is not a direction in this space.** Demonstrated above: distance measures
#    interchangeable usage, so `profit` and `loss` are close. A supervised classifier trained
#    on labeled sentences can still separate them - `03_sentiment_evolution` does exactly
#    that on averaged vectors - because two vectors that sit close together are still distinct
#    points. What the labels have to supply is the direction itself, which is why an
#    unsupervised method that ranks by similarity alone conflates the two.
# 2. **One vector per word, whatever it meant.** `Apple` the company and `apple` the fruit are
#    averaged into a single point, and so are `charge` the fee and `charge` the accusation.
#    The vector alone cannot say which sense an occurrence carries, so a downstream model has
#    to read the surrounding words to tell them apart - which is what the contextual models
#    below do by construction rather than as a repair.
# 3. **Nothing to say about a word it never saw.** A ticker that listed last month, a term of
#    art absent from the corpus, and any typo are out of vocabulary and have no vector at all,
#    not a poor one.
# 4. **No unit above the word.** "Net loss narrowed" is positive and none of its three words
#    is. Averaging or summing the three ignores order and negation, so composition is
#    something the downstream model has to learn rather than something the vectors provide.
#
# Section 10.4's contextual models change what is encoded rather than what can be learned on
# top: a word's vector becomes a function of the sentence it appears in, which addresses the
# first three, and the model emits a vector for the sentence itself, which addresses the
# fourth.
#
# ## Key takeaways
#
# 1. **Co-occurrence measures topic, not meaning.** The method places two words together when
#    they are used in the same frames. Read a nearest-neighbor list as "used like this", and
#    check the opposite of your target word before you trust it as a feature.
# 2. **The tokenizer is part of the model.** Keeping numeric tokens put currency fragments
#    among the neighbors of the most important word in the corpus. Decide what a token is
#    before deciding what the vectors mean.
# 3. **Corpus size buys precision, domain buys coverage.** A large general model separates
#    words a small one confuses; a small domain model has read sentences the large one has
#    not. Neither substitutes for the other, and the useful question is which failure your
#    task can tolerate.
# 4. **Reproducibility costs a worker.** More than one training thread means the update order
#    is the OS's decision and the seed no longer determines the result.

# %% [markdown]
# The trained model and a short summary are written to the chapter's output directory. No
# other notebook reads them - `02_asset_embeddings` trains its own model on portfolios and
# `03_sentiment_evolution` downloads pretrained vectors - so this is here for a reader who
# wants to load the vectors and probe them without paying for the training run again.


# %%
chapter_dir = get_chapter_dir(10)
output_dir = chapter_dir / "output" / "word2vec"
output_dir.mkdir(parents=True, exist_ok=True)
model.save(str(output_dir / "word2vec_financial.model"))
print(f"Model saved to: {output_dir / 'word2vec_financial.model'}")

results_file = output_dir / "results.md"
algorithm = "Skip-gram" if w2v_config["sg"] else "CBOW"
with open(results_file, "w") as f:
    f.write("# Word2Vec training results\n\n")
    f.write("## Configuration\n")
    f.write(f"- Training data: Financial PhraseBank ({len(sentences):,} sentences)\n")
    f.write(f"- Vocabulary size: {len(model.wv):,}\n")
    f.write(f"- Embedding dimension: {model.wv.vector_size}\n")
    f.write(f"- Context window: {w2v_config['window']}\n")
    f.write(f"- Algorithm: {algorithm} with {w2v_config['negative']} negative samples\n\n")
    f.write("## What the vectors encode\n")
    f.write("Co-occurrence, which is topic rather than meaning. On this corpus the nearest\n")
    f.write("neighbor of `profit` is `loss`, because the two are reported in identical\n")
    f.write("sentence frames. Treat a neighbor list as evidence of interchangeable usage.\n")

print(f"Results saved to: {results_file}")
