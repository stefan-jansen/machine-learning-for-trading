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
# # Financial Named Entity Recognition (NER) Fine-Tuning
#
# **Chapter 10: From Text to Features - The Transformer Breakthrough**
# **Section Reference**: See Section 10.4 for Transformers and token classification
#
# **Docker image**: `ml4t-gpu`
#
# > **GPU recommended**: This notebook trains models with PyTorch/CUDA. It will run on CPU
# > but training may be very slow. For GPU acceleration:
# > ```bash
# > docker compose run --rm ml4t-gpu python 10_text_feature_engineering/05_financial_ner_finetuning.py
# > ```
#
#
# ## What this notebook is for
#
# Named entity recognition turns a sentence into structured fields: which spans are
# organizations, which are amounts, which are dates. That is what makes a filing or an
# earnings call queryable, and it is the step between having text and having a table.
#
# The task differs from the sentiment classification in `04_bert_finetuning` in a way that
# causes most of the difficulty: a tag marks part of a sentence rather than the whole of it,
# and a transformer works in subword pieces that do not line up with words. So
# most of the work below is alignment, and the piece worth reading closely is the function
# that maps word-level tags onto subword tokens.
#
# The span is the unit here: a tag names where an entity starts and where it ends, not what
# the sentence as a whole is about.
#
# The data here is generated from templates rather than annotated by hand. That keeps the
# notebook runnable, and it has a consequence the notebook measures rather than glosses: the
# generator repeats itself, so much of the test set is also in the training set and the
# scores are near the ceiling for that reason.
#
# ## Learning objectives
#
# After working through this notebook you will be able to:
#
# - Read and write BIO tags, and say what goes wrong if a multi-word entity opens with `I-`.
# - Align word-level labels to a transformer's subword tokens, and explain which subwords get
#   a label and which get ignored by the loss.
# - Fine-tune a transformer for token classification and score it entity by entity rather
#   than token by token.
# - Check whether a held-out split of generated data is actually held out.
#
# ## Prerequisites
#
# - Section 10.4 of the chapter.
# - `04_bert_finetuning.py` for the Trainer mechanics, which are not re-explained here.
#
# ## Related notebooks
#
# - `04_bert_finetuning.py` - the same Trainer applied to whole-sentence classification
# - `09_filing_text_signals.py` - extracting features from filings at scale

# %%
"""Fine-tune a transformer for financial named entity recognition."""

import json
import random
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

# `multiprocess`, reached through these two, raises a SyntaxWarning at COMPILE time, so a
# module-level filter set afterwards is too late and it reached the committed render carrying
# the absolute path of the environment that produced it.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import evaluate
    from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers import (
    set_seed as set_transformers_seed,
)

from utils.paths import get_chapter_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, show_with_alt

# %% tags=["parameters"]
SEED = 42
N_SAMPLES = 500
N_EPOCHS = 3

# %% [markdown]
# `set_global_seeds` covers Python, NumPy and Torch. The Trainer draws from its own generator
# for shuffling and dropout, which needs seeding separately or the run is not reproducible.


# %%
set_global_seeds(SEED)
set_transformers_seed(SEED)

CONFIG = {
    "random_seed": SEED,
    "n_samples": N_SAMPLES,
    "n_epochs": N_EPOCHS,
    "model": {
        "base": "ProsusAI/finbert",
        "description": "FinBERT - BERT pre-trained on financial text",
    },
    "dataset": {
        "source": "synthetic (teaching-focused)",
        "schema": "IOB2 (Inside-Outside-Beginning variant 2)",
        "entity_types": ["ORG", "MONEY", "DATE", "PER", "PERCENT"],
        "note": "Synthetic data matches chapter's coarse-grained taxonomy",
    },
    "training": {
        "learning_rate": 2e-5,
        "batch_size": 16,
        "weight_decay": 0.01,
        "max_length": 128,
    },
}

print(json.dumps(CONFIG, indent=2))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# %% [markdown]
# ## The data, and why it is generated
#
# Annotated NER corpora are expensive because a person has to mark every span, so this
# notebook generates its sentences from templates instead. The tradeoff is explicit: the
# pipeline that follows is exactly what you would run on CoNLL-2003 or FiNER-139 or your own
# annotations, and the scores it produces on this data are not.
#
# The generator fills five sentence templates from five options per entity slot, tagging as
# it goes so the labels are correct by construction rather than by annotation.


# %%
def generate_synthetic_ner_data(n_samples: int = 500, seed: int = 42):
    """Draw *n_samples* template-generated financial sentences with BIO tags.

    The vocabulary per slot is deliberately small, so the number of distinct sentences this
    can produce is bounded and drawing more samples than that repeats them.
    """
    random.seed(seed)

    orgs = ["Apple Inc.", "Microsoft", "Goldman Sachs", "JPMorgan Chase", "Tesla Motors"]
    people = ["Tim Cook", "Satya Nadella", "Warren Buffett", "Elon Musk", "Janet Yellen"]
    money = ["$500 million", "$1.2 billion", "$50,000", "€10 million", "£5.5 billion"]
    dates = ["Q3 2024", "March 15, 2024", "fiscal year 2023", "last quarter", "January 2025"]
    percents = ["15%", "2.5%", "10 percent", "25.7%", "3.2%"]

    templates = [
        ("{ORG}", "announced", "revenue", "of", "{MONEY}", "for", "{DATE}"),
        ("{PER}", "CEO", "of", "{ORG}", "reported", "growth", "of", "{PERCENT}"),
        ("The", "stock", "of", "{ORG}", "rose", "{PERCENT}", "on", "{DATE}"),
        ("{ORG}", "plans", "to", "invest", "{MONEY}", "in", "new", "facilities"),
        ("{PER}", "sold", "{MONEY}", "worth", "of", "{ORG}", "shares"),
    ]

    samples = []
    for _ in range(n_samples):
        template = random.choice(templates)
        tokens = []
        ner_tags = []

        for word in template:
            if word == "{ORG}":
                org = random.choice(orgs)
                org_tokens = org.split()
                tokens.extend(org_tokens)
                ner_tags.append(1)  # B-ORG
                ner_tags.extend([2] * (len(org_tokens) - 1))  # I-ORG
            elif word == "{PER}":
                per = random.choice(people)
                per_tokens = per.split()
                tokens.extend(per_tokens)
                ner_tags.append(7)  # B-PER
                ner_tags.extend([8] * (len(per_tokens) - 1))  # I-PER
            elif word == "{MONEY}":
                mon = random.choice(money)
                mon_tokens = mon.split()
                tokens.extend(mon_tokens)
                ner_tags.append(3)  # B-MONEY
                ner_tags.extend([4] * (len(mon_tokens) - 1))  # I-MONEY
            elif word == "{DATE}":
                date = random.choice(dates)
                date_tokens = date.split()
                tokens.extend(date_tokens)
                ner_tags.append(5)  # B-DATE
                ner_tags.extend([6] * (len(date_tokens) - 1))  # I-DATE
            elif word == "{PERCENT}":
                pct = random.choice(percents)
                pct_tokens = pct.split()
                tokens.extend(pct_tokens)
                ner_tags.append(9)  # B-PERCENT
                ner_tags.extend([10] * (len(pct_tokens) - 1))  # I-PERCENT
            else:
                tokens.append(word)
                ner_tags.append(0)  # O

        samples.append({"tokens": tokens, "ner_tags": ner_tags})

    return Dataset.from_list(samples)


# %% [markdown]
# The tag vocabulary is fixed here rather than inferred from the data, so a label id means
# the same thing on every run and a class absent from one sample does not renumber the rest.


# %%
def load_ner_dataset():
    """Load financial NER dataset with explicit provenance tracking.

    Uses synthetic data designed to match chapter prose (coarse-grained financial
    entities: ORG, MONEY, DATE, PER, PERCENT). This provides consistent, reproducible
    results for teaching purposes.

    Returns:
        tuple: (dataset, label_list) where label_list is the BIO tag vocabulary
    """
    # Coarse-grained financial NER label scheme (matches chapter prose)
    label_list = [
        "O",  # 0: Outside
        "B-ORG",  # 1: Beginning of organization
        "I-ORG",  # 2: Inside organization
        "B-MONEY",  # 3: Beginning of monetary value
        "I-MONEY",  # 4: Inside monetary value
        "B-DATE",  # 5: Beginning of date
        "I-DATE",  # 6: Inside date
        "B-PER",  # 7: Beginning of person
        "I-PER",  # 8: Inside person
        "B-PERCENT",  # 9: Beginning of percentage
        "I-PERCENT",  # 10: Inside percentage
    ]

    print("\n" + "=" * 70)
    print("DATASET PROVENANCE")
    print("=" * 70)
    print("  Source: Synthetic financial NER data")
    print("  Purpose: Teaching BIO tagging and token classification")
    print(f"  Schema: {CONFIG['dataset']['schema']}")
    print(f"  Entity types: {CONFIG['dataset']['entity_types']}")
    print(f"  Samples: {N_SAMPLES}")
    print("\n  Note: Synthetic data provides controlled examples matching")
    print("  the chapter's coarse-grained entity taxonomy. For production")
    print("  NER, use annotated datasets like CoNLL-2003 or domain-specific")
    print("  corpora with appropriate label mappings.")
    print("=" * 70 + "\n")

    return generate_synthetic_ner_data(n_samples=N_SAMPLES, seed=SEED), label_list


# %%
dataset, LABEL_LIST = load_ner_dataset()

id2label = dict(enumerate(LABEL_LIST))
label2id = {label: i for i, label in id2label.items()}

split = dataset.train_test_split(test_size=0.2, seed=SEED)
print(f"Train: {len(split['train'])}, Test: {len(split['test'])}")

# %% [markdown]
# ### How much of the test set is already in training
#
# The generator can produce a few hundred distinct sentences: four of its templates have
# three entity slots and one has two, with five options each. Every draw is independent and
# with replacement, so long before the sample count approaches that ceiling the same
# sentences come up repeatedly - the collision argument is the birthday problem, not a
# shortage of possibilities.
#
# The consequence is what matters here. A random split of a sample containing duplicates puts
# copies of the same sentence on both sides, so the model is scored partly on sentences it
# was trained on. That is worth counting rather than assuming, because it is the reason the
# scores below look the way they do. The count is over exact token sequences.

# %%
train_sentences = [" ".join(row) for row in split["train"]["tokens"]]
test_sentences = [" ".join(row) for row in split["test"]["tokens"]]
distinct_test = set(test_sentences)
memorized = distinct_test & set(train_sentences)

print(f"Sentences drawn: {len(dataset):,}, distinct: {len(set(train_sentences) | distinct_test):,}")
print(f"Distinct test sentences: {len(distinct_test)}")
print(f"  of which also appear verbatim in training: {len(memorized)}")

# %% [markdown]
# One example, with the tag on each token. `B-` opens an entity, `I-` continues the one
# before it, and `O` is everything outside an entity. A two-word company name is therefore
# `B-ORG` followed by `I-ORG`, which is what lets the scheme mark where one entity ends and
# the next begins.

# %%
example = split["train"][0]
for token, tag in zip(example["tokens"][:10], example["ner_tags"][:10], strict=True):
    print(f"  {token:15} -> {id2label[tag]}")

# %% [markdown]
# ## Aligning labels to subwords
#
# The labels are one per word; the model reads one token per subword, and "JPMorgan" may
# arrive as three of them. Something has to decide which subword carries the word's label.
#
# The convention below gives the label to the first subword of each word and marks the rest
# -100, which is the value PyTorch's cross-entropy ignores. So the loss is computed once per
# word rather than once per subword, and a word that happens to split into many pieces does
# not outweigh a word that does not. Getting this wrong is the most common way an NER
# pipeline trains without error and scores badly.

# %%
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_and_align_labels(examples):
    """Tokenize and align labels with subword tokens."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word gets the label
                label_ids.append(label[word_idx])
            else:
                # Subsequent subwords get -100
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Tokenize dataset
tokenized_dataset = split.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=split["train"].column_names,
)

# %%
# Load model
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(LABEL_LIST),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# %%
# Metrics - try seqeval first, fall back to sklearn if not installed
try:
    seqeval = evaluate.load("seqeval")
    SEQEVAL_AVAILABLE = True
except Exception:
    SEQEVAL_AVAILABLE = False
    print("seqeval not available, using sklearn metrics fallback")
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(eval_pred):
    """Compute metrics for NER evaluation.

    Handles both tuple format (predictions, labels) and EvalPrediction object
    for compatibility across HuggingFace transformers versions.
    """
    # Handle both tuple and EvalPrediction object formats
    if hasattr(eval_pred, "predictions"):
        # EvalPrediction object
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        # Tuple format
        predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [LABEL_LIST[pred] for (pred, lab) in zip(prediction, label, strict=False) if lab != -100]
        for prediction, label in zip(predictions, labels, strict=False)
    ]
    true_labels = [
        [LABEL_LIST[lab] for (pred, lab) in zip(prediction, label, strict=False) if lab != -100]
        for prediction, label in zip(predictions, labels, strict=False)
    ]

    if SEQEVAL_AVAILABLE:
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    else:
        # Flatten for sklearn metrics (token-level, not entity-level)
        flat_preds = [tag for seq in true_predictions for tag in seq]
        flat_labels = [tag for seq in true_labels for tag in seq]
        return {
            "precision": precision_score(
                flat_labels, flat_preds, average="weighted", zero_division=0
            ),
            "recall": recall_score(flat_labels, flat_preds, average="weighted", zero_division=0),
            "f1": f1_score(flat_labels, flat_preds, average="weighted", zero_division=0),
            "accuracy": accuracy_score(flat_labels, flat_preds),
        }


# %%
# Training arguments - save checkpoints under chapter output directory
chapter_dir = get_chapter_dir(10)
output_dir = chapter_dir / "output" / "financial_ner"
output_dir.mkdir(parents=True, exist_ok=True)

# Build training arguments dict with version-compatible parameter names
# transformers 4.36+ uses eval_strategy, older versions use evaluation_strategy
import inspect

eval_strat_key = "eval_strategy"  # Default to newer API
try:
    sig = inspect.signature(TrainingArguments)
    if "evaluation_strategy" in sig.parameters and "eval_strategy" not in sig.parameters:
        eval_strat_key = "evaluation_strategy"
except Exception:
    pass  # Use defaults

training_kwargs = {
    "output_dir": str(output_dir),
    eval_strat_key: "epoch",
    "save_strategy": "epoch",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": N_EPOCHS,
    "weight_decay": 0.01,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "report_to": "none",
    "fp16": torch.cuda.is_available(),
}

training_args = TrainingArguments(**training_kwargs)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %%
print("Training NER model...")
trainer.train()

results = trainer.evaluate()
print("\nTest Results:")
print(f"  Precision: {results['eval_precision']:.3f}")
print(f"  Recall: {results['eval_recall']:.3f}")
print(f"  F1: {results['eval_f1']:.3f}")

# %% [markdown]
# ### What these scores are measuring
#
# The scores are near the ceiling, and the count printed after the split says why: most of
# the distinct test sentences appear verbatim in the training set. The model is being asked
# to reproduce sentences it has already seen, which it can do, and the metric is reporting
# that it did.
#
# So read this number as a plumbing check. It says the tokenizer, the subword-to-word label
# alignment, the collator and the training loop are wired up correctly, which is genuinely
# worth confirming and is the thing most likely to be silently wrong in an NER pipeline. It
# says nothing about whether the model can find an entity it has not seen before.
#
# On a real annotated corpus - CoNLL-2003, FiNER-139, or your own filings - the interesting
# errors appear: boundaries in the wrong place, a ticker read as an ordinary word, an
# organization the model has never encountered. The pipeline below is unchanged for those;
# only the data is.
#
# There is a second thing to check before trusting an NER number anywhere. The canonical
# metric is entity-level, meaning a span counts only when both its full extent and its type
# match, and `seqeval` computes it. Without `seqeval` installed this notebook falls back to a
# token-level score from `sklearn`, which is more lenient because a span with one token wrong
# still earns credit for the rest. On data this saturated the two agree; on real data they do
# not, and the entity-level one is the number to quote.

# %% [markdown]
# ## Reading entities back out
#
# A token-classification model emits one label per subword token. Turning that into the spans
# a downstream system wants means grouping consecutive tokens by their tag, which is what the
# `B-`/`I-` distinction exists for and what the function below does.


# %%
def extract_entities(text: str) -> list[tuple[str, str]]:
    """Extract named entities from text using the fine-tuned model."""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    # Move inputs to same device as model
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Get special tokens to filter out
    special_tokens = set(tokenizer.all_special_tokens)

    # Extract entities
    entities = []
    current_entity = []
    current_type = None

    for token, pred in zip(tokens, predictions[0], strict=False):
        # Skip special tokens like [CLS], [SEP], [PAD]
        if token in special_tokens:
            continue

        label = id2label[pred.item()]

        if label.startswith("B-"):
            # Save previous entity if exists
            if current_entity:
                entity_text = tokenizer.convert_tokens_to_string(current_entity)
                entities.append((entity_text.strip(), current_type))

            # Start new entity
            current_entity = [token]
            current_type = label[2:]

        elif label.startswith("I-") and current_type == label[2:]:
            # Continue current entity
            current_entity.append(token)

        else:
            # End entity
            if current_entity:
                entity_text = tokenizer.convert_tokens_to_string(current_entity)
                entities.append((entity_text.strip(), current_type))
                current_entity = []
                current_type = None

    # Don't forget the last entity
    if current_entity:
        entity_text = tokenizer.convert_tokens_to_string(current_entity)
        entities.append((entity_text.strip(), current_type))

    return entities


# Test sentences
test_sentences = [
    "Apple Inc. reported quarterly revenue of $94.8 billion in January 2024.",
    "Goldman Sachs CEO David Solomon announced a 15% increase in dividends.",
    "Tesla shares rose 5% following Elon Musk's announcement.",
]

print("\n" + "=" * 60)
print("ENTITY EXTRACTION EXAMPLES")
print("=" * 60)

for sentence in test_sentences:
    print(f"\nText: {sentence}")
    entities = extract_entities(sentence)
    if entities:
        print("Entities:")
        for text, etype in entities:
            print(f"  - {text}: {etype}")
    else:
        print("  (No entities detected)")

# %% [markdown]
# ## How many of each type it found, against how many there were
#
# Counting only `B-` tags counts entities. Counting every non-`O` tag counts tagged tokens,
# which double-counts every multi-word span - and since organizations and people here are
# usually two words while percentages are one, that would inflate the types unevenly and make
# the distribution say something about span length rather than about frequency.
#
# Predicted counts alone would also not answer the question the chart implies. The true
# labels are already in hand, so both go on the axis - and on this data the two bars in each
# pair coincide, which is the overlap counted after the split showing up in the output rather
# than a separate result.

# %%
predictions = trainer.predict(tokenized_dataset["test"])
pred_labels = np.argmax(predictions.predictions, axis=2)

predicted_counts, true_counts = Counter(), Counter()
for pred_seq, label_seq in zip(pred_labels, predictions.label_ids, strict=True):
    for predicted, actual in zip(pred_seq, label_seq, strict=True):
        if actual == -100:
            continue
        # `B-` opens an entity, so one `B-` is one entity; `I-` continues the same one.
        if id2label[predicted].startswith("B-"):
            predicted_counts[id2label[predicted][2:]] += 1
        if id2label[actual].startswith("B-"):
            true_counts[id2label[actual][2:]] += 1

entity_types = sorted(set(predicted_counts) | set(true_counts))
print({t: (true_counts[t], predicted_counts[t]) for t in entity_types})

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
positions = np.arange(len(entity_types))
width = 0.38

ax.bar(
    positions - width / 2,
    [true_counts[t] for t in entity_types],
    width,
    label="In the labels",
    color=COLORS["blue"],
)
ax.bar(
    positions + width / 2,
    [predicted_counts[t] for t in entity_types],
    width,
    label="Predicted",
    color=COLORS["amber"],
)

ax.set_xticks(positions)
ax.set_xticklabels(entity_types)
ax.set_xlabel("Entity type")
ax.set_ylabel("Entities in the test split")
ax.set_title("Entity counts by type, labeled against predicted")
ax.legend(fontsize=7)

show_with_alt(
    fig,
    "A grouped bar chart with one pair of bars per entity type, the left bar of each pair "
    "counting the entities in the test labels and the right bar counting those the model "
    "predicted. The two bars in each pair are indistinguishable in height, so the chart reads "
    "as five single bars of differing heights rather than as a comparison, and the types "
    "differ from one another by up to a factor of three.",
)

# %% [markdown]
# ## Downstream Feature Engineering: From Entities to ML Features
#
# Counting the entities a document mentions is the simplest feature that NER makes possible,
# and it is the one to start with: how many organizations a filing names, how many dated
# commitments it makes, how many figures it quotes. These are counts a model can read
# directly, and they exist only because the spans were identified first.


# %%
def extract_entity_features(text: str) -> dict:
    """
    Extract NER-based features from text for ML modeling.

    Returns:
        Dictionary of features derived from extracted entities.
    """
    entities = extract_entities(text)

    # Initialize feature counts
    features = {
        "n_org": 0,  # Number of organizations mentioned
        "n_money": 0,  # Number of monetary values
        "n_date": 0,  # Number of dates
        "n_per": 0,  # Number of people
        "n_percent": 0,  # Number of percentages
        "n_total_entities": 0,  # Total entity count
        "has_money": 0,  # Binary: mentions money?
        "has_multiple_orgs": 0,  # Binary: >1 org mentioned?
    }

    for _, etype in entities:
        features["n_total_entities"] += 1
        if etype == "ORG":
            features["n_org"] += 1
        elif etype == "MONEY":
            features["n_money"] += 1
        elif etype == "DATE":
            features["n_date"] += 1
        elif etype == "PER":
            features["n_per"] += 1
        elif etype == "PERCENT":
            features["n_percent"] += 1

    # Derived features
    features["has_money"] = 1 if features["n_money"] > 0 else 0
    features["has_multiple_orgs"] = 1 if features["n_org"] > 1 else 0

    return features


# Demo on sample financial texts
sample_texts = [
    "Apple Inc. reported quarterly revenue of $94.8 billion in January 2024.",
    "Goldman Sachs CEO David Solomon announced a 15% increase in dividends.",
    "The Federal Reserve raised interest rates by 0.25% following inflation data.",
    "Tesla shares rose 5% after Elon Musk announced new factory plans.",
]

import polars as pl

feature_records = []
for text in sample_texts:
    features = extract_entity_features(text)
    features["text"] = text[:50] + "..." if len(text) > 50 else text
    feature_records.append(features)

features_df = pl.DataFrame(feature_records).select(
    ["text", "n_org", "n_money", "n_per", "n_percent", "has_money", "n_total_entities"]
)
features_df

# %% [markdown]
# ## Key takeaways
#
# 1. **Check whether a generated split is actually held out.** A generator with a bounded
#    vocabulary repeats itself, and a random split then puts the same sentence on both sides.
#    Counting distinct sequences across the two halves takes one line and tells you what the
#    score can mean; without it a saturated metric is indistinguishable from a good model.
# 2. **Subword alignment is where a token-classification pipeline goes wrong quietly.** The
#    label is per word, the model reads subwords, and only the first subword should carry it
#    while the rest are ignored by the loss. Get this wrong and the code runs, trains
#    and reports a number.
# 3. **BIO exists to mark boundaries, not just types.** `B-` opening and `I-` continuing is
#    what distinguishes two adjacent organizations from one two-word organization. A scheme
#    that only labeled tokens by type could not tell those apart.
# 4. **Count entities by their opening tag.** Counting every non-`O` token counts tokens, and
#    since span length varies by entity type that turns a distribution over types into a
#    distribution over how many words each type usually takes.
# 5. **Entity-level and token-level scores are different metrics.** A span counts only when
#    its extent and its type both match, which is what `seqeval` computes and what to quote.
#    A token-level score gives partial credit for a span whose boundary is wrong.
# 6. **The output is a feature, not an answer.** Counts of organizations, amounts and dates
#    per document are structured columns a model can read, and producing them is the reason to
#    run NER over a corpus of filings at all.
