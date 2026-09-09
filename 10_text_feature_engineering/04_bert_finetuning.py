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
# # Fine-tuning a transformer, and checking what its checkpoint already saw
#
# **Chapter 10: text feature engineering**
#
# **Docker image**: `ml4t-gpu`
#
# **Section reference**: Sections 10.4 and 10.5
#
# ## What this notebook is for
#
# Fine-tuning a pre-trained transformer for sentence classification is a short and
# well-supported piece of work: load a checkpoint, tokenize, hand it to the Trainer, read
# the metrics. This notebook does that three times, on one domain-specific checkpoint and
# two general ones, so the mechanics are visible on a task small enough to run.
#
# The comparison it produces is the part worth being careful about. One of the three
# checkpoints, `ProsusAI/finbert`, was itself fine-tuned on the Financial PhraseBank - the
# corpus this notebook's test split is drawn from. Its test score is therefore measured on
# sentences it has already been trained on, and it is not a held-out number. That fact is
# stated where the models are introduced and carried in the results table, because a
# comparison table is read long before any caveat at the end of a notebook.
#
# ## Learning objectives
#
# After working through this notebook you will be able to:
#
# - Fine-tune a pre-trained transformer for sentence classification with the Hugging Face
#   Trainer, and say what each of the training arguments changes.
# - Check whether a published checkpoint was trained on the data you are about to test it
#   on, and say what that does to the number you get.
# - Read a confusion matrix for a three-class sentiment task and identify which pair of
#   classes a model actually confuses.
# - Decide whether a difference between two models on one test split is large enough to act
#   on.
#
# ## Prerequisites
#
# - Sections 10.4 and 10.5 of the chapter.
# - `03_sentiment_evolution` for the baselines these models are being compared against.
#
# ## Related notebooks
#
# - `03_sentiment_evolution.py` - lexicon, TF-IDF and static-embedding baselines
# - `06_finbert_cross_dataset.py` - the same checkpoints evaluated on a different corpus
# - `12_gradient_boosting/10_shap_nlp_sentiment.py` - attributing a text model's decisions
#
# ## What it costs to run
#
# Three fine-tuning runs on one GPU. The training time each takes is measured and reported
# below rather than asserted here, because it depends on the card.

# %%
"""Fine-tune three transformer checkpoints on financial sentiment and compare them."""

import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch

# `multiprocess`, reached through these two, raises a SyntaxWarning at COMPILE time, so a
# module-level filter set after the import is too late and the warning reached the committed
# render carrying the absolute path of whichever checkout produced it.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import evaluate
    from datasets import Dataset, DatasetDict

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers import (
    set_seed as set_transformers_seed,
)

from data import load_financial_phrasebank
from utils.paths import get_chapter_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, show_with_alt

# %% [markdown]
# The two parameters below are what a reduced run overrides. `MAX_TRAIN_STEPS` caps the total
# number of optimizer steps; at -1 the Trainer runs the configured epochs instead. Neither
# declaration carries a trailing comment, because papermill parses this cell line by line and
# a comment on the same line hides the name from it - the override is then dropped in silence
# and the reduced run trains at production size.

# %% tags=["parameters"]
SEED = 42
MAX_TRAIN_STEPS = -1

# %% [markdown]
# `set_global_seeds` covers Python, NumPy and Torch. The Trainer draws from its own generator
# for shuffling and dropout, which needs seeding separately or the run is not reproducible
# even with everything else pinned.


# %%
set_global_seeds(SEED)
set_transformers_seed(SEED)

CONFIG = {
    "random_seed": SEED,
    "dataset": {
        "name": "takala/financial_phrasebank",
        "subset": "sentences_allagree",
        "test_size": 0.15,
        "val_size": 0.15,
    },
    # `saw_phrasebank` is the field that decides how each row of the results table may be
    # read. It is a property of the published checkpoint, not of anything this notebook
    # does, and it is declared here so no comparison below can quietly omit it.
    "models": {
        "finbert": {
            "model_id": "ProsusAI/finbert",
            "description": "BERT already fine-tuned for sentiment on Financial PhraseBank",
            "saw_phrasebank": True,
        },
        "deberta": {
            "model_id": "microsoft/deberta-v3-small",
            "description": "General checkpoint, disentangled attention",
            "saw_phrasebank": False,
        },
        "modernbert": {
            "model_id": "answerdotai/ModernBERT-base",
            "description": "General checkpoint, long context window",
            "saw_phrasebank": False,
        },
    },
    "training": {
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_length": 128,
        "early_stopping_patience": 2,
    },
}

print(json.dumps(CONFIG, indent=2))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# %% [markdown]
# ## The corpus and the split
#
# The `sentences_allagree` subset keeps only sentences every annotator scored the same way.
# The split is stratified on the label, because the three classes are far from balanced and
# an unstratified draw would leave the test set with a different class mix from the training
# set, which moves accuracy for reasons that have nothing to do with the model.

# %%
df = load_financial_phrasebank(agreement="100")
print(f"Dataset size: {len(df):,}")

# %%
dataset_config = CONFIG["dataset"]
held_out_fraction = dataset_config["test_size"] + dataset_config["val_size"]

df_pd = df.to_pandas()
train_pd, temp_pd = train_test_split(
    df_pd, test_size=held_out_fraction, random_state=SEED, stratify=df_pd["label"]
)
val_pd, test_pd = train_test_split(
    temp_pd,
    test_size=dataset_config["test_size"] / held_out_fraction,
    random_state=SEED,
    stratify=temp_pd["label"],
)

train_df = pl.from_pandas(train_pd)
val_df = pl.from_pandas(val_pd)
test_df = pl.from_pandas(test_pd)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


# %% [markdown]
# The Trainer reads a `DatasetDict`, which is built from pandas. Polars has no direct
# conversion into it, so pandas is the boundary here rather than a preference.


# %%
def create_dataset_dict(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame):
    return DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_df.select(["sentence", "label"]).to_pandas(), preserve_index=False
            ),
            "validation": Dataset.from_pandas(
                val_df.select(["sentence", "label"]).to_pandas(), preserve_index=False
            ),
            "test": Dataset.from_pandas(
                test_df.select(["sentence", "label"]).to_pandas(), preserve_index=False
            ),
        }
    )


dataset = create_dataset_dict(train_df, val_df, test_df)

# %% [markdown]
# ## The three checkpoints, and which of them can be read as held out
#
# `ProsusAI/finbert` is a BERT that its authors already fine-tuned for sentiment on the
# Financial PhraseBank. The other two are general checkpoints that have not seen it. All
# three are fine-tuned here on the same training split, so the mechanics are identical, but
# only two of the three produce a test score with the usual meaning.
#
# For FinBERT the test sentences are not unseen: they were in the corpus its published
# weights were trained on. Whatever it scores here is an upper bound inflated by that
# exposure, and it cannot be compared with the other two or attributed to the fine-tuning
# step this notebook performs. It is kept in the comparison because a reader will reach for
# a domain checkpoint first and should see what checking its provenance is worth.

# %%
MODELS = {
    "FinBERT": CONFIG["models"]["finbert"],
    "DeBERTa-v3": CONFIG["models"]["deberta"],
    "ModernBERT": CONFIG["models"]["modernbert"],
}

id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}


# %% [markdown]
# Tokenization does not pad. Padding every sentence to `max_length` would spend most of the
# compute on padding tokens, because these sentences are far shorter than the limit;
# `DataCollatorWithPadding` instead pads each batch to its own longest member.


# %%
def tokenize_function(examples, tokenizer, max_length=128):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=max_length,
        # Note: Don't pad here; use DataCollatorWithPadding for dynamic padding
    )


# %% [markdown]
# Two metrics, and the second is the one to read. Accuracy on a corpus this imbalanced is
# dominated by the majority class; macro F1 averages the per-class scores with equal weight,
# so a model that never predicts the smallest class cannot hide behind the other two.

# %%
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


# %%
def fine_tune_model(model_name: str, spec: dict, dataset: DatasetDict) -> dict:
    """Fine-tune one checkpoint for sentiment classification and score it on the test split."""
    model_path = spec["model_id"]
    print(f"Fine-tuning {model_name} from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Handle models without pad token (proper approach for encoder-only models)
    # Encoder models like BERT don't have eos_token; use [PAD] or add one
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # For BERT-like models, add [PAD] token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Resize embeddings if we added a new token
    model.resize_token_embeddings(len(tokenizer))

    # Handle pad token in model config
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Create data collator for dynamic padding (more efficient than max_length padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Tokenize dataset
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["sentence"],
    )

    # Training arguments - save checkpoints under chapter output directory
    chapter_dir = get_chapter_dir(10)
    output_dir = chapter_dir / "output" / "bert_finetuning" / model_name.lower().replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use CONFIG values consistently
    train_config = CONFIG["training"]
    num_epochs = train_config["num_epochs"]
    batch_size = train_config["batch_size"]
    train_size = len(tokenized["train"])

    max_steps = MAX_TRAIN_STEPS

    # `load_best_model_at_end` requires save_steps to be a multiple of eval_steps, so the two
    # are derived together rather than set independently.
    if max_steps > 0:
        eval_steps = max(10, max_steps // 5)
        # save_steps must be a multiple of eval_steps
        save_steps = eval_steps * 2  # evaluate twice, save once
    else:
        eval_steps = None
        save_steps = None

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": num_epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size * 2,
        "warmup_steps": min(50, max(10, train_size // batch_size // 4)),
        "weight_decay": train_config["weight_decay"],
        "logging_steps": max(10, train_size // batch_size // 3),
        "eval_strategy": "steps" if max_steps > 0 else "epoch",
        "eval_steps": eval_steps,
        "save_strategy": "steps" if max_steps > 0 else "epoch",
        "save_steps": save_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "report_to": "none",  # Disable wandb/tensorboard
        "fp16": torch.cuda.is_available(),
    }

    training_args = TrainingArguments(**training_kwargs)

    # Create trainer with data collator for dynamic padding
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=train_config["early_stopping_patience"])
        ],
    )

    # Train
    start_time = time.time()
    train_result = trainer.train()
    train_time = time.time() - start_time

    # Evaluate on test set
    test_results = trainer.evaluate(tokenized["test"])

    # Get predictions for confusion matrix
    predictions = trainer.predict(tokenized["test"])
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    return {
        "model_name": model_name,
        "accuracy": test_results["eval_accuracy"],
        "f1": test_results["eval_f1"],
        "train_time": train_time,
        "num_params": sum(p.numel() for p in model.parameters()),
        "saw_phrasebank": spec["saw_phrasebank"],
        "y_pred": y_pred,
        "y_true": y_true,
        "train_loss": train_result.training_loss,
    }


# %% [markdown]
# ## Fine-tuning the three

# %%
results = {name: fine_tune_model(name, spec, dataset) for name, spec in MODELS.items()}

# %% [markdown]
# ## The comparison, with the contaminated row marked
#
# The last column is what makes the table readable. A row whose checkpoint had already been
# trained on this corpus is not a held-out measurement, and ranking it against the rows that
# are compares two different things.

# %%
summary_df = pl.DataFrame(
    [
        {
            "Model": r["model_name"],
            "Accuracy": f"{r['accuracy']:.1%}",
            "F1 (macro)": f"{r['f1']:.3f}",
            "Parameters": f"{r['num_params'] / 1e6:.1f}M",
            "Train Time": f"{r['train_time']:.0f}s",
            "Test split held out?": "no, checkpoint saw it" if r["saw_phrasebank"] else "yes",
        }
        for r in results.values()
    ]
)

summary_df

# %% [markdown]
# Two panels, and neither of them ranks the models. Read the left one for how little
# separates the three scores against the range the axis could take, and remember that the
# asterisked model is not measured on held-out data. Read the right one against it: the
# training costs differ from one another far more than the scores do, which is the practical
# finding here. The right panel's times are for whatever device this run used, printed above.

# %%
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["dual_h_tall"])

ax = axes[0]
model_names = list(results.keys())
accuracies = [results[m]["accuracy"] for m in model_names]
f1_scores = [results[m]["f1"] for m in model_names]
bar_labels = [f"{m}*" if results[m]["saw_phrasebank"] else m for m in model_names]

x = np.arange(len(model_names))
width = 0.35

ax.bar(x - width / 2, accuracies, width, label="Accuracy", color=COLORS["blue"])
ax.bar(x + width / 2, f1_scores, width, label="F1 (macro)", color=COLORS["amber"])

ax.set_ylabel("Score")
ax.set_title("Test accuracy and macro F1 by model")
ax.set_xticks(x)
ax.set_xticklabels(bar_labels, fontsize=7)
ax.legend(fontsize=7)
ax.set_ylim(0, 1)

ax = axes[1]
times = [results[m]["train_time"] for m in model_names]
bars = ax.bar(model_names, times, color=COLORS["slate"])
ax.set_ylabel("Seconds")
ax.set_title("Fine-tuning wall-clock time by model")
ax.tick_params(axis="x", labelsize=7)

for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{height:.0f}s",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
    )

show_with_alt(
    fig,
    "Two panels. The left has a pair of bars per model, accuracy and macro F1, on an axis "
    "running from zero to one; all six bars are tall and close to the same height, and the "
    "model whose checkpoint had already seen the test corpus is marked with an asterisk. "
    "The right has one bar per model for training time in seconds, each labeled with its "
    "value, and the three differ from one another far more than the scores on the left do.",
)

# %% [markdown]
# A single accuracy figure says how often a model is right, not what it is wrong about. The
# matrices below say which pair of classes each model confuses, which is the difference
# between a model that is unsure and one that has collapsed a class. Look for whether the
# large off-diagonal cells sit in the same places across the three panels: shared error
# structure is a property of the task and the labels, and it bounds what any of these models
# can do on it.

# %%
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=FIGSIZE["triple_h_tall"])

labels = ["negative", "neutral", "positive"]

for ax, (name, r) in zip(axes, results.items(), strict=True):
    cm = confusion_matrix(r["y_true"], r["y_pred"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 7},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    # Each model's score is in the table above, not repeated in the title, so a re-run does
    # not leave a figure asserting a number the table has already moved.
    ax.set_title(name, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(labelsize=7)

fig.suptitle("Test-set confusion matrices by model")
show_with_alt(
    fig,
    "Three heatmaps side by side, one per model, each a three-by-three grid of counts with "
    "the true class down the side and the predicted class across the bottom. In all three "
    "the diagonal cells carry much larger counts than anything off it, and the off-diagonal "
    "counts that are not near zero sit in the same cells in each panel rather than in "
    "different places.",
)

# %% [markdown]
# ## Key takeaways
#
# 1. **Check what a published checkpoint was trained on before you test it.** `ProsusAI/finbert`
#    was fine-tuned on the whole Financial PhraseBank, so a test split drawn from that corpus
#    measures it on sentences it has already seen. Nothing in the training code, the metrics
#    or the confusion matrix reveals this; it is on the model card, and reading the card is
#    the step. A domain checkpoint is the first thing anyone reaches for, which is exactly why
#    this trap is common.
# 2. **A leaked comparison does not announce itself as one.** The contaminated row sits in the
#    same table as the clean ones and scores highest. Carry the provenance into the
#    table as a column rather than into a caveat at the end of the notebook, because the table
#    is what gets read and quoted.
# 3. **Cross-notebook before-and-after comparisons need the same checkpoint on both sides.**
#    `03_sentiment_evolution` scores `yiyanghkust/finbert-tone`, trained on analyst reports.
#    Reading its number against this notebook's `ProsusAI/finbert` compares two different
#    models, not one model before and after fine-tuning.
# 4. **Accuracy on an imbalanced corpus flatters the majority class.** Macro F1 weights the
#    three classes equally, and the confusion matrix says which pair a model actually confuses.
#    Read all three; a single scalar cannot distinguish a model that is unsure from one that
#    has stopped predicting a class.
# 5. **Fine-tuning cost and fine-tuning benefit are not on the same scale here.** The training
#    times differ from one another by more than the scores do, so on a task this size the
#    choice between these checkpoints is closer to an engineering decision than a modelling one.
#
# ### The scope these numbers have
#
# One stratified split of one corpus, at one seed. Transfer to text from another source is
# measured in `06_finbert_cross_dataset`, and whether any of this carries a tradable signal in
# `07_news_return_signals` and `09_filing_text_signals`. A difference of a point or two
# between two models on a single split of a few hundred test sentences is inside the range a
# different seed moves them.

# %%
for r in results.values():
    held_out = "held-out" if not r["saw_phrasebank"] else "NOT held out, checkpoint saw this corpus"
    print(f"{r['model_name']}: accuracy {r['accuracy']:.1%}, macro F1 {r['f1']:.3f} ({held_out})")
    print(f"  {r['num_params'] / 1e6:.1f}M parameters, {r['train_time']:.0f}s to fine-tune")
