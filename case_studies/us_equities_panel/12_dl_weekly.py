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
# # US equities panel: predicting a week ahead instead of five days ahead
#
# The three sequence notebooks before this one - [`09_dl_nlinear`](09_dl_nlinear.ipynb),
# [`10_dl_lstm`](10_dl_lstm.ipynb) and [`11_dl_tsmixer`](11_dl_tsmixer.ipynb) - read a window of
# daily rows and predict a return five sessions ahead. That is a **multi-step** problem: the
# quantity being predicted spans five periods of the grid the model reads. This notebook changes
# the grid rather than the models, and asks whether that alone is worth anything.
#
# **Sampling on Fridays makes the same label a one-step problem.** `fwd_ret_5d` is the return over
# the next five *sessions*, and a full trading week holds exactly five, so on a Friday-only grid
# each row's label lands on about the next row. Nothing about the label changed; what changed is
# how many steps of the model's own grid it spans. Two things follow, and they pull in opposite
# directions. A one-step target avoids the error compounding that makes a multi-step forecast
# progressively vaguer, and the reader should expect that to help. Against it, subsampling to one
# day in five throws away four fifths of the rows, and a sequence model on a smaller sample is a
# weaker fit. The experiment is worth running because neither effect is obviously larger.
#
# **Two formulations of the same task are compared.** Both read a **lookback** window of the
# twelve most recent weekly feature vectors for a stock - about three months - and both are scored
# on the same rows over the same horizon.
#
# - **Direct regression** treats the window as fixed-length input to a model whose output is one
#   number, the return. `lstm_h64` is a two-layer recurrent network that consumes the window one
#   week at a time and carries a hidden state forward; `nlinear` normalizes the window by its last
#   value and applies a single linear map. Neither has any notion that its output is a future
#   value of one of its inputs.
# - **One-step forecasting** treats the window as the history of a time series and asks for its
#   next value. `NBEATSModel`, from the Darts library, is built for that formulation: a stack of
#   blocks that each fit a piece of the signal and pass the remainder to the next.
#   `darts_output_chunk_length=1` is what makes it predict a single week rather than a sequence.
#
# The comparison is therefore between two ways of writing down the same prediction, at a frequency
# where the second is well posed. On a daily grid it would not be: N-BEATS asked for five days
# would compound its own output four times.
#
# **What the reader should carry away is the reframing, not this panel's numbers.** Whether a
# weekly grid helps depends on how much history the panel has and how strongly the signal decays,
# and both differ by market.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Say what makes a prediction one-step rather than multi-step, and why that is a property of the
#   sampling grid rather than of the label.
# - State the two costs of subsampling a daily panel to weekly and say which one a longer history
#   would relieve.
# - Distinguish direct regression from forecasting as two formulations of one prediction, and say
#   what each assumes about the relationship between input and output.
# - Explain why a walk-forward fold has to be resolved from the label file rather than written into
#   the notebook, and what a hand-written window would be free to do.
# - Say what a training identity has to cover before a run may be skipped as already complete, and
#   what goes wrong when the feature set is outside it.
#
# **Book reference**: Chapter 13, Section 13.9. Table 13.5 reports the results this notebook
# produces.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices,
# and [`02_labels`](02_labels.ipynb) has written `fwd_ret_5d`.
#
# **What it writes**: one training run and one validation prediction set per configuration, in
# `run_log/registry.db` and under `run_log/training/`. These predictions sit on a Friday grid and
# are deliberately **not** entered into the canonical backtest pool that
# [`16_backtest`](16_backtest.ipynb) ranks: the daily models it ranks are scored on every session,
# and a Friday-only series competing against them on the same label would be compared on a
# different set of decision dates rather than on a different model.

# %%
"""Weekly-frequency sequence models: direct regression against one-step forecasting."""

# %%
import os
import shutil
import warnings
from gc import collect
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl
import torch

from case_studies.utils.cv_window import modeling_fold_boundaries
from utils.artifact_specs import load_setup_config, resolve_label_buffer
from utils.modeling import (
    RANDOM_SEED,
    build_modeling_input_lineage,
    load_configs,
    reduce_to_top_entities,
    seed_everything,
)
from utils.paths import get_case_study_dir

# %% [markdown]
# ## The values a run can be given
#
# What each one decides:
#
# - **`LOOKBACK`** is how many weekly observations a model sees before predicting. Twelve is about
#   three months, which is long enough to carry a quarter's momentum and short enough that a stock
#   needs only twelve weeks of history before it contributes any training window at all.
# - **`N_EPOCHS`** is how many passes the fit makes over its training rows. Fifty rather than the
#   daily notebooks' longer schedules, because a Friday grid holds a fifth of the rows and a pass
#   over it is a fifth of the gradient steps.
# - **`MAX_FOLDS`** takes that many of the case study's walk-forward folds, evenly spaced and
#   always including the earliest and the most recent. Four rather than all sixteen: this notebook
#   fits four sequence models and the reframing it tests shows up across the sample's span rather
#   than in the density of folds along it.
# - **`MAX_TRAIN_SEQUENCES`** caps how many lookback windows one fold contributes. A cap is what
#   keeps a fold's memory bounded on a three-thousand-name panel; it is lower here than in the
#   daily notebooks because a weekly grid yields fewer windows to begin with.
# - **`BATCH_SIZE`** is how many windows a gradient step averages over. It affects speed and the
#   noise in each step, not what the model can represent.
# - **`MAX_SYMBOLS`** caps the universe, taking the stocks with the most rows. Zero, the default,
#   keeps all of them.
# - **`FORCE_RETRAIN`** discards fitted state and refits configurations the registry already holds
#   as complete. Leave it off unless the identity below has changed in a way the registry cannot
#   see.

# %%
CASE_STUDY_ID = "us_equities_panel"
PRIMARY_LABEL = "fwd_ret_5d"
NOTEBOOK = "12_dl_weekly"

MAX_TRAIN_SEQUENCES = 200_000

# %% tags=["parameters"]
BATCH_SIZE = 2048
LOOKBACK = 12
MAX_SYMBOLS = 0
FORCE_RETRAIN = False
PREDICTION_SPLIT = "validation"
N_EPOCHS = 50
MAX_FOLDS = 4

# %%
seed_everything(RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
SAVE_ROOT = CASE_DIR / "run_log" / "training" / "deep_learning" / NOTEBOOK
PYTORCH_SAVE_DIR = SAVE_ROOT / "pytorch"
DARTS_SAVE_DIR = SAVE_ROOT / "darts"
if FORCE_RETRAIN and SAVE_ROOT.exists():
    shutil.rmtree(SAVE_ROOT)
PYTORCH_SAVE_DIR.mkdir(parents=True, exist_ok=True)
DARTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = Path("/tmp/dl_weekly_experiment.log")

# %% [markdown]
# ## Load and Subsample to Weekly Frequency
#
# The daily features and labels are subsampled to Fridays, so `fwd_ret_5d` becomes a
# one-step-ahead target: each row's window runs to about the next row's date.
#
# **The windows are mostly, not entirely, non-overlapping, and the label is why.**
# `02_labels` resolves `fwd_ret_5d` five *sessions* ahead. A full trading week holds exactly
# five sessions, so a Friday's window closes on the next Friday. A week carrying a market
# holiday holds four, so the fifth session falls on the Monday after that Friday and the
# window overlaps the next observation's by a session. Counted on the NYSE calendar over the
# span the label file covers, 1990-01-30 to 2018-03-20, of 1,417 consecutive Friday pairs:
# 1,175 (82.9%) close exactly on the next Friday, 196 (13.8%) close after it and overlap, and
# 46 (3.2%) close before it because the next Friday was itself a holiday and carries no row.
# The rate is a property of the exchange calendar rather than of this sample: over
# 2000-02-01 to 2018-03-26 alone it is 82.3%, 14.5% and 3.2%.
#
# The overlap is always exactly one session and never more. A holiday week holds four
# sessions, so the fifth falls on the Monday after the next Friday. That is what makes this
# worth describing rather than removing: sampling every fifth session would close it exactly
# and replace it with a grid that drifts across weekdays and a cadence nobody trades.
#
# What that costs is the mechanical autocorrelation overlapping windows induce, on about one
# week in seven. It is small enough to leave the one-step formulation intact and too large to
# describe as absent.
#
# Non-overlap would not buy independence in any case. Returns cluster in volatility and share
# a market factor across the cross-section, so what non-overlap removes is the correlation the
# construction itself imposes, not the dependence in the data.

# %%
# Load only weekly rows before materializing joins. The full daily join OOM-kills the kernel.
print("Loading weekly features...")
weekly_filter = pl.col("timestamp").dt.weekday() == 5

feat = (
    pl.scan_parquet(CASE_DIR / "features" / "financial.parquet")
    .filter(weekly_filter)
    .collect(streaming=True)
)
print(f"  Weekly financial features: {feat.shape[0]:,} rows, {feat.shape[1]} cols")

mb = (
    pl.scan_parquet(CASE_DIR / "features" / "model_based.parquet")
    .filter(weekly_filter)
    .collect(streaming=True)
)
print(f"  Weekly model-based features: {mb.shape[0]:,} rows, {mb.shape[1]} cols")

# model_based.parquet is keyed on (symbol, timestamp): every estimate behind it is bounded by a
# refit schedule rather than by a fold, so a symbol-session carries one value whichever fold reads
# it and the join is a plain left join that multiplies nothing. `temporal_by_fold` is therefore
# None, which is what `load_modeling_dataset` passes on this shape too.
#
# The key's uniqueness is asserted rather than assumed. A repeat would fan every weekly
# observation out to one row per duplicate, leave duplicate timestamps per symbol, and give
# the sequence builder no fold it could form - which surfaces as "No valid folds created"
# several cells later, a long way from the join that caused it.
assert mb.select("symbol", "timestamp").is_duplicated().sum() == 0, (
    "model_based.parquet repeats a symbol and timestamp; the sequence builder would see "
    "duplicate dates per symbol and create no folds"
)
assert "fold" not in mb.columns, (
    "model_based.parquet carries a fold column, which this stage has no key to read it by: "
    "a stock-session is expected to carry one value"
)

feat_cols = [c for c in feat.columns if c not in ("symbol", "timestamp")]
temporal_feature_names = [c for c in mb.columns if c not in ("symbol", "timestamp")]
temporal_by_fold = None
features = feat.join(mb, on=["symbol", "timestamp"], how="left")
# prepare_fold_sequence_stores builds `use_cols` from feature_names, so a temporal column absent
# from this list is joined and then immediately dropped - the model would train on the financial
# features alone and report nothing about it. load_modeling_dataset carries its temporal names in
# feature_names for the same reason.
feature_names = feat_cols + temporal_feature_names
print(
    f"  Base features: {features.shape[0]:,} rows, {len(feat_cols)} financial "
    f"+ {len(temporal_feature_names)} temporal = {len(feature_names)} features"
)

labels = (
    pl.scan_parquet(CASE_DIR / "labels" / f"{PRIMARY_LABEL}.parquet")
    .filter(weekly_filter)
    .collect(streaming=True)
)
print(f"  Weekly labels: {labels.shape[0]:,} rows")

dataset = features.join(labels, on=["symbol", "timestamp"], how="inner")
del feat, mb, features, labels
collect()

# A capped universe takes the stocks with the most rows, ties broken by name. Taking the
# alphabetically first names instead would select on the spelling of a ticker rather than on how
# much history it carries, and a sequence model needs history: a stock that lists part-way through
# a fold contributes no complete lookback window to it. This is the rule every reduced notebook in
# the case study applies, so a capped run here selects the same universe as a capped run elsewhere.
if MAX_SYMBOLS > 0:
    dataset = reduce_to_top_entities(dataset, "symbol", MAX_SYMBOLS)
    print(f"  Filtered to {MAX_SYMBOLS} symbols: {dataset.shape[0]:,} rows")

# Convert to pandas (pipeline expects pandas)
n_symbols = dataset["symbol"].n_unique()
print(f"  Weekly (Friday): {dataset.shape[0]:,} rows, {n_symbols} symbols")
dataset_pd = dataset.to_pandas()
dataset_pd["timestamp"] = pd.to_datetime(dataset_pd["timestamp"])
del dataset
collect()

# %% [markdown]
# ## Create Walk-Forward CV Splits
#
# The folds are the case study's own, resolved from the label file through
# `modeling_fold_boundaries` - the call [`04_model_based_features`](04_model_based_features.ipynb)
# and `load_modeling_dataset` both resolve them with, so fold *k* here is fold *k* everywhere
# else. `MAX_FOLDS` of them are taken, evenly spaced across the sequence and always including
# the earliest and the most recent, because this notebook fits four sequence models on a weekly
# grid and the full sixteen would cost four times what the comparison needs.
#
# They are resolved rather than written out. A window typed into this notebook is a second
# declaration of the fold design, free to disagree with the one every other stage reads and
# free to reach past the holdout without anything noticing. A window derived from the label
# file cannot, and the assertion below is what establishes it rather than the prose.

# %%
SETUP = load_setup_config(CASE_STUDY_ID)
LABEL_BUFFER = resolve_label_buffer(CASE_STUDY_ID, PRIMARY_LABEL, SETUP)
HOLDOUT_START = pd.Timestamp(str(SETUP["evaluation"]["holdout_start"]))

canonical = sorted(
    modeling_fold_boundaries(CASE_STUDY_ID, PRIMARY_LABEL), key=lambda f: f["val_start"]
)
chosen = sorted(
    {round(i) for i in np.linspace(0, len(canonical) - 1, min(MAX_FOLDS, len(canonical)))}
)
splits = [
    {
        "fold": canonical[i]["fold"],
        "train_start": pd.Timestamp(canonical[i]["train_start"]),
        "train_end": pd.Timestamp(canonical[i]["train_end"]),
        "val_start": pd.Timestamp(canonical[i]["val_start"]),
        "val_end": pd.Timestamp(canonical[i]["val_end"]),
    }
    for i in chosen
]

for split in splits:
    assert split["val_end"] < HOLDOUT_START, (
        f"fold {split['fold']} is scored through {split['val_end'].date()}, inside the holdout "
        f"opening {HOLDOUT_START.date()}"
    )

print(f"CV splits: {len(splits)} of the case study's {len(canonical)} folds, evenly spaced")
for s in splits:
    n_train = dataset_pd[
        (dataset_pd["timestamp"] >= s["train_start"]) & (dataset_pd["timestamp"] <= s["train_end"])
    ].shape[0]
    n_val = dataset_pd[
        (dataset_pd["timestamp"] >= s["val_start"]) & (dataset_pd["timestamp"] <= s["val_end"])
    ].shape[0]
    print(
        f"  Fold {s['fold']}: trained on {s['train_start'].date()} to {s['train_end'].date()} "
        f"({n_train:,} weekly rows), scored over {s['val_start'].date()} to "
        f"{s['val_end'].date()} ({n_val:,})"
    )

# %% [markdown]
# ## The identity these runs register under
#
# `run_dl_cv` skips a configuration whose training hash is already complete, so whatever the
# hash is built from is what a re-run is able to notice. Without an input lineage the hash
# covers the family, the configuration, the label, the fold count, the epochs and the feature
# *names* - and a stage-04 artifact regenerated under a different estimation schedule keeps
# every one of its column names. The corrected values would then never reach a model, and the
# notebook would report the previous run's numbers under them, which a clean registry hides
# because everything retrains.
#
# `build_modeling_input_lineage` is the same payload `load_modeling_dataset` builds for the
# sibling notebooks. It digests the three parquet files this one reads and the fold windows it
# runs, so a changed artifact or a changed window is a changed identity.

# %%
INPUT_LINEAGE = build_modeling_input_lineage(
    artifacts={
        "financial": CASE_DIR / "features" / "financial.parquet",
        "model_based": CASE_DIR / "features" / "model_based.parquet",
        "label": CASE_DIR / "labels" / f"{PRIMARY_LABEL}.parquet",
    },
    feature_names=feature_names,
    splits=splits,
    label_buffer=LABEL_BUFFER,
    task_type="regression",
    eval_label_col=None,
    max_symbols=MAX_SYMBOLS,
    symbols=None,
)
print(f"Input lineage fingerprint {INPUT_LINEAGE['fingerprint'][:12]} over")
for name, record in INPUT_LINEAGE["artifacts"].items():
    print(f"  {name}: {record['sha256'][:12]}, {record['size'] / 1e9:.2f} GB")

# %% [markdown]
# ## The direct-regression arm
#
# Both configurations read the same twelve-week window and emit one number, the return, with no
# structure connecting that output to the inputs it was computed from. `lstm_h64` steps through
# the window one week at a time, carrying a hidden state that summarises everything it has seen;
# `nlinear` subtracts the window's last value, applies one linear map to what is left, and adds
# the value back, so it predicts a change from the most recent observation.
#
# What comes out is used as a ranking signal across stocks on a date, not as a return forecast to
# be believed at face value, which is why the scoring below is a rank correlation.

# %%
from case_studies.utils.deep_learning import run_dl_cv

# Build configs manually with weekly-adjusted lookback
pytorch_configs = []
for name, arch in [("lstm_h64", "lstm"), ("nlinear", "nlinear")]:
    cfg = {
        "config_name": name,
        "family": "deep_learning",
        "library": "pytorch",
        "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "checkpoint_interval": 5,
        "params": {
            "architecture": arch,
            "lookback": LOOKBACK,
            "dropout": 0.1,
        },
    }
    if arch == "lstm":
        cfg["params"]["hidden_size"] = 64
        cfg["params"]["n_layers"] = 2
    pytorch_configs.append(cfg)

print(f"Running {len(pytorch_configs)} PyTorch configs on {device}...")
with open(LOG_FILE, "a") as f:
    f.write("=== PyTorch direct regression (weekly) ===\n")

# %%
pytorch_result = run_dl_cv(
    dataset_pd,
    splits,
    configs=pytorch_configs,
    n_features=len(feature_names),
    feature_names=feature_names,
    temporal_by_fold=temporal_by_fold,
    temporal_keys=["symbol", "timestamp"],
    temporal_feature_names=temporal_feature_names,
    label_col=PRIMARY_LABEL,
    date_col="timestamp",
    entity_col="symbol",
    device=device,
    save_dir=PYTORCH_SAVE_DIR,
    max_train_sequences=MAX_TRAIN_SEQUENCES,
    register=True,
    force_retrain=FORCE_RETRAIN,
    prediction_split=PREDICTION_SPLIT,
    # The feature set belongs in the identity, not only in the training call. A run is skipped
    # when the registry already holds its training hash as complete, so anything the hash does
    # not cover is something a re-run cannot notice has changed.
    identity_params={"feature_names": feature_names},
    input_data_spec=INPUT_LINEAGE,
    case_study=CASE_STUDY_ID,
    notebook=NOTEBOOK,
)

# %%
print("\nDirect regression, highest-IC checkpoint per configuration:")
print(f"  configuration: {pytorch_result['best_config_name']}")
print(f"  epoch: {pytorch_result['best_epoch']}")
print(f"  mean validation IC: {pytorch_result['best_ic']:.4f}")

with open(LOG_FILE, "a") as f:
    f.write(
        f"Best: {pytorch_result['best_config_name']} "
        f"IC={pytorch_result['best_ic']:.4f} "
        f"epoch={pytorch_result['best_epoch']}\n"
    )
    for r in pytorch_result["grid_results"]:
        f.write(f"  {r['config_name']}: IC={r['best_ic']:.4f} epoch={r['best_epoch']}\n")

# %% [markdown]
# ## The forecasting arm
#
# `NBEATSModel` treats the window as the history of a series and predicts its next value. Its
# blocks each fit part of the signal and hand the remainder to the next block, so the fit is built
# up as a sum of pieces rather than as one map from window to output.
#
# `darts_output_chunk_length=1` is the setting that makes this one-step. Asked for five steps on a
# daily grid the same model would feed its own output back in four times, and each of those
# passes carries the previous error forward - which is the compounding a weekly grid removes by
# construction rather than by choosing a better model.
#
# Both arms read the same rows over the same horizon with the same lookback, so the difference
# between them is the formulation and nothing else.

# %%
darts_configs = [
    {
        "config_name": "nbeats",
        "family": "deep_learning",
        "library": "darts",
        "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "checkpoint_interval": 5,
        "params": {
            "architecture": "nbeats",
            "lookback": LOOKBACK,
            "hidden_size": 128,
            "n_blocks": 3,
            "n_layers": 4,
            "dropout": 0.1,
            # Weekly-specific: predict 1 step (1 week) instead of default 5 days
            "darts_output_chunk_length": 1,
            "darts_input_chunk_length": LOOKBACK,
        },
    }
]

print(f"Running DARTS N-BEATS (1-step weekly forecasting) on {device}...")
with open(LOG_FILE, "a") as f:
    f.write("\n=== DARTS N-BEATS (weekly, 1-step) ===\n")

# %%
darts_result = run_dl_cv(
    dataset_pd,
    splits,
    configs=darts_configs,
    n_features=len(feature_names),
    feature_names=feature_names,
    temporal_by_fold=temporal_by_fold,
    temporal_keys=["symbol", "timestamp"],
    temporal_feature_names=temporal_feature_names,
    label_col=PRIMARY_LABEL,
    date_col="timestamp",
    entity_col="symbol",
    device=device,
    save_dir=DARTS_SAVE_DIR,
    max_train_sequences=MAX_TRAIN_SEQUENCES,
    register=True,
    force_retrain=FORCE_RETRAIN,
    # The feature set belongs in the identity, not only in the training call. A run is skipped
    # when the registry already holds its training hash as complete, so anything the hash does
    # not cover is something a re-run cannot notice has changed.
    identity_params={"feature_names": feature_names},
    input_data_spec=INPUT_LINEAGE,
    case_study=CASE_STUDY_ID,
    notebook=NOTEBOOK,
    prediction_split=PREDICTION_SPLIT,
)

# %%
print("\nOne-step forecasting, highest-IC checkpoint:")
print(f"  epoch: {darts_result['best_epoch']}")
print(f"  mean validation IC: {darts_result['best_ic']:.4f}")

with open(LOG_FILE, "a") as f:
    f.write(f"N-BEATS: IC={darts_result['best_ic']:.4f} epoch={darts_result['best_epoch']}\n")

# %% [markdown]
# ## What to notice
#
# The table below puts the three weekly fits beside each other, and then beside the daily
# tabular families already registered on the same label. Two comparisons, and they answer
# different questions.
#
# **Within the weekly grid**, direct regression against one-step forecasting is the comparison
# this notebook was built for: same rows, same horizon, same lookback, two ways of writing the
# prediction down.
#
# **Against the daily families**, the comparison is looser and worth reading with that in mind.
# Those models are scored on every session and these on Fridays only, so the two are measured over
# different sets of decision dates. A difference between them is a difference in the experiment as
# much as in the model, which is also why these predictions stay out of the canonical backtest
# pool.
#
# **What generalizes is the reframing.** Whether sampling to a coarser grid pays depends on how
# much history the panel holds and how fast the signal decays, and a reader applying this to their
# own data should expect the balance between the two costs to land differently.

# %%
print("\n" + "=" * 60)
print("Weekly sequence models: validation IC by formulation")
print("=" * 60)

# Collect DL results
all_results = []
for r in pytorch_result["grid_results"]:
    all_results.append(
        {
            "model": r["config_name"],
            "approach": "direct regression",
            "ic": r["best_ic"],
            "epoch": r["best_epoch"],
        }
    )
all_results.append(
    {
        "model": "nbeats (DARTS)",
        "approach": "1-step forecasting",
        "ic": darts_result["best_ic"],
        "epoch": darts_result["best_epoch"],
    }
)

results_df = pl.DataFrame(all_results).sort("ic", descending=True)
print(results_df)

# %%
# Compare against registry baselines
import sqlite3

db_path = CASE_DIR / "run_log" / "registry.db"
if db_path.exists():
    conn = sqlite3.connect(db_path)
    baselines = pd.read_sql_query(
        """
        SELECT t.family, t.config_name, t.label, AVG(f.ic) as mean_ic
        FROM fold_metrics f
        JOIN prediction_sets ps ON f.prediction_hash = ps.prediction_hash
        JOIN training_runs t ON ps.training_hash = t.training_hash
        WHERE t.label = 'fwd_ret_5d'
          AND t.family IN ('linear', 'gbm', 'tabular_dl')
        GROUP BY t.training_hash, t.family, t.config_name, t.label
        ORDER BY mean_ic DESC
        LIMIT 5
    """,
        conn,
    )
    conn.close()

    print("\nBaseline comparison (fwd_ret_5d, daily CV):")
    print(baselines.to_string(index=False))

print("\nDone. Full log at:", LOG_FILE)
