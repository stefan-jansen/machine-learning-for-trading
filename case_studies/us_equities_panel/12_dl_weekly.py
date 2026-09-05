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

# %%
"""Weekly-frequency DL experiment: direct regression vs 1-step forecasting.

Subsamples daily features/labels to Friday frequency, then compares:
- LSTM, NLinear (direct regression, lookback=12 weeks)
- DARTS N-BEATS (1-step-ahead forecasting, lookback=12 weeks)

Sampling on Fridays makes fwd_ret_5d a 1-step-ahead prediction at that
frequency, which removes the error compounding that makes multi-step daily
forecasting ineffective for cross-sectional ranking. It does not make the
windows non-overlapping; see the note in the loading section.
"""

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

# %%
CASE_STUDY_ID = "us_equities_panel"
PRIMARY_LABEL = "fwd_ret_5d"
NOTEBOOK = "12_dl_weekly"

MAX_TRAIN_SEQUENCES = 200_000  # Lower cap for weekly (fewer total sequences)

# %% tags=["parameters"]
BATCH_SIZE = 2048
LOOKBACK = 12  # 12 weekly observations, so ~3 months
MAX_SYMBOLS = 0
FORCE_RETRAIN = False  # Set True to retrain configs that already have complete hashes
PREDICTION_SPLIT = "validation"
N_EPOCHS = 50  # Shorter training: weekly data has fewer samples per fold
MAX_FOLDS = 4  # Quick experiment: 4 evenly-spaced folds from the 16-fold CV

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
# panel's own span, 1990-01-02 to 2018-03-27, of 1,422 consecutive Friday pairs: 1,180 (83.0%)
# close exactly on the next Friday, 196 (13.8%) close after it and overlap, and 46 (3.2%)
# close before it because the next Friday was itself a holiday and carries no row. The rate is
# a property of the exchange calendar rather than of this sample: over 2000-02-01 to
# 2018-03-26 alone it is 82.3%, 14.5% and 3.2%.
#
# What that costs is the mechanical autocorrelation overlapping windows induce, on about one
# week in seven, always by exactly one session. It is small enough to leave the one-step
# formulation intact and too large to describe as absent. Sampling every fifth session instead
# of every Friday would remove it exactly, at the cost of a grid that drifts across weekdays
# and a cadence nobody trades.
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

# Subsample to symbol filter if needed. Selecting by name takes whichever symbols sort first,
# which says nothing about how much history they carry: on this panel the alphabetical head is
# A, AA, AAL, AAMC, AAN, and AAL and AAMC only list part-way through the fold range. A reduced
# run then builds folds whose training window is empty for most of the universe, every fold falls
# under the sequence floor in prepare_fold_sequence_stores, and the run dies on "No valid folds
# created". reduce_to_top_entities takes the symbols with the most rows, ties broken by name, and
# is what every other reduced notebook here uses.
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
# ## Run Direct Regression Models (LSTM, NLinear)
#
# These models take a lookback window of 12 weekly feature vectors and
# predict `fwd_ret_5d` directly as a scalar. The sequence provides temporal
# context; the output is the cross-sectional ranking signal.

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
    # feature_names is in the identity, not only in the training call. Without identity_params or
    # input_data_spec, _config_identity_params returns None (deep_learning.py:1723-1747) and
    # build_training_spec hashes family, config, label, n_folds, n_epochs and the preset params -
    # so changing what the model trains on leaves the spec hash where it was. With
    # FORCE_RETRAIN False the pre-filter at :1765-1782 then finds the previous run complete and
    # skips it, and the notebook reports the old model's numbers under the new feature set. A
    # clean registry retrains and looks correct, which is why this is invisible locally.
    identity_params={"feature_names": feature_names},
    input_data_spec=INPUT_LINEAGE,
    case_study=CASE_STUDY_ID,
    notebook=NOTEBOOK,
)

# %%
print("\nPyTorch results:")
print(f"  Best config: {pytorch_result['best_config_name']}")
print(f"  Best epoch: {pytorch_result['best_epoch']}")
print(f"  Best IC: {pytorch_result['best_ic']:.4f}")

with open(LOG_FILE, "a") as f:
    f.write(
        f"Best: {pytorch_result['best_config_name']} "
        f"IC={pytorch_result['best_ic']:.4f} "
        f"epoch={pytorch_result['best_epoch']}\n"
    )
    for r in pytorch_result["grid_results"]:
        f.write(f"  {r['config_name']}: IC={r['best_ic']:.4f} epoch={r['best_epoch']}\n")

# %% [markdown]
# ## Run Darts `NBEATSModel` (1-Step Forecasting)
#
# With `darts_output_chunk_length=1`, `NBEATSModel` predicts a single weekly
# return, which eliminates the error compounding that degrades multi-step
# daily forecasting. Both arms read the same rows over the same horizon and differ only in
# the formulation, forecasting against direct regression.

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
    # feature_names is in the identity, not only in the training call. Without identity_params or
    # input_data_spec, _config_identity_params returns None (deep_learning.py:1723-1747) and
    # build_training_spec hashes family, config, label, n_folds, n_epochs and the preset params -
    # so changing what the model trains on leaves the spec hash where it was. With
    # FORCE_RETRAIN False the pre-filter at :1765-1782 then finds the previous run complete and
    # skips it, and the notebook reports the old model's numbers under the new feature set. A
    # clean registry retrains and looks correct, which is why this is invisible locally.
    identity_params={"feature_names": feature_names},
    input_data_spec=INPUT_LINEAGE,
    case_study=CASE_STUDY_ID,
    notebook=NOTEBOOK,
    prediction_split=PREDICTION_SPLIT,
)

# %%
print("\nDARTS N-BEATS results:")
print(f"  Best epoch: {darts_result['best_epoch']}")
print(f"  Best IC: {darts_result['best_ic']:.4f}")

with open(LOG_FILE, "a") as f:
    f.write(f"N-BEATS: IC={darts_result['best_ic']:.4f} epoch={darts_result['best_epoch']}\n")

# %% [markdown]
# ## Summary
#
# Compare direct regression (LSTM, NLinear) against 1-step `NBEATSModel` forecasting,
# and against the tabular baselines (GBM, Ridge, TabM) already in the registry.

# %%
print("\n" + "=" * 60)
print("WEEKLY DL EXPERIMENT RESULTS")
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
