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

The key insight: at weekly frequency, fwd_ret_5d becomes a non-overlapping
1-step-ahead prediction. This removes the error-compounding problem that
makes multi-step daily forecasting ineffective for cross-sectional ranking.
"""

# %%
import os
import shutil
import warnings
from gc import collect
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import polars as pl
import torch

from utils.modeling import RANDOM_SEED, load_configs, seed_everything
from utils.paths import get_case_study_dir

# %%
CASE_STUDY_ID = "us_equities_panel"
PRIMARY_LABEL = "fwd_ret_5d"
NOTEBOOK = "12_dl_weekly"

# Weekly experiment parameters
LOOKBACK_WEEKS = 12  # 12 weekly observations = ~3 months
MAX_FOLDS = 4  # Quick experiment: 4 evenly-spaced folds from the 16-fold CV
MAX_TRAIN_SEQUENCES = 200_000  # Lower cap for weekly (fewer total sequences)
N_EPOCHS = 50  # Shorter training — weekly data has fewer samples per fold

# %% tags=["parameters"]
BATCH_SIZE = 2048
LOOKBACK = 12
MAX_SYMBOLS = 0
FORCE_RETRAIN = False  # Set True to retrain configs that already have complete hashes
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
# We subsample the daily feature/label data to Fridays (weekday=4 in pandas).
# At Friday frequency, `fwd_ret_5d` represents a non-overlapping
# Friday-to-Friday return — each observation is informationally independent.

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

feat_cols = [c for c in feat.columns if c not in ("symbol", "timestamp")]
mb_cols = [c for c in mb.columns if c not in ("symbol", "timestamp")]
features = feat.join(mb, on=["symbol", "timestamp"], how="inner")
feature_names = feat_cols + mb_cols
print(f"  Combined: {features.shape[0]:,} rows, {len(feature_names)} features")

labels = (
    pl.scan_parquet(CASE_DIR / "labels" / f"{PRIMARY_LABEL}.parquet")
    .filter(weekly_filter)
    .collect(streaming=True)
)
print(f"  Weekly labels: {labels.shape[0]:,} rows")

dataset = features.join(labels, on=["symbol", "timestamp"], how="inner")
del feat, mb, features, labels
collect()

# Subsample to symbol filter if needed
if MAX_SYMBOLS > 0:
    symbols = dataset.select("symbol").unique().sort("symbol").head(MAX_SYMBOLS)
    dataset = dataset.join(symbols, on="symbol")
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
# We use 4 evenly-spaced folds from the full date range, each with
# ~8-10 years of training and ~1 year of validation.

# %%
# Build 4 manual splits spanning 2000-2018
fold_specs = [
    {
        "fold": 0,
        "train_start": "2000-01-01",
        "train_end": "2010-01-01",
        "val_start": "2010-02-01",
        "val_end": "2011-01-01",
    },
    {
        "fold": 1,
        "train_start": "2002-01-01",
        "train_end": "2012-01-01",
        "val_start": "2012-02-01",
        "val_end": "2013-01-01",
    },
    {
        "fold": 2,
        "train_start": "2005-01-01",
        "train_end": "2015-01-01",
        "val_start": "2015-02-01",
        "val_end": "2016-01-01",
    },
    {
        "fold": 3,
        "train_start": "2007-01-01",
        "train_end": "2017-01-01",
        "val_start": "2017-02-01",
        "val_end": "2018-01-01",
    },
]

# Limit folds if requested
splits = fold_specs[:MAX_FOLDS]
print(f"CV splits: {len(splits)} folds")
for s in splits:
    n_train = dataset_pd[
        (dataset_pd["timestamp"] >= s["train_start"]) & (dataset_pd["timestamp"] < s["train_end"])
    ].shape[0]
    n_val = dataset_pd[
        (dataset_pd["timestamp"] >= s["val_start"]) & (dataset_pd["timestamp"] < s["val_end"])
    ].shape[0]
    print(f"  Fold {s['fold']}: train={n_train:,}, val={n_val:,}")

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

pytorch_result = run_dl_cv(
    dataset_pd,
    splits,
    configs=pytorch_configs,
    n_features=len(feature_names),
    feature_names=feature_names,
    label_col=PRIMARY_LABEL,
    date_col="timestamp",
    entity_col="symbol",
    device=device,
    save_dir=PYTORCH_SAVE_DIR,
    max_train_sequences=MAX_TRAIN_SEQUENCES,
    register=True,
    force_retrain=FORCE_RETRAIN,
    prediction_split=PREDICTION_SPLIT,
    case_study=CASE_STUDY_ID,
    notebook=NOTEBOOK,
)

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
# ## Run DARTS N-BEATS (1-Step Forecasting)
#
# With `darts_output_chunk_length=1`, N-BEATS predicts a single weekly
# return — eliminating the error compounding that degrades multi-step
# daily forecasting. This is the fair comparison: same horizon, same data,
# but the forecasting formulation vs direct regression.

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

darts_result = run_dl_cv(
    dataset_pd,
    splits,
    configs=darts_configs,
    n_features=len(feature_names),
    feature_names=feature_names,
    label_col=PRIMARY_LABEL,
    date_col="timestamp",
    entity_col="symbol",
    device=device,
    save_dir=DARTS_SAVE_DIR,
    max_train_sequences=MAX_TRAIN_SEQUENCES,
    register=True,
    force_retrain=FORCE_RETRAIN,
    case_study=CASE_STUDY_ID,
    notebook=NOTEBOOK,
    prediction_split=PREDICTION_SPLIT,
)

print("\nDARTS N-BEATS results:")
print(f"  Best epoch: {darts_result['best_epoch']}")
print(f"  Best IC: {darts_result['best_ic']:.4f}")

with open(LOG_FILE, "a") as f:
    f.write(f"N-BEATS: IC={darts_result['best_ic']:.4f} epoch={darts_result['best_epoch']}\n")

# %% [markdown]
# ## Summary
#
# Compare direct regression (LSTM, NLinear) against 1-step N-BEATS forecasting,
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
