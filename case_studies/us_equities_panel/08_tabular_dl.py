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
# # Tabular Deep Learning — US Equities Panel
#
# TabM applies attention-style ensembling over the same 71-feature matrix used by
# linear models and GBMs. On the US equities panel, where both linear and GBM
# produce nearly identical daily predictions, the question is
# whether neural network expressiveness discovers interactions that tree-based
# methods miss. With 3,199 stocks and 9.2 million training rows, the dataset is
# large enough for deep learning to train properly --- but whether the
# cross-sectional signal has sufficient structure to reward the additional
# complexity remains to be seen.
#
# **Learning Objectives**:
# - Test whether attention-based tabular models improve on GBMs for broad panels
# - Compare TabM sizes (small/medium/large) on a 71-feature daily equity dataset
# - Evaluate whether model complexity helps when per-stock signal is weak (IC~0.02)
# - Generate backtesting-ready predictions from the best configuration
#
# **Book Reference**: Chapter 12, Section 12.3 (Deep Learning Alternatives)
#
# **Prerequisites**: `03_financial_features.py`, `04_temporal.py`, [`05_evaluation`](05_evaluation.ipynb)

# %%
"""Tabular DL Grid Search — TabM / TabPFN via walk-forward CV."""

import warnings

import polars as pl
import torch
import yaml

from case_studies.utils.tabular_dl import run_tabm_cv
from utils.modeling import load_configs, load_modeling_dataset
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
PRIMARY_LABEL = ""  # Read from setup.yaml if empty
MAX_SYMBOLS = 0
FORCE_RETRAIN = False  # Set True to retrain configs that already have complete hashes
PREDICTION_SPLIT = "validation"
N_EPOCHS = 100
BATCH_SIZE = 4096
MAX_FOLDS = 0

# %%
# Resolve config from setup.yaml
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

if not PRIMARY_LABEL:
    PRIMARY_LABEL = setup["labels"]["primary"]
    print(f"Label from setup.yaml: {PRIMARY_LABEL}")
else:
    print(f"Label override: {PRIMARY_LABEL}")

tdl_config = setup.get("modeling", {}).get("tabular_dl", {})
MODELS = tdl_config.get("models", ["tabm"])
DEVICE = tdl_config.get("device", "gpu")

include_tabpfn = "tabpfn" in MODELS

device_str = "cuda" if DEVICE == "gpu" and torch.cuda.is_available() else "cpu"
print(f"Case study: {CASE_STUDY_ID}")
print(f"Device: {device_str} | Models: {MODELS}")
print(f"Epochs: {N_EPOCHS} | Batch: {BATCH_SIZE}")

# %% [markdown]
# ## 1. Load Artifacts
#
# Same 71-feature dataset as linear and GBM. The 16 walk-forward folds (10-year
# train, 1-year test) provide stable estimates --- though TabM's epoch-based
# training means runtime scales with the number of epochs rather than tree count.

# %%
mds = load_modeling_dataset(CASE_STUDY_ID, PRIMARY_LABEL, max_symbols=MAX_SYMBOLS)

dataset = mds.dataset
feature_names = mds.feature_names
label_col = mds.label_col
date_col = mds.date_col
entity_col = mds.entity_cols[0] if mds.entity_cols else "symbol"
splits = mds.splits
if MAX_FOLDS:
    splits = splits[:MAX_FOLDS]
n_features = len(feature_names)

print(f"Dataset: {len(dataset):,} rows × {n_features} features")
print(f"Label: {label_col} | Date: {date_col} | Entity: {entity_col}")
for s in splits:
    print(
        f"  Fold {s['fold']}: train {str(s['train_start'])[:10]}\u2192{str(s['train_end'])[:10]}  "
        f"val {str(s['val_start'])[:10]}\u2192{str(s['val_end'])[:10]}"
    )

# %% [markdown]
# ## 1b. Data Diagnostics

# %%
dataset_pd = dataset.to_pandas()

label_nans = dataset_pd[label_col].isna().sum()
feat_nan_rate = dataset_pd[feature_names].isna().mean().mean()
n_entities = dataset_pd[entity_col].nunique()

print(f"Entities: {n_entities}")
print(f"Label NaN: {label_nans:,} / {len(dataset_pd):,} ({label_nans / len(dataset_pd):.1%})")
print(f"Feature NaN rate: {feat_nan_rate:.1%}")

# %% [markdown]
# ## 2. Build Grid
#
# TabM configurations: small (64h×4m), medium (128h×8m), large (256h×16m).
# Optionally includes TabPFN (foundation model, subsampled to 2K training rows).

# %%
tabdl_configs = load_configs(CASE_STUDY_ID, PRIMARY_LABEL, "tabular_dl")

# Apply Papermill overrides to configs (test mode: fewer epochs)
for cfg in tabdl_configs:
    if cfg.get("n_epochs", 100) != N_EPOCHS:
        cfg["n_epochs"] = N_EPOCHS
    if cfg.get("batch_size", 4096) != BATCH_SIZE:
        cfg["batch_size"] = BATCH_SIZE

print(f"Grid: {len(tabdl_configs)} configs × {N_EPOCHS} epochs × {len(splits)} folds")
for cfg in tabdl_configs:
    name = cfg["config_name"]
    params = cfg.get("params", {})
    if name.startswith("tabpfn"):
        print(f"  {name:15s}  max_samples={params.get('max_samples', 2000)}")
    else:
        print(
            f"  {name:15s}  hidden={params['hidden_dim']}  "
            f"members={params['n_members']}  dropout={params['dropout']}"
        )

# %% [markdown]
# ## 3. Run Tabular DL CV
#
# Walk-forward training with IC evaluation at epoch checkpoints.

# %%
result = run_tabm_cv(
    dataset_pd,
    splits,
    feature_names=feature_names,
    label_col=label_col,
    date_col=date_col,
    entity_col=entity_col,
    configs=tabdl_configs,
    n_features=n_features,
    device=device_str,
    save_dir=CASE_DIR / "run_log" / "training" / "tabular_dl",
    register=True,
    force_retrain=FORCE_RETRAIN,
    prediction_split=PREDICTION_SPLIT,
    case_study=CASE_STUDY_ID,
    notebook="08_tabular_dl",
    temporal_by_fold=mds.temporal_by_fold,
    temporal_keys=mds.temporal_keys,
    temporal_feature_names=mds.temporal_feature_names,
)

# %% [markdown]
# ## 4. Grid Results

# %%
grid_results = result["grid_results"]
best_name = result["best_config_name"]
best_epoch = result["best_epoch"]
best_ic = result["best_ic"]

print(f"{'Config':15s} {'Best Epoch':>10s} {'Peak IC':>10s} {'Time':>8s}")
print(f"{'-' * 48}")
for r in grid_results:
    marker = " *" if r["config_name"] == best_name else ""
    print(
        f"{r['config_name']:15s} {r['best_epoch']:10d} {r['best_ic']:+10.4f} "
        f"{r['elapsed_s']:7.1f}s{marker}"
    )
print(f"{'-' * 48}")
print(f"Best: {best_name} @ epoch {best_epoch} (IC={best_ic:+.4f})")

# %% [markdown]
# ## 5. Learning Curves

# %%
curves = result["all_learning_curves"]
if curves.height > 0:
    checkpoints = sorted(curves["epoch"].unique().to_list())
    display_cps = [cp for cp in checkpoints if cp % 50 == 0 or cp == checkpoints[-1]]

    print(f"\n{'Config':15s}", end="")
    for cp in display_cps:
        print(f" {cp:>7d}", end="")
    print()
    print("-" * (15 + 8 * len(display_cps)))

    for r in grid_results:
        cfg_data = curves.filter(pl.col("config") == r["config_name"])
        print(f"{r['config_name']:15s}", end="")
        for cp in display_cps:
            row = cfg_data.filter(pl.col("epoch") == cp)
            if row.height > 0:
                ic_val = row["ic_mean"][0]
                print(f" {ic_val:+7.4f}", end="")
            else:
                print(f" {'N/A':>7s}", end="")
        print()

# %% [markdown]
# ## 6. Fold Metrics

# %%
fold_metrics = result["fold_metrics"]
if fold_metrics.height > 0:
    print(f"\nPer-fold IC ({best_name}):")
    for row in fold_metrics.iter_rows(named=True):
        print(f"  Fold {row['fold_id']}: IC={row['ic_mean']:+.4f}  n_test={row['n_test']:,}")
    mean_ic = fold_metrics["ic_mean"].mean()
    print(f"\n  Mean IC: {mean_ic:+.4f}")

# %% [markdown]
# ## 7. Save Results
#
# Predictions and fold metrics are registered by `run_tabm_cv()`
# during training. Here we record the pipeline results JSON.

# %%
predictions = result["predictions"]
all_predictions = result["all_predictions"]

print(f"predictions.parquet: {predictions.height:,} rows")
print(f"all_predictions.parquet: {all_predictions.height:,} rows")
if curves.height > 0:
    print(f"learning_curves.parquet: {curves.height:,} rows")
if fold_metrics.height > 0:
    print(f"fold_metrics.parquet: {fold_metrics.height} rows")

# %%
# Pipeline results JSON
grid_summary = {
    r["config_name"]: {
        "best_epoch": r["best_epoch"],
        "best_ic": round(r["best_ic"], 6),
        "elapsed_s": round(r["elapsed_s"], 1),
    }
    for r in grid_results
}

val_ic_mean = float(fold_metrics["ic_mean"].mean()) if fold_metrics.height > 0 else None

# %% [markdown]
# ## 8. Key Takeaways
#
# On the broadest equity panel, tabular DL faces the same challenge as GBMs:
# the cross-sectional signal in daily returns is predominantly linear, leaving
# limited room for non-linear models to add value. Attention over 71 features
# × 3,199 stocks creates a high-dimensional interaction space, but the marginal
# IC per stock (~0.02) may not contain enough structure to reward the additional
# parameters. The consistent theme: breadth --- not model sophistication ---
# drives the tradable edge in this dataset.
#
# **Next**: [`09_dl_nlinear`](09_dl_nlinear.ipynb) tests whether a minimal
# temporal baseline can improve on flat-feature tabular modeling before the
# broader sequence-model block.
# extract signal that flat-feature models miss.

# %%
print(f"\n{'=' * 60}")
print(f"Tabular DL Grid Search: {CASE_STUDY_ID}")
print(f"{'=' * 60}")
print(f"Features: {n_features}  |  Folds: {len(splits)}  |  Label: {label_col}")
print(f"Device: {device_str}  |  Epochs: {N_EPOCHS}")
print(f"Grid: {len(tabdl_configs)} configs ({', '.join(MODELS)})")
print(f"{'-' * 60}")
print(f"Best config: {best_name} @ epoch {best_epoch}")
print(f"Validation IC (cross-sectional): {best_ic:+.4f}")
if val_ic_mean is not None:
    print(f"Mean fold IC: {val_ic_mean:+.4f}")
