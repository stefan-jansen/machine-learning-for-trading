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
# # NLinear — NASDAQ-100 Microstructure
#
# NLinear is the minimal temporal baseline for this case study. It asks whether
# the intraday microstructure edge is already captured by last-value
# normalization plus a single linear map before we move to recurrent,
# convolutional, and attention-based architectures.
#
# **Learning Objectives**:
# - Establish a standalone DL baseline for the NASDAQ microstructure case
# - Compare the simplest temporal model against prior linear and GBM results
# - Prepare a clean provenance chain for later DL architecture comparisons
#
# **Book Reference**: Chapter 13
#
# **Prerequisites**: [`06_linear`](06_linear.ipynb), [`07_gbm`](07_gbm.ipynb)

# %%
"""NLinear — nasdaq100_microstructure deep learning."""

import warnings

import polars as pl
import torch
import yaml

from case_studies.utils.analytics import load_best_ic_per_family
from case_studies.utils.deep_learning import run_dl_cv
from utils.modeling import append_holdout_fold_if_needed, load_configs, load_modeling_dataset
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
MODEL = "nlinear"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
N_EPOCHS = 100
LOOKBACK = 60
BATCH_SIZE = 2048
MAX_TRAIN_SEQUENCES = 750_000
MAX_FOLDS = 0
FOLD_IDS = []
FORCE_RETRAIN = False  # Set True to retrain configs that already have complete hashes
PREDICTION_SPLIT = "validation"

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

if not PRIMARY_LABEL:
    PRIMARY_LABEL = setup["labels"]["primary"]
    print(f"Label from setup.yaml: {PRIMARY_LABEL}")
else:
    print(f"Label override: {PRIMARY_LABEL}")

dl_config = setup.get("modeling", {}).get("dl", {})
DEVICE = dl_config.get("device", "gpu")
device_str = "cuda" if DEVICE == "gpu" and torch.cuda.is_available() else "cpu"

print(f"Case study: {CASE_STUDY_ID} | Model: {MODEL}")
print(f"Device: {device_str} | Epochs: {N_EPOCHS} | Lookback: {LOOKBACK}")

# %% [markdown]
# ## 1. Load Data

# %%
mds = load_modeling_dataset(CASE_STUDY_ID, PRIMARY_LABEL, max_symbols=MAX_SYMBOLS)
append_holdout_fold_if_needed(mds, PREDICTION_SPLIT, CASE_STUDY_ID)

dataset = mds.dataset
feature_names = mds.feature_names
label_col = mds.label_col
date_col = mds.date_col
entity_col = mds.entity_cols[0] if mds.entity_cols else "symbol"
splits = mds.splits[:MAX_FOLDS] if MAX_FOLDS else mds.splits
n_features = len(feature_names)

print(f"Dataset: {len(dataset):,} rows × {n_features} features")
print(f"Label: {label_col} | Entity: {entity_col} | Folds: {len(splits)}")

dataset_pd = dataset.to_pandas()
print(f"Entities: {dataset_pd[entity_col].nunique()}")

# %% [markdown]
# ## 2. Prior Baselines

# %%
prior_baselines = {}
_baselines = load_best_ic_per_family(["linear", "gbm"], case_studies=[CASE_STUDY_ID])
if not _baselines.is_empty():
    for row in _baselines.iter_rows(named=True):
        if row["family"] == "linear":
            prior_baselines[f"{row['config_name'].title()} (Ch11)"] = row["ic_mean"]
        elif row["family"] == "gbm":
            prior_baselines["GBM (Ch12)"] = row["ic_mean"]

if prior_baselines:
    for name, ic in prior_baselines.items():
        print(f"  {name}: IC={ic:+.4f}" if ic is not None else f"  {name}: IC=N/A")
else:
    print("  No prior results found — run 06_linear.py and 07_gbm.py first")

# %% [markdown]
# ## 3. NLinear

# %%
dl_configs = load_configs(CASE_STUDY_ID, PRIMARY_LABEL, "deep_learning")
dl_configs = [c for c in dl_configs if c["params"].get("architecture") == MODEL]
if not dl_configs:
    raise ValueError(
        f"No '{MODEL}' configs found — add '{MODEL}' under 'deep_learning:' in the label config"
    )

for cfg in dl_configs:
    cfg["n_epochs"] = N_EPOCHS
    cfg["batch_size"] = BATCH_SIZE
    cfg["params"]["lookback"] = LOOKBACK

print(
    f"Grid: {len(dl_configs)} configs × {dl_configs[0].get('n_epochs', 100)} epochs × "
    f"{len(splits)} folds"
)
for cfg in dl_configs:
    print(f"  {cfg['config_name']}: {cfg['params'].get('architecture', '?')}")

# %%
result = run_dl_cv(
    dataset_pd,
    splits,
    feature_names=feature_names,
    label_col=label_col,
    date_col=date_col,
    entity_col=entity_col,
    configs=dl_configs,
    n_features=n_features,
    device=device_str,
    save_dir=CASE_DIR / "run_log" / "training" / "deep_learning",
    register=True,
    force_retrain=FORCE_RETRAIN,
    prediction_split=PREDICTION_SPLIT,
    case_study=CASE_STUDY_ID,
    notebook="08_dl_nlinear",
    max_train_sequences=MAX_TRAIN_SEQUENCES,
    selected_folds=FOLD_IDS or None,
    temporal_by_fold=mds.temporal_by_fold,
    temporal_keys=mds.temporal_keys,
    temporal_feature_names=mds.temporal_feature_names,
)

# %% [markdown]
# ## 4. Learning Curves

# %%
grid_results = result["grid_results"]
best_name = result["best_config_name"]
best_epoch = result["best_epoch"]
best_ic = result["best_ic"]

curves = result["all_learning_curves"]
if curves.height > 0:
    checkpoints = sorted(curves["epoch"].unique().to_list())
    display_cps = [cp for cp in checkpoints if cp % 20 == 0 or cp == checkpoints[-1]]

    print(f"{'Config':15s}", end="")
    for cp in display_cps:
        print(f" {cp:>7d}", end="")
    print()
    print("-" * (15 + 8 * len(display_cps)))

    for row in grid_results:
        cfg_data = curves.filter(pl.col("config") == row["config_name"])
        print(f"{row['config_name']:15s}", end="")
        for cp in display_cps:
            ep_row = cfg_data.filter(pl.col("epoch") == cp)
            print(f" {ep_row['ic_mean'][0]:+7.4f}" if ep_row.height > 0 else "     N/A", end="")
        print()

# %% [markdown]
# ## 5. Comparison

# %%
comparison_rows = [{"Model": name, "IC": ic} for name, ic in prior_baselines.items()]
comparison_rows.append({"Model": best_name, "IC": best_ic})
comparison = pl.DataFrame(comparison_rows).with_columns(
    pl.when(pl.col("IC") == pl.col("IC").max())
    .then(pl.lit("*"))
    .otherwise(pl.lit(""))
    .alias("Best")
)
comparison

# %% [markdown]
# ## 6. Save Results

# %%
predictions = result["predictions"]
all_predictions = result["all_predictions"]
fold_metrics = result["fold_metrics"]

print(f"Predictions: {predictions.height:,} rows")
print(f"All predictions: {all_predictions.height:,} rows")
val_ic_mean = float(fold_metrics["ic_mean"].mean()) if fold_metrics.height > 0 else None

# %% [markdown]
# ## 7. Key Takeaways
#
# This notebook gives NLinear its own provenance in the NASDAQ sequence. If the
# later architectures outperform it, that gain is now attributable to the model
# itself rather than to an inline baseline buried inside another notebook.
