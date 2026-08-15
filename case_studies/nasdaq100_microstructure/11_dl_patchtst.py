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
# # PatchTST - NASDAQ-100 Microstructure
#
# PatchTST cuts the window into short consecutive segments - patches - and treats
# each patch as one item in a sequence. An attention mechanism then lets every
# patch weigh every other patch directly, so the model can relate two stretches
# of the window without passing information through everything in between.
#
# Patching is what makes that affordable. Attention costs grow with the square of
# the number of items it compares, so attending over 60 individual observations
# is far more expensive than attending over a handful of patches covering the
# same window. Patching also gives each item more content than a single
# observation, which matters when any one minute is mostly noise.
#
# The label looks 15 minutes ahead on a one-minute grid, so consecutive windows
# overlap heavily and neighbouring rows are far from independent. That shapes
# everything below: how windows are built, where folds are cut, and how much of
# the training set is sampled.
#
# **Learning Objectives**:
# - Fit an attention-based sequence model on a panel by declaring one request
#   rather than assembling folds and windows in the notebook
# - Read a learning curve across training epochs and say what it shows about
#   capacity and noise
# - Check that a fitted model produced predictions on every fold it was asked
#   for, before any of those predictions are used
#
# **Book Reference**: Chapter 13
#
# **Prerequisites**: [`05_evaluation`](05_evaluation.ipynb)

# %%
"""PatchTST - nasdaq100_microstructure deep learning."""

import warnings

import matplotlib.pyplot as plt
import polars as pl
import torch
import yaml

from case_studies.utils.deep_learning import run_dl_cv
from utils.modeling import append_holdout_fold_if_needed, load_configs, load_modeling_dataset
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

warnings.filterwarnings("ignore")

# %% [markdown]
# ### Settings
#
# `LOOKBACK` is how many one-minute observations enter each window, so 60 gives
# the model the trailing hour. `MAX_TRAIN_SEQUENCES` caps how many windows are
# drawn per fold: every row starts a window, so an uncapped fold would build
# tens of millions of near-identical overlapping sequences. `N_EPOCHS` is how
# many passes are made over that sample, and checkpoints are written along the
# way so the run can be inspected and resumed at a known epoch rather than only
# at the end.
#
# `MAX_FOLDS` and `FOLD_IDS` restrict which walk-forward folds run. They exist
# for previews; a run that uses them covers less of the history than the fold
# plan declares, and the fold set is part of what the run is registered under.

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
MODEL = "patchtst"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
SEED = 42
N_EPOCHS = 100
LOOKBACK = 60
BATCH_SIZE = 2048
MAX_TRAIN_SEQUENCES = 750_000
MAX_FOLDS = 0
FOLD_IDS = []
FORCE_RETRAIN = False  # Set True to retrain configs that already have complete hashes
PREDICTION_SPLIT = "validation"

# %%
set_global_seeds(SEED)
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

if not PRIMARY_LABEL:
    PRIMARY_LABEL = setup["labels"]["primary"]

# The configured device is part of what the run is registered under, so an
# unavailable accelerator stops the run rather than quietly retraining on CPU
# and registering the result as though it were the requested one.
DEVICE = setup.get("modeling", {}).get("dl", {}).get("device", "gpu")
if DEVICE == "gpu" and not torch.cuda.is_available():
    raise RuntimeError(
        f"{CASE_STUDY_ID}/config/setup.yaml requests modeling.dl.device=gpu and no "
        f"CUDA device is visible. Make the GPU available, or set the device to cpu "
        f"in setup.yaml so the change is recorded with the run."
    )
device_str = "cuda" if DEVICE == "gpu" else "cpu"

print(f"Case study: {CASE_STUDY_ID} | architecture: {MODEL}")
print(f"Label: {PRIMARY_LABEL} (from config/setup.yaml)")
print(f"Device: {device_str} (configured {DEVICE})")
print(f"Window: {LOOKBACK} one-minute observations | training epochs: {N_EPOCHS}")

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
# ## 2. Declare the fitting request
#
# Configurations come from the label's config file rather than from literals
# here, and the notebook keeps only the ones whose architecture is the one it is
# about. The three training settings above are then applied to every retained
# configuration, so what is fitted is visible in one place instead of being
# spread between a config file and the runner's defaults.
#
# Nothing about the windows, folds or gaps is assembled here. The runner receives
# the label's fold plan and the observation cadence and derives the rest, so this
# notebook and the ones for the other architectures cannot drift apart in how
# they cut a window.

# %%
dl_configs = load_configs(CASE_STUDY_ID, PRIMARY_LABEL, "deep_learning")
dl_configs = [c for c in dl_configs if c["params"].get("architecture") == MODEL]
if not dl_configs:
    available = sorted(
        {
            c["params"].get("architecture", "?")
            for c in load_configs(CASE_STUDY_ID, PRIMARY_LABEL, "deep_learning")
        }
    )
    raise ValueError(
        f"No '{MODEL}' configuration in the {PRIMARY_LABEL} deep_learning config. "
        f"Declared architectures: {available}"
    )

for cfg in dl_configs:
    cfg["n_epochs"] = N_EPOCHS
    cfg["batch_size"] = BATCH_SIZE
    cfg["params"]["lookback"] = LOOKBACK

print(f"Fitting {len(dl_configs)} {MODEL} configuration(s) on {len(splits)} folds:")
for cfg in dl_configs:
    print(f"  {cfg['config_name']}: window {LOOKBACK}, batch {BATCH_SIZE}, {N_EPOCHS} epochs")

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
    notebook="11_dl_patchtst",
    max_train_sequences=MAX_TRAIN_SEQUENCES,
    selected_folds=FOLD_IDS or None,
    temporal_by_fold=mds.temporal_by_fold,
    temporal_keys=mds.temporal_keys,
    temporal_feature_names=mds.temporal_feature_names,
    seed=SEED,
)

# %% [markdown]
# ## 3. Learning curves
#
# A checkpoint is written every few epochs and scored on the fold's validation
# window, which gives one curve per configuration. The shape is what to read,
# not any single point on it. A curve that climbs and then falls says the model
# has started fitting noise and the useful capacity was reached earlier; a curve
# that stays flat from the first checkpoint says the extra epochs are buying
# nothing. At this label horizon the values are small in absolute terms, so read
# the direction and the spread between configurations rather than the level.

# %%
curves = result["all_learning_curves"]
if curves.height > 0:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for config_name in sorted(curves["config"].unique().to_list()):
        series = curves.filter(pl.col("config") == config_name).sort("epoch")
        ax.plot(series["epoch"], series["ic_mean"], marker="o", markersize=3, label=config_name)
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation information coefficient")
    ax.set_title(f"{MODEL} validation score across training epochs")
    ax.legend(title="Configuration")

# %% [markdown]
# ## 4. Check what was produced
#
# Before these predictions are used anywhere, confirm the run covered the folds
# it was asked for. A model that failed on one fold still leaves rows in the
# registry for the others, and a downstream average over whatever is present
# reads as a complete result. Comparing the folds returned against the folds
# requested is what separates the two.

# %%
predictions = result["predictions"]
fold_metrics = result["fold_metrics"]

requested_folds = sorted(int(f) for f in (FOLD_IDS or [s["fold"] for s in splits]))
produced_folds = sorted(int(f) for f in predictions["fold"].unique().to_list())
missing_folds = [f for f in requested_folds if f not in produced_folds]

print(f"Folds requested: {requested_folds}")
print(f"Folds with predictions: {produced_folds}")
print(f"Validation rows: {predictions.height:,}")
print(f"Null predictions: {predictions['prediction'].null_count():,}")

if missing_folds:
    raise RuntimeError(
        f"{MODEL} produced no predictions for fold(s) {missing_folds}. The registered "
        f"set is incomplete and must not be compared or backtested."
    )

# %% [markdown]
# ## 5. Key Takeaways
#
# 1. **A sequence model is a request, not a loop.** The window length, fold plan,
#    observation cadence and gap policy are declared once and resolved by the
#    shared runner. A notebook that rebuilds them locally will eventually cut a
#    window differently from its sibling notebooks, and the two results stop
#    being comparable without anything looking wrong.
#
# 2. **Overlapping labels make sequence counts misleading.** Every row starts a
#    window, so a fold holds almost as many sequences as it has rows, and nearly
#    all of them share most of their content. Capping how many are drawn is what
#    keeps the fit tractable; it also means the effective sample is far smaller
#    than the sequence count suggests.
#
# 3. **Check coverage before you compare.** Fold-level completeness is the
#    precondition for any comparison across architectures. `13_model_analysis`
#    is where those comparisons happen, on the population this notebook and its
#    siblings register.
#
# 4. **Patch length is a modelling choice, not a detail.** It sets how many items
#    attention compares and how much each one contains. Longer patches make the
#    fit cheaper and blur short events; shorter patches preserve them and cost
#    more. The window length alone does not determine it.
#
# **Known limitations**: The validation score here is a diagnostic of the fit, not
# a basis for choosing a configuration or a checkpoint. Attention over patches is
# what this architecture contributes, and it is the most expensive of the four to
# fit. Whether that expense is repaid is a question for the comparison in
# `13_model_analysis`, not for this notebook.
