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
# # TSMixer — US Equities Panel
#
# TSMixer alternates time-mixing (across the 60-day lookback) and feature-mixing
# (across 71 features), seeking cross-feature interactions that LSTM's sequential
# gating might miss. On ETFs, TSMixer achieves the highest IC because the
# 99-ETF panel has structured sector and style groupings. The question is whether
# TSMixer's mixing layers find similar structure across 3,199 heterogeneous stocks.
#
# **Learning Objectives**:
# - Test whether cross-feature mixing helps on broad, heterogeneous panels
# - Compare TSMixer with LSTM and linear baseline
# - Evaluate whether dataset structure matters more than architecture choice
#
# **Book Reference**: Chapter 13
#
# **Prerequisites**: [`09_dl_nlinear`](09_dl_nlinear.ipynb),
# [`10_dl_lstm`](10_dl_lstm.ipynb), [`06_linear`](06_linear.ipynb),
# [`07_gbm`](07_gbm.ipynb)

# %%
"""TSMixer — us_equities_panel deep learning."""

import warnings

import numpy as np
import polars as pl
import torch
import yaml

from case_studies.utils.analytics import load_best_ic_per_family
from case_studies.utils.deep_learning import (
    create_model,
    resolve_arch_name,
    run_dl_cv,
)
from utils.modeling import load_configs, load_modeling_dataset
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
MODEL = "tsmixer"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
SYMBOLS = []  # Explicit symbol whitelist (tests pin raw-present symbols); [] = use MAX_SYMBOLS
FORCE_RETRAIN = False  # Set True to retrain configs that already have complete hashes
PREDICTION_SPLIT = "validation"
N_EPOCHS = 100
LOOKBACK = 60
BATCH_SIZE = 2048
MAX_TRAIN_SEQUENCES = 100_000
MC_DROPOUT = False
MAX_FOLDS = 0
FOLD_IDS = []

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
mds = load_modeling_dataset(
    CASE_STUDY_ID, PRIMARY_LABEL, max_symbols=MAX_SYMBOLS, symbols=SYMBOLS or None
)

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
print(f"Label: {label_col} | Entity: {entity_col} | Folds: {len(splits)}")

dataset_pd = dataset.to_pandas()
n_entities = dataset_pd[entity_col].nunique()
print(f"Entities: {n_entities}")

# %% [markdown]
# ## 2. Prior Baselines
#
# Load IC results from earlier pipeline stages (Ch11 linear, Ch12 GBM)
# rather than re-running them here.

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
# ## 3. TSMixer
#
# Primary architecture for this notebook.

# %%
dl_configs = load_configs(CASE_STUDY_ID, PRIMARY_LABEL, "deep_learning")
dl_configs = [c for c in dl_configs if c["params"].get("architecture") == MODEL]

# Apply Papermill overrides to configs (test mode: fewer epochs)
for cfg in dl_configs:
    if cfg.get("n_epochs", 100) != N_EPOCHS:
        cfg["n_epochs"] = N_EPOCHS
    if cfg.get("batch_size", 2048) != BATCH_SIZE:
        cfg["batch_size"] = BATCH_SIZE
    if cfg["params"].get("lookback", 60) != LOOKBACK:
        cfg["params"]["lookback"] = LOOKBACK

print(
    f"Grid: {len(dl_configs)} configs × {dl_configs[0].get('n_epochs', 100)} epochs × "
    f"{len(splits)} folds"
)
for cfg in dl_configs:
    print(
        f"  {cfg['config_name']}: {cfg['params'].get('architecture', '?')} "
        f"({cfg.get('n_epochs', 100)} epochs)"
    )

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
    notebook="11_dl_tsmixer",
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

    for r in grid_results:
        cfg_data = curves.filter(pl.col("config") == r["config_name"])
        print(f"{r['config_name']:15s}", end="")
        for cp in display_cps:
            row = cfg_data.filter(pl.col("epoch") == cp)
            if row.height > 0:
                print(f" {row['ic_mean'][0]:+7.4f}", end="")
            else:
                print(f" {'N/A':>7s}", end="")
        print()

# %% [markdown]
# ## 5. MC Dropout Uncertainty (Optional)

# %%
if MC_DROPOUT:
    from ml4t.diagnostic.metrics import cross_sectional_ic

    from case_studies.utils.deep_learning import mc_dropout_predict
    from case_studies.utils.sequence_dataset import (
        materialize_sequences,
        prepare_fold_sequence_stores,
    )

    dates_series = dataset_pd[date_col]
    last_fold = splits[-1]
    train_mask = (dates_series >= last_fold["train_start"]) & (
        dates_series <= last_fold["train_end"]
    )
    val_mask = (dates_series >= last_fold["val_start"]) & (dates_series <= last_fold["val_end"])

    train_store, val_store, _ = prepare_fold_sequence_stores(
        dataset_pd,
        train_mask=train_mask,
        val_mask=val_mask,
        feature_names=feature_names,
        label_col=label_col,
        date_col=date_col,
        entity_col=entity_col,
        lookback=LOOKBACK,
    )
    X_train_seq, y_train_seq, _, _ = materialize_sequences(train_store)
    X_val_seq, y_val_seq, val_dates, val_entities = materialize_sequences(val_store)

    if len(X_train_seq) > 0 and len(X_val_seq) > 0:
        torch_device = torch.device(device_str)
        best_cfg_dict = dl_configs[0]
        arch_name = best_cfg_dict["params"].get(
            "architecture", resolve_arch_name(best_cfg_dict["config_name"])
        )
        from case_studies.utils.deep_learning import build_arch_kwargs

        best_cfg = build_arch_kwargs(
            best_cfg_dict, n_features, best_cfg_dict["params"].get("lookback", 60)
        )
        mc_model = create_model(arch_name, best_cfg).to(torch_device)

        X_t = torch.FloatTensor(X_train_seq).to(torch_device)
        y_t = torch.FloatTensor(y_train_seq).to(torch_device)
        optimizer = torch.optim.AdamW(mc_model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        mc_model.train()
        for ep in range(min(N_EPOCHS, 50)):
            idx = torch.randperm(len(X_t))
            for s in range(0, len(X_t), BATCH_SIZE):
                batch = idx[s : s + BATCH_SIZE]
                loss = criterion(mc_model(X_t[batch]), y_t[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        X_v = torch.FloatTensor(X_val_seq).to(torch_device)
        mean_pred, std_pred = mc_dropout_predict(mc_model, X_v, n_samples=50)

        median_unc = np.median(std_pred)
        low_unc = std_pred <= median_unc
        high_unc = std_pred > median_unc

        low_frame = pl.DataFrame(
            {
                "date": val_dates[low_unc],
                "symbol": val_entities[low_unc],
                "y_true": y_val_seq[low_unc],
                "y_pred": mean_pred[low_unc],
            }
        )
        ic_low = cross_sectional_ic(
            low_frame,
            low_frame,
            pred_col="y_pred",
            ret_col="y_true",
            date_col="date",
            entity_col="symbol",
            min_obs=5,
        )["ic_mean"]
        high_frame = pl.DataFrame(
            {
                "date": val_dates[high_unc],
                "symbol": val_entities[high_unc],
                "y_true": y_val_seq[high_unc],
                "y_pred": mean_pred[high_unc],
            }
        )
        ic_high = cross_sectional_ic(
            high_frame,
            high_frame,
            pred_col="y_pred",
            ret_col="y_true",
            date_col="date",
            entity_col="symbol",
            min_obs=5,
        )["ic_mean"]
        print("MC Dropout uncertainty analysis:")
        print(f"  Low uncertainty IC:  {ic_low:+.4f} ({low_unc.sum():,} samples)")
        print(f"  High uncertainty IC: {ic_high:+.4f} ({high_unc.sum():,} samples)")
        print(f"  IC gap: {ic_low - ic_high:+.4f}")

        del mc_model, X_t, y_t, X_v
        torch.cuda.empty_cache()
else:
    print("MC Dropout disabled (set MC_DROPOUT=True to enable)")

# %% [markdown]
# ## 6. Comparison

# %%
rows = [(name, ic) for name, ic in prior_baselines.items()]
rows.append((best_name, best_ic))

comparison = pl.DataFrame({"Model": [r[0] for r in rows], "IC": [r[1] for r in rows]})
comparison = comparison.with_columns(
    pl.when(pl.col("IC") == pl.col("IC").max())
    .then(pl.lit("*"))
    .otherwise(pl.lit(""))
    .alias("Best")
)
comparison

# %%
ridge_ic = prior_baselines.get("Ridge (Ch11)", float("nan"))
dl_delta = best_ic - ridge_ic
print(f"DL delta over Ridge: {dl_delta:+.4f}")

# %% [markdown]
# ## 7. Save Results
#
# Predictions and fold metrics are registered by `run_dl_cv()`
# during training. Here we record the pipeline results JSON.

# %%
predictions = result["predictions"]
all_predictions = result["all_predictions"]
fold_metrics = result["fold_metrics"]

print(f"Predictions: {predictions.height:,} rows")
print(f"All predictions: {all_predictions.height:,} rows")

# %%
val_ic_mean = float(fold_metrics["ic_mean"].mean()) if fold_metrics.height > 0 else None

# %% [markdown]
# ## 8. Key Takeaways
#
# TSMixer produces the weakest result across all models on this dataset --- a
# clearly negative DL delta below linear. This is the clearest DL failure in the
# book. The contrast with TSMixer's dominance on ETFs is instructive:
# feature-mixing succeeds when the panel has structured cross-asset groupings
# (ETFs' sector and style categories), but fails on 3,199 heterogeneous stocks
# where no such grouping exists.
#
# Across both DL architectures, the message is consistent: the US equities panel
# rewards cross-sectional breadth, not temporal or feature-interaction modeling.
# Simple linear models capture everything the signal has to offer.
#
# **Next**: [`13_latent_factors`](13_latent_factors.ipynb) tests whether PCA
# and IPCA can extract
# common factors from this broad cross-section.
