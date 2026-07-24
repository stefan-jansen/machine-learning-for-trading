# ---
# jupyter:
#   jupytext:
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
# # GBM Grid Search — US Equities Panel
#
# With 3,199 stocks and 72 features, the question is whether gradient boosting
# can capture non-linear feature interactions that linear models miss. On daily
# returns, the linear baseline suggests the cross-sectional
# relationship is largely linear. The GBM grid (5 leaf profiles × 3 loss
# functions = 15 configurations) tests whether tree-based non-linearity adds
# value on this broad, noisy panel.
#
# **Learning Objectives**:
# - Test whether non-linear models improve on the linear baseline for broad panels
# - Compare loss functions (MSE vs MAE vs Huber) on noisy daily returns
# - Track IC learning curves to identify overfitting on 3,199-stock cross-sections
# - Determine if horizon matters: daily vs weekly vs monthly GBM performance
#
# **Book Reference**: Chapter 12, Section 12.2 (GBM Libraries)
#
# **Prerequisites**: `03_financial_features.py`, `04_temporal.py`, [`05_evaluation`](05_evaluation.ipynb)

# %%
"""GBM Grid Search — config-driven regularization profiles × loss functions."""

import warnings

import numpy as np
import polars as pl
import yaml

from case_studies.utils.gbm import (
    prepare_gbm_folds,
    register_gbm_result,
    train_gbm_config,
)
from case_studies.utils.registry import (
    build_training_spec,
    get_training_dir,
    load_prediction_metrics,
    load_prediction_sets,
    training_hash_from_spec,
    training_run_status,
)
from utils.modeling import load_configs, load_modeling_dataset
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
MAX_FOLDS = 0
FORCE_RETRAIN = False  # Set True to retrain configs that already have complete hashes
PREDICTION_SPLIT = "validation"
TRAIN_SAMPLE_FRAC = 1.0  # <1.0 subsamples training rows per fold (val is never sampled). Use for memory-constrained runs on large datasets.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

if not PRIMARY_LABEL:
    PRIMARY_LABEL = setup["labels"]["primary"]

# Device: read from setup.yaml, fall back to GPU detection
gbm_config = setup.get("modeling", {}).get("gbm", {})
DEVICE = gbm_config.get("device", "cuda")
MAX_BIN = 63  # GPU default
import torch

if DEVICE != "cpu" and not torch.cuda.is_available():
    DEVICE, MAX_BIN = "cpu", 255

print(f"Case study: {CASE_STUDY_ID} | Device: {DEVICE} | max_bin: {MAX_BIN}")

# %% [markdown]
# ## 1. Load Data and Model Configs
#
# GBM configs are defined in `config/training/{label}.yaml` under the `gbm:` key.
# Each config references a preset in `config/lgb/` with the complete
# LightGBM parameter set. To modify the grid, edit the label config file.

# %%
mds = load_modeling_dataset(CASE_STUDY_ID, PRIMARY_LABEL, max_symbols=MAX_SYMBOLS)

dataset = mds.dataset
feature_names = mds.feature_names
label_col = mds.label_col
date_col = mds.date_col
entity_col = mds.entity_cols[0] if mds.entity_cols else "symbol"
splits = mds.splits[: MAX_FOLDS or None]

print(f"Dataset: {len(dataset):,} rows × {len(feature_names)} features")
print(f"Label: {label_col} | Task: {mds.task_type} | Folds: {len(splits)}")

# %%
configs = load_configs(CASE_STUDY_ID, PRIMARY_LABEL, family="gbm")

print(f"\n{len(configs)} configs × {len(splits)} folds\n")
for cfg in configs:
    leaves = cfg["params"].get("num_leaves", 31)
    obj = cfg["params"].get("objective", "regression")
    n_trees = cfg.get("max_iterations", 500)
    print(f"  {cfg['config_name']:25s}  leaves={leaves:3d}  obj={obj}  trees={n_trees}")

# %% [markdown]
# ## 2. Prepare CV Folds
#
# GBM folds use float32 (LightGBM native precision) and skip
# imputation/scaling — gradient boosting handles missing values natively.

# %%
dataset_pd = dataset.to_pandas()
fold_data = prepare_gbm_folds(
    dataset_pd,
    splits,
    feature_names,
    label_col,
    date_col,
    entity_col,
    task_type=mds.task_type,
    class_values=mds.class_values,
    temporal_by_fold=mds.temporal_by_fold,
    temporal_keys=mds.temporal_keys,
    temporal_feature_names=mds.temporal_feature_names,
    train_sample_frac=TRAIN_SAMPLE_FRAC,
)

for f in fold_data:
    print(f"  Fold {f['fold']}: train={f['n_train']:,}  val={f['n_val']:,}")

# %% [markdown]
# ## 3. Train All Configs
#
# For each config, train one LightGBM model per fold to `max_iterations` trees.
# Cross-sectional IC is evaluated at checkpoints (every 50 iterations) to
# detect overfitting — configs that peak early and decay indicate too much capacity.

# %%
results = []
for cfg in configs:
    # Pre-compute registry training dir so boosters go directly there
    spec = build_training_spec(
        cfg["family"],
        cfg["config_name"],
        label_col,
        n_folds=len(fold_data),
        max_bin=MAX_BIN,
        checkpoint_interval=cfg.get("checkpoint_interval", 50),
        train_sample_frac=TRAIN_SAMPLE_FRAC,
    )
    train_dir = get_training_dir(CASE_STUDY_ID, spec)

    # Skip if this config's hash is already complete (unless FORCE_RETRAIN)
    _status = training_run_status(CASE_STUDY_ID, spec)
    _training_hash = training_hash_from_spec(spec)
    _split_rows = load_prediction_sets(
        CASE_STUDY_ID,
        training_hash=_training_hash,
        split=PREDICTION_SPLIT,
    )
    _split_complete = not _split_rows.is_empty()
    if _status.complete and _split_complete and not FORCE_RETRAIN:
        # Already trained + registered: rebuild a minimal result from the
        # registry so the grid + learning-curve sections render on a
        # fully-cached checkout. (A bare `continue` here drops the config
        # from `results`, printing an empty grid when every config is
        # registered.) best_ic is the authoritative registered value;
        # best_iter and the curves come from learning_curves.parquet.
        _pred_hash = _split_rows["prediction_hash"][0]
        _metrics = load_prediction_metrics(CASE_STUDY_ID, prediction_hash=_pred_hash)
        _best_ic = float(_metrics["ic_mean"][0]) if not _metrics.is_empty() else float("nan")
        _curves = []
        _lc_path = train_dir / "learning_curves.parquet"
        if _lc_path.exists():
            _curves = pl.read_parquet(_lc_path).to_dicts()
        _best_iter = 0
        if _curves:
            _best_iter = int(max(_curves, key=lambda c: c["ic_mean"])["iteration"])
        print(
            f"  {cfg['config_name']:25s}  iter={_best_iter:4d}  IC={_best_ic:+.4f}  "
            f"(cached, {_status.summary()})"
        )
        results.append(
            {
                "config_name": cfg["config_name"],
                "best_iter": _best_iter,
                "best_ic": _best_ic,
                "elapsed_s": 0.0,
                "learning_curves": _curves,
                "cached": True,
            }
        )
        continue
    if _status.complete and not _split_complete:
        print(f"  {cfg['config_name']:25s}  RETRAIN — missing {PREDICTION_SPLIT} predictions")
    elif _status.partial:
        print(f"  {cfg['config_name']:25s}  RETRAIN — partial state: {_status.summary()}")

    result = train_gbm_config(
        cfg,
        fold_data,
        feature_names=feature_names,
        device=DEVICE,
        max_bin=MAX_BIN,
        entity_col=entity_col,
        date_col=date_col,
        task_type=mds.task_type,
        class_values=mds.class_values,
        save_dir=train_dir,
    )
    results.append(result)
    print(
        f"  {result['config_name']:25s}  iter={result['best_iter']:4d}  "
        f"IC={result['best_ic']:+.4f}  ({result['elapsed_s']:.0f}s)"
    )

    # Register immediately after training — incremental save protects against
    # interruption losing work on large sweeps.
    register_gbm_result(
        CASE_STUDY_ID,
        result,
        cfg,
        label_col,
        n_folds=len(fold_data),
        max_bin=MAX_BIN,
        entry_point="07_gbm",
        date_col=date_col,
        entity_col=entity_col,
        train_sample_frac=TRAIN_SAMPLE_FRAC,
        prediction_split=PREDICTION_SPLIT,
    )
# %% [markdown]
# ## 4. Grid Results
#
# All configs ranked by peak IC (best checkpoint). The best config × iteration
# combination is selected for downstream prediction generation.

# %%
results.sort(key=lambda r: r["best_ic"], reverse=True)
best = results[0] if results else None

print(f"{'Config':25s}  {'Iter':>5s}  {'IC':>8s}  {'Time':>6s}")
print("-" * 50)
for r in results:
    marker = " *" if r is best else ""
    print(
        f"  {r['config_name']:25s}  {r['best_iter']:5d}  {r['best_ic']:+.4f}  {r['elapsed_s']:5.0f}s{marker}"
    )

if best:
    print(f"\nBest: {best['config_name']} @ {best['best_iter']} trees (IC={best['best_ic']:+.4f})")

# %% [markdown]
# ## 5. Learning Curves
#
# IC at checkpoints (every 50 iterations) for each config. Configs that peak
# early and decay indicate overfitting; those that plateau show good regularization.

# %%
all_curves = pl.DataFrame([c for r in results for c in r["learning_curves"]])
if all_curves.height > 0:
    checkpoints = sorted(all_curves["iteration"].unique().to_list())
    display_cps = [cp for cp in [50, 100, 200, 300, 500] if cp in checkpoints]

    print(f"{'Config':25s}", end="")
    for cp in display_cps:
        print(f" {cp:>7d}", end="")
    print()

    for r in results:
        cfg_data = all_curves.filter(pl.col("config") == r["config_name"])
        print(f"  {r['config_name']:25s}", end="")
        for cp in display_cps:
            row = cfg_data.filter(pl.col("iteration") == cp)
            if row.height > 0:
                print(f" {row['ic_mean'][0]:+7.4f}", end="")
            else:
                print(f" {'N/A':>7s}", end="")
        print()

# %% [markdown]
# ## 6. Registration Complete
#
# Each config was registered immediately after training (see Section 3).
# This protects against interruption — all completed configs are already
# persisted in `run_log/registry.db`.

# %%
print(f"All {len(results)} configs registered.")
# %%

# %% [markdown]
# ## 7. Key Takeaways
#
# GBM adds almost nothing on daily returns (nearly matching linear), but the gap
# opens at longer horizons where GBM clearly achieves a higher IC than linear.
# For the broadest equity panel, the daily cross-section is essentially linear --- the
# Fundamental Law's breadth advantage doesn't require non-linear modeling. MAE
# loss consistently achieves a higher IC than MSE, a practical lesson for noisy return
# prediction: robust loss functions matter more than model complexity.
#
# **Next**: [`08_tabular_dl`](08_tabular_dl.ipynb) and the DL notebooks test whether attention
# mechanisms or temporal architectures add value on this panel.
