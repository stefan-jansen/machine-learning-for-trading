# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Linear Models — US Equities Panel
#
# **Docker image**: `ml4t`
#
# The US equities panel is the broadest dataset in the book: 3,199 stocks spanning
# 1962--2018 with 72 features (63 financial + 9 temporal). With 9.2 million training
# rows across 16 walk-forward folds, this is the Fundamental Law's test case. Can a
# modest per-stock IC combine with large breadth ($\sqrt{3{,}199} \approx 56.6$) to
# produce a tradable information ratio? Linear models on daily returns establish the
# baseline prediction quality across this massive cross-section.
#
# **Learning Objectives**:
# - Train regularized linear models on the largest walk-forward setup in the book
# - Test whether cross-sectional breadth compensates for weak per-stock signal
# - Compare L1 vs L2 regularization on a 72-feature, 3,199-stock panel
# - Generate predictions for downstream backtesting (Ch16--19)
#
# **Book Reference**: Chapter 11, Section 11.2 (Regularized Linear Models)
#
# **Prerequisites**: `03_financial_features.py`, `04_temporal.py`, [`05_evaluation`](05_evaluation.ipynb)

# %%
"""Linear Models — walk-forward cross-validation."""

import time
import warnings
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import polars as pl
import yaml
from ml4t.diagnostic.metrics import cross_sectional_ic
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge

from case_studies.utils.registry import (
    build_training_spec,
    get_training_dir,
    register_prediction_set,
    register_training_run,
)
from utils.modeling import (
    ConfigError,
    load_configs,
    load_modeling_dataset,
    prepare_single_fold,
)
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
FORCE_RETRAIN = False  # Set True to retrain configs that already have complete hashes
PREDICTION_SPLIT = "validation"
TRAIN_SAMPLE_FRAC = 1.0  # <1.0 subsamples training rows per fold (val is never sampled). Use for memory-constrained runs on large datasets.
MAX_FOLDS = 0

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
if not PRIMARY_LABEL:
    setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
    PRIMARY_LABEL = setup["labels"]["primary"]

# %% [markdown]
# ## 1. Load Data and Model Configs
#
# Model configurations are defined in `config/training/{label}.yaml`. Each entry
# references a preset in `config/` — a complete specification of
# the sklearn class and its constructor parameters. To modify the grid,
# edit the label config file: comment out presets or add new ones.

# %%
mds = load_modeling_dataset(CASE_STUDY_ID, PRIMARY_LABEL, max_symbols=MAX_SYMBOLS)

dataset = mds.dataset
feature_names = mds.feature_names
label_col = mds.label_col
date_col = mds.date_col
entity_col = mds.entity_cols[0] if mds.entity_cols else None
splits = mds.splits[: MAX_FOLDS or None]

print(f"Dataset: {len(dataset):,} rows × {len(feature_names)} features")
print(f"Label: {label_col} | Task: {mds.task_type} | Folds: {len(splits)}")

# %%
configs = load_configs(CASE_STUDY_ID, PRIMARY_LABEL, family="linear")

print(f"\n{len(configs)} configs × {len(splits)} folds = {len(configs) * len(splits)} fits\n")
for cfg in configs:
    params_str = (
        ", ".join(f"{k}={v}" for k, v in cfg["params"].items()) if cfg["params"] else "defaults"
    )
    print(f"  {cfg['config_name']:25s}  {cfg['model_class']}({params_str})")

# %% [markdown]
# ## 2. Walk-Forward Cross-Validation (Fold-Major)
#
# The US equities panel is too large (9.2M rows × 72 features × 16 folds)
# to materialize all folds at once. `prepare_single_fold()` accepts the
# Polars DataFrame directly — only the current fold's rows are converted
# to numpy. This avoids both the full pandas copy and the all-folds
# materialization problem.

# %% [markdown]
# ## 3. Walk-Forward Cross-Validation (Fold-Major)
#
# For each fold: preprocess (impute + scale), then fit ALL configs on that
# fold's data. This keeps only one fold's arrays in memory at a time —
# critical for the 9.2M-row panel.

# %%
# sklearn class lookup — maps model_class strings from presets to classes
MODEL_CLASSES = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "LogisticRegression": LogisticRegression,
}

# Per-config accumulators (fold-major loop fills these incrementally)
config_state = {}
for cfg in configs:
    cls = MODEL_CLASSES.get(cfg["model_class"])
    if cls is None:
        raise ConfigError(
            f"Unknown model_class '{cfg['model_class']}' in preset '{cfg['config_name']}'.\n"
            f"Available: {list(MODEL_CLASSES.keys())}"
        )
    config_state[cfg["config_name"]] = {
        "config": cfg,
        "cls": cls,
        "fold_preds": [],
        "fold_ics": [],
        "fold_coefs": [],
        "degenerate": False,
        "started_at": datetime.now(UTC).isoformat(),
        "t0": time.perf_counter(),
    }

n_folds_processed = 0
for split in splits:
    fold = prepare_single_fold(
        dataset,
        split,
        feature_names,
        label_col,
        date_col,
        entity_col,
        temporal_by_fold=mds.temporal_by_fold,
        temporal_keys=mds.temporal_keys,
        temporal_feature_names=mds.temporal_feature_names,
        train_sample_frac=TRAIN_SAMPLE_FRAC,
    )
    if fold is None:
        continue
    n_folds_processed += 1
    print(f"  Fold {fold['fold']}: train={fold['n_train']:,}  val={fold['n_val']:,}")

    for cfg in configs:
        st = config_state[cfg["config_name"]]
        if st["degenerate"]:
            continue

        try:
            model = st["cls"](**cfg["params"])
        except TypeError as e:
            raise ConfigError(
                f"Cannot create {cfg['model_class']} from preset '{cfg['config_name']}'.\n"
                f"Check preset params for {cfg['config_name']}: {e}"
            ) from e

        model.fit(fold["X_train"], fold["y_train"])

        # Check for degenerate model (all coefficients zero — regularization too strong)
        if hasattr(model, "coef_") and np.all(model.coef_ == 0):
            st["degenerate"] = True
            print(f"    {cfg['config_name']:25s}  DEGENERATE at fold {fold['fold']}")
            continue

        # Store coefficients (feature weights + intercept)
        if hasattr(model, "coef_"):
            coefs = model.coef_.ravel() if model.coef_.ndim > 1 else model.coef_
            intercept = model.intercept_ if np.isscalar(model.intercept_) else model.intercept_[0]
            for feat, c in zip(feature_names, coefs, strict=False):
                st["fold_coefs"].append(
                    {
                        "config_name": cfg["config_name"],
                        "fold": fold["fold"],
                        "feature": feat,
                        "coefficient": float(c),
                    }
                )
            st["fold_coefs"].append(
                {
                    "config_name": cfg["config_name"],
                    "fold": fold["fold"],
                    "feature": "_intercept_",
                    "coefficient": float(intercept),
                }
            )

        # Classification: use expected value of class probabilities for IC
        if mds.task_type == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(fold["X_val"])
            preds = proba @ np.array(sorted(mds.class_values), dtype=np.float64)
        else:
            preds = model.predict(fold["X_val"])

        ic_frame = pl.DataFrame(
            {
                "date": fold["dates"],
                "symbol": fold["entities"],
                "y_true": fold["y_val"],
                "y_pred": preds,
            }
        )
        ic = cross_sectional_ic(
            ic_frame,
            ic_frame,
            pred_col="y_pred",
            ret_col="y_true",
            date_col="date",
            entity_col="symbol",
            min_obs=5,
        )["ic_mean"]
        st["fold_ics"].append(ic)

        # Assemble prediction DataFrame for this fold
        if fold["meta_pl"] is not None:
            # Polars path — select join columns, add fold/prediction/actual
            available_cols = [c for c in mds.join_cols if c in fold["meta_pl"].columns]
            pred_df = fold["meta_pl"].select(available_cols).to_pandas()
        else:
            pred_df = fold["meta"][mds.join_cols].copy()
        pred_df["fold"] = fold["fold"]
        pred_df["prediction"] = preds
        pred_df["actual"] = fold["y_val"]
        st["fold_preds"].append(pred_df)

    del fold  # Free fold arrays before preparing next fold

# %%
# Assemble results from per-config accumulators
results = []
for cfg in configs:
    st = config_state[cfg["config_name"]]

    config_elapsed = time.perf_counter() - st["t0"]

    if st["degenerate"]:
        print(
            f"  {cfg['config_name']:25s}  SKIP — all coefficients zero (regularization too strong)"
        )
        results.append(
            {
                "config": cfg,
                "predictions": pd.DataFrame(),
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "fold_ics": [],
                "degenerate": True,
                "started_at": st["started_at"],
                "elapsed_s": config_elapsed,
            }
        )
        continue

    ic_mean = float(np.nanmean(st["fold_ics"]))
    ic_std = float(np.nanstd(st["fold_ics"]))
    print(f"  {cfg['config_name']:25s}  IC={ic_mean:+.4f} ± {ic_std:.4f}  ({config_elapsed:.1f}s)")

    results.append(
        {
            "config": cfg,
            "predictions": pd.concat(st["fold_preds"], ignore_index=True),
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "fold_ics": st["fold_ics"],
            "coefficients": st["fold_coefs"],
            "degenerate": False,
            "started_at": st["started_at"],
            "elapsed_s": config_elapsed,
        }
    )

del config_state  # Free accumulators

# %% [markdown]
# ## 4. Results Summary
#
# Rank configs by mean IC. Group by model family (OLS, Ridge, Lasso, ElasticNet)
# and report the best regularization strength per family.

# %%
# Sort by IC descending
results.sort(key=lambda r: r["ic_mean"] if np.isfinite(r["ic_mean"]) else -np.inf, reverse=True)

active = [r for r in results if not r.get("degenerate")]
degenerate = [r for r in results if r.get("degenerate")]

print(f"{'Config':25s}  {'IC Mean':>9s}  {'IC Std':>8s}")
print("-" * 46)
for r in active:
    print(f"  {r['config']['config_name']:25s}  {r['ic_mean']:+.4f}  {r['ic_std']:.4f}")
if degenerate:
    print(f"\nSkipped ({len(degenerate)} degenerate — all coefficients zero):")
    for r in degenerate:
        print(f"  {r['config']['config_name']}")

best = active[0] if active else None
if best:
    print(f"\nBest: {best['config']['config_name']} (IC={best['ic_mean']:+.4f})")

# %% [markdown]
# ## 5. Register Results
#
# Each config is registered in the unified registry with its predictions,
# IC metrics, and full provenance (training hash = SHA256 of config + label
# + features + folds). Identical configs produce the same hash — re-running
# updates rather than duplicates.

# %%
for r in active:
    cfg = r["config"]
    spec = build_training_spec(
        cfg["family"],
        cfg["config_name"],
        label_col,
        n_folds=n_folds_processed,
        train_sample_frac=TRAIN_SAMPLE_FRAC,
    )
    t_hash = register_training_run(
        CASE_STUDY_ID,
        spec=spec,
        entry_point="06_linear",
        started_at=r.get("started_at"),
        elapsed_s=r.get("elapsed_s"),
    )

    # Save coefficients to registry training dir
    train_dir = get_training_dir(CASE_STUDY_ID, spec)
    coefs = r.get("coefficients", [])
    if coefs:
        pd.DataFrame(coefs).to_parquet(train_dir / "coefficients.parquet", index=False)

    metrics = {"ic_mean": r["ic_mean"], "ic_std": r["ic_std"]}
    register_prediction_set(
        CASE_STUDY_ID,
        t_hash,
        split=PREDICTION_SPLIT,
        predictions=r["predictions"],
        task_type=mds.task_type,
        class_values=mds.class_values or None,
        metrics=metrics,
    )
    print(f"  registered {cfg['config_name']:25s}  IC={r['ic_mean']:+.4f}")

# %%
# Pipeline results JSON
model_results = {}
for r in results:
    name = r["config"]["config_name"]
    if r.get("degenerate"):
        model_results[name] = {"degenerate": True, "reason": "all coefficients zero"}
        continue
    model_results[name] = {
        "ic_mean": round(r["ic_mean"], 6) if np.isfinite(r["ic_mean"]) else None,
        "ic_std": round(r["ic_std"], 6) if np.isfinite(r["ic_std"]) else None,
        "model_class": r["config"]["model_class"],
        "params": r["config"]["params"],
    }

# %% [markdown]
# ## 7. Key Takeaways
#
# ElasticNet IC on daily returns is weak per-stock, but powerful at breadth =
# 3,199. The Fundamental Law says breadth compensates for low IC. Whether the
# portfolio construction chapters can harvest this theoretical IR depends on
# turnover costs (3,199 daily positions is expensive) and risk management.
#
# The stronger 5-day IC hints that weekly rebalancing may be more natural for
# this signal --- a hypothesis the cost analysis (Ch18) will test.
#
# **Next**: [`07_gbm`](07_gbm.ipynb) tests whether non-linear models can improve on the
# linear baseline across 3,199 stocks.
