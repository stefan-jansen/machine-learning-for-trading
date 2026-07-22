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
# # Linear Models: NASDAQ-100 Microstructure
#
# **Docker image**: `ml4t`
#
# NASDAQ-100 minute-bar data offers one of the richest feature spaces in the book:
# 88 financial and temporal features drawn from microstructure fields (signed volume,
# microprice deviation, depth imbalance) across 114 symbols and 13M+ observations.
# The 15-minute prediction horizon creates a distinctive challenge: returns at this
# frequency are tiny (median |return| ~15 bps), so even small IC improvements are
# economically meaningful --- or would be, if execution costs didn't dominate.
#
# This notebook establishes the linear baseline. As later notebooks will show, the
# gross-to-net Sharpe collapse makes this case study the book's starkest lesson in
# cost analysis: even strong validation Sharpes cannot survive 15-minute rebalancing
# costs.
#
# **Learning Objectives**:
# - Establish a linear IC baseline on high-frequency microstructure features
# - Assess whether L1/L2 regularization helps on a wide, liquid cross-section
# - Quantify the signal magnitude in basis points per trade (the cost-relevance test)
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
    load_prediction_metrics,
    load_prediction_sets,
    register_prediction_set,
    register_training_run,
    training_hash_from_spec,
    training_run_status,
)
from utils.modeling import (
    ConfigError,
    append_holdout_fold_if_needed,
    load_configs,
    load_modeling_dataset,
    prepare_cv_folds,
)
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
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
append_holdout_fold_if_needed(mds, PREDICTION_SPLIT, CASE_STUDY_ID)

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
# ## 2. Prepare CV Folds
#
# Each fold preprocesses training data (median imputation for missing features,
# standard scaling) and applies the same transformation to the validation set.

# %%
dataset_pd = dataset.to_pandas()
folds = prepare_cv_folds(
    dataset_pd,
    splits,
    feature_names,
    label_col,
    date_col,
    entity_col,
    temporal_by_fold=mds.temporal_by_fold,
    temporal_keys=mds.temporal_keys,
    temporal_feature_names=mds.temporal_feature_names,
    train_sample_frac=TRAIN_SAMPLE_FRAC,
)

for f in folds:
    print(f"  Fold {f['fold']}: train={f['n_train']:,}  val={f['n_val']:,}")

# %% [markdown]
# ## 3. Walk-Forward Cross-Validation
#
# For each configuration, fit the model on each training fold and predict
# the validation fold. Cross-sectional IC (Spearman rank correlation per
# date, averaged) measures predictive quality.

# %%
# sklearn class lookup — maps model_class strings from presets to classes
MODEL_CLASSES = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "LogisticRegression": LogisticRegression,
}


# %% [markdown]
# ### Fit-Predict-Score Loop
#
# For each config, fit on each training fold, predict the validation fold,
# and compute cross-sectional IC. Coefficients are stored for later inspection.


# %%
def fit_config(cfg, cls, folds, feature_names, mds, date_col, entity_col):
    """Fit one config across all folds, returning predictions, IC, and coefficients."""
    fold_preds = []
    fold_ics = []
    fold_coefs = []
    degenerate = False

    for fold in folds:
        try:
            model = cls(**cfg["params"])
        except TypeError as e:
            raise ConfigError(
                f"Cannot create {cfg['model_class']} from preset '{cfg['config_name']}'.\n"
                f"Check preset params for {cfg['config_name']}: {e}"
            ) from e

        model.fit(fold["X_train"], fold["y_train"])

        # Check for degenerate model (all coefficients zero — regularization too strong)
        if hasattr(model, "coef_") and np.all(model.coef_ == 0):
            degenerate = True
            break

        # Store coefficients (feature weights + intercept)
        if hasattr(model, "coef_"):
            coefs = model.coef_.ravel() if model.coef_.ndim > 1 else model.coef_
            intercept = model.intercept_ if np.isscalar(model.intercept_) else model.intercept_[0]
            for feat, c in zip(feature_names, coefs, strict=False):
                fold_coefs.append(
                    {
                        "config_name": cfg["config_name"],
                        "fold": fold["fold"],
                        "feature": feat,
                        "coefficient": float(c),
                    }
                )
            fold_coefs.append(
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
        fold_ics.append(ic)

        # Assemble prediction DataFrame for this fold
        pred_df = fold["meta"][mds.join_cols].copy()
        pred_df["fold"] = fold["fold"]
        pred_df["prediction"] = preds
        pred_df["actual"] = fold["y_val"]
        fold_preds.append(pred_df)

    return fold_preds, fold_ics, fold_coefs, degenerate


# %%
def run_config(cfg, folds, feature_names, mds, date_col, entity_col):
    """Run one config: validate, fit, and assemble result dict."""
    cls = MODEL_CLASSES.get(cfg["model_class"])
    if cls is None:
        raise ConfigError(
            f"Unknown model_class '{cfg['model_class']}' in preset '{cfg['config_name']}'.\n"
            f"Available: {list(MODEL_CLASSES.keys())}"
        )

    config_started_at = datetime.now(UTC).isoformat()
    config_t0 = time.perf_counter()

    # Skip if this config's hash is already complete (unless FORCE_RETRAIN)
    _early_spec = build_training_spec(
        cfg["family"],
        cfg["config_name"],
        label_col,
        n_folds=len(folds),
        train_sample_frac=TRAIN_SAMPLE_FRAC,
    )
    _status = training_run_status(CASE_STUDY_ID, _early_spec)
    if _status.complete and not FORCE_RETRAIN:
        # Already trained + registered: load the cached IC from the registry
        # and return it as a `cached` result so the Results Summary renders on
        # a fully-cached checkout. (A bare `return None` here drops the config
        # from `results`, printing an empty summary when every config is
        # registered.)
        _t_hash = training_hash_from_spec(_early_spec)
        _pred_sets = load_prediction_sets(
            CASE_STUDY_ID, training_hash=_t_hash, split=PREDICTION_SPLIT
        )
        _ic_mean = np.nan
        _ic_std = np.nan
        if not _pred_sets.is_empty():
            _pred_hash = _pred_sets["prediction_hash"][0]
            _metrics = load_prediction_metrics(CASE_STUDY_ID, prediction_hash=_pred_hash)
            if not _metrics.is_empty():
                _ic_mean = float(_metrics["ic_mean"][0])
                _ic_std = float(_metrics["ic_std"][0])
        print(
            f"  {cfg['config_name']:25s}  IC={_ic_mean:+.4f} ± {_ic_std:.4f}  "
            f"(cached, {_status.summary()})"
        )
        return {
            "config": cfg,
            "predictions": pd.DataFrame(),
            "ic_mean": _ic_mean,
            "ic_std": _ic_std,
            "fold_ics": [],
            "degenerate": False,
            "cached": True,
            "started_at": None,
            "elapsed_s": 0.0,
        }
    if _status.partial:
        print(f"  {cfg['config_name']:25s}  RETRAIN — partial state: {_status.summary()}")

    fold_preds, fold_ics, fold_coefs, degenerate = fit_config(
        cfg, cls, folds, feature_names, mds, date_col, entity_col
    )
    config_elapsed = time.perf_counter() - config_t0

    if degenerate:
        print(
            f"  {cfg['config_name']:25s}  SKIP — all coefficients zero (regularization too strong)"
        )
        return {
            "config": cfg,
            "predictions": pd.DataFrame(),
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "fold_ics": [],
            "degenerate": True,
            "started_at": config_started_at,
            "elapsed_s": config_elapsed,
        }

    ic_mean = float(np.nanmean(fold_ics))
    ic_std = float(np.nanstd(fold_ics))
    print(f"  {cfg['config_name']:25s}  IC={ic_mean:+.4f} ± {ic_std:.4f}  ({config_elapsed:.1f}s)")

    return {
        "config": cfg,
        "predictions": pd.concat(fold_preds, ignore_index=True),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "fold_ics": fold_ics,
        "coefficients": fold_coefs,
        "degenerate": False,
        "started_at": config_started_at,
        "elapsed_s": config_elapsed,
    }


# %%
results = [
    r
    for r in (run_config(cfg, folds, feature_names, mds, date_col, entity_col) for cfg in configs)
    if r is not None
]

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
    if r.get("cached"):
        continue  # already registered on a prior run; predictions not reloaded
    cfg = r["config"]
    spec = build_training_spec(
        cfg["family"],
        cfg["config_name"],
        label_col,
        n_folds=len(folds),
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
# ## 5. Key Takeaways
#
# ElasticNet on 15-minute returns produces weak IC that translates to a fraction
# of a basis point of expected edge per trade --- already dangerously close to
# the 5+ bps execution cost floor for NASDAQ-100 stocks. The linear baseline
# establishes that predictive signal exists, but the economic question is whether
# non-linear models (Ch12 GBM, Ch13 DL) can amplify it enough to overcome costs.
# Spoiler: they cannot --- even the best DL model still falls short when the
# breakeven cost is far below actual execution costs (Ch18).
