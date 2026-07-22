# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CME Futures: Linear Models
#
# **Docker image**: `ml4t`
#
# CME futures span 30 products across 7 sectors (equity indices, treasuries,
# energy, metals, currencies, agriculture, livestock). The entity column is
# `product`, not `symbol`. Carry and term structure features dominate the
# 69-feature set, with roll yield, basis, and curve slope capturing the
# structural hedging pressure that drives futures returns. With 5 folds of
# 8-year training windows, the model spans multiple rate cycles, commodity
# supercycles, and the COVID-era dislocation.
#
# **Learning Objectives**:
# - Compare how L1 vs L2 regularization handles correlated carry features
# - Evaluate whether ElasticNet's mixed penalty improves on pure Lasso
# - Identify which feature families (carry, momentum, volatility) survive selection
# - Generate out-of-fold predictions for Ch16 backtesting
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
    load_configs,
    load_modeling_dataset,
    prepare_cv_folds,
    resolve_linear_params,
)
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "cme_futures"
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

results = []
for cfg in configs:
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
    _training_hash = training_hash_from_spec(_early_spec)
    _split_rows = load_prediction_sets(
        CASE_STUDY_ID,
        training_hash=_training_hash,
        split=PREDICTION_SPLIT,
    )
    _split_complete = not _split_rows.is_empty()
    if _status.complete and _split_complete and not FORCE_RETRAIN:
        # Already trained + registered: load the cached IC from the registry
        # and append it so the Results Summary renders on a fully-cached
        # checkout. (A bare `continue` here drops the config from `results`,
        # printing an empty summary table when every config is registered.)
        _pred_hash = _split_rows["prediction_hash"][0]
        _metrics = load_prediction_metrics(CASE_STUDY_ID, prediction_hash=_pred_hash)
        _ic_mean = float(_metrics["ic_mean"][0]) if not _metrics.is_empty() else np.nan
        _ic_std = float(_metrics["ic_std"][0]) if not _metrics.is_empty() else np.nan
        # ic_n_days is the count of daily cross-sections behind the pooled IC; a
        # config that dropped a fold (undefined fold IC) covers fewer days. Used
        # by the fold-completeness guard in the Results Summary below.
        _ic_n_days = (
            float(_metrics["ic_n_days"][0])
            if not _metrics.is_empty() and _metrics["ic_n_days"][0] is not None
            else np.nan
        )
        print(
            f"  {cfg['config_name']:25s}  IC={_ic_mean:+.4f} ± {_ic_std:.4f}  "
            f"(cached, {_status.summary()})"
        )
        results.append(
            {
                "config": cfg,
                "predictions": pd.DataFrame(),
                "ic_mean": _ic_mean,
                "ic_std": _ic_std,
                "ic_n_days": _ic_n_days,
                "fold_ics": [],
                "degenerate": False,
                "cached": True,
                "started_at": None,
                "elapsed_s": 0.0,
            }
        )
        continue
    if _status.complete and not _split_complete:
        print(f"  {cfg['config_name']:25s}  RETRAIN — missing {PREDICTION_SPLIT} predictions")
    elif _status.partial:
        print(f"  {cfg['config_name']:25s}  RETRAIN — partial state: {_status.summary()}")

    fold_preds = []
    fold_ics = []
    fold_coefs = []

    for fold in folds:
        try:
            model = cls(**resolve_linear_params(cfg, fold["X_train"], fold["y_train"]))
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
    else:
        degenerate = False

    config_elapsed = time.perf_counter() - config_t0

    if degenerate:
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
                "started_at": config_started_at,
                "elapsed_s": config_elapsed,
            }
        )
        continue

    ic_mean = float(np.nanmean(fold_ics))
    ic_std = float(np.nanstd(fold_ics))
    print(f"  {cfg['config_name']:25s}  IC={ic_mean:+.4f} ± {ic_std:.4f}  ({config_elapsed:.1f}s)")

    results.append(
        {
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
    )

# %% [markdown]
# ## 4. Results Summary
#
# Rank configs by mean IC. Group by model family (OLS, Ridge, Lasso, ElasticNet)
# and report the best regularization strength per family.

# %%
# Fold-completeness guard. A config whose cross-sectional IC is undefined on any
# fold (over-shrunk L1 collapses that fold's predictions to a constant, so the
# per-date rank correlation is undefined -> NaN fold IC) has its mean IC averaged
# over the surviving folds only, which biases it upward and is not a valid model
# result. The shared registry leaderboard (12_model_analysis) and every backtest
# selection exclude these via `degenerate_prediction_sql`; apply the same rule
# here so "Best" is a like-for-like full-fold winner, matching the Ch11/Ch12
# narrative (the strongest full-fold linear model is enet_f0.5). A dropped fold
# shows up as a short pooled IC (fewer ic_n_days) for cached configs and as a
# NaN fold IC for configs trained this run.
_day_counts = [r["ic_n_days"] for r in results if np.isfinite(r.get("ic_n_days", np.nan))]
_max_days = max(_day_counts) if _day_counts else None


def _is_incomplete(r: dict) -> bool:
    fold_ics = r.get("fold_ics") or []
    if fold_ics:  # trained this run — inspect the in-memory fold ICs
        return any(not np.isfinite(x) for x in fold_ics)
    d = r.get("ic_n_days", np.nan)  # cached — compare pooled-IC day coverage
    return _max_days is not None and np.isfinite(d) and d < _max_days


# Sort by IC descending
results.sort(key=lambda r: r["ic_mean"] if np.isfinite(r["ic_mean"]) else -np.inf, reverse=True)

zero_coef = [r for r in results if r.get("degenerate")]
active = [r for r in results if not r.get("degenerate")]
complete = [r for r in active if not _is_incomplete(r)]
incomplete = [r for r in active if _is_incomplete(r)]

print(f"{'Config':25s}  {'IC Mean':>9s}  {'IC Std':>8s}")
print("-" * 46)
for r in complete:
    print(f"  {r['config']['config_name']:25s}  {r['ic_mean']:+.4f}  {r['ic_std']:.4f}")
if incomplete:
    print(f"\nExcluded ({len(incomplete)} — undefined IC on 1+ folds, mean biased upward):")
    for r in incomplete:
        print(f"  {r['config']['config_name']:25s}  {r['ic_mean']:+.4f}  (partial folds)")
if zero_coef:
    print(f"\nSkipped ({len(zero_coef)} degenerate — all coefficients zero):")
    for r in zero_coef:
        print(f"  {r['config']['config_name']}")

best = complete[0] if complete else None
if best:
    print(f"\nBest (full-fold): {best['config']['config_name']} (IC={best['ic_mean']:+.4f})")

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
# ## 7. Key Takeaways
#
# Linear IC is consistent with the marginal feature evaluation from Ch8.
# Unregularized OLS overfits the 69 correlated carry and momentum features and
# posts a negative validation IC (-0.024); moderate L1/ElasticNet shrinkage is
# needed to reach a marginally positive value, and the strongest full-fold linear
# model is `enet_f0.5` at +0.0026. The most aggressive L1 configs (`lasso_f0.85`,
# `lasso_f0.7`) print a higher IC, but their cross-sectional IC is undefined on
# the oldest fold, so their mean is averaged over four folds and biased upward;
# the shared leaderboard and backtest selection exclude them, and so does the
# summary above. Either way the signal is weak. The real question for Ch12 is
# whether GBM captures non-linear sector interactions that linear models treat as
# additive.
#
# **Next**: [`07_gbm`](07_gbm.ipynb) tests gradient boosting with 15 tree depth × loss
# configurations to evaluate non-linear feature interactions.
