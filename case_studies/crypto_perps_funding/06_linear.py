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
# # Linear Models - Crypto Perpetual Funding
#
# The feature evaluation finds several negative volatility and premium effects in a narrow
# 19-contract cross-section. Linear models test whether dense or sparse regularization can combine
# the 44 inputs into a stable out-of-sample ranking.
#
# **Learning Objectives**:
# - Train regularized linear models on walk-forward cross-validation folds
# - Compare regularization approaches (L1, L2, elastic net) via out-of-fold IC
# - Exclude incomplete prediction sets from model selection
# - Generate validation predictions for the downstream backtest funnel
#
# **Book Reference**: Chapter 11, Section 11.2 (Regularized Linear Models)
#
# **Prerequisites**: [`04_model_based_features`](04_model_based_features.ipynb) and
# [`05_evaluation`](05_evaluation.ipynb)

# %%
"""Evaluate regularized linear models with walk-forward cross-validation."""

import time
import warnings
from datetime import UTC, datetime

import matplotlib.pyplot as plt
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
    modeling_input_fingerprint,
    read_predictions,
    register_prediction_set,
    register_training_run,
    training_hash_from_spec,
    training_run_status,
)
from utils.cv_splits import load_evaluation_config
from utils.modeling import (
    ConfigError,
    load_configs,
    load_modeling_dataset,
    prepare_cv_folds,
    resolve_linear_params,
)
from utils.paths import get_case_study_dir
from utils.style import COLORS, add_message_title

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "crypto_perps_funding"
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
# references a preset in `config/` - a complete specification of
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
assert len(feature_names) == 44
holdout_start = datetime.fromisoformat(load_evaluation_config(CASE_STUDY_ID)["holdout_start"])
holdout_start = holdout_start.replace(tzinfo=UTC)
label_horizon = pd.Timedelta(mds.label_buffer)
assert all(split["val_end"] + label_horizon < holdout_start for split in splits)

INPUT_FINGERPRINT = modeling_input_fingerprint(
    CASE_DIR,
    PRIMARY_LABEL,
    splits,
    feature_names,
    MAX_SYMBOLS,
)
TRAINING_IDENTITY = {"input_fingerprint": INPUT_FINGERPRINT, "max_symbols": MAX_SYMBOLS}
print(f"Input lineage: {INPUT_FINGERPRINT[:12]}")

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
    eval_label_col=mds.eval_label_col,
)

for f in folds:
    print(f"  Fold {f['fold']}: train={f['n_train']:,}  val={f['n_val']:,}")

# %% [markdown]
# ## 3. Walk-Forward Cross-Validation
#
# The training identity includes the exact label, financial, temporal, and split artifacts. A
# historical prediction set cannot satisfy a rebuilt input fingerprint. Missing current-lineage
# configurations are fitted after train-only imputation and scaling, then scored by averaging IC
# across decision timestamps.

# %%
# sklearn class lookup - maps model_class strings from presets to classes
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
        extra_params=TRAINING_IDENTITY,
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
        # Load the canonical decision-time metric and physical prediction coverage.
        _pred_hash = _split_rows["prediction_hash"][0]
        _metrics = load_prediction_metrics(CASE_STUDY_ID, prediction_hash=_pred_hash)
        _ic_mean = float(_metrics["ic_mean_daily"][0]) if not _metrics.is_empty() else np.nan
        _ic_std = float(_metrics["ic_std_daily"][0]) if not _metrics.is_empty() else np.nan
        _ic_n_days = float(_metrics["ic_n_days"][0]) if not _metrics.is_empty() else np.nan
        _cached_predictions = read_predictions(CASE_STUDY_ID, _pred_hash)
        _n_null = _cached_predictions.select(pl.col("y_score").is_null().sum()).item()
        print(
            f"  {cfg['config_name']:25s}  IC={_ic_mean:+.4f} ± {_ic_std:.4f}  "
            f"n={int(_ic_n_days):,} dates  (cached, {_status.summary()})"
        )
        results.append(
            {
                "config": cfg,
                "predictions": pd.DataFrame(),
                "ic_mean": _ic_mean,
                "ic_std": _ic_std,
                "ic_n_days": _ic_n_days,
                "n_null": _n_null,
                "fold_ics": [],
                "degenerate": False,
                "cached": True,
                "started_at": None,
                "elapsed_s": 0.0,
            }
        )
        continue
    if _status.complete and not _split_complete:
        print(f"  {cfg['config_name']:25s}  RETRAIN - missing {PREDICTION_SPLIT} predictions")
    elif _status.partial:
        print(f"  {cfg['config_name']:25s}  RETRAIN - partial state: {_status.summary()}")

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

        # Check for degenerate model (all coefficients zero - regularization too strong)
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

        # For classification, IC is computed against the continuous return the
        # binary label was derived from. Spearman versus a binary label is not IC.
        ic_target = fold["y_eval"] if mds.eval_label_col else fold["y_val"]
        ic_frame = pl.DataFrame(
            {
                "date": fold["dates"],
                "symbol": fold["entities"],
                "y_true": ic_target,
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
        if mds.eval_label_col:
            pred_df["eval_actual"] = fold["y_eval"]
        fold_preds.append(pred_df)
    else:
        degenerate = False

    config_elapsed = time.perf_counter() - config_t0

    if degenerate:
        print(
            f"  {cfg['config_name']:25s}  SKIP - all coefficients zero (regularization too strong)"
        )
        results.append(
            {
                "config": cfg,
                "predictions": pd.DataFrame(),
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "ic_n_days": np.nan,
                "n_null": np.nan,
                "fold_ics": [],
                "degenerate": True,
                "started_at": config_started_at,
                "elapsed_s": config_elapsed,
            }
        )
        continue

    predictions = pd.concat(fold_preds, ignore_index=True)
    ic_target_col = "eval_actual" if mds.eval_label_col else "actual"
    ic_stats = cross_sectional_ic(
        pl.from_pandas(predictions),
        pl.from_pandas(predictions),
        pred_col="prediction",
        ret_col=ic_target_col,
        date_col=date_col,
        entity_col=entity_col,
        min_obs=5,
    )
    ic_mean = float(ic_stats["ic_mean"])
    ic_std = float(ic_stats["ic_std"])
    ic_n_days = int(ic_stats["n_periods"])
    n_null = int(predictions["prediction"].isna().sum())
    print(
        f"  {cfg['config_name']:25s}  IC={ic_mean:+.4f} ± {ic_std:.4f}  "
        f"n={ic_n_days:,} dates  ({config_elapsed:.1f}s)"
    )

    results.append(
        {
            "config": cfg,
            "predictions": predictions,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_n_days": ic_n_days,
            "n_null": n_null,
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
# Rank only configurations with complete predictions and the maximum number of valid decision-time
# IC observations. Constant or partially null predictions can otherwise manufacture a winner by
# dropping difficult timestamps.

# %%
# Sort by IC descending
results.sort(key=lambda r: r["ic_mean"] if np.isfinite(r["ic_mean"]) else -np.inf, reverse=True)

active = [r for r in results if not r.get("degenerate")]
degenerate = [r for r in results if r.get("degenerate")]

_finite_days = [r["ic_n_days"] for r in active if np.isfinite(r.get("ic_n_days", np.nan))]
_full_days = max(_finite_days) if _finite_days else None
full_cov = [
    r
    for r in active
    if r.get("n_null") == 0 and (_full_days is None or r.get("ic_n_days") == _full_days)
]
partial_cov = [r for r in active if r not in full_cov]

print(f"{'Config':25s}  {'Mean IC':>9s}  {'IC Std':>8s}  {'N Dates':>7s}")
print("-" * 58)
for r in full_cov:
    n_dates = int(r["ic_n_days"]) if np.isfinite(r["ic_n_days"]) else 0
    print(
        f"  {r['config']['config_name']:25s}  {r['ic_mean']:+.4f}  {r['ic_std']:.4f}  {n_dates:7d}"
    )
if partial_cov:
    print("\nIncomplete prediction coverage (excluded from ranking):")
    for r in partial_cov:
        print(
            f"  {r['config']['config_name']:25s}  IC={r['ic_mean']:+.4f}  "
            f"n_dates={r['ic_n_days']:.0f}  n_null={r['n_null']:.0f}"
        )
if degenerate:
    print(f"\nSkipped ({len(degenerate)} degenerate - all coefficients zero):")
    for r in degenerate:
        print(f"  {r['config']['config_name']}")

best = full_cov[0] if full_cov else None
if best:
    print(
        f"\nBest full-coverage config: {best['config']['config_name']} (IC={best['ic_mean']:+.4f})"
    )

# %% [markdown]
# ### Linear model ranking
#
# The complete prediction sets are directly comparable because each covers the same validation
# timestamps. The highlighted bar is the selected linear configuration, and the vertical line marks
# zero IC.


# %%
def display_config(name: str) -> str:
    """Return a compact reader-facing configuration label."""
    if name.startswith("ridge_a"):
        return f"Ridge alpha={float(name.split('_a')[1]):g}"
    if name.startswith("lasso_f"):
        return f"Lasso fraction={name.split('_f')[1]}"
    if name.startswith("enet_f"):
        return f"Elastic net fraction={name.split('_f')[1]}"
    return "OLS" if name == "ols" else name


ranked = full_cov[:20][::-1]
labels = [display_config(r["config"]["config_name"]) for r in ranked]
values = [r["ic_mean"] for r in ranked]
colors = [COLORS["amber"] if r is best else COLORS["blue"] for r in ranked]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(labels, values, color=colors)
ax.axvline(0, color=COLORS["neutral"], linewidth=1)
ax.set_xlabel("Mean decision-time rank IC (validation)")
ax.set_ylabel("Configuration")
add_message_title(
    ax,
    f"{display_config(best['config']['config_name'])} leads, but linear IC is only "
    f"{best['ic_mean']:+.3f}",
)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Ridge penalty path
#
# The Ridge sweep isolates how dense shrinkage changes validation ranking quality while preserving
# the same feature set. The logarithmic penalty axis makes the broad range readable.

# %%
ridge_path = sorted(
    (
        float(r["config"]["config_name"].split("_a")[1]),
        r["ic_mean"],
    )
    for r in full_cov
    if r["config"]["config_name"].startswith("ridge_a")
)
alphas = np.array([row[0] for row in ridge_path])
ridge_ics = np.array([row[1] for row in ridge_path])
peak = int(np.nanargmax(ridge_ics))
ridge_order_span = int(round(np.log10(alphas.max() / alphas.min())))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(np.log10(alphas), ridge_ics, color=COLORS["blue"], marker="o", linewidth=2)
ax.scatter(np.log10(alphas[peak]), ridge_ics[peak], color=COLORS["amber"], s=90, zorder=3)
ax.axhline(0, color=COLORS["neutral"], linewidth=1)
ax.set_xlabel("log10(alpha), Ridge penalty strength")
ax.set_ylabel("Mean decision-time rank IC (validation)")
add_message_title(
    ax,
    f"Ridge validation IC remains weak across {ridge_order_span} orders of penalty strength",
)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Register Results
#
# Uncached runs register predictions under a configuration and input-content hash. Historical
# predictions remain addressable under their old hashes but cannot satisfy the current lineage.
# Publication verification uses an isolated registry so the production registry remains unchanged.

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
        extra_params=TRAINING_IDENTITY,
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
        eval_col="eval_actual" if mds.eval_label_col else None,
        metrics=metrics,
    )
    print(f"  registered {cfg['config_name']:25s}  IC={r['ic_mean']:+.4f}")

# %% [markdown]
# ## 6. What the Linear Baseline Establishes
#
# The current input lineage shows whether dense shrinkage remains preferable to sparse selection.
# All displayed configurations cover the same validation timestamps, so the ranking is not driven
# by missing or constant-prediction periods. Notebook 07 tests whether non-linear interactions
# improve this linear baseline.
#
# **Next**: [`07_gbm`](07_gbm.ipynb) evaluates gradient boosting on the same canonical folds and
# 44-feature frame.
