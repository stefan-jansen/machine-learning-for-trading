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
# # Temporal Convolutional Network for Crypto Perpetual Funding
#
# The TCN tests whether causal, dilated convolutions over 60 bars add predictive
# information beyond the same 44-feature flat-model frame. The cached reader path
# must reconstruct the registered physical result without silently returning an
# empty notebook or training an unrelated model.
#
# **Learning Objectives**:
# - Build causal 60-bar sequences from 39 financial and five fold-temporal features
# - Evaluate TCN checkpoints with decision-time rank IC on purged validation folds
# - Reconstruct the current registered checkpoint surface without rewriting the registry
# - Compare the registered TCN with current-lineage Linear, GBM, TabM, and LSTM results
#
# **Book Reference**: Chapter 13, Sections 13.4 and 13.8
#
# **Prerequisites**: [`04_model_based_features`](04_model_based_features.ipynb),
# [`07_gbm`](07_gbm.ipynb), and [`09_dl_lstm`](09_dl_lstm.ipynb)

# %%
"""Temporal convolutional model for crypto perpetual funding."""

import warnings

import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch
import yaml
from ml4t.diagnostic.metrics import cross_sectional_ic

import utils.style as style
from case_studies.utils.analytics import load_best_ic_per_family
from case_studies.utils.deep_learning import run_dl_cv
from case_studies.utils.registry import (
    build_training_spec,
    compute_fold_metrics_from_predictions,
    load_prediction_metrics,
    load_prediction_sets,
    modeling_input_fingerprint,
    read_predictions,
    training_hash_from_spec,
    training_run_status,
)
from utils.modeling import load_configs, load_modeling_dataset
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "crypto_perps_funding"
MODEL = "tcn"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
FORCE_RETRAIN = False
PREDICTION_SPLIT = "validation"
N_EPOCHS = 100
LOOKBACK = 60
BATCH_SIZE = 2048
MAX_FOLDS = 0
TRAIN_DEVICE = "cuda"
SEED = 42

# %%
set_global_seeds(SEED)
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
PRIMARY_LABEL = PRIMARY_LABEL or setup["labels"]["primary"]

if TRAIN_DEVICE not in {"cpu", "cuda"}:
    raise ValueError("TRAIN_DEVICE must be 'cpu' or 'cuda'")
if TRAIN_DEVICE != "cuda":
    raise ValueError("Crypto publication TCN training requires TRAIN_DEVICE='cuda'")
if not torch.cuda.is_available():
    raise RuntimeError("Crypto TCN requires CUDA, but PyTorch cannot see a CUDA device")

print(f"Case study: {CASE_STUDY_ID} | Model: {MODEL}")
print(f"Device: {TRAIN_DEVICE} | Epochs: {N_EPOCHS} | Lookback: {LOOKBACK}")

# %% [markdown]
# ## 1. Canonical Modeling Frame
#
# The loader assembles the signed 39-feature financial emit with five temporal
# features fitted separately for each fold. The 2024-2025 holdout remains sealed.

# %%
mds = load_modeling_dataset(CASE_STUDY_ID, PRIMARY_LABEL, max_symbols=MAX_SYMBOLS)
dataset = mds.dataset
feature_names = mds.feature_names
label_col = mds.label_col
date_col = mds.date_col
entity_col = mds.entity_cols[0] if mds.entity_cols else "symbol"
splits = mds.splits[:MAX_FOLDS] if MAX_FOLDS else mds.splits
expected_fold_ids = sorted(int(split["fold"]) for split in splits)
dataset_pd = dataset.to_pandas()
INPUT_FINGERPRINT = modeling_input_fingerprint(
    CASE_DIR,
    PRIMARY_LABEL,
    splits,
    feature_names,
    MAX_SYMBOLS,
)
IDENTITY_PARAMS = {
    "device": TRAIN_DEVICE,
    "input_fingerprint": INPUT_FINGERPRINT,
    "max_symbols": MAX_SYMBOLS,
    "max_folds": MAX_FOLDS,
    "batch_size": BATCH_SIZE,
    "lookback": LOOKBACK,
}

financial_features = [name for name in feature_names if name not in mds.temporal_feature_names]
assert len(financial_features) == 39
assert len(mds.temporal_feature_names) == 5
assert len(feature_names) == 44
assert label_col not in feature_names
assert dataset.select(date_col, entity_col).is_duplicated().sum() == 0

print(f"Dataset: {len(dataset):,} rows x {len(feature_names)} features")
print(
    f"Feature assembly: {len(financial_features)} financial + {len(mds.temporal_feature_names)} temporal"
)
print(f"Entities: {dataset[entity_col].n_unique()} | Validation folds: {len(splits)}")
print(f"Input lineage: {INPUT_FINGERPRINT[:12]}")

# %% [markdown]
# ## 2. Same-Lineage Baselines


# %% [markdown]
# The family leaderboard returns only one deep-learning leader. The LSTM baseline
# is therefore reconstructed from its current physical checkpoint surface.


# %%
def _physical_deep_ic(config_name: str) -> float | None:
    """Load and re-score one registered deep-learning prediction set."""
    config = next(
        candidate
        for candidate in load_configs(CASE_STUDY_ID, PRIMARY_LABEL, "deep_learning")
        if candidate["config_name"] == config_name
    )
    spec = build_training_spec(
        config["family"],
        config["config_name"],
        label_col,
        n_folds=len(splits),
        n_epochs=N_EPOCHS,
        extra_params=IDENTITY_PARAMS,
    )
    sets = load_prediction_sets(
        CASE_STUDY_ID,
        training_hash=training_hash_from_spec(spec),
        split=PREDICTION_SPLIT,
    )
    if sets.is_empty():
        return None
    scores = []
    for row in sets.iter_rows(named=True):
        predictions = read_predictions(CASE_STUDY_ID, row["prediction_hash"])
        score = cross_sectional_ic(
            predictions,
            predictions,
            pred_col="y_score",
            ret_col="y_true",
            date_col=date_col,
            entity_col=entity_col,
            min_obs=5,
        )["ic_mean"]
        scores.append(float(score))
    return max(scores)


# %% [markdown]
# Every baseline uses the same current input lineage and isolated registry.


# %%
prior_baselines = {}
baseline_rows = load_best_ic_per_family(
    ["linear", "gbm", "tabular_dl"], case_studies=[CASE_STUDY_ID]
)
for row in baseline_rows.iter_rows(named=True):
    if row["family"] == "linear":
        prior_baselines["Linear"] = float(row["ic_mean"])
    elif row["family"] == "gbm":
        prior_baselines["GBM"] = float(row["ic_mean"])
    elif row["family"] == "tabular_dl":
        prior_baselines["TabM"] = float(row["ic_mean"])

lstm_ic = _physical_deep_ic("lstm_h64")
if lstm_ic is not None:
    prior_baselines["LSTM"] = lstm_ic

for name, ic_value in prior_baselines.items():
    print(f"  {name}: current validation IC={ic_value:+.4f}")

# %% [markdown]
# ## 3. Current Reconstruction or Fresh Training
#
# A complete current-lineage registry exposes every physical prediction set. The
# notebook reads and re-scores that surface without calling the trainer. A missing
# current identity runs the CUDA path and evaluates every physical checkpoint.

# %%
dl_configs = [
    config
    for config in load_configs(CASE_STUDY_ID, PRIMARY_LABEL, "deep_learning")
    if config["params"].get("architecture") == MODEL
]
if not dl_configs:
    raise ValueError(f"No {MODEL!r} configuration is defined")
for config in dl_configs:
    config["n_epochs"] = N_EPOCHS
    config["batch_size"] = BATCH_SIZE
    config["params"]["lookback"] = LOOKBACK

print(f"Grid: {len(dl_configs)} config x {len(splits)} folds x {N_EPOCHS} epochs")


# %% [markdown]
# Physical predictions are the metric oracle. IC is computed within each decision
# timestamp and then averaged, preserving equal weight across decisions.


# %%
def _checkpoint_metrics(config_name: str, epoch: int, frame: pl.DataFrame) -> dict:
    """Score one physical checkpoint on decision-time rank IC."""
    stats = cross_sectional_ic(
        frame,
        frame,
        pred_col="y_score",
        ret_col="y_true",
        date_col=date_col,
        entity_col=entity_col,
        min_obs=5,
    )
    n_nonfinite = frame.select((~pl.col("y_score").is_finite()).fill_null(True).sum()).item()
    fold_dtype = frame.schema["fold_id"]
    fold_ids = sorted(frame["fold_id"].unique().to_list()) if fold_dtype.is_integer() else []
    return {
        "config": config_name,
        "epoch": int(epoch),
        "ic_mean": float(stats["ic_mean"]),
        "ic_std": float(stats["ic_std"]),
        "ic_n_days": int(stats["n_periods"]),
        "n_obs": frame.height,
        "n_null": int(frame["y_score"].null_count() + frame["y_score"].is_nan().sum()),
        "n_nonfinite": int(n_nonfinite),
        "fold_complete": fold_dtype.is_integer() and fold_ids == expected_fold_ids,
    }


# %% [markdown]
# Selection admits only unique, finite, null-free checkpoints with maximum
# validation-date coverage and the exact expected folds.


# %%
def _checkpoint_panel(all_predictions: pl.DataFrame) -> pl.DataFrame:
    """Score every physical config and epoch panel."""
    return pl.DataFrame(
        [
            _checkpoint_metrics(config_name, epoch, frame)
            for (config_name, epoch), frame in all_predictions.partition_by(
                ["config", "epoch"], as_dict=True, maintain_order=True
            ).items()
        ]
    )


# %% [markdown]
# Canonical selection rejects duplicate keys before comparing eligible checkpoint panels.


# %%
def _canonical_result(all_predictions: pl.DataFrame, *, lineage: str) -> dict:
    """Select the complete checkpoint with the highest pooled IC."""
    keys = [date_col, entity_col, "fold_id", "config", "epoch"]
    if all_predictions.select(keys).is_duplicated().any():
        raise ValueError("Duplicate sequence checkpoint prediction keys")
    curves = _checkpoint_panel(all_predictions)
    full_days = int(curves["ic_n_days"].max())
    curves = curves.with_columns(
        (
            (pl.col("ic_n_days") == full_days)
            & (pl.col("n_nonfinite") == 0)
            & pl.col("fold_complete")
            & pl.col("ic_mean").is_finite()
        ).alias("selectable")
    )
    eligible = curves.filter("selectable").sort("ic_mean", descending=True)
    if eligible.is_empty():
        raise ValueError("No finite checkpoint contains every expected validation fold")
    winner = eligible.row(0, named=True)
    selected = all_predictions.filter(
        (pl.col("config") == winner["config"]) & (pl.col("epoch") == winner["epoch"])
    )
    return {
        "lineage": lineage,
        "best_config_name": winner["config"],
        "best_epoch": winner["epoch"],
        "best_ic": winner["ic_mean"],
        "predictions": selected,
        "all_predictions": all_predictions,
        "all_learning_curves": curves,
        "fold_metrics": compute_fold_metrics_from_predictions(
            all_predictions, winner["config"], winner["epoch"], date_col=date_col
        ),
        "prediction_hash": None,
        "uncertainty": None,
    }


# %% [markdown]
# Cached reconstruction requires the exact current identity and checkpoint surface.


# %%
def _current_spec(config: dict) -> dict:
    """Build the content-addressed training identity for one TCN config."""
    return build_training_spec(
        config["family"],
        config["config_name"],
        label_col,
        n_folds=len(splits),
        n_epochs=config.get("n_epochs"),
        extra_params=IDENTITY_PARAMS,
    )


# %% [markdown]
# Registry uncertainty belongs to the selected physical prediction set and is
# loaded separately from the checkpoint curve.


# %%
def _registered_uncertainty(prediction_hash: str) -> dict | None:
    """Load overlap-aware uncertainty for one physical prediction set."""
    frame = load_prediction_metrics(CASE_STUDY_ID, prediction_hash=prediction_hash)
    return frame.row(0, named=True) if frame.height else None


# %% [markdown]
# Prediction hashes are de-duplicated before the physical epoch slices are read.


# %%
def _checkpoint_frames(config: dict, prediction_sets: pl.DataFrame) -> list[pl.DataFrame]:
    """Load each registered physical checkpoint exactly once."""
    frames = []
    seen_hashes = set()
    for prediction_set in prediction_sets.iter_rows(named=True):
        prediction_hash = prediction_set["prediction_hash"]
        if prediction_hash in seen_hashes:
            continue
        seen_hashes.add(prediction_hash)
        frames.append(
            read_predictions(CASE_STUDY_ID, prediction_hash).with_columns(
                pl.lit(config["config_name"]).alias("config"),
                pl.lit(int(prediction_set["checkpoint_value"])).alias("epoch"),
            )
        )
    return frames


# %% [markdown]
# The cached path requires the complete expected epoch surface before scoring.


# %%
def _cached_result(config: dict) -> dict | None:
    """Reconstruct and independently score the current checkpoint surface."""
    spec = _current_spec(config)
    status = training_run_status(CASE_STUDY_ID, spec)
    training_hash = training_hash_from_spec(spec)
    prediction_sets = load_prediction_sets(
        CASE_STUDY_ID, training_hash=training_hash, split=PREDICTION_SPLIT
    )
    if FORCE_RETRAIN or not status.complete or prediction_sets.is_empty():
        return None
    expected_epochs = sorted(
        {
            *range(
                config["checkpoint_interval"],
                config["n_epochs"] + 1,
                config["checkpoint_interval"],
            ),
            config["n_epochs"],
        }
    )
    actual_epochs = sorted(int(value) for value in prediction_sets["checkpoint_value"])
    if actual_epochs != expected_epochs:
        raise ValueError(f"Incomplete TCN checkpoint surface: {actual_epochs}")

    result = _canonical_result(
        pl.concat(_checkpoint_frames(config, prediction_sets), how="diagonal_relaxed"),
        lineage="current registered CUDA predictions",
    )
    selected_set = prediction_sets.filter(pl.col("checkpoint_value") == result["best_epoch"]).row(
        0, named=True
    )
    result["prediction_hash"] = selected_set["prediction_hash"]
    result["uncertainty"] = _registered_uncertainty(selected_set["prediction_hash"])
    return result


# %% [markdown]
# A missing current identity takes the full CUDA training path.


# %%
def _train_fresh_result() -> dict:
    """Train missing TCN checkpoints and apply canonical fresh selection."""
    raw_result = run_dl_cv(
        dataset_pd,
        splits,
        feature_names=feature_names,
        label_col=label_col,
        date_col=date_col,
        entity_col=entity_col,
        configs=dl_configs,
        n_features=len(feature_names),
        device=TRAIN_DEVICE,
        save_dir=CASE_DIR / "run_log" / "training" / "deep_learning",
        register=True,
        force_retrain=FORCE_RETRAIN,
        prediction_split=PREDICTION_SPLIT,
        case_study=CASE_STUDY_ID,
        notebook="10_dl_tcn",
        temporal_by_fold=mds.temporal_by_fold,
        temporal_keys=mds.temporal_keys,
        temporal_feature_names=mds.temporal_feature_names,
        identity_params=IDENTITY_PARAMS,
    )
    result = _canonical_result(
        raw_result["all_predictions"],
        lineage="fresh current CUDA training",
    )
    result["prediction_hash"] = None
    result["uncertainty"] = None
    return result


# %% [markdown]
# Resolve the physical result before any chart or conclusion is assembled.


# %%
result = _cached_result(dl_configs[0])
if result is None:
    result = _train_fresh_result()

best_name = result["best_config_name"]
best_epoch = result["best_epoch"]
best_ic = result["best_ic"]
predictions = result["predictions"]
curves = result["all_learning_curves"]

print(f"Lineage: {result['lineage']}")
print(f"Selected: {best_name} at epoch {best_epoch}, IC={best_ic:+.6f}")
print(f"Coverage: {predictions[date_col].n_unique():,} dates, {predictions.height:,} rows")

# %% [markdown]
# ## 4. Checkpoint History

# %%
if curves.height:
    selected_mask = (
        curves["selected"].to_list()
        if "selected" in curves.columns
        else (curves["epoch"] == best_epoch).to_list()
    )
    colors = [
        style.COLORS["amber"] if selected else style.COLORS["blue"] for selected in selected_mask
    ]
    curve_figure = go.Figure(
        go.Scatter(
            x=curves["epoch"],
            y=curves["ic_mean"],
            mode="lines+markers",
            line=dict(color=style.COLORS["slate"], width=1.5),
            marker=dict(color=colors, size=[14 if selected else 8 for selected in selected_mask]),
            showlegend=False,
        )
    )
    curve_figure.add_hline(y=0, line=dict(color=style.COLORS["silver_muted"], width=1))
    curve_figure.update_layout(
        title=f"The registered TCN checkpoint is epoch {best_epoch} (physical IC {best_ic:+.3f})",
        height=500,
        width=950,
        margin=dict(t=70),
    )
    curve_figure.update_xaxes(title_text="Training epoch")
    curve_figure.update_yaxes(title_text="Validation IC (decision-time mean)")
    curve_figure.show()

# %% [markdown]
# ## 5. Same-Lineage Model Comparison

# %%
baseline_order = ["Linear", "GBM", "TabM", "LSTM"]
comparison_names = [name for name in baseline_order if name in prior_baselines]
comparison_values = [prior_baselines[name] for name in comparison_names]
color_map = {
    "Linear": style.COLORS["slate"],
    "GBM": style.COLORS["copper"],
    "TabM": style.COLORS["amber"],
    "LSTM": style.COLORS["positive"],
}
comparison_colors = [color_map[name] for name in comparison_names]
comparison_names.append(f"TCN ({best_name})")
comparison_values.append(best_ic)
comparison_colors.append(style.COLORS["blue"])
comparison_title = f"Current-lineage TCN selection has validation IC {best_ic:+.3f}"

# %% [markdown]
# The chart keeps all family leaders on the same validation-IC scale.

# %%
comparison_figure = go.Figure(
    go.Bar(
        x=comparison_names,
        y=comparison_values,
        marker_color=comparison_colors,
        text=[f"{value:+.4f}" for value in comparison_values],
        textposition="outside",
        cliponaxis=False,
    )
)
comparison_figure.add_hline(y=0, line=dict(color=style.COLORS["silver_muted"], width=1))
comparison_figure.update_layout(
    title=comparison_title,
    height=500,
    width=950,
    margin=dict(t=70, b=75),
)
comparison_figure.update_xaxes(title_text=None)
comparison_figure.add_annotation(
    text="Model family leader",
    x=0.5,
    y=-0.17,
    xref="paper",
    yref="paper",
    showarrow=False,
)
comparison_figure.update_yaxes(title_text="Mean cross-sectional IC (validation)")
comparison_figure.show()

# %% [markdown]
# ## 6. Validation Diagnostics

# %%
fold_metrics = result["fold_metrics"]
print("Per-fold validation IC:")
for row in fold_metrics.sort("fold_id").iter_rows(named=True):
    print(f"  Fold {row['fold_id']}: IC={row['ic_mean']:+.4f}")

uncertainty = result["uncertainty"]
if uncertainty and uncertainty.get("ic_ci_lo") is not None:
    print(
        f"HAC 95% CI: [{uncertainty['ic_ci_lo']:+.4f}, {uncertainty['ic_ci_hi']:+.4f}] "
        f"| p={uncertainty['ic_p_hac']:.3f}"
    )

# %% [markdown]
# ## 7. Key Takeaways
#
# - **Selection uses physical checkpoint evidence.** All 20 checkpoints are
#   re-scored with equal weight per decision timestamp before selection. Epoch
#   10 leads the TCN curve with validation IC of +0.0044.
# - **The comparison is lineage-consistent.** Linear, GBM, TabM, LSTM, and TCN
#   values all share the current fingerprint-bound modeling frame. The TCN
#   trails all four baselines; GBM remains the family leader at +0.0288.
# - **The sequence contract is explicit.** Every fold uses causal 60-bar,
#   per-symbol windows over 39 financial features plus five fold-fitted temporal
#   features. Its HAC interval [-0.0073, +0.0161] includes zero (p=0.460), so
#   this run does not establish positive TCN rank association.
#
# **Next**: [`11_causal_dml`](11_causal_dml.ipynb) estimates treatment effects,
# then [`12_model_analysis`](12_model_analysis.ipynb) compares the signed model
# families.
