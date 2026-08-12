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
# # PatchTST: S&P 500 Equity Option Analytics
#
# LSTM trailed TabM on the corrected feature lineage. PatchTST tests whether its
# multi-scale patching mechanism captures temporal patterns in IV dynamics that sequential
# gating misses. The hypothesis: IV term structure shifts operate at multiple timescales
# (daily, weekly, monthly), and PatchTST's patching should extract these hierarchical
# patterns more efficiently than LSTM's step-by-step processing.
#
# **Learning Objectives**:
# - Test whether patch-based attention captures multi-scale IV dynamics
# - Compare PatchTST against the full progression: Linear → GBM → TabM → LSTM
# - Assess whether the marginal IV-equity signal survives a more expressive architecture
#
# **Book Reference**: Chapter 13 (Deep Learning for Time Series)
#
# **Prerequisites**: `linear.py`, `gbm.py`, `tabular_dl.py`, `dl_lstm.py` (baselines)

# %%
"""PatchTST — sp500_equity_option_analytics deep learning."""

import sqlite3
import warnings

import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch
import yaml

import utils.style as style
from case_studies.utils.analytics import load_best_ic_per_family, load_model_ic
from case_studies.utils.deep_learning import (
    _resolve_arch_name,
    create_model,
    run_dl_cv,
)
from case_studies.utils.registry import build_training_spec, training_hash_from_spec
from utils.modeling import load_configs, load_modeling_dataset
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")
COLORS = style.COLORS

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_equity_option_analytics"
MODEL = "patchtst"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
FORCE_RETRAIN = False  # Set True to retrain configs that already have complete hashes
PREDICTION_SPLIT = "validation"
N_EPOCHS = 100
LOOKBACK = 60
BATCH_SIZE = 2048
MC_DROPOUT = False
MAX_FOLDS = 0

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
# Load IC results from earlier pipeline stages (Ch11 linear, Ch12 GBM and TabM, Ch13
# LSTM) rather than re-running them here. TabM is the strongest model so far and the
# reference PatchTST has to beat to justify its patch-attention machinery; LSTM is the
# prior sequence model that already tied TabM without improving on it.

# %%
prior_baselines = {}
_baselines = load_best_ic_per_family(["linear", "gbm", "tabular_dl"], case_studies=[CASE_STUDY_ID])
if not _baselines.is_empty():
    for row in _baselines.iter_rows(named=True):
        if row["family"] == "linear":
            prior_baselines[f"{row['config_name'].title()} (Ch11)"] = row["ic_mean"]
        elif row["family"] == "gbm":
            prior_baselines["GBM (Ch12)"] = row["ic_mean"]
        elif row["family"] == "tabular_dl":
            prior_baselines["TabM (Ch12)"] = row["ic_mean"]

# LSTM (Ch13) is a deep_learning config; load_best_ic_per_family returns PatchTST for
# that family (this notebook's own model), so read the best lstm_h64 checkpoint directly.
_dl_ic = load_model_ic(["deep_learning"], split="validation", case_studies=[CASE_STUDY_ID])
if not _dl_ic.is_empty():
    _lstm = _dl_ic.filter((pl.col("label") == label_col) & (pl.col("config_name") == "lstm_h64"))
    if not _lstm.is_empty():
        prior_baselines["LSTM (Ch13)"] = float(_lstm["ic_mean"].max())

if prior_baselines:
    for name, ic in prior_baselines.items():
        print(f"  {name}: IC={ic:+.4f}" if ic is not None else f"  {name}: IC=N/A")
else:
    print("  No prior results found — run 06_linear.py … 09_dl_lstm.py first")

# %% [markdown]
# ## 3. PatchTST
#
# Primary architecture for this notebook.

# %%
dl_configs = load_configs(CASE_STUDY_ID, PRIMARY_LABEL, "deep_learning")
dl_configs = [c for c in dl_configs if c["params"].get("architecture") == MODEL]

if not dl_configs:
    raise ValueError(
        f"No '{MODEL}' configs found — add '{MODEL}' under 'deep_learning:' in the label config"
    )

# Apply Papermill overrides to configs (test mode: fewer epochs)
for cfg in dl_configs:
    if cfg.get("n_epochs", 100) != N_EPOCHS:
        cfg["n_epochs"] = N_EPOCHS
    if cfg.get("batch_size", 2048) != BATCH_SIZE:
        cfg["batch_size"] = BATCH_SIZE
    if cfg["params"].get("lookback", 60) != LOOKBACK:
        cfg["params"]["lookback"] = LOOKBACK

print(
    f"Grid: {len(dl_configs)} configs × {dl_configs[0].get('n_epochs', 100)} epochs × {len(splits)} folds"
)
for cfg in dl_configs:
    print(
        f"  {cfg['config_name']}: {cfg['params'].get('architecture', '?')} ({cfg.get('n_epochs', 100)} epochs)"
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
    notebook=f"dl_{MODEL}",
    temporal_by_fold=mds.temporal_by_fold,
    temporal_keys=mds.temporal_keys,
    temporal_feature_names=mds.temporal_feature_names,
)


# %%
def _load_dl_results_from_registry(case_study, configs, label, split, n_folds):
    """Rebuild the leaderboard/curves/folds from the registry when every config SKIPs.

    ``run_dl_cv`` returns an empty ``grid_results`` when all configs are already
    complete (the reproduce path), so §4-§6 and the figures below need the stored
    validation ICs reloaded from ``registry.db`` (the single source of truth).
    """
    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    con = sqlite3.connect(db_path)
    try:
        grid, curve_rows = [], []
        for cfg in configs:
            spec = build_training_spec(
                cfg["family"],
                cfg["config_name"],
                label,
                n_folds=n_folds,
                n_epochs=cfg.get("n_epochs"),
            )
            thash = training_hash_from_spec(spec)
            rows = con.execute(
                "SELECT ps.checkpoint_value, pm.ic_mean, ps.prediction_hash "
                "FROM prediction_sets ps "
                "JOIN training_runs tr ON tr.training_hash = ps.training_hash "
                "JOIN prediction_metrics pm ON pm.prediction_hash = ps.prediction_hash "
                "WHERE ps.training_hash = ? AND ps.split = ? "
                "AND ps.created_at >= COALESCE(tr.started_at, tr.created_at) "
                "ORDER BY ps.checkpoint_value",
                (thash, split),
            ).fetchall()
            if not rows:
                continue
            elapsed = con.execute(
                "SELECT elapsed_s FROM training_runs WHERE training_hash = ?", (thash,)
            ).fetchone()
            for ckpt, ic_mean, _ in rows:
                curve_rows.append(
                    {"config": cfg["config_name"], "epoch": int(ckpt), "ic_mean": float(ic_mean)}
                )
            best = max(rows, key=lambda r: r[1])
            grid.append(
                {
                    "config_name": cfg["config_name"],
                    "best_epoch": int(best[0]),
                    "best_ic": float(best[1]),
                    "best_prediction_hash": best[2],
                    "elapsed_s": float(elapsed[0]) if elapsed and elapsed[0] is not None else 0.0,
                }
            )
        grid.sort(key=lambda r: r["best_ic"], reverse=True)
        best_row = grid[0] if grid else None
        fold_rows = []
        if best_row is not None:
            for fid, ic, n_ent in con.execute(
                "SELECT fold_id, ic, n_entities FROM fold_metrics "
                "WHERE prediction_hash = ? ORDER BY fold_id",
                (best_row["best_prediction_hash"],),
            ).fetchall():
                fold_rows.append(
                    {"fold_id": int(fid), "ic_mean": float(ic), "n_symbols": int(n_ent or 0)}
                )
        return {
            "grid_results": grid,
            "best_config_name": best_row["config_name"] if best_row else None,
            "best_epoch": best_row["best_epoch"] if best_row else 0,
            "best_ic": best_row["best_ic"] if best_row else float("nan"),
            "predictions": pl.DataFrame(),
            "all_predictions": pl.DataFrame(),
            "fold_metrics": pl.DataFrame(fold_rows),
            "all_learning_curves": pl.DataFrame(curve_rows),
            "training_log": pl.DataFrame(),
        }
    finally:
        con.close()


if not result["grid_results"]:
    print("All configs SKIP'd - reloading the stored leaderboard from the registry.")
    result = _load_dl_results_from_registry(
        CASE_STUDY_ID, dl_configs, label_col, PREDICTION_SPLIT, len(splits)
    )

# %% [markdown]
# ## 4. Validation IC and Fold Stability
#
# A fresh training run displays its checkpoint-by-checkpoint validation IC below. A
# reproduce run reloads only the selected checkpoint from the exact current registry
# execution. It therefore confirms the registered selection but does not reconstruct a
# learning curve from checkpoints written by earlier executions.

# %%
grid_results = result["grid_results"]
best_name = result["best_config_name"]
best_epoch = result["best_epoch"]
best_ic = result["best_ic"]

curves = result["all_learning_curves"]
if curves.height > 0:
    checkpoints = sorted(curves["epoch"].unique().to_list())

    print(f"{'Config':15s}", end="")
    for cp in checkpoints:
        print(f" {cp:>7d}", end="")
    print()
    print("-" * (15 + 8 * len(checkpoints)))

    for r in grid_results:
        cfg_data = curves.filter(pl.col("config") == r["config_name"])
        print(f"{r['config_name']:15s}", end="")
        for cp in checkpoints:
            row = cfg_data.filter(pl.col("epoch") == cp)
            if row.height > 0:
                print(f" {row['ic_mean'][0]:+7.4f}", end="")
            else:
                print(f" {'N/A':>7s}", end="")
        print()

# %% [markdown]
# The mean validation IC is a fragile average of only two walk-forward folds. Splitting
# it out shows how thin the signal is: one fold is strongly positive and one is slightly
# negative, so the headline number rests on a single favourable window rather than a
# stable edge.

# %%
fold_metrics = result["fold_metrics"]
if fold_metrics.height > 0:
    _fm = fold_metrics.sort("fold_id")
    _fold_ids = [f"Fold {i}" for i in _fm["fold_id"].to_list()]
    _fold_ics = _fm["ic_mean"].to_list()
    fig_folds = go.Figure()
    fig_folds.add_trace(
        go.Bar(
            x=_fold_ids,
            y=_fold_ics,
            marker_color=[COLORS["blue"] if v >= 0 else COLORS["copper"] for v in _fold_ics],
            text=[f"{v:+.4f}" for v in _fold_ics],
            textposition="outside",
        )
    )
    fig_folds.add_hline(y=best_ic, line=dict(color=COLORS["amber"], dash="dash"))
    fig_folds.add_hline(y=0.0, line=dict(color=COLORS["slate"], dash="dot"))
    _pad = max(abs(min(_fold_ics)), abs(max(_fold_ics))) * 1.35
    fig_folds.update_yaxes(range=[-_pad, _pad])
    fig_folds.update_layout(
        title_text=f"The {best_ic:+.3f} mean hides fold instability: one fold positive, one negative"
        f"<br><sup>{best_name} at epoch {best_epoch}, per-fold validation IC (fwd_ret_5d); "
        f"dashed line = 2-fold mean {best_ic:+.4f}</sup>",
        yaxis_title="Validation IC",
        height=380,
        showlegend=False,
    )
    fig_folds.show()

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
            "architecture", _resolve_arch_name(best_cfg_dict["config_name"])
        )
        from case_studies.utils.deep_learning import _build_arch_kwargs

        best_cfg = _build_arch_kwargs(
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
# ## 6. Model Progression
#
# PatchTST against the full pipeline it inherits: Ridge (Ch11) → GBM (Ch12) →
# TabM (Ch12) → LSTM (Ch13) → PatchTST (Ch13).

# %%
rows = [(name, ic) for name, ic in prior_baselines.items()]
rows.append(("PatchTST (Ch13)", best_ic))

comparison = pl.DataFrame({"Model": [r[0] for r in rows], "IC": [r[1] for r in rows]})
comparison = comparison.with_columns(
    pl.when(pl.col("IC") == pl.col("IC").max())
    .then(pl.lit("*"))
    .otherwise(pl.lit(""))
    .alias("Best")
)
comparison


# %%
def _baseline_ic(prefix):
    return next(
        (v for k, v in prior_baselines.items() if k.lower().startswith(prefix)), float("nan")
    )


ridge_ic = _baseline_ic("ridge")
gbm_ic = prior_baselines.get("GBM (Ch12)", float("nan"))
tabm_ic = prior_baselines.get("TabM (Ch12)", float("nan"))
lstm_ic = prior_baselines.get("LSTM (Ch13)", float("nan"))

prog_labels = ["Ridge (Ch11)", "GBM (Ch12)", "TabM (Ch12)", "LSTM (Ch13)", "PatchTST (Ch13)"]
prog_ics = [ridge_ic, gbm_ic, tabm_ic, lstm_ic, best_ic]
prog_colors = [
    COLORS["slate"],
    COLORS["slate"],
    COLORS["amber"],
    COLORS["slate"],
    COLORS["blue"],
]

fig_prog = go.Figure()
fig_prog.add_trace(
    go.Bar(
        y=prog_labels,
        x=prog_ics,
        orientation="h",
        marker_color=prog_colors,
        text=[f"{v:+.4f}" for v in prog_ics],
        textposition="outside",
    )
)
fig_prog.add_vline(x=0.0, line=dict(color=COLORS["slate"], dash="dot"))
# categoryarray runs bottom->top; reverse pipeline order to put PatchTST on top.
fig_prog.update_yaxes(
    type="category", categoryorder="array", categoryarray=list(reversed(prog_labels))
)
_xmax = max(prog_ics) * 1.3
_xmin = min(0.0, min(prog_ics) * 1.4)
fig_prog.update_xaxes(range=[_xmin, _xmax])
fig_prog.update_layout(
    title_text="PatchTST trails both LSTM and the best tabular model - no net gain"
    f"<br><sup>fwd_ret_5d, validation IC across the pipeline; "
    f"PatchTST {best_ic:+.4f} vs TabM {tabm_ic:+.4f}</sup>",
    xaxis_title="Validation IC",
    height=380,
    margin=dict(l=140),
    showlegend=False,
)
fig_prog.show()

# %%
print(f"PatchTST delta over Ridge baseline: {best_ic - ridge_ic:+.4f}  (Ridge is negative here)")
print(f"PatchTST delta over TabM (best tabular baseline): {best_ic - tabm_ic:+.4f}")
print(f"PatchTST delta over LSTM (prior sequence model): {best_ic - lstm_ic:+.4f}")

# %% [markdown]
# ## 7. Save Results
#
# Predictions and fold metrics are registered by `run_dl_cv()`
# during training. On the reproduce path every config already has a complete
# training hash, so `run_dl_cv()` re-registers nothing and the stored predictions
# and fold metrics are read back from the registry (the single source of truth).

# %%
fold_metrics = result["fold_metrics"]
val_ic_mean = float(fold_metrics["ic_mean"].mean()) if fold_metrics.height > 0 else None
print(f"Best config: {best_name} @ epoch {best_epoch}")
if val_ic_mean is not None:
    print(f"Mean validation IC across {fold_metrics.height} folds: {val_ic_mean:+.4f}")

# %% [markdown]
# ## 8. Key Takeaways
#
# 1. **PatchTST trails LSTM and TabM**: its selected validation IC is +0.0075, below
#    LSTM (+0.0100) and the best non-sequence model, TabM (+0.0156). Its delta from
#    TabM is -0.0081, so patch-based attention over multi-scale windows adds no edge
#    over the point-in-time tabular snapshot.
# 2. **The edge is not distinguishable from zero**: the daily-IC HAC t-statistic is
#    0.73 (p = 0.47) and the 95% confidence interval [-0.0125, +0.0272] straddles
#    zero. The +0.0075 fold mean averages just two walk-forward folds - one positive
#    (+0.0168), one negative (-0.0019) - so the headline rests on a single
#    favourable window rather than a stable edge.
# 3. **Selection evidence**: the registry carries the selected checkpoint from the
#    current execution. The fresh-run output provides the checkpoint trajectory; a
#    reproduce run does not infer that trajectory or validation-loss behavior from
#    historical registry rows.
# 4. **Model progression**: Ridge to GBM to TabM lifts IC from negative to +0.0156;
#    LSTM and PatchTST then fall back to +0.0100 and +0.0075. Neither sequence model
#    improves on the tabular leader or turns the marginal IC significant. Architecture
#    sophistication does not manufacture signal where the data holds little.
#
# **Next**: `11_latent_factors.py` shifts from supervised prediction to unsupervised
# factor extraction (PCA, IPCA, autoencoders, and a stochastic discount factor) on the
# same feature set.
