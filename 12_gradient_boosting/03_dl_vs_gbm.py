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
# # Deep Learning vs GBMs for Tabular Financial Data
#
# **Docker image**: `ml4t-gpu`
#
# **Chapter 12, Section 12.3**: Deep Learning Alternatives for Tabular Data
#
# ## Purpose
# This notebook compares gradient boosting machines against modern deep learning
# alternatives on the ETF case study, measuring rank IC and training time across
# walk-forward folds. The comparison grounds Section 12.3's decision framework
# with empirical evidence.
#
# ## Learning Objectives
# After completing this notebook, you will be able to:
# - Compare GBM, MLP, TabM, and TabPFN on financial tabular data
# - Evaluate models across walk-forward folds (not a single split)
# - Understand when deep learning offers practical advantages over GBMs
# - Interpret IC and training-time trade-offs for model selection
#
# **Prerequisites**: Requires case study features from
# `case_studies/etfs/features/` and modeling artifacts from Chapter 11 setup.
#
# ## Models Compared
# 1. **LightGBM**: Production GBM baseline
# 2. **MLP**: Sklearn neural network (minimal baseline)
# 3. **TabM**: Rank-one adapter MLP ensemble (Gorishniy et al., ICLR 2025)
# 4. **TabPFN**: Foundation model for small tabular data (if installed)
#
# ## Cross-References
# - **Section 12.3**: Decision framework and benchmark discussion
# - **Related**: `02_gbm_comparison` (GBM library comparison), `04_optuna_tuning` (HPO)

# %% [markdown]
# ## 1. Setup

# %%
"""Deep Learning vs GBMs, compare gradient boosting against modern deep learning for tabular financial data."""

import os
import time
import warnings
from collections import defaultdict
from typing import Any

# lightgbm and torch load before scikit-learn and ml4t.diagnostic: the first OpenMP
# runtime loaded wins the whole process, and an older libcudart otherwise wins symbol
# resolution.
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from IPython.display import Markdown, display
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.modeling import load_modeling_dataset
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_with_alt


def cross_sectional_ic_mean(y_true, y_pred, dates, symbols):
    pred_df = pl.DataFrame({"timestamp": dates, "symbol": symbols, "prediction": y_pred})
    ret_df = pl.DataFrame({"timestamp": dates, "symbol": symbols, "forward_return": y_true})
    ic_per_date = cross_sectional_ic_series(
        pred_df,
        ret_df,
        pred_col="prediction",
        ret_col="forward_return",
        date_col="timestamp",
        entity_col="symbol",
    )
    ic_clean = ic_per_date.drop_nans("ic").drop_nulls("ic")
    return float(ic_clean["ic"].mean()) if ic_clean.height else float("nan")


# %% tags=["parameters"]
# 0 = all folds
MAX_FOLDS = 0
TABM_EPOCHS = 200
# 0 = all symbols
MAX_SYMBOLS = 0
SEED = 42


# %%
set_global_seeds(SEED)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name()}")

# %% [markdown]
# ## 2. Load ETF Features

# %%
mds = load_modeling_dataset("etfs", "fwd_ret_21d", max_symbols=MAX_SYMBOLS)
df = mds.dataset.to_pandas()
date_col = mds.date_col
FEATURE_COLS = mds.feature_names

n_folds = len(mds.splits)
if MAX_FOLDS > 0:
    n_folds = min(n_folds, MAX_FOLDS)

print(f"ETFs: {len(df):,} rows, {len(FEATURE_COLS)} features, {n_folds} walk-forward folds")


# %% [markdown]
# ## 3. Model Definitions
#
# ### TabM: Rank-one Adapter MLP Ensemble
#
# TabM (Gorishniy et al., ICLR 2025) maintains a shared MLP backbone with
# $M$ rank-one scaling adapters. Each adapter creates a diverse ensemble member
# trained simultaneously. Individual members overfit; their average generalizes.
# This implements the core idea with PyTorch directly.


# %% [markdown]
# ### TabM Architecture


# %%
class TabMModel(torch.nn.Module):
    """Rank-1 adapter MLP ensemble for tabular data.

    Shared backbone + M rank-1 scaling vectors = efficient deep ensemble.
    """

    def __init__(self, n_features, hidden_dim=64, n_members=8):
        super().__init__()
        self.n_members: int = int(n_members)

        # Shared backbone
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        # Per-member rank-1 adapters (scaling vectors for last hidden layer)
        self.adapters = torch.nn.Parameter(torch.randn(n_members, hidden_dim) * 0.1)

        # Per-member output heads
        self.heads = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, 1) for _ in range(n_members)])

    def forward(self, x):
        h = self.backbone(x)  # (batch, hidden)
        outputs = []
        for i in range(self.n_members):
            h_adapted = h * self.adapters[i].unsqueeze(0)  # rank-1 scaling
            outputs.append(self.heads[i](h_adapted))
        return torch.stack(outputs, dim=0).mean(dim=0)  # average ensemble


# %% [markdown]
# ### TabM Training


# %%
def train_tabm(X_train, y_train, X_val, y_val, n_features, epochs=200, lr=1e-3, device="cpu"):
    """Train TabM with per-epoch train+val L1/MAE tracking; restore best-val checkpoint."""
    torch.manual_seed(SEED)
    if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
        torch.cuda.manual_seed_all(SEED)
    model = TabMModel(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    Xv_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        train_loss = torch.nn.functional.l1_loss(pred, y_t)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv_t)
            val_loss = torch.nn.functional.l1_loss(val_pred, yv_t)

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    history["best_epoch"] = int(np.argmin(history["val_loss"])) + 1
    history["best_val_loss"] = best_val_loss
    return model, history


# %% [markdown]
# ### TabM Inference


# %%
def predict_tabm(model, X_test, device="cpu"):
    """Generate predictions from TabM model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        return model(X_t).cpu().numpy().ravel()


# %% [markdown]
# ### Minimal MLP Baseline (PyTorch, GPU)
#
# A 64-32 ReLU MLP with Adam and L1 (MAE) loss, the simplest neural
# network we could write, implemented in PyTorch so it runs on the same
# GPU as TabM. This is the "minimal neural baseline" the chapter section
# refers to: fewer parameters than TabM, no ensembling, no architectural
# tricks. Same val/early-stop framework as TabM.


# %%
class TorchMLPModel(torch.nn.Module):
    """64-32 ReLU MLP for tabular regression."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_torch_mlp(
    X_train, y_train, X_val, y_val, n_features, epochs=200, lr=1e-3, patience=20, device="cpu"
):
    """Train a 64-32 MLP with per-epoch val tracking + early stopping on val plateau."""
    torch.manual_seed(SEED)
    if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
        torch.cuda.manual_seed_all(SEED)
    model = TorchMLPModel(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    Xv_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_since_best = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        train_loss = torch.nn.functional.l1_loss(pred, y_t)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv_t)
            val_loss = torch.nn.functional.l1_loss(val_pred, yv_t)

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = int(np.argmin(history["val_loss"])) + 1
    history["best_val_loss"] = best_val_loss
    history["stopped_at"] = epoch + 1
    return model, history


def predict_torch_mlp(model, X_test, device="cpu"):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        return model(X_t).cpu().numpy().ravel()


# %% [markdown]
# ## 4. Walk-Forward Evaluation
#
# We evaluate each model across all walk-forward folds, collecting per-fold IC
# and training time. This matches the temporal validation protocol emphasized
# in Section 12.3: single-split comparisons are unreliable for financial data.
#
# Rather than fix an epoch or tree count in advance, we carve a chronological
# **validation slice** off the end of each fold's train window - `VAL_FRACTION` of it -
# and use it as a held-out set during fitting:
#
# * **LightGBM**: `eval_set=[(X_val, y_val)]` plus `lgb.early_stopping(30)`;
#   we report the iteration count picked by early stopping (`best_iter`).
# * **MLP**: minimal 64-32 ReLU MLP in PyTorch, on whichever device this run has;
#   loop as TabM, with `patience=20` epochs on val-loss plateau.
# * **TabM**: track per-epoch train + val L1 (MAE) loss, restore the
#   best-val checkpoint at predict time, report `best_epoch` per fold.
#
# All three trainable models use **L1 (MAE)** loss, not MSE. On the codebase's
# 9 case-study GBM benchmark, MAE achieves the highest IC on 8 of 9
# regression-primary case studies (see `12_case_study_insights` §3a),
# because financial returns are heavy-tailed and squared-error loss chases
# the few large residuals at the expense of the cross-sectional ranking that
# IC rewards. A LightGBM regressor trained with MSE (`regression_l2`) on this
# fold structure still fits normally, dozens of trees, non-degenerate
# predictions, but its test IC comes out lower, negative on fold 0 where the
# MAE model is positive, which is why every trainable model here defaults to
# MAE.
# * **TabPFN**: zero-shot: no training, no validation needed.
#
# The test slice is held strictly out, none of the early-stopping signals
# touches it, so reported IC remains a fair walk-forward estimate.

# %%
# Preprocessing pipeline for neural models (impute + scale)
preprocess = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


# %% [markdown]
# ### TabPFN Evaluation


# %% [markdown]
# TabPFN ships its weights under a gated license, so the first prediction needs a free
# Prior Labs token: register at https://ux.priorlabs.ai, accept the license, and set
# `TABPFN_TOKEN` (see `.env.example`). The model is free for research and evaluation
# like this notebook and not for commercial use. Where the package or the token is
# missing the notebook says so and carries on without it; the comparison between the
# other three models is unaffected.

# %%
try:
    from tabpfn import TabPFNRegressor
    from tabpfn.errors import TabPFNError

    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("TabPFN not installed, skipping it. Install with: uv pip install tabpfn")


def _eval_tabpfn(
    X_train_scaled, y_train, X_test_scaled, y_test, dates_test, symbols_test, fold_idx
):
    max_samples = min(1000, len(X_train_scaled))
    start = time.time()
    tabpfn = TabPFNRegressor(n_estimators=4)
    tabpfn.fit(X_train_scaled[:max_samples], y_train[:max_samples])
    tabpfn_ic = cross_sectional_ic_mean(
        y_test, tabpfn.predict(X_test_scaled), dates_test, symbols_test
    )
    return f"TabPFN (n={max_samples})", {"ic": tabpfn_ic, "time": time.time() - start}


# %%
LGB_PARAMS: dict[str, Any] = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    objective="regression_l1",  # MAE, highest IC on 8/9 GBM case studies (see 12_case_study_insights §3a)
    random_state=SEED,
    verbose=-1,
)
MLP_EPOCHS = 200  # cap; early stopping on val plateau (patience 20)


# %% [markdown]
# ### Fold Data Preparation


# %%
VAL_FRACTION = 0.2  # last 20% of train (chronological) becomes validation


def _get_fold_data(split, val_frac=VAL_FRACTION):
    """Carve a chronologically-leading val slice off the train window so each
    model has a real held-out set to monitor during fitting."""
    train_mask = (df[date_col] >= split["train_start"]) & (df[date_col] <= split["train_end"])
    test_mask = (df[date_col] >= split["val_start"]) & (df[date_col] <= split["val_end"])

    train_df = df.loc[train_mask].sort_values(date_col)
    n_train = len(train_df)
    n_val = max(int(n_train * val_frac), 1)
    inner_df = train_df.iloc[: n_train - n_val]
    val_df = train_df.iloc[n_train - n_val :]

    X_inner = inner_df[FEATURE_COLS].values
    y_inner = inner_df[mds.label_col].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df[mds.label_col].values
    X_test = df.loc[test_mask, FEATURE_COLS].values
    y_test = df.loc[test_mask, mds.label_col].values
    dates_test = df.loc[test_mask, date_col].values
    symbols_test = df.loc[test_mask, mds.entity_cols[0]].values

    inner_valid = np.isfinite(y_inner)
    val_valid = np.isfinite(y_val)
    test_valid = np.isfinite(y_test)
    X_inner, y_inner = X_inner[inner_valid], y_inner[inner_valid]
    X_val, y_val = X_val[val_valid], y_val[val_valid]
    X_test, y_test = X_test[test_valid], y_test[test_valid]
    dates_test, symbols_test = dates_test[test_valid], symbols_test[test_valid]
    return X_inner, y_inner, X_val, y_val, X_test, y_test, dates_test, symbols_test


# %%
fold_inputs = []
for fold_idx in range(n_folds):
    split = mds.splits[fold_idx]
    fold_inputs.append((fold_idx, split, *_get_fold_data(split)))


# %%
fold_results = defaultdict(list)
tabm_histories = []  # one per fold for the learning-curves figure

for (
    fold_idx,
    split,
    X_inner,
    y_inner,
    X_val,
    y_val,
    X_test,
    y_test,
    dates_test,
    symbols_test,
) in fold_inputs:
    print(
        f"\nFold {fold_idx + 1}/{n_folds}: "
        f"{len(X_inner):,} train / {len(X_val):,} val / {len(X_test):,} test  "
        f"({split['train_start']}→{split['val_end']})"
    )

    X_inner_scaled = preprocess.fit_transform(X_inner)
    X_val_scaled = preprocess.transform(X_val)
    X_test_scaled = preprocess.transform(X_test)

    # LightGBM with explicit val + early stopping
    start = time.time()
    lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
    lgb_model.fit(
        X_inner,
        y_inner,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    lgb_ic = cross_sectional_ic_mean(y_test, lgb_model.predict(X_test), dates_test, symbols_test)
    fold_results["LightGBM"].append(
        {
            "ic": lgb_ic,
            "time": time.time() - start,
            "best_iter": int(lgb_model.best_iteration_ or LGB_PARAMS["n_estimators"]),
        }
    )

    # MLP, torch implementation on GPU, same val/early-stop framework as TabM
    start = time.time()
    mlp_model, mlp_history = train_torch_mlp(
        X_inner_scaled,
        y_inner,
        X_val_scaled,
        y_val,
        n_features=len(FEATURE_COLS),
        epochs=MLP_EPOCHS,
        device=device,
    )
    mlp_ic = cross_sectional_ic_mean(
        y_test,
        predict_torch_mlp(mlp_model, X_test_scaled, device=device),
        dates_test,
        symbols_test,
    )
    fold_results["MLP (64-32)"].append(
        {"ic": mlp_ic, "time": time.time() - start, "best_epoch": mlp_history["best_epoch"]}
    )

    # TabM with per-epoch val tracking + best-checkpoint restore
    start = time.time()
    tabm, history = train_tabm(
        X_inner_scaled,
        y_inner,
        X_val_scaled,
        y_val,
        n_features=len(FEATURE_COLS),
        epochs=TABM_EPOCHS,
        device=device,
    )
    tabm_ic = cross_sectional_ic_mean(
        y_test, predict_tabm(tabm, X_test_scaled, device=device), dates_test, symbols_test
    )
    fold_results["TabM (8-member)"].append(
        {"ic": tabm_ic, "time": time.time() - start, "best_epoch": history["best_epoch"]}
    )
    tabm_histories.append(history)

    # TabPFN, zero-shot, no training, no validation
    if TABPFN_AVAILABLE:
        try:
            tabpfn_name, tabpfn_result = _eval_tabpfn(
                X_inner_scaled, y_inner, X_test_scaled, y_test, dates_test, symbols_test, fold_idx
            )
            if tabpfn_name and tabpfn_result:
                fold_results[tabpfn_name].append(tabpfn_result)
        except TabPFNError as exc:
            TABPFN_AVAILABLE = False
            print(
                "\nTabPFN skipped, it needs a free Prior Labs token to download "
                "its model weights:\n"
                "  1. Register at https://ux.priorlabs.ai and accept the license\n"
                "  2. Copy your API key from https://ux.priorlabs.ai/account\n"
                "  3. Set TABPFN_TOKEN in your .env (see .env.example)\n"
                f"  (reason: {type(exc).__name__})\n"
                "The GBM / MLP / TabM comparison below is unaffected."
            )

    for model_name in fold_results:
        r = fold_results[model_name][-1] if fold_results[model_name] else None
        if r and len(fold_results[model_name]) == fold_idx + 1:
            extra = ""
            if "best_iter" in r:
                extra = f"  best_iter={r['best_iter']}"
            elif "best_epoch" in r:
                extra = f"  best_epoch={r['best_epoch']}"
            print(f"  {model_name:<20s} IC={r['ic']:+.4f}  ({r['time']:.1f}s){extra}")


# %% [markdown]
# ## 5. Results Summary

# %%
summary_rows = []
for model_name, folds in fold_results.items():
    ics = [f["ic"] for f in folds]
    times = [f["time"] for f in folds]
    row = {
        "model": model_name,
        "mean_ic": round(np.mean(ics), 4),
        "std_ic": round(np.std(ics), 4),
        "mean_time_s": round(np.mean(times), 2),
        "n_folds": len(folds),
    }
    # Mean stopping point reported by the early-stopping signal (where applicable)
    if folds and "best_iter" in folds[0]:
        row["mean_best_iter"] = int(np.mean([f["best_iter"] for f in folds]))
    elif folds and "best_epoch" in folds[0]:
        row["mean_best_epoch"] = int(np.mean([f["best_epoch"] for f in folds]))
    summary_rows.append(row)

results_df = pl.DataFrame(summary_rows).sort("mean_ic", descending=True)
results_df

# %% [markdown]
# ## 6. Visualization

# %%
models = results_df["model"].to_list()
mean_ics = results_df["mean_ic"].to_list()
std_ics = results_df["std_ic"].to_list()
mean_times = results_df["mean_time_s"].to_list()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# IC comparison
ax1 = axes[0]
bars1 = ax1.bar(range(len(models)), mean_ics, yerr=std_ics, capsize=4, color=COLORS["blue"])
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=15, ha="right")
ax1.set_ylabel("Rank IC (mean ± std)")
ax1.set_title("Mean rank IC by model, one standard deviation shown")
ax1.axhline(0, color="gray", linewidth=0.5)

# Training time comparison
ax2 = axes[1]
bars2 = ax2.bar(range(len(models)), mean_times, color=COLORS["blue_light"])
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=15, ha="right")
ax2.set_ylabel("Training Time (seconds)")
ax2.set_title("Mean training time per fold")

show_with_alt(
    fig,
    "Two bar panels sharing a model axis. Left: mean rank IC across the walk-forward "
    "folds with an error bar one standard deviation either side, drawn against a line "
    "at zero; every error bar is several times the height of the bar it sits on. "
    "Right: mean training time per fold in seconds.",
)

# %% [markdown]
# ### Learning Curves: Where the Models Actually Stop
#
# The IC table above hides *when* each model converged. The validation slice
# carved off each fold's train window lets us look at three pieces of evidence:
# the per-epoch TabM train/val L1 (MAE) trajectory (how quickly val-loss saturates
# or starts to climb again), the LightGBM `best_iter` per fold (where early
# stopping fired), and the TabM `best_epoch` per fold (which checkpoint we
# actually predicted with). On a high-noise target like 21-day forward returns,
# val-loss curves typically flatten within tens of epochs / hundreds of trees
# , the rest of the schedule is wasted compute or active overfitting.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel A: TabM train + val L1 loss per epoch, all folds overlaid
ax_lc = axes[0]
for fold_idx, history in enumerate(tabm_histories):
    epochs_axis = np.arange(1, len(history["train_loss"]) + 1)
    ax_lc.plot(
        epochs_axis,
        history["train_loss"],
        color=COLORS["blue"],
        alpha=0.35,
        linewidth=1,
        label="TabM train" if fold_idx == 0 else None,
    )
    ax_lc.plot(
        epochs_axis,
        history["val_loss"],
        color=COLORS["amber"],
        alpha=0.7,
        linewidth=1.2,
        label="TabM val" if fold_idx == 0 else None,
    )
    ax_lc.axvline(history["best_epoch"], color=COLORS["amber"], alpha=0.15, linewidth=0.8)

ax_lc.set_xlabel("Epoch")
ax_lc.set_ylabel("L1 (MAE) loss")
ax_lc.set_title("TabM train and validation loss by epoch, one line per fold")
ax_lc.legend(fontsize=8, loc="upper right")

# The iteration each model actually stopped at, per fold.
ax_stop = axes[1]
lgb_best_iters = [r["best_iter"] for r in fold_results["LightGBM"]]
tabm_best_epochs = [r["best_epoch"] for r in fold_results["TabM (8-member)"]]
fold_axis = np.arange(1, n_folds + 1)
ax_stop.plot(
    fold_axis,
    lgb_best_iters,
    "o-",
    color=COLORS["blue"],
    label=f"LightGBM best_iter (cap {LGB_PARAMS['n_estimators']})",
)
ax_stop.plot(
    fold_axis,
    tabm_best_epochs,
    "s-",
    color=COLORS["amber"],
    label=f"TabM best_epoch (cap {TABM_EPOCHS})",
)
ax_stop.axhline(LGB_PARAMS["n_estimators"], color=COLORS["blue"], linestyle=":", alpha=0.4)
ax_stop.axhline(TABM_EPOCHS, color=COLORS["amber"], linestyle=":", alpha=0.4)
ax_stop.set_xlabel("Walk-forward fold")
ax_stop.set_ylabel("Stopping point")
ax_stop.set_title("Selected stopping point per fold, against each cap")
ax_stop.set_xticks(fold_axis)
ax_stop.legend(fontsize=8, loc="best")

show_with_alt(
    fig,
    "Two panels. Left: TabM's L1 loss against epoch, a grey line per fold for training and "
    "an amber line per fold for validation, with a faint vertical line at each fold's "
    "selected epoch; the training lines fall throughout while most validation lines turn "
    "upward. Right: the stopping point each model reached on each fold, against a dotted "
    "line at that model's cap.",
)

# %% [markdown]
# The models are scored on the same folds, so the comparison that means something is
# the paired per-fold difference rather than two error bars that overlap.

# %% tags=["results"]
_fold_ic = {
    name: np.array([fold["ic"] for fold in folds]) for name, folds in fold_results.items() if folds
}
_baseline = "LightGBM"
_lines = []
for _name, _ics in _fold_ic.items():
    if _name == _baseline or len(_ics) != len(_fold_ic[_baseline]):
        continue
    _diff = _fold_ic[_baseline] - _ics
    _lines.append(
        f"- {_baseline} minus {_name}, per fold: mean {_diff.mean():+.4f}, "
        f"standard deviation {_diff.std(ddof=1):.4f}, ahead on "
        f"{int((_diff > 0).sum())} of {len(_diff)} folds."
    )
display(Markdown("\n".join(_lines)))

# %% [markdown]
# **What to read off it.** The error bars in the figure are each model's spread across
# folds, and they overlap heavily. That settles nothing either way, because all three
# models are scored on the same folds: whatever a fold does to one of them it largely
# does to the others, so the quantity to look at is the difference within each fold.
# The results cell above takes it, and reports how often the sign holds as well as how
# large the average difference is. A mean difference smaller than its own spread
# across folds, or a sign that flips on several folds, is not an ordering of
# architectures.
#
# The timing panel is the part that does not need a test. LightGBM trains in a fraction
# of TabM's time per fold and stops well short of its tree budget on every fold, and
# that difference is large enough that hardware and load cannot reverse it. Where two
# models are within noise of each other on accuracy, the one that is an order of
# magnitude cheaper to fit is the one you can afford to refit often.

# %% [markdown]
# ## 7. When to Use Deep Learning
#
# Section 12.3 sets out the decision framework; this notebook supplies the evidence for
# the row of it that covers noisy cross-sectional return prediction. Four things the run
# above establishes, in decreasing order of how sure they are:
#
# - **The cost difference is not close.** LightGBM fits in a fraction of TabM's time per
#   fold and stops well short of its tree budget on every fold. That gap is large enough
#   that hardware and load cannot reverse it, and it decides how often you can afford to
#   refit.
# - **The accuracy difference needs the paired test to mean anything.** The results cell
#   gives the per-fold difference and how often its sign holds. Read that rather than
#   the bar heights.
# - **A minimal MLP is a floor, not a contender.** It is here to show what capacity
#   alone does on tabular financial data, which is the premise Section 12.3 argues from.
# - **TabPFN is a probe you can afford before tuning anything**, when its gated weights
#   are available. Where the token is missing this notebook says so and scores the other
#   three.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Score the models on the same folds and difference them there.** Every model in
#    this notebook sees the same walk-forward folds, so the fold-to-fold swing that
#    dominates each model's error bar is largely shared and cancels in the difference.
#    Comparing the marginal spreads instead is how a comparison this noisy gets read as
#    a ranking.
#
# 2. **The loss function is doing real work on a heavy-tailed target.** L1 costs a large
#    error what it costs, where squared error lets a handful of extreme months set the
#    fit. On a 21-day return that is the difference between fitting the cross-section
#    and fitting its tails.
#
# 3. **Let the data pick the iteration count.** Both families here stop on a held-out
#    slice carved chronologically from the training window, and the stopping panel shows
#    how far below their caps they land. A fixed epoch or tree count is a
#    hyperparameter nobody measured.
#
# 4. **When the accuracy difference sits inside the noise, the cost of refitting is
#    what is left to choose on.** That is an operational argument rather than a
#    statistical one, and the timing panel is where it is made.
#
# **Next**: See `04_optuna_tuning` for Bayesian hyperparameter optimization.
