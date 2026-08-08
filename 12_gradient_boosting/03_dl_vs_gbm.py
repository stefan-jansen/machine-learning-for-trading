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
# 1. **LightGBM** — Production GBM baseline
# 2. **MLP** — Sklearn neural network (minimal baseline)
# 3. **TabM** — Rank-one adapter MLP ensemble (Gorishniy et al., ICLR 2025)
# 4. **TabPFN** — Foundation model for small tabular data (if installed)
#
# ## Cross-References
# - **Section 12.3**: Decision framework and benchmark discussion
# - **Related**: `02_gbm_comparison` (GBM library comparison), `04_optuna_tuning` (HPO)

# %% [markdown]
# ## 1. Setup

# %%
"""Deep Learning vs GBMs — compare gradient boosting against modern deep learning for tabular financial data."""

import os
import time
import warnings
from collections import defaultdict
from typing import Any

# lightgbm and torch must be imported before scikit-learn. All three ship their
# own OpenMP runtime and the first one loaded wins for the whole process; on
# macOS ARM64, loading scikit-learn's libomp first makes LightGBM's first
# multithreaded fit segfault in __kmp_suspend_initialize_thread. Plain `import`
# statements sort ahead of `from ... import` ones, so this order is what isort
# produces and will not drift back. torch also has to precede ml4t.diagnostic,
# which dlopens an older libcudart and otherwise wins symbol resolution.
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.modeling import load_modeling_dataset
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

warnings.filterwarnings("ignore")


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
MAX_FOLDS = 0  # 0 = all folds
TABM_EPOCHS = 200
MAX_SYMBOLS = 0  # 0 = all symbols
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
# A 64-32 ReLU MLP with Adam and L1 (MAE) loss — the simplest neural
# network we could write — implemented in PyTorch so it runs on the same
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
# in Section 12.3 — single-split comparisons are unreliable for financial data.
#
# To avoid the "fixed-epoch / fixed-tree-count" anti-pattern that DL chapters
# warn about, we carve a chronologically-leading **validation slice** (last 20%
# of each fold's train window) and use it as a real held-out set during fitting:
#
# * **LightGBM** — `eval_set=[(X_val, y_val)]` plus `lgb.early_stopping(30)`;
#   we report the iteration count picked by early stopping (`best_iter`).
# * **MLP** — minimal 64-32 ReLU MLP in PyTorch (GPU); same val/early-stop
#   loop as TabM, with `patience=20` epochs on val-loss plateau.
# * **TabM** — track per-epoch train + val L1 (MAE) loss, restore the
#   best-val checkpoint at predict time, report `best_epoch` per fold.
#
# All three trainable models use **L1 (MAE)** loss, not MSE. On the codebase's
# 9 case-study GBM benchmark, MAE achieves the highest IC on 8 of 9
# regression-primary case studies (see `12_case_study_insights` §3a),
# because financial returns are heavy-tailed and squared-error loss chases
# the few large residuals at the expense of the cross-sectional ranking that
# IC rewards. A LightGBM regressor trained with MSE (`regression_l2`) on this
# fold structure still fits normally — dozens of trees, non-degenerate
# predictions — but its test IC comes out lower, negative on fold 0 where the
# MAE model is positive, which is why every trainable model here defaults to
# MAE.
# * **TabPFN** — zero-shot: no training, no validation needed.
#
# The test slice is held strictly out — none of the early-stopping signals
# touches it — so reported IC remains a fair walk-forward estimate.

# %%
# Preprocessing pipeline for neural models (impute + scale)
preprocess = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


# %% [markdown]
# ### TabPFN Evaluation


# %%
# TabPFN ships its weights under a gated license, so the first prediction needs
# a free Prior Labs token (register at https://ux.priorlabs.ai, accept the
# license, set TABPFN_TOKEN — see .env.example). The model is free for research
# and evaluation use like this notebook, but not for commercial/production use.
# If the package or token is missing we skip TabPFN with a clear message rather
# than crash; the GBM / MLP / TabM comparison is unaffected.
try:
    from tabpfn import TabPFNRegressor
    from tabpfn.errors import TabPFNError

    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("TabPFN not installed — skipping it. Install with: uv pip install tabpfn")


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
    objective="regression_l1",  # MAE — highest IC on 8/9 GBM case studies (see 12_case_study_insights §3a)
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

    # MLP — torch implementation on GPU, same val/early-stop framework as TabM
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

    # TabPFN — zero-shot, no training, no validation
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
                "\nTabPFN skipped — it needs a free Prior Labs token to download "
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
ax1.set_title("LightGBM Leads Test IC Across 8 Walk-Forward Folds")
ax1.axhline(0, color="gray", linewidth=0.5)

# Training time comparison
ax2 = axes[1]
bars2 = ax2.bar(range(len(models)), mean_times, color=COLORS["blue_light"])
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=15, ha="right")
ax2.set_ylabel("Training Time (seconds)")
ax2.set_title("TabM's Ensemble Costs the Most Training Time Per Fold")

plt.tight_layout()
plt.show()

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
# — the rest of the schedule is wasted compute or active overfitting.

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
ax_lc.set_title(f"TabM Val Loss Bottoms Early, Then Overfits ({n_folds} Folds)")
ax_lc.legend(fontsize=8, loc="upper right")

# Panel B: LightGBM best_iter and TabM best_epoch per fold
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
ax_stop.set_title("Early Stopping Fires Far Below the Tree/Epoch Caps")
ax_stop.set_xticks(fold_axis)
ax_stop.legend(fontsize=8, loc="best")

plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation**: The results illustrate the high-noise regime of monthly
# return prediction on a small ETF universe. With L1/MAE loss across all
# trainable models, the summary table above ranks **LightGBM first**, then
# TabM, then the minimal 64-32 MLP: capacity alone is not enough to extract
# signal from this feature set. (TabPFN was skipped in this run - its
# gated-license weights need a free Prior Labs token - so only the three
# trainable models are scored here.) LightGBM also trains in well under a
# second per fold, on par with the minimal MLP and several times faster than
# TabM's ensemble, with early stopping firing after a small fraction of the
# tree budget. That reinforces the operational case for GBMs even where a DL
# alternative is within striking distance on accuracy. Wall-clock times are
# hardware-dependent and swing run to run; only the orders of magnitude and
# the TabM-is-slowest ranking are stable.
#
# Read the ordering, not the digits. Per-fold IC swings are larger than the
# gaps between models - each model's per-fold range in the table above spans
# both signs - so a single fold's ranking is not evidence of architectural
# superiority. What carries the argument is that the ordering holds across 8
# chronologically distinct folds. Tuned GBMs in `04_optuna_tuning` widen the
# gap further.

# %% [markdown]
# ## 7. When to Use Deep Learning
#
# See Section 12.3 for the full decision framework. This notebook provides the
# empirical evidence behind the "Recommended family" column. Key findings:
#
# - **LightGBM (with L1/MAE loss + early stopping) leads on this benchmark**,
#   at under a second per fold. Operational case is simple too: minimal
#   infrastructure, tight tuning loop, mature ecosystem.
# - **TabM** is the strongest tabular DL contender, validating its
#   parameter-efficient-ensembling premise without dethroning the GBM.
#   Best-val checkpointing fires well before the 200-epoch cap on most folds
#   and near it on others, so the cap is doing real work for some folds and
#   wasting compute on the rest.
# - **TabPFN** — the gated-license foundation model — was skipped in this
#   production run because its weights need a free Prior Labs token. Install
#   the token (see the skip note above) to add a zero-shot signal-checking
#   probe before investing in tuning.
# - **A minimal 64-32 ReLU MLP** is the simplest neural net you can write and
#   lands visibly below TabM, confirming the chapter's premise that capacity
#   alone is not enough on tabular financial data; you need either an ensemble
#   (TabM), a foundation model (TabPFN), or a real tabular learner (LightGBM).

# %% [markdown]
# ## Key Takeaways
#
# 1. **LightGBM with L1/MAE + early stopping leads**, at under a second per
#    fold. TabM is the strongest tabular DL contender and the minimal 64-32
#    MLP trails both: capacity alone is not a strategy. (TabPFN was skipped
#    this run - its gated-license weights need a Prior Labs token.)
# 2. **The loss function is load-bearing on noisy financial targets.**
#    Swapping LightGBM's objective from MSE to MAE on this 21-day-return
#    target turns fold 0's test IC from negative to positive and makes MAE the
#    benchmark leader: squared error over-weights the heavy-tailed large
#    residuals and erodes the cross-sectional ranking. The codebase's 9
#    case-study GBM benchmark independently confirms MAE achieves the
#    highest IC on most regression-primary case studies (see
#    `12_case_study_insights` §3a).
# 3. **Validation curves matter even when accuracy is noisy.** TabM's
#    train/val L1 trajectory and LightGBM's `best_iter` distribution show
#    where each model actually converges; the IC table alone hides the
#    overfitting risk that fixed-epoch / fixed-tree-count schedules
#    introduce. Always carve a held-out val slice and stop on it.
# 4. **Walk-forward variance is large** - per-fold IC swings wider than the
#    gaps between models make single-split comparisons unreliable. Report
#    mean ± std across folds, not a single number.
# 5. **Tuning matters more than architecture** — default hyperparameters
#    leave performance on the table for every family; Section 12.4 (Optuna)
#    shows how proper hyperparameter tuning closes the remaining gap.
#
# **Next**: See `04_optuna_tuning` for Bayesian hyperparameter optimization.
