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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Temporal Convolutional Network (TCN)
#
# **Docker image**: `ml4t-gpu`
#
# This notebook implements a Temporal Convolutional Network for predicting
# forward returns on ETFs. TCNs use dilated causal convolutions to capture
# long-range temporal dependencies without the sequential bottleneck of
# recurrent networks.
#
# **Learning Objectives**:
# - Implement causal convolutions that prevent information leakage from the future
# - Build a TCN with exponentially growing dilations (1, 2, 4, 8) for efficient
#   receptive field coverage
# - Understand the tradeoff between receptive field size and model depth
# - Compare TCN predictions against a Ridge regression baseline
#
# **Book Reference**: Chapter 13, Section 13.6 (The Full Practitioner Toolkit)
#
# **Prerequisites**: ETF features (`case_studies/etfs/`)

# %%
"""Temporal Convolutional Network — build TCN with dilated causal convolutions for return prediction."""

import warnings

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils.reproducibility import set_global_seeds

warnings.filterwarnings("ignore")

from dl_sequences import create_sequences_multi_asset, load_dl_dataset, train_model

# %% tags=["parameters"]
SEED = 42
EPOCHS = 30
LOOKBACK = 60
BATCH_SIZE = 128
N_CHANNELS = 32
KERNEL_SIZE = 3
DROPOUT = 0.1
LR = 1e-3

# %%

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

set_global_seeds(SEED)

# %% [markdown]
# ## Data Loading
#
# We use ETF features from the case study pipeline, providing diverse
# time series characteristics for testing TCN architectures.

# %%
mds = load_dl_dataset("etfs")

FEATURE_COLS = mds.feature_names[:8]
TARGET_COL = mds.label_col

print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Target: {TARGET_COL}")

# %% [markdown]
# ## Sequence Creation and Temporal Split

# %%
df = mds.dataset.drop_nulls(subset=FEATURE_COLS + [TARGET_COL])
print(f"Rows after dropping nulls: {len(df):,}")

X, y, timestamps, symbols = create_sequences_multi_asset(
    df,
    FEATURE_COLS,
    TARGET_COL,
    LOOKBACK,
    timestamp_col=mds.date_col,
    symbol_col=mds.entity_cols[0],
)
print(f"Sequences: {X.shape[0]:,}, shape: {X.shape}")

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y = np.nan_to_num(y, nan=0.0).astype(np.float32)

# %%
# Date-based 60/20/20 temporal split: all asset-rows for dates < train_end_date
# go to train, etc. Sample-order slicing on the asset-pooled sequences would
# split cross-asset, not temporally.
unique_dates = np.sort(np.unique(timestamps))
train_end_date = unique_dates[int(len(unique_dates) * 0.6)]
val_end_date = unique_dates[int(len(unique_dates) * 0.8)]

train_mask = timestamps < train_end_date
val_mask = (timestamps >= train_end_date) & (timestamps < val_end_date)
test_mask = timestamps >= val_end_date

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]
test_dates, test_symbols = timestamps[test_mask], symbols[test_mask]

print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")


# %% [markdown]
# ### Cross-sectional IC helper
#
# Mean cross-sectional Spearman IC by date — TCN evaluation is on cross-asset
# ranking, not pooled point error, so the same date/entity-aware metric used in
# `01_core_architectures` and `04_transformers` is the right comparison anchor.


# %%
def cross_sectional_ic_mean(y_true, y_pred, dates, syms):
    """Mean cross-sectional Spearman IC across dates."""
    pred_df = pl.DataFrame({"timestamp": dates, "symbol": syms, "prediction": y_pred})
    ret_df = pl.DataFrame({"timestamp": dates, "symbol": syms, "forward_return": y_true})
    ic_per_date = cross_sectional_ic_series(
        pred_df,
        ret_df,
        pred_col="prediction",
        ret_col="forward_return",
        date_col="timestamp",
        entity_col="symbol",
    )
    ic_clean = ic_per_date.drop_nulls("ic")
    return float(ic_clean["ic"].mean()) if ic_clean.height else float("nan")


# %% [markdown]
# > **Note**: This fixed 60/20/20 split is a pedagogical simplification. Production
# > deployment requires the walk-forward validation protocol from Chapter 6, where
# > the model is retrained on expanding windows to avoid temporal data leakage.

# %% [markdown]
# ## TCN Architecture
#
# The TCN consists of stacked causal convolution blocks with exponentially
# increasing dilation factors. Each block uses:
#
# 1. **Causal padding**: Left-pad the input so the convolution only sees past
#    and present timesteps, never the future
# 2. **Dilated convolutions**: Dilation factors of 1, 2, 4, 8 give an
#    exponentially growing receptive field
# 3. **Residual connections**: Enable training deeper networks
#
# The receptive field grows as:
#
# $$R = 1 + \sum_{i=0}^{L-1} 2 \cdot (k-1) \cdot d_i$$
#
# where $k$ is kernel size, $d_i = 2^i$ is the dilation at layer $i$, and
# $L$ is the number of layers.


# %%
class CausalConv1d(nn.Module):
    """1D convolution with causal (left) padding.

    Ensures the output at time t depends only on inputs at times <= t.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.causal_padding,
            dilation=dilation,
        )

    def forward(self, x):
        out = self.conv(x)
        # Trim the right side to enforce causality
        if self.causal_padding > 0:
            out = out[:, :, : -self.causal_padding]
        return out


# %% [markdown]
# ### TCN Block
#
# Each block applies two causal convolutions with batch normalization,
# ReLU activation, and dropout. A residual connection bypasses the block,
# using a 1x1 convolution when channel dimensions change.
#
# > **Strict-online-causality footnote**: BatchNorm uses running batch
# > statistics that are estimated during training and applied at inference
# > time. For strict per-sample online causality (sample-by-sample inference
# > with no dependence on what other samples looked like at training time),
# > the published TCN reference (Bai, Kolter, Koltun 2018) uses WeightNorm
# > on the conv weights instead. We keep BatchNorm here because the original
# > paper's Conv1d activation statistics are load-bearing for the
# > cross-sectional IC signal on this 8-feature ETF panel — alternative
# > normalizations (LayerNorm-over-channels, GroupNorm(1)) were tried in
# > the publication-polish pass and both collapsed TCN IC to near zero
# > while leaving Ridge IC unchanged. The footnote is the right place to
# > flag the strict-causality concern for production deployment.


# %%
class TCNBlock(nn.Module):
    """Residual block with two dilated causal convolutions."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        return self.relu(out + res)


# %% [markdown]
# ### Full TCN Regressor
#
# Stacking blocks with dilations [1, 2, 4, 8] followed by adaptive average
# pooling and a linear head for regression output.


# %%
class TCNRegressor(nn.Module):
    """Temporal Convolutional Network for regression.

    Architecture: Input -> [CausalConv(d=2^i) + ReLU + Dropout] x 4
                  -> AdaptiveAvgPool1d -> Linear -> scalar output.
    """

    def __init__(
        self,
        n_features: int,
        n_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        blocks = []
        for i, d in enumerate(dilations):
            in_ch = n_features if i == 0 else n_channels
            blocks.append(TCNBlock(in_ch, n_channels, kernel_size, d, dropout))

        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_channels, 1)

    def forward(self, x):
        # x input: (batch, seq_len, n_features) -> permute to (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)  # (batch, n_channels)
        return self.fc(x).squeeze(-1)  # (batch,)


# %%
# Calculate receptive field
dilations = (1, 2, 4, 8)
receptive_field = 1 + sum(2 * (KERNEL_SIZE - 1) * d for d in dilations)
print(f"TCN receptive field: {receptive_field} timesteps")
print(f"Lookback window: {LOOKBACK} timesteps")
if receptive_field >= LOOKBACK:
    print("Receptive field covers the full lookback window")

# %% [markdown]
# ## Train the TCN

# %%
model = TCNRegressor(
    n_features=len(FEATURE_COLS),
    n_channels=N_CHANNELS,
    kernel_size=KERNEL_SIZE,
    dropout=DROPOUT,
    dilations=dilations,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"TCN parameters: {n_params:,}")
print(
    f"Architecture: 4 blocks (dilations={list(dilations)}), {N_CHANNELS} channels, kernel={KERNEL_SIZE}"
)

history = train_model(model, X_train, y_train, X_val, y_val, EPOCHS, LR, BATCH_SIZE, DEVICE)

# %% [markdown]
# ## Evaluate on Test Set

# %%
model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_pred = model(X_test_t).cpu().numpy()

test_mse = np.mean((y_pred - y_test) ** 2)
test_ic = cross_sectional_ic_mean(y_test, y_pred, test_dates, test_symbols)

print("\nTCN Test Results:")
print(f"  MSE: {test_mse:.6f}")
print(f"  Spearman IC: {test_ic:.4f}")

# %% [markdown]
# ## Ridge Baseline Comparison

# %%
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_ridge_pred = ridge.predict(X_test_scaled)

ridge_mse = np.mean((y_ridge_pred - y_test) ** 2)
ridge_ic = cross_sectional_ic_mean(y_test, y_ridge_pred, test_dates, test_symbols)

print("\nRidge Baseline Results:")
print(f"  MSE: {ridge_mse:.6f}")
print(f"  Spearman IC: {ridge_ic:.4f}")

# %% [markdown]
# ## sktime Alternative
#
# sktime wraps NeuralForecast's TCN implementation for quick prototyping.
#
# **Dependency note**: sktime's neural forecasters require `neuralforecast`, which
# depends on `ray` — and ray does not yet support Python 3.14
# ([ray-project/ray#56434](https://github.com/ray-project/ray/issues/56434)).
# Once ray adds 3.14 wheels, install via `uv pip install neuralforecast`
# and uncomment the demo below.

# %%
# import pandas as pd
# from sktime.forecasting.neuralforecast import NeuralForecastTCN
#
# from data import load_etfs
#
# spy = load_etfs(symbols=["SPY"]).sort("timestamp")
# spy_pd = spy.select(["timestamp", "close"]).to_pandas().set_index("timestamp")["close"]
# split = int(len(spy_pd) * 0.8)
# y_train_sk = spy_pd.iloc[:split]
#
# forecaster = NeuralForecastTCN(
#     freq="B",
#     input_size=LOOKBACK,
#     max_steps=EPOCHS * 5,
# )
# forecaster.fit(y_train_sk)
# y_pred_sk = forecaster.predict(fh=list(range(1, 11)))
# print(f"sktime NeuralForecastTCN: predicted {len(y_pred_sk)} steps")

# %% [markdown]
# ## Summary

# %%
results_df = pl.DataFrame(
    {
        "Model": ["TCN", "Ridge"],
        "Spearman IC": [test_ic, ridge_ic],
        "MSE": [test_mse, ridge_mse],
    }
)
results_df

# %% [markdown]
# **Interpretation**: on this single-split multivariate ETF-feature setup
# (eight cross-sectional momentum features per ETF, asset-pooled over the
# 60-day lookback), the TCN's cross-sectional Spearman IC narrowly edges the
# Ridge baseline (TCN 0.017 vs Ridge 0.016 — see `results_df` above), but the
# margin is well within single-split noise and the ordering is not stable
# across reruns. The cuDNN-non-deterministic path through the
# dilated causal convolutions drives most of that run-to-run variation; the deeper
# pedagogical point is that TCN's signal at this scale and 8-feature panel
# is at the noise floor and that single-split point estimates are not
# trustworthy. The receptive field of 61 timesteps fully covers the 60-day
# lookback, so no information is discarded by the architecture itself.
# Section 13.9's walk-forward evaluation across more case studies is the
# authoritative TCN reading; treat this notebook as an architecture
# demonstration, not a production estimate.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Causal convolutions prevent lookahead bias**: Left-padding ensures
#    each output depends only on past and present inputs
# 2. **Exponential dilations are efficient**: Dilations of 1, 2, 4, 8 cover
#    a receptive field of 61 timesteps with only 4 layers
# 3. **Fully parallelizable**: Unlike LSTMs, all timesteps are processed
#    simultaneously during both training and inference
# 4. **Fixed receptive field**: The maximum lookback is determined at design
#    time by the dilation schedule -- unlike attention, which adapts dynamically
# 5. **BatchNorm vs strict online-causality**: BatchNorm's running-stats
#    inference behaviour means a deployed model's per-sample output depends
#    on aggregate training-batch statistics. The Bai et al. (2018) reference
#    uses WeightNorm instead for that reason. We keep BatchNorm because, on
#    this 8-feature ETF panel, alternative normalizations (LayerNorm-over-
#    channels, GroupNorm(1)) both collapsed the TCN's cross-sectional IC to
#    near zero. The footnote at the block definition is the right place to
#    flag the strict-causality concern; production deployments targeting
#    streaming inference should swap to WeightNorm and retune
#
# **Next**: See `06_tsmixer` for an MLP-only alternative that achieves
# competitive results without convolutions or attention.
# **Book**: Section 13.6 compares TCN with other non-attention architectures.
