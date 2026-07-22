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
# # TSMixer: MLP-Based Time Series Mixing
#
# **Docker image**: `ml4t-gpu`
#
# This notebook implements TSMixer (Google, 2023) for predicting forward ETF
# returns. TSMixer uses MLPs only — no attention or convolutions — alternating
# between **time-mixing** (a per-feature MLP over the time axis) and
# **feature-mixing** (a per-timestep MLP over the feature axis).
#
# **Learning Objectives**:
# - Implement the TSMixer architecture: alternating time and feature mixing MLPs
# - Understand how transposing the input tensor enables mixing along different axes
# - Compare pure-MLP mixing against a Ridge regression baseline
# - Explore sktime's TinyTimeMixer for foundation-model-based mixing
#
# **Book Reference**: Chapter 13, Section 13.6 (The Full Practitioner Toolkit)
#
# **Prerequisites**: ETF features (`case_studies/etfs/`)

# %%
"""TSMixer — alternating time and feature mixing MLPs for return prediction."""

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
LOOKBACK = 60
D_MODEL = 32
N_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3

# %%

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

set_global_seeds(SEED)

# %% [markdown]
# ## Data Loading
#
# We use ETF features from the case study pipeline. The first 8 features
# provide a diverse mix of momentum, volatility, and cross-sectional signals.

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
# Date-based 60/20/20 temporal split
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
# Mean cross-sectional Spearman IC by date — same metric used in
# `01_core_architectures` and `04_transformers` so TSMixer's signal-quality
# comparison anchors on the same per-date Spearman rank correlation as the
# other Section 13.6 architectures.


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
# ## TSMixer Architecture
#
# TSMixer alternates between two types of MLP blocks:
#
# 1. **Time-mixing**: Transposes to `(batch, features, time)` and applies an MLP
#    along the time axis -- each feature learns its own temporal pattern
# 2. **Feature-mixing**: Applies an MLP along the feature axis -- each timestep
#    learns cross-variate interactions
#
# Both use residual connections and layer normalization. This is conceptually
# similar to the MLP-Mixer vision architecture, adapted for time series.
#
# The mixing operations can be written as:
#
# $$\mathbf{X}' = \text{LayerNorm}\bigl(\mathbf{X} + W_2 \cdot \sigma(W_1 \cdot \mathbf{X}^\top)^\top\bigr)$$
#
# for time-mixing, and similarly without the transpose for feature-mixing.


# %%
class TimeMixingMLP(nn.Module):
    """Mix information across the time dimension.

    Transposes input to (batch, features, time), applies MLP on the time
    axis, then transposes back. This lets each feature channel learn its
    own temporal mixing weights.
    """

    def __init__(self, seq_len: int, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, seq_len)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_features)

    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        return self.norm(x + residual)


# %% [markdown]
# ### Feature-Mixing MLP
#
# Operates directly on the feature dimension at each timestep, learning
# cross-variate interactions without transposing.


# %%
class FeatureMixingMLP(nn.Module):
    """Mix information across the feature dimension.

    Applies MLP on the feature axis at each timestep, enabling
    cross-variate interaction learning.
    """

    def __init__(self, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_features)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.norm(x + residual)


# %% [markdown]
# ### Mixer Block
#
# Pairs one time-mixing step with one feature-mixing step, forming the
# fundamental building block of TSMixer.


# %%
class MixerBlock(nn.Module):
    """One block of TSMixer: time-mixing followed by feature-mixing."""

    def __init__(self, seq_len: int, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.time_mix = TimeMixingMLP(seq_len, n_features, hidden_dim, dropout)
        self.feature_mix = FeatureMixingMLP(n_features, hidden_dim, dropout)

    def forward(self, x):
        x = self.time_mix(x)
        x = self.feature_mix(x)
        return x


# %% [markdown]
# ### TSMixer Regressor
#
# Stacks multiple mixer blocks, then mean-pools over the time dimension
# and projects to a scalar output. Mean pooling (vs. flattening) reduces
# parameter count and improves generalization for longer sequences.


# %%
class TSMixerRegressor(nn.Module):
    """TSMixer for regression: stack mixer blocks, mean pool, linear head."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        n_blocks: int = 2,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[MixerBlock(seq_len, n_features, hidden_dim, dropout) for _ in range(n_blocks)]
        )
        # Mean pool over time, then project features to scalar
        self.head = nn.Linear(n_features, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.blocks(x)
        x = x.mean(dim=1)  # (batch, n_features) - pool over time
        return self.head(x).squeeze(-1)  # (batch,)


# %%
model = TSMixerRegressor(
    seq_len=LOOKBACK,
    n_features=len(FEATURE_COLS),
    n_blocks=N_LAYERS,
    hidden_dim=D_MODEL,
    dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"TSMixer parameters: {n_params:,}")
print(f"Architecture: {N_LAYERS} mixer blocks, hidden_dim={D_MODEL}")
print(f"Input: ({LOOKBACK} timesteps, {len(FEATURE_COLS)} features)")

# %% [markdown]
# ## Train TSMixer

# %%
print("Training TSMixer...")
history = train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    EPOCHS,
    LR,
    BATCH_SIZE,
    DEVICE,
    weight_decay=0.01,
)

# %% [markdown]
# ## Evaluate on Test Set

# %%
model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_pred = model(X_test_t).cpu().numpy()

test_mse = np.mean((y_pred - y_test) ** 2)
test_ic = cross_sectional_ic_mean(y_test, y_pred, test_dates, test_symbols)

print("\nTSMixer Test Results:")
print(f"  MSE: {test_mse:.6f}")
print(f"  Spearman IC: {test_ic:.4f}")

# %% [markdown]
# ## Ridge Baseline Comparison
#
# Flattening the 3D input to 2D and fitting Ridge regression provides a
# simple linear baseline to gauge whether TSMixer's learned mixing adds value.

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
# ## sktime TinyTimeMixer
#
# TinyTimeMixer (TTM) is a pre-trained foundation model based on the TSMixer
# architecture. It was trained on a large corpus of time series and can be
# used zero-shot or fine-tuned. Here we demonstrate the sktime wrapper for
# univariate SPY forecasting.
#
# **Dependency note**: sktime's neural forecasters require `neuralforecast`, which
# depends on `ray` — and ray does not yet support Python 3.14
# ([ray-project/ray#56434](https://github.com/ray-project/ray/issues/56434)).
# Once ray adds 3.14 wheels, install via `uv pip install neuralforecast`
# and uncomment the demo below.

# %%
# import polars as pl
# from sktime.forecasting.ttm import TinyTimeMixerForecaster
#
# from data import load_etfs
#
# # Load SPY univariate series
# spy = load_etfs(symbols=["SPY"]).sort("timestamp")
# spy_pd = spy.select(["timestamp", "close"]).to_pandas().set_index("timestamp")["close"]
#
# split = int(len(spy_pd) * 0.8)
# y_train_sk = spy_pd.iloc[:split]
# fh_sk = list(range(1, 11))
#
# forecaster = TinyTimeMixerForecaster(fit_strategy="zero-shot")
# forecaster.fit(y_train_sk, fh=fh_sk)
# y_pred_sk = forecaster.predict()
# print(f"sktime TinyTimeMixer: predicted {len(y_pred_sk)} steps ahead")
# print(y_pred_sk)

# %% [markdown]
# ## Summary

# %%
results_df = pl.DataFrame(
    {
        "Model": ["TSMixer", "Ridge"],
        "Spearman IC": [test_ic, ridge_ic],
        "MSE": [test_mse, ridge_mse],
        "Parameters": [n_params, None],
    }
)
results_df

# %% [markdown]
# **Interpretation**: on this single-split multivariate ETF-feature setup
# (eight momentum/volatility features per ETF, asset-pooled over the 60-day
# lookback), TSMixer trails Ridge on cross-sectional Spearman IC while both
# are close on MSE — see `results_df` for the point estimates. The MLP mixer
# adds capacity (time-mixing followed by feature-mixing, both with residual
# + LayerNorm) but not ranking signal on this reduced feature panel. At
# roughly 9K parameters TSMixer is about a third the parameter count of the
# Transformer variants in `04_transformers`, so the result is informative
# regardless of direction: a lighter pure-MLP architecture does not beat
# linear regression on these eight features under a single temporal split.
# Whether richer feature sets or walk-forward evaluation change the picture
# is settled in Section 13.9.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Pure MLP architecture**: TSMixer uses no attention or convolutions; on
#    this single-split task it trails the Ridge baseline on cross-sectional IC
#    at roughly 9K parameters — the mixing capacity does not add ranking signal
#    on this 8-feature panel
# 2. **Time-mixing via transpose**: By permuting to `(batch, features, time)`,
#    a standard MLP effectively learns temporal patterns per feature channel
# 3. **Feature-mixing for cross-variate learning**: The alternating design lets
#    the model learn both temporal dynamics and feature interactions
# 4. **Architecturally transparent**: Fewer parameters than Transformers and a
#    clear separation of which block mixes along which axis. "Interpretable"
#    is too strong for the resulting attention-free representation — the MLP
#    weights themselves are opaque even though the block geometry is not
# 5. **TinyTimeMixer**: Foundation model pre-training extends the TSMixer idea
#    to zero-shot and few-shot forecasting
#
# **Next**: See `07_mamba_ssm` for state space models that offer an alternative
# to both attention and MLP mixing.
#
# **Book**: Section 13.6 discusses TSMixer alongside other non-attention architectures.
