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
# returns. TSMixer uses MLPs only - no attention or convolutions - alternating
# between **time-mixing** (a per-feature MLP over the time axis) and
# **feature-mixing** (a per-timestep MLP over the feature axis).
#
# **Learning Objectives**:
# - Implement the TSMixer architecture: alternating time and feature mixing MLPs
# - Understand how transposing the input tensor enables mixing along different axes
# - Compare pure-MLP mixing against a Ridge regression baseline
#
# **Book Reference**: Chapter 13, Section 13.6 (The Full Practitioner Toolkit)
#
# **Prerequisites**: ETF features (`case_studies/etfs/`)

# %%
"""Build TSMixer with alternating time and feature mixing for return prediction."""

import os
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch
import torch.nn as nn
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils.reproducibility import set_global_seeds
from utils.style import COLORS

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
LABEL_HORIZON = 21

# %%

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

set_global_seeds(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# %% [markdown]
# ## Data Loading
#
# We use eight fixed momentum horizons from the ETF case-study pipeline.

# %%
mds = load_dl_dataset("etfs")

FEATURE_COLS = [
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "ret_42d",
    "ret_63d",
    "ret_126d",
    "ret_189d",
    "ret_252d",
]
TARGET_COL = mds.label_col

missing_features = sorted(set(FEATURE_COLS) - set(mds.feature_names))
if missing_features:
    raise ValueError(f"Missing required ETF momentum features: {missing_features}")

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

sequence_order = np.lexsort((symbols.astype(str), timestamps))
X = np.nan_to_num(X[sequence_order], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y = np.nan_to_num(y[sequence_order], nan=0.0).astype(np.float32)
timestamps = timestamps[sequence_order]
symbols = symbols[sequence_order]

# %%
# Date-based 60/20/20 temporal split. The target is a 21-day forward return,
# so labels whose outcome windows cross the next boundary are purged.
unique_dates = np.sort(np.unique(timestamps))
train_boundary_idx = int(len(unique_dates) * 0.6)
val_boundary_idx = int(len(unique_dates) * 0.8)
train_end_date = unique_dates[train_boundary_idx]
val_end_date = unique_dates[val_boundary_idx]
train_label_cutoff = unique_dates[train_boundary_idx - LABEL_HORIZON]
val_label_cutoff = unique_dates[val_boundary_idx - LABEL_HORIZON]

train_mask = timestamps < train_label_cutoff
val_mask = (timestamps >= train_end_date) & (timestamps < val_label_cutoff)
test_mask = timestamps >= val_end_date

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]
test_dates, test_symbols = timestamps[test_mask], symbols[test_mask]

print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
print(
    f"Purged {LABEL_HORIZON} target dates before each boundary: "
    f"validation starts {train_end_date}, test starts {val_end_date}"
)


# %% [markdown]
# ### Cross-sectional IC helper
#
# Mean cross-sectional Spearman IC by date - same metric used in
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
# 1. **Time-mixing**: Transposes to `(batch, features, time)` and applies the
#    same temporal projection to every feature channel
# 2. **Feature-mixing**: Applies an MLP along the feature axis -- each timestep
#    learns cross-variate interactions
#
# Both use pre-normalization and residual connections. This is conceptually
# similar to the MLP-Mixer vision architecture, adapted for time series. The
# block below follows the authors' basic TSMixer implementation: one temporal
# projection followed by a two-layer feature MLP.
#
# The mixing operations can be written as:
#
# $$\mathbf{X}' = \mathbf{X} + \sigma\bigl(W_t \cdot \operatorname{Norm}(\mathbf{X})^\top\bigr)^\top$$
#
# for time-mixing, and similarly without the transpose for feature-mixing.


# %%
class TimeMixingMLP(nn.Module):
    """Mix information across the time dimension.

    Transposes input to (batch, features, time), applies one shared temporal
    projection, then transposes back.
    """

    def __init__(self, seq_len: int, n_features: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm((seq_len, n_features))
        self.temporal = nn.Linear(seq_len, seq_len)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.relu(self.temporal(x))
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        return self.dropout(x) + residual


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

    def __init__(self, seq_len: int, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm((seq_len, n_features))
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x + residual


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
        self.time_mix = TimeMixingMLP(seq_len, n_features, dropout)
        self.feature_mix = FeatureMixingMLP(seq_len, n_features, hidden_dim, dropout)

    def forward(self, x):
        x = self.time_mix(x)
        x = self.feature_mix(x)
        return x


# %% [markdown]
# ### TSMixer Regressor
#
# Stacks multiple mixer blocks and applies the paper's temporal forecast
# projection with output length one. A small feature adapter then maps those
# per-feature forecasts to the single cross-sectional return label.


# %%
class TSMixerRegressor(nn.Module):
    """TSMixer blocks with a one-step temporal head and scalar feature adapter."""

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
        self.temporal_head = nn.Linear(seq_len, 1)
        self.feature_head = nn.Linear(n_features, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.blocks(x)
        x = self.temporal_head(x.permute(0, 2, 1)).squeeze(-1)
        return self.feature_head(x).squeeze(-1)


# %%
set_global_seeds(SEED)
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

# %%
epochs_axis = list(range(1, len(history["train_loss"]) + 1))
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=epochs_axis,
        y=history["train_loss"],
        mode="lines+markers",
        name="Train",
        line_color=COLORS["blue"],
    )
)
fig.add_trace(
    go.Scatter(
        x=epochs_axis,
        y=history["val_loss"],
        mode="lines+markers",
        name="Validation",
        line_color=COLORS["amber"],
    )
)
fig.update_layout(
    title=f"TSMixer stops after {len(epochs_axis)} epochs with no sustained validation gain",
    xaxis_title="Epoch",
    yaxis_title="Mean squared error",
    width=820,
    height=470,
)
fig.show()

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
zero_mse = float(np.mean(y_test**2))

print("\nRidge Baseline Results:")
print(f"  MSE: {ridge_mse:.6f}")
print(f"  Spearman IC: {ridge_ic:.4f}")

# %% [markdown]
# ## Summary

# %%
model_names = ["TSMixer", "Ridge"]
ic_values = [test_ic, ridge_ic]
mse_ratios = [test_mse / zero_mse, ridge_mse / zero_mse]
bar_palette = {"TSMixer": COLORS["blue"], "Ridge": COLORS["slate"]}

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Mean cross-sectional Spearman IC", "MSE relative to zero-return forecast"),
)
for model_name, ic_value, mse_ratio in zip(model_names, ic_values, mse_ratios, strict=True):
    fig.add_trace(
        go.Bar(
            x=[model_name],
            y=[ic_value],
            name=model_name,
            marker_color=bar_palette[model_name],
            text=[f"{ic_value:.3f}"],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[model_name],
            y=[mse_ratio],
            name=model_name,
            marker_color=bar_palette[model_name],
            text=[f"{mse_ratio:.2f}x"],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

ic_leader = model_names[int(np.argmax(ic_values))]
mse_winners = sum(ratio < 1 for ratio in mse_ratios)
fig.add_hline(y=0, line_color=COLORS["neutral"], row=1, col=1)
fig.add_hline(y=1, line_dash="dot", line_color=COLORS["neutral"], row=1, col=2)
fig.update_layout(
    title=f"{ic_leader} leads on rank IC; {mse_winners} of 2 models beat zero-return MSE",
    width=950,
    height=480,
)
fig.update_yaxes(title_text="Spearman IC", row=1, col=1)
fig.update_yaxes(title_text="MSE / zero-return MSE", row=1, col=2)
fig.show()

# %% [markdown]
# The left panel measures cross-sectional ranking, while the right panel asks
# whether either fitted model improves squared error over predicting zero. These
# are distinct questions. This purged single split demonstrates the architecture;
# it does not establish a stable model ranking. Section 13.9 supplies the
# walk-forward comparison across datasets.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Pure MLP architecture**: TSMixer uses no attention or convolutions; on
#    this purged single split, it leads Ridge on cross-sectional IC, although
#    neither model beats the zero-return MSE reference.
# 2. **Time-mixing via transpose**: By permuting to `(batch, features, time)`,
#    one shared projection learns fixed temporal weights for every feature channel
# 3. **Feature-mixing for cross-variate learning**: The alternating design lets
#    the model learn both temporal dynamics and feature interactions
# 4. **Architecturally transparent**: Fewer parameters than Transformers and a
#    clear separation of which block mixes along which axis. "Interpretable"
#    is too strong for the resulting attention-free representation - the MLP
#    weights themselves are opaque even though the block geometry is not
# 5. **Task adapter**: The paper's temporal forecast head supplies one value per
#    feature; a small feature head maps those values to this notebook's scalar label
#
# Deterministic PyTorch algorithms and a fixed cuBLAS workspace make repeated
# executions reproducible on the same software and GPU stack; another environment
# may still produce small floating-point differences.
#
# **Next**: See `07_mamba_ssm` for state space models that offer an alternative
# to both attention and MLP mixing.
#
# **Book**: Section 13.6 discusses TSMixer alongside other non-attention architectures.
