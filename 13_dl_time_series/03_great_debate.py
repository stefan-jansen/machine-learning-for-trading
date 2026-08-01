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
# # The Great Debate: Are Transformers Effective for Time Series?
#
# **Docker image**: `ml4t-gpu`
#
# This notebook tests the core finding from Zeng et al. (2023) that simple
# linear models can outperform Transformers on long-term time series forecasting.
# We implement all three LTSF-Linear variants (Linear, D-Linear, N-Linear), a
# vanilla Transformer, and two naive forecasts on daily SPY returns.
#
# **Learning Objectives**:
# - Implement the Linear, D-Linear, and N-Linear baselines from the LTSF-Linear paper
# - Build a simple Transformer encoder for time series
# - Test whether linear models compete with a Transformer and naive forecasts
# - Understand why this result sparked the "great debate" in the field
#
# **Book Reference**: Chapter 13, Section 13.4 (The Great Debate)
#
# **Prerequisites**: ETF price data via the canonical `load_etfs()` loader

# %%
"""Test whether simple linear models can outperform Transformers on time series."""

import os
import time
import warnings
from datetime import datetime

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch
import torch.nn as nn

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
SEED = 42
LOOKBACK = 96
HORIZON = 24
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
EPOCHS = 30
BATCH_SIZE = 32
START_DATE = "2015-01-01"


# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

set_global_seeds(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# %% [markdown]
# ## Data Preparation
#
# We align several ETF series to a common calendar but run the core
# Linear-versus-Transformer comparison on SPY. This is a deliberate
# simplification: the Zeng et al. critique is about a single-series LTSF
# experiment, and a multi-symbol panel adds confounders (cross-sectional
# variance, multi-asset attention) we do not want to entangle with the
# main question.

# %%
etf_df = load_etfs()

start_dt = datetime.fromisoformat(START_DATE)
SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

close_wide = (
    etf_df.filter(pl.col("symbol").is_in(SYMBOLS) & (pl.col("timestamp") >= start_dt))
    .pivot(on="symbol", values="close", index="timestamp")
    .drop_nulls()
    .sort("timestamp")
)

feature_cols = [c for c in close_wide.columns if c != "timestamp"]
returns = close_wide.with_columns(
    [pl.col(c).pct_change().alias(c) for c in feature_cols]
).drop_nulls()

# SPY is the primary series for the LTSF-style univariate comparison
series = returns["SPY"].to_numpy().astype(np.float32)


# %% [markdown]
# ### Sequence builder
#
# Standard sliding-window construction: each input is `lookback` daily returns
# and each target is the next `horizon` returns. We assign complete forecast
# horizons to one partition and purge the 23 overlapping targets at each boundary.


# %%
def create_sequences(data, lookback, horizon):
    """Sliding lookback → forecast horizon pairs for univariate forecasting."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# %%
X, y = create_sequences(series, LOOKBACK, HORIZON)

n = len(X)
train_target_cutoff = LOOKBACK + int(n * 0.70)
val_target_cutoff = LOOKBACK + int(n * 0.85)
target_start = np.arange(LOOKBACK, len(series) - HORIZON + 1)
target_end = target_start + HORIZON - 1

train_mask = target_end < train_target_cutoff
val_mask = (target_start >= train_target_cutoff) & (target_end < val_target_cutoff)
test_mask = target_start >= val_target_cutoff

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Sequences: {X.shape}, Target: {y.shape}")
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# %% [markdown]
# **Note**: The 70/15/15 cutoffs refer to forecast target dates. No target date
# appears in two partitions. Walk-forward validation (Section 13.7) remains
# preferable when the goal is a production estimate rather than this comparison.

# %% [markdown]
# ## Model Definitions
#
# ### Linear (Plain)
# The simplest possible baseline: a single matrix multiplication from
# lookback window to forecast horizon. No decomposition, no normalization.


# %%
class Linear(nn.Module):
    """Plain linear mapping from lookback to horizon - the simplest baseline."""

    def __init__(self, lookback, horizon):
        super().__init__()
        self.linear = nn.Linear(lookback, horizon)

    def forward(self, x):
        return self.linear(x)


# %% [markdown]
# ### D-Linear
# Decomposes input into trend (moving average) and remainder,
# applies separate linear layers to each.


# %%
class DLinear(nn.Module):
    """Decomposition-Linear: separate linear for trend and remainder."""

    def __init__(self, lookback, horizon, kernel_size=25):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.kernel_size = kernel_size
        # No internal padding - we explicitly edge-pad inputs in `forward`
        # so the moving average mirrors the boundary values instead of
        # injecting zeros, which would bias the trend toward zero near the
        # window edges.
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.linear_trend = nn.Linear(lookback, horizon)
        self.linear_remainder = nn.Linear(lookback, horizon)

    def forward(self, x):
        # x: (batch, lookback)
        pad = self.kernel_size // 2
        # Replicate boundary values so the MA at the edges uses real data
        # instead of zero-padded ghost samples.
        x_padded = nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
        trend = self.avg_pool(x_padded).squeeze(1)[:, : self.lookback]
        remainder = x - trend
        return self.linear_trend(trend) + self.linear_remainder(remainder)


# %% [markdown]
# ### N-Linear
# Normalizes input by subtracting the last value, applies a linear layer,
# then adds back the normalization.


# %%
class NLinear(nn.Module):
    """Normalization-Linear: normalize by last value before linear."""

    def __init__(self, lookback, horizon):
        super().__init__()
        self.linear = nn.Linear(lookback, horizon)

    def forward(self, x):
        # x: (batch, lookback)
        last_val = x[:, -1:]  # (batch, 1)
        x_norm = x - last_val
        forecast = self.linear(x_norm)
        return forecast + last_val


# %% [markdown]
# ### Vanilla Transformer
# Standard Transformer encoder with positional encoding for time series.


# %%
class SimpleTransformer(nn.Module):
    """Vanilla Transformer encoder for time series forecasting."""

    def __init__(self, lookback, horizon, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(lookback * d_model, horizon)

    def forward(self, x):
        # x: (batch, lookback)
        x = x.unsqueeze(-1)  # (batch, lookback, 1)
        x = self.input_proj(x)  # (batch, lookback, d_model)
        x = x + self.pos_encoding
        x = self.encoder(x)
        x = x.flatten(1)  # (batch, lookback * d_model)
        return self.head(x)


# %% [markdown]
# ## Training


# %%
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr=1e-3):
    """Train with MSE loss + early stopping; returns (best-state model, best val MSE)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.FloatTensor(y_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)
    y_v = torch.FloatTensor(y_val).to(DEVICE)
    best_val = float("inf")
    best_state = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_tr))
        for i in range(0, len(indices), batch_size):
            idx = indices[i : i + batch_size]
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 5:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_val


# %% [markdown]
# ## Run Experiments

# %%
model_factories = {
    "Linear": lambda: Linear(LOOKBACK, HORIZON),
    "D-Linear": lambda: DLinear(LOOKBACK, HORIZON),
    "N-Linear": lambda: NLinear(LOOKBACK, HORIZON),
    "Transformer": lambda: SimpleTransformer(LOOKBACK, HORIZON, D_MODEL, N_HEADS, N_LAYERS),
}
models = {}

zero_pred = np.zeros_like(y_test)
repeat_pred = np.repeat(X_test[:, -1:], HORIZON, axis=1)
results = {
    "Zero": {
        "mse": float(np.mean((zero_pred - y_test) ** 2)),
        "mae": float(np.mean(np.abs(zero_pred - y_test))),
        "time": 0.0,
        "params": 0,
    },
    "Closest Repeat": {
        "mse": float(np.mean((repeat_pred - y_test) ** 2)),
        "mae": float(np.mean(np.abs(repeat_pred - y_test))),
        "time": 0.0,
        "params": 0,
    },
}

for name, factory in model_factories.items():
    set_global_seeds(SEED)
    model = factory().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining {name} ({n_params:,} params)...")

    start_time = time.time()
    model, best_val = train_model(model, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)
    train_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()

    mse = np.mean((pred - y_test) ** 2)
    mae = np.mean(np.abs(pred - y_test))

    models[name] = model
    results[name] = {"mse": mse, "mae": mae, "time": train_time, "params": n_params}
    print(f"  MSE={mse:.6f}, MAE={mae:.6f}, Time={train_time:.1f}s")

# %% [markdown]
# ## Results Comparison

# %%
zero_mse = results["Zero"]["mse"]
for result in results.values():
    result["mse_ratio"] = result["mse"] / zero_mse

best_learned_name = min(model_factories, key=lambda name: results[name]["mse"])
transformer_ratio = results["Transformer"]["mse"] / results[best_learned_name]["mse"]
parameter_ratio = results["Transformer"]["params"] / results["Linear"]["params"]

print(
    f"Best learned model: {best_learned_name}; Transformer error is "
    f"{transformer_ratio:.2f}x as large. Zero forecast remains best; "
    f"the Transformer has {parameter_ratio:.1f}x as many parameters as Linear."
)

# %% [markdown]
# **Interpretation**: The linear family beats the Transformer on this SPY task,
# the narrow comparison at the center of Zeng et al. But every learned model
# loses to a zero-return forecast. Daily returns offer little predictable signal,
# so the stronger conclusion is not that a linear network forecasts well. It is
# that additional capacity fails to earn its keep here. The Closest Repeat
# baseline from the original paper is also weak because yesterday's return is a
# poor forecast for every point in the next 24-day horizon.

# %%
ordered_models = ["Zero", "Closest Repeat", "Linear", "D-Linear", "N-Linear", "Transformer"]
bar_colors = [
    COLORS["positive"],
    COLORS["neutral"],
    COLORS["blue"],
    COLORS["slate"],
    COLORS["amber"],
    COLORS["copper"],
]
fig = go.Figure(
    go.Bar(
        x=ordered_models,
        y=[results[name]["mse_ratio"] for name in ordered_models],
        marker_color=bar_colors,
        text=[f"{results[name]['mse_ratio']:.2f}x" for name in ordered_models],
        textposition="outside",
        hovertemplate="%{x}<br>MSE relative to zero: %{y:.2f}x<extra></extra>",
    )
)

fig.update_layout(
    title="Linear models beat the Transformer, but not a zero-return forecast",
    xaxis_title="Forecast",
    yaxis_title="Test MSE relative to zero forecast",
    showlegend=False,
    yaxis_range=[0, max(results[name]["mse_ratio"] for name in ordered_models) * 1.15],
)
fig.add_hline(y=1, line_dash="dot", line_color=COLORS["neutral"])
fig.show()

# %% [markdown]
# ## Shuffle Experiment
#
# Zeng et al. shuffled each input sequence to destroy temporal order. On the
# Exchange and ETTh1 benchmarks, this hurt the linear models much more than the
# Transformers. We repeat the diagnostic on SPY returns without assuming that a
# result from those multivariate benchmarks must transfer.

# %%
# Evaluate all models on original and shuffled test inputs
X_test_t = torch.FloatTensor(X_test).to(DEVICE)

# Shuffle each sample's time steps independently
rng = np.random.default_rng(SEED)
X_shuffled = X_test.copy()
for i in range(len(X_shuffled)):
    rng.shuffle(X_shuffled[i])
X_shuffled_t = torch.FloatTensor(X_shuffled).to(DEVICE)

shuffle_results = []
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        pred_orig = model(X_test_t).cpu().numpy()
        pred_shuf = model(X_shuffled_t).cpu().numpy()

    mse_orig = float(np.mean((pred_orig - y_test) ** 2))
    mse_shuf = float(np.mean((pred_shuf - y_test) ** 2))
    shuffle_results.append(
        {
            "Model": name,
            "MSE (original)": round(mse_orig, 6),
            "MSE (shuffled)": round(mse_shuf, 6),
            "Delta (%)": round(100 * (mse_shuf - mse_orig) / mse_orig, 1),
        }
    )

shuffle_df = pl.DataFrame(shuffle_results)

fig = go.Figure(
    go.Bar(
        x=shuffle_df["Model"],
        y=shuffle_df["Delta (%)"],
        marker_color=[COLORS["blue"], COLORS["slate"], COLORS["amber"], COLORS["copper"]],
        text=[f"{value:+.1f}%" for value in shuffle_df["Delta (%)"]],
        textposition="outside",
        hovertemplate="%{x}<br>MSE change: %{y:+.1f}%<extra></extra>",
    )
)
fig.add_hline(y=0, line_color=COLORS["neutral"])
fig.update_layout(
    title="Shuffling barely changes error on low-signal SPY returns",
    xaxis_title="Model",
    yaxis_title="Change in test MSE after shuffling (%)",
    showlegend=False,
)
fig.show()

# %% [markdown]
# **Finding**: On this SPY split, shuffling has minimal impact on any model.
# The Transformer's MSE is essentially unchanged and the linear models also
# show negligible change. This is a diagnostic on one univariate financial
# series - not a proof that daily equity returns are i.i.d., and not a claim
# that self-attention generally ignores temporal order; the original Zeng et
# al. shuffle gap was demonstrated on weather and electricity data with
# strong periodic patterns, structure that our SPY return series mostly
# lacks. Read the result as: when the input carries little sequential signal,
# preserving sequential order does not help.

# %% [markdown]
# ## Validation-Only Lookback Sensitivity
#
# Zeng et al. found that longer inputs often helped linear models while leaving
# Transformers stable or worse. We test three lookbacks without consulting the
# test set. Every candidate uses the same forecast-date cutoffs, so changing the
# lookback changes available context rather than silently shifting the calendar.

# %%
lookback_values = [48, 96, 192]
lookback_results = []

for lb in lookback_values:
    X_lb, y_lb = create_sequences(series, lb, HORIZON)
    target_start_lb = np.arange(lb, len(series) - HORIZON + 1)
    target_end_lb = target_start_lb + HORIZON - 1
    train_mask_lb = target_end_lb < train_target_cutoff
    val_mask_lb = (target_start_lb >= train_target_cutoff) & (target_end_lb < val_target_cutoff)

    X_train_lb, y_train_lb = X_lb[train_mask_lb], y_lb[train_mask_lb]
    X_val_lb, y_val_lb = X_lb[val_mask_lb], y_lb[val_mask_lb]
    zero_val_mse = float(np.mean(y_val_lb**2))

    lb_factories = {
        "Linear": lambda lb=lb: Linear(lb, HORIZON),
        "D-Linear": lambda lb=lb: DLinear(lb, HORIZON),
        "Transformer": lambda lb=lb: SimpleTransformer(lb, HORIZON, D_MODEL, N_HEADS, N_LAYERS),
    }

    for name, factory in lb_factories.items():
        set_global_seeds(SEED)
        model = factory().to(DEVICE)
        model, _ = train_model(
            model,
            X_train_lb,
            y_train_lb,
            X_val_lb,
            y_val_lb,
            EPOCHS,
            BATCH_SIZE,
        )
        model.eval()
        with torch.no_grad():
            pred_lb = model(torch.FloatTensor(X_val_lb).to(DEVICE)).cpu().numpy()
        mse_lb = float(np.mean((pred_lb - y_val_lb) ** 2))
        lookback_results.append(
            {
                "Lookback": lb,
                "Model": name,
                "MSE ratio": mse_lb / zero_val_mse,
            }
        )
    print(f"  Lookback={lb}: done")

lookback_df = pl.DataFrame(lookback_results)

fig = go.Figure()
for name, color, dash in [
    ("Linear", COLORS["blue"], "solid"),
    ("D-Linear", COLORS["amber"], "dash"),
    ("Transformer", COLORS["copper"], "solid"),
]:
    subset = lookback_df.filter(pl.col("Model") == name)
    show_text = name == "Transformer"
    fig.add_trace(
        go.Scatter(
            x=subset["Lookback"],
            y=subset["MSE ratio"],
            mode="lines+markers+text" if show_text else "lines+markers",
            name=name,
            line=dict(color=color, dash=dash),
            text=[f"{value:.2f}x" for value in subset["MSE ratio"]] if show_text else None,
            textposition="top center",
            hovertemplate=f"{name}<br>Lookback: %{{x}}<br>Relative MSE: %{{y:.2f}}x<extra></extra>",
        )
    )
fig.add_hline(y=1, line_dash="dot", line_color=COLORS["neutral"])
fig.update_layout(
    title="Longer context does not overcome the zero-return baseline",
    xaxis_title="Lookback (trading days)",
    yaxis_title="Validation MSE relative to zero forecast",
    xaxis=dict(tickmode="array", tickvals=lookback_values),
)
fig.show()

# %% [markdown]
# **Finding**: Use this chart as a model-selection diagnostic, not another test
# result. It compares candidates only on validation targets. None beats the
# horizontal zero-forecast reference, and the short three-point curves do not
# support a general claim about how either architecture scales with context.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Linear baselines beat the Transformer on this univariate task**, the narrow
#    comparison highlighted by Zeng et al. (2023), with about 44x fewer parameters
# 2. **No learned model beats zero**: low MSE relative to a complex model is not
#    evidence of useful return predictability; naive forecasts belong in every comparison
# 3. **Three variants**: Linear (plain), D-Linear (trend/remainder decomposition),
#    N-Linear (last-value normalization) - all with minimal parameters
# 4. **Shuffle test does not transfer**: neither architecture reacts much to shuffled
#    SPY returns, unlike the paper's Exchange and ETTh1 benchmark results
# 5. **Lookback sensitivity is validation-only**: fixed target dates prevent leakage
#    and make the comparison independent of the final test set
# 6. The debate motivated better Transformer designs (PatchTST, iTransformer)
#    that directly address these failure modes
#
# **Caveat**: These results are on univariate SPY data. Transformers may perform
# better in multivariate settings with rich covariates where cross-series attention
# extracts meaningful relationships (Section 13.5 explores TFT for this case).
# PyTorch deterministic algorithms and a fixed cuBLAS workspace make repeated
# executions reproducible on the same software and GPU stack; another environment
# may still produce small floating-point differences.
#
# **Next**: See `04_transformers` for modern PatchTST and iTransformer architectures.
