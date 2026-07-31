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
# # Interpretable Forecasting with N-BEATS
#
# **Docker image**: `ml4t-gpu`
#
# This notebook implements the N-BEATS architecture from scratch in PyTorch,
# focusing on its interpretable configuration that decomposes forecasts into
# trend and seasonality components.
#
# **Learning Objectives**:
# - Implement N-BEATS block structure with doubly-residual connections
# - Understand the interpretable basis functions (polynomial trend, Fourier seasonality)
# - Visualize decomposed forecast components
# - Compare interpretable vs. generic configurations
#
# **Book Reference**: Chapter 13, Section 13.2 (Decomposition: N-BEATS).
# Based on Oreshkin et al. (2020), *N-BEATS: Neural Basis Expansion Analysis
# for Interpretable Time Series Forecasting*.
#
# **Prerequisites**: ETF price data (via `load_etfs()` canonical loader)

# %%
"""Interpretable Forecasting with N-BEATS - trend and seasonality decomposition."""

import os
import warnings
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import polars as pl

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from plotly.subplots import make_subplots

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
SEED = 42
LOOKBACK = 60
HORIZON = 10
HIDDEN_SIZE = 256
N_BLOCKS = 3
N_LAYERS = 4
EPOCHS = 50
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
# **Reproducibility.** The fixed seed controls stochastic initialization and
# mini-batch order. Strict PyTorch kernels and a fixed cuBLAS workspace make
# this training path reproducible on the pinned environment; other PyTorch,
# CUDA, or hardware versions can still shift the final decimals.

# %% [markdown]
# ## Data Preparation
#
# N-BEATS operates on univariate time series. We use SPY close prices
# to demonstrate trend and seasonality decomposition.

# %%
etf_df = load_etfs()

start_dt = datetime.fromisoformat(START_DATE)

spy_data = (
    etf_df.filter((pl.col("symbol") == "SPY") & (pl.col("timestamp") >= start_dt))
    .sort("timestamp")
    .select(["timestamp", "close"])
)

prices = spy_data["close"].to_numpy().astype(np.float32)
timestamps = spy_data["timestamp"].to_numpy()

# Fixed forecast-origin boundaries keep every lookback comparison on the same
# calendar windows. Sequences whose targets straddle a boundary are purged.
n_sequences = len(prices) - LOOKBACK - HORIZON + 1
train_seq_end = int(n_sequences * 0.70)
val_seq_end = int(n_sequences * 0.85)
train_target_cutoff = LOOKBACK + train_seq_end
val_target_cutoff = LOOKBACK + val_seq_end

# The training target ends before train_target_cutoff. Its exclusive boundary
# is therefore also the latest safe observation for fitted normalization.
price_mean = prices[:train_target_cutoff].mean()
price_std = prices[:train_target_cutoff].std()
prices_norm = (prices - price_mean) / price_std

print(f"SPY data: {len(prices)} observations")
print(f"Date range: {timestamps[0]} to {timestamps[-1]}")
print(f"Normalization fit on prices[:{train_target_cutoff}] (training window only)")


# %% [markdown]
# ## Create Sequences
#
# N-BEATS takes a lookback window as input and produces a forecast horizon.


# %%
def create_univariate_sequences(data, lookback, horizon):
    """Create (input, target) pairs for univariate forecasting."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


X, y = create_univariate_sequences(prices_norm, LOOKBACK, HORIZON)
print(f"Sequences: X={X.shape}, y={y.shape}")

target_start = np.arange(LOOKBACK, len(prices) - HORIZON + 1)
target_end = target_start + HORIZON - 1

train_mask = target_end < train_target_cutoff
val_mask = (target_start >= train_target_cutoff) & (target_end < val_target_cutoff)
test_mask = target_start >= val_target_cutoff

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
train_val_overlap = max(0, int(target_end[train_mask].max() - target_start[val_mask].min() + 1))
val_test_overlap = max(0, int(target_end[val_mask].max() - target_start[test_mask].min() + 1))
print(
    "Target overlap at train/validation and validation/test boundaries: "
    f"{train_val_overlap}/{val_test_overlap} observations"
)


# %% [markdown]
# ## N-BEATS Block
#
# Each block takes a lookback window as input and produces two outputs:
# - **Backcast**: reconstruction of the input (for residual connections)
# - **Forecast**: prediction of the future horizon
#
# The interpretable version uses constrained basis functions:
# - **Trend stack**: polynomial basis (degree 2-3)
# - **Seasonality stack**: Fourier basis
#
# ### Basis Expansion Formula
#
# The forecast is generated via basis expansion (Section 13.2):
#
# $$\hat{y} = \sum_{i=1}^{|\theta_f|} \theta_{f,i} \cdot g_{f,i}$$
#
# In the code below:
# - `theta_f` (from `self.theta_f(h)`) = learned expansion coefficients
# - `T_fore` / `S_fore` = pre-computed basis matrices ($g_f$ vectors)
# - `torch.einsum("bp,tp->bt", theta_f, T_fore)` = the weighted sum above
#
# For trend, $g_f = [1, t, t^2, t^3]$ (polynomial); for seasonality,
# $g_f = [\sin(2\pi ft), \cos(2\pi ft)]$ (Fourier harmonics).


# %%
class NBEATSBlock(nn.Module):
    """Single N-BEATS block with shared FC stack and separate basis projections."""

    def __init__(self, lookback, horizon, hidden_size, n_layers, basis_type="generic"):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.basis_type = basis_type

        # Shared fully-connected stack
        layers = [nn.Linear(lookback, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.fc_stack = nn.Sequential(*layers)

        if basis_type == "trend":
            # Polynomial basis: coefficients → polynomial evaluation
            self.poly_degree = 3
            self.theta_b = nn.Linear(hidden_size, self.poly_degree + 1)
            self.theta_f = nn.Linear(hidden_size, self.poly_degree + 1)
            # Pre-compute time vectors
            t_back = torch.linspace(0, 1, lookback).unsqueeze(0)
            t_fore = torch.linspace(0, 1, horizon).unsqueeze(0)
            self.register_buffer(
                "T_back",
                torch.stack([t_back**i for i in range(self.poly_degree + 1)], dim=-1).squeeze(0),
            )
            self.register_buffer(
                "T_fore",
                torch.stack([t_fore**i for i in range(self.poly_degree + 1)], dim=-1).squeeze(0),
            )

        elif basis_type == "seasonality":
            # Fourier basis: coefficients → harmonic evaluation
            self.n_harmonics = 5
            n_coeffs = 2 * self.n_harmonics
            self.theta_b = nn.Linear(hidden_size, n_coeffs)
            self.theta_f = nn.Linear(hidden_size, n_coeffs)
            # Pre-compute Fourier basis
            t_back = torch.linspace(0, 1, lookback).unsqueeze(0)
            t_fore = torch.linspace(0, 1, horizon).unsqueeze(0)
            freqs = torch.arange(1, self.n_harmonics + 1).float()
            self.register_buffer("S_back", self._fourier_basis(t_back, freqs))
            self.register_buffer("S_fore", self._fourier_basis(t_fore, freqs))

        else:  # generic
            self.theta_b = nn.Linear(hidden_size, lookback)
            self.theta_f = nn.Linear(hidden_size, horizon)

    @staticmethod
    def _fourier_basis(t, freqs):
        """Create Fourier basis matrix [sin(2pi*f*t), cos(2pi*f*t)]."""
        # t: (1, T), freqs: (H,) → output: (T, 2H)
        t = t.squeeze(0).unsqueeze(-1)  # (T, 1)
        angles = 2 * np.pi * t * freqs.unsqueeze(0)  # (T, H)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (T, 2H)

    def forward(self, x):
        h = self.fc_stack(x)

        if self.basis_type == "trend":
            theta_b = self.theta_b(h)  # (batch, poly_degree+1)
            theta_f = self.theta_f(h)
            backcast = torch.einsum("bp,tp->bt", theta_b, self.T_back)
            forecast = torch.einsum("bp,tp->bt", theta_f, self.T_fore)

        elif self.basis_type == "seasonality":
            theta_b = self.theta_b(h)  # (batch, 2*n_harmonics)
            theta_f = self.theta_f(h)
            backcast = torch.einsum("bh,th->bt", theta_b, self.S_back)
            forecast = torch.einsum("bh,th->bt", theta_f, self.S_fore)

        else:  # generic
            backcast = self.theta_b(h)
            forecast = self.theta_f(h)

        return backcast, forecast


# %% [markdown]
# ## N-BEATS Model
#
# The full model stacks multiple blocks with **doubly-residual** connections:
# each block processes the residual from previous blocks (input minus backcast).


# %%
class NBEATS(nn.Module):
    """N-BEATS with configurable stacks (interpretable or generic)."""

    def __init__(self, lookback, horizon, hidden_size, n_blocks, n_layers, interpretable=True):
        super().__init__()
        self.blocks = nn.ModuleList()

        if interpretable:
            # Trend stack + Seasonality stack (N-BEATS-I)
            for _ in range(n_blocks):
                self.blocks.append(NBEATSBlock(lookback, horizon, hidden_size, n_layers, "trend"))
            for _ in range(n_blocks):
                self.blocks.append(
                    NBEATSBlock(lookback, horizon, hidden_size, n_layers, "seasonality")
                )
        else:
            # All generic blocks (N-BEATS-G)
            for _ in range(n_blocks * 2):
                self.blocks.append(NBEATSBlock(lookback, horizon, hidden_size, n_layers, "generic"))

    def forward(self, x):
        residual = x
        forecast = torch.zeros(x.shape[0], self.blocks[0].horizon, device=x.device)

        block_forecasts = []
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast  # Doubly-residual: update input
            forecast = forecast + block_forecast  # Accumulate forecasts
            block_forecasts.append(block_forecast)

        return forecast, block_forecasts


# %% [markdown]
# ## Training


# %%
def train_nbeats(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr=1e-3):
    """Train N-BEATS with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.FloatTensor(y_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)
    y_v = torch.FloatTensor(y_val).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    patience = 7
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_tr))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            forecast, _ = model(X_tr[batch_idx])
            loss = criterion(forecast, y_tr[batch_idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_forecast, _ = model(X_v)
            val_loss = criterion(val_forecast, y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1}/{epochs}: train={epoch_loss / n_batches:.6f}, val={val_loss:.6f}"
            )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# %% [markdown]
# ## Train N-BEATS Interpretable

# %%
print("Training N-BEATS-I (interpretable)...")
nbeats_i = NBEATS(LOOKBACK, HORIZON, HIDDEN_SIZE, N_BLOCKS, N_LAYERS, interpretable=True).to(DEVICE)
n_params = sum(p.numel() for p in nbeats_i.parameters())
print(f"  Parameters: {n_params:,}")

nbeats_i = train_nbeats(nbeats_i, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)

# %% [markdown]
# ## Train N-BEATS Generic

# %%
print("\nTraining N-BEATS-G (generic)...")
nbeats_g = NBEATS(LOOKBACK, HORIZON, HIDDEN_SIZE, N_BLOCKS, N_LAYERS, interpretable=False).to(
    DEVICE
)
n_params_g = sum(p.numel() for p in nbeats_g.parameters())
print(f"  Parameters: {n_params_g:,}")

nbeats_g = train_nbeats(nbeats_g, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)

# %% [markdown]
# ## Evaluation
#
# A persistence forecast repeats the last observed price through the horizon.
# It is the minimum credible benchmark for a price-level model: a flexible
# architecture has not earned its complexity unless it improves on that rule.
# We report RMSE in training-window standard deviations and the MAE ratio to
# persistence. A ratio above 1 means the model is worse than persistence.

# %%
X_test_t = torch.FloatTensor(X_test).to(DEVICE)

nbeats_i.eval()
nbeats_g.eval()

with torch.no_grad():
    pred_i, block_forecasts_i = nbeats_i(X_test_t)
    pred_g, _ = nbeats_g(X_test_t)

pred_i = pred_i.cpu().numpy()
pred_g = pred_g.cpu().numpy()

# %%
persistence_pred = np.repeat(X_test[:, -1:], HORIZON, axis=1)
persistence_mae = float(np.mean(np.abs(persistence_pred - y_test)))

comparison_df = pl.DataFrame(
    {
        "Model": ["Persistence", "N-BEATS-I", "N-BEATS-G"],
        "RMSE (z)": [
            float(np.sqrt(np.mean((pred - y_test) ** 2)))
            for pred in [persistence_pred, pred_i, pred_g]
        ],
        "MAE / persistence": [
            float(np.mean(np.abs(pred - y_test)) / persistence_mae)
            for pred in [persistence_pred, pred_i, pred_g]
        ],
    }
)

model_ratios = comparison_df.filter(pl.col("Model") != "Persistence")["MAE / persistence"]
benchmark_title = (
    "Both N-BEATS variants trail persistence on the SPY holdout"
    if (model_ratios > 1).all()
    else "At least one N-BEATS variant improves on persistence"
)

fig_benchmark = go.Figure(
    go.Bar(
        x=comparison_df["Model"].to_list(),
        y=comparison_df["MAE / persistence"].to_list(),
        marker_color=[COLORS["neutral"], COLORS["blue"], COLORS["amber"]],
        text=[f"{value:.2f}x" for value in comparison_df["MAE / persistence"]],
        textposition="outside",
    )
)
fig_benchmark.add_hline(y=1.0, line_dash="dash", line_color=COLORS["neutral"])
fig_benchmark.update_layout(
    title=benchmark_title,
    xaxis_title="Forecast",
    yaxis_title="MAE relative to persistence (x)",
    showlegend=False,
)
fig_benchmark.update_yaxes(rangemode="tozero")
fig_benchmark.show()

# %% [markdown]
# **Interpretation**: the persistence line at 1.0 is the relevant floor. In
# this single SPY holdout, neither neural model clears it, although the generic
# variant is closer. The former price-level rank correlation was near one even
# for persistence because adjacent SPY levels share a trend; it was therefore
# not evidence of forecast skill and has been removed. This experiment supports
# the decomposition lesson, not a trading-performance claim.

# %% [markdown]
# ## Component Decomposition
#
# The key advantage of N-BEATS-I: we can examine which blocks contribute
# trend vs. seasonality to the forecast.

# %%
# Extract block-level forecasts for a sample
sample_idx = len(X_test) // 2
block_preds = [bf[sample_idx].cpu().numpy() for bf in block_forecasts_i]

# First N_BLOCKS are trend, last N_BLOCKS are seasonality
trend_forecast = sum(block_preds[:N_BLOCKS])
seasonal_forecast = sum(block_preds[N_BLOCKS:])

# %% [markdown]
# **Denormalization**: The trend component gets the mean added back (it captures
# the price level), while seasonality is zero-centered (periodic deviations around
# the trend). The combined forecast = Trend + Seasonality in original price space.

# %%
trend_denorm = trend_forecast * price_std + price_mean
seasonal_denorm = seasonal_forecast * price_std  # Zero-centered
total_denorm = pred_i[sample_idx] * price_std + price_mean
actual_denorm = y_test[sample_idx] * price_std + price_mean

x_axis = list(range(1, HORIZON + 1))

fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=["Trend Component", "Seasonality Component", "Total Forecast vs Actual"],
    shared_xaxes=True,
    vertical_spacing=0.08,
)

fig.add_trace(
    go.Scatter(x=x_axis, y=trend_denorm, name="Trend", line=dict(color=COLORS["blue"])),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x_axis,
        y=seasonal_denorm,
        name="Seasonality",
        line=dict(color=COLORS["amber"]),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x_axis,
        y=actual_denorm,
        name="Actual",
        line=dict(color=COLORS["neutral"], width=2),
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=x_axis,
        y=total_denorm,
        name="N-BEATS-I",
        line=dict(color=COLORS["blue"], dash="dash"),
    ),
    row=3,
    col=1,
)

fig.update_layout(
    title="N-BEATS-I exposes the trend and seasonal terms behind its forecast",
    height=650,
)
fig.update_xaxes(title_text="Forecast Step", row=3, col=1)
fig.update_yaxes(title_text="Trend contribution ($)", row=1, col=1)
fig.update_yaxes(title_text="Seasonal contribution ($)", row=2, col=1)
fig.update_yaxes(title_text="SPY price ($)", row=3, col=1)
fig.show()

# %% [markdown]
# ## Backcast Visualization
#
# The backcast represents what each block has "understood" about the input.
# The residual (input minus backcast) is passed to the next block, forcing
# progressively finer pattern extraction.

# %%
# Compute block-level backcasts for the same test sample
sample_input = torch.FloatTensor(X_test[sample_idx : sample_idx + 1]).to(DEVICE)
nbeats_i.eval()

with torch.no_grad():
    residual = sample_input.clone()
    backcasts = []
    for block in nbeats_i.blocks:
        backcast, _ = block(residual)
        backcasts.append(backcast.cpu().numpy().flatten())
        residual = residual - backcast

# Denormalize the input back to price level; the residual stays in z-score
# units because adding the training-window mean would shift "structure not yet
# captured" onto the price scale and conflate it with the input curve.
input_denorm = X_test[sample_idx] * price_std + price_mean
residual_z = residual.cpu().numpy().flatten()

x_back = list(range(LOOKBACK))

fig_bc = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=[
        "Original input (price level)",
        "Final residual (z-score, training-window stats)",
    ],
    shared_xaxes=True,
    vertical_spacing=0.12,
)
fig_bc.add_trace(
    go.Scatter(x=x_back, y=input_denorm, name="Original input", line=dict(width=2)), row=1, col=1
)
fig_bc.add_trace(
    go.Scatter(x=x_back, y=residual_z, name="Final residual", line=dict(dash="dot")), row=2, col=1
)
fig_bc.update_xaxes(title_text="Lookback Step", row=2, col=1)
fig_bc.update_yaxes(title_text="Price ($)", row=1, col=1)
fig_bc.update_yaxes(title_text="Residual (z)", row=2, col=1)
fig_bc.update_layout(
    title="Six blocks reduce the input but leave visible residual structure",
    height=500,
)
fig_bc.show()

# %% [markdown]
# **Interpretation**: the final residual is smaller than the normalized input,
# but it is neither centered at zero nor visually structureless. A backcast is
# what the fitted blocks chose to explain; it is not a guarantee that the
# remainder is white noise. The remaining pattern cautions against treating the
# displayed trend and seasonality as a complete economic decomposition.

# %% [markdown]
# ## Lookback Sensitivity
#
# N-BEATS is sensitive to the lookback-to-horizon ratio. Too short a lookback
# starves the decomposition of context; very long lookbacks can introduce noise.
# This is a validation-set model-selection diagnostic. All candidates use the
# same target-date boundaries, and the sealed test window is not touched.

# %%
lookback_values = [20, 40, 60, 120]
sensitivity_results = []

for lb in lookback_values:
    pm_s = prices[:train_target_cutoff].mean()
    ps_s = prices[:train_target_cutoff].std()
    prices_norm_s = (prices - pm_s) / ps_s

    X_s, y_s = create_univariate_sequences(prices_norm_s, lb, HORIZON)
    target_start_s = np.arange(lb, len(prices) - HORIZON + 1)
    target_end_s = target_start_s + HORIZON - 1
    train_mask_s = target_end_s < train_target_cutoff
    val_mask_s = (target_start_s >= train_target_cutoff) & (target_end_s < val_target_cutoff)

    set_global_seeds(SEED)
    model_s = NBEATS(lb, HORIZON, HIDDEN_SIZE, N_BLOCKS, N_LAYERS, interpretable=True).to(DEVICE)
    model_s = train_nbeats(
        model_s,
        X_s[train_mask_s],
        y_s[train_mask_s],
        X_s[val_mask_s],
        y_s[val_mask_s],
        EPOCHS,
        BATCH_SIZE,
    )

    model_s.eval()
    with torch.no_grad():
        pred_s, _ = model_s(torch.FloatTensor(X_s[val_mask_s]).to(DEVICE))
    pred_s = pred_s.cpu().numpy()
    y_s_val = y_s[val_mask_s]
    persistence_s = np.repeat(X_s[val_mask_s, -1:], HORIZON, axis=1)

    rmse_s = float(np.sqrt(np.mean((pred_s - y_s_val) ** 2)))
    mae_ratio_s = float(
        np.mean(np.abs(pred_s - y_s_val)) / np.mean(np.abs(persistence_s - y_s_val))
    )
    sensitivity_results.append(
        {
            "Lookback": lb,
            "Ratio": lb / HORIZON,
            "Validation RMSE (z)": rmse_s,
            "Validation MAE / persistence": mae_ratio_s,
        }
    )
    print(
        f"  Lookback={lb} (ratio={lb / HORIZON:.1f}): "
        f"validation RMSE={rmse_s:.4f}, MAE/persistence={mae_ratio_s:.2f}x"
    )

sensitivity_df = pl.DataFrame(sensitivity_results)
best_lookback = sensitivity_df.sort("Validation MAE / persistence").row(0, named=True)
all_trail_persistence = (sensitivity_df["Validation MAE / persistence"] > 1).all()

fig_sensitivity = go.Figure(
    go.Scatter(
        x=sensitivity_df["Ratio"].to_list(),
        y=sensitivity_df["Validation MAE / persistence"].to_list(),
        mode="lines+markers+text",
        line=dict(color=COLORS["blue"], width=3),
        marker=dict(size=9),
        text=[f"{value:.2f}x" for value in sensitivity_df["Validation MAE / persistence"]],
        textposition="top center",
    )
)
fig_sensitivity.add_hline(y=1.0, line_dash="dash", line_color=COLORS["neutral"])
fig_sensitivity.update_layout(
    title=(
        f"A {best_lookback['Lookback']}-day lookback comes closest, but none beats persistence"
        if all_trail_persistence
        else f"Validation favors a {best_lookback['Lookback']}-day lookback"
    ),
    xaxis_title="Lookback / forecast-horizon ratio (x)",
    yaxis_title="Validation MAE relative to persistence (x)",
    showlegend=False,
)
fig_sensitivity.show()

# %% [markdown]
# **Finding**: lookback length changes the validation error, but this sweep is
# selection evidence rather than a final performance estimate. Persistence is
# the reference line, and the chart makes clear whether added temporal context
# helps enough to clear that elementary forecast. The fixed calendar boundaries
# ensure the candidates face the same validation dates.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Doubly-residual architecture**: Each block refines the input residual AND
#    accumulates its forecast contribution
# 2. **Interpretable decomposition**: Polynomial trend + Fourier seasonality stacks
#    produce human-readable components
# 3. **Generic mode**: Learns arbitrary basis functions, potentially more accurate
#    but loses interpretability
# 4. **No feature engineering**: N-BEATS works directly on the raw price series;
#    that makes persistence the essential benchmark for forecast accuracy
# 5. **Lookback-to-horizon ratio**: Performance is sensitive to this ratio;
#    validation comparison must keep target dates fixed and the test set sealed
#
# **Caveat**: Results on a single equity index (SPY) are not generalizable -
# benchmark on your specific dataset before drawing architecture conclusions.
#
# **Next**: See `03_great_debate` for the Linear vs. Transformer controversy.
