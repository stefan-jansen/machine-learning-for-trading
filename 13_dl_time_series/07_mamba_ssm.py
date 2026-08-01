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
# # Simplified Selective State Space Model (Mamba-like)
#
# **Docker image**: `ml4t-gpu`
#
# This notebook implements a pedagogical version of the Mamba architecture
# (Gu and Dao, 2023) for predicting forward ETF returns. Mamba belongs to
# the family of **Structured State Space Models** (SSMs) that process
# sequences with O(T) complexity -- linear in sequence length -- compared
# to O(T^2) for self-attention.
#
# **Learning Objectives**:
# - Understand the continuous-time state space formulation and its discretization
# - Implement a selective scan mechanism where B, C, and dt are input-dependent
# - Build a multi-layer Mamba-like regressor with gated output projection
# - Compare SSM predictions against a Ridge regression baseline
#
# **Book Reference**: Chapter 13, Section 13.6 (The Full Practitioner Toolkit)
#
# **Note**: This is a *pedagogical* selective SSM written in pure PyTorch
# to expose the inner workings of the selective scan. It captures Mamba's
# defining mechanism - input-dependent $B_t$, $C_t$, and $\Delta_t$ - but
# omits implementation details of the production library, which uses custom
# CUDA kernels for hardware-efficient parallel scans, hardware-aware
# materialization of intermediate states, and additional numerical
# refinements. Treat the implementation here as a faithful sketch of the
# selective-state-space idea rather than a drop-in replacement for the
# reference `mamba_ssm` package.
#
# **Prerequisites**: ETF features (`case_studies/etfs/`)

# %%
"""Simplified Selective State Space Model - pedagogical Mamba implementation for return prediction."""

import os
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils.reproducibility import set_global_seeds
from utils.style import COLORS  # activates the ml4t Plotly template on import

warnings.filterwarnings("ignore")

from dl_sequences import create_sequences_multi_asset, load_dl_dataset, train_model

# %% tags=["parameters"]
SEED = 42
LOOKBACK = 60
D_MODEL = 32
D_STATE = 16
N_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
MAX_TRAIN_SAMPLES = 50_000
MAX_VAL_SAMPLES = 15_000
MAX_TEST_SAMPLES = 15_000
INFER_BATCH_SIZE = 1_024
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

# %% [markdown]
# ### Pedagogical subsampling
#
# The pure-Python selective scan is roughly 100× slower than the production
# CUDA kernel, so we cap each split at a few thousand sequences for tractable
# wall-clock. Critically we subsample by **complete dates**, not row-count: a
# raw `[-MAX_SAMPLES:]` slice would start mid-date and leave a partial
# cross-section, which biases the per-date Spearman IC. We instead keep the
# most recent dates whose total row count fits under the cap.


# %%
def _trim_by_complete_dates(X_arr, y_arr, ts_arr, sym_arr, max_samples):
    if len(X_arr) <= max_samples:
        return X_arr, y_arr, ts_arr, sym_arr
    unique_ts = np.sort(np.unique(ts_arr))[::-1]
    cumulative = 0
    keep_dates: list = []
    for ts in unique_ts:
        n = int((ts_arr == ts).sum())
        if cumulative + n > max_samples and keep_dates:
            break
        cumulative += n
        keep_dates.append(ts)
    keep_mask = np.isin(ts_arr, np.array(keep_dates))
    return X_arr[keep_mask], y_arr[keep_mask], ts_arr[keep_mask], sym_arr[keep_mask]


X_train, y_train, _train_ts, _train_sym = _trim_by_complete_dates(
    X_train, y_train, timestamps[train_mask], symbols[train_mask], MAX_TRAIN_SAMPLES
)
X_val, y_val, _val_ts, _val_sym = _trim_by_complete_dates(
    X_val, y_val, timestamps[val_mask], symbols[val_mask], MAX_VAL_SAMPLES
)
X_test, y_test, test_dates, test_symbols = _trim_by_complete_dates(
    X_test, y_test, test_dates, test_symbols, MAX_TEST_SAMPLES
)


# %% [markdown]
# ### Cross-sectional IC helper


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


print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
print(
    f"Purged {LABEL_HORIZON} target dates before each boundary before complete-date subsampling: "
    f"validation starts {train_end_date}, test starts {val_end_date}"
)

# %% [markdown]
# ## Selective state space - what the code computes
#
# At each timestep $t$ we discretize a continuous-time SSM via zero-order
# hold and run an input-dependent recurrence on a hidden state $h_t$. In
# Mamba the matrices $B_t$, $C_t$ and step size $\Delta_t$ are functions of
# the input $u_t$ (the **selective** ingredient), while $A$ is a learned
# diagonal that stays time-invariant. The state update and output are:
#
# $$h_t = \exp(\Delta_t A)\, h_{t-1} + (\Delta_t B_t)\, u_t$$
# $$y_t = C_t^\top h_t + D \cdot u_t$$
#
# The full continuous-time formulation and the derivation of ZOH live in
# Section 13.6; this notebook focuses on the discrete recurrence as
# implemented in `selective_scan`.

# %% [markdown]
# ## Selective SSM Block
#
# We split the block in three pieces - the recurrence itself, the block's
# parameter geometry (in `__init__`), and the gated forward pass - so each
# can be read on its own.
#
# > **Runtime warning**: `selective_scan` uses a Python `for` loop over
# > `seq_len`, making it ~100× slower than the production Mamba CUDA kernels.
# > Expect several minutes on the full ETF dataset. This is intentional: the
# > loop exposes the recurrence mechanics that hardware-efficient kernels hide.


# %% [markdown]
# ### Selective scan recurrence
#
# Pure-Python implementation of the discrete selective scan: at each step we
# discretize `A` via zero-order hold using the per-step size `dt_t`, then
# update the hidden state and read the output through the input-dependent
# `C_t`. Following the Mamba paper we approximate the input discretization
# as $\bar B_t \approx \Delta_t \cdot B_t$ rather than the full ZOH form.


# %%
def selective_scan(
    log_A: torch.Tensor,
    D: torch.Tensor,
    x_branch: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Run the input-dependent SSM recurrence in pure PyTorch."""
    A = -torch.exp(log_A)
    batch, seq_len, _ = x_branch.shape
    d_inner, _ = log_A.shape
    state = torch.zeros(batch, d_inner, log_A.shape[1], device=x_branch.device)
    outputs = []
    for t in range(seq_len):
        u_t = x_branch[:, t, :]
        B_t = B[:, t, :]
        C_t = C[:, t, :]
        dt_t = dt[:, t].unsqueeze(-1)
        A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
        input_term = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) * u_t.unsqueeze(-1)
        state = state * A_bar + input_term
        outputs.append(torch.einsum("bds,bs->bd", state, C_t) + u_t * D)
    return torch.stack(outputs, dim=1)


# %% [markdown]
# ### Block parameter geometry
#
# `in_proj` doubles the channel count to carry both the SSM branch and the
# gate branch. `x_proj` produces the time-varying `B_t`, `C_t`, and
# `Δ_t`-raw from the SSM branch itself. `log_A` is a learnable diagonal we
# negate before exponentiating so the recurrence is contractive. `D` is the
# direct skip from input to output.


# %%
class SelectiveSSMBlock(nn.Module):
    """Mamba-style block: input-dependent B/C/Δ feeding `selective_scan`."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1)
        self.log_A = nn.Parameter(torch.randn(d_inner, d_state) * 0.5)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_branch, z = self.in_proj(x).chunk(2, dim=-1)
        x_ssm = self.x_proj(x_branch)
        d_state = self.log_A.shape[1]
        B, C, dt_raw = x_ssm.split([d_state, d_state, 1], dim=-1)
        dt = F.softplus(dt_raw.squeeze(-1) + 1.0)
        y = selective_scan(self.log_A, self.D, x_branch, B, C, dt)
        y = self.out_proj(y * F.silu(z))
        return self.dropout(self.norm(y + residual))


# %% [markdown]
# ## Mamba Regressor
#
# Stacks multiple `SelectiveSSMBlock` layers with an input projection and
# a linear head that reads from the last timestep. This mirrors how an
# LSTM uses its final hidden state for prediction.


# %%
class MambaRegressor(nn.Module):
    """Mamba/SSM for regression: input projection, SSM layers, linear head."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        d_state: int = 16,
        n_layers: int = 2,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)

        self.layers = nn.ModuleList(
            [SelectiveSSMBlock(d_model, d_state, expand, dropout) for _ in range(n_layers)]
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        # Use last timestep
        x = x[:, -1, :]
        return self.fc(x).squeeze(-1)


# %%
set_global_seeds(SEED)
model = MambaRegressor(
    n_features=len(FEATURE_COLS),
    d_model=D_MODEL,
    d_state=D_STATE,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"MambaRegressor parameters: {n_params:,}")
print(f"Architecture: {N_LAYERS} SSM layers, d_model={D_MODEL}, d_state={D_STATE}")
print(f"Input: ({LOOKBACK} timesteps, {len(FEATURE_COLS)} features)")

# %% [markdown]
# ## Train Mamba

# %%
print("Training MambaRegressor...")
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
# ### Training convergence
#
# The loss curves trace how the selective-scan model learns before early
# stopping halts it. A validation curve that turns up while the training
# curve keeps falling is the overfitting signal the patience rule watches
# for, and it explains why the run stops well short of the epoch cap.

# %%
fig = go.Figure()
for label, key, color in [
    ("Train", "train_loss", COLORS["blue"]),
    ("Validation", "val_loss", COLORS["amber"]),
]:
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history[key]) + 1)),
            y=history[key],
            mode="lines+markers",
            name=label,
            line={"color": color},
        )
    )
best_epoch = int(np.argmin(history["val_loss"]) + 1)
fig.update_layout(
    title=f"Mamba stops at epoch {len(history['val_loss'])} after validation MSE bottoms at epoch {best_epoch}",
    xaxis_title="Epoch",
    yaxis_title="MSE loss",
)
fig.show()

# %% [markdown]
# ## Evaluate on Test Set

# %%
model.eval()
with torch.no_grad():
    y_pred_batches = []
    for i in range(0, len(X_test), INFER_BATCH_SIZE):
        X_test_t = torch.FloatTensor(X_test[i : i + INFER_BATCH_SIZE]).to(DEVICE)
        y_pred_batches.append(model(X_test_t).cpu().numpy())
    y_pred = np.concatenate(y_pred_batches)

test_mse = np.mean((y_pred - y_test) ** 2)
test_ic = cross_sectional_ic_mean(y_test, y_pred, test_dates, test_symbols)

print("\nMamba Test Results:")
print(f"  MSE: {test_mse:.6f}")
print(f"  Spearman IC: {test_ic:.4f}")

# %% [markdown]
# ## Ridge Baseline Comparison
#
# Flattening the 3D input to 2D and fitting Ridge regression provides a
# simple linear baseline to gauge whether the selective scan adds value.

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
#
# The left panel measures cross-sectional ranking. The right panel compares
# squared error with the zero-return forecast, a natural level benchmark for
# noisy return labels.

# %%
model_names = ["Mamba SSM", "Ridge"]
ic_values = [test_ic, ridge_ic]
mse_ratios = [test_mse / zero_mse, ridge_mse / zero_mse]
bar_palette = {"Mamba SSM": COLORS["blue"], "Ridge": COLORS["slate"]}

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
# ## Interpretation
#
# On this single-split multivariate ETF-feature setup (eight trailing-return
# horizons per ETF, sub-sampled to the most recent
# complete dates so each per-date Spearman uses the full cross-section),
# the paired figure reports both rank IC and squared error relative to zero.
# The loop-based selective scan is trained for only a handful of epochs on a
# capped sample. The point is the mechanism, not the horse race: the
# pure-Python scan exposes the selective-state-space recurrence but runs
# orders of magnitude slower than the production CUDA kernel, which caps the
# training budget and hyperparameter search. This notebook illustrates the
# architecture; it is not an apples-to-apples evaluation of Mamba against
# linear models on cross-sectional return prediction.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Linear complexity**: SSMs process sequences in $O(T)$ time vs $O(T^2)$
#    for self-attention, making them practical for very long sequences
# 2. **Continuous-time formulation**: The state space model is defined in
#    continuous time and discretized via ZOH, providing a principled
#    connection to differential equations and control theory
# 3. **Selective scan**: Making B, C, and $\Delta$ input-dependent lets the
#    model decide what to remember or forget at each step -- this is Mamba's
#    key innovation over fixed-parameter SSMs like S4
# 4. **Gated output**: The SiLU-gated branch (analogous to the gate in
#    LSTMs) provides multiplicative interaction that helps gradient flow
# 5. **Pedagogical vs production**: Our loop-based scan exposes the
#    selective-state-space mechanism but is much slower; production Mamba
#    uses custom CUDA kernels for parallel prefix sums and hardware-aware
#    state materialization
# 6. **Multi-scale variants**: The ms-Mamba architecture deploys parallel
#    Mamba blocks at different sampling rates to capture signals across
#    multiple timescales simultaneously -- see Section 13.6 for details
#
# Deterministic PyTorch algorithms and a fixed cuBLAS workspace make repeated
# executions reproducible on the same software and GPU stack; another environment
# may still produce small floating-point differences.
#
# **Next**: See `08_cnn_image_encoding` for encoding time series as images
# (Gramian Angular Fields, Markov Transition Fields) and classifying with CNNs.
#
# **Book**: Section 13.6 discusses Mamba alongside TCN, TSMixer, and other
# non-attention architectures for time series.
