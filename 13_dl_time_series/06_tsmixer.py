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
# # TSMixer: mixing one axis at a time
#
# **Docker image**: `ml4t-gpu`
#
# `04_transformers` let one position in the window depend on another through
# attention, and `05_tcn` did the same through dilated convolution. Both are devices
# for relating positions. A fully connected layer already relates every input to every
# other, so the question TSMixer (Google, 2023) asks is not whether to use one but
# along which axis to apply it.
#
# Its answer is to alternate, and never to mix both axes in a single layer. A
# **time-mixing** layer applies one shared linear map across the `LOOKBACK` days, the
# same map for every feature. A **feature-mixing** layer applies a small MLP across the
# features, separately at each day. Stacking the two lets a representation depend on
# both axes at a cost of $T^2$ mixing weights along time and $2FH$ across features,
# where a single dense layer over the flattened window would cost $(TF)^2$. The model
# prints its parameter count below, so the arithmetic is checkable.
#
# **Learning objectives**:
# - Read a tensor's axes well enough to say what a permutation before a `Linear` layer
#   changes about which numbers get combined.
# - Build the two mixing layers and say what each one can and cannot represent: the
#   time-mixing map is the same for every feature, and the feature-mixing MLP is the
#   same at every day.
# - Say what the residual connection and the pre-normalisation are for in a stack that
#   has no recurrence and no convolution to stabilise.
# - Score the result against a penalised linear map on the same flattened window,
#   which is the comparison that decides whether the mixing structure earned
#   anything.
#
# **Book Reference**: Chapter 13, Section 13.6 (Alternative architectures and foundation models)
#
# **Prerequisites**: ETF features (`case_studies/etfs/`)

# %%
"""Build TSMixer with alternating time and feature mixing for return prediction."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch
import torch.nn as nn
from dl_sequences import create_sequences_multi_asset, load_dl_dataset, train_model
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

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
per_date = df.group_by(mds.date_col).len().sort(mds.date_col)
print(
    f"{df[mds.date_col].min()} to {df[mds.date_col].max()}, "
    f"{df[mds.entity_cols[0]].n_unique()} funds; funds per date "
    f"{per_date['len'].min()} to {per_date['len'].max()}, median {per_date['len'].median():.0f}"
)
print(
    f"Label {TARGET_COL}: mean {df[TARGET_COL].mean():+.5f}, "
    f"standard deviation {df[TARGET_COL].std():.5f}"
)

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

# %% [markdown]
# ### Splitting by date, with a gap for the label horizon
#
# The split is by date, at fixed fractions of the trading days, and an example belongs
# to the partition the date it carries falls in. The label is a `LABEL_HORIZON`-day
# forward return, so an example dated within that many days of a boundary has an
# outcome resolved by days on the far side; those examples are dropped. Input windows
# may still reach back over a boundary, which is right - at decision time the model has
# every past observation available.

# %%
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
# Mean cross-sectional Spearman IC by date - the same metric used in
# `01_core_architectures`, `04_transformers` and `05_tcn`, so the comparison anchors
# on one per-date rank correlation across the section's architectures.
#
# A date's IC is undefined when a model predicts the same number for every fund on it:
# the predicted ranks are all tied and there is nothing to correlate. The library
# returns `NaN` for such a date, and polars treats `NaN` and null as different values,
# so `drop_nulls` alone leaves it in place and one of them makes the whole mean `NaN`.
# Both are filtered here, and the count of dates the mean was taken over is printed
# beside it, so a model that ties often is visible rather than averaged over whichever
# dates happened to survive.


# %%
def cross_sectional_ic_mean(y_true, y_pred, dates, syms):
    """Mean cross-sectional Spearman IC over the dates where it is defined.

    Returns the mean and the defined/total date counts. Filters both null and NaN,
    since polars `drop_nulls` leaves NaN in place.
    """
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
    defined = ic_per_date.filter(pl.col("ic").is_not_null() & pl.col("ic").is_not_nan())
    mean_ic = float(defined["ic"].mean()) if defined.height else float("nan")
    return {"ic": mean_ic, "n_defined": defined.height, "n_total": ic_per_date.height}


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
    title="Training and validation error per epoch",
    xaxis_title="Epoch",
    yaxis_title="Mean squared error",
    height=470,
)
show_plotly_with_alt(
    fig,
    "A line chart of mean squared error against epoch, with one line for the training "
    "set and one for the validation set. Training stops when the validation line has "
    "gone the required number of epochs without a new minimum.",
)

# %% [markdown]
# ## Evaluate on Test Set

# %%
model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_pred = model(X_test_t).cpu().numpy()

test_mse = np.mean((y_pred - y_test) ** 2)
mixer_ic = cross_sectional_ic_mean(y_test, y_pred, test_dates, test_symbols)
test_ic = mixer_ic["ic"]

print("\nTSMixer Test Results:")
print(f"  MSE: {test_mse:.6f}")
print(f"  Spearman IC: {test_ic:.4f}", end="")
print(f"  (defined on {mixer_ic['n_defined']} of {mixer_ic['n_total']} test dates)")

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
ridge_ic_result = cross_sectional_ic_mean(y_test, y_ridge_pred, test_dates, test_symbols)
ridge_ic = ridge_ic_result["ic"]
zero_mse = float(np.mean(y_test**2))

print("\nRidge Baseline Results:")
print(f"  MSE: {ridge_mse:.6f}")
print(f"  Spearman IC: {ridge_ic:.4f}", end="")
print(f"  (defined on {ridge_ic_result['n_defined']} of {ridge_ic_result['n_total']} test dates)")

# %% [markdown]
# ## The mixer against the linear baseline
#
# Two questions, two panels. The left asks whether the model ordered the funds usefully
# on each date; the right asks whether its predicted return levels were closer than
# predicting zero. A model can do better on one and worse on the other, and both are
# reported because acting on a forecast uses the ordering while fitting one minimises
# the squared error.
#
# The ridge regression is the comparison that decides anything. It sees the same window
# flattened into one vector and fits a penalised linear map straight to the label: $TF$
# coefficients, no hidden representation, and no notion that one axis is time and the
# other is features. Whatever the alternating structure is worth has to appear as a
# difference from that.

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

fig.add_hline(y=0, line_color=COLORS["neutral"], row=1, col=1)
fig.add_hline(y=1, line_dash="dot", line_color=COLORS["neutral"], row=1, col=2)
fig.update_layout(
    title="TSMixer and ridge on the same test split, ranked and levelled",
    height=480,
)
fig.update_yaxes(title_text="Spearman IC", row=1, col=1)
fig.update_yaxes(title_text="MSE / zero-return MSE", row=1, col=2)
show_plotly_with_alt(
    fig,
    "Two bar panels, one bar per model. The left panel gives each model's mean "
    "cross-sectional Spearman IC against a line at zero; the right gives its test MSE "
    "as a multiple of the zero forecast's, against a dotted line at one.",
)

# %% [markdown]
# The left panel measures cross-sectional ranking, while the right panel asks
# whether either fitted model improves squared error over predicting zero. These
# are distinct questions. This purged single split demonstrates the architecture;
# it does not establish a stable model ranking. Section 13.9 supplies the
# walk-forward comparison across datasets.

# %% [markdown]
# ## Key takeaways
#
# 1. **A permutation before a `Linear` decides which numbers get combined.** The two
#    mixing layers hold the same kind of object - a dense matrix - and differ only in
#    the axis it is applied along. Reading the permutation is how you know what a block
#    can represent, and it is the one thing to check when adapting this architecture to
#    a different tensor layout.
# 2. **Each mixing layer is shared along the axis it does not mix.** One temporal map
#    serves all features, and one feature MLP serves all days. That sharing is what
#    keeps the mixing weights at $T^2 + 2FH$ rather than the $(TF)^2$ of a dense layer
#    over the flattened window, and it is also the assumption to doubt first: it says
#    the same temporal pattern matters in every feature.
# 3. **The residual and the pre-normalisation are load-bearing.** With no recurrence
#    and no convolution, a stack of dense layers over a 60-day axis has nothing else
#    keeping its scale in range; every mixing layer here is wrapped in both.
# 4. **The head is an adaptation, not part of the paper.** TSMixer's temporal forecast
#    head produces one value per feature, because the paper forecasts every channel.
#    This notebook's label is one cross-sectional return, so a small feature head maps
#    those per-feature values to a scalar - a modelling choice this notebook makes and
#    the reader should see.
#
# **Known limitations.** One chronological split of one ETF panel, one label horizon,
# one seed, and one block count. The comparison is against one baseline, and a single
# split cannot rank architectures; `12_case_study_insights` is where these families are
# compared across case studies under walk-forward validation. Deterministic PyTorch
# algorithms and a fixed cuBLAS workspace make repeated execution reproduce on the same
# software and GPU; another environment will differ in the final decimals.
#
# **Next**: `07_mamba_ssm` recovers a recurrent state, which both mixing and attention
# gave up, and scales linearly in the sequence length - where this mixer's
# `Linear(T, T)` costs $T^2$ per feature and attention costs $T^2$ outright.
