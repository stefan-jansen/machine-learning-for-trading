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
# # Temporal Convolutional Network
#
# **Docker image**: `ml4t-gpu`
#
# `01_core_architectures` measured the cost of walking a window one day at a time, and
# `04_transformers` answered it with attention. This notebook takes the third route:
# convolution, which reads the whole window in parallel like the fully connected
# network but, unlike it, respects the ordering of the days.
#
# A plain convolution has two problems for forecasting. Its filter reads in both
# directions from each position, so the output at day $t$ mixes in days after $t$; and
# its reach is the width of that filter, so covering sixty days would take a very wide
# filter or very many layers. A **temporal convolutional network** answers both. Its
# convolution is *causal* - the output at day $t$ is a function of days up to $t$ and
# no later - which is what makes every position a legitimate forecast for its own date.
# And it multiplies the gap between the positions each filter reads by two at every
# layer - a *dilation* - so the reach grows geometrically with depth rather than
# linearly.
#
# **Learning objectives**:
# - Build a causal convolution and check which inputs each output can actually see,
#   rather than trusting the word "causal" in the class name.
# - Stack dilated blocks so the receptive field - the span of input one output depends
#   on - covers the whole window, and compute that span rather than assuming it.
# - Explain what a residual connection is doing in a stack this deep, and why the block
#   normalises weights rather than activations.
# - Score the result against a penalised linear map on the same window, which is the
#   comparison that decides whether the convolutional structure earned anything.
#
# **Book Reference**: Chapter 13, Section 13.6 (Alternative architectures and foundation
# models).
#
# **Prerequisites**: `04_transformers`; ETF features from `case_studies/etfs/`.

# %%
"""Build a TCN with dilated causal convolutions for return prediction."""

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
from torch.nn.utils.parametrizations import weight_norm

from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
SEED = 42
EPOCHS = 30
LOOKBACK = 60
BATCH_SIZE = 128
N_CHANNELS = 32
KERNEL_SIZE = 3
DROPOUT = 0.1
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
# ## The data
#
# The same ETF case-study panel `04_transformers` used: eight trailing-return features
# per fund per date, and a 21-day forward return as the label. Keeping the data fixed
# across the chapter's architecture notebooks is what lets their results be read
# against each other at all, and it is why `12_case_study_insights` can aggregate them.
#
# The panel's breadth is worth printing before anything is fitted, because the scoring
# metric is computed across whatever funds exist on each date, and that count moves as
# funds launch.

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
# Mean cross-sectional Spearman IC by date - TCN evaluation is on cross-asset
# ranking, not pooled point error, so the same date/entity-aware metric used in
# `01_core_architectures` and `04_transformers` is the right comparison anchor.
#
# A date's IC is undefined when a model predicts the same number for every fund on
# it: the predicted ranks are all tied and there is nothing to correlate. The library
# returns `NaN` for such a date, and polars treats `NaN` and null as different values,
# so `drop_nulls` alone leaves it in place and one of them makes the whole mean `NaN`.
# Both are filtered here, and the count of dates the mean was actually taken over is
# printed beside it, so a model that ties often is visible rather than averaged over
# whichever dates happened to survive.


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
# > **Note**: This fixed 60/20/20 split is a pedagogical simplification. Its
# > 21-day purge keeps forward-label windows disjoint, but production deployment
# > still requires the expanding walk-forward protocol from Chapter 6.

# %% [markdown]
# ## TCN Architecture
#
# The TCN consists of stacked causal convolution blocks with exponentially
# increasing dilation factors. Each block uses:
#
# 1. **Causal padding**: pad both ends by $(k-1)d$ and drop the right-hand
#    overhang, which is identical to padding only on the left - the output at $t$
#    depends on inputs up to $t$ and no later, and the sequence keeps its length
# 2. **Dilated convolutions**: Dilation factors of 1, 2, 4, 8 give an
#    exponentially growing receptive field
# 3. **Residual connections**: Enable training deeper networks
# 4. **Weight normalization and channel dropout**: Match the reference TCN
#    residual block without mixing information across batch members
#
# The receptive field grows as:
#
# $$R = 1 + \sum_{i=0}^{L-1} 2 \cdot (k-1) \cdot d_i$$
#
# where $k$ is kernel size, $d_i = 2^i$ is the dilation at layer $i$, and
# $L$ is the number of layers.


# %%
class CausalConv1d(nn.Module):
    """1D convolution whose output at time t depends only on inputs at times <= t.

    ``nn.Conv1d`` pads both ends, so the right-hand overhang is dropped; the result
    is identical to padding only on the left, and the sequence keeps its length.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.causal_padding,
                dilation=dilation,
            )
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
# Each block follows Bai, Kolter, and Koltun (2018): two weight-normalized
# causal convolutions, ReLU activations, channel-wise dropout, and a residual
# connection. A 1x1 convolution aligns channel dimensions when necessary.


# %%
class TCNBlock(nn.Module):
    """Residual block with two dilated causal convolutions."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        return self.relu(out + res)


# %% [markdown]
# ### Full TCN Regressor
#
# Stacking blocks with dilations [1, 2, 4, 8] followed by a linear head on
# the final causal state. The final state can use the full receptive field;
# averaging across intermediate states would mix shorter effective histories.


# %%
class TCNRegressor(nn.Module):
    """Temporal Convolutional Network for regression.

    Architecture: Input -> [CausalConv(d=2^i) + ReLU + Dropout] x 4
                  -> final causal state -> Linear -> scalar output.
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
        self.fc = nn.Linear(n_channels, 1)

    def forward(self, x):
        # x input: (batch, seq_len, n_features) -> permute to (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x[:, :, -1]  # final causal state: (batch, n_channels)
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
set_global_seeds(SEED)
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
    "Two lines plotting the temporal convolutional network's training and validation "
    "mean squared error against epoch. The lines stop where early stopping halted "
    "training.",
)

# %% [markdown]
# ## Evaluate on Test Set

# %%
model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_pred = model(X_test_t).cpu().numpy()

test_mse = np.mean((y_pred - y_test) ** 2)
tcn_ic = cross_sectional_ic_mean(y_test, y_pred, test_dates, test_symbols)
test_ic = tcn_ic["ic"]

print("\nTCN Test Results:")
print(f"  MSE: {test_mse:.6f}")
print(f"  Spearman IC: {test_ic:.4f}", end="")
print(f"  (defined on {tcn_ic['n_defined']} of {tcn_ic['n_total']} test dates)")

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
ridge_ic_result = cross_sectional_ic_mean(y_test, y_ridge_pred, test_dates, test_symbols)
ridge_ic = ridge_ic_result["ic"]
zero_mse = float(np.mean(y_test**2))

print("\nRidge Baseline Results:")
print(f"  MSE: {ridge_mse:.6f}")
print(f"  Spearman IC: {ridge_ic:.4f}", end="")
print(f"  (defined on {ridge_ic_result['n_defined']} of {ridge_ic_result['n_total']} test dates)")

# %% [markdown]
# ## The convolutional network against the linear baseline
#
# Two questions, two panels. The left asks whether the model ordered the funds usefully
# on each date; the right asks whether its predicted return levels were closer than
# predicting zero. A model can do better on one and worse on the other, and both are
# reported because acting on a forecast uses the ordering while fitting one minimises
# the squared error.
#
# The ridge regression is the comparison that decides anything. It sees the same window
# flattened into one vector and fits a penalised linear map - no causality constraint,
# no dilation, no notion that the columns are ordered in time. Whatever the dilated
# causal structure is worth has to appear as a difference from that.

# %%
model_names = ["TCN", "Ridge"]
ic_values = [test_ic, ridge_ic]
mse_ratios = [test_mse / zero_mse, ridge_mse / zero_mse]
bar_palette = {"TCN": COLORS["blue"], "Ridge": COLORS["slate"]}

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
fig.add_hline(
    y=1,
    line_dash="dot",
    line_color=COLORS["neutral"],
    annotation_text="zero forecast",
    annotation_position="bottom right",
    row=1,
    col=2,
)
fig.update_layout(
    title="The convolutional network against a penalised linear map",
    height=480,
)
fig.update_yaxes(title_text="Mean daily Spearman IC", row=1, col=1)
fig.update_yaxes(title_text="Test MSE relative to the zero forecast", row=1, col=2)
show_plotly_with_alt(
    fig,
    "Two bar charts comparing the temporal convolutional network with the ridge "
    "baseline. The left gives each one's mean daily cross-sectional rank correlation "
    "against a line at zero; the right gives its test mean squared error as a multiple "
    "of the zero forecast's, against a dotted line at one.",
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
# 1. **Causality is a property of the trim, and it is checkable.** `nn.Conv1d` pads
#    both ends, so dropping the last $(k-1)d$ outputs is what leaves the output at day
#    $t$ a function of days up to $t$ and no later; differentiate an output position
#    with respect to the inputs to confirm that, rather than trusting the argument
#    name. It is not what keeps the target out of the input - the window does that,
#    because every day in the window precedes the target date and the head reads only
#    the final position. Causality is what would make each intermediate position a
#    forecast for its own date, and this notebook uses only the last.
# 2. **Doubling the dilation each layer buys reach geometrically.** Stacking blocks
#    whose dilation doubles makes the receptive field grow like $2^{\text{layers}}$
#    rather than linearly, which is how four blocks reach across the whole window.
#    The arithmetic is printed above rather than restated here, so it follows
#    `KERNEL_SIZE` and the dilation schedule if either changes.
# 3. **Fully parallelizable**: Unlike LSTMs, all timesteps are processed
#    simultaneously during both training and inference
# 4. **Fixed receptive field**: The maximum lookback is determined at design
#    time by the dilation schedule -- unlike attention, which adapts dynamically
# 5. **Reference architecture matters**: Weight normalization, channel-wise
#    dropout, and the final causal state preserve the TCN block's intended
#    inductive bias without mixing batch statistics
#
# **Known limitations.** One chronological split of one ETF panel, one label horizon,
# one seed, and a single dilation schedule - the receptive field was designed to cover
# the window rather than searched for. The comparison is against one baseline, and a
# single split cannot rank architectures; `12_case_study_insights` is where these
# families are compared across case studies under walk-forward validation. Repeated
# execution reproduces on the same software and GPU; another environment will differ in
# the final decimals.
#
# **Next**: `06_tsmixer` drops convolution too, and gets at the same structure with
# nothing but fully connected layers applied along one axis at a time.
