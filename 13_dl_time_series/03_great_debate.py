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
# # Linear Baselines Against a Transformer
#
# **Docker image**: `ml4t-gpu`
#
# In 2022 Zeng and co-authors published a paper whose title was a question - *Are
# Transformers Effective for Time Series Forecasting?* - and whose answer was
# largely no. On nine standard forecasting benchmarks, models consisting of a single
# matrix multiplication matched or bettered a series of Transformer architectures
# that had been reported as the state of the art. The result did not show that
# attention cannot work on sequences. It showed that the papers claiming it worked
# had not been comparing against anything hard.
#
# This notebook reproduces the shape of that comparison on daily SPY returns. It
# builds the three linear models from the paper, a Transformer encoder, and two
# forecasts with no parameters at all, and scores them together. It then runs the
# diagnostic the paper used to argue that the Transformers were not using temporal
# order in the first place: shuffle the days inside each input window and see whether
# anything gets worse.
#
# **Learning objectives**:
# - Build the three LTSF-Linear models - a plain linear map from the window to the
#   forecast, one that splits the window into a smooth part and a remainder first, and
#   one that subtracts the last observation before mapping and adds it back after.
# - Build a Transformer encoder over the same window and see what it costs in
#   parameters relative to a single matrix.
# - Put a forecast of zero and a forecast that repeats the last value into the same
#   table, and read every trained model against them rather than against each other.
# - Destroy the time ordering inside each input window and measure what each model
#   loses, which is how you tell a model that uses sequence from one that does not.
#
# **Book Reference**: Chapter 13, Section 13.4 (Linear baselines versus transformers).
# Zeng et al. (2022), *Are Transformers Effective for Time Series Forecasting?*
#
# **Prerequisites**: `01_core_architectures`; ETF price data via the canonical
# `load_etfs()` loader.

# %%
"""Test whether simple linear models can outperform Transformers on time series."""

import os
import time
from datetime import datetime

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch
import torch.nn as nn
from plotly.subplots import make_subplots

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

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
# ## The data
#
# The comparison runs on one series: SPY's daily returns. That is deliberate. The
# claim under test is about single-series forecasting, and a panel of several funds
# would add a second thing the models differ on - how they handle a cross-section -
# which is not what is being measured here.
#
# Five funds are nonetheless loaded, and only to fix the calendar. Keeping the dates on
# which all five traded gives a single trading calendar with no gaps to explain, and
# the same one the other notebooks in this chapter use. SPY's column is then taken out
# of it.
#
# Returns rather than prices, unlike `02_nbeats_interpretable`. A return series has no
# trend to extrapolate past, so the problem that notebook measured - every test input
# outside the training range - does not arise. What replaces it is a harder problem:
# there is almost nothing to predict, which is exactly why the no-parameter forecasts
# below are hard to beat.

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

series = returns["SPY"].to_numpy().astype(np.float32)
dates = returns["timestamp"].to_numpy()

print(f"{len(series)} trading days shared by all {len(SYMBOLS)} funds, {dates[0]} to {dates[-1]}")
print(
    f"SPY daily return: mean {series.mean():+.5f}, standard deviation "
    f"{series.std():.5f}, so the mean is {abs(series.mean()) / series.std():.3f} "
    f"standard deviations from zero"
)

# %% [markdown]
# ### The series the models have to forecast
#
# Two panels. The left is the return series itself: no trend to speak of, a level that
# stays near zero, and volatility that clusters into bursts. The right is the
# autocorrelation - the correlation between each day's return and the return some
# number of days earlier - with the band inside which a correlation is
# indistinguishable from zero at a sample this size, namely $\pm 1.96/\sqrt{n}$.
#
# Read them together and the notebook's outcome stops being surprising. The printed
# mean is a small fraction of a standard deviation from zero, so forecasting zero is
# close to forecasting the mean. And if almost every autocorrelation sits inside the
# band, then the ordered history of returns has little linear relationship to what
# comes next, which is what every model in this notebook is being asked to find.

# %%
n_lags = 40
centred = series - series.mean()
denom = float((centred**2).sum())
acf = [float((centred[k:] * centred[:-k]).sum()) / denom for k in range(1, n_lags + 1)]
conf = 1.96 / np.sqrt(len(series))

fig_data = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["SPY daily returns", "Autocorrelation of those returns"],
)
fig_data.add_trace(
    go.Scatter(x=dates, y=series, line=dict(color=COLORS["blue"], width=1), name="Return"),
    row=1,
    col=1,
)
fig_data.add_trace(
    go.Bar(x=list(range(1, n_lags + 1)), y=acf, marker_color=COLORS["slate"], name="ACF"),
    row=1,
    col=2,
)
for sign in (1, -1):
    fig_data.add_hline(y=sign * conf, line_dash="dot", line_color=COLORS["neutral"], row=1, col=2)
fig_data.update_xaxes(title_text="Date", row=1, col=1)
fig_data.update_xaxes(title_text="Lag (trading days)", row=1, col=2)
fig_data.update_yaxes(title_text="Daily return", row=1, col=1)
fig_data.update_yaxes(title_text="Correlation with the return that many days earlier", row=1, col=2)
fig_data.update_layout(
    title="What there is to forecast, before any model is fitted", showlegend=False
)
show_plotly_with_alt(
    fig_data,
    "Two panels. The left plots SPY's daily return against date, centred on zero with "
    "volatility arriving in bursts. The right is a bar chart of the correlation "
    "between a day's return and the return a given number of days earlier, for lags "
    "one to forty, with dotted lines marking the band inside which a correlation is "
    "indistinguishable from zero at this sample size.",
)


# %% [markdown]
# ### Sequence builder
#
# Each input is `LOOKBACK` consecutive daily returns and each target is the
# `HORIZON` returns that follow it. Unlike `01_core_architectures`, where the target
# was a single day, the whole path is predicted at once - which is what the linear
# models below map to in one matrix multiplication, and what makes the parameter
# comparison against a Transformer meaningful.


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
# The cutoffs are positions in the return series, and an example belongs to the
# partition its forecast target falls in. A target spans `HORIZON` days, so the
# examples whose horizon would straddle a boundary are dropped rather than assigned to
# one side: no target date appears in two partitions. Chapter 6 sets out the
# walk-forward procedure that a production estimate needs; a single chronological cut
# is enough for a comparison between models that all face the same cut.

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
# ### D-Linear: split the window before mapping it
#
# The window is first separated into a smooth part and what is left over. The smooth
# part is a moving average - each day replaced by the average of the `kernel_size` days
# around it - and the remainder is the window minus that average. Two independent
# linear maps then run, one on each part, and their forecasts are added. The idea is
# that a slow drift and the fluctuations around it are different things to extrapolate,
# and giving each its own matrix lets the model treat them differently.
#
# The averaging needs the window extended at both ends to keep its length, and the
# extension is a repeat of the edge values rather than zeros. Padding with zeros would
# pull the average towards zero at exactly the two places the forecast depends on
# most - the start of the window and, worse, its final day.


# %%
class DLinear(nn.Module):
    """Decomposition-Linear: separate linear for trend and remainder."""

    def __init__(self, lookback, horizon, kernel_size=25):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.kernel_size = kernel_size
        # Padding is applied in `forward` instead, in replicate mode.
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
# ### The Transformer, as the critique found it
#
# Each day of the window is projected into a `D_MODEL`-dimensional vector - a token -
# and a learned position vector is added so the encoder can tell day 3 from day 47,
# which self-attention cannot otherwise do: attention compares every token to every
# other with no notion of which came first. The encoder layers then run, and the head
# flattens all `LOOKBACK` token vectors into one long vector and maps it to the
# forecast.
#
# That head is worth noticing, because it is where most of the parameters go and it is
# a large part of why the parameter counts below differ by the factor they do. It is
# also the design the critique was aimed at: after paying for attention over positions,
# the model concatenates every position anyway. `04_transformers` builds two
# architectures that change what a token is - a patch of consecutive days in one, a
# whole feature's history in the other - rather than keeping this shape.


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

summary_df = pl.DataFrame(
    [
        {
            "Forecast": name,
            "Parameters": results[name]["params"],
            "Test MSE": results[name]["mse"],
            "MSE / zero forecast": round(results[name]["mse_ratio"], 3),
            "Train seconds": round(results[name]["time"], 1),
        }
        for name in ["Zero", "Closest Repeat", "Linear", "D-Linear", "N-Linear", "Transformer"]
    ]
)
summary_df

# %% [markdown]
# ### How to read the table
#
# The column that decides everything is **MSE / zero forecast**. Predicting zero for
# every day of every horizon is what a forecaster does when it has nothing to say, and
# on a return series it is not a straw man: returns average close to zero, so the
# squared error of that forecast is essentially the variance of the returns
# themselves. A model scoring above one has spent its parameters getting further from
# the target than saying nothing would have.
#
# **Closest Repeat** - carry the last observed return forward across the whole horizon
# - is the second no-parameter entry, and it is the one the LTSF papers use. On a
# return series it should do poorly, and for a reason worth stating: yesterday's return
# is a draw from a nearly zero-mean distribution, so repeating it commits to a nonzero
# path for the entire horizon on the strength of one noisy observation. That makes it a
# worse baseline than zero here, which is itself informative - a benchmark's difficulty
# depends on what is being forecast, and importing one from a paper on electricity
# demand does not import its difficulty.
#
# The **Parameters** column is the other half of the argument. Whatever the errors turn
# out to be, they were reached at very different cost, and the linear models have
# nothing in them but one or two matrices from the window to the horizon.

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
    title="Every trained model, measured against saying nothing",
    xaxis_title="Forecast",
    yaxis_title="Test MSE relative to the zero forecast",
    showlegend=False,
    yaxis_range=[0, max(results[name]["mse_ratio"] for name in ordered_models) * 1.15],
)
fig.add_hline(
    y=1,
    line_dash="dot",
    line_color=COLORS["neutral"],
    annotation_text="zero forecast",
    annotation_position="right",
)
show_plotly_with_alt(
    fig,
    "A bar chart of six forecasts, from the two no-parameter rules through the three "
    "linear models to the Transformer, each bar giving its test mean squared error as "
    "a multiple of the zero forecast's, with a dotted line at one.",
)

# %% [markdown]
# ## Does any of them use the ordering?
#
# A model that reads a sequence should get worse when the sequence stops being one.
# The diagnostic is to take each test input window, shuffle its days into a random
# order, and score the model again. The model's weights do not change; only the
# arrangement of what it is shown does. A model whose error jumps was relying on which
# day came when. A model whose error barely moves was reading the window as an
# unordered bag of numbers, whatever its architecture suggests.
#
# This is the test that made the original critique sharp. Zeng and co-authors found
# that shuffling hurt the linear models substantially and the Transformers hardly at
# all on their benchmarks - so the attention layers, whose entire justification is
# modelling relations between positions, were not using position.
#
# The same diagnostic is run here, on a different kind of series, and its outcome has
# to be read against that difference. Those benchmarks are electricity load and
# exchange rates, with daily and weekly cycles to be found. Daily equity returns have
# very little sequential structure for any model to use, so a small shuffle effect
# here says something about the series and not only about the architecture. Both
# readings are available and the section below separates them.

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
    # How much the error moved, and separately how far the predictions themselves
    # moved: the root-mean-square gap between the two prediction sets, divided by the
    # root-mean-square size of the original predictions. Zero means the model emitted
    # the same numbers; one means it disagreed with itself by as much as it was
    # predicting in the first place.
    pred_shift = float(
        np.sqrt(np.mean((pred_shuf - pred_orig) ** 2)) / np.sqrt(np.mean(pred_orig**2))
    )
    shuffle_results.append(
        {
            "Model": name,
            "MSE (original)": round(mse_orig, 6),
            "MSE (shuffled)": round(mse_shuf, 6),
            "Delta (%)": round(100 * (mse_shuf - mse_orig) / mse_orig, 1),
            "Prediction shift": round(pred_shift, 3),
        }
    )

shuffle_df = pl.DataFrame(shuffle_results)
shuffle_df

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Change in error", "How far the predictions moved"],
)
fig.add_trace(
    go.Bar(
        x=shuffle_df["Model"],
        y=shuffle_df["Delta (%)"],
        marker_color=COLORS["blue"],
        text=[f"{value:+.1f}%" for value in shuffle_df["Delta (%)"]],
        textposition="outside",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=shuffle_df["Model"],
        y=shuffle_df["Prediction shift"],
        marker_color=COLORS["amber"],
        text=[f"{value:.3f}" for value in shuffle_df["Prediction shift"]],
        textposition="outside",
    ),
    row=1,
    col=2,
)
fig.add_hline(y=0, line_color=COLORS["neutral"], row=1, col=1)
fig.add_hline(
    y=1.0,
    line_dash="dot",
    line_color=COLORS["neutral"],
    annotation_text="disagreement as large as the forecast",
    annotation_position="bottom right",
    row=1,
    col=2,
)
fig.update_yaxes(title_text="Change in test MSE (%)", row=1, col=1)
fig.update_yaxes(title_text="Prediction shift (0 = unchanged)", row=1, col=2)
fig.update_layout(title="Two different questions about the same shuffle", showlegend=False)
show_plotly_with_alt(
    fig,
    "Two bar charts over the same four trained models. The left gives the percentage "
    "change in test mean squared error when the days inside every input window are "
    "randomly reordered. The right gives how far each model's predictions moved, as the "
    "root-mean-square gap between its original and shuffled predictions divided by the "
    "root-mean-square size of the originals, with zero meaning unchanged.",
)

# %% [markdown]
# The two panels answer different questions, and the difference between them is the
# point of this section.
#
# **The left panel says only how much the average squared error moved.** Error is an
# average over every test window, and an average hides what happened inside it: two
# sets of predictions can both miss a near-zero target by a similar amount while
# disagreeing completely with each other. A bar near zero here means shuffling cost the
# model little accuracy. It does not mean the model produced the same forecast.
#
# **The right panel asks whether the forecast changed at all.** It measures the
# root-mean-square gap between each model's original and shuffled predictions, divided
# by the root-mean-square size of the original predictions. Zero means the model
# emitted the same numbers from the reordered window and therefore did not read
# position - within these test windows and this one shuffle. A value of one means the
# reordering moved the forecast by as much as the whole forecast is worth, and a value
# above one means the two disagree by more than that, which is what happens when the
# reordered prediction is not merely different but points the other way.
#
# It is a gap and not a correlation on purpose. A correlation of one would be satisfied
# by predictions that are twice the originals plus a constant, which is a model whose
# output very much depends on position; correlation measures linear association and not
# agreement, and only a distance answers "did the numbers change".
#
# Read the two panels together and check them against what the critique predicts. Its
# claim is that the linear models depend on position and the Transformer does not, so
# it predicts a lopsided picture on the right - the linear bars high, the Transformer's
# near zero - while the left panel may show very little for anybody. If that is what
# the chart shows, then attention layers whose entire justification is modelling
# relations between positions are producing an output that barely depends on position,
# and no accuracy table would have revealed it.
#
# **What this does not settle.** No uncertainty has been estimated for the left panel,
# so a small bar there is not established as a real effect and neither is a large one.
# Estimating it would take more than counting windows, because these windows overlap:
# consecutive examples share `HORIZON - 1` of their target days, so their errors are
# heavily dependent and the effective number of independent observations is far below
# the number of rows. A test that ignored that would report a confidence interval far
# narrower than the evidence supports. And a model whose predictions move while its
# error does not is reading position and getting nothing for it, which on daily equity
# returns is the expected outcome and is a statement about the series rather than the
# architecture.
#
# The general lesson holds whatever the dataset: a claim that an architecture exploits
# some structure is testable by destroying that structure in the input and re-scoring.
# It costs one forward pass and no accuracy table implies it.

# %% [markdown]
# ## Does a longer input window help?
#
# Zeng and co-authors reported that giving the linear models longer inputs kept
# improving them while the Transformers stayed flat or got worse - a second sign that
# the extra positions were not being used. Three window lengths are fitted here.
#
# Everything is scored on the **validation** windows. Comparing candidates is what
# those are for, and reading the held-back stretch to pick a window length would spend
# it: every number reported from it afterwards would be a report on a choice already
# made using it.
#
# Two things are pinned so the sweep varies only the window. Every candidate faces the
# same target dates, because the cutoffs are fixed in the return series rather than as
# a fraction of each candidate's own sequence count. And every candidate starts its
# training examples on the date the longest window can first reach, so all three are
# fitted on the same days - otherwise the shorter windows would train on more examples
# and a difference in error could be a difference in sample size.

# %%
lookback_values = [48, 96, 192]
sweep_first_target = max(lookback_values)
lookback_results = []

for lb in lookback_values:
    X_lb, y_lb = create_sequences(series, lb, HORIZON)
    target_start_lb = np.arange(lb, len(series) - HORIZON + 1)
    target_end_lb = target_start_lb + HORIZON - 1
    train_mask_lb = (target_start_lb >= sweep_first_target) & (target_end_lb < train_target_cutoff)
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
    print(f"  {lb}-day window, {int(train_mask_lb.sum())} training examples: done")

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
fig.add_hline(
    y=1,
    line_dash="dot",
    line_color=COLORS["neutral"],
    annotation_text="zero forecast",
    annotation_position="right",
)
fig.update_layout(
    title="Validation error against the length of the input window",
    xaxis_title="Input window (trading days)",
    yaxis_title="Validation MSE relative to the zero forecast",
    xaxis=dict(tickmode="array", tickvals=lookback_values),
)
show_plotly_with_alt(
    fig,
    "Three lines, one each for the plain linear model, the decomposition linear "
    "model and the Transformer, plotting validation mean squared error as a multiple "
    "of the zero forecast's against input window length at three settings, with a "
    "dotted line at one.",
)

# %% [markdown]
# Three points per curve is enough to see whether a curve moves and not enough to
# describe how an architecture scales - the original study swept far more settings on
# data with far more signal. Read this as what it is: a check that the window length
# was chosen by measurement rather than by habit, run on the partition reserved for
# choosing things.

# %% [markdown]
# ## Key takeaways
#
# 1. **A comparison between two architectures says nothing until something with no
#    parameters is in the table.** Two models can be ranked against each other while
#    both are worse than predicting zero, and the ranking looks like a finding right up
#    until the third row is added. This is the whole content of the critique the
#    notebook reproduces: the Transformer papers were compared against each other.
# 2. **Import a baseline's identity, not its difficulty.** Repeating the last
#    observation is a demanding benchmark on a price level and a weak one on a return
#    series, because a return is a draw from a nearly zero-mean distribution and
#    repeating it commits to a path on the strength of one noisy number. A benchmark
#    taken from a paper on other data has to be re-argued for yours.
# 3. **Test a structural claim by destroying the structure.** If an architecture is
#    said to exploit temporal order, shuffle the order inside each input and run it
#    again. It costs one forward pass, and no accuracy table implies it.
# 4. **Score that test on the predictions, not only on the error.** An average squared
#    error can sit still while the predictions underneath it change completely, so a
#    flat error bar answers nothing on its own. Measuring the distance between the
#    original and shuffled predictions asks the question directly, and a distance is
#    what it has to be: a correlation of one is satisfied by any affine rescaling, so
#    it cannot tell you the numbers did not change.
# 5. **Choose settings on the validation partition and say that you did.** The window
#    sweep here is model selection, and reporting it as a result would spend the
#    held-back stretch on a choice already made.
#
# **Known limitations.** One series, one chronological split, one seed, one horizon,
# and no hyperparameter search beyond the window sweep - so nothing here ranks these
# architectures in general, and the specific outcome on daily equity returns is
# dominated by how little there is to forecast. The Transformer is the vanilla encoder
# the critique targeted, with a head that flattens every position, rather than any of
# the designs that answered it; `04_transformers` builds two of those. Training times
# in the table are wall-clock on a shared machine and are indicative only.
#
# **Next**: `04_transformers` builds PatchTST and iTransformer, both of which change
# what a token is in order to give attention something positional to work with.
