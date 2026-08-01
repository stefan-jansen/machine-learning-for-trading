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
# # Library Landscape: Unified Forecasting with sktime
#
# **Docker image**: `ml4t-gpu`
#
# This notebook compares raw PyTorch implementations against library-wrapped
# alternatives, positioning **sktime as the unifying meta-framework** for time
# series forecasting. sktime provides a consistent `fit()`/`predict()` API that
# wraps NeuralForecast, PyTorch Forecasting, HuggingFace, and more -- letting
# practitioners swap architectures without rewriting data pipelines.
#
# **Learning Objectives**:
# - Compare raw PyTorch vs library-wrapped implementations
# - Understand sktime's unified fit/predict API for time series
# - See how sktime wraps NeuralForecast, PyTorch Forecasting, and HuggingFace
# - Evaluate when to use raw PyTorch vs library abstractions
#
# **Book Reference**: Chapter 13, Section 13.7 (A Practitioner's Framework)
#
# **Prerequisites**: ETF price data via the canonical `load_etfs()` loader

# %%
"""Library Landscape - compare raw PyTorch vs sktime-wrapped forecasting implementations."""

import tempfile
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import torch
import torch.nn as nn
from darts import TimeSeries
from darts.models import RNNModel
from ml4t.diagnostic.metrics import pooled_ic
from sktime.forecasting.chronos import ChronosForecaster
from sktime.forecasting.patch_tst import PatchTSTForecaster

from utils.reproducibility import set_global_seeds
from utils.style import COLORS  # activates the ml4t Plotly template on import

warnings.filterwarnings("ignore")

from data import load_etfs

# %% tags=["parameters"]
SEED = 42
LOOKBACK = 60
HORIZON = 5
HIDDEN_SIZE = 64
EPOCHS = 30
BATCH_SIZE = 32
START_DATE = "2015-01-01"


# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

set_global_seeds(SEED)

# %% [markdown]
# ## Data: SPY Close Prices
#
# We use a single univariate series (SPY daily closes) so every library sees
# identical input. We prepare two representations: NumPy sequences for raw
# PyTorch, and a pandas Series with business-day index for sktime/Darts.

# %%
etf_df = load_etfs()
start_dt = datetime.fromisoformat(START_DATE)

spy = (
    etf_df.filter((pl.col("symbol") == "SPY") & (pl.col("timestamp") >= start_dt))
    .sort("timestamp")
    .select(["timestamp", "close"])
)

prices = spy["close"].to_numpy().astype(np.float32)

# Fix the train/test boundary in sequence space first, then normalize with
# TRAIN-ONLY price statistics. Using the full series' mean/std here would fold
# test-period price levels into the normalization of the training inputs - a
# full-sample-statistic leak. `train_price_end` is the last price index any
# training sequence can see (input window + forecast horizon).
n_seq = len(prices) - LOOKBACK - HORIZON + 1
split_idx = int(n_seq * 0.8)
train_price_end = split_idx + LOOKBACK + HORIZON
price_mean = prices[:train_price_end].mean()
price_std = prices[:train_price_end].std()
prices_norm = (prices - price_mean) / price_std

print(f"SPY: {len(prices):,} daily observations")

# %% [markdown]
# ### PyTorch Sequences (Normalized)


# %%
def create_sequences(data, lookback, horizon):
    """Create (input, target) pairs for univariate forecasting."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon].mean())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


X, y = create_sequences(prices_norm, LOOKBACK, HORIZON)
# split_idx was fixed above (in price space) so the normalization stays train-only
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_train_t = torch.FloatTensor(X_train).unsqueeze(-1).to(DEVICE)
y_train_t = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
X_test_t = torch.FloatTensor(X_test).unsqueeze(-1).to(DEVICE)

print(f"Sequences: train={len(X_train):,}, test={len(X_test):,}")

# %% [markdown]
# ### sktime / Darts Series (Prices with DatetimeIndex)

# %%
spy_pd = spy.to_pandas()
spy_series = spy_pd.set_index("timestamp")["close"]
spy_series = spy_series.sort_index()
spy_series.index = pd.DatetimeIndex(spy_series.index)

# Reindex to business-day frequency, forward-filling gaps (holidays etc.)
# Libraries like sktime/NeuralForecast and Darts require a regular frequency.
bday_idx = pd.bdate_range(start=spy_series.index.min(), end=spy_series.index.max(), freq="B")
spy_series = spy_series.reindex(bday_idx).ffill().dropna()
spy_series.index.freq = "B"

sk_split = int(len(spy_series) * 0.8)
y_train_sk = spy_series.iloc[:sk_split]
y_test_sk = spy_series.iloc[sk_split : sk_split + HORIZON]
fh_sk = list(range(1, HORIZON + 1))

# %% [markdown]
# ### Evaluation Helper
#
# **Note on Spearman IC**: In this univariate setting, IC reduces to rank
# correlation between predicted and actual values at consecutive time points -
# not the cross-sectional rank correlation used in multi-asset notebooks (04-10).
# We include it for metric consistency across notebooks, but MSE is the primary
# metric here.


# %%
def evaluate(y_true, y_pred):
    """Compute MSE and Spearman IC."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    ic = pooled_ic(y_pred, y_true) if len(y_true) > 5 else float("nan")
    return {"mse": round(mse, 6), "ic": round(ic, 4) if np.isfinite(ic) else None}


results = []  # Collector for final comparison

# %% [markdown]
# ## Part 1: Raw PyTorch LSTM
#
# The "full control" baseline: model definition, training loop, and evaluation
# require roughly 50 lines of boilerplate.


# %%
class LSTMForecaster(nn.Module):
    """Minimal LSTM for univariate forecasting."""

    def __init__(self, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# %%
model = LSTMForecaster(HIDDEN_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

start = time.time()
for epoch in range(EPOCHS):
    model.train()
    indices = torch.randperm(len(X_train_t))
    epoch_loss, n_batches = 0.0, 0
    for i in range(0, len(X_train_t), BATCH_SIZE):
        batch_idx = indices[i : i + BATCH_SIZE]
        loss = criterion(model(X_train_t[batch_idx]), y_train_t[batch_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch + 1}/{EPOCHS}: loss={epoch_loss / n_batches:.6f}")
pytorch_time = time.time() - start

model.eval()
with torch.no_grad():
    preds_pt = model(X_test_t).cpu().numpy().flatten()

m = evaluate(y_test, preds_pt)
results.append(
    {
        "Model": "LSTM",
        "Library": "Raw PyTorch",
        "Lines": "~50",
        "Train Time (s)": round(pytorch_time, 1),
        **m,
        "Trainable": True,
        "Target Scale": "normalized",
        "Evaluation Points": len(y_test),
    }
)
print(f"Raw PyTorch LSTM: {pytorch_time:.1f}s, MSE={m['mse']}, IC={m['ic']}")

# %% [markdown]
# The raw PyTorch LSTM requires defining the model class, training loop, and
# evaluation - approximately 50 lines. Full control over architecture and
# training dynamics, at the cost of significant boilerplate.

# %% [markdown]
# ## Part 2: sktime + NeuralForecast LSTM
#
# sktime wraps NeuralForecast's LSTM: three lines replace the 50-line training
# loop. The underlying recurrent architecture is equivalent.
#
# **Dependency note**: sktime's neural forecasters require `neuralforecast`, which
# depends on `ray` - and Python 3.14 wheels are still pending in
# ([ray-project/ray#56434](https://github.com/ray-project/ray/issues/56434)).
# Once those wheels land, install via `uv pip install neuralforecast`
# and uncomment the demo below. Recent sktime releases also renamed
# `hidden_size` to `encoder_hidden_size`.

# %%
# from sktime.forecasting.neuralforecast import NeuralForecastLSTM
#
# forecaster = NeuralForecastLSTM(
#     freq="B",
#     input_size=LOOKBACK,
#     max_steps=EPOCHS * 10,
#     encoder_hidden_size=HIDDEN_SIZE,
# )
# start = time.time()
# forecaster.fit(y_train_sk)
# y_pred = forecaster.predict(fh=fh_sk)
# t = time.time() - start
#
# m = evaluate(y_test_sk.values, y_pred.values)
# results.append(
#     {
#         "Model": "LSTM",
#         "Library": "sktime/NeuralForecast",
#         "Lines": "~5",
#         "Train Time (s)": round(t, 1),
#         **m,
#         "Trainable": True,
#     }
# )
# print(f"sktime NeuralForecastLSTM: {t:.1f}s, MSE={m['mse']}, IC={m['ic']}")

# %% [markdown]
# sktime reduces the same LSTM to three lines (`fit`/`predict`), delegating to
# NeuralForecast's optimized training loop underneath.

# %% [markdown]
# ## Part 3: sktime + PyTorch Forecasting N-BEATS
#
# N-BEATS (Section 13.2) via sktime's PyTorch Forecasting wrapper. Same
# `fit()`/`predict()` API, different architecture underneath.
#
# **Dependency note**: sktime's PyTorch Forecasting wrapper requires
# `pytorch-forecasting`. The previous run also tripped over a wrapper API
# mismatch around `max_prediction_length`/`max_encoder_length`. Once the
# wrapper signature stabilizes, install `pytorch-forecasting` and uncomment
# the demo below.

# %%
# from sktime.forecasting.pytorchforecasting import PytorchForecastingNBeats
#
# forecaster = PytorchForecastingNBeats(
#     max_prediction_length=HORIZON,
#     max_encoder_length=LOOKBACK,
#     max_epochs=min(EPOCHS, 10),
#     trainer_kwargs={"enable_progress_bar": False, "accelerator": "auto"},
# )
# start = time.time()
# forecaster.fit(y_train_sk)
# y_pred = forecaster.predict(fh=fh_sk)
# t = time.time() - start
#
# m = evaluate(y_test_sk.values, y_pred.values)
# results.append(
#     {
#         "Model": "N-BEATS",
#         "Library": "sktime/PyTorchForecasting",
#         "Lines": "~5",
#         "Train Time (s)": round(t, 1),
#         **m,
#         "Trainable": True,
#     }
# )
# print(f"sktime N-BEATS: {t:.1f}s, MSE={m['mse']}, IC={m['ic']}")

# %% [markdown]
# Same three-line API, different architecture. Swapping LSTM for N-BEATS requires
# changing only the import and constructor - no data pipeline changes.

# %% [markdown]
# ## Part 4: sktime PatchTST
#
# PatchTST (Section 13.5) via sktime's HuggingFace integration. Supports
# full training, fine-tuning, and zero-shot modes.
#
# **Dependency note**: PatchTST uses sktime's HuggingFace backend, not
# `neuralforecast`, so it runs in the current Python 3.14 environment. The
# first run downloads model artifacts and trains into a temporary output
# directory.

# %%
forecaster = PatchTSTForecaster(
    fit_strategy="full",
    config={
        "context_length": LOOKBACK,
        "prediction_length": HORIZON,
        "patch_length": 12,
        "num_hidden_layers": 2,
        "d_model": HIDDEN_SIZE,
    },
    training_args={
        "output_dir": tempfile.mkdtemp(prefix="patchtst_"),
        "num_train_epochs": min(EPOCHS, 10),
        "per_device_train_batch_size": BATCH_SIZE,
        "logging_strategy": "no",
    },
)
start = time.time()
forecaster.fit(y_train_sk, fh=fh_sk)
y_pred = forecaster.predict()
t = time.time() - start

m = evaluate(y_test_sk.values, y_pred.values[:HORIZON])
results.append(
    {
        "Model": "PatchTST",
        "Library": "sktime/HuggingFace",
        "Lines": "~8",
        "Train Time (s)": round(t, 1),
        **m,
        "Trainable": True,
        "Target Scale": "raw_price",
        "Evaluation Points": HORIZON,
    }
)
print(f"sktime PatchTST: {t:.1f}s, MSE={m['mse']}, IC={m['ic']}")

# %% [markdown]
# PatchTST via HuggingFace requires slightly more configuration (patch size,
# training arguments) but still follows the `fit`/`predict` pattern.

# %% [markdown]
# ## Part 5: sktime Chronos (Zero-Shot Foundation Model)
#
# Chronos (Section 13.6) requires **no training**. The `fit()` call registers
# history; all computation happens in `predict()`. This is the ultimate
# rapid-prototyping workflow: instant baseline with zero boilerplate.
#
# **Dependency note**: Chronos also uses sktime's HuggingFace backend and runs
# in the current Python 3.14 environment. The first run downloads the Chronos
# checkpoint before generating zero-shot forecasts.

# %%
forecaster = ChronosForecaster("amazon/chronos-t5-tiny")
start = time.time()
forecaster.fit(y_train_sk)
y_pred = forecaster.predict(fh=fh_sk)
t = time.time() - start

m = evaluate(y_test_sk.values, y_pred.values)
results.append(
    {
        "Model": "Chronos (tiny)",
        "Library": "sktime/HuggingFace",
        "Lines": "~3",
        "Train Time (s)": round(t, 1),
        **m,
        "Trainable": False,
        "Target Scale": "raw_price",
        "Evaluation Points": HORIZON,
    }
)
print(f"sktime Chronos: {t:.1f}s (zero-shot), MSE={m['mse']}, IC={m['ic']}")

# %% [markdown]
# Zero-shot inference eliminates training entirely. Accuracy depends on domain
# match with the pretraining corpus (see Section 13.6 for foundation model details).

# %% [markdown]
# ## Part 6: Darts LSTM (Comparison Point)
#
# Darts is a popular alternative with its own `TimeSeries` container and API.
# This provides one comparison point outside the sktime ecosystem.

# %%
# Use the reindexed business-day series to ensure freq="B" is consistent.
# Cast to float32: Darts TimeSeries default to float64, but the Lightning
# trainer's accelerator="auto" selects MPS on Apple Silicon, and MPS rejects
# float64 tensors. float32 runs identically on CPU, CUDA, and MPS.
spy_bday_df = spy_series.reset_index()
spy_bday_df.columns = ["timestamp", "close"]
ts = TimeSeries.from_dataframe(
    spy_bday_df, time_col="timestamp", value_cols="close", freq="B"
).astype(np.float32)
ts_train = ts[:sk_split]
ts_test = ts[sk_split : sk_split + HORIZON]

darts_model = RNNModel(
    model="LSTM",
    input_chunk_length=LOOKBACK,
    output_chunk_length=HORIZON,
    training_length=LOOKBACK + HORIZON,
    hidden_dim=HIDDEN_SIZE,
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    random_state=SEED,
    pl_trainer_kwargs={"enable_progress_bar": False, "accelerator": "auto"},
)
start = time.time()
darts_model.fit(ts_train)
y_pred_darts = darts_model.predict(HORIZON)
t = time.time() - start

m = evaluate(ts_test.values().flatten()[:HORIZON], y_pred_darts.values().flatten()[:HORIZON])
results.append(
    {
        "Model": "LSTM",
        "Library": "Darts",
        "Lines": "~10",
        "Train Time (s)": round(t, 1),
        **m,
        "Trainable": True,
        "Target Scale": "raw_price",
        "Evaluation Points": HORIZON,
    }
)
print(f"Darts LSTM: {t:.1f}s, MSE={m['mse']}, IC={m['ic']}")

# %% [markdown]
# Darts provides a self-contained ecosystem with its own `TimeSeries` container.
# The `random_state` parameter ensures reproducibility - a feature not all
# wrapper APIs expose.

# %% [markdown]
# ## Comparison Table
#
# All approaches use the same data (SPY close prices). The table below shows
# results from libraries that are currently runnable in this environment.
# Four of the six demos execute end-to-end today: raw PyTorch, PatchTST,
# Chronos, and Darts. The sktime NeuralForecast wrapper is blocked by the
# `ray` dependency on Python 3.14, and the PyTorch Forecasting wrapper is
# blocked by an upstream API mismatch.
#
# **Comparison caveat**: the `MSE` column is not directly comparable across
# rows. The `Target Scale` and `Evaluation Points` columns make the mismatch
# explicit: raw PyTorch fits on a *normalized* target and evaluates on a long
# rolling-window test set, while sktime PatchTST/Chronos and Darts operate on
# raw price levels and evaluate on a single `HORIZON`-step forecast. Both
# axes - scale and `n_eval` - break apples-to-apples comparison. Read the
# table as "implementation experience", not as an accuracy benchmark.

# %%
if results:
    comparison = pl.DataFrame(results)
else:
    comparison = None
    print("No results collected -- check library installations.")

comparison

# %% [markdown]
# ### Boilerplate: lines of code per approach
#
# The clearest scale-independent signal in the table is implementation effort.
# The raw-PyTorch path needs a model class plus a training loop; every wrapped
# approach is a handful of `fit`/`predict` lines. This is the notebook's thesis
# in one view (unlike the MSE column, line count is directly comparable).

# %%
if results:
    _labels = [f"{r['Model']}<br>({r['Library']})" for r in results]
    _lines = [int(str(r["Lines"]).lstrip("~")) for r in results]
    _colors = [COLORS["amber"] if "PyTorch" in r["Library"] else COLORS["blue"] for r in results]
    fig = go.Figure(
        go.Bar(x=_labels, y=_lines, marker_color=_colors, text=_lines, textposition="outside")
    )
    fig.update_layout(
        title="sktime and Darts wrappers replace ~50 lines of raw PyTorch with a handful",
        yaxis_title="Approximate lines of code",
        showlegend=False,
    )
    fig.show()

# %% [markdown]
# ## When to Use Each Approach
#
# The section text (Section 13.7) compares library *capabilities* (panel support,
# foundation models). This table focuses on *implementation experience*:
#
# | Approach | Implementation Effort | Flexibility | Best For |
# |----------|----------------------|-------------|----------|
# | **Raw PyTorch** | ~50 lines, full boilerplate | Full control over everything | Custom architectures, cross-sectional ranking, research |
# | **sktime** | ~3-8 lines | Constrained to `fit`/`predict` API | Rapid prototyping, backend swapping, standardized benchmarks |
# | **Darts** | ~10 lines | Moderate (own `TimeSeries` container) | Probabilistic forecasting, self-contained ecosystem |
# | **Chronos / TSFMs** | ~3 lines, zero training | None (zero-shot only) | Instant baselines, cold-start scenarios |
#
# The key insight: **sktime is not a model library -- it is a meta-framework**.
# It provides a single API surface (`fit`/`predict`/`update`) that delegates to
# NeuralForecast (LSTMs, TCN), PyTorch Forecasting (N-BEATS, TFT), HuggingFace
# (PatchTST, Chronos), and classical methods (ETS, ARIMA). You can benchmark
# an LSTM against Chronos against ARIMA without touching your data pipeline.

# %%
if results:
    print(f"Approaches tested: {len(results)} of 6 demos")
    trainable = [r for r in results if r["Trainable"]]
    zero_shot = [r for r in results if not r["Trainable"]]
    if trainable:
        fastest = min(trainable, key=lambda r: r["Train Time (s)"])
        print(
            f"Fastest trainable: {fastest['Model']} ({fastest['Library']}) at {fastest['Train Time (s)']}s"
        )
    if zero_shot:
        print(f"Zero-shot models: {len(zero_shot)} (no training required)")

# %% [markdown]
# ## Key Takeaways
#
# 1. **sktime provides a unified `fit()`/`predict()` API** across NeuralForecast,
#    PyTorch Forecasting, HuggingFace, and classical models
# 2. **Raw PyTorch gives full control** at the cost of ~10x more boilerplate --
#    justified for custom architectures and research
# 3. **HuggingFace-backed sktime models** (PatchTST, Chronos) run cleanly in
#    this Python 3.14 environment, while sktime's neuralforecast-backed demos
#    are still blocked by `ray`
# 4. **Zero-shot foundation models** (Chronos) eliminate training entirely,
#    providing instant baselines
# 5. **Do not compare MSE across rows unless target scale and evaluation
#    window are identical** -- the `Target Scale` and `Evaluation Points`
#    columns in the comparison table flag the mismatch. Raw PyTorch uses
#    normalized targets and a long rolling-window test; the sktime/Darts
#    rows use raw price levels and a single 5-step forecast. The scale gap
#    alone changes MSE by orders of magnitude; the `n_eval` gap changes its
#    sampling variance. Use this notebook to choose a workflow, not to rank
#    accuracy
# 6. **Start with sktime for prototyping**, drop to raw PyTorch only when you
#    need custom loss functions, architectures, or training dynamics
#
# **Next**: See `09_foundation_models` for a deeper dive into zero-shot TSFMs.
#
# **Book**: Section 13.7 discusses the practitioner framework for choosing tools.
