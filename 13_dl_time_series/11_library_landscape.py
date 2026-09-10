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
# # The same forecast, written six ways
#
# **Docker image**: `ml4t-gpu`
#
# Every notebook in this chapter has been written directly against PyTorch: a model
# class, a training loop, an evaluation block. That is the right way to learn an
# architecture and a poor way to try five of them. This notebook writes the same
# univariate forecast through several libraries and compares the *experience* -
# how many lines, what has to be installed, what the interface constrains.
#
# **sktime** is the one to understand, because it is not a model library. It is an
# interface - `fit`, `predict`, `update` - that delegates to NeuralForecast,
# PyTorch Forecasting, HuggingFace and classical methods, so swapping an LSTM for
# Chronos or for ARIMA is a constructor change rather than a rewrite. **Darts**
# takes the other approach: its own `TimeSeries` container and its own model
# implementations.
#
# **What the table measures is effort, not accuracy.** The rows do not share a target
# scale or a test set, so the error column cannot be read across them - which is
# stated where the table appears and marked in the table's own columns. A last-value
# baseline is included precisely so that the one comparison that *is* like-for-like
# is on the page.
#
# **Learning objectives**:
# - Write the same forecast against a raw framework and against a wrapper, and count
#   what the wrapper actually saved.
# - Read a comparison table and identify which columns can be compared across rows
#   and which cannot.
# - Say what a rank correlation means when the target is a price level rather than a
#   return.
# - Decide when the interface constraint is worth the boilerplate it removes.
#
# **Book Reference**: Chapter 13, Section 13.7 (A practical framework)
#
# **Prerequisites**: ETF price data via the canonical `load_etfs()` loader

# %%
"""Library Landscape - compare raw PyTorch vs sktime-wrapped forecasting implementations."""

import logging
import tempfile
import time
import warnings
from datetime import datetime

# pytorch_lightning emits this at import time, which `from darts...` below triggers, so
# the filter has to be installed before the import rather than alongside the others.
warnings.filterwarnings(
    "ignore",
    message=r".*LeafSpec.*is deprecated",
    module=r".*pytorch_lightning.*",
)

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

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import (  # COLORS activates the ml4t Plotly template on import
    COLORS,
    show_plotly_with_alt,
)

# %% [markdown]
# Three of the libraries below write progress banners to standard error - the accelerator
# they picked, a suggestion to install a logging integration, a deprecation inside
# `transformers` reached through sktime's Chronos wrapper. None of it reports a problem
# and none of it is this notebook's output, so each is quieted by name at its own logger
# rather than by a blanket filter. Anything these libraries raise as a warning still
# reaches the render.

# %%
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

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
n_seq = len(prices) - LOOKBACK - HORIZON + 1
split_idx = int(n_seq * 0.8)
train_price_end = split_idx + LOOKBACK + HORIZON
price_mean = prices[:train_price_end].mean()
price_std = prices[:train_price_end].std()
prices_norm = (prices - price_mean) / price_std

print(f"SPY: {len(prices):,} daily observations")

# %% [markdown]
# The order of the two steps above matters. The train/test boundary is fixed in
# sequence space *first*, and the normalisation statistics are then computed from
# prices up to `train_price_end` only - the last price index any training sequence can
# reach, its input window plus the forecast horizon. Normalising with the full
# series' mean and standard deviation would fold test-period price levels into the
# training inputs, which is a leak that leaves no trace in any split.

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
# **The IC here is not the IC in the rest of the chapter.** Notebooks 04 through 10
# compute a *cross-sectional* rank correlation: on each date, do the predicted ranks
# of the funds match the realised ranks? There is one series here, so there is no
# cross-section, and what this helper computes is the rank correlation between
# predicted and actual values across time.
#
# On a price level that number is close to one for anything that tracks the level,
# including a forecast that just repeats the last observation - the baseline below
# demonstrates it. Read the `ic` column as a check that a model is not producing
# nonsense, and read MSE for anything else.


# %%
def evaluate(y_true, y_pred):
    """Compute MSE and Spearman IC."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    ic = pooled_ic(y_pred, y_true) if len(y_true) > 5 else float("nan")
    return {"mse": round(mse, 6), "ic": round(ic, 4) if np.isfinite(ic) else None}


results = []  # Collector for final comparison

# %% [markdown]
# ### The forecast to beat
#
# Before any library: forecast the next `HORIZON`-day mean as the last observed
# price. It costs one line, it is what every model below has to improve on, and on a
# price level it is hard to beat - which is exactly why it belongs in the table.
#
# The IC column needs the same warning. On a trending price level, any forecast that
# tracks the level correlates with the outcome almost perfectly, so a rank correlation
# near one is a statement about the series, not about the model. The baseline row is
# what makes that visible.

# %%
preds_naive = X_test[:, -1]
m_naive = evaluate(y_test, preds_naive)
results.append(
    {
        "Model": "Last value",
        "Library": "None",
        "Lines": "~1",
        "Fit + Predict (s)": 0.0,
        **m_naive,
        "Trainable": False,
        "Target Scale": "normalized",
        "Evaluation Points": len(y_test),
    }
)
print(f"Last-value baseline: MSE={m_naive['mse']}, IC={m_naive['ic']}")

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
model.eval()
with torch.no_grad():
    preds_pt = model(X_test_t).cpu().numpy().flatten()
pytorch_time = time.time() - start

m = evaluate(y_test, preds_pt)
results.append(
    {
        "Model": "LSTM",
        "Library": "Raw PyTorch",
        "Lines": "~50",
        "Fit + Predict (s)": round(pytorch_time, 1),
        **m,
        "Trainable": True,
        "Target Scale": "normalized",
        "Evaluation Points": len(y_test),
    }
)
print(f"Raw PyTorch LSTM: {pytorch_time:.1f}s, MSE={m['mse']}, IC={m['ic']}")
print(
    f"  against the last-value baseline: MSE={m_naive['mse']}, IC={m_naive['ic']} "
    f"on the same {len(y_test)} points"
)

# %% [markdown]
# The raw PyTorch LSTM requires defining the model class, training loop, and
# evaluation - approximately 50 lines. Full control over architecture and training
# dynamics, at the cost of that boilerplate.
#
# Compare the two lines printed above before reading anything else in this notebook as
# a result. Both are on the same scale and the same test points, which is the only
# such comparison here, and the baseline needed no training at all.

# %% [markdown]
# ## Part 2: sktime + NeuralForecast LSTM
#
# sktime wraps NeuralForecast's LSTM: three lines replace the 50-line training
# loop. The underlying recurrent architecture is equivalent.
#
# **Dependency note**: sktime's neural forecasters require `neuralforecast`, which
# depends on `ray`, and Python 3.14 wheels for ray are still pending
# ([ray-project/ray#56434](https://github.com/ray-project/ray/issues/56434)). This
# section describes the call rather than running it. Note also that recent sktime
# releases renamed the wrapper's `hidden_size` argument to `encoder_hidden_size`.

# %% [markdown]
# The shape of the call, when it can be made, is
# `NeuralForecastLSTM(freq=..., input_size=..., max_steps=..., encoder_hidden_size=...)`
# followed by `fit` and `predict` - the whole of Part 1's model class and training
# loop replaced by a constructor. That is the argument for the wrapper, and this
# environment is also the argument against depending on one: the code above cannot
# run here at all.

# %% [markdown]
# ## Part 3: sktime + PyTorch Forecasting N-BEATS
#
# N-BEATS (Section 13.2) via sktime's PyTorch Forecasting wrapper. Same
# `fit()`/`predict()` API, different architecture underneath.
#
# **Dependency note**: sktime's PyTorch Forecasting wrapper requires
# `pytorch-forecasting`, and the last attempt also hit a wrapper API mismatch around
# `max_prediction_length` and `max_encoder_length`. This section describes the call
# rather than running it.

# %% [markdown]
# The call would be `PytorchForecastingNBeats(max_prediction_length=...,
# max_encoder_length=..., max_epochs=..., trainer_kwargs=...)`, then the same `fit`
# and `predict` as every other forecaster here - which is the point of the interface:
# swapping an LSTM for the N-BEATS of `02_nbeats_interpretable` changes an import and
# a constructor and nothing about the data pipeline.
#
# It is also the second wrapper in a row that this notebook cannot execute, which is
# information about wrappers rather than an aside.

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
        "Fit + Predict (s)": round(t, 1),
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
        "Fit + Predict (s)": round(t, 1),
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

# %% [markdown]
# Two details of the Darts container are worth stating rather than leaving in the
# code. It needs the reindexed business-day series so its declared frequency is
# consistent, and its `TimeSeries` defaults to float64 - which the Lightning trainer's
# `accelerator="auto"` cannot use when it selects MPS on Apple Silicon, since MPS
# rejects float64 tensors. Casting to float32 runs identically on CPU, CUDA and MPS.

# %%
spy_bday_df = spy_series.reset_index()
spy_bday_df.columns = ["timestamp", "close"]
ts = TimeSeries.from_dataframe(
    spy_bday_df, time_col="timestamp", value_cols="close", freq="B"
).astype(np.float32)
ts_train = ts[:sk_split]
ts_test = ts[sk_split : sk_split + HORIZON]

# Re-asserted here rather than only at import: something between the import cell and this
# one puts the LeafSpec deprecation back, and a filter has to be in force where the
# warning is raised, not merely where it was first installed.
warnings.filterwarnings(
    "ignore",
    message=r".*LeafSpec.*is deprecated",
    module=r".*pytorch_lightning.*",
)

# No output_chunk_length: Darts' RNNModel forecasts one step and rolls it forward, so it
# overrides any value passed here. `predict(HORIZON)` still returns HORIZON steps.
darts_model = RNNModel(
    model="LSTM",
    input_chunk_length=LOOKBACK,
    training_length=LOOKBACK + HORIZON,
    hidden_dim=HIDDEN_SIZE,
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    random_state=SEED,
    pl_trainer_kwargs={
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "accelerator": "auto",
    },
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
        "Fit + Predict (s)": round(t, 1),
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
# Every row starts from the same data, SPY close prices, and the table collects
# whichever approaches ran. Of the six library demos attempted, the ones that
# complete in this environment are raw PyTorch, sktime's PatchTST and Chronos, and
# Darts; sktime's NeuralForecast wrapper is blocked by `ray` on Python 3.14 and its
# PyTorch Forecasting wrapper by an upstream API mismatch. The last-value baseline
# needs no library at all.
#
# **Read `Lines` and `Fit + Predict (s)` across rows. Do not read `mse` across rows.**
# The `Target Scale` and `Evaluation Points` columns say why: raw PyTorch and the
# baseline fit a *normalized* target and are scored over hundreds of rolling windows,
# while sktime and Darts work on raw price levels and are scored on a single
# `HORIZON`-step forecast. The scale gap alone moves the number by orders of
# magnitude, and five evaluation points would not support a comparison even if the
# scales matched.
#
# The one pair that *is* comparable is the baseline against raw PyTorch: same scale,
# same test points. That comparison is the only accuracy statement this notebook
# supports, and it is worth reading before the rest of the table.

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
# Implementation effort is what this table measures cleanly. The raw-PyTorch path
# needs a model class and a training loop; each wrapper is a constructor plus `fit`
# and `predict`. Line counts are approximate, but they are counted the same way for
# every row, which is more than the error column can say.

# %%
if results:
    _labels = [f"{r['Model']}<br>({r['Library']})" for r in results]
    _lines = [int(str(r["Lines"]).lstrip("~")) for r in results]
    _colors = [
        COLORS["slate"]
        if r["Library"] == "None"
        else COLORS["amber"]
        if "PyTorch" in r["Library"]
        else COLORS["blue"]
        for r in results
    ]
    fig = go.Figure(
        go.Bar(x=_labels, y=_lines, marker_color=_colors, text=_lines, textposition="outside")
    )
    fig.update_layout(
        title="Approximate lines of code per approach",
        yaxis_title="Approximate lines of code",
        showlegend=False,
    )
    show_plotly_with_alt(
        fig,
        "A bar chart of approximate lines of code, one bar per approach, labelled with "
        "the model and the library. The raw PyTorch bar is amber, the library-wrapped "
        "bars navy, and the no-library baseline slate.",
    )

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
# sktime is not a model library. It is a single interface - `fit`, `predict`,
# `update` - delegating to NeuralForecast (LSTMs, TCN), PyTorch Forecasting (N-BEATS,
# TFT), HuggingFace (PatchTST, Chronos) and classical methods (ETS, ARIMA), so an
# LSTM, a foundation model and an ARIMA can be benchmarked without touching the data
# pipeline. What this notebook also shows is the cost of that indirection: two of the
# backends could not be installed or called here, and the interface does not help
# with a backend that is not there.

# %%
if results:
    _libraries = [r for r in results if r["Library"] != "None"]
    print(f"Library approaches that ran: {len(_libraries)} of 6 attempted")
    trainable = [r for r in _libraries if r["Trainable"]]
    zero_shot = [r for r in _libraries if not r["Trainable"]]
    if trainable:
        fastest = min(trainable, key=lambda r: r["Fit + Predict (s)"])
        print(
            f"Fastest to fit and predict: {fastest['Model']} ({fastest['Library']}) "
            f"at {fastest['Fit + Predict (s)']}s"
        )
    if zero_shot:
        print(f"Zero-shot models: {len(zero_shot)} (no training required)")

# %% [markdown]
# ## Key takeaways
#
# 1. **Two of the table's columns can be compared across rows, and the error column
#    cannot.** Lines of code is one; the other is wall-clock time, which is measured
#    over the same span for every row - the fit and the prediction together - because
#    that is the only boundary all of them share. Chronos does no fitting at all, so
#    separating the two would leave its cell empty and the column unreadable. Error is
#    the one that cannot be compared: the raw-PyTorch row is a normalized target scored
#    over a long rolling test set, the sktime and Darts rows are raw price levels
#    scored on a single `HORIZON`-step forecast. The scale gap moves the number by
#    orders of magnitude and the sample gap moves its variance. `Target Scale` and
#    `Evaluation Points` are in the table so the mismatch cannot be missed.
# 2. **The last-value baseline is the row to read first.** It shares its scale and its
#    test points with the raw-PyTorch row, so those two are genuinely comparable - and
#    it needed no training. On a price level, that is a demanding thing to beat.
# 3. **A rank correlation on a price level is not a measure of forecasting skill.**
#    Any forecast that tracks the level scores near one, including the baseline. That
#    is why this notebook's `ic` column is a diagnostic and the multi-asset notebooks
#    compute a *cross-sectional* IC instead, which asks a question the level cannot
#    answer for you.
# 4. **sktime is an interface, not a model library.** One `fit`/`predict` surface
#    delegating to NeuralForecast, PyTorch Forecasting, HuggingFace and classical
#    methods is what lets an LSTM, Chronos and ARIMA be benchmarked without touching
#    the data pipeline. Darts makes the opposite trade: its own container and its own
#    models, self-contained rather than delegating.
# 5. **What a wrapper costs is the thing it does not expose.** A custom loss, a
#    non-standard training schedule or a cross-sectional target is where the
#    `fit`/`predict` surface stops helping, which is the case the rest of this chapter
#    is written in raw PyTorch for.
# 6. **Installability is part of the comparison.** Not every wrapper ran: the
#    printed count says how many of the attempted demos completed in this environment,
#    and the sections above say what blocked each one.
#
# **Next**: `12_case_study_insights` leaves single-series demonstrations behind and
# aggregates what these architectures did across the book's case studies under
# walk-forward validation.
