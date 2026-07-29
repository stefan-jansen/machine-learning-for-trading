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
# # Core Deep Learning Architectures for Time Series
#
# **Docker image**: `ml4t-gpu`
#
# This notebook compares foundational neural network architectures for return
# prediction: MLP, 1D-CNN, LSTM, and GRU. We evaluate training efficiency versus
# predictive accuracy on pooled single-ETF sequences to understand the tradeoffs
# that motivated newer architectures like N-BEATS and Transformers.
#
# **Learning Objectives**:
# - Implement MLP, CNN, LSTM, and GRU forecasting models in PyTorch
# - Compare training time versus predictive accuracy (Spearman IC)
# - Demonstrate LSTM's sequential bottleneck via sequence-length scaling
# - Establish baselines for comparison with modern architectures
#
# **Book Reference**: Chapter 13, Section 13.1 (The Recurrent Paradigm and Its Discontents).
# See Hochreiter and Schmidhuber (1997) for the original LSTM formulation.
#
# **Prerequisites**: ETF price data (via `load_etfs()` canonical loader)

# %%
"""Core Deep Learning Architectures - compare MLP, CNN, LSTM, and GRU for return prediction."""

import os

# Set before the first CUDA/cuBLAS call so deterministic GEMM kernels are available
# (see the Reproducibility note below); harmless on CPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from ml4t.diagnostic.metrics import cross_sectional_ic_series

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import add_message_title

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
SEED = 42
LOOKBACK = 60
HORIZON = 1
HIDDEN_SIZE = 64
EPOCHS = 50
BATCH_SIZE = 32
SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "USO"]
START_DATE = "2015-01-01"
LOOKBACKS = [30, 60, 120, 240]

# %% [markdown]
# ## Reproducibility
#
# GPU training is not bitwise-deterministic by default: a fixed seed controls a
# model's random choices, but not the order in which parallel CUDA kernels sum
# floating-point values, and that order can shift between runs. We enable
# PyTorch's strict deterministic mode and re-seed before each model, so trained
# parameters and accuracy metrics reproduce exactly on this machine regardless
# of training order. Torch *inference* is already deterministic; only training
# needs this. Results on a different GPU (or with this configuration removed)
# will differ slightly. Elapsed-time measurements remain sensitive to hardware
# and load, so only their broad ranking and scaling pattern carry the argument.


# %%
def seed_all(seed: int = SEED) -> None:
    """Seed Python/NumPy/Torch and re-assert strict deterministic execution."""
    set_global_seeds(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

seed_all(SEED)

# %% [markdown]
# ## Data Acquisition
#
# We use ETF close prices from the case study pipeline, converting to daily returns.
# Eight ETFs span equities, bonds, commodities, and international markets. The
# architecture comparison pools univariate sequences from all eight ETFs so the
# result is not driven by one SPY-only split.

# %%
etf_df = load_etfs()

start_dt = datetime.fromisoformat(START_DATE)

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

print(f"Data: {returns.shape[0]} days, {len(feature_cols)} ETFs")
print(f"Date range: {returns['timestamp'][0]} to {returns['timestamp'][-1]}")

# %% [markdown]
# ## Sequence Construction
#
# Each input is a lookback window of daily returns for one ETF. The target is the
# ETF's **next-day return** - a deliberately noisy single-step objective aligned with
# the direct return-prediction tasks used throughout the book. Pooling sequences from
# all eight ETFs ensures the result is not driven by one symbol's split.

# %% [markdown]
# ### Univariate sequence builder
#
# Slides a `lookback`-length window over one ETF's return series and pairs each
# window with the realised return `horizon` steps ahead.


# %%
def create_sequences(data: np.ndarray, lookback: int, horizon: int):
    """Create univariate input sequences with next-horizon return targets."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback + horizon - 1])
    return np.array(X), np.array(y)


# %% [markdown]
# ### Panel pooling and chronological split
#
# Applies the per-symbol builder to each ETF and locates the 80/20 boundary in
# calendar order. It then purges the training samples whose feature or label
# support reaches the first test window. Test dates stay fixed while the final
# `lookback + horizon - 1` candidate training samples are excluded.


# %%
def create_panel_sequences(
    returns_df: pl.DataFrame,
    symbols: list[str],
    lookback: int,
    horizon: int,
    split_fraction: float = 0.8,
):
    """Create purged chronological train/test splits, then pool symbols."""
    timestamps = returns_df["timestamp"].to_numpy()
    train_X, train_y, test_X, test_y = [], [], [], []
    test_dates_list, test_symbols_list = [], []
    for symbol in symbols:
        series = returns_df[symbol].to_numpy()
        X_symbol, y_symbol = create_sequences(series, lookback, horizon)
        split_idx = int(len(X_symbol) * split_fraction)
        purge_size = lookback + horizon - 1
        train_end = split_idx - purge_size
        if train_end <= 0:
            raise ValueError("Not enough observations for the requested purge")
        train_X.append(X_symbol[:train_end])
        train_y.append(y_symbol[:train_end])
        test_X.append(X_symbol[split_idx:])
        test_y.append(y_symbol[split_idx:])
        n_test = len(X_symbol) - split_idx
        target_offsets = np.arange(n_test) + split_idx + lookback + horizon - 1
        test_dates_list.append(timestamps[target_offsets])
        test_symbols_list.append(np.full(n_test, symbol))
    return (
        np.concatenate(train_X),
        np.concatenate(train_y),
        np.concatenate(test_X),
        np.concatenate(test_y),
        np.concatenate(test_dates_list),
        np.concatenate(test_symbols_list),
    )


# %% [markdown]
# ### Cross-sectional IC helper
#
# Computes the cross-sectional Spearman rank correlation between predicted and
# realised returns across symbols on each date, then averages over the dates where
# it is defined. `min_obs=5` keeps dates with as few as 5 ETFs out of 8 - the
# library default of 10 would discard every date in this 8-symbol panel. On a date
# where a model outputs the *same* value for every symbol, the ranks are tied and
# the correlation is undefined; `cross_sectional_ic_series` returns that as a float
# `NaN`. We exclude those dates from the average and separately report the
# **coverage** (how many dates had a defined IC), so a model that ties on many
# dates is flagged rather than hidden. This matters because polars `drop_nulls`
# does *not* drop `NaN`, so a single tied date would otherwise poison the mean.


# %%
def cross_sectional_ic(y_true, y_pred, dates, symbols):
    """Mean cross-sectional Spearman IC over defined dates, plus date coverage.

    Returns a dict with the mean IC (over dates where it is defined), and the
    counts of defined vs total dates. A date's IC is `NaN` when the model's
    predictions are tied across all symbols (zero rank variance); those dates are
    excluded from the mean (filtering both null and NaN, since polars `drop_nulls`
    leaves NaN in place).
    """
    pred_df = pl.DataFrame({"timestamp": dates, "symbol": symbols, "prediction": y_pred})
    ret_df = pl.DataFrame({"timestamp": dates, "symbol": symbols, "forward_return": y_true})
    ic_per_date = cross_sectional_ic_series(
        pred_df,
        ret_df,
        pred_col="prediction",
        ret_col="forward_return",
        date_col="timestamp",
        entity_col="symbol",
        min_obs=5,
    )
    defined = ic_per_date.filter(pl.col("ic").is_not_null() & pl.col("ic").is_not_nan())
    n_total = ic_per_date.height
    n_defined = defined.height
    mean_ic = float(defined["ic"].mean()) if n_defined else float("nan")
    return {"ic": mean_ic, "n_defined": n_defined, "n_total": n_total}


# %%
X_train, y_train, X_test, y_test, test_dates, test_symbols = create_panel_sequences(
    returns, SYMBOLS, LOOKBACK, HORIZON
)

X_train_t = torch.FloatTensor(X_train).unsqueeze(-1).to(DEVICE)
y_train_t = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
X_test_t = torch.FloatTensor(X_test).unsqueeze(-1).to(DEVICE)
y_test_t = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

print(
    f"Pooled ETF sequences: X_train {X_train.shape}, X_test {X_test.shape} "
    f"| Train: {len(X_train)}, Test: {len(X_test)} | Purge per symbol: "
    f"{LOOKBACK + HORIZON - 1}"
)

# %% [markdown]
# ## Model Definitions
#
# Four architectures with comparable parameter counts for fair comparison.
# The MLP flattens the sequence; the CNN applies same-padded temporal
# convolutions over the lookback window (no look-ahead, but the right half of
# the kernel is padded rather than masked - see `05_tcn` for a strictly causal
# variant); the LSTM and GRU process steps sequentially - the key bottleneck
# discussed in Section 13.1.

# %% [markdown]
# ### MLP Baseline
#
# Flattens the lookback window into a single vector. Fully parallel - no
# sequential dependency between time steps.


# %%
class MLPForecaster(nn.Module):
    def __init__(self, lookback: int, hidden_size: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(lookback, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# %% [markdown]
# ### 1D-CNN
#
# Applies convolutions along the time axis. Captures local patterns within the
# kernel window and processes all positions in parallel.


# %%
class CNNForecaster(nn.Module):
    def __init__(self, lookback: int, hidden_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_size // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# %% [markdown]
# ### LSTM
#
# The hidden state at time $t$ depends on $t{-}1$, creating $O(T)$ sequential
# computation. This is the bottleneck discussed in Section 13.1 - the architecture
# cannot leverage GPU parallelism across time steps.


# %%
class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# %% [markdown]
# ### GRU
#
# The Gated Recurrent Unit merges the LSTM's forget and input gates into a single
# update gate, reducing parameter count. It shares the same $O(T)$ sequential
# dependency but trains faster on smaller datasets due to fewer parameters.


# %%
class GRUForecaster(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])


# %% [markdown]
# ## Training Loop


# %%
def train_model(model, X_train, y_train, epochs, batch_size, model_name):
    """Train a model in place and return its per-epoch loss history."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    n_samples = len(X_train)
    loss_history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(n_samples, device=DEVICE)
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            optimizer.zero_grad()
            predictions = model(X_train[batch_idx])
            loss = criterion(predictions, y_train[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{model_name} Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.6f}")
    return loss_history


# %% [markdown]
# ### Measuring compute cost robustly
#
# Total training wall-clock is a poor efficiency metric on a shared GPU: a busy
# neighbour inflates it unpredictably, and on these tiny batches per-epoch time is
# dominated by Python and launch overhead rather than the architecture's compute.
# Instead we microbenchmark a single forward+backward step on a fixed batch: warm
# up to absorb allocation and kernel initialization, then report the **minimum**
# over many timed reps using CUDA events. This reduces sensitivity to transient
# delays but does not eliminate hardware or load effects. Interpret only the
# broad architecture ranking and scaling pattern.


# %%
BENCH_BATCH = 256


# %%
def benchmark_step(model, x, y, repeats: int = 100, warmup: int = 20) -> float:
    """Return the fastest observed step time; the result remains load-sensitive."""
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    def _one_step() -> None:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        _one_step()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _one_step()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return min(times)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _one_step()
        times.append((time.perf_counter() - t0) * 1000.0)
    return min(times)


# %% [markdown]
# ## Model Comparison
#
# Train all four architectures on the same data and compare efficiency versus accuracy.

# %%
# Constructors, not instances: we re-seed and build each model inside the loop so
# every architecture starts from the same seeded initialization, independent of
# the order in which the models are trained.
model_builders = {
    "MLP": lambda: MLPForecaster(LOOKBACK, HIDDEN_SIZE),
    "CNN": lambda: CNNForecaster(LOOKBACK, HIDDEN_SIZE),
    "LSTM": lambda: LSTMForecaster(HIDDEN_SIZE),
    "GRU": lambda: GRUForecaster(HIDDEN_SIZE),
}

results = {}


# %% [markdown]
# ### Fit and evaluate each architecture

# %%
for name, build in model_builders.items():
    seed_all(SEED)
    model = build()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining {name} ({n_params:,} parameters)...")

    loss_history = train_model(model, X_train_t, y_train_t, EPOCHS, BATCH_SIZE, name)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy().flatten()
        actuals = y_test_t.cpu().numpy().flatten()

    mse = float(np.mean((predictions - actuals) ** 2))
    ic_info = cross_sectional_ic(actuals, predictions, test_dates, test_symbols)

    results[name] = {
        "mse": mse,
        "ic": ic_info["ic"],
        "ic_coverage": (ic_info["n_defined"], ic_info["n_total"]),
        "n_params": n_params,
        "loss_history": loss_history,
    }

# %% [markdown]
# ### Benchmark observed step cost
#
# Timing runs after every accuracy model is fitted, so measurement cannot alter a
# later model's GPU execution path. Each benchmark uses a fresh seeded model.

# %%
for name, build in model_builders.items():
    seed_all(SEED)
    bench_model = build()
    n_bench = min(BENCH_BATCH, len(X_train_t))
    results[name]["step_ms"] = benchmark_step(bench_model, X_train_t[:n_bench], y_train_t[:n_bench])

# %% [markdown]
# ## Results Summary

# %%
comparison_df = pl.DataFrame(
    [
        {
            "Model": name,
            "Parameters": r["n_params"],
            "Step time (ms)": round(r["step_ms"], 3),
            "Test MSE": round(r["mse"], 6),
            # IC is undefined on dates where predictions tie across symbols; if a
            # model has no defined date at all, show "undefined" not a bare NaN.
            "Spearman IC": "undefined" if np.isnan(r["ic"]) else f"{r['ic']:.4f}",
            # Dates where the IC was defined vs total - low coverage means the
            # model ties its cross-section often and the IC is on a biased subset.
            "IC coverage": f"{r['ic_coverage'][0]}/{r['ic_coverage'][1]}",
        }
        for name, r in results.items()
    ]
)
comparison_df

# %% [markdown]
# **Interpretation**: On this single split the point-forecast error (Test MSE) is
# nearly identical across the four architectures, and every cross-sectional
# Spearman IC is small - next-day returns are essentially unpredictable at this
# horizon, so a model that minimises MSE does so mostly by predicting close to the
# mean. The estimates vary around zero, and the **IC coverage** column adds an
# important qualification: the CNN emits the *same* value for all eight ETFs on
# some test dates, so its IC is defined on only a subset, while the MLP, LSTM, and
# GRU rank the cross-section on every date. None of these single-split ICs is a
# basis for selection. The informative and robust difference is therefore in
# **efficiency**, not accuracy: recurrent models (LSTM, GRU) carry more parameters
# and update their hidden state sequentially, while the MLP and CNN process the
# window in parallel. Treat this as an efficiency demonstration, not model
# selection for a production strategy.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Training loss curves
for name, r in results.items():
    axes[0].plot(r["loss_history"], label=name)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE loss")
add_message_title(
    axes[0],
    "MLP keeps cutting training loss",
    subtitle="recurrent and CNN losses plateau early (MSE, 8-ETF next-day returns)",
)
axes[0].legend()

# Right: Per-step compute vs. test MSE scatter. MSE (the training objective) is
# always defined, whereas the cross-sectional IC is undefined on dates where a
# model's predictions tie across symbols - so MSE gives the robust
# efficiency-vs-error view here. The x-axis is the microbenchmarked per-step time
# (lower variance than total wall-clock, but still hardware/load-sensitive).
for name, r in results.items():
    axes[1].scatter(r["step_ms"], r["mse"], s=100, zorder=5)
    axes[1].annotate(
        name,
        (r["step_ms"], r["mse"]),
        textcoords="offset points",
        xytext=(0, -14 if name == "LSTM" else 10),
        ha="center",
    )
axes[1].set_xlabel("Per-step time (ms)")
axes[1].set_ylabel("Test MSE")
add_message_title(
    axes[1],
    "Similar error, very different cost",
    subtitle="Test MSE vs per-step forward+backward time (RTX 3090)",
)

fig.tight_layout()
fig.show()

# %% [markdown]
# **Finding**: The four architectures reach nearly the same test error at very
# different per-step compute costs - the recurrent models buy no measurable
# accuracy for their heavier step. On an essentially unpredictable next-day target
# that is the expected outcome; the lesson is that sequential computation is a cost
# to be justified, which motivates the search for architectures that match this
# accuracy with parallel computation.

# %% [markdown]
# ## Sequence-Length Scaling
#
# The LSTM's $O(T)$ sequential dependency means per-step compute grows with
# sequence length, while the MLP processes any length in parallel (flattened).
# This experiment trains both at increasing lookback windows (for the IC panel) and
# microbenchmarks a single forward+backward step at each setting. Taking the
# minimum reduces transient delays but does not make timing load-invariant, so the
# broad scaling pattern matters more than exact milliseconds. This is the core
# limitation from Section 13.1.

# %%
lookbacks = LOOKBACKS

scaling_results = {}
scaling_batches = {}

for lb in lookbacks:
    X_tr_np, y_tr_np, X_te_np, y_te_np, test_dates_lb, test_symbols_lb = create_panel_sequences(
        returns, SYMBOLS, lb, HORIZON
    )

    X_tr = torch.FloatTensor(X_tr_np).unsqueeze(-1).to(DEVICE)
    y_tr = torch.FloatTensor(y_tr_np).unsqueeze(-1).to(DEVICE)
    X_te = torch.FloatTensor(X_te_np).unsqueeze(-1).to(DEVICE)
    y_te = torch.FloatTensor(y_te_np).unsqueeze(-1).to(DEVICE)
    scaling_batches[lb] = (X_tr, y_tr)

    for arch_name, ModelCls, kwargs in [
        ("MLP", MLPForecaster, {"lookback": lb, "hidden_size": HIDDEN_SIZE}),
        ("LSTM", LSTMForecaster, {"hidden_size": HIDDEN_SIZE}),
    ]:
        seed_all(SEED)
        model = ModelCls(**kwargs)
        train_model(model, X_tr, y_tr, EPOCHS, BATCH_SIZE, f"{arch_name}-{lb}")

        model.eval()
        with torch.no_grad():
            preds = model(X_te).cpu().numpy().flatten()
            acts = y_te.cpu().numpy().flatten()
        ic_val = cross_sectional_ic(acts, preds, test_dates_lb, test_symbols_lb)["ic"]
        scaling_results[(arch_name, lb)] = {"ic": ic_val}

# %% [markdown]
# ### Benchmark each sequence shape
#
# Fresh models keep the timing pass separate from the accuracy fits above.

# %%
for lb in lookbacks:
    X_tr, y_tr = scaling_batches[lb]
    for arch_name, ModelCls, kwargs in [
        ("MLP", MLPForecaster, {"lookback": lb, "hidden_size": HIDDEN_SIZE}),
        ("LSTM", LSTMForecaster, {"hidden_size": HIDDEN_SIZE}),
    ]:
        seed_all(SEED)
        model = ModelCls(**kwargs)
        n_bench = min(BENCH_BATCH, len(X_tr))
        scaling_results[(arch_name, lb)]["time"] = benchmark_step(
            model, X_tr[:n_bench], y_tr[:n_bench]
        )

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for arch in ["MLP", "LSTM"]:
    lbs = [lb for lb in lookbacks if (arch, lb) in scaling_results]
    times = [scaling_results[(arch, lb)]["time"] for lb in lbs]
    ics = [scaling_results[(arch, lb)]["ic"] for lb in lbs]

    axes[0].plot(lbs, times, "o-", label=arch)
    axes[1].plot(lbs, ics, "o-", label=arch)

axes[0].set_xlabel("Lookback window (days)")
axes[0].set_ylabel("Per-step time (ms)")
add_message_title(
    axes[0],
    "LSTM time grows with lookback; MLP remains lower",
    subtitle="Min per-step forward+backward time vs lookback (RTX 3090)",
)
axes[0].legend()

axes[1].set_xlabel("Lookback window (days)")
axes[1].set_ylabel("Spearman IC")
add_message_title(
    axes[1],
    "Predictive accuracy versus lookback window",
    subtitle="Spearman IC, pooled 8-ETF next-day returns",
)
axes[1].legend()

fig.tight_layout()
fig.show()

# %% [markdown]
# **Finding**: LSTM per-step time rises more quickly with lookback length because
# its hidden state is updated one step at a time - the $O(T)$ recurrence. The MLP is
# less sequentially constrained: it flattens longer windows into a larger input
# vector, so its growth comes from a bigger first-layer matmul rather than from
# step-by-step recurrence. The IC plot is noisy, reinforcing that this notebook is
# an architecture demonstration rather than a validated trading model.
#
# **Limitation**: This uses a single 80/20 temporal split, not walk-forward
# validation. Section 13.7 covers proper evaluation methodology.

# %% [markdown]
# ## sktime Alternative
#
# High-level APIs like sktime wrap these architectures in a unified
# `fit()`/`predict()` interface, reducing the training loop above to a few lines.
# This is useful for rapid prototyping before committing to a custom implementation.
#
# **Dependency note**: sktime's neural forecasters require `neuralforecast`, which
# depends on `ray` - and ray does not yet support Python 3.14
# ([ray-project/ray#56434](https://github.com/ray-project/ray/issues/56434)).
# Once ray adds 3.14 wheels, run `uv pip install neuralforecast` and uncomment
# the sktime demo cell below.

# %%
# import pandas as pd
# from sktime.forecasting.neuralforecast import NeuralForecastLSTM
#
# symbol = SYMBOLS[0]
# spy_prices = close_wide.select(["timestamp", symbol]).to_pandas()
# spy_prices = spy_prices.set_index("timestamp")[symbol].dropna()
#
# split = int(len(spy_prices) * 0.8)
# y_train_sk = spy_prices.iloc[:split]
# y_test_sk = spy_prices.iloc[split : split + HORIZON]
#
# forecaster = NeuralForecastLSTM(
#     freq="B",
#     input_size=LOOKBACK,
#     max_steps=EPOCHS * 5,
#     encoder_hidden_size=HIDDEN_SIZE,
# )
# forecaster.fit(y_train_sk)
# y_pred_sk = forecaster.predict(fh=list(range(1, HORIZON + 1)))
#
# mae_sk = float(np.mean(np.abs(y_test_sk.values - y_pred_sk.values)))
# print(f"sktime NeuralForecastLSTM MAE: {mae_sk:.4f}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **The objective is deliberately simple and noisy**: pooled univariate ETF
#    sequences predict next-day returns, so every cross-sectional IC is small and
#    within noise of zero - and a model whose predictions tie across symbols on a
#    date has no defined IC there (watch the IC-coverage column). Accuracy is not
#    what separates these models here.
# 2. **Parallel architectures train efficiently** because the MLP and CNN can process
#    all time steps simultaneously, while recurrent models update hidden state
#    sequentially.
# 3. **IC is not a basis for model selection** on this single illustrative split.
#    Case-study notebooks use walk-forward validation and richer feature sets for
#    publication-grade comparisons.
# 4. **Sequence-length scaling** confirms the $O(T)$ bottleneck: the LSTM's per-step
#    time grows steeply with lookback, while the MLP is less sequentially
#    constrained (its growth comes from a larger first-layer matmul, not recurrence).
# 5. These tradeoffs motivated N-BEATS (parallelizable decomposition) and Transformers
#    (parallelizable attention), explored in subsequent notebooks.
#
# **Next**: See `02_nbeats_interpretable` for the N-BEATS decomposition approach.
