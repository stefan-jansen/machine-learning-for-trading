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
# # Single Asset Backtest with VectorBT
#
# **Docker image**: `ml4t`
#
# This notebook introduces **VectorBT** - a vectorized backtesting library that enables
# fast simulation of trading strategies. We start with a simple single-asset RSI mean-reversion
# strategy on Bitcoin before moving to multi-asset portfolio strategies.
#
# **Learning Objectives:**
# 1. Understand VectorBT's core API and portfolio simulation
# 2. Implement entry/exit signals from technical indicators
# 3. Analyze performance metrics and visualizations
# 4. Compare strategy vs buy-and-hold benchmark
#
# **Book Reference:** Chapter 16, Section 16.3 — vectorized and event-driven engines.
#
# **Prerequisites:** Ch16 NB 01 (backtesting first principles).
#
# **Strategy:** RSI Mean Reversion on BTC
# - **Long Entry**: RSI < 30 (oversold)
# - **Exit**: RSI > 70 (overbought)
# - **Rebalancing**: Daily
# - **Transaction Costs**: 10 bps

# %% [markdown]
# ## Setup

# %%
"""Single Asset Backtest with VectorBT — RSI mean-reversion strategy using vectorized backtesting."""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import polars as pl
import vectorbt as vbt
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from plotly.subplots import make_subplots

from data import load_crypto_perps

vbt.settings["plotting"]["use_widgets"] = False


# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
INITIAL_CASH = 100_000

# %% [markdown]
# ## 1. Data Acquisition
#
# Load BTC/USDT daily bars from the local crypto perpetuals dataset and
# convert to a pandas Series indexed by timestamp — VectorBT expects pandas.

# %%
_crypto = load_crypto_perps()
btc_df = (
    _crypto.filter(
        (pl.col("symbol") == "BTCUSDT")
        & (pl.col("timestamp") >= pl.lit(START_DATE).str.to_datetime().dt.replace_time_zone("UTC"))
        & (pl.col("timestamp") < pl.lit(END_DATE).str.to_datetime().dt.replace_time_zone("UTC"))
    )
    .sort("timestamp")
    .with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    .group_by_dynamic("timestamp", every="1d")
    .agg(
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    )
)

# Convert to pandas Series for VectorBT
btc_pd = btc_df.to_pandas()
btc_pd.set_index("timestamp", inplace=True)
close = btc_pd["close"]

print(f"Loaded {len(btc_df):,} daily bars for BTCUSDT from the crypto perpetuals dataset")
print(f"Date range: {btc_df['timestamp'].min()} to {btc_df['timestamp'].max()}")

# %%
# Preview the data
close.head(10)

# %% [markdown]
# ## 2. RSI Indicator Calculation
#
# VectorBT provides optimized indicator calculations. The RSI indicator
# can be computed for multiple parameter values simultaneously.

# %%
# Single RSI calculation with standard 14-day period
rsi = vbt.RSI.run(close, window=14)

# %%
# Preview RSI values
rsi_series = rsi.rsi
print("RSI Statistics:")
print(f"  Mean: {rsi_series.mean():.1f}")
print(f"  Std:  {rsi_series.std():.1f}")
print(f"  Min:  {rsi_series.min():.1f}")
print(f"  Max:  {rsi_series.max():.1f}")

# %%
# Visualize price and RSI
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3],
    subplot_titles=["BTC/USDT Price", "RSI (14)"],
)

fig.add_trace(go.Scatter(x=close.index, y=close, name="BTC", line=dict(color="blue")), row=1, col=1)

fig.add_trace(
    go.Scatter(x=rsi_series.index, y=rsi_series, name="RSI", line=dict(color="purple")),
    row=2,
    col=1,
)

# Add RSI threshold lines
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)

fig.update_layout(
    height=600,
    title="BTC Price and RSI Indicator",
    showlegend=True,
    xaxis2_title="Date",
    yaxis_title="Price (USDT)",
    yaxis2_title="RSI",
)
fig.show()

# %% [markdown]
# ## 3. Generate Trading Signals
#
# RSI Mean Reversion Logic:
# - **Enter Long**: RSI falls below 30 (oversold → expect bounce)
# - **Exit Long**: RSI rises above 70 (overbought → take profit)

# %%
# Define entry and exit signals
rsi_lower = 30  # Oversold threshold
rsi_upper = 70  # Overbought threshold

# Entry: RSI < lower threshold
entries = rsi_series < rsi_lower

# Exit: RSI > upper threshold
exits = rsi_series > rsi_upper

# %%
# Count signals
print(f"Entry signals: {entries.sum()}")
print(f"Exit signals: {exits.sum()}")

# %% [markdown]
# ## 4. Run Backtest with VectorBT
#
# VectorBT's `Portfolio.from_signals()` is the core backtesting function.
# It simulates a portfolio based on entry/exit signals.

# %%
# Backtest cost parameters
FEES = 0.001  # 10 bps per trade
SLIPPAGE = 0.0005  # 5 bps slippage

# %%
# Run backtest
portfolio = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    size=0.95,  # 95% of capital per trade
    size_type="percent",
    init_cash=INITIAL_CASH,
    fees=FEES,
    slippage=SLIPPAGE,
    freq="1D",
)

# %% [markdown]
# **Portfolio performance summary** (VectorBT's built-in stats):

# %%
portfolio.stats()

# %% [markdown]
# The RSI mean-reversion strategy loses money on BTC — a strongly trending
# asset over 2020–2023. This is expected: mean reversion buys dips that
# continue falling and exits before the major rallies. The low Sharpe (< 0.1)
# and negative expectancy confirm that simple mean reversion is a poor fit
# for crypto's trending regime. This deliberately illustrates why strategy-data
# fit matters more than indicator sophistication.

# %% [markdown]
# ## 5. Analyze Trade Statistics

# %%
# Extract trade details
trades = portfolio.trades.records_readable
print(f"Total trades: {len(trades)}")

# %% [markdown]
# **Trade statistics** (per-trade entry/exit, P&L, holding period summary):

# %%
portfolio.trades.stats()

# %% [markdown]
# **First 10 individual trades** with entry/exit timing and realized P&L:

# %%
display_cols = [
    "Entry Timestamp",
    "Exit Timestamp",
    "Size",
    "Entry Price",
    "Exit Price",
    "PnL",
    "Return",
]
available_cols = [c for c in display_cols if c in trades.columns]
trades[available_cols].head(10)

# %% [markdown]
# ## 6. Performance Visualization

# %%
# Plot portfolio value over time
fig = portfolio.plot()
fig.update_layout(title="RSI Mean Reversion Strategy - Portfolio Value", height=500)
fig.show()

# %%
# Plot drawdown
fig_dd = portfolio.drawdowns.plot()
fig_dd.update_layout(title="Underwater Curve (Drawdowns)", height=400)
fig_dd.show()

# %% [markdown]
# ## 7. Compare to Buy-and-Hold Benchmark
#
# A strategy should beat buy-and-hold to be worthwhile. Let's compare.

# %%
# Create buy-and-hold benchmark
bh_portfolio = vbt.Portfolio.from_holding(close=close, init_cash=INITIAL_CASH, fees=FEES, freq="1D")

# %%
# Compare key metrics
comparison = pd.DataFrame(
    {
        "RSI Strategy": [
            portfolio.total_return() * 100,
            portfolio.sharpe_ratio(),
            portfolio.max_drawdown() * 100,
            portfolio.trades.count(),
            portfolio.sortino_ratio(),
            portfolio.calmar_ratio(),
        ],
        "Buy & Hold": [
            bh_portfolio.total_return() * 100,
            bh_portfolio.sharpe_ratio(),
            bh_portfolio.max_drawdown() * 100,
            0,  # No trades
            bh_portfolio.sortino_ratio(),
            bh_portfolio.calmar_ratio(),
        ],
    },
    index=[
        "Total Return (%)",
        "Sharpe Ratio",
        "Max Drawdown (%)",
        "Total Trades",
        "Sortino Ratio",
        "Calmar Ratio",
    ],
)

# %% [markdown]
# **Strategy vs buy-and-hold comparison** (annualized metrics, daily frequency):

# %%
comparison.round(3)

# %% [markdown]
# Buy-and-hold dominates on return and Sharpe by a wide margin. The strategy's
# only edge is a less severe maximum drawdown (about ten percentage points
# shallower) — its exposure management cuts into the worst losses but cannot
# compensate for missing the major rallies. This comparison is the minimum
# bar any active strategy must clear.

# %%
# Plot cumulative returns comparison
fig = go.Figure()

strategy_cum = portfolio.cumulative_returns() * 100
bh_cum = bh_portfolio.cumulative_returns() * 100

fig.add_trace(
    go.Scatter(x=strategy_cum.index, y=strategy_cum, name="RSI Strategy", line=dict(color="blue"))
)

fig.add_trace(
    go.Scatter(x=bh_cum.index, y=bh_cum, name="Buy & Hold", line=dict(color="gray", dash="dash"))
)

fig.update_layout(
    title="Cumulative Returns: Strategy vs Buy-and-Hold",
    xaxis_title="Date",
    yaxis_title="Cumulative Return (%)",
    height=500,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)
fig.show()

# %% [markdown]
# ## 8. Parameter Optimization
#
# VectorBT excels at parameter sweeps. Let's optimize RSI thresholds.

# %%
# Define parameter ranges
rsi_windows = [7, 14, 21]
lower_thresholds = [20, 25, 30, 35]
upper_thresholds = [65, 70, 75, 80]

# %%
# Run RSI for all window sizes
rsi_multi = vbt.RSI.run(close, window=rsi_windows)

# %%
# Create parameter grid for signals
results = []

for window in rsi_windows:
    rsi_vals = vbt.RSI.run(close, window=window).rsi

    for lower in lower_thresholds:
        for upper in upper_thresholds:
            if lower >= upper:
                continue

            entries = rsi_vals < lower
            exits = rsi_vals > upper

            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                size=0.95,
                size_type="percent",
                init_cash=INITIAL_CASH,
                fees=FEES,
                slippage=SLIPPAGE,
                freq="1D",
            )

            results.append(
                {
                    "window": window,
                    "lower": lower,
                    "upper": upper,
                    "total_return": pf.total_return() * 100,
                    "sharpe": pf.sharpe_ratio(),
                    "max_dd": pf.max_drawdown() * 100,
                    "trades": pf.trades.count(),
                }
            )

# %%
# Convert to DataFrame and find best parameters
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("sharpe", ascending=False)

# %% [markdown]
# **Top 10 RSI parameter combinations by Sharpe ratio** (in-sample sweep):

# %%
results_df.head(10).reset_index(drop=True)

# %%
# Visualize parameter sensitivity
fig = go.Figure(
    data=go.Heatmap(
        x=results_df[results_df["window"] == 14]["lower"],
        y=results_df[results_df["window"] == 14]["upper"],
        z=results_df[results_df["window"] == 14]["sharpe"],
        colorscale="RdYlGn",
        colorbar=dict(title="Sharpe Ratio"),
    )
)

fig.update_layout(
    title="Parameter Sensitivity: RSI Thresholds (14-day window)",
    xaxis_title="Lower Threshold",
    yaxis_title="Upper Threshold",
    height=500,
)
fig.show()

# %% [markdown]
# The highest in-sample Sharpe in the sweep is over an order of magnitude
# above the single-configuration baseline Sharpe printed in §4. That gap is
# the magnitude of in-sample optimization, not a forecast of out-of-sample
# edge: the parameters were selected after seeing the full dataset. VectorBT's
# speed makes such sweeps trivial to run, which makes overfitting risk
# correspondingly higher. The Deflated Sharpe Ratio (NB 12) addresses exactly
# this problem.

# %% [markdown]
# ## 9. Evaluate with ml4t-diagnostic
#
# For comprehensive portfolio analysis, we use ml4t-diagnostic's PortfolioAnalysis.

# %%
# Get daily returns
strategy_returns = portfolio.returns().values
benchmark_returns = bh_portfolio.returns().values

# Create analysis object
analysis = PortfolioAnalysis(
    returns=strategy_returns,
    benchmark=benchmark_returns,
    dates=portfolio.returns().index,
    periods_per_year=365,  # Crypto trades 365 days
)

# Compute comprehensive metrics
metrics = analysis.compute_summary_stats()

# %% [markdown]
# **ml4t-diagnostic comprehensive portfolio evaluation:**

# %%
print(metrics.summary())

# %% [markdown]
# ## See Also
#
# **ml4t-backtest Implementation**: See [`04_single_asset_ml4t_backtest`](04_single_asset_ml4t_backtest.ipynb) for the same
# strategy implemented with ml4t-backtest's event-driven engine. The main difference
# is not capability — VectorBT also supports stops, fills, and risk rules — but
# representation: ml4t-backtest expresses strategy logic as a Python class with
# explicit state, which mirrors how live trading code is typically structured.

# %% [markdown]
# ## Key Takeaways
#
# 1. **VectorBT API**: `Portfolio.from_signals()` is the core function for signal-based backtests
#
# 2. **Signal Generation**: Entry/exit signals are boolean arrays - simple and intuitive
#
# 3. **Parameter Optimization**: Vectorized operations enable fast parameter sweeps
#
# 4. **Transaction Costs**: Always include fees and slippage for realistic results
#
# 5. **Benchmarking**: Compare to buy-and-hold to validate strategy edge
#
# ## Next Steps
#
# - **06_framework_parity**: Compare vectorized and event-driven implementations
# - **08_signal_method_comparison**: Compare signal conversion methods
