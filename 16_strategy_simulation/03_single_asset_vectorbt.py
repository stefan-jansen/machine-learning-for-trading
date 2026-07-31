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
# This notebook introduces **VectorBT**, a vectorized backtesting library that enables
# fast simulation of trading strategies. We start with a simple single-asset RSI mean-reversion
# strategy on Bitcoin before moving to multi-asset portfolio strategies. Every performance result
# is an in-sample teaching output, not a holdout estimate or deployable strategy claim.
#
# **Learning Objectives:**
#
# 1. Understand VectorBT's core API and portfolio simulation
# 2. Implement entry/exit signals from technical indicators
# 3. Analyze performance metrics and visualizations
# 4. Compare strategy vs buy-and-hold benchmark
#
# **Book Reference:** Chapter 16, Section 16.3 - vectorized and event-driven engines.
#
# **Prerequisites:** Ch16 NB 01 (backtesting first principles).
#
# **Strategy:** RSI Mean Reversion on BTC
#
# - **Long Entry**: RSI below the configured lower threshold
# - **Exit**: RSI above the configured upper threshold
# - **Rebalancing**: Daily
# - **Transaction Costs**: Fee plus slippage on each fill

# %% [markdown]
# ## Setup

# %%
"""Single-asset RSI mean-reversion strategy using vectorized backtesting."""

import plotly.graph_objects as go
import polars as pl
import vectorbt as vbt
from IPython.display import Markdown, display
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from plotly.subplots import make_subplots

from data import load_crypto_perps
from utils.style import COLORS, ml4t_diverging

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides for CI
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
INITIAL_CASH = 100_000
RSI_WINDOW = 14
RSI_LOWER = 30
RSI_UPPER = 70
SIZE_FRACTION = 0.95
FEES = 0.001
SLIPPAGE = 0.0005

# %%
display(
    Markdown(
        f"The configured rule uses a **{RSI_WINDOW}-day RSI**, enters below **{RSI_LOWER}**, "
        f"exits above **{RSI_UPPER}**, and allocates **{SIZE_FRACTION:.0%}** of available capital. "
        f"Each fill pays a **{FEES:.2%} fee** plus **{SLIPPAGE:.2%} slippage**."
    )
)

# %% [markdown]
# ## 1. Data Acquisition
#
# Load BTC/USDT daily bars from the local crypto perpetuals dataset and
# convert to pandas Series indexed by timestamp at the VectorBT boundary. The UTC-day aggregation
# is a calendar convention for a continuously traded market, not an exchange session close.

# %%
_crypto = load_crypto_perps(
    symbols=["BTCUSDT"],
    start_date=START_DATE,
    end_date=END_DATE,
)
btc_df = (
    _crypto.filter(
        (pl.col("symbol") == "BTCUSDT")
        & (pl.col("timestamp") >= pl.lit(START_DATE).str.to_datetime().dt.replace_time_zone("UTC"))
        & (pl.col("timestamp") < pl.lit(END_DATE).str.to_datetime().dt.replace_time_zone("UTC"))
    )
    .sort("timestamp")
    .with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    .group_by_dynamic("timestamp", every="1d", group_by="symbol")
    .agg(
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    )
    .sort(["symbol", "timestamp"])
)

# %% [markdown]
# Canonical keys must remain unique through aggregation. Positive, internally consistent OHLC bars
# protect the backtest from malformed price records before the VectorBT boundary.

# %%
assert btc_df.n_unique(["symbol", "timestamp"]) == len(btc_df)
assert btc_df["symbol"].unique().to_list() == ["BTCUSDT"]
price_columns = ["open", "high", "low", "close"]
assert btc_df.select(pl.col(price_columns).is_not_null().all()).row(0) == (True,) * 4
assert btc_df.select((pl.col(price_columns) > 0).all()).row(0) == (True,) * 4
assert btc_df.select((pl.col("high") >= pl.max_horizontal("open", "close", "low")).all()).item()
assert btc_df.select((pl.col("low") <= pl.min_horizontal("open", "close", "high")).all()).item()
assert len(btc_df) > RSI_WINDOW

# Convert at the library boundary; subsequent pandas objects come from VectorBT.
btc_pd = btc_df.to_pandas()
btc_pd.set_index("timestamp", inplace=True)
close = btc_pd["close"]
execution_price = btc_pd["open"]

print(f"Loaded {len(btc_df):,} daily bars for BTCUSDT from the crypto perpetuals dataset")
print(f"Date range: {btc_df['timestamp'].min()} to {btc_df['timestamp'].max()}")

# %%
close.head(10)

# %% [markdown]
# ## 2. RSI Indicator Calculation
#
# VectorBT provides optimized indicator calculations. The RSI indicator
# can be computed for multiple parameter values simultaneously.

# %%
rsi = vbt.RSI.run(close, window=RSI_WINDOW)

# %%
rsi_series = rsi.rsi

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3],
    subplot_titles=["BTC/USDT Price", f"RSI ({RSI_WINDOW})"],
)

fig.add_trace(
    go.Scatter(x=close.index, y=close, name="BTC", line=dict(color=COLORS["blue"])),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(x=rsi_series.index, y=rsi_series, name="RSI", line=dict(color=COLORS["amber"])),
    row=2,
    col=1,
)

fig.add_hline(y=RSI_LOWER, line_dash="dash", line_color=COLORS["positive"], row=2, col=1)
fig.add_hline(y=RSI_UPPER, line_dash="dash", line_color=COLORS["negative"], row=2, col=1)

fig.update_layout(
    height=600,
    title=(
        "RSI thresholds isolate BTC price extremes"
        "<br><sup>Daily BTCUSDT; trailing close-based indicator</sup>"
    ),
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
#
# - **Enter Long**: the prior close's RSI is below the lower threshold
# - **Exit Long**: the prior close's RSI is above the upper threshold
#
# RSI is observed only after the daily close. Shifting both conditions by one row makes the order
# eligible at the next UTC day's open and prevents a same-close look-ahead fill.

# %%
entry_condition = rsi_series < RSI_LOWER
exit_condition = rsi_series > RSI_UPPER
entries = entry_condition.shift(1, fill_value=False)
exits = exit_condition.shift(1, fill_value=False)

# %%
entry_days = int(entries.sum())
exit_days = int(exits.sum())
display(
    Markdown(
        f"The conditions mark **{entry_days} entry-eligible days** and "
        f"**{exit_days} exit-eligible days**. Repeated signals while a position is already "
        "open do not create additional trades."
    )
)

# %% [markdown]
# ## 4. Run Backtest with VectorBT
#
# VectorBT's `Portfolio.from_signals()` is the core backtesting function.
# It simulates a portfolio based on entry/exit signals.

# %%
portfolio = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    price=execution_price,
    open=execution_price,
    size=SIZE_FRACTION,
    size_type="percent",
    init_cash=INITIAL_CASH,
    fees=FEES,
    slippage=SLIPPAGE,
    freq="1D",
)

# %% [markdown]
# **Portfolio performance summary** (VectorBT's built-in strategy statistics):
#
# The matched-cost benchmark is constructed below. This view suppresses VectorBT's implicit
# frictionless close-to-close benchmark so the notebook does not mix execution assumptions.

# %%
strategy_stats = portfolio.stats()
strategy_stats.drop(labels=["Benchmark Return [%]"], errors="ignore")

# %% [markdown]
#
# %%
baseline_return = float(portfolio.total_return())
baseline_sharpe = float(portfolio.sharpe_ratio())
display(
    Markdown(
        f"The configured RSI rule returns **{baseline_return:.2%}** with a "
        f"**{baseline_sharpe:.2f} Sharpe ratio** in this sample. This descriptive output shows "
        "what the rule did, but does not identify a persistent mean-reversion premium or estimate "
        "out-of-sample performance."
    )
)

# %% [markdown]
# ## 5. Analyze Trade Statistics
#
# VectorBT records each trade lifecycle, including any position still open at the sample end. The
# table exposes entry and exit prices so the execution convention remains auditable.

# %%
trades = portfolio.trades.records_readable
display(Markdown(f"The simulation contains **{len(trades)} trade records**."))

# %% [markdown]
# **Trade statistics** (per-trade entry/exit, P&L, holding period summary):

# %%
portfolio.trades.stats()

# %% [markdown]
# **Sample individual trades** with entry/exit timing and realized P&L:

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
#
# Portfolio value reveals when the rule is invested and when capital is idle. The underwater curve
# measures each close relative to the running peak, with zero fixed at the top of the chart.

# %%
portfolio_value = portfolio.value()
fig = go.Figure(
    go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value,
        name="Portfolio value",
        line=dict(color=COLORS["blue"]),
    )
)
fig.update_layout(
    title=(
        "The RSI rule creates an intermittent portfolio path"
        "<br><sup>Net of configured fees and slippage; full sample, in-sample</sup>"
    ),
    xaxis_title="Date",
    yaxis_title="Portfolio Value (USDT)",
    height=450,
    hovermode="x unified",
)
fig.show()

# %%
portfolio_drawdown = portfolio.drawdown() * 100
drawdown_floor = min(float(portfolio_drawdown.min()) * 1.05, -1.0)
fig_dd = go.Figure(
    go.Scatter(
        x=portfolio_drawdown.index,
        y=portfolio_drawdown,
        name="Drawdown",
        line=dict(color=COLORS["negative"]),
        fill="tozeroy",
        fillcolor=COLORS["silver_muted"],
    )
)
fig_dd.update_layout(
    title=(
        "The RSI portfolio spends extended periods below its prior peak"
        "<br><sup>Close-to-close peak-to-trough drawdown; net, in-sample</sup>"
    ),
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    yaxis_range=[drawdown_floor, 0],
    height=400,
    hovermode="x unified",
)
fig_dd.show()

# %% [markdown]
# ## 7. Compare to Buy-and-Hold Benchmark
#
# Buy-and-hold is a simple exposure benchmark. Both paths deploy the same fraction of capital and
# use the same next-open price, fee, and slippage assumptions. The benchmark remains open at the
# sample end, so it does not pay a terminal exit cost.

# %%
bh_portfolio = vbt.Portfolio.from_holding(
    close=close,
    price=execution_price,
    open=execution_price,
    size=SIZE_FRACTION,
    size_type="percent",
    init_cash=INITIAL_CASH,
    fees=FEES,
    slippage=SLIPPAGE,
    freq="1D",
)

# %%
comparison = pl.DataFrame(
    {
        "metric": [
            "Total Return (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Total Trades",
            "Sortino Ratio",
            "Calmar Ratio",
        ],
        "RSI Strategy": [
            float(portfolio.total_return() * 100),
            float(portfolio.sharpe_ratio()),
            float(portfolio.max_drawdown() * 100),
            float(portfolio.trades.count()),
            float(portfolio.sortino_ratio()),
            float(portfolio.calmar_ratio()),
        ],
        "Buy & Hold": [
            float(bh_portfolio.total_return() * 100),
            float(bh_portfolio.sharpe_ratio()),
            float(bh_portfolio.max_drawdown() * 100),
            float(bh_portfolio.trades.count()),
            float(bh_portfolio.sortino_ratio()),
            float(bh_portfolio.calmar_ratio()),
        ],
    }
)

# %% [markdown]
# **Strategy vs buy-and-hold comparison** (daily data; risk ratios use daily annualization):

# %%
comparison.with_columns(pl.col(["RSI Strategy", "Buy & Hold"]).round(3))

# %% [markdown]
#
# %%
benchmark_return = float(bh_portfolio.total_return())
benchmark_sharpe = float(bh_portfolio.sharpe_ratio())
strategy_drawdown = float(portfolio.max_drawdown())
benchmark_drawdown = float(bh_portfolio.max_drawdown())
return_leader = "Buy-and-hold" if benchmark_return > baseline_return else "The RSI strategy"
display(
    Markdown(
        f"**{return_leader}** leads total return in this sample: "
        f"**{benchmark_return:.2%}** for buy-and-hold versus **{baseline_return:.2%}** for RSI. "
        f"Their Sharpe ratios are **{benchmark_sharpe:.2f}** and **{baseline_sharpe:.2f}**; maximum "
        f"drawdowns are **{benchmark_drawdown:.2%}** and **{strategy_drawdown:.2%}**, respectively. "
        "This is an in-sample exposure comparison, not evidence that either rule will dominate "
        "out of sample."
    )
)

# %%
fig = go.Figure()

strategy_cum = portfolio.cumulative_returns() * 100
bh_cum = bh_portfolio.cumulative_returns() * 100

fig.add_trace(
    go.Scatter(
        x=strategy_cum.index,
        y=strategy_cum,
        name="RSI Strategy",
        line=dict(color=COLORS["blue"]),
    )
)

fig.add_trace(
    go.Scatter(
        x=bh_cum.index,
        y=bh_cum,
        name="Buy & Hold",
        line=dict(color=COLORS["neutral"], dash="dash"),
    )
)

fig.update_layout(
    title=(
        f"{return_leader} leads cumulative return in this sample"
        "<br><sup>Matched capital allocation and costs; full sample, in-sample</sup>"
    ),
    xaxis_title="Date",
    yaxis_title="Cumulative Return (%)",
    height=500,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    hovermode="x unified",
)
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], line_width=1)
fig.show()

# %% [markdown]
# ## 8. Parameter Sensitivity
#
# VectorBT makes parameter sweeps inexpensive. This section measures in-sample sensitivity; it does
# not call the best row an optimized deployment choice because no holdout is used.

# %%
rsi_windows = [7, 14, 21]
lower_thresholds = [20, 25, 30, 35]
upper_thresholds = [65, 70, 75, 80]

# %%
results = []

for window in rsi_windows:
    rsi_vals = vbt.RSI.run(close, window=window).rsi

    for lower in lower_thresholds:
        for upper in upper_thresholds:
            if lower >= upper:
                continue

            entries = (rsi_vals < lower).shift(1, fill_value=False)
            exits = (rsi_vals > upper).shift(1, fill_value=False)

            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                price=execution_price,
                open=execution_price,
                size=SIZE_FRACTION,
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
results_df = pl.DataFrame(results)
ranked_results = results_df.filter(pl.col("sharpe").is_finite()).sort("sharpe", descending=True)
assert len(ranked_results) > 0

# %% [markdown]
# **Highest-Sharpe RSI parameter combinations** (in-sample sweep):

# %%
ranked_results.head(10)

# %%
window_results = results_df.filter(pl.col("window") == RSI_WINDOW)
heatmap_z = [
    [
        window_results.filter((pl.col("lower") == lower) & (pl.col("upper") == upper))[
            "sharpe"
        ].item()
        for lower in lower_thresholds
    ]
    for upper in upper_thresholds
]
best_parameters = ranked_results.row(0, named=True)
window_best = (
    window_results.filter(pl.col("sharpe").is_finite())
    .sort("sharpe", descending=True)
    .row(0, named=True)
)

# %% [markdown]
# The heatmap holds the RSI window fixed and exposes the threshold surface. Centering the diverging
# scale at zero distinguishes positive from negative in-sample Sharpe ratios without implying that
# the highest cell is a validated choice.

# %%
fig = go.Figure(
    data=go.Heatmap(
        x=lower_thresholds,
        y=upper_thresholds,
        z=heatmap_z,
        colorscale=ml4t_diverging(),
        zmid=0,
        colorbar=dict(title="Sharpe Ratio"),
    )
)

fig.update_layout(
    title=(
        f"At window {RSI_WINDOW}, in-sample Sharpe peaks at thresholds "
        f"{window_best['lower']}/{window_best['upper']}"
        "<br><sup>Full-sample sensitivity, net of costs; no holdout ranking</sup>"
    ),
    xaxis_title="Lower Threshold",
    yaxis_title="Upper Threshold",
    height=500,
)
fig.show()

# %% [markdown]
#
# %%
display(
    Markdown(
        f"The best grid row has an in-sample Sharpe of **{best_parameters['sharpe']:.2f}**, versus "
        f"**{baseline_sharpe:.2f}** for the configured baseline. That gap measures full-sample "
        "selection, not forecast improvement: every candidate was ranked after observing the same "
        "return path. NB 12 introduces the Deflated Sharpe Ratio for this multiple-testing problem."
    )
)

# %% [markdown]
# ## 9. Evaluate with ml4t-diagnostic
#
# `PortfolioAnalysis` recomputes portfolio statistics from the return series. Agreement on total
# return and maximum drawdown provides a cross-library accounting check; the remaining rows extend
# the report with distribution and benchmark-relative diagnostics.

# %%
strategy_returns = portfolio.returns().values
benchmark_returns = bh_portfolio.returns().values

analysis = PortfolioAnalysis(
    returns=strategy_returns,
    benchmark=benchmark_returns,
    dates=portfolio.returns().index,
    periods_per_year=365,  # Crypto trades 365 days
)

metrics = analysis.compute_summary_stats()
assert abs(metrics.total_return - baseline_return) < 1e-10
assert abs(metrics.max_drawdown - strategy_drawdown) < 1e-10

# %% [markdown]
# **Selected ml4t-diagnostic portfolio statistics:**

# %%
percentage_metrics = [
    "total_return",
    "annual_return",
    "annual_volatility",
    "max_drawdown",
    "var_95",
    "cvar_95",
    "win_rate",
    "avg_win",
    "avg_loss",
    "alpha",
]
metrics_table = (
    metrics.to_dataframe()
    .unpivot(variable_name="metric", value_name="value")
    .filter(~pl.col("metric").is_in(["up_capture", "down_capture"]))
    .with_columns(
        pl.when(pl.col("metric").is_in(percentage_metrics))
        .then((pl.col("value") * 100).round(2))
        .otherwise(pl.col("value").round(3))
        .alias("value"),
        pl.when(pl.col("metric").is_in(percentage_metrics))
        .then(pl.lit("%"))
        .otherwise(pl.lit("unitless"))
        .alias("unit"),
    )
)
metrics_table

# %% [markdown]
# ## See Also
#
# **ml4t-backtest Implementation**: See [`04_single_asset_ml4t_backtest`](04_single_asset_ml4t_backtest.ipynb) for the same
# strategy implemented with ml4t-backtest's event-driven engine. The main difference
# is not capability - VectorBT also supports stops, fills, and risk rules - but
# representation: ml4t-backtest expresses strategy logic as a Python class with
# explicit state, which mirrors how live trading code is typically structured.

# %% [markdown]
# ## Key Takeaways
#
# 1. **VectorBT API**: `Portfolio.from_signals()` is the core function for signal-based backtests
#
# 2. **Signal timing**: Close-derived conditions are shifted to the next UTC-day open
#
# 3. **Parameter sensitivity**: Vectorized operations enable fast parameter sweeps, but ranking
#    candidates on one sample does not establish out-of-sample improvement
#
# 4. **Transaction costs**: Strategy and benchmark use the same fee and slippage convention
#
# 5. **Benchmarking**: A same-budget buy-and-hold path separates timing from passive exposure
#
# ## Next Steps
#
# - **06_framework_parity**: Compare vectorized and event-driven implementations
# - **08_signal_method_comparison**: Compare signal conversion methods
