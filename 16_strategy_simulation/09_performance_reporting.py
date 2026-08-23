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
# # Reporting a backtest so somebody else can audit it
#
# **Docker image**: `ml4t`
#
# ## Purpose
# A finished backtest is a pile of fills, positions and equity values. A report is the small set of
# numbers somebody who did not run it can use to decide whether to believe it. This notebook builds
# that report for one strategy - the RSI rule and execution contract from
# `04_single_asset_ml4t_backtest`, unchanged - and shows what each part of it is for.
#
# Four kinds of evidence are kept apart on purpose, because they answer different questions and
# mixing them is how a report flatters a strategy: the portfolio's own path, its relation to a
# benchmark that faced the same conditions, how much of the time capital was actually at risk, and
# what the round trips looked like once they closed.
#
# ## Learning objectives
#
# - Run the same strategy twice, once with costs and once without, and read the difference as the
#   price of implementing it.
# - Build a benchmark that trades the same instrument on the same dates through the same engine, so
#   that the comparison isolates the trading rule.
# - Report round-trip statistics from closed trades only, and say why the equity curve counts an
#   open position at the end of the sample while the win rate must not.
# - Account for commission and slippage exactly once, and prove it by reconciling trade P&L back to
#   the change in account value.
# - Compute the cost rate at which this backtest would have broken even, and say what that number
#   does and does not cover.
#
# ## Book reference
# Chapter 16, Section 16.5 (performance metrics and reporting).
#
# ## Prerequisites
#
# - `04_single_asset_ml4t_backtest`, which specifies the strategy, the warmup and the fill
#   convention this notebook reports on.

# %% [markdown]
# ## Setup

# %%
"""Build an auditable performance report from protocol-matched BTC backtests."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, Strategy
from ml4t.backtest.analytics import MAEMFEAnalyzer, TradeAnalyzer
from ml4t.backtest.config import ExecutionPrice, ShareType
from ml4t.diagnostic.integration import portfolio_analysis_from_result
from ml4t.diagnostic.visualization import add_annotation
from ml4t.diagnostic.visualization.portfolio import (
    plot_cumulative_returns,
    plot_drawdown_underwater,
    plot_monthly_returns_heatmap,
    plot_rolling_sharpe,
)

from data import load_crypto_perps
from utils.style import COLORS, ml4t_diverging

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides after this cell
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
INITIAL_CASH = 100_000
FEES = 0.001  # 10 bps per fill
SLIPPAGE = 0.0005  # 5 bps per fill
RSI_PERIOD = 14
RSI_LOWER = 30
RSI_UPPER = 70
POSITION_SIZE = 0.95
ROLLING_WINDOW_DAYS = 365
N_BARS = 0  # 0 uses the full date range; a positive value keeps the latest bars

# %% [markdown]
# ### What each setting decides
#
# **RSI period.** How many days of gains and losses the indicator averages. It also fixes the
# warmup: no signal exists until that many bars have passed, and nothing may trade before then.
#
# **Entry and exit levels.** The rule buys when the index falls below the lower level and sells
# when it rises above the upper one. The gap between them is what keeps the strategy from flipping
# on every small move.
#
# **Position size.** The share of equity a long position targets. It is deliberately under one:
# fills happen at the next bar's open at a price nobody knows when the order is placed, and a
# target of exactly one leaves nothing to absorb a gap up.
#
# **Commission and slippage.** Charged per fill. Commission is a fee on notional; slippage is the
# price moving against the order between the decision and the fill. Both are set here and both are
# switched off in the gross run, which is what makes the cost of implementation visible as a
# difference rather than an assertion.
#
# **Bar limit.** Zero uses the whole date range. A positive value keeps only the most recent bars,
# which is how a reduced-scale run is requested without changing any code.

# %%
print(f"Signal:   {RSI_PERIOD}-day RSI, buy below {RSI_LOWER}, sell above {RSI_UPPER}")
print(f"Sizing:   target {POSITION_SIZE:.0%} of equity when long, fractional units")
print(f"Costs:    {FEES:.2%} commission and {SLIPPAGE:.2%} slippage per fill, net run only")
print(f"Capital:  {INITIAL_CASH:,} USDT")
print(f"Sample:   {START_DATE} to {END_DATE}, UTC days")

# %% [markdown]
# ## 1. Rebuild the exact backtest being reported
#
# The strategy is deliberately unchanged from `04_single_asset_ml4t_backtest`. Its RSI is a simple
# rolling mean of gains and losses. The first valid value therefore appears only after the complete
# warmup window. A close-derived signal is submitted after the UTC day closes and fills at the next
# UTC day's open.

# %% [markdown]
# ### RSI indicator


# %%
def compute_rsi(close: pl.Series, period: int = 14) -> pl.Series:
    """Compute RSI from simple rolling mean gains and losses."""
    delta = close.diff()
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)
    avg_gain = gain.rolling_mean(window_size=period, min_samples=period)
    avg_loss = loss.rolling_mean(window_size=period, min_samples=period)

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# %% [markdown]
# ### RSI strategy
#
# Target weights express the allocation rule directly. Fractional sizing avoids the integer-unit
# truncation that is especially material for a high-priced asset such as BTC.


# %%
class RSIMeanReversionStrategy(Strategy):
    """Long-only RSI threshold strategy."""

    def __init__(
        self,
        rsi_lower: float = 30,
        rsi_upper: float = 70,
        position_size: float = 0.95,
    ):
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.position_size = position_size

    def on_data(self, timestamp, data, context, broker):
        """Submit target-weight orders from the close-based RSI state."""
        rsi = context.get("rsi")
        if rsi is None or np.isnan(rsi):
            return

        asset = "BTCUSDT"
        if data.get(asset) is None:
            return

        position = broker.get_position(asset)
        is_in_position = position is not None and position.quantity > 0

        if not is_in_position and rsi < self.rsi_lower:
            broker.order_target_percent(asset, self.position_size)
        elif is_in_position and rsi > self.rsi_upper:
            broker.order_target_percent(asset, 0.0)


# %% [markdown]
# ### A benchmark that faced the same conditions
#
# Buy-and-hold is the right benchmark for a long-only single-asset rule, and it is easy to give it
# an advantage by accident. It runs here through the same engine, on the same bars, with the same
# target exposure, the same next-open fills, the same fractional units, the same calendar and the
# same costs.
#
# It also waits for the same warmup. The RSI strategy cannot place an order until the indicator has
# a value, and a benchmark that buys on the first bar is holding the asset through a window the
# strategy was not allowed to trade. Over this sample that window is a rising one, so the advantage
# would be large and would look like the benchmark winning. Both therefore start when the signal
# does, and the only thing that differs between them is the trading rule.


# %%
class BuyAndHoldStrategy(Strategy):
    """Buy once on the first tradable bar and hold to the end of the sample."""

    def __init__(self, position_size: float = 0.95):
        self.position_size = position_size
        self.submitted = False

    def on_data(self, timestamp, data, context, broker):
        """Enter on the first bar the strategy could also have traded."""
        asset = "BTCUSDT"
        rsi = context.get("rsi")
        if rsi is None or np.isnan(rsi):
            return
        if not self.submitted and data.get(asset) is not None:
            broker.order_target_percent(asset, self.position_size)
            self.submitted = True


# %% [markdown]
# ### Canonical daily bars and indicator context
#
# BTC perpetual data arrive as hourly bars. Aggregation into UTC days preserves the canonical
# `symbol` and `timestamp` keys and validates the OHLC contract before any signal is computed.

# %%
_crypto = load_crypto_perps(
    symbols=["BTCUSDT"],
    start_date=START_DATE,
    end_date=END_DATE,
)
prices_df = (
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
if N_BARS > 0:
    prices_df = prices_df.tail(N_BARS)

price_columns = ["open", "high", "low", "close"]
assert prices_df.n_unique(["symbol", "timestamp"]) == len(prices_df)
assert prices_df["symbol"].unique().to_list() == ["BTCUSDT"]
assert prices_df.select(pl.col(price_columns).is_not_null().all()).row(0) == (True,) * 4
assert prices_df.select((pl.col(price_columns) > 0).all()).row(0) == (True,) * 4
assert prices_df.select((pl.col("high") >= pl.max_horizontal("open", "close", "low")).all()).item()
assert prices_df.select((pl.col("low") <= pl.min_horizontal("open", "close", "high")).all()).item()
assert len(prices_df) > RSI_PERIOD

print(f"Loaded {len(prices_df):,} unique BTC UTC-day bars")

# %%
rsi_values = compute_rsi(prices_df["close"], RSI_PERIOD)
context_df = prices_df.select("timestamp").with_columns(rsi_values.alias("rsi"))
rsi_valid = context_df["rsi"].drop_nulls().drop_nans()

assert len(rsi_valid) <= len(context_df) - RSI_PERIOD
assert rsi_valid.is_between(0, 100, closed="both").all()
print(f"RSI warmup leaves {len(rsi_valid):,} valid observations from {len(context_df):,} bars")

# %% [markdown]
# ### Gross, net, and benchmark runs
#
# Gross and net strategy runs share signals and timing. The gross run removes modeled costs; the net
# run and benchmark both include them. `ShareType.FRACTIONAL`, `calendar="crypto"`, and
# `ExecutionMode.NEXT_BAR` make the sizing, annualization, and fill convention explicit.

# %%
config_gross = BacktestConfig(
    initial_cash=INITIAL_CASH,
    execution_mode=ExecutionMode.NEXT_BAR,
    execution_price=ExecutionPrice.OPEN,
    commission_rate=0.0,
    slippage_rate=0.0,
    share_type=ShareType.FRACTIONAL,
    calendar="crypto",
)
config_net = BacktestConfig(
    initial_cash=INITIAL_CASH,
    execution_mode=ExecutionMode.NEXT_BAR,
    execution_price=ExecutionPrice.OPEN,
    commission_rate=FEES,
    slippage_rate=SLIPPAGE,
    share_type=ShareType.FRACTIONAL,
    calendar="crypto",
)

# %%
strategy_parameters = {
    "rsi_lower": RSI_LOWER,
    "rsi_upper": RSI_UPPER,
    "position_size": POSITION_SIZE,
}
results_gross = Engine(
    feed=DataFeed(prices_df=prices_df, context_df=context_df),
    strategy=RSIMeanReversionStrategy(**strategy_parameters),
    config=config_gross,
).run()
results_net = Engine(
    feed=DataFeed(prices_df=prices_df, context_df=context_df),
    strategy=RSIMeanReversionStrategy(**strategy_parameters),
    config=config_net,
).run()
benchmark_results = Engine(
    feed=DataFeed(prices_df=prices_df, context_df=context_df),
    strategy=BuyAndHoldStrategy(position_size=POSITION_SIZE),
    config=config_net,
).run()

assert results_gross.equity is not None and results_gross.equity.periods_per_year == 365
assert results_net.equity is not None and results_net.equity.periods_per_year == 365
assert benchmark_results.equity is not None and benchmark_results.equity.periods_per_year == 365
assert all(fill.asset == "BTCUSDT" for fill in results_net.fills)
assert all(fill.price_source == "open" for fill in results_net.fills)
assert len(benchmark_results.fills) == 1
assert len(benchmark_results.trades) == 1
assert benchmark_results.trades[0].status == "open"

# %%
run_summary = pl.DataFrame(
    {
        "run": ["RSI gross", "RSI net", "Buy and hold net"],
        "final_value_usdt": [
            float(results_gross["final_value"]),
            float(results_net["final_value"]),
            float(benchmark_results["final_value"]),
        ],
        "total_return_pct": [
            float(results_gross["total_return_pct"]),
            float(results_net["total_return_pct"]),
            float(benchmark_results["total_return_pct"]),
        ],
        "fills": [
            len(results_gross.fills),
            len(results_net.fills),
            len(benchmark_results.fills),
        ],
    }
).with_columns(pl.col(["final_value_usdt", "total_return_pct"]).round(2))
display(run_summary)

# %% [markdown]
# ## 2. Line the three return series up by date
#
# Alpha, beta and the information ratio all take two arrays of returns and assume element $i$ of
# one describes the same day as element $i$ of the other. Nothing in that interface checks it. If
# one series is a day shorter, or starts a day earlier, every benchmark-relative number is computed
# against a shifted series and nothing raises.
#
# So the three results become keyed daily frames first and are joined on their dates, one to one,
# with the join declaring that it expects one row on each side. Only then does anything become an
# array. The assertions after the join are the check that the alignment survived.


# %%
def daily_return_frame(result, column_name: str) -> pl.DataFrame:
    """Return a unique, sorted calendar-day return frame."""
    frame = (
        result.to_daily_pnl(session_aligned=False)
        .select(pl.col("date"), pl.col("return_pct").alias(column_name))
        .sort("date")
    )
    assert frame.n_unique("date") == len(frame)
    return frame


# %%
daily_returns = (
    daily_return_frame(results_gross, "gross_return")
    .join(daily_return_frame(results_net, "net_return"), on="date", how="inner", validate="1:1")
    .join(
        daily_return_frame(benchmark_results, "benchmark_return"),
        on="date",
        how="inner",
        validate="1:1",
    )
)

assert len(daily_returns) == len(results_net.to_daily_pnl(session_aligned=False))
assert daily_returns.select(pl.all().is_not_null().all()).row(0) == (True,) * 4

analysis_gross = portfolio_analysis_from_result(results_gross, calendar="crypto")
analysis_net = portfolio_analysis_from_result(
    results_net,
    calendar="crypto",
    benchmark=daily_returns["benchmark_return"],
)
analysis_benchmark = portfolio_analysis_from_result(benchmark_results, calendar="crypto")

assert analysis_net.dates.to_list() == daily_returns["date"].to_list()
assert np.allclose(analysis_gross.returns, daily_returns["gross_return"], rtol=0, atol=1e-12)
assert np.allclose(analysis_net.returns, daily_returns["net_return"], rtol=0, atol=1e-12)
assert np.allclose(
    analysis_benchmark.returns,
    daily_returns["benchmark_return"],
    rtol=0,
    atol=1e-12,
)

elapsed_days = (daily_returns["date"].max() - daily_returns["date"].min()).days
elapsed_years = elapsed_days / 365.0
assert elapsed_years > 0
print(
    f"Aligned {len(daily_returns):,} crypto daily returns from "
    f"{daily_returns['date'].min()} to {daily_returns['date'].max()}"
)

# %% [markdown]
# Daily return and risk statistics use 365 observations per year, the `crypto` calendar convention.
# Activity rates use the elapsed interval between the first and last keyed observations. Keeping
# those two quantities separate avoids an off-by-one year estimate from dividing row count by 365.

# %% [markdown]
# ## 3. What the portfolio earned, and how it compares

# %%
metrics_gross = analysis_gross.compute_summary_stats()
metrics_net = analysis_net.compute_summary_stats()
metrics_benchmark = analysis_benchmark.compute_summary_stats()
dd_result = analysis_net.compute_drawdown_analysis()
dist = analysis_net.compute_returns_distribution()

# %% [markdown]
# The first table is the portfolio's own path: what it earned, how much it moved, how far it fell.
# The second is its relation to the benchmark, and every number in it comes from the date-aligned
# return vectors above rather than from two separately computed summaries.
#
# Both describe the whole sample. Nothing was held out, so nothing here is an estimate of what the
# rule would do next.

# %%
performance_table = pl.DataFrame(
    {
        "metric": [
            "Total return",
            "Annual return",
            "Annual volatility",
            "Sharpe ratio",
            "Sortino ratio",
            "Maximum drawdown",
            "Calmar ratio",
        ],
        "rsi_gross": [
            metrics_gross.total_return,
            metrics_gross.annual_return,
            metrics_gross.annual_volatility,
            metrics_gross.sharpe_ratio,
            metrics_gross.sortino_ratio,
            metrics_gross.max_drawdown,
            metrics_gross.calmar_ratio,
        ],
        "rsi_net": [
            metrics_net.total_return,
            metrics_net.annual_return,
            metrics_net.annual_volatility,
            metrics_net.sharpe_ratio,
            metrics_net.sortino_ratio,
            metrics_net.max_drawdown,
            metrics_net.calmar_ratio,
        ],
        "buy_hold_net": [
            metrics_benchmark.total_return,
            metrics_benchmark.annual_return,
            metrics_benchmark.annual_volatility,
            metrics_benchmark.sharpe_ratio,
            metrics_benchmark.sortino_ratio,
            metrics_benchmark.max_drawdown,
            metrics_benchmark.calmar_ratio,
        ],
    }
).with_columns(pl.col(["rsi_gross", "rsi_net", "buy_hold_net"]).round(4))

# %% [markdown]
# **Portfolio performance under the matched protocol:**

# %%
display(performance_table)

# %%
benchmark_metrics = pl.DataFrame(
    {
        "metric": ["Annual alpha", "Beta", "Information ratio"],
        "rsi_net_vs_buy_hold": [
            metrics_net.alpha,
            metrics_net.beta,
            metrics_net.information_ratio,
        ],
    }
).with_columns(pl.col("rsi_net_vs_buy_hold").round(4))
display(benchmark_metrics)

# %%
print(f"RSI net total return:        {metrics_net.total_return:.2%}")
print(f"Buy-and-hold total return:   {metrics_benchmark.total_return:.2%}")
print(f"Beta of RSI net to benchmark: {metrics_net.beta:.2f}")

# %% [markdown]
# Beta below one is what a rule that is out of the market much of the time looks like: it cannot
# participate in a move it is not positioned for. That is a statement about exposure, not about
# skill. A strategy can lower its beta to any level by trading less, and the alpha figure in the
# table above is the one that asks whether the time it *was* invested was chosen well.

# %%
fig = plot_cumulative_returns(
    analysis_net,
    benchmark_label="Buy and hold",
    show_benchmark=True,
    log_scale=False,
)
fig.data[0].name = "RSI net"
fig.data[1].name = "Buy and hold"
fig.data[0].line.color = COLORS["blue"]
fig.data[1].line.color = COLORS["neutral"]
for trace in fig.data:
    add_annotation(
        fig,
        text=trace.name,
        x=trace.x[-1],
        y=trace.y[-1],
        xref="x",
        yref="y",
        xshift=8,
        xanchor="left",
        yanchor="middle",
        font={"color": trace.line.color, "size": 11},
    )
fig.update_layout(
    title=(
        "An RSI rule and buy-and-hold over the same bars and the same costs"
        f"<br><sup>Net cumulative return; both target {POSITION_SIZE:.0%} of equity and fill at "
        "the next open</sup>"
    ),
    yaxis_title="Cumulative return (%)",
    height=500,
    margin={"r": 105},
)
fig.show()

# %% [markdown]
# ## 4. How much of the time capital was at risk
#
# A Sharpe ratio says nothing about how much of the sample the strategy was actually invested. Two
# rules with the same Sharpe, one holding a position every day and one holding it a tenth of the
# time, are completely different propositions: the second leaves capital idle that could have been
# doing something else, and its return per unit of *deployed* capital is far higher.
#
# Gross exposure is the absolute value of the positions; net exposure is signed. They are identical
# for a long-only rule, which is worth checking rather than assuming - the assertion below is what
# turns "this strategy never shorts" from a claim into a fact about the run.

# %%
exposure_df = (
    results_net.to_portfolio_state_dataframe()
    .with_columns(
        gross_exposure_pct=pl.col("gross_exposure") / pl.col("equity"),
        net_exposure_pct=pl.col("net_exposure") / pl.col("equity"),
    )
    .sort("timestamp")
)
assert len(exposure_df) == len(prices_df)
assert exposure_df.select(pl.col("equity").is_finite().all()).item()
assert np.allclose(
    exposure_df["gross_exposure_pct"],
    exposure_df["net_exposure_pct"],
    rtol=0,
    atol=1e-12,
)

exposure_summary = pl.DataFrame(
    {
        "metric": [
            "Mean gross exposure",
            "Mean net exposure",
            "Maximum gross exposure",
            "Time invested",
        ],
        "value": [
            exposure_df["gross_exposure_pct"].mean(),
            exposure_df["net_exposure_pct"].mean(),
            exposure_df["gross_exposure_pct"].max(),
            (exposure_df["gross_exposure_pct"] > 1e-12).mean(),
        ],
    }
).with_columns(pl.col("value").round(4))
display(exposure_summary)

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=exposure_df["timestamp"].to_list(),
        y=exposure_df["gross_exposure_pct"].to_list(),
        name="Gross exposure",
        line=dict(color=COLORS["blue"], width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=exposure_df["timestamp"].to_list(),
        y=exposure_df["net_exposure_pct"].to_list(),
        name="Net exposure",
        line=dict(color=COLORS["amber"], width=1, dash="dot"),
    )
)
fig.update_layout(
    title=(
        "Gross and net exposure coincide for the long-only rule"
        "<br><sup>Position value divided by contemporaneous equity; net backtest</sup>"
    ),
    xaxis_title="Date",
    yaxis_title="Share of equity at risk (%)",
    yaxis_tickformat=".0%",
    height=420,
    hovermode="x unified",
)
fig.show()

# %% [markdown]
# ## 5. What the completed round trips looked like
#
# A backtest that ends mid-position leaves one trade unresolved. The equity curve has to carry it,
# because the money is genuinely at risk and marked at the last price. The win rate, the payoff
# ratio and the average holding period must not, because none of those is defined until the trade
# closes. Counting an open position that happens to be up, while a report that ended a week earlier
# would have excluded it, is the easiest way to flatter a strategy without writing a false number.
#
# Every statistic below therefore comes from trades with `status="closed"`, and the assertion
# checks that the count matches what the engine recorded rather than trusting the filter.

# %%
closed_trades = [trade for trade in results_net.trades if trade.status == "closed"]
open_trades = [trade for trade in results_net.trades if trade.status == "open"]
assert len(closed_trades) == int(results_net["num_trades"])
assert len(open_trades) <= 1

trades_df = results_net.to_trades_dataframe().filter(pl.col("status") == "closed")
assert len(trades_df) == len(closed_trades)
assert len(trades_df) > 0
assert trades_df["direction"].unique().to_list() == ["long"]

analyzer = TradeAnalyzer(closed_trades)
mfe_analyzer = MAEMFEAnalyzer(closed_trades)
trade_summary = pl.DataFrame(
    {
        "metric": [
            "Closed round trips",
            "Win rate",
            "Profit factor",
            "Payoff ratio",
            "Average net trade",
            "Average bars held",
            "MFE/MAE edge ratio",
            "Trade efficiency",
            "Mean MFE",
            "Mean MAE",
        ],
        "value": [
            float(analyzer.num_trades),
            analyzer.win_rate,
            analyzer.profit_factor,
            analyzer.payoff_ratio,
            analyzer.avg_trade,
            analyzer.avg_bars_held,
            mfe_analyzer.edge_ratio,
            mfe_analyzer.efficiency,
            mfe_analyzer.mfe_mean,
            mfe_analyzer.mae_mean,
        ],
    }
).with_columns(pl.col("value").round(4))
display(trade_summary)

# %% [markdown]
# MFE and MAE summarize paths that already occurred. They describe how much favorable and adverse
# movement each completed trade experienced. Without a separate validation design, they do not
# select or validate stop-loss and take-profit levels.

# %%
trade_log = trades_df.select(
    "symbol",
    "entry_time",
    "exit_time",
    "direction",
    "pnl",
    "net_return",
    "mfe",
    "mae",
    "exit_reason",
    "bars_held",
)
display(trade_log)

# %%
exit_reasons = (
    trades_df.group_by("exit_reason")
    .agg(
        closed_trades=pl.len(),
        mean_net_pnl=pl.col("pnl").mean(),
        total_net_pnl=pl.col("pnl").sum(),
        win_rate=(pl.col("pnl") > 0).mean(),
    )
    .sort("closed_trades", descending=True)
)
display(exit_reasons)

# %% [markdown]
# ## 6. Charge every cost exactly once
#
# Slippage is already inside the fill price: the price the engine recorded is the one the order was
# actually filled at, worse than the price it was aimed at by exactly the slippage. So computing
# P&L from fill prices and then subtracting the recorded slippage charges it twice, which quietly
# makes a strategy look worse than the engine actually made it. To recover reference-price P&L, remove entry and exit slippage from the
# slipped prices, then subtract slippage dollars and commissions exactly once:
#
# $$
# \text{net P\&L}
# = \text{reference-price P\&L}
# - \text{slippage cost}
# - \text{commission}.
# $$

# %%
all_trades_df = results_net.to_trades_dataframe()
assert all_trades_df["direction"].unique().to_list() == ["long"]

reconciled_trades = (
    all_trades_df.with_columns(
        entry_reference_price=pl.col("entry_price") - pl.col("entry_slippage"),
        exit_reference_price=pl.col("exit_price") + pl.col("exit_slippage"),
    )
    .with_columns(
        reference_gross_pnl=(pl.col("exit_reference_price") - pl.col("entry_reference_price"))
        * pl.col("quantity")
        * pl.col("multiplier"),
        slippage_cost=(pl.col("entry_slippage") + pl.col("exit_slippage"))
        * pl.col("quantity").abs()
        * pl.col("multiplier"),
    )
    .with_columns(
        execution_cost=pl.col("fees") + pl.col("slippage_cost"),
        reconciled_net_pnl=pl.col("reference_gross_pnl") - pl.col("slippage_cost") - pl.col("fees"),
    )
)
assert np.allclose(reconciled_trades["reconciled_net_pnl"], reconciled_trades["pnl"], atol=1e-8)

cost_reconciliation = reconciled_trades.select(
    pl.col("reference_gross_pnl").sum().alias("reference_price_pnl"),
    pl.col("slippage_cost").sum().alias("slippage_cost"),
    pl.col("fees").sum().alias("commission"),
    pl.col("pnl").sum().alias("net_realized_and_marked_pnl"),
)
assert np.isclose(
    cost_reconciliation["reference_price_pnl"].item()
    - cost_reconciliation["slippage_cost"].item()
    - cost_reconciliation["commission"].item(),
    cost_reconciliation["net_realized_and_marked_pnl"].item(),
    atol=1e-8,
)
assert np.isclose(
    cost_reconciliation["net_realized_and_marked_pnl"].item(),
    float(results_net["final_value"]) - INITIAL_CASH,
    atol=1e-8,
)
display(cost_reconciliation)

# %% [markdown]
# ### One-way turnover and break-even cost
#
# Turnover uses every fill, including the entry of an open end-of-sample position. The execution
# base price is recovered directly from the slipped fill price and side. The fill's quote-context
# `reference_price` is not used because it can represent a different price surface. The break-even
# rate is expressed per one-way execution-base notional for the fixed observed path:
#
# $$
# c_{\text{break-even}}
# = \frac{\text{reference-price gross P\&L}}{\text{total one-way reference notional}}.
# $$
#
# This exact ratio for the observed quantities holds their path fixed. It does not model market
# impact or allow a different cost rate to alter future target quantities or signals.

# %%
fills_df = (
    results_net.to_fills_dataframe()
    .with_columns(
        execution_base_price=pl.when(pl.col("side") == "buy")
        .then(pl.col("price") - pl.col("slippage"))
        .otherwise(pl.col("price") + pl.col("slippage")),
    )
    .with_columns(
        reference_notional=pl.col("quantity").abs() * pl.col("execution_base_price"),
        slippage_cost=pl.col("quantity").abs() * pl.col("slippage").abs(),
    )
    .with_columns(fill_cost=pl.col("commission") + pl.col("slippage_cost"))
)

# %% [markdown]
# Aggregate fill notional and costs before annualizing activity over the observed calendar span.

# %%
total_reference_notional = float(fills_df["reference_notional"].sum())
total_fill_cost = float(fills_df["fill_cost"].sum())
average_equity = float(results_net.to_portfolio_state_dataframe()["equity"].mean())
annual_turnover = (total_reference_notional / average_equity) / elapsed_years
observed_cost_bps = total_fill_cost / total_reference_notional * 10_000
reference_price_pnl = float(cost_reconciliation["reference_price_pnl"].item())
assert np.isclose(
    total_fill_cost,
    cost_reconciliation["slippage_cost"].item() + cost_reconciliation["commission"].item(),
    atol=1e-8,
)

if total_reference_notional > 0 and reference_price_pnl > 0:
    breakeven_bps = reference_price_pnl / total_reference_notional * 10_000
    cost_margin_bps = breakeven_bps - observed_cost_bps
else:
    breakeven_bps = np.nan
    cost_margin_bps = np.nan

turnover_table = pl.DataFrame(
    {
        "metric": [
            "Elapsed years",
            "Annual one-way turnover",
            "Observed cost per one-way notional (bps)",
            "Fixed-path break-even cost per one-way notional (bps)",
            "Cost margin (bps)",
        ],
        "value": [
            elapsed_years,
            annual_turnover,
            observed_cost_bps,
            breakeven_bps,
            cost_margin_bps,
        ],
    }
).with_columns(pl.col("value").round(2))
display(turnover_table)

# %%
if np.isfinite(breakeven_bps):
    print(f"Cost actually paid, per one-way notional:  {observed_cost_bps:.1f} bps")
    print(f"Cost at which this path breaks even:       {breakeven_bps:.1f} bps")
    print(f"Margin:                                    {cost_margin_bps:.1f} bps")
else:
    print("Break-even cost is undefined: gross P&L or traded notional is not positive.")

# %% [markdown]
# **On this run the margin does not exist.** Gross P&L is not positive, so there is no cost at
# which the path breaks even: it did not make money at zero cost either, and the table prints both
# figures as NaN. Nothing below is a reading of this run's numbers; it is what the margin means
# where there is one, kept because the quantity is worth understanding before the next run produces
# it.
#
# Where gross P&L is positive, the margin answers "how much more expensive could execution have
# been before this strategy stopped making money", and it is the number to compare against a
# broker's actual schedule before trading anything.
#
# It holds the trade path fixed, which is the assumption that limits it even then. A strategy
# paying more per trade would trade differently: some entries would no longer clear their own cost, positions would
# be held longer, and the sequence of fills would diverge from this one. The margin is therefore an
# upper bound on tolerance, not a forecast of behaviour at a higher cost. It also contains no market
# impact - the slippage here is a fixed rate that does not grow with order size, which Chapter 18
# replaces.

# %% [markdown]
# ## 7. Four views of the same return series
#
# Four views of the same net return series, each answering a question the summary table cannot.
# How long the portfolio spent below its previous peak, not only how far it fell. Whether the
# risk-adjusted return came from the whole sample or from one stretch of it. Which months carried
# the result. And what the distribution of daily returns looks like for a rule that is flat most of
# the time.

# %%
fig = plot_drawdown_underwater(analysis=analysis_net)
fig.data[0].line.color = COLORS["negative"]
fig.data[0].fillcolor = COLORS["negative"]
fig.data[0].opacity = 0.3
fig.data[1].marker.color = COLORS["negative"]
fig.update_layout(
    title=(
        "Time under water, not just the depth of the worst fall"
        "<br><sup>Net peak-to-trough return; zero is the high-water mark</sup>"
    ),
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    height=360,
)
fig.show()

# %%
rolling = analysis_net.compute_rolling_metrics(windows=[ROLLING_WINDOW_DAYS], metrics=["sharpe"])
fig = plot_rolling_sharpe(rolling_result=rolling)
fig.data[0].line.color = COLORS["blue"]
for shape in fig.layout.shapes:
    shape.line.color = COLORS["neutral"]
for annotation in fig.layout.annotations:
    annotation.font.color = COLORS["neutral"]
fig.update_layout(
    title=(
        "Risk-adjusted performance varies across the sample"
        f"<br><sup>{ROLLING_WINDOW_DAYS}-day rolling Sharpe, annualized on a 365-day year</sup>"
    ),
    xaxis_title="Date",
    yaxis_title="Rolling Sharpe ratio",
    height=420,
)
fig.show()

# %%
fig = plot_monthly_returns_heatmap(analysis_net)
fig.data[0].colorscale = ml4t_diverging()
fig.update_layout(
    title=(
        "Monthly outcomes are concentrated in active-position windows"
        "<br><sup>Net calendar-month return; annual column compounds monthly observations</sup>"
    ),
    xaxis_title="Month",
    yaxis_title="Year",
)
fig.show()

# %%
fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=daily_returns["net_return"].to_list(),
        nbinsx=60,
        name="Daily returns",
        marker_color=COLORS["blue"],
        opacity=0.85,
        hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>",
    )
)
fig.add_vline(x=0, line_color=COLORS["neutral"], line_width=1)
fig.add_vline(
    x=dist.var_95,
    line_color=COLORS["negative"],
    line_dash="dot",
    annotation_text=f"VaR 95%: {dist.var_95:.2%}",
    annotation_position="top left",
)
fig.update_layout(
    title=(
        "Daily returns combine many flat days with a heavy active tail"
        "<br><sup>Net calendar-day returns; vertical line marks the empirical 95% VaR</sup>"
    ),
    xaxis_title="Daily return (%)",
    yaxis_title="Number of days",
    xaxis_tickformat=".1%",
    bargap=0.04,
    height=420,
)
fig.show()

# %% [markdown]
# ## 8. The report itself
#
# A compact report keeps portfolio path, benchmark relation, implementation, and trading behavior
# distinct. Exact values are assembled from the freshly computed objects rather than copied into
# markdown.

# %%
report = {
    "Annual return (gross)": f"{metrics_gross.annual_return:.2%}",
    "Annual return (net)": f"{metrics_net.annual_return:.2%}",
    "Annual volatility (net)": f"{metrics_net.annual_volatility:.2%}",
    "Sharpe ratio (net)": f"{metrics_net.sharpe_ratio:.2f}",
    "Sortino ratio (net)": f"{metrics_net.sortino_ratio:.2f}",
    "Maximum drawdown": f"{metrics_net.max_drawdown:.2%}",
    "Maximum drawdown duration": f"{dd_result.max_duration_days} days",
    "Beta vs buy-and-hold": f"{metrics_net.beta:.2f}",
    "Closed round trips": str(analyzer.num_trades),
    "Open trades at sample end": str(len(open_trades)),
    "Win rate": f"{analyzer.win_rate:.1%}",
    "Annual one-way turnover": f"{annual_turnover:.1%}",
    "Mean gross exposure": f"{exposure_df['gross_exposure_pct'].mean():.1%}",
    "Observed one-way cost": f"{observed_cost_bps:.1f} bps",
}
report_df = pl.DataFrame({"metric": list(report), "value": list(report.values())})
display(report_df)

# %% [markdown]
# ## Key takeaways
#
# 1. **A benchmark has to face the same constraints, including the ones that are easy to miss.**
#    A strategy with a warmup cannot trade during it. A buy-and-hold benchmark that can is not
#    measuring the trading rule, it is measuring the warmup, and on a trending sample that
#    difference is larger than most of the effects a report is trying to show.
# 2. **Join before you compare.** Benchmark-relative statistics take two arrays and assume they
#    line up. Joining the two return series on their dates, one to one, and asserting the result
#    before anything is passed as an array is what makes alpha and beta mean what their names say.
# 3. **Equity and round trips answer different questions.** The equity curve has to carry an open
#    position at the end of the sample, because the money is really at risk; the win rate must not,
#    because the trade has not resolved. Drawing both from one pool is how an unresolved loss
#    quietly leaves a report.
# 4. **Charge each cost once and prove it.** Fill prices already contain slippage, so adding the
#    recorded slippage to P&L computed from those prices counts it twice. The reconciliation here
#    recovers reference-price P&L, subtracts commission and slippage once, and asserts the result
#    equals the change in account value. That assertion is the check; the prose is not.
# 5. **A break-even cost rate is more useful than a Sharpe ratio when deciding whether to trade.**
#    It converts the whole backtest into one number a reader can hold against a real broker's fee
#    schedule.
#
# ### Known limitations
#
# - One asset, one rule, one sample, no holdout. Every number describes the period it was computed
#   on and none of them estimates what the rule does next.
# - The entry and exit levels are conventional RSI defaults, not chosen on this data. That is what
#   keeps the sample honest, and it also means nothing here says these are good levels.
# - Slippage is a fixed rate per fill. It does not grow with order size and does not depend on how
#   much volume the market had, so the cost figures understate what a large account would pay.
# - The break-even margin holds the observed trade path fixed. A higher cost would change which
#   trades happen, so the margin bounds tolerance rather than predicting behaviour.
#
# **Next:** `10_regime_backtest_analysis` splits this same report by market state and asks which
# of its numbers hold in each. Section 16.5 covers metric interpretation and the thresholds practitioners use.
