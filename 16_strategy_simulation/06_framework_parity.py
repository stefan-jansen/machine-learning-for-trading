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
# # ETF momentum: the same strategy run two ways
#
# **Docker image**: `ml4t`
#
# ## Purpose
# One rotation rule, one set of monthly target weights, two ways of turning those targets into a
# track record. The first multiplies a matrix of lagged weights by a matrix of returns, which is
# how research code written with pandas or a vectorized library usually starts. The second hands
# the same targets to `ml4t-backtest`, which walks the data one bar at a time, submits orders,
# fills them at a price, and keeps a share count and a cash balance.
#
# Both runs are given the same universe, the same signal, the same dates and the same total cost
# per dollar traded. Whatever separates their ending wealth is therefore the execution protocol:
# when a target is acted on, what price it is filled at, and whether the account has to hold whole
# shares and enough cash.
#
# ## Learning objectives
#
# - Produce one set of monthly target weights and drive two different simulations from it, so that
#   nothing about the signal can account for a difference between their results.
# - Measure how far apart the two finish, in ending wealth and in summary statistics computed by
#   the same code for both.
# - List the assumptions that still differ between them - the moment a target is acted on, the
#   price it fills at, how the cost is charged, whether shares and cash are tracked - and check
#   each one against the code rather than against the name of the library.
# - Say what a high correlation between two daily return series does and does not establish about
#   the two backtests that produced them.
#
# ## Book reference
# Chapter 16, Section 16.3 (the modern backtesting workbench).
#
# ## Prerequisites
#
# - `01_backtest_first_principles`, which builds this ETF momentum protocol from scratch.
# - `03_single_asset_vectorbt` and `04_single_asset_ml4t_backtest`, which introduce the two
#   implementation styles on a single asset.

# %% [markdown]
# ## Two ways to apply one target schedule
#
# | | Array arithmetic | Sequential engine |
# |---|---|---|
# | What it computes | Lagged target weights times asset returns, summed across assets | Orders, fills, share counts and a cash balance, bar by bar |
# | Where it is quick | Sweeping a grid of parameters, because the whole history is one matrix operation | Nowhere in particular; each run walks every bar |
# | What it makes you state | Nothing. Timing, fill price and costs are whatever the arithmetic implies | Everything. The configuration has to name an execution mode, a commission and a slippage rate |
# | What it cannot express | Anything that depends on the account's own state - a stop, an order that fails, running out of cash | - |
#
# The array path here is deliberately the plain version: lagged weights times returns. A vectorized
# portfolio library such as VectorBT can express next-bar fills and per-trade costs too, and doing
# so brings the two much closer together. What this notebook isolates is the arithmetic itself, not
# the ceiling of any particular library.

# %% [markdown]
# ## Setup

# %%
"""ETF momentum protocol parity - compare array and sequential implementations."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from ml4t.backtest import (
    BacktestConfig,
    DataFeed,
    Engine,
    ExecutionMode,
    Strategy,
)
from ml4t.diagnostic.evaluation import PortfolioAnalysis

from data import load_etfs, load_macro
from utils.style import COLORS

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides after this cell
START_DATE = "2010-01-01"
END_DATE = "2024-01-01"
LOOKBACK_PERIOD = 126
TOP_N = 3
REGIME_THRESHOLD = 0.005
INITIAL_CASH = 100_000
ENGINE_COMMISSION = 0.0005
ENGINE_SLIPPAGE = 0.0005

# %% [markdown]
# ## 1. What each setting decides
#
# Both runs get the same universe, the same lookback, the same number of positions and the same
# regime threshold, so nothing the ranking does can account for a difference between them.
#
# They are also charged the same amount per dollar traded. The sequential engine splits its charge
# in two: a commission, taken as a percentage of the traded notional, and a slippage rate, applied
# to the fill price. The library adds the two when it reports a total transaction cost, so the
# array path charges that sum in one piece. Aligning the totals is what leaves the execution
# protocol - the moment a target is acted on, the price it fills at, and whether whole shares and a
# cash balance have to be respected - as the only thing that can separate the two results.
#
# The ETF basket is fixed and hand-picked, and the macro history is a present-day snapshot rather
# than a record of what was published at the time, so the numbers below describe this sample and
# are not an estimate of what the rule would earn live.

# %%
ETF_SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "TLT", "GLD", "VNQ", "DBC"]
ARRAY_TURNOVER_COST = ENGINE_COMMISSION + ENGINE_SLIPPAGE

print("Strategy, shared by both runs")
print(f"  Universe: {len(ETF_SYMBOLS)} US-listed ETFs")
print(f"  Lookback window: {LOOKBACK_PERIOD} sessions")
print(f"  Positions held in a risk-on month: {TOP_N}, equally weighted")
print(f"  Risk-on while the 10Y-2Y spread exceeds: {REGIME_THRESHOLD:.2%}")
print(f"  Sample: {START_DATE} through {END_DATE}")
print("Cost per dollar traded, matched between the two runs")
print(f"  Sequential engine: {ENGINE_COMMISSION * 10_000:.0f} bps commission")
print(f"                   + {ENGINE_SLIPPAGE * 10_000:.0f} bps slippage")
print(f"  Array arithmetic:   {ARRAY_TURNOVER_COST * 10_000:.0f} bps on turnover")

# %% [markdown]
# ## 2. Load the data both runs share

# %%
etf_pl = load_etfs()
etf_pl = etf_pl.with_columns(pl.col("timestamp").dt.date().alias("timestamp"))
etf_pl = etf_pl.filter(
    (pl.col("symbol").is_in(ETF_SYMBOLS))
    & (pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    & (pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
)
assert etf_pl.n_unique(["symbol", "timestamp"]) == len(etf_pl)
required_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
assert set(required_columns) <= set(etf_pl.columns)
assert etf_pl.select(pl.all_horizontal(pl.col(required_columns).is_not_null()).all()).item()
assert etf_pl.select((pl.col("close") > 0).all()).item()
assert etf_pl.select((pl.col("volume") >= 0).all()).item()

# %% [markdown]
# Pivot the canonical long panel at the pandas boundary used by the array
# implementation. Forward filling carries only observations already available.

# %%
close_prices_pl = etf_pl.pivot(on="symbol", index="timestamp", values="close").sort("timestamp")
close_prices = close_prices_pl.to_pandas()
close_prices.set_index("timestamp", inplace=True)

available_symbols = [s for s in ETF_SYMBOLS if s in close_prices.columns]
assert available_symbols == ETF_SYMBOLS, (
    f"missing from the panel: {set(ETF_SYMBOLS) - set(available_symbols)}"
)
close_prices = close_prices[ETF_SYMBOLS].ffill()
assert not close_prices.isna().any().any()

print(f"Panel: {len(close_prices):,} sessions x {len(ETF_SYMBOLS)} ETFs")
print(f"First session: {close_prices.index.min():%Y-%m-%d}")
print(f"Last session:  {close_prices.index.max():%Y-%m-%d}")

# %%
macro_df = load_macro()
assert macro_df.n_unique("timestamp") == len(macro_df)

yield_curve = (
    macro_df.select("timestamp", (pl.col("YIELD_CURVE_SLOPE") / 100).alias("slope"))
    .drop_nulls()
    .to_pandas()
    .set_index("timestamp")
)
yield_curve_aligned = yield_curve.reindex(close_prices.index, method="ffill")
assert not yield_curve_aligned["slope"].isna().any()
regime = (yield_curve_aligned["slope"] > REGIME_THRESHOLD).astype(int)

print(f"Yield-curve observations available: {len(yield_curve):,}")
print(f"Risk-on share of ETF sessions: {regime.mean():.1%}")

# %% [markdown]
# ## 3. Build one set of target weights
#
# Everything from here to the end of this section is shared. The weight matrix it produces is the
# single input both simulations are given, so any difference between their results has to come
# from what each does with it.

# %%
daily_returns = close_prices.pct_change()
cumulative_return = close_prices.pct_change(LOOKBACK_PERIOD)
realized_vol = daily_returns.rolling(LOOKBACK_PERIOD).std() * np.sqrt(252)
momentum_score = cumulative_return / realized_vol
momentum_rank = momentum_score.rank(axis=1, ascending=False)

# %%
weights = pd.DataFrame(np.nan, index=close_prices.index, columns=ETF_SYMBOLS)
month = close_prices.index.to_period("M")
rebalance_dates = pd.DatetimeIndex(close_prices.index.to_series().groupby(month).last())

first_valid_momentum = momentum_rank.dropna(how="all").index[0]
first_day = close_prices.index[0]
weights.loc[first_day, :] = 0.0
weights.loc[first_day, "AGG"] = 0.60
weights.loc[first_day, "TLT"] = 0.40

for date in rebalance_dates:
    weights.loc[date, :] = 0.0
    if date < first_valid_momentum or date not in momentum_rank.index:
        weights.loc[date, "AGG"] = 0.60
        weights.loc[date, "TLT"] = 0.40
        continue

    if regime.loc[date] == 1:
        ranks = momentum_rank.loc[date]
        top_n_etfs = ranks[ranks <= TOP_N].index.tolist()
        weight = 1.0 / len(top_n_etfs) if len(top_n_etfs) > 0 else 0
        for etf in top_n_etfs:
            weights.loc[date, etf] = weight
    else:
        weights.loc[date, "AGG"] = 0.60
        weights.loc[date, "TLT"] = 0.40

weights = weights.ffill()
assert not weights.isna().any().any()
np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, rtol=0, atol=1e-12)

# %% [markdown]
# ## 4. Run it as arithmetic
#
# Three lines carry the whole simulation. Lag the weight matrix by one row, so the weight in force
# over a day is the one the previous close produced. Multiply by the matrix of daily returns and
# sum across assets to get the portfolio return for that day. Subtract a cost proportional to how
# much the target moved.
#
# What this never represents is an account. There are no shares, no cash, and no fill: the
# portfolio is assumed to be at its target continuously and to earn the close-to-close return of
# whatever it targets, which is a different thing from having bought it.

# %%
shifted_weights_vbt = weights.shift(1).fillna(0)
returns_vbt = close_prices.pct_change().fillna(0)
portfolio_returns_vbt = (shifted_weights_vbt * returns_vbt).sum(axis=1)

# A target chosen at t is executed for the t+1 return interval.
target_turnover = weights.diff().fillna(weights).abs().sum(axis=1)
executed_turnover = target_turnover.shift(1).fillna(0)
cost_drag = executed_turnover * ARRAY_TURNOVER_COST

portfolio_returns_vbt_net = portfolio_returns_vbt - cost_drag
equity_vbt = INITIAL_CASH * (1 + portfolio_returns_vbt_net).cumprod()

print(f"Array arithmetic, ending equity: ${equity_vbt.iloc[-1]:,.0f}")
print(f"Turnover charged over the sample: {executed_turnover.sum():.1f}x portfolio value")

# %% [markdown]
# ## 5. Run it as an account
#
# The engine walks the price history one bar at a time and calls a strategy object on each one. The
# strategy here does nothing except read the target weights it is handed and ask the broker to move
# the account to them; the weights arrive through the engine's context frame, which is the same
# matrix section 3 built. Order generation, position sizing and fill simulation are the engine's
# job, and `ExecutionMode.NEXT_BAR` is what tells it that an order placed on one bar fills at the
# next bar's open.


# %%
class WeightRebalanceStrategy(Strategy):
    """Rebalance to precomputed target weights from context.

    Uses broker.rebalance_to_weights() - the Engine handles position
    sizing, order generation, and fill simulation internally.
    """

    def __init__(self, assets):
        self.assets = assets

    def on_data(self, timestamp, data, context, broker):
        target_weights = {}
        for asset in self.assets:
            w = context.get(f"w_{asset}", 0.0)
            if w is not None and not np.isnan(w):
                target_weights[asset] = w
        broker.rebalance_to_weights(target_weights)


# %%
# Prepare data for ml4t-backtest Engine (requires long-format prices)
etf_long = etf_pl.select(
    [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
)

# Build context_df with target weights per timestamp
weight_records = []
for date in weights.index:
    row = {"timestamp": pd.Timestamp(date)}
    for sym in ETF_SYMBOLS:
        w = weights.loc[date, sym]
        row[f"w_{sym}"] = float(w) if not np.isnan(w) else 0.0
    weight_records.append(row)

context_df = pl.DataFrame(weight_records).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
# Align context timestamps with price timestamps
etf_long = etf_long.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

# %%
# Run ml4t-backtest Engine
feed = DataFeed(prices_df=etf_long, context_df=context_df)
strategy = WeightRebalanceStrategy(assets=ETF_SYMBOLS)

config = BacktestConfig(
    initial_cash=INITIAL_CASH,
    execution_mode=ExecutionMode.NEXT_BAR,
    commission_rate=ENGINE_COMMISSION,
    slippage_rate=ENGINE_SLIPPAGE,
    calendar="NYSE",
)
engine = Engine(feed=feed, strategy=strategy, config=config)

results_ml4t = engine.run()

ec_ml4t = results_ml4t["equity"]
equity_ml4t = pd.Series(
    ec_ml4t.values, index=[t.date() if hasattr(t, "timestamp") else t for t in ec_ml4t.timestamps]
)
portfolio_returns_ml4t = equity_ml4t.pct_change().dropna()

print(f"Sequential account, ending equity: ${equity_ml4t.iloc[-1]:,.0f}")
# results["trades"] is not a fill count: the engine appends a Trade only when a position
# is closed, flipped or scaled DOWN, plus every position still open on the last session,
# marked to market. Opens and scale-ups are absent from it. num_fills is the count of
# fills. Both are printed because they answer different questions.
print(f"Fills: {results_ml4t['num_fills']:,}")
print(f"Closed, reduced or still-open positions: {len(results_ml4t['trades']):,}")

# %% [markdown]
# Align both return streams before computing any comparison statistic. This
# removes date-set differences from the framework comparison.

# %%
array_returns = portfolio_returns_vbt_net.copy()
engine_returns = portfolio_returns_ml4t.copy()
array_returns.index = pd.to_datetime(array_returns.index)
engine_returns.index = pd.to_datetime(engine_returns.index)
comparison_returns = pd.concat(
    {"array_based": array_returns, "ml4t_backtest": engine_returns}, axis=1, join="inner"
).dropna()
assert comparison_returns.index.is_unique
assert not comparison_returns.isna().any().any()

# %% [markdown]
# ## 6. Compare the two with one metric implementation
#
# Both return streams go through the same `PortfolioAnalysis` object, so Sharpe, Sortino and
# drawdown are computed by identical code on both sides. That matters more than it sounds: two
# backtesting libraries reporting different Sharpe ratios for the same returns is a common and
# entirely uninformative finding, caused by different annualization or risk-free conventions rather
# than by the strategies. Comparing engines means comparing their return series, not their reports.

# %%
vbt_analysis = PortfolioAnalysis(
    returns=comparison_returns["array_based"].to_numpy(), periods_per_year=252
)
ml4t_analysis = PortfolioAnalysis(
    returns=comparison_returns["ml4t_backtest"].to_numpy(), periods_per_year=252
)

vbt_stats = vbt_analysis.compute_summary_stats()
ml4t_stats = ml4t_analysis.compute_summary_stats()

metrics_labels = [
    "Total Return (%)",
    "CAGR (%)",
    "Volatility (%)",
    "Sharpe",
    "Sortino",
    "Max Drawdown (%)",
]
metric_scales = [100, 100, 100, 1, 1, 100]
vbt_vals = [
    vbt_stats.total_return,
    vbt_stats.annual_return,
    vbt_stats.annual_volatility,
    vbt_stats.sharpe_ratio,
    vbt_stats.sortino_ratio,
    vbt_stats.max_drawdown,
]
ml4t_vals = [
    ml4t_stats.total_return,
    ml4t_stats.annual_return,
    ml4t_stats.annual_volatility,
    ml4t_stats.sharpe_ratio,
    ml4t_stats.sortino_ratio,
    ml4t_stats.max_drawdown,
]

# %% [markdown]
# The table uses the same common-date returns and the same metric implementation.
# Execution and friction assumptions remain intentionally different.

# %%
parity_df = pl.DataFrame(
    {
        "metric": metrics_labels,
        "array_based": [float(v) * scale for v, scale in zip(vbt_vals, metric_scales, strict=True)],
        "ml4t_backtest": [
            float(value) * scale for value, scale in zip(ml4t_vals, metric_scales, strict=True)
        ],
        "array_minus_engine": [
            (float(vbt) - float(engine)) * scale
            for vbt, engine, scale in zip(vbt_vals, ml4t_vals, metric_scales, strict=True)
        ],
    }
)
parity_df

# %%
vbt_total = vbt_stats.total_return
ml4t_total = ml4t_stats.total_return
rel_diff = abs(vbt_total - ml4t_total) / max(abs(vbt_total), 0.0001)
ending_gap = float(equity_vbt.iloc[-1] - equity_ml4t.iloc[-1])
correlation = comparison_returns.corr().iloc[0, 1]

print(f"Common daily returns compared: {len(comparison_returns):,}")
print(f"Correlation between the two return series: {correlation:.4f}")
print(f"Total return, relative difference: {rel_diff:.1%}")
print(f"Ending equity, array minus sequential: ${ending_gap:,.0f}")

# %% [markdown]
# Two things to read off those four numbers, in this order.
#
# The correlation is a statement about days, and it is close to one because both series react to
# the same universe holding almost the same targets. It says the two runs agree about which days
# were good and roughly how good. It says nothing about the level: a small constant difference in
# what each day earns is invisible to a correlation and compounds into everything.
#
# The relative difference in total return is the part that matters, and it is what compounding a
# small daily gap over fourteen years does. Both runs pay the same cost per dollar traded, so what
# is left is the execution protocol: one earns the close-to-close return of a target it is assumed
# to already hold, the other has to buy it at the next open, in whole shares, with the cash it has.
# `07_engine_divergence_anatomy` takes those apart one at a time; here they act together and the
# split between them is not identified.

# %% [markdown]
# ## 7. What the gap looks like over time
#
# The first chart puts both equity curves on one axis. The second plots the difference between
# them, which is the only way to see a slowly widening gap between two lines that look identical.

# %%
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=equity_vbt.index,
        y=equity_vbt,
        name="Array arithmetic",
        line=dict(color=COLORS["blue"], width=2),
    )
)

fig.add_trace(
    go.Scatter(
        x=equity_ml4t.index,
        y=equity_ml4t,
        name="Sequential engine",
        line=dict(color=COLORS["neutral"], width=2, dash="dash"),
    )
)

fig.update_layout(
    title=(
        "Two simulations of one strategy, matched on cost per dollar traded"
        "<br><sup>US-listed ETF basket; monthly rebalance; equity in USD, "
        "starting from the same cash</sup>"
    ),
    xaxis_title="Date",
    yaxis_title="Portfolio equity (USD)",
    height=500,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)
fig.show()

# %%
# Difference over time (align index types)
eq_vbt_dt = equity_vbt.copy()
eq_ml4t_dt = equity_ml4t.copy()
eq_vbt_dt.index = pd.to_datetime(eq_vbt_dt.index)
eq_ml4t_dt.index = pd.to_datetime(eq_ml4t_dt.index)
shared_eq = eq_vbt_dt.index.intersection(eq_ml4t_dt.index)
diff_series = eq_vbt_dt.loc[shared_eq] - eq_ml4t_dt.loc[shared_eq]

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=diff_series.index,
        y=diff_series,
        name="Array minus sequential",
        line=dict(color=COLORS["copper"], width=2),
    )
)

fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])

fig.update_layout(
    title=(
        "The difference between the two accumulates rather than oscillating"
        "<br><sup>Array equity minus sequential equity, in USD; positive means the "
        "arithmetic is ahead</sup>"
    ),
    xaxis_title="Date",
    yaxis_title="Array minus sequential equity (USD)",
    height=400,
)
fig.show()

# %% [markdown]
# ## 8. What is still different between the two runs
#
# Cost per dollar traded is matched. Five things are not, and each one is a modelling assumption
# somebody made rather than a property of a library.
#
# | | Array arithmetic | Sequential engine |
# |---|---|---|
# | What repositioning costs | The target is re-applied every session for free, and turnover is charged only where the weight matrix changes | Every repositioning is a real order: `on_data` calls `rebalance_to_weights` on every session and each resulting trade pays commission and slippage |
# | When a target is acted on | Immediately, as the weight in force over the next close-to-close return | An order placed on that bar, filled on the next one |
# | What price it gets | The close, implicitly, because that is what the return is measured between | The next bar's open, moved against the order by the slippage rate |
# | What the cost is charged on | The change in the target, whether or not a trade could have been made | The notional actually filled |
# | What the account holds | A weight, which can be any real number | Whole shares, bought with the cash on hand |
#
# The first row is the largest and the least visible, and the easy way to state it is wrong. It is
# not that the engine trades and the array form does not. `(weights.shift(1) * returns).sum(axis=1)`
# re-applies the full target on every session after the first - `weights.shift(1).fillna(0)` leaves
# the array account holding nothing on the opening session - so it is silently restored to its
# weights daily too: prices move, the implied holdings drift, and the next line of the formula puts
# them back. What differs is the bill. `cost_drag` is built from `weights.diff()` and carries the
# same one-session lag, so the array side is charged only on the 78 target changes printed below,
# each billed on the session the new target takes effect and starts earning.
#
# The engine cannot do that. Restoring the target means sending orders, and its counts below show
# the price of the same repositioning: 2,911 of those 3,522 attempted rebalances reach the market
# and produce 6,518 fills, every one paying commission and slippage. Both models reposition daily.
# Only one of them pays for it, and that is what this row costs.
#
# The last row is the one that is easy to miss for a different reason. An account that has to buy
# whole shares with a finite cash balance cannot sit exactly on its target, and the residual is not
# centred on zero: rounding down is the only direction that always fits. That is what generates the
# drift the first row then pays to correct.
#
# These five act together here, and this notebook does not separate them.
# `07_engine_divergence_anatomy` changes one at a time and measures each.

# %%
# fillna(weights) so the opening allocation counts, matching how executed_turnover is
# built at the top of section 5; weights.diff() alone leaves the first row NaN and drops it.
_weight_change_dates = int((weights.diff().fillna(weights).abs().sum(axis=1) > 0).sum())
print(f"Sessions in the panel:                  {len(close_prices):,}")
print(f"Dates the weight matrix changes:        {_weight_change_dates:,}")
print(f"Rebalances the engine attempted:        {len(close_prices):,}")
print(f"Of those, ones that produced a trade:   {results_ml4t['num_rebalance_events']:,}")
print(f"Fills:                                  {results_ml4t['num_fills']:,}")
print(f"Turnover charged on the array side:     {executed_turnover.sum():.1f}x portfolio value")

# %% [markdown]
# ## 9. Which one to reach for
#
# | If the question is | Start with | Because |
# |---|---|---|
# | How does this rule behave over a grid of parameters? | Array arithmetic | The whole history is one matrix operation, and a hundred parameter sets are a hundred cheap ones |
# | Is there any signal here at all? | Array arithmetic | A screening pass does not need fills to be right, only rankings |
# | What would this actually have earned? | Sequential engine | Fills, cash and share counts are the difference between a return series and a track record |
# | What happens when an order does not fill, or a stop triggers? | Sequential engine | The array form has nowhere to put a decision that depends on the account's own state |
# | Do two implementations agree? | Either, once their assumptions are written down | Agreement is a property of the protocol, not of the library |
#
# The order matters: decide the trading protocol first, then pick the implementation that makes
# that protocol easiest to write down and check. Reaching for a library first and inferring the
# protocol from what it happens to do is how the assumptions in section 8 get made by accident.

# %% [markdown]
# ## Key takeaways
#
# 1. **Two runs of one strategy can share a signal, a universe and a cost rate and still finish
#    apart.** What separates them is the execution protocol, and every part of it is a choice: the
#    bar an order is placed on, the price it fills at, the granularity of a position, whether cash
#    is a constraint.
# 2. **Correlation between daily return series is nearly useless as a parity check.** A small,
#    persistent difference in daily return compounds into a large difference in wealth while
#    leaving the correlation essentially at one. Compare levels, and compare them over time.
# 3. **Compare return series, not reports.** Sharpe and drawdown depend on annualization and
#    risk-free conventions that differ between libraries, so two engines can disagree on the
#    statistics of a return series they agree about. Run both through one metric implementation.
# 4. **Neither style is the reference implementation.** The array form is not an approximation of
#    the engine, and the engine is not automatically right. The one to trust is whichever encodes
#    the protocol you intended to test.
#
# ### Known limitations
#
# - Both runs use the same fixed, hand-picked basket of ten funds that exist today, so neither is
#   free of selection or survivorship bias.
# - The gap measured here is the joint effect of four differences acting together. Nothing in this
#   notebook attributes a share of it to any one of them.
# - The sequential engine's slippage is a flat rate on the fill price, which does not depend on
#   order size or on how much volume the market had. Chapter 18 replaces it with a cost model that
#   does.
#
# **Next:** `07_engine_divergence_anatomy` changes one execution assumption at a time and measures
# what each is worth. Section 16.3 covers the speed-fidelity trade-off in depth.
