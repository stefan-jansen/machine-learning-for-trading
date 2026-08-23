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
# # Backtesting First Principles: From Returns to Portfolio Simulation
#
# **Docker image**: `ml4t`
#
# ## Purpose
# Build an auditable portfolio simulator from first principles with Polars and NumPy. The
# implementation makes signal timing, next-open execution, transaction costs, cash constraints,
# and benchmark rebalancing explicit before later notebooks introduce backtesting frameworks.
#
# ## Learning objectives
#
# - Write a period's portfolio return as the weighted sum of its assets' returns, using weights
#   that were already tradable when the period began.
# - Rank ten exchange-traded funds each month from their own trailing prices, so the ranking on
#   any date reads only prices published on or before that date.
# - Simulate the resulting trades at the following session's opening price, charge a fee on every
#   dollar traded, and keep the cash balance from going negative.
# - Put a fixed 60/40 stock-bond portfolio through the same simulator, so the two are compared on
#   identical trading dates, prices and fees.
# - Score the finished backtest against pass-or-fail criteria written down before it was run.
#
# ## Book reference
# Chapter 16, Section 16.2 (protocol specification) and Section 16.4 (the non-ML baseline).
#
# ## Prerequisites
#
# - Chapter 6 strategy term-sheet conventions.
# - Familiarity with return, volatility, Sharpe ratio, and drawdown calculations.
#
# Every date in the sample is reported, so nothing here is a holdout estimate: the rule was never
# scored on data it had not already been shown. What protects the exercise instead is that the
# configuration is fixed before the run and is never adjusted to improve what the run produced.

# %% [markdown]
# ## 1. Setup and protocol
#
# Two pieces of information drive the portfolio, both read at the close of the last trading day of
# each month. The first is each fund's own price history. The second is the 10Y-2Y Treasury spread:
# the yield on a ten-year US Treasury note minus the yield on a two-year note. A wide spread means
# long borrowing costs more than short borrowing, which is the ordinary shape of the curve and is
# taken here as the *risk-on* state; a spread at or below the threshold, including an inverted one
# where short rates exceed long rates, is the *risk-off* state.
#
# Orders execute at the next trading day's open. That one-session gap is what keeps the backtest
# honest: a return that begins before the signal existed can never be earned by the strategy. Both
# the momentum portfolio and the benchmark rebalance on those same dates.

# %%
"""Backtesting first principles with point-in-time signals and next-open execution."""

import hashlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import plotly.graph_objects as go
import polars as pl
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.metrics import sharpe_ratio, sortino_ratio
from plotly.subplots import make_subplots

from data import load_etfs, load_macro
from utils import ML4T_DATA_PATH
from utils.style import COLORS

# %% tags=["parameters"]
START_DATE = "2010-01-01"
END_DATE = "2024-01-01"
LOOKBACK_PERIOD = 126
TOP_N = 3
REGIME_THRESHOLD = 0.005
INITIAL_CASH = 100_000.0
FEES = 0.0005

# %%
ETF_UNIVERSE = {
    "SPY": "US Large Cap Equity",
    "QQQ": "US Tech Equity",
    "IWM": "US Small Cap Equity",
    "EFA": "Developed International Equity",
    "EEM": "Emerging Markets Equity",
    "AGG": "US Aggregate Bonds",
    "TLT": "US Long Treasury",
    "GLD": "Gold",
    "VNQ": "US Real Estate",
    "DBC": "Commodities",
}
ETF_SYMBOLS = list(ETF_UNIVERSE)
DEFENSIVE_MIX = {"AGG": 0.60, "TLT": 0.40}
BENCHMARK_MIX = {"SPY": 0.60, "AGG": 0.40}

# %% [markdown]
# ### What each setting decides
#
# **Lookback.** Each fund is scored on its return over a trailing window of daily closes, divided
# by how volatile it was over that same window. The window has to be long enough that a trend is
# distinguishable from noise, and short enough that the score still describes the present rather
# than last year. Roughly half a year of sessions is the usual compromise.
#
# **Breadth.** A risk-on month holds an equally weighted slice of the highest-scoring funds.
# Holding more of them dilutes whatever the ranking knows; holding fewer makes the result turn on
# one fund. Three out of ten keeps the portfolio concentrated enough to express the ranking while
# still spreading a single blow-up across two other positions.
#
# **Regime threshold.** The threshold sits a little above zero rather than at zero, so that a curve
# which has flattened to nearly nothing counts as risk-off. A flat curve is the state that
# historically precedes an inversion, and waiting for the sign to actually change would put the
# portfolio in equities through it.
#
# **Fees.** Every dollar bought or sold pays a proportional fee. The level stands in for a
# commission plus roughly half the bid-ask spread on a liquid US-listed ETF.
#
# **Defensive and benchmark mixes.** Risk-off months hold aggregate bonds and long Treasuries. The
# benchmark is the standard 60/40 stock-bond portfolio, rebalanced on the same dates so that the
# comparison isolates the selection rule and nothing else.

# %%
print("Signal")
print(f"  Lookback window: {LOOKBACK_PERIOD} sessions of daily closes")
print(f"  Held in a risk-on month: {TOP_N} of {len(ETF_SYMBOLS)} funds, equally weighted")
print("Regime")
print(f"  Risk-on while the 10Y-2Y spread exceeds: {REGIME_THRESHOLD:.2%}")
defensive = ", ".join(f"{w:.0%} {s}" for s, w in DEFENSIVE_MIX.items())
benchmark = ", ".join(f"{w:.0%} {s}" for s, w in BENCHMARK_MIX.items())
print(f"  Risk-off allocation: {defensive}")
print("Trading")
print("  Rebalance: last session of each month, filled at the next session's open")
print(f"  Fee per dollar traded: {FEES * 10_000:.0f} basis points")
print(f"  Starting capital: ${INITIAL_CASH:,.0f}")
print("Comparison")
print(f"  Benchmark: {benchmark}, rebalanced on the same dates")
print(f"  Sample: {START_DATE} through {END_DATE}")

# %% [markdown]
# **Universe limitation.** This fixed set is a current, hand-curated teaching universe, not a
# historical constituent list. All ten ETFs had valid observations at the 2010 start, but selecting
# familiar funds with data available today creates selection and survivorship limitations. Results
# therefore describe these ten surviving funds and are not a survivorship-free universe estimate.

# %% [markdown]
# ## 2. Load and validate the market panel
#
# The canonical loader returns long-form Polars data keyed by `symbol` and `timestamp`. Open prices
# determine fills; close prices determine end-of-day portfolio value. Forward filling only carries
# the last observed quote forward, and the complete-case filter removes any pre-inception rows.

# %%
etf_long = load_etfs(
    symbols=ETF_SYMBOLS,
    start_date=START_DATE,
    end_date=END_DATE,
)

duplicate_keys = etf_long.group_by(["symbol", "timestamp"]).len().filter(pl.col("len") > 1).height
assert duplicate_keys == 0, f"Found {duplicate_keys} duplicate ETF keys"

# %%
open_wide = etf_long.pivot(on="symbol", index="timestamp", values="open").sort("timestamp")
close_wide = etf_long.pivot(on="symbol", index="timestamp", values="close").sort("timestamp")
available_symbols = [
    symbol for symbol in ETF_SYMBOLS if symbol in open_wide.columns and symbol in close_wide.columns
]
assert available_symbols == ETF_SYMBOLS, "The configured ETF universe is incomplete"

# %%
price_panel = (
    open_wide.select(
        "timestamp",
        *(pl.col(symbol).alias(f"{symbol}_open") for symbol in ETF_SYMBOLS),
    )
    .join(
        close_wide.select(
            "timestamp",
            *(pl.col(symbol).alias(f"{symbol}_close") for symbol in ETF_SYMBOLS),
        ),
        on="timestamp",
        how="inner",
    )
    .with_columns(pl.exclude("timestamp").forward_fill())
    .drop_nulls()
    .sort("timestamp")
)

dates = price_panel["timestamp"].to_list()
open_prices = price_panel.select(f"{symbol}_open" for symbol in ETF_SYMBOLS).to_numpy()
close_prices = price_panel.select(f"{symbol}_close" for symbol in ETF_SYMBOLS).to_numpy()

assert np.isfinite(open_prices).all() and np.isfinite(close_prices).all()
assert (open_prices > 0).all() and (close_prices > 0).all()

print(f"Loaded {len(dates):,} complete daily bars for {len(ETF_SYMBOLS)} symbols")
print(f"Date range: {dates[0]} to {dates[-1]}")

# %% [markdown]
# Before any score is computed, this is what the ten funds are and how much history the loader
# returned for each. A cross-sectional ranking only means something if every candidate is quoted on
# the ranking date, so what to look for here is a common first bar and an identical bar count: any
# fund starting late would be scored against a shorter window than its rivals, or excluded from the
# early rankings altogether.

# %%
exposures = pl.DataFrame(
    {"symbol": ETF_SYMBOLS, "exposure": [ETF_UNIVERSE[symbol] for symbol in ETF_SYMBOLS]}
)
universe_table = (
    etf_long.group_by("symbol")
    .agg(
        pl.col("timestamp").min().alias("first_bar"),
        pl.col("timestamp").max().alias("last_bar"),
        pl.len().alias("bars"),
    )
    .join(exposures, on="symbol")
    .select("symbol", "exposure", "first_bar", "last_bar", "bars")
    .sort("symbol")
)
universe_table

# %% [markdown]
# ## 3. Align the yield-curve regime point in time
#
# The two series keep different calendars: the ETFs trade on exchange sessions, the Treasury
# yields are published by the Federal Reserve on its own schedule. A *backward as-of join* pairs
# each ETF date with the most recent yield observation dated on or before it, and never with a
# later one. Because orders execute at the following open, the month-end yield reading is already
# published when the fill happens.
#
# What the join cannot fix is the vintage of the file. The local FRED extract is a single current
# snapshot: every series holds the value FRED publishes for that date *today*. Treasury yields are
# revised rarely and by small amounts, so the effect on this rule is minor, but the file is still a
# present-day view of the past. The two facts printed below - how far the snapshot's coverage runs,
# and the hash of the file this run read - are what lets a later run be compared with this one.

# %%
macro_df = load_macro(start_date=START_DATE, end_date=END_DATE)
fred_snapshot_path = ML4T_DATA_PATH / "macro" / "fred_macro.parquet"
fred_snapshot_end = pl.read_parquet(fred_snapshot_path, columns=["date"])["date"].max()
fred_snapshot_hash = hashlib.sha256(fred_snapshot_path.read_bytes()).hexdigest()[:12]

print(f"FRED snapshot covers observations through {fred_snapshot_end}")
print(f"FRED snapshot content hash: sha256:{fred_snapshot_hash}")

yield_curve = macro_df.select(
    "timestamp",
    (pl.col("YIELD_CURVE_SLOPE") / 100).alias("slope"),
).drop_nulls()

# %%
regime_panel = (
    price_panel.select("timestamp")
    .join_asof(yield_curve.sort("timestamp"), on="timestamp", strategy="backward")
    .drop_nulls()
)
assert regime_panel.height == price_panel.height, "Yield-curve history does not cover the panel"

yield_curve_slope = regime_panel["slope"].to_numpy()
regime = yield_curve_slope > REGIME_THRESHOLD
risk_on_share = float(regime.mean())

print(f"Risk-on days: {regime.sum():,} ({risk_on_share:.1%})")
print(f"Risk-off days: {(~regime).sum():,} ({1 - risk_on_share:.1%})")

# %% [markdown]
# ## 4. Compute trailing risk-adjusted momentum
#
# For ETF $i$, the month-end score uses only closes through date $t$:
#
# $$
# m_{i,t} = \frac{P_{i,t}/P_{i,t-L}-1}{\operatorname{sd}(r_{i,t-L+1:t})\sqrt{252}},
# \qquad L=126.
# $$
#
# The cross-sectional rank is formed independently at each date. No full-sample statistic enters
# the score.

# %%
close_frame = price_panel.select(
    "timestamp",
    *(pl.col(f"{symbol}_close").alias(symbol) for symbol in ETF_SYMBOLS),
)

daily_returns_frame = close_frame.select(
    "timestamp",
    *((pl.col(symbol) / pl.col(symbol).shift(1) - 1).alias(symbol) for symbol in ETF_SYMBOLS),
)

# %%
momentum_frame = close_frame.select(
    "timestamp",
    *(
        (
            (pl.col(symbol) / pl.col(symbol).shift(LOOKBACK_PERIOD) - 1)
            / (
                (pl.col(symbol) / pl.col(symbol).shift(1) - 1).rolling_std(LOOKBACK_PERIOD)
                * np.sqrt(252)
            )
        ).alias(symbol)
        for symbol in ETF_SYMBOLS
    ),
)
momentum_scores = momentum_frame.select(ETF_SYMBOLS).to_numpy()

first_valid_idx = int(np.flatnonzero(np.isfinite(momentum_scores).sum(axis=1) >= TOP_N)[0])
print(f"First valid momentum date: {dates[first_valid_idx]}")

# %% [markdown]
# The scores on the last date in the sample, highest at the top. The blue bars are the funds a
# risk-on month would buy. What matters is the spacing rather than the order: where the top scores
# are bunched together the ranking is close to arbitrary, and a small change in the lookback window
# would reshuffle which three are held.

# %%
latest_scores = pl.DataFrame({"symbol": ETF_SYMBOLS, "momentum_score": momentum_scores[-1]}).sort(
    "momentum_score", descending=True
)
latest_top = latest_scores.head(TOP_N)["symbol"].to_list()

fig = go.Figure(
    go.Bar(
        x=latest_scores["momentum_score"],
        y=latest_scores["symbol"],
        orientation="h",
        marker_color=[
            COLORS["blue"] if symbol in latest_top else COLORS["neutral"]
            for symbol in latest_scores["symbol"]
        ],
    )
)
fig.update_layout(
    title=f"The {TOP_N} highest scores are the funds the rule would hold",
    xaxis_title=f"{LOOKBACK_PERIOD}-session return per unit of annualized volatility",
    yaxis_title="ETF symbol",
    height=430,
    yaxis=dict(autorange="reversed"),
    showlegend=False,
)
fig.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"])
fig.show()

# %% [markdown]
# ## 5. Generate target weights that were tradable when the period began
#
# The signal is formed from the closes of the last session of each month, so it cannot be acted on
# until the next session opens. Shifting the whole weight matrix down by one row is what encodes
# that: row $k$ of the executed matrix holds the target the previous session's close produced. The
# very first row is the defensive mix, because on day one no trailing score exists yet.
#
# The check below is against the calendar rather than against the construction. A monthly rule run
# over this sample has to fill once in every month it covers, plus once on the opening day, and any
# other count means the month boundaries were misidentified.

# %%
month_end_signal = np.zeros(len(dates), dtype=bool)
for idx in range(len(dates) - 1):
    month_end_signal[idx] = (dates[idx].year, dates[idx].month) != (
        dates[idx + 1].year,
        dates[idx + 1].month,
    )
month_end_signal[-1] = True

defensive_weights = np.zeros(len(ETF_SYMBOLS))
for symbol, weight in DEFENSIVE_MIX.items():
    defensive_weights[ETF_SYMBOLS.index(symbol)] = weight

# %%
signal_weights = np.zeros_like(close_prices)
current_target = defensive_weights.copy()

for idx in range(len(dates)):
    if month_end_signal[idx]:
        current_target = defensive_weights.copy()
        valid = np.flatnonzero(np.isfinite(momentum_scores[idx]))
        if idx >= first_valid_idx and regime[idx] and len(valid) >= TOP_N:
            ordered = valid[np.argsort(momentum_scores[idx, valid])]
            selected = ordered[-TOP_N:]
            current_target = np.zeros(len(ETF_SYMBOLS))
            current_target[selected] = 1 / TOP_N
    signal_weights[idx] = current_target

# %%
execution_weights = np.vstack([defensive_weights, signal_weights[:-1]])
rebalance_at_open = np.r_[True, month_end_signal[:-1]]

calendar_months = {(day.year, day.month) for day in dates}
assert np.allclose(execution_weights.sum(axis=1), 1.0)
assert rebalance_at_open.sum() == len(calendar_months), "one fill per month, plus the first buy"
print(f"Calendar months in the sample: {len(calendar_months)}")
print(f"Next-open rebalances, including the opening purchase: {rebalance_at_open.sum()}")

# %% [markdown]
# ## 6. Read the regime and the allocation it produced
#
# The two panels share a time axis, so they can be read together: the spread on top, the weights it
# produced underneath. A heatmap carries the weights because ten funds would need ten competing
# colors as lines, and because what the reader is judging is a pattern of presence and absence
# rather than a level. Dark bands on the AGG and TLT rows are the defensive months; a dark band on
# three equity rows at once is a momentum rotation. The dashed line on the top panel is the
# threshold, and every stretch below it should line up with bonds below.

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.35, 0.65],
    subplot_titles=["10Y-2Y Treasury spread", "Target weights, dated to the session they fill on"],
)
fig.add_trace(
    go.Scatter(x=dates, y=yield_curve_slope * 100, line=dict(color=COLORS["blue"])), row=1, col=1
)
fig.add_hline(
    y=REGIME_THRESHOLD * 100, line_dash="dash", line_color=COLORS["neutral"], row=1, col=1
)
fig.add_trace(
    go.Heatmap(
        x=dates,
        y=ETF_SYMBOLS,
        z=execution_weights.T,
        zmin=0,
        zmax=float(execution_weights.max()),
        colorscale=[[0, COLORS["silver"]], [1, COLORS["blue"]]],
        colorbar=dict(title="Weight", tickformat=".0%"),
    ),
    row=2,
    col=1,
)
fig.update_layout(
    title="The portfolio sits in bonds whenever the spread is under the threshold",
    height=700,
    showlegend=False,
)
fig.update_yaxes(title_text="Spread (percentage points)", row=1, col=1)
fig.update_yaxes(title_text="ETF symbol", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.show()

# %% [markdown]
# ## 7. The fundamental return equation
#
# If weights are executable before period $t$ begins, portfolio return is the inner product
#
# $$
# R_{p,t} = \sum_{i=1}^{N} w_{i,t}r_{i,t} = \mathbf{w}_t^\top\mathbf{r}_t.
# $$
#
# A close-derived signal cannot use this equation with the same close-to-close return. The
# simulator below therefore waits until the next open, then values the resulting holdings at each
# close. This explicit event order is the protection against same-bar look-ahead.

# %%
toy_weights = np.array([0.60, 0.40])
toy_asset_returns = np.array([0.01, -0.0025])
toy_portfolio_return = float(toy_weights @ toy_asset_returns)
print(f"Toy portfolio return: {toy_portfolio_return:.3%}")

# %% [markdown]
# ## 8. Simulate next-open fills with a cash constraint
#
# On every scheduled rebalance, the simulator sells first, deducts fees, and then scales purchases
# to the remaining cash. End-of-day equity is marked at the close. This ordering prevents the fee
# financing and negative-cash behavior that a simultaneous full-notional rebalance can introduce.


# %%
def simulate_portfolio(
    opens: np.ndarray,
    closes: np.ndarray,
    target_weights: np.ndarray,
    rebalance_mask: np.ndarray,
    initial_cash: float,
    fee_rate: float,
) -> dict[str, np.ndarray]:
    """Simulate long-only next-open rebalancing with proportional fees."""
    n_bars, n_assets = closes.shape
    holdings = np.zeros(n_assets)
    cash = initial_cash
    equity = np.zeros(n_bars)
    cash_path = np.zeros(n_bars)
    for idx in range(n_bars):
        if rebalance_mask[idx]:
            open_values = holdings * opens[idx]
            open_equity = cash + open_values.sum()
            target_values = target_weights[idx] * open_equity
            sells = np.maximum(open_values - target_values, 0.0)
            holdings -= sells / opens[idx]
            cash += sells.sum() * (1 - fee_rate)
            remaining_values = holdings * opens[idx]
            requested_buys = np.maximum(target_values - remaining_values, 0.0)
            required_cash = requested_buys.sum() * (1 + fee_rate)
            buy_scale = min(1.0, cash / required_cash) if required_cash > 0 else 0.0
            executed_buys = requested_buys * buy_scale
            holdings += executed_buys / opens[idx]
            cash -= executed_buys.sum() * (1 + fee_rate)
        if cash < -1e-8:
            raise RuntimeError(f"Cash constraint violated at bar {idx}: {cash}")
        cash = max(cash, 0.0)
        cash_path[idx] = cash
        equity[idx] = cash + float(holdings @ closes[idx])

    return {"equity": equity, "cash": cash_path}


# %% [markdown]
# The momentum strategy and benchmark use the same starting capital, fee rate, data window, and
# monthly next-open rebalance mask. Only their target-weight rules differ.

# %%
strategy_result = simulate_portfolio(
    open_prices,
    close_prices,
    execution_weights,
    rebalance_at_open,
    INITIAL_CASH,
    FEES,
)

benchmark_weights = np.zeros_like(close_prices)
for symbol, weight in BENCHMARK_MIX.items():
    benchmark_weights[:, ETF_SYMBOLS.index(symbol)] = weight
benchmark_result = simulate_portfolio(
    open_prices,
    close_prices,
    benchmark_weights,
    rebalance_at_open,
    INITIAL_CASH,
    FEES,
)

# %%
portfolio_value = strategy_result["equity"]
benchmark_value = benchmark_result["equity"]
portfolio_returns = (
    np.diff(np.r_[INITIAL_CASH, portfolio_value]) / np.r_[INITIAL_CASH, portfolio_value][:-1]
)
benchmark_returns = (
    np.diff(np.r_[INITIAL_CASH, benchmark_value]) / np.r_[INITIAL_CASH, benchmark_value][:-1]
)

assert strategy_result["cash"].min() >= 0
assert benchmark_result["cash"].min() >= 0
print(f"Final momentum value: ${portfolio_value[-1]:,.2f}")
print(f"Final 60/40 value: ${benchmark_value[-1]:,.2f}")
print(f"Minimum cash balance: ${strategy_result['cash'].min():,.2f}")

# %% [markdown]
# ## 9. Compute one internally consistent metric set
#
# Sharpe and Sortino use mean periodic excess return with a zero annual risk-free rate, matching
# `PortfolioAnalysis`. CAGR remains a separate growth metric rather than being substituted into the
# Sharpe numerator.


# %%
def calculate_metrics(returns: np.ndarray, periods_per_year: int = 252) -> dict[str, float]:
    """Calculate growth, risk, and risk-adjusted performance metrics."""
    total_return = float(np.prod(1 + returns) - 1)
    annual_return = float((1 + total_return) ** (periods_per_year / len(returns)) - 1)
    annual_vol = float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(np.r_[1.0, cumulative])[1:]
    max_dd = float(np.min(cumulative / running_max - 1))

    return {
        "Total Return (%)": total_return * 100,
        "CAGR (%)": annual_return * 100,
        "Annual Volatility (%)": annual_vol * 100,
        "Sharpe Ratio": float(sharpe_ratio(returns, periods_per_year=periods_per_year)),
        "Sortino Ratio": float(sortino_ratio(returns, periods_per_year=periods_per_year)),
        "Max Drawdown (%)": max_dd * 100,
        "Calmar Ratio": annual_return / abs(max_dd) if max_dd else np.nan,
    }


# %%
strategy_metrics = calculate_metrics(portfolio_returns)
benchmark_metrics = calculate_metrics(benchmark_returns)
comparison = pl.DataFrame(
    [
        {"portfolio": "ETF Momentum", **strategy_metrics},
        {"portfolio": "60/40 Benchmark", **benchmark_metrics},
    ]
)
comparison

# %% [markdown]
# ## 10. Score the run against criteria fixed before it
#
# A term sheet is the short written statement of what a strategy has to achieve to be worth
# running, agreed before any backtest is scored against it. Writing it down first is what stops the
# criteria from being adjusted to whatever the run produced. Three lines are enough here: a
# risk-adjusted return floor, a limit on how far the portfolio may fall from its own peak, and a
# requirement to earn more over the sample than the 60/40 mix would have.

# %%
criteria = pl.DataFrame(
    {
        "criterion": ["Sharpe ratio", "Absolute max drawdown", "Total return vs 60/40"],
        "observed": [
            strategy_metrics["Sharpe Ratio"],
            abs(strategy_metrics["Max Drawdown (%)"]) / 100,
            strategy_metrics["Total Return (%)"] / 100,
        ],
        "required": [0.5, 0.25, benchmark_metrics["Total Return (%)"] / 100],
        "passed": [
            strategy_metrics["Sharpe Ratio"] > 0.5,
            abs(strategy_metrics["Max Drawdown (%)"]) < 25,
            strategy_metrics["Total Return (%)"] > benchmark_metrics["Total Return (%)"],
        ],
    }
)
criteria

# %% [markdown]
# A rule that misses part of its own term sheet is still worth keeping as a reference point, and
# that is the role this one plays for the rest of the chapter. Its value is that every assumption
# behind it is visible: the dates, the fills, the fee, the benchmark. A later strategy that claims
# to do better has to be run through the same simulator on the same dates before the claim means
# anything.

# %% [markdown]
# ## 11. Compare cumulative performance and drawdowns
#
# Growth of the initial capital answers who finishes ahead; the underwater curve shows the path
# risk hidden by an endpoint comparison.

# %%
strategy_cum = portfolio_value / INITIAL_CASH - 1
benchmark_cum = benchmark_value / INITIAL_CASH - 1

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=dates,
        y=strategy_cum * 100,
        name="ETF Momentum",
        line=dict(color=COLORS["blue"], width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates,
        y=benchmark_cum * 100,
        name="60/40 Benchmark",
        line=dict(color=COLORS["neutral"], dash="dash"),
    )
)
fig.update_layout(
    title="Momentum and 60/40 compared under one trading protocol",
    xaxis_title="Date",
    yaxis_title="Cumulative net return (%)",
    height=500,
)
fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
fig.show()

# %% [markdown]
# Drawdown is measured from the running peak, with zero at the top and losses below.


# %%
def compute_drawdown(equity: np.ndarray, initial_cash: float) -> np.ndarray:
    """Return percentage drawdown from the running portfolio peak."""
    running_max = np.maximum.accumulate(np.r_[initial_cash, equity])[1:]
    return equity / running_max - 1


# %%
strategy_dd = compute_drawdown(portfolio_value, INITIAL_CASH)
benchmark_dd = compute_drawdown(benchmark_value, INITIAL_CASH)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=dates,
        y=strategy_dd * 100,
        name="ETF Momentum",
        fill="tozeroy",
        line=dict(color=COLORS["blue"]),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates,
        y=benchmark_dd * 100,
        name="60/40 Benchmark",
        line=dict(color=COLORS["neutral"], dash="dash"),
    )
)
fig.update_layout(
    title="Drawdown of each portfolio from its own running peak",
    xaxis_title="Date",
    yaxis_title="Drawdown from running peak (%)",
    height=420,
)
fig.update_yaxes(range=[min(strategy_dd.min(), benchmark_dd.min()) * 110, 0])
fig.show()

# %% [markdown]
# ## 12. Compare performance by contemporaneous regime
#
# This descriptive slice attributes realized strategy returns to the yield-curve state observed on
# that day. It does not claim that the state causes the return difference.

# %%
risk_on_returns = portfolio_returns[regime]
risk_off_returns = portfolio_returns[~regime]
regime_metrics = pl.DataFrame(
    {
        "regime": ["Risk-on", "Risk-off"],
        "days": [len(risk_on_returns), len(risk_off_returns)],
        "mean_daily_return_bps": [
            risk_on_returns.mean() * 10_000,
            risk_off_returns.mean() * 10_000,
        ],
        "annualized_volatility_pct": [
            risk_on_returns.std(ddof=1) * np.sqrt(252) * 100,
            risk_off_returns.std(ddof=1) * np.sqrt(252) * 100,
        ],
    }
)
regime_metrics

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.18,
    subplot_titles=["Mean net return", "Annualized volatility"],
)
fig.add_trace(
    go.Bar(
        x=regime_metrics["regime"],
        y=regime_metrics["mean_daily_return_bps"],
        marker_color=[COLORS["blue"], COLORS["neutral"]],
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=regime_metrics["regime"],
        y=regime_metrics["annualized_volatility_pct"],
        marker_color=[COLORS["blue"], COLORS["neutral"]],
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig.update_layout(
    title="Strategy returns and volatility, split by the state of the curve",
    height=420,
)
fig.update_yaxes(title_text="Mean daily return (bps)", row=1, col=1)
fig.update_yaxes(title_text="Annualized volatility (%)", row=1, col=2)
fig.update_xaxes(title_text="Yield-curve regime", row=1, col=1)
fig.update_xaxes(title_text="Yield-curve regime", row=1, col=2)
fig.show()

# %% [markdown]
# ## 13. Reconcile with `ml4t-diagnostic`
#
# The library receives the same daily return arrays and dates. Its headline Sharpe, Sortino, CAGR,
# volatility, and drawdown should agree with the table above up to display precision.

# %%
analysis = PortfolioAnalysis(
    returns=portfolio_returns,
    benchmark=benchmark_returns,
    dates=dates,
    periods_per_year=252,
)
diagnostic_metrics = analysis.compute_summary_stats()

assert np.isclose(diagnostic_metrics.sharpe_ratio, strategy_metrics["Sharpe Ratio"])
assert np.isclose(diagnostic_metrics.sortino_ratio, strategy_metrics["Sortino Ratio"])
assert np.isclose(diagnostic_metrics.max_drawdown * 100, strategy_metrics["Max Drawdown (%)"])
print(diagnostic_metrics.summary())

# %% [markdown]
# ## Key takeaways
#
# 1. **Event order is part of the model, not an implementation detail.** The gap between the close
#    that produces a signal and the open that fills it is a modelling choice, and moving it changes
#    the result. Write the gap down before running anything, and encode it as a shift of the whole
#    weight matrix rather than as a condition inside a loop, where it is easy to lose.
# 2. **A simulator that cannot go short or borrow has to be told what to do when it runs out of
#    cash.** Selling before buying, charging the fee on each leg, and scaling the purchases to
#    whatever cash is left is one such rule. Simultaneously moving every position to its target is
#    another, and it quietly finances the fees from a negative balance.
# 3. **The benchmark has to run through the same simulator.** Comparing a strategy computed with
#    fills and fees against a benchmark computed from a return series compares two protocols, not
#    two strategies. Both portfolios here trade on the same dates at the same prices.
# 4. **Reconcile against a library before trusting a metric.** Sharpe, Sortino and drawdown all
#    have several defensible conventions, and a hand-rolled version that disagrees with the library
#    by a factor of the annualization is the usual first bug. The assertions in section 13 are the
#    check, not the prose.
#
# ### Known limitations
#
# - The universe is ten funds picked because they exist today, so it carries selection and
#   survivorship bias. Nothing here estimates what the rule would have earned on a universe built
#   from what was investable at the time.
# - The macro series is a single current snapshot, not a release-vintage panel, so the spread used
#   on any historical date may have been revised since.
# - Every date is reported. There is no held-out period, so the sample cannot support a statement
#   about what the rule would do next.
# - Fills happen at the open at no impact and no slippage beyond the flat fee, which is optimistic
#   for anything larger than a small account. Chapter 18 puts a cost model behind those fills.
#
# **Next:** `03_single_asset_vectorbt` and `04_single_asset_ml4t_backtest` implement one RSI rule in
# two engines, and `06_framework_parity` isolates what makes their results differ. Section 16.4
# covers the role of a fixed-rule baseline.
