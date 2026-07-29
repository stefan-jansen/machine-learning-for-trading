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
# # Composable Exit Rules and Portfolio Limits
# **Docker image**: `ml4t`
#
# **Chapter 19: Risk Management**
#
# ## Purpose
# Use the `ml4t.backtest.risk` library to evaluate position-level exit rules,
# compose rule priorities, and evaluate portfolio-level risk limits.
#
# ## Learning Objectives
# After completing this notebook, you will be able to:
# - Configure position-level exit rules (`StopLoss`, `TrailingStop`, `TakeProfit`)
# - Compose multiple rules with priority and boolean logic (`RuleChain`, `AllOf`, `AnyOf`)
# - Set up portfolio-wide risk limits (`MaxDrawdownLimit`, `DailyLossLimit`)
# - Interpret each rule's action, fill price, and reason
#
# ## Book reference
# Sections 19.4 (drawdowns and recovery), 19.7 (adaptive controls), and 19.8
# (kill switches and governance).
#
# ## Prerequisites
# - Comfort with the backtest framework introduced in Chapter 16
# - `02_exit_strategies` and `03_position_sizing_mae_mfe` for the rules'
#   analytical motivation

# %%
"""Configure and evaluate position rules and portfolio limits."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
from ml4t.backtest.risk import (
    AllOf,
    AnyOf,
    DailyLossLimit,
    GrossExposureLimit,
    MaxDrawdownLimit,
    MaxPositionsLimit,
    NetExposureLimit,
    PortfolioState,
    PositionState,
    RuleChain,
    ScaledExit,
    StopLoss,
    TakeProfit,
    TighteningTrailingStop,
    TimeExit,
    TrailingStop,
)

from data import load_etfs
from utils.style import COLORS, ml4t_palette

# %% tags=["parameters"]
N_BARS = 252

# %% [markdown]
# ## 1. Position-Level Exit Rules
#
# The `ml4t.backtest.risk.position` module provides exit rules that operate on
# individual positions. Each rule evaluates a `PositionState` and returns a
# `PositionAction` (HOLD, EXIT_FULL, EXIT_PARTIAL, or ADJUST_STOP).

# %% [markdown]
# ### 1.1 Creating Position States
#
# A compact state factory keeps the examples focused on rule behavior. Long and
# short returns use the entry notional as the common denominator.


# %%
def create_position_state(
    symbol: str = "SPY",
    side: str = "long",
    entry_price: float = 100.0,
    current_price: float = 100.0,
    bars_held: int = 0,
    bar_open: float | None = None,
    bar_high: float | None = None,
    bar_low: float | None = None,
    high_water_mark: float | None = None,
    low_water_mark: float | None = None,
) -> PositionState:
    """Create a PositionState for rule evaluation."""
    if side not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'")

    direction = 1.0 if side == "long" else -1.0
    unrealized_return = direction * (current_price - entry_price) / entry_price
    high_mark = max(entry_price, current_price) if high_water_mark is None else high_water_mark
    low_mark = min(entry_price, current_price) if low_water_mark is None else low_water_mark

    return PositionState(
        asset=symbol,
        side=side,
        entry_price=entry_price,
        current_price=current_price,
        bar_open=current_price if bar_open is None else bar_open,
        bar_high=current_price if bar_high is None else bar_high,
        bar_low=current_price if bar_low is None else bar_low,
        quantity=100.0,
        initial_quantity=100.0,
        unrealized_pnl=direction * (current_price - entry_price) * 100.0,
        unrealized_return=unrealized_return,
        bars_held=bars_held,
        high_water_mark=high_mark,
        low_water_mark=low_mark,
        max_favorable_excursion=max(0, unrealized_return),
        max_adverse_excursion=min(0, unrealized_return),
    )


# %% [markdown]
# ### 1.2 Static Exit Rules
#
# Static rules have fixed thresholds that don't change during the position lifetime.

# %%
stop_loss = StopLoss(pct=0.05)
take_profit = TakeProfit(pct=0.10)
time_exit = TimeExit(max_bars=20)

# %% [markdown]
# **`StopLoss(pct=0.05)`** exits when the price reaches 5% below entry.

# %%
scenarios = [
    ("Entry price", 100.0, 100.0),
    ("Down 3%", 100.0, 97.0),
    ("Down 5% (trigger)", 100.0, 95.0),
    ("Down 7%", 100.0, 93.0),
]

for name, entry, current in scenarios:
    state = create_position_state(entry_price=entry, current_price=current)
    action = stop_loss.evaluate(state)
    status = "TRIGGERED" if action.action.name != "HOLD" else "hold"
    print(f"  {name}: entry=${entry}, current=${current} -> {status}")
    if action.reason:
        print(f"    Reason: {action.reason}")

# %% [markdown]
# **`TakeProfit(pct=0.10)`** exits when the price reaches the 10% target.

# %%
scenarios = [
    ("Entry price", 100.0, 100.0),
    ("Up 5%", 100.0, 105.0),
    ("Above 10% (trigger)", 100.0, 110.01),
    ("Up 15%", 100.0, 115.0),
]

for name, entry, current in scenarios:
    state = create_position_state(entry_price=entry, current_price=current)
    action = take_profit.evaluate(state)
    status = "TRIGGERED" if action.action.name != "HOLD" else "hold"
    print(f"  {name}: entry=${entry}, current=${current} -> {status}")

# %% [markdown]
# **`TimeExit(max_bars=20)`** exits after 20 bars regardless of P&L.

# %%
for bars in [5, 15, 19, 20, 25]:
    state = create_position_state(bars_held=bars)
    action = time_exit.evaluate(state)
    status = "TRIGGERED" if action.action.name != "HOLD" else "hold"
    print(f"  Bars held: {bars} -> {status}")

# %% [markdown]
# ### 1.3 Dynamic Exit Rules
#
# Dynamic rules have thresholds that adapt to position performance.

# %% [markdown]
# **`TrailingStop(pct=0.05)`** exits when price retraces 5% from the running high.
# The rule uses the high-water mark through the prior completed bar, then updates
# that state after the current bar is evaluated.

# %%
trailing_stop = TrailingStop(pct=0.05)

states = [
    create_position_state(entry_price=100, current_price=100),
    create_position_state(entry_price=100, current_price=110),
    create_position_state(entry_price=100, current_price=120),  # New high
    create_position_state(entry_price=100, current_price=116),  # Down 3.3%
    create_position_state(entry_price=100, current_price=114),  # Down 5% (trigger)
]

hwm = 100.0
for state in states:
    state.high_water_mark = hwm
    action = trailing_stop.evaluate(state)
    status = "TRIGGERED" if action.action.name != "HOLD" else "hold"
    trail_level = hwm * 0.95
    print(f"  Price ${state.current_price}: prior HWM=${hwm}, trail=${trail_level:.1f} -> {status}")
    hwm = max(hwm, state.current_price)

# %% [markdown]
# **`TighteningTrailingStop`** shrinks the trail width as the position becomes more
# profitable. The configured schedule uses a 5% trail below 10% return, a 3%
# trail from 10%, and a 2% trail from 20%.

# %%
tightening = TighteningTrailingStop(
    [
        (0.00, 0.05),
        (0.10, 0.03),
        (0.20, 0.02),
    ]
)

for peak_price in [108.0, 116.0, 128.0]:
    current_price = peak_price * 0.96
    state = create_position_state(
        entry_price=100,
        current_price=current_price,
        high_water_mark=peak_price,
    )
    action = tightening.evaluate(state)
    status = action.reason if action.reason else "hold"
    print(f"  Peak ${peak_price:.0f}, current ${current_price:.2f}: {status}")

# %% [markdown]
# **`ScaledExit`** takes partial exits at profit targets. Schedule: +5% -> exit 25%,
# +10% -> exit 33% of remaining, +15% -> exit 50% of remaining. `ScaledExit`
# is stateful, so this one instance follows one position and is reset afterward.

# %%
scaled = ScaledExit(
    [
        (0.05, 0.25),
        (0.10, 0.33),
        (0.15, 0.50),
    ]
)

# Use prices just beyond each decimal threshold so binary floating-point
# representation cannot turn an intended crossing into an equality artifact.
for return_pct in [0.03, 0.0501, 0.08, 0.1001, 0.1501]:
    state = create_position_state(entry_price=100, current_price=100 * (1 + return_pct))
    action = scaled.evaluate(state)
    if action.action.name == "EXIT_PARTIAL":
        print(f"  At {return_pct:.2%}: EXIT {action.pct:.0%} of position ({action.reason})")
    else:
        print(f"  At {return_pct:.2%}: hold")

scaled.reset()

# %% [markdown]
# ## 2. Rule Composition
#
# The `ml4t.backtest.risk` module provides composition patterns to combine rules:
#
# - **RuleChain**: First non-HOLD wins (priority order)
# - **AllOf**: All must trigger (AND logic)
# - **AnyOf**: Any can trigger (OR logic, alias for RuleChain)

# %% [markdown]
# **`RuleChain`** returns the first non-`HOLD` action. Priority order:
# `StopLoss(5%)` > `TakeProfit(10%)` > `TrailingStop(3%)` > `TimeExit(20)`.

# %%
chain = RuleChain(
    [
        StopLoss(pct=0.05),
        TakeProfit(pct=0.10),
        TrailingStop(pct=0.03),
        TimeExit(max_bars=20),
    ]
)

# Test scenarios
test_cases = [
    ("Loss triggers stop", 100, 94, 5, 100),  # Stop loss
    ("Profit triggers TP", 100, 112, 5, 112),  # Take profit
    ("Trail triggers", 100, 108, 5, 115),  # Fell from 115 to 108 (>3%)
    ("Time triggers", 100, 102, 22, 105),  # Held 22 bars
    ("Nothing triggers", 100, 102, 5, 102),  # All hold
]

for name, entry, current, bars, hwm in test_cases:
    state = create_position_state(entry_price=entry, current_price=current, bars_held=bars)
    state.high_water_mark = hwm
    action = chain.evaluate(state)
    if action.action.name != "HOLD":
        print(f"  {name}: {action.reason}")
    else:
        print(f"  {name}: HOLD")

# %% [markdown]
# **`AllOf`** implements AND logic: every constituent rule must trigger. This
# example exits only after at least a 1% gain and a five-bar holding period.

# %%
all_of = AllOf(
    [
        TakeProfit(pct=0.01),
        TimeExit(max_bars=5),
    ]
)

test_cases = [
    ("Profitable, 3 bars", 0.05, 3),
    ("Profitable, 5 bars", 0.05, 5),
    ("Loss, 10 bars", -0.02, 10),
    ("Breakeven, 5 bars", 0.0, 5),
]

for name, ret, bars in test_cases:
    state = create_position_state(entry_price=100, current_price=100 * (1 + ret), bars_held=bars)
    state.unrealized_return = ret
    action = all_of.evaluate(state)
    status = "EXIT" if action.action.name != "HOLD" else "HOLD"
    print(f"  {name} (ret={ret:.0%}, bars={bars}): {status}")

# %% [markdown]
# **`AnyOf`** implements OR logic; the first triggering rule wins. It is equivalent
# to `RuleChain`: `StopLoss(5%)` OR `TakeProfit(10%)`
# OR `TimeExit(20)`.

# %%
any_of = AnyOf(
    [
        StopLoss(pct=0.05),
        TakeProfit(pct=0.10),
        TimeExit(max_bars=20),
    ]
)
{
    "AnyOf type": type(any_of).__name__,
    "rules in chain": len(any_of.rules),
    "doc": (AnyOf.__doc__ or "").strip().splitlines()[0] if AnyOf.__doc__ else "",
}

# %% [markdown]
# ## 3. Portfolio-Level Limits
#
# Portfolio limits operate on the entire portfolio state, not individual positions.
# They implement **kill switches** and **guardrails** discussed in Section 19.8.

# %% [markdown]
# `create_portfolio_state` packages the inputs (equity, high-water mark,
# positions, daily P&L) into a `PortfolioState` so each limit check has a
# single object to reason about. We use it as a scaffold in the demos
# below; in production the backtester reconstructs `PortfolioState` from
# the broker on every bar.


# %%
def create_portfolio_state(
    equity: float = 100000,
    initial_equity: float = 100000,
    high_water_mark: float = 100000,
    num_positions: int = 5,
    positions: dict[str, float] | None = None,
    daily_pnl: float = 0,
) -> PortfolioState:
    """Create a PortfolioState for limit checks."""
    if positions is None:
        positions = {f"ASSET_{i}": equity / 10 for i in range(num_positions)}
    else:
        num_positions = len(positions)

    gross = sum(abs(v) for v in positions.values())
    net = sum(positions.values())

    drawdown = (
        (high_water_mark - equity) / high_water_mark
        if high_water_mark > 0 and equity < high_water_mark
        else 0.0
    )

    return PortfolioState(
        equity=equity,
        initial_equity=initial_equity,
        high_water_mark=high_water_mark,
        current_drawdown=drawdown,
        num_positions=num_positions,
        positions=positions,
        daily_pnl=daily_pnl,
        gross_exposure=gross,
        net_exposure=net,
    )


# %% [markdown]
# **`MaxDrawdownLimit`** warns from 15% drawdown and liquidates from 20%.

# %%
dd_limit = MaxDrawdownLimit(max_drawdown=0.20, warn_threshold=0.15)

for dd_pct in [0.05, 0.10, 0.15, 0.18, 0.20, 0.25]:
    equity = 100000 * (1 - dd_pct)
    state = create_portfolio_state(equity=equity, high_water_mark=100000)
    result = dd_limit.check(state)
    if result.breached:
        print(f"  Drawdown {dd_pct:.0%}: {result.action.upper()} - {result.reason}")
    else:
        print(f"  Drawdown {dd_pct:.0%}: OK")

# %% [markdown]
# **`DailyLossLimit`** liquidates if today's loss exceeds 2% of current equity.

# %%
daily_limit = DailyLossLimit(max_daily_loss_pct=0.02)

for daily_pnl in [500, 0, -1000, -2000, -2500]:
    state = create_portfolio_state(equity=100000, daily_pnl=daily_pnl)
    result = daily_limit.check(state)
    pct = daily_pnl / 100000 * 100
    if result.breached:
        print(f"  Daily P&L ${daily_pnl:+,} ({pct:+.1f}%): {result.action.upper()}")
    else:
        print(f"  Daily P&L ${daily_pnl:+,} ({pct:+.1f}%): OK")

# %% [markdown]
# **`MaxPositionsLimit`** halts when the open-position count reaches 10.

# %%
pos_limit = MaxPositionsLimit(max_positions=10)

for n_pos in [5, 8, 10, 12]:
    positions = {f"ASSET_{i}": 10000 for i in range(n_pos)}
    state = create_portfolio_state(num_positions=n_pos, positions=positions)
    result = pos_limit.check(state)
    if result.breached:
        print(f"  {n_pos} positions: {result.action.upper()} - {result.reason}")
    else:
        print(f"  {n_pos} positions: OK")

# %% [markdown]
# **`GrossExposureLimit`** halts above 150% gross exposure.

# %%
gross_limit = GrossExposureLimit(max_gross_exposure=1.5)

for leverage in [0.8, 1.0, 1.3, 1.5, 2.0]:
    positions = {"LONG": 100000 * leverage / 2, "SHORT": -100000 * leverage / 2}
    state = create_portfolio_state(
        equity=100000,
        positions=positions,
        num_positions=2,
    )
    result = gross_limit.check(state)
    if result.breached:
        print(f"  {leverage:.0%} gross: {result.action.upper()} - {result.reason}")
    else:
        print(f"  {leverage:.0%} gross: OK")

# %% [markdown]
# **`NetExposureLimit`** warns outside the -10% to +10% net-exposure band.

# %%
net_limit = NetExposureLimit(max_net_exposure=0.10, min_net_exposure=-0.10)

test_cases = [
    ("Neutral", {"LONG": 50000, "SHORT": -50000}),
    ("+5% net", {"LONG": 55000, "SHORT": -50000}),
    ("+15% net", {"LONG": 60000, "SHORT": -45000}),
    ("-12% net", {"LONG": 44000, "SHORT": -56000}),
]

for name, positions in test_cases:
    state = create_portfolio_state(equity=100000, positions=positions, num_positions=2)
    result = net_limit.check(state)
    if result.breached:
        print(f"  {name}: {result.action.upper()} - {result.reason}")
    else:
        print(f"  {name}: OK")

# %% [markdown]
# ## 4. Practical Example: Layered Rule Configuration
#
# Position rules and portfolio limits address different decisions. The configuration
# below illustrates both layers; governance still requires escalation, approval, and
# reinstatement procedures outside these classes.

# %%
# Define position-level rules
position_rules = RuleChain(
    [
        StopLoss(pct=0.03),  # 3% hard stop
        TighteningTrailingStop(
            [
                (0.00, 0.05),  # 5% trail initially
                (0.10, 0.03),  # Tighten to 3% at +10%
                (0.20, 0.02),  # Tighten to 2% at +20%
            ]
        ),
        TakeProfit(pct=0.30),  # 30% take profit
        TimeExit(max_bars=60),  # Exit after 60 bars
    ]
)

# Define portfolio-level limits
portfolio_limits = [
    MaxDrawdownLimit(max_drawdown=0.15, warn_threshold=0.10),
    DailyLossLimit(max_daily_loss_pct=0.02),
    MaxPositionsLimit(max_positions=20),
    GrossExposureLimit(max_gross_exposure=1.0),
]

# %% [markdown]
# **Position Rules (RuleChain, in priority order)**:
#
# 1. `StopLoss(3%)`: hard stop on entry-price drawdown
# 2. `TighteningTrailingStop(5% -> 3% -> 2%)`: trail tightens with profit
# 3. `TakeProfit(30%)`: final target
# 4. `TimeExit(60 bars)`: maximum holding period
#
# **Portfolio Limits**:
#
# 1. `MaxDrawdownLimit(15%)`: liquidate at 15% drawdown, warn at 10%
# 2. `DailyLossLimit(2%)`: liquidate above a 2% daily loss
# 3. `MaxPositionsLimit(20)`: halt when the count reaches 20
# 4. `GrossExposureLimit(100%)`: halt above 100% gross exposure

# %% [markdown]
# ### 4.1 Evaluate Illustrative Position Paths

# %%
positions_sim = [
    {
        "symbol": "AAPL",
        "entry_price": 150.0,
        "prices": [150, 148, 146, 145.5, 145],  # Declining -> stop loss
    },
    {
        "symbol": "GOOGL",
        "entry_price": 100.0,
        "prices": [100, 108, 115, 118, 113],  # Rise then trail triggers
    },
    {
        "symbol": "MSFT",
        "entry_price": 300.0,
        "prices": [300, 305, 310, 308, 312],  # Steady rise, hold
    },
]

# %%
for pos in positions_sim:
    print(f"\n{pos['symbol']} (entry: ${pos['entry_price']})")

    hwm = pos["entry_price"]
    for i, price in enumerate(pos["prices"]):
        state = create_position_state(
            symbol=pos["symbol"],
            entry_price=pos["entry_price"],
            current_price=price,
            bars_held=i,
            high_water_mark=hwm,
        )

        action = position_rules.evaluate(state)
        ret = (price / pos["entry_price"] - 1) * 100

        if action.action.name != "HOLD":
            print(f"  Bar {i}: ${price} ({ret:+.1f}%) HWM=${hwm:.0f} -> EXIT: {action.reason}")
            break
        else:
            print(f"  Bar {i}: ${price} ({ret:+.1f}%) HWM=${hwm:.0f} -> hold")
        hwm = max(hwm, price)

# %% [markdown]
# ### 4.2 Portfolio Limit Check

# %%
portfolio = create_portfolio_state(
    equity=92000,  # Down from 100k
    initial_equity=100000,
    high_water_mark=105000,  # Was up 5% at peak
    daily_pnl=-1800,  # Down $1,800 today
    num_positions=8,
    positions={f"POS_{i}": 11500 for i in range(8)},  # 100% gross exposure
)

print("\nPortfolio State:")
print(f"  Equity: ${portfolio.equity:,.0f}")
print(f"  High Water Mark: ${portfolio.high_water_mark:,.0f}")
print(f"  Current Drawdown: {portfolio.current_drawdown:.1%}")
print(f"  Daily P&L: ${portfolio.daily_pnl:+,.0f} ({portfolio.daily_pnl / portfolio.equity:.1%})")
print(f"  Positions: {portfolio.num_positions}")
print(f"  Gross Exposure: {portfolio.gross_exposure / portfolio.equity:.0%}")

print("\nLimit Checks:")
for limit in portfolio_limits:
    result = limit.check(portfolio)
    limit_name = limit.__class__.__name__
    if result.breached:
        print(f"  {limit_name}: {result.action.upper()} - {result.reason}")
    else:
        print(f"  {limit_name}: OK")

# %% [markdown]
# ## 5. Rule Diagnostics
#
# The figures below visualize rule mechanics, not strategy performance. Thresholds
# are fixed before evaluation, and every surface calls the library on controlled
# states rather than clipping or rewriting realized returns.

# %% [markdown]
# ### 5.1 Trigger Timing on a Real Price Path
#
# The position is entered after the first SPY close of 2020. Each subsequent bar
# supplies its observed OHLC range. The trailing rule receives the high-water mark
# through the previous completed bar, which preserves the library's default lagged
# timing and avoids using the current bar's high before evaluating its low.

# %%
spy = (
    load_etfs(symbols=["SPY"], start_date="2020-01-01", end_date="2020-12-31")
    .sort("timestamp")
    .head(N_BARS)
)
entry_date = spy.item(0, "timestamp")
entry_price = float(spy.item(0, "close"))
print(f"Entry after {entry_date}: SPY close ${entry_price:.2f}; bars loaded: {spy.height}")

# %% [markdown]
# `first_trigger` makes the event order explicit. The entry bar is never tested,
# current-bar OHLC is observable to active stop orders, and water marks advance only
# after a bar completes without an exit.


# %%
def first_trigger(frame: pl.DataFrame, rule_name: str, rule: object) -> dict[str, object]:
    """Return the first action from a rule on a long position entered at the first close."""
    entry = float(frame.item(0, "close"))
    high_water_mark = entry
    low_water_mark = entry

    for bars_held, bar in enumerate(frame.iter_rows(named=True), start=0):
        if bars_held == 0:
            continue
        state = create_position_state(
            entry_price=entry,
            current_price=float(bar["close"]),
            bars_held=bars_held,
            bar_open=float(bar["open"]),
            bar_high=float(bar["high"]),
            bar_low=float(bar["low"]),
            high_water_mark=high_water_mark,
            low_water_mark=low_water_mark,
        )
        action = rule.evaluate(state)
        if action.action.name != "HOLD":
            return {
                "rule": rule_name,
                "timestamp": bar["timestamp"],
                "bars_held": bars_held,
                "close": float(bar["close"]),
                "fill_price": action.fill_price,
                "reason": action.reason,
            }
        high_water_mark = max(high_water_mark, float(bar["high"]))
        low_water_mark = min(low_water_mark, float(bar["low"]))

    raise RuntimeError(f"{rule_name} did not trigger within the observed path")


# %% [markdown]
# The same entry and bars feed three independent rules. Their first actions retain
# the library's fill-price convention and human-readable reason.

# %%
timeline_rules = {
    "Stop loss 5%": StopLoss(pct=0.05),
    "Trailing stop 3%": TrailingStop(pct=0.03),
    "Take profit 15%": TakeProfit(pct=0.15),
}
trigger_results = pl.DataFrame(
    [first_trigger(spy, rule_name, rule) for rule_name, rule in timeline_rules.items()]
).sort("timestamp")
trigger_results

# %% [markdown]
# The trigger markers use fill prices, while the line shows daily closes. A shared
# date axis prevents nearby selloff triggers from collapsing into overlapping text.

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=spy["timestamp"].to_list(),
        y=spy["close"].to_list(),
        name="SPY close",
        line=dict(color=COLORS["neutral"], width=2),
    )
)
for result, color in zip(
    trigger_results.iter_rows(named=True),
    ml4t_palette(trigger_results.height, categorical=True),
    strict=True,
):
    fig.add_trace(
        go.Scatter(
            x=[result["timestamp"]],
            y=[result["fill_price"]],
            mode="markers",
            name=result["rule"],
            marker=dict(color=color, size=11, line=dict(color=COLORS["bg_light"], width=1)),
            hovertemplate=(
                f"{result['rule']}<br>%{{x|%Y-%m-%d}}<br>Fill $%{{y:.2f}}<extra></extra>"
            ),
        )
    )
fig.update_layout(
    title=(
        "SPY Trigger Dates Differ Across Exit Rules"
        "<br><sup>Long entry after the first 2020 close; default stop-price fills</sup>"
    ),
    xaxis_title="Date",
    yaxis_title="SPY price ($)",
    height=430,
    legend_title_text="First action",
)
fig.show()

# %% [markdown]
# ### 5.2 Rule Priority Across Position States
#
# A close-only grid isolates composition semantics. When the 20-bar time exit
# overlaps a price rule, `RuleChain` returns the earlier rule in its declared order.

# %%
return_grid = np.arange(-10, 16, 1)
bars_grid = np.arange(0, 31, 2)
exit_chain = RuleChain([StopLoss(pct=0.05), TakeProfit(pct=0.10), TimeExit(max_bars=20)])
exit_code = {"HOLD": 0, "stop_loss": 1, "take_profit": 2, "time_exit": 3}
exit_surface = np.zeros((len(bars_grid), len(return_grid)), dtype=int)

for row_idx, bars_held in enumerate(bars_grid):
    for col_idx, return_pct in enumerate(return_grid):
        state = create_position_state(
            entry_price=100.0,
            current_price=100.0 * (1 + return_pct / 100),
            bars_held=int(bars_held),
        )
        action = exit_chain.evaluate(state)
        reason_key = next((key for key in exit_code if action.reason.startswith(key)), "HOLD")
        exit_surface[row_idx, col_idx] = exit_code[reason_key]

# %% [markdown]
# The categorical map separates `HOLD`, entry stop, profit target, and time exit.
# Its colors encode actions rather than performance.

# %%
exit_colors = [
    COLORS["silver_muted"],
    COLORS["negative"],
    COLORS["positive"],
    COLORS["amber"],
]
exit_scale = [
    [0.0, exit_colors[0]],
    [1 / 6, exit_colors[0]],
    [1 / 6, exit_colors[1]],
    [0.5, exit_colors[1]],
    [0.5, exit_colors[2]],
    [5 / 6, exit_colors[2]],
    [5 / 6, exit_colors[3]],
    [1.0, exit_colors[3]],
]

# %% [markdown]
# The colorbar names each action directly so the categorical codes never require
# interpretation from the reader.

# %%
fig = go.Figure(
    go.Heatmap(
        x=return_grid,
        y=bars_grid,
        z=exit_surface,
        zmin=0,
        zmax=3,
        colorscale=exit_scale,
        colorbar=dict(
            title="First action",
            tickmode="array",
            tickvals=[0, 1, 2, 3],
            ticktext=["HOLD", "STOP", "PROFIT", "TIME"],
        ),
        hovertemplate="Return %{x}%<br>Bars %{y}<br>Action code %{z}<extra></extra>",
    )
)
fig.update_layout(
    title=(
        "Rule Priority Resolves Overlapping Exit Conditions"
        "<br><sup>Close-only state grid; stop loss precedes take profit and time exit</sup>"
    ),
    xaxis_title="Return from entry (%)",
    yaxis_title="Bars held",
    height=430,
)
fig.show()

# %% [markdown]
# ### 5.3 Portfolio Escalation Surface
#
# The portfolio map combines two independent checks without inventing realized
# returns. Each cell constructs a `PortfolioState`, calls both limits, and reports
# the more severe action if both are breached.

# %%
drawdown_grid = np.arange(0, 26, 2.5)
daily_loss_grid = np.arange(0, 4.1, 0.5)
portfolio_surface = np.zeros((len(drawdown_grid), len(daily_loss_grid)), dtype=int)

for row_idx, drawdown_pct in enumerate(drawdown_grid):
    for col_idx, daily_loss_pct in enumerate(daily_loss_grid):
        equity = 100000 * (1 - drawdown_pct / 100)
        state = create_portfolio_state(
            equity=equity,
            high_water_mark=100000,
            daily_pnl=-equity * daily_loss_pct / 100,
            num_positions=0,
            positions={},
        )
        actions = {
            result.action
            for result in (dd_limit.check(state), daily_limit.check(state))
            if result.breached
        }
        portfolio_surface[row_idx, col_idx] = (
            2 if "liquidate" in actions else 1 if "warn" in actions else 0
        )

# %% [markdown]
# The boundaries reflect the exact class contracts: drawdown warns from 15% and
# liquidates from 20%, while daily loss liquidates only when it exceeds 2% of
# current equity.

# %%
portfolio_colors = [COLORS["silver_muted"], COLORS["amber"], COLORS["negative"]]
portfolio_scale = [
    [0.0, portfolio_colors[0]],
    [0.25, portfolio_colors[0]],
    [0.25, portfolio_colors[1]],
    [0.75, portfolio_colors[1]],
    [0.75, portfolio_colors[2]],
    [1.0, portfolio_colors[2]],
]
fig = go.Figure(
    go.Heatmap(
        x=daily_loss_grid,
        y=drawdown_grid,
        z=portfolio_surface,
        zmin=0,
        zmax=2,
        colorscale=portfolio_scale,
        colorbar=dict(
            title="Action",
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=["OK", "WARN", "LIQUIDATE"],
        ),
        hovertemplate="Daily loss %{x}%<br>Drawdown %{y}%<extra></extra>",
    )
)
fig.update_layout(
    title=(
        "Drawdown and Daily-Loss Limits Form an Escalation Surface"
        "<br><sup>MaxDrawdownLimit(20%, warn 15%) and DailyLossLimit(2%)</sup>"
    ),
    xaxis_title="Daily loss (% of current equity)",
    yaxis_title="Drawdown from high-water mark (%)",
    height=430,
)
fig.show()

# %% [markdown]
# ## 6. Demonstrated API Coverage
#
# This inventory summarizes only the classes exercised above. The counts derive
# from the names so they cannot drift from the displayed lists.

# %%
demonstrated_api = {
    "Position rules": (
        "StopLoss",
        "TakeProfit",
        "TimeExit",
        "TrailingStop",
        "TighteningTrailingStop",
        "ScaledExit",
    ),
    "Composition patterns": ("RuleChain", "AllOf", "AnyOf"),
    "Portfolio limits": (
        "MaxDrawdownLimit",
        "DailyLossLimit",
        "MaxPositionsLimit",
        "GrossExposureLimit",
        "NetExposureLimit",
    ),
}
library_coverage = pl.DataFrame(
    {
        "category": list(demonstrated_api),
        "count": [len(names) for names in demonstrated_api.values()],
        "examples": [", ".join(names) for names in demonstrated_api.values()],
    }
)
library_coverage

# %% [markdown]
# **Section integration**:
#
# - §19.4 references `StopLoss`, `TrailingStop`, and `TighteningTrailingStop`
#   for drawdown and recovery management.
# - §19.7 references `RuleChain` for adaptive controls.
# - §19.8 references `MaxDrawdownLimit`, `DailyLossLimit` for kill switches.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Position rules and portfolio limits act at different layers.**
#    The six position rules (`StopLoss`, `TakeProfit`, `TimeExit`,
#    `TrailingStop`, `TighteningTrailingStop`, `ScaledExit`) decide
#    *individual* exits; the five portfolio limits
#    (`MaxDrawdownLimit`, `DailyLossLimit`, `MaxPositionsLimit`,
#    `GrossExposureLimit`, `NetExposureLimit`) decide *system-wide*
#    halts. The two are not interchangeable.
# 2. **Composition expresses ordering, conjunction, and disjunction.**
#    `RuleChain` picks the first non-`HOLD` action;
#    `AllOf` requires every rule to agree; `AnyOf` fires if any rule
#    triggers. Priority is part of the policy and must be declared before
#    evaluating outcomes.
# 3. **Timing is part of every rule's definition.** Active stop orders may use
#    current-bar OHLC, but the default trailing rule receives a water mark from
#    completed prior bars. Entry-bar evaluation would violate that event order.
# 4. **A limit class is an enforcement primitive, not complete governance.**
#    Thresholds come from the risk mandate and still require escalation,
#    approval, and reinstatement procedures. The diagnostic surfaces demonstrate
#    mechanics and do not estimate performance benefits.
#
# API reference: `ml4t.backtest.risk`.
#
# **Next**: [`11_systematic_risk_sweep`](11_systematic_risk_sweep.ipynb)
# applies these position rules to 1D and 2D grids and shows how the
# resulting surfaces interact with MAE/MFE-calibrated cuts.
