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
# # ML4T Library: Volume Participation Limits
#
# **Docker image**: `ml4t`
#
# This notebook demonstrates **VolumeParticipationLimit** from ml4t.backtest.execution
# for realistic institutional order execution:
#
# 1. **Volume Participation Concept**: Why institutions limit market footprint
# 2. **VolumeParticipationLimit API**: Parameters and behavior
# 3. **Partial Fills Over Multiple Bars**: Large orders split automatically
# 4. **Participation Rate Comparison**: 5%, 10%, 25% limits
# 5. **Real-World Scenario**: Large order with volume constraints
# 6. **Integration with Impact Models**: Full execution realism
#
# **Key Insight**: Large institutional orders cannot be filled instantly without
# moving markets. Volume participation limits enforce realistic execution by
# spreading fills across multiple bars based on available liquidity.
#
# **Learning Objectives**
# - Explain why desks cap participation as a fraction of available bar volume
# - Interpret `ExecutionResult` fields from the ML4T execution broker
# - Simulate partial fills across bars and days under different participation caps
# - Combine quantity limits with impact-adjusted fill pricing
#
# **Book Reference:** Chapter 18, Section 18.5 (Execution Algorithms as Controls)
#
# **Prerequisites:** Read [`06_ml4t_execution_demo`](06_ml4t_execution_demo.ipynb) for impact-model APIs and
# [`04_vwap_twap_execution`](04_vwap_twap_execution.ipynb) for benchmark scheduling logic.

# %% [markdown]
# ## Imports & Setup

# %%
"""ML4T Volume Participation - Realistic execution constraints on real NASDAQ-100 liquidity."""

import warnings

warnings.filterwarnings("ignore")

import plotly.graph_objects as go
import polars as pl
from IPython.display import Markdown, display
from ml4t.backtest.execution import (
    SquareRootImpact,
    VolumeParticipationLimit,
)
from plotly.subplots import make_subplots

from data import load_nasdaq100_bars
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

# %% tags=["parameters"]
# A parent order is released against a real intraday sequence of (volume, price)
# intervals built from AlgoSeek NASDAQ-100 minute bars. The participation cap is
# applied to each interval's *actual* traded volume, so completion time and
# realized price come entirely from real liquidity - no synthetic volume curves.
EXEC_SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]  # liquid NASDAQ-100 names
PRIMARY_SYMBOL = "AAPL"  # symbol whose real sessions drive the execution walk
TAQ_START_DATE = "2021-10-01"
TAQ_END_DATE = "2021-12-31"
INTERVAL_MINUTES = 15  # execution grid; 09:30-16:00 -> 26 intervals/session
CALIBRATION_SESSIONS = 20  # completed sessions used to estimate ADV before execution starts
ORDER_PCT_ADV = 0.5  # parent order as a fraction of measured ADV
PARTICIPATION_RATES = [0.05, 0.10, 0.25]
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## Part 1: Why Volume Participation Limits?
#
# Institutional traders face a fundamental constraint: **you cannot execute more
# than a fraction of market volume without moving prices against you**.
#
# ### The Problem
#
# | Order Size | % of Daily Volume | Expected Impact |
# |------------|-------------------|-----------------|
# | 10,000 shares | 1% | Minimal |
# | 100,000 shares | 10% | Moderate |
# | 500,000 shares | 50% | Severe |
#
# ### Industry Practice
#
# Institutional desks typically limit participation to **5-20% of volume**:
# - **5%**: Very conservative (stealth execution)
# - **10%**: Standard (balanced impact/speed)
# - **20%**: Aggressive (urgent execution)
# - **25%+**: Only for very liquid names or urgent situations

# %%
# Demonstrate the VolumeParticipationLimit API
limit = VolumeParticipationLimit(max_participation=0.10)

print("VolumeParticipationLimit Configuration")
print("=" * 50)
print(f"Max Participation Rate: {limit.max_participation:.0%}")
print(f"Min Volume Threshold:   {limit.min_volume:,.0f}")

# Example calculation
order_qty = 50_000  # shares
bar_volume = 100_000  # shares
price = 150.0

result = limit.calculate(order_qty, bar_volume, price)

print(f"\nExample: {order_qty:,} share order, {bar_volume:,} bar volume")
print(f"  Max fillable (10%):  {bar_volume * 0.10:,.0f} shares")
print(f"  Fillable quantity:   {result.fillable_quantity:,.0f} shares")
print(f"  Remaining quantity:  {result.remaining_quantity:,.0f} shares")
print(f"  Participation rate:  {result.participation_rate:.1%}")
print(f"  Is partial fill:     {result.is_partial}")

# %%
display(
    Markdown(
        f"**Finding**: A {order_qty:,.0f}-share parent order facing a "
        f"{bar_volume:,.0f}-share bar releases only "
        f"{result.fillable_quantity:,.0f} shares under a "
        f"{limit.max_participation:.0%} participation cap."
    )
)

# %% [markdown]
# ## Part 2: The ExecutionResult Object
#
# When VolumeParticipationLimit calculates fillable quantity, it returns an
# `ExecutionResult` with complete execution details:
#
# ```python
# @dataclass
# class ExecutionResult:
#     fillable_quantity: float    # Shares that can fill this bar
#     remaining_quantity: float   # Shares queued for next bar
#     adjusted_price: float       # Price (may include impact)
#     impact_cost: float          # Market impact cost
#     participation_rate: float   # Actual % of volume used
# ```

# %%
# Demonstrate different execution scenarios under a 10% participation limit.
limit = VolumeParticipationLimit(max_participation=0.10)
price = 100.0

scenarios = [
    ("Small order (within limit)", 5_000, 100_000),
    ("Medium order (at limit)", 10_000, 100_000),
    ("Large order (exceeds limit)", 50_000, 100_000),
    ("Very large order (5x limit)", 100_000, 100_000),
    ("Low volume bar", 10_000, 10_000),
    ("No volume data", 10_000, None),
]

scenario_rows = []
for name, order_qty, volume in scenarios:
    result = limit.calculate(order_qty, volume, price)
    scenario_rows.append(
        {
            "scenario": name,
            "order_qty": order_qty,
            "bar_volume": volume,
            "fill_qty": result.fillable_quantity,
            "remaining_qty": result.remaining_quantity,
            "participation_rate": result.participation_rate if volume else None,
        }
    )

scenarios_df = pl.DataFrame(scenario_rows)
scenarios_df

# %% [markdown]
# **Finding**: `ExecutionResult` turns a limit rule into operational state. The
# remaining quantity is the broker's queue for the next bar whenever current
# liquidity cannot absorb the order safely.

# %% [markdown]
# ## Part 3: Real Intraday Liquidity
#
# The participation cap is only meaningful against *real* liquidity. We load
# AlgoSeek NASDAQ-100 minute bars, aggregate them onto a 15-minute execution
# grid, and build a single consecutive sequence of intervals (each carrying its
# actual traded volume and volume-weighted price) for one liquid name. The
# parent order walks that real sequence interval by interval.


# %%
def load_intraday_panel(
    symbols: list[str],
    start_date: str,
    end_date: str,
    interval_minutes: int,
) -> pl.DataFrame:
    """Aggregate regular-session minute bars onto an intraday execution grid."""
    session_start = 9 * 60 + 30  # 09:30 as minute-of-day
    session_end = 16 * 60  # 16:00
    return (
        load_nasdaq100_bars(
            start_date=start_date,
            end_date=end_date,
            include_microstructure=True,
            lazy=True,
        )
        .filter(pl.col("symbol").is_in(symbols))
        .select("timestamp", "symbol", "volume", "last_trade_price")
        .filter(pl.col("last_trade_price").is_not_null() & (pl.col("volume") > 0))
        .with_columns(
            minute_of_day=pl.col("timestamp").dt.hour().cast(pl.Int32) * 60
            + pl.col("timestamp").dt.minute().cast(pl.Int32)
        )
        .filter(
            (pl.col("minute_of_day") >= session_start) & (pl.col("minute_of_day") < session_end)
        )
        .with_columns(timestamp=pl.col("timestamp").dt.truncate(f"{interval_minutes}m"))
        .group_by("symbol", "timestamp")
        .agg(
            volume=pl.col("volume").sum(),
            price=(pl.col("last_trade_price") * pl.col("volume")).sum() / pl.col("volume").sum(),
        )
        .with_columns(session=pl.col("timestamp").dt.date())
        .sort("symbol", "timestamp")
        .collect()
    )


# %%
panel = load_intraday_panel(EXEC_SYMBOLS, TAQ_START_DATE, TAQ_END_DATE, INTERVAL_MINUTES)

# Calibrate on completed sessions, then begin execution strictly afterward.
seq = panel.filter(pl.col("symbol") == PRIMARY_SYMBOL).sort("timestamp")
sessions = seq["session"].unique(maintain_order=True).to_list()
calibration_sessions = sessions[:CALIBRATION_SESSIONS]
execution_sessions = sessions[CALIBRATION_SESSIONS:]
calibration = seq.filter(pl.col("session").is_in(calibration_sessions))
execution = seq.filter(pl.col("session").is_in(execution_sessions))

adv = float(calibration.group_by("session").agg(dv=pl.col("volume").sum())["dv"].mean())
order_shares = int(round(ORDER_PCT_ADV * adv))

print(f"Primary symbol: {PRIMARY_SYMBOL}")
print(
    f"Calibration:    {len(calibration_sessions)} completed sessions through "
    f"{calibration_sessions[-1]}"
)
print(f"Execution:      {execution.height:,} intervals from {execution_sessions[0]}")
print(f"Calibration ADV:{adv:>14,.0f} shares")
print(f"Parent order:   {order_shares:,} shares ({ORDER_PCT_ADV:.0%} of ADV)")

# %% [markdown]
# **Finding**: The parent order is a fixed fraction of the symbol's measured ADV,
# so it is genuinely large relative to a single session's liquidity. That is the
# regime where a participation cap actually binds - small orders clear in one
# interval and never exercise the constraint.

# %% [markdown]
# ### Walk the Parent Order Against Real Intervals
#
# A participation-of-volume order reacts as trades print: after each market
# transaction, cumulative child fills may not exceed `cap × cumulative market
# volume`. Aggregating that feasible continuous process to 15-minute bars makes
# the end-of-interval fill exactly `cap × realized interval volume`; the fill
# price is the interval VWAP because the child order participates proportionally
# throughout the interval. The simulation does not know final bar volume at the
# interval open. Only the cap changes between runs.

# %% [markdown]
# A small frame helper adds cumulative shares, cost, and completion. Keeping this
# accounting separate leaves the event loop readable as a notebook cell.


# %%
def execution_frame(rows: list[dict], order_shares: int) -> pl.DataFrame:
    """Convert fill records to a cumulative execution path."""
    df = pl.DataFrame(rows)
    if df.height == 0:
        return df
    return df.with_columns(
        cumulative_shares=pl.col("fill_qty").cum_sum(),
        cumulative_cost=(pl.col("fill_qty") * pl.col("price")).cum_sum(),
    ).with_columns(pct_complete=pl.col("cumulative_shares") / order_shares * 100)


# %% [markdown]
# The walk applies the library limit at every interval, preserves the event
# timestamp, and queues any unfilled inventory for the next observed interval.


# %%
def participation_walk(
    intervals: pl.DataFrame,
    order_shares: int,
    max_participation: float,
    min_volume: float = 0.0,
) -> pl.DataFrame:
    """Release a parent order against a real (volume, price) interval sequence."""
    limit = VolumeParticipationLimit(max_participation=max_participation, min_volume=min_volume)
    remaining = order_shares
    rows: list[dict] = []
    sessions = intervals["session"].unique(maintain_order=True).to_list()
    session_index = {session: i for i, session in enumerate(sessions)}
    for i, row in enumerate(intervals.iter_rows(named=True)):
        if remaining <= 0:
            break
        result = limit.calculate(remaining, float(row["volume"]), float(row["price"]))
        if result.fillable_quantity <= 0:
            continue
        rows.append(
            {
                "bar": len(rows),
                "interval": i,
                "session_number": session_index[row["session"]],
                "timestamp": row["timestamp"],
                "bar_volume": float(row["volume"]),
                "price": float(row["price"]),
                "fill_qty": result.fillable_quantity,
                "remaining": result.remaining_quantity,
                "participation": result.participation_rate,
            }
        )
        remaining = result.remaining_quantity
    if remaining > 0:
        print(
            f"WARNING: parent order not fully filled - {remaining:,} of "
            f"{order_shares:,} shares remain after {len(rows)} intervals "
            f"(data window exhausted before completion)"
        )
    return execution_frame(rows, order_shares)


# %%
# Run the walk for each participation cap against the same real interval sequence.
results = {rate: participation_walk(execution, order_shares, rate) for rate in PARTICIPATION_RATES}

print("Participation Rate Comparison")
print(f"Order: {order_shares:,} shares ({PRIMARY_SYMBOL}, {ORDER_PCT_ADV:.0%} of ADV)")
print("=" * 70)
for rate, df in results.items():
    print(
        f"{rate:>5.0%} limit: {df.height:>3} intervals to complete, "
        f"{df['session_number'].max() + 1:>2} sessions, "
        f"avg participation: {df['participation'].mean():.1%}"
    )

comparison_df = pl.DataFrame(
    [
        {
            "participation_limit": rate,
            "intervals_to_complete": df.height,
            "sessions_to_complete": df["session_number"].max() + 1,
            "avg_participation": df["participation"].mean(),
            "vwap": df["cumulative_cost"][-1] / df["cumulative_shares"][-1],
        }
        for rate, df in results.items()
    ]
)

# %% [markdown]
# **Finding**: Against real liquidity the cap is the only lever that changes, so
# differences in intervals-to-complete and sessions-to-complete are attributable
# directly to footprint discipline. A tighter cap stretches the same parent order
# across more real intervals and more trading sessions.

# %% [markdown]
# ### Plot Completion Paths by Participation Limit

# %%
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=["5% Participation", "10% Participation", "25% Participation"],
)

colors = [COLORS["blue"], COLORS["amber"], COLORS["copper"]]

for i, (rate, df) in enumerate(results.items()):
    if df.height == 0:
        continue
    fig.add_scatter(
        x=df["bar"].to_list(),
        y=df["pct_complete"].to_list(),
        mode="lines+markers",
        name=f"{rate:.0%}",
        line=dict(color=colors[i], width=2),
        marker=dict(size=5),
        row=1,
        col=i + 1,
    )
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color=COLORS["neutral"],
        row=1,
        col=i + 1,
    )

fig.update_xaxes(title_text="Executed interval (count)")
fig.update_yaxes(title_text="Parent order filled (%)", range=[0, 105])
fig.update_layout(
    title="Higher participation caps shorten the completion horizon",
    height=400,
    showlegend=False,
)
fig.show()

# %%
summary = {row["participation_limit"]: row for row in comparison_df.iter_rows(named=True)}
display(
    Markdown(
        "**Finding**: The calibrated half-ADV order clears in "
        f"{summary[0.05]['intervals_to_complete']} intervals "
        f"({summary[0.05]['sessions_to_complete']} sessions) at a 5% cap, "
        f"{summary[0.10]['intervals_to_complete']} intervals "
        f"({summary[0.10]['sessions_to_complete']} sessions) at 10%, and "
        f"{summary[0.25]['intervals_to_complete']} intervals "
        f"({summary[0.25]['sessions_to_complete']} sessions) at 25%. "
        "The faster schedule consumes more of each interval's liquidity and "
        "therefore accepts more market-impact risk per fill."
    )
)

# %% [markdown]
# ## Part 4: Large Order Execution Timeline
#
# The same real walk yields the operational outcomes a portfolio manager trades
# off: intervals and sessions to completion, realized VWAP against real prices,
# and the actual participation share consumed each interval.

# %%
# Summarize each cap's real execution outcome.
for rate, df in results.items():
    if df.height == 0:
        continue
    vwap = df["cumulative_cost"][-1] / df["cumulative_shares"][-1]
    print(f"\n{rate:.0%} Participation Limit:")
    print(f"  Intervals to complete:  {df.height}")
    print(f"  Sessions to complete:   {df['session_number'].max() + 1}")
    print(f"  Realized VWAP:          ${vwap:.4f}")
    print(f"  Avg participation:      {df['participation'].mean():.1%}")

# %%
display(
    Markdown(
        "**Finding**: Realized VWAP differs across caps "
        f"(${summary[0.05]['vwap']:.2f} at 5%, ${summary[0.10]['vwap']:.2f} at 10%, "
        f"and ${summary[0.25]['vwap']:.2f} at 25%) because the schedules span "
        "different market-price windows. This timing or drift risk is distinct "
        "from the participation footprint measured within each interval."
    )
)

# %% [markdown]
# ### Visualize Execution Timeline

# %%
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "Cumulative Fill (%)",
        "Fill Size per Interval",
        "Execution Price Path",
        "Participation Rate",
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.1,
)

colors = {0.05: COLORS["blue"], 0.10: COLORS["amber"], 0.25: COLORS["copper"]}

# %% [markdown]
# ### Trace Helper for the Four-Panel Diagnostic


# %%
def add_execution_traces(fig, df: pl.DataFrame, rate: float, color: str) -> None:
    name = f"{rate:.0%}"
    bars = df["bar"].to_list()
    marker = dict(color=color, size=4, opacity=0.6)
    panels = [
        (df["pct_complete"].to_list(), "lines", dict(color=color, width=2), 1, 1, True),
        (df["fill_qty"].to_list(), "markers", marker, 1, 2, False),
        (df["price"].to_list(), "lines", dict(color=color, width=1), 2, 1, False),
        ([p * 100 for p in df["participation"].to_list()], "markers", marker, 2, 2, False),
    ]
    for values, mode, style, row, col, showlegend in panels:
        fig.add_scatter(
            x=bars,
            y=values,
            mode=mode,
            name=name,
            row=row,
            col=col,
            showlegend=showlegend,
            line=style if mode == "lines" else None,
            marker=style if mode == "markers" else None,
        )


# %%
for rate, df in results.items():
    if df.height == 0:
        continue
    add_execution_traces(fig, df, rate, colors[rate])

# %%
# Finalize execution diagnostics panel
fig.add_hline(y=100, line_dash="dash", line_color=COLORS["neutral"], row=1, col=1)

fig.update_xaxes(title_text="Interval", row=2, col=1)
fig.update_xaxes(title_text="Interval", row=2, col=2)
fig.update_yaxes(title_text="% Complete", row=1, col=1)
fig.update_yaxes(title_text="Shares", row=1, col=2)
fig.update_yaxes(title_text="Price ($)", row=2, col=1)
fig.update_yaxes(title_text="Participation (%)", row=2, col=2)

fig.update_layout(
    title="Tighter caps extend the horizon and spread fills across liquidity",
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
)
fig.show()

# %% [markdown]
# **Finding**: The execution timeline makes the trade-off visible. Conservative
# limits stretch the order over more real intervals and sessions, while aggressive
# limits raise fill size per interval and therefore increase the likelihood of
# adverse impact when liquidity is thin.

# %% [markdown]
# ## Part 5: Combining with Market Impact Models
#
# For complete execution realism, combine:
#
# 1. **VolumeParticipationLimit**: Controls *how much* can fill per bar
# 2. **MarketImpactModel**: Adjusts *price* based on participation
#
# The broker applies both in sequence.

# %% [markdown]
# ### Combined Volume-Limit and Impact Pricing Example

# %%
# Demonstrate combined volume limit + impact
volume_limit = VolumeParticipationLimit(max_participation=0.10)
impact_model = SquareRootImpact(coefficient=0.5, volatility=0.02)

# Scenario
order_qty = 50_000
bar_volume = 100_000
price = 100.0
is_buy = True

# Step 1: Apply volume limit
exec_result = volume_limit.calculate(order_qty, bar_volume, price)
fill_qty = exec_result.fillable_quantity

# Step 2: Apply market impact
impact = impact_model.calculate(fill_qty, price, bar_volume, is_buy)
fill_price = price + impact

# %%
print("Combined Execution Model")
print("=" * 60)
print("\n1. Volume Participation Limit (10%):")
print(f"   Order quantity:     {order_qty:,} shares")
print(f"   Bar volume:         {bar_volume:,} shares")
print(f"   Max fillable:       {bar_volume * 0.10:,.0f} shares")
print(f"   Actual fill:        {fill_qty:,.0f} shares")
print(f"   Remaining:          {exec_result.remaining_quantity:,.0f} shares")

print("\n2. Market Impact (Square Root):")
print(f"   Fill quantity:      {fill_qty:,.0f} shares")
print(f"   Base price:         ${price:.4f}")
print(f"   Price impact:       ${impact:.4f} ({impact / price * 10000:.1f} bps)")
print(f"   Fill price:         ${fill_price:.4f}")

print("\n3. Total Execution Cost:")
notional = fill_qty * price
impact_cost = fill_qty * impact
print(f"   Notional:           ${notional:,.2f}")
print(f"   Impact cost:        ${impact_cost:,.2f}")
print(f"   Total cost:         ${notional + impact_cost:,.2f}")

# %% [markdown]
# **Finding**: Volume limits and market impact answer different questions. The
# limit decides how much inventory may trade now; the impact model decides what
# price concession that permitted slice should pay.

# %% [markdown]
# ## Part 6: Minimum Volume Gate
#
# VolumeParticipationLimit includes a `min_volume` parameter that prevents
# execution on low-volume bars:
#
# ```python
# limit = VolumeParticipationLimit(
#     max_participation=0.10,
#     min_volume=5000,  # Don't execute if bar volume < 5,000
# )
# ```
#
# **Use Cases:**
# - Avoid executing during illiquid periods (lunch hour)
# - Prevent orders on halted or thinly-traded stocks
# - Implement "volume gates" for risk management

# %% [markdown]
# ### Minimum-Volume Gate Demonstration

# %%
# Demonstrate min_volume threshold by comparing two participation limits side by side.
limit_no_gate = VolumeParticipationLimit(max_participation=0.10, min_volume=0)
limit_with_gate = VolumeParticipationLimit(max_participation=0.10, min_volume=5000)

order_qty = 10_000
price = 100.0

volume_levels = [1000, 3000, 5000, 10000, 50000]

gate_rows = []
for vol in volume_levels:
    result_no_gate = limit_no_gate.calculate(order_qty, vol, price)
    result_with_gate = limit_with_gate.calculate(order_qty, vol, price)
    gate_rows.append(
        {
            "bar_volume": vol,
            "no_gate_fill": result_no_gate.fillable_quantity,
            "gate_5k_fill": result_with_gate.fillable_quantity,
            "blocked_by_gate": result_with_gate.fillable_quantity == 0 and vol < 5000,
        }
    )

gate_df = pl.DataFrame(gate_rows)

# %% [markdown]
# The grouped bars isolate the gate's discontinuity: below the threshold the
# permitted fill drops to zero, while both policies agree once volume recovers.

# %%
fig = go.Figure()
fig.add_bar(
    x=[f"{volume:,}" for volume in gate_df["bar_volume"]],
    y=gate_df["no_gate_fill"].to_list(),
    name="No minimum-volume gate",
    marker_color=COLORS["amber"],
)
fig.add_bar(
    x=[f"{volume:,}" for volume in gate_df["bar_volume"]],
    y=gate_df["gate_5k_fill"].to_list(),
    name="5,000-share minimum",
    marker_color=COLORS["blue"],
)
fig.update_layout(
    title="The minimum-volume gate blocks fills below 5,000 shares",
    barmode="group",
    xaxis_title="Realized bar volume (shares)",
    xaxis_type="category",
    yaxis_title="Permitted fill (shares)",
)
fig.show()

# %% [markdown]
# **Finding**: A minimum-volume gate is a second layer of execution discipline.
# It prevents the algorithm from trading mechanically through bars that are too
# thin to support even a small participation rate safely.

# %% [markdown]
# ## Summary
#
# ### VolumeParticipationLimit Key Points
#
# 1. **Purpose**: Enforce realistic execution by limiting fills to a % of volume
# 2. **Partial Fills**: Large orders automatically split across multiple bars
# 3. **Broker Integration**: `_partial_orders` dict tracks remaining quantities
#
# ### Parameter Guidelines
#
# | Parameter | Typical Value | Use Case |
# |-----------|---------------|----------|
# | `max_participation=0.05` | 5% | Stealth execution, minimize impact |
# | `max_participation=0.10` | 10% | Standard institutional |
# | `max_participation=0.25` | 25% | Urgent execution |
# | `min_volume=5000` | 5K+ | Block illiquid periods |
#
# ### Combining with Impact Models
#
# For full realism, combine:
# 1. **VolumeParticipationLimit** - Controls quantity per bar
# 2. **SquareRootImpact** - Adjusts price based on participation
#
# **Next**: See [`06_ml4t_execution_demo`](06_ml4t_execution_demo.ipynb) for market-impact model selection and
# [`08_ml_dynamic_execution`](08_ml_dynamic_execution.ipynb) for adaptive execution policies.

# %%
comparison_df

# %% [markdown]
# **Finding**: Participation caps control the speed-impact trade-off, while
# minimum-volume gates prevent fills in bars where that trade-off is simply not
# worth taking. Together they turn a benchmark schedule into a liquidity-aware
# execution policy.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Participation caps mechanize footprint discipline**: the cap limits each
#    child fill to a fixed share of cumulative market volume, regardless of how
#    aggressive the parent order is. This turns a target schedule into a
#    sequence of executable child orders.
#
# 2. **Partial-fill arithmetic is additive across bars**: `ExecutionResult`
#    returns `fillable_quantity` for the current bar and `remaining_quantity`
#    for the broker queue. Sum of fills equals the parent quantity only when
#    the order completes; otherwise the residual is what the next bar must
#    absorb.
#
# 3. **Completion time scales inversely with the cap**: against real AAPL
#    liquidity, higher caps clear the same calibrated half-ADV order in fewer
#    intervals and sessions. The speedup is roughly proportional to the cap,
#    but so is the participation footprint consumed each interval, and impact rises
#    with that footprint, so the genuine cost of a higher cap is impact risk,
#    not a worse benchmark fill.
#
# 4. **Minimum-volume gates are a second layer of control**: caps without a
#    gate still execute on thin bars, where even a small participation share
#    is dangerous. A `min_volume` threshold blocks fills entirely until
#    liquidity recovers - the right rule for lunch-hour or halted markets.
#
# 5. **Volume limits and impact models answer different questions**: the
#    limit decides *how much* trades now; the impact model decides *what
#    price concession* that permitted slice should pay. Production execution
#    composes both in sequence.
