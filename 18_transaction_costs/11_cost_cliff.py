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
# # The Cost Cliff: Intraday Strategy Reality Check
#
# **Docker image**: `ml4t`
#
# This notebook demonstrates how transaction costs can overwhelm an intraday
# strategy. A strategy that posts a high gross Sharpe can become unprofitable or
# marginal once its turnover is charged an explicit execution-cost stack.
#
# **Key Insight**: The "cost cliff" is the turnover level at which execution-cost
# drag consumes the gross return. Cost must be charged to traded notional, so the
# relevant activity measure is portfolio NAV turnover rather than a raw trade count.
#
# **Why This Matters**:
# - Intraday viability depends on the execution stack a strategy can actually achieve
# - Cost assumptions must be separated from measured market inputs
# - The same hypothetical gross return can survive or fail under different execution stacks
#
# **Learning Objectives**
# - Quantify the gross-to-net effect of explicit intraday cost stacks
# - Compare crossing, worked-order, and passive execution assumptions on the same profile
# - Estimate the turnover ceiling implied by a target net Sharpe
# - Use the cost cliff as a publication-quality sanity check for intraday claims
#
# **Book Reference:** Chapter 18: Section 18.8 (Practical Guardrails)
#
# **Prerequisites:** Read [`02_spread_estimation`](02_spread_estimation.ipynb) for spread realism and
# [`09_frequency_tradeoff`](09_frequency_tradeoff.ipynb) for the slower-frequency guardrail framing.

# %% [markdown]
# ## Setup

# %%
"""The Cost Cliff - Intraday Sharpe collapse and break-even turnover analysis."""

import math
from dataclasses import dataclass
from typing import NamedTuple

import plotly.graph_objects as go
import polars as pl
from IPython.display import Markdown, display

import utils  # noqa: F401
from data import load_nasdaq100_bars
from utils.style import COLORS

# %% tags=["parameters"]
# The retail half-spread that anchors the cost stacks is measured from real
# AlgoSeek NASDAQ-100 minute-bar quotes over the window below.
SPREAD_START_DATE = "2021-12-01"
SPREAD_END_DATE = "2021-12-31"

# %% [markdown]
# ## 1. Anchoring the Spread to Real NASDAQ-100 Quotes
#
# The cost cliff is only credible if the spread that drives it is real. We load
# AlgoSeek NASDAQ-100 minute bars, compute each interval's relative quoted spread,
# and take the **cross-sectional** distribution of per-symbol volume-weighted
# spreads. The median name, not the most liquid mega-cap, is the right anchor
# for a strategy that trades the whole index.


# %%
def measure_nasdaq100_spreads(start_date: str, end_date: str) -> pl.DataFrame:
    """Per-symbol volume-weighted relative quoted spread (bps) over the regular session."""
    return (
        load_nasdaq100_bars(
            start_date=start_date,
            end_date=end_date,
            include_microstructure=True,
            lazy=True,
        )
        .select("symbol", "timestamp", "volume", "close_bid_price", "close_ask_price")
        .filter(
            (pl.col("close_bid_price") > 0)
            & (pl.col("close_ask_price") >= pl.col("close_bid_price"))
            & (pl.col("volume") > 0)
        )
        .with_columns(
            minute_of_day=pl.col("timestamp").dt.hour().cast(pl.Int32) * 60
            + pl.col("timestamp").dt.minute().cast(pl.Int32),
            rel_spread_bps=(
                (pl.col("close_ask_price") - pl.col("close_bid_price"))
                / ((pl.col("close_ask_price") + pl.col("close_bid_price")) / 2)
                * 1e4
            ),
        )
        .filter((pl.col("minute_of_day") >= 570) & (pl.col("minute_of_day") < 960))
        .group_by("symbol")
        .agg(
            vw_rel_spread_bps=(pl.col("rel_spread_bps") * pl.col("volume")).sum()
            / pl.col("volume").sum()
        )
        .sort("vw_rel_spread_bps")
        .collect()
    )


# %% [markdown]
# ### Measure the Empirical Spread Anchor

# %%
spread_df = measure_nasdaq100_spreads(SPREAD_START_DATE, SPREAD_END_DATE)
if spread_df.is_empty():
    raise ValueError("no valid NASDAQ-100 spread observations were available")
median_rel_spread_bps = float(spread_df["vw_rel_spread_bps"].median())
if not math.isfinite(median_rel_spread_bps) or median_rel_spread_bps <= 0:
    raise ValueError("the measured median relative spread must be finite and positive")
# The cost of crossing once is half the quoted spread.
MEASURED_HALF_SPREAD_BPS = median_rel_spread_bps / 2.0

print(f"NASDAQ-100 symbols measured: {spread_df.height}")
print(f"Median per-symbol relative spread: {median_rel_spread_bps:.2f} bps")
print(f"Measured half-spread (cost to cross): {MEASURED_HALF_SPREAD_BPS:.2f} bps")
print(f"Most liquid name:  {spread_df['vw_rel_spread_bps'][0]:.2f} bps")
print(f"Least liquid name: {spread_df['vw_rel_spread_bps'][-1]:.2f} bps")

# %% [markdown]
# ### Inspect the Cross-Sectional Distribution

# %%
fig = go.Figure()
fig.add_histogram(
    x=spread_df["vw_rel_spread_bps"].to_list(), nbinsx=30, marker_color=COLORS["blue"]
)
fig.add_vline(
    x=median_rel_spread_bps,
    line_dash="dash",
    line_color=COLORS["neutral"],
    annotation_text=f"Median {median_rel_spread_bps:.1f} bps",
)
fig.update_layout(
    title="The Median Name Sets a Material Crossing-Cost Anchor",
    xaxis_title="Relative spread (bps)",
    yaxis_title="Number of symbols",
    height=380,
)
fig.show()

# %% [markdown]
# **Finding**: The cross-sectional distribution shows why a mega-cap quote is not
# a representative execution anchor for a broad-universe strategy. The median
# per-symbol spread supplies the empirical crossing-cost input used below. This
# volume-weighted estimate is most relevant when participation follows market volume.
# The sample is the set of symbols present in the licensed data window; the notebook
# makes no point-in-time index-membership or return-selection claim.

# %% [markdown]
# ## 2. Intraday Cost Components
#
# Intraday trading incurs costs at multiple levels. The spread term is the
# measured half-spread above; market impact, slippage, and fees are institutional
# cost assumptions layered on top.


# %%
@dataclass
class IntradayCostStack:
    """Illustrative execution costs in bps of traded notional, per side."""

    spread_half: float = 3.0
    market_impact: float = 2.0
    slippage: float = 1.5
    commission: float = 0.5
    exchange_fee: float = 0.3
    clearing_fee: float = 0.1

    @property
    def one_way_bps(self) -> float:
        """Total one-way cost in bps of traded notional."""
        return (
            self.spread_half
            + self.market_impact
            + self.slippage
            + self.commission
            + self.exchange_fee
            + self.clearing_fee
        )

    @property
    def round_trip_bps(self) -> float:
        """Total round-trip cost in bps of round-trip notional."""
        return 2 * self.one_way_bps


# %% [markdown]
# ### Cost Scenario Presets
#
# Only the crossing spread is measured. The worked-order spread fraction,
# passive spread cost, impact, slippage, and fees are illustrative assumptions.
# They isolate sensitivity to execution quality; they are not estimates of a
# particular broker, institution, or HFT strategy.

# %%
CROSSING = IntradayCostStack(
    spread_half=MEASURED_HALF_SPREAD_BPS,  # crosses and pays the full measured half-spread
    market_impact=3.0,
    slippage=2.0,
    commission=0.0,  # Commission-free broker
    exchange_fee=0.3,
    clearing_fee=0.1,
)

WORKED_ORDER = IntradayCostStack(
    spread_half=0.375 * MEASURED_HALF_SPREAD_BPS,  # works orders to cross only partway
    market_impact=2.5,
    slippage=1.0,
    commission=0.3,
    exchange_fee=0.2,
    clearing_fee=0.05,
)

PASSIVE_LOW_COST = IntradayCostStack(
    spread_half=0.0,  # Assumes passive fills without adverse-selection spread drag
    market_impact=0.5,
    slippage=0.2,
    commission=0.05,
    exchange_fee=-0.2,  # Illustrative maker rebate
    clearing_fee=0.02,
)

# %% [markdown]
# ### Compare Scenario Inputs

# %%
print("Intraday Cost Comparison (bps):")
cost_rows = []
for component in [
    "spread_half",
    "market_impact",
    "slippage",
    "commission",
    "exchange_fee",
    "clearing_fee",
]:
    cost_rows.append(
        {
            "Component": component,
            "Crossing": getattr(CROSSING, component),
            "Worked order": getattr(WORKED_ORDER, component),
            "Passive low-cost": getattr(PASSIVE_LOW_COST, component),
        }
    )
cost_rows.append(
    {
        "Component": "TOTAL (one-way)",
        "Crossing": CROSSING.one_way_bps,
        "Worked order": WORKED_ORDER.one_way_bps,
        "Passive low-cost": PASSIVE_LOW_COST.one_way_bps,
    }
)
cost_rows.append(
    {
        "Component": "TOTAL (round-trip)",
        "Crossing": CROSSING.round_trip_bps,
        "Worked order": WORKED_ORDER.round_trip_bps,
        "Passive low-cost": PASSIVE_LOW_COST.round_trip_bps,
    }
)
pl.DataFrame(cost_rows)

# %% [markdown]
# **Finding**: The round-trip stack sets the baseline hurdle. The crossing
# scenario starts several bps behind before alpha enters the picture. The two
# lower-cost stacks are sensitivity cases whose assumptions must be validated
# against an actual execution process before deployment.

# %% [markdown]
# ## 3. Hypothetical Intraday Strategy Profiles
#
# These profiles are deliberately hypothetical. They specify gross Sharpe,
# annual volatility, and round-trip portfolio NAV turnover. No signal is fit or
# tested here. Daily round-trip turnover measures opened-and-closed notional in
# units of NAV, regardless of how many child orders implement that turnover.


# %%
class IntradayStrategy(NamedTuple):
    """Hypothetical gross performance and portfolio-turnover assumptions."""

    name: str
    gross_sharpe: float  # Gross Sharpe ratio
    annual_vol: float  # Annual volatility
    round_trip_nav_turnover_per_day: float


HIGH_TURNOVER = IntradayStrategy(
    name="High turnover",
    gross_sharpe=2.5,
    annual_vol=0.20,
    round_trip_nav_turnover_per_day=1.0,
)

MODERATE_TURNOVER = IntradayStrategy(
    name="Moderate turnover",
    gross_sharpe=2.0,
    annual_vol=0.15,
    round_trip_nav_turnover_per_day=0.4,
)

LOW_TURNOVER = IntradayStrategy(
    name="Low turnover",
    gross_sharpe=1.5,
    annual_vol=0.12,
    round_trip_nav_turnover_per_day=0.1,
)


# %% [markdown]
# ### Net Performance Calculator
#
# Let $\tau$ denote daily round-trip NAV turnover, $c_{rt}$ the round-trip
# execution cost in basis points, $S_g$ gross Sharpe, and $\sigma$ annual
# volatility. With $D$ trading days and deterministic cost drag,
#
# $$C_{ann} = D\tau\frac{c_{rt}}{10^4}, \qquad
# S_n = \frac{S_g\sigma - C_{ann}}{\sigma}.$$
#
# This approximation changes annual return but not annual volatility.


# %%
def calculate_intraday_net_performance(
    strategy: IntradayStrategy,
    costs: IntradayCostStack,
    trading_days: int = 252,
) -> dict[str, float | str]:
    """Apply deterministic execution-cost drag to a hypothetical gross profile."""
    if strategy.annual_vol <= 0:
        raise ValueError("annual_vol must be positive")
    if strategy.round_trip_nav_turnover_per_day < 0:
        raise ValueError("round-trip NAV turnover cannot be negative")
    if trading_days <= 0:
        raise ValueError("trading_days must be positive")

    gross_return = strategy.gross_sharpe * strategy.annual_vol
    annual_round_trip_nav_turnover = strategy.round_trip_nav_turnover_per_day * trading_days

    annual_cost_bps = annual_round_trip_nav_turnover * costs.round_trip_bps
    annual_cost = annual_cost_bps / 10000
    net_return = gross_return - annual_cost

    net_sharpe = net_return / strategy.annual_vol

    return {
        "strategy": strategy.name,
        "gross_sharpe": strategy.gross_sharpe,
        "gross_return": gross_return,
        "daily_round_trip_nav_turnover": strategy.round_trip_nav_turnover_per_day,
        "annual_round_trip_nav_turnover": annual_round_trip_nav_turnover,
        "round_trip_cost_bps": costs.round_trip_bps,
        "annual_cost_bps": annual_cost_bps,
        "annual_cost": annual_cost,
        "net_return": net_return,
        "net_sharpe": net_sharpe,
    }


# %% [markdown]
# ## 4. The Cost Cliff Demonstration
#
# We now apply each illustrative execution stack to every hypothetical strategy
# profile. The calculation treats cost as a deterministic return drag and holds
# annual volatility fixed. It excludes financing, taxes, passive-fill risk, and
# uncertainty in realized impact, so this is a sensitivity analysis rather than
# a backtest or capacity estimate.

# %%
strategies = [HIGH_TURNOVER, MODERATE_TURNOVER, LOW_TURNOVER]
cost_scenarios = [
    ("Crossing", CROSSING),
    ("Worked order", WORKED_ORDER),
    ("Passive low-cost", PASSIVE_LOW_COST),
]

results = []
for strategy in strategies:
    for cost_name, costs in cost_scenarios:
        perf = calculate_intraday_net_performance(strategy, costs)
        perf["cost_type"] = cost_name
        results.append(perf)

results_df = pl.DataFrame(results)

# %% [markdown]
# ## 5. Visualizing the Cost Cliff

# %%
colors = {
    "Crossing": COLORS["negative"],
    "Worked order": COLORS["amber"],
    "Passive low-cost": COLORS["positive"],
}
patterns = {"Crossing": "/", "Worked order": "x", "Passive low-cost": "."}
fig = go.Figure()
x_labels = [strategy.name for strategy in strategies]
fig.add_trace(
    go.Bar(
        x=x_labels,
        y=[strategy.gross_sharpe for strategy in strategies],
        name="Gross Sharpe",
        marker_color=COLORS["blue"],
        text=[f"{strategy.gross_sharpe:.2f}" for strategy in strategies],
        textposition="outside",
    )
)

for cost_name, _costs in cost_scenarios:
    subset = results_df.filter(pl.col("cost_type") == cost_name)
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=subset["net_sharpe"].to_list(),
            name=cost_name,
            marker_color=colors[cost_name],
            marker_pattern_shape=patterns[cost_name],
            text=[f"{value:.2f}" for value in subset["net_sharpe"]],
            textposition="outside",
        )
    )

# %% [markdown]
# ### Add the Scenario Threshold

# %%
fig.add_hline(
    y=0.5,
    line_dash="dash",
    line_color=COLORS["neutral"],
    annotation_text="Scenario threshold",
)
fig.add_hline(y=0, line_dash="dot", line_color=COLORS["negative"])
fig.update_layout(
    title="Crossing Costs Overwhelm the High-Turnover Profile",
    yaxis_title="Sharpe Ratio",
    xaxis_title="Hypothetical strategy profile",
    barmode="group",
    height=450,
    showlegend=True,
)

fig.show()

# %% [markdown]
# ### Quantitative Reading

# %%
high_crossing = results_df.filter(
    (pl.col("strategy") == HIGH_TURNOVER.name) & (pl.col("cost_type") == "Crossing")
).row(0, named=True)
display(
    Markdown(
        f"**Finding**: for the high-turnover profile, the crossing stack reduces "
        f"Sharpe from **{HIGH_TURNOVER.gross_sharpe:.2f}** to "
        f"**{high_crossing['net_sharpe']:.2f}**, with annual cost drag of "
        f"**{high_crossing['annual_cost']:.1%} of NAV**."
    )
)

# %% [markdown]
# The comparison isolates cost assumptions; it does not establish that passive
# fills or worked-order execution are available to the strategy.

# %% [markdown]
# ## 6. Annual Cost as Percentage of Gross Return

# %% [markdown]
# The ratio below is also the percentage reduction in Sharpe under the fixed-volatility
# approximation, so a separate Sharpe-degradation chart would repeat the same information.

# %%
cost_share_df = results_df.with_columns(
    (pl.col("annual_cost") / pl.col("gross_return") * 100).alias("cost_pct_gross")
)

# %% [markdown]
# ### Compare Return-Budget Consumption

# %%
fig = go.Figure()

strategy_colors = {
    "High turnover": COLORS["negative"],
    "Moderate turnover": COLORS["amber"],
    "Low turnover": COLORS["blue"],
}
strategy_patterns = {"High turnover": "/", "Moderate turnover": "x", "Low turnover": "."}

for strategy in strategies:
    subset = cost_share_df.filter(pl.col("strategy") == strategy.name)

    fig.add_trace(
        go.Bar(
            x=subset["cost_type"].to_list(),
            y=subset["cost_pct_gross"].to_list(),
            name=strategy.name,
            marker_color=strategy_colors[strategy.name],
            marker_pattern_shape=strategy_patterns[strategy.name],
            text=[f"{value:.0f}%" for value in subset["cost_pct_gross"]],
            textposition="outside",
        )
    )

fig.add_hline(
    y=100,
    line_dash="dash",
    line_color=COLORS["negative"],
    annotation_text="100% = gross return consumed",
)

fig.update_layout(
    title="Crossing Costs Consume the High-Turnover Return Budget",
    yaxis_title="Cost as % of Gross Return",
    xaxis_title="Cost Structure",
    barmode="group",
    height=400,
)

fig.show()

# %% [markdown]
# **Finding**: Cost as a share of gross return is the clearest sanity check for
# intraday claims. Once annual cost approaches 100% of gross return, the strategy
# has no margin for model error, slippage misses, or live degradation.

# %% [markdown]
# ## 7. Break-Even Turnover Analysis


# %%
def calculate_break_even_turnover(
    target_net_sharpe: float,
    gross_sharpe: float,
    annual_vol: float,
    costs: IntradayCostStack,
    trading_days: int = 252,
) -> float:
    """Maximum daily round-trip NAV turnover for a target net Sharpe."""
    if annual_vol <= 0:
        raise ValueError("annual_vol must be positive")
    if trading_days <= 0:
        raise ValueError("trading_days must be positive")

    gross_return = gross_sharpe * annual_vol
    target_net_return = target_net_sharpe * annual_vol
    max_annual_cost = max(0.0, gross_return - target_net_return)

    round_trip_cost = costs.round_trip_bps / 10000

    if round_trip_cost <= 0:
        return math.inf if max_annual_cost > 0 else 0.0

    max_annual_turnover = max_annual_cost / round_trip_cost
    return max_annual_turnover / trading_days


# %% [markdown]
# ### Compute the Turnover Ceilings

# %%
be_rows = []
for gs in [1.5, 2.0, 2.5, 3.0]:
    row = {"Gross Sharpe": gs}
    for cost_name, cost_obj in cost_scenarios:
        max_turnover = calculate_break_even_turnover(
            target_net_sharpe=0.5,
            gross_sharpe=gs,
            annual_vol=0.15,
            costs=cost_obj,
        )
        row[cost_name] = max_turnover
    be_rows.append(row)
break_even_df = pl.DataFrame(be_rows)

# %% [markdown]
# ### Compare the Cost Scenarios
#
# The low-cost case permits much more turnover than the crossing case, so the
# vertical axis uses a log scale to keep all three curves legible.

# %%
line_dashes = {
    "Crossing": "solid",
    "Worked order": "dash",
    "Passive low-cost": "dot",
}
marker_symbols = {
    "Crossing": "circle",
    "Worked order": "square",
    "Passive low-cost": "diamond",
}

fig = go.Figure()
for cost_name, _cost_obj in cost_scenarios:
    values = break_even_df[cost_name].to_list()
    fig.add_trace(
        go.Scatter(
            x=break_even_df["Gross Sharpe"].to_list(),
            y=values,
            name=cost_name,
            mode="lines+markers+text",
            line=dict(color=colors[cost_name], dash=line_dashes[cost_name]),
            marker=dict(symbol=marker_symbols[cost_name], size=8),
            text=["", "", "", f"{values[-1]:.2f}"],
            textposition="top center",
        )
    )

fig.update_layout(
    title="Lower Execution Costs Support More Daily Turnover",
    xaxis_title="Gross Sharpe",
    yaxis_title="Maximum daily round-trip NAV turnover (log scale)",
    yaxis_type="log",
    height=430,
)
fig.show()

# %% [markdown]
# **Interpretation**: The break-even curves convert cost assumptions into a
# portfolio-turnover ceiling. A trade-count ceiling would be invalid without the
# notional size of each trade.

# %% [markdown]
# ## 8. Cost-Structure Comparison

# %% [markdown]
# ### Key Insights
#
# 1. **Crossing at high turnover is fragile**: the high-turnover profile can fall
#    below the scenario threshold once spread, impact, and fees are applied.
#
# 2. **Execution quality matters**: lower costs per unit of traded notional
#    materially expand viable turnover capacity relative to crossing.
#
# 3. **Viability threshold**: the selected net-Sharpe threshold is a scenario
#    diagnostic, not a universal deployment rule.
#
# 4. **Why backtests overstate intraday edge**: optimistic slippage and
#    incomplete spread/impact modeling can hide the true cost cliff.
#
# 5. **Practical policy**: reject an intraday proposal when conservative cost
#    assumptions consume its return budget at the intended NAV turnover.

# %% [markdown]
# ## Key Takeaways
#
# - **The cost cliff is a turnover phenomenon**: annualized round-trip cost
#   scales linearly with portfolio NAV turnover, not with an unscaled trade count.
# - **Crossing is the demanding case**: with the spread anchored to the measured
#   median NASDAQ-100 half-spread, the crossing stack supplies the largest return drag.
# - **Cost-as-%-of-gross-return is the cleanest diagnostic**: once annual cost
#   crosses 100% of gross return, the strategy has no margin for model error,
#   live degradation, or slippage misses.
# - **Break-even turnover is a deployment guardrail**: solving for the maximum
#   daily round-trip NAV turnover under a target net Sharpe gives a limit that can
#   be compared directly with a proposed portfolio.
# - **What is measured vs assumed**: the half-spread is measured from real
#   NASDAQ-100 quotes; market impact, slippage, and fees are illustrative cost
#   assumptions, and the gross Sharpe / turnover profiles are hypothetical
#   sensitivity cases. The notebook does not estimate an OFI signal or HFT fills.
#
# **Next**: See [`12_commission_slippage_comparison`](12_commission_slippage_comparison.ipynb)
# for explicit fee decomposition.
#
# **Book**: Chapter 18, Section 18.8 discusses practical guardrails for costs.
