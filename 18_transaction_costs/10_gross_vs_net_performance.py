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

# %% [markdown] tags=[]
# # Gross vs Net Performance Analysis
#
# **Docker image**: `ml4t`
#
# This notebook provides a descriptive framework for analyzing the gap between gross
# (theoretical) and net performance under explicit scenario costs.
#
# **Key Learning Objectives:**
# - Understand the full cost stack from gross to net
# - Apply parameterized costs to return-series scenarios
# - Compute net Sharpe under a parameterised cost stack
# - Compare three archetypes driven by real ETF return series under a common cost stack
#
# **Book Reference:** Chapter 18: Section 18.8 (Practical Guardrails: When Costs Kill a Strategy)
#
# **Prerequisites:** Read [`01_cost_taxonomy`](01_cost_taxonomy.ipynb) for the cost stack and
# [`09_frequency_tradeoff`](09_frequency_tradeoff.ipynb) for turnover-driven breakeven logic.

# %% [markdown] tags=[]
# ## 1. Setup

# %% tags=[]
"""Gross vs Net Performance - descriptive cost-stack scenario arithmetic."""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

# %% tags=["parameters"]
# No heavy computation - runs in seconds. Retained for Papermill compatibility.
SEED = 42
# The three archetypes are driven by real daily ETF return series; only the
# turnover/leverage/short configuration differs between them.
GROSS_START_DATE = "2021-01-01"
GROSS_END_DATE = "2023-12-31"

# %% tags=[]
set_global_seeds(SEED)

# %% [markdown] tags=[]
# ## 2. The Cost Stack
#
# Converting gross to net involves multiple layers:
#
# ```
# Gross Strategy Return
#     - Bid-Ask Spread Costs
#     - Market Impact Costs
#     = Trading P&L
#     - Commission/Fees
#     - Financing Costs (margin interest, borrow costs)
#     = Net Trading P&L
#     - Fund Expenses (mgmt fee, admin)
#     = Investor Net Return
# ```


# %% [markdown] tags=[]
# ### CostStack dataclass: fields
#
# The dataclass groups trading, financing, and fund-expense parameters so a
# single instance carries the full set of frictions used throughout the
# notebook.


# %% tags=[]
@dataclass
class CostStack:
    spread_cost_bps: float = 2.0
    impact_cost_bps: float = 5.0
    commission_bps: float = 1.0
    exchange_fee_bps: float = 0.5
    margin_rate_annual: float = 0.05
    borrow_rate_annual: float = 0.01
    management_fee_annual: float = 0.02
    admin_fee_annual: float = 0.002

    def trading_cost_per_trade(self, trade_size_pct: float = 0.1) -> float:
        scaled_impact = self.impact_cost_bps * np.sqrt(trade_size_pct / 0.1)
        return self.spread_cost_bps + scaled_impact

    def annual_trading_cost(self, annual_turnover: float) -> float:
        cost_bps = self.trading_cost_per_trade()
        commission_bps = self.commission_bps + self.exchange_fee_bps
        total_bps = cost_bps + commission_bps
        return annual_turnover * 2 * (total_bps / 10000)

    def annual_financing_cost(
        self,
        gross_leverage: float = 1.0,
        short_pct: float = 0.0,
    ) -> float:
        margin_cost = max(0, gross_leverage - 1) * self.margin_rate_annual
        # ``short_pct`` is the short notional as a fraction of gross exposure.
        short_notional = gross_leverage * short_pct
        borrow_cost = short_notional * self.borrow_rate_annual
        return margin_cost + borrow_cost

    def annual_expense_cost(self) -> float:
        return self.management_fee_annual + self.admin_fee_annual


# %% [markdown] tags=[]
# ### Default cost stack instance

# %% tags=[]
# Default cost stack
costs = CostStack()

print("Cost Stack Summary:")
print(f"  Trading cost (per trade): {costs.trading_cost_per_trade():.1f} bps")
print(f"  Commission + fees: {costs.commission_bps + costs.exchange_fee_bps:.1f} bps")
print(f"  Margin interest: {costs.margin_rate_annual:.1%} p.a.")
print(f"  Short borrow cost: {costs.borrow_rate_annual:.1%} p.a.")
print(f"  Management fee: {costs.management_fee_annual:.1%} p.a.")

# %% [markdown] tags=[]
# ## 3. Real Return Series and Strategy Configurations
#
# **Scope**: this section drives three strategy *configurations*: a high-turnover
# ETF profile, a leveraged long-short profile, and a low-turnover profile. Each uses
# **real daily ETF return series**, then varies turnover and leverage while holding
# the cost stack fixed to read off how the stack transforms gross into net.
# The gross return series are real (QQQ, a dollar-neutral QQQ-IWM spread, and SPY);
# turnover and leverage are configuration choices, not return-generating assumptions.
# A full real-data ETF momentum backtest is outside this descriptive cost exercise.


# %% [markdown] tags=[]
# ### Load Real ETF Return Series

# %% tags=[]
_panel = load_etfs(
    symbols=["SPY", "QQQ", "IWM"], start_date=GROSS_START_DATE, end_date=GROSS_END_DATE
)
_wide = (
    _panel.sort("symbol", "timestamp")
    .with_columns(r=pl.col("close").pct_change().over("symbol"))
    .pivot(values="r", index="timestamp", on="symbol")
    .sort("timestamp")
    .drop_nulls()
)
spy_ret = _wide["SPY"].to_numpy()
qqq_ret = _wide["QQQ"].to_numpy()
iwm_ret = _wide["IWM"].to_numpy()
ls_ret = qqq_ret - iwm_ret  # dollar-neutral long QQQ / short IWM spread
print(
    f"Loaded {_wide.height} daily returns ({GROSS_START_DATE}..{GROSS_END_DATE}) for SPY, QQQ, IWM"
)


# %% [markdown] tags=[]
# ### Strategy Builder
#
# Wraps a real gross-return series with a turnover/leverage/short configuration.
# Turnover is the per-day one-way turnover implied by the annual figure.


# %% tags=[]
def build_strategy(
    gross_returns: np.ndarray,
    annual_turnover: float,
    gross_leverage: float = 1.0,
    short_pct: float = 0.0,
    name: str = "Strategy",
) -> dict:
    """Pair a real return series with a turnover/leverage configuration."""
    daily_turnover = annual_turnover / 252
    return {
        "name": name,
        "gross_returns": gross_returns,
        "turnover": np.full(len(gross_returns), daily_turnover),
        "annual_turnover": annual_turnover,
        "gross_leverage": gross_leverage,
        "short_pct": short_pct,
    }


# %% tags=[]
# Three descriptive configurations spanning turnover and leverage extremes, each on a real series.

# High-turnover ETF configuration: QQQ, long-only, no leverage, 24x annual turnover.
high_turnover_etf = build_strategy(
    qqq_ret,
    annual_turnover=24.0,  # 2400% annual
    gross_leverage=1.0,
    short_pct=0.0,
    name="High Turnover ETF (24x, Long-Only)",
)

# Leveraged long-short scenario: QQQ-IWM spread, 200% gross, 50% of gross short
# exposure (100% of NAV short), and 6x turnover.
long_short = build_strategy(
    ls_ret,
    annual_turnover=6.0,  # 600% annual
    gross_leverage=2.0,  # 200% gross
    short_pct=0.5,
    name="Leveraged Long-Short (6x, 2x Gross)",
)

# Low-turnover ETF configuration: SPY, long-only, no leverage, 1x annual turnover.
value_strategy = build_strategy(
    spy_ret,
    annual_turnover=1.0,  # 100% annual
    gross_leverage=1.0,
    short_pct=0.0,
    name="Low Turnover (1x, Long-Only)",
)

strategies = [high_turnover_etf, long_short, value_strategy]

# %% [markdown] tags=[]
# ## 4. Apply Costs and Compute Net Returns


# %% tags=[]
def apply_cost_stack(
    strategy: dict,
    costs: CostStack,
) -> dict:
    """Apply full cost stack to get net returns."""
    gross_returns = strategy["gross_returns"]
    turnover = strategy["turnover"]

    # Daily trading costs
    trading_cost_bps = costs.trading_cost_per_trade()
    commission_bps = costs.commission_bps + costs.exchange_fee_bps
    daily_trading_cost = turnover * 2 * ((trading_cost_bps + commission_bps) / 10000)

    # Daily financing costs
    financing_annual = costs.annual_financing_cost(
        strategy["gross_leverage"], strategy["short_pct"]
    )
    daily_financing = financing_annual / 252

    # Daily fund expenses
    expense_annual = costs.annual_expense_cost()
    daily_expense = expense_annual / 252

    # Net returns
    net_returns = gross_returns - daily_trading_cost - daily_financing - daily_expense

    return {
        **strategy,
        "net_returns": net_returns,
        "trading_cost": daily_trading_cost,
        "financing_cost": np.full_like(gross_returns, daily_financing),
        "expense_cost": np.full_like(gross_returns, daily_expense),
    }


# Apply costs
for i, strat in enumerate(strategies):
    strategies[i] = apply_cost_stack(strat, costs)

# %% [markdown] tags=[]
# ## 5. Performance Comparison


# %% tags=[]
def compute_performance(returns: np.ndarray) -> dict:
    """Compute performance metrics."""
    ann_return = np.mean(returns) * 252
    ann_vol = np.std(returns, ddof=1) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    cumulative = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        "Annual Return": ann_return,
        "Annual Vol": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }


# %% tags=[]
# Compute summary rows for the chart annotations and downstream checks.
comparison_rows = []
for strat in strategies:
    gross_perf = compute_performance(strat["gross_returns"])
    net_perf = compute_performance(strat["net_returns"])
    drag = gross_perf["Annual Return"] - net_perf["Annual Return"]
    comparison_rows.append(
        {
            "Configuration": strat["name"],
            "Turnover (x)": strat["annual_turnover"],
            "Gross SR": round(gross_perf["Sharpe Ratio"], 2),
            "Net SR": round(net_perf["Sharpe Ratio"], 2),
            "Cost Drag (%)": round(drag * 100, 1),
        }
    )
for row in comparison_rows:
    print(
        f"{row['Configuration']}: gross SR {row['Gross SR']:.2f} -> "
        f"net SR {row['Net SR']:.2f}; cost drag {row['Cost Drag (%)']:.1f}%"
    )

# %% [markdown] tags=[]
# **Reading**: the computed summary rows show how the same illustrative cost stack
# changes three real ETF return series under different turnover and leverage configurations.
# These are descriptive full-sample scenarios, not realized strategy estimates.

# %% [markdown] tags=[]
# ## 6. Equity Curve Comparison

# %% [markdown] tags=[]
# ### Build cumulative equity series

# %% tags=[]
equity_series = [
    {
        "name": strat["name"],
        "cum_gross": np.cumprod(1 + strat["gross_returns"]),
        "cum_net": np.cumprod(1 + strat["net_returns"]),
        "timestamp": _wide["timestamp"].to_list(),
    }
    for strat in strategies
]

# %% [markdown] tags=[]
# ### Stacked subplot of gross vs net equity curves

# %% tags=[]
fig = make_subplots(
    rows=len(strategies),
    cols=1,
    subplot_titles=[s["name"] for s in equity_series],
    shared_xaxes=True,
    shared_yaxes=True,
)

for i, eq in enumerate(equity_series):
    fig.add_trace(
        go.Scatter(
            x=eq["timestamp"],
            y=eq["cum_gross"],
            mode="lines",
            name="Gross",
            line=dict(color=COLORS["blue"], dash="dash"),
            showlegend=(i == 0),
        ),
        row=i + 1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=eq["timestamp"],
            y=eq["cum_net"],
            mode="lines",
            name="Net",
            line=dict(color=COLORS["amber"]),
            showlegend=(i == 0),
        ),
        row=i + 1,
        col=1,
    )

fig.update_layout(
    title="Gross-to-net gaps compound as explicit costs accumulate",
    height=200 * len(equity_series) + 100,
)
fig.update_xaxes(title_text="Calendar date")
fig.update_yaxes(title_text="Cumulative wealth (start = 1.0)")
fig.show()

# %% [markdown] tags=[]
# **Interpretation**: The equity curves make cost drag path-dependent rather than
# abstract. Small daily deductions compound into visibly different wealth paths,
# especially for the highest-turnover strategy.

# %% [markdown] tags=[]
# ## 7. Cost Attribution

# %% tags=[]
# Cost breakdown for each strategy.
cost_breakdown = []

for strat in strategies:
    # Annual costs
    trading_annual = np.mean(strat["trading_cost"]) * 252
    financing_annual = np.mean(strat["financing_cost"]) * 252
    expense_annual = np.mean(strat["expense_cost"]) * 252
    total_annual = trading_annual + financing_annual + expense_annual

    cost_breakdown.append(
        {
            "Strategy": strat["name"],
            "Trading Costs": trading_annual,
            "Financing Costs": financing_annual,
            "Fund Expenses": expense_annual,
            "Total Costs": total_annual,
        }
    )

cost_df = pl.DataFrame(cost_breakdown)

# %% [markdown] tags=[]
# **Finding**: The computed attribution object separates execution drag from financing drag.
# That distinction matters because lowering turnover will not fix a strategy whose
# economics are dominated by leverage and borrow costs.

# %% [markdown] tags=[]
# The attribution object feeds the grouped chart below. No side-effect files are written.

# %% tags=[]
# Grouped bar chart
fig = go.Figure()
categories = ["Trading Costs", "Financing Costs", "Fund Expenses"]
colors = [COLORS["blue"], COLORS["amber"], COLORS["slate"]]

for i, strat in enumerate(cost_breakdown):
    fig.add_trace(
        go.Bar(
            name=strat["Strategy"],
            x=categories,
            y=[strat[c] for c in categories],
            marker_color=colors[i],
        )
    )

fig.update_layout(
    title="Leverage and turnover determine the annual cost mix",
    yaxis_title="Annual cost (%)",
    yaxis_tickformat=".1%",
    barmode="group",
    height=400,
)
fig.show()

# %% [markdown] tags=[]
# **Finding**: The grouped bars show that "cost" is not a single knob. Different
# strategy archetypes fail for different reasons, so the repair has to target the
# dominant source of drag rather than treat all frictions as interchangeable.

# %% [markdown] tags=[]
# ## 8. Sensitivity Analysis: Costs vs Turnover

# %% tags=[]
# How does net Sharpe vary with turnover for different one-way cost levels?
turnovers = np.linspace(0.5, 30, 50)
one_way_cost_bps = [5, 10, 20, 40]  # one-way trading cost in bps

sensitivity_data = []

for cost_bps in one_way_cost_bps:
    for turnover in turnovers:
        # Gross Sharpe of 1.5, 15% vol
        gross_daily_ret = 1.5 * 0.15 / 252
        # Both annual turnover and cost_bps are one-way; a round trip has two legs.
        daily_cost = 2 * turnover / 252 * (cost_bps / 10000)
        net_daily_ret = gross_daily_ret - daily_cost
        net_sharpe = net_daily_ret * 252 / 0.15

        sensitivity_data.append(
            {
                "Turnover": turnover,
                "One-Way Cost (bps)": cost_bps,
                "Net Sharpe": net_sharpe,
            }
        )

sens_df = pl.DataFrame(sensitivity_data)

# %% tags=[]
fig = go.Figure()

for cost_bps in one_way_cost_bps:
    subset = sens_df.filter(pl.col("One-Way Cost (bps)") == cost_bps)
    fig.add_trace(
        go.Scatter(
            x=subset["Turnover"].to_list(),
            y=subset["Net Sharpe"].to_list(),
            mode="lines",
            name=f"{cost_bps} bps one-way",
        )
    )

fig.add_hline(y=0, line_dash="dash", line_color=COLORS["slate"])
fig.add_hline(
    y=0.5, line_dash="dot", line_color=COLORS["positive"], annotation_text="Net SR = 0.5 reference"
)

fig.update_layout(
    title="Higher one-way costs reduce the turnover compatible with a target Sharpe",
    xaxis_title="Annual One-Way Turnover (x)",
    yaxis_title="Net Sharpe Ratio",
    height=450,
)
fig.show()

# %% [markdown] tags=[]
# **Interpretation**: The sensitivity chart is the general policy rule behind the
# case studies. Each line plots Net Sharpe as a function of annual one-way
# turnover at a fixed one-way cost level. As per-trade one-way costs rise, the
# feasible turnover range contracts sharply even if the gross signal quality
# stays unchanged.

# %% [markdown] tags=[]
# ## 9. Vector-L2 Turnover Diagnostic
#
# Standard turnover measures weight changes: $\sum_i |w_{i,t} - w_{i,t-1}|$.
# This diagnostic measures weight changes caused by the configured risk-input path;
# it does not identify alpha-signal turnover.
#
# A vector L2 turnover diagnostic measures the size of each weight-change vector.
#
# This is an illustrative vector norm, not a factor-portfolio matrix norm or an
# alpha-signal decomposition.

# %% [markdown] tags=[]
# ### Set up the covariance-input scenario
#
# 20 assets, base covariance held fixed, only a small rotation of the first two
# axes each period to simulate regime drift. Minimum-variance weights respond to
# this risk-input change, creating a descriptive covariance-drift scenario.

# %% tags=[]
n_assets = 20
n_periods = 60

np.random.seed(SEED)
base_cov = np.random.randn(n_assets, n_assets)
base_cov = base_cov @ base_cov.T / n_assets + np.eye(n_assets) * 0.5

# %% [markdown] tags=[]
# ### Roll the min-var portfolio through the rotating covariance

# %% tags=[]
turnovers_fro = []
turnovers_l1 = []
prev_weights = np.ones(n_assets) / n_assets  # start equal-weight

for t in range(n_periods):
    angle = 0.05 * t
    rotation = np.eye(n_assets)
    rotation[0, 0] = np.cos(angle)
    rotation[0, 1] = -np.sin(angle)
    rotation[1, 0] = np.sin(angle)
    rotation[1, 1] = np.cos(angle)
    cov_t = rotation @ base_cov @ rotation.T

    inv_cov = np.linalg.inv(cov_t)
    w = inv_cov @ np.ones(n_assets)
    w = w / w.sum()

    delta = w - prev_weights
    turnovers_fro.append(np.linalg.norm(delta))
    turnovers_l1.append(np.sum(np.abs(delta)))
    prev_weights = w

# %% [markdown] tags=[]
# ### Summarize covariance-drift maintenance turnover

# %% tags=[]
maintenance_l1 = np.asarray(turnovers_l1[1:])
maintenance_l2 = np.asarray(turnovers_fro[1:])
print("Covariance-Drift Maintenance Turnover (20-asset minimum-variance scenario)")
print(f"  Mean one-way L1 turnover after construction: {0.5 * np.mean(maintenance_l1):.4f}")
print(f"  Mean vector L2 turnover after construction:   {np.mean(maintenance_l2):.4f}")

# %% [markdown] tags=[]
# **Finding**: after excluding initial portfolio construction, this rotating-covariance
# scenario describes covariance-driven maintenance turnover. It is not an alpha-signal
# decomposition or a claim about a null portfolio.

# %% tags=[]
fig = go.Figure()
fig.add_scatter(
    x=list(range(1, n_periods)),
    y=turnovers_l1[1:],
    mode="lines",
    name="L1 turnover",
    line_color=COLORS["blue"],
)
fig.add_scatter(
    x=list(range(1, n_periods)),
    y=turnovers_fro[1:],
    mode="lines",
    name="Vector L2 turnover",
    line_color=COLORS["amber"],
)
fig.update_layout(
    title="Covariance drift changes minimum-variance weights after construction",
    xaxis_title="Rebalancing period",
    yaxis_title="Weight change (L1 or vector L2)",
    height=350,
)
fig.show()

# %% [markdown] tags=[]
# **Finding**: covariance drift creates maintenance turnover under this fixed-rule
# scenario. The plotted L1 and vector-L2 paths use different norms and are not
# additive alpha-signal decompositions.

# %% [markdown] tags=[]
# ## 10. Mechanism Summary
#
# Each item below restates a relationship between an input dial (turnover, leverage,
# expense ratio, covariance drift) and the cost-stack output. The gross return
# series are real ETF returns; the turnover and leverage are configuration choices,
# so the cost-drag numbers reflect those configurations applied to real returns.
#
# 1. **Trading drag scales with turnover**: the computed summary rows show the cost-stack
#    effect of increasing the annual turnover configuration.
#
# 2. **Financing matters for leverage**: a leveraged long-short configuration pays
#    margin interest and borrow on the short notional. Both are explicit assumptions.
#
# 3. **Fund expenses are a constant drag**: the configured management and admin
#    assumptions apply regardless of gross return.
#
# 4. **Net Sharpe ordering reflects the full cost mix, not gross alone**: the
#    computed rows recompute each configuration's gross and net metrics from the
#    same return path and explicit cost assumptions.
#
# 5. **Covariance drift changes risk inputs**: the §9 scenario describes maintenance
#    weight changes after construction under a rotating covariance matrix.

# %% [markdown] tags=[]
# ## 11. Net Sharpe by Configuration

# %% tags=[]
# Net Sharpe summary across the three parametric configurations.
print("\nNet Sharpe by Configuration:")
viability_rows = []
for strat in strategies:
    gross_sr = compute_performance(strat["gross_returns"])["Sharpe Ratio"]
    net_sr = compute_performance(strat["net_returns"])["Sharpe Ratio"]
    if net_sr > 1.0:
        net_sr_bucket = "Net SR > 1.0"
    elif net_sr > 0.5:
        net_sr_bucket = "Net SR in (0.5, 1.0]"
    else:
        net_sr_bucket = "Net SR <= 0.5"
    viability_rows.append(
        {
            "Configuration": strat["name"],
            "Gross SR": round(gross_sr, 2),
            "Net SR": round(net_sr, 2),
            "Net SR Bucket": net_sr_bucket,
        }
    )

# %% [markdown] tags=[]
# **Mechanism**: the computed Net-Sharpe rows are binned by configuration and range
# into three Net-Sharpe ranges. The bucket boundaries (1.0 and 0.5) are presentation
# thresholds for grouping the demonstration outcomes, not a thumbs-up / thumbs-down
# judgment on whether any of these configurations would be deployable on real data.
# The point of these computed rows is to make the gross-to-net gap visible for each
# configuration of turnover and leverage.
#
# **Next**: See [`11_cost_cliff`](11_cost_cliff.ipynb) for the intraday version of this cost arithmetic and
# [`12_commission_slippage_comparison`](12_commission_slippage_comparison.ipynb) for explicit model-choice sensitivity.

# %% [markdown] tags=[]
# ## Key Takeaways
#
# - **The cost stack is layered**: gross-to-net translation is not a single
#   "cost" deduction; trading frictions, financing, and fund expenses each
#   answer to different design levers. Lowering turnover does not fix a
#   leverage-driven cost problem.
# - **Financing dominates leveraged long-short**: under the default cost stack,
#   the leveraged long-short configuration loses more Sharpe to margin and
#   borrow than to trading frictions, while the high-turnover long-only loses
#   most of its Sharpe to per-trade costs.
# - **Turnover sensitivity is linear under fixed assumptions**: the §8 surface
#   shows net Sharpe declining linearly with turnover for each one-way cost level.
# - **Risk-input drift creates maintenance turnover**: the minimum-variance scenario
#   changes weights as covariance rotates, but it provides no alpha-signal conclusion.
