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
# # Frequency-Dependent Transaction Costs
#
# **Docker image**: `ml4t`
#
# This notebook demonstrates how rebalancing cadence changes both captured signal and transaction
# costs in a self-contained historical illustration.
#
# **Key Insight**: Faster trading does not automatically capture more usable signal. It raises
# turnover, while the signal's decay determines whether acting sooner offsets that extra cost.
#
# **Topics Covered:**
# - Break-even alpha analysis: minimum alpha needed to cover costs
# - Frequency comparison: daily vs weekly vs biweekly vs monthly
# - Cost erosion curves: how Sharpe degrades with frequency
# - Scenario-preferred rebalancing cadence given cost structure
#
# **Learning Objectives**
# - Translate turnover assumptions into break-even alpha thresholds
# - Compare gross and net Sharpe across historical rebalancing cadences
# - Model the interaction between signal decay and transaction costs
# - Use a persistence-cost scenario to explain when a faster signal may still be worth trading
#
# **Book Reference:** Chapter 18: Section 18.8 (Practical Guardrails)
#
# **Prerequisites:** Read [`01_cost_taxonomy`](01_cost_taxonomy.ipynb) for breakeven framing and
# [`10_gross_vs_net_performance`](10_gross_vs_net_performance.ipynb) for the full net-of-cost waterfall.

# %% [markdown]
# ## Setup

# %%
"""Frequency-Dependent Transaction Costs - Rebalancing frequency vs cost tradeoff."""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import Markdown, display
from plotly.subplots import make_subplots

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

# %% tags=["parameters"]
# The historical illustration uses one fixed ETF universe and four cadences.
# The cost, decay, and persistence-cost sections are descriptive scenarios.
SEED = 42
ETF_SYMBOLS = ["SPY", "QQQ", "IWM", "XLF", "EEM", "XLE", "XLU", "FXI"]
GROSS_START_DATE = "2019-01-01"
GROSS_END_DATE = "2023-12-31"
MOMENTUM_LOOKBACK = 63  # trading days (~quarter)
TOP_N = 3  # equal-weight top-N by trailing momentum
SCENARIO_GROSS_SHARPE = 2.0
SCENARIO_ANNUAL_VOL = 0.15
EXAMPLE_DECAY_RATE = 0.05
DECAY_RATES = [0.01, 0.03, 0.05, 0.10, 0.20]

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## 1. Cost Model Assumptions
#
# We parameterize transaction costs as a function of turnover:
# - **Spread cost**: Half the bid-ask spread (paid on each trade)
# - **Market impact allowance**: Fixed bps per one-way trade in each scenario
# - **Commissions**: Fixed bps per trade
#
# Total cost per round-trip = 2 × (half-spread + impact + commission)


# %%
@dataclass
class CostAssumptions:
    """Illustrative transaction cost assumptions."""

    name: str
    spread_bps: float  # Half-spread per trade
    impact_bps: float  # Market impact per trade
    commission_bps: float  # Commission per trade

    @property
    def total_one_way(self) -> float:
        """Total cost per one-way trade in bps."""
        return self.spread_bps + self.impact_bps + self.commission_bps

    @property
    def round_trip(self) -> float:
        """Total round-trip cost in bps."""
        return 2 * self.total_one_way


# %% [markdown]
# ### Illustrative Cost Scenarios
#
# These high-, medium-, and low-friction stacks are teaching assumptions, not estimates for named
# investor types. Each component is a one-way cost; doubling their sum gives the round-trip cost
# applied to the notebook's one-way turnover convention.

# %%
HIGH_FRICTION_COSTS = CostAssumptions(
    name="High-friction scenario",
    spread_bps=3.0,
    impact_bps=2.0,
    commission_bps=0.0,
)

MEDIUM_FRICTION_COSTS = CostAssumptions(
    name="Medium-friction scenario",
    spread_bps=1.0,
    impact_bps=3.0,
    commission_bps=0.5,
)

LOW_FRICTION_COSTS = CostAssumptions(
    name="Low-friction scenario",
    spread_bps=0.2,
    impact_bps=0.5,
    commission_bps=0.1,
)

COST_SCENARIOS = [HIGH_FRICTION_COSTS, MEDIUM_FRICTION_COSTS, LOW_FRICTION_COSTS]

# %%
pl.DataFrame(
    [
        {
            "Scenario": c.name,
            "Spread (bps)": c.spread_bps,
            "Impact (bps)": c.impact_bps,
            "Commission (bps)": c.commission_bps,
            "Round-trip (bps)": c.round_trip,
        }
        for c in COST_SCENARIOS
    ]
)

# %% [markdown]
# **Finding**: The scenario table is the whole problem setup in miniature.
# Frequency only creates value if the gross signal is large enough to survive the
# assumed round-trip cost profile.

# %% [markdown]
# ## 2. A Historical Momentum Illustration at Four Cadences
#
# Rather than assume turnover per cadence, we measure it in a historical illustration. We
# equal-weight the top-ranked subset by lagged trailing momentum within a fixed ETF universe and
# compare several cadences on provider-adjusted closes. The universe and sample are fixed teaching
# inputs, not a point-in-time membership screen or sealed holdout. The comparison is descriptive
# and does not estimate a production-optimal cadence. Between scheduled rebalances, realized asset
# returns drift the portfolio weights; the next turnover charge compares the new target with those
# pre-trade drifted weights.


# %%
def momentum_frequency_backtest(
    prices: np.ndarray, rebalance_days: int, lookback: int, top_n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run a top-N trailing-momentum portfolio at a fixed rebalance cadence.

    `prices` is a (T, S) array of daily closes. A signal observed through close
    t-1 is executed at close t, and the resulting weights earn the t-to-t+1
    return. Returns aligned daily gross returns and one-way turnover.
    """
    if lookback < 1 or rebalance_days < 1:
        raise ValueError("lookback and rebalance_days must be positive")
    n_days, n_assets = prices.shape
    if not 1 <= top_n <= n_assets:
        raise ValueError("top_n must be between one and the number of assets")
    if n_days <= lookback + 2:
        raise ValueError("prices do not cover the lookback and execution lag")

    rets = prices[1:] / prices[:-1] - 1
    held = np.zeros(n_assets)
    port_returns = []
    one_way_turnover = []
    first_execution = lookback + 1
    for t in range(first_execution, n_days - 1):
        daily_turnover = 0.0
        if (t - first_execution) % rebalance_days == 0:
            signal_end = t - 1
            mom = prices[signal_end] / prices[signal_end - lookback] - 1
            new_w = np.zeros(n_assets)
            new_w[np.argsort(mom)[-top_n:]] = 1.0 / top_n
            daily_turnover = 0.5 * np.abs(new_w - held).sum()
            held = new_w
        period_return = float((held * rets[t]).sum())
        port_returns.append(period_return)
        one_way_turnover.append(daily_turnover)
        ending_values = held * (1 + rets[t])
        held = ending_values / ending_values.sum()
    return np.asarray(port_returns), np.asarray(one_way_turnover)


# %% [markdown]
# The reporting helper below annualizes the daily return series using sample volatility. Keeping
# this calculation separate makes the cost-accounting path above independently testable.


# %%
def annualized_sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe of a daily return series."""
    volatility = returns.std(ddof=1)
    return float(returns.mean() / volatility * np.sqrt(252)) if volatility > 0 else 0.0


# %%
_panel = load_etfs(symbols=ETF_SYMBOLS, start_date=GROSS_START_DATE, end_date=GROSS_END_DATE)
_wide = (
    _panel.sort("symbol", "timestamp")
    .pivot(values="close", index="timestamp", on="symbol")
    .sort("timestamp")
    .drop_nulls()
)
_symbols = sorted(ETF_SYMBOLS)
_wide = _wide.select("timestamp", *_symbols)
_prices = _wide.select(_symbols).to_numpy()
assert set(_panel["symbol"].unique()) == set(ETF_SYMBOLS)
assert _panel.select(pl.struct("symbol", "timestamp").n_unique()).item() == _panel.height
print(
    f"Loaded {_wide.height} sessions x {_prices.shape[1]} ETFs "
    f"({GROSS_START_DATE}..{GROSS_END_DATE})"
)

# Measure annual turnover and gross Sharpe for each cadence in the historical illustration.
FREQUENCIES = {
    "Daily": {"trading_days_per_rebalance": 1, "rebalances_per_year": 252},
    "Weekly": {"trading_days_per_rebalance": 5, "rebalances_per_year": 52},
    "Biweekly": {"trading_days_per_rebalance": 10, "rebalances_per_year": 26},
    "Monthly": {"trading_days_per_rebalance": 21, "rebalances_per_year": 12},
}
for freq, params in FREQUENCIES.items():
    port, one_way_turnover = momentum_frequency_backtest(
        _prices, params["trading_days_per_rebalance"], MOMENTUM_LOOKBACK, TOP_N
    )
    params["gross_returns"] = port
    params["one_way_turnover"] = one_way_turnover
    params["annual_turnover"] = float(one_way_turnover.mean() * 252)
    params["gross_sharpe"] = annualized_sharpe(port)
    params["gross_return"] = float(port.mean() * 252)
    params["annual_vol"] = float(port.std(ddof=1) * np.sqrt(252))

# %%
frequency_metrics = pl.DataFrame(
    [
        {
            "Frequency": freq,
            "Rebal/Year": p["rebalances_per_year"],
            "Annual TO (x)": round(p["annual_turnover"], 1),
            "Gross SR": round(p["gross_sharpe"], 2),
            "Ann Vol (%)": round(p["annual_vol"] * 100, 1),
        }
        for freq, p in FREQUENCIES.items()
    ]
)
frequency_metrics

# %%
_daily = FREQUENCIES["Daily"]
_monthly = FREQUENCIES["Monthly"]
display(
    Markdown(
        f"""**Finding**: Turnover is measured, not assumed. In this fixed historical sample, """
        f"""the lagged momentum rule turns over {_monthly["annual_turnover"]:.1f}x annually at """
        f"""monthly cadence and {_daily["annual_turnover"]:.1f}x at daily cadence. Its gross """
        f"""Sharpe is {_monthly["gross_sharpe"]:.2f} monthly and """
        f"""{_daily["gross_sharpe"]:.2f} daily. These are descriptive full-sample estimates, """
        """not sealed-holdout performance."""
    )
)

# %% [markdown]
# ## 3. Break-Even Alpha Analysis
#
# The break-even alpha is the minimum gross alpha needed to cover transaction costs:
#
# $$\text{Break-even Alpha} = \text{Annual One-way Turnover} \times \text{Round-trip Cost}$$
#
# One-way turnover is half the absolute weight change. Multiplying it by round-trip cost charges
# both purchase and sale legs without double-counting. If gross alpha is below this threshold, the
# scenario's cost estimate exceeds the strategy's expected return.


# %%
def calculate_break_even_alpha(annual_turnover: float, round_trip_cost_bps: float) -> float:
    """
    Calculate minimum alpha needed to break even.

    Args:
        annual_turnover: One-way annual turnover as decimal (e.g., 2.5 = 250%)
        round_trip_cost_bps: Round-trip cost in basis points

    Returns:
        Break-even alpha in basis points (annualized)
    """
    return annual_turnover * round_trip_cost_bps


# %%
be_rows = []
for freq, params in FREQUENCIES.items():
    row = {"Frequency": freq}
    for costs in COST_SCENARIOS:
        label = costs.name.removesuffix(" scenario")
        row[label] = round(calculate_break_even_alpha(params["annual_turnover"], costs.round_trip))
    be_rows.append(row)
pl.DataFrame(be_rows)

# %% [markdown]
# **Finding**: Break-even alpha grows linearly with turnover. The daily
# schedule only makes sense when the signal is both strong and short-lived; slower
# cadences preserve more of the edge under the stated cost scenarios.

# %% [markdown]
# ## 4. Net Sharpe by Frequency in the Historical Illustration
#
# Each cadence carries its own measured daily gross return and turnover path. We subtract the
# scenario's cost on each rebalance day before annualizing the net return, volatility, and Sharpe.
# This preserves the timing and volatility contribution of trading costs. It remains an in-sample
# illustration rather than an estimate of future performance.


# %%
def real_net_by_frequency(cost_assumptions: CostAssumptions) -> pl.DataFrame:
    """Net performance per cadence after charging each observed rebalance."""
    results = []
    for freq, params in FREQUENCIES.items():
        gross_daily = params["gross_returns"]
        daily_cost = params["one_way_turnover"] * cost_assumptions.round_trip / 10000
        net_daily = gross_daily - daily_cost
        annual_cost = float(daily_cost.mean() * 252)
        net_return = float(net_daily.mean() * 252)
        net_vol = float(net_daily.std(ddof=1) * np.sqrt(252))
        results.append(
            {
                "frequency": freq,
                "gross_sharpe": params["gross_sharpe"],
                "gross_return": params["gross_return"],
                "annual_turnover": params["annual_turnover"],
                "annual_cost": annual_cost,
                "net_return": net_return,
                "net_sharpe": annualized_sharpe(net_daily),
                "cost_pct_gross": (
                    annual_cost / params["gross_return"]
                    if params["gross_return"] > 0
                    else float("inf")
                ),
                "net_vol": net_vol,
            }
        )
    return pl.DataFrame(results)


# %% [markdown]
# ### Analytical Helper for the Signal-Decay Section
#
# A parametric net-Sharpe-by-frequency curve used later (Section 7) to study how
# signal decay shifts the scenario-preferred cadence. It applies a *single* gross Sharpe to
# every cadence's measured turnover.


# %%
def simulate_frequency_comparison(
    gross_sharpe: float,
    annual_vol: float,
    cost_assumptions: CostAssumptions,
) -> pl.DataFrame:
    """Net performance across cadences for a hypothetical gross Sharpe."""
    results = []
    for freq, params in FREQUENCIES.items():
        gross_return = gross_sharpe * annual_vol
        annual_cost = params["annual_turnover"] * cost_assumptions.round_trip / 10000
        net_return = gross_return - annual_cost
        results.append(
            {
                "frequency": freq,
                "gross_sharpe": gross_sharpe,
                "gross_return": gross_return,
                "annual_turnover": params["annual_turnover"],
                "annual_cost": annual_cost,
                "net_return": net_return,
                "net_sharpe": net_return / annual_vol if annual_vol > 0 else 0,
                "cost_pct_gross": annual_cost / gross_return if gross_return > 0 else float("inf"),
            }
        )
    return pl.DataFrame(results)


# %%
results_df = pl.concat(
    [
        real_net_by_frequency(costs).with_columns(pl.lit(costs.name).alias("cost_type"))
        for costs in [HIGH_FRICTION_COSTS, MEDIUM_FRICTION_COSTS]
    ]
)

results_df.filter(pl.col("cost_type") == HIGH_FRICTION_COSTS.name).select(
    "frequency",
    pl.col("gross_sharpe").round(2),
    pl.col("net_sharpe").round(2),
    (pl.col("annual_cost") * 100).round(1).alias("cost_drag_%"),
)

# %%
_high_friction_results = results_df.filter(pl.col("cost_type") == HIGH_FRICTION_COSTS.name)
_best_historical = _high_friction_results.sort("net_sharpe", descending=True).row(0, named=True)
display(
    Markdown(
        f"""**Finding**: Under the illustrative {HIGH_FRICTION_COSTS.round_trip:.1f} bps """
        f"""round-trip stack, {_best_historical["frequency"].lower()} has the highest net Sharpe """
        """in this full-period sample. This is a descriptive result, not a selected production """
        """cadence."""
    )
)

# %% [markdown]
# ## 5. Visualization: Frequency vs Net Sharpe

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["High friction", "Medium friction"],
    shared_yaxes=True,
)

freq_order = ["Monthly", "Biweekly", "Weekly", "Daily"]

for col, cost_type in enumerate([HIGH_FRICTION_COSTS.name, MEDIUM_FRICTION_COSTS.name], 1):
    subset = results_df.filter(pl.col("cost_type") == cost_type).sort(
        pl.col("frequency").map_elements(lambda x: freq_order.index(x), return_dtype=pl.Int64)
    )
    fig.add_trace(
        go.Scatter(
            x=subset["frequency"].to_list(),
            y=subset["net_sharpe"].to_list(),
            mode="lines+markers",
            name="Net Sharpe",
            line=dict(color=COLORS["blue"], width=3),
            marker=dict(size=10),
            showlegend=(col == 1),
        ),
        row=1,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=subset["frequency"].to_list(),
            y=subset["gross_sharpe"].to_list(),
            mode="lines+markers",
            name="Gross Sharpe",
            line=dict(color=COLORS["amber"], width=2, dash="dash"),
            marker=dict(size=8),
            showlegend=(col == 1),
        ),
        row=1,
        col=col,
    )

# %% [markdown]
# ### Add an Illustrative Sharpe Hurdle

# %%
for col in [1, 2]:
    if col == 1:
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color=COLORS["neutral"],
            annotation_text="Illustrative hurdle",
            row=1,
            col=col,
        )
    else:
        fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["neutral"], row=1, col=col)
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["negative"], row=1, col=col)

fig.update_layout(
    title=(
        "Faster rebalancing compounds signal degradation and trading costs"
        f"<br><sup>Fixed {len(ETF_SYMBOLS)}-ETF illustration, {GROSS_START_DATE} to "
        f"{GROSS_END_DATE}; {MOMENTUM_LOOKBACK}-day signal lagged one close</sup>"
    ),
    yaxis_title="Sharpe Ratio",
    height=500,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    margin=dict(t=160, b=65),
)
fig.update_xaxes(title_text="Rebalancing cadence")

fig.show()

# %% [markdown]
# **Finding**: The gap between the gross (dashed) and net (solid) lines is the cost
# drag, and it widens toward daily cadence in this sample. Even before costs, the
# gross line slopes down as cadence accelerates, so the two effects reinforce rather
# than offset. This full-period comparison is descriptive, not a holdout ranking.

# %% [markdown]
# ## 6. Cost Erosion Analysis
#
# How much of the gross alpha is consumed by costs at each frequency?

# %%
erosion = results_df.filter(pl.col("cost_type") == HIGH_FRICTION_COSTS.name).sort(
    pl.col("frequency").map_elements(lambda x: freq_order.index(x), return_dtype=pl.Int64)
)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=erosion["frequency"].to_list(),
        y=[r * 100 for r in erosion["gross_return"].to_list()],
        name="Gross Return",
        marker_color=COLORS["blue"],
    )
)
fig.add_trace(
    go.Bar(
        x=erosion["frequency"].to_list(),
        y=[r * 100 for r in erosion["net_return"].to_list()],
        name="Net Return",
        marker_color=COLORS["amber"],
    )
)
fig.update_layout(
    title=(
        "Trading costs widen the gross-to-net return gap at faster cadences"
        "<br><sup>High-friction scenario; costs charged on each observed rebalance</sup>"
    ),
    yaxis_title="Annual Return (%)",
    xaxis_title="Rebalancing Frequency",
    barmode="group",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(t=105),
)
fig.show()

# %% [markdown]
# ## 7. Frequency Choice Under Signal Decay
#
# The historical rule above does not estimate a signal-decay function. To show when faster trading
# *can* pay, this section switches
# to a **hypothetical fast-decaying signal**: a fixed gross Sharpe whose captured
# alpha decays exponentially with the delay between rebalances. This is a parametric
# study layered on the measured per-cadence turnover, not a calibrated ETF result.


# %%
def evaluate_decay_scenario(
    gross_sharpe: float,
    annual_vol: float,
    cost_assumptions: CostAssumptions,
    signal_decay_rate: float = 0.1,
) -> dict:
    """Compare scenario net Sharpe after applying a specified signal decay."""
    results = []

    for freq, params in FREQUENCIES.items():
        days_delay = params["trading_days_per_rebalance"]
        decay_factor = np.exp(-signal_decay_rate * days_delay)
        effective_gross_sharpe = gross_sharpe * decay_factor

        sim = simulate_frequency_comparison(effective_gross_sharpe, annual_vol, cost_assumptions)
        freq_result = sim.filter(pl.col("frequency") == freq).to_dicts()[0]
        freq_result["effective_gross_sharpe"] = effective_gross_sharpe
        freq_result["decay_factor"] = decay_factor
        results.append(freq_result)

    results_df = pl.DataFrame(results)
    preferred = results_df.sort("net_sharpe", descending=True).row(0, named=True)

    return {
        "all_results": results_df,
        "preferred_frequency": preferred["frequency"],
        "preferred_net_sharpe": preferred["net_sharpe"],
    }


# %%
# Example: hypothetical fast-decaying signal
result = evaluate_decay_scenario(
    gross_sharpe=SCENARIO_GROSS_SHARPE,
    annual_vol=SCENARIO_ANNUAL_VOL,
    cost_assumptions=HIGH_FRICTION_COSTS,
    signal_decay_rate=EXAMPLE_DECAY_RATE,
)

result["all_results"].select(
    "frequency",
    pl.col("effective_gross_sharpe").round(2).alias("eff_gross_sr"),
    pl.col("decay_factor").round(3),
    pl.col("annual_cost").round(4),
    pl.col("net_sharpe").round(2),
)

# %%
display(
    Markdown(
        f"""**Finding**: At the stated {EXAMPLE_DECAY_RATE:.0%} daily decay and """
        """high-friction assumptions, the """
        f"""scenario-preferred cadence is {result["preferred_frequency"].lower()} with an """
        f"""approximate net Sharpe of {result["preferred_net_sharpe"]:.2f}. This is a sensitivity """
        """calculation, not an ETF performance estimate."""
    )
)

# %% [markdown]
# ## 8. Sensitivity Analysis: Cost vs Signal Decay
#
# The scenario-preferred frequency depends on:
# 1. Cost structure (higher costs favor lower frequency)
# 2. Signal decay rate (faster decay favors higher frequency)

# %%
# Grid search over decay rates
sensitivity_results = []

for decay in DECAY_RATES:
    for costs in [HIGH_FRICTION_COSTS, LOW_FRICTION_COSTS]:
        result = evaluate_decay_scenario(
            gross_sharpe=SCENARIO_GROSS_SHARPE,
            annual_vol=SCENARIO_ANNUAL_VOL,
            cost_assumptions=costs,
            signal_decay_rate=decay,
        )
        sensitivity_results.append(
            {
                "decay_rate": decay,
                "cost_type": costs.name,
                "preferred_freq": result["preferred_frequency"],
                "preferred_net_sharpe": result["preferred_net_sharpe"],
            }
        )

sensitivity_df = pl.DataFrame(sensitivity_results)

# %%
fig = go.Figure()
for index, costs in enumerate([HIGH_FRICTION_COSTS, LOW_FRICTION_COSTS]):
    subset = sensitivity_df.filter(pl.col("cost_type") == costs.name).sort("decay_rate")
    fig.add_trace(
        go.Scatter(
            x=(subset["decay_rate"] * 100).to_list(),
            y=subset["preferred_freq"].to_list(),
            mode="lines+markers",
            name=f"{costs.name} ({costs.round_trip:.1f} bps round-trip)",
            line=dict(
                color=COLORS["blue"] if index == 0 else COLORS["amber"],
                width=3 if index == 0 else 2,
                dash="solid" if index == 0 else "dash",
            ),
            marker=dict(symbol="circle" if index == 0 else "diamond", size=9),
        )
    )
fig.update_layout(
    title=(
        "Faster signal decay shifts the scenario-preferred cadence upward"
        f"<br><sup>Hypothetical gross Sharpe {SCENARIO_GROSS_SHARPE:.1f} and "
        f"{SCENARIO_ANNUAL_VOL:.0%} volatility; historical turnover inputs</sup>"
    ),
    xaxis_title="Assumed signal decay per day (%)",
    yaxis_title="Scenario-preferred cadence",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(t=110),
)
fig.update_yaxes(categoryorder="array", categoryarray=freq_order)
fig.show()

# %% [markdown]
# **Finding**: The crossover is conditional on the stated decay, gross Sharpe, volatility, cost,
# and historical-turnover assumptions. It demonstrates the direction of the tradeoff rather than
# estimating a universally preferred frequency.

# %% [markdown]
# ## 9. Persistence-Cost Score: Alpha-to-Go Intuition
#
# Formal alpha-to-go is a dynamic-optimization quantity that depends on forecasts, risk,
# holdings, and the execution-cost model. This notebook does not estimate that model. Instead, it
# uses a dimensionless teaching proxy to isolate the intended comparative statics for an AR(1)
# persistence parameter $\varphi$ and a cost-pressure parameter $\Gamma$:
#
# $$S(\varphi, \Gamma) = \frac{\varphi}{1 - \varphi + \Gamma}$$
#
# The score rises with persistence and falls with cost pressure. It is not calibrated in bps, is
# not a retention fraction, and can exceed one. It supports scenario ranking only; it does not
# measure realized net alpha or reproduce Paleologo's full alpha-to-go optimization.

# %%
# Persistence-cost score heatmap
phi_values = np.linspace(0.1, 0.99, 50)  # persistence
gamma_values = np.linspace(0.01, 1.0, 50)  # unitless cost-pressure parameter
PHI, GAMMA = np.meshgrid(phi_values, gamma_values)

persistence_cost_score = PHI / (1 - PHI + GAMMA)
score_ticks = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 40.0])

fig = go.Figure(
    data=go.Heatmap(
        z=np.log10(persistence_cost_score),
        customdata=persistence_cost_score,
        x=np.round(phi_values, 2),
        y=np.round(gamma_values, 2),
        colorscale=[
            [0.0, COLORS["silver_muted"]],
            [0.5, COLORS["amber"]],
            [1.0, COLORS["blue"]],
        ],
        colorbar=dict(
            title="Unitless score<br>(log scale)",
            tickvals=np.log10(score_ticks),
            ticktext=[f"{tick:g}" for tick in score_ticks],
        ),
        hovertemplate=(
            "Persistence=%{x:.2f}<br>Cost pressure=%{y:.2f}"
            "<br>Score=%{customdata:.2f}<extra></extra>"
        ),
    )
)
fig.update_layout(
    title=(
        "Persistence lifts the teaching score while cost pressure lowers it"
        "<br><sup>Illustrative proxy only; not a calibrated alpha-to-go estimate</sup>"
    ),
    xaxis_title="Signal Persistence (φ)",
    yaxis_title="Unitless Cost Pressure (Γ)",
    height=500,
    margin=dict(t=105),
)
fig.show()

# %% [markdown]
# **Interpretation**: The score is highest at high persistence and low cost pressure, in the
# bottom-right of the heatmap. Its scale is deliberately not interpreted as a fraction of alpha.
# The logarithmic color scale keeps the rest of the surface visible despite the sharp corner peak;
# it changes only the color mapping, not the score or its ordering. The surface demonstrates that
# persistence and cost can change a signal ranking.

# %%
# Illustrative signal reranking demo
signals = pl.DataFrame(
    {
        "signal": ["Momentum 1m", "Momentum 6m", "Value", "Quality"],
        "raw_ic": [0.04, 0.03, 0.025, 0.02],
        "persistence": [0.3, 0.85, 0.95, 0.92],
        "gamma": [0.8, 0.3, 0.1, 0.05],
    }
)
signals = signals.with_columns(
    (
        pl.col("raw_ic") * pl.col("persistence") / (1 - pl.col("persistence") + pl.col("gamma"))
    ).alias("priority_score")
)
signals = signals.with_columns(
    pl.col("raw_ic").rank(descending=True).alias("raw_rank"),
    pl.col("priority_score").rank(descending=True).alias("score_rank"),
)

# %%
fig = go.Figure()
rank_colors = [COLORS["blue"], COLORS["amber"], COLORS["slate"], COLORS["copper"]]
for row, color in zip(signals.sort("raw_rank").iter_rows(named=True), rank_colors, strict=True):
    fig.add_trace(
        go.Scatter(
            x=["Raw IC rank", "Persistence-cost score rank"],
            y=[row["raw_rank"], row["score_rank"]],
            mode="lines+markers+text",
            name=row["signal"],
            line=dict(color=color, width=2),
            marker=dict(size=9),
            text=[row["signal"], row["signal"]],
            textposition=["top center", "middle right"],
            cliponaxis=False,
        )
    )
fig.update_layout(
    title=(
        "Persistence and cost assumptions can reverse a raw-signal ranking"
        "<br><sup>Illustrative inputs; the score is not a measured cost-adjusted IC</sup>"
    ),
    xaxis_title="Ranking basis",
    yaxis_title="Rank (1 = highest)",
    yaxis=dict(autorange="reversed", tickmode="linear", dtick=1),
    height=500,
    showlegend=False,
    margin=dict(t=105, l=125, r=145),
)
fig.show()

# %% [markdown]
# **Interpretation**: In these hypothetical inputs, short-horizon momentum starts with the highest
# raw IC but ranks last on the persistence-cost proxy. Value and quality move up because their
# assumed persistence is higher and cost pressure is lower. The exercise demonstrates sensitivity
# to assumptions; it is not an empirical comparison of these signals.

# %% [markdown]
# ## 10. Summary Statistics

# %%
# Final summary table uses each cadence's measured gross returns and turnover.
summary_data = []
for costs in [HIGH_FRICTION_COSTS, MEDIUM_FRICTION_COSTS]:
    for freq, params in FREQUENCIES.items():
        be_alpha = calculate_break_even_alpha(params["annual_turnover"], costs.round_trip)
        net_row = (
            real_net_by_frequency(costs).filter(pl.col("frequency") == freq).row(0, named=True)
        )

        summary_data.append(
            {
                "Cost Scenario": costs.name.removesuffix(" scenario").title(),
                "Frequency": freq,
                "Annual TO (x)": round(params["annual_turnover"], 1),
                "Gross SR": round(params["gross_sharpe"], 2),
                "Break-even Alpha (bps)": round(be_alpha),
                "Net Sharpe": round(net_row["net_sharpe"], 2),
            }
        )

summary_df = pl.DataFrame(summary_data)
summary_df

# %% [markdown]
# **Finding**: The summary table compresses the notebook into a usable trading
# rule. Frequency choice should be driven by net Sharpe and break-even alpha
# jointly, not by gross performance or turnover in isolation.

# %% [markdown]
# ## 11. Key Takeaways
#
# %%
_daily_high = _high_friction_results.filter(pl.col("frequency") == "Daily").row(0, named=True)
_monthly_high = _high_friction_results.filter(pl.col("frequency") == "Monthly").row(0, named=True)
display(
    Markdown(
        f"""
1. **Break-even alpha scales with measured turnover**: daily turnover is
   {_daily["annual_turnover"]:.1f}x and requires {_daily["annual_turnover"] * HIGH_FRICTION_COSTS.round_trip:.0f}
   bps under the high-friction scenario; monthly turnover is {_monthly["annual_turnover"]:.1f}x
   and requires {_monthly["annual_turnover"] * HIGH_FRICTION_COSTS.round_trip:.0f} bps.

2. **The historical cadence comparison is descriptive**: in the fixed {GROSS_START_DATE} to
   {GROSS_END_DATE} sample,
   monthly net Sharpe is {_monthly_high["net_sharpe"]:.2f} versus
   {_daily_high["net_sharpe"]:.2f} daily under the high-friction stack. No sealed holdout or
   production-optimal cadence is claimed.

3. **Cost labels are scenarios, not trader estimates**: each stack is a transparent parameterization
   that readers can replace with their own spread, impact, and commission estimates.

4. **Signal decay can change the ranking**: the parametric study shows when acting sooner can offset
   extra turnover, conditional on the stated gross Sharpe, volatility, decay, and cost assumptions.

5. **The persistence-cost score is a teaching proxy**: it demonstrates comparative statics and
   reranking, but it is neither calibrated alpha-to-go nor measured cost-adjusted IC.

6. **Practical rule**: increase frequency only when an independently estimated signal half-life and
   implementable cost model support the extra turnover.
"""
    )
)

# %% [markdown]
# **Next**: See [`10_gross_vs_net_performance`](10_gross_vs_net_performance.ipynb) for full gross-to-net waterfall analysis.
# **Book**: Chapter 18, Section 18.8 discusses practical guardrails for execution costs.
