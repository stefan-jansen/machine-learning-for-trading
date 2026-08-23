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
# # Almgren-Chriss Optimal Execution
#
# **Docker image**: `ml4t`
#
# The previous notebook's schedules were fixed in advance and scored on how closely they tracked
# the day's average price. This one asks a different question: given that trading faster costs more
# in impact and trading slower leaves the position exposed to the market longer, what schedule
# balances the two best - and how does a trader express which of the two worries them more?
#
# **Implementation shortfall** is the benchmark that makes the question well posed. It measures the
# execution against the **arrival price**, the price at the moment the decision to trade was made.
# Under that benchmark a position still unsold when the price falls has cost real money, so
# trading slowly is a risk rather than a virtue.
#
# Almgren and Chriss (2001) solve the resulting problem in closed form. This notebook implements
# the solution, traces out the set of schedules that are not dominated by any other, and simulates
# what each would have paid across many price paths.
#
# **Learning Objectives**
# - State the two costs an execution schedule trades off, and say which one grows with speed
# - Compute the Almgren-Chriss trajectory for a given aversion to timing risk, and read the
#   resulting schedule as a liquidation half-life rather than as an abstract parameter
# - Trace the efficient frontier and say what a move along it buys and costs
# - Separate the part of execution cost a schedule controls from the part it cannot
# - Compare schedules on paired simulated price paths, so the comparison is not sampling noise
# - Read a transaction-cost report and say which of its lines are measurements and which are
#   assumptions
#
# **Book Reference:** Chapter 18, Section 18.6
#
# **Prerequisites:** Read [`04_vwap_twap_execution`](04_vwap_twap_execution.ipynb)
# for benchmark schedules and
# [`03_market_impact_calibration`](03_market_impact_calibration.ipynb) for the
# empirical interpretation of the impact inputs.

# %% [markdown]
# ## The Optimal Execution Problem
#
# Selling a position faces two costs that pull in opposite directions:
#
# 1. **Market impact.** Every share sold pushes the price down a little, and pushing harder in less
#    time pushes further. Selling the whole position at once pays the most impact possible.
# 2. **Timing risk.** A position not yet sold is still exposed to whatever the market does next.
#    Spreading the sale over a week means a week of price uncertainty on a shrinking position.
#
# Trading faster reduces the second and increases the first. There is no schedule that minimizes
# both, so there is no single right answer - only a set of schedules where reducing one cost
# requires accepting more of the other, and a choice about where on that set to sit.

# %% [markdown]
# ## Imports & Settings

# %%
"""Almgren-Chriss Optimal Execution - Efficient frontier, optimal trajectories, and TCA."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import Markdown, display

from utils.reproducibility import set_global_seeds
from utils.style import COLORS, ml4t_palette, show_plotly_with_alt

# %% tags=["parameters"]
N_RISK_AVERSIONS = 100
SEED = 42
N_SIMULATIONS = 1000
TRAJECTORY_RISK_AVERSIONS = [0.0, 1e-8, 1e-7, 1e-6, 1e-5]
FRONTIER_LOG10_RANGE = (-9, -5)
SIMULATION_RISK_AVERSIONS = [0.0, 1e-7, 1e-6]
REPORT_START = datetime(2021, 10, 1, 9, 30)
REPORT_INTERVAL_MINUTES = 15

# %%
set_global_seeds(SEED)

# %% [markdown]
# What each setting decides:
#
# - `TRAJECTORY_RISK_AVERSIONS` are the values of $\lambda$ whose schedules are drawn together, and
#   `FRONTIER_LOG10_RANGE` bounds the sweep behind the frontier. $\lambda$ has no natural scale: it
#   converts a variance in dollars-squared into dollars, so its magnitude depends on the position
#   size and the price. The range here is the one over which the schedule visibly changes for this
#   position; the charts label each schedule by its liquidation half-life, which does not depend on
#   those units.
# - `N_RISK_AVERSIONS` is how finely the frontier is sampled, and affects only how smooth it looks.
# - `N_SIMULATIONS` is the number of simulated price paths per schedule. It must be even, because
#   the paths are drawn in mirror-image pairs so that the average price shock is exactly zero and
#   each schedule's simulated mean lands on its analytical expected cost rather than near it.
# - `SIMULATION_RISK_AVERSIONS` selects the three schedules taken through the simulation.

# %% [markdown]
# ## Part 1: Model Setup
#
# ### Price Dynamics
#
# For a sell program, the unaffected price evolves as:
# $$S_k = S_{k-1} - \gamma n_k + \sigma_P \tau^{1/2} \xi_k$$
#
# where:
# - $S_k$ = price after period $k$
# - $\sigma_P = S_0\sigma/\sqrt{252}$ = daily price volatility in USD
# - $\tau$ = time step
# - $\xi_k \sim N(0,1)$ = random shock
# - $\gamma n_k$ = permanent price impact of selling $n_k$ shares
#
# ### Impact Functions
#
# **Permanent Impact** (information):
# $$g(n) = \gamma n$$
#
# **Temporary Impact** (liquidity):
# $$h(n) = \epsilon \, \text{sign}(n) + \eta \frac{n}{\tau}$$
#
# The two impacts differ in what happens after the order stops. **Permanent impact** stays: it is
# the market's revision of what the asset is worth given that someone was selling, and every
# subsequent trade prints at the lower price. **Temporary impact** is the extra concession paid for
# demanding liquidity right now, and it decays once the pressure is off, so it is charged to the
# slice that caused it and to nobody else.
#
# The simulated sell price subtracts the half-spread and the temporary impact in full, and half of
# the slice's own permanent impact. The half is the standard convention: a slice is filled while
# the price is being walked down by its own trading, so on average it transacts at the midpoint of
# the move it causes. It is also what makes a run with no random shocks reconcile exactly with the
# expected-cost formula in Part 2.


# %%
@dataclass
class AlmgrenChrissParams:
    """Parameters for Almgren-Chriss model."""

    X: int = 100_000  # Shares to liquidate
    T: float = 1.0  # Trading horizon in days
    N: int = 10  # Trading periods
    S0: float = 100.0  # Arrival price in USD
    sigma: float = 0.30  # Annual return volatility
    gamma: float = 25.0  # Permanent-impact bps at 100% ADV
    eta: float = 160.0  # Temporary-impact bps at 100% interval participation
    epsilon: float = 1.0  # Half-spread in bps
    ADV: float = 1_000_000  # Average daily shares

    def __post_init__(self) -> None:
        """Validate inputs and convert impact coefficients to price units."""
        positive = {"X": self.X, "T": self.T, "N": self.N, "S0": self.S0, "ADV": self.ADV}
        if any(value <= 0 for value in positive.values()):
            raise ValueError("Position, horizon, periods, price, and ADV must be positive")
        if self.eta <= 0:
            raise ValueError("Temporary-impact coefficient eta must be positive")
        if any(value < 0 for value in (self.sigma, self.gamma, self.epsilon)):
            raise ValueError("Volatility and remaining cost inputs must be nonnegative")
        self.gamma_price = self.gamma * self.S0 / (self.ADV * 10_000)
        self.eta_price = self.eta * self.S0 / (self.ADV * 10_000)
        self.epsilon_price = self.epsilon * self.S0 / 10_000

    @property
    def tau(self) -> float:
        """Time step."""
        return self.T / self.N

    @property
    def sigma_daily(self) -> float:
        """Daily return volatility."""
        return self.sigma / np.sqrt(252)

    @property
    def sigma_price_daily(self) -> float:
        """Daily price volatility in USD per share."""
        return self.S0 * self.sigma_daily


# %%
params = AlmgrenChrissParams(
    X=100_000,
    T=5.0,
    N=50,
    S0=100.0,
    sigma=0.30,
    gamma=25.0,
    eta=160.0,
    epsilon=1.0,
    ADV=1_000_000,
)

interval_volume = params.ADV * params.tau
twap_slice = params.X / params.N
print(
    f"Selling {params.X:,} shares worth ${params.X * params.S0 / 1e6:.1f}M over {params.T:g} days "
    f"in {params.N} slices\n"
    f"That is {params.X / (params.ADV * params.T):.1%} of the volume traded over the horizon; an "
    f"even slice of {twap_slice:,.0f} shares is {twap_slice / interval_volume:.1%} of the "
    f"{interval_volume:,.0f} shares an interval trades\n"
    f"Price moves {params.sigma_daily:.2%} a day, ${params.sigma_price_daily:.2f} per share, "
    f"which is what an unsold position is exposed to\n"
    f"Selling a full day's volume would move the price {params.gamma:g} bps permanently; a slice "
    f"equal to an interval's entire volume would cost {params.eta:g} bps in temporary impact\n"
    f"Crossing the spread costs {params.epsilon:g} bps, whatever the schedule"
)

# %% [markdown]
# **Where these coefficients come from.** The volatility and the spread are the scale this book's
# own measurements put them on: `02_spread_estimation` found a median quoted spread near two basis
# points on NASDAQ-100 names, half of which is what a crossing order gives up. The temporary-impact
# coefficient is set so that this model charges what the square-root model of
# `03_market_impact_calibration` charges at the participation rate traded here, and the permanent
# coefficient makes lasting impact about a quarter of the total. They are stated figures on a
# defensible scale, not estimates from execution records - which is what calibrating them properly
# would require.

# %% [markdown]
# ## Part 2: Execution Cost and Risk
#
# ### Expected Cost (Implementation Shortfall)
#
# $$E[C] = \frac{1}{2} \gamma X^2 + \epsilon X + \eta \sum_{k=1}^{N} \frac{n_k^2}{\tau}$$
#
# where $n_k$ is shares traded in period $k$.
#
# ### Execution Risk (Variance)
#
# $$V[C] = \sigma_P^2 \sum_{k=1}^{N} \tau \, x_k^2$$
#
# where $x_k$ is the post-trade position exposed to the next price shock.


# %%
def compute_trajectory_from_list(trade_list: np.ndarray, X: int) -> np.ndarray:
    """Convert trade list to position trajectory."""
    trade_list = np.asarray(trade_list, dtype=float)
    if trade_list.ndim != 1 or not np.isfinite(trade_list).all():
        raise ValueError("trade_list must be a finite one-dimensional array")
    if (trade_list < 0).any() or not np.isclose(trade_list.sum(), X):
        raise ValueError("A liquidation schedule must be nonnegative and sum to X")
    trajectory = np.zeros(len(trade_list) + 1)
    trajectory[0] = X
    for k, n in enumerate(trade_list):
        trajectory[k + 1] = trajectory[k] - n
    return trajectory


# %% [markdown]
# #### Expected Cost Functional


# %%
def expected_cost(
    trade_list: np.ndarray,
    params: AlmgrenChrissParams,
) -> float:
    """Compute expected cost (implementation shortfall)."""
    trade_list = np.asarray(trade_list, dtype=float)
    compute_trajectory_from_list(trade_list, params.X)
    X = params.X
    tau = params.tau
    gamma = params.gamma_price
    eta = params.eta_price
    epsilon = params.epsilon_price

    perm_cost = 0.5 * gamma * X**2
    fixed_cost = epsilon * abs(X)
    temp_cost = eta * np.sum(trade_list**2) / tau

    return perm_cost + fixed_cost + temp_cost


# %% [markdown]
# #### Execution Variance Functional


# %%
def execution_variance(
    trade_list: np.ndarray,
    params: AlmgrenChrissParams,
) -> float:
    """Compute execution variance (timing risk)."""
    trajectory = compute_trajectory_from_list(trade_list, params.X)
    sigma = params.sigma_price_daily
    tau = params.tau

    # Each shock occurs after that period's trade, so it reaches post-trade inventory.
    return sigma**2 * tau * np.sum(trajectory[1:] ** 2)


# %% [markdown]
# #### Execution Standard Deviation Helper


# %%
def execution_std(trade_list: np.ndarray, params: AlmgrenChrissParams) -> float:
    """Compute execution standard deviation."""
    return np.sqrt(execution_variance(trade_list, params))


# %%
twap_trades = np.full(params.N, params.X / params.N)

aggressive_trades = np.zeros(params.N)
aggressive_trades[0] = params.X  # All in first period

gradual_trades = np.zeros(params.N)
gradual_trades[:5] = params.X / 5  # First 5 periods

strategies = {
    "TWAP": twap_trades,
    "Aggressive": aggressive_trades,
    "Front-loaded": gradual_trades,
}

notional = params.X * params.S0
strategy_comparison = pl.DataFrame(
    [
        {
            "strategy": name,
            "spread_bps": params.epsilon_price * params.X / notional * 10_000,
            "permanent_bps": 0.5 * params.gamma_price * params.X**2 / notional * 10_000,
            "temporary_bps": params.eta_price * np.sum(trades**2) / params.tau / notional * 10_000,
            "total_cost_bps": expected_cost(trades, params) / notional * 10_000,
            "risk_bps": execution_std(trades, params) / notional * 10_000,
        }
        for name, trades in strategies.items()
    ]
)
strategy_comparison

# %% [markdown]
# **Reading the table**: Only one of the three cost columns responds to the schedule. The spread is
# paid on every share however they are sold, and the permanent impact of liquidating the whole
# position is the same whether it takes an hour or a week - which is why both columns are constant
# down the table. Temporary impact is the schedule's own doing, and it is what the optimization has
# to work with. The risk column is what the schedule buys with it.
#
# Selling everything at once carries no timing risk at all, because nothing is left to be exposed.
# It pays for that with the largest temporary impact available. Selling evenly does the reverse.
# Neither is optimal; they are two corners of a space the next section maps.

# %% [markdown]
# ## Part 3: Optimal Trajectory
#
# ### Mean-Variance Optimization
#
# The trader minimizes:
# $$\min_{n_1, ..., n_N} E[C] + \lambda V[C]$$
#
# where $\lambda$ is the risk aversion parameter.
#
# ### Exact Finite-Period Solution
#
# For the discrete objective used here, define:
# $$\theta = \frac{\lambda \sigma_P^2 \tau^2}{\eta}, \qquad
# \alpha = \operatorname{arccosh}\left(1 + \frac{\theta}{2}\right)
# = 2\operatorname{asinh}\left(\frac{\sqrt{\theta}}{2}\right).$$
#
# The exact inventory path over $N$ periods is:
# $$x_k = X \frac{\sinh\left(\alpha(N-k)\right)}{\sinh(\alpha N)}.$$
#
# The familiar continuous-time parameter is the small-step approximation
# $\kappa = \alpha/\tau \approx \sqrt{\lambda\sigma_P^2/\eta}$. Using that
# approximation directly can misstate the optimum when $\theta$ is not small.
# The numerical scale of $\lambda$ still depends on the dollar units of
# $\sigma_P$ and $\eta$, so the sweep is reported rather than treated as a
# universal calibration.


# %%
def optimal_trajectory(
    params: AlmgrenChrissParams,
    risk_aversion: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the exact finite-period Almgren-Chriss trajectory.

    Parameters
    ----------
    params : AlmgrenChrissParams
    risk_aversion : float
        Risk aversion parameter λ (higher = more aggressive)

    Returns
    -------
    times : array
        Time points
    trajectory : array
        Optimal position at each time
    """
    X = params.X
    T = params.T
    N = params.N
    tau = params.tau
    sigma = params.sigma_price_daily
    eta = params.eta_price

    if risk_aversion < 0:
        raise ValueError("risk_aversion must be nonnegative")
    times = np.linspace(0, T, N + 1)
    theta = risk_aversion * sigma**2 * tau**2 / eta
    if theta == 0:
        trajectory = X * (1 - times / T)
    else:
        alpha = 2 * np.arcsinh(np.sqrt(theta) / 2)
        periods_remaining = np.arange(N, -1, -1, dtype=float)
        scaled_remaining = alpha * periods_remaining
        scaled_horizon = alpha * N
        numerator = np.exp(scaled_remaining - scaled_horizon) * -np.expm1(-2 * scaled_remaining)
        denominator = -np.expm1(-2 * scaled_horizon)
        trajectory = X * numerator / denominator
    trajectory[-1] = 0.0

    return times, trajectory


# %% [markdown]
# #### Convert Trajectory to Trade List


# %%
def trajectory_to_trades(trajectory: np.ndarray) -> np.ndarray:
    """Convert position trajectory to trade list."""
    return -np.diff(trajectory)


# %% [markdown]
# ### Reading a Schedule as a Half-Life
#
# $\lambda$ is hard to interpret directly, so each schedule below is also labelled by its
# **liquidation half-life**: the elapsed time at which half the position has been sold. An even
# schedule has a half-life of exactly half the horizon, and an urgent one reaches half sold much
# sooner. That number is comparable across positions and prices in a way $\lambda$ is not.


# %%
def liquidation_half_life(times: np.ndarray, trajectory: np.ndarray) -> float:
    """Elapsed time at which the remaining position first falls to half its starting size."""
    remaining = trajectory / trajectory[0]
    return float(np.interp(0.5, remaining[::-1], times[::-1]))


# %%
fig = go.Figure()
trajectory_styles = [
    dict(color=COLORS["neutral"], dash="dash", width=1.5),
    dict(color=COLORS["silver"], dash="dot", width=1.5),
    dict(color=COLORS["slate"], dash="dashdot", width=1.5),
    dict(color=COLORS["amber"], dash="solid", width=2),
    dict(color=COLORS["blue"], dash="solid", width=3),
]

for line_style, lambda_ in zip(trajectory_styles, TRAJECTORY_RISK_AVERSIONS):
    times, traj = optimal_trajectory(params, lambda_)
    half_life = liquidation_half_life(times, traj)
    label = f"λ = {lambda_:.0e}" if lambda_ else "λ = 0 (even schedule)"
    _ = fig.add_scatter(
        x=times,
        y=traj / params.X,
        mode="lines",
        name=f"{label}, half sold by day {half_life:.2f}",
        line=line_style,
    )

fig.update_layout(
    title="Aversion to timing risk bends the schedule forward",
    xaxis_title="Elapsed execution time (days)",
    yaxis_title="Remaining position",
    yaxis_tickformat=".0%",
    yaxis_range=[0, 1],
    height=500,
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
)
show_plotly_with_alt(
    fig,
    "Five curves of remaining position against elapsed time, all starting fully invested and "
    "ending at zero. The zero-aversion curve is a straight diagonal; each higher aversion bows "
    "further above it early and flattens along the bottom, selling most of the position in the "
    "first part of the horizon.",
)

# %% [markdown]
# **Reading the chart**: The straight diagonal is the even schedule, which sells the same number of
# shares every period and has no opinion about risk. Each curve above it sells faster early and
# holds a smaller position through the rest of the horizon, which is exactly what reduces the
# exposure the variance term charges for. The half-lives in the legend say how much faster, in
# days, without reference to the units $\lambda$ happens to be measured in.

# %% [markdown]
# ## Part 4: Efficient Frontier of Execution
#
# Sweeping $\lambda$ across its range traces out every schedule the model considers worth
# considering. Plotting each one's expected cost against its risk gives the **efficient frontier**:
# the set of schedules for which no other schedule is both cheaper and less risky. Anything below
# and to the left of the curve is unattainable; anything above and to the right is dominated by a
# point on it.
#
# The curve is what makes the choice concrete. Moving along it is not a matter of finding the
# optimum - every point on it is optimal for someone - but of deciding how many basis points of
# expected cost a basis point of reduced uncertainty is worth.


# %%
def compute_efficient_frontier(
    params: AlmgrenChrissParams,
    n_points: int = 50,
) -> pl.DataFrame:
    """Compute the efficient frontier of execution strategies."""
    lambdas = np.logspace(*FRONTIER_LOG10_RANGE, n_points)

    results = []
    for lambda_ in lambdas:
        _, traj = optimal_trajectory(params, lambda_)
        trades = trajectory_to_trades(traj)

        cost = expected_cost(trades, params)
        risk = execution_std(trades, params)

        results.append(
            {
                "lambda": lambda_,
                "expected_cost": cost,
                "cost_bps": cost / (params.X * params.S0) * 10000,
                "risk_std": risk,
                "risk_bps": risk / (params.X * params.S0) * 10000,
            }
        )

    return pl.DataFrame(results)


# %%
frontier = compute_efficient_frontier(params, n_points=N_RISK_AVERSIONS)

strategies_to_mark = [
    (1e-8, "Patient (λ=1e-8)"),
    (1e-7, "Balanced (λ=1e-7)"),
    (1e-6, "Urgent (λ=1e-6)"),
]

# %%
fig = go.Figure()
frontier_for_plot = frontier.sort("risk_bps")
_ = fig.add_scatter(
    x=frontier_for_plot["risk_bps"].to_list(),
    y=frontier_for_plot["cost_bps"].to_list(),
    mode="lines",
    name="Efficient Frontier",
    line=dict(color=COLORS["blue"], width=3),
)
for color, (lambda_, name) in zip(ml4t_palette(3, categorical=True), strategies_to_mark):
    row = (
        frontier.with_columns(distance=(pl.col("lambda").log10() - np.log10(lambda_)).abs())
        .sort("distance")
        .row(0, named=True)
    )
    _ = fig.add_scatter(
        x=[row["risk_bps"]],
        y=[row["cost_bps"]],
        mode="markers",
        marker=dict(size=11, symbol="star", color=color),
        name=name,
    )
fig.update_layout(
    title="Lower timing risk requires accepting higher expected cost",
    xaxis_title="Implementation-shortfall standard deviation (bps)",
    yaxis_title="Expected implementation shortfall (bps)",
    height=500,
)
show_plotly_with_alt(
    fig,
    "The efficient frontier as a convex curve falling from left to right: high expected cost with "
    "low risk at the upper left, low cost with high risk at the lower right. Three starred points "
    "mark the patient, balanced and urgent schedules along it. The curve is steep at the urgent "
    "end and nearly flat at the patient end.",
)

# %%
cheapest = frontier.sort("cost_bps").row(0, named=True)
safest = frontier.sort("risk_bps").row(0, named=True)
print(
    f"Cheapest schedule on the frontier: {cheapest['cost_bps']:.2f} bps expected cost, "
    f"{cheapest['risk_bps']:.1f} bps risk\n"
    f"Least risky schedule:              {safest['cost_bps']:.2f} bps expected cost, "
    f"{safest['risk_bps']:.1f} bps risk\n"
    f"Buying that reduction in risk costs "
    f"{safest['cost_bps'] - cheapest['cost_bps']:.2f} bps of expected cost, or "
    f"{(safest['cost_bps'] - cheapest['cost_bps']) / (cheapest['risk_bps'] - safest['risk_bps']):.3f}"
    " bps of cost per basis point of risk removed, averaged across the frontier"
)

# %% [markdown]
# **Reading the frontier**: The curve is far from straight, and that is the part worth carrying
# away. At the patient end it is nearly flat: a large reduction in risk costs almost nothing,
# because the schedule is barely trading faster than an even one. At the urgent end it is steep,
# and each further reduction costs several times what the last one did. A trader with no strong
# view about urgency is giving away the cheap part of the curve by staying at the flat end.

# %% [markdown]
# ## Part 5: What Each Schedule Would Actually Have Paid
#
# The frontier reports an expected cost and a standard deviation. Neither says what the
# distribution of outcomes looks like, and a trader cares about the bad tail rather than the
# second moment.
#
# Each schedule is therefore run against a thousand simulated price paths. The paths are shared:
# schedule A and schedule B see the identical sequence of shocks on simulation 37, so any
# difference between them is the schedule and not the draw. The paths also come in mirror-image
# pairs - every path is run again with the sign of every shock flipped - so the shocks average to
# exactly zero and each schedule's simulated mean lands on its analytical expected cost instead of
# somewhere near it.


# %%
def simulate_single_execution(
    params: AlmgrenChrissParams,
    trade_list: np.ndarray,
    shocks: np.ndarray,
) -> dict:
    """Simulate one execution path under Almgren-Chriss price dynamics.

    `shocks` is a pre-generated array of standard-normal draws (one per period)
    so that different strategies can be evaluated against the *same* price path
    for a given simulation index - a paired Monte Carlo design that removes
    sampling noise from the strategy comparison.
    """
    trade_list = np.asarray(trade_list, dtype=float)
    shocks = np.asarray(shocks, dtype=float)
    compute_trajectory_from_list(trade_list, params.X)
    if shocks.shape != trade_list.shape or not np.isfinite(shocks).all():
        raise ValueError("shocks must be finite with one value per trade")
    tau = params.tau
    sigma = params.sigma_price_daily
    gamma = params.gamma_price
    eta = params.eta_price
    epsilon = params.epsilon_price
    price = params.S0
    total_proceeds = 0.0
    prices = [price]

    for k, n_k in enumerate(trade_list):
        exec_price = price - 0.5 * gamma * n_k - epsilon - eta * n_k / tau
        total_proceeds += exec_price * n_k
        price = price - gamma * n_k
        price = price + sigma * np.sqrt(tau) * shocks[k]
        prices.append(price)

    avg_exec_price = total_proceeds / params.X
    is_dollar = params.S0 * params.X - total_proceeds
    is_bps = is_dollar / (params.S0 * params.X) * 10000
    return {
        "avg_exec_price": avg_exec_price,
        "final_price": prices[-1],
        "is_dollar": is_dollar,
        "is_bps": is_bps,
        "total_proceeds": total_proceeds,
    }


# %% [markdown]
# ### Monte Carlo Wrapper for Execution Paths


# %%
def simulate_execution(
    params: AlmgrenChrissParams,
    trade_list: np.ndarray,
    n_simulations: int = 1000,
    seed: int = SEED,
) -> pl.DataFrame:
    """
    Simulate execution with price dynamics.

    Simulation indices share paths across strategies. Consecutive indices use
    antithetic shocks from ``np.random.default_rng(seed + sim // 2)`` so an even
    run count centers each strategy on its analytical expected cost.

    Returns
    -------
    DataFrame with simulation results
    """
    if n_simulations <= 0 or n_simulations % 2:
        raise ValueError("n_simulations must be a positive even integer")
    results = []

    for sim in range(n_simulations):
        pair_index = sim // 2
        rng = np.random.default_rng(seed + pair_index)
        shocks = rng.standard_normal(len(trade_list))
        if sim % 2:
            shocks = -shocks
        result = simulate_single_execution(params, trade_list, shocks)
        result["simulation"] = sim
        results.append(result)

    return pl.DataFrame(results)


# %%
simulation_results = {}

simulation_labels = [
    "Even schedule (λ=0)",
    "Balanced (λ=1e-7)",
    "Urgent (λ=1e-6)",
]
for name, risk_aversion in zip(simulation_labels, SIMULATION_RISK_AVERSIONS):
    _, traj = optimal_trajectory(params, risk_aversion)
    trades = trajectory_to_trades(traj)
    sim_df = simulate_execution(params, trades, n_simulations=N_SIMULATIONS)
    simulation_results[name] = sim_df

# %% [markdown]
# #### TCA Summary Statistics

# %%
tca_summary = pl.DataFrame(
    [
        {
            "strategy": name,
            "is_mean_bps": df["is_bps"].mean(),
            "is_std_bps": df["is_bps"].std(),
            "is_p5_bps": df["is_bps"].quantile(0.05),
            "is_p95_bps": df["is_bps"].quantile(0.95),
        }
        for name, df in simulation_results.items()
    ]
)
tca_summary

# %% [markdown]
# **Reading the table**: The mean is the analytical expected cost, recovered by the simulation. The
# two percentiles are what the mean hides: the range within which nine of ten simulated executions
# landed. A negative shortfall means the execution beat the arrival price, which happens whenever
# the market moved favourably while the position was still being sold - so a wide distribution
# contains both the good outcomes and the bad ones, and it is the same width that produces both.

# %%
fig = go.Figure()
distribution_colors = ml4t_palette(len(simulation_results), categorical=True)
for color, (name, df) in zip(distribution_colors, simulation_results.items()):
    shortfall = np.sort(df["is_bps"].to_numpy())
    cumulative_probability = np.arange(1, len(shortfall) + 1) / len(shortfall)
    _ = fig.add_scatter(
        x=shortfall,
        y=cumulative_probability,
        mode="lines",
        name=name,
        line=dict(color=color, width=2),
    )
risk_neutral_row = tca_summary.row(0, named=True)
risk_averse_row = tca_summary.row(-1, named=True)
dispersion_reduction = risk_neutral_row["is_std_bps"] - risk_averse_row["is_std_bps"]
mean_cost_increase = risk_averse_row["is_mean_bps"] - risk_neutral_row["is_mean_bps"]
fig.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"], line_width=1)
fig.update_layout(
    title="Trading sooner narrows the range of outcomes and shifts it right",
    xaxis_title="Implementation shortfall (bps; positive is worse)",
    yaxis_title="Cumulative probability",
    yaxis_tickformat=".0%",
    yaxis_range=[0, 1],
    height=500,
)
show_plotly_with_alt(
    fig,
    "Three cumulative distribution curves of implementation shortfall. The even schedule's curve "
    "is the shallowest and spans the widest range; the urgent schedule's is much steeper and "
    "narrower, and sits slightly to the right of the others at its centre.",
)

# %% tags=["results"]
display(
    Markdown(
        f"**Reading the curves:** Moving from the even schedule to the urgent one cuts the "
        f"standard deviation of the shortfall by **{dispersion_reduction:.0f} bps**, from "
        f"**{risk_neutral_row['is_std_bps']:.0f}** to **{risk_averse_row['is_std_bps']:.0f}**, and "
        f"raises the mean cost by **{mean_cost_increase:.2f} bps**. The steeper curve is the "
        "narrower distribution: nearly all of its probability is packed into a small range, while "
        "the even schedule's spreads across a range several times as wide in both directions."
    )
)

# %% [markdown]
# ## Part 6: What the Linear Impact Assumption Costs
#
# The closed form in Part 3 exists because temporary impact is linear in the trade rate. That is a
# modelling convenience, and `03_market_impact_calibration` argues the shape is closer to a square
# root. This section prices the same schedule under both to see how much the assumption is worth.
#
# The comparison only means something if the two models are put on the same scale first. An
# exponent and a coefficient are not independent: reusing a coefficient fitted for one exponent
# under another changes the level as well as the shape, and the resulting difference says nothing
# about which shape is right. Both models are therefore matched to charge the same amount at a
# stated reference participation, so what remains between them is the shape alone.
#
# Three further extensions are worth knowing about and are not built here: liquidity that varies
# through the day, so an even schedule is not even in participation terms; volatility that changes
# with the market regime; and a decaying signal, which gives a reason to hurry that has nothing to
# do with risk aversion.


# %% [markdown]
# #### Non-Linear Impact Extension


# %%
@dataclass
class ExtendedParams(AlmgrenChrissParams):
    """Extended parameters with non-linear impact."""

    impact_exponent: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.impact_exponent <= 1:
            raise ValueError("impact_exponent must lie in (0, 1]")


# %% [markdown]
# #### Non-Linear Cost Functional


# %%
def expected_cost_nonlinear(
    trade_list: np.ndarray,
    params: ExtendedParams,
    reference_participation: float,
) -> float:
    """Expected cost with a power-law temporary impact matched to the linear model.

    The coefficient is rescaled so both models charge the same per-share concession at
    `reference_participation`, leaving the exponent as the only difference between them.
    """
    trade_list = np.asarray(trade_list, dtype=float)
    compute_trajectory_from_list(trade_list, params.X)
    if not 0 < reference_participation <= 1:
        raise ValueError("reference_participation must lie in (0, 1]")
    X = params.X
    tau = params.tau
    gamma = params.gamma_price
    epsilon = params.epsilon_price
    alpha = params.impact_exponent

    perm_cost = 0.5 * gamma * X**2
    fixed_cost = epsilon * abs(X)
    interval_market_volume = params.ADV * tau
    participation = trade_list / interval_market_volume
    # Linear charges eta * p at p = reference; the power law charges eta_matched * p**alpha.
    eta_matched = params.eta * reference_participation ** (1 - alpha)
    temp_cost = params.S0 / 10_000 * eta_matched * np.sum(trade_list * participation**alpha)

    return perm_cost + fixed_cost + temp_cost


# %%
extended_params = ExtendedParams(
    X=100_000,
    T=5.0,
    N=50,
    S0=100.0,
    sigma=0.30,
    gamma=25.0,
    eta=160.0,
    epsilon=1.0,
    ADV=1_000_000,
    impact_exponent=0.5,
)

twap_trades = np.full(extended_params.N, extended_params.X / extended_params.N)
reference_participation = twap_slice / interval_volume

print(
    f"Both models matched at {reference_participation:.1%} participation, the rate an even "
    f"schedule trades at\n"
)
print(f"{'schedule':<14}{'participation':>14}{'linear':>12}{'square root':>14}{'ratio':>9}")
for name, trades in strategies.items():
    linear_cost = expected_cost(trades, params)
    nonlinear_cost = expected_cost_nonlinear(trades, extended_params, reference_participation)
    peak = trades.max() / interval_volume
    print(
        f"{name:<14}{peak:>13.0%} {linear_cost:>11,.0f} {nonlinear_cost:>13,.0f} "
        f"{nonlinear_cost / linear_cost:>8.2f}"
    )

# %% [markdown]
# **Reading the comparison**: Matched at the even schedule's own participation rate, the two models
# agree on it almost exactly - the small residual is the fixed spread and permanent impact, which
# neither exponent touches. Away from that point they diverge in the direction the square root
# implies: a schedule that concentrates the order into fewer, larger slices runs at a much higher
# participation, and the square-root model charges it less than the linear one does, because the
# concession per share grows more slowly than proportionally.
#
# The practical consequence runs the other way from what the arithmetic first suggests. A linear
# model overstates the cost of trading fast, so a trader optimizing against it front-loads less
# than they should. The exponent is not a detail of the cost estimate; it changes the schedule.

# %% [markdown]
# ## Part 7: Reading a Transaction Cost Report
#
# **Transaction cost analysis** is what a desk produces after the fact: the fills that happened,
# scored against benchmarks. Three benchmarks answer three different questions.
#
# - Against the **arrival price**, the question is what the decision cost from the moment it was
#   made. This is implementation shortfall, and it charges the schedule for the market moving while
#   it worked.
# - Against the day's **volume-weighted average price**, the question is whether the execution was
#   better or worse than average participation that day. It says nothing about whether that day was
#   a good day to trade.
# - Against the **closing price**, the question is what the position would have cost someone who
#   waited until the end.
#
# The report below is built from the balanced schedule run through one simulated price path, so
# every fill in it came out of the model above rather than being typed in.


# %%
def generate_tca_report(
    executed_trades: pl.DataFrame,
    benchmark_prices: dict,
    params: AlmgrenChrissParams,
) -> dict:
    """Score a set of fills against arrival, VWAP and closing benchmarks."""
    required = {"timestamp", "shares", "exec_price"}
    if not required.issubset(executed_trades.columns):
        raise ValueError(f"executed_trades must contain {sorted(required)}")
    if (executed_trades["shares"] <= 0).any() or (executed_trades["exec_price"] <= 0).any():
        raise ValueError("shares and execution prices must be positive")
    total_shares = executed_trades["shares"].sum()
    total_value = (executed_trades["shares"] * executed_trades["exec_price"]).sum()
    avg_price = total_value / total_shares

    arrival = benchmark_prices["arrival"]
    vwap = benchmark_prices["vwap"]
    close = benchmark_prices["close"]
    if min(arrival, vwap, close) <= 0:
        raise ValueError("benchmark prices must be positive")

    return {
        "total_shares": total_shares,
        "total_value": total_value,
        "avg_exec_price": avg_price,
        "vs_arrival_bps": (arrival - avg_price) / arrival * 10000,
        "vs_vwap_bps": (vwap - avg_price) / vwap * 10000,
        "vs_close_bps": (close - avg_price) / close * 10000,
        "assumed_half_spread_bps": params.epsilon,
    }


# %% [markdown]
# ### Produce One Execution to Report On
#
# The balanced schedule is run against a single simulated price path, and every fill it produced is
# recorded with the price it printed at. The market's own volume-weighted average price over the
# same path is the natural VWAP benchmark, and its last price is the close.


# %%
def executed_fills(
    params: AlmgrenChrissParams,
    trade_list: np.ndarray,
    shocks: np.ndarray,
) -> tuple[pl.DataFrame, dict]:
    """Return the fills one simulated execution produced, and the benchmarks to score them on."""
    tau, sigma = params.tau, params.sigma_price_daily
    gamma, eta, epsilon = params.gamma_price, params.eta_price, params.epsilon_price
    price = params.S0
    fills, unaffected = [], []
    for k, n_k in enumerate(trade_list):
        exec_price = price - 0.5 * gamma * n_k - epsilon - eta * n_k / tau
        fills.append(
            {
                "timestamp": REPORT_START + timedelta(minutes=REPORT_INTERVAL_MINUTES * k),
                "shares": float(n_k),
                "exec_price": exec_price,
            }
        )
        unaffected.append(price)
        price = price - gamma * n_k + sigma * np.sqrt(tau) * shocks[k]
    prices = np.asarray(unaffected)
    return pl.DataFrame(fills), {
        "arrival": params.S0,
        # Every interval trades the same market volume in this model, so the volume-weighted
        # average price over the horizon is the plain mean of the interval prices.
        "vwap": float(prices.mean()),
        "close": float(price),
    }


# %%
_, balanced_traj = optimal_trajectory(params, SIMULATION_RISK_AVERSIONS[1])
balanced_trades = trajectory_to_trades(balanced_traj)
report_shocks = np.random.default_rng(SEED).standard_normal(len(balanced_trades))
example_trades, benchmarks = executed_fills(params, balanced_trades, report_shocks)
tca = generate_tca_report(example_trades, benchmarks, params)

# %%
tca_report = pl.DataFrame(
    [
        {"metric": "Shares executed", "value": f"{tca['total_shares']:,.0f}", "kind": "measured"},
        {"metric": "Proceeds ($)", "value": f"{tca['total_value']:,.0f}", "kind": "measured"},
        {
            "metric": "Average fill price ($)",
            "value": f"{tca['avg_exec_price']:.4f}",
            "kind": "measured",
        },
        {
            "metric": "vs arrival price (bps)",
            "value": f"{tca['vs_arrival_bps']:+.1f}",
            "kind": "measured",
        },
        {
            "metric": "vs market VWAP (bps)",
            "value": f"{tca['vs_vwap_bps']:+.1f}",
            "kind": "measured",
        },
        {
            "metric": "vs closing price (bps)",
            "value": f"{tca['vs_close_bps']:+.1f}",
            "kind": "measured",
        },
        {
            "metric": "Half-spread charged (bps)",
            "value": f"{tca['assumed_half_spread_bps']:.1f}",
            "kind": "assumed",
        },
    ]
)
tca_report

# %% [markdown]
# **Reading the report**: The `kind` column is the point of the table. The first six rows are
# arithmetic on fills that happened and prices that were observed. The last is an input that was
# chosen, and it appears in the report because it was subtracted from every fill.
#
# The closing-price line is the one to be most careful with. Over this horizon the price wanders
# by hundreds of basis points on its own, so on any single path that line reports where the random
# walk happened to end rather than anything about the execution. It is informative averaged over
# many executions and close to meaningless on one, which is why Part 5 ran a thousand.
#
# What the report cannot do is split the shortfall into impact and timing. Doing that requires
# knowing what the price would have done had the order not been sent, and that price does not
# exist anywhere - it is a counterfactual. A report that presents an impact number and a timing
# number is reporting a model's opinion, and the honest version says which model.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Separate the part of execution cost the schedule controls from the part it does not.** The
#    spread is paid on every share and the permanent impact of liquidating a position is the same
#    however long it takes. Only temporary impact responds to the schedule, and only it is worth
#    optimizing. Reporting a total that is mostly fixed cost hides how much room there was.
#
# 2. **Express urgency as a schedule, not as an adjective.** The model turns an aversion to timing
#    risk into a specific trajectory, and the trajectory is comparable across positions once it is
#    read as a half-life. "Trade this one aggressively" is not an instruction anyone can audit.
#
# 3. **The frontier is curved, and where you sit on it matters more than that you are on it.** At
#    the patient end, a large reduction in risk costs almost nothing; at the urgent end each
#    further reduction costs several times the last. Both ends are optimal for someone, and the
#    cheap reductions are at one end only.
#
# 4. **Compare schedules on shared price paths.** Running each schedule on its own random draws
#    means the difference between them contains sampling noise. Using the same shocks for all of
#    them, in mirror-image pairs, removes it and lets the simulated mean reproduce the analytical
#    one exactly.
#
# 5. **Keep impact coefficients in units you can check against a measurement.** Quoting temporary
#    impact as basis points at full interval participation makes it comparable with the square-root
#    coefficient fitted in `03_market_impact_calibration`; quoting it as a raw price coefficient
#    does not, and a value that is wrong by three orders of magnitude looks the same as a right one.
#
# 6. **A cost report can measure benchmark slippage and cannot measure attribution.** Splitting a
#    shortfall into impact and timing needs the price that would have prevailed had the order never
#    been sent. That price is a counterfactual, so any split is a model's output and should be
#    labelled as one.
#
# ### Known limitations
#
# - The impact coefficients are stated rather than fitted to execution records. They are set on the
#   scale this chapter's own measurements imply, which fixes the shape of every result here and
#   leaves the levels resting on that choice.
# - Temporary impact is linear in the trade rate, which is what makes the closed form possible and
#   is not what `03_market_impact_calibration` argues the shape is. Part 6 prices the same schedule
#   under a square-root exponent to show the size of the difference; it does not re-optimize under
#   it, and the optimal schedule under square-root impact is not the one shown here.
# - Volatility, liquidity and the impact coefficients are constant over the horizon. Real intraday
#   liquidity follows the profile in `03_market_impact_calibration`, so an even schedule is not
#   even in participation terms.
# - The price process has no drift and no serial correlation, so the model has no view worth
#   trading on. A decaying signal is precisely what would justify urgency on grounds other than
#   risk aversion, and it is absent here.
# - The simulation fills every slice at the modelled price. Nothing goes unfilled and nothing is
#   crossed by another participant.
#
# **Next**: `08_ml_dynamic_execution` lets the schedule respond to conditions as they arrive
# instead of committing to a trajectory in advance.
#
# **Book**: Chapter 18, Section 18.6.
