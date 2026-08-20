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
# # Mean-variance optimization, and why its answer is so unstable
#
# **Environment**: `uv run` from the repository root
#
# ## Purpose
# Markowitz's result is that for any level of risk there is a portfolio with the highest expected
# return, and that it can be solved for from each asset's expected return and the covariance
# between them. The solution is exact and the
# mathematics is not in dispute. What is in dispute is whether the two inputs can be estimated well
# enough for the answer to mean anything.
#
# This notebook builds the whole machinery from scratch - the feasible region, the frontier, the
# maximum-Sharpe and minimum-variance solutions - and then does the thing that decides whether any
# of it is useful: estimates the inputs on one period, freezes the resulting weights, and applies
# them to a later one. Alongside it run three heuristics that need no expected-return estimate at
# all, which is the comparison that matters.
#
# ## Learning objectives
#
# - Solve for the portfolio with the highest risk-adjusted return, and for the one with the lowest
#   risk, under a long-only, fully-invested constraint.
# - Draw the set of achievable risk-and-return combinations, and the curve that bounds it.
# - Measure how sensitive the covariance matrix is to small changes in its inputs, and read that
#   number as a warning about the optimizer's output rather than about the matrix.
# - Estimate every input on one window, freeze the weights, and score them on a later window
#   against baselines that estimate less.
#
# ## Book reference
# Chapter 17, Section 17.5 (mean-variance optimization and the Markowitz curse).
#
# ## Prerequisites
#
# - Daily ETF prices from the canonical dataset.

# %% [markdown]
# ## The Mean-Variance Framework
#
# MPT solves for the optimal portfolio weights to minimize volatility for a given expected return, or maximize returns for a given level of volatility. The key requisite inputs are expected asset returns, standard deviations, and the covariance matrix.
#
# Diversification works because the variance of portfolio returns depends on the covariance of the assets and can be reduced below the weighted average of the asset variances by including assets with less than perfect correlation. Given a vector, $\omega$, of portfolio weights and the covariance matrix, $\Sigma$, the portfolio variance, $\sigma_{\text{PF}}^2$ is:
#
# $$\sigma_{\text{PF}}^2=\omega^T\Sigma\omega$$
#
# Markowitz showed that maximizing expected portfolio return subject to a target risk has an equivalent dual representation of minimizing portfolio risk subject to a target expected return level, $\mu_{PF}$:
#
# $$
# \begin{align}
# \min_\omega & \quad\quad\sigma^2_{\text{PF}}= \omega^T\Sigma\omega\\
# \text{s.t.} &\quad\quad \mu_{\text{PF}}= \omega^T\mu\\
# &\quad\quad \mathbf{1}^\top \omega =1
# \end{align}
# $$
#
# We calculate an efficient frontier using `scipy.optimize.minimize` and historical estimates for asset returns, standard deviations, and the covariance matrix.

# %%
"""Construct efficient frontiers and test portfolios estimated from historical returns."""

import contextlib
import io

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from ml4t.backtest import (
    BacktestConfig,
    CommissionType,
    DataFeed,
    Engine,
    ExecutionMode,
    Strategy,
)
from ml4t.backtest.config import SlippageType
from ml4t.backtest.execution.rebalancer import RebalanceConfig, TargetWeightExecutor
from ml4t.diagnostic.evaluation import (
    PortfolioAnalysis,
)
from ml4t.diagnostic.evaluation.factor import FactorAnalysis, load_fama_french_5factor
from numpy.random import dirichlet
from scipy.optimize import minimize

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, ml4t_diverging, ml4t_palette

# %% tags=["parameters"]
# Production defaults; Papermill overrides for CI testing
N_PORTFOLIOS = 10000  # Random portfolios for frontier visualization
SEED = 42
TRAIN_END = "2019-12-31"
RISK_FREE_RATE = 0.04
COMMISSION_RATE = 0.0005
SLIPPAGE_RATE = 0.0005

# %%
set_global_seeds(SEED)

ML4T_CATEGORICAL = ml4t_palette(4, categorical=True) + [COLORS["positive"]]
ML4T_SEQUENTIAL = [COLORS["silver_muted"], COLORS["slate"], COLORS["blue"]]

# %% [markdown]
# ## Load Data
#
# We'll work with diversified ETFs from the canonical dataset to demonstrate MVO.
# ETFs are commonly used as building blocks in portfolio construction.

# %%
START = "2015-01-01"
END = "2023-12-31"

# %% [markdown]
# ### ETF Universe by Asset Class

# %%
# Diversified ETF selection across asset classes
SYMBOLS = [
    # US Equities
    "SPY",
    "QQQ",
    "IWM",
    "VTV",
    "VUG",
    # International
    "EFA",
    "EEM",
    "VEA",
    # Fixed Income
    "AGG",
    "TLT",
    "LQD",
    "HYG",
    "TIP",
    "SHY",
    # Alternatives
    "GLD",
    "SLV",
    "VNQ",
    "DBC",
    # Sectors
    "XLF",
    "XLE",
    "XLK",
    "XLV",
    "XLI",
    "XLU",
    "XLP",
    "XLY",
    "XLB",
    # More fixed income
    "BND",
    "IEF",
    "EMB",
]

# %% [markdown]
# ### Load Daily Prices from Canonical Data

# %%
# Load the fixed teaching universe from canonical ETF data.
print(f"Loading {len(SYMBOLS)} ETFs from canonical data...")
etf_data = load_etfs(symbols=SYMBOLS, start_date=START, end_date=END)

# Pivot to a wide Polars panel and fill only from earlier observations.
prices = (
    etf_data.select(["timestamp", "symbol", "close"])
    .pivot(on="symbol", index="timestamp", values="close")
    .sort("timestamp")
    .fill_null(strategy="forward")
    .drop_nulls()
)
print(f"Price data: {prices.height} days, {prices.width - 1} ETFs")
selected_symbols = [c for c in prices.columns if c != "timestamp"]
print(f"Selected ETFs: {selected_symbols[:5]}...")

# %% [markdown]
# The fixed universe intentionally mixes equities, duration, credit, and real assets.
# It is a controlled teaching panel assembled with current-vintage data, not a
# point-in-time reconstruction of the historical ETF opportunity set.

# %% [markdown]
# ### The hurdle rate
#
# The maximum-Sharpe portfolio is the one whose excess return per unit of risk is highest, so it
# needs a rate to measure excess against. The value below is a scenario assumption rather than an
# estimate of what cash actually paid over this sample, and it is a parameter so a reader can move
# it and see what happens.
#
# It only affects one of the two solutions. The minimum-variance portfolio is defined without
# reference to expected returns at all, so it does not move when the hurdle does - which is the
# first hint of why it behaves so differently out of sample.

# %%
print(f"Scenario hurdle: {RISK_FREE_RATE:.1%} a year")

# %% [markdown]
# The risk-free rate affects tangency-portfolio rankings more than minimum-variance
# allocations. When short rates move materially, max-Sharpe portfolios can change
# even if the covariance matrix stays the same.

# %% [markdown]
# ## Compute Returns & Covariance

# %%
# Compute daily returns using Polars
daily_returns = prices.select(
    pl.col("timestamp"), *[pl.col(c).pct_change().alias(c) for c in selected_symbols]
).drop_nulls()

print(f"Daily returns: {daily_returns.shape[0]} observations")
daily_returns.head()

# %%
# Full-sample returns are used only for descriptive geometry below.
geometry_returns_matrix = daily_returns.select(selected_symbols).to_numpy()
periods_per_year = 252

# %% [markdown]
# ### The two inputs, and how unequally they are estimated
#
# The optimizer needs an expected return per asset and a covariance between every pair. The
# asymmetry between how well those two can be estimated is the whole subject of this notebook.
#
# The expected return below is a compound growth rate computed from two prices: the first and the
# last. Every observation in between affects it only through where it leaves the endpoints. That is
# a deliberately spare estimator and it is not unusual - a longer-horizon mean is not obviously
# better - but it means the return vector rests on two numbers per asset.
#
# The covariance rests on every daily observation of every pair, thousands of them. It is estimated
# far more precisely, and it is also the input the optimizer is *less* sensitive to. Both facts
# point the same way: what comes out of a mean-variance optimizer is dominated by the input that is
# known worst.


# %%
def annualize_returns_from_prices(prices_df: pl.DataFrame, symbols: list[str]) -> np.ndarray:
    """Compute CAGR from elapsed calendar time between endpoint prices."""
    first_prices = prices_df.select(symbols).row(0)
    last_prices = prices_df.select(symbols).row(-1)
    first_timestamp = prices_df["timestamp"][0]
    last_timestamp = prices_df["timestamp"][-1]
    elapsed_years = (last_timestamp - first_timestamp).days / 365.2425
    if elapsed_years <= 0:
        raise ValueError("CAGR requires a positive elapsed time")

    return np.array(
        [
            (last / first) ** (1 / elapsed_years) - 1
            for first, last in zip(first_prices, last_prices, strict=True)
        ]
    )


geometry_returns = annualize_returns_from_prices(prices, selected_symbols)
print(f"Annualized returns range: [{geometry_returns.min():.2%}, {geometry_returns.max():.2%}]")

# %%
# Covariance matrix (annualized)
geometry_daily_cov = np.cov(geometry_returns_matrix.T)
geometry_cov = geometry_daily_cov * periods_per_year

print(f"Covariance matrix shape: {geometry_cov.shape}")

# %% [markdown]
# ### Correlation Analysis with Interactive Heatmap

# %%
# Compute correlation matrix
corr_matrix = np.corrcoef(geometry_returns_matrix.T)

# Interactive correlation heatmap
fig = px.imshow(
    corr_matrix,
    x=selected_symbols,
    y=selected_symbols,
    color_continuous_scale=ml4t_diverging(),
    zmin=-1,
    zmax=1,
    title="Diversification comes from distinct equity and duration blocks",
)
fig.update_layout(height=600, width=700)
fig.show()

# %%
# Correlation distribution (lower triangle)
lower_tri = corr_matrix[np.tril_indices(len(corr_matrix), -1)]

fig = px.histogram(
    x=lower_tri,
    nbins=30,
    color_discrete_sequence=[COLORS["blue"]],
    title="Pairwise correlations span hedges to near-lockstep exposures",
    labels={"x": "Correlation", "y": "Count"},
)
fig.add_vline(
    x=lower_tri.mean(),
    line_dash="dash",
    line_color=COLORS["amber"],
    annotation_text=f"Mean: {lower_tri.mean():.2f}",
    annotation_position="top right",
)
fig.update_layout(showlegend=False)
fig.show()

print(f"Correlation stats: Mean={lower_tri.mean():.3f}, Std={lower_tri.std():.3f}")

# %% [markdown]
# **Interpretation**: The average pairwise correlation is well below one, which is
# the raw material that makes diversification possible. But the dispersion matters
# too: if correlations spike together in stress periods, MVO can overstate how much
# protection the cross-section really provides.

# %% [markdown]
# ### How much a small change in the inputs moves the answer
#
# Solving for optimal weights involves inverting the covariance matrix, and inversion amplifies
# error. The condition number says by how much: it is the ratio of the largest to the smallest
# eigenvalue, and it bounds the factor by which a relative error in the input can grow in the
# output.
#
# $$\kappa(\Sigma) = \frac{\sigma_{\text{max}}}{\sigma_{\text{min}}}$$
#
# A value near one means the matrix is nearly a scaled identity and inversion is harmless. A large
# value means some direction in asset space has almost no variance in the sample, the inverse
# divides by that near-zero number, and the optimizer takes enormous positions along it.
#
# What makes the number large is not the assets being risky, it is them being *redundant*: two
# funds that move together leave a direction with almost no independent variation, and a nineteen-
# asset universe of ETFs has many such pairs. The eigenvalue spectrum below is where to see it.

# %%
condition_number = np.linalg.cond(geometry_cov)
print(f"Condition number: {condition_number:.1f}")

if condition_number < 100:
    print("-> Well-conditioned matrix (stable optimization)")
elif condition_number < 1000:
    print("-> Moderately conditioned (acceptable for optimization)")
else:
    print("-> Ill-conditioned (consider regularization)")

# %%
# The covariance eigenvalue spectrum shows why inversion would be unstable.
cov_eigenvalues = np.linalg.eigvalsh(geometry_cov)
eigenvalue_rank = np.arange(1, len(cov_eigenvalues) + 1)

fig = px.line(
    x=eigenvalue_rank,
    y=np.sort(cov_eigenvalues),
    markers=True,
    log_y=True,
    labels={"x": "Ordered Eigenvalue", "y": "Annualized Covariance Eigenvalue"},
    color_discrete_sequence=[COLORS["blue"]],
    title="A wide eigenvalue spread makes covariance inversion fragile",
)
fig.update_layout(height=450)
fig.show()

# %% [markdown]
# The spread between the largest and smallest eigenvalue on a log axis is the condition number,
# read off a chart. The last few points are the directions in which this universe barely moves
# independently, and they are the ones the inverse divides by.
#
# There is a fix and it is not in this notebook. Shrinking the sample covariance towards a simpler
# target lifts the small eigenvalues and pulls the condition number down, at the cost of biasing
# the estimate towards something the data did not say. `03_robust_optimization` does that and
# measures what it buys.

# %% [markdown]
# ## Simulate Random Portfolios
#
# Generate random portfolio weights using the Dirichlet distribution to visualize the feasible region (the "Markowitz Bullet").


# %%
def simulate_portfolios(
    returns: np.ndarray,
    cov: np.ndarray,
    n_portfolios: int,
    rf_rate: float = 0.0,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Simulate random long-only portfolios using a Dirichlet distribution."""
    n_assets = len(returns)

    # Generate weights (small alpha = concentrated, large alpha = uniform)
    alpha = np.full(n_assets, 0.05)
    weights = dirichlet(alpha=alpha, size=n_portfolios)

    # Portfolio metrics
    pf_returns = weights @ returns
    pf_std = np.sqrt((weights @ cov * weights).sum(axis=1))
    pf_sharpe = (pf_returns - rf_rate) / pf_std

    results = pl.DataFrame(
        {
            "std": pf_std,
            "return": pf_returns,
            "sharpe": pf_sharpe,
        }
    )

    return results, weights


# %%
simulated, sim_weights = simulate_portfolios(
    geometry_returns,
    geometry_cov,
    N_PORTFOLIOS,
    rf_rate=RISK_FREE_RATE,
)

simulated.describe()

# %% [markdown]
# ### Interactive Markowitz Bullet

# %%
# Convert to pandas for plotly express
sim_pd = simulated.to_pandas()

fig = px.scatter(
    sim_pd.sample(n=min(N_PORTFOLIOS, len(sim_pd)), random_state=SEED),
    x="std",
    y="return",
    color="sharpe",
    color_continuous_scale=ML4T_SEQUENTIAL,
    opacity=0.5,
    title="Random portfolios reveal a narrow high-return frontier",
    labels={
        "std": "Annualized Volatility",
        "return": "Annualized Return",
        "sharpe": "Sharpe Ratio",
    },
)

# %%
# Mark best portfolios from simulation
max_sharpe_idx = simulated["sharpe"].arg_max()
min_vol_idx = simulated["std"].arg_min()

max_sr = simulated.row(max_sharpe_idx)
min_vol = simulated.row(min_vol_idx)

_ = fig.add_scatter(
    x=[max_sr[0]],
    y=[max_sr[1]],
    mode="markers",
    marker=dict(size=15, color=COLORS["amber"], symbol="star"),
    name=f"Max Sharpe (SR={max_sr[2]:.2f})",
)
fig.add_scatter(
    x=[min_vol[0]],
    y=[min_vol[1]],
    mode="markers",
    marker=dict(size=12, color=COLORS["positive"], symbol="circle"),
    name=f"Min Vol (Vol={min_vol[0]:.2%})",
)

fig.update_layout(
    xaxis_tickformat=".0%",
    yaxis_tickformat=".0%",
    height=600,
    margin=dict(r=130),
    legend=dict(
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=0.02,
        bgcolor=COLORS["bg_light"],
    ),
)
fig.data[0].marker.colorbar.update(title="Sharpe", x=1.04)
fig.show()

print(f"Simulated Max Sharpe: Return={max_sr[1]:.2%}, Vol={max_sr[0]:.2%}, SR={max_sr[2]:.2f}")
print(f"Simulated Min Vol:    Return={min_vol[1]:.2%}, Vol={min_vol[0]:.2%}, SR={min_vol[2]:.2f}")

# %% [markdown]
# **Interpretation**: The random cloud is useful as a sanity check. If the optimizer
# lands only marginally better than a broad sample of random portfolios, the extra
# precision of MVO is not buying much practical advantage.

# %% [markdown]
# ## Portfolio Optimization
#
# Now we solve for optimal portfolios using scipy.optimize.


# %%
def portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    """Portfolio expected return."""
    return weights @ returns


# %% [markdown]
# #### Portfolio Volatility Function


# %%
def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio volatility (standard deviation)."""
    return np.sqrt(weights @ cov @ weights)


# %% [markdown]
# #### Sharpe Ratio Function


# %%
def portfolio_sharpe(weights: np.ndarray, returns: np.ndarray, cov: np.ndarray, rf: float) -> float:
    """Portfolio Sharpe ratio."""
    ret = portfolio_return(weights, returns)
    vol = portfolio_volatility(weights, cov)
    return (ret - rf) / vol


# %% [markdown]
# #### Optimization Objective (Negative Sharpe)


# %%
def neg_sharpe(weights: np.ndarray, returns: np.ndarray, cov: np.ndarray, rf: float) -> float:
    """Negative Sharpe (for minimization)."""
    return -portfolio_sharpe(weights, returns, cov, rf)


# %%
# Asset count for the full-sample portfolio displays below.
n_assets = len(selected_symbols)


def validate_optimization_result(
    result,
    *,
    label: str,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    target_return: float | None = None,
    expected_returns: np.ndarray | None = None,
    erc_cov: np.ndarray | None = None,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Reject invalid solver output before weights reach downstream analysis."""
    if not result.success:
        raise ValueError(f"{label} optimization failed: {result.message}")

    weights = np.asarray(result.x, dtype=float)
    if not np.isfinite(weights).all() or not np.isfinite(result.fun):
        raise ValueError(f"{label} optimization returned non-finite output")

    budget_residual = abs(weights.sum() - 1.0)
    lower_violation = max(0.0, lower_bound - float(weights.min()))
    upper_violation = max(0.0, float(weights.max()) - upper_bound)
    if max(budget_residual, lower_violation, upper_violation) > tolerance:
        raise ValueError(
            f"{label} constraint residuals exceed tolerance: "
            f"budget={budget_residual:.2e}, lower={lower_violation:.2e}, "
            f"upper={upper_violation:.2e}"
        )

    if target_return is not None:
        if expected_returns is None:
            raise ValueError("expected_returns is required for target-return validation")
        return_residual = abs(portfolio_return(weights, expected_returns) - target_return)
        if return_residual > tolerance:
            raise ValueError(
                f"{label} target-return residual {return_residual:.2e} exceeds tolerance"
            )

    if erc_cov is not None:
        portfolio_variance = weights @ erc_cov @ weights
        risk_contributions = weights * (erc_cov @ weights) / portfolio_variance
        erc_residual = float(np.max(np.abs(risk_contributions - 1 / len(weights))))
        if erc_residual > 5e-3:
            raise ValueError(f"{label} risk-contribution residual {erc_residual:.2e} is too large")
        print(f"{label} max risk-contribution residual: {erc_residual:.2e}")

    return weights


# %% [markdown]
# ### Maximum Sharpe Ratio Portfolio


# %%
def optimize_max_sharpe(returns: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    """Find long-only portfolio that maximizes Sharpe ratio."""
    n_inputs = len(returns)
    if cov.shape != (n_inputs, n_inputs):
        raise ValueError("Covariance shape must match the expected-return vector")
    initial_weights = np.full(n_inputs, 1 / n_inputs)
    bounds = ((0, 1),) * n_inputs
    budget_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        neg_sharpe,
        x0=initial_weights,
        args=(returns, cov, rf),
        method="SLSQP",
        bounds=bounds,
        constraints=budget_constraint,
        options={"ftol": 1e-10, "maxiter": 10000},
    )
    return validate_optimization_result(result, label="Max Sharpe")


geometry_max_sharpe_weights = optimize_max_sharpe(geometry_returns, geometry_cov, RISK_FREE_RATE)

print(f"Max Sharpe Return: {portfolio_return(geometry_max_sharpe_weights, geometry_returns):.2%}")
print(f"Max Sharpe Vol:    {portfolio_volatility(geometry_max_sharpe_weights, geometry_cov):.2%}")
print(
    "Max Sharpe SR:     "
    f"{portfolio_sharpe(geometry_max_sharpe_weights, geometry_returns, geometry_cov, RISK_FREE_RATE):.3f}"
)

# %% [markdown]
# ### Minimum Volatility Portfolio


# %%
def optimize_min_vol(returns: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Find long-only minimum volatility portfolio."""
    n_inputs = len(returns)
    if cov.shape != (n_inputs, n_inputs):
        raise ValueError("Covariance shape must match the expected-return vector")
    initial_weights = np.full(n_inputs, 1 / n_inputs)
    bounds = ((0, 1),) * n_inputs
    budget_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        portfolio_volatility,
        x0=initial_weights,
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=budget_constraint,
        options={"ftol": 1e-10, "maxiter": 10000},
    )
    return validate_optimization_result(result, label="Minimum Volatility")


geometry_min_vol_weights = optimize_min_vol(geometry_returns, geometry_cov)

print(f"Min Vol Return: {portfolio_return(geometry_min_vol_weights, geometry_returns):.2%}")
print(f"Min Vol Vol:    {portfolio_volatility(geometry_min_vol_weights, geometry_cov):.2%}")
print(
    "Min Vol SR:     "
    f"{portfolio_sharpe(geometry_min_vol_weights, geometry_returns, geometry_cov, RISK_FREE_RATE):.3f}"
)

# %% [markdown]
# ### Alternative Portfolio Strategies

# %%
# Equal Weight Portfolio
equal_weights = np.full(n_assets, 1 / n_assets)

# Inverse Volatility Portfolio
geometry_asset_vols = np.sqrt(np.diag(geometry_cov))
geometry_inv_vol_weights = (1 / geometry_asset_vols) / (1 / geometry_asset_vols).sum()

# %% [markdown]
# #### Equal Risk Contribution (ERC)
#
# The ERC portfolio equalizes each asset's percentage contribution to total
# portfolio variance. The (percentage) variance contribution for asset $i$ is
# $$RC_i = \frac{w_i\,(\Sigma w)_i}{w^\top \Sigma w}.$$
# These contributions sum to one across assets; we minimize the squared
# deviations of the unnormalized contributions $w_i (\Sigma w)_i$ from the
# target $\sigma_P^2 / N$, which is the same as targeting $RC_i = 1/N$.


# %%
def equal_risk_contribution(cov: np.ndarray) -> np.ndarray:
    """Compute Equal Risk Contribution weights via optimization.

    Solves for weights where each asset contributes equally to total
    portfolio variance: w_i * (Sigma @ w)_i = sigma_P^2 / N for all i.
    """
    n = len(cov)

    def risk_contrib_objective(w):
        port_var = w @ cov @ w
        marginal_contrib = cov @ w
        risk_contrib_pct = w * marginal_contrib / port_var
        return np.sum((risk_contrib_pct - 1 / n) ** 2)

    w0 = np.ones(n) / n
    result = minimize(
        risk_contrib_objective,
        w0,
        method="SLSQP",
        bounds=[(1e-6, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-12, "maxiter": 10000},
    )
    return validate_optimization_result(
        result,
        label="Equal Risk Contribution",
        lower_bound=1e-6,
        erc_cov=cov,
    )


geometry_erc_weights = equal_risk_contribution(geometry_cov)

# %%
# Store all portfolio strategies
geometry_portfolios = {
    "Max Sharpe": geometry_max_sharpe_weights,
    "Min Volatility": geometry_min_vol_weights,
    "Equal Weight": equal_weights,
    "Inverse Vol": geometry_inv_vol_weights,
    "ERC": geometry_erc_weights,
}

# %%
# Compute metrics for all portfolios
portfolio_metrics = []

for name, weights in geometry_portfolios.items():
    ret = portfolio_return(weights, geometry_returns)
    vol = portfolio_volatility(weights, geometry_cov)
    sr = portfolio_sharpe(weights, geometry_returns, geometry_cov, RISK_FREE_RATE)
    n_positions = (np.abs(weights) > 0.001).sum()

    portfolio_metrics.append(
        {
            "Portfolio": name,
            "Return": ret,
            "Volatility": vol,
            "Sharpe": sr,
            "Positions": n_positions,
        }
    )

metrics_df = pl.DataFrame(portfolio_metrics)
metrics_df

# %% [markdown]
# **Interpretation**: This table gives a first read on the return/risk trade-off across
# strategies. Notice that Min Volatility can have a negative Sharpe when the optimized
# return falls below the risk-free rate -- the optimizer minimizes variance without
# regard to the hurdle rate. Equal Risk Contribution (ERC) produces weights that differ
# from Inverse Vol because it accounts for cross-asset correlations, not just standalone
# volatility.

# %% [markdown]
# ## Efficient Frontier


# %%
def compute_efficient_frontier(
    returns: np.ndarray,
    cov: np.ndarray,
    min_vol_weights: np.ndarray,
    n_points: int = 50,
) -> pl.DataFrame:
    """Compute the long-only upper frontier from global minimum variance upward."""
    n_inputs = len(returns)
    if cov.shape != (n_inputs, n_inputs):
        raise ValueError("Covariance shape must match the expected-return vector")
    if len(min_vol_weights) != n_inputs:
        raise ValueError("Minimum-volatility weights must match the expected-return vector")
    initial_weights = np.full(n_inputs, 1 / n_inputs)
    ret_min = portfolio_return(min_vol_weights, returns)
    ret_max = float(returns.max())
    target_returns = np.linspace(ret_min, ret_max, n_points)

    bounds = ((0, 1),) * n_inputs
    frontier_points = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, returns) - t},
        ]

        result = minimize(
            portfolio_volatility,
            x0=initial_weights,
            args=(cov,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 10000},
        )

        try:
            weights = validate_optimization_result(
                result,
                label=f"Frontier target {target:.4f}",
                target_return=target,
                expected_returns=returns,
            )
        except ValueError as exc:
            print(f"Rejected frontier point: {exc}")
        else:
            frontier_points.append(
                {
                    "target_return": target,
                    "volatility": portfolio_volatility(weights, cov),
                }
            )

    return pl.DataFrame(frontier_points)


# %%
frontier = compute_efficient_frontier(
    geometry_returns,
    geometry_cov,
    geometry_min_vol_weights,
    n_points=40,
)
frontier.head()

# %% [markdown]
# ### Plot: Efficient Frontier with All Portfolios
#
# We build the figure incrementally so each visual layer is easy to reason
# about: first the random-portfolio cloud, then the efficient frontier curve,
# then the named portfolio markers, and finally the layout.

# %% [markdown]
# #### Layer A: Simulated-Portfolio Cloud
#
# A subsample of the Dirichlet-generated portfolios provides a visual sanity
# check on the feasible region (the "Markowitz bullet").

# %%
fig = go.Figure()

# Simulated portfolios (sample for performance)
sample_idx = np.random.choice(len(simulated), min(5000, len(simulated)), replace=False)
_ = fig.add_scatter(
    x=simulated["std"].to_numpy()[sample_idx],
    y=simulated["return"].to_numpy()[sample_idx],
    mode="markers",
    marker=dict(
        size=4,
        color=simulated["sharpe"].to_numpy()[sample_idx],
        colorscale=ML4T_SEQUENTIAL,
        opacity=0.3,
        colorbar=dict(title="Sharpe", x=1.04),
    ),
    name="Simulated Portfolios",
)

# %% [markdown]
# #### Layer B: Efficient Frontier Line
#
# Overlay the optimized frontier, the upper boundary of the feasible region.

# %%
# Efficient frontier
_ = fig.add_scatter(
    x=frontier["volatility"].to_list(),
    y=frontier["target_return"].to_list(),
    mode="lines",
    line=dict(color=COLORS["neutral"], width=2, dash="dash"),
    name="Efficient Frontier",
)

# %% [markdown]
# #### Layer C: Named Portfolio Markers
#
# Add the headline portfolios (Max Sharpe, Min Volatility, Equal Weight,
# Inverse Vol, ERC) so we can locate them relative to the frontier.

# %%
# Portfolio markers
markers = {
    "Max Sharpe": ("star", COLORS["amber"], 18),
    "Min Volatility": ("circle", COLORS["positive"], 14),
    "Equal Weight": ("pentagon", COLORS["slate"], 12),
    "Inverse Vol": ("x", COLORS["negative"], 12),
    "ERC": ("triangle-down", COLORS["copper"], 12),
}

for name, weights in geometry_portfolios.items():
    ret = portfolio_return(weights, geometry_returns)
    vol = portfolio_volatility(weights, geometry_cov)
    symbol, color, size = markers[name]

    fig.add_scatter(
        x=[vol],
        y=[ret],
        mode="markers",
        marker=dict(
            symbol=symbol,
            color=color,
            size=size,
            line=dict(width=1, color=COLORS["blue"]),
        ),
        name=name,
    )

# %% [markdown]
# #### Layer D: Layout and Render

# %%
fig.update_layout(
    title="In-sample optimization concentrates at the frontier extremes",
    xaxis_title="Annualized Volatility",
    yaxis_title="Annualized Return",
    xaxis_tickformat=".0%",
    yaxis_tickformat=".0%",
    height=600,
    margin=dict(r=130),
    legend=dict(
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=0.02,
        bgcolor=COLORS["bg_light"],
    ),
)
fig.show()

# %% [markdown]
# **Interpretation**: The efficient frontier traces the upper boundary of the feasible region.
# Portfolios on this curve deliver the highest return for each level of volatility. Several
# patterns are worth noting:
#
# - The **Max Sharpe** portfolio sits on the steepest ray from the risk-free rate to the frontier,
#   but it concentrates heavily in a handful of assets, making it sensitive to estimation error.
# - The **Min Volatility** portfolio clusters at the left tip of the bullet and sacrifices
#   expected return for lower in-sample variance.
# - **Equal Weight**, **Inverse Vol**, and **ERC** fall inside the frontier because this panel
#   defines in-sample geometry. The test section below evaluates frozen train-period weights.
#
# This concentration is the mechanism behind the **Markowitz Curse**: an optimizer can amplify
# small estimation differences into large weight differences. Section 17.5 discusses shrinkage
# and constraint-based remedies.

# %% [markdown]
# ### Portfolio Weights Comparison

# %%
# Create weights DataFrame
weights_df = pl.DataFrame({"symbol": selected_symbols, **geometry_portfolios})

# Melt for plotting
weights_long = weights_df.unpivot(index="symbol", variable_name="Portfolio", value_name="Weight")

fig = px.bar(
    weights_long.to_pandas(),
    x="symbol",
    y="Weight",
    color="Portfolio",
    barmode="group",
    color_discrete_sequence=ML4T_CATEGORICAL,
    title="Expected-return optimization concentrates in two ETFs",
)

# Add equal weight reference line
fig.add_hline(
    y=1 / n_assets,
    line_dash="dash",
    line_color=COLORS["neutral"],
    annotation_text=f"Equal Weight: {1 / n_assets:.1%}",
)

fig.update_layout(
    yaxis_tickformat=".0%",
    height=500,
    width=900,
    xaxis_tickangle=-45,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig.show()

# %%
# Number of significant positions (>0.1%)
for name, weights in geometry_portfolios.items():
    n_pos = (np.abs(weights) > 0.001).sum()
    top_weight = np.max(weights)
    print(f"{name:15s}: {n_pos:2d} positions, max weight: {top_weight:.1%}")

# %% [markdown]
# **Interpretation**: This is where the Markowitz curse becomes concrete. The max-Sharpe
# solution often wins in-sample by concentrating hard in a few names, while inverse-vol
# and equal-weight keep diversification by construction.

# %%
geometry_max_order = np.argsort(geometry_max_sharpe_weights)[::-1]
print("Largest holdings in the in-sample max-Sharpe solution:")
for rank in geometry_max_order[:3]:
    print(f"  {selected_symbols[rank]:<5} {geometry_max_sharpe_weights[rank]:.1%}")
print(
    f"Assets it holds at all: {int((geometry_max_sharpe_weights > 1e-4).sum())}"
    f" of {len(selected_symbols)}"
)
geometry_min_top = int(np.argmax(geometry_min_vol_weights))
print("Largest holding in the minimum-volatility solution:")
print(f"  {selected_symbols[geometry_min_top]:<5} {geometry_min_vol_weights[geometry_min_top]:.1%}")
print(
    f"Assets it holds at all: {int((geometry_min_vol_weights > 1e-4).sum())}"
    f" of {len(selected_symbols)}"
)

# %% [markdown]
# ## Train/Test Portfolio Evaluation
#
# The preceding frontier is explicitly in-sample geometry. For performance evaluation,
# every estimated input and portfolio weight uses observations through `TRAIN_END` only.
# We freeze those weights and apply them to later returns without selecting an allocator
# or changing a parameter from the test result.

# %%
train_prices = prices.filter(pl.col("timestamp") <= pl.lit(TRAIN_END).str.to_datetime())
train_daily_returns = daily_returns.filter(
    pl.col("timestamp") <= pl.lit(TRAIN_END).str.to_datetime()
)
test_daily_returns = daily_returns.filter(pl.col("timestamp") > pl.lit(TRAIN_END).str.to_datetime())

if train_daily_returns.is_empty() or test_daily_returns.is_empty():
    raise ValueError("Both train and test windows must contain returns")
if train_daily_returns["timestamp"].max() >= test_daily_returns["timestamp"].min():
    raise ValueError("Train and test windows overlap")

train_returns_matrix = train_daily_returns.select(selected_symbols).to_numpy()
test_returns_matrix = test_daily_returns.select(selected_symbols).to_numpy()
test_dates = test_daily_returns["timestamp"].to_list()

train_returns = annualize_returns_from_prices(train_prices, selected_symbols)
train_cov = np.cov(train_returns_matrix.T) * periods_per_year

train_max_sharpe_weights = optimize_max_sharpe(train_returns, train_cov, RISK_FREE_RATE)
train_min_vol_weights = optimize_min_vol(train_returns, train_cov)
train_asset_vols = np.sqrt(np.diag(train_cov))
train_inv_vol_weights = (1 / train_asset_vols) / (1 / train_asset_vols).sum()
train_erc_weights = equal_risk_contribution(train_cov)

test_portfolios = {
    "Max Sharpe": train_max_sharpe_weights,
    "Min Volatility": train_min_vol_weights,
    "Equal Weight": equal_weights,
    "Inverse Vol": train_inv_vol_weights,
    "ERC": train_erc_weights,
}

print(
    f"Training window: {train_daily_returns['timestamp'].min()} to "
    f"{train_daily_returns['timestamp'].max()} ({train_daily_returns.height} returns)"
)
print(
    f"Test window: {test_daily_returns['timestamp'].min()} to "
    f"{test_daily_returns['timestamp'].max()} ({test_daily_returns.height} returns)"
)

# Daily constant-target rebalancing is the declared vectorized policy.
test_portfolio_returns = {
    name: test_returns_matrix @ weights for name, weights in test_portfolios.items()
}

# Create DataFrame of portfolio returns
pf_returns_df = pl.DataFrame(
    {
        "timestamp": test_dates,
        **test_portfolio_returns,
    }
)

pf_returns_df.head()

# %% [markdown]
# ### Cumulative Returns Comparison

# %%
# Compute cumulative returns
fig = go.Figure()

for i, name in enumerate(test_portfolios):
    cum_ret = (1 + np.array(test_portfolio_returns[name])).cumprod()
    fig.add_scatter(
        x=test_dates,
        y=cum_ret,
        mode="lines",
        name=name,
        line=dict(color=ML4T_CATEGORICAL[i % len(ML4T_CATEGORICAL)]),
    )

fig.update_layout(
    title=(f"Frozen max-Sharpe weights lead the {int(TRAIN_END[:4]) + 1}-{END[:4]} test"),
    xaxis_title="Date",
    yaxis_title="Cumulative Return (Growth of $1)",
    height=500,
    legend=dict(
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=0.02,
        bgcolor=COLORS["bg_light"],
    ),
)
fig.show()

# %% [markdown]
# ### Portfolio Analysis with ml4t-diagnostic

# %%
# Analyze Max Sharpe portfolio in detail
max_sharpe_test_returns = pl.Series("returns", test_portfolio_returns["Max Sharpe"])

analysis = PortfolioAnalysis(
    returns=max_sharpe_test_returns,
    dates=pl.Series("timestamp", test_dates),
    risk_free=RISK_FREE_RATE,
    periods_per_year=252,
)

# Get comprehensive metrics
metrics = analysis.compute_summary_stats()
print(metrics.summary())

# %% [markdown]
# ### Factor Decomposition with `ml4t-diagnostic`
#
# `PortfolioAnalysis` reports the standard summary statistics for this
# test portfolio. `FactorAnalysis` from the same library answers a different
# question: "**what is driving** this portfolio's returns?". We regress the
# Max-Sharpe daily return series on the Fama-French 5-factor model (Mkt-RF,
# SMB, HML, RMW, CMA) using HAC standard errors. The intercept is the
# factor-adjusted alpha. The loadings describe test-period co-movement with
# market, size, value, profitability, and investment factors. They do not
# establish that those exposures persist beyond this test window.

# %%
factor_load_log = io.StringIO()
with contextlib.redirect_stdout(factor_load_log), contextlib.redirect_stderr(factor_load_log):
    ff_data = load_fama_french_5factor(frequency="daily")
print("Loaded daily Fama-French five-factor data from the configured provider.")

if ff_data.rf_rate is None:
    raise ValueError("Fama-French data must include the aligned risk-free series")

ff_returns_with_rf = ff_data.returns.with_columns(ff_data.rf_rate.rename("RF"))
test_mvo_returns = pl.DataFrame(
    {
        "timestamp": pl.Series(test_dates),
        "ret": np.asarray(test_portfolio_returns["Max Sharpe"]),
    }
)
joined = test_mvo_returns.with_columns(pl.col("timestamp").cast(pl.Date)).join(
    ff_returns_with_rf.with_columns(pl.col("timestamp").cast(pl.Date)),
    on="timestamp",
    how="inner",
)
print(
    f"Aligned {joined.height} daily observations with FF5 factors "
    f"({joined['timestamp'].min()} to {joined['timestamp'].max()})"
)

aligned_factor_data = ff_data.__class__(
    returns=joined.select(["timestamp"] + ff_data.factor_names),
    rf_rate=joined["RF"],
    factor_names=ff_data.factor_names,
    source="fama_french",
    frequency="daily",
)
fa = FactorAnalysis(
    returns=joined["ret"].to_numpy(),
    factor_data=aligned_factor_data,
    periods_per_year=252,
)
factor_model = fa.static_model(hac=True)
print(factor_model.summary())

# %% [markdown]
# Read the table top-down: the **alpha** row is the residual return after
# removing factor exposures (per day, with a HAC t-statistic). The
# **factor loadings** report what fraction of each factor's return the
# portfolio captures during the test period. A high adjusted $R^2$ means the
# observed returns co-moved with the selected factors over this window.

# %%
# Compare all portfolio strategies using PortfolioAnalysis
comparison_metrics = []

for name in test_portfolios:
    returns_series = pl.Series("returns", test_portfolio_returns[name])

    pa = PortfolioAnalysis(
        returns=returns_series,
        dates=pl.Series("timestamp", test_dates),
        risk_free=RISK_FREE_RATE,
        periods_per_year=252,
    )

    metrics = pa.compute_summary_stats()

    comparison_metrics.append(
        {
            "Portfolio": name,
            "Annual Return": metrics.annual_return,
            "Annual Vol": metrics.annual_volatility,
            "Sharpe": metrics.sharpe_ratio,
            "Sortino": metrics.sortino_ratio,
            "Calmar": metrics.calmar_ratio,
            "Max DD": metrics.max_drawdown,
            "Win Rate": metrics.win_rate,
            "VaR 95%": metrics.var_95,
        }
    )

comparison_df = pl.DataFrame(comparison_metrics)
comparison_df

# %% [markdown]
# **Interpretation**: Several patterns stand out:
#
# - **Min Volatility** minimizes train-period variance without considering whether its
#   expected or test return clears the scenario hurdle.
# - **Max Sharpe** targets the steepest excess-return-per-unit-risk, but concentrates
#   heavily in training, making its frozen test result sensitive to estimation error.
# - **ERC** and **Inverse Vol** offer middle-ground diversification. Because ERC
#   equalizes risk *contributions* (accounting for correlations) rather than simply
#   inverting standalone volatility, the two strategies produce different weights and metrics.

# %% [markdown]
# ### Execution-Aware Bridge with ml4t-backtest
#
# The vectorized path rebalances to the frozen max-Sharpe target every day at
# close-to-close returns. The two engine paths use the same daily target schedule with
# next-bar execution and actual test-period OHLCV. A zero-cost engine isolates timing
# and engine mechanics; the cost-aware engine then adds commission and slippage.

# %% [markdown]
# Submit the same frozen target weights on every test bar.


# %%
class DailyTargetWeightStrategy(Strategy):
    def __init__(self, target_weights: dict[str, float], allow_short: bool):
        self.target_weights = target_weights
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(
                min_trade_value=0.0,
                min_weight_change=0.0,
                allow_fractional=True,
                allow_short=allow_short,
            )
        )

    def on_data(self, timestamp, data, context, broker):
        targets = {asset: weight for asset, weight in self.target_weights.items() if asset in data}
        if targets:
            self.executor.execute(targets, data, broker)


# %%
# Build engine inputs from train-only weights and actual test-period OHLCV.
engine_target_weights = {
    asset: float(weight)
    for asset, weight in zip(selected_symbols, test_portfolios["Max Sharpe"], strict=False)
    if abs(float(weight)) > 1e-8
}
allow_short_engine = any(weight < 0 for weight in engine_target_weights.values())

test_prices_long = (
    etf_data.filter(pl.col("timestamp") > pl.lit(TRAIN_END).str.to_date())
    .select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    .sort(["timestamp", "symbol"])
)


# %%
def run_daily_target_engine(*, cost_aware: bool, return_column: str) -> pl.DataFrame:
    """Run daily target rebalancing with or without explicit trading costs."""
    engine = Engine(
        feed=DataFeed(prices_df=test_prices_long),
        strategy=DailyTargetWeightStrategy(
            engine_target_weights,
            allow_short=allow_short_engine,
        ),
        config=BacktestConfig(
            initial_cash=100_000.0,
            execution_mode=ExecutionMode.NEXT_BAR,
            commission_type=(CommissionType.PERCENTAGE if cost_aware else CommissionType.NONE),
            commission_rate=COMMISSION_RATE if cost_aware else 0.0,
            slippage_type=SlippageType.PERCENTAGE if cost_aware else SlippageType.NONE,
            slippage_rate=SLIPPAGE_RATE if cost_aware else 0.0,
            allow_short_selling=allow_short_engine,
        ),
    )
    return (
        engine.run()
        .to_daily_pnl()
        .select(
            pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
            pl.col("return_pct").alias(return_column),
        )
    )


zero_cost_daily = run_daily_target_engine(
    cost_aware=False,
    return_column="zero_cost_return",
)
cost_aware_daily = run_daily_target_engine(
    cost_aware=True,
    return_column="cost_aware_return",
)

vectorized_daily = pl.DataFrame(
    {
        "timestamp": pl.Series(test_dates).cast(pl.Datetime("us")),
        "vectorized_return": np.asarray(test_portfolio_returns["Max Sharpe"]),
    }
)

# %%
# Compare the three paths only on their common test dates.
bridge = (
    vectorized_daily.join(zero_cost_daily, on="timestamp", how="inner")
    .join(cost_aware_daily, on="timestamp", how="inner")
    .drop_nulls(["vectorized_return", "zero_cost_return", "cost_aware_return"])
    .sort("timestamp")
)

bridge_stats = {}
for label, column in {
    "Vectorized": "vectorized_return",
    "Zero-cost engine": "zero_cost_return",
    "Cost-aware engine": "cost_aware_return",
}.items():
    portfolio_analysis = PortfolioAnalysis(
        returns=bridge[column],
        dates=bridge["timestamp"],
        risk_free=RISK_FREE_RATE,
        periods_per_year=252,
    )
    bridge_stats[label] = portfolio_analysis.compute_summary_stats()

# %%
print("Execution bridge (train-only Max Sharpe, daily target rebalancing):")
for label, stats in bridge_stats.items():
    print(
        f"  {label}: Sharpe={stats.sharpe_ratio:.3f}, "
        f"MaxDD={stats.max_drawdown:.2%}, Annual Return={stats.annual_return:.2%}"
    )

# %%
bridge_table = pl.DataFrame(
    [
        {
            "Mode": label,
            "Annual Return": stats.annual_return,
            "Sharpe": stats.sharpe_ratio,
            "Max Drawdown": stats.max_drawdown,
        }
        for label, stats in bridge_stats.items()
    ]
)
bridge_table

# %% [markdown]
# Three rows, two comparisons. The gap between the vectorized result and the zero-cost engine is
# timing and engine mechanics alone: same weights, same prices, no fees on either side. The gap
# between the zero-cost engine and the cost-aware one is the commission and slippage the settings
# block declared, and nothing else. Reading the vectorized figure against the cost-aware one mixes
# the two and attributes the whole difference to costs.

# %%
fig = go.Figure()
bridge_colors = {
    "Vectorized": COLORS["blue"],
    "Zero-cost engine": COLORS["amber"],
    "Cost-aware engine": COLORS["copper"],
}
for label, column in {
    "Vectorized": "vectorized_return",
    "Zero-cost engine": "zero_cost_return",
    "Cost-aware engine": "cost_aware_return",
}.items():
    fig.add_scatter(
        x=bridge["timestamp"],
        y=(1 + bridge[column]).cum_prod(),
        mode="lines",
        name=label,
        line=dict(color=bridge_colors[label]),
    )
fig.update_layout(
    title="Timing explains more of the bridge than trading costs",
    xaxis_title="Date",
    yaxis_title="Growth of $1",
    height=420,
    legend=dict(
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=0.02,
        bgcolor=COLORS["bg_light"],
    ),
)
fig.show()

# %% [markdown]
# ### Drawdown Analysis

# %%
# Drawdown for Max Sharpe portfolio
cum_returns = (1 + np.array(test_portfolio_returns["Max Sharpe"])).cumprod()
running_max = np.maximum.accumulate(cum_returns)
drawdown = (cum_returns - running_max) / running_max

fig = go.Figure()

fig.add_scatter(
    x=test_dates,
    y=drawdown,
    mode="lines",
    fill="tozeroy",
    fillcolor="rgba(239, 68, 68, 0.25)",
    line=dict(color=COLORS["negative"], width=1),
    name="Drawdown",
)

# Mark maximum drawdown
max_dd_idx = np.argmin(drawdown)
fig.add_scatter(
    x=[test_dates[max_dd_idx]],
    y=[drawdown[max_dd_idx]],
    mode="markers",
    marker=dict(size=10, color=COLORS["blue"]),
    name=f"Max DD: {drawdown[max_dd_idx]:.1%}",
)

fig.update_layout(
    title="Losses cluster into a few deep and persistent falls",
    xaxis_title="Date",
    yaxis_title="Drawdown",
    yaxis_tickformat=".0%",
    height=400,
)
fig.show()

# %% [markdown]
# **Interpretation**: The underwater curve is the investor-experience view of MVO.
# A portfolio with a strong average Sharpe can still be hard to hold if losses cluster
# into a small number of deep and persistent drawdowns.

# %% [markdown]
# ### Portfolio Metrics Visualization

# %%
# Risk-return scatter colored by Sharpe ratio
metrics_for_plot = []
for name in test_portfolios:
    row = comparison_df.filter(pl.col("Portfolio") == name).row(0, named=True)
    metrics_for_plot.append(
        {
            "Portfolio": name,
            "Return": row["Annual Return"],
            "Volatility": row["Annual Vol"],
            "Sharpe": row["Sharpe"],
            "MaxDD": row["Max DD"],
        }
    )

plot_df = pl.DataFrame(metrics_for_plot).to_pandas()

fig = px.scatter(
    plot_df,
    x="Volatility",
    y="Return",
    color="Sharpe",
    text="Portfolio",
    size_max=30,
    color_continuous_scale=ml4t_diverging(),
    title="Higher test return comes with concentrated equity risk",
)

fig.update_traces(textposition="top center", marker=dict(size=16), cliponaxis=False)
fig.update_layout(
    xaxis_tickformat=".0%",
    yaxis_tickformat=".0%",
    height=500,
    margin=dict(l=90, r=100, t=90),
)
fig.show()

# %% [markdown]
# ## Key takeaways
#
# 1. **The frontier is a picture of the sample, not a menu of choices.** Every point on it is where
#    the optimizer would have put you *given* estimates drawn from the data it was shown. Drawn on
#    the full sample it is a description; the only way to learn anything from it about the future
#    is to estimate on one window and score on another, which is what the second half does.
# 2. **The optimizer is most sensitive to the input estimated worst.** Expected returns come from
#    two prices per asset; the covariance comes from thousands of observations per pair. The
#    solution moves far more with the first, which is the whole reason maximum-Sharpe weights
#    concentrate into a handful of names and rarely stay there.
# 3. **A high condition number is a statement about redundancy, not about risk.** It grows when
#    assets move together, leaving directions with almost no independent variation for the inverse
#    to divide by. Check it before trusting an optimizer's output, not after being surprised by it.
# 4. **Estimating less is a strategy.** Equal weight estimates nothing, inverse volatility
#    estimates variances only, equal risk contribution estimates the covariance but no returns.
#    Each gives up the ability to express a view in exchange for not having to hold one, and the
#    test window is where that trade is priced.
# 5. **Separate timing from cost when comparing an allocation to its backtest.** The gap between
#    vectorized arithmetic and a zero-cost engine is when trades happen; the gap between that and a
#    cost-aware engine is what they cost. Collapsing the two attributes fills to fees.
#
# ### Known limitations
#
# - One training window and one test window on one universe. A different split date would give
#   different weights and possibly a different ordering, and nothing here estimates how much.
# - The expected-return vector is an endpoint-to-endpoint growth rate. It is a defensible choice
#   and a very noisy one, and the notebook uses it partly because its noisiness is the subject.
# - Weights are frozen across the whole test window with daily rebalancing back to target. A real
#   allocation would re-estimate periodically, which introduces its own turnover and its own
#   sequence of estimation errors.
# - Long-only and fully invested throughout. Allowing shorts widens the frontier and makes the
#   concentration problem considerably worse.
#
# **Next:** [`03_robust_optimization`](03_robust_optimization.ipynb) attacks the estimation problem
# directly, with shrinkage and robust covariance estimators, and measures whether they help.
