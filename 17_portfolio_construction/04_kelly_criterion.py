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
# # How much to bet, and why the answer is unusable as given
#
# **Docker image**: `ml4t`
#
# ## Purpose
# The previous notebooks ask which assets to hold. This one asks how much of the account to put at
# risk, which is a separate question with a clean answer: the fraction that maximizes the expected
# logarithm of terminal wealth. Kelly derived it for a coin toss with known odds, and it extends to
# continuous returns and to portfolios.
#
# The derivation is exact and the answer, applied to estimated market inputs, is unusable as it
# comes out. Following it through from the coin toss to a multi-asset portfolio shows why, and the
# reason is not that the formula is wrong: it optimizes growth and has no term for ruin, no notion
# of a borrowing limit, and it scales directly with an expected-return estimate that nobody can
# make accurately.
#
# ## Learning objectives
#
# - Derive the optimal bet fraction for a binary wager, and show by simulation what betting more or
#   less than it does to the distribution of outcomes.
# - Extend the result to continuous returns, and say which approximation is being made and when it
#   holds.
# - Compute the multi-asset solution, read the leverage it implies, and work out what adverse move
#   would end the account at that leverage.
# - Recompute the same fraction on rolling windows and judge whether an estimate that unstable can
#   be acted on.
#
# ## Book reference
# Chapter 17, Section 17.4 (baseline allocators).
#
# ## Prerequisites
#
# - `02_mean_variance_optimization`, for covariance estimation and its instability.
# - Basic probability: expectation, variance, and the log of a product.

# %% [markdown]
# ## Imports & Settings

# %%
"""Derive and apply Kelly position sizing under uncertainty."""

from collections.abc import Callable

import numpy as np
import plotly.graph_objects as go
import polars as pl
import sympy

# Portfolio analysis
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from plotly.subplots import make_subplots
from scipy.optimize import minimize_scalar
from scipy.stats import binom
from sklearn.covariance import LedoitWolf
from sympy import diff, log, pprint, series, solve, symbols

from data import load_etfs
from utils.paths import get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, ml4t_palette

# %% tags=["parameters"]
# Production defaults; Papermill overrides for CI testing
SEED = 42
RISK_FREE_RATE = 0.02
TRAIN_END = "2019-12-31"

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## Part 1: Kelly Criterion for Binary Outcomes
#
# Kelly began by analyzing games with binary outcomes (like a coin toss) with constant
# win-loss probability. A win returns the invested capital plus a payoff defined by the odds.
#
# **Key variables:**
# - $W_n$: Wealth after n bets
# - $b$: Odds (amount won per unit staked)
# - $p$: Probability of winning
# - $f$: Fraction of current wealth to risk

# %% [markdown]
# ### Derivation of the Kelly Formula
#
# After $n$ trials with $W$ wins and $L = n - W$ losses:
#
# $$W_n = W_0 (1 + bf)^W (1 - f)^L$$
#
# The Kelly criterion maximizes the expected growth rate:
#
# $$g(f) = p \log(1 + bf) + (1-p) \log(1 - f)$$
#
# Taking the derivative and setting to zero yields the optimal fraction:
#
# $$f^* = \frac{bp - (1-p)}{b} = \frac{\text{edge}}{\text{odds}}$$

# %%
# Symbolic derivation using SymPy
f, b, p = symbols("f b p", positive=True, real=True)

# Expected growth rate
growth_rate = p * log(1 + b * f) + (1 - p) * log(1 - f)

print("Growth rate function:")
pprint(growth_rate)

# %%
# First derivative (first-order condition)
first_deriv = diff(growth_rate, f)
print("\nFirst derivative:")
pprint(first_deriv)

# %%
# Solve for optimal f
f_star = solve(first_deriv, f)
print("\nOptimal Kelly fraction:")
pprint(f_star[0])

# Simplified form: f* = (bp + p - 1) / b = (bp - q) / b where q = 1 - p
print("\nSimplified: f* = (edge) / (odds) = (bp - q) / b")

# %%
# Verify second-order condition (must be negative for maximum)
second_deriv = diff(first_deriv, f)
print("Second derivative:")
pprint(second_deriv)

# Evaluate at a specific point to verify it's negative
soc_value = second_deriv.subs([(b, 1), (p, 0.6), (f, 0.2)])
print(f"\nSOC at b=1, p=0.6, f=0.2: {float(soc_value):.4f} (negative = maximum)")

# %% [markdown]
# ### Kelly Fraction Examples


# %%
def compute_kelly_fraction(win_prob: float, odds: float = 1.0) -> float:
    """Compute the Kelly fraction for given win probability and odds."""
    edge = odds * win_prob - (1 - win_prob)
    return edge / odds if edge > 0 else 0.0


# %% [markdown]
# #### Growth Function for a Chosen Bet Fraction


# %%
def compute_growth_rate(win_prob: float, fraction: float, odds: float = 1.0) -> float:
    """Compute expected growth rate for given parameters."""
    if fraction <= 0 or fraction >= 1:
        return -np.inf
    return win_prob * np.log(1 + odds * fraction) + (1 - win_prob) * np.log(1 - fraction)


# Example calculations
examples = [
    (0.55, 1.0),  # Slight edge, even odds
    (0.60, 1.0),  # Good edge, even odds
    (0.55, 2.0),  # Slight edge, 2:1 odds
    (0.70, 0.5),  # Strong edge, unfavorable odds
]

print("Kelly Fraction Examples:")
for prob, odds in examples:
    kelly = compute_kelly_fraction(prob, odds)
    growth = compute_growth_rate(prob, kelly, odds)
    print(f"P(win)={prob:.0%}, Odds={odds}:1 -> Kelly={kelly:.1%}, Growth={growth:.4f}")

# %% [markdown]
# ### Growth Rate as Function of Bet Size

# %%
# Compute growth rates for different probabilities and bet fractions
fractions = np.linspace(0.01, 0.99, 100)
probabilities = [0.55, 0.60, 0.65, 0.70, 0.75]

fig = go.Figure()

colors = ml4t_palette(len(probabilities), categorical=True)

for i, prob in enumerate(probabilities):
    kelly = compute_kelly_fraction(prob)
    growth_rates = [compute_growth_rate(prob, f) for f in fractions]

    fig.add_scatter(
        x=fractions,
        y=growth_rates,
        mode="lines",
        name=f"P={prob:.0%}",
        line=dict(color=colors[i % len(colors)], width=2),
    )

    # Mark optimal Kelly fraction
    optimal_growth = compute_growth_rate(prob, kelly)
    fig.add_scatter(
        x=[kelly],
        y=[optimal_growth],
        mode="markers",
        marker=dict(size=10, color=colors[i % len(colors)], symbol="diamond"),
        showlegend=False,
        hovertemplate=f"Kelly={kelly:.1%}<br>Growth={optimal_growth:.4f}",
    )

# %%
fig.update_layout(
    title="A 55% win probability at even odds peaks at a 10% stake",
    xaxis_title="Bet Fraction (f)",
    yaxis_title="Expected Growth Rate",
    xaxis_tickformat=".0%",
    height=500,
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
)
fig.show()

# %% [markdown]
# ### Simulating Wealth Paths


# %%
def simulate_wealth_paths(
    outcomes: np.ndarray,
    fraction: float,
    odds: float = 1.0,
    initial_wealth: float = 100.0,
) -> np.ndarray:
    """Simulate wealth paths for Kelly betting."""
    # Compute growth factors
    growth_factors = np.where(
        outcomes,
        1 + odds * fraction,  # Win
        1 - fraction,  # Loss
    )

    # Cumulative wealth
    wealth = initial_wealth * np.cumprod(growth_factors, axis=0)
    return wealth


# %% [markdown]
# #### Percentile-Band Plot Helper


# %%
def add_percentile_bands(
    fig: go.Figure, trials: np.ndarray, wealth_pcts: np.ndarray, col_idx: int, showlegend: bool
) -> None:
    """Add 5-95 and 25-75 percentile bands with median path."""
    bands = [
        (4, None),
        (0, "rgba(10, 22, 40, 0.2)"),
        (3, None),
        (1, "rgba(10, 22, 40, 0.4)"),
    ]
    for idx, fillcolor in bands:
        fig.add_scatter(
            x=trials,
            y=wealth_pcts[idx],
            mode="lines",
            line=dict(width=0),
            fill="tonexty" if fillcolor else None,
            fillcolor=fillcolor,
            showlegend=False,
            row=1,
            col=col_idx,
        )
    fig.add_scatter(
        x=trials,
        y=wealth_pcts[2],
        mode="lines",
        line=dict(color=COLORS["blue"], width=2),
        name="Median",
        showlegend=showlegend,
        row=1,
        col=col_idx,
    )


# %%
# Simulate paths for different Kelly multiples
n_trials = 1000
n_simulations = 500
win_prob = 0.55
odds = 1.0
kelly = compute_kelly_fraction(win_prob, odds)
kelly_multiples = [0.25, 0.5, 1.0, 1.5, 2.0]
common_outcomes = np.random.default_rng(SEED).random((n_trials, n_simulations)) < win_prob

fig = make_subplots(
    rows=1,
    cols=len(kelly_multiples),
    subplot_titles=[f"{mult:.0%} Kelly" for mult in kelly_multiples],
    shared_yaxes=True,
)

for i, mult in enumerate(kelly_multiples):
    fraction = kelly * mult
    wealth = simulate_wealth_paths(common_outcomes, fraction, odds)
    wealth_pcts = np.percentile(wealth, [5, 25, 50, 75, 95], axis=1)
    add_percentile_bands(fig, np.arange(n_trials), wealth_pcts, col_idx=i + 1, showlegend=(i == 0))

# %%
fig.update_xaxes(title_text="Trial")
fig.update_yaxes(type="log", title_text="Wealth")
fig.update_layout(
    title="Growth peaks near full Kelly while overbetting widens downside dispersion",
    height=400,
    showlegend=True,
)
fig.show()

print(f"\nOptimal Kelly fraction: {kelly:.1%}")
print(
    f"At 2x Kelly ({2 * kelly:.1%}), left-tail wealth dispersion and drawdown risk increase sharply"
)

# %% [markdown]
# ### Distribution of Terminal Wealth


# %%
def terminal_wealth_distribution(
    n_trials: int,
    win_prob: float,
    fraction: float,
    odds: float = 1.0,
    initial_wealth: float = 100.0,
) -> pl.DataFrame:
    """Compute theoretical distribution of terminal wealth."""
    rv = binom(n=n_trials, p=win_prob)

    results = []
    for n_wins in range(n_trials + 1):
        n_losses = n_trials - n_wins
        log_wealth = (
            np.log(initial_wealth)
            + n_wins * np.log(1 + odds * fraction)
            + n_losses * np.log(1 - fraction)
        )
        prob = rv.pmf(n_wins)
        results.append(
            {
                "n_wins": n_wins,
                "log_wealth": log_wealth,
                "wealth": np.exp(log_wealth),
                "probability": prob,
            }
        )

    return pl.DataFrame(results)


# %%
# Compare terminal wealth distributions
n_trials = 100
win_prob = 0.55
kelly = compute_kelly_fraction(win_prob)

fig = go.Figure()

for mult in [0.5, 1.0, 1.5, 2.0]:
    dist = terminal_wealth_distribution(n_trials, win_prob, kelly * mult)

    # Compute expected log wealth
    expected_log_wealth = (dist["log_wealth"] * dist["probability"]).sum()

    fig.add_scatter(
        x=dist["log_wealth"].to_list(),
        y=dist["probability"].to_list(),
        mode="lines",
        name=f"{mult:.0%} Kelly (E[log W]={expected_log_wealth:.1f})",
        line=dict(width=2),
    )

fig.update_layout(
    title="Full Kelly maximizes the centre and widens the spread",
    xaxis_title="Log(Wealth)",
    yaxis_title="Probability",
    height=450,
)
fig.show()

# %% [markdown]
# ## Part 2: Kelly for Continuous Returns (Single Asset)
#
# For continuous return distributions, we use a Taylor expansion of $\log(1+x)$:
#
# $$\log(1+x) \approx x - \frac{x^2}{2} + \frac{x^3}{3} - ...$$
#
# For small returns, the optimal Kelly fraction becomes:
#
# $$f^* \approx \frac{\mu - r_f}{\sigma^2}$$
#
# where $\mu$ is expected return, $r_f$ is risk-free rate, and $\sigma^2$ is variance.

# %% [markdown]
# ### Taylor Expansion Visualization


# %%
def taylor_polynomial(n_terms: int) -> Callable:
    """Return Taylor polynomial approximation of log(1+x)."""
    x = symbols("x")
    expansion = series(log(1 + x), x, x0=0, n=n_terms).removeO()
    return sympy.lambdify(x, expansion, "numpy")


# %%
# Compare Taylor approximations
x_vals = np.linspace(-0.5, 1.0, 200)
true_log = np.log(1 + x_vals)

fig = go.Figure()

fig.add_scatter(
    x=x_vals,
    y=true_log,
    mode="lines",
    name="log(1+x)",
    line=dict(width=3, color=COLORS["blue"]),
)

colors = ml4t_palette(4, categorical=True)
for i, n in enumerate([2, 3, 4, 5]):
    taylor = taylor_polynomial(n)
    approx = taylor(x_vals)
    fig.add_scatter(
        x=x_vals,
        y=approx,
        mode="lines",
        name=f"Taylor (n={n})",
        line=dict(width=2, dash="dash", color=colors[i]),
    )

fig.update_layout(
    title="Higher-order terms matter as returns move away from zero",
    xaxis_title="x",
    yaxis_title="y",
    yaxis_range=[-2, 1.5],
    height=450,
)
fig.add_vline(x=0, line_dash="dot", line_color=COLORS["neutral"])
fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
fig.show()

# %% [markdown]
# ### Kelly Fraction for Market Returns


# %%
def kelly_fraction_continuous(
    mean_return: float, std_return: float, risk_free: float = 0.0
) -> float:
    """Kelly fraction for normally distributed returns."""
    excess_return = mean_return - risk_free
    return excess_return / (std_return**2)


# %% [markdown]
# #### Empirical Growth Objective


# %%
def empirical_growth_rate(returns: np.ndarray, fraction: float, risk_free: float = 0.0) -> float:
    """Annualized mean log growth over observed daily returns."""
    daily_risk_free = (1 + risk_free) ** (1 / 252) - 1
    wealth_multiples = 1 + daily_risk_free + fraction * (returns - daily_risk_free)
    if np.any(wealth_multiples <= 0):
        return -np.inf
    return float(np.log(wealth_multiples).mean() * 252)


# %% [markdown]
# #### Support-Constrained Numerical Kelly Fraction


# %%
def optimal_kelly_empirical(returns: np.ndarray, risk_free: float = 0.0) -> tuple[float, float]:
    """Optimize growth while keeping wealth positive for every observed return."""
    daily_risk_free = (1 + risk_free) ** (1 / 252) - 1
    excess_returns = returns - daily_risk_free
    worst_excess_return = float(excess_returns.min())
    if worst_excess_return >= 0:
        raise ValueError("Observed returns do not establish a finite leverage boundary")
    max_observed_safe = (1 + daily_risk_free) / -worst_excess_return
    result = minimize_scalar(
        lambda fraction: -empirical_growth_rate(returns, fraction, risk_free),
        bounds=(0.0, max_observed_safe * (1 - 1e-9)),
        method="bounded",
    )
    return float(result.x), max_observed_safe


# %%
# Load SPY as proxy for S&P 500 from canonical data
etf_data = load_etfs()
spy_data = etf_data.filter(pl.col("symbol") == "SPY").sort("timestamp")
sp500_returns = spy_data.select(
    "timestamp",
    pl.col("close").pct_change().alias("return"),
).drop_nulls()

# Compute annual statistics
annual_ret = float(sp500_returns["return"].mean() * 252)
annual_vol = float(sp500_returns["return"].std() * np.sqrt(252))

print(
    f"SPY statistics ({sp500_returns['timestamp'].min().year}-"
    f"{sp500_returns['timestamp'].max().year}):"
)
print(f"  Annual Return: {annual_ret:.2%}")
print(f"  Annual Volatility: {annual_vol:.2%}")
print(f"  Sharpe Ratio: {(annual_ret - RISK_FREE_RATE) / annual_vol:.2f}")

# Both estimators take the same annual risk-free rate
# so the analytical Taylor approximation and the numerical optimizer are
# compared like-for-like.
kelly_approx = kelly_fraction_continuous(annual_ret, annual_vol, RISK_FREE_RATE)
kelly_empirical, max_observed_safe = optimal_kelly_empirical(
    sp500_returns["return"].to_numpy(), RISK_FREE_RATE
)

print("\nKelly Fractions:")
print(f"  Analytical (Taylor approx): {kelly_approx:.1%}")
print(f"  Empirical log-growth optimum: {kelly_empirical:.1%}")
print(f"  Observed-return leverage boundary: {max_observed_safe:.1%}")

# %% [markdown]
# The empirical optimizer keeps wealth positive for every observed SPY return. That is a
# transparent historical-support constraint, not a guarantee against a worse future return.

# %% [markdown]
# ### Rolling Kelly Fraction

# %%
# Compute rolling Kelly fraction
rolling_window = min(252 * 5, max(63, sp500_returns.height // 2))
rolling_window_years = rolling_window / 252

# Rolling mean and std
rolling_stats = sp500_returns.with_columns(
    [
        pl.col("return").rolling_mean(window_size=rolling_window).alias("rolling_mean"),
        pl.col("return").rolling_std(window_size=rolling_window).alias("rolling_std"),
    ]
).drop_nulls()

# Compute Kelly fraction
rolling_stats = rolling_stats.with_columns(
    [
        (
            (pl.col("rolling_mean") * 252 - RISK_FREE_RATE) / (pl.col("rolling_std") ** 2 * 252)
        ).alias("kelly"),
    ]
)

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=[f"Rolling Statistics ({rolling_window_years:.1f}Y Window)", "Kelly Fraction"],
)

# Rolling return and volatility
fig.add_scatter(
    x=rolling_stats["timestamp"].to_list(),
    y=(rolling_stats["rolling_mean"] * 252).to_list(),
    name="Annual Return",
    line=dict(color=COLORS["blue"]),
    row=1,
    col=1,
)
fig.add_scatter(
    x=rolling_stats["timestamp"].to_list(),
    y=(rolling_stats["rolling_std"] * np.sqrt(252)).to_list(),
    name="Annual Volatility",
    line=dict(color=COLORS["amber"]),
    row=1,
    col=1,
)

# Kelly fraction
kelly_values = rolling_stats["kelly"].to_list()
fig.add_scatter(
    x=rolling_stats["timestamp"].to_list(),
    y=kelly_values,
    name="Kelly Fraction",
    line=dict(color=COLORS["copper"]),
    row=2,
    col=1,
)
_ = fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["neutral"], row=2, col=1)

# %%
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Annualized value", tickformat=".0%", row=1, col=1)
fig.update_yaxes(title_text="Kelly fraction", tickformat=".0%", row=2, col=1)
fig.update_layout(
    height=600,
    title="The same formula on rolling windows gives wildly different answers",
)
fig.show()

print("\nKelly Fraction Statistics:")
print(f"  Mean: {np.mean(kelly_values):.1%}")
print(f"  Std:  {np.std(kelly_values):.1%}")
print(f"  Min:  {np.min(kelly_values):.1%}")
print(f"  Max:  {np.max(kelly_values):.1%}")

# %% [markdown]
# ## Part 3: Kelly for Multiple Assets
#
# For a portfolio of $n$ assets, the optimal Kelly allocation is:
#
# $$\mathbf{f}^* = \Sigma^{-1} \boldsymbol{\mu}$$
#
# where $\Sigma^{-1}$ is the **precision matrix** (inverse covariance) and $\boldsymbol{\mu}$
# is the vector of expected excess returns.
#
# This has the same direction as the **maximum Sharpe ratio portfolio** under aligned assumptions.
# Tangency optimization normalizes the risky portfolio; Kelly also determines its leverage.

# %% [markdown]
# ### Load Multi-Asset Data

# %%
# Load data for a set of ETFs from canonical data
SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ"]
START_DATE = "2010-01-01"
END_DATE = "2024-12-01"

# Load from canonical ETF data
multi_etf = etf_data.filter(
    (pl.col("symbol").is_in(SYMBOLS))
    & (pl.col("timestamp") >= pl.lit(START_DATE).str.to_datetime())
    & (pl.col("timestamp") <= pl.lit(END_DATE).str.to_datetime())
)
prices = (
    multi_etf.select(["timestamp", "symbol", "close"])
    .pivot(on="symbol", index="timestamp", values="close")
    .sort("timestamp")
    .fill_null(strategy="forward")
    .drop_nulls()
)
assets = [column for column in prices.columns if column != "timestamp"]
returns = prices.select(
    "timestamp",
    *[pl.col(symbol).pct_change().alias(symbol) for symbol in assets],
).drop_nulls()
train_returns = returns.filter(pl.col("timestamp") <= pl.lit(TRAIN_END).str.to_datetime())
test_returns = returns.filter(pl.col("timestamp") > pl.lit(TRAIN_END).str.to_datetime())

# %%
if train_returns.is_empty() or test_returns.is_empty():
    raise ValueError("Both train and test windows must contain returns")
if train_returns["timestamp"].max() >= test_returns["timestamp"].min():
    raise ValueError("Train and test windows overlap")

print(
    f"Training: {train_returns['timestamp'].min()} to "
    f"{train_returns['timestamp'].max()} ({train_returns.height:,} returns)"
)
print(
    f"Test: {test_returns['timestamp'].min()} to "
    f"{test_returns['timestamp'].max()} ({test_returns.height:,} returns)"
)

# %%
print(f"Assets with complete history: {len(assets)}")

# %% [markdown]
# This fixed eight-ETF set uses current-vintage histories. It demonstrates allocation mechanics;
# it is not a point-in-time, survivorship-free universe study.

# %%
# Compute statistics
train_matrix = train_returns.select(assets).to_numpy()
annual_total_returns = train_matrix.mean(axis=0) * 252
annual_excess_returns = annual_total_returns - RISK_FREE_RATE
annual_cov = np.cov(train_matrix, rowvar=False, ddof=1) * 252

training_moments = pl.DataFrame(
    {
        "symbol": assets,
        "annual_return": annual_total_returns,
        "annual_excess_return": annual_excess_returns,
    }
)
print(f"Training covariance condition number: {np.linalg.cond(annual_cov):.1f}")
training_moments

# %% [markdown]
# ### Kelly Portfolio Allocation

# %%
# Kelly allocation
kelly_allocation = np.linalg.solve(annual_cov, annual_excess_returns)

print(f"Raw Kelly gross leverage: {np.abs(kelly_allocation).sum():.1%}")
print(f"Raw Kelly net exposure: {kelly_allocation.sum():.1%}")

# %%
fig = go.Figure()

fig.add_bar(
    x=assets,
    y=kelly_allocation,
    name="Kelly (Raw)",
    marker_color=COLORS["blue"],
)

# Add reference line for equal weight
fig.add_hline(
    y=1 / len(assets),
    line_dash="dash",
    line_color=COLORS["neutral"],
    annotation_text=f"Equal Weight: {1 / len(assets):.1%}",
    annotation_position="top left",
)

fig.update_layout(
    title="Kelly sizes positions without any notion of a budget",
    xaxis_title="Asset",
    yaxis_title="Allocation (Can Exceed 100%)",
    yaxis_tickformat=".0%",
    height=400,
)
fig.show()

# %% [markdown]
# ### Fractional Kelly for Risk Management
#
# Full Kelly is often too aggressive in practice due to:
# - Estimation error in $\mu$ and $\Sigma$
# - Non-normal return distributions (fat tails)
# - Borrowing constraints
#
# **Half-Kelly** ($f^*/2$) is common in practice, sacrificing some growth for stability.

# %% [markdown]
# The fractional weights below are the raw Kelly solution scaled, with no normalization applied
# afterwards, so each multiple carries genuinely different leverage rather than the same book
# rescaled. Equal weight is included at a gross exposure of exactly one, which is the reference
# every number in the table should be read against.

# %%
kelly_multiples = [0.25, 0.5, 1.0]
test_matrix = test_returns.select(assets).to_numpy()

portfolio_returns = {}
portfolio_gross = {}
for mult in kelly_multiples:
    weights = kelly_allocation * mult
    gross = float(np.abs(weights).sum())

    pf_returns = test_matrix @ weights
    portfolio_returns[f"{mult:.0%} Kelly"] = pf_returns
    portfolio_gross[f"{mult:.0%} Kelly"] = gross

# Add equal weight for comparison (gross = 1.0 by construction)
equal_weights = np.full(len(assets), 1 / len(assets))
portfolio_returns["Equal Weight"] = test_matrix @ equal_weights
portfolio_gross["Equal Weight"] = float(np.abs(equal_weights).sum())

# 1/gross is the threshold only if every position moves against the book by the same
# percentage on the same day - every long down that much and every short up that much.
# It is not a market move: net exposure is far smaller than gross, so a uniform decline
# of this size does not empty the account. It is the worst case the leverage admits.
print("Gross exposure, and the simultaneous adverse move on every position that would")
print("wipe the account out, with no cap applied:")
for name, gross in portfolio_gross.items():
    ruin_move = 1.0 / gross
    print(f"  {name:<14} {gross:6.2f}x gross    wiped out by {ruin_move:.2%} against every leg")

# %% [markdown]
# ### Performance Comparison with ml4t-diagnostic

# %%
dates = test_returns["timestamp"]
comparison_metrics = []

for name, pf_ret in portfolio_returns.items():
    # The lowest the cumulative wealth path reaches, not the worst single period: the
    # section reads the growth chart off this number, and a chart is a path.
    minimum_wealth_multiple = float(np.min(np.cumprod(1 + pf_ret)))
    if minimum_wealth_multiple <= 0:
        raise ValueError(f"{name} reaches nonpositive wealth in the test window")
    pa = PortfolioAnalysis(
        returns=pl.Series("returns", pf_ret),
        dates=dates,
        risk_free=RISK_FREE_RATE,
        periods_per_year=252,
    )

    metrics = pa.compute_summary_stats()
    comparison_metrics.append(
        {
            "strategy": name,
            "annual_return": metrics.annual_return,
            "annual_vol": metrics.annual_volatility,
            "sharpe": metrics.sharpe_ratio,
            "max_dd": metrics.max_drawdown,
            "gross_exposure": portfolio_gross[name],
            "min_wealth_multiple": minimum_wealth_multiple,
        }
    )

metrics_df = pl.DataFrame(comparison_metrics).sort("sharpe", descending=True)
metrics_df

# %% [markdown]
# Every frozen allocation remains above zero wealth over the test observations. This domain check is
# necessary for geometric wealth and drawdown, but it does not rule out ruin on an unseen return.

# %%
# Cumulative returns comparison
fig = go.Figure()

strategy_colors = dict(
    zip(
        portfolio_returns,
        ml4t_palette(len(portfolio_returns), categorical=True),
        strict=True,
    )
)

for name, pf_ret in portfolio_returns.items():
    cum_ret = (1 + pf_ret).cumprod()
    fig.add_scatter(
        x=dates,
        y=cum_ret,
        mode="lines",
        name=name,
        line=dict(color=strategy_colors[name], width=2),
    )

fig.update_layout(
    title="Growth curves that assume free, unlimited, never-called borrowing",
    xaxis_title="Date",
    yaxis_title="Growth of $1",
    height=500,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)
fig.show()

# %%
fig = go.Figure()
for row in metrics_df.iter_rows(named=True):
    fig.add_scatter(
        x=[row["annual_vol"]],
        y=[row["annual_return"]],
        mode="markers+text",
        name=row["strategy"],
        text=[row["strategy"]],
        textposition="middle right" if row["strategy"] == "Equal Weight" else "top center",
        marker=dict(
            size=max(10, abs(row["sharpe"]) * 15),
            color=strategy_colors[row["strategy"]],
        ),
    )
fig.update_layout(
    title="Full Kelly turns negative as volatility overwhelms average return",
    xaxis_title="Annualized volatility",
    yaxis_title="Annualized return",
    xaxis_tickformat=".0%",
    yaxis_tickformat=".0%",
    height=450,
)
fig.show()

# %% [markdown]
# ## Part 4: Practical Considerations
#
# ### When Kelly Works Well
# - **Known edge**: The probability distribution is well-estimated
# - **Many bets**: Long time horizon allows law of large numbers to work
# - **Reinvestment**: Profits are continuously reinvested
#
# ### Challenges in Finance
# - **Estimation error**: Returns are not stationary, $\mu$ and $\sigma$ change
# - **Fat tails**: Real returns have more extreme events than normal
# - **Correlation changes**: Covariance matrix shifts in crises
# - **Leverage costs**: Borrowing is not free
#
# ### Best Practices
# 1. **Use fractional Kelly** (half or quarter) to account for uncertainty
# 2. **Combine with risk constraints** (max position size, VaR limits)
# 3. **Regularize covariance estimates** (Ledoit-Wolf shrinkage)
# 4. **Monitor and adjust** as market conditions change

# %%
# Example: Shrinkage estimator for more stable Kelly allocation
lw = LedoitWolf()
lw.fit(train_matrix)
shrunk_cov = lw.covariance_ * 252

kelly_shrunk = np.linalg.solve(shrunk_cov, annual_excess_returns)

print("Kelly Allocation Comparison:")
comparison = pl.DataFrame(
    {
        "symbol": assets,
        "kelly_raw": kelly_allocation,
        "kelly_shrunk": kelly_shrunk,
        "difference": kelly_allocation - kelly_shrunk,
    }
)
print(comparison.with_columns(pl.col(pl.Float64).round(3)))
print(f"\nTotal leverage (raw):    {np.abs(kelly_allocation).sum():.1%}")
print(f"Total leverage (shrunk): {np.abs(kelly_shrunk).sum():.1%}")

# %%
# Save results
results = comparison.select("symbol", "kelly_raw", "kelly_shrunk").with_columns(
    pl.lit(TRAIN_END).str.to_date().alias("train_end")
)
OUTPUT_DIR = get_output_dir(17, "kelly")
results.write_parquet(OUTPUT_DIR / "kelly_allocations.parquet")
print("Saved Kelly allocations to ch17_kelly/kelly_allocations.parquet")

# %% [markdown]
# ## What the test window actually says

# %%
full_kelly_row = metrics_df.filter(pl.col("strategy") == "100% Kelly").row(0, named=True)
print(f"Gross exposure the training estimates ask for: {np.abs(kelly_allocation).sum():.1f}x")
print(f"At a quarter of that:                          {portfolio_gross['25% Kelly']:.1f}x")
print(
    "Lowest point of the full-Kelly wealth path:    "
    f"{full_kelly_row['min_wealth_multiple']:.3g}x the starting capital"
)

# %% [markdown]
# The last line is the one that decides how to read the growth chart above it. At its worst point
# the full-Kelly path is down to about one ten-millionth of the capital it started with, while
# carrying more than forty times that capital in gross positions - which is the same thing the
# `max_dd` column says when it reads exactly -1.0. The chart's vertical axis is logarithmic, so a
# fall of that size still looks like a line on the page; the number is what says the account is
# gone. No broker holds a position through it, and the arithmetic above says why: at that gross
# exposure it takes only 2.15% against every leg at once - every long down that much and every
# short up that much - to empty the account. That is a worst case rather than a market move, since
# net exposure is a fraction of gross and a uniform decline does not do it; what matters is that
# 46x leverage puts the worst case within a single ordinary session.
#
# Halving the fraction does not rescue it. Half-Kelly bottoms at 0.13x with a 99.2% drawdown, and
# only at a quarter does the worst point stay above 0.76x. What separates them is not the sign of
# the estimates but how much leverage is taken on the strength of them.
#
# So the curves are not a track record. They are what the formula's answer would have produced given
# borrowing that is unlimited, free of interest, and never called, and the value of computing them
# is precisely that the assumption is visible in the leverage number rather than hidden.

# %% [markdown]
# ## Key takeaways
#
# 1. **Kelly answers a question about growth, not about survival.** Maximizing the expected log of
#    terminal wealth is a defensible objective and it says nothing about the path, so the solution
#    is happy to accept a route through near-total loss on the way to a higher expectation.
# 2. **The formula has no notion of a budget.** Dividing expected returns by a covariance matrix
#    produces a number, and nothing in it is bounded by the capital available. Any use of it in
#    practice is the formula plus a leverage constraint, and the constraint is doing at least as
#    much work as the formula.
# 3. **Fractional Kelly is a cap, not a fix.** Taking a quarter of a forty-six-times solution
#    leaves eleven times. The reason to use a fraction is that the inputs are estimated and the
#    solution scales with the error in them; the reason it is not sufficient is that a fraction of
#    an unbounded number is still unbounded.
# 4. **The single-asset rolling estimate is the honest picture of the input problem.** The same
#    formula on the same instrument over rolling five-year windows swings across a range no
#    position sizer could act on. That instability is the estimate, not the market.
# 5. **Check the wealth path, not only the return series.** A leveraged return series can produce a
#    respectable annualized figure while passing through a wealth multiple that would have ended
#    the account. The minimum wealth multiple is one line of code and it is what makes the
#    difference visible.
#
# ### Known limitations
#
# - Borrowing is free, unlimited and never margin-called throughout. Introducing any of a funding
#   rate, a leverage cap, or a liquidation rule changes every result in Part 3.
# - Inputs are estimated once on the training window and frozen. A rolling re-estimate would change
#   the position sizes continuously, and the rolling-Kelly section shows by how much.
# - Returns are treated as normal, which is what makes the mean-over-variance approximation valid.
#   Real returns have fat tails, and the tail is exactly where a leveraged position dies.
# - The universe is a small set of funds selected because they exist today.
#
# **Next:** [`05_factor_allocation_evidence`](05_factor_allocation_evidence.ipynb) asks whether
# asset characteristics explain later returns at all. Section 17.4 covers Kelly sizing among the
# baseline allocators.
