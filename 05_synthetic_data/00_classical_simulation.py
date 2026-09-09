# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     formats: ipynb,py:percent
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
# # Chapter 5: Classical Simulation Methods
#
# **Docker image**: `ml4t`
#
# **Purpose**: Implement and compare classical Monte Carlo methods for generating
# synthetic financial data, building the foundation for the learned generative
# models that follow.
#
# This notebook teaches the **mechanics** of established Monte Carlo methods for
# generating synthetic financial data. We cover both the underlying mathematics
# and practical implementation, then show library shortcuts for production use.
#
# ## Why Simulate?
#
# Historical market data provides only **one path** through an infinite space of
# possibilities. Simulation generates alternative scenarios for:
#
# 1. **Risk Management**: VaR, stress testing, tail risk assessment
# 2. **Backtesting**: Validate strategies beyond historical experience
# 3. **Data Augmentation**: Larger datasets for ML model training
# 4. **Privacy**: Share synthetic data without exposing proprietary signals
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
#
# 1. **Implement** each classical stochastic model from scratch
# 2. **Explain** the assumptions and limitations of each model
# 3. **Choose** the appropriate model for your use case
# 4. **Use** library implementations for production work
#
# **Book Reference**: Chapter 5, Section 5.3 (Classical simulation baselines)
#
# **Prerequisites**: Basic probability and stochastic processes; familiarity
# with NumPy array operations. Requires ETF data from Chapter 2 (`load_etfs()`).
#
# ## Notebook Structure
#
# 1. **Part 1**: Continuous-Time Price Models (GBM, Jump-Diffusion, OU, Heston)
# 2. **Part 2**: Discrete-Time Volatility Model (GARCH with calibration)
# 3. **Part 3**: Bootstrap Methods (IID, Block, Stationary)
# 4. **Part 4**: Model Comparison
#
# ## Statistical Note
#
# We use **log-returns** throughout for consistency with continuous-time SDEs:
# - Log-return: $r_t = \ln(S_t / S_{t-1})$
# - Kurtosis values are **Fisher (excess) kurtosis**: Gaussian = 0
#
# ## References
#
# - Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering"
# - Politis, D. & Romano, J. (1994). "The Stationary Bootstrap"
# - Cont, R. (2001). "Empirical Properties of Asset Returns: Stylized Facts"
# - Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"

# %%
"""Classical Simulation Methods - Educational implementation of stochastic models."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
import seaborn as sns
from arch import arch_model
from arch.bootstrap import IIDBootstrap, MovingBlockBootstrap, StationaryBootstrap
from ml4t.data.providers import SyntheticProvider
from plotly.subplots import make_subplots
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import acf

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt, show_with_alt

# %% tags=["parameters"]
SEED = 42
N_PATHS = 200
N_BOOTSTRAP_REPLICATES = 200

# %% [markdown]
# Every generator below is spawned from `SEED`, so overriding the parameter
# moves every simulation in the notebook rather than only the ones that happen
# to read a global. `STREAMS` names one independent stream per use, and
# `stream()` hands out its generator.

# %%
set_global_seeds(SEED)

STREAMS = (
    "gbm",
    "jump_diffusion",
    "mean_reversion",
    "heston",
    "garch",
    "model_paths",
    "iid_bootstrap",
    "block_bootstrap",
    "stationary_bootstrap",
    "bootstrap_replicates",
)
_SEED_SEQUENCES = dict(zip(STREAMS, np.random.SeedSequence(SEED).spawn(len(STREAMS)), strict=True))


def stream(name: str) -> np.random.Generator:
    """Return the independent generator reserved for *name*."""
    return np.random.default_rng(_SEED_SEQUENCES[name])


# %% [markdown]
# ---
# # Part 1: Continuous-Time Price Models
#
# These models specify stochastic differential equations (SDEs) for price dynamics.
# Each captures different market phenomena:
#
# | Model | Key Feature | Best For |
# |-------|-------------|----------|
# | **GBM** | Log-normal prices, constant vol | Option pricing, baseline |
# | **Jump-Diffusion** | Rare extreme moves | Crash scenarios, tail risk |
# | **Mean-Reversion** | Prices drift to equilibrium | Spreads, commodities, rates |
# | **Heston** | Stochastic volatility | Vol surfaces, leverage effect |
#
# We implement each from scratch using local RNG for reproducibility,
# then show the equivalent library call.

# %% [markdown]
# ## Geometric Brownian Motion
#
# The foundation of quantitative finance. Price $S$ follows the stochastic
# differential equation:
#
# $$dS = \mu S \, dt + \sigma S \, dW$$
#
# where:
# - $\mu$ = drift (expected annual return)
# - $\sigma$ = volatility (annualized)
# - $dW$ = Wiener process increment (Brownian motion)
#
# ### Key Properties
#
# - **Log-returns are Gaussian**: $\ln(S_{t+1}/S_t) \sim N((\mu - \sigma^2/2)\Delta t, \sigma^2 \Delta t)$
# - **Prices are log-normal**: Always positive, no crashes below zero
# - **No memory**: Future returns independent of past (no autocorrelation)
# - **Constant volatility**: Same vol every day (unrealistic)
#
# ### Discretization (Euler-Maruyama)
#
# $$S_{t+\Delta t} = S_t \exp\left((\mu - \frac{\sigma^2}{2}) \Delta t + \sigma \sqrt{\Delta t} \, Z\right)$$
#
# where $Z \sim N(0,1)$.


# %%
def simulate_gbm(
    n_steps: int,
    mu: float,
    sigma: float,
    S0: float = 100.0,
    dt: float = 1 / 252,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate Geometric Brownian Motion price path.

    Parameters
    ----------
    n_steps : int
        Number of time steps
    mu : float
        Annual drift (expected return)
    sigma : float
        Annual volatility
    S0 : float
        Initial price
    dt : float
        Time step (1/252 for daily)
    rng : np.random.Generator, optional
        Random number generator for reproducibility

    Returns
    -------
    np.ndarray
        Price path of length n_steps + 1
    """
    if rng is None:
        rng = np.random.default_rng()

    # Standard normal random draws
    Z = rng.standard_normal(n_steps)

    # Log-return for each step
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative sum gives log-prices, then exponentiate
    log_prices = np.cumsum(log_returns)
    prices = S0 * np.exp(log_prices)

    # Prepend initial price
    return np.insert(prices, 0, S0)


# %% [markdown]
# ### GBM Simulation
#
# Generate two years of daily data (504 trading days) and inspect the return
# distribution. Excess kurtosis should sit near zero: GBM's log-returns are
# Gaussian by construction, and any departure is sampling noise from one path.

# %%
GBM_MU, GBM_SIGMA = 0.08, 0.20
N_STEPS = 504

gbm_prices = simulate_gbm(n_steps=N_STEPS, mu=GBM_MU, sigma=GBM_SIGMA, rng=stream("gbm"))
gbm_returns = np.diff(np.log(gbm_prices))

print(f"GBM simulation: {len(gbm_prices)} prices")
print(f"Annualized return: {gbm_returns.mean() * 252:.2%}")
print(f"Annualized volatility: {gbm_returns.std() * np.sqrt(252):.2%}")
print(f"Skewness: {skew(gbm_returns):.4f} (Gaussian: 0)")
print(f"Excess kurtosis: {kurtosis(gbm_returns, fisher=True, bias=False):.4f} (Gaussian: 0)")

# %% [markdown]
# ### Library Usage: GBM
#
# `SyntheticProvider` in the `ml4t-data` package generates the same processes
# behind an OHLCV interface. It draws from its own generator, so a provider path
# never matches a from-scratch path step for step; what should agree is the
# distribution the two draw from. GBM is the one model whose full parameter set
# the provider exposes (`annual_return`, `annual_volatility`), so it is the one
# case where the two are the same process and the realized moments are
# comparable.

# %%
provider = SyntheticProvider(
    model="gbm", annual_return=GBM_MU, annual_volatility=GBM_SIGMA, seed=SEED
)
df = provider.fetch_ohlcv("SYNTH", "2022-01-01", "2023-12-31", "daily")
provider_gbm_returns = np.diff(np.log(df["close"].to_numpy()))

print(f"SyntheticProvider GBM: {len(df)} bars")
print(f"  realized annual volatility: {provider_gbm_returns.std() * np.sqrt(252):.3f}")
print(f"  from-scratch, same parameters: {gbm_returns.std() * np.sqrt(252):.3f}")
print(f"  requested: {GBM_SIGMA:.3f}")

# %% [markdown]
# ### GBM Limitations
#
# GBM assumes returns are **i.i.d. Gaussian**, which contradicts observed
# "stylized facts" of financial returns:
#
# 1. **Fat tails**: Real returns have excess kurtosis (more extremes than Gaussian)
# 2. **Volatility clustering**: High-vol days tend to follow high-vol days
# 3. **Leverage effect**: Negative returns often increase future volatility
#
# Despite these limitations, GBM remains the workhorse for option pricing
# (Black-Scholes) due to its analytical tractability.

# %% [markdown]
# ## Jump-Diffusion (Merton)
#
# Adds occasional extreme moves to GBM via a compound Poisson process:
#
# $$dS = \mu S \, dt + \sigma S \, dW + S(e^Y - 1) \, dN$$
#
# where:
# - $dN$ = Poisson process with intensity $\lambda$ (jumps per year)
# - $Y \sim N(\mu_J, \sigma_J^2)$ = log jump size
#
# ### Drift Compensator
#
# To ensure $\mu$ represents the *total* expected return (including jumps),
# we subtract the expected jump contribution:
#
# $$k = \mathbb{E}[e^Y - 1] = \exp(\mu_J + \tfrac{1}{2}\sigma_J^2) - 1$$
#
# ### Discretization
#
# $$S_{t+\Delta t} = S_t \exp\left((\mu - \lambda k - \frac{\sigma^2}{2}) \Delta t
#   + \sigma \sqrt{\Delta t} Z + \sum_{i=1}^{N_t} Y_i\right)$$
#
# where $N_t \sim \text{Poisson}(\lambda \Delta t)$.


# %%
def simulate_jump_diffusion(
    n_steps: int,
    mu: float,
    sigma: float,
    lambda_: float,
    mu_jump: float,
    sigma_jump: float,
    S0: float = 100.0,
    dt: float = 1 / 252,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate Merton jump-diffusion price path with compensated drift.

    Parameters
    ----------
    n_steps : int
        Number of time steps
    mu : float
        Total annual drift (including jump contribution)
    sigma : float
        Annual volatility (diffusion part)
    lambda_ : float
        Jump intensity (expected jumps per year)
    mu_jump : float
        Mean of log jump size Y ~ N(mu_jump, sigma_jump^2)
    sigma_jump : float
        Std of log jump size
    S0 : float
        Initial price
    dt : float
        Time step
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    np.ndarray
        Price path of length n_steps + 1
    """
    if rng is None:
        rng = np.random.default_rng()

    # Jump compensator: E[e^Y - 1] so mu remains the total expected return
    k = np.exp(mu_jump + 0.5 * sigma_jump**2) - 1

    # Diffusion component with compensated drift
    Z = rng.standard_normal(n_steps)
    drift_compensated = mu - lambda_ * k - 0.5 * sigma**2
    diffusion = drift_compensated * dt + sigma * np.sqrt(dt) * Z

    # Jump component: compound Poisson
    N_jumps = rng.poisson(lambda_ * dt, n_steps)
    jump_component = np.zeros(n_steps)

    # Vectorized: for steps with jumps, sample and sum log jump sizes
    steps_with_jumps = np.where(N_jumps > 0)[0]
    for t in steps_with_jumps:
        jump_sizes = rng.normal(mu_jump, sigma_jump, N_jumps[t])
        jump_component[t] = np.sum(jump_sizes)

    # Combine and build price path
    log_returns = diffusion + jump_component
    log_prices = np.cumsum(log_returns)
    prices = S0 * np.exp(log_prices)

    return np.insert(prices, 0, S0)


# %% [markdown]
# ### Jump-Diffusion Simulation
#
# Simulate crash-only jumps: a jump intensity of five per year, and a mean log
# jump size that is negative, so each jump is a drop on average. The compensator
# is computed from those same parameters rather than retyped, so changing one
# cannot leave the printed diagnostic describing a different model.

# %%
JD_MU, JD_SIGMA = 0.08, 0.15
JD_LAMBDA, JD_MU_JUMP, JD_SIGMA_JUMP = 5.0, -0.03, 0.04

jd_prices = simulate_jump_diffusion(
    n_steps=N_STEPS,
    mu=JD_MU,
    sigma=JD_SIGMA,
    lambda_=JD_LAMBDA,
    mu_jump=JD_MU_JUMP,
    sigma_jump=JD_SIGMA_JUMP,
    rng=stream("jump_diffusion"),
)
jd_returns = np.diff(np.log(jd_prices))

k = np.exp(JD_MU_JUMP + 0.5 * JD_SIGMA_JUMP**2) - 1
print(f"Jump-Diffusion simulation: {len(jd_prices)} prices")
print(f"Mean jump size: {np.exp(JD_MU_JUMP) - 1:.2%}; intensity {JD_LAMBDA:.0f} per year")
print(f"Jump compensator k: {k:.4f} (subtracted from drift)")
print(f"Annualized return: {jd_returns.mean() * 252:.2%}")
print(f"Annualized volatility: {jd_returns.std() * np.sqrt(252):.2%}")
print(f"Skewness: {skew(jd_returns):.4f}")
print(f"Excess kurtosis: {kurtosis(jd_returns, fisher=True, bias=False):.4f} (> 0 from jumps)")

# %% [markdown]
# ### Library Usage: Jump-Diffusion
#
# The provider's `gbm_jump` model is **not** the model above. It exposes only
# `annual_return` and `annual_volatility`; the jump process is fixed internally
# at five jumps per year with a **zero-mean** jump size. Symmetric jumps produce
# fat tails without skew, while the crash-only jumps above produce both. The
# printed skewness shows the difference, and it is the reason to implement the
# jump process yourself when the asymmetry is the point.

# %%
provider = SyntheticProvider(
    model="gbm_jump", annual_return=JD_MU, annual_volatility=JD_SIGMA, seed=SEED
)
df = provider.fetch_ohlcv("SYNTH", "2022-01-01", "2023-12-31", "daily")
provider_jd_returns = np.diff(np.log(df["close"].to_numpy()))

print(f"SyntheticProvider gbm_jump: {len(df)} bars")
print(f"  skewness, provider (zero-mean jumps):  {skew(provider_jd_returns):+.3f}")
print(f"  skewness, from scratch (down-jumps):   {skew(jd_returns):+.3f}")

# %% [markdown]
# ## Mean-Reversion (Ornstein-Uhlenbeck)
#
# Prices gravitate toward a long-term equilibrium $\theta$:
#
# $$d(\log S) = \kappa(\theta - \log S) \, dt + \sigma \, dW$$
#
# where:
# - $\kappa$ = speed of mean reversion
# - $\theta$ = long-term mean (log-price level)
# - Half-life: $t_{1/2} = \ln(2) / \kappa$
#
# ### Key Properties
#
# - **Stationary**: Prices fluctuate around equilibrium
# - **No trends**: Can't capture bull/bear markets
# - **Negative autocorrelation**: Today's move partially reversed tomorrow
#
# ### Discretization Options
#
# **Euler-Maruyama** (approximate):
# $$X_{t+\Delta t} = X_t + \kappa(\theta - X_t)\Delta t + \sigma\sqrt{\Delta t} Z$$
#
# **Exact transition** (closed-form for OU):
# $$X_{t+\Delta t} = \theta + (X_t - \theta)e^{-\kappa\Delta t}
#   + \sigma\sqrt{\frac{1 - e^{-2\kappa\Delta t}}{2\kappa}} Z$$
#
# We implement both to compare discretization error.


# %%
def simulate_mean_reversion_euler(
    n_steps: int,
    kappa: float,
    theta: float,
    sigma: float,
    S0: float = 100.0,
    dt: float = 1 / 252,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate OU process using Euler-Maruyama discretization.

    Parameters
    ----------
    n_steps : int
        Number of time steps
    kappa : float
        Mean reversion speed (annualized)
    theta : float
        Long-term mean (log-price level)
    sigma : float
        Volatility (annualized)
    S0 : float
        Initial price
    dt : float
        Time step
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    np.ndarray
        Price path of length n_steps + 1
    """
    if rng is None:
        rng = np.random.default_rng()

    log_prices = np.zeros(n_steps + 1)
    log_prices[0] = np.log(S0)

    Z = rng.standard_normal(n_steps)

    for t in range(n_steps):
        log_prices[t + 1] = (
            log_prices[t] + kappa * (theta - log_prices[t]) * dt + sigma * np.sqrt(dt) * Z[t]
        )

    return np.exp(log_prices)


# %% [markdown]
# ### Exact Transition Density
#
# The OU process has a closed-form transition density, eliminating
# discretization error entirely. We implement both to compare accuracy.


# %%
def simulate_mean_reversion_exact(
    n_steps: int,
    kappa: float,
    theta: float,
    sigma: float,
    S0: float = 100.0,
    dt: float = 1 / 252,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate OU process using exact transition density.

    The exact solution eliminates discretization error entirely.
    """
    if rng is None:
        rng = np.random.default_rng()

    log_prices = np.zeros(n_steps + 1)
    log_prices[0] = np.log(S0)

    # Precompute constants
    exp_neg_kappa_dt = np.exp(-kappa * dt)
    std_dev = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))

    Z = rng.standard_normal(n_steps)

    for t in range(n_steps):
        log_prices[t + 1] = theta + (log_prices[t] - theta) * exp_neg_kappa_dt + std_dev * Z[t]

    return np.exp(log_prices)


# %% [markdown]
# ### Mean-Reversion Simulation
#
# The half-life is derived from the reversion speed rather than retyped, so it
# tracks any change to the parameter. Euler and exact are then run from the
# **same** stream of shocks: both consume one standard normal per step in the
# same order, so the two paths differ only by discretization error and the gap
# between them measures exactly that.

# %%
MR_KAPPA, MR_SIGMA = 2.0, 0.15
MR_EQUILIBRIUM = 100.0

mr_prices = simulate_mean_reversion_exact(
    n_steps=N_STEPS,
    kappa=MR_KAPPA,
    theta=np.log(MR_EQUILIBRIUM),
    sigma=MR_SIGMA,
    rng=stream("mean_reversion"),
)
mr_returns = np.diff(np.log(mr_prices))

half_life_days = np.log(2) / MR_KAPPA * 252
print(f"Mean-Reversion (exact) simulation: {len(mr_prices)} prices")
print(f"Half-life: {half_life_days:.0f} trading days")
print(f"Final price: {mr_prices[-1]:.2f} (equilibrium: {MR_EQUILIBRIUM:.0f})")
print(f"Return autocorr(1): {np.corrcoef(mr_returns[:-1], mr_returns[1:])[0, 1]:.4f}")

mr_euler = simulate_mean_reversion_euler(
    n_steps=N_STEPS,
    kappa=MR_KAPPA,
    theta=np.log(MR_EQUILIBRIUM),
    sigma=MR_SIGMA,
    rng=stream("mean_reversion"),
)
max_diff = np.max(np.abs(mr_prices - mr_euler))
print(f"Euler vs exact, same shocks, max price difference: {max_diff:.4f}")

# %% [markdown]
# ### Library Usage: Mean-Reversion
#
# The provider's `mean_revert` model fixes the reversion speed internally at the
# same value used above and reverts to its own `base_price`, so neither the speed
# nor the equilibrium is a parameter you can set. Reversion speed is usually the
# quantity you want to fit for a spread or a rate, which is why the from-scratch
# implementation stays useful.

# %%
provider = SyntheticProvider(
    model="mean_revert", annual_volatility=MR_SIGMA, base_price=MR_EQUILIBRIUM, seed=SEED
)
df = provider.fetch_ohlcv("SYNTH", "2022-01-01", "2023-12-31", "daily")
provider_mr_close = df["close"].to_numpy()

print(f"SyntheticProvider mean_revert: {len(df)} bars")
print(f"  mean price: {provider_mr_close.mean():.2f} (equilibrium {MR_EQUILIBRIUM:.0f})")
print(f"  from-scratch mean price: {mr_prices.mean():.2f}")

# %% [markdown]
# ## Heston (Stochastic Volatility)
#
# Volatility itself is random, following a separate mean-reverting process:
#
# $$dS = \mu S \, dt + \sqrt{v} S \, dW_S$$
# $$dv = \kappa(\theta - v) \, dt + \xi \sqrt{v} \, dW_v$$
# $$\text{Corr}(dW_S, dW_v) = \rho$$
#
# where:
# - $v$ = instantaneous variance
# - $\kappa$ = variance mean-reversion speed
# - $\theta$ = long-term variance
# - $\xi$ = volatility of volatility ("vol of vol")
# - $\rho$ = correlation between price and variance shocks (leverage effect)
#
# ### Key Properties
#
# - **Stochastic volatility**: Vol changes unpredictably
# - **Leverage effect**: $\rho < 0$ means price drops increase volatility
# - **Fat tails**: From randomness in volatility
# - **Volatility clustering**: From mean-reversion in variance
# - **Feller condition**: $2\kappa\theta > \xi^2$ prevents variance from hitting zero
#
# ### Discretization: Full Truncation Euler
#
# To handle potential negative variance, we use **full truncation**:
# apply $\max(v, 0)$ consistently in both drift and diffusion terms.


# %%
def simulate_heston(
    n_steps: int,
    mu: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    S0: float = 100.0,
    dt: float = 1 / 252,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Heston stochastic volatility path using full truncation Euler.

    Full truncation applies max(v, 0) to the current variance before
    computing both drift and diffusion terms, ensuring consistency.

    Parameters
    ----------
    n_steps : int
        Number of time steps
    mu : float
        Drift
    v0 : float
        Initial variance
    kappa : float
        Variance mean-reversion speed
    theta : float
        Long-term variance
    xi : float
        Volatility of variance (vol of vol)
    rho : float
        Correlation between price and variance shocks
    S0 : float
        Initial price
    dt : float
        Time step
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Price path and variance path

    Note
    ----
    For production use, consider Andersen's QE scheme which has better
    accuracy near the boundary.
    """
    if rng is None:
        rng = np.random.default_rng()

    prices = np.zeros(n_steps + 1)
    variance = np.zeros(n_steps + 1)
    prices[0] = S0
    variance[0] = v0

    sqrt_dt = np.sqrt(dt)

    for t in range(n_steps):
        # Full truncation: apply floor BEFORE computing terms
        v_t = max(variance[t], 0)
        sqrt_v_t = np.sqrt(v_t)

        # Correlated Brownian motions
        Z1 = rng.standard_normal()
        Z2 = rng.standard_normal()
        W_v = Z1
        W_S = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        # Variance update (full truncation Euler)
        variance[t + 1] = v_t + kappa * (theta - v_t) * dt + xi * sqrt_v_t * sqrt_dt * W_v
        # Floor the result for next iteration
        variance[t + 1] = max(variance[t + 1], 0)

        # Price update
        prices[t + 1] = prices[t] * np.exp((mu - 0.5 * v_t) * dt + sqrt_v_t * sqrt_dt * W_S)

    return prices, variance


# %% [markdown]
# ### Heston Simulation
#
# The variance starts at its long-run level, so the long-run volatility is the
# square root of `HESTON_THETA`. The Feller condition is evaluated from the same
# named parameters that drive the simulation, and asserted rather than printed as
# advice: a violated Feller condition means the truncation is doing real work and
# the path no longer represents the model the prose describes.

# %%
HESTON_MU = 0.05
HESTON_KAPPA, HESTON_THETA, HESTON_XI, HESTON_RHO = 5.0, 0.04, 0.3, -0.7

feller_lhs = 2 * HESTON_KAPPA * HESTON_THETA
feller_rhs = HESTON_XI**2
assert feller_lhs > feller_rhs, (
    f"Feller condition violated: 2*kappa*theta={feller_lhs:.3f} <= xi^2={feller_rhs:.3f}"
)

heston_prices, heston_var = simulate_heston(
    n_steps=N_STEPS,
    mu=HESTON_MU,
    v0=HESTON_THETA,
    kappa=HESTON_KAPPA,
    theta=HESTON_THETA,
    xi=HESTON_XI,
    rho=HESTON_RHO,
    rng=stream("heston"),
)
heston_returns = np.diff(np.log(heston_prices))

print(f"Heston simulation: {len(heston_prices)} prices")
print(f"Long-run volatility: {np.sqrt(HESTON_THETA):.1%}")
print(f"Feller: 2*kappa*theta = {feller_lhs:.2f} > xi^2 = {feller_rhs:.2f}")
print(f"Realized vol range: {np.sqrt(heston_var).min():.1%} to {np.sqrt(heston_var).max():.1%}")
assert heston_var.min() >= 0, "full truncation should keep variance non-negative"
print(f"Excess kurtosis: {kurtosis(heston_returns, fisher=True, bias=False):.4f}")

# %% [markdown]
# ### Library Usage: Heston
#
# The provider exposes the full Heston parameter set except the initial variance,
# which it always starts at the long-run level `heston_theta` — the same choice
# made above. Passing all four means the provider runs the same process, so the
# realized volatility of the two paths should agree up to sampling noise. Leaving
# `heston_kappa` at its default would silently simulate a different, slower
# reverting variance process.

# %%
provider = SyntheticProvider(
    model="heston",
    annual_return=HESTON_MU,
    heston_kappa=HESTON_KAPPA,
    heston_theta=HESTON_THETA,
    heston_xi=HESTON_XI,
    heston_rho=HESTON_RHO,
    seed=SEED,
)
df = provider.fetch_ohlcv("SYNTH", "2022-01-01", "2023-12-31", "daily")
provider_heston_returns = np.diff(np.log(df["close"].to_numpy()))

print(f"SyntheticProvider heston: {len(df)} bars")
print(f"  realized annual volatility: {provider_heston_returns.std() * np.sqrt(252):.3f}")
print(f"  from-scratch, same parameters: {heston_returns.std() * np.sqrt(252):.3f}")
print(f"  long-run level sqrt(theta): {np.sqrt(HESTON_THETA):.3f}")

# %% [markdown]
# ---
# # Part 2: Discrete-Time Volatility Model (GARCH)
#
# Unlike the continuous-time SDEs above, GARCH is a **discrete-time model**
# for conditional variance. It models how return volatility evolves based on
# past shocks.
#
# ## GARCH(1,1)
#
# $$r_t = \mu + \sigma_t \varepsilon_t, \quad \varepsilon_t \sim N(0,1)$$
# $$\sigma^2_t = \omega + \alpha (r_{t-1} - \mu)^2 + \beta \sigma^2_{t-1}$$
#
# where:
# - $\mu$ = unconditional mean return (drift)
# - $\omega$ = base variance (intercept)
# - $\alpha$ = reaction to recent shocks (news impact)
# - $\beta$ = persistence of past variance (memory)
# - $\alpha + \beta < 1$ required for stationarity
#
# ### Key Properties
#
# - **Volatility clustering**: $\beta > 0$ means vol persists
# - **Fat tails**: From time-varying volatility
# - **Mean-reverting volatility**: Unconditional variance = $\omega / (1 - \alpha - \beta)$
# - **Leverage effect**: Requires asymmetric extensions (GJR-GARCH, EGARCH)
#
# ### GARCH vs Heston
#
# | Aspect | GARCH | Heston |
# |--------|-------|--------|
# | Time | Discrete | Continuous |
# | Leverage | Extensions needed | Built-in ($\rho$) |
# | Calibration | MLE from data | Option surface |
# | Analytical | Limited | Semi-closed form |
#
# ## Calibration: Fitting GARCH to Data
#
# Unlike SDEs where we **choose** parameters (drift, volatility), GARCH is
# typically **fitted** to historical data via maximum likelihood estimation.

# %%
# Load SPY returns for GARCH calibration
etf_data = load_etfs()
spy_close = (
    etf_data.filter(pl.col("symbol") == "SPY")
    .sort("timestamp")
    .select("close")
    .to_series()
    .to_numpy()
)

# Compute log returns in percent (arch library convention; matches exp(cumsum) reconstruction)
spy_log_returns_pct = np.diff(np.log(spy_close)) * 100

print(f"SPY log-returns: {len(spy_log_returns_pct)} observations")
print(f"Mean: {spy_log_returns_pct.mean():.4f}% daily")
print(f"Std: {spy_log_returns_pct.std():.4f}%")

# %% [markdown]
# ### Fit GARCH(1,1) to SPY

# %%
# Fit GARCH(1,1) model
am = arch_model(spy_log_returns_pct, mean="Constant", vol="GARCH", p=1, q=1)
res = am.fit(disp="off")

# Extract calibrated parameters
mu_fit = res.params["mu"]
omega_fit = res.params["omega"]
alpha_fit = res.params["alpha[1]"]
beta_fit = res.params["beta[1]"]

print("Calibrated GARCH(1,1) parameters:")
print(f"  mu (mean):     {mu_fit:.6f}% daily")
print(f"  omega:         {omega_fit:.6f}")
print(f"  alpha (news):  {alpha_fit:.4f}")
print(f"  beta (memory): {beta_fit:.4f}")
print(f"  persistence:   {alpha_fit + beta_fit:.4f}")
print(f"  unconditional vol: {np.sqrt(omega_fit / (1 - alpha_fit - beta_fit)):.4f}% daily")

# %% [markdown]
# ### Simulate from Calibrated Parameters
#
# Now we can simulate new paths using the fitted parameters.


# %%
def simulate_garch(
    n_steps: int,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    sigma0: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate GARCH(1,1) return series with mean.

    Parameters
    ----------
    n_steps : int
        Number of time steps
    mu : float
        Mean return (same units as omega)
    omega : float
        Base variance (intercept)
    alpha : float
        Shock coefficient (news impact)
    beta : float
        Persistence coefficient
    sigma0 : float, optional
        Initial volatility. If None, use unconditional volatility.
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Returns and volatility paths
    """
    if rng is None:
        rng = np.random.default_rng()

    # Unconditional variance
    uncond_var = omega / (1 - alpha - beta)
    if sigma0 is None:
        sigma0 = np.sqrt(uncond_var)

    returns = np.zeros(n_steps)
    sigma = np.zeros(n_steps)
    sigma[0] = sigma0

    for t in range(n_steps):
        # Generate return with mean
        eps = rng.standard_normal()
        returns[t] = mu + sigma[t] * eps

        # Update volatility for next period
        if t < n_steps - 1:
            shock = returns[t] - mu  # Deviation from mean
            sigma[t + 1] = np.sqrt(omega + alpha * shock**2 + beta * sigma[t] ** 2)

    return returns, sigma


# %% [markdown]
# ### GARCH Simulation from Calibrated Parameters
#
# Use the fitted SPY parameters to generate a synthetic path and compare
# the resulting moments with the calibrated values.

# %%
garch_log_returns_pct, garch_vol = simulate_garch(
    n_steps=N_STEPS,
    mu=mu_fit,
    omega=omega_fit,
    alpha=alpha_fit,
    beta=beta_fit,
    rng=stream("garch"),
)

garch_log_returns = garch_log_returns_pct / 100
garch_prices = 100 * np.exp(np.cumsum(np.insert(garch_log_returns, 0, 0)))

print(f"GARCH simulation (calibrated to SPY): {len(garch_prices)} prices")
print(f"Simulated annual return: {garch_log_returns.mean() * 252:.2%}")
print(f"Simulated annual vol: {garch_log_returns.std() * np.sqrt(252):.1%}")
print(f"Excess kurtosis: {kurtosis(garch_log_returns, fisher=True, bias=False):.4f}")

# %% [markdown]
# ### Library Usage: GARCH
#
# The provider takes `garch_alpha` and `garch_beta` but **not** omega: it derives
# omega from `annual_volatility` and the requested frequency, and a `garch_omega`
# argument is accepted, ignored, and warned about. Passing the fitted omega
# therefore does nothing, and the resulting series is calibrated to SPY in its
# persistence but not in its level.
#
# To carry the calibration across, convert the fitted unconditional variance into
# an annual volatility and pass that instead. Because `arch` is fitted on
# returns in percent, the conversion divides by 100 before annualizing.

# %%
uncond_var_pct = omega_fit / (1 - alpha_fit - beta_fit)
fitted_annual_vol = np.sqrt(uncond_var_pct) / 100 * np.sqrt(252)

provider = SyntheticProvider(
    model="garch",
    annual_volatility=fitted_annual_vol,
    garch_alpha=alpha_fit,
    garch_beta=beta_fit,
    seed=SEED,
)
df = provider.fetch_ohlcv("SYNTH", "2022-01-01", "2023-12-31", "daily")
provider_garch_returns = np.diff(np.log(df["close"].to_numpy()))

print(f"SyntheticProvider garch: {len(df)} bars")
print(f"  fitted unconditional annual volatility: {fitted_annual_vol:.3f}")
print(f"  realized, provider:    {provider_garch_returns.std() * np.sqrt(252):.3f}")
print(f"  realized, from scratch: {garch_log_returns.std() * np.sqrt(252):.3f}")

# %% [markdown]
# ---
# ## Visualize All Parametric Models

# %%
# Collect all simulations
all_models = {
    "GBM": gbm_prices,
    "Jump-Diffusion": jd_prices,
    "Mean-Reversion": mr_prices,
    "Heston": heston_prices,
    "GARCH": garch_prices,
}

# %%
# Create comparison plot
fig = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=list(all_models.keys()) + ["Combined (Normalized)"],
    vertical_spacing=0.12,
)

colors = [COLORS["blue"], COLORS["amber"], COLORS["copper"], COLORS["neutral"], COLORS["positive"]]

# Individual plots (no legend - subplot titles identify each)
for idx, (name, prices) in enumerate(all_models.items()):
    row, col = (idx // 3) + 1, (idx % 3) + 1
    fig.add_trace(
        go.Scatter(
            y=prices, mode="lines", name=name, line=dict(color=colors[idx]), showlegend=False
        ),
        row=row,
        col=col,
    )

# Combined normalized plot (with legend for comparison)
for idx, (name, prices) in enumerate(all_models.items()):
    normalized = 100 * prices / prices[0]
    fig.add_trace(
        go.Scatter(y=normalized, mode="lines", name=name, line=dict(color=colors[idx])),
        row=2,
        col=3,
    )

fig.update_layout(
    title_text="One simulated path per model, and the five overlaid",
    height=600,
    width=950,
)
show_plotly_with_alt(
    fig,
    "Six panels: one simulated price path for each of GBM, jump-diffusion, "
    "mean-reversion, Heston and GARCH, and a sixth panel overlaying all five "
    "rebased to 100 at the start.",
)

# %% [markdown]
# ### Normalized Price Paths (Grayscale-Compatible)
#
# A matplotlib companion to the plotly grid above, using distinct line styles
# that remain distinguishable in grayscale print.

# %%
# Line styles for grayscale compatibility - varied grays + distinct patterns
LINE_STYLES = [
    {"linestyle": "-", "linewidth": 1.8, "color": "black"},  # GBM: solid black
    {"linestyle": "--", "linewidth": 1.8, "color": "#404040"},  # Jump-Diffusion: dashed dark gray
    {
        "linestyle": "-.",
        "linewidth": 1.8,
        "color": "#606060",
    },  # Mean-Reversion: dash-dot medium gray
    {"linestyle": ":", "linewidth": 2.2, "color": "#202020"},  # Heston: dotted near-black (thicker)
    {
        "linestyle": (0, (5, 2, 1, 2)),
        "linewidth": 1.8,
        "color": "#808080",
    },  # GARCH: long-dash-dot gray
]

# Built and styled in one cell so the inline backend cannot flush a
# half-constructed figure.
fig, ax = plt.subplots(figsize=(12, 4.5))

# Plot each model normalized to 100
for idx, (name, prices) in enumerate(all_models.items()):
    normalized = 100 * prices / prices[0]
    ax.plot(normalized, label=name, **LINE_STYLES[idx])

# Styling
ax.set_xlabel("Trading Days")
ax.set_ylabel("Normalized Price (Start = 100)")
ax.set_title("Classical Simulation Models: Normalized Price Paths")

# Legend outside plot area to avoid overlap
ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.01, 0.99),
    frameon=True,
    fancybox=False,
    edgecolor="lightgray",
    fontsize=9,
)

# Add horizontal reference line at 100
ax.axhline(y=100, color="gray", linewidth=0.5, linestyle="-", alpha=0.4)

# Despine (seaborn style)
sns.despine(ax=ax)

show_with_alt(
    fig,
    "Five simulated price paths rebased to 100, drawn in distinct line styles "
    "so they stay separable in grayscale, with a reference line at the starting "
    "level of 100.",
)

# %% [markdown]
# These are five single realizations, one per model, and a picture of five
# paths cannot separate a model's properties from the draw that produced them.
# Two things it does show are structural rather than incidental: the
# jump-diffusion panel contains discontinuities that no diffusion path can
# produce, and the mean-reversion panel stays inside a band around its
# equilibrium while the others wander. Everything else — which path looks
# calmest, which drew the deepest drawdown — is a property of the draw. The
# next section measures the model properties across many draws instead.

# %% [markdown]
# ## Model Statistics Comparison
#
# Each model is simulated `N_PATHS` times so the comparison describes the
# **model** rather than one path. The sample excess kurtosis of a Gaussian
# series of length `N_STEPS` has a standard error of $\sqrt{24/T}$, which at
# this path length is large relative to the gaps between several of these
# models, so an ordering read off one path per model is mostly noise. The cell
# below prints it alongside GBM's measured spread.
#
# The models are also **not** run at a common volatility: `GBM_SIGMA` and
# $\sqrt{\texttt{HESTON\_THETA}}$ set one level, `MR_SIGMA` and `JD_SIGMA`
# another, and GARCH inherits whatever the SPY fit implies. The volatility
# column therefore reports the input, not a finding, and drawdown depth is not
# comparable across rows for the same reason. Excess kurtosis and skewness are
# the columns that describe the models, because no parameter sets them
# directly.


# %%
def compute_model_stats(prices: np.ndarray, name: str) -> dict:
    """Compute key statistics for a price path using log-returns."""
    log_returns = np.diff(np.log(prices))
    return {
        "model": name,
        "annual_return": log_returns.mean() * 252,
        "annual_volatility": log_returns.std() * np.sqrt(252),
        "skewness": skew(log_returns),
        "excess_kurtosis": kurtosis(log_returns, fisher=True, bias=False),
        "max_drawdown": np.min(prices / np.maximum.accumulate(prices) - 1),
    }


# %% [markdown]
# ### Simulating a Population of Paths
#
# `simulate_path_population` re-runs each generator `N_PATHS` times from
# independent child streams of the `model_paths` seed, so the paths are
# independent of each other and of the illustrative paths plotted above.


# %%
def simulate_path_population(n_paths: int) -> dict[str, list[np.ndarray]]:
    """Simulate *n_paths* independent price paths for each parametric model."""
    seeds = _SEED_SEQUENCES["model_paths"].spawn(5)
    draw = {
        name: [np.random.default_rng(c) for c in seq.spawn(n_paths)]
        for name, seq in zip(all_models, seeds, strict=True)
    }

    populations: dict[str, list[np.ndarray]] = {}
    populations["GBM"] = [
        simulate_gbm(n_steps=N_STEPS, mu=GBM_MU, sigma=GBM_SIGMA, rng=r) for r in draw["GBM"]
    ]
    populations["Jump-Diffusion"] = [
        simulate_jump_diffusion(
            n_steps=N_STEPS,
            mu=JD_MU,
            sigma=JD_SIGMA,
            lambda_=JD_LAMBDA,
            mu_jump=JD_MU_JUMP,
            sigma_jump=JD_SIGMA_JUMP,
            rng=r,
        )
        for r in draw["Jump-Diffusion"]
    ]
    populations["Mean-Reversion"] = [
        simulate_mean_reversion_exact(
            n_steps=N_STEPS,
            kappa=MR_KAPPA,
            theta=np.log(MR_EQUILIBRIUM),
            sigma=MR_SIGMA,
            rng=r,
        )
        for r in draw["Mean-Reversion"]
    ]
    populations["Heston"] = [
        simulate_heston(
            n_steps=N_STEPS,
            mu=HESTON_MU,
            v0=HESTON_THETA,
            kappa=HESTON_KAPPA,
            theta=HESTON_THETA,
            xi=HESTON_XI,
            rho=HESTON_RHO,
            rng=r,
        )[0]
        for r in draw["Heston"]
    ]
    populations["GARCH"] = [
        100
        * np.exp(
            np.cumsum(
                np.insert(
                    simulate_garch(
                        n_steps=N_STEPS,
                        mu=mu_fit,
                        omega=omega_fit,
                        alpha=alpha_fit,
                        beta=beta_fit,
                        rng=r,
                    )[0]
                    / 100,
                    0,
                    0,
                )
            )
        )
        for r in draw["GARCH"]
    ]
    return populations


path_populations = simulate_path_population(N_PATHS)
assert all(len(v) == N_PATHS for v in path_populations.values())

stats_df = pl.DataFrame(
    [compute_model_stats(path, name) for name, paths in path_populations.items() for path in paths]
)


def summarize(column: str) -> pl.DataFrame:
    """Median and 5th-95th percentile range of *column* by model."""
    return stats_df.group_by("model", maintain_order=True).agg(
        pl.col(column).median().alias("median"),
        pl.col(column).quantile(0.05).alias("p05"),
        pl.col(column).quantile(0.95).alias("p95"),
    )


print(f"Gaussian excess-kurtosis standard error at T={N_STEPS}: {np.sqrt(24 / N_STEPS):.3f}")
print("Volatility each model was parameterized at (annualized):")
print(
    f"  GBM {GBM_SIGMA:.2f}, Heston {np.sqrt(HESTON_THETA):.2f}, "
    f"mean-reversion {MR_SIGMA:.2f}, jump-diffusion diffusion part {JD_SIGMA:.2f}"
)

print(f"\nExcess kurtosis across {N_PATHS} paths per model:")
print(summarize("excess_kurtosis"))
print(f"\nSkewness across {N_PATHS} paths per model:")
print(summarize("skewness"))

# %%
# Return distribution comparison
fig = go.Figure()

for idx, (name, prices) in enumerate(all_models.items()):
    log_returns = np.diff(np.log(prices))
    fig.add_trace(
        go.Histogram(
            x=log_returns,
            name=name,
            opacity=0.6,
            nbinsx=50,
            histnorm="probability density",
            marker_color=colors[idx],
        )
    )

fig.update_layout(
    title="Return distributions differ mainly in the tails",
    xaxis_title="Daily Log-Return",
    yaxis_title="Density",
    barmode="overlay",
)
show_plotly_with_alt(
    fig,
    "Overlaid density histograms of daily log-returns from the five simulated "
    "paths; the bodies largely coincide while the jump-diffusion series extends "
    "furthest into the negative tail.",
)

# %% [markdown]
# ### Which Models Separate
#
# Rather than reading an ordering off the medians, compare each model's excess
# kurtosis against GBM's. GBM is the null here: its log-returns are Gaussian by
# construction, so its spread over `N_PATHS` paths is the sampling noise of the
# statistic at this path length, and a model separates only if its own spread
# sits clear of that.

# %%
kurt_by_model = {
    name: stats_df.filter(pl.col("model") == name)["excess_kurtosis"].to_numpy()
    for name in path_populations
}
gbm_p95 = np.percentile(kurt_by_model["GBM"], 95)

print(f"GBM excess kurtosis, 5th-95th percentile over {N_PATHS} paths: ")
print(f"  [{np.percentile(kurt_by_model['GBM'], 5):.3f}, {gbm_p95:.3f}]")
print("\nShare of paths above the GBM 95th percentile:")
for name, values in kurt_by_model.items():
    share = float((values > gbm_p95).mean())
    print(f"  {name:15s} median {np.median(values):7.3f}   above GBM p95: {share:6.1%}")


def dominance(a: str, b: str) -> float:
    """Share of (a, b) path pairs in which *a* has the higher excess kurtosis."""
    return float((kurt_by_model[a][:, None] > kurt_by_model[b][None, :]).mean())


print("\nPairwise dominance, P(row path exceeds column path):")
fat_tailed = ["Jump-Diffusion", "GARCH", "Heston"]
for a in fat_tailed:
    row = "  ".join(f"{dominance(a, b):.2f}" if a != b else "  - " for b in fat_tailed)
    print(f"  {a:15s} {row}")

# %% [markdown]
# **What the comparison establishes.** Three models generate excess kurtosis and
# they order the same way in every pairwise comparison: jump-diffusion above
# GARCH above Heston. Jump-diffusion is the clearest, exceeding GBM's 95th
# percentile on nearly every path, which is what an explicit jump component is
# for. GARCH exceeds Heston on close to three quarters of path pairs, so the
# ordering between those two is real even though their ranges overlap and a
# single path from each would not have shown it.
#
# Mean-reversion is indistinguishable from GBM on kurtosis: it clears GBM's 95th
# percentile at about the rate chance alone would produce. Its visibly narrower
# price range comes from the lower volatility it was parameterized with and from
# the pull toward equilibrium, neither of which is a tail property.
#
# For risk work: GBM and mean-reversion put no more weight in the tails than a
# Gaussian, so VaR and ES computed from them understate tail loss by
# construction. The magnitudes separating the other three are large enough to
# matter — the jump model's median excess kurtosis is an order of magnitude
# above the other two — so the choice among them is not a rounding difference.

# %% [markdown]
# ---
# # Part 3: Bootstrap Methods
#
# Bootstrap methods resample historical data rather than assuming a parametric
# model. They preserve the **empirical distribution** exactly, including fat tails.
#
# | Method | Block Size | Preserves Autocorrelation | Best For |
# |--------|-----------|--------------------------|----------|
# | **IID Bootstrap** | 1 | No | i.i.d. assumption OK |
# | **Block Bootstrap** | Fixed | Yes (within blocks) | Time series |
# | **Stationary Bootstrap** | Random | Yes (smoother) | Financial returns |
#
# ### Key Trade-off
#
# - **Parametric**: Can generate scenarios *beyond* historical range
# - **Bootstrap**: Preserves empirical distribution *exactly* but limited to observed extremes
#
# ### Consistency Note
#
# We bootstrap **log-returns** to match the parametric models above.

# %% [markdown]
# ## Load Real Data for Bootstrap

# %%
# Compute log-returns (consistent with parametric models)
spy_log_returns = np.diff(np.log(spy_close))

print(f"SPY log-returns: {len(spy_log_returns)} observations")
print(f"Mean: {spy_log_returns.mean():.6f}")
print(f"Std: {spy_log_returns.std():.4f}")
print(f"Skewness: {skew(spy_log_returns):.4f}")
print(
    f"Excess kurtosis: {kurtosis(spy_log_returns, fisher=True, bias=False):.4f} (Fisher, Gaussian=0)"
)

# %% [markdown]
# ## IID Bootstrap
#
# The simplest resampling method: draw individual returns **with replacement**.
#
# ### Algorithm
#
# ```
# For each bootstrap sample of length T:
#     For t = 1 to T:
#         Draw index i uniformly from {1, ..., N}
#         Set r*_t = r_i  (original return i)
#     Return r* = (r*_1, ..., r*_T)
# ```
#
# ### Properties
#
# - **Preserves marginal distribution**: Same histogram as original
# - **Destroys autocorrelation**: Each draw is independent
# - **Fast and simple**: No tuning parameters


# %%
def iid_bootstrap(
    data: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate IID bootstrap sample.

    Parameters
    ----------
    data : np.ndarray
        Original data
    n_samples : int
        Length of bootstrap sample
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    np.ndarray
        Bootstrap sample
    """
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.choice(len(data), size=n_samples, replace=True)
    return data[indices]


# %% [markdown]
# Draw one IID bootstrap sample the same length as the original series and
# compare its moments with the original's.

# %%
iid_sample = iid_bootstrap(spy_log_returns, len(spy_log_returns), rng=stream("iid_bootstrap"))

print("IID Bootstrap vs Original (log-returns):")
print(f"  Mean: {iid_sample.mean():.6f} vs {spy_log_returns.mean():.6f}")
print(f"  Std: {iid_sample.std():.4f} vs {spy_log_returns.std():.4f}")
print(f"  Skew: {skew(iid_sample):.4f} vs {skew(spy_log_returns):.4f}")

# %% [markdown]
# ### Library Usage: IID Bootstrap

# %%
bs = IIDBootstrap(spy_log_returns, seed=SEED)
means = [data[0].mean() for data, _ in bs.bootstrap(100)]
print(f"arch IIDBootstrap (100 samples): mean of means = {np.mean(means):.6f}")

# %% [markdown]
# ## Block Bootstrap
#
# Resample **contiguous blocks** of fixed length to preserve local dependence.
#
# ### Algorithm (Moving Block Bootstrap)
#
# ```
# Choose block length b
# For each bootstrap sample of length T:
#     While sample length < T:
#         Draw start index i uniformly from {1, ..., N-b+1}
#         Append block (r_i, r_{i+1}, ..., r_{i+b-1})
#     Trim to length T
# ```
#
# ### Block Length Selection
#
# - Rule of thumb: $b \approx T^{1/3}$
# - Financial: ~22 days (one month) is common
# - Optimal: Cross-validation on out-of-sample statistics


# %%
def block_bootstrap(
    data: np.ndarray,
    block_size: int,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate moving block bootstrap sample.

    Parameters
    ----------
    data : np.ndarray
        Original data
    block_size : int
        Fixed block length
    n_samples : int
        Length of bootstrap sample
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    np.ndarray
        Bootstrap sample
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(data)
    result = []

    while len(result) < n_samples:
        start = rng.integers(0, n - block_size + 1)
        block = data[start : start + block_size]
        result.extend(block)

    return np.array(result[:n_samples])


# %% [markdown]
# Use 22-day blocks, roughly one trading month.

# %%
block_size = 22
block_sample = block_bootstrap(
    spy_log_returns, block_size, len(spy_log_returns), rng=stream("block_bootstrap")
)

print(f"Block Bootstrap (block_size={block_size}) vs Original:")
print(f"  Mean: {block_sample.mean():.6f} vs {spy_log_returns.mean():.6f}")
print(f"  Std: {block_sample.std():.4f} vs {spy_log_returns.std():.4f}")

# %% [markdown]
# ### Library Usage: Block Bootstrap

# %%
bs = MovingBlockBootstrap(block_size, spy_log_returns, seed=SEED)
means = [data[0].mean() for data, _ in bs.bootstrap(100)]
print(f"arch MovingBlockBootstrap (100 samples): mean of means = {np.mean(means):.6f}")

# %% [markdown]
# ## Stationary Bootstrap
#
# Uses **random block lengths** from a geometric distribution, eliminating
# artificial block boundaries.
#
# ### Algorithm (Politis & Romano 1994)
#
# ```
# Choose expected block length b
# For each bootstrap sample of length T:
#     Set t = 0
#     While t < T:
#         Draw start index i uniformly from {1, ..., N}
#         Draw block length L from Geometric(1/b)
#         Append (r_i, r_{i+1}, ..., r_{i+L-1}) with wrap-around
#         t = t + L
#     Trim to length T
# ```
#
# ### Why "Stationary"?
#
# With random block lengths, the bootstrap distribution is **stationary** -
# each position in the sample has the same marginal distribution.


# %%
def stationary_bootstrap(
    data: np.ndarray,
    expected_block_size: float,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate stationary bootstrap sample.

    Parameters
    ----------
    data : np.ndarray
        Original data
    expected_block_size : float
        Expected block length (geometric distribution parameter)
    n_samples : int
        Length of bootstrap sample
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    np.ndarray
        Bootstrap sample
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(data)
    p = 1.0 / expected_block_size  # Probability of ending block
    result = []

    while len(result) < n_samples:
        pos = rng.integers(0, n)

        while len(result) < n_samples:
            result.append(data[pos])
            pos = (pos + 1) % n  # Wrap around

            if rng.random() < p:
                break

    return np.array(result[:n_samples])


# %% [markdown]
# Use the same expected block length so the two block methods differ only in
# whether the length is fixed or random.

# %%
stat_sample = stationary_bootstrap(
    spy_log_returns, block_size, len(spy_log_returns), rng=stream("stationary_bootstrap")
)

print(f"Stationary Bootstrap (expected block={block_size}) vs Original:")
print(f"  Mean: {stat_sample.mean():.6f} vs {spy_log_returns.mean():.6f}")
print(f"  Std: {stat_sample.std():.4f} vs {spy_log_returns.std():.4f}")

# %% [markdown]
# ### Library Usage: Stationary Bootstrap

# %%
bs = StationaryBootstrap(block_size, spy_log_returns, seed=SEED)
means = [data[0].mean() for data, _ in bs.bootstrap(100)]
print(f"arch StationaryBootstrap (100 samples): mean of means = {np.mean(means):.6f}")

# %% [markdown]
# ## Bootstrap Method Comparison

# %%
# Collect bootstrap samples
bootstrap_samples = {
    "Original": spy_log_returns,
    "IID": iid_sample,
    "Block": block_sample,
    "Stationary": stat_sample,
}

# Compare moments
bootstrap_stats = []
for name, sample in bootstrap_samples.items():
    bootstrap_stats.append(
        {
            "method": name,
            "mean": sample.mean(),
            "std": sample.std(),
            "skew": skew(sample),
            "excess_kurtosis": kurtosis(sample, fisher=True, bias=False),
        }
    )

pl.DataFrame(bootstrap_stats)

# %% [markdown]
# ## Autocorrelation Preservation
#
# The key difference between bootstrap methods is how they handle **temporal
# dependence**. We measure this via autocorrelation of **squared returns**
# (signature of volatility clustering).

# %%
# Compute ACF for squared returns
n_lags = 20
fig = go.Figure()

bootstrap_colors = [COLORS["blue"], COLORS["amber"], COLORS["copper"], COLORS["neutral"]]

for idx, (name, sample) in enumerate(bootstrap_samples.items()):
    squared = sample**2
    acf_values = acf(squared, nlags=n_lags, fft=True)
    fig.add_trace(
        go.Scatter(
            x=list(range(n_lags + 1)),
            y=acf_values,
            mode="lines+markers",
            name=name,
            line=dict(color=bootstrap_colors[idx]),
        )
    )

fig.update_layout(
    title="IID resampling removes volatility clustering; block methods keep it",
    xaxis_title="Lag (days)",
    yaxis_title="Autocorrelation",
)
show_plotly_with_alt(
    fig,
    "Autocorrelation of squared returns against lag for the original SPY series "
    "and three bootstrap resamples; the IID resample sits at zero across all "
    "lags while the block and stationary resamples track the original.",
)

# %% [markdown]
# ### How Much Dependence Each Method Retains
#
# The ACF curves above come from one resample each, and one resample cannot rank
# two methods that differ only in whether the block length is fixed or random.
# Summarize each resample by the sum of its squared-return autocorrelations over
# the first `n_lags` lags, divided by the same sum for the original series, and
# repeat it `N_BOOTSTRAP_REPLICATES` times per method. A value of one means the
# resample carries as much volatility clustering as SPY; zero means none.

# %%
original_dependence = acf(spy_log_returns**2, nlags=n_lags, fft=True)[1:].sum()

resamplers = {
    "IID": lambda rng: iid_bootstrap(spy_log_returns, len(spy_log_returns), rng=rng),
    "Block": lambda rng: block_bootstrap(
        spy_log_returns, block_size, len(spy_log_returns), rng=rng
    ),
    "Stationary": lambda rng: stationary_bootstrap(
        spy_log_returns, block_size, len(spy_log_returns), rng=rng
    ),
}

method_seeds = dict(
    zip(resamplers, _SEED_SEQUENCES["bootstrap_replicates"].spawn(len(resamplers)), strict=True)
)

retention = {}
for method, resample in resamplers.items():
    generators = [
        np.random.default_rng(child) for child in method_seeds[method].spawn(N_BOOTSTRAP_REPLICATES)
    ]
    retention[method] = np.array(
        [
            acf(resample(rng) ** 2, nlags=n_lags, fft=True)[1:].sum() / original_dependence
            for rng in generators
        ]
    )

print(f"Dependence retained, {N_BOOTSTRAP_REPLICATES} replicates per method")
print(f"(sum of ACF of squared returns over lags 1-{n_lags}, as a share of SPY's)")
for method, values in retention.items():
    print(
        f"  {method:11s} median {np.median(values):6.3f}  "
        f"[{np.percentile(values, 5):6.3f}, {np.percentile(values, 95):6.3f}]"
    )

stationary_beats_block = float(
    (retention["Stationary"][:, None] > retention["Block"][None, :]).mean()
)
print(
    f"\nP(a stationary replicate retains more than a block replicate) = "
    f"{stationary_beats_block:.2f}"
)

# %% [markdown]
# ### Bootstrap Key Takeaways
#
# 1. **IID bootstrap** reproduces the marginal distribution — the same histogram,
#    the same fat tails — and retains essentially none of the volatility
#    clustering, which is what resampling one observation at a time implies.
# 2. **Block and stationary bootstrap** both retain roughly two thirds of SPY's
#    squared-return dependence at a 22-day block length. Neither recovers all of
#    it: dependence that spans a block boundary is destroyed whatever the block
#    length, and lengthening blocks to keep more of it leaves fewer distinct
#    blocks to resample.
# 3. **Random block lengths help, modestly.** The stationary bootstrap retains
#    more than the moving block bootstrap in the majority of paired draws, but
#    the two distributions overlap heavily, so the advantage shows up across
#    replicates and is not something a single resample would establish.
#
# For financial returns with volatility clustering, either block method is a
# reasonable default and the choice between them matters less than the block
# length.

# %% [markdown]
# ---
# # Part 4: Model Comparison Summary
#
# ## What Each Method Captures
#
# Fat tails and volatility clustering are separate properties and are measured
# separately above: excess kurtosis across `N_PATHS` paths for the parametric
# models, retained squared-return autocorrelation across
# `N_BOOTSTRAP_REPLICATES` replicates for the resamplers. "Beyond history" asks
# whether the method can emit a value larger than any it was given.
#
# | Method | Excess kurtosis above Gaussian | Volatility clustering | Beyond history |
# |--------|-------------------------------|-----------------------|----------------|
# | GBM | No, by construction | None | Yes |
# | Jump-Diffusion | Yes, and the only clear separation measured here | None; jumps are i.i.d. | Yes |
# | Mean-Reversion | No | None; the dependence is negative autocorrelation in returns, not in squared returns | Yes |
# | Heston | Yes, overlapping with GARCH | Yes, from mean-reverting variance | Yes |
# | GARCH | Yes, overlapping with Heston | Yes, from the variance recursion | Yes |
# | IID Bootstrap | Inherits the sample's | None retained | No |
# | Block Bootstrap | Inherits the sample's | About two thirds retained | No |
# | Stationary Bootstrap | Inherits the sample's | About two thirds retained, slightly more than fixed blocks | No |
#
# ## Key Takeaways
#
# 1. **Parametric models** can emit values beyond anything in the historical
#    sample, which is what makes them usable for stress scenarios, but every
#    property they exhibit is one that was chosen or fitted.
# 2. **Bootstrap methods** reproduce the empirical marginal distribution, fat
#    tails included, without any distributional assumption, and cannot produce a
#    return larger than the largest one observed.
# 3. **No classical model here captures both stylized facts at a magnitude these
#    runs can separate.** The jump model separates on tails and has no
#    clustering; Heston and GARCH have clustering and excess kurtosis but their
#    kurtosis ranges overlap each other across 200 paths.
# 4. **Drift compensation** (jump-diffusion) and **full truncation** (Heston) are
#    implementation details that change what is simulated, not stylistic choices;
#    without the compensator the requested drift is not the realized drift, and
#    without truncation the variance recursion can go negative.
# 5. **Fitting GARCH by maximum likelihood** replaces two chosen numbers with two
#    estimated ones, which is a different claim from producing more realistic
#    paths: persistence comes from the data, the level still has to be carried
#    across explicitly, as the library comparison above shows.
#
# **Next**: See [`01_timegan`](01_timegan.ipynb) for the first learned generative model, which uses
# adversarial training to capture temporal dynamics that classical models miss.
#
# **Book**: Chapter 5, Section 5.4 covers the generative model taxonomy and explains
# why learned models complement (rather than replace) classical simulation.
