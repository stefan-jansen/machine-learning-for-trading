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
# # Options Greeks: From Theory to Computation
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Derive Black-Scholes pricing and Greeks from first principles, implement implied
# volatility via root-finding, and validate the computations against the
# vendor-supplied values in the AlgoSeek options dataset.
#
# ## Learning Objectives
#
# - Implement Black-Scholes call/put pricing and verify put-call parity.
# - Solve for implied volatility numerically using Brent's method.
# - Code all five Greeks (Delta, Gamma, Vega, Theta, Rho) and visualize their
#   behavior across moneyness and time to expiration.
# - Quantify residuals between in-house and vendor-computed Greeks and explain
#   the residual sources.
#
# ## Book Reference
#
# §2.2, "The asset-class market data landscape" - the derivatives part of it.
#
# ## Prerequisites
#
# - Basic calculus (partial derivatives) and the standard normal distribution.
# - Options terminology from `07_sp500_options_eda`.
# - The AlgoSeek S&P 500 options EDA parquet at `$ML4T_DATA_PATH/equities/market/sp500/options_eda/`.

# %%
"""Options Greeks — Black-Scholes pricing, IV computation, and Greeks validation."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import brentq

from data import load_macro, load_sp500_options_eda
from utils.style import COLORS, show_plotly_with_alt

# %% [markdown]
# ### Declared parameters
#
# The validation sections draw a random sample rather than reading whichever rows happen to
# come first, so the seed is declared alongside the sample sizes. `head(n)` on this file would
# return a single trading day, and the sections below say what difference that makes.
#
# There is no fixed risk-free rate here. The one-year Treasury yield fell by more than a
# percentage point over 2020, so any single number is wrong for most of the year. The
# validation section reads a rate per date and prints the range it spans.

# %% tags=["parameters"]
N_GREEKS_VALIDATE = 2000  # Options drawn for the Greeks comparison
N_IV_VALIDATE = 200  # Options drawn for the IV recovery comparison
SAMPLE_SEED = 42

# Illustrative underlying price for the Greeks-behaviour charts in sections 2 to 4. These plot
# the formulas rather than the data, so the level is a round number chosen for readability.
DEMO_SPOT = 100.0

# A constant rate, kept only so the validation can measure what using one costs.
FLAT_RATE_FOR_COMPARISON = 0.015

# %% [markdown]
# ## 1. The Black-Scholes Framework
#
# The Black-Scholes model (1973) provides closed-form solutions for European option
# prices under specific assumptions. Understanding these assumptions is critical for
# practitioners - model limitations explain many real-world pricing phenomena.
#
# ### Model Assumptions
#
# | Assumption | Reality | Implication |
# |------------|---------|-------------|
# | Log-normal returns | Fat tails exist | Underprices tail risk |
# | Constant volatility | Vol changes over time | Need to re-estimate σ |
# | No dividends | Stocks pay dividends | Use dividend-adjusted models |
# | No transaction costs | Costs exist | Greeks less useful for small positions |
# | Continuous trading | Markets close | Weekend/overnight gaps |
# | European exercise | Many options are American | Early exercise premium missed |
# | Constant risk-free rate | Rates vary | Use term-matched rates |
#
# Despite these limitations, Black-Scholes remains the industry standard for
# quoting volatility and computing Greeks. The model's tractability outweighs
# its theoretical shortcomings for most practical applications.

# %% [markdown]
# ### The Black-Scholes Formula
#
# For a European call option:
#
# $$C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$
#
# For a European put option:
#
# $$P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)$$
#
# Where:
# - $S$ = Current stock price
# - $K$ = Strike price
# - $T$ = Time to expiration (in years)
# - $r$ = Risk-free interest rate
# - $\sigma$ = Volatility (annualized standard deviation of log returns)
# - $N(\cdot)$ = Cumulative normal distribution function
#
# And $d_1$, $d_2$ are:
#
# $$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$
#
# $$d_2 = d_1 - \sigma\sqrt{T}$$


# %%
# Small helper functions for d1 and d2 (tightly coupled, <=5 lines each)
def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 parameter for Black-Scholes formula."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 parameter: d2 = d1 - sigma * sqrt(T)."""
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


# %% [markdown]
# ### Call Pricing


# %%
def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes price for a European call option."""
    if T <= 0:
        return max(S - K, 0)  # Intrinsic value at expiration
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return S * stats.norm.cdf(d_1) - K * np.exp(-r * T) * stats.norm.cdf(d_2)


# %% [markdown]
# ### Put Pricing


# %%
def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes price for a European put option."""
    if T <= 0:
        return max(K - S, 0)  # Intrinsic value at expiration
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * stats.norm.cdf(-d_2) - S * stats.norm.cdf(-d_1)


# %% [markdown]
# ### Unified Pricing Function


# %%
def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Black-Scholes option price (call or put)."""
    if option_type.lower() == "call":
        return bs_call_price(S, K, T, r, sigma)
    else:
        return bs_put_price(S, K, T, r, sigma)


# %% [markdown]
# ### Verify Put-Call Parity
#
# A fundamental relationship that must hold for European options:
#
# $$C - P = S - K \cdot e^{-rT}$$
#
# This provides a sanity check for our implementation.

# %%
# Test parameters
S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

call_price = bs_call_price(S, K, T, r, sigma)
put_price = bs_put_price(S, K, T, r, sigma)

# Put-call parity check
lhs = call_price - put_price
rhs = S - K * np.exp(-r * T)

print("=== Black-Scholes Implementation Test ===")
print(f"Parameters: S={S}, K={K}, T={T}, r={r}, σ={sigma}")
print(f"\nCall price: ${call_price:.4f}")
print(f"Put price:  ${put_price:.4f}")
print("\nPut-Call Parity Check:")
print(f"  C - P = {lhs:.6f}")
print(f"  S - Ke^(-rT) = {rhs:.6f}")
print(f"  Difference: {abs(lhs - rhs):.2e}")

# %% [markdown]
# ## 2. Implied Volatility Computation
#
# Implied volatility (IV) is the volatility value that, when plugged into
# Black-Scholes, produces the observed market price. Since there's no closed-form
# solution, we must solve numerically:
#
# $$\text{Find } \sigma^* \text{ such that } BS(S, K, T, r, \sigma^*) = P_{market}$$
#
# We'll use Brent's method (a robust root-finding algorithm) to solve this.


# %%
def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    bounds: tuple = (0.001, 5.0),
) -> float | None:
    """Compute implied volatility using Brent's method.

    Returns the volatility that makes BS price equal market_price, or None.
    """
    if T <= 0:
        return None

    # Define objective function: BS_price(sigma) - market_price = 0
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        # Check if solution exists within bounds
        f_low = objective(bounds[0])
        f_high = objective(bounds[1])

        if f_low * f_high > 0:
            # No sign change - no solution in bounds
            return None

        # Brent's method for root finding
        iv = brentq(objective, bounds[0], bounds[1], xtol=1e-8)
        return iv

    except (ValueError, RuntimeError):
        return None


# %%
# Test IV computation
test_vol = 0.25
test_call_price = bs_call_price(S, K, T, r, test_vol)

recovered_iv = implied_volatility(test_call_price, S, K, T, r, "call")

print("=== Implied Volatility Test ===")
print(f"Original volatility: {test_vol:.4f}")
print(f"Generated call price: ${test_call_price:.4f}")
print(f"Recovered IV: {recovered_iv:.4f}")
print(f"Error: {abs(test_vol - recovered_iv):.2e}")

# %% [markdown]
# ## 3. The Greeks: Measuring Option Sensitivities
#
# Greeks measure how option prices change with respect to various inputs.
# They're essential for:
# - **Hedging**: Neutralizing unwanted exposures
# - **Risk Management**: Understanding portfolio sensitivities
# - **Trading**: Identifying mispriced options
#
# ### Summary of Greeks
#
# | Greek | Symbol | Measures | Formula |
# |-------|--------|----------|---------|
# | Delta | $\Delta$ | ∂V/∂S | Price sensitivity to underlying |
# | Gamma | $\Gamma$ | ∂²V/∂S² | Delta sensitivity to underlying |
# | Vega | $\mathcal{V}$ | ∂V/∂σ | Price sensitivity to volatility |
# | Theta | $\Theta$ | ∂V/∂t | Price sensitivity to time (decay) |
# | Rho | $\rho$ | ∂V/∂r | Price sensitivity to interest rate |

# %% [markdown]
# ### Delta ($\Delta$)
#
# Delta measures the rate of change of option price with respect to the underlying:
#
# $$\Delta_{call} = N(d_1)$$
# $$\Delta_{put} = N(d_1) - 1 = -N(-d_1)$$
#
# **Interpretation**:
# - Call delta ranges from 0 to 1
# - Put delta ranges from -1 to 0
# - At-the-money options have an absolute delta near one half
# - Delta also approximates probability of finishing ITM


# %%
def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Compute Black-Scholes delta."""
    if T <= 0:
        if option_type.lower() == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d_1 = d1(S, K, T, r, sigma)

    if option_type.lower() == "call":
        return stats.norm.cdf(d_1)
    else:
        return stats.norm.cdf(d_1) - 1


# %% [markdown]
# ### Gamma ($\Gamma$)
#
# Gamma measures the rate of change of delta (option's "acceleration"):
#
# $$\Gamma = \frac{N'(d_1)}{S \sigma \sqrt{T}}$$
#
# Where $N'(x)$ is the standard normal PDF.
#
# **Interpretation**:
# - Gamma is highest for ATM options near expiration
# - Same for calls and puts (by put-call parity)
# - High gamma = delta changes rapidly = harder to hedge


# %%
def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute Black-Scholes gamma (same for calls and puts).
    """
    if T <= 0:
        return 0.0

    d_1 = d1(S, K, T, r, sigma)
    return stats.norm.pdf(d_1) / (S * sigma * np.sqrt(T))


# %% [markdown]
# ### Vega ($\mathcal{V}$)
#
# Vega measures sensitivity to implied volatility:
#
# $$\mathcal{V} = S \sqrt{T} \cdot N'(d_1)$$
#
# **Interpretation**:
# - Usually quoted per percentage-point change in volatility (divide by one hundred)
# - Highest for ATM options with longer time to expiration
# - Same for calls and puts


# %%
def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute Black-Scholes vega.
    Returns vega per 1 point (100%) change in volatility.
    Divide by 100 for vega per 1% change.
    """
    if T <= 0:
        return 0.0

    d_1 = d1(S, K, T, r, sigma)
    return S * np.sqrt(T) * stats.norm.pdf(d_1)


# %% [markdown]
# ### Theta ($\Theta$)
#
# Theta measures time decay - how option value erodes as time passes:
#
# $$\Theta_{call} = -\frac{S \sigma N'(d_1)}{2\sqrt{T}} - rKe^{-rT}N(d_2)$$
#
# $$\Theta_{put} = -\frac{S \sigma N'(d_1)}{2\sqrt{T}} + rKe^{-rT}N(-d_2)$$
#
# **Interpretation**:
# - Usually negative (options lose value over time)
# - Accelerates as expiration approaches
# - Deep ITM puts can have positive theta


# %%
def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Compute Black-Scholes theta (per year).
    Divide by 365 for daily theta.
    """
    if T <= 0:
        return 0.0

    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)

    term1 = -S * sigma * stats.norm.pdf(d_1) / (2 * np.sqrt(T))

    if option_type.lower() == "call":
        term2 = -r * K * np.exp(-r * T) * stats.norm.cdf(d_2)
    else:
        term2 = r * K * np.exp(-r * T) * stats.norm.cdf(-d_2)

    return term1 + term2


# %% [markdown]
# ### Rho ($\rho$)
#
# Rho measures sensitivity to interest rates:
#
# $$\rho_{call} = KTe^{-rT}N(d_2)$$
# $$\rho_{put} = -KTe^{-rT}N(-d_2)$$
#
# **Interpretation**:
# - Less important for short-dated options
# - Higher rates benefit calls, hurt puts
# - Often the least-monitored Greek


# %%
def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Compute Black-Scholes rho (per 1 point change in rate).
    Divide by 100 for rho per 1% change.
    """
    if T <= 0:
        return 0.0

    d_2 = d2(S, K, T, r, sigma)

    if option_type.lower() == "call":
        return K * T * np.exp(-r * T) * stats.norm.cdf(d_2)
    else:
        return -K * T * np.exp(-r * T) * stats.norm.cdf(-d_2)


# %% [markdown]
# ### All Greeks Summary Function


# %%
def compute_all_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict:
    """Compute all Greeks for an option."""
    return {
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "vega": vega(S, K, T, r, sigma) / 100,  # Per 1% vol change
        "theta": theta(S, K, T, r, sigma, option_type) / 365,  # Daily
        "rho": rho(S, K, T, r, sigma, option_type) / 100,  # Per 1% rate change
    }


# Test the Greeks
test_greeks = compute_all_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20)

print("=== Greeks for ATM Call (S=K=100, T=0.25yr, σ=20%) ===")
for greek, value in test_greeks.items():
    print(f"{greek.capitalize():>6}: {value:>10.6f}")

# %% [markdown]
# ## 4. Greeks Visualization
#
# Understanding how Greeks behave across different strikes and times to
# expiration is crucial for option traders.

# %%
# Generate data for visualization
strikes = np.linspace(80, 120, 41)
S_0 = DEMO_SPOT
r_0 = 0.05
sigma_0 = 0.20
times = [0.25, 0.5, 1.0]  # 3mo, 6mo, 1yr

# Compute Greeks across strikes for different expirations
greeks_data = []
for T_val in times:
    for K_val in strikes:
        greeks = compute_all_greeks(S_0, K_val, T_val, r_0, sigma_0, "call")
        greeks_data.append(
            {
                "strike": K_val,
                "moneyness": S_0 / K_val,
                "time_to_exp": f"{int(T_val * 12)}M",
                "T": T_val,
                **greeks,
            }
        )

greeks_df = pl.DataFrame(greeks_data)

# %% [markdown]
# ### Delta vs Moneyness
#
# Delta transitions from 0 (deep OTM) to 1 (deep ITM), with the steepest
# slope at ATM. Shorter-dated options have sharper transitions.

# %%
fig = px.line(
    greeks_df.to_pandas(),
    x="strike",
    y="delta",
    color="time_to_exp",
    title="Call delta against strike",
    labels={"strike": "Strike Price ($)", "delta": "Delta", "time_to_exp": "Expiration"},
)
fig.add_vline(x=DEMO_SPOT, line_dash="dash", line_color=COLORS["neutral"], annotation_text="ATM")
fig.update_layout(height=400)
show_plotly_with_alt(
    fig,
    "A curve of call delta against strike price, with a dashed vertical line at the spot. It runs from near one at the lowest strikes down to near zero at the highest, falling most steeply where it crosses the spot line at around one half.",
)

# %% [markdown]
# ### Gamma Concentration Near ATM
#
# Gamma peaks at ATM and increases dramatically as expiration approaches.
# This is why short-dated ATM options are difficult to hedge.

# %%
fig = px.line(
    greeks_df.to_pandas(),
    x="strike",
    y="gamma",
    color="time_to_exp",
    title="Gamma against strike",
    labels={"strike": "Strike Price ($)", "gamma": "Gamma", "time_to_exp": "Expiration"},
)
fig.add_vline(x=DEMO_SPOT, line_dash="dash", line_color=COLORS["neutral"], annotation_text="ATM")
fig.update_layout(height=400)
show_plotly_with_alt(
    fig,
    "A curve of gamma against strike price, with a dashed vertical line at the spot. It is a single hump peaking at the spot line and falling away towards zero on both sides.",
)

# %% [markdown]
# ### Theta Decay Acceleration
#
# Theta (time decay) accelerates as expiration approaches. Options lose
# value faster in their final weeks.

# %%
fig = px.line(
    greeks_df.to_pandas(),
    x="strike",
    y="theta",
    color="time_to_exp",
    title="Daily call theta against strike, by time to expiration",
    labels={
        "strike": "Strike Price ($)",
        "theta": "Theta ($/day)",
        "time_to_exp": "Expiration",
    },
)
fig.add_vline(x=DEMO_SPOT, line_dash="dash", line_color=COLORS["neutral"], annotation_text="ATM")
fig.update_layout(height=400)
show_plotly_with_alt(
    fig,
    "Curves of daily call theta against strike price, one per time to expiration, with a dashed "
    "vertical line at the spot. All lie below zero and reach their most negative value near the "
    "spot line, the shortest-dated curve dipping furthest.",
)

# %% [markdown]
# ### Vega Term Structure
#
# Longer-dated options have higher vega - they're more sensitive to
# volatility changes. This makes sense: more time means more opportunity
# for volatility to impact the final payoff.

# %%
fig = px.line(
    greeks_df.to_pandas(),
    x="strike",
    y="vega",
    color="time_to_exp",
    title="Vega against strike, by time to expiration",
    labels={
        "strike": "Strike Price ($)",
        "vega": "Vega ($/1% vol)",
        "time_to_exp": "Expiration",
    },
)
fig.add_vline(x=DEMO_SPOT, line_dash="dash", line_color=COLORS["neutral"], annotation_text="ATM")
fig.update_layout(height=400)
show_plotly_with_alt(
    fig,
    "Curves of vega against strike price for several times to expiration, with a dashed vertical line at the spot. Each is a hump peaking at the spot line, and the humps are taller for longer times to expiration.",
)

# %% [markdown]
# ## 5. Validation Against AlgoSeek Data
#
# The Greeks above are held against the vendor's pre-computed values for the same contracts.
# Two choices decide whether that comparison means anything, and both are easy to get wrong in
# a way that flatters the result.
#
# **The rate has to move.** Black-Scholes takes a risk-free rate, and 2020 is the wrong year to
# pick one number for: the one-year Treasury yield fell by more than a percentage point over
# the twelve months, and the cell below prints where it started and finished. A constant chosen
# from the start of the sample is roughly right in January and wrong by most of that fall for
# the rest of the year. The rate is read per date from the macro panel instead, and the cost of
# using a constant is measured rather than assumed.
#
# The panel's shortest maturity is one year, and these options have twenty to ninety days to
# run, so this is the nearest available rate rather than a term-matched one. That is a real
# limitation and it is named here rather than papered over.
#
# **The sample has to be a sample.** The filtered frame is sorted, so its first rows are all
# from the earliest dates, so reading the head of it validates the model against a handful of
# trading days while reporting the answer as though it covered the year. The sample below is drawn at
# random, and the cell prints how many days each choice actually reaches.

# %%
rates = (
    load_macro()
    .select(pl.col("timestamp").cast(pl.Date).alias("date"), pl.col("dgs1"))
    .drop_nulls()
    .with_columns((pl.col("dgs1") / 100).alias("risk_free_rate"))
    .select("date", "risk_free_rate")
    .sort("date")
)
_span = rates.filter(pl.col("date").dt.year() == 2020)
print(
    f"One-year Treasury yield in 2020: {_span['risk_free_rate'].min():.3%} to "
    f"{_span['risk_free_rate'].max():.3%}"
)
print(
    f"  first five trading days {_span.head(5)['risk_free_rate'].mean():.3%}, "
    f"last five {_span.tail(5)['risk_free_rate'].mean():.3%}"
)

# %%
options = load_sp500_options_eda(
    symbols=["AAPL"],
    start_date="2020-01-01",
    end_date="2020-12-31",
)

print(f"Loaded {len(options):,} option records")
print(f"Columns: {options.columns}")

# %% [markdown]
# The filter keeps contracts with a month or two to run and a solved implied volatility, which
# is the population the vendor's Greeks are meaningful for.

# %%
options_filtered = (
    options.filter(
        (pl.col("days_to_maturity").is_between(20, 90))
        & (pl.col("implied_vol").is_not_null())
        & (pl.col("implied_vol") > 0.05)
        & (pl.col("implied_vol") < 2.0)
        & (pl.col("delta").is_not_null())
    )
    .with_columns(pl.col("timestamp").cast(pl.Date).alias("date"))
    .join(rates, on="date", how="left")
    .with_columns(pl.col("risk_free_rate").forward_fill())
)

print(f"Filtered to {len(options_filtered):,} options")
print(
    f"  covering {options_filtered['date'].n_unique()} trading days and "
    f"{options_filtered['expiration'].n_unique()} expirations"
)

_head = options_filtered.head(N_GREEKS_VALIDATE)
print(
    f"The first {N_GREEKS_VALIDATE:,} rows cover {_head['date'].n_unique()} trading day(s): "
    f"{_head['date'].min()} to {_head['date'].max()}"
)

greeks_sample = options_filtered.sample(N_GREEKS_VALIDATE, seed=SAMPLE_SEED)
print(f"A random sample of the same size covers {greeks_sample['date'].n_unique()} trading days")
options_filtered.head(5)

# %%
# Compute our Greeks for each option
validation_results = []

for row in greeks_sample.iter_rows(named=True):
    S = row["underlying_price"]
    K = row["strike"]
    T = row["years_to_maturity"]
    sigma = row["implied_vol"]
    r = row["risk_free_rate"]
    opt_type = "call" if row["call_put"] == "C" else "put"

    our_delta = delta(S, K, T, r, sigma, opt_type)
    our_gamma = gamma(S, K, T, r, sigma)
    our_vega = vega(S, K, T, r, sigma) / 100
    our_theta = theta(S, K, T, r, sigma, opt_type) / 365
    flat_delta = delta(S, K, T, FLAT_RATE_FOR_COMPARISON, sigma, opt_type)

    # AlgoSeek values
    algoseek_delta = row["delta"]
    algoseek_gamma = row["gamma"]
    algoseek_vega = row["vega"]
    algoseek_theta = row["theta"]

    validation_results.append(
        {
            "symbol": row["symbol"],
            "date": row["date"],
            "strike": K,
            "spot": S,
            "moneyness": K / S,
            "days_to_exp": row["days_to_maturity"],
            "option_type": opt_type,
            "iv": sigma,
            "flat_delta": flat_delta,
            "our_delta": our_delta,
            "algoseek_delta": algoseek_delta,
            "our_gamma": our_gamma,
            "algoseek_gamma": algoseek_gamma,
            "our_vega": our_vega,
            "algoseek_vega": algoseek_vega,
            "our_theta": our_theta,
            "algoseek_theta": algoseek_theta,
        }
    )

validation_df = pl.DataFrame(validation_results)

# %%
# Calculate validation errors
validation_df = validation_df.with_columns(
    delta_error=(pl.col("our_delta") - pl.col("algoseek_delta")).abs(),
    gamma_error=(pl.col("our_gamma") - pl.col("algoseek_gamma")).abs(),
    vega_error=(pl.col("our_vega") - pl.col("algoseek_vega")).abs(),
    theta_error=(pl.col("our_theta") - pl.col("algoseek_theta")).abs(),
)

# Summary statistics
print("=== Greeks Validation Summary ===")
print(f"Options validated: {len(validation_df)}")
print()

for greek in ["delta", "gamma", "vega", "theta"]:
    error_col = f"{greek}_error"
    stats_row = validation_df.select(
        pl.col(error_col).mean().alias("mean"),
        pl.col(error_col).median().alias("median"),
        pl.col(error_col).max().alias("max"),
    )
    print(f"{greek.capitalize()}:")
    print(f"  Mean error: {stats_row['mean'][0]:.6f}")
    print(f"  Median error: {stats_row['median'][0]:.6f}")
    print(f"  Max error: {stats_row['max'][0]:.6f}")
    print()

# %% [markdown]
# ### Validation Scatter Plots

# %% [markdown]
# All four panels are built in one cell. Splitting figure construction across cells lets
# papermill flush the inline backend mid-render, capturing the delta and gamma panels as a
# figure of their own and leaving vega and theta empty in the published one.

# %%
fig = make_subplots(rows=2, cols=2, subplot_titles=["Delta", "Gamma", "Vega", "Theta"])


# Helper: draw scatter + reference 45° line into one panel.
def _add_validation_panel(row: int, col: int, x_col: str, y_col: str, name: str) -> None:
    x_vals = validation_df[x_col].to_list()
    y_vals = validation_df[y_col].to_list()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(size=4, opacity=0.5),
            name=name,
        ),
        row=row,
        col=col,
    )
    lo = min(min(x_vals), min(y_vals))
    hi = max(max(x_vals), max(y_vals))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(dash="dash", color=COLORS["negative"]),
            showlegend=False,
        ),
        row=row,
        col=col,
    )


_add_validation_panel(1, 1, "algoseek_delta", "our_delta", "Delta")
_add_validation_panel(1, 2, "algoseek_gamma", "our_gamma", "Gamma")
_add_validation_panel(2, 1, "algoseek_vega", "our_vega", "Vega")
_add_validation_panel(2, 2, "algoseek_theta", "our_theta", "Theta")

fig.update_layout(
    height=600,
    title_text="Our Greeks against the vendor's, with the identity line",
    showlegend=False,
)
fig.update_xaxes(title_text="AlgoSeek", row=2, col=1)
fig.update_xaxes(title_text="AlgoSeek", row=2, col=2)
fig.update_yaxes(title_text="Our Calculation", row=1, col=1)
fig.update_yaxes(title_text="Our Calculation", row=2, col=1)

show_plotly_with_alt(
    fig,
    "Four scatter panels, one each for delta, gamma, vega and theta, plotting our computed value against the vendor's with a dashed identity line through each. In every panel the points lie along that line closely enough to obscure it.",
)

# %% [markdown]
# ### Sources of Discrepancy
#
# Several things could produce a gap between two Black-Scholes implementations of the same
# contract: the rate, dividends, the exercise style, the root-finder, and the moment at which
# each side stamped the underlying price. Listing them costs nothing and settles nothing. Two
# of them make predictions this data can test.
#
# **The rate.** If the constant rate matters, replacing it with a per-date one should shrink
# the residual. Both were computed for every option in the sample, so this is a direct
# comparison rather than an argument.

# %%
_flat_err = (validation_df["flat_delta"] - validation_df["algoseek_delta"]).abs()
_dated_err = validation_df["delta_error"]
print(
    f"Delta residual with a flat {FLAT_RATE_FOR_COMPARISON:.1%} rate: "
    f"mean {_flat_err.mean():.5f}, max {_flat_err.max():.4f}"
)
print(
    f"Delta residual with the per-date rate:      "
    f"mean {_dated_err.mean():.5f}, max {_dated_err.max():.4f}"
)
print(f"Reduction in the mean: {1 - _dated_err.mean() / _flat_err.mean():.0%}")

# %% [markdown]
# The rate is a large part of it, and a flat rate was inflating the residual by a substantial
# fraction. That is worth dwelling on, because the flat rate would have looked fine: validated
# against the first rows of the file, which are all from early January, a rate chosen for early
# January is very nearly correct. Two shortcuts that conceal each other is the ordinary way an
# error of this kind stays in place.
#
# **The exercise style.** Every contract in this file is American - the `option_style` column
# says so - and the formulas here are European. Early exercise is worth most on deep
# in-the-money puts, where the holder can take the strike now rather than wait for it. So if
# the exercise style drives the residual, deep in-the-money puts should be the worst rows.

# %%
print(f"option_style values in the file: {options['option_style'].unique().to_list()}")
_by_side = (
    validation_df.with_columns(
        pl.when(pl.col("moneyness") < 0.9)
        .then(pl.lit("K below 0.9 S"))
        .when(pl.col("moneyness") <= 1.1)
        .then(pl.lit("near the money"))
        .otherwise(pl.lit("K above 1.1 S"))
        .alias("bucket")
    )
    .group_by("option_type", "bucket")
    .agg(pl.col("delta_error").mean().alias("mean_error"), pl.len().alias("n"))
    .sort("option_type", "bucket")
)
_by_side

# %% [markdown]
# The prediction does not come through: the residual is largest **near the money** on both
# sides, and deep in-the-money puts carry the smallest of the six groups.
#
# **That does not clear the exercise style, and it is worth being precise about why.** Early
# exercise shows up first in an option's *price*. Its effect on delta deep in the money is
# muted, because an American and a European put that far in the money both have deltas pinned
# near minus one. So this test is looking for an exercise-style effect in the one place where
# delta is least able to reveal it, and a small residual there is weak evidence. What the table
# does establish is where the residual concentrates, which is a fact about the data whatever
# causes it.
#
# Largest near the money is where gamma is highest, and gamma is what turns a difference in
# inputs into a difference in delta. That makes an input difference a plausible contributor and
# gives a way to express the residual in units a reader can judge: divide it by gamma to get
# the change in the underlying price that would produce the same delta gap.

# %%
_implied = validation_df.filter(pl.col("our_gamma") > 0).with_columns(
    (pl.col("delta_error") / pl.col("our_gamma")).alias("implied_spot_gap")
)
print(
    f"Correlation of the delta residual with gamma: "
    f"{validation_df.select(pl.corr('delta_error', 'our_gamma'))[0, 0]:.2f}"
)
print(
    f"Spot difference that would explain the residual: median "
    f"${_implied['implied_spot_gap'].median():.2f} on a median underlying of "
    f"${_implied['spot'].median():.2f}"
)
print(
    f"  as a fraction of the underlying: "
    f"{(_implied['implied_spot_gap'] / _implied['spot']).median():.2%}"
)

# %% [markdown]
# A fraction of a percent of the underlying price would account for what is left. That is a
# re-expression of the residual, not an attribution of it: a spot difference of that size is
# *consistent* with what we see, and so are several other things, including the exercise style
# the previous test could not rule out. What the figure does is put the residual on a scale
# where its size can be judged, and on that scale it is small.
#
# Where the section ends up is one source measured and settled, and a remainder that is bounded
# but not explained. The rate was ours to get wrong and we got it wrong; fixing it accounts for
# a large share. Attributing the rest would need a comparison that holds the pricing model and
# every input fixed and varies one thing at a time, which this file cannot support because it
# does not record what the vendor used. Saying so is the honest end of the analysis, and it is
# a better place to stop than a list of five candidates presented as though naming them were
# the same as testing them.

# %% [markdown]
# ## 6. Computing IV from Market Prices
#
# Let's demonstrate computing implied volatility from the observed option
# prices and compare to the provided IV values.

# %% [markdown]
# One filter has to be added before the comparison means anything. `options_filtered` keeps
# every row with a usable implied volatility, and notebook 07 showed that only about seventy
# percent of this file's volatilities were solved from the quote - the rest are extrapolated,
# interpolated, or taken from the other side of the put-call pair. Comparing our solve against
# a number the vendor did not solve measures the vendor's fallback rule, not either solver.

# %%
iv_candidates = options_filtered.filter(pl.col("iv_convergence") == "Converged")
print(
    f"Rows with a vendor volatility solved from the quote: {len(iv_candidates):,} of "
    f"{len(options_filtered):,}"
)
iv_sample = iv_candidates.sample(N_IV_VALIDATE, seed=SAMPLE_SEED)
iv_validation = []

for row in iv_sample.iter_rows(named=True):
    S = row["underlying_price"]
    K = row["strike"]
    T = row["years_to_maturity"]
    market_price = row["mid_price"]
    opt_type = "call" if row["call_put"] == "C" else "put"

    intrinsic = max(0.0, S - K) if opt_type == "call" else max(0.0, K - S)

    if market_price is not None and market_price > 0:
        computed_iv = implied_volatility(market_price, S, K, T, row["risk_free_rate"], opt_type)

        iv_validation.append(
            {
                "symbol": row["symbol"],
                "strike": K,
                "days_to_exp": row["days_to_maturity"],
                "option_type": opt_type,
                "market_price": market_price,
                "time_value": market_price - intrinsic,
                "computed_iv": computed_iv,
                "algoseek_iv": row["implied_vol"],
            }
        )

iv_df = pl.DataFrame(iv_validation)
iv_solved = iv_df.drop_nulls("computed_iv")
print(f"Options attempted: {len(iv_df)}")
print(f"  solver returned a volatility: {len(iv_solved)}")
print(f"  solver failed:                {len(iv_df) - len(iv_solved)}")

# %% [markdown]
# The failures are counted rather than dropped. A solver that silently discards the cases it
# cannot handle and then averages what is left reports the accuracy of the easy subset.

# %%
iv_df = iv_solved.with_columns(
    iv_diff=(pl.col("computed_iv") - pl.col("algoseek_iv")).abs(),
)

print(f"Absolute difference from the vendor's implied volatility, over {len(iv_df)} options:")
print(f"  median {iv_df['iv_diff'].median():.4f}")
print(f"  mean   {iv_df['iv_diff'].mean():.4f}")
print(f"  90th percentile {iv_df['iv_diff'].quantile(0.9):.4f}")
print(
    f"  within 0.01 of the vendor: {iv_df.filter(pl.col('iv_diff') < 0.01).height} of {len(iv_df)}"
)

# %% [markdown]
# Every option solves, and the agreement is close: the median disagreement is a fifth of a
# volatility point and most rows are within one point of the vendor.
#
# That is worth pausing on, because the same comparison run without the `iv_convergence` filter
# looks very different - solver failures, and a mean many times the median. None of that was
# about the solvers. It was the comparison reaching for vendor volatilities that had been
# extrapolated rather than solved, and then treating the difference as a disagreement between
# two implementations of the same calculation.
#
# The next table splits what remains by time value, and finds no such split.

# %%
iv_by_time_value = (
    iv_df.with_columns(
        pl.when(pl.col("time_value") < 0.05)
        .then(pl.lit("under 5 cents"))
        .when(pl.col("time_value") < 1.0)
        .then(pl.lit("5 cents to a dollar"))
        .otherwise(pl.lit("over a dollar"))
        .alias("time_value_bucket")
    )
    .group_by("time_value_bucket")
    .agg(
        pl.len().alias("n"),
        pl.col("iv_diff").median().alias("median_diff"),
        pl.col("iv_diff").max().alias("max_diff"),
    )
    .sort("median_diff")
)
iv_by_time_value

# %% [markdown]
# The buckets agree with each other, and one bucket is missing: among the rows the vendor
# solved directly there are almost no options with under five cents of time value.
#
# That absence is the finding, and it points at why the unfiltered comparison behaved so badly.
# Implied volatility is recovered by inverting price for volatility, and vega is the
# sensitivity of price to volatility. Where vega is near zero, a change in price too small to
# quote moves the implied volatility a long way. This is not a claim about solvers - given the
# same price, two converged solvers find the same root. It is a claim about the price: a quote
# is known only to the nearest tick, and where vega is small that tick spans a wide range of
# volatilities.
#
# The next cell measures the span directly. Each option is re-solved from a price half a tick
# above and half a tick below its mid, and the gap between the two is how much volatility the
# quote alone leaves undetermined. It is drawn from the unfiltered frame, because the thin
# options are the subject and the vendor's filter has removed them.

# %%
TICK = 0.01


def with_time_value(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the part of the price that is not intrinsic value."""
    intrinsic = (
        pl.when(pl.col("call_put") == "C")
        .then(pl.max_horizontal(pl.lit(0.0), pl.col("underlying_price") - pl.col("strike")))
        .otherwise(pl.max_horizontal(pl.lit(0.0), pl.col("strike") - pl.col("underlying_price")))
    )
    return frame.with_columns((pl.col("mid_price") - intrinsic).alias("time_value"))


def iv_interval_width(row: dict) -> float | None:
    """Volatility spanned by moving the price half a tick either way."""
    args = (row["strike"], row["years_to_maturity"], row["risk_free_rate"])
    opt = "call" if row["call_put"] == "C" else "put"
    lo = implied_volatility(row["mid_price"] - TICK / 2, row["underlying_price"], *args, opt)
    hi = implied_volatility(row["mid_price"] + TICK / 2, row["underlying_price"], *args, opt)
    return None if lo is None or hi is None else abs(hi - lo)


_tick_sample = with_time_value(options_filtered).sample(N_IV_VALIDATE, seed=SAMPLE_SEED)
tick_sensitivity = _tick_sample.select("mid_price", "time_value").with_columns(
    pl.Series("tick_iv_width", [iv_interval_width(r) for r in _tick_sample.iter_rows(named=True)])
)

tick_by_time_value = (
    tick_sensitivity.drop_nulls("tick_iv_width")
    .with_columns(
        pl.when(pl.col("time_value") < 0.05)
        .then(pl.lit("under 5 cents"))
        .when(pl.col("time_value") < 1.0)
        .then(pl.lit("5 cents to a dollar"))
        .otherwise(pl.lit("over a dollar"))
        .alias("time_value_bucket")
    )
    .group_by("time_value_bucket")
    .agg(
        pl.len().alias("n"),
        pl.col("tick_iv_width").median().alias("median_iv_span_of_one_tick"),
    )
    .sort("median_iv_span_of_one_tick")
)
print("Volatility spanned by one tick of price uncertainty:")
tick_by_time_value

# %% [markdown]
# One tick is worth orders of magnitude more volatility on the thin options than on the rest.
# On those rows the quote simply does not determine a volatility to any useful precision, and
# two implementations disagreeing there are not giving different answers to the same question -
# they are answering one the price leaves open.
#
# The vendor reached the same conclusion first, and the next cell shows it did.

# %%
_by_code = (
    with_time_value(options_filtered)
    .with_columns((pl.col("iv_convergence") == "Converged").alias("solved_from_quote"))
    .group_by("solved_from_quote")
    .agg(
        pl.len().alias("rows"),
        (pl.col("time_value") < 0.05).mean().alias("share_under_5_cents"),
    )
    .sort("solved_from_quote")
)
_by_code

# %% [markdown]
# Options with almost no time value are many times rarer among the rows the vendor solved
# directly than among the rest. The `iv_convergence` code and the time-value threshold are two
# ways of naming the same set of contracts: the ones whose price does not pin down a
# volatility. The vendor did not fail to solve those quotes so much as decline to pretend it
# had.
#
# What follows for practice is a filter, not a fix. An implied volatility recovered from a
# near-worthless option is not a measurement and should not become a feature. Filtering on
# `iv_convergence`, as this section does and as `07_sp500_options_eda` recommends, is what
# applying that judgement looks like - and it is why the comparison above was restricted to
# solved rows before any of it was measured.

# %%
# Scatter plot
fig = px.scatter(
    iv_df.to_pandas(),
    x="algoseek_iv",
    y="computed_iv",
    color="option_type",
    title="Recovered implied volatility against the vendor's",
    labels={
        "algoseek_iv": "AlgoSeek IV",
        "computed_iv": "Our Computed IV",
        "option_type": "Type",
    },
    opacity=0.6,
)
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color=COLORS["neutral"]),
        name="Perfect Match",
    )
)
fig.update_layout(height=500)
show_plotly_with_alt(
    fig,
    "A scatter of the implied volatility recovered by our root-finder against the vendor's, with a dashed identity line. The points lie tightly along the line.",
)

# %% [markdown]
# ## 7. Practical Considerations
#
# ### When Black-Scholes Breaks Down
#
# | Scenario | Problem | Alternative Approach |
# |----------|---------|---------------------|
# | Deep OTM options | Log-normal underprices tails | Use jump-diffusion or local vol |
# | Short-dated ATM | Gamma explosion | Use realized vol, not IV |
# | Dividend stocks | Wrong forward price | Use dividend-adjusted models |
# | American options | Early exercise value | Use binomial tree or approximations |
# | Vol clustering | Constant vol assumption | Use GARCH or stochastic vol |
#
# ### The Greeks in Practice
#
# **For Hedging:**
# - Delta-hedge by trading underlying shares
# - Gamma tells you how often to rebalance
# - Vega exposure from vol moves often dominates
#
# **For Trading:**
# - Greeks help identify relative value
# - High gamma + low premium = potential mispricing
# - Theta/vega ratio useful for vol trades
#
# **For Risk Management:**
# - Aggregate portfolio Greeks
# - Stress test under extreme scenarios
# - Greeks are local approximations - large moves need full repricing

# %% [markdown]
# ## 8. Key Takeaways
#
# 1. **The implementation is exact where it can be checked against itself.** Put-call parity
#    holds to numerical precision, and the Brent solver recovers a volatility it was given back
#    to the same order. Those checks establish that the code implements the formulas; they
#    establish nothing at all about whether the formulas describe these contracts.
#
# 2. **A constant risk-free rate is wrong for 2020, and the notebook measures what it costs.**
#    The one-year Treasury yield fell by more than a percentage point across the year, and the
#    range is printed in the validation section rather than repeated here. Reading a
#    rate per date instead of fixing one substantially reduces the delta residual against the
#    vendor, and the size of that reduction is printed above rather than quoted here.
#
# 3. **Two shortcuts were concealing each other.** The rate was chosen to suit early January,
#    and the validation read the first rows of the file - which are all from 2 January. A
#    constant rate validated on the one day it happens to fit looks like a good constant. The
#    sample is now drawn at random across the year, and the notebook prints how many days the
#    head of the file actually covers so the reader can see the trap rather than be told about
#    it.
#
# 4. **The exercise-style test comes out negative and cannot be taken as clearing it.** Every
#    contract here is American while the formulas are European. Early exercise is worth most on
#    deep in-the-money puts, so those rows should carry the largest residual and they carry the
#    smallest. But early exercise moves an option's price, and its effect on delta is muted
#    exactly there, where American and European deltas are both pinned near the bound. A test
#    aimed where its target is least visible is weak evidence either way, and the notebook says
#    so rather than banking the result.
#
# 5. **The remainder is bounded, not explained.** The residual concentrates where gamma is
#    highest, which is where delta is most sensitive to its inputs. Dividing by gamma expresses
#    it as the spot difference that would produce the same gap - a fraction of a percent of the
#    underlying. That puts the disagreement on a judgeable scale and shows it is small; it does
#    not identify a cause. Attribution would need a comparison holding the model and every
#    input fixed, and this file does not record what the vendor used.
#
# 6. **A validation is only as good as the population it compares against.** The implied
#    volatility check originally ran against every row with a usable volatility, and roughly a
#    third of those were extrapolated by the vendor rather than solved from the quote. That
#    produced solver failures and a mean many times the median, all of it a comparison against
#    numbers nobody had solved. Filtering on `iv_convergence` first leaves close agreement on
#    every row.
#
# 7. **The vendor's convergence code is a time-value filter under another name.** Options with
#    almost no time value are two orders of magnitude rarer among the rows solved directly than
#    among the rest, and one tick of price uncertainty spans tens of times more volatility on
#    those options than on the rest. Their prices do not determine a volatility, and the code
#    is the vendor declining to pretend otherwise.
#
# 8. **Greeks are local sensitivities.** The charts show delta steepening and gamma peaking near
#    the money as expiration approaches, which is also the regime where the Black-Scholes
#    assumptions are under the most strain.
#

# ## Next Steps
#
# - Chapter 8: Build features from options data (IV signals, skew measures).
# - Chapter 9: Evaluate IV-based signals for equity prediction.
# - Chapter 12+: ML models using options-derived features.
# - Chapter 16: Strategy backtests including the `sp500_options` short-straddle
#   case study and the `sp500_equity_option_analytics` IV-features case study.
