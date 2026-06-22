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
# Chapter 2 §2.2 (asset-class market data landscape — derivatives).
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

from data import load_sp500_options_eda

# %% tags=["parameters"]
# Production defaults
RISK_FREE_RATE = 0.015  # 3-month Treasury rate, ~typical pre-COVID 2020
N_GREEKS_VALIDATE = 500  # Sample size for Greeks comparison
N_IV_VALIDATE = 200  # Sample size for IV recovery comparison

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
# - ATM options have |Δ| ≈ 0.5
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
# - Usually quoted per 1% change in volatility (divide by 100)
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
S_0 = 100
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
    title="Call Delta vs Strike Price",
    labels={"strike": "Strike Price ($)", "delta": "Delta", "time_to_exp": "Expiration"},
)
fig.add_vline(x=100, line_dash="dash", line_color="gray", annotation_text="ATM")
fig.update_layout(height=400)
fig.show()

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
    title="Gamma vs Strike Price",
    labels={"strike": "Strike Price ($)", "gamma": "Gamma", "time_to_exp": "Expiration"},
)
fig.add_vline(x=100, line_dash="dash", line_color="gray", annotation_text="ATM")
fig.update_layout(height=400)
fig.show()

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
    title="Daily Theta vs Strike Price",
    labels={
        "strike": "Strike Price ($)",
        "theta": "Theta ($/day)",
        "time_to_exp": "Expiration",
    },
)
fig.add_vline(x=100, line_dash="dash", line_color="gray", annotation_text="ATM")
fig.update_layout(height=400)
fig.show()

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
    title="Vega vs Strike Price",
    labels={
        "strike": "Strike Price ($)",
        "vega": "Vega ($/1% vol)",
        "time_to_exp": "Expiration",
    },
)
fig.add_vline(x=100, line_dash="dash", line_color="gray", annotation_text="ATM")
fig.update_layout(height=400)
fig.show()

# %% [markdown]
# ## 5. Validation Against AlgoSeek Data
#
# Now we validate our Greeks computations against the pre-computed values
# in the AlgoSeek options dataset. We'll use a risk-free rate from FRED.

# %%
options = load_sp500_options_eda(
    symbols=["AAPL"],
    start_date="2020-01-01",
    end_date="2020-12-31",
)

print(f"Loaded {len(options):,} option records")
print(f"Columns: {options.columns}")

# %%
# Filter to liquid options for validation
# ATM options with reasonable time to expiration
options_filtered = options.filter(
    (pl.col("days_to_maturity").is_between(20, 90))
    & (pl.col("implied_vol").is_not_null())
    & (pl.col("implied_vol") > 0.05)
    & (pl.col("implied_vol") < 2.0)
    & (pl.col("delta").is_not_null())
)

print(f"Filtered to {len(options_filtered):,} options")
options_filtered.head(5)

# %%
# Compute our Greeks for each option
validation_results = []

for row in options_filtered.head(N_GREEKS_VALIDATE).iter_rows(named=True):
    S = row["underlying_price"]
    K = row["strike"]
    T = row["years_to_maturity"]
    sigma = row["implied_vol"]
    opt_type = "call" if row["call_put"] == "C" else "put"

    # Our computed values
    our_delta = delta(S, K, T, RISK_FREE_RATE, sigma, opt_type)
    our_gamma = gamma(S, K, T, RISK_FREE_RATE, sigma)
    our_vega = vega(S, K, T, RISK_FREE_RATE, sigma) / 100
    our_theta = theta(S, K, T, RISK_FREE_RATE, sigma, opt_type) / 365

    # AlgoSeek values
    algoseek_delta = row["delta"]
    algoseek_gamma = row["gamma"]
    algoseek_vega = row["vega"]
    algoseek_theta = row["theta"]

    validation_results.append(
        {
            "symbol": row["symbol"],
            "strike": K,
            "days_to_exp": row["days_to_maturity"],
            "option_type": opt_type,
            "iv": sigma,
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

# %%
# Build all four panels in a single cell — splitting figure construction across
# cells lets papermill flush the inline backend mid-render and capture the
# Delta+Gamma intermediate, leaving Vega/Theta empty in the published figure.
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
            line=dict(dash="dash", color="red"),
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
    title_text="Our Greeks vs AlgoSeek (perfect = diagonal line)",
    showlegend=False,
)
fig.update_xaxes(title_text="AlgoSeek", row=2, col=1)
fig.update_xaxes(title_text="AlgoSeek", row=2, col=2)
fig.update_yaxes(title_text="Our Calculation", row=1, col=1)
fig.update_yaxes(title_text="Our Calculation", row=2, col=1)

fig.show()

# %% [markdown]
# ### Sources of Discrepancy
#
# Small differences between our calculations and AlgoSeek's pre-computed values
# can arise from:
#
# 1. **Risk-free rate**: We used a constant rate; they may use term-matched rates
# 2. **Dividend handling**: We ignored dividends; they may adjust for them
# 3. **American vs European**: Our formulas assume European exercise
# 4. **Numerical precision**: Different root-finding algorithms
# 5. **Timestamp differences**: Greeks change throughout the day

# %% [markdown]
# ## 6. Computing IV from Market Prices
#
# Let's demonstrate computing implied volatility from the observed option
# prices and compare to the provided IV values.

# %%
# Compute IV for a sample of options
iv_validation = []

for row in options_filtered.head(N_IV_VALIDATE).iter_rows(named=True):
    S = row["underlying_price"]
    K = row["strike"]
    T = row["years_to_maturity"]
    market_price = row["mid_price"]
    opt_type = "call" if row["call_put"] == "C" else "put"

    if market_price is not None and market_price > 0:
        computed_iv = implied_volatility(market_price, S, K, T, RISK_FREE_RATE, opt_type)

        if computed_iv is not None:
            iv_validation.append(
                {
                    "symbol": row["symbol"],
                    "strike": K,
                    "days_to_exp": row["days_to_maturity"],
                    "option_type": opt_type,
                    "market_price": market_price,
                    "computed_iv": computed_iv,
                    "algoseek_iv": row["implied_vol"],
                }
            )

iv_df = pl.DataFrame(iv_validation)

# %%
# Compare IVs
iv_df = iv_df.with_columns(
    iv_diff=(pl.col("computed_iv") - pl.col("algoseek_iv")).abs(),
    iv_pct_diff=((pl.col("computed_iv") - pl.col("algoseek_iv")).abs() / pl.col("algoseek_iv"))
    * 100,
)

print("=== IV Validation Summary ===")
print(f"Options validated: {len(iv_df)}")
print(f"Mean IV difference: {iv_df['iv_diff'].mean():.4f}")
print(f"Median IV difference: {iv_df['iv_diff'].median():.4f}")
print(f"Mean % difference: {iv_df['iv_pct_diff'].mean():.2f}%")

# %%
# Scatter plot
fig = px.scatter(
    iv_df.to_pandas(),
    x="algoseek_iv",
    y="computed_iv",
    color="option_type",
    title="Computed IV vs AlgoSeek IV",
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
        line=dict(dash="dash", color="gray"),
        name="Perfect Match",
    )
)
fig.update_layout(height=500)
fig.show()

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
# 1. Black-Scholes pricing recovers put-call parity to numerical precision
#    (~7e-15 on the test parameters); the same code base solves IV via Brent's
#    method to ~10⁻¹¹ accuracy on a closed-loop test.
# 2. Greeks computed in-house track the AlgoSeek vendor values closely on
#    AAPL options (20–90 DTE, IV ∈ [0.05, 2]): mean absolute errors are
#    ~2×10⁻³ for delta and vega, ~7×10⁻⁵ for gamma, ~1×10⁻³ for daily theta.
# 3. The remaining residual reflects modeling choices that the vendor handles
#    differently — term-matched risk-free rates, dividend treatment, American
#    early-exercise premium, and intraday timestamp drift in the inputs.
# 4. Cross-sectional IV recovery (≈180 options) yields a median absolute IV
#    diff of ~5×10⁻⁴ and a mean of ~0.024, with the residual concentrated in
#    deep OTM contracts where the loss surface is shallow.
# 5. Greeks are local sensitivities — the visualizations show how Δ steepens
#    and Γ peaks near ATM as expiration approaches, which is also the regime
#    where the Black-Scholes assumption set is most stressed.
#
# ## Next Steps
#
# - Chapter 8: Build features from options data (IV signals, skew measures).
# - Chapter 9: Evaluate IV-based signals for equity prediction.
# - Chapter 12+: ML models using options-derived features.
# - Chapter 16: Strategy backtests including the `sp500_options` short-straddle
#   case study and the `sp500_equity_option_analytics` IV-features case study.
