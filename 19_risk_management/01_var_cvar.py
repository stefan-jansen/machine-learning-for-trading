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
# # Value at Risk and Conditional Value at Risk
# **Docker image**: `ml4t`
#
# ## Purpose
# **Value at Risk** answers one question: over the next day, how much could a position lose if the
# day turns out badly but not catastrophically? Pick a confidence level - say the worst one day in
# twenty - and VaR is the loss that is not exceeded on the other nineteen. It is a quantile of the
# loss distribution and nothing more.
#
# That is also its weakness, and **conditional value at risk** is the answer to it. VaR says how far
# down the threshold sits; it says nothing about what happens past it. CVaR, also called expected
# shortfall, is the average loss *given* that the threshold was breached, so it reads the whole tail
# rather than one point on it.
#
# This notebook computes both four different ways on nineteen years of real ETF returns, tests
# whether the numbers hold up against what actually happened, and then asks two questions a
# published VaR figure usually leaves unanswered: how much of it a diversified portfolio removes,
# and how much worse it gets in the market states where it matters.
#
# ## Learning Objectives
# After completing this notebook, you will be able to:
# - Estimate a loss quantile four ways - from the observed returns, from a fitted normal, from a
#   normal corrected for the shape of the observed distribution, and by simulation - and say what
#   each one assumes
# - Test whether a risk estimate was right, by counting how often losses exceeded it and asking
#   whether that count is consistent with the confidence level it claimed
# - Measure how much of a portfolio's tail risk diversification removes, and why the answer is an
#   observation for VaR and a guarantee for CVaR
# - Condition a risk estimate on the market state, using a label that was already known when the
#   return it classifies arrived
# - Score competing volatility forecasts with a loss function that penalizes under-prediction more
#   than over-prediction, which is the asymmetry risk management actually faces
#
# ## Book reference
# - Section 19.3 - Measuring the Tail: VaR and CVaR
# - Section 19.4 - Drawdowns, Path Risk, and Time-to-Recovery
#
# ## Prerequisites
# - Familiarity with daily return series and rolling volatility estimates
# - Comfort reading quantile-based loss metrics and backtest exception counts

# %%
"""Value at Risk and Conditional Value at Risk on real ETF returns."""

import logging
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from arch import arch_model
from IPython.display import Markdown, display
from ml4t.diagnostic.evaluation.distribution import analyze_distribution
from plotly.subplots import make_subplots
from scipy import stats

from data import load_etfs
from utils.paths import get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

# %% tags=["parameters"]
SYMBOL = "SPY"
SEED = 42
PORTFOLIO_SYMBOLS = ["SPY", "AGG", "GLD", "EFA", "EEM"]
START_DATE = "2006-01-01"
END_DATE = "2024-12-31"
ROLLING_WINDOW = 252
REGIME_WINDOW = 63
REGIME_MIN_HISTORY = 252
FORECAST_EVALUATION_START = "2020-01-02"
N_MC_SIMULATIONS = 10_000
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
BACKTEST_CONFIDENCE = 0.95
ROLLING_VAR_WINDOW = 21
EWMA_LAMBDA = 0.94

# %%
OUTPUT_DIR = get_output_dir(19, "var_cvar")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
set_global_seeds(SEED)

# %% [markdown]
# What each setting decides:
#
# - `SYMBOL` carries the single-asset analysis and `PORTFOLIO_SYMBOLS` the diversification one. The
#   five span US equities, US bonds, gold, developed international and emerging markets, so they
#   are the kind of basket whose components do not all fall together.
# - `START_DATE` and `END_DATE` bound the sample. It deliberately opens before 2008, because a
#   tail-risk notebook whose sample contains no crisis measures the wrong thing.
# - `ROLLING_WINDOW` is the trailing window each rolling VaR estimate is computed from, about a
#   trading year. Shorter windows react to a change in volatility sooner and estimate the far tail
#   from fewer observations: the deepest confidence level reported here rests on roughly the two or
#   three worst days in the window.
# - `REGIME_WINDOW` is the window the volatility state is read from, about a quarter, and
#   `REGIME_MIN_HISTORY` is how much history must accumulate before the notebook is willing to
#   split that volatility into terciles at all.
# - `FORECAST_EVALUATION_START` is the date the volatility forecast comparison begins. The GARCH
#   model is estimated strictly before it, so the comparison is out of sample for every method.
# - `CONFIDENCE_LEVELS` are the coverage levels reported side by side, and `BACKTEST_CONFIDENCE`
#   is the one carried into the backtest.
# - `ROLLING_VAR_WINDOW` and `EWMA_LAMBDA` parameterize two of the three volatility forecasts. The
#   lambda is the RiskMetrics convention: each day's variance keeps that share of yesterday's
#   estimate and gives the rest to yesterday's squared return.
# - `N_MC_SIMULATIONS` is how many paths the simulated VaR draws. It affects only the precision of
#   that one estimate.

# %% [markdown]
# ## 1. Load ETF returns
#
# We use SPY for the single-asset VaR analysis and a diversified five-ETF basket
# (US equities, US aggregate bonds, gold, developed international, emerging markets)
# for the portfolio-level diversification benefit.

# %%
etf_panel = (
    load_etfs()
    .filter(
        (pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
        & (pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
        & pl.col("symbol").is_in(PORTFOLIO_SYMBOLS)
    )
    .sort(["symbol", "timestamp"])
)

returns_panel = (
    etf_panel.with_columns(ret=(pl.col("close") / pl.col("close").shift(1) - 1).over("symbol"))
    .drop_nulls("ret")
    .select(["timestamp", "symbol", "ret"])
)

# Single-asset path uses the full SPY history. The portfolio-VaR section below
# uses returns_wide, which intersects calendars across all five ETFs.
returns_spy = (
    returns_panel.filter(pl.col("symbol") == SYMBOL).sort("timestamp").select(["timestamp", "ret"])
)
returns_wide = returns_panel.pivot(values="ret", index="timestamp", on="symbol").drop_nulls()
print(f"Loaded {returns_spy.height:,} {SYMBOL} daily returns (single-asset path)")
print(
    f"Loaded {returns_wide.height:,} aligned daily returns across "
    f"{len(PORTFOLIO_SYMBOLS)} ETFs (portfolio path)"
)
print(f"  {SYMBOL}: {returns_spy['timestamp'].min()} to {returns_spy['timestamp'].max()}")
print(
    f"  Portfolio (aligned): {returns_wide['timestamp'].min()} to {returns_wide['timestamp'].max()}"
)

dates = returns_spy["timestamp"].to_numpy()
returns = returns_spy["ret"].to_numpy()
N_DAYS = len(returns)

# %%
summary_stats = pl.DataFrame(
    {
        "metric": ["Mean (%)", "Std (%)", "Skewness", "Excess kurtosis"],
        SYMBOL: [
            np.mean(returns) * 100,
            np.std(returns) * 100,
            stats.skew(returns),
            stats.kurtosis(returns),
        ],
    }
).with_columns(pl.col(SYMBOL).round(4))
summary_stats

# %% [markdown]
# ## 2. VaR Computation Methods
#
# Each method recovers the same loss quantile from a different distributional lens.
# Historical VaR is non-parametric, parametric VaR assumes Gaussian returns,
# Cornish-Fisher adjusts the Gaussian quantile for skewness and kurtosis, and Monte
# Carlo VaR samples from a fitted Student-t distribution to capture heavy tails.

# %% [markdown]
# ### Historical VaR
#
# Historical VaR uses the empirical return distribution without imposing a parametric shape.


# %%
def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical (non-parametric) VaR: $\\text{VaR}_\\alpha = -q_{1-\\alpha}(r)$."""
    alpha = 1 - confidence
    return -np.percentile(returns, alpha * 100)


# %% [markdown]
# ### Parametric VaR
#
# Gaussian VaR provides a compact benchmark when we assume returns are approximately normal.


# %%
def parametric_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) VaR: $-\\mu + \\sigma\\,\\Phi^{-1}(1-\\alpha)$."""
    mu = np.mean(returns)
    sigma = np.std(returns)
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha)
    return -(mu - z * sigma)


# %% [markdown]
# ### Cornish-Fisher VaR
#
# Cornish-Fisher VaR adjusts the Gaussian quantile for skewness and kurtosis.


# %%
def cornish_fisher_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Cornish-Fisher VaR with a skew/kurtosis adjustment to the Gaussian quantile."""
    mu = np.mean(returns)
    sigma = np.std(returns)
    s = stats.skew(returns)
    k = stats.kurtosis(returns)

    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return -(mu + z_cf * sigma)


# %% [markdown]
# ### Monte Carlo VaR
#
# Monte Carlo VaR estimates the loss threshold by simulating future paths from a fitted
# Student-t distribution.


# %%
def monte_carlo_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    n_simulations: int = N_MC_SIMULATIONS,
    horizon: int = 1,
    seed: int = SEED,
) -> float:
    """Monte Carlo VaR by sampling from a fitted Student-t."""
    df, loc, scale = stats.t.fit(returns)
    rng = np.random.default_rng(seed)
    simulated = rng.standard_t(df, size=(n_simulations, horizon)) * scale + loc
    cumulative = simulated.sum(axis=1)
    alpha = 1 - confidence
    return -np.percentile(cumulative, alpha * 100)


# %% [markdown]
# ### Method comparison
#
# The methods diverge where tail-shape assumptions matter most. In particular, a truncated
# Cornish-Fisher expansion can become unstable when empirical excess kurtosis is large, so
# the estimate needs a coverage backtest rather than automatic preference over simpler methods.

# %%
var_results = []
for conf in CONFIDENCE_LEVELS:
    var_results.append(
        {
            "confidence": conf,
            "historical": historical_var(returns, conf) * 100,
            "parametric": parametric_var(returns, conf) * 100,
            "cornish_fisher": cornish_fisher_var(returns, conf) * 100,
            "monte_carlo": monte_carlo_var(returns, conf) * 100,
        }
    )
var_df = pl.DataFrame(var_results)

# %%
fig = go.Figure()
x = [f"{round(c * 100)}%" for c in CONFIDENCE_LEVELS]
for method in ["historical", "parametric", "cornish_fisher", "monte_carlo"]:
    fig.add_trace(go.Bar(x=x, y=var_df[method], name=method.replace("_", " ").title()))
fig.update_layout(
    title="Tail assumptions dominate deep-confidence VaR",
    xaxis=dict(title="Confidence level", type="category"),
    yaxis_title="VaR (% of NAV)",
    barmode="group",
    height=400,
)
show_plotly_with_alt(
    fig,
    "Grouped bars of VaR by confidence level, four methods per level. All four agree closely at the shallowest level and fan apart at the deepest, where the assumption about tail shape starts to matter.",
)

# %% [markdown]
# ## 3. Conditional Value at Risk (CVaR / Expected Shortfall)
#
# CVaR is the average loss conditional on a VaR breach. Unlike VaR, CVaR is subadditive
# and provides full-tail information - both properties that make it the preferred
# coherent risk measure when tail severity matters more than tail frequency.

# %% [markdown]
# ### Historical CVaR
#
# Historical CVaR averages the realized losses that fall beyond the VaR threshold.


# %%
def historical_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical CVaR (Expected Shortfall): $\\mathbb{E}[L \\mid L \\geq \\text{VaR}_\\alpha]$."""
    alpha = 1 - confidence
    var_q = np.percentile(returns, alpha * 100)
    tail_losses = returns[returns <= var_q]
    return -np.mean(tail_losses)


# %% [markdown]
# ### Parametric CVaR
#
# Parametric CVaR extends the Gaussian assumption to estimate expected tail losses analytically.


# %%
def parametric_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) CVaR: $-\\mu + \\sigma \\frac{\\phi(\\Phi^{-1}(1-\\alpha))}{\\alpha}$."""
    mu = np.mean(returns)
    sigma = np.std(returns)
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha)
    phi_z = stats.norm.pdf(z)
    return -(mu - sigma * phi_z / alpha)


# %% [markdown]
# ### Student-t CVaR
#
# Student-t CVaR captures heavier tails than the Gaussian alternative.


# %%
def student_t_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """CVaR under a fitted Student-t distribution (closed-form expression)."""
    df, loc, scale = stats.t.fit(returns)
    alpha = 1 - confidence
    t_val = stats.t.ppf(alpha, df)
    f_t = stats.t.pdf(t_val, df)
    return -loc + scale * (df + t_val**2) / ((df - 1) * alpha) * f_t


# %% [markdown]
# ### Comparing VaR with CVaR

# %%
risk_measures = []
for conf in CONFIDENCE_LEVELS:
    risk_measures.append(
        {
            "confidence": conf,
            "var_historical": historical_var(returns, conf) * 100,
            "cvar_historical": historical_cvar(returns, conf) * 100,
            "var_parametric": parametric_var(returns, conf) * 100,
            "cvar_parametric": parametric_cvar(returns, conf) * 100,
            "cvar_student_t": student_t_cvar(returns, conf) * 100,
        }
    )
risk_df = pl.DataFrame(risk_measures)

# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=["VaR comparison", "CVaR comparison"])
fig.add_trace(go.Bar(x=x, y=risk_df["var_historical"], name="VaR Historical"), row=1, col=1)
fig.add_trace(go.Bar(x=x, y=risk_df["var_parametric"], name="VaR Parametric"), row=1, col=1)
fig.add_trace(go.Bar(x=x, y=risk_df["cvar_historical"], name="CVaR Historical"), row=1, col=2)
fig.add_trace(go.Bar(x=x, y=risk_df["cvar_parametric"], name="CVaR Parametric"), row=1, col=2)
fig.add_trace(go.Bar(x=x, y=risk_df["cvar_student_t"], name="CVaR Student-t"), row=1, col=2)
fig.update_yaxes(title_text="Risk (%)", row=1, col=1)
fig.update_yaxes(title_text="Risk (%)", row=1, col=2)
for col in [1, 2]:
    fig.update_xaxes(
        title_text="Confidence level",
        type="category",
        tickmode="array",
        tickvals=x,
        ticktext=x,
        row=1,
        col=col,
    )
fig.update_layout(
    title="CVaR reveals loss severity beyond the VaR threshold", barmode="group", height=400
)
show_plotly_with_alt(
    fig,
    "Two grouped bar panels. The left compares VaR estimates across confidence levels, the right compares CVaR estimates. Every CVaR bar stands taller than its VaR counterpart, and the gap widens toward the deeper confidence levels.",
)

# %% [markdown]
# CVaR exceeds VaR at every confidence level, and the gap widens as the confidence level deepens.
# That is the whole point of the measure: the further into the tail the threshold sits, the more
# the losses beyond it differ from the threshold itself. The Student-t estimate sits above the
# Gaussian one at the deepest level, which is what a heavier-tailed fit does.

# %% [markdown]
# ## 4. A Bound That Assumes Nothing About the Distribution
#
# The Cantelli (one-sided Chebyshev) inequality provides a worst-case tail probability
# requiring only finite mean and variance - no distributional assumptions:
#
# $$P(X - \mu \geq k\sigma) \leq \frac{1}{1 + k^2}$$
#
# This is much weaker than Gaussian bounds but applies to *any* distribution, making it
# useful as a conservative stress benchmark when distribution assumptions are suspect.

# %%
k_values = [1.0, 1.5, 2.0, 3.0]
standardized_losses = (np.mean(returns) - returns) / np.std(returns)
cantelli_df = pl.DataFrame(
    {
        "k": k_values,
        "cantelli_bound": [1.0 / (1.0 + k**2) for k in k_values],
        "gaussian_tail": [stats.norm.sf(k) for k in k_values],
        "empirical_spy_tail": [np.mean(standardized_losses >= k) for k in k_values],
    }
).with_columns(pl.exclude("k").round(4))

fig = go.Figure()
for column, label in [
    ("cantelli_bound", "Cantelli bound"),
    ("gaussian_tail", "Gaussian tail"),
    ("empirical_spy_tail", "Empirical SPY tail"),
]:
    fig.add_trace(go.Scatter(x=k_values, y=cantelli_df[column], mode="lines+markers", name=label))
fig.update_layout(
    title="Cantelli remains conservative for observed SPY downside tails",
    xaxis_title="Downside deviation from mean (standard deviations)",
    yaxis_title="Tail probability",
    yaxis_tickformat=".1%",
    height=400,
)
show_plotly_with_alt(
    fig,
    "Three curves of tail probability against standard deviations below the mean. The Cantelli bound sits far above both the Gaussian curve and the observed SPY frequencies at every point.",
)

# %% tags=["results"]
cantelli_k2 = cantelli_df.filter(pl.col("k") == 2.0).row(0, named=True)
display(
    Markdown(
        f"At $k=2$, the Cantelli bound is {cantelli_k2['cantelli_bound']:.1%}, "
        f"compared with {cantelli_k2['gaussian_tail']:.1%} under a Gaussian model and "
        f"{cantelli_k2['empirical_spy_tail']:.1%} in this SPY sample. The bound is "
        "distribution-free, but its conservatism limits its use as a day-to-day forecast."
    )
)

# %% [markdown]
# ## 5. What Shape Is the Return Distribution
#
# `analyze_distribution` and `analyze_tails` from `ml4t.diagnostic` package together the
# moments, normality tests, Hill estimator, and QQ-plot diagnostics. This puts the
# parametric / Cornish-Fisher / Student-t choice on an empirical footing.

# %%
dist_result = analyze_distribution(returns)
tail_analysis = dist_result.tail_analysis_result
distribution_summary = pl.DataFrame(
    {
        "metric": [
            "Jarque-Bera p-value",
            "Shapiro-Wilk p-value",
            "Hill tail index",
            "Normal QQ R-squared",
            "Student-t QQ R-squared",
            "Recommended distribution",
        ],
        "value": [
            f"{dist_result.jarque_bera_result.p_value:.4g}",
            f"{dist_result.shapiro_wilk_result.p_value:.4g}",
            f"{tail_analysis.hill_result.tail_index:.3f}",
            f"{tail_analysis.qq_normal.r_squared:.3f}",
            f"{tail_analysis.qq_t.r_squared:.3f}",
            dist_result.recommended_distribution,
        ],
    }
)
distribution_summary

# %% [markdown]
# ## 6. Did the Estimates Hold Up
#
# A VaR estimate is only as good as its empirical exception rate. The Kupiec
# proportion-of-failures test compares the realized exception rate to the configured
# coverage level under a binomial null.


# %% [markdown]
# ### Kupiec proportion-of-failures test


# %%
def kupiec_test(exceptions: np.ndarray, expected_rate: float) -> tuple[float, float]:
    """Return the Kupiec likelihood-ratio statistic and its p-value."""
    n = len(exceptions)
    x = int(np.sum(exceptions))
    p = expected_rate
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        lr = 2 * (x * np.log(x / (n * p)) + (n - x) * np.log((n - x) / (n * (1 - p))))
    return lr, float(stats.chi2.sf(lr, 1))


# %% [markdown]
# ### Rolling VaR backtest helper


# %%
def backtest_var(
    returns: np.ndarray,
    window: int = ROLLING_WINDOW,
    confidence: float = 0.95,
    method: str = "historical",
) -> dict:
    """Backtest rolling VaR estimates against realized one-step losses."""
    var_func = {
        "historical": historical_var,
        "parametric": parametric_var,
        "cornish_fisher": cornish_fisher_var,
    }[method]

    var_estimates, exceptions = [], []
    for i in range(window, len(returns)):
        past = returns[i - window : i]
        v = var_func(past, confidence)
        var_estimates.append(v)
        exceptions.append(-returns[i] > v)

    var_estimates = np.array(var_estimates)
    exceptions = np.array(exceptions)
    n_exc = int(exceptions.sum())
    rate = n_exc / len(exceptions)
    expected = 1 - confidence
    lr, p_value = kupiec_test(exceptions, expected)

    return {
        "method": method,
        "n_observations": len(exceptions),
        "n_exceptions": n_exc,
        "exception_rate": rate,
        "expected_rate": expected,
        "exception_ratio": rate / expected,
        "kupiec_pvalue": p_value,
        "var_estimates": var_estimates,
        "exceptions": exceptions,
    }


# %%
backtest_results = []
for method in ["historical", "parametric", "cornish_fisher"]:
    backtest_results.append(
        backtest_var(returns, window=ROLLING_WINDOW, confidence=BACKTEST_CONFIDENCE, method=method)
    )

backtest_summary = pl.DataFrame(
    [
        {
            "method": r["method"],
            "exceptions": r["n_exceptions"],
            "rate_pct": r["exception_rate"] * 100,
            "ratio": r["exception_ratio"],
            "kupiec_p": r["kupiec_pvalue"],
            "reject_5pct": bool(r["kupiec_pvalue"] < 0.05),
        }
        for r in backtest_results
    ]
).with_columns(
    pl.col("rate_pct").round(4),
    pl.col("ratio").round(4),
    pl.col("kupiec_p").round(4),
)
backtest_summary

# %% [markdown]
# The exception ratio measures distance from the configured coverage target; the Kupiec test
# asks whether that distance is statistically distinguishable from correct unconditional
# coverage. Failure to reject is evidence about coverage, not proof that a model is valid.

# %% tags=["results"]
best_result = min(backtest_results, key=lambda r: abs(r["exception_ratio"] - 1))
backtest_by_method = {result["method"]: result for result in backtest_results}
historical_result = backtest_by_method["historical"]
parametric_result = backtest_by_method["parametric"]
cornish_fisher_result = backtest_by_method["cornish_fisher"]
display(
    Markdown(
        f"Historical VaR has an exception ratio of {historical_result['exception_ratio']:.2f} "
        f"with Kupiec $p={historical_result['kupiec_pvalue']:.3f}$. Parametric and "
        f"Cornish-Fisher VaR produce ratios of {parametric_result['exception_ratio']:.2f} "
        f"and {cornish_fisher_result['exception_ratio']:.2f}; their Kupiec p-values are "
        f"{parametric_result['kupiec_pvalue']:.3f} and "
        f"{cornish_fisher_result['kupiec_pvalue']:.3f}."
    )
)

# %%
test_dates = dates[ROLLING_WINDOW:]
test_returns = returns[ROLLING_WINDOW:]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=test_returns * 100,
        mode="lines",
        name="Returns",
        line=dict(color=COLORS["neutral"], width=1),
    )
)
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=-best_result["var_estimates"] * 100,
        mode="lines",
        name="VaR (95%)",
        line=dict(color=COLORS["blue"], dash="dash"),
    )
)
mask = best_result["exceptions"]
fig.add_trace(
    go.Scatter(
        x=test_dates[mask],
        y=test_returns[mask] * 100,
        mode="markers",
        name="VaR exceptions",
        marker=dict(color=COLORS["negative"], size=6, symbol="x"),
    )
)
fig.update_layout(
    title="Losses beyond the VaR line cluster rather than arriving evenly",
    xaxis_title="Date",
    yaxis_title="Return (%)",
    height=500,
)
show_plotly_with_alt(
    fig,
    "A daily return series with a dashed VaR line beneath it and crosses marking the days the loss exceeded that line. The crosses are not spread evenly: they bunch tightly in 2008 and 2020 and are sparse for long stretches between.",
)

# %% [markdown]
# ## 7. Rolling VaR and CVaR

# %%
rolling_var, rolling_cvar = [], []
for i in range(ROLLING_WINDOW, len(returns)):
    past = returns[i - ROLLING_WINDOW : i]
    rolling_var.append(historical_var(past, 0.95))
    rolling_cvar.append(historical_cvar(past, 0.95))
rolling_var = np.array(rolling_var)
rolling_cvar = np.array(rolling_cvar)
rolling_dates = dates[ROLLING_WINDOW:]

rolling_summary = pl.DataFrame(
    {
        "metric": ["Mean VaR(95%)", "Mean CVaR(95%)", "Mean CVaR/VaR ratio"],
        "value": [
            f"{np.mean(rolling_var) * 100:.3f}%",
            f"{np.mean(rolling_cvar) * 100:.3f}%",
            f"{np.mean(rolling_cvar) / np.mean(rolling_var):.2f}x",
        ],
    }
)
rolling_summary

# %%
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, subplot_titles=["Rolling VaR and CVaR", "CVaR / VaR ratio"]
)
fig.add_trace(
    go.Scatter(
        x=rolling_dates,
        y=rolling_var * 100,
        mode="lines",
        name="VaR 95%",
        line=dict(color=COLORS["blue"]),
    ),
    row=1,
    col=1,
)
_ = fig

# %% [markdown]
# Both series use a trailing 252-trading-day window of daily, unannualized returns. The lower
# panel normalizes expected shortfall by VaR to show changes in tail severity that are not
# explained by a larger loss threshold alone.

# %%
fig.add_trace(
    go.Scatter(
        x=rolling_dates,
        y=rolling_cvar * 100,
        mode="lines",
        name="CVaR 95%",
        line=dict(color=COLORS["negative"]),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=rolling_dates,
        y=rolling_cvar / rolling_var,
        mode="lines",
        name="Ratio",
        line=dict(color=COLORS["copper"]),
    ),
    row=2,
    col=1,
)
fig.update_yaxes(title_text="Risk (%)", row=1, col=1)
fig.update_yaxes(title_text="CVaR / VaR", row=2, col=1)
fig.update_layout(
    title=(
        "Tail severity remains above the loss threshold"
        "<br><sup>252-trading-day trailing window; daily, unannualized returns</sup>"
    ),
    height=600,
)
show_plotly_with_alt(
    fig,
    "Two stacked panels over the full sample. The upper shows rolling VaR and CVaR moving together, spiking in 2008 and 2020. The lower shows their ratio, which stays above one throughout and moves independently of the level above it.",
)

# %% [markdown]
# ## 8. Tail Risk Conditioned on the Volatility State
#
# Unconditional VaR averages over regimes and masks the tail amplification that occurs
# during high-volatility periods. Each day uses prior-close 63-trading-day daily volatility
# and expanding point-in-time tercile thresholds available at that time. The same-day return
# never defines its own regime.

# %%
regime_frame = returns_spy.with_columns(
    rolling_vol=pl.col("ret").rolling_std(window_size=REGIME_WINDOW).shift(1)
)
rolling_vol = regime_frame["rolling_vol"].to_numpy()
low_threshold = np.full(len(rolling_vol), np.nan)
high_threshold = np.full(len(rolling_vol), np.nan)
regimes = np.full(len(rolling_vol), None, dtype=object)

# %% [markdown]
# At each date, the expanding thresholds use only lagged volatility observations already
# available by that morning. The minimum history avoids unstable early-sample terciles.

# %%
for i, current_vol in enumerate(rolling_vol):
    available_history = rolling_vol[: i + 1]
    available_history = available_history[np.isfinite(available_history)]
    if np.isfinite(current_vol) and len(available_history) >= REGIME_MIN_HISTORY:
        low_threshold[i], high_threshold[i] = np.quantile(available_history, [0.33, 0.67])
        regimes[i] = (
            "Low Vol"
            if current_vol <= low_threshold[i]
            else "Mid Vol"
            if current_vol <= high_threshold[i]
            else "High Vol"
        )

valid_regimes = np.array([regime is not None for regime in regimes])
vol_dates = dates[valid_regimes]
vol_returns = returns[valid_regimes]
regimes = regimes[valid_regimes]

regime_rows = []
regime_stats = {}
for regime in ["Low Vol", "Mid Vol", "High Vol"]:
    mask = regimes == regime
    regime_ret = vol_returns[mask]
    var_val = historical_var(regime_ret, 0.95) * 100
    cvar_val = historical_cvar(regime_ret, 0.95) * 100
    regime_stats[regime] = {"var": var_val, "cvar": cvar_val, "n": int(mask.sum())}
    regime_rows.append(
        {
            "regime": regime,
            "n_days": int(mask.sum()),
            "var_95_pct": var_val,
            "cvar_95_pct": cvar_val,
            "cvar_var_ratio": cvar_val / var_val,
        }
    )
regime_df = pl.DataFrame(regime_rows).with_columns(pl.exclude("regime").round(3))

# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=["VaR by regime", "CVaR by regime"])
regime_names = list(regime_stats.keys())
colors = [COLORS["positive"], COLORS["amber"], COLORS["negative"]]
for col_i, metric in enumerate(["var", "cvar"], 1):
    vals = [regime_stats[r][metric] for r in regime_names]
    fig.add_trace(
        go.Bar(x=regime_names, y=vals, marker_color=colors, showlegend=False),
        row=1,
        col=col_i,
    )
    fig.update_yaxes(title_text=f"{metric.upper()} (%)", row=1, col=col_i)
fig.update_layout(
    title=(
        "High-volatility states amplify SPY tail losses"
        "<br><sup>Prior-close 63-trading-day volatility; expanding point-in-time terciles</sup>"
    ),
    height=400,
)
show_plotly_with_alt(
    fig,
    "Two bar panels, VaR on the left and CVaR on the right, each with one bar per volatility state. Both rise steeply from the low state to the high one.",
)

# %% tags=["results"]
low_vol_stats = regime_stats["Low Vol"]
high_vol_stats = regime_stats["High Vol"]
display(
    Markdown(
        f"CVaR rises from {low_vol_stats['cvar']:.2f}% in the low-volatility state to "
        f"{high_vol_stats['cvar']:.2f}% in the high-volatility state, a "
        f"{high_vol_stats['cvar'] / low_vol_stats['cvar']:.1f}x increase. Because each "
        "label is available before the return it classifies, a risk control can use this "
        "state without same-bar look-ahead."
    )
)

# %% [markdown]
# ## 9. How Deep the Losses Went, and How Long Recovery Took
#
# VaR and CVaR are point-loss measures; drawdowns capture *path* risk - how deep the
# losses go and how long the recovery takes. For the same underlying return series,
# drawdowns and tail measures stress different aspects of the same distribution.


# %%
def drawdown_path(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cumulative wealth, running peak, and drawdown (negative %) series."""
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1
    return wealth, peak, dd


wealth, peak, drawdown = drawdown_path(returns)
max_dd = float(drawdown.min())
max_dd_idx = int(drawdown.argmin())
peak_idx = int(np.argmax(wealth[: max_dd_idx + 1]))
recovery_offset = np.where(wealth[max_dd_idx:] >= peak[max_dd_idx])[0]
recovery_idx = int(max_dd_idx + recovery_offset[0]) if recovery_offset.size else None

drawdown_summary = pl.DataFrame(
    {
        "metric": [
            "Max drawdown",
            "Max-drawdown trough date",
            "Prior peak date",
            "Recovery date",
            "Drawdown duration (trading days)",
            "Recovery duration (trading days)",
        ],
        "value": [
            f"{max_dd * 100:.2f}%",
            str(pd.Timestamp(dates[max_dd_idx]).date()),
            str(pd.Timestamp(dates[peak_idx]).date()),
            str(pd.Timestamp(dates[recovery_idx]).date())
            if recovery_idx is not None
            else "not recovered in sample",
            str(max_dd_idx - peak_idx),
            str(recovery_idx - max_dd_idx) if recovery_idx is not None else "not recovered",
        ],
    }
)
drawdown_summary

# %%
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, subplot_titles=[f"{SYMBOL} growth of $1", "Drawdown"]
)
fig.add_trace(
    go.Scatter(x=dates, y=wealth, mode="lines", name="Wealth", line=dict(color=COLORS["blue"])),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=dates,
        y=peak,
        mode="lines",
        name="Peak",
        line=dict(color=COLORS["neutral"], dash="dash"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=dates,
        y=drawdown * 100,
        mode="lines",
        name="Drawdown",
        fill="tozeroy",
        line=dict(color=COLORS["negative"]),
    ),
    row=2,
    col=1,
)
_ = fig

# %% [markdown]
# The lower panel plots drawdown as a negative percentage with zero at the top, which is the
# convention: the line sits at zero whenever the series is at a new high and falls away from it
# during a decline, so the depth of each trough is read directly off the axis.

# %%
fig.update_yaxes(title_text="Wealth ($1 → x)", row=1, col=1)
fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
fig.update_layout(
    title=f"{SYMBOL} took years to regain its pre-crisis peak",
    height=600,
    showlegend=True,
)
show_plotly_with_alt(
    fig,
    "Two stacked panels sharing a date axis: growth of one dollar with its running peak above, and drawdown below plotted as a negative percentage from a zero line at the top. The 2008 trough is by far the deepest and the recovery back to the previous peak takes years.",
)

# %% [markdown]
# Maximum drawdown is the worst-case path loss in the sample; the trough-to-recovery
# duration is the operational pain that allocators and risk officers actually live
# through. Tail metrics (CVaR) and path metrics (max drawdown) tend to move together
# but the path metric responds to the *sequence* of losses - two distributions with
# identical CVaR can produce very different drawdown experiences.

# %% [markdown]
# ## 10. How Much Diversification Removes
#
# The gap between this portfolio's tail risk and the weighted stand-alone estimates measures
# its empirical diversification benefit. Unlike CVaR, VaR is not generally subadditive, so a
# positive VaR benefit in this sample is an observation rather than a mathematical guarantee.


# %%
def portfolio_tail_risk(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """Compute empirical portfolio VaR and CVaR against weighted stand-alone estimates."""
    portfolio_returns = returns_matrix @ weights
    individual_var = np.array(
        [historical_var(returns_matrix[:, i], confidence) for i in range(returns_matrix.shape[1])]
    )
    individual_cvar = np.array(
        [historical_cvar(returns_matrix[:, i], confidence) for i in range(returns_matrix.shape[1])]
    )
    undiversified_var = float(np.sum(weights * individual_var))
    undiversified_cvar = float(np.sum(weights * individual_cvar))
    diversified_var = float(historical_var(portfolio_returns, confidence))
    diversified_cvar = float(historical_cvar(portfolio_returns, confidence))
    return {
        "portfolio_var": diversified_var,
        "portfolio_cvar": diversified_cvar,
        "undiversified_var": undiversified_var,
        "undiversified_cvar": undiversified_cvar,
        "var_benefit": 1 - diversified_var / undiversified_var,
        "cvar_benefit": 1 - diversified_cvar / undiversified_cvar,
        "portfolio_returns": portfolio_returns,
    }


# %%
asset_returns = returns_wide.select(PORTFOLIO_SYMBOLS).to_numpy()
weights = np.ones(len(PORTFOLIO_SYMBOLS)) / len(PORTFOLIO_SYMBOLS)
port_var_result = portfolio_tail_risk(asset_returns, weights, confidence=0.95)

portfolio_summary = pl.DataFrame(
    {
        "metric": ["VaR", "CVaR"],
        "portfolio_pct": [
            port_var_result["portfolio_var"] * 100,
            port_var_result["portfolio_cvar"] * 100,
        ],
        "weighted_standalone_pct": [
            port_var_result["undiversified_var"] * 100,
            port_var_result["undiversified_cvar"] * 100,
        ],
    }
)

# %%
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=portfolio_summary["metric"],
        y=portfolio_summary["portfolio_pct"],
        name="Equal-weight portfolio",
        marker_color=COLORS["blue"],
    )
)
fig.add_trace(
    go.Bar(
        x=portfolio_summary["metric"],
        y=portfolio_summary["weighted_standalone_pct"],
        name="Weighted stand-alone risk",
        marker_color=COLORS["neutral"],
    )
)
fig.update_layout(
    title="Diversification lowers this portfolio's empirical tail risk",
    xaxis_title="Tail measure (95% confidence)",
    yaxis_title="Loss (% of NAV)",
    barmode="group",
    height=400,
)
show_plotly_with_alt(
    fig,
    "Grouped bars comparing the equal-weight portfolio's VaR and CVaR against the weighted sum of the same measures computed for each fund on its own. The portfolio bars are visibly shorter.",
)

# %% tags=["results"]
display(
    Markdown(
        f"The equal-weight portfolio's VaR is {port_var_result['portfolio_var']:.2%}, "
        f"compared with {port_var_result['undiversified_var']:.2%} for the weighted "
        f"stand-alone estimates, an empirical benefit of {port_var_result['var_benefit']:.1%}. "
        f"CVaR falls by {port_var_result['cvar_benefit']:.1%}; unlike VaR, this benefit is "
        "consistent with CVaR's general subadditivity property."
    )
)

# %% [markdown]
# ### Does the Benefit Hold When It Is Needed
#
# A diversification benefit measured over the whole sample is close to useless if it disappears in
# the states where losses actually happen - correlations between asset classes are famously higher
# in a crisis than in a calm market. Applying the same volatility labels to the aligned ETF panel
# splits the benefit by state. Each label was known before the return it classifies, so a portfolio
# could have been positioned on it.

# %%
regime_labels = pl.DataFrame({"timestamp": vol_dates, "regime": regimes})
portfolio_by_regime = returns_wide.join(regime_labels, on="timestamp", how="inner")
regime_portfolio_rows = []
for regime in ["Low Vol", "Mid Vol", "High Vol"]:
    regime_matrix = portfolio_by_regime.filter(pl.col("regime") == regime).select(PORTFOLIO_SYMBOLS)
    result = portfolio_tail_risk(regime_matrix.to_numpy(), weights, confidence=0.95)
    regime_portfolio_rows.append(
        {
            "regime": regime,
            "n_days": regime_matrix.height,
            "var_benefit_pct": result["var_benefit"] * 100,
            "cvar_benefit_pct": result["cvar_benefit"] * 100,
        }
    )
regime_portfolio_df = pl.DataFrame(regime_portfolio_rows)

# %%
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=regime_portfolio_df["regime"],
        y=regime_portfolio_df["var_benefit_pct"],
        name="VaR benefit",
        marker_color=COLORS["blue"],
    )
)
fig.add_trace(
    go.Bar(
        x=regime_portfolio_df["regime"],
        y=regime_portfolio_df["cvar_benefit_pct"],
        name="CVaR benefit",
        marker_color=COLORS["copper"],
    )
)
fig.update_layout(
    title="Tail-risk diversification changes with the volatility state",
    xaxis_title="Point-in-time volatility state",
    yaxis_title="Empirical diversification benefit (%)",
    barmode="group",
    height=400,
)
show_plotly_with_alt(
    fig,
    "Grouped bars of the VaR and CVaR diversification benefit, one pair per volatility state, showing how much of the standalone tail risk the portfolio removes in each.",
)

# %% tags=["results"]
regime_portfolio_lookup = {row["regime"]: row for row in regime_portfolio_df.iter_rows(named=True)}
low_regime_benefit = regime_portfolio_lookup["Low Vol"]["cvar_benefit_pct"]
high_regime_benefit = regime_portfolio_lookup["High Vol"]["cvar_benefit_pct"]
benefit_direction = "contracts" if high_regime_benefit < low_regime_benefit else "expands"
display(
    Markdown(
        f"The empirical CVaR benefit {benefit_direction} from {low_regime_benefit:.1f}% "
        f"in the low-volatility state to {high_regime_benefit:.1f}% in the "
        "high-volatility state. The result is sample-specific; the point-in-time labels "
        "make the comparison one a risk control could have acted on at the time."
    )
)

# %% [markdown]
# ## 11. Scoring Volatility Forecasts
#
# Risk management depends on accurate volatility forecasts. Two complementary loss
# functions evaluate forecast quality:
#
# $$\text{MSE} = \frac{1}{T}\sum_{t=1}^T (\hat{\sigma}_t^2 - \sigma_t^2)^2$$
#
# $$\text{QLIKE} = \frac{1}{T}\sum_{t=1}^T \left[\log(\hat{\sigma}_t^2) + \frac{\sigma_t^2}{\hat{\sigma}_t^2}\right]$$
#
# QLIKE's asymmetric penalty makes it more appropriate for risk: under-predicting
# volatility (dangerous for risk management) is penalized more than over-predicting.
# We use squared daily returns as the volatility proxy $\sigma_t^2$. See Chapter 9 for
# deeper GARCH treatment; here we use a compact GARCH(1,1) fit solely for comparison.


# %% [markdown]
# ### QLIKE Loss
#
# QLIKE penalizes under-forecasting volatility more sharply than over-forecasting it.


# %%
def qlike_loss(forecast_var: np.ndarray, proxy_var: np.ndarray) -> float:
    """QLIKE loss: $\\log(\\hat{\\sigma}^2) + \\sigma^2_{\\text{proxy}} / \\hat{\\sigma}^2$."""
    mask = (forecast_var > 0) & np.isfinite(forecast_var) & np.isfinite(proxy_var)
    f, p = forecast_var[mask], proxy_var[mask]
    return float(np.mean(np.log(f) + p / f))


# %% [markdown]
# ### Variance MSE
#
# MSE provides a symmetric baseline loss for volatility forecast comparison.


# %%
def variance_mse(forecast_var: np.ndarray, proxy_var: np.ndarray) -> float:
    """Variance MSE: $(\\hat{\\sigma}^2 - \\sigma^2_{\\text{proxy}})^2$."""
    return float(np.mean((forecast_var - proxy_var) ** 2))


# %%
ret_series = pd.Series(returns, index=pd.DatetimeIndex(dates))
proxy_var = ret_series**2

# Rolling and EWMA variance for day t use returns only through t-1. GARCH parameters are
# estimated once on observations before the evaluation boundary; the fixed model then updates
# conditional variance recursively as evaluation-period returns arrive.
rolling_var_21 = ret_series.rolling(ROLLING_VAR_WINDOW).var().shift(1).dropna()
# RiskMetrics EWMA: sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r^2_{t-1}, which is an
# exponentially weighted mean of SQUARED RETURNS. pandas' .ewm().var() is a different estimator -
# a debiased weighted variance about the weighted mean - so it is not what the lambda names.
ewma_var = (ret_series**2).ewm(alpha=1 - EWMA_LAMBDA).mean().shift(1).dropna()
garch_model = arch_model(ret_series * 100, vol="Garch", p=1, q=1, dist="normal")
garch_fit = garch_model.fit(last_obs=FORECAST_EVALUATION_START, disp="off")
garch_forecast = garch_fit.forecast(
    horizon=1,
    start=FORECAST_EVALUATION_START,
    align="target",
    reindex=True,
)
garch_var = (garch_forecast.variance["h.1"] / 10_000).dropna()

common_idx = rolling_var_21.index.intersection(ewma_var.index).intersection(garch_var.index)
common_idx = common_idx[common_idx >= pd.Timestamp(FORECAST_EVALUATION_START)]
proxy_aligned = proxy_var.loc[common_idx].values
rolling_aligned = rolling_var_21.loc[common_idx].values
ewma_aligned = ewma_var.loc[common_idx].values
garch_aligned = garch_var.loc[common_idx].values

forecast_eval_df = pl.DataFrame(
    [
        {
            "method": name,
            "qlike": qlike_loss(fcast, proxy_aligned),
            "mse_x_1e6": variance_mse(fcast, proxy_aligned) * 1e6,
        }
        for name, fcast in [
            ("Rolling 21d", rolling_aligned),
            (f"EWMA λ={EWMA_LAMBDA}", ewma_aligned),
            ("GARCH(1,1)", garch_aligned),
        ]
    ]
).with_columns(pl.exclude("method").round(4))

# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=["QLIKE", "Variance MSE"])
fig.add_trace(
    go.Scatter(
        x=forecast_eval_df["method"],
        y=forecast_eval_df["qlike"],
        mode="markers+text",
        text=[f"{value:.3f}" for value in forecast_eval_df["qlike"]],
        textposition="top center",
        marker=dict(color=COLORS["blue"], size=12),
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=forecast_eval_df["method"],
        y=forecast_eval_df["mse_x_1e6"],
        mode="markers+text",
        text=[f"{value:.3f}" for value in forecast_eval_df["mse_x_1e6"]],
        textposition="top center",
        marker=dict(color=COLORS["copper"], size=12),
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig.update_yaxes(title_text="Average QLIKE (lower is better)", row=1, col=1)
fig.update_yaxes(title_text="MSE (x 1e-6; lower is better)", row=1, col=2)
fig.update_layout(
    title="Out-of-sample loss functions compare volatility forecasts",
    height=400,
)
show_plotly_with_alt(
    fig,
    "Two panels of labelled points comparing three volatility forecasts, by QLIKE on the left and variance MSE on the right. The two loss functions do not order the three methods identically.",
)

# %% tags=["results"]
qlike_winner = forecast_eval_df.sort("qlike").row(0, named=True)
mse_winner = forecast_eval_df.sort("mse_x_1e6").row(0, named=True)
display(
    Markdown(
        f"From {common_idx.min().date()} through {common_idx.max().date()}, "
        f"{qlike_winner['method']} has the lowest QLIKE ({qlike_winner['qlike']:.3f}) and "
        f"{mse_winner['method']} has the lowest variance MSE "
        f"({mse_winner['mse_x_1e6']:.3f} x $10^{{-6}}$). GARCH parameters are estimated "
        "before this interval, while rolling and EWMA forecasts are explicitly lagged. "
        "Squared returns remain a noisy volatility proxy, so small loss gaps should not be "
        "treated as economically decisive."
    )
)

# %% [markdown]
# ## 12. Persist the Two Artifacts the Book's Figures Read
#
# Two figures in the printed chapter are generated from this notebook's output rather than from the
# notebook's own charts, so those two tables are written to disk. Nothing else here is written:
# an artifact no named consumer reads is one more thing that can go stale without anyone noticing.

# %%
var_comparison_df = pl.DataFrame(
    {
        "confidence": [f"{round(c * 100)}%" for c in CONFIDENCE_LEVELS] * 4,
        "method": ["historical"] * 3
        + ["parametric"] * 3
        + ["cornish_fisher"] * 3
        + ["monte_carlo"] * 3,
        "var_pct": list(var_df["historical"])
        + list(var_df["parametric"])
        + list(var_df["cornish_fisher"])
        + list(var_df["monte_carlo"]),
    }
)
var_comparison_df.write_parquet(OUTPUT_DIR / "var_method_comparison.parquet")
regime_df.write_parquet(OUTPUT_DIR / "regime_conditional_risk.parquet")
print(
    f"var_method_comparison.parquet   {len(var_comparison_df)} rows -> book figure 19.2\n"
    f"regime_conditional_risk.parquet {len(regime_df)} rows -> book figure 19.3"
)

# %% tags=["results"]
high_low_cvar_ratio = high_vol_stats["cvar"] / low_vol_stats["cvar"]
display(
    Markdown(
        f"Across {N_DAYS:,} sessions, {best_result['method']} VaR came closest to its exception "
        f"budget, with an exception ratio of {best_result['exception_ratio']:.2f} and Kupiec "
        f"$p={best_result['kupiec_pvalue']:.3f}$. Rolling CVaR averaged "
        f"{np.mean(rolling_cvar) / np.mean(rolling_var):.2f} times rolling VaR. Conditioning on "
        f"the volatility state raised CVaR by a factor of {high_low_cvar_ratio:.1f} between the "
        f"calmest and most stressed terciles. The equal-weight basket removed "
        f"{port_var_result['cvar_benefit']:.1%} of the weighted stand-alone CVaR. "
        f"{SYMBOL}'s worst drawdown was {max_dd:.1%}, taking "
        f"{recovery_idx - max_dd_idx:,} trading days to recover."
    )
)

# %% [markdown]
# ## Key Takeaways
#
# 1. **A risk number is a claim about frequency, so test it against how often it was wrong.** VaR
#    at a stated confidence level predicts an exception rate. Count the exceptions, compare the
#    count to what the level implies, and use a test that says whether the difference is more than
#    sampling noise. An estimate that has never been backtested is an assertion.
#
# 2. **Report severity as well as frequency.** VaR locates a threshold and stops. CVaR averages
#    what happens past it, and the two diverge more the deeper into the tail you go - which is
#    exactly where a risk limit is supposed to bind.
#
# 3. **Correcting a normal distribution for skew and kurtosis is not free.** The Cornish-Fisher
#    expansion is a truncated polynomial, and on returns with heavy tails it can move the quantile
#    in ways that do not improve coverage. Its backtest here does no better than the plain normal.
#
# 4. **Estimate the market state from information that predates the return it labels.** The
#    volatility used to classify each day ends at the previous close, and the thresholds splitting
#    those volatilities expand through time rather than being fitted on the whole sample. A state
#    label that peeks makes any conditional risk number unusable for the control it is meant to
#    drive.
#
# 5. **Check that a diversification benefit is there when it is needed.** A benefit averaged over a
#    long sample can be dominated by the calm periods that make up most of it. Splitting it by
#    volatility state asks the question a portfolio actually faces.
#
# 6. **Choose the loss function that matches the asymmetry of the decision.** Squared error treats
#    an over-forecast and an under-forecast of volatility as equally bad. For risk they are not:
#    under-forecasting is what leaves a position too large going into a bad day, and QLIKE
#    penalizes it more heavily. The two can rank the same three forecasts differently.
#
# ### Known limitations
#
# - Every estimate is unconditional on anything but the volatility state, and one-day-ahead. A
#   ten-day regulatory horizon does not follow from scaling a one-day figure by the square root of
#   ten unless returns are independent, which the exception clustering visible in Section 6 argues
#   against.
# - The Kupiec test checks only how many exceptions occurred, not when. A model that produces the
#   right count but concentrates every exception in one month passes it and is badly wrong.
# - Historical VaR cannot produce a loss larger than the worst one in its window, so it understates
#   risk precisely when the recent past has been calm.
# - The volatility proxy for the forecast comparison is the squared daily return, which is unbiased
#   but extremely noisy. Small differences in either loss function are not decisive.
# - The regime terciles are computed on this sample's own volatility distribution, so the states
#   are relative to what this period contained rather than to any absolute level.
# - The diversification results use five ETFs at equal weights over one sample. They describe this
#   basket over this period.
#
# **Next**: [`06_stress_testing`](06_stress_testing.ipynb) asks what these estimates would have
# said about crises they were not fitted on.
