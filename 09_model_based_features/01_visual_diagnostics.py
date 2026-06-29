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
# # Visual Diagnostics and Stationarity Testing
#
# **Docker image**: `ml4t`
#
# This notebook demonstrates the complete diagnostic workflow for financial
# time series: visual inspection, stationarity tests, autocorrelation analysis,
# and rolling diagnostic features.
#
# **Learning Objectives**:
# - Perform visual diagnostics (time series plot, ACF/PACF, Q-Q plot)
# - Test stationarity using ADF and KPSS with the joint decision matrix
# - Compute Ljung-Box test for residual autocorrelation
# - Build rolling ADF/KPSS statistics as time-varying features
#
# **Book Reference**: Chapter 9, Section 9.1 (Diagnostics and Stationarity Features)
#
# **Prerequisites**: None — this is the starting point for Ch9.

# %%
"""Visual Diagnostics and Stationarity Testing — the diagnostic workflow."""

import warnings

warnings.filterwarnings("ignore")

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display
from ml4t.diagnostic.evaluation.autocorrelation import analyze_autocorrelation
from ml4t.diagnostic.evaluation.distribution import analyze_distribution
from ml4t.diagnostic.evaluation.stationarity import analyze_stationarity
from ml4t.diagnostic.evaluation.volatility import arch_lm_test
from scipy.stats import norm, probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from data import load_etfs, load_macro

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"
ROLLING_WINDOW = 252

# %% [markdown]
# ## Load Data
#
# SPY (S&P 500 ETF) for trending price series and VIX for a mean-reverting
# volatility series — contrasting stationarity behaviors.

# %%
etfs = load_etfs(symbols=["SPY"])
sp500 = etfs.select(["timestamp", "close"]).rename({"close": "value"}).sort("timestamp")

macro = load_macro()
vix = (
    macro.select(["timestamp", "vixcls"]).drop_nulls().rename({"vixcls": "value"}).sort("timestamp")
)

START = datetime.strptime(START_DATE, "%Y-%m-%d")
END = datetime.strptime(END_DATE, "%Y-%m-%d")
sp500 = sp500.filter((pl.col("timestamp") >= START) & (pl.col("timestamp") <= END))
vix = vix.filter((pl.col("timestamp") >= START) & (pl.col("timestamp") <= END))

# Add returns
sp500 = sp500.with_columns(returns=pl.col("value").pct_change() * 100).drop_nulls()

sp500_pd = sp500.to_pandas().set_index("timestamp")
vix_pd = vix.to_pandas().set_index("timestamp")
returns = sp500_pd["returns"]

print(
    f"S&P 500: {len(sp500_pd):,} obs ({sp500_pd.index.min().date()} to {sp500_pd.index.max().date()})"
)
print(f"VIX: {len(vix_pd):,} obs ({vix_pd.index.min().date()} to {vix_pd.index.max().date()})")

# %% [markdown]
# ## Visual Inspection
#
# Start with four plots that reveal trend, volatility clustering, and
# distributional properties at a glance.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

ax = axes[0, 0]
ax.plot(sp500_pd.index, sp500_pd["value"].values, linewidth=0.8)
ax.set_title("S&P 500 Index (Levels)")
ax.set_ylabel("Index Value")

ax = axes[0, 1]
ax.plot(returns.index, returns.values, linewidth=0.5, alpha=0.8)
ax.axhline(0, color="red", linestyle="--", linewidth=0.5)
ax.set_title("S&P 500 Daily Returns (%)")
ax.set_ylabel("Return (%)")

ax = axes[1, 0]
ax.plot(vix_pd.index, vix_pd["value"].values, linewidth=0.8, color="orange")
ax.axhline(20, color="red", linestyle="--", linewidth=0.5, label="VIX=20 threshold")
ax.set_title("VIX Index")
ax.set_ylabel("VIX Level")
ax.legend()

ax = axes[1, 1]
ax.hist(returns.values, bins=100, density=True, alpha=0.7, edgecolor="white")
x = np.linspace(returns.min(), returns.max(), 100)
ax.plot(x, norm.pdf(x, returns.mean(), returns.std()), "r-", linewidth=2, label="Normal")
ax.set_title("Return Distribution (Fat Tails)")
ax.set_xlabel("Daily Return (%)")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Stationarity Testing: ADF + KPSS Decision Matrix
#
# Using two tests with opposite null hypotheses provides robust conclusions:
#
# | ADF Result | KPSS Result | Conclusion |
# |:-----------|:------------|:-----------|
# | Reject H0 | Fail to reject H0 | Stationary (both agree) |
# | Fail to reject H0 | Reject H0 | Non-stationary (both agree) |
# | Reject H0 | Reject H0 | Trend-stationary |
# | Fail to reject H0 | Fail to reject H0 | Inconclusive |


# %%
def run_stationarity_tests(series: pd.Series, name: str) -> dict:
    """Run ADF and KPSS tests with joint interpretation."""
    series = series.dropna()

    adf_stat, adf_pval, adf_lags, nobs, _, _ = adfuller(series, autolag="AIC")
    kpss_stat, kpss_pval, _, _ = kpss(series, regression="c", nlags="auto")

    if adf_pval < 0.05 and kpss_pval > 0.05:
        conclusion = "Stationary (both agree)"
    elif adf_pval > 0.05 and kpss_pval < 0.05:
        conclusion = "Non-stationary (both agree)"
    elif adf_pval < 0.05 and kpss_pval < 0.05:
        conclusion = "Trend-stationary"
    else:
        conclusion = "Inconclusive"

    return {
        "series": name,
        "nobs": nobs,
        "adf_stat": round(adf_stat, 4),
        "adf_pval": round(adf_pval, 4),
        "kpss_stat": round(kpss_stat, 4),
        "kpss_pval": round(kpss_pval, 4),
        "conclusion": conclusion,
    }


results = [
    run_stationarity_tests(sp500_pd["value"], "S&P 500 Levels"),
    run_stationarity_tests(returns, "S&P 500 Returns"),
    run_stationarity_tests(vix_pd["value"], "VIX Levels"),
]

results_df = pd.DataFrame(results)
display(results_df)

# %% [markdown]
# **Finding**: Prices are non-stationary (unit root); returns are stationary.
# VIX is mean-reverting but may show trend-stationarity due to structural shifts.

# %% [markdown]
# ### ml4t-diagnostic: Consensus Stationarity Analysis
#
# The manual approach above requires running two separate tests and interpreting
# a decision matrix. `analyze_stationarity()` runs ADF, KPSS, and Phillips-Perron
# in one call and returns a consensus classification with agreement score.

# %%
for name, series in [
    ("S&P 500 Levels", sp500_pd["value"]),
    ("S&P 500 Returns", returns),
    ("VIX Levels", vix_pd["value"]),
]:
    result = analyze_stationarity(series.dropna().values)
    print(f"{name:20s}: consensus={result.consensus}, agreement={result.agreement_score:.2f}")

# %% [markdown]
# The three-test consensus (ADF + KPSS + Phillips-Perron) is more robust than
# the two-test decision matrix. The agreement score quantifies how strongly
# the tests agree — 1.0 means unanimous, below 0.5 is inconclusive.

# %% [markdown]
# ## Autocorrelation Analysis
#
# ACF/PACF plots reveal the lag structure. Key patterns:
# - Slow ACF decay → non-stationarity
# - Sharp PACF cutoff → AR process (cutoff at lag $p$)
# - Sharp ACF cutoff → MA process (cutoff at lag $q$)


# %%
def plot_correlogram(series: pd.Series, title: str, lags: int = 40):
    """Create diagnostic correlogram: time series, Q-Q, ACF, PACF."""
    series = series.dropna()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    ax.plot(series.index, series.values, linewidth=0.5, alpha=0.7, label="Series")
    rolling_mean = series.rolling(21).mean()
    ax.plot(series.index, rolling_mean.values, linewidth=1.5, color="red", label="21-day MA")
    ax.set_title("Time Series with Rolling Mean")
    ax.legend()

    # Stats annotation
    adf_pval = adfuller(series, autolag="AIC")[1]
    ljung = acorr_ljungbox(series, lags=[10], return_df=True)
    lb_pval = ljung["lb_pvalue"].values[0]
    ax.text(
        0.02,
        0.95,
        f"ADF p={adf_pval:.4f}\nLjung-Box(10) p={lb_pval:.4f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax = axes[0, 1]
    probplot(series, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot (Normal)")
    skew_val = series.skew()
    kurt_val = series.kurtosis()
    ax.text(
        0.02,
        0.95,
        f"Skew: {skew_val:.2f}\nKurtosis: {kurt_val:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax = axes[1, 0]
    plot_acf(series, lags=lags, zero=False, ax=ax)
    ax.set_title("ACF")

    ax = axes[1, 1]
    plot_pacf(series, lags=lags, zero=False, ax=ax, method="ywm")
    ax.set_title("PACF")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


fig = plot_correlogram(returns, "S&P 500 Daily Returns — Correlogram")
plt.show()

# %% [markdown]
# **Observations**: ACF and PACF stay close to zero at every lag, so any linear
# dependence in raw returns is small in magnitude — even where formal tests
# (below) detect it given the long sample. The Q-Q plot fans out at both ends,
# consistent with fat tails and excess kurtosis well above the Gaussian baseline.

# %% [markdown]
# ### ml4t-diagnostic: Autocorrelation Analysis with ARIMA Suggestion
#
# `analyze_autocorrelation()` examines the ACF/PACF patterns and suggests
# ARIMA orders — useful before fitting time series models in NB07.

# %%
acf_result = analyze_autocorrelation(returns.dropna().values)
print("=== ml4t-diagnostic: Autocorrelation Analysis ===")
print(f"Suggested ARIMA order: {acf_result.suggested_arima_order}")

# %% [markdown]
# ## Ljung-Box Test
#
# Formal test for whether the first $m$ autocorrelations are jointly zero.
# Useful for checking model residuals.

# %%
lb_results = acorr_ljungbox(returns, lags=[5, 10, 20, 40], return_df=True)
lb_returns_reject = (lb_results["lb_pvalue"] < 0.05).any()
print(
    f"Returns: {'autocorrelation detected' if lb_returns_reject else 'no significant autocorrelation'}"
)
display(lb_results)

# Check squared returns (volatility clustering)
lb_sq = acorr_ljungbox(returns**2, lags=[5, 10, 20, 40], return_df=True)
lb_sq_reject = (lb_sq["lb_pvalue"] < 0.05).any()
print(f"Squared returns: {'ARCH effects present' if lb_sq_reject else 'no ARCH effects'}")
display(lb_sq)

# %% [markdown]
# **Finding**: Highly significant Ljung-Box statistics on squared returns confirm
# ARCH effects — volatility clusters in time. This motivates the GARCH models
# developed in `08_garch_volatility`.

# %% [markdown]
# ## Rolling Stationarity Features
#
# Stationarity is not a fixed property — it can change over time. Rolling
# ADF/KPSS statistics become time-varying features that detect when
# relationships break down (e.g., cointegration weakening).

# %%
WINDOW = ROLLING_WINDOW  # From parameters cell
STEP = 5  # Compute every 5 days

rolling_stats = []

for end in range(WINDOW, len(returns), STEP):
    window_data = returns.iloc[end - WINDOW : end]
    try:
        adf_result = adfuller(window_data, autolag="AIC")
        kpss_result = kpss(window_data, regression="c", nlags="auto")
        adf_stat = adf_result[0]
        kpss_stat = kpss_result[0]

        # Decision: both agree on stationary?
        adf_reject = adf_result[1] < 0.05
        kpss_not_reject = kpss_result[1] > 0.05
        stationary = int(adf_reject and kpss_not_reject)

        rolling_stats.append(
            {
                "timestamp": returns.index[end],
                "adf_statistic": adf_stat,
                "kpss_statistic": kpss_stat,
                "stationarity_regime": stationary,
            }
        )
    except Exception:
        continue

rolling_df = pd.DataFrame(rolling_stats).set_index("timestamp")

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

ax = axes[0]
ax.plot(rolling_df.index, rolling_df["adf_statistic"], linewidth=0.8)
ax.axhline(-2.86, color="red", linestyle="--", linewidth=0.5, label="5% critical")
ax.set_title("Rolling ADF Statistic (252-Day Window)")
ax.set_ylabel("ADF Statistic")
ax.legend()

ax = axes[1]
ax.plot(rolling_df.index, rolling_df["kpss_statistic"], linewidth=0.8, color="orange")
ax.axhline(0.463, color="red", linestyle="--", linewidth=0.5, label="5% critical")
ax.set_title("Rolling KPSS Statistic (252-Day Window)")
ax.set_ylabel("KPSS Statistic")
ax.legend()

ax = axes[2]
ax.fill_between(rolling_df.index, 0, rolling_df["stationarity_regime"], alpha=0.5, color="green")
ax.set_title("Stationarity Regime (1 = Stationary by Both Tests)")
ax.set_ylabel("Regime")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Distribution Analysis
#
# Beyond stationarity and autocorrelation, the distribution of returns matters
# for model selection: fat tails affect VaR, skewness affects directional bets,
# and ARCH effects motivate GARCH models (NB08).

# %%
dist_result = analyze_distribution(returns.dropna().values)
print("=== ml4t-diagnostic: Distribution Analysis ===")
m = dist_result.moments_result
print(f"Mean: {m.mean:.4f}")
print(f"Std: {m.std:.4f}")
print(f"Skewness: {m.skewness:.4f}")
print(f"Excess kurtosis: {m.excess_kurtosis:.4f}")
print(f"Jarque-Bera p-value: {dist_result.jarque_bera_result.p_value:.6f}")
print(f"Normal: {dist_result.is_normal}")

# %% [markdown]
# **Interpretation**: Excess kurtosis >> 3 confirms fat tails (extreme returns
# more frequent than normal); the Jarque-Bera test strongly rejects normality.
# This motivates heavy-tailed distributions (Student-t) in GARCH models
# and non-parametric approaches for VaR.

# %% [markdown]
# ## ARCH Effects: Bridge to GARCH (NB08)
#
# The Ljung-Box test on squared returns already suggested volatility clustering.
# The formal ARCH-LM test confirms whether conditional heteroskedasticity is
# present — the key prerequisite for GARCH modeling.

# %%
arch_result = arch_lm_test(returns.dropna().values)
print("=== ml4t-diagnostic: ARCH-LM Test ===")
print(f"Test statistic: {arch_result.test_statistic:.4f}")
print(f"P-value: {arch_result.p_value:.6f}")
print(f"ARCH effects: {arch_result.has_arch_effects}")

# %% [markdown]
# **Bridge to NB08**: Strong ARCH effects confirm that constant-variance models
# are inadequate. GARCH(1,1) is the natural next step — see `08_garch_volatility`
# for fitting, diagnostics, and feature extraction.

# %% [markdown]
# ## Feature Catalog: Diagnostic Features
#
# | Feature | Source | Computation | Update |
# |---------|--------|-------------|--------|
# | `adf_statistic` | Unit root test | Rolling 252-day ADF | Weekly |
# | `adf_pvalue` | Unit root test | P-value from ADF | Weekly |
# | `kpss_statistic` | Unit root test | Rolling 252-day KPSS | Weekly |
# | `stationarity_regime` | Combined | Joint decision matrix | Weekly |

# %% [markdown]
# ## Key Takeaways
#
# 1. **Visual inspection first**: time series plot, ACF/PACF, Q-Q reveal
#    trends, dependence, and distributional properties at a glance
# 2. **Use both ADF and KPSS**: opposite null hypotheses give robust
#    conclusions via the decision matrix
# 3. **Returns are stationary, prices are not**: first-differencing is the
#    standard fix, but fractional differencing preserves memory (see NB03)
# 4. **Squared returns show ARCH effects**: volatility clusters, motivating
#    GARCH models (NB08)
# 5. **Rolling stationarity statistics** become time-varying features that
#    detect regime changes in real time
# 6. **ml4t-diagnostic consolidates diagnostics**: `analyze_stationarity()`
#    provides three-test consensus; `analyze_autocorrelation()` suggests ARIMA
#    orders; `analyze_distribution()` and `arch_lm_test()` complete the
#    pre-modeling diagnostic workflow
#
# **Next**: See `02_structural_breaks` for break detection and
# `03_fractional_differencing` for memory-preserving stationarity transforms.
