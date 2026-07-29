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
# # Factor Allocation Evidence: Diversification Across Return Sources
#
# **Docker image**: `ml4t`
#
# This notebook examines the empirical evidence for factor-based allocation using
# nearly a century of data from AQR Capital Management and Kenneth French's Data Library.
# The key question for portfolio construction: which factors diversify, which provide
# crisis insurance, and what does this mean for allocation decisions?
#
# **Learning Objectives**:
# - Assess factor premia using long pre-publication histories
# - Analyze cross-asset evidence from 8 asset classes
# - Quantify the value-momentum negative correlation for diversification
# - Evaluate trend-following as portfolio insurance during crises
# - Understand factor decay (the cautionary tale of the size premium)
#
# **Book Reference**: Chapter 17, §17.4 (Baseline Allocators)
#
# **Prerequisites**: None (uses AQR and Fama-French data, not case study data)

# %% [markdown]
# ## Data Requirements
#
# ### Fama-French Data (automatic)
# Data is fetched automatically from Kenneth French's Data Library on first use.
#
# ### AQR Data (one-time download required)
# AQR factor data must be downloaded once before running this notebook:
#
# ```python
# from ml4t.data.providers import AQRFactorProvider
# AQRFactorProvider.download()  # Downloads ~50MB of Excel files from AQR
# ```
#
# Both data sources are freely available for academic and research use.

# %% [markdown]
# ## Setup

# %%
"""Factor Allocation Evidence - assess diversification across published factor returns."""

import importlib
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import structlog
from IPython.display import Markdown, display
from plotly.subplots import make_subplots
from scipy import stats

from utils import DATA_DIR
from utils.style import COLORS, ml4t_diverging

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

# The provider package imports optional Oanda code whose Python 3.14 syntax warnings are
# unrelated to these factor providers. Scope suppression to that one optional-package import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    provider_module = importlib.import_module("ml4t.data.providers")

AQRFactorProvider = provider_module.AQRFactorProvider
FamaFrenchProvider = provider_module.FamaFrenchProvider


def show_figure(figure: go.Figure) -> None:
    """Render Plotly plus PNG while containing legacy Kaleido deprecations."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        figure.show()


# %% tags=["parameters"]
# Production defaults - Papermill overrides for CI testing
# This notebook uses AQR/Fama-French data (no case study data);
# date range is governed by source-provider availability, no parameters needed.

# %% [markdown]
# ## Initialize Data Providers
#
# The ml4t-data library provides comprehensive access to factor data from both
# AQR Capital Management and Kenneth French's Data Library.

# %%
try:
    aqr = AQRFactorProvider(data_path=DATA_DIR / "factors" / "aqr")
except FileNotFoundError as err:
    raise RuntimeError(
        "AQR factor data is required for this notebook. Run AQRFactorProvider.download() first."
    ) from err

# Fetch French source archives afresh so the signed run exercises download and parsing
# rather than certifying a stale local cache hit. AQR's downloaded workbooks are the
# declared canonical inputs and are reconciled separately in the evidence bundle.
ff = FamaFrenchProvider(cache_path=DATA_DIR / "factors" / "fama-french", use_cache=False)

display(
    Markdown(
        f"The local AQR source exposes **{len(aqr.list_datasets())} datasets** across "
        f"**{len(aqr.list_categories())} categories**. The French provider exposes "
        f"**{len(ff.list_datasets())} datasets** across **{len(ff.list_categories())} "
        "categories**."
    )
)

# %% [markdown]
# These provider checks look mundane, but they matter because the notebook combines
# several sources with different histories and naming conventions. Verifying coverage
# up front avoids drawing strong conclusions from mismatched sample periods later on.

# %% [markdown]
# ## Load Core Factor Data
#
# We'll load the key factor datasets for our analysis:
#
# 1. **Fama-French factors**: FF3, FF5, Momentum (1926-present)
# 2. **AQR equity factors**: QMJ and BAB
# 3. **Cross-asset**: VME (1972-present), TSMOM (1985-present)
# 4. **Long-history**: Century of Factor Premia (1926-2024 in this source vintage)

# %%
# Fama-French core factors
ff3 = ff.fetch("ff3")  # Market, SMB, HML, RF
ff5 = ff.fetch("ff5")  # Adds RMW, CMA
mom = ff.fetch("mom")  # Momentum factor

# Join once-fetched source frames on their common monthly timestamps.
ff4 = ff3.join(mom, on="timestamp", how="inner")
ff6 = ff5.join(mom, on="timestamp", how="inner")

display(
    Markdown(
        f"French coverage spans **{len(ff3):,} FF3 months** "
        f"({ff3['timestamp'].min():%Y-%m} to {ff3['timestamp'].max():%Y-%m}), "
        f"**{len(ff5):,} FF5 months**, and **{len(mom):,} momentum months**."
    )
)

# %%
# AQR factors (if available)
qmj = aqr.fetch("qmj_factors")
bab = aqr.fetch("bab_factors")
vme = aqr.fetch("vme_factors")
tsmom = aqr.fetch("tsmom")
century = aqr.fetch("century_premia")

display(
    Markdown(
        f"AQR coverage includes **{len(qmj):,} QMJ months**, **{len(bab):,} BAB months**, "
        f"**{len(vme):,} VME months**, **{len(tsmom):,} TSMOM months**, and "
        f"**{len(century):,} Century months** through {century['timestamp'].max():%Y-%m}."
    )
)

# %% [markdown]
# ## Return Units and Conventions
#
# The `ml4t.data.providers` library returns all factor data as **monthly decimals**:
# - `0.01` = 1% monthly return
# - `FamaFrenchProvider`: Divides raw French data (percent) by 100 on fetch
# - `AQRFactorProvider`: Returns data as-is (already decimal in source files)
#
# These are **current-vintage published research series**. Providers can revise,
# backfill, or extend their histories. Observation timestamps identify return months,
# not the dates on which the final research series became available to investors.
# The notebook therefore presents historical evidence, not a live-vintage backtest.
#
# **Terminology**:
# - `Mkt-RF`: Market excess return (market return minus risk-free rate)
# - `HML`, `SMB`, `MOM`: Long-short factor portfolios (already excess returns)
# - Missing values: Excluded from all calculations via `dropna()`

# %%
# Convert to pandas for analysis (provider returns Polars)
ff3_pd = ff3.to_pandas().set_index("timestamp")
mom_pd = mom.to_pandas().set_index("timestamp")
ff4_pd = ff4.to_pandas().set_index("timestamp")

# Sanity check: verify decimal units (fail if provider has a bug)
for col in ["Mkt-RF", "HML", "SMB", "MOM"]:
    if col in ff4_pd.columns:
        median_abs = ff4_pd[col].dropna().abs().median()
        assert median_abs < 0.20, (
            f"{col}: median |r| = {median_abs:.2f} suggests percent units (provider bug)"
        )

display(
    Markdown(
        f"The combined French panel contains **{len(ff4_pd):,} monthly observations** from "
        f"**{ff4_pd.index.min():%Y-%m} through {ff4_pd.index.max():%Y-%m}**."
    )
)

# %% [markdown]
# The unit sanity check is worth keeping in a teaching notebook because factor datasets
# are notorious for mixing percent and decimal conventions. A silent unit error would
# completely distort every Sharpe ratio and t-statistic that follows.

# %% [markdown]
# ## Reference Data: NBER Recessions
#
# Source: NBER Business Cycle Dating Committee (https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions).
# The static list includes NBER recessions through the 2020 contraction. Future
# contractions require an explicit source update.

# %%
# NBER recession dates (peak to trough)
RECESSIONS = [
    ("1929-08-01", "1933-03-01"),  # Great Depression
    ("1937-05-01", "1938-06-01"),  # 1937 Recession
    ("1945-02-01", "1945-10-01"),  # Post-WWII
    ("1948-11-01", "1949-10-01"),  # 1948 Recession
    ("1953-07-01", "1954-05-01"),  # 1953 Recession
    ("1957-08-01", "1958-04-01"),  # 1957 Recession
    ("1960-04-01", "1961-02-01"),  # 1960 Recession
    ("1969-12-01", "1970-11-01"),  # 1969 Recession
    ("1973-11-01", "1975-03-01"),  # Oil Crisis
    ("1980-01-01", "1980-07-01"),  # 1980 Recession
    ("1981-07-01", "1982-11-01"),  # Early 80s Recession
    ("1990-07-01", "1991-03-01"),  # 1990 Recession
    ("2001-03-01", "2001-11-01"),  # Dot-com Bust
    ("2007-12-01", "2009-06-01"),  # Global Financial Crisis
    ("2020-02-01", "2020-04-01"),  # COVID-19
]

# Major crisis periods for detailed analysis
CRISES = {
    "1987 Black Monday": ("1987-10-01", "1987-10-31"),
    "1998 LTCM": ("1998-08-01", "1998-10-31"),
    "2000-02 Dot-com": ("2000-03-01", "2002-10-31"),
    "2008-09 GFC": ("2007-10-01", "2009-03-31"),
    "2020 COVID": ("2020-02-01", "2020-03-31"),
    "2022 Inflation": ("2022-01-01", "2022-10-31"),
}

# %% [markdown]
# ---
#
# # Part 1: A Century of Factor Performance
#
# ## Are Factor Premia Real?
#
# One of the strongest criticisms of factor investing is that factor premia were discovered
# through data mining on the 1963-1990 period. The **Century of Factor Premia** dataset
# from AQR addresses this with a history beginning in 1926, much of which predates
# the academic publication of these factors.
#
# > "The fact that value and momentum premia exist in data that predates their discovery
# > provides compelling pre-publication evidence." - Ilmanen et al. (2021)
#
# **Important distinction**: This is "pre-discovery" or "pre-publication" evidence, not
# strictly "out-of-sample" in the workflow sense (which would require pre-registration
# before seeing any data). The evidence is strong because it's less susceptible to
# post hoc window selection.

# %%
# Consistent semantic emphasis across figures. Figures with more than five factors
# use one focal color plus neutral context rather than a categorical rainbow.
factor_colors = {
    "Mkt-RF": COLORS["blue"],
    "HML": COLORS["copper"],
    "SMB": COLORS["neutral"],
    "MOM": COLORS["amber"],
    "Value": COLORS["copper"],
    "Momentum": COLORS["amber"],
    "Carry": COLORS["positive"],
    "Defensive": COLORS["slate"],
}

labels = {
    "Mkt-RF": "Market (Mkt-RF)",
    "HML": "Value (HML)",
    "SMB": "Size (SMB)",
    "MOM": "Momentum (MOM)",
    "RMW": "Profitability (RMW)",
    "CMA": "Investment (CMA)",
    "QMJ": "Quality (QMJ)",
    "BAB": "Low-Vol (BAB)",
    "TSMOM": "Trend (TSMOM)",
}

# %%
cum_returns = (1 + ff4_pd[["Mkt-RF", "HML", "SMB", "MOM"]]).cumprod()
terminal_growth = cum_returns.iloc[-1]

fig = go.Figure()

for col in ["Mkt-RF", "MOM", "HML", "SMB"]:
    fig.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=cum_returns[col],
            name=labels.get(col, col),
            line=dict(color=factor_colors[col], width=2),
            hovertemplate="%{x|%Y-%m}<br>%{y:.2f}x<extra></extra>",
        )
    )

# %% [markdown]
# ### Add recession context
#
# Recession shading separates secular compounding from crisis behavior.

# %%
# Add recession bands
for start, end in RECESSIONS:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    if start_dt >= cum_returns.index.min():
        fig.add_vrect(
            x0=start_dt,
            x1=end_dt,
            fillcolor=COLORS["silver_muted"],
            opacity=0.35,
            layer="below",
            line_width=0,
        )

wealth_tick_values = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000]
wealth_tick_text = [f"{value:g}x" for value in wealth_tick_values]
fig.update_layout(
    title=(
        f"Market and momentum dominate long-run factor wealth<br><sup>Growth of $1 in "
        f"published long-short factor returns, {cum_returns.index.min():%Y-%m} to "
        f"{cum_returns.index.max():%Y-%m}; log scale; shaded bands are NBER recessions</sup>"
    ),
    xaxis_title="Month",
    yaxis_title="Growth of $1 (log scale)",
    yaxis_type="log",
    yaxis=dict(tickmode="array", tickvals=wealth_tick_values, ticktext=wealth_tick_text),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    hovermode="x unified",
    height=500,
)

show_figure(fig)

# %% [markdown]
# ### What the wealth paths establish
#
# The market, value, size, and momentum series are published long-short research
# portfolios. Their full-history wealth paths establish historical magnitude, not
# implementable investor returns. The later inference and period-comparison sections
# test whether the apparent premia are statistically distinguishable and whether SMB
# weakened after its 1981 publication.
#
# > **Important**: These are **long-short factor portfolio returns**, not tradable
# > strategies. The cumulative return charts above represent theoretical wealth paths
# > assuming: (1) perfect rebalancing, (2) zero transaction costs, (3) unlimited
# > leverage capacity, and (4) no financing costs for short positions. Actual
# > implementation of factor strategies involves significant slippage, financing
# > costs, and capacity constraints. Use this data as evidence about published factor
# > returns, not as a literal investable wealth trajectory.

# %%
display(
    Markdown(
        f"Over the shared French history, $1 grows to **${terminal_growth['Mkt-RF']:,.0f}** "
        f"in the market factor and **${terminal_growth['MOM']:,.0f}** in momentum. "
        "Those magnitudes exclude implementation frictions and do not establish that a reader "
        "could have earned the same paths."
    )
)

# %% [markdown]
# ## Century of Factor Premia: Pre-Publication Evidence
#
# The AQR "Century of Factor Premia" dataset provides pre-discovery evidence for
# factor premia. Much of this data (1926-1963) predates the academic publication
# of factors, making it less susceptible to post hoc window selection.
#
# Note: "Pre-discovery" is more accurate than "out-of-sample" here. True out-of-sample
# would require pre-registration of the strategy before seeing any data.

# %%
century_pd = century.to_pandas().set_index("timestamp")
century_columns = [
    "All asset classes Value",
    "All asset classes Momentum",
    "All asset classes Carry",
    "All asset classes Defensive",
]
assert set(century_columns) <= set(century_pd.columns)
century_factors = century_pd[century_columns].dropna()
cum_century = (1 + century_factors).cumprod()

display(
    Markdown(
        f"The aggregate Century panel contributes **{len(century_factors):,} common months** "
        f"from **{century_factors.index.min():%Y-%m} through "
        f"{century_factors.index.max():%Y-%m}**."
    )
)

# %% [markdown]
# The four exact aggregate columns keep the comparison aligned with the narrative.
# They combine the underlying stock-selection and macro sleeves rather than mixing
# regions or showing multiple variants of the same factor.

# %%
fig = go.Figure()

for col in cum_century.columns:
    short_name = col.removeprefix("All asset classes ")
    fig.add_trace(
        go.Scatter(
            x=cum_century.index,
            y=cum_century[col],
            name=short_name,
            line=dict(color=factor_colors[short_name], width=2),
            hovertemplate="%{x|%Y-%m}<br>%{y:.2f}x<extra></extra>",
        )
    )

fig.update_layout(
    title=(
        "Four aggregate premia persist across the Century sample"
        f"<br><sup>AQR published factor returns, {cum_century.index.min():%Y-%m} to "
        f"{cum_century.index.max():%Y-%m}; pre-publication history is not a "
        "preregistered holdout</sup>"
    ),
    xaxis_title="Month",
    yaxis_title="Growth of $1 (log scale)",
    yaxis_type="log",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    height=500,
)
show_figure(fig)

# %% [markdown]
# ---
#
# ## Part 1 Summary: What We've Established (and What We Haven't)
#
# **EXISTENCE EVIDENCE**
# - Factor premia exist in long-history data predating their publication
# - The data is less susceptible to post hoc window selection
# - Multiple factors (value, momentum, carry, defensive) show positive premia
#
# **NOT YET ADDRESSED** (requires separate analysis):
# - **Transaction costs**: Turnover, bid-ask spreads, market impact
# - **Financing costs**: Short rebates, leverage financing, margin requirements
# - **Capacity constraints**: How much capital can these strategies absorb?
# - **Implementation drag**: Rebalancing timing, corporate actions, index changes
#
# **The gap between "evidence of premium" and "tradable strategy" is addressed in**:
# - Chapter 16: Strategy Simulation (transaction cost modeling)
# - Chapter 17: Portfolio Construction (leverage constraints)
# - Chapter 18: Transaction Costs (market impact)
#
# ---

# %% [markdown]
# ---
#
# # Part 2: Statistical Rigor: Do Factors Pass Significance Tests?
#
# Harvey, Liu & Zhu (2016) propose a stricter statistical bar for factor discovery:
# given hundreds of factors tested in the literature, a **t-stat > 3.0** (not 2.0)
# is needed to account for multiple testing.
#
# > "Most claimed research findings in financial economics are likely false."
# > - Harvey, Liu & Zhu (2016)
#
# **Important caveats on the t > 3 threshold**:
#
# 1. **It applies to mean return t-stats** (HAC-adjusted), not Sharpe ratio significance
# 2. **It's a proposed bar for cross-sectional factor discovery**, not a universal law
# 3. **The appropriate threshold depends on context**: pre-registered tests vs. data mining
# 4. **We apply it here conservatively** to the Newey-West HAC t-stat of mean excess returns


# %% [markdown]
# ### Mean Return t-statistic (Newey-West HAC)
#
# Harvey et al. (2016) propose t > 3.0 as a stricter bar for factor discovery.
# This function computes the Newey-West HAC t-statistic for mean excess returns,
# which is what the Harvey threshold applies to.


# %%
def calculate_mean_return_tstat(returns, periods_per_year=12):
    """Calculate the Newey-West mean-return t-statistic used in the Harvey test."""
    if hasattr(returns, "values"):
        returns = returns.values
    returns = returns[~np.isnan(returns)]

    n = len(returns)
    if n < 12:
        return {"mean_tstat": np.nan, "mean_return": np.nan, "nw_se_monthly": np.nan}

    mean_ret = np.mean(returns)

    # Newey-West lag selection
    max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    max_lag = max(1, min(max_lag, n // 4))

    # Compute Newey-West HAC variance
    resid = returns - mean_ret
    gamma_0 = np.sum(resid**2) / n

    gamma_sum = 0
    for j in range(1, max_lag + 1):
        weight = 1 - j / (max_lag + 1)  # Bartlett kernel
        gamma_j = np.sum(resid[j:] * resid[:-j]) / n
        gamma_sum += 2 * weight * gamma_j

    nw_var = (gamma_0 + gamma_sum) / n
    nw_se = np.sqrt(max(nw_var, 1e-10))

    mean_tstat = mean_ret / nw_se if nw_se > 0 else 0

    # Annualization: mean_return scales by periods_per_year; nw_se kept monthly.
    # Inference uses mean_tstat (computed on monthly data), not annualized SE.
    return {
        "mean_tstat": mean_tstat,
        "mean_return": mean_ret * periods_per_year,
        "nw_se_monthly": nw_se,
    }


# %% [markdown]
# ### Sharpe Ratio Uncertainty
#
# The Sharpe ratio t-statistic is **different** from the Harvey threshold. The
# approximation below rescales both the monthly estimate and its standard error to
# annual units, then inflates uncertainty for return autocorrelation. It is a
# transparent Lo-style diagnostic, not the full covariance expression in Lo (2002).


# %%
def compute_autocorrelation_adjustment(returns, max_lag: int) -> tuple[list[float], float]:
    autocorrs = []
    for k in range(1, max_lag + 1):
        if len(returns) > k + 1:
            rho_k = np.corrcoef(returns[:-k], returns[k:])[0, 1]
            autocorrs.append(np.clip(rho_k, -0.95, 0.95))
        else:
            autocorrs.append(0.0)

    lr_var_factor = 1.0
    for k, rho_k in enumerate(autocorrs, start=1):
        weight = 1 - k / (max_lag + 1)
        lr_var_factor += 2 * weight * rho_k
    return autocorrs, max(lr_var_factor, 0.1)


# %% [markdown]
# Reuse the autocorrelation adjustment inside the Sharpe function so the notebook can
# keep the statistical assumptions explicit without burying them in one large cell.


# %%
def calculate_sharpe_stats(returns, periods_per_year=12):
    """Calculate annualized Sharpe statistics with a serial-correlation adjustment."""
    if hasattr(returns, "values"):
        returns = returns.values
    returns = returns[~np.isnan(returns)]

    n = len(returns)
    monthly_mean = np.mean(returns)
    monthly_std = np.std(returns, ddof=1)
    mean_ret = monthly_mean * periods_per_year
    std_ret = monthly_std * np.sqrt(periods_per_year)
    monthly_sharpe = monthly_mean / monthly_std if monthly_std > 0 else 0
    sharpe = monthly_sharpe * np.sqrt(periods_per_year)

    max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    max_lag = max(1, min(max_lag, n // 4))
    autocorrs, lr_var_factor = compute_autocorrelation_adjustment(returns, max_lag)
    se_monthly = np.sqrt((1 + 0.5 * monthly_sharpe**2) / n)
    se_sharpe = se_monthly * np.sqrt(lr_var_factor * periods_per_year)
    sharpe_tstat = sharpe / se_sharpe if se_sharpe > 0 else 0

    ci_lower = sharpe - 1.96 * se_sharpe
    ci_upper = sharpe + 1.96 * se_sharpe
    p_value = 2 * (1 - stats.norm.cdf(abs(sharpe_tstat)))
    mean_stats = calculate_mean_return_tstat(returns, periods_per_year)
    rho1 = autocorrs[0] if autocorrs else 0.0

    return {
        "sharpe": sharpe,
        "sharpe_tstat": sharpe_tstat,
        "t_stat": mean_stats["mean_tstat"],
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ann_return": mean_ret,
        "ann_vol": std_ret,
        "n_months": n,
        "autocorr": rho1,
        "lr_var_factor": lr_var_factor,
        "sharpe_se": se_sharpe,
    }


# %% [markdown]
# ### Maximum Drawdown
#
# Simple utility for computing peak-to-trough drawdown.


# %%
def calculate_max_drawdown(returns):
    """Calculate maximum drawdown from return series."""
    if hasattr(returns, "values"):
        returns = returns.values
    returns = returns[~np.isnan(returns)]
    cum_returns = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / rolling_max - 1
    return np.min(drawdowns)


# %% [markdown]
# ### Compute Factor Statistics
#
# Apply the statistical functions to all available factors from Fama-French and AQR.

# %%
# Calculate statistics for all factors
factor_stats = {}

# French factors use each source's full available history for inference. The
# inner-joined FF4 panel remains appropriate only for shared-history comparisons.
for col in ["Mkt-RF", "HML", "SMB"]:
    factor_stats[col] = calculate_sharpe_stats(ff3_pd[col].dropna())
factor_stats["MOM"] = calculate_sharpe_stats(mom_pd["MOM"].dropna())

# FF5 additional factors (convert once, reuse)
ff5_pd = ff5.to_pandas().set_index("timestamp")
for col in ["RMW", "CMA"]:
    if col in ff5_pd.columns:
        factor_stats[col] = calculate_sharpe_stats(ff5_pd[col].dropna())

# AQR factors
if qmj is not None:
    qmj_pd = qmj.to_pandas().set_index("timestamp")
    if "USA" in qmj_pd.columns:
        factor_stats["QMJ"] = calculate_sharpe_stats(qmj_pd["USA"].dropna())
    elif "Global" in qmj_pd.columns:
        factor_stats["QMJ"] = calculate_sharpe_stats(qmj_pd["Global"].dropna())

if bab is not None:
    bab_pd = bab.to_pandas().set_index("timestamp")
    if "USA" in bab_pd.columns:
        factor_stats["BAB"] = calculate_sharpe_stats(bab_pd["USA"].dropna())
    elif "Global" in bab_pd.columns:
        factor_stats["BAB"] = calculate_sharpe_stats(bab_pd["Global"].dropna())

# %%
stats_df = pd.DataFrame(factor_stats).T
stats_df["significant_harvey"] = stats_df["t_stat"] > 3.0
factor_order = ["Mkt-RF", "MOM", "HML", "RMW", "CMA", "QMJ", "BAB", "SMB"]
factor_order = [f for f in factor_order if f in factor_stats]

# %% [markdown]
# The Harvey threshold belongs on the mean-return t-statistic itself. Bars above
# three clear the conservative discovery screen; annualized Sharpe remains in the
# hover text as a separate economic-magnitude diagnostic.

# %%
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=[labels.get(factor, factor) for factor in factor_order],
        y=[factor_stats[factor]["t_stat"] for factor in factor_order],
        marker_color=[
            COLORS["blue"] if factor_stats[factor]["t_stat"] > 3 else COLORS["neutral"]
            for factor in factor_order
        ],
        customdata=np.array([[factor_stats[factor]["sharpe"]] for factor in factor_order]),
        hovertemplate="<b>%{x}</b><br>Mean t (NW): %{y:.2f}<br>Annualized Sharpe: "
        "%{customdata[0]:.2f}<extra></extra>",
    )
)
fig.add_hline(y=3, line_dash="dash", line_color=COLORS["amber"], line_width=2)
fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)

fig.update_layout(
    title=(
        f"{int(stats_df['significant_harvey'].sum())} of {len(stats_df)} factors clear the "
        "conservative t > 3 screen<br><sup>Newey-West HAC t-statistics for mean monthly "
        "returns; sample histories differ by factor</sup>"
    ),
    xaxis_title="Published factor portfolio",
    yaxis_title="Mean-return t-statistic (Newey-West)",
    showlegend=False,
    height=450,
)

show_figure(fig)

# %% [markdown]
# ### Test the SMB decay claim directly
#
# A full-sample SMB statistic cannot establish post-publication decay. Splitting at
# the 1981 publication year makes the comparison visible while remaining explicitly
# ex-post and descriptive.

# %%
smb_returns = ff3_pd["SMB"].dropna()
smb_periods = {
    "Pre-1981": smb_returns[smb_returns.index < "1981-01-01"],
    "1981-present": smb_returns[smb_returns.index >= "1981-01-01"],
}
smb_period_stats = {name: calculate_sharpe_stats(values) for name, values in smb_periods.items()}

fig = go.Figure(
    go.Bar(
        x=list(smb_period_stats),
        y=[stats["ann_return"] * 100 for stats in smb_period_stats.values()],
        marker_color=[COLORS["blue"], COLORS["neutral"]],
        text=[f"t = {stats['t_stat']:.2f}" for stats in smb_period_stats.values()],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Annualized mean: %{y:.2f}%<br>%{text}<extra></extra>",
    )
)
fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)
fig.update_layout(
    title=(
        "SMB weakens after its 1981 publication"
        "<br><sup>Annualized mean monthly return; labels show Newey-West t-statistics; "
        "ex-post period split</sup>"
    ),
    xaxis_title="Sample period",
    yaxis_title="Annualized mean return (%)",
    height=420,
)
show_figure(fig)

# %%
display(
    Markdown(
        f"SMB's annualized mean falls from **{smb_period_stats['Pre-1981']['ann_return']:.1%}** "
        f"before 1981 (t = {smb_period_stats['Pre-1981']['t_stat']:.2f}) to "
        f"**{smb_period_stats['1981-present']['ann_return']:.1%}** afterward "
        f"(t = {smb_period_stats['1981-present']['t_stat']:.2f}). The split was chosen ex post, "
        "so it documents decay rather than proving publication caused it."
    )
)

# %% [markdown]
# ---
#
# # Part 3: Value and Momentum Everywhere
#
# Asness, Moskowitz & Pedersen (2013) document value and momentum premia not just in US
# equities, but across **8 different asset classes globally** - a result that mitigates
# the single-market data-mining critique applied to factor work on US equities alone.
#
# > "We find consistent value and momentum return premia across eight diverse markets
# > and asset classes." - Asness, Moskowitz & Pedersen (2013)

# %%
vme_pd = vme.to_pandas().set_index("timestamp")
display(
    Markdown(
        f"The VME panel contains **{len(vme_pd):,} months** and "
        f"**{len(vme_pd.columns)} published series**."
    )
)

# %% [markdown]
# Map AQR VME column suffixes (`VALLS_VME_XX90`, `MOMLS_VME_XX90`) to
# display labels. `EQ`, `FX`, `FI`, `COM` cover cross-asset buckets;
# `US90`, `UK90`, `ROE90`, `JP90` cover regional equity blocks.

# %%
asset_class_map = {
    "US90": "US Equities",
    "UK90": "UK Equities",
    "ROE90": "Europe ex-UK",
    "JP90": "Japan",
    "EQ": "All Equities",
    "FX": "Currencies",
    "FI": "Fixed Income",
    "COM": "Commodities",
}
vme_stats = []

# %% [markdown]
# Aggregate value/momentum row (the "Everywhere" line) - uses the bundled
# `VAL` / `MOM` columns rather than any single asset class.

# %%
if "VAL" in vme_pd.columns and "MOM" in vme_pd.columns:
    val_data = vme_pd["VAL"].dropna()
    mom_data = vme_pd["MOM"].dropna()
    if len(val_data) > 12 and len(mom_data) > 12:
        common_idx = val_data.index.intersection(mom_data.index)
        val_stats = calculate_sharpe_stats(val_data.loc[common_idx])
        mom_stats = calculate_sharpe_stats(mom_data.loc[common_idx])
        corr = val_data.loc[common_idx].corr(mom_data.loc[common_idx])
        vme_stats.append(
            {
                "Asset Class": "EVERYWHERE",
                "Value SR": val_stats["sharpe"],
                "Value t": val_stats["t_stat"],
                "Momentum SR": mom_stats["sharpe"],
                "Momentum t": mom_stats["t_stat"],
                "Val-Mom Corr": corr,
                "Common Months": len(common_idx),
            }
        )

# %%
# Then add by asset class.
for suffix, display_name in asset_class_map.items():
    val_col = f"VALLS_VME_{suffix}"
    mom_col = f"MOMLS_VME_{suffix}"
    assert val_col in vme_pd.columns and mom_col in vme_pd.columns
    val_data = vme_pd[val_col].dropna()
    mom_data = vme_pd[mom_col].dropna()
    common_idx = val_data.index.intersection(mom_data.index)
    assert len(common_idx) > 12
    val_stats = calculate_sharpe_stats(val_data.loc[common_idx])
    mom_stats = calculate_sharpe_stats(mom_data.loc[common_idx])
    vme_stats.append(
        {
            "Asset Class": display_name,
            "Value SR": val_stats["sharpe"],
            "Value t": val_stats["t_stat"],
            "Momentum SR": mom_stats["sharpe"],
            "Momentum t": mom_stats["t_stat"],
            "Val-Mom Corr": val_data.loc[common_idx].corr(mom_data.loc[common_idx]),
            "Common Months": len(common_idx),
        }
    )

# %%
assert len(vme_stats) == 9
vme_df = pd.DataFrame(vme_stats)
class_corr = vme_df.loc[vme_df["Asset Class"] != "EVERYWHERE", "Val-Mom Corr"]
median_vme_corr = float(class_corr.median())
min_vme_corr = float(class_corr.min())
max_vme_corr = float(class_corr.max())

# %% [markdown]
# The cross-asset comparison matters more than any single region. The figure below
# separates economic magnitude (Sharpe) from the diversification statistic (correlation).

# %% [markdown]
# Two-panel frame: Sharpe ratios by asset class
# on the left, value-momentum correlation on the right.

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Annualized Sharpe", "Value-Momentum Correlation"),
)

# %% [markdown]
# Left panel - Sharpe-ratio bars for Value and Momentum side-by-side.

# %%
_ = fig.add_trace(
    go.Bar(
        x=vme_df["Asset Class"],
        y=vme_df["Value SR"],
        name="Value",
        marker_color=factor_colors["Value"],
    ),
    row=1,
    col=1,
)
_ = fig.add_trace(
    go.Bar(
        x=vme_df["Asset Class"],
        y=vme_df["Momentum SR"],
        name="Momentum",
        marker_color=factor_colors["Momentum"],
    ),
    row=1,
    col=1,
)

# %% [markdown]
# Right panel - value-momentum correlation, with blue for negative
# (diversifying) observations and red for positive observations.

# %%
_ = fig.add_trace(
    go.Bar(
        x=vme_df["Asset Class"],
        y=vme_df["Val-Mom Corr"],
        name="Correlation",
        marker_color=[
            COLORS["negative"] if value > 0 else COLORS["blue"] for value in vme_df["Val-Mom Corr"]
        ],
        showlegend=False,
    ),
    row=1,
    col=2,
)

# %%
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], row=1, col=2)
fig.update_layout(
    title=(
        "Value and momentum diversify in every VME asset class"
        f"<br><sup>Monthly published returns; median correlation {median_vme_corr:.2f}; "
        f"range {min_vme_corr:.2f} to {max_vme_corr:.2f}</sup>"
    ),
    barmode="group",
    height=480,
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.45),
)
fig.update_xaxes(title_text="Asset class", tickangle=-30)
fig.update_yaxes(title_text="Annualized Sharpe ratio", row=1, col=1)
_ = fig.update_yaxes(title_text="Correlation", range=[-1, 1], row=1, col=2)

# %%
show_figure(fig)

# %%
display(
    Markdown(
        f"Across the eight regional and cross-asset sleeves, value-momentum correlation ranges "
        f"from **{min_vme_corr:.2f} to {max_vme_corr:.2f}**, with a median of "
        f"**{median_vme_corr:.2f}**. The separate aggregate `EVERYWHERE` series has correlation "
        f"**{vme_df.loc[vme_df['Asset Class'] == 'EVERYWHERE', 'Val-Mom Corr'].iloc[0]:.2f}**."
    )
)

# %% [markdown]
# ### The Diversification Benefit
#
# A crucial finding is that **value and momentum are negatively correlated** in
# every VME sleeve shown. This creates diversification potential, but the published
# long-short returns do not include a reader's implementation costs or constraints.
#
# The negative correlation arises because:
# - Value buys beaten-down assets (past losers that are now cheap)
# - Momentum buys recent winners (past winners with positive trend)
#
# These are opposite bets on mean reversion vs. trend continuation.

# %% [markdown]
# ---
#
# # Part 4: Factor Correlations and Diversification

# %%
# Build combined factor dataset for correlation analysis
# Use common period when all factors are available (1972+)

# Start with Fama-French 6 factors
# Cast timestamp to microseconds for consistent join precision
combined_pl = ff6.select(["timestamp", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]).with_columns(
    pl.col("timestamp").cast(pl.Datetime("us"))
)

# Add the AQR factors to the common-period panel.
if qmj is not None and "USA" in qmj.columns:
    qmj_usa = qmj.select([pl.col("timestamp"), pl.col("USA").alias("QMJ")]).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us"))
    )
    combined_pl = combined_pl.join(qmj_usa, on="timestamp", how="full", coalesce=True)

if bab is not None and "USA" in bab.columns:
    bab_usa = bab.select([pl.col("timestamp"), pl.col("USA").alias("BAB")]).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us"))
    )
    combined_pl = combined_pl.join(bab_usa, on="timestamp", how="full", coalesce=True)

# Filter to common period (1972+) and drop nulls
combined_pl = combined_pl.filter(pl.col("timestamp") >= pl.date(1972, 1, 1)).drop_nulls()

display(
    Markdown(
        f"The common eight-factor panel contains **{len(combined_pl):,} months** from "
        f"**{combined_pl['timestamp'].min():%Y-%m} through "
        f"{combined_pl['timestamp'].max():%Y-%m}**."
    )
)

# Convert to pandas only for correlation and visualization
combined_data = combined_pl.to_pandas().set_index("timestamp")

# %%
# Compute the common-period factor correlation matrix.
corr_matrix = combined_data.corr()

# Create labels for display
corr_labels = {col: labels.get(col, col) for col in corr_matrix.columns}
corr_display = corr_matrix.copy()
corr_display.index = [corr_labels.get(c, c) for c in corr_display.index]
corr_display.columns = [corr_labels.get(c, c) for c in corr_display.columns]

# %% [markdown]
# Mask the duplicate upper triangle and use a fixed diverging scale centered at zero.

# %%
corr_plot = corr_display.mask(np.triu(np.ones(corr_display.shape, dtype=bool), k=1))
corr_text = corr_plot.apply(
    lambda column: column.map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
)
fig = go.Figure(
    go.Heatmap(
        z=corr_plot.to_numpy(),
        x=corr_plot.columns,
        y=corr_plot.index,
        text=corr_text.to_numpy(),
        texttemplate="%{text}",
        colorscale=ml4t_diverging(),
        zmin=-1,
        zmax=1,
        zmid=0,
        colorbar=dict(title="Correlation"),
        hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
    )
)

# %% [markdown]
# Add a message-first title and explicit dimensions.

# %%
_ = fig.update_layout(
    title=(
        f"Value and momentum remain diversifying in the common factor panel"
        f"<br><sup>Pairwise correlations on {len(combined_data):,} common monthly rows, "
        f"{combined_data.index.min():%Y-%m} to {combined_data.index.max():%Y-%m}</sup>"
    ),
    height=560,
    margin=dict(l=135, r=90, b=120),
    xaxis_title="Factor",
    yaxis_title="Factor",
)

show_figure(fig)

# %% [markdown]
# ### Key Correlation Insights
#
# 1. **Value vs. Momentum**: The French common-period estimate appears in the visible
#    lower triangle. It is distinct from the stronger cross-asset VME estimates.
#
# 2. **Quality (QMJ) vs. Market**: QMJ has low market beta, providing defensive
#    characteristics during equity downturns.
#
# 3. **Low-Vol (BAB) vs. Market**: BAB is designed to be market-neutral but may still
#    have residual market exposure during extreme moves.

# %% [markdown]
# ---
#
# # Part 5: Crisis Performance: Who Provides "Crisis Alpha"?
#
# One of the most important questions for portfolio construction: **Which factors
# perform well during market crises?** True "crisis alpha" - positive returns during
# equity market drawdowns - is extremely valuable.
#
# These windows are selected ex post. They describe historical co-movement; they do
# not establish that a factor will insure a future crisis.

# %%
# Calculate crisis period returns
crisis_factors = ["Mkt-RF", "HML", "SMB", "MOM"]
if "QMJ" in combined_data.columns:
    crisis_factors.append("QMJ")
if "BAB" in combined_data.columns:
    crisis_factors.append("BAB")

crisis_returns = {}
for crisis_name, (start, end) in CRISES.items():
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    mask = (combined_data.index >= start_dt) & (combined_data.index <= end_dt)
    crisis_data = combined_data[mask]

    if len(crisis_data) > 0:
        crisis_returns[crisis_name] = {}
        for col in crisis_factors:
            if col in crisis_data.columns:
                data = crisis_data[col].dropna()
                if len(data) > 0:
                    cum_ret = (1 + data).prod() - 1
                    crisis_returns[crisis_name][col] = cum_ret

crisis_df = pd.DataFrame(crisis_returns).T

# %%
tsmom_pd = tsmom.to_pandas().set_index("timestamp")
assert "TSMOM" in tsmom_pd.columns
for crisis_name, (start, end) in CRISES.items():
    tsmom_crisis = tsmom_pd.loc[pd.to_datetime(start) : pd.to_datetime(end), "TSMOM"].dropna()
    if len(tsmom_crisis) > 0:
        crisis_df.loc[crisis_name, "TSMOM"] = (1 + tsmom_crisis).prod() - 1

# %% [markdown]
# A signed-return heatmap makes the crisis pattern legible without assigning seven
# categorical colors. Blank cells denote factor histories that had not started.

# %%
available_factors = [
    f for f in ["Mkt-RF", "MOM", "HML", "QMJ", "BAB", "TSMOM", "SMB"] if f in crisis_df.columns
]
crisis_plot = crisis_df[available_factors].rename(columns=labels) * 100
tsmom_positive = int((crisis_df["TSMOM"].dropna() > 0).sum())
tsmom_observed = int(crisis_df["TSMOM"].notna().sum())

fig = go.Figure(
    go.Heatmap(
        z=crisis_plot.to_numpy(),
        x=crisis_plot.columns,
        y=crisis_plot.index,
        text=np.round(crisis_plot.to_numpy(), 1),
        texttemplate="%{text:.1f}%",
        colorscale=ml4t_diverging(),
        zmid=0,
        colorbar=dict(title="Return (%)"),
        hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
    )
)

# %%
fig.update_layout(
    title=(
        f"TSMOM is positive in {tsmom_positive} of {tsmom_observed} observed crisis windows"
        "<br><sup>Ex-post cumulative monthly returns; blanks predate a factor's history; "
        "windows are descriptive, not prospective insurance tests</sup>"
    ),
    xaxis_title="Published factor portfolio",
    yaxis_title="Selected crisis window",
    height=520,
    margin=dict(l=105, r=75, b=100),
)
show_figure(fig)

# %% [markdown]
# **Interpretation**: Crisis performance is where correlations and premia become
# economically meaningful. A factor that compounds nicely on average but fails exactly
# when the rest of the portfolio is under stress is much less valuable in allocation.

# %% [markdown]
# ### Crisis Performance Observations
#
# 1. **Market (Mkt-RF)**: By definition, large negative returns during crises.
#
# 2. **Momentum (MOM)**: Mixed results. Provided crisis alpha in some events but
#    suffered the famous "momentum crash" in 2009 when the trend reversed violently.
#
# 3. **Value (HML)**: Generally negative during crises as cheap stocks get cheaper.
#    The 2020 COVID crash was particularly painful for value.
#
# 4. **Quality (QMJ)**: Its observed crisis returns vary by episode; the heatmap
#    shows when the historical "flight to quality" description does and does not fit.
#
# 5. **Trend (TSMOM)**: The aggregate published series is positive in most, but not
#    all, observed windows. Its 1985 start leaves earlier crises unmeasured.

# %% [markdown]
# ---
#
# # Part 6: Risk and Return in One View

# %% [markdown]
# ### Summary Helper
#
# The final chart uses one numeric row layout across French and AQR sources.


# %%
def append_summary_row(
    summary_stats: list[dict[str, object]],
    factor: str,
    source: str,
    returns: pd.Series,
    stats_dict: dict[str, float],
) -> None:
    """Append one numeric summary row for a factor series."""
    summary_stats.append(
        {
            "Factor": labels.get(factor, factor),
            "Source": source,
            "Period": f"{returns.index.min():%Y}-{returns.index.max():%Y}",
            "Ann. Return": stats_dict["ann_return"],
            "Ann. Volatility": stats_dict["ann_vol"],
            "Sharpe Ratio": stats_dict["sharpe"],
            "t-statistic": stats_dict["t_stat"],
            "Max Drawdown": calculate_max_drawdown(returns),
            "Skewness": returns.skew(),
            "Months": len(returns),
        }
    )


# %%
# Build comprehensive summary table
summary_stats = []

# Fama-French factors
for factor in ["Mkt-RF", "HML", "SMB", "MOM"]:
    if factor in factor_stats:
        source_data = mom_pd if factor == "MOM" else ff3_pd
        returns = source_data[factor].dropna()
        s = factor_stats[factor]

        append_summary_row(summary_stats, factor, "French", returns, s)

# FF5 additional factors (ff5_pd already converted above)
for factor in ["RMW", "CMA"]:
    if factor in factor_stats:
        returns = ff5_pd[factor].dropna()
        s = factor_stats[factor]

        append_summary_row(summary_stats, factor, "French", returns, s)

# AQR factors
for factor, data in [("QMJ", qmj), ("BAB", bab)]:
    if data is not None and factor in factor_stats:
        data_pd = data.to_pandas().set_index("timestamp")
        col = "USA" if "USA" in data_pd.columns else data_pd.columns[0]
        returns = data_pd[col].dropna()
        s = factor_stats[factor]

        append_summary_row(summary_stats, factor, "AQR", returns, s)

summary_df = pd.DataFrame(summary_stats).set_index("Factor")
label_positions = {
    "Market (Mkt-RF)": "middle left",
    "Momentum (MOM)": "top center",
    "Value (HML)": "top center",
    "Size (SMB)": "bottom center",
    "Profitability (RMW)": "top right",
    "Investment (CMA)": "bottom right",
    "Quality (QMJ)": "top right",
    "Low-Vol (BAB)": "top center",
}

# %% [markdown]
# A labeled risk-return map replaces the redundant summary dump. Comparisons remain
# historical and descriptive because factor histories begin on different dates.

# %%
fig = go.Figure(
    go.Scatter(
        x=summary_df["Ann. Volatility"] * 100,
        y=summary_df["Ann. Return"] * 100,
        mode="markers+text",
        text=summary_df.index,
        textposition=[label_positions[label] for label in summary_df.index],
        marker=dict(
            size=12,
            color=[
                COLORS["blue"] if value > 3 else COLORS["neutral"]
                for value in summary_df["t-statistic"]
            ],
        ),
        customdata=summary_df[["Sharpe Ratio", "t-statistic", "Max Drawdown", "Period"]],
        hovertemplate=(
            "<b>%{text}</b><br>Annualized return: %{y:.1f}%<br>Annualized volatility: "
            "%{x:.1f}%<br>Sharpe: %{customdata[0]:.2f}<br>Mean t (NW): "
            "%{customdata[1]:.2f}<br>Max drawdown: %{customdata[2]:.1%}<br>Period: "
            "%{customdata[3]}<extra></extra>"
        ),
    )
)
fig.update_layout(
    title=(
        "BAB and QMJ lead the historical factor risk-return trade-off"
        "<br><sup>Published monthly factor returns; blue markers clear mean t > 3; "
        "sample histories differ</sup>"
    ),
    xaxis_title="Annualized volatility (%)",
    yaxis_title="Annualized mean return (%)",
    height=500,
    margin=dict(l=75, r=75, b=80),
    showlegend=False,
)
show_figure(fig)

# %%
display(
    Markdown(
        f"The cross-asset VME evidence supplies the clearest allocation result: median "
        f"value-momentum correlation is **{median_vme_corr:.2f}** across eight sleeves. "
        f"The inference screen is more selective: **{int(stats_df['significant_harvey'].sum())} "
        f"of {len(stats_df)}** factor means exceed t = 3. TSMOM is positive in "
        f"**{tsmom_positive} of {tsmom_observed}** observed crisis windows, while the explicit "
        "SMB split documents weaker post-1981 performance without claiming publication caused it."
    )
)

# %% [markdown]
# ---
#
# ## Key Takeaways
#
# - **Diversification**: Value and momentum are negatively correlated in every
#   regional and cross-asset VME sleeve shown, making the pair a stronger allocation
#   building block than either factor alone.
# - **Inference**: Historical Sharpe ratios and conservative mean-return t-statistics
#   answer different questions. Long samples help, but they do not remove source
#   revision, publication, financing, turnover, or capacity concerns.
# - **Regime evidence**: TSMOM is the most frequent positive crisis companion in the
#   selected post-1985 windows, not universal insurance. SMB weakens after 1981 in an
#   ex-post split that documents decay without identifying its cause.
#
# **Next**: `08_library_comparison` compares portfolio-construction implementations.
# **Book**: Section 17.4 develops baseline allocators and factor diversification.
#
# ## References
#
# - Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere"
# - Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns"
# - Ilmanen et al. (2021), "How Do Factor Premia Vary Over Time?"
# - Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum"
# - Lo (2002), "The Statistics of Sharpe Ratios"
#
# Full bibliography and Data Library / AQR licensing details live in the
# chapter prose.
