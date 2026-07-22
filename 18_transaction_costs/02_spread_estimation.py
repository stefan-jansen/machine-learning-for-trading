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
# # Spread Estimation from Market Data
#
# **Docker image**: `ml4t`
#
# This notebook estimates bid-ask spreads from OHLCV data using two classical
# estimators, validates them against ground-truth microstructure data, and
# applies them across six OHLCV datasets (options analytics covered separately
# in NB03).
#
# **Learning Objectives:**
# - Implement the Corwin-Schultz (2012) high-low spread estimator
# - Implement the Roll (1984) serial covariance spread estimator
# - Validate both estimators against observed spreads from NASDAQ-100 minute bars
# - Estimate spreads across six OHLCV datasets and compare to setup.yaml assumptions
# - Examine how spreads vary with volatility regimes (VIX conditioning)
#
# **Book Reference:** Chapter 18, Section 18.3 (Spread Estimation)
#
# **Prerequisites:** Access to six OHLCV datasets; NASDAQ-100 minute bars
# with microstructure columns for validation.

# %%
"""Spread Estimation from Market Data — Corwin-Schultz & Roll estimators with validation."""

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import yaml
from _cost_analysis import corwin_schultz_spread, roll_spread

from data import (
    load_cme_futures,
    load_crypto_perps,
    load_etfs,
    load_fx_pairs,
    load_macro,
    load_nasdaq100_bars,
    load_sp500_daily_bars,
)
from utils.paths import get_case_study_source_dir

# %% tags=["parameters"]
MAX_SYMBOLS = 50  # Symbols per dataset for cross-asset comparison (50 is plenty)
NQ100_SYMBOLS = 12  # NASDAQ-100 symbols for microstructure validation (keep small!)
NQ100_START_DATE = "2021-10-01"  # Production validation window
NQ100_END_DATE = "2021-12-31"
CS_WINDOW = 20  # Corwin-Schultz averaging window
ROLL_WINDOW = 20  # Roll estimator window

# %% [markdown]
# ## 1. The Estimation Problem
#
# Only 2 of our datasets include observed bid-ask spreads:
# - **NASDAQ-100 minute bars**: `close_bid_price`, `close_ask_price`, `min_spread`, `max_spread`
# - **S&P 500 options**: `bid`, `ask` columns
#
# For the remaining datasets (ETFs, crypto, futures, FX, US equities), we
# must *estimate* spreads from price data alone. Two classical approaches:
#
# 1. **Corwin-Schultz (2012)**: Uses the relationship between daily high-low
#    ranges and the bid-ask spread. Key insight: daily highs tend to be at
#    the ask, and daily lows at the bid.
#
# 2. **Roll (1984)**: Uses negative serial covariance in returns caused by
#    the bid-ask bounce.

# %% [markdown]
# ## 2. Corwin-Schultz Estimator
#
# The high-low spread estimator decomposes the observed high-low range into a
# volatility component and a spread component.
#
# **Two-period estimator:**
#
# $$\beta = \mathbb{E}\left[\sum_{j=1}^{2} \ln\left(\frac{H_j}{L_j}\right)^2\right]$$
#
# $$\gamma = \ln\left(\frac{H_{[1,2]}}{L_{[1,2]}}\right)^2$$
#
# where $H_{[1,2]}$ and $L_{[1,2]}$ are the high and low over the combined two-period window.
#
# $$\alpha = \frac{\sqrt{2\beta} - \sqrt{\gamma}}{3 - 2\sqrt{2}} - \sqrt{\frac{\gamma}{3 - 2\sqrt{2}}}$$
#
# $$\hat{S} = \frac{2(e^{\alpha} - 1)}{1 + e^{\alpha}}$$
#
# Negative estimates are clamped to zero (occur when volatility dominates).

# %% [markdown]
# ## 3. Roll Estimator
#
# If bid-ask bounce is the dominant source of return serial correlation:
#
# $$\hat{S} = 2\sqrt{-\text{Cov}(\Delta p_t, \Delta p_{t-1})}$$
#
# Only defined when autocovariance is negative (the "efficient market" condition).
# Positive autocovariance (momentum) produces zero estimates.

# %% [markdown]
# ## 4. Ground Truth: NASDAQ-100 Observed Spreads
#
# We aggregate minute-level bid-ask data to daily observed spreads, then
# compare with our estimators.


# %%
def load_nq_microstructure() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Load raw and cleaned NASDAQ-100 microstructure data lazily."""
    raw_nq_lf = load_nasdaq100_bars(
        start_date=NQ100_START_DATE,
        end_date=NQ100_END_DATE,
        include_microstructure=True,
        lazy=True,
    ).select(
        "timestamp",
        "symbol",
        "volume",
        "close_bid_price",
        "close_ask_price",
        "high_trade_price",
        "low_trade_price",
        "first_trade_price",
        "last_trade_price",
    )
    nq_lf = (
        raw_nq_lf.filter(
            pl.col("close_bid_price").is_not_null()
            & pl.col("close_ask_price").is_not_null()
            & (pl.col("close_bid_price") > 0)
            & (pl.col("close_ask_price") > pl.col("close_bid_price"))
        )
        .with_columns(
            mid_price=((pl.col("close_bid_price") + pl.col("close_ask_price")) / 2),
            abs_spread=(pl.col("close_ask_price") - pl.col("close_bid_price")),
        )
        .with_columns(rel_spread=pl.col("abs_spread") / pl.col("mid_price"))
    )
    return raw_nq_lf, nq_lf


# %% [markdown]
# ### Restrict to the Most Liquid NASDAQ-100 Symbols

# %%
raw_nq_lf, nq_lf = load_nq_microstructure()

if NQ100_SYMBOLS > 0:
    top_symbols = (
        raw_nq_lf.group_by("symbol")
        .agg(pl.col("volume").sum())
        .sort("volume", descending=True)
        .head(NQ100_SYMBOLS)
        .collect()["symbol"]
        .to_list()
    )
    raw_nq_lf = raw_nq_lf.filter(pl.col("symbol").is_in(top_symbols))
    nq_lf = nq_lf.filter(pl.col("symbol").is_in(top_symbols))

# %% [markdown]
# **Interpretation**: Restricting the sample to the most liquid NASDAQ-100 names keeps
# validation focused on a realistic large-cap benchmark. It also prevents thin tail names
# from dominating the estimator comparison.

# %% [markdown]
# ### Aggregate Minute Spreads to Daily Observations

# %%
# Aggregate minute bars to one row per (symbol, trading date). Open / close
# use the first / last minute of the day; high / low are intraday extrema;
# observed_spread is volume-weighted across the day.
nq_daily_observed = (
    nq_lf.with_columns(date=pl.col("timestamp").dt.date())
    .sort(["symbol", "timestamp"])
    .group_by(["date", "symbol"])
    .agg(
        observed_spread=((pl.col("rel_spread") * pl.col("volume")).sum() / pl.col("volume").sum()),
        mean_spread=pl.col("rel_spread").mean(),
        median_spread=pl.col("rel_spread").median(),
        high=pl.col("high_trade_price").max(),
        low=pl.col("low_trade_price").min(),
        open=pl.col("first_trade_price").first(),
        close=pl.col("last_trade_price").last(),
        volume=pl.col("volume").sum(),
    )
    .sort(["symbol", "date"])
    .filter(
        pl.col("high").is_not_null()
        & pl.col("low").is_not_null()
        & (pl.col("high") > pl.col("low"))
        & (pl.col("low") > 0)
    )
    .collect()
)

if nq_daily_observed.is_empty():
    raise ValueError(
        "No NASDAQ-100 observed spreads found in the selected window. "
        "Adjust NQ100_START_DATE/NQ100_END_DATE to match available minute-bar data."
    )

print(
    f"Daily observed spreads ({NQ100_START_DATE} to {NQ100_END_DATE}): "
    f"{len(nq_daily_observed):,} symbol-days"
)
median_observed = nq_daily_observed["observed_spread"].median()
print(f"Median observed spread: {median_observed:.4f} ({median_observed * 10_000:.1f} bps)")

# %% [markdown]
# **Finding**: The observed NASDAQ-100 baseline anchors the entire notebook.
# If the median realized spread is only a few basis points, any estimator that
# regularly prints double-digit costs is too pessimistic for large-cap equities.

# %% [markdown]
# ### Apply Both Estimators to NASDAQ-100 Daily Data

# %%
# Apply Corwin-Schultz and Roll estimators per symbol
nq_estimated = (
    nq_daily_observed.sort(["symbol", "date"])
    .with_columns(
        cs_spread=corwin_schultz_spread(pl.col("high"), pl.col("low"), window=CS_WINDOW),
        roll_spread_est=roll_spread(pl.col("close"), window=ROLL_WINDOW),
    )
    .filter(
        pl.col("cs_spread").is_not_null()
        & pl.col("roll_spread_est").is_not_null()
        & pl.col("observed_spread").is_not_null()
    )
)

print(f"Matched observations: {len(nq_estimated):,}")
if len(nq_estimated) == 0:
    raise ValueError(
        "Spread estimators produced no matched observations. "
        "Reduce CS_WINDOW/ROLL_WINDOW or widen the NASDAQ-100 date range."
    )

# %% [markdown]
# **Finding**: The matched sample is the relevant validation set because it is
# the intersection where both ground truth and both estimators exist. Shrinking
# this sample too far makes the validation statistics unstable.

# %% [markdown]
# ## 5. Validation: Estimated vs Observed Spreads
#
# We aggregate to symbol-level medians and compare estimated vs observed.

# %%
# Symbol-level comparison
nq_validation = (
    nq_estimated.group_by("symbol")
    .agg(
        observed_bps=(pl.col("observed_spread").median() * 10_000),
        cs_bps=(pl.col("cs_spread").median() * 10_000),
        roll_bps=(pl.col("roll_spread_est").median() * 10_000),
        adv=pl.col("volume").mean(),
    )
    .sort("observed_bps")
)
if len(nq_validation) < 2:
    raise ValueError(
        "Need at least two symbols with matched observed and estimated spreads for validation."
    )

# %%
# Compute validation metrics
for est_col, name in [("cs_bps", "Corwin-Schultz"), ("roll_bps", "Roll")]:
    obs = nq_validation["observed_bps"].to_numpy()
    est = nq_validation[est_col].to_numpy()
    mask = np.isfinite(obs) & np.isfinite(est)
    obs, est = obs[mask], est[mask]

    corr = np.corrcoef(obs, est)[0, 1]
    mae = np.mean(np.abs(obs - est))
    bias = np.mean(est - obs)
    print(f"{name}: R²={corr**2:.3f}, MAE={mae:.1f} bps, Bias={bias:+.1f} bps")

# %% [markdown]
# **Finding**: These validation statistics tell us whether the estimators are merely
# directionally correlated with truth or actually close enough in level to support
# realistic backtests. Correlation without manageable bias is not sufficient.

# %%
# Validation scatter plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, est_col, name in [
    (axes[0], "cs_bps", "Corwin-Schultz"),
    (axes[1], "roll_bps", "Roll"),
]:
    obs = nq_validation["observed_bps"].to_numpy()
    est = nq_validation[est_col].to_numpy()
    mask = np.isfinite(obs) & np.isfinite(est)

    ax.scatter(obs[mask], est[mask], alpha=0.6, s=30)
    lims = [0, max(obs[mask].max(), est[mask].max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.5, label="45° line")
    ax.set_xlabel("Observed Spread (bps)")
    ax.set_ylabel("Estimated Spread (bps)")
    ax.set_title(f"{name} vs Observed (NASDAQ-100)")

    if mask.sum() > 1:
        corr = np.corrcoef(obs[mask], est[mask])[0, 1]
        ax.text(0.05, 0.92, f"R² = {corr**2:.3f}", transform=ax.transAxes, fontsize=11)
    ax.legend(loc="lower right")

fig.tight_layout()
fig.show()

# %% [markdown]
# **Finding**: On this short, large-cap-only daily sample both estimators
# overestimate the few-bps observed spreads and correlate only weakly with
# ground truth — the high-low and serial-covariance signals are swamped by
# volatility when true spreads are this tight. The estimators are far more
# useful for wider-spread asset classes (FX, crypto, less-liquid futures), as
# the cross-asset section below shows.

# %% [markdown]
# ## 6. Cross-Asset Spread Estimation
#
# We now apply both estimators to all six OHLCV datasets to build a
# cross-asset spread landscape.


# %%
def keep_top_symbols(df: pl.DataFrame, symbol_col: str) -> pl.DataFrame:
    """Optionally restrict a dataset to the most liquid symbols."""
    if MAX_SYMBOLS <= 0:
        return df
    top = df.group_by(symbol_col).agg(pl.col("volume").mean()).sort("volume", descending=True)
    return df.filter(pl.col(symbol_col).is_in(top.head(MAX_SYMBOLS)[symbol_col]))


# %% [markdown]
# ### Assemble the Cross-Asset Daily Dataset Map
#
# **Interpretation**: This helper pattern keeps the cross-asset comparison honest.
# Every dataset is reduced to the same daily OHLCV shape before we compare spread
# regimes across markets.

# %%
datasets = {}

# ETFs
datasets["ETFs"] = keep_top_symbols(load_etfs(), "symbol")

# Crypto perpetuals (8h bars)
datasets["Crypto Perps"] = keep_top_symbols(load_crypto_perps(frequency="8h"), "symbol")

# CME Futures (front month only). Spread estimators run on the roll-adjusted
# series (adj_* → bare OHLC): Roll's serial covariance would otherwise pick up
# roll gaps as spurious spread.
datasets["CME Futures"] = (
    keep_top_symbols(load_cme_futures(tenors=[0]), "product")
    .rename({"session_date": "timestamp", "product": "symbol"})
    .with_columns(
        pl.col("adj_open").alias("open"),
        pl.col("adj_high").alias("high"),
        pl.col("adj_low").alias("low"),
        pl.col("adj_close").alias("close"),
    )
)

# FX Pairs (daily)
fx = load_fx_pairs(frequency="daily")
if MAX_SYMBOLS > 0:
    fx = fx.filter(pl.col("symbol").is_in(fx["symbol"].unique().sort().head(MAX_SYMBOLS)))
datasets["FX Pairs"] = fx

# S&P 500 daily bars (representative US equities — smaller than load_us_equities)
sp500 = load_sp500_daily_bars()
if "date" in sp500.columns and "timestamp" not in sp500.columns:
    sp500 = sp500.rename({"date": "timestamp"})
if MAX_SYMBOLS > 0:
    top = sp500.group_by("symbol").agg(pl.col("volume").mean()).sort("volume", descending=True)
    sp500 = sp500.filter(pl.col("symbol").is_in(top.head(MAX_SYMBOLS)["symbol"]))
datasets["S&P 500 Equities"] = sp500.select(
    "timestamp", "symbol", "open", "high", "low", "close", "volume"
)

# NASDAQ-100 (aggregate minute → daily for estimators)
nq_daily = nq_daily_observed.select(
    pl.col("date").cast(pl.Datetime).alias("timestamp"),
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
datasets["NASDAQ-100"] = nq_daily

print("Datasets loaded:")
for name, df in datasets.items():
    print(f"  {name}: {len(df):>10,} rows, {df['symbol'].n_unique():>5} symbols")

# %% [markdown]
# **Finding**: The dataset inventory tells us how much evidence supports each asset-class
# estimate. A spread comparison built from only a few symbols deserves more caution than
# one built from a broad liquid panel.

# %% [markdown]
# ### Estimate Spreads for Each Dataset

# %%
spread_results = []

for name, df in datasets.items():
    if "timestamp" not in df.columns and "date" in df.columns:
        df = df.with_columns(pl.col("date").alias("timestamp"))
    # Ensure sorted by symbol and time
    df_sorted = df.sort(["symbol", "timestamp"])

    # Apply estimators per symbol
    estimated = df_sorted.with_columns(
        cs_spread=corwin_schultz_spread(pl.col("high"), pl.col("low"), window=CS_WINDOW),
        roll_spread_est=roll_spread(pl.col("close"), window=ROLL_WINDOW),
    ).filter(pl.col("cs_spread").is_not_null() & pl.col("roll_spread_est").is_not_null())

    # Aggregate to symbol-level median spreads
    symbol_spreads = estimated.group_by("symbol").agg(
        cs_bps=(pl.col("cs_spread").median() * 10_000),
        roll_bps=(pl.col("roll_spread_est").median() * 10_000),
    )

    # Store with asset class label
    spread_results.append(
        symbol_spreads.with_columns(
            asset_class=pl.lit(name),
        )
    )

all_spreads = pl.concat(spread_results)
print(f"Total symbol-level estimates: {len(all_spreads):,}")

# %% [markdown]
# **Interpretation**: The symbol count is the effective sample size for the cross-asset
# map. More symbols mean the median spread is describing a market regime rather than a
# handful of names.

# %% [markdown]
# ### Cross-Asset Spread Distribution

# %%
# Mean spread by asset class with p25-p75 range. The CS estimator clamps negative
# values to zero, so for ultra-liquid asset classes (ETFs, S&P 500 equities) the
# bulk of the distribution sits at exactly zero. Plotting the mean with a p25-p75
# range preserves the right-tail signal those classes carry instead of rendering
# as missing data.
class_stats = (
    all_spreads.group_by("asset_class")
    .agg(
        mean_bps=pl.col("cs_bps").mean(),
        p25_bps=pl.col("cs_bps").quantile(0.25),
        p75_bps=pl.col("cs_bps").quantile(0.75),
    )
    .sort("mean_bps")
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(12, 6))
yerr_lower = (class_stats["mean_bps"] - class_stats["p25_bps"]).clip(lower=0)
yerr_upper = (class_stats["p75_bps"] - class_stats["mean_bps"]).clip(lower=0)
bars = ax.bar(
    class_stats["asset_class"],
    class_stats["mean_bps"],
    yerr=[yerr_lower, yerr_upper],
    capsize=4,
)
for bar, mean in zip(bars, class_stats["mean_bps"]):
    ax.annotate(
        f"{mean:.2f} bps",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
ax.set_ylabel("Mean Estimated Spread (bps; bars span p25-p75)")
ax.set_xlabel("")
ax.set_title("Corwin-Schultz Spread Estimates by Asset Class")
ax.tick_params(axis="x", rotation=20)

fig.tight_layout()
fig.show()

# %% [markdown]
# **Finding**: Cross-asset dispersion matters more than small estimator error.
# Futures and major FX pairs live in a different spread regime from single-name
# equities, so the book should not discuss transaction costs as if they were a
# single universal constant.

# %% [markdown]
# ### Summary Statistics

# %%
summary = (
    all_spreads.group_by("asset_class")
    .agg(
        n_symbols=pl.col("symbol").n_unique(),
        cs_median=pl.col("cs_bps").median(),
        cs_p25=pl.col("cs_bps").quantile(0.25),
        cs_p75=pl.col("cs_bps").quantile(0.75),
        roll_median=pl.col("roll_bps").median(),
    )
    .sort("cs_median")
)
summary

# %% [markdown]
# **Finding**: The cross-asset ranking reveals two regimes. For less liquid assets
# (crypto, FX, some futures), the Corwin-Schultz estimator produces meaningful
# non-zero medians. For ultra-liquid equities (ETFs, S&P 500), daily high-low
# ranges are dominated by volatility and the estimator collapses to zero —
# correctly signalling that these markets trade at sub-basis-point spreads beyond
# the resolution of OHLCV-based estimators. The Roll estimator is less affected
# by censoring but tends to overestimate because momentum and other microstructure
# effects beyond the bid-ask bounce also contribute to return serial correlation.

# %% [markdown]
# ## 7. Estimated vs setup.yaml Comparison
#
# Do the cost assumptions in our case study configurations match the
# empirically estimated spreads?

# %%
# Load cost assumptions from all setup.yaml files
CASE_STUDIES = [
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "us_firm_characteristics",
    "fx_pairs",
    "cme_futures",
    "sp500_options",
    "us_equities_panel",
]

assumed_costs = {}
for cs_id in CASE_STUDIES:
    setup_path = get_case_study_source_dir(cs_id) / "config" / "setup.yaml"
    if setup_path.exists():
        setup = yaml.safe_load(setup_path.read_text())
        costs = setup.get("costs", {})
        assumed_costs[cs_id] = costs

# Map case study IDs to asset class names used in spread estimation
cs_to_asset = {
    "etfs": "ETFs",
    "crypto_perps_funding": "Crypto Perps",
    "nasdaq100_microstructure": "NASDAQ-100",
    "sp500_equity_option_analytics": "S&P 500 Equities",
    "us_firm_characteristics": "US Equities",
    "fx_pairs": "FX Pairs",
    "cme_futures": "CME Futures",
    "us_equities_panel": "US Equities",
}

# %% [markdown]
# ### Normalize setup.yaml Spread Assumptions


# %%
def extract_assumed_spread_bps(costs: dict) -> float | None:
    """Normalize heterogeneous setup.yaml spread assumptions to basis points."""
    if "per_leg_cost_bps_range" in costs:
        lo, hi = costs["per_leg_cost_bps_range"]
        return (lo + hi) / 2
    if "fee_schedule" in costs:
        return costs["fee_schedule"].get("taker_bps", 4)
    if "spread_bps" in costs:
        spread_bps = costs["spread_bps"]
        if isinstance(spread_bps, dict):
            vals = []
            for value in spread_bps.values():
                vals.append(sum(value) / len(value) if isinstance(value, list) else value)
            return sum(vals) / len(vals) if vals else None
    if "spread_ticks" in costs:
        return 5
    return None


# %% [markdown]
# ### Compare Empirical Estimates with setup.yaml Assumptions

# %%
comparison_rows = []
for cs_id, costs in assumed_costs.items():
    asset_name = cs_to_asset.get(cs_id)
    if asset_name is None:
        continue

    assumed_bps = extract_assumed_spread_bps(costs)
    asset_spreads = all_spreads.filter(pl.col("asset_class") == asset_name)
    if len(asset_spreads) > 0 and assumed_bps is not None:
        estimated_bps = asset_spreads["cs_bps"].median()
        comparison_rows.append(
            {
                "case_study": cs_id,
                "asset_class": asset_name,
                "assumed_bps": assumed_bps,
                "estimated_bps": estimated_bps,
            }
        )

comparison = pl.DataFrame(comparison_rows).sort("assumed_bps")
comparison

# %% [markdown]
# **Finding**: The setup comparison reveals where the chapter is conservative
# versus optimistic. Conservative assumptions are acceptable for robustness, but
# optimistic ones will bias gross-to-net expectations upward.

# %%
# Grouped bar chart: assumed vs estimated
if comparison_rows:
    comp_pd = comparison.to_pandas()

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(comp_pd))
    width = 0.35

    ax.bar(x - width / 2, comp_pd["assumed_bps"], width, label="Assumed (setup.yaml)")
    ax.bar(x + width / 2, comp_pd["estimated_bps"], width, label="Estimated (Corwin-Schultz)")

    ax.set_xticks(x)
    ax.set_xticklabels(comp_pd["case_study"], rotation=30, ha="right")
    ax.set_ylabel("Spread (bps)")
    ax.set_title("Assumed vs Estimated Spreads")
    ax.legend()

    fig.tight_layout()
    fig.show()

# %% [markdown]
# **Finding**: Several case studies assume costs that differ materially from
# empirical estimates. For highly liquid assets (NASDAQ-100 large caps, major
# FX pairs), assumed costs may be conservative — good for robustness but
# potentially understating strategy performance. For less liquid assets,
# assumed costs may be optimistic, warranting upward revision.

# %% [markdown]
# ## 8. Regime Conditioning: Spreads and Volatility
#
# Bid-ask spreads widen during high-volatility episodes. We examine this
# relationship using VIX quartiles as a regime indicator.

# %%
# Load VIX
macro = load_macro(series=["vixcls"])
date_col = "timestamp" if "timestamp" in macro.columns else "date"
vix = (
    macro.select(
        pl.col(date_col).cast(pl.Date).alias("timestamp"),
        pl.col("vixcls").alias("vix"),
    )
    .filter(pl.col("vix").is_not_null())
    .sort("timestamp")
)

# Compute VIX quartile thresholds
vix_q = vix["vix"].quantile(0.25), vix["vix"].quantile(0.5), vix["vix"].quantile(0.75)
print(f"VIX quartile boundaries: Q1<{vix_q[0]:.1f}, Q2<{vix_q[1]:.1f}, Q3<{vix_q[2]:.1f}")

# %% [markdown]
# **Finding**: These quartile boundaries define the regime labels used below.
# They matter because spread behavior should be compared on a like-for-like
# volatility basis rather than against an unconditional average.

# %%
# Use ETFs as the representative dataset for regime analysis
etf_sorted = datasets["ETFs"].sort(["symbol", "timestamp"])
etf_with_spreads = etf_sorted.with_columns(
    cs_spread=corwin_schultz_spread(pl.col("high"), pl.col("low"), window=CS_WINDOW),
).filter(pl.col("cs_spread").is_not_null())

# Join with VIX
etf_vix = etf_with_spreads.join(
    vix,
    left_on=pl.col("timestamp").cast(pl.Date),
    right_on="timestamp",
    how="inner",
).with_columns(
    cs_bps=pl.col("cs_spread") * 10_000,
    vix_regime=pl.when(pl.col("vix") < vix_q[0])
    .then(pl.lit("Q1 (Low)"))
    .when(pl.col("vix") < vix_q[1])
    .then(pl.lit("Q2"))
    .when(pl.col("vix") < vix_q[2])
    .then(pl.lit("Q3"))
    .otherwise(pl.lit("Q4 (High)")),
)

# %% [markdown]
# ### Spread Distribution by VIX Regime

# %%
# Mean ETF spread by VIX regime. A boxplot collapses to a flat zero line because
# the CS estimator clamps most ETF days to zero; the regime signal lives in the
# right-tail mean, so we plot mean spreads alone (the p90 is zero across all
# regimes — narrative quantifies that separately below).
regime_summary = (
    etf_vix.group_by("vix_regime")
    .agg(mean_bps=pl.col("cs_bps").mean())
    .to_pandas()
    .set_index("vix_regime")
    .reindex(["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(regime_summary["vix_regime"], regime_summary["mean_bps"])
for bar, mean in zip(bars, regime_summary["mean_bps"]):
    ax.annotate(
        f"{mean:.2f} bps",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
ax.set_ylabel("Mean Estimated Spread (bps)")
ax.set_xlabel("VIX Regime")
ax.set_title("ETF Spread Estimates by Volatility Regime")

fig.tight_layout()
fig.show()

# %% [markdown]
# **Finding**: The regime bar chart shows why static spread assumptions fail in
# stressed markets. Mean estimated spreads shift upward in high-VIX states.

# %%
# Regime summary statistics
regime_stats = (
    etf_vix.group_by("vix_regime")
    .agg(
        mean_spread=pl.col("cs_bps").mean(),
        median_spread=pl.col("cs_bps").median(),
        p90_spread=pl.col("cs_bps").quantile(0.90),
        n_obs=pl.len(),
    )
    .sort("mean_spread")
)
regime_stats

# %% [markdown]
# **Finding**: The mean roughly doubles from calm regimes (~2 bps in Q1-Q2) to
# stressed markets (~5 bps in Q4), confirming that non-zero spread events
# concentrate during volatility spikes. Median and p90 sit at zero for ETFs
# (see Section 6 for the censoring mechanism); the mean alone carries the
# regime signal here.

# %% [markdown]
# ### Spread Time Series with VIX Overlay

# %%
# Aggregate daily cross-sectional mean spread for ETFs
etf_ts = (
    etf_vix.group_by(pl.col("timestamp").cast(pl.Date).alias("timestamp"))
    .agg(
        mean_spread=pl.col("cs_bps").mean(),
    )
    .sort("timestamp")
    .join(vix, on="timestamp", how="inner")
)

etf_ts_pd = etf_ts.to_pandas()

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(
    etf_ts_pd["timestamp"],
    etf_ts_pd["mean_spread"],
    alpha=0.7,
    linewidth=0.8,
    label="Mean CS Spread",
)
ax1.set_ylabel("Estimated Spread (bps)")
ax1.set_xlabel("")

ax2 = ax1.twinx()
ax2.fill_between(
    etf_ts_pd["timestamp"],
    etf_ts_pd["vix"],
    alpha=0.15,
    color="orange",
    label="VIX",
)
ax2.set_ylabel("VIX")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.set_title("ETF Spread Estimates and VIX")
fig.tight_layout()
fig.show()

# %% [markdown]
# **Finding**: The mean CS spread co-moves strongly with VIX. Stress episodes
# (March 2020, early 2022) produce clear spread spikes as the high-low range
# incorporates widened bid-ask spreads. This pattern confirms that transaction
# cost assumptions should be regime-aware: a strategy that rebalances actively
# during volatile periods faces materially higher execution costs.
#
# **Implication for backtesting**: Using a single fixed cost assumption
# understates the challenge. The case study `costs.py` notebooks sweep
# a range of cost levels (0-50 bps); the empirical distribution here
# helps readers understand which part of that range is realistic for
# each asset class.

# %% [markdown]
# ## Key Takeaways
#
# 1. **OHLCV estimators have limited resolution on large caps**: Against
#    NASDAQ-100 ground truth aggregated to true daily bars, both the
#    Corwin-Schultz and Roll estimators show large positive bias and weak
#    correlation over a short 64-day, large-cap-only window. The high-low and
#    serial-covariance estimators carry real signal for wider-spread asset
#    classes but lack resolution where realized spreads are only a few bps.
#
# 2. **Estimated spreads span orders of magnitude across asset classes**: FX
#    pairs and CME futures come in at single-digit bps (CS median 2.2 and 0.9);
#    crypto perpetuals run ~16 bps; and on the short NASDAQ-100 sample the CS
#    estimator is the *largest* of all classes (median 42 bps) despite the ~4 bps
#    observed spread — the large positive bias flagged in Takeaway 1. For
#    ultra-liquid ETFs and S&P 500 stocks, the CS estimator falls below its
#    resolution threshold (median = 0).
#
# 3. **Regime dependence is material**: Mean CS spread roughly doubles
#    from low-VIX to high-VIX regimes for ETFs. Stress episodes like
#    March 2020 produce pronounced spread spikes, making fixed-cost
#    backtests optimistic during precisely the periods when rebalancing
#    is most urgent.
#
# 4. **Some setup.yaml assumptions need scrutiny**: Comparing empirical
#    estimates to configured costs identifies both conservative and
#    optimistic assumptions across our nine case studies.
#
# **Next**: See `03_market_impact_calibration` for market impact estimation,
# and `01_cost_taxonomy` for the full cost decomposition framework.
#
# **Book**: Chapter 18.3 discusses spread estimation methodology in detail.
