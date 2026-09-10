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
# # Crypto Premium Index: Funding Rate Arbitrage Data
#
# **Docker image**: `ml4t`
#
# ## Purpose
# Explore the Binance perpetual-futures premium index — the per-period deviation between
# perpetual and spot prices that determines the funding rate paid every 8 hours. The
# notebook profiles 19 USDT-margined perpetuals from 2020-01 to 2025-12 and turns the
# raw premium series into estimated funding APY for the funding-arbitrage case study.
#
# ## Learning Objectives
# - Load the 8-hour premium-index panel and read its schema.
# - Characterize the distribution and time-series behavior of BTC premium.
# - Compare premium volatility across majors and altcoins.
# - Translate premium into Binance's clamped funding rate and an annualized return.
#
# ## Book reference
# Chapter 2, §2.2 (asset-class market data — crypto datasets). The funding-arbitrage
# case study built on this dataset lives in `case_studies/crypto_perps_funding/`.
#
# ## Prerequisites
# - Crypto perpetual + premium parquet files materialized under `ML4T_DATA_PATH`
#   (`data/crypto/market/download.py`).
# - Binance's published funding settlements, which Section 5 scores the formula against.
#   The perpetual downloader does not fetch them; `case_studies/crypto_perps_funding/
#   funding_data.py` does.
# - Loaders `data.load_crypto_premium` and
#   `case_studies.crypto_perps_funding.funding_data.load_funding_rates`.
#
# ---

# %%
"""Crypto Premium Index — Funding rate arbitrage data exploration."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from case_studies.crypto_perps_funding.funding_data import load_funding_rates
from data import load_crypto_premium
from utils.style import COLORS, ml4t_palette, show_plotly_with_alt

# %% [markdown]
# ### Declared parameters
#
# The funding constants are Binance's, not ours: the interest rate and the clamp half-width are
# published contract terms, and the settlement cadence follows from them. They are declared
# rather than written into the arithmetic so a reader can see the whole contract in one place
# and CI can override any of it.

# %% tags=["parameters"]
# Binance's published funding terms, USDT-margined perpetuals.
INTEREST_RATE = 0.0001  # per settlement
FUNDING_CLAMP = 0.0005  # half-width of the clamp on (interest - premium)
FUNDING_INTERVAL_HOURS = 8
PERIODS_PER_DAY = 24 // FUNDING_INTERVAL_HOURS

# Analysis choices.
DEMO_SYMBOL = "BTCUSDT"
ROLLING_WINDOW_DAYS = 30
NOTABLE_PREMIUM = 0.001  # 10 bps: the level the regime split and the frequency stat both use
COLOR_CLIP_BPS = 20
TAIL_CLIP_PCT = 0.5  # dropped from each end before binning the cross-asset histograms

# %% [markdown]
# ---
#
# ## Section 1: Understanding the Premium Index
#
# ### What is the Premium Index?
#
# The **premium index** measures how far the perpetual trades from the underlying index. The
# one-line version is the relative difference between perpetual and spot, and it is worth
# starting there, but the exchange's definition is not that:
#
# $$\text{Premium Index} = \frac{\max(0,\ \text{Impact Bid} - \text{Price Index}) - \max(0,\ \text{Price Index} - \text{Impact Ask})}{\text{Price Index}}$$
#
# The impact bid and ask are the prices at which a fixed notional would fill on each side, so
# the numerator asks how far the perpetual's *executable* quote sits outside the index. Both
# terms are zero whenever the index falls between them, which means the index has a **dead
# zone** the width of the impact spread and returns exactly zero inside it.
#
# That is not a technicality. It is why a large share of this column is exactly zero, why the
# share is far larger for illiquid contracts than for BTC, and why treating those zeros as
# missing data would be a mistake - `10_crypto_perps_eda` measures all three.
#
# ### Key properties
#
# 1. **Positive premium**: the impact bid sits above the index - the perpetual's executable
#    quote is rich
# 2. **Negative premium**: the impact ask sits below the index - the executable quote is
#    cheap
# 3. **Exactly zero**: the index falls inside the impact spread
# 4. **Funding rate**: derived from the premium index, settled every 8 hours on Binance
#
# ### Who pays whom
#
# The premium's sign does not answer this, and the difference is not a corner case. Funding
# is $F = P + \operatorname{clamp}(I - P)$ with a positive interest rate $I$, so the premium
# has to fall below $I - c$ before funding turns negative at all - a zero premium still
# leaves longs paying shorts at exactly the interest rate. Section 5 derives this and
# measures how often each case occurs.
#
# Payment direction follows the funding rate:
#
# - **Funding positive**: longs pay shorts. **Long Spot** + **Short Perpetual** collects it,
#   market-neutral.
# - **Funding negative**: shorts pay longs. **Short Spot** + **Long Perpetual** collects it.

# %%
premium_df = load_crypto_premium(frequency="8h")

print(f"Total rows: {len(premium_df):,}")
print(f"Columns: {premium_df.columns}")
print("\nSchema:")
for col, dtype in premium_df.schema.items():
    print(f"  {col}: {dtype}")

# %%
# Overview by asset
symbol_stats = (
    premium_df.group_by("symbol")
    .agg(
        [
            pl.col("timestamp").min().alias("start"),
            pl.col("timestamp").max().alias("end"),
            pl.len().alias("rows"),
            pl.col("premium_index_close").mean().alias("avg_premium"),
            pl.col("premium_index_close").std().alias("std_premium"),
        ]
    )
    .sort("rows", descending=True)
)

symbol_stats

# %%
# Sample data - BTC premium index
btc_premium = premium_df.filter(pl.col("symbol") == DEMO_SYMBOL).sort("timestamp")

print(f"BTC Premium Index: {len(btc_premium):,} 8h observations")
print(f"Date range: {btc_premium['timestamp'].min()} to {btc_premium['timestamp'].max()}")

btc_premium.head(10)

# %% [markdown]
# ---
#
# ## Section 2: Premium Index Distribution
#
# Understanding the distribution of premium values is crucial for:
# 1. Setting entry/exit thresholds for arbitrage
# 2. Risk management (tail events)
# 3. Comparing opportunities across assets

# %%
# BTC Premium distribution
btc_close = btc_premium["premium_index_close"].to_numpy()

# Convert to basis points for readability
btc_close_bps = btc_close * 10000

fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=btc_close_bps,
        nbinsx=100,
        name="BTC Premium",
        marker_color=COLORS["copper"],
    )
)

# Add vertical lines for mean and +-2 std
mean_val = np.mean(btc_close_bps)
std_val = np.std(btc_close_bps)

fig.add_vline(
    x=mean_val,
    line_dash="dash",
    line_color=COLORS["negative"],
    annotation_text=f"Mean: {mean_val:.1f} bps",
)
fig.add_vline(
    x=mean_val + 2 * std_val,
    line_dash="dot",
    line_color=COLORS["positive"],
    annotation_text=f"+2σ: {mean_val + 2 * std_val:.1f} bps",
)
fig.add_vline(
    x=mean_val - 2 * std_val,
    line_dash="dot",
    line_color=COLORS["positive"],
    annotation_text=f"-2σ: {mean_val - 2 * std_val:.1f} bps",
)

fig.update_layout(
    title=f"{DEMO_SYMBOL} premium index distribution, basis points",
    xaxis_title="Premium Index (bps)",
    yaxis_title="Frequency",
    height=400,
)
show_plotly_with_alt(
    fig,
    "A histogram of the eight-hourly premium index in basis points, with dashed vertical lines marking the mean and two standard deviations either side. The mass is a tall narrow peak close to zero with long thin tails reaching much further left than right.",
)

# %%
print("BTC Premium Statistics:")
print(f"  Mean: {mean_val:.2f} bps")
print(f"  Std:  {std_val:.2f} bps")
print(f"  Min:  {np.min(btc_close_bps):.2f} bps")
print(f"  Max:  {np.max(btc_close_bps):.2f} bps")
print(f"  Skew: {((btc_close_bps - mean_val) ** 3).mean() / std_val**3:.2f}")

# %% [markdown]
# The four panels below drop the outermost half percent from each tail before binning. Limiting
# the axis alone is not enough: bin width is set by the full range, so a contract carrying
# thousand-basis-point dislocations would resolve its whole body into two bins. The guard on an
# empty array is for reduced CI panels, where a symbol may be absent and a percentile of
# nothing raises.

# %%
major_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

fig = make_subplots(rows=2, cols=2, subplot_titles=major_symbols)

colors = ml4t_palette(len(major_symbols), categorical=True)

for idx, (symbol, color) in enumerate(zip(major_symbols, colors, strict=False)):
    row = idx // 2 + 1
    col = idx % 2 + 1

    data = premium_df.filter(pl.col("symbol") == symbol)["premium_index_close"].to_numpy() * 10000

    if data.size:
        lo, hi = np.percentile(data, [TAIL_CLIP_PCT, 100 - TAIL_CLIP_PCT])
        data = data[(data >= lo) & (data <= hi)]

    fig.add_trace(
        go.Histogram(x=data, nbinsx=50, marker_color=color, name=symbol), row=row, col=col
    )

fig.update_layout(
    title="Premium index distribution, four major contracts",
    height=500,
    showlegend=False,
)
show_plotly_with_alt(
    fig,
    "Four histogram panels, one per contract, each showing the premium index in basis points with its outermost half percent of observations removed before binning. All four are centred near zero; the panels widen from the most liquid contract to the least.",
)

# %% [markdown]
# ---
#
# ## Section 3: Time Series Analysis
#
# Premium index varies over time based on market sentiment. Let's analyze:
# 1. Long-term trends
# 2. Regime changes (bull vs bear markets)
# 3. Correlation with price movements

# %% [markdown]
# The rolling window is expressed in days and converted to observations here, because the
# series is on an eight-hour cadence and a window given in rows would mean a different span if
# the cadence ever changed.

# %%
ROLLING_WINDOW_OBS = ROLLING_WINDOW_DAYS * PERIODS_PER_DAY

btc_bps = btc_premium.with_columns(
    (pl.col("premium_index_close") * 10000).alias("premium_bps"),
).with_columns(
    pl.col("premium_index_close")
    .rolling_mean(window_size=ROLLING_WINDOW_OBS)
    .alias("rolling_premium"),
)

# %%
# Plot raw 8h premium and 30-day rolling mean
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=[
        "Premium index, 8h observations (bps)",
        f"{ROLLING_WINDOW_DAYS}-day rolling average (bps)",
    ],
)

fig.add_trace(
    go.Scatter(
        x=btc_bps["timestamp"].to_list(),
        y=btc_bps["premium_bps"].to_list(),
        mode="lines",
        name="8h Premium",
        line=dict(color=COLORS["copper"], width=1),
        opacity=0.6,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=btc_bps["timestamp"].to_list(),
        y=(btc_bps["rolling_premium"] * 10000).to_list(),
        mode="lines",
        name="30-Day Rolling Avg",
        line=dict(color=COLORS["negative"], width=2),
    ),
    row=2,
    col=1,
)
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], row=1, col=1)
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], row=2, col=1)
fig.update_layout(
    height=600,
    showlegend=False,
    title=f"{DEMO_SYMBOL} premium index, raw and rolling average",
)
fig.update_yaxes(title_text="Premium (bps)", row=1, col=1)
fig.update_yaxes(title_text="Premium (bps)", row=2, col=1)
show_plotly_with_alt(
    fig,
    "Two stacked panels sharing a date axis across six years, each with a dashed line at zero. "
    "The upper panel plots the raw eight-hourly premium in basis points, a dense band around "
    "zero with occasional large excursions. The lower panel plots its rolling average, a much "
    "smoother line that sits above zero for a long stretch early in the sample and below it "
    "for a long stretch afterwards.",
)

# %%
# Report the observed BTC range so the reader can size the y-axis.
btc_bps_series = btc_bps["premium_bps"]
print(f"BTC premium range: {btc_bps_series.min():.1f} to {btc_bps_series.max():.1f} bps")

# %%
# Identify premium regimes
btc_regimes = btc_premium.with_columns(
    [
        # Define regimes based on premium level
        pl.when(pl.col("premium_index_close") > NOTABLE_PREMIUM)
        .then(pl.lit("High Premium (Bullish)"))
        .when(pl.col("premium_index_close") < -NOTABLE_PREMIUM)
        .then(pl.lit("Low Premium (Bearish)"))
        .otherwise(pl.lit("Neutral"))
        .alias("regime"),
        # Year for grouping
        pl.col("timestamp").dt.year().alias("year"),
    ]
)

# Regime distribution by year (counts of 8h periods, ~1095 per full year)
regime_dist = (
    btc_regimes.group_by(["year", "regime"])
    .agg(pl.len().alias("periods_8h"))
    .sort(["year", "regime"])
)

regime_dist.pivot(on="regime", index="year", values="periods_8h").fill_null(0)

# %% [markdown]
# ---
#
# ## Section 4: Cross-Asset Premium Comparison
#
# Different cryptocurrencies have different premium dynamics:
# - **BTC/ETH**: Lower volatility, tighter premiums
# - **Altcoins**: Higher volatility, wider premium swings
#
# This affects arbitrage opportunity selection.

# %%
# Calculate premium statistics for all assets
premium_stats = (
    premium_df.group_by("symbol")
    .agg(
        [
            pl.col("premium_index_close").mean().alias("mean_premium"),
            pl.col("premium_index_close").std().alias("std_premium"),
            pl.col("premium_index_close").min().alias("min_premium"),
            pl.col("premium_index_close").max().alias("max_premium"),
            # Percentage of time premium > 10 bps (profitable arbitrage threshold)
            (pl.col("premium_index_close").abs() > NOTABLE_PREMIUM)
            .mean()
            .alias("pct_above_notable"),
        ]
    )
    .sort("std_premium", descending=True)
)

# Convert to basis points for display
premium_stats_bps = premium_stats.with_columns(
    [
        (pl.col("mean_premium") * 10000).round(2).alias("mean_bps"),
        (pl.col("std_premium") * 10000).round(2).alias("std_bps"),
        (pl.col("min_premium") * 10000).round(2).alias("min_bps"),
        (pl.col("max_premium") * 10000).round(2).alias("max_bps"),
        (pl.col("pct_above_notable") * 100).round(1).alias("pct_above_notable"),
    ]
).select(["symbol", "mean_bps", "std_bps", "min_bps", "max_bps", "pct_above_notable"])

premium_stats_bps

# %%
# Scatter: Premium volatility vs mean premium
fig = px.scatter(
    premium_stats_bps.to_pandas(),
    x="std_bps",
    y="mean_bps",
    size="pct_above_notable",
    color="symbol",
    hover_name="symbol",
    title="Mean premium against premium volatility, by contract",
    labels={
        "std_bps": "Premium Volatility (bps)",
        "mean_bps": "Mean Premium (bps)",
        "pct_above_notable": "% of periods beyond the notable level",
    },
)

fig.update_layout(
    height=600,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
    ),
    margin=dict(b=120),
)
show_plotly_with_alt(
    fig,
    "A scatter of mean premium against premium volatility, both in basis points, one marked point per contract, sized by how often the premium exceeds the notable level. Every point sits at or below zero on the vertical axis, and they spread rightwards along the volatility axis with the least liquid contracts furthest out.",
)

print("\nInterpretation:")
print("- Top-right quadrant: High volatility, positive bias (bullish altcoins)")
print("- Larger bubbles: More arbitrage opportunities (premium often > 10bps)")

# %% [markdown]
# The colour scale on the heatmap below is clipped. A handful of contract-months reach far
# enough that, unclipped, they set the range and flatten every other cell to the same shade.
# The clipped extremes are listed after the figure rather than hidden by it.

# %%
monthly_premium = (
    premium_df.with_columns([pl.col("timestamp").dt.strftime("%Y-%m").alias("month")])
    .group_by(["symbol", "month"])
    .agg(pl.col("premium_index_close").mean().alias("avg_premium"))
)

# Pivot for heatmap
heatmap_data = monthly_premium.pivot(on="month", index="symbol", values="avg_premium").sort(
    "symbol"
)

# Get month columns in order
month_cols = sorted([c for c in heatmap_data.columns if c != "symbol"])
assets = heatmap_data["symbol"].to_list()

# Extract values for heatmap
z_values = heatmap_data.select(month_cols).to_numpy() * 10000  # Convert to bps

# %%
fig = go.Figure(
    data=go.Heatmap(
        z=z_values,
        x=month_cols,
        y=assets,
        colorscale="RdBu",
        zmid=0,
        zmin=-COLOR_CLIP_BPS,
        zmax=COLOR_CLIP_BPS,
        colorbar=dict(title="Premium (bps)"),
    )
)

fig.update_layout(
    title="Monthly average premium by contract, colour clipped",
    xaxis_title="Month",
    yaxis_title="Symbol",
    height=600,
)
show_plotly_with_alt(
    fig,
    "A heatmap with months across the horizontal axis and contracts up the vertical, shaded from one colour for negative average premium through neutral at zero to another for positive, with the scale clipped. Most cells sit close to neutral; isolated rows and columns saturate at the negative end, and the top rows are blank where a contract had not listed yet.",
)

# Report extremes that exceed color scale (shown as saturated colors)
extremes = (
    monthly_premium.filter(pl.col("avg_premium").abs() * 10000 > COLOR_CLIP_BPS)
    .with_columns((pl.col("avg_premium") * 10000).round(1).alias("avg_bps"))
    .select(["symbol", "month", "avg_bps"])
    .sort("avg_bps")
)
extremes

# %% [markdown]
# ---
#
# ## Section 5: Funding Rate Estimation
#
# The funding rate is what the premium index is *for*: every eight hours one side of the market
# pays the other, and the rate is computed from the index. Binance's formula is
#
# $$F = P + \operatorname{clamp}\!\left(I - P,\ -c,\ +c\right)$$
#
# where $P$ is the premium index, $I$ the interest rate and $c$ the clamp half-width. Both
# constants are Binance's published contract terms and are declared as `INTEREST_RATE` and
# `FUNDING_CLAMP` in the parameters cell, so the arithmetic below and the reader read the same
# values.
#
# **The clamp is on the difference, not on the premium**, and that changes the result
# qualitatively rather than by a little. Whenever $|I - P|$ is inside the clamp the clamp does
# nothing, the expression reduces to $F = P + (I - P) = I$, and the funding rate is *exactly*
# the interest rate regardless of where the premium sat. Only outside that band does the
# premium reach the funding rate at all.
#
# So the funding rate has a dead zone of its own, on the same pattern as the premium index in
# Section 1: a formula whose output is pinned to a constant over a range of its input. Writing
# the clamp on $P$ instead - $\operatorname{clamp}(P) + I$ - is a natural misreading and it
# removes the dead zone entirely, producing a funding series that varies where the real one is
# flat.
#
# **Annualized:** $\text{APY} = F \times 3 \times 365$, three settlements a day.
#
# ### Pairing an estimate with its settlement
#
# One thing has to be settled before the formula can be checked against anything: what the
# timestamp on a premium bar means. Binance stamps a kline with the time the bar *opens*
# (`open_time` in the archive; `data/crypto/market/download.py` carries it straight through to
# the `timestamp` column). An 8-hour bar stamped 00:00 therefore spans 00:00 to 08:00, and its
# close is the premium as the interval ends - the interval the exchange averages to settle
# funding **at 08:00**.
#
# So a row stamped `t` estimates the settlement at `t + 8h`, not the one at `t`. The bars and
# the settlements both sit on the same 00:00 / 08:00 / 16:00 grid, so joining them on the raw
# timestamp produces a full set of rows and no error at all - it just compares each estimate
# with the settlement one interval too early. The settlement time is computed as a column
# below so the join has to name it.

# %%
premium_col = pl.col("premium_index_close")
btc_funding = btc_premium.with_columns(
    (premium_col + (INTEREST_RATE - premium_col).clip(-FUNDING_CLAMP, FUNDING_CLAMP)).alias(
        "est_funding_rate"
    ),
    (premium_col.clip(-FUNDING_CLAMP, FUNDING_CLAMP) + INTEREST_RATE).alias("clamp_on_premium"),
    (pl.col("timestamp") + pl.duration(hours=FUNDING_INTERVAL_HOURS)).alias("settles_at"),
).with_columns(
    (pl.col("est_funding_rate") * PERIODS_PER_DAY * 365 * 100).alias("annualized_pct"),
)

_pinned = btc_funding.filter((pl.col("est_funding_rate") - INTEREST_RATE).abs() < 1e-12).height
print(
    f"Periods where the clamp binds: "
    f"{btc_funding.filter((INTEREST_RATE - premium_col).abs() > FUNDING_CLAMP).height:,} of "
    f"{len(btc_funding):,} "
    f"({100 * btc_funding.filter((INTEREST_RATE - premium_col).abs() > FUNDING_CLAMP).height / len(btc_funding):.1f}%)"
)
print(
    f"Periods where funding is pinned at exactly the interest rate: {_pinned:,} "
    f"({100 * _pinned / len(btc_funding):.1f}%)"
)

# %% [markdown]
# ### Checking the formula against what was actually charged
#
# The estimate does not have to be taken on trust. Binance publishes the funding rate it
# settled at each interval, and the case study's download keeps it, so the formula can be held
# against six years of the exchange's own numbers. That turns "here is the formula" into a
# claim that can fail - and it is the check that separates the two readings of the clamp.

# %%
realized_funding = load_funding_rates(symbols=[DEMO_SYMBOL]).select(
    "timestamp", "symbol", pl.col("funding_rate").cast(pl.Float64).alias("realized")
)

check = btc_funding.join(
    realized_funding,
    left_on=["settles_at", "symbol"],
    right_on=["timestamp", "symbol"],
    how="inner",
)
print(f"Settlements with both an estimate and a realized rate: {len(check):,}")
for label, col in [
    ("clamp on (interest - premium)", "est_funding_rate"),
    ("clamp on the premium", "clamp_on_premium"),
]:
    err = (pl.col(col) - pl.col("realized")).abs()
    stats = check.select(err.mean().alias("mae"), (err < 1e-9).mean().alias("exact"))
    print(f"  {label:32s} mean abs error {stats['mae'][0]:.6f}   exact {stats['exact'][0]:.1%}")

_realized_pinned = check.filter((pl.col("realized") - INTEREST_RATE).abs() < 1e-12).height
print(
    f"Realized funding exactly at the interest rate: {_realized_pinned:,} of {len(check):,} "
    f"({100 * _realized_pinned / len(check):.1f}%)"
)

# %% [markdown]
# ### The alignment is a claim too, and the same data tests it
#
# The eight-hour shift above was read off the download path rather than measured, and a wrong
# shift would fail silently here for the same reason the raw join does: every offset that is a
# multiple of the settlement interval lands on the grid and returns a nearly full set of rows.
# What separates them is how well the estimate tracks the rate it is paired with. Sweeping the
# offset makes the right one visible instead of assumed.

# %%
for offset in (-FUNDING_INTERVAL_HOURS, 0, FUNDING_INTERVAL_HOURS, 2 * FUNDING_INTERVAL_HOURS):
    paired = btc_funding.with_columns(
        (pl.col("timestamp") + pl.duration(hours=offset)).alias("paired_at")
    ).join(
        realized_funding,
        left_on=["paired_at", "symbol"],
        right_on=["timestamp", "symbol"],
        how="inner",
    )
    err = (pl.col("est_funding_rate") - pl.col("realized")).abs()
    stats = paired.select(
        err.mean().alias("mae"),
        (err < 1e-9).mean().alias("exact"),
        pl.corr("est_funding_rate", "realized").alias("corr"),
    )
    print(
        f"  bar stamped t paired with settlement t{offset:+3d}h:  rows {len(paired):,}   "
        f"mean abs error {stats['mae'][0]:.6f}   exact {stats['exact'][0]:.1%}   "
        f"correlation {stats['corr'][0]:.3f}"
    )

# %% [markdown]
# Every offset joins, and the mean absolute errors are close enough that on their own they
# would not decide anything. The correlation does: it peaks at the one-interval-forward
# pairing and falls away on both sides, which is the pairing where the estimate and the
# realized rate are computed from the same eight hours of premium. The wrong pairings still
# correlate, because funding is persistent from one settlement to the next, so a misalignment
# of this kind looks entirely reasonable in isolation and only the sweep locates it.
#
# With the pairing settled, the formula comparison stands: the published formula reproduces the
# exchange's rate several times more closely than the misreading does, and only it produces the
# point mass that is actually there - better than a third of realized BTC settlements are
# exactly the interest rate, to the last decimal place. A funding series built by clamping the
# premium would show that value almost never.
#
# The agreement is close but not exact, and the reason is worth stating rather than leaving as
# noise. Binance computes the funding rate from a time-weighted average of the premium index
# over the interval, sampled far more finely than the eight-hour bars this notebook has. Using
# the bar's close is a proxy for that average. Of the proxies this file supports it reproduces
# the exchange most closely - the bar's mean and its high-low midpoint both do worse - but it
# remains a proxy, and the residual is what it costs.
#
# The estimate is therefore good enough to reason about the shape of funding and not a
# substitute for the realized series where the realized series exists. The case study uses the
# realized rates.

# %%
avg_funding_rate = float(btc_funding["est_funding_rate"].mean())
ann_min = float(btc_funding["annualized_pct"].min())
ann_max = float(btc_funding["annualized_pct"].max())
ann_mean = float(btc_funding["annualized_pct"].mean())
print("BTC Estimated Funding Rate Analysis:")
print(f"  Average funding rate (per 8h): {avg_funding_rate * 100:.4f}%")
print(f"  Annualized return (avg): {ann_mean:.1f}%")
print(f"  Annualized return (max): {ann_max:.1f}%")
print(f"  Annualized return (min): {ann_min:.1f}%")

# %% [markdown]
# ### The clamp does not bound the funding rate
#
# It is tempting to read the clamp as a cap: five basis points per settlement either side of
# the interest rate, so an APY ceiling somewhere in the tens of percent. That is a misreading
# of the same formula, in the other direction. Outside the dead zone the clamp saturates and
# the expression becomes $F = P \mp c$ for the clamp half-width $c$, which tracks the premium
# wherever it goes. The clamp bounds how far funding can sit *from* the premium, not how
# large it can be.
#
# The misreading's own bounds are worth writing down before testing them, because they are
# not symmetric: $\operatorname{clamp}(P) + I$ runs from $I - c$ to $I + c$, which the
# interest rate shifts off zero. Testing $|F|$ against the upper bound alone would miss every
# violation below the lower one.
#
# The realized series settles it, and the answer is not marginal.

# %%
realized_all = load_funding_rates().select(
    "symbol", pl.col("funding_rate").cast(pl.Float64).alias("realized")
)

_supposed_floor = INTEREST_RATE - FUNDING_CLAMP
_supposed_ceiling = INTEREST_RATE + FUNDING_CLAMP
_beyond = (pl.col("realized") < _supposed_floor) | (pl.col("realized") > _supposed_ceiling)
_btc_realized = realized_all.filter(pl.col("symbol") == DEMO_SYMBOL)["realized"]
_annualize = PERIODS_PER_DAY * 365 * 100

print(
    f"If the clamp bounded funding, it would run {_supposed_floor:.4f} to "
    f"{_supposed_ceiling:.4f} per settlement "
    f"({_supposed_floor * _annualize:.1f}% to {_supposed_ceiling * _annualize:.1f}% APY)"
)
print(
    f"Realized {DEMO_SYMBOL} funding actually ranges "
    f"{_btc_realized.min():.4f} to {_btc_realized.max():.4f} "
    f"({_btc_realized.min() * _annualize:.0f}% to {_btc_realized.max() * _annualize:.0f}% APY)"
)
print(
    f"  settlements outside those bounds: "
    f"{realized_all.filter(pl.col('symbol') == DEMO_SYMBOL).select(_beyond.mean()).item():.2%}"
)

realized_all.group_by("symbol").agg(
    pl.col("realized").mean().alias("mean_rate"),
    pl.col("realized").min().alias("min_observed"),
    pl.col("realized").max().alias("max_observed"),
    _beyond.mean().alias("share_outside_supposed_bounds"),
).sort("min_observed")

# %% [markdown]
# Funding does have a hard cap, but it is a separate mechanism at a far wider level and it is
# set per contract rather than universally. The table above does not show that cap: its
# columns are the largest and smallest rates each contract actually settled at, which bound
# the enforced limit from inside and say nothing about where it sits or whether it moved
# during the sample. What they do establish is enough for the point at hand - the observed
# extremes are already far outside the clamp, and they differ by an order of magnitude across
# the universe. Reading the clamp as the cap understates the tail risk of a funding strategy
# several times over on the most liquid contract and far more on the thin ones.

# %%
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=btc_funding["settles_at"].to_list(),
        y=btc_funding["annualized_pct"].to_list(),
        mode="lines",
        name="Annualized Funding Return",
        line=dict(color=COLORS["copper"], width=1),
    )
)

# Add horizontal lines for reference
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
fig.add_hline(y=20, line_dash="dot", line_color=COLORS["positive"], annotation_text="20% APY")
fig.add_hline(y=-20, line_dash="dot", line_color=COLORS["negative"], annotation_text="-20% APY")

y_padding = 10
fig.update_layout(
    title=f"{DEMO_SYMBOL} estimated annualized funding return",
    xaxis_title="Date",
    yaxis_title="Annualized Return (%)",
    yaxis=dict(range=[ann_min - y_padding, ann_max + y_padding]),
    height=400,
)
show_plotly_with_alt(
    fig,
    "A time series of the estimated annualized funding return in percent across six years, with a dashed line at zero and dotted reference lines at plus and minus twenty percent. The line oscillates around zero, spending long stretches within the reference lines and spiking well beyond them in both directions.",
)

print(f"Annualized funding return range: {ann_min:.1f}% to {ann_max:.1f}%")

# %% [markdown]
# One more count, and it needs labelling carefully. The share of settlements whose estimated
# APY exceeds a chosen level and the share where the clamp binds are different quantities, and
# the relation between them runs one way. Exceeding the threshold requires a binding clamp,
# because funding without one is pinned at the interest rate and annualizes far below any
# threshold worth setting. The clamp binding does not require exceeding the threshold: it binds
# whenever the premium sits more than the clamp half-width from the interest rate, which is a
# routine condition, and the resulting APY is usually nowhere near the threshold.

# %%
APY_THRESHOLD_PCT = 20
high_conviction = btc_funding.filter(pl.col("annualized_pct").abs() > APY_THRESHOLD_PCT)
_clamp_binds = btc_funding.filter((INTEREST_RATE - premium_col).abs() > FUNDING_CLAMP).height

print(
    f"Settlements with |APY| above {APY_THRESHOLD_PCT}%: {len(high_conviction):,} of "
    f"{len(btc_funding):,} ({100 * len(high_conviction) / len(btc_funding):.1f}%)"
)
print(
    f"Settlements where the clamp binds:       {_clamp_binds:,} of {len(btc_funding):,} "
    f"({100 * _clamp_binds / len(btc_funding):.1f}%)"
)
print(
    f"Without a binding clamp funding is exactly {INTEREST_RATE:.4f}, or "
    f"{INTEREST_RATE * _annualize:.2f}% APY, so every settlement above the threshold is one "
    f"where the clamp binds. The converse does not hold."
)

(
    high_conviction.sort("annualized_pct", descending=True)
    .head(10)
    .select(["settles_at", "premium_index_close", "est_funding_rate", "annualized_pct"])
)

# %% [markdown]
# ---
#
# ## Section 6: Loading this data elsewhere
#
# Everything above went through `load_crypto_premium`, which is the loader the rest of the book
# uses. The `ml4t-data` package also exposes `CryptoDataManager`, a class-based API over the
# same files, and the case study reaches for it where it needs the download path as well as the
# read path.
#
# There is nothing to demonstrate here that the sections above have not already shown, so this
# section states which entry point to use rather than importing one to print that it exists.
# `load_crypto_premium` for reading a panel, `CryptoDataManager` when you also need to fetch.

# %% [markdown]
# ---
#
# ## Key Takeaways
#
# 1. **The premium index is not the perpetual-spot difference, and the gap matters.** Binance
#    computes it from the impact bid and ask against the price index, with a numerator that is
#    zero whenever the index sits between them. That dead zone is why a large share of the
#    column is exactly zero, why the share rises as liquidity falls, and why those zeros are
#    measurements rather than gaps. `10_crypto_perps_eda` measures all three.
#
# 2. **The funding clamp is on the difference, not on the premium.** Binance settles
#    $F = P + \operatorname{clamp}(I - P)$, so whenever the premium sits within the clamp of
#    the interest rate the whole expression reduces to the interest rate exactly.
#    Funding then has a dead zone of its own, and better than a third of realized settlements
#    on the most liquid contract sit precisely on it. Clamping the premium instead removes that
#    point mass and reproduces the exchange's own rate several times less accurately - both
#    versions are computed above and scored against six years of published rates.
#
# 3. **The clamp does not cap the funding rate.** Outside the dead zone it saturates and
#    funding tracks the premium with a fixed offset, so the rate is unbounded by that
#    mechanism. Realized settlements run to many times the ceiling the clamp appears to imply,
#    and the real cap is a separate per-contract limit that differs by an order of magnitude
#    across this universe. Reading the clamp as a cap understates a funding strategy's tail
#    badly, and most on the contracts where the tail is worst.
#
# 4. **Two different counts, one label.** The share of settlements whose APY exceeds a chosen
#    threshold and the share where the clamp binds are different quantities, and the
#    implication runs one way: exceeding the threshold requires a binding clamp, while the
#    clamp binds routinely at premiums that move the APY hardly at all. Both are printed
#    above, and they differ by tens of percentage points.
#
# 5. **The estimate is a proxy and says so.** Binance computes funding from a time-weighted
#    average of the premium over the interval; this notebook has eight-hour bars and uses the
#    close, which is the most accurate of the proxies available here and still not exact. Where
#    the realized series exists, use it - the case study does.
#
# 6. **Premium volatility spans the universe by more than an order of magnitude**, and the
#    dispersion is episodic rather than steady: the widest contracts earn their standard
#    deviations in a handful of dislocations. Contract selection for a funding strategy is a
#    choice about which tail to hold, and the per-symbol table above is where that choice is
#    made.
#
# ### Implications for the funding-arbitrage case study
#
# - **Direction**: the mean premium is negative for every contract in this panel, and the mean
#   realized funding rate is *positive* for nearly all of them. The transform is monotone - a
#   larger premium never produces a smaller funding rate - but it does not preserve sign: it
#   has a plateau at the interest rate covering every premium within the clamp of it, and the
#   interest rate is positive. Whether that plateau is enough to carry the mean across zero
#   depends on how much of the distribution sits below the plateau and how far, which is a
#   question about this sample and not an implication. In this sample it does. Read direction
#   off the realized funding column in the per-symbol table above, never off the premium's
#   sign, and treat both as properties of this window rather than laws.
# - **Regimes**: the rolling premium changes sign for long stretches, so a static threshold
#   fires in one regime and never in the other. `case_studies/crypto_perps_funding/` carries
#   the regime-aware version.
# - **Ties**: the premium's point mass at zero reaches a third of observations on the thinnest
#   contracts, so any percentile or z-score feature built on this column is standardising a
#   distribution with a large tie group.
#
# **Next**: `12_fx_pairs_eda` profiles the third 24/7-adjacent dataset —
# G10 FX pairs at 4h cadence — completing the global market-data tour.
