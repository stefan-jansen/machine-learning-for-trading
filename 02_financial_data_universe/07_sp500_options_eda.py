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
# # S&P 500 Options Analytics
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Profile the AlgoSeek S&P 500 options analytics dataset for an 8-symbol 2020 EDA
# slice — chain structure, implied-volatility surfaces, data quality, and an early
# look at the predictive content that motivates the options case studies.
#
# ## Learning Objectives
#
# - Read an option chain and locate strikes, expirations, and call/put pairs.
# - Construct and visualize an implied-volatility smile, term structure, and surface.
# - Apply IV-convergence and Greeks-validity filters to clean options data.
# - Quantify a baseline IV-change → forward-return relationship in the cross section.
#
# ## Book Reference
#
# §2.2, "The asset-class market data landscape" - the derivatives part of it.
#
# ## Prerequisites
#
# - Familiarity with daily OHLC equity data (`01_us_equities_eda`).
# - The AlgoSeek S&P 500 options EDA parquet at `$ML4T_DATA_PATH/sp500_options/`
#   (8 representative underlyings: AAPL, AMZN, BA, GOOGL, JPM, KO, MSFT, XOM).
# - The S&P 500 daily-bar parquet covering the same 2020 window.
#
# Loaders used:
#
# | Dataset | Loader | Coverage |
# |---------|--------|----------|
# | S&P 500 options (EDA slice) | `load_sp500_options_eda()` | 2020, 8 underlyings |
# | S&P 500 daily prices | `load_sp500_daily_bars()` | 2020, same 8 underlyings |

# %%
"""S&P 500 Options Analytics — options chain structure, volatility surfaces, and data quality."""

from datetime import date

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data import load_sp500_daily_bars, load_sp500_options_eda
from utils.style import COLORS, show_plotly_with_alt

# %% [markdown]
# ### Declared parameters
#
# Everything the analysis is bounded by is declared here rather than repeated as a literal
# further down, so a reader can see the whole scope of the notebook in one cell and Papermill
# can override any of it for CI.
#
# The two moneyness bands are the ones that carry an argument. `ATM_BAND` is deliberately
# narrow: an at-the-money implied volatility is meant to be read at a single strike, and
# widening the band mixes in the smile. `CHAIN_BAND` is wide enough to show the smile's shape
# without letting the far wings, where quotes are stale and spreads are enormous, set the
# colour scale.

# %% tags=["parameters"]
START_DATE = "2020-01-01"
END_DATE = "2020-12-31"
DAILY_START_DATE = "2020-01-01"

# Moneyness (strike / spot) bands.
ATM_BAND = (0.98, 1.02)
CHAIN_BAND = (0.7, 1.3)

# Forward-return horizon, in trading days, for the information-content section.
FORWARD_DAYS = 5

# Expirations beyond this are drawn as their own section rather than shown alongside the
# near-dated chain, which carries many more strikes.
SURFACE_MAX_DAYS = 180

# The spread section deliberately reaches further out than CHAIN_BAND, because what happens in
# the wings is its subject.
SPREAD_BAND = (0.5, 1.5)

# "Near the money" for the spread summary; outside it either way is "away from the money".
WING_BAND = (0.9, 1.1)

# Options priced below this are quoted in ticks rather than in a spread, so a percentage
# spread computed from them says more about the tick size than about liquidity.
MIN_MID_PRICE = 0.10

# The S&P 500's closing low of the 2020 drawdown. A dated external fact, not a result of this
# notebook, so it is declared rather than computed - the IV peak beside it on the same chart
# *is* computed, and the point of drawing both is that they are not the same day.
SPX_TROUGH_DATE = "2020-03-23"

# %% [markdown]
# ## 1. Options Primer for ML Practitioners
#
# Before diving into the data, let's establish the key concepts that make options
# data different from—and complementary to—spot market data.
#
# ### What is an option?
#
# An option is a **derivative contract** that gives the holder the right (but not
# obligation) to buy or sell an underlying asset at a specified price (strike)
# by a specified date (expiration).
#
# | Type | Right | Profitable When |
# |------|-------|-----------------|
# | **Call** | Buy at strike | Underlying rises above strike |
# | **Put** | Sell at strike | Underlying falls below strike |
#
# ### Why options data matters for ML
#
# Options prices embed **forward-looking information** that spot prices don't:
#
# 1. **Implied Volatility (IV)** - Market's expectation of future volatility
# 2. **IV Skew** - Relative demand for downside vs upside protection
# 3. **Term Structure** - How expectations change across time horizons
# 4. **Greeks** - Sensitivities that quantify risk exposures
#
# This information can predict:
# - Future realized volatility
# - Underlying price movements (via order flow/positioning)
# - Tail risk events (via skew)
#
# ### Moneyness: in, at and out of the money
#
# Moneyness describes how an option's strike relates to the current spot price:
#
# | Moneyness | Call (Strike vs Spot) | Put (Strike vs Spot) | Characteristics |
# |-----------|----------------------|---------------------|-----------------|
# | **ITM** (In-the-money) | Strike < Spot | Strike > Spot | Has intrinsic value |
# | **ATM** (At-the-money) | Strike ≈ Spot | Strike ≈ Spot | Highest time value |
# | **OTM** (Out-of-the-money) | Strike > Spot | Strike < Spot | Pure time value |
#
# Moneyness in this notebook is **strike divided by spot**. A ratio of one is at the money,
# below one is an in-the-money call and an out-of-the-money put, and above one is the reverse.
#
# ### Option value components
#
# An option's price decomposes into intrinsic value (immediate exercise payoff) and
# time value (the remainder, reflecting optionality):
#
# $$\text{Price} = \text{Intrinsic} + \text{Time}$$
#
# $$\text{Intrinsic}_{\text{call}} = \max(0,\; S - K), \qquad \text{Intrinsic}_{\text{put}} = \max(0,\; K - S)$$
#
# $$\text{Time} = \text{Price} - \text{Intrinsic}$$
#
# Time value reflects:
# - Time remaining until expiration
# - Expected volatility (IV)
# - Interest rates and dividends

# %% [markdown]
# ## 2. Dataset Overview
#
# ### Data schema
#
# | Field | Type | Description |
# |-------|------|-------------|
# | **Identifiers** | | |
# | `timestamp` | Date | Trading date (observation date, EOD snapshot) |
# | `symbol` | String | Underlying ticker (e.g., "AAPL", "MSFT") |
# | `expiration` | Date | Option expiration date |
# | `strike` | Float64 | Strike price in USD |
# | `call_put` | String | "C" for call, "P" for put |
# | **Prices** | | |
# | `bid` | Float64 | Best bid price at close |
# | `ask` | Float64 | Best ask price at close |
# | `mid_price` | Float64 | Mid-market price: (bid + ask) / 2 |
# | `underlying_price` | Float64 | Underlying stock close price |
# | **Time** | | |
# | `days_to_maturity` | Int32 | Calendar days until expiration |
# | **Greeks** | | |
# | `delta` | Float64 | ∂V/∂S - Price sensitivity to underlying |
# | `gamma` | Float64 | ∂²V/∂S² - Delta sensitivity to underlying |
# | `theta` | Float64 | ∂V/∂t - Time decay ($/day, typically negative) |
# | `vega` | Float64 | ∂V/∂σ - Sensitivity to volatility |
# | `rho` | Float64 | ∂V/∂r - Sensitivity to interest rates |
# | **Volatility** | | |
# | `implied_vol` | Float64 | Black-Scholes implied volatility |
# | `iv_convergence` | String | IV solver status (quality indicator) |
#
# ### IV convergence codes
#
# `iv_convergence` records how the vendor's solver arrived at each implied volatility, and it
# is the field that decides which rows are usable. The code is a compound of two parts: what
# the solver was given (`Converged` from a normal quote, `SmallBid` from a bid near zero,
# `IntrVal` from a price at intrinsic value, `Failed` when it could not solve at all) and, where
# the direct solve failed, how the number was produced instead (`FlatExtrapol`, `LinInterp`, or
# `PutCallPair` from put-call parity).
#
# The full set present in the data is read from the data below rather than listed here. A
# hand-written table of solver codes goes stale the first time the vendor adds one, and a
# reader who trusts it will filter against a set that no longer matches the file.
#
# Only `Converged` is used for analysis in this notebook. Everything else is either
# extrapolated, interpolated, or derived from the other side of the pair, and none of those is
# an implied volatility solved from the quote in front of it.

# %%
options = load_sp500_options_eda(
    start_date=START_DATE,
    end_date=END_DATE,
    include_greeks=True,
)

print("=== S&P 500 Options Dataset ===")
print(f"Total rows: {len(options):,}")
print(f"Columns: {len(options.columns)}")
print(f"Date range: {options['timestamp'].min()} to {options['timestamp'].max()}")
print(f"Underlyings: {sorted(options['symbol'].unique().to_list())}")

# %%
daily = load_sp500_daily_bars(
    symbols=sorted(options["symbol"].unique().to_list()),
    start_date=DAILY_START_DATE,
    end_date=END_DATE,
)

print("\n=== S&P 500 Daily Prices ===")
print(f"Total rows: {len(daily):,}")
print(f"Symbols: {daily['symbol'].n_unique()}")
print(f"Date range: {daily['timestamp'].min()} to {daily['timestamp'].max()}")

# %% [markdown]
# ### What the solver actually returned

# %%
convergence_inventory = (
    options.group_by("iv_convergence")
    .len()
    .with_columns((100 * pl.col("len") / pl.sum("len")).alias("pct"))
    .sort("len", descending=True)
)
print(f"Distinct iv_convergence codes in this file: {convergence_inventory.height}")
print(
    f"Rows solved directly from the quote: "
    f"{convergence_inventory.filter(pl.col('iv_convergence') == 'Converged')['pct'][0]:.2f}%"
)
convergence_inventory

# %% [markdown]
# ### Schema preview

# %%
options.head(3)

# %% [markdown]
# ## 3. Option Chain Structure
#
# An **option chain** is the full set of options available for one underlying on one day.
# It spans multiple dimensions:
# - **Strikes**: Many price levels around the current spot
# - **Expirations**: Multiple dates from days to years out
# - **Types**: Calls and puts at each strike/expiration
#
# This creates a 3D grid: `(strike × expiration × call_put)`

# %%
# Options per symbol per day - how dense are the chains?
options_per_symbol = options.group_by(["timestamp", "symbol"]).agg(
    [
        pl.len().alias("n_options"),
        (pl.col("call_put") == "C").sum().alias("n_calls"),
        (pl.col("call_put") == "P").sum().alias("n_puts"),
        pl.col("strike").n_unique().alias("n_strikes"),
        pl.col("expiration").n_unique().alias("n_expirations"),
    ]
)

print("=== Option Chain Density (per symbol per day) ===")
options_per_symbol.select(["n_options", "n_strikes", "n_expirations"]).describe()

# %%
# Visualize: Distribution of chain sizes
fig = px.histogram(
    options_per_symbol.to_pandas(),
    x="n_options",
    nbins=50,
    title="Options per symbol per day",
    labels={"n_options": "Number of Options per Symbol/Day", "count": "Frequency"},
)
median_options = float(options_per_symbol["n_options"].median())
fig.add_vline(x=median_options, line_dash="dash", line_color=COLORS["negative"])
fig.add_annotation(x=median_options, y=0.95, yref="paper", text="Median", showarrow=False)
fig.update_layout(showlegend=False)
show_plotly_with_alt(
    fig,
    "A histogram of how many option contracts each underlying carries on each trading day, with a dashed vertical line marking the median. The bulk of the mass sits in a broad hump with a long tail to the right.",
)

# %% [markdown]
# ### One chain in full: AAPL
#
# Let's examine one complete option chain to understand the structure.

# %%
# AAPL on a specific date
sample_date = options["timestamp"].max()
aapl_day = options.filter((pl.col("symbol") == "AAPL") & (pl.col("timestamp") == sample_date))

spot = aapl_day["underlying_price"][0]

print(f"=== AAPL Option Chain ({sample_date}) ===")
print(f"Underlying price: ${spot:.2f}")
print(f"Total options: {len(aapl_day):,}")
print(f"  Calls: {aapl_day.filter(pl.col('call_put') == 'C').height:,}")
print(f"  Puts: {aapl_day.filter(pl.col('call_put') == 'P').height:,}")
print(f"Expirations: {aapl_day['expiration'].n_unique()}")
print(f"Strikes: {aapl_day['strike'].n_unique()}")
print(f"Strike range: ${aapl_day['strike'].min():.2f} - ${aapl_day['strike'].max():.2f}")

# %%
# Expiration breakdown
exp_breakdown = (
    aapl_day.group_by("expiration")
    .agg([pl.len().alias("n_options"), pl.col("strike").n_unique().alias("n_strikes")])
    .sort("expiration")
)
print("\n=== AAPL Expirations ===")
exp_breakdown.head(10)

# %% [markdown]
# ### The chain as a heatmap
#
# Strikes up the vertical axis, expirations across the horizontal, shaded by implied
# volatility: the whole surface in one picture.
#
# It is drawn on the near-dated part of the chain only. The full chain runs out past two years,
# and those far expirations carry a handful of strikes each. Plotted on a categorical axis they
# take most of the width while the near-dated expirations, which carry almost every strike, are
# crushed into a sliver at the left - so the chart would be mostly empty in the region where
# there is least to see.

# %%
# Prepare data for heatmap - calls only, converged IV, reasonable moneyness
aapl_calls = (
    aapl_day.filter(
        (pl.col("call_put") == "C")
        & (pl.col("iv_convergence") == "Converged")
        & (pl.col("implied_vol") > 0)
        & (pl.col("implied_vol") < 2.0)  # Filter outliers
    )
    .with_columns((pl.col("strike") / pl.col("underlying_price")).alias("moneyness"))
    .filter(pl.col("moneyness").is_between(*CHAIN_BAND))
)

heatmap_source = aapl_calls.filter(pl.col("days_to_maturity") <= SURFACE_MAX_DAYS)
heatmap_data = (
    heatmap_source.select(["strike", "expiration", "implied_vol"])
    .sort(["expiration", "strike"])
    .to_pandas()
    .pivot(index="strike", columns="expiration", values="implied_vol")
)
print(
    f"Heatmap covers {heatmap_data.shape[1]} expirations within {SURFACE_MAX_DAYS} days and "
    f"{heatmap_data.shape[0]} strikes"
)

# %%
fig = go.Figure(
    data=go.Heatmap(
        z=heatmap_data.values,
        x=[str(c) for c in heatmap_data.columns],
        y=heatmap_data.index,
        colorscale=[[0, COLORS["silver"]], [1, COLORS["blue"]]],
        colorbar=dict(title="IV"),
    )
)

spot_float = float(spot)
fig.add_hline(y=spot_float, line_dash="dash", line_color=COLORS["negative"])
fig.add_annotation(
    y=spot_float,
    x=0.02,
    xref="paper",
    yshift=12,
    text="Spot",
    showarrow=False,
    font=dict(color=COLORS["negative"]),
)

fig.update_layout(
    title="AAPL option chain: implied volatility by strike and expiration",
    xaxis_title="Expiration",
    yaxis_title="Strike ($)",
    height=600,
)
show_plotly_with_alt(
    fig,
    "A heatmap with expiration dates along the horizontal axis and strike prices up the vertical axis, shaded from pale for low implied volatility to dark for high, with a dashed horizontal line marking the spot price. The darkest cells sit in the bottom-left corner, at the lowest strikes and nearest expirations. Moving up the strike axis the shading fades to almost nothing in a band around the spot line, then darkens again to a uniform mid grey across the highest strikes. Moving right along the expiration axis the shading flattens out. Many cells are blank, because a strike is listed for some expirations and not others.",
)

# %% [markdown]
# **Reading the heatmap.** A horizontal slice at one strike is a term structure: how the price
# of volatility for that strike changes as the expiration moves out. A vertical slice at one
# expiration is a smile: how it changes as the strike moves away from spot. Darker is higher
# implied volatility.
#
# The darkest cells sit at the bottom left, at the lowest strikes and the nearest expirations.
# That is the crash-protection corner of the chain, and it is the most expensive volatility on
# the board.
#
# Read a column from bottom to top and the smile is there as shading rather than as a curve:
# dark at the low strikes, almost white in a band around the spot line, and darkening again
# across the high strikes. The two ends are not equally dark, which is the skew - with the
# caveat developed in the smile section below that these are calls, so the low-strike end is
# in-the-money calls rather than the out-of-the-money puts the usual explanation names.
# Read a row from left to right and the variation flattens as the expiration moves out, which
# is the term structure converging on a long-run level.
#
# The blank cells are not missing data in the sense of a defect. A strike is listed for some
# expirations and not others, so the grid is genuinely sparse, and any surface model fitted to
# it has to interpolate across those gaps rather than assume them filled.

# %% [markdown]
# ## 4. Volatility Surface Analysis
#
# The **implied volatility surface** is the core representation for options analytics.
# It captures how IV varies across two dimensions:
# 1. **Moneyness** (strike relative to spot) → IV smile/skew
# 2. **Time to expiration** → IV term structure
#
# ### The smile and the skew

# %%
# IV smile for nearest expiration
nearest_exp = aapl_calls["expiration"].min()
aapl_smile = aapl_calls.filter(pl.col("expiration") == nearest_exp).sort("strike")

fig = px.scatter(
    aapl_smile.to_pandas(),
    x="moneyness",
    y="implied_vol",
    title="AAPL implied volatility by moneyness, nearest expiration",
    labels={"moneyness": "Moneyness (Strike/Spot)", "implied_vol": "Implied Volatility"},
    trendline="lowess",
)
fig.add_vline(x=1.0, line_dash="dash", line_color=COLORS["neutral"])
fig.add_annotation(x=1.0, y=0.95, yref="paper", text="ATM", showarrow=False)
show_plotly_with_alt(
    fig,
    "A scatter of implied volatility against moneyness for the nearest expiration, with a smoothed trend line through it and a dashed vertical line at moneyness one. The points trace a clear U: falling from the left edge to a flat minimum just above moneyness one, then rising steadily across the whole right half to finish well above where they started.",
)

# %% [markdown]
# The curve is a smile: implied volatility is lowest close to the money and rises on both
# sides. Strikes far from spot cost more volatility than strikes near it, in either direction.
#
# It is also not symmetric, and that asymmetry is the skew. What needs care is the reading
# usually attached to it. Skew is normally explained as the market paying more for downside
# protection than for upside, and that explanation compares out-of-the-money **puts** with
# out-of-the-money calls. Every point on this curve is a call, so its left half is in-the-money
# calls rather than out-of-the-money puts.
#
# Those are different contracts, and reading one as the other is a step that needs justifying
# rather than assuming. Put-call parity is the justification: it ties a call and a put at the
# same strike and expiration together tightly enough that they should imply nearly the same
# volatility. Whether they do in this file is a question the file can answer, so the next cell
# asks it.

# %%
_exp = nearest_exp
_pair = (
    aapl_day.filter(
        (pl.col("expiration") == _exp)
        & (pl.col("iv_convergence") == "Converged")
        & (pl.col("implied_vol") > 0)
    )
    .select("strike", "call_put", "implied_vol")
    .pivot(on="call_put", index="strike", values="implied_vol")
    .drop_nulls()
    .with_columns((pl.col("C") - pl.col("P")).abs().alias("iv_gap"))
    .sort("strike")
)
print(f"Strikes with a converged IV on both sides at the nearest expiration: {_pair.height}")
if _pair.height:
    print(f"  median absolute call-put IV difference: {_pair['iv_gap'].median():.4f}")
    print(f"  90th percentile: {_pair['iv_gap'].quantile(0.9):.4f}")
    print(f"  largest: {_pair['iv_gap'].max():.4f}")

# %%
# Compare smile across multiple expirations
expirations = sorted(aapl_calls["expiration"].unique().to_list())[:4]  # First 4

smile_data = aapl_calls.filter(pl.col("expiration").is_in(expirations))

fig = px.scatter(
    smile_data.to_pandas(),
    x="moneyness",
    y="implied_vol",
    color="expiration",
    title="AAPL implied volatility by moneyness, four expirations",
    labels={"moneyness": "Moneyness", "implied_vol": "Implied Volatility"},
)
fig.add_vline(x=1.0, line_dash="dash", line_color=COLORS["neutral"])
show_plotly_with_alt(
    fig,
    "A scatter of implied volatility against moneyness, coloured by expiration date, with a dashed vertical line at moneyness one. Each expiration forms its own curve; the curves are stacked rather than overlapping.",
)

# %% [markdown]
# ### The term structure
#
# How does ATM IV vary across expirations?

# %%
# ATM IV term structure (moneyness 0.98-1.02)
atm_term = (
    aapl_calls.filter(pl.col("moneyness").is_between(*ATM_BAND))
    .group_by("expiration")
    .agg(
        [
            pl.col("implied_vol").mean().alias("iv_atm"),
            pl.col("days_to_maturity").first().alias("days"),
        ]
    )
    .sort("expiration")
)

fig = px.line(
    atm_term.to_pandas(),
    x="days",
    y="iv_atm",
    markers=True,
    title="AAPL at-the-money implied volatility by days to expiration",
    labels={"days": "Days to Expiration", "iv_atm": "ATM Implied Volatility"},
)
show_plotly_with_alt(
    fig,
    "A line with markers showing at-the-money implied volatility against days to expiration, running out beyond two years. The line begins low at the shortest expiry, jumps sharply to its highest point within the first month or so, falls back over the next hundred days, and then runs almost flat and slightly below that peak across the entire remaining range.",
)

# %% [markdown]
# **Term structure shapes:**
# - **Contango** (upward sloping): Near-term calm, uncertainty further out
# - **Backwardation** (downward sloping): Near-term stress/event expected
# - **Flat**: Consistent expectations across horizons

# %% [markdown]
# ### Both dimensions at once
#
# Combine moneyness and time dimensions into a single surface visualization.

# %%
# Prepare surface data
surface_data = (
    aapl_calls.filter(pl.col("days_to_maturity") <= SURFACE_MAX_DAYS)
    .select(["moneyness", "days_to_maturity", "implied_vol"])
    .to_pandas()
)

# Create 3D surface
fig = go.Figure(
    data=[
        go.Mesh3d(
            x=surface_data["moneyness"],
            y=surface_data["days_to_maturity"],
            z=surface_data["implied_vol"],
            intensity=surface_data["implied_vol"],
            colorscale=[[0, COLORS["silver"]], [1, COLORS["blue"]]],
            opacity=0.7,
        )
    ]
)

fig.update_layout(
    title="AAPL implied volatility by moneyness and days to expiration",
    scene=dict(
        xaxis_title="Moneyness",
        yaxis_title="Days to Expiration",
        zaxis_title="Implied Volatility",
    ),
    height=600,
)
show_plotly_with_alt(
    fig,
    "A three-dimensional mesh surface with moneyness on one horizontal axis, days to expiration on the other, and implied volatility as height, shaded by the same height. The surface slopes across both axes rather than being flat.",
)

# %% [markdown]
# ## 5. Cross-Sectional Analysis
#
# How does ATM IV differ across the eight underlyings on a single day? The same
# logic scales to the full S&P 500 universe — here we keep the comparison
# tractable on the EDA slice.

# %%
# Compute ATM IV for all symbols on sample date
converged = options.filter(pl.col("iv_convergence") == "Converged")

cross_section = (
    converged.filter(pl.col("timestamp") == sample_date)
    .with_columns((pl.col("strike") / pl.col("underlying_price")).alias("moneyness"))
    .filter(pl.col("moneyness").is_between(*ATM_BAND))
    .filter(pl.col("call_put") == "C")
    .group_by("symbol")
    .agg(
        [
            pl.col("implied_vol").mean().alias("iv_atm"),
            pl.col("underlying_price").first().alias("price"),
        ]
    )
    .sort("iv_atm", descending=True)
)

print(f"=== Cross-Sectional ATM IV ({sample_date}) ===")
print(f"Symbols: {len(cross_section)}")
print(f"IV range: {cross_section['iv_atm'].min():.1%} - {cross_section['iv_atm'].max():.1%}")
print(f"IV median: {cross_section['iv_atm'].median():.1%}")

# %%
# Highest IV names
print("\n=== Highest IV Names ===")
cross_section.head(10)

# %%
# Lowest IV names
print("\n=== Lowest IV Names ===")
cross_section.tail(10)

# %%
# IV distribution across universe
fig = px.histogram(
    cross_section.to_pandas(),
    x="iv_atm",
    nbins=40,
    title="At-the-money implied volatility across the eight underlyings",
    labels={"iv_atm": "ATM Implied Volatility", "count": "Number of Symbols"},
)
median_iv = float(cross_section["iv_atm"].median())
fig.add_vline(x=median_iv, line_dash="dash", line_color=COLORS["negative"])
fig.add_annotation(x=median_iv, y=0.95, yref="paper", text="Median", showarrow=False)
show_plotly_with_alt(
    fig,
    "A histogram of at-the-money implied volatility across the eight underlyings on a single day, with a dashed vertical line at the median. With only eight values the bars are sparse.",
)

# %% [markdown]
# ## 6. Time Series Analysis
#
# How does IV evolve over time? The year 2020 provides an excellent case study
# with the COVID crash in March.

# %%
# Daily aggregate IV statistics
daily_iv = (
    converged.with_columns((pl.col("strike") / pl.col("underlying_price")).alias("moneyness"))
    .filter(pl.col("moneyness").is_between(*ATM_BAND))
    .filter(pl.col("call_put") == "C")
    .group_by("timestamp")
    .agg(
        [
            pl.col("implied_vol").mean().alias("iv_mean"),
            pl.col("implied_vol").median().alias("iv_median"),
            pl.col("implied_vol").quantile(0.1).alias("iv_p10"),
            pl.col("implied_vol").quantile(0.9).alias("iv_p90"),
            pl.col("symbol").n_unique().alias("n_symbols"),
        ]
    )
    .sort("timestamp")
)

# %% [markdown]
# The band and the line are built in a single cell. Splitting a plotly figure across two cells
# renders the half-built version as well, and a chart with no title or axis labels ships into
# the notebook above the finished one.

# %%
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=daily_iv["timestamp"].to_list(),
        y=daily_iv["iv_p90"].to_list(),
        fill=None,
        mode="lines",
        line_color=COLORS["slate"],
        name="P90",
    )
)
fig.add_trace(
    go.Scatter(
        x=daily_iv["timestamp"].to_list(),
        y=daily_iv["iv_p10"].to_list(),
        fill="tonexty",
        mode="lines",
        line_color=COLORS["slate"],
        name="P10-P90 Range",
    )
)
fig.add_trace(
    go.Scatter(
        x=daily_iv["timestamp"].to_list(),
        y=daily_iv["iv_median"].to_list(),
        mode="lines",
        line_color=COLORS["blue"],
        line_width=2,
        name="Median IV",
    )
)

_iv_peak_date = daily_iv.sort("iv_p90", descending=True)[0, "timestamp"]
fig.add_vline(x=_iv_peak_date, line_dash="dash", line_color=COLORS["negative"])
fig.add_vline(x=SPX_TROUGH_DATE, line_dash="dash", line_color=COLORS["positive"])
fig.add_annotation(x=_iv_peak_date, y=0.95, yref="paper", text="IV peak", showarrow=False)
fig.add_annotation(x=SPX_TROUGH_DATE, y=0.90, yref="paper", text="S&P 500 trough", showarrow=False)

fig.update_layout(
    title="At-the-money implied volatility over 2020",
    xaxis_title="Date",
    yaxis_title="Implied Volatility",
    height=500,
)
show_plotly_with_alt(
    fig,
    "A time series across 2020 of the median at-the-money implied volatility across the eight underlyings, drawn as a line inside a shaded band running from the tenth to the ninetieth percentile. The band is narrow and low through January and February, jumps to several times its previous width and height in March, and narrows and falls gradually through the rest of the year without returning to where it started. Two dashed vertical lines mark dates in March.",
)

# %% [markdown]
# %%
_peak = daily_iv.sort("iv_p90", descending=True)[0]
_jan = daily_iv.filter(pl.col("timestamp").dt.month() == 1)
_dec = daily_iv.filter(pl.col("timestamp").dt.month() == 12)
print(f"Highest 90th-percentile ATM IV: {_peak['iv_p90'][0]:.1%} on {_peak['timestamp'][0]}")
print(f"  cross-sectional median that day: {_peak['iv_median'][0]:.1%}")
print(
    f"  days on which the 90th percentile exceeded 100%: {daily_iv.filter(pl.col('iv_p90') > 1.0).height}"
)
print(f"January median ATM IV: {_jan['iv_median'].mean():.1%}")
print(f"December median ATM IV: {_dec['iv_median'].mean():.1%}")

# %% [markdown]
# The spike is sharp and the decay is not. Implied volatility multiplies within weeks in March
# and is still above where it started when the year ends, months after the index itself had
# recovered. That asymmetry is the property that makes implied volatility worth carrying as a
# feature: it is not a restatement of the price move, because it does not come back on the same
# schedule.
#
# The dashed lines make the same point in a second way. The day the market put the highest
# price on future volatility is not the day the market bottomed - the peak in expected
# volatility leads the trough in price. Naming both lines something like "the COVID low" would
# have obscured exactly the gap the chart exists to show.

# %% [markdown]
# ## 7. Execution Cost Proxy: Bid-Ask Spreads
#
# Since we don't have volume or open interest, bid-ask spread serves as our
# primary liquidity/execution cost indicator.

# %%
# Compute spread metrics
spread_analysis = converged.with_columns(
    [
        (pl.col("ask") - pl.col("bid")).alias("spread_abs"),
        ((pl.col("ask") - pl.col("bid")) / pl.col("mid_price")).alias("spread_pct"),
        (pl.col("strike") / pl.col("underlying_price")).alias("moneyness"),
    ]
).filter(pl.col("mid_price") > MIN_MID_PRICE)

print("=== Bid-Ask Spread Statistics ===")
spread_analysis.select(["spread_abs", "spread_pct"]).describe()

# %% [markdown]
# Spread is bucketed by moneyness in five-percent steps, and both the median and the ninetieth
# percentile are reported for each bucket. The median alone would understate the problem: what
# makes a strike untradeable is not its typical spread but how often it is quoted far wider
# than typical, and those two diverge sharply as you move away from the money.

# %%
spread_by_moneyness = (
    spread_analysis.filter(pl.col("moneyness").is_between(*SPREAD_BAND))
    .with_columns((pl.col("moneyness") * 20).round() / 20)
    .group_by("moneyness")
    .agg(
        pl.col("spread_pct").median().alias("median_spread"),
        pl.col("spread_pct").quantile(0.9).alias("p90_spread"),
        (pl.col("spread_pct") > 0.5).mean().alias("share_over_50pct"),
        pl.len().alias("n"),
    )
    .sort("moneyness")
)

_spread_pd = spread_by_moneyness.to_pandas()
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=_spread_pd["moneyness"],
        y=_spread_pd["median_spread"],
        name="Median",
        marker_color=COLORS["blue"],
    )
)
fig.add_trace(
    go.Scatter(
        x=_spread_pd["moneyness"],
        y=_spread_pd["p90_spread"],
        name="90th percentile",
        mode="lines+markers",
        line=dict(color=COLORS["amber"], width=2),
    )
)
fig.update_layout(
    title="Bid-ask spread by moneyness",
    xaxis_title="Moneyness (strike / spot)",
    yaxis_title="Spread as a share of mid price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
fig.add_vline(x=1.0, line_dash="dash", line_color=COLORS["neutral"])
fig.add_annotation(x=1.0, y=0.95, yref="paper", text="ATM", showarrow=False)
show_plotly_with_alt(
    fig,
    "Bars of the median bid-ask spread as a percentage of the mid price, against moneyness buckets, with a dashed vertical line at moneyness one.",
)

# %% [markdown]
# The two summaries below are computed over the quotes themselves rather than over the chart's
# buckets. The buckets hold very unequal numbers of quotes, so a median of per-bucket ninetieth
# percentiles is not the ninetieth percentile of anything, and averaging per-bucket breach
# shares would weight a thin far-wing bucket the same as a crowded one beside the money.
#
# Either band can come back empty, and which one does depends on how wide the chain is: a
# chain with strikes on one side of the wing boundary and none on the other empties that
# side. A quantile of an empty frame is null, and formatting a null against a percentage
# spec raises, so the cell says the band is empty rather than printing a number for a set
# with nothing in it.

# %%
_in_band = spread_analysis.filter(pl.col("moneyness").is_between(*SPREAD_BAND))
_near = _in_band.filter(pl.col("moneyness").is_between(*WING_BAND))
_wings = _in_band.filter(~pl.col("moneyness").is_between(*WING_BAND))

for label, frame in [("Near the money", _near), ("Away from it", _wings)]:
    if not frame.height:
        print(f"{label:16s} n={0:9,}  no contracts in this band, nothing to summarise")
        continue
    print(
        f"{label:16s} n={frame.height:9,}  median {frame['spread_pct'].median():6.1%}  "
        f"90th pct {frame['spread_pct'].quantile(0.9):6.1%}  "
        f"wider than half the mid {(frame['spread_pct'] > 0.5).mean():5.1%}"
    )

# %% [markdown]
# The median spread barely moves between the two groups, and reading only the medians would
# suggest that moneyness costs a strategy almost nothing. The ninetieth percentile tells a
# different story and it is the one that binds: away from the money it is well over twice what
# it is near the money, and quotes wider than half the mid price are several times as common.
#
# The practical consequence is about which statistic to trade on. A cost model calibrated to
# the median spread of this universe would be roughly right at the money and badly optimistic
# in the wings, and the wings are exactly where a strategy buying convexity wants to operate.
#
# This is also as far as the data can take the question. There is no volume and no open
# interest in this file, so spread is standing in for liquidity rather than measuring it, and a
# wide spread on a contract nobody trades means something different from a wide spread on one
# that trades all day.

# %% [markdown]
# ## 8. Data Quality Assessment
#
# ### IV convergence rates

# %%
# Convergence statistics
convergence_stats = (
    options.group_by("iv_convergence")
    .len()
    .with_columns((pl.col("len") / pl.sum("len") * 100).alias("pct"))
    .sort("len", descending=True)
)

print("=== IV Convergence Status ===")
for row in convergence_stats.iter_rows(named=True):
    print(f"  {row['iv_convergence']}: {row['len']:,} ({row['pct']:.2f}%)")

# %% [markdown]
# A horizontal bar chart rather than a pie. The codes are numerous and their shares are wildly
# uneven, so as a pie the small categories become slivers whose labels overlap each other and
# cannot be read at all - and the small categories are the interesting ones here, because they
# are the rows a reader has to decide whether to keep.

# %%
convergence_pd = convergence_stats.to_pandas().sort_values("pct", ascending=True)
fig = go.Figure(
    data=go.Bar(
        x=convergence_pd["pct"],
        y=convergence_pd["iv_convergence"],
        orientation="h",
        text=[f"{v:.2f}%" for v in convergence_pd["pct"]],
        textposition="outside",
        marker_color=COLORS["blue"],
    )
)
fig.update_layout(
    title="Share of rows by IV convergence code",
    xaxis_title="Share of rows (%)",
    yaxis_title="Convergence status",
    height=460,
    margin=dict(l=170, r=100),
    xaxis=dict(range=[0, max(convergence_pd["pct"]) * 1.15]),
)
show_plotly_with_alt(
    fig,
    "A horizontal bar chart of what share of rows each IV convergence code accounts for, sorted with the largest at the top and the percentage written at the end of each bar. One bar is far longer than the rest.",
)

# %% [markdown]
# ### Coverage over time

# %%
# Daily symbol coverage
daily_coverage = (
    converged.group_by("timestamp")
    .agg(
        [
            pl.col("symbol").n_unique().alias("n_symbols"),
            pl.len().alias("n_options"),
        ]
    )
    .sort("timestamp")
)

print("=== Daily Coverage (Converged Options) ===")
print(f"Mean symbols/day: {daily_coverage['n_symbols'].mean():.0f}")
print(f"Min symbols/day: {daily_coverage['n_symbols'].min()}")
print(f"Max symbols/day: {daily_coverage['n_symbols'].max()}")

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=("Symbols with Converged Options", "Total Converged Options"),
)

coverage_pd = daily_coverage.to_pandas()

fig.add_trace(
    go.Scatter(x=coverage_pd["timestamp"], y=coverage_pd["n_symbols"], name="Symbols"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=coverage_pd["timestamp"], y=coverage_pd["n_options"], name="Options"),
    row=2,
    col=1,
)

fig.update_layout(height=500, title="Converged options available per day")
show_plotly_with_alt(
    fig,
    "Two stacked time-series panels sharing a date axis across 2020. The upper panel counts the underlyings with converged options each day and the lower counts the converged contracts.",
)

# %% [markdown]
# ### Greeks validation
#
# Four of these five bounds are arithmetic facts about the Black-Scholes formula rather than
# properties a dataset may or may not have: delta lies in [-1, 1], gamma and vega are
# non-negative for a long option, and an implied volatility is positive. A breach of any of
# them is a defect in the vendor's numbers.
#
# The fifth is not. Theta is non-positive for almost every option but not for all of them: a
# deep in-the-money European put earns interest on the strike it will receive, and that can
# outweigh time decay. So the theta line is expected to fall short of a hundred percent, and
# what matters is whether the shortfall is the size that explanation predicts.
#
# Each line therefore reports its exact share and the number of rows outside the bound, rather
# than a pass or fail against a threshold. A threshold here would be a number chosen to make
# theta pass, and it would hide the one result worth looking at.

# %%
checks = converged.select(
    [
        ((pl.col("delta") >= -1.0) & (pl.col("delta") <= 1.0)).mean().alias("delta_within_pm1"),
        (pl.col("gamma") >= 0).mean().alias("gamma_non_negative"),
        (pl.col("vega") >= 0).mean().alias("vega_non_negative"),
        (pl.col("theta") <= 0).mean().alias("theta_non_positive"),
        (pl.col("implied_vol") > 0).mean().alias("iv_positive"),
    ]
)

print(f"Greeks bounds over {len(converged):,} converged rows:")
for col in checks.columns:
    share = checks[col][0]
    breaches = round((1 - share) * len(converged))
    verdict = "holds" if breaches == 0 else f"{breaches:,} rows outside"
    print(f"  {col:22s} {100 * share:8.4f}%   {verdict}")

# %%
print("\n=== Greeks Summary Statistics ===")
converged.select(["delta", "gamma", "theta", "vega", "implied_vol"]).describe()

# %% [markdown]
# ### Point-in-time validation

# %%
print("=== Point-in-Time Checks ===")

# Expiration must be >= observation date
exp_check = options.filter(pl.col("expiration") < pl.col("timestamp"))
print(f"Expiration < Date violations: {len(exp_check):,}")

# Days to maturity must be non-negative
dtm_check = options.filter(pl.col("days_to_maturity") < 0)
print(f"Negative days_to_maturity: {len(dtm_check):,}")

if len(exp_check) == 0 and len(dtm_check) == 0:
    print("[OK] PASSED - No look-ahead bias detected")
else:
    print("[FAIL] FAILED - Data integrity issue")

# %% [markdown]
# ## 9. Information Content Preview
#
# Why does options data predict underlying returns? Let's examine the IV-return
# relationship.

# %%
# Compute ATM IV per symbol/date
atm_iv = (
    converged.with_columns((pl.col("strike") / pl.col("underlying_price")).alias("moneyness"))
    .filter(pl.col("moneyness").is_between(*ATM_BAND))
    .filter(pl.col("call_put") == "C")
    .with_columns((pl.col("moneyness") - 1.0).abs().alias("atm_distance"))
    .sort(["timestamp", "symbol", "atm_distance"])
    .group_by(["timestamp", "symbol"])
    .first()
    .select(["timestamp", "symbol", "implied_vol", "underlying_price"])
    .rename({"implied_vol": "iv_atm"})
)

# Join with daily prices and compute returns
panel = (
    atm_iv.join(
        daily.select(["timestamp", "symbol", "close"]), on=["timestamp", "symbol"], how="inner"
    )
    .sort(["symbol", "timestamp"])
    .with_columns(
        [
            pl.col("iv_atm").shift(FORWARD_DAYS).over("symbol").alias("iv_atm_lag5"),
            pl.col("close").shift(-FORWARD_DAYS).over("symbol").alias("close_fwd5"),
        ]
    )
    .with_columns(
        [
            (pl.col("iv_atm") - pl.col("iv_atm_lag5")).alias("iv_change_5d"),
            ((pl.col("close_fwd5") / pl.col("close")) - 1).alias("ret_fwd5"),
        ]
    )
    .drop_nulls(subset=["iv_change_5d", "ret_fwd5"])
)

correlation = panel.select(pl.corr("iv_change_5d", "ret_fwd5").alias("corr"))[0, 0]
print(f"Rows: {len(panel):,} over {panel['symbol'].n_unique()} symbols")
print(
    f"Correlation of {FORWARD_DAYS}-day IV change with {FORWARD_DAYS}-day forward return: "
    f"{correlation:+.3f}"
)

# %% [markdown]
# ### What that correlation is, and what it is not
#
# The number is small and negative, and before it is interpreted it is worth being explicit
# about what would make it unreliable.
#
# **The observations are not independent.** Both sides of the correlation are computed over
# rolling windows on daily data, so consecutive rows share four of their five days. The
# effective sample is roughly a fifth of the row count, and any significance test that treated
# these rows as independent would overstate its confidence by more than a factor of two.
#
# **There are eight underlyings, not a cross section.** Everything below aggregates across a
# handful of large-cap names in a single year, so a result that holds for the aggregate can
# rest on one or two of them.
#
# **The year is 2020.** A single episode moved implied volatility and price together and
# violently, and a full-sample correlation could be that episode and nothing else.
#
# The first is a caveat that has to be stated and cannot be removed here. The second and third
# can be checked, so they are.

# %%
_by_symbol = (
    panel.group_by("symbol")
    .agg(pl.corr("iv_change_5d", "ret_fwd5").alias("corr"), pl.len().alias("n"))
    .sort("corr")
)

_crash = panel.filter(
    pl.col("timestamp").dt.date().is_between(date(2020, 2, 15), date(2020, 4, 30))
)
_calm = panel.filter(
    ~pl.col("timestamp").dt.date().is_between(date(2020, 2, 15), date(2020, 4, 30))
)
_spaced = (
    panel.sort(["symbol", "timestamp"])
    .with_columns(pl.int_range(pl.len()).over("symbol").alias("_i"))
    .filter(pl.col("_i") % FORWARD_DAYS == 0)
)

print(f"Whole sample:                     n={len(panel):5,}  corr {correlation:+.3f}")
print(
    f"Excluding 15 Feb to 30 Apr 2020:  n={len(_calm):5,}  corr "
    f"{_calm.select(pl.corr('iv_change_5d', 'ret_fwd5'))[0, 0]:+.3f}"
)
print(
    f"Inside that window only:          n={len(_crash):5,}  corr "
    f"{_crash.select(pl.corr('iv_change_5d', 'ret_fwd5'))[0, 0]:+.3f}"
)
print(
    f"Non-overlapping windows only:     n={len(_spaced):5,}  corr "
    f"{_spaced.select(pl.corr('iv_change_5d', 'ret_fwd5'))[0, 0]:+.3f}"
)
print(
    f"\nPer symbol ({_by_symbol.filter(pl.col('corr') < 0).height} of "
    f"{_by_symbol.height} negative):"
)
_by_symbol

# %% [markdown]
# The relationship is not the crash. Removing the ten weeks around it leaves the correlation
# where it was rather than collapsing it, which is the outcome that would have discredited the
# result and did not happen. Dropping to non-overlapping windows does not weaken it either.
#
# The per-symbol table is the weaker part. The sign is shared by most of the eight names but
# not all of them, and the spread across names is wider than the aggregate figure. With eight
# underlyings that is neither surprising nor reassuring: it is simply too few series to tell a
# common effect from a coincidence among a handful of large-cap technology and financial names
# in one year.
#
# So the honest reading is a hypothesis worth carrying forward, not a finding. Chapter 9
# evaluates it as an information coefficient over a real cross section and a longer history,
# which is where it can be confirmed or discarded.

# %%
# Quintile analysis
panel_ranked = (
    panel.with_columns(pl.col("iv_change_5d").rank().over("timestamp").alias("iv_rank_raw"))
    .with_columns(
        (pl.col("iv_rank_raw") / pl.col("iv_rank_raw").max().over("timestamp") * 100).alias(
            "iv_pct"
        )
    )
    .with_columns(
        pl.when(pl.col("iv_pct") <= 20)
        .then(1)
        .when(pl.col("iv_pct") <= 40)
        .then(2)
        .when(pl.col("iv_pct") <= 60)
        .then(3)
        .when(pl.col("iv_pct") <= 80)
        .then(4)
        .otherwise(5)
        .alias("iv_quintile")
    )
)

quintile_returns = (
    panel_ranked.group_by("iv_quintile")
    .agg(
        [
            pl.col("ret_fwd5").mean().alias("mean_ret"),
            pl.col("ret_fwd5").std().alias("std_ret"),
            pl.len().alias("n_obs"),
        ]
    )
    .sort("iv_quintile")
)

print("Forward returns by IV-change quintile (Q1 = largest decrease, Q5 = largest increase)")
print(
    f"Each quintile is formed within a day across {panel['symbol'].n_unique()} symbols, so a "
    f"'quintile' here holds one or two names"
)
quintile_returns

# %%
fig = px.bar(
    quintile_returns.to_pandas(),
    x="iv_quintile",
    y="mean_ret",
    title="Mean forward return by IV-change quintile",
    labels={
        "iv_quintile": "IV Change Quintile (1=falling, 5=rising)",
        "mean_ret": "Mean 5-Day Return",
    },
)
fig.update_layout(xaxis=dict(tickmode="array", tickvals=[1, 2, 3, 4, 5]))
show_plotly_with_alt(
    fig,
    "Bars of the mean forward return for each of five quintiles of five-day implied-volatility change, ordered from the largest decreases on the left to the largest increases on the right. The first bar is much the tallest, the second is about a third of it, the third is near zero, the fourth dips slightly below zero and the fifth returns to just above it.",
)

# %% [markdown]
# The bars are not monotonic. The two quintiles with the largest implied-volatility declines
# carry clearly positive mean returns and the rest are close to zero, with the last bucket
# turning back up rather than continuing down. So what the picture supports is that large IV
# declines precede positive returns, not that the effect is graded across the range - and a
# correlation, which assumes a straight line, is the wrong summary of a shape like this one.
#
# The quintile bars restate the correlation and inherit every one of its limits, plus one of
# their own: dividing eight names into five buckets leaves one or two names per bucket per day,
# so each bar is an average over a handful of observations rather than over a portfolio. The
# shape is worth looking at and the bar heights are not worth quoting.
#
# What the section establishes is a direction and a reason to look further. Falling implied
# volatility lining up with positive forward returns is consistent with volatility risk premium
# being harvested as fear subsides, and it is also consistent with several other stories this
# data cannot separate. Chapter 9 is where the feature is tested properly.

# %% [markdown]
# ## 10. Data Quality Summary

# %%
total_rows = len(options)
converged_rows = len(converged)
converged_pct = converged_rows / total_rows * 100

print("=" * 70)
print("DATA QUALITY SUMMARY: S&P 500 OPTIONS")
print("=" * 70)

print("\n1. SCALE")
print(f"   Total options records: {total_rows:,}")
print(f"   Converged IV records: {converged_rows:,} ({converged_pct:.1f}%)")
print(f"   Unique underlyings: {options['symbol'].n_unique()}")
print(f"   Date range: {options['timestamp'].min()} to {options['timestamp'].max()}")

print("\n2. COVERAGE")
print(f"   Trading days: {daily_coverage['timestamp'].n_unique()}")
print(f"   Avg symbols/day: {daily_coverage['n_symbols'].mean():.0f}")
print(f"   Avg options/symbol/day: {options_per_symbol['n_options'].mean():.0f}")

print("\n3. DATA QUALITY")
print(f"   Point-in-time: {'PASS' if len(exp_check) == 0 else 'FAIL'}")
print("   Greeks validity: See checks above")
print(f"   Convergence rate: {converged_pct:.1f}%")

print("\n4. EXECUTION PROXY")
print(
    f"   Median ATM spread: {spread_by_moneyness.filter(pl.col('moneyness') == 1.0)['median_spread'][0]:.1%}"
)

print("\n5. INFORMATION CONTENT")
print(f"   IV-Return Correlation: {correlation:.4f}")

print("\n" + "=" * 70)
print("DATASET SUPPORTS TWO CASE STUDIES:")
print("  sp500_equity_option_analytics — IV features used to trade equities")
print("  sp500_options                  — short-straddle harvest with daily delta hedge")
print("=" * 70)

# %% [markdown]
# ## Key Takeaways
#
# 1. **The convergence code decides which rows exist.** Roughly a third of this file carries an
#    implied volatility that was extrapolated, interpolated, or inferred from the other side of
#    the put-call pair rather than solved from the quote. The exact share and the full set of
#    codes are printed above rather than written here, because a hand-kept list of solver codes
#    is wrong the first time the vendor adds one - this file carries twice as many codes as the
#    five a reader would think to look for.
#
# 2. **Structure.** Each underlying carries dozens of expirations and many strikes on each day,
#    so a chain is a three-dimensional grid of strike, expiration and side, and every section
#    here is a slice through it.
#
# 3. **Four of the five Greeks bounds are arithmetic, and one is not.** Delta within plus or
#    minus one, non-negative gamma and vega, and positive implied volatility hold on every
#    converged row. Theta does not, and it should not: a deep in-the-money European put earns
#    interest on the strike it will receive, which can outweigh time decay. The shortfall is
#    small and it is the size that explanation predicts. A threshold that made theta "pass"
#    would have hidden the only line worth reading.
#
# 4. **The volatility surface is visible on this slice.** Smile, skew and term structure all
#    show up on eight names in one year, which is what makes the slice usable for teaching the
#    shape even though it is far too small to support a claim about the shape.
#
# 5. **Implied volatility spikes fast and decays slowly.** The 2020 path multiplies within
#    weeks in March and has still not returned to its January level by December, long after the
#    index recovered. The day expected volatility peaked is not the day the index bottomed, and
#    the chart marks both so the gap between them is visible.
#
# 6. **Spread punishes the wings through its tail, not its median.** The median spread is
#    almost the same near the money and away from it; the ninetieth percentile is not, and
#    quotes wider than half the mid price are several times as common in the wings. A cost
#    model fitted to the median would be roughly right at the money and badly optimistic
#    exactly where a strategy buying convexity wants to trade.
#
# 7. **The IV-return relationship is a hypothesis, not a finding.** The correlation is small
#    and negative, and it is much the same figure with the crash window removed and with the
#    sample thinned to non-overlapping windows - but the rows overlap by construction, there are eight
#    underlyings rather than a cross section, and the sign is not shared by all of them.
#    Chapter 9 tests it where it can be tested.
#
# ## Data Limitations
#
# - **No volume/open interest**: Cannot filter for liquidity directly.
# - **Daily snapshots only**: No intraday dynamics for gamma scalping.
# - **Spread as proxy**: Bid-ask is the only execution cost indicator available.
#
# ## Next Steps
#
# - `08_options_greeks_computation`: Compute Greeks from scratch and validate
#   against the vendor numbers used here.
# - Chapter 8: Build IV surface features for ML.
# - Chapter 9: Model-based feature extraction (PCA, autoencoders).
# - Chapter 12: ML models for equity and options targets.
# - Chapter 16: Backtests using these features.
