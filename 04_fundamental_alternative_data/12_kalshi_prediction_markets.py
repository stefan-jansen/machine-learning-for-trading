# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Kalshi Prediction Markets: Regulated Event Contracts
#
# **Chapter 4: Fundamental and Alternative Data**
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Kalshi is the first CFTC-regulated prediction market in the US, offering binary
# contracts on economic, market, and policy events. This notebook loads real Kalshi
# OHLCV data and demonstrates how to build event probability indicators for ML
# feature engineering and regime detection.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Understand Kalshi contract structure and implied probability pricing
# - Load and explore real OHLCV data from the Kalshi API
# - Build event probability indicators for ML pipelines
# - Assess prediction market data quality for systematic use
#
# ## Cross-References
#
# - **Upstream**: `data/prediction_markets/download.py` (fetches data)
# - **Downstream**: Chapter 8 event features, macro regime indicators
# - **Related**: [`13_polymarket_prediction_markets`](13_polymarket_prediction_markets.ipynb) (crypto-based alternative)

# %%
"""Kalshi Prediction Markets - build event probability indicators from regulated binary contracts."""

import warnings

warnings.filterwarnings("ignore")

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data.prediction_markets.loader import load_kalshi
from utils.paths import get_output_dir
from utils.style import COLORS

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides for CI

# %% [markdown]
# ## 1. Kalshi Contract Structure
#
# Kalshi contracts are binary event contracts that settle at \$0 or \$1.
# The contract price represents the market's implied probability of the event.
#
# | Feature | Description |
# |---------|-------------|
# | **Regulation** | CFTC-regulated (legal in US) |
# | **Settlement** | USD (real dollars) |
# | **Position Limit** | \$25,000 per contract |
# | **Trading Hours** | 24/7 |
# | **Min Tick** | \$0.01 |
#
# ### Ticker Format
#
# `KXFED-27APR-T4.25` decodes as:
# - **KXFED**: Federal Funds Rate series
# - **27APR**: April 2027 FOMC meeting
# - **T4.25**: threshold - contract pays \$1 if rate is **above** 4.25%
#
# The `close` price is the implied probability (0–1) that the rate will
# exceed the threshold at that meeting.

# %% [markdown]
# ## 2. Load Kalshi Data
#
# We load pre-downloaded OHLCV data from the Kalshi API. The download script
# (`data/prediction_markets/download.py`) fetches all configured economic series
# and stores them in canonical OHLCV format.

# %%
df = load_kalshi()

print(f"Loaded {len(df):,} observations across {df['symbol'].n_unique()} contracts")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

df.group_by("symbol").len().rename({"len": "days"}).sort("symbol")

# %%
df.head(10)

# %% [markdown]
# ### Data Integrity Check
#
# These KXFED contracts are thinly traded: only a handful of bars carry any volume, and
# the rest are daily carry-forward snapshots. Two problems follow. First, some fields are
# corrupt: a price of exactly 0 (open, low, or close) while the high still holds the prior
# level is an ingestion artifact, since a live contract trades inside (0, 1) and never
# prints a true zero the same day it closes near 0.9. Second, on a zero-volume day there
# is no genuine intraday range at all, so a `low` far below the `close` is stale rather
# than traded. The one reliable field is `close` (the implied probability). We therefore
# carry the last valid close forward within each contract and rebuild every non-traded
# bar as a flat snapshot at that close, keeping raw OHLC only where the bar actually
# traded. Left unhandled, these artifacts inflate both the close-price range (misranking a
# near-certain contract as "most active") and the intraday range shown in Section 6.

# %%
traded = pl.col("volume") > 0

# Report the clearest ingestion artifacts: a price field of exactly 0 with a positive high.
zero_price_mask = (pl.col("high") > 0.0) & (
    (pl.col("open") == 0.0) | (pl.col("low") == 0.0) | (pl.col("close") == 0.0)
)
artifact_bars = (
    df.filter(zero_price_mask)
    .select("timestamp", "symbol", "open", "high", "low", "close", "volume")
    .sort("timestamp", "symbol")
)
print(f"Zero-price artifact bars (a price field of 0 with a positive high): {artifact_bars.height}")
print(f"Genuinely traded bars (volume > 0): {df.filter(traded).height} of {df.height}")

# Repair: null any close that collapsed to 0, carry the last valid close forward (and
# backward for a leading gap) within each contract, then rebuild every non-traded bar as
# a flat snapshot at that close so no spurious intraday range survives.
df = (
    df.sort("symbol", "timestamp")
    .with_columns(
        pl.when((pl.col("close") == 0.0) & (pl.col("high") > 0.0))
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    .with_columns(pl.col("close").forward_fill().over("symbol").alias("close"))
    .with_columns(pl.col("close").backward_fill().over("symbol").alias("close"))
    .with_columns(
        pl.when(traded).then(pl.col("open")).otherwise(pl.col("close")).alias("open"),
        pl.when(traded).then(pl.col("high")).otherwise(pl.col("close")).alias("high"),
        pl.when(traded).then(pl.col("low")).otherwise(pl.col("close")).alias("low"),
    )
)

artifact_bars

# %% [markdown]
# ## 3. Contract Universe
#
# All contracts are from the KXFED (Federal Reserve) series, covering different
# rate thresholds for upcoming FOMC meetings. Each threshold represents a
# different market expectation about the terminal rate.

# %%
contracts = (
    df.sort("timestamp")
    .group_by("symbol")
    .agg(
        pl.col("close").last().alias("latest_prob"),
        pl.col("close").first().alias("initial_prob"),
        pl.col("volume").sum().alias("total_volume"),
        pl.col("timestamp").min().alias("first_date"),
        pl.col("timestamp").max().alias("last_date"),
        pl.len().alias("observations"),
    )
    .sort("symbol")
)
contracts

# %% [markdown]
# ## 4. Probability Evolution
#
# The implied probability for each contract evolves over time as the market
# incorporates new information about Fed policy. Higher thresholds have lower
# probabilities (less likely the rate exceeds a high level).

# %%
# Volume is near zero across these contracts, so "most active" means the widest range
# in implied probability (on the cleaned data), which flags the genuine battleground
# thresholds rather than a data glitch.
price_range = (
    df.group_by("symbol")
    .agg((pl.col("close").max() - pl.col("close").min()).alias("range"))
    .sort("range", descending=True)
)

top_contracts = price_range.head(3)["symbol"].to_list()

fig = go.Figure()
palette = [COLORS["blue"], COLORS["amber"], COLORS["copper"]]

for sym, color in zip(top_contracts, palette, strict=False):
    data = df.filter(pl.col("symbol") == sym).sort("timestamp").to_pandas()
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["close"],
            mode="lines",
            name=sym,
            line=dict(color=color, width=2),
        )
    )

fig.update_layout(
    title="Battleground Fed-rate thresholds hover near even odds while far thresholds stay pinned",
    xaxis_title="Date",
    yaxis_title="Implied Probability",
    yaxis=dict(tickformat=".0%", range=[0, 1.05]),
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)

fig.show()

# %% [markdown]
# The contracts with rate thresholds near the current rate show the most
# price movement - these are the "battleground" levels where the market
# is genuinely uncertain. Contracts far from the current rate trade near
# 0 or 1 with little movement.

# %% [markdown]
# ## 5. Multi-Threshold View
#
# Looking at all thresholds for a single meeting gives a snapshot of the
# market's full probability distribution over rate outcomes.

# %%
# Group by meeting date prefix
meetings = {}
for sym in df["symbol"].unique().to_list():
    # KXFED-27APR-T4.25 → 27APR
    parts = sym.split("-")
    if len(parts) >= 3:
        meeting = parts[1]
        meetings.setdefault(meeting, []).append(sym)

fig = make_subplots(
    rows=len(meetings),
    cols=1,
    shared_xaxes=True,
    subplot_titles=[f"Meeting: {m}" for m in sorted(meetings.keys())],
    vertical_spacing=0.08,
)

for i, (meeting, symbols) in enumerate(sorted(meetings.items()), 1):
    for sym in sorted(symbols):
        data = df.filter(pl.col("symbol") == sym).sort("timestamp").to_pandas()
        threshold = sym.split("-T")[-1] if "-T" in sym else sym
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=data["close"],
                mode="lines",
                name=f"T{threshold}",
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
        )
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], row=i, col=1)

fig.update_layout(
    height=250 * len(meetings),
    title_text="Each FOMC meeting's threshold ladder maps the market's full rate distribution",
)

fig.show()

# %% [markdown]
# ## 6. Day-over-Day Probability Dynamics
#
# These markets trade on only a handful of days in the sample; the rest are zero-volume
# carry-forward snapshots with no genuine intraday range. Intraday high/low is therefore
# not a meaningful signal here. What moves is the day-over-day implied probability (the
# `close` path), so we summarize each contract by how many days it actually traded and by
# the volatility of its daily probability changes. Contracts whose threshold sits near the
# expected rate show the largest daily moves.

# %%
df_enriched = df.sort("symbol", "timestamp").with_columns(
    pl.col("close").diff().over("symbol").alias("prob_change"),
)

activity = (
    df_enriched.group_by("symbol")
    .agg(
        (pl.col("volume") > 0).sum().alias("traded_days"),
        pl.col("volume").sum().round(1).alias("total_volume"),
        pl.col("prob_change").std().round(4).alias("daily_prob_vol"),
        pl.col("prob_change").abs().max().round(4).alias("max_daily_move"),
    )
    .sort("daily_prob_vol", descending=True, nulls_last=True)
)

activity

# %%
ad = activity.to_pandas()
fig = go.Figure(go.Bar(x=ad["symbol"], y=ad["daily_prob_vol"], marker_color=COLORS["blue"]))
fig.update_layout(
    title="Near-the-money Fed thresholds carry the most daily probability movement",
    xaxis_title="Contract",
    yaxis_title="Std. of daily probability change",
    height=400,
)
fig.update_xaxes(tickangle=-45)
fig.show()

# %% [markdown]
# ## 7. Event Indicators for ML
#
# Transform Kalshi probabilities into ML-ready features. Since `close`
# is already the implied probability, we derive momentum, volatility,
# and regime indicators directly.

# %%
LOOKBACK = 5
VOL_WINDOW = 10

kalshi_features = df.sort("symbol", "timestamp").with_columns(
    (pl.col("close") - pl.col("close").shift(LOOKBACK).over("symbol")).alias("prob_momentum"),
    pl.col("close").diff().rolling_std(VOL_WINDOW).over("symbol").alias("prob_volatility"),
    pl.when(pl.col("close").rolling_std(VOL_WINDOW).over("symbol") > 0)
    .then(
        (pl.col("close") - pl.col("close").rolling_mean(VOL_WINDOW).over("symbol"))
        / pl.col("close").rolling_std(VOL_WINDOW).over("symbol")
    )
    .otherwise(0.0)
    .alias("prob_zscore"),
    ((pl.col("close") > 0.8) | (pl.col("close") < 0.2)).cast(pl.Int8).alias("high_confidence"),
    (pl.col("high") - pl.col("low")).alias("uncertainty"),
)

print(f"Feature matrix: {kalshi_features.shape}")
kalshi_features.select(
    "timestamp", "symbol", "close", "prob_momentum", "prob_volatility", "high_confidence"
).head(10)

# %%
# Feature distributions for the most active contract
active_sym = top_contracts[0]
active_features = kalshi_features.filter(
    (pl.col("symbol") == active_sym) & pl.col("prob_momentum").is_not_null()
)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Probability Momentum", "Probability Z-Score"),
)

fig.add_trace(
    go.Histogram(
        x=active_features["prob_momentum"].to_list(),
        nbinsx=30,
        name="Momentum",
        marker_color=COLORS["blue"],
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Histogram(
        x=active_features["prob_zscore"].to_list(),
        nbinsx=30,
        name="Z-Score",
        marker_color=COLORS["slate"],
    ),
    row=1,
    col=2,
)

fig.update_layout(
    title=f"Probability momentum and z-score center near zero for {active_sym}",
    height=350,
    showlegend=False,
)

fig.show()

# %% [markdown]
# ## 8. Data Quality Assessment
#
# Two checks matter for prediction-market data: whether any bars were corrupt (caught
# and repaired at load), and how much genuine price variation each contract carries.

# %%
print(f"Zero-price artifact bars caught and repaired at load: {artifact_bars.height}")
artifact_bars

# %%
quality_df = (
    df.group_by("symbol")
    .agg(
        pl.len().alias("observations"),
        pl.col("volume").mean().round(3).alias("avg_volume"),
        (pl.col("close").max() - pl.col("close").min()).round(3).alias("price_range"),
    )
    .sort("price_range", descending=True)
)
quality_df

# %%
qd = quality_df.to_pandas()
fig = go.Figure(go.Bar(x=qd["symbol"], y=qd["price_range"], marker_color=COLORS["blue"]))
fig.update_layout(
    title="After repair, close-price range concentrates in near-the-money thresholds",
    xaxis_title="Contract",
    yaxis_title="Close price range",
    height=400,
)
fig.update_xaxes(tickangle=-45)
fig.show()

# %% [markdown]
# After repairing the corrupt bars, price variation reflects real market movement.
# Volume is near zero across the universe, so the range in implied probability, not
# turnover, is the useful activity signal: contracts whose thresholds sit near the
# expected rate move the most and carry the richest information for ML features.

# %% [markdown]
# ## 9. Save Enriched Data

# %%
output_dir = get_output_dir(4, "kalshi")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "kalshi_features.parquet"
kalshi_features.write_parquet(output_file)

print(f"Saved {len(kalshi_features)} observations to {output_file}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Implied probability = close price**: Kalshi contract prices directly encode
#    the market's probability estimate for the event, no transformation needed
#
# 2. **Threshold structure**: Multiple contracts per meeting create a full
#    probability distribution over rate outcomes - richer than a single forecast
#
# 3. **Feature engineering**: Momentum, volatility, and z-score of probability
#    paths provide regime-detection signals for rate-sensitive strategies
#
# 4. **Liquidity caveat**: Economic event contracts are still early-stage;
#    volume is thin compared to traditional derivatives markets
#
# 5. **Screen for corrupt bars first**: thin, carry-forward markets are prone to
#    ingestion artifacts (a positive high with a zero close). Detect and repair them
#    before ranking or feature engineering, or a single bad bar distorts both.
#
# **Next**: See [`13_polymarket_prediction_markets`](13_polymarket_prediction_markets.ipynb) for the higher-liquidity
# crypto-based alternative and cross-platform comparison.
