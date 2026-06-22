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
"""Kalshi Prediction Markets — build event probability indicators from regulated binary contracts."""

import warnings

warnings.filterwarnings("ignore")

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data.prediction_markets.loader import load_kalshi
from utils.paths import get_output_dir
from utils.style import COLORS

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI

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
# - **T4.25**: threshold — contract pays \$1 if rate is **above** 4.25%
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
# Select contracts with the most price variation for visualization
price_range = (
    df.group_by("symbol")
    .agg((pl.col("close").max() - pl.col("close").min()).alias("range"))
    .sort("range", descending=True)
)

# Take top 3 most active contracts
top_contracts = price_range.head(3)["symbol"].to_list()

fig = go.Figure()
palette = [COLORS["blue"], COLORS["amber"], COLORS["slate"]]

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
    title="Fed Rate Probability Evolution (Kalshi)",
    xaxis_title="Date",
    yaxis_title="Implied Probability",
    yaxis=dict(tickformat=".0%", range=[0, 1.05]),
    template="plotly_white",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)

fig.show()

# %% [markdown]
# The contracts with rate thresholds near the current rate show the most
# price movement — these are the "battleground" levels where the market
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
    title_text="Rate Threshold Probabilities by Meeting",
    template="plotly_white",
)

fig.show()

# %% [markdown]
# ## 6. Intraday Price Dynamics
#
# The OHLCV data captures intraday price ranges. The difference between
# high and low within a day reflects intraday uncertainty and information flow.

# %%
# Add intraday range
df_enriched = df.with_columns(
    (pl.col("high") - pl.col("low")).alias("intraday_range"),
)

# Average intraday range by contract
range_stats = (
    df_enriched.group_by("symbol")
    .agg(
        pl.col("intraday_range").mean().alias("avg_range"),
        pl.col("intraday_range").max().alias("max_range"),
        pl.col("volume").mean().alias("avg_volume"),
    )
    .sort("avg_range", descending=True)
)

range_stats

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
    title=f"ML Feature Distributions — {active_sym}",
    height=350,
    showlegend=False,
    template="plotly_white",
)

fig.show()

# %% [markdown]
# ## 8. Data Quality Assessment

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

# %% [markdown]
# Most KXFED contracts show limited volume — Fed rate markets on Kalshi are
# still maturing. The contracts with meaningful price variation (those near the
# current rate threshold) are most useful as ML features.

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
#    probability distribution over rate outcomes — richer than a single forecast
#
# 3. **Feature engineering**: Momentum, volatility, and z-score of probability
#    paths provide regime-detection signals for rate-sensitive strategies
#
# 4. **Liquidity caveat**: Economic event contracts are still early-stage;
#    volume is thin compared to traditional derivatives markets
#
# **Next**: See [`13_polymarket_prediction_markets`](13_polymarket_prediction_markets.ipynb) for the higher-liquidity
# crypto-based alternative and cross-platform comparison.
