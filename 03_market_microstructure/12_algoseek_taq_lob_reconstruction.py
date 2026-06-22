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
# # TAQ LOB Reconstruction: Measuring Trade Aggression
#
# **Chapter 3: Market Microstructure**
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Build a forward-filled NBBO timeline from AlgoSeek TAQ events, classify
# each AAPL trade on 2020-03-16 with the Lee-Ready algorithm, and use the
# resulting buy/sell stream to compute order-imbalance and trade-aggression
# metrics that characterize the crash session.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Interleave trade and quote events on a nanosecond timeline and use
#   forward-fill to attach the prevailing NBBO to each trade.
# - Apply the Lee-Ready quote-test + tick-test cascade and read out
#   buy/sell ratios across the trading day.
# - Generate cumulative-order-imbalance and effective-spread visualizations
#   that quantify "the cost of immediacy during panic".
#
# ## Book reference
#
# Section §3.2 (`Notebooks 15-16 analyze tick-level patterns during the
# March 2020 crash`); §3.3 references the wider stylized-facts pattern.
#
# ## Prerequisites
#
# - AlgoSeek TAQ parquets (AAPL, 2020-03-16) accessible via `load_nasdaq100_taq`.
#
# ## The Lee-Ready Algorithm
#
# Lee and Ready (1991) proposed a simple rule:
#
# 1. **Quote test**: If trade price > midpoint → buyer initiated; < midpoint → seller
# 2. **Tick test**: If at midpoint, use price change: uptick → buy, downtick → sell
#
# Validated to ~94-95% accuracy on modern markets in
# [`15_itch_lee_ready`](15_itch_lee_ready.ipynb) using DataBento's
# ground-truth aggressor labels.

# %%
"""TAQ LOB Reconstruction — measuring trade aggression with Lee-Ready classification."""

from datetime import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data import load_nasdaq100_taq

COLORS = {
    "blue": "#1E3A5F",
    "accent": "#4A90A4",
    "warm": "#8B4513",
    "buy": "#228B22",
    "sell": "#B22222",
    "neutral": "#5D5D5D",
}


# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI

# %% [markdown]
# ## 1. Load and Filter Data
#
# We filter to regular trading hours (9:30 AM - 4:00 PM ET) to avoid pre-market
# artifacts. During pre-market, thin liquidity creates artificially wide spreads
# that would distort our analysis.

# %%
SYMBOL = "AAPL"
DATE = "20200316"
DATE_ISO = f"{DATE[:4]}-{DATE[4:6]}-{DATE[6:]}"

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

taq_raw = load_nasdaq100_taq(symbols=[SYMBOL], start_date=DATE_ISO, end_date=DATE_ISO)

taq = taq_raw.filter(
    (pl.col("timestamp").dt.time() >= MARKET_OPEN) & (pl.col("timestamp").dt.time() <= MARKET_CLOSE)
)

print(f"=== {SYMBOL} on March 16, 2020 ===")
print(f"Raw events: {len(taq_raw):,}")
print(f"Regular hours: {len(taq):,}")

# %% [markdown]
# ## 2. Build the NBBO Timeline
#
# For each trade, we need the prevailing NBBO. The challenge: quotes and trades
# are interleaved in time. We use forward-fill to carry the last known bid/ask
# to each trade timestamp.

# %%
# Extract quote and trade events
bids = (
    taq.filter(pl.col("event_type") == "QUOTE BID")
    .select(["timestamp", pl.col("price").alias("bid"), pl.col("quantity").alias("bid_size")])
    .sort("timestamp")
)

asks = (
    taq.filter(pl.col("event_type") == "QUOTE ASK")
    .select(["timestamp", pl.col("price").alias("ask"), pl.col("quantity").alias("ask_size")])
    .sort("timestamp")
)

trades = (
    taq.filter(pl.col("event_type") == "TRADE")
    .select(
        ["timestamp", pl.col("price").alias("trade_price"), pl.col("quantity").alias("trade_size")]
    )
    .sort("timestamp")
)

print(f"Bid quotes:  {len(bids):,}")
print(f"Ask quotes:  {len(asks):,}")
print(f"Trades:      {len(trades):,}")

# %%
# Combine all events chronologically
bids_marked = bids.with_columns(pl.lit("bid").alias("event"))
asks_marked = asks.with_columns(pl.lit("ask").alias("event"))
trades_marked = trades.with_columns(pl.lit("trade").alias("event"))

all_events = pl.concat(
    [
        bids_marked.select(["timestamp", "event", "bid", "bid_size"]),
        asks_marked.select(["timestamp", "event", "ask", "ask_size"]),
        trades_marked.select(["timestamp", "event", "trade_price", "trade_size"]),
    ],
    how="diagonal",
).sort("timestamp")

# Forward-fill bid/ask to get NBBO at each point
nbbo_at_trades = (
    all_events.with_columns(
        pl.col("bid").forward_fill(),
        pl.col("bid_size").forward_fill(),
        pl.col("ask").forward_fill(),
        pl.col("ask_size").forward_fill(),
    )
    .filter(pl.col("event") == "trade")
    .drop_nulls(subset=["bid", "ask"])
    .with_columns(
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        ((pl.col("ask") - pl.col("bid")) / ((pl.col("ask") + pl.col("bid")) / 2) * 10000).alias(
            "spread_bps"
        ),
        ((pl.col("bid") + pl.col("ask")) / 2).alias("midpoint"),
    )
    .filter(pl.col("spread") > 0)  # Remove crossed/locked markets
)

print(f"\nTrades with valid NBBO: {len(nbbo_at_trades):,}")

# %% [markdown]
# ## 3. Spread at Trade Time
#
# Before classifying trades, let's understand the spread environment they
# executed in. The spread is the "toll" for crossing from passive to aggressive.

# %%
# Spread statistics at trade times
spread_stats = nbbo_at_trades.select(
    pl.col("spread_bps").mean().alias("mean"),
    pl.col("spread_bps").median().alias("median"),
    pl.col("spread_bps").quantile(0.95).alias("p95"),
    pl.col("spread_bps").max().alias("max"),
)

print("=== Spread at Trade Time ===")
print(f"  Mean:   {spread_stats['mean'][0]:.1f} bps")
print(f"  Median: {spread_stats['median'][0]:.1f} bps")
print(f"  95th:   {spread_stats['p95'][0]:.1f} bps")
print(f"  Max:    {spread_stats['max'][0]:.1f} bps")
print("\n  (Normal day: ~1-2 bps median)")

# %%
# Spread distribution
fig = px.histogram(
    nbbo_at_trades.filter(pl.col("spread_bps") < 50).to_pandas(),  # Cap for visibility
    x="spread_bps",
    nbins=100,
    color_discrete_sequence=[COLORS["blue"]],
)

fig.add_vline(
    x=spread_stats["median"][0],
    line_dash="dash",
    line_color=COLORS["warm"],
    annotation_text=f"Median: {spread_stats['median'][0]:.1f} bps",
)

fig.update_layout(
    title=f"Spread Distribution at Trade Time - {SYMBOL} (March 16, 2020)",
    xaxis_title="Spread (bps)",
    yaxis_title="Count",
    height=400,
)

fig.show()

# %% [markdown]
# **Interpretation**: Most trades execute in a 2-5 bps spread environment, but
# the long tail shows moments of stress where spreads widened to 20+ bps. These
# are expensive trades - the aggressor pays a significant premium for immediacy.

# %% [markdown]
# ## 4. Lee-Ready Classification
#
# Now we apply Lee-Ready to classify each trade. The quote test handles ~90%
# of trades; the tick test fills in when prices land exactly at midpoint.

# %%
# Apply Lee-Ready
trades_classified = (
    nbbo_at_trades.with_columns(
        # Quote test: compare to midpoint
        pl.when(pl.col("trade_price") > pl.col("midpoint"))
        .then(pl.lit(1))
        .when(pl.col("trade_price") < pl.col("midpoint"))
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("quote_rule"),
        # Tick test: direction of price change
        pl.col("trade_price").diff().sign().fill_null(0).alias("tick_rule"),
    )
    .with_columns(
        # Final classification
        pl.when(pl.col("quote_rule") != 0)
        .then(pl.col("quote_rule"))
        .otherwise(pl.col("tick_rule"))
        .alias("trade_sign")
    )
    .with_columns(
        pl.when(pl.col("trade_sign") == 1)
        .then(pl.lit("BUY"))
        .when(pl.col("trade_sign") == -1)
        .then(pl.lit("SELL"))
        .otherwise(pl.lit("UNKNOWN"))
        .alias("direction")
    )
)

# %%
# Classification breakdown
classification = (
    trades_classified.group_by("direction")
    .agg(
        pl.len().alias("count"),
        pl.col("trade_size").sum().alias("volume"),
    )
    .with_columns(
        (pl.col("count") / pl.sum("count") * 100).alias("count_pct"),
        (pl.col("volume") / pl.sum("volume") * 100).alias("volume_pct"),
    )
    .sort("volume", descending=True)
)

print("=== Lee-Ready Classification ===")
for row in classification.iter_rows(named=True):
    print(
        f"  {row['direction']:7} {row['count']:>10,} trades ({row['count_pct']:5.1f}%)  "
        f"{row['volume']:>15,} shares ({row['volume_pct']:5.1f}%)"
    )

# %%
# Visualize classification
colors_map = {"BUY": COLORS["buy"], "SELL": COLORS["sell"], "UNKNOWN": COLORS["neutral"]}

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "pie"}, {"type": "pie"}]],
    subplot_titles=("By Trade Count", "By Volume"),
)

fig.add_trace(
    go.Pie(
        labels=classification["direction"].to_list(),
        values=classification["count"].to_list(),
        marker=dict(colors=[colors_map[d] for d in classification["direction"].to_list()]),
        textinfo="label+percent",
        hole=0.4,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Pie(
        labels=classification["direction"].to_list(),
        values=classification["volume"].to_list(),
        marker=dict(colors=[colors_map[d] for d in classification["direction"].to_list()]),
        textinfo="label+percent",
        hole=0.4,
    ),
    row=1,
    col=2,
)

fig.update_layout(
    title=f"Trade Direction (Lee-Ready) - {SYMBOL} (March 16, 2020)",
    height=400,
    showlegend=False,
)

fig.show()

# %% [markdown]
# **What we see**: On this crash day, seller-initiated trades slightly dominate
# both by count and volume. This confirms the intuition that March 16 was a
# day of panic selling - the aggressive side was overwhelmingly sellers
# demanding immediacy.

# %% [markdown]
# ## 5. Order Imbalance Over Time
#
# Order imbalance = (Buy Volume - Sell Volume) / Total Volume
#
# This signal captures the net direction of aggressive trading. Strong positive
# imbalance indicates buying pressure; negative indicates selling.

# %%
# Compute minute-level order imbalance
minute_stats = (
    trades_classified.with_columns(
        (pl.col("trade_size") * pl.col("trade_sign")).alias("signed_volume"),
    )
    .group_by_dynamic("timestamp", every="1m")
    .agg(
        pl.col("trade_price").first().alias("open"),
        pl.col("trade_price").last().alias("close"),
        pl.col("trade_size").sum().alias("volume"),
        pl.col("signed_volume").sum().alias("signed_volume"),
        pl.col("spread_bps").mean().alias("avg_spread"),
        pl.len().alias("trades"),
    )
    .with_columns(
        (pl.col("close") / pl.col("open") - 1).alias("return"),
        (pl.col("signed_volume") / pl.col("volume")).alias("imbalance"),
    )
    .drop_nulls()
)

print(f"Minute bars: {len(minute_stats)}")

# %%
# Correlation between imbalance and returns
corr = minute_stats.select(pl.corr("imbalance", "return"))
print(f"\nImbalance ↔ Return correlation: {corr[0, 0]:.3f}")

# %% [markdown]
# The Pearson correlation between minute-level imbalance and minute returns is
# ~0.08 on this single AAPL day. The relationship is contemporaneous;
# converting it into a tradeable signal requires predicting future imbalance
# rather than reading the realized contemporaneous value.

# %%
# Build three-panel figure: Price, Imbalance, Spread (single cell so the
# figure is emitted once, fully populated — split-cell variants trigger
# papermill's intermediate auto-display and leave the third panel empty).
fig = make_subplots(
    rows=3,
    cols=1,
    row_heights=[0.4, 0.3, 0.3],
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=("Price", "Order Imbalance", "Spread"),
)

fig.add_trace(
    go.Scatter(
        x=minute_stats["timestamp"].to_list(),
        y=minute_stats["close"].to_list(),
        name="Price",
        line=dict(color=COLORS["blue"], width=1),
    ),
    row=1,
    col=1,
)

imbalance_colors = [
    COLORS["buy"] if x > 0 else COLORS["sell"] for x in minute_stats["imbalance"].to_list()
]
fig.add_trace(
    go.Bar(
        x=minute_stats["timestamp"].to_list(),
        y=minute_stats["imbalance"].to_list(),
        name="Imbalance",
        marker_color=imbalance_colors,
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=minute_stats["timestamp"].to_list(),
        y=minute_stats["avg_spread"].to_list(),
        name="Spread",
        line=dict(color=COLORS["warm"], width=1),
        fill="tozeroy",
        fillcolor="rgba(139, 69, 19, 0.2)",
    ),
    row=3,
    col=1,
)

fig.update_layout(
    title=f"Price, Imbalance, and Spread - {SYMBOL} (March 16, 2020)",
    height=600,
    showlegend=False,
)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Imbalance", row=2, col=1)
fig.update_yaxes(title_text="Spread (bps)", row=3, col=1)
fig.update_xaxes(title_text="Time (ET)", row=3, col=1)

fig.show()

# %% [markdown]
# **Reading the panel**:
#
# - **Top (Price)**: The crash unfolds - gap down at open, circuit breaker halt,
#   continued selling, then stabilization
# - **Middle (Imbalance)**: Red bars dominate early (sell pressure), more mixed later
# - **Bottom (Spread)**: Spikes during price dislocations, narrows when calm
#
# The three series are connected: when imbalance is strongly negative (selling),
# price drops, and spreads widen as market makers retreat.

# %%
# Scatter: imbalance vs return
fig = px.scatter(
    minute_stats.to_pandas(),
    x="imbalance",
    y="return",
    color="avg_spread",
    color_continuous_scale="RdYlBu_r",
    opacity=0.6,
)

# Regression line
x = minute_stats["imbalance"].to_numpy()
y = minute_stats["return"].to_numpy()
mask = ~(np.isnan(x) | np.isnan(y))
if mask.sum() > 2:
    z = np.polyfit(x[mask], y[mask], 1)
    p = np.poly1d(z)
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=p(x_line),
            mode="lines",
            name="Trend",
            line=dict(color=COLORS["warm"], width=2, dash="dash"),
        )
    )

fig.update_layout(
    title=f"Imbalance-Return Relationship (r={corr[0, 0]:.2f})",
    xaxis_title="Order Imbalance",
    yaxis_title="Minute Return",
    yaxis=dict(tickformat=".1%"),
    coloraxis_colorbar_title="Spread (bps)",
    height=450,
)

fig.show()

# %% [markdown]
# **The scatter shows a weak positive tilt**: consistent with the ~0.08
# correlation, the cloud is diffuse rather than tight, but buy-dominated minutes
# (imbalance > 0) lean toward positive returns and sell-dominated minutes toward
# negative ones. The color shows that high-spread moments (yellow/red) are often
# extreme imbalance/return events.

# %% [markdown]
# ## 6. Intraday Imbalance Pattern
#
# Does imbalance vary systematically through the day? Let's aggregate by hour.

# %%
hourly_imbalance = (
    minute_stats.with_columns(pl.col("timestamp").dt.hour().alias("hour"))
    .group_by("hour")
    .agg(
        pl.col("imbalance").mean().alias("avg_imbalance"),
        pl.col("volume").sum().alias("total_volume"),
        pl.col("avg_spread").mean().alias("avg_spread"),
    )
    .sort("hour")
)

print("=== Hourly Pattern ===")
print(hourly_imbalance)

# %%
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Bar(
        x=hourly_imbalance["hour"].to_list(),
        y=hourly_imbalance["avg_imbalance"].to_list(),
        name="Avg Imbalance",
        marker_color=[
            COLORS["buy"] if x > 0 else COLORS["sell"]
            for x in hourly_imbalance["avg_imbalance"].to_list()
        ],
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=hourly_imbalance["hour"].to_list(),
        y=hourly_imbalance["avg_spread"].to_list(),
        name="Avg Spread",
        mode="lines+markers",
        line=dict(color=COLORS["warm"], width=2),
        marker=dict(size=8),
    ),
    secondary_y=True,
)

fig.update_layout(
    title=f"Hourly Imbalance and Spread - {SYMBOL} (March 16, 2020)",
    xaxis_title="Hour (ET)",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)

fig.update_yaxes(title_text="Avg Order Imbalance", secondary_y=False)
fig.update_yaxes(title_text="Avg Spread (bps)", secondary_y=True)

fig.show()

# %% [markdown]
# **Selling pressure deepens into the afternoon**: the open (Hour 9) already
# carries negative imbalance (-0.12) and wide spreads (~132 bps), but the
# imbalance becomes *most* negative midday-to-afternoon, reaching -0.16 by
# Hour 14, while the widest spread sits at Hour 12 (~137 bps). Spreads are
# tightest mid-morning and late afternoon (~75 bps in Hours 10 and 14), so the
# pattern is a worsening midday imbalance rather than a clean open-to-close
# moderation.

# %% [markdown]
# ## Key Takeaways
#
# **1. NBBO reconstruction is straightforward**: Forward-fill bid/ask to each
# trade timestamp. This is the foundation for all trade classification.
#
# **2. Lee-Ready classification**: The quote test handles trades away from the
# midpoint; the tick test fills the gap at the midpoint. On March 16, sellers
# accounted for ~58% of classified volume.
#
# **3. Order imbalance co-moves with returns**: contemporaneous correlation of
# ~0.08 at minute frequency for this single-day AAPL sample.
#
# **4. Stress amplifies patterns**: The open was dominated by sell imbalance
# and wide spreads; both moderated through the day.
#
# **5. The three metrics are connected**: Price, imbalance, and spread move
# together - understanding one requires understanding all three.
#
# ## Next Steps
#
# - **Minute Bars**: [`13_algoseek_minute_bars_eda`](13_algoseek_minute_bars_eda.ipynb) - Pre-aggregated data
#   for longer-horizon analysis
# - **Feature Engineering (Ch8)**: Build microstructure features from signed
#   trades for ML models
# - **VPIN (Ch8)**: Volume-synchronized probability of informed trading
