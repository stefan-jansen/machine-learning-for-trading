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
# `15_itch_lee_ready` measures how often that rule agrees with DataBento's own aggressor
# labels, which are the venue's record of which side crossed. This notebook applies the
# rule; that one says how well it does.

# %%
"""TAQ LOB Reconstruction — measuring trade aggression with Lee-Ready classification."""

from datetime import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data import load_nasdaq100_taq
from utils.style import COLORS, show_plotly_with_alt

# Four shades on top of the repository palette: two for a second and third series on one
# axis, and green and red for buyer- and seller-initiated trades, which is the convention
# the rest of the chapter uses.
COLORS = {
    **COLORS,
    "accent": "#4A90A4",
    "warm": "#8B4513",
    "buy": "#228B22",
    "sell": "#B22222",
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
    title=f"{SYMBOL}: spread prevailing at each trade, March 16, 2020",
    xaxis_title="Spread (bps)",
    yaxis_title="Count",
    height=400,
)

show_plotly_with_alt(
    fig,
    "A histogram of the bid-ask spread in basis points at the moment each trade printed, with the horizontal axis capped so the bulk of the distribution is legible and a vertical annotation marking the median.",
)

# %% [markdown]
# Read the mean against the median in the statistics above. Most trades on this day still
# executed against a tight quote, but the distribution has a tail heavy enough that the
# mean sits an order of magnitude above the middle of it, and around the trading halts
# the quote dislocates far enough that the spread runs into the thousands of basis
# points - a substantial fraction of the price itself.
#
# That gap is the reason a mean spread is a poor summary of what trading costs. The
# histogram below caps its horizontal axis so the bulk of the distribution is legible,
# and the tail continues past the right edge; the printed percentiles are where to read
# the tail, not the chart.

# %% [markdown]
# ## 4. Lee-Ready Classification
#
# The quote test settles every trade that printed away from the midpoint, which is most
# of them; the tick test exists for the rest, where the price landed exactly on the
# midpoint and the quote says nothing about who crossed. The counts below say how the
# work divided between them on this day.

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
    title=f"{SYMBOL}: trades classified by the Lee-Ready rule, March 16, 2020",
    height=400,
    showlegend=False,
)

show_plotly_with_alt(
    fig,
    "Two doughnut charts side by side, both split into buyer-initiated in green, seller-initiated in red and unclassified. The first divides the day's trade count between the three and the second divides its volume.",
)

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
# That correlation is contemporaneous: it pairs a minute's imbalance with that same
# minute's return. A positive value says buying pressure and rising prices happen
# together, which is close to a definition - the trades that pushed the price up are the
# ones counted as buys.
#
# It is not a signal, because acting on it would require knowing the minute's imbalance
# before the minute ends. The tradeable question is whether an imbalance already
# observed says anything about the *next* interval's return, which is a different
# measurement on the same two series: lag the imbalance behind the return rather than
# pairing them within a bar. `09_databento_mbo_analysis` makes that one.

# %% [markdown]
# The three series go on one figure with a shared time axis because the question is how
# they move relative to each other: whether the minutes of heaviest one-sided flow are
# the minutes the price moved, and whether either coincides with the widest spreads.

# %%
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
    title=f"{SYMBOL}: price, order imbalance and spread through the session",
    height=600,
    showlegend=False,
)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Imbalance", row=2, col=1)
fig.update_yaxes(title_text="Spread (bps)", row=3, col=1)
fig.update_xaxes(title_text="Time (ET)", row=3, col=1)

show_plotly_with_alt(
    fig,
    "Three stacked panels sharing a clock-time axis over one session: the traded price in dollars, the order imbalance per minute about a zero line, and the prevailing spread in basis points.",
)

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
    title="Minute return against order imbalance, coloured by the prevailing spread",
    xaxis_title="Order Imbalance",
    yaxis_title="Minute Return",
    yaxis=dict(tickformat=".1%"),
    coloraxis_colorbar_title="Spread (bps)",
    height=450,
)

show_plotly_with_alt(
    fig,
    "A scatter of each minute's return against its order imbalance, one point per minute, with the points coloured by the spread prevailing in that minute so the widest-spread minutes can be located within the cloud.",
)

# %% [markdown]
# Two things to read off that scatter. The tilt is the contemporaneous relationship just
# printed, and how diffuse the cloud is around it says how much of a minute's return the
# imbalance accounts for - a tilt in a wide cloud is a weak association, not a strong
# one seen through noise.
#
# The colour is the third variable: where the widest-spread minutes sit in that cloud.
# If they cluster at the extremes of both axes, then the minutes with the largest moves
# and the most one-sided flow are also the minutes when trading them cost the most, which
# is the practical objection to reading this relationship as an opportunity.

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
    title=f"{SYMBOL}: average imbalance and spread by hour of the session",
    xaxis_title="Hour (ET)",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)

fig.update_yaxes(title_text="Avg Order Imbalance", secondary_y=False)
fig.update_yaxes(title_text="Avg Spread (bps)", secondary_y=True)

show_plotly_with_alt(
    fig,
    "A chart with one point per hour of the session, plotting the average order imbalance on the left vertical axis and the average spread in basis points on the right, so the hour at which each reaches its extreme can be compared.",
)

# %% [markdown]
# The two series on that chart do not peak in the same hour, and that is the point of
# plotting them together. A story in which stress arrives at the open and eases through
# the day would show both worst in the first hour and improving after it. Read where each
# series actually reaches its extreme, and whether the hour of the widest spread is the
# hour of the most one-sided flow.

# %% [markdown]
# ## Key Takeaways
#
# **1. A trade needs the quote that prevailed when it printed.** Forward-filling the
# consolidated bid and ask onto each trade timestamp is what makes every classification
# below possible, and the join has to be as-of: each trade takes the most recent quote
# at or before its own timestamp, which is the only quote its sender could have seen.
#
# **2. Lee-Ready is two rules, and the second one is the interesting one.** The quote
# test settles anything that printed away from the midpoint. The tick test handles the
# rest by looking at the direction of the last price change, which is a weaker piece of
# evidence - and how much of the day falls to it is worth knowing before trusting the
# classified totals.
#
# **3. A contemporaneous correlation is not a signal.** Pairing a minute's imbalance with
# that minute's return measures co-movement, and acting on it would require knowing the
# minute before it ended. Lagging the imbalance behind the return asks the tradeable
# question instead, and it is a different measurement with a different answer.
#
# **4. Plot stress measures together and check whether they peak together.** Spread and
# imbalance are both read as stress; if their extremes fall in different hours, they are
# measuring different things and a single 'stress' narrative papers over that.
#
# **5. A mean spread on a dislocated day says very little.** With a tail this heavy the
# mean sits far above the median, and neither one describes what a typical trade paid.
#
# ### Known limitations
#
# - One symbol on one exceptional session, chosen because it is not typical.
# - Lee-Ready is inferred, not observed. `15_itch_lee_ready` compares it against a
#   venue's own aggressor labels; nothing here is validated against ground truth.
# - Every relationship reported is contemporaneous. This notebook makes no forecast and
#   its correlations should not be read as predictive.
#
# ## Next Steps
#
# - **Minute Bars**: [`13_algoseek_minute_bars_eda`](13_algoseek_minute_bars_eda.ipynb) - Pre-aggregated data
#   for longer-horizon analysis
# - **Feature Engineering (Ch8)**: Build microstructure features from signed
#   trades for ML models
# - **VPIN (Ch8)**: Volume-synchronized probability of informed trading
