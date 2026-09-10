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
# # Intraday Patterns: Volume and Volatility Dynamics
#
# **Chapter 3: Market Microstructure**
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Quantify the intraday volume U-shape across high-, medium-, and low-liquidity
# NASDAQ tickers using ITCH-derived trade data, and produce the comparative
# 30-minute-resolution figures that §3.1 and §3.3 reference.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Resample tick-level trades into 30-minute volume bars and recognize the
#   open/close hump versus midday lull.
# - Compare intraday patterns across liquidity tiers (TSLA / mid-tier / illiquid)
#   and quantify the open-vs-midday volume ratio.
# - Connect the U-shape to feature engineering choices in Chapter 8 (time-of-day
#   features, volume-normalized signals).
#
# ## Book reference
#
# Section §3.1 (intraday-flow narrative) and Section §3.3 (stylized-facts
# subsection on intraday U-shape).
#
# ## Prerequisites
#
# - The canonical enriched-trade parquet at
#   `output/ch03/nasdaq_itch/trading_activity/trades.parquet` and the matching
#   `trade_summary.parquet` (used for liquidity-tier symbol selection); both
#   are produced by `05_itch_trading_activity`.
#
# ---

# %% [markdown]
# ## 1. Setup

# %%
"""Intraday Patterns — volume and volatility dynamics from NASDAQ ITCH data."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from data import load_nasdaq_itch
from utils.paths import display_path, get_output_dir, require_chapter_inputs
from utils.style import COLORS, show_with_alt

# %% [markdown]
# ### Declared parameters
#
# `MIN_TRADES_LOW` is the floor a ticker must clear to stand for the low-liquidity tier.
# A ticker that printed a handful of trades all day produces a panel with two points on
# it, which shows nothing about intraday shape; the floor picks the least active name
# that still has one.
#
# `PATTERN_TICKERS` is how many of the most actively traded tickers the aggregate
# U-shape averages over. Averaging over a handful would let one name's day decide the
# shape; averaging over everything would let thousands of barely traded ones bury it.
#
# `PATTERN_FREQ` sets the bar width for that aggregate. Thirty minutes divides the
# session into thirteen bars, which is fine enough to separate the open and the close
# from the middle of the day and coarse enough that each bar holds real volume.

# %% tags=["parameters"]
MIN_TRADES_LOW = 500
PATTERN_TICKERS = 20
PATTERN_FREQ = "30m"

# %%
NASDAQ_ITCH_OUTPUT = get_output_dir(3, "nasdaq_itch")
MESSAGE_DIR = load_nasdaq_itch(get_base_path=True)
TRADING_ACTIVITY_DIR = NASDAQ_ITCH_OUTPUT / "trading_activity"

print(f"Input directory (messages): {display_path(MESSAGE_DIR)}")
print(f"Input directory (trade summary): {display_path(TRADING_ACTIVITY_DIR)}")

# %% [markdown]
# ## 2. Load Trade Data
#
# Load outputs from `05_itch_trading_activity`:
# - `trade_summary.parquet`: Aggregated stats by ticker (for symbol selection)
# - `trades.parquet`: Canonical tick-level trades (for analysis)
#
# `trades.parquet` is the table `05_itch_trading_activity` builds by attributing each
# `E` and `C` execution back to a ticker, so reading it here means the two notebooks
# cannot disagree about what a trade is.

# %%
# Load trade summary and canonical trades from notebook 05
TRADE_SUMMARY_PATH = TRADING_ACTIVITY_DIR / "trade_summary.parquet"
TRADES_PATH = TRADING_ACTIVITY_DIR / "trades.parquet"

# Substituting well-known tickers for the ones this dataset actually traded would let
# the notebook finish with nothing to plot, so stop instead and say what is missing.
require_chapter_inputs(
    {
        MESSAGE_DIR: "01_itch_parser",
        TRADE_SUMMARY_PATH: "05_itch_trading_activity",
        TRADES_PATH: "05_itch_trading_activity",
    }
)

# Load trade summary for ticker selection
trade_summary = pl.read_parquet(TRADE_SUMMARY_PATH)
# Sort explicitly by value to ensure correct selection
trade_summary = trade_summary.sort("total_value", descending=True)
trade_col = next(
    (c for c in ("trade_count", "n_trades", "total_trades") if c in trade_summary.columns),
    None,
)
if trade_col is None:
    # No trade-count column: fall back to the top half by traded value.
    active_summary = trade_summary.head(len(trade_summary) // 2)
else:
    active_summary = trade_summary.filter(pl.col(trade_col) >= MIN_TRADES_LOW)
num_syms = len(active_summary)
high_sym = active_summary["ticker"][0]  # highest value, still traded
mid_sym = active_summary["ticker"][num_syms // 2]  # middle of active band
low_sym = active_summary["ticker"][-1]  # lowest value above min-activity floor
print(f"Loaded trade summary: {len(trade_summary)} tickers; {num_syms} above min-activity floor")

# Load canonical trades (single source of truth for trade extraction)
all_trades = pl.read_parquet(TRADES_PATH)
print(f"Loaded canonical trades: {len(all_trades):,} trades")
if "msg_type" in all_trades.columns:
    msg_breakdown = all_trades.group_by("msg_type").len().sort("msg_type")
    print("  Message type breakdown:")
    for row in msg_breakdown.iter_rows():
        print(f"    {row[0]}: {row[1]:>12,}")

print("\nSelected tickers for analysis:")
print(f"  High liquidity:   {high_sym}")
print(f"  Medium liquidity: {mid_sym}")
print(f"  Low liquidity:    {low_sym}")


# %% [markdown]
# ## 3. Intraday Volume and Price by Liquidity Tier
#
# We resample tick-level trades into intraday bars for one high-, one medium-,
# and one low-liquidity ticker, and read the volume and price panels side by
# side. The bar frequency widens for the illiquid name so its sparse prints
# still form a legible shape.


# %%
def intraday_resample(trades_df: pl.DataFrame, ticker: str, freq: str = "5m") -> pl.DataFrame:
    """
    Filter trades for a single ticker and resample to intraday bars.

    Args:
        trades_df: Canonical trades DataFrame from notebook 05 (trades.parquet)
        ticker: Stock symbol to filter
        freq: Bar frequency (e.g., "5m", "15m", "30m")

    Returns:
        DataFrame with columns: timestamp, shares, value, price, vwap, trade_count
    """
    if trades_df is None or len(trades_df) == 0:
        return pl.DataFrame()

    # Filter to ticker
    df = trades_df.filter(pl.col("ticker") == ticker)
    if len(df) == 0:
        return pl.DataFrame()

    # Ensure we have required columns (compute value if missing)
    required = ["timestamp", "shares", "price"]
    if not all(c in df.columns for c in required):
        return pl.DataFrame()

    if "value" not in df.columns:
        df = df.with_columns((pl.col("shares") * pl.col("price")).alias("value"))

    df = df.select(["timestamp", "shares", "price", "value"]).sort("timestamp")

    # Resample to bars using group_by_dynamic
    bars = df.group_by_dynamic("timestamp", every=freq).agg(
        [
            pl.col("shares").sum().alias("shares"),
            pl.col("value").sum().alias("value"),
            pl.col("price").last().alias("price"),
            pl.len().alias("trade_count"),
        ]
    )

    # Calculate VWAP
    bars = bars.with_columns(
        pl.when(pl.col("shares") > 0)
        .then(pl.col("value") / pl.col("shares"))
        .otherwise(None)
        .alias("vwap")
    )

    bars = bars.drop_nulls(subset=["price"])
    return bars


# %% [markdown]
# ### Plot Intraday Bars
# Resample trades for a ticker and visualize volume and price patterns side by side.


# %%
def plot_intraday_bars(trades_df: pl.DataFrame, ticker: str, freq: str = "5m") -> None:
    """Resample trades for ticker and plot volume and price patterns."""
    bars = intraday_resample(trades_df, ticker, freq)
    if len(bars) == 0:
        print(f"No trade data found for {ticker}.")
        return

    # Convert to pandas for matplotlib
    bars_pd = bars.to_pandas()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{ticker}: intraday trading in {freq} bars", fontsize=14)

    ax1 = axes[0]
    ax1.bar(
        bars_pd["timestamp"],
        bars_pd["shares"],
        alpha=0.7,
        color=COLORS["blue"],
        label="Shares traded",
    )
    ax1.set_ylabel("Shares traded")
    ax1.legend(loc="upper left")

    ax1_2 = ax1.twinx()
    ax1_2.plot(bars_pd["timestamp"], bars_pd["trade_count"], color=COLORS["amber"], label="Trades")
    ax1_2.set_ylabel("Number of trades")
    ax1_2.legend(loc="upper right")

    ax2 = axes[1]
    ax2.plot(
        bars_pd["timestamp"], bars_pd["price"], label="Last trade price", color=COLORS["slate"]
    )
    ax2.plot(
        bars_pd["timestamp"],
        bars_pd["vwap"],
        label="Volume-weighted average price",
        color=COLORS["copper"],
        linestyle="--",
    )
    ax2.set_ylabel("Price ($)")
    ax2.set_xlabel("Time (US/Eastern)")
    ax2.legend()

    show_with_alt(
        fig,
        f"Two stacked panels for {ticker} sharing a clock-time axis over one session. The upper panel is a bar chart of shares traded in each {freq} bar, with a line on a second vertical axis giving the number of trades in the same bar. The lower panel plots two price lines, the last trade price and the volume-weighted average price of the bar, the second dashed.",
    )


# %%
print(f"High-Volume Ticker: {high_sym}")
plot_intraday_bars(all_trades, high_sym, freq="5m")

# %%
print(f"Medium-Volume Ticker: {mid_sym}")
plot_intraday_bars(all_trades, mid_sym, freq="5m")

# %%
print(f"Low-Volume Ticker: {low_sym}")
plot_intraday_bars(all_trades, low_sym, freq="15m")  # Longer bars for sparse data


# %% [markdown]
# ## 4. The Intraday U-Shape
#
# Averaging volume across the top-20 most active tickers reveals the
# characteristic U-shape: trading concentrates at the open and the close and
# thins out at midday.
# - **High at open**: price discovery and overnight-information incorporation.
# - **Low at midday**: the "lunch lull" of reduced institutional activity.
# - **High at close**: portfolio rebalancing, index arbitrage, and MOC orders.
#
# This regularity drives feature construction in Chapter 8: time-of-day
# features encode it directly.


# %%
def compute_intraday_pattern(
    trades_df: pl.DataFrame, tickers: list[str], freq: str = "30m"
) -> pl.DataFrame:
    """
    Compute average intraday patterns across multiple tickers.

    Args:
        trades_df: Canonical trades DataFrame from notebook 05
        tickers: List of stock symbols to analyze
        freq: Bar frequency (e.g., "30m")

    Returns:
        DataFrame with time_slot, vol_pct, trade_count, ticker
    """
    all_patterns = []

    for ticker in tickers:
        bars = intraday_resample(trades_df, ticker, freq)
        if len(bars) == 0:
            continue

        # Extract hour and compute relative metrics
        bars = bars.with_columns(
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.minute().alias("minute"),
        )

        # Compute time-of-day slot (e.g., 9:30 -> 9.5)
        bars = bars.with_columns((pl.col("hour") + pl.col("minute") / 60).alias("time_slot"))

        # Each ticker's bars are expressed as shares of its own daily total, so that a
        # mega-cap and a mid-cap contribute equally to the average shape rather than in
        # proportion to their size.
        total_vol = bars["shares"].sum()
        if total_vol > 0:
            bars = bars.with_columns(
                (pl.col("shares") / total_vol).alias("vol_pct"),
                pl.lit(ticker).alias("ticker"),
            )
            all_patterns.append(bars.select(["time_slot", "vol_pct", "trade_count", "ticker"]))

    if not all_patterns:
        return pl.DataFrame()

    return pl.concat(all_patterns)


# %%
top_tickers = trade_summary.head(PATTERN_TICKERS)["ticker"].to_list()
pattern_df = compute_intraday_pattern(all_trades, top_tickers, freq=PATTERN_FREQ)
assert not pattern_df.is_empty(), (
    f"None of the {len(top_tickers)} most active tickers produced intraday bars; the "
    f"trade table is empty or carries no usable timestamps."
)

hourly_pattern = (
    pattern_df.group_by("time_slot")
    .agg(
        pl.col("vol_pct").mean().alias("avg_vol_pct"),
        pl.col("vol_pct").std().alias("std_vol_pct"),
        pl.col("trade_count").mean().alias("avg_trades"),
    )
    .sort("time_slot")
    # Regular trading hours only: pre- and post-market bars are a different market with
    # its own participants, and mixing them in flattens the shape being measured.
    .filter((pl.col("time_slot") >= 9.5) & (pl.col("time_slot") <= 16))
)

# %%
times = hourly_pattern["time_slot"].to_numpy()
vol_pct = hourly_pattern["avg_vol_pct"].to_numpy() * 100
vol_std = hourly_pattern["std_vol_pct"].to_numpy() * 100
trades = hourly_pattern["avg_trades"].to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].fill_between(
    times,
    vol_pct - vol_std,
    vol_pct + vol_std,
    alpha=0.3,
    color=COLORS["blue"],
    label="±1 standard deviation across tickers",
)
axes[0].plot(times, vol_pct, color=COLORS["blue"], linewidth=2, marker="o", label="Mean")
axes[0].set_xlabel("Time of day (US/Eastern, hours)")
axes[0].set_ylabel("Share of the ticker's daily volume (%)")
axes[0].set_title("Volume by time of day, averaged over the most active tickers")
axes[0].axhline(
    100 / len(times),
    color=COLORS["negative"],
    linestyle="--",
    label="Even across the session",
)
axes[0].legend()
axes[0].set_xlim(9.5, 16)

axes[1].bar(times, trades, width=0.4, alpha=0.7, color=COLORS["blue"])
axes[1].set_xlabel("Time of day (US/Eastern, hours)")
axes[1].set_ylabel(f"Mean trades per {PATTERN_FREQ} bar")
axes[1].set_title("Number of trades by time of day, the same tickers")
axes[1].set_xlim(9.5, 16)

show_with_alt(
    fig,
    "Two panels side by side, both against time of day from the 09:30 open to the 16:00 close. The left plots the mean share of a ticker's daily volume falling in each bar as a line with circular markers, inside a shaded band of one standard deviation across tickers, with a dashed horizontal line marking the level an even split across the session would give. The right is a bar chart of the mean number of trades in each bar over the same hours.",
)

# Name the bars by the clock, not by position: the number of bars follows from
# PATTERN_FREQ, so an index into the middle is not a fixed time of day.


def slot_label(slot: float) -> str:
    """Render a decimal hour such as 12.5 as a clock time."""
    hour, minute = divmod(round(slot * 60), 60)
    return f"{hour:02d}:{minute:02d}"


midday = int(np.argmin(np.abs(times - 12.5)))
print(f"Share of daily volume by {PATTERN_FREQ} bar, averaged over the selected tickers:")
print(f"  Opening bar   ({slot_label(times[0])}): {vol_pct[0]:.1f}%")
print(f"  Midday bar    ({slot_label(times[midday])}): {vol_pct[midday]:.1f}%")
print(f"  Closing bar   ({slot_label(times[-1])}): {vol_pct[-1]:.1f}%")
print(
    f"  Opening and closing bars against the midday bar: "
    f"{(vol_pct[0] + vol_pct[-1]) / (2 * vol_pct[midday]):.1f}x"
)


# %% [markdown]
# ## Key Takeaways
#
# 1. **Volume is not spread evenly across a session.** The opening and closing bars carry
#    a multiple of what a midday bar carries; the figure above draws the level an even
#    split would give, and the printed ratio says by how much the ends exceed the middle.
#    Any statistic computed per bar - a volatility, a spread, an average trade size - is
#    estimated from very different sample sizes depending on when the bar falls.
# 2. **Normalise each ticker before averaging shapes.** Expressing every bar as a share
#    of that ticker's own day is what makes the average a shape rather than a picture of
#    whichever ticker traded most.
# 3. **Read the bar by the clock, not by its index.** The number of bars follows from the
#    chosen frequency, so 'the middle one' is a different time of day at 15 minutes than
#    at 30, and a label written against one is wrong for the other.
# 4. **Time of day is a feature.** Chapter 8 encodes it directly, and this is the shape
#    it encodes.
#
# ### Known limitations
#
# - One venue and one session. The U-shape is a well-established regularity, and one day
#   of one venue illustrates it rather than establishing it.
# - The three per-tier panels are one ticker each, chosen by traded value. They show what
#   the shape looks like at different activity levels; they do not test whether it varies
#   systematically with liquidity, which would need the whole cross-section.
# - Regular trading hours only. Pre- and post-market bars are dropped rather than shown.
#
# **Next**: `07_itch_stylized_facts` for bid-ask bounce and liquidity.
#
# ---
#
# ## Reference
#
# Bouchaud, J.-P., Bonart, J., Donier, J., & Gould, M. (2018).
# *Trades, Quotes and Prices: Financial Markets Under the Microscope*.
# Cambridge University Press.
# [https://doi.org/10.1017/9781009028943](https://doi.org/10.1017/9781009028943)
