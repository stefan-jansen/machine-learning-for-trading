# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NASDAQ-100 Minute Bars - Exploratory Data Analysis
#
# **Chapter 3: Market Microstructure**
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# This notebook provides a comprehensive exploration of the AlgoSeek TAQ minute bar dataset.
# Unlike simple OHLCV data, TAQ bars contain 61 pre-computed columns that capture the
# full microstructure of each trading minute: quote dynamics, trade execution, aggressor
# behavior, and liquidity conditions.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Load and validate TAQ minute bars using the canonical loader
# - Understand the schema: quote-derived vs trade-derived columns
# - Interpret null patterns (bars with no trades are normal, not errors)
# - Analyze trade bucket fields to measure aggressor behavior
# - Interpret pressure and tick direction indicators
# - Distinguish exchange volume from off-exchange (FINRA) activity
# - Connect spread dynamics to liquidity conditions
#
# ## Dataset Overview
#
# | Attribute | Value |
# |-----------|-------|
# | **Source** | AlgoSeek TAQ Minute Bars |
# | **Coverage** | NASDAQ-100 constituents |
# | **Period** | 2020-2021 |
# | **Frequency** | 1-minute bars (continuous, 04:00-20:00 ET) |
# | **Columns** | 61 pre-computed microstructure fields |
#
# ## Book reference
#
# Section §3.2, *The Anatomy of Modern Market Data Feeds* — AlgoSeek
# minute-bars bullet (61 pre-computed microstructure columns serving as
# input for Chapter 7 feature engineering).
#
# ## Prerequisites
#
# - AlgoSeek TAQ minute bars under
#   `data/equities/nasdaq100/minute_bars/year=YYYY/month=MM.parquet` (Hive
#   partitioned). Loaded via `load_nasdaq100_bars`.
# - Companion: `11_algoseek_taq_eda` for the underlying tick-level stream
#   these bars summarize.

# %%
"""NASDAQ-100 Minute Bars EDA — exploring AlgoSeek TAQ minute bar microstructure fields."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import polars as pl

from data import load_nasdaq100_bars
from utils.data_quality import check_ohlc_invariants, describe_coverage, null_rate, per_asset_stats

# Polars display configuration

# %% tags=["parameters"]
# Always limit data to avoid OOM (full NASDAQ-100 is ~50M rows)
MAX_SYMBOLS = 10
END_DATE = "2020-12-31"

# %%
# Select top N tickers by market cap
_ALL_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "GOOG", "AVGO", "COST"]
SAMPLE_TICKERS = _ALL_TICKERS[:MAX_SYMBOLS]

print(f"Tickers: {len(SAMPLE_TICKERS)}")

# %% [markdown]
# ## 1. Load and Inspect
#
# The NASDAQ-100 minute bars are stored in Hive-partitioned Parquet files:
# `equities/nasdaq100/minute_bars/year={YYYY}/month={MM}.parquet`

# %%
# Load data with all microstructure columns
# Limit to 2020 to avoid OOM (full 2020-2021 x 100 symbols is ~50M rows)
df = load_nasdaq100_bars(
    symbols=SAMPLE_TICKERS,
    start_date="2020-01-01",
    end_date=END_DATE,
    include_microstructure=True,
)

print("=== NASDAQ-100 Minute Bars ===")
print(f"Shape: {df.shape}")
print(f"Symbols: {df['symbol'].n_unique()}")
print(f"Memory: {df.estimated_size('mb'):.1f} MB")

# %% [markdown]
# ### Schema Overview
#
# TAQ minute bars contain 61 columns organized into distinct families based on
# the underlying market mechanism they measure.

# %%
# Full schema
print("\n=== Schema (All 61 Columns) ===")
for i, (col, dtype) in enumerate(df.schema.items()):
    print(f"  {i + 1:2d}. {col}: {dtype}")

# %% [markdown]
# ### Column Families
#
# The 61 columns can be grouped into 7 families based on what they measure:
#
# | Family | Columns | What They Measure |
# |--------|---------|-------------------|
# | **Identifiers** | date, symbol, time, timestamp | Row keys |
# | **Quote OHLC** | open/high/low/close_bid/ask_price/size/time | NBBO dynamics |
# | **Trade OHLC** | first/high/low/last_trade_price/size/time | Actual executions |
# | **Spread** | min_spread, max_spread | Bid-ask spread range |
# | **Volume** | volume, finra_volume, total_trades | Exchange vs off-exchange |
# | **Trade Buckets** | trade_at_bid/mid/ask, trade_at_cross | Aggressor direction |
# | **Pressure** | trade_to_mid_vol_weight, uptick/downtick_volume | Directional intensity |

# %%
# Column families — quote OHLC
COLUMN_FAMILIES = {
    "Identifiers": ["timestamp", "symbol", "time", "timestamp", "year", "month"],
    "Quote OHLC (Prices)": [
        "open_bid_price",
        "open_ask_price",
        "high_bid_price",
        "high_ask_price",
        "low_bid_price",
        "low_ask_price",
        "close_bid_price",
        "close_ask_price",
    ],
    "Quote OHLC (Sizes)": [
        "open_bid_size",
        "open_ask_size",
        "high_bid_size",
        "high_ask_size",
        "low_bid_size",
        "low_ask_size",
        "close_bid_size",
        "close_ask_size",
    ],
    "Quote OHLC (Times)": [
        "open_bar_time",
        "high_bid_time",
        "high_ask_time",
        "low_bid_time",
        "low_ask_time",
        "close_bar_time",
    ],
}

# %%
# Column families — trade OHLC
COLUMN_FAMILIES.update(
    {
        "Trade OHLC": [
            "first_trade_price",
            "first_trade_size",
            "first_trade_time",
            "high_trade_price",
            "high_trade_size",
            "high_trade_time",
            "low_trade_price",
            "low_trade_size",
            "low_trade_time",
            "last_trade_price",
            "last_trade_size",
            "last_trade_time",
        ],
        "Spread": ["min_spread", "max_spread"],
        "Volume": ["volume", "finra_volume", "total_trades", "cancel_size"],
    }
)

# %%
# Column families — order flow and pressure indicators
COLUMN_FAMILIES.update(
    {
        "Trade Buckets": [
            "trade_at_bid",
            "trade_at_bid_mid",
            "trade_at_mid",
            "trade_at_mid_ask",
            "trade_at_ask",
            "trade_at_cross",
        ],
        "Tick Direction": [
            "uptick_volume",
            "downtick_volume",
            "repeat_uptick_volume",
            "repeat_downtick_volume",
            "unknown_tick_volume",
        ],
        "Pressure & VWAP": [
            "vwap",
            "finra_vwap",
            "trade_to_mid_vol_weight",
            "trade_to_mid_vol_weight_rel",
            "time_weight_bid",
            "time_weight_ask",
        ],
        "Quote Count": ["nbbo_quote_count"],
    }
)

# %%
print("\n=== Column Families ===")
for family, cols in COLUMN_FAMILIES.items():
    available = [c for c in cols if c in df.columns]
    print(f"  {family}: {len(available)}/{len(cols)} columns")

# %% [markdown]
# ## 2. Coverage Summary
#
# TAQ minute bars are **continuous**: a row exists for every minute from 04:00 ET
# to 20:00 ET, even when no trades occur. This is different from tick data where
# rows only exist when events happen.

# %%
# Overall coverage
cov = describe_coverage(df, time_col="timestamp", asset_col="symbol")
print("=== Coverage ===")
print(f"  Total rows: {cov['rows']:,}")
print(f"  Symbols: {cov['assets']}")
print(f"  Time range: {cov['time_min']} to {cov['time_max']}")
print(f"  Unique timestamps: {cov['unique_times']:,}")

# %%
# Per-symbol statistics
symbol_stats = per_asset_stats(
    df,
    time_col="timestamp",
    asset_col="symbol",
    price_col="last_trade_price",
    volume_col="volume",
)

print("\nPer-Symbol Statistics:")
print(symbol_stats)

# %%
# Trading hours breakdown
# TAQ bars cover extended hours: 04:00-09:30 (pre), 09:30-16:00 (RTH), 16:00-20:00 (post)
df_hours = df.with_columns(pl.col("timestamp").dt.hour().alias("hour"))

hours_dist = (
    df_hours.group_by("hour")
    .agg(pl.len().alias("bars"), pl.col("volume").sum().alias("total_volume"))
    .sort("hour")
)

print("\n=== Bars by Hour (Sample) ===")
print(hours_dist.filter(pl.col("hour").is_in([4, 9, 10, 12, 15, 16, 19])))

# %% [markdown]
# ## 3. Null Patterns: Semantic, Not Errors
#
# TAQ minute bars have two types of columns with different null semantics:
#
# | Column Type | When Null | Meaning |
# |-------------|-----------|---------|
# | **Quote fields** | Rarely | No quote activity (very rare) |
# | **Trade fields** | Frequently | No trades in this minute bar |
#
# Null trade fields are NOT missing data - they correctly indicate "no trading
# activity in this minute." This is common in extended hours and for illiquid names.

# %%
# Null rates by column family
quote_cols = ["open_bid_price", "close_ask_price", "min_spread", "max_spread"]
trade_cols = ["first_trade_price", "last_trade_price", "vwap", "trade_to_mid_vol_weight"]

print("=== Null Rates: Quote Fields (Should Be ~0%) ===")
quote_nulls = null_rate(df, [c for c in quote_cols if c in df.columns])
for row in quote_nulls.iter_rows(named=True):
    print(f"  {row['column']}: {row['null_pct']:.2f}%")

print("\n=== Null Rates: Trade Fields (Expected ~20% with extended hours) ===")
trade_nulls = null_rate(df, [c for c in trade_cols if c in df.columns])
for row in trade_nulls.iter_rows(named=True):
    print(f"  {row['column']}: {row['null_pct']:.1f}%")

print("\nNote: Trade field nulls indicate bars with no trades (normal, not errors)")

# %%
# Null rates by trading session
df_sessions = df.with_columns(
    pl.when(pl.col("timestamp").dt.hour() < 9)
    .then(pl.lit("Pre-market"))
    .when(
        (pl.col("timestamp").dt.hour() >= 9)
        & ((pl.col("timestamp").dt.hour() > 9) | (pl.col("timestamp").dt.minute() >= 30))
        & (pl.col("timestamp").dt.hour() < 16)
    )
    .then(pl.lit("Regular"))
    .otherwise(pl.lit("Post-market"))
    .alias("session")
)

session_nulls = (
    df_sessions.group_by("session")
    .agg(
        pl.len().alias("bars"),
        pl.col("last_trade_price").is_null().mean().alias("null_rate"),
    )
    .sort("session")
)

print("\n=== Trade Null Rates by Session ===")
for row in session_nulls.iter_rows(named=True):
    print(f"  {row['session']}: {row['null_rate'] * 100:.1f}% null ({row['bars']:,} bars)")

# %% [markdown]
# ## 4. Data Quality: OHLC Invariants
#
# Trade OHLC must satisfy invariants: High ≥ Low, High ≥ Open/Close, Low ≤ Open/Close.

# %%
# OHLC invariants (trade prices)
invariants = check_ohlc_invariants(
    df,
    open_col="first_trade_price",
    high_col="high_trade_price",
    low_col="low_trade_price",
    close_col="last_trade_price",
    volume_col="volume",
)

print("=== OHLC Invariants (Trade Prices) ===")
for row in invariants.iter_rows(named=True):
    status = "[OK]" if row["valid_pct"] >= 99.99 else "[WARN]"
    print(f"  {status} {row['check']}: {row['valid_pct']:.2f}%")

# %% [markdown]
# ## 5. Quote OHLC: NBBO Dynamics
#
# Quote fields capture the National Best Bid and Offer (NBBO) at key moments:
#
# | Field Pattern | Meaning |
# |--------------|---------|
# | `open_bid/ask_price` | NBBO at bar start (carried forward if no change) |
# | `high_bid/ask_price` | Highest bid/ask during the bar |
# | `low_bid/ask_price` | Lowest bid/ask during the bar |
# | `close_bid/ask_price` | NBBO at bar end |
#
# Quote fields are **never null** because NBBO is always defined (carried forward).

# %%
# Quote price statistics
quote_stats = df.select(
    [
        pl.col("open_bid_price").mean().alias("avg_bid"),
        pl.col("open_ask_price").mean().alias("avg_ask"),
        (pl.col("open_ask_price") - pl.col("open_bid_price")).mean().alias("avg_spread"),
        pl.col("open_bid_size").mean().alias("avg_bid_size"),
        pl.col("open_ask_size").mean().alias("avg_ask_size"),
    ]
)

print("=== Quote Statistics ===")
print(quote_stats)

# %%
# Compute midpoint and spread in basis points
df = df.with_columns(
    ((pl.col("open_bid_price") + pl.col("open_ask_price")) / 2).alias("mid_price"),
)
df = df.with_columns(
    pl.when(pl.col("mid_price") > 0)
    .then((pl.col("open_ask_price") - pl.col("open_bid_price")) / pl.col("mid_price") * 10_000)
    .otherwise(None)
    .alias("spread_bps"),
)

print("\n=== Spread Distribution (bps) ===")
print(
    df.select(
        pl.col("spread_bps").min().alias("min"),
        pl.col("spread_bps").quantile(0.25).alias("q25"),
        pl.col("spread_bps").median().alias("median"),
        pl.col("spread_bps").quantile(0.75).alias("q75"),
        pl.col("spread_bps").max().alias("max"),
    )
)

# %% [markdown]
# ## 6. Trade OHLC: Actual Executions
#
# Trade fields capture actual executed trades, which may differ from quotes:
#
# | Field | Meaning |
# |-------|---------|
# | `first_trade_price/size/time` | First trade of the minute |
# | `high_trade_price/size/time` | Trade at highest price |
# | `low_trade_price/size/time` | Trade at lowest price |
# | `last_trade_price/size/time` | Final trade of the minute |
#
# Trade fields are **null when no trades occur** in the bar (common in extended hours).

# %%
# Trade statistics (excluding nulls)
trade_stats = df.filter(pl.col("last_trade_price").is_not_null()).select(
    [
        pl.col("first_trade_price").mean().alias("avg_open"),
        pl.col("high_trade_price").mean().alias("avg_high"),
        pl.col("low_trade_price").mean().alias("avg_low"),
        pl.col("last_trade_price").mean().alias("avg_close"),
        pl.col("first_trade_size").mean().alias("avg_first_size"),
        pl.col("last_trade_size").mean().alias("avg_last_size"),
    ]
)

print("=== Trade Statistics (Bars With Trades) ===")
print(trade_stats)

# %%
# Compare trade price to midpoint
df = df.with_columns(
    pl.when(pl.col("last_trade_price").is_not_null() & (pl.col("mid_price") > 0))
    .then((pl.col("last_trade_price") - pl.col("mid_price")) / pl.col("mid_price") * 10_000)
    .otherwise(None)
    .alias("last_trade_vs_mid_bps"),
)

print("\n=== Last Trade vs Midpoint (bps) ===")
print(
    df.filter(pl.col("last_trade_vs_mid_bps").is_not_null()).select(
        pl.col("last_trade_vs_mid_bps").quantile(0.01).alias("p1"),
        pl.col("last_trade_vs_mid_bps").quantile(0.25).alias("q25"),
        pl.col("last_trade_vs_mid_bps").median().alias("median"),
        pl.col("last_trade_vs_mid_bps").quantile(0.75).alias("q75"),
        pl.col("last_trade_vs_mid_bps").quantile(0.99).alias("p99"),
    )
)

# %% [markdown]
# ## 7. Trade Buckets: Aggressor Direction
#
# Trade buckets classify each trade by execution price relative to NBBO at trade time:
#
# | Bucket | Location | Signal |
# |--------|----------|--------|
# | `trade_at_bid` | At or below best bid | **Seller aggressor** (hitting bid) |
# | `trade_at_bid_mid` | Between bid and mid | Seller with price improvement |
# | `trade_at_mid` | At midpoint | Neutral / matched |
# | `trade_at_mid_ask` | Between mid and ask | Buyer with price improvement |
# | `trade_at_ask` | At or above best ask | **Buyer aggressor** (lifting offer) |
# | `trade_at_cross` | Crossed/locked market | Exclude from imbalance (anomaly) |
#
# ### Why This Matters
#
# The **aggressor** is the party crossing the spread to trade immediately. Tracking
# aggressor flow reveals **informed vs uninformed order flow patterns**:
#
# - **High trade_at_ask**: Buyers are aggressive → bullish pressure
# - **High trade_at_bid**: Sellers are aggressive → bearish pressure
#
# > **Price Improvement**: When a seller executes between bid and mid, they received
# > more than the bid price. When a buyer executes between mid and ask, they paid
# > less than the ask. This often indicates off-exchange matching or hidden liquidity.
#
# The `trade_at_cross` bucket corresponds to locked/crossed markets where bid ≥ ask.
# These are excluded from imbalance calculations because aggressor direction is
# undefined.

# %%
# Trade bucket statistics
bucket_cols = [
    "trade_at_bid",
    "trade_at_bid_mid",
    "trade_at_mid",
    "trade_at_mid_ask",
    "trade_at_ask",
    "trade_at_cross",
]
available_buckets = [c for c in bucket_cols if c in df.columns]

if available_buckets:
    bucket_stats = df.select([pl.col(c).sum().alias(c) for c in available_buckets])

    print("=== Trade Bucket Volume Distribution ===")
    total = sum(bucket_stats.row(0))
    for col in available_buckets:
        vol = bucket_stats[col][0]
        pct = 100 * vol / total if total > 0 else 0
        print(f"  {col}: {vol:,} shares ({pct:.1f}%)")

# %%
# Compute Order Flow Imbalance (OFI)
if all(
    c in df.columns
    for c in ["trade_at_bid", "trade_at_bid_mid", "trade_at_ask", "trade_at_mid_ask"]
):
    df = df.with_columns(
        (pl.col("trade_at_ask") + pl.col("trade_at_mid_ask")).alias("buy_aggressor"),
        (pl.col("trade_at_bid") + pl.col("trade_at_bid_mid")).alias("sell_aggressor"),
    )
    df = df.with_columns(
        pl.when((pl.col("buy_aggressor") + pl.col("sell_aggressor")) > 0)
        .then(
            (pl.col("buy_aggressor") - pl.col("sell_aggressor"))
            / (pl.col("buy_aggressor") + pl.col("sell_aggressor"))
        )
        .otherwise(0.0)
        .alias("order_flow_imbalance"),
    )

    print("\n=== Order Flow Imbalance (OFI) ===")
    print("OFI = (buy_aggressor - sell_aggressor) / (buy_aggressor + sell_aggressor)")
    print("Range: -1 (all sellers) to +1 (all buyers)\n")
    print(
        df.select(
            pl.col("order_flow_imbalance").mean().alias("mean"),
            pl.col("order_flow_imbalance").std().alias("std"),
            pl.col("order_flow_imbalance").quantile(0.25).alias("q25"),
            pl.col("order_flow_imbalance").quantile(0.75).alias("q75"),
        )
    )

# %% [markdown]
# ### Trading Hypotheses: OFI Momentum and Reversal
#
# **Momentum Hypothesis**: Persistent aggressor imbalance predicts short-term price direction.
#
# | Signal | Condition | Expected Outcome |
# |--------|-----------|------------------|
# | Strong Buy Pressure | OFI > 0.3 for multiple bars | Price increase |
# | Strong Sell Pressure | OFI < -0.3 for multiple bars | Price decrease |
#
# **Reversal Hypothesis**: Extreme OFI leads to short-term mean reversion.
#
# | Signal | Condition | Expected Outcome |
# |--------|-----------|------------------|
# | Extreme Buying | OFI > 0.7 (exhaustion) | Potential reversal down |
# | Extreme Selling | OFI < -0.7 (capitulation) | Potential reversal up |
#
# These form the basis for the **order flow reversal strategy** developed in later chapters.

# %% [markdown]
# ## 8. Tick Direction: Trade-Level Momentum
#
# Tick direction tracks price movement at each trade:
#
# | Field | Meaning |
# |-------|---------|
# | `uptick_volume` | Volume at prices > previous trade |
# | `downtick_volume` | Volume at prices < previous trade |
# | `repeat_uptick_volume` | Volume at same price after uptick |
# | `repeat_downtick_volume` | Volume at same price after downtick |
# | `unknown_tick_volume` | First trade of day (no prior reference) |
#
# ### Why This Matters
#
# Tick direction captures **price momentum at the trade level**. Unlike simple returns,
# this shows the actual trade-by-trade direction of price movement. An uptick ratio
# persistently above 0.5 indicates buying momentum; below 0.5 indicates selling momentum.
#
# This differs from OFI in that it measures **sequential price changes** rather than
# **aggressor direction**. Both are useful but capture different aspects of order flow.

# %%
tick_cols = ["uptick_volume", "downtick_volume", "repeat_uptick_volume", "repeat_downtick_volume"]
if all(c in df.columns for c in tick_cols):
    df = df.with_columns(
        (
            pl.col("uptick_volume")
            + pl.col("repeat_uptick_volume")
            - pl.col("downtick_volume")
            - pl.col("repeat_downtick_volume")
        ).alias("net_uptick_volume"),
    )
    df = df.with_columns(
        pl.when(
            (
                pl.col("uptick_volume")
                + pl.col("downtick_volume")
                + pl.col("repeat_uptick_volume")
                + pl.col("repeat_downtick_volume")
            )
            > 0
        )
        .then(
            (pl.col("uptick_volume") + pl.col("repeat_uptick_volume"))
            / (
                pl.col("uptick_volume")
                + pl.col("downtick_volume")
                + pl.col("repeat_uptick_volume")
                + pl.col("repeat_downtick_volume")
            )
        )
        .otherwise(0.5)
        .alias("uptick_ratio"),
    )

    print("=== Tick Direction Statistics ===")
    print(
        df.select(
            pl.col("uptick_ratio").mean().alias("avg_uptick_ratio"),
            pl.col("uptick_ratio").std().alias("std_uptick_ratio"),
            pl.col("net_uptick_volume").mean().alias("avg_net_uptick"),
        )
    )

# %% [markdown]
# ## 9. Pressure Indicators: Directional Intensity
#
# Pressure fields measure how far trades execute from the midpoint:
#
# | Field | Formula | Range |
# |-------|---------|-------|
# | `trade_to_mid_vol_weight` | Σ(trade_price - mid) × volume / Σ volume | Dollars |
# | `trade_to_mid_vol_weight_rel` | Same, normalized by spread | Unitless |
#
# ### Why This Matters
#
# - **Positive pressure**: Trades executing above midpoint → buying pressure
# - **Negative pressure**: Trades executing below midpoint → selling pressure
#
# The **relative** version normalizes by spread, making it comparable across stocks
# with different price levels and liquidity. A relative pressure of 0.5 means trades
# are on average executing halfway between the midpoint and the ask (consistent buying).
#
# Unlike OFI which counts shares by bucket, pressure measures the **magnitude** of
# price impact - how aggressively traders are pushing prices away from fair value.

# %%
pressure_cols = ["trade_to_mid_vol_weight", "trade_to_mid_vol_weight_rel"]
if all(c in df.columns for c in pressure_cols):
    print("=== Pressure Indicator Statistics ===")
    print(
        df.filter(pl.col("trade_to_mid_vol_weight_rel").is_not_null()).select(
            pl.col("trade_to_mid_vol_weight").mean().alias("avg_absolute"),
            pl.col("trade_to_mid_vol_weight_rel").mean().alias("avg_relative"),
            pl.col("trade_to_mid_vol_weight_rel").std().alias("std_relative"),
            pl.col("trade_to_mid_vol_weight_rel").quantile(0.10).alias("q10"),
            pl.col("trade_to_mid_vol_weight_rel").quantile(0.90).alias("q90"),
        )
    )

# %% [markdown]
# ## 10. Volume: Exchange vs Off-Exchange (FINRA)
#
# Volume is split between lit exchanges and off-exchange venues:
#
# | Field | Source | Description |
# |-------|--------|-------------|
# | `volume` | Lit exchanges | NYSE, NASDAQ, BATS, etc. |
# | `finra_volume` | Off-exchange | Dark pools, internalizers, OTC |
# | `total_trades` | All | Number of trades in the bar |
#
# **Total Volume = volume + finra_volume**
#
# > **Data Contract**: FINRA Trade Reporting Facility (TRF) collects reports of
# > off-exchange trades (dark pools, internalizers, OTC). These trades are included
# > in the consolidated tape but execute away from lit exchanges.
#
# ### Why This Matters
#
# - **High FINRA share (>40%)**: Institutional activity, larger trades seeking anonymity
# - **Low FINRA share (<20%)**: Retail-dominated, smaller trades
# - **FINRA spikes**: May indicate large block trades seeking minimal market impact
#
# This can be used for **regime filtering** in trading strategies - some signals
# work better in institutional vs retail regimes.

# %%
if "finra_volume" in df.columns:
    df = df.with_columns(
        (pl.col("volume") + pl.col("finra_volume")).alias("total_volume"),
    )
    df = df.with_columns(
        pl.when(pl.col("total_volume") > 0)
        .then(pl.col("finra_volume") / pl.col("total_volume"))
        .otherwise(0.0)
        .alias("finra_share"),
    )

    print("=== Volume Statistics ===")
    print(
        df.select(
            pl.col("volume").sum().alias("exchange_volume"),
            pl.col("finra_volume").sum().alias("finra_volume"),
            pl.col("total_volume").sum().alias("total_volume"),
            (pl.col("finra_volume").sum() / pl.col("total_volume").sum()).alias("finra_share"),
        )
    )

    # FINRA share by hour (intraday pattern)
    df_rth = df.filter((pl.col("timestamp").dt.hour() >= 10) & (pl.col("timestamp").dt.hour() < 16))

    hourly_finra = (
        df_rth.with_columns(pl.col("timestamp").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(pl.col("finra_share").mean().alias("avg_finra_share"))
        .sort("hour")
    )

    print("\n=== FINRA Share by Hour (RTH) ===")
    for row in hourly_finra.iter_rows(named=True):
        print(f"  {row['hour']:02d}:00: {row['avg_finra_share'] * 100:.1f}%")

# %% [markdown]
# ## 11. Spread: Liquidity Conditions
#
# Spread fields capture bid-ask spread dynamics:
#
# | Field | Meaning | Note |
# |-------|---------|------|
# | `min_spread` | Minimum spread during bar | **0 = locked/crossed market** |
# | `max_spread` | Maximum spread during bar | Widening indicates stress |
#
# **min_spread = 0 is an anomaly indicator**, not "free liquidity." It signals
# market stress where bid ≥ ask (locked or crossed market).

# %%
if "min_spread" in df.columns:
    # Spread statistics
    print("=== Spread Statistics ===")
    print(
        df.select(
            pl.col("min_spread").mean().alias("avg_min_spread"),
            pl.col("max_spread").mean().alias("avg_max_spread"),
            (pl.col("min_spread") == 0).mean().alias("locked_market_rate"),
        )
    )

    # Spread by hour
    hourly_spread = (
        df_rth.with_columns(pl.col("timestamp").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(
            pl.col("spread_bps").mean().alias("avg_spread_bps"),
            (pl.col("min_spread") == 0).mean().alias("locked_rate"),
        )
        .sort("hour")
    )

    print("\n=== Spread by Hour (RTH) ===")
    for row in hourly_spread.iter_rows(named=True):
        print(
            f"  {row['hour']:02d}:00: {row['avg_spread_bps']:.1f} bps, "
            f"{row['locked_rate'] * 100:.2f}% locked"
        )

# %% [markdown]
# ## 12. VWAP and Time-Weighted Prices
#
# | Field | Meaning |
# |-------|---------|
# | `vwap` | Volume-weighted average price (exchange trades) |
# | `finra_vwap` | VWAP of off-exchange trades |
# | `time_weight_bid` | Time-weighted average bid |
# | `time_weight_ask` | Time-weighted average ask |

# %%
vwap_cols = ["vwap", "finra_vwap", "time_weight_bid", "time_weight_ask"]
available_vwap = [c for c in vwap_cols if c in df.columns]

if available_vwap:
    print("=== VWAP and Time-Weighted Prices ===")
    for col in available_vwap:
        non_null = df.filter(pl.col(col).is_not_null())
        if len(non_null) > 0:
            avg = non_null[col].mean()
            null_pct = 100 * (len(df) - len(non_null)) / len(df)
            if avg is not None:
                print(f"  {col}: avg=${avg:.2f}, {null_pct:.1f}% null")
            else:
                print(f"  {col}: no valid values, {null_pct:.1f}% null")
        else:
            print(f"  {col}: all null")
else:
    print("=== VWAP columns not present in this dataset ===")

# %% [markdown]
# ## 13. Visualizations: Single Day Deep Dive

# %%
# Pick highest-volume symbol and day
vol_col = "total_volume" if "total_volume" in df.columns else "volume"

top_symbol = (
    df.group_by("symbol")
    .agg(pl.col(vol_col).sum().alias("tv"))
    .sort("tv", descending=True)
    .select("symbol")
    .row(0)[0]
)

top_day = (
    df.filter(pl.col("symbol") == top_symbol)
    .with_columns(pl.col("timestamp").dt.date().alias("timestamp"))
    .group_by("timestamp")
    .agg(pl.col(vol_col).sum().alias("day_vol"))
    .sort("day_vol", descending=True)
    .select("timestamp")
    .row(0)[0]
)

df_day = (
    df.filter(pl.col("symbol") == top_symbol)
    .filter(pl.col("timestamp").dt.date() == pl.lit(top_day))
    .sort("timestamp")
)

print(f"=== Deep Dive: {top_symbol} on {top_day} ({df_day.height:,} bars) ===")

# %%
# Multi-panel microstructure dashboard
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(14, 12), sharex=True)

ts = df_day["timestamp"].to_numpy()

# Panel 1: Price
if "last_trade_price" in df_day.columns:
    axes[0].plot(ts, df_day["last_trade_price"].to_numpy(), label="Last Trade", linewidth=1)
if "mid_price" in df_day.columns:
    axes[0].plot(ts, df_day["mid_price"].to_numpy(), label="Midpoint", alpha=0.7, linestyle="--")
axes[0].set_ylabel("Price ($)")
axes[0].legend(loc="upper left")
axes[0].set_title(f"{top_symbol} - {top_day} - Microstructure Dashboard")

# Panel 2: Spread
if "spread_bps" in df_day.columns:
    axes[1].fill_between(ts, df_day["spread_bps"].to_numpy(), color="orange", alpha=0.5)
    axes[1].set_ylabel("Spread (bps)")

# Panel 3: Volume
if "volume" in df_day.columns and "finra_volume" in df_day.columns:
    axes[2].bar(ts, df_day["volume"].to_numpy(), width=0.0005, label="Exchange", color="steelblue")
    axes[2].bar(
        ts,
        df_day["finra_volume"].to_numpy(),
        width=0.0005,
        bottom=df_day["volume"].to_numpy(),
        label="FINRA",
        color="purple",
        alpha=0.7,
    )
    axes[2].set_ylabel("Volume")
    axes[2].set_yscale("log")
    axes[2].legend(loc="upper right")

# Panel 4: OFI
if "order_flow_imbalance" in df_day.columns:
    ofi = df_day["order_flow_imbalance"].to_numpy()
    colors = ["green" if x > 0 else "red" for x in ofi]
    axes[3].bar(ts, ofi, width=0.0005, color=colors, alpha=0.7)
    axes[3].set_ylabel("OFI")
    axes[3].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[3].set_ylim(-1, 1)

# Panel 5: Uptick ratio
if "uptick_ratio" in df_day.columns:
    axes[4].plot(ts, df_day["uptick_ratio"].to_numpy(), color="green", alpha=0.7)
    axes[4].set_ylabel("Uptick Ratio")
    axes[4].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[4].set_ylim(0, 1)
    axes[4].set_xlabel("Time (ET)")

plt.tight_layout()
plt.show()

# Clean up large arrays to free memory
del ts, fig, axes
import gc

gc.collect()

# %% [markdown]
# ## 14. Intraday Patterns
#
# Key microstructure metrics show distinct patterns across the trading day.

# %%
# Compute hourly averages for RTH (9:30-16:00)
df_rth_hours = (
    df.filter((pl.col("timestamp").dt.hour() >= 10) & (pl.col("timestamp").dt.hour() < 16))
    .with_columns(pl.col("timestamp").dt.hour().alias("hour"))
    .group_by("hour")
    .agg(
        pl.col("spread_bps").mean().alias("avg_spread"),
        pl.col("volume").sum().alias("total_volume"),
        pl.col("finra_share").mean().alias("avg_finra_share")
        if "finra_share" in df.columns
        else pl.lit(0).alias("avg_finra_share"),
        pl.col("order_flow_imbalance").std().alias("ofi_volatility")
        if "order_flow_imbalance" in df.columns
        else pl.lit(0).alias("ofi_volatility"),
    )
    .sort("hour")
)

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

hours = df_rth_hours["hour"].to_numpy()

# Spread pattern
axes[0, 0].plot(hours, df_rth_hours["avg_spread"].to_numpy(), marker="o", color="orange")
axes[0, 0].set_xlabel("Hour (ET)")
axes[0, 0].set_ylabel("Spread (bps)")
axes[0, 0].set_title("Average Spread by Hour")
axes[0, 0].grid(True, alpha=0.3)

# Volume pattern
axes[0, 1].bar(hours, df_rth_hours["total_volume"].to_numpy() / 1e9, color="steelblue")
axes[0, 1].set_xlabel("Hour (ET)")
axes[0, 1].set_ylabel("Volume (Billions)")
axes[0, 1].set_title("Total Volume by Hour")
axes[0, 1].grid(True, alpha=0.3)

# FINRA share pattern
if "finra_share" in df.columns:
    axes[1, 0].plot(
        hours, df_rth_hours["avg_finra_share"].to_numpy() * 100, marker="o", color="purple"
    )
    axes[1, 0].set_xlabel("Hour (ET)")
    axes[1, 0].set_ylabel("FINRA Share (%)")
    axes[1, 0].set_title("Off-Exchange Activity by Hour")
    axes[1, 0].grid(True, alpha=0.3)

# OFI volatility pattern
if "order_flow_imbalance" in df.columns:
    axes[1, 1].plot(hours, df_rth_hours["ofi_volatility"].to_numpy(), marker="o", color="green")
    axes[1, 1].set_xlabel("Hour (ET)")
    axes[1, 1].set_ylabel("OFI Std Dev")
    axes[1, 1].set_title("Order Flow Imbalance Volatility by Hour")
    axes[1, 1].grid(True, alpha=0.3)

plt.suptitle("Intraday Microstructure Patterns", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 15. Trading Mechanism Menu: Connecting Fields to Strategies
#
# The microstructure fields in TAQ minute bars connect to specific trading mechanisms:
#
# ### Intraday Momentum / Trend Persistence
#
# **Features**: `order_flow_imbalance`, `uptick_ratio`, `trade_to_mid_vol_weight`
#
# | Signal | Condition | Strategy |
# |--------|-----------|----------|
# | OFI > 0.3 for 5+ bars | Persistent buying | Follow momentum |
# | Uptick ratio > 0.6 | Price trending up | Hold long positions |
# | Positive pressure consistent | Aggressive buying | Continuation expected |
#
# ### Liquidity Shock / Spread Widening
#
# **Features**: `spread_bps`, `nbbo_quote_count`, `min_spread`
#
# | Signal | Condition | Strategy |
# |--------|-----------|----------|
# | Spread > 2× rolling mean | Liquidity withdrawal | Reduce position size |
# | min_spread = 0 | Locked/crossed market | Fast market, pause trading |
# | Quote count drop | Market makers stepping back | Increase execution urgency |
#
# ### Hidden Liquidity / Institutional Regime
#
# **Features**: `finra_share`, `finra_volume`
#
# | Signal | Condition | Strategy |
# |--------|-----------|----------|
# | FINRA share > 40% | Institutional activity | Larger moves possible |
# | FINRA spike vs previous bars | Block trade | Watch for continuation |
# | FINRA share < 20% | Retail-dominated | Mean reversion may work |
#
# ### Mean Reversion / Exhaustion
#
# **Features**: `order_flow_imbalance` (extreme values)
#
# | Signal | Condition | Strategy |
# |--------|-----------|----------|
# | OFI > 0.7 | Extreme buying (exhaustion) | Fade the move |
# | OFI < -0.7 | Extreme selling (capitulation) | Buy the dip |
#
# > These mechanisms form the foundation for the **order flow reversal strategy**
# > developed in Chapters 7-12, which transforms these raw fields into predictive
# > alpha factors.

# %% [markdown]
# ## Key Takeaways
#
# ### Dataset Structure
#
# 1. **61 columns** organized into 7 families: identifiers, quote OHLC, trade OHLC,
#    spread, volume, trade buckets, and pressure indicators
# 2. **Continuous bars**: Rows exist for every minute 04:00-20:00 ET, even without trades
# 3. **Null semantics**: Trade fields null = no trades (normal), quote fields never null
#
# ### Quote vs Trade Fields
#
# | Aspect | Quote Fields | Trade Fields |
# |--------|--------------|--------------|
# | Source | NBBO updates | Actual executions |
# | Null when | Never (carried forward) | No trades in bar |
# | Examples | open_bid_price, close_ask_price | first_trade_price, vwap |
#
# ### TAQ Field Families and Trading Mechanisms
#
# | Family | Key Fields | Trading Mechanism |
# |--------|------------|-------------------|
# | **Trade Buckets** | trade_at_bid/ask | Aggressor direction → OFI |
# | **Tick Direction** | uptick/downtick_volume | Trade-level momentum |
# | **Pressure** | trade_to_mid_vol_weight | Directional intensity |
# | **FINRA** | finra_volume | Institutional activity |
# | **Spread** | min/max_spread | Liquidity conditions |
#
# ### Key Microstructure Insights
#
# 1. **Order Flow Imbalance** (OFI): Measures aggressor direction (-1 to +1)
#    - Persistent OFI predicts short-term momentum
#    - Extreme OFI (>0.7 or <-0.7) signals potential reversal
# 2. **FINRA Share**: ~45% of volume is off-exchange (institutional activity)
# 3. **Spread Dynamics**: Wider at open/close, tightest midday
# 4. **Locked Markets**: min_spread=0 indicates stress (rare but important)
#
# ### Why AlgoSeek TAQ is Valuable
#
# - **Pre-computed from tick data**: Saves significant processing time
# - **Rich semantics**: Goes beyond OHLCV with aggressor, pressure, FINRA fields
# - **ML-ready**: Fields can be used directly as features or as building blocks
#
# ### Chapter 3 vs Chapter 8 Scope
#
# | This Notebook (Ch3) | Chapter 8 Notebooks |
# |---------------------|---------------------|
# | Understand field meanings | Engineer predictive alpha factors |
# | Data quality and semantics | Kyle's Lambda, Amihud, VPIN |
# | Trading mechanisms explained | Feature selection and orthogonalization |
#
# ### Next Steps
#
# - **Ch3 `11_algoseek_taq_eda`**: TAQ tick-level data exploration
# - **Ch8 `microstructure_features.py`**: Engineer alpha factors (Kyle's Lambda, VPIN)
# - **Ch9**: Evaluate feature predictive power
# - **Ch12**: Build ML models using microstructure features
