# ---
# jupyter:
#   jupytext:
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
# # Order Lifecycle Analysis: From Submission to Cancellation or Execution
#
# **Chapter 3: Market Microstructure**
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Track individual NASDAQ limit orders from submission to first termination
# event (delete, partial cancel, or replace) and to first execution, and
# quantify the resulting cancellation- and execution-rate distributions across
# the ITCH sample.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Reconstruct order lifecycles from add (`A`/`F`), delete (`D`), partial cancel
#   (`X`), replace (`U`), and execute (`E`/`C`) ITCH messages.
# - Measure how often an order ends in a cancellation and how often in a fill, and say
#   why those two rates do not sum to one.
# - Measure how long an order lives before each outcome, on a scale that can show
#   microseconds and hours in the same picture.
# - Separate an order still resting when the sample ends from one that filled and was
#   then deleted, so that neither outcome is counted twice.
#
# ## Book reference
#
# Section §3.3, *From raw messages to the limit order book* - the empirical findings on
# cancellation rates and on time to cancel and to execute.
#
# ## Prerequisites
#
# - Parsed ITCH message parquets at `data/equities/market/microstructure/nasdaq_itch/messages/`
#   (output of `01_itch_parser` or the Rust parser).
# - The last section reads the enriched `E` and `C` parquets that
#   `05_itch_trading_activity` writes, so run that notebook before this one if you want
#   it populated. Everything before it needs only the parsed messages.
#
# ---

# %% [markdown]
# ## What an order's life story is good for
#
# The book shows how much is resting at each price. It does not show how long any of it
# stays there, and that is what decides whether you can trade against it. An order that
# is quoted and withdrawn within a millisecond appears in the book and is not liquidity
# any slower participant can reach.
#
# Four things read directly off the lifecycle:
#
# | Question | What the lifecycle answers |
# |---|---|
# | Will my passive order fill? | The share of orders that ever execute, by size and side |
# | How often must a quote be refreshed? | The distribution of how long orders survive |
# | Is the flow informed? | Whether orders that fill were resting longer or shorter |
# | What will crossing the spread cost? | How much displayed size is still there on arrival |

# %% [markdown]
# ## Setup

# %%
"""Order Lifecycle Analysis — track limit orders from submission to cancellation or execution."""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
from itch_message_specs import MESSAGE_SPECS
from matplotlib.ticker import FuncFormatter

from data import load_nasdaq_itch
from utils.paths import display_path, get_output_dir
from utils.style import COLORS, show_with_alt

# %% [markdown]
# ### Declared parameters
#
# `SYMBOL` picks the one stock whose orders are followed individually. A trading day holds
# a few hundred million messages across the venue and a few million for one active name,
# so the per-order work is done on one symbol; the last section widens back out to the
# whole day with only two columns per message.
#
# `MAX_ORDERS` caps how many rows are read after the symbol filter. `None` reads them all,
# which is what the committed run does. A small cap makes a first pass quick on the same
# code path.

# %% tags=["parameters"]
SYMBOL = "AAPL"
MAX_ORDERS = None

# %%
NASDAQ_ITCH_OUTPUT = get_output_dir(3, "nasdaq_itch")
MESSAGE_DIR = load_nasdaq_itch(get_base_path=True)
OUTPUT_DIR = NASDAQ_ITCH_OUTPUT / "order_lifecycle"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input directory (messages):  {display_path(MESSAGE_DIR)}")
print(f"Output directory (analysis): {display_path(OUTPUT_DIR)}")
print(f"Symbol under analysis: {SYMBOL}")

# %%
if not MESSAGE_DIR.exists():
    print(f"Message directory not found: {display_path(MESSAGE_DIR)}")
    print("Run 01_itch_parser first to parse ITCH data.")
    available = []
else:
    available = sorted([d.name for d in MESSAGE_DIR.iterdir() if d.is_dir()])
    print(f"\nAvailable message types: {available}")

# Check if we have message data to analyze
HAS_MESSAGE_DATA = len(available) > 0

if not HAS_MESSAGE_DATA:
    raise RuntimeError(
        "No parsed ITCH message data found.\n"
        "This notebook requires output from 01_itch_parser (full parse ~60min).\n"
        f"Expected directory: {MESSAGE_DIR}"
    )

# D, X, E, C and U messages carry no ticker, only the numeric stock_locate the R (stock
# directory) messages assign, so the symbol has to be resolved to that number first.
r_dir = MESSAGE_DIR / "R"
if r_dir.exists():
    stock_directory = pl.scan_parquet(r_dir).collect()
    symbol_match = stock_directory.filter(pl.col("stock") == SYMBOL)
    if len(symbol_match) > 0:
        STOCK_LOCATE = symbol_match["stock_locate"][0]
        print(f"\nAnalyzing symbol: {SYMBOL} (stock_locate={STOCK_LOCATE})")
    else:
        raise ValueError(f"Symbol {SYMBOL} not found in R messages")
else:
    raise RuntimeError("R (Stock Directory) messages not found")


# %% [markdown]
# ## Loading one symbol's messages
#
# Functions to load different message types from Parquet files.


# %%
def _load_filtered_df(msg_dir, symbol, stock_locate, columns, limit):
    """Scan parquet, apply symbol/stock_locate filter, select columns, and collect."""
    lf = pl.scan_parquet(msg_dir / "*.parquet")
    schema = lf.collect_schema()

    # Filter before collecting so the predicate is pushed into the Parquet scan and only
    # the matching row groups are read.
    if "stock" in schema and symbol:
        lf = lf.filter(pl.col("stock") == symbol)
    elif "stock_locate" in schema and stock_locate is not None:
        lf = lf.filter(pl.col("stock_locate") == stock_locate)

    if columns:
        # Only select columns that exist
        available_cols = schema.names()
        valid_cols = [c for c in columns if c in available_cols]
        if valid_cols:
            lf = lf.select(valid_cols)

    # Apply row limit AFTER symbol filter
    if limit:
        lf = lf.head(limit)

    return lf.collect()


# %% [markdown]
# ### Post-Process Message DataFrame
# Validate timestamp dtypes and normalize ITCH price4 format after loading.


# %%
def _postprocess_message_df(df, msg_type):
    """Validate timestamp dtypes and normalize ITCH price4 format."""
    # Validate timestamp dtype - critical for correct timing calculations
    if "timestamp" in df.columns:
        ts_dtype = df.schema["timestamp"]
        if not isinstance(ts_dtype, pl.Datetime):
            # If timestamp is integer (nanoseconds since midnight), convert to duration
            if ts_dtype in (pl.Int64, pl.UInt64, pl.Int32, pl.UInt32):
                df = df.with_columns(
                    pl.duration(nanoseconds=pl.col("timestamp")).alias("timestamp")
                )
                print(
                    f"  Warning: {msg_type} timestamps converted from integer (assumed nanoseconds)"
                )

    # Normalize prices from ITCH price4 format (divide by 10000)
    # ITCH stores prices as integers with 4 implied decimal places
    price_cols = ["price", "execution_price"]
    for col in price_cols:
        if col in df.columns:
            # Heuristic: if median price > 10000, assume not yet normalized
            median_price = df.select(pl.col(col).median()).item()
            if median_price is not None and median_price > 10000:
                df = df.with_columns((pl.col(col) / 10000).alias(col))

    return df


# %% [markdown]
# ### Load Message Type
# Load a specific ITCH message type with optional symbol filtering and column selection.


# %%
def load_message_type(
    msg_type: str,
    columns: list | None = None,
    limit: int | None = None,
    symbol: str | None = None,
    stock_locate: int | None = None,
) -> pl.DataFrame:
    """Load a specific message type from Parquet files using Polars.

    Args:
        msg_type: Message type code (A, D, E, etc.)
        columns: Optional list of columns to select
        limit: Optional row limit (useful with large files)
        symbol: Filter by stock symbol (for A, F, P, R messages with 'stock' column)
        stock_locate: Filter by stock_locate ID (for D, E, X, C, U messages)

    Returns:
        DataFrame with requested columns. Timestamps are validated to be Datetime type.
        Empty DataFrame if message type directory doesn't exist.

    Note:
        ITCH message types have different columns:
        - A, F, P, R: Have 'stock' column - filter by symbol
        - D, E, X, C, U: Only have 'stock_locate' - filter by stock_locate ID
        Always filter by symbol/stock_locate to avoid loading all market data.
    """
    msg_dir = MESSAGE_DIR / msg_type
    if not msg_dir.exists():
        return pl.DataFrame()

    # Use MAX_ORDERS as default limit if not specified
    if limit is None:
        limit = MAX_ORDERS

    # Use global symbol/stock_locate if not specified
    if symbol is None:
        symbol = SYMBOL
    if stock_locate is None:
        stock_locate = STOCK_LOCATE

    try:
        df = _load_filtered_df(msg_dir, symbol, stock_locate, columns, limit)
        return _postprocess_message_df(df, msg_type)
    except Exception as e:
        print(f"Error loading {msg_type}: {e}")
        return pl.DataFrame()


# %% [markdown]
# ### Count Messages by Type
#
# Aggregate message counts across all Parquet partitions.


# %%
def count_messages() -> dict[str, int]:
    """Count rows per ITCH message type.

    The parser writes one directory per message type, and other notebooks write derived
    directories beside them, so only the single-letter names the ITCH specification
    defines are counted.
    """
    counts = {}
    for path in MESSAGE_DIR.iterdir():
        if not path.is_dir() or path.name not in MESSAGE_SPECS:
            continue
        try:
            dset = ds.dataset(path.as_posix(), format="parquet")
            counts[path.name] = sum(f.metadata.num_rows for f in dset.get_fragments())
        except (OSError, ValueError):
            continue
    return counts


# %% [markdown]
# ## 1. What the day is made of
#
# Before following a single order, it is worth seeing what the day's traffic consists of.
# The counts span many orders of magnitude - adds and deletes against a handful of
# session events - so the axis is logarithmic; a linear one would show two bars and
# seventeen invisible ones.

# %%
if HAS_MESSAGE_DATA:
    message_counts = count_messages()
    # Sort by count descending
    message_counts = dict(sorted(message_counts.items(), key=lambda x: -x[1]))

    labelled = {f"{code}  {MESSAGE_SPECS[code]['name']}": n for code, n in message_counts.items()}

    fig, ax = plt.subplots(figsize=(10, 6))
    pd.Series(labelled).sort_values().plot.barh(ax=ax, color=COLORS["blue"])
    ax.set_xscale("log")
    ax.set_title("Messages published per ITCH message type, one trading day")
    ax.set_xlabel("Messages (log scale)")
    ax.set_ylabel("")
    show_with_alt(
        fig,
        "A horizontal bar chart with one bar per ITCH message type, each labelled with its letter code and name, sorted with the most numerous type at the top and the rarest at the bottom. The horizontal axis counts messages on a logarithmic scale spanning roughly one to several hundred million, so the shortest bars are single-digit counts and the longest run the full width.",
    )

    print("Message counts:")
    for msg_type, count in message_counts.items():
        print(f"  {msg_type} {MESSAGE_SPECS[msg_type]['name']:<28}: {count:>12,}")

# %% [markdown]
# Read the two ends of that chart against each other. Orders arriving (`A`, `F`) and
# orders leaving without trading (`D`, `X`) sit at one end; executions (`E`, `C`) sit
# well below them. A venue publishes far more quoting than trading, and the sections
# below put a number on how much.

# %% [markdown]
# ## 2. What the orders look like
#
# Load limit order submissions (Add Order messages) to understand order characteristics.

# %%
# Load and combine Add Order messages
if HAS_MESSAGE_DATA:
    # Load Add Order messages (A = anonymous, F = with attribution)
    add_a = load_message_type("A")
    add_f = load_message_type("F")

    # Combine both types (use diagonal_relaxed for schema differences)
    if len(add_a) > 0 and len(add_f) > 0:
        limit_orders = pl.concat([add_a, add_f], how="diagonal_relaxed")
    elif len(add_a) > 0:
        limit_orders = add_a
    elif len(add_f) > 0:
        limit_orders = add_f
    else:
        limit_orders = pl.DataFrame()

    if len(limit_orders) > 0:
        # Standardize columns using rename
        rename_map = {
            "order_reference_number": "order",
            "buy_sell_indicator": "side",
            "timestamp": "submitted",
            "stock": "ticker",
        }
        for old, new in rename_map.items():
            if old in limit_orders.columns:
                limit_orders = limit_orders.rename({old: new})

        # Convert side to numeric using Polars when/then
        # Cast to string first to handle bytes/string mix
        if "side" in limit_orders.columns:
            limit_orders = limit_orders.with_columns(pl.col("side").cast(pl.Utf8).alias("side_str"))
            limit_orders = limit_orders.with_columns(
                pl.when(pl.col("side_str") == "B")
                .then(1)
                .when(pl.col("side_str") == "S")
                .then(-1)
                .otherwise(0)
                .alias("side_num")
            )

        print(f"Loaded {len(limit_orders):,} limit orders")
        print(limit_orders.schema)

# %%
# Price sanity check
if HAS_MESSAGE_DATA and len(limit_orders) > 0 and "price" in limit_orders.columns:
    price_stats = limit_orders.select(
        [
            pl.col("price").min().alias("min"),
            pl.col("price").median().alias("median"),
            pl.col("price").max().alias("max"),
        ]
    ).row(0, named=True)
    print("Price sanity check (should be plausible dollar values):")
    print(
        f"  Min: ${price_stats['min']:.2f}, Median: ${price_stats['median']:.2f}, Max: ${price_stats['max']:.2f}"
    )

    # Warning if prices look scaled (ITCH often uses 1/10000 scaling)
    if price_stats["median"] > 100000:
        print("  WARNING: Prices appear to be scaled integers, not dollars!")
        print("  Check that the ITCH parser normalizes prices correctly.")

# %% [markdown]
# ### How large a typical order is

# %%
if HAS_MESSAGE_DATA and len(limit_orders) > 0 and "shares" in limit_orders.columns:
    # Top order sizes
    order_sizes = (
        limit_orders.group_by(["shares", "side_num"])
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("proportion"))
        .sort("count", descending=True)
    )

    print("Top 10 Most Common Order Sizes:")
    top_10 = order_sizes.head(10)
    for row in top_10.iter_rows(named=True):
        print(
            f"  {row['shares']:>8,.0f} shares (side={row['side_num']:>2}): {row['proportion']:.2%}"
        )

    # Plot distribution - convert to pandas for pivot
    order_sizes_pd = order_sizes.to_pandas()
    size_by_side = order_sizes_pd.pivot_table(
        index="shares", columns="side_num", values="proportion", aggfunc="sum"
    ).rename(columns={1: "Buy", -1: "Sell"})

    fig, ax = plt.subplots(figsize=(10, 5))
    size_by_side.nlargest(15, "Buy").sort_values("Buy").plot.barh(
        ax=ax, color={"Buy": COLORS["positive"], "Sell": COLORS["negative"]}
    )
    ax.set_title(f"{SYMBOL}: the fifteen most common order sizes, by side")
    ax.set_xlabel("Share of orders")
    ax.set_ylabel("Order size (shares)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend(title="Side")
    show_with_alt(
        fig,
        "A horizontal bar chart with one pair of bars per order size, green for buys and red for sells, sorted so the most common size is at the top. The horizontal axis is the share of that side's orders, formatted as a percentage; the vertical axis lists the order sizes in shares.",
    )

# %% [markdown]
# Two things are worth reading off that chart. One size usually takes a large share on
# its own: a hundred shares is the round lot, the unit US equity trading was built
# around, and it is still what an order defaults to. Below it the sizes are a mixture of
# round lots and odd numbers, and odd lots are ordinary rather than exceptional - a
# reminder that the visible book is not made of neat blocks.
#
# Compare the green and red bars at each size rather than reading either alone. Where the
# two sides use the same sizes, order size is a convention rather than a signal.

# %% [markdown]
# ### Where the dollars are

# %%
if (
    HAS_MESSAGE_DATA
    and len(limit_orders) > 0
    and "price" in limit_orders.columns
    and "shares" in limit_orders.columns
):
    # Calculate dollar value per order
    limit_orders = limit_orders.with_columns(
        (pl.col("shares").cast(pl.Float64) * pl.col("price").cast(pl.Float64)).alias("value")
    )

    # Value contribution by order size
    value_by_size = limit_orders.group_by(["shares", "side_num"]).agg(
        pl.col("value").sum().alias("total_value")
    )

    total_value = value_by_size.select(pl.col("total_value").sum()).item()
    value_by_size = value_by_size.with_columns(
        (pl.col("total_value") / total_value).alias("value_share")
    )

    # Convert to pandas for pivot plot
    value_by_size_pd = value_by_size.to_pandas()
    value_pivot = value_by_size_pd.pivot_table(
        index="shares", columns="side_num", values="value_share", aggfunc="sum"
    ).rename(columns={1: "Buy", -1: "Sell"})

    fig, ax = plt.subplots(figsize=(10, 5))
    value_pivot.nlargest(10, "Buy").sort_values("Buy").plot.barh(
        ax=ax, color={"Buy": COLORS["positive"], "Sell": COLORS["negative"]}
    )
    ax.set_title(f"{SYMBOL}: share of submitted dollar value by order size")
    ax.set_xlabel("Share of total dollar value")
    ax.set_ylabel("Order size (shares)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend(title="Side")
    show_with_alt(
        fig,
        "A horizontal bar chart with one pair of bars per order size, green for buys and red for sells, showing each size's share of the total dollar value submitted rather than its share of the order count. Sizes run down the vertical axis and the horizontal axis is a percentage.",
    )

# %% [markdown]
# ## 3. Orders that leave without trading
#
# Modern markets feature extremely high cancellation rates. Let's quantify this.
#
# ### Methodology Note
#
# ITCH has multiple termination message types:
# - **D (Delete)**: Full order cancellation - liquidity withdrawn
# - **X (Cancel)**: Partial cancellation - reduces order size
# - **U (Replace)**: Cancels original, creates new order - quote update
#
# For this analysis:
# - We track **time to first termination event** per order (not all events)
# - Replace (U) terminates the original order but creates a replacement
# - Orders can have multiple events (e.g., partial cancel then delete)
# - We use **first event timestamp** for timing statistics

# %%
# Load termination messages (Delete, Cancel, Replace)
orders_with_cancel = None
orders_with_exec = None
cancelled_orders_df = None
executed_orders_df = None
termination_by_type = None

if HAS_MESSAGE_DATA and len(limit_orders) > 0:
    delete_msgs = load_message_type("D", columns=["order_reference_number", "timestamp"])
    cancel_msgs = load_message_type("X", columns=["order_reference_number", "timestamp"])
    replace_msgs = load_message_type("U", columns=["original_order_reference_number", "timestamp"])

    # Normalize each termination type
    termination_events = []

    if len(delete_msgs) > 0:
        delete_msgs = delete_msgs.rename(
            {"order_reference_number": "order", "timestamp": "terminated"}
        ).with_columns(pl.lit("delete").alias("termination_type"))
        termination_events.append(delete_msgs.select(["order", "terminated", "termination_type"]))

    if len(cancel_msgs) > 0:
        cancel_msgs = cancel_msgs.rename(
            {"order_reference_number": "order", "timestamp": "terminated"}
        ).with_columns(pl.lit("cancel").alias("termination_type"))
        termination_events.append(cancel_msgs.select(["order", "terminated", "termination_type"]))

    if len(replace_msgs) > 0:
        replace_msgs = replace_msgs.rename(
            {"original_order_reference_number": "order", "timestamp": "terminated"}
        ).with_columns(pl.lit("replace").alias("termination_type"))
        termination_events.append(replace_msgs.select(["order", "terminated", "termination_type"]))

# %%
# Deduplicate to first termination event per order and merge with limit orders
if HAS_MESSAGE_DATA and len(limit_orders) > 0 and termination_events:
    all_terminations = pl.concat(termination_events)
    print(f"Total termination events: {len(all_terminations):,}")

    # An order can be named by several termination messages; keep the earliest.
    # Counting every event would weight an order by how often it was touched.
    first_termination = (
        all_terminations.sort("terminated")
        .group_by("order")
        .agg(
            [
                pl.col("terminated").first().alias("cancelled"),
                pl.col("termination_type").first().alias("termination_type"),
            ]
        )
    )

    print(f"Unique orders with termination: {len(first_termination):,}")

    # Track termination breakdown
    termination_by_type = (
        first_termination.group_by("termination_type")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    print("\nTermination breakdown (first event per order):")
    for row in termination_by_type.iter_rows(named=True):
        print(f"  {row['termination_type']:>10}: {row['count']:>12,}")

    # Merge with limit orders to get full lifecycle
    orders_with_cancel = limit_orders.join(first_termination, on="order", how="left")
    print(f"\nOrders with termination data: {len(orders_with_cancel):,}")

# %% [markdown]
# ### How often an order is terminated

# %%
if HAS_MESSAGE_DATA and orders_with_cancel is not None and len(orders_with_cancel) > 0:
    # Calculate termination rate (using first event per order)
    total_orders = orders_with_cancel.select(pl.col("order").n_unique()).item()
    terminated_orders = (
        orders_with_cancel.filter(pl.col("cancelled").is_not_null())
        .select(pl.col("order").n_unique())
        .item()
    )

    termination_rate = terminated_orders / total_orders
    remaining = total_orders - terminated_orders

    print("=" * 50)
    print("ORDER TERMINATION ANALYSIS")
    print("=" * 50)
    print(f"Total unique orders:     {total_orders:>12,}")
    print(f"Orders terminated:       {terminated_orders:>12,}")
    print("  (delete/cancel/replace)")
    print(f"Still live or unknown:   {remaining:>12,}")
    print(f"Termination rate:        {termination_rate:>12.1%}")
    print("=" * 50)

    print("'Terminated' counts the first Delete, Cancel or Replace to name the order.")
    print(
        "'Still live or unknown' holds orders with no termination message in the sample, "
        "including ones that executed; Section 5 separates those."
    )

# %% [markdown]
# Read that rate carefully, because it counts three different events as one outcome:
# `D` withdraws the order entirely, `X` reduces its size, and `U` moves it to another
# price or size by retiring it and issuing a new reference. Only the first is a
# cancellation in the everyday sense.
#
# It is also not the same quantity as "never traded". An order can fill part of its size
# and then be deleted, and it appears in both this rate and the execution rate below.
# Section 5 puts each order in exactly one category.
#
# What a high termination rate describes is a market where quoting is cheap and
# continuous: a resting order is a standing offer that its sender re-prices whenever
# anything it depends on moves, and re-pricing on this feed means retiring one order and
# sending another.

# %% [markdown]
# ### How long an order lives before it is terminated
#
# Termination here is the first `D`, `X` or `U` to name the order, and an order that was
# partly filled before that still counts.
#
# The lifetime is computed from the nanosecond difference rather than from a seconds
# helper, because a large share of these orders live for less than one second and a
# whole-second duration would record all of them as zero. That matters for what follows:
# the median, the quantile table and both histograms are about the sub-second end of the
# distribution, and a truncated duration erases exactly that end.


# %%
# Compute time-to-cancellation statistics
if HAS_MESSAGE_DATA and orders_with_cancel is not None:
    cancelled_orders_df = orders_with_cancel.filter(pl.col("cancelled").is_not_null())

    if len(cancelled_orders_df) > 0:
        cancelled_orders_df = cancelled_orders_df.with_columns(
            ((pl.col("cancelled") - pl.col("submitted")).dt.total_nanoseconds() / 1e9).alias(
                "cancel_time"
            )
        )

        cancel_time_stats = cancelled_orders_df.select(
            [
                pl.col("cancel_time").count().alias("count"),
                pl.col("cancel_time").mean().alias("mean"),
                pl.col("cancel_time").std().alias("std"),
                pl.col("cancel_time").min().alias("min"),
                pl.col("cancel_time").quantile(0.1).alias("10%"),
                pl.col("cancel_time").quantile(0.25).alias("25%"),
                pl.col("cancel_time").quantile(0.5).alias("50%"),
                pl.col("cancel_time").quantile(0.75).alias("75%"),
                pl.col("cancel_time").quantile(0.9).alias("90%"),
                pl.col("cancel_time").quantile(0.95).alias("95%"),
                pl.col("cancel_time").quantile(0.99).alias("99%"),
                pl.col("cancel_time").max().alias("max"),
            ]
        ).row(0, named=True)

        print("Time to Cancellation (seconds):")
        print("-" * 40)
        for stat, value in cancel_time_stats.items():
            print(f"{stat:>10}: {value:>12.4f}")

# %% [markdown]
# Order lifetimes span from microseconds to the length of a session, so the histogram is
# built on a logarithmic time axis. On a linear one every bar but the first would be
# empty, and the first would hide the entire shape.

# %%
if HAS_MESSAGE_DATA and cancelled_orders_df is not None and len(cancelled_orders_df) > 0:
    lifetimes = cancelled_orders_df.filter(pl.col("cancel_time") > 0)["cancel_time"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(lifetimes):
        edges = np.logspace(np.log10(lifetimes.min()), np.log10(lifetimes.max()), 41)
        ax.hist(lifetimes, bins=edges, edgecolor="none", alpha=0.8, color=COLORS["slate"])
        ax.set_xscale("log")
        median_life = float(np.median(lifetimes))
        ax.axvline(
            median_life,
            color="red",
            linestyle="--",
            label=f"Median of the plotted orders: {median_life:.4g} s",
        )
        ax.legend()
    ax.set_title(f"{SYMBOL}: time from submission to first termination")
    ax.set_xlabel("Seconds (log scale)")
    ax.set_ylabel("Number of orders")
    show_with_alt(
        fig,
        "A histogram of order lifetimes on a logarithmic horizontal axis running from well under a millisecond to the length of a trading session. A dashed vertical line marks the median of the plotted orders and is labelled with its value in seconds. The vertical axis counts orders.",
    )

    n_cancelled = len(cancelled_orders_df)
    for label, cutoff in (("1 millisecond", 0.001), ("1 second", 1.0), ("10 seconds", 10.0)):
        share = cancelled_orders_df.filter(pl.col("cancel_time") < cutoff).height / n_cancelled
        print(f"Terminated within {label}: {share:.1%}")
    zero_length = cancelled_orders_df.filter(pl.col("cancel_time") <= 0).height
    print(
        f"Orders with a non-positive lifetime (excluded from the log-scaled figure): "
        f"{zero_length:,} of {n_cancelled:,}"
    )

# %% [markdown]
# ## 4. The orders that trade
#
# Two message types report a fill. `E` executes shares at the order's displayed price;
# `C` executes at a different one, which is how a hidden or price-improved fill is
# reported. Both name an order, and one order can be named several times as it fills in
# pieces, so the two streams are combined and reduced to the first fill per order.
#
# Taking the first fill is what makes 'time to execution' a well-defined quantity here.
# It is the wait before an order started trading, not the time it took to complete.

# %%
if HAS_MESSAGE_DATA and len(limit_orders) > 0:
    exec_e = load_message_type("E", columns=["order_reference_number", "timestamp"])
    exec_c = load_message_type("C", columns=["order_reference_number", "timestamp"])

    executions = []
    if len(exec_e) > 0:
        exec_e = exec_e.rename({"order_reference_number": "order", "timestamp": "executed"})
        executions.append(exec_e.select(["order", "executed"]))

    if len(exec_c) > 0:
        exec_c = exec_c.rename({"order_reference_number": "order", "timestamp": "executed"})
        executions.append(exec_c.select(["order", "executed"]))

    if executions:
        all_executions = pl.concat(executions)

        print(f"Total execution events: {len(all_executions):,}")

        # One order can fill in several pieces; keep the earliest.
        first_execution = (
            all_executions.sort("executed")
            .group_by("order")
            .agg(pl.col("executed").first().alias("executed"))
        )

        print(f"Unique orders with execution: {len(first_execution):,}")

        # Merge with limit orders using deduplicated executions
        orders_with_exec = limit_orders.join(first_execution, on="order", how="left")

# %% [markdown]
# ### How often an order fills

# %%
if HAS_MESSAGE_DATA and orders_with_exec is not None and len(orders_with_exec) > 0:
    # Calculate execution rate
    total_orders = orders_with_exec.select(pl.col("order").n_unique()).item()
    executed_orders = (
        orders_with_exec.filter(pl.col("executed").is_not_null())
        .select(pl.col("order").n_unique())
        .item()
    )

    exec_rate = executed_orders / total_orders

    print("=" * 50)
    print("EXECUTION RATE ANALYSIS")
    print("=" * 50)
    print(f"Total unique orders:     {total_orders:>12,}")
    print(f"Orders executed:         {executed_orders:>12,}")
    print(f"Execution rate:          {exec_rate:>12.1%}")
    print("=" * 50)

# %% [markdown]
# ### How long a fill takes to arrive

# %%
if HAS_MESSAGE_DATA and orders_with_exec is not None:
    # Calculate time from submission to execution
    executed_orders_df = orders_with_exec.filter(pl.col("executed").is_not_null())

    if len(executed_orders_df) > 0:
        executed_orders_df = executed_orders_df.with_columns(
            ((pl.col("executed") - pl.col("submitted")).dt.total_nanoseconds() / 1e9).alias(
                "exec_time"
            )
        )

        # Summary statistics
        exec_time_stats = executed_orders_df.select(
            [
                pl.col("exec_time").count().alias("count"),
                pl.col("exec_time").mean().alias("mean"),
                pl.col("exec_time").std().alias("std"),
                pl.col("exec_time").min().alias("min"),
                pl.col("exec_time").quantile(0.1).alias("10%"),
                pl.col("exec_time").quantile(0.25).alias("25%"),
                pl.col("exec_time").quantile(0.5).alias("50%"),
                pl.col("exec_time").quantile(0.75).alias("75%"),
                pl.col("exec_time").quantile(0.9).alias("90%"),
                pl.col("exec_time").quantile(0.95).alias("95%"),
                pl.col("exec_time").quantile(0.99).alias("99%"),
                pl.col("exec_time").max().alias("max"),
            ]
        ).row(0, named=True)

        print("Time to Execution (seconds):")
        print("-" * 40)
        for stat, value in exec_time_stats.items():
            print(f"{stat:>10}: {value:>12.4f}")

# %% [markdown]
# ## 5. One outcome per order
#
# **Important**: Execution and termination are not mutually exclusive!
# An order can be partially executed and then cancelled.
# Here we build a unified view of order outcomes.

# %%
unified = None

if (
    HAS_MESSAGE_DATA
    and orders_with_cancel is not None
    and orders_with_exec is not None
    and len(limit_orders) > 0
):
    # Build unified outcomes table
    # Start with all orders
    unified = limit_orders.select(["order"]).unique()

    # Add execution flag
    executed_orders_set = (
        orders_with_exec.filter(pl.col("executed").is_not_null()).select("order").unique()
    )
    # implode makes the Series one collection to test membership in. Passing it bare is
    # ambiguous in polars and prints a deprecation into the render.
    unified = unified.with_columns(
        pl.col("order").is_in(executed_orders_set["order"].implode()).alias("has_execution")
    )

    # Add termination flag
    terminated_orders_set = (
        orders_with_cancel.filter(pl.col("cancelled").is_not_null()).select("order").unique()
    )
    unified = unified.with_columns(
        pl.col("order").is_in(terminated_orders_set["order"].implode()).alias("has_termination")
    )

    # Create mutually exclusive outcome categories
    unified = unified.with_columns(
        pl.when(pl.col("has_execution") & pl.col("has_termination"))
        .then(pl.lit("Executed then Terminated"))
        .when(pl.col("has_execution") & ~pl.col("has_termination"))
        .then(pl.lit("Executed Only"))
        .when(~pl.col("has_execution") & pl.col("has_termination"))
        .then(pl.lit("Terminated Only"))
        .otherwise(pl.lit("Still Live/Unknown"))
        .alias("outcome")
    )

# %%
if unified is not None:
    # Compute outcome statistics
    outcome_stats = (
        unified.group_by("outcome")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("rate"))
        .sort("count", descending=True)
    )

    print("=" * 60)
    print("UNIFIED ORDER OUTCOME CLASSIFICATION")
    print("=" * 60)
    total_unified = unified.shape[0]
    print(f"Total unique orders: {total_unified:,}")
    print("-" * 60)
    for row in outcome_stats.iter_rows(named=True):
        print(f"  {row['outcome']:<25}: {row['count']:>10,} ({row['rate']:>6.1%})")
    print("=" * 60)
    print("\nNote: 'Executed then Terminated' means partial fill then cancel/delete.")
    print("This reconciles execution and termination rates into exclusive categories.")

# %% [markdown]
# ## 6. Which happens sooner, a withdrawal or a fill

# %%
if (
    HAS_MESSAGE_DATA
    and cancelled_orders_df is not None
    and len(cancelled_orders_df) > 0
    and executed_orders_df is not None
    and len(executed_orders_df) > 0
):
    # Create comparison table
    print("Time Comparison: Cancellations vs. Executions (seconds)")
    print("=" * 60)
    print(f"{'Statistic':<15} {'Cancellations':>15} {'Executions':>15}")
    print("-" * 60)
    for stat in ["10%", "25%", "50%", "75%", "90%"]:
        cancel_val = cancel_time_stats.get(stat, 0)
        exec_val = exec_time_stats.get(stat, 0)
        print(f"{stat:<15} {cancel_val:>15.4f} {exec_val:>15.4f}")

# %%
if (
    HAS_MESSAGE_DATA
    and cancelled_orders_df is not None
    and len(cancelled_orders_df) > 0
    and executed_orders_df is not None
    and len(executed_orders_df) > 0
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, frame, column, colour, label in (
        (
            axes[0],
            cancelled_orders_df,
            "cancel_time",
            COLORS["negative"],
            "termination",
        ),
        (
            axes[1],
            executed_orders_df,
            "exec_time",
            COLORS["positive"],
            "execution",
        ),
    ):
        values = frame.filter(pl.col(column) > 0)[column].to_numpy()
        if len(values):
            edges = np.logspace(np.log10(values.min()), np.log10(values.max()), 41)
            ax.hist(values, bins=edges, alpha=0.8, color=colour, edgecolor="none")
            ax.set_xscale("log")
        ax.set_title(f"Time from submission to {label}")
        ax.set_xlabel("Seconds (log scale)")
        ax.set_ylabel("Number of orders")

    show_with_alt(
        fig,
        "Two histograms side by side, each on a logarithmic seconds axis running from well under a millisecond to the length of a session. The left panel, in red, counts orders by how long they lived before being terminated; the right, in green, counts orders by how long they lived before their first execution. The two panels have independent vertical scales.",
    )

# %% [markdown]
# ## 7. What the fills looked like
#
# An `E` or `C` message names an order and a quantity, not a side or a limit price. To
# say which way a fill went, or how its price compared with what the sender asked for,
# the execution has to be joined back to the add that created the order.
# `05_itch_trading_activity` does that join once and writes the result, so this section
# reads its output rather than repeating the work. Without it the section prints nothing
# and the rest of the notebook is unaffected.

# %%
# Filter on stock_locate inside the scan: the enriched files cover the whole venue.
ENRICHED_DIR = NASDAQ_ITCH_OUTPUT / "enriched"

if ENRICHED_DIR.exists():
    enriched_e = None
    enriched_c = None

    e_file = ENRICHED_DIR / "E.parquet"
    c_file = ENRICHED_DIR / "C.parquet"

    # Apply stock_locate filter to avoid OOM on full-day data
    if e_file.exists():
        lf = pl.scan_parquet(e_file)
        if "stock_locate" in lf.collect_schema().names():
            lf = lf.filter(pl.col("stock_locate") == STOCK_LOCATE)
        enriched_e = lf.collect()
        print(f"Loaded {len(enriched_e):,} enriched E messages for {SYMBOL}")

    if c_file.exists():
        lf = pl.scan_parquet(c_file)
        if "stock_locate" in lf.collect_schema().names():
            lf = lf.filter(pl.col("stock_locate") == STOCK_LOCATE)
        enriched_c = lf.collect()
        print(f"Loaded {len(enriched_c):,} enriched C messages for {SYMBOL}")
else:
    print(f"Enriched data not found at {display_path(ENRICHED_DIR)}")
    print("Run 05_itch_trading_activity first to generate the enriched E/C/X parquets.")
    enriched_e = None
    enriched_c = None

# %%
# Execution by side (Buy vs Sell)
if enriched_e is not None and "side" in enriched_e.columns:
    side_summary = (
        enriched_e.filter(pl.col("side").is_not_null())
        .group_by("side")
        .agg(
            pl.col("executed_shares").sum().alias("total_shares"),
            pl.len().alias("execution_count"),
        )
    )

    print("\n=== Execution Summary by Side (E messages) ===")
    print(side_summary)

    if len(side_summary) > 0:
        total_exec = side_summary.select(pl.col("execution_count").sum()).item()
        buy_rows = side_summary.filter(pl.col("side") == "B")
        sell_rows = side_summary.filter(pl.col("side") == "S")
        buy_count = buy_rows["execution_count"].item() if len(buy_rows) > 0 else 0
        sell_count = sell_rows["execution_count"].item() if len(sell_rows) > 0 else 0

        print(f"\nBuy executions:  {buy_count:>10,} ({buy_count / total_exec:.1%})")
        print(f"Sell executions: {sell_count:>10,} ({sell_count / total_exec:.1%})")
    else:
        print("  No executions with side information available.")

# %%
# Price improvement analysis (C messages only)
if enriched_c is not None and "price_improvement_raw" in enriched_c.columns:
    # Filter to valid price improvement data
    valid_pi = enriched_c.filter(
        pl.col("price_improvement_raw").is_not_null() & pl.col("side").is_not_null()
    )

    if len(valid_pi) > 0:
        print("\n=== Price Improvement Analysis (C messages) ===")
        print("Note: C messages are executions at a DIFFERENT price than the order's limit.")
        print("Typically for hidden/iceberg orders.")

        # Analyze by side
        # For buys: negative price_improvement_raw = better (paid less than limit)
        # For sells: positive price_improvement_raw = better (received more than limit)
        for side_label, side_code, better_sign in [
            ("Buy", "B", "negative"),
            ("Sell", "S", "positive"),
        ]:
            side_data = valid_pi.filter(pl.col("side") == side_code)
            if len(side_data) == 0:
                continue

            pi = side_data["price_improvement_raw"]
            mean_pi = pi.mean()
            median_pi = pi.median()

            # Convert from price units (4 decimals) to dollars
            mean_pi_dollars = mean_pi / 10000
            median_pi_dollars = median_pi / 10000

            # Count improved vs worse
            if side_code == "B":
                improved = side_data.filter(pl.col("price_improvement_raw") < 0).height
                worse = side_data.filter(pl.col("price_improvement_raw") > 0).height
            else:
                improved = side_data.filter(pl.col("price_improvement_raw") > 0).height
                worse = side_data.filter(pl.col("price_improvement_raw") < 0).height
            same = side_data.filter(pl.col("price_improvement_raw") == 0).height

            print(f"\n{side_label} Orders ({len(side_data):,} executions):")
            print(f"  Mean price diff:   ${mean_pi_dollars:>8.4f}")
            print(f"  Median price diff: ${median_pi_dollars:>8.4f}")
            print(f"  Better than limit: {improved:>8,} ({improved / len(side_data):.1%})")
            print(f"  At limit price:    {same:>8,} ({same / len(side_data):.1%})")
            print(f"  Worse than limit:  {worse:>8,} ({worse / len(side_data):.1%})")

# %% [markdown]
# ### How to read those two tables
#
# The side counts say whether the day's fills were one-directional. Counts and share
# volume can disagree: a balanced count with lopsided volume means the two sides were
# trading in different sizes, which is what directional flow looks like when it is
# worked in pieces.
#
# The price-improvement table compares each `C` fill against the limit its order carried.
# `C` exists precisely for fills that print away from the displayed price, so the
# interesting question is not whether the difference is non-zero but whether it favours
# the sender: better than the limit for a buy means paying less, for a sell receiving
# more. The table splits the fills three ways - better, at, and worse - so a mixture and
# a uniform offset can be told apart, which a mean alone cannot do.

# %% [markdown]
# ## 8. Two narrower timings across the whole venue
#
# Everything so far followed one symbol. Two durations can be computed for every order the
# venue saw that day, because only three columns are needed - the order reference and two
# timestamps - and a scan that reads nothing else fits in memory.
#
# They are not the same two quantities as above, and the difference matters when reading
# them side by side. This pass defines the end of an order's life as its first `D`
# delete, where the per-symbol section counted the first `D`, `X` or `U`; and it defines
# the first fill as the first `E`, where the per-symbol section also counted `C`. So an
# order that was replaced rather than deleted, or filled first by a `C`, is either absent
# here or timed from a later event. Read these as time to deletion and time to first `E`
# fill, over every symbol; the per-symbol numbers cover more event types over one.
#
# The narrowing is deliberate: `D` and `E` are the two largest message types on the feed,
# and restricting to them is what lets the pass read a few hundred million orders without
# a symbol filter.

# %%
if HAS_MESSAGE_DATA:
    print("Loading venue-wide order data (A+F, D, E messages)...")
    started = perf_counter()

    # All Add orders (A + F) - no symbol filter. Section 2 above already builds from
    # whichever of the two the store holds, and this pass has to agree with it: a full
    # session carries both, a partial store need not.
    venue_add_scans = [
        pl.scan_parquet(MESSAGE_DIR / code / "*.parquet").select(
            pl.col("order_reference_number").alias("order"),
            pl.col("timestamp").alias("submitted"),
        )
        for code in ("A", "F")
        if (MESSAGE_DIR / code).is_dir() and any((MESSAGE_DIR / code).glob("*.parquet"))
    ]
    assert venue_add_scans, f"No A or F add messages under {MESSAGE_DIR}"
    venue_adds = pl.concat(venue_add_scans)
    venue_add_count = venue_adds.select(pl.len()).collect().item()
    print(f"Total Add orders (A+F): {venue_add_count:,} ({perf_counter() - started:.1f}s)")

    # --- Time to Cancellation ---
    print("\nComputing time-to-cancellation...")
    cancel_started = perf_counter()
    venue_deletes = (
        pl.scan_parquet(MESSAGE_DIR / "D" / "*.parquet")
        .select(
            pl.col("order_reference_number").alias("order"),
            pl.col("timestamp").alias("deleted"),
        )
        .group_by("order")
        .agg(pl.col("deleted").min())
    )
    venue_cancel_times = (
        venue_adds.join(venue_deletes, on="order", how="inner")
        .with_columns(
            (pl.col("deleted") - pl.col("submitted")).dt.total_nanoseconds().alias("cancel_ns")
        )
        .select("cancel_ns")
        .collect()
    )
    cancel_seconds = venue_cancel_times["cancel_ns"].cast(pl.Float64) / 1e9
    venue_cancelled = len(venue_cancel_times)
    print(
        f"Orders with delete events: {venue_cancelled:,} ({perf_counter() - cancel_started:.1f}s)"
    )

    # An empty series means no add in this store was ever deleted, and every share below
    # would be a proportion of nothing: polars returns None for the mean, which formats
    # as a crash rather than as the absence it is.
    if venue_cancelled:
        print(f"\nTime to deletion (D only), venue-wide ({venue_cancelled:,} orders):")
        print(f"  Within 500 ms:     {(cancel_seconds < 0.5).mean():.1%}")
        print(f"  Within 1 second:   {(cancel_seconds < 1.0).mean():.1%}")
        print(f"  Within 10 seconds: {(cancel_seconds < 10.0).mean():.1%}")
        print(f"  Median:            {cancel_seconds.median():.3f} s")
    else:
        print("\nNo add in this store was deleted, so there is no time to deletion.")
    del venue_cancel_times, cancel_seconds

    # --- Time to Execution ---
    print("\nComputing time-to-execution...")
    exec_started = perf_counter()
    venue_execs = (
        pl.scan_parquet(MESSAGE_DIR / "E" / "*.parquet")
        .select(
            pl.col("order_reference_number").alias("order"),
            pl.col("timestamp").alias("executed"),
        )
        .group_by("order")
        .agg(pl.col("executed").min())
    )
    venue_exec_times = (
        venue_adds.join(venue_execs, on="order", how="inner")
        .with_columns(
            (pl.col("executed") - pl.col("submitted")).dt.total_nanoseconds().alias("exec_ns")
        )
        .select("exec_ns")
        .collect()
    )
    exec_seconds = venue_exec_times["exec_ns"].cast(pl.Float64) / 1e9
    venue_executed = len(venue_exec_times)
    print(
        f"Orders with execution events: {venue_executed:,} ({perf_counter() - exec_started:.1f}s)"
    )

    if venue_executed:
        print(f"\nTime to first E fill, venue-wide ({venue_executed:,} orders):")
        print(f"  Within 1 millisecond: {(exec_seconds < 0.001).mean():.1%}")
        print(f"  Median:               {exec_seconds.median():.3f} s")
        print(f"  Over 40 minutes:      {(exec_seconds > 2400).mean():.1%}")
    else:
        print("\nNo add in this store was executed, so there is no time to first fill.")
    del venue_exec_times, exec_seconds

    print(f"\nTotal runtime: {perf_counter() - started:.1f}s")

# %% [markdown]
# Compare the two lists above against each other rather than reading either alone. Orders
# that are deleted are deleted quickly; orders that fill take longer, and a tail of them
# waits for most of the session. That ordering is the mechanism at work: an order is
# withdrawn as soon as the price it was written against moves, and it fills only when
# someone chooses to cross to it, which is not something its sender controls.
#
# The two populations overlap, because an order can fill part of its size and be deleted
# afterwards, so these are two views of one day rather than two disjoint sets. Both are
# also conditional on their event having happened: an order that was never deleted is
# absent from the first list, and an order that never filled from the second. An order
# that took a partial fill and was still resting at the close is in the second and not
# the first.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Termination and execution are not complements.** An order can fill part of its
#    size and then be withdrawn, so the two rates overlap and do not sum to one. Any
#    statement about how many orders 'never traded' has to come from the unified
#    classification, not from one minus the other.
# 2. **A replace is not a cancellation.** `U` retires a reference and issues a new one,
#    which is what re-pricing a quote looks like on this feed. Counting it as a
#    withdrawal overstates how much liquidity actually left the book.
# 3. **Compute durations from nanoseconds.** A large share of these orders live for less
#    than a second, so a whole-second duration records them as zero and erases the part
#    of the distribution the analysis is about.
# 4. **Plot lifetimes on a log axis.** They span microseconds to hours; on a linear axis
#    the entire distribution lands in the first bin.
# 5. **One symbol is not the venue, and the two passes measure different events.** The
#    per-order work runs on one name because it has to, over `D`, `X`, `U`, `E` and `C`;
#    the venue-wide pass reads three columns over every order but only `D` and `E`. They
#    are reported separately, and a difference between them is partly population and
#    partly definition.
#
# ### Known limitations
#
# - One venue and one session. NASDAQ-routed orders only, on a single day.
# - Every outcome here is one observed inside the session. An order still on the book at
#   the close has not been seen to terminate, which is not the same as not terminating.
#   And resting is not exclusive of either population: a partially filled order still
#   rests and contributes its first fill to the execution timing, and a partial cancel
#   (`X`) counts as a termination here while leaving the remainder on the book.
# - Time to execution is time to the *first* fill. An order filled in several pieces
#   contributes the first one, so this is not how long an order took to complete.
# - Hidden orders never appear as adds, so nothing here describes their lifecycle.
#
# ### Next Steps
#
# - **Notebook 05**: `itch_trading_activity` - Message counts and volume concentration
# - **Notebook 06**: `itch_intraday_patterns` - U-shape in volume and volatility
# - **Chapter 8**: Build predictive features from order flow data

# %% [markdown]
# ---
#
# ## References
#
# - Harris, L. (2003). *Trading and Exchanges: Market Microstructure for Practitioners*.
#   Oxford University Press.
# - Bouchaud, J.-P., Bonart, J., Donier, J., & Gould, M. (2018).
#   *Trades, Quotes and Prices: Financial Markets Under the Microscope*.
#   Cambridge University Press.
#   [https://doi.org/10.1017/9781009028943](https://doi.org/10.1017/9781009028943)
