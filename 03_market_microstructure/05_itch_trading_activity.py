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
# # Trading Activity Overview: NASDAQ Market Structure
#
# **Chapter 3: Market Microstructure**
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Provide the chapter's market-wide ITCH context: aggregate message-type
# composition, dollar-volume concentration across tickers, and the canonical
# enriched-trade parquet that `06_itch_intraday_patterns` and
# `07_itch_stylized_facts` reuse.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Tally ITCH message types across a multi-day sample and place add/cancel/
#   execute frequencies in proportion.
# - Build a stock-attributed trade table by joining `E`/`C`/`X` messages to the
#   `R`-based `stock_locate → stock` directory and to `A`/`F` add-order
#   attributes (with `U` replace lineage).
# - Quantify dollar-volume concentration (Pareto-style) over the universe.
#
# ## Book reference
#
# Section §3.3, *From raw messages to the limit order book* - the market-wide statistics
# paragraph cites this notebook.
#
# ## Prerequisites
#
# - Parsed ITCH message parquets at `data/equities/market/microstructure/nasdaq_itch/messages/`
#   (output of `01_itch_parser` or the Rust parser).
# - Notebooks 06 and 07 read the canonical trade table this notebook writes
#   under `03_market_microstructure/output/nasdaq_itch/trading_activity/`.
#
# ---

# %% [markdown]
# ## 1. Setup and Configuration

# %%
"""Trading Activity Overview — high-level view of NASDAQ trading activity using TotalView-ITCH data."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
from IPython.display import display
from itch_message_specs import MESSAGE_SPECS

from data import load_nasdaq_itch
from utils.paths import display_path, get_output_dir
from utils.style import COLORS, show_with_alt

# %% [markdown]
# ### Declared parameters
#
# `MAX_ROWS` caps how many rows of each message type are read; zero means read them all,
# which is what the committed run does. A small cap exercises the same code in seconds.
#
# `REBUILD_ENRICHED` decides whether the enrichment below is recomputed. The enriched
# files are large and take minutes to build, so a reader who already has them can skip
# the work; the committed run rebuilds them, because a result read from a cache is not a
# result the notebook reproduced.

# %% tags=["parameters"]
MAX_ROWS = 0
REBUILD_ENRICHED = True

# %%
NASDAQ_ITCH_OUTPUT = get_output_dir(3, "nasdaq_itch")
MESSAGE_DIR = load_nasdaq_itch(get_base_path=True)
OUTPUT_DIR = NASDAQ_ITCH_OUTPUT / "trading_activity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ENRICHED_DIR = NASDAQ_ITCH_OUTPUT / "enriched"

ROW_LIMIT = MAX_ROWS or None

print(f"Input directory (messages):  {display_path(MESSAGE_DIR)}")
print(f"Output directory (analysis): {display_path(OUTPUT_DIR)}")

if not MESSAGE_DIR.exists():
    print(f"Message directory not found: {display_path(MESSAGE_DIR)}")
    print("Run 01_itch_parser first to parse ITCH data.")
    available = []
else:
    available = sorted([d.name for d in MESSAGE_DIR.iterdir() if d.is_dir()])
    print(f"Available message types: {available}")

# Check if we have message data to analyze
HAS_MESSAGE_DATA = len(available) > 0

if not HAS_MESSAGE_DATA:
    raise RuntimeError(
        "No parsed ITCH message data found.\n"
        "This notebook requires output from 01_itch_parser (full parse ~60min).\n"
        f"Expected directory: {display_path(MESSAGE_DIR)}"
    )


# %% [markdown]
# ## 2. Counting Messages by Type
#
# Each message type is stored in its own Parquet subdirectory. Let's see how many messages
# of each type exist in our dataset.


# %%
def count_parquet_rows(base_dir: Path) -> dict[str, int]:
    """Count rows per ITCH message type.

    Only the single-letter codes the ITCH specification defines are counted, so a
    directory some other tool left in the store cannot appear in the chart as though
    the venue had published it.

    Args:
        base_dir: Directory containing subfolders like A/, C/, E/.

    Returns:
        Mapping from message-type code to total row count.
    """
    message_counts = {}
    for path in sorted(base_dir.iterdir()):
        if not path.is_dir() or path.name not in MESSAGE_SPECS:
            continue
        try:
            dset = ds.dataset(path.as_posix(), format="parquet")
            message_counts[path.name] = sum(frag.metadata.num_rows for frag in dset.get_fragments())
        except (OSError, ValueError) as exc:
            print(f"Could not read {path.name}: {type(exc).__name__}: {exc}")
    return message_counts


# %%
if HAS_MESSAGE_DATA:
    message_summary = count_parquet_rows(MESSAGE_DIR)
    labelled = {f"{code}  {MESSAGE_SPECS[code]['name']}": n for code, n in message_summary.items()}

    fig, ax = plt.subplots(figsize=(9, 6))
    pd.Series(labelled).sort_values().plot.barh(ax=ax, color=COLORS["blue"])
    ax.set_xscale("log")
    ax.set_title("Messages published per ITCH message type")
    ax.set_xlabel("Messages (log scale)")
    ax.set_ylabel("")
    show_with_alt(
        fig,
        "A horizontal bar chart with one bar per ITCH message type, labelled with its letter code and name and sorted with the most numerous type at the top and the rarest at the bottom. The horizontal axis counts messages on a logarithmic scale, so single-digit counts and counts in the hundreds of millions are both readable on the same chart.",
    )

    print("Message counts:")
    for msg_type, count in sorted(message_summary.items(), key=lambda x: -x[1]):
        print(f"  {msg_type} {MESSAGE_SPECS[msg_type]['name']:<28}: {count:>12,}")

# %% [markdown]
# ### Message Type Reference
#
# | Type | Name | Description |
# |------|------|-------------|
# | **A** | Add Order | New limit order enters the book |
# | **F** | Add Order (Attributed) | Same as A, with market participant ID |
# | **E** | Order Executed | Partial/full execution |
# | **C** | Order Executed w/Price | Execution at different price |
# | **X** | Order Cancel | Partial cancellation |
# | **D** | Order Delete | Full removal from book |
# | **P** | Trade | Non-displayed execution |
# | **Q** | Cross Trade | Opening/closing cross |

# %% [markdown]
# ## 3. Execution Attribution: Enriching E/C Messages
#
# E (Order Executed) and C (Order Executed with Price) messages contain `stock_locate`
# but not the stock symbol. We enrich them by joining to:
#
# 1. **R messages**: `stock_locate → stock` mapping (basic attribution)
# 2. **A/F messages**: `order_ref → (price, side)` (execution quality analysis)
#
# This enables filtering trades by stock symbol and analyzing execution quality.
#
# The result is written to `03_market_microstructure/output/nasdaq_itch/enriched/` and read by
# `04_itch_order_lifecycle_analysis` and `07_itch_stylized_facts`, so the join runs once
# for the whole chapter rather than three times. It goes under the chapter's output
# directory rather than beside the parsed messages because it is derived here, and the
# message store is whatever the reader has mounted at `ML4T_DATA_PATH` - a shared
# checkout, a synced folder, a read-only mount.


# %%
def _build_stock_directory(message_dir: Path) -> pl.DataFrame | None:
    """Load stock_locate -> stock mapping from R (Stock Directory) messages."""
    r_path = message_dir / "R"
    if not r_path.exists():
        print("No R (Stock Directory) messages found. Cannot enrich.")
        return None

    print("Building stock_locate -> stock mapping from R messages...")
    stock_directory = (
        pl.scan_parquet(r_path / "*.parquet").select("stock_locate", "stock").collect()
    )
    print(f"  {len(stock_directory):,} stocks in directory")
    return stock_directory


# %% [markdown]
# ### Build Order Attributes
# Map order references to their price, side, and stock from A/F messages.


# %%
def _build_order_attrs(message_dir: Path) -> pl.DataFrame | None:
    """Build order_ref -> (price, side, stock) mapping from A/F messages."""
    print("Building order_ref -> attributes mapping from A/F messages...")

    a_path = message_dir / "A"
    if not a_path.exists():
        print("No A (Add Order) messages found. Cannot enrich.")
        return None

    orders_a = pl.scan_parquet(a_path / "*.parquet").select(
        "order_reference_number",
        "price",
        "buy_sell_indicator",
        "stock",
    )

    # F messages have same structure plus attribution (MPID)
    f_path = message_dir / "F"
    if f_path.exists() and list(f_path.glob("*.parquet")):
        orders_f = pl.scan_parquet(f_path / "*.parquet").select(
            "order_reference_number",
            "price",
            "buy_sell_indicator",
            "stock",
        )
        order_attrs = pl.concat([orders_a, orders_f]).collect()
    else:
        order_attrs = orders_a.collect()

    print(f"  {len(order_attrs):,} orders with attributes")
    return order_attrs


# %% [markdown]
# ### Follow the replace chains
#
# A `U` message retires one order reference and issues another, and the new reference
# never appears in an add, so an execution against it cannot be attributed from the adds
# alone. Resolving one hop is not enough: `A → U → U` leaves the second `U`'s parent
# undefined until the first has been resolved, and a chain can run several deep.
#
# Following one hop per pass would need as many passes as the longest chain, and on a
# full day that is roughly ten thousand: a market maker rewrites the same quote all
# session. So the chain is collapsed by pointer doubling instead. Each pass replaces
# every reference's recorded parent with *its* parent, halving the remaining depth, and a
# chain of ten thousand closes in fourteen passes rather than ten thousand.
#
# Once every reference points at the add that started its chain, side and ticker come
# from that add, and the price comes from the `U` message itself, since that is what the
# replacement is quoting at. Whatever is still unresolved is reported rather than dropped
# in silence.
#
# The pass cap stops a cycle. Each pass doubles the depth a pointer covers, so forty of
# them reach a chain length no order book will ever produce, and a run that hits the cap
# has found references pointing at each other rather than a very long chain.


# %%
MAX_REPLACEMENT_PASSES = 40


def _apply_replacements(message_dir: Path, order_attrs: pl.DataFrame) -> pl.DataFrame:
    """Resolve U (Replace) chains so every new order reference inherits its attributes.

    Collapses each chain by pointer doubling, which costs a pass per doubling of the
    depth rather than a pass per hop.
    """
    u_path = message_dir / "U"
    if not (u_path.exists() and list(u_path.glob("*.parquet"))):
        return order_attrs

    print("Processing U (Replace) messages for order lineage...")
    # One row per new reference: who it replaced, and what it quotes at. A reference the
    # day issues twice would break the uniqueness the caller asserts, so keep the first.
    links = (
        pl.scan_parquet(u_path / "*.parquet")
        .select(
            pl.col("new_order_reference_number").alias("order_reference_number"),
            pl.col("original_order_reference_number").alias("ancestor"),
            "price",
        )
        .collect()
        .unique(subset="order_reference_number", keep="first")
    )
    total_replacements = links.height

    passes = 0
    while passes < MAX_REPLACEMENT_PASSES:
        doubled = links.join(
            links.select(
                pl.col("order_reference_number").alias("ancestor"),
                pl.col("ancestor").alias("grandparent"),
            ),
            on="ancestor",
            how="left",
        )
        # An ancestor that is itself a replaced reference still has further to go. When
        # none is, every chain already points at the add that started it.
        if not doubled["grandparent"].is_not_null().any():
            break
        links = doubled.select(
            "order_reference_number",
            pl.coalesce("grandparent", "ancestor").alias("ancestor"),
            "price",
        )
        passes += 1

    newly = links.join(
        order_attrs.select("order_reference_number", "buy_sell_indicator", "stock"),
        left_on="ancestor",
        right_on="order_reference_number",
        how="inner",
    ).select("order_reference_number", "price", "buy_sell_indicator", "stock")

    unresolved = total_replacements - newly.height
    print(
        f"  {newly.height:,} of {total_replacements:,} replacements "
        f"resolved over {passes} doubling pass(es)"
    )
    if unresolved:
        print(
            f"  {unresolved:,} replacements name a parent this sample never saw and "
            f"stay unattributed"
        )
    return pl.concat([order_attrs, newly])


# %% [markdown]
# ### Enrich E Messages
# Add stock symbol and order attributes to Order Executed messages.


# %%
def _enrich_e_messages(
    message_dir: Path, enriched_dir: Path, stock_directory: pl.DataFrame, order_attrs: pl.DataFrame
) -> int | None:
    """Enrich E (Order Executed) messages with stock and order attributes."""
    e_path = message_dir / "E"
    if not e_path.exists():
        return None

    print("Enriching E (Order Executed) messages...")
    executions_e = pl.scan_parquet(e_path / "*.parquet").collect()

    # Join to get stock from stock_locate
    enriched_e = executions_e.join(stock_directory, on="stock_locate", how="left")

    # Join to get order attributes (price, side)
    enriched_e = enriched_e.join(
        order_attrs.select(
            "order_reference_number",
            pl.col("price").alias("order_price"),
            pl.col("buy_sell_indicator").alias("side"),
        ),
        on="order_reference_number",
        how="left",
    )

    # E messages execute at the order's limit price
    enriched_e = enriched_e.with_columns(pl.col("order_price").alias("execution_price"))

    enriched_e.write_parquet(enriched_dir / "E.parquet")
    matched = enriched_e.filter(pl.col("stock").is_not_null()).height
    count = len(enriched_e)
    print(f"  {count:,} executions enriched ({matched:,} with stock match)")
    return count


# %% [markdown]
# ### Enrich C Messages
# Add stock symbol and order attributes to Order Executed with Price messages.


# %%
def _enrich_c_messages(
    message_dir: Path, enriched_dir: Path, stock_directory: pl.DataFrame, order_attrs: pl.DataFrame
) -> int | None:
    """Enrich C (Order Executed with Price) messages — includes price improvement."""
    c_path = message_dir / "C"
    if not c_path.exists():
        return None

    print("Enriching C (Order Executed with Price) messages...")
    executions_c = pl.scan_parquet(c_path / "*.parquet").collect()

    enriched_c = executions_c.join(stock_directory, on="stock_locate", how="left")

    enriched_c = enriched_c.join(
        order_attrs.select(
            "order_reference_number",
            pl.col("price").alias("order_price"),
            pl.col("buy_sell_indicator").alias("side"),
        ),
        on="order_reference_number",
        how="left",
    )

    # C messages have execution_price - can compute price improvement
    enriched_c = enriched_c.with_columns(
        (pl.col("execution_price").cast(pl.Int64) - pl.col("order_price").cast(pl.Int64)).alias(
            "price_improvement_raw"
        )
    )

    enriched_c.write_parquet(enriched_dir / "C.parquet")
    count = len(enriched_c)
    print(f"  {count:,} executions with price enriched")
    return count


# %% [markdown]
# ### Enrich X Messages
# Add stock symbol to Order Cancel messages.


# %%
def _enrich_x_messages(
    message_dir: Path, enriched_dir: Path, stock_directory: pl.DataFrame
) -> int | None:
    """Enrich X (Order Cancel) messages with stock symbol."""
    x_path = message_dir / "X"
    if not x_path.exists():
        return None

    print("Enriching X (Order Cancel) messages...")
    cancels = pl.scan_parquet(x_path / "*.parquet").collect()

    enriched_x = cancels.join(stock_directory, on="stock_locate", how="left")

    enriched_x.write_parquet(enriched_dir / "X.parquet")
    count = len(enriched_x)
    print(f"  {count:,} cancellations enriched")
    return count


# %% [markdown]
# ### Enrich Execution Messages
# Orchestrate enrichment of E/C/X messages with stock symbols and order attributes.


# %%
def enrich_execution_messages(message_dir: Path, enriched_dir: Path) -> dict[str, int]:
    """
    Enrich E/C/X messages with stock symbol and order attributes.

    Creates enriched Parquet files that enable:
    - Filtering executions by stock symbol
    - Execution quality analysis (fill price vs limit price)
    - Fill rate analysis by order characteristics

    Args:
        message_dir: Parsed ITCH message store, read only.
        enriched_dir: Where the enriched files are written. Under the chapter's output
            directory, not beside the messages: this is derived data, and writing it
            into the data root puts it in whatever the reader has mounted there.

    Returns count of enriched messages by type.
    """
    enriched_dir.mkdir(parents=True, exist_ok=True)

    stock_directory = _build_stock_directory(message_dir)
    if stock_directory is None:
        return {}

    order_attrs = _build_order_attrs(message_dir)
    if order_attrs is None:
        return {}
    order_attrs = _apply_replacements(message_dir, order_attrs)

    # Every enrichment below joins executions to this table on the order reference. A
    # duplicated reference would multiply execution rows and inflate every count and sum
    # downstream, so the key is asserted unique before any of them run.
    n_refs = order_attrs["order_reference_number"].n_unique()
    assert n_refs == order_attrs.height, (
        f"order attributes hold {order_attrs.height:,} rows for {n_refs:,} distinct order "
        f"references; joining on a duplicated key would multiply executions"
    )

    counts = {}
    for label, fn in [
        ("E", lambda: _enrich_e_messages(message_dir, enriched_dir, stock_directory, order_attrs)),
        ("C", lambda: _enrich_c_messages(message_dir, enriched_dir, stock_directory, order_attrs)),
        ("X", lambda: _enrich_x_messages(message_dir, enriched_dir, stock_directory)),
    ]:
        result = fn()
        if result is not None:
            counts[label] = result

    print(f"\nEnriched files saved to: {display_path(enriched_dir)}")
    return counts


# %%
if HAS_MESSAGE_DATA and (MESSAGE_DIR / "R").exists():
    already_built = (ENRICHED_DIR / "E.parquet").exists()
    if REBUILD_ENRICHED or not already_built:
        print("Running execution enrichment...")
        enrichment_counts = enrich_execution_messages(MESSAGE_DIR, ENRICHED_DIR)
        print("\nEnrichment summary:")
        for msg_type, count in enrichment_counts.items():
            print(f"  {msg_type}: {count:,} messages")
    else:
        print(
            f"Reusing the enriched files already in {display_path(ENRICHED_DIR)}; set "
            f"REBUILD_ENRICHED to rebuild them from the parsed messages."
        )

# %% [markdown]
# ## 4. Trade Volume and Value by Ticker
#
# We analyze **executions** (trades) from message types that reflect actual trades:
# - `'C'`: Order Executed with Price
# - `'E'`: Order Executed
# - `'P'`: Trade (regular)
# - `'Q'`: Cross Trade


# %%
def _unify_columns(df: pl.DataFrame, msg_type: str) -> pl.DataFrame | None:
    """Normalize column names across ITCH message types to a common schema."""
    cols = {c.lower(): c for c in df.columns}

    # Unify 'shares' column
    if "executed_shares" in cols:
        df = df.with_columns(pl.col(cols["executed_shares"]).cast(pl.Float64).alias("shares"))
    elif "shares" in cols:
        df = df.with_columns(pl.col(cols["shares"]).cast(pl.Float64).alias("shares"))
    else:
        return None

    # Unify 'price' column
    if "execution_price" in cols:
        df = df.with_columns(pl.col(cols["execution_price"]).cast(pl.Float64).alias("price"))
    elif "cross_price" in cols:
        df = df.with_columns(pl.col(cols["cross_price"]).cast(pl.Float64).alias("price"))
    elif "price" in cols:
        df = df.with_columns(pl.col(cols["price"]).cast(pl.Float64).alias("price"))
    else:
        return None

    # Handle timestamp
    if "timestamp" in cols:
        df = df.with_columns(pl.col(cols["timestamp"]).alias("timestamp"))

    # Handle ticker (may be 'stock' or 'ticker')
    if "ticker" in cols:
        df = df.with_columns(pl.col(cols["ticker"]).alias("ticker"))
    elif "stock" in cols:
        df = df.with_columns(pl.col(cols["stock"]).alias("ticker"))
    else:
        return None

    df = df.with_columns(pl.lit(msg_type).alias("msg_type"))

    keep_cols = ["timestamp", "ticker", "shares", "price", "msg_type"]
    existing = [c for c in keep_cols if c in df.columns]
    return df.select(existing)


# %% [markdown]
# ### Load Single Message Type
# Load one ITCH execution message type from enriched or raw parquet files.


# %%
def _load_single_msg_type(
    base_dir: Path, enriched_dir: Path, msg_type: str, max_rows: int | None
) -> pl.DataFrame | None:
    """Load a single ITCH execution message type (C, E, P, or Q)."""
    enriched_file = enriched_dir / f"{msg_type}.parquet"
    msg_folder = base_dir / msg_type

    try:
        if enriched_file.exists():
            lf = pl.scan_parquet(enriched_file)
            if max_rows:
                lf = lf.head(max_rows)
            df = lf.collect()
        elif msg_folder.is_dir():
            lf = pl.scan_parquet(msg_folder / "*.parquet")
            if max_rows:
                lf = lf.head(max_rows)
            df = lf.collect()
        else:
            return None

        if len(df) == 0:
            return None

        return _unify_columns(df, msg_type)

    except (OSError, FileNotFoundError, pl.exceptions.ComputeError) as e:
        print(f"Error loading {msg_type}: {e}")
    except Exception as e:
        print(f"Unexpected error loading {msg_type}: {type(e).__name__}: {e}")
    return None


# %% [markdown]
# ### Normalize Trades
# Concatenate trade DataFrames, clean tickers, normalize ITCH prices, and compute trade value.


# %%
def _normalize_trades(trades: list[pl.DataFrame], type_counts: dict[str, int]) -> pl.DataFrame:
    """Concat trade DataFrames, clean tickers, normalize prices, compute value."""
    all_trades = pl.concat(trades)
    all_trades = all_trades.drop_nulls(subset=["ticker", "shares", "price"])

    # Strip ITCH ticker padding (8-char fixed width with trailing spaces)
    all_trades = all_trades.with_columns(pl.col("ticker").str.strip_chars().alias("ticker"))

    # Normalize prices from ITCH price4 format (divide by 10000)
    # Only normalize if prices appear to be scaled integers (median > 10000)
    median_price = all_trades.select(pl.col("price").median()).item()
    if median_price is not None and median_price > 10000:
        all_trades = all_trades.with_columns((pl.col("price") / 10000).alias("price"))

    all_trades = all_trades.with_columns((pl.col("shares") * pl.col("price")).alias("value"))

    # Print message type breakdown
    if type_counts:
        print("Execution message type breakdown:")
        for mt, count in sorted(type_counts.items()):
            print(f"  {mt}: {count:>12,}")

    return all_trades


# %% [markdown]
# ### Load Executions
# Load and combine C, E, P, Q execution messages into a unified trades DataFrame.


# %%
def load_executions(
    base_dir: Path, enriched_dir: Path, max_rows: int | None = None
) -> pl.DataFrame:
    """
    Load execution data from C, E, P, Q message types using Polars.

    For E and C messages, prefers enriched files (with stock symbol).
    Falls back to raw files for P/Q which already have stock.

    Returns DataFrame with columns: timestamp, ticker, shares, price, value, msg_type
    """
    trades = []
    type_counts = {}

    for msg_type in ["C", "E", "P", "Q"]:
        df = _load_single_msg_type(base_dir, enriched_dir, msg_type, max_rows)
        if df is not None:
            type_counts[msg_type] = len(df)
            trades.append(df)

    if not trades:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "ticker": pl.Utf8,
                "shares": pl.Float64,
                "price": pl.Float64,
                "value": pl.Float64,
                "msg_type": pl.Utf8,
            }
        )

    return _normalize_trades(trades, type_counts)


# %%
if HAS_MESSAGE_DATA:
    trade_df = load_executions(MESSAGE_DIR, ENRICHED_DIR, max_rows=ROW_LIMIT)
    print(f"\nLoaded {len(trade_df):,} trades total")
    print(trade_df.schema)

    # Price sanity check - ITCH prices may be scaled (often 1/10000)
    # The parser should normalize to dollars; verify with sample prices
    if "price" in trade_df.columns and len(trade_df) > 0:
        price_stats = trade_df.select(
            [
                pl.col("price").min().alias("min"),
                pl.col("price").median().alias("median"),
                pl.col("price").max().alias("max"),
            ]
        ).row(0, named=True)
        print("\nPrice sanity check (should be plausible dollar values):")
        print(
            f"  Min: ${price_stats['min']:.2f}, Median: ${price_stats['median']:.2f}, Max: ${price_stats['max']:.2f}"
        )
        if price_stats["median"] > 100000:
            print("  WARNING: Prices appear scaled - check parser normalization!")

    # Timestamp type check
    if "timestamp" in trade_df.columns and len(trade_df) > 0:
        ts_dtype = trade_df.schema["timestamp"]
        print(f"\nTimestamp dtype: {ts_dtype}")
        # In Polars, check dtype by comparing base_type or string representation
        is_datetime = (
            ts_dtype.base_type() == pl.Datetime if hasattr(ts_dtype, "base_type") else False
        )
        if not is_datetime:
            print("  WARNING: Timestamps may need conversion for time-based analysis")

    print("\nSample trades:")
    display(trade_df.head().to_pandas())

# %% [markdown]
# ### Which tickers the day's dollars went through
#
# Trades are summed per ticker in two currencies: share count and dollar value. The two
# rank differently, because a share of a $3 stock and a share of a $300 stock are not
# comparable quantities, and it is the dollar ranking that says where the day's risk was
# transferred.

# %%
if HAS_MESSAGE_DATA and len(trade_df) > 0:
    trade_summary = trade_df.group_by("ticker").agg(
        [
            pl.col("shares").sum().alias("total_shares"),
            pl.col("value").sum().alias("total_value"),
            pl.col("shares").count().alias("trade_count"),
        ]
    )

    # Calculate shares
    total_value = trade_summary.select(pl.col("total_value").sum()).item()
    total_shares = trade_summary.select(pl.col("total_shares").sum()).item()

    trade_summary = trade_summary.with_columns(
        [
            (pl.col("total_value") / total_value).alias("share_of_value"),
            (pl.col("total_shares") / total_shares).alias("share_of_volume"),
        ]
    )

    trade_summary = trade_summary.sort("total_value", descending=True)

    print("Top 10 Tickers by Dollar Volume:")
    display(trade_summary.head(10).to_pandas())

    # Concentration analysis
    top_50 = trade_summary.head(50)
    sum_top_50 = top_50.select(pl.col("share_of_value").sum()).item()
    print(f"\nConcentration: Top 50 tickers account for {sum_top_50:.1%} of total dollar volume")

# %%
if HAS_MESSAGE_DATA and len(trade_df) > 0:
    # Plot cumulative concentration
    cum_val = trade_summary.select(pl.col("share_of_value").cum_sum()).to_series().to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(cum_val)), cum_val, linewidth=2)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="50% of value")
    ax.axhline(0.8, color="orange", linestyle="--", alpha=0.7, label="80% of value")

    # Find how many tickers needed for 50% and 80%
    n_50 = np.searchsorted(cum_val, 0.5) + 1
    n_80 = np.searchsorted(cum_val, 0.8) + 1

    ax.axvline(n_50, color="red", linestyle=":", alpha=0.5)
    ax.axvline(n_80, color="orange", linestyle=":", alpha=0.5)

    ax.set_title("Cumulative share of traded dollar value by ticker rank")
    ax.set_ylabel("Cumulative share of dollar volume")
    ax.set_xlabel("Ticker rank by traded value")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    ax.set_xlim(0, min(500, len(cum_val)))
    show_with_alt(
        fig,
        "A cumulative curve rising steeply from the origin and then flattening, plotting the running share of the day's traded dollar value against ticker rank, with rank on the horizontal axis truncated at five hundred and the share on the vertical axis as a percentage. Two dashed horizontal reference lines mark the fifty and eighty percent levels, each meeting a dotted vertical line at the rank where the curve crosses it.",
    )

    print(f"Tickers reaching 50% of traded value: {n_50}")
    print(f"Tickers reaching 80% of traded value: {n_80}")

# %% [markdown]
# Read the two crossings off that curve. The rank at which it reaches half the day's value,
# and the rank at which it reaches four fifths, are printed below it. Both are small
# relative to the several thousand tickers the venue quotes, and the gap between them says
# how quickly the tail thins out.
#
# What follows from it is practical: a universe screened on liquidity is not a small
# restriction of the market but almost all of it, and a strategy built on the names past
# the flat part of this curve is trading in a different regime from the ones before it.

# %% [markdown]
# ## Save Output for Downstream Notebooks
#
# Save the trade summary for use by `06_itch_intraday_patterns` and
# `07_itch_stylized_facts`.

# %%
if HAS_MESSAGE_DATA and len(trade_df) > 0:
    # Save trade summary for downstream notebooks
    output_path = OUTPUT_DIR / "trade_summary.parquet"
    trade_summary.write_parquet(output_path)
    print(f"Saved per-ticker trade summary to {display_path(output_path)}")

    trades_path = OUTPUT_DIR / "trades.parquet"
    trade_df.write_parquet(trades_path)
    print(f"Saved the trade table to {display_path(trades_path)}")
    print(f"  {len(trade_df):,} trades across {trade_df['ticker'].n_unique():,} tickers")

# %% [markdown]
# ## Key Takeaways
#
# 1. **A venue publishes far more quoting than trading.** Adds and deletes dominate the
#    message counts and executions are a small share of them, which is the same fact
#    `04_itch_order_lifecycle_analysis` measures per order.
# 2. **An execution message does not say what was traded.** `E` and `C` carry a numeric
#    `stock_locate` and an order reference, so the ticker comes from the `R` directory
#    and the side and limit price from the add that created the order.
# 3. **Follow replace chains to the end.** A `U` issues a new reference whose parent may
#    itself be a `U`, so resolving one hop attributes some executions and silently loses
#    the rest. The resolution here repeats until it stops finding anything.
# 4. **Assert the key before a left join on it.** A duplicated order reference multiplies
#    execution rows, and every count and sum computed afterwards is wrong by a factor
#    nothing reports.
# 5. **Rank by dollars, not by shares.** The two orderings differ, and it is the dollar
#    ranking that says where the day's risk moved.
# 6. **Do the join once.** Three later notebooks read the enriched files this one writes,
#    so the attribution has one definition rather than four.
#
# ### Known limitations
#
# - One venue, one session. NASDAQ-routed activity only, so a ticker's share here is its
#   share of this venue rather than of its consolidated volume.
# - `P` trades are non-displayed and have no resting order to attribute to, so execution
#   quality against a limit price is undefined for them.
# - A replacement whose parent falls outside the sample stays unattributed; the count is
#   printed rather than absorbed.
#
# **Next**: `06_itch_intraday_patterns` reads the trade table written above.
#
# ---
#
# ## Reference
#
# Bouchaud, J.-P., Bonart, J., Donier, J., & Gould, M. (2018).
# *Trades, Quotes and Prices: Financial Markets Under the Microscope*.
# Cambridge University Press.
# [https://doi.org/10.1017/9781009028943](https://doi.org/10.1017/9781009028943)
