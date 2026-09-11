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
# # Order Book Reconstruction from NASDAQ ITCH Messages
#
# **Chapter 3: Market Microstructure**
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Reconstruct the **limit order book (LOB)** for a single symbol-day from NASDAQ
# ITCH message-by-order (MBO) data, producing a one-second-resolution top-of-book
# snapshot series with per-second order-flow imbalance (OFI). The reconstruction tracks
# every resting order internally; each snapshot records the highest bid and the lowest
# ask, with the shares resting at each.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Turn a day of `A`/`F`/`D`/`X`/`E`/`C`/`U` messages into a book that knows, at every
#   moment, how many shares rest at each price on each side.
# - Follow a Replace (`U`) message, which retires one order reference and issues a new
#   one, and say why a chain of them defeats a reconstruction that only reads adds.
# - Check a reconstruction by counting crossed quotes, where the bid sits above the ask,
#   which cannot happen in a real book and so counts reconstruction errors.
# - Compute order-flow imbalance per second: shares added to the bid minus shares taken
#   off it, less the same for the ask.
#
# ## Book reference
#
# Section §3.3, *From Raw Messages to the Limit Order Book*.
#
# ## Prerequisites
#
# - Parsed ITCH message parquets at `data/equities/market/microstructure/nasdaq_itch/messages/`
#   (output of `01_itch_parser` or the Rust parser).
# - Familiarity with §3.2 (data feed taxonomy) and §3.3 (LOB reconstruction
#   algorithm).
#
# **Output**: per-second LOB snapshots saved to
# `03_market_microstructure/output/nasdaq_itch/order_book/{SYMBOL}/lob_snapshots.parquet`.

# %% [markdown]
# ## Setup

# %%
"""Order Book Reconstruction from NASDAQ ITCH Messages — build limit order book from MBO data."""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from limit_orderbook import (
    get_stock_locate_mapping,
    load_itch_messages,
    reconstruct_lob_with_ofi,
)

from data.equities.loader import load_nasdaq_itch
from utils.paths import display_path, get_output_dir
from utils.style import show_with_alt

# %% [markdown]
# ### Declared parameters
#
# `SYMBOL` and `TRADING_DATE` choose the one symbol-day to reconstruct; ITCH is a
# venue-wide feed, and one book describes one symbol on one day.
#
# `START_TIME` and `END_TIME` bound the snapshots that are *kept*, not the messages that
# are read. Regular trading hours on a US equity venue run 09:30 to 16:00 Eastern, and
# those are the hours a reader wants a book for. Messages before `START_TIME` still have
# to be processed, because an order added at 04:00 in the pre-market can be deleted at
# 09:31, and the book cannot subtract shares it never added.
#
# `MESSAGE_LIMIT` caps how many messages of each type are read. A whole symbol-day of
# AAPL is a few million messages, which is a minute of work, so a smaller cap is what
# makes a first pass quick while exercising the same code path. `None` reads them all,
# which is what the committed run does.
#
# `SNAPSHOT_FREQ` sets how often the book is written down. One second is fine enough to
# see liquidity move and coarse enough that a trading day fits in a frame of a few tens of
# thousands of rows.

# %% tags=["parameters"]
SYMBOL = "AAPL"
TRADING_DATE = "2020-01-30"
START_TIME = "09:30:00"
END_TIME = "16:00:00"
MESSAGE_LIMIT = None
SNAPSHOT_FREQ = "1s"

# %% [markdown]
# Batch runs over many symbols set `ITCH_SYMBOL` in the environment rather than editing
# the cell above; the parameter is the default when the variable is unset.

# %%
symbol = os.environ.get("ITCH_SYMBOL", SYMBOL)

ITCH_DIR = load_nasdaq_itch(get_base_path=True)
OUTPUT_DIR = get_output_dir(3, "nasdaq_itch") / "order_book"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input messages:   {display_path(ITCH_DIR)}")
print(f"Output directory: {display_path(OUTPUT_DIR)}")

# %%
# Validate parsed ITCH data — produced by 01_itch_parser or Rust parser
assert ITCH_DIR.exists(), (
    f"Parsed ITCH data not found at {ITCH_DIR}.\n"
    "Run the ITCH pipeline first:\n"
    "  1. Download: uv run python data/equities/market/microstructure/nasdaq_itch_download.py\n"
    "  2. Parse:    Run 01_itch_parser.py (Section 4) or Rust parser (Section 6)"
)
msg_types = sorted([d.name for d in ITCH_DIR.iterdir() if d.is_dir()])
assert len(msg_types) > 0, f"No message types in {ITCH_DIR} — run 01_itch_parser first"
print(f"Available message types: {msg_types}")

# %% [markdown]
# ## 1. Understanding ITCH Message Types
#
# NASDAQ ITCH v5.0 provides message-by-order (MBO) data with these key types:
#
# | Type | Name | Description |
# |------|------|-------------|
# | **A** | Add Order | New limit order enters the book |
# | **F** | Add Order (MPID) | Same as A, with market participant ID |
# | **E** | Order Executed | Partial/full execution |
# | **C** | Order Executed w/Price | Execution at different price (hidden orders) |
# | **X** | Order Cancel | Partial cancellation |
# | **D** | Order Delete | Full removal from book |
# | **U** | Order Replace | Modify price/size (cancel + add) |
# | **P** | Trade | Non-displayed execution |
#
# **Price format**: an integer with four implied decimal places, so the field 3212000 is a
# price of three hundred twenty-one dollars and twenty cents.
#
# **Timezone note**: ITCH timestamps are nanoseconds since midnight in US/Eastern (exchange local time).
# The data is timezone-naive; convert to America/New_York before cross-source joins.


# %% [markdown]
# ## 2. Load Messages for Target Symbol
#
# We load all message types needed for LOB reconstruction and **pre-join** them
# to attach price and side information to D/X/E messages.

# %%
# Get the stock_locate ID for our symbol from R messages
stock_map = get_stock_locate_mapping(ITCH_DIR)
assert symbol in stock_map, f"Symbol {symbol} not found in stock directory"
symbol_locate = stock_map[symbol]
print(f"Symbol {symbol} has stock_locate = {symbol_locate}")

# %% [markdown]
# Add orders arrive as two message types: `A` carries no member identifier and `F` does.
# The reconstruction ignores the identifier, so the book is built from both. A trading
# day always carries both, and `load_itch_messages` returns None for a type the store
# does not hold, so an absent one means the store is partial - a parse stopped early, or
# a reduced fixture - and the book is built from whichever is there.

# %%
# Add orders (A and F types) - have 'stock' column
add_a = load_itch_messages(ITCH_DIR, "A", symbol=symbol, max_messages=MESSAGE_LIMIT)
add_f = load_itch_messages(ITCH_DIR, "F", symbol=symbol, max_messages=MESSAGE_LIMIT)

# Combine A and F (select common columns)
common_cols = [
    "stock_locate",
    "tracking_number",
    "timestamp",
    "order_reference_number",
    "buy_sell_indicator",
    "shares",
    "stock",
    "price",
]
add_frames = [df.select(common_cols) for df in (add_a, add_f) if df is not None]
assert add_frames, (
    f"No add messages of either type for {symbol} in {ITCH_DIR}. The book cannot be built "
    "without them; check that the parse wrote the A and F directories."
)
if add_f is None:
    print("No F (add with attribution) messages in this store; building the book from A alone")
add_orders = pl.concat(add_frames)

# D, X, E, C, U messages don't have 'stock' column - use stock_locate filtering
deletes = load_itch_messages(ITCH_DIR, "D", stock_locate=symbol_locate, max_messages=MESSAGE_LIMIT)
cancels = load_itch_messages(ITCH_DIR, "X", stock_locate=symbol_locate, max_messages=MESSAGE_LIMIT)
executions = load_itch_messages(
    ITCH_DIR, "E", stock_locate=symbol_locate, max_messages=MESSAGE_LIMIT
)
executions_c = load_itch_messages(
    ITCH_DIR, "C", stock_locate=symbol_locate, max_messages=MESSAGE_LIMIT
)
replaces = load_itch_messages(ITCH_DIR, "U", stock_locate=symbol_locate, max_messages=MESSAGE_LIMIT)

# P messages (trades) have 'stock' column
trades = load_itch_messages(ITCH_DIR, "P", symbol=symbol, max_messages=MESSAGE_LIMIT)


def n_messages(frame: pl.DataFrame | None) -> int:
    """Message count, 0 for a type the store does not hold."""
    return 0 if frame is None else len(frame)


def until(frame: pl.DataFrame | None, cutoff: datetime) -> pl.DataFrame | None:
    """Messages up to `cutoff`, passing an absent type through as None."""
    return None if frame is None else frame.filter(pl.col("timestamp") <= cutoff)


# %% [markdown]
# ### Why the add messages are not enough on their own
#
# A `D`, `X` or `E` message names only an order reference number. To know which price
# level it acts on, you need the order that reference belongs to. The obvious move is to
# look every reference up in the `A`/`F` adds - and it loses a large part of the day,
# because a Replace (`U`) message retires one reference and issues a *new* one. After
# `A → U`, the live order is the one `U` created, and a later `D` names that reference,
# which never appeared in an add.
#
# So the reconstruction keeps its own pool of live orders and adds `U` results to it as
# it goes, rather than resolving references against the adds up front. The cell below
# measures what the up-front approach would have missed on this symbol-day.
#
# The measurement only means that on a complete day. `MESSAGE_LIMIT` truncates each
# message type independently, so under a cap a reference can be missing simply because
# its add was past the cut; the cell says which case it is reporting.


# %%
def share_not_in_adds(frame: pl.DataFrame, ref_col: str) -> tuple[int, int]:
    """Count references in `ref_col` that no A/F add message ever created."""
    if frame is None or frame.height == 0:
        return 0, 0
    missing = (
        frame.select(ref_col)
        .join(add_refs, left_on=ref_col, right_on="order_reference_number", how="anti")
        .height
    )
    return missing, frame.height


add_refs = add_orders.select("order_reference_number").unique()
complete_day = MESSAGE_LIMIT is None
u_reason = (
    "came from another U rather than an add"
    if complete_day
    else "is absent from the loaded add sample"
)
dxe_reason = "no add ever created" if complete_day else "absent from the loaded add sample"
if not complete_day:
    print(
        f"MESSAGE_LIMIT={MESSAGE_LIMIT:,} truncates each message type separately, so the "
        f"counts below describe this sample, not the day."
    )

missing_u, total_u = share_not_in_adds(replaces, "original_order_reference_number")
if total_u:
    print(f"\nReplace (U) messages: {total_u:,}")
    print(f"  ...whose replaced order {u_reason}: {missing_u:,} ({missing_u / total_u * 100:.1f}%)")

missing_dxe, total_dxe = 0, 0
for frame in (deletes, cancels, executions):
    missing, total = share_not_in_adds(frame, "order_reference_number")
    missing_dxe += missing
    total_dxe += total
if total_dxe:
    print(f"\nD/X/E messages: {total_dxe:,}")
    print(
        f"  ...naming an order {dxe_reason}: {missing_dxe:,} ({missing_dxe / total_dxe * 100:.1f}%)"
    )

# %%
print(f"\nSymbol: {symbol}")
print(f"Trading Date: {TRADING_DATE}")
print("\nMessage counts:")
print(f"  Add orders (A+F): {len(add_orders):,}")
print(f"  Deletes (D): {n_messages(deletes):,}")
print(f"  Cancels (X): {n_messages(cancels):,}")
print(f"  Executions (E): {n_messages(executions):,}")
print(f"  Replaces (U): {n_messages(replaces):,}")
print(f"  Trades (P): {n_messages(trades):,}")

# %% [markdown]
# The first rows of the add messages show the fields the reconstruction reads:
# `order_reference_number` is the identity a later `D`, `X` or `E` will name,
# `buy_sell_indicator` puts the order on a side, and `price` and `shares` say where it
# rests and how much of it.

# %%
add_orders.head(5)

# %% [markdown]
# Not every add is an attempt to trade. Market-peg orders and orders parked far from the
# touch sit at sentinel prices near zero or in the hundreds of thousands, so the outright
# minimum and maximum say nothing about where the symbol traded. The first and
# ninety-ninth percentiles bound where displayed liquidity actually rests.

# %%
add_prices = add_orders["price"]
print(f"Price range (min to max):        ${add_prices.min():,.2f} - ${add_prices.max():,.2f}")
print(
    f"Central range (1st to 99th pct): "
    f"${add_prices.quantile(0.01):,.2f} - ${add_prices.quantile(0.99):,.2f}"
)

# %% [markdown]
# ## 3. Order Book Reconstruction Algorithm
#
# ### Two structures, and why both are needed
#
# The reconstruction carries an **order pool** and a **book**. The pool maps each live
# order reference to its side, price and *remaining* shares. The book maps each price to
# the total shares resting there on each side:
#
# ```
# pool = {order_ref: (side, price, shares_remaining), ...}
# book = {
#     "B": {price: total_shares, ...},  # bids
#     "S": {price: total_shares, ...},  # asks
# }
# ```
#
# The book alone cannot process a message, because `D` and `E` name an order and not a
# price. The pool alone cannot answer what the top of the book is without a scan. So each
# message reads the pool to find the level it acts on and then moves shares on the book:
#
# | Message | Pool | Book |
# |---|---|---|
# | `A`/`F` add | record the new order | add its shares at its price |
# | `D` delete | drop the order | subtract whatever remained of it |
# | `X` cancel | reduce remaining shares | subtract the cancelled shares |
# | `E` execute | reduce remaining shares | subtract the executed shares |
# | `C` execute with price | reduce remaining shares | subtract them at the resting price |
# | `U` replace | retire the old reference, record the new one | subtract at the old price, add at the new |
#
# The remaining-shares bookkeeping is what makes `D` correct. An order added for 500
# shares that has already executed 300 leaves 200 on the book, and the delete must remove
# 200. A reconstruction that subtracts the original 500 drives the level negative.
#
# `reconstruct_lob_with_ofi` in `limit_orderbook` does this in a compiled loop, and
# `14_itch_bar_sampling` calls the same module.

# %% [markdown]
# ## 4. Run Reconstruction

# %% [markdown]
# Every message from the start of the day up to `END_TIME` is processed, and only the
# snapshots from `START_TIME` onwards are kept. The two boundaries differ because the
# pool has to be warm before the first snapshot is meaningful: an order added at 04:00
# in the pre-market, partly executed at 05:00 and deleted at 09:31 leaves the book
# correctly only if all three messages were seen. Start reading at 09:30 and the delete
# arrives for an order the pool has never heard of.

# %%
start_time = datetime.strptime(f"{TRADING_DATE} {START_TIME}", "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime(f"{TRADING_DATE} {END_TIME}", "%Y-%m-%d %H:%M:%S")

add_all = add_orders.filter(pl.col("timestamp") <= end_time)
del_all = until(deletes, end_time)
can_all = until(cancels, end_time)
exec_all = until(executions, end_time)
exec_c_all = until(executions_c, end_time)
rep_all = until(replaces, end_time)

print(f"Messages for LOB reconstruction (up to {end_time.time()}):")
print(f"  Add orders: {len(add_all):,}")
print(f"  Deletes: {n_messages(del_all):,}")
print(f"  Cancels: {n_messages(can_all):,}")
print(f"  Executions (E): {n_messages(exec_all):,}")
print(f"  Executions (C): {n_messages(exec_c_all):,}")
print(f"  Replaces: {n_messages(rep_all):,}")

# %% [markdown]
# The reconstruction runs the message loop in a compiled kernel and accumulates
# order-flow imbalance as it goes, so the pass that builds the book is also the pass that
# measures the flow into it.

# %%
lob = reconstruct_lob_with_ofi(
    add_all,
    del_all,
    can_all,
    exec_all,
    executions_c=exec_c_all,
    replaces=rep_all,
    snapshot_freq=SNAPSHOT_FREQ,
)

# Filter to RTH snapshots only (reconstruction processes all messages from start of day)
if len(lob) > 0:
    lob = lob.filter(pl.col("timestamp") >= start_time)

assert len(lob) > 0, (
    f"LOB reconstruction returned 0 snapshots for {symbol} on {TRADING_DATE}. "
    "Check that the trading date has parsed ITCH messages on disk."
)

# %%
print(f"LOB snapshots: {len(lob):,}")

# %%
lob.head()

# %% [markdown]
# ### Spread validity
#
# A crossed quote is a snapshot whose highest bid sits above its lowest ask. A real book
# cannot be in that state - the two orders would have traded - so every crossed snapshot
# is a reconstruction error: a message dropped, or shares subtracted from the wrong level.
# The count below is the reconstruction's own error rate.

# %%
valid_count = (lob["spread"] > 0).sum()
crossed_count = (lob["spread"] < 0).sum()
print(f"Valid spreads (spread > 0): {valid_count:,} ({valid_count / len(lob) * 100:.1f}%)")
print(f"Crossed quotes (spread < 0): {crossed_count:,} ({crossed_count / len(lob) * 100:.1f}%)")

# %% [markdown]
# ### Order Flow Imbalance per second
#
# OFI = (bid adds − bid removes) − (ask adds − ask removes), aggregated to one
# second. Cumulative OFI tracks net buying/selling pressure within the trading
# day.

# %%
lob.select(
    pl.col("ofi").mean().alias("mean"),
    pl.col("ofi").std().alias("std"),
    pl.col("ofi").quantile(0.5).alias("median"),
    pl.col("ofi").min().alias("min"),
    pl.col("ofi").max().alias("max"),
)

# %% [markdown]
# ## 5. Visualize Order Book Dynamics
#
# For detailed spread and imbalance analysis over time, see **`03_itch_lob_analysis`**.
# This section focuses on market depth which shows the reconstructed book structure.

# %%
# Create output directory for symbol
symbol_dir = OUTPUT_DIR / symbol
symbol_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### Market depth through the session
#
# Two stacked panels share a time axis over the trading day. The top panel is the signed
# order-flow imbalance summed within each minute; the bottom is the shares resting at the
# highest bid and the lowest ask, averaged within each minute.

# %%
ALT_LOB_DYNAMICS = "Two stacked line charts sharing a time axis across one trading session. The upper panel plots order-flow imbalance per minute as a single dark line oscillating about a dashed zero line, with the vertical range clipped to the first and ninety-ninth percentiles. The lower panel plots two lines, bid depth in green and ask depth in red, showing the shares resting at the top of the book in each minute."

# %%
lob_pd = lob.to_pandas().set_index("timestamp")
ofi_1m = (
    lob_pd[["ofi", "bid_size_0", "ask_size_0"]]
    .resample("1min")
    .agg({"ofi": "sum", "bid_size_0": "mean", "ask_size_0": "mean"})
)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1 = axes[0]
ofi_series = ofi_1m["ofi"].fillna(0)
ofi_series.plot(ax=ax1, color="#1E3A5F", linewidth=1.0, label="1-min OFI")
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ofi_values = ofi_series.to_numpy()
if len(ofi_values):
    low, high = np.nanpercentile(ofi_values, [1, 99])
    pad = max(abs(low), abs(high)) * 0.15
    ax1.set_ylim(low - pad, high + pad)
ax1.set_title(f"{symbol} order-flow imbalance per minute")
ax1.set_ylabel("OFI (shares per minute)")
ax1.legend(loc="upper right")

ax2 = axes[1]
ofi_1m["bid_size_0"].plot(ax=ax2, label="Bid depth (top of book)", color="green", alpha=0.7)
ofi_1m["ask_size_0"].plot(ax=ax2, label="Ask depth (top of book)", color="red", alpha=0.7)
ax2.set_title(f"{symbol} shares resting at the best bid and the best ask, per minute")
ax2.set_ylabel("Shares")
ax2.set_xlabel("Time (US/Eastern)")
ax2.legend()

show_with_alt(fig, ALT_LOB_DYNAMICS)

# %% [markdown]
# `03_itch_lob_analysis` takes these snapshots further, into depth imbalance and whether
# order flow anticipates the next price move.

# %% [markdown]
# ## 6. Save Results

# %%
output_file = symbol_dir / "lob_snapshots.parquet"
lob.write_parquet(output_file)
print(f"Saved LOB snapshots to: {display_path(output_file)}")
print(f"Rows: {lob.height:,}  Columns: {lob.width}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Track what remains of an order, not what it started as.** An add of 500 shares
#    followed by executions of 100 and 200 leaves 200 on the book, and the delete that
#    ends it removes 200. Subtracting the original size drives the price level negative
#    and the error stays there for the rest of the session.
# 2. **A replace is a new order.** `U` retires one reference and issues another, so a
#    reconstruction that resolves references only against the adds loses every order that
#    has been replaced. The counts printed earlier in this notebook say how much of the
#    day that is for this symbol.
# 3. **Read from the start of the day, snapshot from the open.** The two windows are
#    different: the pool has to see the pre-market adds that later messages will name.
# 4. **`C` reports a price the book never showed.** It carries its own `execution_price`,
#    which is where the trade printed; the shares it removes still come off the order's
#    resting price, because that is where they were displayed. `P` trades are
#    non-displayed throughout, so they never entered the visible book and do not change
#    it.
# 5. **Crossed quotes are the reconstruction's error rate.** They cannot occur in a real
#    book, so their share is a direct check rather than a market observation.
#
# ### Known limitations
#
# - One venue. ITCH carries NASDAQ-routed activity, so this book is NASDAQ's, not the
#   consolidated quote across all US venues.
# - The snapshots record the top of the book. The pool holds every price level, but what
#   is written out is the highest bid, the lowest ask, and the shares resting at each.
# - Hidden liquidity is invisible by construction: an order that was never displayed
#   never entered the book, and only its execution (`P`) is observable.
#
# ### Next Steps
#
# - **`03_itch_lob_analysis`**: Spread dynamics, OFI predictability, liquidity spectrum
# - **Chapter 8**: Feature engineering using LOB metrics
# - **Chapter 19**: Price impact modeling using depth and imbalance
#
# ---
#
# ## References
#
# - Bouchaud, J.-P., Bonart, J., Donier, J., & Gould, M. (2018).
#   *Trades, Quotes and Prices: Financial Markets Under the Microscope*.
#   Cambridge University Press.
#   [https://doi.org/10.1017/9781009028943](https://doi.org/10.1017/9781009028943)
#
# - Gould, M. D., Porter, M. A., Williams, S., McDonald, M., Fenn, D. J., & Howison, S. D. (2013).
#   "Limit order books." *Quantitative Finance*, 13(11), 1709-1742.
#   [https://doi.org/10.1080/14697688.2013.803148](https://doi.org/10.1080/14697688.2013.803148)
