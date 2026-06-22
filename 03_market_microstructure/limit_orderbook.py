"""
Limit Order Book Utilities

Shared functions for LOB reconstruction from ITCH messages.
Used by:
- 03_market_microstructure/02_itch_lob_reconstruction.py (LOB snapshots)
- 03_market_microstructure/14_itch_bar_sampling.py (Lee-Ready trade classification)

Key insight: Orders can be created by Replace (U) messages through chains:
A → U → U → U. Messages reference orders by order_reference_number,
which may have been created by any prior Add (A/F) or Replace (U) message.

Includes Numba-accelerated version for production use with OFI computation.
"""

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numba
import numpy as np
import polars as pl
from numba import float64, int64
from numba.typed import Dict as NumbaDict
from tqdm.auto import tqdm


def get_stock_locate_mapping(itch_dir: Path) -> dict[str, int]:
    """Load stock → stock_locate mapping from R (Stock Directory) messages.

    The R message provides the official mapping between stock symbols and
    their numeric stock_locate identifiers used in all other ITCH messages.

    Parameters
    ----------
    itch_dir : Path
        Directory containing parsed ITCH message subdirectories (A/, D/, R/, etc.)

    Returns
    -------
    dict[str, int]
        Mapping from stock symbol to stock_locate ID
    """
    r_dir = itch_dir / "R"
    if not r_dir.exists():
        return {}
    df = pl.scan_parquet(r_dir).collect()
    return dict(zip(df["stock"].to_list(), df["stock_locate"].to_list(), strict=False))


def load_itch_messages(
    itch_dir: Path,
    msg_type: str,
    symbol: str = None,
    stock_locate: int = None,
    max_messages: int = None,
) -> pl.DataFrame | None:
    """Load parsed ITCH messages from parquet.

    Parameters
    ----------
    itch_dir : Path
        Directory containing parsed ITCH message subdirectories
    msg_type : str
        ITCH message type (A, D, E, X, P, R, etc.)
    symbol : str, optional
        Filter to specific stock symbol (for messages with 'stock' column: A, F, P, R)
    stock_locate : int, optional
        Filter by stock_locate ID (for messages without 'stock' column: D, X, E, U)
    max_messages : int, optional
        Maximum messages to return (applied AFTER filtering)

    Returns
    -------
    pl.DataFrame or None
        Parsed messages with converted price, or None if no data

    Notes
    -----
    ITCH message types have different columns:
    - A, F (Add): Have 'stock' column
    - D, X, E, U (Modify): Only have 'stock_locate' - need stock_locate ID to filter
    - P (Trade): Has 'stock' column
    - R (Stock Directory): Maps stock_locate → stock symbol
    """
    msg_dir = itch_dir / msg_type
    if not msg_dir.exists():
        return None

    # Use lazy scan with predicate pushdown for memory efficiency
    lf = pl.scan_parquet(msg_dir)

    # Get schema to check available columns
    schema = lf.collect_schema()

    # Filter by stock symbol if column exists (predicate pushdown)
    if symbol and "stock" in schema:
        lf = lf.filter(pl.col("stock") == symbol)

    # Filter by stock_locate ID (for D, X, E, U messages that lack 'stock')
    if stock_locate is not None and "stock_locate" in schema:
        lf = lf.filter(pl.col("stock_locate") == stock_locate)

    # Apply row limit AFTER filtering
    if max_messages is not None:
        lf = lf.head(max_messages)

    # Collect after all filters applied (predicate pushdown optimization)
    df = lf.collect()

    # Convert price from price4 format (divide by 10000)
    price_cols = ["price", "execution_price"]
    for col in price_cols:
        if col in df.columns:
            df = df.with_columns((pl.col(col) / 10000).alias(col))

    return df


def load_messages_for_symbol(
    itch_dir: Path,
    symbol: str,
    stock_locate: int = None,
    max_messages: int = None,
    start_time: datetime = None,
    end_time: datetime = None,
) -> dict[str, pl.DataFrame]:
    """Load all message types for a single symbol.

    Convenience function that loads A, F, D, X, E, C, U, P messages
    for a given symbol, applying time filtering if specified.

    Parameters
    ----------
    itch_dir : Path
        Directory containing parsed ITCH message subdirectories
    symbol : str
        Stock symbol (e.g., "AAPL")
    stock_locate : int, optional
        Stock locate ID. If None, will be looked up from R messages.
    max_messages : int, optional
        Maximum messages per type
    start_time : datetime, optional
        Filter messages >= this time
    end_time : datetime, optional
        Filter messages <= this time

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary with keys 'A', 'F', 'D', 'X', 'E', 'C', 'U', 'P'
        containing filtered DataFrames
    """
    # Get stock_locate if not provided
    if stock_locate is None:
        mapping = get_stock_locate_mapping(itch_dir)
        stock_locate = mapping.get(symbol)
        if stock_locate is None:
            raise ValueError(f"Symbol {symbol} not found in stock directory")

    messages = {}

    # Message types with 'stock' column (filter by symbol)
    for msg_type in ["A", "F", "P"]:
        df = load_itch_messages(itch_dir, msg_type, symbol=symbol, max_messages=max_messages)
        if df is not None and len(df) > 0:
            if start_time and "timestamp" in df.columns:
                df = df.filter(pl.col("timestamp") >= start_time)
            if end_time and "timestamp" in df.columns:
                df = df.filter(pl.col("timestamp") <= end_time)
            messages[msg_type] = df
        else:
            messages[msg_type] = pl.DataFrame()

    # Message types with 'stock_locate' column (filter by ID)
    for msg_type in ["D", "X", "E", "C", "U"]:
        df = load_itch_messages(
            itch_dir, msg_type, stock_locate=stock_locate, max_messages=max_messages
        )
        if df is not None and len(df) > 0:
            if start_time and "timestamp" in df.columns:
                df = df.filter(pl.col("timestamp") >= start_time)
            if end_time and "timestamp" in df.columns:
                df = df.filter(pl.col("timestamp") <= end_time)
            messages[msg_type] = df
        else:
            messages[msg_type] = pl.DataFrame()

    return messages


def _get_snapshot(book: dict, n_levels: int, timestamp) -> dict | None:
    """Extract top N levels from current book state.

    Parameters
    ----------
    book : dict
        Book state with keys 'B' (bids) and 'S' (asks),
        each mapping price -> size
    n_levels : int
        Number of price levels to extract per side
    timestamp : datetime
        Timestamp for this snapshot

    Returns
    -------
    dict or None
        Snapshot with bid/ask prices and sizes, or None if incomplete
    """
    snapshot = {"timestamp": timestamp}

    # Bids: highest prices first
    bids = sorted(book["B"].items(), key=lambda x: -x[0])[:n_levels]
    for i, (price, size) in enumerate(bids):
        snapshot[f"bid_price_{i}"] = price
        snapshot[f"bid_size_{i}"] = size

    # Asks: lowest prices first
    asks = sorted(book["S"].items(), key=lambda x: x[0])[:n_levels]
    for i, (price, size) in enumerate(asks):
        snapshot[f"ask_price_{i}"] = price
        snapshot[f"ask_size_{i}"] = size

    # Only return if we have both sides
    if bids and asks:
        snapshot["best_bid"] = bids[0][0]
        snapshot["best_ask"] = asks[0][0]
        snapshot["spread"] = asks[0][0] - bids[0][0]
        snapshot["mid_price"] = (asks[0][0] + bids[0][0]) / 2
        return snapshot
    return None


def reconstruct_lob(
    add_orders: pl.DataFrame,
    deletes: pl.DataFrame,
    cancels: pl.DataFrame,
    executions: pl.DataFrame,
    executions_c: pl.DataFrame | None = None,
    replaces: pl.DataFrame | None = None,
    n_levels: int = 10,
    snapshot_freq: str = "1s",
    snapshot_start: datetime | None = None,
    show_progress: bool = True,
) -> pl.DataFrame:
    """
    Reconstruct limit order book from ITCH messages using order state tracking.

    This implementation correctly tracks remaining shares per order, following the
    reference C++ implementation (martinobdl/ITCH) and ML4T 2nd edition pattern.

    Key insight: Delete (D) messages must use the CURRENT remaining shares, not
    the original shares from the Add message. An order may have been partially
    executed or cancelled before deletion.

    Parameters
    ----------
    add_orders : pl.DataFrame
        Combined A and F messages with add orders
    deletes : pl.DataFrame
        D messages (full order deletion)
    cancels : pl.DataFrame
        X messages (partial cancellation) - have cancelled_shares
    executions : pl.DataFrame
        E messages (order executions) - have executed_shares
    executions_c : pl.DataFrame, optional
        C messages (order executed with price)
    replaces : pl.DataFrame, optional
        U messages (order replacements)
    n_levels : int
        Number of price levels to track on each side
    snapshot_freq : str
        Frequency for LOB snapshots (e.g., '1s', '100ms')
    snapshot_start : datetime, optional
        Only generate snapshots after this time
    show_progress : bool
        Whether to show progress bar

    Returns
    -------
    pl.DataFrame
        Time series of LOB snapshots with bid/ask prices and sizes
    """
    # Order state tracking - maps order_ref -> {side, price, shares (remaining)}
    submitted_orders: dict[int, dict] = {}

    # Price-level book
    book = {"B": Counter(), "S": Counter()}

    # Combine all messages
    all_messages = []

    # Add orders (A/F)
    for row in add_orders.iter_rows(named=True):
        all_messages.append(
            {
                "timestamp": row["timestamp"],
                "tracking_number": row.get("tracking_number", 0),
                "type": "A",
                "order_ref": row["order_reference_number"],
                "side": row["buy_sell_indicator"],
                "price": row["price"],
                "shares": row["shares"],
            }
        )

    # Delete orders (D)
    for row in deletes.iter_rows(named=True):
        all_messages.append(
            {
                "timestamp": row["timestamp"],
                "tracking_number": row.get("tracking_number", 0),
                "type": "D",
                "order_ref": row["order_reference_number"],
            }
        )

    # Cancel orders (X)
    for row in cancels.iter_rows(named=True):
        all_messages.append(
            {
                "timestamp": row["timestamp"],
                "tracking_number": row.get("tracking_number", 0),
                "type": "X",
                "order_ref": row["order_reference_number"],
                "shares": row["cancelled_shares"],
            }
        )

    # Execute orders (E)
    for row in executions.iter_rows(named=True):
        all_messages.append(
            {
                "timestamp": row["timestamp"],
                "tracking_number": row.get("tracking_number", 0),
                "type": "E",
                "order_ref": row["order_reference_number"],
                "shares": row["executed_shares"],
            }
        )

    # Execute with price (C)
    if executions_c is not None and len(executions_c) > 0:
        for row in executions_c.iter_rows(named=True):
            all_messages.append(
                {
                    "timestamp": row["timestamp"],
                    "tracking_number": row.get("tracking_number", 0),
                    "type": "C",
                    "order_ref": row["order_reference_number"],
                    "shares": row["executed_shares"],
                }
            )

    # Replace orders (U)
    if replaces is not None and len(replaces) > 0:
        for row in replaces.iter_rows(named=True):
            all_messages.append(
                {
                    "timestamp": row["timestamp"],
                    "tracking_number": row.get("tracking_number", 0),
                    "type": "U",
                    "order_ref": row["new_order_reference_number"],
                    "old_order_ref": row["original_order_reference_number"],
                    "side": row.get("original_side"),
                    "price": row["price"],
                    "shares": row["shares"],
                }
            )

    # Sort by (timestamp, tracking_number) to preserve exchange sequence
    all_messages.sort(key=lambda x: (x["timestamp"], x["tracking_number"]))
    print(f"Total messages to process: {len(all_messages):,}")

    # Track statistics
    crossed_count = 0

    # Generate snapshots
    snapshots = []
    last_snapshot_time = None
    freq_map = {
        "1s": timedelta(seconds=1),
        "100ms": timedelta(milliseconds=100),
        "500ms": timedelta(milliseconds=500),
        "5s": timedelta(seconds=5),
        "10s": timedelta(seconds=10),
        "1min": timedelta(minutes=1),
    }
    snapshot_delta = freq_map.get(snapshot_freq, timedelta(seconds=1))

    iterator = tqdm(all_messages, desc="Reconstructing LOB") if show_progress else all_messages

    for msg in iterator:
        ts = msg["timestamp"]
        msg_type = msg["type"]
        order_ref = msg.get("order_ref")

        if msg_type == "A":
            side = msg["side"]
            price = msg["price"]
            shares = msg["shares"]
            submitted_orders[order_ref] = {"side": side, "price": price, "shares": shares}
            book[side][price] += shares

        elif msg_type == "D":
            order = submitted_orders.pop(order_ref, None)
            if order:
                side = order["side"]
                price = order["price"]
                remaining = order["shares"]
                book[side][price] -= remaining
                if book[side][price] <= 0:
                    del book[side][price]

        elif msg_type == "X" or msg_type in ("E", "C"):
            shares = msg["shares"]
            order = submitted_orders.get(order_ref)
            if order:
                order["shares"] -= shares
                book[order["side"]][order["price"]] -= shares
                if book[order["side"]][order["price"]] <= 0:
                    del book[order["side"]][order["price"]]
                if order["shares"] <= 0:
                    submitted_orders.pop(order_ref, None)

        elif msg_type == "U":
            old_ref = msg.get("old_order_ref")
            old_order = submitted_orders.pop(old_ref, None)
            if old_order:
                book[old_order["side"]][old_order["price"]] -= old_order["shares"]
                if book[old_order["side"]][old_order["price"]] <= 0:
                    del book[old_order["side"]][old_order["price"]]

            side = msg.get("side") or (old_order["side"] if old_order else None)
            if side:
                price = msg["price"]
                shares = msg["shares"]
                submitted_orders[order_ref] = {"side": side, "price": price, "shares": shares}
                book[side][price] += shares

        # Check for crossed book
        if book["B"] and book["S"]:
            best_bid = max(book["B"].keys())
            best_ask = min(book["S"].keys())
            if best_bid > best_ask:
                crossed_count += 1

        # Take snapshot at regular intervals
        should_snapshot = snapshot_start is None or ts >= snapshot_start
        if should_snapshot and (
            last_snapshot_time is None or (ts - last_snapshot_time) >= snapshot_delta
        ):
            snapshot = _get_snapshot(book, n_levels, ts)
            if snapshot:
                snapshots.append(snapshot)
            last_snapshot_time = ts

    print(f"Generated {len(snapshots):,} snapshots")
    if crossed_count > 0:
        print(f"WARNING: {crossed_count:,} crossed book states detected")

    if snapshots:
        return pl.DataFrame(snapshots)
    return pl.DataFrame()


# =============================================================================
# Numba-Accelerated LOB Reconstruction with OFI
# =============================================================================

# Message type codes for Numba
MSG_ADD = 0
MSG_DELETE = 1
MSG_CANCEL = 2
MSG_EXECUTE = 3
MSG_REPLACE = 4

# Side codes
SIDE_BID = 0
SIDE_ASK = 1


@numba.jit(nopython=True)
def _numba_reconstruct_lob(
    timestamps: np.ndarray,
    tracking_numbers: np.ndarray,
    msg_types: np.ndarray,
    order_refs: np.ndarray,
    old_order_refs: np.ndarray,
    sides: np.ndarray,
    prices: np.ndarray,
    shares: np.ndarray,
    snapshot_interval_ns: int,
    n_levels: int,
) -> tuple:
    """
    Core Numba kernel for LOB reconstruction with OFI.

    Returns tuple of arrays for post-processing into DataFrame.
    """
    n_messages = len(timestamps)

    # Order state: order_ref -> (side, price_int, shares)
    # price_int = price * 10000 (to avoid float keys)
    order_sides = NumbaDict.empty(key_type=int64, value_type=int64)
    order_prices = NumbaDict.empty(key_type=int64, value_type=int64)
    order_shares = NumbaDict.empty(key_type=int64, value_type=int64)

    # Book state: price_int -> total_shares (separate for bid/ask)
    bid_book = NumbaDict.empty(key_type=int64, value_type=int64)
    ask_book = NumbaDict.empty(key_type=int64, value_type=int64)

    # Pre-allocate snapshot arrays (worst case: one per message)
    max_snapshots = n_messages // 100 + 10000  # Reasonable upper bound
    snap_timestamps = np.zeros(max_snapshots, dtype=np.int64)
    snap_best_bid = np.zeros(max_snapshots, dtype=np.float64)
    snap_best_ask = np.zeros(max_snapshots, dtype=np.float64)
    snap_mid_price = np.zeros(max_snapshots, dtype=np.float64)
    snap_spread = np.zeros(max_snapshots, dtype=np.float64)
    snap_bid_size_0 = np.zeros(max_snapshots, dtype=np.int64)
    snap_ask_size_0 = np.zeros(max_snapshots, dtype=np.int64)

    # OFI accumulators per snapshot interval
    snap_ofi = np.zeros(max_snapshots, dtype=np.float64)
    snap_bid_add = np.zeros(max_snapshots, dtype=np.int64)
    snap_bid_remove = np.zeros(max_snapshots, dtype=np.int64)
    snap_ask_add = np.zeros(max_snapshots, dtype=np.int64)
    snap_ask_remove = np.zeros(max_snapshots, dtype=np.int64)

    # Current interval OFI accumulators
    curr_bid_add = int64(0)
    curr_bid_remove = int64(0)
    curr_ask_add = int64(0)
    curr_ask_remove = int64(0)

    # Tracking
    n_snapshots = 0
    last_snapshot_ts = int64(-1)
    crossed_count = 0

    for i in range(n_messages):
        ts = timestamps[i]
        msg_type = msg_types[i]
        order_ref = order_refs[i]
        price = prices[i]
        share_count = shares[i]
        side = sides[i]

        price_int = int64(price * 10000 + 0.5)  # Round to avoid float issues

        if msg_type == MSG_ADD:
            # Add order to book
            order_sides[order_ref] = side
            order_prices[order_ref] = price_int
            order_shares[order_ref] = share_count

            if side == SIDE_BID:
                if price_int in bid_book:
                    bid_book[price_int] += share_count
                else:
                    bid_book[price_int] = share_count
                curr_bid_add += share_count
            else:
                if price_int in ask_book:
                    ask_book[price_int] += share_count
                else:
                    ask_book[price_int] = share_count
                curr_ask_add += share_count

        elif msg_type == MSG_DELETE:
            # Full deletion - remove remaining shares
            if order_ref in order_sides:
                o_side = order_sides[order_ref]
                o_price = order_prices[order_ref]
                o_shares = order_shares[order_ref]

                if o_side == SIDE_BID:
                    if o_price in bid_book:
                        bid_book[o_price] -= o_shares
                        if bid_book[o_price] <= 0:
                            del bid_book[o_price]
                    curr_bid_remove += o_shares
                else:
                    if o_price in ask_book:
                        ask_book[o_price] -= o_shares
                        if ask_book[o_price] <= 0:
                            del ask_book[o_price]
                    curr_ask_remove += o_shares

                del order_sides[order_ref]
                del order_prices[order_ref]
                del order_shares[order_ref]

        elif msg_type in (MSG_CANCEL, MSG_EXECUTE):
            # Partial cancellation or execution
            if order_ref in order_sides:
                o_side = order_sides[order_ref]
                o_price = order_prices[order_ref]

                order_shares[order_ref] -= share_count

                if o_side == SIDE_BID:
                    if o_price in bid_book:
                        bid_book[o_price] -= share_count
                        if bid_book[o_price] <= 0:
                            del bid_book[o_price]
                    curr_bid_remove += share_count
                else:
                    if o_price in ask_book:
                        ask_book[o_price] -= share_count
                        if ask_book[o_price] <= 0:
                            del ask_book[o_price]
                    curr_ask_remove += share_count

                if order_shares[order_ref] <= 0:
                    del order_sides[order_ref]
                    del order_prices[order_ref]
                    del order_shares[order_ref]

        elif msg_type == MSG_REPLACE:
            # Replace: delete old order, add new
            old_ref = old_order_refs[i]
            if old_ref in order_sides:
                o_side = order_sides[old_ref]
                o_price = order_prices[old_ref]
                o_shares = order_shares[old_ref]

                if o_side == SIDE_BID:
                    if o_price in bid_book:
                        bid_book[o_price] -= o_shares
                        if bid_book[o_price] <= 0:
                            del bid_book[o_price]
                    curr_bid_remove += o_shares
                else:
                    if o_price in ask_book:
                        ask_book[o_price] -= o_shares
                        if ask_book[o_price] <= 0:
                            del ask_book[o_price]
                    curr_ask_remove += o_shares

                # New order inherits side from old order
                order_sides[order_ref] = o_side
                order_prices[order_ref] = price_int
                order_shares[order_ref] = share_count

                if o_side == SIDE_BID:
                    if price_int in bid_book:
                        bid_book[price_int] += share_count
                    else:
                        bid_book[price_int] = share_count
                    curr_bid_add += share_count
                else:
                    if price_int in ask_book:
                        ask_book[price_int] += share_count
                    else:
                        ask_book[price_int] = share_count
                    curr_ask_add += share_count

                del order_sides[old_ref]
                del order_prices[old_ref]
                del order_shares[old_ref]

        # Check for crossed book
        if len(bid_book) > 0 and len(ask_book) > 0:
            best_bid_int = int64(0)
            for p in bid_book:
                if p > best_bid_int:
                    best_bid_int = p
            best_ask_int = int64(9999999999)
            for p in ask_book:
                if p < best_ask_int:
                    best_ask_int = p
            if best_bid_int > best_ask_int:
                crossed_count += 1

        # Take snapshot at regular intervals
        if last_snapshot_ts < 0 or (ts - last_snapshot_ts) >= snapshot_interval_ns:
            if len(bid_book) > 0 and len(ask_book) > 0:
                # Find best bid and ask
                best_bid_int = int64(0)
                best_bid_size = int64(0)
                for p in bid_book:
                    if p > best_bid_int:
                        best_bid_int = p
                        best_bid_size = bid_book[p]

                best_ask_int = int64(9999999999)
                best_ask_size = int64(0)
                for p in ask_book:
                    if p < best_ask_int:
                        best_ask_int = p
                        best_ask_size = ask_book[p]

                # Convert back to float prices
                best_bid = best_bid_int / 10000.0
                best_ask = best_ask_int / 10000.0

                # Compute OFI for this interval
                ofi = float64(curr_bid_add - curr_bid_remove) - float64(
                    curr_ask_add - curr_ask_remove
                )

                # Record snapshot
                snap_timestamps[n_snapshots] = ts
                snap_best_bid[n_snapshots] = best_bid
                snap_best_ask[n_snapshots] = best_ask
                snap_mid_price[n_snapshots] = (best_bid + best_ask) / 2.0
                snap_spread[n_snapshots] = best_ask - best_bid
                snap_bid_size_0[n_snapshots] = best_bid_size
                snap_ask_size_0[n_snapshots] = best_ask_size
                snap_ofi[n_snapshots] = ofi
                snap_bid_add[n_snapshots] = curr_bid_add
                snap_bid_remove[n_snapshots] = curr_bid_remove
                snap_ask_add[n_snapshots] = curr_ask_add
                snap_ask_remove[n_snapshots] = curr_ask_remove

                n_snapshots += 1
                last_snapshot_ts = ts

                # Reset OFI accumulators for next interval
                curr_bid_add = int64(0)
                curr_bid_remove = int64(0)
                curr_ask_add = int64(0)
                curr_ask_remove = int64(0)

    return (
        snap_timestamps[:n_snapshots],
        snap_best_bid[:n_snapshots],
        snap_best_ask[:n_snapshots],
        snap_mid_price[:n_snapshots],
        snap_spread[:n_snapshots],
        snap_bid_size_0[:n_snapshots],
        snap_ask_size_0[:n_snapshots],
        snap_ofi[:n_snapshots],
        snap_bid_add[:n_snapshots],
        snap_bid_remove[:n_snapshots],
        snap_ask_add[:n_snapshots],
        snap_ask_remove[:n_snapshots],
        crossed_count,
    )


def reconstruct_lob_with_ofi(
    add_orders: pl.DataFrame,
    deletes: pl.DataFrame,
    cancels: pl.DataFrame,
    executions: pl.DataFrame,
    executions_c: pl.DataFrame | None = None,
    replaces: pl.DataFrame | None = None,
    n_levels: int = 10,
    snapshot_freq: str = "1s",
    show_progress: bool = True,
) -> pl.DataFrame:
    """
    Numba-accelerated LOB reconstruction with OFI computation.

    ~10-50x faster than Python version for large message sets.
    Computes Order Flow Imbalance (OFI) during the reconstruction pass.

    OFI = (Bid Adds - Bid Removes) - (Ask Adds - Ask Removes)

    This captures the net order flow pressure: positive OFI indicates
    buying pressure (more bids added/asks removed), negative indicates
    selling pressure.

    Parameters
    ----------
    add_orders : pl.DataFrame
        Combined A and F messages with add orders
    deletes : pl.DataFrame
        D messages (full order deletion)
    cancels : pl.DataFrame
        X messages (partial cancellation)
    executions : pl.DataFrame
        E messages (order executions)
    executions_c : pl.DataFrame, optional
        C messages (order executed with price)
    replaces : pl.DataFrame, optional
        U messages (order replacements)
    n_levels : int
        Number of price levels to track (currently returns top-of-book)
    snapshot_freq : str
        Frequency for LOB snapshots (e.g., '1s', '100ms', '1min')
    show_progress : bool
        Whether to show progress (pre-processing only, Numba is fast)

    Returns
    -------
    pl.DataFrame
        Time series of LOB snapshots with columns:
        - timestamp: Snapshot time
        - best_bid, best_ask, mid_price, spread: Quote data
        - bid_size_0, ask_size_0: Top-of-book depth
        - ofi: Order Flow Imbalance for the interval
        - bid_add, bid_remove, ask_add, ask_remove: OFI components
    """
    # Parse snapshot frequency to nanoseconds
    freq_ns_map = {
        "100ms": 100_000_000,
        "500ms": 500_000_000,
        "1s": 1_000_000_000,
        "5s": 5_000_000_000,
        "10s": 10_000_000_000,
        "1min": 60_000_000_000,
    }
    snapshot_interval_ns = freq_ns_map.get(snapshot_freq, 1_000_000_000)

    # Build message arrays
    if show_progress:
        print("Preparing messages for Numba kernel...")

    # Count total messages for pre-allocation
    n_add = len(add_orders)
    n_del = len(deletes) if deletes is not None else 0
    n_cancel = len(cancels) if cancels is not None else 0
    n_exec = len(executions) if executions is not None else 0
    n_exec_c = len(executions_c) if executions_c is not None and len(executions_c) > 0 else 0
    n_replace = len(replaces) if replaces is not None and len(replaces) > 0 else 0
    n_total = n_add + n_del + n_cancel + n_exec + n_exec_c + n_replace

    # Pre-allocate arrays
    timestamps = np.zeros(n_total, dtype=np.int64)
    tracking_numbers = np.zeros(n_total, dtype=np.int64)
    msg_types = np.zeros(n_total, dtype=np.int64)
    order_refs = np.zeros(n_total, dtype=np.int64)
    old_order_refs = np.zeros(n_total, dtype=np.int64)
    sides = np.zeros(n_total, dtype=np.int64)
    prices = np.zeros(n_total, dtype=np.float64)
    shares_arr = np.zeros(n_total, dtype=np.int64)

    idx = 0

    # Add orders (A/F)
    if n_add > 0:
        ts_col = add_orders["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        timestamps[idx : idx + n_add] = ts_col
        tracking_numbers[idx : idx + n_add] = (
            add_orders["tracking_number"].to_numpy()
            if "tracking_number" in add_orders.columns
            else np.zeros(n_add, dtype=np.int64)
        )
        msg_types[idx : idx + n_add] = MSG_ADD
        order_refs[idx : idx + n_add] = add_orders["order_reference_number"].to_numpy()
        # Convert side: 'B' -> 0, 'S' -> 1
        side_strs = add_orders["buy_sell_indicator"].to_list()
        sides[idx : idx + n_add] = np.array([SIDE_BID if s == "B" else SIDE_ASK for s in side_strs])
        prices[idx : idx + n_add] = add_orders["price"].to_numpy()
        shares_arr[idx : idx + n_add] = add_orders["shares"].to_numpy()
        idx += n_add

    # Delete orders (D)
    if n_del > 0:
        ts_col = deletes["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        timestamps[idx : idx + n_del] = ts_col
        tracking_numbers[idx : idx + n_del] = (
            deletes["tracking_number"].to_numpy()
            if "tracking_number" in deletes.columns
            else np.zeros(n_del, dtype=np.int64)
        )
        msg_types[idx : idx + n_del] = MSG_DELETE
        order_refs[idx : idx + n_del] = deletes["order_reference_number"].to_numpy()
        idx += n_del

    # Cancel orders (X)
    if n_cancel > 0:
        ts_col = cancels["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        timestamps[idx : idx + n_cancel] = ts_col
        tracking_numbers[idx : idx + n_cancel] = (
            cancels["tracking_number"].to_numpy()
            if "tracking_number" in cancels.columns
            else np.zeros(n_cancel, dtype=np.int64)
        )
        msg_types[idx : idx + n_cancel] = MSG_CANCEL
        order_refs[idx : idx + n_cancel] = cancels["order_reference_number"].to_numpy()
        shares_arr[idx : idx + n_cancel] = cancels["cancelled_shares"].to_numpy()
        idx += n_cancel

    # Execute orders (E)
    if n_exec > 0:
        ts_col = executions["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        timestamps[idx : idx + n_exec] = ts_col
        tracking_numbers[idx : idx + n_exec] = (
            executions["tracking_number"].to_numpy()
            if "tracking_number" in executions.columns
            else np.zeros(n_exec, dtype=np.int64)
        )
        msg_types[idx : idx + n_exec] = MSG_EXECUTE
        order_refs[idx : idx + n_exec] = executions["order_reference_number"].to_numpy()
        shares_arr[idx : idx + n_exec] = executions["executed_shares"].to_numpy()
        idx += n_exec

    # Execute with price (C)
    if n_exec_c > 0:
        ts_col = executions_c["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        timestamps[idx : idx + n_exec_c] = ts_col
        tracking_numbers[idx : idx + n_exec_c] = (
            executions_c["tracking_number"].to_numpy()
            if "tracking_number" in executions_c.columns
            else np.zeros(n_exec_c, dtype=np.int64)
        )
        msg_types[idx : idx + n_exec_c] = MSG_EXECUTE
        order_refs[idx : idx + n_exec_c] = executions_c["order_reference_number"].to_numpy()
        shares_arr[idx : idx + n_exec_c] = executions_c["executed_shares"].to_numpy()
        idx += n_exec_c

    # Replace orders (U)
    if n_replace > 0:
        ts_col = replaces["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        timestamps[idx : idx + n_replace] = ts_col
        tracking_numbers[idx : idx + n_replace] = (
            replaces["tracking_number"].to_numpy()
            if "tracking_number" in replaces.columns
            else np.zeros(n_replace, dtype=np.int64)
        )
        msg_types[idx : idx + n_replace] = MSG_REPLACE
        order_refs[idx : idx + n_replace] = replaces["new_order_reference_number"].to_numpy()
        old_order_refs[idx : idx + n_replace] = replaces[
            "original_order_reference_number"
        ].to_numpy()
        prices[idx : idx + n_replace] = replaces["price"].to_numpy()
        shares_arr[idx : idx + n_replace] = replaces["shares"].to_numpy()
        idx += n_replace

    # Sort by (timestamp, tracking_number)
    sort_idx = np.lexsort((tracking_numbers, timestamps))
    timestamps = timestamps[sort_idx]
    tracking_numbers = tracking_numbers[sort_idx]
    msg_types = msg_types[sort_idx]
    order_refs = order_refs[sort_idx]
    old_order_refs = old_order_refs[sort_idx]
    sides = sides[sort_idx]
    prices = prices[sort_idx]
    shares_arr = shares_arr[sort_idx]

    if show_progress:
        print(f"Processing {n_total:,} messages with Numba...")

    # Run Numba kernel
    result = _numba_reconstruct_lob(
        timestamps,
        tracking_numbers,
        msg_types,
        order_refs,
        old_order_refs,
        sides,
        prices,
        shares_arr,
        snapshot_interval_ns,
        n_levels,
    )

    (
        snap_ts,
        snap_best_bid,
        snap_best_ask,
        snap_mid,
        snap_spread,
        snap_bid_size,
        snap_ask_size,
        snap_ofi,
        snap_bid_add,
        snap_bid_remove,
        snap_ask_add,
        snap_ask_remove,
        crossed_count,
    ) = result

    if show_progress:
        print(f"Generated {len(snap_ts):,} snapshots")
        if crossed_count > 0:
            print(f"WARNING: {crossed_count:,} crossed book states detected")

    if len(snap_ts) == 0:
        return pl.DataFrame()

    # Convert timestamps back to datetime
    timestamps_dt = snap_ts.astype("datetime64[ns]")

    # Build DataFrame
    df = pl.DataFrame(
        {
            "timestamp": timestamps_dt,
            "best_bid": snap_best_bid,
            "best_ask": snap_best_ask,
            "mid_price": snap_mid,
            "spread": snap_spread,
            "bid_size_0": snap_bid_size,
            "ask_size_0": snap_ask_size,
            "ofi": snap_ofi,
            "bid_add": snap_bid_add,
            "bid_remove": snap_bid_remove,
            "ask_add": snap_ask_add,
            "ask_remove": snap_ask_remove,
        }
    )

    return df


def classify_trades_lee_ready(
    itch_dir: Path,
    symbol: str,
    start_time: datetime = None,
    end_time: datetime = None,
    show_progress: bool = True,
) -> pl.DataFrame:
    """
    Classify trade direction using Lee-Ready algorithm with proper LOB reconstruction.

    Lee-Ready (1991) classifies trades by comparing trade price to quote midpoint:
    1. Quote test: price > midpoint → buy, price < midpoint → sell
    2. Tick test (fallback): uptick → buy, downtick → sell

    This function maintains LOB state while processing trades, using the same
    correct reconstruction logic as reconstruct_lob().

    Parameters
    ----------
    itch_dir : Path
        Directory containing parsed ITCH message subdirectories
    symbol : str
        Stock symbol (e.g., "AAPL")
    start_time : datetime, optional
        Filter to trades >= this time
    end_time : datetime, optional
        Filter to trades <= this time
    show_progress : bool
        Whether to show progress bar

    Returns
    -------
    pl.DataFrame
        Trades with columns: timestamp, price, shares, side
        where side is 1 (buy), -1 (sell), or 0 (at midpoint/unknown)
    """
    # Load all messages for this symbol
    messages = load_messages_for_symbol(itch_dir, symbol, start_time=start_time, end_time=end_time)

    # Combine A and F adds (F messages have extra 'attribution' column we don't need)
    add_a = messages.get("A", pl.DataFrame())
    add_f = messages.get("F", pl.DataFrame())
    if len(add_f) > 0 and "attribution" in add_f.columns:
        add_f = add_f.drop("attribution")
    if len(add_a) > 0 and len(add_f) > 0:
        add_orders = pl.concat([add_a, add_f])
    elif len(add_a) > 0:
        add_orders = add_a
    elif len(add_f) > 0:
        add_orders = add_f
    else:
        print("No add orders found")
        return pl.DataFrame()

    trades = messages.get("P", pl.DataFrame())
    if len(trades) == 0:
        print("No trades found")
        return pl.DataFrame()

    # Build unified message stream for LOB + trades
    all_messages = []

    # Add orders
    for row in add_orders.iter_rows(named=True):
        all_messages.append(
            {
                "timestamp": row["timestamp"],
                "tracking_number": row.get("tracking_number", 0),
                "type": "A",
                "order_ref": row["order_reference_number"],
                "side": row["buy_sell_indicator"],
                "price": row["price"],
                "shares": row["shares"],
            }
        )

    # Delete orders
    for row in messages.get("D", pl.DataFrame()).iter_rows(named=True):
        all_messages.append(
            {
                "timestamp": row["timestamp"],
                "tracking_number": row.get("tracking_number", 0),
                "type": "D",
                "order_ref": row["order_reference_number"],
            }
        )

    # Cancel orders
    for row in messages.get("X", pl.DataFrame()).iter_rows(named=True):
        all_messages.append(
            {
                "timestamp": row["timestamp"],
                "tracking_number": row.get("tracking_number", 0),
                "type": "X",
                "order_ref": row["order_reference_number"],
                "shares": row["cancelled_shares"],
            }
        )

    # Execute orders
    for row in messages.get("E", pl.DataFrame()).iter_rows(named=True):
        all_messages.append(
            {
                "timestamp": row["timestamp"],
                "tracking_number": row.get("tracking_number", 0),
                "type": "E",
                "order_ref": row["order_reference_number"],
                "shares": row["executed_shares"],
            }
        )

    # Execute with price
    for row in messages.get("C", pl.DataFrame()).iter_rows(named=True):
        all_messages.append(
            {
                "timestamp": row["timestamp"],
                "tracking_number": row.get("tracking_number", 0),
                "type": "C",
                "order_ref": row["order_reference_number"],
                "shares": row["executed_shares"],
            }
        )

    # Replace orders
    for row in messages.get("U", pl.DataFrame()).iter_rows(named=True):
        all_messages.append(
            {
                "timestamp": row["timestamp"],
                "tracking_number": row.get("tracking_number", 0),
                "type": "U",
                "order_ref": row["new_order_reference_number"],
                "old_order_ref": row["original_order_reference_number"],
                "side": row.get("original_side"),
                "price": row["price"],
                "shares": row["shares"],
            }
        )

    # Trades (P messages)
    for row in trades.iter_rows(named=True):
        all_messages.append(
            {
                "timestamp": row["timestamp"],
                "tracking_number": row.get("tracking_number", 0),
                "type": "P",
                "price": row["price"],
                "shares": row["shares"],
            }
        )

    # Sort by (timestamp, tracking_number) to preserve exchange sequence
    all_messages.sort(key=lambda x: (x["timestamp"], x["tracking_number"]))
    print(f"Processing {len(all_messages):,} messages for Lee-Ready classification...")

    # Process messages and classify trades
    submitted_orders: dict[int, dict] = {}
    book = {"B": Counter(), "S": Counter()}
    classified_trades = []
    last_price = None
    last_tick_dir = 0

    iterator = tqdm(all_messages, desc="Lee-Ready") if show_progress else all_messages

    for msg in iterator:
        msg_type = msg["type"]

        if msg_type == "A":
            order_ref = msg["order_ref"]
            side = msg["side"]
            price = msg["price"]
            shares = msg["shares"]
            submitted_orders[order_ref] = {"side": side, "price": price, "shares": shares}
            book[side][price] += shares

        elif msg_type == "D":
            order_ref = msg["order_ref"]
            order = submitted_orders.pop(order_ref, None)
            if order:
                book[order["side"]][order["price"]] -= order["shares"]
                if book[order["side"]][order["price"]] <= 0:
                    del book[order["side"]][order["price"]]

        elif msg_type == "X" or msg_type in ("E", "C"):
            order_ref = msg["order_ref"]
            shares = msg["shares"]
            order = submitted_orders.get(order_ref)
            if order:
                order["shares"] -= shares
                book[order["side"]][order["price"]] -= shares
                if book[order["side"]][order["price"]] <= 0:
                    del book[order["side"]][order["price"]]
                if order["shares"] <= 0:
                    submitted_orders.pop(order_ref, None)

        elif msg_type == "U":
            old_ref = msg.get("old_order_ref")
            order_ref = msg["order_ref"]
            old_order = submitted_orders.pop(old_ref, None)
            if old_order:
                book[old_order["side"]][old_order["price"]] -= old_order["shares"]
                if book[old_order["side"]][old_order["price"]] <= 0:
                    del book[old_order["side"]][old_order["price"]]

            side = msg.get("side") or (old_order["side"] if old_order else None)
            if side:
                price = msg["price"]
                shares = msg["shares"]
                submitted_orders[order_ref] = {"side": side, "price": price, "shares": shares}
                book[side][price] += shares

        elif msg_type == "P":
            # Trade - classify using Lee-Ready
            trade_price = msg["price"]
            trade_shares = msg["shares"]
            trade_ts = msg["timestamp"]

            # Get current midpoint
            if book["B"] and book["S"]:
                best_bid = max(book["B"].keys())
                best_ask = min(book["S"].keys())
                midpoint = (best_bid + best_ask) / 2

                # Quote test
                if trade_price > midpoint:
                    side = 1  # Buy
                elif trade_price < midpoint:
                    side = -1  # Sell
                else:
                    # At midpoint - use tick test
                    if last_price is not None:
                        if trade_price > last_price:
                            last_tick_dir = 1
                        elif trade_price < last_price:
                            last_tick_dir = -1
                    side = last_tick_dir
            else:
                # No book - use tick test only
                if last_price is not None:
                    if trade_price > last_price:
                        side = 1
                    elif trade_price < last_price:
                        side = -1
                    else:
                        side = last_tick_dir
                else:
                    side = 0

            classified_trades.append(
                {
                    "timestamp": trade_ts,
                    "price": trade_price,
                    "shares": trade_shares,
                    "side": side,
                }
            )
            last_price = trade_price

    print(f"Classified {len(classified_trades):,} trades")

    if classified_trades:
        result = pl.DataFrame(classified_trades)
        # Report classification breakdown
        buy_count = (result["side"] == 1).sum()
        sell_count = (result["side"] == -1).sum()
        unknown_count = (result["side"] == 0).sum()
        print(f"  Buys: {buy_count:,} ({100 * buy_count / len(result):.1f}%)")
        print(f"  Sells: {sell_count:,} ({100 * sell_count / len(result):.1f}%)")
        if unknown_count > 0:
            print(f"  Unknown: {unknown_count:,} ({100 * unknown_count / len(result):.1f}%)")
        return result

    return pl.DataFrame()
