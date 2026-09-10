# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # NASDAQ TotalView-ITCH: Order Book Data Parsing
#
# **Chapter 3: Market Microstructure**
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# This notebook demonstrates how to parse NASDAQ's TotalView-ITCH binary protocol.
# Understanding MBO (message-by-order) data is foundational for microstructure-based ML features.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
# - Read one ITCH v5.0 message out of a binary file: find where it starts, how long it is,
#   and which of the twenty-odd message types it is.
# - Unpack a message's fields with Python's `struct` module, and convert the two encodings
#   ITCH uses - a nanosecond offset from midnight, and a price held as an integer with four
#   implied decimal places.
# - Run the parse over a full trading day, writing each message type to its own Parquet
#   partition so that a day that does not fit in memory still lands on disk.
# - Load an already-parsed day and check that every message type reads back.
# - Say when the Python parser is the right tool and when the Rust one is.
#
# ## Book reference
#
# Section §3.3, *From raw messages to the limit order book* - the binary-parsing-at-scale
# subsection and Table 3.1. §3.2 places ITCH among the other feeds.
#
# ## Cross-References
#
# - **Downstream**: `02_itch_lob_reconstruction` (builds order book from these messages)
# - **Related**: `09_databento_mbo_analysis` (alternative MBO data source)
#
# ## Data Requirements
#
# ITCH sample data can be downloaded using:
# ```bash
# python data/equities/market/microstructure/nasdaq_itch_download.py --list   # List available files
# python data/equities/market/microstructure/nasdaq_itch_download.py          # Download default sample
# ```
#
# Files are ~5GB compressed from: https://emi.nasdaq.com/ITCH/

# %% [markdown]
# ## ITCH Message Types
#
# The [ITCH v5.0 specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf) defines 20+ message types:
#
# | Type | Name | Description |
# |------|------|-------------|
# | **S** | System Event | Market open/close events |
# | **R** | Stock Directory | Ticker information and characteristics |
# | **H** | Trading Action | Trading halts, pauses, and resumptions |
# | **Y** | Reg SHO Restriction | Short sale price test restrictions |
# | **L** | Market Participant | Market maker positions |
# | **V** | MWCB Decline Level | Market-wide circuit breaker levels |
# | **W** | MWCB Status | Circuit breaker breach status |
# | **A** | Add Order | New limit order enters the book |
# | **F** | Add Order (MPID) | Same as A, with market participant ID |
# | **E** | Order Executed | Partial/full execution against standing order |
# | **C** | Order Executed w/Price | Execution at different price (hidden orders) |
# | **X** | Order Cancel | Partial cancellation |
# | **D** | Order Delete | Full removal from book |
# | **U** | Order Replace | Modify price/size (cancel + add) |
# | **P** | Trade | Non-displayed execution |
# | **Q** | Cross Trade | Opening/closing cross |
# | **B** | Broken Trade | Trade cancellation |
# | **I** | NOII | Net Order Imbalance Indicator (auction) |
# | **J** | LULD Auction Collar | Limit up-limit down price bands |
# | **K** | IPO Quoting Period | IPO quotation timing |
#
# By combining these messages chronologically, we can reconstruct the order book at any point in time.

# %%
"""NASDAQ TotalView-ITCH: Order Book Data Parsing — parse ITCH binary protocol into structured messages."""

import gzip
import os
import shutil
import struct
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from time import time

import polars as pl
from itch_message_specs import (
    FMT_DICT,
    MESSAGE_SPECS,
    NT_DICT,
    flush_to_parquet,
    parse_price4,
    parse_timestamp,
    print_message_formats,
)
from tqdm.auto import tqdm

from data import load_nasdaq_itch
from utils.paths import display_path

# %% tags=["parameters"]
SKIP_PARSING = False

# %% [markdown]
# The parse reads one directory and writes another. `load_nasdaq_itch(get_base_path=True)`
# resolves the parsed-message root from the repository's data configuration rather than a
# path typed here, so the same notebook runs against a local checkout and inside the Docker
# image. The raw binary sits in a `raw/` directory beside it, which is where the download
# script puts it.

# %%
MESSAGE_DIR = load_nasdaq_itch(get_base_path=True)
MESSAGE_DIR.mkdir(parents=True, exist_ok=True)
ITCH_RAW_DIR = MESSAGE_DIR.parent / "raw"

print(f"Raw ITCH binary (input):  {display_path(ITCH_RAW_DIR)}")
print(f"Parsed messages (output): {display_path(MESSAGE_DIR)}")
raw_present = ITCH_RAW_DIR.exists() and next(ITCH_RAW_DIR.iterdir(), None) is not None
print(f"Raw binary present: {raw_present}")
if not raw_present:
    print(
        "  Fetch it first: "
        "uv run python data/equities/market/microstructure/nasdaq_itch_download.py"
    )

# %% [markdown]
# ## 1. Message Specifications
#
# Each ITCH message has a fixed binary structure. The format is defined in `itch_message_specs.py`
# using Python's `struct` module. Format codes:
# - `H` = unsigned short (2 bytes)
# - `I` = unsigned int (4 bytes)
# - `Q` = unsigned long long (8 bytes)
# - `s` = char (1 byte), `Ns` = N chars
# - `>` = big-endian byte order

# %%
# Show message formats (loaded from utils/itch_message_specs.py)
print_message_formats()

# %% [markdown]
# ## 2. Binary Parsing Example
#
# Let's demonstrate how binary parsing works by creating and parsing a sample Add Order message.

# %% [markdown]
# Packing a message by hand is the quickest way to see the layout. An Add Order carries the
# format `>HH6sQsI8sI`: big-endian, two unsigned shorts, a six-byte timestamp, an eight-byte
# order reference, a one-character side, the share count, an eight-character padded ticker
# and the price. Unpacking it below returns exactly these fields.

# %%
sample_add_order = struct.pack(
    ">HH6sQsI8sI",
    1234,  # stock_locate
    5678,  # tracking_number
    b"\x00\x00\x00\x00\x00\x01",  # timestamp (1 nanosecond)
    9876543210,  # order_reference_number
    b"B",  # buy_sell_indicator
    100,  # shares
    b"AAPL    ",  # stock (padded to 8 chars)
    1500000,  # price: four implied decimals, so this is $150
)

print(f"Raw Add Order message ({len(sample_add_order)} bytes):")
print(f"  Hex: {sample_add_order.hex()}")

# %%
# Parse the binary data using our struct format
parsed = struct.unpack(FMT_DICT["A"], sample_add_order)
add_order = NT_DICT["A"]._make(parsed)

print("Parsed Add Order Message:")
print("-" * 40)
for field, value in add_order._asdict().items():
    if isinstance(value, bytes):
        value = value.decode("ascii").strip()
    print(f"  {field:25}: {value}")

# %%
# Apply conversions using helper functions from utils.itch_message_specs
ts_ns = parse_timestamp(add_order.timestamp)
price = parse_price4(add_order.price)

print(f"Timestamp: {ts_ns:,} nanoseconds = {ts_ns / 1e9:.9f} seconds after midnight")
print(f"Price: ${price:.4f}")

# %% [markdown]
# ## 3. Loading Pre-Parsed ITCH Data
#
# If you've already parsed ITCH data (using the Rust parser or Python parser), you can load
# the pre-parsed messages directly. This is the recommended approach for analysis.

# %%
# Check what data is available locally
print("ITCH Data Pipeline Status:")
print("-" * 50)

# Step 1: Raw binary from download
raw_files = []
if ITCH_RAW_DIR.exists():
    raw_files = list(ITCH_RAW_DIR.glob("*.gz")) + list(ITCH_RAW_DIR.glob("*.bin"))
print(f"Raw binary files: {len(raw_files)}")
for f in raw_files:
    print(f"  {f.name} ({f.stat().st_size / 1e9:.2f} GB)")

# Step 2: Parsed messages (single uppercase letter = message type)
parsed_types = (
    [
        d
        for d in sorted(MESSAGE_DIR.iterdir())
        if d.is_dir() and len(d.name) == 1 and d.name.isupper()
    ]
    if MESSAGE_DIR.exists()
    else []
)
parsed_with_data = [d for d in parsed_types if list(d.glob("*.parquet"))]
print(f"Parsed message types: {len(parsed_with_data)}")
for msg_dir in parsed_with_data:
    name = MESSAGE_SPECS.get(msg_dir.name, {}).get("name", "Unknown")
    n_files = len(list(msg_dir.glob("*.parquet")))
    print(f"  {msg_dir.name} ({name}): {n_files} files")

# %%
# Validate: at minimum we need parsed data to continue
trade_dir = MESSAGE_DIR / "P"
assert trade_dir.exists() and list(trade_dir.glob("*.parquet")), (
    f"No parsed ITCH data at {MESSAGE_DIR}.\n"
    "To set up the data pipeline:\n"
    "  1. Download raw data:  uv run python data/equities/market/microstructure/nasdaq_itch_download.py\n"
    "  2. Parse (this notebook, Section 4) or use Rust parser (Section 6)\n"
    "  3. Parsed messages go to: data/equities/market/microstructure/nasdaq_itch/messages/"
)

trades = pl.read_parquet(trade_dir / "*.parquet")
print(f"\nLoaded {len(trades):,} trade messages")
print(f"Columns: {trades.columns}")

# %% [markdown]
# ## 4. Full Parser Implementation
#
# The parser below is written to be read. It processes one message at a time in Python, which
# is what makes each step visible and also what makes a full trading day take about twenty
# minutes. Section 6 covers the Rust parser, which emits the same Parquet schema and is what
# you would run over many days.


# %% [markdown]
# ### Parser Helpers
#
# The parse splits into three functions. `read_frame` takes the next message off the file
# and hands back its type and its raw bytes; `decode_message` turns those bytes into a
# dictionary of Python values; and `parse_itch_file` runs the loop, buffering decoded
# messages and writing them out in batches.


# %%
def read_frame(f, pbar) -> tuple[str, bytes] | None:
    """Read one ITCH message frame: 2-byte length + 1-byte type + payload.

    Returns (msg_type, payload) on success, or None on EOF/truncation.
    """
    # 2-byte big-endian length prefix (message size including type byte)
    length_bytes = f.read(2)
    if len(length_bytes) < 2:
        return None
    pbar.update(2)

    msg_size = int.from_bytes(length_bytes, "big")

    # 1-byte message type
    msg_type_byte = f.read(1)
    if len(msg_type_byte) < 1:
        print(f"\nWarning: Truncated message at byte {f.tell()}, expected type byte")
        return None
    pbar.update(1)

    msg_type = msg_type_byte.decode("ascii")

    # Payload (msg_size includes type byte, so payload is msg_size - 1)
    payload = f.read(msg_size - 1)
    if len(payload) < msg_size - 1:
        print(f"\nWarning: Truncated payload for message type {msg_type}")
        return None
    pbar.update(msg_size - 1)

    return msg_type, payload


# %% [markdown]
# Decode binary payload into a Python dict, converting raw timestamp bytes
# to nanosecond integers and byte strings to stripped ASCII.


# %%
def decode_message(msg_type: str, payload: bytes) -> dict | None:
    """Unpack binary payload into a dict, converting timestamps and strings.

    Returns parsed message dict, or None on struct error.
    """
    try:
        parsed = struct.unpack(FMT_DICT[msg_type], payload)
        msg = NT_DICT[msg_type]._make(parsed)._asdict()
    except struct.error:
        return None

    # Convert timestamp: nanoseconds since midnight
    if "timestamp" in msg:
        msg["timestamp"] = int.from_bytes(msg["timestamp"], "big")

    # Decode string fields
    for field, value in msg.items():
        if isinstance(value, bytes):
            msg[field] = value.decode("ascii").strip()

    return msg


# %% [markdown]
# The main parser reads the binary file sequentially, buffering decoded messages
# and flushing to Parquet periodically to bound memory usage.


# %%
def parse_itch_file(
    itch_file: Path,
    trading_day: date,
    output_dir: Path,
    max_buffered_messages: int = 10_000_000,
    max_messages: int | None = None,
) -> dict[str, int]:
    """Parse ITCH binary file and store messages as Parquet.

    Args:
        itch_file: Path to binary ITCH file (.bin, not .gz).
        trading_day: Trading date for timestamp construction.
        output_dir: Directory for Parquet output (one subdir per message type).
        max_buffered_messages: Flush threshold (total buffered messages).
        max_messages: Optional limit for testing.

    Returns:
        Dictionary with message type counts.
    """
    # Midnight timestamp for the trading day (ITCH timestamps are nanoseconds offset)
    base_ts = datetime(trading_day.year, trading_day.month, trading_day.day)
    file_counters: dict[str, int] = defaultdict(int)

    buffers = defaultdict(list)
    counts = Counter()
    file_size = itch_file.stat().st_size
    start_time = time()

    with (
        itch_file.open("rb") as f,
        tqdm(total=file_size, desc="Parsing ITCH", unit="B", unit_scale=True) as pbar,
    ):
        while True:
            if max_messages and sum(counts.values()) >= max_messages:
                print(f"\nLimit reached: {max_messages:,} messages")
                break

            frame = read_frame(f, pbar)
            if frame is None:
                break
            msg_type, payload = frame
            counts[msg_type] += 1

            if msg_type not in FMT_DICT:
                continue

            msg = decode_message(msg_type, payload)
            if msg is None:
                continue

            # Check for end of messages
            if msg_type == "S" and msg.get("event_code") == "C":
                print("\nEnd of Messages")
                flush_to_parquet(buffers, output_dir, base_ts, file_counters)
                break

            buffers[msg_type].append(msg)

            # Periodic flush
            if sum(len(v) for v in buffers.values()) >= max_buffered_messages:
                flush_to_parquet(buffers, output_dir, base_ts, file_counters)

    # Final flush
    if any(buffers.values()):
        flush_to_parquet(buffers, output_dir, base_ts, file_counters)

    elapsed = time() - start_time
    total = sum(counts.values())
    print(f"Parsed {total:,} messages in {elapsed:.1f}s ({total / elapsed:,.0f} msg/s)")

    return dict(counts)


# %%
# Locate ITCH data file (skip if SKIP_PARSING is set)
if SKIP_PARSING:
    print("SKIP_PARSING=True: skipping ITCH binary parsing (uses pre-parsed data)")
    itch_file = None
    counts = {}
else:
    # Clear any existing parsed data to avoid schema conflicts
    # (Different parser versions may produce different schemas)
    # Set ITCH_KEEP_EXISTING=1 to skip cleanup and use existing data
    clear_existing = os.environ.get("ITCH_KEEP_EXISTING", "0") != "1"
    if clear_existing and MESSAGE_DIR.exists() and list(MESSAGE_DIR.glob("*/part-*.parquet")):
        print(f"Clearing existing parsed data in {MESSAGE_DIR}")
        print("  (Set ITCH_KEEP_EXISTING=1 to keep existing data)")
        shutil.rmtree(MESSAGE_DIR)
        MESSAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Find ITCH file (compressed or uncompressed)
    gz_files = list(ITCH_RAW_DIR.glob("*.gz")) if ITCH_RAW_DIR.exists() else []
    bin_files = list(ITCH_RAW_DIR.glob("*.bin")) if ITCH_RAW_DIR.exists() else []

    if not gz_files and not bin_files:
        raise FileNotFoundError(
            f"No raw ITCH binary found at {ITCH_RAW_DIR}.\n"
            "Download first:\n"
            "  uv run python data/equities/market/microstructure/nasdaq_itch_download.py"
        )

# %%
# Decompress if needed and extract trading date
if not SKIP_PARSING and (gz_files or bin_files):
    # Prefer uncompressed, otherwise decompress
    if bin_files:
        itch_file = bin_files[0]
        print(f"Found uncompressed: {itch_file.name}")
    else:
        gz_file = gz_files[0]
        itch_file = gz_file.with_suffix(".bin")

        if not itch_file.exists():
            print(f"Decompressing {gz_file.name}...")
            with gzip.open(gz_file, "rb") as f_in, open(itch_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"Created: {itch_file.name} ({itch_file.stat().st_size / 1e9:.1f} GB)")
        else:
            print(f"Found: {itch_file.name}")

    # Extract trading date from filename (format: MMDDYYYY.NASDAQ_ITCH50.bin)
    date_str = itch_file.stem.split(".")[0]
    if len(date_str) == 8 and date_str.isdigit():
        trading_day = date(int(date_str[4:8]), int(date_str[:2]), int(date_str[2:4]))
    else:
        trading_day = date(2020, 1, 30)  # Fallback

    print(f"Trading day: {trading_day}")

    # Parse ITCH file (full-day parse: ~22 min on the reference machine)
    if itch_file and itch_file.exists():
        counts = parse_itch_file(
            itch_file=itch_file,
            trading_day=trading_day,
            output_dir=MESSAGE_DIR,
            # max_messages=1_000_000,  # Remove this line for full parse
        )

        print("\nMessage counts:")
        for msg_type, count in sorted(counts.items(), key=lambda x: -x[1]):
            name = MESSAGE_SPECS.get(msg_type, {}).get("name", "Unknown")
            print(f"  {msg_type} ({name:25}): {count:>10,}")

# %% [markdown]
# ## 5. Message Type Analysis
#
# After parsing, we can analyze message distributions.

# %%
# Message type distribution — use lazy scan to count without loading all data
print("Parsed Message Types:")
print("-" * 50)
for msg_dir in sorted(MESSAGE_DIR.iterdir()):
    if (
        msg_dir.is_dir()
        and len(msg_dir.name) == 1
        and msg_dir.name.isupper()
        and list(msg_dir.glob("*.parquet"))
    ):
        count = pl.scan_parquet(msg_dir / "*.parquet").select(pl.len()).collect().item()
        name = MESSAGE_SPECS.get(msg_dir.name, {}).get("name", "Unknown")
        print(f"  {msg_dir.name} ({name:25}): {count:>12,} messages")

# %%
# Schema compatibility check — verify we can read each message type
print("Schema Compatibility Check:")
print("-" * 50)
for msg_dir in sorted(MESSAGE_DIR.iterdir()):
    if (
        msg_dir.is_dir()
        and len(msg_dir.name) == 1
        and msg_dir.name.isupper()
        and list(msg_dir.glob("*.parquet"))
    ):
        try:
            sample = pl.scan_parquet(msg_dir / "*.parquet").head(5).collect()
            name = MESSAGE_SPECS.get(msg_dir.name, {}).get("name", "Unknown")
            print(f"  [OK] {msg_dir.name} ({name:25}): cols={list(sample.columns)[:4]}...")
        except Exception as e:
            print(f"  [FAIL] {msg_dir.name}: {e}")

# %% [markdown]
# ## 6. Production Parsing with Rust
#
# The Python parser above decodes one message per loop iteration, and a trading day holds a
# few hundred million of them. The same protocol parsed in Rust reads the file through a
# memory map and unpacks each message without copying it first, so it neither pays the
# per-message interpreter overhead nor holds the decoded messages in memory.
#
# **Repository**: [github.com/ml4t/itch-parser](https://github.com/ml4t/itch-parser)
#
# The figures in the book's Table 3.1, for the same 13 GB file on one machine, are about
# twenty-three minutes and roughly 8 GB of memory in Python against under five minutes and
# under 500 MB in Rust. Wall-clock timings move with disk and CPU, so read them as an order
# of magnitude rather than a ratio: the gap is large enough that it decides which parser you
# reach for, and not stable enough to quote to a decimal place. The throughput this notebook
# printed above is the Python side of the same comparison, measured on the machine that ran
# it.
#
# ### Installation
#
# ```bash
# # Clone the repository
# git clone https://github.com/ml4t/itch-parser.git
# cd itch-parser
#
# # Build release binary
# cargo build --release
# ```
#
# ### Usage
#
# ```bash
# # Parse ITCH file (works with .gz or uncompressed)
# ./target/release/itch_parser <input_file> <output_dir> <MMDDYYYY>
#
# # Example
# ./target/release/itch_parser data/01302020.NASDAQ_ITCH50.gz ./messages 01302020
# ```
#
# Output is identical Parquet files partitioned by message type, compatible with
# the Python code in this notebook and downstream analysis.
#
# ### When to Use Which
#
# | Use Case | Recommendation |
# |----------|---------------|
# | Learning the protocol | Python (this notebook) |
# | Debugging parse issues | Python |
# | Processing a single day | Either |
# | Multi-day backtesting | **Rust** |
# | Production pipeline | **Rust** |
#
# Both parsers write the same Parquet schema, so a day parsed either way feeds every
# notebook that follows without change.

# %% [markdown]
# ## Key Takeaways
#
# 1. **The protocol is message-by-order.** Every event names the individual order it acts
#    on, stamped to the nanosecond, which is what makes book reconstruction possible at all.
# 2. **Six message types carry the book.** `A` and `F` add an order, `E` and `C` execute
#    against one, `X` cancels part of one, `D` deletes one, `U` replaces one. The rest
#    describe the session around them.
# 3. **Numbers arrive encoded.** Prices are integers with four implied decimal places, and
#    timestamps are nanoseconds since midnight, so both need converting before use.
# 4. **Parse in batches, not in one pass.** Buffering by message type and flushing to
#    Parquet is what keeps a day that does not fit in memory from having to.
# 5. **Which parser depends on how many days you need.** Python reads clearly and takes
#    about twenty minutes per day; Rust emits the same schema in a fraction of that.
#
# ### Known limitations
#
# - The parse covers one venue. NASDAQ ITCH sees NASDAQ-routed activity, not the
#   consolidated tape, so counts here are a venue's share of a symbol's trading rather than
#   all of it.
# - Message types outside `FMT_DICT` are counted and skipped rather than decoded.
# - The Rust timings quoted above were measured elsewhere, on one machine; this notebook
#   times only its own Python parse.
#
# ### Next Steps
#
# - **Order Book Reconstruction**: `02_itch_lob_reconstruction`
# - **Trading Activity Overview**: `05_itch_trading_activity` (includes E/C enrichment)
#
# ---
#
# ## Reference
#
# Bouchaud, J.-P., Bonart, J., Donier, J., & Gould, M. (2018).
# *Trades, Quotes and Prices: Financial Markets Under the Microscope*.
# Cambridge University Press.
# [https://doi.org/10.1017/9781009028943](https://doi.org/10.1017/9781009028943)
