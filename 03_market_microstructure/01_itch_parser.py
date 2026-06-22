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
# - Understand the ITCH 5.0 binary message format
# - Parse ITCH messages using Python's `struct` module
# - Load pre-parsed ITCH data for analysis
# - Choose between Python (educational) and Rust (production) parsers
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
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from time import time

warnings.filterwarnings("ignore")

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

# %% tags=["parameters"]
SKIP_PARSING = False

# %%
# Data paths
# Canonical parsed messages location (both read and write target)
MESSAGE_DIR = load_nasdaq_itch(get_base_path=True)
MESSAGE_DIR.mkdir(parents=True, exist_ok=True)

# Raw ITCH binary input (from download script)
ITCH_RAW_DIR = MESSAGE_DIR.parent / "raw"

print(f"Raw ITCH data (input): {ITCH_RAW_DIR}")
print(f"Parsed messages (output): {MESSAGE_DIR}")
_raw_present = ITCH_RAW_DIR.exists() and next(ITCH_RAW_DIR.iterdir(), None) is not None
print(f"Raw data exists: {_raw_present}")
if not _raw_present:
    print("  (run NB00_itch_download.py first to fetch the ITCH binary)")

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

# %%
# Create a sample Add Order message to demonstrate parsing
# Format: >HH6sQsI8sI (big-endian)

sample_add_order = struct.pack(
    ">HH6sQsI8sI",
    1234,  # stock_locate
    5678,  # tracking_number
    b"\x00\x00\x00\x00\x00\x01",  # timestamp (1 nanosecond)
    9876543210,  # order_reference_number
    b"B",  # buy_sell_indicator
    100,  # shares
    b"AAPL    ",  # stock (padded to 8 chars)
    1500000,  # price (150.0000 in price4 format)
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
# This Python parser is for **educational purposes**. For production use with large files,
# use the Rust parser (see Section 6) which provides order-of-magnitude speedups.


# %% [markdown]
# ### Parser Helpers
#
# We split the parser into three functions: `_read_frame` reads one binary message
# frame, `_decode_message` unpacks and converts it, and `parse_itch_file` orchestrates
# the loop with buffered Parquet writes.


# %%
def _read_frame(f, pbar) -> tuple[str, bytes] | None:
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
def _decode_message(msg_type: str, payload: bytes) -> dict | None:
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


# %% — single function body, helpers already extracted
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

            frame = _read_frame(f, pbar)
            if frame is None:
                break
            msg_type, payload = frame
            counts[msg_type] += 1

            if msg_type not in FMT_DICT:
                continue

            msg = _decode_message(msg_type, payload)
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
# The Python parser above is educational but slow for full-day files.
# For production use, we provide a **Rust parser** that is an order of magnitude faster.
#
# **Repository**: [github.com/ml4t/itch-parser](https://github.com/ml4t/itch-parser)
#
# ### Performance Characteristics
#
# | Aspect | Python | Rust |
# |--------|--------|------|
# | Speed | Baseline | **10-20× faster** |
# | Memory | High (buffers in RAM) | Low (streaming) |
# | Use case | Learning, debugging | Production pipelines |
#
# Actual speedups depend on disk I/O and CPU. The Rust parser uses memory-mapped
# I/O and zero-copy parsing, which provides substantial gains on modern hardware.
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

# %% [markdown]
# ## Key Takeaways
#
# 1. **ITCH Protocol**: Binary message-by-order format with nanosecond precision
# 2. **Message Types**: A/F (add), E/C (execute), X (cancel), D (delete), U (replace)
# 3. **Price Format**: Integers with 4 implied decimals (150.0000 → 1500000)
# 4. **Parser Choice**: Python for learning, Rust for production (20× faster)
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
