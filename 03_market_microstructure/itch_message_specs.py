"""NASDAQ TotalView-ITCH 5.0 Message Specifications.

This module defines the binary message formats for the ITCH protocol.
Used by both Python (educational) and Rust (production) parsers.

Reference: https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf
"""

import struct
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import polars as pl

# ITCH 5.0 Message Specifications
# Each message has a fixed binary structure defined by field name and struct format code.
# Format codes: H=uint16, I=uint32, Q=uint64, s=char, Ns=N chars
MESSAGE_SPECS = {
    "S": {  # System Event
        "name": "System Event",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("event_code", "s"),
        ],
    },
    "R": {  # Stock Directory
        "name": "Stock Directory",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("stock", "8s"),
            ("market_category", "s"),
            ("financial_status", "s"),
            ("round_lot_size", "I"),
            ("round_lots_only", "s"),
            ("issue_classification", "s"),
            ("issue_subtype", "2s"),
            ("authenticity", "s"),
            ("short_sale_threshold", "s"),
            ("ipo_flag", "s"),
            ("luld_reference_price_tier", "s"),
            ("etp_flag", "s"),
            ("etp_leverage_factor", "I"),
            ("inverse_indicator", "s"),
        ],
    },
    "A": {  # Add Order (No MPID)
        "name": "Add Order",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("order_reference_number", "Q"),
            ("buy_sell_indicator", "s"),
            ("shares", "I"),
            ("stock", "8s"),
            ("price", "I"),
        ],
    },
    "F": {  # Add Order (With MPID)
        "name": "Add Order MPID",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("order_reference_number", "Q"),
            ("buy_sell_indicator", "s"),
            ("shares", "I"),
            ("stock", "8s"),
            ("price", "I"),
            ("attribution", "4s"),
        ],
    },
    "E": {  # Order Executed
        "name": "Order Executed",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("order_reference_number", "Q"),
            ("executed_shares", "I"),
            ("match_number", "Q"),
        ],
    },
    "C": {  # Order Executed with Price
        "name": "Order Executed with Price",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("order_reference_number", "Q"),
            ("executed_shares", "I"),
            ("match_number", "Q"),
            ("printable", "s"),
            ("execution_price", "I"),
        ],
    },
    "X": {  # Order Cancel
        "name": "Order Cancel",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("order_reference_number", "Q"),
            ("cancelled_shares", "I"),
        ],
    },
    "D": {  # Order Delete
        "name": "Order Delete",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("order_reference_number", "Q"),
        ],
    },
    "U": {  # Order Replace
        "name": "Order Replace",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("original_order_reference_number", "Q"),
            ("new_order_reference_number", "Q"),
            ("shares", "I"),
            ("price", "I"),
        ],
    },
    "P": {  # Trade (Non-Cross)
        "name": "Trade",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("order_reference_number", "Q"),
            ("buy_sell_indicator", "s"),
            ("shares", "I"),
            ("stock", "8s"),
            ("price", "I"),
            ("match_number", "Q"),
        ],
    },
    "Q": {  # Cross Trade
        "name": "Cross Trade",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("shares", "Q"),
            ("stock", "8s"),
            ("cross_price", "I"),
            ("match_number", "Q"),
            ("cross_type", "s"),
        ],
    },
    "H": {  # Stock Trading Action
        "name": "Trading Action",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("stock", "8s"),
            ("trading_state", "s"),
            ("reserved", "s"),
            ("reason", "4s"),
        ],
    },
    "Y": {  # Reg SHO Short Sale Price Test Restricted Indicator
        "name": "Reg SHO Restriction",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("stock", "8s"),
            ("reg_sho_action", "s"),
        ],
    },
    "L": {  # Market Participant Position
        "name": "Market Participant Position",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("mpid", "4s"),
            ("stock", "8s"),
            ("primary_market_maker", "s"),
            ("market_maker_mode", "s"),
            ("market_participant_state", "s"),
        ],
    },
    "V": {  # MWCB Decline Level
        "name": "MWCB Decline Level",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("level_1", "Q"),
            ("level_2", "Q"),
            ("level_3", "Q"),
        ],
    },
    "W": {  # MWCB Status
        "name": "MWCB Status",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("breached_level", "s"),
        ],
    },
    "J": {  # LULD Auction Collar
        "name": "LULD Auction Collar",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("stock", "8s"),
            ("auction_collar_reference_price", "I"),
            ("upper_auction_collar_price", "I"),
            ("lower_auction_collar_price", "I"),
            ("auction_collar_extension", "I"),
        ],
    },
    "K": {  # IPO Quoting Period Update
        "name": "IPO Quoting Period",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("stock", "8s"),
            ("ipo_quotation_release_time", "I"),
            ("ipo_quotation_release_qualifier", "s"),
            ("ipo_price", "I"),
        ],
    },
    "B": {  # Broken Trade
        "name": "Broken Trade",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("match_number", "Q"),
        ],
    },
    "I": {  # NOII (Net Order Imbalance Indicator)
        "name": "NOII",
        "fields": [
            ("stock_locate", "H"),
            ("tracking_number", "H"),
            ("timestamp", "6s"),
            ("paired_shares", "Q"),
            ("imbalance_shares", "Q"),
            ("imbalance_direction", "s"),
            ("stock", "8s"),
            ("far_price", "I"),
            ("near_price", "I"),
            ("current_reference_price", "I"),
            ("cross_type", "s"),
            ("price_variation_indicator", "s"),
        ],
    },
}

# System event codes (for S messages)
EVENT_CODES = {
    "O": "Start of Messages",
    "S": "Start of System Hours",
    "Q": "Start of Market Hours",
    "M": "End of Market Hours",
    "E": "End of System Hours",
    "C": "End of Messages",
}

# Mapping from struct format codes to Polars data types
# Used for explicit casting in Parquet output
_STRUCT_TO_POLARS = {
    "H": pl.UInt16,  # unsigned short (2 bytes)
    "I": pl.UInt32,  # unsigned int (4 bytes)
    "Q": pl.UInt64,  # unsigned long long (8 bytes)
}


def build_parsers(specs: dict | None = None) -> tuple[dict, dict]:
    """Build namedtuples and struct format strings for each message type.

    Args:
        specs: Message specifications dict. Defaults to MESSAGE_SPECS.

    Returns:
        Tuple of (namedtuples_dict, format_strings_dict)
    """
    if specs is None:
        specs = MESSAGE_SPECS

    namedtuples = {}
    formats = {}

    for msg_type, spec in specs.items():
        fields = [f[0] for f in spec["fields"]]
        fmt_codes = [f[1] for f in spec["fields"]]

        namedtuples[msg_type] = namedtuple(msg_type, fields)
        formats[msg_type] = ">" + "".join(fmt_codes)  # Big-endian

    return namedtuples, formats


def build_parquet_schemas(specs: dict | None = None) -> dict[str, dict[str, pl.DataType]]:
    """Generate Parquet schema definitions from message specifications.

    Only includes fields that need explicit casting (numeric types).
    String fields and timestamp are handled specially during parsing.

    Args:
        specs: Message specifications dict. Defaults to MESSAGE_SPECS.

    Returns:
        Dict mapping message type to column->dtype mappings.
    """
    if specs is None:
        specs = MESSAGE_SPECS

    schemas = {}

    for msg_type, spec in specs.items():
        schema = {}
        for field_name, fmt_code in spec["fields"]:
            # Skip timestamp (handled specially) and string fields
            if fmt_code == "6s" or fmt_code.endswith("s"):
                continue

            # Map struct format to Polars type
            if fmt_code in _STRUCT_TO_POLARS:
                schema[field_name] = _STRUCT_TO_POLARS[fmt_code]

        schemas[msg_type] = schema

    return schemas


# Pre-built parsers and schemas for convenience
NT_DICT, FMT_DICT = build_parsers()
PARQUET_SCHEMAS = build_parquet_schemas()


def parse_timestamp(ts_bytes: bytes) -> int:
    """Convert 6-byte ITCH timestamp to nanoseconds since midnight."""
    return int.from_bytes(ts_bytes, byteorder="big")


def parse_price4(price_int: int) -> float:
    """Convert ITCH price4 format (4 implied decimals) to float."""
    return price_int / 10000


def get_message_size(msg_type: str) -> int:
    """Get the byte size of a message type (excluding length prefix)."""
    if msg_type not in FMT_DICT:
        raise ValueError(f"Unknown message type: {msg_type}")
    return struct.calcsize(FMT_DICT[msg_type])


def print_message_formats() -> None:
    """Print all message formats with their sizes."""
    print("ITCH Message Formats:\n")
    for msg_type, spec in MESSAGE_SPECS.items():
        fmt = FMT_DICT[msg_type]
        size = struct.calcsize(fmt)
        print(f"  {msg_type} ({spec['name']:25}): {size:2} bytes - {fmt}")


def flush_to_parquet(
    buffers: dict[str, list[dict]],
    output_dir: Path,
    base_ts: datetime,
    file_counters: dict[str, int],
    schemas: dict[str, dict[str, pl.DataType]] | None = None,
) -> None:
    """Write message buffers to Parquet files with unique names.

    Args:
        buffers: Dict mapping message type to list of parsed message dicts.
        output_dir: Directory for Parquet output (one subdir per message type).
        base_ts: Base timestamp (midnight) for datetime conversion.
        file_counters: Mutable dict tracking file indices per message type.
        schemas: Optional schema overrides. Defaults to PARQUET_SCHEMAS.
    """

    if schemas is None:
        schemas = PARQUET_SCHEMAS

    for msg_type, records in buffers.items():
        if not records:
            continue
        df = pl.DataFrame(records)

        # Cast to correct types to match Rust parser schema
        if msg_type in schemas:
            cast_exprs = [
                pl.col(col).cast(dtype)
                for col, dtype in schemas[msg_type].items()
                if col in df.columns
            ]
            if cast_exprs:
                df = df.with_columns(cast_exprs)

        # Convert timestamp from nanoseconds-since-midnight to datetime
        # Uses Polars native datetime with nanosecond precision
        if "timestamp" in df.columns:
            df = df.with_columns(
                (
                    pl.lit(base_ts).cast(pl.Datetime("ns"))
                    + pl.duration(nanoseconds=pl.col("timestamp"))
                ).alias("timestamp")
            )

        out_dir = output_dir / msg_type
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write with unique filename per flush to avoid overwriting
        file_idx = file_counters[msg_type]
        df.write_parquet(out_dir / f"part-{file_idx:06d}.parquet")
        file_counters[msg_type] += 1

    buffers.clear()
