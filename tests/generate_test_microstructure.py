"""Generate minimal synthetic test data for Ch03 microstructure notebooks.

Creates test-sized datasets that match the schemas expected by Ch03 notebooks
and related notebooks (Ch02 futures individual, Ch04 prediction markets).

Writes to ~/ml4t/test-data/data/ which serves as ML4T_DATA_PATH
in CI.

Usage:
    uv run python tests/generate_test_microstructure.py
"""

from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# ── Output root ──────────────────────────────────────────────────────────────
TEST_DATA_ROOT = Path.home() / "ml4t" / "test-data" / "data"

# Seed for reproducibility
RNG = np.random.default_rng(42)


# ═════════════════════════════════════════════════════════════════════════════
# 1. ITCH Parsed Messages (for NB 02-10)
# ═════════════════════════════════════════════════════════════════════════════
# These go into the ITCH messages path that load_nasdaq_itch() resolves:
#   ML4T_DATA_PATH / "equities" / "market" / "microstructure" / "nasdaq_itch" / "messages"
# Notebooks 02-10 read via utils.limit_orderbook.load_itch_messages(itch_dir, msg_type, symbol)


def _ns_timestamp(hour: int, minute: int, second: int = 0, micro: int = 0) -> datetime:
    """Create a nanosecond-precision datetime on the ITCH trading day (2020-01-30)."""
    return datetime(2020, 1, 30, hour, minute, second, micro)


def generate_itch_messages() -> None:
    """Generate all ITCH message type parquet files."""
    itch_dir = (
        TEST_DATA_ROOT / "equities" / "market" / "microstructure" / "nasdaq_itch" / "messages"
    )

    # ── R (Stock Directory) ──────────────────────────────────────────────
    r_dir = itch_dir / "R"
    r_dir.mkdir(parents=True, exist_ok=True)
    r_df = pl.DataFrame(
        {
            "stock_locate": pl.Series([1, 2, 3], dtype=pl.UInt16),
            "tracking_number": pl.Series([0, 0, 0], dtype=pl.UInt16),
            "timestamp": [
                _ns_timestamp(4, 0, 0),
                _ns_timestamp(4, 0, 0),
                _ns_timestamp(4, 0, 0),
            ],
            "stock": ["AAPL", "MSFT", "NVDA"],
            "market_category": ["Q", "Q", "Q"],
            "financial_status": ["N", "N", "N"],
            "round_lot_size": pl.Series([100, 100, 100], dtype=pl.UInt32),
            "round_lots_only": ["N", "N", "N"],
            "issue_classification": ["C", "C", "C"],
            "issue_subtype": ["Z", "Z", "Z"],
            "authenticity": ["P", "P", "P"],
            "short_sale_threshold": ["N", "N", "N"],
            "ipo_flag": ["N", "N", "N"],
            "luld_reference_price_tier": ["1", "1", "1"],
            "etp_flag": ["N", "N", "N"],
            "etp_leverage_factor": pl.Series([0, 0, 0], dtype=pl.UInt32),
            "inverse_indicator": ["N", "N", "N"],
        }
    ).cast({"timestamp": pl.Datetime("ns")})
    r_df.write_parquet(r_dir / "part-000000.parquet")

    # ── S (System Event) ─────────────────────────────────────────────────
    s_dir = itch_dir / "S"
    s_dir.mkdir(parents=True, exist_ok=True)
    s_df = pl.DataFrame(
        {
            "stock_locate": pl.Series([0, 0, 0, 0], dtype=pl.UInt16),
            "tracking_number": pl.Series([0, 0, 0, 0], dtype=pl.UInt16),
            "timestamp": [
                _ns_timestamp(4, 0, 0),
                _ns_timestamp(9, 30, 0),
                _ns_timestamp(16, 0, 0),
                _ns_timestamp(20, 0, 0),
            ],
            "event_code": ["O", "Q", "M", "C"],
        }
    ).cast({"timestamp": pl.Datetime("ns")})
    s_df.write_parquet(s_dir / "part-000000.parquet")

    # ── A (Add Order) ────────────────────────────────────────────────────
    # 20 orders for AAPL (stock_locate=1), spanning 10:00 to 15:00
    a_dir = itch_dir / "A"
    a_dir.mkdir(parents=True, exist_ok=True)

    n_orders = 20
    base_price_aapl = 320.0  # AAPL price circa Jan 2020
    order_refs = list(range(1001, 1001 + n_orders))
    sides = ["B" if i % 2 == 0 else "S" for i in range(n_orders)]
    shares = [int(RNG.integers(100, 1001)) for _ in range(n_orders)]
    # Prices: bids slightly below base, asks slightly above
    prices = []
    for i, side in enumerate(sides):
        offset = RNG.uniform(0.01, 0.50)
        if side == "B":
            prices.append(round(base_price_aapl - offset, 4))
        else:
            prices.append(round(base_price_aapl + offset, 4))

    # Timestamps spaced across 10:00-15:00 (300 minutes = 18000 seconds)
    a_timestamps = [
        _ns_timestamp(10, 0) + timedelta(seconds=int(i * 18000 / n_orders)) for i in range(n_orders)
    ]

    # Prices stored as ITCH price4 integers (multiply by 10000) per spec
    a_df = pl.DataFrame(
        {
            "stock_locate": pl.Series([1] * n_orders, dtype=pl.UInt16),
            "tracking_number": pl.Series([0] * n_orders, dtype=pl.UInt16),
            "timestamp": a_timestamps,
            "order_reference_number": pl.Series(order_refs, dtype=pl.UInt64),
            "buy_sell_indicator": sides,
            "shares": pl.Series(shares, dtype=pl.UInt32),
            "stock": ["AAPL"] * n_orders,
            "price": pl.Series([int(p * 10000) for p in prices], dtype=pl.UInt32),
        }
    ).cast({"timestamp": pl.Datetime("ns")})
    a_df.write_parquet(a_dir / "part-000000.parquet")

    # ── D (Order Delete) ─────────────────────────────────────────────────
    d_dir = itch_dir / "D"
    d_dir.mkdir(parents=True, exist_ok=True)
    delete_refs = [1001, 1003, 1005, 1007, 1009]
    d_df = pl.DataFrame(
        {
            "stock_locate": pl.Series([1] * 5, dtype=pl.UInt16),
            "tracking_number": pl.Series([0] * 5, dtype=pl.UInt16),
            "timestamp": [
                a_timestamps[0] + timedelta(seconds=30),
                a_timestamps[2] + timedelta(seconds=30),
                a_timestamps[4] + timedelta(seconds=30),
                a_timestamps[6] + timedelta(seconds=30),
                a_timestamps[8] + timedelta(seconds=30),
            ],
            "order_reference_number": pl.Series(delete_refs, dtype=pl.UInt64),
        }
    ).cast({"timestamp": pl.Datetime("ns")})
    d_df.write_parquet(d_dir / "part-000000.parquet")

    # ── E (Order Executed) ───────────────────────────────────────────────
    e_dir = itch_dir / "E"
    e_dir.mkdir(parents=True, exist_ok=True)
    exec_refs = [1002, 1004, 1006, 1008, 1010, 1012, 1014, 1016]
    exec_shares = [min(shares[r - 1001] // 2, 200) for r in exec_refs]
    e_df = pl.DataFrame(
        {
            "stock_locate": pl.Series([1] * 8, dtype=pl.UInt16),
            "tracking_number": pl.Series([0] * 8, dtype=pl.UInt16),
            "timestamp": [a_timestamps[r - 1001] + timedelta(seconds=60) for r in exec_refs],
            "order_reference_number": pl.Series(exec_refs, dtype=pl.UInt64),
            "executed_shares": pl.Series(exec_shares, dtype=pl.UInt32),
            "match_number": pl.Series(list(range(5001, 5009)), dtype=pl.UInt64),
        }
    ).cast({"timestamp": pl.Datetime("ns")})
    e_df.write_parquet(e_dir / "part-000000.parquet")

    # ── X (Order Cancel) ─────────────────────────────────────────────────
    x_dir = itch_dir / "X"
    x_dir.mkdir(parents=True, exist_ok=True)
    cancel_refs = [1011, 1013, 1015]
    cancel_shares = [shares[r - 1001] // 3 for r in cancel_refs]
    x_df = pl.DataFrame(
        {
            "stock_locate": pl.Series([1] * 3, dtype=pl.UInt16),
            "tracking_number": pl.Series([0] * 3, dtype=pl.UInt16),
            "timestamp": [a_timestamps[r - 1001] + timedelta(seconds=45) for r in cancel_refs],
            "order_reference_number": pl.Series(cancel_refs, dtype=pl.UInt64),
            "cancelled_shares": pl.Series(cancel_shares, dtype=pl.UInt32),
        }
    ).cast({"timestamp": pl.Datetime("ns")})
    x_df.write_parquet(x_dir / "part-000000.parquet")

    # ── C (Order Executed with Price) ────────────────────────────────────
    c_dir = itch_dir / "C"
    c_dir.mkdir(parents=True, exist_ok=True)
    c_refs = [1017, 1018]
    c_df = pl.DataFrame(
        {
            "stock_locate": pl.Series([1] * 2, dtype=pl.UInt16),
            "tracking_number": pl.Series([0] * 2, dtype=pl.UInt16),
            "timestamp": [
                a_timestamps[16] + timedelta(seconds=90),
                a_timestamps[17] + timedelta(seconds=90),
            ],
            "order_reference_number": pl.Series(c_refs, dtype=pl.UInt64),
            "executed_shares": pl.Series([shares[16] // 4, shares[17] // 4], dtype=pl.UInt32),
            "match_number": pl.Series([6001, 6002], dtype=pl.UInt64),
            "printable": ["Y", "Y"],
            "execution_price": pl.Series(
                [int(prices[16] * 10000), int(prices[17] * 10000)], dtype=pl.UInt32
            ),
        }
    ).cast({"timestamp": pl.Datetime("ns")})
    c_df.write_parquet(c_dir / "part-000000.parquet")

    # ── P (Non-Cross Trade) ──────────────────────────────────────────────
    p_dir = itch_dir / "P"
    p_dir.mkdir(parents=True, exist_ok=True)
    p_df = pl.DataFrame(
        {
            "stock_locate": pl.Series([1, 1, 1], dtype=pl.UInt16),
            "tracking_number": pl.Series([0, 0, 0], dtype=pl.UInt16),
            "timestamp": [
                _ns_timestamp(11, 30, 0),
                _ns_timestamp(13, 0, 0),
                _ns_timestamp(14, 30, 0),
            ],
            "order_reference_number": pl.Series([2001, 2002, 2003], dtype=pl.UInt64),
            "buy_sell_indicator": ["B", "S", "B"],
            "shares": pl.Series([200, 150, 300], dtype=pl.UInt32),
            "stock": ["AAPL", "AAPL", "AAPL"],
            "price": pl.Series([int(base_price_aapl * 10000)] * 3, dtype=pl.UInt32),
            "match_number": pl.Series([7001, 7002, 7003], dtype=pl.UInt64),
        }
    ).cast({"timestamp": pl.Datetime("ns")})
    p_df.write_parquet(p_dir / "part-000000.parquet")

    # ── U (Order Replace) ────────────────────────────────────────────────
    u_dir = itch_dir / "U"
    u_dir.mkdir(parents=True, exist_ok=True)
    u_df = pl.DataFrame(
        {
            "stock_locate": pl.Series([1, 1], dtype=pl.UInt16),
            "tracking_number": pl.Series([0, 0], dtype=pl.UInt16),
            "timestamp": [
                a_timestamps[18] + timedelta(seconds=20),
                a_timestamps[19] + timedelta(seconds=20),
            ],
            "original_order_reference_number": pl.Series([1019, 1020], dtype=pl.UInt64),
            "new_order_reference_number": pl.Series([3001, 3002], dtype=pl.UInt64),
            "shares": pl.Series([500, 600], dtype=pl.UInt32),
            "price": pl.Series(
                [int((base_price_aapl - 0.10) * 10000), int((base_price_aapl + 0.10) * 10000)],
                dtype=pl.UInt32,
            ),
        }
    ).cast({"timestamp": pl.Datetime("ns")})
    u_df.write_parquet(u_dir / "part-000000.parquet")

    print(f"  ITCH messages written to {itch_dir}")
    for sub in sorted(itch_dir.iterdir()):
        if sub.is_dir() and sub.name != "enriched":
            n = pl.scan_parquet(sub / "*.parquet").select(pl.len()).collect().item()
            print(f"    {sub.name}/: {n} rows")


# ═════════════════════════════════════════════════════════════════════════════
# 2. DataBento MBO (for NB 09-13)
# ═════════════════════════════════════════════════════════════════════════════
# Path: ML4T_DATA_PATH / "equities" / "market" / "microstructure" / "market_by_order" / "NVDA"
# File naming: xnas-itch-YYYYMMDD.mbo.dbn.parquet (DataBento convention)
# NB09, NB12 expect "timestamp" column. NB10, NB11, NB13 also need it.
# We provide BOTH ts_event and timestamp (same values) for compatibility.


def _generate_mbo_day(base_date: datetime, base_price_nano: int, start_order_id: int) -> list[dict]:
    """Generate one day of MBO messages with realistic bid/ask structure.

    Returns a list of row dicts (not yet a DataFrame).
    """
    rows: list[dict] = []
    order_id = start_order_id
    n_cycles = 50  # 50 cycles spread across 6.5 hours of trading

    for cycle in range(n_cycles):
        cycle_start_ms = cycle * 468_000  # ~7.8 min per cycle

        # Phase 1: Adds (build book) - 15 orders per cycle
        for i in range(15):
            ts = base_date + timedelta(milliseconds=cycle_start_ms + i * 100)
            side = "B" if i % 2 == 0 else "A"
            if side == "B":
                price_offset = -RNG.integers(1, 51) * 10_000_000
            else:
                price_offset = RNG.integers(1, 51) * 10_000_000
            price = base_price_nano + price_offset
            size = int(RNG.integers(1, 501))
            rows.append(
                {
                    "ts_event": ts,
                    "ts_recv": ts + timedelta(microseconds=int(RNG.integers(1, 100))),
                    "action": "A",
                    "side": side,
                    "price": price,
                    "size": size,
                    "order_id": order_id,
                    "flags": 0,
                    "publisher_id": 39,
                }
            )
            order_id += 1

        # Phase 2: Modifications - 3 per cycle
        for i in range(3):
            ts = base_date + timedelta(milliseconds=cycle_start_ms + 1500 + i * 200)
            mod_order = order_id - 15 + i * 5
            mod_side = "B" if i % 2 == 0 else "A"
            if mod_side == "B":
                price_offset = -RNG.integers(1, 31) * 10_000_000
            else:
                price_offset = RNG.integers(1, 31) * 10_000_000
            rows.append(
                {
                    "ts_event": ts,
                    "ts_recv": ts + timedelta(microseconds=int(RNG.integers(1, 100))),
                    "action": "M",
                    "side": mod_side,
                    "price": base_price_nano + price_offset,
                    "size": int(RNG.integers(1, 300)),
                    "order_id": mod_order,
                    "flags": 0,
                    "publisher_id": 39,
                }
            )

        # Phase 3: Cancels - 3 per cycle
        for i in range(3):
            ts = base_date + timedelta(milliseconds=cycle_start_ms + 2100 + i * 200)
            cancel_order = order_id - 14 + i * 5
            rows.append(
                {
                    "ts_event": ts,
                    "ts_recv": ts + timedelta(microseconds=int(RNG.integers(1, 100))),
                    "action": "C",
                    "side": "A" if i % 2 == 0 else "B",
                    "price": base_price_nano,
                    "size": 0,
                    "order_id": cancel_order,
                    "flags": 0,
                    "publisher_id": 39,
                }
            )

        # Phase 4: Fills (F) and Trades (T) - 10 per cycle
        for i in range(10):
            ts = base_date + timedelta(milliseconds=cycle_start_ms + 2700 + i * 300)
            fill_order = order_id - 13 + i
            fill_size = int(RNG.integers(1, 200))
            # Biased aggressor side for realistic imbalance runs.
            # Runs of 10 consecutive cycles (~100 trades) with 95% bias,
            # creating sustained imbalance that triggers bar boundaries.
            # This mimics real institutional order flow patterns.
            run_idx = cycle // 10
            if run_idx % 2 == 0:
                aggressor = "B" if RNG.random() < 0.95 else "A"
            else:
                aggressor = "A" if RNG.random() < 0.95 else "B"
            trade_price = base_price_nano + RNG.integers(-5, 6) * 10_000_000
            fill_side = "A" if aggressor == "B" else "B"
            rows.append(
                {
                    "ts_event": ts,
                    "ts_recv": ts + timedelta(microseconds=int(RNG.integers(1, 100))),
                    "action": "F",
                    "side": fill_side,
                    "price": trade_price,
                    "size": fill_size,
                    "order_id": fill_order,
                    "flags": 128,
                    "publisher_id": 39,
                }
            )
            rows.append(
                {
                    "ts_event": ts + timedelta(microseconds=1),
                    "ts_recv": ts + timedelta(microseconds=int(RNG.integers(2, 150))),
                    "action": "T",
                    "side": aggressor,
                    "price": trade_price,
                    "size": fill_size,
                    "order_id": fill_order,
                    "flags": 128,
                    "publisher_id": 39,
                }
            )

    return rows


def generate_mbo_data() -> None:
    """Generate synthetic DataBento MBO tick data for NVDA.

    Key schema requirements from notebooks:
    - NB09 (lee_ready): expects "timestamp" column, reads parquet directly
    - NB10 (information_bars): expects filename like xnas-itch-YYYYMMDD.mbo.dbn.parquet
    - NB11 (lob_reconstruction): expects "ts_event" column, reads parquet directly
    - NB12 (mbo_analysis): expects "timestamp" column, reads parquet directly
    - NB13 (bar_sampling): expects "timestamp" column, filename like xnas-itch-*

    We include both ts_event and timestamp columns, and use DataBento file naming.
    We also generate enough data (spread across hours) for meaningful analysis.
    """
    mbo_dir = TEST_DATA_ROOT / "equities" / "market" / "microstructure" / "market_by_order" / "NVDA"
    mbo_dir.mkdir(parents=True, exist_ok=True)

    # Remove old file if it exists (was named 20241104.parquet before)
    old_file = mbo_dir / "20241104.parquet"
    if old_file.exists():
        old_file.unlink()

    base_price_nano = 140_000_000_000  # $140 in nanodollars

    # Generate 3 days of data. NB13 (bar_sampling) computes day-to-day CV which
    # needs >= 2 days. NB10 (information_bars) also benefits from more trades.
    trading_days = [
        datetime(2024, 11, 4, 14, 30, 0),  # Monday 9:30 AM ET in UTC
        datetime(2024, 11, 5, 14, 30, 0),  # Tuesday
        datetime(2024, 11, 6, 14, 30, 0),  # Wednesday
    ]

    for day_idx, base_date in enumerate(trading_days):
        rows = _generate_mbo_day(base_date, base_price_nano, 100_000 + day_idx * 10_000)

        df = (
            pl.DataFrame(rows)
            .cast(
                {
                    "ts_event": pl.Datetime("ns"),
                    "ts_recv": pl.Datetime("ns"),
                    "price": pl.Int64,
                    "size": pl.Int64,
                    "order_id": pl.Int64,
                    "flags": pl.Int64,
                    "publisher_id": pl.Int64,
                }
            )
            .sort("ts_event")
        )

        # Add canonical "timestamp" column (same as ts_event) for notebooks that expect it.
        # NB09, NB12, NB13 use "timestamp"; NB10, NB11 use "ts_event".
        df = df.with_columns(pl.col("ts_event").alias("timestamp"))

        # Write with DataBento filename convention: xnas-itch-YYYYMMDD.mbo.dbn.parquet
        # NB10 and NB13 parse the filename: file_path.name.split("-")[2].split(".")[0]
        date_str = base_date.strftime("%Y%m%d")
        out_file = mbo_dir / f"xnas-itch-{date_str}.mbo.dbn.parquet"
        df.write_parquet(out_file)
        print(f"  MBO day {date_str}: {len(df)} rows -> {out_file}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. AlgoSeek TAQ (for NB 15-16)
# ═════════════════════════════════════════════════════════════════════════════
# Path: ML4T_DATA_PATH / "equities" / "market" / "microstructure" / "trade_and_quotes" / "symbol=AAPL" / "data.parquet"
# NB15 expects: TRADE, QUOTE BID, QUOTE ASK, QUOTE BID NB, QUOTE ASK NB event types
# NB15 does spread analysis using NBBO quotes and trade size distribution


def generate_taq_data() -> None:
    """Generate synthetic AlgoSeek TAQ tick data for AAPL on 2020-03-16.

    Key schema requirements from notebooks:
    - NB15 (taq_eda): Needs TRADE, QUOTE BID NB, QUOTE ASK NB event types
      for spread analysis. Needs enough trades for size distribution.
    - NB16 (taq_lob): Needs QUOTE BID/ASK for LOB reconstruction.

    We generate ~600 events with realistic distributions.
    """
    taq_dir = (
        TEST_DATA_ROOT
        / "equities"
        / "market"
        / "microstructure"
        / "trade_and_quotes"
        / "symbol=AAPL"
    )
    taq_dir.mkdir(parents=True, exist_ok=True)

    # March 16, 2020: AAPL around $250, huge volatility day
    base_date = datetime(2020, 3, 16)
    base_price = 250.0
    exchanges = ["Q", "N", "Z", "P", "K"]

    rows = []
    # Generate ~600 rows: mix of trades, exchange quotes, and NBBO quotes
    # NB15 needs: TRADE events for trade analysis, QUOTE BID NB / QUOTE ASK NB for spread
    for i in range(600):
        # Random time between 9:30 and 16:00 (6.5 hours = 23400 seconds)
        seconds_offset = int(RNG.integers(0, 23400))
        ts = base_date + timedelta(
            hours=9, minutes=30, seconds=seconds_offset, microseconds=int(RNG.integers(0, 999999))
        )

        # Event type distribution:
        # ~15% trades, ~20% NBBO bids, ~20% NBBO asks, ~20% exchange bids, ~20% exchange asks
        # We need QUOTE BID NB and QUOTE ASK NB for NB15's spread analysis
        r = RNG.random()
        if r < 0.15:
            event_type = "TRADE"
            price = round(base_price + RNG.normal(0, 5), 2)
            quantity = int(RNG.integers(10, 10001))
        elif r < 0.35:
            event_type = "QUOTE BID NB"
            price = round(base_price - abs(RNG.normal(0.03, 0.10)), 2)
            quantity = int(RNG.integers(100, 5001))
        elif r < 0.55:
            event_type = "QUOTE ASK NB"
            price = round(base_price + abs(RNG.normal(0.03, 0.10)), 2)
            quantity = int(RNG.integers(100, 5001))
        elif r < 0.75:
            event_type = "QUOTE BID"
            price = round(base_price - abs(RNG.normal(0.05, 0.20)), 2)
            quantity = int(RNG.integers(100, 5001))
        else:
            event_type = "QUOTE ASK"
            price = round(base_price + abs(RNG.normal(0.05, 0.20)), 2)
            quantity = int(RNG.integers(100, 5001))

        rows.append(
            {
                "timestamp": ts,
                "event_type": event_type,
                "price": price,
                "quantity": quantity,
                "exchange": exchanges[int(RNG.integers(0, len(exchanges)))],
                "conditions": "00000000",
            }
        )

    df = (
        pl.DataFrame(rows)
        .cast(
            {
                "timestamp": pl.Datetime("us"),
                "price": pl.Float64,
                "quantity": pl.Int64,
            }
        )
        .sort("timestamp")
    )

    df.write_parquet(taq_dir / "data.parquet")
    print(f"  TAQ data: {len(df)} rows -> {taq_dir / 'data.parquet'}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. IEX Parsed Data (for NB 14)
# ═════════════════════════════════════════════════════════════════════════════
# Path: ML4T_DATA_PATH / "equities" / "market" / "microstructure" / "iex" / "deep" / "parsed" / {type}/


def generate_iex_data() -> None:
    """Generate synthetic IEX DEEP parsed data."""
    parsed_dir = (
        TEST_DATA_ROOT / "equities" / "market" / "microstructure" / "iex" / "deep" / "parsed"
    )

    base_date = datetime(2025, 1, 15, 14, 30, 0)  # 9:30 AM ET in UTC
    base_price = 240.0  # AAPL-ish

    # ── Quotes ───────────────────────────────────────────────────────────
    quotes_dir = parsed_dir / "quotes"
    quotes_dir.mkdir(parents=True, exist_ok=True)

    quote_rows = []
    for i in range(30):
        ts = base_date + timedelta(seconds=i * 60)
        spread = round(abs(RNG.normal(0.02, 0.01)), 4)
        mid = base_price + RNG.normal(0, 0.5)
        quote_rows.append(
            {
                "timestamp": ts,
                "symbol": "AAPL",
                "bid_price": round(mid - spread / 2, 2),
                "bid_size": int(RNG.integers(100, 5001)),
                "ask_price": round(mid + spread / 2, 2),
                "ask_size": int(RNG.integers(100, 5001)),
            }
        )

    quotes_df = pl.DataFrame(quote_rows).cast(
        {
            "timestamp": pl.Datetime("ns"),
            "bid_price": pl.Float64,
            "ask_price": pl.Float64,
            "bid_size": pl.Int64,
            "ask_size": pl.Int64,
        }
    )
    quotes_df.write_parquet(quotes_dir / "data.parquet")
    print(f"  IEX quotes: {len(quotes_df)} rows -> {quotes_dir / 'data.parquet'}")

    # ── Trades ───────────────────────────────────────────────────────────
    trades_dir = parsed_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    trade_rows = []
    for i in range(20):
        ts = base_date + timedelta(seconds=i * 90 + int(RNG.integers(0, 30)))
        trade_rows.append(
            {
                "timestamp": ts,
                "symbol": "AAPL",
                "price": round(base_price + RNG.normal(0, 0.3), 2),
                "size": int(RNG.integers(1, 501)),
            }
        )

    trades_df = pl.DataFrame(trade_rows).cast(
        {
            "timestamp": pl.Datetime("ns"),
            "price": pl.Float64,
            "size": pl.Int64,
        }
    )
    trades_df.write_parquet(trades_dir / "data.parquet")
    print(f"  IEX trades: {len(trades_df)} rows -> {trades_dir / 'data.parquet'}")

    # ── Price Levels ─────────────────────────────────────────────────────
    price_levels_dir = parsed_dir / "price_levels"
    price_levels_dir.mkdir(parents=True, exist_ok=True)

    pl_rows = []
    for i in range(40):
        ts = base_date + timedelta(seconds=i * 45)
        side = "bid" if i % 2 == 0 else "ask"
        offset = RNG.uniform(0.01, 0.50)
        price = round(base_price - offset if side == "bid" else base_price + offset, 2)
        pl_rows.append(
            {
                "timestamp": ts,
                "symbol": "AAPL",
                "side": side,
                "price": price,
                "size": int(RNG.integers(100, 3001)),
            }
        )

    pl_df = pl.DataFrame(pl_rows).cast(
        {
            "timestamp": pl.Datetime("ns"),
            "price": pl.Float64,
            "size": pl.Int64,
        }
    )
    pl_df.write_parquet(price_levels_dir / "data.parquet")
    print(f"  IEX price_levels: {len(pl_df)} rows -> {price_levels_dir / 'data.parquet'}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. CME Individual Contracts (for Ch02 NB 04-06)
# ═════════════════════════════════════════════════════════════════════════════
# Path: ML4T_DATA_PATH / "futures" / "market" / "individual" / "{PRODUCT}" / "data.parquet"
# Schema matches what load_cme_futures(continuous=False) returns:
#   timestamp (datetime[ns, UTC]), rtype, publisher_id, instrument_id,
#   open, high, low, close, volume, product
#
# NB06 (futures_continuous) needs:
# - Multiple contracts with OVERLAPPING date ranges
# - Volume patterns that make front-month detection possible
# - Enough contracts for roll detection to produce adj_close


def generate_individual_futures() -> None:
    """Generate synthetic CME individual contract data for ES, NQ, CL.

    Key requirements from NB06 (continuous construction):
    - Contracts must overlap in time (concurrent trading)
    - Front month should have highest volume (for volume-based roll detection)
    - Need at least 3 contracts with clear roll transitions
    - Need enough data points for roll gaps to produce adj_close
    """
    individual_dir = TEST_DATA_ROOT / "futures" / "market" / "individual"

    products = {
        "ES": {"base_price": 4500.0, "tick": 0.25},
        "NQ": {"base_price": 15500.0, "tick": 0.25},
        "CL": {"base_price": 75.0, "tick": 0.01},
    }

    # Contract months: H=March, M=June, U=Sep, Z=Dec
    # Instrument IDs encode contract month. Simulate 4 quarterly contracts
    # overlapping across 2024, with volume-based rolls.
    contract_specs = [
        # (instrument_id, start_day_offset, end_day_offset, is_front_until_day)
        # Contract 1 (H24): front month days 0-29, then rolls to contract 2
        (49701, 0, 59, 29),
        # Contract 2 (M24): front month days 30-89, then rolls to contract 3
        (49702, 15, 119, 89),
        # Contract 3 (U24): front month days 90-149, then rolls to contract 4
        (49703, 75, 179, 149),
        # Contract 4 (Z24): front month from day 150 onward
        (49704, 135, 209, 209),
    ]

    for product, cfg in products.items():
        prod_dir = individual_dir / product
        prod_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        start = datetime(2024, 1, 2, 0, 0, 0)

        for inst_id, start_day, end_day, front_until in contract_specs:
            # Adjust instrument_id per product to be unique
            if product == "NQ":
                inst_id += 1000
            elif product == "CL":
                inst_id += 2000

            for day_offset in range(start_day, end_day + 1):
                # Generate one bar per day (24 hours apart for hourly-like data)
                ts = start + timedelta(days=day_offset)

                # Price drifts slightly
                base = cfg["base_price"] + RNG.normal(0, cfg["base_price"] * 0.002)
                o = round(base, 2)
                h = round(base + abs(RNG.normal(0, cfg["base_price"] * 0.001)), 2)
                l = round(base - abs(RNG.normal(0, cfg["base_price"] * 0.001)), 2)
                c = round(base + RNG.normal(0, cfg["base_price"] * 0.0005), 2)

                # Volume: high when front month, low when back month
                if day_offset <= front_until:
                    vol = int(RNG.integers(10000, 50001))  # Front month: high volume
                else:
                    vol = int(RNG.integers(100, 3001))  # Back month: low volume

                rows.append(
                    {
                        "timestamp": ts,
                        "rtype": 35,
                        "publisher_id": 1,
                        "instrument_id": inst_id,
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": vol,
                        "product": product,
                    }
                )

        df = (
            pl.DataFrame(rows)
            .cast(
                {
                    "timestamp": pl.Datetime("ns", time_zone="UTC"),
                    "rtype": pl.UInt8,
                    "publisher_id": pl.UInt16,
                    "instrument_id": pl.UInt32,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.UInt64,
                }
            )
            .sort("timestamp")
        )

        df.write_parquet(prod_dir / "data.parquet")
        print(f"  Futures individual {product}: {len(df)} rows -> {prod_dir / 'data.parquet'}")


# ═════════════════════════════════════════════════════════════════════════════
# 6. Kalshi Events (for Ch04 NB 13)
# ═════════════════════════════════════════════════════════════════════════════
# Path: ML4T_DATA_PATH / "prediction_markets" / "kalshi_events.parquet"
# Schema: timestamp (Date), symbol (str), open/high/low/close (Float64), volume (Int64)


def generate_kalshi_data() -> None:
    """Generate synthetic Kalshi prediction market data."""
    pm_dir = TEST_DATA_ROOT / "prediction_markets"
    pm_dir.mkdir(parents=True, exist_ok=True)

    # 5 contracts, ~10 days each = ~50 rows
    contracts = [
        "KXFED-27APR-T4.25",
        "KXFED-27APR-T4.50",
        "KXFED-27JUN-T4.00",
        "KXINFL-27MAR-T3.0",
        "KXGDP-27Q1-T2.0",
    ]

    rows = []
    base_date = date(2027, 3, 1)

    for contract in contracts:
        # Each contract gets a base probability and drifts
        base_prob = RNG.uniform(0.2, 0.8)
        for day in range(10):
            d = base_date + timedelta(days=day)
            # Random walk for probability
            base_prob = max(0.01, min(0.99, base_prob + RNG.normal(0, 0.03)))
            o = round(base_prob, 2)
            h = round(min(0.99, base_prob + abs(RNG.normal(0, 0.02))), 2)
            l = round(max(0.01, base_prob - abs(RNG.normal(0, 0.02))), 2)
            c = round(max(0.01, min(0.99, base_prob + RNG.normal(0, 0.01))), 2)
            vol = int(RNG.integers(50, 5001))

            rows.append(
                {
                    "timestamp": d,
                    "symbol": contract,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": vol,
                }
            )

    df = (
        pl.DataFrame(rows)
        .cast(
            {
                "timestamp": pl.Date,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )
        .sort(["symbol", "timestamp"])
    )

    df.write_parquet(pm_dir / "kalshi_events.parquet")
    print(f"  Kalshi events: {len(df)} rows -> {pm_dir / 'kalshi_events.parquet'}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print(f"Generating test microstructure data in {TEST_DATA_ROOT}\n")

    print("1. ITCH Parsed Messages")
    generate_itch_messages()

    print("\n2. DataBento MBO")
    generate_mbo_data()

    print("\n3. AlgoSeek TAQ")
    generate_taq_data()

    print("\n4. IEX Parsed Data")
    generate_iex_data()

    print("\n5. CME Individual Futures")
    generate_individual_futures()

    print("\n6. Kalshi Prediction Markets")
    generate_kalshi_data()

    print("\nDone.")


if __name__ == "__main__":
    main()
