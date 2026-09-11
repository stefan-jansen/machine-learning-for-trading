"""The LOB snapshot buffer must be sized from the snapshot interval.

``_numba_reconstruct_lob`` runs in nopython mode, which does no bounds checking, so a
snapshot buffer too small for the number of snapshots is an out-of-bounds heap write
rather than an exception. The buffer used to be ``n_messages // 100 + 10000``, which has
nothing to do with the interval: shortening ``SNAPSHOT_FREQ`` - a declared parameter of
``02_itch_lob_reconstruction`` - raised the snapshot count without raising the buffer.
The case below produces 20,000 snapshots against that formula's 10,400 and aborted the
interpreter with ``free(): invalid pointer`` before the fix.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Loaded by path rather than imported by name: chapter directories are number-prefixed
# and are only on sys.path inside a notebook run, so `import limit_orderbook` resolves
# for the notebooks beside it and for nothing under tests/.
_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "03_market_microstructure" / "limit_orderbook.py"
)
_spec = importlib.util.spec_from_file_location("ch03_limit_orderbook", _MODULE_PATH)
limit_orderbook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(limit_orderbook)
reconstruct_lob_with_ofi = limit_orderbook.reconstruct_lob_with_ofi

N_MESSAGES = 40_000
SPAN_SECONDS = 3_600


def _adds() -> pl.DataFrame:
    """One add per 90 ms for an hour, alternating sides over two price levels.

    Two levels rather than 40,000 keeps the kernel's best-price scan O(1) per message;
    the orders are never removed, so both sides of the book stay populated and every
    interval boundary produces a snapshot.
    """
    step_ns = int(SPAN_SECONDS * 1e9) // N_MESSAGES
    base = np.datetime64("2026-01-02T14:30:00", "ns").astype("int64")
    offsets = np.arange(N_MESSAGES, dtype=np.int64) * step_ns
    is_bid = np.arange(N_MESSAGES) % 2 == 0
    return pl.DataFrame(
        {
            "timestamp": pl.Series(base + offsets, dtype=pl.Int64).cast(pl.Datetime("ns")),
            "tracking_number": np.zeros(N_MESSAGES, dtype=np.int64),
            "order_reference_number": np.arange(1, N_MESSAGES + 1, dtype=np.int64),
            "buy_sell_indicator": np.where(is_bid, "B", "S"),
            "shares": np.full(N_MESSAGES, 100, dtype=np.int64),
            "price": np.where(is_bid, 100.0, 100.01),
        }
    )


def _reconstruct(freq: str) -> pl.DataFrame:
    adds = _adds()
    empty = adds.head(0)
    return reconstruct_lob_with_ofi(
        adds, empty, empty, empty, snapshot_freq=freq, show_progress=False
    )


@pytest.mark.parametrize(
    ("freq", "expected"),
    [("1s", 3_334), ("500ms", 6_667), ("100ms", 20_000)],
)
def test_snapshot_count_follows_the_interval(freq: str, expected: int) -> None:
    assert len(_reconstruct(freq)) == expected


def test_a_short_interval_exceeds_the_old_message_count_buffer() -> None:
    """The regression itself: more snapshots than ``n // 100 + 10000`` would have held."""
    lob = _reconstruct("100ms")
    assert len(lob) > N_MESSAGES // 100 + 10_000


def test_snapshots_are_ordered_and_at_least_one_interval_apart() -> None:
    gaps = _reconstruct("500ms")["timestamp"].diff().drop_nulls().dt.total_nanoseconds()
    assert gaps.min() >= 500_000_000


def test_an_unrecognised_frequency_is_refused() -> None:
    """A polars-style '1m' is not in the map, and one-second snapshots mislabel the axis."""
    with pytest.raises(ValueError, match="1m"):
        _reconstruct("1m")
