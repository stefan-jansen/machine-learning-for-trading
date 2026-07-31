"""Tests for contract-following straddle premium moves."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from case_studies.sp500_options._straddle_moves import straddle_premium_moves

SESSIONS = [date(2020, 1, 6) + timedelta(days=offset) for offset in range(5)]
EXPIRY = date(2020, 2, 21)


def _legs(rows: list[dict]) -> pl.LazyFrame:
    """One row per leg, session and contract, in the raw chain's column names."""
    return pl.DataFrame(rows).lazy()


def _contract(symbol: str, strike: float, prices: dict[date, float]) -> list[dict]:
    """Two legs a session, each priced at half the straddle premium, one tick wide."""
    return [
        {
            "symbol": symbol,
            "strike": strike,
            "expiration": EXPIRY,
            "timestamp": session,
            "call_put": leg,
            "bid": premium / 2 - 0.01,
            "ask": premium / 2 + 0.01,
        }
        for session, premium in prices.items()
        for leg in ("C", "P")
    ]


def _entries(*keys: tuple[str, float, date]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"symbol": s, "strike": k, "expiration": EXPIRY, "timestamp": t} for s, k, t in keys]
    )


def test_move_is_measured_within_one_contract() -> None:
    """A second strike quoting different prices must not enter the first strike's move."""
    prices = dict(zip(SESSIONS, [10.0, 11.0, 12.0, 13.0, 14.0], strict=True))
    decoy = dict(zip(SESSIONS, [99.0, 1.0, 99.0, 1.0, 99.0], strict=True))
    moves = straddle_premium_moves(
        _legs(_contract("AAA", 100.0, prices) + _contract("AAA", 105.0, decoy)),
        _entries(("AAA", 100.0, SESSIONS[0])),
        horizons=[2],
    )
    assert moves["h2"].item() == pytest.approx(0.2)  # 12.0 / 10.0 - 1


def test_a_gap_yields_no_move_rather_than_a_mistimed_one() -> None:
    """Skipping a session must null the horizon, not silently shorten it."""
    quoted = {SESSIONS[0]: 10.0, SESSIONS[1]: 11.0, SESSIONS[3]: 20.0, SESSIONS[4]: 21.0}
    # A second symbol keeps SESSIONS[2] in the chains, so the offset still counts it.
    other = _contract("CCC", 50.0, dict(zip(SESSIONS, [5.0] * 5, strict=True)))
    moves = straddle_premium_moves(
        _legs(_contract("BBB", 100.0, quoted) + other),
        _entries(("BBB", 100.0, SESSIONS[0])),
        horizons=[2],
    )
    assert moves["h2"].item() is None


def test_only_the_supplied_entries_are_followed() -> None:
    """A contract's whole life is read, but only selected entries come back."""
    prices = dict(zip(SESSIONS, [10.0, 11.0, 12.0, 13.0, 14.0], strict=True))
    moves = straddle_premium_moves(
        _legs(_contract("AAA", 100.0, prices)),
        _entries(("AAA", 100.0, SESSIONS[1]), ("AAA", 100.0, SESSIONS[2])),
        horizons=[1],
    )
    assert sorted(moves["timestamp"].to_list()) == SESSIONS[1:3]
    assert moves.sort("timestamp")["h1"].to_list() == pytest.approx([1 / 11, 1 / 12])


def test_an_unpaired_leg_is_dropped() -> None:
    """A session quoting only the call is not a straddle."""
    prices = dict(zip(SESSIONS, [10.0, 11.0, 12.0, 13.0, 14.0], strict=True))
    rows = [leg for leg in _contract("AAA", 100.0, prices) if leg["timestamp"] != SESSIONS[1]]
    rows.append(
        {
            "symbol": "AAA",
            "strike": 100.0,
            "expiration": EXPIRY,
            "timestamp": SESSIONS[1],
            "call_put": "C",
            "bid": 5.0,
            "ask": 5.5,
        }
    )
    moves = straddle_premium_moves(
        _legs(rows),
        _entries(("AAA", 100.0, SESSIONS[0]), ("AAA", 100.0, SESSIONS[1])),
        horizons=[1],
    )
    assert moves["timestamp"].to_list() == [SESSIONS[0]]  # the unpaired session is not an entry
    assert moves["h1"].item() is None  # and SESSIONS[1] is not a destination either
