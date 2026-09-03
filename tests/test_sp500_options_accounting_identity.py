"""The specialized option accounting identity covers the settlement regime guard.

The guard decides whether a position held past a corporate action settles the listed strike
against a restated underlying or falls to the liquidation path, and the two book different
returns. A hashed spec that does not name it lets a run completed before the guard be served
from the registry afterwards, returning pre-fix numbers under a corrected engine.
"""

from __future__ import annotations

import pytest

from case_studies.sp500_options._htm_backtest import (
    SPLIT_GUARD_HIGH,
    SPLIT_GUARD_LOW,
    option_accounting_parameters,
)
from case_studies.utils.registry.specs import canonical_json, compute_hash


def _hash(spec: dict) -> str:
    return compute_hash(canonical_json(spec))


@pytest.fixture
def signal() -> dict:
    return {"n_roll": 4, "option_spread_fraction": 1.0}


def test_guard_band_is_part_of_the_identity(signal: dict) -> None:
    params = option_accounting_parameters(signal)
    assert params["settlement_regime_guard_low"] == SPLIT_GUARD_LOW
    assert params["settlement_regime_guard_high"] == SPLIT_GUARD_HIGH
    assert params["schema_version"] >= 2


def test_moving_the_band_moves_the_hash(signal: dict) -> None:
    params = option_accounting_parameters(signal)
    widened = dict(params, settlement_regime_guard_low=SPLIT_GUARD_LOW / 2)
    assert _hash(params) != _hash(widened)


def test_schema_version_moves_the_hash(signal: dict) -> None:
    params = option_accounting_parameters(signal)
    previous = dict(params, schema_version=1)
    assert _hash(params) != _hash(previous)
