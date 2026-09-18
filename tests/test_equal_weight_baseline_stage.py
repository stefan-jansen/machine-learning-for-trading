"""The equal-weight baseline lives at the signal stage, not in the allocator menu.

`20_strategy_synthesis/05_portfolio_allocation` read its baseline as
`allocator == "equal_weight"` over allocation-stage rows. `equal_weight` left every case
study's allocator menu on the ruling that equal weight IS the baseline, so the filter matched
nothing and the section named "Equal-Weight Baseline vs Best Allocator" reported no number for
any case study.
"""

import json

import pytest

from case_studies.utils.analytics import extract_allocator, is_unallocated

# The three spec shapes the nine registries actually hold, read 2026-09-18.
SIGNAL_STAGE = {
    "rebalance": {"cadence": "monthly_month_end", "mode": "engine", "step": 1},
    "signal": {"long_short": False, "method": "equal_weight_top_k", "top_k": 5},
}
SIGNAL_STAGE_SPELLED_OUT = {
    "allocation": {"long_short": True, "method": "equal_weight", "top_k": 5},
    "rebalance": {"cadence": "15_minute", "mode": "engine", "step": 1},
    "signal": {"long_short": True, "method": "equal_weight_top_k", "top_k": 5},
}
ALLOCATION_STAGE = {
    "allocation": {"long_short": False, "method": "hrp", "top_k": 10},
    "rebalance": {"cadence": "monthly_month_end", "mode": "engine", "step": 1},
    "signal": {"long_short": False, "method": "equal_weight_top_k", "top_k": 10},
}


def _spec(strategy: dict) -> str:
    """The canonical envelope `strategy_view` recognizes - version 2 with a backtest_config."""
    return json.dumps(
        {
            "version": 2,
            "chapter": "ch17",
            "preset_id": "test",
            "backtest_config": {"commission": {"rate": 0.0}, "slippage": {"rate": 0.0}},
            "strategy": strategy,
        }
    )


def test_a_signal_stage_row_with_no_allocation_block_is_the_baseline():
    assert is_unallocated(_spec(SIGNAL_STAGE))


def test_the_older_spelling_is_the_same_baseline():
    """30 nasdaq100_microstructure rows write `allocation.method = equal_weight` instead."""
    assert is_unallocated(_spec(SIGNAL_STAGE_SPELLED_OUT))


def test_an_allocator_is_not_the_baseline():
    assert not is_unallocated(_spec(ALLOCATION_STAGE))


@pytest.mark.parametrize(
    "method",
    [
        "hrp",
        "risk_parity",
        "inverse_vol",
        "score_weighted",
        "mvo_ledoit_wolf",
        "conformal_weighted",
    ],
)
def test_no_allocator_on_the_menu_counts_as_the_baseline(method):
    strategy = {**ALLOCATION_STAGE, "allocation": {"method": method, "top_k": 10}}
    assert not is_unallocated(_spec(strategy))


def test_the_allocator_name_cannot_find_the_baseline():
    """Why the filter had to change: `extract_allocator` cannot see an absent block."""
    assert extract_allocator(_spec(SIGNAL_STAGE)) == "unknown"
    assert extract_allocator(_spec(SIGNAL_STAGE)) != "equal_weight"
