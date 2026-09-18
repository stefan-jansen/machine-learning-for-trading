"""The equal-weight baseline lives at the signal stage, not in the allocator menu.

`20_strategy_synthesis/05_portfolio_allocation` read its baseline as
`allocator == "equal_weight"` over allocation-stage rows. `equal_weight` left every case
study's allocator menu on the ruling that equal weight IS the baseline, so the filter matched
nothing and the section named "Equal-Weight Baseline vs Best Allocator" reported no number for
any case study.
"""

import ast
import json
from pathlib import Path

import polars as pl
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


# --- The scatter interpretation must not outrun the points it describes -------------------
#
# An earlier version asserted that every point sat in one region and that no weak-signal case
# study had reached the comparison. That was true of the single point the broken equal-weight
# filter left behind, and the cell kept printing it once there were eight. `MAX_CASE_STUDIES`
# makes a one-row and a single-sign frame reachable, so both are tested here.

NOTEBOOK_ALLOC = Path(__file__).parents[1] / "20_strategy_synthesis" / "05_portfolio_allocation.py"


def _load_interpretation():
    tree = ast.parse(NOTEBOOK_ALLOC.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "uplift_interpretation"
    )
    namespace: dict = {"pl": pl}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK_ALLOC), "exec"),
        namespace,
    )
    return namespace["uplift_interpretation"]


def _frame(rows: list[tuple[str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "display_name": [r[0] for r in rows],
            "ew_sharpe": [r[1] for r in rows],
            "uplift": [r[2] for r in rows],
        }
    )


def test_one_case_study_is_not_both_the_largest_gain_and_the_largest_loss():
    text = _load_interpretation()(_frame([("ETFs", 0.65, 0.23)]))
    assert "largest gain is ETFs" in text
    assert "largest loss" not in text


def test_all_positive_uplifts_do_not_get_a_loss_sentence():
    text = _load_interpretation()(_frame([("ETFs", 0.65, 0.23), ("FX", 0.20, 0.11)]))
    assert "largest loss" not in text
    assert "helps in every case study here" in text


def test_all_negative_uplifts_do_not_get_a_gain_sentence():
    text = _load_interpretation()(_frame([("Crypto", 1.56, -0.73), ("US Equities", 1.05, -1.11)]))
    assert "largest gain" not in text
    assert "hurts in every case study here" in text


def test_a_single_sign_result_withholds_the_conclusion_about_baseline_strength():
    text = _load_interpretation()(_frame([("ETFs", 0.65, 0.23)]))
    assert "cannot say whether" in text
    assert "not decided by it alone" not in text


def test_overlapping_baselines_support_the_conclusion():
    """Helped and hurt span the same baseline range, so strength does not separate them."""
    text = _load_interpretation()(
        _frame([("ETFs", 0.65, 0.23), ("SP500", 1.60, 0.26), ("Crypto", 1.56, -0.73)])
    )
    assert "baselines overlap" in text
    assert "not decided by it alone" in text


def test_a_separation_in_the_predicted_direction_says_so():
    """The mechanism predicts allocation is harmful where the ranking is weak.

    So the separation it predicts has the hurt group's baselines BELOW the helped group's -
    weak signal, allocator redistributing noise. Here the loss sits at 0.20 and the gains at
    2.80 and 3.00.
    """
    text = _load_interpretation()(
        _frame([("ETFs", 3.00, 0.23), ("SP500", 2.80, 0.26), ("Crypto", 0.20, -0.73)])
    )
    assert "hurts has a weaker baseline than every one it helps" in text
    assert "the direction the mechanism below predicts" in text
    assert "not decidable from these" in text


def test_a_separation_in_the_reverse_direction_is_not_called_support():
    """Losses at the strong end and gains at the weak end contradict the mechanism."""
    text = _load_interpretation()(
        _frame([("ETFs", 0.20, 0.23), ("SP500", 0.30, 0.26), ("Crypto", 3.00, -0.73)])
    )
    assert "hurts has a stronger baseline than every one it helps" in text
    assert "opposite of what the mechanism below predicts" in text
    assert "the direction the mechanism below predicts" not in text


def test_a_zero_uplift_is_counted_as_neither_helped_nor_hurt():
    text = _load_interpretation()(_frame([("ETFs", 0.65, 0.23), ("FX", 0.20, 0.0)]))
    assert "helps in 1 of them and hurts in 0, and changes nothing in 1." in text
    assert "helps in every case study here" not in text
    assert "never hurts here" in text


def test_an_all_zero_frame_does_not_claim_allocation_hurts_everywhere():
    text = _load_interpretation()(_frame([("ETFs", 0.65, 0.0), ("FX", 0.20, 0.0)]))
    assert "changes nothing in any case study here" in text
    assert "hurts in every case study here" not in text
    assert "largest gain" not in text and "largest loss" not in text


def test_an_empty_frame_reports_an_empty_plane():
    assert "plane is empty" in _load_interpretation()(_frame([]))
