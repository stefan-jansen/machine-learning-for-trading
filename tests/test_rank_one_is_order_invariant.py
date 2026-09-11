"""The reported winner is a function of the data, not of how the frame arrived.

`ml4t/agent-workspace#333` is the class where a visible precaution - a seed, a sort - pins one
dimension of randomness and leaves another free. Here the precaution is `sort(by,
descending=True)`, and what it leaves free is the tie: `head(1)` then returns whichever tied
row happened to come first, so the allocator or overlay a case study reports moves between
runs that computed identical numbers.

Exact ties in these quantities are measured, not hypothetical: across the nine live registries
on 2026-09-11, 14 (case study, stage) pairs hold at least one exactly repeated Sharpe, with
multiplicity up to 6, and one stage holds two rows tied at the maximum.
"""

from __future__ import annotations

import itertools

import polars as pl

from case_studies.utils.strategy_analysis import rank_one


def _tied_at_the_top() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "risk_name": ["trailing_stop", "daily_loss", "max_drawdown"],
            "sharpe": [1.25, 1.25, 0.80],
        }
    )


def test_a_tie_at_the_top_resolves_the_same_way_from_every_row_order():
    """Every permutation of a tied frame reports the same winner."""
    frame = _tied_at_the_top()
    winners = {
        rank_one(frame[list(order)], by="sharpe", name="risk_name")["risk_name"][0]
        for order in itertools.permutations(range(frame.height))
    }
    assert winners == {"daily_loss"}


def test_the_one_key_sort_this_replaced_does_not_survive_the_same_permutations():
    """The control: without the named tiebreak the answer moves, which is the defect.

    Asserting the fix alone would pass just as well against a frame that never ties, and the
    point is that this frame does.
    """
    frame = _tied_at_the_top()
    winners = {
        frame[list(order)].sort("sharpe", descending=True).head(1)["risk_name"][0]
        for order in itertools.permutations(range(frame.height))
    }
    assert winners == {"trailing_stop", "daily_loss"}


def test_an_untied_frame_reports_the_largest_and_the_tiebreak_changes_nothing():
    frame = pl.DataFrame(
        {"allocator": ["hrp", "mean_variance", "equal_weight"], "best_sharpe": [0.9, 1.4, 0.3]}
    )
    for order in itertools.permutations(range(frame.height)):
        top = rank_one(frame[list(order)], by="best_sharpe", name="allocator")
        assert top["allocator"][0] == "mean_variance"
        assert top["best_sharpe"][0] == 1.4


def test_nulls_in_the_ranked_column_do_not_win():
    """polars sorts nulls FIRST under descending=True, so the default would report one.

    No live registry holds a null Sharpe today, so this is a contract rather than a repair of
    a current number - the frame is built by a join, and the code reading the result already
    guards a null max_drawdown.
    """
    frame = pl.DataFrame(
        {"risk_name": ["a", "b", "c"], "sharpe": [None, 0.5, 0.4]},
        schema={"risk_name": pl.String, "sharpe": pl.Float64},
    )
    assert rank_one(frame, by="sharpe", name="risk_name")["risk_name"][0] == "b"
