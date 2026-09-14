"""A leaderboard row identical to one above it in every column is not a second result.

A configuration keeps its results when a spec field is added that it never set: the
backtest identity is new, the numbers are not. `best` ranks over every generation in
the registry, so both rows compete and a ten-row table can show five configurations.

Measured 2026-09-14: `us_firm_characteristics` holds 80 of 240 allocation configurations
and 912 of 2,276 signal configurations twice, `etfs` 424 of 1,767, `crypto_perps_funding`
273 of 2,109. One pair's specs differ in `account.lock_notional_update_mode` and
`position_sizing.share_rounding` alone, both implicit defaults the schema made explicit.

The rule is deliberately the narrowest one that fixes it, and these tests pin both
halves: a row identical in every column but `backtest_hash` is dropped, and a row that
differs anywhere a caller can see is kept.
"""

from __future__ import annotations

import polars as pl

from case_studies.utils.backtest_explorer import _drop_rows_a_reader_cannot_tell_apart


def _frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


BASE = {
    "backtest_hash": "aaaa",
    "prediction_hash": "p1",
    "source": "gbm/leaves_31_huber",
    "sharpe": 3.5829,
    "top_k": 20,
}


def test_a_re_keyed_identity_collapses_to_one_row() -> None:
    df = _frame([BASE, {**BASE, "backtest_hash": "bbbb"}])
    out = _drop_rows_a_reader_cannot_tell_apart(df)
    assert out.height == 1
    assert out["backtest_hash"].to_list() == ["aaaa"], "the first row under the query order wins"


def test_a_row_differing_anywhere_visible_is_kept() -> None:
    """The negative control. Without it the rule could collapse everything and still pass."""
    cases = [
        ("sharpe", 3.5828),
        ("top_k", 50),
        ("source", "gbm/leaves_31_mae"),
        ("prediction_hash", "p2"),
    ]
    for column, other in cases:
        df = _frame([BASE, {**BASE, "backtest_hash": "bbbb", column: other}])
        out = _drop_rows_a_reader_cannot_tell_apart(df)
        assert out.height == 2, f"rows differing in {column} are two results"


def test_a_float_difference_below_display_precision_is_still_a_difference() -> None:
    """Four of the measured pairs agree only past the sixth decimal; they are not collapsed.

    The rule keys on the value, not on how it renders, so a pair the table would print
    identically still counts as two rows. That is the conservative direction: this
    function never removes a row whose numbers differ.
    """
    df = _frame([BASE, {**BASE, "backtest_hash": "bbbb", "sharpe": 3.5829000001}])
    assert _drop_rows_a_reader_cannot_tell_apart(df).height == 2


def test_nulls_compare_equal_so_a_re_keyed_row_with_missing_metrics_collapses() -> None:
    row = {**BASE, "sharpe": None}
    df = _frame([row, {**row, "backtest_hash": "bbbb"}])
    assert _drop_rows_a_reader_cannot_tell_apart(df).height == 1


def test_order_is_preserved() -> None:
    rows = [
        {**BASE, "backtest_hash": "aaaa", "sharpe": 3.6},
        {**BASE, "backtest_hash": "bbbb", "sharpe": 3.6},
        {**BASE, "backtest_hash": "cccc", "sharpe": 3.4},
    ]
    out = _drop_rows_a_reader_cannot_tell_apart(_frame(rows))
    assert out["sharpe"].to_list() == [3.6, 3.4]
    assert out["backtest_hash"].to_list() == ["aaaa", "cccc"]
