"""A bound on prediction age changes the result, so it has to change the identity.

ml4t/agent-workspace#1135. `17_costs.py` aligns minute predictions onto coarser bars with a
backward as-of join. That join produces a null only *before* a symbol's series begins, so
dropping nulls trims the leading edge and nothing else: every bar after a symbol's series
*ends* reuses its last score for as long as the panel runs. Measured on this registry's
`fwd_dir_15m` predictions, eleven of 113 symbols stop mid-sample and the oldest match the
join produced was 252 sessions - the whole panel.

Bounding that age fixes what the backtest is computed from and touches neither
`prediction_hash` nor the strategy spec, so without a declaration a bounded run and an
unbounded one hash alike and the second is served the first's result. That is
ml4t/agent-workspace#911 exactly, one level over: there the reduction was the price panel,
here it is the aligned predictions, and in both the caller has to declare what it did before
it hashes.

`prediction_age_declaration` builds the declaration and
`build_backtest_spec(prediction_age=...)` puts it in `strategy.signal`, which is hashed
whole. A caller that declares nothing produces the spec it produced before the parameter
existed, which is what leaves every registered backtest at the identity it was written
under - the first two tests hold both halves of that.
"""

from __future__ import annotations

from copy import deepcopy

import polars as pl
import pytest

from case_studies.utils.backtest_loaders import get_backtest_config
from case_studies.utils.backtest_presets import build_backtest_spec, prediction_age_declaration
from case_studies.utils.registry.specs import backtest_hash_from_parts

CASE_STUDY = "nasdaq100_microstructure"


def _prices() -> pl.DataFrame:
    timestamps = pl.datetime_range(
        pl.datetime(2020, 6, 30), pl.datetime(2020, 7, 30), "1d", eager=True
    )
    return pl.DataFrame(
        {"timestamp": timestamps, "symbol": "AAPL", "close": 100.0},
    )


def _spec(**kwargs) -> dict:
    config = get_backtest_config(CASE_STUDY)
    base = dict(
        prices=_prices(),
        prediction_hash="p" * 12,
        initial_cash=1_000_000.0,
        signal={"method": "equal_weight_top_k", "top_k": 5},
        chapter="ch18",
        label=config.primary_label,
    )
    return build_backtest_spec(CASE_STUDY, config, **deepcopy(base), **kwargs)


def _hash(spec: dict) -> str:
    return backtest_hash_from_parts("p" * 12, spec)


def _declaration(*, dropped: int = 19_667, kept: int = 283_974, symbols: int = 12) -> dict:
    return prediction_age_declaration(
        max_age_sessions=1, dropped=dropped, kept=kept, symbols_dropped=symbols
    )


def test_a_cadence_that_drops_nothing_hashes_exactly_as_it_did_before() -> None:
    """The compatibility half, and why the key is emitted only when it is earned."""
    without = _spec()
    explicit_none = _spec(prediction_age=None)

    assert "prediction_age" not in without["strategy"]["signal"]
    assert _hash(without) == _hash(explicit_none)


def test_a_bound_that_drops_rows_gets_an_identity_of_its_own() -> None:
    """The defect. Without this the fix is invisible to a registry that already holds a row.

    `prediction_hash` names the prediction set, not the subset of it this run was computed
    from, and the strategy spec is untouched by a filter on the aligned frame. So a warm
    re-run of an already-swept cadence returns the unbounded numbers and nothing says so.
    """
    assert _hash(_spec()) != _hash(_spec(prediction_age=_declaration()))


def test_two_different_drops_are_two_identities() -> None:
    """The counts are in the declaration, not only the bound.

    Two runs can declare the same tolerance and be computed from different rows: the
    prediction set behind them moves on every refit, and a universe that left earlier is a
    different portfolio at the same bound. A declaration carrying only `max_age_sessions`
    would collapse those into one identity, which is the failure this whole key exists to
    prevent rather than a smaller version of it.
    """
    thirty_minute = _spec(prediction_age=_declaration(dropped=19_667, kept=283_974))
    four_hour = _spec(prediction_age=_declaration(dropped=3_593, kept=51_762))

    assert _hash(thirty_minute) != _hash(four_hour)


def test_a_bound_that_removed_nothing_cannot_be_declared() -> None:
    """A declaration for a bound that did nothing would re-key a run for no change in it.

    The refusal is what keeps the compatibility half above true by construction rather than
    by the caller remembering: there is no way to spell "bounded, dropped zero" in a spec.
    """
    with pytest.raises(ValueError, match="removed rows"):
        prediction_age_declaration(max_age_sessions=1, dropped=0, kept=283_974, symbols_dropped=0)


def test_the_declaration_reaches_the_hashed_block_rather_than_sitting_beside_it() -> None:
    """Where it lands is the whole mechanism, so it is asserted rather than assumed.

    `strategy.signal` is hashed whole. A key written anywhere `_hashable_strategy_spec`
    strips, or outside `strategy`, would read as a declaration and change no identity - which
    is indistinguishable from the defect it is meant to close.
    """
    spec = _spec(prediction_age=_declaration())

    assert spec["strategy"]["signal"]["prediction_age"] == {
        "max_age_sessions": 1,
        "dropped": 19_667,
        "kept": 283_974,
        "symbols_dropped": 12,
    }
