"""A wider risk-overlay width extends the narrow selection rather than reordering it.

`test_risk_width_is_overridable.py` asserts every risk notebook *reads* its declared width; it
reads source and cannot say what a width of three actually selects. This one runs the selection.

Two properties, and each is a way the widening could be wrong while the parameter is plainly
honoured:

- **the first parent never moves.** Every recorded run selected at the declared width of one, so
  a widening that reordered the ranking would silently restate published results as something
  else. `ranked[:n]` has to be a prefix, not a re-selection.
- **a width above the population size is the population**, not an error and not a short read,
  because the expansion launches at 999 against sets of tens.

`crypto_perps_funding`'s helper is the one exercised directly, because it is the only one of the
three that is a function rather than an expression inside a notebook cell. `cme_futures` and
`fx_pairs` slice the same kind of ranking inline - `rank_by_validation_sharpe(...)[:n]` and
`sorted(eligible, key=_eligible_order)[:n]` - and the prefix property is what those slices rest
on, asserted here on the shared ranking rule they all use.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from case_studies.research import comparison
from case_studies.research.comparison import CandidateSet


@dataclass(frozen=True)
class _Opened:
    hash: str
    complete: bool = True


class _Backtests:
    def __init__(self, table: pl.DataFrame) -> None:
        self._table = table

    def table(self, **_: object) -> pl.DataFrame:
        return self._table


class _Study:
    def __init__(self, table: pl.DataFrame) -> None:
        self.backtests = _Backtests(table)


def _set(monkeypatch, rows: list[tuple[str, float]]) -> CandidateSet:
    monkeypatch.setattr(
        comparison,
        "Result",
        type("R", (), {"open": staticmethod(lambda _study, value, **_: _Opened(value))}),
    )
    table = pl.DataFrame(
        {
            "backtest_hash": [name for name, _ in rows],
            "sharpe": [sharpe for _, sharpe in rows],
            "split": ["validation"] * len(rows),
            "execution_tier": ["canonical"] * len(rows),
            "stage": ["allocation"] * len(rows),
            "ruin": [0.0] * len(rows),
        },
        schema_overrides={"sharpe": pl.Float64, "ruin": pl.Float64},
    )
    return CandidateSet(
        study=_Study(table),
        hash="set1",
        name="crypto-signal-allocation-fwd_ret_1d",
        member_kind="backtest",
        members=tuple(name for name, _ in rows),
        comparison_contract={},
    )


ROWS = [("mid", 1.0), ("top", 3.0), ("low", -1.0), ("second", 2.0)]


def test_the_declared_width_selects_what_it_always_did(monkeypatch) -> None:
    candidate_set = _set(monkeypatch, ROWS)
    widened = candidate_set.ranked_validation_sharpe(limit=1)
    assert [result.hash for result in widened] == ["top"]
    assert candidate_set.best_validation_sharpe().hash == "top", (
        "the single-parent path the recorded runs took and the limit=1 slice that replaced it "
        "have to name the same configuration, or every published overlay is attributed to a "
        "parent it was not run over"
    )


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_a_wider_slice_is_a_prefix_of_the_narrow_one(monkeypatch, width: int) -> None:
    candidate_set = _set(monkeypatch, ROWS)
    full = [result.hash for result in candidate_set.ranked_validation_sharpe()]
    assert full == ["top", "second", "mid", "low"]
    assert [result.hash for result in candidate_set.ranked_validation_sharpe(limit=width)] == (
        full[:width]
    )


def test_a_width_above_the_population_returns_the_population(monkeypatch) -> None:
    """The expansion launches at 999 against sets of tens; that is not an error."""
    candidate_set = _set(monkeypatch, ROWS)
    assert len(candidate_set.ranked_validation_sharpe(limit=999)) == len(ROWS)


def test_a_width_below_one_is_refused(monkeypatch) -> None:
    """Zero parents is an empty overlay grid reported as a completed sweep."""
    candidate_set = _set(monkeypatch, ROWS)
    with pytest.raises(ValueError, match="limit must be positive"):
        candidate_set.ranked_validation_sharpe(limit=0)
