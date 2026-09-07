"""The canonical validation ranking demotes a bankrupt member instead of refusing.

ml4t/agent-workspace#1079, canonical half. The preview half is
`rank_by_validation_sharpe` in `cme_futures/research_workflow.py`, fixed in
f84b0ec0; the two apply one rule and have to keep agreeing.

The refusal here is not a filter that drops a row. The `sharpe.is_not_null()`
filter removes the member, and then a height check finds fewer rows than the set
has members and rejects the whole set. So a ruined member has to survive the
filter to be ranked at all - adding a sort key alone changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from case_studies.research import comparison
from case_studies.research.comparison import CandidateSet


@dataclass(frozen=True)
class _Opened:
    complete: bool = True


class _Backtests:
    def __init__(self, table: pl.DataFrame) -> None:
        self._table = table

    def table(self, **_: object) -> pl.DataFrame:
        return self._table


class _Study:
    def __init__(self, table: pl.DataFrame) -> None:
        self.backtests = _Backtests(table)


def _table(rows, *, with_ruin: bool = True) -> pl.DataFrame:
    data = {
        "backtest_hash": [r[0] for r in rows],
        "sharpe": [r[1] for r in rows],
        "split": ["validation"] * len(rows),
        "execution_tier": ["canonical"] * len(rows),
        "stage": ["signal"] * len(rows),
    }
    if with_ruin:
        data["ruin"] = [r[2] for r in rows]
    return pl.DataFrame(data, schema_overrides={"sharpe": pl.Float64, "ruin": pl.Float64})


def _rank(monkeypatch, rows, members, *, with_ruin: bool = True) -> list[str]:
    monkeypatch.setattr(
        comparison, "Result", type("R", (), {"open": staticmethod(lambda *_: _Opened())})
    )
    candidate_set = CandidateSet(
        study=_Study(_table(rows, with_ruin=with_ruin)),
        hash="set1",
        name="cohort",
        member_kind="backtest",
        members=tuple(members),
        comparison_contract={},
    )
    return list(candidate_set._ranked_validation_hashes())


def test_a_bankrupt_member_ranks_last_rather_than_rejecting_the_set(monkeypatch) -> None:
    """Eleven solvent members and one bankrupt one is a ranking of twelve."""
    rows = [("solid", 1.2, 0.0), ("broke", None, 1.0), ("better", 2.0, 0.0)]
    assert _rank(monkeypatch, rows, ["solid", "broke", "better"]) == ["better", "solid", "broke"]


def test_the_bankrupt_member_is_never_the_best(monkeypatch) -> None:
    """`best_validation_sharpe` takes the head, so ordering last is the protection."""
    rows = [("broke", None, 1.0), ("thin", -0.4, 0.0)]
    assert _rank(monkeypatch, rows, ["broke", "thin"])[0] == "thin"


def test_bankrupt_members_break_their_tie_on_identity(monkeypatch) -> None:
    """The same tie-break the rankable members get, for the same reason."""
    rows = [("b", None, 1.0), ("a", None, 1.0), ("live", 0.1, 0.0)]
    assert _rank(monkeypatch, rows, ["b", "a", "live"]) == ["live", "a", "b"]


def test_a_null_sharpe_without_the_flag_still_rejects_the_set(monkeypatch) -> None:
    """That is the other reason a Sharpe is null, and it is still an ineligible member."""
    rows = [("measured", 0.5, 0.0), ("unmeasured", None, 0.0)]
    with pytest.raises(ValueError, match="ineligible selection member"):
        _rank(monkeypatch, rows, ["measured", "unmeasured"])


def test_a_registry_written_before_the_ruin_column_keeps_the_old_rule(monkeypatch) -> None:
    """No column is not the same as no bankrupt member; there a null means one thing."""
    rows = [("measured", 0.5, None), ("unmeasured", None, None)]
    with pytest.raises(ValueError, match="ineligible selection member"):
        _rank(monkeypatch, rows, ["measured", "unmeasured"], with_ruin=False)


def test_an_all_solvent_set_ranks_exactly_as_before(monkeypatch) -> None:
    """The common case must not move."""
    rows = [("mid", 1.0, 0.0), ("top", 3.0, 0.0), ("low", -1.0, 0.0)]
    assert _rank(monkeypatch, rows, ["mid", "top", "low"]) == ["top", "mid", "low"]


def test_a_set_whose_members_all_went_bankrupt_has_nothing_to_select(monkeypatch) -> None:
    """Ordering last protects the selection only while something solvent is ahead.

    `best_validation_sharpe` takes the head unconditionally, so an all-bankrupt set
    would hand back a bankrupt selection - the outcome #920 exists to prevent,
    reached through the fix for it.
    """
    rows = [("a", None, 1.0), ("b", None, 1.0)]
    with pytest.raises(ValueError, match="no solvent member to select"):
        _rank(monkeypatch, rows, ["a", "b"])
