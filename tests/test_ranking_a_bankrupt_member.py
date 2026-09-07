"""A bankrupt backtest ranks last; an unmeasured one still refuses the ranking.

ml4t/agent-workspace#1079. The ruin stop (#920) registers `sharpe` as null for a
path whose equity reached zero, on purpose - a bankrupt path must not produce a
rankable Sharpe. `rank_by_validation_sharpe` read a null Sharpe as "not measured"
and refused the whole ranking, so a sweep with one bankrupt member and several
solvent ones ranked nothing at all. Measured on CI run 34157373964, where
cme_futures 14_portfolio_management, 15_risk_management, 16_costs and
19_strategy_analysis all failed at that refusal once #1003 let the sweep register
backtests that trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from case_studies.cme_futures.research_workflow import rank_by_validation_sharpe


@dataclass(frozen=True)
class _Result:
    hash: str


class _Backtests:
    def __init__(self, table: pl.DataFrame) -> None:
        self._table = table

    def table(self, *, include_preview: bool = False) -> pl.DataFrame:
        return self._table


class _Study:
    def __init__(self, table: pl.DataFrame) -> None:
        self.backtests = _Backtests(table)


def _table(rows: list[tuple[str, float | None, float | None]], *, with_ruin: bool = True):
    data = {
        "backtest_hash": [r[0] for r in rows],
        "sharpe": [r[1] for r in rows],
    }
    if with_ruin:
        data["ruin"] = [r[2] for r in rows]
    return pl.DataFrame(data, schema_overrides={"sharpe": pl.Float64, "ruin": pl.Float64})


def _rank(rows, hashes, *, with_ruin=True):
    study = _Study(_table(rows, with_ruin=with_ruin))
    return [r.hash for r in rank_by_validation_sharpe(study, [_Result(h) for h in hashes])]


def test_a_bankrupt_member_ranks_last_instead_of_refusing_the_set() -> None:
    """One ruined member and two solvent ones is a ranking of three, not an error."""
    rows = [("solid", 1.2, 0.0), ("broke", None, 1.0), ("better", 2.0, 0.0)]
    assert _rank(rows, ["solid", "broke", "better"]) == ["better", "solid", "broke"]


def test_the_bankrupt_member_is_never_the_selection() -> None:
    """`best_validation_sharpe` takes the head, so last is the whole protection."""
    rows = [("broke", None, 1.0), ("thin", -0.4, 0.0)]
    assert _rank(rows, ["broke", "thin"])[0] == "thin"


def test_several_bankrupt_members_keep_a_deterministic_order() -> None:
    """Ties among the unrankable break on identity, as they do among the rankable."""
    rows = [("b", None, 1.0), ("a", None, 1.0), ("live", 0.1, 0.0)]
    assert _rank(rows, ["b", "a", "live"]) == ["live", "a", "b"]


def test_a_null_sharpe_without_the_ruin_flag_still_refuses() -> None:
    """That is the other reason a Sharpe is null, and it is still a broken run."""
    rows = [("measured", 0.5, 0.0), ("unmeasured", None, 0.0)]
    with pytest.raises(ValueError, match="no validation Sharpe recorded"):
        _rank(rows, ["measured", "unmeasured"])


def test_a_null_ruin_flag_is_not_a_bankruptcy() -> None:
    """Null in that column means the run predates the measurement, not that it survived."""
    rows = [("measured", 0.5, None), ("unmeasured", None, None)]
    with pytest.raises(ValueError, match="no validation Sharpe recorded"):
        _rank(rows, ["measured", "unmeasured"])


def test_a_registry_without_the_column_keeps_the_old_rule() -> None:
    """Written before #920: no `ruin` column at all, so a null Sharpe means one thing."""
    rows = [("measured", 0.5, None), ("unmeasured", None, None)]
    with pytest.raises(ValueError, match="no validation Sharpe recorded"):
        _rank(rows, ["measured", "unmeasured"], with_ruin=False)


def test_an_all_solvent_ranking_is_unchanged() -> None:
    """The common case has to sort exactly as it did before the ruin column existed."""
    rows = [("mid", 1.0, 0.0), ("top", 3.0, 0.0), ("low", -1.0, 0.0)]
    assert _rank(rows, ["mid", "top", "low"]) == ["top", "mid", "low"]
