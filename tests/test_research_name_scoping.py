"""Isolating a narrowed run has to isolate the whole chain, not just what it writes.

`POPULATION_NAME` gave a run its own name for the population it publishes. Nothing gave
it its own name for the population it reads, so 14/15/16 resolved their upstream by the
canonical name while publishing under the isolated one. In a workspace holding the
canonical populations that allocates over the full baselines and freezes the result under
a name that says it is narrowed; in a fresh workspace it raises instead, so the version
that corrupts is the one that only happens where the data already exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from case_studies.research import research_name

NOTEBOOKS = Path(__file__).resolve().parents[1] / "case_studies" / "fx_pairs"

CANONICAL = [
    ("fx_pairs", "validation-predictions", "fx_pairs:validation-predictions"),
    ("fx_pairs", "equal-weight-baselines", "fx_pairs:equal-weight-baselines"),
    ("fx_pairs", "allocation-backtests", "fx_pairs:allocation-backtests"),
    ("fx_pairs", "cost-sensitivity-backtests", "fx_pairs:cost-sensitivity-backtests"),
    ("fx_pairs", "risk-overlay-backtests", "fx_pairs:risk-overlay-backtests"),
    ("fx_pairs", "holdout-candidates", "fx_pairs:holdout-candidates"),
    (
        "fx_pairs",
        "fwd_ret_5d:equal-weight-candidates",
        "fx_pairs:fwd_ret_5d:equal-weight-candidates",
    ),
]


@pytest.mark.parametrize(("case_study", "suffix", "expected"), CANONICAL)
def test_an_unscoped_name_is_the_published_name(case_study, suffix, expected) -> None:
    """A population is immutable per name, so these strings are not free to move. Routing
    them through a helper must reproduce them exactly; a changed separator would orphan
    every population already published under the old one."""
    assert research_name(case_study, suffix) == expected
    assert research_name(case_study, suffix, scope="") == expected


def test_a_scope_isolates_every_name_it_touches() -> None:
    """The defect was a name that stayed canonical while its sibling was isolated, so the
    property under test is that no canonical name survives scoping."""
    scope = "fx_pairs:preflight-baselines"
    scoped = {research_name("fx_pairs", suffix, scope=scope) for _, suffix, _ in CANONICAL}
    canonical = {expected for _, _, expected in CANONICAL}

    assert scoped.isdisjoint(canonical)
    assert len(scoped) == len(CANONICAL), "scoping collapsed two distinct names into one"
    assert all(name.startswith(f"{scope}:") for name in scoped)


def test_every_population_name_in_the_fx_phase_two_chain_is_scoped() -> None:
    """The reader/writer agreement is only observable across notebooks, so this reads them.

    A unit test of the helper cannot see the defect: both sides called a correct function,
    and one of them was not calling it at all. What fails here against the previous code is
    `14_portfolio_management` resolving `f"{CASE_STUDY_ID}:equal-weight-baselines"` directly
    while `13_backtest` published a scoped name.
    """
    pattern = re.compile(r"(OfficialPopulation|CandidateSet)\.(one|create)\(([^)]*?)\bname=([^,]+)")
    unscoped: list[str] = []
    for stem in ("13_backtest", "14_portfolio_management", "15_costs", "16_risk_management"):
        source = (NOTEBOOKS / f"{stem}.py").read_text(encoding="utf-8")
        for match in pattern.finditer(source):
            name_expr = match.group(4).strip()
            if not name_expr.startswith("research_name("):
                unscoped.append(f"{stem}: {match.group(1)}.{match.group(2)} name={name_expr}")

    assert not unscoped, (
        "a population name that does not go through research_name cannot be isolated, "
        "so a narrowed run reads or writes the canonical one: " + "; ".join(unscoped)
    )
