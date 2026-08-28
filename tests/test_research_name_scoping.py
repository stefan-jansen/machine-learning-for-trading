"""Isolating a narrowed run has to isolate the whole chain, not just what it writes.

`POPULATION_NAME` gave a run its own name for the population it publishes. Nothing gave
it its own name for the population it reads, so 14/15/16 resolved their upstream by the
canonical name while publishing under the isolated one. In a workspace holding the
canonical populations that allocates over the full baselines and freezes the result under
a name that says it is narrowed; in a fresh workspace it raises instead, so the version
that corrupts is the one that only happens where the data already exists.
"""

from __future__ import annotations

import ast
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


def _is_research_name(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "research_name"
    )


def test_every_population_name_in_the_fx_phase_two_chain_is_scoped() -> None:
    """The reader/writer agreement is only observable across notebooks, so this reads them.

    A unit test of the helper cannot see the defect: both sides called a correct function,
    and one of them was not calling it at all. What fails here against the previous code is
    `14_portfolio_management` resolving `f"{CASE_STUDY_ID}:equal-weight-baselines"` directly
    while `13_backtest` published a scoped name.

    Read with `ast` rather than a regex. The regex this replaced required that no `)` appear
    between the call's opening paren and `name=`, so `OfficialPopulation.one(open_study(...),
    name=...)` produced no match at all and the violation list stayed empty - it passed on
    exactly the code it exists to catch. A checker that finds nothing has to be
    indistinguishable from a checker that finds nothing wrong, which is why the count below
    is asserted as well as the violations.
    """
    receivers = {"OfficialPopulation", "CandidateSet"}
    methods = {"one", "create"}
    unscoped: list[str] = []
    checked = 0
    for stem in ("13_backtest", "14_portfolio_management", "15_costs", "16_risk_management"):
        tree = ast.parse((NOTEBOOKS / f"{stem}.py").read_text(encoding="utf-8"))
        # A name may be bound to a local first. That is not a way around the rule - the binding
        # still has to come from `research_name` - and it is how a notebook passes one name to
        # both `create` and `population_supersedes` without writing the call twice, which is
        # where two copies of a name would drift apart. Only a binding whose right-hand side is
        # a `research_name(` call counts, so an f-string assigned to a variable still fails.
        from_research_name = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and _is_research_name(node.value)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in methods:
                continue
            if not (isinstance(func.value, ast.Name) and func.value.id in receivers):
                continue
            name_kw = next((k for k in node.keywords if k.arg == "name"), None)
            if name_kw is None:
                continue
            checked += 1
            expr = name_kw.value
            scoped = _is_research_name(expr) or (
                isinstance(expr, ast.Name) and expr.id in from_research_name
            )
            if not scoped:
                unscoped.append(f"{stem}: {func.value.id}.{func.attr} name={ast.unparse(expr)}")

    assert not unscoped, (
        "a population name that does not go through research_name cannot be isolated, "
        "so a narrowed run reads or writes the canonical one: " + "; ".join(unscoped)
    )
    assert checked >= len(CANONICAL), (
        f"only {checked} population name(s) found across the four notebooks, fewer than the "
        f"{len(CANONICAL)} canonical names they publish and read. The scan is not reaching "
        "the calls, so an unscoped name would not be reported either"
    )
