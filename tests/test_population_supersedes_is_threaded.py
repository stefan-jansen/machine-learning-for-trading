"""Every writer of an official population must be able to name the generation it retires.

`OfficialPopulation.create` refuses a changed member list under an existing name unless the
caller says which generation it replaces, and it refuses at write time. A wrapper that does not
take `supersedes` therefore cannot answer that refusal from a notebook parameter: the first time
anything moves an identity, the sweep raises and the only way forward is to edit the module. That
is what happened to `cme_futures`' four backtest sweeps and it is the class this test closes -
`crypto_perps_funding`'s `freeze_official_model_population` documents the same gap.

The check is over source rather than over a live registry because the defect is structural: the
wrapper compiles, runs, and produces correct populations right up to the run that needs the
parameter. Nothing at runtime distinguishes a wrapper that threads `supersedes` from one that does
not until that run, so there is no earlier behavioural signal to assert on.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((REPO / "case_studies").glob("*/research_workflow.py"))


def _enclosing_function(tree: ast.Module, node: ast.AST) -> ast.FunctionDef | None:
    for candidate in ast.walk(tree):
        if isinstance(candidate, ast.FunctionDef) and node in set(ast.walk(candidate)):
            return candidate
    return None


def _population_creates(tree: ast.Module) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "OfficialPopulation"
    ]


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.parent.name)
def test_every_population_writer_threads_supersedes(path: Path) -> None:
    tree = ast.parse(path.read_text())
    calls = _population_creates(tree)
    if not calls:
        pytest.skip(f"{path.parent.name} writes no official population here")
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "supersedes" in keywords, (
            f"{path.relative_to(REPO)}: OfficialPopulation.create on line {call.lineno} does not "
            "pass supersedes, so no caller can name the generation it retires"
        )
        function = _enclosing_function(tree, call)
        assert function is not None, f"{path.relative_to(REPO)}: create call outside a function"
        parameters = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
        assert "supersedes" in parameters, (
            f"{path.relative_to(REPO)}: {function.name} passes supersedes but does not take it, "
            "so the value cannot come from a notebook parameter"
        )


def test_the_cme_backtest_sweeps_still_have_no_notebook_parameter() -> None:
    """The half of ml4t/agent-workspace#1009 that is fixed in the module but not in the notebooks.

    `run_official_backtest_requests` now takes `supersedes`, so nothing has to edit the module.
    The sweeps that publish a `cme_futures` backtest population passed nothing, because adding a
    papermill parameter changes the paired `.py` and the provenance gate then requires the
    notebook to be re-executed - and under `rebalance.step` entering the backtest identity
    (public 83141459), a re-run resolves 992 hashes none of which is registered, so it would
    supersede the whole baseline population to land a parameter.

    **That reason has expired, and `13_backtest` is the first of the four to move.** The cost it
    described was triggering a re-run that supersedes the baseline population. cme_futures is
    being regenerated from stage 04 forward - its `model_based.parquet` already carries a `fold`
    column its own notebook can no longer emit - so that supersession is happening regardless of
    this parameter, and there is no longer a cost to avoid. The notebook's outputs and stamp are
    cleared rather than re-executed, so it claims nothing until the regeneration runs it.

    One sweep keeps the contract, because a test naming it still says something true and
    deleting it early would retire the only thing recording the gap. `13_backtest`, `16_costs`
    and `15_risk_management` took their parameters on 2026-09-06, each in its own visit. Delete
    this test in the commit that moves `14_portfolio_management`.
    """
    sweeps = ("14_portfolio_management",)
    for stem in sweeps:
        source = (REPO / "case_studies" / "cme_futures" / f"{stem}.py").read_text()
        assert "SUPERSEDES_" not in source, (
            f"{stem} now declares a supersedes parameter - thread it into "
            "run_official_backtest_requests and delete this test"
        )
