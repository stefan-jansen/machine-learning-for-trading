"""The crypto holdout must read its own inputs, and be reported only for its own lineage."""

from __future__ import annotations

import ast
from pathlib import Path

CASE_STUDY = Path(__file__).resolve().parents[1] / "case_studies" / "crypto_perps_funding"
NOTEBOOK = CASE_STUDY / "18_holdout_backtest.py"
ANALYSIS = CASE_STUDY / "19_strategy_analysis.py"


def _tree() -> ast.Module:
    return ast.parse(NOTEBOOK.read_text(encoding="utf-8"))


def test_holdout_backtest_settles_funding() -> None:
    """Funding is this case study's subject, and the runner defaults it to absent.

    ``run_backtest`` takes ``funding_rates=None``, so a call that omits the keyword produces a
    holdout with no funding cashflows at all - registering, reporting a Sharpe, and matching
    every expected row count while pricing a perpetual-futures strategy without the payment
    that defines it. Validation runs reach the same engine through ``run_backtests``, which
    resolves funding itself, so the two are only comparable when this call passes it.
    """
    calls = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_backtest"
    ]
    assert calls, f"no run_backtest call found in {NOTEBOOK.name}"
    omissions = [
        f"{NOTEBOOK.name}:{node.lineno} omits funding_rates"
        for node in calls
        if "funding_rates" not in {keyword.arg for keyword in node.keywords}
    ]
    assert not omissions, "\n".join(omissions)


def test_holdout_backtest_rebuilds_input_identity() -> None:
    """The carrier's spec describes the validation window, so its digests cannot be inherited.

    ``input_identity`` records the digests of the data a run actually read. The holdout reuses
    the carrier's registered specification for everything that must not change, and a spec
    carried across wholesale therefore names the validation price panel and the validation
    funding settlements. Consumers check that entry against the panel they hold, so an
    inherited digest is a record that is wrong rather than merely stale.
    """
    assigned = {
        target.slice.value
        for node in ast.walk(_tree())
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == "spec"
        and isinstance(target.slice, ast.Constant)
    }
    assert "input_identity" in assigned, (
        f"{NOTEBOOK.name} never assigns spec['input_identity'], so it inherits the carrier's"
    )


def test_strategy_analysis_reports_only_its_own_holdout() -> None:
    """``stage='holdout'`` names a window, not a lineage.

    Every row produced from a holdout prediction set carries that stage, including a run of a
    different allocator over the same window and one left behind by a superseded selection. A
    metrics query keyed on the stage alone therefore reports whatever holdout rows the registry
    happens to hold, beside an analysis of the configuration this notebook selected - which is
    the same failure as reporting two strategies as one, arrived at from the other end.

    The lineage link is ``resolve_solvent_carrier``'s ``holdout_backtest_hash``, which matches
    the holdout backtest to the carrier by strategy specification. The query has to be
    restricted to it, which means being parameterised rather than filtered on the stage alone.
    """
    source = ANALYSIS.read_text(encoding="utf-8")
    assert "holdout_backtest_hash" in source, (
        f"{ANALYSIS.name} never resolves the holdout backtest for the carrier it analyses"
    )
    holdout_queries = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute"
        and any(
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and "stage = 'holdout'" in arg.value
            for arg in node.args
        )
    ]
    assert holdout_queries, f"{ANALYSIS.name} runs no holdout metrics query to check"
    unrestricted = [
        f"{ANALYSIS.name}:{node.lineno} filters on stage alone"
        for node in holdout_queries
        if len(node.args) < 2
    ]
    assert not unrestricted, "\n".join(unrestricted)
