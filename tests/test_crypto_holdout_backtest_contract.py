"""The crypto holdout backtest must read, and be identified by, its own inputs."""

from __future__ import annotations

import ast
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "case_studies"
    / "crypto_perps_funding"
    / "18_holdout_backtest.py"
)


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
