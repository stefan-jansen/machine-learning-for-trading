"""The crypto cost sweep must re-price the strategy that was selected, whole."""

from __future__ import annotations

import ast
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "case_studies" / "crypto_perps_funding" / "16_costs.py"
)

# The three components that make a selected configuration what it is. `costs` is the swept
# dimension and is deliberately not in this set; every other component has to survive the
# sweep unchanged or the curve describes a different strategy.
STRATEGY_COMPONENTS = {"signal", "allocation", "risk"}


def test_cost_sweep_forwards_every_selected_strategy_component() -> None:
    """Risk management runs before this stage, so a selection can be a risk overlay.

    The stage that fails here is the one that drops it: `run_backtests` defaults `risk` to
    `None`, so a sweep that simply omits the keyword prices the overlay's signal and
    allocation without its control, and reports the resulting difference as a cost effect.
    Nothing else catches that - the run succeeds, the row counts match, and the curve is
    wrong.

    Read with `ast` rather than a regex so a call spanning several lines, or one whose
    arguments contain their own parentheses, is still seen. `checked` is asserted as well as
    the omissions, because a checker that parsed nothing would otherwise be
    indistinguishable from a checker that found nothing wrong.
    """
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    checked = 0
    omissions: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "run_backtests"):
            continue
        checked += 1
        passed = {keyword.arg for keyword in node.keywords}
        missing = STRATEGY_COMPONENTS - passed
        if missing:
            omissions.append(f"{NOTEBOOK.name}:{node.lineno} omits {', '.join(sorted(missing))}")
    assert checked, f"no run_backtests call found in {NOTEBOOK.name}"
    assert not omissions, "\n".join(omissions)
