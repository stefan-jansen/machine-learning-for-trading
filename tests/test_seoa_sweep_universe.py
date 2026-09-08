"""The seoa CI universe must realize the grid the case study declares, and the check that
says so must not sit behind a papermill parameter.

`cs-sp500_equity_option_analytics` was green while `14_backtest` swept nothing. Two
parameters did it between them. `MAX_SYMBOLS: 3` left no declared `k` realizable, so
`get_entry_schemes_for` returned `[]` and the sweep ran a `6 predictions x 0 schemes`
grid. `TOP_K: 1` took the `if TOP_K:` branch, which is where the feasibility check lived,
so nothing said so. The analysis downstream then summarised 52 backtests the fixture
already shipped, and the job's green tick was one of the seven conditions the case study
was declared complete against (ml4t/agent-workspace#1084).

Both halves are pinned, because either alone leaves the failure reachable: a universe that
realizes the grid, and a check that no parameter can route around.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from case_studies.utils.sweep_config import get_entry_schemes_for, get_top_k_values_for, load_sweep
from tests.pm_helpers import invocations_for

CASE_STUDY = "sp500_equity_option_analytics"
REPO_ROOT = Path(__file__).parent.parent
NOTEBOOK = REPO_ROOT / "case_studies" / CASE_STUDY / "14_backtest.py"
SWEEPING_NOTEBOOK = f"case_studies/{CASE_STUDY}/14_backtest"
# seoa declares no `backtest.long_short`, and `get_backtest_config` resolves it to False.
LONG_SHORT = False


def _max_symbols() -> int:
    """The universe every run of the sweeping notebook is given.

    Read through `invocations_for` rather than off a `parameters` block: the notebook
    runs once per declared label now, and this file's question - can the CI universe
    realize the grid - is the same for each of them only while they agree. Asserting
    that they do is what keeps this a single number.
    """
    overrides = yaml.safe_load((REPO_ROOT / "tests" / "overrides.yaml").read_text())
    caps = {
        run.parameters["MAX_SYMBOLS"]
        for run in invocations_for(overrides[SWEEPING_NOTEBOOK], key=SWEEPING_NOTEBOOK)
    }
    assert len(caps) == 1, f"{SWEEPING_NOTEBOOK}: runs disagree on MAX_SYMBOLS: {sorted(caps)}"
    return caps.pop()


def _declared_labels() -> list[str]:
    return sorted(load_sweep(CASE_STUDY).get("top_k_grid") or {})


def test_the_ci_universe_realizes_every_declared_concentration():
    """Not just one. A universe admitting `ew_top5` alone still runs a third of the grid.

    Fails at `MAX_SYMBOLS: 3` (no scheme at all) and at 6 (one of three).
    """
    max_symbols = _max_symbols()
    grid = load_sweep(CASE_STUDY)["top_k_grid"]
    shortfalls = {}
    for label in _declared_labels():
        declared = [int(k) for k in grid[label]]
        realized = get_top_k_values_for(CASE_STUDY, label, max_symbols)
        missing = sorted(set(declared) - set(int(k) for k in realized))
        if missing:
            shortfalls[label] = missing
    assert not shortfalls, (
        f"MAX_SYMBOLS={max_symbols} cannot realize every declared k: {shortfalls}. "
        f"The fixture would assert a grid it does not run."
    )


def test_every_declared_label_admits_an_entry_scheme():
    max_symbols = _max_symbols()
    starved = [
        label
        for label in _declared_labels()
        if not get_entry_schemes_for(CASE_STUDY, label, max_symbols, long_short=LONG_SHORT)
    ]
    assert not starved, f"MAX_SYMBOLS={max_symbols} leaves no entry scheme for {starved}"


def test_a_universe_below_the_smallest_declared_k_realizes_nothing():
    """The premise: the filter is what empties the grid, so the test above can fail."""
    label = _declared_labels()[0]
    smallest = min(int(k) for k in load_sweep(CASE_STUDY)["top_k_grid"][label])
    assert get_entry_schemes_for(CASE_STUDY, label, smallest, long_short=LONG_SHORT) == []
    assert get_entry_schemes_for(CASE_STUDY, label, smallest + 1, long_short=LONG_SHORT) != []


def test_the_feasibility_check_is_not_behind_a_parameter():
    """`get_top_k_values_for` raises on an unrealizable grid, so calling it IS the check.

    It used to sit in the `else` of `if TOP_K:`, which meant supplying TOP_K skipped the
    check as well as the default it was there to provide. Fails on the pre-fix notebook.
    """
    tree = ast.parse(NOTEBOOK.read_text())

    calls_under_parameter_branch = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if "TOP_K" not in names:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "get_top_k_values_for"
            ):
                calls_under_parameter_branch.append(node.lineno)

    assert not calls_under_parameter_branch, (
        f"{NOTEBOOK.name} calls get_top_k_values_for inside a branch on TOP_K "
        f"(line {calls_under_parameter_branch}). Supplying TOP_K then skips the "
        f"feasibility check, which is how ml4t/agent-workspace#1084 stayed silent."
    )

    called_somewhere = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "get_top_k_values_for"
        for n in ast.walk(tree)
    )
    assert called_somewhere, f"{NOTEBOOK.name} never calls get_top_k_values_for at all"
