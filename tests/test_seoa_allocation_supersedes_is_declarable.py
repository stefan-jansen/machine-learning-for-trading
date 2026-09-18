"""A parameter the notebook re-assigns is not a parameter, and the freeze needs this one.

`15_portfolio_management` publishes an official allocation population, and
`OfficialPopulation.create` refuses a changed membership unless the caller names the
snapshot it supersedes. `SUPERSEDES_ALLOCATION_POPULATIONS` is how a run names it.

Until 2026-09-18 that map was assigned in an ordinary cell rather than the `parameters`
cell. Papermill injects its overrides immediately after the `parameters` cell, so the
committed literal ran afterwards and overwrote the injected value before
`population_supersedes` read it. A widened private run that passed exactly the
declaration it was asked for was refused as though it had passed nothing:

    ValueError: a changed population named
    'sp500_equity_option_analytics-allocation-fwd_ret_5d-fe0f6ceeb71c'
    must explicitly supersedes b718079501e6

The two sibling notebooks guarded by the same freeze take this as a real parameter -
`cme_futures` as `SUPERSEDES_ALLOCATION_POPULATION`, `crypto_perps_funding` as
`SUPERSEDES_ALLOCATION` - so this notebook was the only one whose freeze could not be
satisfied from outside its source.

Both halves are pinned. The name has to be a parameter, and no parameter may be
re-assigned at module level afterwards, because the second is what silently undid the
first.
"""

from __future__ import annotations

import ast
from pathlib import Path

CASE_STUDY = "sp500_equity_option_analytics"
REPO_ROOT = Path(__file__).parent.parent
NOTEBOOK = REPO_ROOT / "case_studies" / CASE_STUDY / "15_portfolio_management.py"
SUPERSEDES = "SUPERSEDES_ALLOCATION_POPULATIONS"


def _source() -> str:
    return NOTEBOOK.read_text()


def _parameters_cell_bounds(source: str) -> tuple[int, int]:
    """1-indexed [start, end) line numbers of the `parameters`-tagged cell's body."""
    lines = source.splitlines()
    start = None
    for i, line in enumerate(lines, start=1):
        if line.startswith("# %%") and "parameters" in line:
            start = i + 1
            break
    assert start is not None, f"{NOTEBOOK.name} has no cell tagged parameters"
    for i in range(start, len(lines) + 1):
        if lines[i - 1].startswith("# %%"):
            return start, i
    return start, len(lines) + 1


def _module_level_assignments(source: str) -> list[tuple[str, int]]:
    """(name, lineno) for every module-level binding, including annotated ones."""
    names: list[tuple[str, int]] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                names.append((target.id, node.lineno))
    return names


def test_the_allocation_supersedes_map_is_a_papermill_parameter() -> None:
    source = _source()
    start, end = _parameters_cell_bounds(source)
    in_cell = [name for name, lineno in _module_level_assignments(source) if start <= lineno < end]
    assert SUPERSEDES in in_cell, (
        f"{SUPERSEDES} is not assigned in the parameters cell of {NOTEBOOK.name}, so papermill "
        "cannot set it and the allocation freeze cannot be satisfied from outside the source"
    )


def test_no_parameter_is_reassigned_after_the_parameters_cell() -> None:
    """The general form of the same defect, which is what made it invisible.

    An injected override is written just after the parameters cell. Any later module-level
    binding of the same name discards it, and nothing reports that.
    """
    source = _source()
    start, end = _parameters_cell_bounds(source)
    assignments = _module_level_assignments(source)
    parameters = {name for name, lineno in assignments if start <= lineno < end}
    clobbered = sorted(
        {
            f"{name} (line {lineno})"
            for name, lineno in assignments
            if name in parameters and lineno >= end
        }
    )
    assert not clobbered, (
        f"{NOTEBOOK.name} re-assigns parameters after the parameters cell, which discards "
        f"whatever papermill injected: {', '.join(clobbered)}"
    )
