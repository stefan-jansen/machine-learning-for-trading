"""Every allocation notebook has to honour a width the launcher passes.

Papermill binds an override only into a name the parameters cell already holds. A notebook
that omits `TOP_N_PREDICTIONS` and calls `get_top_n_predictions` unconditionally therefore
sweeps the width its `setup.yaml` declares, prints one advisory `Passed unknown parameter`
line among a few hundred, and **exits 0** - the registry gains rows, every identity and
isolation check passes, and nothing says the width did not move. On 2026-09-18 that cost a
full canonical-tier `sp500_options` sweep launched at 999 against 46 advancing prediction
sets, caught only because a private pin recorded a number the sweep never reached
(`ml4t/agent-workspace#1200`).

Four of the nine had the defect: `cme_futures`, `crypto_perps_funding`, `sp500_options` and
`us_equities_panel`. The issue named the first three; the fourth is the same shape.

Both halves are asserted, because either alone passes on a notebook that still ignores the
override. Declaring the name without reading it is the failure mode papermill cannot see, and
reading a name the parameters cell does not bind is a `NameError` at run time rather than a
silent narrowing - so the pair is what makes the parameter mean anything.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.pm_helpers import (
    PARAMETERS_CELL_MARKER,
    _percent_cell_bounds,
    _top_level_bindings,
)

CASE_STUDIES = Path(__file__).resolve().parents[1] / "case_studies"
WIDTH_PARAMETER = "TOP_N_PREDICTIONS"
DECLARED_WIDTH_HELPER = "get_top_n_predictions"


def _allocation_notebooks() -> list[Path]:
    return sorted(CASE_STUDIES.glob("*/*_portfolio_management.py"))


def _parameters_cell_bounds(source: str) -> tuple[int, int] | None:
    return next(
        (
            (first, last)
            for header, first, last in _percent_cell_bounds(source)
            if PARAMETERS_CELL_MARKER in header
        ),
        None,
    )


def _depends_on_the_override(tree: ast.Module, call: ast.Call) -> bool:
    """Whether `TOP_N_PREDICTIONS` reaches the statement that reads the declared width.

    Either it is in the same expression, so the declared width is one branch of it, or an
    enclosing `if` tests it, so the declared width is the fallback. Both are honouring the
    override; a call that neither guards nor mentions it overwrites whatever was passed.
    """
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]
    current: ast.AST | None = call
    while current is not None:
        if isinstance(current, ast.If) and any(
            isinstance(name, ast.Name) and name.id == WIDTH_PARAMETER
            for name in ast.walk(current.test)
        ):
            return True
        if isinstance(current, (ast.Assign, ast.AnnAssign)) and any(
            isinstance(name, ast.Name) and name.id == WIDTH_PARAMETER
            for name in ast.walk(current.value)  # type: ignore[arg-type]
        ):
            return True
        current = getattr(current, "parent", None)
    return False


def test_every_case_study_has_an_allocation_notebook_to_check() -> None:
    """The glob is the test's subject, so an empty one would pass everything below."""
    found = {path.parent.name for path in _allocation_notebooks()}
    expected = {path.name for path in CASE_STUDIES.iterdir() if (path / "config").is_dir()}
    assert found == expected, f"allocation notebooks missing for {sorted(expected - found)}"


@pytest.mark.parametrize(
    "notebook", _allocation_notebooks(), ids=lambda p: f"{p.parent.name}/{p.stem}"
)
def test_the_parameters_cell_binds_the_width(notebook: Path) -> None:
    source = notebook.read_text(encoding="utf-8")
    bounds = _parameters_cell_bounds(source)
    assert bounds is not None, f"{notebook} has no parameters cell"
    first, last = bounds
    tree = ast.parse(source, filename=str(notebook))
    declared = {name for name, line in _top_level_bindings(tree) if first <= line <= last}
    assert WIDTH_PARAMETER in declared, (
        f"{notebook.parent.name}/{notebook.name} does not bind {WIDTH_PARAMETER} in its "
        "parameters cell, so papermill drops an override and the sweep runs at the width "
        "setup.yaml declares while exiting 0"
    )


@pytest.mark.parametrize(
    "notebook", _allocation_notebooks(), ids=lambda p: f"{p.parent.name}/{p.stem}"
)
def test_the_declared_width_is_only_a_fallback(notebook: Path) -> None:
    source = notebook.read_text(encoding="utf-8")
    bounds = _parameters_cell_bounds(source)
    assert bounds is not None
    first, last = bounds
    tree = ast.parse(source, filename=str(notebook))
    reads_after_the_cell = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == WIDTH_PARAMETER
        and isinstance(node.ctx, ast.Load)
        and not (first <= node.lineno <= last)
    ]
    assert reads_after_the_cell, (
        f"{notebook.parent.name}/{notebook.name} binds {WIDTH_PARAMETER} and never reads it, "
        "so an override is accepted and discarded"
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == DECLARED_WIDTH_HELPER
    ]
    assert calls, f"{notebook} never reads the declared width, so nothing sets a default"
    for call in calls:
        assert _depends_on_the_override(tree, call), (
            f"{notebook.parent.name}/{notebook.name} reads the declared width at line "
            f"{call.lineno} without {WIDTH_PARAMETER} reaching that statement, so an override "
            "is accepted and the sweep runs at the width setup.yaml declares"
        )
