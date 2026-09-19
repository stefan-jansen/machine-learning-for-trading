"""Every allocation notebook has to honour a width the launcher passes.

Papermill injects every parameter it is given into a cell of its own, so the name is always
bound at run time. What decides whether the override does anything is whether the notebook
reads it. Four allocation notebooks called `get_top_n_predictions` unconditionally and never
read `TOP_N_PREDICTIONS`, so a run that passed one swept the width `setup.yaml` declares,
printed one advisory `Passed unknown parameter` line among a few hundred, and **exited 0** -
the registry gained rows, every identity and isolation check passed, and nothing said the
width had not moved. On 2026-09-18 that cost a full canonical-tier `sp500_options` sweep
launched at 999 against 46 advancing prediction sets, caught only because a private pin
recorded a number the sweep never reached (`ml4t/agent-workspace#1200`).

Three assertions, because each passes on a notebook the other two catch:

- the parameters cell binds the name. Papermill's injected cell comes after that one, so a
  notebook that does not bind it has the name only from the injection and raises `NameError`
  on any run that does not pass it;
- papermill's inspector can see it there. This is not what makes the override land - measured
  2026-09-19, a cell holding `X: int | None = None` still receives `X = 111` and prints it -
  but the inspector drives the `Passed unknown parameter` warning, so a name it cannot parse
  makes that line fire on a parameter the notebook does honour. The warning is the only
  runtime signal this failure has, and a false one is worse than none;
- `TOP_N_PREDICTIONS` reaches the statement that reads the declared width, or the override is
  bound and then overwritten. That is the defect itself, and the shape `fx_pairs` had, where
  the name was bound, read only by the narrowing guard, and the width came from
  `TOP_N_CONFIGS`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.pm_helpers import (
    PARAMETERS_CELL_MARKER,
    _papermill_visible,
    _percent_cell_bounds,
    parameters_cell_names,
)

CASE_STUDIES = Path(__file__).resolve().parents[1] / "case_studies"
WIDTH_PARAMETER = "TOP_N_PREDICTIONS"
DECLARED_WIDTH_HELPER = "get_top_n_predictions"


def _allocation_notebooks() -> list[Path]:
    return sorted(CASE_STUDIES.glob("*/*_portfolio_management.py"))


def _parameter_cell_spans(source: str) -> list[tuple[int, int]]:
    return [
        (lo, hi)
        for header, lo, hi in _percent_cell_bounds(source)
        if PARAMETERS_CELL_MARKER in header
    ]


def _depends_on_the_override(tree: ast.Module, call: ast.Call) -> bool:
    """Whether `TOP_N_PREDICTIONS` reaches the statement that reads the declared width.

    Either an enclosing `if` tests it, so the declared width is the fallback, or it is in the
    assigned expression itself, so the declared width is one branch of it. A call that neither
    guards on it nor mentions it overwrites whatever the launcher passed.
    """
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]

    def mentions(node: ast.AST | None) -> bool:
        return node is not None and any(
            isinstance(name, ast.Name) and name.id == WIDTH_PARAMETER for name in ast.walk(node)
        )

    current: ast.AST | None = call
    while current is not None:
        if isinstance(current, ast.If) and mentions(current.test):
            return True
        if isinstance(current, (ast.Assign, ast.AnnAssign)) and mentions(current.value):
            return True
        current = getattr(current, "parent", None)
    return False


def test_every_case_study_has_an_allocation_notebook_to_check() -> None:
    """The glob is the subject of everything below, so an empty one would pass it all."""
    found = {path.parent.name for path in _allocation_notebooks()}
    expected = {path.name for path in CASE_STUDIES.iterdir() if (path / "config").is_dir()}
    assert found == expected, f"allocation notebooks missing for {sorted(expected - found)}"


@pytest.mark.parametrize(
    "notebook", _allocation_notebooks(), ids=lambda p: f"{p.parent.name}/{p.stem}"
)
def test_the_parameters_cell_binds_the_width(notebook: Path) -> None:
    assert WIDTH_PARAMETER in parameters_cell_names(notebook), (
        f"{notebook.parent.name}/{notebook.name} does not bind {WIDTH_PARAMETER} in its "
        "parameters cell, so papermill drops an override and the sweep runs at the width "
        "setup.yaml declares while exiting 0"
    )


@pytest.mark.parametrize(
    "notebook", _allocation_notebooks(), ids=lambda p: f"{p.parent.name}/{p.stem}"
)
def test_papermill_can_see_the_width(notebook: Path) -> None:
    """So `Passed unknown parameter` stays a true signal, not one fired on a honoured name."""
    assert notebook.with_suffix(".ipynb").exists(), f"{notebook} has no paired notebook"
    visible = _papermill_visible(notebook)
    assert visible is not None, (
        f"papermill could not inspect {notebook.with_suffix('.ipynb').name} at all, which is a "
        "malformed or unreadable notebook rather than a missing parameter"
    )
    assert WIDTH_PARAMETER in visible, (
        f"papermill's inspector cannot see {WIDTH_PARAMETER} in {notebook.parent.name}/"
        f"{notebook.name}; it splits the parameters cell on '=' line by line, so a PEP 604 "
        "annotation or a trailing comment containing '=' hides a name that is bound in plain "
        "Python, and every run that passes the width is told it was unknown"
    )


@pytest.mark.parametrize(
    "notebook", _allocation_notebooks(), ids=lambda p: f"{p.parent.name}/{p.stem}"
)
def test_the_declared_width_is_only_a_fallback(notebook: Path) -> None:
    source = notebook.read_text(encoding="utf-8")
    spans = _parameter_cell_spans(source)
    assert spans, f"{notebook} has no parameters cell"
    tree = ast.parse(source, filename=str(notebook))
    reads_outside_the_cell = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == WIDTH_PARAMETER
        and isinstance(node.ctx, ast.Load)
        and not any(lo <= node.lineno <= hi for lo, hi in spans)
    ]
    assert reads_outside_the_cell, (
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
