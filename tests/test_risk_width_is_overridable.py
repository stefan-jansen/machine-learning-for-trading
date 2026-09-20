"""Every risk-overlay notebook has to honour a width the launcher passes.

The sibling of `test_allocation_width_is_overridable.py`, one stage later, and its absence is
why the divergence it checks survived the stage-06 rebuild. Every case study declares
`backtest.sweep.top_n_predictions.risk_overlay: 1`, so one parent per label is the shipped
design and the notebooks agreed on the answer. They did not agree on where it came from. Four
read the declaration through `TOP_N_COMBOS`; `cme_futures`, `crypto_perps_funding` and
`fx_pairs` spelled the same 1 as a literal - `rank_by_validation_sharpe(...)[0]`,
`CandidateSet.one(...).best_validation_sharpe()`, `min(eligible, key=...)` - and
`us_equities_panel` read the declaration but bound no parameter. A launch at a wider width
exited 0 on all four, registered the rows the declared width produces, and printed one advisory
`Passed unknown parameter: TOP_N_COMBOS` line. Measured 2026-09-20, after a weekend sweep whose
risk lanes ran at width 1 while reporting 999.

The assertions are the allocation test's three, for the same reasons, plus the one that names
this stage's own failure: a notebook that reads the declared width without the override
reaching that statement.

`sp500_options` is exempt, and the exemption is decidable rather than a name in a list: it
declares no position-level and no portfolio-level risk controls, so its risk notebook registers
nothing and there is no grid for a width to size. `test_the_exemption_is_earned` fails if that
stops being true, which is the negative control that keeps the exemption honest.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from case_studies.utils.sweep_config import (
    get_portfolio_risk_controls,
    get_position_risk_controls,
    get_top_n_predictions,
)
from tests.pm_helpers import (
    PARAMETERS_CELL_MARKER,
    _papermill_visible,
    _percent_cell_bounds,
    parameters_cell_names,
)

CASE_STUDIES = Path(__file__).resolve().parents[1] / "case_studies"
WIDTH_PARAMETER = "TOP_N_COMBOS"
DECLARED_WIDTH_HELPER = "get_top_n_predictions"


def _declares_no_controls(case_study: str) -> bool:
    return not get_position_risk_controls(case_study) and not get_portfolio_risk_controls(
        case_study
    )


def _risk_notebooks() -> list[Path]:
    return sorted(
        path
        for path in CASE_STUDIES.glob("*/*_risk_management.py")
        if not _declares_no_controls(path.parent.name)
    )


def _parameter_cell_spans(source: str) -> list[tuple[int, int]]:
    return [
        (lo, hi)
        for header, lo, hi in _percent_cell_bounds(source)
        if PARAMETERS_CELL_MARKER in header
    ]


def _depends_on_the_override(tree: ast.Module, call: ast.Call) -> bool:
    """Whether `TOP_N_COMBOS` reaches the statement that reads the declared width."""
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
        if isinstance(current, ast.IfExp) and mentions(current.test):
            return True
        current = getattr(current, "parent", None)
    return False


def test_every_case_study_has_a_risk_notebook_to_check() -> None:
    """The glob is the subject of everything below, so an empty one would pass it all."""
    found = {path.parent.name for path in CASE_STUDIES.glob("*/*_risk_management.py")}
    expected = {path.name for path in CASE_STUDIES.iterdir() if (path / "config").is_dir()}
    assert found == expected, f"risk notebooks missing for {sorted(expected - found)}"


def test_the_exemption_is_earned() -> None:
    """`sp500_options` is skipped above only while it declares no risk controls at all."""
    exempt = {
        path.parent.name
        for path in CASE_STUDIES.glob("*/*_risk_management.py")
        if _declares_no_controls(path.parent.name)
    }
    assert exempt == {"sp500_options"}, (
        f"the set of case studies declaring no risk controls moved to {sorted(exempt)}; a new "
        "member needs its own reasoning and a departing one needs the width parameter"
    )
    assert get_top_n_predictions("sp500_options", "risk_overlay") >= 1, (
        "sp500_options still declares a risk_overlay width, so the declaration and the empty "
        "control set disagree about whether this stage sweeps anything"
    )


def test_every_case_study_declares_one_parent_per_label() -> None:
    """The shipped width, which is what makes a literal `1` in a notebook invisible."""
    declared = {
        path.parent.name: get_top_n_predictions(path.parent.name, "risk_overlay")
        for path in CASE_STUDIES.glob("*/*_risk_management.py")
    }
    assert set(declared.values()) == {1}, (
        f"the declared risk-overlay widths are no longer uniform: {declared}. That is allowed, "
        "but this test's premise - that a hardcoded 1 agrees with every declaration and so "
        "cannot be caught by comparing results - no longer holds."
    )


@pytest.mark.parametrize("notebook", _risk_notebooks(), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_the_parameters_cell_binds_the_width(notebook: Path) -> None:
    assert WIDTH_PARAMETER in parameters_cell_names(notebook), (
        f"{notebook.parent.name}/{notebook.name} does not bind {WIDTH_PARAMETER} in its "
        "parameters cell, so papermill drops an override and the overlay grid sits on the "
        "number setup.yaml declares while the run exits 0"
    )


@pytest.mark.parametrize("notebook", _risk_notebooks(), ids=lambda p: f"{p.parent.name}/{p.stem}")
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


@pytest.mark.parametrize("notebook", _risk_notebooks(), ids=lambda p: f"{p.parent.name}/{p.stem}")
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
            "is accepted and the overlay grid sits on the declared number"
        )
