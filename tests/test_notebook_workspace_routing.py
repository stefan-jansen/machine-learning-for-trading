"""Every notebook that offers a WORKSPACE parameter must honour it at both tiers.

`open_study` answers a canonical run with no workspace through `Study.regenerate`, which
writes through the generated-artifact symlinks to the released case directory. A notebook that
reads `WORKSPACE` only on its preview branch therefore accepts the parameter, reports the
workspace it was asked for, and registers its rows in the published store - with no exception
and no warning. On 2026-09-15 that left 60 allocation rows and 201 files in `sp500_options`
(ml4t/agent-workspace#1100), and the run's exit status said it had refused at the freeze.

The shape that does it is an `open_study` call taking a workspace under a hardcoded
`execution_tier="preview"`. The whole fleet passes the `EXECUTION_TIER` parameter instead, so
the canonical branch routes too, and this asks every notebook for that.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CASE_STUDIES = REPO_ROOT / "case_studies"


def _workspace_notebooks() -> list[Path]:
    return sorted(
        path
        for path in CASE_STUDIES.glob("*/[0-9]*.py")
        if 'WORKSPACE: str = ""' in path.read_text()
    )


def unrouted_open_study_calls(source: str) -> list[str]:
    """Report every `open_study` call that takes a workspace under a hardcoded tier.

    The tier has to reach the call as the notebook's own `EXECUTION_TIER` parameter. A string
    literal there pins the call to one tier, and the only tier anybody pins it to is the
    preview - which is the defect: the canonical branch then never passes the workspace on.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "open_study":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        if "workspace" not in keywords:
            continue
        tier = keywords.get("execution_tier")
        if isinstance(tier, ast.Constant):
            found.append(f"line {node.lineno}: execution_tier={tier.value!r}")
    return found


@pytest.mark.parametrize("path", _workspace_notebooks(), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_a_notebook_offering_a_workspace_routes_it_at_both_tiers(path: Path) -> None:
    unrouted = unrouted_open_study_calls(path.read_text())
    assert not unrouted, (
        f"{path.relative_to(REPO_ROOT)} declares a WORKSPACE parameter and opens its study with "
        f"a hardcoded tier: {'; '.join(unrouted)}. A canonical run given a workspace then falls "
        "through to Study.regenerate and registers in the released case directory while the "
        "caller reads the workspace. Pass execution_tier=EXECUTION_TIER and "
        "workspace=WORKSPACE or None."
    )


def test_the_checker_reports_the_shape_it_exists_to_catch() -> None:
    """Negative control: the pre-#1100 shape, which every fixed notebook used to carry."""
    unrouted = unrouted_open_study_calls(
        'WORKSPACE: str = ""\n'
        'if EXECUTION_TIER == "preview":\n'
        "    study = open_study(\n"
        "        CASE_STUDY_ID,\n"
        '        execution_tier="preview",\n'
        "        workspace=WORKSPACE,\n"
        "    )\n"
    )
    assert unrouted == ["line 3: execution_tier='preview'"]


def test_the_checker_passes_the_shape_the_fleet_uses() -> None:
    assert not unrouted_open_study_calls(
        'WORKSPACE: str = ""\n'
        "study = open_study(\n"
        "    CASE_STUDY_ID,\n"
        "    execution_tier=EXECUTION_TIER,\n"
        "    workspace=WORKSPACE or None,\n"
        ")\n"
    )


def test_every_case_study_with_a_workspace_parameter_is_covered() -> None:
    """A checker that selects nothing passes vacuously, so say what it selected."""
    notebooks = _workspace_notebooks()
    case_studies = {path.parent.name for path in notebooks}
    assert len(notebooks) > 100, f"only {len(notebooks)} notebooks declare a WORKSPACE parameter"
    assert len(case_studies) == 9, sorted(case_studies)
