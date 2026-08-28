"""A notebook that opens a study must let its tier be chosen.

``open_study(case_study, execution_tier="canonical")`` with no workspace is
``Study.regenerate``: the maintainer's handle for rewriting a case study's generated
artifacts in place. It refuses unless ``features``, ``labels`` and ``run_log`` are directory
symlinks, which they are in a maintainer worktree and are not anywhere else - a clean clone
has no such directories, and the CI fixture copies them (``tests/conftest.py``,
"We copy (not symlink) so we can patch presets for fast testing"). A notebook that hardcodes
that call therefore passes for its author and fails for everyone, including CI, and it fails
on the study handle rather than on anything the notebook is about.

Passing ``workspace`` at the call site does not decide this: ``workspace=WORKSPACE or None``
with an empty default is ``workspace=None``. What decides it is whether ``EXECUTION_TIER`` and
``WORKSPACE`` are both bound in the ``parameters`` cell, because
``pm_helpers.research_preview_parameters`` binds preview and an isolated workspace exactly
then - so the regenerate branch is never reached under the harness. That is what this checks,
by the same reading of the same cell the harness does.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.pm_helpers import PARAMETERS_CELL_MARKER, _percent_cell_bounds, _top_level_bindings

REPO_ROOT = Path(__file__).resolve().parent.parent

# Notebooks that call ``open_study`` and declare neither name, so the harness cannot route
# them and they reach ``Study.regenerate`` wherever the generated directories are ordinary
# directories. Owned by nasdaq100_microstructure; measured 2026-08-27. Removing the parameters
# from a notebook is not what this list is for - it exists so the guard can land while the two
# it names are fixed by their owner.
KNOWN_REGENERATION_ONLY: frozenset[str] = frozenset(
    {
        "case_studies/nasdaq100_microstructure/13_model_analysis.py",
        "case_studies/nasdaq100_microstructure/14_backtest.py",
    }
)


def _parameter_cell_bindings(source: str, path: Path) -> set[str]:
    bounds = next(
        (
            (first, last)
            for header, first, last in _percent_cell_bounds(source)
            if PARAMETERS_CELL_MARKER in header
        ),
        None,
    )
    if bounds is None:
        return set()
    first, last = bounds
    tree = ast.parse(source, filename=str(path))
    return {name for name, line in _top_level_bindings(tree) if first <= line <= last}


def _notebooks_opening_a_study() -> list[Path]:
    return [
        path
        for path in sorted((REPO_ROOT / "case_studies").rglob("[0-9][0-9]*.py"))
        if "open_study(" in path.read_text(encoding="utf-8")
    ]


def _offenders() -> list[str]:
    offenders = []
    for path in _notebooks_opening_a_study():
        source = path.read_text(encoding="utf-8")
        declared = _parameter_cell_bindings(source, path)
        if not {"EXECUTION_TIER", "WORKSPACE"} <= declared:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    return offenders


def test_every_notebook_opening_a_study_declares_its_tier() -> None:
    notebooks = _notebooks_opening_a_study()
    assert len(notebooks) > 50, f"only {len(notebooks)} notebooks scanned; the glob missed some"
    unlisted = sorted(set(_offenders()) - KNOWN_REGENERATION_ONLY)
    assert not unlisted, (
        "these notebooks call open_study but bind neither EXECUTION_TIER nor WORKSPACE in their "
        f"parameters cell, so nothing can stop them reaching Study.regenerate: {unlisted}. "
        'Declare EXECUTION_TIER = "canonical" and WORKSPACE: str = "" in the parameters cell and '
        "pass them to open_study."
    )


def test_no_stale_regeneration_only_entry() -> None:
    fixed = sorted(
        entry
        for entry in KNOWN_REGENERATION_ONLY - set(_offenders())
        if (REPO_ROOT / entry).exists()
    )
    assert not fixed, (
        f"these notebooks now declare both parameters: {fixed}. Remove them from "
        "KNOWN_REGENERATION_ONLY so the guard covers them."
    )
