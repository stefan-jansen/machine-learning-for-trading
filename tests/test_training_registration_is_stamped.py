"""Every notebook that registers a training run must go through the stamped path.

``case_studies/utils/registry/specs.py`` sets ``IDENTITY_VERSION = 3`` and
``build_training_spec`` - the pre-migration builder - never stamps it, on any family
branch. An unstamped spec projects at the legacy version, ``catalog.py`` maps it to
``legacy`` and never ``current``, and a row that is not ``current`` is never ``complete``,
so ``official_prediction_catalog`` resolves it away.

Nothing raises when that happens. The notebook prints a normal IC table, registers its
rows, and the official population comes back empty; ``load_prediction_index`` does not
filter on ``complete``, so a backtest stage downstream consumes the rows happily.
``nasdaq100_microstructure``'s ``14_backtest`` preview ran 936 backtests to completion off
prediction sets whose official catalog resolved to zero rows - fifteen hours of compute on
a population that did not officially exist. That is what this test exists to stop
recurring, because the failure is silent at exactly the point where it looks like success.

``run_dl_cv`` is the last function still on the unstamped builder. Its migrated sibling in
the same file, ``ResolvedSpec.create``, stamps unconditionally, and ``run_model_population``
reaches that one - so which of the two a notebook calls decides whether its rows can be
traded, and the two calls look equally ordinary at the call site.

Measured on 2026-09-05: one caller remains, in a parked case study. The point of pinning it
now is that the count can only go up by someone adding a call that looks like every other
one, and no test, gate or notebook output would say so.

Tracked as ml4t/agent-workspace#919.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_STUDIES = REPO_ROOT / "case_studies"

# `us_equities_panel` is parked: it holds its worktree, branch and registry as they are and
# gets no run, so its notebook cannot register anything until it is unparked. Named here
# rather than skipped silently, so that unparking the case study fails this test and makes
# the migration a precondition of the first canonical deep-learning run rather than a
# discovery after it.
#
# The entry is a migration to do, never a notebook to delete: `12_dl_weekly` is cited by
# Table 13.5 in the book and competes for selection, so removing it to satisfy this test
# would take a published result with it.
KNOWN_UNMIGRATED = {"us_equities_panel/12_dl_weekly.py"}


def _notebook_sources() -> list[Path]:
    """Every case-study notebook, excluding the shared helper modules under `utils/`."""
    return sorted(
        path
        for path in CASE_STUDIES.glob("*/*.py")
        if path.parent.name != "utils" and not path.name.startswith("_")
    )


def _calls_unstamped_runner(source: str) -> bool:
    """True when the module calls ``run_dl_cv``.

    Matched on the AST rather than on the text, so the name appearing in a markdown cell,
    a docstring or a comment - which is how several of these notebooks discuss the
    migration - does not count as calling it.
    """
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "run_dl_cv":
            return True
    return False


@pytest.mark.parametrize("notebook", _notebook_sources(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_notebook_registers_through_the_unstamped_builder(notebook: Path) -> None:
    relative = f"{notebook.parent.name}/{notebook.name}"
    calls_it = _calls_unstamped_runner(notebook.read_text())

    if relative in KNOWN_UNMIGRATED:
        assert calls_it, (
            f"{relative} is listed as unmigrated and no longer calls run_dl_cv. Remove it "
            "from KNOWN_UNMIGRATED - the list is what makes the remaining exposure legible, "
            "and an entry that no longer describes anything hides that the work is done."
        )
        pytest.skip(f"{relative}: known unmigrated, and its case study is parked")

    assert not calls_it, (
        f"{relative} calls run_dl_cv, which builds its specs through build_training_spec "
        "and stamps no identity_version. Its rows would register at the legacy version, "
        "resolve as incomplete, and be filtered out of every official population - with "
        "nothing raising. Call run_model_population instead, which reaches ResolvedSpec."
    )


def test_the_unstamped_builder_still_stamps_nothing() -> None:
    """The premise of the test above, checked rather than assumed.

    If ``build_training_spec`` ever starts stamping ``identity_version``, the parametrized
    test is guarding against a defect that no longer exists and should be deleted rather
    than left to look like it is protecting something.
    """
    from case_studies.utils.registry import build_training_spec

    # A real preset, because the builder resolves one by name. Which family it is does not
    # matter here - the issue confirmed that no branch of this function stamps the version.
    spec = build_training_spec("deep_learning", "nlinear", "label", n_folds=2)
    assert "identity_version" not in spec, (
        "build_training_spec now stamps identity_version, so run_dl_cv is no longer the "
        "unstamped path and this whole file has nothing left to protect. Delete it, and "
        "close ml4t/agent-workspace#919 citing the commit that changed the builder."
    )
