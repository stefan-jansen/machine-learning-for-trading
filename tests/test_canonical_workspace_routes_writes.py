"""A canonical run given a workspace must write into it, in every case study.

Five case studies define their own ``open_study`` in ``research_workflow.py`` rather than using
the shared one, and they did not agree on what a workspace means. ``sp500_options`` and
``sp500_equity_option_analytics`` read it on the preview branch only: a canonical run that passed
one read from the workspace and returned ``Study.regenerate``, which writes to the released case
directory. Neither raised and neither warned. On 2026-09-15 a rehearsal against a private registry
left 60 allocation rows and 60 artifact directories in the published ``sp500_options`` store that
way, and the run's own exit status said it had refused at the freeze.

The shape is what makes it worth a test rather than a fix: the wrong behaviour is silent, it is
per case study, and nothing downstream distinguishes a row a workspace run wrote from one the
released path wrote. This asserts the contract across every case study that has an ``open_study``,
so the next one added is covered without anyone remembering to add a case here.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

# Every case study whose research_workflow defines its own open_study. The shared
# case_studies.research.open_study takes the case study positionally and is covered by
# tests/test_research_workspace.py; these five shadow it.
LOCAL_OPEN_STUDY = [
    "cme_futures",
    "crypto_perps_funding",
    "sp500_equity_option_analytics",
    "sp500_options",
    "us_firm_characteristics",
]


def _seed_workspace(workspace: Path, case_study: str) -> Path:
    """Write the manifest `Study.open` looks for, so it adopts rather than copying the release.

    Without it `Study.open` runs `create_experiment`, which physically copies the canonical
    features and labels - minutes and gigabytes for a contract this can establish without them.
    """
    target = workspace / case_study
    (target / "config").mkdir(parents=True)
    (target / ".study.json").write_text(
        json.dumps({"schema_version": 1, "case_study": case_study}) + "\n"
    )
    return target


@pytest.mark.parametrize("case_study", LOCAL_OPEN_STUDY)
def test_canonical_with_a_workspace_writes_into_it(case_study, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ML4T_OUTPUT_DIR", raising=False)
    workspace = tmp_path / "private"
    target = _seed_workspace(workspace, case_study)

    module = importlib.import_module(f"case_studies.{case_study}.research_workflow")
    study = module.open_study(execution_tier="canonical", workspace=workspace)

    assert study.storage_root("canonical") == target, (
        f"{case_study}.open_study ignored the workspace on a canonical run and would have "
        f"written to {study.storage_root('canonical')}"
    )


def _study_bindings(source: Path) -> list[tuple[int, ast.Call]]:
    """Every ``study = open_study(...)`` in a notebook, with the line it is on.

    The name is the contract. Across the nine case studies there are 142 of these and four
    ``_workspace_study = open_study(...)``, and nothing else binds a study. ``study`` is the one
    the notebook computes and registers with; the underscore-prefixed ones are secondary handles
    that pass a workspace unconditionally and are followed by an explicit ``storage_root`` guard
    refusing a run that reads one registry and writes another.
    """
    tree = ast.parse(source.read_text(), filename=str(source))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "study" for t in node.targets):
            continue
        for inner in ast.walk(node.value):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "open_study"
            ):
                calls.append((inner.lineno, inner))
    return calls


NOTEBOOKS_BINDING_A_STUDY = sorted(
    path
    for path in (Path(__file__).resolve().parents[1] / "case_studies").glob("*/[0-9]*.py")
    if _study_bindings(path)
)


@pytest.mark.parametrize(
    "notebook", NOTEBOOKS_BINDING_A_STUDY, ids=lambda p: f"{p.parent.name}/{p.stem}"
)
def test_every_notebook_decides_where_its_study_writes(notebook: Path) -> None:
    """A correct ``open_study`` does not help a call site that never passes the workspace.

    The test above covers the function; this covers the caller, and the two fail separately.
    ``us_equities_panel`` opens its study through the shared ``open_study``, which routes a
    canonical run correctly when it is given a workspace, and five of its notebooks passed one on
    the preview branch only. On 2026-09-19 a rehearsal lane registered five allocation rows, seven
    populations and 35 artifact directories in the published store that way, and replaced the
    official allocation population with sixty members of which fifty-five had never been computed.

    It reads the source rather than running the notebook because the behaviour is reachable only
    by executing one, which is hours per case study, and because the defect is the call site: an
    absent keyword, in the branch a preview-tier test never enters. ``workspace=None`` is a
    decision and passes; saying nothing is not a decision.
    """
    missing = [
        lineno
        for lineno, call in _study_bindings(notebook)
        if not any(keyword.arg == "workspace" for keyword in call.keywords)
    ]
    assert not missing, (
        f"{notebook.parent.name}/{notebook.name} binds its study without a workspace at "
        f"line(s) {', '.join(str(n) for n in missing)}. A canonical-tier run that sets one "
        "would read it and write to the published store instead."
    )
