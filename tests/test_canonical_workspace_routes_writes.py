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
