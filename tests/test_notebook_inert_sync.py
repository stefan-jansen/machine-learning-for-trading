"""An inert code edit must be able to land, and must not be able to carry an output with it.

`sync-prose`, `sync-alt`, `sync-paths` and `sync-imports` each recognise their edit
mechanically. Some inert edits are not recognisable that way - dropping a conjunct from a
guard the run's own parameters already decided, threading an argument that enters no stored
identity - and where the case study has frozen its populations, a re-run is refused rather
than merely expensive. `sync-inert` is the escape hatch, and these tests pin what keeps it
honest: the outputs must be exactly the recorded run's, the source must actually have moved,
the claim must be written down, and `executed_at` must survive.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("tmp_repo")

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "notebook_provenance", REPO / ".github" / "scripts" / "notebook_provenance.py"
)
provenance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(provenance)

REASON = "the removed conjunct was true under this run's own parameters"

PY_BEFORE = """# %% [markdown]
# # A notebook

# %%
WORKSPACE = ""

# %%
CANONICAL = True and not WORKSPACE
print(CANONICAL)
"""


def _write_pair(root: Path, py_source: str) -> tuple[Path, Path]:
    py = root / "demo.py"
    py.write_text(py_source, encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupytext",
            "--to",
            "ipynb",
            "--output",
            str(root / "demo.ipynb"),
            str(py),
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    nb_path = root / "demo.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    n = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            n += 1
            cell["execution_count"] = n
            cell["outputs"] = [
                {
                    "output_type": "execute_result",
                    "execution_count": n,
                    "data": {"text/plain": ["True"]},
                    "metadata": {},
                }
            ]
    nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    return py, nb_path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    return tmp_path


def _stamp(nb_path: Path, py: Path) -> None:
    subprocess.run(["git", "add", py.name], cwd=py.parent, check=True)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    nb["metadata"][provenance.STAMP_KEY] = {
        "source_py_blob": provenance.git_blob(py),
        "outputs_digest": provenance.outputs_digest(nb),
        "library_digest": "0" * 12,
        "executed_at": "2026-08-01T00:00:00+00:00",
        "executor": "ml4t-gpu",
        "production": True,
        "parameters": {},
    }
    nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")


def test_an_inert_code_edit_lands_with_its_outputs_and_its_original_run_time(repo):
    py, nb_path = _write_pair(repo, PY_BEFORE)
    _stamp(nb_path, py)
    before = json.loads(nb_path.read_text())
    n_outputs = sum(len(c.get("outputs", [])) for c in before["cells"])

    py.write_text(PY_BEFORE.replace("True and not WORKSPACE", "True"))
    provenance.sync_inert(nb_path, REASON)

    after = json.loads(nb_path.read_text())
    assert sum(len(c.get("outputs", [])) for c in after["cells"]) == n_outputs
    stamp = after["metadata"][provenance.STAMP_KEY]
    assert stamp["source_py_blob"] == provenance.git_blob(py)
    # The run happened then, by that executor, and produced those outputs. Only the
    # source moved, so those three are the part of the record that must not be rewritten.
    assert stamp["executed_at"] == "2026-08-01T00:00:00+00:00"
    assert stamp["executor"] == "ml4t-gpu"
    assert stamp["outputs_digest"] == provenance.outputs_digest(after)
    assert stamp["inert_edit"]["reason"] == REASON
    assert stamp["inert_edit"]["previous_source_py_blob"] != stamp["source_py_blob"]
    # And the edit really is in the notebook, not only in the .py.
    code = "".join("".join(c["source"]) for c in after["cells"] if c["cell_type"] == "code")
    assert "not WORKSPACE" not in code


def test_an_edited_output_cannot_ride_along(repo):
    py, nb_path = _write_pair(repo, PY_BEFORE)
    _stamp(nb_path, py)
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and cell["outputs"]:
            cell["outputs"][0]["data"]["text/plain"] = ["False"]
            break
    nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    py.write_text(PY_BEFORE.replace("True and not WORKSPACE", "True"))
    with pytest.raises(SystemExit, match="not the ones the stamp records"):
        provenance.sync_inert(nb_path, REASON)


def test_an_unmoved_source_is_refused(repo):
    py, nb_path = _write_pair(repo, PY_BEFORE)
    _stamp(nb_path, py)
    with pytest.raises(SystemExit, match="nothing to fold in"):
        provenance.sync_inert(nb_path, REASON)


def test_a_reason_that_says_nothing_is_refused(repo):
    py, nb_path = _write_pair(repo, PY_BEFORE)
    _stamp(nb_path, py)
    py.write_text(PY_BEFORE.replace("True and not WORKSPACE", "True"))
    with pytest.raises(SystemExit, match="must say why"):
        provenance.sync_inert(nb_path, "inert")


def test_an_unstamped_notebook_is_refused(repo):
    py, nb_path = _write_pair(repo, PY_BEFORE)
    subprocess.run(["git", "add", py.name], cwd=py.parent, check=True)
    with pytest.raises(SystemExit, match="carries no provenance stamp"):
        provenance.sync_inert(nb_path, REASON)
