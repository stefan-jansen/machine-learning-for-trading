"""A prose-only edit must not cost a re-run, and must not be able to hide a code change.

`notebook_provenance.py` stamps a notebook with the `.py` blob that produced it, so any
edit to the `.py` reads as stale. Rewording a markdown cell in place was already forgiven;
changing the SHAPE of the prose - deleting a cell, merging two, adding one - was not, and
that is exactly what bringing a notebook under the three-tagged-results-cells rule needs.
The gate was stricter than the physics: a markdown cell is a comment block in the `.py`,
each code cell carries its own `# %%` marker, and markdown produces no outputs.

`sync-prose` folds the new prose into the executed notebook and re-points the stamp,
without executing anything. These tests pin the two properties that make that honest: the
outputs survive, and any change to a code cell is refused.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "notebook_provenance", REPO / ".github" / "scripts" / "notebook_provenance.py"
)
provenance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(provenance)


PY_BEFORE = """# %% [markdown]
# # A notebook
# Some prose.

# %% [markdown]
# A second paragraph, in its own cell.

# %%
total = 1 + 1
total

# %%
print(total)
"""


def _write_pair(root: Path, py_source: str) -> tuple[Path, Path]:
    py = root / "demo.py"
    py.write_text(py_source, encoding="utf-8")
    subprocess.run(
        [
            ".venv/bin/jupytext" if (REPO / ".venv/bin/jupytext").exists() else "jupytext",
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
    # Stand in for an execution: real outputs, real execution counts.
    n = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            n += 1
            cell["execution_count"] = n
            cell["outputs"] = [
                {
                    "output_type": "execute_result",
                    "execution_count": n,
                    "data": {"text/plain": "2"},
                    "metadata": {},
                }
            ]
    nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    return py, nb_path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo, because the stamp names a blob and the tool reads it back."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    return tmp_path


def _stamp(nb_path: Path, py: Path) -> None:
    # `git cat-file` reads the object store, and hashing a file does not put it there.
    # In the repo the .py is committed, so the blob exists; here it has to be added.
    subprocess.run(["git", "add", py.name], cwd=py.parent, check=True)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    nb["metadata"][provenance.STAMP_KEY] = {
        "source_py_blob": provenance.git_blob(py),
        "executed_at": "2026-08-01T00:00:00+00:00",
        "executor": "ml4t-gpu",
        "production": True,
        "parameters": {},
    }
    nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")


def test_deleting_a_markdown_cell_keeps_the_outputs_and_the_stamp(repo):
    py, nb_path = _write_pair(repo, PY_BEFORE)
    _stamp(nb_path, py)
    before = json.loads(nb_path.read_text())
    n_outputs = sum(len(c.get("outputs", [])) for c in before["cells"])
    assert n_outputs == 2

    py.write_text(
        PY_BEFORE.replace("# %% [markdown]\n# A second paragraph, in its own cell.\n\n", "")
    )
    provenance.sync_prose(nb_path)

    after = json.loads(nb_path.read_text())
    assert sum(len(c.get("outputs", [])) for c in after["cells"]) == n_outputs
    stamp = after["metadata"][provenance.STAMP_KEY]
    assert stamp["source_py_blob"] == provenance.git_blob(py)
    # The execution really did happen then, by that executor. Only the blob moves.
    assert stamp["executed_at"] == "2026-08-01T00:00:00+00:00"
    assert stamp["executor"] == "ml4t-gpu"
    assert "without re-executing" in stamp["notes"]


def test_merging_two_markdown_cells_is_a_prose_edit(repo):
    py, nb_path = _write_pair(repo, PY_BEFORE)
    _stamp(nb_path, py)
    py.write_text(
        PY_BEFORE.replace(
            "# Some prose.\n\n# %% [markdown]\n# A second paragraph, in its own cell.\n",
            "# Some prose.\n#\n# A second paragraph, folded in.\n",
        )
    )
    provenance.sync_prose(nb_path)
    after = json.loads(nb_path.read_text())
    assert sum(len(c.get("outputs", [])) for c in after["cells"]) == 2
    assert "folded in" in "".join(
        "".join(c["source"]) for c in after["cells"] if c["cell_type"] == "markdown"
    )


def test_a_changed_constant_is_refused(repo):
    py, nb_path = _write_pair(repo, PY_BEFORE)
    _stamp(nb_path, py)
    py.write_text(PY_BEFORE.replace("total = 1 + 1", "total = 1 + 2"))
    with pytest.raises(SystemExit, match="not a prose-only edit"):
        provenance.sync_prose(nb_path)


def test_an_added_code_cell_is_refused(repo):
    py, nb_path = _write_pair(repo, PY_BEFORE)
    _stamp(nb_path, py)
    py.write_text(PY_BEFORE + "\n# %%\nprint('extra')\n")
    with pytest.raises(SystemExit, match="code cells in the executed source"):
        provenance.sync_prose(nb_path)


def test_retagging_a_code_cell_is_refused(repo):
    """The marker carries the tags, and a tag decides whether a cell is a parameters cell."""
    py, nb_path = _write_pair(repo, PY_BEFORE)
    _stamp(nb_path, py)
    py.write_text(
        PY_BEFORE.replace("# %%\ntotal = 1 + 1", '# %% tags=["parameters"]\ntotal = 1 + 1')
    )
    with pytest.raises(SystemExit, match="not a prose-only edit"):
        provenance.sync_prose(nb_path)


def test_an_unstamped_notebook_is_refused(repo):
    py, nb_path = _write_pair(repo, PY_BEFORE)
    with pytest.raises(SystemExit, match="no provenance stamp"):
        provenance.sync_prose(nb_path)
