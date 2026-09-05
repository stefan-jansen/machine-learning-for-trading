"""The stamp records what the run produced, and the code that produced it.

`source_py_blob` answers one question: has the paired `.py` changed since somebody
stamped this notebook. Two things a reader sees are outside it.

**The outputs.** A notebook whose `.py` is untouched passes the staleness check
carrying outputs from any earlier run. That is a path a superseded result actually
took: `nbconvert --execute --inplace` under `nohup ... &` exits 0 without rewriting
the notebook, so the stamp goes on a file still holding the previous run's figures,
and every check in the repository passes it.

**The library.** The numbers in a case study are computed in `case_studies/utils/*.py`,
which the stamp never covered. #606 changed `causal.py` under all nine case studies at
once; `cme_futures` found by hand that its committed outputs described code no longer in
the tree - `analysis_rows` 88695 -> 88633 on `fwd_ret_5d`, and every `request_hash` with
it. Nothing in the repository could answer "did this run use the code that is here now".

Both are recorded at stamp time and recomputed by `check`. The outputs digest fails the
gate, because a mismatch means the notebook is not the run it claims. The library digest
is reported and does not fail: on the measurement in #917 it would block at least five
notebooks across three case studies on the day it landed, and turning it into a failure
is a separate decision from being able to see it.

A stamp written before either field existed has neither, and is counted rather than
failed - 423 of them at the time this landed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))

import notebook_provenance  # noqa: E402
from notebook_provenance import (  # noqa: E402
    check_all,
    library_digest,
    outputs_digest,
    repo_local_sources,
    stamp_notebook,
)


def _code_cell(source: str, outputs: list[dict], execution_count: int | None = 1) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": outputs,
        "execution_count": execution_count,
    }


def _stream(text: str) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": [text]}


def _notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


# -----------------------------------------------------------------------------
# outputs_digest
# -----------------------------------------------------------------------------


def test_a_changed_output_changes_the_digest():
    """The whole point: the stored result is part of what the stamp covers."""
    before = _notebook([_code_cell("print(sharpe)", [_stream("0.81\n")])])
    after = _notebook([_code_cell("print(sharpe)", [_stream("0.42\n")])])

    assert outputs_digest(before) != outputs_digest(after)


def test_re_running_the_same_cells_renumbers_without_moving_the_digest():
    """`execution_count` is the kernel's counter, not a result.

    A notebook re-executed to the same values must keep its digest, or every
    re-run would read as a changed result and the class would be noise.
    """
    outputs = [_stream("0.81\n")]
    first = _notebook([_code_cell("print(sharpe)", outputs, execution_count=1)])
    second = _notebook([_code_cell("print(sharpe)", outputs, execution_count=7)])

    assert outputs_digest(first) == outputs_digest(second)


def test_an_execution_count_inside_an_output_is_ignored_too():
    """`execute_result` carries its own counter, nested one level down."""
    result = {"output_type": "execute_result", "data": {"text/plain": ["3"]}}
    first = _notebook([_code_cell("1 + 2", [{**result, "execution_count": 1}])])
    second = _notebook([_code_cell("1 + 2", [{**result, "execution_count": 9}])])

    assert outputs_digest(first) == outputs_digest(second)


def test_editing_prose_does_not_change_the_digest():
    """`sync-prose` exists so a markdown edit costs no re-run; this keeps it true."""
    code = _code_cell("print(sharpe)", [_stream("0.81\n")])
    before = _notebook([{"cell_type": "markdown", "metadata": {}, "source": "## A"}, code])
    after = _notebook([{"cell_type": "markdown", "metadata": {}, "source": "## B"}, code])

    assert outputs_digest(before) == outputs_digest(after)


def test_the_digest_is_over_the_parsed_notebook_not_its_bytes(tmp_path: Path):
    """Reformatting the JSON must not read as a changed result."""
    nb = _notebook([_code_cell("print(sharpe)", [_stream("0.81\n")])])
    compact = json.loads(json.dumps(nb, separators=(",", ":")))
    spaced = json.loads(json.dumps(nb, indent=4))

    assert outputs_digest(compact) == outputs_digest(spaced)


# -----------------------------------------------------------------------------
# library_digest
# -----------------------------------------------------------------------------


def test_a_case_study_notebook_reaches_the_helpers_that_compute_its_numbers():
    """The concrete case #917 was filed from.

    `case_studies/utils/causal.py` is where a causal notebook's numbers come from,
    and it is exactly what `source_py_blob` does not cover.
    """
    py = REPO_ROOT / "case_studies/etfs/12_causal_dml.py"
    if not py.exists():
        pytest.skip("the etfs causal notebook is not in this tree")

    reached = {p.relative_to(REPO_ROOT).as_posix() for p in repo_local_sources(py)}

    assert "case_studies/utils/causal.py" in reached


def test_the_import_graph_is_transitive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A module the notebook imports indirectly still counts.

    A one-level scan would miss `causal.py` entirely for any notebook that reaches
    it through `case_studies/research/`, which is how most of them do.
    """
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    (tmp_path / "leaf.py").write_text("VALUE = 1\n")
    (tmp_path / "middle.py").write_text("import leaf\n")
    entry = tmp_path / "entry.py"
    entry.write_text("import middle\n")

    reached = {p.name for p in repo_local_sources(entry)}

    assert reached == {"middle.py", "leaf.py"}


def test_a_module_imported_by_name_from_a_package_is_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`from case_studies.utils import causal` names the module in the alias.

    `node.module` is the package there, so resolving only that finds
    `__init__.py` and stops - one short of the file that actually changed.
    """
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "causal.py").write_text("EFFECT = 1\n")
    entry = tmp_path / "entry.py"
    entry.write_text("from pkg import causal\n")

    reached = {p.relative_to(tmp_path).as_posix() for p in repo_local_sources(entry)}

    assert "pkg/causal.py" in reached


def test_an_installed_package_is_not_repo_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Third-party versions are uv.lock's business, and check_env_matches_lock's."""
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    entry = tmp_path / "entry.py"
    entry.write_text("import json\nimport numpy as np\nfrom pathlib import Path\n")

    assert repo_local_sources(entry) == []


def test_moving_a_module_is_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The digest is keyed by path as well as content.

    Two files with identical bytes at different paths are not the same import
    graph, and a rename that changes what a notebook resolves must show.
    """
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    entry = tmp_path / "entry.py"
    entry.write_text("import helper\n")
    before = library_digest(entry)

    (tmp_path / "helper.py").rename(tmp_path / "renamed.py")
    entry.write_text("import renamed\n")

    assert library_digest(entry) != before


def test_a_notebook_with_no_repo_imports_still_has_a_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """ "No repo-local imports" and "not recorded" must not look the same."""
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    entry = tmp_path / "entry.py"
    entry.write_text("import json\n")

    assert library_digest(entry)


# -----------------------------------------------------------------------------
# The gate
# -----------------------------------------------------------------------------


def _seed_stamped_pair(root: Path, source: str = "print(1)\n") -> tuple[Path, Path]:
    py = root / "nb.py"
    py.write_text(f"# %%\n{source}")
    nb_path = root / "nb.ipynb"
    nb_path.write_text(json.dumps(_notebook([_code_cell(source, [_stream("1\n")])])))
    stamp_notebook(nb_path, executor="test", parameters={})
    return py, nb_path


def test_replacing_the_outputs_fails_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A stamped notebook holding outputs no run of it produced.

    This is the `--execute --inplace` failure: the stamp claims a run, the `.py`
    is untouched so the staleness check passes, and the figures are the previous
    run's.
    """
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(notebook_provenance, "iter_notebooks", lambda: [tmp_path / "nb.ipynb"])
    _, nb_path = _seed_stamped_pair(tmp_path)
    assert check_all().outputs_changed == []

    nb = json.loads(nb_path.read_text())
    nb["cells"][0]["outputs"] = [_stream("999\n")]
    nb_path.write_text(json.dumps(nb))

    result = check_all()
    assert result.outputs_changed == ["nb.ipynb"]
    assert result.stale == [], "the .py is untouched, which is the whole problem"


def test_moving_an_imported_module_is_reported_but_does_not_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Library drift is named and printed, not enforced - #917's own sequencing."""
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(notebook_provenance, "iter_notebooks", lambda: [tmp_path / "nb.ipynb"])
    (tmp_path / "helper.py").write_text("EFFECT = 1\n")
    _seed_stamped_pair(tmp_path, source="import helper\nprint(helper.EFFECT)\n")
    assert check_all().library_drift == []

    (tmp_path / "helper.py").write_text("EFFECT = 2\n")

    result = check_all()
    assert result.library_drift == ["nb.ipynb"]
    assert result.outputs_changed == [], "the stored outputs are untouched"
    assert result.stale == [], "and so is the .py"


def test_a_stamp_predating_the_digests_is_counted_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Every stamp written before this change lacks both fields.

    Failing them would turn every branch red at once for a defect none of them
    has. Counting them keeps the backfill visible instead.
    """
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(notebook_provenance, "iter_notebooks", lambda: [tmp_path / "nb.ipynb"])
    _, nb_path = _seed_stamped_pair(tmp_path)

    nb = json.loads(nb_path.read_text())
    stamp = nb["metadata"][notebook_provenance.STAMP_KEY]
    del stamp["outputs_digest"]
    del stamp["library_digest"]
    nb["cells"][0]["outputs"] = [_stream("anything at all\n")]
    nb_path.write_text(json.dumps(nb))

    result = check_all()
    assert result.undigested == ["nb.ipynb"]
    assert result.outputs_changed == []
    assert result.library_drift == []


def test_stamping_records_both_digests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    _, nb_path = _seed_stamped_pair(tmp_path)

    stamp = json.loads(nb_path.read_text())["metadata"][notebook_provenance.STAMP_KEY]

    assert stamp["outputs_digest"] and stamp["library_digest"]


def test_folding_prose_in_keeps_the_outputs_digest_valid(tmp_path: Path, monkeypatch):
    """`sync-prose` must stay free, which is the whole reason it exists.

    It rewrites `source_py_blob` and keeps the executed outputs. If the outputs
    digest did not survive that, every prose fold would be reported as a changed
    result and the cheapest correction in the repository would cost a re-run
    again - the exact cost sync-prose was built to remove.
    """
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(notebook_provenance, "iter_notebooks", lambda: [tmp_path / "nb.ipynb"])

    py = tmp_path / "nb.py"
    py.write_text("# %% [markdown]\n# A paragraph.\n\n# %%\nprint(1)\n")
    subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "ipynb", "--output", "nb.ipynb", "nb.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    nb_path = tmp_path / "nb.ipynb"
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = 1
            cell["outputs"] = [_stream("1\n")]
    nb_path.write_text(json.dumps(nb, indent=1) + "\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    stamp_notebook(nb_path, executor="test", parameters={})
    assert check_all().outputs_changed == []

    py.write_text(
        "# %% [markdown]\n# A rewritten paragraph, and a second sentence.\n\n# %%\nprint(1)\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    notebook_provenance.sync_prose(nb_path)

    result = check_all()
    assert result.outputs_changed == []
    assert result.stale == [], "sync-prose re-points the blob, so nothing is stale"
