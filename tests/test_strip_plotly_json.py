"""Stripping the embedded Plotly JSON must lose a payload and nothing else.

`strip_plotly_json.py` removes `application/vnd.plotly.v1+json` from outputs that
already carry a static image of the same figure, because that payload is what makes a
25.3 MB notebook out of 0.03 MB of source and leaves a reader on github.com staring at
an empty pane. `sync-plotly` folds the strip into the provenance stamp, which is only
sound because re-executing cannot produce the strip - it writes the payload back.

Two things therefore have to hold, and both are tested here rather than argued:
the stripper must refuse an output whose Plotly payload is the only copy of the
figure, and `sync-plotly` must refuse a working copy that carries anything beyond
the strip.
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
SCRIPTS = REPO / ".github" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


provenance = _load("notebook_provenance")
stripper = _load("strip_plotly_json")

PLOTLY = stripper.PLOTLY_MIME
# Stands in for the megabytes of serialized traces a real figure carries.
FIGURE = {"data": [{"x": [1, 2, 3], "y": [4, 5, 6], "type": "scatter"}], "layout": {}}


def _nb(outputs: list[dict]) -> dict:
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# A notebook"]},
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": ["fig.show()"],
                "outputs": outputs,
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _plotly_output(*, fallback: bool) -> dict:
    data = {PLOTLY: FIGURE}
    metadata = {}
    if fallback:
        data["image/png"] = "aGVsbG8=\n"
        metadata["image/png"] = {"alt": "A scatter of three points."}
    return {"output_type": "display_data", "data": data, "metadata": metadata}


def _raw(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


# --- the stripper ---------------------------------------------------------------


def test_a_payload_with_a_static_fallback_is_stripped():
    out, stripped, skipped = stripper.strip_plotly(_raw(_nb([_plotly_output(fallback=True)])))
    assert (stripped, skipped) == (1, 0)
    data = json.loads(out)["cells"][1]["outputs"][0]["data"]
    assert PLOTLY not in data
    assert data["image/png"] == "aGVsbG8=\n"


def test_the_alt_text_on_the_surviving_image_is_untouched():
    """The PNG becomes the only rendering, so its description has to survive."""
    out, _, _ = stripper.strip_plotly(_raw(_nb([_plotly_output(fallback=True)])))
    metadata = json.loads(out)["cells"][1]["outputs"][0]["metadata"]
    assert metadata["image/png"]["alt"] == "A scatter of three points."


def test_a_payload_with_no_static_fallback_is_left_alone():
    """Negative control. Stripping this one deletes the only copy of the figure.

    All 654 Plotly outputs in the repository carry a PNG today, so this case does not
    occur and the guard would be invisible if nothing exercised it. A figure rendered
    without a static image is what makes it occur, which is a `plotly.io` renderer
    setting away.
    """
    raw = _raw(_nb([_plotly_output(fallback=False)]))
    out, stripped, skipped = stripper.strip_plotly(raw)
    assert (stripped, skipped) == (0, 1)
    assert out == raw
    assert PLOTLY in json.loads(out)["cells"][1]["outputs"][0]["data"]


def test_a_notebook_with_no_plotly_output_is_returned_byte_identical():
    """Not merely "unchanged": a reserialize would churn 491 notebooks into the diff.

    The input is deliberately NOT in the serialization the stripper writes - it is
    indented by four and its keys are in a different order - because a fixture that
    already matches that form cannot tell a byte-identical return from a reserialize,
    and the assertion below would hold whatever the code did.
    """
    nb = _nb([{"output_type": "stream", "name": "stdout", "text": ["0.31\n"]}])
    raw = json.dumps(nb, indent=4, sort_keys=True) + "\n"
    assert raw != _raw(nb), "fixture must differ from the stripper's own serialization"

    out, stripped, skipped = stripper.strip_plotly(raw)
    assert (stripped, skipped) == (0, 0)
    assert out == raw


def test_other_mime_types_in_the_same_output_survive():
    output = _plotly_output(fallback=True)
    output["data"]["text/html"] = ["<div>table</div>"]
    output["data"]["text/plain"] = ["<Figure>"]
    out, _, _ = stripper.strip_plotly(_raw(_nb([output])))
    data = json.loads(out)["cells"][1]["outputs"][0]["data"]
    assert set(data) == {"image/png", "text/html", "text/plain"}


def test_a_relative_path_is_reported_not_crashed(tmp_path, monkeypatch, capsys):
    """How anyone actually types a path, and it raised ValueError.

    `relative_to(REPO_ROOT)` on an unresolved relative argument raises, but only on
    the branch that prints a path - so a run over an already-clean tree returned 0 and
    the bug stayed invisible. Caught by review on the first push, not by this suite.
    """
    nb_path = tmp_path / "probe.ipynb"
    nb_path.write_text(_raw(_nb([_plotly_output(fallback=True)])), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["strip_plotly_json.py", "--check", "probe.ipynb"])

    assert stripper.main() == 1
    assert "probe.ipynb" in capsys.readouterr().out


def test_a_notebook_outside_the_repo_is_named_absolutely(tmp_path, monkeypatch, capsys):
    """A scratch copy under review is a legitimate argument, not a crash."""
    nb_path = tmp_path / "outside.ipynb"
    nb_path.write_text(_raw(_nb([_plotly_output(fallback=False)])), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["strip_plotly_json.py", "--check", str(nb_path)])

    assert stripper.main() == 0
    assert str(nb_path) in capsys.readouterr().out


# --- folding it into the stamp --------------------------------------------------


@pytest.fixture
def repo(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    return tmp_path


def _committed(repo: Path, nb: dict) -> Path:
    """A stamped notebook, committed, with the stamp agreeing with its own outputs."""
    py = repo / "demo.py"
    py.write_text("# %%\nfig.show()\n", encoding="utf-8")
    subprocess.run(["git", "add", "demo.py"], cwd=repo, check=True)

    nb_path = repo / "demo.ipynb"
    nb["metadata"][provenance.STAMP_KEY] = {
        "source_py_blob": provenance.git_blob(py),
        "outputs_digest": provenance.outputs_digest(nb),
        "library_digest": "0" * 12,
        "executed_at": "2026-08-01T00:00:00+00:00",
        "executor": "ml4t-gpu",
        "production": True,
        "parameters": {},
    }
    nb_path.write_text(_raw(nb), encoding="utf-8")
    subprocess.run(["git", "add", "demo.ipynb"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "executed"], cwd=repo, check=True)
    return nb_path


def test_sync_plotly_keeps_the_run_and_moves_only_the_outputs_digest(repo):
    nb_path = _committed(repo, _nb([_plotly_output(fallback=True)]))
    before = json.loads(nb_path.read_text())["metadata"][provenance.STAMP_KEY]

    stripped, count, _ = stripper.strip_plotly(nb_path.read_text())
    assert count == 1
    nb_path.write_text(stripped, encoding="utf-8")

    digest = provenance.sync_plotly(nb_path)
    after = json.loads(nb_path.read_text())["metadata"][provenance.STAMP_KEY]

    assert after["outputs_digest"] == digest != before["outputs_digest"]
    # The run itself is untouched: this is not a claim that anything was re-executed.
    assert after["executed_at"] == before["executed_at"] == "2026-08-01T00:00:00+00:00"
    assert after["executor"] == before["executor"]
    assert after["source_py_blob"] == before["source_py_blob"]
    assert "Plotly" in after["notes"]


def test_sync_plotly_is_resumable(repo):
    """A batch over 203 notebooks must survive being stopped halfway."""
    nb_path = _committed(repo, _nb([_plotly_output(fallback=True)]))
    nb_path.write_text(stripper.strip_plotly(nb_path.read_text())[0], encoding="utf-8")
    first = provenance.sync_plotly(nb_path)
    assert provenance.sync_plotly(nb_path) == first


def test_an_edited_output_cannot_ride_along_with_the_strip(repo):
    """The whole soundness argument is "nothing else moved", so prove it is enforced."""
    nb_path = _committed(repo, _nb([_plotly_output(fallback=True)]))
    nb = json.loads(stripper.strip_plotly(nb_path.read_text())[0])
    nb["cells"][1]["outputs"].append(
        {"output_type": "stream", "name": "stdout", "text": ["Sharpe 1.41\n"]}
    )
    nb_path.write_text(_raw(nb), encoding="utf-8")

    with pytest.raises(SystemExit, match="something else changed too"):
        provenance.sync_plotly(nb_path)


def test_an_unstripped_notebook_whose_outputs_moved_is_refused(repo):
    """A plain stale notebook must not be launderable through this command."""
    nb_path = _committed(repo, _nb([_plotly_output(fallback=True)]))
    nb = json.loads(nb_path.read_text())
    nb["cells"][1]["outputs"][0]["data"]["image/png"] = "ZGlmZmVyZW50\n"
    nb_path.write_text(_raw(nb), encoding="utf-8")

    with pytest.raises(SystemExit, match="something else changed too"):
        provenance.sync_plotly(nb_path)


def test_an_unstamped_notebook_is_refused(repo):
    nb = _nb([_plotly_output(fallback=True)])
    nb_path = repo / "demo.ipynb"
    nb_path.write_text(_raw(nb), encoding="utf-8")
    subprocess.run(["git", "add", "demo.ipynb"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "unstamped"], cwd=repo, check=True)
    nb_path.write_text(stripper.strip_plotly(nb_path.read_text())[0], encoding="utf-8")

    with pytest.raises(SystemExit, match="carries no provenance stamp"):
        provenance.sync_plotly(nb_path)


def test_a_stamp_predating_outputs_digest_is_left_alone(repo):
    """Undated is not stale, and the difference is six real notebooks.

    A stamp written before `outputs_digest` existed pins nothing, and the gate counts
    such a notebook rather than failing it. The fold has nothing to invalidate and
    nothing to re-stamp, and must not write a digest - that would assert these outputs
    are the ones the run produced, which is exactly what the absent field means nobody
    can say. Before this case was handled the comparison read `<digest> != None` and
    refused the notebook as "stale before this rewrite", which is a false claim about
    `18_transaction_costs/04_vwap_twap_execution.ipynb` and five others.
    """
    nb = _nb([_plotly_output(fallback=True)])
    nb_path = _committed(repo, nb)
    committed = json.loads(nb_path.read_text())
    del committed["metadata"][provenance.STAMP_KEY]["outputs_digest"]
    nb_path.write_text(_raw(committed), encoding="utf-8")
    subprocess.run(["git", "add", "demo.ipynb"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "stamp with no digest"], cwd=repo, check=True)

    nb_path.write_text(stripper.strip_plotly(nb_path.read_text())[0], encoding="utf-8")
    assert provenance.sync_plotly(nb_path) == ""

    after = json.loads(nb_path.read_text())["metadata"][provenance.STAMP_KEY]
    assert "outputs_digest" not in after
    assert PLOTLY not in json.loads(nb_path.read_text())["cells"][1]["outputs"][0]["data"]


def test_an_edited_output_is_refused_even_with_no_outputs_digest(repo):
    """The no-digest branch writes nothing, so it is the easiest place to skip the check.

    Returning early before the rewrite comparison would wave through any edit at all on
    these six notebooks, which is the opposite of what an absent digest should buy.
    """
    nb_path = _committed(repo, _nb([_plotly_output(fallback=True)]))
    committed = json.loads(nb_path.read_text())
    del committed["metadata"][provenance.STAMP_KEY]["outputs_digest"]
    nb_path.write_text(_raw(committed), encoding="utf-8")
    subprocess.run(["git", "add", "demo.ipynb"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "stamp with no digest"], cwd=repo, check=True)

    nb = json.loads(stripper.strip_plotly(nb_path.read_text())[0])
    nb["cells"][1]["outputs"].append(
        {"output_type": "stream", "name": "stdout", "text": ["Sharpe 1.41\n"]}
    )
    nb_path.write_text(_raw(nb), encoding="utf-8")

    with pytest.raises(SystemExit, match="something else changed too"):
        provenance.sync_plotly(nb_path)


def test_sync_paths_still_works_through_the_shared_helper(repo):
    """`sync_paths` and `sync_plotly` share their guards; this pins the older one."""
    output = {
        "output_type": "stream",
        "name": "stdout",
        "text": [f"wrote {Path.home()}/ml4t/public-demo/out.parquet\n"],
    }
    nb_path = _committed(repo, _nb([output]))

    sanitize = _load("sanitize_notebook_paths")
    rewritten = sanitize.sanitize_notebook(nb_path.read_text())[0]
    assert rewritten != nb_path.read_text()
    nb_path.write_text(rewritten, encoding="utf-8")

    before = json.loads(
        subprocess.run(
            ["git", "show", "HEAD:demo.ipynb"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout
    )["metadata"][provenance.STAMP_KEY]
    digest = provenance.sync_paths(nb_path)
    after = json.loads(nb_path.read_text())["metadata"][provenance.STAMP_KEY]
    assert after["outputs_digest"] == digest != before["outputs_digest"]
    assert after["executed_at"] == before["executed_at"]
    assert "paths sanitized" in after["notes"]


def test_stamping_a_fresh_run_strips_before_it_digests(repo):
    """Closes the ordering trap rather than documenting it.

    Every re-run writes the Plotly payload back. If stamping digested the payload and
    the strip happened afterwards, the committed notebook's outputs would no longer be
    the ones the digest describes, and the gate would report OUTPUTS CHANGED on a
    notebook whose numbers never moved. Stamping strips first, so there is no order for
    an author to get wrong.
    """
    py = repo / "demo.py"
    py.write_text("# %%\nfig.show()\n", encoding="utf-8")
    subprocess.run(["git", "add", "demo.py"], cwd=repo, check=True)

    nb_path = repo / "demo.ipynb"
    nb_path.write_text(_raw(_nb([_plotly_output(fallback=True)])), encoding="utf-8")

    stamp = provenance.stamp_notebook(nb_path, "ml4t-gpu", parameters={})

    on_disk = json.loads(nb_path.read_text())
    assert PLOTLY not in on_disk["cells"][1]["outputs"][0]["data"]
    assert on_disk["cells"][1]["outputs"][0]["data"]["image/png"] == "aGVsbG8=\n"
    # The digest describes the file as committed, which is the whole point.
    assert stamp["outputs_digest"] == provenance.outputs_digest(on_disk)
    assert stripper.strip_plotly(nb_path.read_text())[1] == 0


def test_stamping_leaves_a_payload_that_is_the_only_copy(repo):
    """Negative control: stamping must not quietly delete an unbacked figure."""
    py = repo / "demo.py"
    py.write_text("# %%\nfig.show()\n", encoding="utf-8")
    subprocess.run(["git", "add", "demo.py"], cwd=repo, check=True)

    nb_path = repo / "demo.ipynb"
    nb_path.write_text(_raw(_nb([_plotly_output(fallback=False)])), encoding="utf-8")

    provenance.stamp_notebook(nb_path, "ml4t-gpu", parameters={})
    assert PLOTLY in json.loads(nb_path.read_text())["cells"][1]["outputs"][0]["data"]
