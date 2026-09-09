"""Correcting a figure description must not cost a re-run, and must not be free of proof.

Two halves, and until 2026-09-08 only one of them existed.

`alt_text_only_drift` decides whether a `.py` that drifted from its stamp drifted only in
alt text. Its second half is meant to require that the outputs on disk already carry the
corrected string, so "edited the .py and left the executed alt saying the old thing" stays
stale. It read those alts out of the NOTEBOOK's own cell source, which is the source that
produced those very outputs - so it compared an execution against itself, was true of every
executed notebook, and said nothing about the edit under adjudication. Measured: appending a
sentence to a plain-literal alt in `cme_futures/03_financial_features.py` was reported
ALT-TEXT ONLY and allowed to commit, with the rendered alt still saying the old thing.

Making that check real leaves the corrected alt genuinely stale, which is correct and, on
its own, useless: nothing could write the new string into the output metadata, so the
documented cheap path could never be reached. `sync-alt` is that half. It is sound because
`show_plotly_with_alt` takes the image from `fig._repr_mimebundle_()`, which never sees the
alt string, so the file it produces is byte-for-byte the file a re-run would produce.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "notebook_provenance", REPO / ".github" / "scripts" / "notebook_provenance.py"
)
provenance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(provenance)

PLAIN = """# %% [markdown]
# # A notebook

# %%
from utils.style import show_plotly_with_alt

show_plotly_with_alt(fig, "Bars sorted from tallest at the left to shortest at the right.")
"""

COMPUTED = """# %% [markdown]
# # A notebook

# %%
from utils.style import show_plotly_with_alt

show_plotly_with_alt(fig, f"IC of {ic:.3f} over {n} folds.")
"""


def _write_pair(root: Path, py_source: str, alt: str) -> tuple[Path, Path]:
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
        if cell["cell_type"] != "code":
            continue
        n += 1
        cell["execution_count"] = n
        cell["outputs"] = [
            {
                "output_type": "display_data",
                "data": {"image/png": "iVBOR"},
                "metadata": {"image/png": {"alt": alt}},
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


def _stamp(nb_path: Path, py: Path) -> str:
    subprocess.run(["git", "add", py.name], cwd=py.parent, check=True)
    blob = provenance.git_blob(py)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    nb["metadata"][provenance.STAMP_KEY] = {
        "source_py_blob": blob,
        "executed_at": "2026-08-01T00:00:00+00:00",
        "executor": "ml4t-gpu",
        "production": True,
        "parameters": {},
    }
    nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    return blob


def _drift(nb_path: Path, py: Path, blob: str) -> bool:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return provenance.alt_text_only_drift(blob, py, nb)


def test_an_alt_the_outputs_do_not_carry_is_not_alt_text_only_drift(repo):
    """The defect this file exists for: the gate used to allow this."""
    py, nb_path = _write_pair(
        repo, PLAIN, "Bars sorted from tallest at the left to shortest at the right."
    )
    blob = _stamp(nb_path, py)
    py.write_text(PLAIN.replace("shortest at the right.", "shortest at the right edge."), "utf-8")
    assert _drift(nb_path, py, blob) is False


def test_an_alt_the_outputs_do_carry_is_alt_text_only_drift(repo):
    """The no-hit half. Without it the test above passes against a gate that always says no."""
    py, nb_path = _write_pair(
        repo, PLAIN, "Bars sorted from tallest at the left to shortest at the right edge."
    )
    blob = _stamp(nb_path, py)
    py.write_text(PLAIN.replace("shortest at the right.", "shortest at the right edge."), "utf-8")
    assert _drift(nb_path, py, blob) is True


def test_sync_alt_writes_the_correction_into_the_outputs(repo):
    py, nb_path = _write_pair(
        repo, PLAIN, "Bars sorted from tallest at the left to shortest at the right."
    )
    blob = _stamp(nb_path, py)
    py.write_text(PLAIN.replace("shortest at the right.", "shortest at the right edge."), "utf-8")
    provenance.sync_alt(nb_path)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    alts = [
        (o["metadata"]["image/png"]["alt"])
        for c in nb["cells"]
        for o in c.get("outputs", [])
        if "image/png" in (o.get("data") or {})
    ]
    assert alts == ["Bars sorted from tallest at the left to shortest at the right edge."]
    assert nb["metadata"][provenance.STAMP_KEY]["executed_at"] == "2026-08-01T00:00:00+00:00"
    assert nb["metadata"][provenance.STAMP_KEY]["source_py_blob"] != blob
    subprocess.run(["git", "add", py.name], cwd=py.parent, check=True)
    assert _drift(nb_path, py, nb["metadata"][provenance.STAMP_KEY]["source_py_blob"]) is True


def test_sync_alt_keeps_the_interpolated_values_of_a_computed_alt(repo):
    """The values need a kernel; the prose around them does not."""
    py, nb_path = _write_pair(repo, COMPUTED, "IC of 0.031 over 5 folds.")
    _stamp(nb_path, py)
    py.write_text(
        COMPUTED.replace(
            'f"IC of {ic:.3f} over {n} folds."', 'f"Rank IC of {ic:.3f} across {n} folds."'
        ),
        "utf-8",
    )
    provenance.sync_alt(nb_path)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    alt = next(
        o["metadata"]["image/png"]["alt"]
        for c in nb["cells"]
        for o in c.get("outputs", [])
        if "image/png" in (o.get("data") or {})
    )
    assert alt == "Rank IC of 0.031 across 5 folds."


def test_sync_alt_refuses_a_changed_constant(repo):
    """It is the alt exception, not a way past the gate."""
    py, nb_path = _write_pair(
        repo, PLAIN, "Bars sorted from tallest at the left to shortest at the right."
    )
    _stamp(nb_path, py)
    py.write_text(
        PLAIN.replace("show_plotly_with_alt(fig,", "show_plotly_with_alt(other_fig,"), "utf-8"
    )
    with pytest.raises(SystemExit, match="Re-run the notebook"):
        provenance.sync_alt(nb_path)


def test_sync_alt_refuses_a_computed_alt_that_interpolates_differently(repo):
    """A different number of values is a change to what the alt asserts, not to its wording.

    Refused before the splice is reached: an added ``FormattedValue`` is a node, so the
    blanked-AST comparison sees it and reports a changed code cell. The message names the
    cell rather than the interpolation, which is accurate - what changed is the call.
    """
    py, nb_path = _write_pair(repo, COMPUTED, "IC of 0.031 over 5 folds.")
    _stamp(nb_path, py)
    py.write_text(
        COMPUTED.replace(
            'f"IC of {ic:.3f} over {n} folds."', 'f"IC of {ic:.3f} over {n} folds, worst {lo:.3f}."'
        ),
        "utf-8",
    )
    with pytest.raises(SystemExit, match="Re-run the notebook"):
        provenance.sync_alt(nb_path)


@pytest.mark.parametrize(
    ("old_segments", "new_segments", "carried", "expected"),
    [
        (
            ("IC of ", " over ", " folds."),
            ("Rank IC of ", " across ", " folds."),
            "IC of 0.031 over 5 folds.",
            "Rank IC of 0.031 across 5 folds.",
        ),
        # A single segment interpolates nothing, so the whole alt is prose and must match.
        (("Sharpe 1.24",), ("Annualised Sharpe 1.24",), "Sharpe 1.24", "Annualised Sharpe 1.24"),
        (("Sharpe ",), ("Annualised Sharpe ",), "Sharpe 1.24", None),
        # Scanning left to right takes the FIRST occurrence of the delimiter, which here is
        # the decimal point inside the value: the value reads as "0" and "031." is appended
        # after the new prose, producing "IC 0 overall.031." and stamping it as current.
        # Anchoring the last segment to the end of the string is what rejects that reading.
        (("IC ", "."), ("IC ", " overall."), "IC 0.031.", "IC 0.031 overall."),
        # More than one split is consistent with the source, so there is no telling which
        # the notebook meant. Refused rather than guessed.
        ((" ", " ", " "), ("x ", " y ", " z"), "a b c d", None),
        # One more gap to fill than there are values to fill it with.
        (("A ", " chart."), ("A ", " plot.", " Extra."), "A bar chart.", None),
        # The carried alt was not produced by these segments at all.
        (
            ("IC of ", " over ", " folds."),
            ("Rank IC of ", " across ", " folds."),
            "something else entirely",
            None,
        ),
    ],
)
def test_splice_keeps_the_values_and_refuses_what_it_cannot_place(
    old_segments, new_segments, carried, expected
):
    """The splice is what makes a computed alt correctable without a kernel, so it is
    pinned directly - `sync_alt` refuses these shapes earlier and cannot reach them."""
    assert provenance._splice_alt(old_segments, new_segments, carried) == expected


def test_code_bodies_drops_the_preamble_so_cells_line_up(repo):
    """`_percent_cells` labels the pre-marker header `code`; a notebook has no cell for it.

    Off by one here does not misalign - it rejects every notebook, so the alt exception
    silently stops working and every corrected alt is priced at a re-run again.
    """
    src = "# ---\n# jupyter: meta\n# ---\n\n# %% [markdown]\n# Prose.\n\n# %%\nx = 1\n"
    assert len(provenance._code_bodies(src)) == 1
    assert provenance._code_bodies(src)[0].strip() == "x = 1"
