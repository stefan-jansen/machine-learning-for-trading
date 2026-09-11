"""The gate has to name the cheapest command that makes its own claim true again.

`source_py_blob` compares whole blobs, so every `.py` edit reads as stale. The report then
has to say what to do about it, and until this landed it had two answers: `sync-prose` when
nothing but markdown moved, and "re-run in the canonical env" for everything else. A pass
that rewrites figure alt text and moves results into tagged cells - which is what conforming
a notebook to the figure-description rule *is* - fell into the second bucket and was priced
at a full execution. It is a `sync-alt`, and `sync-alt` already accepted it; only the report
sent the author somewhere else.

The two classifiers differ in one argument. `drift_is_prose_only` compares with the alt
literals left in, `drift_is_alt_and_prose_only` blanks them, and the second therefore asks
whether anything *other than* an alt literal moved in a code cell. Markdown is free in both,
because both drop the pure-markdown cells: adding, deleting, merging or retagging one cannot
change what a code cell computes.

What must keep failing is the point of the tests below. A changed constant, a moved `# %%`
boundary between two code cells, and a new statement are all executable drift, and blanking
alt literals does not make them anything else.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))

import notebook_provenance  # noqa: E402
from notebook_provenance import (  # noqa: E402
    drift_is_alt_and_prose_only,
    drift_is_prose_only,
)

EXECUTED = """# %% [markdown] tags=[]
# # A heading
#
# A paragraph of prose.

# %% tags=[]
import pandas as pd

THRESHOLD = 21

# %% [markdown] tags=[]
# Another paragraph.

# %% tags=[]
fig = make_figure(THRESHOLD)
show_plotly_with_alt(
    fig,
    "A line of the count per year. It rises steadily and then falls away sharply.",
)
"""


def _blob(text: str) -> str:
    """Write *text* into the object store and return its hash, as a stamp would have."""
    out = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=REPO_ROOT,
        input=text,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


@pytest.fixture
def stamped() -> str:
    return _blob(EXECUTED)


def _py(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "nb.py"
    path.write_text(text, encoding="utf-8")
    return path


# --- what the new bucket must accept -------------------------------------------------

ALT_ONLY = EXECUTED.replace(
    '"A line of the count per year. It rises steadily and then falls away sharply.",',
    '"A line of the count of distinct symbols per year, with the year on the horizontal axis.",',
)

ALT_PLUS_MARKDOWN = ALT_ONLY.replace(
    "# %% [markdown] tags=[]\n# Another paragraph.\n",
    '# %% [markdown] tags=["results"]\n# ### Results\n#\n# The reading goes here.\n\n'
    "# %% [markdown] tags=[]\n# Another paragraph, reworded.\n",
)

RETAG_ONLY = EXECUTED.replace(
    "# %% [markdown] tags=[]\n# Another paragraph.",
    '# %% [markdown] tags=["results"]\n# Another paragraph.',
)


@pytest.mark.parametrize(
    "edited",
    [
        pytest.param(ALT_ONLY, id="alt literal alone"),
        pytest.param(ALT_PLUS_MARKDOWN, id="alt plus a new results cell and a reworded one"),
    ],
)
def test_alt_bucket_accepts(stamped: str, tmp_path: Path, edited: str) -> None:
    assert drift_is_alt_and_prose_only(stamped, _py(tmp_path, edited))


def test_markdown_alone_is_prose_not_alt(stamped: str, tmp_path: Path) -> None:
    """A markdown-only edit belongs to `sync-prose`, the cheaper of the two.

    Both classifiers accept it, and the report asks `drift_is_prose_only` first, so the
    author is sent to the command that does not touch output metadata at all.
    """
    py = _py(tmp_path, RETAG_ONLY)
    assert drift_is_prose_only(stamped, py)
    assert drift_is_alt_and_prose_only(stamped, py)


def test_alt_change_is_not_prose_only(stamped: str, tmp_path: Path) -> None:
    """The buckets do not overlap where it matters: `sync-prose` must refuse an alt edit.

    It keeps the outputs, so an alt the output metadata does not carry would be stamped as
    current while the notebook still renders the old sentence.
    """
    assert not drift_is_prose_only(stamped, _py(tmp_path, ALT_ONLY))


# --- what must still be a re-run ----------------------------------------------------

CONSTANT_MOVED = ALT_ONLY.replace("THRESHOLD = 21", "THRESHOLD = 42")
STATEMENT_ADDED = ALT_ONLY.replace("import pandas as pd\n", "import pandas as pd\nimport numpy\n")
BOUNDARY_MOVED = ALT_ONLY.replace(
    "import pandas as pd\n\nTHRESHOLD", "import pandas as pd\n\n# %% tags=[]\nTHRESHOLD"
)
SEMICOLON_ADDED = ALT_ONLY.replace("fig = make_figure(THRESHOLD)", "fig = make_figure(THRESHOLD);")


@pytest.mark.parametrize(
    "edited",
    [
        pytest.param(CONSTANT_MOVED, id="a constant changed"),
        pytest.param(STATEMENT_ADDED, id="a statement added"),
        pytest.param(BOUNDARY_MOVED, id="a code-cell boundary moved"),
        pytest.param(SEMICOLON_ADDED, id="display suppression changed"),
    ],
)
def test_alt_bucket_refuses_executable_drift(stamped: str, tmp_path: Path, edited: str) -> None:
    """Blanking the alt literals must not forgive anything else in a code cell."""
    py = _py(tmp_path, edited)
    assert not drift_is_alt_and_prose_only(stamped, py)
    assert not drift_is_prose_only(stamped, py)


def test_a_missing_stamped_blob_is_never_forgiven(tmp_path: Path) -> None:
    """With nothing to compare against, the report must not soften. 40 hex digits, no object."""
    assert not drift_is_alt_and_prose_only("0" * 40, _py(tmp_path, ALT_ONLY))


# --- the prose extraction pass ------------------------------------------------------


def test_prose_prints_markdown_and_no_code(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The review pass the commands exist to serve: read the writing, touch no code."""
    py = _py(tmp_path, EXECUTED)
    args = type("Args", (), {"notebooks": [str(py)], "all": False})()
    assert notebook_provenance._cmd_prose(args) == 0
    out = capsys.readouterr().out
    assert "A paragraph of prose." in out
    assert "Another paragraph." in out
    assert "markdown cell 1" in out and "markdown cell 2" in out
    assert "THRESHOLD" not in out
    assert "show_plotly_with_alt" not in out


def test_prose_reports_a_cells_tags(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """An editor moving results into tagged cells has to see which cells are already tagged."""
    py = _py(tmp_path, RETAG_ONLY)
    args = type("Args", (), {"notebooks": [str(py)], "all": False})()
    notebook_provenance._cmd_prose(args)
    assert 'tags=["results"]' in capsys.readouterr().out
