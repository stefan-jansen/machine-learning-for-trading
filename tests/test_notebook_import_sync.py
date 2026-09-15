"""The fourth sync tier: removing an import nothing references is not a re-run.

`source_py_blob` compares whole blobs, so deleting `import numpy` reads as stale and the
report says re-run. AGENTS.md rule 6 already says it does not owe one - a fix that changes
no computed value never does - but the obligation and the gate are two different things.
The gate refuses the commit, rule 5 says never bypass a hook, and without a tier the dead
import in a 734-output notebook cannot land at all. Measured on etfs/13_model_analysis:
fix the .py, `jupytext --update` the pair, outputs untouched, and `check` still exits 1
with STALE.

What this tier is NOT is safe by construction, and that is the difference from the three
before it. A markdown cell is a comment block in the .py; an alt literal reaches the output
as metadata the image bytes never see; a sanitized path is compared against the sanitizer's
own output. An unreferenced import can change what a code cell computes, by side effect,
and no amount of reading the source separates that case from a dead one - `import torch`
for cudart symbol ordering has zero references by construction. The refusals below are the
narrowing: an import carrying `# noqa: F401`, an import carrying any other comment, an
import that was added rather than removed, an import whose removal exposes a trailing
expression to Jupyter's display, and anything else in a code cell.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))

import notebook_provenance  # noqa: E402
from notebook_provenance import (  # noqa: E402
    _comparable,
    _ends_in_an_expression,
    _unused_imports_removed,
    code_cells_only,
    drift_is_prose_only,
    drift_is_unused_import_only,
)

EXECUTED = """# %% [markdown] tags=[]
# # A heading
#
# A paragraph of prose.

# %% tags=[]
import json
from pathlib import Path

import pandas as pd

THRESHOLD = 21

# %% [markdown] tags=[]
# Another paragraph.

# %% tags=[]
frame = pd.DataFrame({"n": [THRESHOLD]})
show_with_alt(frame, "A one-row table of the threshold.")
"""

CLEANED = EXECUTED.replace("import json\nfrom pathlib import Path\n\n", "")


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


# --- the normalizer ------------------------------------------------------------------


def test_normalizer_ignores_the_repository_config() -> None:
    """`--isolated` is load-bearing: pyproject.toml switches F401 off repo-wide.

    Reading that config would return the input unchanged, every comparison would trivially
    hold, and the classifier would accept a genuinely executable drift as an import removal.
    """
    assert "F401" in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    normalized = _unused_imports_removed(EXECUTED)
    assert "import json" not in normalized
    assert "from pathlib import Path" not in normalized
    # pandas is referenced two cells down, so it is not an unused import and must survive.
    assert "import pandas as pd" in normalized


def test_normalizer_keeps_a_suppressed_import() -> None:
    """A `# noqa: F401` import survives normalization, which is what makes its removal visible."""
    source = EXECUTED.replace("import json\n", "import torch  # noqa: F401\n")
    assert "import torch  # noqa: F401" in _unused_imports_removed(source)


# --- what the tier must accept -------------------------------------------------------

CLEANED_PLUS_MARKDOWN = CLEANED.replace(
    "# %% [markdown] tags=[]\n# Another paragraph.",
    '# %% [markdown] tags=["results"]\n# ### Results\n#\n# The reading goes here.',
)


@pytest.mark.parametrize(
    "edited,expected",
    [
        pytest.param(CLEANED, ["Path", "json"], id="two unreferenced imports removed"),
        pytest.param(
            CLEANED_PLUS_MARKDOWN, ["Path", "json"], id="removal plus a retagged markdown cell"
        ),
    ],
)
def test_tier_accepts(stamped: str, tmp_path: Path, edited: str, expected: list[str]) -> None:
    ok, removed = drift_is_unused_import_only(stamped, _py(tmp_path, edited))
    assert ok
    assert sorted(removed) == expected


def test_prose_tier_refuses_an_import_removal(stamped: str, tmp_path: Path) -> None:
    """The tiers do not overlap. An import removal changes an AST, so sync-prose must refuse."""
    assert not drift_is_prose_only(stamped, _py(tmp_path, CLEANED))


def test_tier_refuses_a_markdown_only_edit(stamped: str, tmp_path: Path) -> None:
    """Nothing was removed, so this belongs to sync-prose and the report must send it there."""
    edited = EXECUTED.replace("# Another paragraph.", "# Another paragraph, reworded.")
    ok, removed = drift_is_unused_import_only(stamped, _py(tmp_path, edited))
    assert not ok
    assert removed == []


# --- what must still be a re-run ------------------------------------------------------
#
# Each case names the source that was EXECUTED as well as the edit, because two of the
# refusals are about what the stamped source carried. Reusing one stamped fixture for all
# of them made both annotation cases collapse into the partial-cleanup case, and the
# comment refusal then passed its test while contributing nothing - a mutation that
# deleted it changed no result.

ONE_DEAD = EXECUTED.replace("from pathlib import Path\n", "")
ANNOTATED_NOQA = ONE_DEAD.replace("import json\n", "import torch  # noqa: F401\n")
ANNOTATED_PROSE = ONE_DEAD.replace("import json\n", "import json  # kept for the next stage\n")

CASES = [
    pytest.param(
        EXECUTED,
        EXECUTED.replace("import json\n", ""),
        id="one dead import removed and another left behind",
    ),
    pytest.param(
        ANNOTATED_NOQA,
        ANNOTATED_NOQA.replace("import torch  # noqa: F401\n", ""),
        id="a noqa-suppressed import removed",
    ),
    pytest.param(
        ANNOTATED_PROSE,
        ANNOTATED_PROSE.replace("import json  # kept for the next stage\n", ""),
        id="an import carrying any other comment removed",
    ),
    pytest.param(
        EXECUTED,
        CLEANED.replace("import pandas as pd\n", "import pandas as pd\nimport numpy\n"),
        id="an unreferenced import added",
    ),
    pytest.param(
        EXECUTED, CLEANED.replace("import pandas as pd\n\n", ""), id="a referenced import removed"
    ),
    pytest.param(
        EXECUTED, CLEANED.replace("THRESHOLD = 21", "THRESHOLD = 42"), id="a constant changed"
    ),
    pytest.param(
        EXECUTED,
        CLEANED.replace("THRESHOLD = 21", "THRESHOLD = 21\nOFFSET = 3\nprint(OFFSET)"),
        id="a statement added",
    ),
    pytest.param(
        EXECUTED,
        CLEANED.replace(
            "import pandas as pd\n\nTHRESHOLD", "import pandas as pd\n\n# %% tags=[]\nTHRESHOLD"
        ),
        id="a code-cell boundary moved",
    ),
    pytest.param(
        EXECUTED,
        CLEANED.replace(
            'show_with_alt(frame, "A one-row table of the threshold.")',
            'show_with_alt(frame, "A one-row table of the threshold.");',
        ),
        id="display suppression changed",
    ),
    pytest.param(
        EXECUTED,
        CLEANED.replace(
            "# %% tags=[]\nframe = pd.DataFrame", '# %% tags=["results"]\nframe = pd.DataFrame'
        ),
        id="a code cell retagged",
    ),
]


@pytest.mark.parametrize("executed,edited", CASES)
def test_tier_refuses(tmp_path: Path, executed: str, edited: str) -> None:
    ok, _removed = drift_is_unused_import_only(_blob(executed), _py(tmp_path, edited))
    assert not ok


def test_a_ruff_that_cannot_be_imported_is_a_refusal_not_a_no_op(
    monkeypatch, tmp_path: Path
) -> None:
    """A missing ruff exits 1, exactly as a lint finding does. It must not read as "nothing to fix".

    `test-unit` installed no ruff. `python -m ruff` there exits 1 with "No module named ruff",
    and the first normaliser accepted that as a completed run, read the unmodified file back,
    and reported every drift as executable. The tier was inert and nothing said so - three
    assertion failures in this file were the only symptom, and in a job that did not run them
    there would have been none: `sync-imports` would refuse every notebook with a message about
    code cells having moved, which reads exactly like a correct refusal.
    """
    real = importlib.util.find_spec

    monkeypatch.setattr(
        notebook_provenance.importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "ruff" else real(name, *a, **k),
    )
    assert _unused_imports_removed(EXECUTED) is None
    ok, removed = drift_is_unused_import_only(_blob(EXECUTED), _py(tmp_path, CLEANED))
    assert not ok
    assert removed == []


def test_a_warning_on_stderr_does_not_disarm_the_normalizer(monkeypatch) -> None:
    """Ruff writes warnings to stderr from runs that ran and succeeded, so emptiness is not the test.

    Measured on ruff 0.15.14: `check --no-cache <empty dir>` exits 0 with "No Python files found
    under the given path(s)" on stderr, and `check --isolated --select D203,D211,F401` exits 1
    with twelve lines of real diagnostics on stdout AND an incompatible-rules warning on stderr.
    Neither can fire under the argv this function uses - one `--select` cannot conflict, the path
    is an explicit existing file - but both are one flag or one ruff release away, and a
    normaliser that refused on any stderr would go inert in the silent direction again.
    """
    real_run = notebook_provenance.subprocess.run

    def warn_but_work(cmd, *args, **kwargs):
        result = real_run(cmd, *args, **kwargs)
        if len(cmd) > 2 and cmd[1:3] == ["-m", "ruff"]:
            return subprocess.CompletedProcess(
                cmd, result.returncode, result.stdout, "warning: something ruff felt like saying\n"
            )
        return result

    monkeypatch.setattr(notebook_provenance.subprocess, "run", warn_but_work)
    normalized = _unused_imports_removed(EXECUTED)
    assert normalized is not None
    assert "import json" not in normalized
    assert "import pandas as pd" in normalized


def test_tier_refuses_a_stamped_blob_that_is_gone(tmp_path: Path) -> None:
    """No stored source means no comparison, and an unreadable comparison never softens a report."""
    ok, removed = drift_is_unused_import_only("0" * 40, _py(tmp_path, CLEANED))
    assert not ok
    assert removed == []


# --- the display refusal --------------------------------------------------------------
#
# A way removing an import changes a notebook's output without changing what it computes,
# and one the other comparisons are all blind to: both sides normalize to the same source,
# the code-cell ASTs agree, and preserved output counts are what a notebook that has not
# been re-run looks like either way.

TRAILING_IMPORT = """# %% [markdown] tags=[]
# # A heading

# %% tags=[]
import pandas as pd

frame = pd.DataFrame({"n": [21]})
frame
import json
"""

LEADING_IMPORT = """# %% [markdown] tags=[]
# # A heading

# %% tags=[]
import json

import pandas as pd

frame = pd.DataFrame({"n": [21]})
frame
"""

NO_IMPORT = """# %% [markdown] tags=[]
# # A heading

# %% tags=[]
import pandas as pd

frame = pd.DataFrame({"n": [21]})
frame
"""


def test_tier_refuses_a_removal_that_exposes_a_trailing_expression(tmp_path: Path) -> None:
    """`frame` then `import json` displays nothing; drop the import and it renders a table.

    The committed notebook holds the cell's outputs from an execution where the import was
    final, so accepting this would restamp a source whose next execution produces an output
    the .ipynb does not have. Every other comparison the tier makes passes on this pair,
    which is why the refusal has to be its own check rather than a consequence of one.
    """
    ok, removed = drift_is_unused_import_only(_blob(TRAILING_IMPORT), _py(tmp_path, NO_IMPORT))
    assert not ok
    assert removed == []


def test_tier_accepts_a_removal_in_front_of_a_trailing_expression(tmp_path: Path) -> None:
    """`frame` is the final statement before and after, so the display does not change.

    The negative half of the rule. Without it the guard could refuse every cell that ends
    in an expression and still pass its positive test, which is conservative rather than
    decidable.
    """
    ok, removed = drift_is_unused_import_only(_blob(LEADING_IMPORT), _py(tmp_path, NO_IMPORT))
    assert ok
    assert removed == ["json"]


def test_nothing_the_tier_compares_separates_the_two_display_cases() -> None:
    """The comparison the classifier makes cannot tell the refused pair from the accepted one.

    Pins what the refusal is for. Both stamped sources normalize to code cells with the same
    AST - a trailing import and a leading one are the same statements in a different order,
    and both are gone after the fix - so the check that rejects one and accepts the other has
    to be its own, not a consequence of the AST comparison. If a later change makes that
    comparison distinguish these, this fails and the display check can be re-examined rather
    than left in place answering a question something else now answers.
    """
    trailing = code_cells_only(_comparable(_unused_imports_removed(TRAILING_IMPORT)))
    leading = code_cells_only(_comparable(_unused_imports_removed(LEADING_IMPORT)))
    assert trailing is not None
    assert trailing == leading


@pytest.mark.parametrize(
    "body,expected",
    [
        pytest.param("frame = 1\nframe", True, id="a trailing expression"),
        pytest.param("frame = 1\nframe\nimport json", False, id="a trailing import"),
        pytest.param("frame = 1", False, id="a trailing assignment"),
        pytest.param("frame = 1\nframe;", True, id="a semicolon does not change the statement"),
        pytest.param("", False, id="an empty cell"),
        pytest.param("frame =", None, id="a body that does not parse"),
    ],
)
def test_the_display_predicate(body: str, expected: bool | None) -> None:
    """What Jupyter renders is the last statement's kind, and nothing else about the cell.

    A semicolon suppresses the display at run time without changing the statement, so it is
    True here; `_comparable` carries the semicolon flags separately and the prose tier
    already refuses a change to them.
    """
    assert _ends_in_an_expression(body) is expected
