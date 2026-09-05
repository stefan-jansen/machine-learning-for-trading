"""The quarantine list must name things that exist.

`test-unit` runs `pytest tests/` minus `.github/ci/unit-test-quarantine.txt`. That is
the whole point of the subtraction: a file nobody writes down is gated, so the gate
follows the directory instead of drifting from it.

The subtraction has one failure mode and it is silent in both directions. An entry
naming a path that no longer exists excludes nothing, and pytest does not complain
about an `--ignore` that matches nothing, so the file reads as quarantined while it is
actually running - or, if it was renamed, reads as quarantined while its replacement
runs ungated. An entry naming a test function that has been renamed does the same. In
neither case does anything fail; the list just stops describing reality, which is
exactly how the enumerated lists this replaced went stale.

So: every entry resolves, or this fails.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Anchored to this file, not to the cwd. The workflow step reads the list from the
# repo root and so did an earlier version of this module, which made
# `pytest tests/test_quarantine_list.py` from anywhere else a collection error
# instead of the failure the assertions below describe.
REPO_ROOT = Path(__file__).resolve().parent.parent
QUARANTINE = REPO_ROOT / ".github/ci/unit-test-quarantine.txt"


def _raw_lines() -> list[str]:
    return QUARANTINE.read_text().splitlines() if QUARANTINE.is_file() else []


def _entries() -> list[str]:
    """Parse the list the way the workflow step parses it, trailing comments included."""
    out = []
    for line in _raw_lines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            out.append(entry)
    return out


RAW_LINES = _raw_lines()
ENTRIES = _entries()


def test_the_parser_sees_every_line_that_names_a_path() -> None:
    """Guard the guard, without pinning how long the list is.

    A count threshold would read a legitimate shortening as a parser failure: the
    torch block is 25 of the current entries and is meant to go away, which would
    have turned that cleanup red under a message blaming the parser. What actually
    needs guarding is that the parser does not drop a line the file writes down, so
    compare it against an independent count of the lines that name a `.py` path.
    """
    assert QUARANTINE.is_file(), f"{QUARANTINE} is missing; the workflow step reads it by path"
    assert ENTRIES, f"{QUARANTINE} parsed to nothing; every check below would pass vacuously"
    naming_a_path = [line for line in RAW_LINES if ".py" in line.split("#", 1)[0]]
    assert len(ENTRIES) == len(naming_a_path), (
        f"{len(naming_a_path)} lines in {QUARANTINE} name a .py path but the parser "
        f"produced {len(ENTRIES)} entries; the two have diverged and whatever it "
        "dropped is running ungated"
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=ENTRIES)
def test_every_entry_names_a_file_that_exists(entry: str) -> None:
    path = REPO_ROOT / entry.split("::", 1)[0]
    assert path.is_file(), (
        f"{entry} names {path}, which does not exist. pytest accepts an --ignore or "
        "--deselect that matches nothing, so this entry excludes nothing and the file "
        "it was written for is either gone or running ungated under a new name. Delete "
        "the entry or point it at the current path."
    )


NODEIDS = [e for e in ENTRIES if "::" in e]


@pytest.mark.parametrize("entry", NODEIDS, ids=NODEIDS)
def test_every_deselected_test_is_defined_where_it_says(entry: str) -> None:
    path_part, _, test_name = entry.partition("::")
    path = REPO_ROOT / path_part
    if not path.is_file():
        pytest.skip("covered by test_every_entry_names_a_file_that_exists")
    # The name as written, without a parametrization suffix: --deselect on the bare
    # function name drops every case of a parametrized test, which is how
    # test_model_notebook is excluded.
    wanted = test_name.split("[", 1)[0]
    defined = {
        node.name
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert wanted in defined, (
        f"{entry} deselects {wanted!r}, which {path} does not define. The test was "
        "renamed or removed, so this deselect matches nothing and whatever replaced it "
        "is running."
    )


def test_no_entry_is_both_ignored_and_deselected() -> None:
    """A file-level ignore swallows any nodeid under it, so both is a contradiction.

    Reading the pair, the nodeid looks like the narrower rule that is in force. It is
    not: the ignore wins and the rest of the file is excluded too, which is more than
    the author wrote down.
    """
    ignored = {e for e in ENTRIES if "::" not in e}
    overlapping = sorted(e for e in NODEIDS if e.split("::", 1)[0] in ignored)
    assert not overlapping, (
        "these entries deselect a single test in a file that is also ignored whole: "
        f"{overlapping}. The ignore wins. Drop one of the two so the list says what "
        "is actually excluded."
    )


def _section(marker: str) -> list[str]:
    """The entries under the heading containing *marker*.

    Sections in the quarantine file are separated by full-width `# ---` rules, so a
    heading owns every entry until the next one.
    """
    # Only a comment run opened by a full-width `# ---` rule is a section heading;
    # the shorter runs are per-entry annotations, which sit inside a section and must
    # not close it. Everything after a heading belongs to it until the next heading.
    block: list[str] = []
    heading: list[str] | None = None
    mine = False
    for line in RAW_LINES:
        if line.startswith("# ---"):
            # The rule is drawn both above and below a heading. The second one must
            # not wipe the text collected between them.
            if heading is None:
                heading = []
            continue
        if line.startswith("#"):
            if heading is not None:
                heading.append(line)
            continue
        entry = line.split("#", 1)[0].strip()
        if not entry:
            continue
        if heading is not None:
            mine = any(marker in h for h in heading)
            heading = None
        if mine:
            block.append(entry)
    return block


def test_the_image_job_runs_exactly_what_is_quarantined_for_it() -> None:
    """The two lists are one decision written in two files, so they have to agree.

    Excluding a file here moves it to `test-unit-image`. Nothing makes that true
    except the job naming the same file, and a file dropped from one side is
    invisible from the other: removed from the job it silently runs nowhere, and
    removed from here it goes red in test-unit for want of torch. That is the same
    exclude-more-than-you-meant failure this whole file exists to catch.
    """
    _assert_section_matches_job(
        "modelling environment", "  test-unit-image:", "\n  test-chapters:", exclusive=True
    )


def test_the_data_job_runs_exactly_what_is_quarantined_for_it() -> None:
    """The same decision, for the section that needs the test-data checkout.

    It needed the check more than the image section did. From 2026-08-24 to
    2026-09-03 this section excluded three files from `test-unit` and named no job
    that ran them, and `test-unit-data` - which has the dataset they want - did not
    list them, so they ran nowhere. That is the enumeration defect this whole
    subtraction replaced, reappearing one level down because the job on the other
    side of the exclusion still enumerates by hand.
    """
    _assert_section_matches_job(
        "Needs a dataset", "  test-unit-data:", "\n  test-unit-image:", exclusive=False
    )


def _assert_section_matches_job(
    marker: str, job_header: str, next_header: str, *, exclusive: bool
) -> None:
    """Every test a quarantine section excludes is named by the job that takes it.

    Nothing makes "excluded here, run there" true except the two lists agreeing, and a
    file dropped from the job is invisible from here: it silently runs nowhere. That is
    the same exclude-more-than-you-meant failure this file exists to catch.

    ``exclusive`` is the other direction, and the two sections differ on it. A
    modelling-environment file cannot be collected in `test-unit` at all - it raises on
    the import - so a file the image job runs and this list does not exclude is a red
    `test-unit`, and the equality holds both ways. A data-dependent file collects fine
    and *skips*, so `test-unit-data` legitimately runs files that stay in the sweep; the
    two the job has always run are exactly that. Requiring equality there would demand
    they be quarantined, which would give up the collection they still provide.

    A `--deselect` in the job is deliberately not counted. The quarantine file says what
    `test-unit` does not run; a test deselected inside a file the other job *does* run
    is that job's own business and is justified in the workflow beside it.
    """
    quarantined = set(_section(marker))
    assert quarantined, f"the {marker!r} section of the quarantine file is empty"

    workflow = (REPO_ROOT / ".github/workflows/test.yml").read_text()
    _, _, after = workflow.partition(job_header)
    job, _, _ = after.partition(next_header)
    assert job, f"no {job_header.strip()} job in .github/workflows/test.yml"
    in_job = {
        token
        for line in job.splitlines()
        for token in [line.strip().rstrip("\\").strip()]
        if token.startswith("tests/") and ".py" in token
    }

    assert not (quarantined - in_job), (
        f"the quarantine's {marker!r} section excludes these from test-unit and the "
        f"{job_header.strip()} job does not run them, so they run in no job at all: "
        f"{sorted(quarantined - in_job)}"
    )
    if exclusive:
        assert not (in_job - quarantined), (
            f"the {job_header.strip()} job runs these and the quarantine's {marker!r} "
            f"section does not exclude them, so test-unit collects them too and goes "
            f"red on the import: {sorted(in_job - quarantined)}"
        )
