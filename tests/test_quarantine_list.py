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

QUARANTINE = Path(".github/ci/unit-test-quarantine.txt")


def _entries() -> list[str]:
    """Parse the list the way the workflow step parses it, trailing comments included."""
    out = []
    for line in QUARANTINE.read_text().splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            out.append(entry)
    return out


ENTRIES = _entries()


def test_the_list_is_not_empty() -> None:
    """Guard the guard: an unreadable or relocated file would pass everything below."""
    assert QUARANTINE.is_file(), f"{QUARANTINE} is missing; the workflow step reads it by path"
    assert len(ENTRIES) > 20, (
        f"only {len(ENTRIES)} entries parsed from {QUARANTINE}; the parser and the file "
        "have diverged, and every check below would pass vacuously"
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=ENTRIES)
def test_every_entry_names_a_file_that_exists(entry: str) -> None:
    path = Path(entry.split("::", 1)[0])
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
    path = Path(path_part)
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
