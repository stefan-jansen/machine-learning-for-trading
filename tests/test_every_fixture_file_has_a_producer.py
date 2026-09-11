"""Most of the fixture had no producer, and no test could see that.

`tests/test_fixture_manifest_matches_builders.py` checks every *declared* dataset
against its builder and against the data on disk.
`tests/test_one_producer_per_fixture_file.py` checks that the two producers do not
write the same path. Neither looks at the fixture root and asks which files no
producer accounts for, so a file could sit in the fixture for a year with nothing
able to rebuild it and every test still pass - which is how 149 of 327 files got
there.

A file with no producer cannot be regenerated when production moves, cannot be
checked against production, and cannot be explained. The backlog turned up three
files byte-identical to each other at pre-migration paths that nothing reads, one
holding the whole FinancialPhraseBank corpus under a filename that promises the
unanimous subset, an options panel built from a different universe than the one its
loader documents - so four of the eight symbols a chapter-8 notebook asks for came
back empty under CI - and a 20,000-row trade-only panel at `nasdaq100_taq/`, which
no code resolves at all: `load_nasdaq100_taq` reads `trade_and_quotes/symbol=*`.

`UNPRODUCED` is down to that last one, and it is the ratchet: a new fixture file with
no producer fails immediately, an exemption has to be named here, and a file that
gains a producer has to leave the list. It only shrinks.

Two producers are declared, and two is the whole list. `create_test_data.py` derives
from production and each `Dataset` names what it owns; `generate_test_microstructure.py`
is synthetic and `generate_all` returns what it writes. `generate_skip_data.py` used to
be a third, writing two paths into the fixture and declaring neither; both turned out to
be paths no loader resolves, so it was cut back to `intermediates/` instead of declared.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import generate_test_microstructure as generator  # noqa: E402

from tests.create_test_data import DATASETS  # noqa: E402

# One file is left, and it is a deletion rather than a declaration: no code resolves
# `nasdaq100_taq/`. `load_nasdaq100_taq` reads `trade_and_quotes/symbol=*`, which
# `generate_test_microstructure.py` writes, and both chapter-3 TAQ notebooks reach the
# data through that loader. The file itself is 20,000 rows evenly spaced 117 seconds
# apart, five symbols over twenty March 2020 sessions, `event_type` "trade" on every
# row and no quote events at all - so it is neither the AlgoSeek tick sample the
# `nasdaq100_taq` name promises nor anything the Lee-Ready section could be run on.
# It stays here for one more change: removing it from this list and deleting it from
# the fixture cannot be done in one commit, because whichever half lands first turns
# a test red on every branch that has not merged the other.
UNPRODUCED = frozenset({"equities/market/microstructure/nasdaq100_taq/data.parquet"})


def _owned_by_a_dataset(on_disk: set[str]) -> set[str]:
    """The files `create_test_data.py` claims, with directory entries expanded.

    Expanded against what is on disk rather than listed: `Dataset.owns` may name a
    directory, and the point of the check is which real files are covered.
    """
    covered: set[str] = set()
    for dataset in DATASETS:
        for owned in dataset.owns:
            prefix = owned.as_posix()
            covered |= {path for path in on_disk if path == prefix or path.startswith(f"{prefix}/")}
    return covered


@pytest.fixture(scope="module")
def fixture_root(test_data_dir: Path) -> Path:
    """The test-data checkout, or a skip.

    Production has no manifest, and against production these declarations are a
    category error rather than a failure.
    """
    if not (test_data_dir / "manifest.json").is_file():
        pytest.skip(f"{test_data_dir} is not a test-data checkout (no manifest.json)")
    return test_data_dir


@pytest.fixture(scope="module")
def on_disk(fixture_root: Path) -> set[str]:
    """Every fixture data file, relative to the root. The manifest describes, so it
    is not itself a fixture file."""
    return {
        path.relative_to(fixture_root).as_posix()
        for path in fixture_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }


@pytest.fixture(scope="module")
def produced(on_disk: set[str], tmp_path_factory) -> set[str]:
    """Everything a declared producer writes."""
    root = tmp_path_factory.mktemp("generated")
    generated = {
        path.relative_to(root).as_posix() for path in generator.generate_all(root, quiet=True)
    }
    return _owned_by_a_dataset(on_disk) | generated


def test_every_fixture_file_has_a_producer(on_disk: set[str], produced: set[str]) -> None:
    """The check itself. A new fixture file with no builder fails here."""
    orphans = sorted(on_disk - produced - UNPRODUCED)
    assert not orphans, (
        f"{len(orphans)} fixture files are written by no declared producer and are not "
        f"in UNPRODUCED: {orphans}. Declare each in tests/create_test_data.py as a "
        "Dataset, or write it from tests/generate_test_microstructure.py. Adding it to "
        "UNPRODUCED is not the remedy: that list only shrinks."
    )


def test_no_exempt_path_has_gained_a_producer(produced: set[str]) -> None:
    """The ratchet. Declaring a file is only half of retiring it from the backlog."""
    retired = sorted(UNPRODUCED & produced)
    assert not retired, (
        f"{retired} now have a producer and must be removed from UNPRODUCED. Left in "
        "place the list stops measuring the backlog and starts hiding a regression."
    )


def test_no_exempt_path_has_left_the_fixture(on_disk: set[str]) -> None:
    """A deleted file leaves the list too, so it never grants a future file cover."""
    gone = sorted(UNPRODUCED - on_disk)
    assert not gone, f"{gone} are in UNPRODUCED and not in the fixture; remove them from it."


def test_the_producers_are_what_cover_the_rest(on_disk: set[str], produced: set[str]) -> None:
    """Negative selftest.

    The three tests above would pass on a `produced` that resolved to nothing, as
    long as UNPRODUCED happened to list the whole fixture - and would pass just as
    well if `Dataset.owns` expansion silently matched no files, which is the failure
    mode a path-prefix match invites. Both make the check decorative. So: the
    producers must account for the fixture that UNPRODUCED does not, exactly.
    """
    assert produced, "no declared producer resolved to any file on disk"
    uncovered = on_disk - UNPRODUCED
    assert uncovered <= produced
    assert len(uncovered) > len(UNPRODUCED), (
        f"{len(uncovered)} of {len(on_disk)} fixture files have a producer against "
        f"{len(UNPRODUCED)} that do not; the backlog is no longer the minority and "
        "this test is the wrong shape for it."
    )
