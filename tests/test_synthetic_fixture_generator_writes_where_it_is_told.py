"""The synthetic generator must write where it is told, and must not surprise a reader.

Both halves exist because of one incident on 2026-09-11. Every generator function took
its output root from the module-level ``TEST_DATA_ROOT``, so the only thing the script
could do was overwrite ``~/ml4t/test-data/data``; and there was no argument parser, so
``--help`` - typed to find out what the script did - was ignored and ``main()`` ran. It
replaced four fixture files with smaller synthetic ones, including a
``futures/market/individual/ES/data.parquet`` that is byte-identical to production:
19,361 rows became 345.

The files were tracked and were restored from git. What could not be restored by git is
the reason it was possible, which is what these tests hold in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

import generate_test_microstructure as generator  # noqa: E402


def test_generate_all_writes_under_the_root_it_is_given(tmp_path):
    """The contract that makes the script inspectable at all."""
    written = generator.generate_all(tmp_path, quiet=True)

    assert written, "generated nothing"
    for path in written:
        assert path.is_relative_to(tmp_path), f"{path} escaped the requested root"


def test_generate_all_leaves_the_live_fixture_alone(tmp_path):
    """The no-hit half: the test above passes even if the live root is written too."""
    live = generator.TEST_DATA_ROOT
    before = (
        {p: p.stat().st_mtime_ns for p in live.rglob("*") if p.is_file()} if live.is_dir() else {}
    )

    generator.generate_all(tmp_path, quiet=True)

    after = (
        {p: p.stat().st_mtime_ns for p in live.rglob("*") if p.is_file()} if live.is_dir() else {}
    )
    assert after == before, "writing to a scratch root touched the live fixture"


def test_generate_all_reseeds_so_two_calls_agree(tmp_path):
    """Without the reseed one module-level generator carries its position across calls.

    A `Dataset.build` that produced different bytes on its second call in a process
    could not be checked against anything, which is the whole point of declaring it.
    """
    first = {
        p.relative_to(tmp_path / "a"): p.read_bytes()
        for p in generator.generate_all(tmp_path / "a", quiet=True)
    }
    second = {
        p.relative_to(tmp_path / "b"): p.read_bytes()
        for p in generator.generate_all(tmp_path / "b", quiet=True)
    }

    assert first.keys() == second.keys()
    assert first == second, "a second call in the same process produced different bytes"


def test_an_unrecognized_flag_is_an_error_rather_than_a_write():
    """`--help` ran the script and destroyed four files. Argparse is what stops that."""
    with pytest.raises(SystemExit) as excinfo:
        generator.main(["--nonsense"])
    assert excinfo.value.code != 0
