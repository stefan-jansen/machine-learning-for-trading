"""The fixture seed has to leave the output dir holding the fixture and nothing else.

CI checks the test-data repo out into a clean container on every run. A lane re-runs
into an existing ML4T_OUTPUT_DIR, so anything the seed leaves behind is a difference
between what CI measures and what the lane measures, on identical code.
"""

from pathlib import Path

from tests.conftest import SEEDED_SUBDIRS, seed_case_study_intermediates


def _fixture(root: Path, **subdirs: str) -> Path:
    src = root / "etfs"
    for name, content in subdirs.items():
        (src / name).mkdir(parents=True, exist_ok=True)
        (src / name / f"{name}.txt").write_text(content, encoding="utf-8")
    src.mkdir(parents=True, exist_ok=True)
    (src / "eligibility.csv").write_text("symbol\nSPY\n", encoding="utf-8")
    return src


def test_seed_replaces_a_registry_left_by_a_previous_run(tmp_path: Path) -> None:
    """run_log/registry.db is the one that matters: a previous run's registry carries
    every training run, prediction set and backtest that run registered, and the
    notebooks then resolve identities the fixture does not carry."""
    src = _fixture(tmp_path / "intermediates", run_log="fixture registry")
    dst = tmp_path / "out" / "etfs"
    (dst / "run_log").mkdir(parents=True)
    (dst / "run_log" / "run_log.txt").write_text("previous run registry", encoding="utf-8")

    seed_case_study_intermediates(src, dst)

    assert (dst / "run_log" / "run_log.txt").read_text(encoding="utf-8") == "fixture registry"


def test_seed_removes_a_subdir_the_fixture_no_longer_ships(tmp_path: Path) -> None:
    """Nothing overwrites a subdir with no source, so a leftover there survives every
    later run while reading as fixture data."""
    src = _fixture(tmp_path / "intermediates", labels="fixture labels")
    dst = tmp_path / "out" / "etfs"
    (dst / "results").mkdir(parents=True)
    (dst / "results" / "stale.json").write_text("{}", encoding="utf-8")

    seed_case_study_intermediates(src, dst)

    assert not (dst / "results").exists()
    assert (dst / "labels" / "labels.txt").read_text(encoding="utf-8") == "fixture labels"


def test_seed_refreshes_top_level_files(tmp_path: Path) -> None:
    src = _fixture(tmp_path / "intermediates", labels="fixture labels")
    dst = tmp_path / "out" / "etfs"
    dst.mkdir(parents=True)
    (dst / "eligibility.csv").write_text("symbol\nSTALE\n", encoding="utf-8")

    seed_case_study_intermediates(src, dst)

    assert (dst / "eligibility.csv").read_text(encoding="utf-8") == "symbol\nSPY\n"


def test_seed_is_idempotent(tmp_path: Path) -> None:
    src = _fixture(tmp_path / "intermediates", run_log="fixture registry", labels="fixture labels")
    dst = tmp_path / "out" / "etfs"

    seed_case_study_intermediates(src, dst)
    first = sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*"))
    seed_case_study_intermediates(src, dst)

    assert sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*")) == first
    assert (dst / "run_log" / "run_log.txt").read_text(encoding="utf-8") == "fixture registry"


def test_every_seeded_subdir_is_reset(tmp_path: Path) -> None:
    """Whatever the list is, a stale copy of each entry does not survive the seed."""
    src = _fixture(tmp_path / "intermediates")
    dst = tmp_path / "out" / "etfs"
    for name in SEEDED_SUBDIRS:
        (dst / name).mkdir(parents=True)
        (dst / name / "stale").write_text("x", encoding="utf-8")

    seed_case_study_intermediates(src, dst)

    assert [name for name in SEEDED_SUBDIRS if (dst / name).exists()] == []
