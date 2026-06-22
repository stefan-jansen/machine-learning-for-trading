"""Tests for utils/paths.py — chapter/case-study path resolution.

Pins the redirection semantics of ML4T_OUTPUT_DIR (test isolation env var)
and the input validation on the chapter/strategy registries. The redirection
behavior is load-bearing: every case-study pipeline notebook depends on it
to avoid overwriting production artifacts during tests.
"""

from __future__ import annotations

import pytest

from utils.paths import (
    CHAPTERS,
    REPO_ROOT,
    STRATEGY_IDS,
    get_case_study_dir,
    get_chapter_dir,
    get_output_dir,
)

# -----------------------------------------------------------------------------
# Registry invariants
# -----------------------------------------------------------------------------


def test_chapters_registry_covers_1_through_27() -> None:
    """The book ships 27 chapters; the registry must enumerate all of them."""
    assert set(CHAPTERS.keys()) == set(range(1, 28))


def test_chapter_directory_names_are_prefixed_by_number() -> None:
    """Notebook discovery in tests/pm_helpers relies on the NN_ prefix pattern."""
    for n, dirname in CHAPTERS.items():
        assert dirname.startswith(f"{n:02d}_"), f"chapter {n} dir {dirname!r} missing prefix"


def test_strategy_ids_match_case_studies_dir() -> None:
    """STRATEGY_IDS should mirror case_studies/<id>/ on disk (structure invariant)."""
    cs_root = REPO_ROOT / "case_studies"
    on_disk = {
        p.name
        for p in cs_root.iterdir()
        if p.is_dir() and not p.name.startswith(("_", ".")) and p.name not in {"utils", "config"}
    }
    # STRATEGY_IDS is the enforced registry; on_disk may contain extras (ignored_subdirs)
    # but every declared id must exist on disk.
    missing = STRATEGY_IDS - on_disk
    assert not missing, f"STRATEGY_IDS declare non-existent case studies: {missing}"


# -----------------------------------------------------------------------------
# get_chapter_dir
# -----------------------------------------------------------------------------


def test_get_chapter_dir_returns_absolute_path() -> None:
    path = get_chapter_dir(7)
    assert path.is_absolute()
    assert path.name == "07_defining_the_learning_task"


def test_get_chapter_dir_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="Invalid chapter"):
        get_chapter_dir(99)


def test_get_chapter_dir_rejects_zero() -> None:
    with pytest.raises(ValueError, match="Invalid chapter"):
        get_chapter_dir(0)


# -----------------------------------------------------------------------------
# get_output_dir — test-mode redirection
# -----------------------------------------------------------------------------


def test_get_output_dir_production_path(tmp_path, monkeypatch) -> None:
    """No env var → writes under the chapter dir."""
    monkeypatch.delenv("ML4T_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("ML4T_CHAPTER_OUTPUT_DIR", raising=False)

    # Use create=False to avoid making a directory in the real repo.
    path = get_output_dir(7, "etfs", create=False)
    assert path == get_chapter_dir(7) / "output" / "etfs"


def test_get_output_dir_redirects_under_ml4t_output_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("ML4T_CHAPTER_OUTPUT_DIR", raising=False)

    path = get_output_dir(7, "etfs")
    assert path == tmp_path / "ch07_etfs"
    assert path.exists()


def test_get_output_dir_chapter_specific_env_wins_over_global(tmp_path, monkeypatch) -> None:
    """ML4T_CHAPTER_OUTPUT_DIR should take precedence over ML4T_OUTPUT_DIR."""
    ch_dir = tmp_path / "chapter-only"
    global_dir = tmp_path / "global"
    monkeypatch.setenv("ML4T_CHAPTER_OUTPUT_DIR", str(ch_dir))
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(global_dir))

    path = get_output_dir(11, "etfs")
    assert path.parent == ch_dir
    assert not str(path).startswith(str(global_dir))


def test_get_output_dir_zero_pads_chapter_number(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("ML4T_CHAPTER_OUTPUT_DIR", raising=False)

    assert get_output_dir(3, "x").name == "ch03_x"
    assert get_output_dir(11, "x").name == "ch11_x"


# -----------------------------------------------------------------------------
# get_case_study_dir — test-mode redirection
# -----------------------------------------------------------------------------


def test_get_case_study_dir_production_path(monkeypatch) -> None:
    """No env var → writes under case_studies/{id}/."""
    monkeypatch.delenv("ML4T_OUTPUT_DIR", raising=False)

    path = get_case_study_dir("etfs", create=False)
    assert path == REPO_ROOT / "case_studies" / "etfs"


def test_get_case_study_dir_redirects_under_ml4t_output_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))

    path = get_case_study_dir("etfs")
    assert path == tmp_path / "etfs"
    assert path.exists()


def test_get_case_study_dir_create_false_does_not_mkdir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))

    path = get_case_study_dir("etfs", create=False)
    assert not path.exists()


def test_get_case_study_dir_create_true_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))

    first = get_case_study_dir("etfs", create=True)
    second = get_case_study_dir("etfs", create=True)
    assert first == second
    assert first.exists()
