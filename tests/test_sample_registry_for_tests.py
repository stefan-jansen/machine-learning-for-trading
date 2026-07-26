"""Tests for the registry sampler's output-root guard.

Step 3 of the regeneration path unlinks each destination registry before opening
its source, and a production registry.db is 43-180 MB and gitignored. A wrong
--output therefore destroys the results source of truth with nothing to restore it
from, which is the one failure in this path that cannot be undone by re-running it.
"""

from pathlib import Path

from tests.sample_registry_for_tests import (
    CASE_STUDY_IDS,
    CODE_CS_DIR,
    DEFAULT_INTERMEDIATES_DIR,
    rejected_output_root,
)


def test_the_intended_output_root_is_accepted() -> None:
    assert rejected_output_root(DEFAULT_INTERMEDIATES_DIR) is None


def test_a_case_studies_tree_is_rejected() -> None:
    """Named-directory rule, not path equality: in a worktree CODE_CS_DIR is the
    worktree's own tree while the canonical registries live in ~/ml4t/code."""
    assert rejected_output_root(CODE_CS_DIR) is not None
    assert rejected_output_root(Path.home() / "ml4t" / "code" / "case_studies") is not None
    assert rejected_output_root(CODE_CS_DIR / "etfs") is not None


def test_a_root_resolving_onto_a_source_registry_is_rejected() -> None:
    """The destination for cs_id is <root>/<cs_id>/run_log/registry.db, so a root one
    level above a case-study tree collides with the source even under another name."""
    assert rejected_output_root(CODE_CS_DIR.parent / "case_studies") is not None


def test_a_symlinked_destination_is_rejected(tmp_path: Path) -> None:
    """The worktree setup symlinks each case study's run_log to the canonical one, so
    a destination that only looks separate is the normal case, not an exotic one."""
    root = tmp_path / "intermediates"
    (root / "etfs").mkdir(parents=True)
    (root / "etfs" / "run_log").symlink_to(CODE_CS_DIR / "etfs" / "run_log")

    assert rejected_output_root(root) is not None


def test_a_symlinked_case_study_directory_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "intermediates"
    root.mkdir(parents=True)
    (root / "etfs").symlink_to(CODE_CS_DIR / "etfs")

    assert rejected_output_root(root) is not None


def test_every_case_study_is_covered_by_the_check(tmp_path: Path) -> None:
    """The guard iterates CASE_STUDY_IDS; an empty list would make it vacuous."""
    assert len(CASE_STUDY_IDS) == 9
    assert rejected_output_root(tmp_path) is None
