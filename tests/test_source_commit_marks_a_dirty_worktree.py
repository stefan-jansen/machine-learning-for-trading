"""A recorded commit has to say whether it describes the tree the run read.

`git_commit` on a registry row, and `source_commit` inside `runtime_provenance`,
were both HEAD at write time with nothing consulting the worktree. A run started
from a checkout carrying uncommitted edits attested a source that could not have
produced it, and the row read exactly like one whose source is addressable
(ml4t/agent-workspace#1162).

Marking is only possible going forward: an unmarked historical row is not a clean
one, because the tree it would have been compared against is gone.

This module imports nothing heavier than `utils.runtime`, which is stdlib-only, so
it runs in the torch-free `test-unit` job.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from case_studies.utils.runtime import source_commit, worktree_marker


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "model.py").write_text("VERSION = 1\n")
    _git(root, "add", "model.py")
    _git(root, "commit", "-q", "-m", "first")
    return root


def test_a_clean_worktree_records_the_commit_alone(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD")

    assert worktree_marker(repo) == ""
    assert source_commit(repo) == head


def test_a_modified_tracked_file_marks_the_commit(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "model.py").write_text("VERSION = 2\n")

    assert worktree_marker(repo) == "+dirty"
    assert source_commit(repo) == f"{head}+dirty"


def test_an_untracked_file_marks_the_commit_separately(repo: Path) -> None:
    """An untracked module is importable, so it is part of what the run read.

    This is the case a `git diff --quiet` check would miss, and the reason the
    marker is taken from `git status --porcelain` instead. It is reported apart
    from `+dirty` because it is the weaker failure, and because `py_vs_head` in
    `ml4t/agents` already separates the two for the run-level marker.
    """
    (repo / "scratch_model.py").write_text("VERSION = 99\n")

    assert worktree_marker(repo) == "+untracked"
    assert source_commit(repo).endswith("+untracked")


def test_a_tracked_change_outranks_an_untracked_file(repo: Path) -> None:
    """With both present the row has to report the stronger failure."""
    (repo / "model.py").write_text("VERSION = 6\n")
    (repo / "scratch_model.py").write_text("VERSION = 99\n")

    assert worktree_marker(repo) == "+dirty"


def test_a_staged_change_marks_the_commit(repo: Path) -> None:
    (repo / "model.py").write_text("VERSION = 3\n")
    _git(repo, "add", "model.py")

    assert worktree_marker(repo) == "+dirty"


def test_committing_the_change_clears_the_marker(repo: Path) -> None:
    (repo / "model.py").write_text("VERSION = 4\n")
    _git(repo, "add", "model.py")
    _git(repo, "commit", "-q", "-m", "second")

    assert worktree_marker(repo) == ""
    assert source_commit(repo) == _git(repo, "rev-parse", "HEAD")


def test_no_repository_is_unknown_rather_than_clean(tmp_path: Path) -> None:
    """Absence of an answer must not read as a clean tree."""
    outside = tmp_path / "not-a-repo"
    outside.mkdir()

    assert worktree_marker(outside) == "+unknown"
    assert source_commit(outside) == "unknown"


def test_the_registry_column_carries_the_same_marker(repo: Path, monkeypatch) -> None:
    """`_git_hash` resolves against the working directory, which is the notebook's."""
    from case_studies.utils.registry.store import _git_hash

    monkeypatch.chdir(repo)
    short = _git(repo, "rev-parse", "--short", "HEAD")
    assert _git_hash() == short

    (repo / "model.py").write_text("VERSION = 5\n")
    assert _git_hash() == f"{short}+dirty"
