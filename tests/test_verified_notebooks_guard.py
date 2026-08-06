"""The gate that stops a batch rewrite from silently voiding a verification.

``.github/scripts/check_verified_notebooks.py`` fails a commit that changes a ``.py``
listed in ``.verified-notebooks.tsv`` unless the same commit drops or updates its row.
Four batch rewrites in the week to 2026-08-06 erased roughly thirty verifications
without anything noticing, PR #478 rewriting 34 notebooks in one commit.

The gate is only worth having if it actually fires, so each test below builds a throwaway
git repository, stages a real change, and runs the script the way pre-commit runs it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "check_verified_notebooks.py"
MANIFEST_NAME = ".verified-notebooks.tsv"
HEADER = "path\tpy_blob\tverified_at\tverifier\n"
NOTEBOOK = "case_studies/demo/01_feasibility_analysis.py"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


def _run_gate(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the gate as pre-commit does: from the repo root, no filenames passed."""
    return subprocess.run(
        [sys.executable, str(repo / ".github" / "scripts" / SCRIPT.name), *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repo holding one committed notebook and an empty manifest."""
    repo = tmp_path / "repo"
    (repo / ".github" / "scripts").mkdir(parents=True)
    (repo / "case_studies" / "demo").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / ".github" / "scripts" / SCRIPT.name)

    (repo / NOTEBOOK).write_text("# %%\nprint('verified')\n")
    (repo / MANIFEST_NAME).write_text(HEADER)

    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial")
    return repo


def _verify(repo: Path, path: str = NOTEBOOK) -> str:
    """Record ``path`` as verified at its committed blob. Returns the blob hash."""
    blob = _git(repo, "rev-parse", f"HEAD:{path}").stdout.strip()
    manifest = repo / MANIFEST_NAME
    manifest.write_text(manifest.read_text() + f"{path}\t{blob}\t2026-08-06\ttester\n")
    _git(repo, "add", MANIFEST_NAME)
    return blob


def test_changing_a_verified_notebook_is_rejected(repo: Path) -> None:
    """The case PR #478 was: the .py moves, the row claiming it was verified does not."""
    _verify(repo)
    (repo / NOTEBOOK).write_text("# %%\nprint('rewritten by a batch pass')\n")
    _git(repo, "add", NOTEBOOK)

    result = _run_gate(repo)

    assert result.returncode == 1, result.stdout
    assert NOTEBOOK in result.stdout
    assert "verified" in result.stdout.lower()


def test_dropping_the_row_in_the_same_commit_is_allowed(repo: Path) -> None:
    """Discarding a verification stays possible; it has to appear in the diff."""
    _verify(repo)
    (repo / NOTEBOOK).write_text("# %%\nprint('rewritten, and the row goes with it')\n")
    manifest = repo / MANIFEST_NAME
    manifest.write_text(HEADER)
    _git(repo, "add", NOTEBOOK, MANIFEST_NAME)

    assert _run_gate(repo).returncode == 0


def test_updating_the_row_to_the_new_blob_is_allowed(repo: Path) -> None:
    """A re-verification records the new blob, which is consent for that exact version."""
    _verify(repo)
    (repo / NOTEBOOK).write_text("# %%\nprint('re-verified at this version')\n")
    _git(repo, "add", NOTEBOOK)
    new_blob = _git(repo, "ls-files", "--stage", "--", NOTEBOOK).stdout.split()[1]
    (repo / MANIFEST_NAME).write_text(HEADER + f"{NOTEBOOK}\t{new_blob}\t2026-08-07\ttester\n")
    _git(repo, "add", MANIFEST_NAME)

    assert _run_gate(repo).returncode == 0


def test_deleting_a_verified_notebook_is_rejected(repo: Path) -> None:
    """Removing the file voids the verdict as surely as rewriting it."""
    _verify(repo)
    _git(repo, "rm", "-q", NOTEBOOK)

    result = _run_gate(repo)

    assert result.returncode == 1, result.stdout
    assert NOTEBOOK in result.stdout
    assert "removes it" in result.stdout


def test_renaming_a_verified_notebook_is_rejected(repo: Path) -> None:
    """A rename leaves the row pointing at nothing and the content somewhere unvouched for."""
    _verify(repo)
    _git(repo, "mv", NOTEBOOK, "case_studies/demo/01_renamed.py")

    result = _run_gate(repo)

    assert result.returncode == 1, result.stdout
    assert NOTEBOOK in result.stdout


def test_moving_the_row_with_the_rename_is_allowed(repo: Path) -> None:
    """Consent for a rename is moving the row, which puts the decision in the diff."""
    _verify(repo)
    renamed = "case_studies/demo/01_renamed.py"
    _git(repo, "mv", NOTEBOOK, renamed)
    blob = _git(repo, "ls-files", "--stage", "--", renamed).stdout.split()[1]
    (repo / MANIFEST_NAME).write_text(HEADER + f"{renamed}\t{blob}\t2026-08-06\ttester\n")
    _git(repo, "add", MANIFEST_NAME)

    assert _run_gate(repo).returncode == 0


def test_audit_reports_a_verified_notebook_that_is_gone(repo: Path) -> None:
    """--audit must not read a missing file as an unchanged one."""
    _verify(repo)
    _git(repo, "commit", "-qm", "record the verification")
    (repo / NOTEBOOK).unlink()

    result = _run_gate(repo, "--audit")

    assert result.returncode == 1, result.stdout
    assert "GONE" in result.stdout
    assert NOTEBOOK in result.stdout


def test_an_unlisted_notebook_is_untouched(repo: Path) -> None:
    """Nothing is verified yet, so the gate must not stand in the way of ordinary work."""
    (repo / NOTEBOOK).write_text("# %%\nprint('nobody has verified this')\n")
    _git(repo, "add", NOTEBOOK)

    assert _run_gate(repo).returncode == 0


def test_audit_reports_a_verified_notebook_the_worktree_has_moved(repo: Path) -> None:
    """--audit answers "does the manifest still describe the tree", staged or not."""
    _verify(repo)
    _git(repo, "commit", "-qm", "record the verification")
    (repo / NOTEBOOK).write_text("# %%\nprint('edited, never staged')\n")

    result = _run_gate(repo, "--audit")

    assert result.returncode == 1, result.stdout
    assert NOTEBOOK in result.stdout

    (repo / NOTEBOOK).write_text("# %%\nprint('verified')\n")
    assert _run_gate(repo, "--audit").returncode == 0


def test_the_installed_hook_names_the_installed_script() -> None:
    """The gate is only in force if .pre-commit-config.yaml actually calls it."""
    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text()

    assert "id: verified-notebooks" in config
    assert ".github/scripts/check_verified_notebooks.py" in config
    assert SCRIPT.exists()
    assert (REPO_ROOT / MANIFEST_NAME).exists()
