"""`sitecustomize` decides where every entry point looks for data, so its rules are pinned here.

The rule under test: the default is `<repo>/data`, which is right for a reader with one clone
and wrong for a linked `git worktree`, because a worktree gets the tracked skeleton of `data/`
and none of the gitignored datasets. Those live only in the main working tree.

Both directions matter. Falling back for a reader would silently move their data root; not
falling back in a worktree tells them to download tens of gigabytes already on the disk.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_sitecustomize():
    """Import `sitecustomize` under its own name so the installed one is not shadowed."""
    spec = importlib.util.spec_from_file_location(
        "_sitecustomize_under_test", REPO_ROOT / "sitecustomize.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sc = load_sitecustomize()


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("ML4T_DATA_PATH", raising=False)
    monkeypatch.delenv("ML4T_DATA_PATH_IS_DEFAULT", raising=False)


# --- what counts as a populated data directory ------------------------------------------------


def test_the_tracked_skeleton_is_not_datasets(tmp_path):
    """`data/etfs/market/` is committed and arrives empty, so a directory test is not enough."""
    (tmp_path / "etfs" / "market").mkdir(parents=True)
    (tmp_path / "etfs" / "market" / "config.yaml").write_text("name: etfs\n")
    (tmp_path / "etfs" / "market" / "download.py").write_text("# downloader\n")
    assert not sc._has_datasets(tmp_path)


@pytest.mark.parametrize("rel", ["etfs/market/etf_universe.parquet", "factors/mom_daily.parquet"])
def test_a_parquet_at_either_documented_depth_counts(tmp_path, rel):
    target = tmp_path / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    assert sc._has_datasets(tmp_path)


def test_a_missing_directory_is_not_datasets(tmp_path):
    assert not sc._has_datasets(tmp_path / "nope")


# --- telling a worktree from a clone ----------------------------------------------------------


def test_the_main_working_tree_reports_no_main_worktree():
    """The reader's case. `--git-dir` and `--git-common-dir` are equal, so there is no fallback."""
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    main_tree = Path(common).parent
    assert sc._main_worktree(main_tree) is None


def test_a_linked_worktree_reports_the_tree_that_holds_the_data():
    if sc._main_worktree(REPO_ROOT) is None:
        pytest.skip("this checkout is the main working tree")
    assert (sc._main_worktree(REPO_ROOT) / ".git").exists()


def test_a_directory_outside_any_repository_is_not_a_worktree(tmp_path):
    assert sc._main_worktree(tmp_path) is None


# --- the anchoring rule -----------------------------------------------------------------------


def test_an_explicit_value_is_never_touched(monkeypatch, tmp_path):
    monkeypatch.setenv("ML4T_DATA_PATH", str(tmp_path))
    monkeypatch.delenv("ML4T_DATA_PATH_IS_DEFAULT", raising=False)
    sc._anchor_data_root(REPO_ROOT)
    assert os.environ["ML4T_DATA_PATH"] == str(tmp_path)
    assert "ML4T_DATA_PATH_IS_DEFAULT" not in os.environ


def test_a_populated_repo_data_directory_wins_and_is_marked_default(clean_env, tmp_path):
    """The reader with one clone: no fallback runs, and the default is still marked as one."""
    (tmp_path / "data" / "etfs").mkdir(parents=True)
    (tmp_path / "data" / "etfs" / "bars.parquet").write_bytes(b"")
    sc._anchor_data_root(tmp_path)
    assert os.environ["ML4T_DATA_PATH"] == str(tmp_path / "data")
    assert os.environ["ML4T_DATA_PATH_IS_DEFAULT"] == "1"


def test_an_unpopulated_non_worktree_keeps_the_repo_default(clean_env, tmp_path):
    """No datasets and no main tree to borrow from: the answer is unchanged, and the error
    downstream is the correct 'you have not downloaded the data yet'."""
    (tmp_path / "data" / "etfs" / "market").mkdir(parents=True)
    sc._anchor_data_root(tmp_path)
    assert os.environ["ML4T_DATA_PATH"] == str(tmp_path / "data")


def test_a_linked_worktree_falls_back_to_the_main_working_tree(clean_env):
    main_tree = sc._main_worktree(REPO_ROOT)
    if main_tree is None:
        pytest.skip("this checkout is the main working tree")
    if sc._has_datasets(REPO_ROOT / "data"):
        pytest.skip("this worktree has its own datasets")
    sc._anchor_data_root(REPO_ROOT)
    assert os.environ["ML4T_DATA_PATH"] == str(main_tree / "data")
    assert os.environ["ML4T_DATA_PATH_IS_DEFAULT"] == "1"


def test_a_broken_git_still_anchors_the_variable(clean_env, tmp_path, monkeypatch):
    """The regression the fallback introduced, pinned shut.

    The worktree probe runs BEFORE `ML4T_DATA_PATH` is assigned. If it propagated an error the
    module-level guard would swallow it and the variable would be left unset entirely - worse
    than the `<repo>/data` default it was meant to refine. A tarball install and a container
    with no git both take this path.
    """

    def explode(*_args, **_kwargs):
        raise FileNotFoundError("git: command not found")

    monkeypatch.setattr(sc.subprocess, "run", explode)
    (tmp_path / "data" / "etfs" / "market").mkdir(parents=True)
    sc._anchor_data_root(tmp_path)
    assert os.environ["ML4T_DATA_PATH"] == str(tmp_path / "data")
    assert os.environ["ML4T_DATA_PATH_IS_DEFAULT"] == "1"


def test_a_hung_git_is_not_an_error_either(clean_env, tmp_path, monkeypatch):
    def hang(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=10)

    monkeypatch.setattr(sc.subprocess, "run", hang)
    (tmp_path / "data").mkdir()
    sc._anchor_data_root(tmp_path)
    assert os.environ["ML4T_DATA_PATH"] == str(tmp_path / "data")


def test_the_resolved_root_is_the_one_the_loader_uses():
    """End to end, in whatever tree this runs: the anchored value is what `utils` reports."""
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; from utils import ML4T_DATA_PATH;"
            " print(os.environ['ML4T_DATA_PATH']); print(ML4T_DATA_PATH)",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        pytest.skip(f"utils not importable here: {out.stderr.strip()[-200:]}")
    anchored, resolved = out.stdout.split()
    assert Path(anchored).resolve() == Path(resolved).resolve()
    assert sc._has_datasets(Path(resolved))
