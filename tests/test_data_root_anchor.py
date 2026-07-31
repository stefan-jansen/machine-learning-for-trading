"""Guard: the data root is anchored to the repo, not to the working directory.

``resolve_data_root()`` falls back to ``cwd/data`` when ``ML4T_DATA_PATH`` is
unset, so the answer used to depend on where the process started. Jupyter Lab
starts every kernel in the notebook's own chapter directory, which made every
data-loading notebook fail on the local ``uv`` path with a FileNotFoundError
naming a path under the chapter directory.

``sitecustomize.py`` now sets the variable at interpreter startup. These tests
pin the three things that has to get right: an explicit value in the environment
wins, ``.env`` is honored (so the notebooks that import ``utils.config`` and the
ones that do not agree on the answer), and a chapter directory as cwd resolves to
the same place as the repo root.

``tests/test_chapter_imports.py`` covers the other half of the same hook.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER_DIR = REPO_ROOT / "01_process_is_edge"

PROBE = "import os; print(os.environ.get('ML4T_DATA_PATH', ''))"


def _load_sitecustomize():
    """Load *this repo's* sitecustomize.py by path.

    A bare ``import sitecustomize`` returns whichever copy the running
    interpreter resolved at startup, which in a shared-venv worktree is another
    checkout's and on Ubuntu is the distro's. The unit tests below are about this
    file, so they load this file.
    """
    spec = importlib.util.spec_from_file_location(
        "ml4t_sitecustomize_under_test", REPO_ROOT / "sitecustomize.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Executing it runs the anchor for real; put the variable back so loading the
    # module under test cannot change the environment the rest of the suite sees.
    before = os.environ.get("ML4T_DATA_PATH")
    try:
        spec.loader.exec_module(mod)
    finally:
        if before is None:
            os.environ.pop("ML4T_DATA_PATH", None)
        else:
            os.environ["ML4T_DATA_PATH"] = before
    return mod


def _probe(cwd: Path, env_overrides: dict[str, str] | None = None) -> str:
    """Read ML4T_DATA_PATH out of a fresh interpreter started in *cwd*."""
    env = dict(os.environ)
    env.pop("ML4T_DATA_PATH", None)
    env.update(env_overrides or {})
    out = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


@pytest.fixture
def anchor_ran() -> None:
    """Skip when the startup hook is shadowed rather than reporting a false failure.

    On a venv built against a Debian/Ubuntu system interpreter the distro's own
    sitecustomize.py wins and this repo's never runs. That is a real defect, but
    it is the one ``scripts/verify_installation.py`` diagnoses by name; these
    tests are about what the hook does once it runs.
    """
    if not _probe(REPO_ROOT):
        pytest.skip("sitecustomize.py did not run in this environment")


def test_repo_root_resolves_to_repo_data(anchor_ran) -> None:
    assert Path(_probe(REPO_ROOT)) == REPO_ROOT / "data"


def test_chapter_directory_resolves_to_the_same_place(anchor_ran) -> None:
    """The defect itself: this used to be <repo>/01_process_is_edge/data."""
    assert Path(_probe(CHAPTER_DIR)) == REPO_ROOT / "data"


def test_explicit_environment_value_wins(anchor_ran, tmp_path) -> None:
    assert _probe(CHAPTER_DIR, {"ML4T_DATA_PATH": str(tmp_path)}) == str(tmp_path)


def test_dotenv_value_is_honored(tmp_path, monkeypatch) -> None:
    """A reader who follows .env.example must get the same answer everywhere.

    ``utils.config`` calls ``load_dotenv``, so notebooks that import it always
    honored ``.env``; notebooks that do not import it silently ignored the value.
    The startup hook reads the one variable so both groups agree.
    """
    sitecustomize = _load_sitecustomize()

    monkeypatch.setattr(sitecustomize, "_data_root_from_dotenv", lambda _root: str(tmp_path))
    monkeypatch.delenv("ML4T_DATA_PATH", raising=False)
    sitecustomize._anchor_data_root(REPO_ROOT)
    assert os.environ["ML4T_DATA_PATH"] == str(tmp_path)


def test_dotenv_relative_path_is_relative_to_the_repo(monkeypatch) -> None:
    """Otherwise the setting reintroduces the cwd dependence it is used to avoid."""
    sitecustomize = _load_sitecustomize()

    monkeypatch.setattr(sitecustomize, "_data_root_from_dotenv", lambda _root: "elsewhere/data")
    monkeypatch.delenv("ML4T_DATA_PATH", raising=False)
    sitecustomize._anchor_data_root(REPO_ROOT)
    assert Path(os.environ["ML4T_DATA_PATH"]) == REPO_ROOT / "elsewhere" / "data"


def test_dotenv_parser_reads_the_value(tmp_path) -> None:
    sitecustomize = _load_sitecustomize()

    (tmp_path / ".env").write_text(
        "# a comment\nOTHER=1\nML4T_DATA_PATH='/mnt/big/data'\nML4T_PATH=/nope\n"
    )
    assert sitecustomize._data_root_from_dotenv(tmp_path) == "/mnt/big/data"


def test_dotenv_parser_ignores_a_similar_name(tmp_path) -> None:
    sitecustomize = _load_sitecustomize()

    (tmp_path / ".env").write_text("ML4T_DATA_PATH_OLD=/wrong\n")
    assert sitecustomize._data_root_from_dotenv(tmp_path) is None


def test_dotenv_parser_tolerates_a_missing_file(tmp_path) -> None:
    sitecustomize = _load_sitecustomize()

    assert sitecustomize._data_root_from_dotenv(tmp_path) is None
