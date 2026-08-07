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


def _expected_default_root() -> Path:
    """Where the hook lands with nothing configured.

    ``<repo>/data``, except in a linked ``git worktree``, which gets the tracked
    skeleton of ``data/`` and none of the gitignored datasets under it: those live
    only in the main working tree, so the hook borrows that one rather than naming
    a directory that exists and holds nothing.
    """
    sitecustomize = _load_sitecustomize()
    default = REPO_ROOT / "data"
    if sitecustomize._has_datasets(default):
        return default
    main_tree = sitecustomize._main_worktree(REPO_ROOT)
    if main_tree is not None and sitecustomize._has_datasets(main_tree / "data"):
        return main_tree / "data"
    return default


def _expected_root() -> Path:
    """What the hook should resolve to on *this* machine.

    Not unconditionally ``<repo>/data``: a reader who keeps data on another drive
    sets ML4T_DATA_PATH in ``.env``, and honoring that is the whole point of
    reading the file. Asserting the repo path regardless would fail on every
    correctly-configured machine, including this one.

    *Which* of the possible roots the rule picks is pinned in
    ``tests/test_sitecustomize_data_root.py``, against a repository and a linked
    worktree built for the test. What the tests here add is the other half: a
    fresh interpreter started in any directory arrives at that same answer.
    """
    configured = _load_sitecustomize()._data_root_from_dotenv(REPO_ROOT)
    if not configured:
        return _expected_default_root()
    path = Path(configured).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def test_repo_root_resolves_to_the_configured_root(anchor_ran) -> None:
    assert Path(_probe(REPO_ROOT)) == _expected_root()


def test_chapter_directory_resolves_to_the_same_place(anchor_ran) -> None:
    """The defect itself: this used to be <repo>/01_process_is_edge/data."""
    assert Path(_probe(CHAPTER_DIR)) == _expected_root()


def test_every_directory_agrees(anchor_ran) -> None:
    """Whatever the answer is, it must not depend on where the process started."""
    seen = {_probe(d) for d in (REPO_ROOT, CHAPTER_DIR, REPO_ROOT / "tests", Path.home())}
    assert len(seen) == 1, seen


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


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("ML4T_DATA_PATH=/mnt/big/data", "/mnt/big/data"),
        ('ML4T_DATA_PATH="/mnt/big/data"', "/mnt/big/data"),
        ("ML4T_DATA_PATH=/mnt/big/data # external disk", "/mnt/big/data"),
        ("export ML4T_DATA_PATH=/mnt/big/data", "/mnt/big/data"),
        ("ML4T_DATA_PATH=", None),
    ],
    ids=["plain", "quoted", "inline-comment", "export", "empty"],
)
def test_dotenv_parser_matches_load_dotenv(tmp_path, line, expected) -> None:
    """Every one of these is valid .env syntax a reader may write.

    A hand-rolled parser gets the last three wrong, and a wrong value here is one
    nothing downstream can fix: ``utils.config`` calls ``load_dotenv`` with
    ``override=False``, so whatever the hook sets is what the run uses.
    """
    sitecustomize = _load_sitecustomize()

    (tmp_path / ".env").write_text(line + "\n")
    assert sitecustomize._data_root_from_dotenv(tmp_path) == expected


def test_dotenv_parser_interpolates(tmp_path) -> None:
    sitecustomize = _load_sitecustomize()

    (tmp_path / ".env").write_text("ROOT=/mnt/big\nML4T_DATA_PATH=${ROOT}/data\n")
    assert sitecustomize._data_root_from_dotenv(tmp_path) == "/mnt/big/data"


def test_the_default_is_marked_as_a_default(monkeypatch) -> None:
    """tests/conftest.py distinguishes on this.

    The tracked ``data/`` tree is never empty, so an unmarked default would look
    like a configured data root and hide the populated test-data checkout,
    silently skipping every data-dependent notebook test.
    """
    sitecustomize = _load_sitecustomize()

    monkeypatch.setattr(sitecustomize, "_data_root_from_dotenv", lambda _root: None)
    monkeypatch.delenv("ML4T_DATA_PATH", raising=False)
    monkeypatch.delenv("ML4T_DATA_PATH_IS_DEFAULT", raising=False)
    sitecustomize._anchor_data_root(REPO_ROOT)
    assert Path(os.environ["ML4T_DATA_PATH"]) == _expected_default_root()
    assert os.environ["ML4T_DATA_PATH_IS_DEFAULT"] == "1"


def test_a_configured_value_is_not_marked_as_a_default(monkeypatch, tmp_path) -> None:
    sitecustomize = _load_sitecustomize()

    monkeypatch.setattr(sitecustomize, "_data_root_from_dotenv", lambda _root: str(tmp_path))
    monkeypatch.delenv("ML4T_DATA_PATH", raising=False)
    monkeypatch.delenv("ML4T_DATA_PATH_IS_DEFAULT", raising=False)
    sitecustomize._anchor_data_root(REPO_ROOT)
    assert "ML4T_DATA_PATH_IS_DEFAULT" not in os.environ


def test_a_generated_env_omits_the_anchored_default() -> None:
    """Otherwise the marker is defeated one step later.

    conftest writes a .env when a clean clone has none. Writing the marked
    default into it promotes the default to an explicit setting, which the .env
    branch of _resolve_data_path() then returns — the same silent skip the marker
    exists to prevent, one indirection further along.
    """
    from tests.conftest import generated_env_contents

    written = generated_env_contents(
        REPO_ROOT,
        {"ML4T_DATA_PATH": str(REPO_ROOT / "data"), "ML4T_DATA_PATH_IS_DEFAULT": "1"},
    )
    assert "ML4T_DATA_PATH=" not in written
    assert f"ML4T_PATH={REPO_ROOT}" in written


def test_a_generated_env_keeps_a_chosen_data_path() -> None:
    from tests.conftest import generated_env_contents

    written = generated_env_contents(REPO_ROOT, {"ML4T_DATA_PATH": "/mnt/big/data"})
    assert "ML4T_DATA_PATH=/mnt/big/data" in written


def test_conftest_ignores_the_anchored_default(monkeypatch) -> None:
    """The regression the marker exists to prevent, at the site that consumes it."""
    import tests.conftest as conftest

    monkeypatch.setenv("ML4T_DATA_PATH", str(REPO_ROOT / "data"))
    monkeypatch.setenv("ML4T_DATA_PATH_IS_DEFAULT", "1")
    resolved = conftest._resolve_data_path()
    # Whatever it picks, it must not be the source tree taken on the strength of
    # being non-empty. Step 4 may still return it, but only if it holds parquet.
    if resolved == REPO_ROOT / "data":
        assert list((REPO_ROOT / "data").glob("*/*.parquet"))
