"""Anchor the repo's chapter imports and data root to the repo, not to ``cwd``.

Python imports ``sitecustomize`` automatically at interpreter startup whenever
it is found on the path. This module is loaded in both supported environments:

* Docker — the repo is bind-mounted at ``/app`` with ``PYTHONPATH=/app``, so
  ``/app/sitecustomize.py`` is on the startup path.
* Local ``uv`` install — declared as a top-level ``py-module`` in
  ``pyproject.toml``, so the editable finder resolves ``sitecustomize`` to this
  file and ``site`` imports it at startup.

Debian and Ubuntu ship their own ``/usr/lib/pythonX.Y/sitecustomize.py``, which
shadows this file whenever the venv is built on the system interpreter — the
stdlib directory precedes the editable finder. ``pyproject.toml`` therefore
pins ``python-preference = "only-managed"`` so ``uv`` downloads its own CPython
rather than reusing the distro's. ``scripts/verify_installation.py`` checks that
this module actually loaded, because nothing else reports that it did not.

Two things are set up here, both anchored to this file's directory so they do
not depend on the working directory:

**Chapter imports.** Chapter directories (e.g. ``25_live_trading``) are
number-prefixed so the repo lists them in reading order, which makes them
invalid Python package names, so their helper modules (``async_utils``,
``limit_orderbook``, ``rl_environments`` …) are importable only when the chapter
directory is on ``sys.path``. Every ``NN_*`` directory is appended (append, not
insert, so nothing shadows stdlib or installed packages). Chapter helper module
names do not collide, so a flat append is unambiguous.

**The data root.** ``resolve_data_root()`` falls back to ``cwd/data`` when
``ML4T_DATA_PATH`` is unset, so the answer depends on where the process started.
Jupyter Lab starts each kernel in the notebook's own chapter directory, which
made every data-loading notebook fail on the local ``uv`` path. Setting the
variable here gives every entry point the same answer: an explicit value in the
environment wins, then ``.env``, then ``<repo>/data``. Reading ``.env`` here is
what makes the two groups of notebooks agree — those that import
``utils.config`` (which calls ``load_dotenv``) and those that do not. When the
value is the ``<repo>/data`` default rather than anyone's choice,
``ML4T_DATA_PATH_IS_DEFAULT`` says so, which is what lets ``tests/conftest.py``
still prefer a populated test-data checkout over the tracked source tree.

Nothing here may raise: this module runs at the startup of every interpreter in
the environment, so a failure would break the whole install rather than one
notebook. Failing means leaving the variable unset, never setting a guess —
``utils.config`` calls ``load_dotenv(override=False)``, so a wrong value set
here is one nothing downstream can correct.
"""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

for _chapter_dir in sorted(_REPO_ROOT.glob("[0-9][0-9]_*")):
    if _chapter_dir.is_dir():
        _entry = str(_chapter_dir)
        if _entry not in sys.path:
            sys.path.append(_entry)


def _data_root_from_dotenv(repo_root: Path) -> str | None:
    """Read ``ML4T_DATA_PATH`` out of ``<repo>/.env``, or report that it cannot.

    Returns the configured value, or ``None`` when the file does not exist or
    does not set the variable. Raises when there is a ``.env`` whose value cannot
    be read the way ``load_dotenv`` would read it — the caller must then leave the
    environment alone, because ``utils.config`` calls ``load_dotenv(override=False)``
    and so cannot correct a wrong value set here.

    ``python-dotenv`` is imported lazily and only when there is a file to parse.
    Importing it unconditionally costs ~15 ms on every interpreter start in the
    environment, which roughly doubles bare startup; hand-rolling the parse
    instead gets ``export KEY=``, inline comments and ``${VAR}`` interpolation
    wrong, and each of those is a value a reader may legitimately write.
    """
    env_file = repo_root / ".env"
    if not env_file.is_file():
        return None
    from dotenv import dotenv_values

    return dotenv_values(env_file).get("ML4T_DATA_PATH") or None


def _has_datasets(data_root: Path) -> bool:
    """True where a data directory holds datasets and not just the tracked skeleton.

    ``data/`` is committed for ``download.py``, each dataset's ``config.yaml`` and the dataset
    cards, while every parquet under it is gitignored. So the directory existing proves nothing,
    and neither does a subdirectory: ``data/etfs/market/`` is tracked and arrives empty. Datasets
    land at ``data/<domain>/*.parquet`` or ``data/<domain>/<source>/*.parquet``; both globs stop
    at the first match and neither descends into the full tree.
    """
    return any(next(data_root.glob(p), None) for p in ("*/*.parquet", "*/*/*.parquet"))


def _main_worktree(repo_root: Path) -> Path | None:
    """The repository's main working tree, or ``None`` if ``repo_root`` is not a linked worktree.

    ``--git-dir`` and ``--git-common-dir`` are equal in the main working tree and differ in a
    linked one, and the common dir's parent IS the main working tree. Both are plumbing commands
    with a stable contract.

    Every failure returns ``None`` rather than propagating, and that is load-bearing: this runs
    before ``ML4T_DATA_PATH`` is assigned, so an escaping error would reach the module-level
    guard and leave the variable unset entirely - worse than the default it was refining. No
    git, no git on PATH, a tarball install and a hung invocation all fall back to the default.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-dir", "--git-common-dir"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001 - startup hook: a missing or hung git is not an error here
        return None
    # One path per line, and splitting on whitespace instead would break on any checkout whose
    # path contains a space: the two paths would arrive as three or more parts, fail the length
    # test, and silently disable the fallback for exactly the reader who cannot tell why.
    parts = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if out.returncode != 0 or len(parts) != 2 or parts[0] == parts[1]:
        return None
    return Path(parts[1]).parent


def _anchor_data_root(repo_root: Path) -> None:
    if os.environ.get("ML4T_DATA_PATH"):
        return
    configured = _data_root_from_dotenv(repo_root)
    # A relative path in .env is relative to the repo, not to the process cwd —
    # otherwise the setting reintroduces the defect it is being used to avoid.
    data_root = Path(configured).expanduser() if configured else repo_root / "data"
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    if configured is None and not _has_datasets(data_root):
        # A reader has one clone, downloads into <repo>/data, and never reaches this branch.
        # A linked `git worktree` does: it is a second working directory of one repository, so
        # it gets the tracked skeleton of data/ and none of the gitignored datasets, which live
        # only in the main working tree. Defaulting to the worktree's own data/ therefore names
        # a directory that exists and holds nothing, and the reader is told to download tens of
        # gigabytes already on the disk.
        #
        # Symlinking the datasets into the worktree is the other obvious fix and is worse: a
        # directory symlink under data/ is not gitignored, so a `git add -A` commits an absolute
        # machine-specific path into the repository.
        main_tree = _main_worktree(repo_root)
        if main_tree is not None and _has_datasets(main_tree / "data"):
            data_root = main_tree / "data"
    os.environ["ML4T_DATA_PATH"] = str(data_root)
    if configured is None:
        # This value is the repo-anchored default, not something anyone asked for.
        # tests/conftest.py needs the difference: the tracked data/ tree is never
        # empty, so an unmarked default would hide the populated test-data
        # checkout and silently skip every data-dependent notebook test.
        os.environ["ML4T_DATA_PATH_IS_DEFAULT"] = "1"


try:
    _anchor_data_root(_REPO_ROOT)
except Exception:  # noqa: BLE001 - startup hook: never break the interpreter
    # Leaving ML4T_DATA_PATH unset is the safe failure: load_dotenv still applies
    # the reader's .env later, which a wrong value set here would have blocked.
    pass
