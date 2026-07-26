#!/usr/bin/env python3
"""Fail if any shipped ``.py`` file imports a module that does not exist.

This is the guard for the Chapter 21 class of bug: ``03_market_making_ppo.py``
shipped to readers with ``from market_making_env import ...`` while
``market_making_env.py`` was never added to the repo. Every reader who opened
that notebook hit ``ModuleNotFoundError`` on the import cell.

The check is deliberately **static** — no imports are executed and nothing needs
to be installed. That is what lets it run in the fast ``guards`` job on every PR
instead of only inside a built Docker image. An import name passes if it is:

1. Python standard library, or
2. resolvable to files in this repo from a directory that is genuinely on
   ``sys.path`` for that file — its own directory, the repo root, an ancestor
   that notebooks are run from, or a path the file itself inserts — with every
   component of a dotted name checked, or
3. a declared third-party dependency (``pyproject.toml`` dependencies, optional
   dependency groups, dependency-groups, and ``uv.lock``, plus
   ``IMAGE_OVERRIDES`` for packages that live only in a non-default Docker
   image), under the name it is actually imported as.

Relative imports are resolved against the importing file's own package, in both
forms: ``from .mod import x`` requires ``mod`` to be a module on disk, and
``from . import x`` accepts either a submodule or a name the package's
``__init__.py`` binds, because both make that statement work.

Anything else is, by elimination, a module the author expected to exist in the
repo and that is not there.

Adding a dependency therefore means declaring it in ``pyproject.toml`` — which
is required anyway. Only when a package's *import* name differs from its
*distribution* name (``bs4`` from ``beautifulsoup4``) does it need an entry, and
that mapping already exists in ``envs/test_all_imports.py``; this guard reuses
it rather than keeping a second copy.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from functools import cache, lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.scan_imports import SKIP_DIRS, STDLIB  # noqa: E402
from envs.test_all_imports import IMPORT_MAP  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# import name -> distribution that provides it, inverted from the repo's single
# canonical mapping in envs/test_all_imports.py.
IMPORT_ALIASES: dict[str, str] = {imp: dist for dist, imp in IMPORT_MAP.items()}


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def declared_dependencies(root: Path) -> set[str]:
    """Every distribution name declared anywhere in ``pyproject.toml``."""
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    specs: list[str] = list(project.get("dependencies", []))
    for group in (project.get("optional-dependencies") or {}).values():
        specs.extend(group)
    for group in (data.get("dependency-groups") or {}).values():
        specs.extend(s for s in group if isinstance(s, str))
    return {_normalize(re.split(r"[<>=!\[;~ ]", s)[0]) for s in specs}


def locked_distributions(root: Path) -> set[str]:
    """Every distribution in ``uv.lock`` — the resolved environment readers get.

    Using the lock file rather than only ``pyproject.toml`` means transitive
    packages that a notebook imports directly (``joblib``, ``networkx``) are
    recognized without hand-maintaining a list.
    """
    lock = root / "uv.lock"
    if not lock.is_file():
        return set()
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    return {_normalize(pkg["name"]) for pkg in data.get("package", []) if "name" in pkg}


def third_party_names(root: Path) -> set[str]:
    """Import names that are legitimately expected to come from site-packages.

    A distribution contributes only the name it is actually imported under.
    Declaring ``scikit-learn`` permits ``import sklearn``, not
    ``import scikit_learn`` -- the latter would fail at runtime, so accepting it
    here would be a false negative.
    """
    from envs.scan_imports import IMAGE_OVERRIDES

    dist_to_import = {_normalize(dist): imp for dist, imp in IMPORT_MAP.items()}
    available = declared_dependencies(root) | locked_distributions(root)
    names = set()
    for dist in available:
        names.add(_normalize(dist_to_import.get(dist, dist)))
    names.update(_normalize(k) for k in IMAGE_OVERRIDES)
    return names


def imports_with_lines(path: Path) -> list[tuple[str, int, int, bool]]:
    """Import names in ``path`` as ``(name, lineno, level, bare)``.

    Dotted names are kept whole. ``from utils.missing import x`` raises
    ``ModuleNotFoundError`` even though ``utils`` exists, so resolution has to
    walk the full path; only dependency *classification* uses the top-level
    component.

    ``level`` is the relative-import depth: 0 for absolute imports, 1 for
    ``from . import x``, 2 for ``from .. import x``. Relative imports name repo
    files by definition, so they are checked rather than skipped.

    ``bare`` marks ``from . import x``, where ``x`` may be either a submodule or
    a name the package's ``__init__.py`` binds. Both are legal, so a bare alias
    resolves against either; ``from .x import y`` requires ``x`` to be a module
    and is checked more strictly.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    found: list[tuple[str, int, int, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno, 0, False))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                if node.module:
                    found.append((node.module, node.lineno, node.level, False))
                else:
                    for alias in node.names:
                        found.append((alias.name, node.lineno, node.level, True))
            elif node.module:
                found.append((node.module, node.lineno, 0, False))
    return found


@cache
def package_init_names(package: Path) -> frozenset[str]:
    """Names a package's ``__init__.py`` binds other than by relative import.

    ``from . import x`` also succeeds when ``x`` is a symbol the ``__init__.py``
    defines rather than a submodule on disk. Bindings that come from a relative
    import are deliberately excluded: those are the statements being checked, so
    counting them would let the guard satisfy itself with the very line in
    question.
    """
    init = package / "__init__.py"
    if not init.is_file():
        return frozenset()
    try:
        tree = ast.parse(init.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names.update(alias.asname or alias.name for alias in node.names)
    return frozenset(names)


def explicit_path_inserts(path: Path, root: Path) -> list[Path]:
    """Directories the file itself puts on ``sys.path``.

    ``tests/test_notebook_sync.py`` does ``sys.path.insert(0, REPO_ROOT / "scripts")``
    before importing ``notebook_provenance``. That is a real resolution root, so
    the guard honors it instead of demanding an allowlist entry.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    roots: list[Path] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in {"insert", "append"}
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "path"
        ):
            continue
        for literal in (s for s in ast.walk(node) if isinstance(s, ast.Constant)):
            if isinstance(literal.value, str) and literal.value:
                roots.append(root / literal.value.lstrip("/"))
    return roots


@lru_cache(maxsize=1)
def entry_point_dirs(root: Path) -> frozenset[Path]:
    """Directories a notebook is actually run from.

    A directory qualifies if it holds a numbered notebook script (``01_x.py``).
    Those are the entry points, so those directories -- and only those -- become
    ``sys.path[0]`` in practice. Treating *every* ancestor as importable would
    approve bare sibling imports that fail at runtime.
    """
    dirs: set[Path] = set()
    for path in root.rglob("[0-9][0-9]_*.py"):
        parts = path.relative_to(root).parts
        if any(p in SKIP_DIRS for p in parts):
            continue
        dirs.add(path.parent)
    return frozenset(dirs)


def is_entry_point(source: Path) -> bool:
    """True if ``source`` is something that gets run directly.

    Three signals, all meaning "python runs this file, so its directory is
    ``sys.path[0]``": a numbered notebook script (``01_x.py``), a test module,
    or a ``__main__`` guard.
    """
    name = source.name
    if (len(name) > 3 and name[:2].isdigit() and name[2] == "_") or name.startswith("test_"):
        return True
    if name == "conftest.py":
        return True
    try:
        return "__main__" in source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False


def _resolution_roots(source: Path, root: Path) -> list[Path]:
    """Directories that are on ``sys.path`` when ``source`` is imported.

    The importing file's own directory is ``sys.path[0]`` when it is the script
    being run; the repo root is the working directory the project documents for
    notebook runs; an ancestor counts only if notebooks live there and can be
    the entry point (this is how ``16_strategy_simulation/validation/adapters/``
    imports ``validation``); plus whatever the file itself inserts.
    """
    entries = entry_point_dirs(root)
    bases: list[Path] = [root]
    # A file's own directory is sys.path[0] only when that file is what gets
    # run. For a module that is merely imported, its directory is not on the
    # path, so admitting it unconditionally would approve bare sibling imports
    # that fail at runtime.
    if source.parent in entries or is_entry_point(source):
        bases.append(source.parent)
    parent = source.parent
    while parent != root and root in parent.parents:
        parent = parent.parent
        if parent in entries:
            bases.append(parent)
    bases.extend(explicit_path_inserts(source, root))
    return bases


def _resolves_under(package: Path, parts: list[str]) -> bool:
    """True if ``parts`` names a module or package under ``package``."""
    for component in parts[:-1]:
        package = package / component
        if not package.is_dir():
            return False
    leaf = parts[-1]
    return (package / f"{leaf}.py").is_file() or (package / leaf).is_dir()


def resolves_relative(name: str, level: int, source: Path, root: Path, bare: bool = False) -> bool:
    """True if a relative import resolves.

    The anchor is the importing file's package, walked up ``level - 1`` times.

    ``from .mod import x`` needs ``mod`` to be a module on disk. ``from . import
    x`` (``bare``) is satisfied by either a submodule or a name the package's
    ``__init__.py`` binds, since both make the statement work at runtime.
    """
    anchor = source.parent
    for _ in range(level - 1):
        anchor = anchor.parent
        if root not in anchor.parents and anchor != root:
            return False
    if _resolves_under(anchor, name.split(".")):
        return True
    return bare and name in package_init_names(anchor)


def resolves_in_repo(name: str, source: Path, root: Path) -> bool:
    """True if the dotted module ``name`` resolves to repo files.

    Every component of a dotted path must exist: ``from utils.missing import x``
    fails at import time even though ``utils`` is a real package, so it must
    fail here too.

    Intermediate components need only be directories. Most of this repo's
    packages are PEP 420 namespace packages -- ``data/equities/`` has no
    ``__init__.py`` yet ``data.equities.loader`` imports correctly -- so
    requiring ``__init__.py`` would reject 53 working imports.
    """
    parts = name.split(".")
    return any(_resolves_under(base, parts) for base in _resolution_roots(source, root))


def iter_source_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if any(p in SKIP_DIRS for p in parts):
            continue
        yield path


def main() -> int:
    root = REPO_ROOT
    allowed = third_party_names(root)
    failures: list[tuple[Path, int, str]] = []

    for path in iter_source_files(root):
        for name, lineno, level, bare in imports_with_lines(path):
            if level > 0:
                if not resolves_relative(name, level, path, root, bare):
                    failures.append((path.relative_to(root), lineno, "." * level + name))
                continue
            # Third-party and stdlib are classified on the top-level component:
            # submodules of an installed distribution cannot be checked without
            # importing it, which this guard deliberately does not do.
            top = name.split(".")[0]
            if top in STDLIB or _normalize(top) in allowed:
                continue
            if resolves_in_repo(name, path, root):
                continue
            failures.append((path.relative_to(root), lineno, name))

    if failures:
        print(f"{len(failures)} unresolvable import(s):\n", file=sys.stderr)
        for rel, lineno, name in failures:
            print(f"  {rel}:{lineno}: cannot resolve {name!r}", file=sys.stderr)
        print(
            "\nEach import above is neither stdlib, nor a file in this repo, nor a\n"
            "declared dependency. Either add the missing module, declare the\n"
            "package in pyproject.toml, or (if its import name differs from its\n"
            "distribution name) add it to IMPORT_ALIASES in this script.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: every import in {sum(1 for _ in iter_source_files(root))} files resolves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
