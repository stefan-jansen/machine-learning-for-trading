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

Relative imports are resolved against the importing file's own package, and both
forms require a module on disk: ``from .mod import x`` needs ``mod.py``, and
``from . import x`` needs ``x.py``.

Anything else is, by elimination, a module the author expected to exist in the
repo and that is not there.

**What this guard deliberately does not check: when code runs.** It answers one
question - does a module by this name exist where this file can see it - and says
nothing about import-time ordering. That leaves one residual in each direction,
both accepted:

* **A false positive, reported here and working at runtime.** ``from . import
  NAME`` where ``NAME`` is only bound by the package's ``__init__.py`` (a
  constant, a function, a re-export) is legal Python and is flagged anyway.
  Requiring a module on disk is what closes the bug class: ``deepm/__init__.py``
  does ``from . import configs, dataset, ...``, and once a name an initializer
  binds counts as resolution, deciding which of those nine submodules must exist
  becomes a static model of execution order. No file in this repo uses the
  rejected form; one that did would get a named failure from this script and a
  one-line fix (``from .constants import NAME``).
* **A false negative, passing here and failing at runtime.** A ``sys.path``
  mutation counts wherever it sits in the file, so one placed below its import,
  or inside a function that is never called, is honored. Executing the affected
  import raises ``ModuleNotFoundError`` -- immediately for a module-level import,
  and whenever it is first reached for one deferred inside a function or branch.

Neither residual is silent, which is what separates them from the bug this guard
exists to prevent: the first stops CI with a named file and line, the second
stops whoever runs that import. Modeling Python's execution order statically to
close them costs more false positives than it buys, so the contract stays at
existence.

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
from functools import lru_cache
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

    The two sides are cased differently on purpose. **Distribution** names are
    case-insensitive (PEP 503), so they are normalized for lookup. **Import**
    names are not: ``import NumPy`` raises ``ModuleNotFoundError`` on Linux, so
    the names returned here keep the casing they are imported under and callers
    must match them exactly. Where an import name is *derived* from a
    distribution name it is the normalized form, which is the lowercase spelling
    that nearly every distribution actually uses; the exceptions are what
    ``IMPORT_MAP`` is for, and a package whose import name is capitalized needs an
    entry there just as one whose name differs entirely does.
    """
    from envs.scan_imports import IMAGE_OVERRIDES

    dist_to_import = {_normalize(dist): imp for dist, imp in IMPORT_MAP.items()}
    available = declared_dependencies(root) | locked_distributions(root)
    names = set()
    for dist in available:
        mapped = dist_to_import.get(dist)
        names.add(mapped if mapped is not None else _normalize(dist))
    # IMAGE_OVERRIDES is keyed by import name, so those are taken as written too.
    names.update(IMAGE_OVERRIDES)
    return names


def imports_with_lines(path: Path) -> list[tuple[str, int, int]]:
    """Import names in ``path`` as ``(name, lineno, level)``.

    Dotted names are kept whole. ``from utils.missing import x`` raises
    ``ModuleNotFoundError`` even though ``utils`` exists, so resolution has to
    walk the full path; only dependency *classification* uses the top-level
    component.

    ``level`` is the relative-import depth: 0 for absolute imports, 1 for
    ``from . import x``, 2 for ``from .. import x``. Relative imports name repo
    files by definition, so they are checked rather than skipped.

    ``from . import x`` contributes each imported alias, since ``x`` is the
    module name being asked for. ``lineno`` is carried for the error message.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    found: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno, 0))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                if node.module:
                    found.append((node.module, node.lineno, node.level))
                else:
                    for alias in node.names:
                        found.append((alias.name, node.lineno, node.level))
            elif node.module:
                found.append((node.module, node.lineno, 0))
    return found


#: Fields the grammar uses to hold a single bound identifier. Checking fields
#: rather than node types means a binding form that is not enumerated anywhere --
#: a match-statement capture, a type parameter, something a future release adds --
#: is still caught: ``name`` covers ``FunctionDef``/``ClassDef``/``ExceptHandler``/
#: ``MatchAs``/``MatchStar``/``TypeVar``, ``arg`` covers parameters, and ``rest``
#: covers ``MatchMapping``.
_BINDING_FIELDS = ("name", "arg", "rest")


def _binds_the_name(node: ast.AST, name: str) -> bool:
    """True if ``node`` binds ``name`` through an identifier field or a
    ``global``/``nonlocal`` declaration."""
    if any(getattr(node, field, None) == name for field in _BINDING_FIELDS):
        return True
    declared = getattr(node, "names", None)
    return isinstance(declared, list) and name in declared


def _sys_is_the_module(tree: ast.Module) -> bool:
    """True if the name ``sys`` in this file is the standard-library module.

    The file must carry a plain ``import sys`` **at module level** -- a nested
    one binds only inside its own function, so a module-level
    ``sys.path.append(...)`` beside it raises ``NameError`` and never touches the
    path -- and the name must never be bound any other way anywhere in the file.
    An alias (``import config as sys``), a reassignment, a parameter, a ``for``
    target, a ``case`` capture: any of them make ``sys.path.append("scripts")``
    an unrelated call that would otherwise approve a broken import. A wildcard
    import disqualifies the file too, since what it binds cannot be read here.

    Whole-file rather than per-scope on purpose. Which binding is live at a given
    call is a scope-and-order question, and this guard does not answer those; a
    file that both imports ``sys`` and rebinds the name anywhere simply gets no
    credit for its mutations, which costs a false positive at worst.
    """
    if not any(
        isinstance(node, ast.Import)
        and any(alias.name == "sys" and alias.asname is None for alias in node.names)
        for node in tree.body
    ):
        return False
    for node in ast.walk(tree):
        # Two nodes carry a name in a binding field without binding anything.
        # `alias` holds the imported name, which the Import/ImportFrom branches
        # below inspect instead; `keyword` holds a keyword-argument name, so
        # `configure(sys=sys)` passes the module rather than rebinding it.
        if isinstance(node, ast.alias | ast.keyword):
            continue
        if isinstance(node, ast.Import):
            if any(alias.asname == "sys" for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                return False
            if any((alias.asname or alias.name) == "sys" for alias in node.names):
                return False
        elif isinstance(node, ast.Name):
            if node.id == "sys" and isinstance(node.ctx, ast.Store | ast.Del):
                return False
        elif _binds_the_name(node, "sys"):
            return False
    return True


def explicit_path_inserts(path: Path, root: Path) -> list[Path]:
    """Directories the file itself puts on ``sys.path``.

    ``tests/test_notebook_sync.py`` does ``sys.path.insert(0, REPO_ROOT / "scripts")``
    before importing ``notebook_provenance``. That is a real resolution root, so
    the guard honors it instead of demanding an allowlist entry.

    Only a directory inside the repo is returned. An absolute entry keeps its own
    semantics rather than being reinterpreted relative to the repo, and one that
    lands outside cannot hold a module this repo ships, so it grants nothing.

    The receiver has to be the standard-library ``sys.path``. Matching any
    ``x.path.append()`` would let an unrelated call turn a broken import into a
    resolved one, which is a false negative in the bug class this guard exists
    for, so both the attribute spelling and the binding of the name are checked
    (see :func:`_sys_is_the_module`). Every mutation in this repo is a plain
    ``sys.path.insert`` in a file that imports ``sys`` unaliased.

    Position and scope are not consulted: a mutation anywhere in the file counts
    for every import in it. That admits the file whose insert sits below its
    import or inside a function nobody calls. Deciding it statically means
    modeling execution order -- which path reaches which import -- and that
    rejects legitimate shapes (``if condition: sys.path.insert(...); import
    tool``, an import deferred inside a function defined above the insert) to
    catch a failure that raises as soon as the affected import runs.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    if not _sys_is_the_module(tree):
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
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "sys"
        ):
            continue
        for literal in (s for s in ast.walk(node) if isinstance(s, ast.Constant)):
            if isinstance(literal.value, str) and literal.value:
                # An absolute entry stays absolute. Rewriting "/scripts" to
                # <repo>/scripts claimed a directory Python never searches, and
                # a root outside the repo cannot hold a module this repo ships,
                # so it is no resolution root here either way.
                inserted = Path(literal.value)
                inserted = inserted if inserted.is_absolute() else root / inserted
                if inserted == root or root in inserted.parents:
                    roots.append(inserted)
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
    imports ``validation``); plus whatever the file inserts itself.
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


def resolves_relative(name: str, level: int, source: Path, root: Path) -> bool:
    """True if a relative import resolves to a module on disk.

    The anchor is the importing file's package, walked up ``level - 1`` times.
    Both forms require a module: ``from .mod import x`` needs ``mod.py``, and
    ``from . import x`` needs ``x.py``. The second is the form that let the
    Chapter 21 bug class recur, and demanding a file is what closes it -- see the
    module docstring for the legal shape this rejects and why that is accepted.
    """
    anchor = source.parent
    for _ in range(level - 1):
        anchor = anchor.parent
        if root not in anchor.parents and anchor != root:
            return False
    return _resolves_under(anchor, name.split("."))


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
        for name, lineno, level in imports_with_lines(path):
            if level > 0:
                if not resolves_relative(name, level, path, root):
                    failures.append((path.relative_to(root), lineno, "." * level + name))
                continue
            # Third-party and stdlib are classified on the top-level component:
            # submodules of an installed distribution cannot be checked without
            # importing it, which this guard deliberately does not do.
            # Matched exactly: imports are case-sensitive, so `import NumPy` is a
            # runtime failure even though the distribution is spelled any way.
            top = name.split(".")[0]
            if top in STDLIB or top in allowed:
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
