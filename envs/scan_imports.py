#!/usr/bin/env python3
"""AST-based import scanner for the ML4T book codebase.

Walks every ``.py`` file in the source tree, extracts the third-party
top-level imports, and classifies them by Docker image (ml4t / py312 /
benchmark / rapids). The scanner is the source of truth that
``test_all_imports.py`` and ``tests/test_import_coverage.py`` use to
detect drift — if a new chapter pulls in a dependency that isn't installed
in the image a reader built, the scanner will flag it.

First-party modules are auto-detected (any ``.py`` stem or ``__init__.py``
package directory in the repo tree is treated as local and excluded).
Stdlib modules are filtered via ``sys.stdlib_module_names``.

Usage
-----

As a library::

    from envs.scan_imports import scan_repo, classify, IMAGE_OVERRIDES

    external = scan_repo()                 # set[str] of 3rd-party import names
    groups = classify(external)            # {image_id: set[str]}

As a CLI (inside a Docker container or locally)::

    python envs/scan_imports.py                    # test ml4t image
    python envs/scan_imports.py --image py312      # only py312 packages
    python envs/scan_imports.py --list             # list, don't import
    python envs/scan_imports.py --verbose          # show all successes

Exit code: 0 if every expected import succeeded, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories to skip entirely (virtualenvs, caches, archives, .git)
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".venv",
        ".git",
        "__pycache__",
        "_archive",
        ".agents",
        ".claude",
        ".ruff_cache",
        ".pytest_cache",
        ".github",
        "build",
        "dist",
    }
)

# Python stdlib — auto-filtered using 3.10+ API
STDLIB: frozenset[str] = frozenset(sys.stdlib_module_names)

# Docker-image classification for packages that are NOT in the default ml4t
# image. Anything not listed here defaults to ``ml4t``.
#
# When adding a new image or dependency, extend this dict. The coverage
# test (tests/test_import_coverage.py) enforces that everything the scanner
# discovers is either importable in the current image or classified out.
IMAGE_OVERRIDES: dict[str, str] = {
    # py312 (Python 3.12 image — packages without 3.14 wheels)
    "signatory": "py312",
    "gensim": "py312",
    "pfhedge": "py312",
    "causalimpact": "py312",  # tfcausalimpact (TFP BSTS); Ch15/06 — caps at py<3.13
    # benchmark (Ch2/Ch3 storage-layer comparison — database clients)
    "arcticdb": "benchmark",
    "clickhouse_connect": "benchmark",
    "duckdb": "benchmark",
    "influxdb_client": "benchmark",
    "psycopg2": "benchmark",
    "pykx": "benchmark",
    "questdb": "benchmark",
    "tables": "benchmark",
    # optional — broker / market-data SDKs readers may skip
    "alpaca": "optional",
    "ib_async": "optional",
    "okx": "optional",
    "databento": "optional",
    "voyageai": "optional",
    # optional — notebooks guard these with try/except ImportError and
    # provide a fallback path. Readers can install on demand.
    "openai": "optional",  # Ch22/02, Ch24/agents — alternative LLM provider
}

VALID_IMAGES: tuple[str, ...] = ("ml4t", "py312", "benchmark", "rapids", "optional")


def _first_party_names(root: Path) -> set[str]:
    """Auto-detect first-party module names by scanning the tree.

    A name is first-party if any ``{name}.py`` file or ``{name}/__init__.py``
    package directory lives inside ``root`` (honoring SKIP_DIRS).
    """
    names: set[str] = set()
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).parts
        if any(p in SKIP_DIRS for p in parts):
            continue
        names.add(path.stem)
        if (path.parent / "__init__.py").exists():
            p = path.parent
            while (p / "__init__.py").exists() and p != root:
                names.add(p.name)
                p = p.parent
    return names


def extract_imports_from_file(path: Path) -> set[str]:
    """Extract every absolute top-level import name from a ``.py`` file via AST.

    Walks the whole tree (including imports inside functions, ``try`` blocks,
    and ``TYPE_CHECKING`` guards) — those still represent real dependencies
    that need to resolve in the target environment.

    Relative imports (``from . import X``) are ignored: they point at
    first-party code.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names.add(node.module.split(".")[0])
    return names


def scan_repo(root: Path | None = None) -> set[str]:
    """Return the set of external (third-party) top-level imports used in the repo.

    External means: not in the Python standard library, not first-party
    code from this repo. First-party is auto-detected from the file tree.
    """
    root = root or REPO_ROOT
    first_party = _first_party_names(root)
    imports: set[str] = set()
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).parts
        if any(p in SKIP_DIRS for p in parts):
            continue
        imports.update(extract_imports_from_file(path))
    return {n for n in imports if n not in STDLIB and n not in first_party}


def classify(imports: set[str]) -> dict[str, set[str]]:
    """Partition ``imports`` by Docker image using ``IMAGE_OVERRIDES``.

    Returns a dict keyed by image id, with every known image pre-populated
    so downstream code can rely on all keys being present.
    """
    groups: dict[str, set[str]] = {image: set() for image in VALID_IMAGES}
    for imp in imports:
        groups[IMAGE_OVERRIDES.get(imp, "ml4t")].add(imp)
    return groups


def try_import(module_name: str) -> tuple[bool, str]:
    """Attempt ``importlib.import_module(module_name)``.

    Treats FileNotFoundError / NotADirectoryError as success — those happen
    when a package imports OK but its configured data directory doesn't
    exist locally, which is not an environment setup failure.
    """
    try:
        importlib.import_module(module_name)
        return True, ""
    except (FileNotFoundError, NotADirectoryError):
        return True, ""
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:  # noqa: BLE001 — report anything that broke the import
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan and verify ML4T book imports by Docker image"
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("ML4T_IMAGE", "ml4t"),
        choices=VALID_IMAGES,
        help="Which image's expected imports to verify (default: $ML4T_IMAGE or 'ml4t')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the scanned imports for the selected image and exit (no import test)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print successful imports too")
    args = parser.parse_args()

    external = scan_repo()
    groups = classify(external)
    expected = sorted(groups[args.image])

    print(f"ML4T import scanner — image: {args.image}")
    print(f"  scanned {len(external)} external imports across the repo")
    print(f"  {len(expected)} classified for the {args.image!r} image")

    if args.list:
        for pkg in expected:
            print(f"  {pkg}")
        return 0

    failures: list[tuple[str, str]] = []
    successes: list[str] = []
    for pkg in expected:
        ok, err = try_import(pkg)
        if ok:
            successes.append(pkg)
        else:
            failures.append((pkg, err))

    if args.verbose:
        print("\nSuccess:")
        for pkg in successes:
            print(f"  ok   {pkg}")

    if failures:
        print("\nFailed imports:")
        for pkg, err in failures:
            print(f"  FAIL {pkg}: {err[:120]}")
        print(f"\n{len(failures)} failure(s) out of {len(expected)} expected imports")
        return 1

    print(f"\nAll {len(expected)} imports succeeded in the {args.image!r} image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
