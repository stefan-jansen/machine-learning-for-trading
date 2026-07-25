"""Tests for the import-resolution guard.

The guard exists because Chapter 21 shipped `03_market_making_ppo.py` with
`from market_making_env import ...` while `market_making_env.py` was never
committed. These tests pin the behaviour that catches that, and the resolution
rules that keep it from crying wolf on legitimate imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_import_resolution import (  # noqa: E402
    imports_with_lines,
    resolves_in_repo,
    resolves_relative,
    third_party_names,
)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A miniature repo: a chapter dir with a notebook, a sibling helper,
    a nested package, and a root-level package."""
    (tmp_path / "07_chapter").mkdir()
    (tmp_path / "07_chapter" / "00_entry.py").write_text("")  # makes it an entry-point dir
    (tmp_path / "07_chapter" / "helper.py").write_text("VALUE = 1\n")
    (tmp_path / "07_chapter" / "pkg").mkdir()
    (tmp_path / "07_chapter" / "pkg" / "__init__.py").write_text("")
    (tmp_path / "07_chapter" / "pkg" / "deep").mkdir()
    (tmp_path / "07_chapter" / "pkg" / "deep" / "mod.py").write_text("")
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils" / "__init__.py").write_text("")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "tool.py").write_text("")
    return tmp_path


def test_missing_sibling_module_does_not_resolve(tree: Path) -> None:
    """The Chapter 21 bug itself."""
    nb = tree / "07_chapter" / "01_notebook.py"
    nb.write_text("from market_making_env import MarketMakingEnv\n")
    assert not resolves_in_repo("market_making_env", nb, tree)


def test_present_sibling_module_resolves(tree: Path) -> None:
    nb = tree / "07_chapter" / "01_notebook.py"
    nb.write_text("from helper import VALUE\n")
    assert resolves_in_repo("helper", nb, tree)


def test_root_level_package_resolves(tree: Path) -> None:
    nb = tree / "07_chapter" / "01_notebook.py"
    nb.write_text("from utils import x\n")
    assert resolves_in_repo("utils", nb, tree)


def test_ancestor_package_resolves_from_nested_module(tree: Path) -> None:
    """`16_strategy_simulation/validation/adapters/*.py` imports `validation`,
    which is on sys.path when the entry point is a notebook in the chapter dir."""
    mod = tree / "07_chapter" / "pkg" / "deep" / "mod.py"
    assert resolves_in_repo("pkg", mod, tree)


def test_explicit_sys_path_insert_is_honored(tree: Path) -> None:
    """Tests that insert `scripts/` on sys.path may import from it."""
    test_file = tree / "test_thing.py"
    test_file.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        'sys.path.insert(0, str(Path(__file__).parent / "scripts"))\n'
        "import tool\n"
    )
    assert resolves_in_repo("tool", test_file, tree)


def test_unrelated_name_still_fails_with_sys_path_insert(tree: Path) -> None:
    """A sys.path.insert must not blanket-approve every unresolved name."""
    test_file = tree / "test_thing.py"
    test_file.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        'sys.path.insert(0, str(Path(__file__).parent / "scripts"))\n'
        "import nonexistent_module\n"
    )
    assert not resolves_in_repo("nonexistent_module", test_file, tree)


def test_imports_are_extracted_with_line_numbers(tmp_path: Path) -> None:
    """Dotted names are kept whole so submodules can be resolved."""
    src = tmp_path / "m.py"
    src.write_text("import os\n\nfrom pkg.sub import thing\nimport a.b as ab\n")
    found = {name: (lineno, level) for name, lineno, level in imports_with_lines(src)}
    assert found == {"os": (1, 0), "pkg.sub": (3, 0), "a.b": (4, 0)}


def test_missing_submodule_of_existing_package_does_not_resolve(tree: Path) -> None:
    """`from utils.missing import x` raises ModuleNotFoundError even though
    `utils` exists, so checking only the top-level component is not enough."""
    nb = tree / "07_chapter" / "01_notebook.py"
    nb.write_text("from utils.missing import thing\n")
    assert resolves_in_repo("utils", nb, tree)
    assert not resolves_in_repo("utils.missing", nb, tree)


def test_namespace_package_submodule_resolves(tree: Path) -> None:
    """PEP 420: `data/equities/` has no __init__.py yet imports fine."""
    (tree / "data").mkdir()
    (tree / "data" / "equities").mkdir()
    (tree / "data" / "equities" / "loader.py").write_text("")
    nb = tree / "07_chapter" / "01_notebook.py"
    nb.write_text("from data.equities.loader import load\n")
    assert resolves_in_repo("data.equities.loader", nb, tree)
    assert not resolves_in_repo("data.equities.absent", nb, tree)


def test_relative_imports_are_captured_with_their_level(tmp_path: Path) -> None:
    """`from .mod import x` is checked; `from . import x` is not, because `x`
    may be a symbol re-exported by the package __init__ rather than a module."""
    src = tmp_path / "m.py"
    src.write_text("from . import sibling\nfrom .mod import thing\nfrom ..up import y\n")
    assert imports_with_lines(src) == [("mod", 2, 1), ("up", 3, 2)]


def test_missing_relative_import_does_not_resolve(tree: Path) -> None:
    """`from .missing import value` names a repo file by definition."""
    mod = tree / "07_chapter" / "pkg" / "thing.py"
    mod.write_text("from .missing import value\n")
    assert not resolves_relative("missing", 1, mod, tree)
    (tree / "07_chapter" / "pkg" / "missing.py").write_text("value = 1\n")
    assert resolves_relative("missing", 1, mod, tree)


def test_parent_relative_import_resolves(tree: Path) -> None:
    """`from ..helper import x` anchors one package higher."""
    mod = tree / "07_chapter" / "pkg" / "thing.py"
    mod.write_text("from ..helper import VALUE\n")
    assert resolves_relative("helper", 2, mod, tree)
    assert not resolves_relative("absent", 2, mod, tree)


def test_distribution_name_is_not_accepted_as_import_name() -> None:
    """Declaring `scikit-learn` permits `import sklearn`, not `import
    scikit_learn` — accepting the latter would be a false negative."""
    names = third_party_names(REPO_ROOT)
    assert "sklearn" in names
    assert "scikit_learn" not in names
    assert "tsfm_public" in names
    assert "granite_tsfm" not in names


def test_non_entry_point_ancestor_is_not_a_resolution_root(tmp_path: Path) -> None:
    """A bare sibling import must not resolve via an intermediate directory
    that nothing is ever run from."""
    (tmp_path / "07_chapter").mkdir()
    (tmp_path / "07_chapter" / "01_notebook.py").write_text("")  # makes it an entry point
    (tmp_path / "07_chapter" / "sub").mkdir()
    (tmp_path / "07_chapter" / "sub" / "deep").mkdir()
    (tmp_path / "07_chapter" / "sub" / "neighbour.py").write_text("")
    (tmp_path / "07_chapter" / "chapter_helper.py").write_text("")
    deep = tmp_path / "07_chapter" / "sub" / "deep" / "mod.py"
    deep.write_text("import neighbour\n")
    # `07_chapter` holds notebooks, so it is a real entry point root.
    assert resolves_in_repo("chapter_helper", deep, tmp_path)
    # `07_chapter/sub` is not run from, so `neighbour` is not importable there.
    assert not resolves_in_repo("neighbour", deep, tmp_path)


def test_declared_dependencies_are_recognized() -> None:
    """Spot-check against the real repo: distributions and aliased import names."""
    names = third_party_names(REPO_ROOT)
    for name in ("polars", "numpy", "sklearn", "bs4", "yaml", "chronos", "tsfm_public"):
        assert name in names, f"{name} should be a recognized third-party import"


def test_repo_is_clean() -> None:
    """The whole repo passes. This is the gate CI runs."""
    from check_import_resolution import main

    assert main() == 0
