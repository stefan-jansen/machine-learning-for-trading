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
    _init_body,
    imports_with_lines,
    package_init_names,
    resolves_in_repo,
    resolves_relative,
    third_party_names,
)


def _clear_init_caches() -> None:
    """Both initializer caches key on the package path, which these tests reuse
    while rewriting the file underneath it."""
    package_init_names.cache_clear()
    _init_body.cache_clear()


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
    assert resolves_in_repo("tool", test_file, tree, lineno=4)


def test_unrelated_name_still_fails_with_sys_path_insert(tree: Path) -> None:
    """A sys.path.insert must not blanket-approve every unresolved name."""
    test_file = tree / "test_thing.py"
    test_file.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        'sys.path.insert(0, str(Path(__file__).parent / "scripts"))\n'
        "import nonexistent_module\n"
    )
    assert not resolves_in_repo("nonexistent_module", test_file, tree, lineno=4)


def test_sys_path_insert_below_the_import_does_not_help_it(tree: Path) -> None:
    """The insert runs after the import, so the import still fails at runtime."""
    test_file = tree / "test_thing.py"
    test_file.write_text('import sys\nimport tool\nsys.path.insert(0, "scripts")\n')
    assert not resolves_in_repo("tool", test_file, tree, lineno=2)
    # Above it, the same insert is in force.
    test_file.write_text('import sys\nsys.path.insert(0, "scripts")\nimport tool\n')
    assert resolves_in_repo("tool", test_file, tree, lineno=3)


def test_sys_path_insert_inside_a_function_does_not_count(tree: Path) -> None:
    """A mutation in a function body, a branch, or a `try` may never run, so it
    cannot be treated as putting a directory on sys.path."""
    test_file = tree / "test_thing.py"
    test_file.write_text(
        'import sys\n\n\ndef setup():\n    sys.path.insert(0, "scripts")\n\n\nimport tool\n'
    )
    assert not resolves_in_repo("tool", test_file, tree, lineno=8)


def test_imports_are_extracted_with_line_numbers(tmp_path: Path) -> None:
    """Dotted names are kept whole so submodules can be resolved."""
    src = tmp_path / "m.py"
    src.write_text("import os\n\nfrom pkg.sub import thing\nimport a.b as ab\n")
    found = {name: (lineno, level) for name, lineno, level, _ in imports_with_lines(src)}
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
    """Both forms are captured. `from . import x` is flagged bare, because `x`
    may be a submodule or a name the package __init__ binds."""
    src = tmp_path / "m.py"
    src.write_text("from . import sibling\nfrom .mod import thing\nfrom ..up import y\n")
    assert imports_with_lines(src) == [
        ("sibling", 1, 1, True),
        ("mod", 2, 1, False),
        ("up", 3, 2, False),
    ]


def test_bare_relative_import_of_missing_submodule_does_not_resolve(tree: Path) -> None:
    """`from . import missing` is the form that let the Chapter 21 class of bug
    recur: it was skipped entirely, so a deleted sibling passed the guard."""
    pkg = tree / "07_chapter" / "pkg"
    mod = pkg / "thing.py"
    mod.write_text("from . import missing\n")
    assert not resolves_relative("missing", 1, mod, tree, bare=True)
    (pkg / "missing.py").write_text("value = 1\n")
    assert resolves_relative("missing", 1, mod, tree, bare=True)


def test_bare_relative_import_accepts_a_name_the_init_binds(tree: Path) -> None:
    """`from . import CONSTANT` is legal when the package __init__ defines it,
    so requiring a module on disk would reject working code."""
    pkg = tree / "07_chapter" / "pkg"
    (pkg / "__init__.py").write_text("CONSTANT = 3\n\n\ndef helper():\n    pass\n")
    mod = pkg / "thing.py"
    mod.write_text("from . import CONSTANT, helper\n")
    assert resolves_relative("CONSTANT", 1, mod, tree, bare=True)
    assert resolves_relative("helper", 1, mod, tree, bare=True)
    assert not resolves_relative("absent", 1, mod, tree, bare=True)
    # The strict form still demands a module: a symbol named CONSTANT does not
    # make `from .CONSTANT import x` importable.
    assert not resolves_relative("CONSTANT", 1, mod, tree)


def test_init_relative_imports_do_not_satisfy_themselves(tree: Path) -> None:
    """`deepm/__init__.py` does `from . import configs, ...`. Counting names an
    __init__ binds *by relative import* would let that line prove itself, so
    those bindings are excluded and the submodules must actually exist."""
    pkg = tree / "07_chapter" / "pkg"
    init = pkg / "__init__.py"
    init.write_text("from . import configs\n")
    assert not resolves_relative("configs", 1, init, tree, bare=True)
    (pkg / "configs.py").write_text("")
    _clear_init_caches()
    assert resolves_relative("configs", 1, init, tree, bare=True)


def test_init_binding_below_the_import_does_not_satisfy_it(tree: Path) -> None:
    """`from . import missing` followed by `missing = 1` raises at runtime: the
    import runs first. Ignoring statement order left a variant of the original
    hole open, since any later binding of the name looked like a definition."""
    pkg = tree / "07_chapter" / "pkg"
    init = pkg / "__init__.py"
    init.write_text("from . import late\n\nlate = 1\n")
    _clear_init_caches()
    assert not resolves_relative("late", 1, init, tree, bare=True, lineno=1)
    # Above the import the same binding is real, so the statement does work.
    init.write_text("early = 1\n\nfrom . import early\n")
    _clear_init_caches()
    assert resolves_relative("early", 1, init, tree, bare=True, lineno=3)


def test_same_line_binding_above_the_import_is_visible(tree: Path) -> None:
    """`early = 1; from . import early` is legal and puts both statements on one
    line, so ordering has to compare statement position, not line number."""
    pkg = tree / "07_chapter" / "pkg"
    init = pkg / "__init__.py"
    init.write_text("early = 1; from . import early\n")
    _clear_init_caches()
    assert resolves_relative("early", 1, init, tree, bare=True, lineno=1)
    # Reversed on the same line, the import runs first and fails.
    init.write_text("from . import late; late = 1\n")
    _clear_init_caches()
    assert not resolves_relative("late", 1, init, tree, bare=True, lineno=1)


def test_sibling_the_init_does_not_import_sees_every_binding(tree: Path) -> None:
    """A submodule the initializer never imports is loaded only after it has run
    to completion, so a binding anywhere in it is in place."""
    pkg = tree / "07_chapter" / "pkg"
    (pkg / "__init__.py").write_text("from .helper import setup\n\nLIMIT = 5\n")
    (pkg / "helper.py").write_text("def setup():\n    pass\n")
    mod = pkg / "thing.py"
    mod.write_text("from . import LIMIT\n")
    _clear_init_caches()
    assert resolves_relative("LIMIT", 1, mod, tree, bare=True, lineno=1)


def test_sibling_imported_by_the_init_sees_only_what_ran_before_it(tree: Path) -> None:
    """Circular initialization: the __init__ imports `thing` before defining
    `LIMIT`, and `thing` does `from . import LIMIT`. That raises at runtime, so
    treating every sibling as fully initialized was a false negative."""
    pkg = tree / "07_chapter" / "pkg"
    mod = pkg / "thing.py"
    mod.write_text("from . import LIMIT\n")

    (pkg / "__init__.py").write_text("from . import thing\n\nLIMIT = 5\n")
    _clear_init_caches()
    assert not resolves_relative("LIMIT", 1, mod, tree, bare=True, lineno=1)

    # Defined before the import, the same binding is there when `thing` loads.
    (pkg / "__init__.py").write_text("LIMIT = 5\n\nfrom . import thing\n")
    _clear_init_caches()
    assert resolves_relative("LIMIT", 1, mod, tree, bare=True, lineno=1)

    # The explicit form imports the sibling just the same.
    (pkg / "__init__.py").write_text("from .thing import anything\n\nLIMIT = 5\n")
    _clear_init_caches()
    assert not resolves_relative("LIMIT", 1, mod, tree, bare=True, lineno=1)


def test_bare_relative_import_accepts_an_init_re_export(tree: Path) -> None:
    """`from .constants import VALUE` in the __init__ makes `from . import VALUE`
    work in a sibling. Excluding *every* relative binding rejected that working
    code; only the bare form has to be excluded, and `constants` is checked on
    its own line anyway."""
    pkg = tree / "07_chapter" / "pkg"
    (pkg / "constants.py").write_text("VALUE = 1\n")
    (pkg / "__init__.py").write_text("from .constants import VALUE\n")
    mod = pkg / "thing.py"
    mod.write_text("from . import VALUE\n")
    _clear_init_caches()
    assert resolves_relative("VALUE", 1, mod, tree, bare=True, lineno=1)
    # A re-export from a module that does not exist buys nothing: the __init__'s
    # own `from .absent import ...` line is what fails.
    (pkg / "__init__.py").write_text("from .absent import OTHER\n")
    _clear_init_caches()
    assert resolves_relative("OTHER", 1, mod, tree, bare=True, lineno=1)
    init = pkg / "__init__.py"
    assert not resolves_relative("absent", 1, init, tree, lineno=1)


def test_annotation_without_a_value_is_not_a_binding(tree: Path) -> None:
    """`NAME: int` only records an annotation; nothing named NAME exists, so
    `from . import NAME` fails at runtime unless a NAME submodule is there."""
    pkg = tree / "07_chapter" / "pkg"
    (pkg / "__init__.py").write_text("ANNOTATED: int\n")
    mod = pkg / "thing.py"
    mod.write_text("from . import ANNOTATED\n")
    _clear_init_caches()
    assert not resolves_relative("ANNOTATED", 1, mod, tree, bare=True, lineno=1)
    # With a value it does bind.
    (pkg / "__init__.py").write_text("ANNOTATED: int = 1\n")
    _clear_init_caches()
    assert resolves_relative("ANNOTATED", 1, mod, tree, bare=True, lineno=1)


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
