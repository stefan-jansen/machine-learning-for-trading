"""Tests for the import-resolution guard.

The guard exists because Chapter 21 shipped `03_market_making_ppo.py` with
`from market_making_env import ...` while `market_making_env.py` was never
committed. These tests pin the behaviour that catches that, and the resolution
rules that keep it from crying wolf on legitimate imports.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))

from check_import_resolution import (  # noqa: E402
    _path_builtins_are_readable,
    imports_with_lines,
    iter_source_files,
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


def test_an_unrelated_path_append_is_not_a_sys_path_mutation(tree: Path) -> None:
    """Matching any `x.path.append()` would let an unrelated call approve a broken
    import - a false negative in the exact bug class this guard covers."""
    test_file = tree / "test_thing.py"
    test_file.write_text('import config\nconfig.path.append("scripts")\nimport tool\n')
    assert not resolves_in_repo("tool", test_file, tree)
    # Spelled on sys, the same call is a real resolution root.
    test_file.write_text('import sys\nsys.path.append("scripts")\nimport tool\n')
    assert resolves_in_repo("tool", test_file, tree)


@pytest.mark.parametrize(
    "source",
    [
        'import config as sys\nsys.path.append("scripts")\nimport tool\n',
        'import sys\nsys = object()\nsys.path.append("scripts")\nimport tool\n',
        'from config import loader as sys\nsys.path.append("scripts")\nimport tool\n',
        'def f(sys):\n    sys.path.append("scripts")\n\n\nimport tool\n',
        'for sys in items:\n    sys.path.append("scripts")\n\nimport tool\n',
        'import sys\ntry:\n    pass\nexcept ValueError as sys:\n    pass\nsys.path.append("s")\nimport tool\n',
        # A `case` capture binds outside ast.Name, so type enumeration missed it.
        'import sys\nmatch v:\n    case sys:\n        pass\nsys.path.append("scripts")\nimport tool\n',
        # A wildcard import may export its own `sys`; what it binds is unreadable.
        'import sys\nfrom config import *\nsys.path.append("scripts")\nimport tool\n',
        # Nested: the import binds inside `f` only, so the module-level call raises
        # NameError and never touches the path.
        'def f():\n    import sys\n\n\nsys.path.append("scripts")\nimport tool\n',
        # A `global` declaration is not a binding, but the assignment under it is.
        'import sys\nsys.path.append("scripts")\n\n\ndef clobber():\n'
        "    global sys\n    sys = None\n\n\nimport tool\n",
        # No `import sys` at all: whatever `sys` is here, it is not the module.
        'sys.path.append("scripts")\nimport tool\n',
    ],
)
def test_a_sys_that_is_not_the_module_grants_nothing(tree: Path, source: str) -> None:
    """The name has to be the stdlib module bound at module level, not just spelled
    `sys`. An alias, a reassignment, a parameter, a loop or `case` target, or an
    import nested in a function makes `sys.path.append("scripts")` an unrelated or
    failing call, and honoring it would approve a broken import."""
    test_file = tree / "test_thing.py"
    test_file.write_text(source)
    assert not resolves_in_repo("tool", test_file, tree)


@pytest.mark.parametrize(
    "source",
    [
        # `keyword.arg` holds a keyword-argument name in the same field a binding
        # uses, so reading fields without excluding it disqualified this file.
        'import sys\nsys.path.insert(0, "scripts")\nconfigure(sys=sys)\nimport tool\n',
        # A local named `sys_path`, an attribute named `sys`, a string "sys": none
        # of them bind the name.
        'import sys\nsys.path.insert(0, "scripts")\nsys_path = cfg.sys\nlabel = "sys"\nimport tool\n',
        # `global sys` only names the scope an assignment would write to. Reading
        # sys.path through it binds nothing.
        'import sys\nsys.path.insert(0, "scripts")\n\n\ndef show():\n'
        "    global sys\n    print(sys.path)\n\n\nimport tool\n",
    ],
)
def test_using_the_name_without_binding_it_keeps_the_credit(tree: Path, source: str) -> None:
    """Passing the module on, or naming something near it, is not a rebinding. A
    check that read these as bindings would reject the file's real path insert and
    fail a working import."""
    test_file = tree / "test_thing.py"
    test_file.write_text(source)
    assert resolves_in_repo("tool", test_file, tree)


def test_a_guarded_or_deferred_sys_path_insert_still_counts(tree: Path) -> None:
    """Position and scope are not modeled, so an insert inside a conditional or a
    function counts for the imports in the file. The guard asks whether the module
    exists, not whether the insert has run: `tool` is committed either way, and an
    insert that truly never runs raises a named ImportError on first use."""
    guarded = tree / "test_guarded.py"
    guarded.write_text(
        'import sys\n\nif True:\n    sys.path.insert(0, "scripts")\n    import tool\n'
    )
    assert resolves_in_repo("tool", guarded, tree)
    deferred = tree / "test_deferred.py"
    deferred.write_text(
        'import sys\n\n\ndef later():\n    import tool\n\n\nsys.path.insert(0, "scripts")\n'
    )
    assert resolves_in_repo("tool", deferred, tree)


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
    """Both forms are captured, with the depth that anchors them."""
    src = tmp_path / "m.py"
    src.write_text("from . import sibling\nfrom .mod import thing\nfrom ..up import y\n")
    assert imports_with_lines(src) == [
        ("sibling", 1, 1),
        ("mod", 2, 1),
        ("up", 3, 2),
    ]


def test_bare_relative_import_of_missing_submodule_does_not_resolve(tree: Path) -> None:
    """`from . import missing` is the form that let the Chapter 21 class of bug
    recur: it was skipped entirely, so a deleted sibling passed the guard."""
    pkg = tree / "07_chapter" / "pkg"
    mod = pkg / "thing.py"
    mod.write_text("from . import missing\n")
    assert not resolves_relative("missing", 1, mod, tree)
    (pkg / "missing.py").write_text("value = 1\n")
    assert resolves_relative("missing", 1, mod, tree)


def test_bare_relative_import_demands_a_module_not_an_init_binding(tree: Path) -> None:
    """The narrowed contract, and the one legal shape it rejects.

    `from . import CONSTANT` works at runtime when the package __init__ binds
    CONSTANT, and the guard reports it anyway. Accepting initializer bindings is
    what reopens the bug class this guard exists for: `deepm/__init__.py` does
    `from . import configs, dataset, ...`, and once a name an __init__ binds
    counts as resolution, deciding which of those nine submodules must exist on
    disk turns into a static model of Python's execution order. No file in this
    repo relies on the rejected form; one that did would get a named error and a
    one-line fix (`from .constants import CONSTANT`).
    """
    pkg = tree / "07_chapter" / "pkg"
    (pkg / "__init__.py").write_text("CONSTANT = 3\n\n\ndef helper():\n    pass\n")
    mod = pkg / "thing.py"
    mod.write_text("from . import CONSTANT, helper\n")
    assert not resolves_relative("CONSTANT", 1, mod, tree)
    assert not resolves_relative("helper", 1, mod, tree)
    # A submodule of that name is what resolves.
    (pkg / "CONSTANT.py").write_text("")
    assert resolves_relative("CONSTANT", 1, mod, tree)


def test_init_relative_imports_do_not_satisfy_themselves(tree: Path) -> None:
    """`deepm/__init__.py` does `from . import configs, ...` - the exact shape of
    the shipped bug. The initializer naming the submodule proves nothing; the
    file has to be there."""
    pkg = tree / "07_chapter" / "pkg"
    init = pkg / "__init__.py"
    init.write_text("from . import configs\n")
    assert not resolves_relative("configs", 1, init, tree)
    (pkg / "configs.py").write_text("")
    assert resolves_relative("configs", 1, init, tree)


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


def test_import_names_are_case_sensitive() -> None:
    """`import NumPy` raises ModuleNotFoundError on Linux, so the accepted names
    carry the casing they are imported under. Distribution names stay
    case-insensitive per PEP 503, which is why `ipython` maps to `IPython` rather
    than being accepted in either spelling."""
    names = third_party_names(REPO_ROOT)
    assert "numpy" in names
    assert "NumPy" not in names
    assert "IPython" in names
    assert "ipython" not in names


def test_an_absolute_sys_path_entry_is_not_reinterpreted(tree: Path) -> None:
    """Rewriting "/scripts" to <repo>/scripts claimed a directory Python never
    searches, so a broken import resolved against the repo's own scripts/."""
    test_file = tree / "test_thing.py"
    test_file.write_text('import sys\nsys.path.insert(0, "/scripts")\nimport tool\n')
    assert not resolves_in_repo("tool", test_file, tree)
    # An absolute path that really points into the repo is a resolution root.
    test_file.write_text(f'import sys\nsys.path.insert(0, "{tree / "scripts"}")\nimport tool\n')
    assert resolves_in_repo("tool", test_file, tree)


def test_a_multi_segment_path_insert_names_one_directory(tmp_path: Path) -> None:
    """`REPO_ROOT / ".github" / "scripts"` is one root written as two literals.
    Reading each constant on its own produced `<repo>/.github` and
    `<repo>/scripts` — neither the directory meant, so the import failed."""
    root = tmp_path / "repo"
    (root / ".github" / "scripts").mkdir(parents=True)
    (root / ".github" / "scripts" / "tool.py").write_text("")
    test_file = root / "test_thing.py"
    test_file.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "REPO_ROOT = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))\n'
        "import tool\n"
    )
    assert resolves_in_repo("tool", test_file, root)

    # The segments are not a pool of independent roots: a module sitting at only
    # one of them does not satisfy the import.
    (root / "scripts").mkdir()
    (root / "scripts" / "stray.py").write_text("")
    assert not resolves_in_repo("stray", test_file, root)


def test_a_base_outside_the_checkout_grants_nothing(tmp_path: Path) -> None:
    """Reading only the literals made `EXTERNAL_ROOT / "scripts"` and
    `REPO_ROOT / "scripts"` the same entry, so an import passed on the strength
    of a module this repo ships at a path the running code never searches."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "tool.py").write_text("")
    test_file = root / "test_thing.py"

    test_file.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        'EXTERNAL_ROOT = Path("/somewhere/else")\n'
        'sys.path.insert(0, str(EXTERNAL_ROOT / "scripts"))\n'
        "import tool\n"
    )
    assert not resolves_in_repo("tool", test_file, root)

    # The same entry against a base that really is the repo root is honored.
    test_file.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "REPO_ROOT = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(REPO_ROOT / "scripts"))\n'
        "import tool\n"
    )
    assert resolves_in_repo("tool", test_file, root)


def test_an_unreadable_base_grants_nothing(tmp_path: Path) -> None:
    """A name this reader cannot follow — computed, imported, rebound — must not
    fall back to the repo root. Reporting an import it cannot justify is the
    cheaper error; approving one it cannot check is the bug class itself."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "tool.py").write_text("")
    test_file = root / "test_thing.py"

    for base in (
        "BASE = os.environ['SOMEWHERE']",
        "from config import BASE",
        'BASE = Path("/a")\nBASE = Path("/b")',  # rebound: which one is live?
    ):
        test_file.write_text(
            "import sys\n"
            "import os\n"
            "from pathlib import Path\n"
            f"{base}\n"
            'sys.path.insert(0, str(BASE / "scripts"))\n'
            "import tool\n"
        )
        assert not resolves_in_repo("tool", test_file, root), base


def test_an_unreadable_first_assignment_still_counts_as_a_binding(tmp_path: Path) -> None:
    """Counting only the assignments this reader understands let an unreadable
    first one go unrecorded, so the readable second one looked like the single
    binding and lent its directory to an insert that ran against the first."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "tool.py").write_text("")
    test_file = root / "test_thing.py"
    test_file.write_text(
        "import sys\n"
        "import os\n"
        "from pathlib import Path\n"
        "BASE = os.environ['SOMEWHERE']\n"
        'sys.path.insert(0, str(BASE / "scripts"))\n'
        "import tool\n"
        "BASE = Path(__file__).resolve().parent\n"
    )
    assert not resolves_in_repo("tool", test_file, root)


@pytest.mark.parametrize(
    "preamble",
    [
        "from mypath import Path",  # not pathlib's
        "from pathlib import Path as _P\nPath = _P\n",  # rebound
        "from pathlib import Path\ndef str(x):\n    return x\n",  # builtin shadowed
    ],
)
def test_a_shadowed_path_or_str_grants_nothing(tmp_path: Path, preamble: str) -> None:
    """`Path` and `str` are spellings, not guarantees. A local, imported or
    rebound one can name a directory outside the checkout at runtime while the
    expression still reads as an in-repo path."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "tool.py").write_text("")
    test_file = root / "test_thing.py"
    test_file.write_text(
        f"import sys\n{preamble}\n"
        "REPO_ROOT = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(REPO_ROOT / "scripts"))\n'
        "import tool\n"
    )
    assert not resolves_in_repo("tool", test_file, root)


@pytest.mark.parametrize(
    "preamble",
    [
        "from mypath import Path",  # not pathlib's
        "from pathlib import Path as _P\nPath = _P\n",  # rebound
        "from pathlib import Path\ndef str(x):\n    return x\n",  # builtin shadowed
    ],
)
def test_a_shadowed_path_or_str_grants_nothing_written_inline(
    tmp_path: Path, preamble: str
) -> None:
    """The same claim written as one expression rather than through a
    module-level name. Validating the names only left this form trusted, so a
    shadowed `Path` or `str` could still name a directory outside the checkout
    at runtime while the insert read as an in-repo path."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "tool.py").write_text("")
    test_file = root / "test_thing.py"
    test_file.write_text(
        f"import sys\n{preamble}\n"
        'sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))\n'
        "import tool\n"
    )
    assert not resolves_in_repo("tool", test_file, root)


@pytest.mark.parametrize(
    "pathlib_import",
    [
        "def helper():\n    from pathlib import Path\n",  # binds inside the function only
        "class Holder:\n    from pathlib import Path\n",  # binds as a class attribute
    ],
)
def test_a_scoped_pathlib_import_grants_nothing(tmp_path: Path, pathlib_import: str) -> None:
    """An import inside a function or class does not bind the module-level name,
    so the `Path(__file__)` beside it raises NameError and the insert it builds
    never runs. Reading the file as trustworthy approved an import on the
    strength of a `sys.path` entry that never existed."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "tool.py").write_text("")
    test_file = root / "test_thing.py"
    test_file.write_text(
        f"import sys\n{pathlib_import}\n"
        'sys.path.insert(0, Path(__file__).resolve().parent / "scripts")\n'
        "import tool\n"
    )
    assert not resolves_in_repo("tool", test_file, root)


def test_a_plain_string_entry_needs_no_pathlib_import(tmp_path: Path) -> None:
    """A literal entry is readable whether or not the file imports `Path`.
    Refusing the whole name map when `Path` is unreadable rejected
    `SCRIPTS = "scripts"`, which needs neither `Path` nor `str` to be read."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "tool.py").write_text("")
    test_file = root / "test_thing.py"
    test_file.write_text(
        'import sys\n\nSCRIPTS = "scripts"\nsys.path.insert(0, SCRIPTS)\nimport tool\n'
    )
    assert resolves_in_repo("tool", test_file, root)


@pytest.mark.parametrize(
    ("base", "entry"),
    [
        (".", 'BASE / "scripts"'),  # TypeError: unsupported operand str / str
        ("scripts/inner", "BASE.parent"),  # AttributeError: str has no .parent
        (".", 'BASE.resolve() / "scripts"'),  # AttributeError: str has no .resolve
    ],
)
def test_a_string_base_cannot_be_joined_or_walked(tmp_path: Path, base: str, entry: str) -> None:
    """`"." / "scripts"` raises TypeError, so the insert never runs and the
    directory is no resolution root. Reading every literal as a `Path` lost the
    runtime type and granted a root for an entry that never existed.

    Each shape is written so that misreading it yields `<repo>/scripts`, where
    the module sits — landing on the repo root instead would assert nothing,
    since the root is a resolution root regardless."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "tool.py").write_text("")
    test_file = root / "test_thing.py"
    # The pathlib import is what makes this a test of the runtime type rather
    # than of the readability check: without it the whole name map is refused
    # anyway, and the assertion would hold for the wrong reason.
    test_file.write_text(
        "import sys\nfrom pathlib import Path\n\n"
        f'BASE = "{base}"\nsys.path.insert(0, {entry})\nimport tool\n'
    )
    assert not resolves_in_repo("tool", test_file, root)


def test_a_wildcard_import_makes_path_and_str_unreadable() -> None:
    """What a wildcard binds cannot be read here, so it can shadow `Path` or
    `str` unseen. `_sys_is_the_module` rejects such a file before any insert is
    read, which is why this is asserted on the predicate itself: the predicate
    has to hold on its own rather than on a caller's unrelated check."""
    clean = "import sys\nfrom pathlib import Path\n"
    assert _path_builtins_are_readable(ast.parse(clean))
    assert not _path_builtins_are_readable(ast.parse(clean + "from custom import *\n"))


def test_dot_github_scripts_is_scanned(tmp_path: Path) -> None:
    """SKIP_DIRS drops `.github` wholesale, which would silently exempt the
    guards and notebook tooling that live in `.github/scripts/` — including this
    guard itself — from the check they exist to enforce."""
    root = tmp_path / "repo"
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "scripts").mkdir()
    (root / ".github" / "scripts" / "guard.py").write_text("")
    (root / ".github" / "workflows" / "helper.py").write_text("")
    (root / ".github" / "scripts" / "__pycache__").mkdir()
    (root / ".github" / "scripts" / "__pycache__" / "guard.py").write_text("")

    scanned = {p.relative_to(root).as_posix() for p in iter_source_files(root)}
    assert ".github/scripts/guard.py" in scanned
    assert ".github/workflows/helper.py" not in scanned
    assert ".github/scripts/__pycache__/guard.py" not in scanned


def test_a_path_entry_cannot_walk_out_of_the_repo(tmp_path: Path) -> None:
    """Lexical containment was not enough: `<repo>/../external` lists `<repo>`
    among its parents, and an in-repo symlink can point anywhere. Either would let
    a module outside the checkout satisfy a missing import."""
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "tool.py").write_text("")
    test_file = root / "test_thing.py"

    test_file.write_text('import sys\nsys.path.insert(0, "../external")\nimport tool\n')
    assert not resolves_in_repo("tool", test_file, root)

    (root / "linked").symlink_to(outside)
    test_file.write_text('import sys\nsys.path.insert(0, "linked")\nimport tool\n')
    assert not resolves_in_repo("tool", test_file, root)

    # A symlink that stays inside the repo still resolves.
    inside = root / "scripts"
    inside.mkdir()
    (inside / "helper.py").write_text("")
    (root / "shortcut").symlink_to(inside)
    test_file.write_text('import sys\nsys.path.insert(0, "shortcut")\nimport helper\n')
    assert resolves_in_repo("helper", test_file, root)


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
