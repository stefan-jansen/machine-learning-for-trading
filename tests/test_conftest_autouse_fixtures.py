"""The autouse fixtures in ``conftest.py`` run against every test in the repository.

That reach is what makes them worth a test of their own: a fixture that works in the
environment its author had, and raises in one CI job's environment, turns every test in
that job into a teardown error while the tests themselves pass. ``test-unit`` checks out
no test-data and its Chapter 21 step does not override the workflow-level
``ML4T_DATA_PATH``, so an autouse teardown that imports ``case_studies.research`` reported
70 passed and 70 errors there against tests that read nothing from disk.

The throwaway suite has to live under ``tests/`` because that is the only place
``tests/conftest.py`` applies. A suite written to ``tmp_path`` loads no conftest at all and
would pass whatever the fixture does.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


def test_autouse_teardown_survives_a_data_path_that_does_not_exist() -> None:
    """A test that reads nothing from disk must not fail in teardown for want of a dataset.

    ``case_studies.research`` imports ``utils.config``, which raises ``FileNotFoundError`` at
    import time when ``ML4T_DATA_PATH`` names a directory that is not there. Reaching for that
    module from an autouse teardown charges every test in the run for a dependency it never
    asked for. Reading ``sys.modules`` answers the only question the teardown has - did
    anything import the module and leave state in it - without importing anything.
    """
    suite = Path(tempfile.mkdtemp(prefix="autouse_probe_", dir=TESTS_DIR))
    try:
        (suite / "test_reads_nothing.py").write_text(
            "def test_arithmetic():\n    assert 1 + 1 == 2\n"
        )
        env = dict(os.environ)
        env["ML4T_DATA_PATH"] = str(REPO_ROOT / "no-such-data-directory")
        env["PYTHONPATH"] = str(REPO_ROOT)
        env.pop("ML4T_DATA_PATH_IS_DEFAULT", None)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(suite), "-p", "no:cacheprovider", "-q"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(suite, ignore_errors=True)

    combined = result.stdout + result.stderr
    assert "Data directory not found" not in combined, combined
    assert "error" not in result.stdout, combined
    assert result.returncode == 0, combined


def _fixture_defs(tree: ast.Module) -> list[ast.FunctionDef]:
    """Top-level functions decorated with ``pytest.fixture``, however it is spelled."""
    out: list[ast.FunctionDef] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            if name == "fixture":
                out.append(node)
                break
    return out


def _is_autouse(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        for kw in dec.keywords:
            if kw.arg == "autouse" and isinstance(kw.value, ast.Constant) and kw.value.value:
                return True
    return False


def _pops_the_output_root(node: ast.FunctionDef) -> bool:
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "pop"):
            continue
        if not (isinstance(func.value, ast.Attribute) and func.value.attr == "environ"):
            continue
        if call.args and isinstance(call.args[0], ast.Constant):
            if call.args[0].value == "ML4T_OUTPUT_DIR":
                return True
    return False


def _test_modules() -> list[Path]:
    return sorted(p for p in TESTS_DIR.rglob("test_*.py") if p.name != "conftest.py")


def test_no_test_module_redefines_a_conftest_autouse_fixture() -> None:
    """A same-named fixture in a test module replaces the conftest one for that module.

    pytest resolves a fixture from the most specific definition, so a module that defines
    ``_restore_output_root`` does not run alongside ``tests/conftest.py``'s fixture of that
    name - it runs instead of it. Both copies existed here and they did not agree:
    conftest's puts ``ML4T_OUTPUT_DIR`` back to the value ``seeded_output_dir`` installed,
    and the module copies deleted it. Measured 2026-09-15 on
    ``pytest tests/test_cme_futures_research.py tests/test_artifact_specs.py``: two failures
    reading ``Missing prerequisites for 'us_equities_panel': features/financial.parquet,
    labels/fwd_ret_1d.parquet`` against files that are on disk, because every later test
    resolved ``get_case_study_dir`` against the committed ``case_studies/`` tree.

    An override is legitimate in general. An override of an autouse fixture that restores
    process-global state is not: nothing at the call site says the conftest version stopped
    running, and the tests that pay for it are in other files.

    The same command passes in one checkout and fails in another, so a green run is not
    evidence the bug is gone. A `--case-study` worktree symlinks
    `case_studies/<cs>/{features,labels}` to the canonical store, and `get_case_study_dir`
    then resolves to real artifacts whatever the variable says; a checkout without those
    links has no fallback. What distinguishes a fix from a tree that hides it is the variable
    being restored, not the run being green.
    """
    conftest = ast.parse((TESTS_DIR / "conftest.py").read_text())
    protected = {f.name for f in _fixture_defs(conftest) if _is_autouse(f)}
    assert protected, "tests/conftest.py defines no autouse fixtures - the check is vacuous"

    offenders = []
    for path in _test_modules():
        for fixture in _fixture_defs(ast.parse(path.read_text())):
            if fixture.name in protected:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{fixture.lineno} {fixture.name}")
    assert not offenders, (
        "these fixtures replace an autouse fixture of the same name in tests/conftest.py, "
        "so conftest's version never runs for that module: " + ", ".join(offenders)
    )


def test_no_test_module_deletes_the_output_root_in_a_fixture() -> None:
    """Deleting ``ML4T_OUTPUT_DIR`` is not the same as restoring it, and it outlives the file.

    ``seeded_output_dir`` is session-scoped and writes the variable exactly once, so a fixture
    that pops it in teardown removes it for the rest of the worker rather than for the rest of
    the module. ``tests/conftest.py`` carries the restoring version; a module-level copy that
    pops defeats it even when it does not shadow it by name.

    A pop inside a test body is untouched - ``test_research_workspace`` pops the variable to
    assert what ``Study.at`` does without one, which is the behaviour under test.
    """
    offenders = []
    for path in _test_modules():
        for fixture in _fixture_defs(ast.parse(path.read_text())):
            if _pops_the_output_root(fixture):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{fixture.lineno} {fixture.name}")
    assert not offenders, (
        "these fixtures delete ML4T_OUTPUT_DIR instead of restoring it, which removes the "
        "seeded output root for every later test in the worker: " + ", ".join(offenders)
    )


def test_the_session_output_dir_is_not_adopted_from_a_leaking_fixture() -> None:
    """The session seeds the directory it started with, not one a later fixture installed.

    ``seeded_output_dir`` is session-scoped and used to read ``ML4T_OUTPUT_DIR`` live, at
    whatever point the first test needing it ran. Every higher-scoped fixture for that test
    has already set up by then, and the autouse restore cannot see any of them: it is
    function-scoped, so the value it captures as "before" is taken after they ran, and it
    faithfully puts their leak back. A module-scoped fixture in any file that writes the
    variable therefore chose the output directory for the whole worker, and every later
    consumer resolved artifacts inside a temp tree belonging to a finished test - which is
    what ``test_us_equities_pilot_helpers_preserve_current_outputs`` reads as artifacts
    missing that are present on disk, while passing when its own file runs alone.

    The probe suite runs in file order, with the leaking module sorting first, and asserts
    the seeded directory is outside the leaked tree. Reverting the fixture to
    ``os.environ.get`` fails it.

    It has to live under ``tests/`` because that is the only place ``tests/conftest.py``
    applies, and it runs in a subprocess because the session fixture under test is created
    once per session and this session has already created it.
    """
    suite = Path(tempfile.mkdtemp(prefix="output_dir_probe_", dir=TESTS_DIR))
    leaked = suite / "leaked-output-dir"
    try:
        (suite / "test_a_leaks.py").write_text(
            "import os\n"
            "import pytest\n"
            "\n"
            "LEAKED = os.environ['PROBE_LEAKED_DIR']\n"
            "\n"
            "\n"
            "@pytest.fixture(scope='module', autouse=True)\n"
            "def _leak():\n"
            "    # A module-scoped fixture sets up before every function-scoped one for the\n"
            "    # first test in this file, so conftest's restore captures the leak as the\n"
            "    # value to restore to.\n"
            "    os.environ['ML4T_OUTPUT_DIR'] = LEAKED\n"
            "    yield\n"
            "\n"
            "\n"
            "def test_leaks():\n"
            "    assert os.environ['ML4T_OUTPUT_DIR'] == LEAKED\n"
        )
        (suite / "test_b_seeds.py").write_text(
            "import os\n"
            "from pathlib import Path\n"
            "\n"
            "LEAKED = Path(os.environ['PROBE_LEAKED_DIR'])\n"
            "\n"
            "\n"
            "def test_the_seeded_dir_is_not_the_leaked_one(seeded_output_dir):\n"
            "    seeded = Path(seeded_output_dir)\n"
            "    assert LEAKED not in (seeded, *seeded.parents), (\n"
            "        f'seeded {seeded} inside the leaked tree {LEAKED}'\n"
            "    )\n"
        )
        env = dict(os.environ)
        env["PROBE_LEAKED_DIR"] = str(leaked)
        env["PYTHONPATH"] = str(REPO_ROOT)
        # The starting value is what the fixture must honour, so the probe starts it unset and
        # lets the fixture mint its own directory. A run that adopts the leak lands inside it.
        env.pop("ML4T_OUTPUT_DIR", None)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(suite),
                "-p",
                "no:cacheprovider",
                "-p",
                "no:randomly",
                "-q",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(suite, ignore_errors=True)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
