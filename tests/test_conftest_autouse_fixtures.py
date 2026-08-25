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
