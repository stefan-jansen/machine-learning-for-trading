"""The weekly issue opener must key on the notebook, and must close on a real pass.

Two defects this guards. The first shipped: the workflow's inline
`name="([^"]*)"` matched inside `classname="`, so every weekly failure was filed
as `weekly-flake: tests.test_chapter_notebooks` and the failing notebook never
reached the title. The second is the one the close step could introduce: a
skipped notebook is not a passing notebook, and closing its issue on a skip
would retire an unfixed failure.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "junit_outcomes.py"

_spec = importlib.util.spec_from_file_location("junit_outcomes", SCRIPT)
assert _spec and _spec.loader
junit_outcomes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(junit_outcomes)

NB = "22_rag_financial_research::02_domain_embeddings_comparison.py"

CHAPTER_REPORT = f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="4">
  <testcase classname="tests.test_chapter_notebooks"
            name="test_chapter_notebook[{NB}]" time="11.4">
    <failure message="Failed">couldn't connect to 'https://huggingface.co'</failure>
  </testcase>
  <testcase classname="tests.test_chapter_notebooks"
            name="test_chapter_notebook[22_rag_financial_research::03_hybrid_retrieval.py]"
            time="31.0" />
  <testcase classname="tests.test_chapter_notebooks"
            name="test_chapter_notebook[24_autonomous_agents::11_research_operator.py]" time="0.0">
    <skipped type="pytest.skip" message="Skipped: Requires ~/ml4t/skills/" />
  </testcase>
  <testcase classname="tests.test_chapter_notebooks"
            name="test_chapter_notebook[25_live_trading::02_etfs_deployment_loop.py]" time="31.0" />
</testsuite></testsuites>
"""

DRIFT_REPORT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="2">
  <testcase classname="tests.test_external_drift" name="test_etf_yahoo_reachable" time="0.4">
    <error message="schema">column moved</error>
  </testcase>
  <testcase classname="tests.test_external_drift" name="test_fred_reachable" time="0.2" />
</testsuite></testsuites>
"""


def test_the_failing_id_is_the_notebook_not_the_test_class() -> None:
    outcomes = junit_outcomes.classify(CHAPTER_REPORT, "notebook")
    assert outcomes["failed"] == [NB]
    # The defect this replaces: the class name reaching the issue title.
    assert "tests.test_chapter_notebooks" not in outcomes["failed"]


def test_a_passing_notebook_is_reported_so_its_issue_can_close() -> None:
    outcomes = junit_outcomes.classify(CHAPTER_REPORT, "notebook")
    assert outcomes["passed"] == [
        "22_rag_financial_research::03_hybrid_retrieval.py",
        "25_live_trading::02_etfs_deployment_loop.py",
    ]


def test_a_skipped_notebook_is_not_a_passing_one() -> None:
    outcomes = junit_outcomes.classify(CHAPTER_REPORT, "notebook")
    skipped = "24_autonomous_agents::11_research_operator.py"
    assert outcomes["skipped"] == [skipped]
    assert skipped not in outcomes["passed"]
    assert skipped not in outcomes["failed"]


def test_a_reran_notebook_that_failed_once_stays_failed() -> None:
    """pytest-rerunfailures writes the failed attempt and the passing retry."""
    rerun = CHAPTER_REPORT.replace(
        "</testsuite>",
        f'<testcase classname="tests.test_chapter_notebooks" '
        f'name="test_chapter_notebook[{NB}]" time="9.0" /></testsuite>',
    )
    outcomes = junit_outcomes.classify(rerun, "notebook")
    assert outcomes["failed"] == [NB]
    assert NB not in outcomes["passed"]


def test_the_drift_id_strips_the_pytest_decoration() -> None:
    outcomes = junit_outcomes.classify(DRIFT_REPORT, "drift")
    assert outcomes["failed"] == ["etf_yahoo"]
    assert outcomes["passed"] == ["fred"]


def test_an_error_child_counts_as_a_failure() -> None:
    assert junit_outcomes.classify(DRIFT_REPORT, "drift")["failed"] == ["etf_yahoo"]


def test_the_script_prints_json_and_tolerates_an_absent_report(tmp_path: Path) -> None:
    report = tmp_path / "junit.xml"
    report.write_text(CHAPTER_REPORT)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(report), "--id-style", "notebook"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(out.stdout)["failed"] == [NB]

    missing = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "nope.xml"), "--id-style", "drift"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(missing.stdout) == {"failed": [], "passed": [], "skipped": []}
