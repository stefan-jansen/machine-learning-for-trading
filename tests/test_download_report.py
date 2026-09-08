"""The per-dataset result of `download_all.py` must survive a lost log.

`Reader install` reaches seven third-party sources and fails as a whole when one of
them is down. Which one failed decides the response - an upstream outage is waited
out, a total failure points at egress from the runner - and that name lived only in
the step log. Run 33988501549 failed on Windows and `gh run view --log` returned
nothing, so the diagnosis had to be dug out of the raw job-log endpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github" / "scripts"))

from publish_download_report import main as publish_main  # noqa: E402
from publish_download_report import render  # noqa: E402

from data.download_all import write_report  # noqa: E402

# The seven free datasets, and the one that was down on 2026-09-05.
FREE_RESULTS = {
    "ETFs": True,
    "Crypto": False,
    "Prediction Markets": True,
    "CFTC COT": True,
    "Fama-French": True,
    "AQR": True,
    "Firm Characteristics": True,
}


def test_report_names_the_dataset_that_failed(tmp_path: Path) -> None:
    report = tmp_path / "download-report.json"

    write_report(report, FREE_RESULTS, mode="free-only")

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["failed"] == ["Crypto"]
    assert payload["completed"] == 6
    assert payload["total"] == 7
    assert payload["datasets"]["Crypto"] is False
    assert payload["datasets"]["ETFs"] is True


def test_report_distinguishes_one_outage_from_a_total_failure(tmp_path: Path) -> None:
    """Six of seven failing and one of seven failing must not read the same."""
    one_down = tmp_path / "one.json"
    all_down = tmp_path / "all.json"

    write_report(one_down, FREE_RESULTS, mode="free-only")
    write_report(all_down, dict.fromkeys(FREE_RESULTS, False), mode="free-only")

    assert len(json.loads(one_down.read_text())["failed"]) == 1
    assert len(json.loads(all_down.read_text())["failed"]) == 7


def test_no_report_path_writes_nothing(tmp_path: Path) -> None:
    """A reader running the script by hand gets the printed summary and no file."""
    write_report(None, FREE_RESULTS, mode="free-only")

    assert list(tmp_path.iterdir()) == []


def test_rendered_summary_names_the_failure(tmp_path: Path) -> None:
    report = tmp_path / "download-report.json"
    write_report(report, FREE_RESULTS, mode="free-only")

    markdown = render(json.loads(report.read_text(encoding="utf-8")))

    assert "6 of 7 datasets arrived." in markdown
    assert "| Crypto | **FAILED** |" in markdown
    assert "| ETFs | OK |" in markdown
    assert "Failed: Crypto." in markdown


def test_publisher_appends_to_the_step_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "download-report.json"
    write_report(report, FREE_RESULTS, mode="free-only")
    step_summary = tmp_path / "step-summary.md"
    step_summary.write_text("earlier content\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))

    assert publish_main(["publish", str(report)]) == 0

    written = step_summary.read_text(encoding="utf-8")
    assert written.startswith("earlier content\n")
    assert "| Crypto | **FAILED** |" in written


def test_publisher_is_quiet_when_the_download_never_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step runs with if: always(), so a job that died earlier must not fail here."""
    step_summary = tmp_path / "step-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))

    assert publish_main(["publish", str(tmp_path / "absent.json")]) == 0
    assert not step_summary.exists()
