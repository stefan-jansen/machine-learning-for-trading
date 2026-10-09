"""The weekly issue lifecycle: what opens, what closes, and what must never close.

Four defects guard this file. One shipped: the workflow's inline
``name="([^"]*)"`` matched the ``name="`` inside ``classname="``, so every weekly
failure was filed as ``weekly-flake: tests.test_chapter_notebooks`` and no issue
ever named the failing notebook. One was the absence of any close path, which is
why a single issue stayed open for weeks after the plumbing around it changed.

The other two are what a close path can get wrong, and they are the reason these
tests exist rather than a smoke check:

- Closing on something that is not an executed pass. A skipped notebook, a JUnit
  report that was never written, and a failure to list the existing issues all look
  like "no failures reported" and would retire a live failure.
- Opening a second issue for a failure that already has one.

The transitions are read as data from ``reconcile`` and then replayed through
``apply`` against a client that records calls, so both the planning and the writing
are the production code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "weekly_issues.py"
_spec = importlib.util.spec_from_file_location("weekly_issues", SCRIPT)
assert _spec and _spec.loader
weekly_issues = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(weekly_issues)

NB = "22_rag_financial_research::02_domain_embeddings_comparison.py"
PASSING = "22_rag_financial_research::03_hybrid_retrieval.py"
SKIPPED = "24_autonomous_agents::11_research_operator.py"
PREFIX = "weekly-flake"
RUN = "https://github.com/o/r/actions/runs/1"

CHAPTER_REPORT = f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="4">
  <testcase classname="tests.test_chapter_notebooks"
            name="test_chapter_notebook[{NB}]" time="11.4">
    <failure message="Failed">couldn't connect to 'https://huggingface.co'</failure>
  </testcase>
  <testcase classname="tests.test_chapter_notebooks"
            name="test_chapter_notebook[{PASSING}]" time="31.0" />
  <testcase classname="tests.test_chapter_notebooks"
            name="test_chapter_notebook[{SKIPPED}]" time="0.0">
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


class FakeIssues:
    """Records the writes, so a test asserts on what the production code did."""

    def __init__(self, issues: list[dict] | None = None, *, list_raises: bool = False) -> None:
        self.issues = issues or []
        self.list_raises = list_raises
        self.calls: list[tuple] = []
        self.next_number = 9001

    def list_by_label(self, label: str) -> list[dict]:
        if self.list_raises:
            raise OSError("502 Bad Gateway")
        return [i for i in self.issues if label in i.get("labels", [label])]

    def create(self, title: str, body: str, labels: list[str]) -> int:
        number = self.next_number
        self.next_number += 1
        self.calls.append(("create", number, title))
        return number

    def comment(self, number: int, body: str) -> None:
        self.calls.append(("comment", number, body.split(":")[0]))

    def set_state(self, number: int, state: str, reason: str | None = None) -> None:
        self.calls.append(("set_state", number, state, reason))


def issue(number: int, case_id: str, state: str, **extra: object) -> dict:
    return {"number": number, "title": f"{PREFIX}: {case_id}", "state": state, **extra}


def run(outcomes: dict, issues: list[dict]) -> list[tuple]:
    """reconcile + apply: the two halves the workflow calls, over a fake client."""
    client = FakeIssues(issues)
    weekly_issues.apply(
        weekly_issues.reconcile(outcomes, client.list_by_label(PREFIX), PREFIX, RUN), client
    )
    return client.calls


# --------------------------------------------------------------------------- parser


def test_the_failing_id_is_the_notebook_not_the_test_class() -> None:
    outcomes = weekly_issues.classify(CHAPTER_REPORT, "notebook")
    assert outcomes["failed"] == [NB]
    assert "tests.test_chapter_notebooks" not in outcomes["failed"]


def test_a_passing_notebook_is_reported_so_its_issue_can_close() -> None:
    outcomes = weekly_issues.classify(CHAPTER_REPORT, "notebook")
    assert outcomes["passed"] == [PASSING, "25_live_trading::02_etfs_deployment_loop.py"]


def test_a_skipped_notebook_is_in_neither_bucket() -> None:
    outcomes = weekly_issues.classify(CHAPTER_REPORT, "notebook")
    assert outcomes["skipped"] == [SKIPPED]
    assert SKIPPED not in outcomes["passed"]
    assert SKIPPED not in outcomes["failed"]


def test_a_reran_notebook_that_failed_once_stays_failed() -> None:
    """pytest-rerunfailures writes the failed attempt and the passing retry."""
    rerun = CHAPTER_REPORT.replace(
        "</testsuite>",
        f'<testcase classname="tests.test_chapter_notebooks" '
        f'name="test_chapter_notebook[{NB}]" time="9.0" /></testsuite>',
    )
    outcomes = weekly_issues.classify(rerun, "notebook")
    assert outcomes["failed"] == [NB]
    assert NB not in outcomes["passed"]


def test_the_drift_id_strips_the_pytest_decoration() -> None:
    outcomes = weekly_issues.classify(DRIFT_REPORT, "drift")
    assert outcomes["failed"] == ["etf_yahoo"]
    assert outcomes["passed"] == ["fred"]


def test_a_missing_report_raises_rather_than_reading_as_a_clean_week(tmp_path: Path) -> None:
    with pytest.raises(weekly_issues.ReportMissing):
        weekly_issues.read_outcomes(tmp_path / "nope.xml", "notebook")


# ----------------------------------------------------------------------- transitions


def test_a_failure_with_no_issue_opens_exactly_one() -> None:
    calls = run({"failed": [NB], "passed": [], "skipped": []}, [])
    assert [c[0] for c in calls] == ["create"]
    assert calls[0][2] == f"{PREFIX}: {NB}"


def test_an_unchanged_failure_comments_once_and_opens_nothing() -> None:
    calls = run({"failed": [NB], "passed": [], "skipped": []}, [issue(1046, NB, "open")])
    assert [c[0] for c in calls] == ["comment"]
    assert calls[0][1] == 1046


def test_a_failure_after_a_close_reopens_the_same_issue() -> None:
    calls = run({"failed": [NB], "passed": [], "skipped": []}, [issue(1046, NB, "closed")])
    assert [(c[0], c[1]) for c in calls] == [("set_state", 1046), ("comment", 1046)]
    assert calls[0][2] == "open"


def test_an_executed_pass_closes_the_issue_as_completed() -> None:
    calls = run({"failed": [], "passed": [NB], "skipped": []}, [issue(1046, NB, "open")])
    assert [c[0] for c in calls] == ["comment", "set_state"]
    assert calls[1][2:] == ("closed", "completed")


def test_a_skip_closes_nothing() -> None:
    """A notebook that did not run has not been fixed."""
    calls = run({"failed": [], "passed": [], "skipped": [NB]}, [issue(1046, NB, "open")])
    assert calls == []


def test_a_pass_on_an_already_closed_issue_writes_nothing() -> None:
    calls = run({"failed": [], "passed": [NB], "skipped": []}, [issue(1046, NB, "closed")])
    assert calls == []


def test_a_pull_request_with_a_matching_title_is_not_mistaken_for_the_issue() -> None:
    calls = run(
        {"failed": [NB], "passed": [], "skipped": []},
        [issue(7, NB, "open", pull_request={"url": "..."})],
    )
    assert [c[0] for c in calls] == ["create"], "a PR must not absorb the failure"


def test_an_open_duplicate_wins_over_a_closed_one() -> None:
    calls = run(
        {"failed": [NB], "passed": [], "skipped": []},
        [issue(10, NB, "closed"), issue(20, NB, "open")],
    )
    assert [(c[0], c[1]) for c in calls] == [("comment", 20)]


def test_one_run_can_open_one_issue_and_close_another() -> None:
    calls = run(
        {"failed": [NB], "passed": [PASSING], "skipped": []},
        [issue(1046, PASSING, "open")],
    )
    assert [c[0] for c in calls] == ["create", "comment", "set_state"]
    assert calls[2][1] == 1046


# ------------------------------------------------------------------- failure is not a pass


def test_a_missing_report_exits_non_zero_and_touches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    code = weekly_issues.main(
        [
            "--junit",
            str(tmp_path / "absent.xml"),
            "--id-style",
            "notebook",
            "--prefix",
            PREFIX,
            "--repo",
            "o/r",
        ]
    )
    assert code == 2
    assert "::error::" in capsys.readouterr().out


def test_an_unreadable_issue_list_exits_non_zero_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acting on "no issue exists" would open a duplicate of every standing one."""
    report = tmp_path / "junit.xml"
    report.write_text(CHAPTER_REPORT)
    client = FakeIssues([issue(1046, NB, "open")], list_raises=True)
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setattr(weekly_issues, "RestIssues", lambda repo, token: client)
    code = weekly_issues.main(
        ["--junit", str(report), "--id-style", "notebook", "--prefix", PREFIX, "--repo", "o/r"]
    )
    assert code == 2
    assert client.calls == []


def test_a_missing_token_exits_non_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = tmp_path / "junit.xml"
    report.write_text(CHAPTER_REPORT)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    code = weekly_issues.main(
        ["--junit", str(report), "--id-style", "notebook", "--prefix", PREFIX, "--repo", "o/r"]
    )
    assert code == 2


# ------------------------------------------------- a re-run does not comment twice


def run_with_comments(outcomes: dict, issues: list[dict], comments: dict) -> list[tuple]:
    client = FakeIssues(issues)
    weekly_issues.apply(
        weekly_issues.reconcile(
            outcomes, client.list_by_label(PREFIX), PREFIX, RUN, comments=comments
        ),
        client,
    )
    return client.calls


def test_a_rerun_of_the_same_run_does_not_comment_twice() -> None:
    """Re-running a failed job keeps the run id, so the body would be identical."""
    calls = run_with_comments(
        {"failed": [NB], "passed": [], "skipped": []},
        [issue(1046, NB, "open")],
        {1046: [f"Failed again in the weekly run: {RUN}"]},
    )
    assert calls == []


def test_a_later_run_does_comment() -> None:
    calls = run_with_comments(
        {"failed": [NB], "passed": [], "skipped": []},
        [issue(1046, NB, "open")],
        {1046: ["Failed again in the weekly run: https://github.com/o/r/actions/runs/0"]},
    )
    assert [c[0] for c in calls] == ["comment"]


def test_dedup_never_suppresses_the_reopen() -> None:
    """The issue state is the thing that matters; the comment is the notification."""
    calls = run_with_comments(
        {"failed": [NB], "passed": [], "skipped": []},
        [issue(1046, NB, "closed")],
        {1046: [f"Failed again after closing: {RUN}"]},
    )
    assert [(c[0], c[1], c[2]) for c in calls] == [("set_state", 1046, "open")]


# ------------------------------------------------------- retiring the class-name issue

LEGACY = weekly_issues.LEGACY_IDS[PREFIX]


def test_the_class_name_issue_is_retired_as_not_planned_not_completed() -> None:
    """A wrong title is not an executed passing run, and must not be recorded as one."""
    calls = run({"failed": [NB], "passed": [], "skipped": []}, [issue(1046, LEGACY, "open")])
    assert ("set_state", 1046, "closed", "not_planned") in calls
    assert not any(c[0] == "set_state" and c[3] == "completed" for c in calls)


def test_retiring_it_names_where_the_failures_went_and_opens_the_accurate_issue() -> None:
    client = FakeIssues([issue(1046, LEGACY, "open")])
    plan = weekly_issues.reconcile(
        {"failed": [NB], "passed": [], "skipped": []}, client.issues, PREFIX, RUN
    )
    kinds = [a["action"] for a in plan]
    assert kinds == ["create", "comment", "close_not_planned"]
    assert plan[0]["title"] == f"{PREFIX}: {NB}"
    assert NB in plan[1]["body"]
    assert "not planned" in plan[1]["body"]


def test_a_quiet_week_still_retires_it_and_says_so() -> None:
    """Its title can never match an id, so leaving it open is leaving it forever."""
    plan = weekly_issues.reconcile(
        {"failed": [], "passed": [], "skipped": []}, [issue(1046, LEGACY, "open")], PREFIX, RUN
    )
    assert [a["action"] for a in plan] == ["comment", "close_not_planned"]
    assert "no accurate issue to point at yet" in plan[0]["body"]


def test_an_already_closed_class_name_issue_is_left_alone() -> None:
    calls = run({"failed": [], "passed": [], "skipped": []}, [issue(1046, LEGACY, "closed")])
    assert calls == []


def test_retirement_does_not_fire_twice_on_a_rerun() -> None:
    """Closed on the first run, so the second finds nothing open to retire."""
    first = FakeIssues([issue(1046, LEGACY, "open")])
    weekly_issues.apply(
        weekly_issues.reconcile(
            {"failed": [], "passed": [], "skipped": []}, first.issues, PREFIX, RUN
        ),
        first,
    )
    assert any(c[0] == "set_state" for c in first.calls)
    second = FakeIssues([issue(1046, LEGACY, "closed")])
    weekly_issues.apply(
        weekly_issues.reconcile(
            {"failed": [], "passed": [], "skipped": []}, second.issues, PREFIX, RUN
        ),
        second,
    )
    assert second.calls == []


def test_the_drift_job_has_its_own_class_name_issue_to_retire() -> None:
    assert weekly_issues.LEGACY_IDS["external-drift"] == "tests.test_external_drift"


# ------------------------------------------- the comment lookup reaches the recent ones


class FakeTransport:
    """Stands in for the HTTP call, so RestIssues' real pagination is exercised."""

    def __init__(self, pages: list[list[dict]]) -> None:
        self.pages = pages
        self.requested: list[str] = []

    def __call__(self, method: str, path: str, payload: dict | None = None) -> object:
        self.requested.append(path)
        page = int(path.rsplit("page=", 1)[1])
        return self.pages[page - 1] if page <= len(self.pages) else []


def rest_with(pages: list[list[dict]]) -> tuple[object, FakeTransport]:
    client = weekly_issues.RestIssues("o/r", "token")
    transport = FakeTransport(pages)
    client._call = transport  # noqa: SLF001 - the seam this test exists to drive
    return client, transport


def test_the_comment_lookup_reads_past_the_first_page() -> None:
    """Page 1 is the OLDEST hundred: the endpoint returns oldest first and has no sort.

    Reading only page 1 would miss every recent run URL on a long thread, so a job
    re-run would comment a second time - the duplicate this lookup prevents.
    """
    old_page = [{"body": f"old {i}"} for i in range(100)]
    recent = [{"body": f"Failed again in the weekly run: {RUN}"}]
    client, transport = rest_with([old_page, recent])
    bodies = client.comments(1046)
    assert len(transport.requested) == 2, "it stopped at the first page"
    assert weekly_issues.already_reported(bodies, RUN)


def test_the_comment_lookup_stops_on_a_short_page() -> None:
    client, transport = rest_with([[{"body": "only one"}]])
    assert client.comments(1046) == ["only one"]
    assert len(transport.requested) == 1


def test_the_comment_lookup_is_capped(capsys: pytest.CaptureFixture[str]) -> None:
    """A pathological thread must not turn one dedup check into endless requests."""
    full = [{"body": "x"} for _ in range(100)]
    client, transport = rest_with([full] * 50)
    client.comments(1046)
    assert len(transport.requested) == weekly_issues.RestIssues.COMMENT_PAGE_CAP
    assert "::warning::" in capsys.readouterr().out


def test_a_longer_run_id_does_not_suppress_a_shorter_one() -> None:
    """`runs/1` is a substring of `runs/10`, and suppressing wrongly loses a report."""
    assert not weekly_issues.already_reported(["Failed again: .../runs/10"], ".../runs/1")
    assert weekly_issues.already_reported(["Failed again: .../runs/1"], ".../runs/1")
    assert weekly_issues.already_reported(["Failed: .../runs/1 and more"], ".../runs/1")
    assert weekly_issues.already_reported(["a: .../runs/10", "b: .../runs/1"], ".../runs/1")


def test_no_run_url_suppresses_nothing() -> None:
    assert not weekly_issues.already_reported(["anything"], "")
