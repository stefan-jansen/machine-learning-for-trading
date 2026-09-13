"""The lag report has to answer the question the run-level view got wrong.

On 2026-09-12 every `Tests` run on `main` read `cancelled` for eleven hours, and the
newest run carrying any conclusion was a `failure` ten merges back on one job that had
already passed since. Both halves of that are encoded here: a superseded verdict must
lose to a newer one, and a job that has not reported on anything merged since must be
visible even though nothing is red.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ci_job_lag", REPO_ROOT / ".github" / "scripts" / "ci_job_lag.py"
)
assert _spec and _spec.loader
ci_job_lag = importlib.util.module_from_spec(_spec)
# Registered before exec: the module defines a dataclass, and `dataclasses` resolves the
# defining module out of `sys.modules` to read its annotations.
sys.modules["ci_job_lag"] = ci_job_lag
_spec.loader.exec_module(ci_job_lag)

JobRow = ci_job_lag.JobRow
newest_verdicts = ci_job_lag.newest_verdicts

TIP, ONE_BACK, FIVE_BACK = "aaa", "bbb", "fff"
HISTORY = [TIP, ONE_BACK, "ccc", "ddd", "eee", FIVE_BACK]


def _row(sha: str, name: str, conclusion: str | None, status: str = "completed") -> JobRow:
    return JobRow(sha=sha, name=name, status=status, conclusion=conclusion, completed_at=None)


def test_a_newer_pass_supersedes_an_older_failure() -> None:
    """The exact shape that made the run list lie: the red is real and it is not current."""

    verdicts = newest_verdicts(
        [_row(FIVE_BACK, "ch22", "failure"), _row(ONE_BACK, "ch22", "success")], HISTORY
    )

    assert verdicts["ch22"] == (ONE_BACK, "success", 1)


def test_a_newer_failure_supersedes_an_older_pass() -> None:
    """And the other direction, so the rule is recency and not optimism."""

    verdicts = newest_verdicts(
        [_row(FIVE_BACK, "ch22", "success"), _row(ONE_BACK, "ch22", "failure")], HISTORY
    )

    assert verdicts["ch22"] == (ONE_BACK, "failure", 1)


def test_recency_is_position_on_main_not_the_order_rows_arrive() -> None:
    """A run started earlier can finish later; the question is which commit it is about."""

    rows = [_row(ONE_BACK, "lint", "success"), _row(FIVE_BACK, "lint", "success")]
    assert newest_verdicts(rows, HISTORY)["lint"][2] == 1
    assert newest_verdicts(list(reversed(rows)), HISTORY)["lint"][2] == 1


def test_a_cancelled_job_carries_no_verdict() -> None:
    """Which is the whole subject: a cancelled run says nothing about the tree it left."""

    verdicts = newest_verdicts(
        [_row(TIP, "cs-etfs", "cancelled"), _row(FIVE_BACK, "cs-etfs", "success")], HISTORY
    )

    assert verdicts["cs-etfs"] == (FIVE_BACK, "success", 5)


@pytest.mark.parametrize("conclusion", [None, "", "null"])
def test_an_unfinished_job_carries_no_verdict(conclusion: str | None) -> None:
    """A job still running on the tip has not reported on it yet."""

    verdicts = newest_verdicts(
        [
            _row(TIP, "cs-etfs", conclusion, status="in_progress"),
            _row(FIVE_BACK, "cs-etfs", "success"),
        ],
        HISTORY,
    )

    assert verdicts["cs-etfs"] == (FIVE_BACK, "success", 5)


def test_a_skip_is_a_verdict() -> None:
    """A job the path filter correctly did not run HAS reported on that tree.

    Treating a skip as no verdict would report a permanent gap for every job outside the
    diff, which is most of them on most commits.
    """

    verdicts = newest_verdicts([_row(TIP, "test-benchmark", "skipped")], HISTORY)

    assert verdicts["test-benchmark"] == (TIP, "skipped", 0)


def test_an_unexpanded_matrix_name_is_not_a_job() -> None:
    """`cs-${{ matrix.case-study }}` is what a matrix skipped before expansion reports as.

    It names no job, so counting it would show a gap that can never close. Both spellings
    appear in the API for this repo.
    """

    rows = [
        _row(TIP, "cs-${{ matrix.case-study }}", "skipped"),
        _row(TIP, "matrix.name", "skipped"),
        _row(TIP, "lint", "success"),
    ]

    assert sorted(newest_verdicts(rows, HISTORY)) == ["lint"]


def test_a_commit_outside_the_history_window_is_dropped() -> None:
    """A run older than the window cannot be placed, and a guessed position is worse."""

    assert newest_verdicts([_row("zzz", "lint", "success")], HISTORY) == {}


def test_a_pull_request_head_says_nothing_about_main() -> None:
    """The rule that a skip is a verdict holds only because this reads `main`.

    A pull request runs what its diff touches, so its skip means "this diff did not need
    this job" - not "this tree has been tested". Four of the five merges that opened the
    gap skipped these jobs correctly, so counting a PR run would score them as covered and
    the hole would read as zero lag. The API filter is the first guard; this is the second,
    and it holds whatever the query returns.
    """

    pr_head = "pr-head-not-on-main"
    rows = [_row(pr_head, "cs-etfs", "skipped"), _row(FIVE_BACK, "cs-etfs", "success")]

    assert newest_verdicts(rows, HISTORY)["cs-etfs"] == (FIVE_BACK, "success", 5)


def test_the_threshold_decides_the_exit_status(capsys: pytest.CaptureFixture[str]) -> None:
    verdicts = {"lint": (TIP, "success", 0), "cs-etfs": (FIVE_BACK, "success", 5)}

    assert ci_job_lag.report(dict(verdicts), max_lag=5) == 0
    assert ci_job_lag.report(dict(verdicts), max_lag=4) == 1
    assert "cs-etfs at 5" in capsys.readouterr().out


def test_a_window_with_no_verdict_at_all_is_loud() -> None:
    """Silence and health are not the same reading, which is the defect in one line."""

    assert ci_job_lag.report({}, max_lag=4) == 1
