#!/usr/bin/env python3
"""File, reopen and close one issue per failing weekly notebook or data source.

The repository carried a single permanently open ``weekly-flake`` issue for weeks,
for two reasons that have nothing to do with each other.

The title was the test class. Both weekly jobs parsed their JUnit report with
``new RegExp(`${name}="([^"]*)"`)`` applied to the whole ``<testcase ...>`` tag, and
``classname="`` contains the substring ``name="``. Every failure deduped onto
``weekly-flake: tests.test_chapter_notebooks``, so no issue ever named the notebook
that failed.

Nothing ever closed an issue. Both jobs only created and commented, so a flake that
got fixed left a standing open issue with no path back to closed.

Parsing and the transitions live here, in one module with tests, rather than as two
copies of a regex and two copies of an issue loop inside workflow YAML. The YAML
regex is what produced the wrong titles, and nothing in a workflow file can be run
against a case.

What must stay true, and what the tests pin:

- A failure opens one issue, or comments on the one that exists, or reopens the one
  that was closed. It never opens a second issue for the same id.
- A notebook that *executed and passed* closes its issue.
- A skip is not a pass. A missing report is not a pass. A failure to list the
  existing issues is not a pass. None of them closes anything, and the last two
  exit non-zero so the job surfaces them.
- An unchanged failure comments once per run and does not duplicate the issue. A
  re-run of the same workflow run does not comment twice, because a comment already
  reporting that run's URL is not repeated.
- The issues filed under the old class-name titles are retired once, by this code
  rather than by hand. They are closed as `not_planned`, never as `completed`: a
  title repair is not an executed successful run, and conflating the two would
  record that a notebook passed when nothing ran.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol
from xml.etree import ElementTree

ID_STYLES = ("notebook", "drift")
API = "https://api.github.com"

# What the broken parser put in the title: the test class, for every failure. One
# issue per prefix absorbed every weekly failure for weeks (#1046 for the chapter
# job). No notebook or source id will ever match these, so the close path cannot
# reach them and they would stay open forever beside the accurate ones.
LEGACY_IDS = {
    "weekly-flake": "tests.test_chapter_notebooks",
    "external-drift": "tests.test_external_drift",
}


class ReportMissing(RuntimeError):
    """The pytest step wrote no JUnit report, so nothing is known about any notebook."""


def testcase_id(name: str, style: str) -> str:
    """Reduce a pytest testcase name to the id the issue title is keyed on.

    ``test_chapter_notebook[22_rag/02_x.py]`` -> ``22_rag/02_x.py``
    ``test_etf_yahoo_reachable``              -> ``etf_yahoo``
    """
    if style == "notebook":
        start = name.find("[")
        if start == -1 or not name.endswith("]"):
            return name
        return name[start + 1 : -1]
    if style == "drift":
        return name.removeprefix("test_").removesuffix("_reachable")
    msg = f"unknown id style {style!r}; expected one of {ID_STYLES}"
    raise ValueError(msg)


def classify(xml: str, style: str) -> dict[str, list[str]]:
    """Split every testcase in a JUnit report into failed, passed and skipped ids."""
    root = ElementTree.fromstring(xml)
    outcomes: dict[str, list[str]] = {"failed": [], "passed": [], "skipped": []}
    for case in root.iter("testcase"):
        name = case.get("name")
        if not name:
            continue
        case_id = testcase_id(name, style)
        if case.find("failure") is not None or case.find("error") is not None:
            bucket = "failed"
        elif case.find("skipped") is not None:
            bucket = "skipped"
        else:
            bucket = "passed"
        if case_id not in outcomes[bucket]:
            outcomes[bucket].append(case_id)
    # A parametrization that failed once and passed on a rerun is still a failure to
    # report: pytest-rerunfailures writes the failed attempt and the passing retry as
    # separate testcases, and an id must not land in two buckets.
    outcomes["passed"] = [i for i in outcomes["passed"] if i not in outcomes["failed"]]
    outcomes["skipped"] = [
        i
        for i in outcomes["skipped"]
        if i not in outcomes["failed"] and i not in outcomes["passed"]
    ]
    return outcomes


def read_outcomes(junit: Path, style: str) -> dict[str, list[str]]:
    if not junit.exists():
        msg = (
            f"{junit} does not exist. The pytest step wrote no report, so no notebook "
            f"can be said to have passed and nothing is closed."
        )
        raise ReportMissing(msg)
    return classify(junit.read_text(), style)


def owned_issue(issues: list[dict], title: str) -> dict | None:
    """The one issue a title owns.

    Pull requests are dropped: ``GET /issues`` returns them too. An open issue wins
    over a closed one with the same title and the newest of two open ones wins, so a
    historical duplicate cannot make this comment on a thread nobody reads.
    """
    candidates = [i for i in issues if i.get("title") == title and not i.get("pull_request")]
    if not candidates:
        return None
    return max(candidates, key=lambda i: (i.get("state") == "open", i.get("number", 0)))


def already_reported(comments: list[str], run_url: str) -> bool:
    """Has this run already been commented on this issue?

    A re-run of a failed job keeps the same run id, so without this a job re-run
    posts the same "failed again" line a second time on every standing issue.

    The match needs a boundary. A run URL ends in the run id, and a plain substring
    test makes `.../runs/1` match a comment about `.../runs/10`, which would suppress
    a real notification rather than a duplicate one. Suppressing wrongly is the worse
    error of the two, so the character after the URL has to be a non-digit.
    """
    if not run_url:
        return False
    for body in comments:
        start = 0
        while (at := body.find(run_url, start)) != -1:
            after = at + len(run_url)
            if after >= len(body) or not body[after].isdigit():
                return True
            start = after
    return False


def reconcile(
    outcomes: dict[str, list[str]],
    issues: list[dict],
    prefix: str,
    run_url: str,
    intro: str = "",
    comments: dict[int, list[str]] | None = None,
) -> list[dict]:
    """The writes this run should make, as data, so a test can read them.

    Returns a list of ``{"action", ...}`` in the order they must be applied: a reopen
    precedes the comment that explains it, and a closing comment precedes the close.
    ``comments`` maps an issue number to its existing comment bodies; an empty or
    absent entry means none are known and nothing is suppressed.
    """
    actions: list[dict] = []
    comments = comments or {}

    for case_id in outcomes["failed"]:
        title = f"{prefix}: {case_id}"
        existing = owned_issue(issues, title)
        if existing is None:
            actions.append(
                {
                    "action": "create",
                    "title": title,
                    "labels": [prefix],
                    "body": (
                        f"{intro}\n\n`{case_id}`\n\nRun: {run_url}\n\n"
                        f"This issue is opened, commented and closed by the weekly "
                        f"workflow. It closes on the first run in which this executes "
                        f"and passes, so an open one means a standing failure. A skipped "
                        f"run does not close it."
                    ).strip(),
                }
            )
            continue
        seen = already_reported(comments.get(existing["number"], []), run_url)
        if existing["state"] == "closed":
            actions.append({"action": "reopen", "number": existing["number"]})
            if not seen:
                actions.append(
                    {
                        "action": "comment",
                        "number": existing["number"],
                        "body": f"Failed again after closing: {run_url}",
                    }
                )
        elif not seen:
            actions.append(
                {
                    "action": "comment",
                    "number": existing["number"],
                    "body": f"Failed again in the weekly run: {run_url}",
                }
            )

    for case_id in outcomes["passed"]:
        existing = owned_issue(issues, f"{prefix}: {case_id}")
        if existing is None or existing["state"] != "open":
            continue
        actions.append(
            {
                "action": "comment",
                "number": existing["number"],
                "body": f"Passed in the weekly run: {run_url}. Closing; it reopens if it fails again.",
            }
        )
        actions.append({"action": "close", "number": existing["number"]})

    actions += retire_legacy(issues, prefix, run_url, outcomes["failed"])
    return actions


def retire_legacy(issues: list[dict], prefix: str, run_url: str, failed: list[str]) -> list[dict]:
    """Close the class-name issue this parser replaces, once, preserving its thread.

    Closed as ``not_planned``: the title was wrong, which is not the same event as a
    notebook executing and passing, and ``completed`` is reserved for that. The issue
    is closed rather than retitled so its comment history stays where anyone
    following it will find it, and the comment says where the accurate issues are.
    """
    legacy_id = LEGACY_IDS.get(prefix)
    if legacy_id is None:
        return []
    existing = owned_issue(issues, f"{prefix}: {legacy_id}")
    if existing is None or existing["state"] != "open":
        return []
    if failed:
        where = "The failures it collected are now tracked per id: " + ", ".join(
            f"`{prefix}: {case_id}`" for case_id in failed
        )
    else:
        where = (
            "Nothing failed in this run, so there is no accurate issue to point at yet. "
            "One opens per failing id from here on."
        )
    return [
        {
            "action": "comment",
            "number": existing["number"],
            "body": (
                f"Retiring this issue. Its title is the pytest class rather than the "
                f'notebook, because the workflow read the `name="` inside '
                f'`classname="`, so every weekly failure deduped onto this one thread '
                f"and none of them named what failed.\n\n{where}\n\nClosing as not "
                f"planned rather than completed: the title was wrong, which is not the "
                f"same thing as a notebook running and passing. Nothing here is "
                f"deleted, and the comments above stay as the record.\n\nRun: {run_url}"
            ),
        },
        {"action": "close_not_planned", "number": existing["number"]},
    ]


class Issues(Protocol):
    def list_by_label(self, label: str) -> list[dict]: ...
    def comments(self, number: int) -> list[str]: ...
    def create(self, title: str, body: str, labels: list[str]) -> int: ...
    def comment(self, number: int, body: str) -> None: ...
    def set_state(self, number: int, state: str, reason: str | None = None) -> None: ...


class RestIssues:
    """The GitHub issues API over urllib, so the job needs no extra dependency."""

    def __init__(self, repo: str, token: str) -> None:
        self.repo = repo
        self.token = token

    def _call(self, method: str, path: str, payload: dict | None = None) -> object:
        request = urllib.request.Request(  # noqa: S310 - a fixed https API host
            f"{API}{path}",
            method=method,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return json.loads(response.read() or "null")

    def list_by_label(self, label: str) -> list[dict]:
        # state=all, and listed rather than searched: the search index lags by up to a
        # minute, so an issue a job just created can come back absent and be created a
        # second time. Closed issues are listed because a notebook that flaps belongs
        # in one thread that reopens, not in a new issue every Monday.
        out: list[dict] = []
        page = 1
        while True:
            batch = self._call(
                "GET",
                f"/repos/{self.repo}/issues?state=all&labels={label}&per_page=100&page={page}",
            )
            if not batch:
                return out
            out.extend(batch)
            page += 1

    # The comments endpoint returns oldest first and takes no sort parameter, so
    # page 1 is the OLDEST hundred. Reading only that page would miss every recent
    # run URL on a thread with more than a hundred comments, and a re-run of a job
    # would post a duplicate - which is the thing this lookup exists to prevent.
    # Paginate to the end instead, with a cap so a pathological thread cannot turn
    # one dedup check into hundreds of requests.
    COMMENT_PAGE_CAP = 20

    def comments(self, number: int) -> list[str]:
        out: list[str] = []
        for page in range(1, self.COMMENT_PAGE_CAP + 1):
            batch = self._call(
                "GET",
                f"/repos/{self.repo}/issues/{number}/comments?per_page=100&page={page}",
            )
            if not batch:
                return out
            out.extend(c.get("body") or "" for c in batch)
            if len(batch) < 100:
                return out
        print(
            f"::warning::#{number} has more than {self.COMMENT_PAGE_CAP * 100} comments; "
            f"only the first were read, so a duplicate comment is possible"
        )
        return out

    def create(self, title: str, body: str, labels: list[str]) -> int:
        created = self._call(
            "POST", f"/repos/{self.repo}/issues", {"title": title, "body": body, "labels": labels}
        )
        return created["number"]

    def comment(self, number: int, body: str) -> None:
        self._call("POST", f"/repos/{self.repo}/issues/{number}/comments", {"body": body})

    def set_state(self, number: int, state: str, reason: str | None = None) -> None:
        payload: dict = {"state": state}
        if reason:
            payload["state_reason"] = reason
        self._call("PATCH", f"/repos/{self.repo}/issues/{number}", payload)


def apply(actions: list[dict], client: Issues) -> list[str]:
    """Perform the planned writes and return one log line each."""
    log = []
    for action in actions:
        kind = action["action"]
        if kind == "create":
            number = client.create(action["title"], action["body"], action["labels"])
            log.append(f"opened #{number} {action['title']}")
        elif kind == "comment":
            client.comment(action["number"], action["body"])
            log.append(f"commented #{action['number']}")
        elif kind == "reopen":
            client.set_state(action["number"], "open")
            log.append(f"reopened #{action['number']}")
        elif kind == "close":
            client.set_state(action["number"], "closed", "completed")
            log.append(f"closed #{action['number']} as completed")
        elif kind == "close_not_planned":
            client.set_state(action["number"], "closed", "not_planned")
            log.append(f"closed #{action['number']} as not planned")
        else:
            msg = f"unknown action {kind!r}"
            raise ValueError(msg)
    return log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--id-style", choices=ID_STYLES, required=True)
    parser.add_argument("--prefix", required=True, help="issue title prefix and label")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--run-url", default="")
    parser.add_argument("--intro", default="")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the planned writes and make none"
    )
    args = parser.parse_args(argv)

    try:
        outcomes = read_outcomes(args.junit, args.id_style)
    except ReportMissing as exc:
        # Not a pass. Exit non-zero so the job shows it rather than reporting a clean
        # week on a report that was never written.
        print(f"::error::{exc}")
        return 2
    print(json.dumps(outcomes, indent=2))

    if args.dry_run:
        for action in reconcile(outcomes, [], args.prefix, args.run_url, args.intro):
            print(f"would {action['action']}: {action.get('title') or action.get('number')}")
        return 0

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token or not args.repo:
        print("::error::GITHUB_TOKEN and --repo are required to touch issues")
        return 2

    client = RestIssues(args.repo, token)
    try:
        issues = client.list_by_label(args.prefix)
    except (OSError, ValueError) as exc:
        # OSError rather than URLError alone: a socket timeout or a DNS failure from
        # urlopen arrives as a bare OSError, and ValueError covers a body that is not
        # the JSON the API promised. Any of them leaves the issue state unknown.
        # Acting on "no issue exists" would open a duplicate of every standing one,
        # so nothing is written.
        print(f"::error::could not list `{args.prefix}` issues, so nothing was changed: {exc}")
        return 2

    # Comments are fetched only for the issues this run could write to - the ones
    # whose title matches a failing id, plus the legacy thread - so a quiet week
    # costs no extra requests.
    wanted = {f"{args.prefix}: {case_id}" for case_id in outcomes["failed"]}
    legacy = LEGACY_IDS.get(args.prefix)
    if legacy:
        wanted.add(f"{args.prefix}: {legacy}")
    comments: dict[int, list[str]] = {}
    for issue in issues:
        if issue.get("title") in wanted and not issue.get("pull_request"):
            try:
                comments[issue["number"]] = client.comments(issue["number"])
            except (OSError, ValueError) as exc:
                # Not knowing is not the same as knowing it was not reported, but the
                # cost of being wrong here is one duplicate comment, so the run goes
                # on rather than failing over a notification.
                print(f"::warning::could not read comments on #{issue['number']}: {exc}")

    plan = reconcile(outcomes, issues, args.prefix, args.run_url, args.intro, comments)
    for line in apply(plan, client):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
