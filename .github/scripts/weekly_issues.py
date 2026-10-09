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
- An unchanged failure comments once per run and does not duplicate the issue.
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


def reconcile(
    outcomes: dict[str, list[str]],
    issues: list[dict],
    prefix: str,
    run_url: str,
    intro: str = "",
) -> list[dict]:
    """The writes this run should make, as data, so a test can read them.

    Returns a list of ``{"action", ...}`` in the order they must be applied: a reopen
    precedes the comment that explains it, and a closing comment precedes the close.
    """
    actions: list[dict] = []

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
        if existing["state"] == "closed":
            actions.append({"action": "reopen", "number": existing["number"]})
            actions.append(
                {
                    "action": "comment",
                    "number": existing["number"],
                    "body": f"Failed again after closing: {run_url}",
                }
            )
        else:
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

    return actions


class Issues(Protocol):
    def list_by_label(self, label: str) -> list[dict]: ...
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
            log.append(f"closed #{action['number']}")
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

    for line in apply(reconcile(outcomes, issues, args.prefix, args.run_url, args.intro), client):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
