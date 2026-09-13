"""How many merges back each job's last completed conclusion on `main` sits.

The run-level conclusion is the wrong instrument for "is `main` green", and this
script exists because reading it that way was wrong for eleven hours on
2026-09-12. `Tests` collapses a burst of merges to the newest tree
(`.github/workflows/test.yml`), so a superseded run ends `cancelled` and carries no
verdict. When merges land faster than the slowest jobs finish, every run in the
list reads `cancelled` and the newest run that *does* carry a conclusion can be
many merges old - on that night it was a `failure` ten merges back, on one job,
already superseded by a pass. A summary that reads the top of the run list sees
either nothing or that stale red, and neither is the state of `main`.

The jobs are the instrument. A job that completed on the current tip has reported
on it whatever its run did afterwards, and a job whose newest completed conclusion
belongs to an older commit has reported on nothing since - which is a coverage
gap, not a failure, and it is invisible in every view that reports colours.

Lag is counted in merges rather than in hours on purpose: a quiet weekend is not a
gap, and an hour of heavy merging is.

    gh auth status && python .github/scripts/ci_job_lag.py --max-lag 4

Exit status: 0 when every job is within the threshold, 1 when one is not, 2 when
the data could not be read.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass

REPO = "stefan-jansen/machine-learning-for-trading"
WORKFLOW = "Tests"
BRANCH = "main"

# A matrix job that is skipped before its matrix expands is reported under the
# unexpanded expression rather than under a job name. It names no job, so counting it
# would report a permanent gap for something that never runs.
UNEXPANDED = ("${{", "matrix.")

# `cancelled` is the absence of a verdict, which is the whole subject here. `skipped`
# IS a verdict - a job the path filter correctly did not run has reported on that tree.
NO_VERDICT = {"cancelled", None, "", "null"}


@dataclass(frozen=True)
class JobRow:
    sha: str
    name: str
    status: str
    conclusion: str | None
    completed_at: str | None


def newest_verdicts(rows: list[JobRow], main_shas: list[str]) -> dict[str, tuple[str, str, int]]:
    """Per job, the newest completed non-cancelled conclusion and its lag in merges.

    "Newest" is by position in `main_shas`, not by timestamp: a run started earlier can
    finish later, and the question is which commit the verdict is about.
    """
    depth = {sha: i for i, sha in enumerate(main_shas)}
    best: dict[str, tuple[str, str, int]] = {}
    for row in rows:
        if any(token in row.name for token in UNEXPANDED):
            continue
        if row.status != "completed" or row.conclusion in NO_VERDICT:
            continue
        lag = depth.get(row.sha)
        if lag is None:
            continue
        if row.name not in best or lag < best[row.name][2]:
            best[row.name] = (row.sha, row.conclusion, lag)
    return best


def _gh_lines(path: str, jq: str) -> list[str]:
    """`gh api --paginate` with a jq projection, read as lines.

    Not as JSON: `--paginate` concatenates one document per page, so a caller that
    reassembles them has to split on a brace boundary that also occurs inside string
    values. Projecting to lines in `gh` avoids inventing that parser.
    """
    out = subprocess.run(
        ["gh", "api", path, "--paginate", "-q", jq],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(f"gh api {path} failed: {out.stderr.strip()[:400]}")
    return [line for line in out.stdout.splitlines() if line.strip()]


def fetch_rows(runs_to_scan: int) -> tuple[list[JobRow], list[str]]:
    runs = [
        line.split("\t")
        for line in _gh_lines(
            f"repos/{REPO}/actions/runs?branch={BRANCH}&per_page=100",
            r'.workflow_runs[] | select(.name=="' + WORKFLOW + r'") | "\(.id)\t\(.head_sha)"',
        )
    ][:runs_to_scan]

    rows: list[JobRow] = []
    for run_id, sha in runs:
        for line in _gh_lines(
            f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100",
            r'.jobs[] | "\(.name)\t\(.status)\t\(.conclusion)\t\(.completed_at)"',
        ):
            name, status, conclusion, completed_at = line.split("\t")
            rows.append(
                JobRow(
                    sha=sha,
                    name=name,
                    status=status,
                    conclusion=None if conclusion == "null" else conclusion,
                    completed_at=None if completed_at == "null" else completed_at,
                )
            )
    return rows, [sha for _id, sha in runs]


def main_history(depth: int) -> list[str]:
    """`main`'s commits, newest first.

    Tried in order because the three contexts this runs in name the same history
    differently: a working checkout has `origin/main`, an `actions/checkout` of main has
    a local `main`, and a detached checkout has only `HEAD`. A wrong ref is not a
    fallback worth taking, so all three are the same branch or the run fails.
    """
    for ref in (f"origin/{BRANCH}", BRANCH, "HEAD"):
        out = subprocess.run(
            ["git", "log", ref, "--format=%H", f"-{depth}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0 and out.stdout.split():
            return out.stdout.split()
    raise RuntimeError(f"no git history for {BRANCH} (tried origin/{BRANCH}, {BRANCH}, HEAD)")


def report(verdicts: dict[str, tuple[str, str, int]], max_lag: int) -> int:
    if not verdicts:
        print("no completed job conclusions found on main in the scanned window")
        return 1

    width = max(len(name) for name in verdicts)
    over = []
    for name in sorted(verdicts):
        sha, conclusion, lag = verdicts[name]
        flag = "  <-- stale" if lag > max_lag else ""
        print(f"{name:<{width}}  {conclusion:<8} {lag:>3} merges back  {sha[:8]}{flag}")
        if lag > max_lag:
            over.append((name, lag))

    failing = sorted(
        (name, v[0]) for name, v in verdicts.items() if v[1] not in {"success", "skipped"}
    )
    print()
    if failing:
        print(f"{len(failing)} job(s) whose newest verdict is not a pass:")
        for name, sha in failing:
            print(f"  {name} on {sha[:8]}")
    if over:
        print(f"{len(over)} job(s) more than {max_lag} merges behind main's tip:")
        for name, lag in sorted(over, key=lambda pair: -pair[1]):
            print(f"  {name} at {lag}")
        print("Nothing has run these on the trees merged since. See ml4t/agent-workspace#1166.")
        return 1
    print(f"every job has reported within {max_lag} merges of main's tip")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # 4, from the 2026-09-12 measurement: 31 of 37 jobs sat at 1 merge back during a
    # heavy evening and the six longest sat at 5, so 4 separates "a merge landed while
    # the matrix was running" from "these jobs are not converging".
    parser.add_argument("--max-lag", type=int, default=4)
    parser.add_argument("--runs", type=int, default=25, help="how many main Tests runs to scan")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows, run_shas = fetch_rows(args.runs)
        history = main_history(max(len(run_shas) * 2, 40))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    return report(newest_verdicts(rows, history), args.max_lag)


if __name__ == "__main__":
    raise SystemExit(main())
