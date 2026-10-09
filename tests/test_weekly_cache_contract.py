"""The weekly checkpoint cache has to survive a red run and a quiet week.

Two defects this pins, both of which left a green-looking workflow that re-downloaded
1.659 GB of checkpoints:

`actions/cache` declares ``post-if: success()``, so its save step runs only when the
job succeeded. The weekly job fails on exactly the notebook whose download the cache
exists to remove, so a cold run populated the cache in its prefetch step and then
discarded it, every week. The fix is ``actions/cache/restore`` plus
``actions/cache/save`` placed before pytest, and what this file checks is the ordering
rather than the words: a save after pytest is a save that a failing notebook can skip.

GitHub removes "cache entries that have not been accessed in over 7 days". A workflow
scheduled only on Monday touches its entry every 168 hours plus the scheduler's delay,
so the entry can expire between two consecutive green runs, and one missed week
guarantees it. The fix is a restore-only job on the days between, and what this file
checks is the actual gap the crons produce, including with one scheduled run dropped.

The routing is checked by evaluating each job's real `if` expression against the event
payloads GitHub delivers, not by matching its text. The evaluator refuses syntax it
does not understand, because an expression it silently read as false would make every
assertion below pass.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "weekly-external.yml"
WF = yaml.safe_load(WORKFLOW.read_text())
# `on:` is the YAML 1.1 boolean true, which is why this reads WF[True].
TRIGGERS = WF[True]
JOBS = WF["jobs"]
NOTEBOOK_JOBS = ("weekly-chapters", "weekly-drift")
TOUCH_JOB = "hf-cache-touch"
MONDAY_CRON = "0 6 * * 1"


# --------------------------------------------------------------------------- routing


class Unsupported(Exception):
    """The expression used syntax this evaluator does not implement."""


def evaluate(expr: str, context: dict[str, object]) -> bool:
    """Evaluate the subset of GitHub's expression language these `if`s use.

    Supports `||`, `&&`, `==`, `!=`, parentheses, single-quoted literals and dotted
    context lookups. Anything else raises, so a condition rewritten into syntax this
    does not cover fails the suite instead of reading as false.
    """
    expr = " ".join(expr.split())
    value, rest = _or_expr(_tokenize(expr), context)
    if rest:
        raise Unsupported(f"trailing tokens {rest!r} in {expr!r}")
    return value


_TOKEN = re.compile(r"\s*(\|\||&&|==|!=|\(|\)|'[^']*'|[A-Za-z_][\w.]*)")


def _tokenize(expr: str) -> list[str]:
    tokens, pos = [], 0
    while pos < len(expr):
        match = _TOKEN.match(expr, pos)
        if not match:
            raise Unsupported(f"cannot tokenize {expr[pos:]!r}")
        tokens.append(match.group(1))
        pos = match.end()
    return tokens


def _or_expr(tokens: list[str], ctx: dict[str, object]) -> tuple[bool, list[str]]:
    value, tokens = _and_expr(tokens, ctx)
    while tokens and tokens[0] == "||":
        right, tokens = _and_expr(tokens[1:], ctx)
        value = value or right
    return value, tokens


def _and_expr(tokens: list[str], ctx: dict[str, object]) -> tuple[bool, list[str]]:
    value, tokens = _comparison(tokens, ctx)
    while tokens and tokens[0] == "&&":
        right, tokens = _comparison(tokens[1:], ctx)
        value = value and right
    return value, tokens


def _comparison(tokens: list[str], ctx: dict[str, object]) -> tuple[bool, list[str]]:
    if not tokens:
        raise Unsupported("expression ended early")
    if tokens[0] == "(":
        value, tokens = _or_expr(tokens[1:], ctx)
        if not tokens or tokens[0] != ")":
            raise Unsupported("unbalanced parenthesis")
        return value, tokens[1:]
    left, tokens = _operand(tokens, ctx)
    if not tokens or tokens[0] not in ("==", "!="):
        raise Unsupported(f"bare operand {left!r}: only == and != are implemented")
    op, (right, tokens) = tokens[0], _operand(tokens[1:], ctx)
    return (left == right) if op == "==" else (left != right), tokens


def _operand(tokens: list[str], ctx: dict[str, object]) -> tuple[object, list[str]]:
    if not tokens:
        raise Unsupported("operand expected")
    token = tokens[0]
    if token.startswith("'"):
        return token.strip("'"), tokens[1:]
    if token in ("(", ")", "||", "&&", "==", "!="):
        raise Unsupported(f"operand expected, got {token!r}")
    # An unset context value is null in GitHub's language, and `null != 'touch'` is
    # true there as it is here. That is what makes a schedule event skip the chapters
    # job through `inputs.mode`, so the default has to be None rather than a KeyError.
    value: object = ctx
    for part in token.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value, tokens[1:]


def _context(event: str, schedule: str | None = None, mode: str | None = None) -> dict:
    return {
        "github": {"event_name": event, "event": {"schedule": schedule}},
        "inputs": {"mode": mode},
    }


def runs(job: str, context: dict[str, object]) -> bool:
    condition = JOBS[job].get("if")
    return True if condition is None else evaluate(str(condition), context)


def test_the_evaluator_refuses_what_it_cannot_read() -> None:
    """Otherwise every routing assertion below would pass on an expression it ignored."""
    ctx = _context("schedule", MONDAY_CRON)
    assert evaluate("github.event.schedule == '0 6 * * 1'", ctx) is True
    assert evaluate("github.event.schedule != '0 6 * * 1'", ctx) is False
    assert evaluate("github.event_name == 'push' || github.event_name == 'schedule'", ctx) is True
    assert evaluate("(github.event_name == 'schedule') && inputs.mode != 'touch'", ctx) is True
    for bad in (
        "success()",
        "!cancelled()",
        "github.ref == 'x' && contains(github.ref, 'y')",
        "github.event_name",
        "startsWith(github.ref, 'refs/')",
    ):
        with pytest.raises(Unsupported):
            evaluate(bad, ctx)


@pytest.mark.parametrize(
    ("event", "schedule", "mode", "chapters", "touch"),
    [
        ("schedule", "0 6 * * 1", None, True, False),
        ("schedule", "0 6 * * 3", None, False, True),
        ("schedule", "0 6 * * 5", None, False, True),
        ("workflow_dispatch", None, "full", True, False),
        ("workflow_dispatch", None, "touch", False, True),
        # A dispatch from the API may omit a defaulted input entirely.
        ("workflow_dispatch", None, None, True, False),
    ],
)
def test_each_event_routes_to_exactly_the_jobs_it_should(
    event: str, schedule: str | None, mode: str | None, chapters: bool, touch: bool
) -> None:
    """The midweek crons must not run the notebooks, and Monday must not skip them.

    A midweek cron that reached the notebook jobs would turn a 60-minute weekly run
    into three, and a Monday that reached only the touch job would never save a cache.
    """
    ctx = _context(event, schedule, mode)
    assert runs("weekly-chapters", ctx) is chapters
    assert runs("weekly-drift", ctx) is chapters
    assert runs(TOUCH_JOB, ctx) is touch


def test_every_scheduled_cron_routes_somewhere() -> None:
    """A cron no job answers is a scheduled run that does nothing and reports green."""
    for cron in [entry["cron"] for entry in TRIGGERS["schedule"]]:
        ctx = _context("schedule", cron)
        assert any(runs(job, ctx) for job in (*NOTEBOOK_JOBS, TOUCH_JOB)), cron


# --------------------------------------------------------------------------- retention

WEEK = 7 * 24 * 60
EVICTION_DAYS = 7  # "not been accessed in over 7 days", GitHub's documented policy


def _minutes_into_week(cron: str) -> int:
    minute, hour, dom, month, dow = cron.split()
    if dom != "*" or month != "*" or not dow.isdigit():
        raise Unsupported(f"{cron!r} is not a plain weekly cron; the gap maths assumes one")
    # cron day-of-week is 0=Sunday; measure the week from Monday 00:00.
    return ((int(dow) - 1) % 7) * 24 * 60 + int(hour) * 60 + int(minute)


def _max_gap_days(occurrences: list[int]) -> float:
    assert occurrences
    points = sorted(occurrences)
    gaps = [b - a for a, b in zip(points, points[1:])]
    gaps.append(WEEK - points[-1] + points[0])  # across the week boundary
    return max(gaps) / (24 * 60)


def test_the_schedule_touches_the_cache_inside_the_eviction_window() -> None:
    crons = [entry["cron"] for entry in TRIGGERS["schedule"]]
    occurrences = [_minutes_into_week(c) for c in crons]
    assert _max_gap_days(occurrences) < EVICTION_DAYS, (
        f"{crons} leaves the cache untouched for {_max_gap_days(occurrences)} days"
    )


def test_one_missed_scheduled_run_still_stays_inside_the_window() -> None:
    """GitHub drops scheduled runs under load, and a missed week was the old failure.

    With Monday alone the gap after one miss is 14 days; the assertion is what forces
    more than one touch day, and it leaves room for the scheduler's queueing delay.
    """
    crons = [entry["cron"] for entry in TRIGGERS["schedule"]]
    occurrences = [_minutes_into_week(c) for c in crons]
    worst = max(
        _max_gap_days(occurrences[:i] + occurrences[i + 1 :]) for i in range(len(occurrences))
    )
    assert worst < EVICTION_DAYS, f"dropping one run of {crons} leaves a {worst}-day gap"


# ------------------------------------------------------------------- save and identity


def _steps(job: str) -> list[dict]:
    return JOBS[job]["steps"]


def _index(job: str, predicate) -> int:
    for i, step in enumerate(_steps(job)):
        if predicate(step):
            return i
    raise AssertionError(f"{job} has no step matching {predicate}")


def _uses(prefix: str):
    return lambda step: str(step.get("uses", "")).startswith(prefix)


def test_nothing_uses_the_bare_cache_action() -> None:
    """`actions/cache@` saves in a post step declared `post-if: success()`.

    Under it, a weekly run whose notebooks fail - the state this branch exists to fix -
    populates the cache in its prefetch and then drops it on the floor.
    """
    text = WORKFLOW.read_text()
    assert "actions/cache@" not in text
    assert "actions/cache/restore@" in text
    assert "actions/cache/save@" in text


def test_the_cache_is_saved_before_the_notebooks_can_fail() -> None:
    job = "weekly-chapters"
    prefetch = _index(job, lambda s: "hf_prefetch.py fetch" in str(s.get("run", "")))
    save = _index(job, _uses("actions/cache/save"))
    pytest_step = _index(job, lambda s: s.get("id") == "pytest_chapters")
    assert prefetch < save < pytest_step, (
        "the save has to sit between the prefetch that populates the cache and the "
        "pytest step that can fail, or a red run loses the download again"
    )


def test_the_save_does_not_depend_on_the_notebook_outcome() -> None:
    """Its only condition is the cache miss; a hit has nothing new to store."""
    save = _steps("weekly-chapters")[_index("weekly-chapters", _uses("actions/cache/save"))]
    condition = str(save.get("if", ""))
    assert "cache-hit" in condition
    for forbidden in ("pytest", "success()", "failure()", "always()"):
        assert forbidden not in condition, f"the save reads {forbidden} and can be skipped"


CACHE_SITES = (
    ("weekly-chapters", "actions/cache/restore"),
    ("weekly-chapters", "actions/cache/save"),
    (TOUCH_JOB, "actions/cache/restore"),
)


def test_restore_and_save_and_the_touch_job_name_one_entry() -> None:
    """Three sites, one key and one path: a touch of a different entry touches nothing."""
    sites = []
    for job, prefix in CACHE_SITES:
        step = _steps(job)[_index(job, _uses(prefix))]
        sites.append((step["with"]["key"], step["with"]["path"]))
    assert len(set(sites)) == 1, f"the three cache steps disagree: {sites}"
    key, path = sites[0]
    assert "hf_prefetch.py key" in str(_steps(TOUCH_JOB)), "the touch job invents its own key"
    assert "${{ steps.hfkey.outputs.key }}" in key
    assert ".hf-cache" in path


def test_the_cache_path_is_a_literal_the_two_jobs_resolve_the_same_way() -> None:
    """An entry is addressed by its key AND by a version hashed from the `path` input.

    So two jobs that agree on the key and differ on the path address different entries.
    `${{ github.workspace }}` is exactly that difference: it is /__w/<repo>/<repo>
    inside a job container and /home/runner/work/<repo>/<repo> on a host runner, so a
    host-based touch job would have missed Monday's entry every week while reporting a
    plausible run. Equal YAML text is not enough to catch it, which is why the
    assertion is on the absence of any expression in the path.
    """
    for job, prefix in CACHE_SITES:
        path = _steps(job)[_index(job, _uses(prefix))]["with"]["path"]
        assert "${{" not in path, (
            f"{job}'s {prefix} path is an expression ({path!r}); two jobs can resolve it "
            "differently and then they are not sharing a cache entry"
        )
        assert not path.startswith("/"), f"{job}'s {prefix} path is absolute: {path!r}"


def test_the_jobs_that_share_the_entry_share_a_container() -> None:
    """The other half of the cache version is the compression method.

    actions/cache picks zstd or gzip from what the environment has, so a host runner
    and this image can hash different versions from the same path. Running the touch
    job in the image that saved the entry settles the path and the compressor at once.
    """
    images = {JOBS[job].get("container", {}).get("image") for job, _ in CACHE_SITES}
    assert len(images) == 1 and None not in images, (
        f"the jobs sharing the cache entry run in different environments: {images}"
    )


def test_the_touch_job_reads_the_key_from_the_manifest_like_the_notebook_job() -> None:
    """A second copy of the key would drift from the first the next time a pin moves."""
    for job in ("weekly-chapters", TOUCH_JOB):
        step = _steps(job)[_index(job, lambda s: s.get("id") == "hfkey")]
        assert "hf_prefetch.py key" in str(step["run"])


def test_the_touch_job_only_restores() -> None:
    """It exists to update a last-accessed date, not to run or save anything."""
    steps = _steps(TOUCH_JOB)
    assert not any(_uses("actions/cache/save")(s) for s in steps)
    assert not any("pytest" in str(s.get("run", "")) for s in steps)
    assert not any("hf_prefetch.py fetch" in str(s.get("run", "")) for s in steps)
    restore = steps[_index(TOUCH_JOB, _uses("actions/cache/restore"))]
    # lookup-only is not documented to update the last-accessed date. A full restore
    # downloads the archive, which is unambiguously an access.
    assert restore["with"].get("lookup-only") in (None, False)
    assert restore["with"].get("fail-on-cache-miss") in (None, False)


def test_a_miss_in_the_touch_job_is_reported_and_not_fatal() -> None:
    """A miss is the normal state after a pin moves; Monday repopulates it."""
    report = _steps(TOUCH_JOB)[-1]
    run = str(report["run"])
    assert "MISS" in run and "::warning::" in run
    assert "last_accessed_at" in run, (
        "the job has to print the date it claims to move, or its effect stays unverified"
    )
