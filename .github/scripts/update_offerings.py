#!/usr/bin/env python3
"""Refresh the courses, workshops, and free lessons block in README.md.

Every offering this repository advertises is scheduled on Maven, and a date in
a README goes stale silently. The July 30, 2026 reader's guide sat at the top of
this file for four weeks after it ran, and readers kept signing up for it.

So the offerings are not written by hand. The public instructor profile at
https://maven.com/stefan-jansen embeds the schedule as JSON in its Next.js
payload, needs no authentication, and is the same record the Maven pages
themselves render. This script reads it, drops everything already past, and
rewrites two marked regions of README.md:

    <!-- offerings:next start --> ... <!-- offerings:next end -->
    <!-- offerings:all start -->  ... <!-- offerings:all end -->

Nothing outside those markers is touched, so the file stays hand-edited
everywhere else.

    update_offerings.py            # rewrite README.md in place
    update_offerings.py --check    # exit 1 if the block is out of date
    update_offerings.py --print    # render to stdout, touch nothing

Times come back as UTC instants and are rendered in both US Eastern, which is
what every other channel says, and UTC, which a reader in any timezone can
convert without knowing US daylight-saving rules.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PROFILE_URL = "https://maven.com/stefan-jansen"
COURSES_URL = (
    "https://ml4trading.io/courses/"
    "?utm_source=github&utm_medium=readme&utm_campaign=ml4t3e&utm_content=offerings"
)
README = Path(__file__).resolve().parents[2] / "README.md"
EASTERN = ZoneInfo("America/New_York")
USER_AGENT = "ml4t-readme-offerings"

NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)

# One-line positioning per offering, keyed by Maven course slug. The profile's
# own course_description is marketing-page prose, too long for a table row.
BLURBS = {
    "research-to-production": "Take one research idea from a question to a costed, "
    "monitored strategy, with the evidence trail that makes the result checkable.",
    "agent-engineering": "Build a multi-agent forecasting system whose reasoning is "
    "auditable end to end.",
    "loop-engineering": "Get reliable work out of coding agents: harness design, "
    "verification, and recovery from a bad run.",
}


def fetch_profile(url: str = PROFILE_URL) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", "ignore")
    match = NEXT_DATA_RE.search(body)
    if not match:
        raise SystemExit(f"no __NEXT_DATA__ payload in {url}; the page layout changed")
    return json.loads(match.group(1))["props"]["pageProps"]


def parse_instant(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def when(start: datetime, *, long: bool = False) -> str:
    """`Wed, Sep 2, 12:00 PM ET / 16:00 UTC`, or the spelled-out form."""
    local = start.astimezone(EASTERN)
    day = local.strftime("%A, %B %-d, %Y" if long else "%a, %b %-d")
    return f"{day}, {local.strftime('%-I:%M %p')} ET / {start.strftime('%H:%M')} UTC"


def span(start: datetime, end: datetime | None) -> str:
    """`Sep 16 – Dec 2, 2026` across days, `Sep 19, 2026` within one."""
    first = start.astimezone(EASTERN)
    if end is None or end.astimezone(EASTERN).date() == first.date():
        return first.strftime("%b %-d, %Y")
    return f"{first.strftime('%b %-d')} – {end.astimezone(EASTERN).strftime('%b %-d, %Y')}"


def collect(props: dict, now: datetime) -> tuple[list[dict], list[dict]]:
    """Return (upcoming free lessons, upcoming paid cohorts), soonest first."""
    lessons = []
    for item in props.get("free_items", {}).get("items", []):
        start = parse_instant(item.get("start_datetime"))
        if start is None or start <= now:
            continue
        lessons.append(
            {
                "title": item["title"],
                "url": f"https://maven.com/p/{item['slug']}",
                "start": start,
                "minutes": item.get("duration_minutes"),
            }
        )

    cohorts = []
    for course in props.get("courses", []) + props.get("paid_workshop_courses", []):
        cohort = course.get("next_live_cohort") or {}
        start = parse_instant(cohort.get("start_date"))
        if start is None or start <= now:
            continue
        cohorts.append(
            {
                "title": course["course_name"],
                "slug": course["course_slug"],
                "url": f"https://maven.com/stefan-jansen/{course['course_slug']}",
                "start": start,
                "end": parse_instant(cohort.get("end_date")),
                "format": course.get("course_format"),
            }
        )

    return sorted(lessons, key=lambda x: x["start"]), sorted(cohorts, key=lambda x: x["start"])


def render_next(lessons: list[dict]) -> str:
    """The one-line callout under the intro. Free lesson if there is one."""
    if not lessons:
        return (
            f"> **Live sessions and cohorts:** [courses and workshops]({COURSES_URL}) "
            "lists what is scheduled next, and the "
            "[**Insights** newsletter](https://insights.ml4trading.io/) carries the "
            "research between them."
        )
    nxt = lessons[0]
    minutes = f"{nxt['minutes']}-minute " if nxt.get("minutes") else ""
    return (
        f"> **Next free session:** [{nxt['title']}]({nxt['url']}), a {minutes}live "
        f"session on **{when(nxt['start'], long=True)}**. "
        f"[All courses, workshops, and free lessons]({COURSES_URL})."
    )


def render_all(lessons: list[dict], cohorts: list[dict]) -> str:
    out: list[str] = []

    if cohorts:
        out += [
            "**Cohorts and workshops.** Live, scheduled, and worked through with direct "
            "feedback on your own research.",
            "",
            "| Starts | Offering | What you leave with |",
            "|--------|----------|---------------------|",
        ]
        for c in cohorts:
            blurb = BLURBS.get(c["slug"], "")
            out.append(f"| {span(c['start'], c['end'])} | [{c['title']}]({c['url']}) | {blurb} |")
        out.append("")

    if lessons:
        out += [
            "**Free live sessions.** Thirty minutes to an hour, no cost, recording sent "
            "to everyone who registers.",
            "",
            "| When | Session |",
            "|------|---------|",
        ]
        for lesson in lessons:
            out.append(f"| {when(lesson['start'])} | [{lesson['title']}]({lesson['url']}) |")
        out.append("")

    if not cohorts and not lessons:
        out.append(
            f"Nothing is on the calendar right now. [Courses and workshops]({COURSES_URL}) "
            "lists new dates as they are scheduled."
        )
        out.append("")

    out.append(
        "*Between cohorts, the [**Insights** newsletter](https://insights.ml4trading.io/) "
        "covers the same ground weekly, source by source.*"
    )
    return "\n".join(out)


def splice(text: str, name: str, body: str) -> str:
    start, end = f"<!-- offerings:{name} start -->", f"<!-- offerings:{name} end -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise SystemExit(f"markers {start} / {end} not found in {README}")
    return pattern.sub(f"{start}\n{body}\n{end}", text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if README is out of date")
    ap.add_argument("--print", dest="show", action="store_true", help="render to stdout only")
    args = ap.parse_args()

    try:
        props = fetch_profile()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"could not read {PROFILE_URL}: {exc}", file=sys.stderr)
        return 2

    now = datetime.now(UTC)
    lessons, cohorts = collect(props, now)
    nxt, allblock = render_next(lessons), render_all(lessons, cohorts)

    if args.show:
        print(nxt, "", allblock, sep="\n")
        return 0

    original = README.read_text()
    updated = splice(splice(original, "next", nxt), "all", allblock)

    if updated == original:
        print("README offerings are current")
        return 0
    if args.check:
        print(
            "README offerings are stale; run .github/scripts/update_offerings.py", file=sys.stderr
        )
        return 1

    README.write_text(updated)
    print(f"README updated: {len(cohorts)} cohorts, {len(lessons)} free lessons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
