#!/usr/bin/env python3
"""Refresh the courses, workshops, and free lessons block in README.md.

Every offering this repository advertises is scheduled on Maven, and a date in
a README goes stale silently. The July 30, 2026 reader's guide sat at the top of
this file for four weeks after it ran, and readers kept signing up for it.

So the offerings are not written by hand. The public instructor profile at
https://maven.com/stefan-jansen embeds the schedule as JSON in its Next.js
payload, needs no authentication, and is the same record the Maven pages
themselves render. This script reads it, drops everything already past, and
rewrites one marked region of README.md:

    <!-- offerings:all start -->  ... <!-- offerings:all end -->

There used to be a second region, a one-line "Next free session" callout under the
intro. A cron that runs daily cannot keep a callout about a session that starts at
11:00 ET honest: it reads as "next" for about twenty hours after it has happened,
which it did on the Sep 30 session. The courses section carries the same sessions
with their dates attached, where a past one is visibly past.

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
from datetime import UTC, datetime
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

# What a reader is actually choosing between, keyed by Maven course slug. Maven
# carries no theme field - its own grouping is by format and date, which is the cut
# the three tables used to make. A reader does not arrive asking "what is scheduled
# and what is self-paced"; they arrive asking "is this about trading or about
# agents", so the themes are stated here and the schedule fills them in.
#
# A slug absent from this map lands in OTHER_THEME rather than being dropped: a
# newly launched course must never vanish from the front page because nobody
# updated a dict, which is the failure #1130 and #1132 were both instances of.
THEMES = {
    "ml4t-foundations": "ml4t",
    "research-to-production": "ml4t",
    "ml4t-ai-agents": "ml4t",
    "agent-engineering": "agents",
    "loop-engineering": "coding-agents",
}
OTHER_THEME = "ml4t"

# Rendered in this order, each with the heading and the one line under it.
THEME_ORDER = ["ml4t", "agents", "coding-agents"]
THEME_HEADINGS = {
    "ml4t": (
        "Machine Learning for Trading",
        "The workflow this repository implements, taught end to end.",
    ),
    "agents": (
        "Agentic systems",
        "Designing multi-agent systems whose reasoning can be audited.",
    ),
    "coding-agents": (
        "Coding agents",
        "Getting reliable work out of the agents you code with.",
    ),
}

# Displayed names. Maven's course_name carries the "ML for Trading:" prefix that the
# section heading above the table already supplies, and repeating it four times on
# one page is most of why the offerings block reads as a wall.
DISPLAY_NAMES = {
    "ml4t-foundations": "Foundations",
    "research-to-production": "Research to Production",
}

# One-line positioning per offering, keyed by Maven course slug, for the courses
# whose own course_description is marketing-page prose too long for a table row.
# A course absent here falls back to its course_description, so a newly launched
# cohort never renders an empty cell on the front page; see blurb_for().
BLURBS = {
    "research-to-production": "Take one research idea from a question to a costed, "
    "monitored strategy, with the evidence trail that makes the result checkable.",
    "agent-engineering": "Go from agent fundamentals to a live multi-agent system, "
    "with an evaluation harness that says whether it works.",
    "loop-engineering": "Get reliable work out of coding agents: harness design, "
    "verification, and recovery from a bad run.",
    "ml4t-foundations": "Build the ML for Trading pipeline yourself, end to end, and "
    "the evidence to say what it does and does not establish.",
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


def collect(props: dict, now: datetime) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (upcoming free lessons, upcoming live cohorts, standing offerings).

    The first two are dated and sort soonest first. The third is what a reader can
    buy today that carries no date, and it exists because a table keyed on a start
    date shows nothing for an offering that has none: `ml4t-foundations` sells a
    self-paced seat and was absent from the README entirely, and
    `research-to-production` disappears from it between cohorts, which on
    2026-09-24 was both full courses missing from the page at once.
    """
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

    # Every scheduled cohort, from `course_cohorts`, not one per course from
    # `next_live_cohort`. That field holds a single cohort, so a course with two
    # dates on the calendar advertised only the nearer one: `agent-engineering`
    # was listed on Maven for Oct 3 and Nov 21, 2026 and the README showed Oct 3
    # alone. It is also None while a cohort is running with no successor
    # scheduled, which dropped `research-to-production` out of the table entirely.
    # `course_cohorts` is keyed by `course_id`, so the course record supplies the
    # name, slug and description and the cohort supplies the dates.
    courses_by_id = {
        course.get("course_id") or course.get("id"): course
        for course in props.get("courses", []) + props.get("paid_workshop_courses", [])
    }

    cohorts = []
    for cohort in props.get("course_cohorts", []):
        # A self-paced entry has no start date and belongs in no schedule; an
        # unlisted one is not on sale. Neither is a date a reader can act on.
        if cohort.get("visibility") != "listed" or cohort.get("type") != "live":
            continue
        course = courses_by_id.get(cohort.get("course_id"))
        if course is None:
            continue
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
                "description": (course.get("course_description") or "").strip(),
            }
        )

    # What a reader can act on with no date attached. A listed self-paced cohort is
    # one: the seat is on sale and the start is whenever the reader starts. A full
    # course with nothing scheduled is the other, because the course page keeps
    # selling and collecting a waitlist while a cohort runs. A workshop is not: it
    # is a single event, so with no date there is nothing to point anyone at.
    scheduled = {c["slug"] for c in cohorts}
    standing = []
    for course in courses_by_id.values():
        slug = course.get("course_slug")
        self_paced = any(
            cohort.get("course_id") == (course.get("course_id") or course.get("id"))
            and cohort.get("type") == "self_paced"
            and cohort.get("visibility") == "listed"
            for cohort in props.get("course_cohorts", [])
        )
        if self_paced:
            mode = "Self-paced, start any time"
        elif slug not in scheduled and course.get("course_format") == "full_course":
            mode = "Next cohort not yet scheduled"
        else:
            continue
        standing.append(
            {
                "title": course["course_name"],
                "slug": slug,
                "url": f"https://maven.com/stefan-jansen/{slug}",
                "mode": mode,
                "description": (course.get("course_description") or "").strip(),
            }
        )

    return (
        sorted(lessons, key=lambda x: x["start"]),
        sorted(cohorts, key=lambda x: x["start"]),
        sorted(standing, key=lambda x: x["title"]),
    )


# A free lesson has no course slug to key on - Maven gives it an opaque one
# (`efe730`) - so its theme is read off the title. Most specific first: a session
# about coding agents says so, a session that names trading belongs with the
# trading material however many times it says "agent", and what is left that
# mentions an agent at all is agentic systems.
LESSON_THEME_KEYWORDS = [
    ("coding agent", "coding-agents"),
    ("trading", "ml4t"),
    ("agent", "agents"),
]


def theme_for_lesson(title: str) -> str:
    lowered = title.lower()
    for keyword, theme in LESSON_THEME_KEYWORDS:
        if keyword in lowered:
            return theme
    return OTHER_THEME


def display_name(slug: str, course_name: str) -> str:
    return DISPLAY_NAMES.get(slug, course_name)


def blurb_for(cohort: dict) -> str:
    """The `What you leave with` cell, never empty.

    A hand-written BLURBS entry wins. Absent one - which is every newly launched
    cohort, since the table is edited by hand and the schedule is not - fall back
    to the profile's own course_description, first sentence only. PR #887 shipped
    an empty cell for `ml4t-ai-agents` this way, on the front page of the repo.
    """
    written = BLURBS.get(cohort["slug"])
    if written:
        return written
    description = cohort.get("description", "")
    if not description:
        raise SystemExit(
            f"{cohort['slug']}: no BLURBS entry and no course_description on the Maven "
            "profile, so the offerings table would ship an empty cell. Add a BLURBS entry."
        )
    first, _, _ = description.partition(". ")
    return first.rstrip(".") + "."


def render_all(lessons: list[dict], cohorts: list[dict], standing: list[dict]) -> str:
    """One table per theme, each followed by that theme's free sessions.

    The three tables this replaces were cut by scheduling mechanics - dated cohort,
    undated course, free lesson - which is the generator's own distinction and not a
    question anybody arrives with. It also buried the self-paced course: `Foundations`
    is the one most readers of this repository want and it sat alone under a heading
    reading "Courses with no date on the calendar", which says nothing is happening.

    A `When` cell therefore carries a date span, `Self-paced` or `Waitlist`, and the
    standing offerings sort first so a reader can start today without reading a date
    column to find out they can.
    """
    by_theme: dict[str, dict[str, list]] = {
        theme: {"offerings": [], "lessons": []} for theme in THEME_ORDER
    }

    for item in standing:
        theme = THEMES.get(item["slug"], OTHER_THEME)
        by_theme[theme]["offerings"].append(
            {
                "sort": (0, item["title"]),
                "when": "Self-paced" if "Self-paced" in item["mode"] else "Waitlist",
                "title": display_name(item["slug"], item["title"]),
                "url": item["url"],
                "blurb": blurb_for(item),
            }
        )
    # One row per course, carrying every date it is scheduled for. Advertising only
    # the nearer of two cohorts was #1130; advertising both as two rows with the same
    # title and the same blurb is the same information twice.
    merged: dict[str, dict] = {}
    for item in cohorts:
        row = merged.get(item["slug"])
        if row is None:
            theme = THEMES.get(item["slug"], OTHER_THEME)
            row = merged[item["slug"]] = {
                "theme": theme,
                "sort": (1, item["start"].isoformat()),
                "spans": [],
                "title": display_name(item["slug"], item["title"]),
                "url": item["url"],
                "blurb": blurb_for(item),
            }
        row["spans"].append(span(item["start"], item["end"]))
    for row in merged.values():
        theme = row.pop("theme")
        row["when"] = " · ".join(row.pop("spans"))
        by_theme[theme]["offerings"].append(row)
    for lesson in lessons:
        by_theme[theme_for_lesson(lesson["title"])]["lessons"].append(lesson)

    out: list[str] = []
    for theme in THEME_ORDER:
        group = by_theme[theme]
        if not group["offerings"] and not group["lessons"]:
            continue
        heading, standfirst = THEME_HEADINGS[theme]
        out += [f"### {heading}", "", standfirst, ""]

        if group["offerings"]:
            out += [
                "| When | Offering | What you leave with |",
                "|------|----------|---------------------|",
            ]
            for item in sorted(group["offerings"], key=lambda x: x["sort"]):
                out.append(
                    f"| {item['when']} | [{item['title']}]({item['url']}) | {item['blurb']} |"
                )
            out.append("")

        if group["lessons"]:
            sessions = " · ".join(
                f"[{x['title']}]({x['url']}) ({when(x['start'])})" for x in group["lessons"]
            )
            out += [
                f"**Free live sessions**, with the recording sent to everyone who registers: {sessions}",
                "",
            ]

    if not out:
        out += [
            f"Nothing is on the calendar right now. [Courses and workshops]({COURSES_URL}) "
            "lists new dates as they are scheduled.",
            "",
        ]

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
    lessons, cohorts, standing = collect(props, now)
    allblock = render_all(lessons, cohorts, standing)

    if args.show:
        print(allblock)
        return 0

    original = README.read_text()
    updated = splice(original, "all", allblock)

    if updated == original:
        print("README offerings are current")
        return 0
    if args.check:
        print(
            "README offerings are stale; run .github/scripts/update_offerings.py", file=sys.stderr
        )
        return 1

    README.write_text(updated)
    print(
        f"README updated: {len(cohorts)} cohorts, {len(standing)} without a date, "
        f"{len(lessons)} free lessons"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
