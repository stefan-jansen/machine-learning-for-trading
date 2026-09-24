"""`collect` reads every scheduled cohort, not one per course.

The README's cohort table is generated from the Maven profile. Until 2026-09-24 it
was built from each course's `next_live_cohort`, which holds a single cohort, so a
course with two dates on the calendar advertised only the nearer one and a course
whose current cohort was already running advertised nothing at all. Both shapes are
in the live payload, and neither is visible in the rendered block, which is why this
ran for months without anyone noticing.
"""

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "update_offerings", REPO / ".github" / "scripts" / "update_offerings.py"
)
offerings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(offerings)

NOW = datetime(2026, 9, 24, tzinfo=UTC)


def _course(course_id: int, slug: str, name: str, next_live: dict | None = None) -> dict:
    return {
        "course_id": course_id,
        "course_slug": slug,
        "course_name": name,
        "course_description": f"{name}. Second sentence.",
        "next_live_cohort": next_live,
    }


def _cohort(course_id: int, start: str, **over) -> dict:
    base = {
        "course_id": course_id,
        "start_date": start,
        "end_date": start,
        "visibility": "listed",
        "type": "live",
        "name": "Cohort",
    }
    base.update(over)
    return base


def _props(courses: list[dict], cohorts: list[dict]) -> dict:
    return {
        "courses": [],
        "paid_workshop_courses": courses,
        "course_cohorts": cohorts,
        "free_items": {"items": []},
    }


def test_two_future_cohorts_of_one_course_both_render():
    """The shape that hid `agent-engineering` Cohort 4, Nov 21 2026."""
    props = _props(
        [_course(1, "agent-engineering", "Agents", {"start_date": "2026-10-03T14:00:00Z"})],
        [
            _cohort(1, "2026-10-03T14:00:00Z"),
            _cohort(1, "2026-11-21T15:00:00Z"),
        ],
    )
    _, cohorts = offerings.collect(props, NOW)
    assert [c["start"].date().isoformat() for c in cohorts] == ["2026-10-03", "2026-11-21"]


def test_a_course_with_no_next_live_cohort_still_renders_its_scheduled_one():
    """`next_live_cohort` is None while a cohort runs with no successor booked."""
    props = _props(
        [_course(2, "research-to-production", "R2P", None)],
        [_cohort(2, "2026-12-01T14:00:00Z")],
    )
    _, cohorts = offerings.collect(props, NOW)
    assert [c["slug"] for c in cohorts] == ["research-to-production"]


def test_past_unlisted_and_self_paced_cohorts_are_dropped():
    """The negative control: reading `course_cohorts` must not widen what is advertised."""
    props = _props(
        [_course(3, "workshop", "Workshop", None)],
        [
            _cohort(3, "2026-06-27T14:00:00Z"),  # past
            _cohort(3, "2026-12-01T14:00:00Z", visibility="unlisted"),
            _cohort(3, None, type="self_paced"),
            _cohort(3, "2026-12-08T14:00:00Z"),  # the only one a reader can book
        ],
    )
    _, cohorts = offerings.collect(props, NOW)
    assert [c["start"].date().isoformat() for c in cohorts] == ["2026-12-08"]


def test_a_cohort_whose_course_is_absent_is_dropped():
    """`course_cohorts` covers the whole school; only listed courses have a URL."""
    props = _props([_course(4, "kept", "Kept", None)], [_cohort(99, "2026-12-01T14:00:00Z")])
    _, cohorts = offerings.collect(props, NOW)
    assert cohorts == []
