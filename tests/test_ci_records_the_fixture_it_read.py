"""Every job that checks out the fixture records which commit of it it read.

`ml4t/third-edition-test-data` is checked out at ten sites across three workflows,
none of them with a `ref:`, so every job takes that repository's default branch at the
moment it runs. The fixture moved 26 times in the first thirteen days of September, six
of them on one day, so a job on an unchanged tree can read a different fixture from the
job before it and fail for a reason no commit in this repository explains.

Nothing recorded which one. A red check was then indistinguishable from a red check
caused by a fixture that moved underneath it, and the cost was paid repeatedly and
separately: on 2026-09-04, on 2026-09-11, and again on 2026-09-13, sessions each spent
time establishing that a failure was not theirs (ml4t/agent-workspace#1168).

One line per job answers it, and it is worth asserting rather than trusting, because a
new checkout site added without one puts that job back where all ten were.

The pin that question deferred landed afterwards: every site now takes `ref: ci`, a tag
in the fixture repository advanced deliberately rather than by every push to its default
branch. Recording and pinning answer different halves - the pin decides which fixture a
job reads, the recording says which one it got - so both are asserted here, and a site
that arrives with neither is the state all ten were in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
WORKFLOWS = REPO_ROOT / ".github/workflows"
FIXTURE_REPO = "ml4t/third-edition-test-data"
RECORDING_STEP = "Record the fixture this job read"
FIXTURE_REF = "ci"


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOWS.glob("*.yml"))


def _steps(document: dict) -> list[tuple[str, list[dict]]]:
    return [
        (name, job.get("steps") or [])
        for name, job in (document.get("jobs") or {}).items()
        if isinstance(job, dict)
    ]


def _checks_out_the_fixture(step: dict) -> bool:
    return (step.get("with") or {}).get("repository") == FIXTURE_REPO


def _fixture_checkouts() -> list[tuple[Path, str, int]]:
    """Every (workflow, job, step index) that checks the fixture out."""
    found = []
    for path in _workflow_files():
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, steps in _steps(document):
            found.extend(
                (path, job_name, index)
                for index, step in enumerate(steps)
                if _checks_out_the_fixture(step)
            )
    return found


def test_the_fixture_is_checked_out_somewhere() -> None:
    """The finder is reading the workflows it thinks it is.

    Without this, a rename of the fixture repository or a change to the checkout
    action's shape empties the list and every assertion below passes over nothing.
    """
    assert len(_fixture_checkouts()) >= 10


@pytest.mark.parametrize(
    ("path", "job_name", "index"),
    _fixture_checkouts(),
    ids=lambda value: value.name if isinstance(value, Path) else str(value),
)
def test_a_fixture_checkout_is_followed_by_the_recording_step(
    path: Path, job_name: str, index: int
) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = dict(_steps(document))[job_name]
    following = steps[index + 1] if index + 1 < len(steps) else {}

    assert following.get("name") == RECORDING_STEP, (
        f"{path.name} job {job_name!r}: the step after the {FIXTURE_REPO} checkout is "
        f"{following.get('name')!r}. Without the recording step this job cannot say which "
        "fixture it read, and a failure it did not cause reads as one it did."
    )


@pytest.mark.parametrize(
    ("path", "job_name", "index"),
    _fixture_checkouts(),
    ids=lambda value: value.name if isinstance(value, Path) else str(value),
)
def test_the_recording_step_runs_under_the_same_condition_as_its_checkout(
    path: Path, job_name: str, index: int
) -> None:
    """A recording step that runs when the checkout did not fails on an absent directory.

    Six of the ten checkouts are guarded by `steps.reach.outputs.have_key`, which is
    false on a fork pull request, where no deploy key is reachable and `test-data/` is
    never created.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = dict(_steps(document))[job_name]

    assert steps[index + 1].get("if") == steps[index].get("if")


@pytest.mark.parametrize(
    ("path", "job_name", "index"),
    _fixture_checkouts(),
    ids=lambda value: value.name if isinstance(value, Path) else str(value),
)
def test_a_fixture_checkout_is_pinned_to_the_ci_tag(path: Path, job_name: str, index: int) -> None:
    """No site may take the fixture's default branch.

    A checkout with no `ref:` reads `main` at the moment it runs, so one push to the
    fixture repository changes the inputs of every open pull request at once. The three
    incidents in the module docstring are that, and an unpinned site added later is the
    same defect however many pinned ones surround it.

    The assertion is on the exact tag rather than on `ref:` being present, because a site
    pinned to a different ref reads a different fixture from the other nine and produces
    a disagreement no single job can report.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = dict(_steps(document))[job_name]
    ref = (steps[index].get("with") or {}).get("ref")

    assert ref == FIXTURE_REF, (
        f"{path.name} job {job_name!r}: the {FIXTURE_REPO} checkout takes ref {ref!r}, not "
        f"{FIXTURE_REF!r}. An unpinned site reads whatever the fixture's default branch "
        "holds at that moment, which is what pinning removed."
    )
