"""Every job that checks out the private test data survives a fork pull request.

`secrets.TEST_DATA_DEPLOY_KEY` is not exposed to a `pull_request` from a fork, so
`actions/checkout` falls back to the default token, which cannot see
`ml4t/third-edition-test-data`. The checkout fails and the job fails with it, at a
step that has nothing to do with the contribution. Measured on #595, a two-line
packaging fix from an outside contributor: 26 red checks, none of them theirs.

The fix is to decide once whether the run can reach the data and gate the
data-dependent steps on that. 26 jobs share one workflow file, so what keeps this
from coming back is not the six edits but this test: a data-dependent job added
without the gate fails here.

The gate is a step and not the job's `if:` deliberately. A job-level `if:` that
evaluates false creates no check run at all, and a required context with no check
run blocks the pull request forever - which is what #613 was fixed for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).parent.parent / ".github" / "workflows"
TEST_DATA_REPO = "ml4t/third-edition-test-data"
REACH_OUTPUT = "steps.reach.outputs.have_key"


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def _triggers(workflow: dict) -> dict:
    """The workflow's `on:` block. YAML 1.1 reads a bare ``on`` as the boolean True."""
    triggers = workflow.get("on", workflow.get(True)) or {}
    return triggers if isinstance(triggers, dict) else dict.fromkeys(triggers)


def _jobs_that_check_out_test_data() -> list[tuple[str, str, dict]]:
    """(workflow name, job name, job) for every job checking out the test data.

    Only workflows a pull request can trigger. A workflow_dispatch- or
    schedule-only workflow always runs from this repository, so the deploy key is
    always there and a missing one is a real failure to report rather than a state
    to skip past.
    """
    found = []
    for path in _workflow_files():
        workflow = yaml.safe_load(path.read_text()) or {}
        if "pull_request" not in _triggers(workflow):
            continue
        for job_name, job in (workflow.get("jobs") or {}).items():
            steps = job.get("steps") or []
            if any(
                str((step.get("with") or {}).get("repository", "")) == TEST_DATA_REPO
                for step in steps
            ):
                found.append((path.name, job_name, job))
    return found


def test_at_least_one_job_checks_out_the_test_data() -> None:
    """Guards the guard: a rename that stops matching would empty every test below."""
    assert _jobs_that_check_out_test_data()


@pytest.mark.parametrize(
    ("workflow", "job_name", "job"),
    _jobs_that_check_out_test_data(),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_the_reach_decision_is_the_first_step(workflow: str, job_name: str, job: dict) -> None:
    """It has to precede the container's own setup, not just the checkout."""
    first = (job.get("steps") or [])[0]
    assert first.get("id") == "reach", (
        f"{workflow}::{job_name} checks out {TEST_DATA_REPO} but its first step is "
        f"{first.get('name') or first.get('uses')!r}, not the reach decision"
    )
    assert "TEST_DATA_DEPLOY_KEY" in str(first.get("run", "")), (
        f"{workflow}::{job_name}'s reach step does not read TEST_DATA_DEPLOY_KEY"
    )


@pytest.mark.parametrize(
    ("workflow", "job_name", "job"),
    _jobs_that_check_out_test_data(),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_every_step_after_the_reach_decision_is_gated_on_it(
    workflow: str, job_name: str, job: dict
) -> None:
    ungated = [
        step.get("name") or step.get("uses")
        for step in (job.get("steps") or [])[1:]
        if REACH_OUTPUT not in str(step.get("if", ""))
    ]
    assert not ungated, (
        f"{workflow}::{job_name} runs these steps without the test data being "
        f"reachable: {ungated}. Gate them on {REACH_OUTPUT}."
    )


@pytest.mark.parametrize(
    ("workflow", "job_name", "job"),
    _jobs_that_check_out_test_data(),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_the_job_says_out_loud_that_it_asserted_nothing(
    workflow: str, job_name: str, job: dict
) -> None:
    """A green job that ran nothing has to report that, or the merge is misinformed."""
    reach = (job.get("steps") or [])[0]
    assert "::notice::" in str(reach.get("run", "")), (
        f"{workflow}::{job_name} skips its work on a fork pull request without saying so"
    )


@pytest.mark.parametrize(
    ("workflow", "job_name", "job"),
    _jobs_that_check_out_test_data(),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_the_gate_is_not_the_jobs_own_if(workflow: str, job_name: str, job: dict) -> None:
    assert REACH_OUTPUT not in str(job.get("if", "")), (
        f"{workflow}::{job_name} gates the job rather than its steps; a job-level `if:` "
        "that evaluates false creates no check run, which a required context can never "
        "satisfy (#613)"
    )
