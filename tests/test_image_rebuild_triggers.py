"""A file that decides what an image installs must also rebuild that image.

`test-unit-image` is required and runs the repository's tests inside
`ml4t/ml4t:latest`. Its first step, `.github/scripts/check_env_matches_lock.py`,
refuses to measure anything in an environment that does not match `uv.lock` -
correctly, because a green job in a drifted image is not evidence about the
commit under test.

Until 2026-09-05 nothing rebuilt the published images when the lock moved:
`.github/workflows/docker-publish.yml` ran on a `v3.*` tag or a manual dispatch
and on nothing else. A lock bump therefore staled `ml4t/ml4t:latest`, the guard
refused it, and `main` plus every open pull request went red until someone
dispatched a rebuild by hand. The gap was not in the guard, which did exactly
what it exists to do; it was that a lock bump could not trigger the rebuild that
would have kept the guard satisfiable.

The trigger is now a `changes` job whose per-image path filters decide which
builds run on a push to `main`. That list is the thing that goes stale next: a
Dockerfile that starts reading a new dependency manifest, with nothing added to
the filter, reintroduces the same outage in a form no one is looking for.

Rebuilding after merge is only half of it, and on its own it is the wrong half.
A pull request that changes the lock cannot pass the guard either: the published
image can only be built from a lock that has landed, and the lock cannot land
without the check that needs the rebuild. So `test.yml` builds a candidate image
from the commit under test whenever those same inputs move, and `test-unit-image`
measures in that instead of in `latest`.

Both halves read the same list, and this derives it from the Dockerfiles rather
than restating either workflow. Every path a Dockerfile copies out of the build
context to decide what gets installed - a lock or a dependency manifest - must
appear in that image's filter, along with the Dockerfile itself. It fails on the
edit that adds the manifest, not on the release that discovers it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).parent.parent
WORKFLOW = REPO_ROOT / ".github/workflows/docker-publish.yml"
TEST_WORKFLOW = REPO_ROOT / ".github/workflows/test.yml"

# Filter name in the `changes` job -> the Dockerfile it builds. The names are the
# workflow's own outputs, so a renamed job output fails here rather than silently
# leaving an image unfiltered.
IMAGES = {
    "ml4t": Path("envs/ml4t/Dockerfile"),
    "py312": Path("envs/py312/Dockerfile"),
    "benchmark": Path("envs/benchmark/Dockerfile"),
}

# What "decides what gets installed" means, concretely: a lock, or a manifest the
# resolve reads. A kernel.json or a test script is copied into the image without
# changing a single installed distribution, and rebuilding for one would mean a
# multi-arch build on every unrelated edit.
DEPENDENCY_SUFFIXES = (".lock", ".toml")


def _context_dependency_paths(dockerfile: Path) -> set[str]:
    """Repo-relative lock and manifest paths *dockerfile* copies from the build context.

    `COPY --from=...` pulls from another stage or an external image rather than
    from the repository, so those sources are not repo paths and cannot be
    changed by a commit. Everything else on a `COPY` line except the final
    destination is a context path.
    """
    found: set[str] = set()
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\s*COPY\b", line):
            continue
        parts = line.split()[1:]
        if any(p.startswith("--from=") for p in parts):
            continue
        sources = [p for p in parts if not p.startswith("--")][:-1]
        found.update(s for s in sources if s.endswith(DEPENDENCY_SUFFIXES))
    return found


def _filters(workflow_path: Path = WORKFLOW) -> dict[str, list[str]]:
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    step = next(
        s
        for s in workflow["jobs"]["changes"]["steps"]
        if str(s.get("uses", "")).startswith("dorny/paths-filter")
    )
    return {name: list(paths) for name, paths in yaml.safe_load(step["with"]["filters"]).items()}


@pytest.mark.parametrize("image", sorted(IMAGES))
def test_every_dependency_input_rebuilds_its_image(image: str):
    """The filter covers every lock and manifest the build actually reads."""
    dockerfile = IMAGES[image]
    filters = _filters()

    assert image in filters, f"docker-publish.yml has no `{image}` path filter"
    declared = set(filters[image])
    required = _context_dependency_paths(REPO_ROOT / dockerfile) | {dockerfile.as_posix()}
    missing = sorted(required - declared)

    assert not missing, (
        f"{dockerfile} reads {missing} to decide what it installs, but a change to "
        f"{'them' if len(missing) > 1 else 'it'} does not rebuild `{image}`. Add "
        f"{'them' if len(missing) > 1 else 'it'} to the `{image}` filter in "
        "docker-publish.yml, or the image drifts from the lock and `test-unit-image` "
        "goes red on main with no way to fix it but a manual dispatch."
    )


def test_a_push_to_main_can_trigger_a_build():
    """The trigger the outage was missing.

    Without a branch push in `on`, no lock bump reaches this workflow at all and
    the per-image filters above have nothing to filter.
    """
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML resolves the bare key `on` to the boolean True.
    triggers = workflow.get("on") or workflow[True]

    assert "main" in (triggers["push"].get("branches") or []), (
        "docker-publish.yml does not run on a push to main, so nothing rebuilds the "
        "published images when uv.lock moves"
    )


def test_the_tag_trigger_is_not_path_filtered():
    """A release tag must build regardless of what its commit touched.

    A `paths:` under `push` applies to the tag trigger as well as the branch one,
    so filtering there would make `v3.x` build nothing whenever the tagged commit
    happened not to touch a lock. The filtering belongs in the `changes` job,
    which the build jobs consult only for a branch push.
    """
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow.get("on") or workflow[True]

    assert "paths" not in triggers["push"], (
        "a `paths:` filter under `push` also filters the `v3.*` tag trigger - "
        "put the filtering in the `changes` job instead"
    )

    for job, spec in workflow["jobs"].items():
        if not job.startswith("build-"):
            continue
        assert "github.ref_type == 'tag'" in str(spec.get("if", "")), (
            f"{job} does not exempt a tag push from the `changes` filter, so a "
            "release tag would build nothing"
        )


def test_a_lock_change_measures_test_unit_image_in_a_candidate():
    """The pre-merge half, and the reason a lock bump is mergeable at all.

    `test-unit-image` is required and refuses to measure in an image that does not
    match `uv.lock`. Rebuilding after merge does not help a pull request that
    changes the lock: the published image can only be built from a lock that has
    landed, and the lock cannot land without the check that needs the rebuild. So
    the same inputs that stale `ml4t/ml4t:latest` must also make this workflow
    build a candidate from the commit under test.
    """
    required = _context_dependency_paths(REPO_ROOT / IMAGES["ml4t"]) | {IMAGES["ml4t"].as_posix()}
    declared = set(_filters(TEST_WORKFLOW).get("image-inputs", []))
    missing = sorted(required - declared)

    assert not missing, (
        f"a change to {missing} moves what the image installs, but test.yml would still "
        "measure test-unit-image in the published `latest`. Add "
        f"{'them' if len(missing) > 1 else 'it'} to the `image-inputs` filter, or that "
        "pull request cannot pass a required check and cannot merge."
    )


def test_the_candidate_build_never_publishes():
    """A pull request's image must not be able to move a published tag.

    The candidate exists to be measured, not to ship: anyone can open a pull
    request, and a build step that pushed would let one replace the image every
    other job in the repository trusts.
    """
    workflow = yaml.safe_load(TEST_WORKFLOW.read_text(encoding="utf-8"))
    build = workflow["jobs"]["build-image-candidate"]

    for step in build["steps"]:
        with_ = step.get("with") or {}
        assert not with_.get("push"), f"{step.get('name')} pushes the candidate image"
        assert "push=true" not in str(with_.get("outputs", "")), (
            f"{step.get('name')} pushes the candidate image by digest"
        )
        assert not str(step.get("uses", "")).startswith("docker/login-action"), (
            "the candidate build authenticates to a registry, so it can publish"
        )


def test_a_broken_candidate_fails_the_required_check():
    """The failure mode that would make the whole thing decorative.

    A skipped job counts as a satisfied required check. So excluding a failed
    candidate in the job's own `if` - the obvious way to write "do not fall back
    to `latest`" - would skip `test-unit-image` exactly when the image could not
    be built, and a lock bump whose image is broken would become mergeable with
    the image tests never having run.

    The job therefore runs unconditionally and a step turns a broken candidate
    into a red check. `build-image-candidate` itself is not in the required set,
    so this step is the only thing standing between a failed build and a green
    pull request.
    """
    workflow = yaml.safe_load(TEST_WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["test-unit-image"]

    assert str(job["if"]).strip() == "always()", (
        "test-unit-image must run unconditionally: any other condition can skip it, "
        "and a skipped required check is reported as satisfied"
    )

    guard = next(
        (
            step
            for step in job["steps"]
            if "needs.build-image-candidate.result == 'failure'" in str(step.get("if", ""))
        ),
        None,
    )
    assert guard is not None, (
        "nothing fails test-unit-image when the candidate build fails, so a lock bump "
        "whose image does not build would pass the check that exists to measure it"
    )
    assert "cancelled" in str(guard["if"]), (
        "the guard ignores a cancelled candidate build, which leaves the same hole"
    )
    assert "exit 1" in str(guard.get("run", "")), f"{guard.get('name')} does not fail the job"


# The jobs that execute in the ml4t image and must therefore check it against the
# lock. Named rather than only discovered, because both ways of reaching that image
# are structural: a job-level `container:`, or a step that resolves the image and
# runs `docker run`. A restructuring that moved a job out of both forms would empty
# a purely discovered set and leave the tests below asserting nothing, which is how
# this test previously stopped covering the three notebook jobs.
JOBS_THAT_RUN_IN_THE_ML4T_IMAGE = frozenset(
    {"test-unit-image", "test-chapters", "test-case-studies", "test-neo4j"}
)


def _jobs_in_the_published_ml4t_image() -> dict[str, dict]:
    """Jobs that execute in `ml4t/ml4t`, keyed by job name.

    Two shapes count. A job-level `container:` pins the published image before any
    step runs. A step-level `docker run` against the reference the
    `resolve-ml4t-image` action produces runs either the published image or a
    candidate built from this commit, which is what lets a lock bump be measured in
    the environment it declares.

    Only the ml4t image. `ml4t-py312` and `ml4t-benchmark` install their own
    dependency sets on top of a constraint derived from the root lock, so their
    installed distributions legitimately differ from it and the guard below does not
    apply.
    """
    workflow = yaml.safe_load(TEST_WORKFLOW.read_text(encoding="utf-8"))
    out = {}
    for name, spec in workflow["jobs"].items():
        container = spec.get("container")
        image = container.get("image") if isinstance(container, dict) else container
        steps = spec.get("steps") or []
        # Either shape of step-level resolution: the shared composite action, or the
        # inline resolve test-unit-image carries. Both end in a `docker run` against
        # `steps.image.outputs.ref`, which is what actually names the image.
        resolves = any(
            "resolve-ml4t-image" in str(step.get("uses", ""))
            or "steps.image.outputs.ref" in str(step.get("run", ""))
            for step in steps
        )
        if image == "ml4t/ml4t:latest" or resolves:
            out[name] = spec
    return out


def test_the_guarded_set_is_the_set_that_runs_in_the_image():
    """The two tests below assert over a discovered set, so an empty one proves nothing.

    Both filter that set and assert the filter caught nothing. If the discovery stops
    matching - as it did when these jobs moved from `container:` to `docker run` - the
    set empties, both tests pass, and the coverage they exist for is gone with no
    failure anywhere. Pinning the expected membership is what makes that a red test
    rather than a silent one.
    """
    assert set(_jobs_in_the_published_ml4t_image()) == JOBS_THAT_RUN_IN_THE_ML4T_IMAGE


def test_every_job_in_the_published_image_checks_it_against_the_lock():
    """Every job that runs in the ml4t image checks that image against the lock.

    These jobs choose their image at the step, so a commit that moves the lock is
    measured in an image built from that commit. The guard is what catches the case
    that remains: an image that is not what the lock declares, whether because the
    candidate was not built or because the published image has drifted.

    Without it the drift arrives as arithmetic. On the commit that moved
    ml4t-diagnostic to 0.1.4 it surfaced as thirteen fold-ordering errors inside
    `cs-cme_futures` and a red `ch26`, none of which names the environment - the same
    shape that cost four rounds of investigation aimed at a notebook the last time an
    image drifted off the lock.

    This is the check that was missing when `test-unit-image` was pointed at a
    candidate and these three jobs were not.
    """
    unguarded = sorted(
        name
        for name, spec in _jobs_in_the_published_ml4t_image().items()
        if not any(
            "check_env_matches_lock.py" in str(step.get("run", "")) for step in spec["steps"]
        )
    )

    assert not unguarded, (
        f"{unguarded} execute inside ml4t/ml4t:latest without first checking it against "
        "uv.lock, so a lock bump reaches them as a numerical failure in a notebook "
        "rather than as a statement about the environment"
    )


def test_the_guard_runs_before_anything_it_would_explain():
    """A guard after the work it qualifies explains a failure that already happened."""
    for name, spec in _jobs_in_the_published_ml4t_image().items():
        steps = [str(step.get("run", "")) for step in spec["steps"]]
        guard = next(i for i, s in enumerate(steps) if "check_env_matches_lock.py" in s)
        work = [i for i, step in enumerate(spec["steps"]) if "pytest" in str(step.get("run", ""))]

        assert not work or guard < min(work), (
            f"{name} runs its tests before checking the image against the lock"
        )
