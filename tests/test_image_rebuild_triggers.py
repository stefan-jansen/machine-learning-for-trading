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

So this derives the answer from the Dockerfiles rather than restating the
workflow. Every path a Dockerfile copies out of the build context to decide what
gets installed - a lock or a dependency manifest - must appear in that image's
filter, along with the Dockerfile itself. It fails on the edit that adds the
manifest, not on the release that discovers it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).parent.parent
WORKFLOW = REPO_ROOT / ".github/workflows/docker-publish.yml"

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


def _filters() -> dict[str, list[str]]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
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
