"""The publish gate is the documented reader check, and it sees every ml4t subpackage.

`docker-publish.yml`'s `smoke-ml4t` job is the last thing that runs before an image
reaches Docker Hub. Until 2026-09-08 it ran only `envs/test_all_imports.py --scan`,
which cannot see a broken subpackage: `envs/scan_imports.py` reduces every import to
its top-level name, so `from ml4t.live.brokers.ib import IBBroker` arrives as `ml4t`,
and importing `ml4t` never executes `ml4t/live/__init__.py`.

That is not hypothetical. The image published on 2026-09-04 carried an
`import ml4t.live` that raised `RuntimeError: There is no current event loop in
thread 'MainThread'`; every Chapter 25 notebook failed before its first cell, and
`docker compose run --rm ml4t python scripts/verify_installation.py` - the one
PASS/FAIL command `docs/installation.md` names - exited 1 on a correct install.
`smoke-ml4t` was green on both arches on every publish. Measured by reinstalling the
defect into the published image: the scan step exits 0, `verify_installation.py`
exits 1 and names `ml4t-live`.

The fix was to run the documented reader check as the publish gate. Two things have
to stay true for that to keep meaning anything, and neither is self-enforcing:

1. The gate and the documented check are the same command. Left as a convention it
   drifts, and the drift is invisible - both halves keep working, they just stop
   being about each other.
2. The gate names every `ml4t.<sub>` the repository imports. `ml4t.models` was
   already missing when this was written, so the gate was blind to one of the six
   before it ever ran in the publish job.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_DOC = REPO_ROOT / "docs" / "installation.md"
PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-publish.yml"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_installation.py"

_SKIP_DIRS = {".venv", ".git", "node_modules", "__pycache__", ".ipynb_checkpoints", "build"}


def _imported_ml4t_subpackages() -> set[str]:
    """Every ``ml4t.<sub>`` the repository's Python actually imports, by AST.

    A grep would also collect the ones that appear in prose - `24_autonomous_agents`
    names a future `ml4t.agent` in a comment - and the gate would then be asked to
    check a package nothing imports.
    """
    found: set[str] = set()
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                parts = name.split(".")
                if parts[0] == "ml4t" and len(parts) >= 2:
                    found.add(f"ml4t.{parts[1]}")
    return found


def _gate_checked_subpackages() -> set[str]:
    """What ``check_ml4t_libraries()`` asks for, recorded rather than imported.

    Driving the real function is what keeps this from going stale against a
    restatement of its contents; importing the packages for real would make a unit
    test depend on the whole ml4t stack being installed.
    """
    spec = importlib.util.spec_from_file_location("_verify_installation", VERIFY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    asked: list[str] = []
    module.check_import = lambda category, import_name, label=None: asked.append(import_name)
    module.check_from_import = lambda *a, **k: None
    module.check_ml4t_libraries()

    return {name for name in asked if name.startswith("ml4t.")}


def _documented_docker_gate() -> str:
    """The script `docs/installation.md` tells a Docker reader to run to verify."""
    section = re.search(
        r"^## Verify Your Installation\s*$(.*?)^## ", INSTALL_DOC.read_text(), re.M | re.S
    )
    assert section, "docs/installation.md no longer has a 'Verify Your Installation' section"

    commands = re.findall(
        r"docker compose run [^\n]*?\bml4t\b[^\n]*?\bpython\s+(\S+\.py)", section.group(1)
    )
    assert commands, (
        f"that section names no `docker compose run ... python <script>`:\n{section.group(1)}"
    )
    return commands[0]


def _jobs_publication_waits_for(workflow: dict, publishing_job: str = "merge-ml4t") -> set[str]:
    """Every job that has to succeed before ``publishing_job`` runs, transitively.

    `merge-ml4t` is where `:latest` and the version tag start pointing at the new
    build. A check that is not upstream of it cannot stop a bad image reaching
    readers, however loudly it fails.
    """
    jobs = workflow["jobs"]
    upstream: set[str] = set()
    frontier = [publishing_job]
    while frontier:
        name = frontier.pop()
        needs = jobs.get(name, {}).get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        for dependency in needs:
            if dependency not in upstream:
                upstream.add(dependency)
                frontier.append(dependency)
    return upstream


def _scripts_run_by(workflow: dict, job_names: set[str]) -> set[str]:
    """Every ``python <script>.py`` those jobs run inside a container."""
    scripts: set[str] = set()
    for name in job_names:
        for step in workflow["jobs"].get(name, {}).get("steps", []):
            scripts.update(re.findall(r"\bpython\s+(/app/\S+\.py)", step.get("run", "")))
    return scripts


def test_the_documented_reader_check_gates_publication():
    """The gate has to run before the tags move, not after.

    Verifying a published image is a detector: it goes red once readers can
    already pull the break. `smoke-ml4t` needs `merge-ml4t`, so the check has to
    live somewhere `merge-ml4t` waits for instead.
    """
    workflow = yaml.safe_load(PUBLISH_WORKFLOW.read_text())
    documented = _documented_docker_gate()

    # The doc names the script relative to the repo root; the workflow mounts that
    # root at /app.
    expected = f"/app/{documented.lstrip('./')}"

    before_publish = _jobs_publication_waits_for(workflow)
    gating = _scripts_run_by(workflow, before_publish)

    assert expected in gating, (
        f"docs/installation.md tells a Docker reader to run {documented!r} to verify the "
        f"install, but no job merge-ml4t waits for runs it - only {sorted(gating)}. Running "
        "it after merge-ml4t publishes `:latest` reports a broken image to a reader who can "
        "already pull it."
    )


def test_gate_checks_every_ml4t_subpackage_the_repo_imports():
    imported = _imported_ml4t_subpackages()
    checked = _gate_checked_subpackages()

    assert imported, "the AST scan found no ml4t.* imports at all, which cannot be right"

    missing = sorted(imported - checked)
    assert not missing, (
        f"{VERIFY_SCRIPT.name} does not check {missing}, which the repository imports. "
        "Nothing else looks at this granularity - the publish smoke test scans top-level "
        "names - so a break in one of these reaches Docker Hub green."
    )


def test_gate_does_not_ask_for_subpackages_nothing_imports():
    """The other direction, so the list stays a description rather than a wish."""
    imported = _imported_ml4t_subpackages()
    checked = _gate_checked_subpackages()

    stale = sorted(checked - imported)
    assert not stale, (
        f"{VERIFY_SCRIPT.name} checks {stale}, which nothing in the repository imports. "
        "Either an import was removed and the gate was not, or the name is wrong."
    )
