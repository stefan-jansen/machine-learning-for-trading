"""Test case study pipeline notebooks via Papermill parameter injection.

Each case study notebook runs independently against pre-generated intermediates
(labels, features, predictions, registries) stored in the test-data repo.
A failure in one notebook does NOT cascade to skip later notebooks.

Stages are auto-discovered: any [0-9][0-9]_*.py file in a case study
directory is treated as a pipeline stage.

Usage:
    # All case studies
    pytest tests/test_case_studies.py -v

    # Specific case study
    pytest tests/test_case_studies.py -v -k "etfs"

    # Specific stage
    pytest tests/test_case_studies.py -v -k "03_features"
"""

from pathlib import Path

import pytest

from tests.awaiting_rebuild import unmet_reason
from tests.pm_helpers import (
    STAGE_RE,
    current_test_tier,
    get_overrides,
    get_tier,
    invocations_for,
    missing_required_env,
    run_notebook,
    stage_sort_key,
)

REPO_ROOT = Path(__file__).parent.parent

# Stages that completed in THIS pytest session, as `(case_study, stage)`. Module level
# because it describes the workspace, which the session shares; an xdist worker has its own
# and will therefore skip what it did not run rather than read another worker's registry.
_STAGES_RUN_HERE: set[tuple[str, str]] = set()


def _required_stages(overrides: dict) -> list[str]:
    """Stages this notebook needs to have run in the same workspace, from `requires_stage`.

    Accepts one name or a list, so a notebook that reads two upstream tiers does not need a
    second key.
    """
    declared = overrides.get("requires_stage")
    if not declared:
        return []
    return [declared] if isinstance(declared, str) else list(declared)


# All case studies
CASE_STUDIES = [
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "us_firm_characteristics",
    "fx_pairs",
    "cme_futures",
    "sp500_options",
    "us_equities_panel",
]


def _collect_case_study_tests():
    """Collect case study pipeline runs as (case_study, stage, path, invocation) tuples.

    Auto-discovers files matching ^\\d{2}[a-z]?_ in each case study directory,
    sorted numerically. Skips helper files (starting with _).

    One notebook is one run unless `tests/overrides.yaml` declares several under
    `invocations`, in which case each becomes its own test - its own timeout, its own
    pass or fail, and its id in the test id, so a failure names the run. Three case
    studies take their label as a parameter and declare more than one, and a chain that
    exercises one label of five asserts nothing about the other four.

    The expansion happens here rather than inside the test because a loop inside one
    test would give the whole notebook one timeout and one verdict, and would report
    five label runs as a single line whichever of them failed.

    Ordering stays stage-major: every invocation of stage 14 runs before any invocation
    of stage 15. That is what a per-label chain needs - `15_portfolio_management`
    selects what `14_backtest` registered for its own label - and the stage-major order
    satisfies it for every label at once rather than per chain.
    """
    tests = []
    for cs in CASE_STUDIES:
        cs_dir = REPO_ROOT / "case_studies" / cs
        if not cs_dir.exists():
            continue

        for notebook in sorted(cs_dir.glob("[0-9][0-9]*.py"), key=stage_sort_key):
            if notebook.name.startswith("_"):
                continue
            if not STAGE_RE.match(notebook.name):
                continue
            stage = notebook.stem  # e.g., "06_linear" or "11a_pca"
            key = str(notebook.relative_to(REPO_ROOT).with_suffix(""))
            for invocation in invocations_for(get_overrides(key), key=key):
                tests.append((cs, stage, notebook, invocation))

    return tests


CASE_STUDY_TESTS = _collect_case_study_tests()

# Spelled out rather than left to pytest, which renders a Path as `notebook_path146` and a
# NamedTuple as `invocation7` - positions in a collection order, which change when a notebook
# is added. `-k <case study>` is how the cs-* jobs select their matrix cell, and it matches
# against this string.
CASE_STUDY_IDS = [
    f"{cs}-{stage}" + (f"-{run.id}" if run.id else "") for cs, stage, _, run in CASE_STUDY_TESTS
]

print(f"Found {len(CASE_STUDY_TESTS)} case study pipeline runs to test")


@pytest.mark.parametrize(
    "case_study,stage,notebook_path,invocation",
    CASE_STUDY_TESTS,
    ids=CASE_STUDY_IDS,
)
def test_case_study_pipeline(
    case_study, stage, notebook_path, invocation, populated_data_dir, seeded_output_dir
):
    """Execute a case study pipeline stage via Papermill.

    Each notebook runs independently — intermediates (labels, features,
    predictions, registries) are pre-generated in the test-data repo.

    With one declared exception. A preview downstream stage selects its inputs by execution
    tier, and the seeded registry holds no preview rows: `tests/conftest.py` records why the
    preview root is deliberately left empty, having measured that seeding it makes
    `14_backtest` refuse to rank anything. So those stages read only what an earlier
    notebook registered in the same workspace, and a notebook that declares
    `requires_stage` is skipped rather than failed when that stage has not run here.
    """
    # Check case-study-level skip (e.g., "case_studies/nasdaq100_microstructure")
    cs_key = f"case_studies/{case_study}"
    cs_overrides = get_overrides(cs_key)
    if cs_overrides.get("skip"):
        pytest.skip(f"Skipped: {cs_overrides.get('skip_reason', 'case study skipped')}")

    rel_path = notebook_path.relative_to(REPO_ROOT).with_suffix("")
    overrides = get_overrides(str(rel_path))

    # Tier routing: skip when NB tier doesn't match the current run tier.
    nb_tier = get_tier(overrides)
    run_tier = current_test_tier()
    if nb_tier != run_tier:
        pytest.skip(f"Tier {nb_tier} — current run tier is {run_tier}")

    # Skip if overrides say so
    if overrides.get("skip"):
        reason = overrides.get("skip_reason", "marked skip in overrides")
        pytest.skip(f"Skipped: {reason}")

    # Inputs the rebuild has not produced yet. Unlike `skip`, this is checked against the
    # registry every run and stops applying the moment the input exists, so it cannot outlive
    # its reason; tests/test_awaiting_rebuild.py fails the build on a declaration that has.
    if declaration := overrides.get("awaiting_rebuild"):
        if reason := unmet_reason(declaration):
            pytest.skip(f"{reason} (#{declaration['issue']})")

    # A stage whose inputs exist only in this workspace. See the docstring: seeding the
    # preview root was measured to do more harm than the gap it fills, so the dependency is
    # declared instead of removed. A full in-order run satisfies it and is unaffected; a
    # focused run or one distributed across workers with separate workspaces skips here,
    # with the prerequisite named, rather than failing several cells in on a refusal that
    # reads like a broken pipeline.
    for prerequisite in _required_stages(overrides):
        if (case_study, prerequisite) not in _STAGES_RUN_HERE:
            pytest.skip(
                f"Requires {case_study}::{prerequisite} in the same workspace, which has "
                f"not run in this session. It registers the preview-tier rows this stage "
                f"selects, and the seeded registry holds none by design."
            )

    # Credentials the notebook cannot run without.
    if absent := missing_required_env(overrides):
        pytest.skip(f"Requires {', '.join(absent)} (unset in this environment)")

    # Check required imports (e.g., gensim, signatory, duckdb)
    requires = overrides.get("requires_import")
    if requires:
        pkg = requires if isinstance(requires, str) else requires[0]
        try:
            __import__(pkg)
        except ImportError:
            pytest.skip(f"Requires {pkg} (not installed in this Docker image)")

    # Check GPU requirement
    if overrides.get("gpu"):
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("GPU required but not available")
        except ImportError:
            pytest.skip("GPU required but torch not installed")

    timeout = overrides.get("timeout", 300)

    result = run_notebook(
        py_path=notebook_path,
        parameters=invocation.parameters,
        timeout=timeout,
        output_dir=seeded_output_dir,
        data_dir=populated_data_dir,
        research_preview=overrides.get("research_preview", True),
    )

    if result["status"] != "error":
        # Keyed by stage rather than by invocation, and `requires_stage` reads it the same
        # way. That is sufficient rather than approximate: collection is stage-major, so
        # every invocation of a producer stage has run before any invocation of a consumer,
        # and `test_a_required_stage_offers_every_invocation_its_dependant_declares` holds
        # the two fan-outs to the same ids. A prerequisite that has to name one particular
        # run would need more than this, and nothing declares one.
        _STAGES_RUN_HERE.add((case_study, stage))

    if result["status"] == "error":
        named = stage if invocation.id is None else f"{stage}[{invocation.id}]"
        pytest.fail(
            f"\n{'=' * 70}\n"
            f"Pipeline failed: {case_study}::{named}\n"
            f"{'=' * 70}\n"
            f"Error: {result['error']}\n"
            f"{'=' * 70}\n"
        )


# Custom test IDs
def pytest_collection_modifyitems(items):
    """Set readable test IDs for case study tests."""
    for item in items:
        if "test_case_study_pipeline" in item.name and hasattr(item, "callspec"):
            cs = item.callspec.params.get("case_study", "")
            stage = item.callspec.params.get("stage", "")
            item._nodeid = f"{item.parent.nodeid}::{cs}::{stage}"
