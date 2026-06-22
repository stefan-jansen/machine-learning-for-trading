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

import re
from pathlib import Path

import pytest

from tests.pm_helpers import current_test_tier, get_overrides, get_tier, run_notebook

REPO_ROOT = Path(__file__).parent.parent

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

# Pattern for numbered pipeline stages — allows optional single-letter suffix
# (e.g., 10a_pca, 11b_ipca) for per-estimator notebook splits.
_STAGE_RE = re.compile(r"^\d{2}[a-z]?_")


def _collect_case_study_tests():
    """Collect all case study pipeline notebooks as (case_study, stage, path) tuples.

    Auto-discovers files matching ^\\d{2}[a-z]?_ in each case study directory,
    sorted numerically. Skips helper files (starting with _).
    """
    tests = []
    for cs in CASE_STUDIES:
        cs_dir = REPO_ROOT / "case_studies" / cs
        if not cs_dir.exists():
            continue

        for notebook in sorted(cs_dir.glob("[0-9][0-9]*.py")):
            if notebook.name.startswith("_"):
                continue
            if not _STAGE_RE.match(notebook.name):
                continue
            stage = notebook.stem  # e.g., "06_linear" or "11a_pca"
            tests.append((cs, stage, notebook))

    return tests


CASE_STUDY_TESTS = _collect_case_study_tests()

print(f"Found {len(CASE_STUDY_TESTS)} case study pipeline notebooks to test")


@pytest.mark.parametrize(
    "case_study,stage,notebook_path",
    CASE_STUDY_TESTS,
    ids=lambda *args: None,  # Custom IDs below
)
def test_case_study_pipeline(
    case_study, stage, notebook_path, populated_data_dir, seeded_output_dir
):
    """Execute a case study pipeline stage via Papermill.

    Each notebook runs independently — intermediates (labels, features,
    predictions, registries) are pre-generated in the test-data repo.
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
    parameters = overrides.get("parameters", {})

    result = run_notebook(
        py_path=notebook_path,
        parameters=parameters,
        timeout=timeout,
        output_dir=seeded_output_dir,
        data_dir=populated_data_dir,
    )

    if result["status"] == "error":
        pytest.fail(
            f"\n{'=' * 70}\n"
            f"Pipeline failed: {case_study}::{stage}\n"
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
