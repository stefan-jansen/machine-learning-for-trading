"""Test chapter teaching notebooks via Papermill parameter injection.

Instead of the legacy TEST=1 environment variable (which creates divergent code paths),
this module uses Papermill to inject medium-scale parameter overrides into notebooks.
The same code path always runs; only the scale differs.

When ML4T_OUTPUT_DIR is set and contains pre-generated intermediates (from
generate_intermediates.py), chapter notebooks that depend on case study artifacts
(labels, features, predictions) will find them. This is seeded in CI by copying
intermediates from the test-data repo into ML4T_OUTPUT_DIR before running tests.

Usage:
    # All chapters
    pytest tests/test_chapter_notebooks.py -v

    # Specific chapter
    pytest tests/test_chapter_notebooks.py -v -k "ch05"

    # Specific notebook
    pytest tests/test_chapter_notebooks.py -v -k "tailgan"
"""

from pathlib import Path

import pytest

from tests.pm_helpers import (
    collect_chapter_notebooks,
    current_test_tier,
    get_overrides,
    get_tier,
    run_notebook,
)

REPO_ROOT = Path(__file__).parent.parent

# Collect all chapter teaching notebooks (Ch01-Ch26)
CHAPTER_RANGE = range(1, 27)
CHAPTER_NOTEBOOKS = collect_chapter_notebooks(REPO_ROOT, CHAPTER_RANGE)

# Also collect per-dataset card notebooks (data/*/dataset_card.py, data/*/*/dataset_card.py)
for notebook in sorted(REPO_ROOT.glob("data/**/dataset_card.py")):
    CHAPTER_NOTEBOOKS.append(notebook)

print(f"Found {len(CHAPTER_NOTEBOOKS)} chapter notebooks to test")


@pytest.mark.parametrize(
    "notebook_path",
    CHAPTER_NOTEBOOKS,
    ids=lambda p: p.relative_to(REPO_ROOT).as_posix().replace("/", "::"),
)
def test_chapter_notebook(notebook_path, populated_data_dir, seeded_output_dir):
    """Execute a chapter notebook via Papermill with medium-scale overrides.

    Each notebook runs with:
    - Production defaults (what readers see)
    - Papermill-injected overrides from tests/overrides.yaml (medium scale)
    - ML4T_OUTPUT_DIR set to seeded output dir (has case study configs)
    - MPLBACKEND=Agg, PLOTLY_RENDERER=json (headless rendering)

    Markers (applied at collection time via conftest.py):
    - ``pytest -m gpu`` — run only GPU-requiring notebooks
    - ``pytest -m "not gpu"`` — run only CPU notebooks
    """
    rel_path = notebook_path.relative_to(REPO_ROOT).with_suffix("")
    overrides = get_overrides(str(rel_path))

    # Tier routing: skip when NB tier doesn't match the current run tier.
    # Default tier is per_commit; weekly/on_demand NBs require their dedicated
    # workflow to set ML4T_TEST_TIER explicitly.
    nb_tier = get_tier(overrides)
    run_tier = current_test_tier()
    if nb_tier != run_tier:
        pytest.skip(f"Tier {nb_tier} — current run tier is {run_tier}")

    # Skip if overrides say so (e.g., missing test data)
    if overrides.get("skip"):
        pytest.skip(f"Skipped: {overrides.get('skip_reason', 'marked skip in overrides')}")

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

    # Data layer notebooks expect to run from their own directory (for config.yaml)
    notebook_cwd = notebook_path.parent if "data/" in str(rel_path) else None

    result = run_notebook(
        py_path=notebook_path,
        parameters=parameters,
        timeout=timeout,
        output_dir=seeded_output_dir,
        data_dir=populated_data_dir,
        cwd=notebook_cwd,
    )

    if result["status"] == "error":
        pytest.fail(
            f"\n{'=' * 70}\n"
            f"Notebook failed: {rel_path}\n"
            f"{'=' * 70}\n"
            f"Error: {result['error']}\n"
            f"{'=' * 70}\n"
        )
