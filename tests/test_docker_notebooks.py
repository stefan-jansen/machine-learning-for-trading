"""Test notebooks that require Docker environments (py312, neo4j, benchmark).

Same as test_chapter_notebooks.py but IGNORES skip flags from overrides.yaml.
These notebooks are skipped in uv-native runs (missing modules like signatory,
gensim, esig, tfcausalimpact, or Neo4j) but CAN run inside their respective
Docker images.

The skip flag stays in overrides.yaml so the uv-native runner still skips them.
This file runs them in Docker where the dependencies are available.

Usage:
    # Py312 notebooks (signatory, gensim, esig, pfhedge, tfcausalimpact, torch CUDA bug)
    python -m pytest tests/test_docker_notebooks.py -v -k "03_sigcwgan or ..."

    # Neo4j notebooks
    python -m pytest tests/test_docker_notebooks.py -v -k "08_8k_event_extraction or ..."

    # Benchmark notebooks
    python -m pytest tests/test_docker_notebooks.py -v -k "18_storage_benchmark_database or ..."
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

# Collect all chapter teaching notebooks (Ch01-Ch26) + data layer
CHAPTER_RANGE = range(1, 27)
CHAPTER_NOTEBOOKS = collect_chapter_notebooks(REPO_ROOT, CHAPTER_RANGE)
for notebook in sorted(REPO_ROOT.glob("data/**/dataset_card.py")):
    CHAPTER_NOTEBOOKS.append(notebook)


@pytest.mark.parametrize(
    "notebook_path",
    CHAPTER_NOTEBOOKS,
    ids=lambda p: p.relative_to(REPO_ROOT).as_posix().replace("/", "::"),
)
def test_docker_notebook(notebook_path, populated_data_dir, seeded_output_dir):
    """Execute a notebook via Papermill, ignoring skip flags.

    Identical to test_chapter_notebook() but does NOT honor the 'skip' key
    in overrides.yaml. This allows Docker-based CI jobs to run notebooks
    that are skipped in the uv-native environment due to missing modules.

    GPU skips are still honored (Docker CI runners have no GPU).
    """
    rel_path = notebook_path.relative_to(REPO_ROOT).with_suffix("")
    overrides = get_overrides(str(rel_path))

    # Tier routing still applies — Docker tests are normally per_commit, but
    # this honors the same env-driven gating used by the uv-native runners.
    nb_tier = get_tier(overrides)
    run_tier = current_test_tier()
    if nb_tier != run_tier:
        pytest.skip(f"Tier {nb_tier} — current run tier is {run_tier}")

    # NOTE: We intentionally do NOT check overrides.get("skip") here.
    # That's the whole point of this file — Docker provides the missing deps.

    # GPU requirement still applies (CI runners have no GPU)
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
            f"Notebook failed: {rel_path}\n"
            f"{'=' * 70}\n"
            f"Error: {result['error']}\n"
            f"{'=' * 70}\n"
        )
