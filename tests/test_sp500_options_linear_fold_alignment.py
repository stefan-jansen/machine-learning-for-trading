"""Notebook-level checks on the S&P 500 options linear notebook.

This file used to load seventeen private helper functions out of the notebook by parsing its
source, and exercise each one: fold alignment, holdout sealing, prediction-key contracts, IC
uncertainty, cache validation and workspace isolation. Those helpers were copies living inside
one notebook. `06_linear` now goes through the shared research API, the copies are gone, and the
behaviours are tested against the implementation that actually runs:

    workspace isolation, symlink and regeneration rules  tests/test_research_workspace.py
    fold geometry, ordering, embargo, holdout sealing    tests/test_cv_splits.py
    temporal fold validation and warmup allowance        tests/test_cv_splits.py
    fold preparation and streaming                       tests/test_folds.py
    cached results and checkpoint assembly               tests/test_cv_cached_results.py
    training and prediction identity                     tests/test_fold_resolution.py
    IC, AUC and their uncertainty                        tests/test_registry_metrics.py

Testing a notebook's private copy of a rule cannot fail when the shared implementation breaks,
which is the reason those tests are not reproduced here.

What remains is the part that is about this notebook as an artifact and has no shared home: it
must stay inside the cell-length limit clause T6 of the notebook standard sets, and it must not
carry result claims in its prose that a re-run can falsify.
"""

from __future__ import annotations

from pathlib import Path

NOTEBOOK = Path("case_studies/sp500_options/06_linear.py")

# T6: "Target 15-30 lines, hard max 40." A cell past that is split at its natural seam.
MAX_CELL_LINES = 40

# Prose that asserted an outcome rather than a method. Every number in this case study moves on a
# re-run, so a sentence naming one is wrong the moment it is re-executed.
STALE_CLAIMS = (
    "linear signal here is faint",
    "best config barely clears zero",
    "declines monotonically",
    "aggressive L1 selection is worst",
)


def _code_cells() -> list[tuple[int, int]]:
    """(1-based start line, body line count) for every code cell in the paired script."""
    lines = NOTEBOOK.read_text().splitlines()
    markers = [index for index, line in enumerate(lines) if line.startswith("# %%")]
    markers.append(len(lines))
    return [
        (start + 1, end - start - 1)
        for start, end in zip(markers, markers[1:], strict=False)
        if "[markdown]" not in lines[start]
    ]


def test_no_code_cell_exceeds_the_standards_hard_maximum() -> None:
    oversized = [(line, count) for line, count in _code_cells() if count > MAX_CELL_LINES]
    assert oversized == [], (
        f"cells over {MAX_CELL_LINES} lines at (line, length): {oversized}. "
        "Split at a load/transform/compute/visualize seam with a markdown transition."
    )


def test_prose_states_method_rather_than_a_result_a_rerun_would_falsify() -> None:
    source = NOTEBOOK.read_text()
    present = [claim for claim in STALE_CLAIMS if claim in source]
    assert present == [], f"prose asserts an outcome that a re-run can falsify: {present}"
