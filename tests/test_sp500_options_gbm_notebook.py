"""Notebook-level checks on the S&P 500 options GBM notebook.

The sibling `tests/test_sp500_options_linear_fold_alignment.py` explains why the notebook's own
behaviours are tested against the shared research implementation rather than against private
copies parsed out of the source. This file is the `07_gbm` half of what has no shared home: the
notebook must stay inside the cell-length limit clause T6 of the notebook standard sets, and its
prose must not assert a result that re-executing it can falsify.

The stale claims listed below are not hypothetical. Every one of them was published in the
notebook's conclusion and contradicted by the figure directly above it the first time the
notebook was executed against the rebuilt registry.
"""

from __future__ import annotations

from pathlib import Path

NOTEBOOK = Path("case_studies/sp500_options/07_gbm.py")

# T6: "Target 15-30 lines, hard max 40." A cell past that is split at its natural seam.
MAX_CELL_LINES = 40

# Each of these asserted an outcome the executed figures contradict. Huber does not separate from
# absolute error at all - the two interleave, and two Huber configurations end below zero - and
# ten configurations peak inside the first fifth of training, not twelve.
STALE_CLAIMS = (
    "in the predicted order, and almost without overlap",
    "Every Huber configuration\n# ends on the right side of it",
    "Absolute error sits between them",
    "Twelve of the fifteen",
    "bar stands above the line at the left of the ranking",
    "the amber absolute-error lines cross back and forth near it",
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
