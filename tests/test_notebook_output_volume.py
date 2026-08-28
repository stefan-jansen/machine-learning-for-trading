"""A cell that emits tens of thousands of outputs has a logger writing into the artifact.

`case_studies/etfs/10_dl_tsmixer.ipynb` is 11.9 MB and 441,860 lines - 4x the next largest
notebook in the repository by line count, and 118x its own sibling `09_dl_lstm`. None of it is
result: Darts constructs a fresh Lightning trainer for every five-epoch checkpoint increment, and
each construction emits "GPU available", "TPU available", "LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES",
a litlogger advertisement and a "`Trainer.fit` stopped" line. Eight folds times a hundred epochs
in five-epoch steps is 160 trainers, and one cell carried 62,901 output entries.

The consequence is reader-facing: a notebook that size does not render on GitHub and does not open
in JupyterLab, so the chapter it belongs to has no readable artifact at all.

Outputs per cell rather than bytes or lines is what separates this from legitimate size. The two
largest notebooks in the repository are larger in bytes and are image-heavy, which is content; a
plot is one output entry however many bytes it holds. Measured across every notebook over 200 KB
on 2026-08-27:

    62,901  case_studies/etfs/10_dl_tsmixer.ipynb     <- the defect
     5,629  05_synthetic_data/01_timegan.ipynb
     2,068  case_studies/cme_futures/09_dl_lstm.ipynb
       828  case_studies/etfs/09_dl_lstm.ipynb

The cap sits above the whole legitimate distribution with room to spare, so it fails on a logger
loose in a cell and not on a notebook that prints a lot.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# An order of magnitude above the largest legitimate producer (timegan, 5,629) and an order of
# magnitude below the defect (62,901). Raising this to admit a notebook is the wrong fix: silence
# the logger, as `darts_forecasting._trainer_kwargs` already does for the progress bar.
MAX_OUTPUTS_PER_CELL = 8_000


def _tracked_notebooks() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [REPO / name for name in listed]


def _worst_cell(path: Path) -> tuple[int, int]:
    """(index, output count) of the cell emitting the most outputs."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    counts = [(len(cell.get("outputs", [])), index) for index, cell in enumerate(notebook["cells"])]
    if not counts:
        return (-1, 0)
    count, index = max(counts)
    return (index, count)


@pytest.mark.parametrize("path", _tracked_notebooks(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_no_cell_buries_its_results_in_log_output(path: Path) -> None:
    if not path.exists():
        pytest.skip("tracked but not present in this worktree")
    index, count = _worst_cell(path)
    name = path.relative_to(REPO) if path.is_relative_to(REPO) else path.name
    assert count <= MAX_OUTPUTS_PER_CELL, (
        f"{name} cell {index} carries {count:,} output entries, over the "
        f"{MAX_OUTPUTS_PER_CELL:,} cap. That is a logger writing into the artifact rather than a "
        "result: silence it at the source and re-run, rather than raising the cap."
    )


def test_the_cap_catches_a_logger_loose_in_a_cell(tmp_path: Path) -> None:
    """The guard fails on the shape rather than only on today's tree.

    The notebook it was written for is re-run as part of the same change, so by the time this
    file lands the tracked artifact is clean and the parameterized case above passes on every
    notebook. Without this, that green would be indistinguishable from a guard that never fires.
    """
    path = tmp_path / "noisy.ipynb"
    noisy = {
        "cells": [
            {
                "cell_type": "code",
                "source": [],
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": ["GPU available\n"]}
                ]
                * (MAX_OUTPUTS_PER_CELL + 1),
            }
        ]
    }
    path.write_text(json.dumps(noisy), encoding="utf-8")
    index, count = _worst_cell(path)
    assert (index, count) == (0, MAX_OUTPUTS_PER_CELL + 1)
    with pytest.raises(AssertionError, match="logger writing into the artifact"):
        test_no_cell_buries_its_results_in_log_output(path)
