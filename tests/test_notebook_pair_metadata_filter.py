"""A paired notebook must filter papermill's cell metadata out of its ``.py``.

Papermill stamps ``metadata.papermill`` on every cell it executes. Without
``cell_metadata_filter: tags,-all`` in the jupytext header that stamp round-trips
into the ``.py`` on the next ``jupytext --sync``, so the pair disagrees with what
is committed and JupyterLab's contents manager refuses to open the notebook with a
``File Load Error`` (public #372).

The defect is invisible until something re-executes: a committed pair whose ``.py``
carries the markers inline agrees with its ``.ipynb`` and opens fine. Five stage
01-05 notebooks sat in that state, and a sweep that re-executes all 44 would have
turned every one of them into a tree the ``notebook-sync`` pre-commit gate refuses.

Nothing checked the header, which is why it could be dropped by editing a notebook
by hand or by adding one from a template that predates the convention.

The guard covers every paired case-study notebook. The stage 06+ repair found 113
notebooks without ``cell_metadata_filter`` and five sources with inline markers.
Leaving the test scoped to stages 01-05 would let the same failure recur on the
downstream notebooks before their production execution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILTER = "tags,-all"


def _paired_notebooks() -> list[Path]:
    """Every ``.ipynb`` under ``case_studies`` with a paired ``.py`` source."""
    return sorted(
        nb
        for nb in (REPO_ROOT / "case_studies").glob("*/*.ipynb")
        if nb.with_suffix(".py").exists()
    )


@pytest.mark.parametrize("notebook", _paired_notebooks(), ids=lambda p: p.stem)
def test_the_jupytext_header_filters_cell_metadata(notebook: Path) -> None:
    metadata = json.loads(notebook.read_text())["metadata"]
    jupytext = metadata.get("jupytext", {})
    assert jupytext.get("cell_metadata_filter") == REQUIRED_FILTER, (
        f"{notebook.relative_to(REPO_ROOT)} has no cell_metadata_filter, so the next "
        f"production run writes papermill's cell metadata back into its .py and the "
        f"pair stops opening in JupyterLab"
    )


@pytest.mark.parametrize("notebook", _paired_notebooks(), ids=lambda p: p.stem)
def test_the_paired_py_carries_no_papermill_marker(notebook: Path) -> None:
    """The state the filter prevents, read from the other side of the pair.

    A ``.py`` carrying ``papermill={...}`` on its cell markers is a pair that already
    round-tripped once. It opens today and desyncs on the next run, which is the
    trap: the header check above would pass on a file someone has since re-added the
    header to without stripping what the earlier runs left behind.
    """
    source = notebook.with_suffix(".py").read_text()
    assert "papermill={" not in source, (
        f"{notebook.with_suffix('.py').relative_to(REPO_ROOT)} carries inline papermill "
        f"cell metadata from an earlier run; strip it and re-execute"
    )
