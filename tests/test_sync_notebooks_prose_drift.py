"""`sync_notebooks.py` must see a change confined to prose, not only to code.

The tool compared code cells alone, so editing a markdown cell in the `.py` left the
`.ipynb` holding the old text while the tool printed `Nothing to sync` and the pair was
committed divergent. Measured on `nasdaq100_microstructure`: renumbering `16_costs` and
`17_risk_management` changed prose in three `.py` files and the tool reported two.

The `doc_only` category was written for this case and was unreachable, because `classify`
returned `None` whenever the code matched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "_sync_notebooks_under_test", REPO_ROOT / "scripts" / "sync_notebooks.py"
)
sn = importlib.util.module_from_spec(spec)
# `Drift` is a dataclass, and `dataclasses` resolves its annotations through
# `sys.modules[cls.__module__]`. Registering the module before executing it is what
# makes that lookup succeed for a module loaded by path.
sys.modules[spec.name] = sn
spec.loader.exec_module(sn)

CODE = "x = 1"
PROSE = "The holdout is resolved in 17 and read in 18."


def _pair(tmp_path: Path, py_prose: str, nb_prose: str) -> tuple[Path, Path]:
    py = tmp_path / "12_causal_dml.py"
    py.write_text(
        f"# %% [markdown]\n# {py_prose}\n\n# %%\n{CODE}\n",
        encoding="utf-8",
    )
    ipynb = tmp_path / "12_causal_dml.ipynb"
    ipynb.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": [nb_prose]},
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": [CODE],
                        "outputs": [],
                        "execution_count": 1,
                    },
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    return py, ipynb


def test_a_pair_whose_prose_agrees_is_not_drift(tmp_path):
    py, ipynb = _pair(tmp_path, PROSE, PROSE)
    assert sn.classify(py, ipynb) is None


def test_prose_only_drift_is_reported(tmp_path):
    py, ipynb = _pair(tmp_path, "The holdout is resolved in 19 and read in 20.", PROSE)
    drift = sn.classify(py, ipynb)
    assert drift is not None, "a markdown-only change was invisible to the tool"
    assert drift.code_diff is False
    assert drift.category == "doc_only"


def test_doc_only_drift_is_safe_to_sync_forward(tmp_path):
    """`--safe-only` already listed `doc_only` as a target; nothing could produce one."""
    py, ipynb = _pair(tmp_path, "Renumbered: 19 and 20.", PROSE)
    assert sn.classify(py, ipynb).category in ("doc_only", "code_drift_no_outputs")


@pytest.mark.parametrize("marker", ["# %% [markdown]", '# %% [markdown] tags=["intro"]'])
def test_the_comment_prefix_is_stripped_before_comparing(tmp_path, marker):
    """A `.py` markdown cell is comment-prefixed and the `.ipynb`'s is not. Comparing the
    two without stripping `# ` would report every pair in the repo as drifted."""
    py = tmp_path / "n.py"
    py.write_text(f"{marker}\n# Line one.\n#\n# Line two.\n", encoding="utf-8")
    assert sn.py_markdown_cells(py) == ["Line one.\n\nLine two."]


def test_code_drift_still_wins_its_own_category(tmp_path):
    py, ipynb = _pair(tmp_path, PROSE, PROSE)
    py.write_text(f"# %% [markdown]\n# {PROSE}\n\n# %%\nx = 2\n", encoding="utf-8")
    drift = sn.classify(py, ipynb)
    assert drift.code_diff is True
    assert drift.category == "code_drift_no_outputs"
