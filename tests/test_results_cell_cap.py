"""N7: results discussion belongs to two notebooks per case study, and nowhere else.

`rules/notebook-standards.md` N1 and N7 say where a number may appear in a notebook:
in a figure, or in a markdown cell tagged `results`. Every case study has one
model-analysis notebook and one strategy-analysis notebook that may carry as many
tagged cells as the argument needs; every other notebook is capped at three and
usually wants one.

Nothing checked this, so it drifted. Sixteen notebooks were over the cap when the
check was written on 2026-08-25, and each is recorded below with its count. The list
is a ratchet, not an exemption: a notebook that is not on it may not exceed the cap,
and a notebook on it may not get worse. Removing an entry is how the debt is paid.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# A cell whose `# %%` line carries the `results` tag. jupytext writes the tag list on
# the cell marker in the .py, which is why the .py is what is read here - the tag lives
# in .ipynb cell metadata, where it is invisible to a reader and awkward to grep.
RESULTS_CELL = re.compile(r'^# %%.*tags=\[[^\]]*"results"', re.MULTILINE)

CAP = 3

# Where a result is interpreted, compared and argued about. Exempt by N7.
EXEMPT_SUFFIXES = ("model_analysis", "strategy_analysis")

# Over the cap on 2026-08-25, with the count on that day. Each entry is a notebook whose
# results discussion needs consolidating into at most three tagged cells. Fixing one costs
# a re-run, because any change to the .py makes the executed .ipynb stale, so these are
# paid off as each notebook is next re-run for another reason rather than in one sweep.
KNOWN_OVER_CAP = {
    "case_studies/cme_futures/06_linear.py": 4,
    "case_studies/cme_futures/07_gbm.py": 6,
    "case_studies/crypto_perps_funding/07_gbm.py": 6,
    # Written 2026-08-23 and over the cap on the day this list was measured, but on an
    # unmerged branch, so the sweep could not see it. Same debt as the entries above
    # rather than new drift. The four cells are the prediction catalog, the decision
    # clock, the coverage check and the sweep result; the first three fold into one
    # cell about what is being backtested. Paid off on the notebook's next canonical
    # run, which waits on #636 - crypto_perps_funding cannot re-execute 13 until the
    # 196 moved deep-learning prediction identities are migrated.
    "case_studies/crypto_perps_funding/13_backtest.py": 4,
    "case_studies/etfs/06_linear.py": 4,
    "case_studies/etfs/07_gbm.py": 8,
    "case_studies/fx_pairs/06_linear.py": 4,
    "case_studies/fx_pairs/07_gbm.py": 6,
    # Arrived with #612, after this list was first measured. nasdaq100_microstructure owns it.
    "case_studies/nasdaq100_microstructure/07_gbm.py": 4,
    "case_studies/us_equities_panel/16_backtest.py": 5,
    "case_studies/us_equities_panel/17_portfolio_management.py": 5,
    "case_studies/us_equities_panel/18_costs.py": 5,
    "case_studies/us_equities_panel/19_risk_management.py": 4,
    "case_studies/us_firm_characteristics/06_gbm.py": 4,
}


def _capped_notebooks() -> list[Path]:
    return sorted(
        p for p in REPO.glob("case_studies/*/[0-9]*.py") if not p.stem.endswith(EXEMPT_SUFFIXES)
    )


def _count(path: Path) -> int:
    return len(RESULTS_CELL.findall(path.read_text(encoding="utf-8", errors="replace")))


@pytest.mark.parametrize("path", _capped_notebooks(), ids=lambda p: p.name)
def test_results_cells_within_cap(path: Path) -> None:
    rel = path.relative_to(REPO).as_posix()
    count = _count(path)
    allowed = KNOWN_OVER_CAP.get(rel, CAP)
    assert count <= allowed, (
        f"{rel} has {count} cells tagged `results`; N7 caps a notebook that is not the "
        f"case study's model-analysis or strategy-analysis at {CAP}"
        + (
            f" (recorded at {allowed} on 2026-08-25 and may not grow)"
            if rel in KNOWN_OVER_CAP
            else ""
        )
        + ". Consolidate the discussion into the model-analysis notebook, or fold the "
        "cells together. Raising the recorded count is not a fix."
    )


def test_known_over_cap_list_has_no_stale_entries() -> None:
    """A notebook that has been brought back under the cap comes off the list.

    Without this the list only ever grows stale, and a later reader cannot tell which
    entries are real debt and which are notebooks somebody already fixed.
    """
    fixed = {
        rel: _count(REPO / rel)
        for rel, recorded in KNOWN_OVER_CAP.items()
        if (REPO / rel).exists() and _count(REPO / rel) <= CAP
    }
    assert not fixed, (
        "these are back within the cap and their entries in KNOWN_OVER_CAP should be "
        f"deleted: {fixed}"
    )


def test_known_over_cap_list_names_real_files() -> None:
    missing = [rel for rel in KNOWN_OVER_CAP if not (REPO / rel).exists()]
    assert not missing, f"KNOWN_OVER_CAP names files that do not exist: {missing}"
