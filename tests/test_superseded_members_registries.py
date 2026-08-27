"""`superseded_members` against real registries, at four shapes a fixture does not reach.

The unit suite in `test_population_supersedes.py` builds its own populations, and five
successive forms of this function passed it while being wrong - three returning an empty
set, one over-retiring, one under-retiring systematically. Every one was caught by a real
registry or by review, none by the unit suite alone. These are the four registries the
fleet measured on 2026-08-25, each chosen for a shape the others do not test:

| case study | retired | what it catches |
|---|---|---|
| `etfs` | 324 | Transitive closure. `etfs-gbm-validation-v1` runs four generations, `linear` and `ipca` three each, so a form that finds the first edge and stops under-reports. |
| `us_firm_characteristics` | 135 | Four independent single-hop chains with uneven counts (30/3/30/72), so a form that stops at one chain misses the rest and an over-retiring form cannot coincidentally match. |
| `sp500_equity_option_analytics` | 123 | A doubled catalog, visible directly: 915 prediction rows = 792 live + 123 retired, zero overlap. |
| `crypto_perps_funding` | 400 | The edge outlives the rows. Its refit deleted generation A's `prediction_sets` rows and only the `official_population_members` record survives, so a form reading supersession out of `prediction_sets` returns empty here and correct on the other three. |

The last one is why this file exists rather than a single spot check. Three of the four
registries would let a wrong implementation look right, and `crypto_perps_funding` reading
clean is not evidence the filter is unnecessary there - exposure is decided by whether a
refit deleted or layered, never by the notebook.

Skipped where the artifacts are not present, which is every CI runner: these read the
canonical production registries under `~/ml4t/artifacts`, not the test fixture.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.research.population import superseded_members

ARTIFACTS = Path.home() / "ml4t" / "artifacts" / "case_studies"

# Measured 2026-08-25, each by the pane that owns that case study.
RETIRED_PREDICTIONS = {
    "etfs": 324,
    "us_firm_characteristics": 135,
    "sp500_equity_option_analytics": 123,
    "crypto_perps_funding": 400,
}


class _SingleRootStudy:
    """The shape `superseded_members` needs, pointed at one registry root.

    A production case study has no workspace: `study.root` and `study.release_case_root`
    are the same directory, which is the configuration these registries are in.
    """

    read_only = False

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def release_case_root(self) -> Path:
        return self.root


def _study(case_study: str) -> _SingleRootStudy:
    root = ARTIFACTS / case_study
    if not (root / "run_log" / "registry.db").exists():
        pytest.skip(f"no production registry for {case_study} on this machine")
    return _SingleRootStudy(root)


@pytest.mark.parametrize(("case_study", "expected"), sorted(RETIRED_PREDICTIONS.items()))
def test_retired_prediction_count(case_study: str, expected: int) -> None:
    assert len(superseded_members(_study(case_study), member_kind="prediction")) == expected


def test_a_retired_generation_and_the_live_catalog_do_not_overlap() -> None:
    """What the doubling looks like from the catalog's side.

    `sp500_equity_option_analytics` holds both generations as complete, current rows, so
    the count a backtest would sweep is the sum. Asserting the disjointness rather than
    the totals is what says the retired set is a clean second generation and not an
    arbitrary subset.
    """
    study = _study("sp500_equity_option_analytics")
    retired = superseded_members(study, member_kind="prediction")
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        registered = {row[0] for row in db.execute("SELECT prediction_hash FROM prediction_sets")}
    assert retired <= registered
    assert len(registered - retired) == len(registered) - len(retired)


def test_an_edge_outlives_the_rows_it_retired() -> None:
    """`crypto_perps_funding` deleted the rows and kept the membership record.

    Nothing of generation A is left in `prediction_sets`; only
    `official_population_members` still lists it, which is what keeps the edge readable as
    history. A form that derives supersession from the prediction rows returns empty here
    and correct everywhere else, which is the failure this case exists to catch.
    """
    study = _study("crypto_perps_funding")
    retired = superseded_members(study, member_kind="prediction")
    assert retired
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        registered = {row[0] for row in db.execute("SELECT prediction_hash FROM prediction_sets")}
    assert not (retired & registered)
