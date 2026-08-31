"""`superseded_members` against real registries, at four shapes a fixture does not reach.

The unit suite in `test_population_supersedes.py` builds its own populations, and five
successive forms of this function passed it while being wrong - three returning an empty
set, one over-retiring, one under-retiring systematically. Every one was caught by a real
registry or by review, none by the unit suite alone. These are the four registries the
fleet measured on 2026-08-25, each chosen for a shape the others do not test:

| case study | retired | what it catches |
|---|---|---|
| `etfs` | 364 | Transitive closure. `etfs-gbm-validation-v1` runs four generations, `linear` and `ipca` three each, so a form that finds the first edge and stops under-reports. Was 324 until 2026-08-28, when `09_dl_lstm` refit and `etfs-lstm-validation-v1` retired its 40 members in one edge (`1c04632dec9c` -> `9d09b33a058c`); the difference is that edge exactly. |
| `us_firm_characteristics` | 135 | Four independent single-hop chains with uneven counts (30/3/30/72), so a form that stops at one chain misses the rest and an over-retiring form cannot coincidentally match. |
| `sp500_equity_option_analytics` | 123 | A doubled catalog, visible directly: 915 prediction rows = 792 live + 123 retired, zero overlap. |
| `crypto_perps_funding` | 400 | The edge outlives the rows. Its refit deleted generation A's `prediction_sets` rows and only the `official_population_members` record survives, so a form reading supersession out of `prediction_sets` returns empty here and correct on the other three. |
| `cme_futures` | 656 | A live population lists retired members. Four dead names left by two renames are in force under their own names and hold 178 identities that `linear` and `gbm` have refit past, so the global reading - retired by someone, listed by nobody in force - un-retires all 178 and the per-name reading does not. |
| `nasdaq100_microstructure` | 150 | A superseded generation that retires nothing. Its `linear-validation-v1` refit grew 16 members to 61 and re-listed all 16, so those 16 are superseded and still published; only `gbm-validation-v1` retires (150 of 150). A form comparing whole generations rather than members returns 166 - isolated to one edge here, where `etfs` shows the same defect only as a total. Added 2026-08-27. |

`crypto_perps_funding` is why this file exists rather than a single spot check: three of the
others would let a wrong implementation look right, and its reading clean is not evidence the
filter is unnecessary there - exposure is decided by whether a refit deleted or layered, never
by the notebook.

`cme_futures` is here for the *other* decision in this function: that retirement is asked per
name. Under the global reading - retired by someone, listed by nobody in force - `etfs`,
`us_firm_characteristics` and `sp500_equity_option_analytics` all still read correctly, so
without a registry that overlaps, half of what this function decides is untested. Its dead
names are not a hypothetical stale snapshot; they are two renames' worth of leftovers, in
force in the only sense the registry has, which is that nothing supersedes them. They list
178 identities `linear` and `gbm` have refit past, and the global reading stops counting
those as retired - so a sweep runs over two generations with nothing about the run looking
wrong. `crypto_perps_funding` happens to overlap too, for an unrelated reason, which is why
the assertion below is on the named cause rather than on a count.

Skipped where the artifacts are not present, which is every CI runner: these read the
canonical production registries under `~/ml4t/artifacts`, not the test fixture.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.research.population import superseded_members, superseded_members_at

ARTIFACTS = Path.home() / "ml4t" / "artifacts" / "case_studies"

# Measured 2026-08-25, each by the pane that owns that case study.
RETIRED_PREDICTIONS = {
    "etfs": 364,
    "us_firm_characteristics": 135,
    "sp500_equity_option_analytics": 123,
    "crypto_perps_funding": 400,
    "cme_futures": 656,
    # Added 2026-08-27.
    "nasdaq100_microstructure": 150,
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

    The disjointness has to be stated against the generations in force, not against the
    catalog. `len(registered - retired) == len(registered) - len(retired)` follows from
    `retired <= registered` by counting alone: it holds for any subset whatsoever and
    cannot fail, so it asserted nothing. What is worth checking is that nothing this name
    still publishes is also something it retired - which `cme_futures` shows is not a
    given, since a population left in force by a rename can list retired identities.
    """
    study = _study("sp500_equity_option_analytics")
    retired = superseded_members(study, member_kind="prediction")
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        registered = {row[0] for row in db.execute("SELECT prediction_hash FROM prediction_sets")}
        rows = db.execute(
            "SELECT population_hash, supersedes_hash FROM official_populations "
            "WHERE member_kind = 'prediction'"
        ).fetchall()
        members: dict[str, set[str]] = {}
        for population_hash, member_hash in db.execute(
            "SELECT population_hash, member_hash FROM official_population_members"
        ):
            members.setdefault(population_hash, set()).add(member_hash)

    assert retired
    assert retired <= registered
    superseded = {row[1] for row in rows if row[1] is not None}
    in_force = {row[0] for row in rows if row[0] not in superseded}
    live = set().union(*(members.get(h, set()) for h in in_force))
    assert live
    assert not (live & retired)


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


def test_a_dead_name_still_in_force_does_not_un_retire_what_it_lists() -> None:
    """`cme_futures` measures the per-name rule against a registry that defeats the global one.

    Its `linear` and `gbm` populations were published three times under two earlier names
    before the current one, and those earlier names were never superseded - a rename leaves
    them in force, listing members their producer has since refit past. So the two readings
    disagree here by construction, and the disagreement is exactly the overlap asserted
    below: identities that are retired under the name that published them and listed by a
    population nothing supersedes.

    The assertion is on the overlap being non-empty, not on its size, because the size is a
    property of how many configurations were fitted and would move under a re-run; that a
    dead name lists retired members at all is the property this registry contributes.
    """
    study = _study("cme_futures")
    retired = superseded_members(study, member_kind="prediction")
    assert retired

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        rows = db.execute(
            "SELECT population_hash, name, supersedes_hash FROM official_populations "
            "WHERE member_kind = 'prediction'"
        ).fetchall()
        members: dict[str, set[str]] = {}
        for population_hash, member_hash in db.execute(
            "SELECT population_hash, member_hash FROM official_population_members"
        ):
            members.setdefault(population_hash, set()).add(member_hash)

    superseded = {row[2] for row in rows if row[2] is not None}
    in_force = {h for h, _, _ in rows if h not in superseded}
    listed_by_something_in_force = set().union(*(members.get(h, set()) for h in in_force))

    # The global reading keeps only what nothing in force still lists. The per-name reading
    # is what `superseded_members` returns, and here the two differ.
    global_reading = retired - listed_by_something_in_force
    assert global_reading < retired, "no live population lists a retired identity"

    un_retired = retired & listed_by_something_in_force
    holders = {name for h, name, _ in rows if h in in_force and members.get(h, set()) & un_retired}
    assert holders, "the overlap has no holder, so this registry no longer tests the rule"
    assert all(
        name not in ("cme_futures-linear-validation-v1", "cme_futures-gbm-validation-v1")
        for name in holders
    ), "a name is listing identities it also retired, which no reading of the lineage allows"


def test_a_superseded_generation_whose_members_are_all_re_listed_retires_none() -> None:
    """The case that separates member-wise comparison from whole-generation comparison.

    `nasdaq100_microstructure` carries two supersedes edges. `gbm-validation-v1` replaced 150
    members with a disjoint 500 and retires all 150. `linear-validation-v1` grew 16 members to
    61 and kept every one of the 16, so its superseded generation retires nothing at all - the
    name moved past that *snapshot*, not past those identities.

    A form that retires a superseded generation wholesale returns 166 here. Measured
    2026-08-27, `etfs` also moves under that form and the other three do not, so this is the
    second of two registries that separate them - and the only one where the difference is a
    single edge with a stated cause, rather than a total across four generations.
    """
    study = _study("nasdaq100_microstructure")
    retired = superseded_members(study, member_kind="prediction")
    assert len(retired) == 150

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        generations = db.execute(
            "SELECT population_hash, name, supersedes_hash FROM official_populations "
            "WHERE member_kind = 'prediction'"
        ).fetchall()
        members = {
            population_hash: {
                row[0]
                for row in db.execute(
                    "SELECT member_hash FROM official_population_members WHERE population_hash = ?",
                    (population_hash,),
                )
            }
            for population_hash, _, _ in generations
        }

    superseded = {row[2] for row in generations if row[2] is not None}
    assert len(superseded) == 2, "both names must have refit for this case to be the one described"
    wholesale = set().union(*(members[population_hash] for population_hash in superseded))
    assert len(wholesale) == 166
    assert retired < wholesale, (
        "retiring superseded generations wholesale would retire 16 identities that "
        "linear-validation-v1 still publishes"
    )


def test_superseded_members_at_answers_for_the_root_it_is_given() -> None:
    """The root-based form a notebook uses when it holds a directory and no study.

    `14_backtest` reads its catalog through `prediction_rows_at(CASE_DIR)` precisely so that
    no `Study.open` runs and no `activate()` re-points the notebook mid-run. Asking the
    lineage question through a study there would answer for whichever registry the activation
    selected, so the retired set and the catalog it filters could describe different
    registries. Same reduction, same answer, one named root.
    """
    study = _study("nasdaq100_microstructure")
    assert superseded_members_at(study.root, member_kind="prediction") == superseded_members(
        study, member_kind="prediction"
    )


def test_superseded_members_at_is_empty_for_a_root_holding_no_registry(tmp_path: Path) -> None:
    """A reader's clean clone has no registry, and that is not an error.

    The distinction that matters is between "no generation was ever written" and "the query
    could not read". The first is answered with nothing; the second must raise, because a
    swallowed failure here is indistinguishable from a clean filter and would silently admit
    every retired row.
    """
    assert superseded_members_at(tmp_path, member_kind="prediction") == frozenset()
