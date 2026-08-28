"""The family table has to answer over the same population the selection ran on.

`best` and `compare_allocators` take `prediction_hashes`; `compare_families` did not, so a
notebook that scoped its sweep and its shortlist to the live population still reported family
medians computed over every registered row, retired generations included. The two tables then
disagree by construction: the leader comes from one population and the family it belongs to is
summarised over another.

The coverage bar is the part that makes this more than a row count. `full_coverage_prediction_sql`
keeps only the rows whose `ic_n_days` equals the maximum for their `(split, family, label)`, and
that maximum has to be taken within the population being asked about - otherwise a retired
full-coverage generation sets a bar that excludes every live row of its own family, and the
family disappears from its own comparison.
"""

from __future__ import annotations

import sqlite3

from case_studies.utils.backtest_explorer import BacktestExplorer
from tests.test_full_coverage_selection import _build_registry

# `deflated_sharpe` looks up the registered cohort row for the leader it ranks. With no such
# row every variant reports NULL selection-bias columns, which is the shape this fixture takes:
# what is under test is which variants the table ranks, not what it computes for them.
_COHORT_METRICS = """
    CREATE TABLE IF NOT EXISTS cohort_metrics (
        cohort_type TEXT, stage TEXT, leader_hash TEXT, k_variants INTEGER,
        n_trials_effective_mp REAL, n_trials_effective_er REAL,
        dsr_raw REAL, dsr_raw_pvalue REAL, dsr_mp REAL, dsr_mp_pvalue REAL,
        dsr_er REAL, dsr_er_pvalue REAL, expected_max_sharpe_er REAL,
        ras_leader REAL, ras_pvalue REAL, reality_check_pvalue REAL, pbo REAL
    );
"""


def _explorer(tmp_path):
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.executescript(_COHORT_METRICS)
    return BacktestExplorer("test", case_dir=case_dir)


def test_the_unscoped_table_is_unchanged(tmp_path) -> None:
    families = _explorer(tmp_path).compare_families()
    assert dict(zip(families["family"], families["n"], strict=True)) == {
        "gbm": 2,
        "tabular_dl": 1,
    }


def test_a_scoped_table_counts_only_the_population(tmp_path) -> None:
    families = _explorer(tmp_path).compare_families(prediction_hashes=["full_a", "tabular"])
    assert dict(zip(families["family"], families["n"], strict=True)) == {
        "gbm": 1,
        "tabular_dl": 1,
    }


def test_the_coverage_bar_is_taken_inside_the_population(tmp_path) -> None:
    """`partial` has the lowest coverage in `gbm` and is the only gbm row in this population.

    Unscoped it is dropped, because `full_a` and `full_b` set a bar it does not meet. Scoped,
    the bar is its own, and dropping it here would erase the family from the comparison
    entirely rather than narrowing it.
    """
    families = _explorer(tmp_path).compare_families(prediction_hashes=["partial", "tabular"])
    assert dict(zip(families["family"], families["n"], strict=True)) == {
        "gbm": 1,
        "tabular_dl": 1,
    }


def test_an_empty_population_compares_nothing(tmp_path) -> None:
    assert _explorer(tmp_path).compare_families(prediction_hashes=[]).is_empty()


def test_the_deflated_sharpe_table_counts_only_the_population(tmp_path) -> None:
    """K is "how many variants were tried", and a retired generation is not one of them.

    Left unscoped the table ranks retired backtests beside live ones, so the leader whose
    selection bias is being corrected can be a configuration its own publisher has replaced,
    and the correction is computed against a trial count that includes it.
    """
    explorer = _explorer(tmp_path)

    assert explorer.deflated_sharpe(top_n=10).height == 3
    scoped = explorer.deflated_sharpe(top_n=10, prediction_hashes=["full_a", "tabular"])
    assert scoped.height == 2
    assert set(scoped["source"].to_list()) == {"gbm/full_a", "tabular_dl/tabular"}


def test_the_search_context_counts_only_the_population(tmp_path) -> None:
    """ "How exceptional is the leader" is a question about the trials that were run.

    A retired generation's backtests are still registered, so an unscoped count prices the
    selection against a sweep larger than the one the shortlist was drawn from - and the
    leader's percentile is computed within a distribution it was never compared against.
    """
    explorer = _explorer(tmp_path)

    assert explorer.search_context()["total"] == 3
    assert explorer.search_context(prediction_hashes=["full_a", "tabular"])["total"] == 2
    assert explorer.search_context(prediction_hashes=[]) == {}
