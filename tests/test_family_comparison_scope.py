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

import polars as pl

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


def _register_family_cohort(case_dir, prediction_hash: str, *, k_variants: int) -> None:
    """Register a `cohort_metrics` row for the backtest on *prediction_hash*.

    `k_variants` is written independently of the rows actually present, because that is the
    situation under test: the correction was computed when the sweep was wider, and the table
    keeps it long after a later sweep narrowed the population.
    """
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        (leader_hash,) = db.execute(
            "SELECT backtest_hash FROM backtest_runs WHERE prediction_hash = ?",
            (prediction_hash,),
        ).fetchone()
        db.execute(
            "INSERT INTO cohort_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "family",
                "signal",
                leader_hash,
                k_variants,
                1.5,
                1.5,
                0.9,
                0.01,
                0.9,
                0.01,
                0.9,
                0.01,
                0.4,
                0.8,
                0.02,
                0.03,
                0.25,
            ),
        )


def _row(table, source: str) -> dict:
    return table.filter(pl.col("source") == source).row(0, named=True)


def test_a_cohort_matching_the_population_reports_its_correction(tmp_path) -> None:
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.executescript(_COHORT_METRICS)
    # gbm holds two full-coverage variants unscoped, which is what this correction was
    # computed over, so unscoped the stored cohort is the cohort being ranked.
    _register_family_cohort(case_dir, "full_a", k_variants=2)

    row = _row(BacktestExplorer("test", case_dir=case_dir).deflated_sharpe(top_n=10), "gbm/full_a")

    assert row["is_best"] is True
    assert row["deflated_sharpe"] == 0.9
    assert row["pbo"] == 0.25
    assert row["k_variants"] == 2
    assert row["k_variants_scoped"] is None


def test_a_correction_computed_over_a_wider_cohort_is_withheld_from_the_scoped_table(
    tmp_path,
) -> None:
    """The defect this covers: scoping `best` and reading `cohort_metrics` unscoped.

    A deflated Sharpe is a correction for how many variants were tried. Scoping removes
    variants, so the stored correction was computed against strictly more trials than the
    reader selected from, and reporting it here would describe a sweep that did not happen.
    Reporting it is the visible half; the dangerous half is that a scoped leader which was
    never the wider cohort's leader gets no row at all and so appears to need no correction.
    Both counts survive so the reader can see which population the correction belonged to.
    """
    case_dir = tmp_path / "case"
    _build_registry(case_dir)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.executescript(_COHORT_METRICS)
    _register_family_cohort(case_dir, "full_a", k_variants=2)

    scoped = BacktestExplorer("test", case_dir=case_dir).deflated_sharpe(
        top_n=10, prediction_hashes=["full_a", "tabular"]
    )
    row = _row(scoped, "gbm/full_a")

    assert row["is_best"] is True, "it is still the cohort's registered leader"
    assert row["deflated_sharpe"] is None
    assert row["dsr_pvalue"] is None
    assert row["significant"] is None
    assert row["pbo"] is None
    assert row["reality_check_pvalue"] is None
    assert row["k_variants"] == 2, "the count the stored correction was computed over"
    assert row["k_variants_scoped"] == 1, "the count in hand: scoping left one gbm variant"
    assert row["sharpe"] is not None, "the uncorrected Sharpe is unaffected by scoping"
