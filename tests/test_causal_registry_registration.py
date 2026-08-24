"""Idempotence contract for causal-result registration."""

from __future__ import annotations

import sqlite3

from case_studies.utils.causal import REFUTATION_ALPHA, classify_refutation
from case_studies.utils.registry.registration import register_causal_run


def _register(case_dir, *, effect: float, started_at: str, elapsed_s: float) -> None:
    register_causal_run(
        "test_case",
        "causal123",
        label="fwd_ret_5d",
        treatment="ivrv_spread",
        confounders_json='["rv_20"]',
        embargo=10,
        n_folds=5,
        n_obs=100,
        dml_effect=effect,
        dml_se_hac=0.02,
        p_value_hac=0.25,
        naive_effect=-0.02,
        confounding_bias_pct=-0.5,
        refutation_p=0.01,
        spec_json='{"family":"causal_dml"}',
        notebook="12_causal_dml",
        started_at=started_at,
        elapsed_s=elapsed_s,
        case_dir=case_dir,
    )


def _row(case_dir) -> tuple:
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        return db.execute(
            "SELECT dml_effect, started_at, elapsed_s, git_commit, created_at "
            "FROM causal_runs WHERE causal_hash='causal123'"
        ).fetchone()


def test_identical_causal_result_does_not_refresh_execution_provenance(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    _register(case_dir, effect=-0.0228, started_at="first", elapsed_s=10.0)
    first = _row(case_dir)

    _register(case_dir, effect=-0.0228, started_at="second", elapsed_s=20.0)
    assert _row(case_dir) == first

    _register(case_dir, effect=-0.0230, started_at="third", elapsed_s=30.0)
    changed = _row(case_dir)
    assert changed[0:3] == (-0.0230, "third", 30.0)
    assert changed[4] == first[4]


def test_the_draw_count_is_persisted_so_the_verdict_can_be_recomputed(tmp_path) -> None:
    """A reader holding only `refutation_p` cannot tell a run that failed from one whose
    draws could never have rejected. The plus-one correction floors the p-value at
    1 / (n + 1), so a preview run of ten draws reports 0.09 at best and any bare
    threshold republishes it as "Fails". Persisting the count is what lets every reader
    reach the same verdict from the same rule."""
    case_dir = tmp_path / "test_case"
    register_causal_run(
        "test_case",
        "causal_underpowered",
        label="fwd_ret_5d",
        treatment="ivrv_spread",
        confounders_json='["rv_20"]',
        embargo=10,
        n_folds=5,
        n_obs=100,
        dml_effect=-0.02,
        dml_se_hac=0.02,
        p_value_hac=0.25,
        naive_effect=-0.02,
        confounding_bias_pct=-0.5,
        refutation_p=1 / 11,
        refutation_n_successful=10,
        spec_json='{"family":"causal_dml"}',
        notebook="12_causal_dml",
        started_at="first",
        elapsed_s=1.0,
        case_dir=case_dir,
    )

    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        stored = db.execute(
            "SELECT refutation_p, refutation_n_successful FROM causal_runs "
            "WHERE causal_hash='causal_underpowered'"
        ).fetchone()

    assert stored[1] == 10
    assert classify_refutation(stored[0], stored[1]) == "Underpowered"
    # The two-way rule a reader would otherwise apply to the same p-value.
    assert ("Passes" if stored[0] < REFUTATION_ALPHA else "Fails") == "Fails"


def test_a_registry_written_before_the_draw_count_existed_gains_the_column(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    _register(case_dir, effect=-0.0228, started_at="first", elapsed_s=10.0)
    db_path = case_dir / "run_log" / "registry.db"
    with sqlite3.connect(db_path) as db:
        db.execute("ALTER TABLE causal_runs DROP COLUMN refutation_n_successful")
        assert "refutation_n_successful" not in {
            row[1] for row in db.execute("PRAGMA table_info(causal_runs)").fetchall()
        }

    _register(case_dir, effect=-0.0228, started_at="second", elapsed_s=11.0)

    with sqlite3.connect(db_path) as db:
        assert "refutation_n_successful" in {
            row[1] for row in db.execute("PRAGMA table_info(causal_runs)").fetchall()
        }
