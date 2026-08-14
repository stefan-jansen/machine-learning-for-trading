"""Idempotence contract for causal-result registration."""

from __future__ import annotations

import sqlite3

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
