"""The registry row says which estimator produced `dml_se_hac`, and a reader can see it.

`causal_runs` stored `dml_se_hac` and `p_value_hac` with nothing that said what produced
them, so the two robust estimators - Driscoll-Kraay when the caller supplies decision-time
groups, Newey-West when it does not - were indistinguishable in the row, and so was an HC0
fallback. A column, not part of `spec_json`, so it moves no `causal_hash` and invalidates
no registered row: the same trade `refutation_n_successful`, `refutation_placebo_json` and
`refutation_frozen_fraction` were added under.

A row written before the column carries NULL, which is the truthful answer rather than a
guess. The value cannot be reconstructed: the fallback copied the HC0 number into `se_hac`
bit for bit, so nothing stored separates it from a robust result.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from case_studies.research.causal import CausalResult
from case_studies.utils import causal
from case_studies.utils.registry.registration import register_causal_run


def _study(case_dir):
    return SimpleNamespace(root=case_dir, output_root=None, release_case_root=case_dir)


def _fields(covariance_type: str | None) -> dict:
    return {
        "label": "fwd_ret_5d",
        "treatment": "ivrv_spread",
        "confounders_json": '["rv_20"]',
        "embargo": 10,
        "n_folds": 5,
        "n_obs": 100,
        "dml_effect": -0.02,
        "dml_se_hac": 0.02,
        "covariance_type": covariance_type,
        "p_value_hac": 0.25,
        "naive_effect": -0.02,
        "confounding_bias_pct": -0.5,
        "refutation_p": 0.1,
        "refutation_n_successful": 40,
        "spec_json": '{"family":"causal_dml","identity_version":3}',
        "notebook": "12_causal_dml",
        "elapsed_s": 1.0,
    }


def test_the_estimator_name_reaches_the_row_and_reads_back(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    register_causal_run(
        "test_case", "causal_dk", started_at="first", case_dir=case_dir, **_fields("driscoll_kraay")
    )

    result = CausalResult.open(_study(case_dir), "causal_dk")

    assert result.metrics["covariance_type"] == "driscoll_kraay"


def test_a_registry_written_before_the_column_existed_can_still_be_read(tmp_path) -> None:
    """`CausalResult.open` reads through a plain connection rather than the migrating
    opener, so naming the column unconditionally raises `OperationalError` on every
    pre-migration registry - the failure `refutation_n_successful` already caused once."""
    case_dir = tmp_path / "test_case"
    register_causal_run(
        "test_case", "causal_dk", started_at="first", case_dir=case_dir, **_fields("driscoll_kraay")
    )
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute("ALTER TABLE causal_runs DROP COLUMN covariance_type")

    result = CausalResult.open(_study(case_dir), "causal_dk")

    assert result.metrics["covariance_type"] is None
    assert result.metrics["dml_se_hac"] == pytest.approx(0.02)


def test_filling_the_column_on_an_existing_row_is_not_a_conflict(tmp_path) -> None:
    """An identity-version-3 row is immutable. A row registered before the column existed
    carries NULL, and a re-registration that knows the estimator fills it; treating that as
    a conflict would make the migration break re-registration of identical results."""
    case_dir = tmp_path / "test_case"
    register_causal_run(
        "test_case", "causal_immutable", started_at="first", case_dir=case_dir, **_fields(None)
    )
    register_causal_run(
        "test_case",
        "causal_immutable",
        started_at="second",
        case_dir=case_dir,
        **_fields("driscoll_kraay"),
    )

    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        stored = db.execute(
            "SELECT covariance_type FROM causal_runs WHERE causal_hash = ?", ("causal_immutable",)
        ).fetchone()[0]
    assert stored == "driscoll_kraay"


def test_changing_a_recorded_estimator_is_a_conflict(tmp_path) -> None:
    """Filling NULL is a migration; changing one name to another says the same identity
    produced its standard error two different ways, which is exactly what immutability is
    for. Without this the column would be silently rewritable and could not be trusted."""
    case_dir = tmp_path / "test_case"
    register_causal_run(
        "test_case",
        "causal_immutable",
        started_at="first",
        case_dir=case_dir,
        **_fields("driscoll_kraay"),
    )

    with pytest.raises(ValueError, match="covariance_type"):
        register_causal_run(
            "test_case",
            "causal_immutable",
            started_at="second",
            case_dir=case_dir,
            **_fields("newey_west"),
        )


def test_the_production_path_records_an_estimator_rather_than_null(tmp_path) -> None:
    """The guard against the next caller forgetting.

    `covariance_type` defaults to None on `register_causal_run` so the pre-migration
    meaning stays available, which means a production path that omits it writes NULL and
    reads back as "written before the column existed". Only `causal.py` registers rows in
    production, and it must pass what `manual_dml_timeseries` reported.
    """
    case_dir = tmp_path / "test_case"
    results = {
        "dml_result": {
            "theta": 0.02,
            "se_hac": 0.01,
            "n_obs": 120,
            "covariance_type": "driscoll_kraay",
        },
        "p_value_hac": 0.04,
        "naive_effect": 0.03,
        "confounding_bias_pct": 50.0,
        "refutation": {"empirical_p": 0.1, "n_successful": 40},
    }
    causal_hash = causal.register_causal_run(
        "test_case",
        "fwd_ret_5d",
        results,
        treatment_col="ivrv_spread",
        confounder_cols=["rv_20"],
        n_folds=5,
        embargo=10,
        block_size=126,
        n_placebo=40,
        case_dir=case_dir,
    )

    assert CausalResult.open(_study(case_dir), causal_hash).metrics["covariance_type"] == (
        "driscoll_kraay"
    )
