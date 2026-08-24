"""Idempotence contract for causal-result registration."""

from __future__ import annotations

import sqlite3

import pytest

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


_RESULTS = {
    "dml_result": {"n_obs": 100, "theta": -0.0008, "se_hac": 0.01},
    "refutation": {"empirical_p": 0.4},
    "p_value_hac": 0.3,
    "naive_effect": -0.0042,
    "confounding_bias_pct": -450.0,
}

_BASE = {
    "label": "fwd_ret_1m",
    "treatment_col": "r12_2",
    "confounder_cols": ["Beta", "IdioVol"],
    "n_folds": 5,
    "embargo": 1,
    "time_col": "timestamp",
    "block_size": 1,
    "n_placebo": 100,
    "seed": 42,
    "horizon": 0,
    "max_samples": 250_000,
    "max_symbols": 0,
    "development_end": "2015-12-01",
    "config_name": "dml_250k",
}

# One entry per knob that changes the estimate, with a value that differs from the
# baseline. A knob outside the identity lets two different estimates share one
# causal_hash, and registration then takes its ON CONFLICT DO UPDATE branch, so the
# second run replaces the first in place instead of landing beside it.
_VARIANTS = {
    "label": "fwd_class_1m",
    "treatment_col": "ST_REV",
    "confounder_cols": ["Beta", "IdioVol", "LME"],
    "n_folds": 3,
    "embargo": 2,
    "block_size": 3,
    "n_placebo": 50,
    "seed": 7,
    "horizon": 1,
    "max_samples": 50_000,
    "max_symbols": 5,
    "development_end": "2014-12-01",
    "config_name": "dml",
}


def _causal_hash(case_dir, **overrides) -> str:
    from case_studies.utils.causal import register_causal_run as register

    return register("test_case", results=_RESULTS, case_dir=case_dir, **{**_BASE, **overrides})


def test_identical_arguments_reproduce_the_causal_hash(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    assert _causal_hash(case_dir) == _causal_hash(case_dir)


@pytest.mark.parametrize("knob", sorted(_VARIANTS))
def test_every_knob_that_changes_the_estimate_changes_the_causal_hash(tmp_path, knob) -> None:
    case_dir = tmp_path / "test_case"
    baseline = _causal_hash(case_dir)
    assert _causal_hash(case_dir, **{knob: _VARIANTS[knob]}) != baseline, (
        f"{knob} is outside the causal identity"
    )


def test_registering_under_two_identities_keeps_both_rows(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    wide = _causal_hash(case_dir)
    narrow = _causal_hash(case_dir, max_samples=50_000)

    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        stored = {row[0] for row in db.execute("SELECT causal_hash FROM causal_runs")}
    assert stored == {wide, narrow}
