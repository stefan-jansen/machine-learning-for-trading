from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from case_studies.research import CausalResult, LabelDefinition, Study
from case_studies.utils import causal
from case_studies.utils.registry.specs import training_hash_from_spec
from tests.test_research_workspace import _seed_release


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _causal_fixture(tmp_path, monkeypatch, entity: str = "symbol"):
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    setup_path = study.root / "config" / "setup.yaml"
    setup_path.write_text(
        setup_path.read_text() + "causal:\n  treatment: treatment\n  confounders: [confounder]\n"
    )
    rows = []
    for timestamp_index, timestamp in enumerate(
        pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 9, 16),
            interval="8h",
            eager=True,
            time_zone="UTC",
        )
    ):
        for symbol_index in range(6):
            rows.append(
                {
                    entity: f"S{symbol_index}",
                    "timestamp": timestamp,
                    "feature": float(symbol_index),
                    "treatment": float(symbol_index) + timestamp_index / 100,
                    "confounder": float(timestamp_index % 5),
                    "fwd_ret_8h": (symbol_index - 2.5) / 100 + timestamp_index / 10000,
                }
            )
    frame = pl.DataFrame(rows)
    label = study.labels.publish(
        LabelDefinition("fwd_ret_8h", "regression", "8H"),
        frame.rename({entity: "symbol"}).select("symbol", "timestamp", "fwd_ret_8h"),
    )
    mds = SimpleNamespace(
        dataset=frame,
        feature_names=["feature", "treatment", "confounder"],
        label_col="fwd_ret_8h",
        label_buffer="8H",
        date_col="timestamp",
        entity_cols=[entity],
        input_lineage={
            "artifacts": {"financial": {"sha256": "features-v1", "size": 1}},
            "fingerprint": "fixture-v1",
        },
    )
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *args, **kwargs: mds)
    monkeypatch.setattr(
        "utils.modeling.load_configs",
        lambda *args, **kwargs: [
            {
                "config_name": "dml",
                "family": "causal_dml",
                "n_folds": 5,
                "n_placebo": 100,
                "params": {"max_depth": 3, "max_iter": 50},
                "seed": 42,
            }
        ],
    )
    return study, label, frame


def test_causal_request_resolves_timing_gap_and_preview_identity(tmp_path, monkeypatch) -> None:
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch)

    resolved = study.causal(
        method="dml",
        label=label.name,
        execution_tier="preview",
        preview_reductions={
            "max_samples": 240,
            "max_symbols": 6,
            "n_folds": 2,
            "n_placebo": 10,
        },
    ).resolve()
    computation = resolved.spec["computation"]

    assert resolved.spec["identity_version"] == 3
    assert computation["estimand"]["outcome"] == "fwd_ret_8h"
    assert computation["estimand"]["treatment"] == "treatment"
    assert computation["estimand"]["treatment_observed_at"] == "decision_timestamp"
    assert computation["refutation"]["temporal_gap_policy"] == "reset"
    assert computation["refutation"]["observation_cadence"] == "0 days 08:00:00"
    assert computation["analysis_population"]["n_rows"] <= 240
    assert computation["model"]["nuisance_params"]["max_iter"] == 50
    assert computation["preview_reductions"]["n_folds"] == 2


def test_canonical_causal_uses_the_full_declared_population(tmp_path, monkeypatch) -> None:
    study, label, frame = _causal_fixture(tmp_path, monkeypatch)

    canonical = study.causal(method="dml", label=label.name).resolve()
    preview = study.causal(
        method="dml",
        label=label.name,
        execution_tier="preview",
        preview_reductions={
            "max_samples": 60,
            "max_symbols": 6,
            "n_folds": 2,
            "n_placebo": 10,
        },
    ).resolve()

    canonical_population = canonical.spec["computation"]["analysis_population"]
    preview_population = preview.spec["computation"]["analysis_population"]
    assert canonical_population["n_rows"] == frame.height
    assert canonical_population["max_samples"] == 0
    assert "preview_reductions" not in canonical.spec["computation"]
    assert preview_population["n_rows"] == 60
    assert preview_population["max_samples"] == 60
    assert canonical.identity != preview.identity


def _set_holdout_start(setup_path, value: str) -> None:
    setup_path.write_text(
        re.sub(
            r"holdout_start: *'?[^'\n]+'?",
            f"holdout_start: '{value}'",
            setup_path.read_text(),
        )
    )


def test_canonical_causal_population_stops_before_the_holdout_endpoint(
    tmp_path, monkeypatch
) -> None:
    """The DML sample must end one outcome horizon before the holdout, and fail closed."""
    study, label, frame = _causal_fixture(tmp_path, monkeypatch)
    setup_path = study.root / "config" / "setup.yaml"

    # The seeded holdout_start (2024-01-11) is past the fixture's last row, so the seal never
    # binds there. Move it inside the fixture's own span to exercise the cutoff itself.
    _set_holdout_start(setup_path, "2024-01-05")
    resolved = study.causal(method="dml", label=label.name).resolve()
    computation = resolved.spec["computation"]

    cutoff = datetime(2024, 1, 5, tzinfo=UTC) - timedelta(hours=8)
    expected = frame.filter(pl.col("timestamp") < cutoff)
    assert computation["estimand"]["holdout_endpoint_cutoff"] == cutoff.isoformat()
    assert computation["analysis_population"]["n_rows"] == expected.height
    # The seal must actually remove rows here, or this test would pass on a no-op filter.
    assert 0 < expected.height < frame.height

    # An empty pre-holdout frame fails closed rather than estimating on whatever is left.
    _set_holdout_start(setup_path, "2023-12-01")
    with pytest.raises(ValueError, match="empty pre-holdout analysis frame"):
        study.causal(method="dml", label=label.name).resolve()


def test_causal_pins_the_thread_pool_and_records_it_in_identity(tmp_path, monkeypatch) -> None:
    """The placebo loop is only bit-reproducible at a fixed thread count.

    `os.environ.setdefault("OMP_NUM_THREADS", "1")` at import does not achieve that: every
    notebook imports sklearn through case_studies.research first, so the pools are already
    built. Measured that way the openmp pool stays at 16. The limit therefore has to be
    applied at run time, and recorded, so two runs at different counts cannot share a hash.
    """
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch)
    resolved = study.causal(method="dml", label=label.name).resolve()

    numerics = resolved.spec["computation"]["numerics"]
    assert numerics["thread_limit"] == causal.DML_THREAD_LIMIT
    assert numerics["deterministic_reduction"] is True

    # The limit must be identity-bearing: a different count is a different result, not the
    # same result computed differently.
    other = deepcopy(resolved.spec)
    other["computation"]["numerics"]["thread_limit"] = causal.DML_THREAD_LIMIT + 1
    assert training_hash_from_spec(other) != training_hash_from_spec(resolved.spec)

    # The resolver's job is to pass the recorded limit down; binding the pool is
    # run_dml_analysis's, so that direct callers get it too - see
    # test_run_dml_analysis_pins_the_pool_for_direct_callers.
    passed: list[int] = []

    def capture(*args, **kwargs):
        passed.append(kwargs["thread_limit"])
        return {
            "dml_result": {"theta": 0.02, "se_hac": 0.01, "n_obs": 120},
            "p_value_hac": 0.04,
            "naive_effect": 0.03,
            "confounding_bias_pct": 50.0,
            "refutation": {"empirical_p": 0.1},
            "started_at": "2026-08-15T00:00:00+00:00",
            "elapsed_s": 1.0,
        }

    monkeypatch.setattr(causal, "run_dml_analysis", capture)
    study.causal(method="dml", label=label.name).run()

    assert passed == [causal.DML_THREAD_LIMIT], (
        f"resolver passed thread_limit={passed}, not [{causal.DML_THREAD_LIMIT}]"
    )


def test_manual_dml_timeseries_pins_the_pool_for_every_caller(monkeypatch) -> None:
    """The pin lives where the nuisance models are fitted, so no caller can bypass it.

    Six case-study DML stages call run_dml_analysis directly, and the chapter-15 notebooks
    15_causal_estimation/03_econml_dml.py and 04_dml_crypto_regime.py call
    manual_dml_timeseries themselves at nine sites, with model_y/model_t unset so they fit the
    default HistGradientBoostingRegressor. All of them import sklearn before this module, so
    the OMP_NUM_THREADS setdefault is inert for every one - while
    cme_futures/12_model_analysis.py:1190 and sp500_options/11_model_analysis.py:992 tell the
    reader the nuisance models are pinned.
    """
    import threadpoolctl

    observed: list[int] = []
    original = causal._walk_forward_indices

    def record(*args, **kwargs):
        observed.extend(
            info["num_threads"]
            for info in threadpoolctl.threadpool_info()
            if info["user_api"] in {"openmp", "blas"}
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(causal, "_walk_forward_indices", record)

    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=(n, 1))
    t = x[:, 0] * 0.5 + rng.normal(scale=0.1, size=n)
    y = t * 0.3 + rng.normal(scale=0.1, size=n)
    causal.manual_dml_timeseries(y, t, x, n_folds=2, embargo=1)

    assert observed, "no thread pool was observed inside manual_dml_timeseries"
    assert set(observed) == {causal.DML_THREAD_LIMIT}, (
        f"manual_dml_timeseries ran with pools at {sorted(set(observed))}, "
        f"not {causal.DML_THREAD_LIMIT}"
    )


def test_run_dml_analysis_pins_the_naive_ols_too(monkeypatch) -> None:
    """The naive-OLS comparison runs after manual_dml_timeseries returns.

    np.linalg.lstsq on a tall design reaches threaded BLAS, so naive_effect - and
    confounding_bias_pct, which is a difference between it and the pinned theta - would vary
    with the ambient pool while the resolved spec records deterministic_reduction: True.
    Observed with manual_dml_timeseries stubbed, so only the outer pin can be in effect.
    """
    import threadpoolctl

    observed: list[int] = []
    n = 200

    def stub(Y, T, X, **kwargs):
        observed.extend(
            info["num_threads"]
            for info in threadpoolctl.threadpool_info()
            if info["user_api"] in {"openmp", "blas"}
        )
        rng = np.random.default_rng(1)
        return {
            "Y_res": rng.normal(size=n),
            "T_res": rng.normal(size=n),
            "theta": 0.02,
            "se_hac": 0.01,
            "n_obs": n,
            "t_stat": 2.0,
            "p_value_hac": 0.04,
            "hac_lags": 1,
            "n_entities": 1,
            "n_periods": n,
            "hac_maxlags": 1,
            "covariance_type": "newey_west",
        }

    monkeypatch.setattr(causal, "manual_dml_timeseries", stub)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 1))
    t = x[:, 0] * 0.5 + rng.normal(scale=0.1, size=n)
    y = t * 0.3 + rng.normal(scale=0.1, size=n)
    causal.run_dml_analysis(
        pd.DataFrame({"y": y, "t": t, "x": x[:, 0]}),
        "t",
        "y",
        ["x"],
        n_folds=2,
        embargo=1,
        n_placebo=0,
    )

    assert observed, "manual_dml_timeseries was not reached"
    assert set(observed) == {causal.DML_THREAD_LIMIT}, (
        f"run_dml_analysis ran with pools at {sorted(set(observed))}, not {causal.DML_THREAD_LIMIT}"
    )


def test_preview_causal_requires_every_reduction(tmp_path, monkeypatch) -> None:
    """A preview that omits max_samples would resolve the full population.

    This path no longer falls back to the shared preset's max_samples, so a partial
    reduction set is an uncapped preview - the opposite of what the tier is for. The
    guard's message already claimed every reduction; now it enforces it.
    """
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch)

    with pytest.raises(
        ValueError, match=r"must declare every reduction; missing \['max_samples'\]"
    ):
        study.causal(
            method="dml",
            label=label.name,
            execution_tier="preview",
            preview_reductions={"max_symbols": 6, "n_folds": 2, "n_placebo": 10},
        ).resolve()


def test_canonical_causal_ignores_a_sample_cap_declared_in_the_preset(
    tmp_path, monkeypatch
) -> None:
    """The shared preset cannot cap a canonical sample.

    config/dml/dml.yaml still declares max_samples for the six case-study DML stages that read
    it as their own default and have not migrated to this path. This resolver must ignore it:
    a canonical run uses the full declared population, and a reduction reaches it only through
    preview_reductions, which a canonical request refuses.
    """
    study, label, frame = _causal_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "utils.modeling.load_configs",
        lambda *args, **kwargs: [
            {
                "config_name": "dml",
                "family": "causal_dml",
                "max_samples": 50_000,
                "n_folds": 5,
                "n_placebo": 100,
                "params": {},
                "seed": 42,
            }
        ],
    )

    resolved = study.causal(method="dml", label=label.name).resolve()
    population = resolved.spec["computation"]["analysis_population"]

    assert population["max_samples"] == 0
    assert population["n_rows"] == frame.height

    with pytest.raises(ValueError, match="canonical causal requests cannot declare preview"):
        study.causal(
            method="dml", label=label.name, preview_reductions={"max_samples": 60}
        ).resolve()


def test_causal_run_registers_once_and_reopens_after_restart(tmp_path, monkeypatch) -> None:
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch)

    monkeypatch.setattr(
        causal,
        "run_dml_analysis",
        lambda *args, **kwargs: {
            "dml_result": {"theta": 0.02, "se_hac": 0.01, "n_obs": 120},
            "p_value_hac": 0.04,
            "naive_effect": 0.03,
            "confounding_bias_pct": 50.0,
            "refutation": {"empirical_p": 0.1},
            "started_at": "2026-08-13T00:00:00+00:00",
            "elapsed_s": 1.0,
        },
    )
    request = study.causal(
        method="dml",
        label=label.name,
        execution_tier="preview",
        preview_reductions={
            "max_samples": 240,
            "max_symbols": 6,
            "n_folds": 2,
            "n_placebo": 10,
        },
    )

    first = request.run()
    reopened = CausalResult.open(study, first.hash, include_preview=True)
    second = request.run()

    assert first.complete
    assert reopened.hash == first.hash == second.hash
    assert CausalResult.one(study, label=label.name, execution_tier="preview").hash == first.hash
    assert reopened.metrics["dml_effect"] == 0.02
    assert json.loads(json.dumps(reopened.spec)) == reopened.spec


def test_causal_cache_accepts_provenance_only_drift(tmp_path, monkeypatch) -> None:
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch)
    calls = 0

    def run_analysis(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "dml_result": {"theta": 0.02, "se_hac": 0.01, "n_obs": 120},
            "p_value_hac": 0.04,
            "naive_effect": 0.03,
            "confounding_bias_pct": 50.0,
            "refutation": {"empirical_p": 0.1},
        }

    monkeypatch.setattr(causal, "run_dml_analysis", run_analysis)
    resolved = study.causal(
        method="dml",
        label=label.name,
        execution_tier="preview",
        preview_reductions={
            "max_samples": 240,
            "max_symbols": 6,
            "n_folds": 2,
            "n_placebo": 10,
        },
    ).resolve()
    first = resolved.run()
    provenance_only = deepcopy(resolved.spec)
    provenance_only["provenance"]["baseline_commit"] = "new-provenance-only-commit"

    assert training_hash_from_spec(provenance_only) == first.hash
    second = causal.run_resolved_causal_request(study, provenance_only, resolved._context)

    assert second.hash == first.hash
    assert calls == 1


def test_block_permutation_resets_at_temporal_gap() -> None:
    values = np.arange(8)
    timestamps = np.array(
        [
            "2024-01-01T00:00:00",
            "2024-01-01T08:00:00",
            "2024-01-01T16:00:00",
            "2024-01-02T00:00:00",
            "2024-01-04T00:00:00",
            "2024-01-04T08:00:00",
            "2024-01-04T16:00:00",
            "2024-01-05T00:00:00",
        ],
        dtype="datetime64[s]",
    )

    permuted = causal.block_permute(
        values,
        block_size=2,
        rng=np.random.default_rng(7),
        groups=timestamps,
        expected_step="8h",
    )

    assert set(permuted[:4]) == set(values[:4])
    assert set(permuted[4:]) == set(values[4:])


@pytest.mark.parametrize("entity", ["symbol", "product"])
def test_causal_resolver_accepts_either_canonical_entity_key(tmp_path, monkeypatch, entity) -> None:
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch, entity=entity)

    resolved = study.causal(
        method="dml",
        label=label.name,
        execution_tier="preview",
        preview_reductions={"max_samples": 240, "max_symbols": 6, "n_folds": 2, "n_placebo": 10},
    ).resolve()

    assert resolved.spec["computation"]["analysis_population"]["n_rows"] > 0


def test_causal_resolver_rejects_an_unsupported_entity_key(tmp_path, monkeypatch) -> None:
    study, label, _frame = _causal_fixture(tmp_path, monkeypatch, entity="ticker")

    with pytest.raises(ValueError, match="does not support entity key 'ticker'"):
        study.causal(
            method="dml",
            label=label.name,
            execution_tier="preview",
            preview_reductions={
                "max_samples": 240,
                "max_symbols": 6,
                "n_folds": 2,
                "n_placebo": 10,
            },
        ).resolve()
