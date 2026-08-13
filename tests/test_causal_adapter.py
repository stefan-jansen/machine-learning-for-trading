from __future__ import annotations

import json
import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
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


def _causal_fixture(tmp_path, monkeypatch):
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
                    "symbol": f"S{symbol_index}",
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
        frame.select("symbol", "timestamp", "fwd_ret_8h"),
    )
    mds = SimpleNamespace(
        dataset=frame,
        feature_names=["feature", "treatment", "confounder"],
        label_col="fwd_ret_8h",
        label_buffer="8H",
        date_col="timestamp",
        entity_cols=["symbol"],
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
                "max_samples": 50000,
                "n_folds": 5,
                "n_placebo": 100,
                "params": {"max_depth": 3, "max_iter": 50},
                "seed": 42,
            }
        ],
    )
    return study, label


def test_causal_request_resolves_timing_gap_and_preview_identity(tmp_path, monkeypatch) -> None:
    study, label = _causal_fixture(tmp_path, monkeypatch)

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


def test_causal_run_registers_once_and_reopens_after_restart(tmp_path, monkeypatch) -> None:
    study, label = _causal_fixture(tmp_path, monkeypatch)

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
    study, label = _causal_fixture(tmp_path, monkeypatch)
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
