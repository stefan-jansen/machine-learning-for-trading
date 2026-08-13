from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from case_studies.research.causal import CausalRequest, CausalResult, causal_runtime_identity
from case_studies.research.workspace import Study
from case_studies.utils.registry import training_hash_from_spec
from tests.test_research_workspace import _seed_release


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    timestamps = np.repeat(pd.date_range("2020-01-01", periods=90, freq="B"), 2)
    confounder = rng.normal(size=len(timestamps))
    treatment = 0.4 * confounder + rng.normal(size=len(timestamps))
    outcome = 0.2 * treatment + 0.3 * confounder + rng.normal(size=len(timestamps))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": np.tile(["A", "B"], 90),
            "treatment": treatment,
            "fwd_ret_21d": outcome,
            "confounder": confounder,
        }
    )


def _request(study: Study, **changes: Any) -> CausalRequest:
    request: dict[str, Any] = {
        "label": "fwd_ret_21d",
        "treatment": "treatment",
        "confounders": ["confounder"],
        "n_folds": 2,
        "embargo": 1,
        "observation_frequency": "1D",
        "horizon": 1,
        "block_size": 2,
        "n_placebo": 10,
        "seed": 7,
        "time_col": "timestamp",
        "entity_col": "symbol",
        "development_end": "2020-06-01",
        "source_identity": {
            "label_artifact": "label-a",
            "feature_artifacts": {"financial": "features-a"},
        },
        "runtime_identity": {"python": "test", "sklearn": "test"},
        "notebook": "12_causal_dml",
    }
    request.update(changes)
    return CausalRequest(study=study, **request)


def test_causal_result_survives_restart_and_cannot_enter_strategy(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)

    result = _request(study).run(_panel())
    reopened_study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    reopened = CausalResult.open(reopened_study, result.hash)

    assert isinstance(reopened, CausalResult)
    assert reopened.complete
    assert reopened.spec()["input_identity"]["analysis_frame"]
    assert reopened.registry_record()["n_obs"] > 0
    with pytest.raises(TypeError, match="PredictionResult"):
        reopened_study.strategy(
            prediction=reopened,
            signal={"method": "equal_weight_top_k", "top_k": 1},
        )


def test_causal_preview_stays_out_of_canonical_registry(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)

    preview = _request(
        study,
        execution_tier="preview",
        preview_reductions={"max_decision_times": 80},
    ).run(_panel())

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM causal_runs").fetchone()[0] == 0
    assert CausalResult.open(study, preview.hash, include_preview=True).complete
    with pytest.raises(KeyError):
        CausalResult.open(study, preview.hash)


def test_causal_failure_does_not_register_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from case_studies.utils import causal as causal_adapter

    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )

    def fallback_result(*_args, **_kwargs):
        return {
            "dml_result": {
                "theta": 0.1,
                "se_hac": 0.02,
                "n_obs": 100,
                "covariance_type": "hc0_fallback",
                "covariance_error": "singular",
            },
            "p_value_hac": 0.01,
            "naive_effect": 0.2,
            "confounding_bias_pct": 50.0,
            "refutation": {"empirical_p": 0.01},
        }

    monkeypatch.setattr(causal_adapter, "run_dml_analysis", fallback_result)
    with pytest.raises(RuntimeError, match="required robust covariance"):
        _request(study).run(_panel())

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM causal_runs").fetchone()[0] == 0


def test_causal_resolver_pins_fold_boundaries_and_analysis_values(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    request = _request(study)
    panel = _panel()

    first, _ = request.resolve(panel)
    changed = panel.copy()
    changed.loc[changed.index[-1], "confounder"] += 1.0
    second, _ = request.resolve(changed)

    assert len(first["cv"]["folds"]) == 2
    assert first["cv"]["folds"][0]["test_start"] < first["cv"]["folds"][1]["test_start"]
    assert first["causal"]["hac_maxlags"] >= first["causal"]["horizon"] - 1
    assert training_hash_from_spec(first) != training_hash_from_spec(second)


def test_identical_complete_causal_result_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from case_studies.utils import causal as causal_adapter

    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    request = _request(study)
    first = request.run(_panel())

    def unexpected_recompute(*_args, **_kwargs):
        raise AssertionError("an exact complete causal result must be reused")

    monkeypatch.setattr(causal_adapter, "run_dml_analysis", unexpected_recompute)
    second = request.run(_panel())

    assert second.hash == first.hash
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM causal_runs").fetchone()[0] == 1


def test_causal_resolver_rejects_hac_bandwidth_beyond_resolved_periods(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )

    with pytest.raises(ValueError, match="requires more than"):
        _request(study, hac_maxlags=100).resolve(_panel())


def test_causal_runtime_identity_records_numerical_packages() -> None:
    runtime = causal_runtime_identity()

    assert runtime["python"]
    assert set(runtime["packages"]) == {
        "numpy",
        "pandas",
        "polars",
        "scikit-learn",
        "statsmodels",
    }


def test_study_causal_request_resolves_through_registered_adapter(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    direct = _request(study)
    public = study.causal(
        label="fwd_ret_21d",
        treatment="treatment",
        confounders=["confounder"],
        n_folds=2,
        embargo=1,
        observation_frequency="1D",
        horizon=1,
        block_size=2,
        n_placebo=10,
        seed=7,
        time_col="timestamp",
        entity_col="symbol",
        development_end="2020-06-01",
        source_identity={
            "label_artifact": "label-a",
            "feature_artifacts": {"financial": "features-a"},
        },
        runtime_identity={"python": "test", "sklearn": "test"},
        notebook="12_causal_dml",
        adapter="dml",
    )

    direct_spec, _ = direct.resolve(_panel())
    public_spec, _ = public.resolve(_panel())
    assert public_spec == direct_spec
    assert public_spec["causal"]["adapter"] == "dml"
    assert public_spec["causal"]["adapter_module"] == "case_studies.utils.causal"


def test_causal_resolver_rejects_duplicate_panel_keys(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    panel = _panel()
    duplicated = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate decision-time and entity"):
        _request(study).resolve(duplicated)
