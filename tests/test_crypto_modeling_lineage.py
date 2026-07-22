"""Regression tests for Crypto model-input cache identity."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from case_studies.utils import deep_learning
from case_studies.utils.registry import (
    build_training_spec,
    modeling_input_fingerprint,
    training_hash_from_spec,
)


def test_crypto_training_hash_changes_with_modeling_artifact(tmp_path) -> None:
    """A rebuilt temporal artifact must not reuse a historical prediction hash."""
    case_dir = tmp_path / "crypto_perps_funding"
    inputs = {
        "features/financial.parquet": b"financial-v1",
        "features/model_based.parquet": b"temporal-v1",
        "labels/fwd_ret_8h.parquet": b"labels-v1",
        "config/setup.yaml": b"evaluation: label-clock-v1",
    }
    for relative, content in inputs.items():
        path = case_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    splits = [
        {
            "fold": 0,
            "train_start": datetime(2020, 1, 1, tzinfo=UTC),
            "train_end": datetime(2022, 12, 31, tzinfo=UTC),
            "val_start": datetime(2023, 1, 1, tzinfo=UTC),
            "val_end": datetime(2023, 12, 31, 8, tzinfo=UTC),
        }
    ]
    feature_names = ["financial_feature", "temporal_feature"]
    first = modeling_input_fingerprint(case_dir, "fwd_ret_8h", splits, feature_names, 0)
    old_spec = {"family": "linear", "label": "fwd_ret_8h", "seed": 42, "params": {}}
    first_spec = {
        **old_spec,
        "params": {"input_fingerprint": first, "max_symbols": 0},
    }

    (case_dir / "features/model_based.parquet").write_bytes(b"temporal-v2")
    second = modeling_input_fingerprint(case_dir, "fwd_ret_8h", splits, feature_names, 0)
    second_spec = {
        **old_spec,
        "params": {"input_fingerprint": second, "max_symbols": 0},
    }

    assert first != second
    assert training_hash_from_spec(old_spec) != training_hash_from_spec(first_spec)
    assert training_hash_from_spec(first_spec) != training_hash_from_spec(second_spec)


def test_crypto_hybrid_registry_uses_output_presets(tmp_path, monkeypatch) -> None:
    """Hybrid execution must resolve presets beside the isolated case output."""
    preset = tmp_path / "config" / "ols" / "ols.yaml"
    preset.parent.mkdir(parents=True)
    preset.write_text("model_class: LinearRegression\nparams: {}\n")
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))

    spec = build_training_spec("linear", "ols", "fwd_ret_8h", n_folds=2)

    assert spec["config_name"] == "ols"
    assert spec["family"] == "linear"
    assert spec["n_folds"] == 2


def test_crypto_dl_cuda_request_fails_closed(monkeypatch) -> None:
    """A CUDA-requested sequence model must never continue on CPU."""
    monkeypatch.setattr(deep_learning.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        deep_learning.run_dl_cv(
            None,
            [],
            configs=[],
            n_features=0,
            feature_names=[],
            label_col="fwd_ret_8h",
            date_col="timestamp",
            device="cuda",
        )


def test_crypto_dl_registration_keeps_current_lineage(tmp_path, monkeypatch) -> None:
    """A sequence-model checkpoint must retain the current input identity."""
    captured = {}
    preset = tmp_path / "config/lstm/lstm_h64.yaml"
    preset.parent.mkdir(parents=True)
    preset.write_text(
        "library: pytorch\nn_epochs: 2\nparams:\n  architecture: lstm\n  lookback: 3\n"
    )
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))

    def capture_training_run(_case_study_id, *, spec, **_kwargs):
        captured["spec"] = spec
        return training_hash_from_spec(spec)

    monkeypatch.setattr(
        "case_studies.utils.registry.registration.register_training_run",
        capture_training_run,
    )
    monkeypatch.setattr(
        "case_studies.utils.registry.registration.register_prediction_set",
        lambda *_args, **_kwargs: None,
    )

    deep_learning._register_dl_config(
        case_study="crypto_perps_funding",
        label="fwd_ret_8h",
        config_name="lstm_h64",
        architecture="lstm",
        n_epochs=2,
        best_epoch=1,
        lookback=3,
        n_folds=2,
        ic_mean=0.0,
        predictions=[],
        identity_params={"device": "cuda", "input_fingerprint": "current-lineage"},
    )

    assert captured["spec"]["params"]["device"] == "cuda"
    assert captured["spec"]["params"]["input_fingerprint"] == "current-lineage"


def test_crypto_dl_checkpoint_metric_equal_weights_decision_times() -> None:
    """Sequence checkpoints must weight each decision timestamp equally."""
    first = datetime(2023, 1, 1, tzinfo=UTC)
    second = datetime(2023, 1, 2, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "timestamp": [first] * 5 + [second] * 10,
            "symbol": [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(10)],
            "y_score": [float(i) for i in range(5)] + [float(i) for i in range(10)],
            "y_true": [float(i) for i in range(5)] + [float(9 - i) for i in range(10)],
        }
    )

    metrics = deep_learning._decision_time_checkpoint_metrics(
        frame,
        date_col="timestamp",
        entity_col="symbol",
    )

    assert metrics["ic_n_days"] == 2
    assert metrics["ic_mean"] == pytest.approx(0.0, abs=1e-12)
