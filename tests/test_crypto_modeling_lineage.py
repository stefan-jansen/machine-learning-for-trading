"""Regression tests for Crypto model-input cache identity."""

from __future__ import annotations

from datetime import UTC, datetime

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
