"""Execution-contract tests for the shared native LightGBM runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from case_studies.utils import gbm


def test_cpu_runtime_params_are_explicit_and_deterministic() -> None:
    params = gbm.lightgbm_runtime_params("cpu", num_threads=4, seed=17)

    assert params == {
        "device_type": "cpu",
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 4,
        "seed": 17,
        "data_random_seed": 17,
        "feature_fraction_seed": 17,
        "bagging_seed": 17,
        "drop_seed": 17,
        "extra_seed": 17,
        "objective_seed": 17,
    }


def test_cpu_runtime_params_reject_invalid_thread_count() -> None:
    with pytest.raises(ValueError, match="num_threads must be at least 1"):
        gbm.lightgbm_runtime_params("cpu", num_threads=0)


def test_gpu_runtime_params_fail_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gbm, "_best_gpu_device", lambda _library: None)

    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        gbm.lightgbm_runtime_params("gpu")


def test_gpu_runtime_params_record_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gbm, "_best_gpu_device", lambda _library: "cuda")

    assert gbm.lightgbm_runtime_params("cuda") == {"device_type": "cuda"}


def test_runtime_params_reject_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unsupported LightGBM device"):
        gbm.lightgbm_runtime_params("tpu")


def test_prepare_gbm_folds_keeps_continuous_classification_target() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "symbol": ["a", "b", "c"],
            "feature": [1.0, 2.0, 3.0],
            "label": [0.0, 1.0, 1.0],
            "return": [-0.2, 0.3, 0.4],
        }
    )
    splits = [
        {
            "fold": 0,
            "train_start": pd.Timestamp("2020-01-01"),
            "train_end": pd.Timestamp("2020-01-01"),
            "val_start": pd.Timestamp("2020-01-02"),
            "val_end": pd.Timestamp("2020-01-03"),
        }
    ]

    [fold] = gbm.prepare_gbm_folds(
        frame,
        splits,
        ["feature"],
        "label",
        "timestamp",
        "symbol",
        task_type="classification",
        class_values=[0, 1],
        eval_label_col="return",
    )

    np.testing.assert_array_equal(fold["y_val"], np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(fold["y_eval"], np.array([0.3, 0.4], dtype=np.float32))


def test_classification_ic_uses_continuous_eval_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_targets: list[np.ndarray] = []

    def capture_ic(frame, *_args, **_kwargs):
        captured_targets.append(frame["y_true"].to_numpy())
        return {"ic_mean": float(np.mean(frame["y_true"].to_numpy()))}

    monkeypatch.setattr(gbm, "cross_sectional_ic", capture_ic)
    rng = np.random.default_rng(11)
    x_train = rng.normal(size=(80, 2)).astype(np.float32)
    y_train = (x_train[:, 0] > 0).astype(np.float32)
    x_val = rng.normal(size=(20, 2)).astype(np.float32)
    y_val = (x_val[:, 0] > 0).astype(np.float32)
    y_eval = np.linspace(-0.5, 0.5, len(x_val), dtype=np.float32)
    dates = np.repeat(np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[D]"), 10)
    fold = {
        "fold": 0,
        "X_train": x_train,
        "y_train": y_train,
        "y_train_lgb": y_train,
        "X_val": x_val,
        "y_val": y_val,
        "y_val_lgb": y_val,
        "y_eval": y_eval,
        "dates": dates,
        "entities": np.tile(np.arange(10), 2),
        "n_train": len(x_train),
        "n_val": len(x_val),
    }
    config = {
        "config_name": "classification_eval_probe",
        "params": {"objective": "binary", "num_leaves": 3},
        "max_iterations": 2,
        "checkpoint_interval": 2,
    }

    result = gbm.train_gbm_config(
        config,
        [fold],
        feature_names=["x0", "x1"],
        device="cpu",
        num_threads=1,
        task_type="classification",
        class_values=[0, 1],
    )

    assert captured_targets
    assert all(np.array_equal(target, y_eval) for target in captured_targets)
    assert np.array_equal(result["predictions"][0]["y_true"], y_val)
    assert np.array_equal(result["predictions"][0]["y_eval"], y_eval)
