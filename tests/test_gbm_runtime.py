"""Execution-contract tests for the shared native LightGBM runner."""

from __future__ import annotations

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
