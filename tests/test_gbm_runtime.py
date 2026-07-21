"""Execution-contract tests for the shared native LightGBM runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from case_studies.utils import gbm, registry


def _gbm_prediction_entry(
    fold: int,
    timestamps: list[pd.Timestamp],
    prediction_sign: int,
    n_trees: int,
) -> dict:
    dates = np.repeat(np.array(timestamps, dtype="datetime64[ns]"), 5)
    actual = np.tile(np.arange(5, dtype=np.float64), len(timestamps))
    prediction = actual if prediction_sign > 0 else -actual
    return {
        "dates": dates,
        "entities": np.tile(np.arange(5), len(timestamps)),
        "y_true": actual,
        "y_eval": None,
        "y_pred": prediction,
        "fold": fold,
        "n_trees": n_trees,
    }


def _cached_prediction_frame(*, include_eval: bool = True) -> pl.DataFrame:
    timestamps = np.repeat(pd.date_range("2020-01-01", periods=2, freq="MS"), 5)
    actual = np.tile(np.arange(5, dtype=float), 2)
    frame = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": np.tile(np.arange(5), 2),
            "fold": np.repeat([0, 1], 5),
            "prediction": actual,
            "actual": (actual > 2).astype(float),
        }
    )
    if include_eval:
        frame = frame.with_columns(pl.Series("eval_actual", actual))
    return frame


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


def test_checkpoint_metrics_average_the_complete_monthly_series() -> None:
    predictions = [
        _gbm_prediction_entry(0, [pd.Timestamp("2020-01-01")], 1, 50),
        _gbm_prediction_entry(
            1,
            [
                pd.Timestamp("2020-02-01"),
                pd.Timestamp("2020-03-01"),
                pd.Timestamp("2020-04-01"),
            ],
            -1,
            50,
        ),
    ]

    metrics = gbm._checkpoint_metrics_from_predictions(predictions, [50])

    assert metrics[50]["ic_mean"] == pytest.approx(-0.5)
    assert metrics[50]["ic_std"] == pytest.approx(1.0)
    assert np.mean([1.0, -1.0]) == 0.0


def test_gbm_training_identity_covers_every_effective_input() -> None:
    config = {
        "family": "gbm",
        "config_name": "default_binary",
        "max_iterations": 500,
        "checkpoint_interval": 50,
    }
    split = {
        "fold": 0,
        "train_start": pd.Timestamp("2010-01-01"),
        "train_end": pd.Timestamp("2018-12-31"),
        "val_start": pd.Timestamp("2019-01-01"),
        "val_end": pd.Timestamp("2019-12-31"),
    }
    runtime = gbm.lightgbm_runtime_params("cpu", num_threads=8, seed=42)
    baseline = gbm.build_gbm_training_spec(
        config,
        label_col="fwd_class_1m",
        n_folds=1,
        max_bin=255,
        runtime_params=runtime,
        feature_names=["value", "size"],
        splits=[split],
        eval_label_col="fwd_ret_1m",
        task_type="classification",
        class_values=[0, 1],
        seed=42,
    )
    baseline_hash = registry.training_hash_from_spec(baseline)
    variants = [
        {"feature_names": ["value", "quality"]},
        {"eval_label_col": "fwd_ret_3m"},
        {"splits": [{**split, "val_end": pd.Timestamp("2020-01-31")}]},
        {"seed": 17, "runtime_params": gbm.lightgbm_runtime_params("cpu", seed=17)},
    ]
    for changes in variants:
        kwargs = {
            "label_col": "fwd_class_1m",
            "n_folds": 1,
            "max_bin": 255,
            "runtime_params": runtime,
            "feature_names": ["value", "size"],
            "splits": [split],
            "eval_label_col": "fwd_ret_1m",
            "task_type": "classification",
            "class_values": [0, 1],
            "seed": 42,
            **changes,
        }
        assert (
            registry.training_hash_from_spec(gbm.build_gbm_training_spec(config, **kwargs))
            != baseline_hash
        )


def test_gbm_cache_requires_curves_and_continuous_eval_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    frame = _cached_prediction_frame(include_eval=False)
    prediction_dir = tmp_path / "prediction"
    prediction_dir.mkdir()
    frame.write_parquet(prediction_dir / "predictions.parquet")
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    monkeypatch.setattr(
        registry,
        "load_prediction_sets",
        lambda *_args, **_kwargs: pl.DataFrame(
            {
                "prediction_hash": ["prediction"],
                "checkpoint_value": [100],
                "checkpoint_kind": ["iteration"],
            }
        ),
    )
    monkeypatch.setattr(registry, "prediction_dir", lambda *_args: prediction_dir)
    monkeypatch.setattr(registry, "get_training_dir", lambda *_args: training_dir)
    monkeypatch.setattr(
        registry,
        "load_prediction_metrics",
        lambda *_args, **_kwargs: pl.DataFrame({"ic_mean": [1.0], "ic_std": [0.0]}),
    )
    keys = _cached_prediction_frame().select("timestamp", "symbol", "fold")

    with pytest.raises(ValueError, match="learning curves"):
        gbm.load_cached_gbm_config(
            case_study="probe",
            training_spec={"family": "gbm", "label": "label", "seed": 42},
            config_name="default_binary",
            prediction_split="validation",
            date_col="timestamp",
            entity_col="symbol",
            eval_col="eval_actual",
            expected_iterations=(50, 100),
            expected_keys=keys,
        )

    pl.DataFrame(
        {
            "config": ["default_binary", "default_binary"],
            "iteration": [50, 100],
            "ic_mean": [0.5, 1.0],
            "ic_std": [0.1, 0.0],
        }
    ).write_parquet(training_dir / "learning_curves.parquet")
    with pytest.raises(ValueError, match="eval_actual"):
        gbm.load_cached_gbm_config(
            case_study="probe",
            training_spec={"family": "gbm", "label": "label", "seed": 42},
            config_name="default_binary",
            prediction_split="validation",
            date_col="timestamp",
            entity_col="symbol",
            eval_col="eval_actual",
            expected_iterations=(50, 100),
            expected_keys=keys,
        )


def test_complete_gbm_cache_replays_the_canonical_metric(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    frame = _cached_prediction_frame()
    prediction_dir = tmp_path / "prediction"
    prediction_dir.mkdir()
    frame.write_parquet(prediction_dir / "predictions.parquet")
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    pl.DataFrame(
        {
            "config": ["default_binary", "default_binary"],
            "iteration": [50, 100],
            "ic_mean": [0.5, 1.0],
            "ic_std": [0.1, 0.0],
        }
    ).write_parquet(training_dir / "learning_curves.parquet")
    monkeypatch.setattr(
        registry,
        "load_prediction_sets",
        lambda *_args, **_kwargs: pl.DataFrame(
            {
                "prediction_hash": ["prediction"],
                "checkpoint_value": [100],
                "checkpoint_kind": ["iteration"],
            }
        ),
    )
    monkeypatch.setattr(registry, "prediction_dir", lambda *_args: prediction_dir)
    monkeypatch.setattr(registry, "get_training_dir", lambda *_args: training_dir)
    monkeypatch.setattr(
        registry,
        "load_prediction_metrics",
        lambda *_args, **_kwargs: pl.DataFrame({"ic_mean": [1.0], "ic_std": [0.0]}),
    )

    result, curves = gbm.load_cached_gbm_config(
        case_study="probe",
        training_spec={"family": "gbm", "label": "label", "seed": 42},
        config_name="default_binary",
        prediction_split="validation",
        date_col="timestamp",
        entity_col="symbol",
        eval_col="eval_actual",
        expected_iterations=(50, 100),
        expected_keys=frame.select("timestamp", "symbol", "fold"),
    )

    assert result["cached"] is True
    assert result["best_iter"] == 100
    assert result["best_ic"] == 1.0
    assert len(curves) == 2


def test_gbm_registration_uses_the_lookup_spec_and_iteration_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured = {}
    spec = {
        "family": "gbm",
        "config_name": "default_binary",
        "label": "fwd_class_1m",
        "n_folds": 1,
        "seed": 17,
        "params": {"runtime": {"seed": 17}},
    }

    def capture_training(_case_study, *, spec, **_kwargs):
        captured["spec"] = spec
        return "training"

    monkeypatch.setattr(registry, "register_training_run", capture_training)

    def capture_prediction(*_args, **kwargs):
        captured["prediction"] = kwargs
        return "prediction"

    monkeypatch.setattr(registry, "register_prediction_set", capture_prediction)
    monkeypatch.setattr(registry, "get_training_dir", lambda *_args: tmp_path)
    result = {
        "config_name": "default_binary",
        "best_iter": 100,
        "best_ic": 0.25,
        "best_ic_std": 0.10,
        "elapsed_s": 1.0,
        "learning_curves": [],
        "fold_metrics": [],
        "predictions": [
            {
                "dates": np.array(["2020-01-01"], dtype="datetime64[D]"),
                "entities": np.array(["a"]),
                "y_true": np.array([1.0]),
                "y_eval": np.array([0.2]),
                "y_pred": np.array([0.4]),
                "fold": 0,
                "n_trees": 100,
            }
        ],
    }
    config = {
        "family": "gbm",
        "config_name": "default_binary",
        "checkpoint_interval": 50,
    }

    gbm.register_gbm_result(
        "probe",
        result,
        config,
        "fwd_class_1m",
        n_folds=1,
        max_bin=255,
        task_type="classification",
        class_values=[0, 1],
        eval_col="eval_actual",
        training_spec=spec,
    )

    assert captured["spec"] == spec
    assert captured["prediction"]["checkpoint_value"] == 100
    assert captured["prediction"]["checkpoint_kind"] == "iteration"
    assert captured["prediction"]["eval_col"] == "eval_actual"
