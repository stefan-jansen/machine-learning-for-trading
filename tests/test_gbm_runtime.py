"""Execution-contract tests for the shared native LightGBM runner."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

from case_studies.utils import gbm, registry
from case_studies.utils.registry import store
from utils.config import REPO_ROOT


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
    with pytest.raises(ValueError, match="num_threads must be at least 1"):
        gbm.lightgbm_runtime_params("cuda", num_threads=0)


def test_gpu_runtime_params_fail_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gbm, "_best_gpu_device", lambda _library: None)

    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        gbm.lightgbm_runtime_params("gpu")


def test_gpu_runtime_params_record_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gbm, "_best_gpu_device", lambda _library: "cuda")

    params = gbm.lightgbm_runtime_params("cuda", num_threads=3)
    assert params["device_type"] == "cuda"
    assert params["num_threads"] == 3
    assert {
        params[key]
        for key in (
            "bagging_seed",
            "data_random_seed",
            "drop_seed",
            "extra_seed",
            "feature_fraction_seed",
            "objective_seed",
            "seed",
        )
    } == {42}


def test_runtime_params_reject_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unsupported LightGBM device"):
        gbm.lightgbm_runtime_params("tpu")


def test_runtime_override_is_optional_and_normalizes_gpu_alias() -> None:
    assert gbm.resolve_gbm_device("", "cpu") == "cpu"
    assert gbm.resolve_gbm_device("cuda", "cpu") == "cuda"
    assert gbm.resolve_gbm_device(None, "gpu") == "cuda"


def test_execution_config_keeps_numerical_parameters_independent_of_device() -> None:
    cpu = gbm.resolve_gbm_execution_config({"device": "cpu", "max_bin": 63, "num_threads": 8})
    cuda = gbm.resolve_gbm_execution_config({"device": "cuda", "max_bin": 63, "num_threads": 8})

    assert cpu == ("cpu", 63, 8)
    assert cuda == ("cuda", 63, 8)


def test_execution_config_requires_declared_max_bin() -> None:
    with pytest.raises(ValueError, match="max_bin must be declared"):
        gbm.resolve_gbm_execution_config({"device": "cpu"})


def test_us_firm_gbm_defaults_use_the_reproducible_reader_backend() -> None:
    setup_path = REPO_ROOT / "case_studies/us_firm_characteristics/config/setup.yaml"
    setup = yaml.safe_load(setup_path.read_text())
    gbm_config = setup["modeling"]["gbm"]

    assert gbm.resolve_gbm_execution_config(gbm_config) == (
        "cpu",
        gbm.GBM_DEFAULT_MAX_BIN,
        gbm.DEFAULT_GBM_CPU_THREADS,
    )


def test_every_case_study_gbm_setup_resolves_the_shared_execution_contract() -> None:
    resolved = {}
    for setup_path in sorted((REPO_ROOT / "case_studies").glob("*/config/setup.yaml")):
        setup = yaml.safe_load(setup_path.read_text()) or {}
        gbm_config = (setup.get("modeling") or {}).get("gbm")
        if gbm_config is not None:
            resolved[setup_path.parents[1].name] = gbm.resolve_gbm_execution_config(gbm_config)

    assert set(resolved) == {
        "cme_futures",
        "crypto_perps_funding",
        "etfs",
        "fx_pairs",
        "nasdaq100_microstructure",
        "sp500_equity_option_analytics",
        "sp500_options",
        "us_equities_panel",
        "us_firm_characteristics",
    }
    assert {max_bin for _, max_bin, _ in resolved.values()} == {gbm.GBM_DEFAULT_MAX_BIN}


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


def test_cpu_training_repeats_bit_exactly() -> None:
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(240, 4)).astype(np.float32)
    y_train = (0.4 * x_train[:, 0] - 0.3 * x_train[:, 1]).astype(np.float32)
    x_val = rng.normal(size=(80, 4)).astype(np.float32)
    y_val = (0.4 * x_val[:, 0] - 0.3 * x_val[:, 1]).astype(np.float32)
    fold = {
        "fold": 0,
        "X_train": x_train,
        "y_train": y_train,
        "y_train_lgb": y_train,
        "X_val": x_val,
        "y_val": y_val,
        "y_val_lgb": y_val,
        "y_eval": None,
        "dates": np.repeat(
            np.array(pd.date_range("2020-01-01", periods=8, freq="MS"), dtype="datetime64[ns]"),
            10,
        ),
        "entities": np.tile(np.arange(10), 8),
        "n_train": len(x_train),
        "n_val": len(x_val),
    }
    config = {
        "config_name": "determinism_probe",
        "params": {
            "objective": "regression",
            "num_leaves": 7,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
        },
        "max_iterations": 20,
        "checkpoint_interval": 10,
    }

    def train_once():
        return gbm.train_gbm_config(
            config,
            [fold],
            feature_names=[f"x{i}" for i in range(4)],
            device="cpu",
            num_threads=1,
            seed=42,
            max_bin=gbm.GBM_DEFAULT_MAX_BIN,
        )

    first = train_once()
    second = train_once()
    np.testing.assert_array_equal(
        first["predictions"][0]["y_pred"], second["predictions"][0]["y_pred"]
    )
    assert first["learning_curves"] == second["learning_curves"]


def test_max_bin_changes_the_fitted_model() -> None:
    """max_bin is not cosmetic: it must be declared, never inherited from a device branch.

    Changing it moves every fitted result, which is why the correction from 63 to 255
    supersedes the populations fitted before it rather than adding to them. This pins the
    empirical consequence that
    ``test_gbm_training_identity_covers_every_declared_numerical_input`` pins for the hash.
    """
    n_dates, n_entities, n_features = 60, 20, 6
    rng = np.random.default_rng(0)
    n = n_dates * n_entities
    x = rng.normal(size=(n, n_features)).astype(np.float32)
    y = (0.4 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(scale=1.0, size=n)).astype(np.float32)
    dates = np.repeat(np.arange(n_dates), n_entities)
    entities = np.tile(np.arange(n_entities), n_dates).astype(str)

    folds = []
    cut = n_dates // 3
    for f in range(2):
        tr = dates < cut * (f + 1)
        va = (dates >= cut * (f + 1)) & (dates < cut * (f + 2))
        folds.append(
            {
                "fold": f,
                "X_train": x[tr],
                "y_train": y[tr],
                "y_train_lgb": y[tr],
                "X_val": x[va],
                "y_val": y[va],
                "y_val_lgb": y[va],
                "y_eval": None,
                "dates": dates[va],
                "entities": entities[va],
                "n_train": int(tr.sum()),
                "n_val": int(va.sum()),
            }
        )

    config = {
        "config_name": "max_bin_probe",
        "family": "gbm",
        "max_iterations": 40,
        "checkpoint_interval": 20,
        "params": {
            "objective": "regression_l1",
            "num_leaves": 7,
            "learning_rate": 0.1,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": 42,
        },
    }

    def train(max_bin: int) -> dict[tuple[int, int], np.ndarray]:
        result = gbm.train_gbm_config(
            dict(config),
            folds,
            feature_names=[f"f{i}" for i in range(n_features)],
            device="cpu",
            max_bin=max_bin,
            entity_col="symbol",
            date_col="date",
            task_type="regression",
            save_dir=None,
        )
        # Key on (fold, n_trees) so the comparison is checkpoint-for-checkpoint.
        return {(e["fold"], e["n_trees"]): np.asarray(e["y_pred"]) for e in result["predictions"]}

    # Compare the predictions themselves, not an aggregate IC: two binnings can
    # preserve the cross-sectional ranking and so score an identical Spearman IC
    # while fitting different models, which would let this guard pass vacuously.
    coarse, fine = train(63), train(255)
    assert coarse.keys() == fine.keys(), "max_bin must not change the checkpoint grid"
    assert any(not np.array_equal(coarse[k], fine[k]) for k in coarse), (
        "max_bin should change the fitted model; a guard has gone stale"
    )


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


def test_gbm_training_identity_covers_every_declared_numerical_input() -> None:
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
    baseline = gbm.build_gbm_training_spec(
        config,
        label_col="fwd_class_1m",
        n_folds=1,
        max_bin=255,
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
        {"seed": 17},
        {"max_bin": 127},
    ]
    for changes in variants:
        kwargs = {
            "label_col": "fwd_class_1m",
            "n_folds": 1,
            "max_bin": 255,
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


def test_gbm_training_identity_excludes_runtime_provenance() -> None:
    config = {
        "family": "gbm",
        "config_name": "default_binary",
        "max_iterations": 500,
        "checkpoint_interval": 50,
    }
    kwargs = {
        "label_col": "fwd_class_1m",
        "n_folds": 1,
        "max_bin": 63,
        "feature_names": ["value", "size"],
        "splits": [
            {
                "fold": 0,
                "train_start": pd.Timestamp("2010-01-01"),
                "train_end": pd.Timestamp("2018-12-31"),
                "val_start": pd.Timestamp("2019-01-01"),
                "val_end": pd.Timestamp("2019-12-31"),
            }
        ],
        "eval_label_col": "fwd_ret_1m",
        "task_type": "classification",
        "class_values": [0, 1],
        "seed": 42,
    }

    identity = gbm.build_gbm_training_spec(config, **kwargs)
    assert "device_type" not in identity["params"]
    assert "num_threads" not in identity["params"]


def test_training_registration_records_runtime_without_changing_hash(tmp_path) -> None:
    spec = {
        "config_name": "probe",
        "family": "gbm",
        "feature_sets": ["financial"],
        "label": "fwd_ret_1m",
        "library": "lightgbm",
        "n_folds": 1,
        "params": {"max_bin": 63},
        "seed": 42,
    }
    runtime = {"device_type": "cpu", "num_threads": 8}

    training_hash = registry.register_training_run(
        "probe",
        spec,
        case_dir=tmp_path,
        runtime_provenance=runtime,
    )

    assert training_hash == registry.training_hash_from_spec(spec)
    assert (
        json.loads((tmp_path / "run_log" / "training" / training_hash / "runtime.json").read_text())
        == runtime
    )
    with closing(sqlite3.connect(tmp_path / "run_log" / "registry.db")) as db:
        [runtime_json] = db.execute(
            "SELECT runtime_json FROM training_runs WHERE training_hash = ?", (training_hash,)
        ).fetchone()
    assert json.loads(runtime_json) == runtime


def test_legacy_huber_registration_hashes_declared_fold_scale(
    monkeypatch,
    tmp_path,
) -> None:
    captured = {}

    def capture_training(_case_study, *, spec, **_kwargs):
        captured["spec"] = spec
        return registry.training_hash_from_spec(spec)

    monkeypatch.setattr(registry, "register_training_run", capture_training)
    monkeypatch.setattr(registry, "get_training_dir", lambda *_args: tmp_path)
    monkeypatch.setattr(registry, "register_prediction_set", lambda *_args, **_kwargs: "unused")
    result = {
        "best_iter": 50,
        "best_ic": 0.0,
        "best_ic_std": 0.0,
        "config_name": "default_huber",
        "elapsed_s": 0.1,
        "fold_metrics": [],
        "learning_curves": [],
        "predictions": [],
    }
    config = {
        "checkpoint_interval": 50,
        "config_name": "default_huber",
        "family": "gbm",
    }

    gbm.register_gbm_result(
        "probe",
        result,
        config,
        "fwd_ret_1m",
        n_folds=2,
        max_bin=63,
    )

    assert captured["spec"]["params"]["huber_alpha_scale"] == 0.5


def test_runtime_provenance_migrates_a_training_only_registry(tmp_path) -> None:
    run_log = tmp_path / "run_log"
    run_log.mkdir()
    with closing(sqlite3.connect(run_log / "registry.db")) as db:
        db.execute(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                label TEXT NOT NULL,
                config_name TEXT,
                spec_json TEXT,
                created_at TEXT NOT NULL,
                git_commit TEXT,
                entry_point TEXT,
                started_at TEXT,
                elapsed_s REAL
            )
            """
        )

    db = store._open_registry(tmp_path)
    try:
        columns = {row[1] for row in db.execute("PRAGMA table_info(training_runs)")}
    finally:
        db.close()
    assert "runtime_json" in columns


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
        return registry.training_hash_from_spec(spec)

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


def test_crypto_gbm_takes_its_device_from_setup_yaml() -> None:
    """The device the boundary fits on is the one ``setup.yaml`` declares.

    ``07_gbm`` no longer names a device. It calls the shared model boundary, which
    reads ``modeling.gbm`` from the case study's own setup and lets an explicit
    request field, and nothing else, override it. The guard this replaces lived in
    ``test_holdout_boundary.py`` and grepped the notebook's source, so it went on
    passing against text the notebook had stopped containing.
    """
    from types import SimpleNamespace

    root = REPO_ROOT / "case_studies" / "crypto_perps_funding"
    setup = yaml.safe_load((root / "config" / "setup.yaml").read_text())
    assert setup["modeling"]["gbm"]["device"] == "cpu"

    study = SimpleNamespace(root=root)
    assert gbm._gbm_execution_settings(study, {})[0] == "cpu"
    assert gbm._gbm_execution_settings(study, {"device": "gpu"})[0] == "cuda"
