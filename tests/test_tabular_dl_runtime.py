"""Execution-contract tests for the shared TabM runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from case_studies.utils import registry, tabular_dl


def _cache_frame(*, include_eval: bool = True) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "timestamp": [pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01")],
            "symbol": [1, 2],
            "y_true": [0.0, 1.0],
            "y_score": [0.1, 0.2],
            "fold_id": [0, 1],
        }
    )
    if include_eval:
        frame = frame.with_columns(pl.Series("eval_actual", [-0.1, 0.3]))
    return frame


def _install_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    frames: dict[int, pl.DataFrame],
) -> None:
    rows = []
    for epoch, frame in frames.items():
        prediction_hash = f"prediction-{epoch}"
        target = tmp_path / prediction_hash
        target.mkdir()
        frame.write_parquet(target / "predictions.parquet")
        rows.append(
            {
                "prediction_hash": prediction_hash,
                "checkpoint_value": epoch,
                "checkpoint_kind": "epoch",
            }
        )
    monkeypatch.setattr(
        registry,
        "load_prediction_sets",
        lambda *_args, **_kwargs: pl.DataFrame(rows),
    )
    monkeypatch.setattr(registry, "prediction_dir", lambda _case_study, value: tmp_path / value)


def _classification_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=5, freq="MS")
    rows = []
    for timestamp in timestamps:
        for symbol in range(50):
            feature = float(symbol - 25)
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "feature": feature,
                    "label": float(feature > 0),
                    "return": feature / 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_classification_cv_keeps_continuous_evaluation_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, np.ndarray] = {}

    def capture_fold(**kwargs):
        captured["fit"] = kwargs["y_val"].copy()
        captured["eval"] = kwargs["y_eval_val"].copy()
        predictions = np.arange(len(kwargs["y_val"]), dtype=np.float32)
        return {1: 0.25}, {1: predictions}, {1: 0.5}

    monkeypatch.setattr(tabular_dl, "_train_tabm_fold", capture_fold)
    splits = [
        {
            "fold": 0,
            "train_start": pd.Timestamp("2020-01-01"),
            "train_end": pd.Timestamp("2020-03-01"),
            "val_start": pd.Timestamp("2020-04-01"),
            "val_end": pd.Timestamp("2020-05-01"),
        }
    ]
    configs = [
        {
            "config_name": "tabm_probe",
            "params": {"hidden_dim": 4, "n_members": 2, "dropout": 0.0},
            "n_epochs": 1,
            "batch_size": 32,
            "checkpoint_interval": 1,
        }
    ]

    tabular_dl.run_tabm_cv(
        _classification_frame(),
        splits,
        configs=configs,
        n_features=1,
        feature_names=["feature"],
        label_col="label",
        eval_label_col="return",
        task_type="classification",
        class_values=[0, 1],
        class_weights_by_fold={0: (1.0, 1.0)},
        date_col="timestamp",
        entity_col="symbol",
        device="cpu",
        save_dir=tmp_path,
    )

    assert set(np.unique(captured["fit"])) == {0.0, 1.0}
    expected = np.tile(np.arange(-25, 25, dtype=np.float32) / 100.0, 2)
    np.testing.assert_array_equal(captured["eval"], expected)


def test_classification_requires_continuous_evaluation_target(tmp_path) -> None:
    with pytest.raises(ValueError, match="eval_label_col"):
        tabular_dl.run_tabm_cv(
            _classification_frame(),
            [],
            configs=[],
            n_features=1,
            feature_names=["feature"],
            label_col="label",
            task_type="classification",
            class_values=[0, 1],
            date_col="timestamp",
            device="cpu",
            save_dir=tmp_path,
        )


def test_fold_persistence_keeps_fit_and_evaluation_targets(tmp_path) -> None:
    tabular_dl.flush_fold_predictions(
        tmp_path,
        "tabm_probe",
        0,
        {25: np.array([0.1, 0.2])},
        np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[D]"),
        np.array([1, 2]),
        np.array([0.0, 1.0]),
        "timestamp",
        "symbol",
        eval_actual=np.array([-0.1, 0.3]),
        eval_col="eval_actual",
    )

    saved = pl.read_parquet(tmp_path / "tabm_probe_fold0.parquet")
    assert saved["y_true"].to_list() == [0.0, 1.0]
    assert saved["eval_actual"].to_list() == [-0.1, 0.3]


def test_cuda_request_fails_instead_of_silently_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tabular_dl.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        tabular_dl.resolve_torch_device("cuda")


def test_runtime_spec_records_strict_determinism() -> None:
    assert tabular_dl.tabm_runtime_spec("cpu", seed=17) == {
        "device": "cpu",
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "num_threads": 8,
        "seed": 17,
    }


def test_tabm_training_identity_covers_every_effective_input() -> None:
    config = {
        "family": "tabular_dl",
        "config_name": "tabm_s",
        "n_epochs": 100,
        "batch_size": 4096,
        "checkpoint_interval": 25,
    }
    runtime = tabular_dl.tabm_runtime_spec("cpu", seed=42)
    baseline = tabular_dl._build_tabm_training_spec(
        config,
        label_col="fwd_class_1m",
        n_folds=10,
        feature_names=["value", "size"],
        eval_label_col="fwd_ret_1m",
        task_type="classification",
        class_values=[0, 1],
        runtime_spec=runtime,
        seed=42,
    )
    variants = [
        {**config, "batch_size": 2048},
        {**config, "n_epochs": 75},
        {**config, "checkpoint_interval": 15},
    ]
    specs = [
        tabular_dl._build_tabm_training_spec(
            variant,
            label_col="fwd_class_1m",
            n_folds=10,
            feature_names=["value", "size"],
            eval_label_col="fwd_ret_1m",
            task_type="classification",
            class_values=[0, 1],
            runtime_spec=runtime,
            seed=42,
        )
        for variant in variants
    ]
    specs.extend(
        [
            tabular_dl._build_tabm_training_spec(
                config,
                label_col="fwd_class_1m",
                n_folds=10,
                feature_names=["value", "quality"],
                eval_label_col="fwd_ret_1m",
                task_type="classification",
                class_values=[0, 1],
                runtime_spec=runtime,
                seed=42,
            ),
            tabular_dl._build_tabm_training_spec(
                config,
                label_col="fwd_class_1m",
                n_folds=10,
                feature_names=["value", "size"],
                eval_label_col="fwd_ret_3m",
                task_type="classification",
                class_values=[0, 1],
                runtime_spec=runtime,
                seed=42,
            ),
            tabular_dl._build_tabm_training_spec(
                config,
                label_col="fwd_class_1m",
                n_folds=10,
                feature_names=["value", "size"],
                eval_label_col="fwd_ret_1m",
                task_type="classification",
                class_values=[0, 1],
                runtime_spec=tabular_dl.tabm_runtime_spec("cpu", seed=17),
                seed=17,
            ),
        ]
    )

    baseline_hash = registry.training_hash_from_spec(baseline)
    assert all(registry.training_hash_from_spec(spec) != baseline_hash for spec in specs)
    assert baseline["params"]["batch_size"] == 4096
    assert baseline["params"]["eval_label_col"] == "fwd_ret_1m"
    assert baseline["params"]["feature_names"] == ["value", "size"]
    assert baseline["params"]["task_type"] == "classification"
    assert baseline["seed"] == 42


def test_cached_classification_fails_closed_without_eval_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    frame = _cache_frame(include_eval=False)
    _install_cache(monkeypatch, tmp_path, {25: frame})

    with pytest.raises(ValueError, match="eval_actual"):
        tabular_dl._load_cached_tabm_config(
            case_study="probe",
            training_spec={"family": "tabular_dl", "label": "label", "seed": 42},
            config_name="tabm_probe",
            prediction_split="validation",
            date_col="timestamp",
            entity_col="symbol",
            eval_col="eval_actual",
            expected_checkpoints=(25,),
            expected_keys=frame.select("timestamp", "symbol", "fold_id"),
        )


def test_cached_replay_requires_the_exact_checkpoint_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    frame = _cache_frame()
    _install_cache(monkeypatch, tmp_path, {25: frame})

    with pytest.raises(ValueError, match="checkpoints"):
        tabular_dl._load_cached_tabm_config(
            case_study="probe",
            training_spec={"family": "tabular_dl", "label": "label", "seed": 42},
            config_name="tabm_probe",
            prediction_split="validation",
            date_col="timestamp",
            entity_col="symbol",
            eval_col="eval_actual",
            expected_checkpoints=(25, 50, 75, 100),
            expected_keys=frame.select("timestamp", "symbol", "fold_id"),
        )


@pytest.mark.parametrize("defect", ["null_checkpoint", "wrong_kind"])
def test_cached_replay_rejects_contradictory_checkpoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    defect: str,
) -> None:
    frame = _cache_frame()
    _install_cache(monkeypatch, tmp_path, {25: frame})
    if defect == "null_checkpoint":
        prediction_sets = pl.DataFrame(
            {
                "prediction_hash": ["prediction-25", "ignored-final"],
                "checkpoint_value": [25, None],
                "checkpoint_kind": ["epoch", "final"],
            }
        )
    else:
        prediction_sets = pl.DataFrame(
            {
                "prediction_hash": ["prediction-25"],
                "checkpoint_value": [25],
                "checkpoint_kind": ["final"],
            }
        )
    monkeypatch.setattr(
        registry,
        "load_prediction_sets",
        lambda *_args, **_kwargs: prediction_sets,
    )

    with pytest.raises(ValueError, match="checkpoint"):
        tabular_dl._load_cached_tabm_config(
            case_study="probe",
            training_spec={"family": "tabular_dl", "label": "label", "seed": 42},
            config_name="tabm_probe",
            prediction_split="validation",
            date_col="timestamp",
            entity_col="symbol",
            eval_col="eval_actual",
            expected_checkpoints=(25,),
            expected_keys=frame.select("timestamp", "symbol", "fold_id"),
        )


@pytest.mark.parametrize("defect", ["duplicate", "coverage", "schema"])
def test_cached_replay_rejects_invalid_prediction_panels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    defect: str,
) -> None:
    expected = _cache_frame()
    frame = expected
    if defect == "duplicate":
        frame = pl.concat([expected, expected.head(1)])
    elif defect == "coverage":
        frame = expected.head(1)
    else:
        frame = expected.rename({"timestamp": "date"})
    _install_cache(monkeypatch, tmp_path, {25: frame})

    with pytest.raises(ValueError, match="schema|duplicate|coverage"):
        tabular_dl._load_cached_tabm_config(
            case_study="probe",
            training_spec={"family": "tabular_dl", "label": "label", "seed": 42},
            config_name="tabm_probe",
            prediction_split="validation",
            date_col="timestamp",
            entity_col="symbol",
            eval_col="eval_actual",
            expected_checkpoints=(25,),
            expected_keys=expected.select("timestamp", "symbol", "fold_id"),
        )


@pytest.mark.parametrize("corrupted_metric", ["ic_mean_daily", "ic_std_daily"])
def test_cached_replay_rejects_corrupted_daily_registry_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    corrupted_metric: str,
) -> None:
    timestamps = [pd.Timestamp("2020-04-01")] * 5 + [pd.Timestamp("2020-05-01")] * 5
    frame = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": list(range(5)) * 2,
            "y_true": list(range(5)) * 2,
            "y_score": list(range(5)) + list(reversed(range(5))),
            "fold_id": [0] * 5 + [1] * 5,
            "eval_actual": list(range(5)) * 2,
        }
    )
    _install_cache(monkeypatch, tmp_path, {25: frame})
    metric = tabular_dl.cross_sectional_ic(
        frame,
        frame,
        pred_col="y_score",
        ret_col="eval_actual",
        date_col="timestamp",
        entity_col="symbol",
        method="spearman",
        min_obs=5,
    )
    registry_metrics = {
        "ic_mean_daily": float(metric["ic_mean"]),
        "ic_std_daily": float(metric["ic_std"]),
    }
    registry_metrics[corrupted_metric] += 0.25
    monkeypatch.setattr(
        registry,
        "load_prediction_metrics",
        lambda *_args, **_kwargs: pl.DataFrame([registry_metrics]),
    )

    with pytest.raises(ValueError, match="daily metric mismatch"):
        tabular_dl._load_cached_tabm_config(
            case_study="probe",
            training_spec={"family": "tabular_dl", "label": "label", "seed": 42},
            config_name="tabm_probe",
            prediction_split="validation",
            date_col="timestamp",
            entity_col="symbol",
            eval_col="eval_actual",
            expected_checkpoints=(25,),
            expected_keys=frame.select("timestamp", "symbol", "fold_id"),
        )


def test_cached_replay_selects_full_decision_time_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    dates = (
        [pd.Timestamp("2020-04-01")] * 5
        + [pd.Timestamp("2020-05-01")] * 5
        + [pd.Timestamp("2020-06-01")] * 5
    )
    symbols = list(range(5)) * 3
    actual = list(range(5)) * 3
    full = pl.DataFrame(
        {
            "timestamp": dates,
            "symbol": symbols,
            "y_true": actual,
            "y_score": list(reversed(range(5))) + list(range(5)) * 2,
            "fold_id": [0] * 5 + [1] * 5 + [2] * 5,
        }
    ).with_columns(pl.col("y_score").cast(pl.Float64))
    partial = full.with_columns(
        pl.when(pl.col("timestamp") == pd.Timestamp("2020-06-01"))
        .then(0.0)
        .otherwise(pl.col("y_true"))
        .alias("y_score")
    )
    frames = {10: full, 20: partial}
    _install_cache(monkeypatch, tmp_path, frames)

    metrics = {}
    for epoch, frame in frames.items():
        stats = tabular_dl.cross_sectional_ic(
            frame,
            frame,
            pred_col="y_score",
            ret_col="y_true",
            date_col="timestamp",
            entity_col="symbol",
            method="spearman",
            min_obs=5,
        )
        metrics[f"prediction-{epoch}"] = pl.DataFrame(
            {
                "ic_mean_daily": [stats["ic_mean"]],
                "ic_std_daily": [stats["ic_std"]],
            }
        )
    monkeypatch.setattr(
        registry,
        "load_prediction_metrics",
        lambda _case_study, *, prediction_hash: metrics[prediction_hash],
    )

    result, _, curves = tabular_dl._load_cached_tabm_config(
        case_study="probe",
        training_spec={"family": "tabular_dl", "label": "label", "seed": 42},
        config_name="tabm_probe",
        prediction_split="validation",
        date_col="timestamp",
        entity_col="symbol",
        eval_col=None,
        expected_checkpoints=(10, 20),
        expected_keys=full.select("timestamp", "symbol", "fold_id"),
    )

    assert {row["epoch"]: row["ic_n_days"] for row in curves} == {10: 3, 20: 2}
    assert result["best_epoch"] == 10


def test_training_without_save_dir_keeps_checkpoint_eligibility_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _classification_frame()
    splits = [
        {
            "fold": 0,
            "train_start": pd.Timestamp("2020-01-01"),
            "train_end": pd.Timestamp("2020-03-01"),
            "val_start": pd.Timestamp("2020-04-01"),
            "val_end": pd.Timestamp("2020-05-01"),
        }
    ]

    def fake_fold(**kwargs):
        actual = kwargs["y_eval_val"].astype(np.float32)
        return {1: 0.1, 2: 0.9}, {1: actual, 2: np.full_like(actual, np.inf)}, {}

    monkeypatch.setattr(tabular_dl, "TabMModel", lambda **_kwargs: object())
    monkeypatch.setattr(tabular_dl, "_train_tabm_fold", fake_fold)

    result = tabular_dl.run_tabm_cv(
        frame,
        splits,
        configs=[
            {
                "family": "tabular_dl",
                "config_name": "tabm_probe",
                "params": {},
                "n_epochs": 2,
                "checkpoint_interval": 1,
            }
        ],
        n_features=1,
        feature_names=["feature"],
        label_col="label",
        date_col="timestamp",
        entity_col="symbol",
        device="cpu",
        save_dir=None,
    )

    curves = {row["epoch"]: row for row in result["all_learning_curves"].to_dicts()}
    assert result["best_epoch"] == 1
    assert curves[1]["n_invalid"] == 0
    assert curves[2]["n_invalid"] == 100
    assert result["all_predictions"].height == 200


def test_complete_registry_replays_predictions_instead_of_returning_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class CompleteStatus:
        complete = True
        partial = False

        @staticmethod
        def summary() -> str:
            return "complete"

    cached_predictions = pl.DataFrame(
        {
            "timestamp": [pd.Timestamp("2020-04-01")] * 50,
            "symbol": list(range(50)),
            "y_true": np.arange(50, dtype=float),
            "y_score": np.arange(50, dtype=float),
            "fold_id": [0] * 50,
            "config": ["tabm_probe"] * 50,
            "epoch": [25] * 50,
        }
    )
    cached_result = {
        "config_name": "tabm_probe",
        "best_epoch": 25,
        "best_ic": 1.0,
        "elapsed_s": 0.0,
        "started_at": None,
        "cached": True,
    }
    cached_curves = [{"config": "tabm_probe", "epoch": 25, "ic_mean": 1.0, "ic_std": 0.0}]

    monkeypatch.setattr(
        registry,
        "build_training_spec",
        lambda *_args, **_kwargs: {"family": "tabular_dl", "label": "label", "seed": 42},
    )
    monkeypatch.setattr(registry, "training_hash_from_spec", lambda _spec: "training")
    monkeypatch.setattr(registry, "training_run_status", lambda *_args: CompleteStatus())
    monkeypatch.setattr(
        registry,
        "load_prediction_sets",
        lambda *_args, **_kwargs: pl.DataFrame({"prediction_hash": ["prediction"]}),
    )
    monkeypatch.setattr(
        tabular_dl,
        "_load_cached_tabm_config",
        lambda **_kwargs: (cached_result, cached_predictions, cached_curves),
    )

    result = tabular_dl.run_tabm_cv(
        _classification_frame(),
        [
            {
                "fold": 0,
                "train_start": "2020-01-01",
                "train_end": "2020-03-01",
                "val_start": "2020-04-01",
                "val_end": "2020-04-01",
            }
        ],
        configs=[
            {
                "family": "tabular_dl",
                "config_name": "tabm_probe",
                "params": {"hidden_dim": 4, "n_members": 2, "dropout": 0.0},
                "n_epochs": 25,
                "checkpoint_interval": 25,
            }
        ],
        n_features=1,
        feature_names=["feature"],
        label_col="return",
        date_col="timestamp",
        entity_col="symbol",
        device="cpu",
        save_dir=tmp_path,
        register=True,
        case_study="probe",
    )

    assert result["best_config_name"] == "tabm_probe"
    assert result["predictions"].height == 50

    with pytest.raises(ValueError, match="at least one fold"):
        tabular_dl.run_tabm_cv(
            _classification_frame(),
            [],
            configs=[
                {
                    "family": "tabular_dl",
                    "config_name": "tabm_probe",
                    "params": {"hidden_dim": 4, "n_members": 2, "dropout": 0.0},
                    "n_epochs": 25,
                    "checkpoint_interval": 25,
                }
            ],
            n_features=1,
            feature_names=["feature"],
            label_col="return",
            date_col="timestamp",
            entity_col="symbol",
            device="cpu",
            save_dir=tmp_path,
            register=True,
            case_study="probe",
        )
    assert result["grid_results"][0]["cached"] is True


def test_direct_registered_batch_preserves_completed_sibling_on_later_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class MissingStatus:
        complete = False
        partial = False

    registered: list[str] = []

    def train_candidate(*, model, X_val, **_kwargs):
        hidden_dim = int(model.backbone[0].out_features)
        if hidden_dim == 8:
            raise RuntimeError("injected later candidate failure")
        predictions = np.asarray(X_val[:, 0], dtype=np.float64) + hidden_dim / 100
        return {1: 0.1}, {1: predictions}, {1: 0.01}

    monkeypatch.setattr(tabular_dl, "_train_tabm_fold", train_candidate)
    monkeypatch.setattr(registry, "training_run_status", lambda *_args: MissingStatus())
    monkeypatch.setattr(registry, "load_prediction_sets", lambda *_args, **_kwargs: pl.DataFrame())
    monkeypatch.setattr(registry, "training_hash_from_spec", lambda _spec: "training")
    monkeypatch.setattr(
        tabular_dl,
        "_register_tabm_config",
        lambda **kwargs: registered.append(kwargs["config_name"]) or "training",
    )

    with pytest.raises(RuntimeError, match="injected later candidate failure"):
        tabular_dl.run_tabm_cv(
            _classification_frame(),
            [
                {
                    "fold": 0,
                    "train_start": pd.Timestamp("2020-01-01"),
                    "train_end": pd.Timestamp("2020-03-01"),
                    "val_start": pd.Timestamp("2020-04-01"),
                    "val_end": pd.Timestamp("2020-05-01"),
                }
            ],
            configs=[
                {
                    "family": "tabular_dl",
                    "config_name": "tabm_s",
                    "params": {"hidden_dim": 4, "n_members": 2, "dropout": 0.0},
                    "n_epochs": 1,
                    "batch_size": 32,
                    "checkpoint_interval": 1,
                },
                {
                    "family": "tabular_dl",
                    "config_name": "tabm_m",
                    "params": {"hidden_dim": 8, "n_members": 2, "dropout": 0.0},
                    "n_epochs": 1,
                    "batch_size": 32,
                    "checkpoint_interval": 1,
                },
            ],
            n_features=1,
            feature_names=["feature"],
            label_col="return",
            date_col="timestamp",
            entity_col="symbol",
            device="cpu",
            save_dir=tmp_path,
            register=True,
            case_study="probe",
        )

    assert registered == ["tabm_s"]
    assert (tmp_path / "return" / "_incremental" / "tabm_s_fold0.parquet").exists()


def test_saved_artifact_message_does_not_leak_absolute_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    predictions = pl.DataFrame(
        {
            "timestamp": [pd.Timestamp("2020-04-01")],
            "symbol": [1],
            "y_true": [0.1],
            "y_score": [0.2],
            "fold_id": [0],
            "config": ["tabm_probe"],
            "epoch": [25],
        }
    )
    monkeypatch.setattr(
        tabular_dl,
        "compute_fold_metrics_from_predictions",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )

    tabular_dl._assemble_tabm_results(
        config_results=[
            {
                "config_name": "tabm_probe",
                "best_epoch": 25,
                "best_ic": 0.1,
                "elapsed_s": 0.0,
            }
        ],
        all_predictions=predictions,
        curve_rows=[],
        training_rows=[],
        save_dir=tmp_path / "private" / "fwd_ret_1m",
        date_col="timestamp",
        entity_col="symbol",
        eval_col=None,
    )

    output = capsys.readouterr().out
    assert str(tmp_path) not in output
    assert "Saved TabM artifacts for fwd_ret_1m" in output


def test_tabm_selection_rejects_higher_ic_with_partial_decision_time_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")]
    full = pl.DataFrame(
        {
            "timestamp": np.repeat(dates, 5),
            "symbol": list(range(5)) * 2,
            "y_true": list(range(5)) * 2,
            "y_score": list(range(5)) * 2,
            "fold_id": [0] * 5 + [1] * 5,
            "config": ["full"] * 10,
            "epoch": [25] * 10,
        }
    )
    partial = full.head(5).with_columns(pl.lit("partial").alias("config"))
    monkeypatch.setattr(
        tabular_dl,
        "compute_fold_metrics_from_predictions",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )

    result = tabular_dl._assemble_tabm_results(
        config_results=[
            {
                "config_name": "partial",
                "best_epoch": 25,
                "best_ic": 0.9,
                "ic_n_days": 1,
                "n_invalid": 0,
                "elapsed_s": 0.0,
            },
            {
                "config_name": "full",
                "best_epoch": 25,
                "best_ic": 0.2,
                "ic_n_days": 2,
                "n_invalid": 0,
                "elapsed_s": 0.0,
            },
        ],
        all_predictions=pl.concat([partial, full]),
        curve_rows=[],
        training_rows=[],
        save_dir=None,
        date_col="timestamp",
        entity_col="symbol",
        eval_col=None,
    )

    assert result["best_config_name"] == "full"
    assert result["grid_results"][0]["selectable"] is True
    assert result["grid_results"][1]["selectable"] is False
