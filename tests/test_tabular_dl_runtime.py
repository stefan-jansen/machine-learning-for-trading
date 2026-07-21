"""Execution-contract tests for the shared TabM runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from case_studies.utils import registry, tabular_dl


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
        predictions = np.zeros(len(kwargs["y_val"]), dtype=np.float32)
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
            "timestamp": [pd.Timestamp("2020-04-01")] * 5,
            "symbol": list(range(5)),
            "y_true": np.arange(5, dtype=float),
            "y_score": np.arange(5, dtype=float),
            "fold_id": [0] * 5,
            "config": ["tabm_probe"] * 5,
            "epoch": [25] * 5,
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

    assert result["best_config_name"] == "tabm_probe"
    assert result["predictions"].height == 5
    assert result["grid_results"][0]["cached"] is True
