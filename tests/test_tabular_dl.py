import numpy as np
import pandas as pd
import torch

from case_studies.utils.deep_model_state import restore_deep_model
from case_studies.utils.tabular_dl import (
    TabMModel,
    _predict_in_chunks,
    run_tabm_cv,
)


def test_run_tabm_cv_returns_fold_metrics_after_training(tmp_path):
    timestamps = pd.date_range("2020-01-01", periods=15, freq="D")
    dataset = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, 10),
            "symbol": np.tile([f"S{i}" for i in range(10)], 15),
            "feature": np.linspace(-1.0, 1.0, 150),
            "target": np.sin(np.arange(150) / 10),
        }
    )
    splits = [
        {
            "fold": 0,
            "train_start": timestamps[0],
            "train_end": timestamps[9],
            "val_start": timestamps[10],
            "val_end": timestamps[-1],
        }
    ]
    configs = [
        {
            "config_name": "tabm_test",
            "params": {"hidden_dim": 4, "n_members": 2, "dropout": 0.0},
            "n_epochs": 1,
            "batch_size": 32,
            "checkpoint_interval": 1,
        }
    ]

    checkpoint_root = tmp_path / "checkpoints"
    result = run_tabm_cv(
        dataset,
        splits,
        configs=configs,
        n_features=1,
        feature_names=["feature"],
        label_col="target",
        date_col="timestamp",
        device="cpu",
        save_dir=tmp_path,
        checkpoint_root=checkpoint_root,
    )

    assert result["best_config_name"] == "tabm_test"
    assert result["fold_metrics"].height == 1
    assert result["fold_metrics"]["n_entities"].to_list() == [10]

    checkpoint = checkpoint_root / "tabm_test" / "fold_00" / "epoch_0001.pt"
    restored, preprocessing, metadata = restore_deep_model(
        checkpoint,
        lambda architecture, kwargs: TabMModel(**kwargs) if architecture == "tabm" else None,
    )
    validation = dataset[dataset["timestamp"] >= timestamps[10]].sort_values(
        ["timestamp", "symbol"]
    )
    raw = validation[["feature"]].to_numpy(dtype=np.float32)
    imputed = np.where(np.isnan(raw), preprocessing["imputer_statistics"], raw)
    transformed = (imputed - preprocessing["scaler_mean"]) / preprocessing["scaler_scale"]
    actual = _predict_in_chunks(restored, transformed, torch.device("cpu"))
    expected = (
        result["all_predictions"]
        .filter(
            (result["all_predictions"]["config"] == "tabm_test")
            & (result["all_predictions"]["epoch"] == 1)
        )
        .sort("timestamp", "symbol")["y_score"]
        .to_numpy()
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
    assert metadata["checkpoint_value"] == 1
