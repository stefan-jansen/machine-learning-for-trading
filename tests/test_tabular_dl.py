import numpy as np
import pandas as pd

from case_studies.utils.tabular_dl import run_tabm_cv


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
    )

    assert result["best_config_name"] == "tabm_test"
    assert result["fold_metrics"].height == 1
    assert result["fold_metrics"]["n_entities"].to_list() == [10]
