from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import torch

from case_studies.utils.deep_learning import create_model, run_dl_cv
from case_studies.utils.deep_model_state import restore_deep_model
from case_studies.utils.sequence_dataset import (
    materialize_sequences,
    prepare_fold_sequence_stores,
)


def test_sequence_runner_checkpoint_reconstructs_registered_values(tmp_path) -> None:
    timestamps = pd.date_range("2020-01-01", periods=30, freq="B")
    dataset = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": f"S{symbol}",
                "feature": float(symbol) + day / 100.0,
                "target": float(symbol) / 10.0 + np.sin(day / 5.0),
            }
            for symbol in range(10)
            for day, timestamp in enumerate(timestamps)
        ]
    )
    split = {
        "fold": 0,
        "train_start": timestamps[0],
        "train_end": timestamps[19],
        "val_start": timestamps[20],
        "val_end": timestamps[-1],
    }
    config = {
        "family": "deep_learning",
        "config_name": "nlinear",
        "params": {"architecture": "nlinear", "lookback": 2, "dropout": 0.0},
        "n_epochs": 1,
        "batch_size": 64,
        "checkpoint_interval": 1,
    }
    checkpoint_root = tmp_path / "checkpoints"

    result = run_dl_cv(
        dataset,
        [split],
        configs=[config],
        n_features=1,
        feature_names=["feature"],
        label_col="target",
        date_col="timestamp",
        entity_col="symbol",
        device="cpu",
        save_dir=tmp_path / "run",
        checkpoint_root=checkpoint_root,
    )

    checkpoint = checkpoint_root / "nlinear" / "fold_00" / "epoch_0001.pt"
    restored, preprocessing, metadata = restore_deep_model(checkpoint, create_model)
    train_mask = (dataset["timestamp"] >= split["train_start"]) & (
        dataset["timestamp"] <= split["train_end"]
    )
    val_mask = (dataset["timestamp"] >= split["val_start"]) & (
        dataset["timestamp"] <= split["val_end"]
    )
    _, val_store, _ = prepare_fold_sequence_stores(
        dataset,
        train_mask=train_mask,
        val_mask=val_mask,
        feature_names=["feature"],
        label_col="target",
        date_col="timestamp",
        entity_col="symbol",
        lookback=2,
        val_start=split["val_start"],
    )
    X_val, _y_val, val_dates, val_entities = materialize_sequences(val_store)
    with torch.no_grad():
        scores = restored(torch.from_numpy(X_val)).numpy()
    reconstructed = pl.DataFrame(
        {"timestamp": val_dates, "symbol": val_entities, "restored": scores}
    ).sort("timestamp", "symbol")
    produced = (
        result["all_predictions"]
        .filter((pl.col("config") == "nlinear") & (pl.col("epoch") == 1) & (pl.col("fold_id") == 0))
        .sort("timestamp", "symbol")
    )

    np.testing.assert_allclose(
        reconstructed["restored"].to_numpy(),
        produced["y_score"].to_numpy(),
        rtol=1e-7,
        atol=1e-8,
    )
    np.testing.assert_array_equal(preprocessing["mean"], val_store.feature_mean)
    np.testing.assert_array_equal(preprocessing["scale"], val_store.feature_scale)
    assert metadata["checkpoint_value"] == 1
