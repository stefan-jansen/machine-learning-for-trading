from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from darts.models import TSMixerModel

from case_studies.utils.darts_forecasting import (
    BASE_TARGET_COL,
    darts_checkpoint_path,
    darts_validation_keys,
    load_darts_checkpoint,
    run_darts_cv,
    validate_darts_checkpoint_population,
    write_darts_checkpoint,
)
from case_studies.utils.registry import evaluate_prediction_coverage


def _fit_tiny_tsmixer() -> tuple[TSMixerModel, TimeSeries]:
    series = TimeSeries.from_values(np.arange(24, dtype=np.float32))
    model = TSMixerModel(
        input_chunk_length=4,
        output_chunk_length=1,
        hidden_size=4,
        ff_size=4,
        num_blocks=1,
        n_epochs=1,
        batch_size=8,
        random_state=7,
        save_checkpoints=False,
        force_reset=True,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "logger": False,
        },
    )
    model.fit(series, verbose=False)
    return model, series


def test_darts_checkpoint_reconstructs_identical_predictions(tmp_path) -> None:
    model, series = _fit_tiny_tsmixer()
    expected = model.predict(2, series=series).values()
    root = tmp_path / "models"
    path = darts_checkpoint_path(root, "tsmixer", 0, 1)

    write_darts_checkpoint(
        path,
        model=model,
        architecture="tsmixer",
        metadata={
            "config_name": "tsmixer",
            "fold": 0,
            "checkpoint_kind": "epoch",
            "checkpoint_value": 1,
        },
    )
    validate_darts_checkpoint_population(
        root,
        config_name="tsmixer",
        fold_ids=(0,),
        checkpoints=(1,),
        architecture="tsmixer",
    )
    restored, metadata = load_darts_checkpoint(path)
    actual = restored.predict(2, series=series).values()

    np.testing.assert_array_equal(actual, expected)
    assert metadata["checkpoint_value"] == 1


def test_darts_checkpoint_population_rejects_missing_weights(tmp_path) -> None:
    model, _series = _fit_tiny_tsmixer()
    root = tmp_path / "models"
    path = darts_checkpoint_path(root, "tsmixer", 0, 1)
    write_darts_checkpoint(
        path,
        model=model,
        architecture="tsmixer",
        metadata={
            "config_name": "tsmixer",
            "fold": 0,
            "checkpoint_kind": "epoch",
            "checkpoint_value": 1,
        },
    )
    path.with_suffix(".pt.ckpt").unlink()

    with pytest.raises(ValueError, match="population is incomplete"):
        validate_darts_checkpoint_population(
            root,
            config_name="tsmixer",
            fold_ids=(0,),
            checkpoints=(1,),
            architecture="tsmixer",
        )


def test_darts_runner_persists_state_with_exact_prediction_keys(tmp_path, monkeypatch) -> None:
    dates = pd.date_range("2024-01-02", periods=16, freq="B")
    dataset = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": f"S{symbol}",
                "feature": symbol + day / 10,
                "fwd_ret_1d": np.sin(day / 3) + symbol / 100,
            }
            for symbol in range(6)
            for day, timestamp in enumerate(dates)
        ]
    )

    def attach_base_target(frame, _case_study, _date_col):
        return frame.assign(**{BASE_TARGET_COL: np.log1p(frame["fwd_ret_1d"] / 10)})

    monkeypatch.setattr(
        "case_studies.utils.darts_forecasting._attach_base_target", attach_base_target
    )
    config = {
        "family": "deep_learning",
        "library": "darts",
        "config_name": "tsmixer_probe",
        "params": {
            "architecture": "tsmixer",
            "lookback": 4,
            "hidden_dim": 4,
            "n_blocks": 1,
            "dropout": 0.0,
        },
        "n_epochs": 1,
        "batch_size": 32,
        "checkpoint_interval": 1,
    }
    split = {
        "fold": 0,
        "train_start": dates[0],
        "train_end": dates[9],
        "val_start": dates[10],
        "val_end": dates[-1],
    }
    model_root = tmp_path / "models"

    result = run_darts_cv(
        dataset,
        [split],
        configs=[config],
        feature_names=["feature"],
        label_col="fwd_ret_1d",
        date_col="timestamp",
        entity_col="symbol",
        device="cpu",
        save_dir=tmp_path / "run",
        max_train_sequences=0,
        register=False,
        case_study="etfs",
        notebook=None,
        checkpoint_root=model_root,
    )
    expected = darts_validation_keys(
        dataset,
        [split],
        config=config,
        feature_names=["feature"],
        label_col="fwd_ret_1d",
        date_col="timestamp",
        entity_col="symbol",
        case_study="etfs",
    )
    predictions = result["all_predictions"].rename({"fold_id": "fold"})

    assert evaluate_prediction_coverage(expected, predictions).complete
    assert (
        len(
            validate_darts_checkpoint_population(
                model_root,
                config_name="tsmixer_probe",
                fold_ids=(0,),
                checkpoints=(1,),
                architecture="tsmixer",
            )
        )
        == 1
    )
