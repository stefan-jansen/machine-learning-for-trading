from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from case_studies.utils.deep_model_state import (
    checkpoint_sidecar,
    declared_epoch_checkpoints,
    deep_checkpoint_path,
    load_deep_checkpoint,
    restore_deep_model,
    validate_deep_checkpoint_population,
    write_deep_checkpoint,
)


def _factory(_architecture: str, kwargs) -> nn.Module:
    return nn.Linear(int(kwargs["n_features"]), 1)


def test_checkpoint_reconstructs_identical_predictions_and_preprocessing(tmp_path) -> None:
    torch.manual_seed(7)
    model = nn.Linear(2, 1)
    raw = np.array([[2.0, 7.0], [4.0, 11.0]], dtype=np.float32)
    preprocessing = {
        "feature_names": ["value", "quality"],
        "mean": np.array([1.0, 3.0], dtype=np.float32),
        "scale": np.array([2.0, 4.0], dtype=np.float32),
    }
    transformed = (raw - preprocessing["mean"]) / preprocessing["scale"]
    expected = model(torch.from_numpy(transformed)).detach().numpy()
    path = tmp_path / "fold_00" / "epoch_005.pt"

    write_deep_checkpoint(
        path,
        model=model,
        architecture="linear-probe",
        model_kwargs={"n_features": 2},
        preprocessing=preprocessing,
        metadata={"config_name": "probe", "fold": 0, "checkpoint_value": 5},
    )
    restored, restored_preprocessing, metadata = restore_deep_model(path, _factory)
    restored_input = (raw - restored_preprocessing["mean"]) / restored_preprocessing["scale"]
    actual = restored(torch.from_numpy(restored_input)).detach().numpy()

    np.testing.assert_array_equal(actual, expected)
    assert restored_preprocessing["feature_names"] == ["value", "quality"]
    assert metadata == {"config_name": "probe", "fold": 0, "checkpoint_value": 5}


def test_declared_checkpoint_schedule_includes_the_final_epoch() -> None:
    assert declared_epoch_checkpoints(10, 5) == (5, 10)
    assert declared_epoch_checkpoints(11, 5) == (5, 10, 11)


def test_checkpoint_is_immutable_and_digest_verified(tmp_path) -> None:
    model = nn.Linear(1, 1)
    path = tmp_path / "epoch_001.pt"
    request = {
        "model": model,
        "architecture": "linear-probe",
        "model_kwargs": {"n_features": 1},
        "preprocessing": {"mean": np.array([0.0]), "scale": np.array([1.0])},
        "metadata": {"checkpoint_value": 1},
    }
    write_deep_checkpoint(path, **request)
    write_deep_checkpoint(path, **request)

    changed = nn.Linear(1, 1)
    with torch.no_grad():
        changed.weight.add_(1.0)
    with pytest.raises(FileExistsError, match="immutable checkpoint conflict"):
        write_deep_checkpoint(path, **{**request, "model": changed})

    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        load_deep_checkpoint(path)

    record = json.loads(checkpoint_sidecar(path).read_text())
    assert record["schema_version"] == 1


def test_checkpoint_population_requires_exact_folds_epochs_and_metadata(tmp_path) -> None:
    model = nn.Linear(1, 1)
    root = tmp_path / "checkpoints"
    for fold in (0, 1):
        for epoch in (5, 10):
            write_deep_checkpoint(
                deep_checkpoint_path(root, "probe", fold, epoch),
                model=model,
                architecture="linear-probe",
                model_kwargs={"n_features": 1},
                preprocessing={"mean": np.array([0.0]), "scale": np.array([1.0])},
                metadata={
                    "config_name": "probe",
                    "fold": fold,
                    "checkpoint_kind": "epoch",
                    "checkpoint_value": epoch,
                },
            )

    paths = validate_deep_checkpoint_population(
        root,
        config_name="probe",
        fold_ids=(0, 1),
        checkpoints=(5, 10),
        architecture="linear-probe",
    )
    assert len(paths) == 4

    deep_checkpoint_path(root, "probe", 1, 10).unlink()
    with pytest.raises(ValueError, match="population is incomplete"):
        validate_deep_checkpoint_population(
            root,
            config_name="probe",
            fold_ids=(0, 1),
            checkpoints=(5, 10),
        )
