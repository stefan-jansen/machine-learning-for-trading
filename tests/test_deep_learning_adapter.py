from __future__ import annotations

import os
from importlib.metadata import version
from types import SimpleNamespace

import polars as pl
import pytest

from case_studies.research import LabelDefinition, Study
from case_studies.utils import deep_learning, tabular_dl
from tests.test_research_workspace import _seed_release


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def test_sequence_resolver_builds_complete_v2_request(tmp_path, monkeypatch) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    dates = [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
    ]
    frame = pl.DataFrame(
        {
            "symbol": [f"S{symbol}" for symbol in range(3) for _ in dates],
            "timestamp": dates * 3,
            "feature": [float(index) for index in range(24)],
            "fwd_ret_1d": [float(index % 3) / 100 for index in range(24)],
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    label = study.labels.publish(
        LabelDefinition("fwd_ret_1d", "regression", "1D"),
        frame.select("symbol", "timestamp", "fwd_ret_1d"),
    )
    splits = [
        {
            "fold": 0,
            "train_start": "2024-01-02",
            "train_end": "2024-01-05",
            "val_start": "2024-01-08",
            "val_end": "2024-01-11",
        }
    ]
    mds = SimpleNamespace(
        dataset=frame,
        feature_names=["feature"],
        label_col="fwd_ret_1d",
        date_col="timestamp",
        entity_cols=["symbol"],
        splits=splits,
        task_type="regression",
        class_values=[],
        temporal_by_fold=None,
        temporal_keys=[],
        temporal_feature_names=[],
        input_lineage={
            "artifacts": {"financial": {"sha256": "features-v1", "size": 1}},
            "fingerprint": "fixture-v1",
        },
    )
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *args, **kwargs: mds)
    monkeypatch.setattr(
        "utils.modeling.load_configs",
        lambda *args, **kwargs: [
            {
                "batch_size": 64,
                "checkpoint_interval": 2,
                "n_epochs": 4,
                "params": {"architecture": "nlinear", "dropout": 0.0, "lookback": 2},
                "config_name": "nlinear_probe",
                "family": "deep_learning",
                "library": "pytorch",
            }
        ],
    )

    resolved = study.model(
        family="deep_learning",
        label=label.name,
        config_name="nlinear_probe",
        overrides={"device": "cpu", "n_epochs": 3},
    ).resolve()
    spec = resolved.spec
    context = resolved._context

    assert spec["identity_version"] == 2
    assert spec["label_artifact"]["digest"] == label.digest
    assert spec["model"]["params"]["n_epochs"] == 3
    assert [row["value"] for row in spec["checkpoint_schedule"]] == [2, 3]
    assert spec["expected_prediction_keys"]["n_rows"] == 12
    assert spec["sampling"] == {"max_symbols": 0, "max_train_sequences": 0}
    assert context.expected_keys.height == 12
    assert context.config["n_epochs"] == 3


def test_darts_request_resolves_installed_runtime_identity(tmp_path, monkeypatch) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    dates = [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
    ]
    frame = pl.DataFrame(
        {
            "symbol": [f"S{symbol}" for symbol in range(3) for _ in dates],
            "timestamp": dates * 3,
            "feature": [float(index) for index in range(18)],
            "fwd_ret_1d": [float(index % 3) / 100 for index in range(18)],
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    label = study.labels.publish(
        LabelDefinition("fwd_ret_1d", "regression", "1D"),
        frame.select("symbol", "timestamp", "fwd_ret_1d"),
    )
    split = {
        "fold": 0,
        "train_start": "2024-01-02",
        "train_end": "2024-01-05",
        "val_start": "2024-01-08",
        "val_end": "2024-01-09",
    }
    mds = SimpleNamespace(
        dataset=frame,
        feature_names=["feature"],
        label_col="fwd_ret_1d",
        date_col="timestamp",
        entity_cols=["symbol"],
        splits=[split],
        task_type="regression",
        class_values=[],
        temporal_by_fold=None,
        temporal_keys=[],
        temporal_feature_names=[],
        input_lineage={
            "artifacts": {"financial": {"sha256": "features-v1", "size": 1}},
            "fingerprint": "fixture-v1",
        },
    )
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *args, **kwargs: mds)
    monkeypatch.setattr(
        "utils.modeling.load_configs",
        lambda *args, **kwargs: [
            {
                "batch_size": 64,
                "checkpoint_interval": 1,
                "n_epochs": 1,
                "params": {"architecture": "tsmixer", "lookback": 2},
                "config_name": "tsmixer_probe",
                "family": "deep_learning",
                "library": "darts",
            }
        ],
    )
    expected = (
        frame.filter(pl.col("timestamp") >= pl.date(2024, 1, 8))
        .select("symbol", "timestamp")
        .with_columns(pl.lit(0, dtype=pl.Int64).alias("fold"))
    )
    monkeypatch.setattr(
        "case_studies.utils.darts_forecasting.darts_validation_keys",
        lambda *args, **kwargs: expected,
    )
    monkeypatch.setattr(
        "case_studies.utils.darts_forecasting.darts_training_identity",
        lambda *args, **kwargs: {
            "input_data_spec": mds.input_lineage,
            "input_chunk_length": 2,
            "output_chunk_length": 1,
            "max_train_sequences": 0,
        },
    )

    resolved = study.model(
        family="deep_learning",
        label=label.name,
        config_name="tsmixer_probe",
        overrides={"device": "cpu"},
    ).resolve()

    assert resolved.spec["runtime_identity"]["darts"] == version("darts")
    assert resolved.spec["model"]["implementation"] == "darts"


@pytest.mark.parametrize(
    "split_resolver", [deep_learning._sequence_splits, tabular_dl._tabm_splits]
)
def test_deep_adapters_reject_custom_cv_with_stale_temporal_geometry(split_resolver) -> None:
    canonical = {
        "fold": 0,
        "train_start": "2020-01-01",
        "train_end": "2020-12-31",
        "val_start": "2021-01-01",
        "val_end": "2021-12-31",
    }
    requested = {**canonical, "val_start": "2020-07-01"}
    resolved_cv = SimpleNamespace(
        normalized_folds=(requested,),
        as_dict=lambda: {"folds": [requested]},
    )
    cv = SimpleNamespace(resolve=lambda *args, **kwargs: resolved_cv)
    mds = SimpleNamespace(
        dataset=pl.DataFrame({"timestamp": ["2020-01-01"]}).with_columns(
            pl.col("timestamp").str.to_date()
        ),
        date_col="timestamp",
        splits=[canonical],
        temporal_by_fold=pl.DataFrame({"fold": [0], "timestamp": ["2021-01-01"]}).with_columns(
            pl.col("timestamp").str.to_date()
        ),
    )

    with pytest.raises(ValueError, match="Custom CV cannot reuse fold-specific temporal features"):
        split_resolver(mds, {"cv": cv, "preview_reductions": {}})
