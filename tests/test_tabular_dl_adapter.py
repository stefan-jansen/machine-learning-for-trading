from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace

import polars as pl
import pytest

from case_studies.research import LabelDefinition, Study
from tests.test_research_workspace import _seed_release


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def test_tabm_resolver_builds_complete_resolved_request(tmp_path, monkeypatch) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    frame = pl.DataFrame(
        {
            "symbol": [f"S{index}" for index in range(6)] * 4,
            "timestamp": [
                date
                for date in ("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04")
                for _ in range(6)
            ],
            "feature": [float(index) for index in range(24)],
            "fwd_ret_1d": [float(index % 6) / 100 for index in range(24)],
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    label = study.labels.publish(
        LabelDefinition("fwd_ret_1d", "regression", "1D"),
        frame.select("symbol", "timestamp", "fwd_ret_1d"),
    )
    splits = [
        {
            "fold": 0,
            "train_start": "2024-01-01",
            "train_end": "2024-01-02",
            "val_start": "2024-01-03",
            "val_end": "2024-01-04",
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
        eval_label_col=None,
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
                "checkpoint_interval": 5,
                "n_epochs": 10,
                "params": {"dropout": 0.0, "hidden_dim": 4, "n_members": 2},
                "config_name": "tabm_probe",
                "family": "tabular_dl",
                "library": "tabm",
            }
        ],
    )

    resolved = study.model(
        family="tabular_dl",
        label=label.name,
        config_name="tabm_probe",
        overrides={"device": "cpu", "n_epochs": 11},
    ).resolve()
    spec = resolved.spec
    context = resolved._context

    assert spec["identity_version"] == 3
    assert spec["resolved_spec_schema"] == "ml4t.resolved-spec/v1"
    assert set(spec) == {
        "identity_version",
        "resolved_spec_schema",
        "execution_tier",
        "family",
        "label",
        "seed",
        "config_name",
        "computation",
        "provenance",
    }
    computation = spec["computation"]
    assert set(computation) == {
        "label_artifact",
        "feature_artifacts",
        "feature_names",
        "task",
        "cv",
        "model",
        "preprocessing",
        "checkpoint_schedule",
        "expected_prediction_keys",
        "input_data_spec",
        "sampling",
        "numerics",
        "source_identity",
        "runtime_identity",
    }
    assert computation["label_artifact"]["digest"] == label.digest
    assert computation["model"]["params"]["n_epochs"] == 11
    assert [row["value"] for row in computation["checkpoint_schedule"]] == [5, 10, 11]
    assert computation["expected_prediction_keys"] == {
        "digest": computation["expected_prediction_keys"]["digest"],
        "n_rows": 12,
        "n_folds": 1,
    }
    assert context.expected_keys.height == 12
    assert context.config["n_epochs"] == 11


def test_tabm_classification_resolves_targets_imbalance_and_preview_checkpoints(
    tmp_path, monkeypatch
) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    symbols = [f"S{index}" for index in range(6)]
    timestamps = [
        date for date in ("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04") for _ in symbols
    ]
    classes = [0, 0, 0, 0, 0, 1] * 4
    frame = pl.DataFrame(
        {
            "symbol": symbols * 4,
            "timestamp": timestamps,
            "feature": [float(index) for index in range(24)],
            "direction": classes,
            "fwd_ret_1d": [float(index % 6 - 3) / 100 for index in range(24)],
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    label = study.labels.publish(
        LabelDefinition("direction", "classification", "1D", "fwd_ret_1d"),
        frame.select("symbol", "timestamp", "direction", "fwd_ret_1d"),
    )
    splits = [
        {
            "fold": 0,
            "train_start": "2024-01-01",
            "train_end": "2024-01-02",
            "val_start": "2024-01-03",
            "val_end": "2024-01-04",
        }
    ]
    mds = SimpleNamespace(
        dataset=frame,
        feature_names=["feature"],
        label_col="direction",
        date_col="timestamp",
        entity_cols=["symbol"],
        splits=splits,
        task_type="classification",
        class_values=[0, 1],
        eval_label_col="fwd_ret_1d",
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
                "checkpoint_interval": 5,
                "n_epochs": 10,
                "params": {"dropout": 0.0, "hidden_dim": 4, "n_members": 2},
                "config_name": "tabm_probe",
                "family": "tabular_dl",
                "library": "tabm",
            }
        ],
    )

    resolved = study.model(
        family="tabular_dl",
        label=label.name,
        config_name="tabm_probe",
        overrides={"device": "cpu", "class_weight": "balanced"},
        execution_tier="preview",
        preview_reductions={"folds": [0], "n_epochs": 2, "checkpoint_interval": 1},
    ).resolve()
    computation = resolved.spec["computation"]

    assert computation["task"]["type"] == "classification"
    assert computation["task"]["continuous_eval_label"] == "fwd_ret_1d"
    assert computation["task"]["imbalance"] == {
        "method": "balanced",
        "effective_class_weights_by_fold": {"0": [0.6, 3.0]},
    }
    assert computation["task"]["metrics"] == [
        "ic",
        "auc_roc",
        "log_loss",
        "accuracy",
        "balanced_accuracy",
    ]
    assert computation["preview_reductions"] == {
        "checkpoint_interval": 1,
        "folds": [0],
        "n_epochs": 2,
    }
    assert computation["checkpoint_schedule"] == [
        {"kind": "epoch", "value": 1},
        {"kind": "epoch", "value": 2},
    ]
    assert resolved._context.class_weights_by_fold == {0: (0.6, 3.0)}

    mds.dataset = frame.with_columns(pl.Series("direction", [0, 1, 2, 0, 1, 2] * 4))
    mds.class_values = [0, 1, 2]
    multiclass = study.model(
        family="tabular_dl",
        label=label.name,
        config_name="tabm_probe",
        overrides={"device": "cpu", "class_weight": "balanced"},
        execution_tier="preview",
        preview_reductions={"folds": [0], "n_epochs": 2, "checkpoint_interval": 1},
    ).resolve()

    assert multiclass.spec["computation"]["task"]["metrics"] == [
        "ic",
        "accuracy",
        "balanced_accuracy",
    ]


def _entity_keyed_dataset(entity: str) -> tuple[pl.DataFrame, SimpleNamespace, list[dict]]:
    """Build a four-session panel whose entity column carries the given canonical name."""
    frame = pl.DataFrame(
        {
            entity: [f"S{index}" for index in range(6)] * 4,
            "timestamp": [
                date
                for date in ("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04")
                for _ in range(6)
            ],
            "feature": [float(index) for index in range(24)],
            "fwd_ret_1d": [float(index % 6) / 100 for index in range(24)],
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    splits = [
        {
            "fold": 0,
            "train_start": "2024-01-01",
            "train_end": "2024-01-02",
            "val_start": "2024-01-03",
            "val_end": "2024-01-04",
        }
    ]
    mds = SimpleNamespace(
        dataset=frame,
        feature_names=["feature"],
        label_col="fwd_ret_1d",
        date_col="timestamp",
        entity_cols=[entity],
        splits=splits,
        task_type="regression",
        class_values=[],
        eval_label_col=None,
        temporal_by_fold=None,
        temporal_keys=[],
        temporal_feature_names=[],
        input_lineage={
            "artifacts": {"financial": {"sha256": "features-v1", "size": 1}},
            "fingerprint": "fixture-v1",
        },
    )
    return frame, mds, splits


def _tabm_config() -> list[dict]:
    return [
        {
            "batch_size": 64,
            "checkpoint_interval": 5,
            "n_epochs": 10,
            "params": {"dropout": 0.0, "hidden_dim": 4, "n_members": 2},
            "config_name": "tabm_probe",
            "family": "tabular_dl",
            "library": "tabm",
        }
    ]


@pytest.mark.parametrize("entity", ["symbol", "product"])
def test_tabm_resolver_accepts_either_canonical_entity_key(tmp_path, monkeypatch, entity) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    frame, mds, _ = _entity_keyed_dataset(entity)
    label = study.labels.publish(
        LabelDefinition("fwd_ret_1d", "regression", "1D"),
        frame.rename({entity: "symbol"}).select("symbol", "timestamp", "fwd_ret_1d"),
    )
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *a, **k: mds)
    monkeypatch.setattr("utils.modeling.load_configs", lambda *a, **k: _tabm_config())

    resolved = study.model(
        family="tabular_dl",
        label=label.name,
        config_name="tabm_probe",
        overrides={"device": "cpu"},
    ).resolve()

    assert resolved._context.entity_col == entity
    assert resolved._context.expected_keys.columns == ["symbol", "timestamp", "fold"]
    assert resolved._context.expected_keys.height == 12


def test_tabm_resolver_rejects_an_unsupported_entity_key(tmp_path, monkeypatch) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    frame, mds, _ = _entity_keyed_dataset("ticker")
    label = study.labels.publish(
        LabelDefinition("fwd_ret_1d", "regression", "1D"),
        frame.rename({"ticker": "symbol"}).select("symbol", "timestamp", "fwd_ret_1d"),
    )
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *a, **k: mds)
    monkeypatch.setattr("utils.modeling.load_configs", lambda *a, **k: _tabm_config())

    with pytest.raises(ValueError, match="does not support entity key 'ticker'"):
        study.model(
            family="tabular_dl",
            label=label.name,
            config_name="tabm_probe",
            overrides={"device": "cpu"},
        ).resolve()


@pytest.mark.parametrize("entity", ["symbol", "product"])
def test_tabm_publishes_predictions_under_the_expected_key_names(monkeypatch, entity) -> None:
    """The runner emits the reader-facing entity key; the registry contract expects `symbol`."""
    from case_studies.utils import tabular_dl

    expected_keys = pl.DataFrame(
        {
            "symbol": ["ES", "NQ"],
            "timestamp": [datetime(2024, 1, 3), datetime(2024, 1, 3)],
            "fold": [0, 0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
    all_predictions = pl.DataFrame(
        {
            "config": ["tabm_probe"] * 2,
            "epoch": [2, 2],
            entity: ["ES", "NQ"],
            "timestamp": [datetime(2024, 1, 3), datetime(2024, 1, 3)],
            "fold_id": [0, 0],
            "y_true": [0.01, -0.01],
            "y_score": [0.02, -0.02],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    context = SimpleNamespace(
        config={"config_name": "tabm_probe"},
        date_col="timestamp",
        entity_col=entity,
        expected_keys=expected_keys,
        task_type="regression",
        class_values=(),
        eval_label_col=None,
        label_col="fwd_ret_1d",
        prediction_split="validation",
        published_checkpoints=None,
    )
    published = []

    def capture(_training, **kwargs):
        published.append(kwargs["predictions"])
        return kwargs["predictions"]

    study = SimpleNamespace(results=SimpleNamespace(publish_predictions=capture))
    spec = {"computation": {"checkpoint_schedule": [{"kind": "epoch", "value": 2}]}}

    tabular_dl._publish_tabm_predictions(
        study, spec, context, object(), {"all_predictions": all_predictions}
    )

    assert len(published) == 1
    frame = published[0]
    assert "symbol" in frame.columns and entity not in set(frame.columns) - {"symbol"}
    assert frame.schema["symbol"] == expected_keys.schema["symbol"]
    assert frame.schema["timestamp"] == expected_keys.schema["timestamp"]
    assert frame.schema["fold"] == expected_keys.schema["fold"]
    assert (
        frame.select("symbol", "timestamp", "fold")
        .sort("symbol")
        .equals(expected_keys.sort("symbol"))
    )
