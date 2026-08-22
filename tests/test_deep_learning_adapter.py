from __future__ import annotations

import os
from datetime import datetime
from importlib.metadata import version
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from case_studies.research import CVSpec, LabelDefinition, Study
from case_studies.utils import deep_learning, tabular_dl
from tests.test_research_workspace import _seed_release


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _resolve_nlinear_request(
    tmp_path, monkeypatch, entity: str = "symbol", library: str = "pytorch"
):
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
            entity: [f"S{symbol}" for symbol in range(3) for _ in dates],
            "timestamp": dates * 3,
            "feature": [float(index) for index in range(24)],
            "fwd_ret_1d": [float(index % 3) / 100 for index in range(24)],
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    label = study.labels.publish(
        LabelDefinition("fwd_ret_1d", "regression", "1D"),
        frame.rename({entity: "symbol"}).select("symbol", "timestamp", "fwd_ret_1d"),
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
        entity_cols=[entity],
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
                "params": {
                    "architecture": "tsmixer" if library == "darts" else "nlinear",
                    "dropout": 0.0,
                    "lookback": 2,
                },
                "config_name": "nlinear_probe",
                "family": "deep_learning",
                "library": library,
            }
        ],
    )

    resolved = study.model(
        family="deep_learning",
        label=label.name,
        config_name="nlinear_probe",
        overrides={"device": "cpu", "n_epochs": 3},
    ).resolve()
    return study, label, resolved


def test_sequence_resolver_builds_complete_resolved_request(tmp_path, monkeypatch) -> None:
    _study, label, resolved = _resolve_nlinear_request(tmp_path, monkeypatch)
    spec = resolved.spec
    context = resolved._context

    computation = spec["computation"]
    assert spec["identity_version"] == 3
    assert computation["label_artifact"]["digest"] == label.digest
    assert computation["model"]["params"]["n_epochs"] == 3
    assert [row["value"] for row in computation["checkpoint_schedule"]] == [2, 3]
    assert computation["expected_prediction_keys"]["n_rows"] == 12
    assert computation["sampling"] == {"max_symbols": 0, "max_train_sequences": 0}
    assert context.expected_keys.height == 12
    assert context.config["n_epochs"] == 3


def test_cached_sequence_run_resolves_predictions_at_the_published_identity(
    tmp_path, monkeypatch
) -> None:
    from case_studies.research.results import Result
    from case_studies.utils.registry import prediction_hash_from_parts, training_hash_from_spec

    study, _label, resolved = _resolve_nlinear_request(tmp_path, monkeypatch)
    spec = resolved.spec
    requested: list[str] = []

    @classmethod
    def _record(cls, study_arg, result_hash, **kwargs):
        requested.append(result_hash)
        return SimpleNamespace(hash=result_hash)

    monkeypatch.setattr(Result, "open", _record)

    assert deep_learning._cached_sequence_run(study, spec, resolved._context) is None

    training_hash = training_hash_from_spec(spec)
    expected = [
        prediction_hash_from_parts(
            training_hash,
            row["value"],
            "validation",
            checkpoint_kind="epoch",
            identity_version=spec["identity_version"],
        )
        for row in spec["computation"]["checkpoint_schedule"]
    ]
    assert requested == [training_hash, *expected]


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
            "base_target_data_spec": {"kind": "one_period_return"},
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

    computation = resolved.spec["computation"]
    assert computation["runtime_identity"]["darts"] == version("darts")
    assert computation["model"]["implementation"] == "darts"


def test_weekly_nbeats_request_applies_identity_cadence_before_cv(tmp_path, monkeypatch) -> None:
    release = _seed_release(tmp_path)
    (release / "case_studies" / "etfs").rename(release / "case_studies" / "us_equities_panel")
    study = Study.open(
        "us_equities_panel",
        workspace=tmp_path / "workspace",
        release_root=release,
    )
    dates = pl.date_range(pl.date(2023, 1, 2), pl.date(2024, 3, 29), eager=True).filter(
        pl.date_range(pl.date(2023, 1, 2), pl.date(2024, 3, 29), eager=True).dt.weekday() <= 5
    )
    frame = pl.DataFrame(
        {
            "symbol": [f"S{symbol}" for symbol in range(3) for _ in dates],
            "timestamp": dates.to_list() * 3,
            "feature": [float(index) for index in range(3 * len(dates))],
            "temporal": [float(index % 7) for index in range(3 * len(dates))],
            "fwd_ret_5d": [float(index % 3) / 100 for index in range(3 * len(dates))],
        }
    )
    label = study.labels.publish(
        LabelDefinition("fwd_ret_5d", "regression", "5D"),
        frame.select("symbol", "timestamp", "fwd_ret_5d"),
    )
    cv = CVSpec.walk_forward(
        training_window="80D",
        validation_window="20D",
        retrain_every="20D",
        folds=(0,),
        horizon="5D",
        gap="5D",
        holdout_start="2024-02-01",
        holdout_end="2024-03-29",
        calendar="NYSE",
    )
    canonical_folds = [
        dict(fold) for fold in cv.resolve(frame.select("timestamp").unique()).normalized_folds
    ]
    temporal_by_fold = (
        frame.select("symbol", "timestamp", "temporal")
        .with_columns(pl.lit(0, dtype=pl.Int64).alias("fold"))
        .to_pandas()
    )
    mds = SimpleNamespace(
        dataset=frame,
        feature_names=["feature", "temporal"],
        label_col="fwd_ret_5d",
        date_col="timestamp",
        entity_cols=["symbol"],
        splits=canonical_folds,
        task_type="regression",
        class_values=[],
        temporal_by_fold=temporal_by_fold,
        temporal_keys=["symbol", "timestamp"],
        temporal_feature_names=["temporal"],
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
                "params": {
                    "architecture": "nbeats",
                    "decision_cadence": "weekly_friday",
                    "lookback": 12,
                    "darts_input_chunk_length": 12,
                    "darts_output_chunk_length": 2,
                    "darts_target": "lagged_label",
                },
                "config_name": "nbeats_weekly",
                "family": "deep_learning",
                "library": "darts",
            }
        ],
    )
    resolved = study.model(
        family="deep_learning",
        label=label.name,
        config_name="nbeats_weekly",
        overrides={"device": "cpu"},
        cv=cv,
    ).resolve()

    context = resolved._context
    observed_timestamps = sorted(context.dataset_pd["timestamp"].unique())
    split = context.splits[0]
    assert len(observed_timestamps) < len(dates)
    assert all(timestamp.weekday() == 4 for timestamp in observed_timestamps)
    assert list(context.splits) == canonical_folds
    assert pd.Timestamp(split["val_start"]) - pd.Timestamp(split["train_end"]) >= pd.Timedelta("5D")
    computation = resolved.spec["computation"]
    assert computation["cv"]["request"]["decision_cadence"] is None
    assert computation["cv"]["request"]["gap"] == "5D"
    assert computation["model"]["params"]["decision_cadence"] == "weekly_friday"
    assert computation["preprocessing"]["decision_cadence"] == "weekly_friday"
    eligible_timestamps = [
        timestamp
        for timestamp in observed_timestamps
        if pd.Timestamp(split["val_start"]) <= timestamp <= pd.Timestamp(split["val_end"])
    ]
    expected = pl.DataFrame(
        {
            "symbol": [f"S{symbol}" for symbol in range(3) for _ in eligible_timestamps],
            "timestamp": eligible_timestamps * 3,
            "fold": [0] * (3 * len(eligible_timestamps)),
        }
    ).sort("symbol", "timestamp", "fold")
    assert context.expected_keys.equals(expected)
    assert computation["expected_prediction_keys"]["n_rows"] == expected.height


def test_run_dl_cv_applies_preset_cadence_before_backend(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-01", "2024-01-19")
    dataset = pd.DataFrame(
        {
            "symbol": "S0",
            "timestamp": dates,
            "feature": range(len(dates)),
            "fwd_ret_5d": 0.01,
        }
    )
    config = {
        "batch_size": 8,
        "checkpoint_interval": 1,
        "config_name": "nbeats_weekly",
        "family": "deep_learning",
        "library": "darts",
        "n_epochs": 1,
        "params": {
            "architecture": "nbeats",
            "decision_cadence": "weekly_friday",
            "darts_input_chunk_length": 2,
            "darts_output_chunk_length": 2,
            "darts_target": "lagged_label",
            "lookback": 2,
        },
    }
    observed: dict[str, pd.DataFrame] = {}
    sentinel = {"all_predictions": pl.DataFrame()}

    def capture_backend(dataset_pd, _splits, **_kwargs):
        observed["dataset"] = dataset_pd
        return sentinel

    monkeypatch.setattr(
        "case_studies.utils.darts_forecasting.run_darts_cv",
        capture_backend,
    )

    result = deep_learning.run_dl_cv(
        dataset,
        [
            {
                "fold": 0,
                "train_start": dates[0],
                "train_end": dates[7],
                "val_start": dates[8],
                "val_end": dates[-1],
            }
        ],
        configs=[config],
        n_features=1,
        feature_names=["feature"],
        label_col="fwd_ret_5d",
        date_col="timestamp",
        device="cpu",
        case_study="us_equities_panel",
    )

    assert result is sentinel
    assert observed["dataset"]["timestamp"].dt.weekday.eq(4).all()


def test_sequence_adapter_rejects_unknown_decision_cadence() -> None:
    dataset = pd.DataFrame({"timestamp": pd.bdate_range("2024-01-01", periods=5)})

    with pytest.raises(ValueError, match="unsupported sequence decision cadence"):
        deep_learning._select_sequence_observations(
            dataset,
            date_col="timestamp",
            cadence="weekly_fri",
            calendar="NYSE",
        )


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


@pytest.mark.parametrize("entity", ["symbol", "product"])
def test_sequence_resolver_accepts_either_canonical_entity_key(
    tmp_path, monkeypatch, entity
) -> None:
    _study, _label, resolved = _resolve_nlinear_request(tmp_path, monkeypatch, entity=entity)

    assert resolved._context.entity_col == entity
    assert resolved._context.expected_keys.columns == ["symbol", "timestamp", "fold"]
    assert resolved._context.expected_keys.height > 0


def test_sequence_resolver_rejects_an_unsupported_entity_key(tmp_path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="does not support entity key 'ticker'"):
        _resolve_nlinear_request(tmp_path, monkeypatch, entity="ticker")


def test_darts_presets_refuse_a_non_symbol_entity_key(tmp_path, monkeypatch) -> None:
    """The Darts key builder names its keys after the entity column, unlike the other three."""
    with pytest.raises(ValueError, match="Darts presets require the symbol entity key"):
        _resolve_nlinear_request(tmp_path, monkeypatch, entity="product", library="darts")


@pytest.mark.parametrize("entity", ["symbol", "product"])
def test_sequence_publishes_predictions_under_the_expected_key_names(entity) -> None:
    """The runner emits the reader-facing entity key; the registry contract expects `symbol`."""
    from case_studies.utils import deep_learning

    expected_keys = pl.DataFrame(
        {
            "symbol": ["ES", "NQ"],
            "timestamp": [datetime(2024, 1, 8), datetime(2024, 1, 8)],
            "fold": [0, 0],
        }
    )
    all_predictions = pl.DataFrame(
        {
            "config": ["nlinear_probe"] * 2,
            "epoch": [2, 2],
            entity: ["ES", "NQ"],
            "timestamp": [datetime(2024, 1, 8), datetime(2024, 1, 8)],
            "fold_id": [0, 0],
            "y_true": [0.01, -0.01],
            "y_score": [0.02, -0.02],
        }
    )
    context = SimpleNamespace(
        config={"config_name": "nlinear_probe"},
        entity_col=entity,
        expected_keys=expected_keys,
        label_col="fwd_ret_1d",
        prediction_split="validation",
        published_checkpoints=None,
    )
    published = []

    def capture(_training, **kwargs):
        published.append(kwargs["predictions"])
        return kwargs["predictions"]

    study = SimpleNamespace(results=SimpleNamespace(publish_predictions=capture))
    computation = {"checkpoint_schedule": [{"kind": "epoch", "value": 2}]}

    results = deep_learning._publish_sequence_predictions(
        study, computation, context, object(), {"all_predictions": all_predictions}
    )

    assert len(results) == 1 and len(published) == 1
    frame = published[0]
    assert "symbol" in frame.columns
    assert entity not in set(frame.columns) - {"symbol"}
    assert (
        frame.select("symbol", "timestamp", "fold")
        .sort("symbol")
        .equals(expected_keys.sort("symbol"))
    )
