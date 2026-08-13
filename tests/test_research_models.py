from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from case_studies.research import (
    CVSpec,
    LabelDefinition,
    ModelRun,
    Study,
    get_adapter,
    register_adapter,
    registered_adapters,
)
from case_studies.research.results import ResultsCatalog
from case_studies.utils import gbm as gbm_utils
from case_studies.utils import linear
from tests.test_research_workspace import _seed_release
from utils import modeling


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _linear_study(tmp_path, monkeypatch):
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    symbols = [f"S{index}" for index in range(6)]
    rows = []
    for date_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            x1 = float(symbol_index - 2.5)
            x2 = float(date_index) + x1 / 10
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": date,
                    "x1": x1,
                    "x2": x2,
                    "fwd_ret_1d": 0.03 * x1 + 0.01 * x2,
                }
            )
    dataset = pl.DataFrame(rows).with_columns(pl.col("timestamp").str.to_date())
    study.labels.publish(
        LabelDefinition("fwd_ret_1d", "regression", "1D"),
        dataset.select("symbol", "timestamp", "fwd_ret_1d"),
    )
    splits = [
        {
            "fold": 0,
            "train_start": "2024-01-01",
            "train_end": "2024-01-02",
            "val_start": "2024-01-03",
            "val_end": "2024-01-03",
        },
        {
            "fold": 1,
            "train_start": "2024-01-01",
            "train_end": "2024-01-03",
            "val_start": "2024-01-04",
            "val_end": "2024-01-04",
        },
    ]
    modeling_dataset = SimpleNamespace(
        dataset=dataset,
        feature_names=["x1", "x2"],
        label_col="fwd_ret_1d",
        date_col="timestamp",
        entity_cols=["symbol"],
        join_cols=["symbol", "timestamp"],
        splits=splits,
        task_type="regression",
        class_values=[],
        temporal_by_fold=None,
        temporal_keys=[],
        temporal_feature_names=[],
        eval_label_col=None,
        input_lineage={
            "artifacts": {
                "financial": {"sha256": "features-v1", "size": 1},
                "label": {"sha256": "label-v1", "size": 1},
            },
            "feature_names": ["x1", "x2"],
            "splits": splits,
            "fingerprint": "fixture-v1",
        },
    )
    monkeypatch.setattr(linear, "load_modeling_dataset", lambda *args, **kwargs: modeling_dataset)
    monkeypatch.setattr(
        linear,
        "_load_preset",
        lambda config_name: {
            "config_name": config_name,
            "family": "linear",
            "library": "sklearn",
            "model_class": "Ridge",
            "params": {"alpha": 1.0},
        },
    )
    return study


def test_linear_notebook_and_public_request_resolve_identically(tmp_path, monkeypatch) -> None:
    study = _linear_study(tmp_path, monkeypatch)
    notebook_request = study.model(
        family="linear",
        label="fwd_ret_1d",
        config_name="ridge",
        overrides={"alpha": 2.5},
    )
    api_request = study.model(
        family="linear",
        label="fwd_ret_1d",
        config_name="ridge",
        overrides={"alpha": 2.5},
    )

    notebook_resolved = notebook_request.resolve()
    api_resolved = api_request.resolve()

    assert notebook_resolved.identity == api_resolved.identity
    assert notebook_resolved.spec == api_resolved.spec
    assert notebook_resolved.spec["model"]["effective_params_by_fold"]["0"]["alpha"] == 2.5


def test_linear_runner_persists_complete_reusable_result(tmp_path, monkeypatch) -> None:
    study = _linear_study(tmp_path, monkeypatch)
    request = study.model(family="linear", label="fwd_ret_1d", config_name="ridge")

    fresh = request.run()
    cached = request.run()

    assert isinstance(fresh, ModelRun)
    assert fresh.training.hash == cached.training.hash
    assert fresh.predictions[0].hash == cached.predictions[0].hash
    assert fresh.predictions[0].complete
    assert fresh.predictions[0].coverage()["n_expected"] == 12
    assert fresh.predictions[0].load().select("symbol", "timestamp", "fold").height == 12
    model_dir = fresh.training.root / "run_log" / "training" / fresh.training.hash / "models"
    assert sorted(path.name for path in model_dir.glob("fold_*.joblib")) == [
        "fold_0.joblib",
        "fold_1.joblib",
    ]


def test_linear_override_changes_training_identity(tmp_path, monkeypatch) -> None:
    study = _linear_study(tmp_path, monkeypatch)
    base = study.model(family="linear", label="fwd_ret_1d", config_name="ridge").resolve()
    changed = study.model(
        family="linear",
        label="fwd_ret_1d",
        config_name="ridge",
        overrides={"alpha": 3.0},
    ).resolve()

    assert base.identity != changed.identity


def test_linear_effective_params_bind_seed_for_stochastic_estimators() -> None:
    config = {
        "model_class": "LogisticRegression",
        "params": {"solver": "liblinear"},
    }
    folds = [{"fold": 0, "X_train": np.ones((4, 2)), "y_train": np.array([0, 1, 0, 1])}]

    effective = linear._effective_params(config, {}, folds)

    assert effective["0"]["random_state"] == 42


def test_custom_cv_uses_label_timeline_not_feature_availability() -> None:
    timeline = pl.DataFrame(
        {"timestamp": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), eager=True)}
    )
    request = {
        "cv": CVSpec.walk_forward(
            training_window="20D",
            validation_window="5D",
            folds=(0,),
            horizon="0D",
        ),
        "preview_reductions": {},
    }
    full = SimpleNamespace(splits=[], date_col="timestamp", dataset=timeline)
    reduced = SimpleNamespace(splits=[], date_col="timestamp", dataset=timeline.tail(12))

    full_splits, full_record = linear._select_splits(full, request, timeline)
    reduced_splits, reduced_record = linear._select_splits(reduced, request, timeline)

    assert full_splits == reduced_splits
    assert full_record == reduced_record


def test_product_entity_is_normalized_at_prediction_boundary() -> None:
    meta = pl.DataFrame(
        {"product": ["ES", "NQ"], "timestamp": ["2024-01-01", "2024-01-01"]}
    ).to_pandas()

    expected = linear._expected_keys([{"fold": 0, "meta": meta}], "product", "timestamp")

    assert expected.columns == ["symbol", "timestamp", "fold"]
    assert expected.get_column("symbol").to_list() == ["ES", "NQ"]


def _gbm_study(tmp_path, monkeypatch):
    study = _linear_study(tmp_path, monkeypatch)
    modeling_dataset = linear.load_modeling_dataset("etfs", "fwd_ret_1d")
    monkeypatch.setattr(modeling, "load_modeling_dataset", lambda *args, **kwargs: modeling_dataset)
    monkeypatch.setattr(
        modeling,
        "load_configs",
        lambda *args, **kwargs: [
            {
                "checkpoint_interval": 2,
                "config_name": "leaves_7_mse",
                "family": "gbm",
                "library": "lightgbm",
                "max_iterations": 4,
                "params": {
                    "learning_rate": 0.1,
                    "min_child_samples": 2,
                    "num_leaves": 7,
                    "objective": "regression",
                    "seed": 42,
                },
            }
        ],
    )

    def fake_train(config, fold_data, *, save_dir, **kwargs):
        booster_dir = save_dir / "boosters"
        booster_dir.mkdir(parents=True)
        checkpoints = gbm_utils.gbm_checkpoint_iterations(config)
        predictions = []
        for fold in fold_data:
            (booster_dir / f"fold_{fold['fold']}.txt").write_text(f"fold={fold['fold']}\n")
            for checkpoint in checkpoints:
                predictions.append(
                    {
                        "dates": fold["dates"],
                        "entities": fold["entities"],
                        "y_true": fold["y_val"],
                        "y_eval": fold.get("y_eval"),
                        "y_pred": fold["X_val"][:, 0] + checkpoint / 100,
                        "fold": fold["fold"],
                        "n_trees": checkpoint,
                    }
                )
        return {
            "learning_curves": [
                {
                    "config": config["config_name"],
                    "iteration": checkpoint,
                    "ic_mean": 0.01 * checkpoint,
                    "ic_std": 0.0,
                }
                for checkpoint in checkpoints
            ],
            "predictions": predictions,
        }

    monkeypatch.setattr(gbm_utils, "train_gbm_config", fake_train)
    return study


def test_gbm_runner_persists_every_declared_checkpoint(tmp_path, monkeypatch) -> None:
    study = _gbm_study(tmp_path, monkeypatch)
    request = study.model(
        family="gbm",
        label="fwd_ret_1d",
        config_name="leaves_7_mse",
        overrides={"device": "cpu", "max_bin": 63},
    )

    resolved = request.resolve()
    fresh = resolved.run()
    cached = request.run()

    assert [item["value"] for item in resolved.spec["checkpoint_schedule"]] == [2, 4]
    assert len(fresh.predictions) == len(cached.predictions) == 2
    assert [result.hash for result in fresh.predictions] == [
        result.hash for result in cached.predictions
    ]
    assert all(result.complete for result in fresh.predictions)
    assert all(result.coverage()["n_expected"] == 12 for result in fresh.predictions)
    model_dir = fresh.training.root / "run_log" / "training" / fresh.training.hash / "models"
    assert sorted(path.name for path in (model_dir / "boosters").glob("*.txt")) == [
        "fold_0.txt",
        "fold_1.txt",
    ]


def test_gbm_runner_replays_valid_models_after_partial_registration(tmp_path, monkeypatch) -> None:
    study = _gbm_study(tmp_path, monkeypatch)
    request = study.model(
        family="gbm",
        label="fwd_ret_1d",
        config_name="leaves_7_mse",
        overrides={"device": "cpu", "max_bin": 63},
    )
    original_publish = ResultsCatalog.publish_predictions
    calls = 0

    def interrupt_after_first(catalog, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted registration")
        return original_publish(catalog, *args, **kwargs)

    monkeypatch.setattr(ResultsCatalog, "publish_predictions", interrupt_after_first)
    with pytest.raises(RuntimeError, match="interrupted registration"):
        request.run()
    monkeypatch.setattr(ResultsCatalog, "publish_predictions", original_publish)
    monkeypatch.setattr(
        gbm_utils,
        "_predict_from_gbm_models",
        lambda model_dir, spec, context: {
            "learning_curves": [],
            "predictions": [
                {
                    "dates": fold["dates"],
                    "entities": fold["entities"],
                    "y_true": fold["y_val"],
                    "y_eval": fold.get("y_eval"),
                    "y_pred": fold["X_val"][:, 0] + checkpoint["value"] / 100,
                    "fold": fold["fold"],
                    "n_trees": checkpoint["value"],
                }
                for fold in context.folds
                for checkpoint in spec["checkpoint_schedule"]
            ],
        },
    )
    monkeypatch.setattr(
        gbm_utils,
        "train_gbm_config",
        lambda *args, **kwargs: pytest.fail("valid fitted state must not retrain"),
    )

    recovered = request.run()

    assert len(recovered.predictions) == 2
    assert all(result.complete for result in recovered.predictions)


def test_huber_threshold_is_fold_scaled_and_hash_covered() -> None:
    folds = [
        {"fold": 0, "y_train": np.array([-0.02, 0.0, 0.02])},
        {"fold": 1, "y_train": np.array([-0.2, 0.0, 0.2])},
    ]
    config = {
        "huber_alpha_scale": 0.5,
        "params": {"objective": "huber", "seed": 42},
    }

    effective = gbm_utils._gbm_effective_params_by_fold(
        config,
        folds,
        device="cpu",
        max_bin=63,
        num_threads=1,
        seed=42,
    )

    assert effective["0"]["alpha"] == pytest.approx(0.5 * np.std(folds[0]["y_train"]))
    assert effective["1"]["alpha"] == pytest.approx(0.5 * np.std(folds[1]["y_train"]))
    assert effective["0"]["alpha"] != effective["1"]["alpha"]


def test_preview_request_requires_hash_covered_reductions(tmp_path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    with pytest.raises(ValueError, match="declare every reduction"):
        study.model(
            family="linear",
            label="fwd_ret_21d",
            config_name="ridge",
            execution_tier="preview",
        )


def test_model_and_causal_adapters_have_one_extension_seam() -> None:
    register_adapter("model", "fixture_family", "case_studies.utils.linear")
    register_adapter("causal", "fixture_causal", "case_studies.utils.causal")

    assert get_adapter("model", "fixture_family") is linear
    assert get_adapter("causal", "fixture_causal").__name__ == "case_studies.utils.causal"
    assert "tabular_dl" in {binding.name for binding in registered_adapters("model")}
    assert "dml" in {binding.name for binding in registered_adapters("causal")}
