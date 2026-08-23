from __future__ import annotations

import gc
import json
import os
import weakref
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
import torch
import yaml

from case_studies.research import (
    CVSpec,
    LabelDefinition,
    ModelRun,
    Study,
    get_adapter,
    plan_models,
    register_adapter,
    registered_adapters,
    run_models,
)
from case_studies.research.results import ResultsCatalog
from case_studies.utils import gbm as gbm_utils
from case_studies.utils import linear, tabular_dl
from case_studies.utils.latent_factors import adapter as latent_adapter
from case_studies.utils.latent_factors import case_study as latent_case_study
from case_studies.utils.latent_factors.cae import run_cae_fold
from case_studies.utils.latent_factors.case_study import LatentFactorCaseStudyContext
from case_studies.utils.latent_factors.cv import _expected_latent_checkpoints, load_fold_extras
from case_studies.utils.latent_factors.ipca import run_ipca_fold
from case_studies.utils.latent_factors.library_bridge import (
    predict_latent_fold_from_artifact,
)
from case_studies.utils.latent_factors.sae import run_sae_fold
from case_studies.utils.latent_factors.sdf import run_sdf_fold
from tests.test_research_registry import _predictions
from tests.test_research_workspace import _seed_release
from utils import modeling


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _linear_study(tmp_path, monkeypatch, *, n_symbols: int = 6):
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    symbols = [f"S{index}" for index in range(n_symbols)]
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
        temporal_artifact_splits=[],
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


def _tabm_study(tmp_path, monkeypatch):
    study = _linear_study(tmp_path, monkeypatch, n_symbols=60)
    modeling_dataset = linear.load_modeling_dataset("etfs", "fwd_ret_1d")
    loads: list[tuple[str, str, int]] = []

    def load_dataset(case_study, label, *, max_symbols=0):
        loads.append((case_study, label, max_symbols))
        return modeling_dataset

    configs = [
        {
            "batch_size": 64,
            "checkpoint_interval": 1,
            "config_name": name,
            "family": "tabular_dl",
            "library": "tabm",
            "n_epochs": 1,
            "params": {"dropout": 0.0, "hidden_dim": hidden, "n_members": 2},
        }
        for name, hidden in (("tabm_s", 4), ("tabm_m", 8))
    ]
    monkeypatch.setattr(modeling, "load_modeling_dataset", load_dataset)
    monkeypatch.setattr(modeling, "load_configs", lambda *args, **kwargs: configs)
    return study, modeling_dataset, loads


def test_tabm_public_batch_materializes_and_prepares_compatible_panel_once(
    tmp_path,
    monkeypatch,
) -> None:
    study, modeling_dataset, loads = _tabm_study(tmp_path, monkeypatch)
    conversions = 0
    executions: list[tuple[str, ...]] = []
    original_to_pandas = pl.DataFrame.to_pandas

    def counted_to_pandas(frame, *args, **kwargs):
        nonlocal conversions
        if frame is modeling_dataset.dataset:
            conversions += 1
        return original_to_pandas(frame, *args, **kwargs)

    def fake_run_tabm_cv(
        dataset_pd,
        splits,
        *,
        configs,
        checkpoint_root,
        _recovery=None,
        **kwargs,
    ):
        executions.append(tuple(config["config_name"] for config in configs))
        prediction_frames = []
        for config_index, config in enumerate(configs):
            config_name = config["config_name"]
            candidate_key = config.get("_execution_key", config_name)
            for split in splits:
                fold = int(split["fold"])
                root = _recovery.model_root(candidate_key) if _recovery else checkpoint_root
                checkpoint = root / config_name / f"fold_{fold:02d}" / "epoch_0001.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text("fixture\n")
                valid = dataset_pd["timestamp"].between(
                    split["val_start"], split["val_end"], inclusive="both"
                )
                frame = dataset_pd.loc[valid]
                prediction_frames.append(
                    pl.DataFrame(
                        {
                            "timestamp": frame["timestamp"],
                            "symbol": frame["symbol"],
                            "y_true": frame["fwd_ret_1d"],
                            "y_score": frame["x1"] + config_index / 100,
                            "fold_id": [fold] * len(frame),
                            "config": [candidate_key] * len(frame),
                            "epoch": [1] * len(frame),
                        }
                    )
                )
        all_predictions = pl.concat(prediction_frames)
        return {
            "all_learning_curves": pl.DataFrame(),
            "all_predictions": all_predictions,
            "execution_diagnostics": {
                "base_fold_preparation_s": 0.2,
                "base_fold_preparations": len(splits),
                "candidate_fit_s": {
                    config.get("_execution_key", config["config_name"]): 0.1 for config in configs
                },
            },
            "fold_metrics": pl.DataFrame(),
            "grid_results": [
                {
                    "best_epoch": 1,
                    "best_ic": 0.0,
                    "config_name": config.get("_execution_key", config["config_name"]),
                    "elapsed_s": 0.0,
                }
                for config in configs
            ],
            "predictions": all_predictions,
            "training_log": pl.DataFrame(),
        }

    monkeypatch.setattr(pl.DataFrame, "to_pandas", counted_to_pandas)
    monkeypatch.setattr(tabular_dl, "run_tabm_cv", fake_run_tabm_cv)
    monkeypatch.setattr(
        "case_studies.utils.deep_model_state.validate_deep_checkpoint_population",
        lambda *args, **kwargs: (),
    )
    requests = [
        study.model(
            family="tabular_dl",
            label="fwd_ret_1d",
            config_name=config_name,
            overrides={"device": "cpu", "num_threads": 1},
        )
        for config_name in ("tabm_s", "tabm_m")
    ]

    plan = plan_models(study, requests=requests)
    population = plan.create_population(
        name="planned-tabm-checkpoints",
    )
    assert loads == [("etfs", "fwd_ret_1d", 0)]
    assert conversions == 1
    assert executions == []
    with pytest.raises(ValueError, match="incomplete"):
        population.require_complete()
    result = plan.run()

    assert loads == [("etfs", "fwd_ret_1d", 0)]
    assert conversions == 1
    assert executions == [("tabm_s", "tabm_m")]
    assert len(result.runs) == 2
    assert len({run.training.hash for run in result.runs}) == 2
    assert all(run.predictions[0].complete for run in result.runs)
    assert all(run.diagnostics["execution_order"] == "fold_major" for run in result.runs)
    assert all(run.diagnostics["base_fold_preparations"] == 2 for run in result.runs)
    assert all(run.diagnostics["compatibility_group_size"] == 2 for run in result.runs)
    assert all(run.diagnostics["disk_fold_cache"] is False for run in result.runs)
    assert population.require_complete() == plan.expected_prediction_hashes


def test_tabm_cv_releases_each_prepared_fold_after_all_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    _, modeling_dataset, _ = _tabm_study(tmp_path, monkeypatch)
    configs = modeling.load_configs("etfs", "fwd_ret_1d", "tabular_dl")
    original_prepare = tabular_dl._prepare_tabm_fold
    released_arrays: list[weakref.ReferenceType[np.ndarray]] = []
    prepared_folds: list[int] = []
    training_order: list[tuple[str, int]] = []

    def observed_prepare(*args, **kwargs):
        gc.collect()
        if released_arrays:
            assert released_arrays[-1]() is None
        fold = original_prepare(*args, **kwargs)
        prepared_folds.append(int(fold["fold"]))
        released_arrays.append(weakref.ref(fold["X_train"]))
        return fold

    def fake_train_tabm_fold(*, model, X_val, val_dates, **kwargs):
        hidden_dim = int(model.backbone[0].out_features)
        training_order.append((str(val_dates[0])[:10], hidden_dim))
        predictions = np.asarray(X_val[:, 0], dtype=np.float64)
        return {1: 0.1}, {1: predictions}, {1: 0.01}

    monkeypatch.setattr(tabular_dl, "_prepare_tabm_fold", observed_prepare)
    monkeypatch.setattr(tabular_dl, "_train_tabm_fold", fake_train_tabm_fold)

    result = tabular_dl.run_tabm_cv(
        modeling_dataset.dataset.to_pandas(),
        modeling_dataset.splits,
        configs=configs,
        n_features=len(modeling_dataset.feature_names),
        feature_names=modeling_dataset.feature_names,
        label_col=modeling_dataset.label_col,
        date_col=modeling_dataset.date_col,
        entity_col=modeling_dataset.entity_cols[0],
        device="cpu",
        seed=42,
        num_threads=1,
        strict=True,
    )

    assert prepared_folds == [0, 1]
    assert training_order == [
        ("2024-01-03", 4),
        ("2024-01-03", 8),
        ("2024-01-04", 4),
        ("2024-01-04", 8),
    ]
    assert result["all_predictions"].height == 240
    assert result["execution_diagnostics"]["base_fold_preparations"] == 2


def test_tabm_cv_rejects_empty_fold_population(tmp_path, monkeypatch) -> None:
    _, modeling_dataset, _ = _tabm_study(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="at least one fold"):
        tabular_dl.run_tabm_cv(
            modeling_dataset.dataset.to_pandas(),
            [],
            configs=modeling.load_configs("etfs", "fwd_ret_1d", "tabular_dl")[:1],
            n_features=len(modeling_dataset.feature_names),
            feature_names=modeling_dataset.feature_names,
            label_col=modeling_dataset.label_col,
            date_col=modeling_dataset.date_col,
            device="cpu",
        )


def test_tabm_batch_reuses_completed_fold_after_interruption(tmp_path, monkeypatch) -> None:
    study, _, _ = _tabm_study(tmp_path, monkeypatch)
    calls: list[str] = []
    fail_second_fold = True

    def interrupted_train(*, model, X_val, val_dates, state_callback=None, **kwargs):
        nonlocal fail_second_fold
        validation_date = str(val_dates[0])[:10]
        calls.append(validation_date)
        if validation_date == "2024-01-04" and fail_second_fold:
            fail_second_fold = False
            raise RuntimeError("injected TabM interruption")
        if state_callback is not None:
            state_callback(1, model)
        predictions = np.asarray(X_val[:, 0], dtype=np.float64)
        return {1: 0.1}, {1: predictions}, {1: 0.01}

    monkeypatch.setattr(tabular_dl, "_train_tabm_fold", interrupted_train)
    request = study.model(
        family="tabular_dl",
        label="fwd_ret_1d",
        config_name="tabm_s",
        overrides={"device": "cpu", "num_threads": 1},
    )

    with pytest.raises(RuntimeError, match="injected TabM interruption"):
        request.run()
    recovered = request.run()

    assert calls == ["2024-01-03", "2024-01-04", "2024-01-04"]
    assert recovered.diagnostics["reused_folds"] == [0]
    assert recovered.diagnostics["fitted_folds"] == [1]
    assert recovered.diagnostics["base_fold_preparations"] == 1
    assert recovered.predictions[0].complete
    assert recovered.predictions[0].coverage()["n_expected"] == 120
    training_log = pl.read_parquet(
        recovered.training.root
        / "run_log"
        / "training"
        / recovered.training.hash
        / "diagnostics"
        / "training_log.parquet"
    )
    assert set(training_log["fold"]) == {0, 1}


def test_tabm_candidate_failure_preserves_completed_sibling(tmp_path, monkeypatch) -> None:
    study, _, _ = _tabm_study(tmp_path, monkeypatch)
    calls: list[tuple[int, str]] = []
    fail_small = True

    def candidate_train(*, model, X_val, val_dates, state_callback=None, **kwargs):
        hidden_dim = int(model.backbone[0].out_features)
        validation_date = str(val_dates[0])[:10]
        calls.append((hidden_dim, validation_date))
        if hidden_dim == 4 and fail_small:
            raise RuntimeError("injected TabM candidate failure")
        if state_callback is not None:
            state_callback(1, model)
        predictions = np.asarray(X_val[:, 0], dtype=np.float64) + hidden_dim / 100
        return {1: 0.1}, {1: predictions}, {1: 0.01}

    monkeypatch.setattr(tabular_dl, "_train_tabm_fold", candidate_train)
    requests = [
        study.model(
            family="tabular_dl",
            label="fwd_ret_1d",
            config_name=config_name,
            overrides={"device": "cpu", "num_threads": 1},
        )
        for config_name in ("tabm_s", "tabm_m")
    ]
    plan = plan_models(study, requests=requests)
    population = plan.create_population(
        name="planned-tabm-recovery",
    )

    with pytest.raises(RuntimeError, match="injected TabM candidate failure"):
        plan.run()
    first_calls = list(calls)
    with pytest.raises(ValueError, match="incomplete"):
        population.require_complete()
    fail_small = False
    recovered_plan = plan_models(study, requests=requests)
    assert recovered_plan.expected_prediction_hashes == plan.expected_prediction_hashes
    recovered = recovered_plan.run()

    assert first_calls == [
        (4, "2024-01-03"),
        (8, "2024-01-03"),
        (8, "2024-01-04"),
    ]
    assert calls[len(first_calls) :] == [
        (4, "2024-01-03"),
        (4, "2024-01-04"),
    ]
    assert recovered.runs[1].diagnostics["reused"] is True
    assert {run.diagnostics["base_fold_preparations"] for run in recovered.runs} == {2}
    assert len({run.diagnostics["preparation_fraction"] for run in recovered.runs}) == 1
    assert recovered.runs[1].diagnostics["candidate_fit_s"] == 0.0
    assert all(run.predictions[0].complete for run in recovered.runs)
    assert population.require_complete() == plan.expected_prediction_hashes


def test_tabm_batch_matches_individual_resolved_execution(tmp_path, monkeypatch) -> None:
    batch_study, _, _ = _tabm_study(tmp_path / "batch", monkeypatch)
    individual_study, _, _ = _tabm_study(tmp_path / "individual", monkeypatch)

    def deterministic_train(*, model, X_val, state_callback=None, **kwargs):
        if state_callback is not None:
            state_callback(1, model)
        hidden_dim = int(model.backbone[0].out_features)
        predictions = np.asarray(X_val[:, 0], dtype=np.float64) + hidden_dim / 100
        return {1: 0.1}, {1: predictions}, {1: 0.01}

    monkeypatch.setattr(tabular_dl, "_train_tabm_fold", deterministic_train)
    batch = run_models(
        batch_study,
        requests=[
            batch_study.model(
                family="tabular_dl",
                label="fwd_ret_1d",
                config_name=config_name,
                overrides={"device": "cpu", "num_threads": 1},
            )
            for config_name in ("tabm_s", "tabm_m")
        ],
    )
    individual = (
        individual_study.model(
            family="tabular_dl",
            label="fwd_ret_1d",
            config_name="tabm_s",
            overrides={"device": "cpu", "num_threads": 1},
        )
        .resolve()
        .run()
    )

    assert batch.runs[0].training.hash == individual.training.hash
    assert batch.runs[0].predictions[0].hash == individual.predictions[0].hash
    assert batch.runs[0].predictions[0].load().equals(individual.predictions[0].load())


def test_tabm_batch_rejects_corrupt_checkpoint_and_refits_only_its_fold(
    tmp_path,
    monkeypatch,
) -> None:
    study, _, _ = _tabm_study(tmp_path, monkeypatch)
    calls: list[str] = []

    def deterministic_train(*, model, X_val, val_dates, state_callback=None, **kwargs):
        calls.append(str(val_dates[0])[:10])
        if state_callback is not None:
            state_callback(1, model)
        predictions = np.asarray(X_val[:, 0], dtype=np.float64)
        return {1: 0.1}, {1: predictions}, {1: 0.01}

    monkeypatch.setattr(tabular_dl, "_train_tabm_fold", deterministic_train)
    request = study.model(
        family="tabular_dl",
        label="fwd_ret_1d",
        config_name="tabm_s",
        overrides={"device": "cpu", "num_threads": 1},
    )
    original = request.run()
    model_root = original.training.root / "run_log" / "training" / original.training.hash / "models"
    checkpoint = model_root / original.training.hash / "fold_00" / "epoch_0001.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")

    recovered = request.run()

    assert calls == ["2024-01-03", "2024-01-04", "2024-01-03"]
    assert recovered.training.hash == original.training.hash
    assert recovered.diagnostics["fitted_folds"] == [0]
    assert recovered.diagnostics["reused_folds"] == [1]
    assert recovered.diagnostics["base_fold_preparations"] == 1
    training_log = pl.read_parquet(
        recovered.training.root
        / "run_log"
        / "training"
        / recovered.training.hash
        / "diagnostics"
        / "training_log.parquet"
    )
    assert set(training_log["fold"]) == {0, 1}
    invalid = (
        original.training.root / "run_log" / "training" / original.training.hash / "invalid_folds"
    )
    assert len(list(invalid.glob("fold_0.*"))) == 1


def test_tabm_saved_weight_checkpoint_replays_validation_predictions(
    tmp_path,
    monkeypatch,
) -> None:
    from case_studies.utils.deep_model_state import restore_deep_model

    study, modeling_dataset, _ = _tabm_study(tmp_path, monkeypatch)
    run = study.model(
        family="tabular_dl",
        label="fwd_ret_1d",
        config_name="tabm_s",
        overrides={"device": "cpu", "num_threads": 1},
    ).run()
    model_root = run.training.root / "run_log" / "training" / run.training.hash / "models"
    checkpoint = model_root / run.training.hash / "fold_00" / "epoch_0001.pt"
    model, preprocessing, metadata = restore_deep_model(
        checkpoint,
        lambda architecture, kwargs: tabular_dl.TabMModel(**kwargs),
    )
    sorted_dataset = (
        modeling_dataset.dataset.to_pandas()
        .sort_values(["timestamp", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    fold = tabular_dl._prepare_tabm_fold(
        sorted_dataset,
        modeling_dataset.splits[0],
        feature_names=modeling_dataset.feature_names,
        label_col=modeling_dataset.label_col,
        eval_label_col=None,
        date_col=modeling_dataset.date_col,
        entity_col=modeling_dataset.entity_cols[0],
        temporal_by_fold=None,
        temporal_keys=[],
        temporal_feature_names=[],
    )
    replay = tabular_dl._predict_in_chunks(model, fold["X_val"], torch.device("cpu"))
    stored = (
        run.predictions[0]
        .load()
        .filter(pl.col("fold") == 0)
        .sort("timestamp", "symbol")
        .get_column("prediction")
        .to_numpy()
    )

    assert metadata["fold"] == 0
    assert preprocessing["feature_names"] == modeling_dataset.feature_names
    np.testing.assert_allclose(replay, stored, rtol=0.0, atol=1e-7)


def test_tabm_corrupt_diagnostics_are_rebuilt_from_completed_folds(tmp_path, monkeypatch) -> None:
    study, _, _ = _tabm_study(tmp_path, monkeypatch)
    prepared_folds: list[int] = []
    original_prepare = tabular_dl._prepare_tabm_fold

    def observed_prepare(*args, **kwargs):
        fold = original_prepare(*args, **kwargs)
        prepared_folds.append(int(fold["fold"]))
        return fold

    monkeypatch.setattr(tabular_dl, "_prepare_tabm_fold", observed_prepare)
    request = study.model(
        family="tabular_dl",
        label="fwd_ret_1d",
        config_name="tabm_s",
        overrides={"device": "cpu", "num_threads": 1},
    )
    original = request.run()
    training_log = (
        original.training.root
        / "run_log"
        / "training"
        / original.training.hash
        / "diagnostics"
        / "training_log.parquet"
    )
    training_log.write_bytes(b"truncated")

    repaired = request.run()

    assert repaired.training.hash == original.training.hash
    assert repaired.diagnostics["base_fold_preparations"] == 0
    assert prepared_folds == [0, 1]
    assert set(pl.read_parquet(training_log)["fold"]) == {0, 1}

    selected_path = training_log.parent / "predictions.parquet"
    obsolete = (
        pl.read_parquet(selected_path)
        .drop("model_id")
        .with_columns(
            pl.lit("tabm_s").alias("config"),
            pl.lit(1, dtype=pl.Int32).alias("epoch"),
        )
    )
    obsolete.write_parquet(selected_path)

    request.run()

    selected = pl.read_parquet(selected_path)
    assert "model_id" in selected.columns
    assert {"config", "epoch"}.isdisjoint(selected.columns)


def test_tabm_candidate_order_does_not_change_identities_or_predictions(
    tmp_path,
    monkeypatch,
) -> None:
    forward_study, _, _ = _tabm_study(tmp_path / "forward", monkeypatch)
    reverse_study, _, _ = _tabm_study(tmp_path / "reverse", monkeypatch)

    def deterministic_train(*, model, X_val, state_callback=None, **kwargs):
        if state_callback is not None:
            state_callback(1, model)
        hidden_dim = int(model.backbone[0].out_features)
        predictions = np.asarray(X_val[:, 0], dtype=np.float64) + hidden_dim / 100
        return {1: 0.1}, {1: predictions}, {1: 0.01}

    monkeypatch.setattr(tabular_dl, "_train_tabm_fold", deterministic_train)

    def execute(study, names):
        result = run_models(
            study,
            requests=[
                study.model(
                    family="tabular_dl",
                    label="fwd_ret_1d",
                    config_name=name,
                    overrides={"device": "cpu", "num_threads": 1},
                )
                for name in names
            ],
        )
        return {run.training.spec()["config_name"]: run for run in result.runs}

    forward = execute(forward_study, ("tabm_s", "tabm_m"))
    reverse = execute(reverse_study, ("tabm_m", "tabm_s"))

    assert set(forward) == set(reverse)
    for name in forward:
        assert forward[name].training.hash == reverse[name].training.hash
        assert forward[name].predictions[0].hash == reverse[name].predictions[0].hash
        assert forward[name].predictions[0].load().equals(reverse[name].predictions[0].load())


def test_tabm_variants_from_one_named_preset_keep_separate_identities(
    tmp_path,
    monkeypatch,
) -> None:
    study, _, _ = _tabm_study(tmp_path, monkeypatch)
    prepared_folds: list[int] = []
    original_prepare = tabular_dl._prepare_tabm_fold

    def observed_prepare(*args, **kwargs):
        fold = original_prepare(*args, **kwargs)
        prepared_folds.append(int(fold["fold"]))
        return fold

    def deterministic_train(*, model, X_val, state_callback=None, **kwargs):
        if state_callback is not None:
            state_callback(1, model)
        hidden_dim = int(model.backbone[0].out_features)
        predictions = np.asarray(X_val[:, 0], dtype=np.float64) + hidden_dim / 100
        return {1: 0.1}, {1: predictions}, {1: 0.01}

    monkeypatch.setattr(tabular_dl, "_train_tabm_fold", deterministic_train)
    monkeypatch.setattr(tabular_dl, "_prepare_tabm_fold", observed_prepare)
    result = run_models(
        study,
        requests=[
            study.model(
                family="tabular_dl",
                label="fwd_ret_1d",
                config_name="tabm_s",
                overrides={"device": "cpu", "hidden_dim": hidden, "num_threads": 1},
            )
            for hidden in (5, 6)
        ],
    )

    assert len({run.training.hash for run in result.runs}) == 2
    assert len({run.predictions[0].hash for run in result.runs}) == 2
    assert prepared_folds == [0, 1]
    assert {run.diagnostics["base_fold_preparations"] for run in result.runs} == {2}
    assert {run.training.spec()["identity_version"] for run in result.runs} == {3}
    assert {run.training.spec()["resolved_spec_schema"] for run in result.runs} == {
        "ml4t.resolved-spec/v1"
    }
    assert [
        run.training.spec()["computation"]["model"]["params"]["hidden_dim"] for run in result.runs
    ] == [5, 6]
    for run in result.runs:
        diagnostics = run.training.root / "run_log" / "training" / run.training.hash / "diagnostics"
        assert {path.name for path in diagnostics.iterdir()} == {
            "all_predictions.parquet",
            "learning_curves.parquet",
            "predictions.parquet",
            "result.json",
            "training_log.parquet",
        }
        selected = pl.read_parquet(diagnostics / "predictions.parquet")
        assert "model_id" in selected.columns
        assert {"config", "epoch"}.isdisjoint(selected.columns)


def test_tabm_duplicate_requests_share_one_execution(tmp_path, monkeypatch) -> None:
    study, _, _ = _tabm_study(tmp_path, monkeypatch)
    prepared_folds: list[int] = []
    original_prepare = tabular_dl._prepare_tabm_fold

    def observed_prepare(*args, **kwargs):
        fold = original_prepare(*args, **kwargs)
        prepared_folds.append(int(fold["fold"]))
        return fold

    def deterministic_train(*, model, X_val, state_callback=None, **kwargs):
        if state_callback is not None:
            state_callback(1, model)
        predictions = np.asarray(X_val[:, 0], dtype=np.float64)
        return {1: 0.1}, {1: predictions}, {1: 0.01}

    monkeypatch.setattr(tabular_dl, "_train_tabm_fold", deterministic_train)
    monkeypatch.setattr(tabular_dl, "_prepare_tabm_fold", observed_prepare)
    request = study.model(
        family="tabular_dl",
        label="fwd_ret_1d",
        config_name="tabm_s",
        overrides={"device": "cpu", "num_threads": 1},
    )

    result = run_models(study, requests=[request, request])

    assert len(result.runs) == 2
    assert result.runs[0].training.hash == result.runs[1].training.hash
    assert result.runs[0].predictions[0].hash == result.runs[1].predictions[0].hash
    assert prepared_folds == [0, 1]
    assert {run.diagnostics["base_fold_preparations"] for run in result.runs} == {2}


def test_tabm_equivalent_named_presets_share_identity_stable_checkpoints(
    tmp_path, monkeypatch
) -> None:
    study, _, _ = _tabm_study(tmp_path, monkeypatch)
    prepared_folds: list[int] = []
    original_prepare = tabular_dl._prepare_tabm_fold

    def observed_prepare(*args, **kwargs):
        fold = original_prepare(*args, **kwargs)
        prepared_folds.append(int(fold["fold"]))
        return fold

    monkeypatch.setattr(tabular_dl, "_prepare_tabm_fold", observed_prepare)
    requests = [
        study.model(
            family="tabular_dl",
            label="fwd_ret_1d",
            config_name=name,
            overrides={"device": "cpu", "hidden_dim": 6, "num_threads": 1},
        )
        for name in ("tabm_s", "tabm_m")
    ]

    result = run_models(study, requests=requests)
    replay = requests[1].run()

    assert result.runs[0].training.hash == result.runs[1].training.hash == replay.training.hash
    assert result.runs[0].predictions[0].hash == result.runs[1].predictions[0].hash
    assert prepared_folds == [0, 1]
    checkpoint_root = (
        replay.training.root
        / "run_log"
        / "training"
        / replay.training.hash
        / "models"
        / replay.training.hash
    )
    assert {path.parent.name for path in checkpoint_root.glob("fold_*/epoch_0001.pt")} == {
        "fold_00",
        "fold_01",
    }


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
    assert (
        notebook_resolved.spec["computation"]["model"]["effective_params_by_fold"]["0"]["alpha"]
        == 2.5
    )


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


def test_thread_limit_does_not_leak_into_families_that_already_record_threads(
    tmp_path, monkeypatch
) -> None:
    """gbm, tabular_dl and deep_learning must be byte-identical before and after this change.

    They already record thread state - gbm carries num_threads in _GBM_REQUEST_FIELDS,
    tabular_dl and deep_learning carry numerics blocks. Adding a second thread field to a
    family that already has one moves its training identities and buys nothing. The pinning
    helper is shared; the `numerics.thread_limit` field is applied to linear and causal only.
    """
    gbm_study = _gbm_study(tmp_path / "gbm", monkeypatch)
    gbm_spec = (
        gbm_study.model(
            family="gbm",
            label="fwd_ret_1d",
            config_name="leaves_7_mse",
            overrides={"device": "cpu", "max_bin": 63},
        )
        .resolve()
        .spec
    )
    gbm_numerics = gbm_spec["computation"].get("numerics", {})
    assert "thread_limit" not in gbm_numerics, (
        "thread_limit leaked into gbm, which already records num_threads; "
        "this moves every gbm training identity"
    )

    tabm_study, *_ = _tabm_study(tmp_path / "tabm", monkeypatch)
    tabm_spec = (
        tabm_study.model(family="tabular_dl", label="fwd_ret_1d", config_name="tabm_s")
        .resolve()
        .spec
    )
    assert "thread_limit" not in tabm_spec["computation"].get("numerics", {}), (
        "thread_limit leaked into tabular_dl, which already carries a numerics block"
    )

    # deep_learning builds its own numerics block at utils/deep_learning.py, so it is the third
    # family this scope protects and equally likely to be caught by a future widening.
    from tests.test_deep_learning_adapter import _resolve_nlinear_request

    _, _, sequence_resolved = _resolve_nlinear_request(tmp_path / "sequence", monkeypatch)
    sequence_spec = sequence_resolved.spec
    assert "thread_limit" not in sequence_spec["computation"].get("numerics", {}), (
        "thread_limit leaked into deep_learning, which already carries a numerics block"
    )
    assert sequence_spec["computation"]["numerics"]["num_threads"]

    # latent_factors records num_threads too (utils/latent_factors/adapter.py), so it is the
    # fourth family this scope protects and equally exposed to a future widening.
    latent_study = _latent_study(tmp_path / "latent", monkeypatch)
    latent_spec = (
        latent_study.model(
            family="latent_factors",
            label="fwd_ret_1d",
            config_name="pca",
            overrides={"device": "cpu"},
        )
        .resolve()
        .spec
    )
    assert "thread_limit" not in latent_spec["computation"].get("numerics", {}), (
        "thread_limit leaked into latent_factors, which already records num_threads"
    )


def test_linear_pins_the_thread_pool_and_records_it_in_identity(tmp_path, monkeypatch) -> None:
    """A linear fit is a deterministic function of the thread pool, not of the data alone.

    Coordinate descent and the BLAS kernels reduce in thread order, so two runs can agree on
    training_hash and prediction_hash and still differ in the coefficients. Measured on real
    crypto data: lasso_f0.08 at identical training_hash produced prediction digests
    37352b2ff14a4b6a uncapped (pools 16/24) against c81ca6b5302e1dc2 capped. Recording the
    limit is what makes the identity cover the computation.
    """
    import threadpoolctl

    study = _linear_study(tmp_path, monkeypatch)
    request = study.model(family="linear", label="fwd_ret_1d", config_name="ridge")
    resolved = request.resolve()

    numerics = resolved.spec["computation"]["numerics"]
    assert numerics["thread_limit"] == linear.LINEAR_THREAD_LIMIT
    assert numerics["deterministic_reduction"] is True

    other = deepcopy(resolved.spec)
    other["computation"]["numerics"]["thread_limit"] = linear.LINEAR_THREAD_LIMIT + 1
    assert linear.training_hash_from_spec(other) != linear.training_hash_from_spec(resolved.spec)

    # Observed from inside the limited block rather than by trusting the source: the runner
    # calls _fold_predictions immediately after model.fit, within the same context manager.
    observed: list[int] = []
    original_predictions = linear._fold_predictions

    def recording_predictions(model, fold, context):
        observed.extend(
            info["num_threads"]
            for info in threadpoolctl.threadpool_info()
            if info["user_api"] in {"openmp", "blas"}
        )
        return original_predictions(model, fold, context)

    monkeypatch.setattr(linear, "_fold_predictions", recording_predictions)
    request.run()

    assert observed, "no thread pool was observed during the linear fit"
    assert set(observed) == {linear.LINEAR_THREAD_LIMIT}, (
        f"linear fitted with pools at {sorted(set(observed))}, not {linear.LINEAR_THREAD_LIMIT}"
    )


def _observe_fold_preparation(monkeypatch) -> list[int]:
    """Record which folds are actually built, in the order they are built.

    Preparation is shared across configurations and cached between runs, so the seam that says
    how much work a run did is where a fold is constructed - not where one is asked for.
    """
    from case_studies.utils import folds as folds_module

    folds_module.clear_memo()
    built: list[int] = []
    original = folds_module.standardized_fold

    def observed(raw, *args, **kwargs):
        built.append(int(raw.fold))
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(folds_module, "standardized_fold", observed)
    return built


def _observe_fold_sets(monkeypatch) -> list[tuple[int, float]]:
    """Record every fold built, paired with the sampling fraction it was built under.

    Two configurations that subsample differently are not fitted on the same rows, so they need
    separate fold sets; this is where that separation is visible.
    """
    from case_studies.utils import folds as folds_module

    folds_module.clear_memo()
    built: list[tuple[int, float]] = []
    original = folds_module.iter_raw_folds

    # `iter_raw_folds`, not `prepare_raw_folds`: the batch paths stream folds so that only one is
    # alive at a time, and `prepare_raw_folds` is now the list() wrapper no consumer calls.
    # Observing the wrapper recorded nothing while the run underneath prepared every fold.
    def observed(mds, splits, *, train_sample_frac=1.0, **kwargs):
        for fold in original(mds, splits, train_sample_frac=train_sample_frac, **kwargs):
            built.append((int(fold.fold), float(train_sample_frac)))
            yield fold

    monkeypatch.setattr(folds_module, "iter_raw_folds", observed)
    return built


def test_a_fold_set_too_large_to_hold_is_released_as_it_is_consumed(tmp_path, monkeypatch) -> None:
    """Holding every fold is worth 0.9 GB on etfs and 44 GB on nasdaq100_microstructure.

    Above the budget nothing is retained, so a panel run costs one fold at a time rather than the
    whole set - the bound the fold-major batch loop was written for.
    """
    from case_studies.utils import folds as folds_module

    study = _linear_study(tmp_path, monkeypatch)
    monkeypatch.setenv("ML4T_FOLD_MEMO_BUDGET_BYTES", "0")
    folds_module.clear_memo()

    study.model(family="linear", label="fwd_ret_1d", config_name="ridge").run()

    assert not folds_module._STANDARDIZED_MEMO
    assert not folds_module._RAW_MEMO


def test_a_fold_set_within_budget_is_held_and_shared(tmp_path, monkeypatch) -> None:
    from case_studies.utils import folds as folds_module

    study = _linear_study(tmp_path, monkeypatch)
    folds_module.clear_memo()

    study.model(family="linear", label="fwd_ret_1d", config_name="ridge").run()

    assert folds_module._STANDARDIZED_MEMO


def test_linear_batch_is_fold_major_and_matches_individual_execution(tmp_path, monkeypatch) -> None:
    study = _linear_study(tmp_path / "batch", monkeypatch)
    individual_study = _linear_study(tmp_path / "individual", monkeypatch)
    prepared_folds = _observe_fold_preparation(monkeypatch)
    requests = [
        study.model(
            family="linear",
            label="fwd_ret_1d",
            config_name="ridge",
            overrides={"alpha": alpha},
        )
        for alpha in (1.0, 2.0)
    ]

    batch = run_models(study, requests=requests)
    batch_prepared_folds = list(prepared_folds)
    individual = (
        individual_study.model(
            family="linear",
            label="fwd_ret_1d",
            config_name="ridge",
            overrides={"alpha": 1.0},
        )
        .resolve()
        .run()
    )
    gc.collect()

    # Each fold is built once for the whole batch. It used to be built once per configuration
    # per pass, which is what made resolving 28 etfs configurations cost 313 seconds.
    assert batch_prepared_folds == [0, 1]
    assert batch.runs[0].training.hash == individual.training.hash
    batch_predictions = batch.runs[0].predictions[0].load()
    individual_predictions = individual.predictions[0].load()
    assert batch_predictions.select("symbol", "timestamp", "fold", "actual").equals(
        individual_predictions.select("symbol", "timestamp", "fold", "actual")
    )
    np.testing.assert_allclose(
        batch_predictions["prediction"],
        individual_predictions["prediction"],
        rtol=1e-12,
        atol=1e-15,
    )
    assert all(run.diagnostics["execution_order"] == "fold_major" for run in batch.runs)
    assert all(run.diagnostics["compatibility_group_size"] == 2 for run in batch.runs)
    assert all(run.diagnostics["disk_fold_cache"] is False for run in batch.runs)


def test_linear_batch_resolves_fold_dependent_parameters_before_fitting(
    tmp_path, monkeypatch
) -> None:
    study = _linear_study(tmp_path, monkeypatch)

    def load_preset(config_name):
        if config_name == "lasso":
            return {
                "config_name": config_name,
                "family": "linear",
                "library": "sklearn",
                "model_class": "Lasso",
                "params": {"alpha_frac": 0.5, "max_iter": 5000},
            }
        return {
            "config_name": config_name,
            "family": "linear",
            "library": "sklearn",
            "model_class": "Ridge",
            "params": {"alpha": 1.0},
        }

    monkeypatch.setattr(linear, "_load_preset", load_preset)
    prepared_folds = _observe_fold_preparation(monkeypatch)

    batch = run_models(
        study,
        requests=[
            study.model(
                family="linear",
                label="fwd_ret_1d",
                config_name=config_name,
            )
            for config_name in ("ridge", "lasso")
        ],
    )

    # Lasso's alpha is a fraction of each fold's own degeneracy threshold, so it cannot be known
    # until that fold exists. Both folds are built once and both configurations read them.
    assert prepared_folds == [0, 1]
    assert all(run.predictions[0].complete for run in batch.runs)
    lasso_params = batch.runs[1].training.spec()["computation"]["model"]["effective_params_by_fold"]
    assert set(lasso_params) == {"0", "1"}
    assert all(params["alpha"] > 0 for params in lasso_params.values())
    assert all("alpha_frac" not in params for params in lasso_params.values())
    assert all(run.diagnostics["base_fold_preparations"] == 2 for run in batch.runs)


def test_linear_model_plan_reuses_one_materialization_and_one_execution_fold_pass(
    tmp_path, monkeypatch
) -> None:
    study = _linear_study(tmp_path, monkeypatch)
    original_load = linear.load_modeling_dataset
    loads = 0

    def observed_load(*args, **kwargs):
        nonlocal loads
        loads += 1
        return original_load(*args, **kwargs)

    def load_preset(config_name):
        if config_name == "lasso":
            return {
                "config_name": config_name,
                "family": "linear",
                "library": "sklearn",
                "model_class": "Lasso",
                "params": {"alpha_frac": 0.5, "max_iter": 5000},
            }
        return {
            "config_name": config_name,
            "family": "linear",
            "library": "sklearn",
            "model_class": "Ridge",
            "params": {"alpha": 1.0},
        }

    monkeypatch.setattr(linear, "load_modeling_dataset", observed_load)
    monkeypatch.setattr(linear, "_load_preset", load_preset)
    prepared_folds = _observe_fold_preparation(monkeypatch)
    requests = [
        study.model(family="linear", label="fwd_ret_1d", config_name=config_name)
        for config_name in ("ridge", "lasso")
    ]

    plan = plan_models(study, requests=requests)
    population = plan.create_population(
        name="planned-linear-checkpoints",
    )
    with pytest.raises(ValueError, match="incomplete"):
        population.require_complete()
    execution = plan.run()

    assert loads == 1
    assert prepared_folds == [0, 1]
    assert tuple(run.training.hash for run in execution.runs) == plan.expected_training_hashes
    assert (
        tuple(prediction.hash for run in execution.runs for prediction in run.predictions)
        == plan.expected_prediction_hashes
    )
    # Planning materialised both folds; execution built none, which is what "one
    # materialization and one execution fold pass" now costs.
    assert all(run.diagnostics["base_fold_preparations"] == 0 for run in execution.runs)
    assert population.require_complete() == plan.expected_prediction_hashes


def test_linear_batch_separates_incompatible_sampling_and_is_order_invariant(
    tmp_path, monkeypatch
) -> None:
    first = _linear_study(tmp_path / "first", monkeypatch)
    second = _linear_study(tmp_path / "second", monkeypatch)
    original_load = linear.load_modeling_dataset
    load_count = 0

    def observed_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return original_load(*args, **kwargs)

    preparation = _observe_fold_sets(monkeypatch)
    monkeypatch.setattr(linear, "load_modeling_dataset", observed_load)

    def requests(study, order):
        return [
            study.model(
                family="linear",
                label="fwd_ret_1d",
                config_name="ridge",
                overrides={"alpha": alpha},
                execution_tier="preview",
                preview_reductions={
                    "folds": [0, 1],
                    "train_sample_frac": sample,
                },
            )
            for alpha, sample in order
        ]

    forward = run_models(
        first,
        requests=requests(first, [(1.0, 0.75), (2.0, 0.75), (3.0, 0.5)]),
    )
    forward_preparation = list(preparation)
    forward_load_count = load_count
    preparation.clear()
    load_count = 0
    # The second study has the same inputs, so it would otherwise be served the held fold sets -
    # correct, but it would measure nothing about the order the second run prepares them in.
    from case_studies.utils import folds as _folds_module

    _folds_module.clear_memo()
    linear.clear_input_memo()
    reverse = run_models(
        second,
        requests=requests(second, [(3.0, 0.5), (2.0, 0.75), (1.0, 0.75)]),
    )

    assert forward_preparation == [(0, 0.75), (1, 0.75), (0, 0.5), (1, 0.5)]
    assert forward_load_count == 1
    assert preparation == [(0, 0.5), (1, 0.5), (0, 0.75), (1, 0.75)]
    assert load_count == 1
    forward_by_alpha = {
        run.training.spec()["computation"]["model"]["effective_params_by_fold"]["0"]["alpha"]: run
        for run in forward.runs
    }
    reverse_by_alpha = {
        run.training.spec()["computation"]["model"]["effective_params_by_fold"]["0"]["alpha"]: run
        for run in reverse.runs
    }
    assert set(forward_by_alpha) == set(reverse_by_alpha) == {1.0, 2.0, 3.0}
    for alpha in forward_by_alpha:
        assert forward_by_alpha[alpha].training.hash == reverse_by_alpha[alpha].training.hash
        assert (
            forward_by_alpha[alpha]
            .predictions[0]
            .load()
            .equals(reverse_by_alpha[alpha].predictions[0].load())
        )


def test_linear_batch_failure_preserves_and_reuses_completed_candidate_folds(
    tmp_path, monkeypatch
) -> None:
    study = _linear_study(tmp_path, monkeypatch)
    original_fit = linear.Ridge.fit
    alpha_two_fits = 0

    def interrupted_fit(model, *args, **kwargs):
        nonlocal alpha_two_fits
        if model.alpha == 2.0:
            alpha_two_fits += 1
            if alpha_two_fits == 2:
                raise RuntimeError("injected candidate failure")
        return original_fit(model, *args, **kwargs)

    monkeypatch.setattr(linear.Ridge, "fit", interrupted_fit)
    requests = [
        study.model(
            family="linear",
            label="fwd_ret_1d",
            config_name="ridge",
            overrides={"alpha": alpha},
        )
        for alpha in (1.0, 2.0, 3.0)
    ]
    plan = plan_models(study, requests=requests)
    population = plan.create_population(
        name="planned-linear-recovery",
    )

    with pytest.raises(RuntimeError, match="injected candidate failure"):
        plan.run()

    completed = study.predictions.table().filter(pl.col("complete"))
    assert completed.height == 2
    with pytest.raises(ValueError, match="incomplete"):
        population.require_complete()
    monkeypatch.setattr(linear.Ridge, "fit", original_fit)
    recovered_plan = plan_models(study, requests=requests)
    assert recovered_plan.expected_prediction_hashes == plan.expected_prediction_hashes
    recovered = recovered_plan.run()
    recovered_by_alpha = {
        run.training.spec()["computation"]["model"]["effective_params_by_fold"]["0"]["alpha"]: run
        for run in recovered.runs
    }

    assert recovered_by_alpha[1.0].diagnostics["cache_hit"] is True
    assert recovered_by_alpha[3.0].diagnostics["cache_hit"] is True
    assert recovered_by_alpha[2.0].diagnostics["reused_folds"] == [0]
    assert recovered_by_alpha[2.0].diagnostics["fitted_folds"] == [1]
    # The interrupted run built these folds; recovery refits one of them without rebuilding it.
    assert {run.diagnostics["base_fold_preparations"] for run in recovered.runs} == {0}
    assert study.predictions.table().filter(pl.col("complete")).height == 3
    assert population.require_complete() == plan.expected_prediction_hashes


def test_linear_batch_rejects_a_modified_fitted_preprocessor(tmp_path, monkeypatch) -> None:
    study = _linear_study(tmp_path, monkeypatch)
    requests = [
        study.model(
            family="linear",
            label="fwd_ret_1d",
            config_name="ridge",
            overrides={"alpha": alpha},
        )
        for alpha in (1.0, 2.0)
    ]
    first = run_models(study, requests=requests)
    artifact = (
        first.runs[0].training.root
        / "run_log"
        / "training"
        / first.runs[0].training.hash
        / "models"
        / "fold_0.joblib"
    )
    payload = linear.joblib.load(artifact)
    payload["preprocessor"][-1].mean_[0] += 1.0
    linear.joblib.dump(payload, artifact)

    recovered = run_models(study, requests=requests)

    assert recovered.runs[0].diagnostics["reused_folds"] == [1]
    assert recovered.runs[0].diagnostics["fitted_folds"] == [0]
    assert recovered.runs[0].diagnostics["base_fold_preparations"] == 0
    assert recovered.runs[1].diagnostics["cache_hit"] is True


def test_linear_feature_order_and_random_state_are_identity_bearing(tmp_path, monkeypatch) -> None:
    study = _linear_study(tmp_path, monkeypatch)
    original_load = linear.load_modeling_dataset
    base = study.model(
        family="linear",
        label="fwd_ret_1d",
        config_name="ridge",
    ).resolve()
    changed_seed = study.model(
        family="linear",
        label="fwd_ret_1d",
        config_name="ridge",
        overrides={"random_state": 7},
    ).resolve()
    modeling_dataset = original_load("etfs", "fwd_ret_1d")
    reversed_lineage = dict(modeling_dataset.input_lineage)
    reversed_lineage["feature_names"] = ["x2", "x1"]
    reversed_features = SimpleNamespace(
        **{
            **vars(modeling_dataset),
            "feature_names": ["x2", "x1"],
            "input_lineage": reversed_lineage,
        }
    )
    monkeypatch.setattr(linear, "load_modeling_dataset", lambda *args, **kwargs: reversed_features)
    changed_order = study.model(
        family="linear",
        label="fwd_ret_1d",
        config_name="ridge",
    ).resolve()

    assert base.identity != changed_seed.identity
    assert base.identity != changed_order.identity


def test_linear_runner_replays_valid_models_after_registration_interrupt(
    tmp_path, monkeypatch
) -> None:
    study = _linear_study(tmp_path, monkeypatch)
    request = study.model(family="linear", label="fwd_ret_1d", config_name="ridge")
    original_publish = ResultsCatalog.publish_predictions
    attempted_predictions = None

    def interrupt_registration(*args, **kwargs):
        nonlocal attempted_predictions
        attempted_predictions = kwargs["predictions"].clone()
        raise RuntimeError("interrupted registration")

    monkeypatch.setattr(ResultsCatalog, "publish_predictions", interrupt_registration)
    with pytest.raises(RuntimeError, match="interrupted registration"):
        request.run()
    resolved = request.resolve()
    training_hash = linear.training_hash_from_spec(resolved.spec)
    model_dir = study.storage_root() / "run_log" / "training" / training_hash / "models"
    fitted_digests = {
        path.name: linear._sha256(path) for path in sorted(model_dir.glob("fold_*.joblib"))
    }
    monkeypatch.setattr(ResultsCatalog, "publish_predictions", original_publish)
    monkeypatch.setattr(
        linear.Ridge,
        "fit",
        lambda *args, **kwargs: pytest.fail("valid fitted state must not retrain"),
    )

    recovered = request.run()

    assert recovered.predictions[0].complete
    assert recovered.predictions[0].coverage()["n_expected"] == 12
    assert attempted_predictions is not None
    assert recovered.predictions[0].load().equals(attempted_predictions)
    assert fitted_digests == {
        path.name: linear._sha256(path) for path in sorted(model_dir.glob("fold_*.joblib"))
    }
    assert recovered.diagnostics == {
        "cache_hit": False,
        "reused_folds": [0, 1],
        "fitted_folds": [],
    }
    elapsed, resources = _recorded_runtime(study, recovered.training.hash)
    assert elapsed > 0
    assert resources["process_peak_rss_bytes"] > 0


def _recorded_runtime(study, training_hash: str) -> tuple[float, dict]:
    """What the registry says a training run cost.

    The measurement lives on the row rather than in the run's ``runtime.json``, which is compared
    byte for byte when the same identity is registered again and so cannot carry one.
    """
    import sqlite3

    with sqlite3.connect(study.storage_root() / "run_log" / "registry.db") as db:
        row = db.execute(
            "SELECT elapsed_s, runtime_json FROM training_runs WHERE training_hash = ?",
            (training_hash,),
        ).fetchone()
    assert row is not None, f"no training row for {training_hash}"
    return row[0], (json.loads(row[1]) if row[1] else {}).get("resources", {})


def test_linear_records_what_a_run_cost_against_its_registry_row(tmp_path, monkeypatch) -> None:
    """A run that does not record its own cost cannot be used to schedule the next one.

    Every row the current path produced carried a NULL ``elapsed_s`` while the value sat in the
    run's ``runtime.json``, where no query looks.
    """
    study = _linear_study(tmp_path, monkeypatch)
    run = study.model(family="linear", label="fwd_ret_1d", config_name="ridge").run()

    elapsed, resources = _recorded_runtime(study, run.training.hash)

    assert elapsed > 0
    assert resources["cpu_s"] > 0
    assert resources["cores_used"] > 0
    assert resources["process_peak_rss_bytes"] > 0


def test_a_failed_runtime_update_leaves_the_row_as_it_was(tmp_path, monkeypatch) -> None:
    """The measurement is written after the run, so a failure there must not lose the row."""
    from case_studies.utils.registry import registration

    study = _linear_study(tmp_path, monkeypatch)
    run = study.model(family="linear", label="fwd_ret_1d", config_name="ridge").run()
    before = _recorded_runtime(study, run.training.hash)

    # The failure has to land after the UPDATE. Patching canonical_json, as this used to, raises
    # while the argument tuple is still being built, so the row survived because nothing was ever
    # written and the run this describes never happened. Failing the commit is the real case: the
    # statement ran against the row, and its effect must not survive.
    class FailsToCommit:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def commit(self):
            raise RuntimeError("interrupted runtime update")

    real_open = registration._open_registry
    monkeypatch.setattr(
        registration, "_open_registry", lambda case_dir: FailsToCommit(real_open(case_dir))
    )
    with pytest.raises(RuntimeError, match="interrupted runtime update"):
        registration.record_training_runtime(
            study.case_study,
            run.training.hash,
            case_dir=study.storage_root(),
            measured={"elapsed_s": 999.0},
        )

    monkeypatch.setattr(registration, "_open_registry", real_open)
    assert _recorded_runtime(study, run.training.hash) == before


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


@pytest.mark.parametrize("family", ["linear", "gbm", "latent_factors"])
def test_model_adapters_reject_custom_cv_for_fold_scoped_temporal_features(
    family,
    tmp_path,
    monkeypatch,
) -> None:
    cv = CVSpec.walk_forward(
        training_window="2D",
        validation_window="1D",
        folds=(0,),
    )
    if family == "linear":
        study = _linear_study(tmp_path / family, monkeypatch)
        context = linear.load_modeling_dataset("etfs", "fwd_ret_1d")
        selector = "case_studies.utils.linear._select_splits"
        config_name = "ridge"
    elif family == "gbm":
        study = _gbm_study(tmp_path / family, monkeypatch)
        context = modeling.load_modeling_dataset("etfs", "fwd_ret_1d")
        selector = "case_studies.utils.gbm._gbm_select_splits"
        config_name = "leaves_7_mse"
    else:
        study = _latent_study(tmp_path / family, monkeypatch)
        context = latent_case_study.load_case_study_context("etfs")
        selector = "case_studies.utils.latent_factors.adapter._select_splits"
        config_name = "pca"
    context.temporal_by_fold = pl.DataFrame({"fold": [0]}).to_pandas()
    context.temporal_keys = ["symbol", "timestamp"]
    context.temporal_feature_names = ["temporal_value"]
    context.temporal_artifact_splits = [dict(context.splits[0])]
    variant_splits = [{**context.splits[0], "train_end": "2024-01-01"}]
    context.splits = variant_splits
    monkeypatch.setattr(
        selector, lambda *_args, **_kwargs: (variant_splits, {"identity": "variant-buffer"})
    )

    with pytest.raises(ValueError, match="incompatible with fold-scoped temporal features"):
        study.model(
            family=family,
            label="fwd_ret_1d",
            config_name=config_name,
            cv=cv,
            overrides={"device": "cpu", "max_bin": 63} if family == "gbm" else {},
        ).resolve()


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

    assert resolved.spec["identity_version"] == 3
    assert resolved.spec["resolved_spec_schema"] == "ml4t.resolved-spec/v1"
    assert [item["value"] for item in resolved.spec["computation"]["checkpoint_schedule"]] == [2, 4]
    assert len(fresh.predictions) == len(cached.predictions) == 2
    assert [result.hash for result in fresh.predictions] == [
        result.hash for result in cached.predictions
    ]
    assert all(result.complete for result in fresh.predictions)
    assert all(result.coverage()["n_expected"] == 12 for result in fresh.predictions)
    assert set(study.predictions.table()["identity_status"]) == {"current"}
    model_dir = fresh.training.root / "run_log" / "training" / fresh.training.hash / "models"
    assert sorted(path.name for path in (model_dir / "boosters").glob("*.txt")) == [
        "fold_0.txt",
        "fold_1.txt",
    ]


def test_gbm_batch_is_fold_major_and_matches_individual_execution(tmp_path, monkeypatch) -> None:
    # A fold set above the memo budget is not held, and that is the case this guards: on
    # us_equities_panel one set is 90 GB, so the batch path must release each fold as it takes the
    # next. Below the budget the set is deliberately held and shared, which no large panel reaches.
    monkeypatch.setenv("ML4T_FOLD_MEMO_BUDGET_BYTES", "1")
    from case_studies.utils import folds as fold_utils

    fold_utils.clear_memo()
    study = _gbm_study(tmp_path / "batch", monkeypatch)
    individual_study = _gbm_study(tmp_path / "individual", monkeypatch)
    original_prepare = gbm_utils.prepare_gbm_folds_from_mds
    prepared: list[list[int]] = []
    released_arrays: list[weakref.ReferenceType[np.ndarray]] = []

    def observed_prepare(mds, splits, *args, **kwargs):
        gc.collect()
        if released_arrays:
            assert released_arrays[-1]() is None
        prepared.append([int(split["fold"]) for split in splits])
        folds = original_prepare(mds, splits, *args, **kwargs)
        released_arrays.append(weakref.ref(folds[0]["X_train"]))
        return folds

    monkeypatch.setattr(gbm_utils, "prepare_gbm_folds_from_mds", observed_prepare)
    requests = [
        study.model(
            family="gbm",
            label="fwd_ret_1d",
            config_name="leaves_7_mse",
            overrides={"device": "cpu", "max_bin": 63, "learning_rate": learning_rate},
        )
        for learning_rate in (0.1, 0.2)
    ]

    plan = plan_models(study, requests=requests)
    population = plan.create_population(
        name="planned-gbm-checkpoints",
    )
    with pytest.raises(ValueError, match="incomplete"):
        population.require_complete()
    batch = plan.run()
    batch_preparations = list(prepared)
    monkeypatch.setattr(gbm_utils, "prepare_gbm_folds_from_mds", original_prepare)
    individual = (
        individual_study.model(
            family="gbm",
            label="fwd_ret_1d",
            config_name="leaves_7_mse",
            overrides={"device": "cpu", "max_bin": 63, "learning_rate": 0.1},
        )
        .resolve()
        .run()
    )
    gc.collect()

    assert batch_preparations == [[0], [1]]
    assert released_arrays[-1]() is None
    assert batch.runs[0].training.hash == individual.training.hash
    assert [prediction.hash for prediction in batch.runs[0].predictions] == [
        prediction.hash for prediction in individual.predictions
    ]
    for batch_prediction, individual_prediction in zip(
        batch.runs[0].predictions,
        individual.predictions,
        strict=True,
    ):
        assert batch_prediction.load().equals(individual_prediction.load())
    assert all(run.diagnostics["execution_order"] == "fold_major" for run in batch.runs)
    assert all(run.diagnostics["compatibility_group_size"] == 2 for run in batch.runs)
    assert all(run.diagnostics["base_fold_preparations"] == 2 for run in batch.runs)
    assert all(run.diagnostics["disk_fold_cache"] is False for run in batch.runs)
    assert population.require_complete() == plan.expected_prediction_hashes


def test_gbm_batch_resolves_fold_dependent_huber_parameters(tmp_path, monkeypatch) -> None:
    study = _gbm_study(tmp_path, monkeypatch)
    original_prepare = gbm_utils.prepare_gbm_folds_from_mds
    prepared: list[int] = []
    prepared_label_std: dict[str, float] = {}

    def configs(*_args, **_kwargs):
        common = {
            "checkpoint_interval": 2,
            "family": "gbm",
            "library": "lightgbm",
            "max_iterations": 4,
        }
        return [
            {
                **common,
                "config_name": "mse",
                "params": {"learning_rate": 0.1, "objective": "regression"},
            },
            {
                **common,
                "config_name": "huber",
                "huber_alpha_scale": 0.5,
                "params": {"learning_rate": 0.1, "objective": "huber"},
            },
        ]

    def observed_prepare(*args, **kwargs):
        folds = original_prepare(*args, **kwargs)
        prepared.extend(int(fold["fold"]) for fold in folds)
        prepared_label_std.update(
            {str(int(fold["fold"])): float(np.std(fold["y_train"])) for fold in folds}
        )
        return folds

    monkeypatch.setattr(modeling, "load_configs", configs)
    monkeypatch.setattr(gbm_utils, "prepare_gbm_folds_from_mds", observed_prepare)

    batch = run_models(
        study,
        requests=[
            study.model(
                family="gbm",
                label="fwd_ret_1d",
                config_name=config_name,
                overrides={"device": "cpu", "max_bin": 63},
            )
            for config_name in ("mse", "huber")
        ],
    )

    assert prepared == [0, 1]
    params = batch.runs[1].training.spec()["computation"]["model"]["effective_params_by_fold"]
    assert params["0"]["alpha"] == pytest.approx(0.5 * prepared_label_std["0"])
    assert params["1"]["alpha"] == pytest.approx(0.5 * prepared_label_std["1"])
    assert all(run.diagnostics["base_fold_preparations"] == 2 for run in batch.runs)


def test_gbm_batch_separates_sampling_and_is_order_invariant(tmp_path, monkeypatch) -> None:
    first = _gbm_study(tmp_path / "first", monkeypatch)
    second = _gbm_study(tmp_path / "second", monkeypatch)
    original_prepare = gbm_utils.prepare_gbm_folds_from_mds
    original_load = modeling.load_modeling_dataset
    preparation: list[tuple[int, float]] = []
    load_count = 0

    def observed_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return original_load(*args, **kwargs)

    def observed_prepare(*args, **kwargs):
        folds = original_prepare(*args, **kwargs)
        preparation.extend(
            (int(fold["fold"]), float(kwargs["train_sample_frac"])) for fold in folds
        )
        return folds

    monkeypatch.setattr(modeling, "load_modeling_dataset", observed_load)
    monkeypatch.setattr(gbm_utils, "prepare_gbm_folds_from_mds", observed_prepare)

    def requests(study, order):
        return [
            study.model(
                family="gbm",
                label="fwd_ret_1d",
                config_name="leaves_7_mse",
                overrides={
                    "device": "cpu",
                    "learning_rate": learning_rate,
                    "max_bin": 63,
                },
                execution_tier="preview",
                preview_reductions={
                    "folds": [0, 1],
                    "train_sample_frac": sample,
                },
            )
            for learning_rate, sample in order
        ]

    forward = run_models(
        first,
        requests=requests(first, [(0.1, 0.75), (0.2, 0.75), (0.3, 0.5)]),
    )
    forward_preparation = list(preparation)
    forward_load_count = load_count
    preparation.clear()
    load_count = 0
    reverse = run_models(
        second,
        requests=requests(second, [(0.3, 0.5), (0.2, 0.75), (0.1, 0.75)]),
    )

    assert forward_preparation == [(0, 0.75), (1, 0.75), (0, 0.5), (1, 0.5)]
    assert forward_load_count == 1
    assert preparation == [(0, 0.5), (1, 0.5), (0, 0.75), (1, 0.75)]
    assert load_count == 1
    forward_by_rate = {
        run.training.spec()["computation"]["model"]["effective_params_by_fold"]["0"][
            "learning_rate"
        ]: run
        for run in forward.runs
    }
    reverse_by_rate = {
        run.training.spec()["computation"]["model"]["effective_params_by_fold"]["0"][
            "learning_rate"
        ]: run
        for run in reverse.runs
    }
    assert set(forward_by_rate) == set(reverse_by_rate) == {0.1, 0.2, 0.3}
    for learning_rate in forward_by_rate:
        assert (
            forward_by_rate[learning_rate].training.hash
            == reverse_by_rate[learning_rate].training.hash
        )
        for forward_prediction, reverse_prediction in zip(
            forward_by_rate[learning_rate].predictions,
            reverse_by_rate[learning_rate].predictions,
            strict=True,
        ):
            assert forward_prediction.load().equals(reverse_prediction.load())


def test_gbm_batch_failure_preserves_siblings_and_completed_folds(tmp_path, monkeypatch) -> None:
    study = _gbm_study(tmp_path, monkeypatch)
    original_train = gbm_utils.train_gbm_config
    rate_two_fits = 0

    def interrupted_train(*args, **kwargs):
        nonlocal rate_two_fits
        params = next(iter(kwargs["effective_params_by_fold"].values()))
        if params["learning_rate"] == 0.2:
            rate_two_fits += 1
            if rate_two_fits == 2:
                raise RuntimeError("injected GBM candidate failure")
        return original_train(*args, **kwargs)

    monkeypatch.setattr(gbm_utils, "train_gbm_config", interrupted_train)
    requests = [
        study.model(
            family="gbm",
            label="fwd_ret_1d",
            config_name="leaves_7_mse",
            overrides={"device": "cpu", "max_bin": 63, "learning_rate": rate},
        )
        for rate in (0.1, 0.2, 0.3)
    ]
    plan = plan_models(study, requests=requests)
    population = plan.create_population(
        name="planned-gbm-recovery",
    )

    with pytest.raises(RuntimeError, match="injected GBM candidate failure"):
        plan.run()

    assert study.predictions.table().filter(pl.col("complete")).height == 4
    with pytest.raises(ValueError, match="incomplete"):
        population.require_complete()
    monkeypatch.setattr(gbm_utils, "train_gbm_config", original_train)
    recovered_plan = plan_models(study, requests=requests)
    assert recovered_plan.expected_prediction_hashes == plan.expected_prediction_hashes
    recovered = recovered_plan.run()
    recovered_by_rate = {
        run.training.spec()["computation"]["model"]["effective_params_by_fold"]["0"][
            "learning_rate"
        ]: run
        for run in recovered.runs
    }

    assert recovered_by_rate[0.1].diagnostics["cache_hit"] is True
    assert recovered_by_rate[0.3].diagnostics["cache_hit"] is True
    assert recovered_by_rate[0.2].diagnostics["reused_folds"] == [0]
    assert recovered_by_rate[0.2].diagnostics["fitted_folds"] == [1]
    assert {run.diagnostics["base_fold_preparations"] for run in recovered.runs} == {1}
    assert study.predictions.table().filter(pl.col("complete")).height == 6
    assert population.require_complete() == plan.expected_prediction_hashes


def test_gbm_batch_rejects_a_modified_booster(tmp_path, monkeypatch) -> None:
    study = _gbm_study(tmp_path, monkeypatch)
    requests = [
        study.model(
            family="gbm",
            label="fwd_ret_1d",
            config_name="leaves_7_mse",
            overrides={"device": "cpu", "max_bin": 63, "learning_rate": rate},
        )
        for rate in (0.1, 0.2)
    ]
    first = run_models(study, requests=requests)
    artifact = (
        first.runs[0].training.root
        / "run_log"
        / "training"
        / first.runs[0].training.hash
        / "models"
        / "boosters"
        / "fold_0.txt"
    )
    artifact.write_text("modified\n")

    recovered = run_models(study, requests=requests)

    assert recovered.runs[0].diagnostics["reused_folds"] == [1]
    assert recovered.runs[0].diagnostics["fitted_folds"] == [0]
    # Gradient boosting still prepares its own folds; it moves onto the shared preparation
    # before its rebuild starts, and this count drops to zero then.
    assert recovered.runs[0].diagnostics["base_fold_preparations"] == 1
    assert recovered.runs[1].diagnostics["cache_hit"] is True


def test_gbm_request_uses_setup_execution_config_then_request_overrides(
    tmp_path, monkeypatch
) -> None:
    study = _gbm_study(tmp_path, monkeypatch)
    setup_path = study.root / "config" / "setup.yaml"
    setup = yaml.safe_load(setup_path.read_text())
    setup["modeling"] = {"gbm": {"device": "cpu", "max_bin": 127, "num_threads": 3}}
    setup_path.write_text(yaml.safe_dump(setup, sort_keys=False))

    configured = study.model(family="gbm", label="fwd_ret_1d", config_name="leaves_7_mse").resolve()
    overridden = study.model(
        family="gbm",
        label="fwd_ret_1d",
        config_name="leaves_7_mse",
        overrides={"max_bin": 63, "num_threads": 1},
    ).resolve()

    configured_params = configured.spec["computation"]["model"]["effective_params_by_fold"]["0"]
    overridden_params = overridden.spec["computation"]["model"]["effective_params_by_fold"]["0"]
    assert (configured_params["max_bin"], configured_params["num_threads"]) == (127, 3)
    assert (overridden_params["max_bin"], overridden_params["num_threads"]) == (63, 1)


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

    class ReplayBooster:
        def __init__(self, *, model_file):
            self.model_file = model_file

        def predict(self, values, *, num_iteration):
            return values[:, 0] + num_iteration / 100

    monkeypatch.setattr("lightgbm.Booster", ReplayBooster)
    monkeypatch.setattr(
        gbm_utils,
        "train_gbm_config",
        lambda *args, **kwargs: pytest.fail("valid fitted state must not retrain"),
    )

    recovered = request.run()

    assert len(recovered.predictions) == 2
    assert all(result.complete for result in recovered.predictions)
    curves = (
        recovered.training.root
        / "run_log"
        / "training"
        / recovered.training.hash
        / "learning_curves.parquet"
    )
    assert pl.read_parquet(curves).get_column("iteration").to_list() == [2, 4]


def test_gbm_runner_finalizes_curves_after_prediction_registration_interrupt(
    tmp_path, monkeypatch
) -> None:
    study = _gbm_study(tmp_path, monkeypatch)
    request = study.model(
        family="gbm",
        label="fwd_ret_1d",
        config_name="leaves_7_mse",
        overrides={"device": "cpu", "max_bin": 63},
    )
    original_write = gbm_utils._write_learning_curves

    def interrupt_curve_write(*args, **kwargs):
        raise RuntimeError("interrupted curve finalization")

    monkeypatch.setattr(gbm_utils, "_write_learning_curves", interrupt_curve_write)
    with pytest.raises(RuntimeError, match="interrupted curve finalization"):
        request.run()
    monkeypatch.setattr(gbm_utils, "_write_learning_curves", original_write)

    class ReplayBooster:
        def __init__(self, *, model_file):
            self.model_file = model_file

        def predict(self, values, *, num_iteration):
            return values[:, 0] + num_iteration / 100

    monkeypatch.setattr("lightgbm.Booster", ReplayBooster)
    monkeypatch.setattr(
        gbm_utils,
        "train_gbm_config",
        lambda *args, **kwargs: pytest.fail("valid fitted state must not retrain"),
    )

    recovered = request.run()
    curves_path = (
        recovered.training.root
        / "run_log"
        / "training"
        / recovered.training.hash
        / "learning_curves.parquet"
    )

    assert [result.complete for result in recovered.predictions] == [True, True]
    assert gbm_utils._valid_learning_curves(curves_path, request.resolve().spec)


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


def test_lightgbm_huber_alpha_is_a_residual_unit_threshold() -> None:
    import lightgbm as lgb

    rng = np.random.default_rng(7)
    features = rng.normal(size=(200, 3))
    labels = 0.02 * features[:, 0] + 0.01 * features[:, 1]
    labels += rng.normal(scale=0.01, size=200)
    labels[::20] += 0.15
    common = {
        "deterministic": True,
        "force_col_wise": True,
        "metric": "None",
        "min_data_in_leaf": 5,
        "num_leaves": 7,
        "num_threads": 1,
        "seed": 42,
        "verbosity": -1,
    }
    data = lgb.Dataset(features, label=labels)
    mse = lgb.train({**common, "objective": "regression"}, data, num_boost_round=20)
    alpha = gbm_utils._scaled_huber_alpha(0.5, labels)
    huber = lgb.train(
        {**common, "alpha": alpha, "objective": "huber"},
        data,
        num_boost_round=20,
    )
    label_scale = 10.0
    scaled_huber = lgb.train(
        {**common, "alpha": label_scale * alpha, "objective": "huber"},
        lgb.Dataset(features, label=label_scale * labels),
        num_boost_round=20,
    )

    mse_predictions = np.asarray(mse.predict(features), dtype=np.float64)
    huber_predictions = np.asarray(huber.predict(features), dtype=np.float64)
    scaled_predictions = np.asarray(scaled_huber.predict(features), dtype=np.float64)
    assert alpha == pytest.approx(0.5 * np.std(labels))
    assert np.max(np.abs(mse_predictions - huber_predictions)) > 0.01
    np.testing.assert_allclose(
        scaled_predictions,
        label_scale * huber_predictions,
        rtol=1e-5,
        atol=1e-7,
    )


def test_real_huber_preset_keeps_legacy_training_surface(monkeypatch) -> None:
    config = next(
        config
        for config in modeling.load_configs("etfs", "fwd_ret_21d", "gbm")
        if config["config_name"] == "default_huber"
    )
    captured = []

    class Booster:
        def predict(self, values, *, num_iteration):
            del num_iteration
            return values[:, 0]

        def feature_importance(self, *, importance_type):
            del importance_type
            return np.array([1.0])

    def fake_train(params, *_args, **_kwargs):
        captured.append(dict(params))
        return Booster()

    monkeypatch.setattr("lightgbm.train", fake_train)
    folds = []
    for fold, scale in enumerate((0.01, 0.10)):
        values = np.arange(6, dtype=float) * scale
        folds.append(
            {
                "fold": fold,
                "X_train": values[:, None],
                "X_val": values[:, None],
                "y_train": values,
                "y_train_lgb": values,
                "y_val": values,
                "y_eval": None,
                "dates": np.array([f"2024-01-0{fold + 1}"] * 6, dtype="datetime64[D]"),
                "entities": np.array([f"S{index}" for index in range(6)]),
                "n_train": 6,
                "n_val": 6,
            }
        )

    result = gbm_utils.train_gbm_config(
        config,
        folds,
        feature_names=["value"],
        device="cpu",
        max_bin=63,
        num_threads=1,
    )

    effective = result["effective_params_by_fold"]
    assert [params["alpha"] for params in captured] == [
        effective["0"]["alpha"],
        effective["1"]["alpha"],
    ]
    assert effective["0"]["alpha"] != effective["1"]["alpha"]


def test_multiclass_effective_params_use_resolved_class_count() -> None:
    folds = [{"fold": 0, "y_train": np.array([0, 1, 2])}]
    config = {
        "params": {
            "num_class": 5,
            "objective": "multiclass",
            "seed": 42,
        }
    }

    effective = gbm_utils._gbm_effective_params_by_fold(
        config,
        folds,
        device="cpu",
        max_bin=63,
        num_threads=1,
        seed=42,
        task_type="classification",
        class_values=[-1, 0, 1],
    )

    assert effective["0"]["num_class"] == 3


@pytest.mark.parametrize("model_name", ["pca", "ipca"])
def test_real_pca_presets_resolve_checkpoint_metadata(model_name) -> None:
    presets = latent_case_study._load_preset_model_kwargs("etfs", "fwd_ret_21d")
    case = cast(LatentFactorCaseStudyContext, SimpleNamespace(model_kwargs=presets))

    model_kwargs, _, _, _, checkpoint_metadata = latent_adapter._resolve_model_configuration(
        case,
        model_name,
        {},
        {},
    )

    assert checkpoint_metadata == {"checkpoint_interval": 0}
    assert "checkpoint_interval" not in model_kwargs


def test_legacy_latent_surface_excludes_internal_checkpoint_aliases() -> None:
    presets = latent_case_study._load_preset_model_kwargs(
        "us_firm_characteristics",
        "fwd_ret_1m",
    )

    assert len(_expected_latent_checkpoints("cae", n_epochs=50, model_kwargs=presets["cae"])) == 10
    assert len(_expected_latent_checkpoints("sdf", n_epochs=0, model_kwargs=presets["sdf"])) == 5
    assert 0 in _expected_latent_checkpoints(
        "cae",
        n_epochs=50,
        model_kwargs=presets["cae"],
        include_internal_aliases=True,
    )


def test_sdf_identity_hashes_resolved_fallback_macro_values(tmp_path, monkeypatch) -> None:
    _latent_study(tmp_path, monkeypatch)
    context = latent_case_study.load_case_study_context("etfs")
    dates = context.dataset.get_column("timestamp").unique().sort()
    context.macro_panel = pl.DataFrame(
        {
            "timestamp": dates,
            "market_state": np.linspace(0.0, 1.0, len(dates)),
        }
    )
    context.macro_context_spec = {
        "policy": "load_macro_fallback",
        "version": "v1",
    }

    first = latent_adapter._resolved_macro_digest(context)
    context.macro_panel = context.macro_panel.with_columns(pl.col("market_state") + 0.01)
    second = latent_adapter._resolved_macro_digest(context)

    assert first != second


def test_macro_disabled_sdf_request_retains_disabled_identity(tmp_path, monkeypatch) -> None:
    study = _latent_study(tmp_path, monkeypatch)
    context = latent_case_study.load_case_study_context("etfs")
    context.macro_context_spec = {
        "alignment": "none",
        "availability_lag_days": 0,
        "input_digest": None,
        "policy": "disabled",
        "series": [],
        "version": "v1",
    }
    context.model_kwargs["sdf"] = {
        "beta_checkpoint_epochs": [1],
        "beta_default_checkpoint": 1,
        "beta_n_epochs": 1,
        "checkpoint_epochs": [1],
        "n_epochs_cond": 1,
        "n_epochs_moment": 1,
        "n_epochs_unc": 1,
    }

    resolved = study.model(
        family="latent_factors",
        label="fwd_ret_1d",
        config_name="sdf",
        overrides={"device": "cpu", "use_macro": False},
    ).resolve()

    assert resolved.spec["computation"]["macro_context"] == context.macro_context_spec


def test_latent_numerical_runtime_changes_training_identity(tmp_path, monkeypatch) -> None:
    study = _latent_study(tmp_path, monkeypatch)
    requests = [
        {"deterministic_algorithms": True, "device": "cpu", "num_threads": 1},
        {"deterministic_algorithms": True, "device": "cuda", "num_threads": 1},
        {"deterministic_algorithms": True, "device": "cpu", "num_threads": 2},
        {"deterministic_algorithms": False, "device": "cpu", "num_threads": 1},
    ]

    identities = {
        study.model(
            family="latent_factors",
            label="fwd_ret_1d",
            config_name="pca",
            overrides=runtime,
        )
        .resolve()
        .identity
        for runtime in requests
    }

    assert len(identities) == len(requests)


def test_real_sdf_preset_resolves_reduced_preview_schedule(tmp_path, monkeypatch) -> None:
    presets = latent_case_study._load_preset_model_kwargs("etfs", "fwd_ret_21d")
    study = _latent_study(tmp_path, monkeypatch)
    context = latent_case_study.load_case_study_context("etfs")
    context.model_kwargs["sdf"] = presets["sdf"]
    context.macro_context_spec = {
        "alignment": "none",
        "availability_lag_days": 0,
        "input_digest": None,
        "policy": "disabled",
        "series": [],
        "version": "v1",
    }

    resolved = study.model(
        family="latent_factors",
        label="fwd_ret_1d",
        config_name="sdf",
        overrides={"device": "cpu", "use_macro": False},
        execution_tier="preview",
        preview_reductions={
            "n_epochs_cond": 1,
            "n_epochs_moment": 1,
            "n_epochs_unc": 1,
        },
    ).resolve()

    assert resolved.spec["computation"]["model"]["params"]["checkpoint_epochs"] == [1]
    assert [item["value"] for item in resolved.spec["computation"]["checkpoint_schedule"]] == [
        1,
        2,
    ]


@pytest.mark.parametrize(
    ("model_name", "reduction"),
    [
        ("pca", {"n_epochs": 1}),
        ("sae", {"n_factors": 2}),
        ("sdf", {"n_epochs": 1}),
        ("sdf", {"n_factors": 2}),
    ],
)
def test_latent_preview_rejects_model_specific_noop_reductions(
    tmp_path, monkeypatch, model_name, reduction
) -> None:
    study = _latent_study(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match=f"unsupported {model_name} preview reductions"):
        study.model(
            family="latent_factors",
            label="fwd_ret_1d",
            config_name=model_name,
            overrides={"device": "cpu", "use_macro": False},
            execution_tier="preview",
            preview_reductions={"folds": [0], **reduction},
        ).resolve()


def test_sdf_rejects_unimplemented_expected_return_mapper() -> None:
    case = cast(
        LatentFactorCaseStudyContext,
        SimpleNamespace(
            model_kwargs={
                "sdf": {
                    "expected_return_mapper": "neural",
                    "output_mode": "expected_returns",
                }
            }
        ),
    )

    with pytest.raises(ValueError, match="supports only 'linear'"):
        latent_adapter._resolve_model_configuration(case, "sdf", {}, {})


def test_cae_fitted_artifact_reconstructs_every_checkpoint(tmp_path) -> None:
    rng = np.random.default_rng(42)
    chars_train = rng.normal(size=(4, 5, 3)).astype(np.float32)
    chars_val = rng.normal(size=(2, 5, 3)).astype(np.float32)
    returns_train = rng.normal(size=(4, 5)).astype(np.float32)
    returns_val = rng.normal(size=(2, 5)).astype(np.float32)
    artifact_dir = tmp_path / "cae"

    fresh, _ = run_cae_fold(
        chars_train,
        returns_train,
        chars_val,
        returns_val,
        n_factors=2,
        n_epochs=2,
        checkpoint_interval=1,
        batch_size=20,
        device="cpu",
        artifact_dir=artifact_dir,
    )
    loaded = predict_latent_fold_from_artifact(
        "cae",
        artifact_dir=artifact_dir,
        chars_train=chars_train,
        returns_train=returns_train,
        chars_val=chars_val,
        returns_val=returns_val,
        device="cpu",
    )

    assert sorted(fresh) == sorted(loaded) == [0, 1, 2]
    assert all(np.array_equal(fresh[epoch], loaded[epoch]) for epoch in fresh)
    assert sorted(path.name for path in artifact_dir.glob("*.ml4t")) == [
        "forecaster_0.ml4t",
        "forecaster_1.ml4t",
        "forecaster_2.ml4t",
        "model.ml4t",
    ]


def test_sdf_fitted_heads_reconstruct_every_checkpoint(tmp_path) -> None:
    rng = np.random.default_rng(42)
    chars_train = rng.normal(size=(10, 5, 3)).astype(np.float32)
    chars_val = rng.normal(size=(3, 5, 3)).astype(np.float32)
    returns_train = rng.normal(scale=0.05, size=(10, 5)).astype(np.float32)
    returns_val = rng.normal(scale=0.05, size=(3, 5)).astype(np.float32)
    macro_train = rng.normal(size=(10, 2)).astype(np.float32)
    macro_val = rng.normal(size=(3, 2)).astype(np.float32)
    artifact_dir = tmp_path / "sdf"
    kwargs: dict[str, Any] = {
        "state_dim_sdf": 2,
        "state_dim_moment": 2,
        "hidden_dim": 4,
        "n_instruments": 2,
        "n_epochs_unc": 1,
        "n_epochs_moment": 1,
        "n_epochs_cond": 1,
        "checkpoint_epochs": [1],
        "beta_n_epochs": 1,
        "beta_checkpoint_epochs": [1],
        "beta_default_checkpoint": 1,
        "output_mode": "beta_network",
        "device": "cpu",
    }

    fresh, _ = run_sdf_fold(
        chars_train,
        returns_train,
        chars_val,
        returns_val,
        macro_train=macro_train,
        macro_val=macro_val,
        artifact_dir=artifact_dir,
        **kwargs,
    )
    loaded = predict_latent_fold_from_artifact(
        "sdf",
        artifact_dir=artifact_dir,
        chars_train=chars_train,
        returns_train=returns_train,
        chars_val=chars_val,
        returns_val=returns_val,
        macro_train=macro_train,
        macro_val=macro_val,
        output_mode="beta_network",
        device="cpu",
    )

    expected = _expected_latent_checkpoints(
        "sdf",
        n_epochs=0,
        model_kwargs=kwargs,
        include_internal_aliases=True,
    )
    assert sorted(fresh) == sorted(loaded) == list(expected) == [-3, -2, -1, 0, 1, 2]
    assert all(np.array_equal(fresh[checkpoint], loaded[checkpoint]) for checkpoint in fresh)


@pytest.mark.parametrize("model_name", ["ipca", "sae"])
def test_other_latent_fitted_artifacts_reconstruct_predictions(tmp_path, model_name) -> None:
    rng = np.random.default_rng(42)
    chars_train = rng.normal(size=(12, 5, 3)).astype(np.float32)
    chars_val = rng.normal(size=(3, 5, 3)).astype(np.float32)
    returns_train = rng.normal(scale=0.05, size=(12, 5)).astype(np.float32)
    returns_val = rng.normal(scale=0.05, size=(3, 5)).astype(np.float32)
    artifact_dir = tmp_path / model_name
    if model_name == "ipca":
        predictions, _ = run_ipca_fold(
            chars_train,
            returns_train,
            chars_val,
            returns_val,
            n_factors=2,
            max_iter=20,
            tol=1.0,
            artifact_dir=artifact_dir,
        )
        fresh = {0: predictions}
    else:
        fresh, _ = run_sae_fold(
            chars_train,
            returns_train,
            chars_val,
            returns_val,
            n_factors=2,
            n_epochs=2,
            checkpoint_interval=1,
            main_hidden_units=[4, 4, 4, 4],
            aux_hidden_dim=4,
            bottleneck_dim=2,
            device="cpu",
            artifact_dir=artifact_dir,
        )
    loaded = predict_latent_fold_from_artifact(
        model_name,
        artifact_dir=artifact_dir,
        chars_train=chars_train,
        returns_train=returns_train,
        chars_val=chars_val,
        returns_val=returns_val,
        device="cpu",
    )

    assert sorted(fresh) == sorted(loaded)
    assert all(np.array_equal(fresh[checkpoint], loaded[checkpoint]) for checkpoint in fresh)


def _latent_study(tmp_path, monkeypatch):
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    dates = [f"2024-01-{day:02d}" for day in range(1, 17)]
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
                    "fwd_ret_1d": 0.03 * x1 + 0.002 * date_index * x1 + 0.01 * x2,
                }
            )
    frame = pl.DataFrame(rows).with_columns(pl.col("timestamp").str.to_date())
    study.labels.publish(
        LabelDefinition("fwd_ret_1d", "regression", "1D"),
        frame.select("symbol", "timestamp", "fwd_ret_1d"),
    )
    splits = [
        {
            "fold": 0,
            "train_start": "2024-01-01",
            "train_end": "2024-01-10",
            "val_start": "2024-01-11",
            "val_end": "2024-01-13",
        },
        {
            "fold": 1,
            "train_start": "2024-01-01",
            "train_end": "2024-01-13",
            "val_start": "2024-01-14",
            "val_end": "2024-01-16",
        },
    ]
    context = LatentFactorCaseStudyContext(
        case_study_id="etfs",
        case_dir=study.root,
        setup={},
        primary_label="fwd_ret_1d",
        variant_labels=[],
        model_kwargs={"pca": {"n_factors": 1}},
        setup_model_kwargs={},
        persistent_entities=True,
        macro_panel=None,
        macro_context_spec=None,
        input_data_spec={
            "version": "v1",
            "files": [
                {"role": "financial", "sha256": "sha256:features-v1"},
                {"role": "label", "sha256": "sha256:label-v1"},
            ],
            "input_digest": "sha256:fixture-v1",
        },
        dataset=frame,
        feature_names=["x1", "x2"],
        task_type="regression",
        class_values=[],
        eval_label_col=None,
        date_col="timestamp",
        entity_col="symbol",
        splits=splits,
        temporal_by_fold=None,
        temporal_keys=[],
        temporal_feature_names=[],
        temporal_artifact_splits=[],
        device="cpu",
        num_threads=1,
    )
    monkeypatch.setattr(
        "case_studies.utils.latent_factors.case_study.load_case_study_context",
        lambda *args, **kwargs: context,
    )
    monkeypatch.setattr(latent_adapter, "_source_identity", lambda: {"fixture": "v1"})
    return study


def test_latent_runner_persists_and_reconstructs_fitted_state(tmp_path, monkeypatch) -> None:
    study = _latent_study(tmp_path, monkeypatch)
    request = study.model(
        family="latent_factors",
        label="fwd_ret_1d",
        config_name="pca",
        overrides={"device": "cpu", "n_factors": 1},
    )

    resolved = request.resolve()
    fresh = resolved.run()
    cached = request.run()

    assert resolved.spec["identity_version"] == 3
    assert resolved.spec["resolved_spec_schema"] == "ml4t.resolved-spec/v1"
    assert resolved.spec["computation"]["checkpoint_schedule"] == [{"kind": "epoch", "value": 0}]
    assert fresh.training.hash == cached.training.hash
    assert fresh.predictions[0].hash == cached.predictions[0].hash
    assert fresh.predictions[0].complete
    assert fresh.predictions[0].coverage()["n_expected"] == 36
    assert set(study.predictions.table()["identity_status"]) == {"current"}
    model_dir = fresh.training.root / "run_log" / "training" / fresh.training.hash / "models"
    assert sorted(model_dir.glob("pca/artifacts/fold_*/model.ml4t")) == [
        model_dir / "pca" / "artifacts" / "fold_0" / "model.ml4t",
        model_dir / "pca" / "artifacts" / "fold_1" / "model.ml4t",
    ]
    extras = load_fold_extras("etfs", fresh.training.hash)
    assert extras is not None
    assert [int(row["fold_id"]) for row in extras] == [0, 1]


def test_latent_runner_reuses_fitted_state_after_registration_interrupt(
    tmp_path, monkeypatch
) -> None:
    study = _latent_study(tmp_path, monkeypatch)
    request = study.model(
        family="latent_factors",
        label="fwd_ret_1d",
        config_name="pca",
        overrides={"device": "cpu", "n_factors": 1},
    )
    original_publish = ResultsCatalog.publish_predictions

    def interrupt_registration(*args, **kwargs):
        raise RuntimeError("interrupted registration")

    monkeypatch.setattr(ResultsCatalog, "publish_predictions", interrupt_registration)
    with pytest.raises(RuntimeError, match="interrupted registration"):
        request.run()
    monkeypatch.setattr(ResultsCatalog, "publish_predictions", original_publish)
    monkeypatch.setattr(
        latent_adapter,
        "run_latent_factor_cv",
        lambda *args, **kwargs: pytest.fail("valid fitted state must not retrain"),
    )

    recovered = request.run()

    assert len(recovered.predictions) == 1
    assert recovered.predictions[0].complete


def test_latent_runner_recovers_interrupted_fold_diagnostics_publication(
    tmp_path, monkeypatch
) -> None:
    study = _latent_study(tmp_path, monkeypatch)
    request = study.model(
        family="latent_factors",
        label="fwd_ret_1d",
        config_name="pca",
        overrides={"device": "cpu", "n_factors": 1},
    )
    original_publish = latent_adapter._publish_fold_extras

    def interrupt_publication(*args, **kwargs):
        raise RuntimeError("interrupted diagnostics publication")

    monkeypatch.setattr(latent_adapter, "_publish_fold_extras", interrupt_publication)
    with pytest.raises(RuntimeError, match="interrupted diagnostics publication"):
        request.run()
    monkeypatch.setattr(latent_adapter, "_publish_fold_extras", original_publish)
    monkeypatch.setattr(
        latent_adapter,
        "run_latent_factor_cv",
        lambda *args, **kwargs: pytest.fail("valid fitted state must not retrain"),
    )

    recovered = request.run()

    extras = load_fold_extras("etfs", recovered.training.hash)
    assert extras is not None
    assert [int(row["fold_id"]) for row in extras] == [0, 1]
    assert recovered.predictions[0].complete


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


def test_version_3_linear_preview_registers_identity_covered_reductions(
    tmp_path, monkeypatch
) -> None:
    study = _linear_study(tmp_path, monkeypatch)
    resolved = study.model(
        family="linear",
        label="fwd_ret_1d",
        config_name="ridge",
        execution_tier="preview",
        preview_reductions={"folds": [0]},
    ).resolve()

    training = study.results.register_training(
        resolved.spec,
        execution_tier="preview",
    )
    frame = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )

    assert resolved.spec["computation"]["preview_reductions"] == {"folds": [0]}
    assert training.execution_tier == "preview"
    assert prediction.complete
    assert training.complete


def test_model_and_causal_adapters_have_one_extension_seam() -> None:
    register_adapter("model", "fixture_family", "case_studies.utils.linear")
    register_adapter("causal", "fixture_causal", "case_studies.utils.causal")

    assert get_adapter("model", "fixture_family") is linear
    assert get_adapter("causal", "fixture_causal").__name__ == "case_studies.utils.causal"
    assert "tabular_dl" in {binding.name for binding in registered_adapters("model")}
    assert "dml" in {binding.name for binding in registered_adapters("causal")}


def test_published_logistic_presets_resolve_to_the_model_their_name_claims() -> None:
    """A preset name must describe the model it produces, for every preset in the family.

    `logistic_none` declared only max_iter and solver, so it inherited scikit-learn's defaults
    of penalty="l2", C=1.0 and fitted coefficients identical to `logistic_l2_C1.0`. The
    published menu advertised an unpenalized baseline that has never existed, and six training
    menus across four case studies reference it.

    Checking `logistic_none` alone would leave the rule unenforced everywhere else: the six
    `logistic_l2_*` presets take their l2 from the same constructor default, and a collision
    check catches a mistyped C only when it happens to collide with a sibling - `C: 5.0` on
    `logistic_l2_C10.0` would pass. Deriving the expectation from the stem checks the claim
    each name makes rather than only that the names differ.
    """
    import re
    from pathlib import Path

    from sklearn.linear_model import LogisticRegression

    from utils.paths import REPO_ROOT

    preset_dir = Path(REPO_ROOT) / "case_studies" / "config" / "logistic"
    presets = {
        path.stem: yaml.safe_load(path.read_text())["params"]
        for path in sorted(preset_dir.glob("*.yaml"))
    }
    assert presets, "no published logistic presets found"

    effective = {}
    for name, params in presets.items():
        resolved = LogisticRegression(**params).get_params()
        effective[name] = (resolved["penalty"], resolved["C"], resolved["solver"])

        stem = name.removeprefix("logistic_")
        if stem == "none":
            expected_penalty, expected_c = None, None
        else:
            match = re.fullmatch(r"(l1|l2)_C([0-9.]+)", stem)
            assert match, f"unrecognised logistic preset name {name!r}"
            expected_penalty, expected_c = match.group(1), float(match.group(2))

        assert resolved["penalty"] == expected_penalty, (
            f"{name} resolves to penalty={resolved['penalty']!r}, "
            f"but its name claims {expected_penalty!r}"
        )
        if expected_c is not None:
            assert resolved["C"] == expected_c, (
                f"{name} resolves to C={resolved['C']}, but its name claims {expected_c}"
            )

    duplicates = {
        signature: sorted(n for n, sig in effective.items() if sig == signature)
        for signature in set(effective.values())
        if sum(sig == signature for sig in effective.values()) > 1
    }
    assert not duplicates, f"published logistic presets resolve to one model: {duplicates}"


def test_peak_rss_is_read_in_the_unit_the_platform_reports(monkeypatch) -> None:
    """Linux reports ru_maxrss in kilobytes and macOS in bytes; the same reading is not both."""
    from case_studies.utils import runtime

    monkeypatch.setattr(runtime, "sys", type("_S", (), {"platform": "linux"}))
    on_linux = runtime.peak_rss_bytes()
    monkeypatch.setattr(runtime, "sys", type("_S", (), {"platform": "darwin"}))
    on_macos = runtime.peak_rss_bytes()

    assert on_linux == on_macos * 1024
