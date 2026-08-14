from __future__ import annotations

import gc
import json
import os
import weakref
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
import yaml

from case_studies.research import (
    CVSpec,
    LabelDefinition,
    ModelRun,
    Study,
    get_adapter,
    register_adapter,
    registered_adapters,
    run_models,
)
from case_studies.research.results import ResultsCatalog
from case_studies.utils import gbm as gbm_utils
from case_studies.utils import linear
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


def test_linear_batch_is_fold_major_and_matches_individual_execution(tmp_path, monkeypatch) -> None:
    study = _linear_study(tmp_path / "batch", monkeypatch)
    individual_study = _linear_study(tmp_path / "individual", monkeypatch)
    original_prepare = modeling.prepare_single_fold
    prepared_folds: list[int] = []
    released_arrays: list[weakref.ReferenceType[np.ndarray]] = []

    def observed_prepare(*args, **kwargs):
        gc.collect()
        if released_arrays:
            assert released_arrays[-1]() is None
        fold = original_prepare(*args, **kwargs)
        assert fold is not None
        prepared_folds.append(int(fold["fold"]))
        released_arrays.append(weakref.ref(fold["X_train"]))
        return fold

    monkeypatch.setattr(linear, "prepare_single_fold", observed_prepare, raising=False)
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
    monkeypatch.setattr(linear, "prepare_single_fold", original_prepare)
    individual = individual_study.model(
        family="linear",
        label="fwd_ret_1d",
        config_name="ridge",
        overrides={"alpha": 1.0},
    ).run()
    gc.collect()

    assert batch_prepared_folds == [0, 1]
    assert released_arrays[-1]() is None
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


def test_linear_batch_separates_incompatible_sampling_and_is_order_invariant(
    tmp_path, monkeypatch
) -> None:
    first = _linear_study(tmp_path / "first", monkeypatch)
    second = _linear_study(tmp_path / "second", monkeypatch)
    original_prepare = modeling.prepare_single_fold
    original_load = linear.load_modeling_dataset
    preparation: list[tuple[int, float]] = []
    load_count = 0

    def observed_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return original_load(*args, **kwargs)

    def observed_prepare(*args, **kwargs):
        fold = original_prepare(*args, **kwargs)
        assert fold is not None
        preparation.append((int(fold["fold"]), float(kwargs["train_sample_frac"])))
        return fold

    monkeypatch.setattr(linear, "prepare_single_fold", observed_prepare, raising=False)
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

    with pytest.raises(RuntimeError, match="injected candidate failure"):
        run_models(study, requests=requests)

    completed = study.predictions.table().filter(pl.col("complete"))
    assert completed.height == 2
    monkeypatch.setattr(linear.Ridge, "fit", original_fit)
    recovered = run_models(study, requests=requests)
    recovered_by_alpha = {
        run.training.spec()["computation"]["model"]["effective_params_by_fold"]["0"]["alpha"]: run
        for run in recovered.runs
    }

    assert recovered_by_alpha[1.0].diagnostics["cache_hit"] is True
    assert recovered_by_alpha[3.0].diagnostics["cache_hit"] is True
    assert recovered_by_alpha[2.0].diagnostics["reused_folds"] == [0]
    assert recovered_by_alpha[2.0].diagnostics["fitted_folds"] == [1]
    assert recovered_by_alpha[2.0].diagnostics["base_fold_preparations"] == 1
    assert study.predictions.table().filter(pl.col("complete")).height == 3


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
    assert recovered.runs[0].diagnostics["base_fold_preparations"] == 1
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
    runtime = json.loads((model_dir.parent / "runtime.json").read_text())
    assert runtime["elapsed_s"] > 0


def test_linear_runtime_update_is_atomic(tmp_path, monkeypatch) -> None:
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text('{"status": "registered"}\n')

    def interrupt_replace(*args, **kwargs):
        raise RuntimeError("interrupted runtime update")

    monkeypatch.setattr(linear.os, "replace", interrupt_replace)
    with pytest.raises(RuntimeError, match="interrupted runtime update"):
        linear._write_runtime_fields(runtime_path, elapsed_s=1.0)

    assert json.loads(runtime_path.read_text()) == {"status": "registered"}
    assert list(tmp_path.glob(".runtime.json.*.tmp")) == []


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
