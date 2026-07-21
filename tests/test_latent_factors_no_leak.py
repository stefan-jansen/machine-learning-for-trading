"""Regression tests for the latent factor forecasting contracts."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from case_studies.utils.latent_factors.panel import align_macro_to_dates, compute_managed_portfolios


def _install_latent_registry_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    corrupt_daily_std: bool = False,
) -> tuple[dict, pl.DataFrame]:
    from case_studies.utils import registry
    from case_studies.utils.latent_factors.cv import _normalize_prediction_keys

    training_spec = {
        "config_name": "cae",
        "family": "latent_factors",
        "label": "return",
        "n_epochs": 5,
        "params": {"input_digest": "input-a"},
        "seed": 42,
    }
    training_hash = registry.training_hash_from_spec(training_spec)
    timestamps = [datetime(2020, 4, 1)] * 5 + [datetime(2020, 5, 1)] * 5
    base = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": list(range(5)) * 2,
            "y_true": list(range(5)) * 2,
            "y_score": list(range(5)) + list(reversed(range(5))),
            "fold_id": [0] * 5 + [1] * 5,
        }
    )
    prediction_rows = []
    metrics_by_hash = {}
    for epoch in (0, 5):
        prediction_hash = f"prediction-{epoch}"
        frame = base.with_columns(pl.lit(epoch).alias("epoch"))
        target = tmp_path / prediction_hash
        target.mkdir()
        frame.write_parquet(target / "predictions.parquet")
        prediction_rows.append(
            {
                "prediction_hash": prediction_hash,
                "checkpoint_value": epoch,
                "checkpoint_kind": "epoch",
            }
        )
        from ml4t.diagnostic.metrics import cross_sectional_ic

        metric = cross_sectional_ic(
            frame,
            frame,
            pred_col="y_score",
            ret_col="y_true",
            date_col="timestamp",
            entity_col="symbol",
            method="spearman",
            min_obs=5,
        )
        metrics_by_hash[prediction_hash] = pl.DataFrame(
            {
                "ic_mean_daily": [float(metric["ic_mean"])],
                "ic_std_daily": [float(metric["ic_std"]) + (0.25 if corrupt_daily_std else 0.0)],
            }
        )

    monkeypatch.setattr(
        registry,
        "load_training_runs",
        lambda *_args, **_kwargs: pl.DataFrame(
            {
                "training_hash": [training_hash],
                "spec_json": [json.dumps(training_spec)],
            }
        ),
    )
    monkeypatch.setattr(
        registry,
        "load_prediction_sets",
        lambda *_args, **_kwargs: pl.DataFrame(prediction_rows),
    )
    monkeypatch.setattr(
        registry,
        "load_prediction_metrics",
        lambda *_args, prediction_hash=None, **_kwargs: metrics_by_hash[prediction_hash],
    )
    monkeypatch.setattr(registry, "prediction_dir", lambda _case_study, value: tmp_path / value)
    expected_keys = _normalize_prediction_keys(base, "timestamp", "symbol")
    return training_spec, expected_keys


def test_managed_portfolios_are_cross_sectionally_shared() -> None:
    rng = np.random.default_rng(1)
    chars = rng.normal(size=(12, 8, 4)).astype(np.float32)
    returns = rng.normal(size=(12, 8)).astype(np.float32)
    portfolios = compute_managed_portfolios(chars, returns)

    for date_idx in range(portfolios.shape[0]):
        assert np.allclose(portfolios[date_idx, :1, :], portfolios[date_idx]), (
            f"managed portfolios vary within date {date_idx}"
        )


def test_managed_portfolios_use_current_date_only() -> None:
    rng = np.random.default_rng(2)
    chars = rng.normal(size=(10, 6, 3)).astype(np.float32)
    returns = rng.normal(size=(10, 6)).astype(np.float32)

    portfolios_a = compute_managed_portfolios(chars, returns)
    perturbed = returns.copy()
    perturbed[4] += 100.0
    portfolios_b = compute_managed_portfolios(chars, perturbed)

    changed = np.abs(portfolios_a - portfolios_b).max(axis=(1, 2))
    assert changed[4] > 0.0
    assert np.all(changed[np.arange(len(changed)) != 4] == 0.0)


def test_macro_alignment_never_backfills_from_future() -> None:
    dates = [datetime(2019, 12, 31), datetime(2020, 1, 31), datetime(2020, 2, 29)]
    macro = pl.DataFrame(
        {
            "timestamp": [datetime(2020, 1, 15), datetime(2020, 2, 15)],
            "rate": [1.0, 2.0],
        }
    )

    aligned, features = align_macro_to_dates(macro, dates)

    assert features == ["rate"]
    assert aligned[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_macro_panel_applies_allowlist_and_availability_lag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from case_studies.utils.latent_factors import case_study

    captured: dict[str, object] = {}

    def fake_load_macro(*, series=None):
        captured["series"] = series
        return pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 31)],
                "dgs10": [1.5],
                "vixcls": [15.0],
            }
        )

    monkeypatch.setattr(case_study, "load_macro", fake_load_macro)
    panel = case_study._load_macro_panel(
        {
            "macro_series": ["dgs10", "vixcls"],
            "macro_availability_lag_days": 1,
        }
    )

    assert captured["series"] == ["dgs10", "vixcls"]
    assert panel["timestamp"].to_list() == [datetime(2020, 2, 1)]


def test_pca_entity_eligibility_is_learned_from_train_only() -> None:
    """Validation coverage must not decide which entities enter the PCA panel."""
    from case_studies.utils.latent_factors.cv import _prepare_fold_inputs

    train_dates = pl.date_range(
        datetime(2020, 1, 1),
        datetime(2020, 1, 12),
        interval="1d",
        eager=True,
    ).to_list()
    val_dates = pl.date_range(
        datetime(2020, 1, 13),
        datetime(2020, 1, 15),
        interval="1d",
        eager=True,
    ).to_list()

    rows: list[dict[str, object]] = []
    for idx, timestamp in enumerate(train_dates + val_dates):
        rows.append({"timestamp": timestamp, "symbol": "A", "feature": idx, "label": idx / 100})
    for idx, timestamp in enumerate(train_dates[:9] + val_dates):
        rows.append({"timestamp": timestamp, "symbol": "B", "feature": idx, "label": idx / 100})

    inputs = _prepare_fold_inputs(
        dataset=pl.DataFrame(rows),
        split={
            "fold": 0,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "val_start": val_dates[0],
            "val_end": val_dates[-1],
        },
        feature_names=["feature"],
        label_col="label",
        date_col="timestamp",
        entity_col="symbol",
        eval_label_col=None,
        macro_panel=None,
        need_pca_inputs=True,
    )

    assert inputs is not None
    assert inputs["pca"]["returns_train"].shape == (12, 1)
    assert set(inputs["pca"]["val_entities"].ravel()) == {"A"}
    assert inputs["ragged"]["returns_val"].shape == (3, 2)


def test_pca_only_fold_preparation_skips_ragged_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from case_studies.utils.latent_factors import cv

    timestamps = pl.date_range(
        datetime(2020, 1, 1),
        datetime(2020, 1, 15),
        interval="1d",
        eager=True,
    ).to_list()
    dataset = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["A"] * len(timestamps),
            "feature": range(len(timestamps)),
            "label": [value / 100 for value in range(len(timestamps))],
        }
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("PCA-only folds must not build the ragged panel")

    monkeypatch.setattr(cv, "prepare_ragged_panel_data", fail_if_called)
    inputs = cv._prepare_fold_inputs(
        dataset=dataset,
        split={
            "fold": 0,
            "train_start": timestamps[0],
            "train_end": timestamps[11],
            "val_start": timestamps[12],
            "val_end": timestamps[-1],
        },
        feature_names=["feature"],
        label_col="label",
        date_col="timestamp",
        entity_col="symbol",
        eval_label_col=None,
        macro_panel=None,
        need_pca_inputs=True,
        need_ragged_inputs=False,
    )

    assert inputs is not None
    assert inputs["ragged"] is None
    assert inputs["pca"]["returns_train"].shape == (12, 1)


def test_cae_validation_batch_receives_validation_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from case_studies.utils.latent_factors import library_bridge

    rng = np.random.default_rng(123)
    chars_train = rng.normal(size=(12, 8, 4)).astype(np.float32)
    returns_train = rng.normal(size=(12, 8)).astype(np.float32) * 0.02
    chars_val = rng.normal(size=(5, 8, 4)).astype(np.float32)
    returns_val = rng.normal(size=(5, 8)).astype(np.float32) * 0.02

    captured: dict[str, np.ndarray] = {}

    def capture_pipeline(**kwargs):
        captured["validation_returns"] = kwargs["val_batch"].returns.copy()
        return {
            "checkpoint_predictions": {0: np.zeros_like(returns_val)},
            "checkpoint_epochs": [0],
        }

    monkeypatch.setattr(library_bridge, "_run_checkpointed_latent_pipeline", capture_pipeline)

    library_bridge.run_cae_fold_with_library(
        chars_train,
        returns_train,
        chars_val=chars_val,
        returns_val=returns_val,
        n_factors=2,
        n_epochs=2,
        n_ensemble=1,
        hidden_units=(8,),
        checkpoint_epochs=[2],
    )

    assert np.array_equal(captured["validation_returns"], returns_val)


def test_latent_cuda_request_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from case_studies.utils.latent_factors import library_bridge

    monkeypatch.setattr(library_bridge.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        library_bridge.configure_latent_torch_runtime(
            "cuda",
            seed=42,
            num_threads=8,
            deterministic_algorithms=True,
        )


def test_latent_training_identity_includes_input_splits_and_runtime() -> None:
    from case_studies.utils import registry
    from case_studies.utils.latent_factors.cv import _apply_latent_factor_runtime_spec

    kwargs = {
        "spec": {
            "family": "latent_factors",
            "config_name": "cae",
            "label": "return",
            "seed": 42,
        },
        "n_factors": 5,
        "n_epochs": 50,
        "model_kwargs": {"checkpoint_interval": 5},
        "fold_extras": [],
        "feature_names": ["value", "size"],
        "splits": [
            {
                "fold": 0,
                "train_start": datetime(2020, 1, 1),
                "train_end": datetime(2020, 12, 31),
                "val_start": datetime(2021, 1, 1),
                "val_end": datetime(2021, 12, 31),
            }
        ],
        "task_type": "regression",
        "class_values": None,
        "eval_label_col": None,
        "input_digest": "input-a",
        "macro_digest": None,
        "runtime_spec": {
            "device": "cuda",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "num_threads": 8,
            "seed": 42,
        },
    }
    spec = _apply_latent_factor_runtime_spec(**kwargs)
    params = spec["params"]

    assert params["feature_names"] == ["value", "size"]
    assert params["input_digest"] == "input-a"
    assert params["runtime"]["device"] == "cuda"
    assert params["splits"][0]["val_end"] == "2021-12-31T00:00:00"

    baseline_hash = registry.training_hash_from_spec(spec)
    for field, value in (
        ("input_digest", "input-b"),
        ("feature_names", ["size", "value"]),
    ):
        changed = _apply_latent_factor_runtime_spec(**{**kwargs, field: value})
        assert registry.training_hash_from_spec(changed) != baseline_hash

    changed_runtime = _apply_latent_factor_runtime_spec(
        **{
            **kwargs,
            "runtime_spec": {**kwargs["runtime_spec"], "device": "cpu"},
        }
    )
    assert registry.training_hash_from_spec(changed_runtime) != baseline_hash


def test_latent_input_digest_is_order_stable_and_value_sensitive() -> None:
    from case_studies.utils.latent_factors.cv import _latent_input_digest

    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2021, 2, 1), datetime(2021, 1, 1)],
            "symbol": ["B", "A"],
            "value": [2.0, 1.0],
            "return": [0.2, 0.1],
        }
    )
    splits = [
        {
            "fold": 0,
            "train_start": datetime(2021, 1, 1),
            "train_end": datetime(2021, 1, 1),
            "val_start": datetime(2021, 2, 1),
            "val_end": datetime(2021, 2, 1),
        }
    ]

    digest = _latent_input_digest(
        frame,
        feature_names=["value"],
        label_col="return",
        eval_label_col=None,
        date_col="timestamp",
        entity_col="symbol",
        splits=splits,
    )
    reordered = _latent_input_digest(
        frame.reverse(),
        feature_names=["value"],
        label_col="return",
        eval_label_col=None,
        date_col="timestamp",
        entity_col="symbol",
        splits=splits,
    )
    changed = _latent_input_digest(
        frame.with_columns(
            pl.when(pl.col("symbol") == "A").then(9.0).otherwise(pl.col("value")).alias("value")
        ),
        feature_names=["value"],
        label_col="return",
        eval_label_col=None,
        date_col="timestamp",
        entity_col="symbol",
        splits=splits,
    )

    assert reordered == digest
    assert changed != digest


def test_latent_input_digest_excludes_sealed_holdout() -> None:
    from case_studies.utils import registry
    from case_studies.utils.latent_factors.cv import (
        _apply_latent_factor_runtime_spec,
        _latent_input_digest,
    )

    timestamps = [datetime(2020, month, 1) for month in range(1, 7)]
    frame = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["A"] * len(timestamps),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "return": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    splits = [
        {
            "fold": 0,
            "train_start": datetime(2020, 1, 1),
            "train_end": datetime(2020, 3, 1),
            "val_start": datetime(2020, 4, 1),
            "val_end": datetime(2020, 5, 1),
        }
    ]
    digest_kwargs = {
        "feature_names": ["value"],
        "label_col": "return",
        "eval_label_col": None,
        "date_col": "timestamp",
        "entity_col": "symbol",
        "splits": splits,
    }
    digest = _latent_input_digest(frame, **digest_kwargs)
    holdout_changed = _latent_input_digest(
        frame.with_columns(
            pl.when(pl.col("timestamp") > datetime(2020, 5, 1))
            .then(999.0)
            .otherwise(pl.col(column))
            .alias(column)
            for column in ("value", "return")
        ),
        **digest_kwargs,
    )
    development_changed = _latent_input_digest(
        frame.with_columns(
            pl.when(pl.col("timestamp") == datetime(2020, 2, 1))
            .then(999.0)
            .otherwise(pl.col("value"))
            .alias("value")
        ),
        **digest_kwargs,
    )

    spec_kwargs = {
        "spec": {
            "family": "latent_factors",
            "config_name": "cae",
            "label": "return",
            "seed": 42,
        },
        "n_factors": 5,
        "n_epochs": 50,
        "model_kwargs": {"checkpoint_interval": 5},
        "fold_extras": [],
        "feature_names": ["value"],
        "splits": splits,
        "task_type": "regression",
        "class_values": None,
        "eval_label_col": None,
        "macro_digest": None,
        "runtime_spec": {
            "device": "cuda",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "num_threads": 8,
            "seed": 42,
        },
    }
    training_hash = registry.training_hash_from_spec(
        _apply_latent_factor_runtime_spec(**spec_kwargs, input_digest=digest)
    )
    holdout_training_hash = registry.training_hash_from_spec(
        _apply_latent_factor_runtime_spec(**spec_kwargs, input_digest=holdout_changed)
    )

    assert holdout_changed == digest
    assert holdout_training_hash == training_hash
    assert development_changed != digest


def test_sdf_expected_spec_includes_library_output_defaults() -> None:
    from case_studies.utils import registry
    from case_studies.utils.latent_factors.cv import _build_expected_latent_training_spec

    model_kwargs = {
        "n_epochs_unc": 256,
        "n_epochs_moment": 64,
        "n_epochs_cond": 1024,
        "checkpoint_epochs": [256, 512, 768, 1024, 1280],
        "beta_n_epochs": 256,
        "beta_checkpoint_epochs": [256],
        "beta_default_checkpoint": 256,
    }
    splits = [
        {
            "fold": 0,
            "train_start": datetime(2020, 1, 1),
            "train_end": datetime(2020, 12, 31),
            "val_start": datetime(2021, 1, 1),
            "val_end": datetime(2021, 12, 31),
        }
    ]
    spec, checkpoints = _build_expected_latent_training_spec(
        model_name="sdf",
        label_col="return",
        n_factors=5,
        n_epochs=50,
        model_kwargs=model_kwargs,
        feature_names=["value"],
        splits=splits,
        task_type="regression",
        class_values=None,
        eval_label_col=None,
        input_digest="input-a",
        macro_digest="macro-a",
        runtime_spec={
            "device": "cuda",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "num_threads": 8,
            "seed": 42,
        },
    )
    actual = dict(spec)
    actual["output_mode"] = "beta_network"
    actual["expected_return_mapper"] = "linear"

    assert checkpoints == (256, 512, 768, 1024, 1280)
    assert spec["output_mode"] == "beta_network"
    assert spec["expected_return_mapper"] == "linear"
    assert registry.training_hash_from_spec(spec) == registry.training_hash_from_spec(actual)


def test_latent_registration_builds_complete_training_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from case_studies.utils import registry
    from case_studies.utils.latent_factors.cv import _register_model_predictions

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        registry,
        "build_training_spec",
        lambda *_args, **_kwargs: {
            "family": "latent_factors",
            "config_name": "cae",
            "label": "return",
            "seed": 42,
        },
    )

    def capture_training_run(_case_study_id: str, *, spec, **_kwargs) -> str:
        captured["spec"] = spec
        return "training-hash"

    monkeypatch.setattr(registry, "register_training_run", capture_training_run)

    def capture_prediction_set(*_args, checkpoint_value=None, **_kwargs) -> str:
        captured.setdefault("checkpoints", []).append(checkpoint_value)
        return "prediction-hash"

    monkeypatch.setattr(registry, "register_prediction_set", capture_prediction_set)

    _register_model_predictions(
        case_study_id="probe",
        model_name="cae",
        label_col="return",
        n_epochs=5,
        n_factors=5,
        notebook="08b_conditional_autoencoder",
        prediction_split="validation",
        task_type="regression",
        class_values=None,
        eval_label_col=None,
        started_at="2026-07-21T00:00:00+00:00",
        elapsed=1.0,
        model_kwargs={"checkpoint_interval": 5},
        fold_extras=[{"checkpoint_epochs": [0, 5]}],
        fold_ics_df=pl.DataFrame({"fold_id": [0, 0], "epoch": [0, 5], "ic_mean": [0.1, 0.2]}),
        preds_df=pl.DataFrame(
            {
                "timestamp": [datetime(2021, 1, 1), datetime(2021, 1, 1)],
                "symbol": ["A", "A"],
                "y_true": [0.1, 0.1],
                "y_score": [0.2, 0.3],
                "fold_id": [0, 0],
                "config_name": ["cae", "cae"],
                "epoch": [0, 5],
            }
        ),
        feature_names=["value"],
        splits=[
            {
                "fold": 0,
                "train_start": datetime(2020, 1, 1),
                "train_end": datetime(2020, 12, 31),
                "val_start": datetime(2021, 1, 1),
                "val_end": datetime(2021, 12, 31),
            }
        ],
        input_digest="input-a",
        macro_digest=None,
        runtime_spec={
            "device": "cuda",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "num_threads": 8,
            "seed": 42,
        },
    )

    spec = captured["spec"]
    assert isinstance(spec, dict)
    assert spec["n_epochs"] == 5
    assert spec["checkpoint_interval"] == 5
    assert spec["checkpoint_epochs"] == [5]
    assert spec["params"]["input_digest"] == "input-a"
    assert captured["checkpoints"] == [5]


@pytest.mark.gpu
def test_cae_predictions_independent_of_validation_returns() -> None:
    """End-to-end regression: perturbing validation returns must not change predictions.

    Restores the byte-identical fit-twice check that the wiring-only test above
    cannot enforce — guards against future changes in
    `_run_checkpointed_latent_pipeline` (or anything downstream of
    `model.fit(..., validation_batch=val_batch)`) that accidentally let
    validation returns influence the fitted model.
    """
    pytest.importorskip("torch")
    from case_studies.utils.latent_factors.cae import run_cae_fold

    rng = np.random.default_rng(31)
    chars_train = rng.normal(size=(12, 8, 4)).astype(np.float32)
    returns_train = rng.normal(size=(12, 8)).astype(np.float32) * 0.02
    chars_val = rng.normal(size=(5, 8, 4)).astype(np.float32)
    returns_val = rng.normal(size=(5, 8)).astype(np.float32) * 0.02

    base_preds_by_epoch, _ = run_cae_fold(
        chars_train,
        returns_train,
        chars_val,
        returns_val,
        n_factors=2,
        n_epochs=2,
        checkpoint_epochs=[2],
        hidden_units=(8,),
        log_fn=lambda *args, **kwargs: None,
    )
    perturbed_val = returns_val.copy()
    perturbed_val += 100.0
    perturbed_preds_by_epoch, _ = run_cae_fold(
        chars_train,
        returns_train,
        chars_val,
        perturbed_val,
        n_factors=2,
        n_epochs=2,
        checkpoint_epochs=[2],
        hidden_units=(8,),
        log_fn=lambda *args, **kwargs: None,
    )

    base_preds = base_preds_by_epoch[2]
    perturbed_preds = perturbed_preds_by_epoch[2]
    assert np.array_equal(base_preds, perturbed_preds), (
        "CAE predictions changed when validation returns were perturbed by +100 — "
        "validation returns are leaking into the fitted model"
    )


@pytest.mark.gpu
def test_cae_classification_uses_continuous_factor_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from case_studies.utils.latent_factors import library_bridge

    captured: dict[str, np.ndarray] = {}
    original_cross_section_batch = library_bridge._cross_section_batch

    def capture_batch(
        characteristics: np.ndarray,
        *,
        returns: np.ndarray | None = None,
        factor_returns: np.ndarray | None = None,
        context_features: np.ndarray | None = None,
    ):
        if factor_returns is not None:
            captured["factor_returns"] = factor_returns.copy()
        return original_cross_section_batch(
            characteristics,
            returns=returns,
            factor_returns=factor_returns,
            context_features=context_features,
        )

    monkeypatch.setattr(library_bridge, "_cross_section_batch", capture_batch)

    rng = np.random.default_rng(7)
    chars = rng.normal(size=(24, 10, 5)).astype(np.float32)
    class_labels = (rng.random(size=(24, 10)) > 0.5).astype(np.float32)
    factor_returns = rng.normal(size=(24, 10)).astype(np.float32) * 0.02

    from case_studies.utils.latent_factors.cae import run_cae_fold

    run_cae_fold(
        chars[:18],
        class_labels[:18],
        chars[18:],
        class_labels[18:],
        n_factors=2,
        factor_returns_train=factor_returns[:18],
        n_epochs=1,
        checkpoint_epochs=[1],
        hidden_units=(8,),
        task_type="classification",
        log_fn=lambda *args, **kwargs: None,
    )

    assert np.array_equal(captured["factor_returns"], factor_returns[:18])


def test_reporting_epoch_defaults_to_last_checkpoint() -> None:
    from case_studies.utils.latent_factors.cv import _select_reporting_epoch

    metrics = pl.DataFrame(
        {
            "fold_id": [0, 0, 1, 1],
            "epoch": [5, 10, 5, 10],
            "ic_mean": [0.12, 0.03, 0.11, 0.02],
        }
    )

    epoch, mean_ic = _select_reporting_epoch(
        metrics,
        checkpoint_selection_policy="fixed",
        reporting_epoch=None,
    )

    assert epoch == 10
    assert mean_ic == pytest.approx(0.025)


def test_reporting_epoch_excludes_validation_selected_checkpoint_zero_by_default() -> None:
    from case_studies.utils.latent_factors.cv import _select_reporting_epoch

    metrics = pl.DataFrame(
        {
            "fold_id": [0, 0, 1, 1],
            "epoch": [0, 10, 0, 10],
            "ic_mean": [0.04, 0.03, 0.05, 0.02],
        }
    )

    epoch, mean_ic = _select_reporting_epoch(
        metrics,
        checkpoint_selection_policy="fixed",
        reporting_epoch=None,
    )

    assert epoch == 10
    assert mean_ic == pytest.approx(0.025)


def test_reporting_epoch_allows_explicit_checkpoint_zero() -> None:
    from case_studies.utils.latent_factors.cv import _select_reporting_epoch

    metrics = pl.DataFrame(
        {
            "fold_id": [0, 0, 1, 1],
            "epoch": [0, 10, 0, 10],
            "ic_mean": [0.04, 0.03, 0.05, 0.02],
        }
    )

    epoch, mean_ic = _select_reporting_epoch(
        metrics,
        checkpoint_selection_policy="fixed",
        reporting_epoch=0,
    )

    assert epoch == 0
    assert mean_ic == pytest.approx(0.045)


def test_latent_registry_cache_replays_exact_complete_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from case_studies.utils.latent_factors.cv import _load_registered_latent_factor

    training_spec, expected_keys = _install_latent_registry_cache(monkeypatch, tmp_path)
    result = _load_registered_latent_factor(
        "probe",
        model_name="cae",
        training_spec=training_spec,
        prediction_split="validation",
        expected_checkpoints=(0, 5),
        expected_keys=expected_keys,
        date_col="timestamp",
        entity_col="symbol",
        eval_label_col=None,
    )

    assert result is not None
    metrics, predictions = result
    assert metrics.shape == (4, 3)
    assert predictions.shape == (20, 6)


def test_latent_registry_cache_rejects_corrupted_daily_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from case_studies.utils.latent_factors.cv import _load_registered_latent_factor

    training_spec, expected_keys = _install_latent_registry_cache(
        monkeypatch,
        tmp_path,
        corrupt_daily_std=True,
    )

    with pytest.raises(ValueError, match="daily metric mismatch"):
        _load_registered_latent_factor(
            "probe",
            model_name="cae",
            training_spec=training_spec,
            prediction_split="validation",
            expected_checkpoints=(0, 5),
            expected_keys=expected_keys,
            date_col="timestamp",
            entity_col="symbol",
            eval_label_col=None,
        )


def test_latent_registry_cache_requires_exact_checkpoint_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from case_studies.utils.latent_factors.cv import _load_registered_latent_factor

    training_spec, expected_keys = _install_latent_registry_cache(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="checkpoint count"):
        _load_registered_latent_factor(
            "probe",
            model_name="cae",
            training_spec=training_spec,
            prediction_split="validation",
            expected_checkpoints=(0, 5, 10),
            expected_keys=expected_keys,
            date_col="timestamp",
            entity_col="symbol",
            eval_label_col=None,
        )


def test_prediction_frame_preserves_temporal_timestamp_dtype() -> None:
    from case_studies.utils.latent_factors.cv import _build_prediction_frame

    predictions = np.array([[0.1, np.nan, 0.3], [0.4, 0.5, np.nan]], dtype=np.float64)
    returns_val = np.array([[0.0, np.nan, 1.0], [1.0, 2.0, np.nan]], dtype=np.float64)
    val_dates = np.array(["2024-01-31", "2024-02-29"], dtype="datetime64[ns]")
    val_entities = np.array(
        [["A", "B", "C"], ["A", "B", "C"]],
        dtype=object,
    )

    frame = _build_prediction_frame(
        predictions=predictions,
        returns_val=returns_val,
        eval_returns_val=None,
        val_dates=val_dates,
        val_entities=val_entities,
        fold_id=0,
        model_name="ipca",
        epoch=0,
    )

    assert frame is not None
    assert frame["timestamp"].dtype.is_temporal()
    assert frame["timestamp"].to_list() == [
        datetime(2024, 1, 31),
        datetime(2024, 1, 31),
        datetime(2024, 2, 29),
        datetime(2024, 2, 29),
    ]


def test_rebalance_scoring_thins_to_declared_schedule() -> None:
    from case_studies.utils.latent_factors.cv import _compute_frame_ic, _score_prediction_frame

    frame = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 15),
                datetime(2024, 1, 15),
                datetime(2024, 1, 15),
                datetime(2024, 1, 15),
                datetime(2024, 1, 15),
                datetime(2024, 1, 31),
                datetime(2024, 1, 31),
                datetime(2024, 1, 31),
                datetime(2024, 1, 31),
                datetime(2024, 1, 31),
                datetime(2024, 2, 15),
                datetime(2024, 2, 15),
                datetime(2024, 2, 15),
                datetime(2024, 2, 15),
                datetime(2024, 2, 15),
                datetime(2024, 2, 29),
                datetime(2024, 2, 29),
                datetime(2024, 2, 29),
                datetime(2024, 2, 29),
                datetime(2024, 2, 29),
            ],
            "symbol": ["A", "B", "C", "D", "E"] * 4,
            "y_true": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                4.0,
                3.0,
                2.0,
                1.0,
                0.0,
                4.0,
                3.0,
                2.0,
                1.0,
                0.0,
            ],
            "y_score": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                4.0,
                3.0,
                2.0,
                1.0,
                0.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                4.0,
                3.0,
                2.0,
                1.0,
                0.0,
            ],
            "fold_id": [0] * 20,
            "config_name": ["cae"] * 20,
            "epoch": [10] * 20,
        }
    )

    _, full_periods = _compute_frame_ic(frame)
    thinned = _score_prediction_frame(
        frame,
        score_dates="rebalance",
        score_cadence="monthly_month_end",
        score_rebalance_step=1,
    )
    _, thinned_periods = _compute_frame_ic(thinned)

    assert full_periods == 4
    assert thinned_periods == 2
    assert thinned is not None
    assert thinned["timestamp"].unique().sort().to_list() == [
        datetime(2024, 1, 31),
        datetime(2024, 2, 29),
    ]
