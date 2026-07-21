"""Regression tests for the latent factor forecasting contracts."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from case_studies.utils.latent_factors.panel import compute_managed_portfolios


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


def test_reporting_epoch_prefers_validation_best_checkpoint_zero() -> None:
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

    assert epoch == 0
    assert mean_ic == pytest.approx(0.045)


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


def test_ipca_solver_controls_change_training_identity() -> None:
    from case_studies.utils.latent_factors.cv import _apply_latent_factor_runtime_spec
    from case_studies.utils.registry.specs import training_hash_from_spec

    base = {
        "config_name": "ipca",
        "family": "latent_factors",
        "label": "fwd_ret_21d",
        "n_folds": 8,
        "params": {"n_factors": 5},
        "seed": 42,
    }
    old = _apply_latent_factor_runtime_spec(
        spec=base,
        n_factors=5,
        n_epochs=50,
        model_kwargs={"max_iter": 100},
        fold_extras=[],
    )
    corrected = _apply_latent_factor_runtime_spec(
        spec=base,
        n_factors=5,
        n_epochs=50,
        model_kwargs={"max_iter": 10_000},
        fold_extras=[],
    )

    assert old["params"]["max_iter"] == 100
    assert corrected["params"]["max_iter"] == 10_000
    assert training_hash_from_spec(old) != training_hash_from_spec(corrected)


def test_ipca_wrapper_defers_to_library_iteration_default() -> None:
    import inspect

    from ml4t.models.configs import IPCAConfig

    from case_studies.utils.latent_factors.ipca import run_ipca_fold

    default = inspect.signature(run_ipca_fold).parameters["max_iter"].default
    assert default == IPCAConfig().max_iter


def test_ipca_nonconvergence_blocks_registration() -> None:
    from case_studies.utils.latent_factors.cv import _require_ipca_convergence

    _require_ipca_convergence("cae", [{"fold_id": 0, "converged": False}])
    _require_ipca_convergence("ipca", [{"fold_id": 0, "converged": True}])
    with pytest.raises(RuntimeError, match=r"folds \[1\].*refusing to register"):
        _require_ipca_convergence("ipca", [{"fold_id": 1, "converged": False}])


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


def test_case_study_default_scores_ic_at_every_prediction_timestamp() -> None:
    from case_studies.utils.latent_factors.cv import _resolve_metric_policy

    policy = _resolve_metric_policy(
        case_study_id="etfs",
        label_col="fwd_ret_21d",
        checkpoint_selection_policy=None,
        reporting_epoch=None,
        score_dates="auto",
        score_cadence=None,
        score_rebalance_step=None,
    )

    assert policy["score_dates"] == "all"
    assert policy["score_cadence"] == ""
    assert policy["score_rebalance_step"] == 1


def test_latent_cv_replaces_placeholder_with_fold_temporal_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from case_studies.utils.latent_factors import cv

    dates = pl.date_range(
        datetime(2020, 1, 1),
        datetime(2020, 1, 15),
        interval="1d",
        eager=True,
    )
    symbols = ["A", "B", "C"]
    dataset = pl.DataFrame(
        {
            "timestamp": [date for date in dates for _ in symbols],
            "symbol": symbols * len(dates),
            "base_feature": [0.0, 1.0, 2.0] * len(dates),
            # load_modeling_dataset joins fold 0 as a schema placeholder.
            "temporal_feature": [1.0, 2.0, 3.0] * len(dates),
            "fwd_ret": [0.01, 0.02, 0.03] * len(dates),
        }
    )
    temporal_by_fold = pl.DataFrame(
        {
            "timestamp": [date for date in dates for _ in symbols],
            "symbol": symbols * len(dates),
            "fold": [1] * (len(dates) * len(symbols)),
            "temporal_feature": [3.0, 2.0, 1.0] * len(dates),
        }
    ).to_pandas()
    split = {
        "fold": 1,
        "train_start": datetime(2020, 1, 1),
        "train_end": datetime(2020, 1, 12),
        "val_start": datetime(2020, 1, 13),
        "val_end": datetime(2020, 1, 15),
    }

    captured: dict[str, np.ndarray] = {}

    def capture_fold(
        chars_train: np.ndarray,
        returns_train: np.ndarray,
        chars_val: np.ndarray,
        returns_val: np.ndarray,
        n_factors: int,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del returns_train, chars_val, n_factors
        captured["chars_train"] = chars_train.copy()
        return np.zeros_like(returns_val), {"converged": True}

    monkeypatch.setitem(cv._MODEL_RUNNERS, "ipca", capture_fold)
    cv.run_latent_factor_cv(
        panel_data=None,
        splits=[split],
        models=["ipca"],
        n_factors=2,
        use_cache=False,
        dataset=dataset,
        feature_names=["base_feature", "temporal_feature"],
        label_col="fwd_ret",
        date_col="timestamp",
        entity_col="symbol",
        temporal_by_fold=temporal_by_fold,
        temporal_keys=["timestamp", "symbol"],
        temporal_feature_names=["temporal_feature"],
    )

    assert captured["chars_train"][0, :, 1].tolist() == pytest.approx([0.5, 0.0, -0.5])


def test_fold_temporal_assembly_changes_training_identity() -> None:
    from case_studies.utils.latent_factors.cv import (
        TEMPORAL_FEATURE_ASSEMBLY,
        _apply_latent_factor_runtime_spec,
    )
    from case_studies.utils.registry.specs import training_hash_from_spec

    base = {
        "config_name": "ipca",
        "family": "latent_factors",
        "label": "fwd_ret_21d",
        "n_folds": 8,
        "params": {"n_factors": 5},
        "seed": 42,
    }
    placeholder = _apply_latent_factor_runtime_spec(
        spec=base,
        n_factors=5,
        n_epochs=50,
        model_kwargs={},
        fold_extras=[],
    )
    fold_scoped = _apply_latent_factor_runtime_spec(
        spec=base,
        n_factors=5,
        n_epochs=50,
        model_kwargs={},
        fold_extras=[],
        temporal_feature_assembly=TEMPORAL_FEATURE_ASSEMBLY,
    )

    assert fold_scoped["temporal_feature_assembly"] == TEMPORAL_FEATURE_ASSEMBLY
    assert training_hash_from_spec(placeholder) != training_hash_from_spec(fold_scoped)
