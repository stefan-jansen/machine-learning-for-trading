"""Adapters from case-study fold inputs to ml4t-models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ml4t.models.asset_prediction import SAEModel
from ml4t.models.configs import (
    CAEConfig,
    IPCAConfig,
    PCAConfig,
    SAEConfig,
    StochasticDiscountFactorConfig,
)
from ml4t.models.forecasters import ExpandingMeanFactorForecaster
from ml4t.models.latent_factors import CAEModel, IPCAModel, PCAModel
from ml4t.models.mappers import BetaLambdaMapper
from ml4t.models.stochastic_discount_factor import (
    LinearStochasticDiscountFactorReturnMapper,
    StochasticDiscountFactorBetaNetworkHead,
    StochasticDiscountFactorModel,
)
from ml4t.models.types import CrossSectionBatch, PersistentPanelBatch

from case_studies.utils.latent_factors.common import TaskType, summarize_predictions

# CAE/SAE/SDF inherit `device="cpu"` from `AssetPredictionConfig` /
# `LatentFactorConfig` defaults in ml4t-models, so without an explicit
# override the case-study runs train on CPU even when CUDA is available.
# Fall back to "cpu" only when no GPU is present so notebook execution
# doesn't break on CPU-only machines.
_PREFERRED_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_pca_fold_with_library(
    returns_train: np.ndarray,
    returns_val: np.ndarray,
    *,
    n_factors: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    _validate_persistent_returns(returns_train, returns_val)
    train_batch = PersistentPanelBatch(
        returns=returns_train,
        timestamps=tuple(range(returns_train.shape[0])),
        asset_ids=_asset_ids(returns_train.shape[1]),
    )
    val_batch = PersistentPanelBatch(
        timestamps=tuple(range(returns_val.shape[0])),
        asset_ids=_asset_ids(returns_val.shape[1]),
    )

    model = PCAModel(PCAConfig(n_factors=n_factors))
    fit = model.fit(train_batch)
    train_state = model.extract(train_batch)
    val_state = model.extract(val_batch)
    forecaster = ExpandingMeanFactorForecaster()
    forecaster.fit(train_state)
    forecast = forecaster.predict(val_state)
    predictions = (
        BetaLambdaMapper().predict(val_state, forecast).expected_returns.astype(np.float32)
    )
    predictions[~np.isfinite(returns_val)] = np.nan

    centered = returns_train.astype(np.float64) - np.nanmean(returns_train, axis=0, keepdims=True)
    centered = np.where(np.isfinite(centered), centered, 0.0)
    total_variance = float(np.var(centered, axis=0, ddof=0).sum())
    factor_variance = np.var(train_state.factor_returns, axis=0, ddof=0)
    variance_ratio = (
        factor_variance / total_variance
        if total_variance > 0
        else np.zeros(n_factors, dtype=np.float64)
    )
    extras = {
        "n_factors": n_factors,
        "asset_mean": np.nanmean(returns_train, axis=0).tolist(),
        "factor_premium": np.nanmean(train_state.factor_returns, axis=0).tolist(),
        "loadings": train_state.asset_betas[0].tolist(),
        "explained_variance_ratio": variance_ratio.tolist(),
        "train_metrics": dict(fit.train_metrics),
    }
    return predictions, extras


def run_ipca_fold_with_library(
    chars_train: np.ndarray,
    returns_train: np.ndarray,
    chars_val: np.ndarray,
    returns_val: np.ndarray,
    *,
    n_factors: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    factor_ridge: float = 1e-6,
    gamma_ridge: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_batch = _cross_section_batch(chars_train, returns=returns_train)
    val_batch = _cross_section_batch(chars_val)

    model = IPCAModel(
        IPCAConfig(
            n_factors=n_factors,
            max_iter=max_iter,
            tol=tol,
            factor_ridge=factor_ridge,
            gamma_ridge=gamma_ridge,
        )
    )
    fit = model.fit(train_batch)
    train_state = model.extract(train_batch)
    val_state = model.extract(val_batch)
    forecaster = ExpandingMeanFactorForecaster()
    forecaster.fit(train_state)
    forecast = forecaster.predict(val_state)
    predictions = (
        BetaLambdaMapper().predict(val_state, forecast).expected_returns.astype(np.float32)
    )
    predictions[~np.isfinite(returns_val)] = np.nan

    extras = {
        "n_factors": n_factors,
        "n_instruments": int(chars_train.shape[2] + 1),
        "iterations": int(val_state.metadata.get("fit_iterations", 0)),
        "converged": bool(val_state.metadata.get("fit_converged", fit.converged)),
        "factor_premium": np.nanmean(train_state.factor_returns, axis=0).tolist(),
        "gamma": model.gamma.tolist(),
        "train_metrics": dict(fit.train_metrics),
    }
    return predictions, extras


def run_cae_fold_with_library(
    chars_train: np.ndarray,
    returns_train: np.ndarray,
    chars_val: np.ndarray,
    returns_val: np.ndarray,
    *,
    n_factors: int,
    factor_returns_train: np.ndarray | None = None,
    n_epochs: int = 50,
    checkpoint_interval: int | None = 5,
    checkpoint_epochs: list[int] | None = None,
    n_ensemble: int = 1,
    hidden_units: tuple[int, ...] = (32,),
    lambda_l1: float = 1e-4,
    batch_size: int = 10_000,
    lr: float = 1e-3,
    task_type: TaskType = "regression",
    seed: int = 42,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    train_batch = _cross_section_batch(
        chars_train,
        returns=returns_train,
        factor_returns=factor_returns_train,
    )
    val_batch = _cross_section_batch(chars_val, returns=returns_val)

    model = CAEModel(
        CAEConfig(
            n_factors=n_factors,
            task_type=task_type,
            hidden_units=hidden_units,
            n_ensemble=n_ensemble,
            n_epochs=n_epochs,
            checkpoint_interval=checkpoint_interval,
            checkpoint_epochs=tuple(checkpoint_epochs or ()),
            lr=lr,
            lambda_l1=lambda_l1,
            batch_size=batch_size,
            device=_PREFERRED_DEVICE,
            seed=seed,
        )
    )
    extras = _run_checkpointed_latent_pipeline(
        model=model,
        train_batch=train_batch,
        val_batch=val_batch,
        returns_val=returns_val,
        task_type=task_type,
    )
    extras["factor_source"] = (
        "continuous_returns" if factor_returns_train is not None else "label_column"
    )
    return extras.pop("checkpoint_predictions"), extras


def run_sae_fold_with_library(
    chars_train: np.ndarray,
    returns_train: np.ndarray,
    chars_val: np.ndarray,
    returns_val: np.ndarray,
    *,
    factor_returns_train: np.ndarray | None = None,
    n_epochs: int = 50,
    checkpoint_interval: int | None = 5,
    checkpoint_epochs: list[int] | None = None,
    lr: float = 1e-4,
    bottleneck_dim: int = 96,
    aux_hidden_dim: int = 96,
    main_hidden_units: list[int] | None = None,
    hidden_units: list[int] | None = None,
    dropout_rates: list[float] | None = None,
    noise_std: float = 0.035,
    alpha: float = 1.0,
    aux_weight: float = 1.0,
    task_type: TaskType = "regression",
    seed: int = 42,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    train_batch = _cross_section_batch(
        chars_train,
        returns=returns_train,
        factor_returns=factor_returns_train,
    )
    val_batch = _cross_section_batch(chars_val)

    model = SAEModel(
        SAEConfig(
            task_type=task_type,
            bottleneck_dim=bottleneck_dim,
            aux_hidden_dim=aux_hidden_dim,
            main_hidden_units=tuple(main_hidden_units or hidden_units or (896, 448, 448, 256)),
            dropout_rates=None if dropout_rates is None else tuple(dropout_rates),
            noise_std=noise_std,
            alpha=alpha,
            aux_weight=aux_weight,
            n_epochs=n_epochs,
            checkpoint_interval=checkpoint_interval,
            checkpoint_epochs=tuple(checkpoint_epochs or ()),
            lr=lr,
            device=_PREFERRED_DEVICE,
            seed=seed,
        )
    )
    extras = _run_checkpointed_signal_pipeline(
        model=model,
        train_batch=train_batch,
        val_batch=val_batch,
        returns_val=returns_val,
        task_type=task_type,
    )
    return extras.pop("checkpoint_predictions"), extras


def run_sdf_fold_with_library(
    chars_train: np.ndarray,
    returns_train: np.ndarray,
    chars_val: np.ndarray,
    returns_val: np.ndarray,
    *,
    macro_train: np.ndarray | None = None,
    macro_val: np.ndarray | None = None,
    state_dim_sdf: int = 4,
    state_dim_moment: int = 32,
    hidden_dim: int = 64,
    n_instruments: int = 8,
    dropout: float = 0.05,
    n_epochs_unc: int = 256,
    n_epochs_moment: int = 64,
    n_epochs_cond: int = 1024,
    checkpoint_interval: int | None = None,
    checkpoint_epochs: list[int] | None = None,
    beta_n_epochs: int = 256,
    beta_checkpoint_interval: int | None = None,
    beta_checkpoint_epochs: list[int] | None = None,
    beta_default_checkpoint: int | None = None,
    output_mode: str = "beta_network",
    expected_return_mapper: str = "linear",
    burn_in_epochs: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 42,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    train_batch = _cross_section_batch(
        chars_train,
        returns=returns_train,
        context_features=macro_train,
    )
    val_batch = _cross_section_batch(chars_val, returns=returns_val, context_features=macro_val)

    model = StochasticDiscountFactorModel(
        StochasticDiscountFactorConfig(
            output_mode="weights",
            state_dim_sdf=state_dim_sdf,
            state_dim_moment=state_dim_moment,
            hidden_dim=hidden_dim,
            n_instruments=n_instruments,
            dropout=dropout,
            n_epochs_unc=n_epochs_unc,
            n_epochs_moment=n_epochs_moment,
            n_epochs_cond=n_epochs_cond,
            checkpoint_interval=checkpoint_interval,
            checkpoint_epochs=tuple(checkpoint_epochs or ()),
            beta_n_epochs=beta_n_epochs,
            beta_checkpoint_interval=beta_checkpoint_interval,
            beta_checkpoint_epochs=tuple(beta_checkpoint_epochs or ()),
            beta_default_checkpoint=beta_default_checkpoint,
            expected_return_mapper=expected_return_mapper,
            burn_in_epochs=burn_in_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=_PREFERRED_DEVICE,
            seed=seed,
        )
    )
    fit = model.fit(train_batch, validation_batch=val_batch)

    checkpoint_predictions: dict[int, np.ndarray] = {}
    checkpoint_metrics: dict[str, dict[str, float | int | None]] = {}
    beta_head_epochs: dict[str, int | None] = {}

    for epoch in model.available_checkpoints:
        checkpoint_label = _sdf_checkpoint_label(epoch, n_epochs_unc=n_epochs_unc)
        train_state = model.extract(train_batch, checkpoint=epoch)
        val_state = model.extract(val_batch, checkpoint=epoch)
        if output_mode == "weights":
            predictions = val_state.asset_weights.astype(np.float32)
            beta_head_epochs[str(checkpoint_label)] = None
        elif output_mode == "beta_network":
            beta_head = StochasticDiscountFactorBetaNetworkHead(model.config)
            beta_fit = beta_head.fit(train_state, train_batch)
            predictions = beta_head.predict(val_batch).signal_values.astype(np.float32)
            beta_head_epochs[str(checkpoint_label)] = beta_fit.best_epoch
        elif output_mode == "expected_returns":
            mapper = LinearStochasticDiscountFactorReturnMapper()
            mapper.fit(train_state, train_batch)
            predictions = mapper.predict(val_state).expected_returns.astype(np.float32)
            beta_head_epochs[str(checkpoint_label)] = None
        else:
            raise ValueError(f"Unsupported output_mode: {output_mode!r}")
        predictions[~np.isfinite(returns_val)] = np.nan
        checkpoint_predictions[checkpoint_label] = predictions
        checkpoint_metrics[str(checkpoint_label)] = summarize_predictions(
            returns_val,
            predictions,
            task_type="regression",
        )

    extras = {
        "n_epochs_unc": n_epochs_unc,
        "n_epochs_moment": n_epochs_moment,
        "n_epochs_cond": n_epochs_cond,
        "checkpoint_epochs": [
            _sdf_checkpoint_label(epoch, n_epochs_unc=n_epochs_unc)
            for epoch in model.available_checkpoints
        ],
        "library_checkpoints": list(model.available_checkpoints),
        "beta_n_epochs": model.config.beta_n_epochs,
        "beta_checkpoint_epochs": list(model.config.beta_checkpoint_epochs),
        "beta_default_checkpoint": model.config.beta_default_checkpoint,
        "output_mode": output_mode,
        "expected_return_mapper": expected_return_mapper,
        "beta_head_best_epochs": beta_head_epochs,
        "checkpoint_metrics": checkpoint_metrics,
        "training_history": list(fit.history),
        "train_metrics": dict(fit.train_metrics),
        "sdf_sharpe": _latest_sdf_sharpe(fit.history),
    }
    return checkpoint_predictions, extras


def _run_checkpointed_latent_pipeline(
    *,
    model: CAEModel,
    train_batch: CrossSectionBatch,
    val_batch: CrossSectionBatch,
    returns_val: np.ndarray,
    task_type: TaskType,
) -> dict[str, Any]:
    fit = model.fit(train_batch, validation_batch=val_batch)

    checkpoint_predictions: dict[int, np.ndarray] = {}
    checkpoint_metrics: dict[str, dict[str, float | int | None]] = {}

    for epoch in model.available_checkpoints:
        train_state = model.extract(train_batch, checkpoint=epoch)
        val_state = model.extract(val_batch, checkpoint=epoch)
        forecaster = ExpandingMeanFactorForecaster()
        forecaster.fit(train_state)
        forecast = forecaster.predict(val_state)
        predictions = (
            BetaLambdaMapper().predict(val_state, forecast).expected_returns.astype(np.float32)
        )
        if task_type == "classification":
            predictions = _sigmoid(predictions).astype(np.float32)
        predictions[~np.isfinite(returns_val)] = np.nan
        checkpoint_predictions[int(epoch)] = predictions
        checkpoint_metrics[str(epoch)] = summarize_predictions(
            returns_val,
            predictions,
            task_type=task_type,
        )

    return {
        "n_epochs": int(model.config.n_epochs),
        "checkpoint_epochs": list(model.available_checkpoints),
        "task_type": task_type,
        "checkpoint_metrics": checkpoint_metrics,
        "train_history": list(fit.history),
        "train_metrics": dict(fit.train_metrics),
        "checkpoint_predictions": checkpoint_predictions,
    }


def _run_checkpointed_signal_pipeline(
    *,
    model: SAEModel,
    train_batch: CrossSectionBatch,
    val_batch: CrossSectionBatch,
    returns_val: np.ndarray,
    task_type: TaskType,
) -> dict[str, Any]:
    fit = model.fit(train_batch)

    checkpoint_predictions: dict[int, np.ndarray] = {}
    checkpoint_metrics: dict[str, dict[str, float | int | None]] = {}

    for epoch in model.available_checkpoints:
        predictions = model.predict(val_batch, checkpoint=epoch).signal_values.astype(np.float32)
        predictions[~np.isfinite(returns_val)] = np.nan
        checkpoint_predictions[int(epoch)] = predictions
        checkpoint_metrics[str(epoch)] = summarize_predictions(
            returns_val,
            predictions,
            task_type=task_type,
        )

    return {
        "n_epochs": int(model.config.n_epochs),
        "checkpoint_epochs": list(model.available_checkpoints),
        "task_type": task_type,
        "checkpoint_metrics": checkpoint_metrics,
        "train_history": list(fit.history),
        "train_metrics": dict(fit.train_metrics),
        "checkpoint_predictions": checkpoint_predictions,
    }


def _cross_section_batch(
    characteristics: np.ndarray,
    *,
    returns: np.ndarray | None = None,
    factor_returns: np.ndarray | None = None,
    context_features: np.ndarray | None = None,
) -> CrossSectionBatch:
    mask = np.isfinite(characteristics).all(axis=2)
    return CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        factor_returns=factor_returns,
        context_features=context_features,
        mask=mask,
        timestamps=tuple(range(characteristics.shape[0])),
        asset_ids=_asset_ids(characteristics.shape[1]),
    )


def _validate_persistent_returns(returns_train: np.ndarray, returns_val: np.ndarray) -> None:
    if returns_train.ndim != 2 or returns_val.ndim != 2:
        raise ValueError("returns_train and returns_val must be 2D")
    if returns_train.shape[1] != returns_val.shape[1]:
        raise ValueError("returns_train and returns_val must share the entity axis")


def _asset_ids(n_assets: int) -> tuple[str, ...]:
    return tuple(f"asset_{idx}" for idx in range(n_assets))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _sdf_checkpoint_label(checkpoint: tuple[str, int], *, n_epochs_unc: int) -> int:
    phase, epoch = checkpoint
    if checkpoint == ("conditional", -1):
        return 0
    if checkpoint == ("conditional", 0):
        return -1
    if checkpoint == ("unconditional", -1):
        return -2
    if checkpoint == ("unconditional", 0):
        return -3
    return int(epoch if phase == "unconditional" else n_epochs_unc + epoch)


def _latest_sdf_sharpe(history: tuple[dict[str, float | str], ...]) -> float | None:
    for entry in reversed(history):
        sharpe = entry.get("train_sharpe")
        if isinstance(sharpe, (int, float)) and np.isfinite(sharpe):
            return float(sharpe)
    return None
