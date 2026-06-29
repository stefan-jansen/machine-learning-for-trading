"""Supervised autoencoder on dated cross-sections."""

from __future__ import annotations

from typing import Any

import numpy as np

from case_studies.utils.latent_factors.common import TaskType
from case_studies.utils.latent_factors.library_bridge import run_sae_fold_with_library


def run_sae_fold(
    chars_train: np.ndarray,
    returns_train: np.ndarray,
    chars_val: np.ndarray,
    returns_val: np.ndarray,
    n_factors: int,
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
    log_fn=print,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Train the SAE and emit predictions on the requested checkpoint grid."""
    # `n_factors` is part of the runner-API contract for parity with PCA/IPCA/CAE/SDF
    # but the SAE has no n_factors knob — `bottleneck_dim` plays that role.
    del log_fn, n_factors
    return run_sae_fold_with_library(
        chars_train,
        returns_train,
        chars_val,
        returns_val,
        factor_returns_train=factor_returns_train,
        n_epochs=n_epochs,
        checkpoint_interval=checkpoint_interval,
        checkpoint_epochs=checkpoint_epochs,
        lr=lr,
        bottleneck_dim=bottleneck_dim,
        aux_hidden_dim=aux_hidden_dim,
        main_hidden_units=main_hidden_units,
        hidden_units=hidden_units,
        dropout_rates=dropout_rates,
        noise_std=noise_std,
        alpha=alpha,
        aux_weight=aux_weight,
        task_type=task_type,
    )
