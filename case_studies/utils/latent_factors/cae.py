"""Conditional autoencoder on dated cross-sections."""

from __future__ import annotations

from typing import Any

import numpy as np

from case_studies.utils.latent_factors.common import TaskType
from case_studies.utils.latent_factors.library_bridge import run_cae_fold_with_library


def run_cae_fold(
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
    n_ensemble: int = 1,
    hidden_units: tuple[int, ...] = (32,),
    lambda_l1: float = 1e-4,
    batch_size: int = 10_000,
    lr: float = 1e-3,
    task_type: TaskType = "regression",
    seed: int = 42,
    log_fn=print,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Train the CAE and emit forecasts from the requested checkpoint grid."""
    del log_fn
    return run_cae_fold_with_library(
        chars_train,
        returns_train,
        chars_val,
        returns_val,
        n_factors=n_factors,
        factor_returns_train=factor_returns_train,
        n_epochs=n_epochs,
        checkpoint_interval=checkpoint_interval,
        checkpoint_epochs=checkpoint_epochs,
        n_ensemble=n_ensemble,
        hidden_units=hidden_units,
        lambda_l1=lambda_l1,
        batch_size=batch_size,
        lr=lr,
        task_type=task_type,
        seed=seed,
    )
