"""Supervised autoencoder on dated cross-sections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from case_studies.utils.latent_factors.common import TaskType
from case_studies.utils.latent_factors.library_bridge import run_sae_fold_with_library

# Bumped when a change to this module would change a fitted SAE result. It enters
# every sae training identity through `adapter._source_identity`, which declares behaviour
# rather than hashing these bytes - see that function for why.
SAE_RUNNER_VERSION = 2


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
    batch_size: int = 10_000,
    task_type: TaskType = "regression",
    seed: int = 42,
    device: str = "cpu",
    log_fn=print,
    artifact_dir: Path | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Train the SAE and emit predictions on the requested checkpoint grid.

    ``batch_size`` is declared here rather than left to the library, and it is the same
    default its sibling ``run_cae_fold`` carries. Omitting it is what this runner used to do,
    and ``SAEConfig.batch_size`` defaults to ``None``, which the training loop reads as one
    batch holding the entire training window - roughly a quarter of a million rows on a
    daily equity panel, which exhausts a 24 GB card. That was never a decision about gradient
    estimation; it was the one parameter of the pair that nobody passed. A case study that
    wants a different value declares it under ``modeling.latent_factors.model_kwargs.sae``,
    where it reaches the fit through the same route as every other model argument and is
    hashed into the training identity along with them.
    """
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
        batch_size=batch_size,
        task_type=task_type,
        seed=seed,
        device=device,
        artifact_dir=artifact_dir,
    )
