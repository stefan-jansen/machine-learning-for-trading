"""Stochastic discount factor network on dated cross-sections."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from case_studies.utils.latent_factors.library_bridge import run_sdf_fold_with_library

SDFOutputMode = Literal["weights", "expected_returns", "beta_network"]


def run_sdf_fold(
    chars_train: np.ndarray,
    returns_train: np.ndarray,
    chars_val: np.ndarray,
    returns_val: np.ndarray,
    *,
    macro_train: np.ndarray | None = None,
    macro_val: np.ndarray | None = None,
    n_factors: int = 5,
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
    output_mode: SDFOutputMode = "beta_network",
    expected_return_mapper: str = "linear",
    burn_in_epochs: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    log_fn=print,
    seed: int = 42,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Train the SDF network and emit checkpoint predictions."""
    del n_factors, log_fn
    return run_sdf_fold_with_library(
        chars_train,
        returns_train,
        chars_val,
        returns_val,
        macro_train=macro_train,
        macro_val=macro_val,
        state_dim_sdf=state_dim_sdf,
        state_dim_moment=state_dim_moment,
        hidden_dim=hidden_dim,
        n_instruments=n_instruments,
        dropout=dropout,
        n_epochs_unc=n_epochs_unc,
        n_epochs_moment=n_epochs_moment,
        n_epochs_cond=n_epochs_cond,
        checkpoint_interval=checkpoint_interval,
        checkpoint_epochs=checkpoint_epochs,
        beta_n_epochs=beta_n_epochs,
        beta_checkpoint_interval=beta_checkpoint_interval,
        beta_checkpoint_epochs=beta_checkpoint_epochs,
        beta_default_checkpoint=beta_default_checkpoint,
        output_mode=output_mode,
        expected_return_mapper=expected_return_mapper,
        burn_in_epochs=burn_in_epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
    )
