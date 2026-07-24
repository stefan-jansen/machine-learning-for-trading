"""Configuration dataclasses for DeePM-style training and features."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Feature engineering hyperparameters.

    Defaults match the DeePM paper's feature definitions:
    1. Volatility-normalized returns for horizons h in {1, 21, 63, 252}.
    2. MACD filters at (S, L) in {(8, 24), (16, 48), (32, 96)}.
    3. Log-price z-scores for windows l in {21, 252}.
    4. Robust clipping using rolling median and MAD over 252 days.
    """

    vol_span: int = 63
    vol_eps: float = 1e-8

    ret_horizons: tuple[int, ...] = (1, 21, 63, 252)

    macd_pairs: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96))
    macd_renorm_window: int = 252

    zscore_windows: tuple[int, ...] = (21, 252)

    clip_window: int = 252
    clip_k: float = 5.0
    clip_mad_scale: float = 1.48

    include_existence: bool = True

    feature_set: Literal["signal", "raw_momentum"] = "signal"
    raw_momentum_max_horizon: int = 252


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Neural policy architecture hyperparameters."""

    d_model: int = 128
    n_heads: int = 4
    dropout: float = 0.4

    lstm_layers: int = 1
    temporal_mha_layers: int = 1

    cross_attention_heads: int = 4
    cross_attention_lag: int = 1

    macro_gnn_heads: int = 4

    asset_embedding_dim: int = 16
    group_embedding_dim: int = 8

    use_group_embedding: bool = True
    use_cost_in_context: bool = True

    vvsn_hidden_dim: int = 128
    adapter_hidden_mult: int = 4


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training and objective hyperparameters."""

    seq_len: int = 84
    burn_in: int = 21

    batch_size: int = 64

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    annualization_factor: float = 252.0
    sharpe_eps: float = 1e-8

    # Transaction cost scaling in Eq. (13)
    gamma_cost: float = 0.5

    # Robust objective (Eq. (31) and Eq. (33))
    softmin_tau: float = 0.2
    softmin_lambda: float = 0.1

    max_iters: int = 1000
    eval_every: int = 25

    # Validation metric smoothing and early stopping
    metric_ema_alpha: float = 0.45
    metric_min_delta: float = 0.001
    early_stopping_patience: int = 50
    early_stopping_burn_in_iters: int = 50

    seed: int = 7
    device: str = "cpu"


def validate_training_config(cfg: TrainingConfig) -> None:
    """Validate invariants required by the loss and dataset."""
    if cfg.seq_len <= 2:
        raise ValueError("seq_len must be > 2")
    if cfg.burn_in < 0 or cfg.burn_in >= cfg.seq_len - 1:
        raise ValueError("burn_in must satisfy 0 <= burn_in < seq_len - 1")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.softmin_tau <= 0.0:
        raise ValueError("softmin_tau must be positive")


def as_tuple(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)
