"""Shared helpers for latent factor runners."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl
from ml4t.diagnostic.metrics import cross_sectional_ic

TaskType = Literal["regression", "classification"]


def resolve_checkpoint_epochs(
    max_epoch: int,
    *,
    checkpoint_interval: int | None = 5,
    checkpoint_epochs: Sequence[int] | None = None,
    include_final: bool = True,
) -> list[int]:
    """Resolve the checkpoint grid for a training run."""
    if max_epoch < 1:
        raise ValueError(f"max_epoch must be positive; got {max_epoch}")

    if checkpoint_epochs is not None:
        epochs = sorted({int(epoch) for epoch in checkpoint_epochs if 1 <= int(epoch) <= max_epoch})
        if not epochs:
            raise ValueError("checkpoint_epochs did not contain a valid epoch")
    elif checkpoint_interval is None or checkpoint_interval <= 0:
        epochs = [max_epoch]
    else:
        epochs = list(range(int(checkpoint_interval), max_epoch + 1, int(checkpoint_interval)))
        if not epochs:
            epochs = [max_epoch]

    if include_final and max_epoch not in epochs:
        epochs.append(max_epoch)
    return sorted(set(epochs))


def summarize_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    task_type: TaskType,
) -> dict[str, float | int | None]:
    """Summarize validation predictions with task-appropriate metrics."""
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    if not valid.any():
        if task_type == "classification":
            return {
                "n_validation_obs": 0,
                "validation_auc": None,
                "validation_log_loss": None,
            }
        return {
            "n_validation_obs": 0,
            "validation_mean_cs_ic": None,
        }

    if task_type == "classification":
        y_true_valid = y_true[valid]
        y_score_valid = y_score[valid]
        return {
            "n_validation_obs": int(valid.sum()),
            "validation_auc": _binary_auc(y_true_valid, y_score_valid),
            "validation_log_loss": _binary_log_loss(y_true_valid, y_score_valid),
        }

    return {
        "n_validation_obs": int(valid.sum()),
        "validation_mean_cs_ic": mean_cross_sectional_spearman(y_true, y_score),
    }


def mean_cross_sectional_spearman(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float | None:
    """Average cross-sectional Spearman IC across dates."""
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    frames: list[pl.DataFrame] = []
    asset_ids = [f"asset_{idx}" for idx in range(y_true.shape[1])]
    for date_idx in range(y_true.shape[0]):
        valid = np.isfinite(y_true[date_idx]) & np.isfinite(y_score[date_idx])
        if valid.sum() < 3:
            continue
        asset_idx = np.nonzero(valid)[0]
        frames.append(
            pl.DataFrame(
                {
                    "date": [date_idx] * int(valid.sum()),
                    "entity": [asset_ids[idx] for idx in asset_idx],
                    "prediction": y_score[date_idx, valid].astype(np.float64).tolist(),
                    "forward_return": y_true[date_idx, valid].astype(np.float64).tolist(),
                }
            )
        )

    if not frames:
        return None

    frame = pl.concat(frames)
    stats = cross_sectional_ic(
        predictions=frame.select(["date", "entity", "prediction"]),
        returns=frame.select(["date", "entity", "forward_return"]),
        pred_col="prediction",
        ret_col="forward_return",
        date_col="date",
        entity_col="entity",
        method="spearman",
        min_obs=3,
    )
    if int(stats["n_periods"]) == 0:
        return None
    return float(stats["ic_mean"])


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with stable tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = (np.asarray(y_true) > 0.5).astype(np.int8)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(np.asarray(y_score, dtype=np.float64))
    rank_sum_pos = float(ranks[y == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _binary_log_loss(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = (np.asarray(y_true) > 0.5).astype(np.float64)
    p = np.clip(np.asarray(y_score, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


__all__ = [
    "TaskType",
    "average_ranks",
    "mean_cross_sectional_spearman",
    "resolve_checkpoint_epochs",
    "summarize_predictions",
]
