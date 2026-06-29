"""Data preparation utilities for latent factor case studies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from scipy import stats


def prepare_ragged_panel_data(
    dataset: pl.DataFrame,
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str,
    max_entities: int = 0,
    eval_label_col: str | None = None,
    macro_panel: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a dated cross-sectional panel with per-date observed assets only.

    The returned arrays are padded to the maximum cross-section size within the
    input window. The slot axis is date-local and does not imply stable entity
    identity across time.
    """
    df = _sort_panel_frame(dataset, date_col=date_col, entity_col=entity_col)
    if max_entities > 0:
        df = _limit_entities(df, entity_col=entity_col, max_entities=max_entities)

    groups = df.partition_by(date_col, maintain_order=True)
    if not groups:
        raise ValueError("Dataset produced no dated cross-sections")

    dates = [group[date_col][0] for group in groups]
    counts = np.asarray([group.height for group in groups], dtype=np.int32)
    n_dates = len(groups)
    n_slots = int(counts.max())
    n_features = len(feature_names)

    chars = np.full((n_dates, n_slots, n_features), np.nan, dtype=np.float32)
    returns = np.full((n_dates, n_slots), np.nan, dtype=np.float32)
    eval_returns = np.full((n_dates, n_slots), np.nan, dtype=np.float32) if eval_label_col else None
    entities = np.full((n_dates, n_slots), None, dtype=object)

    for date_idx, group in enumerate(groups):
        n_obs = group.height
        chars[date_idx, :n_obs] = group.select(feature_names).to_numpy().astype(np.float32)
        returns[date_idx, :n_obs] = (
            group.select(label_col).to_numpy().reshape(-1).astype(np.float32)
        )
        if eval_returns is not None:
            eval_returns[date_idx, :n_obs] = (
                group.select(eval_label_col).to_numpy().reshape(-1).astype(np.float32)
            )
        entities[date_idx, :n_obs] = np.asarray(group[entity_col].to_list(), dtype=object)

    macro = None
    macro_features: list[str] | None = None
    if macro_panel is not None:
        macro, macro_features = align_macro_to_dates(macro_panel, dates, date_col)

    return {
        "chars": chars,
        "returns": returns,
        "eval_returns": eval_returns,
        "dates": np.asarray(dates, dtype="datetime64[ns]"),
        "entities": entities,
        "counts": counts,
        "entity_col": entity_col,
        "macro": macro,
        "macro_features": macro_features,
    }


def prepare_panel_data(
    dataset: pl.DataFrame,
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str,
    max_entities: int = 0,
    min_coverage: float = 0.5,
    eval_label_col: str | None = None,
    macro_panel: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a persistent-entity panel for stable-ID models such as PCA."""
    df = _sort_panel_frame(dataset, date_col=date_col, entity_col=entity_col)

    n_dates_total = df[date_col].n_unique()
    min_dates = max(int(n_dates_total * min_coverage), 10)

    eligible = (
        df.group_by(entity_col)
        .len()
        .filter(pl.col("len") >= min_dates)
        .sort(["len", entity_col], descending=[True, False])
    )
    if max_entities > 0:
        eligible = eligible.head(max_entities)

    entities = sorted(eligible[entity_col].to_list())
    if not entities:
        raise ValueError("No entities met the persistent-panel coverage requirement")

    df = df.filter(pl.col(entity_col).is_in(entities)).sort(date_col, entity_col)
    dates = sorted(df[date_col].unique().to_list())

    n_dates = len(dates)
    n_entities = len(entities)
    n_features = len(feature_names)

    chars = np.full((n_dates, n_entities, n_features), np.nan, dtype=np.float32)
    returns = np.full((n_dates, n_entities), np.nan, dtype=np.float32)
    eval_returns = (
        np.full((n_dates, n_entities), np.nan, dtype=np.float32) if eval_label_col else None
    )
    entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}

    groups = df.partition_by(date_col, maintain_order=True)
    for date_idx, group in enumerate(groups):
        for row in group.iter_rows(named=True):
            entity_idx = entity_to_idx.get(row[entity_col])
            if entity_idx is None:
                continue
            chars[date_idx, entity_idx] = np.asarray(
                [row.get(feature, np.nan) for feature in feature_names],
                dtype=np.float32,
            )
            returns[date_idx, entity_idx] = np.float32(row.get(label_col, np.nan))
            if eval_returns is not None:
                eval_returns[date_idx, entity_idx] = np.float32(row.get(eval_label_col, np.nan))

    macro = None
    macro_features: list[str] | None = None
    if macro_panel is not None:
        macro, macro_features = align_macro_to_dates(macro_panel, dates, date_col)

    return {
        "chars": chars,
        "returns": returns,
        "eval_returns": eval_returns,
        "dates": np.asarray(dates, dtype="datetime64[ns]"),
        "entities": np.asarray(entities, dtype=object),
        "entity_col": entity_col,
        "macro": macro,
        "macro_features": macro_features,
    }


def rank_normalize_cross_section(chars: np.ndarray) -> np.ndarray:
    """Rank-normalize each date's characteristics to the [-0.5, 0.5] interval."""
    arr = np.asarray(chars, dtype=np.float32)
    original_ndim = arr.ndim
    if original_ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"chars must be 2D or 3D; got shape {arr.shape}")

    ranked = np.zeros_like(arr, dtype=np.float32)
    _, _, n_features = arr.shape

    for date_idx in range(arr.shape[0]):
        for feature_idx in range(n_features):
            values = arr[date_idx, :, feature_idx]
            valid = np.isfinite(values)
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            if n_valid == 1:
                ranked[date_idx, valid, feature_idx] = 0.0
                continue
            ranks = stats.rankdata(values[valid], method="average")
            ranked[date_idx, valid, feature_idx] = ((ranks - 1.0) / (n_valid - 1.0) - 0.5).astype(
                np.float32
            )

    return ranked[0] if original_ndim == 2 else ranked


def compute_managed_portfolios(
    chars: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """Compute diagonal characteristic-managed portfolios for each date."""
    if chars.ndim != 3:
        raise ValueError(f"chars must be 3D (T, N, L); got shape {chars.shape}")
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2D (T, N); got shape {returns.shape}")
    if chars.shape[:2] != returns.shape:
        raise ValueError(
            f"chars and returns disagree on (T, N): {chars.shape[:2]} vs {returns.shape}"
        )

    n_dates, n_slots, n_features = chars.shape
    ones = np.ones((n_dates, n_slots, 1), dtype=np.float32)
    chars_aug = np.concatenate([chars.astype(np.float32, copy=False), ones], axis=2)
    portfolios = np.zeros((n_dates, n_slots, n_features + 1), dtype=np.float32)
    eps = 1e-8

    for date_idx in range(n_dates):
        z_t = chars_aug[date_idx]
        r_t = returns[date_idx]
        valid = np.isfinite(r_t) & np.isfinite(z_t).all(axis=1)
        if not valid.any():
            continue
        z_valid = z_t[valid].astype(np.float64)
        r_valid = r_t[valid].astype(np.float64)
        numerator = (z_valid * r_valid[:, None]).sum(axis=0)
        denominator = (z_valid**2).sum(axis=0)
        x_t = numerator / np.maximum(denominator, eps)
        portfolios[date_idx] = np.broadcast_to(
            x_t.astype(np.float32)[None, :],
            (n_slots, n_features + 1),
        )

    return portfolios


def align_macro_to_dates(
    macro_panel: pl.DataFrame,
    dates: Sequence[object],
    date_col: str = "timestamp",
) -> tuple[np.ndarray, list[str]]:
    """Align macro features to case-study dates with backward as-of joins."""
    macro = macro_panel.clone()
    if hasattr(macro[date_col].dtype, "time_zone") and macro[date_col].dtype.time_zone is not None:
        macro = macro.with_columns(pl.col(date_col).dt.replace_time_zone(None))

    feature_cols = [column for column in macro.columns if column != date_col]
    if not feature_cols:
        return np.zeros((len(dates), 0), dtype=np.float32), []

    date_frame = pl.DataFrame({date_col: list(dates)}).sort(date_col)
    aligned = (
        date_frame.join_asof(macro.sort(date_col), on=date_col, strategy="backward")
        .fill_null(strategy="backward")
        .fill_null(strategy="forward")
    )
    return aligned.select(feature_cols).to_numpy().astype(np.float32), feature_cols


def _sort_panel_frame(
    dataset: pl.DataFrame,
    *,
    date_col: str,
    entity_col: str,
) -> pl.DataFrame:
    df = dataset.sort(date_col, entity_col)
    if hasattr(df[date_col].dtype, "time_zone") and df[date_col].dtype.time_zone is not None:
        df = df.with_columns(pl.col(date_col).dt.replace_time_zone(None))
    return df


def _limit_entities(
    dataset: pl.DataFrame,
    *,
    entity_col: str,
    max_entities: int,
) -> pl.DataFrame:
    top_entities = (
        dataset.group_by(entity_col)
        .len()
        .sort(["len", entity_col], descending=[True, False])
        .head(max_entities)[entity_col]
        .to_list()
    )
    return dataset.filter(pl.col(entity_col).is_in(top_entities))


__all__ = [
    "align_macro_to_dates",
    "compute_managed_portfolios",
    "prepare_panel_data",
    "prepare_ragged_panel_data",
    "rank_normalize_cross_section",
]
