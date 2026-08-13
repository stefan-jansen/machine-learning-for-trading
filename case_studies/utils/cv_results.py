"""Assemble cached and freshly trained CV results under one selection contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import polars as pl

from case_studies.utils.registry import (
    build_training_spec,
    compute_fold_metrics_from_predictions,
    load_prediction_metrics,
    load_prediction_sets,
    read_predictions,
    training_hash_from_spec,
)


def _as_frame(value: pl.DataFrame | Iterable[dict[str, Any]] | None) -> pl.DataFrame:
    if value is None:
        return pl.DataFrame()
    if isinstance(value, pl.DataFrame):
        return value
    rows = list(value)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _concat_frames(frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    populated = [frame for frame in frames if frame.height > 0]
    return pl.concat(populated, how="diagonal_relaxed") if populated else pl.DataFrame()


def assemble_cv_result(
    curves: pl.DataFrame | Iterable[dict[str, Any]],
    all_predictions: pl.DataFrame,
    *,
    date_col: str,
    entity_col: str,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
    training_log: pl.DataFrame | Iterable[dict[str, Any]] | None = None,
    require_full_coverage: bool = True,
) -> dict[str, Any]:
    """Build the common CV return value and select only comparable checkpoints.

    ``ic_n_days`` is the comparability gate. When it is available, the winner is
    selected only among checkpoints covering the maximum number of decision dates.
    This prevents a checkpoint with missing cross-sections from winning on an easier
    subset. Older registries without ``ic_n_days`` retain their historical raw-IC
    selection behavior.
    """
    curve_df = _as_frame(curves)
    if curve_df.is_empty():
        raise ValueError("Cannot assemble CV results without checkpoint metrics.")
    if all_predictions.is_empty():
        raise ValueError("Cannot assemble CV results without prediction rows.")

    required = {"config", "epoch", "ic_mean"}
    missing = required.difference(curve_df.columns)
    if missing:
        raise ValueError(f"Checkpoint metrics are missing required columns: {sorted(missing)}")

    if "ic_std" not in curve_df.columns:
        curve_df = curve_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("ic_std"))
    if "ic_n_days" not in curve_df.columns:
        curve_df = curve_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("ic_n_days"))
    curve_df = curve_df.with_columns(pl.col("epoch").cast(pl.Int64))

    prediction_coverage = pl.DataFrame()
    prediction_keys = {"config", "epoch", "y_score"}
    if all_predictions.height > 0 and prediction_keys.issubset(all_predictions.columns):
        prediction_coverage = (
            all_predictions.with_columns(pl.col("epoch").cast(pl.Int64))
            .group_by(["config", "epoch"])
            .agg(
                (
                    pl.col("y_score").is_null().sum()
                    + pl.col("y_score").is_nan().sum()
                    + pl.col("y_score").is_infinite().sum()
                ).alias("_prediction_n_invalid"),
                pl.lit(True).alias("_has_prediction_rows"),
            )
        )

    if "n_invalid" not in curve_df.columns:
        curve_df = curve_df.with_columns(pl.lit(0, dtype=pl.Int64).alias("n_invalid"))
    if prediction_coverage.height > 0:
        curve_df = curve_df.join(prediction_coverage, on=["config", "epoch"], how="left")
        curve_df = curve_df.with_columns(
            pl.col("_prediction_n_invalid").fill_null(pl.col("n_invalid")).alias("n_invalid"),
            pl.col("_has_prediction_rows").fill_null(False),
        ).drop("_prediction_n_invalid")

    curve_df = curve_df.with_columns(
        pl.col("epoch").cast(pl.Int64),
        pl.col("ic_mean").cast(pl.Float64),
        pl.col("ic_std").cast(pl.Float64),
        pl.col("ic_n_days").cast(pl.Float64),
        pl.col("n_invalid").cast(pl.Int64),
    ).sort(["config", "epoch"])

    eligibility = pl.col("ic_mean").is_finite() & (pl.col("n_invalid") == 0)
    if "_has_prediction_rows" in curve_df.columns:
        eligibility &= pl.col("_has_prediction_rows")
    eligible = curve_df.filter(eligibility)
    full_coverage_days: float | None = None
    if require_full_coverage and not eligible.is_empty():
        finite_days = eligible.filter(pl.col("ic_n_days").is_finite())
        if finite_days.height == eligible.height:
            full_coverage_days = float(cast(int | float, finite_days["ic_n_days"].max()))
            eligible = finite_days.filter(pl.col("ic_n_days") == full_coverage_days)
        elif not finite_days.is_empty():
            raise ValueError(
                "Checkpoint coverage is missing for part of the candidate set; "
                "cannot compare cached and fresh results safely."
            )

    if eligible.is_empty():
        # Report the scored-day counts: ic_n_days == 0 for every checkpoint means no
        # decision date reached the cross-sectional IC min_obs threshold, which is a
        # universe-too-small problem rather than a training failure.
        scored = curve_df.select("config", "epoch", "ic_mean", "ic_n_days").to_dicts()
        raise ValueError(
            "No finite, full-coverage checkpoint is available for selection. "
            f"Considered {len(scored)} checkpoint(s): {scored}"
        )

    metadata = metadata or {}
    grid: list[dict[str, Any]] = []
    for config_name in eligible["config"].unique(maintain_order=True).to_list():
        config_rows = eligible.filter(pl.col("config") == config_name).sort(
            "ic_mean", descending=True
        )
        best = config_rows.row(0, named=True)
        config_meta = metadata.get(config_name, {})
        selected_predictions = pl.DataFrame()
        if all_predictions.height > 0 and {"config", "epoch"}.issubset(all_predictions.columns):
            selected_predictions = all_predictions.filter(
                (pl.col("config") == config_name) & (pl.col("epoch") == int(best["epoch"]))
            )
        if "fold_id" in selected_predictions.columns:
            n_folds = int(selected_predictions["fold_id"].n_unique())
        else:
            n_folds = int(config_meta.get("n_folds") or 0)
        grid_row = dict(config_meta)
        grid_row.update(
            {
                "config_name": config_name,
                "best_epoch": int(best["epoch"]),
                "best_ic": float(best["ic_mean"]),
                "ic_n_days": (
                    float(best["ic_n_days"]) if best["ic_n_days"] is not None else float("nan")
                ),
                "n_invalid": int(best["n_invalid"]),
                "n_folds": n_folds,
                "selectable": True,
                "elapsed_s": float(config_meta.get("elapsed_s") or 0.0),
                "started_at": config_meta.get("started_at"),
            }
        )
        grid.append(grid_row)

    grid.sort(key=lambda row: row["best_ic"], reverse=True)
    best = grid[0]

    predictions = pl.DataFrame()
    if all_predictions.height > 0 and {"config", "epoch"}.issubset(all_predictions.columns):
        predictions = all_predictions.filter(
            (pl.col("config") == best["config_name"]) & (pl.col("epoch") == best["best_epoch"])
        )
        if predictions.height > 0:
            predictions = predictions.with_columns(
                pl.lit(best["config_name"]).alias("model_id")
            ).drop("config", "epoch")

    return {
        "grid_results": grid,
        "best_config_name": best["config_name"],
        "best_epoch": best["best_epoch"],
        "best_ic": best["best_ic"],
        "predictions": predictions,
        "all_predictions": all_predictions,
        "fold_metrics": compute_fold_metrics_from_predictions(
            all_predictions,
            best["config_name"],
            best["best_epoch"],
            date_col=date_col,
            entity_col=entity_col,
        ),
        "all_learning_curves": curve_df.drop("_has_prediction_rows", strict=False),
        "training_log": _as_frame(training_log),
        "full_coverage_days": full_coverage_days,
    }


def rebuild_cv_result_from_registry(
    case_study: str,
    configs: Sequence[dict[str, Any]],
    *,
    label_col: str,
    n_folds: int,
    prediction_split: str,
    date_col: str,
    entity_col: str,
    require_full_coverage: bool = True,
    training_specs: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rebuild the training-path result contract without training or writes."""
    curves: list[dict[str, Any]] = []
    prediction_frames: list[pl.DataFrame] = []
    missing_configs: list[str] = []

    for cfg in configs:
        spec = (
            training_specs[cfg["config_name"]]
            if training_specs is not None
            else build_training_spec(
                cfg["family"],
                cfg["config_name"],
                label_col,
                n_folds=n_folds,
                n_epochs=cfg.get("n_epochs"),
            )
        )
        training_hash = training_hash_from_spec(spec)
        prediction_sets = load_prediction_sets(
            case_study,
            training_hash=training_hash,
            split=prediction_split,
        )
        if prediction_sets.is_empty():
            missing_configs.append(cfg["config_name"])
            continue

        prediction_sets = prediction_sets.sort(["checkpoint_value", "prediction_hash"]).unique(
            "checkpoint_value", keep="first", maintain_order=True
        )

        config_has_metrics = False
        for prediction_set in prediction_sets.iter_rows(named=True):
            prediction_hash = prediction_set["prediction_hash"]
            metrics = load_prediction_metrics(case_study, prediction_hash=prediction_hash)
            if metrics.is_empty() or metrics["ic_mean"][0] is None:
                continue
            config_has_metrics = True
            checkpoint = int(prediction_set["checkpoint_value"] or 0)
            curves.append(
                {
                    "config": cfg["config_name"],
                    "epoch": checkpoint,
                    "ic_mean": float(metrics["ic_mean"][0]),
                    "ic_std": (
                        float(metrics["ic_std"][0])
                        if "ic_std" in metrics.columns and metrics["ic_std"][0] is not None
                        else float("nan")
                    ),
                    "ic_n_days": (
                        float(metrics["ic_n_days"][0])
                        if "ic_n_days" in metrics.columns and metrics["ic_n_days"][0] is not None
                        else float("nan")
                    ),
                }
            )
            predictions = read_predictions(case_study, prediction_hash)
            if predictions.height > 0:
                prediction_frames.append(
                    predictions.with_columns(
                        pl.lit(cfg["config_name"]).alias("config"),
                        pl.lit(checkpoint).alias("epoch"),
                    )
                )
        if not config_has_metrics:
            missing_configs.append(cfg["config_name"])

    if missing_configs:
        names = ", ".join(sorted(set(missing_configs)))
        raise ValueError(
            f"Registered CV state is incomplete for {case_study}/{prediction_split}: {names}"
        )

    return assemble_cv_result(
        curves,
        _concat_frames(prediction_frames),
        date_col=date_col,
        entity_col=entity_col,
        require_full_coverage=require_full_coverage,
    )


def combine_cv_results(
    results: Sequence[dict[str, Any]],
    *,
    date_col: str,
    entity_col: str,
    require_full_coverage: bool = True,
) -> dict[str, Any]:
    """Combine disjoint cached and fresh results into one comparable leaderboard."""
    metadata: dict[str, dict[str, Any]] = {}
    for result in results:
        for row in result.get("grid_results", []):
            metadata[row["config_name"]] = row

    return assemble_cv_result(
        _concat_frames(_as_frame(result.get("all_learning_curves")) for result in results),
        _concat_frames(_as_frame(result.get("all_predictions")) for result in results),
        date_col=date_col,
        entity_col=entity_col,
        metadata=metadata,
        training_log=_concat_frames(_as_frame(result.get("training_log")) for result in results),
        require_full_coverage=require_full_coverage,
    )
