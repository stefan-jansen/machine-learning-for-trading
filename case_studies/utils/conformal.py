"""Per-prediction split-conformal widths for position sizing.

Validation widths follow the strict ``walk_forward_v2`` contract. The first
chronological validation fold has no conformal allocation. Every later fold
uses residuals from earlier validation folds only. An entity with insufficient
prior residuals receives a pooled width from that same prior out-of-sample
calibration pool, so allocation never changes the selected basket by silently
dropping an uncalibrated entity.

Storage: alongside ``predictions.parquet`` in the same prediction-hash directory.
Writes are alpha-aware: a new alpha is appended to any existing
``conformal_widths.parquet`` (rows for the same alpha are replaced), so the
single artifact can carry multiple alphas. The output always uses ``symbol``
as the entity column (matching ``backtest_loaders.load_predictions_for_backtest``'s
normalization), regardless of whether the source predictions.parquet uses
``product`` or ``stock_id``.
"""

from __future__ import annotations

import copy
import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from utils.paths import get_case_study_dir

ID_COLS: tuple[str, ...] = ("symbol", "product")

# Legacy → canonical column rename map. Older prediction parquets (pre-IC
# unification) use {fold, prediction, actual}; newer use {fold_id, y_score, y_true}.
_LEGACY_RENAME: dict[str, str] = {
    "fold": "fold_id",
    "prediction": "y_score",
    "actual": "y_true",
}

DEFAULT_ALPHA: float = 0.20
DEFAULT_MIN_CALIBRATION_N: int = 30
CALIBRATION_VERSION: str = "walk_forward_v2"
POOLED_FALLBACK: str = "pooled_prior_oos"

HOLDOUT_CONFORMAL_EMBARGO_STEPS: dict[str, int] = {
    "etfs/fwd_ret_21d": 21,
    "cme_futures/fwd_ret_5d": 5,
    "cme_futures/fwd_ret_21d": 21,
    "fx_pairs/fwd_ret_1d": 1,
    "fx_pairs/fwd_ret_5d": 5,
    "fx_pairs/fwd_ret_21d": 21,
    "crypto_perps_funding/fwd_ret_24h": 3,
    "crypto_perps_funding/fwd_ret_8h": 1,
    "nasdaq100_microstructure/fwd_ret_15m": 1,
    "nasdaq100_microstructure/fwd_ret_60m": 4,
    "nasdaq100_microstructure/fwd_ret_5m": 1,
    "sp500_equity_option_analytics/fwd_ret_5d": 5,
    "sp500_equity_option_analytics/fwd_ret_risk_adj_5d": 5,
    "us_equities_panel/fwd_ret_5d": 5,
    "us_equities_panel/fwd_ret_1d": 1,
    "us_equities_panel/fwd_ret_21d": 21,
    "us_firm_characteristics/fwd_ret_1m_win": 1,
    "us_firm_characteristics/fwd_ret_1m": 1,
    "us_firm_characteristics/fwd_class_1m": 1,
}


def split_conformal_coverage(
    predictions: pl.DataFrame,
    *,
    levels: tuple[float, ...] = (0.80, 0.90, 0.95),
    min_rows: int = 30,
) -> list[dict[str, float | int]]:
    """Measure split-conformal coverage with chronological calibration.

    The earliest validation fold supplies both the absolute-residual quantile
    and the target scale. Later folds are evaluation data only. Quantiles use
    the exact finite-sample order statistic rather than interpolation.
    """
    renames = {
        legacy: canonical
        for legacy, canonical in _LEGACY_RENAME.items()
        if legacy in predictions.columns and canonical not in predictions.columns
    }
    if renames:
        predictions = predictions.rename(renames)

    required = {"timestamp", "fold_id", "y_true", "y_score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction artifact missing {sorted(missing)}")
    predictions = predictions.drop_nulls(required).with_columns(
        (pl.col("y_true") - pl.col("y_score")).abs().alias("abs_resid")
    )
    if predictions.is_empty():
        raise ValueError("prediction artifact has no finite predictions")
    for column in ("y_true", "y_score", "abs_resid"):
        if not np.isfinite(predictions[column].to_numpy()).all():
            raise ValueError(f"prediction artifact has non-finite {column} values")

    fold_windows = (
        predictions.group_by("fold_id")
        .agg(pl.col("timestamp").min().alias("validation_start"))
        .sort("validation_start")
    )
    if fold_windows.height < 2:
        raise ValueError("conformal coverage requires at least two validation folds")
    calibration_fold = fold_windows["fold_id"][0]
    calibration = predictions.filter(pl.col("fold_id") == calibration_fold)
    test = predictions.filter(pl.col("fold_id") != calibration_fold)
    if calibration.height < min_rows or test.height < min_rows:
        raise ValueError(f"conformal calibration or test panel has fewer than {min_rows} rows")

    scale = float(calibration["y_true"].std() or 0.0)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("conformal calibration target scale is invalid")
    calibration_residuals = np.sort(calibration["abs_resid"].to_numpy())
    test_residuals = test["abs_resid"].to_numpy()
    n_calibration = len(calibration_residuals)

    rows: list[dict[str, float | int]] = []
    for level in levels:
        if not 0 < level < 1:
            raise ValueError(f"invalid conformal level: {level}")
        rank = math.ceil((n_calibration + 1) * level)
        quantile = float("inf") if rank > n_calibration else float(calibration_residuals[rank - 1])
        rows.append(
            {
                "nominal_level": float(level),
                "empirical_coverage": float((test_residuals <= quantile).mean()),
                "mean_interval_width_frac_std": float((2.0 * quantile) / scale),
                "n_test": int(len(test_residuals)),
            }
        )
    return rows


def holdout_conformal_embargo_steps(case_study: str, label: str) -> int:
    """Return the reviewed label horizon in prediction data steps."""
    key = f"{case_study}/{label}"
    try:
        return HOLDOUT_CONFORMAL_EMBARGO_STEPS[key]
    except KeyError as error:
        raise KeyError(
            f"No conformal holdout embargo is defined for {key}; add a reviewed data-step value"
        ) from error


def ensure_conformal_calibration_identity(strategy_spec: dict[str, Any]) -> dict[str, Any]:
    """Return a spec whose conformal allocation carries its full identity."""
    spec = copy.deepcopy(strategy_spec)
    strategy = spec.get("strategy")
    if not isinstance(strategy, dict):
        return spec
    allocation = strategy.get("allocation")
    if not isinstance(allocation, dict) or allocation.get("method") != "conformal_weighted":
        return spec

    requested = allocation.get("calibration_version")
    if requested not in (None, CALIBRATION_VERSION):
        raise ValueError(
            f"Unsupported conformal calibration_version={requested!r}; "
            f"expected {CALIBRATION_VERSION!r}"
        )
    allocation["calibration_version"] = CALIBRATION_VERSION
    allocation.setdefault("min_calibration_n", DEFAULT_MIN_CALIBRATION_N)
    allocation.setdefault("sparse_fallback", POOLED_FALLBACK)
    return spec


def _detect_id_col(columns: list[str]) -> str:
    for c in ID_COLS:
        if c in columns:
            return c
    raise ValueError(
        f"predictions.parquet has no canonical entity column "
        f"(expected one of {ID_COLS}); found {columns}"
    )


def _predictions_dir(
    case_study: str,
    prediction_hash: str,
    *,
    case_dir: Path | None = None,
) -> Path:
    resolved_case_dir = case_dir or get_case_study_dir(case_study)
    return resolved_case_dir / "run_log" / "predictions" / prediction_hash


def _write_widths(
    path: Path,
    new_widths: pl.DataFrame,
    alpha: float,
    *,
    immutable: bool = False,
) -> None:
    """Persist widths to ``path``, merging by alpha.

    If ``path`` already exists, rows with the same ``alpha`` are dropped and
    replaced by ``new_widths``; rows with other alphas are preserved. This
    keeps a single file able to carry multiple alphas, which matches what
    ``load_conformal_widths`` expects when filtering on ``alpha``.
    """
    merged = new_widths
    if path.exists():
        # Tolerate a partially-written file from a concurrent worker — the
        # parallel sweep can race two workers onto the same prediction_hash
        # when both auto-generate widths via load_conformal_widths(). Treat an
        # unreadable existing file as "no prior widths" and overwrite.
        try:
            existing = pl.read_parquet(path)
        except (pl.exceptions.ComputeError, pl.exceptions.NoDataError, OSError, EOFError):
            if immutable:
                raise ValueError(f"locked conformal artifact is unreadable: {path}") from None
            # A zero-byte or missing-magic-bytes file from a half-finished
            # concurrent write surfaces as NoDataError/OSError, not just
            # ComputeError — treat any unreadable file as "no prior widths".
            existing = None
        if existing is not None:
            if "calibration_version" not in existing.columns:
                raise ValueError(
                    f"Legacy conformal artifact at {path}; preserve it in the pre-fix "
                    "snapshot and remove it from the live candidate before regeneration"
                )
            versions = set(existing["calibration_version"].unique().to_list())
            if versions != {CALIBRATION_VERSION}:
                raise ValueError(
                    f"Refusing to mix conformal calibration versions in {path}: {versions}"
                )
            # Float equality on alpha is fine here: we write Float64 and read
            # back Float64; both sides round-trip bit-identically through parquet.
            same_alpha = existing.filter(pl.col("alpha") == alpha)
            if immutable and not same_alpha.is_empty():
                from case_studies.utils.artifact_digest import value_digest

                if value_digest(same_alpha) != value_digest(new_widths):
                    raise ValueError(f"locked conformal artifact conflicts with {path}")
                return
            keep = existing.filter(pl.col("alpha") != alpha)
            merged = pl.concat([keep, new_widths], how="diagonal_relaxed")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        merged.write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compute_conformal_widths(
    case_study: str,
    prediction_hash: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_calibration_n: int = DEFAULT_MIN_CALIBRATION_N,
    write: bool = True,
    case_dir: Path | None = None,
) -> pl.DataFrame:
    """Compute and optionally persist strict walk-forward conformal widths.

    Returns one row per (timestamp, entity) for which a width could be
    calibrated: columns ``[timestamp, <id_col>, fold_id, width, alpha,
    calibration_n]``. Width = 2·q_{1-α}(|y_true − y_score|) on prior-fold
    residuals for that entity. The first chronological fold emits no widths.
    Later folds use earlier validation folds only. Entities below
    ``min_calibration_n`` use a pooled quantile from the same prior-only pool.

    Writes are alpha-aware (see module docstring): an existing
    ``conformal_widths.parquet`` for the same prediction hash retains rows
    at other alphas; rows at this ``alpha`` are replaced.

    Raises ``ValueError`` when:
      * predictions.parquet is missing or has < 2 fold ids (degenerate
        ``fold_id``); the function requires at least two folds;
      * no fold yields any entity meeting ``min_calibration_n`` after the
        prior-fold (or fallback) filter.
    """
    pred_dir = _predictions_dir(case_study, prediction_hash, case_dir=case_dir)
    pred_path = pred_dir / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.parquet not found: {pred_path}")

    preds = pl.read_parquet(pred_path)
    legacy_present = {k: v for k, v in _LEGACY_RENAME.items() if k in preds.columns}
    if legacy_present:
        preds = preds.rename(legacy_present)
    src_id_col = _detect_id_col(preds.columns)
    # Canonical: emit widths keyed by "symbol", matching backtest_loaders normalization.
    if src_id_col != "symbol":
        preds = preds.rename({src_id_col: "symbol"})
    id_col = "symbol"

    required = {"timestamp", id_col, "y_true", "y_score", "fold_id"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(
            f"{case_study}/{prediction_hash}: predictions.parquet missing "
            f"columns {sorted(missing)}; got {preds.columns}"
        )

    preds = preds.filter(
        pl.col("y_true").is_not_null() & pl.col("y_score").is_not_null()
    ).with_columns(abs_resid=(pl.col("y_true") - pl.col("y_score")).abs())

    folds = sorted(preds["fold_id"].unique().to_list())
    if not folds or len(folds) < 2:
        raise ValueError(
            f"{case_study}/{prediction_hash}: degenerate fold_id "
            f"(n_folds={len(folds)}); expanding-window calibration needs ≥2 folds"
        )

    # Fold IDs are NOT reliably chronological across case studies. Some CV
    # schemes label the latest fold as fold_id=0 (nasdaq100, crypto, …) while
    # others label it as the highest fold_id (us_equities_panel fwd_ret_5d/21d).
    # Derive walk-forward order from each fold's earliest timestamp instead.
    fold_ts = preds.group_by("fold_id").agg(ts_min=pl.col("timestamp").min()).sort("ts_min")
    fold_chronological = fold_ts["fold_id"].to_list()
    fold_widths_rows: list[pl.DataFrame] = []
    for i, k in enumerate(fold_chronological):
        prior = fold_chronological[:i]
        if not prior:
            continue
        cal = preds.filter(pl.col("fold_id").is_in(prior))
        if cal.is_empty():
            continue
        target_symbols = preds.filter(pl.col("fold_id") == k).select(id_col).unique()
        per_symbol = (
            cal.group_by(id_col)
            .agg(
                q=pl.col("abs_resid").quantile(1.0 - alpha, interpolation="higher"),
                calibration_n=pl.len(),
            )
            .filter(pl.col("calibration_n") >= min_calibration_n)
            .with_columns(
                fold_id=pl.lit(k, dtype=pl.Int64),
                width=2.0 * pl.col("q"),
                alpha=pl.lit(alpha, dtype=pl.Float64),
                calibration_scope=pl.lit("symbol"),
                calibration_version=pl.lit(CALIBRATION_VERSION),
            )
            .select(
                id_col,
                "fold_id",
                "width",
                "alpha",
                "calibration_n",
                "calibration_scope",
                "calibration_version",
            )
        )
        missing_symbols = target_symbols.join(per_symbol.select(id_col), on=id_col, how="anti")
        if not missing_symbols.is_empty() and cal.height >= min_calibration_n:
            pooled_q = cal["abs_resid"].quantile(1.0 - alpha, interpolation="higher")
            if pooled_q is None:
                raise ValueError(f"{case_study}/{prediction_hash}: pooled quantile is undefined")
            pooled = missing_symbols.with_columns(
                fold_id=pl.lit(k),
                width=pl.lit(2.0 * float(pooled_q), dtype=pl.Float64),
                alpha=pl.lit(alpha, dtype=pl.Float64),
                calibration_n=pl.lit(cal.height, dtype=pl.UInt32),
                calibration_scope=pl.lit("pooled"),
                calibration_version=pl.lit(CALIBRATION_VERSION),
            ).select(per_symbol.columns)
            per_symbol = pl.concat([per_symbol, pooled], how="vertical_relaxed")
        if not per_symbol.is_empty():
            fold_widths_rows.append(per_symbol)

    if not fold_widths_rows:
        raise ValueError(
            f"{case_study}/{prediction_hash}: no fold had prior-fold "
            f"calibration data after applying min_calibration_n={min_calibration_n}"
        )

    fold_widths = pl.concat(fold_widths_rows)

    timestamps = preds.select("timestamp", id_col, "fold_id").unique()
    widths = (
        timestamps.join(fold_widths, on=[id_col, "fold_id"], how="inner")
        .select(
            "timestamp",
            id_col,
            "fold_id",
            "width",
            "alpha",
            "calibration_n",
            "calibration_scope",
            "calibration_version",
        )
        .sort("timestamp", id_col)
    )

    if write:
        # `pred_dir` honours the `case_dir` override the predictions were read
        # from; recomputing it without that override would write the widths into
        # an unrelated run log.
        _write_widths(pred_dir / "conformal_widths.parquet", widths, alpha)

    return widths


def compute_holdout_conformal_widths(
    case_study: str,
    val_prediction_hash: str,
    holdout_prediction_hash: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_calibration_n: int = DEFAULT_MIN_CALIBRATION_N,
    embargo_steps: int = 0,
    write: bool = True,
    immutable: bool = False,
) -> pl.DataFrame:
    """Pooled per-symbol split-conformal widths for the holdout window.

    Calibration set: all validation residuals for the val prediction set,
    pooled across folds within each symbol. Prediction set: every
    (timestamp, symbol) pair in the holdout predictions parquet.

    Per-symbol pooled q_{1-α}(|y_true - y_score|) is broadcast across the
    holdout window for that symbol. Symbols with fewer than
    ``min_calibration_n`` validation residuals receive the pooled width from
    all embargoed validation residuals.

    ``embargo_steps`` drops the trailing ``embargo_steps`` distinct val
    timestamps from the calibration set. Required when the label has a
    non-zero forward-return horizon ``h``: a residual at val timestamp ``t``
    depends on returns realized over ``(t, t+h]``; if ``t+h`` falls inside
    the holdout window, the residual leaks holdout-period price information
    into the calibration. Set this to the label's horizon expressed in
    data-step units — e.g. ``21`` for ``fwd_ret_21d`` on a daily trading
    calendar; ``3`` for ``fwd_ret_24h`` on 8-hourly crypto data; ``1`` for
    ``fwd_ret_15m`` on 15-minute bars.

    Output schema matches ``compute_conformal_widths``'s val output:
    ``[timestamp, symbol, fold_id, width, alpha, calibration_n]`` with
    ``fold_id = -1`` as a sentinel meaning "holdout, no fold partition".
    """
    val_dir = _predictions_dir(case_study, val_prediction_hash)
    val_path = val_dir / "predictions.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"val predictions.parquet not found: {val_path}")

    val_preds = pl.read_parquet(val_path)
    legacy_val = {k: v for k, v in _LEGACY_RENAME.items() if k in val_preds.columns}
    if legacy_val:
        val_preds = val_preds.rename(legacy_val)
    src_id_val = _detect_id_col(val_preds.columns)
    if src_id_val != "symbol":
        val_preds = val_preds.rename({src_id_val: "symbol"})

    required = {"timestamp", "symbol", "y_true", "y_score"}
    missing = required - set(val_preds.columns)
    if missing:
        raise ValueError(
            f"{case_study}/{val_prediction_hash}: val predictions.parquet missing "
            f"columns {sorted(missing)}; got {val_preds.columns}"
        )

    val_preds = val_preds.filter(
        pl.col("y_true").is_not_null() & pl.col("y_score").is_not_null()
    ).with_columns(abs_resid=(pl.col("y_true") - pl.col("y_score")).abs())

    if embargo_steps > 0:
        unique_ts = sorted(val_preds.select("timestamp").unique().to_series().to_list())
        if len(unique_ts) <= embargo_steps:
            raise ValueError(
                f"{case_study}/{val_prediction_hash}: embargo_steps={embargo_steps} "
                f">= n_val_timestamps={len(unique_ts)}; no calibration data left "
                f"after embargo"
            )
        cutoff_ts = unique_ts[-embargo_steps - 1]
        val_preds = val_preds.filter(pl.col("timestamp") <= cutoff_ts)

    per_symbol_widths = (
        val_preds.group_by("symbol")
        .agg(
            q=pl.col("abs_resid").quantile(1.0 - alpha, interpolation="higher"),
            calibration_n=pl.len(),
        )
        .filter(pl.col("calibration_n") >= min_calibration_n)
        .with_columns(
            width=2.0 * pl.col("q"),
            alpha=pl.lit(alpha, dtype=pl.Float64),
            calibration_scope=pl.lit("symbol"),
            calibration_version=pl.lit(CALIBRATION_VERSION),
        )
        .select(
            "symbol",
            "width",
            "alpha",
            "calibration_n",
            "calibration_scope",
            "calibration_version",
        )
    )

    if val_preds.height < min_calibration_n:
        raise ValueError(
            f"{case_study}/{val_prediction_hash}: only {val_preds.height} embargoed "
            f"validation residuals; need at least {min_calibration_n}"
        )

    ho_dir = _predictions_dir(case_study, holdout_prediction_hash)
    ho_path = ho_dir / "predictions.parquet"
    if not ho_path.exists():
        raise FileNotFoundError(f"holdout predictions.parquet not found: {ho_path}")

    ho_preds = pl.read_parquet(ho_path)
    legacy_ho = {k: v for k, v in _LEGACY_RENAME.items() if k in ho_preds.columns}
    if legacy_ho:
        ho_preds = ho_preds.rename(legacy_ho)
    src_id_ho = _detect_id_col(ho_preds.columns)
    if src_id_ho != "symbol":
        ho_preds = ho_preds.rename({src_id_ho: "symbol"})

    ho_required = {"timestamp", "symbol"}
    ho_missing = ho_required - set(ho_preds.columns)
    if ho_missing:
        raise ValueError(
            f"{case_study}/{holdout_prediction_hash}: holdout predictions.parquet "
            f"missing columns {sorted(ho_missing)}; got {ho_preds.columns}"
        )

    ho_keys = ho_preds.select("timestamp", "symbol").unique()
    missing_symbols = (
        ho_keys.select("symbol")
        .unique()
        .join(per_symbol_widths.select("symbol"), on="symbol", how="anti")
    )
    if not missing_symbols.is_empty():
        pooled_q = val_preds["abs_resid"].quantile(1.0 - alpha, interpolation="higher")
        if pooled_q is None:
            raise ValueError(
                f"{case_study}/{val_prediction_hash}: holdout pooled quantile is undefined"
            )
        pooled = missing_symbols.with_columns(
            width=pl.lit(2.0 * float(pooled_q), dtype=pl.Float64),
            alpha=pl.lit(alpha, dtype=pl.Float64),
            calibration_n=pl.lit(val_preds.height, dtype=pl.UInt32),
            calibration_scope=pl.lit("pooled"),
            calibration_version=pl.lit(CALIBRATION_VERSION),
        ).select(per_symbol_widths.columns)
        per_symbol_widths = pl.concat([per_symbol_widths, pooled], how="vertical_relaxed")

    widths = (
        ho_keys.join(per_symbol_widths, on="symbol", how="inner")
        .with_columns(fold_id=pl.lit(-1, dtype=pl.Int64))
        .select(
            "timestamp",
            "symbol",
            "fold_id",
            "width",
            "alpha",
            "calibration_n",
            "calibration_scope",
            "calibration_version",
        )
        .sort("timestamp", "symbol")
    )

    if widths.is_empty():
        raise ValueError(
            f"{case_study}/{holdout_prediction_hash}: pooled-width join with "
            f"holdout predictions produced no rows. Holdout symbol set may not "
            f"overlap with val-calibrated symbols."
        )

    if write:
        out = ho_dir / "conformal_widths.parquet"
        _write_widths(out, widths, alpha, immutable=immutable)

    return widths


def load_conformal_widths(
    case_study: str,
    prediction_hash: str,
    *,
    alpha: float | None = None,
    min_calibration_n: int = DEFAULT_MIN_CALIBRATION_N,
    calibration_version: str = CALIBRATION_VERSION,
) -> pl.DataFrame:
    """Load persisted widths. Filters to a specific alpha if supplied.

    Auto-generates ``conformal_widths.parquet`` via ``compute_conformal_widths``
    when missing so the conformal_weighted allocator works end-to-end inside the
    canonical sweep without a separate widths-bootstrap step. Only the default
    alpha is computed on auto-generation; callers asking for a non-default alpha
    on a fresh prediction set should compute widths up-front.
    """
    path = _predictions_dir(case_study, prediction_hash) / "conformal_widths.parquet"
    if not path.exists():
        compute_conformal_widths(case_study, prediction_hash, min_calibration_n=min_calibration_n)
    df = pl.read_parquet(path)
    if "calibration_version" not in df.columns:
        raise ValueError(
            f"Legacy conformal artifact at {path}; preserve and regenerate it before use"
        )
    df = df.filter(pl.col("calibration_version") == calibration_version)
    if df.is_empty():
        raise ValueError(f"No widths for calibration_version={calibration_version!r} in {path}")
    if alpha is not None:
        available = sorted(set(df["alpha"].to_list()))
        df = df.filter(pl.col("alpha") == alpha)
        if df.is_empty():
            raise ValueError(f"No widths at alpha={alpha} in {path}; available alphas: {available}")
    return df


def coverage_summary(case_study: str, prediction_hash: str, *, alpha: float | None = None) -> dict:
    """Per-fold coverage and width-dispersion diagnostics (no side effects)."""
    pred_dir = _predictions_dir(case_study, prediction_hash)
    preds = pl.read_parquet(pred_dir / "predictions.parquet")
    src_id_col = _detect_id_col(preds.columns)
    # Widths file always uses canonical "symbol" (see compute_conformal_widths).
    widths = load_conformal_widths(case_study, prediction_hash, alpha=alpha)
    id_col = "symbol"

    n_total = preds[src_id_col].n_unique()
    folds = sorted(widths["fold_id"].unique().to_list())
    by_fold = []
    for k in folds:
        wk = widths.filter(pl.col("fold_id") == k).select(id_col, "width").unique()
        n_with = wk.height
        w_min = float(wk["width"].min()) if n_with else float("nan")
        w_max = float(wk["width"].max()) if n_with else float("nan")
        by_fold.append(
            {
                "fold_id": k,
                "n_with_width": n_with,
                "n_total": n_total,
                "frac_covered": n_with / n_total if n_total else 0.0,
                "mean_width": float(wk["width"].mean()) if n_with else float("nan"),
                "median_width": float(wk["width"].median()) if n_with else float("nan"),
                "width_p10": float(wk["width"].quantile(0.10)) if n_with else float("nan"),
                "width_p90": float(wk["width"].quantile(0.90)) if n_with else float("nan"),
                "max_min_ratio": (w_max / max(w_min, 1e-12)) if n_with else float("nan"),
            }
        )
    return {
        "case_study": case_study,
        "prediction_hash": prediction_hash,
        "id_col": id_col,
        "n_entities": n_total,
        "n_folds_with_widths": len(folds),
        "by_fold": by_fold,
    }
