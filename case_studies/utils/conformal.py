"""Per-prediction Mondrian split-conformal widths for position sizing.

Walk-forward, expanding-window calibration:
fold-k width for entity i is 2·q_{1-α}(|y_true − y_score|) over folds {0..k-1}
restricted to entity i. The chronologically earliest fold has no walk-forward
prior; it falls back to cross-conformal calibration pooling all OTHER
validation folds. By construction "all OTHER folds" for the earliest fold are
all *later* folds, so the calibration is forward-looking rather than strictly
walk-forward; coverage still holds at the validation-set aggregate level
because the val set is jointly OOS relative to training (same trade-off
``compute_holdout_conformal_widths`` makes). Treat the earliest fold's
``coverage_summary`` row as a separate cohort from the strictly walk-forward
folds — they are not directly comparable. Entities with fewer than
``min_calibration_n`` calibration residuals get no width for that fold (the
allocator drops them from the top-K selection at runtime).

Storage: alongside ``predictions.parquet`` in the same prediction-hash directory.
Writes are alpha-aware: a new alpha is appended to any existing
``conformal_widths.parquet`` (rows for the same alpha are replaced), so the
single artifact can carry multiple alphas. The output always uses ``symbol``
as the entity column (matching ``backtest_loaders.load_predictions_for_backtest``'s
normalization), regardless of whether the source predictions.parquet uses
``product`` or ``stock_id``.
"""

from __future__ import annotations

from pathlib import Path

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


def _detect_id_col(columns: list[str]) -> str:
    for c in ID_COLS:
        if c in columns:
            return c
    raise ValueError(
        f"predictions.parquet has no canonical entity column "
        f"(expected one of {ID_COLS}); found {columns}"
    )


def _predictions_dir(case_study: str, prediction_hash: str) -> Path:
    return get_case_study_dir(case_study) / "run_log" / "predictions" / prediction_hash


def _write_widths(path: Path, new_widths: pl.DataFrame, alpha: float) -> None:
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
            # A zero-byte or missing-magic-bytes file from a half-finished
            # concurrent write surfaces as NoDataError/OSError, not just
            # ComputeError — treat any unreadable file as "no prior widths".
            existing = None
        if existing is not None:
            # Float equality on alpha is fine here: we write Float64 and read
            # back Float64; both sides round-trip bit-identically through parquet.
            keep = existing.filter(pl.col("alpha") != alpha)
            merged = pl.concat([keep, new_widths], how="diagonal_relaxed")
    merged.write_parquet(path)


def compute_conformal_widths(
    case_study: str,
    prediction_hash: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_calibration_n: int = DEFAULT_MIN_CALIBRATION_N,
    write: bool = True,
) -> pl.DataFrame:
    """Compute and (optionally) persist Mondrian split-conformal widths.

    Returns one row per (timestamp, entity) for which a width could be
    calibrated: columns ``[timestamp, <id_col>, fold_id, width, alpha,
    calibration_n]``. Width = 2·q_{1-α}(|y_true − y_score|) on prior-fold
    residuals for that entity. The chronologically earliest fold has no
    walk-forward prior; it falls back to cross-conformal calibration
    pooling all OTHER validation folds (mirroring the holdout pattern).
    Note that for the earliest fold "all OTHER folds" are by construction
    all *later* folds, so its calibration is forward-looking rather than
    strictly walk-forward — its ``coverage_summary`` row should not be
    compared apples-to-apples with the strictly walk-forward folds.

    Writes are alpha-aware (see module docstring): an existing
    ``conformal_widths.parquet`` for the same prediction hash retains rows
    at other alphas; rows at this ``alpha`` are replaced.

    Raises ``ValueError`` when:
      * predictions.parquet is missing or has < 2 fold ids (degenerate
        ``fold_id``); the function requires ≥2 folds to define a walk-forward
        prior or a cross-conformal fallback;
      * no fold yields any entity meeting ``min_calibration_n`` after the
        prior-fold (or fallback) filter.
    """
    pred_dir = _predictions_dir(case_study, prediction_hash)
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
    # Build the per-fold calibration pool. For all but the chronologically
    # earliest fold this is the strictly walk-forward prefix of prior folds.
    # For the earliest fold we fall back to "all OTHER folds" — see module
    # docstring for the forward-looking-pool caveat.
    prior_folds_for: dict[int, list[int]] = {}
    for i, f in enumerate(fold_chronological):
        prior_folds_for[f] = (
            fold_chronological[:i] if i > 0 else [g for g in fold_chronological if g != f]
        )

    fold_widths_rows: list[pl.DataFrame] = []
    for k in folds:
        prior = prior_folds_for[k]
        cal = preds.filter(pl.col("fold_id").is_in(prior))
        if cal.is_empty():
            continue
        widths_k = (
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
            )
            .select(id_col, "fold_id", "width", "alpha", "calibration_n")
        )
        fold_widths_rows.append(widths_k)

    if not fold_widths_rows:
        raise ValueError(
            f"{case_study}/{prediction_hash}: no fold had prior-fold "
            f"calibration data after applying min_calibration_n={min_calibration_n}"
        )

    fold_widths = pl.concat(fold_widths_rows)

    timestamps = preds.select("timestamp", id_col, "fold_id").unique()
    widths = (
        timestamps.join(fold_widths, on=[id_col, "fold_id"], how="inner")
        .select("timestamp", id_col, "fold_id", "width", "alpha", "calibration_n")
        .sort("timestamp", id_col)
    )

    if write:
        out = _predictions_dir(case_study, prediction_hash) / "conformal_widths.parquet"
        _write_widths(out, widths, alpha)

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
) -> pl.DataFrame:
    """Pooled per-symbol split-conformal widths for the holdout window.

    Calibration set: all validation residuals for the val prediction set,
    pooled across folds within each symbol. Prediction set: every
    (timestamp, symbol) pair in the holdout predictions parquet.

    Per-symbol pooled q_{1-α}(|y_true − y_score|) is broadcast across the
    holdout window for that symbol — one width per symbol, applied uniformly
    over the holdout timestamps. Symbols with fewer than
    ``min_calibration_n`` val residuals get no width (the allocator drops
    them from the top-K selection at runtime).

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
        )
        .select("symbol", "width", "alpha", "calibration_n")
    )

    if per_symbol_widths.is_empty():
        raise ValueError(
            f"{case_study}/{val_prediction_hash}: no symbol has ≥{min_calibration_n} "
            f"val residuals; cannot compute pooled per-symbol widths"
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

    widths = (
        ho_keys.join(per_symbol_widths, on="symbol", how="inner")
        .with_columns(fold_id=pl.lit(-1, dtype=pl.Int64))
        .select("timestamp", "symbol", "fold_id", "width", "alpha", "calibration_n")
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
        _write_widths(out, widths, alpha)

    return widths


def load_conformal_widths(
    case_study: str, prediction_hash: str, *, alpha: float | None = None
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
        compute_conformal_widths(case_study, prediction_hash)
    df = pl.read_parquet(path)
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
