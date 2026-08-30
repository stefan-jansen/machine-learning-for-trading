"""Per-prediction conformal widths for position sizing.

Validation widths follow the ``walk_forward_v3`` contract, which is one rule for
every fold: the width used at timestamp ``t`` for an entity is calibrated on every
residual for that entity at or before ``t - h``, where ``h`` is the label horizon in
prediction data steps. Residuals from earlier folds and from the current fold's own
elapsed history are both eligible. An entity with fewer than ``min_calibration_n``
eligible residuals receives a pooled width from every entity's eligible residuals, so
allocation never changes the selected basket by silently dropping an uncalibrated entity.

``h`` is an embargo and it is load-bearing in both directions. Without it, a residual at
``t'`` whose forward return realizes over ``(t', t'+h]`` carries information from after
the decision it sizes - which the prior-fold-only rule this replaces did at every fold
boundary, not just at the holdout. With it, the earliest fold no longer has to abstain: it
calibrates on its own elapsed history after a warm-up rather than sitting out entirely,
which on a two-fold split is the difference between forfeiting half the evaluation period
and forfeiting a warm-up.

**No coverage guarantee is claimed or consumed.** Split conformal's finite-sample coverage
requires the calibration and test scores to be exchangeable, and return residuals are
heteroskedastic and regime-dependent, so they are not. What the allocator does with these
widths is inverse-uncertainty sizing with a conformal quantile standing in for a
volatility estimate; nothing downstream reads an interval or a coverage level.

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

from utils.modeling import conformal_quantile
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
CALIBRATION_VERSION: str = "walk_forward_v3"
POOLED_FALLBACK: str = "pooled_prior_oos"

HOLDOUT_CONFORMAL_EMBARGO_STEPS: dict[str, int] = {
    "etfs/fwd_ret_21d": 21,
    "etfs/fwd_ret_5d": 5,
    "cme_futures/fwd_ret_5d": 5,
    "cme_futures/fwd_ret_21d": 21,
    "fx_pairs/fwd_ret_1d": 1,
    "fx_pairs/fwd_ret_5d": 5,
    "fx_pairs/fwd_ret_21d": 21,
    "crypto_perps_funding/fwd_ret_24h": 3,
    "crypto_perps_funding/fwd_ret_8h": 1,
    "crypto_perps_funding/fwd_dir_8h": 1,
    "crypto_perps_funding/fwd_dir_8h_3c": 1,
    "nasdaq100_microstructure/fwd_ret_15m": 1,
    "nasdaq100_microstructure/fwd_ret_60m": 4,
    "nasdaq100_microstructure/fwd_ret_5m": 1,
    "sp500_equity_option_analytics/fwd_ret_5d": 5,
    "sp500_equity_option_analytics/fwd_ret_risk_adj_5d": 5,
    # The three below joined the table when this case study declared the conformal_weighted
    # allocator, which sizes every declared label rather than the primary alone. Each value is
    # the label's own horizon from labels.horizons in sessions, the panel's step: a ten-session
    # return needs ten sessions of embargo for the calibration residuals to stop overlapping the
    # holdout, and the direction labels reach exactly as far as the returns they are signs of.
    "sp500_equity_option_analytics/fwd_ret_10d": 10,
    "sp500_equity_option_analytics/fwd_dir_5d": 5,
    "sp500_equity_option_analytics/fwd_dir_10d": 10,
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
        quantile = conformal_quantile(calibration_residuals, level)
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


def _expanding_calibration(frame: pl.DataFrame, *, alpha: float) -> pl.DataFrame:
    """Per grid step: the quantile and count of every residual up to and including it.

    One row per distinct ``step`` in ``frame``, carrying what a caller standing at that
    step already knows. The embargo is applied by the as-of join that reads this table,
    not here, so the same construction serves the per-entity and the pooled pool.
    """
    ordered = frame.sort("step")
    window = max(ordered.height, 1)
    return (
        ordered.with_columns(
            cal_q=pl.col("abs_resid").rolling_quantile(
                quantile=1.0 - alpha,
                window_size=window,
                min_samples=1,
                interpolation="higher",
            ),
            cal_n=pl.int_range(1, pl.len() + 1, dtype=pl.UInt32),
        )
        .group_by("step", maintain_order=True)
        .agg(cal_q=pl.col("cal_q").last(), cal_n=pl.col("cal_n").last())
        .with_columns(pl.col("step").cast(pl.Int64))
    )


def compute_conformal_widths(
    case_study: str,
    prediction_hash: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_calibration_n: int = DEFAULT_MIN_CALIBRATION_N,
    write: bool = True,
    case_dir: Path | None = None,
    label: str | None = None,
    embargo_steps: int | None = None,
) -> pl.DataFrame:
    """Compute and optionally persist expanding walk-forward conformal widths.

    Returns one row per (timestamp, entity) for which a width could be
    calibrated: columns ``[timestamp, <id_col>, fold_id, width, alpha,
    calibration_n, calibration_scope, calibration_version]``.

    Calibration rule, one rule for every fold: the width used at timestamp
    ``t`` for entity ``s`` is ``2·q_{1-α}`` of ``|y_true − y_score|`` over
    every residual for ``s`` at timestamps at or before ``t − h``, where ``h``
    is the label horizon in prediction data steps. Residuals from earlier
    folds and from the current fold's own elapsed history are both eligible,
    which is what lets the earliest fold trade after a warm-up instead of
    sitting out entirely.

    ``h`` is the embargo, and it is the same quantity
    :func:`compute_holdout_conformal_widths` applies at the validation/holdout
    boundary: a residual at ``t'`` depends on the return realized over
    ``(t', t'+h]``, so a residual with ``t' + h > t`` carries information from
    after the decision. Pass ``embargo_steps`` directly or pass ``label`` to
    take the reviewed value from :data:`HOLDOUT_CONFORMAL_EMBARGO_STEPS`.

    Entities with fewer than ``min_calibration_n`` eligible residuals of their
    own use a pooled quantile over every entity's eligible residuals, so
    allocation never changes the selected basket by silently dropping an
    uncalibrated entity.

    Writes are alpha-aware (see module docstring): an existing
    ``conformal_widths.parquet`` for the same prediction hash retains rows
    at other alphas; rows at this ``alpha`` are replaced.

    Raises ``ValueError`` when neither ``embargo_steps`` nor a ``label`` with a
    reviewed embargo is given, when ``embargo_steps`` is below one, or when no
    (timestamp, entity) pair clears the warm-up.
    """
    pred_dir = _predictions_dir(case_study, prediction_hash, case_dir=case_dir)
    pred_path = pred_dir / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.parquet not found: {pred_path}")

    if embargo_steps is None:
        if label is None:
            raise ValueError(
                f"{case_study}/{prediction_hash}: conformal calibration needs the label "
                "horizon as an embargo - pass embargo_steps, or label to look up the "
                "reviewed value"
            )
        embargo_steps = holdout_conformal_embargo_steps(case_study, label)
    if embargo_steps < 1:
        raise ValueError(
            f"{case_study}/{prediction_hash}: embargo_steps={embargo_steps} would calibrate "
            "on a residual that is not yet realized at the decision it sizes"
        )

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
    if preds.is_empty():
        raise ValueError(f"{case_study}/{prediction_hash}: no residuals to calibrate on")

    # The embargo is expressed in data steps, so it is applied on the position of a
    # timestamp in the prediction set's own grid rather than on the calendar. A gap in
    # the grid is a gap in the data, and stepping over it is what "h steps back" means.
    grid = preds.select("timestamp").unique().sort("timestamp").with_row_index("step")
    preds = preds.join(grid, on="timestamp", how="inner")

    per_entity = pl.concat(
        [
            _expanding_calibration(group, alpha=alpha).with_columns(pl.lit(entity).alias(id_col))
            for (entity,), group in preds.group_by([id_col], maintain_order=True)
        ]
    )
    pooled = _expanding_calibration(preds, alpha=alpha)

    targets = (
        preds.select("timestamp", id_col, "fold_id", "step")
        .unique()
        .with_columns(known_by=pl.col("step").cast(pl.Int64) - embargo_steps)
        # Sorted within entity as well as globally: join_asof cannot verify sortedness once
        # `by` groups are given, and an unsorted group silently resolves to the wrong row.
        .sort(id_col, "known_by")
    )
    resolved = (
        targets.join_asof(
            per_entity.sort("step").rename(
                {"cal_q": "entity_q", "cal_n": "entity_n", "step": "entity_step"}
            ),
            left_on="known_by",
            right_on="entity_step",
            by=id_col,
            strategy="backward",
        )
        .sort("known_by")
        .join_asof(
            pooled.sort("step").rename(
                {"cal_q": "pooled_q", "cal_n": "pooled_n", "step": "pooled_step"}
            ),
            left_on="known_by",
            right_on="pooled_step",
            strategy="backward",
        )
    )

    enough_own = pl.col("entity_n") >= min_calibration_n
    enough_pooled = pl.col("pooled_n") >= min_calibration_n
    widths = (
        resolved.with_columns(
            width=pl.when(enough_own)
            .then(2.0 * pl.col("entity_q"))
            .when(enough_pooled)
            .then(2.0 * pl.col("pooled_q"))
            .otherwise(None),
            calibration_n=pl.when(enough_own)
            .then(pl.col("entity_n"))
            .otherwise(pl.col("pooled_n")),
            calibration_scope=pl.when(enough_own)
            .then(pl.lit("symbol"))
            .otherwise(pl.lit("pooled")),
        )
        .drop_nulls("width")
        .with_columns(
            alpha=pl.lit(alpha, dtype=pl.Float64),
            calibration_version=pl.lit(CALIBRATION_VERSION),
        )
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

    if widths.is_empty():
        raise ValueError(
            f"{case_study}/{prediction_hash}: no decision clears a warm-up of "
            f"min_calibration_n={min_calibration_n} residuals plus an embargo of "
            f"{embargo_steps} steps"
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
    label: str | None = None,
    embargo_steps: int | None = None,
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
        compute_conformal_widths(
            case_study,
            prediction_hash,
            min_calibration_n=min_calibration_n,
            label=label,
            embargo_steps=embargo_steps,
        )
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
