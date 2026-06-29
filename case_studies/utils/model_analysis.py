"""Shared utilities for per-case-study model analysis notebooks.

Loads all predictions from the registry, computes cross-sectional IC,
learning curves, fold-level diagnostics, prediction agreement, bucket
monotonicity, and regime-conditional analysis.

Usage:
    from case_studies.utils.model_analysis import (
        load_predictions,
        model_summary_table,
        fold_performance_matrix,
        prediction_bucket_monotonicity,
        prediction_correlation_matrix,
        regime_conditional_ic,
    )
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Import torch before ml4t.diagnostic. ml4t.diagnostic transitively loads the
# `cuda` Python package, which dlopens the older system `libcudart.so.12`
# (12.0.146) and wins the symbol resolution; subsequent torch imports then
# fail with `undefined symbol: cudaGetDriverEntryPointByVersion`. Loading
# torch first ensures its bundled CUDA runtime wins. Same pattern as in
# `case_studies/utils/latent_factors/__init__.py`.
import torch  # noqa: F401
from ml4t.diagnostic.metrics import cross_sectional_ic

from utils.paths import get_case_study_dir

from .notebook_contracts import degenerate_prediction_sql

# ---------------------------------------------------------------------------
# Fast metrics from registry (no raw prediction loading needed)
# ---------------------------------------------------------------------------


def load_metrics_from_registry(
    case_study_id: str,
    label: str | None = None,
    families: list[str] | None = None,
) -> pl.DataFrame:
    """Load pre-computed IC metrics directly from the registry.

    Much faster than loading raw predictions — queries prediction_metrics
    table which stores ic_mean/ic_std computed during training runs.

    Returns DataFrame with columns:
        family, config_name, label, checkpoint_value, checkpoint_kind,
        ic_mean, ic_std
    """
    case_dir = get_case_study_dir(case_study_id)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return pl.DataFrame()

    db = sqlite3.connect(str(db_path))

    # Probe which uncertainty columns exist (older registries may predate the
    # daily-uncertainty backfill).
    pm_cols = {row[1] for row in db.execute("PRAGMA table_info(prediction_metrics)").fetchall()}
    unc_cols = [
        c
        for c in (
            "ic_mean_daily",
            "ic_se_hac",
            "ic_ci_lo",
            "ic_ci_hi",
            "ic_boot_lo",
            "ic_boot_hi",
            "ic_t_hac",
            "ic_p_hac",
            "ic_n_days",
            "ic_hac_lag",
            "auc_mean_daily",
            "auc_se_hac",
            "auc_ci_lo",
            "auc_ci_hi",
            "auc_n_days",
        )
        if c in pm_cols
    ]
    extra_cols = "".join(f", pm.{c}" for c in unc_cols)

    # Keep rows that have either the daily-pooled IC (post-backfill) or the
    # legacy fold-aggregated `ic_mean` (pre-backfill or partial migration).
    # ``ic_mean_daily`` may be present as a column but null on un-backfilled
    # rows; the disjunction surfaces both flavors so a partial backfill does
    # not silently drop rows from the leaderboard.
    has_daily = "ic_mean_daily" in pm_cols
    primary_filter = (
        "(pm.ic_mean_daily IS NOT NULL OR pm.ic_mean IS NOT NULL)"
        if has_daily
        else "pm.ic_mean IS NOT NULL"
    )
    query = f"""
        SELECT
            t.family,
            t.config_name,
            t.label,
            p.checkpoint_value,
            p.checkpoint_kind,
            pm.ic_mean,
            pm.ic_std{extra_cols}
        FROM prediction_metrics pm
        JOIN prediction_sets p ON pm.prediction_hash = p.prediction_hash
        JOIN training_runs t ON p.training_hash = t.training_hash
        WHERE p.split = 'validation'
          AND {primary_filter}
          {degenerate_prediction_sql("p.prediction_hash")}
    """
    params: list[Any] = []

    if label is not None:
        query += " AND t.label = ?"
        params.append(label)
    if families is not None:
        placeholders = ",".join("?" * len(families))
        query += f" AND t.family IN ({placeholders})"
        params.extend(families)

    rows = db.execute(query, params).fetchall()
    db.close()

    if not rows:
        return pl.DataFrame()

    schema: dict[str, Any] = {
        "family": pl.Utf8,
        "config_name": pl.Utf8,
        "label": pl.Utf8,
        "checkpoint_value": pl.Int64,
        "checkpoint_kind": pl.Utf8,
        "ic_mean": pl.Float64,
        "ic_std": pl.Float64,
    }
    for c in unc_cols:
        schema[c] = pl.Float64

    df = pl.DataFrame(rows, schema=schema, orient="row")

    # Deduplicate (some configs have multiple training runs) — keep best.
    agg_exprs = [pl.col("ic_mean").mean(), pl.col("ic_std").mean()]
    for c in unc_cols:
        agg_exprs.append(pl.col(c).mean())

    df = df.group_by(["family", "config_name", "label", "checkpoint_value", "checkpoint_kind"]).agg(
        agg_exprs
    )

    return df.sort("ic_mean", descending=True)


def load_fold_metrics_from_registry(
    case_study_id: str,
    label: str | None = None,
    families: list[str] | None = None,
) -> pl.DataFrame:
    """Load per-fold IC metrics from the fold_metrics registry table.

    Returns DataFrame with columns:
        family, config_name, label, checkpoint_value, fold_id, ic, ic_std,
        n_entities, rmse, mae
    """
    case_dir = get_case_study_dir(case_study_id)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return pl.DataFrame()

    db = sqlite3.connect(str(db_path))

    # Check if fold_metrics table exists
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fold_metrics'"
    ).fetchall()
    if not tables:
        db.close()
        return pl.DataFrame()

    # Only select columns that exist on `fold_metrics` per the live schema
    # (prediction-side fold_metrics has no n_periods / n_obs — those live on
    # backtest_fold_metrics). Older callers selected them and crashed.
    query = """
        SELECT
            t.family,
            t.config_name,
            t.label,
            p.checkpoint_value,
            fm.fold_id,
            fm.ic,
            fm.ic_std,
            fm.n_entities,
            fm.rmse,
            fm.mae
        FROM fold_metrics fm
        JOIN prediction_sets p ON fm.prediction_hash = p.prediction_hash
        JOIN training_runs t ON p.training_hash = t.training_hash
        WHERE p.split = 'validation'
    """
    params: list[Any] = []

    if label is not None:
        query += " AND t.label = ?"
        params.append(label)
    if families is not None:
        placeholders = ",".join("?" * len(families))
        query += f" AND t.family IN ({placeholders})"
        params.extend(families)

    rows = db.execute(query, params).fetchall()
    db.close()

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(
        rows,
        schema={
            "family": pl.Utf8,
            "config_name": pl.Utf8,
            "label": pl.Utf8,
            "checkpoint_value": pl.Int64,
            "fold_id": pl.Int64,
            "ic": pl.Float64,
            "ic_std": pl.Float64,
            "n_entities": pl.Float64,
            "rmse": pl.Float64,
            "mae": pl.Float64,
        },
        orient="row",
    )

    # Deduplicate reruns: keep latest per (config, label, checkpoint, fold)
    df = df.unique(
        subset=["family", "config_name", "label", "checkpoint_value", "fold_id"],
        keep="last",
    )

    return df.sort("family", "config_name", "label", "checkpoint_value", "fold_id")


def load_all_metrics(
    case_study_id: str,
    label: str | None = None,
) -> pl.DataFrame:
    """Load pre-computed IC metrics from the registry.

    Fast path for leaderboard and learning curves — no raw prediction loading.
    """
    result = load_metrics_from_registry(case_study_id, label=label)
    if result.height == 0:
        return pl.DataFrame()

    return result.sort("ic_mean", descending=True)


def best_model_per_family_fast(
    metrics: pl.DataFrame,
) -> pl.DataFrame:
    """Find the best (config, checkpoint) per family from pre-computed metrics.

    Much faster than select_best_per_family() which recomputes IC from raw predictions.
    """
    if metrics.height == 0:
        return metrics

    return (
        metrics.filter(pl.col("ic_mean").is_not_null())
        .sort("ic_mean", descending=True)
        .group_by("family")
        .first()
        .sort("ic_mean", descending=True)
    )


# ---------------------------------------------------------------------------
# Load predictions from registry
# ---------------------------------------------------------------------------


def load_predictions(
    case_study_id: str,
    *,
    family: str | None = None,
    label: str | None = None,
    config_name: str | None = None,
    checkpoint_value: int | None = None,
    split: str = "validation",
) -> pl.DataFrame:
    """Load predictions from the registry.

    Queries prediction_sets joined with training_runs, then reads the
    matching prediction parquets. Filter as narrowly as possible to
    avoid loading unnecessary data.

    Parameters
    ----------
    family : str, optional
        Model family (e.g., 'linear', 'gbm'). Strongly recommended.
    label : str, optional
        Target label (e.g., 'fwd_ret_21d').
    config_name : str, optional
        Specific config (e.g., 'ridge_a10000000.0').
    checkpoint_value : int, optional
        Specific checkpoint epoch/trees.
    split : str
        'validation' or 'holdout'.

    Returns
    -------
    pl.DataFrame with columns:
        timestamp, symbol, y_true, y_score, fold_id,
        family, config_name, label, checkpoint_value, checkpoint_kind,
        training_hash, prediction_hash
    """
    case_dir = get_case_study_dir(case_study_id)
    db_path = case_dir / "run_log" / "registry.db"

    if not db_path.exists():
        raise FileNotFoundError(f"Registry not found: {db_path}")

    db = sqlite3.connect(str(db_path))

    query = """
        SELECT
            t.training_hash,
            t.family,
            t.config_name,
            t.label,
            p.prediction_hash,
            p.checkpoint_value,
            p.checkpoint_kind,
            p.split
        FROM training_runs t
        JOIN prediction_sets p ON t.training_hash = p.training_hash
        WHERE p.split = ?
    """
    params: list[Any] = [split]

    if label is not None:
        query += " AND t.label = ?"
        params.append(label)
    if family is not None:
        query += " AND t.family = ?"
        params.append(family)
    if config_name is not None:
        query += " AND t.config_name = ?"
        params.append(config_name)
    if checkpoint_value is not None:
        query += " AND p.checkpoint_value = ?"
        params.append(checkpoint_value)

    rows = db.execute(query, params).fetchall()
    db.close()

    if not rows:
        return pl.DataFrame()

    pred_dir = case_dir / "run_log" / "predictions"
    frames = []

    for t_hash, fam, config, lbl, p_hash, cp_val, cp_kind, sp in rows:
        parquet_path = pred_dir / p_hash / "predictions.parquet"
        if not parquet_path.exists():
            continue

        df = pl.read_parquet(parquet_path)

        # Normalize legacy column names to canonical schema
        renames = {}
        if "actual" in df.columns and "y_true" not in df.columns:
            renames["actual"] = "y_true"
        if "prediction" in df.columns and "y_score" not in df.columns:
            renames["prediction"] = "y_score"
        if "fold" in df.columns and "fold_id" not in df.columns:
            renames["fold"] = "fold_id"
        if renames:
            df = df.rename(renames)

        # Normalize timestamp dtype: registries hold a mix of tz-naive and
        # tz-aware (UTC) Datetime[ms]. Cast to tz-naive ms so diagonal_relaxed
        # concat can determine a single supertype.
        if "timestamp" in df.columns:
            ts_dtype = df.schema["timestamp"]
            if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
                df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

        # Add metadata columns
        df = df.with_columns(
            pl.lit(fam).alias("family"),
            pl.lit(config).alias("config_name"),
            pl.lit(lbl).alias("label"),
            pl.lit(cp_val).alias("checkpoint_value"),
            pl.lit(cp_kind).alias("checkpoint_kind"),
            pl.lit(t_hash).alias("training_hash"),
            pl.lit(p_hash).alias("prediction_hash"),
        )
        frames.append(df)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# Model summary table
# ---------------------------------------------------------------------------


def model_summary_table(
    all_preds: pl.DataFrame,
    date_col: str = "timestamp",
    *,
    horizon: int = 1,
    n_boot: int = 1000,
) -> pl.DataFrame:
    """Build summary table comparing all models with daily-pooled uncertainty.

    Groups by (family, config_name, checkpoint_value) and reports the daily
    cross-sectional IC pooled across folds together with naive, HAC and block
    bootstrap intervals from :func:`compute_ic_uncertainty`. The fold-level
    `ic_mean` / `ic_std` are still emitted for backward compatibility but
    SHOULD NOT be used as the primary uncertainty estimate — N≈8–16 folds is
    too small a sample.

    Parameters
    ----------
    horizon : int
        Forward-return horizon (in label-step units). Sets the HAC lag and the
        bootstrap block length. Defaults to 1; pass the actual horizon (e.g. 21
        for fwd_ret_21d) for autocorrelation-aware SE.
    """
    if all_preds.height == 0:
        return pl.DataFrame()

    from ml4t.diagnostic.metrics import (
        compute_ic_uncertainty,
        cross_sectional_ic_series,
    )

    groups = all_preds.group_by(["family", "config_name", "label", "checkpoint_value"]).agg(
        pl.len().alias("n_predictions")
    )

    results = []
    for row in groups.iter_rows(named=True):
        mask = (
            (pl.col("family") == row["family"])
            & (pl.col("config_name") == row["config_name"])
            & (pl.col("label") == row["label"])
        )
        if row["checkpoint_value"] is not None:
            mask = mask & (pl.col("checkpoint_value") == row["checkpoint_value"])

        subset = all_preds.filter(mask)
        _entity = "symbol" if "symbol" in subset.columns else None
        stats = cross_sectional_ic(
            subset,
            subset,
            pred_col="y_score",
            ret_col="y_true",
            date_col=date_col,
            entity_col=_entity,
            method="spearman",
            min_obs=5,
        )

        # Daily-pooled IC + uncertainty (HAC + block bootstrap).
        daily_ic = cross_sectional_ic_series(
            subset,
            subset,
            pred_col="y_score",
            ret_col="y_true",
            date_col=date_col,
            entity_col=_entity,
            method="spearman",
            min_obs=5,
        )
        unc: dict[str, float] = {}
        if isinstance(daily_ic, pl.DataFrame) and daily_ic.drop_nulls("ic").height >= 3:
            u = compute_ic_uncertainty(
                daily_ic.drop_nulls("ic").select("ic"),
                horizon=int(max(1, horizon)),
                n_boot=n_boot,
            )
            unc = {
                "ic_mean_daily": u["mean_ic"],
                "ic_se_naive": u["se_naive"],
                "ic_naive_lo": u["ci_naive_lower"],
                "ic_naive_hi": u["ci_naive_upper"],
                "ic_se_hac": u["se_hac"],
                "ic_ci_lo": u["ci_hac_lower"],
                "ic_ci_hi": u["ci_hac_upper"],
                "ic_t_hac": u["t_hac"],
                "ic_p_hac": u["p_hac"],
                "ic_hac_lag": float(u["hac_lag"]),
                "ic_n_days": float(u["n_days"]),
                "ic_boot_lo": u["ci_boot_lower"],
                "ic_boot_hi": u["ci_boot_upper"],
            }

        n_folds = subset["fold_id"].n_unique() if "fold_id" in subset.columns else 0

        results.append(
            {
                "family": row["family"],
                "config_name": row["config_name"],
                "label": row["label"],
                "checkpoint": row["checkpoint_value"],
                "n_folds": n_folds,
                **stats,
                **unc,
            }
        )

    result_df = pl.DataFrame(results)
    return result_df.sort("ic_mean", descending=True)


# ---------------------------------------------------------------------------
# Learning curve data
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Best model selection
# ---------------------------------------------------------------------------


def select_best_checkpoint(
    all_preds: pl.DataFrame,
    date_col: str = "timestamp",
) -> pl.DataFrame:
    """For each (family, config), find the best checkpoint by mean IC.

    Returns DataFrame with best checkpoint per config, with IC stats.
    """
    summary = model_summary_table(all_preds, date_col)
    if summary.height == 0:
        return summary

    # For each (family, config_name, label), keep the checkpoint with highest IC
    best = (
        summary.sort("ic_mean", descending=True)
        .group_by(["family", "config_name", "label"])
        .first()
    )

    return best.sort("ic_mean", descending=True)


def select_best_per_family(
    all_preds: pl.DataFrame,
    date_col: str = "timestamp",
) -> pl.DataFrame:
    """For each family, find the single best (config, checkpoint).

    Returns one row per family, sorted by IC.
    """
    best_configs = select_best_checkpoint(all_preds, date_col)
    if best_configs.height == 0:
        return best_configs

    best_family = (
        best_configs.sort("ic_mean", descending=True).group_by(["family", "label"]).first()
    )

    return best_family.sort("ic_mean", descending=True)


# ---------------------------------------------------------------------------
# Fold-level performance matrix (Visual #2 from spec)
# ---------------------------------------------------------------------------


def fold_performance_matrix(
    all_preds: pl.DataFrame,
    models: list[dict] | None = None,
    date_col: str = "timestamp",
) -> pl.DataFrame:
    """Build fold × model performance matrix.

    Parameters
    ----------
    models : list[dict], optional
        List of dicts with 'family', 'config_name', 'checkpoint' keys.
        If None, uses best checkpoint per (family, config).

    Returns
    -------
    pl.DataFrame with columns: model_label, fold_id, ic_mean
    """
    if models is None:
        best = select_best_checkpoint(all_preds, date_col)
        models = [
            {"family": r["family"], "config_name": r["config_name"], "checkpoint": r["checkpoint"]}
            for r in best.iter_rows(named=True)
        ]

    results = []
    for m in models:
        mask = (pl.col("family") == m["family"]) & (pl.col("config_name") == m["config_name"])
        if m.get("checkpoint") is not None:
            mask = mask & (pl.col("checkpoint_value") == m["checkpoint"])

        subset = all_preds.filter(mask)
        if subset.height == 0:
            continue

        label = f"{m['family']}/{m['config_name']}"
        for fold in sorted(subset["fold_id"].unique().to_list()):
            fold_preds = subset.filter(pl.col("fold_id") == fold)
            _entity = "symbol" if "symbol" in fold_preds.columns else None
            stats = cross_sectional_ic(
                fold_preds,
                fold_preds,
                pred_col="y_score",
                ret_col="y_true",
                date_col=date_col,
                entity_col=_entity,
                method="spearman",
                min_obs=5,
            )
            results.append(
                {
                    "model_label": label,
                    "fold_id": fold,
                    "ic_mean": stats["ic_mean"],
                    "ic_t": stats["ic_t"],
                    "n_periods": stats["n_periods"],
                }
            )

    return pl.DataFrame(results) if results else pl.DataFrame()


# ---------------------------------------------------------------------------
# Prediction bucket monotonicity (Visual #4 from spec)
# ---------------------------------------------------------------------------


def prediction_bucket_monotonicity(
    preds: pl.DataFrame,
    n_buckets: int = 10,
    date_col: str = "timestamp",
) -> pl.DataFrame:
    """Compute mean realized return by prediction score bucket.

    For each date, ranks predictions into quantile buckets, then averages
    realized returns per bucket across all dates. Monotonic increase from
    low to high buckets indicates ranking ability.

    Returns DataFrame with columns: bucket, mean_return, mean_score, n_obs.
    """
    dates = preds[date_col].unique().sort().to_list()
    bucket_returns: dict[int, list[float]] = {i: [] for i in range(n_buckets)}
    bucket_scores: dict[int, list[float]] = {i: [] for i in range(n_buckets)}

    for d in dates:
        day = preds.filter(pl.col(date_col) == d)
        scores = day["y_score"].to_numpy()
        returns = day["y_true"].to_numpy()
        valid = ~(np.isnan(scores) | np.isnan(returns))
        if valid.sum() < n_buckets:
            continue

        s, r = scores[valid], returns[valid]
        # Rank into buckets
        ranks = np.argsort(np.argsort(s))  # 0..n-1
        buckets = np.minimum(ranks * n_buckets // len(s), n_buckets - 1)

        for b in range(n_buckets):
            mask_b = buckets == b
            if mask_b.any():
                bucket_returns[b].append(float(r[mask_b].mean()))
                bucket_scores[b].append(float(s[mask_b].mean()))

    results = []
    for b in range(n_buckets):
        if bucket_returns[b]:
            results.append(
                {
                    "bucket": b + 1,
                    "mean_return": float(np.mean(bucket_returns[b])),
                    "mean_score": float(np.mean(bucket_scores[b])),
                    "n_obs": len(bucket_returns[b]),
                }
            )

    return pl.DataFrame(results) if results else pl.DataFrame()


# ---------------------------------------------------------------------------
# Prediction correlation matrix (Visual #6 from spec)
# ---------------------------------------------------------------------------


def prediction_correlation_matrix(
    all_preds: pl.DataFrame,
    models: list[dict] | None = None,
    date_col: str = "timestamp",
    entity_col: str = "symbol",
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise Spearman correlation between model predictions.

    Aligns predictions by (date, entity) and computes rank correlation.
    Only uses dates where ALL models have predictions.

    Returns (correlation_matrix, model_labels).
    """
    if models is None:
        best = select_best_per_family(all_preds, date_col)
        models = [
            {"family": r["family"], "config_name": r["config_name"], "checkpoint": r["checkpoint"]}
            for r in best.iter_rows(named=True)
        ]

    # Extract predictions per model, keyed by (date, entity)
    model_preds = {}
    labels = []

    for m in models:
        mask = (pl.col("family") == m["family"]) & (pl.col("config_name") == m["config_name"])
        if m.get("checkpoint") is not None:
            mask = mask & (pl.col("checkpoint_value") == m["checkpoint"])

        subset = all_preds.filter(mask)
        if subset.height == 0:
            continue

        label = f"{m['family']}/{m['config_name']}"
        # Create a key for merge
        keyed = subset.select([date_col, entity_col, "y_score"]).rename({"y_score": label})
        model_preds[label] = keyed
        labels.append(label)

    if len(labels) < 2:
        return np.array([]), labels

    # Merge all on (date, entity)
    merged = model_preds[labels[0]]
    for lbl in labels[1:]:
        merged = merged.join(model_preds[lbl], on=[date_col, entity_col], how="inner")

    # Compute pairwise Spearman correlation via polars rank + pearson
    ranked = merged.select([pl.col(lbl).rank().alias(lbl) for lbl in labels])
    n = len(labels)
    corr = np.eye(n)
    # Use polars corr for each pair (avoids scipy UDF overhead)
    for i in range(n):
        for j in range(i + 1, n):
            r = ranked.select(pl.corr(labels[i], labels[j])).item()
            if r is not None:
                corr[i, j] = r
                corr[j, i] = r

    return corr, labels


# ---------------------------------------------------------------------------
# Regime-conditional IC (Visual #8 from spec)
# ---------------------------------------------------------------------------


def regime_conditional_ic(
    preds: pl.DataFrame,
    regime_col: str = "volatility_regime",
    regime_values: list | None = None,
    date_col: str = "timestamp",
) -> pl.DataFrame:
    """Compute IC conditioned on a regime variable.

    If regime_col is not in the predictions, tries to compute it from y_true
    (e.g., volatility regime from rolling std of cross-sectional returns).

    Returns DataFrame with columns: regime, ic_mean, ic_std, ic_t, n_periods.
    """
    if regime_col not in preds.columns:
        # Auto-compute volatility regime from cross-sectional return dispersion
        daily_std = (
            preds.group_by(date_col).agg(pl.col("y_true").std().alias("cs_std")).sort(date_col)
        )
        median_std = daily_std["cs_std"].median()
        daily_std = daily_std.with_columns(
            pl.when(pl.col("cs_std") > median_std)
            .then(pl.lit("high_vol"))
            .otherwise(pl.lit("low_vol"))
            .alias(regime_col)
        )
        preds = preds.join(daily_std.select([date_col, regime_col]), on=date_col, how="left")

    if regime_values is None:
        regime_values = sorted(preds[regime_col].unique().drop_nulls().to_list())

    results = []
    for regime in regime_values:
        subset = preds.filter(pl.col(regime_col) == regime)
        _entity = "symbol" if "symbol" in subset.columns else None
        stats = cross_sectional_ic(
            subset,
            subset,
            pred_col="y_score",
            ret_col="y_true",
            date_col=date_col,
            entity_col=_entity,
            method="spearman",
            min_obs=5,
        )
        results.append({"regime": regime, **stats})

    return pl.DataFrame(results) if results else pl.DataFrame()


# ---------------------------------------------------------------------------
# Feature importance loading (Visual #7 from spec)
# ---------------------------------------------------------------------------


def load_gbm_feature_importance(
    case_study_id: str,
    label: str | None = None,
    top_n: int = 15,
) -> pl.DataFrame | None:
    """Load GBM feature importance from saved booster files.

    Looks for LightGBM booster .txt files in run_log/training/{hash}/boosters/.
    Extracts gain-based importance per fold.

    Returns DataFrame with columns: config_name, fold_id, feature, importance.
    Returns None if no booster files found.
    """
    case_dir = get_case_study_dir(case_study_id)
    db_path = case_dir / "run_log" / "registry.db"

    if not db_path.exists():
        return None

    db = sqlite3.connect(str(db_path))
    query = "SELECT training_hash, config_name FROM training_runs WHERE family = 'gbm'"
    params: list = []
    if label is not None:
        query += " AND label = ?"
        params.append(label)
    rows = db.execute(query, params).fetchall()
    db.close()

    if not rows:
        return None

    try:
        import lightgbm as lgb
    except ImportError:
        return None

    results = []
    for t_hash, config_name in rows:
        booster_dir = case_dir / "run_log" / "training" / t_hash / "boosters"
        if not booster_dir.exists():
            # Also check under run_log/models/{hash}/boosters (older layout)
            booster_dir = case_dir / "run_log" / "models" / t_hash / "boosters"
        if not booster_dir.exists():
            continue

        for booster_file in sorted(booster_dir.glob("*.txt")):
            # Extract fold from filename: fold_0.txt or {config}_fold0.txt
            name = booster_file.stem
            fold_str = name.split("fold")[-1].lstrip("_") if "fold" in name else "0"
            try:
                fold_id = int(fold_str)
            except ValueError:
                continue

            model = lgb.Booster(model_file=str(booster_file))
            importance = model.feature_importance(importance_type="gain")
            feature_names = model.feature_name()

            for feat, imp in zip(feature_names, importance, strict=False):
                results.append(
                    {
                        "config_name": config_name,
                        "fold_id": fold_id,
                        "feature": feat,
                        "importance": float(imp),
                    }
                )

    if not results:
        return None

    df = pl.DataFrame(results)

    # Normalize per (config, fold) to [0, 1]
    df = df.with_columns(
        (pl.col("importance") / pl.col("importance").max().over(["config_name", "fold_id"])).alias(
            "importance_norm"
        )
    )

    # Filter to top_n features by mean importance across folds
    top_features = (
        df.group_by("feature")
        .agg(pl.col("importance_norm").mean().alias("mean_imp"))
        .sort("mean_imp", descending=True)
        .head(top_n)["feature"]
        .to_list()
    )

    return df.filter(pl.col("feature").is_in(top_features))


def load_linear_coefficients(
    case_study_id: str,
    label: str | None = None,
    top_n: int = 15,
) -> pl.DataFrame | None:
    """Load linear model coefficients from registry training dirs.

    Returns DataFrame with columns: config_name, fold, feature, coefficient.
    Returns None if no coefficient files found.
    """
    case_dir = get_case_study_dir(case_study_id)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return None

    db = sqlite3.connect(str(db_path))
    query = "SELECT training_hash, config_name FROM training_runs WHERE family = 'linear'"
    params: list = []
    if label is not None:
        query += " AND label = ?"
        params.append(label)
    rows = db.execute(query, params).fetchall()
    db.close()

    if not rows:
        return None

    frames = []
    for t_hash, config_name in rows:
        coef_path = case_dir / "run_log" / "training" / t_hash / "coefficients.parquet"
        if not coef_path.exists():
            continue
        try:
            df = pl.read_parquet(coef_path)
            if "config_name" not in df.columns:
                df = df.with_columns(pl.lit(config_name).alias("config_name"))
            frames.append(df)
        except Exception as exc:
            warnings.warn(f"Failed to read coefficients {coef_path}: {exc}", stacklevel=2)
            continue

    if not frames:
        return None

    result = pl.concat(frames)

    # Filter to top_n features by mean |coefficient| across folds (exclude intercept)
    top_features = (
        result.filter(pl.col("feature") != "_intercept_")
        .group_by("feature")
        .agg(pl.col("coefficient").abs().mean().alias("mean_abs"))
        .sort("mean_abs", descending=True)
        .head(top_n)["feature"]
        .to_list()
    )

    return result.filter(
        pl.col("feature").is_in(top_features) | (pl.col("feature") == "_intercept_")
    )


def load_gbm_learning_curves(
    case_study_id: str,
    label: str | None = None,
) -> pl.DataFrame | None:
    """Load GBM learning curves from registry training dirs.

    Returns DataFrame with columns: config, iteration, ic_mean, ic_std.
    Returns None if no learning curve files found.
    """
    case_dir = get_case_study_dir(case_study_id)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return None

    db = sqlite3.connect(str(db_path))
    query = "SELECT training_hash, config_name FROM training_runs WHERE family = 'gbm'"
    params: list = []
    if label is not None:
        query += " AND label = ?"
        params.append(label)
    rows = db.execute(query, params).fetchall()
    db.close()

    if not rows:
        return None

    frames = []
    for t_hash, config_name in rows:
        lc_path = case_dir / "run_log" / "training" / t_hash / "learning_curves.parquet"
        if not lc_path.exists():
            continue
        try:
            df = pl.read_parquet(lc_path)
            if "dataset" not in df.columns:
                df = df.with_columns(pl.lit(case_study_id).alias("dataset"))
            frames.append(df)
        except Exception as exc:
            warnings.warn(f"Failed to read learning curves {lc_path}: {exc}", stacklevel=2)
            continue

    if not frames:
        return None

    return pl.concat(frames)


def load_gbm_fold_metrics(
    case_study_id: str,
    label: str | None = None,
) -> pl.DataFrame | None:
    """Load GBM per-fold IC metrics from the registry fold_metrics table.

    First checks the fold_metrics DB table (populated during training for all CS).
    Falls back to parquet files in training dirs if the table is empty.

    Returns DataFrame with columns: config_name, fold_id, ic_mean.
    Returns None if no fold metrics found.
    """
    case_dir = get_case_study_dir(case_study_id)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return None

    db = sqlite3.connect(str(db_path))

    # Primary source: fold_metrics table in registry DB
    try:
        query = """
            SELECT t.config_name, fm.fold_id, fm.ic AS ic_mean
            FROM fold_metrics fm
            JOIN prediction_sets p ON fm.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            WHERE t.family = 'gbm'
              AND p.split != 'holdout'
        """
        params: list = []
        if label is not None:
            query += " AND t.label = ?"
            params.append(label)
        rows = db.execute(query, params).fetchall()
    except sqlite3.Error as exc:
        warnings.warn(
            f"fold_metrics query failed for {case_study_id} (falling back to parquet): {exc}",
            stacklevel=2,
        )
        rows = []
    db.close()

    if rows:
        df = pl.DataFrame(
            {
                "config_name": [r[0] for r in rows],
                "fold_id": [r[1] for r in rows],
                "ic_mean": [r[2] for r in rows],
            },
        )
        df = df.with_columns(pl.lit(case_study_id).alias("dataset"))
        return df

    # Fallback: parquet files in training dirs
    db = sqlite3.connect(str(db_path))
    t_query = "SELECT training_hash, config_name FROM training_runs WHERE family = 'gbm'"
    t_params: list = []
    if label is not None:
        t_query += " AND label = ?"
        t_params.append(label)
    t_rows = db.execute(t_query, t_params).fetchall()
    db.close()

    frames = []
    for t_hash, config_name in t_rows:
        fm_path = case_dir / "run_log" / "training" / t_hash / "fold_metrics.parquet"
        if not fm_path.exists():
            continue
        try:
            df = pl.read_parquet(fm_path)
            if "config_name" not in df.columns:
                df = df.with_columns(pl.lit(config_name).alias("config_name"))
            if "dataset" not in df.columns:
                df = df.with_columns(pl.lit(case_study_id).alias("dataset"))
            frames.append(df)
        except Exception as exc:
            warnings.warn(f"Failed to read fold metrics {fm_path}: {exc}", stacklevel=2)
            continue

    if not frames:
        return None

    return pl.concat(frames)


# ---------------------------------------------------------------------------
# Daily-IC time series + indistinguishable-config detection
# ---------------------------------------------------------------------------


def load_daily_metrics_series(
    case_study_id: str,
    prediction_hash: str,
) -> pl.DataFrame:
    """Load the per-fold daily IC (and AUC if present) parquet for one prediction set.

    Returns the frame at `run_log/predictions/{hash}/daily_metrics.parquet`
    (shipped with the downloaded case-study artifacts). Use this for
    rolling-IC plots and re-running the bootstrap on the daily series
    without re-touching raw predictions. Empty DataFrame if missing.
    """
    case_dir = get_case_study_dir(case_study_id)
    path = case_dir / "run_log" / "predictions" / prediction_hash / "daily_metrics.parquet"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def indistinguishable_groups(
    metrics: pl.DataFrame,
    *,
    ic_col: str = "ic_mean_daily",
    lo_col: str = "ic_ci_lo",
    hi_col: str = "ic_ci_hi",
) -> pl.DataFrame:
    """Cluster rows whose HAC CIs overlap into indistinguishable groups.

    A simple sweep on the IC point estimate that emits a ``group`` integer per
    row: rows are in the same group iff their CIs intersect with the running
    upper bound. Useful for annotating leaderboards with "configs in group N
    are statistically indistinguishable at this CI level".

    Returns the input frame augmented with a ``group`` column.
    """
    if metrics.height == 0 or any(c not in metrics.columns for c in (ic_col, lo_col, hi_col)):
        return metrics

    ordered = metrics.sort(ic_col, descending=True, nulls_last=True)
    rows = list(ordered.iter_rows(named=True))
    groups: list[int] = []
    g = 0
    running_lo = float("inf")
    for r in rows:
        ic = r.get(ic_col)
        lo = r.get(lo_col)
        hi = r.get(hi_col)
        if ic is None or lo is None or hi is None:
            groups.append(-1)
            continue
        if not groups:
            groups.append(g)
            running_lo = float(lo)
            continue
        if float(hi) >= running_lo:
            groups.append(g)
            running_lo = max(running_lo, float(lo))
        else:
            g += 1
            groups.append(g)
            running_lo = float(lo)

    return ordered.with_columns(pl.Series("group", groups))
