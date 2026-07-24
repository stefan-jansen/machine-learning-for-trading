from __future__ import annotations

import polars as pl

from utils.paths import get_case_study_source_dir


def normalize_demo_predictions(df: pl.DataFrame, asset_column: str) -> pl.DataFrame:
    """Normalize prediction artifacts to ``timestamp``, ``asset_column``, ``prediction``."""
    rename_map = {}
    # Normalize entity column: accept "asset" or "symbol" as source
    if asset_column not in df.columns:
        if "asset" in df.columns:
            rename_map["asset"] = asset_column
        elif "symbol" in df.columns:
            rename_map["symbol"] = asset_column
    if "prediction" not in df.columns and "y_score" in df.columns:
        rename_map["y_score"] = "prediction"
    # Normalize legacy "date" column to canonical "timestamp"
    if "date" in df.columns and "timestamp" not in df.columns:
        rename_map["date"] = "timestamp"
    if rename_map:
        df = df.rename(rename_map)

    required = {"timestamp", asset_column, "prediction"}
    if required - set(df.columns):
        raise ValueError(
            f"Prediction artifact missing required columns: {required - set(df.columns)}"
        )

    ts = df["timestamp"]
    if ts.dtype == pl.Utf8:
        df = df.with_columns(pl.col("timestamp").str.to_date())
    elif ts.dtype in (pl.Datetime, pl.Date):
        df = df.with_columns(pl.col("timestamp").dt.date())

    return (
        df.select(["timestamp", asset_column, "prediction"])
        .group_by(["timestamp", asset_column])
        .agg(pl.col("prediction").mean().alias("prediction"))
        .sort([asset_column, "timestamp"])
    )


def load_demo_predictions(strategy_id: str, horizon: int, asset_column: str) -> pl.DataFrame | None:
    """Load prediction artifacts for live-demo notebooks.

    Searches committed model directories first, then falls back to the
    content-addressed registry (run_log/predictions/) which is the
    primary output of the model training pipeline. The registry fallback
    returns the sealed **holdout** split — the once-touched out-of-sample
    set — never validation, so a deployment export never ships predictions
    from the data used to select the model.
    """
    import os
    import sqlite3
    from pathlib import Path

    source_dir = get_case_study_source_dir(strategy_id)
    models_dir = source_dir / "models"
    candidates = [
        models_dir / "gbm" / f"fwd_ret_{horizon}d" / "predictions.parquet",
        models_dir / "linear" / f"fwd_ret_{horizon}d" / "predictions.parquet",
        models_dir / "deep_learning" / f"fwd_ret_{horizon}d" / "predictions.parquet",
        models_dir / "tabular_dl" / f"fwd_ret_{horizon}d" / "predictions.parquet",
    ]

    # Also check seeded predictions in ML4T_OUTPUT_DIR (CI / test mode)
    output_dir = os.environ.get("ML4T_OUTPUT_DIR", "")
    if output_dir:
        seeded_dir = Path(output_dir) / strategy_id / "models"
        candidates.append(seeded_dir / f"predictions_reg_{horizon}d.parquet")

    for path in candidates:
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        has_time = "date" in df.columns or "timestamp" in df.columns
        has_entity = "asset" in df.columns or "symbol" in df.columns
        has_score = "y_score" in df.columns or "prediction" in df.columns
        if not (has_time and has_entity and has_score):
            continue
        return normalize_demo_predictions(df, asset_column)

    # Fall back to registry — export the sealed HOLDOUT prediction set, never
    # validation. A deployment bridge that exported validation predictions would
    # be backtesting on the data used to select the model (leakage). The holdout
    # is the once-touched out-of-sample set, so it is the only honest thing to
    # ship to an external backtester.
    #
    # Pick that holdout by the *selected winner*, not by raw IC: the carrier is
    # the cross-stage validation-Sharpe rank-1 config, pooling the selection
    # stages (signal/allocation/risk_overlay). cost_sensitivity is a
    # perturbation, not a selection axis, and the holdout stage is the sealed
    # set itself, so both are excluded. The winner's training run owns exactly
    # one holdout prediction set (the "one holdout per case study" rule), which
    # is what we export. This also derives the label/horizon from the winner
    # rather than trusting the caller's `horizon`. The IC-sorted query is kept
    # only as a fallback for case studies that have predictions but no backtest
    # stages recorded.
    registry_db = source_dir / "run_log" / "registry.db"
    if registry_db.exists():
        conn = sqlite3.connect(str(registry_db))
        pred_hash = None
        winner = conn.execute(
            """SELECT ps.training_hash
               FROM backtest_runs br
               JOIN backtest_metrics bm ON br.backtest_hash = bm.backtest_hash
               JOIN prediction_sets ps ON br.prediction_hash = ps.prediction_hash
               WHERE br.stage IN ('signal', 'allocation', 'risk_overlay')
               ORDER BY bm.sharpe DESC LIMIT 1""",
        ).fetchone()
        if winner:
            row = conn.execute(
                """SELECT prediction_hash FROM prediction_sets
                   WHERE training_hash = ? AND split = 'holdout'
                   ORDER BY prediction_hash LIMIT 1""",
                (winner[0],),
            ).fetchone()
            if row:
                pred_hash = row[0]
        if pred_hash is None:
            label = f"fwd_ret_{horizon}d"
            row = conn.execute(
                """SELECT ps.prediction_hash
                   FROM prediction_sets ps
                   JOIN training_runs tr ON ps.training_hash = tr.training_hash
                   JOIN prediction_metrics pm ON ps.prediction_hash = pm.prediction_hash
                   WHERE tr.label = ? AND ps.split = 'holdout'
                   ORDER BY pm.ic_mean DESC LIMIT 1""",
                (label,),
            ).fetchone()
            if row:
                pred_hash = row[0]
        conn.close()
        if pred_hash:
            pred_path = source_dir / "run_log" / "predictions" / pred_hash / "predictions.parquet"
            if pred_path.exists():
                df = pl.read_parquet(pred_path)
                return normalize_demo_predictions(df, asset_column)

    return None
