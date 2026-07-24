"""Shared signal-to-weight conversion for Chapter 16 validation.

Converts ML prediction artifacts into a target-weight time series that all
three frameworks (ml4t-backtest, VectorBT, Backtrader) consume identically.

This eliminates signal-generation differences and isolates execution-engine parity.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from utils import ML4T_DATA_PATH
from utils.paths import get_case_study_dir

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

COST_BPS: dict[str, float] = {
    "etfs": 10,
    "crypto_perps_funding": 3,
    "fx_pairs": 3,
    "cme_futures": 5,
    "sp500_equity_option_analytics": 7,
    "nasdaq100_microstructure": 5,
    "us_equities_panel": 12,
}

SKIP_CASE_STUDIES = {"sp500_options", "us_firm_characteristics"}


# --------------------------------------------------------------------------- #
# Price loading (reuses patterns from run_linear_backtests.py)
# --------------------------------------------------------------------------- #
def _base_rename(df: pl.DataFrame, ts_col: str, asset_col: str) -> pl.DataFrame:
    """Rename timestamp and asset columns to standard names."""
    renames = {}
    if ts_col != "timestamp":
        renames[ts_col] = "timestamp"
    if asset_col != "symbol":
        renames[asset_col] = "symbol"
    if renames:
        df = df.rename(renames)
    if df["timestamp"].dtype == pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    df = df.drop_nulls("close")
    return df


def _first_existing_path(*candidates: Path) -> Path:
    """Return the first available artifact path from current and legacy layouts."""
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Artifact not found. Checked:\n{checked}")


def _resolve_column(df: pl.DataFrame, *candidates: str) -> str:
    """Return the first matching column name from a list of schema variants."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"None of the expected columns exist: {candidates}. Found: {df.columns}")


_LOADERS: dict[str, str] = {
    "etfs": "data.load_etfs",
    "crypto_perps_funding": "data.load_crypto_perps",
    "fx_pairs": "data.load_fx_pairs",
    "cme_futures": "data.load_cme_futures",
    "us_equities_panel": "data.load_us_equities",
    "nasdaq100_microstructure": "data.load_nasdaq100_bars",
    "sp500_equity_option_analytics": "data.load_sp500_daily_bars",
    "us_firm_characteristics": "data.load_firm_characteristics",
}


def load_prices(case_study: str) -> pl.DataFrame:
    """Load OHLCV prices with columns: [timestamp, asset, open, high, low, close, volume]."""
    case_dir = get_case_study_dir(case_study, create=False)
    candidates = [
        case_dir / "data" / "labels" / "prices.parquet",
        case_dir / "labels" / "prices.parquet",
        ML4T_DATA_PATH / "intermediates" / case_study / "labels" / "prices.parquet",
    ]

    prices_path = None
    for p in candidates:
        if p.exists():
            prices_path = p
            break

    if prices_path is not None:
        df = pl.read_parquet(prices_path)
    elif case_study in _LOADERS:
        # Fall back to data provider when cached prices don't exist
        import importlib

        module_name, func_name = _LOADERS[case_study].rsplit(".", 1)
        mod = importlib.import_module(module_name)
        df = getattr(mod, func_name)()
    else:
        checked = "\n".join(f"- {p}" for p in candidates)
        raise FileNotFoundError(
            f"No prices found for '{case_study}'. Checked:\n{checked}\n"
            f"No data loader registered. Run the labels pipeline first."
        )

    # CME daily bars carry `session_date` rather than `timestamp`.
    ts_col = _resolve_column(df, "timestamp", "date", "session_date")
    asset_col = _resolve_column(df, "symbol", "asset", "product", "stock_id")

    # Front month only. Cached label artifacts name the tenor `position`; the
    # canonical loader returns `tenor`. Without this the panel carries three rows
    # per (date, product).
    if case_study == "cme_futures":
        tenor_col = next((c for c in ("position", "tenor") if c in df.columns), None)
        if tenor_col:
            df = df.filter(pl.col(tenor_col) == 0)

    if case_study == "us_equities_panel" and "adj_open" in df.columns:
        df = df.select(
            ts_col,
            asset_col,
            pl.col("adj_open").alias("open"),
            pl.col("adj_high").alias("high"),
            pl.col("adj_low").alias("low"),
            pl.col("adj_close").alias("close"),
            pl.col("adj_volume").alias("volume"),
        )
    elif case_study == "cme_futures" and "adj_close" in df.columns:
        # Backtest fills ride the roll-adjusted series; volume is unadjusted.
        df = df.select(
            ts_col,
            asset_col,
            pl.col("adj_open").alias("open"),
            pl.col("adj_high").alias("high"),
            pl.col("adj_low").alias("low"),
            pl.col("adj_close").alias("close"),
            "volume",
        )
    elif case_study == "nasdaq100_microstructure" and "mid_close" in df.columns:
        df = df.select(
            ts_col,
            asset_col,
            pl.col("mid_close").alias("open"),
            pl.col("mid_close").alias("high"),
            pl.col("mid_close").alias("low"),
            pl.col("mid_close").alias("close"),
            pl.col("volume").cast(pl.Float64),
        )
    else:
        df = df.select(ts_col, asset_col, "open", "high", "low", "close", "volume")

    return _base_rename(df, ts_col, asset_col)


def _find_linear_predictions(case_study: str, label: str) -> Path:
    """Find linear prediction parquet via the model registry.

    Prefers consolidated runs (multiple models in one file) over per-config runs.
    Falls back to the per-config run with the highest IC.
    """
    case_dir = get_case_study_dir(case_study, create=False)
    run_log = case_dir / "run_log"

    # Current schema: registry.db with training_runs + prediction_sets
    registry_path = run_log / "registry.db"
    if registry_path.exists():
        db = sqlite3.connect(str(registry_path))
        try:
            # Find linear prediction sets, ordered by IC
            rows = db.execute(
                "SELECT ps.prediction_hash "
                "FROM training_runs tr "
                "JOIN prediction_sets ps ON tr.training_hash = ps.training_hash "
                "LEFT JOIN prediction_metrics pm ON ps.prediction_hash = pm.prediction_hash "
                "WHERE tr.family = 'linear' AND tr.label = ? "
                "ORDER BY pm.ic_mean DESC NULLS LAST, tr.created_at DESC",
                (label,),
            ).fetchall()
            for (pred_hash,) in rows:
                pred = run_log / "predictions" / pred_hash / "predictions.parquet"
                if pred.exists():
                    return pred
        finally:
            db.close()

    # Legacy schema: models.db with model_runs table
    for db_path in [run_log / "models.db", run_log / "models" / "models.db"]:
        if not db_path.exists():
            continue
        db = sqlite3.connect(str(db_path))
        try:
            rows = db.execute(
                "SELECT hash, config_json FROM model_runs "
                "WHERE family='linear' AND label=? AND config_json LIKE '%n_specs%' "
                "ORDER BY created_at DESC",
                (label,),
            ).fetchall()
            for run_hash, _ in rows:
                pred = run_log / "models" / "runs" / run_hash / "predictions.parquet"
                if pred.exists():
                    return pred

            rows = db.execute(
                "SELECT hash, ic_mean FROM model_runs "
                "WHERE family='linear' AND label=? "
                "ORDER BY ic_mean DESC",
                (label,),
            ).fetchall()
            for run_hash, _ in rows:
                pred = run_log / "models" / "runs" / run_hash / "predictions.parquet"
                if pred.exists():
                    return pred
        finally:
            db.close()

    # Legacy path fallback (models/linear/{label}/predictions.parquet)
    legacy_candidates = [
        case_dir / "models" / "linear" / label / "predictions.parquet",
        REPO_ROOT
        / "case_studies"
        / case_study
        / "models"
        / "linear"
        / label
        / "predictions.parquet",
    ]
    for c in legacy_candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"No linear predictions found for {case_study}/{label}. "
        f"Run the linear model notebook first."
    )


def _detect_linear_labels(case_study: str) -> list[str]:
    """Detect available linear labels from the model registry."""
    case_dir = get_case_study_dir(case_study, create=False)
    run_log = case_dir / "run_log"

    # Current schema: registry.db with training_runs
    registry_path = run_log / "registry.db"
    if registry_path.exists():
        db = sqlite3.connect(str(registry_path))
        try:
            rows = db.execute(
                "SELECT DISTINCT label FROM training_runs WHERE family='linear'"
            ).fetchall()
            if rows:
                return sorted(r[0] for r in rows)
        finally:
            db.close()

    # Legacy schema: models.db with model_runs
    for db_path in [run_log / "models.db", run_log / "models" / "models.db"]:
        if not db_path.exists():
            continue
        db = sqlite3.connect(str(db_path))
        try:
            rows = db.execute(
                "SELECT DISTINCT label FROM model_runs WHERE family='linear'"
            ).fetchall()
            if rows:
                return sorted(r[0] for r in rows)
        finally:
            db.close()

    # Legacy fallback: scan models/linear/ directory
    linear_dir = REPO_ROOT / "case_studies" / case_study / "models" / "linear"
    if linear_dir.exists():
        return [d.name for d in sorted(linear_dir.iterdir()) if d.is_dir()]

    return []


def load_signals(case_study: str, label: str, best_model: str) -> pl.DataFrame:
    """Load prediction signals with columns: [timestamp, symbol, prediction]."""
    pred_path = _find_linear_predictions(case_study, label)
    df = pl.read_parquet(pred_path)

    # Filter by model if the column exists (legacy consolidated files)
    if "model" in df.columns:
        df = df.filter(pl.col("model") == best_model)

    ts_col = _resolve_column(df, "timestamp", "date")
    asset_col = _resolve_column(df, "symbol", "asset", "product", "stock_id")
    score_col = _resolve_column(df, "prediction", "y_score")

    result = df.select(
        pl.col(ts_col).alias("timestamp"),
        pl.col(asset_col).cast(pl.String).alias("symbol"),
        pl.col(score_col).alias("prediction"),
    )
    if result["timestamp"].dtype == pl.Date:
        result = result.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    return result


def get_best_model(case_study: str, label: str) -> str:
    """Get best linear model from registry, validated against available predictions."""
    from case_studies.utils.analytics import load_best_ic_per_family

    best_model = "elastic_net"

    _best = load_best_ic_per_family(["linear"], case_studies=[case_study])
    if not _best.is_empty():
        _match = _best.filter(pl.col("label") == label)
        if not _match.is_empty():
            best_model = str(_match["config_name"][0])

    # Validate against available models in predictions
    try:
        pred_path = _find_linear_predictions(case_study, label)
    except FileNotFoundError:
        return best_model

    schema = pl.read_parquet_schema(pred_path)
    if "model" not in schema:
        return best_model

    available_models = (
        pl.read_parquet(pred_path, columns=["model"])["model"]
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    if best_model in available_models:
        return best_model

    # Best model not in predictions — pick the best available by IC from registry
    _all_linear = load_best_ic_per_family(["linear"], case_studies=[case_study])
    if not _all_linear.is_empty():
        ranked_available = sorted(
            (
                (m, float(_all_linear.filter(pl.col("config_name") == m)["ic_mean"][0]))
                for m in available_models
                if not _all_linear.filter(pl.col("config_name") == m).is_empty()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked_available:
            fallback_model = ranked_available[0][0]
            print(
                f"Best model '{best_model}' missing from {pred_path.name}; "
                f"using available model '{fallback_model}' instead."
            )
            return fallback_model

    return best_model


# --------------------------------------------------------------------------- #
# Weight generation
# --------------------------------------------------------------------------- #
def predictions_to_weights(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    strategy: str = "top_quintile",
    long_only: bool = True,
    min_assets: int = 5,
) -> pd.DataFrame:
    """Convert predictions to a target-weight DataFrame.

    Returns:
        pd.DataFrame with DatetimeIndex, one column per asset, values = target weights.
        All NaN rows (dates without enough signals) are dropped.
    """
    # Align timezone/time-unit between prices and signals
    signals = _align_timestamps(prices, signals)

    # Get dates where we have signals
    signal_dates = signals["timestamp"].unique().sort()

    # Get all assets
    all_assets = sorted(prices["symbol"].unique().to_list())

    rows = []
    dates = []

    for dt in signal_dates:
        sigs = signals.filter(pl.col("timestamp") == dt)
        if len(sigs) < min_assets:
            continue

        if strategy == "top_quintile":
            weights = _top_quintile_weights(sigs, all_assets, long_only)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        rows.append(weights)
        dates.append(dt)

    if not rows:
        return pd.DataFrame()

    # Convert Polars datetime to pandas
    dates_pd = pd.DatetimeIndex(
        [d.to_pydatetime() if hasattr(d, "to_pydatetime") else d for d in dates]
    )

    weights_df = pd.DataFrame(rows, index=dates_pd, columns=all_assets).fillna(0.0)

    # Round to 6 decimals to avoid floating-point noise across frameworks
    weights_df = weights_df.round(6)

    return weights_df


def _top_quintile_weights(
    signals: pl.DataFrame, all_assets: list[str], long_only: bool
) -> dict[str, float]:
    """Compute top-quintile equal-weight allocation."""
    sorted_sigs = signals.sort("prediction", descending=True)
    n = len(sorted_sigs)
    top_n = max(1, n // 5)
    weight = round(1.0 / top_n, 6)

    top_assets = sorted_sigs["symbol"].head(top_n).to_list()
    return {a: weight if a in top_assets else 0.0 for a in all_assets}


def _align_timestamps(prices: pl.DataFrame, signals: pl.DataFrame) -> pl.DataFrame:
    """Align signal timestamps to match prices dtype."""
    if prices["timestamp"].dtype != signals["timestamp"].dtype:
        if hasattr(prices["timestamp"].dtype, "time_zone") and prices["timestamp"].dtype.time_zone:
            prices_tz = prices["timestamp"].dtype.time_zone
        else:
            prices_tz = None
        if (
            hasattr(signals["timestamp"].dtype, "time_zone")
            and signals["timestamp"].dtype.time_zone
        ):
            signals = signals.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        if prices_tz:
            signals = signals.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

        target_tu = (
            prices["timestamp"].dtype.time_unit
            if hasattr(prices["timestamp"].dtype, "time_unit")
            else "us"
        )
        signals = signals.with_columns(pl.col("timestamp").cast(pl.Datetime(target_tu)))
    return signals


# --------------------------------------------------------------------------- #
# Convenience: load everything for a case study
# --------------------------------------------------------------------------- #
def _truncate_to_common_end(
    prices: pl.DataFrame, weights: pd.DataFrame, coverage_pct: float = 0.90
) -> tuple[pl.DataFrame, pd.DataFrame]:
    """Truncate simulation to last date where >=coverage_pct of weighted assets have prices.

    When assets have different end dates, VBT and ml4t handle stale positions
    differently (VBT closes on NaN, ml4t carries at last known price). This
    truncates to the point where most assets still have coverage.

    Delistments (assets ending months/years early) are fine — both engines handle
    them. The problem is the trailing edge where a few assets continue past the
    majority, leaving positions un-valued in some engines.
    """
    ever_weighted = set(c for c in weights.columns if (weights[c] != 0).any())
    if not ever_weighted:
        return prices, weights

    asset_ends = (
        prices.filter(pl.col("symbol").is_in(ever_weighted))
        .group_by("symbol")
        .agg(pl.col("timestamp").max().alias("end"))
    )
    n_assets = len(asset_ends)
    if n_assets < 2:
        return prices, weights

    # Sort end dates descending and find the cutoff where coverage_pct of assets are covered
    end_dates = asset_ends["end"].sort(descending=True).to_list()
    cutoff_idx = int(n_assets * coverage_pct) - 1
    cutoff_idx = max(0, min(cutoff_idx, n_assets - 1))
    cutoff_date = end_dates[cutoff_idx]

    max_end = end_dates[0]
    if cutoff_date == max_end:
        return prices, weights

    prices = prices.filter(pl.col("timestamp") <= cutoff_date)
    cutoff_pd = pd.Timestamp(cutoff_date)
    if cutoff_pd.tzinfo is not None:
        cutoff_pd = cutoff_pd.tz_localize(None)
    if weights.index.tz is not None:
        cutoff_pd = cutoff_pd.tz_localize(weights.index.tz)
    weights = weights.loc[:cutoff_pd]

    return prices, weights


def load_case_study_data(
    case_study: str,
    label: str | None = None,
    model: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pd.DataFrame, dict]:
    """Load prices, signals, and pre-computed weights for a case study.

    Returns:
        (prices_pl, signals_pl, weights_pd, metadata)
    """
    if case_study in SKIP_CASE_STUDIES:
        raise ValueError(f"Cannot backtest {case_study}")

    # Auto-detect label if not provided
    if label is None:
        labels = _detect_linear_labels(case_study)
        if not labels:
            raise FileNotFoundError(
                f"No linear model labels found for {case_study}. "
                f"Run the linear model notebook first."
            )
        label = labels[0]

    if model is None:
        model = get_best_model(case_study, label)

    prices = load_prices(case_study)
    signals = load_signals(case_study, label, model)

    n_assets = signals["symbol"].n_unique()
    min_assets = min(5, max(2, n_assets // 5))

    weights = predictions_to_weights(
        prices, signals, strategy="top_quintile", min_assets=min_assets
    )

    # Truncate trailing coverage gaps to prevent engine divergence
    prices, weights = _truncate_to_common_end(prices, weights)

    metadata = {
        "case_study": case_study,
        "label": label,
        "model": model,
        "cost_bps": COST_BPS.get(case_study, 10),
        "n_assets": n_assets,
        "n_signal_dates": len(weights),
        "min_assets": min_assets,
    }

    return prices, signals, weights, metadata
