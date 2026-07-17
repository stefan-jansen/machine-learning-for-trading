"""Universal loader for Ch16-19 backtesting case study notebooks.

Replaces the outdated 16_strategy_simulation/code/prediction_loader.py.
Handles schema normalization across all 9 case studies and 5 model families.

Functions:
    load_backtest_predictions(): Load + normalize prediction artifacts
    load_backtest_prices(): Load + normalize price data for DataFeed
    build_target_weights(): Convert predictions → portfolio weights (delegated to utils.signals)
    get_backtest_config(): Extract costs, calendar, rebalance config from setup.yaml
    compute_allocator_metrics(): Compute 11-metric allocator summary in one call
    compute_dsr_table(): DSR for all model variants (selection-bias accounting)
    extract_daily_returns_frame(): Extract daily returns from BacktestResult
    aggregate_timestamped_returns_to_daily(): Aggregate timestamped returns to daily
    infer_session_alignment(): Infer whether returns need session alignment

Usage:
    from case_studies.utils.backtest_loaders import (
        load_backtest_predictions,
        load_backtest_prices,
        build_target_weights,
        get_backtest_config,
        compute_allocator_metrics,
        compute_dsr_table,
    )
"""

from __future__ import annotations

import re
import sqlite3
import warnings
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import yaml

from case_studies.utils.notebook_contracts import degenerate_prediction_sql
from case_studies.utils.registry import model_source
from case_studies.utils.signals import build_target_weights
from utils.artifact_specs import resolve_market_runtime, resolve_market_semantics
from utils.paths import get_case_study_dir

if TYPE_CHECKING:
    from ml4t.backtest.result import BacktestResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_CASE_STUDIES = [
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "us_firm_characteristics",
    "fx_pairs",
    "cme_futures",
    "sp500_options",
    "us_equities_panel",
]

MODEL_FAMILIES = ["linear", "gbm", "tabular_dl", "deep_learning", "latent_factors"]

# Columns that could be entity identifiers in predictions
_ENTITY_COLS = {"symbol", "product", "stock_id", "entity", "instrument_id", "asset"}

# Columns that could be time identifiers
_TIME_COLS = {"date", "timestamp", "session_date"}

# Normalized output schema for predictions
_PRED_SCHEMA = ["timestamp", "symbol", "y_score", "y_true", "fold_id", "model_id", "source"]

# Case studies that use vectorized (non-Engine) backtesting
VECTORIZED_CASE_STUDIES = {"us_firm_characteristics", "sp500_options"}


@cache
def _load_backtest_preset_config(case_study_id: str) -> dict:
    """Load the case-study backtest preset if present."""
    case_dir = get_case_study_dir(case_study_id)
    path = case_dir / "config" / "backtest" / "base.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _preset_requests_quotes(case_study_id: str) -> bool:
    """Return True when the backtest preset requires bid/ask columns."""
    preset = _load_backtest_preset_config(case_study_id)
    feed = preset.get("feed", {})
    execution = preset.get("execution", {})
    return bool(feed.get("bid_col") or feed.get("ask_col")) or (
        execution.get("execution_price") in {"bid", "ask", "quote_mid", "quote_side"}
        or execution.get("mark_price") in {"bid", "ask", "quote_mid", "quote_side"}
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BacktestPredictions:
    """Container for normalized prediction data."""

    predictions: pl.DataFrame  # [timestamp, symbol, y_score, y_true, fold_id, model_id, source]
    case_study_id: str
    label: str
    model_families: list[str]
    n_assets: int
    n_timestamps: int
    date_range: tuple[str, str]
    sources: dict[str, int] = field(default_factory=dict)  # {family: n_rows}
    registry_entries: list[dict] = field(default_factory=list)


@dataclass
class BacktestConfig:
    """Configuration extracted from setup.yaml for backtesting."""

    case_study_id: str
    primary_label: str
    label_buffer: str
    calendar: str
    cadence: str
    execution_delay: str
    commission_bps: float  # Normalized single number (midpoint of range)
    slippage_bps: float  # Estimated slippage
    costs_class: str  # "material" or "negligible"
    long_short: bool
    holdout_start: str
    holdout_end: str
    n_splits: int
    raw_costs: dict  # Original costs section from setup.yaml
    min_weight_change: float = 0.005  # Engine rebalance threshold (skip < this)
    min_trade_value: float = 100.0  # Engine rebalance threshold (skip < this $)
    initial_cash: float = 100_000.0  # SSOT — set from setup.yaml::execution.initial_cash
    share_type: str = "integer"  # SSOT — set from setup.yaml::execution.share_type


def load_contract_specs_from_yaml(yaml_path: Path | None = None):
    """Load futures contract specs from YAML, deriving multipliers from tick values."""
    from ml4t.backtest import AssetClass, ContractSpec

    if yaml_path is None:
        repo_root = Path(__file__).resolve().parents[2]  # case_studies/utils/ → repo root
        candidates = [
            repo_root / "data" / "futures" / "market" / "futures_specs.yaml",
            repo_root / "data" / "futures" / "futures_specs.yaml",
            repo_root / "data" / "_archive" / "config" / "futures_specs.yaml",
        ]
        yaml_path = next((path for path in candidates if path.exists()), candidates[0])

    with yaml_path.open() as f:
        raw = yaml.safe_load(f)

    specs = {}
    for symbol, info in raw["products"].items():
        init_pct = info.get("initial_margin_pct")
        maint_pct = info.get("maintenance_margin_pct")
        if (init_pct is None) != (maint_pct is None):
            raise ValueError(
                f"{symbol}: must specify both initial_margin_pct and "
                f"maintenance_margin_pct or neither "
                f"(got init={init_pct}, maint={maint_pct})"
            )
        margin_pct = (init_pct, maint_pct) if init_pct is not None else None
        specs[symbol] = ContractSpec(
            symbol=symbol,
            asset_class=AssetClass.FUTURE,
            multiplier=info["tick_value"] / info["tick_size"],
            tick_size=info["tick_size"],
            margin_pct=margin_pct,
        )
    return specs


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------


def _detect_entity_col(df: pl.DataFrame) -> str | None:
    """Detect entity column from a DataFrame."""
    for col in _ENTITY_COLS:
        if col in df.columns:
            return col
    return None


def _detect_time_col(df: pl.DataFrame) -> str | None:
    """Detect time column from a DataFrame."""
    for col in _TIME_COLS:
        if col in df.columns:
            return col
    return None


def _normalize_predictions(
    df: pl.DataFrame,
    source: str,
    case_study_id: str,
) -> pl.DataFrame:
    """Normalize a prediction DataFrame to the canonical schema.

    Handles two prediction schemas:
    - Linear: [date/timestamp, entity, fold, model, prediction, actual]
    - GBM/DL/Latent: [date/timestamp, entity, y_true, y_score, fold_id, model_id]

    Returns: [timestamp, symbol, y_score, y_true, fold_id, model_id, source]
    """
    time_col = _detect_time_col(df)
    entity_col = _detect_entity_col(df)

    if time_col is None:
        msg = f"No time column found in {source} predictions for {case_study_id}. Columns: {df.columns}"
        raise ValueError(msg)

    # --- Rename columns to canonical names ---
    renames = {}

    # Time → timestamp
    if time_col != "timestamp":
        renames[time_col] = "timestamp"

    # Entity → symbol
    if entity_col and entity_col != "symbol" and "symbol" not in df.columns:
        renames[entity_col] = "symbol"

    # Linear schema: prediction→y_score, actual→y_true, fold→fold_id, model→model_id
    if "prediction" in df.columns:
        renames["prediction"] = "y_score"
        renames["actual"] = "y_true"
        renames["fold"] = "fold_id"
        renames["model"] = "model_id"
    elif "model_id" not in df.columns and "config" in df.columns:
        renames["config"] = "model_id"

    if renames:
        df = df.rename(renames)

    if case_study_id == "cme_futures" and "position" in df.columns:
        df = df.filter(pl.col("position") == 0)

    # --- Type normalization ---

    # Cast date types to Datetime for consistent timestamp column
    ts_dtype = df.schema["timestamp"]
    if ts_dtype == pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    elif ts_dtype == pl.String or ts_dtype == pl.Utf8:
        # String timestamps (e.g., latent_factors PCA) — parse to Datetime
        df = df.with_columns(pl.col("timestamp").str.to_datetime().cast(pl.Datetime("us")))
    elif hasattr(ts_dtype, "time_zone") and ts_dtype.time_zone:
        # Strip timezone (crypto has UTC)
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    # Ensure fold_id is Int64
    if "fold_id" in df.columns and df.schema["fold_id"] != pl.Int64:
        df = df.with_columns(pl.col("fold_id").cast(pl.Int64))

    # Ensure model_id is String
    if "model_id" in df.columns and df.schema["model_id"] != pl.String:
        df = df.with_columns(pl.col("model_id").cast(pl.String))

    # Ensure symbol is String (us_firm has UInt32 stock_id)
    if "symbol" in df.columns and df.schema["symbol"] != pl.String:
        df = df.with_columns(pl.col("symbol").cast(pl.String))

    # Add source column
    df = df.with_columns(pl.lit(source).alias("source"))

    # Select only canonical columns (drop extras like position, instrument_id duplicates)
    keep = [c for c in _PRED_SCHEMA if c in df.columns]
    return df.select(keep)


def _load_registry_prediction_frames(
    case_study_id: str,
    case_dir: Path,
    label: str,
    model_families: list[str],
    split: str,
    best_only: bool,
) -> tuple[list[pl.DataFrame], dict[str, int], list[dict]]:
    # Use the new registry at run_log/registry.db (SSOT since registry redesign)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return [], {}, []

    requested = set(model_families)

    # Query prediction_sets JOIN training_runs for label + split filtering
    conn = sqlite3.connect(str(db_path))
    try:
        query = """
            SELECT
                p.prediction_hash,
                p.training_hash,
                p.split,
                t.family,
                t.config_name,
                t.label,
                t.created_at
            FROM prediction_sets p
            JOIN training_runs t ON p.training_hash = t.training_hash
            WHERE t.label = ?
        """
        # Drop prediction sets with any constant-prediction (NULL-IC) fold so a
        # degenerate L1/EN model is never backtested — see degenerate_prediction_sql().
        query += degenerate_prediction_sql("p.prediction_hash")
        params: list[str] = [label]

        if split == "validation":
            query += " AND p.split = 'validation'"
        elif split == "holdout":
            query += " AND p.split = 'holdout'"
        # split == "all" → no additional filter

        rows = conn.execute(query, params).fetchall()
        col_names = [
            "prediction_hash",
            "training_hash",
            "split",
            "family",
            "config_name",
            "label",
            "created_at",
        ]
    finally:
        conn.close()

    frames: list[pl.DataFrame] = []
    sources: dict[str, int] = {}
    entries: list[dict] = []

    for row_tuple in rows:
        row = dict(zip(col_names, row_tuple, strict=False))
        family = str(row.get("family", "")).strip()
        if family not in requested:
            continue

        prediction_hash = row["prediction_hash"]
        pred_path = case_dir / "run_log" / "predictions" / prediction_hash / "predictions.parquet"
        if not pred_path.exists():
            continue

        try:
            raw = pl.read_parquet(pred_path)
        except Exception as exc:
            warnings.warn(f"Failed to read predictions {pred_path}: {exc}", stacklevel=2)
            continue
        if raw.is_empty():
            continue

        source = model_source(family, row.get("config_name"))
        run_split = row.get("split", "validation")
        if split == "all" and run_split == "holdout":
            source = f"{source}/holdout"
        normalized = _normalize_predictions(raw, source, case_study_id)

        if best_only and family != "latent_factors":
            normalized = _select_best_predictions(normalized)

        frames.append(normalized)
        source_counts = normalized.group_by("source").agg(n=pl.len())
        for count_row in source_counts.iter_rows(named=True):
            source_name = count_row["source"]
            sources[source_name] = sources.get(source_name, 0) + count_row["n"]

        entries.append(
            {
                "hash": prediction_hash,
                "family": family,
                "label": row.get("label"),
                "created_at": row.get("created_at"),
                "source": source,
                "predictions_path": str(pred_path),
            }
        )

    return frames, sources, entries


def _load_cme_front_month_targets(case_dir: Path, label: str) -> pl.DataFrame | None:
    label_path = case_dir / "labels" / f"{label}.parquet"
    if not label_path.exists():
        return None
    try:
        labels = pl.read_parquet(label_path)
    except Exception as exc:
        warnings.warn(f"Failed to read labels {label_path}: {exc}", stacklevel=2)
        return None
    if label not in labels.columns or "position" not in labels.columns:
        return None
    time_col = (
        "timestamp"
        if "timestamp" in labels.columns
        else "date"
        if "date" in labels.columns
        else None
    )
    asset_col = (
        "product"
        if "product" in labels.columns
        else "symbol"
        if "symbol" in labels.columns
        else None
    )
    if time_col is None or asset_col is None:
        return None
    return (
        labels.filter(pl.col("position") == 0)
        .select(
            [
                pl.col(time_col).cast(pl.Datetime("us")).alias("timestamp"),
                pl.col(asset_col).cast(pl.String).alias("symbol"),
                pl.col(label).cast(pl.Float64).alias("_front_y_true"),
            ]
        )
        .unique(subset=["timestamp", "symbol"], keep="first")
    )


def _select_best_predictions(df: pl.DataFrame) -> pl.DataFrame:
    """Select best model per fold from multi-model prediction files.

    For files with multiple models (linear has 9+, GBM has multiple configs),
    select the model with highest mean |y_score| correlation with y_true per fold.
    """
    if "model_id" not in df.columns:
        return df

    n_models = df["model_id"].n_unique()
    if n_models <= 1:
        return df

    # Compute rank IC per model across all folds
    model_ics = (
        df.group_by("model_id")
        .agg(
            ic=pl.corr("y_score", "y_true", method="spearman"),
            n_obs=pl.len(),
        )
        .sort("ic", descending=True)
    )

    best_model = model_ics.row(0, named=True)["model_id"]
    return df.filter(pl.col("model_id") == best_model)


def load_backtest_predictions(
    case_study_id: str,
    label: str | None = None,
    model_families: list[str] | None = None,
    best_only: bool = True,
    split: str = "validation",
    use_registry: bool | None = None,  # Deprecated — registry is always primary
) -> BacktestPredictions:
    """Load and normalize prediction artifacts for backtesting.

    The model registry (``registry.db`` / ``models.db``) is the source of truth.
    Predictions are loaded from content-addressed run directories
    (``run_log/models/runs/{hash}/`` or ``models/runs/{hash}/``), keyed by
    the registry's ``model_runs`` table.

    Args:
        case_study_id: Case study identifier (e.g., "etfs", "cme_futures")
        label: Target label (e.g., "fwd_ret_21d"). None = primary from setup.yaml.
        model_families: List of families to load. None = all available.
        best_only: If True, select best model per family. If False, return all.
        split: Which prediction split to load: "validation", "holdout", or "all".
        use_registry: Deprecated — ignored. Registry is always used.

    Returns:
        BacktestPredictions with normalized [timestamp, symbol, y_score, y_true,
        fold_id, model_id, source] DataFrame.
    """
    case_dir = get_case_study_dir(case_study_id)

    # Resolve label from setup.yaml if not provided
    if label is None:
        setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
        label = setup["labels"]["primary"]

    if model_families is None:
        model_families = MODEL_FAMILIES

    valid_splits = {"validation", "holdout", "all"}
    if split not in valid_splits:
        msg = f"Invalid split='{split}'. Must be one of {sorted(valid_splits)}"
        raise ValueError(msg)

    cme_front_targets = (
        _load_cme_front_month_targets(case_dir, label) if case_study_id == "cme_futures" else None
    )

    # --- Primary source: registry-backed content-addressed runs ---
    frames, sources, registry_entries = _load_registry_prediction_frames(
        case_study_id=case_study_id,
        case_dir=case_dir,
        label=label,
        model_families=model_families,
        split=split,
        best_only=best_only,
    )

    if not frames:
        msg = f"No predictions found for {case_study_id}/{label} in families {model_families}"
        raise FileNotFoundError(msg)

    predictions = pl.concat(frames, how="diagonal_relaxed")
    if case_study_id == "cme_futures":
        if (
            cme_front_targets is not None
            and "timestamp" in predictions.columns
            and cme_front_targets.schema.get("timestamp") != predictions.schema.get("timestamp")
        ):
            cme_front_targets = cme_front_targets.with_columns(
                pl.col("timestamp").cast(predictions.schema["timestamp"])
            )
        base_sort = ["timestamp", "symbol", "source"]
        fold_sort = ["fold_id"] if "fold_id" in predictions.columns else []
        fold_desc = [True] if fold_sort else []
        if cme_front_targets is not None and {"timestamp", "symbol", "y_true"}.issubset(
            predictions.columns
        ):
            predictions = predictions.join(
                cme_front_targets, on=["timestamp", "symbol"], how="left"
            )
            predictions = predictions.with_columns(
                (pl.col("y_true") - pl.col("_front_y_true")).abs().alias("_front_err")
            )
            predictions = predictions.sort(
                by=base_sort + ["_front_err"] + fold_sort,
                descending=[False, False, False, False] + fold_desc,
                nulls_last=True,
            )
            predictions = predictions.unique(subset=["timestamp", "symbol", "source"], keep="first")
            predictions = predictions.drop(["_front_y_true", "_front_err"])
        else:
            predictions = predictions.sort(
                by=base_sort + fold_sort,
                descending=[False, False, False] + fold_desc,
                nulls_last=True,
            ).unique(subset=["timestamp", "symbol", "source"], keep="first")
        predictions = predictions.sort(["source", "timestamp", "symbol"])
        sources = {
            row["source"]: row["n"]
            for row in predictions.group_by("source").agg(pl.len().alias("n")).iter_rows(named=True)
        }

    # Compute summary stats
    n_assets = predictions["symbol"].n_unique() if "symbol" in predictions.columns else 0
    ts_col = "timestamp"
    n_timestamps = predictions[ts_col].n_unique()
    date_range = (
        str(predictions[ts_col].min()),
        str(predictions[ts_col].max()),
    )

    return BacktestPredictions(
        predictions=predictions,
        case_study_id=case_study_id,
        label=label,
        model_families=[f for f in model_families if any(s.startswith(f) for s in sources)],
        n_assets=n_assets,
        n_timestamps=n_timestamps,
        date_range=date_range,
        sources=sources,
        registry_entries=registry_entries,
    )


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------

# Per-case-study price normalization config
_PRICE_CONFIG = {
    "etfs": {
        "entity_col": "symbol",
        "time_col": "timestamp",
        "close_col": "close",
        "ohlcv": True,
        "loader": "etfs",
        "drop_cols": [],
    },
    "crypto_perps_funding": {
        "entity_col": "symbol",
        "time_col": "timestamp",
        "close_col": "close",
        "ohlcv": True,
        "loader": "crypto_perps",
        "drop_cols": [
            "premium_index_open",
            "premium_index_high",
            "premium_index_low",
            "premium_index_close",
        ],
    },
    "nasdaq100_microstructure": {
        "entity_col": "symbol",
        "time_col": "timestamp",
        "close_col": "close",
        "ohlcv": True,
        "loader": "nasdaq100_bars",
        "drop_cols": [],
    },
    "sp500_equity_option_analytics": {
        "entity_col": "symbol",
        "time_col": "timestamp",
        "close_col": "close",
        "ohlcv": True,
        "loader": "sp500_daily_bars",
        "drop_cols": [
            "sec_id",
            "adj_factor",
            "vol_adj_factor",
            "adjustment_factor",
            "adjustment_reason",
        ],
    },
    "us_firm_characteristics": {
        "entity_col": "symbol",
        "time_col": "timestamp",
        "close_col": None,  # No OHLCV — uses ret column for decile portfolio
        "ohlcv": False,
        # No loader — uses materialized prices.parquet (returns-only, no OHLCV source)
        "drop_cols": [],
    },
    "fx_pairs": {
        "entity_col": "symbol",
        "time_col": "timestamp",
        "close_col": "close",
        "ohlcv": True,
        "loader": "fx_pairs",
        "drop_cols": [],
    },
    "cme_futures": {
        "entity_col": "product",
        "time_col": "session_date",
        "close_col": "adj_close",
        "ohlcv": True,
        "loader": "cme_futures",
        # Backtest fills ride the roll-adjusted series (continuous PnL); rename
        # adj_* → bare OHLC and drop the raw term-structure series + roll multiplier.
        "rename_cols": {
            "adj_open": "open",
            "adj_high": "high",
            "adj_low": "low",
            "adj_close": "close",
        },
        "drop_cols": [
            "raw_open",
            "raw_high",
            "raw_low",
            "raw_close",
            "cum_ratio",
            "bar_count",
            "session_start",
            "session_end",
        ],
        "filter": {"tenor": 0},  # Front-month only
    },
    "sp500_options": {
        "entity_col": "symbol",
        "time_col": "timestamp",
        "close_col": "underlying_price",  # No standard OHLCV
        "ohlcv": False,
        "loader": "sp500_options_straddles",
        "drop_cols": ["qc_any_estimated_iv"],
    },
    "us_equities_panel": {
        "entity_col": "symbol",
        "time_col": "timestamp",
        "close_col": "adj_close",
        "ohlcv": True,
        "loader": "us_equities",
        "rename_cols": {
            "adj_open": "open",
            "adj_high": "high",
            "adj_low": "low",
            "adj_close": "close",
            "adj_volume": "volume",
        },
        # Drop raw OHLCV before adj_ rename to avoid duplicate columns
        "drop_cols": [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ex-dividend",
            "split_ratio",
            "returns",
            "adv_21d",
        ],
    },
}


def _load_via_canonical(
    loader_name: str,
    max_symbols: int = 0,
    frequency: str = "",
    include_quotes: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Dispatch to canonical data loaders instead of reading prices.parquet."""
    if loader_name == "etfs":
        from data import load_etfs

        return load_etfs(max_symbols=max_symbols, start_date=start_date, end_date=end_date)
    if loader_name == "crypto_perps":
        from data import load_crypto_perps

        return load_crypto_perps(
            frequency="8h",
            max_symbols=max_symbols,
            start_date=start_date,
            end_date=end_date,
        )
    if loader_name == "fx_pairs":
        from data import load_fx_pairs

        return load_fx_pairs(
            frequency="daily",
            max_symbols=max_symbols,
            start_date=start_date,
            end_date=end_date,
        )
    if loader_name == "cme_futures":
        from data import load_cme_futures

        return load_cme_futures(
            max_symbols=max_symbols,
            start_date=start_date,
            end_date=end_date,
        )
    if loader_name == "sp500_daily_bars":
        from data.equities.loader import load_sp500_daily_bars

        return load_sp500_daily_bars(
            max_symbols=max_symbols,
            start_date=start_date,
            end_date=end_date,
        )
    if loader_name == "sp500_options_straddles":
        from data import load_sp500_options_straddles

        return load_sp500_options_straddles(
            max_symbols=max_symbols,
            start_date=start_date,
            end_date=end_date,
        )
    if loader_name == "us_equities":
        from data import load_us_equities

        return load_us_equities(
            max_symbols=max_symbols,
            start_date=start_date,
            end_date=end_date,
        )
    if loader_name == "nasdaq100_bars":
        from data.equities.loader import load_nasdaq100_bars

        freq = frequency or "15m"
        return load_nasdaq100_bars(
            frequency=freq,
            max_symbols=max_symbols,
            include_quotes=include_quotes,
            start_date=start_date,
            end_date=end_date,
        )
    msg = f"Unknown loader: {loader_name}"
    raise ValueError(msg)


def load_backtest_prices(
    case_study_id: str,
    max_symbols: int = 0,
    frequency: str = "",
    include_quotes: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load and normalize price data for DataFeed consumption.

    Returns a DataFrame with columns [timestamp, symbol, open, high, low, close, volume]
    for standard case studies, or case-study-specific columns for special cases
    (us_firm: ret, sp500_options: instrument-level).

    Args:
        case_study_id: Case study identifier
        max_symbols: Limit universe size (0 = all)
        frequency: Bar frequency override for loader-backed case studies (e.g. "1m",
            "15m", "1h"). Empty string uses the default for the case study.
        include_quotes: If True, include bid/ask OHLCV columns (loader-backed only).
            Use for risk-layer stop monitoring with bid/ask-aware execution.
        start_date: Optional lower bound (``YYYY-MM-DD``) pushed into the parquet
            read for row-group pruning. Callers SHOULD pass the canonical
            ``(cs, label, split)`` window so memory scales with the window
            rather than the full history — see ``load_backtest_prices_for``
            for the convenience that resolves the window from
            ``canonical_window``.
        end_date: Optional upper bound (``YYYY-MM-DD``) pushed into the parquet
            read.

    Returns:
        Normalized price DataFrame ready for DataFeed
    """
    config = dict(_PRICE_CONFIG[case_study_id])
    runtime = resolve_market_runtime(case_study_id)
    if runtime:
        for key in ("entity_col", "time_col", "close_col", "ohlcv", "loader", "prices_file"):
            if runtime.get(key) is not None:
                config[key] = runtime[key]
        if "drop_cols" in runtime:
            config["drop_cols"] = list(runtime["drop_cols"])
        if "rename_cols" in runtime:
            config["rename_cols"] = dict(runtime["rename_cols"])
        if "filter" in runtime:
            config["filter"] = dict(runtime["filter"])

    effective_frequency = frequency or str(runtime.get("frequency", ""))
    effective_include_quotes = (
        include_quotes
        or bool(runtime.get("include_quotes", False))
        or _preset_requests_quotes(case_study_id)
    )
    loader_name = config.get("loader")

    # Canonical loader dispatch — avoids materialized prices.parquet for most CS
    if loader_name:
        df = _load_via_canonical(
            loader_name,
            max_symbols,
            effective_frequency,
            effective_include_quotes,
            start_date=start_date,
            end_date=end_date,
        )
        # Apply post-load filters (e.g., CME front-month: tenor=0)
        if "filter" in config:
            for col, val in config["filter"].items():
                df = df.filter(pl.col(col) == val)
    else:
        # File-backed fallback: US Firms (returns-only) and SP500 Options (straddles).
        # Apply date filters lazily so parquet row-group pruning kicks in.
        case_dir = get_case_study_dir(case_study_id)
        prices_file = config.get("prices_file", "prices.parquet")
        prices_path = case_dir / "labels" / prices_file
        lf = pl.scan_parquet(prices_path)
        if "filter" in config:
            for col, val in config["filter"].items():
                lf = lf.filter(pl.col(col) == val)
        # Date pushdown — resolve dtype-aware comparison
        if start_date or end_date:
            ts_col = config.get("time_col", "timestamp")
            if ts_col not in lf.collect_schema().names():
                ts_col = "timestamp" if "timestamp" in lf.collect_schema().names() else "date"
            ts_type = lf.collect_schema()[ts_col]
            tz = getattr(ts_type, "time_zone", None)
            is_date = ts_type == pl.Date
            if start_date:
                lit = (
                    pl.lit(start_date).str.to_date()
                    if is_date
                    else pl.lit(start_date).str.to_datetime()
                )
                if tz and not is_date:
                    lit = lit.dt.replace_time_zone(tz)
                lf = lf.filter(pl.col(ts_col) >= lit)
            if end_date:
                if is_date:
                    lf = lf.filter(pl.col(ts_col) <= pl.lit(end_date).str.to_date())
                else:
                    end_lit = pl.lit(end_date).str.to_datetime()
                    if tz:
                        end_lit = end_lit.dt.replace_time_zone(tz)
                    lf = lf.filter(pl.col(ts_col) < end_lit + pl.duration(days=1))
        df = lf.collect()

    # Drop unwanted columns
    drop = [c for c in config.get("drop_cols", []) if c in df.columns]
    if drop:
        df = df.drop(drop)

    # Apply renames (us_equities adj_ columns)
    if "rename_cols" in config:
        renames = {k: v for k, v in config["rename_cols"].items() if k in df.columns}
        if renames:
            df = df.rename(renames)

    # Rename close column if non-standard
    close_col = config.get("close_col")
    if close_col and close_col != "close" and close_col in df.columns:
        df = df.rename({close_col: "close"})

    # Rename entity → symbol
    entity_col = config["entity_col"]
    if entity_col not in df.columns:
        detected_entity = _detect_entity_col(df)
        if detected_entity is None:
            msg = f"No entity column found for {case_study_id}. Columns: {df.columns}"
            raise KeyError(msg)
        entity_col = detected_entity
    if entity_col != "symbol" and entity_col in df.columns:
        df = df.rename({entity_col: "symbol"})

    # Ensure symbol is String
    if "symbol" in df.columns and df.schema["symbol"] != pl.String:
        df = df.with_columns(pl.col("symbol").cast(pl.String))

    # Rename time → timestamp and cast to Datetime
    time_col = config["time_col"]
    if time_col not in df.columns:
        detected_time = _detect_time_col(df)
        if detected_time is None:
            msg = f"No time column found for {case_study_id}. Columns: {df.columns}"
            raise KeyError(msg)
        time_col = detected_time
    if time_col != "timestamp" and time_col in df.columns:
        df = df.rename({time_col: "timestamp"})

    if df.schema["timestamp"] == pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
    elif hasattr(df.schema["timestamp"], "time_zone") and df.schema["timestamp"].time_zone:
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    # Drop filter columns that are no longer needed (e.g., position for CME)
    if "filter" in config:
        for col in config["filter"]:
            if col in df.columns:
                df = df.drop(col)

    # Universe reduction
    if max_symbols > 0 and "symbol" in df.columns:
        top_symbols = (
            df.group_by("symbol")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .head(max_symbols)["symbol"]
        )
        df = df.filter(pl.col("symbol").is_in(top_symbols))

    return df.sort("timestamp", "symbol")


@cache
def _load_case_setup_yaml(case_study_id: str) -> dict:
    """Cached read of the case study's setup.yaml. Returns {} when missing."""
    setup_path = get_case_study_dir(case_study_id) / "config" / "setup.yaml"
    if not setup_path.exists():
        return {}
    with setup_path.open() as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def warmup_periods_for(case_study_id: str) -> int:
    """Resolve the per-CS warmup period count from setup.yaml.

    Returns ``max(execution.allocator_lookback, max sweep allocator
    {vol_window, lookback})``. Replaces the hardcoded ``warmup_periods=126``
    constant that previously coupled call sites to the daily-cadence
    default by hand; non-daily CSes need a different bar count (crypto
    240, nasdaq100 520, us_firm 12).

    Returns 0 when no allocator window is declared (the unbounded
    fallback in ``load_backtest_prices_for`` will then skip the
    prefix-day calculation entirely).
    """
    setup = _load_case_setup_yaml(case_study_id)
    execution = setup.get("execution") or {}
    candidates: list[int] = []
    base = execution.get("allocator_lookback")
    if base is not None:
        candidates.append(int(base))
    backtest = setup.get("backtest") or {}
    sweep = backtest.get("sweep") or {}
    allocators = sweep.get("allocators") or []
    for alloc in allocators:
        if not isinstance(alloc, dict):
            continue
        for key in ("vol_window", "lookback"):
            value = alloc.get(key)
            if value is not None:
                candidates.append(int(value))
    return max(candidates) if candidates else 0


# Calendar-day spacing per allocator-window bar, indexed by setup.yaml
# cadence / bar_frequency tokens. Daily cadences allow 1.5× to absorb
# weekends + market holidays; intraday cadences are pure trading-time
# (no weekend allowance needed — the price loader's start_date filter
# only sees timestamps that exist). Monthly month-end approximates a
# calendar-month spacing.
_CADENCE_CALENDAR_DAYS_PER_PERIOD: dict[str, float] = {
    # Daily cadences
    "daily_close": 1.5,
    "daily_ny_close": 1.5,
    # Weekly cadences
    "weekly_friday_close": 7.0,
    "weekly_friday": 7.0,
    # 8-hour funding
    "8_hour_funding_aligned": 1.0 / 3.0,
    # Intraday equity microstructure: ~26 fifteen-minute bars per RTH
    # trading day; multiply by 1.4 to account for weekends.
    "15_minute": (1.0 / 26.0) * 1.4,
    # Monthly month-end
    "monthly_month_end": 31.0,
}


def _calendar_days_per_period(case_study_id: str) -> float:
    """Calendar-day spacing per allocator-window bar for this case study.

    Reads ``decision.entry_cadence`` (or ``decision.cadence`` or
    ``decision.bar_frequency``) and returns the calendar-day multiplier
    used to walk the start_date back during a warmup-prefix load. Falls
    back to the daily 1.5× heuristic when the cadence token isn't
    recognized — old behavior for unknown CSes.
    """
    setup = _load_case_setup_yaml(case_study_id)
    decision = setup.get("decision") or {}
    cadence = (
        decision.get("entry_cadence") or decision.get("cadence") or decision.get("bar_frequency")
    )
    if cadence and cadence in _CADENCE_CALENDAR_DAYS_PER_PERIOD:
        return _CADENCE_CALENDAR_DAYS_PER_PERIOD[cadence]
    return 1.5


def load_backtest_prices_for(
    case_study_id: str,
    label: str,
    split: Literal["validation", "holdout"] = "validation",
    warmup_periods: int = 0,
    **kwargs,
) -> pl.DataFrame:
    """Load prices pre-windowed to ``canonical_window(case_study_id, label, split)``.

    When ``warmup_periods > 0``, the start of the load window is left
    unconstrained so rolling-vol allocators (``inverse_vol`` /
    ``risk_parity`` / ``hrp`` / ``mvo_ledoit_wolf``) see pre-window history
    and produce data-driven weights at the first rebalance instead of
    falling back to the median-imputed warmup. The end of the window is
    always capped to the canonical window end; the engine's port_ret only
    covers rebalance timestamps from the predictions, so the extra prefix
    history is consumed by the allocator's rolling window but does not
    enter return aggregation.

    Args:
        case_study_id: Case study identifier.
        label: Target label (e.g. ``"fwd_ret_21d"``).
        split: ``"validation"`` (default) for the union-of-folds window, or
            ``"holdout"`` for ``setup.yaml::evaluation.{holdout_start,
            holdout_end}``.
        warmup_periods: Number of pre-window periods the caller's allocator
            needs (typically ``strategy.allocation.vol_window`` or
            ``lookback``). When > 0, the start of the load window is
            dropped so the parquet read returns the full prefix up to the
            canonical window end. When 0 (default), only the canonical
            window is loaded.
        **kwargs: Forwarded to :func:`load_backtest_prices` (``max_symbols``,
            ``frequency``, ``include_quotes``). Explicit ``start_date`` or
            ``end_date`` in ``kwargs`` take precedence over the canonical
            window.
    """
    import math
    from datetime import timedelta

    from case_studies.utils.cv_window import canonical_window

    win = canonical_window(case_study_id, label, split=split)
    if win is not None:
        kwargs.setdefault("end_date", win[1].isoformat())
        if warmup_periods <= 0:
            kwargs.setdefault("start_date", win[0].isoformat())
        else:
            # Bounded warmup: walk start_date back by ~warmup_periods
            # allocator-window bars, expressed as calendar days using the
            # per-CS cadence multiplier from
            # ``_CADENCE_CALENDAR_DAYS_PER_PERIOD``. Daily cadences use 1.5×
            # (weekend + holiday allowance), weekly 7×, monthly 31×, 8h
            # ~0.33×, 15-min ~0.054×. Previously this was a flat 1.5×
            # which over-allocated by ~50× on the intraday
            # nasdaq100_microstructure CS (520 periods × 15-min bars).
            # ``math.ceil`` is load-bearing: float-arithmetic truncation
            # (e.g. ``520 * (1.0/26.0) * 1.4`` can land at 27.999...)
            # would silently under-provision the warmup window by one
            # bar on intraday CSes. The floor of 7 days ensures the
            # parquet read covers at least a full calendar week even
            # when ``warmup_periods`` is tiny (e.g. monthly us_firm
            # with 12 periods).
            cal_per_period = _calendar_days_per_period(case_study_id)
            prefix_days = max(math.ceil(warmup_periods * cal_per_period), 7)
            kwargs.setdefault("start_date", (win[0] - timedelta(days=prefix_days)).isoformat())
    return load_backtest_prices(case_study_id, **kwargs)


# ---------------------------------------------------------------------------
# Calendar-aware schedule resolution
# ---------------------------------------------------------------------------


def resolve_rebalance_timestamps(
    available_timestamps: pl.Series,
    cadence: str,
    calendar: str = "NYSE",
) -> pl.Series:
    """Resolve exact rebalance timestamps from cadence + calendar + available data.

    Instead of counting elapsed seconds or stepping by fixed intervals, this
    function selects the actual timestamps that match the declared schedule:

    - ``monthly_month_end`` → last available timestamp in each calendar month
    - ``weekly_friday_close`` / ``weekly_friday`` → last available timestamp
      in each ISO week (typically Friday, or Thursday if Friday is a holiday)
    - ``daily_*`` → every available timestamp
    - ``8_hour_*`` / ``15_min`` → every available timestamp (fixed-interval
      cadences where the data is already at the correct granularity)

    Parameters
    ----------
    available_timestamps : pl.Series
        Sorted unique timestamps from predictions or prices.
    cadence : str
        Rebalance cadence from setup.yaml.
    calendar : str
        Trading calendar name (used for future session filtering).

    Returns
    -------
    pl.Series
        Subset of available_timestamps matching the declared schedule.
    """
    if available_timestamps.is_empty():
        return available_timestamps

    ts = available_timestamps.unique().sort()

    if cadence == "monthly_month_end":
        # Last available session in each calendar month
        df = pl.DataFrame({"ts": ts}).with_columns(
            year=pl.col("ts").dt.year(),
            month=pl.col("ts").dt.month(),
        )
        month_ends = (
            df.group_by("year", "month").agg(pl.col("ts").max().alias("rebal_ts")).sort("rebal_ts")
        )
        return month_ends["rebal_ts"]

    if cadence in {"weekly", "weekly_friday", "weekly_friday_close"}:
        # Last available session in each ISO week
        df = pl.DataFrame({"ts": ts}).with_columns(
            iso_year=pl.col("ts").dt.iso_year(),
            iso_week=pl.col("ts").dt.week(),
        )
        week_ends = (
            df.group_by("iso_year", "iso_week")
            .agg(pl.col("ts").max().alias("rebal_ts"))
            .sort("rebal_ts")
        )
        return week_ends["rebal_ts"]

    # All other cadences: daily, 8_hour, 15_min, etc.
    # The data is already at the correct granularity — every timestamp is valid.
    return ts


# ---------------------------------------------------------------------------
# Rebalance thinning
# ---------------------------------------------------------------------------


@cache
def get_rebalance_step(case_study: str, label: str) -> int:
    """Return the per-label vectorized-backtest thinning step, from setup.yaml.

    The step is the number of schedule slots to advance per trade so that
    holding periods don't overlap. It is a design-time property of the
    (case study, label) pair — authored in each case study's
    ``config/setup.yaml`` under ``labels.rebalance_step``. No inference
    happens at runtime.

    Parameters
    ----------
    case_study : str
        Case study identifier (e.g., ``"sp500_options"``).
    label : str
        Label name (e.g., ``"ret_to_expiry"``).

    Returns
    -------
    int
        Rebalance step (>= 1).

    Raises
    ------
    KeyError
        If ``labels.rebalance_step[label]`` is missing from setup.yaml.
        New labels must be registered explicitly.
    """
    # Always read the source-of-truth setup.yaml under CASE_STUDIES_DIR, not the
    # ML4T_OUTPUT_DIR-redirected get_case_study_dir() path: the rebalance-step
    # declaration is configuration, not output, and tests must see the real value.
    from utils import CASE_STUDIES_DIR

    setup_path = CASE_STUDIES_DIR / case_study / "config" / "setup.yaml"
    setup = yaml.safe_load(setup_path.read_text())
    steps = (setup.get("labels") or {}).get("rebalance_step") or {}
    if label not in steps:
        raise KeyError(
            f"labels.rebalance_step[{label!r}] not declared in "
            f"case_studies/{case_study}/config/setup.yaml. Add it explicitly — "
            f"the step is (schedule cadence, label horizon)-dependent and must "
            f"not be inferred at runtime."
        )
    step = int(steps[label])
    if step < 1:
        raise ValueError(
            f"labels.rebalance_step[{label!r}] = {step!r} in "
            f"case_studies/{case_study}/config/setup.yaml — must be >= 1."
        )
    return step


def thin_to_rebalance_dates(
    predictions: pl.DataFrame,
    cadence: str = "",
    step: int = 1,
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """Thin predictions to non-overlapping rebalance dates.

    Two-step thinning for vectorized backtests:

    1. **Calendar alignment** — filter prediction timestamps to those that
       match the declared rebalance cadence (e.g., only Fridays for
       ``weekly_friday``, only month-ends for ``monthly_month_end``).
    2. **Non-overlapping thinning** — keep every ``step``-th calendar-aligned
       date so forward-return windows don't overlap. The caller supplies
       ``step`` via :func:`get_rebalance_step`, which looks it up from
       ``labels.rebalance_step`` in the case study's setup.yaml.

    Parameters
    ----------
    predictions : pl.DataFrame
        Must contain ``time_col`` (default ``"timestamp"``).
    cadence : str
        Rebalance cadence from setup.yaml (e.g., ``"monthly_month_end"``).
    step : int
        Number of schedule slots to advance per trade (1 = keep every
        calendar-aligned date). Must be >= 1.
    time_col : str
        Timestamp column name.

    Returns
    -------
    pl.DataFrame
        Filtered DataFrame with rows at non-overlapping rebalance times.
    """
    all_dates = predictions[time_col].unique().sort()
    n_dates = len(all_dates)
    if n_dates <= 1:
        return predictions

    # Step 1: Calendar-aware schedule resolution
    schedule_dates = resolve_rebalance_timestamps(all_dates, cadence)

    # Step 2: Apply design-time non-overlapping step
    if step > 1:
        schedule_dates = schedule_dates.gather_every(step)

    # Semi-join to filter — avoids Polars is_in precision mismatch
    # (group_by().agg(max) can change Datetime precision)
    schedule_df = pl.DataFrame({time_col: schedule_dates})
    if schedule_df[time_col].dtype != predictions[time_col].dtype:
        schedule_df = schedule_df.cast({time_col: predictions[time_col].dtype})
    return predictions.join(schedule_df, on=time_col, how="semi")


# ---------------------------------------------------------------------------
# Shared allocator metrics
# ---------------------------------------------------------------------------


def _periods_per_year_from_ann_factor(ann_factor: float) -> int:
    """Convert annualization factor (sqrt(N)) back to periods per year."""
    return max(1, round(ann_factor**2))


def compute_allocator_metrics(
    port_returns: pl.Series | list[float],
    weights_df: pl.DataFrame | None = None,
    ann_factor: float = np.sqrt(252),
    time_col: str = "timestamp",
    cost_rate: float = 0.0,
) -> dict:
    """Compute allocator summary metrics using ml4t-diagnostic PortfolioAnalysis.

    Args:
        port_returns: Series or list of per-period gross returns. If cost_rate > 0,
            turnover-adjusted net returns are computed internally.
        weights_df: Optional DataFrame with [timestamp, symbol, weight] for
            computing concentration and turnover metrics.
        ann_factor: Annualization factor (sqrt of periods per year).
        time_col: Time column name in weights_df.
        cost_rate: Per-period cost rate applied to turnover (e.g., 0.001 for 10 bps).
            When > 0, net returns = gross - turnover * cost_rate.

    Returns:
        Dict with return-based metrics from PortfolioAnalysis plus weight-based
        metrics (turnover, HHI, effective bets, max weight).
    """
    import numpy as _np
    from ml4t.diagnostic.evaluation import PortfolioAnalysis

    if isinstance(port_returns, pl.Series):
        rets = port_returns.to_numpy()
    else:
        rets = _np.array(port_returns)

    rets = rets[~_np.isnan(rets)]
    _empty_keys = [
        "sharpe",
        "sortino",
        "calmar",
        "omega",
        "total_return",
        "annual_return",
        "max_drawdown",
        "max_dd_duration",
        "var_95",
        "cvar_95",
        "win_rate",
        "profit_factor",
        "stability",
        "avg_turnover",
        "avg_hhi",
        "eff_bets",
        "avg_max_weight",
    ]
    if len(rets) == 0:
        return {k: 0.0 for k in _empty_keys}

    # --- Weight-based metrics (computed first for cost deduction) ---
    avg_turnover = 0.0
    avg_hhi = 0.0
    eff_bets = 0.0
    avg_max_weight = 0.0
    turnover_per_period = None

    if weights_df is not None and len(weights_df) > 0:
        hhi_ts = (
            weights_df.with_columns(w2=pl.col("weight") ** 2)
            .group_by(time_col)
            .agg(hhi=pl.col("w2").sum(), max_w=pl.col("weight").abs().max())
        )
        avg_hhi = float(hhi_ts["hhi"].mean())
        avg_max_weight = float(hhi_ts["max_w"].mean())
        eff_bets = 1.0 / avg_hhi if avg_hhi > 0 else 0.0

        w_lag = weights_df.sort(time_col, "symbol").with_columns(
            prev_w=pl.col("weight").shift(1).over("symbol").fill_null(0.0)
        )
        to_ts = (
            w_lag.with_columns(delta=(pl.col("weight") - pl.col("prev_w")).abs())
            .group_by(time_col)
            .agg(turnover=pl.col("delta").sum())
        )
        avg_turnover = float(to_ts["turnover"].mean())
        if cost_rate > 0:
            turnover_per_period = to_ts["turnover"].to_numpy()

    # Deduct transaction costs if cost_rate provided
    if cost_rate > 0 and turnover_per_period is not None:
        n = min(len(rets), len(turnover_per_period))
        rets = rets[:n] - turnover_per_period[:n] * cost_rate
    elif cost_rate > 0:
        rets = rets - avg_turnover * cost_rate

    # --- Return-based metrics via PortfolioAnalysis ---
    periods_per_year = _periods_per_year_from_ann_factor(ann_factor)
    pa = PortfolioAnalysis(pl.Series("returns", rets), periods_per_year=periods_per_year)
    stats = pa.compute_summary_stats()
    dd = pa.compute_drawdown_analysis()

    def _safe_round(value: object, digits: int = 4) -> float:
        if isinstance(value, complex):
            value = value.real
        return round(float(value), digits)

    return {
        "sharpe": _safe_round(stats.sharpe_ratio, 4),
        "sortino": _safe_round(stats.sortino_ratio, 4),
        "calmar": _safe_round(stats.calmar_ratio, 4),
        "omega": _safe_round(stats.omega_ratio, 4),
        "total_return": _safe_round(stats.total_return, 6),
        "annual_return": _safe_round(stats.annual_return, 6),
        "max_drawdown": _safe_round(stats.max_drawdown, 6),
        "max_dd_duration": int(dd.max_duration_days),
        "var_95": _safe_round(stats.var_95, 6),
        "cvar_95": _safe_round(stats.cvar_95, 6),
        "win_rate": _safe_round(stats.win_rate, 4),
        "profit_factor": _safe_round(stats.profit_factor, 4),
        "stability": _safe_round(stats.stability, 4),
        "avg_turnover": round(avg_turnover, 6),
        "avg_hhi": round(avg_hhi, 6),
        "eff_bets": round(eff_bets, 2),
        "avg_max_weight": round(avg_max_weight, 6),
    }


def compute_dsr_table(
    returns_by_source: dict[str, pl.Series | np.ndarray],
    periods_per_year: int = 252,
) -> pl.DataFrame:
    """Rank model variants by Sharpe with raw-K selection-bias adjustment for the best.

    Ad-hoc utility for one-off DSR analysis over a custom returns dict. Uses
    **raw-K** trial counting (no Marchenko-Pastur or effective-rank
    correction), which overcounts trials when variants are correlated.

    For headline / persisted DSR numbers, prefer the cohort_metrics table:

        BacktestExplorer(cs).deflated_sharpe(stage="signal")

    which surfaces the effective-rank (ER) DSR — the library maintainer's
    recommended default — alongside MP and raw-K for sensitivity.

    Each variant gets its own Sharpe ratio and individual PSR (probability of
    skill without multiple-testing correction). The best variant additionally
    gets DSR columns showing how selection bias across K tested strategies
    deflates the observed Sharpe.

    Args:
        returns_by_source: Dict mapping model name to return series.
        periods_per_year: Annualization periods.

    Returns:
        DataFrame sorted by Sharpe (descending) with columns: source, sharpe,
        psr_pvalue, deflated_sharpe, expected_max_sharpe, dsr_pvalue,
        significant, is_best.
    """
    from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

    freq_map = {252: "daily", 52: "weekly", 12: "monthly", 1: "annual"}
    frequency = freq_map.get(periods_per_year, "daily")

    all_returns = []
    names = []
    for name, ret in returns_by_source.items():
        arr = ret.to_numpy() if isinstance(ret, pl.Series) else np.asarray(ret)
        all_returns.append(arr)
        names.append(name)

    if not all_returns:
        return pl.DataFrame(
            schema={
                "source": pl.Utf8,
                "sharpe": pl.Float64,
                "psr_pvalue": pl.Float64,
                "deflated_sharpe": pl.Float64,
                "expected_max_sharpe": pl.Float64,
                "dsr_pvalue": pl.Float64,
                "significant": pl.Boolean,
                "is_best": pl.Boolean,
            }
        )

    # Per-variant PSR (individual probability of skill, no multiple-testing correction)
    per_variant_psr = {}
    sharpes = {}
    for i, name in enumerate(names):
        arr = all_returns[i]
        sr = float(np.mean(arr) / max(np.std(arr, ddof=1), 1e-8) * np.sqrt(periods_per_year))
        sharpes[name] = sr
        try:
            psr = deflated_sharpe_ratio(
                [arr], frequency=frequency, periods_per_year=periods_per_year
            )
            per_variant_psr[name] = psr
        except Exception as exc:
            warnings.warn(f"DSR computation failed for {name}: {exc}", stacklevel=2)
            per_variant_psr[name] = None

    # Aggregate DSR across all variants (selection-bias adjustment for best-of-K)
    # Filter out zero-variance return series (e.g. constant/all-zero returns from test data)
    valid_returns = [r for r in all_returns if np.std(r, ddof=1) > 1e-10]
    if not valid_returns:
        return pl.DataFrame(
            schema={
                "source": pl.Utf8,
                "sharpe": pl.Float64,
                "psr_pvalue": pl.Float64,
                "deflated_sharpe": pl.Float64,
                "expected_max_sharpe": pl.Float64,
                "dsr_pvalue": pl.Float64,
                "significant": pl.Boolean,
                "is_best": pl.Boolean,
            }
        )
    dsr = deflated_sharpe_ratio(
        valid_returns, frequency=frequency, periods_per_year=periods_per_year
    )

    # Identify best variant by Sharpe
    best_name = max(sharpes, key=sharpes.get)

    rows = []
    for name in names:
        psr = per_variant_psr.get(name)
        is_best = name == best_name
        rows.append(
            {
                "source": name,
                "sharpe": round(sharpes[name], 4),
                "psr_pvalue": round(psr.p_value, 4) if psr else None,
                "deflated_sharpe": round(dsr.deflated_sharpe, 4) if is_best else None,
                "expected_max_sharpe": round(dsr.expected_max_sharpe, 4) if is_best else None,
                "dsr_pvalue": round(dsr.p_value, 4) if is_best else None,
                "significant": bool(dsr.is_significant) if is_best else None,
                "is_best": is_best,
            }
        )

    # Sort by Sharpe descending so best is row 0
    df = pl.DataFrame(rows).sort("sharpe", descending=True)
    return df


def print_stage_dsr_summary(
    explorer,
    *,
    stages: tuple[str, ...] = ("signal", "allocation", "cost_sensitivity", "risk_overlay"),
    top_n: int = 20,
    head: int = 10,
) -> None:
    """Print per-stage DSR / PSR tables for a case-study explorer.

    The selection-bias question — "after K variants were tried, does the leader
    have skill?" — is well-defined at every pipeline stage, not just the
    signal stage. This helper iterates the four stages, prints the leader
    table for each one (with PSR per variant + DSR for the leader), and
    silently skips stages that have no data.
    """
    for stage in stages:
        try:
            table = explorer.deflated_sharpe(stage=stage, top_n=top_n)
        except ValueError as exc:
            if "zero variance" in str(exc).lower():
                print(f"\n--- DSR @ {stage}: skipped ({exc}) ---")
                continue
            raise
        except Exception as exc:  # pragma: no cover
            print(f"\n--- DSR @ {stage}: error ({exc}) ---")
            continue
        if table is None or table.is_empty():
            continue
        print(f"\n--- DSR @ {stage} (K={table.height}) ---")
        print(table.head(head))


def infer_session_alignment(calendar: str | None) -> bool:
    """Infer whether returns should be aligned to trading sessions."""
    return bool(calendar and "CME" in str(calendar).upper())


def _extract_session_aligned_returns(result: BacktestResult) -> pl.DataFrame:
    """Rebuild session-aligned returns when ml4t-backtest's helper hits dtype issues."""
    from zoneinfo import ZoneInfo

    from ml4t.backtest.sessions import SessionConfig, assign_session_date

    equity_df = (
        result.to_equity_dataframe()
        .select("timestamp", "equity")
        .with_columns(pl.col("equity").cast(pl.Float64))
    )
    if equity_df.is_empty():
        return pl.DataFrame({"timestamp": pl.Series([], dtype=pl.Date), "daily_return": []})

    session_config = SessionConfig(
        calendar=result.config.calendar,
        timezone=result.config.timezone,
        session_start_time=getattr(result.config, "session_start_time", None),
    )
    tz = ZoneInfo(session_config.timezone)
    session_start_hour = session_config.get_session_start_hour()
    session_start_minute = session_config.get_session_start_minute()
    timestamps = equity_df["timestamp"].to_list()
    session_dates = [
        assign_session_date(ts, tz, session_start_hour, session_start_minute) for ts in timestamps
    ]

    daily = (
        pl.DataFrame(
            {
                "timestamp": timestamps,
                "equity": equity_df["equity"].to_list(),
                "session_date": session_dates,
            },
            strict=False,
        )
        .group_by("session_date")
        .agg(
            pl.col("equity").first().alias("open_equity"),
            pl.col("equity").last().alias("close_equity"),
        )
        .sort("session_date")
    )

    prev_close = daily.select(pl.col("close_equity").shift(1)).to_series()
    return (
        daily.with_columns(
            ((pl.col("close_equity") - prev_close) / prev_close)
            .fill_null(0.0)
            .alias("daily_return")
        )
        .select(pl.col("session_date").cast(pl.Date).alias("timestamp"), "daily_return")
        .sort("timestamp")
        .unique("timestamp", keep="last")
    )


def extract_daily_returns_frame(
    result: BacktestResult,
    calendar: str | None = None,
    session_aligned: bool | None = None,
) -> pl.DataFrame:
    """Extract daily returns with dates from BacktestResult.

    Prefers `to_daily_pnl()` so output includes date/session_date labels.
    """
    if session_aligned is None:
        cal = calendar
        if cal is None and hasattr(result, "config") and result.config is not None:
            cal = getattr(result.config, "calendar", None)
        session_aligned = infer_session_alignment(cal)

    if hasattr(result, "to_daily_pnl"):
        try:
            daily = result.to_daily_pnl(session_aligned=session_aligned)
        except TypeError:
            if session_aligned and getattr(result, "config", None) is not None:
                return _extract_session_aligned_returns(result)
            daily = None
        if daily is not None:
            if daily.is_empty():
                return pl.DataFrame({"timestamp": pl.Series([], dtype=pl.Date), "daily_return": []})
            date_col = "session_date" if "session_date" in daily.columns else "date"
            if date_col not in daily.columns:
                msg = f"to_daily_pnl() missing date column. Columns: {daily.columns}"
                raise ValueError(msg)
            return (
                daily.select(
                    pl.col(date_col).cast(pl.Date).alias("timestamp"),
                    pl.col("return_pct").cast(pl.Float64).alias("daily_return"),
                )
                .sort("timestamp")
                .unique("timestamp", keep="last")
            )

    if hasattr(result, "to_daily_returns"):
        daily_returns = result.to_daily_returns(
            calendar=calendar,
            session_aligned=session_aligned,
        )
        if not isinstance(daily_returns, pl.Series):
            daily_returns = pl.Series("daily_return", np.asarray(daily_returns, dtype=float))
        return pl.DataFrame(
            {"date_idx": np.arange(len(daily_returns)), "daily_return": daily_returns}
        )

    if hasattr(result, "to_returns_series"):
        rets = result.to_returns_series()
        if not isinstance(rets, pl.Series):
            rets = pl.Series("daily_return", np.asarray(rets, dtype=float))
        return pl.DataFrame({"date_idx": np.arange(len(rets)), "daily_return": rets})

    msg = "BacktestResult has no daily or period returns export method"
    raise AttributeError(msg)


def aggregate_timestamped_returns_to_daily(
    returns_df: pl.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    return_col: str = "ret",
    calendar: str | None = None,
    session_aligned: bool | None = None,
) -> pl.DataFrame:
    """Aggregate timestamped period returns to daily returns.

    For CME-style sessions, uses session-date assignment when available.
    """
    if returns_df.is_empty():
        return pl.DataFrame({"timestamp": pl.Series([], dtype=pl.Date), "daily_return": []})

    if timestamp_col not in returns_df.columns or return_col not in returns_df.columns:
        msg = f"Expected columns '{timestamp_col}' and '{return_col}'. Got: {returns_df.columns}"
        raise ValueError(msg)

    out = returns_df.select([timestamp_col, return_col]).drop_nulls()
    if out.is_empty():
        return pl.DataFrame({"timestamp": pl.Series([], dtype=pl.Date), "daily_return": []})

    if out[timestamp_col].dtype == pl.Utf8:
        out = out.with_columns(pl.col(timestamp_col).str.to_datetime(strict=False))

    if session_aligned is None:
        session_aligned = infer_session_alignment(calendar)

    if session_aligned:
        try:
            from zoneinfo import ZoneInfo

            from ml4t.backtest.sessions import SessionConfig, assign_session_date

            cfg = SessionConfig(calendar=str(calendar or "CME_Equity"))
            tz = ZoneInfo(cfg.timezone)
            sh = cfg.get_session_start_hour()
            sm = cfg.get_session_start_minute()
            ts = out[timestamp_col].to_list()
            session_dates = [assign_session_date(t, tz, sh, sm).date() for t in ts]
            out = out.with_columns(pl.Series("timestamp", session_dates, dtype=pl.Date))
        except Exception:
            out = out.with_columns(pl.col(timestamp_col).dt.date().alias("timestamp"))
    else:
        out = out.with_columns(pl.col(timestamp_col).dt.date().alias("timestamp"))

    return (
        out.group_by("timestamp")
        .agg(daily_return=((1.0 + pl.col(return_col)).product() - 1.0))
        .sort("timestamp")
    )


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------


def get_backtest_config(case_study_id: str) -> BacktestConfig:
    """Extract backtesting configuration from setup.yaml.

    Normalizes the heterogeneous costs sections into a uniform
    (commission_bps, slippage_bps) pair.

    Args:
        case_study_id: Case study identifier

    Returns:
        BacktestConfig with normalized cost and execution parameters
    """
    case_dir = get_case_study_dir(case_study_id)
    setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
    market_semantics = resolve_market_semantics(case_study_id, setup)

    labels = setup["labels"]
    evaluation = setup["evaluation"]
    decision = setup.get("decision", {})
    mapping = setup.get("mapping", {})
    costs = setup.get("costs", {})

    # Normalize costs to (commission_bps, slippage_bps)
    commission_bps, slippage_bps = _normalize_costs(costs, case_study_id)

    # Determine long/short from mapping state-space tokens.
    # One-sided short states (e.g., short_straddle_hedged) should not trigger cross-sectional
    # long/short construction.
    position_space = str(mapping.get("position_state_space", "long_only")).strip().lower()
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", position_space) if tok]
    long_short = "long" in tokens and "short" in tokens

    # Determine cadence: entry_cadence > cadence > bar_frequency > default
    cadence = (
        decision.get("entry_cadence")
        or decision.get("cadence")
        or decision.get("bar_frequency")
        or "monthly_month_end"
    )

    backtest_block = setup.get("backtest", {}) or {}
    rebalance_block = backtest_block.get("rebalance", {}) or {}
    default_rebal = rebalance_block.get("default", {}) or {}

    # Engine-level execution defaults: single source of truth. Notebooks must
    # never declare local INITIAL_CASH / SHARE_TYPE constants. Falls back to
    # the previous notebook defaults during migration; the CS's setup.yaml
    # should declare an ``execution:`` block explicitly.
    execution = setup.get("execution", {}) or {}
    initial_cash = float(execution.get("initial_cash", 100_000.0))
    share_type = str(execution.get("share_type", "integer"))

    return BacktestConfig(
        case_study_id=case_study_id,
        primary_label=labels["primary"],
        label_buffer=labels.get("buffer", ""),
        calendar=market_semantics.get("calendar") or evaluation.get("calendar", "NYSE"),
        cadence=cadence,
        execution_delay=decision.get("execution_delay", "next_bar_open"),
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        costs_class=costs.get("class", "material"),
        long_short=long_short,
        holdout_start=evaluation.get("holdout_start", ""),
        holdout_end=evaluation.get("holdout_end", ""),
        n_splits=evaluation.get("n_splits", 1),
        raw_costs=costs,
        min_weight_change=float(default_rebal.get("min_weight_change", 0.005)),
        min_trade_value=float(default_rebal.get("min_trade_value", 100.0)),
        initial_cash=initial_cash,
        share_type=share_type,
    )


def get_benchmark_rebalance_thresholds(case_study_id: str) -> tuple[float, float]:
    """Return (min_weight_change, min_trade_value) for the benchmark profile.

    Read from setup.yaml:backtest.rebalance.benchmark. The benchmark — full-
    universe equal-weight — needs thresholds disabled (per-asset weight = 1/N
    is below the default 0.5% for any reasonable universe), so this profile
    typically returns (0.0, 0.0). Falls back to the default profile if no
    benchmark block is declared.
    """
    case_dir = get_case_study_dir(case_study_id)
    setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
    rebal = (setup.get("backtest", {}) or {}).get("rebalance", {}) or {}
    bench = rebal.get("benchmark") or rebal.get("default") or {}
    return (
        float(bench.get("min_weight_change", 0.0)),
        float(bench.get("min_trade_value", 0.0)),
    )


def _normalize_costs(costs: dict, case_study_id: str) -> tuple[float, float]:
    """Convert heterogeneous cost structures to (commission_bps, slippage_bps).

    Returns:
        (commission_bps, slippage_bps) — both in basis points
    """
    if not costs or costs.get("class") == "negligible":
        return 0.0, 0.0

    # Most case studies: per_leg_cost_bps_range → midpoint
    if "per_leg_cost_bps_range" in costs:
        lo, hi = costs["per_leg_cost_bps_range"]
        midpoint = (lo + hi) / 2
        # Split roughly 60/40 between commission and slippage
        return midpoint * 0.6, midpoint * 0.4

    # Crypto: fee_schedule with taker/maker
    if "fee_schedule" in costs:
        fee = costs["fee_schedule"]
        taker = fee.get("taker_bps", 4)
        maker = fee.get("maker_bps", 2)
        # Use taker as conservative estimate (most retail trades are taker)
        return float(taker), 1.0  # Minimal slippage for liquid crypto

    # Round trip cost
    if "round_trip_cost_bps" in costs:
        rt = costs["round_trip_cost_bps"]
        per_leg = rt / 2
        return per_leg * 0.6, per_leg * 0.4

    # Fallback: covers cme_futures (commission_per_contract + spread_ticks)
    # and other non-standard cost structures. For CME futures, exact bps
    # conversion requires contract-specific notional; 7 bps total is a
    # reasonable aggregate across the 30-product universe.
    return 5.0, 2.0
