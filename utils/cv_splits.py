"""Cross-validation split generation for case study pipelines.

Reads the ``evaluation`` section from ``setup.yaml`` and generates
walk-forward date boundaries by delegating to ml4t-diagnostic's
``WalkForwardCV``. This is the single source of truth for CV splits
used by all case studies (Ch11+).

Usage:
    from utils.cv_splits import generate_cv_splits, load_evaluation_config, make_walk_forward_config

    # Date-boundary splits
    splits = generate_cv_splits(dataset, case_study_id="etfs", label_buffer="21D")
    for split in splits:
        train_mask = (df[date_col] >= split["train_start"]) & (df[date_col] <= split["train_end"])
        val_mask   = (df[date_col] >= split["val_start"])   & (df[date_col] <= split["val_end"])

    # WalkForwardConfig for library integration
    config = make_walk_forward_config("etfs", label_horizon="21D")

Design decisions:
    - Delegates fold generation to ml4t-diagnostic's WalkForwardCV
    - Calendar-aware splitting (NYSE, CME, etc.) replaces broken ppd arithmetic
    - Operates on unique dates (handles panel data correctly)
    - Rolling training windows (respects train_size from config)
    - Backward stepping from holdout boundary
    - label_buffer is provided at call time (depends on label, not config)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import yaml

from utils.artifact_specs import resolve_market_semantics
from utils.paths import get_case_study_dir

if TYPE_CHECKING:
    from ml4t.diagnostic.splitters.config import WalkForwardConfig


# ---------------------------------------------------------------------------
# Calendar name mapping: setup.yaml → pandas_market_calendars exchange names
# ---------------------------------------------------------------------------
_CALENDAR_MAP: dict[str, str | None] = {
    "NYSE": "NYSE",
    "CME": "CME_Equity",
    "FX": "CME_FX",
    "crypto": None,  # 24/7 trading, no calendar
}


def _map_calendar_id(calendar: str | None) -> str | None:
    """Map setup.yaml calendar name to pandas_market_calendars exchange name.

    Returns None for 24/7 markets (crypto) to disable calendar-aware splitting.
    Unknown names are passed through unchanged (will error in the library if invalid).
    """
    if calendar is None:
        return None
    return _CALENDAR_MAP.get(calendar, calendar)


def _normalize_duration(s: str) -> str:
    """Strip ISO 8601 prefix (P, PT) and normalize unit aliases.

    Examples: P5Y → 5Y, P1Y → 1Y, PT8H → 8h, 21D → 21D (unchanged).
    """
    s = re.sub(r"^P?T?", "", s)
    s = re.sub(r"(\d+)H$", r"\1h", s)
    s = re.sub(r"(\d+)T$", r"\1min", s)
    return s


def _normalize_label_buffer(s: str) -> str:
    """Normalize label buffer for pd.Timedelta compatibility.

    Strips ISO prefix, normalizes units, and converts month-based
    durations to day equivalents since pd.Timedelta rejects 'M' as ambiguous.
    """
    s = _normalize_duration(s)
    m = re.match(r"^(\d+)M$", s)
    if m:
        return f"{int(m.group(1)) * 30}D"
    return s


def load_evaluation_config(case_study_id: str) -> dict[str, Any]:
    """Read the evaluation section from setup.yaml.

    Parameters
    ----------
    case_study_id : str
        Case study identifier (e.g., "etfs", "crypto_perps_funding").

    Returns
    -------
    dict
        Evaluation config with keys: n_splits, train_size, val_size,
        holdout_start, holdout_end, calendar.
    """
    import os

    setup_path = get_case_study_dir(case_study_id) / "config" / "setup.yaml"
    setup: dict[str, Any] = {}
    if setup_path.exists():
        with open(setup_path) as f:
            setup = yaml.safe_load(f) or {}
    if "evaluation" not in setup:
        # Under ML4T_OUTPUT_DIR isolation, the redirected setup.yaml may
        # be absent or lack hand-curated sections. Fall back to source.
        test_output = os.environ.get("ML4T_OUTPUT_DIR")
        if test_output:
            from utils import CASE_STUDIES_DIR

            source_path = CASE_STUDIES_DIR / case_study_id / "config" / "setup.yaml"
            if source_path.exists():
                with open(source_path) as f:
                    setup = yaml.safe_load(f) or {}
    if "evaluation" not in setup:
        raise KeyError(
            f"No 'evaluation' section in {setup_path}. "
            f"Expected keys: n_splits, train_size, val_size, holdout_start, holdout_end, calendar."
        )
    evaluation = dict(setup["evaluation"])
    market_semantics = resolve_market_semantics(case_study_id, setup)
    if market_semantics.get("calendar") and not evaluation.get("calendar"):
        evaluation["calendar"] = market_semantics["calendar"]
    return evaluation


def make_walk_forward_config(
    case_study_id: str,
    label_horizon: str = "0D",
    date_col: str = "timestamp",
) -> WalkForwardConfig:
    """Create a WalkForwardConfig from a case study's setup.yaml.

    Bridges the setup.yaml evaluation section to the ml4t-diagnostic
    library's WalkForwardConfig, using its built-in aliases
    (val_size→test_size, holdout_start→test_start, etc.).

    Parameters
    ----------
    case_study_id : str
        Case study identifier (e.g., "etfs").
    label_horizon : str, default "0D"
        Label buffer as duration string (e.g., "21D" for fwd_ret_21d).
    date_col : str, default "timestamp"
        Timestamp column name for the dataset.

    Returns
    -------
    WalkForwardConfig
        Configured for the case study's walk-forward protocol.
    """
    from ml4t.diagnostic.splitters import WalkForwardConfig

    eval_config = load_evaluation_config(case_study_id)
    calendar_id = _map_calendar_id(eval_config.get("calendar"))

    # For D-unit buffers with a calendar, pass as int (trading days)
    normalized_horizon: int | str = _normalize_label_buffer(label_horizon)
    if calendar_id is not None and isinstance(normalized_horizon, str):
        d_match = re.match(r"^(\d+)D$", normalized_horizon)
        if d_match:
            normalized_horizon = int(d_match.group(1))

    return WalkForwardConfig(
        n_splits=eval_config["n_splits"],
        train_size=_normalize_duration(str(eval_config["train_size"])),
        val_size=_normalize_duration(str(eval_config["val_size"])),
        holdout_start=eval_config.get("holdout_start"),
        holdout_end=eval_config.get("holdout_end"),
        label_horizon=normalized_horizon,
        calendar_id=calendar_id,
        timestamp_col=date_col,
        fold_direction="backward",
    )


def make_wf_config(
    case_study_id: str,
    label_horizon: str = "0D",
    date_col: str = "timestamp",
) -> WalkForwardConfig:
    """Backward-compatible alias for make_walk_forward_config."""
    return make_walk_forward_config(
        case_study_id=case_study_id,
        label_horizon=label_horizon,
        date_col=date_col,
    )


def generate_cv_splits(
    dataset: pl.DataFrame | pd.DataFrame,
    case_study_id: str | None = None,
    setup_path: Path | None = None,
    label_buffer: str = "0D",
    date_col: str = "timestamp",
    *,
    cv_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate walk-forward date splits from evaluation config.

    Delegates to ml4t-diagnostic's ``WalkForwardCV`` for calendar-aware
    fold generation. Reads the ``evaluation`` section from ``setup.yaml``
    (via ``case_study_id`` or ``setup_path``).

    Parameters
    ----------
    dataset : pl.DataFrame or pd.DataFrame
        Dataset with a date/timestamp column. Only used to extract unique
        timestamps -- the full panel rows are not needed.
    case_study_id : str, optional
        Case study identifier. Used to locate setup.yaml.
    setup_path : Path, optional
        Explicit path to setup.yaml. Takes precedence over case_study_id.
    label_buffer : str, default "0D"
        Gap between train_end and val_start sized to the label horizon.
        Determined by the label being trained on (e.g., "21D" for fwd_ret_21d).
    date_col : str, default "timestamp"
        Name of the date/timestamp column.
    cv_config : dict, optional
        Pass a cv_config dict directly (e.g. from cv_config.json).
        If provided, case_study_id and setup_path are ignored.

    Returns
    -------
    list[dict]
        List of split dicts with keys: ``fold``, ``train_start``,
        ``train_end``, ``val_start``, ``val_end``.
    """
    from ml4t.diagnostic.splitters import WalkForwardCV
    from ml4t.diagnostic.splitters.config import WalkForwardConfig as LibWalkForwardConfig

    # Legacy path: pre-computed explicit splits
    if cv_config is not None and "splits" in cv_config:
        return cv_config["splits"]

    # Normalize label buffer (strip ISO prefix, convert M → days)
    label_buffer = _normalize_label_buffer(label_buffer)

    # Load evaluation config
    if cv_config is not None:
        # Legacy cv_config dict
        test_size_key = "val_size" if "val_size" in cv_config else "test_size"
        holdout_start_key = "holdout_start" if "holdout_start" in cv_config else "test_start"
        holdout_end_key = "holdout_end" if "holdout_end" in cv_config else "test_end"
        eval_config = {
            "n_splits": cv_config["n_splits"],
            "train_size": str(cv_config["train_size"]),
            "val_size": str(cv_config[test_size_key]),
            "holdout_start": cv_config.get(holdout_start_key),
            "holdout_end": cv_config.get(holdout_end_key),
            "calendar": cv_config.get("calendar"),
        }
    elif setup_path is not None:
        with open(setup_path) as f:
            setup = yaml.safe_load(f)
        eval_config = dict(setup["evaluation"])
    elif case_study_id is not None:
        eval_config = load_evaluation_config(case_study_id)
    else:
        raise ValueError("Provide either case_study_id, setup_path, or cv_config")

    # Map calendar name to library exchange name
    calendar_id = _map_calendar_id(eval_config.get("calendar"))

    # For D-unit buffers with a calendar, pass label_horizon as int so the
    # library interprets it as trading days (not calendar days). This fixes
    # the under-buffering where "21D" → pd.Timedelta("21 days") → ~15 trading
    # days instead of the intended 21 trading days.
    label_horizon: int | str = label_buffer
    if calendar_id is not None:
        d_match = re.match(r"^(\d+)D$", label_buffer)
        if d_match:
            label_horizon = int(d_match.group(1))

    # Build WalkForwardConfig (library Pydantic model)
    config = LibWalkForwardConfig(
        n_splits=eval_config["n_splits"],
        train_size=_normalize_duration(str(eval_config["train_size"])),
        val_size=_normalize_duration(str(eval_config["val_size"])),
        holdout_start=eval_config.get("holdout_start"),
        holdout_end=eval_config.get("holdout_end"),
        label_horizon=label_horizon,
        calendar_id=calendar_id,
        fold_direction="backward",
    )

    # Extract sorted unique timestamps from the dataset
    if isinstance(dataset, pl.DataFrame):
        unique_ts = dataset.select(date_col).unique().sort(date_col).to_series().to_pandas()
    else:
        unique_ts = pd.Series(sorted(dataset[date_col].dropna().unique()))

    if len(unique_ts) == 0:
        raise ValueError("No timestamps found in dataset")

    # Build a single-column DataFrame with DatetimeIndex for the splitter
    ts_index = pd.DatetimeIndex(unique_ts)
    input_tz_naive = ts_index.tz is None
    if input_tz_naive:
        ts_index = ts_index.tz_localize("UTC")
    ts_df = pd.DataFrame(
        {"_dummy": np.zeros(len(ts_index), dtype=np.int8)},
        index=ts_index,
    )

    # Create WalkForwardCV with rolling window (expanding=False)
    cv = WalkForwardCV(config=config)
    cv.expanding = False

    # Generate splits and extract date boundaries.
    # Match tz-awareness to the input data so comparisons work.
    def _ts(idx):
        t = ts_index[idx]
        return t.tz_localize(None) if input_tz_naive else t

    splits = []
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(ts_df)):
        splits.append(
            {
                "fold": fold_i,
                "train_start": _ts(train_idx[0]),
                "train_end": _ts(train_idx[-1]),
                "val_start": _ts(val_idx[0]),
                "val_end": _ts(val_idx[-1]),
            }
        )

    return splits
