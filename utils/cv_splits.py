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
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import yaml

from utils.artifact_specs import (
    DEFAULT_LABEL_BUFFER_UNIT,
    LABEL_BUFFER_UNITS,
    resolve_market_semantics,
)
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

    Examples: P5Y → 5YE, P1Y → 1YE, PT8H → 8h, 21D → 21D (unchanged).
    """
    s = re.sub(r"^P?T?", "", s)
    s = re.sub(r"(\d+)H$", r"\1h", s)
    s = re.sub(r"(\d+)T$", r"\1min", s)
    s = re.sub(r"(\d+)Y$", r"\1YE", s)
    return s


def normalize_label_buffer(s: str) -> str:
    """Normalize label buffer for pd.Timedelta compatibility.

    Strips ISO prefix, normalizes units, and converts month-based
    durations to day equivalents since pd.Timedelta rejects 'M' as ambiguous.
    """
    s = _normalize_duration(s)
    m = re.match(r"^(\d+)M$", s)
    if m:
        return f"{int(m.group(1)) * 30}D"
    return s


def _horizon_for_config(
    normalized_buffer: str,
    *,
    calendar_id: str | None,
    buffer_unit: str,
) -> int | str:
    """Turn a normalized buffer into what the splitter should count.

    A ``D`` buffer is passed as an ``int`` so the library counts **sessions**, which is
    right for a session-gridded panel: "21D" as ``pd.Timedelta("21 days")`` is about 15
    sessions, and under-buffering the holdout boundary leaks. It is wrong for a
    calendar-anchored horizon such as ``sp500_options``' 35 days to option expiry, where
    counting 35 sessions over-trims by about two weeks.

    The duration cannot say which it is, so the label declares it -
    ``utils.artifact_specs.resolve_label_buffer_unit``. Without a calendar there are no
    sessions to count and the duration is the only reading available.
    """
    if buffer_unit not in LABEL_BUFFER_UNITS:
        raise ValueError(f"buffer_unit is {buffer_unit!r}, not one of {list(LABEL_BUFFER_UNITS)}")
    if buffer_unit != "sessions" or calendar_id is None:
        return normalized_buffer
    d_match = re.match(r"^(\d+)D$", normalized_buffer)
    return int(d_match.group(1)) if d_match else normalized_buffer


def _purge_holdout_touching_validation(
    val_idx: np.ndarray,
    timestamps: pd.DatetimeIndex,
    *,
    holdout_start: str | None,
    outcome_horizon: str,
    calendar_id: str | None,
    buffer_unit: str = DEFAULT_LABEL_BUFFER_UNIT,
) -> np.ndarray:
    """Exclude validation signals whose label endpoint reaches the holdout.

    ``buffer_unit`` decides how ``outcome_horizon`` is read, the same way it decides it
    for the fold geometry: sessions counted back from the boundary's position, or a
    calendar duration subtracted from the boundary itself. A calendar-anchored horizon
    read as sessions purges further than the label reaches.
    """
    if not holdout_start or outcome_horizon in {"", "0D", "0H"}:
        return val_idx

    boundary = pd.Timestamp(holdout_start)
    if timestamps.tz is not None:
        boundary = (
            boundary.tz_localize(timestamps.tz)
            if boundary.tzinfo is None
            else boundary.tz_convert(timestamps.tz)
        )
    elif boundary.tzinfo is not None:
        boundary = boundary.tz_localize(None)

    trading_day_match = re.fullmatch(r"(\d+)D", outcome_horizon)
    if calendar_id is not None and trading_day_match and buffer_unit == "sessions":
        horizon = int(trading_day_match.group(1))
        holdout_pos = int(timestamps.searchsorted(boundary, side="left"))
        return val_idx[val_idx < holdout_pos - horizon]

    cutoff = boundary - pd.Timedelta(outcome_horizon)
    return val_idx[timestamps[val_idx] < cutoff]


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
    *,
    buffer_unit: str = DEFAULT_LABEL_BUFFER_UNIT,
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
    normalized_horizon = _horizon_for_config(
        normalize_label_buffer(label_horizon), calendar_id=calendar_id, buffer_unit=buffer_unit
    )

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
    *,
    buffer_unit: str = DEFAULT_LABEL_BUFFER_UNIT,
) -> WalkForwardConfig:
    """Backward-compatible alias for make_walk_forward_config."""
    return make_walk_forward_config(
        case_study_id=case_study_id,
        label_horizon=label_horizon,
        date_col=date_col,
        buffer_unit=buffer_unit,
    )


def generate_cv_splits(
    dataset: pl.DataFrame | pd.DataFrame,
    case_study_id: str | None = None,
    setup_path: Path | None = None,
    label_buffer: str = "0D",
    outcome_horizon: str | None = None,
    date_col: str = "timestamp",
    *,
    buffer_unit: str = DEFAULT_LABEL_BUFFER_UNIT,
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
    outcome_horizon : str, optional
        Forward-outcome horizon used to seal validation before holdout. This may
        be shorter than a deliberately conservative train-to-validation buffer.
    date_col : str, default "timestamp"
        Name of the date/timestamp column.
    cv_config : dict, optional
        Pass a cv_config dict directly (e.g. from cv_config.json).
        If provided, case_study_id and setup_path are ignored.

    Returns
    -------
    list[dict]
        Split dicts with keys ``fold``, ``train_start``, ``train_end``,
        ``val_start``, ``val_end``, **ordered oldest first**. Fold 0 validates
        on the earliest window and carries the earliest ``train_start``; the
        last element is the most recent fold. The order is asserted before the
        list is returned, so it cannot change silently.

        Index it only when you mean a position in that order. For "the most
        recent fold" and "everything available before the holdout", call
        :func:`most_recent_split` and :func:`earliest_train_start`, which read
        the boundaries rather than the position and are correct whatever order
        the list is in - they did not change when the order did.
    """
    from ml4t.diagnostic.splitters import WalkForwardCV
    from ml4t.diagnostic.splitters.config import WalkForwardConfig as LibWalkForwardConfig

    # Legacy path: pre-computed explicit splits. Held to the same contract as the
    # generated ones, because the caller cannot tell which path produced its list
    # and reads fold 0 the same way either way.
    if cv_config is not None and "splits" in cv_config:
        precomputed = cv_config["splits"]
        _assert_chronological(precomputed, source="the precomputed splits in cv_config")
        return precomputed

    # Normalize label buffer (strip ISO prefix, convert M → days)
    label_buffer = normalize_label_buffer(label_buffer)
    outcome_horizon = normalize_label_buffer(outcome_horizon or label_buffer)

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
            "step_size": cv_config.get("step_size"),
            "expanding": bool(cv_config.get("expanding", False)),
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
    label_horizon = _horizon_for_config(
        label_buffer, calendar_id=calendar_id, buffer_unit=buffer_unit
    )

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
        step_size=eval_config.get("step_size"),
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

    # Create WalkForwardCV with the resolved rolling or expanding behavior.
    cv = WalkForwardCV(config=config)
    cv.expanding = bool(eval_config.get("expanding", False))

    # Generate splits and extract date boundaries.
    # Match tz-awareness to the input data so comparisons work.
    def _ts(idx):
        t = ts_index[idx]
        return t.tz_localize(None) if input_tz_naive else t

    splits = []
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(ts_df)):
        val_idx = _purge_holdout_touching_validation(
            val_idx,
            ts_index,
            holdout_start=eval_config.get("holdout_start"),
            outcome_horizon=outcome_horizon,
            calendar_id=calendar_id,
            buffer_unit=buffer_unit,
        )
        if len(val_idx) == 0:
            raise ValueError(
                f"Fold {fold_i} has no validation timestamps after purging labels that "
                "touch the holdout boundary"
            )
        splits.append(
            {
                "fold": fold_i,
                "train_start": _ts(train_idx[0]),
                "train_end": _ts(train_idx[-1]),
                "val_start": _ts(val_idx[0]),
                "val_end": _ts(val_idx[-1]),
            }
        )

    _assert_chronological(splits)
    return splits


def _assert_chronological(
    splits: list[dict[str, Any]],
    source: str = "generate_cv_splits",
) -> None:
    """Fail if the folds are not ordered oldest first.

    ``ml4t-diagnostic`` 0.1.4 constructs the backward validation windows from the
    held-out test boundary and then emits the completed folds chronologically, so
    fold 0 validates on the earliest window and the fold id increases with time.
    Every earlier release emitted the same windows in the opposite order. Roughly
    forty call sites read that order - some by indexing, some by writing the fold
    id into an artifact a later stage reads back by id - and a library change that
    reversed it again would leave all of them running while quietly meaning the
    opposite. This turns that into an immediate failure.

    It applies to a ``cv_config`` carrying explicit splits too. A caller cannot
    tell which path produced its list, so a stored fold set that still runs newest
    first hands fold id 0 to the latest window while everything built through the
    generated path now hands it to the earliest, and the two meanings meet in a
    join. Two committed configs carry precomputed splits, and this named them the
    wrong way round until 2026-09-07. Both now run oldest first and both agree with
    what their case study has registered:
    ``us_firm_characteristics/config/cv_config.json`` was renumbered by #791, which
    re-ran the case study rather than migrating its rows.
    ``fx_pairs/config/cv_config.json`` ran newest first, fold 0 validating from
    2023-01-03 down to fold 7 at 2016-01-05, and was refused here until #1073
    renumbered it. That one cost no re-run: its 148 registered training specs were
    already written by 0.1.4's generator and carry ascending ids, so the committed
    file was a stale record rather than an input - no notebook reads it, because
    ``generate_cv_splits`` takes the precomputed path only for a caller that passes
    ``cv_config=`` explicitly. ``us_equities_panel``'s config carries no ``splits``
    list at all and goes through the generated path, so it is not in question.

    ``tests/test_cv_splits.py`` asserts that state directly on the committed files,
    so it is executable rather than a comment that can go stale the way this one
    did.
    """
    val_starts = [_split_value(s, "val_start", "test_start") for s in splits]
    if any(later <= earlier for earlier, later in zip(val_starts, val_starts[1:], strict=False)):
        raise RuntimeError(
            f"{source} produced folds that are not ordered oldest first: "
            f"val_starts {[str(v) for v in val_starts]}. Fold 0 is read as the "
            "earliest fold everywhere, and stage-04 artifacts carry these ids, so a "
            "descending set joins each fold against the wrong end of the sample. "
            "Renumber the source rather than reversing it at the call site."
        )
    # The ids, not just the order. Reversing a descending list leaves fold 0 on the
    # newest window while the list reads oldest first, and every join is by id.
    ids = [s["fold"] for s in splits]
    if ids != list(range(len(splits))):
        raise RuntimeError(
            f"{source} produced fold ids {ids} against list positions "
            f"{list(range(len(splits)))}. The list runs oldest first, so fold 0 is "
            "the earliest fold and the ids have to follow the positions - a "
            "downstream artifact is joined on the id, never on the position."
        )


def _split_value(split: dict[str, Any], *names: str) -> Any:
    """Read the first key a split carries, so a stored config's spelling still resolves."""
    for name in names:
        if split.get(name) is not None:
            return split[name]
    raise KeyError(f"split carries none of {names}: {sorted(split)}")


def most_recent_split(splits: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The fold whose validation window ends last.

    Reads the boundaries rather than a list position, so it is correct whichever
    end of the list that fold sits at. Use it wherever a caller means "the latest
    fold" - ``splits[-1]`` under the name ``last_fold`` takes the *earliest* one.
    """
    if not splits:
        raise ValueError("No splits to choose from")
    return max(splits, key=lambda s: pd.Timestamp(s["val_end"]))


def earliest_train_start(splits: Sequence[dict[str, Any]]) -> pd.Timestamp:
    """The earliest training start across the folds - "everything available".

    A holdout retrain trains on the whole history before the holdout boundary,
    which is ``min(train_start)`` over the fold set and never one fold's own
    start. Reading a single fold's ``train_start`` hands the retrain a shorter
    window than it should have, whichever end of the list that fold sits at.
    """
    if not splits:
        raise ValueError("No splits to choose from")
    return min(pd.Timestamp(s["train_start"]) for s in splits)


def select_folds(
    splits: Sequence[dict[str, Any]],
    fold_ids: Iterable[int],
) -> list[dict[str, Any]]:
    """The folds carrying *fold_ids*, in the order they appear in *splits*.

    A reduction has to say **which** folds it keeps. A count off one end of an
    ordered list does not: ``splits[:2]`` kept the two most recent folds before
    ml4t-diagnostic 0.1.4 and keeps the two earliest after it, and neither reading
    is written down anywhere, so the same code silently became a different
    experiment. Three case studies reduced their fold set that way (#1076).

    Naming the ids is also what makes the reduction checkable against the windows:
    a reader can hold ``[0, 1]`` against the fold table, and cannot hold ``[:2]``
    against anything without knowing which release produced the list.

    This is the same contract the model families already apply to the ``folds``
    key of a preview reduction (``case_studies/utils/linear.py`` and
    ``case_studies/utils/gbm.py`` both filter by id and refuse an id the fold set
    does not carry). ``MAX_FOLDS = n`` is the count form of it, and
    ``tests/pm_helpers.py::PREVIEW_TRANSLATED_PARAMETERS`` is where the harness
    turns that count into ids for every notebook that takes ``PREVIEW_REDUCTIONS``:
    ``list(range(n))``, the earliest n. A notebook that reads ``MAX_FOLDS``
    directly passes ``range(MAX_FOLDS)`` here and means the same thing by it.

    Raises
    ------
    ValueError
        If *fold_ids* is empty, or names an id the fold set does not carry. A
        reduction that silently keeps fewer folds than it asked for reports under
        the same name as one that got what it asked for.
    """
    requested = [int(fold_id) for fold_id in fold_ids]
    if not requested:
        raise ValueError("select_folds was given no fold ids; a reduction has to keep some fold")
    available = {int(_split_value(s, "fold")): s for s in splits}
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(
            f"fold reduction names {missing}, which the fold set does not carry - "
            f"it has {sorted(available)}. Reduce to ids that exist rather than to a "
            "count, so a set with fewer folds than expected fails here instead of "
            "reporting a smaller experiment under the same name."
        )
    wanted = set(requested)
    return [s for s in splits if int(_split_value(s, "fold")) in wanted]
