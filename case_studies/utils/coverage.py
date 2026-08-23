"""Absolute coverage checks against the declared validation geometry.

The pipeline already had three coverage guards before this module, and all three
compare a result against its *peers*: ``full_coverage_prediction_sql`` keeps rows whose
``ic_n_days`` equals the maximum for the same ``(split, family, label)``, the period
counts compare one backtest's observation count against another's, and
``rank_returns_on_common_support`` intersects the periods two results share. A relative
guard cannot see a failure that moves every peer the same way. When a fold is stale,
when a validation boundary does not line up, or when a join upstream of the backtest
drops a whole window, every candidate loses the same sessions, they all still agree
with each other, and all three guards pass.

This module compares against the **declaration** instead: the fold boundaries in
``setup.yaml``, and the sessions that actually exist in the label artifact inside them.
That is an absolute reference, so it fails when every peer is wrong together.

Three conditions, all required, from ``reference/CASE_STUDY_PIPELINE.md`` section 3:

1. the folds present are the folds declared, and each one's timestamps lie inside its
   declared window;
2. every session inside a declared fold carries at least one prediction;
3. the declared folds account for the whole declared window.

**A check that cannot run must never look like a check that passed.** Every entry point
here raises when the declaration or the label artifact is unavailable, rather than
returning an empty report. The alternative was measured on 2026-08-23: a registry query
that returned ``[]`` for a missing file made "wrote somewhere else" and "wrote nothing"
the same observation, and the assertion on top of it reported the wrong defect for a
day.

Coverage is not trading. An allocator that holds nothing for a month has complete
coverage and no positions; a flat day is an observation of zero, not an abstention.
Whether a strategy that barely trades is worth reporting is a separate question with a
separate answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from case_studies.utils.notebook_contracts import _ENTITY_ALIASES, _first_present

__all__ = [
    "CoverageError",
    "CoverageGap",
    "CoverageReport",
    "declared_sessions",
    "check_prediction_coverage",
    "check_backtest_input_coverage",
]

_TIME_ALIASES = ("timestamp", "date", "datetime", "ts")
_SCORE_ALIASES = ("prediction", "y_score", "y_pred", "score", "signal", "weight")


class CoverageError(RuntimeError):
    """Raised when coverage is incomplete, or when it cannot be evaluated at all."""


@dataclass(frozen=True)
class CoverageGap:
    """One way the observed sessions depart from the declared ones."""

    kind: str
    fold: int | None
    detail: str
    n: int

    def __str__(self) -> str:
        where = "window" if self.fold is None else f"fold {self.fold}"
        return f"[{self.kind}] {where}: {self.detail}"


@dataclass(frozen=True)
class CoverageReport:
    case_study: str
    label: str
    split: str
    source: str
    declared_folds: int
    expected_sessions: int
    observed_sessions: int
    gaps: tuple[CoverageGap, ...]

    @property
    def complete(self) -> bool:
        return not self.gaps

    def summary(self) -> str:
        head = (
            f"{self.case_study}/{self.label}/{self.split} {self.source}: "
            f"{self.observed_sessions} of {self.expected_sessions} declared sessions "
            f"across {self.declared_folds} folds"
        )
        if self.complete:
            return f"{head} - complete"
        return "\n".join([f"{head} - INCOMPLETE"] + [f"  {gap}" for gap in self.gaps])

    def raise_if_incomplete(self) -> None:
        if not self.complete:
            raise CoverageError(self.summary())


def _as_date(value) -> date:
    return value if isinstance(value, date) else value.date()


def _time_column(columns: list[str]) -> str:
    column = _first_present(columns, _TIME_ALIASES)
    if column is None:
        raise CoverageError(
            f"no time column among {_TIME_ALIASES} in columns {sorted(columns)}; "
            "coverage cannot be evaluated"
        )
    return column


def _label_artifact(case_study: str, label: str, case_dir: Path | None) -> Path:
    if case_dir is None:
        from utils.paths import get_case_study_dir

        case_dir = get_case_study_dir(case_study)
    path = Path(case_dir) / "labels" / f"{label}.parquet"
    if not path.exists():
        raise CoverageError(
            f"label artifact {path} does not exist; the session axis is undefined and "
            "coverage cannot be evaluated"
        )
    return path


def _declared_windows(
    case_study: str, label: str, split: str
) -> list[tuple[int | None, date, date]]:
    """The declared per-fold windows for ``split``, oldest first."""
    from case_studies.utils.cv_window import canonical_window, fold_boundaries

    if split == "holdout":
        window = canonical_window(case_study, label, split="holdout")
        if window is None:
            raise CoverageError(
                f"{case_study}/{label}: no holdout window configured; coverage cannot be evaluated"
            )
        return [(None, _as_date(window[0]), _as_date(window[1]))]

    folds = fold_boundaries(case_study, label)
    if not folds:
        raise CoverageError(
            f"{case_study}/{label}: no CV fold boundaries derivable from setup.yaml; "
            "coverage cannot be evaluated"
        )
    windows = [(int(f["fold"]), _as_date(f["val_start"]), _as_date(f["val_end"])) for f in folds]
    return sorted(windows, key=lambda w: w[1])


def _session_axis(case_study: str, label: str, case_dir: Path | None) -> pl.Series:
    """Every timestamp the case study could have predicted, oldest first."""
    path = _label_artifact(case_study, label, case_dir)
    columns = pl.scan_parquet(path).collect_schema().names()
    time_col = _time_column(columns)
    if label not in columns:
        raise CoverageError(f"{path} has no column named {label!r}; coverage cannot be evaluated")

    axis = (
        pl.scan_parquet(path)
        .filter(pl.col(label).is_not_null())
        .select(pl.col(time_col))
        .unique()
        .sort(time_col)
        .collect()
        .get_column(time_col)
    )
    if axis.is_empty():
        raise CoverageError(
            f"{path} carries no non-null {label!r}; the session axis is empty and "
            "coverage cannot be evaluated"
        )
    return axis


def _sealed(case_study: str, label: str, axis: pl.Series) -> pl.Series:
    """Drop the sessions the outcome-horizon seal removes from validation.

    The declared ``val_end`` is a calendar date and the seal is sized in the label's own
    horizon, which can be shorter than a day. On an 8-hourly case study the last fold
    therefore ends mid-date, and a date-granular window over-declares by up to one
    cadence. Rather than re-derive the rule, this calls the same
    ``_purge_holdout_touching_validation`` the fold generator calls, so the expectation
    agrees with the folds by construction instead of by reimplementation.
    """
    import numpy as np
    import pandas as pd

    from case_studies.utils.cv_window import _holdout_window, _load_setup_yaml
    from utils.artifact_specs import resolve_label_horizon, resolve_market_semantics
    from utils.cv_splits import _purge_holdout_touching_validation

    holdout = _holdout_window(case_study)
    if holdout is None:
        return axis
    setup = _load_setup_yaml(case_study)
    horizon = resolve_label_horizon(case_study, label, setup)
    if not horizon:
        return axis
    calendar = resolve_market_semantics(case_study, setup).get("calendar")

    stamps = pd.DatetimeIndex(axis.to_list())
    kept = _purge_holdout_touching_validation(
        np.arange(len(stamps)),
        stamps,
        holdout_start=str(holdout[0]),
        outcome_horizon=str(horizon),
        calendar_id=calendar,
    )
    return axis.gather(kept.tolist())


def declared_sessions(
    case_study: str,
    label: str,
    *,
    split: str = "validation",
    case_dir: Path | None = None,
) -> dict[int | None, list]:
    """Sessions each declared fold contains, keyed by fold id (``None`` for holdout).

    A session is a timestamp that exists in the label artifact with a non-null label, at
    the case study's own cadence. Reading the axis from the data the case study actually
    trades, rather than from a synthetic calendar, is what makes this work unchanged for
    daily equities, 8-hourly perpetuals and minute-bar microstructure.

    For ``split='validation'`` the axis is sealed against the holdout first, so the last
    fold is not expected to predict a session whose outcome lands inside it.
    """
    windows = _declared_windows(case_study, label, split)
    axis = _session_axis(case_study, label, case_dir)
    if split != "holdout":
        axis = _sealed(case_study, label, axis)

    axis_dates = axis.dt.date() if axis.dtype != pl.Date else axis
    frame = pl.DataFrame({"session": axis, "on": axis_dates})

    sessions: dict[int | None, list] = {}
    for fold, start, end in windows:
        inside = frame.filter(pl.col("on").is_between(start, end))
        sessions[fold] = inside.get_column("session").to_list()
    return sessions


def _coverage(
    frame: pl.DataFrame,
    *,
    case_study: str,
    label: str,
    split: str,
    source: str,
    case_dir: Path | None,
    fold_column: str | None,
) -> CoverageReport:
    if frame.is_empty():
        raise CoverageError(
            f"{case_study}/{label}/{split} {source}: frame is empty; a check that cannot "
            "run must not read as a pass"
        )

    time_col = _time_column(frame.columns)
    windows = _declared_windows(case_study, label, split)
    expected = declared_sessions(case_study, label, split=split, case_dir=case_dir)

    observed = frame.select(pl.col(time_col)).unique().get_column(time_col)
    observed_set = set(observed.to_list())

    gaps: list[CoverageGap] = []

    # (1) The folds present are the folds declared.
    if fold_column and fold_column in frame.columns and split != "holdout":
        declared_ids = {fold for fold, _, _ in windows if fold is not None}
        present_ids = set(frame.get_column(fold_column).unique().to_list())
        for missing in sorted(declared_ids - present_ids):
            gaps.append(
                CoverageGap(
                    "missing_fold",
                    missing,
                    "declared in setup.yaml, absent from the frame",
                    0,
                )
            )
        for extra in sorted(present_ids - declared_ids):
            gaps.append(
                CoverageGap(
                    "undeclared_fold",
                    extra,
                    "present in the frame, not declared in setup.yaml - a stale fold",
                    0,
                )
            )
        bounds = {fold: (start, end) for fold, start, end in windows if fold is not None}
        for fold in sorted(declared_ids & present_ids):
            start, end = bounds[fold]
            stamps = frame.filter(pl.col(fold_column) == fold).get_column(time_col).unique()
            if stamps.is_empty():
                continue
            as_dates = stamps.dt.date() if stamps.dtype != pl.Date else stamps
            outside = int((~as_dates.is_between(start, end)).sum())
            if outside:
                gaps.append(
                    CoverageGap(
                        "out_of_window",
                        fold,
                        f"{outside} timestamps outside the declared [{start}, {end}]",
                        outside,
                    )
                )

    # (2) Every session inside a declared fold carries at least one row.
    expected_total = 0
    for fold, sessions in expected.items():
        expected_total += len(sessions)
        missing = [s for s in sessions if s not in observed_set]
        if missing:
            gaps.append(
                CoverageGap(
                    "missing_sessions",
                    fold,
                    f"{len(missing)} of {len(sessions)} declared sessions absent, "
                    f"first {missing[0]}, last {missing[-1]}",
                    len(missing),
                )
            )

    # (3) The declared folds account for the whole declared window.
    if split != "holdout" and len(windows) > 1:
        for (_, _, end), (next_fold, next_start, _) in zip(windows, windows[1:]):
            if next_start <= end:
                continue
            unaccounted = [
                s for s in _sessions_between(case_study, label, end, next_start, case_dir)
            ]
            if unaccounted:
                gaps.append(
                    CoverageGap(
                        "unaccounted_window",
                        next_fold,
                        f"{len(unaccounted)} sessions between {end} and {next_start} "
                        "belong to no declared fold",
                        len(unaccounted),
                    )
                )

    return CoverageReport(
        case_study=case_study,
        label=label,
        split=split,
        source=source,
        declared_folds=len(windows),
        expected_sessions=expected_total,
        observed_sessions=len(observed_set),
        gaps=tuple(gaps),
    )


def _sessions_between(
    case_study: str, label: str, after: date, before: date, case_dir: Path | None
) -> list:
    """Sessions strictly between two dates, from the label artifact's own axis."""
    axis = _sealed(case_study, label, _session_axis(case_study, label, case_dir))
    as_dates = axis.dt.date() if axis.dtype != pl.Date else axis
    keep = (as_dates > after) & (as_dates < before)
    return axis.filter(keep).to_list()


def check_prediction_coverage(
    predictions: pl.DataFrame,
    case_study: str,
    label: str,
    *,
    split: str = "validation",
    case_dir: Path | None = None,
    fold_column: str | None = "fold_id",
    raise_on_gap: bool = True,
) -> CoverageReport:
    """Assert a prediction set covers the declared validation geometry.

    Call this where the predictions are produced, before anything downstream reads
    them. ``raise_on_gap=False`` returns the report for a notebook that wants to
    display it before failing.
    """
    report = _coverage(
        predictions,
        case_study=case_study,
        label=label,
        split=split,
        source="predictions",
        case_dir=case_dir,
        fold_column=fold_column,
    )
    if raise_on_gap:
        report.raise_if_incomplete()
    return report


def check_backtest_input_coverage(
    signals: pl.DataFrame,
    case_study: str,
    label: str,
    *,
    split: str = "validation",
    case_dir: Path | None = None,
    fold_column: str | None = "fold_id",
    raise_on_gap: bool = True,
) -> CoverageReport:
    """Assert the frame a backtest is about to consume still covers the declaration.

    Separate from ``check_prediction_coverage`` because a complete prediction set can
    be reduced to a partial one on the way in. Measured in ``crypto_perps_funding``: an
    inner join against conformal widths dropped an entire fold in silence, because
    ``backtest_runner`` drops unsupported *timestamps* while raising on unsupported
    *symbols* (``backtest_runner.py:2239``). Checking the output would not have found
    it - the backtest reported a full set of returns, because a flat day is an
    observation of zero.
    """
    report = _coverage(
        signals,
        case_study=case_study,
        label=label,
        split=split,
        source="backtest input",
        case_dir=case_dir,
        fold_column=fold_column,
    )
    if raise_on_gap:
        report.raise_if_incomplete()
    return report
