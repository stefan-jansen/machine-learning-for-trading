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

Three conditions, all required:

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

from case_studies.utils.notebook_contracts import _first_present

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


def _normalize_time(series: pl.Series) -> pl.Series:
    """Put a time column into the one representation both sides are compared in.

    The declaration and the frame genuinely disagree in this repo:
    ``crypto_perps_funding/labels/fwd_ret_8h.parquet`` carries ``Datetime('ms', 'UTC')``
    while the prediction diagnostics carry ``Datetime('ms', None)``, and
    ``backtest_loaders.normalize_prediction_columns`` strips the zone and casts ``Date``
    to ``Datetime`` before the backtest ever sees the frame. A tz-aware value never
    equals a naive one, so comparing them as raw objects reports a correct prediction
    set as 100% missing. This mirrors ``backtest_loaders.py`` rather than inventing a
    second convention, so the gate agrees with what the pipeline itself does.
    """
    dtype = series.dtype
    if dtype == pl.Date:
        return series.cast(pl.Datetime("us"))
    if dtype in (pl.String, pl.Utf8):
        return series.str.to_datetime().cast(pl.Datetime("us"))
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone:
            series = series.dt.replace_time_zone(None)
        return series.cast(pl.Datetime("us"))
    return series


def _time_column(columns: list[str]) -> str:
    column = _first_present(columns, _TIME_ALIASES)
    if column is None:
        raise CoverageError(
            f"no time column among {_TIME_ALIASES} in columns {sorted(columns)}; "
            "coverage cannot be evaluated"
        )
    return column


_FOLD_ALIASES = ("fold_id", "fold")


def _resolve_fold_column(columns: list[str], fold_column) -> str | None:
    """Find the fold column under either name the pipeline uses.

    The linear and GBM path writes ``fold`` (``prediction_folds/fold_*.parquet``,
    ``deep_learning.py`` renames ``fold_id`` to ``fold`` on publish) and
    ``backtest_loaders`` renames it back to ``fold_id`` only on the way into the
    backtest. A ``fold_id``-only lookup therefore skipped condition (1) in silence on
    half the artifacts, at the producer site the docstring sends callers to - which is
    the "a check that cannot run must not read as a pass" failure this module exists to
    prevent, in the module itself.
    """
    if fold_column is None:
        return None
    names = (fold_column,) if isinstance(fold_column, str) else tuple(fold_column)
    for name in names:
        if name in columns:
            return name
    return None


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
    return _normalize_time(axis)


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
    from utils.cv_splits import (
        _map_calendar_id,
        _purge_holdout_touching_validation,
        normalize_label_buffer,
    )

    holdout = _holdout_window(case_study)
    if holdout is None:
        return axis
    setup = _load_setup_yaml(case_study)
    horizon = resolve_label_horizon(case_study, label, setup)
    if not horizon:
        return axis
    # generate_cv_splits maps the calendar id and normalizes the buffer before calling
    # the purge function; passing the raw values here would take a different branch of it
    # for a 24/7 case study with an NdD-shaped horizon, and the expectation would then
    # disagree with the folds it claims to agree with by construction.
    calendar = _map_calendar_id(resolve_market_semantics(case_study, setup).get("calendar"))

    stamps = pd.DatetimeIndex(axis.to_list())
    kept = _purge_holdout_touching_validation(
        np.arange(len(stamps)),
        stamps,
        holdout_start=str(holdout[0]),
        outcome_horizon=normalize_label_buffer(str(horizon)),
        calendar_id=calendar,
    )
    return axis.gather(kept.tolist())


def declared_sessions(
    case_study: str,
    label: str,
    *,
    split: str = "validation",
    case_dir: Path | None = None,
    decision_axis: pl.Series | None = None,
) -> dict[int | None, list]:
    """Sessions each declared fold contains, keyed by fold id (``None`` for holdout).

    A session is a timestamp that exists in the label artifact with a non-null label, at
    the case study's own cadence. Reading the axis from the data the case study actually
    trades, rather than from a synthetic calendar, is what makes this work unchanged for
    daily equities, 8-hourly perpetuals and minute-bar microstructure.

    ``decision_axis`` narrows that to the moments a model could actually have decided at,
    and a caller with a feature panel narrower than its label file has to pass it. The two
    differ whenever an input feed has an outage: a forward return is computed from prices
    alone and survives it, while every feature built on the missing feed does not, so the
    label artifact declares sessions no model was ever in a position to predict.
    ``crypto_perps_funding`` has two - the premium-index feed is out for 57 days from
    2021-08-27, and the reduced CI panel loses a further 31 days from 2022-11-02 that the
    full universe covers from another contract. Without this the gate reports a model
    incomplete for not predicting where it was blind.

    It narrows and never widens: a timestamp absent from the label artifact is not made a
    session by appearing in the panel.

    For ``split='validation'`` the axis is sealed against the holdout first, so the last
    fold is not expected to predict a session whose outcome lands inside it.
    """
    windows = _declared_windows(case_study, label, split)
    axis = _session_axis(case_study, label, case_dir)
    if split != "holdout":
        axis = _sealed(case_study, label, axis)
    if decision_axis is not None:
        observable = _normalize_time(decision_axis.unique())
        axis = axis.filter(axis.is_in(observable.implode()))
        if axis.is_empty():
            raise CoverageError(
                f"{case_study}/{label}: the decision axis and the label artifact share no "
                "timestamp, so no session could be declared"
            )

    axis_dates = axis.dt.date() if axis.dtype != pl.Date else axis
    frame = pl.DataFrame({"session": axis, "on": axis_dates})

    sessions: dict[int | None, list] = {}
    for fold, start, end in windows:
        inside = frame.filter(pl.col("on").is_between(start, end))
        sessions[fold] = inside.get_column("session").to_list()
    return sessions


def _reject_label_mismatch(
    frame: pl.DataFrame, *, case_study: str, label: str, split: str, source: str
) -> None:
    """Refuse to check a frame against a label it was not produced under.

    The declared axis is sized by the label's own outcome horizon, so passing the
    case study's primary label while handing in a variant's predictions produces a
    small, plausible, entirely spurious gap. Measured in ``crypto_perps_funding``:
    the 8-hour label declares 2,189 validation sessions and the 24-hour label 2,187,
    because a decision at 2023-12-31 00:00 realizes inside the holdout under a
    24-hour horizon and is purged. Checking a 24-hour artifact against the 8-hour
    label therefore reports exactly two missing sessions and nothing is wrong.

    The mismatch is only visible in one direction from the timestamps themselves - a
    shorter declared horizon makes the observed frame a strict subset, which no
    condition here can distinguish from a genuine gap. So it is caught from the
    frame's own ``label`` column where one exists, and the check is silently skipped
    where it does not rather than being asserted on absent evidence.
    """
    if "label" not in frame.columns:
        return
    present = frame.get_column("label").unique().drop_nulls().to_list()
    if not present or present == [label]:
        return
    raise CoverageError(
        f"{case_study}/{label}/{split} {source}: the frame carries label(s) "
        f"{sorted(str(v) for v in present)}, not {label!r}. The declared session axis "
        "is sized by the label's outcome horizon, so checking one label's predictions "
        "against another's declaration reports a gap that is not there. Pass the label "
        "the frame was produced under."
    )


def _coverage(
    frame: pl.DataFrame,
    *,
    case_study: str,
    label: str,
    split: str,
    source: str,
    case_dir: Path | None,
    fold_column: tuple[str, ...] | str | None,
    decision_axis: pl.Series | None = None,
) -> CoverageReport:
    if frame.is_empty():
        raise CoverageError(
            f"{case_study}/{label}/{split} {source}: frame is empty; a check that cannot "
            "run must not read as a pass"
        )

    _reject_label_mismatch(frame, case_study=case_study, label=label, split=split, source=source)

    time_col = _time_column(frame.columns)
    windows = _declared_windows(case_study, label, split)
    expected = declared_sessions(
        case_study, label, split=split, case_dir=case_dir, decision_axis=decision_axis
    )

    observed = _normalize_time(frame.select(pl.col(time_col)).unique().get_column(time_col))
    observed_set = set(observed.to_list())

    resolved_fold = _resolve_fold_column(frame.columns, fold_column)
    if fold_column is not None and resolved_fold is None and split != "holdout":
        raise CoverageError(
            f"{case_study}/{label}/{split} {source}: no fold column among "
            f"{_FOLD_ALIASES} in columns {sorted(frame.columns)}. The fold checks cannot "
            "run, and a check that cannot run must not read as a pass; pass "
            "fold_column=None to ask for session coverage only."
        )

    gaps: list[CoverageGap] = []

    # (1) The folds present are the folds declared.
    if resolved_fold and split != "holdout":
        declared_ids = {fold for fold, _, _ in windows if fold is not None}
        present_ids = set(frame.get_column(resolved_fold).unique().to_list())
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
            stamps = _normalize_time(
                frame.filter(pl.col(resolved_fold) == fold).get_column(time_col).unique()
            )
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

    # (2) Every session inside a declared fold carries at least one row. On the prediction
    #     path `check_prediction_coverage` has already dropped rows with no score, so a row
    #     there is a prediction; a backtest input frame is not required to carry one.
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
                s
                for s in _sessions_between(
                    case_study, label, end, next_start, case_dir, decision_axis
                )
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

    # (4) Nothing observed outside every declared window. Condition (1) covers this per
    # fold, but only where a fold column exists and only for validation; without this a
    # holdout frame carrying validation or post-holdout sessions reports complete, and
    # observed_sessions counts rows the declaration never asked for, so the summary can
    # read "N of N" while the frame carries extras.
    declared_all = {session for sessions in expected.values() for session in sessions}
    extras = sorted(observed_set - declared_all)
    if extras:
        gaps.append(
            CoverageGap(
                "out_of_window",
                None,
                f"{len(extras)} timestamps belong to no declared {split} window, "
                f"first {extras[0]}, last {extras[-1]}",
                len(extras),
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
    case_study: str,
    label: str,
    after: date,
    before: date,
    case_dir: Path | None,
    decision_axis: pl.Series | None = None,
) -> list:
    """Sessions strictly between two dates, from the label artifact's own axis.

    ``decision_axis`` narrows it the same way it narrows the per-fold expectation, and for the
    same reason: a feed outage between two folds leaves the label artifact carrying sessions no
    model could have decided at, and reporting them as unaccounted for is the gate answering a
    question about the feed as though it were about the folds.
    """
    axis = _sealed(case_study, label, _session_axis(case_study, label, case_dir))
    if decision_axis is not None:
        observable = _normalize_time(decision_axis.unique())
        axis = axis.filter(axis.is_in(observable.implode()))
    as_dates = axis.dt.date() if axis.dtype != pl.Date else axis
    keep = (as_dates > after) & (as_dates < before)
    return axis.filter(keep).to_list()


def _scored_rows(frame: pl.DataFrame, *, case_study: str, label: str, split: str) -> pl.DataFrame:
    """Restrict a prediction frame to the rows that actually carry a score.

    ``_coverage`` counts a session as observed when a row exists for it. On the
    prediction path that is not what the module promises: a frame with no score column
    at all, or one whose scores are all null, described a complete set of decisions
    that were never made.
    """
    score_col = _first_present(frame.columns, _SCORE_ALIASES)
    if score_col is None:
        raise CoverageError(
            f"{case_study}/{label}/{split} predictions: no prediction column among "
            f"{_SCORE_ALIASES} in columns {sorted(frame.columns)}. A frame with no score "
            "is not a prediction set, and a check that cannot run must not read as a pass."
        )
    scored = frame.filter(pl.col(score_col).is_not_null())
    if scored.is_empty():
        raise CoverageError(
            f"{case_study}/{label}/{split} predictions: every value in {score_col!r} is "
            f"null across {frame.height} rows; the frame carries sessions but no predictions."
        )
    return scored


def check_prediction_coverage(
    predictions: pl.DataFrame,
    case_study: str,
    label: str,
    *,
    split: str = "validation",
    case_dir: Path | None = None,
    fold_column: tuple[str, ...] | str | None = _FOLD_ALIASES,
    raise_on_gap: bool = True,
    decision_axis: pl.Series | None = None,
) -> CoverageReport:
    """Assert a prediction set covers the declared validation geometry.

    Call this where the predictions are produced, before anything downstream reads
    them. ``raise_on_gap=False`` returns the report for a notebook that wants to
    display it before failing.

    Condition (2) is read here as the module docstring states it - a session carries a
    **prediction**, not merely a row. The score column is required and rows with a null
    score are dropped before the geometry is measured, so a frame carrying a timestamp
    for every declared session and no usable score reports the gap rather than
    ``complete``. The requirement sits here and not in ``_coverage`` because
    ``check_backtest_input_coverage`` shares that helper and a backtest input frame is
    not required to carry a score column.
    """
    scored = _scored_rows(predictions, case_study=case_study, label=label, split=split)
    report = _coverage(
        scored,
        case_study=case_study,
        label=label,
        split=split,
        source="predictions",
        case_dir=case_dir,
        fold_column=fold_column,
        decision_axis=decision_axis,
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
    fold_column: tuple[str, ...] | str | None = _FOLD_ALIASES,
    raise_on_gap: bool = True,
    decision_axis: pl.Series | None = None,
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
        decision_axis=decision_axis,
    )
    if raise_on_gap:
        report.raise_if_incomplete()
    return report
