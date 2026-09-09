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

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import polars as pl
from ml4t.diagnostic.splitters.calendar import TradingCalendar

from case_studies.utils.notebook_contracts import _first_present, _is_finite

__all__ = [
    "CoverageError",
    "absent_calendar_sessions",
    "assert_sessions_complete",
    "CoverageGap",
    "CoverageReport",
    "declared_sessions",
    "check_prediction_coverage",
    "check_backtest_input_coverage",
    "CrossSectionReport",
    "declared_cross_section",
    "check_prediction_cross_section",
    "feature_panel_keys",
    "BACKTEST_COVERAGE_MINIMUM",
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


def absent_calendar_sessions(
    session_dates: Iterable[date],
    *,
    calendar: str,
    known_absent: Iterable[date] = (),
) -> list[date]:
    """Sessions the exchange held between the first and last of *session_dates*, that are not
    in *session_dates* and are not declared in *known_absent*.

    A panel is normally checked the other way round: each date it carries is asked whether the
    exchange was open, and the dates that fail are dropped as stray prints. That direction
    cannot see this one. A session the exchange held and the archive never printed leaves no
    row to test, so nothing raises, every query succeeds, and one day's rows are simply gone.
    `us_equities_panel`'s single missing session was found by two counts of an unrelated
    quantity differing by one, which is the only way a defect of this shape surfaces on its own.

    It matters because every rolling window downstream reads its input in order and treats
    consecutive elements as consecutive sessions. A variance recursion, a fractional-difference
    convolution and a rolling average all price the gap across a missing session as one day's
    move.

    *known_absent* is the declaration, not a suppression: a caller states the sessions it has
    established are missing upstream, and anything else is returned for the caller to refuse.

    A date counts as a session when it settles itself, which is the rule
    :meth:`TradingCalendar.get_sessions` applies and the same one a stray-print filter uses -
    so the two directions cannot disagree about what a session is. Running it over every
    calendar day of the span enumerates what the exchange held, rather than classifying only
    the dates the archive happens to carry.

    Args:
        session_dates: The panel's session index. Order and duplicates do not matter.
        calendar: Exchange calendar name, as ``config/setup.yaml``'s ``evaluation.calendar``
            gives it.
        known_absent: Sessions already established as missing upstream.

    Returns:
        The undeclared absent sessions, earliest first. Empty when the panel is complete.
    """
    present = {d.date() if isinstance(d, datetime) else d for d in session_dates}
    if not present:
        return []

    span = pd.date_range(str(min(present)), str(max(present)), freq="D", tz="UTC")
    settling = TradingCalendar(calendar).get_sessions(pd.DatetimeIndex(span))
    held = set(pd.DatetimeIndex(settling.to_numpy()).date) & set(span.date)
    declared = {d.date() if isinstance(d, datetime) else d for d in known_absent}
    return sorted(held - present - declared)


def assert_sessions_complete(
    session_dates: Iterable[date],
    *,
    calendar: str,
    known_absent: Iterable[date] = (),
    source: str,
) -> list[date]:
    """Refuse a session index missing a session the exchange held and nobody declared.

    The refusing half of :func:`absent_calendar_sessions`, so the message that explains why a
    gap matters is written once rather than pasted into each notebook that builds an index.

    Returns the declared absences that fall inside this index's own span, so a caller can print
    what it deliberately tolerated. A caller that tolerates nothing gets an empty list.

    :raises CoverageError: on any absent session that *known_absent* does not name.
    """
    undeclared = absent_calendar_sessions(
        session_dates, calendar=calendar, known_absent=known_absent
    )
    if undeclared:
        shown = ", ".join(str(d) for d in undeclared[:10])
        more = "" if len(undeclared) <= 10 else f" (and {len(undeclared) - 10} more)"
        raise CoverageError(
            f"{source}: {len(undeclared)} {calendar} session(s) the exchange held that this "
            f"panel does not carry and nothing declared: {shown}{more}. A rolling window reads "
            "its input in order and treats consecutive elements as consecutive sessions, so a "
            "missing session is priced as though the gap across it were one period's move. "
            "Establish whether the session is absent upstream or dropped in a join, then either "
            "fix the join or declare the session."
        )
    present = {d.date() if isinstance(d, datetime) else d for d in session_dates}
    if not present:
        return []
    first, last = min(present), max(present)
    return sorted(
        d
        for d in (x.date() if isinstance(x, datetime) else x for x in known_absent)
        if first <= d <= last
    )


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
    folds: Sequence[int] | None = None,
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
    if folds is not None:
        # A reduced run fits the folds it declared and no others, so measuring it against all of
        # them reports a gap that is the reduction itself. Narrowing here rather than in the
        # caller keeps every condition below comparing against a declaration: the windows are
        # still setup.yaml's, and a fold present in the frame but outside the subset is still
        # refused as undeclared.
        keep = {int(fold) for fold in folds}
        if not keep:
            raise CoverageError(
                f"{case_study}/{label}/{split}: an empty fold subset asks for no coverage at all"
            )
        declared_ids = {fold for fold, _, _ in windows if fold is not None}
        unknown = sorted(keep - declared_ids)
        if unknown:
            raise CoverageError(
                f"{case_study}/{label}/{split}: folds {unknown} are not declared in setup.yaml, "
                f"which declares {sorted(declared_ids)}; a subset can only narrow"
            )
        windows = [window for window in windows if window[0] in keep]
        expected = {fold: sessions for fold, sessions in expected.items() if fold in keep}

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

    Null is not the whole of it. NaN and infinity are non-null and rank against nothing,
    so a session holding only those is as empty of decisions as one holding only nulls -
    ``_is_finite`` is the same reading the IC series already applies, and on a non-float
    column being non-null is the whole of the condition.
    """
    score_col = _first_present(frame.columns, _SCORE_ALIASES)
    if score_col is None:
        raise CoverageError(
            f"{case_study}/{label}/{split} predictions: no prediction column among "
            f"{_SCORE_ALIASES} in columns {sorted(frame.columns)}. A frame with no score "
            "is not a prediction set, and a check that cannot run must not read as a pass."
        )
    scored = frame.filter(_is_finite(frame.schema[score_col], score_col))
    if scored.is_empty():
        raise CoverageError(
            f"{case_study}/{label}/{split} predictions: no finite value in {score_col!r} across "
            f"{frame.height} rows; the frame carries sessions but no predictions."
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
    folds: Sequence[int] | None = None,
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
        folds=folds,
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


# ---------------------------------------------------------------------------
# The cross-section, which the session checks above cannot see
# ---------------------------------------------------------------------------
#
# Conditions (1)-(3) at the top of this module are all about the time axis: which
# folds, which sessions inside them, and whether the folds span the window. A family
# that scores every session for half the symbols satisfies all three. That is not a
# hypothetical - `case_studies/utils/sequence_dataset.py:215` drops a symbol from a
# fold outright when it holds fewer than `lookback + 1` bars, and `:233` drops every
# endpoint whose preceding window straddles a period gap. Measured on 2026-09-07, the
# `deep_learning` family never scores 258 of 529 symbols in
# `sp500_equity_option_analytics` and 288 of 522 in `sp500_options`, while its session
# coverage is complete and every peer guard agrees with it.
#
# What makes it invisible rather than merely wrong: `expected_prediction_keys` in the
# training identity records what the model *declared* it would deliver, and the
# completeness machinery checks delivery against that declaration. A family that
# narrows the universe narrows its own declaration in the same step, so it is complete
# by its own account and is then ranked against families that were not narrowed.
#
# The declaration this compares against is the label instead: every entity carrying a
# non-null label at a declared session is owed a prediction.

_ENTITY_ALIASES = ("symbol", "product", "asset", "ticker", "pair", "instrument")


def _entity_column(columns: list[str], *, where: str) -> str:
    column = _first_present(columns, _ENTITY_ALIASES)
    if column is None:
        raise CoverageError(
            f"no entity column among {_ENTITY_ALIASES} in {where} columns {sorted(columns)}; "
            "the cross-section cannot be evaluated"
        )
    return column


@dataclass(frozen=True)
class CrossSectionReport:
    """How much of the declared ``(entity, session)`` grid a result actually carries.

    Two denominators, because a family that delivered nothing and a family that was
    handed nothing produce the same shortfall and are different defects. ``expected`` is
    what the label declares. ``achievable`` is the part of that a caller's input panel
    actually reaches, and is ``None`` when no panel was supplied.

    Measured in ``sp500_equity_option_analytics``: 53,712 of 248,460 declared pairs
    (21.6%) carry a label and no row in ``features/financial.parquet``. Every family
    misses exactly those, so a gate denominated on the label alone refuses all five for
    a shortfall none of them caused - while ``deep_learning``'s own further 68,564 and
    ``latent_factors``' own further 16,979, which are the real findings, get no more
    weight than the floor.
    """

    case_study: str
    label: str
    split: str
    source: str
    expected: int
    delivered: int
    achievable: int | None
    delivered_achievable: int | None
    never_scored: tuple[str, ...]
    partially_scored: tuple[str, ...]
    entities_declared: int
    per_fold: tuple[tuple[int | None, int, int], ...]

    @property
    def coverage(self) -> float:
        """Share of what the label declares. The honest denominator, and the reported one."""
        return self.delivered / self.expected if self.expected else 0.0

    @property
    def achievable_coverage(self) -> float | None:
        """Share of what the input panel actually offered, or ``None`` if none was given."""
        if self.achievable is None:
            return None
        return self.delivered_achievable / self.achievable if self.achievable else 0.0

    @property
    def accountable_coverage(self) -> float:
        """What a gate should refuse on: the model's own shortfall, not its input's.

        Falls back to :attr:`coverage` when no panel was supplied, so a caller that
        cannot name the achievable set is held to the label rather than to nothing.
        """
        narrowed = self.achievable_coverage
        return self.coverage if narrowed is None else narrowed

    @property
    def missing(self) -> int:
        return self.expected - self.delivered

    @property
    def complete(self) -> bool:
        return self.missing == 0

    def summary(self) -> str:
        head = (
            f"{self.case_study}/{self.label}/{self.split} {self.source}: "
            f"{self.delivered} of {self.expected} declared (entity, session) pairs "
            f"({self.coverage:.1%}) across {self.entities_declared} entities"
        )
        if self.achievable is not None:
            head += (
                f"; of the {self.achievable} its input panel offered it carries "
                f"{self.delivered_achievable} ({self.accountable_coverage:.1%})"
            )
        if self.complete:
            return f"{head} - complete"
        lines = [f"{head} - {self.missing} missing"]
        if self.never_scored:
            shown = ", ".join(self.never_scored[:10])
            more = f" (+{len(self.never_scored) - 10} more)" if len(self.never_scored) > 10 else ""
            lines.append(f"  never scored ({len(self.never_scored)}): {shown}{more}")
        if self.partially_scored:
            lines.append(f"  scored in part ({len(self.partially_scored)} entities)")
        for fold, delivered, expected in self.per_fold:
            if delivered < expected:
                where = "window" if fold is None else f"fold {fold}"
                lines.append(f"  {where}: {delivered} of {expected}")
        return "\n".join(lines)

    def raise_if_below(self, minimum: float) -> None:
        if self.accountable_coverage < minimum:
            raise CoverageError(
                f"coverage {self.accountable_coverage:.1%} is below {minimum:.1%}\n{self.summary()}"
            )


def declared_cross_section(
    case_study: str,
    label: str,
    *,
    split: str = "validation",
    case_dir: Path | None = None,
    decision_axis: pl.Series | None = None,
) -> pl.DataFrame:
    """Every ``(fold, entity, session)`` the case study owes a prediction for.

    The sessions come from ``declared_sessions``, so the seal, the fold boundaries and
    any ``decision_axis`` narrowing apply here unchanged rather than being re-derived.
    The entities come from the label artifact at those sessions: an entity carrying a
    non-null label is one a model was in a position to rank.

    Returned columns are ``fold``, ``entity`` and ``session``. The entity column is
    renamed because the two sides disagree - ``cme_futures`` keys its labels by
    ``product`` and its prediction panels by ``symbol``, so comparing by the source
    name would report every row missing.
    """
    sessions = declared_sessions(
        case_study, label, split=split, case_dir=case_dir, decision_axis=decision_axis
    )
    path = _label_artifact(case_study, label, case_dir)
    columns = pl.scan_parquet(path).collect_schema().names()
    time_col = _time_column(columns)
    entity_col = _entity_column(columns, where=f"label artifact {path.name}")
    if label not in columns:
        raise CoverageError(f"{path} has no column named {label!r}; the cross-section is undefined")

    panel = (
        pl.scan_parquet(path)
        .filter(pl.col(label).is_not_null())
        .select(
            pl.col(entity_col).cast(pl.String).alias("entity"),
            pl.col(time_col).alias("session"),
        )
        .unique()
        .collect()
    )
    panel = panel.with_columns(_normalize_time(panel.get_column("session")).alias("session"))

    parts = [
        panel.filter(pl.col("session").is_in(pl.Series(values).implode())).with_columns(
            pl.lit(fold, dtype=pl.Int64).alias("fold")
        )
        for fold, values in sessions.items()
        if values
    ]
    if not parts:
        raise CoverageError(
            f"{case_study}/{label}/{split}: no declared fold carries a session, so the "
            "cross-section is empty and coverage cannot be evaluated"
        )
    return pl.concat(parts).select("fold", "entity", "session")


def check_prediction_cross_section(
    predictions: pl.DataFrame,
    case_study: str,
    label: str,
    *,
    split: str = "validation",
    source: str = "predictions",
    case_dir: Path | None = None,
    decision_axis: pl.Series | None = None,
    input_panel: pl.DataFrame | None = None,
    folds: Collection[int] | None = None,
    minimum: float | None = None,
) -> CrossSectionReport:
    """Measure a prediction set against the ``(entity, session)`` grid the label declares.

    ``input_panel`` is the frame the stage was fitted on - a feature panel, a merged
    design matrix. Supplying it splits the shortfall into the part the stage inherited
    and the part it caused, and ``minimum`` is then applied to the latter. Any frame
    carrying an entity and a time column will do; only its distinct keys are read.

    ``minimum`` raises when coverage falls below it; the default returns the report and
    leaves the judgement to the caller, because the legitimate reasons for a shortfall
    (a warm-up window, a universe a family genuinely cannot trade) are not distinguishable
    from the defective ones by the number alone.

    An entity absent from every fold and one missing its first weeks produce the same
    percentage and are different failures, so ``never_scored`` and ``partially_scored``
    are reported apart.

    ``folds`` is the fold axis the run was asked to produce, and it is not derivable from
    the case study: ``declared_sessions`` reads the fold windows from the configuration,
    which lists every configured fold whatever the run did. A run that fitted a subset is
    then charged for folds it was never asked to produce, and reads at the ratio of the two
    counts however complete it is. Pass the folds the run declares and the shortfall is
    measured inside them; the symbol axis is untouched, so a family that lost names within
    the folds it ran is still charged for them.
    """
    scored = _scored_rows(predictions, case_study=case_study, label=label, split=split)
    entity_col = _entity_column(scored.columns, where=f"{source} frame")
    time_col = _time_column(scored.columns)
    got = scored.select(
        pl.col(entity_col).cast(pl.String).alias("entity"),
        pl.col(time_col).alias("session"),
    ).unique()
    got = got.with_columns(_normalize_time(got.get_column("session")).alias("session"))

    want = declared_cross_section(
        case_study, label, split=split, case_dir=case_dir, decision_axis=decision_axis
    )
    if folds is not None:
        kept = sorted(folds)
        want = want.filter(pl.col("fold").is_in(pl.Series(kept, dtype=pl.Int64).implode()))
        if want.is_empty():
            raise CoverageError(
                f"{case_study}/{label}/{split}: the run declares fold(s) {kept} and the "
                "configuration declares none of them, so the cross-section is empty and "
                "coverage cannot be evaluated"
            )
    delivered = want.join(got, on=["entity", "session"], how="semi")
    missing = want.join(got, on=["entity", "session"], how="anti")

    achievable = delivered_achievable = None
    if input_panel is not None:
        # `feature_panel_keys` hands back the canonical two columns already; a caller
        # passing a raw panel gets them resolved. Without this branch the canonical shape
        # is the one shape that fails, because "entity" is not among the source names a
        # panel is allowed to use.
        if {"entity", "session"} <= set(input_panel.columns):
            panel_entity, panel_time = "entity", "session"
        else:
            panel_entity = _entity_column(input_panel.columns, where="input panel")
            panel_time = _time_column(input_panel.columns)
        offered = input_panel.select(
            pl.col(panel_entity).cast(pl.String).alias("entity"),
            pl.col(panel_time).alias("session"),
        ).unique()
        offered = offered.with_columns(
            _normalize_time(offered.get_column("session")).alias("session")
        )
        reachable = want.join(offered, on=["entity", "session"], how="semi")
        achievable = reachable.height
        delivered_achievable = reachable.join(got, on=["entity", "session"], how="semi").height

    scored_entities = set(delivered.get_column("entity").unique().to_list())
    missing_entities = set(missing.get_column("entity").unique().to_list())
    per_fold = tuple(
        (
            None if row["fold"] is None else int(row["fold"]),
            int(row["delivered"]),
            int(row["expected"]),
        )
        for row in want.group_by("fold")
        .agg(expected=pl.len())
        .join(delivered.group_by("fold").agg(delivered=pl.len()), on="fold", how="left")
        .with_columns(pl.col("delivered").fill_null(0))
        .sort("fold", nulls_last=True)
        .iter_rows(named=True)
    )

    report = CrossSectionReport(
        case_study=case_study,
        label=label,
        split=split,
        source=source,
        expected=want.height,
        delivered=delivered.height,
        achievable=achievable,
        delivered_achievable=delivered_achievable,
        never_scored=tuple(sorted(missing_entities - scored_entities)),
        partially_scored=tuple(sorted(missing_entities & scored_entities)),
        entities_declared=want.get_column("entity").n_unique(),
        per_fold=per_fold,
    )
    if minimum is not None:
        report.raise_if_below(minimum)
    return report


#: Share of the achievable cross-section a prediction set must carry to be backtested.
#: A model that scores 98% of what it was handed has a warm-up or a holiday; one that
#: scores two thirds has dropped a universe, and ranking it against a complete peer
#: compares two different experiments. Measured 2026-09-08 across the seven case
#: studies: linear, gbm and tabular_dl deliver 100% everywhere, so this refuses nothing
#: that is whole and is not a threshold tuned to admit a known-bad family.
BACKTEST_COVERAGE_MINIMUM = 0.98


def feature_panel_keys(case_dir: Path | str) -> pl.DataFrame | None:
    """The ``(entity, session)`` pairs the case study's modeling panel offers a model.

    This is the ceiling the model families share: a pair carrying a label but no feature
    row is one no model was in a position to score, and charging it to the models hides
    the families that lost rows they were given. Returns ``None`` when the case study has
    no ``features/financial.parquet``, which leaves the caller measuring against the
    label alone.

    **The financial panel alone, not an intersection across ``features/``.**
    ``load_modeling_dataset`` builds the dataset by scanning ``features/financial.parquet``
    and LEFT-joining ``features/model_based.parquet`` onto it (``utils/modeling.py``, the
    two ``how="left"`` joins), so a key the financial panel carries survives whether or not
    the model-based panel has a row for it - with null model-based columns, which every
    family imputes. Those rows are in the dataset and every family is asked to score them.

    Intersecting the two panels therefore removes keys the models were handed, and it
    removes them from the DENOMINATOR: a family that dropped exactly the rows
    ``model_based`` lacks reads as complete. Measured 2026-09-09 on the registered panels,
    as the share of financial keys the intersection discards:

        case study                      financial   model_based   intersection   discarded
        sp500_equity_option_analytics     481,184       480,938        378,789      21.28%
        crypto_perps_funding               99,877        98,741         92,125       7.76%
        etfs                              404,500       470,662        404,500       0.00%

    A guard that exists to catch a family scoring two thirds of its cross-section cannot
    be measured against a universe 21% narrower than the one the family was given.
    """
    path = Path(case_dir) / "features" / "financial.parquet"
    if not path.is_file():
        return None
    names = pl.scan_parquet(path).collect_schema().names()
    entity = _first_present(names, _ENTITY_ALIASES)
    time_col = _first_present(names, _TIME_ALIASES)
    if entity is None or time_col is None:
        return None
    offered = (
        pl.scan_parquet(path)
        .select(
            pl.col(entity).cast(pl.String).alias("entity"),
            pl.col(time_col).alias("session"),
        )
        .unique()
        .collect()
    )
    return offered.with_columns(_normalize_time(offered.get_column("session")).alias("session"))
