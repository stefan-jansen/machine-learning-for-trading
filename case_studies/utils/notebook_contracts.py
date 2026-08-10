from __future__ import annotations

from pathlib import Path

import polars as pl

# Families excluded from ALL backtest sweeps — predictions lack y_score column
_BACKTEST_EXCLUDED_FAMILIES: set[str] = {"causal_dml"}

# The minimum cross-section `cross_sectional_ic_series` needs before it will return a
# coefficient for a date. 5 at every case-study call site; the library default is 10.
IC_MIN_OBS = 5

# The predictions parquet schema is not uniform across the nine case studies, so the
# validity rule has to resolve its own column names before it can count anything.
# Measured: etfs / fx_pairs / us_firm_characteristics use prediction+actual+symbol,
# sp500_equity_option_analytics uses y_score+y_true+symbol, cme_futures keys on product.
_PREDICTION_ALIASES = ("prediction", "y_score", "y_pred", "score")
# `eval_actual` first: for a classification label the stored IC is computed against
# the continuous return, which `registry/store.py:712-745` writes under that name
# beside the class target. Resolving `actual`/`y_true` ahead of it would judge a
# date's validity on the class column while the metric used the return.
_ACTUAL_ALIASES = ("eval_actual", "actual", "y_true", "realized", "target")
_ENTITY_ALIASES = ("symbol", "product", "asset", "entity")


def _first_present(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    return next((a for a in aliases if a in columns), None)


def _is_finite(dtype: pl.DataType, column: str) -> pl.Expr:
    """True where the column holds a real number, as the IC series requires.

    Null, NaN and infinity each mean the entity contributes nothing to that date's
    rank correlation. ``is_finite`` is undefined on a non-float column, where being
    non-null is the whole of the condition.
    """
    if dtype.is_float():
        return pl.col(column).is_not_null() & pl.col(column).is_finite()
    return pl.col(column).is_not_null()


def excluded_families(case_study: str, *, for_backtest: bool = False) -> set[str]:
    return set(_BACKTEST_EXCLUDED_FAMILIES) if for_backtest else set()


def excluded_family_sql(
    case_study: str, family_column: str = "family", *, for_backtest: bool = False
) -> tuple[str, list[str]]:
    excluded = sorted(excluded_families(case_study, for_backtest=for_backtest))
    if not excluded:
        return "", []

    placeholders = ", ".join("?" for _ in excluded)
    return f" AND {family_column} NOT IN ({placeholders})", excluded


def degenerate_prediction_sql(prediction_hash_column: str = "p.prediction_hash") -> str:
    """SQL clause excluding prediction sets with any constant-prediction fold.

    When a regularized linear model (LASSO / ElasticNet at high ``alpha_frac``)
    shrinks every coefficient to zero on a fold, that fold's predictions are
    constant and its IC is undefined — stored as NULL in ``fold_metrics.ic``.
    The pooled daily IC is then computed over the surviving folds only, which
    biases it (typically upward) and is not a valid model result. Such
    prediction sets must never be selected for backtesting or any follow-on
    leaderboard.

    Returns a fragment beginning with ``" AND "`` suitable for appending to a
    WHERE clause; takes no bound parameters. Pass the column expression naming
    ``prediction_hash`` in the surrounding query (default ``p.prediction_hash``).
    """
    return (
        f" AND {prediction_hash_column} NOT IN "
        "(SELECT prediction_hash FROM fold_metrics WHERE ic IS NULL)"
    )


def full_coverage_prediction_sql(
    prediction_set_alias: str = "p",
    training_run_alias: str = "t",
    prediction_metric_alias: str = "pm",
) -> str:
    """SQL clause retaining the maximum-coverage rows for each family and label.

    A model family can contain checkpoints evaluated on fewer decision dates than
    its peers even when no fold-level IC is NULL. Comparing or selecting those
    rows against full-coverage checkpoints changes the evaluation sample. The
    eligible surface therefore keeps rows whose ``ic_n_days`` equals the maximum
    for the same ``(split, family, label)``.

    The surrounding query must join ``prediction_sets``, ``training_runs``, and
    ``prediction_metrics`` under the supplied aliases. The returned fragment
    begins with ``" AND "`` and takes no bound parameters.
    """
    return f"""
        AND {prediction_metric_alias}.ic_n_days IS NOT NULL
        AND {prediction_metric_alias}.ic_n_days = (
            SELECT MAX(pm_full.ic_n_days)
            FROM prediction_sets p_full
            JOIN training_runs t_full
              ON p_full.training_hash = t_full.training_hash
            JOIN prediction_metrics pm_full
              ON p_full.prediction_hash = pm_full.prediction_hash
            WHERE p_full.split = {prediction_set_alias}.split
              AND t_full.family = {training_run_alias}.family
              AND t_full.label = {training_run_alias}.label
        )
    """


def canonical_coverage_days(
    case_study: str,
    label: str,
    split: str,
    prediction_hash: str,
    case_dir: Path | None = None,
) -> int | None:
    """Count of a prediction set's *scorable* decision dates inside ``canonical_window``.

    ``ic_n_days`` (stored in ``prediction_metrics`` at registration time) counts every
    date in the predictions parquet passed to the metrics function at write time, which
    can include dates outside the *current* ``canonical_window`` when a prediction set
    predates a CV-window change. Comparing that raw stored count across prediction sets
    whose underlying arrays differ only by such out-of-window dates makes
    ``full_coverage_prediction_sql`` exclude sets that cover the modeling window
    identically to their peers.

    This recomputes coverage directly from the prediction parquet, bounded to the
    window, for use by the ``coverage_window="canonical"`` path of
    ``resolve_best_predictions`` / ``resolve_best_backtest_runs``.

    **A date counts only where its cross-section could be scored.** This stands in for
    ``ic_n_days``, which counts the days ``cross_sectional_ic_series`` actually produced
    a coefficient for, and that function nulls a day unless at least ``min_obs``
    entities carry a finite prediction *and* a finite realized return. Counting rows
    present would let a prediction set with a row every day but two usable names on some
    of them read as full coverage while the raw path correctly discounted it, and the
    two counts are compared against each other. ``min_obs`` is 5 at every case-study
    call site.

    Returns ``None`` when the window, the parquet, or the columns the validity rule
    needs are unavailable. Callers must treat that as "cannot evaluate", not as zero
    coverage.
    """
    from case_studies.utils.cv_window import canonical_window
    from utils.paths import get_case_study_dir

    window = canonical_window(case_study, label, split=split)
    if window is None:
        return None
    if case_dir is None:
        case_dir = get_case_study_dir(case_study)
    path = case_dir / "run_log" / "predictions" / prediction_hash / "predictions.parquet"
    if not path.exists():
        return None
    cols = pl.scan_parquet(path).collect_schema().names()
    date_col = "timestamp" if "timestamp" in cols else ("date" if "date" in cols else None)
    if date_col is None:
        return None
    prediction_col = _first_present(cols, _PREDICTION_ALIASES)
    actual_col = _first_present(cols, _ACTUAL_ALIASES)
    if prediction_col is None or actual_col is None:
        return None
    entity_col = _first_present(cols, _ENTITY_ALIASES)

    selected = [date_col, prediction_col, actual_col] + ([entity_col] if entity_col else [])
    frame = pl.scan_parquet(path).select(selected).collect()
    if frame.is_empty():
        return 0

    # Cast to a calendar date BEFORE grouping, never after: on an intraday case
    # study the column is a Datetime and every timestamp within a day is distinct,
    # so grouping first counts one decision date many times and the count stops
    # being comparable to the daily `ic_n_days` it stands in for.
    if frame.schema[date_col] != pl.Date:
        try:
            frame = frame.with_columns(pl.col(date_col).cast(pl.Date))
        except pl.exceptions.PolarsError:
            # A column named `timestamp` that is not a calendar date is one more
            # "cannot evaluate" case, so it degrades to None like the others
            # rather than raising past every caller.
            return None

    lo, hi = window
    scorable = frame.filter(
        pl.col(date_col).is_between(lo, hi)
        & _is_finite(frame.schema[prediction_col], prediction_col)
        & _is_finite(frame.schema[actual_col], actual_col)
    )
    if scorable.is_empty():
        return 0
    # The cross-section is a set of entities, so a duplicated entity on a date does
    # not widen it. Without an entity column a row is the best available proxy.
    breadth = pl.col(entity_col).n_unique() if entity_col else pl.len()
    per_date = scorable.group_by(date_col).agg(
        breadth.alias("_breadth"),
        # A rank correlation is undefined where either side is the same value for
        # every name on the date, so breadth alone would count a date the IC series
        # returns null for. n_unique rather than a variance: Spearman ranks, and a
        # constant column has one rank whatever its spread.
        pl.col(prediction_col).n_unique().alias("_pred_levels"),
        pl.col(actual_col).n_unique().alias("_actual_levels"),
    )
    defined = (
        (per_date["_breadth"] >= IC_MIN_OBS)
        & (per_date["_pred_levels"] > 1)
        & (per_date["_actual_levels"] > 1)
    )
    return int(defined.sum())


def filter_active_model_rows(
    df: pl.DataFrame,
    case_study: str,
    *,
    family_col: str = "family",
) -> pl.DataFrame:
    if df.is_empty() or family_col not in df.columns:
        return df

    excluded = excluded_families(case_study)
    if not excluded:
        return df

    return df.filter(~pl.col(family_col).is_in(sorted(excluded)))
