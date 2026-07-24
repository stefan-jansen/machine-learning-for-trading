from __future__ import annotations

import polars as pl

# Families excluded from ALL backtest sweeps — predictions lack y_score column
_BACKTEST_EXCLUDED_FAMILIES: set[str] = {"causal_dml"}


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
