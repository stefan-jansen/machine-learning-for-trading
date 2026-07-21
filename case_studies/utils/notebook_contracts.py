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

    Returns a fragment beginning with ``" AND "`` for direct use after an
    existing ``WHERE`` predicate. Callers that assemble a predicate list may
    remove that prefix before joining conditions. Pass the column expression
    naming ``prediction_hash`` in the surrounding query (default
    ``p.prediction_hash``); the fragment takes no bound parameters.
    """
    return (
        " AND NOT EXISTS ("
        "SELECT 1 FROM fold_metrics AS fm "
        f"WHERE fm.prediction_hash = {prediction_hash_column} AND fm.ic IS NULL)"
    )


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
