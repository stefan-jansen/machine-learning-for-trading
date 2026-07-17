"""Metric computation for predictions and backtests."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_prediction_fold_metrics(
    predictions,
    *,
    y_true_col: str = "y_true",
    y_score_col: str = "y_score",
    fold_col: str = "fold_id",
    date_col: str = "timestamp",
    entity_col: str = "symbol",
    task_type: str = "regression",
    class_values: list | None = None,
    eval_col: str | None = None,
    label: str | None = None,
) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    """Compute standardized metrics from a predictions DataFrame.

    Uses the provided ``task_type`` to decide which metrics to compute.

    For ``task_type="classification"``, ``eval_col`` must be provided and must
    name a column holding the continuous return that the binary/categorical
    label was derived from. IC is computed against that continuous return;
    AUC, log_loss, accuracy, etc. are computed against the binary ``y_true_col``.
    Computing IC against a binary label collapses to ``2·(AUC − 0.5)`` and is
    not a valid Spearman rank correlation against returns.

    Returns (headline_metrics, fold_metrics) where:
    - headline_metrics: aggregated across all folds
    - fold_metrics: per-fold breakdown keyed by fold_id

    Regression metrics: ic, ic_std, rmse, mae, n_entities
    Classification metrics: ic, ic_std, auc_roc, log_loss, brier_score,
        accuracy, balanced_accuracy, auc_pr, n_entities
    """
    import numpy as np
    import polars as pl
    from ml4t.diagnostic.metrics import (
        compute_auc_uncertainty,
        compute_ic_uncertainty,
        cross_sectional_auc_series,
        cross_sectional_ic,
        cross_sectional_ic_series,
    )

    from utils.modeling import compute_classification_metrics

    if not isinstance(predictions, pl.DataFrame):
        predictions = pl.from_pandas(predictions)

    if class_values is None:
        class_values = []

    is_classification = task_type == "classification"
    if is_classification:
        if not eval_col:
            raise ValueError(
                "compute_prediction_fold_metrics(task_type='classification') requires "
                "eval_col — the continuous return column to compute IC against. "
                "Computing IC vs the binary label is 2·(AUC − 0.5) in disguise."
            )
        if eval_col not in predictions.columns:
            raise KeyError(
                f"eval_col {eval_col!r} not present in predictions DataFrame "
                f"(columns: {predictions.columns}). The caller must materialize the "
                f"continuous-return column on every prediction row before registering."
            )
        ic_target_col = eval_col
    else:
        ic_target_col = y_true_col

    folds = sorted(predictions[fold_col].unique().drop_nulls().to_list())
    fold_results = {}

    for fold_id in folds:
        fold_preds = predictions.filter(pl.col(fold_col) == fold_id)

        # Per-date cross-sectional IC — pass polars frame directly (no numpy round-trip).
        _entity = entity_col if entity_col and entity_col in fold_preds.columns else None
        ic_result = cross_sectional_ic(
            fold_preds,
            fold_preds,
            pred_col=y_score_col,
            ret_col=ic_target_col,
            date_col=date_col,
            entity_col=_entity,
            method="spearman",
            min_obs=5,
        )

        yt_fold = fold_preds[y_true_col].to_numpy().astype(float)
        yp_fold = fold_preds[y_score_col].to_numpy().astype(float)
        valid_all = ~(np.isnan(yt_fold) | np.isnan(yp_fold))

        fold_m: dict[str, float] = {
            "ic": ic_result["ic_mean"],
            "ic_std": ic_result["ic_std"],
            "n_entities": int(fold_preds[entity_col].n_unique())
            if entity_col in fold_preds.columns
            else 0,
        }

        if not is_classification:
            fold_m["rmse"] = (
                float(np.sqrt(np.mean((yt_fold[valid_all] - yp_fold[valid_all]) ** 2)))
                if valid_all.any()
                else 0.0
            )
            fold_m["mae"] = (
                float(np.mean(np.abs(yt_fold[valid_all] - yp_fold[valid_all])))
                if valid_all.any()
                else 0.0
            )
        else:
            # Classification metrics: AUC/log_loss/accuracy on the binary y_true.
            cls_m = compute_classification_metrics(yt_fold, yp_fold, class_values)
            # Multiclass ordinal labels (e.g. {-1, 0, 1}) don't get a single
            # auc_roc from compute_classification_metrics. Derive one by
            # collapsing to "up vs not-up" — the natural directional signal
            # for §6b symmetric panels — and persist it as auc_roc so the
            # symmetric panel is a pure registry query regardless of
            # whether the label is binary or 3-class.
            if len(class_values) > 2 and "auc_roc" not in cls_m:
                from sklearn.metrics import roc_auc_score

                yb01 = (yt_fold[valid_all] > 0).astype(int)
                yp_v = yp_fold[valid_all]
                if 0 < yb01.sum() < len(yb01):
                    cls_m["auc_roc"] = float(roc_auc_score(yb01, yp_v))
            fold_m.update(cls_m)

        fold_results[fold_id] = fold_m

    # Headline aggregates — IC always computed
    fold_ics = [fm["ic"] for fm in fold_results.values()]
    headline: dict[str, float | str] = {
        "ic_mean": float(np.mean(fold_ics)) if fold_ics else 0.0,
        "ic_std": float(np.std(fold_ics)) if len(fold_ics) > 1 else 0.0,
        "ic_t": float(np.mean(fold_ics) / (np.std(fold_ics) / np.sqrt(len(fold_ics))))
        if len(fold_ics) > 1 and np.std(fold_ics) > 0
        else 0.0,
        "n_folds": len(folds),
        "pct_positive": float(np.mean([ic > 0 for ic in fold_ics])) if fold_ics else 0.0,
        "task_type": "classification" if task_type == "classification" else "regression",
    }

    # Aggregate classification headline metrics (mean across folds)
    if task_type == "classification":
        cls_metric_names = [
            "auc_roc",
            "auc_pr",
            "log_loss",
            "brier_score",
            "accuracy",
            "balanced_accuracy",
        ]
        for m_name in cls_metric_names:
            vals = [fm[m_name] for fm in fold_results.values() if m_name in fm]
            if vals:
                headline[m_name] = float(np.mean(vals))

    # ---- Daily-pooled uncertainty (HAC + block bootstrap) -----------------
    # The unit of observation is the date, not the asset prediction. Pool all
    # OOS dates across folds into a single series and compute HAC SE +
    # stationary block-bootstrap CI. This is what `model_analysis` notebooks
    # use for headline IC/AUC and CIs.
    horizon = _infer_horizon_from_label(label)
    _entity = entity_col if entity_col and entity_col in predictions.columns else None

    daily_ic = cross_sectional_ic_series(
        predictions,
        predictions,
        pred_col=y_score_col,
        ret_col=ic_target_col,
        date_col=date_col,
        entity_col=_entity,
        method="spearman",
        min_obs=5,
    )
    if isinstance(daily_ic, pl.DataFrame) and daily_ic.drop_nulls("ic").height >= 3:
        ic_unc = compute_ic_uncertainty(
            daily_ic.drop_nulls("ic").select("ic"),
            horizon=int(max(1, horizon)),
            n_boot=1000,
        )
        headline.update(
            {
                "ic_mean_daily": ic_unc["mean_ic"],
                "ic_std_daily": ic_unc["std_ic"],
                "ic_n_days": float(ic_unc["n_days"]),
                "ic_pct_positive": ic_unc["pct_positive"],
                "ic_se_naive": ic_unc["se_naive"],
                "ic_naive_lo": ic_unc["ci_naive_lower"],
                "ic_naive_hi": ic_unc["ci_naive_upper"],
                "ic_se_hac": ic_unc["se_hac"],
                "ic_ci_lo": ic_unc["ci_hac_lower"],
                "ic_ci_hi": ic_unc["ci_hac_upper"],
                "ic_t_hac": ic_unc["t_hac"],
                "ic_p_hac": ic_unc["p_hac"],
                "ic_hac_lag": float(ic_unc["hac_lag"]),
                "ic_boot_lo": ic_unc["ci_boot_lower"],
                "ic_boot_hi": ic_unc["ci_boot_upper"],
                "ic_boot_block": ic_unc["boot_block_size"],
            }
        )

    if is_classification:
        # Daily AUC + uncertainty when the label is binary 0/1.
        unique_classes = predictions[y_true_col].drop_nulls().unique().sort().to_list()
        if set(int(v) for v in unique_classes) <= {0, 1} and len(unique_classes) == 2:
            daily_auc = cross_sectional_auc_series(
                predictions,
                predictions,
                pred_col=y_score_col,
                label_col=y_true_col,
                date_col=date_col,
                entity_col=_entity,
                min_obs=5,
            )
            if isinstance(daily_auc, pl.DataFrame) and daily_auc.drop_nulls("auc").height >= 3:
                auc_unc = compute_auc_uncertainty(
                    daily_auc.drop_nulls("auc").select("auc"),
                    horizon=int(max(1, horizon)),
                    n_boot=1000,
                )
                headline.update(
                    {
                        "auc_mean_daily": auc_unc["mean_auc"],
                        "auc_std_daily": auc_unc["std_auc"],
                        "auc_n_days": float(auc_unc["n_days"]),
                        "auc_pct_above_null": auc_unc["pct_above_null"],
                        "auc_se_naive": auc_unc["se_naive"],
                        "auc_naive_lo": auc_unc["ci_naive_lower"],
                        "auc_naive_hi": auc_unc["ci_naive_upper"],
                        "auc_se_hac": auc_unc["se_hac"],
                        "auc_ci_lo": auc_unc["ci_hac_lower"],
                        "auc_ci_hi": auc_unc["ci_hac_upper"],
                        "auc_t_hac": auc_unc["t_hac"],
                        "auc_p_hac": auc_unc["p_hac"],
                        "auc_hac_lag": float(auc_unc["hac_lag"]),
                        "auc_boot_lo": auc_unc["ci_boot_lower"],
                        "auc_boot_hi": auc_unc["ci_boot_upper"],
                        "auc_boot_block": auc_unc["boot_block_size"],
                    }
                )

    return headline, fold_results


def _infer_horizon_from_label(label: str | None) -> int:
    """Resolve forward-return horizon (in label-step units) from a label name.

    `fwd_ret_5d` -> 5, `fwd_dir_21d` -> 21, `fwd_class_1m` -> 21,
    `fwd_carry_8h` -> 1 (one 8h bar). Defaults to 1 when label is missing.
    Callers should always pass `label=` so the HAC lag matches horizon-1.

    Some deployed labels carry no `<number><unit>` token. `ret_to_expiry`
    (S&P 500 options, hold-to-expiry) overlaps ~35 trading days per its
    `buffer: 35D` setup — resolving it to 1 silently under-lags the HAC
    bandwidth (this was the Ch13 §13.9 exposure). It is resolved by name here.
    Any *other* non-parsing label triggers a warning and a conservative
    fallback of 1; the caller should pass an explicit horizon at the call site.
    """
    if not label:
        return 1
    import re

    s = label.lower()
    # Named horizons for labels that do not carry a <number><unit> token.
    if "to_expiry" in s:
        return 35
    m = re.search(r"(\d+)\s*([dhwm])", s)
    if not m:
        import warnings

        warnings.warn(
            f"_infer_horizon_from_label: label {label!r} has no <number><unit> "
            "token and is not a named horizon; defaulting to horizon=1, which "
            "under-lags the HAC bandwidth for any overlapping label. Pass an "
            "explicit horizon at the call site.",
            stacklevel=2,
        )
        return 1
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "d":
        return n
    if unit == "h":
        return max(1, n // 8)
    if unit == "w":
        return n * 5
    if unit == "m":
        return n * 21
    return n


def compute_backtest_fold_metrics(
    daily_returns,
    case_study_id: str,
    label: str = "",
    *,
    periods_per_year: int = 0,
) -> dict[int, dict[str, float]]:
    """Compute per-fold backtest metrics by slicing daily returns at fold boundaries.

    Uses the evaluation config from setup.yaml to determine fold boundaries,
    then computes PortfolioAnalysis metrics on each fold's return slice.

    Parameters
    ----------
    daily_returns : pl.DataFrame
        [timestamp, daily_return] — full backtest return series.
    case_study_id : str
        Case study identifier for loading setup.yaml.
    label : str
        Label name (e.g., "fwd_ret_21d") — used to compute label buffer
        for fold boundary calculation.
    periods_per_year : int
        Annualization factor. If 0, auto-detected from data frequency.

    Returns
    -------
    dict[int, dict[str, float]]
        {fold_id: {metric: value, ...}, ...}
    """
    import re

    import polars as pl

    from case_studies.utils.backtest_runner import compute_portfolio_metrics
    from case_studies.utils.cv_window import fold_boundaries

    if not isinstance(daily_returns, pl.DataFrame):
        daily_returns = pl.from_pandas(daily_returns)

    # Determine periods_per_year from case study calendar or data frequency
    if periods_per_year == 0 or periods_per_year is None:
        # Try to get calendar from setup.yaml → exchange_calendars
        try:
            from case_studies.utils.backtest_runner import calendar_periods_per_year
            from utils.cv_splits import load_evaluation_config

            eval_cfg = load_evaluation_config(case_study_id)
            calendar = eval_cfg.get("calendar", "NYSE")
            periods_per_year = calendar_periods_per_year(calendar)
        except (KeyError, FileNotFoundError, ImportError):
            # Fallback: estimate from data frequency
            n_obs = len(daily_returns)
            if n_obs > 1:
                ts = daily_returns["timestamp"].unique().sort()
                span_days = (
                    (ts[-1] - ts[0]).total_seconds() / 86400
                    if hasattr(ts[-1] - ts[0], "total_seconds")
                    else float(
                        (ts[-1] - ts[0]).cast(pl.Duration("ms")).dt.total_milliseconds() / 86400000
                    )
                )
                span_years = span_days / 365.25
                obs_per_year = n_obs / span_years if span_years > 0.01 else 252
                if obs_per_year > 350:
                    periods_per_year = 365
                elif obs_per_year > 200:
                    periods_per_year = 252
                elif obs_per_year > 40:
                    periods_per_year = 52
                elif obs_per_year > 8:
                    periods_per_year = 12
                else:
                    periods_per_year = max(1, int(obs_per_year))
            else:
                periods_per_year = 252

    # Infer label buffer from label name (e.g., "fwd_ret_21d" → "21D")
    label_buffer = "0D"
    if label:
        m = re.search(r"(\d+)[dD]", label)
        if m:
            label_buffer = f"{m.group(1)}D"
        elif "8h" in label.lower():
            label_buffer = "1D"

    # Fold boundaries derived from the modeling dataset (same source as
    # canonical_window). Avoids passing val-only daily_returns to the
    # walk-forward splitter — that fails when train_size > val window length
    # (e.g., 10Y train_size on an 8Y val window).
    splits = fold_boundaries(case_study_id, label)
    if not splits:
        logger.warning(
            "Cannot load fold boundaries for %s/%s — skipping fold metrics", case_study_id, label
        )
        return {}

    fold_results: dict[int, dict[str, float]] = {}

    # Cast timestamps to Date for uniform comparison with fold boundaries
    ts_dtype = daily_returns["timestamp"].dtype
    if ts_dtype != pl.Date:
        daily_returns = daily_returns.with_columns(pl.col("timestamp").cast(pl.Date))

    for split in splits:
        fold_id = split["fold"]
        val_start = str(split["val_start"])[:10]  # "YYYY-MM-DD"
        val_end = str(split["val_end"])[:10]

        from datetime import date as date_cls

        start_date = date_cls.fromisoformat(val_start)
        end_date = date_cls.fromisoformat(val_end)

        mask = (daily_returns["timestamp"] >= start_date) & (daily_returns["timestamp"] <= end_date)

        fold_returns = daily_returns.filter(mask)

        if len(fold_returns) < 5:
            logger.debug("Fold %d has only %d returns — skipping", fold_id, len(fold_returns))
            continue

        returns_arr = fold_returns["daily_return"].to_numpy()
        fold_metrics = compute_portfolio_metrics(returns_arr, periods_per_year=periods_per_year)

        # Add fold metadata
        fold_metrics["n_days"] = len(fold_returns)

        fold_results[fold_id] = fold_metrics

    return fold_results


def compute_classification_metrics_from_predictions(
    predictions,
    *,
    y_true_col: str = "actual",
    y_score_col: str = "prediction",
    fold_col: str = "fold",
    eval_col: str | None = None,
    label: str | None = None,
    date_col: str = "timestamp",
    entity_col: str = "symbol",
    class_values: list | None = None,
) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    """Compute classification metrics (AUC/log_loss/brier/accuracy) for an
    existing predictions DataFrame.

    This is a thin wrapper around :func:`compute_prediction_fold_metrics`
    pinned to ``task_type="classification"`` that auto-derives
    ``class_values`` from the unique values of ``y_true_col`` when not
    provided. It exists so notebooks and the registry-AUC backfill script
    invoke the same code path: a future training run that registers a
    classification pred-set populates ``auc_roc`` via
    ``compute_prediction_fold_metrics``; a backfill of pre-existing
    pred-sets populates ``auc_roc`` via this function — the same metric
    function (``compute_classification_metrics``) underlies both.

    When ``eval_col`` (continuous return) is missing from the predictions
    parquet, IC-vs-returns cannot be computed; only the classification
    metrics (AUC/log_loss/etc.) are returned. The headline IC fields are
    populated as 0.0 / NaN to keep the schema stable.

    Returns (headline_metrics, fold_metrics).
    """
    import polars as pl

    if not isinstance(predictions, pl.DataFrame):
        predictions = pl.from_pandas(predictions)

    if class_values is None:
        class_values = (
            predictions[y_true_col].drop_nulls().unique().sort().cast(pl.Float64).to_list()
        )
        # Cast to int when the float values are whole numbers (typical for
        # categorical labels stored as float32). Preserves the {-1, 0, 1}
        # tri-state ordering used by `compute_classification_metrics`.
        if all(float(v).is_integer() for v in class_values):
            class_values = [int(v) for v in class_values]

    # If eval_col is missing, fall back to passing y_true as the IC target;
    # ic computed against the binary label is meaningless but the upstream
    # function still produces classification metrics. Headline IC will be
    # collapsed to 2*(AUC-0.5) — we discard the IC fields downstream when
    # eval_col is absent and only persist the AUC family.
    have_eval = eval_col and eval_col in predictions.columns
    if have_eval:
        return compute_prediction_fold_metrics(
            predictions,
            y_true_col=y_true_col,
            y_score_col=y_score_col,
            fold_col=fold_col,
            date_col=date_col,
            entity_col=entity_col,
            task_type="classification",
            class_values=class_values,
            eval_col=eval_col,
            label=label,
        )

    # No eval_col — compute classification metrics per-fold by hand.
    import numpy as np

    from utils.modeling import compute_classification_metrics

    folds = sorted(predictions[fold_col].unique().drop_nulls().to_list())
    fold_results: dict[int, dict[str, float]] = {}
    for fold_id in folds:
        fold_preds = predictions.filter(pl.col(fold_col) == fold_id)
        yt = fold_preds[y_true_col].to_numpy().astype(float)
        yp = fold_preds[y_score_col].to_numpy().astype(float)
        cls_m = compute_classification_metrics(yt, yp, class_values)
        # Multiclass ordinal labels (e.g. {-1, 0, 1}) don't get a single
        # auc_roc from compute_classification_metrics. Derive one by
        # collapsing to "up vs not-up" — the natural directional signal
        # for §6b symmetric panels — and persist it as auc_roc.
        if len(class_values) > 2 and "auc_roc" not in cls_m:
            from sklearn.metrics import roc_auc_score

            valid = np.isfinite(yt) & np.isfinite(yp)
            yb01 = (yt[valid] > 0).astype(int)
            if 0 < yb01.sum() < len(yb01):
                cls_m["auc_roc"] = float(roc_auc_score(yb01, yp[valid]))
        fold_results[fold_id] = cls_m

    # Headline = mean across folds for each metric that all folds produced.
    headline: dict[str, float] = {"task_type": "classification"}
    if fold_results:
        keys = set().union(*(fm.keys() for fm in fold_results.values()))
        for k in keys:
            vals = [fm[k] for fm in fold_results.values() if k in fm]
            if vals:
                headline[k] = float(np.mean(vals))

    return headline, fold_results


def compute_regression_vs_binary_auc(
    predictions,
    binary_labels,
    *,
    y_score_col: str = "prediction",
    binary_col: str = "y_binary",
    join_keys: tuple[str, ...] = ("symbol", "timestamp"),
) -> dict[str, float]:
    """Compute AUC of a regression score vs a sibling binary direction label.

    Used for Cohort B of the AUC backfill: a regression pred-set
    (``fwd_ret_5d``) gets an AUC computed against the matching binary
    direction label (``fwd_dir_5d``) joined on (symbol, timestamp). The
    regression score is treated as a ranking signal — higher score implies
    "more likely up" — and AUC measures whether high scores rank above
    low ones with respect to the actual direction.

    Parameters
    ----------
    predictions : pl.DataFrame
        Regression pred-set with columns including ``y_score_col`` and
        ``join_keys``.
    binary_labels : pl.DataFrame
        Sibling label parquet with the binary direction column
        ``binary_col`` ∈ {0, 1} and matching ``join_keys``.
    y_score_col : str
        Continuous score column in ``predictions``.
    binary_col : str
        Binary {0, 1} direction column in ``binary_labels``.
    join_keys : tuple of str
        Join keys (default ``(symbol, timestamp)``).

    Returns
    -------
    dict[str, float]
        ``{"auc_roc": float, "auc_pr": float, "n_obs": int}``. Empty dict
        if the join produces fewer than 100 rows or the binary column is
        degenerate.
    """
    import numpy as np
    import polars as pl
    from sklearn.metrics import average_precision_score, roc_auc_score

    if not isinstance(predictions, pl.DataFrame):
        predictions = pl.from_pandas(predictions)
    if not isinstance(binary_labels, pl.DataFrame):
        binary_labels = pl.from_pandas(binary_labels)

    join = predictions.join(
        binary_labels.select([*join_keys, binary_col]),
        on=list(join_keys),
        how="inner",
    )
    if join.height < 100:
        return {}
    yb = join[binary_col].to_numpy().astype(float)
    ys = join[y_score_col].to_numpy().astype(float)
    valid = np.isfinite(ys) & np.isfinite(yb)
    if valid.sum() < 100:
        return {}
    yb, ys = yb[valid], ys[valid]
    # Coerce to {0,1}: treat positive direction as class 1.
    yb01 = (yb > 0).astype(int)
    if yb01.sum() == 0 or yb01.sum() == len(yb01):
        return {}
    return {
        "auc_roc": float(roc_auc_score(yb01, ys)),
        "auc_pr": float(average_precision_score(yb01, ys)),
        "n_obs": int(len(yb01)),
    }


def compute_fold_metrics_from_predictions(
    all_predictions,
    best_config: str,
    best_epoch: int,
    date_col: str = "timestamp",
    entity_col: str = "symbol",
):
    """Compute per-fold cross-sectional IC from a registered predictions table.

    Filters to the best (config, epoch) and groups by fold_id, returning a
    polars DataFrame with [fold_id, ic_mean, n_test].

    Used by deep_learning / tabular_dl / darts_forecasting runners to assemble
    a fold_metrics summary at the end of CV.
    """
    import polars as pl
    from ml4t.diagnostic.metrics import cross_sectional_ic

    if all_predictions.height == 0 or best_config is None:
        return pl.DataFrame()

    best_preds = all_predictions.filter(
        (pl.col("config") == best_config) & (pl.col("epoch") == best_epoch)
    )
    if best_preds.height == 0:
        return pl.DataFrame()

    rows = []
    for fold_id in sorted(best_preds["fold_id"].unique().to_list()):
        fold_df = best_preds.filter(pl.col("fold_id") == fold_id)
        _entity = entity_col if entity_col and entity_col in fold_df.columns else None
        result = cross_sectional_ic(
            fold_df,
            fold_df,
            pred_col="y_score",
            ret_col="y_true",
            date_col=date_col,
            entity_col=_entity,
            method="spearman",
            min_obs=5,
        )
        rows.append(
            {
                "fold_id": fold_id,
                "ic_mean": result["ic_mean"],
                "n_test": fold_df.height,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()
