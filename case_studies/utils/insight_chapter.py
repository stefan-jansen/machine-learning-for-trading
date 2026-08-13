"""Cross-case-study aggregation for chapter insight notebooks (Ch11–Ch15).

Wraps the per-case-study helpers in `model_analysis` with a thin "collect across
N case studies" layer plus the plotters the insight chapters draw.

The chapters ask one question of nine case studies: which configuration ranked
first, and how did it behave across folds? So the flow is always *select once,
then derive* — `collect_rank1_per_cs` picks the winning configuration per case
study, and everything else takes that frame rather than running its own
selection and risking a different answer.

Selection compares only what is comparable. A configuration evaluated on three
folds and one evaluated on six do not have comparable ICs, so candidates are
first restricted to those covering every declared fold and the same number of
days; only then does the highest daily-pooled IC win.

Usage::

    from case_studies.utils.insight_chapter import (
        collect_rank1_per_cs,
        collect_fold_ic_per_cs,
        plot_cross_cs_forest,
    )
    from case_studies.utils.analytics import CASE_STUDY_IDS

    rank1 = collect_rank1_per_cs(CASE_STUDY_IDS, family="linear")
    folds = collect_fold_ic_per_cs(rank1)
    plot_cross_cs_forest(rank1, family="linear",
                         title="Linear: rank-1 per case study (primary label)")
"""

from __future__ import annotations

import contextlib
import json
import math
import sqlite3
from collections.abc import Callable, Iterable

import polars as pl

# ml4t.diagnostic dlopens cudart; load torch first so its bundled CUDA
# runtime wins. Same pattern as case_studies/utils/model_analysis.py.
import torch  # noqa: F401

from case_studies.utils.analytics import PRIMARY_LABELS, SHORT_NAMES
from case_studies.utils.conformal import split_conformal_coverage
from utils.paths import get_case_study_dir

LabelResolver = Callable[[str], str | None]


class RegistrySelectionError(ValueError):
    """Raised when registry candidates cannot support a comparable rank-one selection."""


def compare_ic_on_shared_timestamps(
    left: pl.DataFrame,
    right: pl.DataFrame,
) -> dict[str, float | int]:
    """Average two IC series over their exact evaluation-timestamp intersection."""
    required = {"date", "ic"}
    for name, frame in (("left", left), ("right", right)):
        missing = required - set(frame.columns)
        if missing:
            raise RegistrySelectionError(f"{name} daily IC is missing {sorted(missing)}")

    def _daily(frame: pl.DataFrame, value_name: str) -> pl.DataFrame:
        return (
            frame.select("date", "ic")
            .drop_nulls()
            .with_columns(pl.col("date").cast(pl.Datetime("ns")))
            .filter(pl.col("ic").is_finite())
            .group_by("date")
            .agg(pl.col("ic").mean().alias(value_name))
        )

    shared = _daily(left, "left_ic").join(_daily(right, "right_ic"), on="date", how="inner")
    if shared.is_empty():
        raise RegistrySelectionError("IC series have no shared evaluation timestamps")
    return {
        "left_ic": float(shared["left_ic"].mean()),
        "right_ic": float(shared["right_ic"].mean()),
        "n_timestamps": shared.height,
    }


def _resolve_label(case_study: str, label_resolver: LabelResolver | None) -> str:
    if label_resolver is None:
        return PRIMARY_LABELS[case_study]
    out = label_resolver(case_study)
    return out if out is not None else PRIMARY_LABELS[case_study]


def select_rank1(
    metrics: pl.DataFrame,
    folds: pl.DataFrame,
    *,
    expected_fold_ids: Iterable[int],
) -> dict:
    """Select the best configuration among those that are comparable.

    A candidate is comparable when it reports every declared fold with a finite
    IC and covers the same number of days as the best-covered candidate. Ranking
    IC across candidates that saw different folds or different days would reward
    a short evaluation window rather than a better model.
    """
    metric_columns = {
        "prediction_hash",
        "training_hash",
        "config_name",
        "ic_mean_daily",
        "ic_n_days",
        "ic_se_hac",
        "ic_ci_lo",
        "ic_ci_hi",
        "ic_t_hac",
        "ic_p_hac",
        "ic_hac_lag",
    }
    fold_columns = {"prediction_hash", "fold_id", "ic"}
    missing_metrics = metric_columns - set(metrics.columns)
    missing_folds = fold_columns - set(folds.columns)
    if missing_metrics or missing_folds:
        raise RegistrySelectionError(
            f"missing selection columns: metrics={sorted(missing_metrics)}, "
            f"folds={sorted(missing_folds)}"
        )
    if metrics.is_empty():
        raise RegistrySelectionError("no registry candidates")

    expected = set(expected_fold_ids)
    if not expected:
        raise RegistrySelectionError("expected fold set is empty")

    eligible: list[dict] = []
    finite_columns = [
        "ic_mean_daily",
        "ic_n_days",
        "ic_se_hac",
        "ic_ci_lo",
        "ic_ci_hi",
        "ic_t_hac",
        "ic_p_hac",
        "ic_hac_lag",
    ]
    for row in metrics.iter_rows(named=True):
        prediction_hash = row["prediction_hash"]
        if any(
            row[column] is None or not math.isfinite(float(row[column]))
            for column in finite_columns
        ):
            continue
        if float(row["ic_n_days"]) <= 0:
            continue

        candidate_folds = folds.filter(pl.col("prediction_hash") == prediction_hash)
        fold_ids = candidate_folds["fold_id"].to_list()
        if len(fold_ids) != len(set(fold_ids)) or set(fold_ids) != expected:
            continue
        if any(value is None or not math.isfinite(float(value)) for value in candidate_folds["ic"]):
            continue
        eligible.append(row)

    if not eligible:
        raise RegistrySelectionError("no finite candidate covers every declared fold")
    full_days = max(float(row["ic_n_days"]) for row in eligible)
    comparable = [row for row in eligible if float(row["ic_n_days"]) == full_days]
    return max(comparable, key=lambda row: float(row["ic_mean_daily"]))


def _raw_primary_candidates(
    case_study: str, family: str, label: str
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    if not db_path.exists():
        return pl.DataFrame(), pl.DataFrame(), 0
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """
            SELECT p.prediction_hash, t.training_hash, t.family, t.config_name, t.label,
                   p.checkpoint_value, p.checkpoint_kind, t.git_commit, t.spec_json,
                   t.created_at AS training_created_at, pm.computed_at,
                   pm.ic_mean, pm.ic_std, pm.ic_mean_daily, pm.ic_n_days,
                   pm.ic_se_hac, pm.ic_ci_lo, pm.ic_ci_hi, pm.ic_t_hac,
                   pm.ic_p_hac, pm.ic_hac_lag
            FROM training_runs t
            JOIN prediction_sets p ON p.training_hash = t.training_hash
            JOIN prediction_metrics pm ON pm.prediction_hash = p.prediction_hash
            WHERE t.family = ? AND t.label = ? AND p.split = 'validation'
            """,
            (family, label),
        ).fetchall()
        fold_rows = db.execute(
            """
            SELECT p.prediction_hash, fm.fold_id, fm.ic
            FROM training_runs t
            JOIN prediction_sets p ON p.training_hash = t.training_hash
            JOIN fold_metrics fm ON fm.prediction_hash = p.prediction_hash
            WHERE t.family = ? AND t.label = ? AND p.split = 'validation'
            """,
            (family, label),
        ).fetchall()
    if not rows:
        return pl.DataFrame(), pl.DataFrame(), 0
    metrics = pl.DataFrame([dict(row) for row in rows], infer_schema_length=None)
    fold_metrics = pl.DataFrame([dict(row) for row in fold_rows], infer_schema_length=None)
    declared_folds = []
    for spec_json in metrics["spec_json"].drop_nulls().to_list():
        with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
            value = int(json.loads(spec_json).get("n_folds", 0))
            if value > 0:
                declared_folds.append(value)
    return metrics, fold_metrics, max(declared_folds, default=0)


def collect_rank1_per_cs(
    case_studies: Iterable[str],
    family: str,
    label_resolver: LabelResolver | None = None,
) -> pl.DataFrame:
    """Rank-one configuration per case study, by daily-pooled IC.

    A case study with no registry rows for this family is left out of the
    result, so an absent case study shows up as a missing row rather than an
    exception. Every other collector takes the frame this returns.
    """
    rows = []
    for case_study in case_studies:
        label = _resolve_label(case_study, label_resolver)
        metrics, folds, n_folds = _raw_primary_candidates(case_study, family, label)
        if metrics.is_empty():
            continue
        if n_folds <= 0:
            raise RegistrySelectionError(f"{case_study}/{family}/{label}: n_folds is not declared")
        try:
            row = select_rank1(metrics, folds, expected_fold_ids=range(n_folds))
        except RegistrySelectionError as exc:
            raise RegistrySelectionError(f"{case_study}/{family}/{label}: {exc}") from exc
        row.update(
            case_study=case_study,
            short_name=SHORT_NAMES.get(case_study, case_study),
        )
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def collect_fold_ic_per_cs(rank1: pl.DataFrame) -> pl.DataFrame:
    """Per-fold IC for each configuration `collect_rank1_per_cs` selected."""
    rows = []
    for selected in rank1.iter_rows(named=True):
        db_path = get_case_study_dir(selected["case_study"]) / "run_log" / "registry.db"
        uri = f"file:{db_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            db.row_factory = sqlite3.Row
            fold_rows = db.execute(
                """
                SELECT fold_id, ic, ic_std, n_entities, rmse, mae
                FROM fold_metrics WHERE prediction_hash = ? ORDER BY fold_id
                """,
                (selected["prediction_hash"],),
            ).fetchall()
        for fold in fold_rows:
            rows.append(
                {
                    "case_study": selected["case_study"],
                    "short_name": selected["short_name"],
                    "family": selected["family"],
                    "config_name": selected["config_name"],
                    "label": selected["label"],
                    "prediction_hash": selected["prediction_hash"],
                    **dict(fold),
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def collect_checkpoint_fold_trajectories(rank1: pl.DataFrame) -> pl.DataFrame:
    """Per-fold IC at every checkpoint of each selected training run."""
    rows = []
    for selected in rank1.iter_rows(named=True):
        spec = json.loads(selected["spec_json"])
        n_folds = int(spec.get("n_folds", 0))
        if n_folds <= 0:
            raise RegistrySelectionError(
                f"{selected['case_study']}/{selected['training_hash']}: n_folds is not declared"
            )
        db_path = get_case_study_dir(selected["case_study"]) / "run_log" / "registry.db"
        uri = f"file:{db_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            db.row_factory = sqlite3.Row
            checkpoint_rows = db.execute(
                """
                SELECT p.prediction_hash, p.checkpoint_value, fm.fold_id, fm.ic
                FROM prediction_sets p
                JOIN fold_metrics fm ON fm.prediction_hash = p.prediction_hash
                WHERE p.training_hash = ? AND p.split = 'validation'
                ORDER BY p.checkpoint_value, fm.fold_id
                """,
                (selected["training_hash"],),
            ).fetchall()
        checkpoints = pl.DataFrame([dict(row) for row in checkpoint_rows], infer_schema_length=None)
        if checkpoints.is_empty():
            raise RegistrySelectionError(
                f"{selected['case_study']}/{selected['training_hash']}: no checkpoint folds"
            )
        expected_folds = set(range(n_folds))
        for checkpoint in checkpoints["checkpoint_value"].unique().sort().to_list():
            current = checkpoints.filter(pl.col("checkpoint_value") == checkpoint)
            fold_ids = current["fold_id"].to_list()
            if len(fold_ids) != len(set(fold_ids)) or set(fold_ids) != expected_folds:
                raise RegistrySelectionError(
                    f"{selected['case_study']}/{selected['training_hash']}/{checkpoint}: "
                    "checkpoint fold coverage is incomplete or duplicate"
                )
            if any(value is None or not math.isfinite(float(value)) for value in current["ic"]):
                raise RegistrySelectionError(
                    f"{selected['case_study']}/{selected['training_hash']}/{checkpoint}: "
                    "checkpoint IC is non-finite"
                )
            for row in current.iter_rows(named=True):
                rows.append(
                    {
                        "case_study": selected["case_study"],
                        "short_name": selected["short_name"],
                        "config_name": selected["config_name"],
                        "training_hash": selected["training_hash"],
                        **row,
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def conformal_coverage_for_selected_prediction(
    selected: dict,
    *,
    levels: tuple[float, ...] = (0.80, 0.90, 0.95),
) -> pl.DataFrame:
    """Measure split-conformal coverage for one exact selected prediction artifact."""
    required = {
        "case_study",
        "family",
        "config_name",
        "prediction_hash",
        "spec_json",
    }
    missing = required - set(selected)
    if missing:
        raise RegistrySelectionError(f"selected row missing conformal fields: {sorted(missing)}")

    spec = json.loads(selected["spec_json"])
    n_folds = int(spec.get("n_folds", 0))
    if n_folds < 2:
        raise RegistrySelectionError(
            f"{selected['case_study']}/{selected['prediction_hash']}: "
            "conformal coverage requires at least two declared folds"
        )

    prediction_path = (
        get_case_study_dir(selected["case_study"])
        / "run_log"
        / "predictions"
        / selected["prediction_hash"]
        / "predictions.parquet"
    )
    if not prediction_path.exists():
        raise RegistrySelectionError(f"missing prediction artifact: {prediction_path}")
    predictions = pl.read_parquet(prediction_path)
    required_columns = {"timestamp", "y_true", "y_score", "fold_id"}
    legacy_columns = {"actual": "y_true", "prediction": "y_score", "fold": "fold_id"}
    present_legacy = {
        legacy: canonical
        for legacy, canonical in legacy_columns.items()
        if legacy in predictions.columns and canonical not in predictions.columns
    }
    if present_legacy:
        predictions = predictions.rename(present_legacy)
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise RegistrySelectionError(
            f"{selected['case_study']}/{selected['prediction_hash']}: "
            f"prediction artifact missing {sorted(missing_columns)}"
        )
    usable = predictions.drop_nulls(required_columns)
    for column in ("y_true", "y_score"):
        usable = usable.filter(pl.col(column).cast(pl.Float64, strict=False).is_finite())
    fold_ids = sorted(usable["fold_id"].unique().to_list())
    if fold_ids != list(range(n_folds)):
        raise RegistrySelectionError(
            f"{selected['case_study']}/{selected['prediction_hash']}: "
            f"expected fold IDs {list(range(n_folds))}, observed {fold_ids}"
        )
    try:
        coverage_rows = split_conformal_coverage(predictions, levels=levels)
    except ValueError as error:
        raise RegistrySelectionError(
            f"{selected['case_study']}/{selected['prediction_hash']}: {error}"
        ) from error
    rows = [
        {
            "case_study": selected["case_study"],
            "family": selected["family"],
            "config_name": selected["config_name"],
            "prediction_hash": selected["prediction_hash"],
            **row,
        }
        for row in coverage_rows
    ]
    return pl.DataFrame(rows)


def collect_grid_per_cs(
    case_studies: Iterable[str],
    family: str,
    label_resolver: LabelResolver | None = None,
    config_parser: Callable[[str], dict] | None = None,
) -> pl.DataFrame:
    """Best checkpoint of every configuration, per case study.

    `collect_rank1_per_cs` keeps one row per case study; this keeps the whole
    grid, so a chapter can show the spread the winner came from.
    """
    rows = []
    for case_study in case_studies:
        label = _resolve_label(case_study, label_resolver)
        metrics, folds, n_folds = _raw_primary_candidates(case_study, family, label)
        if metrics.is_empty() or n_folds <= 0:
            continue
        selected = []
        for config_name in metrics["config_name"].unique().to_list():
            config_metrics = metrics.filter(pl.col("config_name") == config_name)
            try:
                selected.append(
                    select_rank1(config_metrics, folds, expected_fold_ids=range(n_folds))
                )
            except RegistrySelectionError:
                continue
        if not selected:
            continue
        full_days = max(float(row["ic_n_days"]) for row in selected)
        for selected_row in selected:
            if float(selected_row["ic_n_days"]) != full_days:
                continue
            row = {
                **selected_row,
                "case_study": case_study,
                "short_name": SHORT_NAMES.get(case_study, case_study),
            }
            if config_parser is not None:
                row.update(config_parser(row["config_name"]))
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def collect_multi_label_per_cs(
    case_studies: Iterable[str],
    family: str,
    labels: list[str] | Callable[[str], list[str]],
) -> pl.DataFrame:
    """Rank-one configuration per case study and label.

    `labels` is either one list applied to every case study, or a callable
    `case_study -> list[label]` where the label sets differ.
    """
    rows = []
    for case_study in case_studies:
        case_labels = labels(case_study) if callable(labels) else labels
        for label in case_labels:
            metrics, folds, n_folds = _raw_primary_candidates(case_study, family, label)
            if metrics.is_empty():
                continue
            if n_folds <= 0:
                raise RegistrySelectionError(
                    f"{case_study}/{family}/{label}: n_folds is not declared"
                )
            try:
                selected = select_rank1(metrics, folds, expected_fold_ids=range(n_folds))
            except RegistrySelectionError as exc:
                raise RegistrySelectionError(f"{case_study}/{family}/{label}: {exc}") from exc
            selected.update(
                case_study=case_study,
                short_name=SHORT_NAMES.get(case_study, case_study),
            )
            rows.append(selected)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def collect_gbm_checkpoint_trajectories(rank1: pl.DataFrame) -> pl.DataFrame:
    """Learning curve for each selected GBM training run."""
    from case_studies.utils.registry import get_training_dir

    frames = []
    for row in rank1.iter_rows(named=True):
        spec = json.loads(row["spec_json"])
        path = get_training_dir(row["case_study"], spec) / "learning_curves.parquet"
        if not path.exists():
            raise RegistrySelectionError(f"missing learning curve for {row['training_hash']}")
        curve = pl.read_parquet(path).filter(pl.col("config") == row["config_name"])
        if curve.is_empty() or "iteration" not in curve.columns:
            raise RegistrySelectionError(f"incomplete learning curve for {row['training_hash']}")
        trajectory = (
            curve.group_by("iteration")
            .agg(
                pl.col("ic_mean").mean().alias("ic_mean"),
                pl.col("ic_std").mean().alias("ic_std"),
            )
            .sort("iteration")
            .with_columns(
                pl.lit(row["case_study"]).alias("case_study"),
                pl.lit(row["short_name"]).alias("short_name"),
                pl.lit(row["config_name"]).alias("config_name"),
                pl.lit(row["training_hash"]).alias("training_hash"),
            )
        )
        frames.append(trajectory)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def load_gbm_feature_importance(
    case_study: str,
    training_hash: str,
    config_name: str,
    *,
    top_n: int,
) -> pl.DataFrame:
    """Load booster importance from one selected GBM training identity."""
    import lightgbm as lgb

    case_dir = get_case_study_dir(case_study)
    booster_dir = case_dir / "run_log" / "training" / training_hash / "boosters"
    if not booster_dir.exists():
        booster_dir = case_dir / "run_log" / "models" / training_hash / "boosters"
    if not booster_dir.exists():
        return pl.DataFrame()

    rows = []
    for booster_file in sorted(booster_dir.glob("*.txt")):
        fold_text = booster_file.stem.split("fold")[-1].lstrip("_")
        with contextlib.suppress(ValueError):
            fold_id = int(fold_text)
            model = lgb.Booster(model_file=str(booster_file))
            rows.extend(
                {
                    "config_name": config_name,
                    "training_hash": training_hash,
                    "fold_id": fold_id,
                    "feature": feature,
                    "importance": float(importance),
                }
                for feature, importance in zip(
                    model.feature_name(),
                    model.feature_importance(importance_type="gain"),
                    strict=False,
                )
            )
    if not rows:
        return pl.DataFrame()
    result = pl.DataFrame(rows).with_columns(
        fold_max=pl.col("importance").max().over(["config_name", "fold_id"])
    )
    if result.filter(pl.col("fold_max") <= 0).height:
        raise RegistrySelectionError(f"{case_study}/{training_hash}: zero feature importance")
    result = result.with_columns(importance_norm=pl.col("importance") / pl.col("fold_max")).drop(
        "fold_max"
    )
    top_features = (
        result.group_by("feature")
        .agg(pl.col("importance_norm").mean().alias("mean_importance"))
        .sort("mean_importance", descending=True)
        .head(top_n)["feature"]
        .to_list()
    )
    return result.filter(pl.col("feature").is_in(top_features))


def plot_cross_cs_forest(
    df: pl.DataFrame,
    family: str,
    title: str,
    *,
    sort_by: str = "ic_mean_daily",
    figsize: tuple[float, float] | None = None,
    sig_t: float = 2.0,
):
    """Forest of rank-1 daily-pooled IC ± HAC 95% CI per case study.

    Marker style encodes whether the HAC CI excludes zero:
      - filled circle: |t_hac| > sig_t (distinguishable from zero)
      - open circle:  |t_hac| ≤ sig_t (overlaps zero)

    Y-axis order is ascending by `sort_by`, so the largest value sits at top.

    Returns (fig, ax).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from utils.style import COLORS

    if df.is_empty():
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.text(0.5, 0.5, f"No {family} runs in registry", ha="center", va="center")
        ax.set_title(title)
        ax.set_axis_off()
        return fig, ax

    d = df.sort(sort_by, descending=False, nulls_last=False).to_pandas()
    n = len(d)
    if figsize is None:
        figsize = (7.5, max(2.5, 0.45 * n + 1.2))
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)
    ic = d["ic_mean_daily"].to_numpy()
    lo = d["ic_ci_lo"].to_numpy()
    hi = d["ic_ci_hi"].to_numpy()
    t_hac = d["ic_t_hac"].to_numpy() if "ic_t_hac" in d.columns else np.full(n, np.nan)

    ax.errorbar(
        ic,
        y,
        xerr=[ic - lo, hi - ic],
        fmt="none",
        color=COLORS["neutral"],
        lw=1.0,
        capsize=3,
    )
    t_arr = np.asarray(t_hac, dtype=float)
    sig = (np.abs(t_arr) > sig_t) & ~np.isnan(t_arr)
    if sig.any():
        ax.scatter(
            ic[sig],
            y[sig],
            s=60,
            marker="o",
            facecolor=COLORS["blue"],
            edgecolor=COLORS["blue"],
            zorder=3,
            label=f"|t_hac| > {sig_t:g}",
        )
    if (~sig).any():
        ax.scatter(
            ic[~sig],
            y[~sig],
            s=60,
            marker="o",
            facecolor="white",
            edgecolor=COLORS["blue"],
            zorder=3,
            label=f"|t_hac| ≤ {sig_t:g}",
        )

    ax.axvline(0, color=COLORS["neutral"], lw=0.7, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d["short_name"].tolist())
    ax.set_xlabel("Daily-pooled IC (HAC 95% CI)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    return fig, ax


def plot_per_fold_violin(
    fold_df: pl.DataFrame,
    order: list[str],
    *,
    title: str,
    figsize: tuple[float, float] | None = None,
    jitter_color: str = "#3B82F6",
):
    """Box-plus-scatter of per-fold IC across case studies.

    `fold_df` must carry columns `short_name` and `ic`. `order` is the CS
    display order (left → right). CSs absent from `fold_df` are skipped.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    present = [c for c in order if c in fold_df["short_name"].unique().to_list()]
    if not present:
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.text(0.5, 0.5, "No fold IC data", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    if figsize is None:
        figsize = (max(6.5, 1.0 * len(present) + 2), 4.5)
    fig, ax = plt.subplots(figsize=figsize)
    data = [fold_df.filter(pl.col("short_name") == cs)["ic"].to_numpy() for cs in present]
    positions = np.arange(len(present))
    ax.boxplot(data, positions=positions, widths=0.55, showfliers=True)
    for i, arr in enumerate(data):
        if len(arr):
            ax.scatter(np.full(len(arr), i), arr, alpha=0.5, s=14, color=jitter_color)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(present, rotation=30, ha="right")
    ax.set_ylabel("Per-fold Spearman IC")
    ax.set_title(title)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    return fig, ax


def _rank1_full_coverage_hash(case_study: str, label: str) -> str | None:
    """Validation prediction_hash of the highest daily-IC linear config with NO dropped fold.

    Selection is coverage-aware: a configuration whose path zeroes out on some
    fold leaves that fold's IC NULL and pools its daily IC over fewer days,
    which can make a naive ``MAX(ic_mean_daily)`` crown a config scored on a
    non-comparable subset. We therefore rank only among configs whose
    fold_metrics carry no NULL IC. Returns None if the registry has no
    full-coverage linear row for that label.
    """
    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            """
            SELECT p.prediction_hash, pm.ic_mean_daily, pm.ic_n_days,
                   (SELECT COUNT(*) FROM fold_metrics fm
                    WHERE fm.prediction_hash = p.prediction_hash AND fm.ic IS NULL) AS n_null
            FROM training_runs t
            JOIN prediction_sets p ON p.training_hash = t.training_hash AND p.split = 'validation'
            JOIN prediction_metrics pm ON pm.prediction_hash = p.prediction_hash
            WHERE t.family = 'linear' AND t.label = ? AND pm.ic_mean_daily IS NOT NULL
            """,
            (label,),
        ).fetchall()
    full = [r for r in rows if r[2] is not None and r[3] == 0]
    if not full:
        return None
    max_n_days = max(r[2] for r in full)
    full = [r for r in full if r[2] == max_n_days]
    full.sort(key=lambda r: -r[1])
    return full[0][0]


def plot_rolling_daily_ic(
    case_studies: Iterable[str],
    *,
    window: int = 63,
    label_resolver: LabelResolver | None = None,
    common_window: bool = True,
    title: str = "Persistence of linear ranking signal (rolling daily IC)",
    figsize: tuple[float, float] = (10, 4),
    colors: list[str] | None = None,
    selected_prediction_hashes: dict[str, str] | None = None,
):
    """Rolling-mean daily-IC persistence chart for the rank-1 linear fit per case study.

    For each case study, take the coverage-aware rank-1 linear configuration
    (highest daily IC with no dropped fold), load its per-day IC series from
    ``daily_metrics.parquet``, average across overlapping folds per calendar
    day, and plot a ``window``-day rolling mean (63 ≈ three trading months).

    Case studies cover different validation windows, so they cannot share a
    calendar axis unless their periods overlap. With ``common_window=True``
    the series are clipped to the intersection of all input case studies'
    spans — pass only case studies whose windows overlap (e.g. ``etfs`` and
    ``fx_pairs`` over 2016–2023). Case studies without a daily-IC series are
    skipped.

    Returns (fig, ax).
    """
    import matplotlib.pyplot as plt

    from case_studies.utils.model_analysis import load_daily_metrics_series

    if colors is None:
        from utils.style import COLORS

        colors = [COLORS["blue"], COLORS["amber"], COLORS["copper"], COLORS["slate"]]

    series: dict[str, pl.DataFrame] = {}
    for cs in case_studies:
        label = _resolve_label(cs, label_resolver)
        h = (
            selected_prediction_hashes.get(cs)
            if selected_prediction_hashes is not None
            else _rank1_full_coverage_hash(cs, label)
        )
        if h is None:
            continue
        d = load_daily_metrics_series(cs, h)
        if d.is_empty() or "date" not in d.columns:
            continue
        roll = (
            d.group_by("date")
            .agg(pl.col("ic").mean())
            .sort("date")
            .with_columns(
                pl.col("ic").rolling_mean(window_size=window, min_periods=window).alias("roll")
            )
            .drop_nulls("roll")
        )
        if not roll.is_empty():
            series[cs] = roll

    fig, ax = plt.subplots(figsize=figsize)
    if not series:
        ax.text(0.5, 0.5, "No daily-IC series available", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    lo = max(s["date"].min() for s in series.values()) if common_window else None
    hi = min(s["date"].max() for s in series.values()) if common_window else None
    for idx, (cs, roll) in enumerate(series.items()):
        sub = roll
        if common_window:
            sub = roll.filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
        ax.plot(
            sub["date"].to_numpy(),
            sub["roll"].to_numpy(),
            color=colors[idx % len(colors)],
            linewidth=1.6,
            label=SHORT_NAMES.get(cs, cs),
        )
    from utils.style import COLORS

    ax.axhline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"{window}-day rolling daily IC")
    ax.set_xlabel("Validation date")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    return fig, ax


# Horizon → trading-day mapping shared by all chapter insight notebooks.
HORIZON_DAYS: dict[str, float] = {
    "fwd_ret_5m": 5 / (6.5 * 60),
    "fwd_ret_15m": 15 / (6.5 * 60),
    "fwd_ret_60m": 60 / (6.5 * 60),
    "fwd_ret_8h": 1.0 / 3,
    "fwd_ret_24h": 1.0,
    "fwd_ret_1d": 1.0,
    "fwd_ret_5d": 5.0,
    "fwd_ret_10d": 10.0,
    "fwd_ret_21d": 21.0,
    "fwd_ret_1m": 21.0,
    "fwd_ret_3m": 63.0,
    "fwd_ret_1m_win": 21.0,
    "fwd_ret_risk_adj_5d": 5.0,
}


def plot_multi_label_horizon(
    horizon_df: pl.DataFrame,
    *,
    title: str,
    min_labels_per_cs: int = 2,
    figsize: tuple[float, float] = (10, 5),
    palette: list[str] | None = None,
):
    """Faceted horizon plot: daily-pooled IC vs trading-day horizon per CS.

    `horizon_df` must carry `short_name`, `label`, `ic_mean_daily`, `ic_ci_lo`,
    `ic_ci_hi`. CSs with fewer than `min_labels_per_cs` mapped horizons are
    omitted from the figure (a coverage fact, not a defect).
    """
    import matplotlib.pyplot as plt

    plot_df = horizon_df.with_columns(
        horizon_days=pl.col("label").replace_strict(HORIZON_DAYS, default=None).cast(pl.Float64),
    ).filter(pl.col("horizon_days").is_not_null())
    multi_cs = (
        plot_df.group_by("short_name")
        .len()
        .filter(pl.col("len") >= min_labels_per_cs)["short_name"]
        .to_list()
    )
    plot_df = plot_df.filter(pl.col("short_name").is_in(multi_cs))
    if plot_df.is_empty():
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.text(0.5, 0.5, "No multi-horizon coverage", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)
    if palette is None:
        from utils.style import COLORS

        palette = [
            COLORS["blue"],
            COLORS["amber"],
            COLORS["copper"],
            COLORS["positive"],
            COLORS["negative"],
            COLORS["neutral"],
            COLORS["slate"],
            COLORS["amber_light"],
        ]
    cs_sorted = sorted(plot_df["short_name"].unique().to_list())
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    for idx, cs in enumerate(cs_sorted):
        sub = plot_df.filter(pl.col("short_name") == cs).sort("horizon_days")
        if sub.height < 2:
            continue
        x = sub["horizon_days"].to_numpy()
        ic = sub["ic_mean_daily"].to_numpy()
        lo = sub["ic_ci_lo"].to_numpy()
        hi = sub["ic_ci_hi"].to_numpy()
        color = palette[idx % len(palette)]
        ax.fill_between(x, lo, hi, color=color, alpha=0.12)
        ax.plot(
            x,
            ic,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            color=color,
            label=cs,
            linewidth=1.6,
            markersize=6,
            alpha=0.9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Horizon (trading days, log scale)")
    ax.set_ylabel("Daily-pooled IC (HAC 95 % CI band)")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=8, ncol=2)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    return fig, ax


def parse_gbm_config(config: str) -> dict:
    """Decode a GBM `config_name` into profile / loss / leaves / objective_kind.

    Examples
    --------
    >>> parse_gbm_config("leaves_31_huber")
    {"profile": "leaves_31", "loss": "huber", "leaves": 31, "objective_kind": "regression"}
    >>> parse_gbm_config("default_binary")
    {"profile": "default", "loss": "binary", "leaves": None, "objective_kind": "classification"}
    """
    out = {"profile": config, "loss": "unknown", "leaves": None, "objective_kind": "regression"}
    parts = config.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("mse", "mae", "huber"):
        out["profile"], out["loss"] = parts
    elif config.endswith("_binary"):
        out["loss"] = "binary"
        out["objective_kind"] = "classification"
        out["profile"] = config.removesuffix("_binary")
    if "leaves_" in out["profile"]:
        with contextlib.suppress(ValueError, IndexError):
            out["leaves"] = int(out["profile"].split("_")[1])
    return out
