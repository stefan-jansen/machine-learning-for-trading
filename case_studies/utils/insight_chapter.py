"""Cross-case-study aggregation for chapter insight notebooks (Ch11–Ch15).

Wraps the per-case-study spine-v2 helpers in `model_analysis` with a thin
"collect across N case studies" layer plus a forest plotter.

Usage::

    from case_studies.utils.insight_chapter import (
        collect_rank1_per_cs,
        collect_fold_ic_per_cs,
        collect_multi_label_per_cs,
        collect_grid_per_cs,
        collect_gbm_checkpoint_trajectories,
        parse_gbm_config,
        plot_cross_cs_forest,
    )
    from case_studies.utils.analytics import CASE_STUDY_IDS

    rank1 = collect_rank1_per_cs(CASE_STUDY_IDS, family="linear")
    plot_cross_cs_forest(rank1, family="linear",
                         title="Linear: rank-1 per case study (primary label)")
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Callable, Iterable

import polars as pl

# ml4t.diagnostic dlopens cudart; load torch first so its bundled CUDA
# runtime wins. Same pattern as case_studies/utils/model_analysis.py.
import torch  # noqa: F401

from case_studies.utils.analytics import PRIMARY_LABELS, SHORT_NAMES
from case_studies.utils.model_analysis import (
    load_fold_metrics_from_registry,
    load_metrics_from_registry,
)
from utils.paths import get_case_study_dir

LabelResolver = Callable[[str], str | None]

# Canonical schema for collect_rank1_per_cs() — returned as an empty
# DataFrame when no CS has populated registry rows. Lets downstream
# `.select("short_name", ...)` callers see "(no rows)" instead of a
# cryptic ColumnNotFoundError.
_RANK1_SCHEMA: dict[str, pl.DataType] = {
    "case_study": pl.Utf8,
    "short_name": pl.Utf8,
    "family": pl.Utf8,
    "label": pl.Utf8,
    "config_name": pl.Utf8,
    "checkpoint_value": pl.Float64,
    "checkpoint_kind": pl.Utf8,
    "ic_mean": pl.Float64,
    "ic_std": pl.Float64,
    "ic_mean_daily": pl.Float64,
    "ic_se_hac": pl.Float64,
    "ic_ci_lo": pl.Float64,
    "ic_ci_hi": pl.Float64,
    "ic_t_hac": pl.Float64,
    "ic_p_hac": pl.Float64,
    "ic_n_days": pl.Int64,
    "ic_hac_lag": pl.Int64,
    "ic_boot_lo": pl.Float64,
    "ic_boot_hi": pl.Float64,
}


def _resolve_label(case_study: str, label_resolver: LabelResolver | None) -> str:
    if label_resolver is None:
        return PRIMARY_LABELS[case_study]
    out = label_resolver(case_study)
    return out if out is not None else PRIMARY_LABELS[case_study]


def collect_rank1_per_cs(
    case_studies: Iterable[str],
    family: str,
    label_resolver: LabelResolver | None = None,
) -> pl.DataFrame:
    """Rank-1 (config, checkpoint) per case study by daily-pooled IC.

    For each CS, query the registry for `(family, label)` rows where
    `label = label_resolver(cs) or PRIMARY_LABELS[cs]`, keep rows with
    `ic_mean_daily` populated, and return the highest-IC row.

    Returns columns: case_study, short_name, family, label, config_name,
    checkpoint_value, checkpoint_kind, ic_mean, ic_std, ic_mean_daily,
    ic_se_hac, ic_ci_lo, ic_ci_hi, ic_t_hac, ic_p_hac, ic_n_days,
    ic_hac_lag, ic_boot_lo, ic_boot_hi.
    """
    frames = []
    for cs in case_studies:
        label = _resolve_label(cs, label_resolver)
        df = load_metrics_from_registry(cs, label=label, families=[family])
        if df.is_empty():
            continue
        df = df.filter(pl.col("ic_mean_daily").is_not_null())
        if df.is_empty():
            continue
        best = (
            df.sort("ic_mean_daily", descending=True)
            .head(1)
            .with_columns(
                pl.lit(cs).alias("case_study"),
                pl.lit(SHORT_NAMES.get(cs, cs)).alias("short_name"),
            )
        )
        frames.append(best)
    if not frames:
        return pl.DataFrame(schema=_RANK1_SCHEMA)
    out = pl.concat(frames, how="diagonal_relaxed")
    front = ["case_study", "short_name", "family", "label", "config_name"]
    rest = [c for c in out.columns if c not in front]
    return out.select(front + rest)


def collect_fold_ic_per_cs(
    case_studies: Iterable[str],
    family: str,
    label_resolver: LabelResolver | None = None,
) -> pl.DataFrame:
    """Per-fold IC for the rank-1 (config, checkpoint) per CS.

    Uses :func:`collect_rank1_per_cs` to identify the rank-1 row per CS, then
    pulls fold-level IC for that exact (config, checkpoint) from
    `fold_metrics`. Linear models without a checkpoint use a null-safe match.

    Returns columns: case_study, short_name, family, config_name, label,
    fold_id, ic, ic_std, n_entities, rmse, mae.
    """
    rank1 = collect_rank1_per_cs(case_studies, family, label_resolver)
    if rank1.is_empty():
        return pl.DataFrame()

    frames = []
    for row in rank1.iter_rows(named=True):
        cs = row["case_study"]
        folds = load_fold_metrics_from_registry(cs, label=row["label"], families=[family])
        if folds.is_empty():
            continue
        cp = row["checkpoint_value"]
        cond = pl.col("config_name") == row["config_name"]
        cond = cond & (
            pl.col("checkpoint_value").is_null()
            if cp is None
            else (pl.col("checkpoint_value") == cp)
        )
        f = folds.filter(cond)
        if f.is_empty():
            continue
        f = f.with_columns(
            pl.lit(cs).alias("case_study"),
            pl.lit(SHORT_NAMES.get(cs, cs)).alias("short_name"),
        )
        frames.append(
            f.select(
                "case_study",
                "short_name",
                "family",
                "config_name",
                "label",
                "fold_id",
                "ic",
                "ic_std",
                "n_entities",
                "rmse",
                "mae",
            )
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def collect_multi_label_per_cs(
    case_studies: Iterable[str],
    family: str,
    labels: list[str] | Callable[[str], list[str]],
) -> pl.DataFrame:
    """Rank-1 (config, checkpoint) per (CS, label) by daily-pooled IC.

    `labels` is either a fixed label list applied to every CS or a callable
    `cs -> list[label]` for CS-specific label sets. Missing (CS, label) pairs
    are silently skipped — coverage gaps surface as absent rows.

    Returns columns: case_study, short_name, family, label, config_name,
    checkpoint_value, checkpoint_kind, ic_mean_daily, ic_ci_lo, ic_ci_hi,
    ic_t_hac, ic_n_days.
    """
    frames = []
    for cs in case_studies:
        cs_labels = labels(cs) if callable(labels) else labels
        for lbl in cs_labels:
            df = load_metrics_from_registry(cs, label=lbl, families=[family])
            if df.is_empty():
                continue
            df = df.filter(pl.col("ic_mean_daily").is_not_null())
            if df.is_empty():
                continue
            best = (
                df.sort("ic_mean_daily", descending=True)
                .head(1)
                .with_columns(
                    pl.lit(cs).alias("case_study"),
                    pl.lit(SHORT_NAMES.get(cs, cs)).alias("short_name"),
                )
            )
            frames.append(best)
    if not frames:
        return pl.DataFrame()
    out = pl.concat(frames, how="diagonal_relaxed")
    keep = [
        "case_study",
        "short_name",
        "family",
        "label",
        "config_name",
        "checkpoint_value",
        "checkpoint_kind",
        "ic_mean_daily",
        "ic_ci_lo",
        "ic_ci_hi",
        "ic_t_hac",
        "ic_n_days",
    ]
    return out.select([c for c in keep if c in out.columns])


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
        color="#444",
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
            facecolor="#1f77b4",
            edgecolor="#1f77b4",
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
            edgecolor="#1f77b4",
            zorder=3,
            label=f"|t_hac| ≤ {sig_t:g}",
        )

    ax.axvline(0, color="#888", lw=0.7, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d["short_name"].tolist())
    ax.set_xlabel("Daily-pooled IC (HAC 95% CI)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
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
            SELECT p.prediction_hash, pm.ic_mean_daily,
                   (SELECT COUNT(*) FROM fold_metrics fm
                    WHERE fm.prediction_hash = p.prediction_hash AND fm.ic IS NULL) AS n_null
            FROM training_runs t
            JOIN prediction_sets p ON p.training_hash = t.training_hash AND p.split = 'validation'
            JOIN prediction_metrics pm ON pm.prediction_hash = p.prediction_hash
            WHERE t.family = 'linear' AND t.label = ? AND pm.ic_mean_daily IS NOT NULL
            """,
            (label,),
        ).fetchall()
    full = [r for r in rows if r[2] == 0]
    if not full:
        return None
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
        h = _rank1_full_coverage_hash(cs, label)
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
    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"{window}-day rolling daily IC")
    ax.set_xlabel("Validation date")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
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


def collect_grid_per_cs(
    case_studies: Iterable[str],
    family: str,
    label_resolver: LabelResolver | None = None,
    config_parser: Callable[[str], dict] | None = None,
) -> pl.DataFrame:
    """Per-(CS, config) rank-1 IC by daily-pooled IC, primary label only.

    For each CS, loads the family metrics on the resolved label, then groups
    by `config_name` and keeps the highest-IC row per config. If
    `config_parser` is supplied (e.g. :func:`parse_gbm_config` for GBM), its
    keys are merged onto each output row as flat columns.

    Returns a long-form frame with at minimum:
    `case_study, short_name, config_name, ic_mean_daily, ic_ci_lo, ic_ci_hi,
    ic_t_hac, checkpoint_value` plus any keys produced by `config_parser`.

    Built explicitly with `schema_overrides` so that mixed `None`/`int` columns
    (e.g. `checkpoint_value`, `leaves`) don't trip polars' first-row schema
    inference.
    """
    rows = []
    for cs in case_studies:
        label = _resolve_label(cs, label_resolver)
        df = load_metrics_from_registry(cs, label=label, families=[family])
        if df.is_empty():
            continue
        df = df.filter(pl.col("ic_mean_daily").is_not_null())
        if df.is_empty():
            continue
        best = (
            df.sort("ic_mean_daily", descending=True, nulls_last=True)
            .group_by("config_name")
            .first()
        )
        for r in best.iter_rows(named=True):
            row = {
                "case_study": cs,
                "short_name": SHORT_NAMES.get(cs, cs),
                "config_name": r["config_name"],
                "ic_mean_daily": r["ic_mean_daily"],
                "ic_ci_lo": r["ic_ci_lo"],
                "ic_ci_hi": r["ic_ci_hi"],
                "ic_t_hac": r["ic_t_hac"],
                "checkpoint_value": r["checkpoint_value"],
            }
            if config_parser is not None:
                row.update(config_parser(r["config_name"]))
            rows.append(row)
    overrides: dict[str, pl.DataType] = {
        "checkpoint_value": pl.Int64,
        "leaves": pl.Int64,
    }
    if not rows:
        return pl.DataFrame(
            schema={
                "case_study": pl.Utf8,
                "short_name": pl.Utf8,
                "config_name": pl.Utf8,
                "ic_mean_daily": pl.Float64,
                "ic_ci_lo": pl.Float64,
                "ic_ci_hi": pl.Float64,
                "ic_t_hac": pl.Float64,
                "checkpoint_value": pl.Int64,
            }
        )
    return pl.DataFrame(rows, schema_overrides=overrides)


def collect_gbm_checkpoint_trajectories(
    case_studies: Iterable[str],
    label_resolver: LabelResolver | None = None,
) -> pl.DataFrame:
    """Per-checkpoint IC trajectory for each case study's rank-1 GBM config.

    The boosting runner records mean cross-sectional IC at every
    ``checkpoint_interval`` (default 50 trees) up to ``n_trees`` and writes
    the result to ``learning_curves.parquet`` in the training directory. The
    `prediction_metrics` table only stores the early-stopped final IC, so
    trajectories must be loaded from these parquet files directly.

    For each CS, this helper:

    1. Finds the rank-1 GBM `(config, training_hash)` on the resolved label
       by daily-pooled IC.
    2. Reads `learning_curves.parquet` from that training run's directory.
    3. Returns a tidy long frame keyed by `(case_study, short_name,
       config_name, iteration, ic_mean, ic_std)`.

    CSes whose rank-1 lacks a learning-curve file are silently skipped.
    """
    from case_studies.utils.registry import get_training_dir

    frames = []
    for cs in case_studies:
        label = _resolve_label(cs, label_resolver)
        db_path = get_case_study_dir(cs) / "run_log" / "registry.db"
        if not db_path.exists():
            continue
        with sqlite3.connect(db_path) as db:
            cur = db.cursor()
            cur.execute(
                """
                SELECT tr.config_name, tr.spec_json
                FROM prediction_sets ps
                JOIN prediction_metrics pm ON pm.prediction_hash = ps.prediction_hash
                JOIN training_runs tr ON tr.training_hash = ps.training_hash
                WHERE tr.family = 'gbm'
                  AND ps.split = 'validation'
                  AND tr.label = ?
                  AND pm.ic_mean_daily IS NOT NULL
                ORDER BY pm.ic_mean_daily DESC
                LIMIT 1
                """,
                (label,),
            )
            row = cur.fetchone()
        if row is None:
            continue
        cfg, spec_json = row
        spec = json.loads(spec_json)
        lc_path = get_training_dir(cs, spec) / "learning_curves.parquet"
        if not lc_path.exists():
            continue
        lc = pl.read_parquet(lc_path)
        if lc.is_empty() or "iteration" not in lc.columns:
            continue
        traj = (
            lc.filter(pl.col("config") == cfg)
            .group_by("iteration")
            .agg(
                pl.col("ic_mean").mean().alias("ic_mean"),
                pl.col("ic_std").mean().alias("ic_std"),
            )
            .sort("iteration")
            .with_columns(
                pl.lit(cs).alias("case_study"),
                pl.lit(SHORT_NAMES.get(cs, cs)).alias("short_name"),
                pl.lit(cfg).alias("config_name"),
            )
            .select("case_study", "short_name", "config_name", "iteration", "ic_mean", "ic_std")
        )
        if not traj.is_empty():
            frames.append(traj)
    if not frames:
        return pl.DataFrame(
            schema={
                "case_study": pl.Utf8,
                "short_name": pl.Utf8,
                "config_name": pl.Utf8,
                "iteration": pl.Int64,
                "ic_mean": pl.Float64,
                "ic_std": pl.Float64,
            }
        )
    return pl.concat(frames)
