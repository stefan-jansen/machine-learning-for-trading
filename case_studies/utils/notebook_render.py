"""Rendering helpers for per-CS ``*_model_analysis.py`` and chapter insight notebooks.

Composes with :mod:`case_studies.utils.analytics` (registry queries) to produce
uncertainty-aware tables and figures. All helpers return either polars
DataFrames or matplotlib Figures.

Tables (return polars):
    - :func:`holdout_decay_table` — val IC ± CI vs holdout IC ± CI per family
    - :func:`selection_adjusted_leader_table` — DSR / PBO / RC / k-variant per leader

Figures (return matplotlib Figure):
    - :func:`headline_forest_plot` — forest plot of IC ± CI sorted by point estimate
    - :func:`fold_heatmap_with_ci` — fold × family IC heatmap with significance shading

Axes overlays (mutate caller-supplied ax):
    - :func:`regime_coverage_strip` — color-coded strip below fold-IC distribution

Classification-aware helpers (filled at Phase C start):
    - :func:`classification_triple` — AUC ± CI, accuracy ± CI, IC-on-continuous ± CI
    - :func:`cross_task_matrix` — {regression model, classification model} × {IC, AUC} 2×2
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from case_studies.utils.analytics import (
    PRIMARY_LABELS,
    SHORT_NAMES,
    _query,
    _registry_path,
)
from case_studies.utils.notebook_contracts import degenerate_prediction_sql
from utils.style import COLORS

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def holdout_decay_table(
    case_study: str,
    *,
    label: str | None = None,
    families: list[str] | None = None,
) -> pl.DataFrame:
    """Validation vs holdout IC with 95% CIs and decay for the rank-1 leader.

    By design (Ch16 selection workflow) each case study has at most one
    holdout-retrained model — the signal-stage rank-1 leader. This table
    reports its val→holdout decay with a row per family: families that were
    not selected at rank-1 show null holdout columns.

    Always loads fresh from ``prediction_sets`` — no pre-computed artifact.
    Prefers ``ic_mean_daily`` + HAC CI; falls back to legacy ``ic_mean`` where
    the daily-pooled backfill hasn't run (currently: all holdout splits).

    Returns columns:
        family, config_name, label,
        val_ic, val_ci_lo, val_ci_hi, val_ic_source,
        ho_ic,  ho_ci_lo,  ho_ci_hi,  ho_ic_source,
        decay_pp, decay_pct
    """
    label = label or PRIMARY_LABELS[case_study]
    db = _registry_path(case_study)
    if not db.exists():
        return pl.DataFrame()

    family_clause = ""
    params: list = [label]
    if families:
        placeholders = ",".join("?" * len(families))
        family_clause = f"AND t.family IN ({placeholders})"
        params.extend(families)

    # Prefer daily-pooled IC + HAC CI; fall back to legacy ic_mean where the
    # daily-pooled backfill hasn't run (notably: holdout splits, as of 2026-04-30).
    sql = f"""
        SELECT
            t.family,
            t.config_name,
            t.label,
            p.split,
            COALESCE(pm.ic_mean_daily, pm.ic_mean) AS ic,
            pm.ic_ci_lo,
            pm.ic_ci_hi,
            pm.ic_n_days,
            CASE WHEN pm.ic_mean_daily IS NOT NULL THEN 'daily_hac' ELSE 'fold_mean' END AS ic_source
        FROM training_runs t
        JOIN prediction_sets p ON t.training_hash = p.training_hash
        JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
        WHERE t.label = ?
          AND p.split IN ('validation', 'holdout')
          {family_clause}
          AND COALESCE(pm.ic_mean_daily, pm.ic_mean) IS NOT NULL
    """
    rows = _query(db, sql, tuple(params))
    if rows.is_empty():
        return pl.DataFrame()

    # Holdout retrains are at most one per family (the signal-stage rank-1
    # leader). For those families the row's config_name and val_ic must come
    # from the SAME config that was retrained — not the validation IC rank-1
    # config, which need not be the same model. Joining only on `family`
    # would mix two different configs into a single row (e.g. an ETF case
    # where validation IC rank-1 = nlinear and the holdout retrain target =
    # lstm_h64) and mis-attribute lstm_h64's holdout IC to nlinear.
    ho_leaders = (
        rows.filter(pl.col("split") == "holdout")
        .sort("ic", descending=True, nulls_last=True)
        .group_by("family")
        .first()
        .select(
            "family",
            "config_name",
            pl.col("ic").alias("ho_ic"),
            pl.col("ic_ci_lo").alias("ho_ci_lo"),
            pl.col("ic_ci_hi").alias("ho_ci_hi"),
            pl.col("ic_source").alias("ho_ic_source"),
        )
    )

    # For families with a holdout retrain, pull val IC for the same (family, config).
    # Multiple validation predictions can exist for one config (e.g. DL checkpoints);
    # take the highest-IC one to mirror the validation rank-1 selection logic.
    val_rows = rows.filter(pl.col("split") == "validation")
    val_for_ho = (
        val_rows.join(
            ho_leaders.select("family", "config_name"),
            on=["family", "config_name"],
            how="inner",
        )
        .sort("ic", descending=True, nulls_last=True)
        .group_by(["family", "config_name"])
        .first()
        .select(
            "family",
            "config_name",
            "label",
            pl.col("ic").alias("val_ic"),
            pl.col("ic_ci_lo").alias("val_ci_lo"),
            pl.col("ic_ci_hi").alias("val_ci_hi"),
            pl.col("ic_source").alias("val_ic_source"),
        )
    )

    # Families without a holdout retrain still surface their validation IC
    # rank-1 leader; ho_* columns will be null after the left-join below.
    ho_families = ho_leaders.select("family")
    val_no_ho = (
        val_rows.join(ho_families, on="family", how="anti")
        .sort("ic", descending=True, nulls_last=True)
        .group_by("family")
        .first()
        .select(
            "family",
            "config_name",
            "label",
            pl.col("ic").alias("val_ic"),
            pl.col("ic_ci_lo").alias("val_ci_lo"),
            pl.col("ic_ci_hi").alias("val_ci_hi"),
            pl.col("ic_source").alias("val_ic_source"),
        )
    )

    val = pl.concat([val_for_ho, val_no_ho])
    out = val.join(ho_leaders.drop("config_name"), on="family", how="left")
    out = out.with_columns(
        decay_pp=(pl.col("ho_ic") - pl.col("val_ic")),
        decay_pct=pl.when(pl.col("val_ic").abs() > 0)
        .then((pl.col("ho_ic") - pl.col("val_ic")) / pl.col("val_ic").abs() * 100.0)
        .otherwise(None),
    )
    return out.sort("val_ic", descending=True, nulls_last=True)


def selection_adjusted_leader_table(
    case_study: str,
    *,
    stage: str = "signal",
    label: str | None = None,
) -> pl.DataFrame:
    """Per-family rank-1 backtest with selection-adjusted statistics.

    Joins ``backtest_metrics`` with ``training_runs`` and LEFT JOINs the
    persisted ``cohort_metrics`` (cohort_type='family') for the
    leader-hash. The legacy column names (``dsr``, ``dsr_pvalue``,
    ``expected_max_sharpe``) carry the **effective-rank (ER) DSR** — the
    library maintainer's recommended default. ``dsr_mp`` and ``dsr_raw``
    are surfaced alongside for sensitivity. Non-leader family rows have
    NULL selection-bias columns.

    Returns columns:
        family, config_name, label,
        sharpe, sharpe_ci95_lo, sharpe_ci95_hi,
        psr_pvalue, dsr, dsr_pvalue, expected_max_sharpe,
        dsr_mp, dsr_mp_pvalue, dsr_raw, dsr_raw_pvalue,
        n_trials_effective_er, n_trials_effective_mp,
        ras_leader, ras_pvalue,
        reality_check_pvalue, pbo, k_variants
    """
    db = _registry_path(case_study)
    if not db.exists():
        return pl.DataFrame()

    label_clause = ""
    params: list = [stage]
    if label:
        label_clause = "AND t.label = ?"
        params.append(label)

    sql = f"""
        SELECT
            t.family,
            t.config_name,
            t.label,
            bm.sharpe,
            bm.sharpe_ci95_lo,
            bm.sharpe_ci95_hi,
            bm.psr_pvalue,
            cm.dsr_er                 AS dsr,
            cm.dsr_er_pvalue          AS dsr_pvalue,
            cm.expected_max_sharpe_er AS expected_max_sharpe,
            cm.dsr_mp,
            cm.dsr_mp_pvalue,
            cm.dsr_raw,
            cm.dsr_raw_pvalue,
            cm.n_trials_effective_er,
            cm.n_trials_effective_mp,
            cm.ras_leader,
            cm.ras_pvalue,
            cm.reality_check_pvalue,
            cm.pbo,
            cm.k_variants
        FROM backtest_runs b
        JOIN backtest_metrics bm ON b.backtest_hash = bm.backtest_hash
        JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
        JOIN training_runs t ON p.training_hash = t.training_hash
        LEFT JOIN cohort_metrics cm
               ON cm.cohort_type = 'family'
              AND cm.stage = b.stage
              AND cm.label = t.label
              AND cm.family = t.family
              AND cm.leader_hash = b.backtest_hash
        WHERE b.stage = ?
          {label_clause}
          AND bm.sharpe IS NOT NULL
          {degenerate_prediction_sql("p.prediction_hash")}
    """
    rows = _query(db, sql, tuple(params))
    if rows.is_empty():
        return pl.DataFrame()

    # Force Float64 dtype on numeric columns that can come back as all-NULL
    # under the LEFT JOIN (polars infers Null dtype otherwise, which breaks
    # downstream ``.round()`` calls).
    float_cols = [
        "dsr",
        "dsr_pvalue",
        "expected_max_sharpe",
        "dsr_mp",
        "dsr_mp_pvalue",
        "dsr_raw",
        "dsr_raw_pvalue",
        "n_trials_effective_er",
        "n_trials_effective_mp",
        "ras_leader",
        "ras_pvalue",
        "reality_check_pvalue",
        "pbo",
    ]
    casts = [pl.col(c).cast(pl.Float64) for c in float_cols if c in rows.columns]
    if casts:
        rows = rows.with_columns(casts)

    leaders = rows.sort("sharpe", descending=True, nulls_last=True).group_by("family").first()
    return leaders.sort("sharpe", descending=True, nulls_last=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def headline_forest_plot(
    df: pl.DataFrame,
    *,
    ic_col: str = "ic_mean_daily",
    ci_lo_col: str = "ic_ci_lo",
    ci_hi_col: str = "ic_ci_hi",
    label_col: str = "config_name",
    family_col: str = "family",
    task_type_col: str | None = None,
    title: str = "",
    figsize: tuple[float, float] = (8.0, None),  # type: ignore[assignment]
) -> Figure:
    """Forest plot of point estimates with 95% CIs, sorted by point estimate.

    CI bars that include zero are drawn in muted gray; non-zero-crossing CIs
    use the family color from :data:`utils.style.COLORS`. The zero line is
    drawn as a dashed reference.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain ``ic_col``, ``ci_lo_col``, ``ci_hi_col``, ``label_col``,
        ``family_col``. Optional ``task_type_col`` adds a task-type tag.
    """
    required = {ic_col, ci_lo_col, ci_hi_col, label_col, family_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"forest plot missing columns: {missing}")

    sorted_df = df.sort(ic_col, descending=False, nulls_last=True).drop_nulls(ic_col)
    n = sorted_df.height
    if n == 0:
        raise ValueError("forest plot received empty (or fully-null) data")

    height = figsize[1] if figsize[1] is not None else max(2.5, 0.32 * n + 1.0)
    fig, ax = plt.subplots(figsize=(figsize[0], height), constrained_layout=True)

    ic = sorted_df[ic_col].to_numpy()
    lo = sorted_df[ci_lo_col].to_numpy()
    hi = sorted_df[ci_hi_col].to_numpy()
    families = sorted_df[family_col].to_list()
    labels = sorted_df[label_col].to_list()
    if task_type_col and task_type_col in sorted_df.columns:
        task_types = sorted_df[task_type_col].to_list()
        labels = [f"{lbl}  [{tt}]" if tt else lbl for lbl, tt in zip(labels, task_types)]

    family_palette = {
        "linear": COLORS.get("blue", "C0"),
        "gbm": COLORS.get("orange", "C1"),
        "deep_learning": COLORS.get("green", "C2"),
        "tabular_dl": COLORS.get("purple", "C3"),
        "latent_factors": COLORS.get("red", "C4"),
        "causal": COLORS.get("brown", "C5"),
        "benchmark": COLORS.get("gray", "C7"),
    }

    y_positions = np.arange(n)
    for i, (point, lo_i, hi_i, fam) in enumerate(zip(ic, lo, hi, families)):
        # numpy float arrays carry NaN for nulls — np.isfinite catches both
        # None-cast-to-NaN and explicit NaNs; bare ``is not None`` would
        # always be True after ``to_numpy()`` and let NaN values reach plot.
        ci_valid = bool(np.isfinite(lo_i) and np.isfinite(hi_i))
        crosses_zero = ci_valid and lo_i <= 0 <= hi_i
        color = "#999999" if (not ci_valid or crosses_zero) else family_palette.get(fam, "#444444")
        if ci_valid:
            ax.plot([lo_i, hi_i], [i, i], color=color, linewidth=2.0, alpha=0.85)
        if np.isfinite(point):
            ax.plot(point, i, marker="o", color=color, markersize=6, zorder=3)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Information Coefficient (daily-pooled, 95% HAC CI)")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":", alpha=0.3)
    return fig


def fold_heatmap_with_ci(
    case_study: str,
    label: str | None = None,
    *,
    families: list[str] | None = None,
    significance_threshold: float = 0.05,
    title: str = "",
) -> Figure:
    """Heatmap of fold IC × family with cells where p-value > threshold dimmed.

    Pulls from ``fold_metrics`` joined to ``training_runs``. Each (family, fold)
    cell shows the rank-1 config IC for that (family, fold). Cells whose HAC
    t-statistic gives p > ``significance_threshold`` are rendered in gray.

    Returns
    -------
    matplotlib.figure.Figure
    """
    label = label or PRIMARY_LABELS[case_study]
    db = _registry_path(case_study)
    if not db.exists():
        raise FileNotFoundError(f"no registry for {case_study}")

    family_clause = ""
    params: list = [label]
    if families:
        placeholders = ",".join("?" * len(families))
        family_clause = f"AND t.family IN ({placeholders})"
        params.extend(families)

    sql = f"""
        SELECT
            t.family,
            t.config_name,
            fm.fold_id,
            fm.ic,
            fm.ic_std,
            fm.n_entities
        FROM training_runs t
        JOIN prediction_sets p ON t.training_hash = p.training_hash
        JOIN fold_metrics fm ON p.prediction_hash = fm.prediction_hash
        WHERE t.label = ?
          AND p.split = 'validation'
          {family_clause}
          AND fm.ic IS NOT NULL
          {degenerate_prediction_sql("p.prediction_hash")}
    """
    df = _query(db, sql, tuple(params))
    if df.is_empty():
        raise ValueError(f"no fold metrics for {case_study} / {label}")

    leaders = (
        df.group_by(["family", "config_name"])
        .agg(pl.col("ic").mean().alias("avg_ic"))
        .sort("avg_ic", descending=True, nulls_last=True)
        .group_by("family")
        .first()
        .select("family", "config_name")
    )
    df = df.join(leaders, on=["family", "config_name"], how="inner")

    # Approx |t| = |ic| / SE(ic), with SE = ic_std / sqrt(n_entities). Folds with
    # missing ic_std or n_entities fall back to a non-significant cell.
    df = df.with_columns(
        t_approx=pl.when((pl.col("ic_std") > 0) & (pl.col("n_entities") > 0))
        .then(pl.col("ic").abs() / (pl.col("ic_std") / pl.col("n_entities").sqrt()))
        .otherwise(0.0)
    )

    pivot = df.pivot(values="ic", index="family", on="fold_id", aggregate_function="first")
    family_order = pivot["family"].to_list()
    fold_cols = [c for c in pivot.columns if c != "family"]
    fold_cols.sort(key=lambda c: int(c) if str(c).isdigit() else c)
    matrix = pivot.select(fold_cols).to_numpy()

    sig_pivot = df.pivot(
        values="t_approx", index="family", on="fold_id", aggregate_function="first"
    ).select(fold_cols)
    t_matrix = sig_pivot.to_numpy()
    with np.errstate(invalid="ignore"):
        p_matrix = 2.0 * (1.0 - _phi(np.abs(t_matrix)))
    significant = p_matrix <= significance_threshold

    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.5 * len(fold_cols) + 2.0), 0.5 * len(family_order) + 1.5),
        constrained_layout=True,
    )
    vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 0.05
    cmap = plt.get_cmap("RdBu_r")
    for i, fam in enumerate(family_order):
        for j, fold in enumerate(fold_cols):
            value = matrix[i, j]
            sig = significant[i, j] if significant.shape == matrix.shape else False
            if value is None or np.isnan(value):
                continue
            color = cmap(0.5 + 0.5 * (value / vmax) if vmax > 0 else 0.5)
            if not sig:
                color = (0.85, 0.85, 0.85, 1.0)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white"))
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    ax.set_xlim(0, len(fold_cols))
    ax.set_ylim(0, len(family_order))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(fold_cols)) + 0.5)
    ax.set_xticklabels([str(c) for c in fold_cols])
    ax.set_yticks(np.arange(len(family_order)) + 0.5)
    ax.set_yticklabels(family_order)
    ax.set_xlabel("Fold")
    if title:
        ax.set_title(title)
    return fig


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF (kept private; avoid scipy dependency)."""
    # Abramowitz & Stegun 26.2.17 erf approximation; max abs error 1.5e-7
    a1, a2, a3, a4, a5, p = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
        0.3275911,
    )
    sign = np.where(x < 0, -1.0, 1.0)
    ax = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return 0.5 * (1.0 + sign * y)


# ---------------------------------------------------------------------------
# Axes overlays
# ---------------------------------------------------------------------------


def regime_coverage_strip(
    ax: Axes,
    fold_dates: list[tuple[str, str]],
    regime_lookup: dict[str, str],
    *,
    palette: dict[str, str] | None = None,
    strip_height_frac: float = 0.06,
) -> None:
    """Draw a color-coded regime strip below an existing fold-IC distribution.

    Mutates ``ax`` in place. Adds a thin horizontal strip below the existing
    plot area, color-coded by which regime each fold falls into.

    Parameters
    ----------
    ax : matplotlib Axes
        The existing fold-IC axes (e.g. boxplot or strip plot of fold ICs).
    fold_dates : list of (start_iso, end_iso)
        One entry per fold, in the same order as the x-axis.
    regime_lookup : dict[str, str]
        Maps a fold key (e.g. ``"2020-Q1"`` or fold start date) to a regime label.
    palette : dict[str, str], optional
        Maps regime labels to colors. Defaults to a categorical palette.
    """
    n_folds = len(fold_dates)
    if n_folds == 0:
        return

    palette = palette or {
        "calm": COLORS.get("blue", "#1f77b4"),
        "stress": COLORS.get("red", "#d62728"),
        "drift": COLORS.get("orange", "#ff7f0e"),
        "structural_break": COLORS.get("purple", "#9467bd"),
    }

    y0, y1 = ax.get_ylim()
    span = y1 - y0
    strip_top = y0
    strip_bot = y0 - strip_height_frac * span

    for i, (start, end) in enumerate(fold_dates):
        regime = regime_lookup.get(start, regime_lookup.get(end, "calm"))
        color = palette.get(regime, "#cccccc")
        ax.add_patch(
            plt.Rectangle(
                (i + 0.5, strip_bot),
                1.0,
                strip_top - strip_bot,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                clip_on=False,
            )
        )

    ax.set_ylim(strip_bot - 0.02 * span, y1)


# ---------------------------------------------------------------------------
# Classification-aware (Phase C)
# ---------------------------------------------------------------------------


def classification_triple(
    case_study: str,
    classification_label: str,
    regression_label: str,
    *,
    families: list[str] | None = None,
    n_boot: int = 2000,
    block_length: int | None = None,
) -> pl.DataFrame:
    """AUC ± CI, accuracy ± CI, and IC-on-continuous ± CI per family rank-1 config.

    Loads the rank-1 prediction set per family for ``classification_label``,
    pulls AUC and accuracy from ``prediction_metrics``, and computes the IC-vs-
    continuous-return on the matched continuous label.

    Returns columns:
        family, config_name, classification_label, regression_label,
        auc, auc_ci_lo, auc_ci_hi,
        accuracy, accuracy_ci_lo, accuracy_ci_hi,
        ic_continuous, ic_ci_lo, ic_ci_hi

    Status: stub — block-bootstrap CI for AUC/accuracy not yet wired.
    Implemented at the start of Phase C (us_firm / sp500_eo / crypto).
    """
    raise NotImplementedError(
        "classification_triple is a Phase-C helper; implementation is queued for the "
        "us_firm / sp500_eo / crypto_perps walkthrough."
    )


def cross_task_matrix(
    case_study: str,
    regression_label: str,
    classification_label: str,
    *,
    families: list[str] | None = None,
) -> pl.DataFrame:
    """{regression model, classification model} × {IC vs continuous, AUC vs binary}.

    Empirical curiosity table — does the regression model accidentally classify
    well? Does the classification model's continuous score correlate with the
    continuous return?

    Coverage caveat: only feasible for families that trained on both label
    types. Confirmed coverage: ``linear`` and ``gbm``. ``latent_factors`` has 1
    classification run for ``us_firm_characteristics`` only. ``deep_learning``
    has zero classification runs across all CSs.

    Returns columns:
        family, config_name,
        ic_from_regression_model,   ic_reg_ci_lo,   ic_reg_ci_hi,
        ic_from_classification_model, ic_cls_ci_lo, ic_cls_ci_hi,
        auc_from_regression_model,  auc_reg_ci_lo, auc_reg_ci_hi,
        auc_from_classification_model, auc_cls_ci_lo, auc_cls_ci_hi

    Status: stub — implementation queued for Phase C.
    """
    raise NotImplementedError(
        "cross_task_matrix is a Phase-C helper; implementation is queued for the "
        "us_firm / sp500_eo / crypto_perps walkthrough."
    )


# ---------------------------------------------------------------------------
# Conformal coverage diagnostic (spine v2 §7)
# ---------------------------------------------------------------------------


def conformal_coverage_diagnostic(
    case_study: str,
    label: str | None = None,
    *,
    levels: tuple[float, ...] = (0.80, 0.90, 0.95),
    families: list[str] | None = None,
) -> pl.DataFrame:
    """Per-family inductive split-conformal coverage at nominal levels.

    For each family's rank-1 validation config (by ``ic_mean_daily``), loads
    OOF predictions and uses **fold 0** as a calibration set to derive a
    symmetric absolute-residual quantile, then measures empirical coverage
    on the remaining folds at each nominal level. Interval width is reported
    as a fraction of the actuals' standard deviation, so families with
    different return scales are comparable.

    Returns columns:
        family, config_name, nominal_level,
        empirical_coverage, mean_interval_width_frac_std, n_test
    """
    label = label or PRIMARY_LABELS[case_study]
    db = _registry_path(case_study)
    if not db.exists():
        return pl.DataFrame()

    family_clause = ""
    params: list = [label]
    if families:
        placeholders = ",".join("?" * len(families))
        family_clause = f"AND t.family IN ({placeholders})"
        params.extend(families)

    sql = f"""
        SELECT
            t.family,
            t.config_name,
            p.prediction_hash,
            pm.ic_mean_daily
        FROM training_runs t
        JOIN prediction_sets p ON t.training_hash = p.training_hash
        JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
        WHERE t.label = ?
          AND p.split = 'validation'
          AND pm.ic_mean_daily IS NOT NULL
          {family_clause}
    """
    rows = _query(db, sql, tuple(params))
    if rows.is_empty():
        return pl.DataFrame()

    leaders = (
        rows.sort("ic_mean_daily", descending=True, nulls_last=True).group_by("family").first()
    )

    pred_dir = db.parent / "predictions"
    out_rows: list[dict] = []
    for fam, cfg, p_hash in zip(
        leaders["family"].to_list(),
        leaders["config_name"].to_list(),
        leaders["prediction_hash"].to_list(),
    ):
        pq = pred_dir / p_hash / "predictions.parquet"
        if not pq.exists():
            continue
        df = pl.read_parquet(pq)
        ren = {}
        if "actual" in df.columns and "y_true" not in df.columns:
            ren["actual"] = "y_true"
        if "prediction" in df.columns and "y_score" not in df.columns:
            ren["prediction"] = "y_score"
        if "fold" in df.columns and "fold_id" not in df.columns:
            ren["fold"] = "fold_id"
        if ren:
            df = df.rename(ren)
        if "y_true" not in df.columns or "y_score" not in df.columns:
            continue
        df = df.drop_nulls(["y_true", "y_score"])
        if df.height == 0 or "fold_id" not in df.columns:
            continue

        df = df.with_columns((pl.col("y_true") - pl.col("y_score")).abs().alias("abs_resid"))
        scale = float(df["y_true"].std() or 0.0)
        if not np.isfinite(scale) or scale == 0:
            continue

        fold_ids = sorted(df["fold_id"].unique().to_list())
        if len(fold_ids) < 2:
            continue

        cal = df.filter(pl.col("fold_id") == fold_ids[0])
        tst = df.filter(pl.col("fold_id").is_in(fold_ids[1:]))
        if cal.height < 30 or tst.height < 30:
            continue

        cal_res = cal["abs_resid"].to_numpy()
        tst_res = tst["abs_resid"].to_numpy()
        n_cal = len(cal_res)
        for level in levels:
            alpha = 1.0 - level
            q_level = min(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0)
            q_hat = float(np.quantile(cal_res, q_level))
            cov = float((tst_res <= q_hat).mean())
            width_std = (2.0 * q_hat) / scale
            out_rows.append(
                {
                    "family": fam,
                    "config_name": cfg,
                    "nominal_level": float(level),
                    "empirical_coverage": cov,
                    "mean_interval_width_frac_std": float(width_std),
                    "n_test": int(len(tst_res)),
                }
            )

    if not out_rows:
        return pl.DataFrame()
    return pl.DataFrame(out_rows).sort(["family", "nominal_level"])


__all__ = [
    "holdout_decay_table",
    "selection_adjusted_leader_table",
    "headline_forest_plot",
    "fold_heatmap_with_ci",
    "regime_coverage_strip",
    "classification_triple",
    "cross_task_matrix",
    "conformal_coverage_diagnostic",
]
