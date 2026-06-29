"""Shared visualization helpers for model_analysis notebooks.

Each function renders one figure and optionally prints summary statistics.
All functions accept pre-computed data (from model_analysis.py helpers)
and produce matplotlib figures. The notebooks provide the narrative
context; these functions handle the rendering.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from utils.style import COLORS

# ---------------------------------------------------------------------------
# Figure 1: Cross-Validation Timeline
# ---------------------------------------------------------------------------


def plot_cv_timeline(
    fold_ranges: pl.DataFrame,
    n_splits: int,
    holdout_start: str | None = None,
    date_col: str = "timestamp",
) -> None:
    """Plot walk-forward fold validation windows as horizontal bars."""
    if fold_ranges.height == 0:
        return

    fig, ax = plt.subplots(figsize=(12, max(4, n_splits * 0.6)))

    for row in fold_ranges.iter_rows(named=True):
        fold = row["fold_id"]
        vs = row["val_start"]
        ve = row["val_end"]
        ax.barh(
            fold,
            (ve - vs).days,
            left=vs,
            height=0.6,
            color=COLORS.get("amber", "#F59E0B"),
            alpha=0.8,
            label="Validation" if fold == 0 else "",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Fold")
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f"Fold {i}" for i in range(n_splits)])
    ax.invert_yaxis()
    ax.set_title("Walk-Forward Cross-Validation Design")

    if holdout_start:
        import pandas as pd

        ax.axvline(
            pd.Timestamp(holdout_start),
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Holdout start",
        )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        dict(zip(labels, handles, strict=False)).values(),
        dict(zip(labels, handles, strict=False)).keys(),
        loc="lower right",
    )
    fig.tight_layout()
    fig.show()


# ---------------------------------------------------------------------------
# Figure 2: Fold-by-Model Performance Heatmap
# ---------------------------------------------------------------------------


def plot_fold_heatmap(
    fold_ic: pl.DataFrame,
) -> tuple[list[str], list[str], np.ndarray]:
    """Plot fold × model IC heatmap with mean annotations.

    Returns (model_labels, fold_cols, matrix) for downstream use.
    """
    if fold_ic.height == 0:
        return [], [], np.array([])

    pivot = fold_ic.pivot(on="fold_id", index="model_label", values="ic_mean")
    model_labels = pivot["model_label"].to_list()
    fold_cols = [c for c in pivot.columns if c != "model_label"]
    matrix = pivot.select(fold_cols).to_numpy()
    row_means = np.nanmean(matrix, axis=1)

    n_models = len(model_labels)
    n_folds = len(fold_cols)

    fig, ax = plt.subplots(figsize=(max(8, n_folds * 1.2), max(4, n_models * 0.8)))

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.01)
    im = ax.imshow(matrix, cmap="RdYlBu", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(n_models):
        for j in range(n_folds):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n_folds))
    ax.set_xticklabels([f"Fold {c}" for c in fold_cols], rotation=45, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_labels)
    ax.set_title("Validation IC by Model Family and Fold")

    for i, mean in enumerate(row_means):
        ax.text(
            n_folds + 0.3,
            i,
            f"{mean:+.3f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.text(n_folds + 0.3, -0.7, "Mean", ha="left", va="center", fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=ax, label="IC", shrink=0.8)
    fig.show()

    return model_labels, fold_cols, matrix


# ---------------------------------------------------------------------------
# Figure 3: Fold Performance Distribution Boxplot
# ---------------------------------------------------------------------------


def plot_fold_boxplot(fold_ic: pl.DataFrame) -> None:
    """Boxplot with jittered scatter of fold-level IC per model family."""
    if fold_ic.height == 0:
        return

    families = fold_ic["model_label"].unique().sort().to_list()
    n_families = len(families)

    fig, ax = plt.subplots(figsize=(max(8, n_families * 1.5), 5))

    bp_data = []
    for fam in families:
        vals = fold_ic.filter(pl.col("model_label") == fam)["ic_mean"].to_numpy()
        bp_data.append(vals)

    bp = ax.boxplot(
        bp_data,
        positions=list(range(n_families)),
        widths=0.5,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"),
    )

    palette = list(COLORS.values())[:n_families]
    for patch, color in zip(bp["boxes"], palette, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    rng = np.random.default_rng(42)
    for i, (fam, vals) in enumerate(zip(families, bp_data, strict=False)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full_like(vals, i) + jitter,
            vals,
            color=palette[i % len(palette)],
            alpha=0.7,
            s=30,
            zorder=5,
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(list(range(n_families)))
    ax.set_xticklabels([l.split("/")[0] for l in families], rotation=30, ha="right")
    ax.set_ylabel("Mean IC per Fold")
    ax.set_title("Fold Performance Distribution by Model Family")
    fig.tight_layout()
    fig.show()


# ---------------------------------------------------------------------------
# Figure 4: Prediction Bucket Monotonicity
# ---------------------------------------------------------------------------


def plot_bucket_monotonicity(
    bucket_results: dict[str, pl.DataFrame],
    n_buckets: int,
    unconditional_mean: float | None = None,
    label_name: str = "Forward Return",
    cost_range: list[int] | None = None,
) -> None:
    """Plot mean return per prediction bucket for each model family."""
    if not bucket_results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    palette_items = list(COLORS.items())
    for i, (label, buckets) in enumerate(bucket_results.items()):
        color = palette_items[i % len(palette_items)][1]
        x = buckets["bucket"].to_numpy()
        y = buckets["mean_return"].to_numpy()
        ax.plot(x, y, marker="o", label=label, color=color, linewidth=2)

    if unconditional_mean is not None:
        ax.axhline(
            unconditional_mean,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label=f"Unconditional mean ({unconditional_mean:.4f})",
        )

    ax.set_xlabel(f"Prediction Bucket (1 = lowest, {n_buckets} = highest)")
    ax.set_ylabel(f"Mean Realized {label_name}")
    ax.set_title("Do Higher Predictions Correspond to Higher Realized Returns?")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.show()

    # Cost context
    if cost_range:
        print(
            f"\nTop-bottom bucket spread vs trading costs ({cost_range[0]}–{cost_range[1]} bps per leg):"
        )
        for label, buckets in bucket_results.items():
            top = buckets.filter(pl.col("bucket") == n_buckets)["mean_return"][0]
            bottom = buckets.filter(pl.col("bucket") == 1)["mean_return"][0]
            spread = top - bottom
            spread_bps = spread * 10000
            cost_ratio_low = spread_bps / (2 * cost_range[0])
            cost_ratio_high = spread_bps / (2 * cost_range[1])
            print(
                f"  {label:20s}  spread={spread_bps:+.0f} bps  "
                f"edge/cost={cost_ratio_low:.1f}–{cost_ratio_high:.1f}x"
            )


# ---------------------------------------------------------------------------
# Figure 5: Prediction Correlation Heatmap
# ---------------------------------------------------------------------------


def plot_correlation_matrix(
    corr_matrix: np.ndarray,
    labels: list[str],
) -> None:
    """Plot pairwise prediction correlation heatmap."""
    if corr_matrix.size == 0 or len(labels) < 2:
        return

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n)))

    im = ax.imshow(corr_matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    short_labels = [l.split("/")[0] for l in labels]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels)
    ax.set_title("Pairwise Prediction Rank Correlation")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.show()

    off_diag = corr_matrix[np.triu_indices(n, k=1)]
    print(f"\nAverage pairwise correlation: {off_diag.mean():.2f}")
    print(f"Range: {off_diag.min():.2f} to {off_diag.max():.2f}")


# ---------------------------------------------------------------------------
# Figure 6: Learning Curves
# ---------------------------------------------------------------------------


def plot_learning_curves(
    cp_data: pl.DataFrame,
    cp_families: list[str],
) -> None:
    """Plot IC vs checkpoint for each config within each family."""
    if not cp_families or cp_data.height == 0:
        return

    n_panels = len(cp_families)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), squeeze=False)

    for idx, family in enumerate(sorted(cp_families)):
        ax = axes[idx, 0]
        fam_data = cp_data.filter(pl.col("family") == family)

        for config in sorted(fam_data["config_name"].unique().to_list()):
            cfg_data = fam_data.filter(pl.col("config_name") == config).sort("checkpoint_value")
            x = cfg_data["checkpoint_value"].to_numpy()
            y = cfg_data["ic_mean"].to_numpy()

            ax.plot(x, y, marker=".", label=config, linewidth=1.5)

            if "ic_std" in cfg_data.columns:
                y_std = cfg_data["ic_std"].to_numpy()
                valid = ~np.isnan(y_std)
                if valid.any():
                    ax.fill_between(x[valid], (y - y_std)[valid], (y + y_std)[valid], alpha=0.15)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Checkpoint (epoch / trees)")
        ax.set_ylabel("Mean IC (across folds)")
        ax.set_title(f"Learning Curve: {family}")
        ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.show()


# ---------------------------------------------------------------------------
# Figure 7: Feature Importance Stability Heatmap
# ---------------------------------------------------------------------------


def plot_feature_importance_heatmap(
    importance_df: pl.DataFrame,
    top_n: int = 15,
) -> None:
    """Plot feature importance (normalized) across folds as a heatmap."""
    if importance_df is None or importance_df.height == 0:
        return

    pivot = (
        importance_df.group_by(["feature", "fold_id"])
        .agg(pl.col("importance_norm").mean())
        .pivot(on="fold_id", index="feature", values="importance_norm")
    )

    fold_cols = [c for c in pivot.columns if c != "feature"]
    features = pivot["feature"].to_list()
    imp_matrix = pivot.select(fold_cols).to_numpy()
    mean_imp = np.nanmean(imp_matrix, axis=1)
    sort_idx = np.argsort(mean_imp)[::-1]

    n_show = min(top_n, len(features))
    features_sorted = [features[i] for i in sort_idx[:n_show]]
    matrix_sorted = imp_matrix[sort_idx[:n_show]]

    fig, ax = plt.subplots(figsize=(max(8, len(fold_cols)), max(6, n_show * 0.4)))

    im = ax.imshow(matrix_sorted, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    for i in range(n_show):
        for j in range(len(fold_cols)):
            val = matrix_sorted[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    ax.set_xticks(range(len(fold_cols)))
    ax.set_xticklabels([f"Fold {c}" for c in fold_cols], rotation=45, ha="right")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(features_sorted)
    ax.set_title("Feature Importance Stability Across Folds")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.show()

    # Recurrence summary
    n_total_folds = importance_df["fold_id"].n_unique()
    top5_per_fold = (
        importance_df.sort(["fold_id", "importance_norm"], descending=[False, True])
        .group_by("fold_id")
        .head(5)
    )
    recurrence = (
        top5_per_fold.group_by("feature")
        .agg(pl.len().alias("n_top5"))
        .sort("n_top5", descending=True)
    )
    persistent = recurrence.filter(pl.col("n_top5") >= n_total_folds * 0.75)
    if persistent.height > 0:
        print(f"\nPersistent features (top-5 in ≥75% of folds): {persistent['feature'].to_list()}")


# ---------------------------------------------------------------------------
# Figure 8: Regime-Conditional Performance Bars
# ---------------------------------------------------------------------------


def plot_regime_bars(
    regime_df: pl.DataFrame,
) -> None:
    """Grouped bar chart of IC by volatility regime per family."""
    if regime_df.height == 0:
        return

    regimes = sorted(regime_df["regime"].unique().to_list())
    families = sorted(regime_df["family"].unique().to_list())
    n_fam = len(families)

    fig, ax = plt.subplots(figsize=(max(8, n_fam * 2), 5))

    x = np.arange(n_fam)
    width = 0.35
    colors_regime = {
        "low_vol": COLORS.get("blue", "#3B82F6"),
        "high_vol": COLORS.get("amber", "#F59E0B"),
    }

    has_hac = "ic_se_hac" in regime_df.columns
    for i, regime in enumerate(regimes):
        regime_data = regime_df.filter(pl.col("regime") == regime)
        ics, ses = [], []
        for fam in families:
            fam_data = regime_data.filter(pl.col("family") == fam)
            if fam_data.height > 0:
                # Prefer HAC SE when the daily-uncertainty backfill ran.
                # Fall back to fold-std/sqrt(n) only when HAC is missing.
                if has_hac and fam_data["ic_se_hac"][0] is not None:
                    ic = (
                        fam_data.get_column("ic_mean_daily")[0]
                        if "ic_mean_daily" in fam_data.columns
                        else fam_data["ic_mean"][0]
                    )
                    se = fam_data["ic_se_hac"][0]
                else:
                    ic = fam_data["ic_mean"][0]
                    std = fam_data["ic_std"][0]
                    n = fam_data["n_periods"][0]
                    se = std / np.sqrt(max(n, 1))
                ics.append(ic)
                ses.append(se)
            else:
                ics.append(0)
                ses.append(0)

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            ics,
            width,
            yerr=ses,
            label=regime.replace("_", " ").title(),
            color=colors_regime.get(regime, f"C{i}"),
            alpha=0.8,
            capsize=3,
        )

        for j, (bar, ic) in enumerate(zip(bars, ics, strict=False)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{ic:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("Mean IC")
    ax.set_title("Model Performance by Volatility Regime")
    ax.legend()
    fig.tight_layout()
    fig.show()


# ---------------------------------------------------------------------------
# HAC-CI leaderboard + rolling daily-IC plot
# ---------------------------------------------------------------------------


def plot_hac_ci_leaderboard(
    metrics: pl.DataFrame,
    *,
    label_col: str = "config_name",
    family_col: str = "family",
    ic_col: str = "ic_mean_daily",
    lo_col: str = "ic_ci_lo",
    hi_col: str = "ic_ci_hi",
    boot_lo_col: str = "ic_boot_lo",
    boot_hi_col: str = "ic_boot_hi",
    title: str = "Daily-pooled IC ± HAC 95% CI",
    top_n: int | None = 25,
) -> None:
    """Dot-plot leaderboard of daily-pooled IC with HAC CIs.

    Each row is one model config; the dot is the daily-IC point estimate, the
    thick bar is the HAC 95% CI, and a faint outer bar is the bootstrap CI
    when present. Configs with overlapping HAC CIs are visually clustered by
    a faint shaded band so the reader sees which gaps are within noise.
    """
    if metrics.height == 0 or ic_col not in metrics.columns:
        return

    df = metrics.sort(ic_col, descending=True, nulls_last=True)
    if top_n is not None and df.height > top_n:
        df = df.head(top_n)

    n = df.height
    fig, ax = plt.subplots(figsize=(8, max(3.5, n * 0.28)))

    family_order = list(dict.fromkeys(df[family_col].to_list()))
    palette = {
        f: COLORS.get(c, f"C{i}")
        for i, (f, c) in enumerate(
            zip(family_order, ("blue", "amber", "emerald", "violet", "rose", "teal"), strict=False)
        )
    }

    y = np.arange(n)[::-1]  # top-to-bottom highest-IC-first
    has_boot = boot_lo_col in df.columns and boot_hi_col in df.columns

    # Indistinguishable-CI shading: bands of overlapping CIs.
    if {lo_col, hi_col}.issubset(df.columns):
        ic_vals = df[ic_col].to_numpy()
        lo_vals = df[lo_col].to_numpy()
        hi_vals = df[hi_col].to_numpy()
        running_lo = float("inf")
        band_start = None
        band_idx = 0
        for k in range(n):
            lo, hi = lo_vals[k], hi_vals[k]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            if band_start is None:
                band_start = k
                running_lo = lo
                continue
            if hi >= running_lo:
                running_lo = max(running_lo, lo)
            else:
                if k - band_start >= 2:
                    ax.axhspan(
                        y[k - 1] - 0.45,
                        y[band_start] + 0.45,
                        color=("0.92" if band_idx % 2 == 0 else "0.96"),
                        zorder=0,
                    )
                    band_idx += 1
                band_start = k
                running_lo = lo
        if band_start is not None and n - band_start >= 2:
            ax.axhspan(
                y[n - 1] - 0.45,
                y[band_start] + 0.45,
                color=("0.92" if band_idx % 2 == 0 else "0.96"),
                zorder=0,
            )

    for k in range(n):
        row = df.row(k, named=True)
        fam = row.get(family_col, "?")
        col = palette.get(fam, "0.4")
        ic = row.get(ic_col)
        lo = row.get(lo_col)
        hi = row.get(hi_col)
        if ic is None or not np.isfinite(ic):
            continue
        if has_boot:
            blo = row.get(boot_lo_col)
            bhi = row.get(boot_hi_col)
            if blo is not None and bhi is not None:
                ax.hlines(y[k], blo, bhi, color="0.7", linewidth=1.0, zorder=2)
        if lo is not None and hi is not None:
            ax.hlines(y[k], lo, hi, color=col, linewidth=2.5, zorder=3)
        ax.plot(ic, y[k], "o", color=col, markersize=5, zorder=4)

    ax.axvline(0, color="0.5", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r[family_col]} / {r[label_col]}" for r in df.iter_rows(named=True)],
        fontsize=7,
    )
    ax.set_xlabel("Daily-pooled IC")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.show()


def plot_label_horizon_forest(
    metrics: pl.DataFrame,
    *,
    families: list[str] | None = None,
    labels: list[str] | None = None,
    label_display: dict[str, str] | None = None,
    family_display: dict[str, str] | None = None,
    ic_col: str = "ic_mean_daily",
    lo_col: str = "ic_ci_lo",
    hi_col: str = "ic_ci_hi",
    family_col: str = "family",
    label_col: str = "label",
    title: str = "",
) -> None:
    """Small-multiples forest of rank-1 IC ± HAC 95% CI per (family, label).

    Each subplot is one label/horizon; within a subplot, families occupy
    fixed y positions in caller-supplied order. Tiles where a (family, label)
    pair has no run are drawn as a gray "no run" stub at zero so the gap is
    visible. CIs that straddle zero render in muted gray; CIs that exclude
    zero render in the family color from :data:`utils.style.COLORS`.

    Parameters
    ----------
    metrics
        Long-format frame with one row per (family, label) rank-1 config.
        Columns required: ``family_col``, ``label_col``, ``ic_col``,
        ``lo_col``, ``hi_col``.
    families
        Display order for families along the y-axis. Defaults to the unique
        family list as seen in ``metrics`` (sorted).
    labels
        Display order for labels across subplots. Defaults to the unique
        label list as seen in ``metrics`` (sorted).
    """
    if metrics is None or metrics.height == 0 or ic_col not in metrics.columns:
        return

    fams = list(families) if families else sorted(metrics[family_col].unique().to_list())
    lbls = list(labels) if labels else sorted(metrics[label_col].unique().to_list())
    n_lab = len(lbls)
    n_fam = len(fams)
    if n_lab == 0 or n_fam == 0:
        return

    family_palette = {
        "linear": COLORS.get("blue", "C0"),
        "gbm": COLORS.get("orange", "C1"),
        "deep_learning": COLORS.get("green", "C2"),
        "tabular_dl": COLORS.get("purple", "C3"),
        "latent_factors": COLORS.get("red", "C4"),
        "causal": COLORS.get("brown", "C5"),
        "causal_dml": COLORS.get("brown", "C5"),
        "benchmark": COLORS.get("gray", "C7"),
    }
    label_display = label_display or {}
    family_display = family_display or {}

    fig, axes = plt.subplots(
        1,
        n_lab,
        figsize=(3.2 * n_lab + 0.5, max(2.5, 0.45 * n_fam + 1.2)),
        sharey=True,
        constrained_layout=True,
    )
    if n_lab == 1:
        axes = [axes]

    y_pos = np.arange(n_fam)
    for ax, lbl in zip(axes, lbls):
        sub = metrics.filter(pl.col(label_col) == lbl)
        sub_map = {r[family_col]: r for r in sub.iter_rows(named=True)}
        for i, fam in enumerate(fams):
            row = sub_map.get(fam)
            if row is None or row.get(ic_col) is None or not np.isfinite(row.get(ic_col)):
                ax.text(
                    0.0,
                    y_pos[i],
                    "no run",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="0.55",
                    style="italic",
                )
                continue
            ic = float(row[ic_col])
            lo = row.get(lo_col)
            hi = row.get(hi_col)
            ci_valid = lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi)
            crosses_zero = bool(ci_valid and lo <= 0 <= hi)
            color = (
                "#999999" if (not ci_valid or crosses_zero) else family_palette.get(fam, "#444444")
            )
            if ci_valid:
                ax.plot([lo, hi], [y_pos[i], y_pos[i]], color=color, linewidth=2.0, alpha=0.85)
            ax.plot(ic, y_pos[i], marker="o", color=color, markersize=6, zorder=3)

        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(label_display.get(lbl, lbl), fontsize=10)
        ax.grid(True, axis="x", linestyle=":", alpha=0.3)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(
        [family_display.get(f, f) for f in fams],
        fontsize=9,
    )
    axes[0].invert_yaxis()
    fig.supxlabel("Information Coefficient (daily-pooled, 95% HAC CI)", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.show()


def plot_rolling_daily_ic(
    daily_metrics: pl.DataFrame,
    *,
    window: int = 63,
    label: str = "",
) -> None:
    """Plot rolling mean of daily IC with a faint shaded band for daily noise.

    Expects a frame with columns ``[fold_id, date, ic, n_obs]`` (the
    `daily_metrics.parquet` written by the backfill). Pools across folds by
    sorting on ``date`` and computing the rolling mean.
    """
    if daily_metrics is None or daily_metrics.height == 0 or "ic" not in daily_metrics.columns:
        return

    df = daily_metrics.drop_nulls("ic").sort("date")
    if df.height < window:
        window = max(5, df.height // 4)

    dates = df["date"].to_numpy()
    ic = df["ic"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(dates, ic, color="0.7", linewidth=0.4, alpha=0.6, label="Daily IC")
    if window > 1 and df.height >= window:
        roll_mean = (
            df.with_columns(pl.col("ic").rolling_mean(window).alias("__roll"))
            .get_column("__roll")
            .to_numpy()
        )
        ax.plot(
            dates,
            roll_mean,
            color=COLORS.get("blue", "#3B82F6"),
            linewidth=1.6,
            label=f"Rolling mean ({window}d)",
        )

    ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cross-sectional IC")
    ax.set_title(f"Daily IC time series{(' — ' + label) if label else ''}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.show()
