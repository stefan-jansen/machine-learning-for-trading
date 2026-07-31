"""Diagnostics shared by the nine ``01_feasibility_analysis`` notebooks.

Each helper here answers a question the stage asks of every case study, in a form
that survives being applied to a panel. The notebooks keep the loading, the
figures and the interpretation; what lives here is the statistic whose correct
version is longer than a cell should be.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

__all__ = ["exceedance_curve", "fold_timeline", "panel_acf"]


def fold_timeline(ax, splits: list[dict], *, holdout: tuple[str, str]) -> None:
    """Draw walk-forward folds on ``ax`` from the boundaries a caller already has.

    :func:`ml4t.diagnostic.visualization.cv_plots.plot_cv_folds` takes a splitter and
    re-splits whatever timeline it is handed, so the picture it draws is a second
    computation that can disagree with the one the notebook reports - it did, by
    eleven days, on a timeline truncated at the last validation date. This draws the
    boundaries themselves, so the figure and the folds cannot come apart.

    Parameters
    ----------
    splits
        As returned by ``utils.cv_splits.generate_cv_splits``: ``train_start``,
        ``train_end``, ``val_start``, ``val_end`` per fold. The span between
        ``train_end`` and ``val_start`` is the purge gap and is drawn as such.
    holdout
        Start and end of the sealed block, shaded behind the folds.
    """
    from matplotlib.patches import Patch

    from utils.style import COLORS

    bands = [
        ("train_start", "train_end", COLORS["blue"]),
        ("val_start", "val_end", COLORS["amber"]),
    ]
    for row, split in enumerate(splits):
        for lo, hi, color in bands:
            ax.barh(row, split[hi] - split[lo], left=split[lo], height=0.62, color=color)
        ax.barh(
            row,
            split["val_start"] - split["train_end"],
            left=split["train_end"],
            height=0.62,
            color=COLORS["silver_muted"],
        )
    ax.axvspan(*(np.datetime64(d) for d in holdout), color=COLORS["copper"], alpha=0.25)
    ax.set_yticks(range(len(splits)), [f"Fold {s['fold'] + 1}" for s in splits])
    ax.invert_yaxis()
    ax.legend(
        handles=[
            Patch(color=COLORS["blue"], label="train"),
            Patch(color=COLORS["silver_muted"], label="purge"),
            Patch(color=COLORS["amber"], label="validation"),
            Patch(color=COLORS["copper"], alpha=0.25, label="sealed holdout"),
        ],
        frameon=False,
        fontsize=8,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )


def panel_acf(
    frame: pl.DataFrame,
    *,
    entity_col: str,
    value_col: str,
    max_lags: int,
    min_obs: int = 30,
) -> pl.DataFrame:
    """Within-entity autocorrelation of a panel series, pooled across entities.

    A single-series ACF over a stacked panel measures dependence across the
    cross-section wherever one entity's last observation meets the next entity's
    first. This computes the ACF separately per entity with
    :func:`ml4t.diagnostic.evaluation.autocorrelation.compute_acf` and returns the
    cross-entity mean at each lag, with the 10th and 90th percentiles of the
    per-entity curves and the white-noise band a single entity's sample implies.

    Parameters
    ----------
    frame
        Long panel with one row per entity and period, already at the cadence the
        autocorrelation should be read at.
    entity_col, value_col
        Entity identifier and the carrier series to correlate with its own past.
    max_lags
        Highest lag returned. Entities with fewer than ``max_lags + 1``
        observations are skipped.
    min_obs
        Minimum observations an entity needs to contribute a curve.

    Returns
    -------
    pl.DataFrame
        Columns ``lag``, ``acf``, ``acf_p10``, ``acf_p90``, ``band``, ``n_entities``.
    """
    from ml4t.diagnostic.evaluation.autocorrelation import compute_acf

    logger = logging.getLogger("ml4t.diagnostic.evaluation.autocorrelation")
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        curves: list[np.ndarray] = []
        lengths: list[int] = []
        for block in frame.partition_by(entity_col, maintain_order=True):
            series = block[value_col].drop_nulls().to_numpy()
            if len(series) < max(min_obs, max_lags + 1):
                continue
            curves.append(np.asarray(compute_acf(series, nlags=max_lags).values)[: max_lags + 1])
            lengths.append(len(series))
    finally:
        logger.setLevel(previous_level)

    if not curves:
        raise ValueError(f"no {entity_col} carries {min_obs} observations of {value_col}")

    stacked = np.vstack(curves)
    band = 1.96 / np.sqrt(float(np.mean(lengths)))
    return pl.DataFrame(
        {
            "lag": np.arange(max_lags + 1),
            "acf": stacked.mean(axis=0),
            "acf_p10": np.percentile(stacked, 10, axis=0),
            "acf_p90": np.percentile(stacked, 90, axis=0),
            "band": np.full(max_lags + 1, band),
            "n_entities": np.full(max_lags + 1, len(curves)),
        }
    )


def exceedance_curve(values: np.ndarray, n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Magnitudes and the fraction of ``values`` that exceeds each of them.

    The survival function of ``|return|``, thinned to ``n_points`` quantiles so a
    multi-million-row panel draws as a curve rather than a raster. Read against a
    cost line it gives the fraction of moves that clears the round trip.

    Returns
    -------
    tuple of ndarray
        Magnitudes, ascending, and the fraction of the sample at or above each.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("no finite values to build an exceedance curve from")
    finite.sort()
    if finite.size > n_points:
        take = np.unique(np.linspace(0, finite.size - 1, n_points).astype(int))
        magnitudes = finite[take]
        fraction = 1.0 - take / finite.size
    else:
        magnitudes = finite
        fraction = 1.0 - np.arange(finite.size) / finite.size
    return magnitudes, fraction
