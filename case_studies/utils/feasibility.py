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


def fold_timeline(cv, timeline: pl.DataFrame, *, timestamp_col: str = "timestamp", title: str):
    """Walk-forward folds as a timeline, exportable to a static image.

    :func:`ml4t.diagnostic.visualization.cv_plots.plot_cv_folds` draws each block as
    a horizontal bar whose ``base`` is a pandas ``Timestamp`` and whose width is a
    duration in milliseconds. The static-image path serializes with ``orjson``,
    which rejects the ``Timestamp``; round-tripping through Plotly's own JSON
    encoder fixes that but leaves the x axis looking numeric, so it is declared a
    date axis explicitly. Without both steps the committed notebook carries either
    no PNG or one whose axis reads in epoch milliseconds.

    Parameters
    ----------
    cv
        A configured splitter. ``plot_cv_folds`` calls ``.split()`` itself, so this
        takes the splitter object and not the list of boundaries.
    timeline
        One column of timestamps covering the whole sample, holdout included, so
        the sealed block is drawn rather than inferred.
    """
    import plotly.io as pio
    from ml4t.diagnostic.visualization.cv_plots import plot_cv_folds

    figure = plot_cv_folds(
        cv,
        timeline,
        timestamp_col=timestamp_col,
        show_purge_gaps=True,
        show_test_period=True,
        title=title,
    )
    normalized = pio.from_json(figure.to_json())
    normalized.update_xaxes(type="date")
    normalized.update_layout(margin={"l": 110})
    return normalized


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
