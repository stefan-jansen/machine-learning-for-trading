"""Diagnostics shared by the nine ``01_feasibility_analysis`` notebooks.

Each helper here answers a question the stage asks of every case study, in a form
that survives being applied to a panel. The notebooks keep the loading, the
figures and the interpretation; what lives here is the statistic whose correct
version is longer than a cell should be.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["cross_sectional_persistence", "exceedance_curve", "fold_timeline", "panel_acf"]


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
        As returned by ``utils.cv_splits.generate_cv_splits``: ``fold``,
        ``train_start``, ``train_end``, ``val_start``, ``val_end`` per fold. Pass
        them in the order the splitter returned; the rows are drawn earliest-first
        here so the picture runs forward in time. The span between ``train_end``
        and ``val_start`` is the purge gap and is drawn as such.

        Each row is labelled with the splitter's own ``fold``, which numbers folds
        from zero backwards from the most recent, so the labels count down as the
        rows move forward. Every later stage prints and keys its tables on that
        same number, so relabelling the rows by position would make this figure's
        "Fold 0" a different fold from the one the rest of the case study reports.
    holdout
        Start and end of the holdout, shaded behind the folds.
    """
    from matplotlib.patches import Patch

    from utils.style import COLORS

    bands = [
        ("train_start", "train_end", COLORS["blue"]),
        ("val_start", "val_end", COLORS["amber"]),
    ]
    splits = sorted(splits, key=lambda split: split["train_start"])
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
    ax.set_yticks(range(len(splits)), [f"Fold {s['fold']}" for s in splits])
    ax.invert_yaxis()
    ax.legend(
        handles=[
            Patch(color=COLORS["blue"], label="training"),
            Patch(color=COLORS["silver_muted"], label="purge gap"),
            Patch(color=COLORS["amber"], label="validation"),
            Patch(color=COLORS["copper"], alpha=0.25, label="holdout"),
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
    period_col: str | None = None,
) -> pl.DataFrame:
    """Within-entity autocorrelation of a panel series, pooled across entities.

    A single-series ACF over a stacked panel measures dependence across the
    cross-section wherever one entity's last observation meets the next entity's
    first. This computes the ACF separately per entity and returns the cross-entity
    mean at each lag, with the 10th and 90th percentiles of the per-entity curves
    and two references for reading it.

    The per-entity estimator is the one
    :func:`ml4t.diagnostic.evaluation.autocorrelation.compute_acf` uses, and
    ``tests/test_feasibility_helpers.py`` pins the two together on a series with no
    gaps. It is computed here rather than called because the library's estimator has
    no gap-aware form: its ``missing="drop"`` closes the gaps and its
    ``missing="conservative"`` keeps them but shrinks every coefficient by the share
    of pairs the gaps removed, which on a panel with 20% of periods missing reported
    0.48 for a first-order autocorrelation of 0.60.

    Parameters
    ----------
    frame
        Long panel with one row per entity and period, already at the cadence the
        autocorrelation should be read at.
    entity_col, value_col
        Entity identifier and the series to correlate with its own past.
    max_lags
        Highest lag returned. Entities with fewer than ``max_lags + 1``
        observations are skipped.
    min_obs
        Minimum observations an entity needs to contribute a curve.
    period_col
        Integer period index, one step per period of the cadence, unique within an
        entity. Supply it wherever an entity can be missing a period. Each entity's
        series is then laid out on its own dense period grid and correlated with the
        gaps in place, so a curve reported at lag *k* pairs observations *k* periods
        apart. Without it the rows are taken as consecutive, and a pair straddling a
        missing period is counted at too short a lag. On the ``us_equities_panel``
        development window that mislabels one pair in ten thousand; on an intraday
        panel, or a carrier defined on only some periods, it is far higher.

    Returns
    -------
    pl.DataFrame
        Columns ``lag``, ``acf``, ``acf_p10``, ``acf_p90``, ``band``, ``pooled_se``,
        ``n_entities``, ``obs_per_entity``, ``n_dropped``.

        ``band`` is ``1.96 / sqrt(T)`` for the mean entity length ``T``: the white
        noise reference for *one* entity's curve, which is the right reference for a
        daily panel and useless on an intraday one, where the entity that avoids
        pooling across session boundaries carries a few dozen observations and the
        band is drawn wider than the estimate. ``pooled_se`` is the standard error of
        the plotted quantity itself, the cross-entity mean, and stays usable there.
        ``obs_per_entity`` is the mean entity length the band is built from.

        ``n_entities`` is counted **per lag**. A sparse entity can have pairs at lag
        1 and none at lag 12, and it contributes wherever it has them rather than
        being dropped from the whole curve; where the count moves across lags, the
        mean at each lag is over a different set and the column says so.
        ``n_dropped`` counts the entities that contributed at no lag at all, which is
        what a constant series does - it has no autocorrelation to report.
    """
    curves: list[np.ndarray] = []
    lengths: list[int] = []
    dropped = 0
    for block in frame.partition_by(entity_col, maintain_order=True):
        series = _entity_series(block, value_col=value_col, period_col=period_col)
        observed = int(np.count_nonzero(~np.isnan(series)))
        if observed < max(min_obs, max_lags + 1):
            continue
        curve = _lag_exact_acf(series, max_lags)
        # A zero-variance entity has nothing to report at any lag. Pooling it with a
        # plain mean would propagate its NaN to every lag and draw the figure empty
        # with nothing raised; one firm in 10,587 did exactly that.
        if not np.any(np.isfinite(curve[1:])):
            dropped += 1
            continue
        curves.append(curve)
        lengths.append(observed)

    if not curves:
        raise ValueError(f"no {entity_col} carries {min_obs} observations of {value_col}")

    stacked = np.vstack(curves)
    finite = np.isfinite(stacked)
    obs_per_entity = float(np.mean(lengths))

    def per_lag(reduce) -> np.ndarray:
        out = np.full(max_lags + 1, np.nan)
        for lag in range(max_lags + 1):
            column = stacked[finite[:, lag], lag]
            if column.size:
                out[lag] = reduce(column)
        return out

    return pl.DataFrame(
        {
            "lag": np.arange(max_lags + 1),
            "acf": per_lag(np.mean),
            "acf_p10": per_lag(lambda c: np.percentile(c, 10)),
            "acf_p90": per_lag(lambda c: np.percentile(c, 90)),
            "band": np.full(max_lags + 1, 1.96 / np.sqrt(obs_per_entity)),
            "pooled_se": per_lag(
                lambda c: c.std(ddof=1) / np.sqrt(c.size) if c.size > 1 else np.nan
            ),
            "n_entities": finite.sum(axis=0),
            "obs_per_entity": np.full(max_lags + 1, obs_per_entity),
            "n_dropped": np.full(max_lags + 1, dropped),
        }
    )


def _lag_exact_acf(series: np.ndarray, max_lags: int) -> np.ndarray:
    """Autocorrelation of one series, pairing observations by position, not by row.

    ``series`` is laid out one element per period, NaN where the entity has none.
    At each lag the sum runs over the pairs that exist and is divided by how many
    there are, so a gap costs the pairs it straddles and nothing else. The
    ``(T - lag) / T`` taper and the division by the whole-series variance are what
    ``statsmodels`` applies, which is what makes this equal
    :func:`ml4t.diagnostic.evaluation.autocorrelation.compute_acf` on a series with
    no gaps.

    Returns all-NaN for a constant series, which has no autocorrelation.
    """
    observed = ~np.isnan(series)
    n_observed = int(np.count_nonzero(observed))
    total = float(len(series))
    out = np.full(max_lags + 1, np.nan)
    if n_observed == 0 or total <= max_lags:
        return out

    centred = np.where(observed, series - series[observed].mean(), 0.0)
    variance = float(np.dot(centred, centred) / n_observed)
    if variance <= 0.0:
        return out

    out[0] = 1.0
    for lag in range(1, max_lags + 1):
        pairs = observed[:-lag] & observed[lag:]
        n_pairs = int(np.count_nonzero(pairs))
        if n_pairs == 0:
            continue
        cross = float(np.dot(centred[:-lag], centred[lag:]))
        out[lag] = (cross / n_pairs) * ((total - lag) / total) / variance
    return out


def _entity_series(
    block: pl.DataFrame,
    *,
    value_col: str,
    period_col: str | None,
) -> np.ndarray:
    """One entity's values, on its own dense period grid when it has one.

    Without ``period_col`` the rows are taken as consecutive and nulls are dropped,
    which closes the gaps. With it, a missing period stays a hole.
    """
    if period_col is None:
        return block[value_col].drop_nulls().to_numpy().astype(float)

    periods = block[period_col].to_numpy().astype(np.int64)
    values = block[value_col].cast(pl.Float64).to_numpy()
    order = np.argsort(periods, kind="mergesort")
    periods, values = periods[order], values[order]

    grid = np.full(int(periods[-1] - periods[0]) + 1, np.nan)
    grid[periods - periods[0]] = values
    # Trim to the observed span: a leading or trailing hole carries no pair, and
    # leaving it in would lengthen T and taper every coefficient for nothing.
    observed = np.flatnonzero(~np.isnan(grid))
    if observed.size == 0:
        return np.empty(0)
    return grid[observed[0] : observed[-1] + 1]


def cross_sectional_persistence(
    frame: pl.DataFrame,
    *,
    time_col: str,
    entity_col: str,
    value_col: str,
    max_lags: int,
    min_entities: int = 20,
) -> pl.DataFrame:
    """How much of an ordering across entities survives a given number of periods.

    The companion to :func:`panel_acf`, for a strategy that trades a rotation rather
    than a per-entity series. ``panel_acf`` asks whether an entity holds its own
    level and needs an uninterrupted series to do it, so an entity that stops
    quoting for a week cannot contribute at all; where membership turns over, that
    restricts the statistic to the entities least like the ones the turnover is
    about. This ranks ``value_col`` inside each period and correlates the ranking
    with the ranking ``lag`` periods later over the entities present in both, which
    is the quantity a rebalance pays for and which no entity is excluded from.

    Parameters
    ----------
    frame
        Long panel, one row per entity and period, at the cadence the ordering is
        rebuilt on - decision dates for a strategy that re-ranks on a schedule.
    time_col, entity_col, value_col
        Period, entity identifier, and the quantity the ordering is built from.
    max_lags
        Highest lag returned, counted in periods of ``time_col``.
    min_entities
        Fewest entities a pair of periods needs in common to contribute.

    Returns
    -------
    pl.DataFrame
        Columns ``lag``, ``rho``, ``rho_p10``, ``rho_p90``, ``band``, ``n_pairs``.
        ``band`` is the correlation a pair of unrelated orderings of the median
        overlap would exceed one time in twenty.
    """
    periods = frame[time_col].unique().sort().to_list()
    ranks = {
        period: dict(
            zip(
                block[entity_col].to_list(),
                block[value_col].rank().to_list(),
                strict=True,
            )
        )
        for period, block in zip(
            periods,
            frame.sort(time_col).partition_by(time_col, maintain_order=True),
            strict=True,
        )
    }

    rows, overlaps = [], []
    for lag in range(1, max_lags + 1):
        correlations = []
        for start, end in zip(periods, periods[lag:], strict=False):
            first, second = ranks[start], ranks[end]
            shared = first.keys() & second.keys()
            if len(shared) < min_entities:
                continue
            order = sorted(shared)
            left = np.array([first[e] for e in order])
            right = np.array([second[e] for e in order])
            # Re-rank inside the overlap: a rank taken over the whole period is not a
            # rank over the entities the two periods share.
            left = np.argsort(np.argsort(left))
            right = np.argsort(np.argsort(right))
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
            overlaps.append(len(shared))
        if not correlations:
            continue
        curve = np.array(correlations)
        rows.append(
            {
                "lag": lag,
                "rho": float(curve.mean()),
                "rho_p10": float(np.percentile(curve, 10)),
                "rho_p90": float(np.percentile(curve, 90)),
                "n_pairs": len(curve),
            }
        )

    if not rows:
        raise ValueError(f"no pair of {time_col} values shares {min_entities} {entity_col} values")

    return pl.DataFrame(rows).with_columns(
        pl.lit(1.96 / np.sqrt(float(np.median(overlaps)))).alias("band")
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
