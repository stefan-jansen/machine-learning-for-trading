"""Panel diagnostics for overlapping labels, shared across the case studies.

Both statistics here answer the same question - how much independent information a
per-bar label with a multi-bar horizon actually carries - and both are wrong in the
same way when computed carelessly: on one entity rather than the panel, or with the
concurrency of overlapping windows ignored.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from ml4t.engineer.labeling import calculate_label_uniqueness


def panel_autocorrelation(
    frame: pl.DataFrame,
    column: str,
    *,
    max_lag: int,
    entity_col: str = "symbol",
) -> np.ndarray:
    """Autocorrelation of *column* at lags 1..max_lag, pooled across entities.

    The lag is taken within an entity, so no pair spans two entities, and the
    column is demeaned within its entity before pooling. Without the demeaning a
    panel whose entities sit at different levels reports that level dispersion as
    persistence: a series that is constant inside every entity, and so has no
    autocorrelation to speak of, would come back at 1.0.

    A single-entity estimate is a claim about that entity, and the two disagree
    most at the lag that matters - the label horizon.
    """
    centred = frame.with_columns(
        (pl.col(column) - pl.col(column).mean().over(entity_col)).alias("_centred")
    )
    return np.array(
        [
            centred.with_columns(pl.col("_centred").shift(-lag).over(entity_col).alias("_lagged"))
            .select(pl.corr("_centred", "_lagged"))
            .item()
            for lag in range(1, max_lag + 1)
        ]
    )


def effective_sample_size(
    frame: pl.DataFrame,
    *,
    horizon: int,
    entity_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> tuple[int, float]:
    """Return (rows, N_eff) for a label sampled every bar over *horizon* bars.

    ``N_eff`` is Chapter 7.2's average-uniqueness sum: each row is weighted by the
    share of its forward window no concurrent label also spans. Concurrency is a
    property of one entity's overlapping windows, so the weights are computed per
    entity and summed.

    *frame* is expected to hold only rows with a non-null label, so every row has a
    complete ``horizon``-bar forward window even though the bars closing the last
    ``horizon`` windows are not themselves rows of *frame*. The endpoints are
    therefore left uncapped and the concurrency array is extended to ``n + horizon``.

    Capping the endpoints at ``n`` instead leaves concurrency on the retained bars
    exactly as it was - a label covers bar ``t < n`` under both conventions - and
    changes only the boundary windows, by discarding their closing bars. Those bars
    are the tail, where the fewest windows are still open, so they carry the largest
    ``1/c_t`` terms in the average; dropping them removes a window's most unique
    part and lowers its weight. Capping therefore *understates* ``N_eff`` whenever
    ``horizon <= n`` - checked exhaustively for every such pair up to ``n = 59``,
    and 19,064 against 19,112 on the etfs monthly label.

    It reverses once the horizon is long relative to the group, where capping
    removes most of every window rather than a tail: at ``n = 2, horizon = 4`` the
    capped sum is 1.25 against 1.20. A group here is one entity's *labelled* rows,
    so a symbol with barely more bars than the horizon reaches that regime, and no
    directional claim should be made about it.
    """
    rows, weight = 0, 0.0
    for _, group in frame.sort(timestamp_col).group_by([entity_col], maintain_order=True):
        n = group.height
        events = np.arange(n)
        weights = calculate_label_uniqueness(events, events + horizon, n_bars=n + horizon)
        rows += n
        weight += float(weights.sum())
    return rows, weight
