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

    The lag is taken within an entity, so no pair spans two entities, but the
    correlation is computed over the pooled pairs. A single-entity estimate is a
    claim about that entity, and the two disagree most at the lag that matters.
    """
    return np.array(
        [
            frame.with_columns(pl.col(column).shift(-lag).over(entity_col).alias("_lagged"))
            .select(pl.corr(column, "_lagged"))
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
    """
    rows, weight = 0, 0.0
    for _, group in frame.sort(timestamp_col).group_by([entity_col], maintain_order=True):
        n = group.height
        events = np.arange(n)
        weights = calculate_label_uniqueness(events, np.minimum(events + horizon, n), n_bars=n)
        rows += n
        weight += float(weights.sum())
    return rows, weight
