"""Panel diagnostics for overlapping labels, shared across the case studies.

Both statistics here answer the same question - how much independent information a
per-bar label with a multi-bar horizon actually carries - and both are wrong in the
same three ways when computed carelessly: on one entity rather than the panel, with
the concurrency of overlapping windows ignored, or with the frame's row order
mistaken for the grid the horizon is counted in.

The third is why both take `bar_col`. A diagnostics frame usually holds only rows
with a non-null label, and where a bar is missing - an outage, a settlement an
exchange skipped, a symbol that had not listed - the surviving rows close over the
hole. Counting positions among survivors then makes the two rows either side of a
hole adjacent, so windows that share nothing appear to overlap and windows `lag`
apart on the grid are pooled with windows further apart. `bar_col` names each row's
position on the grid the label's horizon is measured in, which the caller builds
from the frame the label was built on, before any row was dropped. Only differences
within an entity are read, so any affine origin will do.
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
    bar_col: str,
    entity_col: str = "symbol",
) -> np.ndarray:
    """Autocorrelation of *column* at lags 1..max_lag, pooled across entities.

    A pair is kept only if both rows belong to the same entity and their `bar_col`
    positions differ by exactly the lag, so no pair spans two entities and none
    spans a hole in the grid. The column is demeaned within its entity before
    pooling: without the demeaning a panel whose entities sit at different levels
    reports that level dispersion as persistence, and a series that is constant
    inside every entity - so with no autocorrelation to speak of - would come back
    at 1.0.

    A single-entity estimate is a claim about that entity, and the two disagree
    most at the lag that matters - the label horizon. A lag with no surviving pair
    is reported as NaN rather than dropped, so the returned array always has
    `max_lag` entries and the lag axis of a figure drawn from it stays honest.
    """
    centred = frame.select(
        entity_col,
        pl.col(bar_col).alias("_bar"),
        (pl.col(column) - pl.col(column).mean().over(entity_col)).alias("_centred"),
    )
    out = []
    for lag in range(1, max_lag + 1):
        lagged = centred.select(
            entity_col,
            (pl.col("_bar") - lag).alias("_bar"),
            pl.col("_centred").alias("_lagged"),
        )
        pairs = centred.join(lagged, on=[entity_col, "_bar"], how="inner")
        value = pairs.select(pl.corr("_centred", "_lagged")).item() if pairs.height else None
        out.append(np.nan if value is None else value)
    return np.array(out, dtype=float)


def effective_sample_size(
    frame: pl.DataFrame,
    *,
    bar_col: str,
    horizon: int | None = None,
    horizon_col: str | None = None,
    entity_col: str = "symbol",
) -> tuple[int, float]:
    """Return (rows, N_eff) for a label sampled every bar over *horizon* bars.

    Pass ``horizon_col`` instead of ``horizon`` where the window is not the same length
    for every row - an event label that resolves when a barrier is hit or when a contract
    expires. The column holds each row's window in the same units as ``bar_col``, and a
    single ``horizon`` is the special case where every row carries the same value. A
    median window standing in for a variable one prices the overlap of a label none of
    the rows has.

    ``N_eff`` is Chapter 7.2's average-uniqueness sum: each row is weighted by the
    share of its forward window no concurrent label also spans. Concurrency is a
    property of one entity's overlapping windows, so the weights are computed per
    entity and summed, over the entity's own grid positions - a window that starts
    on the far side of a hole is concurrent with nothing on the near side.

    **What a label occupies is ``horizon`` return intervals, not ``horizon + 1``
    bars.** The label at bar *i* is $P_{i+h}/P_i - 1$, so it consumes the returns
    realised over bars $i{+}1 \\ldots i{+}h$ - *h* of them - and the label at *i+1*
    shares $h-1$ of those, which is the overlap the audit record prints. Passing a
    closed bar interval ``[i, i+h]`` instead counts the anchor bar as consumed and
    makes every label span ``h+1`` units, so consecutive labels appear to share one
    interval even when they share none.

    The one-session horizon is the case that settles it: consecutive one-day
    forward returns are built from disjoint returns and are fully independent, so
    every weight must be 1 and ``N_eff`` must equal ``N``. The closed-bar form
    returns ``N/2`` there. On a gapless grid average uniqueness converges to
    ``1/h``, so ``N_eff`` tends to ``N/h`` - the reference value the stage standard
    cites - and a grid with holes sits above it, because a hole ends an overlap
    early.

    *frame* is expected to hold only rows with a non-null label, so every row has a
    complete forward window even though the bars closing the last few are not
    themselves rows of *frame*; the endpoints are left uncapped and the concurrency
    array extended past the last window's end rather than truncated, which would
    shorten exactly those windows.
    """
    if (horizon is None) == (horizon_col is None):
        raise ValueError("pass exactly one of horizon and horizon_col")
    # `maintain_order=True` is what makes the total reproducible. Summing floats is not
    # associative, and polars does not fix the order groups come back in, so the same frame
    # summed twice differs in the last bits. Printed as an integer that lands on either side
    # of a rounding boundary: sp500_options' fwd_ret_10d reported N_eff 39,746 on one run and
    # 39,747 on the next, from identical inputs and an unchanged label digest.
    rows, weight = 0, 0.0
    for _, group in frame.group_by([entity_col], maintain_order=True):
        bars = group[bar_col].to_numpy()
        order = np.argsort(bars)
        events = bars[order] - bars.min()
        windows = horizon if horizon_col is None else group[horizon_col].to_numpy()[order]
        ends = events + windows - 1
        weights = calculate_label_uniqueness(events, ends, n_bars=int(ends.max()) + 1)
        rows += group.height
        weight += float(weights.sum())
    return rows, weight
