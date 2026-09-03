"""The daily IC series `90_ic_diagnostic` measures every feature against.

Extracted from the notebook so the two ways an IC series comes back empty can be tested
directly. They need different answers and the distinction is invisible in the result:
`cross_sectional_ic_series` writes a null `ic` both for a date that carried fewer than
`min_obs` valid observations and for a date that carried enough with every prediction or
every return tied, and dropping the nulls collapses the two into one empty frame.
"""

from __future__ import annotations

import polars as pl
from ml4t.diagnostic.metrics import cross_sectional_ic_series


def daily_ic(
    panel: pl.DataFrame,
    *,
    pred_col: str,
    ret_col: str,
    min_symbols_per_date: int,
    described_as: str,
) -> pl.DataFrame:
    """The scored daily IC series, or a refusal naming which of the two causes emptied it.

    Attributing an undefined correlation to insufficient breadth sends a reader to widen a
    panel that is already wide enough, and the panel then stays wrong for a reason nobody is
    looking at. `n_obs` separates the two, so this reports the one that actually happened.
    """
    series = cross_sectional_ic_series(
        panel,
        panel,
        pred_col=pred_col,
        ret_col=ret_col,
        date_col="timestamp",
        entity_col="symbol",
        min_obs=min_symbols_per_date,
    )
    scored = series.drop_nulls("ic")
    if scored.height:
        return scored
    wide_enough = series.filter(pl.col("n_obs") >= min_symbols_per_date).height
    if not wide_enough:
        raise RuntimeError(
            f"no date in the validation panel carries {min_symbols_per_date} names, so "
            f"{described_as} has no daily IC series to average; the panel spans "
            f"{panel['symbol'].n_unique()} names across {series.height} dates in total"
        )
    raise RuntimeError(
        f"{wide_enough} of {series.height} dates carry {min_symbols_per_date} names, so "
        f"breadth is not the problem: the rank correlation for {described_as} is undefined on "
        "every one of them, which is what a feature or a target that is constant within the "
        "date produces"
    )
