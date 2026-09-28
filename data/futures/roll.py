"""Roll factors for ratio back-adjustment of continuous futures series.

A continuous series switches from one contract to the next between two consecutive rows. The
back-adjustment factor at that switch has to compare the two contracts at the same moment:
the ratio of their prices then is the gap the switch introduces. Dividing the new contract's
first price by the old contract's last price mixes two moments whenever the rows are not
adjacent in time, and at a switch across a weekend the factor absorbs the whole weekend's
move, so the adjusted series shows no return where a holder earned one.

The same-moment pair comes from the other tenors of the same product: on the hour a front
month rolls, the contract it rolls into is usually quoted as the first deferred.
"""

from __future__ import annotations

import polars as pl

SAME_HOUR_AT_LAST_OLD_BAR = "same_hour_at_last_old_bar"
SAME_HOUR_AT_FIRST_NEW_BAR = "same_hour_at_first_new_bar"
DIFFERENT_TIMES = "different_times"


def roll_factors(hourly: pl.DataFrame) -> pl.DataFrame:
    """Return one row per contract switch in each (product, tenor) series.

    ``hourly`` holds continuous hourly bars with ``product``, ``tenor``, ``timestamp``,
    ``instrument_id``, ``open`` and ``close``. A switch is a row whose ``instrument_id``
    differs from the previous row of its series; ``timestamp`` in the result is that row's,
    which is where ``ratio`` applies: bars before it are multiplied by it.

    The ratio is the new contract's close over the old contract's close on one hour, taken
    from any tenor of the product:

    - on the old contract's last bar in the series, when the new contract traded then; the
      return across the switch is then the new contract's;
    - otherwise on the new contract's first bar, when the old contract traded then; the
      return across the switch is then the old contract's;
    - otherwise no same-hour pair exists in ``hourly``, and the ratio falls back to the new
      contract's first open over the old contract's last close, with ``basis`` set to
      ``"different_times"`` and ``gap_hours`` saying how far apart those two prices are.
    """
    series = hourly.sort("product", "tenor", "timestamp").with_columns(
        pl.col("instrument_id").shift(1).over("product", "tenor").alias("old_instrument_id"),
        pl.col("timestamp").shift(1).over("product", "tenor").alias("old_timestamp"),
        pl.col("close").shift(1).over("product", "tenor").alias("old_close"),
    )
    switches = series.filter(
        pl.col("old_instrument_id").is_not_null()
        & (pl.col("instrument_id") != pl.col("old_instrument_id"))
    ).select(
        "product",
        "tenor",
        "timestamp",
        "old_timestamp",
        "old_instrument_id",
        pl.col("instrument_id").alias("new_instrument_id"),
        "old_close",
        pl.col("open").alias("new_open"),
        pl.col("close").alias("new_close"),
    )

    quotes = hourly.select("product", "instrument_id", "timestamp", "close").unique(
        ["product", "instrument_id", "timestamp"], keep="first"
    )
    new_at_old = quotes.rename(
        {
            "instrument_id": "new_instrument_id",
            "timestamp": "old_timestamp",
            "close": "new_close_at_old",
        }
    )
    old_at_new = quotes.rename({"instrument_id": "old_instrument_id", "close": "old_close_at_new"})
    switches = switches.join(
        new_at_old, on=["product", "new_instrument_id", "old_timestamp"], how="left"
    ).join(old_at_new, on=["product", "old_instrument_id", "timestamp"], how="left")

    at_old = pl.col("new_close_at_old").is_not_null()
    at_new = pl.col("old_close_at_new").is_not_null()
    return switches.with_columns(
        pl.when(at_old)
        .then(pl.col("new_close_at_old") / pl.col("old_close"))
        .when(at_new)
        .then(pl.col("new_close") / pl.col("old_close_at_new"))
        .otherwise(pl.col("new_open") / pl.col("old_close"))
        .alias("ratio"),
        pl.when(at_old)
        .then(pl.col("old_timestamp"))
        .when(at_new)
        .then(pl.col("timestamp"))
        .otherwise(None)
        .alias("pair_timestamp"),
        pl.when(at_old)
        .then(pl.lit(SAME_HOUR_AT_LAST_OLD_BAR))
        .when(at_new)
        .then(pl.lit(SAME_HOUR_AT_FIRST_NEW_BAR))
        .otherwise(pl.lit(DIFFERENT_TIMES))
        .alias("basis"),
        ((pl.col("timestamp") - pl.col("old_timestamp")).dt.total_minutes() / 60).alias(
            "gap_hours"
        ),
    ).select(
        "product",
        "tenor",
        "timestamp",
        "old_instrument_id",
        "new_instrument_id",
        "pair_timestamp",
        "ratio",
        "basis",
        "gap_hours",
    )
