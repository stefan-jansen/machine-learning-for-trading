"""A `Date` and a `Datetime` of the same instant have to compare equal as coverage keys.

`_canonical_key_frame` cast every key column but `fold_id` straight to `String`, which
renders the dtype rather than the instant: `2016-01-29` against
`2016-01-29 00:00:00.000`. The two never join, so a registry frame meeting a dataset frame
from another loader reported every expected row missing and every actual row extra.

A 100% mismatch in both directions is the signature. It reads as a data problem, which is
why it is worth pinning: the number that would send someone looking at the data is produced
entirely by the dtypes.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from case_studies.utils.registry.completeness import evaluate_prediction_coverage

DAYS = [dt.date(2016, 1, 29), dt.date(2016, 2, 1), dt.date(2016, 2, 2)]


def _keys(timestamps: pl.Series) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "timestamp": timestamps,
            "fold_id": [0, 0, 1],
        }
    )


def _dates() -> pl.Series:
    return pl.Series("timestamp", DAYS, dtype=pl.Date)


def _datetimes(unit: str) -> pl.Series:
    return pl.Series(
        "timestamp", [dt.datetime.combine(d, dt.time()) for d in DAYS], dtype=pl.Datetime(unit)
    )


def _predictions(timestamps: pl.Series) -> pl.DataFrame:
    return _keys(timestamps).with_columns(y_score=pl.Series([0.1, 0.2, 0.3]))


def test_a_date_expectation_matches_millisecond_datetime_predictions() -> None:
    coverage = evaluate_prediction_coverage(_keys(_dates()), _predictions(_datetimes("ms")))

    assert coverage.n_missing == 0
    assert coverage.n_extra == 0


def test_the_two_datetime_units_of_one_instant_also_match() -> None:
    coverage = evaluate_prediction_coverage(_keys(_datetimes("us")), _predictions(_datetimes("ns")))

    assert coverage.n_missing == 0
    assert coverage.n_extra == 0


def test_a_genuinely_different_day_is_still_reported() -> None:
    """The normalization must not make everything match."""
    shifted = pl.Series("timestamp", [d + dt.timedelta(days=7) for d in DAYS], dtype=pl.Date)

    coverage = evaluate_prediction_coverage(_keys(_dates()), _predictions(shifted))

    assert coverage.n_missing == 3
    assert coverage.n_extra == 3
