"""The shared-sample IC comparison must read what `load_predictions` actually returns."""

import polars as pl

from case_studies.utils.model_analysis import common_sample_daily_ic


def _frame(rows):
    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.String,
            "timestamp": pl.Date,
            "y_score": pl.Float64,
            "y_true": pl.Float64,
        },
    )


def _rows(dates, symbols, score):
    import datetime as dt

    return [
        {
            "symbol": sym,
            "timestamp": dt.date(2020, 1, day),
            "y_score": score(day, i),
            "y_true": float(i),
        }
        for day in dates
        for i, sym in enumerate(symbols)
    ]


def test_reads_load_predictions_column_names():
    """`prediction`/`actual` do not exist on a `load_predictions` frame."""
    frame = _frame(_rows([1, 2], ["A", "B", "C"], lambda d, i: float(i)))
    ics, n_dates, n_rows = common_sample_daily_ic({"one": frame, "two": frame})
    assert n_dates == 2
    assert n_rows == 6
    # Score ranks match target ranks exactly, so the daily IC is +1.
    assert ics["one"] == 1.0


def test_intersects_on_exact_entity_date_keys():
    """A shared date with a different cross-section must not be compared whole."""
    wide = _frame(_rows([1, 2], ["A", "B", "C"], lambda d, i: float(i)))
    narrow = _frame(_rows([1, 2], ["A", "B"], lambda d, i: float(i)))
    _, n_dates, n_rows = common_sample_daily_ic({"wide": wide, "narrow": narrow})
    assert n_dates == 2
    # Only the two symbols both sets carry, not the three the wider one has.
    assert n_rows == 4


def test_disjoint_sets_yield_no_comparison():
    import datetime as dt

    a = _frame(_rows([1], ["A"], lambda d, i: 1.0))
    b = _frame([{"symbol": "Z", "timestamp": dt.date(2021, 5, 5), "y_score": 1.0, "y_true": 1.0}])
    ics, n_dates, n_rows = common_sample_daily_ic({"a": a, "b": b})
    assert ics == {} and n_dates == 0 and n_rows == 0


def test_single_entity_dates_are_dropped():
    """One entity on a date carries no cross-sectional correlation."""
    one = _frame(_rows([1], ["A"], lambda d, i: 1.0))
    ics, n_dates, _ = common_sample_daily_ic({"only": one})
    assert n_dates == 1
    assert ics["only"] != ics["only"]  # NaN: no date survived the >= 2 entity filter
