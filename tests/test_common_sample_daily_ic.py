"""The shared-sample IC comparison must read what `load_predictions` actually returns."""

from datetime import date

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


# --- one registry, two conventions for the same trading day -------------------------
#
# etfs stores `timestamp` as Date in 277 of its prediction files and as midnight Datetime
# in 123, because the families were registered by notebooks written months apart. Polars
# raises SchemaError on a join whose key dtypes differ rather than returning nothing, so
# comparing across those families failed outright at `09_dl_lstm`. The stored predictions
# are correct; only the reader was wrong.


def _pred(as_datetime: bool, *, shift: float = 0.0, hours: int = 0) -> pl.DataFrame:
    days = [date(2024, 1, d) for d in range(1, 6)]
    ts = pl.Series("timestamp", days * 4)
    if as_datetime:
        ts = ts.cast(pl.Datetime("us"))
    frame = pl.DataFrame(
        {
            "symbol": [s for s in "ABCD" for _ in days],
            "timestamp": ts,
            "y_score": [float(i) + shift for i in range(20)],
            "y_true": [float(i % 7) for i in range(20)],
        }
    )
    if hours:
        frame = frame.with_columns(pl.col("timestamp") + pl.duration(hours=hours))
    return frame


def test_date_and_midnight_datetime_intersect_as_the_same_day():
    ics, n_dates, n_rows = common_sample_daily_ic(
        {"flat": _pred(False), "lstm": _pred(True, shift=0.5)}
    )
    assert (n_dates, n_rows) == (5, 20)
    assert set(ics) == {"flat", "lstm"}


def test_a_single_dtype_is_left_alone():
    _, n_dates, n_rows = common_sample_daily_ic({"a": _pred(False), "b": _pred(False, shift=0.5)})
    assert (n_dates, n_rows) == (5, 20)


def test_a_real_time_of_day_is_never_truncated_into_a_match():
    """The failure the widening direction exists to avoid.

    Narrowing Datetime to Date would make 14:00 equal to midnight and report a shared
    cross-section that does not exist. An empty intersection is the true answer about two
    prediction sets that share no key.
    """
    _, n_dates, n_rows = common_sample_daily_ic(
        {"flat": _pred(False), "intraday": _pred(True, hours=14)}
    )
    assert (n_dates, n_rows) == (0, 0)
