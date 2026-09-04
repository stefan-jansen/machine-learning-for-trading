"""Every family's prediction artifact carries the same timestamp dtype.

`gbm`, `linear` and `tabular_dl` wrote `Datetime(_, 'UTC')`; `deep_learning` reaches the
registry through `flush_fold_predictions`, whose dates come from a numpy `datetime64`
array and are therefore naive. Measured on crypto_perps_funding (2026-08-28): 578
artifacts UTC-aware and 100 naive - same label, same split, same folds, same 19 symbols,
same 2,189 decision times, identical instants. A tz-aware value never equals a naive one,
so an exact join on (timestamp, symbol) between the two families returned nothing, and
code assuming one dtype across a case study's artifacts dropped rows instead of failing.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.registry.store import _timestamps_as_utc


def _frame(zone: str | None) -> pl.DataFrame:
    stamps = [dt.datetime(2024, 1, 1, 8), dt.datetime(2024, 1, 1, 16)]
    frame = pl.DataFrame({"timestamp": stamps, "symbol": ["BTC", "ETH"], "y_score": [0.1, 0.2]})
    if zone is None:
        return frame
    return frame.with_columns(pl.col("timestamp").dt.replace_time_zone(zone))


def test_a_naive_decision_time_is_localized_not_converted() -> None:
    localized = _timestamps_as_utc(_frame(None))

    assert localized.schema["timestamp"].time_zone == "UTC"
    assert localized["timestamp"].dt.replace_time_zone(None).to_list() == (
        _frame(None)["timestamp"].to_list()
    )


def test_the_two_families_join_after_normalization() -> None:
    naive = _timestamps_as_utc(_frame(None))
    aware = _timestamps_as_utc(_frame("UTC").rename({"y_score": "y_score_other"}))

    assert naive.join(aware, on=["timestamp", "symbol"], how="inner").height == 2


def test_normalizing_does_not_move_the_artifact_digest() -> None:
    """value_digest is zone-insensitive, so a rewritten artifact keeps its identity.

    That is what lets this sit on the writer without any immutable-artifact check
    firing on a prediction set that already exists on disk.
    """
    assert value_digest(_frame(None)) == value_digest(_timestamps_as_utc(_frame(None)))


def test_an_already_aware_frame_and_its_time_unit_are_left_alone() -> None:
    """The unit is deliberately untouched: value_digest *is* time-unit sensitive."""
    aware_ms = _frame("UTC").with_columns(pl.col("timestamp").dt.cast_time_unit("ms"))

    assert _timestamps_as_utc(aware_ms).schema == aware_ms.schema


def test_a_frame_with_no_time_column_passes_through() -> None:
    frame = pl.DataFrame({"symbol": ["BTC"], "y_score": [0.1]})

    assert _timestamps_as_utc(frame).equals(frame)


def test_a_pandas_frame_is_localized_in_place_rather_than_skipped() -> None:
    """The legacy branch and the pandas side of the versioned one never convert.

    Normalizing only Polars frames left those paths writing naive timestamps and recording
    a naive `schema_json`, which is the mismatch this exists to remove.
    """
    import pandas as pd

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 16:00"]),
            "symbol": ["BTC", "ETH"],
            "prediction": [0.1, 0.2],
        }
    )

    localized = _timestamps_as_utc(frame)

    assert str(localized["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert pl.from_pandas(localized).schema["timestamp"].time_zone == "UTC"
    # In place, not converted: the caller's frame type survives.
    assert isinstance(localized, pd.DataFrame)
    # And the instants are unchanged.
    assert localized["timestamp"].dt.tz_localize(None).tolist() == frame["timestamp"].tolist()


def test_an_already_aware_pandas_frame_is_left_alone() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 08:00"]).tz_localize("UTC"),
            "symbol": ["BTC"],
            "prediction": [0.1],
        }
    )

    assert _timestamps_as_utc(frame) is frame


def test_none_passes_through() -> None:
    """`register_prediction_set` accepts predictions=None on the legacy path."""
    assert _timestamps_as_utc(None) is None
