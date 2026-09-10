"""A prediction stamped on a different clock than its label still scores.

`compute_cross_sectional_direction_auc` aligned a `Date` label against a `Datetime`
prediction and refused everything else, so two `Datetime`s differing only in time unit
raised into the caller's bare `except` and the metric was never written. Measured on
`crypto_perps_funding`: 120 of 120 `deep_learning` validation sets on the primary label
carried a NULL `direction_label`, against 0 of 202 for gbm, linear and tabular_dl.

The instants were identical. `deep_learning` and `latent_factors` reach the registry
through `flush_fold_predictions`, whose dates come from a numpy `datetime64` array, so
their artifacts carry `us` where every label and feature panel carries `ms` - 1,548
artifacts across seven case studies.

**The cast may only ever go onto the prediction side.** `value_digest` is time-unit
sensitive and zone-insensitive, so rewriting a prediction frame's unit moves
`computation.expected_prediction_keys.digest` and the artifact stops reproducing the digest
its own spec records; 110 registered training runs sit on that. `registry/store.py::
_timestamps_as_utc` normalised the zone for exactly that reason and says in as many words
that it leaves the unit alone. Reading is the only side that may cast, which is what these
tests pin.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.registry.metrics import (
    _cast_to_label_clock,
    compute_cross_sectional_direction_auc,
)

# `cross_sectional_auc_series` needs `min_obs` rows per date and both classes present, so a
# fixture below eight symbols yields no AUC on any date and the function correctly returns {}.
_SYMBOLS = ("BTC", "ETH", "SOL", "ADA", "DOT", "AVAX", "LINK", "XRP")


def _panel(n: int) -> tuple[list[dt.datetime], list[str]]:
    start = dt.datetime(2024, 1, 1)
    stamps = [start + dt.timedelta(hours=8 * i) for i in range(n)]
    return (
        [s for s in stamps for _ in _SYMBOLS],
        [sym for _ in stamps for sym in _SYMBOLS],
    )


def _predictions(unit: str, zone: str | None, n: int = 200) -> pl.DataFrame:
    """`n` decision times over eight symbols, scored by rank within each date."""
    stamps, symbols = _panel(n)
    frame = pl.DataFrame(
        {
            "timestamp": stamps,
            "symbol": symbols,
            "y_score": [j / len(_SYMBOLS) for _ in range(n) for j in range(len(_SYMBOLS))],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime(unit)))
    if zone:
        frame = frame.with_columns(pl.col("timestamp").dt.replace_time_zone(zone))
    return frame


def _direction(unit: str, zone: str | None, n: int = 200) -> pl.DataFrame:
    """The sibling direction label: the top half of each date's cross-section went up."""
    stamps, symbols = _panel(n)
    frame = pl.DataFrame(
        {
            "timestamp": stamps,
            "symbol": symbols,
            "fwd_dir_8h": [
                1 if j >= len(_SYMBOLS) // 2 else 0 for _ in range(n) for j in range(len(_SYMBOLS))
            ],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime(unit)))
    if zone:
        frame = frame.with_columns(pl.col("timestamp").dt.replace_time_zone(zone))
    return frame


def test_microsecond_predictions_score_against_millisecond_labels():
    """The exact shape that lost 120 runs."""
    out = compute_cross_sectional_direction_auc(
        _predictions("us", "UTC"), _direction("ms", "UTC"), direction_col="fwd_dir_8h", horizon=8
    )

    assert out, "a us/ms pair produced no metric block"
    assert out["direction_label"] == "fwd_dir_8h"
    assert 0.0 <= out["auc_mean_daily"] <= 1.0


def test_same_clock_still_scores():
    out = compute_cross_sectional_direction_auc(
        _predictions("ms", "UTC"), _direction("ms", "UTC"), direction_col="fwd_dir_8h", horizon=8
    )

    assert out["direction_label"] == "fwd_dir_8h"


def test_the_two_clocks_agree_on_the_answer():
    """Aligning the clock must not change the measurement, only make it possible."""
    same = compute_cross_sectional_direction_auc(
        _predictions("ms", "UTC"), _direction("ms", "UTC"), direction_col="fwd_dir_8h", horizon=8
    )
    across = compute_cross_sectional_direction_auc(
        _predictions("us", "UTC"), _direction("ms", "UTC"), direction_col="fwd_dir_8h", horizon=8
    )

    assert across["auc_mean_daily"] == pytest.approx(same["auc_mean_daily"])
    assert across["auc_n_days"] == same["auc_n_days"]


def test_a_narrowing_cast_that_would_truncate_is_refused():
    """Silence here would be a join that quietly matches fewer rows."""
    finer = _predictions("us", "UTC", n=10).with_columns(
        pl.col("timestamp") + pl.duration(microseconds=1)
    )

    with pytest.raises(ValueError, match="without losing precision"):
        _cast_to_label_clock(finer, "timestamp", pl.Datetime("ms", "UTC"))


def test_a_lossless_narrowing_cast_is_allowed():
    coarse = _predictions("us", "UTC", n=10)
    out = _cast_to_label_clock(coarse, "timestamp", pl.Datetime("ms", "UTC"))

    assert out.schema["timestamp"] == pl.Datetime("ms", "UTC")
    assert out.get_column("timestamp").to_list() == coarse.get_column("timestamp").to_list()


def test_value_digest_is_unit_sensitive_and_zone_insensitive():
    """Why the cast goes onto the prediction side at read time and never onto the artifact.

    If this ever flips, the reasoning above it is void: a unit-insensitive digest would make
    rewriting the artifacts free, and a zone-sensitive one would mean the 2026-08-28 zone
    normalisation had silently moved 110 registered identities.
    """
    stamps = [dt.datetime(2024, 1, 1, 8), dt.datetime(2024, 1, 1, 16)]
    naive_us = pl.DataFrame({"timestamp": stamps, "symbol": ["BTC", "ETH"], "fold": [0, 0]})
    utc_us = naive_us.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
    utc_ms = utc_us.with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC")))
    keys = ("symbol", "timestamp", "fold")

    assert value_digest(naive_us, keys) == value_digest(utc_us, keys), "digest gained a zone"
    assert value_digest(utc_us, keys) != value_digest(utc_ms, keys), "digest lost its unit"


def test_a_date_label_still_narrows_the_prediction_stamp():
    """The pre-existing alignment, which the new branch must not shadow."""
    daily = _direction("ms", None).with_columns(pl.col("timestamp").cast(pl.Date))
    out = compute_cross_sectional_direction_auc(
        _predictions("us", None), daily, direction_col="fwd_dir_8h", horizon=8
    )

    assert out["direction_label"] == "fwd_dir_8h"


def test_an_unalignable_key_still_raises():
    """Only Datetime-to-Datetime and Date-to-Datetime are alignable; the rest must say so."""
    text_keyed = _direction("ms", "UTC").with_columns(
        pl.col("timestamp").dt.strftime("%Y-%m-%dT%H:%M:%S")
    )

    with pytest.raises(TypeError, match="cannot align join key"):
        compute_cross_sectional_direction_auc(
            _predictions("us", "UTC"), text_keyed, direction_col="fwd_dir_8h", horizon=8
        )
