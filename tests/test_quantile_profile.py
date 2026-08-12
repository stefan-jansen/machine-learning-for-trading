"""Tests for the shared quantile-profile helper in case_studies/utils/feature_engineering.py.

Pins:
- the two averages are taken in order - across assets inside one decision time, then across
  decision times - so a decision time quoting many assets does not outweigh one quoting few.
  Pooling every row into a single average instead was the construction in five of the nine
  stage-05 notebooks, and it is silent: the profile still comes out, weighted by cross-section
  size rather than by time.
- quantiles are assigned inside each decision time, so a period when the whole panel sat high
  does not fill the top quantile with that period's assets.
- a decision time below the cross-section floor, or carrying fewer distinct feature values than
  quantiles, is dropped rather than split on ties.
- the excess-return mode takes the label relative to its own decision time's average before the
  quantiles are formed.

Eight stage-05 notebooks call this, so a regression here is a regression in eight rendered
pages at once.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from case_studies.utils.feature_engineering import quantile_profile


def _panel(rows: list[tuple[date, str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["timestamp", "symbol", "feature", "label"], orient="row")


def _two_sessions_of_different_widths() -> pl.DataFrame:
    """One wide session where the feature works, one narrow session where it reverses.

    The wide session carries 50 assets and the narrow one 5, and their profiles are equal
    and opposite. Weighted by row the wide session decides the answer; weighted by session
    the two cancel exactly.
    """
    wide, narrow = date(2020, 1, 2), date(2020, 1, 3)
    rows = []
    for i in range(50):
        rows.append((wide, f"W{i}", float(i), float(i) / 50.0))
    for i in range(5):
        rows.append((narrow, f"N{i}", float(i), -float(i) / 5.0))
    return _panel(rows)


def test_sessions_are_weighted_equally_not_by_cross_section() -> None:
    frame = _two_sessions_of_different_widths()
    profile = quantile_profile(frame, "feature", "label", date_col="timestamp")
    assert profile is not None

    pooled = (
        frame.with_columns(
            ((pl.col("feature").rank().over("timestamp") - 1) / pl.len().over("timestamp") * 5)
            .floor()
            .clip(0, 4)
            .alias("quantile")
        )
        .group_by("quantile")
        .agg(pl.col("label").mean())
        .sort("quantile")["label"]
        .to_list()
    )

    # The two sessions are built to cancel: the wide one climbs 0.8 across the
    # quantiles and the narrow one falls 0.8, so a diagnostic that weights them
    # equally reports a feature with no usable shape at all.
    assert profile.spread == pytest.approx(0.0, abs=1e-9)
    # Pooling lets the 50-asset session outvote the 5-asset one ten to one, so the
    # reversal barely registers and the same feature reads as a clean staircase.
    assert pooled[-1] - pooled[0] == pytest.approx(0.6545, abs=1e-3)


def test_quantiles_are_cut_inside_each_decision_time() -> None:
    """A level shift between sessions must not decide which quantile an asset lands in.

    Session two sits entirely above session one. Cut over the pooled sample, every
    session-two asset lands in the top quantiles; cut within the session, each session
    fills all five.
    """
    rows = []
    for offset, base in ((0, 0.0), (1, 100.0)):
        when = date(2020, 1, 2) + timedelta(days=offset)
        for i in range(10):
            rows.append((when, f"S{i}", base + float(i), float(i)))
    frame = _panel(rows)

    profile = quantile_profile(frame, "feature", "label", date_col="timestamp")
    assert profile is not None
    assert profile.periods_used == 2
    # Both sessions rank their assets identically, so the profile is the label ladder
    # itself: quantile k holds ranks 2k and 2k+1, averaging 0.5, 2.5, 4.5, 6.5, 8.5.
    assert profile.means == pytest.approx([0.5, 2.5, 4.5, 6.5, 8.5])
    assert profile.monotonicity == pytest.approx(1.0)


def test_thin_and_tied_decision_times_are_dropped() -> None:
    """Below the floor, or with fewer distinct values than quantiles, there is no split."""
    wide = date(2020, 1, 2)
    thin = date(2020, 1, 3)
    tied = date(2020, 1, 6)
    rows = [(wide, f"W{i}", float(i), float(i)) for i in range(10)]
    rows += [(thin, f"T{i}", float(i), float(i)) for i in range(3)]
    rows += [(tied, f"C{i}", 1.0 if i % 2 else 0.0, float(i)) for i in range(10)]
    frame = _panel(rows)

    profile = quantile_profile(frame, "feature", "label", date_col="timestamp")
    assert profile is not None
    assert profile.periods_available == 3
    assert profile.periods_used == 1


def test_a_feature_with_no_usable_decision_time_has_no_profile() -> None:
    rows = [(date(2020, 1, 2), f"A{i}", 1.0, float(i)) for i in range(10)]
    assert quantile_profile(_panel(rows), "feature", "label", date_col="timestamp") is None


def test_min_cross_section_is_a_floor_over_the_quantile_count() -> None:
    rows = [(date(2020, 1, 2), f"A{i}", float(i), float(i)) for i in range(10)]
    frame = _panel(rows)
    assert quantile_profile(frame, "feature", "label", date_col="timestamp") is not None
    assert (
        quantile_profile(frame, "feature", "label", date_col="timestamp", min_cross_section=20)
        is None
    )


def test_excess_mode_removes_the_decision_time_average() -> None:
    """Every quantile earns the period's drift; only the difference is collectable."""
    rows = []
    for offset, drift in ((0, 0.0), (1, 10.0)):
        when = date(2020, 1, 2) + timedelta(days=offset)
        for i in range(10):
            rows.append((when, f"S{i}", float(i), drift + float(i)))
    frame = _panel(rows)

    raw = quantile_profile(frame, "feature", "label", date_col="timestamp")
    excess = quantile_profile(
        frame, "feature", "label", date_col="timestamp", demean_within_date=True
    )
    assert raw is not None and excess is not None
    # The drift lifts every quantile by the same 5.0 and leaves the spread untouched.
    assert raw.means == pytest.approx([5.5, 7.5, 9.5, 11.5, 13.5])
    assert excess.means == pytest.approx([-4.0, -2.0, 0.0, 2.0, 4.0])
    assert excess.spread == pytest.approx(raw.spread)
