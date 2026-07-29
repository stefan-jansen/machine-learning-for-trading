"""Panel autocorrelation and effective sample size, against known answers.

Both statistics are cheap to compute wrongly in a way that looks plausible: an
autocorrelation that silently pairs one entity's last row with the next entity's
first, and an effective sample size that returns the row count when the windows
do not in fact overlap.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from ml4t.engineer.labeling import calculate_label_uniqueness

from case_studies.utils.label_diagnostics import effective_sample_size, panel_autocorrelation


def _panel(n_per_symbol: int = 60, symbols: tuple[str, ...] = ("A", "B")) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "symbol": [s for s in symbols for _ in range(n_per_symbol)],
            "timestamp": list(range(n_per_symbol)) * len(symbols),
            "x": rng.normal(size=n_per_symbol * len(symbols)),
        }
    )


def test_autocorrelation_of_a_repeating_series_is_one_at_the_period() -> None:
    period = 4
    frame = pl.DataFrame(
        {
            "symbol": ["A"] * 40,
            "timestamp": list(range(40)),
            "x": [float(i % period) for i in range(40)],
        }
    )
    acf = panel_autocorrelation(frame, "x", max_lag=period)
    assert acf[period - 1] == 1.0
    assert acf[0] < 0


def test_level_differences_between_entities_are_not_persistence() -> None:
    """A is constant and B is constant at another level: there is no within-entity
    variation, so there is no autocorrelation to report. Pooling raw values would
    return 1.0 here, reading the cross-sectional level gap as persistence."""
    frame = pl.DataFrame(
        {
            "symbol": ["A"] * 10 + ["B"] * 10,
            "timestamp": list(range(10)) * 2,
            "x": [1.0] * 10 + [5.0] * 10,
        }
    )
    acf = panel_autocorrelation(frame, "x", max_lag=3)
    assert np.all(np.isnan(acf)), acf


def test_a_one_session_horizon_loses_no_information() -> None:
    """The case that fixes the convention.

    A one-session forward return at t is built from the single return realised
    between t and t+1, and the one at t+1 from the next. They share nothing, so
    every label is fully unique and N_eff equals N. Treating a horizon-h label as
    a closed bar interval [i, i+h] instead makes it span h+1 units, so consecutive
    labels appear to share one even here, and this returns N/2.
    """
    frame = _panel(n_per_symbol=50)
    rows, n_eff = effective_sample_size(frame, horizon=1)
    assert rows == 100
    assert n_eff == pytest.approx(rows)


def test_overlapping_windows_shrink_the_effective_count_toward_n_over_h() -> None:
    horizon = 10
    frame = _panel(n_per_symbol=1000)
    rows, n_eff = effective_sample_size(frame, horizon=horizon)
    assert rows == 2000
    assert n_eff < rows
    # A label spans h return intervals and its neighbour shares h-1 of them, so
    # average uniqueness tends to 1/h - the reference the stage standard cites.
    assert n_eff / rows == pytest.approx(1 / horizon, abs=0.005)


@pytest.mark.parametrize(
    ("n", "horizon", "expected"),
    [
        # h=1: each label occupies one return interval and no two coincide, so every
        # weight is 1.
        (3, 1, 3.0),
        # n=4, h=2: intervals [0,1], [1,2], [2,3], [3,4] over bars 0..4, so
        # concurrency is (1, 2, 2, 2, 1) and the weights are (3/4, 1/2, 1/2, 3/4).
        (4, 2, 2.5),
        # n=5, h=3: weights (11/18, 7/18, 1/3, 7/18, 11/18).
        (5, 3, 7 / 3),
    ],
)
def test_effective_sample_size_matches_uniqueness_computed_by_hand(
    n: int, horizon: int, expected: float
) -> None:
    """Pins both conventions at once: h intervals per label, and complete windows
    at the boundary rather than endpoints truncated at the row count."""
    frame = _panel(n_per_symbol=n, symbols=("A",))
    rows, n_eff = effective_sample_size(frame, horizon=horizon)
    assert rows == n
    assert n_eff == pytest.approx(expected)


def test_the_anchor_bar_is_not_counted_as_consumed() -> None:
    """Regression guard for the off-by-one this helper was built with.

    Counting the anchor makes a label span horizon + 1 units, which inflates the
    apparent overlap by exactly one interval at every horizon and drives N_eff to
    N/(h+1) instead of N/h.
    """
    n, horizon = 400, 4
    ev = np.arange(n)
    correct = calculate_label_uniqueness(ev, ev + horizon - 1, n_bars=n + horizon - 1).sum()
    with_anchor = calculate_label_uniqueness(ev, ev + horizon, n_bars=n + horizon).sum()
    assert correct / n == pytest.approx(1 / horizon, abs=0.005)
    assert with_anchor / n == pytest.approx(1 / (horizon + 1), abs=0.005)


def test_effective_sample_size_is_computed_per_entity() -> None:
    """One symbol of 2n rows carries less independent information than two of n:
    concurrency stops at an entity boundary."""
    horizon = 10
    one = _panel(n_per_symbol=200, symbols=("A",))
    two = _panel(n_per_symbol=100, symbols=("A", "B"))
    assert effective_sample_size(one, horizon=horizon)[0] == 200
    assert effective_sample_size(two, horizon=horizon)[0] == 200
    assert (
        effective_sample_size(two, horizon=horizon)[1]
        > effective_sample_size(one, horizon=horizon)[1]
    )
