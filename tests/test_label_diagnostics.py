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


def test_a_one_bar_horizon_halves_the_effective_count() -> None:
    """The library's window is closed at both ends, so a horizon-h label spans h+1
    bars and consecutive labels always share one. At h=1 that is half the window,
    which is why average uniqueness converges to 1/(h+1) rather than to 1/h."""
    frame = _panel(n_per_symbol=50)
    rows, n_eff = effective_sample_size(frame, horizon=1)
    assert rows == 100
    assert n_eff == pytest.approx(rows / 2, rel=0.02)


def test_overlapping_windows_shrink_the_effective_count_toward_n_over_h() -> None:
    horizon = 10
    frame = _panel(n_per_symbol=1000)
    rows, n_eff = effective_sample_size(frame, horizon=horizon)
    assert rows == 2000
    assert n_eff < rows
    # Each row lends most of its window to its neighbours; the limit is N/(h+1).
    assert n_eff / rows == pytest.approx(1 / (horizon + 1), abs=0.005)


@pytest.mark.parametrize(
    ("n", "horizon", "expected"),
    [
        # n=3, h=1. Windows are closed at both ends over bars 0..3, so concurrency is
        # (1, 2, 2, 1) and the uniqueness weights are (0.75, 0.5, 0.75).
        (3, 1, 2.0),
        # n=4, h=2. Concurrency over bars 0..5 is (1, 2, 3, 3, 2, 1); the weights are
        # (11/18, 7/18, 7/18, 11/18).
        (4, 2, 2.0),
    ],
)
def test_effective_sample_size_matches_uniqueness_computed_by_hand(
    n: int, horizon: int, expected: float
) -> None:
    """Pins the boundary convention, which is the whole difficulty here.

    Every row of the frame carries a complete forward window, but the bars that
    close the last `horizon` windows are not rows of the frame. Capping the
    endpoints at `n` leaves concurrency on the retained bars untouched and discards
    those closing bars, which are the least concurrent and so the most unique part
    of each boundary window - so for these cases, where the horizon is well under
    the group size, it comes out below these values.
    """
    frame = _panel(n_per_symbol=n, symbols=("A",))
    rows, n_eff = effective_sample_size(frame, horizon=horizon)
    assert rows == n
    assert n_eff == pytest.approx(expected)


def test_a_horizon_longer_than_the_group_is_not_the_ordinary_regime() -> None:
    """Guards the bound on the docstring's directional claim.

    Capping the endpoints understates N_eff while the horizon fits inside the
    group, because it discards the low-concurrency tail of each window. Once the
    horizon is long relative to the group it removes most of every window instead
    and the comparison reverses. A group is one entity's *labelled* rows, so a
    symbol with barely more bars than the horizon lands here.
    """
    ev = np.arange(2)
    uncapped = calculate_label_uniqueness(ev, ev + 4, n_bars=2 + 4).sum()
    capped = calculate_label_uniqueness(ev, np.minimum(ev + 4, 2), n_bars=2).sum()
    assert capped > uncapped

    ev = np.arange(50)
    uncapped = calculate_label_uniqueness(ev, ev + 10, n_bars=60).sum()
    capped = calculate_label_uniqueness(ev, np.minimum(ev + 10, 50), n_bars=50).sum()
    assert capped < uncapped


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
