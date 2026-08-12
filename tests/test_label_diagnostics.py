"""Panel autocorrelation and effective sample size, against known answers.

Both statistics are cheap to compute wrongly in a way that looks plausible: an
autocorrelation that silently pairs one entity's last row with the next entity's
first, an effective sample size that returns the row count when the windows do not
in fact overlap, and either one closing over a hole in the grid so that windows
sharing nothing are counted as adjacent.
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
            "bar": list(range(n_per_symbol)) * len(symbols),
            "x": rng.normal(size=n_per_symbol * len(symbols)),
        }
    )


def test_autocorrelation_of_a_repeating_series_is_one_at_the_period() -> None:
    period = 4
    frame = pl.DataFrame(
        {
            "symbol": ["A"] * 40,
            "timestamp": list(range(40)),
            "bar": list(range(40)),
            "x": [float(i % period) for i in range(40)],
        }
    )
    acf = panel_autocorrelation(frame, "x", max_lag=period, bar_col="bar")
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
            "bar": list(range(10)) * 2,
            "x": [1.0] * 10 + [5.0] * 10,
        }
    )
    acf = panel_autocorrelation(frame, "x", max_lag=3, bar_col="bar")
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
    rows, n_eff = effective_sample_size(frame, horizon=1, bar_col="bar")
    assert rows == 100
    assert n_eff == pytest.approx(rows)


def test_overlapping_windows_shrink_the_effective_count_toward_n_over_h() -> None:
    horizon = 10
    frame = _panel(n_per_symbol=1000)
    rows, n_eff = effective_sample_size(frame, horizon=horizon, bar_col="bar")
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
    rows, n_eff = effective_sample_size(frame, horizon=horizon, bar_col="bar")
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


def _gapped_alternating(blocks: tuple[tuple[int, int], ...]) -> pl.DataFrame:
    """One entity sampled on a grid with holes, valued +1/-1 by grid parity.

    *blocks* is a sequence of (first bar, length) runs, so the bars between two
    runs are absent from the grid rather than absent from the data.
    """
    bars = [b for start, length in blocks for b in range(start, start + length)]
    return pl.DataFrame(
        {
            "symbol": ["A"] * len(bars),
            "bar": bars,
            "x": [1.0 if b % 2 == 0 else -1.0 for b in bars],
        }
    )


def test_a_hole_in_the_grid_is_not_read_as_adjacency() -> None:
    """The defect this argument exists to stop.

    The series alternates with grid parity, so its true lag-1 autocorrelation is
    exactly -1. Every run here starts on an even bar, so pairing rows by their
    position among the surviving rows joins the last row of one run to the first of
    the next - two rows of the same parity - and reports persistence where the grid
    says there is none.
    """
    frame = _gapped_alternating(((0, 3), (4, 3), (8, 3), (12, 3)))
    acf = panel_autocorrelation(frame, "x", max_lag=1, bar_col="bar")
    assert acf[0] == pytest.approx(-1.0)

    positional = frame.with_columns(pl.int_range(pl.len()).over("symbol").alias("row"))
    spurious = panel_autocorrelation(positional, "x", max_lag=1, bar_col="row")[0]
    assert spurious == pytest.approx(-4 / 7)  # three of the eleven pairs span a hole


def test_labels_either_side_of_a_hole_do_not_overlap() -> None:
    """Two runs far enough apart share no forward window, so the effective count is
    twice one run's - not the smaller number that treating the six rows as six
    consecutive bars returns."""
    horizon = 3
    two_runs = _gapped_alternating(((0, 3), (10, 3)))
    one_run = _gapped_alternating(((0, 3),))

    rows, n_eff = effective_sample_size(two_runs, horizon=horizon, bar_col="bar")
    assert rows == 6
    assert n_eff == pytest.approx(
        2 * effective_sample_size(one_run, horizon=horizon, bar_col="bar")[1]
    )
    assert n_eff == pytest.approx(10 / 3)  # 5/3 per run, computed by hand as above

    positional = two_runs.with_columns(pl.int_range(pl.len()).over("symbol").alias("row"))
    contiguous = effective_sample_size(positional, horizon=horizon, bar_col="row")[1]
    assert contiguous == pytest.approx(8 / 3)  # six consecutive bars, overlap overstated


def test_bars_are_taken_from_the_grid_regardless_of_row_order() -> None:
    """The grid column, not the frame's order, decides what is adjacent."""
    frame = _gapped_alternating(((0, 8),))
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=0)
    assert panel_autocorrelation(shuffled, "x", max_lag=2, bar_col="bar") == pytest.approx(
        panel_autocorrelation(frame, "x", max_lag=2, bar_col="bar")
    )
    assert effective_sample_size(shuffled, horizon=3, bar_col="bar") == pytest.approx(
        effective_sample_size(frame, horizon=3, bar_col="bar")
    )


def test_effective_sample_size_is_computed_per_entity() -> None:
    """One symbol of 2n rows carries less independent information than two of n:
    concurrency stops at an entity boundary."""
    horizon = 10
    one = _panel(n_per_symbol=200, symbols=("A",))
    two = _panel(n_per_symbol=100, symbols=("A", "B"))
    assert effective_sample_size(one, horizon=horizon, bar_col="bar")[0] == 200
    assert effective_sample_size(two, horizon=horizon, bar_col="bar")[0] == 200
    assert (
        effective_sample_size(two, horizon=horizon, bar_col="bar")[1]
        > effective_sample_size(one, horizon=horizon, bar_col="bar")[1]
    )


def test_effective_sample_size_sums_entities_in_a_fixed_order() -> None:
    """The same frame returns the same float every call, to the last bit.

    `effective_sample_size` accumulates one float per entity, and adding floats is not
    associative, so a total near a rounding boundary lands on either side of it depending
    on the order the groups came back in. `sp500_options/02_labels` reported N_eff 39,746
    on one production run and 39,747 on the next, from identical inputs and unchanged
    label digests. Entities of deliberately unequal size make the summation order visible:
    the exact total differs between permutations, so a run that fixes the order and a run
    that does not cannot both satisfy the assertion below.
    """
    sizes = [4000] + [7] * 120
    symbols = [f"S{i:03d}" for i in range(len(sizes))]
    frame = pl.DataFrame(
        {
            "symbol": [s for s, n in zip(symbols, sizes, strict=True) for _ in range(n)],
            "bar": [b for n in sizes for b in range(n)],
        }
    )
    first = effective_sample_size(frame, horizon=10, bar_col="bar")
    for _ in range(4):
        assert effective_sample_size(frame, horizon=10, bar_col="bar") == first

    # The fixed order is the order the entities appear in, so summing the per-entity
    # results left to right in that order reproduces the total exactly.
    by_hand = 0.0
    for symbol in symbols:
        by_hand += effective_sample_size(
            frame.filter(pl.col("symbol") == symbol), horizon=10, bar_col="bar"
        )[1]
    assert first[1] == by_hand
