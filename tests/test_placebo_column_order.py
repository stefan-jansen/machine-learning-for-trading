"""The placebo universe must not depend on the column order of the frame it was handed.

`run_placebo_benchmark` draws COLUMN POSITIONS with `rng.choice(n_symbols, ...)`, so which
symbols a random book holds was decided by the order of `daily_returns_wide`. A caller
building that frame with `pivot` gets first-appearance order, which is a property of the
parquet it read rather than of any code - so regenerating an upstream artifact in a
different row order silently changed the distribution the strategy is compared against,
at the same seed, with the seed making it look pinned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from case_studies.utils.factor_attribution import run_placebo_benchmark


def _universe(seed: int = 0, n_symbols: int = 12, n_days: int = 300):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    # Distinct per-symbol means, so a book of a different set of symbols has a different
    # return: with identical columns the test could not tell the two orders apart.
    rets = pd.DataFrame(
        rng.standard_normal((n_days, n_symbols)) * 0.01 + np.linspace(-0.002, 0.002, n_symbols),
        index=dates,
        columns=symbols,
    )
    factors = pd.DataFrame(
        rng.standard_normal((n_days, 6)) * 0.005,
        index=dates,
        columns=["mkt_rf", "smb", "hml", "rmw", "cma", "mom"],
    )
    factors["rf"] = 0.0
    return rets, factors


def test_a_shuffled_column_order_gives_the_same_placebo_distribution() -> None:
    rets, factors = _universe()
    shuffled = rets[list(reversed(rets.columns))]

    a = run_placebo_benchmark(rets, factors, n_sims=40, top_k=3, seed=42)
    b = run_placebo_benchmark(shuffled, factors, n_sims=40, top_k=3, seed=42)

    assert a["alpha_ann_mean"] == pytest.approx(b["alpha_ann_mean"])
    assert a["alpha_ann_std"] == pytest.approx(b["alpha_ann_std"])
    assert a["r_squared_mean"] == pytest.approx(b["r_squared_mean"])


def test_the_sorted_order_is_the_one_that_is_kept() -> None:
    """Both current callers pass sorted columns, so the fix must not move their numbers.

    Reordering has to converge on the order those callers already had, not on some third
    canonical form - otherwise every placebo figure in the corpus moves for a defect that
    was not reaching them.
    """
    rets, factors = _universe()
    assert list(rets.columns) == sorted(rets.columns)

    a = run_placebo_benchmark(rets, factors, n_sims=40, top_k=3, seed=42)
    b = run_placebo_benchmark(
        rets[list(reversed(rets.columns))], factors, n_sims=40, top_k=3, seed=42
    )

    assert a["alpha_ann_mean"] == pytest.approx(b["alpha_ann_mean"])


def test_a_dropped_all_nan_column_does_not_shift_the_rest() -> None:
    """`dropna(axis=1, how='all')` runs before the sort, so an empty symbol must not
    displace the others - it is removed from the universe, not renamed into a neighbour."""
    rets, factors = _universe()
    with_hole = rets.copy()
    with_hole["S05"] = np.nan

    a = run_placebo_benchmark(with_hole, factors, n_sims=40, top_k=3, seed=42)
    b = run_placebo_benchmark(
        with_hole[list(reversed(with_hole.columns))], factors, n_sims=40, top_k=3, seed=42
    )

    assert a["alpha_ann_mean"] == pytest.approx(b["alpha_ann_mean"])
