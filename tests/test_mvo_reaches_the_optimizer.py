"""`compute_mvo_weights` must not silently become equal weight.

Two things decide whether an `mvo_ledoit_wolf` row is an optimization or its own
fallback, and until 2026-09-24 neither was visible anywhere. The observation floor
was `max(top_k, lookback // 2)`, and `top_k` is a cross-section width that
`_apply_allocation` defaults to the size of the whole traded universe when the
allocation spec omits it. On `sp500_options` that put the floor at 268 observations
against a 63-day lookback and on `us_equities_panel` at 3,113 against 126: floors no
window can reach, so every rebalance returned equal weight under an MVO label, with
nothing in the registry row saying so.
"""

import logging

import numpy as np
import polars as pl

from case_studies.utils.allocation import compute_mvo_weights

DAYS = 200
SYMBOLS = [f"S{i:02d}" for i in range(6)]


def _panel(n_days: int = DAYS, symbols: list[str] | None = None) -> pl.DataFrame:
    symbols = symbols or SYMBOLS
    rng = np.random.default_rng(7)
    dates = pl.date_range(
        pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=n_days - 1), eager=True
    )
    rows = []
    for j, sym in enumerate(symbols):
        price = 100.0 * np.cumprod(1.0 + rng.normal(0.0004 * (j + 1), 0.01, n_days))
        rows.append(pl.DataFrame({"timestamp": dates, "symbol": sym, "close": price}))
    return pl.concat(rows).cast({"timestamp": pl.Datetime("us")})


def _predictions(prices: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(11)
    return prices.select("timestamp", "symbol").with_columns(
        y_score=pl.Series(rng.normal(size=prices.height))
    )


def _is_equal_weight(weights: pl.DataFrame) -> pl.DataFrame:
    return weights.group_by("timestamp").agg(
        flat=(pl.col("weight").max() - pl.col("weight").min()).abs() < 1e-12
    )


def test_a_universe_wider_than_the_lookback_still_optimizes():
    """The shape that made MVO equal weight on every sp500_options rebalance.

    `top_k` arrives as the universe size, far larger than any window the lookback can
    supply. The floor is a count of observations, so it must not scale with the
    cross-section the caller declared.
    """
    prices = _panel()
    weights = compute_mvo_weights(
        _predictions(prices), prices, top_k=268, lookback=63, max_weight=1.0
    )
    assert weights.height > 0
    flat = _is_equal_weight(weights)
    assert not flat["flat"].all(), "every rebalance fell back to equal weight"


def test_the_floor_still_refuses_fewer_observations_than_the_cross_section():
    """The negative control, and it has to reach the floor rather than the guard above it.

    A window long enough to clear `height < lookback // 2` but holding fewer complete
    rows than the cross-section is wide: 40 names against a 63-day lookback whose rows
    are punched through with gaps. The estimate spans more assets than it has
    observations, which is the condition the floor exists for.
    """
    symbols = [f"W{i:02d}" for i in range(40)]
    prices = _panel(n_days=120, symbols=symbols)
    # Punch a gap into one name on most dates, so the row-wise NaN drop leaves fewer
    # complete observations than there are columns.
    gapped = prices.with_columns(
        close=pl.when((pl.col("symbol") == symbols[0]) & (pl.col("timestamp").dt.day() % 3 != 0))
        .then(None)
        .otherwise(pl.col("close"))
    )
    weights = compute_mvo_weights(
        _predictions(gapped), gapped, top_k=40, lookback=63, max_weight=1.0
    )
    flat = _is_equal_weight(weights)
    assert flat["flat"].all(), (
        "40 columns with fewer than 40 complete rows is not an estimable covariance"
    )


def test_a_mostly_degenerate_run_says_so(caplog):
    """An allocator that became its own fallback reads exactly like one that ran."""
    prices = _panel(n_days=20)
    with caplog.at_level(logging.WARNING, logger="case_studies.utils.allocation"):
        compute_mvo_weights(_predictions(prices), prices, top_k=6, lookback=126, max_weight=1.0)
    assert any("returned equal weight on" in r.message for r in caplog.records)


def test_a_healthy_run_is_quiet(caplog):
    """The other side of the same control: no warning when the optimizer does the work."""
    prices = _panel()
    with caplog.at_level(logging.WARNING, logger="case_studies.utils.allocation"):
        compute_mvo_weights(_predictions(prices), prices, top_k=6, lookback=63, max_weight=1.0)
    assert not [r for r in caplog.records if "returned equal weight on" in r.message]
