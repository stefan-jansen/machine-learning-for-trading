"""Behavior of the shared cross-sectional IC adapter in utils.modeling.

The defect these pin (#493): a date whose predictions all tie has an undefined
Spearman coefficient. ml4t-diagnostic 0.1.1 reports it as NaN and 0.1.2 as null,
and in polars `drop_nulls` removes only the second. One survivor turns the mean
of the entire series into NaN, so a notebook reports `nan` for its headline
metric, or selects an arbitrary model, with no error anywhere.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from utils.modeling import cross_sectional_ic_mean

N_ENTITIES = 12


def _date_block(day: str, predictions: np.ndarray, returns: np.ndarray):
    entities = np.array([f"S{i}" for i in range(len(predictions))])
    dates = np.array([day] * len(predictions))
    return dates, entities, predictions, returns


def _concat(*blocks):
    return tuple(np.concatenate(parts) for parts in zip(*blocks, strict=True))


def _ranked_block(day: str, *, tied: bool):
    ordered = np.arange(N_ENTITIES, dtype=float)
    predictions = np.full(N_ENTITIES, 0.5) if tied else ordered / 10.0
    returns = (ordered % 5) / 100.0
    return _date_block(day, predictions, returns)


def test_the_tied_date_really_is_undefined() -> None:
    """The case the next test relies on: the library cannot score a tied date.

    Without this, a library version that started scoring tied dates would make
    the poisoning test pass for the wrong reason and stop measuring anything.
    """
    import polars as pl
    from ml4t.diagnostic.metrics import cross_sectional_ic_series

    dates, entities, predictions, returns = _ranked_block("2020-01-02", tied=True)
    ic = cross_sectional_ic_series(
        pl.DataFrame({"timestamp": dates, "symbol": entities, "prediction": predictions}),
        pl.DataFrame({"timestamp": dates, "symbol": entities, "forward_return": returns}),
        pred_col="prediction",
        ret_col="forward_return",
        date_col="timestamp",
        entity_col="symbol",
        min_obs=10,
    )
    undefined = ic["ic"].null_count() + int(ic["ic"].is_nan().fill_null(False).sum())
    assert undefined == ic.height


def test_a_tied_date_does_not_poison_the_mean() -> None:
    """One undefined date must be dropped, not propagated into the average."""
    defined_only = _ranked_block("2020-01-01", tied=False)
    with_a_tied_date = _concat(defined_only, _ranked_block("2020-01-02", tied=True))

    dates, entities, predictions, returns = defined_only
    one_date = cross_sectional_ic_mean(returns, predictions, dates, entities)

    dates, entities, predictions, returns = with_a_tied_date
    two_dates = cross_sectional_ic_mean(returns, predictions, dates, entities)

    assert math.isfinite(two_dates)
    assert two_dates == pytest.approx(one_date)


def test_every_date_undefined_returns_nan_not_zero() -> None:
    """A model that predicts one constant has no IC, and NaN is what says so."""
    dates, entities, predictions, returns = _concat(
        _ranked_block("2020-01-01", tied=True),
        _ranked_block("2020-01-02", tied=True),
    )

    assert math.isnan(cross_sectional_ic_mean(returns, predictions, dates, entities))


def test_min_obs_excludes_a_thin_cross_section() -> None:
    """Below min_obs a date is not scored, so a thin panel scores no dates."""
    thin = 4
    ordered = np.arange(thin, dtype=float)
    dates, entities, predictions, returns = _date_block(
        "2020-01-01", ordered / 10.0, (ordered % 3) / 100.0
    )

    assert math.isnan(cross_sectional_ic_mean(returns, predictions, dates, entities, min_obs=10))
    assert math.isfinite(
        cross_sectional_ic_mean(returns, predictions, dates, entities, min_obs=thin)
    )
