"""Tests for `utils.modeling.cross_sectional_ic_mean`.

The one definition of the numpy-array adapter that chapters 11, 12 and 13 use to
score a fold. It replaced 17 copy-pasted definitions, 13 of which took the mean
over a series that could still hold NaN (#493).
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.modeling import cross_sectional_ic_mean


def _panel(n_dates: int = 4, n_entities: int = 12, tied_date: int | None = None):
    """Aligned (y_true, y_pred, dates, entities) arrays, optionally with a tied date."""
    rng = np.random.default_rng(0)
    y_true, y_pred, dates, entities = [], [], [], []
    for d in range(n_dates):
        truth = rng.normal(size=n_entities)
        pred = (
            np.full(n_entities, 0.5)
            if d == tied_date
            else 0.7 * truth + 0.3 * rng.normal(size=n_entities)
        )
        y_true.extend(truth)
        y_pred.extend(pred)
        dates.extend([f"2024-01-{d + 1:02d}"] * n_entities)
        entities.extend(f"S{e:02d}" for e in range(n_entities))
    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(dates),
        np.asarray(entities),
    )


def test_positive_ic_when_predictions_track_returns() -> None:
    ic = cross_sectional_ic_mean(*_panel())
    assert np.isfinite(ic)
    assert ic > 0.5


def test_a_tied_date_is_excluded_not_propagated() -> None:
    """The #493 regression: one undefined date must not make the mean NaN."""
    with_tie = cross_sectional_ic_mean(*_panel(tied_date=1))
    assert np.isfinite(with_tie), "a tied date poisoned the mean"

    # And it is excluded rather than counted as zero: dropping the tied date from
    # the input entirely must give the same answer.
    y_true, y_pred, dates, entities = _panel(tied_date=1)
    keep = dates != "2024-01-02"
    assert with_tie == pytest.approx(
        cross_sectional_ic_mean(y_true[keep], y_pred[keep], dates[keep], entities[keep])
    )


def test_all_dates_tied_returns_nan_not_zero() -> None:
    """No date has a coefficient, so there is no average to report."""
    y_true, y_pred, dates, entities = _panel()
    ic = cross_sectional_ic_mean(y_true, np.full_like(y_pred, 0.5), dates, entities)
    assert np.isnan(ic)


def test_min_obs_excludes_a_thin_cross_section() -> None:
    """12_gradient_boosting/06_optuna_multi_asset lowers the floor to 3."""
    y_true, y_pred, dates, entities = _panel(n_entities=4)

    assert np.isnan(cross_sectional_ic_mean(y_true, y_pred, dates, entities))
    assert np.isfinite(cross_sectional_ic_mean(y_true, y_pred, dates, entities, min_obs=3))


def test_a_nan_from_a_pre_0_1_2_library_is_still_filtered(monkeypatch) -> None:
    """Belt-and-braces: 0.1.1 returned NaN for an undefined date, and `drop_nulls`
    does not remove NaN. Simulate that library so the guard is actually exercised
    rather than shadowed by the 0.1.2 floor pinned in pyproject.toml."""
    import polars as pl
    from ml4t.diagnostic import metrics as diagnostic_metrics

    def old_library_series(*_args, **_kwargs):
        return pl.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "n_obs": [12, 12, 12],
                "ic": [0.2, float("nan"), -0.4],
            }
        )

    monkeypatch.setattr(diagnostic_metrics, "cross_sectional_ic_series", old_library_series)

    # Precondition: the naive reading of that frame really is NaN.
    assert np.isnan(old_library_series().drop_nulls("ic")["ic"].mean())

    assert cross_sectional_ic_mean(*_panel()) == pytest.approx(-0.1)
