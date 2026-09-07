"""A fold that opens after the account is gone reports the ruin, not zeros.

Second half of ml4t/agent-workspace#920. The engine stops a bankrupt path at
-100% and zeroes every later period, so a fold sliced out of the tail is all
zeros - and a slice of zeros is indistinguishable from a fold in which the
strategy simply did not trade. `compute_portfolio_metrics` on it reported
`ruin=0`, a Sharpe of 0 and no drawdown: a clean row for a period in which the
book did not exist.
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from case_studies.utils.registry.metrics import compute_backtest_fold_metrics

# Ruin lands on 2024-02-10, inside fold 1. Fold 2 opens after it.
FOLDS = [
    {"fold": 1, "val_start": date(2024, 2, 1), "val_end": date(2024, 2, 20)},
    {"fold": 2, "val_start": date(2024, 2, 21), "val_end": date(2024, 3, 11)},
]


@pytest.fixture
def stopped_returns() -> pl.DataFrame:
    from case_studies.utils.backtest_runner import apply_ruin_stop

    raw = [0.01] * 9 + [-1.2] + [0.01] * 30
    stopped, index = apply_ruin_stop(raw)
    assert index == 9
    return pl.DataFrame(
        {
            "timestamp": pl.date_range(
                date(2024, 2, 1), date(2024, 2, 1) + pl.duration(days=39), "1d", eager=True
            ),
            "daily_return": stopped,
        }
    )


def _fold_metrics(monkeypatch, frame: pl.DataFrame) -> dict:
    import case_studies.utils.registry.metrics as metrics_module

    monkeypatch.setattr(metrics_module, "fold_boundaries", lambda *_a, **_k: FOLDS, raising=False)
    import case_studies.utils.cv_window as cv_window

    monkeypatch.setattr(cv_window, "fold_boundaries", lambda *_a, **_k: FOLDS)
    return compute_backtest_fold_metrics(frame, "demo", label="fwd_ret_1d", periods_per_year=252)


def test_the_fold_holding_the_ruin_finds_it_for_itself(monkeypatch, stopped_returns) -> None:
    folds = _fold_metrics(monkeypatch, stopped_returns)
    assert folds[1]["ruin"] == 1.0
    assert folds[1]["max_drawdown"] == -1.0
    assert math.isnan(folds[1]["sharpe"])


def test_a_fold_after_the_ruin_is_told(monkeypatch, stopped_returns) -> None:
    """The defect: all zeros read as a quiet fold with a Sharpe of nothing."""
    folds = _fold_metrics(monkeypatch, stopped_returns)
    assert folds[2]["ruin"] == 1.0
    assert math.isnan(folds[2]["sharpe"])
    # The whole path's index, so the fold rows and the overall row agree on where.
    assert folds[2]["ruin_period"] == 9.0


def test_a_solvent_path_leaves_every_fold_alone(monkeypatch) -> None:
    """The control: no ruin anywhere, so no fold is marked."""
    frame = pl.DataFrame(
        {
            "timestamp": pl.date_range(
                date(2024, 2, 1), date(2024, 2, 1) + pl.duration(days=39), "1d", eager=True
            ),
            "daily_return": [0.01] * 40,
        }
    )
    folds = _fold_metrics(monkeypatch, frame)
    assert folds[1]["ruin"] == 0.0
    assert folds[2]["ruin"] == 0.0
    assert not math.isnan(folds[1]["sharpe"])
