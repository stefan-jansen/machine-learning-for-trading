"""The reported epoch IC is the mean over decision dates, not the mean over folds."""

# torch loads its CUDA extension against whichever runtime is already in the process, so
# it has to come before ml4t pulls in its own.
# isort: off
import torch  # noqa: F401

# isort: on
import numpy as np
import polars as pl
from ml4t.diagnostic.metrics import cross_sectional_ic

from case_studies.utils.latent_factors.cv import _epoch_daily_ic, _select_reporting_epoch

EPOCH = 50


def _fold_predictions(*, fold_id: int, n_dates: int, start_day: int, rng: np.random.Generator):
    """One fold of daily cross-sections, ten symbols wide."""
    symbols = [f"S{i}" for i in range(10)]
    n_symbols = len(symbols)
    days = np.arange(
        np.datetime64("2020-01-01") + np.timedelta64(start_day, "D"),
        np.datetime64("2020-01-01") + np.timedelta64(start_day + n_dates, "D"),
        dtype="datetime64[D]",
    ).astype("datetime64[ms]")
    truth = rng.normal(size=(n_dates, n_symbols))
    # Correlation strength differs by fold so the two averages cannot coincide.
    noise = rng.normal(size=(n_dates, n_symbols)) * (0.5 if fold_id == 0 else 3.0)
    return pl.DataFrame(
        {
            "timestamp": np.repeat(days, n_symbols),
            "symbol": symbols * n_dates,
            "y_true": truth.ravel(),
            "y_score": (truth + noise).ravel(),
            "fold_id": np.full(n_dates * n_symbols, fold_id, dtype=np.int64),
            "epoch": np.full(n_dates * n_symbols, EPOCH, dtype=np.int64),
        }
    )


def _pooled_daily_ic(predictions: pl.DataFrame) -> float:
    return float(
        cross_sectional_ic(
            predictions,
            predictions,
            pred_col="y_score",
            ret_col="y_true",
            date_col="timestamp",
            entity_col="symbol",
            method="spearman",
            min_obs=5,
        )["ic_mean"]
    )


def _fold_metrics(predictions: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for fold_id in predictions["fold_id"].unique().sort():
        fold = predictions.filter(pl.col("fold_id") == fold_id)
        metric = cross_sectional_ic(
            fold,
            fold,
            pred_col="y_score",
            ret_col="y_true",
            date_col="timestamp",
            entity_col="symbol",
            method="spearman",
            min_obs=5,
        )
        rows.append(
            {
                "epoch": EPOCH,
                "fold_id": int(fold_id),
                "ic_mean": float(metric["ic_mean"]),
                "n_scored_dates": int(metric["n_periods"]),
            }
        )
    return pl.DataFrame(rows)


def _unequal_folds() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    short = _fold_predictions(fold_id=0, n_dates=20, start_day=0, rng=rng)
    long = _fold_predictions(fold_id=1, n_dates=200, start_day=40, rng=rng)
    return pl.concat([short, long])


def test_epoch_ic_equals_the_pooled_daily_mean() -> None:
    predictions = _unequal_folds()
    metrics = _fold_metrics(predictions)

    reported = float(_epoch_daily_ic(metrics)["mean_ic"][0])

    assert np.isclose(reported, _pooled_daily_ic(predictions), rtol=0, atol=1e-12)


def test_epoch_ic_is_not_the_mean_of_fold_means() -> None:
    """Folds of unequal length make the two statistics genuinely different numbers."""
    predictions = _unequal_folds()
    metrics = _fold_metrics(predictions)

    reported = float(_epoch_daily_ic(metrics)["mean_ic"][0])
    mean_of_fold_means = float(metrics["ic_mean"].mean())

    assert abs(reported - mean_of_fold_means) > 1.1e-4


def test_reporting_epoch_carries_the_dated_average() -> None:
    metrics = _fold_metrics(_unequal_folds())

    epoch, mean_ic = _select_reporting_epoch(
        metrics,
        checkpoint_selection_policy="fixed",
        reporting_epoch=EPOCH,
    )

    assert epoch == EPOCH
    assert np.isclose(mean_ic, float(_epoch_daily_ic(metrics)["mean_ic"][0]), rtol=0, atol=1e-12)
