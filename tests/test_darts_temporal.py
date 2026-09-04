import datetime as dt

import pandas as pd

from case_studies.utils.darts_forecasting import _overlay_fold_temporal_features


def test_darts_fold_temporal_overlay_uses_requested_fold_values():
    timestamps = pd.to_datetime(["2020-01-01", "2020-01-02"])
    dataset = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["A", "A"],
            "garch_cond_vol": [-1.0, -1.0],
            "other": [1.0, 2.0],
        }
    )
    temporal = pd.DataFrame(
        {
            "timestamp": [*timestamps, *timestamps],
            "symbol": ["A", "A", "A", "A"],
            "fold": [0, 0, 1, 1],
            "garch_cond_vol": [0.1, 0.2, 1.1, 1.2],
        }
    )

    fold = _overlay_fold_temporal_features(
        dataset,
        {
            "fold": 1,
            "train_start": timestamps[0],
            # `train_end` is part of the declared fold geometry (`_TEMPORAL_FOLD_FIELDS`) and
            # `_prepare_fold_series` already requires it. The overlay reads it too now, to bound
            # its uncovered-row trim to the training half. Both rows here are covered, so the
            # trim is a no-op and the values below are what they always were.
            "train_end": timestamps[1],
            "val_end": timestamps[1],
        },
        "timestamp",
        temporal,
        ["timestamp", "symbol"],
        ["garch_cond_vol"],
    )

    assert fold["garch_cond_vol"].to_list() == [1.1, 1.2]
    assert fold["other"].to_list() == [1.0, 2.0]


def _daily_dates(n: int) -> list[dt.date]:
    base = dt.date(2020, 1, 1)
    return [base + dt.timedelta(days=i) for i in range(n)]


def test_the_overlay_scopes_its_contiguity_check_per_symbol():
    """Artifact coverage is ragged across symbols; the check must not read that as a hole.

    etfs' fold 0 has six distinct artifact first-dates across its 100 symbols. Checked globally,
    one symbol's legitimate leading trim falls strictly inside another symbol's date range and
    reads as an interior hole, so a fold that is fine is refused. This is what passing
    `entity_col` through every `_overlay_fold_temporal_features` call site buys.
    """
    dates = pd.to_datetime(_daily_dates(100))
    dataset = pd.DataFrame(
        {
            "timestamp": [*dates, *dates],
            "symbol": ["A"] * len(dates) + ["B"] * len(dates),
            "garch_cond_vol": [-1.0] * (2 * len(dates)),
        }
    )
    # A is covered from index 4, B from index 9 - a trim for each, no hole in either.
    covered = [(s, d) for s, cut in (("A", 4), ("B", 9)) for d in dates[cut:]]
    temporal = pd.DataFrame(
        {
            "timestamp": [d for _, d in covered],
            "symbol": [s for s, _ in covered],
            "fold": [0] * len(covered),
            "garch_cond_vol": [0.5] * len(covered),
        }
    )

    fold = _overlay_fold_temporal_features(
        dataset,
        {
            "fold": 0,
            "train_start": dates[0],
            "train_end": dates[-1],
            "val_end": dates[-1],
        },
        "timestamp",
        temporal,
        ["timestamp", "symbol"],
        ["garch_cond_vol"],
        "symbol",
    )

    assert (fold["symbol"] == "A").sum() == 96
    assert (fold["symbol"] == "B").sum() == 91
