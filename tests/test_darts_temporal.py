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
