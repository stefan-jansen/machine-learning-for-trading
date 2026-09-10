from __future__ import annotations

import numpy as np
import polars as pl

from case_studies.utils.folds import clear_memo
from case_studies.utils.gbm import prepare_gbm_folds_from_mds
from utils.modeling import ModelingDataset


def test_classification_folds_preserve_continuous_evaluation_target() -> None:
    """The discrete label is what is fitted; the continuous return is what IC is measured on.

    Both have to survive fold preparation, because losing the continuous column silently turns
    a classification case study's IC into a correlation between a prediction and a class index.
    """
    dates = [
        pl.datetime(2020, 1, 2),
        pl.datetime(2020, 1, 3),
        pl.datetime(2020, 1, 6),
        pl.datetime(2020, 1, 7),
    ]
    frame = pl.select(
        timestamp=pl.concat_list(dates).explode(),
        symbol=pl.lit("A"),
        feature=pl.Series([0.0, 1.0, 2.0, 3.0]),
        fwd_dir_5d=pl.Series([0.0, 1.0, 0.0, 1.0]),
        fwd_ret_5d=pl.Series([-0.03, 0.02, -0.01, 0.04]),
    )
    splits = [
        {
            "fold": 0,
            "train_start": frame["timestamp"][0],
            "train_end": frame["timestamp"][1],
            "val_start": frame["timestamp"][2],
            "val_end": frame["timestamp"][3],
        }
    ]
    mds = ModelingDataset(
        dataset=frame,
        feature_names=["feature"],
        label_col="fwd_dir_5d",
        date_col="timestamp",
        entity_cols=["symbol"],
        join_cols=["symbol", "timestamp"],
        splits=splits,
        label_buffer="5d",
        task_type="classification",
        class_values=[0.0, 1.0],
        eval_label_col="fwd_ret_5d",
    )

    clear_memo()
    [fold] = prepare_gbm_folds_from_mds(mds, splits, use_cache=False)

    np.testing.assert_array_equal(fold["y_val"], np.array([0.0, 1.0]))
    np.testing.assert_array_equal(fold["y_eval"], np.array([-0.01, 0.04]))
    # LightGBM takes contiguous 0-indexed classes; the declared values stay in y_val.
    np.testing.assert_array_equal(fold["y_val_lgb"], np.array([0, 1], dtype=np.int32))
