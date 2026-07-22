from __future__ import annotations

import numpy as np
import pandas as pd

from case_studies.utils.gbm import prepare_gbm_folds


def test_classification_folds_preserve_continuous_evaluation_target() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    dataset = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["A"] * 4,
            "feature": [0.0, 1.0, 2.0, 3.0],
            "fwd_dir_5d": [0, 1, 0, 1],
            "fwd_ret_5d": [-0.03, 0.02, -0.01, 0.04],
        }
    )
    splits = [
        {
            "fold": 0,
            "train_start": dates[0],
            "train_end": dates[1],
            "val_start": dates[2],
            "val_end": dates[3],
        }
    ]

    fold = prepare_gbm_folds(
        dataset,
        splits,
        ["feature"],
        "fwd_dir_5d",
        "timestamp",
        task_type="classification",
        class_values=[0, 1],
        eval_label_col="fwd_ret_5d",
    )[0]

    np.testing.assert_array_equal(fold["y_val"], np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(fold["y_eval"], np.array([-0.01, 0.04], dtype=np.float32))
