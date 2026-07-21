from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from case_studies.utils.causal import block_permute_panel_time, manual_dml_timeseries


def test_panel_dml_splits_and_embargo_use_complete_dates() -> None:
    dates = np.repeat(pd.date_range("2020-01-01", periods=120, freq="D"), 2)
    x = np.arange(len(dates), dtype=float)
    confounders = np.column_stack((x, x**2))
    treatment = 0.2 * x + np.tile([0.0, 1.0], 120)
    outcome = 0.5 * treatment + 0.1 * x

    result = manual_dml_timeseries(
        outcome,
        treatment,
        confounders,
        n_folds=2,
        embargo=2,
        model_y=LinearRegression(),
        model_t=LinearRegression(),
        return_residuals=True,
        time_values=dates,
    )

    valid = ~np.isnan(result["Y_res"])
    valid_counts = pd.Series(valid).groupby(dates).sum()

    assert valid_counts.loc[pd.Timestamp("2020-02-10")] == 0
    assert valid_counts.loc[pd.Timestamp("2020-02-11")] == 0
    assert valid_counts.loc[pd.Timestamp("2020-02-12")] == 2
    assert set(valid_counts.unique()) <= {0, 2}


def test_panel_block_permutation_keeps_complete_dates_together() -> None:
    dates = np.repeat(pd.date_range("2020-01-01", periods=8, freq="D"), 3)
    values = np.repeat(np.arange(8), 3)

    permuted = block_permute_panel_time(
        values,
        dates,
        block_size=2,
        rng=np.random.default_rng(7),
    )

    assert all(len(set(chunk)) == 1 for chunk in permuted.reshape(-1, 3))
