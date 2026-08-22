from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from case_studies.utils.causal import (
    _resolve_panel_columns,
    _walk_forward_indices,
    block_permute,
    embargo_from_buffer,
    manual_dml_timeseries,
    observation_step,
    run_dml_analysis,
)


def test_resolve_panel_columns_uses_canonical_schema() -> None:
    frame = pd.DataFrame({"timestamp": [1, 1], "symbol": ["a", "b"]})

    assert _resolve_panel_columns(frame, None, None) == ("timestamp", "symbol")


def test_resolve_panel_columns_rejects_unidentified_panel() -> None:
    frame = pd.DataFrame({"when": [1, 1], "value": [2.0, 3.0]})

    try:
        _resolve_panel_columns(frame, "when", None)
    except ValueError as error:
        assert "entity_col" in str(error)
    else:
        raise AssertionError("duplicated decision times need an entity column")


def test_panel_walk_forward_keeps_dates_whole_and_embargoes_dates() -> None:
    dates = np.repeat(np.arange(24), 4)

    folds = _walk_forward_indices(
        n_rows=len(dates),
        n_folds=3,
        embargo=2,
        groups=dates,
    )

    assert len(folds) == 3
    for train_idx, test_idx in folds:
        train_dates = dates[train_idx]
        test_dates = dates[test_idx]
        assert set(train_dates).isdisjoint(test_dates)
        assert int(test_dates.min() - train_dates.max()) == 3
        assert all(np.count_nonzero(test_dates == date) == 4 for date in np.unique(test_dates))


def test_monthly_label_buffer_is_one_monthly_decision_group() -> None:
    assert embargo_from_buffer("1M", periods_per_year=12) == 1


def test_observation_step_measures_the_grid_and_ignores_session_gaps() -> None:
    """The mode is the bar size; overnight and weekend gaps are outliers."""
    session_one = pd.date_range("2021-01-04 09:30", periods=390, freq="1min")
    session_two = pd.date_range("2021-01-05 09:30", periods=390, freq="1min")
    frame = pd.DataFrame({"timestamp": session_one.append(session_two)})

    assert observation_step(frame) == pd.Timedelta("1min")


def test_observation_step_reads_a_polars_frame() -> None:
    import polars as pl

    frame = pl.DataFrame({"timestamp": pd.date_range("2021-01-04", periods=10, freq="1D")})

    assert observation_step(frame) == pd.Timedelta("1D")


def test_embargo_counts_label_horizons_on_the_observed_grid() -> None:
    """A 15-minute label on a one-minute panel embargoes 15 periods, not one."""
    assert embargo_from_buffer("15min", observed_step=pd.Timedelta("1min")) == 15
    assert embargo_from_buffer("15min", observed_step="15min") == 1
    assert embargo_from_buffer("8H", observed_step=pd.Timedelta("8h")) == 1
    assert embargo_from_buffer("21D", observed_step=pd.Timedelta("1D")) == 21


def test_embargo_without_an_observed_step_assumes_a_bar_size() -> None:
    """The fallback is wrong by the ratio between the assumed bar and the real one.

    It pins the assumption rather than blessing the answer: "15min" resolves to
    one period whatever the panel is recorded at, which is right on a 15-minute
    grid and wrong by fifteen on a one-minute grid.
    """
    assert embargo_from_buffer("15min") == 1
    assert embargo_from_buffer("15min", observed_step=pd.Timedelta("15min")) == 1
    assert embargo_from_buffer("15min", observed_step=pd.Timedelta("1min")) == 15


def test_embargo_rejects_a_month_buffer_against_an_observation_step() -> None:
    """A month has no fixed length, so it cannot be divided by a bar size."""
    assert embargo_from_buffer("1M", periods_per_year=12) == 1
    try:
        embargo_from_buffer("1M", observed_step=pd.Timedelta("1D"))
    except ValueError as error:
        assert "month" in str(error).lower()
    else:
        raise AssertionError("a month buffer must not resolve against an observation step")


def test_embargo_rounds_a_partial_period_up() -> None:
    """Half a period of gap is not a gap; the label still resolves inside it."""
    assert embargo_from_buffer("10min", observed_step=pd.Timedelta("4min")) == 3


def test_embargo_rejects_a_nonpositive_observed_step() -> None:
    try:
        embargo_from_buffer("15min", observed_step=pd.Timedelta(0))
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("a zero-length grid step must not resolve to an embargo")


def test_panel_walk_forward_rejects_unsorted_groups() -> None:
    dates = np.array([0, 0, 2, 2, 1, 1])

    try:
        _walk_forward_indices(n_rows=len(dates), n_folds=1, embargo=0, groups=dates)
    except ValueError as error:
        assert "sorted" in str(error)
    else:
        raise AssertionError("unsorted panel groups must be rejected")


def test_panel_cross_fitting_residualizes_complete_dates() -> None:
    counts = np.resize(np.array([3, 5, 4]), 80)
    dates = np.repeat(np.arange(len(counts)), counts)
    rng = np.random.default_rng(42)
    treatment = rng.normal(size=len(dates))
    confounders = rng.normal(size=(len(dates), 2))
    outcome = treatment + rng.normal(size=len(dates))

    result = manual_dml_timeseries(
        outcome,
        treatment,
        confounders,
        n_folds=3,
        embargo=2,
        model_y=DummyRegressor(),
        model_t=DummyRegressor(),
        return_residuals=True,
        groups=dates,
    )

    valid = ~np.isnan(result["T_res"])
    for date in np.unique(dates):
        date_valid = valid[dates == date]
        assert date_valid.all() or not date_valid.any()
    assert result["n_periods"] == len(np.unique(dates[valid]))
    assert result["covariance_type"] == "driscoll_kraay"
    assert result["hac_maxlags"] < int(valid.sum() ** (1 / 3))


def test_panel_block_permutation_preserves_each_entity_history() -> None:
    dates = np.array([1, 1, 2, 2, 3, 3, 4, 5])
    units = np.array(["a", "b", "a", "b", "a", "b", "a", "a"])
    treatment = np.array([11, 21, 12, 22, 13, 23, 14, 15])

    permuted = block_permute(
        treatment,
        block_size=2,
        rng=np.random.default_rng(7),
        groups=dates,
        units=units,
    )

    for unit in np.unique(units):
        assert sorted(permuted[units == unit]) == sorted(treatment[units == unit])
    assert not np.array_equal(permuted, treatment)


def test_panel_block_permutation_requires_entity_column() -> None:
    dates = np.array([1, 1, 2, 2])

    try:
        block_permute(np.arange(4), block_size=1, groups=dates)
    except ValueError as error:
        assert "units are required" in str(error)
    else:
        raise AssertionError("panel permutation without units must be rejected")


def test_dml_comparisons_use_the_observed_second_stage_sample_and_folds() -> None:
    rng = np.random.default_rng(19)
    n_dates = 100
    n_entities = 2
    dates = np.repeat(pd.date_range("2020-01-01", periods=n_dates, freq="B"), n_entities)
    symbols = np.tile(["A", "B"], n_dates)
    confounder = rng.normal(size=len(dates))
    treatment = 0.4 * confounder + rng.normal(size=len(dates))
    outcome = 0.3 * treatment + 0.2 * confounder + rng.normal(size=len(dates))
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": symbols,
            "treatment": treatment,
            "outcome": outcome,
            "confounder": confounder,
        }
    )

    result = run_dml_analysis(
        frame,
        treatment_col="treatment",
        outcome_col="outcome",
        confounder_cols=["confounder"],
        n_folds=2,
        embargo=1,
        n_placebo=10,
        block_size=3,
        seed=7,
        horizon=1,
        time_col="timestamp",
        entity_col="symbol",
    )

    residuals = result["dml_result"]
    valid = np.isfinite(residuals["T_res"]) & np.isfinite(residuals["Y_res"])
    same_sample = frame.loc[valid]
    expected_naive = np.linalg.lstsq(
        np.column_stack([np.ones(valid.sum()), same_sample["treatment"]]),
        same_sample["outcome"],
        rcond=None,
    )[0][1]

    assert result["naive_effect"] == expected_naive
    assert result["naive_n_obs"] == residuals["n_obs"]
    assert result["refutation"]["n_folds"] == 2
    assert set(result["refutation"]["placebo_n_obs"]) == {residuals["n_obs"]}
