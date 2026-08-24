from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from case_studies.utils import causal
from case_studies.utils.causal import (
    _placebo_is_unchanged,
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
    """Treatment moves within an entity's own history and never across entities."""
    dates = np.concatenate([np.arange(1, 13), np.arange(1, 13)])
    units = np.array(["a"] * 12 + ["b"] * 12)
    treatment = np.concatenate([np.arange(11, 23), np.arange(101, 113)])
    order = np.argsort(dates, kind="stable")
    dates, units, treatment = dates[order], units[order], treatment[order]

    permuted = block_permute(
        treatment,
        block_size=3,
        rng=np.random.default_rng(7),
        groups=dates,
        units=units,
    )

    for unit in np.unique(units):
        assert sorted(permuted[units == unit]) == sorted(treatment[units == unit])
    assert not np.array_equal(permuted, treatment)


def test_a_segment_too_short_for_two_blocks_is_left_intact() -> None:
    """It is not shuffled: shuffling destroys the dependence the blocks preserve.

    One entity in a panel can be shorter than two blocks while the others are not,
    so this is per-segment behaviour rather than an error. The case where it happens
    to every segment is caught by `_assert_placebo_permutation_possible`.
    """
    dates = np.concatenate([np.arange(1, 13), np.arange(1, 5)])
    units = np.array(["long"] * 12 + ["short"] * 4)
    treatment = np.concatenate([np.arange(11, 23), np.arange(101, 105)])
    order = np.argsort(dates, kind="stable")
    dates, units, treatment = dates[order], units[order], treatment[order]

    permuted = block_permute(
        treatment,
        block_size=3,
        rng=np.random.default_rng(7),
        groups=dates,
        units=units,
    )

    short = units == "short"
    assert np.array_equal(permuted[short], treatment[short])
    assert not np.array_equal(permuted[~short], treatment[~short])


def test_a_weekend_is_not_a_gap_in_a_daily_series() -> None:
    """The defect this pins: splitting at every diff != cadence cut a daily series
    into five-row weeks, none of which could hold two blocks of a useful size."""
    dates = pd.bdate_range("2024-01-01", periods=30).to_numpy()
    treatment = np.arange(30, dtype=float)

    permuted = block_permute(
        treatment,
        block_size=5,
        rng=np.random.default_rng(3),
        groups=dates,
        expected_step="1D",
    )

    assert sorted(permuted) == sorted(treatment)
    assert not np.array_equal(permuted, treatment)
    # Whole weeks moved: four of every five steps stay contiguous.
    assert np.sum(np.diff(permuted) == 1.0) >= 30 - 30 // 5 - 1


def test_a_real_hole_in_the_series_is_still_a_gap() -> None:
    """Six cadences apart is past the four-cadence tolerance, so the two runs of
    observations are permuted independently and nothing crosses between them."""
    dates = np.array(
        [
            "2024-01-01T00:00:00",
            "2024-01-01T08:00:00",
            "2024-01-01T16:00:00",
            "2024-01-02T00:00:00",
            "2024-01-04T00:00:00",
            "2024-01-04T08:00:00",
            "2024-01-04T16:00:00",
            "2024-01-05T00:00:00",
        ],
        dtype="datetime64[s]",
    )
    treatment = np.arange(8)

    permuted = block_permute(
        treatment,
        block_size=2,
        rng=np.random.default_rng(7),
        groups=dates,
        expected_step="8h",
    )

    assert set(permuted[:4]) == set(treatment[:4])
    assert set(permuted[4:]) == set(treatment[4:])


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


def _two_entity_panel(n_dates: int, seed: int = 19) -> pd.DataFrame:
    """A small balanced daily panel that DML can residualize."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2020-01-01", periods=n_dates, freq="B"), 2)
    confounder = rng.normal(size=len(dates))
    treatment = 0.4 * confounder + rng.normal(size=len(dates))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": np.tile(["A", "B"], n_dates),
            "treatment": treatment,
            "outcome": 0.3 * treatment + 0.2 * confounder + rng.normal(size=len(dates)),
            "confounder": confounder,
        }
    )


def test_a_block_longer_than_every_contiguous_run_fails_the_refutation() -> None:
    """The guard the short-segment path depends on: if no segment can hold two
    blocks, every placebo IS the observed treatment and p = 1 is an artefact."""
    frame = _two_entity_panel(n_dates=100)

    try:
        run_dml_analysis(
            frame,
            treatment_col="treatment",
            outcome_col="outcome",
            confounder_cols=["confounder"],
            n_folds=2,
            embargo=1,
            n_placebo=10,
            block_size=101,
            seed=7,
            horizon=1,
            time_col="timestamp",
            entity_col="symbol",
        )
    except ValueError as error:
        assert "block_size=101" in str(error)
        assert "all 10 placebo draws" in str(error)
    else:
        raise AssertionError("a block longer than every contiguous run must fail")


def test_one_identity_draw_does_not_abort_a_permutable_series() -> None:
    """`rng.permutation` returns the identity by chance - one time in two at two
    blocks. Failing on a single unchanged draw aborts runs that are structurally
    fine, with a message asserting something false about the data. The condition is
    every draw coming back unchanged, not any of them."""
    frame = _two_entity_panel(n_dates=100)
    n_placebo = 30

    unchanged = 0
    real_moved_mask = causal._placebo_moved_mask

    def counting_moved_mask(original, permuted):
        nonlocal unchanged
        moved = real_moved_mask(original, permuted)
        unchanged += not moved.any()
        return moved

    causal._placebo_moved_mask = counting_moved_mask
    try:
        result = run_dml_analysis(
            frame,
            treatment_col="treatment",
            outcome_col="outcome",
            confounder_cols=["confounder"],
            n_folds=2,
            embargo=1,
            n_placebo=n_placebo,
            block_size=50,
            seed=3,
            horizon=1,
            time_col="timestamp",
            entity_col="symbol",
        )
    finally:
        causal._placebo_moved_mask = real_moved_mask

    # Without a draw that came back unchanged the run says nothing about the guard:
    # it would pass under the per-draw abort this replaces.
    assert 0 < unchanged < n_placebo, f"{unchanged} of {n_placebo} draws were the identity"
    assert result["refutation"]["n_successful"] >= 10


def test_the_placebo_guard_sees_through_missing_treatment_values() -> None:
    """`np.array_equal` calls two arrays different wherever either holds a NaN, so
    comparing raw would report every identity draw as a real permutation on any
    frame the resolver's drop_nulls() never touched."""
    treatment = np.array([1.0, np.nan, 3.0, 4.0])

    assert _placebo_is_unchanged(treatment, treatment.copy())
    assert not _placebo_is_unchanged(treatment, np.array([3.0, np.nan, 1.0, 4.0]))
    assert not _placebo_is_unchanged(treatment, np.array([1.0, 2.0, 3.0, 4.0]))


def test_an_explicit_gap_tolerance_moves_the_boundary() -> None:
    """The default clears a weekend at four cadences. A caller that passes its own
    tolerance decides where a series stops being contiguous."""
    dates = pd.bdate_range("2024-01-01", periods=20).to_numpy()
    treatment = np.arange(20, dtype=float)

    tolerant = block_permute(
        treatment,
        block_size=8,
        rng=np.random.default_rng(5),
        groups=dates,
        expected_step="1D",
    )
    strict = block_permute(
        treatment,
        block_size=8,
        rng=np.random.default_rng(5),
        groups=dates,
        expected_step="1D",
        gap_tolerance="1D",
    )

    assert not np.array_equal(tolerant, treatment)
    assert np.array_equal(strict, treatment)


def test_a_partly_frozen_panel_reports_how_much_never_moved() -> None:
    """A unit too short to hold two blocks is returned intact, which is right for that
    unit and invisible in the result unless it is measured. Those rows sit at their
    observed values in every placebo, so the placebo effects cluster on the observed
    effect and the p-value is pulled toward 1 - a "Fails" that looks measured. The
    all-draws-unchanged guard does not fire, because most of the panel does move."""
    rng = np.random.default_rng(11)
    long_dates = pd.date_range("2020-01-01", periods=200, freq="B")
    short_dates = long_dates[:10]
    frames = []
    for symbol, dates in (("LONG", long_dates), ("SHORT", short_dates)):
        confounder = rng.normal(size=len(dates))
        treatment = 0.4 * confounder + rng.normal(size=len(dates))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "treatment": treatment,
                    "outcome": 0.3 * treatment + 0.2 * confounder + rng.normal(size=len(dates)),
                    "confounder": confounder,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])

    with pytest.warns(UserWarning, match="never moved"):
        result = run_dml_analysis(
            frame,
            treatment_col="treatment",
            outcome_col="outcome",
            confounder_cols=["confounder"],
            n_folds=2,
            embargo=1,
            n_placebo=20,
            block_size=20,
            seed=5,
            horizon=1,
            time_col="timestamp",
            entity_col="symbol",
        )

    refutation = result["refutation"]
    # The ten SHORT rows cannot hold two blocks of twenty; the two hundred LONG ones can.
    assert refutation["placebo_frozen_fraction"] == pytest.approx(10 / 210, abs=1e-9)
    assert 0 < refutation["placebo_moved_fraction"] < 1
