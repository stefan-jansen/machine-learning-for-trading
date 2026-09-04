"""What the coverage figure a notebook prints is measured on.

The figure describes the estimator `conformal_weighted` sizes positions with -
`walk_forward_widths`, per symbol with a pooled fallback, calibrated on everything known at
`t - h`, quantile at `interpolation="higher"`. It used to describe a second estimator built to
different rules in all four respects: one pooled quantile over every symbol, fixed on the
earliest fold, no embargo, exact order statistic. Each test below separates the two.
"""

from __future__ import annotations

import polars as pl
import pytest

from case_studies.utils.conformal import (
    walk_forward_conformal_coverage,
    walk_forward_widths,
)


def _panel(residuals: dict[str, list[float]]) -> pl.DataFrame:
    """One row per (day, symbol), with `y_true - y_score` exactly as `residuals` says.

    Every symbol supplies the same number of steps, so the day grid is the row index and an
    embargo of h steps is h days back. `y_score` ramps so the outcomes have a spread for the
    width to be reported against, and `fold_id` labels the later half 0: a reader that takes
    fold ids for chronological order gets this panel backwards.
    """
    lengths = {len(values) for values in residuals.values()}
    assert len(lengths) == 1, "every symbol supplies the same number of steps"
    steps = lengths.pop()
    days = [f"2020-{1 + step // 28:02d}-{1 + step % 28:02d}" for step in range(steps)]
    scores = [float(step) / steps for step in range(steps)]
    return pl.DataFrame(
        {
            "timestamp": [day for day in days for _ in residuals],
            "symbol": [symbol for _ in days for symbol in residuals],
            "y_true": [
                scores[step] + residuals[symbol][step]
                for step in range(steps)
                for symbol in residuals
            ],
            "y_score": [scores[step] for step in range(steps) for _ in residuals],
            "fold_id": [1 if step < steps // 2 else 0 for step in range(steps) for _ in residuals],
        }
    )


def _widths(panel: pl.DataFrame, **kwargs) -> pl.DataFrame:
    prepared = panel.with_columns(abs_resid=(pl.col("y_true") - pl.col("y_score")).abs())
    return walk_forward_widths(prepared, id_col="symbol", context="test", **kwargs)


# One quiet symbol and one an order of magnitude wider. Pooling is the difference that reaches
# the portfolio: `compute_conformal_weights` normalizes 1/width within each side at each
# timestamp, so only the dispersion *across* symbols survives - the axis a pooled quantile
# removes.
TWO_SCALES = {"CALM": [0.1] * 80, "WILD": [10.0] * 80}


def test_a_symbol_is_sized_on_its_own_residuals_once_it_has_enough_of_them() -> None:
    widths = _widths(_panel(TWO_SCALES), alpha=0.2, min_calibration_n=30, embargo_steps=1)
    own = widths.filter(pl.col("calibration_scope") == "symbol")

    by_symbol = {
        symbol: values for symbol, values in own.group_by("symbol").agg(pl.col("width")).iter_rows()
    }
    assert by_symbol["CALM"] == pytest.approx([0.2] * len(by_symbol["CALM"]))
    assert by_symbol["WILD"] == pytest.approx([20.0] * len(by_symbol["WILD"]))


def test_the_pooled_quantile_is_the_fallback_and_not_the_rule() -> None:
    """A pooled width over both symbols is 20.0 everywhere, which is CALM's answer only until
    CALM has 30 residuals of its own. Reporting 20.0 for CALM throughout is the reading this
    replaced, and it is the one number a per-symbol estimator never produces for CALM.
    """
    widths = _widths(_panel(TWO_SCALES), alpha=0.2, min_calibration_n=30, embargo_steps=1)
    calm = widths.filter(pl.col("symbol") == "CALM")

    pooled_widths = calm.filter(pl.col("calibration_scope") == "pooled")["width"].to_list()
    assert pooled_widths == pytest.approx([20.0] * len(pooled_widths))
    assert calm.filter(pl.col("calibration_scope") == "symbol").height > 0

    # Raise the warm-up above what any symbol can reach on its own and every decision falls
    # back to the pool, which is what the fallback is for: allocation never drops a name.
    pooled_only = _widths(_panel(TWO_SCALES), alpha=0.2, min_calibration_n=100, embargo_steps=1)
    assert pooled_only["calibration_scope"].unique().to_list() == ["pooled"]
    pooled = pooled_only["width"].to_list()
    assert pooled == pytest.approx([20.0] * len(pooled))


def test_a_residual_the_decision_could_not_have_known_does_not_calibrate_it() -> None:
    """The embargo is the label horizon, and the estimator this replaced had none.

    Every large residual sits in the last four steps of the panel. At an embargo of one step
    they reach the final decisions; at twenty they reach none, and the widths must differ.
    """
    residuals = [0.1] * 76 + [9.0] * 4
    panel = _panel({"AAA": list(residuals), "BBB": list(residuals)})

    near = walk_forward_conformal_coverage(panel, embargo_steps=1, levels=(0.90,))[0]
    far = walk_forward_conformal_coverage(panel, embargo_steps=20, levels=(0.90,))[0]

    assert near["mean_interval_width_frac_std"] > far["mean_interval_width_frac_std"]
    # A wider embargo also withholds more decisions from calibration entirely.
    assert far["n_uncalibrated"] > near["n_uncalibrated"]
    assert far["n_test"] < near["n_test"]


def test_the_width_is_twice_a_residual_the_calibration_set_attains() -> None:
    """`interpolation="higher"` is the sizing rule, so no interpolated quantile appears.

    Half the residuals are 0.1 and half are 5.0, so every width is 0.2 or 10.0 - never
    something between them, which a linear-interpolation quantile would produce.
    """
    residuals = [0.1] * 40 + [5.0] * 40
    widths = _widths(_panel({"AAA": residuals}), alpha=0.2, min_calibration_n=30, embargo_steps=1)

    distinct = sorted({round(width, 9) for width in widths["width"].to_list()})
    assert distinct == pytest.approx([0.2, 10.0])


def test_every_decision_is_either_measured_or_reported_as_uncalibrated() -> None:
    """A coverage figure describes what it covers and says how much it does not.

    The warm-up leaves the earliest decisions with no eligible residual. Counting them as
    missed would understate coverage and counting them as covered would overstate it, so they
    are neither: they are excluded from `n_test` and reported separately.
    """
    panel = _panel(TWO_SCALES)
    row = walk_forward_conformal_coverage(panel, embargo_steps=1, levels=(0.80,))[0]

    assert row["n_test"] + row["n_uncalibrated"] == panel.height
    assert row["n_uncalibrated"] > 0


def test_a_panel_with_no_entity_column_is_refused() -> None:
    """Per-symbol calibration needs to know the symbol, and pooling silently instead is the
    defect this replaced. An artifact without one is refused rather than measured pooled.
    """
    panel = _panel({"AAA": [0.1] * 80}).drop("symbol")
    with pytest.raises(ValueError, match="no canonical entity column"):
        walk_forward_conformal_coverage(panel, embargo_steps=1, levels=(0.80,))


def test_a_negative_embargo_is_refused() -> None:
    with pytest.raises(ValueError, match="not a label horizon"):
        walk_forward_conformal_coverage(_panel(TWO_SCALES), embargo_steps=-1)


def test_a_zero_embargo_is_admitted_for_a_zero_horizon_label() -> None:
    """`HOLDOUT_CONFORMAL_EMBARGO_STEPS` records 0 for `us_firm_characteristics`' labels: the
    row is dated by the month the return was earned, so the outcome is realised at the
    observation and no residual reaches forward.
    """
    row = walk_forward_conformal_coverage(_panel(TWO_SCALES), embargo_steps=0, levels=(0.80,))[0]
    assert row["n_test"] > 0
