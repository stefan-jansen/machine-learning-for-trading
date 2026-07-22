"""Regression tests for the spine kill-gate helpers in
``case_studies.utils.strategy_analysis``.

These gates decide publication go/no-go for each case study. The functions
accept ``float | None`` but the values reaching them come from the registry,
where a paired bootstrap that could not be computed (no row in
``backtest_paired_metrics``) surfaces as ``NaN`` rather than ``None``. Because
every comparison against ``NaN`` is ``False``, an unguarded ``NaN`` silently
falls through to ``pass`` / ``straddles_zero`` instead of ``no_data`` — a
false green light on a strategy with no holdout evidence. The ``_missing``
guard treats ``NaN`` exactly like ``None``; these tests pin that behavior.
"""

import math

import pytest

from case_studies.utils import strategy_analysis as sa

NAN = math.nan


class TestCIStatus:
    @pytest.mark.parametrize("lo,hi", [(None, 1.0), (1.0, None), (None, None)])
    def test_none_bounds_are_no_data(self, lo, hi):
        assert sa.ci_status(lo, hi) == "no_data"

    @pytest.mark.parametrize("lo,hi", [(NAN, 1.0), (1.0, NAN), (NAN, NAN)])
    def test_nan_bounds_are_no_data(self, lo, hi):
        # Regression: an unguarded NaN previously returned "straddles_zero".
        assert sa.ci_status(lo, hi) == "no_data"

    def test_excludes_zero_positive(self):
        assert sa.ci_status(0.5, 1.5) == "excludes_zero_strong"

    def test_excludes_zero_negative(self):
        assert sa.ci_status(-1.5, -0.5) == "excludes_zero_strong"

    def test_straddles_zero(self):
        assert sa.ci_status(-0.5, 0.5) == "straddles_zero"


class TestGate1:
    def test_none_is_no_data(self):
        assert sa.gate1_validation_sharpe_geq_zero(None) == "no_data"

    def test_nan_is_no_data(self):
        # Regression: an unguarded NaN previously returned "fail".
        assert sa.gate1_validation_sharpe_geq_zero(NAN) == "no_data"

    def test_nonnegative_passes(self):
        assert sa.gate1_validation_sharpe_geq_zero(0.0) == "pass"
        assert sa.gate1_validation_sharpe_geq_zero(0.1) == "pass"

    def test_negative_fails(self):
        assert sa.gate1_validation_sharpe_geq_zero(-0.1) == "fail"


class TestGate2:
    def test_no_data_status_is_no_data(self):
        assert sa.gate2_holdout_diff_not_excludes_zero_negatively("no_data", 0.5) == "no_data"

    def test_none_diff_is_no_data(self):
        assert (
            sa.gate2_holdout_diff_not_excludes_zero_negatively("straddles_zero", None) == "no_data"
        )

    @pytest.mark.parametrize("status", ["straddles_zero", "excludes_zero_strong"])
    def test_nan_diff_is_no_data(self, status):
        # Regression: an unguarded NaN diff previously returned "pass" — a false
        # green light on a strategy whose holdout paired metric was never computed.
        assert sa.gate2_holdout_diff_not_excludes_zero_negatively(status, NAN) == "no_data"

    def test_strong_negative_fails(self):
        assert (
            sa.gate2_holdout_diff_not_excludes_zero_negatively("excludes_zero_strong", -0.5)
            == "fail"
        )

    def test_strong_positive_passes(self):
        assert (
            sa.gate2_holdout_diff_not_excludes_zero_negatively("excludes_zero_strong", 0.5)
            == "pass"
        )

    def test_straddles_zero_passes(self):
        assert sa.gate2_holdout_diff_not_excludes_zero_negatively("straddles_zero", -0.5) == "pass"


def test_gate_passes_maps_no_data_to_none():
    # no_data must serialize to None, never coerced to True.
    assert sa.gate_passes("pass") is True
    assert sa.gate_passes("fail") is False
    assert sa.gate_passes("no_data") is None
