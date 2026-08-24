"""The three properties the block-permutation refutation rests on.

All three were broken together, which is what made the test easy to pass and its
result impossible to read:

- The block was sized by the label horizon alone. On this panel the horizon, the
  embargo and the cadence all coincide at one bar, so the "block" permutation was
  a full within-symbol shuffle - destroying exactly the serial dependence the
  placebo exists to keep - even though the treatment is a 14-day z-score
  autocorrelated over 42 bars. The block now spans the longer of the two scales;
  `tests/test_causal_adapter.py` pins that resolution.
- The p-value omitted the plus-one correction, so a run in which no placebo
  reached the observed effect published `p = 0.000`.
- The pass/fail label was emitted at placebo counts too small to produce it. With
  the correction in place, fewer than 20 successful placebos cannot score below
  5 %, so the label read "Fails" regardless of the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from case_studies.utils.causal import (
    REFUTATION_UNRESOLVED,
    _treatment_persistence_steps,
    block_permute,
    classify_refutation,
    empirical_permutation_p,
)
from utils.paths import REPO_ROOT


class TestEmpiricalPermutationP:
    def test_no_placebo_reaches_the_effect_reports_the_resolution_not_zero(self) -> None:
        """100 permutations cannot establish p = 0; they can establish p <= 1/101."""
        placebo = np.full(100, 0.01)

        p = empirical_permutation_p(placebo, observed_effect=5.0)

        assert p == pytest.approx(1.0 / 101.0)
        assert p > 0.0

    def test_the_floor_is_set_by_how_many_permutations_were_run(self) -> None:
        few = empirical_permutation_p(np.full(10, 0.01), observed_effect=5.0)
        many = empirical_permutation_p(np.full(1000, 0.01), observed_effect=5.0)

        assert few == pytest.approx(1.0 / 11.0)
        assert many == pytest.approx(1.0 / 1001.0)
        assert many < few

    def test_every_placebo_reaching_the_effect_reports_one(self) -> None:
        """An effect the permutation reproduces every time is not evidence."""
        assert empirical_permutation_p(np.full(20, 1.0), observed_effect=0.5) == 1.0

    def test_the_count_is_two_sided(self) -> None:
        """A negative placebo of the same magnitude is as extreme as a positive one."""
        placebo = np.array([-2.0, -2.0, 0.0, 0.0])

        assert empirical_permutation_p(placebo, observed_effect=1.0) == pytest.approx(3.0 / 5.0)
        assert empirical_permutation_p(placebo, observed_effect=-1.0) == pytest.approx(3.0 / 5.0)

    def test_the_fraction_is_otherwise_the_plain_one(self) -> None:
        """Away from the boundary the correction is the only difference."""
        placebo = np.array([3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        assert empirical_permutation_p(placebo, observed_effect=1.0) == pytest.approx(4.0 / 10.0)


class TestBlockSizePreservesDependence:
    """Why taking block_size from a unit embargo emptied the refutation."""

    @staticmethod
    def _one_symbol(n: int = 24):
        dates = pd.to_datetime([f"2024-01-{d + 1:02d}" for d in range(n)]).to_numpy()
        units = np.array(["BTC"] * n)
        return np.arange(n, dtype=float), dates, units

    def test_block_size_one_leaves_no_contiguous_pair(self) -> None:
        """The degenerate case: block_size = 1 is an iid shuffle."""
        arr, dates, units = self._one_symbol()

        permuted = block_permute(
            arr,
            block_size=1,
            rng=np.random.default_rng(0),
            groups=dates,
            units=units,
            expected_step="1D",
        )

        adjacent = np.sum(np.diff(permuted) == 1.0)
        assert sorted(permuted) == sorted(arr)
        assert adjacent <= 2, "a unit block size left the series essentially in order"

    def test_a_horizon_sized_block_keeps_the_series_locally_intact(self) -> None:
        """What the permutation is for: shuffle position, keep local structure."""
        arr, dates, units = self._one_symbol()
        block = 6

        permuted = block_permute(
            arr,
            block_size=block,
            rng=np.random.default_rng(0),
            groups=dates,
            units=units,
            expected_step="1D",
        )

        adjacent = np.sum(np.diff(permuted) == 1.0)
        assert sorted(permuted) == sorted(arr)
        # Four blocks of six: five of every six steps stay contiguous.
        assert adjacent >= len(arr) - len(arr) // block - 1


class TestClassificationResolution:
    """A pass/fail label is only meaningful when the placebo count can produce it.

    With the plus-one correction the smallest attainable p-value is
    `1 / (n + 1)`, so below 20 successful placebos every run scores at or above
    5 % and the label reads "Fails" whatever the data show.
    """

    @pytest.mark.parametrize("n_placebo", [1, 10, 19])
    def test_too_few_placebos_report_no_resolution_rather_than_a_verdict(
        self, n_placebo: int
    ) -> None:
        smallest_attainable = 1.0 / (n_placebo + 1)

        assert classify_refutation(smallest_attainable, n_placebo) == REFUTATION_UNRESOLVED

    def test_twenty_placebos_are_enough_to_decide(self) -> None:
        """1/21 is below 5 %, so the smallest attainable p-value can now pass."""
        assert classify_refutation(1.0 / 21, 20) == "Passes"
        assert classify_refutation(0.5, 20) == "Fails"

    def test_the_count_is_optional_so_the_notebook_callers_keep_working(self) -> None:
        """Seven case-study notebooks call this with the p-value alone."""
        assert classify_refutation(0.01) == "Passes"
        assert classify_refutation(0.5) == "Fails"


class TestTreatmentWindowLookup:
    """`features.windows` is not one shape across the fleet, and the resolver is shared.

    Crypto declares suffix-keyed maps (`premium_zscore: {14d: 42}`), but etfs
    declares a bare int (`skip_recent: 21`) and lists (`momentum: [5, 10, 21,
    ...]`), and sp500_options mixes all three. A lookup that assumes the mapping
    raises AttributeError inside the shared DML resolver for a case study that
    declared nothing wrong.
    """

    def test_a_suffix_keyed_window_is_read(self) -> None:
        setup = {"features": {"windows": {"premium_zscore": {"7d": 21, "14d": 42}}}}

        assert _treatment_persistence_steps(setup, "premium_zscore_14d") == 42

    @pytest.mark.parametrize(
        ("windows", "treatment"),
        [
            ({"skip_recent": 21}, "skip_recent_6_1"),
            ({"momentum": [5, 10, 21, 42]}, "momentum_21d"),
            ({"vrp": None}, "vrp_21d"),
        ],
    )
    def test_a_shape_that_cannot_name_this_column_returns_none_instead_of_raising(
        self, windows: dict, treatment: str
    ) -> None:
        """Guessing which list element built the treatment would put a wrong number
        behind a right-looking block size, which is the defect this whole change fixes."""
        assert _treatment_persistence_steps({"features": {"windows": windows}}, treatment) is None

    def test_an_absent_register_is_not_an_error(self) -> None:
        assert _treatment_persistence_steps({}, "anything_14d") is None

    def test_the_real_crypto_register_resolves_the_declared_treatment(self) -> None:
        """The case that shipped wrong: an 8h horizon against a 42-bar treatment."""
        setup = yaml.safe_load(
            (REPO_ROOT / "case_studies/crypto_perps_funding/config/setup.yaml").read_text()
        )

        steps = _treatment_persistence_steps(setup, setup["causal"]["treatment"])

        assert steps == 42
