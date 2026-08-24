"""The two properties the block-permutation refutation rests on.

Both were broken together, and the pair is what made the test easy to pass and
its result impossible: `block_size` was taken from `embargo_periods`, so at the
common `embargo_periods = 1` the "block" permutation was an iid shuffle that
destroys exactly the serial dependence the placebo is supposed to keep; and the
p-value omitted the plus-one correction, so a run in which no placebo reached
the observed effect published `p = 0.000`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from case_studies.utils.causal import (
    REFUTATION_ALPHA,
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


class TestUnderpoweredRefutation:
    """The plus-one correction floors the p-value, so few draws cannot reject."""

    def test_nineteen_draws_cannot_reach_the_alpha_the_classifier_tests(self) -> None:
        # The most extreme result 19 draws can produce: no placebo reaches the observed
        # effect. Even then the reported p-value is at the threshold, not below it.
        strongest = empirical_permutation_p(np.zeros(19), observed_effect=1.0)

        assert strongest == 1 / 20
        assert not strongest < REFUTATION_ALPHA

    def test_a_draw_count_that_cannot_reject_is_reported_as_underpowered(self) -> None:
        # "Fails" here would be untrue by construction - the same defect as p = 0.000
        # at the other end of the range - because no data could have produced "Passes".
        assert classify_refutation(1 / 20, n_successful=19) == "Underpowered"
        assert classify_refutation(1.0, n_successful=10) == "Underpowered"

    def test_twenty_draws_are_enough_to_answer(self) -> None:
        assert classify_refutation(1 / 21, n_successful=20) == "Passes"
        assert classify_refutation(0.9, n_successful=20) == "Fails"

    def test_a_caller_without_the_draw_count_keeps_the_two_way_answer(self) -> None:
        assert classify_refutation(0.01) == "Passes"
        assert classify_refutation(0.9) == "Fails"


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
