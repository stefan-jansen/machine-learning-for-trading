"""A refutation that was asked for and did not arrive leaves the result incomplete.

`register_causal_run` writes a null `refutation_p` and a null draw count when the placebo
refits mostly failed, and `CausalResult.complete` used to return true anyway. Two things
follow, and the second is the reason this is not merely a formatting bug: a reader
formats and divides nulls, and the runner's cache probe serves that row to every later
run, so the fit that lost its refutation is also the last fit anyone performs. Nothing
after it recomputes what went missing.

`n_placebo` comes from the spec because "no refutation was asked for" and "one was asked
for and did not arrive" are indistinguishable in the columns, and only the first is
complete.
"""

from __future__ import annotations

import json

import pytest

from case_studies.research.causal import CausalResult

BASE_METRICS = {
    "n_obs": 640_333,
    "dml_effect": 0.0093,
    "dml_se_hac": 0.0049,
    "p_value_hac": 0.06,
    "naive_effect": 0.003,
    "confounding_bias_pct": -67.5,
    "refutation_p": 1.0,
    "refutation_n_successful": 100,
    "refutation_class": "Fails",
}


def _result(*, n_placebo: int, **metric_overrides) -> CausalResult:
    spec = {
        "family": "causal_dml",
        "identity_version": 3,
        "execution_tier": "canonical",
        "computation": {"refutation": {"n_placebo": n_placebo, "block_size": 12}},
    }
    return CausalResult(
        study=None,
        hash="aaaa11112222",
        spec=json.loads(json.dumps(spec)),
        metrics={**BASE_METRICS, **metric_overrides},
        execution_tier="canonical",
    )


class TestARefutationThatWasAskedFor:
    def test_a_run_that_produced_one_is_complete(self) -> None:
        assert _result(n_placebo=100).complete

    def test_a_run_that_did_not_produce_one_is_not_complete(self) -> None:
        assert not _result(n_placebo=100, refutation_p=None).complete

    def test_a_missing_draw_count_alone_does_not_make_it_incomplete(self) -> None:
        """`refutation_n_successful` arrived with a migration, so NULL there is a schema
        fact rather than a missing refutation.

        Requiring it would declare every row written before that column existed
        incomplete and send runs back to refit results already on record - the opposite
        of the error this contract exists to catch. A reader needing the count, for the
        p-value's floor or for a verdict, handles its absence itself.
        """
        assert _result(n_placebo=100, refutation_n_successful=None).complete


class TestARefutationThatWasNot:
    def test_a_configuration_declaring_no_placebos_is_complete_without_one(self) -> None:
        """Otherwise the contract would refuse a design that never asked for a refutation.

        This is the case that makes reading `n_placebo` necessary rather than tidy: on the
        metrics alone it is identical to the failure above.
        """
        result = _result(n_placebo=0, refutation_p=None, refutation_n_successful=None)
        assert result.complete

    def test_a_missing_refutation_block_is_read_as_none_asked_for(self) -> None:
        result = _result(n_placebo=0, refutation_p=None, refutation_n_successful=None)
        del result.spec["computation"]["refutation"]
        assert result.complete


class TestTheRestOfTheContractStillHolds:
    @pytest.mark.parametrize(
        "broken",
        [{"n_obs": 0}, {"dml_effect": None}, {"dml_se_hac": None}],
    )
    def test_a_complete_refutation_does_not_excuse_a_missing_estimate(self, broken) -> None:
        assert not _result(n_placebo=100, **broken).complete
