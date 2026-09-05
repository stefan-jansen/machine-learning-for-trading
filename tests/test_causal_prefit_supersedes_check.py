"""The refusal a causal run earns has to arrive before the fit, not after it.

`_causal_source_identity` hashes the whole of `case_studies/utils/causal.py` into the
resolved spec, so any edit to that module - including one made for a different case
study - gives every resolver-based fit a new identity. A notebook that declares no
predecessor then misses the cache, pays the full DML fit and every placebo refit, and is
refused at the write for leaving two current identities. On a panel of this size that is
an hour spent to be told a hash the registry could have named before the first fold
(ml4t/agent-workspace#953).

`check_causal_supersedes` calls the write-time rule itself rather than restating it, so
what these tests pin is that the two answer identically. A second copy of the condition
would either refuse a run that would have registered, or start one that cannot.
"""

from __future__ import annotations

import pytest

from case_studies.utils.registry.registration import (
    check_causal_supersedes,
    register_causal_run,
)

SPEC = '{"family":"causal_dml","identity_version":3,"execution_tier":"canonical"}'
LABEL = "fwd_ret_1m"


def _register(case_dir, causal_hash: str, *, supersedes: str | None = None) -> None:
    register_causal_run(
        "test_case",
        causal_hash,
        label=LABEL,
        treatment="r12_2",
        confounders_json='["Beta"]',
        embargo=1,
        n_folds=5,
        n_obs=100,
        dml_effect=0.01,
        dml_se_hac=0.02,
        p_value_hac=0.25,
        naive_effect=0.02,
        confounding_bias_pct=1.0,
        refutation_p=0.5,
        refutation_n_successful=100,
        spec_json=SPEC,
        notebook="09_causal_dml",
        started_at="2026-08-27T00:00:00Z",
        elapsed_s=1.0,
        supersedes_hash=supersedes,
        case_dir=case_dir,
    )


def _check(case_dir, causal_hash: str, *, supersedes: str | None = None) -> None:
    check_causal_supersedes(
        "test_case",
        causal_hash,
        label=LABEL,
        tier="canonical",
        supersedes_hash=supersedes,
        case_dir=case_dir,
    )


class TestWhatItStops:
    def test_a_second_identity_declaring_nothing_is_refused_up_front(self, tmp_path) -> None:
        _register(tmp_path, "aaaa11112222")
        with pytest.raises(ValueError, match="set SUPERSEDES_CAUSAL to aaaa11112222"):
            _check(tmp_path, "bbbb33334444")

    def test_the_refusal_names_the_hash_the_run_has_to_declare(self, tmp_path) -> None:
        # The whole point of moving the check earlier: the message is actionable, and it
        # arrives while acting on it is still cheap.
        _register(tmp_path, "aaaa11112222")
        with pytest.raises(ValueError) as raised:
            _check(tmp_path, "bbbb33334444")
        assert "aaaa11112222" in str(raised.value)

    def test_a_declaration_naming_something_the_registry_does_not_hold_is_refused(
        self, tmp_path
    ) -> None:
        _register(tmp_path, "aaaa11112222")
        with pytest.raises(ValueError, match="not a current canonical identity"):
            _check(tmp_path, "bbbb33334444", supersedes="feedfacefeed")


class TestWhatItLetsThrough:
    def test_the_first_identity_for_a_label_needs_no_predecessor(self, tmp_path) -> None:
        _check(tmp_path, "aaaa11112222")

    def test_a_reader_with_no_registry_at_all_is_not_refused(self, tmp_path) -> None:
        # The ordinary state of a clean clone: `run_log/` is gitignored, so the first
        # person to run this notebook has no causal rows and nothing to retire.
        _check(tmp_path / "fresh", "aaaa11112222")

    def test_a_declared_predecessor_the_registry_holds_passes(self, tmp_path) -> None:
        _register(tmp_path, "aaaa11112222")
        _check(tmp_path, "bbbb33334444", supersedes="aaaa11112222")

    def test_re_running_the_identity_already_on_record_passes(self, tmp_path) -> None:
        # A re-run resolves the same hash and the runner serves it from the cache, but the
        # check runs on the path where the cache missed for some other reason, so it must
        # not refuse the row it is about to rewrite.
        _register(tmp_path, "aaaa11112222")
        _check(tmp_path, "aaaa11112222")


class TestItIsTheSameRule:
    @pytest.mark.parametrize(
        ("declared", "expected"),
        [(None, False), ("feedfacefeed", False), ("aaaa11112222", True)],
    )
    def test_the_check_and_the_write_agree_on_every_declaration(
        self, tmp_path, declared, expected
    ) -> None:
        """A disagreement here is the defect this shares one derivation to prevent.

        Either direction is a real failure: a check stricter than the write refuses a run
        that would have registered, and a check looser than the write is the hour of
        compute this exists to save, spent anyway.
        """
        _register(tmp_path, "aaaa11112222")

        def survives(call) -> bool:
            try:
                call()
            except ValueError:
                return False
            return True

        checked = survives(lambda: _check(tmp_path, "bbbb33334444", supersedes=declared))
        written = survives(lambda: _register(tmp_path, "bbbb33334444", supersedes=declared))
        assert checked == written == expected
