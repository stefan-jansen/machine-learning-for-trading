"""A causal refit has to be recoverable, and nothing tested that it was.

`CausalResult.one` resolves a label to exactly one current identity. A refit under a
changed `case_studies/utils/causal.py` produces a second, and before this the registry
had no way to say which one is live: no supersedes column, no tiebreak, no recency rule.
`cme_futures` reached that state after re-running `11_causal_dml` - four canonical rows
for two labels - and `12_model_analysis` exited 1 with no way forward but hand-editing
the registry.

The mechanism mirrors `official_populations`, where a changed population under an
existing name must name the hash it supersedes. Recency is deliberately not the fallback:
`created_at` ties on a fast refit, and it would be the only recency rule in a registry
that is otherwise entirely spec-addressed.

Not one of the four tests in `tests/test_causal_result_read.py` covers two identities;
they are all about the refutation draw count. That absence is why this shipped.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from case_studies.research.causal import CausalResult
from case_studies.utils.registry.registration import register_causal_run

SPEC = '{"family":"causal_dml","identity_version":3}'
LABEL = "fwd_ret_5d"


def _register(case_dir, causal_hash, *, effect=-0.02, supersedes=None) -> None:
    register_causal_run(
        "test_case",
        causal_hash,
        label=LABEL,
        treatment="ivrv_spread",
        confounders_json='["rv_20"]',
        embargo=10,
        n_folds=5,
        n_obs=100,
        dml_effect=effect,
        dml_se_hac=0.02,
        p_value_hac=0.25,
        naive_effect=-0.02,
        confounding_bias_pct=-0.5,
        refutation_p=1 / 101,
        refutation_n_successful=100,
        spec_json=SPEC,
        notebook="11_causal_dml",
        started_at="first",
        elapsed_s=1.0,
        supersedes_hash=supersedes,
        case_dir=case_dir,
    )


def _study(case_dir):
    return SimpleNamespace(
        root=case_dir,
        output_root=None,
        storage_root=lambda _tier: case_dir,
    )


def test_one_identity_resolves(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    _register(case_dir, "causal_first")

    assert CausalResult.one(_study(case_dir), label=LABEL).hash == "causal_first"


def test_a_second_identity_is_refused_at_the_write(tmp_path) -> None:
    """Where the failure belongs: the run that caused it, not a downstream notebook.

    Registering freely and failing at read time is what actually happened, and it puts
    the error hours away from its cause and in a different notebook.
    """
    case_dir = tmp_path / "test_case"
    _register(case_dir, "causal_first")

    with pytest.raises(ValueError, match="SUPERSEDES_CAUSAL"):
        _register(case_dir, "causal_second", effect=-0.03)


def test_a_declared_refit_retires_its_predecessor(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    _register(case_dir, "causal_first")
    _register(case_dir, "causal_second", effect=-0.03, supersedes="causal_first")

    resolved = CausalResult.one(_study(case_dir), label=LABEL)
    assert resolved.hash == "causal_second"
    assert resolved.metrics["dml_effect"] == pytest.approx(-0.03)


def test_a_chain_of_two_refits_resolves_to_the_tip(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    _register(case_dir, "causal_first")
    _register(case_dir, "causal_second", effect=-0.03, supersedes="causal_first")
    _register(case_dir, "causal_third", effect=-0.04, supersedes="causal_second")

    assert CausalResult.one(_study(case_dir), label=LABEL).hash == "causal_third"


def test_superseding_something_that_is_not_current_is_refused(tmp_path) -> None:
    """The declaration is checked, not just recorded.

    A typo'd or already-retired predecessor would otherwise leave the newer run live and
    the older one live too, which is the state this whole mechanism exists to prevent -
    reached by the one route that looks like it was handled.
    """
    case_dir = tmp_path / "test_case"
    _register(case_dir, "causal_first")

    with pytest.raises(ValueError, match="not a current canonical identity"):
        _register(case_dir, "causal_second", effect=-0.03, supersedes="causal_typo")


def test_a_run_cannot_supersede_itself(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    with pytest.raises(ValueError, match="cannot supersede itself"):
        _register(case_dir, "causal_first", supersedes="causal_first")


def test_a_registry_written_before_the_column_can_still_be_read(tmp_path) -> None:
    """`CausalResult.one` reads through a plain connection, not the migrating opener.

    Naming `supersedes_hash` unconditionally would raise OperationalError on every
    registry written before it existed - the same defect `refutation_n_successful`
    caused once already, and the reason that read probes rather than assumes.
    """
    case_dir = tmp_path / "test_case"
    _register(case_dir, "causal_first")
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute("ALTER TABLE causal_runs DROP COLUMN supersedes_hash")

    assert CausalResult.one(_study(case_dir), label=LABEL).hash == "causal_first"


def test_two_undeclared_identities_say_what_to_do(tmp_path) -> None:
    """The state cme_futures was left in, and what a reader gets out of it.

    The write-time refusal above stops this arising from here on, but four rows already
    exist in a production registry. The message has to name the candidates and the
    parameter, because the person reading it is in the downstream notebook.
    """
    case_dir = tmp_path / "test_case"
    _register(case_dir, "causal_first")
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute(
            "INSERT INTO causal_runs (causal_hash, label, spec_json, created_at) "
            "VALUES ('causal_second', ?, ?, 'now')",
            (LABEL, SPEC),
        )

    with pytest.raises(ValueError, match="resolved to 2 identities") as raised:
        CausalResult.one(_study(case_dir), label=LABEL)
    assert "SUPERSEDES_CAUSAL" in str(raised.value)
    assert "causal_first" in str(raised.value)
