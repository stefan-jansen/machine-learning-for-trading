"""A causal hash must not come back as "unknown" from a registry that holds it.

`Result.open` consults `training_runs`, `prediction_sets` and `backtest_runs`, and then
raises `KeyError: Unknown result hash`. Causal runs live in `causal_runs` in the same
registry file, so a session checking whether a hash can be migrated was told the row did
not exist - about a row sitting in the same database. That is the one answer that leads
away from the truth, and it mattered because `migrate_equivalent_training_identity` does
not reach causal rows either, so the honest answer is "it exists and it will refit".
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from case_studies.research.results import Result
from case_studies.utils.registry.registration import register_causal_run

SPEC = '{"family":"causal_dml","identity_version":3}'


def _register(case_dir, causal_hash: str) -> None:
    register_causal_run(
        "test_case",
        causal_hash,
        label="fwd_ret_5d",
        treatment="ivrv_spread",
        confounders_json='["rv_20"]',
        embargo=10,
        n_folds=5,
        n_obs=100,
        dml_effect=-0.02,
        dml_se_hac=0.02,
        p_value_hac=0.25,
        naive_effect=-0.02,
        confounding_bias_pct=-0.5,
        refutation_p=0.25,
        refutation_n_successful=10,
        spec_json=SPEC,
        notebook="12_causal_dml",
        started_at="first",
        elapsed_s=1.0,
        case_dir=case_dir,
    )


def _study(case_dir):
    return SimpleNamespace(
        root=case_dir,
        output_root=None,
        release_case_root=case_dir,
        read_only=False,
        case_study="test_case",
    )


def test_a_causal_hash_is_named_rather_than_called_unknown(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    _register(case_dir, "ca6a93f0d776")

    with pytest.raises(KeyError) as excinfo:
        Result.open(_study(case_dir), "ca6a93f0d776")

    message = str(excinfo.value)
    assert "Unknown result hash" not in message
    assert "causal run" in message
    # The reader needs the call that does work, not only the news that this one does not.
    assert "CausalResult.open" in message
    # And the pricing consequence, which is why the wrong answer was expensive.
    assert "migrate_equivalent_training_identity" in message


def test_a_hash_in_no_table_is_still_unknown(tmp_path) -> None:
    """The causal check must not turn every miss into a causal claim."""
    case_dir = tmp_path / "test_case"
    _register(case_dir, "ca6a93f0d776")

    with pytest.raises(KeyError, match="Unknown result hash"):
        Result.open(_study(case_dir), "deadbeefdead")


def test_a_registry_with_no_causal_table_is_still_unknown(tmp_path) -> None:
    """A release seeded before any causal run has no `causal_runs`, and naming an absent
    table in a SELECT is an error rather than an empty result."""
    case_dir = tmp_path / "test_case"
    (case_dir / "run_log").mkdir(parents=True)
    import sqlite3

    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute("CREATE TABLE training_runs (training_hash TEXT)")

    with pytest.raises(KeyError, match="Unknown result hash"):
        Result.open(_study(case_dir), "deadbeefdead")
