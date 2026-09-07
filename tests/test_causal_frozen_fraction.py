"""The placebo frozen fraction survives into the artifact, so a reader can regenerate it.

`run_dml_analysis` computes `placebo_frozen_fraction` and warns that it must be read
alongside the p-value. `causal_runs` had no column for it, so a notebook that reads its
result back from the registry - which is what makes a re-run reproduce - could not report
the number its own warning names.

The gap is invisible exactly when it matters. On a cache-hit re-run the fit does not
execute, so the warning does not fire either: the diagnostic was present in a fresh run's
stdout and absent from what a reader regenerates. A frozen draw is one where the block is
long enough that permutation cannot move the rows, so the fraction rises with block size,
and #623 multiplied block sizes across the fleet by between 12x and 50x.

It is a metric and not part of the identity. `causal_hash` comes from `spec_json`, so the
column moves no hash and invalidates no registered row - the same trade
`refutation_n_successful` and `refutation_placebo_json` were added under.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from case_studies.research.causal import CausalResult
from case_studies.utils import causal
from case_studies.utils.registry.registration import register_causal_run

FROZEN = 0.023


def _results(frozen_fraction: float | None) -> dict:
    refutation = {"empirical_p": 0.1, "n_successful": 40}
    if frozen_fraction is not None:
        refutation["placebo_frozen_fraction"] = frozen_fraction
    return {
        "dml_result": {"theta": 0.02, "se_hac": 0.01, "n_obs": 120},
        "p_value_hac": 0.04,
        "naive_effect": 0.03,
        "confounding_bias_pct": 50.0,
        "refutation": refutation,
    }


def _study(case_dir):
    return SimpleNamespace(root=case_dir, output_root=None, release_case_root=case_dir)


def _register(case_dir, frozen_fraction: float | None) -> str:
    return causal.register_causal_run(
        "test_case",
        "fwd_ret_5d",
        _results(frozen_fraction),
        treatment_col="ivrv_spread",
        confounder_cols=["rv_20"],
        n_folds=5,
        embargo=10,
        block_size=126,
        n_placebo=40,
        case_dir=case_dir,
    )


def test_a_registered_run_carries_the_number_its_own_warning_names(tmp_path) -> None:
    """The whole chain: what `run_dml_analysis` returns reaches the row, and reading the
    row back - which is a cache hit, with no fit and therefore no warning - reports it."""
    case_dir = tmp_path / "test_case"
    causal_hash = _register(case_dir, FROZEN)

    result = CausalResult.open(_study(case_dir), causal_hash)

    assert result.metrics["placebo_frozen_fraction"] == pytest.approx(FROZEN)


def test_a_run_that_computed_no_fraction_reads_back_as_none(tmp_path) -> None:
    """A refutation that produced too few successful placebos returns an empty dict, and
    an absent diagnostic must read as absent rather than as zero - zero is the claim that
    permutation moved every row."""
    case_dir = tmp_path / "test_case"
    causal_hash = _register(case_dir, None)

    assert (
        CausalResult.open(_study(case_dir), causal_hash).metrics["placebo_frozen_fraction"] is None
    )


def test_a_registry_written_before_the_column_existed_can_still_be_read(tmp_path) -> None:
    """`CausalResult.open` reads through a plain connection rather than the migrating
    opener, on purpose, so naming the column unconditionally raises OperationalError on
    every pre-migration registry - the failure `refutation_n_successful` already caused
    once."""
    case_dir = tmp_path / "test_case"
    causal_hash = _register(case_dir, FROZEN)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute("ALTER TABLE causal_runs DROP COLUMN refutation_frozen_fraction")

    result = CausalResult.open(_study(case_dir), causal_hash)

    assert result.metrics["placebo_frozen_fraction"] is None
    assert result.metrics["refutation_p"] == pytest.approx(0.1)


def _register_immutable(case_dir, frozen_fraction: float | None, started_at: str) -> None:
    """An identity-version-3 row, which is the immutable kind. The wrapper in `causal.py`
    registers specs that carry no version, and `SUPPORTED_IDENTITY_VERSIONS` is what turns
    the conflict check on, so the immutability contract has to be exercised here."""
    register_causal_run(
        "test_case",
        "causal_immutable",
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
        refutation_p=0.1,
        refutation_n_successful=40,
        refutation_frozen_fraction=frozen_fraction,
        spec_json='{"family":"causal_dml","identity_version":3}',
        notebook="12_causal_dml",
        started_at=started_at,
        elapsed_s=1.0,
        case_dir=case_dir,
    )


def test_filling_the_column_on_an_existing_row_is_not_a_conflict(tmp_path) -> None:
    """A row registered before the column existed carries NULL, and a re-registration that
    recomputes the fraction fills it. Treating that as an immutability conflict would make
    the migration break re-registration of results that are identical."""
    case_dir = tmp_path / "test_case"
    _register_immutable(case_dir, None, "first")
    _register_immutable(case_dir, FROZEN, "second")

    assert CausalResult.open(_study(case_dir), "causal_immutable").metrics[
        "placebo_frozen_fraction"
    ] == pytest.approx(FROZEN)


def test_a_stored_fraction_that_changes_is_still_a_conflict(tmp_path) -> None:
    """Filling a NULL is a schema gap closing. A stored value moving is a different run
    claiming an identity that is immutable, and must still be refused - by name, so the
    message says which column would change."""
    case_dir = tmp_path / "test_case"
    _register_immutable(case_dir, FROZEN, "first")
    with pytest.raises(ValueError, match="refutation_frozen_fraction"):
        _register_immutable(case_dir, FROZEN * 2, "second")
