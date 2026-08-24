"""Reading a causal result must survive a registry written before the column existed.

`refutation_n_successful` arrived with a migration. `CausalResult.open` reads through a
plain connection rather than the migrating opener - on purpose, since a read that
rewrites the schema of a registry it was only asked to look at is a write - so naming the
column unconditionally raised OperationalError on every pre-migration registry. The cache
probe in `run_resolved_causal_request` catches only KeyError, so that error escaped past
the probe, the registering write that would have migrated the database never ran, and the
next attempt failed the same way.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from case_studies.research.causal import CausalResult
from case_studies.utils.registry.registration import register_causal_run

SPEC = '{"family":"causal_dml","identity_version":3}'


def _register(case_dir, *, refutation_p, n_successful) -> None:
    register_causal_run(
        "test_case",
        "causal_read",
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
        refutation_p=refutation_p,
        refutation_n_successful=n_successful,
        spec_json=SPEC,
        notebook="12_causal_dml",
        started_at="first",
        elapsed_s=1.0,
        case_dir=case_dir,
    )


def _study(case_dir):
    return SimpleNamespace(root=case_dir, output_root=None)


def _drop_the_column(case_dir) -> None:
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute("ALTER TABLE causal_runs DROP COLUMN refutation_n_successful")


def test_a_registry_without_the_draw_count_can_still_be_read(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    _register(case_dir, refutation_p=1 / 11, n_successful=10)
    _drop_the_column(case_dir)

    result = CausalResult.open(_study(case_dir), "causal_read")

    assert result.metrics["refutation_p"] == pytest.approx(1 / 11)
    assert result.metrics["refutation_n_successful"] is None


def test_an_unknown_draw_count_yields_no_verdict(tmp_path) -> None:
    """A p-value of 0.09 is "Fails" under a bare threshold and "Underpowered" under ten
    draws. With the count unknown the reader cannot tell which, and reporting the first is
    exactly the mistake deriving the class centrally exists to prevent."""
    case_dir = tmp_path / "test_case"
    _register(case_dir, refutation_p=1 / 11, n_successful=10)
    _drop_the_column(case_dir)

    result = CausalResult.open(_study(case_dir), "causal_read")

    assert result.metrics["refutation_class"] is None


def test_a_known_draw_count_still_yields_its_verdict(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    _register(case_dir, refutation_p=1 / 11, n_successful=10)

    result = CausalResult.open(_study(case_dir), "causal_read")

    assert result.metrics["refutation_n_successful"] == 10
    assert result.metrics["refutation_class"] == "Underpowered"


def test_a_null_draw_count_in_a_present_column_also_yields_no_verdict(tmp_path) -> None:
    """The column present and NULL is a different state from the column absent, and only
    this one isolates the classification. With the column dropped both tests above fail on
    the read, so neither of them can tell whether the verdict rule was fixed."""
    case_dir = tmp_path / "test_case"
    _register(case_dir, refutation_p=1 / 11, n_successful=10)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute("UPDATE causal_runs SET refutation_n_successful = NULL")

    result = CausalResult.open(_study(case_dir), "causal_read")

    assert result.metrics["refutation_n_successful"] is None
    assert result.metrics["refutation_class"] is None
