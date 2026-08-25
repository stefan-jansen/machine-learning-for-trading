"""Idempotence contract for causal-result registration."""

from __future__ import annotations

import sqlite3

from case_studies.utils.causal import REFUTATION_ALPHA, classify_refutation
from case_studies.utils.registry.registration import register_causal_run

# A spec whose identity_version is supported is what puts a row on the immutable path;
# without one, register_causal_run overwrites freely and an immutability test proves
# nothing.
IMMUTABLE_SPEC = '{"family":"causal_dml","identity_version":3}'


def _register(case_dir, *, effect: float, started_at: str, elapsed_s: float) -> None:
    register_causal_run(
        "test_case",
        "causal123",
        label="fwd_ret_5d",
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
        refutation_p=0.01,
        spec_json='{"family":"causal_dml"}',
        notebook="12_causal_dml",
        started_at=started_at,
        elapsed_s=elapsed_s,
        case_dir=case_dir,
    )


def _row(case_dir) -> tuple:
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        return db.execute(
            "SELECT dml_effect, started_at, elapsed_s, git_commit, created_at "
            "FROM causal_runs WHERE causal_hash='causal123'"
        ).fetchone()


def test_identical_causal_result_does_not_refresh_execution_provenance(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    _register(case_dir, effect=-0.0228, started_at="first", elapsed_s=10.0)
    first = _row(case_dir)

    _register(case_dir, effect=-0.0228, started_at="second", elapsed_s=20.0)
    assert _row(case_dir) == first

    _register(case_dir, effect=-0.0230, started_at="third", elapsed_s=30.0)
    changed = _row(case_dir)
    assert changed[0:3] == (-0.0230, "third", 30.0)
    assert changed[4] == first[4]


def _register_immutable(case_dir, **overrides) -> None:
    """Registers through the immutable path, which `_register` above does not reach.
    `register_causal_run` enforces immutability only when the spec carries an
    `identity_version` it recognises, so a spec without one is accepted and silently
    overwritten. A test built on such a spec passes whatever the check does, including
    nothing. Every assertion about a conflict, or about a conflict correctly not being
    raised, has to come through here."""
    fields = dict(
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
        refutation_p=1 / 11,
        refutation_n_successful=10,
        spec_json=IMMUTABLE_SPEC,
        notebook="12_causal_dml",
        started_at="first",
        elapsed_s=1.0,
        case_dir=case_dir,
    )
    fields.update(overrides)
    register_causal_run("test_case", "causal_immutable", **fields)


def _stored(case_dir, column: str):
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        return db.execute(
            f"SELECT {column} FROM causal_runs WHERE causal_hash='causal_immutable'"
        ).fetchone()[0]


def test_the_draw_count_is_persisted_so_the_verdict_can_be_recomputed(tmp_path) -> None:
    """A reader holding only `refutation_p` cannot tell a run that failed from one whose
    draws could never have rejected. The plus-one correction floors the p-value at
    1 / (n + 1), so a preview run of ten draws reports 0.09 at best and any bare
    threshold republishes it as a verdict the data never supported. Persisting the count
    is what lets every reader reach the same verdict from the same rule."""
    case_dir = tmp_path / "test_case"
    _register_immutable(case_dir)

    assert _stored(case_dir, "refutation_n_successful") == 10
    assert classify_refutation(_stored(case_dir, "refutation_p"), 10) == "Underpowered"
    # The two-way rule a reader would otherwise apply to the same p-value.
    assert (_stored(case_dir, "refutation_p") < REFUTATION_ALPHA) is False


def test_filling_a_column_that_did_not_exist_yet_is_not_a_conflict(tmp_path) -> None:
    """Rows written before `refutation_n_successful` existed carry NULL there. Recording
    the count on a re-registration of the identical result must not read as the result
    having changed: an upgrade that breaks re-registration of unchanged results is the
    same shape as a fix that forces a refit without moving a number."""
    case_dir = tmp_path / "test_case"
    _register_immutable(case_dir)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute("UPDATE causal_runs SET refutation_n_successful = NULL")

    _register_immutable(case_dir, refutation_n_successful=100, started_at="second")

    assert _stored(case_dir, "refutation_n_successful") == 100


def test_a_value_that_actually_changes_is_still_a_conflict(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    _register_immutable(case_dir)

    try:
        _register_immutable(case_dir, dml_effect=-0.9999)
    except ValueError as error:
        assert "dml_effect" in str(error)
    else:
        raise AssertionError("a changed effect must not be accepted onto an immutable row")


def test_a_registry_written_before_the_draw_count_existed_gains_the_column(tmp_path) -> None:
    case_dir = tmp_path / "test_case"
    _register_immutable(case_dir)
    db_path = case_dir / "run_log" / "registry.db"
    with sqlite3.connect(db_path) as db:
        db.execute("ALTER TABLE causal_runs DROP COLUMN refutation_n_successful")

    _register_immutable(case_dir, started_at="second")

    with sqlite3.connect(db_path) as db:
        assert "refutation_n_successful" in {
            row[1] for row in db.execute("PRAGMA table_info(causal_runs)").fetchall()
        }


def test_a_nullable_column_gaining_a_value_is_still_a_conflict(tmp_path) -> None:
    """`refutation_p` is NULL whenever the refutation produced too few successful
    placebos, so a stored NULL there means "this run could not answer", not "the registry
    had nowhere to put it". A later run that does produce a p-value has changed. Only a
    column a migration added may be filled on NULL; widening that to every nullable column
    writes a changed result onto an immutable row."""
    case_dir = tmp_path / "test_case"
    _register_immutable(case_dir, refutation_p=None)

    try:
        _register_immutable(case_dir, refutation_p=0.01, started_at="second")
    except ValueError as error:
        assert "refutation_p" in str(error)
    else:
        raise AssertionError("a refutation p-value appearing where there was none is a change")


def test_a_migrated_column_that_changes_from_a_stored_value_names_itself(tmp_path) -> None:
    """The backfill exists for NULL, not for the column. A migrated column whose stored
    value is present and different - a recording convention that changes the count, or a
    re-registration passing None where a number was stored - is a real difference, and
    excluding it from the message by name raises naming nothing. That empty message is
    what this file already fixed once from the NULL side."""
    case_dir = tmp_path / "test_case"
    _register_immutable(case_dir, refutation_n_successful=1000)

    try:
        _register_immutable(case_dir, refutation_n_successful=998, started_at="second")
    except ValueError as error:
        assert "refutation_n_successful" in str(error), f"the conflict named nothing: {error}"
    else:
        raise AssertionError("a changed draw count on an immutable row must not be accepted")


def _wrapper_hash(case_dir, **knobs) -> str:
    """Register through the notebook-facing wrapper and return the identity it computed."""
    from case_studies.utils.causal import register_causal_run as wrapper

    results = {
        "dml_result": {"n_obs": 100, "theta": -0.0228, "se_hac": 0.02},
        "p_value_hac": 0.25,
        "naive_effect": -0.02,
        "confounding_bias_pct": -0.5,
        "refutation": {"empirical_p": 0.01},
    }
    return wrapper(
        "test_case",
        "fwd_ret_5d",
        results,
        treatment_col="mom_skip",
        confounder_cols=["vol_21"],
        n_folds=5,
        embargo=10,
        case_dir=case_dir,
        **knobs,
    )


def test_entity_cap_is_part_of_the_causal_identity(tmp_path) -> None:
    """A panel thinned to N entities is a different estimate, not the same one re-run."""
    case_dir = tmp_path / "test_case"
    full = _wrapper_hash(case_dir, max_symbols=0)
    reduced = _wrapper_hash(case_dir, max_symbols=5)
    assert full != reduced


def test_wrapper_writes_the_registry_where_it_was_told(tmp_path) -> None:
    """The wrapper accepted `case_dir` and dropped it, falling back to the real case directory."""
    case_dir = tmp_path / "test_case"
    _wrapper_hash(case_dir)
    assert (case_dir / "run_log" / "registry.db").is_file()
