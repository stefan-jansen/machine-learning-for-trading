from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import CandidateSet, PredictionResult, Result, Study
from case_studies.utils.registry import (
    backtest_hash_from_parts,
    prediction_hash_from_parts,
    training_hash_from_spec,
)
from case_studies.utils.registry.completeness import evaluate_prediction_coverage
from case_studies.utils.registry.registration import register_backtest_run
from case_studies.utils.registry.store import _open_registry
from tests.test_research_workspace import _seed_release


def _study(tmp_path: Path) -> Study:
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )


def _training_spec(**changes) -> dict:
    spec = {
        "identity_version": 2,
        "family": "linear",
        "label": "fwd_ret_21d",
        "label_artifact": "label-a",
        "feature_artifacts": {"financial": "features-a"},
        "feature_names": ["momentum", "volatility"],
        "cv": {"folds": [{"fold": 0, "val_start": "2024-01-05"}]},
        "model": {"class": "Ridge", "params": {"alpha": 1.0}},
        "numerics": {"seed": 42, "precision": "float64"},
        "execution_tier": "canonical",
        "seed": 42,
    }
    spec.update(changes)
    return spec


def _predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": ["2024-01-05", "2024-01-05"],
            "fold_id": [0, 0],
            "y_true": [0.01, -0.02],
            "y_score": [0.02, -0.01],
        }
    ).with_columns(pl.col("timestamp").str.to_date())


def test_prediction_coverage_preserves_cme_product_and_position_keys() -> None:
    expected = pl.DataFrame(
        {
            "product": ["ES", "ES"],
            "position": [1, 2],
            "timestamp": ["2024-01-05", "2024-01-05"],
            "fold": [0, 0],
        }
    )
    predictions = expected.rename({"fold": "fold_id"}).with_columns(
        pl.Series("y_score", [0.1, 0.2])
    )

    coverage = evaluate_prediction_coverage(expected, predictions)

    assert coverage.complete
    assert coverage.n_expected == 2


def test_version_2_identity_is_semantic_and_legacy_hashes_are_pinned() -> None:
    base = _training_spec()
    same = {**base, "display_name": "Reader experiment", "notebook_path": "/tmp/book.ipynb"}
    assert training_hash_from_spec(base) == training_hash_from_spec(same)

    for mutation in (
        {"label_artifact": "label-b"},
        {"feature_artifacts": {"financial": "features-b"}},
        {"cv": {"folds": [{"fold": 0, "val_start": "2024-01-06"}]}},
        {"model": {"class": "Ridge", "params": {"alpha": 2.0}}},
        {"execution_tier": "preview"},
    ):
        assert training_hash_from_spec(base) != training_hash_from_spec({**base, **mutation})

    legacy = {"family": "linear", "label": "fwd_ret_21d", "seed": 42}
    assert training_hash_from_spec(legacy) == "a32ccdc1db2e"
    assert prediction_hash_from_parts("abc", None, "validation") == "f8d8ffe712a3"
    assert backtest_hash_from_parts("pred", {"top_k": 10}) == "921c296be434"


def test_checkpoint_kind_is_part_of_version_2_prediction_identity() -> None:
    a = prediction_hash_from_parts(
        "training", 10, "validation", checkpoint_kind="epoch", identity_version=2
    )
    b = prediction_hash_from_parts(
        "training", 10, "validation", checkpoint_kind="tree_limit", identity_version=2
    )
    assert a != b


def test_legacy_result_reopens_without_being_inferred_complete(tmp_path: Path) -> None:
    study = _study(tmp_path)
    db = _open_registry(study.root)
    try:
        db.execute(
            "INSERT INTO training_runs "
            "(training_hash, family, label, spec_json, created_at) VALUES (?,?,?,?,?)",
            ("legacy-training", "linear", "fwd_ret_21d", "{}", "2024-01-01"),
        )
        db.execute(
            "INSERT INTO prediction_sets "
            "(prediction_hash, training_hash, split, created_at) VALUES (?,?,?,?)",
            ("legacy-prediction", "legacy-training", "validation", "2024-01-01"),
        )
        db.commit()
    finally:
        db.close()

    reopened = Result.open(study, "legacy-prediction")

    assert isinstance(reopened, PredictionResult)
    assert reopened.hash == "legacy-prediction"
    assert reopened.identity_version is None
    assert not reopened.complete
    assert reopened.coverage() is None


def test_read_only_legacy_registry_reopens_without_schema_writes(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    db_path = release / "case_studies" / "etfs" / "run_log" / "registry.db"
    with closing(sqlite3.connect(db_path)) as db:
        db.execute(
            "CREATE TABLE training_runs ("
            "training_hash TEXT PRIMARY KEY, family TEXT NOT NULL, label TEXT NOT NULL, "
            "spec_json TEXT, created_at TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE prediction_sets ("
            "prediction_hash TEXT PRIMARY KEY, training_hash TEXT NOT NULL, "
            "split TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO training_runs VALUES (?,?,?,?,?)",
            ("legacy-training", "linear", "fwd_ret_21d", "{}", "2024-01-01"),
        )
        db.execute(
            "INSERT INTO prediction_sets VALUES (?,?,?,?)",
            ("legacy-prediction", "legacy-training", "validation", "2024-01-01"),
        )
        db.commit()
    before = db_path.read_bytes()

    study = Study.open("etfs", release_root=release)
    reopened = Result.open(study, "legacy-prediction")

    assert reopened.identity_version is None
    assert not reopened.complete
    assert db_path.read_bytes() == before


def test_workspace_open_migrates_copied_legacy_registry(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    db_path = release / "case_studies" / "etfs" / "run_log" / "registry.db"
    with closing(sqlite3.connect(db_path)) as db:
        db.execute(
            "CREATE TABLE training_runs ("
            "training_hash TEXT PRIMARY KEY, family TEXT NOT NULL, label TEXT NOT NULL, "
            "spec_json TEXT, created_at TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE prediction_sets ("
            "prediction_hash TEXT PRIMARY KEY, training_hash TEXT NOT NULL, "
            "split TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        db.commit()

    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    with closing(sqlite3.connect(study.root / "run_log" / "registry.db")) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(training_runs)")}
        coverage_table = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'prediction_coverage'"
        ).fetchone()

    assert {"identity_version", "execution_tier"} <= columns
    assert coverage_table == (1,)

    training = study.results.register_training(_training_spec())
    frame = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )

    assert training.complete
    assert prediction.complete


def test_legacy_registry_schema_migrates_additively(tmp_path: Path) -> None:
    case_dir = tmp_path / "legacy" / "etfs"
    db_path = case_dir / "run_log" / "registry.db"
    db_path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(db_path)) as db:
        db.execute(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                label TEXT NOT NULL,
                config_name TEXT,
                spec_json TEXT,
                created_at TEXT NOT NULL,
                git_commit TEXT,
                entry_point TEXT
            )
            """
        )
        db.execute(
            "INSERT INTO training_runs "
            "(training_hash, family, label, spec_json, created_at) VALUES (?,?,?,?,?)",
            ("legacy-training", "linear", "fwd_ret_21d", "{}", "2024-01-01"),
        )
        db.commit()

    migrated = _open_registry(case_dir)
    try:
        columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(training_runs)").fetchall()
        }
        tables = {
            row[0]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        row = migrated.execute(
            "SELECT training_hash, identity_version, execution_tier FROM training_runs"
        ).fetchone()
    finally:
        migrated.close()

    assert {"identity_version", "execution_tier"} <= columns
    assert {"prediction_coverage", "candidate_sets"} <= tables
    assert row == ("legacy-training", None, None)


def test_invalid_coverage_registration_is_atomic(tmp_path: Path) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    actual = _predictions().head(1)
    expected = _predictions().select("symbol", "timestamp", "fold_id")

    with pytest.raises(ValueError, match="coverage"):
        study.results.publish_predictions(
            training,
            checkpoint_kind="final",
            checkpoint_value=None,
            split="validation",
            predictions=actual,
            expected_keys=expected,
        )

    with closing(sqlite3.connect(study.root / "run_log" / "registry.db")) as db:
        assert db.execute("SELECT COUNT(*) FROM prediction_sets").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM prediction_coverage").fetchone()[0] == 0
    assert not list((study.root / "run_log" / "predictions").glob("*"))


def test_version_2_registration_is_immutable(tmp_path: Path) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec(display_name="first"))
    assert (
        study.results.register_training(_training_spec(display_name="second")).hash == training.hash
    )
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )

    with pytest.raises(ValueError, match="artifact conflict"):
        study.results.publish_predictions(
            training,
            checkpoint_kind="final",
            checkpoint_value=None,
            split="validation",
            predictions=frame.with_columns((pl.col("y_score") + 1).alias("y_score")),
            expected_keys=expected,
        )

    assert prediction.load().get_column("y_score").to_list() == [0.02, -0.01]


def test_training_registration_preserves_conflicting_orphan_and_adopts_exact_retry(
    tmp_path: Path,
) -> None:
    study = _study(tmp_path)
    spec = _training_spec()
    training_hash = training_hash_from_spec(spec)
    artifact = study.root / "run_log" / "training" / training_hash / "spec.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    conflicting = {**spec, "seed": 99}
    artifact.write_text(json.dumps(conflicting))

    with pytest.raises(ValueError, match="training spec artifact conflict"):
        study.results.register_training(spec)

    assert json.loads(artifact.read_text()) == conflicting
    artifact.write_text(json.dumps(spec))
    training = study.results.register_training(spec)
    assert training.hash == training_hash


def test_prediction_registration_preserves_conflicting_orphan_and_adopts_exact_retry(
    tmp_path: Path,
) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    prediction_hash = prediction_hash_from_parts(
        training.hash,
        None,
        "validation",
        checkpoint_kind="final",
        identity_version=2,
    )
    artifact = study.root / "run_log" / "predictions" / prediction_hash / "predictions.parquet"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    conflicting = frame.with_columns((pl.col("y_score") + 1).alias("y_score"))
    conflicting.write_parquet(artifact)
    conflicting_bytes = artifact.read_bytes()

    with pytest.raises(ValueError, match="prediction artifact conflict"):
        study.results.publish_predictions(
            training,
            checkpoint_kind="final",
            checkpoint_value=None,
            split="validation",
            predictions=frame,
            expected_keys=expected,
        )

    assert artifact.read_bytes() == conflicting_bytes
    assert pl.read_parquet(artifact).equals(conflicting)
    frame.write_parquet(artifact)
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )
    assert prediction.hash == prediction_hash


def test_backtest_registration_preserves_conflicting_orphan_and_adopts_exact_retry(
    tmp_path: Path,
) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )
    result = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="vectorized",
    ).run(
        prices=pl.DataFrame(
            {
                "symbol": ["A", "B"],
                "timestamp": [date(2024, 1, 5), date(2024, 1, 5)],
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.5, 99.5],
                "volume": [1_000, 1_000],
            }
        )
    )
    artifact = study.root / "run_log" / "backtest" / result.hash / "daily_returns.parquet"
    original = pl.read_parquet(artifact)
    strategy_spec = result.spec()
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute("DELETE FROM backtest_metrics WHERE backtest_hash = ?", (result.hash,))
        db.execute("DELETE FROM backtest_runs WHERE backtest_hash = ?", (result.hash,))
        db.commit()
    conflicting = original.with_columns((pl.col("daily_return") + 0.01).alias("daily_return"))

    with pytest.raises(ValueError, match="backtest artifact conflict"):
        register_backtest_run(
            "etfs",
            prediction.hash,
            strategy_spec,
            returns=conflicting,
            metrics={"sharpe": 0.0, "sharpe_se_lo": 0.0},
            case_dir=study.root,
        )

    assert pl.read_parquet(artifact).equals(original)
    assert (
        register_backtest_run(
            "etfs",
            prediction.hash,
            strategy_spec,
            returns=original,
            metrics={"sharpe": 0.0, "sharpe_se_lo": 0.0},
            case_dir=study.root,
        )
        == result.hash
    )


def test_checkpoint_prediction_schema_must_match(tmp_path: Path) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    study.results.publish_predictions(
        training,
        checkpoint_kind="epoch",
        checkpoint_value=1,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )

    with pytest.raises(ValueError, match="schema"):
        study.results.publish_predictions(
            training,
            checkpoint_kind="epoch",
            checkpoint_value=2,
            split="validation",
            predictions=frame.with_columns(pl.lit("extra").alias("diagnostic")),
            expected_keys=expected,
        )


def test_preview_requires_identity_covered_reductions(tmp_path: Path) -> None:
    study = _study(tmp_path)
    assert study.output_root is not None

    with pytest.raises(ValueError, match="identity-cover"):
        study.results.register_training(
            _training_spec(execution_tier="preview"), execution_tier="preview"
        )

    assert not (study.output_root / ".preview" / "etfs" / "run_log").exists()


def test_candidate_sets_are_immutable_and_validate_protocols(tmp_path: Path) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    first = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )
    second = study.results.publish_predictions(
        training,
        checkpoint_kind="epoch",
        checkpoint_value=2,
        split="validation",
        predictions=frame.with_columns((pl.col("y_score") * 2).alias("y_score")),
        expected_keys=expected,
    )

    original = CandidateSet.create(study, "baseline", [first])
    extended = original.extend("baseline-plus", [second])

    assert original.members == (first.hash,)
    assert extended.members == (first.hash, second.hash)
    assert original.hash != extended.hash
    assert CandidateSet.open(study, original.hash).members == (first.hash,)

    other_training = study.results.register_training(
        _training_spec(cv={"folds": [{"fold": 1, "val_start": "2024-01-06"}]})
    )
    incompatible = study.results.publish_predictions(
        other_training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )
    with pytest.raises(ValueError, match="protocol"):
        CandidateSet.create(study, "invalid", [first, incompatible])

    comparison = {"comparable_fields": ["cv"]}
    ordered = CandidateSet.create(
        study,
        "cv-comparison",
        [first, incompatible],
        comparison_contract=comparison,
    )
    reversed_set = CandidateSet.create(
        study,
        "cv-comparison-reversed",
        [incompatible, first],
        comparison_contract=comparison,
    )
    third = study.results.publish_predictions(
        study.results.register_training(
            _training_spec(cv={"folds": [{"fold": 2, "val_start": "2024-01-07"}]})
        ),
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )
    extended_comparison = ordered.extend("cv-comparison-plus", [third])

    assert ordered.hash == reversed_set.hash
    assert ordered.comparison_contract["protocol"].get("cv") is None
    assert set(extended_comparison.members) == {first.hash, incompatible.hash, third.hash}


def test_a_changed_candidate_set_supersedes_the_one_it_replaces(tmp_path: Path) -> None:
    """A re-run that changes membership must say what it replaces, and the name must resolve.

    The stage that freezes a candidate set is re-run whenever anything upstream of it is
    corrected, and the admitted membership moves with it. Without a recorded lineage the name
    carries two live identities and every reader of it raises instead of resolving.
    """
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    first = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )
    second = study.results.publish_predictions(
        training,
        checkpoint_kind="epoch",
        checkpoint_value=2,
        split="validation",
        predictions=frame.with_columns((pl.col("y_score") * 2).alias("y_score")),
        expected_keys=expected,
    )

    original = CandidateSet.create(study, "pool", [first])
    assert CandidateSet.one(study, name="pool").hash == original.hash
    assert CandidateSet.create(study, "pool", [first]).hash == original.hash

    with pytest.raises(ValueError, match="must explicitly supersedes"):
        CandidateSet.create(study, "pool", [first, second])

    with pytest.raises(ValueError, match="cannot supersede"):
        CandidateSet.create(study, "fresh", [second], supersedes=original.hash)

    replacement = CandidateSet.create(study, "pool", [first, second], supersedes=original.hash)
    assert replacement.supersedes == original.hash
    assert CandidateSet.one(study, name="pool").hash == replacement.hash
    assert CandidateSet.open(study, original.hash).members == (first.hash,)

    third = study.results.publish_predictions(
        training,
        checkpoint_kind="epoch",
        checkpoint_value=3,
        split="validation",
        predictions=frame.with_columns((pl.col("y_score") * 3).alias("y_score")),
        expected_keys=expected,
    )
    head = CandidateSet.create(study, "pool", [first, second, third], supersedes=replacement.hash)
    assert CandidateSet.one(study, name="pool").hash == head.hash


def _two_validation_predictions(study: Study):
    """Two comparable validation predictions from one training run."""
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    first = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )
    second = study.results.publish_predictions(
        training,
        checkpoint_kind="epoch",
        checkpoint_value=2,
        split="validation",
        predictions=frame.with_columns((pl.col("y_score") * 2).alias("y_score")),
        expected_keys=expected,
    )
    return first, second


def test_a_set_requested_under_a_second_name_is_bound_to_that_name(tmp_path: Path) -> None:
    """The case a union produces: a set whose members already have a name needs its own.

    `cme_futures` freezes `signal`, then freezes `pre-overlay` as the union of `signal` and
    `allocation`. An allocation stage that registered nothing new for a label makes the union
    equal to `signal`, so the two names describe the same members. Identity is the members, so
    one row is written; both names still have to resolve, and a caller that asked for
    `pre-overlay` has to be able to read it back.
    """
    study = _study(tmp_path)
    first, _ = _two_validation_predictions(study)

    signal = CandidateSet.create(study, "signal", [first])
    pre_overlay = CandidateSet.create(study, "pre-overlay", [first])

    assert pre_overlay.hash == signal.hash
    assert pre_overlay.name == "pre-overlay"

    # Resolution answers with the binding that was asked for. `open` reads by hash and has no
    # way to know which of a set's names was meant, so returning its answer unaltered handed
    # the caller a `pre-overlay` set named `signal` - which is the disagreement between object
    # and registry this whole change exists to remove, moved one method along.
    resolved = CandidateSet.one(study, name="pre-overlay")
    assert resolved.hash == signal.hash
    assert resolved.name == "pre-overlay"
    assert CandidateSet.one(study, name="signal").name == "signal"


def test_a_second_name_supersedes_nothing_and_retires_no_other_name(tmp_path: Path) -> None:
    """Binding a name is not a generation of any other name.

    Naming an existing set is not a change to the comparison it already names, so it needs no
    `supersedes` and must not be readable as one - otherwise freezing `pre-overlay` would put
    `signal` out of force and every reader of `signal` would raise.
    """
    study = _study(tmp_path)
    first, second = _two_validation_predictions(study)

    signal = CandidateSet.create(study, "signal", [first])
    CandidateSet.create(study, "pre-overlay", [first])
    assert CandidateSet.one(study, name="signal").hash == signal.hash

    # Lineage is per name: replacing `signal` leaves the set `pre-overlay` names in force.
    replacement = CandidateSet.create(study, "signal", [first, second], supersedes=signal.hash)
    live = CandidateSet.one(study, name="signal")
    assert live.hash == replacement.hash
    # The lineage the resolved binding reports is that name's, not the identity row's.
    assert live.supersedes == signal.hash
    still = CandidateSet.one(study, name="pre-overlay")
    assert still.hash == signal.hash
    assert still.supersedes is None


def test_a_changed_set_under_a_bound_second_name_still_has_to_supersede(tmp_path: Path) -> None:
    """The refusal the binding table must not weaken.

    A name that resolves to a live set is a name a changed set cannot take silently, whether or
    not that set was first written under a different name.
    """
    study = _study(tmp_path)
    first, second = _two_validation_predictions(study)

    CandidateSet.create(study, "signal", [first])
    pre_overlay = CandidateSet.create(study, "pre-overlay", [first])

    with pytest.raises(ValueError, match="must explicitly supersedes"):
        CandidateSet.create(study, "pre-overlay", [first, second])

    widened = CandidateSet.create(
        study, "pre-overlay", [first, second], supersedes=pre_overlay.hash
    )
    assert CandidateSet.one(study, name="pre-overlay").hash == widened.hash


def test_re_creating_a_bound_name_is_the_re_run_and_writes_nothing(tmp_path: Path) -> None:
    study = _study(tmp_path)
    first, second = _two_validation_predictions(study)

    original = CandidateSet.create(study, "pool", [first])
    replacement = CandidateSet.create(study, "pool", [first, second], supersedes=original.hash)

    again = CandidateSet.create(study, "pool", [first, second], supersedes=original.hash)
    assert again.hash == replacement.hash
    assert again.supersedes == original.hash
    with closing(sqlite3.connect(study.root / "run_log" / "registry.db")) as db:
        bindings = db.execute(
            "SELECT count(*) FROM candidate_set_names WHERE name = 'pool'"
        ).fetchone()[0]
    assert bindings == 2

    # Two generations, one live. The ambiguity message counts heads rather than rows, so a
    # retired generation is not reported as a second live identity.
    assert CandidateSet.one(study, name="pool").hash == replacement.hash


def test_a_registry_written_before_the_binding_table_still_resolves_its_names(
    tmp_path: Path,
) -> None:
    """The reader's clone: `candidate_sets.name` is the only binding such a registry has.

    Resolution reads the binding table where there is one. A registry written by an earlier
    version has none, and dropping it here is that registry exactly - one name per identity row,
    in the column that used to carry it.
    """
    study = _study(tmp_path)
    first, _ = _two_validation_predictions(study)
    original = CandidateSet.create(study, "pool", [first])

    with closing(sqlite3.connect(study.root / "run_log" / "registry.db")) as db:
        db.execute("DROP TABLE candidate_set_names")
        db.commit()

    assert CandidateSet.one(study, name="pool").hash == original.hash

    # Opening it for writing restores the binding from the identity row, so the name resolves
    # by the same path as one written today.
    with closing(_open_registry(study.root)) as db:
        assert db.execute(
            "SELECT set_hash FROM candidate_set_names WHERE name = 'pool'"
        ).fetchone() == (original.hash,)


def _metrics_registry(tmp_path: Path, task_types: list) -> Path:
    """A registry whose `prediction_metrics` rows carry the given `task_type` values."""
    _open_registry(tmp_path).close()
    db_path = tmp_path / "run_log" / "registry.db"
    with closing(sqlite3.connect(db_path)) as db:
        db.executemany(
            "INSERT INTO prediction_metrics (prediction_hash, task_type, computed_at) "
            "VALUES (?, ?, '2026-01-01T00:00:00Z')",
            [(f"p{index}", value) for index, value in enumerate(task_types)],
        )
        db.commit()
    return db_path


def test_a_registry_holding_legacy_numeric_task_types_is_rewritten_on_open(
    tmp_path: Path,
) -> None:
    """The conversion the migration exists for, which no test covered."""
    db_path = _metrics_registry(tmp_path, [1.0, 0.0, "classification"])

    _open_registry(tmp_path).close()

    with closing(sqlite3.connect(db_path)) as db:
        values = sorted(
            row[0] for row in db.execute("SELECT task_type FROM prediction_metrics").fetchall()
        )
    assert values == ["classification", "classification", "regression"]


def test_opening_a_migrated_registry_does_not_wait_for_the_write_lock(tmp_path: Path) -> None:
    """The pass-through, which is the one every open after the first takes.

    `_open_registry` is on every path that touches a registry, and an `UPDATE` asks for the
    write lock whether or not a row matches. With `busy_timeout` at 60s that turns one open
    contended by any other writer into a minute of waiting, for a rewrite that has had nothing
    to do since the last legacy row was converted - nine of them is a notebook's whole cell
    budget.

    Timed against a held lock rather than counted, because the count cannot see this: an
    `UPDATE` matching no row leaves `total_changes` at zero while still having asked.
    """
    db_path = _metrics_registry(tmp_path, ["classification", "regression"])

    with closing(sqlite3.connect(db_path, timeout=1.0)) as holder:
        holder.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        db = _open_registry(tmp_path)
        elapsed = time.monotonic() - started
        db.close()
        holder.rollback()

    # A request for the lock would block on `busy_timeout`, which `_open_registry` sets to 60s.
    assert elapsed < 5.0, f"opening a migrated registry waited {elapsed:.1f}s for a write lock"


def test_partial_and_preview_results_are_rejected_from_canonical_sets(tmp_path: Path) -> None:
    study = _study(tmp_path)
    canonical_training = study.results.register_training(_training_spec())
    frame = _predictions()
    partial = study.results.publish_predictions(
        canonical_training,
        checkpoint_kind="epoch",
        checkpoint_value=1,
        split="validation",
        predictions=frame.head(1),
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
        allow_partial=True,
    )
    with pytest.raises(ValueError, match="partial"):
        CandidateSet.create(study, "partial", [partial])
    with pytest.raises(ValueError, match="partial"):
        study.strategy(
            prediction=partial,
            signal={"method": "equal_weight_top_k", "top_k": 1},
            execution_mode="vectorized",
        )

    preview_training = study.results.register_training(
        _training_spec(execution_tier="preview", preview_reductions={"folds": [0], "max_rows": 2}),
        execution_tier="preview",
    )
    preview = study.results.publish_predictions(
        preview_training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )

    with pytest.raises(ValueError, match="preview"):
        CandidateSet.create(study, "canonical", [preview])
    with pytest.raises(KeyError):
        Result.open(study, preview.hash)
    assert Result.open(study, preview.hash, include_preview=True).hash == preview.hash


def test_fitted_states_come_back_in_fold_order_not_filename_order(tmp_path: Path) -> None:
    """A lexicographic sort puts fold_10 before fold_2, and one case study declares 16 splits."""
    import joblib

    from case_studies.research import TrainingResult

    study = _study(tmp_path)
    db = _open_registry(study.root)
    try:
        db.execute(
            "INSERT INTO training_runs "
            "(training_hash, family, label, spec_json, created_at) VALUES (?,?,?,?,?)",
            ("many-folds", "linear", "fwd_ret_21d", "{}", "2024-01-01"),
        )
        db.commit()
    finally:
        db.close()

    models = study.root / "run_log" / "training" / "many-folds" / "models"
    models.mkdir(parents=True)
    for fold in range(12):
        joblib.dump({"fold": fold}, models / f"fold_{fold}.joblib")

    reopened = Result.open(study, "many-folds")

    assert isinstance(reopened, TrainingResult)
    assert [state["fold"] for state in reopened.fitted_states()] == list(range(12))


def test_fold_metrics_come_back_in_fold_order_with_the_values_registered(tmp_path: Path) -> None:
    """`ic_mean` is the equal-weight mean of these rows, so a notebook arguing from their spread -
    how often the sign changes, how far the cross-section narrows - is reading them rather than the
    average. That argument is only checkable if the rows come back complete and in fold order: a
    lexicographic sort would put fold 10 before fold 2 and silently reorder the sequence the prose
    walks through."""
    study = _study(tmp_path)
    registered = [
        (0, -0.076172, 0.330584, 92.0),
        (2, -0.003361, 0.175347, 88.0),
        (10, 0.115767, 0.302743, 86.0),
    ]
    db = _open_registry(study.root)
    try:
        db.execute(
            "INSERT INTO training_runs "
            "(training_hash, family, label, spec_json, created_at) VALUES (?,?,?,?,?)",
            ("folded-training", "latent_factors", "fwd_ret_21d", "{}", "2024-01-01"),
        )
        db.execute(
            "INSERT INTO prediction_sets "
            "(prediction_hash, training_hash, split, created_at) VALUES (?,?,?,?)",
            ("folded-prediction", "folded-training", "validation", "2024-01-01"),
        )
        for fold_id, ic, ic_std, n_entities in reversed(registered):
            db.execute(
                "INSERT INTO fold_metrics "
                "(prediction_hash, fold_id, computed_at, ic, ic_std, n_entities) "
                "VALUES (?,?,?,?,?,?)",
                ("folded-prediction", fold_id, "2024-01-01", ic, ic_std, n_entities),
            )
        db.commit()
    finally:
        db.close()

    folds = Result.open(study, "folded-prediction").folds()

    assert folds.columns == ["fold_id", "ic", "ic_std", "n_entities"]
    assert [tuple(row) for row in folds.iter_rows()] == registered


def test_a_prediction_set_with_no_registered_folds_returns_an_empty_frame(tmp_path: Path) -> None:
    """Reading the folds of a result that has none must not raise. A legacy row carries no fold
    metrics at all, and a caller that has to guard the call before making it will not make it."""
    study = _study(tmp_path)
    db = _open_registry(study.root)
    try:
        db.execute(
            "INSERT INTO training_runs "
            "(training_hash, family, label, spec_json, created_at) VALUES (?,?,?,?,?)",
            ("foldless-training", "linear", "fwd_ret_21d", "{}", "2024-01-01"),
        )
        db.execute(
            "INSERT INTO prediction_sets "
            "(prediction_hash, training_hash, split, created_at) VALUES (?,?,?,?)",
            ("foldless-prediction", "foldless-training", "validation", "2024-01-01"),
        )
        db.commit()
    finally:
        db.close()

    folds = Result.open(study, "foldless-prediction").folds()

    assert folds.height == 0
    assert folds.columns == ["fold_id", "ic", "ic_std", "n_entities"]


def test_the_refusal_names_the_partial_member_and_the_sense_it_is_partial_in(
    tmp_path: Path,
) -> None:
    """`partial results cannot enter a candidate set` used to name neither."""
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    partial = study.results.publish_predictions(
        training,
        checkpoint_kind="epoch",
        checkpoint_value=1,
        split="validation",
        predictions=frame.head(1),
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
        allow_partial=True,
    )
    with pytest.raises(ValueError) as raised:
        CandidateSet.create(study, "partial", [partial])
    message = str(raised.value)
    assert partial.hash in message, "the refusal must say which member"
    assert "coverage" in message or "fold_metrics" in message, (
        f"the refusal must say in which sense, got: {message}"
    )


def test_the_catalog_column_is_necessary_and_not_sufficient(tmp_path: Path) -> None:
    """The registry column can say complete where the on-disk check says otherwise.

    This is the disagreement that made a notebook's own guard pass at one cell and the
    freeze refuse the same rows at another. The column reads the registry only; deleting
    the artifact leaves every registry row it inspects untouched.
    """
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )
    assert prediction.complete
    assert study.results.partial_members([prediction.hash]) == []

    artifact = (
        study.storage_root("canonical")
        / "run_log"
        / "predictions"
        / prediction.hash
        / "predictions.parquet"
    )
    assert artifact.is_file()
    artifact.unlink()

    reopened = Result.open(study, prediction.hash)
    reason = reopened.completeness()
    assert reason is not None, "an absent artifact is not a complete result"
    assert str(artifact) in reason, f"the reason must name the file, got: {reason}"

    partial = study.results.partial_members([prediction.hash])
    assert [member_hash for member_hash, _ in partial] == [prediction.hash]

    with pytest.raises(ValueError) as raised:
        CandidateSet.create(study, "gone", [reopened])
    assert prediction.hash in str(raised.value)
