from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import (
    EligibilityManifest,
    OfficialPopulation,
    Result,
    Study,
    run_backtests,
    run_models,
)
from case_studies.utils import linear
from case_studies.utils.registry import metrics as registry_metrics
from case_studies.utils.registry import prediction_hash_from_parts
from tests.test_research_contract_catalog import _publish, _resolved_spec, _tree_digest
from tests.test_research_flow import _prices
from tests.test_research_models import _linear_study
from tests.test_research_registry import _predictions, _training_spec
from tests.test_research_workspace import _seed_release


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _study(tmp_path: Path) -> Study:
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )


def _publish_prediction(study: Study, *, alpha: float, checkpoint: int) -> str:
    training = study.results.register_training(_resolved_spec(alpha=alpha))
    frame = _predictions().with_columns((pl.col("y_score") * alpha).alias("y_score"))
    return study.results.publish_predictions(
        training,
        checkpoint_kind="epoch",
        checkpoint_value=checkpoint,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    ).hash


def _backtest_count(study: Study) -> int:
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        return db.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]


def test_run_models_returns_catalog_rows_and_diagnostics(tmp_path: Path, monkeypatch) -> None:
    study = _linear_study(tmp_path, monkeypatch)
    request = study.model(family="linear", label="fwd_ret_1d", config_name="ridge")

    execution = run_models(study, requests=[request])

    assert execution.catalog_rows.height == 1
    assert (
        execution.catalog_rows.item(0, "prediction_hash") == execution.runs[0].predictions[0].hash
    )
    assert execution.catalog_rows.item(0, "identity_status") == "current"
    assert execution.catalog_rows.item(0, "complete") is True
    assert execution.diagnostics[0]["status"] == "completed"
    assert execution.diagnostics[0]["fitted_folds"] == [0, 1]


def test_backtest_selection_validation_fails_before_any_write(tmp_path: Path) -> None:
    study = _study(tmp_path)
    prediction_hash = _publish_prediction(study, alpha=1.0, checkpoint=1)
    selected = study.predictions.table().filter(pl.col("prediction_hash") == prediction_hash)

    with pytest.raises(ValueError, match="required identity columns"):
        run_backtests(
            study,
            predictions=selected.drop("training_hash"),
            signal={"method": "equal_weight_top_k", "top_k": 1},
            prices=_prices(),
        )
    with pytest.raises(ValueError, match="duplicate|ambiguous"):
        run_backtests(
            study,
            predictions=pl.concat([selected, selected]),
            signal={"method": "equal_weight_top_k", "top_k": 1},
            prices=_prices(),
        )

    assert _backtest_count(study) == 0


def test_prediction_publication_rejects_eligibility_manifest_mismatch(tmp_path: Path) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    manifest = EligibilityManifest.resolve(
        frame.select(pl.col("symbol"), pl.col("timestamp"), pl.col("fold_id").alias("fold")),
        source_identity={"labels": "labels-a", "features": "features-a"},
        logic_identity={"implementation": "eligibility-a"},
    )

    with pytest.raises(ValueError, match="coverage is partial"):
        study.results.publish_predictions(
            training,
            checkpoint_kind="final",
            checkpoint_value=None,
            split="validation",
            predictions=frame.head(1),
            expected_keys=manifest,
        )

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM prediction_sets").fetchone() == (0,)


def test_catalog_one_reports_fields_that_disambiguate_matches(tmp_path: Path) -> None:
    study = _study(tmp_path)
    _publish_prediction(study, alpha=1.0, checkpoint=1)
    _publish_prediction(study, alpha=2.0, checkpoint=2)

    with pytest.raises(ValueError, match="checkpoint_value.*training_hash"):
        study.predictions.one(config_name="ridge")


def test_n_catalog_rows_fan_out_to_n_independent_backtests(tmp_path: Path) -> None:
    study = _study(tmp_path)
    first = _publish_prediction(study, alpha=1.0, checkpoint=1)
    second = _publish_prediction(study, alpha=2.0, checkpoint=2)
    selected = study.predictions.table().filter(pl.col("prediction_hash").is_in([first, second]))

    execution = run_backtests(
        study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        prices=_prices(),
    )

    assert len(execution.results) == 2
    assert len({result.hash for result in execution.results}) == 2
    assert {item["prediction_hash"] for item in execution.diagnostics} == {first, second}
    assert _backtest_count(study) == 2


def test_released_catalog_prediction_backtests_into_workspace_only(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    release_case = release / "case_studies" / "etfs"
    prediction_hash = _publish(release_case, spec=_resolved_spec())
    before = _tree_digest(release_case)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    selected = study.predictions.table().filter(pl.col("prediction_hash") == prediction_hash)

    execution = run_backtests(
        study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        prices=_prices(),
    )
    reopened = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)

    assert len(execution.results) == 1
    assert _backtest_count(reopened) == 1
    assert reopened.predictions.table().item(0, "origin") == "released"
    assert _tree_digest(release_case) == before


def test_official_population_cannot_silently_omit_failed_member(tmp_path: Path) -> None:
    study = _study(tmp_path)
    complete = _publish_prediction(study, alpha=1.0, checkpoint=1)
    missing = "missing-prediction"
    snapshot = OfficialPopulation.create(
        study,
        name="linear-baseline",
        member_kind="prediction",
        members=[complete, missing],
    )

    with pytest.raises(ValueError, match="missing-prediction"):
        snapshot.require_complete()
    with pytest.raises(ValueError, match="supersedes"):
        OfficialPopulation.create(
            study,
            name="linear-baseline",
            member_kind="prediction",
            members=[complete],
        )

    replacement = OfficialPopulation.create(
        study,
        name="linear-baseline",
        member_kind="prediction",
        members=[complete],
        supersedes=snapshot.hash,
    )
    assert replacement.require_complete() == (complete,)
    assert OfficialPopulation.open(study, snapshot.hash).members == (complete, missing)


def test_interrupted_linear_run_reuses_completed_fold_on_retry(tmp_path: Path, monkeypatch) -> None:
    study = _linear_study(tmp_path, monkeypatch)
    request = study.model(family="linear", label="fwd_ret_1d", config_name="ridge")
    original_fit = linear.Ridge.fit
    calls = 0

    def interrupt_second_fit(self, x, y, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted second fold")
        return original_fit(self, x, y, **kwargs)

    monkeypatch.setattr(linear.Ridge, "fit", interrupt_second_fit)
    with pytest.raises(RuntimeError, match="interrupted second fold"):
        run_models(study, requests=[request])

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM candidate_fold_completions").fetchone() == (1,)
        assert db.execute(
            "SELECT status FROM execution_attempts ORDER BY started_at"
        ).fetchall() == [("failed",)]

    execution = run_models(study, requests=[request])

    assert calls == 3
    assert execution.diagnostics[0]["reused_folds"] == [0]
    assert execution.diagnostics[0]["fitted_folds"] == [1]
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM candidate_fold_completions").fetchone() == (2,)
        assert db.execute(
            "SELECT status FROM execution_attempts ORDER BY started_at"
        ).fetchall() == [("failed",), ("completed",)]


def test_prediction_retry_finishes_metrics_without_new_identity(
    tmp_path: Path, monkeypatch
) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    original_compute = registry_metrics.compute_prediction_fold_metrics

    def interrupt_metrics(*args, **kwargs):
        raise RuntimeError("interrupted metric finalization")

    monkeypatch.setattr(registry_metrics, "compute_prediction_fold_metrics", interrupt_metrics)
    with pytest.raises(RuntimeError, match="interrupted metric finalization"):
        study.results.publish_predictions(
            training,
            checkpoint_kind="final",
            checkpoint_value=None,
            split="validation",
            predictions=frame,
            expected_keys=expected,
        )
    prediction_hash = prediction_hash_from_parts(
        training.hash,
        None,
        "validation",
        checkpoint_kind="final",
        identity_version=2,
    )
    interrupted = Result.open(study, prediction_hash)
    assert not interrupted.complete

    monkeypatch.setattr(registry_metrics, "compute_prediction_fold_metrics", original_compute)
    finalized = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )

    assert finalized.hash == prediction_hash
    assert finalized.complete
