from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import (
    CandidateSet,
    DecisionArtifact,
    EligibilityManifest,
    OfficialPopulation,
    PredictionResult,
    ResolvedModelRequest,
    Result,
    StateTransitionPolicy,
    Study,
    plan_backtests,
    run_backtests,
    run_models,
    snapshot_official_models,
)
from case_studies.utils import linear
from case_studies.utils.registry import metrics as registry_metrics
from case_studies.utils.registry import (
    prediction_hash_from_parts,
    register_backtest_fold_metrics,
    register_backtest_run,
)
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


def test_individual_and_batch_linear_execution_are_equivalent(tmp_path: Path, monkeypatch) -> None:
    individual_study = _linear_study(tmp_path / "individual", monkeypatch)
    individual_runs = []
    for alpha in (1.0, 2.0):
        request = individual_study.model(
            family="linear",
            label="fwd_ret_1d",
            config_name="ridge",
            overrides={"alpha": alpha},
        )
        individual_runs.append(run_models(individual_study, requests=[request]).runs[0])

    batch_study = _linear_study(tmp_path / "batch", monkeypatch)
    batch_requests = [
        batch_study.model(
            family="linear",
            label="fwd_ret_1d",
            config_name="ridge",
            overrides={"alpha": alpha},
        )
        for alpha in (1.0, 2.0)
    ]
    batch_runs = run_models(batch_study, requests=batch_requests).runs

    assert [run.training.hash for run in individual_runs] == [
        run.training.hash for run in batch_runs
    ]
    for individual, batch in zip(individual_runs, batch_runs, strict=True):
        assert individual.predictions[0].hash == batch.predictions[0].hash
        assert individual.predictions[0].load().equals(batch.predictions[0].load())


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


def test_prediction_catalog_freezes_only_authoritative_polars_rows(tmp_path: Path) -> None:
    study = _study(tmp_path)
    first = _publish_prediction(study, alpha=1.0, checkpoint=1)
    second = _publish_prediction(study, alpha=2.0, checkpoint=2)
    selected = study.predictions.table().filter(pl.col("prediction_hash").is_in([first, second]))

    frozen = study.predictions.freeze(selected, name="visible-model-selection")

    assert frozen.member_kind == "prediction"
    assert frozen.members == tuple(sorted((first, second)))
    assert CandidateSet.one(study, name="visible-model-selection").hash == frozen.hash
    altered = selected.with_columns(pl.lit("altered-training").alias("training_hash"))
    with pytest.raises(ValueError, match="altered lineage.*training_hash"):
        study.predictions.freeze(altered, name="altered")
    with pytest.raises(ValueError, match="duplicate.*ambiguous"):
        study.predictions.freeze(pl.concat([selected, selected]), name="duplicate")

    training = study.results.register_training(_resolved_spec(alpha=3.0))
    frame = _predictions()
    partial = study.results.publish_predictions(
        training,
        checkpoint_kind="epoch",
        checkpoint_value=3,
        split="validation",
        predictions=frame.head(1),
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
        allow_partial=True,
    )
    partial_selection = study.predictions.table().filter(pl.col("prediction_hash") == partial.hash)
    with pytest.raises(ValueError, match="partial"):
        study.predictions.freeze(partial_selection, name="partial")


def test_version_3_candidate_protocol_uses_computation_cv_identity(tmp_path: Path) -> None:
    study = _study(tmp_path)
    first_hash = _publish_prediction(study, alpha=1.0, checkpoint=1)
    changed = _resolved_spec(alpha=2.0)
    changed["computation"]["cv"]["identity"] = "cv-b"
    second_hash = _publish(study.root, spec=changed, score_shift=0.1)
    first = Result.open(study, first_hash)
    second = Result.open(study, second_hash)

    with pytest.raises(ValueError, match="protocol-incompatible.*cv"):
        CandidateSet.create(study, "incompatible-cv", [first, second])


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
    assert execution.population is not None
    assert execution.population.require_complete() == tuple(
        sorted(result.hash for result in execution.results)
    )
    assert "backtest_hash" in execution.catalog_rows.columns
    assert execution.catalog_rows.height == 2
    assert set(execution.catalog_rows["prediction_hash"]) == {first, second}
    candidate_set = study.backtests.freeze(
        execution.catalog_rows,
        name="two-backtests",
    )
    with pytest.raises(ValueError, match="disambiguate.*checkpoint_value"):
        study.backtests.one(config_name="ridge")
    with pytest.raises(ValueError, match="required identity columns"):
        study.backtests.freeze(
            execution.catalog_rows.drop("training_hash"),
            name="missing-backtest-lineage",
        )
    assert CandidateSet.one(study, name="two-backtests").hash == candidate_set.hash
    assert len(candidate_set.ranked_validation_sharpe()) == 2
    assert len(candidate_set.ranked_validation_sharpe(limit=1)) == 1


def test_backtest_catalog_is_typed_and_filters_nested_model_and_strategy_fields(
    tmp_path: Path,
) -> None:
    study = _study(tmp_path)
    prediction_hash = _publish_prediction(study, alpha=1.0, checkpoint=1)
    selected = study.predictions.table().filter(pl.col("prediction_hash") == prediction_hash)

    execution = run_backtests(
        study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        prices=_prices(),
        allocation={"method": "equal_weight"},
    )
    catalog = study.backtests.table()
    filtered = catalog.filter(
        (pl.col("model__params__alpha") == 1.0)
        & (pl.col("strategy__signal__top_k") == 1)
        & (pl.col("strategy__allocation__method") == "equal_weight")
    )

    assert filtered.height == 1
    assert filtered.item(0, "backtest_hash") == execution.results[0].hash
    assert filtered.item(0, "completion_state") == "complete"
    assert filtered.item(0, "complete") is True
    assert filtered.schema["model__params__alpha"] == pl.Float64
    assert filtered.schema["strategy__signal__top_k"] == pl.Int64
    assert filtered.schema["sharpe"] == pl.Float64
    assert filtered.item(0, "spec_json") == execution.results[0].registry_record()["spec_json"]
    altered = filtered.with_columns(pl.lit("wrong-prediction").alias("prediction_hash"))
    with pytest.raises(ValueError, match="altered lineage.*prediction_hash"):
        study.backtests.freeze(altered, name="altered-backtest")
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "DELETE FROM backtest_metrics WHERE backtest_hash = ?",
            (execution.results[0].hash,),
        )
        db.commit()
    partial = study.backtests.table().filter(pl.col("backtest_hash") == execution.results[0].hash)
    with pytest.raises(ValueError, match="partial"):
        study.backtests.freeze(partial, name="partial-backtest")


def test_backtest_planning_resolves_every_identity_without_writes(tmp_path: Path) -> None:
    study = _study(tmp_path)
    first = _publish_prediction(study, alpha=1.0, checkpoint=1)
    second = _publish_prediction(study, alpha=2.0, checkpoint=2)
    selected = study.predictions.table().filter(pl.col("prediction_hash").is_in([first, second]))

    plan = plan_backtests(
        study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        prices=_prices(),
    )

    assert len(plan.members) == 2
    assert len(set(plan.expected_hashes)) == 2
    assert {member.prediction_hash for member in plan.members} == {first, second}
    assert _backtest_count(study) == 0


def test_explicit_backtest_bridge_preserves_strategy_identity_and_returns(tmp_path: Path) -> None:
    direct_study = _study(tmp_path / "direct")
    direct_prediction_hash = _publish_prediction(
        direct_study,
        alpha=1.0,
        checkpoint=1,
    )
    direct_prediction = Result.open(direct_study, direct_prediction_hash)
    assert isinstance(direct_prediction, PredictionResult)
    direct_strategy = direct_study.strategy(
        prediction=direct_prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
    )
    expected_identity = direct_strategy.identity(prices=_prices())
    direct_result = direct_strategy.run(prices=_prices())

    explicit_study = _study(tmp_path / "explicit")
    explicit_prediction_hash = _publish_prediction(
        explicit_study,
        alpha=1.0,
        checkpoint=1,
    )
    selected = explicit_study.predictions.table().filter(
        pl.col("prediction_hash") == explicit_prediction_hash
    )
    plan = plan_backtests(
        explicit_study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        prices=_prices(),
    )
    explicit_result = run_backtests(
        explicit_study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        prices=_prices(),
    ).results[0]

    assert plan.expected_hashes == (expected_identity,)
    assert direct_result.hash == explicit_result.hash == expected_identity
    direct_returns = (
        direct_result.root / "run_log" / "backtest" / direct_result.hash / "daily_returns.parquet"
    )
    explicit_returns = (
        explicit_result.root
        / "run_log"
        / "backtest"
        / explicit_result.hash
        / "daily_returns.parquet"
    )
    assert pl.read_parquet(direct_returns).equals(pl.read_parquet(explicit_returns))


def test_backtest_population_precedes_writes_and_retry_reuses_completed_results(
    tmp_path: Path, monkeypatch
) -> None:
    from case_studies.research.strategy import Strategy

    study = _study(tmp_path)
    first = _publish_prediction(study, alpha=1.0, checkpoint=1)
    second = _publish_prediction(study, alpha=2.0, checkpoint=2)
    selected = study.predictions.table().filter(pl.col("prediction_hash").is_in([first, second]))
    planned = plan_backtests(
        study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        prices=_prices(),
    )
    original_run = Strategy.run
    calls = 0
    observed_population_hash = None

    def fail_second(self, *, prices=None):
        nonlocal calls, observed_population_hash
        calls += 1
        population = OfficialPopulation.one(study, name="planned-backtests")
        observed_population_hash = population.hash
        assert population.members == tuple(sorted(planned.expected_hashes))
        if calls == 1:
            assert _backtest_count(study) == 0
            return original_run(self, prices=prices)
        raise RuntimeError("induced second backtest failure")

    monkeypatch.setattr(Strategy, "run", fail_second)
    with pytest.raises(RuntimeError, match="induced second backtest failure"):
        run_backtests(
            study,
            predictions=selected,
            signal={"method": "equal_weight_top_k", "top_k": 1},
            prices=_prices(),
            population_name="planned-backtests",
        )

    population = OfficialPopulation.open(study, str(observed_population_hash))
    with pytest.raises(ValueError, match="missing"):
        population.require_complete()
    assert _backtest_count(study) == 1

    monkeypatch.setattr(Strategy, "run", original_run)
    retried = run_backtests(
        study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        prices=_prices(),
        population_name="planned-backtests",
    )

    assert retried.population_hash == observed_population_hash
    assert [item["status"] for item in retried.diagnostics] == ["reused", "completed"]
    assert retried.population is not None
    assert retried.population.require_complete() == tuple(sorted(planned.expected_hashes))
    assert [result.hash for result in retried.results] == list(planned.expected_hashes)

    interrupted_hash = retried.results[0].hash
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "DELETE FROM backtest_metrics WHERE backtest_hash = ?",
            (interrupted_hash,),
        )
        db.commit()
    with pytest.raises(ValueError, match="partial"):
        retried.population.require_complete()

    repaired = run_backtests(
        study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        prices=_prices(),
        population_name="planned-backtests",
    )

    assert [item["status"] for item in repaired.diagnostics] == ["completed", "reused"]
    assert repaired.population is not None
    assert repaired.population.require_complete() == tuple(sorted(planned.expected_hashes))


def test_strategy_change_creates_a_backtest_without_retraining(tmp_path: Path) -> None:
    study = _study(tmp_path)
    prediction_hash = _publish_prediction(study, alpha=1.0, checkpoint=1)
    selected = study.predictions.table().filter(pl.col("prediction_hash") == prediction_hash)
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        before = {
            table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("training_runs", "prediction_sets")
        }

    first = run_backtests(
        study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        prices=_prices(),
    )
    second = run_backtests(
        study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 2},
        prices=_prices(),
    )

    assert first.results[0].hash != second.results[0].hash
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        after = {
            table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("training_runs", "prediction_sets")
        }
    assert after == before


def test_custom_stateful_decision_backtests_but_requires_promotion_for_candidates(
    tmp_path: Path,
) -> None:
    study = _study(tmp_path)
    prediction_hash = _publish_prediction(study, alpha=1.0, checkpoint=1)
    selected = study.predictions.table().filter(pl.col("prediction_hash") == prediction_hash)
    decisions = (
        _predictions()
        .select("symbol", "timestamp")
        .with_columns(pl.Series("position", [1.0, -1.0]))
    )
    artifact = DecisionArtifact.publish(
        study,
        kind="target_positions",
        decisions=decisions,
        prediction_hashes=[prediction_hash],
        parameters={"generator": "ordinary_python", "cadence": "1d"},
        state_transition_policy=StateTransitionPolicy(
            fold_boundary="liquidate",
            temporal_gap="reset",
        ),
    )

    execution = run_backtests(
        study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        prices=_prices(),
        execution_mode="vectorized",
        decision=artifact,
    )
    result = execution.results[0]

    assert execution.population is None
    assert result.spec()["decision_artifact"]["hash"] == artifact.hash
    assert result.spec()["decision_artifact"]["state_transition_policy"] == {
        "fold_boundary": "liquidate",
        "temporal_gap": "reset",
    }
    with pytest.raises(ValueError, match="exploratory decision"):
        CandidateSet.create(study, "exploratory-decision", [result])
    with pytest.raises(ValueError, match="exploratory decision"):
        OfficialPopulation.create(
            study,
            name="exploratory-decision",
            member_kind="backtest",
            members=[result.hash],
        )
    with pytest.raises(ValueError, match="exploratory.*official population"):
        run_backtests(
            study,
            predictions=selected,
            signal={"method": "equal_weight_top_k", "top_k": 1},
            prices=_prices(),
            execution_mode="vectorized",
            decision=artifact,
            population_name="exploratory-must-not-be-official",
        )


def test_preview_prediction_is_excluded_from_official_population(
    tmp_path: Path, monkeypatch
) -> None:
    study = _linear_study(tmp_path, monkeypatch)
    preview = (
        run_models(
            study,
            requests=[
                study.model(
                    family="linear",
                    label="fwd_ret_1d",
                    config_name="ridge",
                    execution_tier="preview",
                    preview_reductions={"folds": [0]},
                )
            ],
        )
        .runs[0]
        .predictions[0]
    )

    assert study.predictions.table().is_empty()
    preview_selection = study.predictions.table(include_preview=True)
    assert preview_selection.height == 1
    with pytest.raises(ValueError, match="preview.*candidate set"):
        study.predictions.freeze(preview_selection, name="preview-must-not-freeze")
    with pytest.raises(ValueError, match="preview.*cannot enter"):
        OfficialPopulation.create(
            study,
            name="preview-must-not-enter-official",
            member_kind="prediction",
            members=[preview.hash],
        )
    preview_returns = pl.DataFrame({"timestamp": ["2024-01-05"], "return": [0.01]}).with_columns(
        pl.col("timestamp").str.to_date()
    )
    preview_backtest_hash = register_backtest_run(
        "etfs",
        preview.hash,
        {
            "identity_version": 2,
            "execution_tier": "preview",
            "strategy": {"signal": {"method": "equal_weight_top_k", "top_k": 1}},
        },
        stage="signal",
        returns=preview_returns,
        metrics={"sharpe": 1.0},
        case_dir=preview.root,
    )
    preview_backtest_rows = study.backtests.table(include_preview=True).filter(
        pl.col("backtest_hash") == preview_backtest_hash
    )
    assert preview_backtest_rows.item(0, "complete") is True
    with pytest.raises(ValueError, match="preview.*candidate set"):
        study.backtests.freeze(
            preview_backtest_rows,
            name="preview-backtest-must-not-freeze",
        )
    with pytest.raises(ValueError, match="preview.*official population"):
        run_backtests(
            study,
            predictions=preview_selection,
            signal={"method": "equal_weight_top_k", "top_k": 1},
            prices=_prices(),
            execution_mode="vectorized",
            population_name="preview-not-official",
        )
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM official_populations").fetchone() == (0,)


def test_backtest_immutability_uses_the_same_semantic_projection_as_hashing(
    tmp_path: Path,
) -> None:
    study = _study(tmp_path)
    prediction_hash = _publish_prediction(study, alpha=1.0, checkpoint=1)
    returns = pl.DataFrame({"timestamp": ["2024-01-05"], "return": [0.01]}).with_columns(
        pl.col("timestamp").str.to_date()
    )
    first = {
        "identity_version": 3,
        "execution_tier": "canonical",
        "strategy": {
            "signal": {
                "method": "equal_weight_top_k",
                "top_k": 1,
                "direction": "long_only",
            }
        },
        "backtest_config": {"metadata": {"preset_path": "/first/preset.yaml"}},
    }
    equivalent = {
        "identity_version": 3,
        "execution_tier": "canonical",
        "strategy": {"signal": {"method": "equal_weight_top_k", "top_k": 1}},
        "backtest_config": {"metadata": {"preset_path": "/other/preset.yaml"}},
    }

    first_hash = register_backtest_run(
        "etfs",
        prediction_hash,
        first,
        stage="signal",
        returns=returns,
        metrics={"sharpe": 1.0},
        case_dir=study.root,
    )
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        first_spec_json = db.execute(
            "SELECT spec_json FROM backtest_runs WHERE backtest_hash = ?",
            (first_hash,),
        ).fetchone()[0]
        db.execute("DELETE FROM backtest_metrics WHERE backtest_hash = ?", (first_hash,))
        db.commit()
    second_hash = register_backtest_run(
        "etfs",
        prediction_hash,
        equivalent,
        stage="signal",
        returns=returns,
        metrics={"sharpe": 1.0},
        case_dir=study.root,
    )

    assert first_hash == second_hash
    assert Result.open(study, second_hash).complete
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert (
            db.execute(
                "SELECT spec_json FROM backtest_runs WHERE backtest_hash = ?",
                (second_hash,),
            ).fetchone()[0]
            == first_spec_json
        )


@pytest.mark.parametrize("identity_version", [3, None], ids=["versioned", "unversioned"])
def test_backtest_retry_rejects_changed_existing_execution_artifacts(
    tmp_path: Path, identity_version: int | None
) -> None:
    study = _study(tmp_path)
    prediction_hash = _publish_prediction(study, alpha=1.0, checkpoint=1)
    returns = pl.DataFrame({"timestamp": ["2024-01-05"], "return": [0.01]}).with_columns(
        pl.col("timestamp").str.to_date()
    )
    original_trades = pl.DataFrame({"symbol": ["SPY"], "pnl": [1.0]})
    changed_trades = pl.DataFrame({"symbol": ["SPY"], "pnl": [2.0]})
    strategy: dict = {
        "execution_tier": "canonical",
        "strategy": {"signal": {"method": "equal_weight_top_k", "top_k": 1}},
    }
    if identity_version is not None:
        strategy["identity_version"] = identity_version

    backtest_hash = register_backtest_run(
        "etfs",
        prediction_hash,
        strategy,
        stage="signal",
        returns=returns,
        trades=original_trades,
        metrics={"sharpe": 1.0},
        case_dir=study.root,
    )
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute("DELETE FROM backtest_metrics WHERE backtest_hash = ?", (backtest_hash,))
        db.commit()

    with pytest.raises(ValueError, match="immutable backtest artifact conflict"):
        register_backtest_run(
            "etfs",
            prediction_hash,
            strategy,
            stage="signal",
            returns=returns,
            trades=changed_trades,
            metrics={"sharpe": 1.0},
            case_dir=study.root,
        )

    stored_trades = pl.read_parquet(
        study.root / "run_log" / "backtest" / backtest_hash / "trades.parquet"
    )
    assert stored_trades.equals(original_trades)


@pytest.mark.parametrize("existing_sharpe", [1.0, None], ids=["stale", "partial"])
def test_unversioned_backtest_retry_updates_metrics_without_rewriting_artifacts(
    tmp_path: Path, existing_sharpe: float | None
) -> None:
    study = _study(tmp_path)
    prediction_hash = _publish_prediction(study, alpha=1.0, checkpoint=1)
    returns = pl.DataFrame({"timestamp": ["2024-01-05"], "return": [0.01]}).with_columns(
        pl.col("timestamp").str.to_date()
    )
    trades = pl.DataFrame({"symbol": ["SPY"], "pnl": [1.0]})
    strategy = {
        "execution_tier": "canonical",
        "strategy": {"signal": {"method": "equal_weight_top_k", "top_k": 1}},
    }
    backtest_hash = register_backtest_run(
        "etfs",
        prediction_hash,
        strategy,
        stage="signal",
        returns=returns,
        trades=trades,
        metrics={"sharpe": 1.0},
        case_dir=study.root,
    )
    if existing_sharpe is None:
        with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
            db.execute(
                "UPDATE backtest_metrics SET sharpe = NULL WHERE backtest_hash = ?",
                (backtest_hash,),
            )
            db.commit()

    retried_hash = register_backtest_run(
        "etfs",
        prediction_hash,
        strategy,
        stage="signal",
        returns=returns,
        trades=trades,
        metrics={"sharpe": 2.0},
        case_dir=study.root,
    )

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        stored_sharpe = db.execute(
            "SELECT sharpe FROM backtest_metrics WHERE backtest_hash = ?",
            (backtest_hash,),
        ).fetchone()[0]
    assert retried_hash == backtest_hash
    assert stored_sharpe == 2.0
    assert pl.read_parquet(
        study.root / "run_log" / "backtest" / backtest_hash / "trades.parquet"
    ).equals(trades)


def test_unversioned_metric_retry_preserves_unsupplied_and_fold_metrics(tmp_path: Path) -> None:
    study = _study(tmp_path)
    prediction_hash = _publish_prediction(study, alpha=1.0, checkpoint=1)
    returns = pl.DataFrame({"timestamp": ["2024-01-05"], "return": [0.01]}).with_columns(
        pl.col("timestamp").str.to_date()
    )
    strategy = {
        "execution_tier": "canonical",
        "strategy": {"signal": {"method": "equal_weight_top_k", "top_k": 1}},
    }
    backtest_hash = register_backtest_run(
        "etfs",
        prediction_hash,
        strategy,
        stage="signal",
        returns=returns,
        metrics={"sharpe": 1.0, "max_drawdown": -0.1},
        case_dir=study.root,
    )
    register_backtest_fold_metrics(
        "etfs",
        backtest_hash,
        {0: {"sharpe": 0.5, "max_drawdown": -0.2}},
        case_dir=study.root,
    )

    register_backtest_run(
        "etfs",
        prediction_hash,
        strategy,
        stage="signal",
        returns=returns,
        metrics={"sharpe": 2.0},
        case_dir=study.root,
    )

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        headline = db.execute(
            "SELECT sharpe, max_drawdown FROM backtest_metrics WHERE backtest_hash = ?",
            (backtest_hash,),
        ).fetchone()
        fold = db.execute(
            "SELECT sharpe, max_drawdown FROM backtest_fold_metrics "
            "WHERE backtest_hash = ? AND fold_id = 0",
            (backtest_hash,),
        ).fetchone()
    assert headline == (2.0, -0.1)
    assert fold == (0.5, -0.2)


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
    assert execution.catalog_rows.item(0, "complete") is True
    assert execution.catalog_rows.item(0, "origin") == "workspace"
    assert _backtest_count(reopened) == 1
    assert reopened.backtests.table().item(0, "complete") is True
    assert reopened.predictions.table().item(0, "origin") == "released"
    assert _tree_digest(release_case) == before

    alternate_release = _seed_release(tmp_path / "alternate", marker="alternate")
    relocated = Study.open("etfs", workspace=tmp_path / "workspace", release_root=alternate_release)
    relocated_prediction = Result.open(relocated, prediction_hash)
    assert isinstance(relocated_prediction, PredictionResult)
    assert relocated_prediction.root == release_case
    assert relocated_prediction.load().height == _predictions().height


def test_workspace_can_freeze_and_rank_a_released_backtest(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    release_case = release / "case_studies" / "etfs"
    prediction_hash = _publish(release_case, spec=_resolved_spec())
    returns = pl.DataFrame({"timestamp": ["2024-01-05"], "return": [0.01]}).with_columns(
        pl.col("timestamp").str.to_date()
    )
    backtest_hash = register_backtest_run(
        "etfs",
        prediction_hash,
        {
            "identity_version": 2,
            "execution_tier": "canonical",
            "strategy": {"signal": {"method": "equal_weight_top_k", "top_k": 1}},
        },
        stage="signal",
        returns=returns,
        metrics={"sharpe": 1.0},
        case_dir=release_case,
    )
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    selected = study.backtests.table().filter(pl.col("backtest_hash") == backtest_hash)

    frozen = study.backtests.freeze(selected, name="released-backtest")
    ranked = frozen.ranked_validation_sharpe()

    assert selected.item(0, "origin") == "released"
    assert ranked[0].hash == backtest_hash
    assert ranked[0].origin == "released"


def test_quick_start_uses_human_fields_and_exposes_lineage_within_budget(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    release_case = release / "case_studies" / "etfs"
    _publish(release_case, spec=_resolved_spec())
    started = time.perf_counter()

    study = Study.open("etfs", workspace=tmp_path / "reader-workspace", release_root=release)
    selected = study.predictions.table().filter(
        (pl.col("label") == "fwd_ret_21d")
        & (pl.col("family") == "linear")
        & (pl.col("config_name") == "ridge")
        & (pl.col("split") == "validation")
        & pl.col("complete")
    )
    backtest = run_backtests(
        study,
        predictions=selected,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        prices=_prices(),
    ).results[0]
    lineage = backtest.lineage()

    assert selected.height == 1
    assert lineage["prediction_hash"] == selected.item(0, "prediction_hash")
    assert lineage["training_spec"]["config_name"] == "ridge"
    assert time.perf_counter() - started < 15 * 60


def test_unfinished_training_cannot_satisfy_an_official_population(tmp_path: Path) -> None:
    study = _study(tmp_path)
    training = study.results.register_training(_resolved_spec())
    population = OfficialPopulation.create(
        study,
        name="training-population",
        member_kind="training",
        members=[training.hash],
    )
    assert OfficialPopulation.one(study, name="training-population").hash == population.hash

    assert training.complete is False
    with pytest.raises(ValueError, match="partial"):
        population.require_complete()

    attempt = study.executions.start(training.hash)
    attempt.finish("completed", {"fitted_folds": [0]})
    assert Result.open(study, training.hash).complete is True
    assert population.require_complete() == (training.hash,)


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


def test_refit_under_a_changed_parameter_needs_the_snapshot_it_replaces(tmp_path: Path) -> None:
    """A population is prediction identities, so a changed estimator parameter replaces it.

    This is the `max_bin` refit: the configuration menu is untouched and every member name is the
    same, but each prediction hashes differently, so the same population name now describes a
    different set. Without naming what it replaces the second snapshot is refused, and the lineage
    that says which run produced which predictions would not exist.
    """
    study = _study(tmp_path)

    def request(alpha: float) -> ResolvedModelRequest:
        spec = _resolved_spec(alpha=alpha)
        spec["computation"]["checkpoint_schedule"] = [{"kind": "epoch", "value": 1}]
        return ResolvedModelRequest(study=study, family="linear", spec=spec, _context=None)

    first = snapshot_official_models(study, [request(1.0)], population_name="gbm-validation-v1")
    (before,) = first.members

    second_members = snapshot_official_models(
        study, [request(2.0)], population_name="other-name"
    ).members
    assert second_members != first.members, "a changed parameter must move the prediction identity"

    with pytest.raises(ValueError, match="supersedes"):
        snapshot_official_models(study, [request(2.0)], population_name="gbm-validation-v1")

    replacement = snapshot_official_models(
        study,
        [request(2.0)],
        population_name="gbm-validation-v1",
        supersedes=first.hash,
    )
    assert replacement.members == second_members
    assert replacement.hash != first.hash
    assert OfficialPopulation.open(study, first.hash).members == (before,)


def test_resolving_a_population_by_name_returns_the_generation_in_force(tmp_path: Path) -> None:
    """Downstream notebooks ask by name, so superseding must not make the name ambiguous."""
    study = _study(tmp_path)

    def request(alpha: float) -> ResolvedModelRequest:
        spec = _resolved_spec(alpha=alpha)
        spec["computation"]["checkpoint_schedule"] = [{"kind": "epoch", "value": 1}]
        return ResolvedModelRequest(study=study, family="linear", spec=spec, _context=None)

    first = snapshot_official_models(study, [request(1.0)], population_name="gbm-v1")
    assert OfficialPopulation.one(study, name="gbm-v1").hash == first.hash

    second = snapshot_official_models(
        study, [request(2.0)], population_name="gbm-v1", supersedes=first.hash
    )
    assert OfficialPopulation.one(study, name="gbm-v1").hash == second.hash
    assert OfficialPopulation.open(study, first.hash).members == first.members

    third = snapshot_official_models(
        study, [request(3.0)], population_name="gbm-v1", supersedes=second.hash
    )
    assert OfficialPopulation.one(study, name="gbm-v1").hash == third.hash


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
