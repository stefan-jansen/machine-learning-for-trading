from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import (
    CandidateSet,
    DecisionArtifact,
    StateTransitionPolicy,
    Study,
)
from case_studies.research.strategy import apply_state_transition_policy
from tests.test_research_contract_catalog import _publish, _resolved_spec
from tests.test_research_flow import _patch_holdout_prices, _prices
from tests.test_research_registry import _predictions
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


def _decisions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": [date(2024, 1, 5), date(2024, 1, 5)],
            "position": [1.0, 0.0],
        }
    )


def test_fold_boundary_liquidates_unchanged_positions_before_next_fold() -> None:
    decisions = pl.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "timestamp": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
            ],
            "fold": [0, 0, 1],
            "weight": pl.Series([1.0, 1.0, 1.0], dtype=pl.Float32),
        }
    )

    transitioned = apply_state_transition_policy(
        decisions,
        policy={"fold_boundary": "liquidate", "temporal_gap": "continue"},
        cadence="1d",
    )

    assert transitioned.get_column("weight").to_list() == [1.0, 1.0, 1.0]
    assert transitioned.get_column("_state_transition").to_list() == [False, False, True]


def test_state_transition_preserves_millisecond_utc_timestamp_dtype() -> None:
    timestamps = pl.Series(
        "timestamp",
        [datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2)],
        dtype=pl.Datetime("ms", "UTC"),
    )
    weights = pl.DataFrame(
        {
            "symbol": ["A", "A"],
            "timestamp": timestamps,
            "fold": [0, 1],
            "weight": [1.0, 1.0],
        }
    )

    transitioned = apply_state_transition_policy(
        weights,
        policy={"fold_boundary": "liquidate", "temporal_gap": "continue"},
        cadence="1d",
    )

    assert transitioned.schema["timestamp"] == pl.Datetime("ms", "UTC")
    assert transitioned.get_column("_state_transition").to_list() == [False, True]


def test_temporal_gap_resets_unchanged_positions_before_gap() -> None:
    decisions = pl.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "timestamp": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 5),
            ],
            "fold": [0, 0, 0],
            "weight": pl.Series([1.0, 1.0, 1.0], dtype=pl.Float32),
        }
    )

    transitioned = apply_state_transition_policy(
        decisions,
        policy={"fold_boundary": "continue", "temporal_gap": "reset"},
        cadence="1d",
        price_keys=pl.DataFrame(
            {
                "symbol": ["A"] * 5,
                "timestamp": [date(2024, 1, day) for day in range(1, 6)],
            }
        ),
    )

    assert transitioned.get_column("timestamp").to_list() == [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 5),
    ]
    assert transitioned.get_column("weight").to_list() == [1.0, 1.0, 0.0, 1.0]
    assert transitioned.schema["weight"] == pl.Float32
    assert transitioned.get_column("_state_transition").to_list() == [
        False,
        False,
        True,
        False,
    ]


def test_rebalance_thinned_weights_do_not_read_as_observation_gaps() -> None:
    """A coarser weight frame than the declared cadence is not a gap.

    precompute_weights thins to the non-overlapping rebalance schedule, so crypto's
    fwd_ret_24h (rebalance_step 3) produces weights 24h apart while 16_risk_management
    declares an 8h cadence. Running the gap test over the weight frame made every ordinary
    rebalance look like a missing observation, which flattened the book one bar later and
    left it flat for two of every three bars.
    """
    observations = [datetime(2024, 1, 1, hour, tzinfo=UTC) for hour in (0, 8, 16)] + [
        datetime(2024, 1, 2, hour, tzinfo=UTC) for hour in (0, 8, 16)
    ]
    # Weights only on the rebalance schedule: every third observation.
    rebalanced = [observations[0], observations[3]]
    weights = pl.DataFrame(
        {
            "symbol": ["A"] * len(rebalanced),
            "timestamp": rebalanced,
            "fold": [0] * len(rebalanced),
            "weight": [1.0] * len(rebalanced),
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    timeline = pl.DataFrame(
        {"timestamp": observations, "fold": [0] * len(observations)}
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    prices = pl.DataFrame(
        {"symbol": ["A"] * len(observations), "timestamp": observations}
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

    transitioned = apply_state_transition_policy(
        weights,
        policy={"fold_boundary": "liquidate", "temporal_gap": "reset"},
        cadence="8h",
        price_keys=prices,
        timeline=timeline,
    )

    assert not transitioned.get_column("_state_transition").any(), (
        "a rebalance interval was misread as an observation gap"
    )
    assert transitioned.height == weights.height, (
        "synthetic flat rows were inserted for ordinary rebalance intervals"
    )


def test_fold_boundary_off_the_weight_grid_liquidates_at_the_boundary() -> None:
    """A declared liquidation happens when it is declared, not at the next rebalance.

    With a thinned weight frame the fold boundary usually falls between weight rows. Snapping
    the mark forward would carry the old fold's book into the new fold, and because the snapped
    timestamp is itself a rebalance bar the engine would flatten and re-apply new targets on the
    same bar - a round trip for no change in exposure.
    """
    observations = [datetime(2024, 1, 1, hour, tzinfo=UTC) for hour in (0, 8, 16)]
    # Weights only at 00:00; the fold changes at 08:00, which is not a weight timestamp.
    weights = pl.DataFrame(
        {"symbol": ["A"], "timestamp": [observations[0]], "fold": [0], "weight": [1.0]}
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    timeline = pl.DataFrame({"timestamp": observations, "fold": [0, 1, 1]}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )
    prices = pl.DataFrame({"symbol": ["A"] * 3, "timestamp": observations}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )

    transitioned = apply_state_transition_policy(
        weights,
        policy={"fold_boundary": "liquidate", "temporal_gap": "continue"},
        cadence="8h",
        price_keys=prices,
        timeline=timeline,
    )

    marked = transitioned.filter(pl.col("_state_transition"))
    assert marked.height == 1, "the fold boundary produced no executable reset"
    assert marked.item(0, "timestamp") == observations[1], (
        "the liquidation was snapped away from the boundary it was declared for"
    )
    assert marked.item(0, "weight") == 0.0, "the reset did not flatten the position"


def test_a_fold_boundary_and_a_gap_at_one_moment_insert_one_flat_row() -> None:
    """Both policies can resolve to the same off-grid moment; the reset must still execute.

    The on-grid path dedupes through a set. Without the same guard on the flat-row path the
    moment gets two identical rows, and `_target_weights_by_timestamp` rejects duplicate
    (symbol, timestamp) pairs - so the declared reset would abort the backtest rather than run.
    """
    observations = [
        datetime(2024, 1, 1, 0, tzinfo=UTC),
        # 08:00 missing: a gap, and the fold changes across it too
        datetime(2024, 1, 1, 16, tzinfo=UTC),
    ]
    weights = pl.DataFrame(
        {"symbol": ["A"], "timestamp": [observations[0]], "fold": [0], "weight": [1.0]}
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    timeline = pl.DataFrame({"timestamp": observations, "fold": [0, 1]}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )
    # 08:00 has no price bar, so the gap branch falls through to the same _mark call the fold
    # branch makes for 16:00.
    prices = pl.DataFrame({"symbol": ["A", "A"], "timestamp": observations}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )

    transitioned = apply_state_transition_policy(
        weights,
        policy={"fold_boundary": "liquidate", "temporal_gap": "reset"},
        cadence="8h",
        price_keys=prices,
        timeline=timeline,
    )

    keys = transitioned.select("symbol", "timestamp")
    assert keys.height == keys.unique().height, (
        "the same moment produced duplicate rows, which the backtest runner rejects"
    )
    assert transitioned.filter(pl.col("_state_transition")).height == 1


def test_a_declared_reset_that_cannot_be_represented_fails_closed() -> None:
    """A reset with no price bar to execute on is a contract violation, not a no-op."""
    observations = [datetime(2024, 1, 1, hour, tzinfo=UTC) for hour in (0, 8)]
    weights = pl.DataFrame(
        {"symbol": ["A"], "timestamp": [observations[0]], "fold": [0], "weight": [1.0]}
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    timeline = pl.DataFrame({"timestamp": observations, "fold": [0, 1]}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )
    # 08:00 has no price bar, so the declared liquidation cannot be executed anywhere.
    prices = pl.DataFrame({"symbol": ["A"], "timestamp": [observations[0]]}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )

    with pytest.raises(ValueError, match="cannot be represented on the weight grid"):
        apply_state_transition_policy(
            weights,
            policy={"fold_boundary": "liquidate", "temporal_gap": "continue"},
            cadence="8h",
            price_keys=prices,
            timeline=timeline,
        )


def test_a_real_observation_gap_still_resets_thinned_weights() -> None:
    """The counterpart: a hole in the observation grid must still reset state."""
    observations = [
        datetime(2024, 1, 1, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 8, tzinfo=UTC),
        # 16:00 is missing - a real gap in the observation grid
        datetime(2024, 1, 2, 0, tzinfo=UTC),
    ]
    weights = pl.DataFrame(
        {
            "symbol": ["A", "A"],
            "timestamp": [observations[0], observations[2]],
            "fold": [0, 0],
            "weight": [1.0, 1.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    timeline = pl.DataFrame({"timestamp": observations, "fold": [0, 0, 0]}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )
    prices = pl.DataFrame(
        {
            "symbol": ["A"] * 4,
            "timestamp": observations + [datetime(2024, 1, 1, 16, tzinfo=UTC)],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

    transitioned = apply_state_transition_policy(
        weights,
        policy={"fold_boundary": "liquidate", "temporal_gap": "reset"},
        cadence="8h",
        price_keys=prices,
        timeline=timeline,
    )

    assert transitioned.get_column("_state_transition").any(), (
        "a real hole in the observation grid did not reset state"
    )


def test_generated_strategy_weights_execute_declared_fold_and_gap_policy(
    tmp_path: Path,
) -> None:
    study = _study(tmp_path)
    timestamps = [date(2024, 1, day) for day in (1, 2, 5)]
    frame = pl.DataFrame(
        {
            "symbol": [symbol for timestamp in timestamps for symbol in ("A", "B")],
            "timestamp": [timestamp for timestamp in timestamps for _ in ("A", "B")],
            "fold_id": [fold for fold in (0, 0, 1) for _ in ("A", "B")],
            "y_true": [0.01, -0.01] * len(timestamps),
            "y_score": [0.02, -0.02] * len(timestamps),
        }
    )
    training = study.results.register_training(_resolved_spec())
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )
    prices = pl.DataFrame(
        {
            "symbol": [symbol for day in range(1, 6) for symbol in ("A", "B")],
            "timestamp": [date(2024, 1, day) for day in range(1, 6) for _ in ("A", "B")],
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1_000] * 10,
        }
    )
    strategy = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        risk={
            "state_transition_policy": {
                "fold_boundary": "liquidate",
                "temporal_gap": "reset",
            },
            "state_transition_cadence": "1d",
        },
    )

    weights = strategy._risk_state_weights(frame, prices, strategy.resolve(prices=prices))

    assert weights is not None
    transitions = weights.filter(pl.col("_state_transition")).get_column("timestamp").unique()
    assert {timestamp.date() for timestamp in transitions} == {
        date(2024, 1, 3),
        date(2024, 1, 5),
    }


def _prediction(study: Study) -> str:
    training = study.results.register_training(_resolved_spec())
    frame = _predictions()
    return study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    ).hash


def test_stateful_decision_artifact_requires_policy_before_write(tmp_path: Path) -> None:
    study = _study(tmp_path)

    with pytest.raises(ValueError, match="state transition policy"):
        DecisionArtifact.publish(
            study,
            kind="target_positions",
            decisions=_decisions(),
            prediction_hashes=["prediction-a"],
            parameters={"entry_threshold": 0.5},
        )

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM decision_artifacts").fetchone() == (0,)
    assert not (study.root / "run_log" / "decisions").exists()


def test_typed_decision_artifact_survives_restart_with_state_policy(tmp_path: Path) -> None:
    study = _study(tmp_path)
    policy = StateTransitionPolicy(fold_boundary="liquidate", temporal_gap="reset")
    prediction_hash = _prediction(study)
    published = DecisionArtifact.publish(
        study,
        kind="target_positions",
        decisions=_decisions(),
        prediction_hashes=[prediction_hash],
        parameters={"entry_threshold": 0.5},
        state_transition_policy=policy,
    )
    reopened_study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=study.release_root
    )
    reopened = DecisionArtifact.open(reopened_study, published.hash)

    assert reopened.kind == "target_positions"
    assert reopened.spec["state_transition_policy"] == {
        "fold_boundary": "liquidate",
        "temporal_gap": "reset",
    }
    assert reopened.load().equals(_decisions())


def test_identical_decision_publishers_and_orphan_recovery_preserve_artifact(
    tmp_path: Path,
) -> None:
    study = _study(tmp_path)
    prediction_hash = _prediction(study)
    policy = StateTransitionPolicy(fold_boundary="reset", temporal_gap="reset")

    def publish() -> DecisionArtifact:
        return DecisionArtifact.publish(
            study,
            kind="target_positions",
            decisions=_decisions(),
            prediction_hashes=[prediction_hash],
            parameters={"threshold": 0.5},
            state_transition_policy=policy,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        published = list(pool.map(lambda _: publish(), range(2)))

    assert published[0].hash == published[1].hash
    assert published[0].path.is_file()
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "DELETE FROM decision_artifacts WHERE decision_hash = ?",
            (published[0].hash,),
        )
        db.commit()

    recovered = publish()
    assert recovered.load().equals(_decisions())


def test_released_decision_artifact_opens_without_workspace_copy(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    release_case = release / "case_studies" / "etfs"
    prediction_hash = _publish(release_case, spec=_resolved_spec())
    writable_release = Study("etfs", release_case, release, release.parent, False, {})
    published = DecisionArtifact.publish(
        writable_release,
        kind="target_positions",
        decisions=_decisions(),
        prediction_hashes=[prediction_hash],
        parameters={},
        state_transition_policy=StateTransitionPolicy("reset", "reset"),
    )
    workspace = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)

    reopened = DecisionArtifact.open(workspace, published.hash)

    assert reopened.root == release_case
    assert reopened.load().equals(_decisions())


def test_canonical_decision_promotion_requires_replay_evidence(tmp_path: Path) -> None:
    study = _study(tmp_path)
    policy = StateTransitionPolicy(fold_boundary="reset", temporal_gap="reset")
    prediction_hash = _prediction(study)

    with pytest.raises(ValueError, match="missing evidence"):
        DecisionArtifact.publish(
            study,
            kind="target_positions",
            decisions=_decisions(),
            prediction_hashes=[prediction_hash],
            parameters={},
            source_identity={"module": "research.custom"},
            state_transition_policy=policy,
            canonical=True,
        )

    with pytest.raises(ValueError, match="exact prediction_hashes"):
        DecisionArtifact.publish(
            study,
            kind="target_positions",
            decisions=_decisions(),
            prediction_hashes=[prediction_hash],
            parameters={},
            source_identity={
                "module": "case_studies.research.decisions",
                "source_digest": "source-a",
                "declared_inputs": {"prediction_hashes": ["undisclosed-other-input"]},
                "determinism": {"seed": 42},
                "clean_replay_digest": "not-reached",
            },
            state_transition_policy=policy,
            canonical=True,
        )

    exploratory = DecisionArtifact.publish(
        study,
        kind="target_positions",
        decisions=_decisions(),
        prediction_hashes=[prediction_hash],
        parameters={},
        state_transition_policy=policy,
    )
    evidence = {
        "module": "case_studies.research.decisions",
        "source_digest": "source-a",
        "declared_inputs": {"prediction_hashes": [prediction_hash]},
        "determinism": {"seed": 42},
        "clean_replay_digest": exploratory.spec["artifact_digest"],
    }
    canonical = DecisionArtifact.publish(
        study,
        kind="target_positions",
        decisions=_decisions(),
        prediction_hashes=[prediction_hash],
        parameters={},
        source_identity=evidence,
        state_transition_policy=policy,
        canonical=True,
    )

    assert not exploratory.canonical
    assert canonical.canonical
    assert canonical.hash != exploratory.hash


def test_holdout_stages_then_transitions_in_one_atomic_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path)
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    validation_spec = _resolved_spec()
    validation_training = study.results.register_training(validation_spec)
    validation_prediction = study.results.publish_predictions(
        validation_training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )
    request = {
        "signal": {"method": "equal_weight_top_k", "top_k": 1},
        "execution_mode": "vectorized",
    }
    validation_backtest = study.strategy(prediction=validation_prediction, **request).run(
        prices=_prices()
    )
    candidates = CandidateSet.create(study, "selection", [validation_backtest])
    holdout_spec = _resolved_spec()
    holdout_spec["computation"]["cv"] = {
        "identity": "holdout-cv",
        "request": {"phase": "holdout", "train_end": "2024-01-04"},
    }
    holdout_prices = _prices().with_columns(pl.lit(date(2024, 1, 11)).alias("timestamp"))
    _patch_holdout_prices(monkeypatch, holdout_prices)
    lock = study.lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=validation_backtest.hash,
        selection_evidence={"metric": "validation_backtest_sharpe"},
        holdout_training_spec=holdout_spec,
    )
    assert (
        study.lifecycle.lock(
            candidate_set_hash=candidates.hash,
            selected_backtest_hash=validation_backtest.hash,
            selection_evidence={"metric": "validation_backtest_sharpe"},
            holdout_training_spec=holdout_spec,
        ).hash
        == lock.hash
    )
    holdout_training = study.results.register_training(holdout_spec)
    holdout_frame = frame.with_columns(pl.lit(date(2024, 1, 11)).alias("timestamp"))
    holdout_prediction = study.results.publish_predictions(
        holdout_training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="holdout",
        predictions=holdout_frame,
        expected_keys=holdout_frame.select("symbol", "timestamp", "fold_id"),
    )
    holdout_backtest = study.strategy(prediction=holdout_prediction, **request).run()

    staged = study.lifecycle.stage_holdout(
        lock.hash,
        holdout_training_hash=holdout_training.hash,
        holdout_prediction_hash=holdout_prediction.hash,
        holdout_backtest_hash=holdout_backtest.hash,
    )
    assert staged.state == "LOCKED"
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_staging").fetchone() == (1,)
        assert db.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone() == (0,)

    from case_studies.research import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "load_backtest_prices_for",
        lambda *args, **kwargs: holdout_prices.with_columns((pl.col("close") + 1).alias("close")),
    )
    with pytest.raises(ValueError, match="exact complete canonical lineage"):
        study.lifecycle.finalize_holdout(lock.hash)
    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    monkeypatch.setattr(
        lifecycle,
        "load_backtest_prices_for",
        lambda *args, **kwargs: holdout_prices,
    )

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "CREATE TRIGGER force_holdout_failure BEFORE UPDATE OF state ON research_locks "
            "WHEN NEW.state = 'HOLDOUT_EVALUATED' BEGIN "
            "SELECT RAISE(ABORT, 'forced holdout failure'); END"
        )
        db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="forced holdout failure"):
        study.lifecycle.finalize_holdout(lock.hash)
    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone() == (0,)
        db.execute("DROP TRIGGER force_holdout_failure")
        db.commit()

    evaluated = study.lifecycle.finalize_holdout(lock.hash)
    assert evaluated.state == "HOLDOUT_EVALUATED"
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone() == (1,)
