from __future__ import annotations

import hashlib
import inspect
import os
import sqlite3
from copy import deepcopy
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from case_studies.research import (
    CandidateSet,
    DecisionArtifact,
    LabelDefinition,
    ModelRun,
    ResearchLock,
    ResolvedModelRequest,
    StateTransitionPolicy,
    Study,
    registered_adapters,
    run_locked_holdout,
)
from case_studies.utils.artifact_digest import value_digest
from tests.test_research_contract_catalog import _resolved_spec
from tests.test_research_flow import _patch_holdout_prices, _prices
from tests.test_research_models import _gbm_study, _linear_study
from tests.test_research_registry import _predictions
from tests.test_research_workspace import _seed_release


def replay_ranked_positions(
    predictions: pl.DataFrame,
    *,
    long_count: int = 1,
    short_count: int = 1,
    universe: list[str] | None = None,
) -> pl.DataFrame:
    score_column = "prediction" if "prediction" in predictions.columns else "y_score"
    fold_column = "fold" if "fold" in predictions.columns else "fold_id"
    eligible = (
        predictions.filter(pl.col("symbol").is_in(universe))
        if universe is not None
        else predictions
    )
    ranked = eligible.with_columns(
        pl.col(score_column).rank("ordinal").over("timestamp").alias("ascending"),
        pl.col(score_column).rank("ordinal", descending=True).over("timestamp").alias("descending"),
    )
    return (
        ranked.with_columns(
            pl.when(pl.col("descending") <= long_count)
            .then(1.0)
            .when(pl.col("ascending") <= short_count)
            .then(-1.0)
            .otherwise(0.0)
            .alias("position")
        )
        .select("symbol", "timestamp", fold_column, "position")
        .sort("timestamp", "symbol")
    )


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _locked_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    decision: DecisionArtifact | None = None,
):
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    validation_spec = _resolved_spec()
    validation_spec["computation"]["checkpoint_schedule"] = [{"kind": "final", "value": None}]
    validation_training = study.results.register_training(validation_spec)
    frame = _predictions()
    validation_prediction = study.results.publish_predictions(
        validation_training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )
    request = {
        "signal": {"method": "equal_weight_top_k", "top_k": 1},
        "execution_mode": "vectorized",
    }
    validation_backtest = study.strategy(
        prediction=validation_prediction,
        decision=decision,
        **request,
    ).run(prices=_prices())
    candidates = CandidateSet.create(study, "locked-selection", [validation_backtest])
    holdout_spec = deepcopy(validation_spec)
    holdout_spec["computation"]["cv"] = {
        "identity": "holdout-cv",
        "split": "holdout",
        "train_start": "2024-01-01",
        "train_end": "2024-01-10",
        "evaluation_start": "2024-01-11",
        "evaluation_end": "2024-01-11",
    }
    holdout_prices = _prices().with_columns(pl.lit(date(2024, 1, 11)).alias("timestamp"))
    _patch_holdout_prices(monkeypatch, holdout_prices)
    lock = study.lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=validation_backtest.hash,
        selection_evidence={"metric": "validation_backtest_sharpe"},
        holdout_training_spec=holdout_spec,
    )
    return study, lock, holdout_prices


def _install_fixture_adapter(
    monkeypatch: pytest.MonkeyPatch,
    holdout_prices: pl.DataFrame,
    *,
    interrupt_after_stage: bool = False,
    published_checkpoint: tuple[str, int | None] = ("final", None),
):
    from case_studies.research import models

    fit_calls: list[str] = []

    def reconstruct(study, spec, *, checkpoint_kind, checkpoint_value):
        assert (checkpoint_kind, checkpoint_value) == ("final", None)
        return ResolvedModelRequest(study, "linear", spec, holdout_prices)

    def run(study, spec, context):
        training = study.results.register_training(spec)
        model = training.root / "run_log" / "training" / training.hash / "models" / "fitted.bin"
        model.parent.mkdir(parents=True, exist_ok=True)
        if not model.exists():
            fit_calls.append(training.hash)
            model.write_bytes(b"fitted-model-state")
        frame = _predictions().with_columns(pl.lit(date(2024, 1, 11)).alias("timestamp"))
        prediction = study.results.publish_predictions(
            training,
            checkpoint_kind=published_checkpoint[0],
            checkpoint_value=published_checkpoint[1],
            split="holdout",
            predictions=frame,
            expected_keys=frame.select("symbol", "timestamp", "fold_id"),
        )
        return ModelRun(training, (prediction,))

    def validate(study, spec, context, run):
        model = (
            run.training.root / "run_log" / "training" / run.training.hash / "models" / "fitted.bin"
        )
        if not model.is_file():
            raise ValueError("missing fitted state")
        return hashlib.sha256(model.read_bytes()).hexdigest()

    adapter = SimpleNamespace(
        reconstruct_locked_request=reconstruct,
        run_resolved_request=run,
        validate_locked_run=validate,
    )
    monkeypatch.setattr(models, "get_adapter", lambda kind, name: adapter)
    if interrupt_after_stage:
        from case_studies.research import lifecycle

        original = lifecycle.Lifecycle.finalize_holdout
        interrupted = False

        def finalize(self, lock_hash):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise RuntimeError("injected finalization interruption")
            return original(self, lock_hash)

        monkeypatch.setattr(lifecycle.Lifecycle, "finalize_holdout", finalize)
    return fit_calls


def test_every_registered_model_family_owns_locked_reconstruction() -> None:
    for binding in registered_adapters("model"):
        module = binding.load()
        assert callable(getattr(module, "reconstruct_locked_request", None)), binding.name
        assert callable(getattr(module, "validate_locked_run", None)), binding.name


def test_latent_holdout_retry_preserves_conflicting_fold_diagnostics(tmp_path: Path) -> None:
    from case_studies.utils.latent_factors import adapter

    model_dir = tmp_path / "models"
    train_dir = tmp_path / "training"
    model_dir.mkdir()
    train_dir.mkdir()
    source = model_dir / "fold_extras.json"
    target = train_dir / "fold_extras.json"
    source.write_text('[{"fold_id": 0, "converged": true}]\n')
    target.write_text('[{"fold_id": 0, "converged": false}]\n')
    conflicting = target.read_bytes()

    with pytest.raises(ValueError, match="diagnostics conflict"):
        adapter._publish_fold_extras(model_dir, train_dir, immutable=True)

    assert target.read_bytes() == conflicting


def test_cme_family_eligibility_keys_use_the_internal_symbol_contract() -> None:
    """Every family names the eligibility entity `symbol`, whatever the reader key is.

    cme_futures is the only product-keyed case study, so it is the sole exerciser of
    this rename. The locked reconstruction validates a recorded eligibility digest, so
    one family drifting off the shared naming would make its locks unreproducible.
    """
    from case_studies.utils import gbm, linear, tabular_dl
    from case_studies.utils.latent_factors.cv import _build_prediction_frame
    from case_studies.utils.sequence_dataset import sequence_validation_keys

    dates = [date(2024, 1, day) for day in range(1, 5)]
    dataset = pl.DataFrame(
        {
            "product": [product for product in ("ES", "NQ") for _ in dates],
            "timestamp": dates * 2,
            "feature": list(range(8)),
            "fwd_ret_1d": [value / 100 for value in range(8)],
        }
    )
    split = {
        "fold": 0,
        "train_start": dates[0],
        "train_end": dates[1],
        "val_start": dates[2],
        "val_end": dates[3],
    }
    mds = SimpleNamespace(
        dataset=dataset,
        entity_cols=["product"],
        date_col="timestamp",
        label_col="fwd_ret_1d",
        eval_label_col=None,
    )
    fold = {
        "fold": 0,
        "entities": np.array(["ES", "NQ"]),
        "dates": np.array([dates[2], dates[2]], dtype="datetime64[D]"),
        "n_val": 2,
    }

    linear_keys = linear._expected_keys_from_dataset(
        dataset,
        [split],
        entity_col="product",
        date_col="timestamp",
        label_col="fwd_ret_1d",
    )
    gbm_keys = gbm._gbm_expected_keys([fold], "product", "timestamp")
    tabm_keys = tabular_dl._tabm_expected_keys(mds, [split])
    sequence_keys = sequence_validation_keys(
        dataset.to_pandas(),
        [split],
        label_col="fwd_ret_1d",
        date_col="timestamp",
        entity_col="product",
        lookback=1,
    )
    latent = _build_prediction_frame(
        predictions=np.array([[0.1, -0.1]]),
        returns_val=np.array([[0.01, -0.01]]),
        eval_returns_val=None,
        val_dates=np.array([dates[2]], dtype="datetime64[D]"),
        val_entities=np.array([["ES", "NQ"]]),
        fold_id=0,
        model_name="pca",
        epoch=0,
    )

    # Every builder names the entity `symbol`, whatever reader key it was given. The
    # runners that carry the reader key further rename it at the publish boundary.
    for frame in (linear_keys, gbm_keys, tabm_keys, sequence_keys, latent):
        assert frame is not None
        assert "symbol" in frame.columns
        assert "product" not in frame.columns


def _real_linear_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    eligibility_row_delta: int = 0,
):
    from case_studies.utils import backtest_loaders, cv_window

    monkeypatch.setattr(backtest_loaders, "get_rebalance_step", lambda *args, **kwargs: 1)
    monkeypatch.setattr(cv_window, "canonical_window", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv_window, "fold_boundaries", lambda *args, **kwargs: [])
    study = _linear_study(tmp_path, monkeypatch)
    validation = study.model(family="linear", label="fwd_ret_1d", config_name="ridge").resolve()
    validation_run = validation.run()
    validation_prediction = validation_run.predictions[0]
    validation_prices = (
        validation_prediction.load()
        .select("symbol", "timestamp")
        .unique()
        .with_columns(
            pl.lit(100.0).alias("open"),
            pl.lit(101.0).alias("high"),
            pl.lit(99.0).alias("low"),
            (pl.lit(100.0) + pl.int_range(pl.len()).cast(pl.Float64) / 100).alias("close"),
            pl.lit(1_000).alias("volume"),
        )
        .sort("timestamp", "symbol")
    )
    validation_backtest = study.strategy(
        prediction=validation_prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="vectorized",
    ).run(prices=validation_prices)
    candidates = CandidateSet.create(study, "linear-lock", [validation_backtest])

    holdout_spec = deepcopy(validation.spec)
    holdout_fold = {
        "fold": 2,
        "train_start": "2024-01-01",
        "train_end": "2024-01-03",
        "val_start": "2024-01-04",
        "val_end": "2024-01-04",
    }
    holdout_spec["computation"]["cv"] = {
        "identity": "linear-holdout",
        "split": "holdout",
        "folds": [holdout_fold],
    }
    params = validation.spec["computation"]["model"]["effective_params_by_fold"]["1"]
    holdout_spec["computation"]["model"]["effective_params_by_fold"] = {"2": params}
    holdout_keys = (
        validation_prices.filter(pl.col("timestamp") == date(2024, 1, 4))
        .select("symbol", "timestamp")
        .with_columns(pl.lit(2, dtype=pl.Int64).alias("fold"))
        .sort("symbol", "timestamp", "fold")
    )
    holdout_spec["computation"]["expected_prediction_keys"] = {
        "digest": value_digest(holdout_keys, ("symbol", "timestamp", "fold")),
        "n_rows": holdout_keys.height + eligibility_row_delta,
        "n_folds": 1,
    }
    holdout_prices = validation_prices.filter(pl.col("timestamp") == date(2024, 1, 4))

    def load_prices(case_study, label, *, split, warmup_periods=0):
        assert (case_study, label, split, warmup_periods) == (
            "etfs",
            "fwd_ret_1d",
            "holdout",
            0,
        )
        return holdout_prices

    from case_studies.research import lifecycle, strategy

    monkeypatch.setattr(lifecycle, "load_backtest_prices_for", load_prices)
    monkeypatch.setattr(strategy, "load_backtest_prices_for", load_prices)
    monkeypatch.setattr(
        cv_window,
        "canonical_window",
        lambda case_study, label, *, split: (date(2024, 1, 4), date(2024, 1, 4)),
    )
    lock = study.lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=validation_backtest.hash,
        selection_evidence={"metric": "validation_backtest_sharpe"},
        holdout_training_spec=holdout_spec,
    )
    return study, lock, holdout_keys


def _real_gbm_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from case_studies.utils import backtest_loaders, cv_window
    from case_studies.utils import gbm as gbm_adapter

    monkeypatch.setattr(backtest_loaders, "get_rebalance_step", lambda *args, **kwargs: 1)
    monkeypatch.setattr(cv_window, "canonical_window", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv_window, "fold_boundaries", lambda *args, **kwargs: [])
    real_train = gbm_adapter.train_gbm_config
    study = _gbm_study(tmp_path, monkeypatch)
    monkeypatch.setattr(gbm_adapter, "train_gbm_config", real_train)
    validation = study.model(
        family="gbm",
        label="fwd_ret_1d",
        config_name="leaves_7_mse",
        overrides={"device": "cpu", "max_bin": 63},
    ).resolve()
    validation_run = validation.run()
    validation_prediction = validation_run.predictions[0]
    validation_prices = (
        validation_prediction.load()
        .select("symbol", "timestamp")
        .unique()
        .with_columns(
            pl.lit(100.0).alias("open"),
            pl.lit(101.0).alias("high"),
            pl.lit(99.0).alias("low"),
            (pl.lit(100.0) + pl.int_range(pl.len()).cast(pl.Float64) / 100).alias("close"),
            pl.lit(1_000).alias("volume"),
        )
        .sort("timestamp", "symbol")
    )
    validation_backtest = study.strategy(
        prediction=validation_prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="vectorized",
    ).run(prices=validation_prices)
    candidates = CandidateSet.create(study, "gbm-lock", [validation_backtest])
    holdout_spec = deepcopy(validation.spec)
    holdout_spec["computation"]["cv"] = {
        "identity": "gbm-holdout",
        "split": "holdout",
        "folds": [
            {
                "fold": 2,
                "train_start": "2024-01-01",
                "train_end": "2024-01-03",
                "val_start": "2024-01-04",
                "val_end": "2024-01-04",
            }
        ],
    }
    params = validation.spec["computation"]["model"]["effective_params_by_fold"]["1"]
    holdout_spec["computation"]["model"]["effective_params_by_fold"] = {"2": params}
    holdout_keys = (
        validation_prices.filter(pl.col("timestamp") == date(2024, 1, 4))
        .select("symbol", "timestamp")
        .with_columns(pl.lit(2, dtype=pl.Int64).alias("fold"))
        .sort("symbol", "timestamp", "fold")
    )
    holdout_spec["computation"]["expected_prediction_keys"] = {
        "digest": value_digest(holdout_keys, ("symbol", "timestamp", "fold")),
        "n_rows": holdout_keys.height,
        "n_folds": 1,
    }
    holdout_prices = validation_prices.filter(pl.col("timestamp") == date(2024, 1, 4))

    def load_prices(case_study, label, *, split, warmup_periods=0):
        assert (case_study, label, split, warmup_periods) == (
            "etfs",
            "fwd_ret_1d",
            "holdout",
            0,
        )
        return holdout_prices

    from case_studies.research import lifecycle, strategy

    monkeypatch.setattr(lifecycle, "load_backtest_prices_for", load_prices)
    monkeypatch.setattr(strategy, "load_backtest_prices_for", load_prices)
    monkeypatch.setattr(
        cv_window,
        "canonical_window",
        lambda case_study, label, *, split: (date(2024, 1, 4), date(2024, 1, 4)),
    )
    lock = study.lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=validation_backtest.hash,
        selection_evidence={"metric": "validation_backtest_sharpe"},
        holdout_training_spec=holdout_spec,
    )
    return study, lock


def test_one_public_operation_fits_and_finalizes_exact_locked_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    fit_calls = _install_fixture_adapter(monkeypatch, prices)

    execution = run_locked_holdout(lock)

    assert execution.lock.state == "HOLDOUT_EVALUATED"
    assert execution.training.hash == lock.record["holdout_training_hash"]
    assert execution.prediction.registry_record()["split"] == "holdout"
    assert fit_calls == [execution.training.hash]
    assert study.lifecycle.holdout_lineage(lock.hash) == {
        "lock_hash": lock.hash,
        "holdout_training_hash": execution.training.hash,
        "holdout_prediction_hash": execution.prediction.hash,
        "holdout_backtest_hash": execution.backtest.hash,
        "fitted_state_digest": execution.fitted_state_digest,
    }
    with pytest.raises(ValueError, match="LOCKED"):
        run_locked_holdout(lock)


def test_locked_execution_reopens_and_rejects_a_changed_lock_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, _ = _locked_study(tmp_path, monkeypatch)
    changed_record = deepcopy(lock.record)
    changed_record["checkpoint_kind"] = "epoch"
    changed = ResearchLock(study, lock.hash, lock.state, changed_record)

    with pytest.raises(ValueError, match="differs from its immutable registry record"):
        run_locked_holdout(changed)

    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    assert not (study.root / "run_log" / "training" / lock.record["holdout_training_hash"]).exists()


def test_locked_execution_ignores_model_and_strategy_preset_drift_and_fits_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, holdout_keys = _real_linear_lock(tmp_path, monkeypatch)
    from case_studies.research import strategy
    from case_studies.utils import backtest_runner, linear

    monkeypatch.setattr(
        linear,
        "_load_preset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preset was reread")),
    )
    monkeypatch.setattr(
        strategy,
        "get_backtest_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("strategy preset was reread")),
    )
    monkeypatch.setattr(
        backtest_runner,
        "get_backtest_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runner preset was reread")),
    )
    monkeypatch.setattr(
        backtest_runner,
        "ensure_backtest_spec",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("spec was re-resolved")),
    )

    execution = run_locked_holdout(lock)

    assert execution.lock.state == "HOLDOUT_EVALUATED"
    assert execution.prediction.coverage()["n_expected"] == holdout_keys.height
    model_dir = (
        execution.training.root / "run_log" / "training" / execution.training.hash / "models"
    )
    assert [path.name for path in model_dir.glob("fold_*.joblib")] == ["fold_2.joblib"]


def test_linear_adapter_reuses_validated_fit_after_finalization_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, lock, _ = _real_linear_lock(tmp_path, monkeypatch)
    from case_studies.research import lifecycle

    original = lifecycle.Lifecycle.finalize_holdout
    calls = 0

    def interrupt_once(self, lock_hash):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected finalization interruption")
        return original(self, lock_hash)

    monkeypatch.setattr(lifecycle.Lifecycle, "finalize_holdout", interrupt_once)
    with pytest.raises(RuntimeError, match="interruption"):
        run_locked_holdout(lock)
    model = (
        lock.study.root
        / "run_log"
        / "training"
        / lock.record["holdout_training_hash"]
        / "models"
        / "fold_2.joblib"
    )
    fitted_digest = hashlib.sha256(model.read_bytes()).hexdigest()
    fitted_mtime = model.stat().st_mtime_ns

    execution = run_locked_holdout(lock)

    assert execution.lock.state == "HOLDOUT_EVALUATED"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == fitted_digest
    assert model.stat().st_mtime_ns == fitted_mtime


def test_linear_adapter_adopts_exact_uncommitted_fold_after_process_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, _ = _real_linear_lock(tmp_path, monkeypatch)
    from case_studies.research import lifecycle

    original = lifecycle.Lifecycle.finalize_holdout
    monkeypatch.setattr(
        lifecycle.Lifecycle,
        "finalize_holdout",
        lambda self, lock_hash: (_ for _ in ()).throw(RuntimeError("interruption")),
    )
    with pytest.raises(RuntimeError, match="interruption"):
        run_locked_holdout(lock)
    model = (
        study.root
        / "run_log"
        / "training"
        / lock.record["holdout_training_hash"]
        / "models"
        / "fold_2.joblib"
    )
    model_digest = hashlib.sha256(model.read_bytes()).hexdigest()
    model_mtime = model.stat().st_mtime_ns
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "DELETE FROM candidate_fold_completions WHERE training_hash = ?",
            (lock.record["holdout_training_hash"],),
        )
        db.commit()
    monkeypatch.setattr(lifecycle.Lifecycle, "finalize_holdout", original)

    execution = run_locked_holdout(lock)

    assert execution.lock.state == "HOLDOUT_EVALUATED"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == model_digest
    assert model.stat().st_mtime_ns == model_mtime


def test_linear_adapter_preserves_conflicting_uncommitted_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, _ = _real_linear_lock(tmp_path, monkeypatch)
    from case_studies.research import lifecycle

    original = lifecycle.Lifecycle.finalize_holdout
    monkeypatch.setattr(
        lifecycle.Lifecycle,
        "finalize_holdout",
        lambda self, lock_hash: (_ for _ in ()).throw(RuntimeError("interruption")),
    )
    with pytest.raises(RuntimeError, match="interruption"):
        run_locked_holdout(lock)
    model = (
        study.root
        / "run_log"
        / "training"
        / lock.record["holdout_training_hash"]
        / "models"
        / "fold_2.joblib"
    )
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "DELETE FROM candidate_fold_completions WHERE training_hash = ?",
            (lock.record["holdout_training_hash"],),
        )
        db.commit()
    model.write_bytes(b"conflicting-uncommitted-fit")
    conflicting_bytes = model.read_bytes()
    monkeypatch.setattr(lifecycle.Lifecycle, "finalize_holdout", original)

    with pytest.raises(ValueError, match="conflicting uncommitted artifacts"):
        run_locked_holdout(lock)

    assert model.read_bytes() == conflicting_bytes
    assert study.lifecycle.open(lock.hash).state == "LOCKED"


def test_linear_adapter_rejects_locked_eligibility_mismatch_before_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, _ = _real_linear_lock(tmp_path, monkeypatch, eligibility_row_delta=1)

    with pytest.raises(ValueError, match="eligibility mismatch"):
        run_locked_holdout(lock)

    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    assert not (
        study.root / "run_log" / "training" / lock.record["holdout_training_hash"] / "models"
    ).exists()


def test_checkpointed_gbm_adapter_fits_once_and_publishes_only_locked_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock = _real_gbm_lock(tmp_path, monkeypatch)
    from case_studies.utils import registry
    from case_studies.utils.registry import registration

    def forbidden_delete(*args, **kwargs):
        raise AssertionError("canonical holdout execution cannot delete historical predictions")

    monkeypatch.setattr(registry, "clear_prediction_sets", forbidden_delete)
    monkeypatch.setattr(registration, "clear_prediction_sets", forbidden_delete)
    assert "config_name" not in lock.record["holdout_training_spec"]

    execution = run_locked_holdout(lock)

    assert execution.lock.state == "HOLDOUT_EVALUATED"
    assert (
        execution.prediction.registry_record()["checkpoint_kind"],
        execution.prediction.registry_record()["checkpoint_value"],
    ) == ("iteration", 2)
    holdout_rows = study.predictions.table().filter(pl.col("split") == "holdout")
    assert holdout_rows.select("checkpoint_kind", "checkpoint_value").rows() == [("iteration", 2)]


def test_interrupted_identical_request_reuses_fitted_state_and_staged_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    fit_calls = _install_fixture_adapter(monkeypatch, prices, interrupt_after_stage=True)

    with pytest.raises(RuntimeError, match="interruption"):
        run_locked_holdout(lock)
    assert study.lifecycle.open(lock.hash).state == "LOCKED"

    execution = run_locked_holdout(lock)

    assert execution.lock.state == "HOLDOUT_EVALUATED"
    assert fit_calls == [execution.training.hash]
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_staging").fetchone() == (1,)
        assert db.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone() == (1,)


def test_final_transaction_failure_keeps_lock_and_evaluation_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    _install_fixture_adapter(monkeypatch, prices)
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "CREATE TRIGGER reject_holdout_evaluation BEFORE UPDATE OF state ON research_locks "
            "WHEN NEW.state = 'HOLDOUT_EVALUATED' BEGIN "
            "SELECT RAISE(ABORT, 'injected final transition failure'); END"
        )
        db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="injected final transition failure"):
        run_locked_holdout(lock)

    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_staging").fetchone() == (1,)
        assert db.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone() == (0,)
        db.execute("DROP TRIGGER reject_holdout_evaluation")
        db.commit()

    assert run_locked_holdout(lock).lock.state == "HOLDOUT_EVALUATED"


def test_fitted_state_change_during_backtest_fails_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    _install_fixture_adapter(monkeypatch, prices)
    from case_studies.research.holdout import LockedStrategyReplay

    original_run = LockedStrategyReplay.run

    def run_then_change_fitted_state(self, prediction):
        result = original_run(self, prediction)
        model = (
            study.root
            / "run_log"
            / "training"
            / lock.record["holdout_training_hash"]
            / "models"
            / "fitted.bin"
        )
        model.write_bytes(b"changed-during-backtest")
        return result

    monkeypatch.setattr(LockedStrategyReplay, "run", run_then_change_fitted_state)

    with pytest.raises(ValueError, match="changed during holdout execution"):
        run_locked_holdout(lock)

    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_staging").fetchone() == (0,)


def test_interrupted_retry_rejects_changed_prediction_artifact_without_deleting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    _install_fixture_adapter(monkeypatch, prices, interrupt_after_stage=True)

    with pytest.raises(RuntimeError, match="interruption"):
        run_locked_holdout(lock)
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        prediction_hash = db.execute(
            "SELECT holdout_prediction_hash FROM holdout_staging WHERE lock_hash = ?",
            (lock.hash,),
        ).fetchone()[0]
    artifact = study.root / "run_log" / "predictions" / prediction_hash / "predictions.parquet"
    changed = pl.read_parquet(artifact).with_columns((pl.col("y_score") + 1.0).alias("y_score"))
    changed.write_parquet(artifact)
    changed_digest = value_digest(changed)

    with pytest.raises(ValueError, match="immutable prediction artifact conflict"):
        run_locked_holdout(lock)

    assert value_digest(pl.read_parquet(artifact)) == changed_digest
    assert study.lifecycle.open(lock.hash).state == "LOCKED"


def test_interrupted_retry_rejects_changed_fitted_state_without_replacing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, _ = _real_linear_lock(tmp_path, monkeypatch)
    from case_studies.research import lifecycle

    original = lifecycle.Lifecycle.finalize_holdout

    def interrupt(self, lock_hash):
        raise RuntimeError("injected finalization interruption")

    monkeypatch.setattr(lifecycle.Lifecycle, "finalize_holdout", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        run_locked_holdout(lock)
    model = (
        study.root
        / "run_log"
        / "training"
        / lock.record["holdout_training_hash"]
        / "models"
        / "fold_2.joblib"
    )
    model.write_bytes(b"changed-completed-fit")
    changed_digest = hashlib.sha256(model.read_bytes()).hexdigest()
    monkeypatch.setattr(lifecycle.Lifecycle, "finalize_holdout", original)

    with pytest.raises(ValueError, match="conflicting persisted artifacts"):
        run_locked_holdout(lock)

    assert hashlib.sha256(model.read_bytes()).hexdigest() == changed_digest
    assert study.lifecycle.open(lock.hash).state == "LOCKED"


def test_interrupted_retry_rejects_changed_backtest_artifact_without_deleting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    _install_fixture_adapter(monkeypatch, prices, interrupt_after_stage=True)

    with pytest.raises(RuntimeError, match="interruption"):
        run_locked_holdout(lock)
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        backtest_hash = db.execute(
            "SELECT holdout_backtest_hash FROM holdout_staging WHERE lock_hash = ?",
            (lock.hash,),
        ).fetchone()[0]
    artifact = study.root / "run_log" / "backtest" / backtest_hash / "daily_returns.parquet"
    changed = pl.read_parquet(artifact).with_columns(
        (pl.col("daily_return") + 0.01).alias("daily_return")
    )
    changed.write_parquet(artifact)
    changed_digest = value_digest(changed)

    with pytest.raises(ValueError, match="immutable backtest artifact conflict"):
        run_locked_holdout(lock)

    assert value_digest(pl.read_parquet(artifact)) == changed_digest
    assert study.lifecycle.open(lock.hash).state == "LOCKED"


def test_wrong_checkpoint_is_rejected_without_lifecycle_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    _install_fixture_adapter(monkeypatch, prices, published_checkpoint=("epoch", 1))

    with pytest.raises(ValueError, match="wrong holdout prediction"):
        run_locked_holdout(lock)

    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_staging").fetchone() == (0,)


def test_conflicting_staged_lineage_is_preserved_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, lock, prices = _locked_study(tmp_path, monkeypatch)
    _install_fixture_adapter(monkeypatch, prices, interrupt_after_stage=True)

    with pytest.raises(RuntimeError, match="interruption"):
        run_locked_holdout(lock)
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "UPDATE holdout_staging SET holdout_backtest_hash = ? WHERE lock_hash = ?",
            ("conflicting-backtest", lock.hash),
        )
        db.commit()

    with pytest.raises(ValueError, match="staged holdout lineage conflict"):
        run_locked_holdout(lock)

    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute(
            "SELECT holdout_backtest_hash FROM holdout_staging WHERE lock_hash = ?",
            (lock.hash,),
        ).fetchone() == ("conflicting-backtest",)


def test_missing_decision_holdout_transformation_fails_before_model_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    validation_spec = _resolved_spec()
    validation_spec["computation"]["checkpoint_schedule"] = [{"kind": "final", "value": None}]
    training = study.results.register_training(validation_spec)
    frame = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )
    positions = frame.select("symbol", "timestamp").with_columns(pl.Series("position", [1.0, -1.0]))
    decision = DecisionArtifact.publish(
        study,
        kind="target_positions",
        decisions=positions,
        prediction_hashes=[prediction.hash],
        parameters={"cadence": "1d"},
        source_identity={
            "module": "case_studies.research.decisions",
            "source_digest": "source-a",
            "declared_inputs": {"prediction_hashes": [prediction.hash]},
            "determinism": {"deterministic": True},
            "clean_replay_digest": value_digest(positions),
        },
        state_transition_policy=StateTransitionPolicy("liquidate", "reset"),
        canonical=True,
    )
    request = {
        "signal": {"method": "equal_weight_top_k", "top_k": 1},
        "execution_mode": "vectorized",
    }
    backtest = study.strategy(prediction=prediction, decision=decision, **request).run(
        prices=_prices()
    )
    candidates = CandidateSet.create(study, "decision-selection", [backtest])
    holdout_spec = deepcopy(validation_spec)
    holdout_spec["computation"]["cv"] = {"identity": "holdout", "split": "holdout"}
    lock = study.lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=backtest.hash,
        selection_evidence={"metric": "validation_backtest_sharpe"},
        holdout_training_spec=holdout_spec,
    )
    adapter_called = False

    def unexpected_adapter(*args, **kwargs):
        nonlocal adapter_called
        adapter_called = True
        raise AssertionError("model adapter must not run")

    from case_studies.research import models

    monkeypatch.setattr(models, "get_adapter", unexpected_adapter)

    with pytest.raises(ValueError, match="no reproducible holdout transformation"):
        run_locked_holdout(lock)

    assert not adapter_called
    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    assert not (study.root / "run_log" / "training" / lock.record["holdout_training_hash"]).exists()


def test_stateful_decision_is_recomputed_for_holdout_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    validation_spec = _resolved_spec()
    validation_spec["computation"]["checkpoint_schedule"] = [{"kind": "final", "value": None}]
    training = study.results.register_training(validation_spec)
    frame = _predictions().rename({"fold_id": "fold", "y_true": "actual", "y_score": "prediction"})
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold"),
    )
    positions = replay_ranked_positions(frame, universe=["A", "B"])
    source_digest = hashlib.sha256(inspect.getsource(replay_ranked_positions).encode()).hexdigest()
    decision = DecisionArtifact.publish(
        study,
        kind="target_positions",
        decisions=positions,
        prediction_hashes=[prediction.hash],
        parameters={"long_count": 1, "short_count": 1, "cadence": "1d"},
        source_identity={
            "module": __name__,
            "source_digest": source_digest,
            "declared_inputs": {
                "prediction_hashes": [prediction.hash],
                "universe": ["A", "B"],
            },
            "determinism": {"deterministic": True},
            "clean_replay_digest": value_digest(positions),
            "holdout_replay": {"version": 1, "function": "replay_ranked_positions"},
        },
        state_transition_policy=StateTransitionPolicy("liquidate", "reset"),
        canonical=True,
    )
    request = {
        "signal": {"method": "equal_weight_top_k", "top_k": 1},
        "execution_mode": "vectorized",
    }
    backtest = study.strategy(prediction=prediction, decision=decision, **request).run(
        prices=_prices()
    )
    candidates = CandidateSet.create(study, "stateful-selection", [backtest])
    holdout_spec = deepcopy(validation_spec)
    holdout_spec["computation"]["cv"] = {"identity": "holdout", "split": "holdout"}
    prices = _prices().with_columns(pl.lit(date(2024, 1, 11)).alias("timestamp"))
    _patch_holdout_prices(monkeypatch, prices)
    lock = study.lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=backtest.hash,
        selection_evidence={"metric": "validation_backtest_sharpe"},
        holdout_training_spec=holdout_spec,
    )
    _install_fixture_adapter(monkeypatch, prices)

    execution = run_locked_holdout(lock)

    holdout_spec_record = execution.backtest.spec()["decision_artifact"]
    assert holdout_spec_record["hash"] != decision.hash
    replayed = DecisionArtifact.open(study, holdout_spec_record["hash"])
    assert replayed.spec["prediction_hashes"] == [execution.prediction.hash]
    assert replayed.spec["state_transition_policy"] == {
        "fold_boundary": "liquidate",
        "temporal_gap": "reset",
    }
    assert replayed.spec["source_identity"]["declared_inputs"]["universe"] == ["A", "B"]


def _install_model_parameters(computation, by_fold) -> None:
    computation["model"]["effective_params_by_fold"] = by_fold


def _install_class_weights(computation, by_fold) -> None:
    computation["task"] = {
        "type": "classification",
        "imbalance": {"method": "balanced", "effective_class_weights_by_fold": by_fold},
    }


def _selection_for_fold_parameters(
    tmp_path: Path,
    validation_by_fold,
    install=_install_model_parameters,
):
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    validation_spec = _resolved_spec()
    validation_spec["computation"]["checkpoint_schedule"] = [{"kind": "final", "value": None}]
    install(validation_spec["computation"], validation_by_fold)
    training = study.results.register_training(validation_spec)
    frame = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )
    backtest = study.strategy(
        prediction=prediction,
        decision=None,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="vectorized",
    ).run(prices=_prices())
    candidates = CandidateSet.create(study, "fold-parameter-selection", [backtest])
    return study, candidates, backtest, validation_spec


def _lock_with_holdout_parameters(
    study,
    candidates,
    backtest,
    validation_spec,
    holdout_by_fold,
    install=_install_model_parameters,
):
    holdout_spec = deepcopy(validation_spec)
    holdout_spec["computation"]["cv"] = {"identity": "holdout-cv", "split": "holdout"}
    install(holdout_spec["computation"], holdout_by_fold)
    return study.lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=backtest.hash,
        selection_evidence={"metric": "validation_backtest_sharpe"},
        holdout_training_spec=holdout_spec,
    )


def test_lock_rejects_holdout_parameters_that_contradict_the_selected_training(
    tmp_path: Path,
) -> None:
    study, candidates, backtest, validation_spec = _selection_for_fold_parameters(
        tmp_path,
        {"0": {"alpha": 1.0}, "1": {"alpha": 1.0}},
    )

    with pytest.raises(ValueError, match=r"\{'alpha': 1000\.0\} != \{'alpha': 1\.0\}"):
        _lock_with_holdout_parameters(
            study, candidates, backtest, validation_spec, {"2": {"alpha": 1000.0}}
        )

    assert study.lifecycle.state == "DEVELOPMENT"


def test_lock_accepts_holdout_parameters_matching_a_fold_invariant_selection(
    tmp_path: Path,
) -> None:
    study, candidates, backtest, validation_spec = _selection_for_fold_parameters(
        tmp_path,
        {"0": {"alpha": 1.0}, "1": {"alpha": 1.0}},
    )

    lock = _lock_with_holdout_parameters(
        study, candidates, backtest, validation_spec, {"2": {"alpha": 1.0}}
    )

    assert lock.state == "LOCKED"


def test_lock_allows_holdout_parameters_the_validation_folds_prove_are_fold_derived(
    tmp_path: Path,
) -> None:
    study, candidates, backtest, validation_spec = _selection_for_fold_parameters(
        tmp_path,
        {"0": {"alpha": 1.0}, "1": {"alpha": 2.0}},
    )

    lock = _lock_with_holdout_parameters(
        study, candidates, backtest, validation_spec, {"2": {"alpha": 3.0}}
    )

    assert lock.state == "LOCKED"


def test_completeness_survives_registry_rows_written_before_artifact_digests(
    tmp_path: Path,
) -> None:
    from case_studies.research.results import Result

    study, candidates, backtest, _ = _selection_for_fold_parameters(tmp_path, {"0": {"alpha": 1.0}})
    assert Result.open(study, backtest.hash).complete

    # Reproduce a registry that predates both columns. A freshly created table
    # declares artifact_digest NOT NULL, but _migrate_registry adds it to an existing
    # table with ALTER TABLE ... ADD COLUMN, which is nullable and unbackfilled, so
    # every row such a registry already held carries NULL.
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        ddl = db.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'prediction_coverage'"
        ).fetchone()[0]
        migrated_ddl = ddl.replace("artifact_digest      TEXT NOT NULL", "artifact_digest TEXT")
        assert migrated_ddl != ddl, "prediction_coverage DDL no longer pins artifact_digest"
        db.execute("ALTER TABLE prediction_coverage RENAME TO prediction_coverage_pinned")
        db.execute(migrated_ddl)
        db.execute("INSERT INTO prediction_coverage SELECT * FROM prediction_coverage_pinned")
        db.execute("DROP TABLE prediction_coverage_pinned")
        db.execute("UPDATE prediction_coverage SET artifact_digest = NULL")
        db.execute("UPDATE backtest_runs SET artifact_digests_json = NULL")

    legacy = Result.open(study, backtest.hash)
    assert legacy.complete
    assert CandidateSet.create(study, "legacy-selection", [legacy]).members == (legacy.hash,)


def test_lock_accepts_list_valued_class_weights_matching_the_selected_training(
    tmp_path: Path,
) -> None:
    study, candidates, backtest, validation_spec = _selection_for_fold_parameters(
        tmp_path, {"0": [0.6, 3.0], "1": [0.6, 3.0]}, install=_install_class_weights
    )

    lock = _lock_with_holdout_parameters(
        study,
        candidates,
        backtest,
        validation_spec,
        {"2": [0.6, 3.0]},
        install=_install_class_weights,
    )

    assert lock.state == "LOCKED"


def test_lock_rejects_list_valued_class_weights_contradicting_the_selected_training(
    tmp_path: Path,
) -> None:
    study, candidates, backtest, validation_spec = _selection_for_fold_parameters(
        tmp_path, {"0": [0.6, 3.0], "1": [0.6, 3.0]}, install=_install_class_weights
    )

    with pytest.raises(ValueError, match=r"\[9\.9, 0\.1\] != \[0\.6, 3\.0\]"):
        _lock_with_holdout_parameters(
            study,
            candidates,
            backtest,
            validation_spec,
            {"2": [9.9, 0.1]},
            install=_install_class_weights,
        )

    assert study.lifecycle.state == "DEVELOPMENT"


def _darts_sequence_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cadence=None):
    """Resolve a canonical Darts request the way the case study would."""
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    dates = [f"2024-01-{day:02d}" for day in range(2, 13)]
    frame = pl.DataFrame(
        {
            "symbol": [f"S{symbol}" for symbol in range(3) for _ in dates],
            "timestamp": dates * 3,
            "feature": [float(index) for index in range(3 * len(dates))],
            "fwd_ret_1d": [float(index % 3) / 100 for index in range(3 * len(dates))],
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    label = study.labels.publish(
        LabelDefinition("fwd_ret_1d", "regression", "1D"),
        frame.select("symbol", "timestamp", "fwd_ret_1d"),
    )
    mds = SimpleNamespace(
        dataset=frame,
        feature_names=["feature"],
        label_col="fwd_ret_1d",
        date_col="timestamp",
        entity_cols=["symbol"],
        splits=[
            {
                "fold": 0,
                "train_start": "2024-01-02",
                "train_end": "2024-01-10",
                "val_start": "2024-01-11",
                "val_end": "2024-01-12",
            }
        ],
        task_type="regression",
        class_values=[],
        temporal_by_fold=None,
        temporal_keys=[],
        temporal_feature_names=[],
        input_lineage={
            "artifacts": {"financial": {"sha256": "features-v1", "size": 1}},
            "fingerprint": "fixture-v1",
        },
    )
    monkeypatch.setattr("utils.modeling.load_modeling_dataset", lambda *a, **k: mds)
    # The etfs fixture resolves no calendar_id, which a real case study always has and
    # the locked reconstruction requires.
    monkeypatch.setattr(
        "utils.cv_splits.make_walk_forward_config",
        lambda *a, **k: SimpleNamespace(calendar_id="crypto-24x7"),
    )
    monkeypatch.setattr(
        "utils.modeling.load_configs",
        lambda *a, **k: [
            {
                "batch_size": 8,
                "checkpoint_interval": 1,
                "n_epochs": 1,
                "params": {
                    "architecture": "tsmixer",
                    "lookback": 2,
                    **({"decision_cadence": cadence} if cadence else {}),
                },
                "config_name": "tsmixer_probe",
                "family": "deep_learning",
                "library": "darts",
            }
        ],
    )
    expected_keys = (
        frame.filter(pl.col("timestamp") >= pl.date(2024, 1, 11))
        .select("symbol", "timestamp")
        .with_columns(pl.lit(0, dtype=pl.Int64).alias("fold"))
    )
    monkeypatch.setattr(
        "case_studies.utils.darts_forecasting.darts_validation_keys",
        lambda *a, **k: expected_keys,
    )
    monkeypatch.setattr(
        "case_studies.utils.darts_forecasting.darts_training_identity",
        lambda *a, **k: {
            "base_target_data_spec": {"kind": "one_period_return", "source": "close"},
            "input_data_spec": mds.input_lineage,
            "input_chunk_length": 2,
            "output_chunk_length": 1,
            "max_train_sequences": 0,
        },
    )
    resolved = study.model(
        family="deep_learning",
        label=label.name,
        config_name="tsmixer_probe",
        overrides={"device": "cpu"},
    ).resolve()
    return study, resolved.spec


def test_locked_sequence_reconstruction_round_trips_its_own_resolved_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reconstruction must accept exactly what the resolver produced.

    The Darts preprocessing block is the part that drifted: it records the resolved
    base-target spec and the decision cadence, and the reconstruction expected a
    hardcoded string and omitted the cadence key, so no Darts model could reconstruct.
    """
    from case_studies.utils import deep_learning

    study, spec = _darts_sequence_spec(tmp_path, monkeypatch)
    holdout_spec = deepcopy(spec)
    holdout_spec["computation"]["cv"] = {
        "identity": "holdout-cv",
        "split": "holdout",
        "train_start": "2024-01-02",
        "train_end": "2024-01-10",
        "evaluation_start": "2024-01-11",
        "evaluation_end": "2024-01-12",
    }

    request = deep_learning.reconstruct_locked_request(
        study,
        holdout_spec,
        checkpoint_kind="epoch",
        checkpoint_value=1,
    )

    assert request._context.prediction_split == "holdout"
    assert request._context.published_checkpoints == (1,)
    assert request._context.config["params"]["architecture"] == "tsmixer"

    # The accepting call above is only evidence if a spec the resolver could not have
    # produced is rejected. Each of these is what the reconstruction used to expect.
    for mutation in (
        {"base_target": "one_period_log_return"},
        {"decision_cadence": "weekly_friday"},
    ):
        drifted = deepcopy(holdout_spec)
        drifted["computation"]["preprocessing"].update(mutation)
        with pytest.raises(ValueError, match="preprocessing does not match"):
            deep_learning.reconstruct_locked_request(
                study, drifted, checkpoint_kind="epoch", checkpoint_value=1
            )

    dropped = deepcopy(holdout_spec)
    del dropped["computation"]["preprocessing"]["decision_cadence"]
    with pytest.raises(ValueError, match="preprocessing does not match"):
        deep_learning.reconstruct_locked_request(
            study, dropped, checkpoint_kind="epoch", checkpoint_value=1
        )


def _holdout_cv_block(**extra):
    return {
        "identity": "holdout-cv",
        "split": "holdout",
        "train_start": "2024-01-02",
        "train_end": "2024-01-10",
        "evaluation_start": "2024-01-11",
        "evaluation_end": "2024-01-12",
        **extra,
    }


def test_cadence_selected_lock_without_a_recorded_calendar_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from case_studies.utils import deep_learning

    study, spec = _darts_sequence_spec(tmp_path, monkeypatch, cadence="weekly_friday")
    holdout_spec = deepcopy(spec)
    holdout_spec["computation"]["cv"] = _holdout_cv_block()

    with pytest.raises(ValueError, match="does not record its calendar"):
        deep_learning.reconstruct_locked_request(
            study, holdout_spec, checkpoint_kind="epoch", checkpoint_value=1
        )


@pytest.mark.parametrize("calendar", [None, "NYSE"])
def test_cadence_selected_lock_reconstructs_from_a_recorded_calendar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calendar
) -> None:
    """A recorded calendar of None is a reproducible choice, not a missing one.

    CVSpec.calendar is None whenever it was left unset and the resolved record still
    writes the key, so a canonical weekly run that never named a calendar must
    reconstruct rather than be refused.
    """
    from case_studies.utils import deep_learning

    study, spec = _darts_sequence_spec(tmp_path, monkeypatch, cadence="weekly_friday")
    holdout_spec = deepcopy(spec)
    holdout_spec["computation"]["cv"] = _holdout_cv_block(calendar=calendar)

    request = deep_learning.reconstruct_locked_request(
        study, holdout_spec, checkpoint_kind="epoch", checkpoint_value=1
    )

    # Thinned to one observation per week, so the other nine fixture days are gone.
    # The fixture runs over consecutive calendar days, so the first week ends on
    # Sunday 2024-01-07 rather than on a Friday.
    observed = sorted(request._context.dataset_pd["timestamp"].unique())
    assert [stamp.strftime("%Y-%m-%d") for stamp in observed] == ["2024-01-07", "2024-01-12"]
    assert request._context.config["params"]["decision_cadence"] == "weekly_friday"


def test_cadence_selected_lock_accepts_the_calendar_from_a_nested_cv_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from case_studies.utils import deep_learning

    study, spec = _darts_sequence_spec(tmp_path, monkeypatch, cadence="weekly_friday")
    holdout_spec = deepcopy(spec)
    holdout_spec["computation"]["cv"] = _holdout_cv_block(request={"calendar": None})

    request = deep_learning.reconstruct_locked_request(
        study, holdout_spec, checkpoint_kind="epoch", checkpoint_value=1
    )

    assert request._context.prediction_split == "holdout"


def test_digest_verification_is_not_served_stale_after_an_artifact_changes(
    tmp_path: Path,
) -> None:
    """Completeness memoizes the digest read; a replaced artifact must still be caught.

    The cache key carries the artifact's size and nanosecond mtime precisely so that a
    rewritten file misses it. Without that, a population scan that had already read an
    artifact would keep reporting it complete after it was replaced.
    """
    from case_studies.research.results import Result

    study, _candidates, backtest, _spec = _selection_for_fold_parameters(
        tmp_path, {"0": {"alpha": 1.0}}
    )
    assert Result.open(study, backtest.hash).complete

    returns = study.root / "run_log" / "backtest" / backtest.hash / "daily_returns.parquet"
    original = pl.read_parquet(returns)
    original.with_columns(pl.col(original.columns[-1]) * 2.0).write_parquet(returns)

    assert not Result.open(study, backtest.hash).complete


def test_resolved_spec_only_requires_the_rebalance_thresholds_it_no_longer_fills(
    tmp_path: Path,
) -> None:
    """Skipping ensure_backtest_spec means the caller must supply what it would have.

    Without this the omission surfaces as a bare KeyError inside the rebalance logic,
    far from the spec that caused it.
    """
    from case_studies.utils.backtest_runner import run_backtest

    study, _candidates, backtest, _spec = _selection_for_fold_parameters(
        tmp_path, {"0": {"alpha": 1.0}}
    )
    resolved = deepcopy(backtest.spec())
    prediction_hash = backtest.registry_record()["prediction_hash"]
    resolved["backtest_config"]["metadata"]["prediction_hash"] = prediction_hash
    resolved["strategy"]["rebalance"].pop("min_trade_value")

    with pytest.raises(ValueError, match=r"omits rebalance \['min_trade_value'\]"):
        run_backtest(
            study.case_study,
            prediction_hash,
            resolved,
            prices=_prices(),
            predictions=_predictions(),
            resolved_spec_only=True,
        )


def test_locked_reconstruction_leaves_linear_training_identity_unmoved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This branch must not move a linear training identity the thread pin just fixed.

    #548 made numerics.thread_limit identity-bearing for linear, so every linear lock
    was re-taken against those identities. The locked reconstruction added here reads
    the resolved spec and must not contribute to it.
    """
    from case_studies.utils import linear
    from case_studies.utils.artifact_digest import value_digest

    study = _linear_study(tmp_path, monkeypatch)
    request = study.model(family="linear", label="fwd_ret_1d", config_name="ridge")
    resolved = request.resolve()
    computation = resolved.spec["computation"]

    assert computation["numerics"]["thread_limit"] == linear.LINEAR_THREAD_LIMIT
    assert linear.training_hash_from_spec(request.resolve().spec) == linear.training_hash_from_spec(
        resolved.spec
    )

    holdout_spec = deepcopy(resolved.spec)
    holdout_spec["computation"]["cv"] = {"identity": "holdout-cv", "split": "holdout"}
    assert linear.training_hash_from_spec(holdout_spec) != linear.training_hash_from_spec(
        resolved.spec
    )

    # Why the eligibility digest is taken over a fixed key tuple rather than over the
    # frame's columns: the fixed tuple is insensitive to a builder gaining a column,
    # so it cannot move an identity that is already registered. Deriving the tuple from
    # the frame makes every added column a silent identity change.
    keys = pl.DataFrame({"symbol": ["ES"], "timestamp": [date(2024, 1, 2)], "fold": [0]})
    widened = keys.with_columns(pl.lit(1.0).alias("weight"))
    fixed = ("symbol", "timestamp", "fold")
    assert value_digest(widened, fixed) == value_digest(keys, fixed)
    assert value_digest(widened, tuple(widened.columns)) != value_digest(keys, fixed)


def test_tabm_checkpoint_ic_survives_product_entities() -> None:
    """The checkpoint IC must actually compute for a product-keyed study.

    _train_tabm_fold builds its IC frame with the literal `symbol` key. When the
    cross_sectional_ic call was left passing the reader key, the join raised
    ColumnNotFoundError - and run_tabm_cv catches Exception per config, so a fresh
    cme_futures run reported every config unavailable and skipped all training
    rather than failing. Nothing else exercises this: the runtime suites monkeypatch
    _train_tabm_fold, so the IC call never runs, and none drives a product key.

    The fixture gives each timestamp min_obs entities, so a NaN IC is a failure
    rather than the expected shape of a too-thin cross-section.
    """
    import numpy as np
    import torch
    from torch import nn

    from case_studies.utils import tabular_dl

    products = ["ES", "NQ", "CL", "GC", "ZN", "6E"]
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    rows, features = len(products) * len(dates), 3
    rng = np.random.default_rng(0)
    val_entities = np.array([product for _ in dates for product in products])
    val_dates = np.array([np.datetime64(day) for day in dates for _ in products])

    class _Probe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(features, 1)

        def forward(self, batch):
            return self.linear(batch).squeeze(-1)

    ics, preds, losses = tabular_dl._train_tabm_fold(
        model=_Probe(),
        X_train=rng.normal(size=(rows, features)).astype(np.float32),
        y_train=rng.normal(size=rows).astype(np.float32),
        X_val=rng.normal(size=(rows, features)).astype(np.float32),
        y_val=rng.normal(size=rows).astype(np.float32),
        y_eval_val=rng.normal(size=rows).astype(np.float32),
        val_dates=val_dates,
        val_entities=val_entities,
        n_epochs=1,
        batch_size=8,
        checkpoint_interval=1,
        device=torch.device("cpu"),
    )

    assert set(ics) == {1}
    assert np.isfinite(ics[1]), "product-keyed cross-section produced no usable IC"
    assert preds[1].shape == (rows,)
    assert np.isfinite(losses[1])
