from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import CandidateSet, PredictionResult, Result, Strategy, Study
from tests.test_research_registry import _predictions, _training_spec
from tests.test_research_workspace import _seed_release


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": ["2024-01-05", "2024-01-05"],
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.5, 99.5],
            "volume": [1_000, 1_000],
        }
    ).with_columns(pl.col("timestamp").str.to_date())


def _patch_holdout_prices(monkeypatch: pytest.MonkeyPatch, prices: pl.DataFrame) -> list[int]:
    from case_studies.research import lifecycle, strategy

    warmups: list[int] = []

    def load_prices(
        case_study: str,
        label: str,
        *,
        split: str,
        warmup_periods: int = 0,
    ):
        assert (case_study, label, split) == ("etfs", "fwd_ret_21d", "holdout")
        warmups.append(warmup_periods)
        return prices

    monkeypatch.setattr(lifecycle, "load_backtest_prices_for", load_prices)
    monkeypatch.setattr(strategy, "load_backtest_prices_for", load_prices)
    return warmups


def _publish_validation_prediction(study: Study) -> PredictionResult:
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    return study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )


def test_fake_prediction_to_backtest_flow_survives_restart(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    training = study.results.register_training(_training_spec())
    predictions = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=predictions,
        expected_keys=predictions.select("symbol", "timestamp", "fold_id"),
    )
    direct = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="vectorized",
    )
    first = direct.run(prices=_prices())

    reopened_study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    reopened_prediction = Result.open(reopened_study, prediction.hash)
    notebook_request = {
        "prediction": reopened_prediction,
        "signal": {"method": "equal_weight_top_k", "top_k": 1},
        "execution_mode": "vectorized",
    }
    notebook_style = Strategy.from_request(reopened_study, notebook_request)
    second = notebook_style.run(prices=_prices())

    assert reopened_prediction.hash == prediction.hash
    assert direct.identity(prices=_prices()) == notebook_style.identity(prices=_prices())
    assert first.hash == second.hash


def test_strategy_identity_covers_prices_costs_and_rejects_unknown_fields(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
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
    base = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="vectorized",
    )
    changed_costs = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        costs={"commission_bps": 5.0, "slippage_bps": 2.0},
        execution_mode="vectorized",
    )
    changed_prices = _prices().with_columns((pl.col("close") + 1).alias("close"))

    assert base.identity(prices=_prices()) != changed_costs.identity(prices=_prices())
    assert base.identity(prices=_prices()) != base.identity(prices=changed_prices)
    with pytest.raises(ValueError, match="unsupported"):
        study.strategy(
            prediction=prediction,
            signal={"method": "equal_weight_top_k", "top_k": 1},
            typo=True,
        )


def test_strategy_normalizes_conformal_identity_before_hashing(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
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
    strategy = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        allocation={"method": "conformal_weighted", "alpha": 0.2},
        execution_mode="vectorized",
    )

    spec = strategy.resolve(prices=_prices())
    allocation = spec["strategy"]["allocation"]

    assert allocation["calibration_version"] == "walk_forward_v2"
    assert allocation["min_calibration_n"] == 30
    assert allocation["sparse_fallback"] == "pooled_prior_oos"


def test_strategy_default_lookback_ignores_larger_unrelated_sweep_variant(
    tmp_path: Path,
) -> None:
    from case_studies.utils.backtest_loaders import warmup_periods_for

    release = _seed_release(tmp_path)
    setup_path = release / "case_studies" / "etfs" / "config" / "setup.yaml"
    setup_path.write_text(
        setup_path.read_text()
        + "execution:\n"
        + "  allocator_lookback: 3\n"
        + "backtest:\n"
        + "  sweep:\n"
        + "    allocators:\n"
        + "      - method: risk_parity\n"
        + "        vol_window: 99\n"
    )
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    prediction = _publish_validation_prediction(study)
    strategy = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        allocation={"method": "inverse_vol"},
        execution_mode="vectorized",
    )

    spec = strategy.resolve(prices=_prices())

    assert warmup_periods_for("etfs") == 99
    assert spec["strategy"]["allocation"]["vol_window"] == 3


def test_strategy_preserves_explicit_lookback_alias_without_injecting_vol_window(
    tmp_path: Path,
) -> None:
    from case_studies.research.strategy import strategy_warmup_periods

    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    prediction = _publish_validation_prediction(study)
    strategy = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        allocation={"method": "inverse_vol", "lookback": 7},
        execution_mode="vectorized",
    )

    spec = strategy.resolve(prices=_prices())
    allocation = spec["strategy"]["allocation"]

    assert allocation["lookback"] == 7
    assert "vol_window" not in allocation
    assert strategy_warmup_periods(spec) == 7


def test_preview_prediction_to_backtest_flow_stays_isolated(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    training = study.results.register_training(
        _training_spec(
            execution_tier="preview",
            preview_reductions={"folds": [0], "max_rows": 2},
        ),
        execution_tier="preview",
    )
    frame = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )
    preview_backtest = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="vectorized",
    ).run(prices=_prices())

    canonical_db = study.root / "run_log" / "registry.db"
    with closing(sqlite3.connect(canonical_db)) as db:
        assert db.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0] == 0
    assert preview_backtest.execution_tier == "preview"
    assert Result.open(study, preview_backtest.hash, include_preview=True).complete
    with pytest.raises(KeyError):
        Result.open(study, preview_backtest.hash)


def test_lock_transition_failure_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
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
    backtest = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="vectorized",
    ).run(prices=_prices())
    candidates = CandidateSet.create(study, "selection", [backtest])
    assert candidates.best_validation_sharpe().hash == backtest.hash
    _patch_holdout_prices(
        monkeypatch,
        _prices().with_columns(pl.lit(date(2024, 1, 11)).alias("timestamp")),
    )
    invalid_holdout_spec = _training_spec(
        cv={"phase": "holdout", "train_end": "2024-01-04"},
        model={"class": "Ridge", "params": {"alpha": 2.0}},
    )
    with pytest.raises(ValueError, match="only in CV interval"):
        study.lifecycle.lock(
            candidate_set_hash=candidates.hash,
            selected_backtest_hash=backtest.hash,
            selection_evidence={"metric": "validation_backtest_sharpe"},
            holdout_training_spec=invalid_holdout_spec,
        )
    with closing(sqlite3.connect(study.root / "run_log" / "registry.db")) as db:
        assert db.execute("SELECT COUNT(*) FROM research_locks").fetchone()[0] == 0

    lock = study.lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=backtest.hash,
        selection_evidence={"metric": "validation_backtest_sharpe"},
        holdout_training_spec=_training_spec(cv={"phase": "holdout", "train_end": "2024-01-04"}),
    )

    with pytest.raises(ValueError, match="exact complete canonical lineage"):
        study.lifecycle.record_holdout(
            lock.hash,
            holdout_training_hash=training.hash,
            holdout_prediction_hash=prediction.hash,
            holdout_backtest_hash=backtest.hash,
        )

    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    with closing(sqlite3.connect(study.root / "run_log" / "registry.db")) as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone()[0] == 0


def test_locked_rolling_allocator_holdout_preserves_warmup_and_transitions_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    validation_training = study.results.register_training(_training_spec())
    validation_prediction = study.results.publish_predictions(
        validation_training,
        checkpoint_kind="epoch",
        checkpoint_value=3,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )
    request = {
        "signal": {"method": "equal_weight_top_k", "top_k": 1},
        "allocation": {"method": "inverse_vol", "vol_window": 2},
        "execution_mode": "vectorized",
    }
    validation_backtest = study.strategy(
        prediction=validation_prediction,
        **request,
    ).run(prices=_prices())
    candidates = CandidateSet.create(study, "selection", [validation_backtest])
    holdout_spec = _training_spec(cv={"phase": "holdout", "train_end": "2024-01-04"})
    holdout_prices = pl.concat(
        [
            _prices().with_columns(
                pl.lit(date(2024, 1, day)).alias("timestamp"),
                (pl.col("close") + day / 10).alias("close"),
            )
            for day in range(8, 12)
        ]
    )
    warmups = _patch_holdout_prices(monkeypatch, holdout_prices)
    lock = study.lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=validation_backtest.hash,
        selection_evidence={"metric": "validation_backtest_sharpe"},
        holdout_training_spec=holdout_spec,
    )
    holdout_training = study.results.register_training(holdout_spec)
    holdout_frame = frame.with_columns(pl.lit(date(2024, 1, 11)).alias("timestamp"))
    holdout_expected = holdout_frame.select("symbol", "timestamp", "fold_id")
    wrong_checkpoint = study.results.publish_predictions(
        holdout_training,
        checkpoint_kind="epoch",
        checkpoint_value=4,
        split="holdout",
        predictions=holdout_frame,
        expected_keys=holdout_expected,
    )
    with pytest.raises(ValueError, match="locked retraining contract"):
        study.strategy(prediction=wrong_checkpoint, **request)

    holdout_prediction = study.results.publish_predictions(
        holdout_training,
        checkpoint_kind="epoch",
        checkpoint_value=3,
        split="holdout",
        predictions=holdout_frame,
        expected_keys=holdout_expected,
    )
    holdout_backtest = study.strategy(
        prediction=holdout_prediction,
        **request,
    ).run()

    with pytest.raises(ValueError, match="canonical holdout prices"):
        study.strategy(prediction=holdout_prediction, **request).run(prices=holdout_prices)
    with pytest.raises(ValueError, match="locked validation strategy"):
        study.strategy(
            prediction=holdout_prediction,
            signal={"method": "equal_weight_top_k", "top_k": 2},
            allocation={"method": "inverse_vol", "vol_window": 2},
            execution_mode="vectorized",
        ).run()

    from case_studies.research import lifecycle

    changed_prices = holdout_prices.with_columns((pl.col("close") + 1).alias("close"))
    monkeypatch.setattr(
        lifecycle,
        "load_backtest_prices_for",
        lambda case_study, label, *, split, warmup_periods=0: changed_prices,
    )
    with pytest.raises(ValueError, match="exact complete canonical lineage"):
        study.lifecycle.record_holdout(
            lock.hash,
            holdout_training_hash=holdout_training.hash,
            holdout_prediction_hash=holdout_prediction.hash,
            holdout_backtest_hash=holdout_backtest.hash,
        )
    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    with closing(sqlite3.connect(study.root / "run_log" / "registry.db")) as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone()[0] == 0
    restored_warmups = _patch_holdout_prices(monkeypatch, holdout_prices)

    evaluated = study.lifecycle.record_holdout(
        lock.hash,
        holdout_training_hash=holdout_training.hash,
        holdout_prediction_hash=holdout_prediction.hash,
        holdout_backtest_hash=holdout_backtest.hash,
    )

    assert evaluated.state == "HOLDOUT_EVALUATED"
    assert warmups + restored_warmups == [2, 2, 2, 2]
    with pytest.raises(ValueError, match="LOCKED"):
        study.lifecycle.record_holdout(
            lock.hash,
            holdout_training_hash=holdout_training.hash,
            holdout_prediction_hash=holdout_prediction.hash,
            holdout_backtest_hash=holdout_backtest.hash,
        )


def test_locked_conformal_holdout_uses_validation_residuals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    timestamps = pl.date_range(date(2023, 12, 18), date(2024, 1, 9), eager=True)
    validation = pl.DataFrame(
        {
            "timestamp": [value for value in timestamps for _ in range(2)],
            "symbol": [symbol for _ in timestamps for symbol in ("A", "B")],
            "fold_id": [
                fold for index in range(len(timestamps)) for fold in ([int(index >= 15)] * 2)
            ],
            "y_true": [0.01, -0.02] * len(timestamps),
            "y_score": [0.02, -0.01] * len(timestamps),
        }
    )
    prices = pl.DataFrame(
        {
            "timestamp": [value for value in timestamps for _ in range(2)],
            "symbol": [symbol for _ in timestamps for symbol in ("A", "B")],
            "close": [
                100.0 + index if symbol == "A" else 100.0 - 0.3 * index
                for index in range(len(timestamps))
                for symbol in ("A", "B")
            ],
            "volume": [1_000] * (2 * len(timestamps)),
        }
    ).with_columns(
        open=pl.col("close"),
        high=pl.col("close") + 1,
        low=pl.col("close") - 1,
    )
    training = study.results.register_training(_training_spec())
    validation_prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=validation,
        expected_keys=validation.select("symbol", "timestamp", "fold_id"),
    )
    request = {
        "signal": {"method": "equal_weight_top_k", "top_k": 1},
        "allocation": {
            "method": "conformal_weighted",
            "alpha": 0.2,
            "min_calibration_n": 1,
        },
        "execution_mode": "vectorized",
    }
    validation_backtest = study.strategy(
        prediction=validation_prediction,
        **request,
    ).run(prices=prices)
    holdout_prices = _prices().with_columns(pl.lit(date(2024, 1, 11)).alias("timestamp"))
    _patch_holdout_prices(monkeypatch, holdout_prices)
    candidates = CandidateSet.create(study, "conformal-selection", [validation_backtest])
    holdout_spec = _training_spec(cv={"phase": "holdout", "train_end": "2024-01-09"})
    lock = study.lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=validation_backtest.hash,
        selection_evidence={"metric": "validation_backtest_sharpe"},
        holdout_training_spec=holdout_spec,
    )
    holdout_training = study.results.register_training(holdout_spec)
    holdout_frame = _predictions().with_columns(pl.lit(date(2024, 1, 11)).alias("timestamp"))
    holdout_prediction = study.results.publish_predictions(
        holdout_training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="holdout",
        predictions=holdout_frame,
        expected_keys=holdout_frame.select("symbol", "timestamp", "fold_id"),
    )

    holdout_backtest = study.strategy(prediction=holdout_prediction, **request).run()
    widths = pl.read_parquet(
        study.root
        / "run_log"
        / "predictions"
        / holdout_prediction.hash
        / "conformal_widths.parquet"
    )
    evaluated = study.lifecycle.record_holdout(
        lock.hash,
        holdout_training_hash=holdout_training.hash,
        holdout_prediction_hash=holdout_prediction.hash,
        holdout_backtest_hash=holdout_backtest.hash,
    )

    assert widths.get_column("fold_id").unique().to_list() == [-1]
    assert widths.get_column("calibration_n").min() == 2
    assert evaluated.state == "HOLDOUT_EVALUATED"
