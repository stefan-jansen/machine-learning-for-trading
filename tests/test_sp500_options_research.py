from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from case_studies.research import (
    DecisionArtifact,
    OfficialPopulation,
    ResolvedModelRequest,
    Study,
)
from case_studies.sp500_options._htm_backtest import (
    _apply_cohort_allocator,
    _compute_cohort_daily_pnl,
    _load_option_lifecycle,
    _select_cohorts,
    option_source_identity,
    run_htm_daily_mtm,
)
from case_studies.sp500_options.research_workflow import (
    model_request_catalog,
    open_study,
    option_decision_dates,
    option_trade_calendar,
    paired_sharpe_on_common_support,
    publish_short_straddle_decisions,
    resolved_model_plan,
    run_official_backtest_requests,
    strategy_request_frame,
)
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.registry.store import _open_registry
from scripts.prove_sp500_options_interface import _seed_real_preview_prediction
from tests.test_research_contract_catalog import _resolved_spec
from utils.paths import REPO_ROOT


def _study(tmp_path: Path) -> Study:
    output_root = tmp_path / "workspace"
    root = output_root / "sp500_options"
    root.mkdir(parents=True)
    (root / "config").symlink_to(
        REPO_ROOT / "case_studies" / "sp500_options" / "config",
        target_is_directory=True,
    )
    (output_root / "config").symlink_to(
        REPO_ROOT / "case_studies" / "config",
        target_is_directory=True,
    )
    _open_registry(root).close()
    return Study(
        case_study="sp500_options",
        root=root,
        release_root=tmp_path / "release",
        output_root=output_root,
        read_only=False,
        manifest={"schema_version": 1, "case_study": "sp500_options"},
    )


def _prediction(
    study: Study,
    *,
    execution_tier: str = "canonical",
    alpha: float = 1.0,
):
    spec = _resolved_spec(alpha=alpha)
    spec["label"] = "ret_to_expiry"
    spec["execution_tier"] = execution_tier
    if execution_tier == "preview":
        spec["computation"]["preview_reductions"] = {"test_rows": 1}
        study.activate("preview")
    training = study.results.register_training(spec, execution_tier=execution_tier)
    frame = pl.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": [datetime(2024, 1, 5)],
            "fold": [0],
            "actual": [0.01],
            "prediction": [0.02],
        }
    )
    return study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold"),
    )


def test_resolved_model_plan_accepts_flat_sequence_specs(tmp_path: Path) -> None:
    study = _study(tmp_path)
    expected = pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": [datetime(2024, 1, 5), datetime(2024, 1, 5)],
            "fold": [0, 0],
        }
    )
    spec = {
        "identity_version": 2,
        "execution_tier": "preview",
        "family": "deep_learning",
        "label": "ret_to_expiry",
        "seed": 42,
        "config_name": "nlinear",
        "task": {"type": "regression", "class_values": []},
        "feature_names": ["feature_a", "feature_b"],
        "checkpoint_schedule": [{"kind": "epoch", "value": 5}],
        "preview_reductions": {"max_symbols": 2},
    }
    request = ResolvedModelRequest(
        study,
        "deep_learning",
        spec,
        SimpleNamespace(expected_keys=expected),
    )

    plan = resolved_model_plan((request,))

    assert plan.select(
        "family", "config_name", "feature_count", "eligible_entities", "checkpoints"
    ).row(0) == ("deep_learning", "nlinear", 2, 2, 1)


def test_preview_study_activates_before_model_catalog_resolution(tmp_path: Path) -> None:
    study = open_study(execution_tier="preview", workspace=tmp_path)

    catalog = model_request_catalog("linear", config_names=["ridge_a1.0"])

    assert study.output_root == tmp_path
    assert catalog.to_dicts() == [
        {"family": "linear", "label": "ret_to_expiry", "config_name": "ridge_a1.0"}
    ]


def _contract_returns() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature_date": [date(2024, 1, 5)],
            "symbol": ["A"],
            "strike": [100.0],
            "expiration": [date(2024, 1, 10)],
            "entry_date": [date(2024, 1, 8)],
            "entry_straddle_mid": [10.0],
            "entry_call_mid": [6.0],
            "entry_call_bid": [5.5],
            "entry_call_ask": [6.5],
            "entry_put_mid": [4.0],
            "entry_put_bid": [3.5],
            "entry_put_ask": [4.5],
            "exit_found_10d": [False],
        }
    )


def _predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [date(2024, 1, 5)],
            "symbol": ["A"],
            "y_score": [0.3],
            "y_true": [0.1],
        }
    )


def _write_raw_options(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    rows = []
    observations = (
        (date(2024, 1, 8), 100.0, 6.0, 4.0, 0.6, -0.4),
        (date(2024, 1, 9), 102.0, 5.5, 3.5, 0.7, -0.35),
        # The quoted expiry midpoint is deliberately not intrinsic. The lifecycle
        # loader must replace it with cash settlement: max(101-100, 0) + 0 = 1.
        (date(2024, 1, 10), 101.0, 3.0, 2.0, 0.8, -0.1),
    )
    for timestamp, underlying, call_mid, put_mid, call_delta, put_delta in observations:
        rows.extend(
            [
                {
                    "date": timestamp,
                    "symbol": "A",
                    "strike": 100.0,
                    "expiration": date(2024, 1, 10),
                    "call_put": "C",
                    "mid_price": call_mid,
                    "bid": max(call_mid - 0.5, 0.0),
                    "ask": call_mid + 0.5,
                    "delta": call_delta,
                    "underlying_price": underlying,
                },
                {
                    "date": timestamp,
                    "symbol": "A",
                    "strike": 100.0,
                    "expiration": date(2024, 1, 10),
                    "call_put": "P",
                    "mid_price": put_mid,
                    "bid": max(put_mid - 0.5, 0.0),
                    "ask": put_mid + 0.5,
                    "delta": put_delta,
                    "underlying_price": underlying,
                },
            ]
        )
    pl.DataFrame(rows).write_parquet(raw_dir / "year=2024.parquet")


def test_hold_to_expiry_selection_does_not_require_a_ten_day_exit_quote() -> None:
    cohorts = _select_cohorts(_predictions(), _contract_returns(), top_k=1)

    assert cohorts.height == 1
    assert cohorts.select("symbol", "timestamp", "strike", "expiration").row(0) == (
        "A",
        date(2024, 1, 5),
        100.0,
        date(2024, 1, 10),
    )


def test_real_prediction_subset_is_identity_covered_and_preview_only(tmp_path: Path) -> None:
    study = _study(tmp_path)
    source_hash = "released-source"
    source_path = (
        study.release_root
        / "case_studies"
        / "sp500_options"
        / "run_log"
        / "predictions"
        / source_hash
        / "predictions.parquet"
    )
    source_path.parent.mkdir(parents=True)
    timestamps = [datetime(2024, 1, day) for day in range(2, 7)]
    pl.DataFrame(
        {
            "symbol": [symbol for timestamp in timestamps for symbol in ("A", "B", "C")],
            "timestamp": [timestamp for timestamp in timestamps for _ in range(3)],
            "fold": [0] * 15,
            "prediction": [float(index) / 100 for index in range(15)],
            "actual": [float(index) / 200 for index in range(15)],
        }
    ).write_parquet(source_path)

    prediction = _seed_real_preview_prediction(
        study,
        source_prediction_hash=source_hash,
        max_symbols=2,
        max_sessions=5,
    )
    replayed = _seed_real_preview_prediction(
        study,
        source_prediction_hash=source_hash,
        max_symbols=2,
        max_sessions=5,
    )
    computation = prediction.lineage()["training_spec"]["computation"]

    assert prediction.complete
    assert replayed.hash == prediction.hash
    assert prediction.execution_tier == "preview"
    assert prediction.load().shape == (10, 5)
    assert computation["input_data_spec"]["source_prediction_hash"] == source_hash
    assert computation["preview_reductions"] == {
        "source_prediction_hash": source_hash,
        "folds": [0],
        "max_symbols": 2,
        "max_sessions": 5,
        "date_start": "2024-01-02 00:00:00",
        "date_end": "2024-01-06 00:00:00",
    }
    with pytest.raises(ValueError, match="preview run cannot create an official population"):
        OfficialPopulation.create(
            study,
            name="real-preview-fixture-must-not-enter",
            member_kind="prediction",
            members=[prediction.hash],
        )


def test_cash_settlement_and_stateful_delta_hedge(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    cohorts = _select_cohorts(_predictions(), _contract_returns(), top_k=1)
    lifecycle = _load_option_lifecycle(cohorts, raw_dir)

    expiry = lifecycle.filter(pl.col("date") == pl.col("expiration")).row(0, named=True)
    assert expiry["call_mid"] == pytest.approx(1.0)
    assert expiry["put_mid"] == pytest.approx(0.0)
    assert expiry["instr_mid"] == pytest.approx(1.0)
    assert expiry["cash_settled"] is True

    daily = _compute_cohort_daily_pnl(
        cohorts,
        lifecycle,
        delta_hedge=True,
        hedge_spread_bps=0.0,
        equity_commission_per_share=1.0,
        option_commission_per_contract=0.0,
        delta_threshold=0.10,
        option_spread_fraction=1.0,
    )

    assert daily.get_column("hedge_position").to_list() == pytest.approx([0.2, 0.35, 0.0])
    assert daily.get_column("hedge_trade").to_list() == pytest.approx([0.2, 0.15, -0.35])
    assert daily.get_column("hedge_pnl_norm").to_list() == pytest.approx([0.0, 0.04, -0.035])
    assert daily.get_column("hedge_cost_norm").to_list() == pytest.approx([0.02, 0.015, 0.035])
    assert daily.get_column("premium_pnl_norm").to_list() == pytest.approx([0.0, 0.1, 0.8])


def test_lifecycle_rejects_a_missing_contract_leg_date(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    raw_path = raw_dir / "year=2024.parquet"
    pl.read_parquet(raw_path).filter(
        ~((pl.col("date") == date(2024, 1, 9)) & (pl.col("call_put") == "P"))
    ).write_parquet(raw_path)
    cohorts = _select_cohorts(_predictions(), _contract_returns(), top_k=1)

    with pytest.raises(ValueError, match="missing 1 lifecycle dates"):
        _load_option_lifecycle(cohorts, raw_dir)


def test_supplied_lifecycle_cannot_drop_cash_settlement(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    cohorts = _select_cohorts(_predictions(), _contract_returns(), top_k=1)
    lifecycle = _load_option_lifecycle(cohorts, raw_dir).filter(
        pl.col("date") < pl.col("expiration")
    )

    with pytest.raises(ValueError, match="cash-settle every selected contract"):
        _compute_cohort_daily_pnl(
            cohorts,
            lifecycle,
            delta_hedge=True,
            hedge_spread_bps=0.0,
            equity_commission_per_share=0.0,
            option_commission_per_contract=0.0,
            delta_threshold=0.1,
        )


def test_typed_contract_rows_match_direct_option_selection(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    raw_dir = tmp_path / "raw"
    labels_dir.mkdir()
    _contract_returns().write_parquet(labels_dir / "contract_returns.parquet")
    _write_raw_options(raw_dir)
    predictions = _predictions()
    decisions = _select_cohorts(predictions, _contract_returns(), top_k=1)

    direct = run_htm_daily_mtm(
        "sp500_options",
        predictions,
        labels_dir,
        raw_dir,
        top_k=1,
        delta_threshold=0.10,
        hedge_spread_bps=0.0,
        equity_commission_per_share=0.0,
        option_commission_per_contract=0.0,
    )
    typed = run_htm_daily_mtm(
        "sp500_options",
        predictions,
        labels_dir,
        raw_dir,
        decisions=decisions,
        top_k=1,
        delta_threshold=0.10,
        hedge_spread_bps=0.0,
        equity_commission_per_share=0.0,
        option_commission_per_contract=0.0,
    )

    assert typed["daily_returns"].equals(direct["daily_returns"])
    assert typed["metrics"] == direct["metrics"]


def test_short_straddle_decision_artifact_keeps_contract_identity(tmp_path: Path) -> None:
    study = _study(tmp_path)
    prediction = _prediction(study)
    decisions = _select_cohorts(_predictions(), _contract_returns(), top_k=1).drop("y_score")

    artifact = DecisionArtifact.publish(
        study,
        kind="short_straddles",
        decisions=decisions,
        prediction_hashes=[prediction.hash],
        parameters={
            "decision_cadence": "weekly_friday",
            "entry_policy": "next_session_close",
            "exit_policy": "hold_to_expiry",
            "settlement_policy": "cash_intrinsic_at_expiration",
        },
    )

    assert artifact.spec["decision_keys"] == ["symbol", "timestamp"]
    assert (
        artifact.load()
        .select("symbol", "timestamp", "strike", "expiration", "entry_date", "weight")
        .equals(
            decisions.select("symbol", "timestamp", "strike", "expiration", "entry_date", "weight")
        )
    )

    with pytest.raises(ValueError, match="expiration"):
        DecisionArtifact.publish(
            study,
            kind="short_straddles",
            decisions=decisions.drop("expiration"),
            prediction_hashes=[prediction.hash],
            parameters={"exit_policy": "hold_to_expiry"},
        )


def test_reader_boundary_publishes_contracts_consumed_by_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from case_studies.sp500_options import research_workflow

    study = _study(tmp_path)
    prediction = _prediction(study)
    labels_dir = tmp_path / "labels"
    raw_dir = tmp_path / "raw"
    labels_dir.mkdir()
    _contract_returns().write_parquet(labels_dir / "contract_returns.parquet")
    _write_raw_options(raw_dir)
    monkeypatch.setattr(research_workflow, "option_data_paths", lambda: (labels_dir, raw_dir))
    prices = pl.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": [datetime(2024, 1, 5)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000],
        }
    )
    signal = {"method": "equal_weight_top_k", "top_k": 1}

    decision = publish_short_straddle_decisions(
        prediction,
        prices=prices,
        signal=signal,
    )
    strategy = study.strategy(prediction=prediction, signal=signal, decision=decision)
    consumed = strategy._decision_weights(prices)

    assert decision.kind == "short_straddles"
    assert consumed is not None
    assert consumed.select(
        "symbol", "timestamp", "strike", "expiration", "entry_date", "weight", "fold"
    ).equals(
        decision.load().select(
            "symbol", "timestamp", "strike", "expiration", "entry_date", "weight", "fold"
        )
    )
    with pytest.raises(ValueError, match="clean-process replay digest"):
        publish_short_straddle_decisions(
            prediction,
            prices=prices,
            signal=signal,
            canonical=True,
        )
    canonical = publish_short_straddle_decisions(
        prediction,
        prices=prices,
        signal=signal,
        canonical=True,
        clean_replay_digest=value_digest(decision.load()),
    )
    declared_inputs = canonical.spec["source_identity"]["declared_inputs"]
    assert declared_inputs["option_contract_returns"]
    # The raw option chain is a declared input, not just the labels table: the decision
    # reads strikes and quotes straight out of it, so a replay against a different chain
    # is reading different data and must be refused rather than silently republished.
    assert declared_inputs["option_sources"] == option_source_identity(labels_dir, raw_dir)
    assert canonical.spec["source_identity"]["holdout_replay"] == {
        "version": 1,
        "function": "resolve_short_straddle_decisions",
    }


def test_typed_decision_runs_through_registered_option_backtest_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from case_studies.sp500_options import _htm_backtest, research_workflow
    from case_studies.utils import cv_window

    study = _study(tmp_path)
    prediction = _prediction(study)
    labels_dir = tmp_path / "labels"
    raw_dir = tmp_path / "raw"
    labels_dir.mkdir()
    _contract_returns().write_parquet(labels_dir / "contract_returns.parquet")
    _write_raw_options(raw_dir)
    monkeypatch.setattr(research_workflow, "option_data_paths", lambda: (labels_dir, raw_dir))
    monkeypatch.setattr(_htm_backtest, "option_data_paths", lambda: (labels_dir, raw_dir))
    monkeypatch.setattr(cv_window, "canonical_window", lambda *args, **kwargs: None)
    prices = pl.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": [datetime(2024, 1, 5)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000],
        }
    )
    signal = {"method": "equal_weight_top_k", "top_k": 1}
    decision = publish_short_straddle_decisions(
        prediction,
        prices=prices,
        signal=signal,
    )
    lifecycle = _load_option_lifecycle(decision.load(), raw_dir)

    result = study.strategy(
        prediction=prediction,
        signal=signal,
        decision=decision,
    ).run(prices=prices, option_lifecycle=lifecycle)
    spec = result.spec()

    assert result.complete
    assert spec["decision_artifact"]["kind"] == "short_straddles"
    assert spec["decision_artifact"]["hash"] == decision.hash
    assert spec["input_identity"]["option_contract_returns"]
    assert spec["input_identity"]["option_lifecycle"]
    assert spec["options_market"]["settlement"] == "cash_intrinsic_at_expiration"
    assert spec["options_accounting"]["delta_threshold"] == pytest.approx(0.1)
    assert spec["options_accounting"]["option_commission_per_contract"] == pytest.approx(0.65)
    assert spec["options_accounting"]["equity_commission_per_share"] == pytest.approx(0.005)
    assert spec["options_accounting"]["option_contract_multiplier"] == 100


def test_preview_option_requests_resolve_preview_catalog_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from case_studies.sp500_options import _htm_backtest, research_workflow
    from case_studies.utils import cv_window

    study = _study(tmp_path)
    prediction = _prediction(study, execution_tier="preview")
    labels_dir = study.root / "labels"
    raw_dir = tmp_path / "raw"
    labels_dir.mkdir()
    _contract_returns().write_parquet(labels_dir / "contract_returns.parquet")
    _write_raw_options(raw_dir)
    monkeypatch.setattr(research_workflow, "option_data_paths", lambda: (labels_dir, raw_dir))
    monkeypatch.setattr(_htm_backtest, "option_data_paths", lambda: (labels_dir, raw_dir))
    monkeypatch.setattr(cv_window, "canonical_window", lambda *args, **kwargs: None)
    prices = pl.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": [datetime(2024, 1, 5)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000],
        }
    )
    signal = {"method": "equal_weight_top_k", "top_k": 1}
    decisions = research_workflow.resolve_short_straddle_decisions(
        prediction,
        prices=prices,
        signal=signal,
    )
    monkeypatch.setattr(research_workflow, "load_backtest_prices_for", lambda *a, **k: prices)
    monkeypatch.setattr(
        research_workflow,
        "_clean_replay_digests",
        lambda *_args, **_kwargs: {"preview-top-1": value_digest(decisions)},
    )
    requests = strategy_request_frame(
        [
            {
                "request_name": "preview-top-1",
                "prediction_hash": prediction.hash,
                "label": "ret_to_expiry",
                "signal": signal,
                "allocation": None,
                "risk": None,
                "costs": None,
            }
        ]
    )

    execution = run_official_backtest_requests(study, requests, population_name=None)

    assert execution.population is None
    assert len(execution.results) == 1
    assert execution.results[0].execution_tier == "preview"
    assert execution.catalog_rows.item(0, "prediction_hash") == prediction.hash


def test_option_request_fanout_preserves_each_prediction_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from case_studies.sp500_options import _htm_backtest, research_workflow
    from case_studies.utils import cv_window

    study = _study(tmp_path)
    predictions = (
        _prediction(study, execution_tier="preview", alpha=1.0),
        _prediction(study, execution_tier="preview", alpha=2.0),
    )
    labels_dir = study.root / "labels"
    raw_dir = tmp_path / "raw"
    labels_dir.mkdir()
    _contract_returns().write_parquet(labels_dir / "contract_returns.parquet")
    _write_raw_options(raw_dir)
    monkeypatch.setattr(research_workflow, "option_data_paths", lambda: (labels_dir, raw_dir))
    monkeypatch.setattr(_htm_backtest, "option_data_paths", lambda: (labels_dir, raw_dir))
    monkeypatch.setattr(cv_window, "canonical_window", lambda *args, **kwargs: None)
    prices = pl.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": [datetime(2024, 1, 5)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000],
        }
    )
    signal = {"method": "equal_weight_top_k", "top_k": 1}
    digest = value_digest(
        research_workflow.resolve_short_straddle_decisions(
            predictions[0],
            prices=prices,
            signal=signal,
        )
    )
    monkeypatch.setattr(research_workflow, "load_backtest_prices_for", lambda *a, **k: prices)
    monkeypatch.setattr(
        research_workflow,
        "_clean_replay_digests",
        lambda *_args, **_kwargs: {"first": digest, "second": digest},
    )
    requests = strategy_request_frame(
        [
            {
                "request_name": name,
                "prediction_hash": prediction.hash,
                "label": "ret_to_expiry",
                "signal": signal,
                "allocation": None,
                "risk": None,
                "costs": None,
            }
            for name, prediction in zip(("first", "second"), predictions, strict=True)
        ]
    )

    execution = run_official_backtest_requests(study, requests, population_name=None)

    assert len(execution.results) == 2
    assert len({result.hash for result in execution.results}) == 2
    assert {result.registry_record()["prediction_hash"] for result in execution.results} == {
        prediction.hash for prediction in predictions
    }


def test_unsupported_option_overrides_fail_before_backtesting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from case_studies.sp500_options import _htm_backtest, research_workflow

    study = _study(tmp_path)
    prediction = _prediction(study)
    labels_dir = tmp_path / "labels"
    raw_dir = tmp_path / "raw"
    labels_dir.mkdir()
    _contract_returns().write_parquet(labels_dir / "contract_returns.parquet")
    _write_raw_options(raw_dir)
    monkeypatch.setattr(research_workflow, "option_data_paths", lambda: (labels_dir, raw_dir))
    monkeypatch.setattr(_htm_backtest, "option_data_paths", lambda: (labels_dir, raw_dir))
    prices = pl.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": [datetime(2024, 1, 5)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000],
        }
    )
    signal = {"method": "equal_weight_top_k", "top_k": 1}
    decision = publish_short_straddle_decisions(prediction, prices=prices, signal=signal)

    with pytest.raises(ValueError, match="does not support risk overlays"):
        study.strategy(
            prediction=prediction,
            signal=signal,
            risk={"method": "vol_target"},
            decision=decision,
        ).resolve(prices=prices)
    with pytest.raises(ValueError, match="generic costs are unsupported"):
        study.strategy(
            prediction=prediction,
            signal=signal,
            costs={"model": "percentage", "commission_bps": 1.0, "slippage_bps": 1.0},
            decision=decision,
        ).resolve(prices=prices)

    assert study.backtests.table(include_preview=True).is_empty()


def test_official_population_is_snapshotted_before_option_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from case_studies.research.strategy import Strategy
    from case_studies.sp500_options import _htm_backtest, research_workflow
    from case_studies.utils import cv_window

    study = _study(tmp_path)
    prediction = _prediction(study)
    labels_dir = study.root / "labels"
    raw_dir = tmp_path / "raw"
    labels_dir.mkdir()
    _contract_returns().write_parquet(labels_dir / "contract_returns.parquet")
    _write_raw_options(raw_dir)
    monkeypatch.setattr(research_workflow, "option_data_paths", lambda: (labels_dir, raw_dir))
    monkeypatch.setattr(_htm_backtest, "option_data_paths", lambda: (labels_dir, raw_dir))
    monkeypatch.setattr(cv_window, "canonical_window", lambda *args, **kwargs: None)
    prices = pl.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": [datetime(2024, 1, 5)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000],
        }
    )
    monkeypatch.setattr(research_workflow, "load_backtest_prices_for", lambda *a, **k: prices)
    population_name = "sp500-options-test-population"

    def fail_after_snapshot(self, **kwargs):
        del self, kwargs
        population = OfficialPopulation.one(study, name=population_name)
        assert len(population.members) == 1
        raise RuntimeError("injected option execution failure")

    monkeypatch.setattr(Strategy, "run", fail_after_snapshot)
    requests = strategy_request_frame(
        [
            {
                "request_name": "liquid-top-1",
                "prediction_hash": prediction.hash,
                "label": "ret_to_expiry",
                "signal": {"method": "equal_weight_top_k", "top_k": 1},
                "allocation": None,
                "risk": None,
                "costs": None,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="injected option execution failure"):
        run_official_backtest_requests(study, requests, population_name=population_name)

    population = OfficialPopulation.one(study, name=population_name)
    with pytest.raises(ValueError, match="incomplete"):
        population.require_complete()


def _equal_weight_cohorts() -> pl.DataFrame:
    """Two symbols entered on one date at equal weight, with unequal scores."""
    return pl.DataFrame(
        {
            "timestamp": [date(2024, 1, 5), date(2024, 1, 5)],
            "symbol": ["A", "B"],
            "y_score": [0.3, 0.1],
            "weight": [0.5, 0.5],
        }
    )


def test_score_weighted_allocation_reweights_an_equal_weight_entry(tmp_path: Path) -> None:
    """The allocator sizes by score even when the signal selected at equal weight.

    The allocator sweep pairs one baseline against several weighting rules and copies
    the baseline's signal verbatim, so `score_weighted` has to compute its own weights.
    Reading them off the signal made the run realize equal weight while being published
    as score-weighted.
    """
    weighted = _apply_cohort_allocator(
        _equal_weight_cohorts(),
        tmp_path,
        {"method": "score_weighted", "top_k": 2},
    )

    weights = dict(zip(weighted["symbol"], weighted["weight"], strict=True))
    assert weights["A"] == pytest.approx(0.75)
    assert weights["B"] == pytest.approx(0.25)


def test_equal_weight_allocation_leaves_the_signal_weights_alone(tmp_path: Path) -> None:
    cohorts = _equal_weight_cohorts()
    unchanged = _apply_cohort_allocator(cohorts, tmp_path, {"method": "equal_weight"})

    assert unchanged.equals(cohorts)


def test_conformal_weighted_allocation_is_dispatched_not_refused(tmp_path: Path) -> None:
    """The declared allocator menu lists `conformal_weighted`, so the path must run it.

    Without a prediction hash there are no calibrated widths to size by, and the
    failure has to say that rather than report the method as unsupported.
    """
    with pytest.raises(ValueError, match="must pass prediction_hash"):
        _apply_cohort_allocator(
            _equal_weight_cohorts(),
            tmp_path,
            {"method": "conformal_weighted", "top_k": 2},
        )


def test_a_lifecycle_gap_does_not_shape_the_decision_universe(tmp_path: Path) -> None:
    """A quote gap after the decision date must not decide what could be ranked on it.

    The screen reads the paired chain from entry through expiration, so every date it looks at
    is later than the decision it would filter. Two candidates make the difference visible: A
    scores higher and has the gap, B scores lower and is complete. A screen that ran before the
    ranking would drop A and hand the cohort to B; running it on the selection keeps A ranked
    first and reports that its lifecycle cannot be accounted for.
    """
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    chain = pl.read_parquet(raw_dir / "year=2024.parquet")
    runner_up = chain.with_columns(symbol=pl.lit("B"))
    # Drop one leg of one session for A: the session still exists in the chain, so the gap is
    # that contract's own rather than a hole in the calendar every contract shares.
    gap = (pl.col("date") == date(2024, 1, 9)) & (pl.col("call_put") == "C")
    pl.concat([chain.filter(~gap), runner_up]).write_parquet(raw_dir / "year=2024.parquet")

    predictions = pl.concat(
        [_predictions(), _predictions().with_columns(symbol=pl.lit("B"), y_score=pl.lit(0.1))]
    )
    contract_returns = pl.concat(
        [_contract_returns(), _contract_returns().with_columns(symbol=pl.lit("B"))]
    )

    ranked = _select_cohorts(predictions, contract_returns, top_k=1)
    assert ranked.get_column("symbol").to_list() == ["A"]

    with pytest.raises(ValueError, match="complete paired lifecycle") as refusal:
        _select_cohorts(predictions, contract_returns, top_k=1, raw_options_dir=raw_dir)
    # The cohort the screen refused is A's, so the ranking was not handed to the runner-up.
    assert "'symbol': 'A'" in str(refusal.value)


def test_a_missing_entry_quote_does_not_shape_the_decision_universe() -> None:
    """The quote read on the session after the decision cannot decide what was rankable.

    A outscores B and has no entry call quote. Dropping A before the ranking would hand the
    cohort to B and say nothing; the selection must still be A's, and the refusal must name it.
    """
    predictions = pl.concat(
        [_predictions(), _predictions().with_columns(symbol=pl.lit("B"), y_score=pl.lit(0.1))]
    )
    contract_returns = pl.concat(
        [
            _contract_returns().with_columns(entry_call_mid=pl.lit(None, dtype=pl.Float64)),
            _contract_returns().with_columns(symbol=pl.lit("B")),
        ]
    )

    with pytest.raises(ValueError, match="no complete entry quote") as refusal:
        _select_cohorts(predictions, contract_returns, top_k=1)
    assert "'symbol': 'A'" in str(refusal.value)


def _week_of_candidates(sessions: list[date]) -> pl.DataFrame:
    """A contract-returns artifact with one candidate on each of the given decision dates."""
    return pl.DataFrame(
        {
            "feature_date": sessions,
            "symbol": ["A"] * len(sessions),
            "strike": [100.0] * len(sessions),
            "expiration": [date(2024, 1, 19)] * len(sessions),
            "entry_date": [date(2024, 1, 16)] * len(sessions),
            "entry_straddle_mid": [10.0] * len(sessions),
        }
    )


def test_the_displayed_calendar_follows_the_schedule_the_predictions_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prediction set that stops short of Friday enters on Thursday, and the table says so.

    The engine resolves ``weekly_friday`` from the prediction frame it is handed, so a week
    whose Friday is absent from the predictions rebalances on the Thursday. Reading the
    schedule off the complete contract artifact instead would display the Friday - a session
    no backtest here enters on - and hide the Thursday it does.
    """
    from case_studies.sp500_options import research_workflow

    study = _study(tmp_path)
    spec = _resolved_spec(alpha=1.0)
    spec["label"] = "ret_to_expiry"
    training = study.results.register_training(spec)
    thursday, friday = date(2024, 1, 11), date(2024, 1, 12)
    frame = pl.DataFrame(
        {
            "symbol": ["A", "A"],
            "timestamp": [datetime(2024, 1, 10), datetime(2024, 1, 11)],
            "fold": [0, 0],
            "actual": [0.01, 0.02],
            "prediction": [0.02, 0.03],
        }
    )
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold"),
    )

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    # The artifact carries the Friday the predictions never reach.
    _week_of_candidates([date(2024, 1, 10), thursday, friday]).write_parquet(
        labels_dir / "contract_returns.parquet"
    )
    monkeypatch.setattr(
        research_workflow, "option_data_paths", lambda: (labels_dir, tmp_path / "raw")
    )

    decision_dates = option_decision_dates(study, [prediction.hash])
    calendar = option_trade_calendar(decision_dates)

    assert decision_dates.to_list() == [thursday]
    assert calendar.get_column("decision_date").to_list() == [thursday]


def test_conformal_weighted_sizes_by_width_and_drops_uncalibrated_dates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The calibrated dates get inverse-width weights; the uncalibrated one is removed.

    An entry date earlier than the first calibration window has no prior-only width, so it
    cannot be sized this way. Filling it with equal weight would publish a run as
    conformal-weighted while realizing the weighting the allocator exists to replace, so
    those cohorts leave the result instead.
    """
    from case_studies.utils import conformal

    prediction_hash = "conformalfixture01"
    calibrated, uncalibrated = date(2024, 1, 12), date(2024, 1, 5)
    widths_dir = tmp_path / "case" / "run_log" / "predictions" / prediction_hash
    widths_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [calibrated, calibrated],
            "symbol": ["A", "B"],
            # B's interval is three times as wide, so it takes a quarter of the cohort.
            "width": [1.0, 3.0],
            "alpha": [conformal.DEFAULT_ALPHA, conformal.DEFAULT_ALPHA],
            "calibration_version": [conformal.CALIBRATION_VERSION] * 2,
        }
    ).write_parquet(widths_dir / "conformal_widths.parquet")
    monkeypatch.setattr(conformal, "get_case_study_dir", lambda _case: tmp_path / "case")

    cohorts = pl.DataFrame(
        {
            "timestamp": [uncalibrated, uncalibrated, calibrated, calibrated],
            "symbol": ["A", "B", "A", "B"],
            "y_score": [0.3, 0.1, 0.3, 0.1],
            "weight": [0.5, 0.5, 0.5, 0.5],
        }
    )

    sized = _apply_cohort_allocator(
        cohorts,
        tmp_path / "raw",
        {"method": "conformal_weighted", "top_k": 2, "floor_quantile": 0.0},
        prediction_hash=prediction_hash,
    )

    assert sized.get_column("timestamp").unique().to_list() == [calibrated]
    weights = dict(zip(sized["symbol"], sized["weight"], strict=True))
    assert weights["A"] == pytest.approx(0.75)
    assert weights["B"] == pytest.approx(0.25)


def test_a_pair_is_compared_on_the_dates_both_backtests_traded(tmp_path: Path) -> None:
    """An allocator that starts trading later is not credited with the period it skipped.

    `conformal_weighted` has no weight for an entry date with no prior-only calibration
    window, so its series starts later than the baseline it is built from. Reading each
    registered Sharpe compares two different stretches of market; both sides are recomputed
    on the dates they share.
    """
    from case_studies.utils.registry.registration import register_backtest_run

    study = _study(tmp_path)
    prediction = _prediction(study)
    sessions = [datetime(2024, 1, day) for day in range(2, 12)]
    # The baseline loses money over the first half and makes it over the second. A comparison
    # on its full series would read a different baseline than the one the variant competed with.
    baseline_returns = [-0.05, -0.03, -0.06, -0.04, -0.05, 0.02, 0.03, 0.01, 0.04, 0.02]
    baseline_hash = register_backtest_run(
        "sp500_options",
        prediction.hash,
        {
            "execution_tier": "canonical",
            "strategy": {"signal": {"method": "equal_weight_top_k", "top_k": 1}},
        },
        stage="signal",
        returns=pl.DataFrame({"timestamp": sessions, "return": baseline_returns}),
        metrics={"sharpe": 0.1},
        case_dir=study.root,
    )
    variant_hash = register_backtest_run(
        "sp500_options",
        prediction.hash,
        {
            "execution_tier": "canonical",
            "strategy": {
                "signal": {"method": "equal_weight_top_k", "top_k": 1},
                "allocation": {"method": "conformal_weighted"},
            },
        },
        stage="allocation",
        returns=pl.DataFrame({"timestamp": sessions[5:], "return": [0.03, 0.05, 0.02, 0.06, 0.04]}),
        metrics={"sharpe": 5.0},
        case_dir=study.root,
    )

    paired = paired_sharpe_on_common_support(
        study,
        pl.DataFrame({"baseline_hash": [baseline_hash], "backtest_hash": [variant_hash]}),
    )

    row = paired.row(0, named=True)
    assert row["n_periods"] == 5
    assert row["baseline_periods"] == 10
    # The baseline's own series straddles zero; on the common support it is the winning half.
    assert row["baseline_sharpe"] > 0
