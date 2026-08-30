from __future__ import annotations

import sqlite3
import sys
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
    split: str = "validation",
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
        split=split,
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


def test_preview_study_activates_before_model_catalog_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preview tier is activated inside `open_study`, before anything resolves config.

    The branch under test only runs when the generated artifact directories are symlinks, which
    is true of a working checkout and false of a clean clone, so the precondition is built here
    rather than depending on the machine. What activation does is redirect `ML4T_OUTPUT_DIR` at
    the workspace's `.preview`; asserting `study.output_root` instead would hold with the
    activation removed, because the constructor sets it either way.
    """
    import os

    from case_studies.sp500_options import research_workflow

    repo_root = tmp_path / "repo"
    case_dir = repo_root / "case_studies" / "sp500_options"
    case_dir.mkdir(parents=True)
    (repo_root / "case_studies" / "config").symlink_to(
        REPO_ROOT / "case_studies" / "config", target_is_directory=True
    )
    generated = tmp_path / "generated"
    for name in ("features", "labels", "run_log"):
        (generated / name).mkdir(parents=True)
        (case_dir / name).symlink_to(generated / name, target_is_directory=True)
    (case_dir / "config").symlink_to(
        REPO_ROOT / "case_studies" / "sp500_options" / "config", target_is_directory=True
    )
    monkeypatch.setattr(research_workflow, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        research_workflow.subprocess, "check_output", lambda *_args, **_kwargs: "deadbeef\n"
    )

    workspace = tmp_path / "workspace"
    study = open_study(execution_tier="preview", workspace=workspace)

    catalog = model_request_catalog("linear", config_names=["ridge_a1.0"])

    assert os.environ["ML4T_OUTPUT_DIR"] == str(workspace / ".preview")
    assert (workspace / ".preview" / "sp500_options" / "config").exists()
    assert study.output_root == workspace
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


def _release_prediction(study: Study, *, label: str = "ret_to_expiry") -> str:
    """Register a complete canonical validation prediction in the release registry."""
    from case_studies.utils.registry import register_prediction_set, register_training_run

    case_dir = study.release_root / "case_studies" / "sp500_options"
    spec = _resolved_spec(alpha=1.0)
    spec["label"] = label
    spec["execution_tier"] = "canonical"
    training_hash = register_training_run("sp500_options", spec, case_dir=case_dir)
    timestamps = [datetime(2024, 1, day) for day in range(2, 7)]
    frame = pl.DataFrame(
        {
            "symbol": [symbol for _ in timestamps for symbol in ("A", "B", "C")],
            "timestamp": [timestamp for timestamp in timestamps for _ in range(3)],
            "fold": [0] * 15,
            "prediction": [float(index) / 100 for index in range(15)],
            "actual": [float(index) / 200 for index in range(15)],
        }
    )
    return register_prediction_set(
        "sp500_options",
        training_hash,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold"),
        case_dir=case_dir,
    )


def test_the_fixture_refuses_a_source_that_is_not_a_released_ret_to_expiry_prediction(
    tmp_path: Path,
) -> None:
    """The fixture republishes its source as a complete `ret_to_expiry` validation prediction.

    It used to read whatever parquet sat at the constructed path, so an artifact scored on
    another label entered the registry under a `ret_to_expiry` identity with new lineage and
    nothing said otherwise.
    """
    study = _study(tmp_path)
    wrong_label_hash = _release_prediction(study, label="fwd_ret_5d")

    with pytest.raises(ValueError, match="the fixture requires"):
        _seed_real_preview_prediction(
            study,
            source_prediction_hash=wrong_label_hash,
            max_symbols=2,
            max_sessions=5,
        )


def test_real_prediction_subset_is_identity_covered_and_preview_only(tmp_path: Path) -> None:
    study = _study(tmp_path)
    source_hash = _release_prediction(study)

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
    """One quoted leg is a defect, not the end of the position.

    A chain that quotes neither leg has stopped carrying the contract; a chain that quotes one
    still carries it and has lost the other. Both fail to produce a paired row, so the lifecycle
    has to tell them apart explicitly - otherwise a half-written session would end the position
    early and book a liquidation against it.
    """
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    raw_path = raw_dir / "year=2024.parquet"
    pl.read_parquet(raw_path).filter(
        ~((pl.col("date") == date(2024, 1, 9)) & (pl.col("call_put") == "P"))
    ).write_parquet(raw_path)
    cohorts = _select_cohorts(_predictions(), _contract_returns(), top_k=1)

    with pytest.raises(ValueError, match="missing 1 contract-leg dates"):
        _load_option_lifecycle(cohorts, raw_dir)


def _cross_the_call_quote(raw_dir: Path, on: date) -> None:
    """Make the call's bid exceed its ask on one session, as the vendor chain does."""
    raw_path = raw_dir / "year=2024.parquet"
    crossed = (pl.col("date") == on) & (pl.col("call_put") == "C")
    pl.read_parquet(raw_path).with_columns(
        bid=pl.when(crossed).then(pl.lit(0.02)).otherwise(pl.col("bid")),
        ask=pl.when(crossed).then(pl.lit(0.01)).otherwise(pl.col("ask")),
    ).write_parquet(raw_path)


def test_a_crossed_quote_on_the_expiration_session_still_settles_at_intrinsic(
    tmp_path: Path,
) -> None:
    """The expiry mids are discarded for cash settlement, so their order cannot matter.

    COST 295 expiring 2020-01-03 closed its last session bid 0.02 / ask 0.01. Nothing
    reads that quote - the straddle settles at max(underlying - strike, 0) + max(strike
    - underlying, 0) - and rejecting it halted the whole backtest over one crossed row.
    """
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    _cross_the_call_quote(raw_dir, date(2024, 1, 10))
    cohorts = _select_cohorts(_predictions(), _contract_returns(), top_k=1)

    lifecycle = _load_option_lifecycle(cohorts, raw_dir)
    settled = lifecycle.filter(pl.col("cash_settled"))
    assert settled.height == 1
    assert settled.get_column("date").item() == date(2024, 1, 10)
    assert settled.get_column("instr_mid").item() == pytest.approx(1.0)


def test_a_crossed_quote_before_expiration_is_still_rejected(tmp_path: Path) -> None:
    """The exemption is the expiration session and nothing wider: 01-09 marks the position."""
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    _cross_the_call_quote(raw_dir, date(2024, 1, 9))
    cohorts = _select_cohorts(_predictions(), _contract_returns(), top_k=1)

    with pytest.raises(ValueError, match="invalid quote rows"):
        _load_option_lifecycle(cohorts, raw_dir)


def test_supplied_lifecycle_cannot_drop_the_end_of_a_position(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    cohorts = _select_cohorts(_predictions(), _contract_returns(), top_k=1)
    lifecycle = _load_option_lifecycle(cohorts, raw_dir).filter(
        pl.col("date") < pl.col("expiration")
    )

    with pytest.raises(ValueError, match="end every selected contract"):
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
    lifecycle_source = option_source_identity(labels_dir, raw_dir)["raw_lifecycle"]

    strategy = study.strategy(prediction=prediction, signal=signal, decision=decision)
    # A frame the identity does not account for cannot be the one the returns are computed from.
    with pytest.raises(ValueError, match="declare the raw files"):
        strategy.run(prices=prices, option_lifecycle=lifecycle)
    result = strategy.run(
        prices=prices,
        option_lifecycle=lifecycle,
        option_lifecycle_source=lifecycle_source,
    )
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


def _chain_with_a_terminated_contract(raw_dir: Path) -> None:
    """A is quoted on its entry session only; B is quoted throughout.

    That is the shape a corporate action leaves in this chain: the contract is adjusted onto a
    strike the slice does not carry, so its quotes stop and never resume. The calendar still
    holds 2024-01-09 and the expiration session, because B is quoted on both.

    A stops after 2024-01-08 rather than after 2024-01-09 so that the session the exit is booked
    on, the first with no quote, falls strictly inside the holding period. Stopping a day later
    would put it on the expiration session itself, where a liquidation and a cash settlement
    would be indistinguishable by date and the test could not tell which rule had fired.
    """
    _write_raw_options(raw_dir)
    chain = pl.read_parquet(raw_dir / "year=2024.parquet")
    runner_up = chain.with_columns(symbol=pl.lit("B"))
    ended = chain.filter(pl.col("date") < date(2024, 1, 9))
    pl.concat([ended, runner_up]).write_parquet(raw_dir / "year=2024.parquet")


def _two_names() -> tuple[pl.DataFrame, pl.DataFrame]:
    predictions = pl.concat(
        [_predictions(), _predictions().with_columns(symbol=pl.lit("B"), y_score=pl.lit(0.1))]
    )
    contract_returns = pl.concat(
        [_contract_returns(), _contract_returns().with_columns(symbol=pl.lit("B"))]
    )
    return predictions, contract_returns


def test_a_contract_ended_by_a_corporate_action_is_liquidated_not_settled(
    tmp_path: Path,
) -> None:
    """The position is opened, then bought back where the quotes stop. It is not withheld.

    A outscores B and its contract stops being quoted before expiration, which is what a
    corporate action leaves in this chain: the position continues under an adjusted strike the
    slice does not carry. Withholding the position would condition the realized portfolio on an
    event nobody knew about at entry and would throw away the P&L it earned before that event.

    What it is *not* is a cash settlement. Settlement happens at expiration, at intrinsic,
    against no counterparty. This is a trade against the last mark the chain carried, and the
    accounting has to charge it as one.

    The exit is dated to the first session with no quote, not to the last session with one. On
    the last quoted session the holder has no reason to act and no way to know it was the last;
    they learn that the following morning. Dating it a day earlier picks the exit date with
    hindsight, which is what the gate objected to.
    """
    raw_dir = tmp_path / "raw"
    _chain_with_a_terminated_contract(raw_dir)
    predictions, contract_returns = _two_names()

    cohorts = _select_cohorts(predictions, contract_returns, top_k=1, raw_options_dir=raw_dir)
    assert cohorts.get_column("symbol").to_list() == ["A"]

    lifecycle = _load_option_lifecycle(cohorts, raw_dir)
    ended = lifecycle.filter(pl.col("liquidated"))
    assert ended.get_column("date").to_list() == [date(2024, 1, 9)]
    assert ended.get_column("date").item() < ended.get_column("expiration").item()
    # The exit is booked against the mark the previous session carried, so the liquidation row
    # repeats 2024-01-08's quote rather than inventing one for a session nothing was quoted on.
    prior = lifecycle.filter(pl.col("date") == date(2024, 1, 8))
    assert ended.get_column("call_mid").item() == prior.get_column("call_mid").item()
    assert ended.get_column("call_ask").item() == prior.get_column("call_ask").item()
    assert ended.get_column("put_mid").item() == prior.get_column("put_mid").item()
    # A bought-back straddle carries no delta into the sessions that follow.
    assert ended.get_column("instr_delta").item() == 0.0
    # Nothing reached expiry, so nothing settled.
    assert lifecycle.filter(pl.col("cash_settled")).is_empty()


def test_a_liquidated_contract_pays_the_exit_spread(tmp_path: Path) -> None:
    """A buy-to-close against the previous session's mark costs the spread, like any other exit.

    Marking it out at the midpoint for free was the defect: it made a position the market
    stopped quoting cheaper to leave than one the strategy chose to leave, which is backwards.
    """
    raw_dir = tmp_path / "raw"
    _chain_with_a_terminated_contract(raw_dir)
    predictions, contract_returns = _two_names()

    cohorts = _select_cohorts(predictions, contract_returns, top_k=1, raw_options_dir=raw_dir)
    lifecycle = _load_option_lifecycle(cohorts, raw_dir)
    daily = _compute_cohort_daily_pnl(
        cohorts,
        lifecycle,
        delta_hedge=False,
        hedge_spread_bps=0.0,
        equity_commission_per_share=0.0,
        option_commission_per_contract=0.0,
        delta_threshold=0.1,
    )
    charged = daily.filter(pl.col("exit_cost_norm") > 0.0)
    assert charged.get_column("date").to_list() == [date(2024, 1, 9)]
    assert charged.get_column("liquidated").to_list() == [True]


def test_a_position_survives_an_interior_session_nobody_quoted(tmp_path: Path) -> None:
    """A gap in the middle of a life is an unmarked session, not the end of the position.

    This is the end-to-end version deliberately: cohort selection already accepted a contract
    with an interior gap, and the lifecycle then refused it, so the two disagreed about the same
    data and a run could pass selection and die later. Stopping at `_select_cohorts` would not
    have caught that. The test therefore loads the lifecycle and computes the daily P&L across
    the gap.

    Nothing is invented for the unquoted session. The position simply is not remarked, so the
    move across the gap is recognised on the session quotes resume - which here is expiration,
    where the straddle settles at intrinsic.
    """
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    chain = pl.read_parquet(raw_dir / "year=2024.parquet")
    # B keeps the calendar intact; A loses both legs on the interior session only.
    runner_up = chain.with_columns(symbol=pl.lit("B"))
    gapped = chain.filter(pl.col("date") != date(2024, 1, 9))
    pl.concat([gapped, runner_up]).write_parquet(raw_dir / "year=2024.parquet")
    predictions, contract_returns = _two_names()

    cohorts = _select_cohorts(predictions, contract_returns, top_k=1, raw_options_dir=raw_dir)
    assert cohorts.get_column("symbol").to_list() == ["A"]

    lifecycle = _load_option_lifecycle(cohorts, raw_dir)
    assert lifecycle.get_column("date").to_list() == [date(2024, 1, 8), date(2024, 1, 10)]
    # It reached expiration, so it settled. A gap that closes is not a liquidation.
    assert lifecycle.filter(pl.col("liquidated")).is_empty()
    assert lifecycle.filter(pl.col("cash_settled")).get_column("date").to_list() == [
        date(2024, 1, 10)
    ]

    daily = _compute_cohort_daily_pnl(
        cohorts,
        lifecycle,
        delta_hedge=False,
        hedge_spread_bps=0.0,
        equity_commission_per_share=0.0,
        option_commission_per_contract=0.0,
        delta_threshold=0.1,
    )
    assert daily.height == 2
    # Cash settlement is not a trade, so nothing is charged for leaving.
    assert daily.get_column("exit_cost_norm").to_list() == pytest.approx([0.0, 0.0])
    # Entry straddle is 6.0 + 4.0 = 10.0; settlement is max(101 - 100, 0) + 0 = 1.0. The short
    # position gains the whole 9.0 decline, recognised on the session quotes resume.
    assert daily.get_column("premium_pnl_norm").to_list() == pytest.approx([0.0, 0.9])


def test_expiration_settles_from_whichever_leg_the_chain_still_quotes(tmp_path: Path) -> None:
    """Intrinsic needs the underlying and the strike, and either leg carries the underlying.

    Requiring a pair at expiration dropped the settlement row whenever the chain stopped
    quoting the worthless leg, which is exactly when it tends to stop. The position then had no
    end and the run either truncated it or refused it, for a leg whose price is not read.
    """
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    chain = pl.read_parquet(raw_dir / "year=2024.parquet")
    single_leg = chain.filter(
        ~((pl.col("date") == date(2024, 1, 10)) & (pl.col("call_put") == "P"))
    )
    single_leg.write_parquet(raw_dir / "year=2024.parquet")
    cohorts = _select_cohorts(_predictions(), _contract_returns(), top_k=1)

    lifecycle = _load_option_lifecycle(cohorts, raw_dir)
    settled = lifecycle.filter(pl.col("cash_settled"))
    assert settled.get_column("date").to_list() == [date(2024, 1, 10)]
    # Settled at intrinsic from the underlying and the strike, not from the surviving quote.
    assert settled.get_column("call_mid").item() == pytest.approx(1.0)
    assert settled.get_column("put_mid").item() == pytest.approx(0.0)
    assert lifecycle.filter(pl.col("liquidated")).is_empty()


def test_a_position_that_can_be_neither_settled_nor_liquidated_stops_the_run(
    tmp_path: Path,
) -> None:
    """Rule 4: refuse, rather than drop the position and report the rest.

    Dropping it would publish a portfolio that silently excluded a position the strategy held,
    which is the same defect as reweighting the survivors, arrived at by omission.
    """
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    chain = pl.read_parquet(raw_dir / "year=2024.parquet")
    # The chain stops for every name before expiration, so there is no later session to book a
    # liquidation on and no expiration session to settle at.
    truncated = chain.filter(pl.col("date") < date(2024, 1, 10))
    truncated.write_parquet(raw_dir / "year=2024.parquet")
    cohorts = _select_cohorts(_predictions(), _contract_returns(), top_k=1)

    with pytest.raises(ValueError, match="neither settled at expiration nor liquidated"):
        _load_option_lifecycle(cohorts, raw_dir)


def test_a_corporate_action_does_not_reweight_the_names_beside_it(tmp_path: Path) -> None:
    """The cohort's weights are the ones the decision date set, whatever happens afterwards.

    Both names are selected at k=2 and A's contract is ended by a corporate action. Dropping A
    and handing its half to B would size B on information that did not exist when the cohort
    was formed, and would report a portfolio concentrated in the name that happened to survive.
    Each still carries a half.
    """
    raw_dir = tmp_path / "raw"
    _chain_with_a_terminated_contract(raw_dir)
    predictions, contract_returns = _two_names()

    cohorts = _select_cohorts(predictions, contract_returns, top_k=2, raw_options_dir=raw_dir)

    assert sorted(cohorts.get_column("symbol").to_list()) == ["A", "B"]
    assert cohorts.get_column("weight").to_list() == [0.5, 0.5]


def test_a_session_the_contract_is_not_quoted_on_is_not_a_defect(tmp_path: Path) -> None:
    """A corporate action can take a contract out of the chain for one session and give it back.

    PFE is the live case: its contract is quoted from entry to expiration except on the day its
    spinoff took effect. There is no quote to mark the position at on that session, but the
    contract is intact and nothing about the chain is broken, so this must not raise the way a
    chain that lost one leg of a session does.
    """
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    chain = pl.read_parquet(raw_dir / "year=2024.parquet")
    runner_up = chain.with_columns(symbol=pl.lit("B"))
    # Both legs of an interior session, so the contract is absent rather than half-quoted.
    interior = chain.filter(pl.col("date") != date(2024, 1, 9))
    pl.concat([interior, runner_up]).write_parquet(raw_dir / "year=2024.parquet")
    predictions, contract_returns = _two_names()

    cohorts = _select_cohorts(predictions, contract_returns, top_k=2, raw_options_dir=raw_dir)

    assert sorted(cohorts.get_column("symbol").to_list()) == ["A", "B"]


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

    decision_dates = option_decision_dates(
        study,
        [prediction.hash],
        prices=pl.DataFrame(
            {
                "symbol": ["A", "A"],
                "timestamp": [datetime(2024, 1, 10), datetime(2024, 1, 11)],
                "close": [100.0, 101.0],
            }
        ),
        signal={"universe_filter": "full"},
    )
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
    # Neither side may be the registered metric: on its own ten sessions the baseline is the
    # 0.1 it registered, and on the five it shares with the variant it is the winning half.
    assert row["baseline_sharpe"] != pytest.approx(0.1)
    assert row["baseline_sharpe"] > 0
    assert row["allocation_sharpe"] != pytest.approx(5.0)


def test_a_preview_pair_is_found_in_the_preview_namespace(tmp_path: Path) -> None:
    """A preview run of the allocation notebook compares preview backtests.

    Preview results live under `output_root/.preview/<case>`, which `Result.open` reaches only
    when asked. Opening the pair without asking raised `KeyError` on the documented preview path
    of `13_portfolio_management`, at the cell that used to read the catalog with the same flag.
    """
    from case_studies.utils.registry.registration import register_backtest_run

    study = _study(tmp_path)
    prediction = _prediction(study, execution_tier="preview")
    preview_root = study.storage_root("preview")
    sessions = [datetime(2024, 1, day) for day in range(2, 8)]
    strategy = {
        "execution_tier": "preview",
        "strategy": {"signal": {"method": "equal_weight_top_k", "top_k": 1}},
    }
    baseline_hash = register_backtest_run(
        "sp500_options",
        prediction.hash,
        strategy,
        stage="signal",
        returns=pl.DataFrame(
            {"timestamp": sessions, "return": [0.01, -0.02, 0.03, -0.01, 0.02, 0.01]}
        ),
        metrics={"sharpe": 0.1},
        case_dir=preview_root,
    )
    variant_hash = register_backtest_run(
        "sp500_options",
        prediction.hash,
        {**strategy, "strategy": {**strategy["strategy"], "allocation": {"method": "hrp"}}},
        stage="allocation",
        returns=pl.DataFrame({"timestamp": sessions[2:], "return": [0.04, -0.01, 0.03, 0.02]}),
        metrics={"sharpe": 1.0},
        case_dir=preview_root,
    )

    paired = paired_sharpe_on_common_support(
        study,
        pl.DataFrame({"baseline_hash": [baseline_hash], "backtest_hash": [variant_hash]}),
        include_preview=True,
    )

    assert paired.row(0, named=True)["n_periods"] == 4


def test_a_partial_resolved_set_cannot_snapshot_the_catalog_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`resolved_requests` avoids resolving twice; it does not narrow the population.

    A stale or partial set used to snapshot under the catalog's name and report complete, so
    every configuration it omitted left the comparison the population exists to define without
    anything saying so.
    """
    from case_studies.sp500_options import research_workflow

    monkeypatch.setattr(
        research_workflow,
        "OfficialPopulation",
        SimpleNamespace(create=lambda *a, **k: pytest.fail("a partial set reached the snapshot")),
    )
    catalog = pl.DataFrame(
        {
            "family": ["linear", "linear"],
            "label": ["ret_to_expiry", "ret_to_expiry"],
            "config_name": ["ridge_a1.0", "ridge_a10.0"],
        }
    )
    partial = (
        SimpleNamespace(
            family="linear",
            spec={
                "execution_tier": "canonical",
                "label": "ret_to_expiry",
                "config_name": "ridge_a1.0",
            },
        ),
    )

    with pytest.raises(ValueError, match="do not match the declared catalog"):
        research_workflow.snapshot_official_model_catalog(
            SimpleNamespace(),
            catalog,
            population_name="sp500-options-linear-validation-v1",
            resolved_requests=partial,
        )


def test_the_liquid_filter_moves_the_decision_to_an_earlier_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A week whose Friday the filter empties is entered on the Thursday.

    `resolve_short_straddle_decisions` applies the declared universe filter before ranking, so
    the frame the engine resolves `weekly_friday` from is the filtered one. Resolving the
    displayed schedule from the unfiltered predictions names a Friday nothing enters on.
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
            "timestamp": [datetime(2024, 1, 11), datetime(2024, 1, 12)],
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

    # Five symbols each session, so the tightest one clears the 0.20 quantile on its own. A is
    # tightest on Thursday and widest on Friday, which is the only difference between the days.
    universe = ["A", "B", "C", "D", "E"]
    prices = pl.DataFrame(
        {
            "symbol": universe * 2,
            "timestamp": [datetime(2024, 1, 11)] * 5 + [datetime(2024, 1, 12)] * 5,
            "close": [100.0] * 10,
            "instr_rel_spread": [0.01, 0.02, 0.03, 0.04, 0.05, 0.09, 0.02, 0.03, 0.04, 0.05],
        }
    )

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    _week_of_candidates([thursday, friday]).write_parquet(labels_dir / "contract_returns.parquet")
    monkeypatch.setattr(
        research_workflow, "option_data_paths", lambda: (labels_dir, tmp_path / "raw")
    )

    decision_dates = option_decision_dates(
        study,
        [prediction.hash],
        prices=prices,
        signal={"universe_filter": "liquid"},
    )

    assert decision_dates.to_list() == [thursday]
    assert option_trade_calendar(decision_dates).get_column("decision_date").to_list() == [thursday]


def test_an_unaccounted_lifecycle_refuses_before_any_conformal_width_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal has to come first, or a run that cannot register still writes calibration.

    `compute_holdout_conformal_widths(..., write=True)` publishes widths keyed by the holdout
    prediction. A supplied lifecycle the identity does not account for makes the run unable to
    register, so reaching the widths write first leaves an artifact behind that no registered
    backtest explains.
    """
    import json

    from case_studies.research import strategy as strategy_module
    from case_studies.sp500_options import _htm_backtest, research_workflow
    from case_studies.utils import cv_window

    class _WidthsWritten(Exception):
        pass

    study = _study(tmp_path)
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
    allocation = {"method": "conformal_weighted", "alpha": 0.2, "min_calibration_n": 2}

    validation = _prediction(study)
    validation_decision = publish_short_straddle_decisions(validation, prices=prices, signal=signal)
    locked_spec = study.strategy(
        prediction=validation,
        signal=signal,
        decision=validation_decision,
        allocation=allocation,
    ).resolve(prices=prices)
    # The lock stores the strategy, not the runtime object the projection drops from both sides.
    locked_spec.pop("_runtime_backtest_config", None)

    holdout = _prediction(study, split="holdout")
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "INSERT INTO research_locks (lock_hash, lock_json, state, created_at) "
            "VALUES (?, ?, 'LOCKED', '2024-01-06T00:00:00Z')",
            (
                "lockhash00000001",
                json.dumps(
                    {
                        "holdout_training_hash": holdout.registry_record()["training_hash"],
                        "checkpoint_kind": "final",
                        "checkpoint_value": None,
                        "prediction_hash": validation.hash,
                        "strategy_spec": locked_spec,
                    }
                ),
            ),
        )
        db.commit()
    holdout_decision = publish_short_straddle_decisions(holdout, prices=prices, signal=signal)
    lifecycle = _load_option_lifecycle(holdout_decision.load(), raw_dir)
    lifecycle_source = option_source_identity(labels_dir, raw_dir)["raw_lifecycle"]

    def _widths(*args, **kwargs):
        raise _WidthsWritten

    monkeypatch.setattr(strategy_module, "compute_holdout_conformal_widths", _widths)
    # No reviewed embargo is registered for this case study and label; the ordering under test
    # is the engine's, not that table's.
    monkeypatch.setattr(strategy_module, "holdout_conformal_embargo_steps", lambda *a, **k: 0)
    monkeypatch.setattr(strategy_module, "load_backtest_prices_for", lambda *a, **k: prices)
    strategy = study.strategy(
        prediction=holdout,
        signal=signal,
        decision=holdout_decision,
        allocation=allocation,
    )

    with pytest.raises(ValueError, match="declare the raw files"):
        strategy.run(option_lifecycle=lifecycle)

    # The widths branch is live in this setup, so the refusal above ran ahead of it rather
    # than the branch simply never being reached.
    with pytest.raises(_WidthsWritten):
        strategy.run(option_lifecycle=lifecycle, option_lifecycle_source=lifecycle_source)


def test_the_research_workflow_imports_without_torch() -> None:
    """The `test-unit` job installs no torch, and this module imports the workflow.

    `case_studies/utils/deep_learning.py` imports torch at module scope, so a module-scope
    import of anything from it puts torch on the critical path of a required per-commit gate
    that excludes torch deliberately - 25 test files sit behind it and installing it would
    cost gigabytes on every commit. The failure is a collection error naming this file, which
    says nothing about which import added the edge, so the property is pinned here where the
    edge would be introduced.
    """
    import builtins
    import importlib

    real_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    module = "case_studies.sp500_options.research_workflow"
    saved = {key: value for key, value in sys.modules.items() if key.startswith("torch")}
    for key in saved:
        del sys.modules[key]
    del sys.modules[module]
    builtins.__import__ = without_torch
    try:
        importlib.import_module(module)
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)


def _exit_costs(raw_dir: Path, *, exit_at_max_days: int) -> list[float]:
    cohorts = _select_cohorts(_predictions(), _contract_returns(), top_k=1)
    daily = _compute_cohort_daily_pnl(
        cohorts,
        _load_option_lifecycle(cohorts, raw_dir),
        delta_hedge=False,
        hedge_spread_bps=0.0,
        equity_commission_per_share=0.0,
        option_commission_per_contract=0.0,
        delta_threshold=0.10,
        option_spread_fraction=1.0,
        exit_at_max_days=exit_at_max_days,
    )
    return daily.get_column("exit_cost_norm").to_list()


def test_a_round_trip_that_reaches_expiration_pays_no_exit_spread(tmp_path: Path) -> None:
    """Cash settlement is not a market exit, so there is no spread to cross.

    The expiration quote is exempt from the crossed-quote check because nothing reads
    it. Charging an ask-based exit cost there would read it, and on a crossed session
    ask - mid is negative, which books the exit as a gain.
    """
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)
    _cross_the_call_quote(raw_dir, date(2024, 1, 10))

    # The window covers the whole lifecycle, so the last held session is the expiry.
    assert _exit_costs(raw_dir, exit_at_max_days=5) == pytest.approx([0.0, 0.0, 0.0])


def test_a_round_trip_that_exits_before_expiration_still_pays_the_spread(
    tmp_path: Path,
) -> None:
    """The exemption is the expiration session and nothing wider."""
    raw_dir = tmp_path / "raw"
    _write_raw_options(raw_dir)

    costs = _exit_costs(raw_dir, exit_at_max_days=1)

    assert costs[-1] > 0.0
    assert costs[:-1] == pytest.approx([0.0] * (len(costs) - 1))
