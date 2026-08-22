#!/usr/bin/env python3
"""Prove the typed S&P 500 option path on a reduced real prediction set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from case_studies.research import OfficialPopulation, PredictionResult, ResolvedSpec, Study
from case_studies.sp500_options._htm_backtest import (
    _load_option_lifecycle,
    option_data_paths,
    run_htm_daily_mtm,
)
from case_studies.sp500_options.research_workflow import (
    _clean_replay_digests,
    open_study,
    publish_short_straddle_decisions,
    resolve_short_straddle_decisions,
    run_official_backtest_requests,
    selected_prediction,
    strategy_request_frame,
)
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import load_backtest_prices_for
from case_studies.utils.backtest_runner import apply_universe_filter, normalize_prediction_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prediction-hash")
    source.add_argument("--source-prediction-hash")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--max-sessions", type=int, default=65)
    return parser.parse_args()


def _seed_real_preview_prediction(
    study: Study,
    *,
    source_prediction_hash: str,
    max_symbols: int,
    max_sessions: int,
) -> PredictionResult:
    if max_symbols < 2 or max_sessions < 5:
        raise ValueError("the real preview fixture requires at least two symbols and five sessions")
    source_path = (
        study.release_root
        / "case_studies"
        / "sp500_options"
        / "run_log"
        / "predictions"
        / source_prediction_hash
        / "predictions.parquet"
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"released source prediction is missing: {source_path}")
    source = pl.read_parquet(source_path)
    required = {"symbol", "timestamp", "fold", "prediction", "actual"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"released source prediction is missing columns: {sorted(missing)}")
    fold_value = source.get_column("fold").min()
    if not isinstance(fold_value, int):
        raise TypeError("released source prediction fold values must be integers")
    fold = fold_value
    folded = source.filter(pl.col("fold") == fold)
    symbols = (
        folded.group_by("symbol")
        .len()
        .sort(["len", "symbol"], descending=[True, False])
        .head(max_symbols)
        .get_column("symbol")
    )
    sessions = folded.get_column("timestamp").unique().sort().head(max_sessions)
    preview = (
        folded.filter(
            pl.col("symbol").is_in(symbols.implode()),
            pl.col("timestamp").is_in(sessions.implode()),
        )
        .select("symbol", "timestamp", "fold", "prediction", "actual")
        .sort("timestamp", "symbol")
    )
    if preview.get_column("symbol").n_unique() != max_symbols:
        raise ValueError("real preview fixture did not retain the declared symbol population")
    if preview.get_column("timestamp").n_unique() != max_sessions:
        raise ValueError("real preview fixture did not retain the declared session population")
    expected_keys = preview.select("symbol", "timestamp", "fold")
    reductions = {
        "source_prediction_hash": source_prediction_hash,
        "folds": [fold],
        "max_symbols": max_symbols,
        "max_sessions": max_sessions,
        "date_start": str(preview.get_column("timestamp").min()),
        "date_end": str(preview.get_column("timestamp").max()),
    }
    spec = ResolvedSpec.create(
        family="options_interface_fixture",
        label="ret_to_expiry",
        seed=0,
        computation={
            "model": {
                "class": "released_prediction_subset",
                "source_prediction_hash": source_prediction_hash,
            },
            "cv": {"identity": "released-validation-fold-subset", "request": {"folds": [fold]}},
            "checkpoint_schedule": [{"kind": "final", "value": None}],
            "expected_prediction_keys": {
                "digest": value_digest(expected_keys),
                "n_rows": expected_keys.height,
                "n_folds": 1,
            },
            "input_data_spec": {
                "source_prediction_hash": source_prediction_hash,
                "source_prediction_digest": value_digest(source),
            },
            "preview_reductions": reductions,
        },
        provenance={"purpose": "reduced-real-options-interface-proof"},
        config_name=f"released-{source_prediction_hash}",
        execution_tier="preview",
    ).as_dict()
    training = study.results.register_training(spec, execution_tier="preview")
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=preview,
        expected_keys=expected_keys,
    )
    if not prediction.complete:
        raise RuntimeError("real preview prediction fixture is incomplete")
    return prediction


def main() -> None:
    args = parse_args()
    study = open_study(execution_tier="preview", workspace=args.workspace)
    if args.source_prediction_hash is not None:
        prediction = _seed_real_preview_prediction(
            study,
            source_prediction_hash=args.source_prediction_hash,
            max_symbols=args.max_symbols,
            max_sessions=args.max_sessions,
        )
    else:
        catalog = study.predictions.table(include_preview=True).filter(
            pl.col("prediction_hash") == args.prediction_hash
        )
        if catalog.height != 1 or not catalog.item(0, "complete"):
            raise ValueError("proof prediction is absent, ambiguous, or incomplete")
        if catalog.item(0, "execution_tier") != "preview":
            raise ValueError("the reduced proof requires a preview prediction")
        prediction = selected_prediction(study, catalog.row(0, named=True))
    signal = {
        "method": "equal_weight_top_k",
        "top_k": args.top_k,
        "universe_filter": "liquid",
    }
    prices = load_backtest_prices_for(
        "sp500_options",
        "ret_to_expiry",
        split="validation",
    )
    prepared_decisions = resolve_short_straddle_decisions(
        prediction,
        prices=prices,
        signal=signal,
    )
    local_decision_digest = value_digest(prepared_decisions)
    clean_replay = _clean_replay_digests(
        study,
        [
            {
                "request_name": "reduced-liquid-option-proof",
                "prediction_hash": prediction.hash,
                "label": "ret_to_expiry",
                "signal": signal,
                "allocation": None,
                "execution_tier": prediction.execution_tier,
            }
        ],
    )
    if clean_replay != {"reduced-liquid-option-proof": local_decision_digest}:
        raise RuntimeError("clean-process option decisions differ from the prepared proof")
    decision = publish_short_straddle_decisions(
        prediction,
        prices=prices,
        signal=signal,
        canonical=False,
    )
    labels_dir, raw_options_dir = option_data_paths()
    decisions = decision.load()
    lifecycle = _load_option_lifecycle(decisions, raw_options_dir)
    raw_predictions = normalize_prediction_columns(prediction.load())
    filtered_predictions = apply_universe_filter(
        raw_predictions,
        prices,
        "sp500_options",
        signal,
        prediction_hash=prediction.hash,
    )
    direct = run_htm_daily_mtm(
        "sp500_options",
        filtered_predictions,
        labels_dir,
        raw_options_dir,
        top_k=args.top_k,
        option_lifecycle=lifecycle,
    )
    typed = run_htm_daily_mtm(
        "sp500_options",
        filtered_predictions,
        labels_dir,
        raw_options_dir,
        top_k=args.top_k,
        decisions=decisions,
        option_lifecycle=lifecycle,
    )
    if not typed["daily_returns"].equals(direct["daily_returns"]):
        raise RuntimeError("typed and direct option returns differ")
    requests = strategy_request_frame(
        [
            {
                "request_name": "reduced-liquid-option-proof",
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
    replay_execution = run_official_backtest_requests(study, requests, population_name=None)
    result = execution.results[0]
    replay = replay_execution.results[0]
    if replay.hash != result.hash:
        raise RuntimeError("option backtest replay changed identity")
    if execution.catalog_rows.item(0, "decision_hash") != decision.hash:
        raise RuntimeError("reader-facing option execution changed the decision identity")
    try:
        OfficialPopulation.create(
            study,
            name="sp500-options-preview-must-fail",
            member_kind="backtest",
            members=[result.hash],
        )
    except ValueError as error:
        if "preview" not in str(error):
            raise
    else:
        raise RuntimeError("preview option result entered an official population")
    expiry_rows = lifecycle.filter(pl.col("cash_settled"))
    report = {
        "prediction_hash": prediction.hash,
        "decision_hash": decision.hash,
        "backtest_hash": result.hash,
        "decision_rows": decisions.height,
        "decision_dates": decisions.get_column("timestamp").n_unique(),
        "contracts": decisions.n_unique(["symbol", "strike", "expiration"]),
        "lifecycle_rows": lifecycle.height,
        "settlement_rows": expiry_rows.height,
        "return_rows": typed["daily_returns"].height,
        "replay_hash": replay.hash,
        "typed_direct_equal": True,
        "clean_decision_replay": True,
        "preview_excluded": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("OPTION_GATE_REPORT=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
