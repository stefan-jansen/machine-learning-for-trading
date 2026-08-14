#!/usr/bin/env python3
"""Prove the typed S&P 500 option path on a reduced real prediction set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from case_studies.research import OfficialPopulation
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
    selected_prediction,
)
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import load_backtest_prices_for
from case_studies.utils.backtest_runner import apply_universe_filter, normalize_prediction_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prediction-hash", required=True)
    parser.add_argument("--top-k", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = open_study(execution_tier="preview", workspace=args.workspace)
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
    strategy = study.strategy(prediction=prediction, signal=signal, decision=decision)
    result = strategy.run(prices=prices, option_lifecycle=lifecycle)
    replay = strategy.run(prices=prices, option_lifecycle=lifecycle)
    if replay.hash != result.hash:
        raise RuntimeError("option backtest replay changed identity")
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
