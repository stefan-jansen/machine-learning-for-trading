"""Run the reduced real-data proof for the crypto research interface."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import polars as pl

from case_studies.crypto_perps_funding.research_workflow import (
    model_request_catalog,
    open_study,
    publish_exploratory_positions,
    resolve_model_requests,
    target_positions,
)
from case_studies.research import CandidateSet, DecisionArtifact, OfficialPopulation, Study
from case_studies.research.catalog import CATALOG_VERSION, RESERVED_COLUMNS
from case_studies.research.decisions import StateTransitionPolicy
from case_studies.research.execution import run_backtests
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import load_backtest_prices_for
from utils.paths import REPO_ROOT

CASE_STUDY = "crypto_perps_funding"


def prove(workspace: Path) -> dict[str, object]:
    assert CATALOG_VERSION == 1
    assert {
        "training_hash",
        "prediction_hash",
        "checkpoint_kind",
        "checkpoint_value",
        "execution_tier",
        "complete",
    } <= set(RESERVED_COLUMNS)
    study = open_study(execution_tier="preview", workspace=workspace)
    request_catalog = model_request_catalog(
        "tabular_dl", labels=("fwd_dir_8h",), config_prefix="tabm"
    ).filter(pl.col("config_name") == "tabm_s")
    overrides = {"class_weight": "balanced", "device": "cuda"}
    preview_reductions = {
        "checkpoint_interval": 1,
        "folds": [0],
        "max_symbols": 6,
        "n_epochs": 2,
    }
    resolved = resolve_model_requests(
        study,
        request_catalog,
        execution_tier="preview",
        overrides=overrides,
        preview_reductions=preview_reductions,
    )[0]
    computation = resolved.spec["computation"]
    task = computation["task"]
    assert task["type"] == "classification"
    assert task["continuous_eval_label"] == "fwd_ret_8h"
    assert task["imbalance"]["method"] == "balanced"
    assert task["metrics"] == [
        "ic",
        "auc_roc",
        "log_loss",
        "accuracy",
        "balanced_accuracy",
    ]
    assert [item["value"] for item in computation["checkpoint_schedule"]] == [1, 2]

    run = resolved.run()
    assert len(run.predictions) == 2
    final_prediction = run.predictions[-1]
    prediction_frame = final_prediction.load()
    expected = resolved._context.expected_keys
    actual = prediction_frame.select("symbol", "timestamp", "fold")
    assert actual.height == actual.n_unique(["symbol", "timestamp", "fold"])
    assert actual.join(expected, on=["symbol", "timestamp", "fold"], how="anti").is_empty()
    assert expected.join(actual, on=["symbol", "timestamp", "fold"], how="anti").is_empty()
    timestamps = prediction_frame.get_column("timestamp").unique().sort()
    observed_steps = timestamps.diff().drop_nulls().unique().to_list()
    assert observed_steps == [timedelta(hours=8)]
    assert prediction_frame.get_column("prediction").is_finite().all()

    restarted = resolve_model_requests(
        study,
        request_catalog,
        execution_tier="preview",
        overrides=overrides,
        preview_reductions=preview_reductions,
    )[0].run()
    assert restarted.training.hash == run.training.hash
    assert tuple(item.hash for item in restarted.predictions) == tuple(
        item.hash for item in run.predictions
    )
    assert value_digest(restarted.predictions[-1].load()) == value_digest(prediction_frame)

    preview_catalog = study.predictions.table(include_preview=True)
    canonical_catalog = study.predictions.table(include_preview=False)
    assert final_prediction.hash in preview_catalog.get_column("prediction_hash").to_list()
    if not canonical_catalog.is_empty():
        assert (
            final_prediction.hash not in canonical_catalog.get_column("prediction_hash").to_list()
        )
    try:
        OfficialPopulation.create(
            study,
            name="preview-must-not-enter-official-population",
            member_kind="prediction",
            members=[final_prediction.hash],
        )
    except ValueError as error:
        assert "preview" in str(error)
    else:
        raise AssertionError("preview prediction entered an official population")

    causal = study.causal(
        method="dml",
        label="fwd_ret_8h",
        execution_tier="preview",
        preview_reductions={
            "max_samples": 1200,
            "max_symbols": 6,
            "n_folds": 2,
            "n_placebo": 10,
        },
        overrides={"nuisance_params": {"max_iter": 10}},
    )
    resolved_causal = causal.resolve()
    assert resolved_causal.spec["computation"]["refutation"]["temporal_gap_policy"] == "reset"
    causal_result = resolved_causal.run()
    assert causal_result.complete
    assert causal.resolve().run().hash == causal_result.hash

    positions = target_positions(prediction_frame)
    state_policy = StateTransitionPolicy(fold_boundary="liquidate", temporal_gap="reset")
    decision = publish_exploratory_positions(study, final_prediction.hash, prediction_frame)
    assert decision.load().sort("timestamp", "symbol").equals(positions)
    prices = load_backtest_prices_for(
        CASE_STUDY,
        label="fwd_dir_8h",
        split="validation",
    )
    selected = preview_catalog.filter(pl.col("prediction_hash") == final_prediction.hash)
    execution = run_backtests(
        study,
        predictions=selected,
        signal={"method": "precomputed_positions", "top_k": 1},
        decision=decision,
        prices=prices,
    )
    backtest = execution.results[0]
    assert backtest.spec()["decision_artifact"]["state_transition_policy"] == {
        "fold_boundary": "liquidate",
        "temporal_gap": "reset",
    }
    backtest_dir = backtest.root / "run_log" / "backtest" / backtest.hash
    fills = pl.read_parquet(backtest_dir / "fills.parquet")
    portfolio_state = pl.read_parquet(backtest_dir / "portfolio_state.parquet")
    with sqlite3.connect(backtest.root / "run_log" / "registry.db") as db:
        metrics = db.execute(
            "SELECT num_trades, funding_pnl, funding_events, funding_settlements "
            "FROM backtest_metrics WHERE backtest_hash = ?",
            (backtest.hash,),
        ).fetchone()
    assert metrics is not None
    num_trades, funding_pnl, funding_events, funding_settlements = metrics
    assert fills.height > 0
    assert portfolio_state.filter(pl.col("gross_exposure") > 0).height > 0
    assert num_trades > 0
    assert funding_pnl != 0
    assert funding_events > 0
    assert funding_settlements > 0
    try:
        CandidateSet.create(study, "exploratory-must-not-be-canonical", [backtest])
    except ValueError as error:
        assert "preview" in str(error) or "exploratory" in str(error)
    else:
        raise AssertionError("exploratory decision backtest entered a candidate set")
    try:
        OfficialPopulation.create(
            study,
            name="exploratory-must-not-be-canonical",
            member_kind="backtest",
            members=[backtest.hash],
        )
    except ValueError as error:
        assert "preview" in str(error) or "exploratory" in str(error)
    else:
        raise AssertionError("exploratory decision backtest entered an official population")

    try:
        DecisionArtifact.publish(
            study,
            kind="target_positions",
            decisions=positions,
            prediction_hashes=[final_prediction.hash],
            parameters={"long_count": 1, "short_count": 1},
            state_transition_policy=state_policy,
            canonical=True,
            source_identity={
                "module": "scripts.prove_crypto_interface",
                "source_digest": "undisclosed",
                "declared_inputs": {},
                "determinism": {"deterministic": True},
                "clean_replay_digest": value_digest(positions),
            },
        )
    except (KeyError, ValueError) as error:
        assert "canonical decision" in str(error) or "Unknown result hash" in str(error)
    else:
        raise AssertionError("undisclosed decision generation was promoted")

    return {
        "backtest_hash": backtest.hash,
        "causal_hash": causal_result.hash,
        "checkpoint_prediction_hashes": [item.hash for item in run.predictions],
        "eligible_rows": expected.height,
        "funding_events": int(funding_events),
        "funding_pnl": float(funding_pnl),
        "funding_settlements": int(funding_settlements),
        "num_fills": fills.height,
        "num_trades": int(num_trades),
        "training_hash": run.training.hash,
        "workspace": str(study.storage_root("preview")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    print(json.dumps(prove(args.workspace), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
