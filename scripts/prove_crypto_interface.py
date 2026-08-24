"""Run the reduced real-data proof for the crypto research interface."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import polars as pl

from case_studies.crypto_perps_funding.research_workflow import (
    model_request_catalog,
    model_requests,
    open_study,
    plan_model_catalog,
    publish_exploratory_positions,
    target_positions,
)
from case_studies.research import (
    CandidateSet,
    CausalResult,
    DecisionArtifact,
    OfficialPopulation,
    Study,
)
from case_studies.research.catalog import CATALOG_VERSION, RESERVED_COLUMNS
from case_studies.research.decisions import StateTransitionPolicy
from case_studies.research.execution import run_backtests
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import load_backtest_prices_for
from utils.paths import REPO_ROOT

CASE_STUDY = "crypto_perps_funding"
# The ETF catalog snapshot brackets the proof, so pointing this at a second
# checkout that only exists on one machine makes the whole proof unrunnable
# anywhere else. Default to this checkout and let the environment override it.
ETF_RELEASE_ROOT = Path(os.environ.get("ML4T_ETF_RELEASE_ROOT", REPO_ROOT))
ETF_IDENTITY_COLUMNS = (
    "family",
    "config_name",
    "label",
    "checkpoint_kind",
    "checkpoint_value",
    "training_hash",
    "prediction_hash",
)


def _etf_catalog_snapshot() -> tuple[int, str] | None:
    """Digest a second case study's catalog, to prove the crypto run leaves it alone.

    Returns None when the resolved root holds no published ETF predictions. The
    check is only meaningful against a populated catalog, and a checkout that has
    not run the ETF stages has nothing to compare - failing there would make the
    whole crypto proof unrunnable on every machine but one. Point
    ML4T_ETF_RELEASE_ROOT at a populated checkout to exercise it.
    """
    study = Study.open("etfs", release_root=ETF_RELEASE_ROOT)
    catalog = study.predictions.table(include_preview=False)
    if catalog.is_empty():
        print(
            f"[note] no published ETF predictions under {ETF_RELEASE_ROOT}; "
            "skipping the cross-study catalog isolation check",
            flush=True,
        )
        return None
    missing = set(RESERVED_COLUMNS) - set(catalog.columns)
    if missing:
        raise RuntimeError(f"ETF compatibility catalog is missing columns: {sorted(missing)}")
    identities = catalog.select(*ETF_IDENTITY_COLUMNS).sort(*ETF_IDENTITY_COLUMNS)
    return identities.height, value_digest(identities)


def _stage(name: str) -> None:
    """Mark a phase boundary so a long phase is distinguishable from a hang in the log."""
    print(f"[{datetime.now(UTC).isoformat()}] {name}", flush=True)


def prove(
    workspace: Path,
    *,
    canonical_dml: bool = False,
    require_notebook_results: bool = False,
) -> dict[str, object]:
    assert CATALOG_VERSION == 1
    assert {
        "training_hash",
        "prediction_hash",
        "checkpoint_kind",
        "checkpoint_value",
        "execution_tier",
        "complete",
    } <= set(RESERVED_COLUMNS)
    _stage("etf catalog snapshot")
    etf_catalog_before = _etf_catalog_snapshot()
    study = open_study(execution_tier="preview", workspace=workspace)
    request_catalog = model_request_catalog(
        "tabular_dl", labels=("fwd_dir_8h",), config_prefix="tabm"
    ).filter(pl.col("config_name") == "tabm_s")
    overrides = {"batch_size": 32, "class_weight": "balanced", "device": "cuda"}
    preview_reductions = {
        "checkpoint_interval": 1,
        "folds": [0, 1],
        "max_symbols": 6,
        "n_epochs": 2,
    }
    resolved = model_requests(
        study,
        request_catalog,
        execution_tier="preview",
        overrides=overrides,
        preview_reductions=preview_reductions,
    )[0].resolve()
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

    # The notebooks plan the population before fitting; planning must resolve the same
    # identities the direct request path resolves, or the frozen population is a fiction.
    plan = plan_model_catalog(
        study,
        request_catalog,
        execution_tier="preview",
        overrides=overrides,
        preview_reductions=preview_reductions,
    )
    assert plan.expected_training_hashes == (resolved.identity,)
    planned_prediction_hashes = plan.expected_prediction_hashes
    assert len(planned_prediction_hashes) == 2
    assert [member.checkpoint_value for member in plan.members] == [1, 2]

    notebook_prediction_digests: dict[str, str] = {}
    if require_notebook_results:
        projected_hashes = set(planned_prediction_hashes)
        existing_rows = study.predictions.table(include_preview=True).filter(
            pl.col("training_hash") == resolved.identity
        )
        assert set(existing_rows.get_column("prediction_hash")) == projected_hashes
        assert existing_rows.get_column("complete").all()
        preview_root = study.storage_root("preview")
        notebook_prediction_digests = {
            prediction_hash: value_digest(
                pl.read_parquet(
                    preview_root
                    / "run_log"
                    / "predictions"
                    / prediction_hash
                    / "predictions.parquet"
                )
            )
            for prediction_hash in projected_hashes
        }

    _stage("tabular_dl fit")
    run = resolved.run()
    assert len(run.predictions) == 2
    assert tuple(item.hash for item in run.predictions) == planned_prediction_hashes
    final_prediction = run.predictions[-1]
    prediction_frame = final_prediction.load()
    expected = resolved._context.expected_keys
    actual = prediction_frame.select("symbol", "timestamp", "fold")
    assert actual.height == actual.n_unique(["symbol", "timestamp", "fold"])
    assert actual.join(expected, on=["symbol", "timestamp", "fold"], how="anti").is_empty()
    assert expected.join(actual, on=["symbol", "timestamp", "fold"], how="anti").is_empty()
    assert prediction_frame.get_column("fold").n_unique() == 2
    for fold_frame in prediction_frame.partition_by("fold"):
        timestamps = fold_frame.get_column("timestamp").unique().sort()
        assert timestamps.diff().drop_nulls().unique().to_list() == [timedelta(hours=8)]
    assert prediction_frame.get_column("prediction").is_finite().all()

    restarted = (
        model_requests(
            study,
            request_catalog,
            execution_tier="preview",
            overrides=overrides,
            preview_reductions=preview_reductions,
        )[0]
        .resolve()
        .run()
    )
    assert restarted.training.hash == run.training.hash
    assert tuple(item.hash for item in restarted.predictions) == tuple(
        item.hash for item in run.predictions
    )
    assert value_digest(restarted.predictions[-1].load()) == value_digest(prediction_frame)
    if require_notebook_results:
        assert notebook_prediction_digests == {
            prediction.hash: value_digest(prediction.load()) for prediction in run.predictions
        }

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
    notebook_causal = None
    if require_notebook_results:
        notebook_causal = CausalResult.one(
            study,
            label="fwd_ret_8h",
            execution_tier="preview",
        )
        assert notebook_causal.hash == resolved_causal.identity
        assert notebook_causal.spec == resolved_causal.spec
    causal_result = resolved_causal.run()
    assert causal_result.complete
    assert causal.resolve().run().hash == causal_result.hash
    if notebook_causal is not None:
        assert causal_result.metrics == notebook_causal.metrics

    preview_population = resolved_causal.spec["computation"]["analysis_population"]
    canonical_summary: dict[str, object] = {}
    if canonical_dml:
        _stage("canonical DML on the full declared population")
        canonical_study = open_study(execution_tier="canonical", workspace=workspace)
        canonical_causal = canonical_study.causal(
            method="dml",
            label="fwd_ret_8h",
            execution_tier="canonical",
        ).resolve()
        canonical_population = canonical_causal.spec["computation"]["analysis_population"]
        assert canonical_population["max_samples"] == 0
        assert "preview_reductions" not in canonical_causal.spec["computation"]
        assert canonical_population["n_rows"] > preview_population["n_rows"]
        assert (
            canonical_causal.spec["computation"]["feature_artifacts"]
            == resolved_causal.spec["computation"]["feature_artifacts"]
        )
        canonical_causal_result = canonical_causal.run()
        assert canonical_causal_result.complete
        assert canonical_causal_result.execution_tier == "canonical"
        assert canonical_causal.run().hash == canonical_causal_result.hash
        canonical_summary = {
            "canonical_causal_hash": canonical_causal_result.hash,
            "canonical_causal_rows": canonical_population["n_rows"],
        }

    first_fold = prediction_frame.get_column("fold").min()
    first_fold_timestamps = (
        prediction_frame.filter(pl.col("fold") == first_fold)
        .get_column("timestamp")
        .unique()
        .sort()
    )
    omitted_timestamp = first_fold_timestamps[first_fold_timestamps.len() // 2]
    decision_predictions = prediction_frame.filter(pl.col("timestamp") != omitted_timestamp)
    positions = target_positions(decision_predictions)
    state_policy = StateTransitionPolicy(fold_boundary="liquidate", temporal_gap="reset")
    decision = publish_exploratory_positions(study, final_prediction.hash, decision_predictions)
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
    weights = pl.read_parquet(backtest_dir / "weights.parquet")
    transition_times = set(
        weights.filter(pl.col("_state_transition"))
        .select(
            pl.col("timestamp")
            .dt.replace_time_zone(None)
            .cast(pl.Datetime("ms"))
            .alias("timestamp")
        )
        .get_column("timestamp")
        .unique()
        .to_list()
    )
    decision_timeline = (
        decision.load()
        .select(
            pl.col("timestamp")
            .dt.replace_time_zone(None)
            .cast(pl.Datetime("ms"))
            .alias("timestamp"),
            "fold",
        )
        .unique()
        .sort("timestamp")
    )
    fold_boundaries = set(
        decision_timeline.filter(pl.col("fold") != pl.col("fold").shift(1))
        .get_column("timestamp")
        .to_list()
    )
    timeline = decision_timeline.get_column("timestamp").to_list()
    gaps = [
        (previous, current)
        for previous, current in pairwise(timeline)
        if current - previous > timedelta(hours=8)
    ]
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
    assert fold_boundaries
    assert gaps
    assert fold_boundaries <= transition_times
    normalized_omitted_timestamp = (
        pl.Series("timestamp", [omitted_timestamp])
        .dt.replace_time_zone(None)
        .cast(pl.Datetime("ms"))
        .item()
    )
    assert normalized_omitted_timestamp in transition_times
    assert all(
        any(previous < transition <= current for transition in transition_times)
        for previous, current in gaps
    )
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

    _stage("etf catalog re-check")
    etf_catalog_after = _etf_catalog_snapshot()
    assert etf_catalog_after == etf_catalog_before

    return {
        "backtest_hash": backtest.hash,
        "causal_hash": causal_result.hash,
        "checkpoint_prediction_hashes": [item.hash for item in run.predictions],
        "eligible_rows": expected.height,
        "etf_catalog_identity_digest": etf_catalog_after[1] if etf_catalog_after else None,
        "etf_catalog_rows": etf_catalog_after[0] if etf_catalog_after else None,
        "funding_events": int(funding_events),
        "funding_pnl": float(funding_pnl),
        "funding_settlements": int(funding_settlements),
        "num_fills": fills.height,
        "num_trades": int(num_trades),
        "notebook_equivalence": require_notebook_results,
        "omitted_gap_timestamp": str(omitted_timestamp),
        "preview_causal_rows": preview_population["n_rows"],
        "state_transition_timestamps": len(transition_times),
        "training_hash": run.training.hash,
        "workspace": str(study.storage_root("preview")),
        **canonical_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument(
        "--canonical-dml",
        action="store_true",
        help="also execute the full canonical DML population in the case-study artifact root",
    )
    parser.add_argument(
        "--require-notebook-results",
        action="store_true",
        help="fail unless matching reduced notebook artifacts already exist in the workspace",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prove(
                args.workspace,
                canonical_dml=args.canonical_dml,
                require_notebook_results=args.require_notebook_results,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
