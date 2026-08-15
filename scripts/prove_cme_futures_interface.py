"""Run the reduced real-data proof for the CME futures research interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import polars as pl

from case_studies.cme_futures.research_workflow import (
    load_futures_price_path,
    model_request_catalog,
    open_study,
    publish_product_weights,
    resolve_model_requests,
    run_resolved_model_requests,
)
from case_studies.research import CandidateSet, OfficialPopulation
from case_studies.research.execution import run_backtests
from case_studies.utils.artifact_digest import value_digest

CASE_STUDY = "cme_futures"


def _product_keys(frame: pl.DataFrame) -> pl.DataFrame:
    entity_columns = [column for column in ("symbol", "product") if column in frame.columns]
    fold_columns = [column for column in ("fold", "fold_id") if column in frame.columns]
    if len(entity_columns) != 1 or len(fold_columns) != 1:
        raise ValueError("prediction eligibility requires one entity key and one fold key")
    result = frame.select(entity_columns[0], "timestamp", fold_columns[0])
    return result.rename({entity_columns[0]: "product", fold_columns[0]: "fold"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fitted_state_rows(study, training_hash: str) -> list[tuple]:
    root = study.storage_root("preview")
    with sqlite3.connect(root / "run_log" / "registry.db") as db:
        rows = db.execute(
            "SELECT fold_id, fitted_state_path, fitted_state_digest, "
            "prediction_shard_path, prediction_shard_digest "
            "FROM candidate_fold_completions WHERE training_hash = ? ORDER BY fold_id",
            (training_hash,),
        ).fetchall()
    if not rows:
        raise AssertionError("reduced model run persisted no fitted-state rows")
    for _, fitted_path, fitted_digest, shard_path, shard_digest in rows:
        fitted = root / fitted_path
        shard = root / shard_path
        assert fitted.is_file() and _sha256(fitted) == fitted_digest
        assert shard.is_file() and _sha256(shard) == shard_digest
    return rows


def _returns(result) -> pl.DataFrame:
    path = result.root / "run_log" / "backtest" / result.hash / "daily_returns.parquet"
    return pl.read_parquet(path)


def _reject_preview_population(study, *, member_kind: str, member_hash: str) -> None:
    try:
        OfficialPopulation.create(
            study,
            name=f"preview-{member_kind}-must-not-enter-official",
            member_kind=member_kind,
            members=[member_hash],
        )
    except ValueError as error:
        assert "preview" in str(error) or "exploratory" in str(error)
    else:
        raise AssertionError(f"preview {member_kind} entered an official population")


def prove(workspace: Path) -> dict[str, object]:
    study = open_study(execution_tier="preview", workspace=workspace)
    request_catalog = model_request_catalog(
        "linear",
        labels=("fwd_ret_5d",),
        config_names=("ols",),
    )
    preview_reductions = {"folds": [0], "max_symbols": 6}
    resolved = resolve_model_requests(
        study,
        request_catalog,
        execution_tier="preview",
        preview_reductions=preview_reductions,
    )[0]
    assert resolved.spec["computation"]["preview_reductions"] == preview_reductions

    execution = run_resolved_model_requests(study, [resolved])
    run = execution.runs[0]
    prediction = run.predictions[-1]
    prediction_frame = prediction.load()
    expected = _product_keys(resolved._context.expected_keys)
    actual = _product_keys(prediction_frame)
    key_columns = ["product", "timestamp", "fold"]
    assert actual.height == actual.n_unique(key_columns)
    assert actual.join(expected, on=key_columns, how="anti").is_empty()
    assert expected.join(actual, on=key_columns, how="anti").is_empty()
    assert actual.get_column("fold").unique().to_list() == [0]
    assert 1 < actual.get_column("product").n_unique() <= 6
    coverage = prediction.coverage()
    assert coverage is not None and coverage["status"] == "complete"
    assert coverage["n_expected"] == coverage["n_actual"] == actual.height
    state_rows = _fitted_state_rows(study, run.training.hash)

    restarted = run_resolved_model_requests(study, [resolved]).runs[0]
    assert restarted.training.hash == run.training.hash
    assert [item.hash for item in restarted.predictions] == [item.hash for item in run.predictions]
    assert value_digest(restarted.predictions[-1].load()) == value_digest(prediction_frame)
    assert _fitted_state_rows(study, restarted.training.hash) == state_rows

    preview_catalog = study.predictions.table(include_preview=True)
    canonical_catalog = study.predictions.table(include_preview=False)
    selected = preview_catalog.filter(pl.col("prediction_hash") == prediction.hash)
    assert selected.height == 1 and selected.item(0, "complete") is True
    if not canonical_catalog.is_empty():
        assert prediction.hash not in canonical_catalog.get_column("prediction_hash").to_list()
    _reject_preview_population(study, member_kind="prediction", member_hash=prediction.hash)

    products = set(actual.get_column("product"))
    price_path = load_futures_price_path(
        "fwd_ret_5d",
        split="validation",
        products=sorted(products),
    )
    assert set(price_path.prices.get_column("product")) == products
    assert "symbol" not in price_path.prices.columns
    assert price_path.audit.get_column("position").unique().to_list() == [0]
    assert price_path.roll_transitions.height > 0
    assert price_path.roll_transitions.get_column("roll_adjustment_factor").is_finite().all()
    assert set(price_path.expiry_rules.get_column("product")) == products

    signal = {"method": "equal_weight_top_k", "top_k": 2}
    allocation = {"method": "inverse_vol", "vol_window": 20}
    decision = publish_product_weights(
        prediction,
        prices=price_path.prices,
        signal=signal,
        allocation=allocation,
    )
    assert decision.spec["decision_keys"] == ["product", "timestamp"]
    resolved_signal = decision.spec["parameters"]["signal"]
    resolved_allocation = decision.spec["parameters"]["allocation"]
    assert resolved_signal["long_short"] is True
    assert resolved_allocation["long_short"] is True
    assert decision.load().filter(pl.col("weight") < 0).height > 0
    assert decision.load().filter(pl.col("weight") > 0).height > 0
    typed = run_backtests(
        study,
        predictions=selected,
        signal=resolved_signal,
        allocation=resolved_allocation,
        decision=decision,
        prices=price_path.prices,
    ).results[0]
    direct = study.strategy(
        prediction=prediction,
        signal=resolved_signal,
        allocation=resolved_allocation,
    ).run(prices=price_path.prices)
    assert _returns(typed).equals(_returns(direct))
    typed_spec = typed.spec()
    assert typed_spec["entity_contract"]["reader_key"] == "product"
    assert typed_spec["decision_artifact"]["hash"] == decision.hash
    assert typed_spec["decision_artifact"]["decision_keys"] == ["product", "timestamp"]
    assert typed_spec["futures_market"]["roll"]["type"] == "volume"
    assert typed_spec["futures_market"]["contract_position"] == 0
    assert set(typed_spec["futures_market"]["expiry"]["products"]) == products
    assert typed.lineage()["prediction_hash"] == prediction.hash
    backtest_dir = typed.root / "run_log" / "backtest" / typed.hash
    fills = pl.read_parquet(backtest_dir / "fills.parquet")
    assert fills.height > 0
    with sqlite3.connect(typed.root / "run_log" / "registry.db") as db:
        metrics = db.execute(
            "SELECT sharpe, num_trades FROM backtest_metrics WHERE backtest_hash = ?",
            (typed.hash,),
        ).fetchone()
    assert metrics is not None
    sharpe, num_trades = metrics
    assert math.isfinite(sharpe) and num_trades > 0
    try:
        CandidateSet.create(study, "preview-backtest-must-not-rank", [typed])
    except ValueError as error:
        assert "preview" in str(error) or "exploratory" in str(error)
    else:
        raise AssertionError("preview backtest entered a candidate set")
    _reject_preview_population(study, member_kind="backtest", member_hash=typed.hash)

    return {
        "backtest_hash": typed.hash,
        "eligible_rows": actual.height,
        "fitted_state_rows": len(state_rows),
        "num_fills": fills.height,
        "num_products": len(products),
        "num_trades": int(num_trades),
        "prediction_hash": prediction.hash,
        "roll_transitions": price_path.roll_transitions.height,
        "sharpe": float(sharpe),
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
