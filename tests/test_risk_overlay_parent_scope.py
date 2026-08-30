"""A risk overlay belongs to one allocation, not to every allocation of its prediction.

`etfs/16_risk_management` sweeps the top allocation *combinations* - a prediction appears once per
allocator and `top_k` - and then reads the overlays back. Scoping that read on the prediction
hash alone pulls in overlays sitting on combinations the sweep did not advance, and those rank
beside the ones it did.
"""

from __future__ import annotations

import json
import sqlite3

from case_studies.utils.backtest_explorer import BacktestExplorer


def _spec(allocator: str, top_k: int, risk_name: str | None) -> str:
    strategy: dict = {
        "signal": {"top_k": top_k},
        "allocation": {"method": allocator},
    }
    if risk_name is not None:
        strategy["risk"] = {
            "name": risk_name,
            "position_rules": [{"type": "stop_loss"}],
        }
    return json.dumps({"version": 2, "strategy": strategy, "backtest_config": {}})


def _registry(case_dir):
    run_log = case_dir / "run_log"
    run_log.mkdir(parents=True)
    with sqlite3.connect(run_log / "registry.db") as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT
            );
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, spec_json TEXT, stage TEXT
            );
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY, sharpe REAL, max_drawdown REAL, num_trades INTEGER
            );
            """
        )
        db.execute("INSERT INTO training_runs VALUES ('t','gbm','cfg','fwd_ret_21d')")
        db.execute("INSERT INTO prediction_sets VALUES ('p','t','validation')")
        rows = [
            # (hash, spec, stage, sharpe)
            ("alloc_advanced", _spec("mean_variance", 20, None), "allocation", 1.0),
            ("alloc_other", _spec("equal_weight", 50, None), "allocation", 1.0),
            ("ovl_advanced", _spec("mean_variance", 20, "advanced_rule"), "risk_overlay", 1.2),
            ("ovl_other", _spec("equal_weight", 50, "other_rule"), "risk_overlay", 9.9),
        ]
        for b_hash, spec, stage, sharpe in rows:
            db.execute("INSERT INTO backtest_runs VALUES (?,?,?,?)", (b_hash, "p", spec, stage))
            db.execute("INSERT INTO backtest_metrics VALUES (?,?,?,?)", (b_hash, sharpe, -0.1, 10))
    return BacktestExplorer("test", case_dir=case_dir)


def test_scoping_on_the_prediction_admits_overlays_the_sweep_did_not_advance(tmp_path) -> None:
    explorer = _registry(tmp_path / "case")

    pooled = explorer.risk_impact(prediction_hashes=["p"])

    assert set(pooled["risk_name"].to_list()) == {"advanced_rule", "other_rule"}
    assert pooled.sort("sharpe", descending=True).row(0, named=True)["risk_name"] == "other_rule"


def test_scoping_on_the_allocation_parent_keeps_only_its_own_overlays(tmp_path) -> None:
    explorer = _registry(tmp_path / "case")

    scoped = explorer.risk_impact(parents=[("p", "mean_variance", 20)])

    assert scoped["risk_name"].to_list() == ["advanced_rule"]


def test_a_parent_matched_on_two_of_three_fields_is_not_a_match(tmp_path) -> None:
    """Same prediction and allocator, different `top_k`: a different combination."""
    explorer = _registry(tmp_path / "case")

    assert explorer.risk_impact(parents=[("p", "mean_variance", 50)]).is_empty()
