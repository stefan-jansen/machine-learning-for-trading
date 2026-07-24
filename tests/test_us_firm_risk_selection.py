from __future__ import annotations

import ast
from pathlib import Path

import polars as pl


def _load_selector():
    source = (
        Path(__file__).parents[1]
        / "case_studies"
        / "us_firm_characteristics"
        / "14_risk_management.py"
    ).read_text()
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_pre_risk_runs"
    )
    namespace: dict[str, object] = {"pl": pl}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "14_risk_management.py", "exec"),
        namespace,
    )
    return namespace["_resolve_pre_risk_runs"], namespace


def test_risk_parent_is_best_of_baseline_and_allocation() -> None:
    selector, namespace = _load_selector()
    calls: list[str] = []

    def fake_resolver(case_study, label, *, split, stage, top_n):
        calls.append(stage)
        sharpe = 2.632140 if stage == "signal" else 2.592095
        return pl.DataFrame(
            {
                "backtest_hash": [f"{stage}_hash"],
                "prediction_hash": ["prediction"],
                "spec_json": ["{}"],
                "sharpe": [sharpe],
            }
        )

    namespace["resolve_best_backtest_runs"] = fake_resolver
    result = selector("us_firm_characteristics", "fwd_ret_1m", split="validation", top_n=1)

    assert calls == ["signal", "allocation"]
    assert result["backtest_hash"].to_list() == ["signal_hash"]
