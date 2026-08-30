from __future__ import annotations

import ast
from pathlib import Path

import polars as pl


def _load_selector():
    source = (
        Path(__file__).parents[1] / "case_studies" / "us_firm_characteristics" / "14_costs.py"
    ).read_text()
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_pre_cost_runs"
    )
    namespace: dict[str, object] = {"pl": pl}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "14_costs.py", "exec"), namespace)
    return namespace["_resolve_pre_cost_runs"], namespace


def test_cost_parent_is_the_best_across_the_canonical_stages() -> None:
    """Cost sensitivity sweeps the winner across the three stages the canonical rank-1 uses.

    The union is what makes an un-overlaid configuration reachable: the risk stage files a
    row per named control and none for the strategy without one, so drawing from
    ``risk_overlay`` alone would force an overlay on even where every control hurt.
    """
    selector, namespace = _load_selector()
    calls: list[str] = []

    def fake_resolver(case_study, label, *, split, stage, top_n):
        calls.append(stage)
        sharpe = 2.632140 if stage == "allocation" else 2.592095
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

    assert calls == ["signal", "allocation", "risk_overlay"]
    # The un-overlaid allocation run is stronger here, so it is what gets swept.
    assert result["backtest_hash"].to_list() == ["allocation_hash"]


def test_cost_rows_are_not_in_the_pool() -> None:
    """The pool must match the canonical selection, and must exclude cost rows."""
    selector, namespace = _load_selector()
    calls: list[str] = []

    def fake_resolver(case_study, label, *, split, stage, top_n):
        calls.append(stage)
        return pl.DataFrame(
            {
                "backtest_hash": [f"{stage}_hash"],
                "prediction_hash": ["prediction"],
                "spec_json": ["{}"],
                "sharpe": [1.0],
            }
        )

    namespace["resolve_best_backtest_runs"] = fake_resolver
    selector("us_firm_characteristics", "fwd_ret_1m", split="validation", top_n=1)
    assert "cost_sensitivity" not in calls
    assert "cost_sensitivity" not in calls


def test_insolvent_leader_falls_through_to_the_solvent_run_behind_it() -> None:
    """A bankrupt leader must not take the slot, nor cost its stage the slot.

    The stages are asked for their whole ranked list and truncated after the solvency
    filter. Were each stage truncated to ``top_n`` first, the insolvent overlay leader
    below would consume the one slot that stage gets and its solvent runner-up would never
    be considered - with ``top_n=1`` the whole stage drops out silently.
    """
    selector, namespace = _load_selector()

    def fake_resolver(case_study, label, *, split, stage, top_n):
        # Honours top_n, as the real resolver does: it truncates in SQL. That is what makes
        # the assertion below fail if the caller asks for top_n rather than the ranked pool.
        if stage == "risk_overlay":
            frame = pl.DataFrame(
                {
                    "backtest_hash": ["overlay_ruined", "overlay_solvent"],
                    "prediction_hash": ["prediction", "prediction"],
                    "spec_json": ["{}", "{}"],
                    "sharpe": [9.9, 2.7],
                }
            )
        else:
            frame = pl.DataFrame(
                {
                    "backtest_hash": ["alloc_solvent"],
                    "prediction_hash": ["prediction"],
                    "spec_json": ["{}"],
                    "sharpe": [2.5],
                }
            )
        return frame.head(top_n)

    namespace["resolve_best_backtest_runs"] = fake_resolver
    result = selector(
        "us_firm_characteristics",
        "fwd_ret_1m",
        split="validation",
        top_n=1,
        solvent_hashes=lambda hashes: {h for h in hashes if h != "overlay_ruined"},
    )

    # 9.9 is the bankrupt run's Sharpe, computed on a balance that no longer exists.
    assert result["backtest_hash"].to_list() == ["overlay_solvent"]


def test_no_solvency_filter_leaves_the_ranking_alone() -> None:
    """Passing no filter selects on Sharpe alone, so the ranking is testable on its own."""
    selector, namespace = _load_selector()

    def fake_resolver(case_study, label, *, split, stage, top_n):
        return pl.DataFrame(
            {
                "backtest_hash": [f"{stage}_hash"],
                "prediction_hash": ["prediction"],
                "spec_json": ["{}"],
                "sharpe": [9.9 if stage == "risk_overlay" else 2.5],
            }
        )

    namespace["resolve_best_backtest_runs"] = fake_resolver
    result = selector("us_firm_characteristics", "fwd_ret_1m", split="validation", top_n=1)
    assert result["backtest_hash"].to_list() == ["risk_overlay_hash"]
