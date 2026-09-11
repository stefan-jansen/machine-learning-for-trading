"""FX allocation selection keeps one baseline result per model configuration."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import pytest

NOTEBOOK = Path("case_studies/fx_pairs/14_portfolio_management.py")


class _Result:
    def __init__(
        self,
        result_hash: str,
        *,
        label: str,
        family: str,
        config_name: str,
        prediction_hash: str,
        top_k: int,
    ) -> None:
        self.hash = result_hash
        self._lineage = {
            "training_spec": {
                "label": label,
                "family": family,
                "config_name": config_name,
            }
        }
        self._prediction_hash = prediction_hash
        self._spec = {"strategy": {"signal": {"method": "equal_weight_top_k", "top_k": top_k}}}

    def lineage(self) -> dict[str, Any]:
        return self._lineage

    def registry_record(self) -> dict[str, str]:
        return {"prediction_hash": self._prediction_hash}

    def spec(self) -> dict[str, Any]:
        return self._spec


def _selection_functions() -> dict[str, Any]:
    tree = ast.parse(NOTEBOOK.read_text())
    names = {
        "_resolve_baseline_scope",
        "_scope_baseline_labels",
        "_non_allocation_projection",
        "_result_config",
        "_select_configuration_survivors",
        "_baseline_top_k",
    }
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, Any] = {
        "Iterable": list,
        "BacktestResult": _Result,
        "deepcopy": copy.deepcopy,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_a_scoped_preview_can_read_the_canonical_baseline() -> None:
    resolve = _selection_functions()["_resolve_baseline_scope"]

    assert resolve("fx_pairs:preflight", None) == "fx_pairs:preflight"
    assert resolve("fx_pairs:preflight", "") == ""


def test_a_scoped_run_iterates_only_the_labels_its_catalog_holds() -> None:
    """The canonical baseline carries every label; a narrowed run holds one."""
    scope = _selection_functions()["_scope_baseline_labels"]
    every = ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_21d"]

    assert scope(every, ["fwd_ret_5d"], "fx_pairs:partial") == ["fwd_ret_5d"]


def test_an_unscoped_run_is_never_narrowed() -> None:
    """Narrowing the canonical path would hide the disagreement its guard reports.

    The caller checks baseline labels against catalog labels for equality before this
    runs, and that check is the one that must fire. Silently intersecting here would
    turn a real coverage failure into a smaller, passing run.
    """
    scope = _selection_functions()["_scope_baseline_labels"]
    every = ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_21d"]

    assert scope(every, ["fwd_ret_5d"], "") == every


def test_a_scoped_run_sharing_no_label_with_the_baseline_is_refused() -> None:
    """Returning an empty list would make the allocation loop silently do nothing."""
    scope = _selection_functions()["_scope_baseline_labels"]

    with pytest.raises(RuntimeError, match="none of the catalog's labels"):
        scope(["fwd_ret_1d"], ["fwd_ret_60m"], "fx_pairs:partial")


def test_a_worktree_path_does_not_make_an_allocation_differ_from_its_baseline() -> None:
    """`preset_path` is an absolute machine path and must never be compared.

    It is excluded from the identity hash by `_HASH_EXCLUDED_METADATA` for exactly
    this reason. Comparing it makes the notebook refuse its own siblings whenever the
    baseline was registered from a different checkout than the allocation run, which
    is the ordinary case when two notebooks run from two worktrees.
    """
    project = _selection_functions()["_non_allocation_projection"]

    def spec(preset: str) -> dict[str, Any]:
        return {
            "strategy": {"signal": {"top_k": 5}, "allocation": {"method": "inverse_vol"}},
            "backtest_config": {"metadata": {"chapter": 19, "preset_path": preset}},
        }

    from_fx13 = project(spec("/home/stefan/ml4t/public-fx13/config/base.yaml"), drop_prices=False)
    from_fx14 = project(spec("/home/stefan/ml4t/public-fx14/config/base.yaml"), drop_prices=False)

    assert from_fx13 == from_fx14


def test_the_projection_still_sees_a_real_strategy_change() -> None:
    """Dropping provenance must not make the check unable to fail."""
    project = _selection_functions()["_non_allocation_projection"]

    def spec(top_k: int) -> dict[str, Any]:
        return {
            "strategy": {"signal": {"top_k": top_k}, "allocation": {"method": "inverse_vol"}},
            "backtest_config": {"metadata": {"chapter": 19, "preset_path": "/a/base.yaml"}},
        }

    assert project(spec(5), drop_prices=False) != project(spec(10), drop_prices=False)


def test_selection_keeps_the_best_checkpoint_and_mapping_per_configuration() -> None:
    functions = _selection_functions()
    best_a = _Result(
        "bt-a-best",
        label="fwd_ret_1d",
        family="tcn",
        config_name="base",
        prediction_hash="p-a-best",
        top_k=5,
    )
    worse_a = _Result(
        "bt-a-worse",
        label="fwd_ret_1d",
        family="tcn",
        config_name="base",
        prediction_hash="p-a-worse",
        top_k=10,
    )
    best_b = _Result(
        "bt-b-best",
        label="fwd_ret_1d",
        family="linear",
        config_name="ridge",
        prediction_hash="p-b-best",
        top_k=15,
    )

    selected = functions["_select_configuration_survivors"]([best_a, worse_a, best_b], 2)

    assert [result.registry_record()["prediction_hash"] for result in selected] == [
        "p-a-best",
        "p-b-best",
    ]
    assert [functions["_baseline_top_k"](result) for result in selected] == [5, 15]
