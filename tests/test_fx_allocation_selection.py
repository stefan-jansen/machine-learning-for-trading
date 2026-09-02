"""FX allocation selection keeps one baseline result per model configuration."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

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
        "_result_config",
        "_select_configuration_survivors",
        "_baseline_top_k",
    }
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, Any] = {"Iterable": list, "BacktestResult": _Result}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_a_scoped_preview_can_read_the_canonical_baseline() -> None:
    resolve = _selection_functions()["_resolve_baseline_scope"]

    assert resolve("fx_pairs:preflight", None) == "fx_pairs:preflight"
    assert resolve("fx_pairs:preflight", "") == ""


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
