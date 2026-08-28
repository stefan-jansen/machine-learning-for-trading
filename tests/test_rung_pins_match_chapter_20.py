"""The rung pins in `paired_metrics` must say the same thing as Ch20's.

`18_strategy_analysis` derives its own cohort and paired-metric inputs rather than depending on
`20_strategy_synthesis`, which means the pinned selection is now written down twice: once in
`_CLUSTER_RUNG_RESTRICTIONS` for the chapter, once in `RUNG_PINS` for the case-study notebooks.

Two definitions of the same selection drift silently, and the drift is invisible in output: a
wrong pin does not fail, it selects a different carrier and reports it with equal confidence.
So the duplication is allowed and pinned here rather than left to inspection.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from case_studies.utils.paired_metrics import RUNG_PINS

CH20 = Path(__file__).resolve().parents[1] / "20_strategy_synthesis" / "01_aggregate_synthesis.py"


def _chapter_20_pins() -> dict:
    """Load Ch20's table without executing the notebook body.

    The module is a papermill script whose import-time work needs a registry, so the constant is
    read by parsing rather than importing.
    """
    import ast

    tree = ast.parse(CH20.read_text(encoding="utf-8"), filename=str(CH20))
    for node in tree.body:
        # The chapter annotates the assignment, so it parses as `AnnAssign`, not `Assign`.
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_CLUSTER_RUNG_RESTRICTIONS" and node.value is not None:
                return node.value
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_CLUSTER_RUNG_RESTRICTIONS" for t in node.targets
        ):
            return node.value
    raise AssertionError(f"_CLUSTER_RUNG_RESTRICTIONS is no longer defined in {CH20}")


def _entry_names(pins_node) -> set[str]:
    import ast

    return {key.value for key in pins_node.keys if isinstance(key, ast.Constant)}


def _scalar_fields(pins_node, name: str) -> dict[str, object]:
    import ast

    for key, value in zip(pins_node.keys, pins_node.values, strict=True):
        if not (isinstance(key, ast.Constant) and key.value == name):
            continue
        fields: dict[str, object] = {}
        for field_key, field_value in zip(value.keys, value.values, strict=True):
            if not isinstance(field_key, ast.Constant):
                continue
            if field_key.value in ("universe_filter", "exit_at_max_days") and isinstance(
                field_value, ast.Constant
            ):
                fields[field_key.value] = field_value.value
        return fields
    raise AssertionError(f"{name} is not pinned in chapter 20")


def test_the_same_case_studies_are_pinned() -> None:
    """A case study pinned in one place and not the other is the drift that matters most.

    Adding a pin to the chapter alone leaves that case study's own notebook selecting across
    rungs; adding one here alone pins a selection the chapter does not make.
    """
    assert _entry_names(_chapter_20_pins()) == set(RUNG_PINS)


@pytest.mark.parametrize("case_study", sorted(RUNG_PINS))
def test_the_scalar_scope_agrees(case_study: str) -> None:
    """`universe_filter` and `exit_at_max_days` are the pin's SQL-path half.

    They are compared as values because they are values. The polars predicate beside them is
    compared separately, since two expressions can be equal without being identical objects.
    """
    chapter = _scalar_fields(_chapter_20_pins(), case_study)
    here = RUNG_PINS[case_study]

    assert chapter["universe_filter"] == here["universe_filter"]
    assert chapter["exit_at_max_days"] == here["exit_at_max_days"]


@pytest.mark.parametrize("case_study", sorted(RUNG_PINS))
def test_the_predicate_names_the_same_columns_and_values(case_study: str) -> None:
    """A predicate's serialised form, which is what a reader would compare by hand.

    Comparing the rendered expression rather than the object catches the substantive changes -
    a different column, a different literal, a dropped conjunct - without asserting that two
    independently built expressions are the same instance.
    """
    import ast

    chapter_src = ""
    pins = _chapter_20_pins()
    for key, value in zip(pins.keys, pins.values, strict=True):
        if isinstance(key, ast.Constant) and key.value == case_study:
            for field_key, field_value in zip(value.keys, value.values, strict=True):
                if isinstance(field_key, ast.Constant) and field_key.value == "predicate":
                    chapter_src = ast.unparse(field_value)

    assert chapter_src, f"chapter 20 has no predicate for {case_study}"

    # Ch20 stores its predicates in module-level names; resolve them to the expression source.
    if chapter_src.startswith("_"):
        tree = ast.parse(CH20.read_text(encoding="utf-8"), filename=str(CH20))
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == chapter_src and node.value is not None:
                    chapter_src = ast.unparse(node.value)
                    break
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == chapter_src for t in node.targets
            ):
                chapter_src = ast.unparse(node.value)
                break

    rendered = str(RUNG_PINS[case_study]["predicate"])
    for column in ("universe_filter", "exit_at_max_days", "family"):
        assert (column in chapter_src) == (column in rendered), (
            f"{case_study}: chapter 20 and RUNG_PINS disagree about {column!r}\n"
            f"  chapter 20: {chapter_src}\n  RUNG_PINS:  {rendered}"
        )
    for literal in ("liquid", "cost_feasible", "ensemble"):
        assert (literal in chapter_src) == (literal in rendered), (
            f"{case_study}: chapter 20 and RUNG_PINS disagree about {literal!r}\n"
            f"  chapter 20: {chapter_src}\n  RUNG_PINS:  {rendered}"
        )
