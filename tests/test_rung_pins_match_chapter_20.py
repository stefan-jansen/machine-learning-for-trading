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

import polars as pl
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


def _chapter_20_predicate(case_study: str):
    """Build the chapter's predicate as an expression, not as text.

    Its source is evaluated with `pl` in scope. Comparing rendered strings was the first
    version of this test and it could not fail on the changes that matter: `|` for `&`, `!=`
    for `==`, `is_not_null()` for `is_null()` all name the same columns and the same literals
    while selecting a different carrier.
    """
    import ast

    pins = _chapter_20_pins()
    source = ""
    for key, value in zip(pins.keys, pins.values, strict=True):
        if isinstance(key, ast.Constant) and key.value == case_study:
            for field_key, field_value in zip(value.keys, value.values, strict=True):
                if isinstance(field_key, ast.Constant) and field_key.value == "predicate":
                    source = ast.unparse(field_value)
    assert source, f"chapter 20 has no predicate for {case_study}"

    if source.startswith("_"):
        tree = ast.parse(CH20.read_text(encoding="utf-8"), filename=str(CH20))
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == source and node.value is not None:
                    source = ast.unparse(node.value)
                    break
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == source for t in node.targets
            ):
                source = ast.unparse(node.value)
                break

    return eval(source, {"pl": pl})  # noqa: S307 - repo source, read from a fixed path


# Every combination the two pins can discriminate on. `exit_at_max_days` carries a null and a
# value because sp500_options pins on its nullity; `universe_filter` carries both pinned values
# and the "full" that rung-1 and rung-2 share; `family` separates nasdaq's ensemble carrier.
_TRUTH_TABLE = pl.DataFrame(
    [
        {"universe_filter": uf, "family": fam, "exit_at_max_days": exit_days}
        for uf in ("full", "liquid", "cost_feasible", None)
        for fam in ("ensemble", "linear", None)
        for exit_days in (None, 5)
    ],
    schema={"universe_filter": pl.String, "family": pl.String, "exit_at_max_days": pl.Int64},
)


@pytest.mark.parametrize("case_study", sorted(RUNG_PINS))
def test_the_predicate_selects_exactly_the_same_rows(case_study: str) -> None:
    """The comparison that catches an inverted operator: both pins, evaluated, row by row.

    A pin is a selection, so two pins agree exactly when they select the same rows. Anything
    weaker - the columns they mention, the literals they contain - passes on a predicate that
    picks the opposite set.
    """
    chapter = _TRUTH_TABLE.select(_chapter_20_predicate(case_study).alias("hit"))["hit"]
    here = _TRUTH_TABLE.select(RUNG_PINS[case_study]["predicate"].alias("hit"))["hit"]

    disagreements = _TRUTH_TABLE.filter(chapter.ne_missing(here))
    assert disagreements.is_empty(), (
        f"{case_study}: the two pins select different rows\n{disagreements}"
    )

    # A pin that selects nothing over this table would make the comparison above vacuous.
    assert here.fill_null(False).any(), f"{case_study}: the pin matches no row in the truth table"
