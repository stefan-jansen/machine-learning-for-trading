"""Repository-wide guard on skip-recent (12-1) momentum construction.

Returns compound, so momentum that skips the most recent month is the price
ratio ``P[t-21] / P[t-252] - 1``. Subtracting a 1-month return from a 12-month
return is only a first-order approximation and is not a return over any window.
That shortcut shipped in three case studies before it was caught, so this test
scans every case-study feature notebook for the pattern rather than checking a
single hand-picked file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CASE_STUDIES = Path(__file__).parents[1] / "case_studies"

SKIP_ALIAS_MARKERS = ("skip_recent", "skip", "12_1", "12m_skip")


def _feature_sources() -> list[Path]:
    return sorted(CASE_STUDIES.glob("*/0*_*features*.py"))


def _references_column(node: ast.AST) -> bool:
    """True if the expression reads a DataFrame column, e.g. ``pl.col("ret_21d")``."""
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "col"
        for child in ast.walk(node)
    )


def _subtraction_aliases(source: Path) -> list[str]:
    """Return skip-momentum aliases built by subtracting one column from another.

    The correct construction ends in ``- 1`` to turn a price ratio into a
    return, so a bare subtraction is not enough to flag: both operands must
    read columns.
    """
    offenders = []
    for node in ast.walk(ast.parse(source.read_text())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "alias" or not node.args:
            continue
        name = node.args[0]
        if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
            continue
        if not any(marker in name.value.lower() for marker in SKIP_ALIAS_MARKERS):
            continue
        for child in ast.walk(node.func.value):
            if (
                isinstance(child, ast.BinOp)
                and isinstance(child.op, ast.Sub)
                and _references_column(child.left)
                and _references_column(child.right)
            ):
                offenders.append(name.value)
                break
    return offenders


@pytest.mark.parametrize("source", _feature_sources(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_skip_momentum_is_not_a_difference_of_returns(source: Path) -> None:
    offenders = _subtraction_aliases(source)
    assert not offenders, (
        f"{source.relative_to(CASE_STUDIES.parent)} builds {offenders} by subtracting returns. "
        "Skip-recent momentum must divide prices: "
        "close.shift(21) / close.shift(252) - 1."
    )


def test_guard_detects_the_original_defect(tmp_path: Path) -> None:
    """The scan must actually fire on the construction that shipped."""
    bad = tmp_path / "bad_features.py"
    bad.write_text(
        'df = df.with_columns((pl.col("ret_252d") - pl.col("ret_21d")).alias("ret_12m_skip"))\n'
    )
    assert _subtraction_aliases(bad) == ["ret_12m_skip"]
