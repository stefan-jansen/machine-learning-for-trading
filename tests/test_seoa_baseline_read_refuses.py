"""An empty population-scoped baseline read must say what is wrong, not divide by zero.

`sp500_equity_option_analytics/14_backtest` scopes its baseline read by the prediction
sets the registry's populations publish. When that read comes back empty the next thing
to touch it is

    100 * all_baselines.filter(pl.col("sharpe") > 0).height / len(all_baselines)

so the run failed as `ZeroDivisionError: division by zero`, which reads like a defect in
the analysis. It was not: the populations were in force with 30 registered members, none
of which carried a signal backtest, because the fixture had been generated through stage
08 and nothing backtests a population the model notebooks declare at stage 06 and 07
(ml4t/agent-workspace#1086).

The refusal is lifted out of the notebook and executed here, rather than restated, so this
cannot pass against a notebook that no longer carries it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import polars as pl
import pytest

NOTEBOOK = (
    Path(__file__).parent.parent
    / "case_studies"
    / "sp500_equity_option_analytics"
    / "14_backtest.py"
)
GUARD_TEST = "CURRENT_MEMBERS is not None and all_baselines.is_empty()"


def _lift_guard(source: str) -> str:
    """The notebook's own refusal block, as source, found by what it tests."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == GUARD_TEST:
            return ast.unparse(node)
    raise AssertionError(
        f"no top-level `if {GUARD_TEST}:` in {NOTEBOOK.name}; the baseline read is "
        f"unguarded and an empty population-scoped read will divide by zero"
    )


class _Explorer:
    """Only what the refusal uses: the count of signal rows in the registry."""

    def __init__(self, signal_rows: int) -> None:
        self._rows = signal_rows

    def specs(self, stages):  # noqa: ARG002 - the guard passes "signal"
        return pl.DataFrame(
            {"backtest_hash": [f"b{i}" for i in range(self._rows)]},
            schema={"backtest_hash": pl.Utf8},
        )


def _run_guard(*, members, baselines_height: int, signal_rows: int, populations):
    namespace = {
        "CURRENT_MEMBERS": members,
        "CURRENT_POPULATIONS": populations,
        "all_baselines": pl.DataFrame(
            {"sharpe": [0.1] * baselines_height}, schema={"sharpe": pl.Float64}
        ),
        "explorer": _Explorer(signal_rows),
        "pl": pl,
    }
    exec(_lift_guard(NOTEBOOK.read_text()), namespace)  # noqa: S102


def test_the_notebook_carries_the_guard():
    """Fails on the pre-fix notebook, where no such block exists."""
    assert GUARD_TEST in _lift_guard(NOTEBOOK.read_text())


def test_a_population_whose_members_have_no_backtest_is_refused():
    with pytest.raises(RuntimeError) as excinfo:
        _run_guard(
            members=frozenset(f"m{i}" for i in range(30)),
            baselines_height=0,
            signal_rows=52,
            populations=["seoa-gbm-validation-v1", "seoa-linear-validation-v1"],
        )
    message = str(excinfo.value)
    # What the next reader needs: the scope, its size, and that the registry is not empty.
    assert "30" in message, message
    assert "52" in message, message
    assert "seoa-gbm-validation-v1" in message, message
    assert "seoa-linear-validation-v1" in message, message


def test_the_refusal_does_not_degrade_to_a_pass():
    """An empty scoped read must stop the notebook, not be summarised as a result."""
    with pytest.raises(RuntimeError):
        _run_guard(
            members=frozenset({"m0"}),
            baselines_height=0,
            signal_rows=1,
            populations=["p"],
        )


def test_a_populated_read_is_not_refused():
    """The premise: the guard fires on emptiness, not on the population existing."""
    _run_guard(
        members=frozenset({"m0"}),
        baselines_height=3,
        signal_rows=52,
        populations=["p"],
    )


def test_an_unscoped_read_is_not_refused():
    """No populations means no filter, which is a fixture or a clean clone, not a fault."""
    _run_guard(members=None, baselines_height=0, signal_rows=52, populations=[])
