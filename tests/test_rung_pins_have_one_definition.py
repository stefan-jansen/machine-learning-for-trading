"""The rung pins are defined once, and each pin selects exactly what its mirror keys say.

`18_strategy_analysis` derives its own cohort and paired-metric inputs rather than depending on
`20_strategy_synthesis`, so the pinned selection used to be written down twice: once as
`_CLUSTER_RUNG_RESTRICTIONS` for the chapter, once as `RUNG_PINS` for the case-study notebooks.
Both write `backtest_paired_metrics`. Two definitions of one selection drift silently and the
drift is invisible in output - a wrong pin does not fail, it selects a different carrier and
reports it with equal confidence.

This file used to pin the duplication by parsing the chapter's literal and comparing it. The
chapter imports `RUNG_PINS` now, so what is checked is that it still does and that no rival
literal has reappeared. That is strictly stronger: the earlier version could only notice a
divergence after someone introduced one.

The truth table stays, because the other half of a pin is still duplicated in a way no import
removes: each entry carries plain `universe_filter` / `exit_at_max_days` / `label` values for
the SQL paths that cannot take a polars expression, beside the expression itself. Those two
halves can disagree, so they are compared by selection rather than by inspection.
"""

from __future__ import annotations

import ast
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.paired_metrics import RUNG_PINS

CH20 = Path(__file__).resolve().parents[1] / "20_strategy_synthesis" / "01_aggregate_synthesis.py"


def test_chapter_20_imports_the_pins_rather_than_restating_them() -> None:
    source = CH20.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CH20))

    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "case_studies.utils.paired_metrics"
        and any(alias.name == "RUNG_PINS" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert imported, (
        f"{CH20.name} no longer imports RUNG_PINS from paired_metrics. If it has gone back to "
        "defining its own table, the two will drift and each will overwrite the other's "
        "backtest_paired_metrics rows with a differently-selected lineage."
    )

    # An import plus a shadowing assignment is the same duplication wearing the same name.
    rival = [
        node
        for node in tree.body
        if (isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "RUNG_PINS")
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "RUNG_PINS" for t in node.targets)
        )
    ]
    assert not rival, f"{CH20.name} imports RUNG_PINS and then reassigns it"


# Every combination the pins can discriminate on. `exit_at_max_days` carries a null and a value
# because sp500_options pins on its nullity; `universe_filter` carries both pinned values and the
# "full" that rung-1 and rung-2 share; `label` separates the four labels nasdaq's cost-feasible
# sweep spans. `family` is carried because no pin should be able to narrow on a column its
# scalar half cannot express without the superset check below catching it.
_TRUTH_TABLE = pl.DataFrame(
    [
        {"universe_filter": uf, "family": fam, "exit_at_max_days": exit_days, "label": label}
        for uf in ("full", "liquid", "cost_feasible", None)
        for fam in ("ensemble", "linear", None)
        for exit_days in (None, 5)
        for label in ("fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m", "ret_to_expiry", None)
    ],
    schema={
        "universe_filter": pl.String,
        "family": pl.String,
        "exit_at_max_days": pl.Int64,
        "label": pl.String,
    },
)


@pytest.mark.parametrize("case_study", sorted(RUNG_PINS))
def test_the_predicate_agrees_with_its_scalar_mirror(case_study: str) -> None:
    """The expression and the plain values beside it must select the same rows.

    Comparing the columns they mention or the literals they contain would pass on a predicate
    that picks the opposite set, so they are compared by evaluating both over every combination
    the pins can tell apart. A pin omitting a key it declares is the failure this catches - and
    it is the one that was live: nasdaq's predicate matched three labels while its scalar half
    named one.
    """
    pin = RUNG_PINS[case_study]

    # Only the columns the predicate actually constrains. A scalar key naming a column the
    # predicate ignores is not a disagreement: `exit_at_max_days: None` reads as "must be null"
    # for sp500_options, whose predicate says so, and as "not applicable" for nasdaq100, whose
    # equities carry no expiry. The dict cannot tell those apart, so the predicate decides which
    # columns are in scope and the mirror is checked only on those.
    constrained = set(pin["predicate"].meta.root_names())

    mirror = pl.lit(True)
    if "universe_filter" in constrained and "universe_filter" in pin:
        mirror = mirror & (pl.col("universe_filter") == pin["universe_filter"])
    if "label" in constrained and "label" in pin:
        mirror = mirror & (pl.col("label") == pin["label"])
    if "exit_at_max_days" in constrained and pin.get("exit_at_max_days", "unset") is None:
        mirror = mirror & pl.col("exit_at_max_days").is_null()

    by_predicate = _TRUTH_TABLE.select(pin["predicate"].alias("hit"))["hit"]
    by_mirror = _TRUTH_TABLE.select(mirror.alias("hit"))["hit"]

    # The predicate may narrow further than the mirror can express, so the mirror must be a
    # superset and never the other way round. No pin uses that latitude today: nasdaq's
    # `family` clause was the one that did, and it is gone.
    escaped = _TRUTH_TABLE.filter(by_predicate.fill_null(False) & ~by_mirror.fill_null(False))
    assert escaped.is_empty(), (
        f"{case_study}: the predicate selects rows its scalar half excludes\n{escaped}"
    )
    assert by_predicate.fill_null(False).any(), (
        f"{case_study}: the pin matches no row in the truth table, so the check above is vacuous"
    )
    # And the mirror must not be vacuous either: a pin whose scalar half constrains nothing
    # would let the superset check pass on any predicate at all.
    assert not by_mirror.fill_null(False).all(), (
        f"{case_study}: the scalar half excludes no row, so it constrains nothing"
    )
