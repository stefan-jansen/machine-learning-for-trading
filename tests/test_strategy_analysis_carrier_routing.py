"""Every strategy-analysis notebook takes its carrier from the shared resolver.

A strategy-analysis notebook reports one configuration: the one the case study's holdout
notebooks refitted and priced. Those resolve it through
`case_studies.utils.strategy_analysis.resolve_solvent_carrier`. A notebook that ranks a
Sharpe column itself is running a second selection against the same registry, and the two
answer differently for reasons that have nothing to do with which strategy is better:

* a stored Sharpe is computed over whatever span its own configuration priced, so ranking
  the column rewards the most forgiving window. The resolver re-ranks the field on the
  timestamps every candidate prices;
* the resolver applies `LABEL_RESTRICTIONS`, `UNIVERSE_RESTRICTIONS` and `CARRIER_PINS`,
  and admits only what the case study's populations still publish;
* it refuses a carrier whose equity reached zero. A long-short book with no margin call
  keeps compounding through zero, so its reported Sharpe is arithmetic on a balance that
  no longer exists, and it can be high enough to top a ranking.

When the two disagree the failure is quiet and reads as the wrong thing: the notebook asks
for the holdout replay of a configuration the holdout notebooks never ran, finds none, and
reports that the holdout has not been produced.

Read by parsing rather than by importing. These modules are notebooks: importing one
executes it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOLVER = "resolve_solvent_carrier"
NOTEBOOKS = sorted(REPO_ROOT.glob("case_studies/*/[0-9]*_strategy_analysis.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), str(path))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.parts[-2])
def test_the_carrier_is_resolved_through_the_shared_resolver(path: Path) -> None:
    """Imported from the shared module, and actually called.

    Both halves are needed and neither is redundant. An import with no call is a notebook
    that still selects some other way; a call to a name bound somewhere else is a second
    implementation wearing the right name, which is the arrangement
    `tests/test_holdout_selection_is_single_sourced.py` exists because of.
    """
    tree = _tree(path)
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "case_studies.utils.strategy_analysis"
        for alias in node.names
    }
    assert RESOLVER in imported, (
        f"{path.relative_to(REPO_ROOT)} does not import {RESOLVER} from "
        "case_studies.utils.strategy_analysis, so whatever it reports as the carrier is "
        "not what the holdout notebooks ran"
    )
    called = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == RESOLVER
        for node in ast.walk(tree)
    )
    assert called, (
        f"{path.relative_to(REPO_ROOT)} imports {RESOLVER} and never calls it, so the "
        "carrier it reports still comes from somewhere else"
    )


def test_the_corpus_this_covers_is_not_empty() -> None:
    """Guard the guard.

    The parametrization is a glob, so a rename that no longer matches turns every case
    above into zero cases and the file passes while checking nothing. There are nine case
    studies with a strategy-analysis notebook and there is no reason for that to shrink.
    """
    assert len(NOTEBOOKS) >= 9, (
        f"only {len(NOTEBOOKS)} strategy-analysis notebooks matched "
        f"case_studies/*/[0-9]*_strategy_analysis.py: "
        f"{[str(p.relative_to(REPO_ROOT)) for p in NOTEBOOKS]}"
    )
