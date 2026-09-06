"""Every stage of a reduced case-study run covers the same symbols.

`tests/overrides.yaml` injects `MAX_SYMBOLS: 5` into `nasdaq100_microstructure`
stages 01-05 so the pipeline runs small under CI. The stages have to agree on
which five, or the labels and financial features cover one set while the temporal
features cover another, and a symbol only one side chose joins to null features -
a wrong answer that runs clean rather than a failure.

Measured on this fixture before the rules were unified: `02_labels` and
`03_financial_features` reduced through the loader to a seeded random sample,
{AAPL, AMD, CMCSA, CSCO, SIRI}, while `04_model_based_features` took the five
most-observed, {AAPL, AMD, AMZN, FB, TSLA}. Three of five had no temporal
features at all.

The rules are one rule now (`utils.data_quality.top_entities`), which
`tests/test_data_quality.py` pins without data, and `04` and `05` reach it rather
than sorting for themselves. They did not until 2026-09-05, and the production
panel is where that mattered rather than the fixture: every name quoting the whole
window sits on the same padded minute grid, so 115 symbols tie at one row count and
a descending sort over them returns the group-by's order. Two runs of the same
reduced 04 configuration, same code, same data, chose {AAPL, MSFT, TSLA} and
{AMZN, FB, GOOG}. The fixture never showed it because its cut at five happened to
be clear.

So what is checked here is that neither notebook has an expression of its own to be
unstable with, and that the sizes the overrides inject still agree. Neither needs
data. The fixture-backed check that no tie straddled the cut is gone with the
expression that needed it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
CASE_STUDY = "nasdaq100_microstructure"
REDUCED_STAGES = ("02_labels", "03_financial_features", "04_model_based_features", "05_evaluation")


def _injected_max_symbols(stage: str) -> int | None:
    overrides = yaml.safe_load((REPO_ROOT / "tests" / "overrides.yaml").read_text()) or {}
    entry = overrides.get(f"case_studies/{CASE_STUDY}/{stage}") or {}
    value = (entry.get("parameters") or {}).get("MAX_SYMBOLS")
    return int(value) if value else None


def test_the_overrides_still_reduce_these_stages() -> None:
    """Guards the tests below: with no injection they would assert nothing."""
    injected = {stage: _injected_max_symbols(stage) for stage in REDUCED_STAGES}
    assert all(injected.values()), f"no MAX_SYMBOLS injected for {injected}"


def test_every_reduced_stage_is_given_the_same_size() -> None:
    """Equal counts are not sufficient, but unequal ones are already a split."""
    sizes = {_injected_max_symbols(stage) for stage in REDUCED_STAGES}
    assert len(sizes) == 1, f"stages 02-05 are reduced to different sizes: {sizes}"


@pytest.mark.parametrize("stage", ("04_model_based_features", "05_evaluation"))
def test_a_reducing_stage_reaches_the_shared_rule(stage: str) -> None:
    """02 and 03 reduce through the loader; 04 and 05 reduce for themselves.

    Whether they reduce the *same way* is the whole question, and it is answered by which
    function they call rather than by what a particular fixture's row counts happen to be.
    """
    source = (REPO_ROOT / "case_studies" / CASE_STUDY / f"{stage}.py").read_text()
    assert "top_entities" in source, f"{stage} does not reach utils.data_quality.top_entities"


@pytest.mark.parametrize("stage", ("04_model_based_features", "05_evaluation"))
def test_a_reducing_stage_has_no_expression_of_its_own(stage: str) -> None:
    """A local count-and-take is the shape that was unstable, so it is the shape refused.

    Matched on the parsed source rather than on text: `group_by(...)` followed anywhere in
    the same expression by `head(...)` is a reduction of the entity axis whatever it spells
    the count. `top_entities` is the only place that chain may live.
    """
    import ast

    tree = ast.parse((REPO_ROOT / "case_studies" / CASE_STUDY / f"{stage}.py").read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain, cursor = [], node
        while isinstance(cursor, ast.Call) and isinstance(cursor.func, ast.Attribute):
            chain.append(cursor.func.attr)
            cursor = cursor.func.value
        if "head" in chain and "group_by" in chain and "sort" in chain:
            offenders.append((node.lineno, ".".join(reversed(chain))))
    assert offenders == [], (
        f"{stage} reduces its entity axis with an expression of its own: {offenders}. "
        "Every reduction goes through utils.data_quality.top_entities, which breaks the "
        "row-count tie on the entity name; an untied sort returns the group-by's order and "
        "picked two different universes on two runs of the same configuration."
    )
