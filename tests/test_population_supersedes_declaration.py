"""The notebook parameter that says which population snapshot a refit retires.

`OfficialPopulation.create` refuses to publish a changed list under a name that already
has one unless the run names the snapshot it supersedes - `case_studies/research/
population.py:246`, `a changed population named ... must explicitly supersedes ...`.

The refusal is cheap, and that is what makes it easy to leave unanswered.
`run_model_population` creates the population before it fits anything - `execution.py:584`
and `:596` both snapshot, then call `run_official_model_subset` - so a notebook missing the
parameter fails in seconds rather than after the sweep. What it cannot do is proceed: the
run that changes the member list cannot publish until someone edits the notebook, and a
notebook is edited under a provenance gate rather than at a prompt.

`create` matches on the member list first, so a re-run publishing a byte-identical list
returns the existing population whatever `supersedes` says. The parameter is dead weight on
every re-run and the only way through on the one that widens or narrows the list - which is
exactly the shape of thing a rewrite drops without noticing it was load-bearing.

The only way to answer the refusal is a papermill-settable parameter, so this checks the
three things that have to hold together: the name exists, it is in the tagged cell where
papermill can set it, and it reaches the call. Any one of them alone is inert.

Found on `origin/rescue/nasdaq100-07-gbm-rewrite-main`, a rewrite of `07_gbm` that dropped
the parameter its predecessor carried, against a registry that already holds two
populations named `nasdaq100_microstructure-gbm-validation-v1`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Notebooks owned by another lane that call the publish without the parameter. Each entry is
# asserted to still be earned by `test_every_exemption_is_still_earned`, so it has to be
# deleted in the same change that adds the parameter rather than outliving it unread.
KNOWN_UNDECLARED = {"case_studies/crypto_perps_funding/06_linear.py"}


def _publishes_a_population(source: str) -> bool:
    return any(
        isinstance(node, ast.Call) and _call_name(node) == "run_model_population"
        for node in ast.walk(ast.parse(source))
    )


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _discover() -> list[str]:
    """Every case-study notebook that publishes a population.

    Discovered rather than listed: a new notebook that publishes one is exactly the case
    this guards, and a hardcoded list would not see it. Restricted to the numbered notebooks
    so `case_studies/research/execution.py`, which defines the call, is not swept up.
    """
    found = []
    for path in sorted((REPO / "case_studies").glob("*/[0-9]*.py")):
        if _publishes_a_population(path.read_text()):
            found.append(str(path.relative_to(REPO)))
    return found


NOTEBOOKS = _discover()
GUARDED = [path for path in NOTEBOOKS if path not in KNOWN_UNDECLARED]


def test_the_corpus_publishes_populations_at_all():
    """Without this the parametrized checks below can pass on an empty list."""
    assert len(NOTEBOOKS) > 10, f"only {len(NOTEBOOKS)} publishing notebooks discovered"
    assert GUARDED


@pytest.mark.parametrize("path", GUARDED)
def test_the_parameter_is_in_the_parameters_cell(path: str):
    """Papermill only overrides names in the tagged cell; elsewhere it is a constant."""
    cells = (REPO / path).read_text().split("\n# %%")
    tagged = [cell for cell in cells if cell.startswith(' tags=["parameters"]')]
    assert tagged, f"{path} has no parameters cell"
    assert any("SUPERSEDES_POPULATION" in cell for cell in tagged), (
        f"{path} publishes a population without a SUPERSEDES_POPULATION parameter papermill "
        "can set, so a second run cannot answer the publish-time refusal"
    )


@pytest.mark.parametrize("path", GUARDED)
def test_the_parameter_reaches_the_publish(path: str):
    """Declaring it and not passing it is the same as not declaring it."""
    tree = ast.parse((REPO / path).read_text())
    passed = [
        keyword
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "run_model_population"
        for keyword in node.keywords
        if keyword.arg == "supersedes"
    ]
    assert passed, (
        f"{path} declares SUPERSEDES_POPULATION but no run_model_population() call takes it "
        "as supersedes="
    )


@pytest.mark.parametrize("path", sorted(KNOWN_UNDECLARED))
def test_every_exemption_is_still_earned(path: str):
    """An exemption for a notebook that now declares it hides the next regression."""
    assert path in NOTEBOOKS, f"{path} no longer publishes a population; drop the exemption"
    assert "SUPERSEDES_POPULATION" not in (REPO / path).read_text(), (
        f"{path} now declares SUPERSEDES_POPULATION; remove it from KNOWN_UNDECLARED so the "
        "check guards it"
    )
