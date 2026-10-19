"""A declared supersedes value reaches the registry through a resolver, never raw.

`OfficialPopulation.create` refuses both ends of the lineage. It refuses a changed member
list that names no predecessor - `case_studies/research/population.py`, `a changed
population named ... must explicitly supersedes ...` - and it refuses a *first* generation
that names one: `first population version cannot supersede another snapshot`. A notebook
answers the first refusal by setting its `SUPERSEDES_*` parameter, and that same edit walks
into the second, because `run_log/` is gitignored and every reader starts with an empty
registry. The author's run publishes generation two; the reader's clone dies on the publish.

`population_supersedes`, `candidate_set_supersedes`, `causal_supersedes` and
`supersedes_for_run` exist to decide that, and they withhold the declaration wherever
offering it would be refused. So the declaration is only ever correct when it is resolved,
and `SUPERSEDES_LIVE` makes resolution mandatory rather than optional: the string `"live"`
is not a hash, so passing it raw is refused against any registry that holds a generation.

The error this prevents, in the shape it arose: on 2026-10-15 the four `08_tabular_dl`
notebooks for etfs, cme_futures, crypto_perps_funding and sp500_equity_option_analytics
passed their parameter straight into the publish call. All four were harmless only because
their declared value was empty, and all four were about to be given a hash to answer a
refusal after the corrected TabM ensemble re-keyed every training identity. Every other
notebook in the corpus already routed its declaration through a resolver, so the structural
difference, not the value, is what this test reads.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CASE_STUDIES = REPO / "case_studies"

RESOLVERS = frozenset(
    {
        "population_supersedes",
        "candidate_set_supersedes",
        "causal_supersedes",
        "supersedes_for_run",
    }
)

# The keyword a publish call takes the resolved value under. `declared=` is a resolver's own
# parameter, so a name reaching one of those is already being resolved.
PUBLISH_KEYWORDS = frozenset({"supersedes"})


def _declared_supersedes(tree: ast.Module) -> set[str]:
    """Module-level `SUPERSEDES_*` names whose default is a non-empty string."""
    declared: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        else:
            continue
        if value is None:
            continue
        # A non-empty literal, or a reference to the SUPERSEDES_LIVE constant. Either way the
        # value is something `create` refuses to receive raw against a reader's registry.
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            non_empty = value.value != ""
        elif isinstance(value, ast.Name):
            non_empty = value.id == "SUPERSEDES_LIVE"
        else:
            non_empty = False
        if not non_empty:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.startswith("SUPERSEDES_"):
                declared.add(target.id)
    return declared


def _names_in(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _raw_publishes(tree: ast.Module, declared: set[str]) -> list[tuple[str, int]]:
    """Publish calls taking a declared name without a resolver between them."""
    findings: list[tuple[str, int]] = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in RESOLVERS:
            continue
        for keyword in call.keywords:
            if keyword.arg not in PUBLISH_KEYWORDS or keyword.value is None:
                continue
            used = _names_in(keyword.value) & declared
            if not used:
                continue
            # A resolver anywhere inside the argument is what makes it resolved, which covers
            # both `supersedes=resolver(...)` and a dict comprehension over several lineages.
            resolved = any(
                (
                    inner.func.attr
                    if isinstance(inner.func, ast.Attribute)
                    else getattr(inner.func, "id", "")
                )
                in RESOLVERS
                for inner in ast.walk(keyword.value)
                if isinstance(inner, ast.Call)
            )
            if not resolved:
                findings.append((sorted(used)[0], keyword.value.lineno))
    return findings


def _notebook_sources() -> list[Path]:
    return sorted(
        path for path in CASE_STUDIES.glob("*/[0-9]*.py") if not path.name.startswith("_")
    )


def test_every_declared_supersedes_is_resolved_before_the_publish() -> None:
    offenders: dict[str, list[tuple[str, int]]] = {}
    scanned = 0
    for path in _notebook_sources():
        tree = ast.parse(path.read_text())
        declared = _declared_supersedes(tree)
        if not declared:
            continue
        scanned += 1
        findings = _raw_publishes(tree, declared)
        if findings:
            offenders[str(path.relative_to(REPO))] = findings
    assert scanned >= 20, (
        f"only {scanned} notebooks declare a non-empty supersedes value; the scan found "
        "nothing to check and would pass vacuously"
    )
    assert not offenders, (
        "a declared supersedes value reaches the publish unresolved, which refuses a "
        f"reader's clean clone after the fit: {offenders}"
    )


def test_the_unresolved_shape_is_detected() -> None:
    """The negative selftest: the shape this rule forbids has to be refused.

    Written as source rather than taken from a notebook, because the corpus holds no
    violation to point at - which is the thing that makes an all-pass result uninformative
    on its own.
    """
    offending = ast.parse(
        "SUPERSEDES_POPULATION: str = '33cefe348c67'\n"
        "population = plan.create_population(\n"
        "    name=population_name, supersedes=SUPERSEDES_POPULATION or None\n"
        ")\n"
    )
    declared = _declared_supersedes(offending)
    assert declared == {"SUPERSEDES_POPULATION"}
    assert _raw_publishes(offending, declared) == [("SUPERSEDES_POPULATION", 3)]

    resolved = ast.parse(
        "SUPERSEDES_POPULATION: str = SUPERSEDES_LIVE\n"
        "population = plan.create_population(\n"
        "    name=population_name,\n"
        "    supersedes=population_supersedes(\n"
        "        study, name=population_name, declared=SUPERSEDES_POPULATION\n"
        "    ),\n"
        ")\n"
    )
    declared = _declared_supersedes(resolved)
    assert declared == {"SUPERSEDES_POPULATION"}
    assert _raw_publishes(resolved, declared) == []

    # An empty default is the one case the raw pass is harmless in, and it is how every
    # notebook that has never published a second generation reads.
    empty = ast.parse(
        "SUPERSEDES_POPULATION: str = ''\n"
        "population = plan.create_population(\n"
        "    name=population_name, supersedes=SUPERSEDES_POPULATION or None\n"
        ")\n"
    )
    assert _declared_supersedes(empty) == set()
