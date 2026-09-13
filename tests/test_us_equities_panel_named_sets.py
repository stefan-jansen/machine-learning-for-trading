"""Every named set `15` and `16` open must be frozen by a modelling notebook.

`15_model_analysis` and `16_backtest` do not read the registry. They name candidate sets
and open them with `CandidateSet.one`, which exists so the strategy chain is an explicit
list rather than whatever rows happen to be present. The names are string literals in one
notebook and the `study.predictions.freeze` calls that create them are string literals in
another, so nothing connects them and a name can be requested that nothing writes.

That is what happened. `eb7ac4e2` moved `15` from two opaque hashes onto named sets by
transcribing the README's pipeline table, which lists a stage per family. `06_linear` and
`07_gbm` publish a *population* and never call `freeze`, so four of the names it wrote -
`us-equities-fwd-ret-1d-linear-v1`, `-gbm-v1`, and their two `-diagnostics-v1`
counterparts - were produced by nobody. `12_dl_weekly` publishes nothing at all, so its
two candidate-set names are in the same state (a third weekly name, the official
population `us-equities-weekly-checkpoints-v1`, is opened by `15` through
`OfficialPopulation.one` and is not covered here).

The two notebooks fail differently, which is why this went unnoticed. `15` raises when
`CandidateSet.one` cannot find a name. `16` has no completeness check against the model
populations: it opens the names it was given, so a missing name is a silently narrower
strategy chain. The README's headline result is a 5-day GBM lineage, and before this test
`16` named no 5-day GBM set at all.

These tests read source rather than executing: the freeze calls fire only on an
unnarrowed canonical CUDA run, which is a production run costing hours, so there is no
reduced execution that reaches them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

CASE_DIR = Path("case_studies/us_equities_panel")

# Notebooks that open named sets, and the parameter lists they open.
CONSUMERS = {
    "15_model_analysis.py": ["PREDICTION_SET_NAMES", "DIAGNOSTIC_SET_NAMES"],
    "16_backtest.py": ["PREDICTION_SET_NAMES"],
}

# Notebooks that freeze named sets. `12_dl_weekly` is absent because it freezes nothing, and
# nothing asks it to: its predictions sit on a Friday grid and are not candidates in the
# strategy chain. See the disposition note below.
PRODUCERS = [
    "06_linear.py",
    "07_gbm.py",
    "08_tabular_dl.py",
    "09_dl_nlinear.py",
    "10_dl_lstm.py",
    "11_dl_tsmixer.py",
    "13a_pca.py",
    "13b_ipca.py",
]

# `12_dl_weekly`'s disposition, settled: it runs and registers, and it names no set that the
# strategy chain opens. Its `fwd_ret_5d` predictions are scored on Fridays only, so ranking them
# against the daily families on the same label would compare series measured over different sets
# of decision dates. `15` and `16` therefore name no `-weekly-` set, and nothing here is pending.
PENDING_WEEKLY: set[str] = set()


def _literal_string_list(path: Path, name: str) -> list[str]:
    """Return the string literals assigned to a module-level list parameter."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if name not in targets:
            continue
        if not isinstance(node.value, ast.List):
            pytest.fail(f"{path.name}: {name} is not a list literal")
        return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    pytest.fail(f"{path.name}: no assignment to {name}")


def _frozen_names(path: Path) -> set[str]:
    """Names passed to `study.predictions.freeze(..., name=...)`, f-strings resolved.

    The producers build the name as `f"us-equities-{label_name}-linear-v1"`, where
    `label_name` is a label with underscores replaced by dashes. Expanding that against
    the three declared labels is the point: a producer that fits one label while the
    consumer names three is exactly the defect, and it has to be visible here.

    A producer may bind that f-string to a local first and pass the local, which is what
    happens where the same name is also handed to `candidate_set_supersedes` - repeating
    the literal at both call sites is how the two drift apart. So a bare `ast.Name` is
    resolved against the module's assignments before it is given up on.

    **Anything still unresolved fails the test rather than contributing nothing.** Returning
    an empty set for a form the parser does not recognise makes every consumer name look
    unfrozen, which reads as sixteen missing producers rather than as one unparsed call -
    and that is exactly what a `name=<local>` call produced before this branch.
    """
    labels = ["fwd-ret-1d", "fwd-ret-5d", "fwd-ret-21d"]
    source = path.read_text()
    names: set[str] = set()
    tree = ast.parse(source)

    bindings: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node.value

    def resolve(value: ast.expr, origin: str) -> None:
        if isinstance(value, ast.Name):
            bound = bindings.get(value.id)
            if bound is None:
                pytest.fail(f"{path.name}: freeze name {value.id!r} is never assigned")
            resolve(bound, f"{origin} via {value.id}")
            return
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            names.add(value.value)
            return
        if isinstance(value, ast.JoinedStr):
            template = "".join(
                part.value if isinstance(part, ast.Constant) else "{}" for part in value.values
            )
            if template.count("{}") != 1:
                pytest.fail(f"{path.name}: cannot resolve freeze name {template!r}")
            names.update(template.format(label) for label in labels)
            return
        pytest.fail(f"{path.name}: cannot resolve the freeze name at {origin}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "freeze"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "name":
                resolve(keyword.value, f"line {keyword.value.lineno}")
    return names


@pytest.fixture(scope="module")
def produced() -> set[str]:
    names: set[str] = set()
    for stem in PRODUCERS:
        names |= _frozen_names(CASE_DIR / stem)
    return names


@pytest.mark.parametrize(
    ("consumer", "parameter"),
    [(nb, param) for nb, params in CONSUMERS.items() for param in params],
)
def test_every_named_set_is_frozen_by_a_producer(
    consumer: str, parameter: str, produced: set[str]
) -> None:
    requested = _literal_string_list(CASE_DIR / consumer, parameter)
    assert requested, f"{consumer}: {parameter} is empty"
    missing = sorted(set(requested) - produced - PENDING_WEEKLY)
    assert not missing, (
        f"{consumer} opens {parameter} entries that no notebook freezes: {missing}. "
        "Either a producer is missing its `study.predictions.freeze` call, or the name "
        "is stale and should come out of the list."
    )


# How many labels each producer fits, which is what decides how many horizons a consumer may
# name. `06_linear` and `07_gbm` take a `LABELS` parameter and loop, so they fit every declared
# label. Every other producer reads `PRIMARY_LABEL or setup["labels"]["primary"]` and fits one,
# and `13a_pca`/`13b_ipca` fit the labels whose `config/training/{label}.yaml` declares
# `latent_factors`, which is the primary label alone. A name at a horizon its producer never
# reaches raises in `15` and silently narrows the strategy chain in `16`; a producer fitting a
# horizon no consumer names throws that fit away.
ALL_HORIZONS = {"1d", "5d", "21d"}
PRIMARY_HORIZON = {"1d"}
EXPECTED_HORIZONS = {
    "linear": ALL_HORIZONS,
    "gbm": ALL_HORIZONS,
    "tabular-dl": PRIMARY_HORIZON,
    "nlinear": PRIMARY_HORIZON,
    "lstm": PRIMARY_HORIZON,
    "tsmixer": PRIMARY_HORIZON,
    "pca": PRIMARY_HORIZON,
    "ipca": PRIMARY_HORIZON,
}


def test_producers_cover_every_label_the_consumers_ask_for() -> None:
    """A family must be named at exactly the horizons its producer fits.

    The narrower version of this defect: `06` fits all three declared labels in one
    population, so a consumer naming only the 1-day set drops two thirds of what was
    fitted, and drops it silently in `16`. The reverse costs more - a consumer naming a
    horizon nothing freezes makes `CandidateSet.one` raise at the first cell of a stage
    that has to run before any backtest exists.
    """
    # Per list, not over their union. The defect this test was written for is `16` naming no
    # 5-day GBM set while `15` named one, and a union of the two consumers cannot see it.
    for consumer, parameters in CONSUMERS.items():
        for parameter in parameters:
            requested = set(_literal_string_list(CASE_DIR / consumer, parameter))

            by_family: dict[str, set[str]] = {}
            for name in requested - PENDING_WEEKLY:
                match = re.fullmatch(r"us-equities-fwd-ret-(\d+d)-(.+?)(-diagnostics)?-v1", name)
                assert match, f"{consumer}: unrecognised set name shape: {name}"
                horizon, family, diagnostics = match.groups()
                by_family.setdefault(f"{family}{diagnostics or ''}", set()).add(horizon)

            unknown = sorted(
                family
                for family in by_family
                if family.removesuffix("-diagnostics") not in EXPECTED_HORIZONS
            )
            assert not unknown, (
                f"{consumer}: no declared horizon coverage for {unknown}. Add an "
                "EXPECTED_HORIZONS entry saying how many labels the producer fits."
            )

            for family in sorted(by_family):
                expected = EXPECTED_HORIZONS[family.removesuffix("-diagnostics")]
                assert by_family[family] == expected, (
                    f"{consumer} {parameter}: {family} is fitted at {sorted(expected)} "
                    f"but named at {sorted(by_family[family])}"
                )


def test_diagnostic_sets_are_bounded() -> None:
    """`15` holds every diagnostic frame in memory and correlates them pairwise.

    Cost is quadratic in diagnostic members, so a diagnostic set must be a bounded subset
    of its family rather than the whole grid. `06` declares sixteen configurations and
    `07` declares fifteen at several checkpoints each; naming the full set as its own
    diagnostic set is what makes `15` unrunnable on this panel.
    """
    for stem in ("06_linear.py", "07_gbm.py", "08_tabular_dl.py"):
        declared = _literal_string_list(CASE_DIR / stem, "DIAGNOSTIC_CONFIG_NAMES")
        assert declared, f"{stem}: DIAGNOSTIC_CONFIG_NAMES is empty"
        assert len(declared) <= 3, (
            f"{stem}: {len(declared)} diagnostic configurations. `15` correlates these "
            "pairwise across every declared label; keep the subset small."
        )
