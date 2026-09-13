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


# Families the strategy chain does not name, and the one reason each is allowed to be missing.
# They are two constants rather than one because they are drained by different things: a pending
# entry is removed by the run that produces its sets, and a ruled entry is removed by nothing.
# Merging them would mean that whoever empties the list after the full backtest pass either
# re-admits a family a ruling dropped or leaves out one whose producer has since run.
#
# The reason lives in the value, not in a comment above the entry, because a comment does not
# move when the entry does.

# A scope ruling, and nothing else. AGENTS.md is explicit that expense alone does not qualify:
# a notebook excluded this way keeps correct committed source and cleared outputs and is still
# runnable by a reader, so the names below are absent from the chain and present in the repo.
EXCLUDED_BY_RULING = {
    "tsmixer": (
        "dropped by Stefan 2026-09-12, agents 30d3766bf: 60-80 h of GPU for a third "
        "deep-learning architecture on the primary label, where nlinear and lstm cost about "
        "50 between them and 08_tabular_dl already covers a third branch of the literature. "
        "11_dl_tsmixer.py stays in the repo and registers nothing."
    ),
}

# A producer that has not run yet. `CandidateSet.one` raises on a name nothing froze, so a
# stage that has to run before the sets exist cannot name them. Every entry here is a
# narrower strategy chain than the case study intends, so each one is temporary by
# construction and the run that produces its sets removes it.
PENDING_PRODUCTION: dict[str, str] = {}

# The third list `15` opens by name. `OFFICIAL_POPULATION_NAMES` holds checkpoint populations
# rather than candidate sets, so it carries no horizon and is shaped
# `us-equities-{family}-checkpoints-v1`. It is checked for family coverage for the same reason
# the other two are: `OfficialPopulation.one` raises on a name nothing published, and before
# this it was the one consumer list with nothing reading it at all.
POPULATION_CONSUMERS = {"15_model_analysis.py": ["OFFICIAL_POPULATION_NAMES"]}

SET_NAME = re.compile(r"us-equities-fwd-ret-(\d+d)-(.+?)(-diagnostics)?-v1")
POPULATION_NAME = re.compile(r"us-equities-(.+?)-checkpoints-v1")


def _families_named(consumer: str, parameter: str, pattern: re.Pattern[str]) -> set[str]:
    """The model families a consumer list names, with any `-diagnostics` suffix removed."""
    families = set()
    for name in set(_literal_string_list(CASE_DIR / consumer, parameter)) - PENDING_WEEKLY:
        match = pattern.fullmatch(name)
        assert match, f"{consumer} {parameter}: unrecognised name shape: {name}"
        families.add(match.group(2) if pattern is SET_NAME else match.group(1))
    return families


def _all_consumer_lists() -> list[tuple[str, str, re.Pattern[str]]]:
    lists = [(nb, param, SET_NAME) for nb, params in CONSUMERS.items() for param in params]
    lists += [
        (nb, param, POPULATION_NAME)
        for nb, params in POPULATION_CONSUMERS.items()
        for param in params
    ]
    return lists


def test_the_chain_names_every_family_no_declaration_excuses() -> None:
    """A family missing from a consumer list has to say why it is missing.

    `test_producers_cover_every_label_the_consumers_ask_for` builds its family table from the
    names that are PRESENT, so it says nothing about a family that is absent altogether -
    dropping five families from `16_backtest` passes it green. That is the same blind spot in
    the other direction from the one this module was written for, and it is the one an interim
    run would walk into: `16` can be narrowed to the sets that happen to exist, publish a
    narrower comparison than the case study intends, and never fail anything.
    """
    excused = set(EXCLUDED_BY_RULING) | set(PENDING_PRODUCTION)
    for consumer, parameter, pattern in _all_consumer_lists():
        missing = sorted(
            set(EXPECTED_HORIZONS) - excused - _families_named(consumer, parameter, pattern)
        )
        assert not missing, (
            f"{consumer} {parameter} names no set for {missing}. Either the family belongs in "
            "the list, or say why it is absent: EXCLUDED_BY_RULING for a scope ruling, "
            "PENDING_PRODUCTION for a producer that has not run yet."
        )


def test_an_excused_family_is_not_also_named() -> None:
    """Excusing a family and naming it anyway is a declaration nobody reads."""
    excused = set(EXCLUDED_BY_RULING) | set(PENDING_PRODUCTION)
    for consumer, parameter, pattern in _all_consumer_lists():
        named = sorted(excused & _families_named(consumer, parameter, pattern))
        assert not named, (
            f"{consumer} {parameter} names {named}, which EXCLUDED_BY_RULING or "
            "PENDING_PRODUCTION says is absent. Remove the name or remove the entry."
        )


def test_every_producer_this_module_names_still_exists() -> None:
    """`_frozen_names` reads each producer, so a deleted one fails as a missing file.

    Checking it here turns `FileNotFoundError` inside a fixture into a sentence saying which
    notebook went and what still refers to it.
    """
    missing = sorted(stem for stem in PRODUCERS if not (CASE_DIR / stem).exists())
    assert not missing, f"PRODUCERS names notebooks that are not in the repo: {missing}"


def test_a_ruled_exclusion_still_describes_this_case_study(produced: set[str]) -> None:
    """A ruling excludes a family from the chain; it does not delete the notebook.

    Nothing else in the repo asserts that. The register row keeps the notebook at `ready`,
    and `register.sh check` compares a row against main rather than asking whether the file
    is there, so an exclusion left standing over a deleted producer would read as an
    ordinary exclusion. That state is what the ruling that created this constant forbids.
    """
    for family, reason in EXCLUDED_BY_RULING.items():
        assert reason.strip(), f"EXCLUDED_BY_RULING[{family!r}] has no reason recorded"
        assert family in EXPECTED_HORIZONS, (
            f"EXCLUDED_BY_RULING names {family!r}, which this case study has no producer for. "
            "An exclusion is a statement about a family that exists."
        )
        frozen = {name for name in produced if SET_NAME.fullmatch(name)}
        families = {SET_NAME.fullmatch(name).group(2) for name in frozen}
        assert family in families, (
            f"EXCLUDED_BY_RULING names {family!r} but no notebook in PRODUCERS freezes a "
            f"{family} set any more. A ruling excludes a family from the strategy chain and "
            "leaves the notebook in the repo, so the producer going missing is a separate "
            "change that has to be made deliberately."
        )
