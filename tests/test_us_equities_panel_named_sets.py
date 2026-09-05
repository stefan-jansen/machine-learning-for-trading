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

# Notebooks that freeze named sets. `12_dl_weekly` is deliberately absent: it freezes
# nothing today, which is why the `-weekly-` names are carried as pending below.
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

# Names `12_dl_weekly` would have to freeze once its disposition is settled. Listing them
# here rather than deleting them from `15`/`16` keeps the gap visible: whoever migrates or
# retires `12` has to come here and decide, instead of finding a silently shorter chain.
PENDING_WEEKLY = {
    "us-equities-fwd-ret-5d-weekly-v1",
    "us-equities-fwd-ret-5d-weekly-diagnostics-v1",
}


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
    """
    labels = ["fwd-ret-1d", "fwd-ret-5d", "fwd-ret-21d"]
    source = path.read_text()
    names: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "freeze"):
            continue
        for keyword in node.keywords:
            if keyword.arg != "name":
                continue
            value = keyword.value
            if isinstance(value, ast.Constant):
                names.add(value.value)
            elif isinstance(value, ast.JoinedStr):
                template = "".join(
                    part.value if isinstance(part, ast.Constant) else "{}" for part in value.values
                )
                if template.count("{}") == 1:
                    names.update(template.format(label) for label in labels)
                else:
                    pytest.fail(f"{path.name}: cannot resolve freeze name {template!r}")
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


def test_producers_cover_every_label_the_consumers_ask_for() -> None:
    """A family named at three labels must be frozen at three labels.

    The narrower version of this defect: `06` fits all three declared labels in one
    population, so a consumer naming only the 1-day set drops two thirds of what was
    fitted, and drops it silently in `16`.
    """
    requested: set[str] = set()
    for consumer, parameters in CONSUMERS.items():
        for parameter in parameters:
            requested.update(_literal_string_list(CASE_DIR / consumer, parameter))

    by_family: dict[str, set[str]] = {}
    for name in requested - PENDING_WEEKLY:
        match = re.fullmatch(r"us-equities-fwd-ret-(\d+d)-(.+?)(-diagnostics)?-v1", name)
        assert match, f"unrecognised set name shape: {name}"
        horizon, family, diagnostics = match.groups()
        by_family.setdefault(f"{family}{diagnostics or ''}", set()).add(horizon)

    multi_label = {"linear", "gbm", "linear-diagnostics", "gbm-diagnostics", "pca", "ipca"}
    for family in sorted(multi_label & set(by_family)):
        assert by_family[family] == {"1d", "5d", "21d"}, (
            f"{family} is fitted on all three declared labels but named at "
            f"{sorted(by_family[family])}"
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
