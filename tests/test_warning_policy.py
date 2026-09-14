"""The notebook warning policy keeps `case_studies.*` diagnostics audible.

The defect pinned here is ml4t/agent-workspace#1078: `warnings.filterwarnings("ignore")`
at notebook import silences every warning, so a diagnostic raised with
`warnings.warn` reaches nobody reading the executed notebook.

Every test drives a real `warnings.warn` from a real imported module. An earlier
version of this file used `warnings.warn_explicit` with a filename, which takes a
different branch: `warn_explicit` derives the module it matches from the filename,
while `warnings.warn` passes the caller's dotted `__name__`. The tests passed
against a pattern that matched nothing in production.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import warnings

import pytest

from case_studies.utils.warning_policy import (
    apply_notebook_warning_policy,
    is_case_studies_module,
)

ARCH_NOISE = "y is poorly scaled, which may affect convergence of the optimizer"
CASE_STUDY_DIAGNOSTIC = "Sortedness of columns cannot be checked when 'by' groups provided"


def _module_that_warns(tmp_path, dotted_name: str) -> types.ModuleType:
    """Import a module under `dotted_name` whose `warn()` raises a warning.

    The dotted name is what `warnings` matches a filter's `module` against, so it
    has to be the real name of a real imported module for the test to mean
    anything.
    """
    source = tmp_path / f"{dotted_name.replace('.', '_')}.py"
    source.write_text(
        "import warnings\ndef warn(message, category):\n    warnings.warn(message, category)\n"
    )
    spec = importlib.util.spec_from_file_location(dotted_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    assert module.__name__ == dotted_name
    return module


@pytest.fixture
def case_study_module(tmp_path):
    name = "case_studies.utils._warning_probe"
    try:
        yield _module_that_warns(tmp_path, name)
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def third_party_module(tmp_path):
    name = "arch.univariate._warning_probe"
    try:
        yield _module_that_warns(tmp_path, name)
    finally:
        sys.modules.pop(name, None)


def _audible(emit) -> list[str]:
    """Messages that survive the policy, with a blanket ignore also installed.

    The blanket ignore is what the notebooks carry today. Installing it first and
    the policy second is the migration state a notebook passes through, and it is
    the case where a keep-audible rule that matches nothing looks identical to one
    that works.
    """
    with warnings.catch_warnings(record=True) as recorded:
        warnings.resetwarnings()
        warnings.filterwarnings("ignore")
        apply_notebook_warning_policy()
        emit()
    return [str(w.message) for w in recorded]


def test_blanket_ignore_hides_a_case_studies_diagnostic(case_study_module):
    """The defect, reproduced against the line the notebooks carry today."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.resetwarnings()
        warnings.filterwarnings("ignore")
        case_study_module.warn(CASE_STUDY_DIAGNOSTIC, UserWarning)
    assert [str(w.message) for w in recorded] == []


def test_policy_keeps_a_case_studies_diagnostic_audible(case_study_module):
    survived = _audible(lambda: case_study_module.warn(CASE_STUDY_DIAGNOSTIC, UserWarning))
    assert survived == [CASE_STUDY_DIAGNOSTIC]


def test_policy_silences_the_measured_third_party_noise(third_party_module):
    """arch's DataScaleWarning subclasses Warning, not UserWarning."""

    class DataScaleWarning(Warning):
        pass

    with warnings.catch_warnings(record=True) as recorded:
        warnings.resetwarnings()
        warnings.simplefilter("always")
        apply_notebook_warning_policy()
        third_party_module.warn(ARCH_NOISE, DataScaleWarning)
    assert [str(w.message) for w in recorded] == []


def test_a_ignore_narrowed_to_userwarning_would_not_reach_it(third_party_module):
    """Why the entry does not name a category: the control for the test above."""

    class DataScaleWarning(Warning):
        pass

    with warnings.catch_warnings(record=True) as recorded:
        warnings.resetwarnings()
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=UserWarning, module="arch")
        third_party_module.warn(ARCH_NOISE, DataScaleWarning)
    assert [str(w.message) for w in recorded] == [ARCH_NOISE]


def test_policy_does_not_silence_a_third_party_warning_that_reports_a_result(third_party_module):
    """A model that did not converge is a diagnostic, not noise."""
    from sklearn.exceptions import ConvergenceWarning

    message = "The max_iter was reached which means the coef_ did not converge"
    with warnings.catch_warnings(record=True) as recorded:
        warnings.resetwarnings()
        warnings.simplefilter("always")
        apply_notebook_warning_policy()
        third_party_module.warn(message, ConvergenceWarning)
    assert [str(w.message) for w in recorded] == [message]


def test_policy_is_idempotent(case_study_module):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.resetwarnings()
        warnings.filterwarnings("ignore")
        apply_notebook_warning_policy()
        apply_notebook_warning_policy()
        case_study_module.warn(CASE_STUDY_DIAGNOSTIC, UserWarning)
    assert [str(w.message) for w in recorded] == [CASE_STUDY_DIAGNOSTIC]


def test_keep_audible_rule_does_not_reach_a_lookalike_package(tmp_path):
    """Negative selftest: a package whose name merely starts the same way."""
    name = "case_studies_notes.scratch"
    module = _module_that_warns(tmp_path, name)
    try:
        survived = _audible(lambda: module.warn("a note", UserWarning))
    finally:
        sys.modules.pop(name, None)
    assert survived == []


@pytest.mark.parametrize(
    ("dotted", "expected"),
    [
        ("case_studies.utils.conformal", True),
        ("case_studies.research", True),
        ("case_studies", True),
        ("case_studies_notes.scratch", False),
        ("arch.univariate.base", False),
        ("utils.style", False),
    ],
)
def test_module_pattern_selects_only_the_package(dotted: str, expected: bool):
    assert is_case_studies_module(dotted) is expected
