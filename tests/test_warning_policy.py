"""The notebook warning policy keeps `case_studies.*` diagnostics audible.

The defect these tests pin is #1078: `warnings.filterwarnings("ignore")` at
notebook import silences every warning, so a diagnostic a library raises with
`warnings.warn` reaches nobody reading the executed notebook.
"""

from __future__ import annotations

import warnings

import pytest

from case_studies.utils.warning_policy import (
    apply_notebook_warning_policy,
    is_case_studies_module,
)

CASE_STUDY_SOURCE = "/home/x/ml4t/public/case_studies/utils/conformal.py"
ARCH_SOURCE = "/home/x/ml4t/public/.venv/lib/python3.14/site-packages/arch/univariate/base.py"
SKLEARN_SOURCE = (
    "/home/x/ml4t/public/.venv/lib/python3.14/site-packages/sklearn/linear_model/_sag.py"
)

ARCH_NOISE = "y is poorly scaled, which may affect convergence of the optimizer"
CASE_STUDY_DIAGNOSTIC = "Sortedness of columns cannot be checked when 'by' groups provided"


def _emit(message: str, source: str, category: type[Warning] = UserWarning) -> None:
    """Raise `message` as if it came from `source`.

    `warn_explicit` is what `warnings.warn` calls once it has resolved the
    caller's frame, so passing the filename directly reproduces how a real
    warning from that file is matched against the filters.
    """
    warnings.warn_explicit(message, category, source, 515)


def _under_policy(emit: list[tuple[str, str, type[Warning]]]) -> list[str]:
    """Return the messages that survive the policy."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.resetwarnings()
        apply_notebook_warning_policy()
        for message, source, category in emit:
            _emit(message, source, category)
    return [str(w.message) for w in recorded]


def test_blanket_ignore_hides_a_case_studies_diagnostic():
    """The defect, reproduced: the line the notebooks carry today silences it."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.resetwarnings()
        warnings.filterwarnings("ignore")
        _emit(CASE_STUDY_DIAGNOSTIC, CASE_STUDY_SOURCE)
    assert [str(w.message) for w in recorded] == []


def test_policy_keeps_a_case_studies_diagnostic_audible():
    survived = _under_policy([(CASE_STUDY_DIAGNOSTIC, CASE_STUDY_SOURCE, UserWarning)])
    assert survived == [CASE_STUDY_DIAGNOSTIC]


def test_policy_silences_the_measured_third_party_noise():
    """arch's DataScaleWarning subclasses Warning, not UserWarning."""

    class DataScaleWarning(Warning):
        pass

    survived = _under_policy([(ARCH_NOISE, ARCH_SOURCE, DataScaleWarning)])
    assert survived == []


def test_policy_does_not_silence_a_third_party_warning_that_reports_a_result():
    """A model that did not converge is a diagnostic, not noise."""
    from sklearn.exceptions import ConvergenceWarning

    message = "The max_iter was reached which means the coef_ did not converge"
    survived = _under_policy([(message, SKLEARN_SOURCE, ConvergenceWarning)])
    assert survived == [message]


def test_policy_is_idempotent():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.resetwarnings()
        apply_notebook_warning_policy()
        apply_notebook_warning_policy()
        _emit(CASE_STUDY_DIAGNOSTIC, CASE_STUDY_SOURCE)
        _emit(ARCH_NOISE, ARCH_SOURCE)
    assert [str(w.message) for w in recorded] == [CASE_STUDY_DIAGNOSTIC]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (CASE_STUDY_SOURCE, True),
        ("/home/x/ml4t/public/case_studies/research.py", True),
        (r"C:\ml4t\public\case_studies\utils\conformal.py", True),
        # Negative selftest: the pattern must not reach anything outside the
        # package, including a path that merely contains the word.
        (ARCH_SOURCE, False),
        ("/home/x/ml4t/public/utils/style.py", False),
        ("/home/x/my_case_studies_notes/scratch.py", False),
    ],
)
def test_module_pattern_selects_only_the_package(path: str, expected: bool):
    assert is_case_studies_module(path) is expected
