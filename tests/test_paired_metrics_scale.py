"""The paired-bootstrap scale comes from the case study, not from a default."""

from __future__ import annotations

import inspect

import pytest

from case_studies.utils.paired_metrics import _min_paired_n, populate_paired_metrics
from case_studies.utils.uncertainty import periods_per_year_from_setup

# Declared cadences that are not the 252 the old default assumed. A case study added
# here with a different declaration is caught by the first test without being listed.
NON_DAILY = ("us_firm_characteristics", "crypto_perps_funding")


def test_annualization_defaults_to_the_case_study_declaration() -> None:
    """Omitting the scale must give the case study its own, not somebody else's.

    `populate_paired_metrics` took `freq: str = "daily"` and resolved it to 252.
    That is right for the six case studies that annualize daily and wrong for the
    rest, silently: on a monthly case study every Sharpe difference and interval it
    wrote was scaled by sqrt(252) instead of sqrt(12).

    The signature is what this asserts, because calling the function needs a
    populated registry. A default of `None` is what routes the omitted case to the
    declaration; a literal default is what reintroduces the defect.
    """
    signature = inspect.signature(populate_paired_metrics)
    assert "freq" not in signature.parameters, (
        "`freq` names a cadence that only gets converted back to a number, and its "
        "literal default is what gave non-daily case studies the wrong scale"
    )
    parameter = signature.parameters["periods_per_year"]
    assert parameter.default is None, (
        "the default must be None so the case study's own "
        "evaluation.periods_per_year is read; a literal here is the old defect"
    )


@pytest.mark.parametrize("case_study", NON_DAILY)
def test_the_declaration_these_case_studies_rely_on_is_not_daily(case_study: str) -> None:
    """Guards the premise: if these became 252 the test above would prove nothing."""
    assert periods_per_year_from_setup(case_study) != 252


def test_the_minimum_length_rule_admits_a_twelve_month_holdout() -> None:
    """A monthly holdout is twelve observations by design, not a short series.

    `_min_paired_n(252)` returns 21, so under the old default both holdout pairs
    were skipped and the table was missing rows with nothing recording it. That is
    the half of the defect where the numbers are absent rather than mis-scaled.
    """
    assert _min_paired_n(periods_per_year_from_setup("us_firm_characteristics")) <= 12
    assert _min_paired_n(252) == 21
