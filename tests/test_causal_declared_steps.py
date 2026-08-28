"""A buffer declared in months has to be counted in months.

`resolve_causal_request` converts the declared CV buffer and outcome horizon into counts of
the panel's own observations, because that is what the per-entity seal, the placebo block and
the HAC bandwidth are all denominated in. It did the conversion by dividing
`pd.Timedelta(declaration)` by the measured cadence, which cannot be reached at all for
`us_firm_characteristics`: pandas refuses `1M` outright, since January and February are
different lengths and no single answer is right for both. The notebook failed on its first
cell that resolves a request, before any fit.

Counting months as observations is only valid where an observation is a month, so the panel
is checked rather than assumed. Every other unit is a fixed span and keeps the division it
already had, which is what leaves the four converted case studies on the identities they
have published.
"""

from __future__ import annotations

import pandas as pd
import pytest

from case_studies.utils.causal import _declared_steps

MONTHLY = pd.Timedelta(days=30)
DAILY = pd.Timedelta(days=1)
EIGHT_HOURLY = pd.Timedelta(hours=8)


class TestACalendarMonth:
    def test_one_month_is_one_observation_on_a_monthly_panel(self) -> None:
        assert _declared_steps("1M", MONTHLY, field="labels.buffer") == (1, "1M")

    @pytest.mark.parametrize("cadence", [pd.Timedelta(days=28), pd.Timedelta(days=31)])
    def test_the_month_the_sample_lands_on_does_not_change_the_count(self, cadence) -> None:
        # The modal gap of a monthly panel is whichever month is most common in the sample,
        # so a check that demanded one exact length would pass or fail on the calendar.
        assert _declared_steps("3M", cadence, field="labels.buffer")[0] == 3

    def test_a_daily_panel_cannot_honour_a_month(self) -> None:
        # The alternative is substituting a nominal month, which is wrong by however far
        # this panel's month differs from thirty days and does not show in the result.
        with pytest.raises(ValueError, match="only on a panel whose observations are months"):
            _declared_steps("1M", DAILY, field="labels.buffer")


class TestAFixedDuration:
    @pytest.mark.parametrize(
        ("declaration", "cadence", "steps"),
        [
            ("5D", DAILY, 5),
            ("21D", DAILY, 21),
            ("1D", DAILY, 1),
            ("8H", EIGHT_HOURLY, 1),
            ("0D", DAILY, 1),
        ],
    )
    def test_the_count_is_the_span_over_the_cadence(self, declaration, cadence, steps) -> None:
        assert _declared_steps(declaration, cadence, field="labels.buffer")[0] == steps

    def test_the_recorded_string_is_the_one_the_published_specs_carry(self) -> None:
        # `estimand.outcome_horizon` is hashed into the causal identity, so this string is
        # not free to move: changing it would give every converted case study a second
        # identity for a fit that did not change.
        assert _declared_steps("5D", DAILY, field="labels.horizons")[1] == "5 days 00:00:00"

    def test_the_deprecated_uppercase_hour_alias_still_parses(self) -> None:
        assert _declared_steps("8H", EIGHT_HOURLY, field="labels.buffer") == (
            1,
            "0 days 08:00:00",
        )

    def test_a_span_shorter_than_a_bar_still_covers_the_bar_it_resolves_inside(self) -> None:
        assert _declared_steps("15min", pd.Timedelta(days=1), field="labels.buffer")[0] == 1
