"""A label buffer is a duration, and the duration alone does not say what it counts.

`21D` on a daily equity panel means 21 sessions; subtracting 21 calendar days from the
holdout boundary leaks about six sessions into it. `sp500_options` declares `35D` for
`ret_to_expiry`, a genuine calendar horizon to option expiry; counting 35 sessions there
trims about seven weeks where five is correct. Both readings were live and neither was
declared, so every consumer guessed independently and each guess was wrong for one class
of label.
"""

from __future__ import annotations

import pytest

from utils.artifact_specs import DEFAULT_LABEL_BUFFER_UNIT, resolve_label_buffer_unit
from utils.cv_splits import _horizon_for_config, _purge_holdout_touching_validation


def test_the_default_is_sessions_because_that_is_what_a_D_buffer_already_meant() -> None:
    assert DEFAULT_LABEL_BUFFER_UNIT == "sessions"
    assert resolve_label_buffer_unit("etfs", "fwd_ret_21d", {"labels": {}}) == "sessions"


def test_setup_declares_the_unit_for_the_primary_label() -> None:
    setup = {"labels": {"primary": "ret_to_expiry", "buffer": "35D", "buffer_unit": "calendar"}}

    assert resolve_label_buffer_unit("sp500_options", "ret_to_expiry", setup) == "calendar"


def test_setup_declares_the_unit_per_variant() -> None:
    setup = {
        "labels": {
            "primary": "fwd_ret_5d",
            "variants": ["ret_to_expiry"],
            "variant_buffer_units": {"ret_to_expiry": "calendar"},
        }
    }

    assert resolve_label_buffer_unit("sp500_options", "fwd_ret_5d", setup) == "sessions"
    assert resolve_label_buffer_unit("sp500_options", "ret_to_expiry", setup) == "calendar"


def test_an_unknown_unit_is_refused_rather_than_read_as_the_default() -> None:
    setup = {"labels": {"primary": "ret_to_expiry", "buffer": "35D", "buffer_unit": "days"}}

    with pytest.raises(ValueError, match="buffer_unit is 'days'"):
        resolve_label_buffer_unit("sp500_options", "ret_to_expiry", setup)


def test_a_session_buffer_is_counted_and_a_calendar_one_is_not() -> None:
    """The int is what makes the splitter count sessions; the string is a duration."""
    assert _horizon_for_config("35D", calendar_id="NYSE", buffer_unit="sessions") == 35
    assert _horizon_for_config("35D", calendar_id="NYSE", buffer_unit="calendar") == "35D"


def test_without_a_calendar_there_are_no_sessions_to_count() -> None:
    assert _horizon_for_config("8h", calendar_id=None, buffer_unit="sessions") == "8h"
    assert _horizon_for_config("1D", calendar_id=None, buffer_unit="sessions") == "1D"


def test_the_purge_reads_the_same_declaration_as_the_fold_geometry() -> None:
    """A calendar horizon purged as sessions removes decisions the label never reaches."""
    import numpy as np
    import pandas as pd

    sessions = pd.bdate_range("2020-11-02", "2021-01-29")
    val_idx = np.arange(len(sessions))
    boundary = "2021-01-01"

    as_sessions = _purge_holdout_touching_validation(
        val_idx,
        sessions,
        holdout_start=boundary,
        outcome_horizon="35D",
        calendar_id="NYSE",
        buffer_unit="sessions",
    )
    as_calendar = _purge_holdout_touching_validation(
        val_idx,
        sessions,
        holdout_start=boundary,
        outcome_horizon="35D",
        calendar_id="NYSE",
        buffer_unit="calendar",
    )

    assert len(as_sessions) < len(as_calendar)
    assert sessions[as_calendar[-1]] < pd.Timestamp(boundary) - pd.Timedelta("35D")
