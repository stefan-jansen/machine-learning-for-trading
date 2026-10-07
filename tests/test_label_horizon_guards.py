"""``labels.buffer`` of ``16min`` has to become a minute embargo, not a crash.

NASDAQ-100's setup declares that buffer. ``load_protocol`` used to slice off
one character and call ``int("16mi")``, so ``get_cv_config`` for that study
raised before it could record an embargo.
"""

from __future__ import annotations

from utils.modeling import _label_horizon_guards, _label_horizon_to_iso, load_protocol


def test_a_minute_word_is_not_sliced_down_to_its_last_character() -> None:
    assert _label_horizon_guards("16min") == {"label_horizon_minutes": 16}
    assert _label_horizon_to_iso(_label_horizon_guards("16min")) == "PT16M"


def test_single_character_units_keep_their_readings() -> None:
    assert _label_horizon_guards("21D") == {"label_horizon_days": 21}
    assert _label_horizon_guards("8H") == {"label_horizon_hours": 8}
    assert _label_horizon_guards("8h") == {"label_horizon_hours": 8}
    assert _label_horizon_guards("60m") == {"label_horizon_minutes": 60}
    assert _label_horizon_guards("15T") == {"label_horizon_minutes": 15}
    # A month is stored as 30 days; pd.Timedelta rejects "M".
    assert _label_horizon_guards("1M") == {"label_horizon_days": 30}


def test_nasdaq_setup_embargo_is_sixteen_minutes() -> None:
    guards = load_protocol("nasdaq100_microstructure")["leakage_guards"]
    assert guards == {"label_horizon_minutes": 16}
    assert _label_horizon_to_iso(guards) == "PT16M"
