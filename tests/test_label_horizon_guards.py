"""An unreadable ``labels.buffer`` must raise, not become an embargo of zero.

Two defects, one shipped and one latent. ``load_protocol`` sliced one character
off the end of the buffer string, so ``nasdaq100_microstructure``'s ``16min``
matched no branch; the chain then returned an empty guard dict and
``_label_horizon_to_iso`` turned that into ``"P0D"``. The case study ran its
walk-forward with NO embargo between train and test, and nothing raised, logged
or recorded that it had happened. The crash the first report described would
have been the better outcome.

So these tests pin two things: every buffer the nine case studies actually
declare keeps the horizon it has always had, and anything the parser cannot read
raises with a message that names the value, the accepted formats and the file.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from utils.modeling import (
    _label_horizon_guards,
    _label_horizon_to_iso,
    get_cv_config,
    load_protocol,
)

# The buffer each case study declares, and the embargo it has to keep producing.
# Read off config/setup.yaml; a change here is a change to a published protocol.
DECLARED = {
    "cme_futures": "P5D",
    "crypto_perps_funding": "PT8H",
    "etfs": "P21D",
    "fx_pairs": "P1D",
    "nasdaq100_microstructure": "PT16M",
    "sp500_equity_option_analytics": "P10D",
    "sp500_options": "P35D",
    "us_equities_panel": "P1D",
    "us_firm_characteristics": "P30D",
}


def _setup(tmp_path: Path, buffer: str) -> Path:
    """Write a minimal case study setup.yaml the real loader will read."""
    case = tmp_path / "probe" / "config"
    case.mkdir(parents=True)
    (case / "setup.yaml").write_text(
        textwrap.dedent(f"""
            evaluation:
              n_splits: 3
              train_size: 5Y
              val_size: 1Y
            labels:
              buffer: {buffer}
        """)
    )
    return tmp_path


@pytest.mark.parametrize(("case_study", "embargo"), sorted(DECLARED.items()))
def test_every_declared_buffer_keeps_its_embargo(case_study: str, embargo: str) -> None:
    assert get_cv_config(case_study).embargo_td == embargo
    assert get_cv_config(case_study).label_horizon == embargo


def test_a_minute_word_is_not_sliced_down_to_its_last_character() -> None:
    assert _label_horizon_guards("16min") == {"label_horizon_minutes": 16}
    assert _label_horizon_to_iso(_label_horizon_guards("16min")) == "PT16M"


def test_single_character_units_keep_their_readings() -> None:
    assert _label_horizon_guards("21D") == {"label_horizon_days": 21}
    assert _label_horizon_guards("8H") == {"label_horizon_hours": 8}
    assert _label_horizon_guards("8h") == {"label_horizon_hours": 8}
    assert _label_horizon_guards("60m") == {"label_horizon_minutes": 60}
    assert _label_horizon_guards("15T") == {"label_horizon_minutes": 15}
    # A month is carried as 30 days; pd.Timedelta rejects "M" as ambiguous.
    assert _label_horizon_guards("1M") == {"label_horizon_days": 30}


def test_iso_and_compound_durations_parse_instead_of_vanishing() -> None:
    """None of these matched the old chain, so each one meant no embargo."""
    assert _label_horizon_guards("P21D") == {"label_horizon_days": 21}
    assert _label_horizon_guards("PT16M") == {"label_horizon_minutes": 16}
    assert _label_horizon_guards("5W") == {"label_horizon_days": 35}
    assert _label_horizon_guards("2h30min") == {"label_horizon_minutes": 150}


def test_a_zero_buffer_is_the_only_way_to_declare_no_embargo() -> None:
    assert _label_horizon_to_iso(_label_horizon_guards("0D")) == "P0D"


@pytest.mark.parametrize("buffer", ["abc", "", "   ", "21 days or so", "fortnight"])
def test_an_unreadable_buffer_raises_and_names_the_accepted_formats(buffer: str) -> None:
    with pytest.raises(ValueError, match=r"labels\.buffer") as raised:
        _label_horizon_guards(buffer)
    assert "16min" in str(raised.value), "the message has to show a format that works"


def test_a_negative_buffer_raises() -> None:
    with pytest.raises(ValueError, match="cannot run backwards"):
        _label_horizon_guards("-5D")


def test_a_sub_minute_buffer_raises_rather_than_rounding_to_zero() -> None:
    with pytest.raises(ValueError, match="not a whole number of minutes"):
        _label_horizon_guards("30s")


@pytest.mark.parametrize("buffer", ["0.5s", "0.9s", "100ms", "1us"])
def test_a_sub_second_buffer_raises_rather_than_truncating_to_no_embargo(buffer: str) -> None:
    """The silent zero, reached the other way: int() truncates to 0 seconds.

    0 seconds is a whole number of days, so it passed the day test and came back as
    ``P0D`` - the same no-embargo this parser exists to refuse, under a value that
    looks like a declared buffer.
    """
    with pytest.raises(ValueError, match="fraction of a second"):
        _label_horizon_guards(buffer)


def test_a_fractional_duration_that_lands_on_a_whole_unit_still_parses() -> None:
    assert _label_horizon_guards("1.5D") == {"label_horizon_hours": 36}
    assert _label_horizon_guards("2.5h") == {"label_horizon_minutes": 150}


def test_guards_with_no_horizon_raise_instead_of_returning_a_zero_embargo() -> None:
    """The silent fallback itself: an empty dict used to read as ``"P0D"``."""
    with pytest.raises(ValueError, match="no label horizon"):
        _label_horizon_to_iso({})


def test_the_end_user_path_raises_and_names_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """How this actually arrives: a setup.yaml declares a buffer nothing reads.

    Before the fix this returned a config with ``embargo_td="P0D"`` and no
    warning, which is the state nasdaq100_microstructure was in.
    """
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(_setup(tmp_path, "5 fortnights")))
    with pytest.raises(ValueError) as raised:
        get_cv_config("probe")
    message = str(raised.value)
    assert "5 fortnights" in message
    assert "setup.yaml" in message, "the user has to be told which file to edit"


def test_the_end_user_path_reads_a_valid_buffer_the_old_chain_missed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(_setup(tmp_path, "PT30M")))
    assert get_cv_config("probe").embargo_td == "PT30M"
    assert load_protocol("probe")["leakage_guards"] == {"label_horizon_minutes": 30}
