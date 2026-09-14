"""A library diagnostic must survive the blanket filter the notebook installs.

175 files still call `warnings.filterwarnings("ignore")` at import, so a guard written on
`warnings.warn` alone reports nothing to the person reading the executed notebook whether or
not it fired. `warn_the_reader` emits on both channels. These tests run the condition the
NOTEBOOK runs under - the blanket filter installed - which is the thing `pytest.warns`
cannot see, because it installs a filter of its own.
"""

from __future__ import annotations

import warnings

import pytest

from case_studies.utils.warning_policy import reset_reader_warnings, warn_the_reader


@pytest.fixture(autouse=True)
def _forget_previous_reports():
    reset_reader_warnings()
    yield
    reset_reader_warnings()


class TestItSurvivesTheBlanketFilter:
    def test_the_printed_line_lands_even_under_a_bare_ignore(self, capsys):
        """The whole point. `warnings.warn` here reaches nobody; the print does."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            warn_the_reader("the panel cannot price 3 symbols", source="check_universe")
        out = capsys.readouterr().out
        assert "  WARN check_universe: the panel cannot price 3 symbols" in out

    def test_the_warning_still_fires_for_a_library_caller(self):
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            warn_the_reader("something", source="fn")
        assert len(raised) == 1
        assert "something" in str(raised[0].message)

    def test_the_category_reaches_the_caller(self):
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            warn_the_reader("degraded", source="fn", category=RuntimeWarning)
        assert raised[0].category is RuntimeWarning

    def test_the_default_category_is_userwarning(self):
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            warn_the_reader("plain", source="fn")
        assert raised[0].category is UserWarning


class TestDeduplication:
    def test_a_repeat_prints_once(self, capsys):
        assert warn_the_reader("same condition", source="fn") is True
        assert warn_the_reader("same condition", source="fn") is False
        assert capsys.readouterr().out.count("WARN fn:") == 1

    def test_the_key_decides_what_counts_as_a_repeat(self, capsys):
        """A sweep calls one function per scheme with a message carrying moving counts.
        A coarse key is what turns twelve lines into one."""
        for scheme in range(12):
            warn_the_reader(
                f"scheme {scheme}: 3 of 40 symbols unpriced",
                source="fn",
                key=("unpriced", "etfs", "fwd_ret_1d"),
            )
        assert capsys.readouterr().out.count("WARN fn:") == 1

    def test_without_a_key_a_moving_message_prints_every_time(self, capsys):
        for scheme in range(3):
            warn_the_reader(f"scheme {scheme}", source="fn")
        assert capsys.readouterr().out.count("WARN fn:") == 3

    def test_a_different_condition_still_prints(self, capsys):
        warn_the_reader("a", source="fn", key="first")
        warn_the_reader("b", source="fn", key="second")
        assert capsys.readouterr().out.count("WARN fn:") == 2

    def test_the_warning_is_not_deduplicated_with_the_print(self):
        """A test may be counting warnings; suppressing the second would change its answer."""
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            warn_the_reader("same", source="fn")
            warn_the_reader("same", source="fn")
        assert len(raised) == 2


class TestShape:
    def test_the_line_carries_the_source_and_a_two_space_indent(self, capsys):
        warn_the_reader("message text", source="some_function")
        assert capsys.readouterr().out == "  WARN some_function: message text\n"
