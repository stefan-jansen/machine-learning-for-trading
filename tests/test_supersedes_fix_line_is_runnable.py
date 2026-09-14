"""The undeclared-generation fix line must print a command that works.

The line told the reader to declare the generation "in the freezing notebook's
SUPERSEDES_* mapping, in the worktree you are about to launch". That is a source edit,
every freezing notebook that has already run is stamped, so it returns STALE and owes a
full re-render - 44.15 h across the five ``us_equities_panel`` notebooks that freeze the
twelve live generations (ml4t/agent-workspace#1175).

The launch-time route it now names has one trap, which is why this is a test rather than
a reading. ``papermill -p`` is scalar-only: ``_resolve_type`` returns True/False/None/int/
float and otherwise the bare string, so ``-p SUPERSEDES_SETS '{"x": "live"}'`` injects a
str and ``SUPERSEDES_SETS.get(...)`` raises AttributeError at the freeze, after the fit -
strictly worse than the re-render it was meant to avoid. ``nb-run.sh``'s ``KEY:json=``
form passes a one-key YAML document (``papermill -y``) and delivers a real dict. This
pins that the emitted line names that form and that parsing it the way the runner does
yields a mapping.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "check_supersedes_literals", REPO / "scripts" / "check_supersedes_literals.py"
)
assert _spec and _spec.loader
_checker = importlib.util.module_from_spec(_spec)
sys.modules["check_supersedes_literals"] = _checker
_spec.loader.exec_module(_checker)


def _fix_line() -> str:
    """The checker's own message for one undeclared generation, not a copy of it."""
    finding = _checker.Finding(
        "us_equities_panel",
        "13a_pca.py",
        "55f27fb9218f",
        "undeclared",
        "detail",
        "SUPERSEDES_SETS",
        _checker.SUPERSEDES_LIVE,
        "us-equities-fwd-ret-1d-pca-v1",
    )
    return _checker.undeclared_fix(finding)


def test_the_fix_line_names_the_structured_parameter_form() -> None:
    line = _fix_line()
    assert "SUPERSEDES_SETS:json=" in line
    # `-p` is the form that silently delivers a string. It must not be what is suggested.
    assert "-p SUPERSEDES_SETS" not in line


def test_the_suggested_value_parses_the_way_nb_run_parses_it() -> None:
    """`nb-run.sh` splits on `:json=` and `json.loads` the right-hand side."""
    line = _fix_line()
    match = re.search(r"(\w+):json='(.*)' to nb-run\.sh", line)
    assert match, line
    key, raw = match.group(1), match.group(2)
    value = json.loads(raw)
    assert isinstance(value, dict), f"{raw!r} must be a mapping, not {type(value).__name__}"
    assert value == {"us-equities-fwd-ret-1d-pca-v1": "live"}
    # What the runner hands papermill: a one-key YAML document, which is why it survives
    # as a dict where `-p` would not.
    assert json.loads(json.dumps({key: value})) == {"SUPERSEDES_SETS": value}


def test_papermill_p_would_not_survive_as_a_mapping() -> None:
    """The reason the line cannot say `-p`, asserted against papermill's own coercion."""
    from papermill.cli import _resolve_type

    delivered = _resolve_type('{"us-equities-fwd-ret-1d-pca-v1": "live"}')
    assert isinstance(delivered, str)
    assert not hasattr(delivered, "get")


def _findings(*specs) -> list:
    """Findings for one notebook, as `check_all` would return them."""
    return [
        _checker.Finding(
            "us_equities_panel",
            "07_gbm.py",
            declared,
            status,
            "detail",
            parameter,
            "live",
            label,
        )
        for parameter, label, declared, status in specs
    ]


class TestOneLaunchCarriesEveryEntry:
    """A per-finding fix line is wrong for any notebook owing more than one entry.

    Papermill injects a parameter by REPLACING the notebook's binding, so a launch
    carrying a one-key `SUPERSEDES_SETS` mapping drops every other key the notebook
    declared and each of those becomes undeclared - refused at the same freeze the fix
    was meant to clear, after the same fit. Measured on `us_equities_panel` 2026-09-14:
    `07_gbm` owes six entries and five other notebooks owe two each, so following the
    one-key lines literally leaves every one of them carrying a single key.
    """

    def test_the_entries_land_in_one_mapping(self) -> None:
        parameters = _checker.launch_parameters(
            _findings(
                ("SUPERSEDES_SETS", "gbm-1d", "464646b3bd65", "behind"),
                ("SUPERSEDES_SETS", "gbm-1d-diagnostics", "30766ca63471", "behind"),
                ("SUPERSEDES_SETS", "gbm-5d", "", "undeclared"),
            )
        )
        assert len(parameters) == 1
        mapping = json.loads(parameters[0].split(":json=", 1)[1].strip("'"))
        assert mapping == {
            "gbm-1d": "live",
            "gbm-1d-diagnostics": "live",
            "gbm-5d": "live",
        }

    def test_a_live_entry_is_carried_too(self) -> None:
        """Dropping it from the mapping is what makes it undeclared on the next run."""
        parameters = _checker.launch_parameters(
            _findings(
                ("SUPERSEDES_SETS", "gbm-1d", "464646b3bd65", "behind"),
                ("SUPERSEDES_SETS", "gbm-21d", "8957551a2b6b", "live"),
            )
        )
        mapping = json.loads(parameters[0].split(":json=", 1)[1].strip("'"))
        assert set(mapping) == {"gbm-1d", "gbm-21d"}

    def test_a_scalar_declaration_takes_the_scalar_form(self) -> None:
        """`SUPERSEDES_POPULATION` is a bare string; the mapping form injects a dict."""
        parameters = _checker.launch_parameters(
            _findings(("SUPERSEDES_POPULATION", None, "1ce92c9f8dc0", "behind"))
        )
        assert parameters == ["SUPERSEDES_POPULATION=live"]

    def test_the_two_forms_compose_in_one_launch(self) -> None:
        """`07_gbm` declares both, which is where getting the form wrong is easy."""
        parameters = _checker.launch_parameters(
            _findings(
                ("SUPERSEDES_POPULATION", None, "1ce92c9f8dc0", "behind"),
                ("SUPERSEDES_SETS", "gbm-1d", "464646b3bd65", "behind"),
            )
        )
        line = _checker.launch_line("us_equities_panel", "07_gbm.py", parameters)
        assert line.strip().startswith("nb-run.sh us_equities_panel 07_gbm ")
        assert "SUPERSEDES_POPULATION=live" in line
        assert "SUPERSEDES_SETS:json=" in line
        assert "SUPERSEDES_POPULATION:json=" not in line

    def test_the_causal_declaration_is_left_out(self) -> None:
        """`causal_supersedes` never offers the sentinel, so naming it would misdirect."""
        parameters = _checker.launch_parameters(
            _findings(
                (_checker._CAUSAL_NAME, "fwd_ret_1m", "96b84e61bab8", "behind"),
                ("SUPERSEDES_SETS", "gbm-1d", "464646b3bd65", "behind"),
            )
        )
        assert parameters == ['SUPERSEDES_SETS:json=\'{"gbm-1d": "live"}\'']

    def test_an_unresolved_finding_is_left_out(self) -> None:
        """The checker does not know what that notebook declares, so it cannot say."""
        assert (
            _checker.launch_parameters(
                _findings(("SUPERSEDES_SETS", "gbm-1d", "464646b3bd65", "unresolved"))
            )
            == []
        )


class TestEveryRefusalPathSaysWhatToType:
    """A path that stops a chain and prints no command is the failure this change exists
    to fix, one level up: the reader is told what is wrong and not what to run.

    The findings are supplied rather than read from the live registry. Reading it would
    make these pass only while `us_equities_panel` happens to owe something, so fixing the
    declarations would break the test or, worse, leave it passing vacuously.
    """

    FINDINGS = [
        _checker.Finding(
            "us_equities_panel",
            "07_gbm.py",
            "464646b3bd65",
            "behind",
            "one generation behind",
            "SUPERSEDES_SETS",
            "live",
            "gbm-1d",
        ),
        _checker.Finding(
            "us_equities_panel",
            "07_gbm.py",
            "",
            "undeclared",
            "no declaration names it",
            "SUPERSEDES_SETS",
            "live",
            "gbm-5d",
        ),
    ]

    def _stderr(self, monkeypatch, *argv: str, findings=None) -> str:
        import contextlib
        import io

        monkeypatch.setattr(
            _checker, "check_all", lambda **_: list(self.FINDINGS if findings is None else findings)
        )
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer), contextlib.redirect_stdout(io.StringIO()):
            _checker.main(["--case-study", "us_equities_panel", *argv])
        return buffer.getvalue()

    def test_the_plain_refusal_prints_the_commands_once(self, monkeypatch) -> None:
        output = self._stderr(monkeypatch)
        assert output.count("The launch each of these needs") == 1
        assert "nb-run.sh us_equities_panel 07_gbm " in output

    def test_require_declarations_prints_them_before_it_returns(self, monkeypatch) -> None:
        """It returns above the refused-literal block, so it used to print none at all."""
        output = self._stderr(monkeypatch, "--require-declarations")
        assert "nb-run.sh us_equities_panel 07_gbm " in output
        assert "Refusing because --require-declarations was passed." in output

    def test_allow_stale_still_prints_them(self, monkeypatch) -> None:
        output = self._stderr(monkeypatch, "--allow-stale-supersedes")
        assert output.count("The launch each of these needs") == 1

    def test_the_undeclared_only_warning_prints_them(self, monkeypatch) -> None:
        """It refuses nothing, which is exactly when acting on it is cheap."""
        undeclared_only = [f for f in self.FINDINGS if f.status == "undeclared"]
        output = self._stderr(monkeypatch, findings=undeclared_only)
        assert "nb-run.sh us_equities_panel 07_gbm " in output

    def test_the_command_carries_both_entries(self, monkeypatch) -> None:
        """One behind and one undeclared entry, one mapping, both keys."""
        output = self._stderr(monkeypatch)
        # The per-finding fix line also names nb-run.sh, so match the command itself.
        line = next(x for x in output.splitlines() if x.strip().startswith("nb-run.sh "))
        mapping = json.loads(line.split(":json=", 1)[1].strip("'"))
        assert mapping == {"gbm-1d": "live", "gbm-5d": "live"}

    def test_no_command_is_offered_for_an_unattributed_generation(self) -> None:
        """`_undeclared_heads` writes "-" when it cannot say which notebook freezes one."""
        import io

        finding = _checker.Finding(
            "us_equities_panel",
            _checker._UNATTRIBUTED,
            "",
            "undeclared",
            "detail",
            "SUPERSEDES_SETS",
            "live",
            "orphan-set-v1",
        )
        buffer = io.StringIO()
        _checker._print_launch_lines([finding], buffer)
        assert buffer.getvalue() == ""
