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
