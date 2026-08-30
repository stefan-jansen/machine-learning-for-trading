"""A causal row a reader cannot resolve is worse than no causal row.

`current_causal_identities` skips any row whose spec does not carry the current identity
version, and that set is what `CausalResult.one` resolves and what the two-identities check
is computed from. A row without it is registered, sits in the table, and answers nothing.

Two paths write causal rows. `resolve_causal_request` produces an `ml4t.resolved-spec/v1`
payload carrying the version; the thin wrapper `register_causal_run` in
`case_studies/utils/causal.py` does not, and cannot be made to - `project_training_identity`
refuses version 3 without that payload. So a notebook that reaches `register_causal_run`
registers something no reader sees, whatever it names the result it passes in.

Measured against the fleet registries on 2026-08-25: crypto 2/2 rows visible, cme 6/6, fx
3/3, all written by the resolver. us_firm_characteristics 0/3 written by the
wrapper; etfs' one row was written by the wrapper and its notebook has since been converted,
so its next run registers a resolvable one. sp500_options holds one visible row, but it was written by the resolver before
that notebook moved to the wrapper, so its next run registers an invisible one;
nasdaq100_microstructure and sp500_equity_option_analytics have registered nothing yet and
would register invisible rows on their first run.

The guard keys on `register_causal_run` rather than on any particular call spelling. An
earlier version matched the literal `results = run_dml_analysis(`, which a notebook slips
past by assigning to another name or by fitting inline and only registering - exactly the
row this test exists to prevent.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Still building their own spec instead of going through `resolve_causal_request`.
# Delete an entry when its notebook is converted; the test below fails if one is
# converted and left here.
UNCONVERTED = {
    "case_studies/nasdaq100_microstructure/12_causal_dml.py",
    "case_studies/sp500_options/10_causal_dml.py",
    "case_studies/sp500_equity_option_analytics/12_causal_dml.py",
}

WRAPPER = "register_causal_run"


def _reaches_the_wrapper(source: str) -> bool:
    """True when this module imports or calls the unversioned registration wrapper.

    Both spellings count. An import from `case_studies.utils.causal` is the one the five
    notebooks use; an attribute call (`causal.register_causal_run(...)`) reaches the same
    function without an import naming it. An `import ... as` alias is still caught, because
    `alias.name` is what was imported, not what it was bound to.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").endswith("causal") and any(
                alias.name == WRAPPER for alias in node.names
            ):
                return True
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == WRAPPER:
                return True
    return False


def _direct_callers(root: Path) -> set[str]:
    """Every numbered case-study notebook under `root` that reaches the wrapper."""
    return {
        p.relative_to(root).as_posix()
        for p in sorted(root.glob("case_studies/*/[0-9]*.py"))
        if _reaches_the_wrapper(p.read_text(encoding="utf-8"))
    }


def _unlisted(root: Path, allowlist: set[str]) -> set[str]:
    return _direct_callers(root) - allowlist


def _stale(root: Path, allowlist: set[str]) -> set[str]:
    return allowlist - _direct_callers(root)


def test_no_new_notebook_registers_an_unresolvable_causal_row() -> None:
    new = _unlisted(REPO, UNCONVERTED)
    assert not new, (
        f"{sorted(new)} reach register_causal_run, so the rows they register carry no "
        "identity version and no reader resolves them. Route the notebook through "
        "resolve_causal_request / run_resolved_causal_request, as crypto_perps_funding, "
        "cme_futures and fx_pairs do."
    )


def test_a_converted_notebook_comes_off_the_list() -> None:
    stale = _stale(REPO, UNCONVERTED)
    assert not stale, f"these no longer reach the wrapper and should be removed: {sorted(stale)}"


def _write(root: Path, rel: str, source: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


CONVERTED = """
from case_studies.utils.causal import resolve_causal_request, run_resolved_causal_request

result = run_resolved_causal_request(study, resolve_causal_request(study, request))
"""

# The spelling the previous regex missed: the result is not named `results`, and the
# registration is a bare call rather than an assignment.
OFFENDER = """
from case_studies.utils.causal import register_causal_run, run_dml_analysis

outcome = run_dml_analysis(frame, treatment="t", outcome="y")
register_causal_run("some_case_study", outcome, notebook="06_causal")
"""


@pytest.fixture
def fleet(tmp_path: Path) -> Path:
    """A miniature repo: one converted notebook, one wrapper caller."""
    _write(tmp_path, "case_studies/converted_study/11_causal_dml.py", CONVERTED)
    _write(tmp_path, "case_studies/legacy_study/12_causal_dml.py", OFFENDER)
    return tmp_path


def test_the_guard_passes_when_the_allowlist_describes_the_tree(fleet: Path) -> None:
    allowlist = {"case_studies/legacy_study/12_causal_dml.py"}
    assert not _unlisted(fleet, allowlist)
    assert not _stale(fleet, allowlist)


def test_a_new_offender_fails_the_guard(fleet: Path) -> None:
    """The sixth offender: a notebook nobody allowlisted, in the spelling the regex missed."""
    _write(fleet, "case_studies/new_study/09_causal_dml.py", OFFENDER)
    allowlist = {"case_studies/legacy_study/12_causal_dml.py"}
    assert _unlisted(fleet, allowlist) == {"case_studies/new_study/09_causal_dml.py"}


def test_an_offender_reaching_the_wrapper_by_attribute_fails_the_guard(fleet: Path) -> None:
    _write(
        fleet,
        "case_studies/new_study/09_causal_dml.py",
        "from case_studies.utils import causal\n\ncausal.register_causal_run('x', {})\n",
    )
    assert _unlisted(fleet, set()) >= {"case_studies/new_study/09_causal_dml.py"}


def test_a_stale_allowlist_entry_fails_the_guard(fleet: Path) -> None:
    """Converting a notebook and leaving it on the list has to fail, or the list never shrinks."""
    _write(fleet, "case_studies/legacy_study/12_causal_dml.py", CONVERTED)
    allowlist = {"case_studies/legacy_study/12_causal_dml.py"}
    assert _stale(fleet, allowlist) == allowlist


def test_a_resolver_routed_notebook_is_not_flagged(fleet: Path) -> None:
    assert "case_studies/converted_study/11_causal_dml.py" not in _direct_callers(fleet)
