"""A causal row a reader cannot resolve is worse than no causal row.

`current_causal_identities` skips any row whose spec does not carry the current identity
version, and that set is what `CausalResult.one` resolves and what the two-identities check
is computed from. A row without it is registered, sits in the table, and answers nothing.

Two paths write causal rows. `resolve_causal_request` produces an `ml4t.resolved-spec/v1`
payload carrying the version; the thin wrapper in `case_studies/utils/causal.py` does not,
and cannot be made to - `project_training_identity` refuses version 3 without that payload.
So a notebook that calls `run_dml_analysis` directly registers something no reader sees.

Measured 2026-08-25: crypto, cme, fx and sp500_options route through the resolver and every
row is visible; etfs and us_firm_characteristics call the wrapper and not one of their four
rows is. This test names the notebooks that still have to be converted, so the list can only
shrink and a new notebook cannot quietly join it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Still building their own spec instead of going through `resolve_causal_request`.
# Delete an entry when its notebook is converted; the test below fails if one is
# converted and left here.
UNCONVERTED = {
    "case_studies/etfs/12_causal_dml.py",
    "case_studies/us_firm_characteristics/09_causal_dml.py",
    "case_studies/nasdaq100_microstructure/12_causal_dml.py",
    "case_studies/sp500_options/10_causal_dml.py",
    "case_studies/sp500_equity_option_analytics/12_causal_dml.py",
}

DIRECT_CALL = re.compile(r"^\s*results = run_dml_analysis\(", re.MULTILINE)


def _direct_callers() -> set[str]:
    return {
        p.relative_to(REPO).as_posix()
        for p in sorted(REPO.glob("case_studies/*/[0-9]*causal_dml.py"))
        if DIRECT_CALL.search(p.read_text(encoding="utf-8"))
    }


def test_no_new_notebook_builds_its_own_causal_spec() -> None:
    new = _direct_callers() - UNCONVERTED
    assert not new, (
        f"{sorted(new)} call run_dml_analysis directly, so the rows they register carry no "
        "identity version and no reader resolves them. Route the notebook through "
        "resolve_causal_request / run_resolved_causal_request, as crypto_perps_funding, "
        "cme_futures and fx_pairs do."
    )


def test_converted_notebooks_come_off_the_list() -> None:
    stale = UNCONVERTED - _direct_callers()
    assert not stale, f"these no longer build their own spec and should be removed: {sorted(stale)}"
