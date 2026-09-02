"""Every label a conformal case study can be run on needs a reviewed embargo.

`holdout_conformal_embargo_steps` refuses an unknown `case_study/label` rather
than guessing, and the notebooks that reach it take `LABEL` as a parameter. So a
case study that declares a `conformal_weighted` allocator and a label variant
with no entry raises `KeyError` the first time anyone runs that variant - not at
import, not in CI, but partway through a production run.

The pairing is what this checks: entries are reviewed values, so the test asserts
one exists, never what it should be.
"""

from __future__ import annotations

import pytest
import yaml

from case_studies.utils.conformal import HOLDOUT_CONFORMAL_EMBARGO_STEPS
from utils.paths import REPO_ROOT

SETUPS = sorted((REPO_ROOT / "case_studies").glob("*/config/setup.yaml"))


def _allocator_methods(node):
    """Collect allocator methods wherever they are declared.

    The key sits at `backtest.sweep.allocators` today. Walking for it rather
    than naming the path keeps a case study that nests it elsewhere from
    silently reporting no allocators, which would skip its check instead of
    failing it.
    """
    found = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "allocators" and isinstance(value, list):
                found |= {a.get("method") for a in value if isinstance(a, dict)}
            found |= _allocator_methods(value)
    elif isinstance(node, list):
        for value in node:
            found |= _allocator_methods(value)
    return found


def _declared(setup_path):
    setup = yaml.safe_load(setup_path.read_text())
    labels = setup.get("labels") or {}
    declared = [labels.get("primary"), *(labels.get("variants") or [])]
    return [label for label in declared if label], _allocator_methods(setup)


def test_case_studies_are_discovered():
    """A glob that collects nothing makes every parametrized case below vacuous."""
    assert SETUPS, "no case-study setup.yaml files discovered"


def test_the_conformal_allocator_is_actually_detected():
    """Every case skipping is indistinguishable from every case passing.

    The first version of this module read `backtest.rebalance.allocators`, a path
    that does not exist, so all nine case studies skipped and the suite reported
    green. This asserts the parse still finds the declaration it is keyed on.
    """
    conformal = [p.parts[-3] for p in SETUPS if "conformal_weighted" in _declared(p)[1]]
    assert conformal, "no case study parsed as declaring a conformal_weighted allocator"


@pytest.mark.parametrize("setup_path", SETUPS, ids=lambda p: p.parts[-3])
def test_every_conformal_label_has_a_reviewed_embargo(setup_path):
    case_study = setup_path.parts[-3]
    labels, methods = _declared(setup_path)
    if "conformal_weighted" not in methods:
        pytest.skip(f"{case_study} declares no conformal allocator")
    missing = [
        label for label in labels if f"{case_study}/{label}" not in HOLDOUT_CONFORMAL_EMBARGO_STEPS
    ]
    assert not missing, (
        f"{case_study} declares a conformal_weighted allocator and the label(s) {missing}, "
        f"which have no HOLDOUT_CONFORMAL_EMBARGO_STEPS entry; add a reviewed data-step value"
    )
