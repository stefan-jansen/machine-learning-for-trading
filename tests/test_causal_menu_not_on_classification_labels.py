"""A `causal_dml` menu declared on a classification label is an instruction to run nonsense.

`case_studies/utils/causal.py` builds both DML nuisance models as
`HistGradientBoostingRegressor` and has no classification branch, so estimating the declared
menu on a class target fits a regressor to it and registers the result as a canonical causal
estimate. Nothing runs these declarations today - every causal notebook estimates on
`labels.primary`, which is continuous everywhere - so the cost is not a wrong number on disk.
It is that the declaration and the production count disagree, and only the declaration is
discoverable: the next reader sees a causal result that the config says should exist, and
extends a notebook to produce it. That is ml4t/agent-workspace#396, and comments alone did not
stop it coming back.

A classification label is identified by `labels.classification_eval_label` in
`config/setup.yaml`, which names the continuous return each class target is derived from. That
is the corpus's own declaration of which labels are classification, so the check does not
depend on how a label happens to be named.
"""

from __future__ import annotations

import pytest
import yaml

from utils.paths import REPO_ROOT

# Anchored to REPO_ROOT rather than the working directory, so running pytest from elsewhere
# cannot silently collect nothing and pass.
SETUPS = sorted((REPO_ROOT / "case_studies").glob("*/config/setup.yaml"))

# `(case_study, label)` pairs that still declare the key, each owned by another lane. An
# exemption here is asserted to still be needed by `test_every_exemption_is_still_earned`, so
# the entry has to be deleted in the same change that fixes the config rather than outliving
# it as a permanently unread allowance.
KNOWN_UNFIXED = {("us_firm_characteristics", "fwd_class_1m")}


def _classification_labels(setup_path) -> list[str]:
    setup = yaml.safe_load(setup_path.read_text()) or {}
    declared = (setup.get("labels") or {}).get("classification_eval_label") or {}
    return sorted(declared)


def _declares_causal_dml(case_study: str, label: str) -> bool:
    menu = REPO_ROOT / "case_studies" / case_study / "config" / "training" / f"{label}.yaml"
    if not menu.exists():
        return False
    return "causal_dml" in (yaml.safe_load(menu.read_text()) or {})


CASES = [(path.parts[-3], label) for path in SETUPS for label in _classification_labels(path)]


def test_the_corpus_declares_classification_labels_at_all():
    """Without this the parametrized check below can pass on an empty list."""
    assert SETUPS, "no case studies discovered"
    assert CASES, "no classification labels discovered; the setup.yaml key must have moved"


@pytest.mark.parametrize("case_study,label", CASES, ids=lambda v: str(v))
def test_no_causal_menu_on_a_classification_label(case_study: str, label: str):
    if (case_study, label) in KNOWN_UNFIXED:
        pytest.skip(f"{case_study}/{label} is a known unfixed instance of #396")
    assert not _declares_causal_dml(case_study, label), (
        f"{case_study}/config/training/{label}.yaml declares `causal_dml` on a classification "
        "label; the DML nuisances are regressors and there is no classification branch"
    )


@pytest.mark.parametrize("case_study,label", sorted(KNOWN_UNFIXED), ids=lambda v: str(v))
def test_every_exemption_is_still_earned(case_study: str, label: str):
    """An exemption for a config that no longer declares the key hides the next regression."""
    assert _declares_causal_dml(case_study, label), (
        f"{case_study}/{label} no longer declares `causal_dml`; remove it from KNOWN_UNFIXED "
        "so the check guards it"
    )
