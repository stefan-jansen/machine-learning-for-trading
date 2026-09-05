"""A family whose runner refuses a class target must not be declared on a classification label.

Two families in this corpus fit regression estimators only, and both refuse rather than adapt:

* ``causal_dml`` builds both DML nuisance models as ``HistGradientBoostingRegressor``
  (``case_studies/utils/causal.py``) and has no classification branch at all.
* ``deep_learning`` refuses explicitly - ``case_studies/utils/deep_learning.py:374``, "sequence
  runner currently supports regression labels only" - inside ``resolve_model_request``, so a
  declared configuration on a class label stops at planning.

Neither refusal costs a fit, and that is why these declarations survive: nothing runs them, so
nothing complains. What they cost is the declared menu. The next reader compares what the
config declares against what the registry holds, sees members missing, and extends a notebook
to produce them - which for ``causal_dml`` means fitting a regressor to a class target and
registering the result as a canonical causal estimate, and for ``deep_learning`` means a run
that either narrows to the regression labels silently or stops at planning after the earlier
families have already been fitted. That is ml4t/agent-workspace#396, generalized once the same
shape turned up in the sequence family.

``linear`` and ``gbm`` are not the target: both have real classification branches, and the
classification labels declare them on purpose.

A classification label is identified by ``labels.classification_eval_label`` in
``config/setup.yaml``, which names the continuous return each class target is derived from.
That is the corpus's own declaration of which labels are classification, so the check does not
depend on how a label happens to be named.
"""

from __future__ import annotations

import pytest
import yaml

from utils.paths import REPO_ROOT

# Anchored to REPO_ROOT rather than the working directory, so running pytest from elsewhere
# cannot silently collect nothing and pass.
SETUPS = sorted((REPO_ROOT / "case_studies").glob("*/config/setup.yaml"))

# The families whose runners refuse a class target, and where each refusal lives, so a reader
# hitting a failure can confirm it rather than take this file's word for it.
REGRESSION_ONLY_FAMILIES = {
    "causal_dml": "case_studies/utils/causal.py builds both DML nuisances as regressors",
    "deep_learning": "case_studies/utils/deep_learning.py:374 refuses a non-regression label",
}

# `(case_study, label, family)` triples that still declare one, each owned by another lane.
# Every entry is asserted to still be earned by `test_every_exemption_is_still_earned`, so it
# has to be deleted in the same change that fixes the config rather than outliving it unread -
# which is how the previous version of this list went stale within a day.
KNOWN_UNFIXED = {
    ("crypto_perps_funding", "fwd_dir_8h", "deep_learning"),
    ("crypto_perps_funding", "fwd_dir_8h_3c", "deep_learning"),
}


def _classification_labels(setup_path) -> list[str]:
    setup = yaml.safe_load(setup_path.read_text()) or {}
    declared = (setup.get("labels") or {}).get("classification_eval_label") or {}
    return sorted(declared)


def _declares(case_study: str, label: str, family: str) -> bool:
    menu = REPO_ROOT / "case_studies" / case_study / "config" / "training" / f"{label}.yaml"
    if not menu.exists():
        return False
    return bool((yaml.safe_load(menu.read_text()) or {}).get(family))


CASES = [
    (path.parts[-3], label, family)
    for path in SETUPS
    for label in _classification_labels(path)
    for family in sorted(REGRESSION_ONLY_FAMILIES)
]


def test_the_corpus_declares_classification_labels_at_all():
    """Without this the parametrized check below can pass on an empty list."""
    assert SETUPS, "no case studies discovered"
    assert CASES, "no classification labels discovered; the setup.yaml key must have moved"


@pytest.mark.parametrize("case_study,label,family", CASES, ids=lambda v: str(v))
def test_no_regression_only_family_on_a_classification_label(
    case_study: str, label: str, family: str
):
    if (case_study, label, family) in KNOWN_UNFIXED:
        pytest.skip(f"{case_study}/{label} declares {family}: known unfixed instance of #396")
    assert not _declares(case_study, label, family), (
        f"{case_study}/config/training/{label}.yaml declares `{family}` on a classification "
        f"label, and {REGRESSION_ONLY_FAMILIES[family]}"
    )


@pytest.mark.parametrize("case_study,label,family", sorted(KNOWN_UNFIXED), ids=lambda v: str(v))
def test_every_exemption_is_still_earned(case_study: str, label: str, family: str):
    """An exemption for a config that no longer declares it hides the next regression."""
    assert _declares(case_study, label, family), (
        f"{case_study}/{label} no longer declares `{family}`; remove it from KNOWN_UNFIXED so "
        "the check guards it"
    )
