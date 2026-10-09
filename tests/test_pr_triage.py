"""The triage comment must name a real problem and stay silent otherwise.

A bot that asks for a split on a one-defect pull request teaches contributors to
ignore it, so each case below pins both directions: the finding fires on the shape
it exists for, and does not fire on the compliant shape.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "pr_triage.py"
_spec = importlib.util.spec_from_file_location("pr_triage", SCRIPT)
assert _spec and _spec.loader
pr_triage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr_triage)

COMPLIANT_BODY = "Fixes the 16min buffer. Closes #1145. Written with Claude's help."
COMPLIANT_FILES = [
    "09_model_based_features/13_regime_as_feature.py",
    "09_model_based_features/13_regime_as_feature.ipynb",
    "tests/test_label_horizon_guards.py",
]


def ids(files: list[str], body: str) -> set[str]:
    return {f["id"] for f in pr_triage.find(files, body)}


def test_a_compliant_pull_request_gets_no_findings() -> None:
    assert pr_triage.find(COMPLIANT_FILES, COMPLIANT_BODY) == []
    assert "checks pass" in pr_triage.render(1, [])


def test_two_chapters_in_one_pull_request_asks_for_a_split() -> None:
    bundled = COMPLIANT_FILES + ["08_financial_features/01_price_volume_features.py"]
    assert "one-change-per-pr" in ids(bundled, COMPLIANT_BODY)


def test_a_fix_plus_its_test_is_not_a_bundle() -> None:
    """`tests/` and `utils/` are shared, so a fix and its test are one area."""
    with_shared = COMPLIANT_FILES + ["utils/modeling.py", ".github/workflows/test.yml"]
    assert "one-change-per-pr" not in ids(with_shared, COMPLIANT_BODY)


def test_a_body_that_only_mentions_an_issue_does_not_close_it() -> None:
    assert "no-owning-issue" in ids(COMPLIANT_FILES, "See #1145 for context.")
    assert "no-owning-issue" not in ids(COMPLIANT_FILES, "Closes #1145")
    assert "no-owning-issue" not in ids(COMPLIANT_FILES, "fixed #1145")
    assert "no-owning-issue" not in ids(
        COMPLIANT_FILES,
        "Resolves https://github.com/stefan-jansen/machine-learning-for-trading/issues/1145",
    )


def test_an_ipynb_without_its_py_is_flagged_as_a_hand_edit() -> None:
    hand_edited = ["09_model_based_features/13_regime_as_feature.ipynb", "tests/test_x.py"]
    found = [
        f for f in pr_triage.find(hand_edited, COMPLIANT_BODY) if f["id"] == "ipynb-without-py"
    ]
    assert found and found[0]["level"] == "block"
    assert "13_regime_as_feature.ipynb" in found[0]["text"]
    # The paired case must stay silent, or every notebook fix trips it.
    assert "ipynb-without-py" not in ids(COMPLIANT_FILES, COMPLIANT_BODY)


def test_a_code_change_with_no_test_is_asked_for_one() -> None:
    no_test = ["09_model_based_features/13_regime_as_feature.py"]
    assert "no-test" in ids(no_test, COMPLIANT_BODY)
    assert "no-test" not in ids(COMPLIANT_FILES, COMPLIANT_BODY)


def test_silence_about_ai_tools_is_a_note_not_a_blocker() -> None:
    quiet = [
        f for f in pr_triage.find(COMPLIANT_FILES, "Closes #1145") if f["id"] == "no-ai-disclosure"
    ]
    assert quiet and quiet[0]["level"] == "note"
    assert "no-ai-disclosure" not in ids(COMPLIANT_FILES, "Closes #1145. No AI was used.")


def test_the_comment_carries_the_marker_the_updater_matches_on() -> None:
    rendered = pr_triage.render(7, pr_triage.find(["a_chapter/x.py"], ""))
    assert rendered.startswith("<!-- pr-triage -->")
    assert "Needs a change" in rendered or "Please fix" in rendered
