"""The triage report must name a real problem, stay silent otherwise, and publish nothing.

A report that asks for a split on a one-defect pull request teaches a maintainer to
ignore it, so each case below pins both directions: the finding fires on the shape it
exists for, and does not fire on the compliant shape.

The last group checks the authority rather than the text. Triage is advisory: it saves
findings for a maintainer and holds no token that could post a comment, a review, a
label or a close. An earlier version posted and edited a comment by itself from a
`workflow_run` job, so these assert the capability is gone and not merely unused.
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


def test_the_report_states_every_finding_and_drafts_a_reply_for_a_person_to_send() -> None:
    findings = pr_triage.find(["a_chapter/x.py"], "")
    rendered = pr_triage.render(7, findings)
    assert "## Triage: PR #7" in rendered
    assert "Needs a change" in rendered or "Please fix" in rendered
    # Each finding has to survive into the draft, or the analysis is lost between the
    # report a maintainer reads and the reply they send.
    draft = rendered.split("### Suggested reply", 1)[1]
    for f in findings:
        assert f["text"] in draft
    assert "Paste this if you want to send it" in rendered


def test_the_report_does_not_claim_to_post_itself() -> None:
    """It is a file in a build artifact. Wording that promises otherwise is wrong."""
    rendered = pr_triage.render(7, pr_triage.find(["a_chapter/x.py"], ""))
    assert "<!-- pr-triage -->" not in rendered, "the comment-dedup marker has no job now"
    for stale in ("this comment updates itself", "Push a change and this comment"):
        assert stale not in rendered


# ---------------------------------------------------- the authority, not the wording

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
TRIAGE = WORKFLOWS / "pr-triage.yml"


def _workflow(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def test_triage_runs_with_no_write_permission_at_all() -> None:
    """The grant is the limit. A prompt or a script cannot exceed it."""
    wf = _workflow(TRIAGE)
    assert wf["permissions"] == {"contents": "read"}
    assert "permissions" not in wf["jobs"]["report"], (
        "a job-level grant would override the read-only workflow grant"
    )


def test_no_workflow_publishes_the_triage_findings() -> None:
    """The `workflow_run` publisher is deleted, not disabled.

    A disabled publisher is one edit from posting again, and the commit that removed
    it is the only thing saying why. A workflow triggered by this one's completion,
    with pull-requests: write, is the shape to refuse.
    """
    import yaml

    for path in sorted(WORKFLOWS.glob("*.yml")):
        wf = yaml.safe_load(path.read_text())
        triggers = wf.get(True) or wf.get("on") or {}
        if not isinstance(triggers, dict):
            continue
        downstream = triggers.get("workflow_run") or {}
        assert "PR triage" not in (downstream.get("workflows") or []), (
            f"{path.name} runs off PR triage and can publish its findings"
        )


def test_the_triage_job_calls_no_publishing_api() -> None:
    text = TRIAGE.read_text()
    for forbidden in (
        "createComment",
        "updateComment",
        "createReview",
        "addLabels",
        "github-script",
        "pull-requests: write",
        "issues: write",
        "pull_request_target",
    ):
        assert forbidden not in text.replace("`pull_request_target` is not used", ""), (
            f"pr-triage.yml reaches for {forbidden}"
        )


def test_triage_skips_maintainers_and_runs_on_outside_contributions() -> None:
    """The real `if`, evaluated. The report is wrong by construction on our own PRs.

    Only the one expression shape this gate uses is implemented, and anything else
    raises: a condition read as false would make the whole table below vacuous.
    """
    import json
    import re

    condition = " ".join(str(_workflow(TRIAGE)["jobs"]["report"]["if"]).split())
    match = re.fullmatch(
        r"!contains\(fromJSON\('(\[.*\])'\), github\.event\.pull_request\.author_association\)",
        condition,
    )
    assert match, f"the gate is no longer the shape this test can evaluate: {condition}"
    excluded = json.loads(match.group(1))
    assert set(excluded) == {"OWNER", "MEMBER", "COLLABORATOR"}
    for association in ("CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", "NONE"):
        assert association not in excluded, f"{association} would be skipped"
