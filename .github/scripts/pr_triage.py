#!/usr/bin/env python3
"""Answer the four questions a reviewer asks before reading a pull request's code.

Each finding below is decidable from the pull request's metadata and its diff, so
this is a deterministic check rather than a judgement: an AI-written bundle of
nine unrelated fixes is identifiable before anyone spends a build on it, which is
the cost this exists to move back to the contributor.

Writes ``triage.md`` (the report a maintainer reads, with a reply they can paste if
they choose to send one) and ``triage.json`` (the same findings, machine-readable).
Nothing here posts, labels, closes or reviews anything, and the workflow that runs it
holds no token that could. Exits 0 whatever it finds: it reports, it does not gate.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

# A pull request "closes" an issue only through a keyword GitHub itself honours.
# "See #12" or "related to #12" leaves the issue open after the merge.
CLOSES = re.compile(
    r"\b(close[sd]?|fix(e[sd])?|resolve[sd]?)\b\s*:?\s*"
    r"(#\d+|https://github\.com/[\w.-]+/[\w.-]+/issues/\d+)",
    re.IGNORECASE,
)
AI_DISCLOSED = re.compile(
    r"\b(ai|llm|claude|copilot|chatgpt|gpt-|codex|cursor|gemini|agent)\b", re.IGNORECASE
)
# A notebook's .ipynb is generated from its .py. A commit carrying the .ipynb
# alone did not come through jupytext, so the pair is out of sync and the
# provenance stamp no longer describes the source beside it.
NOTEBOOK = re.compile(r"^(.+)\.ipynb$")


def changed_files(base: str, head: str) -> list[str]:
    merge_base = subprocess.run(
        ["git", "merge-base", base, head], capture_output=True, text=True, check=True
    ).stdout.strip()
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{merge_base}..{head}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def top_level_areas(paths: list[str]) -> set[str]:
    """The chapter or top-level directory each path belongs to.

    Two chapters in one pull request is the signal that it bundles unrelated
    fixes. A test beside the fix is not a second area, so ``tests/`` and the
    repository's shared directories do not count.
    """
    shared = {"tests", "utils", "data", ".github", "docs"}
    areas = set()
    for path in paths:
        head = path.split("/")[0]
        if head in shared or "/" not in path:
            continue
        areas.add(head)
    return areas


def find(paths: list[str], body: str) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    areas = top_level_areas(paths)
    if len(areas) > 1:
        findings.append(
            {
                "id": "one-change-per-pr",
                "level": "ask",
                "text": (
                    f"This touches {len(areas)} chapters or case studies "
                    f"({', '.join(sorted(areas))}). CONTRIBUTING.md asks for one defect "
                    "per pull request: one doubtful change in a bundle blocks every good "
                    "one beside it. Please split it."
                ),
            }
        )

    if not CLOSES.search(body or ""):
        findings.append(
            {
                "id": "no-owning-issue",
                "level": "ask",
                "text": (
                    "No issue is closed by this. Open one with a minimal reproduction "
                    "first, then say `Closes #N` here, so the defect is recorded "
                    "independently of the fix."
                ),
            }
        )

    stems = {m.group(1) for p in paths if (m := NOTEBOOK.match(p))}
    unpaired = sorted(stem for stem in stems if f"{stem}.py" not in paths)
    if unpaired:
        findings.append(
            {
                "id": "ipynb-without-py",
                "level": "block",
                "text": (
                    "An `.ipynb` changed without its `.py`: "
                    + ", ".join(f"`{s}.ipynb`" for s in unpaired)
                    + ". The `.py` is the source and the `.ipynb` is generated from it, "
                    "with real outputs stamped in `metadata.ml4t_provenance`. Edit the "
                    "`.py` and sync it; CONTRIBUTING.md gives the command, which differs "
                    "for a text change and a code change."
                ),
            }
        )

    touches_code = any(p.endswith((".py", ".ipynb")) and not p.startswith("tests/") for p in paths)
    has_test = any(p.startswith("tests/") for p in paths)
    if touches_code and not has_test:
        findings.append(
            {
                "id": "no-test",
                "level": "ask",
                "text": (
                    "No test under `tests/` changed. CONTRIBUTING.md asks for one that "
                    "fails before the change and passes with it. If the change computes "
                    "nothing (text, a figure label, a comment), say so and this does not "
                    "apply."
                ),
            }
        )

    if not AI_DISCLOSED.search(body or ""):
        findings.append(
            {
                "id": "no-ai-disclosure",
                "level": "note",
                "text": (
                    "The description does not say whether AI tools were used. Using them "
                    "is fine and saying so is required: it tells a reviewer which claims "
                    "to check rather than read."
                ),
            }
        )

    return findings


MARK = {"block": "**Needs a change**", "ask": "**Please fix**", "note": "Note"}
CONTRIBUTING = (
    "https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/CONTRIBUTING.md"
)


def render(pr: int, findings: list[dict[str, str]]) -> str:
    """The maintainer's report, and under it a reply they can paste or ignore.

    Two audiences in one file deliberately. The findings are addressed to whoever is
    deciding what to do with the pull request; the reply is addressed to the
    contributor and is sent by a person or not at all. An earlier version of this was
    the comment itself, which is why it spoke only to the contributor and promised to
    update itself.
    """
    if not findings:
        return (
            f"## Triage: PR #{pr}\n\n"
            "**Contribution checks pass.** One area, an owning issue, a test, notebook "
            "pairs intact, AI use stated. Nothing mechanical to raise, so review the "
            "code itself.\n"
        )

    lines = [
        f"## Triage: PR #{pr}",
        "",
        f"Mechanical checks against [CONTRIBUTING.md]({CONTRIBUTING}) found "
        f"{len(findings)} item(s). None of this is a verdict on the code, and nothing "
        "has been posted, labelled or closed.",
        "",
    ]
    for f in findings:
        lines.append(f"- {MARK[f['level']]} (`{f['id']}`): {f['text']}")
    lines += [
        "",
        "### Suggested reply",
        "",
        "Paste this if you want to send it, after editing whatever does not fit. It is "
        "a draft, not an action taken.",
        "",
        "```markdown",
        "Thanks for the pull request. Checks against CONTRIBUTING.md raise the "
        "following before anyone reviews the code itself.",
        "",
    ]
    lines += [f"- {f['text']}" for f in findings]
    lines += ["```", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", type=Path, default=Path("triage.md"))
    parser.add_argument("--body", default=None, help="PR body; read from $PR_BODY if absent")
    args = parser.parse_args(argv)

    import os

    body = args.body if args.body is not None else os.environ.get("PR_BODY", "")
    paths = changed_files(args.base, args.head)
    findings = find(paths, body)

    args.out.write_text(render(args.pr, findings))
    Path("triage.json").write_text(
        json.dumps({"pr": args.pr, "findings": findings, "files": paths}, indent=2)
    )
    for f in findings:
        print(f"{f['level']}: {f['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
