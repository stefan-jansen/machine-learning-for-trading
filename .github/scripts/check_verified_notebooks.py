"""Gate: a notebook that has been verified may not be changed by accident.

Verification is an assertion about one exact version of a file. Four times in the seven days to
2026-08-06, a batch "bring everything to the standard" commit landed on top of notebooks that had
already been verified - PR #478 did it to 34 of 45 in a single commit - and every one of those
verdicts silently stopped describing anything that exists. Nothing in the tooling noticed, because
nothing recorded which version had been verified.

``.verified-notebooks.tsv`` records it: one row per verified notebook, with the git blob hash of the
``.py`` at the moment its verifier signed off. This gate fails a commit that changes a listed ``.py``
while leaving its row intact. Discarding a verification stays possible and becomes explicit - drop or
update the row in the same commit, which puts it in the diff where a reviewer sees it.

Usage::

    uv run python .github/scripts/check_verified_notebooks.py            # staged changes
    uv run python .github/scripts/check_verified_notebooks.py --audit    # worktree vs the manifest
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / ".verified-notebooks.tsv"
HEADER = ("path", "py_blob", "verified_at", "verifier")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout


def _parse(text: str) -> dict[str, str]:
    """path -> py_blob, from a manifest's text. Blank lines and the header are skipped."""
    rows: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if parts[0] == HEADER[0]:
            continue
        if len(parts) < 2:
            raise SystemExit(f"{MANIFEST.name}: malformed row: {line!r}")
        rows[parts[0]] = parts[1]
    return rows


def _staged_paths() -> list[str]:
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACMRT")
    return [p for p in out.splitlines() if p.endswith(".py")]


def _staged_blob(path: str) -> str | None:
    """The blob hash git would commit for ``path``, or None if it is not staged."""
    out = _git("ls-files", "--stage", "--", path).split()
    return out[1] if len(out) >= 2 else None


def main(argv: list[str]) -> int:
    if not MANIFEST.exists():
        return 0

    audit = "--audit" in argv
    if audit:
        recorded = _parse(MANIFEST.read_text())
        changed = [
            (p, b)
            for p, b in recorded.items()
            if (REPO_ROOT / p).exists() and _git("hash-object", str(REPO_ROOT / p)).strip() != b
        ]
        for path, blob in changed:
            print(f"CHANGED  {path}  verified at {blob[:8]}, worktree differs")
        print(f"--- {len(changed)} of {len(recorded)} verified notebooks have moved ---")
        return 1 if changed else 0

    # The manifest as this commit will leave it: a row dropped or updated here is consent.
    after = _parse(_git("show", ":" + MANIFEST.name) or MANIFEST.read_text())

    violations = []
    for path in _staged_paths():
        recorded = after.get(path)
        if recorded is None:
            continue
        staged = _staged_blob(path)
        if staged is not None and staged != recorded:
            violations.append((path, recorded, staged))

    if not violations:
        return 0

    print("a verified notebook is being changed while its verification still stands:")
    for path, recorded, staged in violations:
        print(f"  {path}\n      verified at {recorded[:8]}, this commit writes {staged[:8]}")
    print(
        f"\nA verdict describes one exact version of the file. Changing it here voids the verdict\n"
        f"without saying so, which is how four batch rewrites erased ~30 verifications in the week\n"
        f"to 2026-08-06.\n"
        f"\nIf the change is intended, drop or update the row in {MANIFEST.name} in this same commit\n"
        f"and re-run the verifier afterwards. If it is not, unstage the notebook."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
