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


def _index_blobs(paths: list[str]) -> dict[str, str]:
    """path -> the blob git would commit for it. Absent means the commit has no such file.

    Asking about the listed paths rather than about the staged ones is what makes a
    deletion and a rename visible. Both leave the recorded path with no blob in the index,
    and a rename additionally writes the same content somewhere the manifest cannot vouch
    for, so neither can be distinguished from "still there" by looking at what changed.
    """
    if not paths:
        return {}
    out = _git("ls-files", "--stage", "-z", "--", *paths)
    blobs: dict[str, str] = {}
    for entry in out.split("\0"):
        if not entry:
            continue
        meta, _, path = entry.partition("\t")
        fields = meta.split()
        if len(fields) >= 2 and path:
            blobs[path] = fields[1]
    return blobs


def main(argv: list[str]) -> int:
    if not MANIFEST.exists():
        return 0

    audit = "--audit" in argv
    if audit:
        recorded = _parse(MANIFEST.read_text())
        moved = []
        for path, blob in recorded.items():
            target = REPO_ROOT / path
            if not target.exists():
                moved.append((path, blob, "GONE", "no such file in the worktree"))
            elif _git("hash-object", str(target)).strip() != blob:
                moved.append((path, blob, "CHANGED", "worktree differs"))
        for path, blob, kind, why in moved:
            print(f"{kind:8} {path}  verified at {blob[:8]}, {why}")
        print(f"--- {len(moved)} of {len(recorded)} verified notebooks have moved ---")
        return 1 if moved else 0

    # The manifest as this commit will leave it: a row dropped or updated here is consent.
    after = _parse(_git("show", ":" + MANIFEST.name) or MANIFEST.read_text())

    violations = []
    blobs = _index_blobs(list(after))
    for path, recorded in after.items():
        staged = blobs.get(path)
        if staged != recorded:
            violations.append((path, recorded, staged))

    if not violations:
        return 0

    print("a verified notebook is being changed while its verification still stands:")
    for path, recorded, staged in violations:
        wrote = f"this commit writes {staged[:8]}" if staged else "this commit removes it"
        print(f"  {path}\n      verified at {recorded[:8]}, {wrote}")
    print(
        f"\nA verdict describes one exact version of the file. Changing it here voids the verdict\n"
        f"without saying so, which is how four batch rewrites erased ~30 verifications in the week\n"
        f"to 2026-08-06.\n"
        f"\nIf the change is intended, drop or update the row in {MANIFEST.name} in this same commit\n"
        f"and re-run the verifier afterwards. If it is not, unstage the notebook.\n"
        f"A renamed notebook is a removal here and an unverified file at the new path: move the\n"
        f"row too, and it is the verifier who decides whether the verdict survives the move."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
