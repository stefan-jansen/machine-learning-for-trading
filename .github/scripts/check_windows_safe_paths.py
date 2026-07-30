#!/usr/bin/env python3
"""Fail if any tracked path cannot exist on Windows.

Git on Windows refuses to check out a path containing a character NTFS reserves,
so a single bad filename makes the whole repository unclonable there - not the
one file, the clone. That is what happened with

    data/factors/aqr/source/ESG_efficient_frontier_portfolios_vF.xlsx?sc_lang=en

which reached readers as issue #433: the `?sc_lang=en` is a URL query string that
the AQR downloader carried into the filename. Every Windows reader hit

    error: invalid path 'data/factors/aqr/source/ESG_..._vF.xlsx?sc_lang=en'

and could not check the tree out at all.

Static check over `git ls-files` - no imports, no network, no data.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict

# Characters NTFS reserves. Backslash and forward slash are omitted: git already
# uses `/` as the separator and rejects `\` in tracked names.
RESERVED_CHARS = set('<>:"|?*') | {chr(c) for c in range(32)}

# Device names, reserved with or without an extension, in any case.
RESERVED_NAMES = (
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

_RESERVED_CHAR_RE = re.compile(f"[{re.escape(''.join(sorted(RESERVED_CHARS)))}]")


def tracked_paths() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def check(paths: list[str]) -> list[str]:
    """Return one human-readable failure line per offending path."""
    failures: list[str] = []
    lowered: dict[str, list[str]] = defaultdict(list)

    for path in paths:
        lowered[path.lower()].append(path)

        for segment in path.split("/"):
            bad = sorted({m.group() for m in _RESERVED_CHAR_RE.finditer(segment)})
            if bad:
                shown = ", ".join(repr(c) for c in bad)
                failures.append(f"{path}\n    reserved character(s) on Windows: {shown}")

            # `NUL.txt` is as reserved as `NUL`; the stem is what matters.
            if segment.split(".")[0].upper() in RESERVED_NAMES:
                failures.append(f"{path}\n    reserved device name on Windows: {segment!r}")

            # Windows silently strips these, so the checked-out name would differ.
            if segment != segment.rstrip(" .") and segment not in (".", ".."):
                failures.append(f"{path}\n    trailing space or dot: {segment!r}")

    # Windows filesystems are case-insensitive: two paths differing only in case
    # collide on checkout and one silently overwrites the other.
    for group in lowered.values():
        if len(group) > 1:
            failures.append(
                "case-insensitive collision on Windows:\n    " + "\n    ".join(sorted(group))
            )

    return failures


def main() -> int:
    failures = check(tracked_paths())
    if failures:
        print("Tracked paths that break `git clone` on Windows:\n", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print(
            f"\n{len(failures)} problem(s). Rename with `git mv`; a path like this makes the "
            "whole repository unclonable on Windows, not just the file.",
            file=sys.stderr,
        )
        return 1

    print("OK: all tracked paths are valid on Windows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
