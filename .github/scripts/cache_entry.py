#!/usr/bin/env python3
"""Print what the Actions cache API says about one entry, after a restore touched it.

The midweek `hf-cache-touch` job exists on one claim: restoring an entry updates its
last-accessed date, and GitHub deletes an entry nobody has accessed in over seven
days. This prints the date, so the claim is measured in the run's own summary rather
than assumed. If `last_accessed_at` is not the day the job ran, a restore does not
count as an access and the retention approach is wrong, not merely unproven.

Reads nothing and writes nothing. Exits 0 even when the entry is absent or the API
refuses: the job it reports on is already past its useful work by then, and a
reporting step that fails a green job teaches people to ignore the job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def entries(repo: str, token: str) -> list[dict]:
    request = urllib.request.Request(  # noqa: S310 - github.com over https, fixed path
        f"https://api.github.com/repos/{repo}/actions/caches?per_page=100",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read())["actions_caches"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--key", required=True, help="the cache key to report on")
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("no GITHUB_TOKEN, so the cache API cannot be read")
        return 0
    try:
        found = [e for e in entries(args.repo, token) if e["key"] == args.key]
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        # Named rather than bare: an unexpected exception here is a defect in this
        # script and should surface as one.
        print(f"could not read the cache API: {exc}")
        return 0

    if not found:
        print(f"the API lists no cache entry under {args.key}")
        return 0
    for entry in found:
        print(
            f"{entry['key']}  version {entry.get('version', '?')[:12]}  "
            f"{entry['size_in_bytes']} bytes  "
            f"created {entry.get('created_at')}  last accessed {entry.get('last_accessed_at')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
