"""Publish `download_all.py --report` to the GitHub Actions run summary.

`Reader install` reaches seven third-party sources. When one of them is down the job
fails with an exit code, and which dataset failed is the whole diagnosis: one source
failing is an outage to wait out, all seven failing points at egress from the runner,
which is ours to fix. That distinction lived only in the step log, and a log that cannot
be retrieved (`gh run view --log` returned nothing for run 33988501549) leaves nothing
behind. The step summary is part of the run's own record and is served by the API
independently of the log.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def render(report: dict[str, object]) -> str:
    datasets = report.get("datasets") or {}
    failed = report.get("failed") or []
    lines = [
        f"### Free dataset download ({report.get('mode', 'unknown')} mode)",
        "",
        f"{report.get('completed')} of {report.get('total')} datasets arrived.",
        "",
        "| Dataset | Result |",
        "| --- | --- |",
    ]
    lines += [
        f"| {name} | {'OK' if arrived else '**FAILED**'} |" for name, arrived in datasets.items()
    ]
    lines.append("")
    if failed:
        lines += [
            f"Failed: {', '.join(failed)}.",
            "",
            "One source failing is an upstream outage. Every source failing points at "
            "network egress from the runner.",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    report_path = Path(argv[1]) if len(argv) > 1 else Path("download-report.json")
    if not report_path.is_file():
        print(f"No download report at {report_path}; nothing to publish.")
        return 0

    summary = render(json.loads(report_path.read_text(encoding="utf-8")))
    print(summary)

    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if not destination:
        return 0
    with Path(destination).open("a", encoding="utf-8") as handle:
        handle.write(summary + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
