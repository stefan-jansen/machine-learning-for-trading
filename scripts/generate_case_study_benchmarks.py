#!/usr/bin/env python
"""Regenerate full-universe equal-weight benchmarks from canonical label artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from case_studies.utils.benchmark import generate_benchmark  # noqa: E402
from utils.paths import get_case_study_dir  # noqa: E402


def _continuous_labels(case_study: str) -> list[str]:
    setup_path = get_case_study_dir(case_study, create=False) / "config" / "setup.yaml"
    setup = yaml.safe_load(setup_path.read_text())
    labels = setup["labels"]
    classification = set((labels.get("classification_eval_label") or {}).keys())
    return [
        label
        for label in [labels["primary"], *(labels.get("variants") or [])]
        if label not in classification
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-study", required=True)
    parser.add_argument("--label", action="append", dest="labels")
    args = parser.parse_args()

    labels = args.labels or _continuous_labels(args.case_study)
    for label in labels:
        parquet_path, json_path = generate_benchmark(args.case_study, label)
        print(f"{label}: {parquet_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
