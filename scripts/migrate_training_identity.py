#!/usr/bin/env python3
"""Reuse a complete model run after a non-computational identity migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from case_studies.research import Study
from case_studies.utils.registry import migrate_equivalent_training_identity
from utils.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-study", required=True)
    parser.add_argument("--source-training-hash", required=True)
    parser.add_argument("--target-spec", required=True, type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--release-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    target_spec = json.loads(args.target_spec.read_text())
    if args.workspace is None:
        study = Study.regenerate(args.case_study, release_root=args.release_root)
    else:
        study = Study.open(
            args.case_study,
            workspace=args.workspace,
            release_root=args.release_root,
        )
    migration = migrate_equivalent_training_identity(
        study,
        args.source_training_hash,
        target_spec,
    )
    print(
        json.dumps(
            {
                "created": migration.created,
                "prediction_map": migration.prediction_map,
                "source_training_hash": migration.source_training_hash,
                "target_training_hash": migration.target_training_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
