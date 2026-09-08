#!/usr/bin/env python3
"""Relabel a registry's stored fold ids onto the chronological numbering, without refitting.

Two steps, and the first writes nothing::

    scripts/migrate_fold_renumbering.py --case-study us_firm_characteristics
    scripts/migrate_fold_renumbering.py --case-study us_firm_characteristics --apply

The plan reproduces every derived identity from what is stored before proposing to rewrite
it, so a registry it cannot explain produces refusals and no plan. ``--apply`` refuses while
any refusal stands: a partial migration leaves a registry that is neither migrated nor not,
which is worse than either.

The tree it replaces is kept at ``run_log.pre-fold-renumber`` rather than deleted, so the
migration is reversible by swapping the two directories back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from case_studies.utils.registry.fold_renumbering import (
    apply_fold_renumbering,
    plan_fold_renumbering,
)
from utils.paths import get_case_study_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-study", required=True)
    parser.add_argument("--case-dir", type=Path, help="override the resolved case study root")
    parser.add_argument("--plan-out", type=Path, help="write the full plan as JSON")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="rewrite the registry; without it nothing is written",
    )
    args = parser.parse_args()

    case_dir = args.case_dir or get_case_study_dir(args.case_study)
    plan = plan_fold_renumbering(case_dir)

    summary = {
        "case_study": args.case_study,
        "case_dir": str(case_dir),
        "training_runs_renumbered": len(plan.remaps),
        "training_runs_unchanged": len(plan.unchanged),
        "training_runs_incomplete": plan.incomplete,
        "prediction_sets": len(plan.prediction_map),
        "backtests": len(plan.backtest_map),
        "populations": len(plan.population_map),
        "cohort_digests_reproduced": len(plan.cohort_digests),
        "cohorts_needing_recomputation": plan.unresolved_cohorts,
        "refusals": plan.refusals,
    }
    if args.plan_out is not None:
        args.plan_out.write_text(json.dumps(plan.as_json(), indent=1, sort_keys=True) + "\n")
        summary["plan_written_to"] = str(args.plan_out)

    if args.apply:
        summary["report"] = apply_fold_renumbering(case_dir, plan)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
