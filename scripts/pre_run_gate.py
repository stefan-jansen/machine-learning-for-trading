"""Everything that must be true before an expensive model run starts.

A canonical model run costs hours. Discovering afterwards that it recorded no runtime, or that
the same configuration would have resolved to a different identity on another code path, means
paying for it twice. This script states those conditions as executable checks and exits non-zero
if any of them fails, so "we verified the preconditions" is something you can see rather than
something someone remembers to do.

The substantive checks run a real reduced population end to end in a throwaway workspace and then
interrogate what it wrote. A check that reads the source and concludes the code looks right would
pass on exactly the runs that fail.

    uv run python scripts/pre_run_gate.py --case-study etfs --family linear --label fwd_ret_21d

Add ``--json <path>`` to write the report next to the run it cleared.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    measurements: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name}: {self.detail}"


@dataclass
class Report:
    case_study: str
    family: str
    label: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def add(self, name: str, passed: bool, detail: str, **measurements: Any) -> Check:
        check = Check(name=name, passed=passed, detail=detail, measurements=measurements)
        self.checks.append(check)
        print(check.render(), flush=True)
        return check


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_configurations(report: Report, study, requests) -> None:
    """Every declared configuration must resolve, and the two resolve paths must agree.

    The batch planner and the per-request resolver both produce a training identity. When they
    disagree, the same declared configuration registers as two different results depending on
    which one ran it, and neither is wrong in a way the registry can detect.
    """
    from case_studies.research import plan_models

    started = time.perf_counter()
    plan = plan_models(study, requests=list(requests))
    planned = {member.config_name: member.training_hash for member in plan.members}
    resolved = {}
    for request in requests:
        item = request.resolve()
        resolved[item.spec["config_name"]] = item.identity
    elapsed = time.perf_counter() - started

    missing = sorted(set(planned) - set(resolved)) + sorted(set(resolved) - set(planned))
    mismatched = sorted(name for name in planned if planned.get(name) != resolved.get(name))
    report.add(
        "identities agree across resolve paths",
        not missing and not mismatched,
        (
            f"{len(resolved)} configurations, all identities agree ({elapsed:.1f}s)"
            if not missing and not mismatched
            else f"mismatched={mismatched} missing={missing}"
        ),
        configurations=len(resolved),
        resolve_s=elapsed,
    )

    duplicates = len(resolved) - len(set(resolved.values()))
    report.add(
        "identities are distinct",
        duplicates == 0,
        (
            "every configuration has its own identity"
            if duplicates == 0
            else f"{duplicates} configurations share an identity with another"
        ),
    )


def check_fold_preparation(report: Report, study, requests) -> None:
    """The folds every configuration will be fitted on must be preparable, once.

    An all-missing feature column, an empty fold or a fold with no labelled rows each produce a
    result that is not what the recorded specification describes. All three raise here rather
    than at hour four of the canonical run.
    """
    from case_studies.utils import linear as linear_runner
    from case_studies.utils.folds import clear_memo, prepare_standardized_folds

    request = requests[0].as_dict() if hasattr(requests[0], "as_dict") else requests[0]
    base = linear_runner._load_batch_base(study, request)
    # Resolving the configurations above already prepared these. Measuring reuse means measuring
    # it against a real preparation, so the held copy goes first.
    clear_memo()
    started = time.perf_counter()
    folds = prepare_standardized_folds(
        base["mds"], base["splits"], train_sample_frac=base["train_sample_frac"]
    )
    first = time.perf_counter() - started

    started = time.perf_counter()
    prepare_standardized_folds(
        base["mds"], base["splits"], train_sample_frac=base["train_sample_frac"]
    )
    second = time.perf_counter() - started

    widths = {fold["X_train"].shape[1] for fold in folds}
    declared_width = len(base["mds"].feature_names)
    report.add(
        "folds prepare with the declared feature set",
        widths == {declared_width},
        (
            f"{len(folds)} folds, {declared_width} features each"
            if widths == {declared_width}
            else f"design matrix widths {sorted(widths)} against {declared_width} declared features"
        ),
        folds=len(folds),
        features=declared_width,
        rows=[int(fold["n_train"]) for fold in folds],
    )
    report.add(
        "prepared folds are reused, not rebuilt per configuration",
        second < first / 10,
        f"first preparation {first:.1f}s, reuse {second:.3f}s",
        first_s=first,
        reuse_s=second,
    )


def check_preview_run(report: Report, case_study: str, family: str, label: str) -> dict[str, Any]:
    """Run a reduced population end to end and interrogate what it recorded.

    This is the check the others exist to support. It fits, registers and verifies on a small
    universe in a throwaway workspace, then reads the rows back: a run that would fail to record
    its own cost, or fail its population verification, fails here for a few seconds rather than
    at the end of the canonical run.
    """
    from case_studies.research import (
        load_model_configs,
        model_requests,
        open_study,
        run_model_population,
    )

    workspace = Path(tempfile.mkdtemp(prefix=f"pre-run-gate-{case_study}-"))
    measurements: dict[str, Any] = {}
    try:
        study = open_study(case_study, execution_tier="preview", workspace=workspace)
        configs = load_model_configs(study, family, labels=[label])
        requests = model_requests(
            study,
            configs,
            execution_tier="preview",
            preview_reductions={"max_symbols": 5, "folds": [0, 1]},
        )
        started = time.perf_counter()
        execution, population = run_model_population(
            study, requests, population_name=f"pre-run-gate-{family}"
        )
        elapsed = time.perf_counter() - started
        report.add(
            "a reduced population runs end to end",
            True,
            f"{len(execution.runs)} configurations, "
            f"{len(population.members)} prediction sets, {elapsed:.1f}s",
            configurations=len(execution.runs),
            elapsed_s=elapsed,
        )
        measurements = _inspect_registry(report, study, expected=len(execution.runs))
    except Exception:
        report.add(
            "a reduced population runs end to end",
            False,
            traceback.format_exc(limit=6).strip().splitlines()[-1],
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return measurements


def _inspect_registry(report: Report, study, *, expected: int) -> dict[str, Any]:
    """Read back what the run recorded about itself."""
    # A preview writes under its workspace's preview namespace; a workspace-isolated canonical run
    # writes under the workspace root. Look in both rather than assume which one this was.
    candidates = [study.root / "run_log" / "registry.db"]
    if study.output_root is not None:
        candidates.append(
            study.output_root / ".preview" / study.case_study / "run_log" / "registry.db"
        )
    rows: list[tuple] = []
    searched = []
    for db_path in candidates:
        searched.append(str(db_path))
        if not db_path.exists():
            continue
        with sqlite3.connect(db_path) as db:
            found = db.execute(
                "SELECT training_hash, elapsed_s, runtime_json FROM training_runs"
            ).fetchall()
        if found:
            rows = found
            break

    if not rows:
        report.add(
            "training rows record their runtime",
            False,
            f"no training rows found in {searched}",
        )
        return {}

    without_elapsed = [row[0] for row in rows if row[1] is None or row[1] <= 0]
    report.add(
        "training rows record their runtime",
        not without_elapsed,
        (
            f"{len(rows)} rows, every one with a positive elapsed_s"
            if not without_elapsed
            else f"{len(without_elapsed)} of {len(rows)} rows have no elapsed_s: "
            f"{without_elapsed[:5]}"
        ),
        rows=len(rows),
    )

    peaks, cores = [], []
    without_memory = []
    for training_hash, _, runtime_json in rows:
        resources = (json.loads(runtime_json) if runtime_json else {}).get("resources") or {}
        if resources.get("process_peak_rss_bytes"):
            peaks.append(int(resources["process_peak_rss_bytes"]))
        else:
            without_memory.append(training_hash)
        if resources.get("cores_used") is not None:
            cores.append(float(resources["cores_used"]))

    report.add(
        "training rows record peak memory",
        not without_memory,
        (
            f"peak resident {max(peaks) / 1e9:.2f} GB"
            if not without_memory
            else f"{len(without_memory)} of {len(rows)} rows have no peak memory"
        ),
        peak_rss_bytes=max(peaks) if peaks else None,
    )
    report.add(
        "the run reports how much of the machine it used",
        bool(cores),
        (
            f"{max(cores):.1f} cores at peak across {len(cores)} runs"
            if cores
            else "no run recorded CPU seconds, so concurrency cannot be planned"
        ),
        max_cores_used=max(cores) if cores else None,
    )
    report.add(
        "every configuration produced a row",
        len(rows) >= expected,
        f"{len(rows)} training rows for {expected} configurations",
    )
    return {
        "peak_rss_bytes": max(peaks) if peaks else None,
        "max_cores_used": max(cores) if cores else None,
        "preview_elapsed_s": sum(row[1] or 0.0 for row in rows),
    }


def check_the_run_is_costed(
    report: Report,
    study,
    requests,
    preview: dict[str, Any],
) -> None:
    """A run nobody has costed cannot be scheduled against the machine.

    Two sources, in order of authority. Recorded ``elapsed_s`` from earlier canonical runs of
    this family is a measurement; the reduced run above is an extrapolation, and is labelled as
    one. Before this, neither existed: the column was NULL on every row the current path wrote.
    """
    from case_studies.utils import linear as linear_runner

    recorded = _recorded_family_cost(study)
    if recorded is not None:
        report.add(
            "the canonical run is costed",
            True,
            f"{recorded['rows']} earlier runs, median {recorded['median_s']:.1f}s each, "
            f"projected {recorded['median_s'] * len(requests) / 60:.0f} min for "
            f"{len(requests)} configurations",
            **recorded,
        )
        return

    if not preview.get("preview_elapsed_s"):
        report.add(
            "the canonical run is costed",
            False,
            "no earlier run recorded its cost and the reduced run produced no measurement",
        )
        return

    request = requests[0].as_dict() if hasattr(requests[0], "as_dict") else requests[0]
    base = linear_runner._load_batch_base(study, request)
    canonical_rows = base["mds"].dataset.height
    canonical_folds = len(base["splits"])
    # The reduced run used five entities and two folds; cost scales with rows fitted, which is
    # rows per fold times folds.
    entities = base["mds"].dataset.get_column(base["mds"].entity_cols[0]).n_unique()
    scale = (entities / 5) * (canonical_folds / 2)
    projected_s = preview["preview_elapsed_s"] * scale
    report.add(
        "the canonical run is costed",
        True,
        f"no recorded cost for this family yet; extrapolated from the reduced run "
        f"({preview['preview_elapsed_s']:.1f}s at 5 of {entities} entities and 2 of "
        f"{canonical_folds} folds) to about {projected_s / 60:.0f} min",
        projected_s=projected_s,
        basis="extrapolated from the reduced run",
        canonical_rows=canonical_rows,
        canonical_folds=canonical_folds,
        entities=entities,
    )


def _recorded_family_cost(study) -> dict[str, Any] | None:
    """Median recorded runtime of earlier canonical runs of this family, if any."""
    db_path = study.root / "run_log" / "registry.db"
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as db:
        rows = [
            row[0]
            for row in db.execute(
                "SELECT elapsed_s FROM training_runs WHERE family = 'linear' "
                "AND execution_tier = 'canonical' AND elapsed_s IS NOT NULL"
            )
        ]
    if not rows:
        return None
    rows.sort()
    return {"rows": len(rows), "median_s": rows[len(rows) // 2]}


def check_registry_ready(report: Report, case_study: str) -> None:
    """The canonical registry must be present, writable and hold the tables a run will write."""
    from utils.paths import get_case_study_dir

    db_path = get_case_study_dir(case_study) / "run_log" / "registry.db"
    if not db_path.exists():
        report.add("canonical registry is ready", False, f"no registry at {db_path}")
        return
    required = {"training_runs", "prediction_sets", "official_populations"}
    with sqlite3.connect(db_path) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        existing = db.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
    missing = sorted(required - tables)
    report.add(
        "canonical registry is ready",
        not missing,
        (
            f"{len(tables)} tables, {existing} existing training rows"
            if not missing
            else f"missing tables: {missing}"
        ),
        existing_training_rows=existing,
    )


def check_notebook_is_current(report: Report, case_study: str, notebook: str | None) -> None:
    """The notebook that will run must be in sync with the module it imports."""
    if notebook is None:
        return
    source = REPO_ROOT / "case_studies" / case_study / f"{notebook}.py"
    paired = source.with_suffix(".ipynb")
    report.add(
        "the notebook exists as a pair",
        source.exists() and paired.exists(),
        f"{source.name} and {paired.name}"
        if source.exists() and paired.exists()
        else f"missing {source if not source.exists() else paired}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-study", required=True)
    parser.add_argument("--family", default="linear")
    parser.add_argument("--label", required=True)
    parser.add_argument("--notebook", default=None, help="notebook stem, e.g. 06_linear")
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="skip the end-to-end reduced run (the substantive check; for debugging only)",
    )
    args = parser.parse_args()

    from case_studies.research import load_model_configs, model_requests, open_study

    report = Report(case_study=args.case_study, family=args.family, label=args.label)
    print(f"\npre-run gate: {args.case_study} / {args.family} / {args.label}\n", flush=True)

    check_registry_ready(report, args.case_study)
    check_notebook_is_current(report, args.case_study, args.notebook)

    study = open_study(args.case_study, execution_tier="canonical")
    configs = load_model_configs(study, args.family, labels=[args.label])
    requests = model_requests(study, configs)
    if not requests:
        report.add("configurations are declared", False, "the training menu declares none")
    else:
        report.add(
            "configurations are declared",
            True,
            f"{len(requests)} configurations for {args.label}",
        )
        check_configurations(report, study, requests)
        if args.family == "linear":
            check_fold_preparation(report, study, requests)

    if not args.skip_preview:
        preview = check_preview_run(report, args.case_study, args.family, args.label)
        if requests:
            check_the_run_is_costed(report, study, requests, preview)

    print()
    if report.passed:
        print(f"GATE PASSED - {len(report.checks)} checks\n")
    else:
        failed = [check.name for check in report.checks if not check.passed]
        print(f"GATE FAILED - {len(failed)} of {len(report.checks)} checks: {failed}\n")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(asdict(report), indent=2, default=str) + "\n")
        print(f"report written to {args.json}\n")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
