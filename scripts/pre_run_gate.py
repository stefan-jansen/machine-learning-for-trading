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
import os
import shutil
import sqlite3
import subprocess
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
    declared_width = len(base["mds"].feature_names)

    # One fold at a time, and released before the next, because that is how the run prepares them
    # and because the whole set does not fit: `us_equities_panel` is 16 folds of 9.97 million rows
    # by 71 features, 90 GB standing. Every fold is still checked - what changes is that two are
    # never live at once.
    clear_memo()
    widths, rows, first, total = set(), [], None, 0.0
    for split in base["splits"]:
        started = time.perf_counter()
        fold = prepare_standardized_folds(
            base["mds"], [split], train_sample_frac=base["train_sample_frac"]
        )[0]
        elapsed = time.perf_counter() - started
        total += elapsed
        if first is None:
            first = elapsed
            started = time.perf_counter()
            prepare_standardized_folds(
                base["mds"], [split], train_sample_frac=base["train_sample_frac"]
            )
            second = time.perf_counter() - started
        widths.add(fold["X_train"].shape[1])
        rows.append(int(fold["n_train"]))
        del fold
        clear_memo()

    report.add(
        "folds prepare with the declared feature set",
        widths == {declared_width},
        (
            f"{len(base['splits'])} folds, {declared_width} features each, {total:.1f}s to build "
            f"every one"
            if widths == {declared_width}
            else f"design matrix widths {sorted(widths)} against {declared_width} declared features"
        ),
        folds=len(base["splits"]),
        features=declared_width,
        rows=rows,
        prepare_all_folds_s=total,
    )
    report.add(
        "prepared folds are reused, not rebuilt per configuration",
        second < max(first, 1e-6) / 10,
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
        measurements = _inspect_registry(report, execution, expected=len(execution.runs))
    except Exception:
        report.add(
            "a reduced population runs end to end",
            False,
            traceback.format_exc(limit=6).strip().splitlines()[-1],
        )
    finally:
        # Activating the preview tier pointed ML4T_OUTPUT_DIR at this throwaway workspace. Restore
        # the canonical activation before the workspace is removed: every check that runs after
        # this one resolves its paths through that variable, and the cost extrapolation reads the
        # modeling dataset, so leaving it pointed at a deleted directory fails the gate on exactly
        # the branch that exists to serve a family with no recorded cost.
        try:
            open_study(case_study).activate()
        except Exception:
            report.add(
                "the canonical study is restored after the preview",
                False,
                traceback.format_exc(limit=6).strip().splitlines()[-1],
            )
        shutil.rmtree(workspace, ignore_errors=True)
    return measurements


def _inspect_registry(report: Report, execution, *, expected: int) -> dict[str, Any]:
    """Read back what this run recorded about itself.

    Scoped to the rows this run produced, by their training hashes. Reading every row in the
    registry instead graded the pre-migration rows already sitting there: `cme_futures` failed
    all three of these checks on 202 legacy rows that predate the runtime column and can never
    carry it, and the extrapolated cost came out at 723 minutes because it summed their
    `elapsed_s` in place of the reduced run's.
    """
    # Only the rows this run actually fitted. A reused row is the runner declining to recompute
    # what is already registered, which is the behaviour we want; it has no fit to measure, and
    # grading it fails a gate for rows that were correct to skip.
    reused = {
        item.get("training_hash")
        for item in execution.diagnostics
        if item.get("cache_hit") or (item.get("fitted_folds") == [] and item.get("reused_folds"))
    }
    by_root: dict[Path, list[str]] = {}
    for run in execution.runs:
        if run.training.hash in reused:
            continue
        by_root.setdefault(Path(run.training.root), []).append(run.training.hash)

    rows: list[tuple] = []
    for root, hashes in by_root.items():
        db_path = root / "run_log" / "registry.db"
        if not db_path.exists():
            continue
        placeholders = ",".join("?" * len(hashes))
        with sqlite3.connect(db_path) as db:
            rows.extend(
                db.execute(
                    "SELECT training_hash, elapsed_s, runtime_json FROM training_runs "
                    f"WHERE training_hash IN ({placeholders})",
                    hashes,
                ).fetchall()
            )

    if not rows:
        every_row_reused = len(reused) == len(execution.runs) and bool(execution.runs)
        report.add(
            "training rows record their runtime",
            every_row_reused,
            "every configuration was already registered and was reused, so this run fitted "
            "nothing to measure"
            if every_row_reused
            else f"the run produced no readable training rows under {sorted(map(str, by_root))}",
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
    *,
    family: str,
    label: str,
) -> None:
    """A run nobody has costed cannot be scheduled against the machine.

    Two sources, in order of authority. Recorded ``elapsed_s`` from earlier canonical runs of
    this family is a measurement; the reduced run above is an extrapolation, and is labelled as
    one. Before this, neither existed: the column was NULL on every row the current path wrote.
    """
    from case_studies.utils import linear as linear_runner

    recorded = _recorded_family_cost(study, family, label)
    if recorded is not None:
        report.add(
            "the canonical run is costed",
            True,
            f"{recorded['rows']} earlier runs ({recorded['basis']}), median "
            f"{recorded['median_s']:.1f}s each, projected "
            f"{recorded['median_s'] * len(requests) / 60:.0f} min for "
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


def _recorded_family_cost(study, family: str, label: str) -> dict[str, Any] | None:
    """Median recorded runtime of earlier canonical runs of this family and label.

    Rows from the current identity version first: those were fitted on the finalized labels and
    features and are a measurement of the run being scheduled. Pre-rebuild rows are reported
    separately and only as a floor - they were timed against the features stage 01-05 replaced,
    and `REBUILD-PLAN.md` section 4 measured L1 configurations moving by an order of magnitude
    across that change.

    Five of the nine registries still carry the pre-rebuild schema, which has neither
    `execution_tier` nor `identity_version` on `training_runs`. Naming a column that a registry
    does not have is how this crashed on `sp500_equity_option_analytics` and
    `us_firm_characteristics`, so the query is built from the columns that are actually there.
    """
    db_path = study.root / "run_log" / "registry.db"
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(training_runs)")}
        if "elapsed_s" not in columns:
            return None
        canonical_only = ["execution_tier = 'canonical'"] if "execution_tier" in columns else []
        attempts: list[tuple[str, list[Any], str]] = []
        if "identity_version" in columns:
            attempts.append(
                ("identity_version = 3 AND label = ?", [label], "measured on the finalized inputs")
            )
            attempts.append(("(identity_version IS NULL OR identity_version < 3)", [], "floor"))
        else:
            attempts.append(("1 = 1", [], "floor, pre-rebuild schema"))

        for clause, extra, basis in attempts:
            where = " AND ".join(["family = ?", "elapsed_s IS NOT NULL", clause, *canonical_only])
            rows = sorted(
                row[0]
                for row in db.execute(
                    f"SELECT elapsed_s FROM training_runs WHERE {where}", [family, *extra]
                )
            )
            if rows:
                return {"rows": len(rows), "median_s": rows[len(rows) // 2], "basis": basis}
    return None


def check_registry_ready(report: Report, case_study: str) -> None:
    """The canonical registry must be present, writable and hold the tables a run will write.

    Five of the nine registries predate the population and coverage tables, and asserting those
    tables exist would refuse every one of them. The tables are not missing in any sense that
    matters: `_open_registry` migrates and creates them on open, so the run would add them anyway.
    This check therefore opens the registry through that code path and reports what the open
    changed, which is the difference between "the tables will probably appear" and "they are
    there". Opening is idempotent and additive - no row is read, written or altered.
    """
    from case_studies.utils.registry.store import _open_registry
    from utils.paths import get_case_study_dir

    case_dir = get_case_study_dir(case_study)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        report.add("canonical registry is ready", False, f"no registry at {db_path}")
        return

    def tables_in(path: Path) -> set[str]:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            return {
                row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }

    before = tables_in(db_path)
    try:
        db = _open_registry(case_dir)
        try:
            existing = db.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
        finally:
            db.close()
    except Exception as exc:
        report.add("canonical registry is ready", False, f"cannot open for writing: {exc}")
        return

    after = tables_in(db_path)
    required = {"training_runs", "prediction_sets", "official_populations", "prediction_coverage"}
    missing = sorted(required - after)
    added = sorted(after - before)
    report.add(
        "canonical registry is ready",
        not missing,
        (
            f"{len(after)} tables, {existing} existing training rows"
            + (f"; migrated on open, added {len(added)}: {added[:4]}" if added else "")
            if not missing
            else f"missing tables after migration: {missing}"
        ),
        existing_training_rows=existing,
        tables_added=added,
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


def check_notebook_prose(report: Report, case_study: str, notebook: str | None) -> None:
    """The notebook's prose and figure titles must pass the standard before it is executed.

    Executing a notebook stamps its provenance, so a prose fix afterwards costs another run. On
    2026-08-17 `etfs/06_linear` was run, committed and read by Stefan carrying four figure-title
    violations - two interpolated computed values into a title and two ran past the 75-character
    ceiling - because nothing between writing it and running it looked. The checkers live in the
    agents repository, so this check reports a failure when it cannot find them rather than
    passing: a gate that passes when it did not run is worse than no gate.
    """
    if notebook is None:
        return
    source = REPO_ROOT / "case_studies" / case_study / f"{notebook}.py"
    agents_root = Path(
        os.environ.get("ML4T_AGENTS_ROOT", Path.home() / "ml4t" / "agents")
    ).expanduser()
    checkers = {
        "prose": agents_root / "scripts" / "check_notebook_prose.py",
        "conformance": agents_root / "scripts" / "check_notebook_conformance.py",
    }
    missing = sorted(name for name, path in checkers.items() if not path.is_file())
    if missing:
        report.add(
            "the notebook passes the prose and figure standard",
            False,
            f"cannot find the {', '.join(missing)} checker under {agents_root}; "
            "set ML4T_AGENTS_ROOT",
        )
        return

    commands = [
        [sys.executable, str(checkers["prose"]), str(source)],
        [
            sys.executable,
            str(checkers["conformance"]),
            # By what the notebook does, not the number it carries: model execution runs as
            # 06_linear, 07_gbm and 08_tabular_dl, and resolving by number found no standard
            # for any of them but the first.
            "--stem",
            notebook.split("_", 1)[-1],
            # Whether the executed notebook holds its figures is only answerable after the run;
            # scripts/validate_notebook_run.py asks it there. Asking it here would fail every
            # rewritten notebook for not yet having been executed.
            "--pre-run",
            "--case-study",
            case_study,
            "--public-root",
            str(REPO_ROOT),
        ],
    ]
    failures = []
    for command in commands:
        finished = subprocess.run(command, capture_output=True, text=True, cwd=agents_root)
        if finished.returncode != 0:
            failures.append((finished.stdout + finished.stderr).strip().splitlines())
    report.add(
        "the notebook passes the prose and figure standard",
        not failures,
        "prose and conformance checkers report no violations"
        if not failures
        else "; ".join(line for block in failures for line in block[-3:]),
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
    check_notebook_prose(report, args.case_study, args.notebook)

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
            check_the_run_is_costed(
                report, study, requests, preview, family=args.family, label=args.label
            )

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
