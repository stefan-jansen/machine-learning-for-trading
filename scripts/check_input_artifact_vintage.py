#!/usr/bin/env python3
"""Every input artifact on disk is a vintage the registry's runs were fitted on.

A kept-registry case study has four pre-flight resolutions that can refuse a run: input
artifact vintage, population hash, causal identity, and candidate sets.
``scripts/check_supersedes_literals.py`` covers the last three. This covers the first, which
until now had no warning but the run itself stopping (ml4t/agent-workspace#1123).

``_enforce_input_artifact_vintage`` refuses a run that would join a population fitted on a
different vintage of the same input artifact. Regenerate a stage-03 or stage-04 artifact and
every later run refuses until ``declare_artifact_supersession`` records which sha the new one
replaces. On 2026-09-10 ``us_equities_panel/06_linear`` hit exactly that at plan time:
``model_based.parquet`` on disk hashed ``0e74d15f`` while the registered training runs for
``fwd_ret_1d`` had been fitted against ``86ece972``.

**What this costs and what it saves, because the two differ from the other three.** This
refusal fires BEFORE the fit - ``register_training_run`` runs ahead of it on every path - so
it costs a launch and a queue slot rather than compute. The value is not a loss prevented but
a chain that can be queued knowing it will not stop ten seconds in.

**The declared supersession is what makes a second vintage legitimate**, so a check that read
only the disk and the runs would report every deliberate regeneration as a defect. The
condition is narrower: the sha on disk differs from what the registry's runs pin, *and*
``artifact_supersessions`` holds no row retiring the pinned one. That is
``input_artifact_vintage_conflicts``, and this script calls it rather than restating it - a
pre-flight that passes where registration refuses is a green light for a chain that will not
run.

**What is NOT checked is reported, because a checker that answers green for what it did not
look at is the failure this whole class is about.** Two things fall outside.

An artifact NAME it cannot locate: three resolve from the case study's own specs
(``financial``, ``model_based``, ``label``) and ``eval_label`` through the classification
mapping the loader uses. Any other name is reported ``unresolved``.

A training run whose spec records its inputs in a shape ``_input_artifact_shas`` does not
read, reported ``unchecked`` per family. Nothing in the nine live registries is in that state
as of ml4t/agent-workspace#1137: the rule reads all three recorded shapes now - ``artifacts``,
the ``deep_learning`` payload nested one level deeper at
``input_data_spec.input_data_spec.artifacts``, and the ``latent_factors`` ``files`` list of
``{role, sha256}`` with a ``sha256:`` prefix (ml4t/agent-workspace#891). Before that widening
the two missed 119 runs, 77 sequence and 42 latent, which neither registration nor this
script checked.

The category stays, and stays able to fire, because that is the whole shape of this file: a
checker that walks only what the rule returns would report a registry of nothing but
unreadable runs as entirely clean. It is still reported rather than fatal - a pre-flight
stricter than the rule it previews would refuse a chain registration would accept, which is a
different failure and a worse one.

Usage::

    python scripts/check_input_artifact_vintage.py                    # every case study
    python scripts/check_input_artifact_vintage.py --case-study etfs  # one
    python scripts/check_input_artifact_vintage.py --json

Exits non-zero when any artifact is on an undeclared vintage.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_studies.utils.registry.registration import (  # noqa: E402
    _declared_artifact_supersessions,
    _input_artifact_shas,
    _registered_artifact_shas,
    input_artifact_vintage_conflicts,
)
from utils.artifact_specs import load_feature_spec, load_label_spec  # noqa: E402
from utils.modeling import (
    _sha256_file,  # noqa: E402
    get_classification_eval_label,  # noqa: E402
)


@dataclass(frozen=True)
class Finding:
    case_study: str
    label: str
    artifact: str
    # current            the runs for this label were fitted on the file that is on disk
    # superseded         the disk vintage is pinned and every other pinned one is retired
    # mixed              the disk vintage is pinned and another one is not retired
    # accepted_replacement  the disk vintage is pinned by no run yet and retires every pinned one
    # undeclared         registration would refuse this run
    # unresolved         an artifact name this script cannot locate
    # unchecked          a run whose inputs are recorded in a shape the rule does not read
    # absent             pinned by a run and not on disk
    status: str
    on_disk: str | None
    pinned: tuple[str, ...]
    detail: str

    @property
    def is_failure(self) -> bool:
        return self.status == "undeclared"


def _default_artifacts_root() -> Path:
    return Path.home() / "ml4t" / "artifacts" / "case_studies"


def _relative(spec: dict | None, default: str) -> str:
    """Where the producer was told to write it, defaulting to the canonical layout.

    ``utils.artifact_specs.resolve_storage_path`` joins this to ``get_case_study_dir``, which
    answers with the WORKTREE's case_studies/<id>. The artifacts are not there - a checkout
    carries a gitignored ``run_log`` symlink and no features or labels at all - so the
    relative half is taken from the spec and joined to the artifacts root instead.
    """
    if spec is None:
        return default
    return str((spec.get("storage") or {}).get("path", default))


def _artifact_path(case_dir: Path, case_study: str, name: str, label: str) -> Path | None:
    if name == "financial":
        return case_dir / _relative(
            load_feature_spec(case_study, "financial"), "features/financial.parquet"
        )
    if name == "model_based":
        return case_dir / _relative(
            load_feature_spec(case_study, "model_based"), "features/model_based.parquet"
        )
    if name == "label":
        return case_dir / _relative(load_label_spec(case_study, label), f"labels/{label}.parquet")
    if name == "eval_label":
        try:
            eval_label = get_classification_eval_label(case_study, label)
        except (KeyError, ValueError, OSError):
            # KeyError: the label is a regression target and declares no eval label, which is
            # the common case. OSError: no setup.yaml, which is every synthetic case study.
            # Either way the name cannot be located, and the caller reports it as unchecked
            # rather than passing it.
            return None
        return case_dir / _relative(
            load_label_spec(case_study, eval_label), f"labels/{eval_label}.parquet"
        )
    return None


def _unread_input_shapes(con: sqlite3.Connection, case_study: str) -> list[Finding]:
    """Runs whose recorded inputs ``_input_artifact_shas`` returns nothing for.

    Silence here would be the same defect one level up, and it was: the rule read
    ``computation.input_data_spec.artifacts`` alone while two producers recorded somewhere
    else, and a checker that only walks what the rule returns reports a registry of nothing
    but those runs as entirely clean.

    Written against what the rule RETURNS rather than against a list of shapes, so widening
    the rule empties this by itself and a producer that invents a fourth shape lands here
    without anyone editing it.
    """
    try:
        rows = list(con.execute("SELECT family, label, spec_json FROM training_runs"))
    except sqlite3.DatabaseError:
        return []
    unread: dict[tuple[str, str], int] = {}
    for family, label, spec_json in rows:
        if _input_artifact_shas(spec_json):
            continue
        try:
            nested = (json.loads(spec_json) or {}).get("computation", {}).get("input_data_spec", {})
        except (TypeError, ValueError):
            nested = {}
        if "files" in nested:
            shape = "input_data_spec.files"
        elif isinstance(nested.get("input_data_spec"), dict):
            shape = "input_data_spec.input_data_spec.artifacts"
        else:
            shape = "no artifact shas recorded"
        unread[(str(family), shape)] = unread.get((str(family), shape), 0) + 1
    return [
        Finding(
            case_study=case_study,
            label="",
            artifact=family,
            status="unchecked",
            on_disk=None,
            pinned=(),
            detail=(
                f"{count} {family} run(s) record their inputs at {shape}, which the vintage "
                "rule does not read, so registration does not check their vintage either and "
                "neither does this"
            ),
        )
        for (family, shape), count in sorted(unread.items())
    ]


def check_case_study(
    case_study: str, *, artifacts_root: Path, digests: dict[Path, str] | None = None
) -> list[Finding]:
    digests = {} if digests is None else digests
    case_dir = artifacts_root / case_study
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.is_file():
        return []

    def sha(path: Path) -> str | None:
        if not path.is_file():
            return None
        if path not in digests:
            digests[path] = _sha256_file(path)
        return digests[path]

    findings: list[Finding] = []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        try:
            labels = sorted(
                str(row[0])
                for row in con.execute("SELECT DISTINCT label FROM training_runs")
                if row[0]
            )
        except sqlite3.DatabaseError as exc:
            return [
                Finding(
                    case_study=case_study,
                    label="",
                    artifact="",
                    status="unresolved",
                    on_disk=None,
                    pinned=(),
                    detail=f"registry unreadable: {exc}",
                )
            ]
        findings.extend(_unread_input_shapes(con, case_study))
        for label in labels:
            registered = _registered_artifact_shas(con, label=label)
            incoming: dict[str, str] = {}
            for name, pinned in sorted(registered.items()):
                path = _artifact_path(case_dir, case_study, name, label)
                if path is None:
                    findings.append(
                        Finding(
                            case_study=case_study,
                            label=label,
                            artifact=name,
                            status="unresolved",
                            on_disk=None,
                            pinned=tuple(sorted(pinned)),
                            detail=(
                                f"{len(pinned)} sha(s) pinned for an artifact name this script "
                                "cannot locate, so its vintage is unchecked"
                            ),
                        )
                    )
                    continue
                on_disk = sha(path)
                if on_disk is None:
                    findings.append(
                        Finding(
                            case_study=case_study,
                            label=label,
                            artifact=name,
                            status="absent",
                            on_disk=None,
                            pinned=tuple(sorted(pinned)),
                            detail=f"{path} is pinned by {len(pinned)} run(s) and is not on disk",
                        )
                    )
                    continue
                incoming[name] = on_disk

            conflicts = {
                conflict.artifact_name: conflict
                for conflict in input_artifact_vintage_conflicts(
                    con, label=label, incoming=incoming
                )
            }
            for name, on_disk in incoming.items():
                pinned = tuple(sorted(registered.get(name, set())))
                if name in conflicts:
                    findings.append(
                        Finding(
                            case_study=case_study,
                            label=label,
                            artifact=name,
                            status="undeclared",
                            on_disk=on_disk,
                            pinned=pinned,
                            detail=conflicts[name].message,
                        )
                    )
                    continue
                # Not a conflict, which leaves three distinguishable states. Lumping them
                # under "current" claimed runs had been fitted on a file no run had touched,
                # and --quiet then hid it.
                retired = _declared_artifact_supersessions(con, artifact_name=name, sha256=on_disk)
                others = tuple(sha for sha in pinned if sha != on_disk)
                if on_disk not in pinned:
                    findings.append(
                        Finding(
                            case_study=case_study,
                            label=label,
                            artifact=name,
                            status="accepted_replacement",
                            on_disk=on_disk,
                            pinned=pinned,
                            detail=(
                                "no run for this label was fitted on the file now on disk; "
                                f"a declaration retires the {len(pinned)} vintage(s) that were, "
                                "so registration accepts the next run"
                            ),
                        )
                    )
                elif not others:
                    findings.append(
                        Finding(
                            case_study=case_study,
                            label=label,
                            artifact=name,
                            status="current",
                            on_disk=on_disk,
                            pinned=pinned,
                            detail="the runs registered for this label were fitted on this file",
                        )
                    )
                elif all(sha in retired for sha in others):
                    findings.append(
                        Finding(
                            case_study=case_study,
                            label=label,
                            artifact=name,
                            status="superseded",
                            on_disk=on_disk,
                            pinned=pinned,
                            detail=(
                                f"the registry holds {len(pinned)} vintages, and a declaration "
                                "retires every one the disk vintage replaced"
                            ),
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            case_study=case_study,
                            label=label,
                            artifact=name,
                            status="mixed",
                            on_disk=on_disk,
                            pinned=pinned,
                            detail=(
                                f"the registry holds {len(pinned)} vintages and nothing retires "
                                f"{', '.join(sha[:8] for sha in others if sha not in retired)}. "
                                "Registration stays silent because the file on disk is one of "
                                "them, so the mixture is already in the population"
                            ),
                        )
                    )
    finally:
        con.close()
    return findings


def check_all(*, artifacts_root: Path, only: str | None = None) -> list[Finding]:
    if not artifacts_root.is_dir():
        return []
    names = (
        [only]
        if only
        else sorted(p.name for p in artifacts_root.iterdir() if (p / "run_log").is_dir())
    )
    digests: dict[Path, str] = {}
    findings: list[Finding] = []
    for name in names:
        findings.extend(check_case_study(name, artifacts_root=artifacts_root, digests=digests))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case-study", default=None, help="check one instead of all")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=_default_artifacts_root(),
        help="where the per-case-study run_log/ directories live",
    )
    parser.add_argument("--json", action="store_true", help="emit findings as JSON")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the artifacts that are not on a vintage the registry carries",
    )
    args = parser.parse_args(argv)

    findings = check_all(artifacts_root=args.artifacts_root, only=args.case_study)

    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=1, default=list))
    else:
        for finding in findings:
            if args.quiet and finding.status == "current":
                continue
            head = f"{finding.status.upper():<11} {finding.case_study}/{finding.label}"
            print(f"{head}  {finding.artifact}")
            print(f"            {finding.detail}")
        counts = {
            status: sum(f.status == status for f in findings)
            for status in (
                "current",
                "superseded",
                "mixed",
                "accepted_replacement",
                "undeclared",
                "unresolved",
                "unchecked",
                "absent",
            )
        }
        print(
            f"{len(findings)} artifact/label pair(s): "
            + ", ".join(f"{n} {status}" for status, n in counts.items() if n)
        )

    return 1 if any(f.is_failure for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
