#!/usr/bin/env python3
"""Every committed supersedes literal still names a live generation.

A notebook that has published an official population and then moved a training identity has
to name the snapshot it replaces, or ``OfficialPopulation.create`` refuses the write. That
hash is committed source, so it goes stale whenever the registry tip moves without the
literal moving with it - a reset, a refit, a chain someone else advanced.

**Nothing that runs before a production chain can catch that.** A preview never resolves a
supersedes hash at all: ``supersedes_for_run`` returns ``None`` for any tier but
``canonical``, and that guard is correct - a preview is discarded with its workspace, has no
lineage to extend, and ``run_model_population`` refuses one that carries a hash. So a smoke
pass says nothing about the literals, and on 2026-09-06 twelve ``sp500_options`` notebooks
passed smoke and three failed the real chain fifteen minutes later on eleven stale literals.

CI cannot catch it either, and that is why this is a script rather than only a test:
``run_log/`` is gitignored and no registry is tracked, so a checkout has nothing to check
against. This runs where the registries live, before a chain is queued.

**It is keyed on the hash, not on the population name, and that is not incidental.** Only
some notebooks build their name as ``POPULATION_NAME or "a-literal"``; the rest use an
f-string over the label set or a module constant. A name-keyed scan cannot read those
statically - it skips them and reports green, which on the corpus this was written against
meant missing 8 of 17 literals including 3 of the 6 stale ones. Looking the declared hash up
in ``official_populations`` and resolving the lineage of whatever name owns it reads all of
them. Do not simplify this back to a name lookup.

Usage::

    python scripts/check_supersedes_literals.py                    # every case study
    python scripts/check_supersedes_literals.py --case-study etfs  # one
    python scripts/check_supersedes_literals.py --json             # machine-readable

**What a stale literal does and does not cost, because the difference decides what this
refuses.** ``OfficialPopulation.create`` reads the declared predecessor only when the member
list has moved: an unchanged re-run matches on members and returns the published population
without ever looking at it. So a stale literal is a latent fault, not a certain failure - it
fires on the run whose membership changes, which is the refit a chain is queued for, and
which is exactly the run that has already paid for its fit by the time it is refused. That
is why this refuses rather than warns, and why the refusal is waivable by name.

Exit status is 1 when a literal is dead AND the registry can name the head it should have
named. `unresolved` - the hash is in no lineage and the notebook's population name cannot be
read without executing it - is reported and never blocks: refusing there would be the check
asserting knowledge it does not have. A case study with no registry on disk is reported and
does not fail either; that is a reader's clone. ``--allow-stale-supersedes`` waives the
refusal for a run whose membership is known to be unchanged.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from case_studies.utils.registry.store import current_causal_identities  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Read with the AST rather than a pattern. A regex has to guess the annotation and the
# quoting, and both vary in the committed corpus: `cme_futures/09_dl_lstm.py` annotates
# `str | None`, and `fx_pairs/11_causal_dml.py` wraps its JSON in parentheses across two
# lines. A pattern that misses either reads the declaration as absent and reports nothing,
# which is the failure this script exists to prevent, arriving inside the script itself.
_POPULATION_NAME = "SUPERSEDES_POPULATION"
_CAUSAL_NAME = "SUPERSEDES_CAUSAL"


def _declared_literals(source: str) -> dict[str, str]:
    """The string value assigned to each SUPERSEDES_* name, whatever the assignment shape."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in (_POPULATION_NAME, _CAUSAL_NAME):
                found[target.id] = value.value
    return found


@dataclass
class Finding:
    case_study: str
    notebook: str
    declared: str
    status: str
    detail: str
    parameter: str = _POPULATION_NAME
    # The value to paste, where it can be determined: the current head, or "" for a genuine
    # first generation. None means it could not be worked out and the author has to look.
    remedy: str | None = None
    # The label a per-label causal declaration named it under. A repair has to replace that
    # entry in the mapping, not the whole declaration: `supersedes_for` rejects a bare hash
    # from a notebook that fits several labels.
    label: str | None = None

    @property
    def is_stale(self) -> bool:
        return self.status == "stale"


_POPULATION_NAME_ASSIGN = re.compile(
    r'^population_name\s*=\s*POPULATION_NAME\s*or\s*"([^"]+)"', re.M
)


def _population_name(source: str) -> str | None:
    """The default population name, where the notebook states one as a literal.

    Deliberately partial. Most notebooks write `POPULATION_NAME or "a-literal"`, but some
    build the name from an f-string over the label set or a module constant, and those
    cannot be read without executing the notebook. A caller must treat `None` as "unknown",
    never as "no generation exists" - reading it that way is how a name-keyed scan silently
    skips a third of the corpus.
    """
    match = _POPULATION_NAME_ASSIGN.search(source)
    return match.group(1) if match else None


def _registry_for(case_study: str, artifacts_root: Path) -> Path:
    return artifacts_root / case_study / "run_log" / "registry.db"


def _current_generation(db: sqlite3.Connection, name: str) -> tuple[str, str | None] | None:
    """The snapshot in force under *name*: the one nothing supersedes.

    The same rule as ``OfficialPopulation.one``. A fork - two snapshots nothing supersedes -
    has no defensible answer and is reported rather than picked between.
    """
    rows = db.execute(
        "SELECT population_hash, supersedes_hash FROM official_populations WHERE name = ?",
        (name,),
    ).fetchall()
    if not rows:
        return None
    superseded = {row[1] for row in rows if row[1] is not None}
    heads = [row for row in rows if row[0] not in superseded]
    if len(heads) != 1:
        raise ValueError(f"{len(heads)} current snapshots among {len(rows)} under {name!r}")
    return heads[0]


def _declared_causal_hashes(literal: str) -> list[tuple[str, str | None]]:
    """``(hash, label)`` for each hash a ``SUPERSEDES_CAUSAL`` declaration names.

    A per-label declaration names its label, which is what lets a stale one be answered with
    the identity to paste. A bare hash does not, and the label is only knowable from the run.
    """
    text = literal.strip()
    if not text:
        return []
    if text.startswith("{"):
        try:
            mapping = json.loads(text)
        except json.JSONDecodeError:
            # `supersedes_for` raises on this before the fit; nothing to resolve here.
            return []
        return [(str(v), str(k)) for k, v in mapping.items() if v]
    return [(text, None)]


def _causal_remedy(db: sqlite3.Connection, label: str | None) -> str | None:
    """The identity a stale causal declaration should name, where the label is known.

    Exactly one current identity is an answer worth printing. None, or more than one, is not,
    and guessing between them is how an author is sent to paste the wrong hash.
    """
    if not label:
        return None
    try:
        current = list(current_causal_identities(db, label=label))
    except sqlite3.OperationalError:
        return None
    return current[0] if len(current) == 1 else None


def _check_causal(case_study: str, notebook: Path, registry: Path, literal: str) -> list[Finding]:
    """Classify each hash a ``SUPERSEDES_CAUSAL`` declaration names.

    Only one verdict here is a failure, and the boundary is narrower than it first looks.
    ``causal_supersedes`` offers the hash when it is a CURRENT identity for the label and
    withholds it otherwise - and withholding is not automatically wrong. A notebook that
    already published its successor and was left unchanged still declares the predecessor;
    the hash is withheld, the runner resolves to the cached successor, and nothing fails.
    Telling that author to "fix" the literal would send them to declare a hash that already
    records theirs as its predecessor.

    What cannot be right in any of those readings is a hash the registry has never held.
    There is nothing to resolve to and nothing cached, so the write is refused after the DML
    fit and every placebo refit. That is the only causal status this script fails on.

    Currency comes from ``current_causal_identities``, the resolver itself, rather than from
    a supersedes-column scan here: it also excludes rows carrying an outdated identity
    version and rows from a non-canonical tier, and a reimplementation that ignored either
    would approve a hash the canonical notebook then withholds.
    """
    findings: list[Finding] = []
    for declared, declared_label in _declared_causal_hashes(literal):
        if not registry.exists():
            findings.append(
                Finding(
                    case_study,
                    notebook.name,
                    declared,
                    "no-registry",
                    "no registry on disk; a clone withholds this and publishes on its own",
                    _CAUSAL_NAME,
                )
            )
            continue
        db = sqlite3.connect(f"file:{registry}?mode=ro", uri=True)
        try:
            rows = db.execute(
                "SELECT label FROM causal_runs WHERE causal_hash = ?", (declared,)
            ).fetchall()
            if not rows:
                # Absent is not evidence on its own, exactly as on the population path: an
                # empty or freshly reset `causal_runs` withholds the hash and registration
                # accepts the first identity, so refusing here would block a valid first run.
                # It is wrong only when a current identity already exists for the label this
                # declaration names - and a bare declaration does not name one.
                current_for_label = (
                    set(current_causal_identities(db, label=declared_label))
                    if declared_label
                    else set()
                )
                if current_for_label:
                    findings.append(
                        Finding(
                            case_study,
                            notebook.name,
                            declared,
                            "stale",
                            f"no causal run with this hash exists, and {declared_label!r} "
                            f"already resolves to {sorted(current_for_label)[0]}; the write "
                            "is refused after the fit and every placebo refit",
                            _CAUSAL_NAME,
                            _causal_remedy(db, declared_label),
                            declared_label,
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            case_study,
                            notebook.name,
                            declared,
                            "unresolved",
                            "no causal run with this hash exists, and no current identity was "
                            "found for the label it is declared under - so it is either dead "
                            "or waiting for a first publication, and which cannot be told "
                            "from here",
                            _CAUSAL_NAME,
                            None,
                            declared_label,
                        )
                    )
                continue
            label = rows[0][0]
            current = set(current_causal_identities(db, label=label))
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            findings.append(
                Finding(case_study, notebook.name, declared, "no-registry", str(exc), _CAUSAL_NAME)
            )
            continue
        finally:
            db.close()

        if declared in current:
            status, detail = "live", f"a current causal identity for {label!r}"
        else:
            # Reported, not failed. See the docstring: an unchanged re-run after a
            # successful supersession looks exactly like this and is correct.
            status, detail = (
                "superseded",
                f"no longer current for {label!r}; correct for an unchanged re-run, which "
                "resolves to the cached successor, and wrong only if this run publishes a "
                "new identity",
            )
        findings.append(Finding(case_study, notebook.name, declared, status, detail, _CAUSAL_NAME))
    return findings


def check_case_study(case_study: str, *, repo_root: Path, artifacts_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    registry = _registry_for(case_study, artifacts_root)

    for notebook in sorted((repo_root / "case_studies" / case_study).glob("[0-9]*.py")):
        declared_by_name = _declared_literals(notebook.read_text(encoding="utf-8", errors="ignore"))
        causal = declared_by_name.get(_CAUSAL_NAME, "")
        if causal:
            findings.extend(_check_causal(case_study, notebook, registry, causal))

        declared = declared_by_name.get(_POPULATION_NAME, "")
        if not declared:
            continue

        if not registry.exists():
            findings.append(
                Finding(
                    case_study,
                    notebook.name,
                    declared,
                    "no-registry",
                    "no registry on disk; a clone publishes generation one and withholds this",
                )
            )
            continue

        db = sqlite3.connect(f"file:{registry}?mode=ro", uri=True)
        try:
            row = db.execute(
                "SELECT name FROM official_populations WHERE population_hash = ?", (declared,)
            ).fetchone()
            if row is None:
                # An absent hash is not evidence on its own. `population_supersedes` withholds
                # a hash it cannot place and `create` then publishes generation one, which is
                # exactly right on a reset or freshly initialised registry - so failing here
                # would block runs that succeed. The declaration is only wrong if a generation
                # already exists under the name THIS notebook publishes under, which needs the
                # name; where the name is an f-string or a module constant it cannot be read,
                # and the honest answer is that this one could not be resolved.
                statically_named = _population_name(
                    notebook.read_text(encoding="utf-8", errors="ignore")
                )
                head = None
                if statically_named:
                    try:
                        head = _current_generation(db, statically_named)
                    except ValueError as exc:
                        findings.append(
                            Finding(case_study, notebook.name, declared, "forked", str(exc))
                        )
                        continue
                if head is not None:
                    tip, tip_supersedes = head
                    findings.append(
                        Finding(
                            case_study,
                            notebook.name,
                            declared,
                            "stale",
                            f"{statically_named!r} is at {tip} (superseding {tip_supersedes}) "
                            "and this hash is in no lineage the registry holds",
                            _POPULATION_NAME,
                            tip,
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            case_study,
                            notebook.name,
                            declared,
                            "unresolved",
                            "this hash is in no lineage the registry holds, and no generation "
                            "was found under the name this notebook publishes under - so it "
                            "is either dead or waiting for a first publication, and which one "
                            "cannot be told from here",
                        )
                    )
                continue
            name = row[0]
            try:
                head = _current_generation(db, name)
            except ValueError as exc:
                findings.append(
                    Finding(
                        case_study, notebook.name, declared, "forked", str(exc), _POPULATION_NAME
                    )
                )
                continue
        except sqlite3.OperationalError as exc:
            # A registry that predates the table is the reader's case again, not a stale
            # literal. Anything else - a lock timeout, an I/O error - is not evidence.
            if "no such table" not in str(exc):
                raise
            findings.append(Finding(case_study, notebook.name, declared, "no-registry", f"{exc}"))
            continue
        finally:
            db.close()

        assert head is not None  # a hash we just found has at least its own row
        tip, tip_supersedes = head
        remedy = None
        if declared == tip:
            status, detail = "live", f"names the tip of {name!r}; a refit publishes over it"
        elif declared == tip_supersedes:
            status, detail = (
                "live",
                f"names what the tip of {name!r} replaced; a re-run resolves to it",
            )
        else:
            status, detail = (
                "stale",
                f"{name!r} is at {tip} (superseding {tip_supersedes}); this literal is neither",
            )
            remedy = tip
        findings.append(
            Finding(case_study, notebook.name, declared, status, detail, _POPULATION_NAME, remedy)
        )

    return findings


def check_all(*, repo_root: Path, artifacts_root: Path, only: str | None = None) -> list[Finding]:
    case_studies = (
        [only]
        if only
        else sorted(p.name for p in (repo_root / "case_studies").iterdir() if p.is_dir())
    )
    findings: list[Finding] = []
    for case_study in case_studies:
        if not (repo_root / "case_studies" / case_study).is_dir():
            continue
        findings.extend(
            check_case_study(case_study, repo_root=repo_root, artifacts_root=artifacts_root)
        )
    return findings


def _default_artifacts_root() -> Path:
    return Path.home() / "ml4t" / "artifacts" / "case_studies"


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
        "--allow-stale-supersedes",
        action="store_true",
        help=(
            "report stale literals without failing. For a run whose membership is unchanged, "
            "where the literal is never read. Deliberately not a general --force: one gate, "
            "named after what it waives."
        ),
    )
    args = parser.parse_args(argv)

    findings = check_all(
        repo_root=REPO_ROOT, artifacts_root=args.artifacts_root, only=args.case_study
    )

    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=1))
    else:
        for finding in findings:
            mark = "STALE" if finding.is_stale else finding.status.upper()
            print(f"{mark:<11} {finding.case_study}/{finding.notebook}  {finding.declared}")
            print(f"            {finding.detail}")
        checked = len(findings)
        stale = sum(f.is_stale for f in findings)
        print(f"\n{checked} declared literal(s); {stale} stale")

    stale = [f for f in findings if f.is_stale]
    unresolved = [f for f in findings if f.status == "unresolved"]

    for finding in unresolved:
        # Warned, never blocked. Refusing on "I could not resolve this" would be the check
        # asserting knowledge it does not have.
        print(
            f"\nUNRESOLVED case_studies/{finding.case_study}/{finding.notebook}\n"
            f"      {finding.parameter} = {finding.declared!r}\n"
            f"      {finding.detail}",
            file=sys.stderr,
        )

    if stale and not args.allow_stale_supersedes:
        # Addressed to the person about to queue a chain, because that is the only person for
        # whom this is free. Editing the literal makes the paired .ipynb stale, and the only
        # sanctioned way to commit that drops its outputs - a live render lost, unless a run
        # is coming anyway. Theirs is.
        print(
            "\nA stale literal does not make any committed output wrong, and it does not "
            "fail every run: an unchanged re-run returns the published population without "
            "ever reading it. It fails the run whose membership MOVES - which is the refit "
            "you are about to queue - after that run has paid for its fit:\n",
            file=sys.stderr,
        )
        for finding in stale:
            if finding.remedy and finding.label:
                # A per-label declaration is a mapping and the notebook fits several labels,
                # so `supersedes_for` rejects a bare hash. Replacing the whole declaration
                # with the remedy would break the run this message is trying to save.
                fix = (
                    f"replace the {finding.label!r} entry in the mapping with "
                    f'"{finding.remedy}", leaving the other entries as they are'
                )
            elif finding.remedy:
                fix = f'set it to "{finding.remedy}"'
            else:
                fix = "look up the current identity for the label this notebook fits"
            print(
                f"  case_studies/{finding.case_study}/{finding.notebook}\n"
                f"      {finding.parameter} = {finding.declared!r} is dead\n"
                f"      {finding.detail}\n"
                f"      fix: {fix}",
                file=sys.stderr,
            )
        print(
            "\nFix it now - you are about to pay for the run that re-renders the notebook "
            "you have to clear. If you know this run's membership is unchanged, so the "
            "literal is never read, pass --allow-stale-supersedes.",
            file=sys.stderr,
        )
        return 1
    if stale:
        print(
            f"\n{len(stale)} stale literal(s) allowed by --allow-stale-supersedes.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
