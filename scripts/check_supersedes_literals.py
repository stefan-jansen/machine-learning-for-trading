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

**Two lineages, and every declaration name.** A hash is looked up in ``official_populations``
and in ``candidate_set_names``, which have the same shape and the same resolver rule -
``candidate_set_supersedes`` offers a declared hash when it is the head or what the head
replaced, exactly as ``population_supersedes`` does. And every ``SUPERSEDES_*`` constant is
read, not a list of two: the corpus holds 24 distinct names, and the candidate-set ones hold a
``dict`` keyed by set name rather than a string.

**The declaration that is ABSENT is the one that costs a fit**, and no literal check can see
it. ``us_equities_panel/06_linear`` ran with ``SUPERSEDES_SETS = {}`` - nothing to classify, so
every literal passed - and was refused at registration after 78 minutes of cold fit and 19.2 h
of registered fit over 64 configs: ``a changed candidate set named
'us-equities-fwd-ret-1d-linear-v1' must explicitly supersedes 454f73021f33``. So the check runs
per LIVE GENERATION as well as per literal: a candidate set with a head that no declaration in
the case study names, by name or by hash, is reported with the hash to paste.

The condition there is a head existing at check time, never a name appearing in source.
Declaring nothing is correct until a generation exists to supersede - ``create`` refuses a
first version that claims to replace one - so a rule keyed on the source would have failed
``06_linear`` on the run where it was right. Reported by default and fatal only under
``--require-declarations``, because the declaration is resolved in the worktree at launch
rather than committed ahead of the run: a notebook on ``main`` is expected to name only what
it has already replaced.

Usage::

    python scripts/check_supersedes_literals.py                    # every case study
    python scripts/check_supersedes_literals.py --case-study etfs  # one
    python scripts/check_supersedes_literals.py --json             # machine-readable
    python scripts/check_supersedes_literals.py --case-study etfs --require-declarations

**A literal one generation behind is a refusal, not a pass.** The resolvers offer a declared
hash when it is the tip OR when it is what the tip replaced, and the second arm made this script
report "live: names what the tip of X replaced; a re-run resolves to it". That reading is true of
an unchanged re-run and false of every other run: ``create`` accepts the tip and nothing else,
so the run whose membership MOVES is refused at the freeze. On 2026-09-11 that was 19 of the 25
literals this script called live, including the two in ``us_equities_panel/06_linear`` that had
just cost 78 minutes of cold fitting - reproduced end to end against ``OfficialPopulation.create``
and ``CandidateSet.create`` in ``tests/test_supersedes_declares_intent.py``. They are reported
``behind`` and refuse exactly as ``stale`` does.

**The repair this prints is ``"live"``, never the current hash.** Pasting the tip is correct
until the next publish of whatever freezes that lineage, which is the loop these literals have
been in since #944. ``SUPERSEDES_LIVE`` names the lineage rather than quoting a generation of it
and is resolved against the registry at run time, so a declaration that carries it is reported
``intent`` and has nothing left to go stale - and it is right on a reader's clean clone, where
there is no generation and the resolvers withhold it.

**What a stale literal does and does not cost, because the difference decides what this
refuses.** ``OfficialPopulation.create`` reads the declared predecessor only when the member
list has moved: an unchanged re-run matches on members and returns the published population
without ever looking at it. So a stale literal is a latent fault, not a certain failure - it
fires on the run whose membership changes, which is the refit a chain is queued for, and
which is exactly the run that has already paid for its fit by the time it is refused. That
is why this refuses rather than warns, and why the refusal is waivable by name.

Exit status is 1 when a literal is refused at the freeze - dead, or one generation behind -
AND the registry can name the head it should have named. `unresolved` - the hash is in no lineage and the notebook's population name cannot be
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

from case_studies.research.population import SUPERSEDES_LIVE  # noqa: E402
from case_studies.utils.registry.store import current_causal_identities  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Read with the AST rather than a pattern. A regex has to guess the annotation and the
# quoting, and both vary in the committed corpus: `cme_futures/09_dl_lstm.py` annotates
# `str | None`, and `fx_pairs/11_causal_dml.py` wraps its JSON in parentheses across two
# lines. A pattern that misses either reads the declaration as absent and reports nothing,
# which is the failure this script exists to prevent, arriving inside the script itself.
_POPULATION_NAME = "SUPERSEDES_POPULATION"
_CAUSAL_NAME = "SUPERSEDES_CAUSAL"
_SUPERSEDES_PREFIX = "SUPERSEDES"

# Every `SUPERSEDES_*` constant, not the two that were named here first. The committed corpus
# uses 24 distinct names - `SUPERSEDES_SETS`, `SUPERSEDES_CANDIDATE_SETS`,
# `SUPERSEDES_MODEL_POPULATION`, `SUPERSEDES_COST_BACKTESTS` and 20 more - and reading two of
# them is how `us_equities_panel` reported "0 declared literal(s); 0 stale" on 2026-09-11 while
# `06_linear.py:124` declared two live ones. Keying on the prefix rather than a list is what
# keeps the 25th name from being invisible the day it is written.
#
# The value is a string for a single lineage and a `dict[name, hash]` where one notebook freezes
# several - and a dict is not an `ast.Constant`, so the old reader skipped every dict-valued
# declaration in the corpus whatever it was called.


def _declared_literals(source: str) -> dict[str, str | dict[str, str]]:
    """The value assigned to each ``SUPERSEDES*`` name, whatever the assignment shape."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    found: dict[str, str | dict[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or not target.id.startswith(_SUPERSEDES_PREFIX):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                # Built at runtime. Unreadable here and reported as such by the caller rather
                # than guessed at, which is the same answer `_population_name` gives.
                continue
            if isinstance(value, str) or (
                isinstance(value, dict)
                and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())
            ):
                found[target.id] = value
    return found


def _declared_pairs(value: str | dict[str, str]) -> list[tuple[str | None, str]]:
    """``(lineage name, hash)`` for a declaration, where a dict states its own names."""
    if isinstance(value, dict):
        return [(name, h) for name, h in value.items() if h]
    return [(None, value)] if value else []


# Every verdict this script can reach. Named once, because the corpus scan's allowlist was a
# second copy of it and drifted: `undeclared` was added in #944 and never reached the test, which
# skips wherever there is no registry - so the drift was invisible in CI by construction and the
# scan asserted a status set two releases old on the one machine that runs it.
STATUSES = (
    "live",  # the declaration names the tip; a refit publishes over it
    "intent",  # SUPERSEDES_LIVE: the lineage is named and the generation is looked up
    "behind",  # names the generation the tip replaced; refused on a membership move
    "stale",  # names no generation of a lineage that has one; refused
    "superseded",  # causal only: no longer current, which an unchanged re-run is entitled to
    "undeclared",  # a live generation no declaration in the case study names
    "unresolved",  # in no lineage, and nothing here can say whether that is dead or first
    "forked",  # two generations nothing supersedes; no defensible answer
    "no-registry",  # a reader's clone
)


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
    def refused_at_the_freeze(self) -> bool:
        """Whether ``create`` refuses this declaration on the run that moves its members.

        Two statuses do, and reading only ``stale`` is how 19 of the corpus's 25 ``live``
        verdicts were reported green while every one of them was a refusal. ``behind`` is the
        common one: the literal names the generation the tip replaced, which the resolver offers
        and ``create`` then rejects because it accepts the tip and nothing else.
        """
        return self.status in ("stale", "behind")


def _intent_finding(case_study: str, notebook: str, parameter: str, name: str | None) -> Finding:
    """A declaration that names the live generation instead of quoting a hash.

    Nothing to classify and nothing that can decay. It is reported rather than dropped because
    a name that appears nowhere in the output reads as a name nobody declared, which is the
    absence this script exists to make visible.
    """
    return Finding(
        case_study,
        notebook,
        SUPERSEDES_LIVE,
        "intent",
        f"names the live generation of {name!r} rather than a hash; nothing here goes stale"
        if name
        else "names the live generation of whatever lineage this notebook publishes under, "
        "rather than a hash; nothing here goes stale",
        parameter,
        label=name,
    )


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


# The two lineage tables a declared hash can live in. They have the same shape - a name, a
# hash, and the hash it supersedes - and `candidate_set_supersedes` makes the same decision
# about a candidate set that `population_supersedes` makes about a population: it offers the
# declared hash when it is the head or what the head replaced, and withholds it otherwise
# (`case_studies/research/comparison.py:112-123`). So one classifier serves both, and the
# only thing that differs is which table to read.
_LINEAGES: tuple[tuple[str, str], ...] = (
    ("official_populations", "population_hash"),
    ("candidate_set_names", "set_hash"),
)


def _current_generation(
    db: sqlite3.Connection,
    name: str,
    *,
    table: str = "official_populations",
    hash_column: str = "population_hash",
) -> tuple[str, str | None] | None:
    """The snapshot in force under *name*: the one nothing supersedes.

    The same rule as ``OfficialPopulation.one`` and ``CandidateSet.one``. A fork - two
    snapshots nothing supersedes - has no defensible answer and is reported rather than
    picked between.
    """
    rows = db.execute(
        f"SELECT {hash_column}, supersedes_hash FROM {table} WHERE name = ?",  # noqa: S608
        (name,),
    ).fetchall()
    if not rows:
        return None
    superseded = {row[1] for row in rows if row[1] is not None}
    heads = [row for row in rows if row[0] not in superseded]
    if len(heads) != 1:
        raise ValueError(f"{len(heads)} current snapshots among {len(rows)} under {name!r}")
    return heads[0]


def _owning_lineage(db: sqlite3.Connection, declared: str) -> tuple[str, str, str] | None:
    """``(table, hash column, name)`` for whichever lineage holds *declared*.

    Keyed on the hash rather than the name, which is the rule this whole script is built on:
    the declaration's variable name does not say which lineage it belongs to, and on the
    committed corpus the same file declares a population hash and a candidate-set hash under
    two different ``SUPERSEDES_*`` names.
    """
    for table, hash_column in _LINEAGES:
        try:
            row = db.execute(
                f"SELECT name FROM {table} WHERE {hash_column} = ?",  # noqa: S608
                (declared,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            continue
        if row is not None:
            return table, hash_column, row[0]
    return None


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


def _check_lineage_literal(
    case_study: str,
    notebook: Path,
    registry: Path,
    parameter: str,
    pairs: list[tuple[str | None, str]],
) -> list[Finding]:
    """Classify a declared hash against whichever lineage table holds it.

    This is the population classifier applied to every other ``SUPERSEDES_*`` name, and the
    verdicts are the same three because the resolver is the same: a hash that names the head
    publishes over it, a hash that names what the head replaced resolves to the head, and
    anything else is withheld and refused by ``create`` after the fit is paid for.

    A dict-valued declaration states the lineage name itself, which is strictly more than the
    population path can read: when the hash is in no lineage at all, the name still says
    whether a generation exists to supersede, so a dead hash can be answered with the head to
    paste instead of being reported as unresolved.
    """
    findings: list[Finding] = []
    if not registry.exists():
        return [
            Finding(
                case_study,
                notebook.name,
                declared,
                "no-registry",
                "no registry on disk; a clone publishes generation one and withholds this",
                parameter,
                label=name,
            )
            for name, declared in pairs
        ]

    db = sqlite3.connect(f"file:{registry}?mode=ro", uri=True)
    try:
        for name, declared in pairs:
            if declared == SUPERSEDES_LIVE:
                findings.append(_intent_finding(case_study, notebook.name, parameter, name))
                continue
            owner = _owning_lineage(db, declared)
            if owner is None:
                # Absent is not evidence on its own - a reset or a reader's empty registry
                # withholds the hash and `create` publishes generation one. It is wrong only
                # when a generation already exists under the name this declaration states.
                head = None
                if name:
                    for table, hash_column in _LINEAGES:
                        try:
                            head = _current_generation(
                                db, name, table=table, hash_column=hash_column
                            )
                        except sqlite3.OperationalError:
                            continue
                        except ValueError as exc:
                            findings.append(
                                Finding(
                                    case_study,
                                    notebook.name,
                                    declared,
                                    "forked",
                                    str(exc),
                                    parameter,
                                    label=name,
                                )
                            )
                            head = None
                            break
                        if head is not None:
                            break
                if head is not None:
                    tip, tip_supersedes = head
                    findings.append(
                        Finding(
                            case_study,
                            notebook.name,
                            declared,
                            "stale",
                            f"{name!r} is at {tip} (superseding {tip_supersedes}) and this "
                            "hash is in no lineage the registry holds",
                            parameter,
                            SUPERSEDES_LIVE,
                            name,
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
                            f"was found under {name!r}"
                            if name
                            else "this hash is in no lineage the registry holds, and the "
                            "declaration states no name to look one up under",
                            parameter,
                            label=name,
                        )
                    )
                continue

            table, hash_column, owner_name = owner
            try:
                head = _current_generation(db, owner_name, table=table, hash_column=hash_column)
            except ValueError as exc:
                findings.append(
                    Finding(
                        case_study,
                        notebook.name,
                        declared,
                        "forked",
                        str(exc),
                        parameter,
                        label=name,
                    )
                )
                continue
            assert head is not None  # a hash we just found has at least its own row
            tip, tip_supersedes = head
            remedy = None
            if declared == tip:
                status, detail = (
                    "live",
                    f"names the tip of {owner_name!r}; a refit publishes over it",
                )
            elif declared == tip_supersedes:
                status, detail = (
                    "behind",
                    f"{owner_name!r} is at {tip} and this names the generation {tip} replaced. "
                    "An unchanged re-run resolves to the tip and is fine; the run that MOVES "
                    "its members is refused at the freeze, because create accepts the tip and "
                    "nothing else",
                )
                remedy = SUPERSEDES_LIVE
            else:
                status, detail = (
                    "stale",
                    f"{owner_name!r} is at {tip} (superseding {tip_supersedes}); this literal "
                    "is neither",
                )
                remedy = SUPERSEDES_LIVE
            findings.append(
                Finding(
                    case_study, notebook.name, declared, status, detail, parameter, remedy, name
                )
            )
    finally:
        db.close()
    return findings


# A notebook that freezes a candidate set. Only these owe a declaration: a notebook that
# reads one with `CandidateSet.one` is handed whatever generation is in force and supersedes
# nothing.
_FREEZES = re.compile(r"candidate_set_supersedes|CandidateSet\.create|\.freeze\(")

# An f-string that could name a lineage, as a SQL LIKE pattern. `f"us-equities-{label}-linear-v1"`
# becomes `us-equities-%-linear-v1`, which is what lets a static reader enumerate the sets a
# notebook will freeze without executing it - the dict keys cannot do that, because the missing
# declaration is exactly the entry that is not in the dict.
_MIN_TEMPLATE_LITERAL = 8


def _freeze_templates(source: str) -> set[str]:
    """LIKE patterns for the lineage names *source* builds, where they can be read.

    Degenerate templates are dropped rather than guessed at. `f"{a}-{b}"` becomes `%-%` and
    matches every name in the table, which would attribute the whole registry to one notebook;
    so a pattern needs real literal text and must not open with a wildcard.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    patterns: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        pattern = ""
        literal_chars = 0
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                pattern += part.value.replace("%", r"\%").replace("_", r"\_")
                literal_chars += len(part.value)
            else:
                pattern += "%"
        if (
            literal_chars >= _MIN_TEMPLATE_LITERAL
            and "%" in pattern
            and not pattern.startswith("%")
        ):
            patterns.add(pattern)
    return patterns


def _undeclared_heads(case_study: str, registry: Path, notebooks: list[Path]) -> list[Finding]:
    """Live candidate-set generations that no declaration in this case study covers.

    This is the half that costs a fit, and it is not a stale-literal check. `us_equities_panel`
    on 2026-09-10 declared `SUPERSEDES_SETS = {}` - nothing to classify, so every literal check
    passes - and `06_linear` then ran 78 minutes of cold fit, 19.2 h of registered fit over 64
    configs, and was refused at registration with `a changed candidate set named
    'us-equities-fwd-ret-1d-linear-v1' must explicitly supersedes 454f73021f33`.

    The condition is a live head existing NOW, not a name appearing in source. A first
    generation must not be reported: before that run `fwd_ret_5d` and `fwd_ret_21d` had no
    generation and declaring nothing for them was correct, and `create` refuses a first version
    that claims to replace one. They have heads now, so the next run that moves their members
    is refused unless they are declared.

    Coverage is by name OR by hash, because a string-valued declaration names no set and only
    its hash can place it - the same reason this script keys on hashes everywhere else.
    """
    if not registry.exists():
        return []
    db = sqlite3.connect(f"file:{registry}?mode=ro", uri=True)
    try:
        try:
            rows = db.execute(
                "SELECT name, set_hash, supersedes_hash FROM candidate_set_names"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            return []

        by_name: dict[str, list[tuple[str, str | None]]] = {}
        for name, set_hash, supersedes in rows:
            by_name.setdefault(name, []).append((set_hash, supersedes))
        heads: dict[str, tuple[str, str | None]] = {}
        for name, generations in by_name.items():
            superseded = {s for _, s in generations if s}
            live = [g for g in generations if g[0] not in superseded]
            if len(live) == 1:
                heads[name] = live[0]

        declared_names: set[str] = set()
        declared_hashes: set[str] = set()
        owners: dict[str, list[str]] = {}
        for notebook in notebooks:
            source = notebook.read_text(encoding="utf-8", errors="ignore")
            for value in _declared_literals(source).values():
                for lineage_name, declared in _declared_pairs(value):
                    if lineage_name:
                        declared_names.add(lineage_name)
                    declared_hashes.add(declared)
            if not _FREEZES.search(source):
                continue
            for pattern in _freeze_templates(source):
                for name in heads:
                    if db.execute("SELECT ? LIKE ? ESCAPE '\\'", (name, pattern)).fetchone()[0]:
                        owners.setdefault(name, []).append(notebook.name)

        findings: list[Finding] = []
        for name, (tip, tip_supersedes) in sorted(heads.items()):
            if name in declared_names or tip in declared_hashes:
                continue
            if tip_supersedes and tip_supersedes in declared_hashes:
                continue
            attributed = sorted(set(owners.get(name, [])))
            # Attribution is an annotation, never the key. A name whose template no notebook
            # states - `cme_futures` builds all eleven of its names as plain literals - is
            # still a live head nothing declares, and reporting it under the case study is
            # more useful than dropping it because the owner could not be worked out.
            if len(attributed) == 1:
                notebook_name, where = attributed[0], f"{attributed[0]} freezes it"
            else:
                notebook_name = "-"
                where = (
                    f"frozen by one of {attributed}"
                    if attributed
                    else "no notebook states this name as a readable template"
                )
            findings.append(
                Finding(
                    case_study,
                    notebook_name,
                    tip,
                    "undeclared",
                    f"{name!r} is live at {tip} and no SUPERSEDES_* declaration in this case "
                    f"study names it or its hash; {where}. The next run that moves its "
                    "members is refused at the freeze, after the fit",
                    "SUPERSEDES_SETS",
                    SUPERSEDES_LIVE,
                    name,
                )
            )
        return findings
    finally:
        db.close()


def check_case_study(case_study: str, *, repo_root: Path, artifacts_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    registry = _registry_for(case_study, artifacts_root)
    notebooks = sorted((repo_root / "case_studies" / case_study).glob("[0-9]*.py"))
    findings.extend(_undeclared_heads(case_study, registry, notebooks))

    for notebook in notebooks:
        declared_by_name = _declared_literals(notebook.read_text(encoding="utf-8", errors="ignore"))
        causal = declared_by_name.get(_CAUSAL_NAME, "")
        if causal and isinstance(causal, str):
            findings.extend(_check_causal(case_study, notebook, registry, causal))

        # Every other `SUPERSEDES_*` name goes through the shared lineage classifier. The
        # population one keeps its own branch below because it can fall back on
        # `_population_name` to read the name out of the source when the hash is absent.
        for parameter, value in sorted(declared_by_name.items()):
            if parameter in (_CAUSAL_NAME, _POPULATION_NAME):
                continue
            pairs = _declared_pairs(value)
            if pairs:
                findings.extend(
                    _check_lineage_literal(case_study, notebook, registry, parameter, pairs)
                )

        declared = declared_by_name.get(_POPULATION_NAME, "")
        if not declared or not isinstance(declared, str):
            continue

        if declared == SUPERSEDES_LIVE:
            # The one declaration this script does not have to read a name for. Every other
            # population verdict needs `_population_name`, which is deliberately partial - a
            # name built from an f-string or a module constant cannot be read without executing
            # the notebook, and that is what leaves five of the corpus's literals unresolvable.
            # The sentinel is resolved at run time against the name the call site already holds,
            # so there is nothing here to look up and nothing that can drift.
            findings.append(_intent_finding(case_study, notebook.name, _POPULATION_NAME, None))
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
                            SUPERSEDES_LIVE,
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
                "behind",
                f"{name!r} is at {tip} and this names the generation {tip} replaced. An "
                "unchanged re-run resolves to the tip and is fine; the run that MOVES its "
                "members is refused at the freeze, because create accepts the tip and nothing "
                "else",
            )
            remedy = SUPERSEDES_LIVE
        else:
            status, detail = (
                "stale",
                f"{name!r} is at {tip} (superseding {tip_supersedes}); this literal is neither",
            )
            remedy = SUPERSEDES_LIVE
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
        "--require-declarations",
        action="store_true",
        help=(
            "exit non-zero when a live candidate-set generation is declared nowhere. Off by "
            "default and meant for the launch of the case study being queued: the declaration "
            "is resolved in the worktree at launch, not committed ahead of the run, so a "
            "notebook on main is expected to name only what it has already replaced."
        ),
    )
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
            print(
                f"{finding.status.upper():<11} {finding.case_study}/{finding.notebook}  "
                f"{finding.declared}"
            )
            print(f"            {finding.detail}")
        undeclared_count = sum(f.status == "undeclared" for f in findings)
        checked = len(findings) - undeclared_count
        refused = sum(f.refused_at_the_freeze for f in findings)
        behind = sum(f.status == "behind" for f in findings)
        print(
            f"\n{checked} declared literal(s); {refused} refused at the freeze "
            f"({behind} of them one generation behind); "
            f"{undeclared_count} live generation(s) declared nowhere"
        )

    stale = [f for f in findings if f.refused_at_the_freeze]
    unresolved = [f for f in findings if f.status == "unresolved"]
    undeclared = [f for f in findings if f.status == "undeclared"]

    if undeclared:
        # The expensive one. A stale literal at least tells the reader a lineage exists; an
        # absent declaration reads as "nothing to check" and every literal check passes, which
        # is how `us_equities_panel/06_linear` reached registration having already paid for
        # 19.2 h of registered fit.
        print(
            f"\n{len(undeclared)} live candidate-set generation(s) that no declaration names. "
            "Each is refused at the freeze - after the fit - on the next run that moves its "
            "members:\n",
            file=sys.stderr,
        )
        for finding in undeclared:
            print(
                f"  {finding.case_study}: {finding.label}\n"
                f"      {finding.detail}\n"
                f'      fix: declare "{finding.label}": "{finding.remedy}" in the freezing '
                "notebook's SUPERSEDES_* mapping, in the worktree you are about to launch",
                file=sys.stderr,
            )

    for finding in unresolved:
        # Warned, never blocked. Refusing on "I could not resolve this" would be the check
        # asserting knowledge it does not have.
        print(
            f"\nUNRESOLVED case_studies/{finding.case_study}/{finding.notebook}\n"
            f"      {finding.parameter} = {finding.declared!r}\n"
            f"      {finding.detail}",
            file=sys.stderr,
        )

    if undeclared and args.require_declarations:
        print(
            "\nRefusing because --require-declarations was passed.",
            file=sys.stderr,
        )
        return 1

    if stale and not args.allow_stale_supersedes:
        # Addressed to the person about to queue a chain, because that is the only person for
        # whom this is free. Editing the literal makes the paired .ipynb stale, and the only
        # sanctioned way to commit that drops its outputs - a live render lost, unless a run
        # is coming anyway. Theirs is.
        print(
            "\nA refused literal does not make any committed output wrong, and it does not "
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
            reading = "is dead" if finding.status == "stale" else "is one generation behind"
            print(
                f"  case_studies/{finding.case_study}/{finding.notebook}\n"
                f"      {finding.parameter} = {finding.declared!r} {reading}\n"
                f"      {finding.detail}\n"
                f"      fix: {fix}",
                file=sys.stderr,
            )
        print(
            '\nPaste "live" rather than the hash. A hash is a quotation of a value the '
            "registry moves on every publish, so pasting the current one buys you until the "
            'next run of whatever freezes it; "live" names the lineage instead and is '
            "resolved against the registry at run time, so it is correct for a clean clone "
            "too. Fix it now - you are about to pay for the run that re-renders the notebook "
            "you have to clear. If you know this run's membership is unchanged, so the "
            "literal is never read, pass --allow-stale-supersedes.",
            file=sys.stderr,
        )
        return 1
    if stale:
        print(
            f"\n{len(stale)} refused literal(s) allowed by --allow-stale-supersedes.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
