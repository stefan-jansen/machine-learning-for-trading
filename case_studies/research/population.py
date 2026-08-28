from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from case_studies.utils.registry.specs import canonical_json, compute_hash
from case_studies.utils.registry.store import _open_registry, _utc_now

from .contracts import ExecutionTier
from .results import Result

if TYPE_CHECKING:
    from .workspace import Study


def _connect(case_dir: Path) -> sqlite3.Connection:
    """Open one case study's registry for reading, with the timeouts every other reader uses.

    A registry is written by notebooks, backfills and scripts that run at the same time, so a
    read that takes SQLite's five-second default raises ``database is locked`` under nothing
    worse than ordinary contention. ``_open_registry`` sets 120s on the driver and 60s
    server-side for exactly that reason; a read here has the same contention and needs the same
    patience. It cannot call ``_open_registry`` itself, which runs the schema DDL and would
    create the tables a clean clone is being asked about.
    """
    db = sqlite3.connect(str(case_dir / "run_log" / "registry.db"), timeout=120.0)
    db.execute("PRAGMA busy_timeout = 60000")
    return db


def research_name(case_study_id: str, suffix: str, *, scope: str = "") -> str:
    """Name one published artifact, isolated as a whole chain when a run is narrowed.

    A run that narrows the catalog must not publish under the canonical name, because a
    population is immutable per name and a partial snapshot would be frozen as the real
    one. Isolating only what a stage *writes* is not enough. A later stage that resolves
    its upstream by the canonical name reads the full population, computes over it, and
    freezes that under the isolated name it was given - so the name says narrowed and the
    contents are not, which is worse than either being wrong on its own. In a workspace
    that does not hold the canonical populations the same run raises instead, so the
    defect is invisible exactly where the populations already exist.

    Both the writing stage and the reading stage call this with the same scope, so one
    knob isolates every name in the chain. With no scope the result is byte-identical to
    the canonical name, which is what lets a narrowed run be configured without moving
    any published population.

    A scope names one run of the chain, not one notebook in it. Giving each stage its own
    scope isolates each stage from every other stage as well as from the canonical names,
    so the second stage looks for an upstream population under a name the first stage
    never wrote. Every notebook in one narrowed run takes the same scope.
    """
    return f"{scope}:{suffix}" if scope else f"{case_study_id}:{suffix}"


def _preview_is_active(study: Study) -> bool:
    """Whether this study is a preview, asked of the study rather than of the environment.

    The tier is a property of how the study was opened, and `Study` holds it. Every earlier
    version of this read `ML4T_OUTPUT_DIR` instead, which `Study.activate` stamps and never
    clears, and that answers a different question: "is a preview active anywhere in this
    process", not "is this study a preview". The two diverge whenever one process holds two
    studies, and they diverged in both directions.

    Reading the environment unscoped withheld the declared hash from a canonical study that
    merely ran second, after any preview had been opened. The notebook then refits everything
    and dies at registration for naming no predecessor - the expense the declaration exists to
    prevent. Scoping the read to the study's own output root fixed that and opened the reverse:
    a canonical study sharing that output root and activating later cleared the `.preview`
    stamp, so the preview read as canonical and had its run refused.

    Neither reading could reach the third case at all. `open_study` returns an isolated study
    through `Study.open` when the case study's generated directories are not symlinks, which is
    every CI checkout and every clean clone, and `Study.open` activated as canonical - so an
    isolated preview was stamped canonical and `_refuse_preview_activation` never ran on the
    one path CI exercises. No amount of scoping an environment read reaches that, because the
    marker it would scope was never written.

    The field is not the whole answer, because `activate` takes a tier per call: every model
    adapter opens one study and activates whichever tier the run asked for, so a study opened
    canonical can be writing as a preview right now. The field answers what the study was
    opened for; the active output root answers what it is writing as. It is a preview if
    either says so, and the root is compared against this study's own preview directory so a
    second study's activation cannot answer for this one.
    """
    if study.execution_tier is ExecutionTier.PREVIEW:
        return True
    active = os.environ.get("ML4T_OUTPUT_DIR")
    if not active or study.output_root is None:
        return False
    return Path(active).resolve() == (Path(study.output_root) / ".preview").resolve()


def _refuse_preview_activation(study: Study) -> None:
    """A population is written to the canonical registry whatever tier is active.

    That is correct - a population is canonical by definition - but it means a preview run that
    reaches this code writes into the shared registry rather than its own workspace. The member
    check below cannot catch it: a population is snapshotted *before* its members are fitted, so
    every lookup misses and every member passes. On 2026-08-16 a preview notebook test left a
    28-member population in the canonical etfs registry that way, and the next canonical run was
    refused because a population of that name already existed with different members.

    Callers guard this too. This is the guard that does not depend on remembering.
    """
    if _preview_is_active(study):
        raise ValueError(
            "a preview run cannot create an official population: it would be written to the "
            "canonical registry"
        )


@dataclass(frozen=True)
class OfficialPopulation:
    study: Study
    hash: str
    name: str
    member_kind: str
    members: tuple[str, ...]
    supersedes: str | None

    @classmethod
    def create(
        cls,
        study: Study,
        *,
        name: str,
        member_kind: str,
        members: list[str] | tuple[str, ...],
        supersedes: str | None = None,
    ) -> OfficialPopulation:
        study.require_writable()
        _refuse_preview_activation(study)
        if member_kind not in {"training", "prediction", "backtest"}:
            raise ValueError("official population member_kind is not supported")
        normalized = tuple(dict.fromkeys(str(member) for member in members))
        if not normalized or len(normalized) != len(members):
            raise ValueError("official population members must be non-empty and unique")
        for member_hash in normalized:
            try:
                result = Result.open(study, member_hash, include_preview=True)
            except KeyError:
                continue
            if result.kind != member_kind:
                raise ValueError(
                    f"official population member {member_hash} has kind {result.kind}, "
                    f"not {member_kind}"
                )
            if result.execution_tier == "preview":
                raise ValueError(
                    f"preview result {member_hash} cannot enter an official population"
                )
            if (
                member_kind == "backtest"
                and (result.spec().get("decision_artifact") or {}).get("canonical") is False
            ):
                raise ValueError(
                    f"exploratory decision backtest {member_hash} cannot enter an official population"
                )
        snapshot = {
            "schema_version": 1,
            "name": name,
            "member_kind": member_kind,
            "members": list(normalized),
            "supersedes": supersedes,
        }
        population_hash = compute_hash(canonical_json(snapshot))
        db = _open_registry(study.root)
        try:
            latest = db.execute(
                "SELECT population_hash, snapshot_json FROM official_populations WHERE name = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (name,),
            ).fetchone()
            if latest is not None:
                if latest[0] == population_hash:
                    return cls.open(study, population_hash)
                # A population is the list of identities published under a name; `supersedes`
                # is lineage about that list and is stored in its own column beside it. Because
                # it also sits inside the hashed snapshot, a second generation's hash depends on
                # which generation it replaced - so the notebook that produced it, re-run
                # unchanged, computes a different hash from the same members and reads as a
                # change. Every notebook publishing a second generation declares
                # `SUPERSEDES_POPULATION = ""` and states in prose that it is the first, which
                # is what that failure looks like from the reader's side. Matching on the
                # members makes reproducing the published list a no-op, whatever the caller
                # says it supersedes, and leaves the guard below to the case where the list
                # genuinely differs.
                published = json.loads(latest[1])
                same_list = published.get("members") == list(normalized)
                if same_list and published.get("member_kind") == member_kind:
                    return cls.open(study, latest[0])
                if supersedes != latest[0]:
                    raise ValueError(
                        f"a changed population named {name!r} must explicitly supersedes "
                        f"{latest[0]}"
                    )
            elif supersedes is not None:
                raise ValueError("first population version cannot supersede another snapshot")
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO official_populations "
                "(population_hash, name, member_kind, snapshot_json, supersedes_hash, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    population_hash,
                    name,
                    member_kind,
                    canonical_json(snapshot),
                    supersedes,
                    _utc_now(),
                ),
            )
            db.executemany(
                "INSERT INTO official_population_members "
                "(population_hash, member_hash, ordinal) VALUES (?,?,?)",
                [
                    (population_hash, member_hash, ordinal)
                    for ordinal, member_hash in enumerate(normalized)
                ],
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return cls(study, population_hash, name, member_kind, normalized, supersedes)

    @classmethod
    def one(cls, study: Study, *, name: str) -> OfficialPopulation:
        """Resolve the current immutable population by name, without a hash handoff.

        A name accumulates a snapshot per generation: refitting the same configurations under a
        corrected estimator parameter produces different prediction identities, so the second run
        supersedes the first rather than replacing it. Every earlier snapshot stays readable by
        hash, which is what makes the lineage worth writing down, but a caller asking by name
        wants the generation in force. That is the one snapshot in the chain that nothing
        supersedes; two of those under a single name means the chain forked and no answer is
        defensible.
        """
        with _connect(study.root) as db:
            rows = db.execute(
                "SELECT population_hash, supersedes_hash FROM official_populations "
                "WHERE name = ? ORDER BY population_hash",
                (name,),
            ).fetchall()
        superseded = {row[1] for row in rows if row[1] is not None}
        current = [row[0] for row in rows if row[0] not in superseded]
        if len(current) != 1:
            raise ValueError(
                f"official population name {name!r} resolved to {len(current)} current "
                f"identities among {len(rows)} snapshots"
            )
        return cls.open(study, current[0])

    @classmethod
    def open(cls, study: Study, population_hash: str) -> OfficialPopulation:
        with _connect(study.root) as db:
            row = db.execute(
                "SELECT name, member_kind, snapshot_json, supersedes_hash "
                "FROM official_populations WHERE population_hash = ?",
                (population_hash,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown official population {population_hash!r}")
        snapshot = json.loads(row[2])
        return cls(
            study,
            population_hash,
            row[0],
            row[1],
            tuple(snapshot["members"]),
            row[3],
        )

    def require_complete(self) -> tuple[str, ...]:
        failures = []
        for member_hash in self.members:
            try:
                result = Result.open(self.study, member_hash)
            except KeyError:
                failures.append(f"{member_hash}:missing")
                continue
            if result.kind != self.member_kind:
                failures.append(f"{member_hash}:kind={result.kind}")
            elif not result.complete:
                failures.append(f"{member_hash}:partial")
        if failures:
            raise ValueError(
                f"official population {self.hash} is incomplete: {', '.join(failures)}"
            )
        return self.members


def population_supersedes(study: Study, *, name: str, declared: str | None) -> str | None:
    """Decide whether a declared population hash may be offered to :meth:`OfficialPopulation.create`.

    A notebook that has published a population and then moved a training identity has to name the
    snapshot it replaces, or ``create`` refuses the write. The hash it names is committed source,
    so the same declaration has to be right in three situations the notebook cannot tell apart on
    its own, and offering it unconditionally is wrong in two of them.

    - **A clean clone.** ``run_log/`` is gitignored, so a reader starts with an empty registry -
      often with no ``official_populations`` table at all, which raises ``OperationalError``
      rather than ``ValueError``. ``create`` refuses a first version that claims to supersede
      something, so the declared hash must be withheld and the reader's run publishes generation
      one. This is the ordinary case for anyone who is not the author.
    - **The re-run.** The generation in force is the one this declaration produced, which is
      ``current.supersedes == declared``. Offering the hash recomputes the same snapshot, so the
      notebook resolves to the population it published instead of writing a new one.
    - **The refit.** The declaration names the tip itself, ``current.hash == declared``, and
      offering it publishes the next generation over that tip.

    Anything else is withheld, and ``create`` then refuses and names the hash it requires - a
    better answer than this function guessing. Note that the two matching conditions are both
    needed: testing only the first withholds the hash from an author holding generation one who
    declares it in order to publish generation two, and testing only whether any generation exists
    gets the second run on a clean clone wrong, because run 1 writes a generation whose own
    ``supersedes`` is ``None`` and offering the hash again there writes a generation nobody asked
    for.

    A narrowed run passes here too and needs no special case: a caller-chosen ``POPULATION_NAME``
    has no prior generation, so the lookup fails and the hash is withheld. So does a preview,
    whose isolated registry holds no generation under this name either.
    """
    if not declared:
        return None
    if _preview_is_active(study):
        # Decided before the registry is consulted, because in a maintainer worktree the
        # registry a preview reads is the canonical one. Asking it first returns a real
        # generation, this function offers the hash, and `run_model_population` then refuses
        # the run outright - "preview populations cannot supersede a snapshot" - before the
        # first fit. A preview never creates an official population, so there is never a
        # snapshot for it to supersede and the answer does not depend on what is on disk.
        return None
    try:
        current = OfficialPopulation.one(study, name=name)
    except (ValueError, KeyError):
        # No generation under this name: `one` raises `ValueError` when the query returns no
        # current snapshot, and `open` raises `KeyError` for a hash the table does not hold.
        return None
    except sqlite3.OperationalError as error:
        if "no such table" not in str(error):
            # A lock timeout, an I/O error and a half-migrated schema are not evidence that
            # nothing has been published. Swallowing them withholds the hash, and the notebook
            # then pays for a full refit before `create` refuses the write for naming no
            # predecessor. Same rule as `_lineage`.
            raise
        return None
    return declared if declared in (current.supersedes, current.hash) else None


def _lineage(
    case_dir: Path, member_kind: str
) -> tuple[list[tuple[str, str, str | None]], dict[str, set[str]]]:
    """One registry's population lineage: its generations, and what each one lists.

    Opened with the timeouts ``_open_registry`` applies, because concurrent writers are the
    expected case for a registry and a momentary lock would otherwise raise instantly - and the
    caller turns any failure here into "nothing is retired", which is the silent wrong answer
    this whole module exists to prevent. Opening through ``_open_registry`` itself would create
    the database and its schema as a side effect, which is wrong for a read against a release
    root that may legitimately not exist.

    A missing file or a missing table means no generation was ever written and is answered with
    nothing, which is the ordinary state of a reader's clean clone. Every other
    ``OperationalError`` propagates: a lock timeout, an I/O error and a half-migrated schema are
    not evidence that nothing has been retired.
    """
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return [], {}
    db = _connect(case_dir)
    try:
        try:
            rows = db.execute(
                "SELECT population_hash, name, supersedes_hash FROM official_populations "
                "WHERE member_kind = ?",
                (member_kind,),
            ).fetchall()
        except sqlite3.OperationalError as error:
            if "no such table" not in str(error):
                raise
            return [], {}
        members: dict[str, set[str]] = {}
        for population_hash, member_hash in db.execute(
            "SELECT population_hash, member_hash FROM official_population_members"
        ):
            members.setdefault(population_hash, set()).add(member_hash)
    finally:
        db.close()
    return rows, members


def superseded_members(study: Study, *, member_kind: str = "prediction") -> frozenset[str]:
    """Identities whose own publisher has moved past them.

    A downstream stage asks the registry which results it should consume, and the obvious
    answer - every row that is ``complete`` and whose ``identity_status`` is ``"current"`` -
    does not answer it. ``identity_status`` is derived from ``identity_version``
    (``catalog.py``), which is the schema number the row was written under. It says the
    registry still understands the row. It says nothing about whether the row is the one its
    producer publishes, because that is a property of the population lineage and is recorded
    in a different table.

    The two agree until a model notebook refits. Then the notebook publishes a second
    generation under the same name, the first generation's members stay in the registry -
    complete, and current under a schema version that has not moved - and a stage selecting on
    the catalog alone sweeps both. It does not fail: it succeeds over twice the population,
    freezes the retired generation into whatever it publishes, and everything downstream
    inherits a set that mixes two answers to the same question. That is worse than a refusal,
    because nothing about the run looks wrong.

    So ask the lineage instead. **The question is asked per name, and the name is the whole
    point.** A name is one publisher's answer to one question, and its chain of generations is
    that publisher changing its mind; a member is retired when the name that published it has
    moved past it. Within a name, the comparison is member-wise rather than whole-generation:
    a refit that moves three of ten identities retires three, and the seven that did not move
    are still what that name publishes.

    Asking it globally instead - "retired by someone, and listed by nobody in force" - is the
    version that looks equivalent and is not, because the same identity is legitimately listed
    under several names. A narrowed or preview run freezes its own snapshot of whatever the
    catalog held on the day it ran, and that snapshot stays in force under its own name
    forever. Measured on ``fx_pairs`` 2026-08-25: refitting ``tabular_dl`` retired 72
    prediction sets, and ``fx_pairs:preflight-baselines``, frozen the day before, still listed
    all 72 - so the global form returned nothing retired and the sweep would have run over both
    generations exactly as if the check were absent. A stale snapshot under an unrelated name
    cannot un-retire another name's superseded generation.

    Both registries are read, in the order :class:`PredictionCatalog` overlays them. A workspace
    study offers released rows the workspace registry does not hold, and their lineage lives in
    the released registry - which ``Study.open`` never copies. Reading only ``study.root`` there
    returns nothing retired, because the workspace's ``official_populations`` table is created
    schema-complete and empty, and the filter is a no-op again by a different route. A population
    hash is content-addressed, so the same generation seen in both is the same generation and the
    merge is a union rather than a precedence rule.

    That union covers the ``supersedes`` edges too, and it has to. The two roots disagree in the
    ordinary case rather than the exotic one: an author who copies a workspace registry into the
    release root and then refits leaves the release root holding generation A as an unsuperseded
    tip while the workspace holds A -> B. **A stale root is not independent evidence that A is
    still published.** It is an older copy of the same content-addressed chain, and supersession
    is monotone - an edge is only ever added, never retracted - so a hash superseded in either
    root is superseded. Deciding tip-ness per root and keeping any root's tip alive instead lets
    the lagging root veto every retirement it knows a name for, which is the state right after
    every release, and the catalog goes on overlaying generation A's rows into that workspace.
    """
    rows: list[tuple[str, str, str | None]] = []
    members: dict[str, set[str]] = {}
    roots = [study.release_case_root]
    if not study.read_only and study.root != study.release_case_root:
        roots.append(study.root)
    seen_populations: set[str] = set()
    for root in roots:
        root_rows, root_members = _lineage(root, member_kind)
        for row in root_rows:
            if row[0] in seen_populations:
                continue
            seen_populations.add(row[0])
            rows.append(row)
        for population_hash, member_hashes in root_members.items():
            members.setdefault(population_hash, set()).update(member_hashes)
    return _retired(rows, members)


def _retired(
    rows: list[tuple[str, str, str | None]], members: dict[str, set[str]]
) -> frozenset[str]:
    """Reduce one or more registries' lineage to the members their publishers have moved past.

    Split out from :func:`superseded_members` so the root-based caller reaches the same
    reduction rather than a second copy of it. The per-name, member-wise comparison below is
    the part that is easy to restate wrongly - the module docstring above says why the global
    form and the whole-generation form both look equivalent and are not - so there is exactly
    one implementation of it.
    """
    if not any(row[2] is not None for row in rows):
        return frozenset()

    by_name: dict[str, list[tuple[str, str | None]]] = {}
    for population_hash, name, supersedes in rows:
        by_name.setdefault(name, []).append((population_hash, supersedes))

    retired: set[str] = set()
    for generations in by_name.values():
        superseded = {supersedes for _, supersedes in generations if supersedes is not None}
        if not superseded:
            continue
        # The generations this name still stands behind. Normally one; a forked chain leaves
        # more, and taking the union is the conservative reading - a member any surviving tip
        # still lists is not retired, so a fork can only under-report.
        in_force: set[str] = set()
        for population_hash, _ in generations:
            if population_hash not in superseded:
                in_force |= members.get(population_hash, set())
        for population_hash in superseded:
            retired |= members.get(population_hash, set()) - in_force
    return frozenset(retired)


def published_population_names_at(
    case_dir: str | Path, *, member_kind: str = "prediction"
) -> frozenset[str]:
    """Every population name this registry has ever published, in force or superseded.

    The question a reader needs before it can interpret a name that will not resolve. An empty
    answer means the registry does not declare populations at all - a fixture, or a registry
    written before the mechanism existed - and a comparison there rests on catalog
    admissibility, which is a weaker claim but a statable one. A non-empty answer means the
    registry does declare them, and a name that resolves to nothing in *that* registry is a
    broken lineage rather than an absent mechanism.

    Answering both with "the name did not resolve" is what makes the two indistinguishable, and
    they call for opposite responses: proceed and say so, or refuse.
    """
    rows, _ = _lineage(Path(case_dir), member_kind)
    return frozenset(name for _, name, _ in rows)


def superseded_members_at(
    case_dir: str | Path, *, member_kind: str = "prediction"
) -> frozenset[str]:
    """:func:`superseded_members` for a case directory the caller has already resolved.

    A downstream notebook that read its rows through ``prediction_rows_at`` has deliberately
    not opened a ``Study``: every ``Study.open`` branch ends in ``activate()``, which re-points
    the rest of the notebook at whichever registry the activation selected. Asking the lineage
    question through a ``Study`` here would reintroduce exactly that, and would answer for a
    different registry than the rows being filtered - so the retired set and the catalog would
    disagree by construction and the join would be meaningless.

    One root, because the caller named it. :func:`superseded_members` unions the release and
    workspace roots because a ``Study`` legitimately overlays both and a hash superseded in
    either is superseded; a caller who resolved a single directory is asking about that
    directory's registry, which is the same one ``prediction_rows_at`` reads.
    """
    rows, members = _lineage(Path(case_dir), member_kind)
    return _retired(rows, members)
