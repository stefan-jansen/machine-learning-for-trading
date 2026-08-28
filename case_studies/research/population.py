from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from case_studies.utils.registry.specs import canonical_json, compute_hash
from case_studies.utils.registry.store import _open_registry, _utc_now

from .results import Result

if TYPE_CHECKING:
    from .workspace import Study


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


def _refuse_preview_activation() -> None:
    """A population is written to the canonical registry whatever tier is active.

    That is correct - a population is canonical by definition - but it means a preview run that
    reaches this code writes into the shared registry rather than its own workspace. The member
    check below cannot catch it: a population is snapshotted *before* its members are fitted, so
    every lookup misses and every member passes. On 2026-08-16 a preview notebook test left a
    28-member population in the canonical etfs registry that way, and the next canonical run was
    refused because a population of that name already existed with different members.

    Callers guard this too. This is the guard that does not depend on remembering.
    """
    import os
    from pathlib import Path

    active = os.environ.get("ML4T_OUTPUT_DIR")
    if active and Path(active).name == ".preview":
        raise ValueError(
            "a preview run cannot create an official population: it would be written to the "
            "canonical registry"
        )


def _registry_roots(study: Study) -> list[Path]:
    """The registries a read consults, released first.

    ``Study.open`` never copies the released ``run_log`` into a workspace, and the prediction
    catalog overlays the two, so a read that consults ``study.root`` alone answers for a
    different set of rows than the one a notebook goes on to filter.
    """
    roots = [study.release_case_root]
    if not study.read_only and study.root != study.release_case_root:
        roots.append(study.root)
    return roots


def _rows(root: Path, query: str, params: tuple) -> list[tuple]:
    """Run one query against a registry root, treating an absent table as no rows.

    A clean clone has no ``official_populations`` at all - ``run_log/`` is gitignored. Any
    other operational error is a real fault and is not swallowed.
    """
    db_path = root / "run_log" / "registry.db"
    if not db_path.is_file():
        return []
    with sqlite3.connect(db_path) as db:
        try:
            return db.execute(query, params).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            return []


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
        _refuse_preview_activation()
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
        seen: dict[str, str | None] = {}
        for root in _registry_roots(study):
            for population_hash, supersedes in _rows(
                root,
                "SELECT population_hash, supersedes_hash FROM official_populations "
                "WHERE name = ? ORDER BY population_hash",
                (name,),
            ):
                # Content-addressed, so the same hash seen in both roots is the same snapshot.
                # Tip-ness is decided over the union rather than per root, for the reason
                # `superseded_members` gives: a lagging root would otherwise veto every
                # retirement it knows a name for, which is the state right after every release.
                seen.setdefault(population_hash, supersedes)
        rows = sorted(seen.items())
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
        row = None
        for root in _registry_roots(study):
            found = _rows(
                root,
                "SELECT name, member_kind, snapshot_json, supersedes_hash "
                "FROM official_populations WHERE population_hash = ?",
                (population_hash,),
            )
            if found:
                row = found[0]
                break
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
    try:
        current = OfficialPopulation.one(study, name=name)
    except (ValueError, sqlite3.OperationalError, KeyError):
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
    db = sqlite3.connect(str(db_path), timeout=120.0)
    try:
        db.execute("PRAGMA busy_timeout = 60000")
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
    roots = _registry_roots(study)
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


def current_prediction_members(study: Study, *, verify_members: bool = True) -> frozenset[str]:
    """Every prediction identity the case study currently publishes, and no others.

    Six notebooks in a case study need the same set, and building it in each of them got the
    same subtlety wrong six times. Two steps are required and neither is sufficient alone.

    The first is the union over names of what each name publishes now, through
    :meth:`OfficialPopulation.one`, which resolves the one generation in a name's chain that
    nothing supersedes and refuses rather than guessing if the chain has forked. Building the
    set from ``official_population_members`` directly instead would sweep every generation.

    The second is subtracting :func:`superseded_members`, and the union alone is not enough
    because a name is not the only thing that lists an identity. A narrowed or preview run
    freezes its own snapshot of whatever the catalog held that day, and that snapshot stays in
    force under its own name forever - so a member its own publisher has since retired is
    still listed by the frozen name, and the union puts it back. Retirement is a statement by
    the name that published the member; another name's stale snapshot does not answer it.

    ``member_kind`` is filtered rather than assumed: the column exists because a population
    can hold something other than predictions, and a backtest population's members are not
    prediction hashes.

    ``verify_members`` controls :meth:`OfficialPopulation.require_complete`, which asks a
    different question - whether each published member's artifact is on disk and complete -
    and is separable from which identities are published. Every notebook wants it, so it is
    on by default; a caller that only needs the set, on a clean clone whose ``run_log/`` is
    gitignored, turns it off rather than getting an error about artifacts it never asked for.

    Both roots are read, in the order :func:`superseded_members` reads them, and for the same
    reason: ``Study.open`` never copies the released ``run_log`` into a workspace, so a
    workspace study whose names came from ``study.root`` alone would publish a set the
    prediction catalog then overlays released rows into - subtracting retirements computed
    across both roots from a union computed from one.
    """
    return frozenset().union(
        *current_prediction_populations(study, verify_members=verify_members).values()
    )


def current_prediction_populations(
    study: Study, *, verify_members: bool = True
) -> dict[str, frozenset[str]]:
    """The same set as :func:`current_prediction_members`, kept split by publishing name.

    The union is what a stage selecting across the whole case study wants, and it is what
    :func:`current_prediction_members` returns. A stage reporting on one family wants the
    names apart, because "every member this population publishes reached the catalog" is a
    question only a name can answer: a member absent from the catalog cannot be attributed to
    a family on its own, but the population that lists it can be, from the members that did
    arrive.

    Retirement is subtracted per name rather than left to the caller, for the reason given
    above: a frozen snapshot lists members its own publisher has since retired, so a per-name
    view that skipped the subtraction would hand them back one name at a time.

    The tips are resolved before retirement is read, and the order is not arbitrary. A writer
    publishing a successor between the two reads is the expected case - a registry is written
    by notebooks and backfills that run at the same time - and reading retirement first pairs a
    stale retirement set with fresh tips, which is the one combination that admits a member the
    successor retired. The other order pairs fresh retirement with stale tips, which can only
    exclude a member that is still published: a later run sees it again, and nothing downstream
    freezes a generation its publisher has moved past.
    """
    names = sorted(
        {
            row[0]
            for root in _registry_roots(study)
            for row in _rows(
                root,
                "SELECT DISTINCT name FROM official_populations WHERE member_kind = ?",
                ("prediction",),
            )
        }
    )
    tips: dict[str, OfficialPopulation] = {}
    for name in names:
        population = OfficialPopulation.one(study, name=name)
        if verify_members:
            population.require_complete()
        tips[name] = population
    retired = superseded_members(study, member_kind="prediction")
    return {name: frozenset(tip.members) - retired for name, tip in tips.items()}
