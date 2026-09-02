"""The field a holdout selection is made over, built one way for every reader of it.

Five notebooks in a case study need the same answer to "what may this selection choose
from": the stage that freezes the field, and the four downstream stages that read the
frozen set - or, in a clean clone that has no ``run_log``, rebuild it live. Each of them
had its own copy of the construction, and the copies disagreed in two ways that changed
published numbers:

- **The freeze spanned every declared label and the fallbacks spanned one.** So a case
  study whose selection is a variant label got that label from the frozen set and the
  primary from the fallback, and which one a reader saw depended on whether
  ``candidate_sets`` existed in their registry.
- **The label used downstream was the primary, not the winner's.** The selection ranged
  over every label while the window loading, schedule thinning and analysis that followed
  it were all keyed to ``labels.primary``. A 10-day or classification winner was evaluated
  under a 5-day regression contract, and the rendered notebook reported one label while
  selecting another.

Both are closed by having one construction and reading the label off the selection.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from .comparison import CandidateSet
from .configs import sweep_labels
from .population import OfficialPopulation

if TYPE_CHECKING:
    from .results import Result
    from .workspace import Study

#: The validation stages that compete for the holdout. A baseline may win outright wherever
#: neither allocation nor risk earned its complexity, so all three are in the field.
FIELD_STAGES: tuple[str, ...] = ("signal", "allocation", "risk_overlay")

#: The one stage every declared label has to reach. The baseline is the equal-weight backtest
#: every label is measured at, so it is where a label being absent means a run is unfinished.
#: Reaching it is a floor, not a completion check - what says the baseline sweep finished is
#: its plan, the same as every stage after it.
COVERAGE_STAGE: str = "signal"


@dataclass(frozen=True)
class SelectionField:
    """What a holdout selection may choose from, and what it chose."""

    name: str
    members: tuple[str, ...]
    selected: Result
    label: str
    source: str
    candidate_set: CandidateSet | None

    @property
    def frozen(self) -> bool:
        """Whether this came from a recorded candidate set rather than a live ranking."""
        return self.candidate_set is not None


def _recorded_set_count(registry: Any, name: str) -> int:
    """How many generations of ``name`` this registry records, 0 where it records none.

    Asked of the database rather than inferred from an exception. ``CandidateSet.one``
    raises ``ValueError`` for two unrelated conditions - the name resolves to no
    unsuperseded set, and it resolves to several - and only the first means "this registry
    has no frozen selection". Catching both would send an AMBIGUOUS set, which is a refit
    that left two generations live and needs a person to say which supersedes which,
    silently down the live-ranking path.
    """
    try:
        with sqlite3.connect(f"file:{registry}?mode=ro", uri=True) as db:
            return int(
                db.execute(
                    "SELECT COUNT(*) FROM candidate_sets WHERE name = ?", (name,)
                ).fetchone()[0]
            )
    except sqlite3.OperationalError:
        # No `candidate_sets` table: a registry that predates them, or a reader's clean clone.
        return 0


#: The population a sweep publishes its planned backtests under, per stage.
#:
#: The baseline is here too. It was left out on the reading that coverage - one rankable row
#: per declared label - is enough to say it finished, and that is wrong twice. The baseline is
#: a grid, predictions by entry scheme, so an interruption leaves the same shape every other
#: interrupted sweep leaves: the stage present, each surviving prediction fully represented,
#: and nothing to compare the count against. And a grid that has since changed - an entry
#: scheme withdrawn, the prediction population refitted - leaves rows that are still rankable,
#: so they stay eligible and can win a selection no current plan contains. One rankable row is
#: evidence the sweep started, and coverage is still asked for exactly that; it is not evidence
#: the sweep finished, and only the plan answers that.
PLAN_STAGE_KEYS: Mapping[str, str] = {
    "signal": "baseline",
    "allocation": "allocation",
    "risk_overlay": "risk",
}


def predictions_identity(prediction_hashes: Collection[str] | None) -> str:
    """A short, stable name for the set of prediction sets in force.

    The plan names below carry it, which is what makes "has this sweep run against the
    predictions in force" an equality rather than an inference. Every inference that was tried
    instead answered one direction and missed the other: comparing a plan's members against the
    current predictions catches a prediction the refit removed, and cannot see one it added,
    because the backtests that would ride the new prediction do not exist until the sweep runs.
    A digest of the whole set moves on either.

    ``None`` means the case study declares no prediction populations, so there is no generation
    for a plan to be behind, and every plan under that name is asked completeness only.
    """
    if prediction_hashes is None:
        return "unrestricted"
    return members_digest(prediction_hashes)


def members_digest(hashes: Collection[str]) -> str:
    """A short, stable name for a set of registered identities, order-independent."""
    return hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest()[:12]


def sweep_plan_name(case_study: str, label: str, stage: str, predictions: str) -> str:
    """The official population a sweep publishes its planned backtests under.

    Owned here rather than built at each call site. The freeze and the four notebooks that
    rebuild the field live have to name the same populations, and a convention spelled out in
    five places is a convention that can differ in one of them.

    ``predictions`` is :func:`predictions_identity` of the populations the sweep planned
    against, so the name identifies one (label, stage, predictions) triple. A plan that is
    absent under the current name is a sweep that has not run against the current predictions -
    whether they were added to, removed from, or replaced wholesale - and the freeze declines
    for the same reason it declines a sweep that never ran at all.
    """
    return f"{case_study}-{PLAN_STAGE_KEYS[stage]}-{label}-{predictions}"


#: Which stages a stage's grid is derived from, upstream first.
#:
#: The allocation grid is the baseline sweep's leading configurations by allocator and
#: concentration; the risk grid is one carrier chosen out of the baseline and allocation rows.
#: So a change to an upstream grid is a change to what the downstream grid should have been,
#: and the identity below carries it.
UPSTREAM_STAGES: Mapping[str, tuple[str, ...]] = {
    "signal": (),
    "allocation": ("signal",),
    "risk_overlay": ("signal", "allocation"),
}


def sweep_generation(plan_hash: str, upstream: Sequence[str] = ()) -> str:
    """Which grid, derived from which upstream grids, a recorded attempt was an attempt at.

    The plan name carries the predictions it was planned against, and not the grid: a sweep
    whose declared configurations change - an allocator added, a schedule withdrawn - supersedes
    the plan under the same name. Attempts numbered by plan name alone would then span the two
    grids, and the previous grid's successful attempt would be the latest one on record for a
    grid that has not run. Naming the attempts after the generation makes each one start its own
    numbering, so an unexecuted new grid has no attempts rather than an inherited one.

    **The plan's own identity, not a digest of its members.** A grid that changes from A to B
    and back to A publishes a third population, and its hash differs from the first A's because
    ``supersedes`` is inside the hashed snapshot - so the returning grid starts fresh numbering
    rather than inheriting the attempts the original A left. A member digest cannot tell the two
    A generations apart, and an interruption between publishing the plan and opening its attempt
    would then let the original A's attestation stand for the new one.

    **``upstream`` is the identities of the plans this grid was derived from.** A downstream plan
    is complete and attested against the grid that was upstream when it ran, and neither says
    anything once that upstream grid is superseded: the allocation plan built on ten advancing
    configurations stays complete after the baseline sweep re-runs and advances a different ten.
    Folding the upstream identities in makes the downstream attestation belong to one upstream
    generation, so superseding an upstream plan leaves the dependent grid unattested until its
    own sweep re-runs.
    """
    return members_digest((plan_hash, *upstream))


def sweep_attempt_name(plan_name: str, generation: str, attempt: int) -> str:
    """The population a sweep publishes when it starts executing its plan, once per attempt."""
    return f"{plan_name}-g{generation}-attempt-{attempt}"


def sweep_attestation_name(plan_name: str, generation: str, attempt: int) -> str:
    """The population a sweep publishes when that attempt finished with no failure.

    The plan alone cannot say this. It is published *before* the sweep runs, so that an
    interruption leaves a plan that is visibly short rather than a previous generation still
    in force, and completeness is then asked of its members. But a member can fail to execute
    and still leave the plan complete: where the failure is a registered artifact that
    disagrees with the prediction window in force, the row is a registered, internally whole
    backtest, and ``require_complete`` is answering a question about the member rather than
    about the run. The sweep raises on such a failure, which stops *that* process and tells a
    later reader nothing - and the freeze is a later reader, in another notebook and usually
    another process.

    **The attestation is per attempt, and the reader requires the latest attempt to carry
    one.** A single attestation keyed on the plan would be indelible: a sweep that succeeded
    once, then failed on a re-run of the identical grid - which is exactly the stale-artifact
    case - would leave the first run's attestation standing, and the freeze would accept the
    failed attempt on the strength of it. Numbering the attempts makes "did the last run of
    this sweep finish" answerable, which is the actual question. Each name is used once, so no
    attempt or attestation ever needs a supersedes declaration.
    """
    return f"{plan_name}-g{generation}-swept-{attempt}"


def _attempts(names: Collection[str], plan_name: str, generation: str) -> list[int]:
    """The attempt numbers this registry records for this generation of ``plan_name``."""
    prefix = f"{plan_name}-g{generation}-attempt-"
    found = []
    for name in names:
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                found.append(int(suffix))
    return sorted(found)


def open_sweep_attempt(study: Study, plan: OfficialPopulation, upstream: Sequence[str] = ()) -> int:
    """Record that this run is about to execute ``plan``, and return the attempt number.

    Called immediately after the plan is published and before any member executes, so that a
    run which dies part-way leaves an attempt with no attestation behind it. The members are
    the plan's own, sorted, so the record is addressable and carries what was attempted.

    ``upstream`` is what :func:`upstream_plan_hashes` returned for this sweep, and the caller
    has it because it had to require those plans before it could rank anything.
    """
    names = _population_names(study) or set()
    generation = sweep_generation(plan.hash, upstream)
    attempt = (max(_attempts(names, plan.name, generation), default=0)) + 1
    OfficialPopulation.create(
        study,
        name=sweep_attempt_name(plan.name, generation, attempt),
        member_kind="backtest",
        members=sorted(set(plan.members)),
    )
    return attempt


def attest_sweep(
    study: Study, plan: OfficialPopulation, attempt: int, upstream: Sequence[str] = ()
) -> OfficialPopulation:
    """Record that attempt ``attempt`` of ``plan`` executed in full, with no failure.

    Called by a sweep notebook after it has raised on any failure, so reaching it is the
    statement being recorded. One call rather than a name built at each site, for the reason
    :func:`sweep_plan_name` is owned here: three notebooks publish plans and the freeze reads
    all three, and a convention spelled out in four places can differ in one of them.
    """
    return OfficialPopulation.create(
        study,
        name=sweep_attestation_name(plan.name, sweep_generation(plan.hash, upstream), attempt),
        member_kind="backtest",
        members=sorted(set(plan.members)),
    )


def _attested(names: set[str] | None, plan: OfficialPopulation, upstream: Sequence[str]) -> bool:
    """Whether the latest recorded attempt at this plan's grid finished without a failure.

    An absent attempt is not a success. Plans recorded before attempts existed have none, and
    their sweeps are exactly the runs whose outcome was never written down. The question is
    asked of one generation of the grid, derived from one generation of the grids upstream of
    it: see :func:`sweep_generation`.
    """
    if names is None:
        return False
    generation = sweep_generation(plan.hash, upstream)
    attempts = _attempts(names, plan.name, generation)
    if not attempts:
        return False
    return sweep_attestation_name(plan.name, generation, attempts[-1]) in names


def _population_names(study: Study) -> set[str] | None:
    """Every official population name this registry records, or ``None`` where it has no table.

    ``None`` is the registry that predates official populations, which is the one state in
    which an absent plan is not evidence of anything. Any other failure to read propagates:
    a lock timeout is not a case study that publishes no plans.
    """
    # The same timeouts every other reader of this registry uses. The field construction now
    # makes this read once per (label, stage) rather than once per label, so a sweep writing in
    # another process is that much more likely to be holding the lock when it happens, and
    # SQLite's five-second default would surface as a reconstruction that aborts rather than
    # waits. A timeout that does expire still propagates: a locked registry is not a case study
    # that publishes no plans.
    try:
        with sqlite3.connect(
            f"file:{study.root / 'run_log' / 'registry.db'}?mode=ro", uri=True, timeout=120.0
        ) as db:
            db.execute("PRAGMA busy_timeout = 60000")
            return {row[0] for row in db.execute("SELECT name FROM official_populations")}
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return None
        raise


def publishes_sweep_plans(study: Study, case_study: str) -> bool:
    """Whether this registry records any sweep plan for this case study, under any generation.

    What separates a case study whose sweeps publish plans from one whose sweeps predate them.
    For the first, a plan absent under the name the current predictions imply means the sweep
    has not run against them, and the field cannot be built. For the second there is nothing to
    look up and the stages are admitted whole, which is the field those case studies were
    published with.

    Asked of the whole name space rather than declared, because a case study that has published
    a plan cannot un-publish one: the old generations stay readable by name forever.
    """
    names = _population_names(study)
    if names is None:
        return False
    prefixes = tuple(f"{case_study}-{key}-" for key in PLAN_STAGE_KEYS.values())
    return any(name.startswith(prefixes) for name in names)


def _upstream_hashes(
    study: Study, case_study: str, label: str, stage: str, predictions: str
) -> tuple[str, ...]:
    """The identities of the plans this stage's grid was derived from, upstream first.

    Requiring them is what makes a downstream sweep unable to rank an upstream sweep that has
    not finished. That is not hypothetical: this case study's allocation and risk plans were
    once published while its baseline sweep was still running, and three of the ten
    configurations they advanced were not the ten the finished baseline gives.

    Reads through :func:`_current_plan`, so each upstream plan is required complete and
    attested against its own upstream in turn, and the recursion ends at the baseline, which
    has none.

    An upstream plan that is absent contributes nothing rather than raising here. Whether a
    case study is allowed to have one absent is a question about the field, and
    :func:`planned_backtests` already answers it at every stage the field is built from - and
    at the two stages a sweep notebook requires before it ranks. Raising here as well would
    make a single stage unaskable in isolation, which is how ``unfinished_sweep_plans`` reports
    per stage.
    """
    hashes: list[str] = []
    for upstream in UPSTREAM_STAGES.get(stage, ()):
        plan = _current_plan(study, case_study, label, upstream, predictions)
        if plan is not None:
            hashes.append(plan.hash)
    return tuple(hashes)


def _current_plan(
    study: Study, case_study: str, label: str, stage: str, predictions: str
) -> OfficialPopulation | None:
    """The plan in force for this stage, complete and attested, or ``None`` where none is recorded.

    Absent is distinct from empty: ``create`` refuses an empty member list, so a recorded plan
    always admits something. It is also distinct from unreadable and from ambiguous - a name
    resolving to two current identities is a forked lineage that needs a person, and reading
    either of those as absence would drop the membership filter and admit historical rows.

    A recorded plan is required to be complete before its members are handed back. Plans are
    published *before* their sweep executes, so a current-but-incomplete plan is exactly what
    an interruption leaves, and filtering to it would rank the part of the grid that finished.
    The freeze asks the same question through :func:`unfinished_sweep_plans` and declines; a
    live rebuild has to reach the same answer, or a reader rebuilding the field mid-sweep gets
    a selection the freeze would have refused to make.

    Completeness is necessary and not sufficient, so the attestation is required too. See
    :func:`sweep_attestation_name`: a member that failed against a registered artifact from a
    different prediction window leaves the plan complete, and only the run knows it failed.
    """
    name = sweep_plan_name(case_study, label, stage, predictions)
    names = _population_names(study)
    if names is None or name not in names:
        return None
    plan = OfficialPopulation.one(study, name=name)
    plan.require_complete()
    upstream = _upstream_hashes(study, case_study, label, stage, predictions)
    if not _attested(names, plan, upstream):
        raise ValueError(
            f"sweep plan {name} is complete but records no attestation for the grid it now "
            "describes, so the run that filled it either reported failures, did not finish, "
            "or ran against an upstream grid that has since been superseded; re-run that sweep"
        )
    return plan


def _plan_members(
    study: Study, case_study: str, label: str, stage: str, predictions: str
) -> set[str] | None:
    """The backtests the plan in force admits, or ``None`` where no plan is recorded."""
    plan = _current_plan(study, case_study, label, stage, predictions)
    return None if plan is None else set(plan.members)


def upstream_plan_hashes(
    study: Study,
    *,
    case_study: str,
    label: str,
    stage: str,
    prediction_hashes: Collection[str] | None,
) -> tuple[str, ...]:
    """What a sweep notebook passes to :func:`open_sweep_attempt` and :func:`attest_sweep`.

    Calling it is also the requirement: it raises unless every upstream sweep this grid is
    derived from has published a plan against the prediction sets in force, filled it, and
    attested it. A sweep that ranks before calling it is ranking rows that no current upstream
    plan need contain.
    """
    return _upstream_hashes(
        study, case_study, label, stage, predictions_identity(prediction_hashes)
    )


def planned_backtests(
    study: Study,
    *,
    case_study: str,
    label: str,
    stage: str,
    prediction_hashes: Collection[str] | None,
) -> set[str] | None:
    """The backtests a stage's plan admits, for a caller that ranks that stage's rows.

    ``None`` where this case study records no plan for any stage, which is how the case studies
    that predate plans keep the field they were published with. Where it records plans and this
    one is absent, the sweep has not run against the prediction sets in force and ranking its
    historical rows would rank another generation's grid, so it raises.
    """
    predictions = predictions_identity(prediction_hashes)
    members = _plan_members(study, case_study, label, stage, predictions)
    if members is None and publishes_sweep_plans(study, case_study):
        raise RuntimeError(
            f"{case_study} publishes sweep plans, and none is recorded for {label} "
            f"at the {stage!r} stage against the prediction sets in force ({predictions}). "
            "That sweep has not been run against them, so the rows this stage does have "
            "belong to another generation. Run the sweep for this label."
        )
    return members


def unfinished_sweep_plans(
    study: Study,
    *,
    case_study: str,
    labels: Sequence[str],
    prediction_hashes: Collection[str] | None = None,
    stages: Sequence[str] = tuple(PLAN_STAGE_KEYS),
) -> list[str]:
    """Which of the declared labels' sweep plans are absent or not fully executed, one line each.

    A sweep notebook computes every backtest identity it intends to register before it
    registers any of them, and publishes that list as an official population. This asks each
    plan whether every member is present and complete, and that is the only question about a
    sweep's completion the registry can actually answer.

    It is asked because none of the alternatives work. An interrupted sweep leaves rows that
    look exactly like a smaller finished one: the stage is present, the surviving model
    configurations are each fully represented, and the row count is simply lower with nothing
    to compare it against. Stage presence, row counts and configuration counts were each tried
    and each accepts an interruption as complete. The plan is the comparison they lacked.

    **Every declared label is asked, not a subset inferred from the rows.** An earlier version
    waited only on labels that already had rows past their baseline, so that whichever label
    ran first could seal the field while the rest had not started - the two states are
    identical in the registry, and a set frozen in the first one locks the others out
    permanently. A label whose sweep is genuinely not intended does not exist here: the
    allocation notebook raises rather than advancing fewer configurations than it declared, so
    there is no path that stops a label at its baseline. Should one ever be added, it has to
    publish that decision, because nothing in the rows distinguishes it from an unstarted run.

    **Which predictions a plan ran against is in its name, not inferred from its members.** A
    plan supersedes only when its own sweep re-runs, so after a refit the previous generation
    would otherwise still be the plan in force under a fixed name - and still complete, because
    its members are still registered. Two inferences were tried and each answered one direction:
    asking whether the members ride predictions still in force catches a prediction the refit
    removed, and cannot see one it added, because the backtests that would ride a new prediction
    do not exist until the sweep runs. The name carries :func:`predictions_identity` instead, so
    a sweep that has not run against the predictions in force is simply absent under the name
    being looked up, and absent is a case this already handles.

    Absent and incomplete are reported the same way on purpose. A plan that was never written
    is a sweep that either has not run or ran before plans were recorded, and neither is a
    sweep whose results may be sealed into an immutable set.

    ``OperationalError`` is caught alongside the rest because a registry that predates official
    populations has no table to query, which is an absent plan and not a broken one.
    """
    predictions = predictions_identity(prediction_hashes)
    names = _population_names(study)
    unfinished: list[str] = []
    for label in labels:
        for stage in stages:
            name = sweep_plan_name(case_study, label, stage, predictions)
            try:
                plan = OfficialPopulation.one(study, name=name)
                plan.require_complete()
                upstream = _upstream_hashes(study, case_study, label, stage, predictions)
                if not _attested(names, plan, upstream):
                    raise ValueError(
                        "complete, but no attestation for the grid it now describes - the run "
                        "that filled it reported failures, did not finish, or ran against an "
                        "upstream grid that has since been superseded"
                    )
            except (KeyError, ValueError, sqlite3.OperationalError) as exc:
                unfinished.append(f"{label} {stage} ({name}): {exc}")
    return unfinished


def _complete_only(study: Study, rows: pl.DataFrame, *, planned: bool) -> pl.DataFrame:
    """The rows whose backtest is registered whole, refusing rather than ranking around the rest.

    Completeness used to be applied after the field was built, by the live ranking alone. That
    put it after the coverage check and after the freeze had already accepted the row, so a
    label whose only baseline was half-written counted towards coverage and then vanished from
    the ranking - and a reader's clean clone, which takes the live path, could select a
    configuration the frozen path had refused. ``CandidateSet.create`` refuses partial members,
    so the freeze reaches the same answer; the two paths now reach it at the same point.

    Where a plan decided the membership, incomplete is a hard stop: the plan was published
    before its sweep ran and required complete on the way in, so a member that is not complete
    now means the registry no longer holds what the plan says it does, and ranking the rest
    would publish a grid the sweep never finished. Where no plan did - the case studies that
    predate them - the member is dropped, which is what the live ranking always did with it;
    what changes is that it is dropped before the coverage tally rather than after.
    """
    if rows.is_empty():
        return rows
    incomplete: list[str] = []
    for member_hash in rows["backtest_hash"].to_list():
        reason = study.results.open(member_hash).completeness()
        if reason is not None:
            incomplete.append(f"{member_hash} ({reason})")
    if not incomplete:
        return rows
    if planned:
        raise RuntimeError(
            "a sweep plan admits validation backtests that are registered but not complete, "
            "so the registry no longer holds the grid the plan describes and the field cannot "
            "be built until that sweep is re-run: " + "; ".join(incomplete[:5])
        )
    dropped = {entry.split(" ", 1)[0] for entry in incomplete}
    return rows.filter(~pl.col("backtest_hash").is_in(list(dropped)))


def resolve_field_members(
    study: Study,
    *,
    case_study: str,
    prediction_hashes: Any,
    resolve_best_backtest_runs: Any,
    stages: tuple[str, ...] = FIELD_STAGES,
    coverage_stage: str = COVERAGE_STAGE,
) -> pl.DataFrame:
    """Every eligible validation backtest across the declared labels and the field's stages.

    **Every declared label is backtested equal-weight**, and that is what makes the labels
    comparable, so a declared label with no baseline rows means the run is unfinished. The
    set is immutable under its name, so freezing there publishes a field nothing can add to.

    **Whether a stage is finished is not readable from its rows, the baseline included.** Each
    sweep plans a grid - predictions by entry scheme, configurations by top-k by allocator, one
    carrier by risk control - and an interrupted run leaves rows that look like a smaller
    finished grid from every angle: the stage is present, and each surviving configuration is
    fully represented. Presence, row counts and configuration counts were all tried and each of
    them accepts an interruption as complete. Coverage catches a label with no baseline at all
    and says nothing about a baseline that stopped part-way.

    What answers it is the plan itself. The sweep notebooks compute every expected backtest
    identity before executing, and publish that list as an official population, so completion
    is ``require_complete`` on a recorded plan rather than an inference from what happens to be
    in the registry. That check belongs where the field is frozen, since it is the freeze that
    is irreversible; this function builds the field and the caller decides whether it may be
    sealed.

    **Where a plan is recorded, it also decides membership, and it has to be complete.**
    Checking a plan only for completeness left it decorative: a superseded grid - an entry
    scheme withdrawn, an allocator withdrawn, a top-k level dropped - leaves rows whose
    predictions are still current, so they stayed eligible and could win the selection even
    though no current plan contains them. Restricting each stage to its plan's members makes
    the published field the grid the sweep actually declared, and requiring the plan complete
    keeps a rebuild run mid-sweep from ranking the half of the grid that finished. A stage with
    no recorded plan is admitted whole, which is how the case studies that predate plans keep
    the field they were published with - and that is decided by whether this registry has ever
    recorded a plan for this case study, not by whether this one happens to be missing. Where it
    has, an absent plan is a sweep that has not run against the predictions in force, and
    building the field anyway would hand every reader who rebuilds it live a different
    membership from the one that was frozen. It raises instead.

    **A plan is asked about the grid it describes now, not the grid it described when it ran.**
    Its attestation is named after a generation that carries the identities of the plans
    upstream of it, so a baseline sweep that re-runs and advances a different ten configurations
    leaves the allocation and risk plans built on the previous ten unattested until their own
    sweeps re-run. Without that they stay complete and attested indefinitely, which is the state
    this case study was actually in: its allocation and risk plans were published an hour before
    the baseline sweep that feeds them finished.

    Rows with no Sharpe are dropped before any of this. They are ineligible by
    construction, since the selection ranks on validation backtest Sharpe, and leaving them
    in makes them count towards coverage and then fails the whole frozen set later, when
    ``best_validation_sharpe`` rejects it for holding a member it cannot rank. Members that are
    registered but not complete are settled at the same point, by :func:`_complete_only`, and
    for the same reason: both used to be handled after the coverage tally, so a label whose only
    baseline was unrankable or half-written satisfied coverage and then left the field.
    """
    labels = sweep_labels(study)
    frames: list[pl.DataFrame] = []
    reached: dict[str, set[str]] = {label: set() for label in labels}
    for label in labels:
        for stage in stages:
            rows = resolve_best_backtest_runs(
                case_study,
                label,
                split="validation",
                stage=stage,
                top_n=9999,
                prediction_hashes=prediction_hashes,
            )
            if "sharpe" in rows.columns:
                rows = rows.filter(pl.col("sharpe").is_not_null())
            admitted: set[str] | None = None
            if stage in PLAN_STAGE_KEYS:
                admitted = planned_backtests(
                    study,
                    case_study=case_study,
                    label=label,
                    stage=stage,
                    prediction_hashes=prediction_hashes,
                )
                if admitted is not None:
                    rows = rows.filter(pl.col("backtest_hash").is_in(list(admitted)))
            rows = _complete_only(study, rows, planned=admitted is not None)
            if rows.is_empty():
                continue
            reached[label].add(stage)
            frames.append(rows)

    uncovered = [label for label in labels if coverage_stage not in reached[label]]
    if uncovered:
        raise RuntimeError(
            f"the holdout field cannot be frozen while these declared labels have no rankable "
            f"validation backtests at the {coverage_stage!r} stage: {uncovered}. Every declared "
            "label is backtested equal-weight, so an absent one means the run is unfinished, and "
            "the set is immutable under its name, so no later run could add it."
        )

    if not frames:
        raise RuntimeError(
            "the holdout field cannot be frozen: no declared label has a rankable validation "
            "backtest at any stage."
        )
    return pl.concat(frames, how="diagonal_relaxed").unique("backtest_hash")


def label_of(study: Study, result: Result) -> str:
    """The label a backtest was fitted against, read from its training run.

    The selection ranges over every declared label, so the label the stages after it must
    use is a property of what won - never ``labels.primary``, which is only the winner's
    label by coincidence.
    """
    registry = study.root / "run_log" / "registry.db"
    with sqlite3.connect(f"file:{registry}?mode=ro", uri=True) as db:
        row = db.execute(
            """
            SELECT t.label
            FROM backtest_runs b
            JOIN prediction_sets p ON p.prediction_hash = b.prediction_hash
            JOIN training_runs t ON t.training_hash = p.training_hash
            WHERE b.backtest_hash = ?
            """,
            (result.hash,),
        ).fetchone()
    if row is None or not row[0]:
        raise RuntimeError(
            f"backtest {result.hash} has no label in this registry, so the stages after the "
            "selection have no label contract to run under"
        )
    return str(row[0])


def open_selection_field(
    study: Study,
    *,
    case_study: str,
    name: str,
    prediction_hashes: Any,
    resolve_best_backtest_runs: Any,
    stages: tuple[str, ...] = FIELD_STAGES,
) -> SelectionField:
    """The frozen field where one is recorded, the same field rebuilt live where none is.

    Reading the recorded set is the stronger path: it is immutable, so it cannot follow an
    upstream change. Rebuilding is the same rule applied live and cannot notice that
    something moved. Which one ran is on the returned object, and both now span every
    declared label - previously only the frozen path did, so a reader's clean clone could
    select a different configuration from the published one.
    """
    registry = study.root / "run_log" / "registry.db"
    if _recorded_set_count(registry, name):
        candidates = CandidateSet.one(study, name=name)
        if candidates.member_kind != "backtest":
            raise RuntimeError(
                f"candidate set {candidates.hash} holds {candidates.member_kind} members; "
                "the holdout selection requires backtests"
            )
        selected = candidates.best_validation_sharpe()
        return SelectionField(
            name=name,
            members=tuple(candidates.members),
            selected=selected,
            label=label_of(study, selected),
            source=f"frozen candidate set {candidates.hash} ({len(candidates.members)} members)",
            candidate_set=candidates,
        )

    live = resolve_field_members(
        study,
        case_study=case_study,
        prediction_hashes=prediction_hashes,
        resolve_best_backtest_runs=resolve_best_backtest_runs,
        stages=stages,
    )
    # A candidate with no registered Sharpe cannot be ranked, so it is not eligible - and
    # `completeness()` does not catch it, because a backtest is complete once a metrics ROW
    # exists whether or not that row carries a Sharpe. Polars sorts nulls first on a descending
    # sort, so without this a null-Sharpe row would sort above every real one and be selected.
    # Completeness itself is `resolve_field_members`' job now, ahead of the coverage check, so
    # that the live field and the frozen one are the same set rather than the same set minus
    # whatever the live path quietly dropped.
    rankable = live.filter(pl.col("sharpe").is_not_null())
    complete: list[Result] = [
        study.results.open(row["backtest_hash"])
        for row in rankable.sort("sharpe", "backtest_hash", descending=[True, False]).iter_rows(
            named=True
        )
    ]
    if not complete:
        raise RuntimeError(
            f"no candidate set {name!r} in this registry and none of the {live.height} "
            "eligible validation backtests carries a Sharpe, so there is no selection to "
            "carry forward"
        )
    selected = complete[0]
    return SelectionField(
        name=name,
        members=tuple(member.hash for member in complete),
        selected=selected,
        label=label_of(study, selected),
        source=(
            f"live ranking of {len(complete)} complete validation backtests "
            f"(no {name!r} in this registry)"
        ),
        candidate_set=None,
    )
