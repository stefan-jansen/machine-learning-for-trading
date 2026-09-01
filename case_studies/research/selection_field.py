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

import sqlite3
from collections.abc import Mapping
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
#: The stages after it are where the comparison pays off: a label the baselines show to be
#: dominated is dropped rather than carried through allocation and risk for symmetry.
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


def unfinished_sweep_plans(study: Study, *, plan_names: Mapping[str, str]) -> list[str]:
    """Which of the recorded sweep plans are absent or not fully executed, one line each.

    A sweep notebook computes every backtest identity it intends to register before it
    registers any of them, and publishes that list as an official population. This asks each
    named plan whether every member is present and complete, and that is the only question
    about a sweep's completion the registry can actually answer.

    It is asked because none of the alternatives work. An interrupted sweep leaves rows that
    look exactly like a smaller finished one: the stage is present, the surviving model
    configurations are each fully represented, and the row count is simply lower with nothing
    to compare it against. Stage presence, row counts and configuration counts were each tried
    and each accepts an interruption as complete. The plan is the comparison they lacked.

    Absent and incomplete are reported the same way on purpose. A plan that was never written
    is a sweep that either has not run or ran before plans were recorded, and neither is a
    sweep whose results may be sealed into an immutable set.

    ``OperationalError`` is caught alongside the rest because a registry that predates official
    populations has no table to query, which is an absent plan and not a broken one.
    """
    unfinished: list[str] = []
    for description, name in plan_names.items():
        try:
            OfficialPopulation.one(study, name=name).require_complete()
        except (KeyError, ValueError, sqlite3.OperationalError) as exc:
            unfinished.append(f"{description}: {exc}")
    return unfinished


def advancing_labels(
    reached: Mapping[str, set[str]],
    *,
    coverage_stage: str = COVERAGE_STAGE,
) -> list[str]:
    """The labels whose sweeps the freeze has to wait for, read off the field itself.

    The funnel permits a label to stop after its baseline: dominated there, it earns no
    allocation or overlay sweep. So the freeze cannot require a plan from every declared label,
    or a field containing a deliberately dropped label could never be sealed. Nor can it require
    one from no label, which is what let a half-run sweep be frozen in the first place.

    A label is waited for when it has rows past the coverage stage in the field being frozen -
    the field the caller has already resolved, restricted to the prediction populations in
    force. That is a property of the current field rather than a declaration about intent, and
    it separates the three cases without anyone writing anything down:

    - **Never advanced.** No rows past the baseline, so nothing is waited for, and nothing but
      its baseline rows is in the field either. A deliberate drop and a sweep nobody has run yet
      are the same state and want the same answer.
    - **Interrupted.** Its sweep published a plan before executing, so the label has rows past
      the baseline *and* a plan whose members are not all registered. It is waited for, and
      ``unfinished_sweep_plans`` reports it.
    - **Finished.** Rows and a complete plan, so the wait is satisfied.

    An earlier version of this asked whether a plan had ever been recorded under the label's
    name. That reads the wrong generation: after a refit, a label that advanced under superseded
    predictions still has its old plan on record while none of its old rows are eligible, so it
    could never stop advancing. The field already answers the question the plan was standing in
    for.
    """
    return sorted(label for label, stages in reached.items() if stages - {coverage_stage})


def resolve_field_members(
    study: Study,
    *,
    case_study: str,
    prediction_hashes: Any,
    resolve_best_backtest_runs: Any,
    stages: tuple[str, ...] = FIELD_STAGES,
    coverage_stage: str = COVERAGE_STAGE,
) -> tuple[pl.DataFrame, dict[str, set[str]]]:
    """Every eligible validation backtest across the declared labels and the field's stages.

    Advancing past the baseline is a decision, so completeness cannot be asked uniformly.

    **Every declared label is backtested equal-weight**, and that is what makes the labels
    comparable, so a declared label with no baseline rows means the run is unfinished. The
    set is immutable under its name, so freezing there publishes a field nothing can add to.

    **Whether the stages past the baseline are finished is not asked here, because no reading
    of the rows can answer it.** Each sweep plans a grid - configurations by top-k by
    allocator, or one carrier by risk control - and an interrupted run leaves rows that look
    like a smaller finished grid from every angle: the stage is present, and each surviving
    configuration is fully represented. Presence, row counts and configuration counts were all
    tried and each of them accepts an interruption as complete.

    What answers it is the plan itself. The sweep notebooks compute every expected backtest
    identity before executing, and publish that list as an official population, so completion
    is ``require_complete`` on a recorded plan rather than an inference from what happens to be
    in the registry. That check belongs where the field is frozen, since it is the freeze that
    is irreversible; this function builds the field and the caller decides whether it may be
    sealed.

    Rows with no Sharpe are dropped before any of this. They are ineligible by
    construction, since the selection ranks on validation backtest Sharpe, and leaving them
    in makes them count towards coverage and then fails the whole frozen set later, when
    ``best_validation_sharpe`` rejects it for holding a member it cannot rank.

    Returns the field and the stages each label reached in it. The second is what
    :func:`advancing_labels` reads to decide which labels the freeze waits for, and it is
    returned rather than recomputed because it is a by-product of building the field and a
    second pass would resolve a different one wherever the registry moved in between.
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
    return pl.concat(frames, how="diagonal_relaxed").unique("backtest_hash"), reached


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
    # `CandidateSet.create` refuses partial members, so a frozen field is complete by
    # construction and everything downstream may assume it; `resolve_best_backtest_runs` ranks
    # on registered metrics and applies no such filter. Checking only the selection would leave
    # the two fields different in exactly the way that matters to a reader of the membership.
    #
    # A candidate with no registered Sharpe cannot be ranked, so it is not eligible - and
    # `completeness()` does not catch it, because a backtest is complete once a metrics ROW
    # exists whether or not that row carries a Sharpe. Polars sorts nulls first on a descending
    # sort, so without this a null-Sharpe row would sort above every real one and be selected.
    rankable = live.filter(pl.col("sharpe").is_not_null())
    complete: list[Result] = []
    incomplete: list[str] = []
    for row in rankable.sort("sharpe", "backtest_hash", descending=[True, False]).iter_rows(
        named=True
    ):
        member = study.results.open(row["backtest_hash"])
        reason = member.completeness()
        if reason is None:
            complete.append(member)
        else:
            incomplete.append(f"{row['backtest_hash']} ({reason})")
    if not complete:
        raise RuntimeError(
            f"no candidate set {name!r} in this registry and none of the {live.height} "
            "eligible validation backtests is both rankable and complete, so there is no "
            "selection to carry forward: " + "; ".join(incomplete[:5])
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
