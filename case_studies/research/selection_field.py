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


def config_counts(registry: Any, prediction_hashes: Any) -> int:
    """How many distinct ``(family, config_name)`` pairs those predictions were fitted under.

    The funnel counts model configurations, not backtest rows: ``top_n_predictions[stage]``
    admits that many ``(family, config_name)`` pairs, and each of them contributes as many
    rows as there are allocators, top-k values and overlays. Counting rows would compare a
    number against a cut it is not measured in.

    The pairs live on ``training_runs`` and reach a backtest through its prediction, so this
    is the same join ``label_of`` makes for one row.
    """
    hashes = sorted(set(prediction_hashes))
    if not hashes:
        return 0
    placeholders = ",".join("?" * len(hashes))
    with sqlite3.connect(f"file:{registry}?mode=ro", uri=True) as db:
        return int(
            db.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT t.family, t.config_name
                    FROM prediction_sets p
                    JOIN training_runs t ON t.training_hash = p.training_hash
                    WHERE p.prediction_hash IN ({placeholders})
                )
                """,
                hashes,
            ).fetchone()[0]
        )


def advancing_shortfalls(
    reached: Mapping[str, Mapping[str, int]],
    *,
    advancing: tuple[str, ...],
    stage_cuts: Mapping[str, int] | None,
) -> dict[str, str]:
    """The labels whose advance past the baseline is part-way done, and why for each.

    *reached* maps each declared label to how many distinct model configurations it carries
    at each advancing stage; *stage_cuts* is ``top_n_predictions`` for those stages, which is
    how many configurations the funnel admits there. Both are the caller's, because the cut
    lives in ``config/setup.yaml`` under the case study and this module is handed a study,
    not a case-study name to look one up by.

    Two readings, and both are counted rather than inferred from a stage being present:

    - **Short of the cut.** A stage that admits ten configurations and carries three is a
      sweep interrupted between them. Presence at the stage says nothing about this, which is
      how a run stopped one configuration in could freeze a field missing nine.
    - **Started and not continued.** Configurations at one advancing stage and none at the
      next is the same interruption one stage later.

    A label with nothing at any advancing stage is left alone. It is either a comparison that
    ended at the baseline or a sweep that has not begun, and **the registry cannot tell those
    apart** - neither leaves a record. Refusing on it would block the first, which is a
    completed decision; the caller sees the count and can say which it is.

    Each stage is measured against its own cut because the cuts differ by an order of
    magnitude: allocation advances ten configurations and the overlay stages advance one.
    """
    shortfalls: dict[str, str] = {}
    for label, counts in reached.items():
        present = [stage for stage in advancing if counts.get(stage, 0)]
        if not present:
            continue
        reasons: list[str] = []
        for stage in advancing:
            found = counts.get(stage, 0)
            cut = (stage_cuts or {}).get(stage)
            if not found:
                reasons.append(f"{stage}: none")
            elif cut and cut > 0 and found < cut:
                reasons.append(f"{stage}: {found} of {cut} configurations")
        if reasons:
            shortfalls[label] = "; ".join(reasons)
    return shortfalls


def resolve_field_members(
    study: Study,
    *,
    case_study: str,
    prediction_hashes: Any,
    resolve_best_backtest_runs: Any,
    stages: tuple[str, ...] = FIELD_STAGES,
    coverage_stage: str = COVERAGE_STAGE,
    stage_cuts: Mapping[str, int] | None = None,
) -> pl.DataFrame:
    """Every eligible validation backtest across the declared labels and the field's stages.

    Advancing past the baseline is a decision, so completeness cannot be asked uniformly.

    **Every declared label is backtested equal-weight**, and that is what makes the labels
    comparable, so a declared label with no baseline rows means the run is unfinished. The
    set is immutable under its name, so freezing there publishes a field nothing can add to.

    **What advances past the baseline is a decision the registry does not record.** A label
    the comparison shows to be dominated stops at the baseline deliberately, and requiring it
    in allocation and risk overlay would order backtests whose only purpose is filling a
    matrix. Nothing distinguishes that from a sweep that has not started, so a label with no
    post-baseline rows at all is left to the caller rather than refused.

    **What the registry does answer is whether an advance that started has finished**, and
    that is asked in the funnel's own unit. ``top_n_predictions[stage]`` admits a number of
    model configurations, so a stage carrying fewer configurations than its cut is a sweep
    interrupted part-way, and one carrying configurations while the next carries none is the
    same interruption one stage later. Both are counted per label, against that label's own
    rows and that stage's own cut, because the allocation notebook applies the cut inside one
    label and the overlay stages admit an order of magnitude fewer configurations than
    allocation does.

    Counting rather than declaring is what survives more work arriving. A written-down
    advancement set is correct until a new configuration outranks the ones on it, and then it
    refuses the very sweep that should now run; a count re-reads the registry.

    Rows with no Sharpe are dropped before any of this. They are ineligible by
    construction, since the selection ranks on validation backtest Sharpe, and leaving them
    in makes them count towards coverage and then fails the whole frozen set later, when
    ``best_validation_sharpe`` rejects it for holding a member it cannot rank.
    """
    labels = sweep_labels(study)
    registry = study.root / "run_log" / "registry.db"
    frames: list[pl.DataFrame] = []
    reached: dict[str, dict[str, int]] = {label: {} for label in labels}
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
            reached[label][stage] = config_counts(registry, rows["prediction_hash"].to_list())
            frames.append(rows)

    uncovered = [label for label in labels if not reached[label].get(coverage_stage)]
    if uncovered:
        raise RuntimeError(
            f"the holdout field cannot be frozen while these declared labels have no rankable "
            f"validation backtests at the {coverage_stage!r} stage: {uncovered}. Every declared "
            "label is backtested equal-weight, so an absent one means the run is unfinished, and "
            "the set is immutable under its name, so no later run could add it."
        )

    advancing = tuple(stage for stage in stages if stage != coverage_stage)
    shortfalls = advancing_shortfalls(reached, advancing=advancing, stage_cuts=stage_cuts)
    if shortfalls:
        raise RuntimeError(
            f"the holdout field cannot be frozen while these labels are part-way through "
            f"advancing: {shortfalls}. Each count is the distinct model configurations that "
            f"label carries at that stage, against the {dict(stage_cuts or {})} the funnel "
            "admits there. A label short of its cut, or carrying configurations at one "
            "advancing stage and none at the next, is a sweep still running, and freezing now "
            "would permanently exclude candidates that could have won."
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
    stage_cuts: Mapping[str, int] | None = None,
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
        stage_cuts=stage_cuts,
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
