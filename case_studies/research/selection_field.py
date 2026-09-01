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


def derive_advancing_labels(
    baseline_rows: pl.DataFrame,
    *,
    top_n: int | None,
) -> frozenset[str]:
    """The labels the baselines favour, taken from the pooled baseline ranking.

    The funnel that feeds *stage* keeps the best ``top_n_predictions[stage]`` model
    configurations, so that is the unit ranked here: one ``(label, family, config_name)``
    triple contributes its best baseline Sharpe, and the labels appearing in the top N are
    the ones the comparison carried forward. The pool spans labels because comparing them is
    the whole point - within one label the same query answers which configurations advance.

    *top_n* is the caller's cut, read from ``backtest.sweep.top_n_predictions`` where the
    caller has a case study to read it from. It is a parameter rather than a lookup inside
    here because this module is also used by readers rebuilding the field in a clean clone,
    where resolving a case-study name to a config path is exactly what is unavailable.

    Returns every label when the rows cannot support the ranking - no cut given, no rows at
    all, or a frame without the family and configuration columns. An empty answer would
    silently excuse every label from the completeness check below, which is the opposite of
    what an unanswerable question should do.
    """
    if baseline_rows.is_empty():
        return frozenset()
    every = frozenset(baseline_rows["label"].unique().to_list())
    required = {"label", "family", "config_name", "sharpe"}
    if top_n is None or top_n <= 0 or not required.issubset(baseline_rows.columns):
        return every
    ranked = (
        baseline_rows.filter(pl.col("sharpe").is_not_null())
        .group_by("label", "family", "config_name")
        .agg(pl.col("sharpe").max().alias("best_sharpe"))
        .sort("best_sharpe", descending=True)
        .head(top_n)
    )
    if ranked.is_empty():
        return every
    return frozenset(ranked["label"].unique().to_list())


def resolve_field_members(
    study: Study,
    *,
    case_study: str,
    prediction_hashes: Any,
    resolve_best_backtest_runs: Any,
    stages: tuple[str, ...] = FIELD_STAGES,
    coverage_stage: str = COVERAGE_STAGE,
    advancing_top_n: int | None = None,
) -> pl.DataFrame:
    """Every eligible validation backtest across the declared labels and the field's stages.

    Advancing past the baseline is a decision, so completeness cannot be asked uniformly.

    **Every declared label is backtested equal-weight**, and that is what makes the labels
    comparable, so a declared label with no baseline rows means the run is unfinished. The
    set is immutable under its name, so freezing there publishes a field nothing can add to.

    **What advances past the baseline is whichever labels the baselines favour.** A label
    the comparison shows to be dominated stops there deliberately, and requiring it in
    allocation and risk overlay would order backtests whose only purpose is filling a
    matrix. So a label absent from every post-baseline stage is a completed decision.

    **Which labels those are is derived from the baselines, not guessed from what ran.**
    Ranking the pooled baseline results answers "which labels did the comparison favour"
    directly, so a label that is favoured and has no allocation rows is an unfinished sweep
    rather than a decision - the distinction reading post-baseline rows alone cannot make.
    The unit ranked is the model configuration, matching the funnel: ``top_n_predictions``
    counts one ``(family, config_name)`` pair per candidate, and the pool spans labels
    because the same configuration on two labels is two candidates.

    Deriving rather than declaring is what survives more work arriving. A written-down
    advancement set is correct until a new configuration promotes a label that was dominated,
    and then it refuses the very sweep that should now run; a derived set re-ranks itself.

    **A label part-way through advancing is refused either way.** It is the sequential-run
    failure: a run mid-way through the second advancing label's sweep has that label's
    allocation rows and not its risk rows, and freezing then excludes candidates that could
    have won. So both readings are applied - a favoured label missing any advancing stage,
    and any label that reached one advancing stage and not the next.

    Rows with no Sharpe are dropped before any of this. They are ineligible by
    construction, since the selection ranks on validation backtest Sharpe, and leaving them
    in makes them count towards coverage and then fails the whole frozen set later, when
    ``best_validation_sharpe`` rejects it for holding a member it cannot rank.
    """
    labels = sweep_labels(study)
    frames: list[pl.DataFrame] = []
    baseline_frames: list[pl.DataFrame] = []
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
            if stage == coverage_stage:
                baseline_frames.append(rows.with_columns(pl.lit(label).alias("label")))

    baseline_rows = (
        pl.concat(baseline_frames, how="diagonal_relaxed") if baseline_frames else pl.DataFrame()
    )
    uncovered = [label for label in labels if coverage_stage not in reached[label]]
    if uncovered:
        raise RuntimeError(
            f"the holdout field cannot be frozen while these declared labels have no rankable "
            f"validation backtests at the {coverage_stage!r} stage: {uncovered}. Every declared "
            "label is backtested equal-weight, so an absent one means the run is unfinished, and "
            "the set is immutable under its name, so no later run could add it."
        )

    advancing = {stage for stage in stages if stage != coverage_stage}
    favoured = derive_advancing_labels(baseline_rows, top_n=advancing_top_n)
    unstarted = {
        label: sorted(advancing - reached[label])
        for label in labels
        if label in favoured and not advancing.issubset(reached[label])
    }
    partial = {
        label: sorted(advancing - reached[label])
        for label in labels
        if reached[label] & advancing and not advancing.issubset(reached[label])
    }
    missing = {**partial, **unstarted}
    if missing:
        raise RuntimeError(
            f"the holdout field cannot be frozen while these labels are part-way through "
            f"advancing: {missing} (stage: missing). The baselines favour {sorted(favoured)}, "
            "so those have to be present in every advancing stage; and a label that reached one "
            "advancing stage and not the next is a sweep still running whatever the baselines "
            "said. Freezing now would permanently exclude candidates that could have won."
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
    advancing_top_n: int | None = None,
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
        advancing_top_n=advancing_top_n,
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
