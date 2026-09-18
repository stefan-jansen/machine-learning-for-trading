from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from case_studies.utils.coverage import CrossSectionReport

# Families excluded from ALL backtest sweeps — predictions lack y_score column
_BACKTEST_EXCLUDED_FAMILIES: set[str] = {"causal_dml"}

# The minimum cross-section `cross_sectional_ic_series` needs before it will return a
# coefficient for a date. 5 at every case-study call site; the library default is 10.
IC_MIN_OBS = 5


def defined_ic(frame: pl.DataFrame, ic_col: str = "ic") -> pl.DataFrame:
    """Drop the dates of an IC series whose coefficient is undefined.

    A date with fewer than `IC_MIN_OBS` names, or with every prediction or every
    return tied, has no rank correlation. ml4t-diagnostic 0.1.2 and later report
    such a date as null, but polars treats null and NaN as different values and
    `daily_metrics.parquet` files written before that release carry NaN instead.
    A `drop_nulls` alone therefore leaves the NaN in place, and one NaN turns any
    mean, std or rolling mean taken afterwards into NaN.

    Use this wherever an IC series is read back from disk or crosses into a
    statistic, rather than `drop_nulls(ic_col)`.
    """
    if ic_col not in frame.columns:
        return frame
    return frame.filter(pl.col(ic_col).is_not_null() & pl.col(ic_col).is_finite())


# The predictions parquet schema is not uniform across the nine case studies, so the
# validity rule has to resolve its own column names before it can count anything.
# Measured: etfs / fx_pairs / us_firm_characteristics use prediction+actual+symbol,
# sp500_equity_option_analytics uses y_score+y_true+symbol, cme_futures keys on product.
_PREDICTION_ALIASES = ("prediction", "y_score", "y_pred", "score")
# `eval_actual` first: for a classification label the stored IC is computed against
# the continuous return, which `registry/store.py:712-745` writes under that name
# beside the class target. Resolving `actual`/`y_true` ahead of it would judge a
# date's validity on the class column while the metric used the return.
_ACTUAL_ALIASES = ("eval_actual", "actual", "y_true", "realized", "target")
_ENTITY_ALIASES = ("symbol", "product", "asset", "entity")


def _first_present(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    return next((a for a in aliases if a in columns), None)


def _is_finite(dtype: pl.DataType, column: str) -> pl.Expr:
    """True where the column holds a real number, as the IC series requires.

    Null, NaN and infinity each mean the entity contributes nothing to that date's
    rank correlation. ``is_finite`` is undefined on a non-float column, where being
    non-null is the whole of the condition.
    """
    if dtype.is_float():
        return pl.col(column).is_not_null() & pl.col(column).is_finite()
    return pl.col(column).is_not_null()


def excluded_families(case_study: str, *, for_backtest: bool = False) -> set[str]:
    return set(_BACKTEST_EXCLUDED_FAMILIES) if for_backtest else set()


def excluded_family_sql(
    case_study: str, family_column: str = "family", *, for_backtest: bool = False
) -> tuple[str, list[str]]:
    excluded = sorted(excluded_families(case_study, for_backtest=for_backtest))
    if not excluded:
        return "", []

    placeholders = ", ".join("?" for _ in excluded)
    return f" AND {family_column} NOT IN ({placeholders})", excluded


# A constant fold reaches the registry in one of two shapes, and the test has to carry both.
# Five of the nine registries store NULL for it; `nasdaq100_microstructure` stores a denormal
# instead, because its predictions are constant to display precision without being bit-identical,
# so the daily IC series exists and averages to about 2e-16 rather than collapsing to undefined.
#
# `1e-12` is not a tuned number, it is the middle of an empty band. Measured across all nine
# registries on 2026-09-12: 24 rows fall under 1e-12, all of them nasdaq's four LASSO/ElasticNet
# configurations on the three continuous labels, minimum 2.0048e-16. The smallest legitimate
# |ic| anywhere is 3.1941e-07 (`sp500_equity_option_analytics`), then 4.0948e-06, 4.4127e-06,
# 5.9011e-06 and 6.5706e-06. Nine orders of magnitude separate the two groups, so any threshold
# between 1e-12 and 1e-9 selects exactly the same rows and no other case study moves.
#
# `ic_std` is in the test because `ic` alone cannot distinguish the two ways a fold reaches a
# near-zero average. `fold_metrics.ic` is `ic_result["ic_mean"]` (`registry/metrics.py:121`), the
# mean of the fold's per-cross-section Spearman ICs, so a fold that ranks perfectly well but
# whose daily ICs cancel would also average to nearly nothing. That fold is a real result and
# must not be excluded. The two cases separate on dispersion rather than on the mean: a
# cancelling series has a large `ic_std`, a constant one has none.
#
# Measured across all nine registries on 2026-09-12, 38,518 fold rows. The 24 rows with a
# denormal `ic` carry `ic_std` of 1.95e-17 to 2.02e-17; the smallest `ic_std` on any other row
# is 1.69e-03 (`us_equities_panel`), and the largest is 0.269. Fourteen orders of magnitude,
# and no row anywhere has a non-null `ic` with a null `ic_std`, so the conjunction needs no
# null branch. The reason to prefer this form is not the width of that gap: it is that
# cancellation is excluded by what the clause tests rather than by the absence of an example.
_DEGENERATE_IC_EPS = 1e-12
_DEGENERATE_SUBQUERY = (
    "SELECT prediction_hash FROM fold_metrics WHERE ic IS NULL "
    f"OR (abs(ic) < {_DEGENERATE_IC_EPS} AND ic_std < {_DEGENERATE_IC_EPS})"
)


def degenerate_prediction_sql(prediction_hash_column: str = "p.prediction_hash") -> str:
    """SQL clause excluding prediction sets with any constant-prediction fold.

    When a regularized linear model (LASSO / ElasticNet at high ``alpha_frac``)
    shrinks every coefficient to zero on a fold, that fold's predictions are
    constant and its IC carries no information. The pooled daily IC is then
    computed over a fold that ranks nothing, which biases it (typically upward)
    and is not a valid model result. Such prediction sets must never be selected
    for backtesting or any follow-on leaderboard.

    **The IC is not always NULL, and testing only for NULL is how this stayed
    invisible.** A fold whose every prediction ties does collapse to an undefined
    correlation and is stored as NULL; a fold whose predictions are constant to
    display precision but not bit-identical produces a defined, denormal IC
    instead. Both are the same defect and the clause excludes both - see
    ``_DEGENERATE_IC_EPS`` for the measurement behind the threshold.

    Returns a fragment beginning with ``" AND "`` suitable for appending to a
    WHERE clause; takes no bound parameters. Pass the column expression naming
    ``prediction_hash`` in the surrounding query (default ``p.prediction_hash``).
    """
    return f" AND {prediction_hash_column} NOT IN ({_DEGENERATE_SUBQUERY})"


def degenerate_prediction_hashes(case_dir: Path) -> set[str]:
    """The prediction sets ``degenerate_prediction_sql`` excludes, as a set.

    Same rule and same source, for a caller that has to reason about the exclusion rather than
    apply it. A population is declared before anything is fitted and degeneracy is only visible
    afterwards, so a cross-check between a declared population and a leaderboard has to allow
    for the rows the leaderboard drops - otherwise it reports a correct exclusion as a missing
    member. Returns an empty set when the registry or the table is absent.
    """
    import sqlite3

    db_path = Path(case_dir) / "run_log" / "registry.db"
    if not db_path.is_file():
        return set()
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        if not db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fold_metrics'"
        ).fetchone():
            return set()
        return {row[0] for row in db.execute(f"SELECT DISTINCT {_DEGENERATE_SUBQUERY[7:]}")}


def incompletely_registered_predictions(case_dir: Path, hashes: Iterable[str]) -> dict[str, str]:
    """Which of ``hashes`` are registered but not finished, and what is short in each.

    A headline row in ``prediction_metrics`` is not evidence that a prediction set arrived.
    Coverage, the headline metrics and the per-fold metrics are committed as separate writes,
    so a run interrupted between them leaves a hash that every metrics query returns and that
    `PredictionResult.complete` rejects. A leaderboard reading the headline alone then scores a
    member over the folds it managed, and a short window is an easier window.

    Checks coverage reporting ``complete``, one ``fold_metrics`` row per expected fold, and the
    predictions parquet on disk. A member with no coverage row at all is not reported here -
    see :func:`predictions_without_coverage`, which separates a gap in the evidence from a run
    that stopped part way. The file is checked for existence and not read:
    `PredictionResult.complete` re-digests it, which is minutes of I/O for a caller running this
    over a thousand members on every execution, and the failure that catches is corruption after
    registration rather than the interrupted registration at issue. A member whose artifact was
    deleted or never written is a different matter - the registry still lists it, the leaderboard
    still ranks it, and `load_predictions` then returns nothing for it without saying so.

    Returns ``{hash: reason}`` for the members that fall short, empty when they all arrived or
    when the registry has no coverage table to check them against.
    """
    run_log = Path(case_dir) / "run_log"
    db_path = run_log / "registry.db"
    predictions_dir = run_log / "predictions"
    wanted = sorted(set(hashes))
    if not wanted or not db_path.is_file():
        return {}
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if not {"prediction_coverage", "fold_metrics"} <= tables:
            return {}
        # The counts, not only `status`. `status` is a stored verdict that no stored row can
        # contradict: registration raises when coverage is partial and `allow_partial`
        # defaults to False everywhere in production, so the refused evaluations are never
        # written and the column reads `complete` in all 6,480 rows across all nine
        # registries. Reading it alone therefore repeats what the row's existence already
        # said, while the counts are the evidence it was drawn from and a row can carry them
        # in contradiction to it.
        #
        # On today's corpus it cannot, and this is not the change that fixes that. Measured
        # 2026-09-18, all five signals below are clean on all 6,480 rows - `n_missing`,
        # `n_extra` and `n_duplicates` zero and the two digests equal and non-null - including
        # the 140 in `sp500_equity_option_analytics` that `prediction_admissibility` rules
        # inadmissible. They are all computed against the same `expected_keys`, the frame the
        # family's own adapter prepared, so a prediction that narrowed its own declared
        # universe agrees with itself on every one of them. The declared-universe comparison
        # is `prediction_admissibility`'s, and its columns are where it is now recorded.
        coverage = {
            row[0]: row[1:]
            for row in db.execute(
                "SELECT prediction_hash, status, n_folds_expected, n_missing, n_extra, "
                "n_duplicates, expected_key_digest, actual_key_digest FROM prediction_coverage"
            )
        }
        folds = dict(
            db.execute("SELECT prediction_hash, COUNT(*) FROM fold_metrics GROUP BY 1").fetchall()
        )
    short: dict[str, str] = {}
    for member in wanted:
        if member not in coverage:
            # Not reported. A member with no coverage row at all is not evidence of an
            # interrupted run: coverage arrived as a later migration, so a registry written
            # before it, or a producer that wrote metrics without it, leaves the row absent
            # while the prediction set is whole. Measured on etfs, where all 40 such members
            # had their `prediction_sets` row and their parquet. `predictions_without_coverage`
            # reports them separately, as a gap in the evidence rather than a partial run.
            continue
        status, expected, n_missing, n_extra, n_duplicates, want_digest, got_digest = coverage[
            member
        ]
        actual = folds.get(member, 0)
        artifact = predictions_dir / member / "predictions.parquet"
        gaps = [
            f"{count} {name}"
            for count, name in (
                (n_missing, "missing"),
                (n_extra, "extra"),
                (n_duplicates, "duplicate"),
            )
            if count
        ]
        if gaps:
            short[member] = "coverage " + ", ".join(gaps) + " key(s)"
        elif want_digest != got_digest:
            short[member] = "coverage key set differs from the expected one"
        elif status != "complete":
            short[member] = f"coverage {status}"
        elif expected is not None and actual != expected:
            short[member] = f"{actual} of {expected} folds scored"
        elif not artifact.is_file():
            short[member] = "no predictions.parquet"
    return short


def predictions_without_coverage(case_dir: Path, hashes: Iterable[str]) -> set[str]:
    """Which of ``hashes`` the registry holds no ``prediction_coverage`` row for.

    Separated from :func:`incompletely_registered_predictions` because the two look alike and
    call for opposite responses. A coverage row that says something other than ``complete``, or
    a fold count short of what that row declares expected, is a run that stopped part way and
    must not be ranked. An absent row is a gap in the evidence: ``prediction_coverage`` arrived
    as a later migration, so a registry written before it - or a producer that wrote metrics
    without it - leaves the row missing while the prediction set itself is whole. Measured on
    etfs, where all 40 such members had both their ``prediction_sets`` row and their parquet.

    Report these; do not refuse on them. Returns an empty set where the registry or the table is
    absent, since neither says anything about a particular member.
    """
    db_path = Path(case_dir) / "run_log" / "registry.db"
    wanted = set(hashes)
    if not wanted or not db_path.is_file():
        return set()
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        if not db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_coverage'"
        ).fetchone():
            return set()
        covered = {row[0] for row in db.execute("SELECT prediction_hash FROM prediction_coverage")}
    return wanted - covered


def prediction_members_in_force(
    study,
    case_dir: Path | None = None,
    *,
    candidates: Iterable[str] | None = None,
) -> tuple[frozenset[str] | None, list[str]]:
    """Every prediction set the registry's populations currently publish, and the notes to print.

    A downstream sweep does not name the families it draws from - it ranks whatever is
    admissible - so the question here is not `declared_population_members`' "did these declared
    names resolve" but "which members are in force at all". The answer takes two steps and
    neither is enough alone.

    A population is immutable and every generation stays readable, so a candidate pool built
    straight from the registry counts a refitted configuration twice: nothing in the read path
    filters on supersession (`case_studies/utils/registry/queries.py` contains no occurrence of
    ``supersed``), and the published leaders are then fewer distinct strategies than they look.
    Resolving each published name through `OfficialPopulation.one` takes the generation nothing
    supersedes - it refuses rather than guesses if a chain has forked. Subtracting the retired
    members is still needed after that, because a narrowed or preview run freezes its own
    snapshot of whatever the catalog held that day and stays in force under its own name
    forever, so the union alone hands a retired generation back through the frozen name that
    still lists it.

    Refuses if any member is registered but unfinished, by the rule
    :func:`incompletely_registered_predictions` applies - a pool that is selected from cannot
    carry a member scored over fewer folds than it was asked for.

    ``candidates`` is the members the caller can actually act on, and passing it is the
    difference between checking a sweep's own pool and checking the registry. Without it this
    reads every published member of every label: on ``nasdaq100_microstructure/14_backtest``
    that was 784 members, 94.3 GB of ``predictions.parquet`` at 11.0 s each, **8,608 s before
    the first backtest** - and the run then swept 162 of them, because the other 537 belong to
    ``fwd_ret_60m``, ``fwd_ret_5m`` and ``fwd_dir_15m``, which that run can never sweep. 68% of
    the reads were for rows the caller had already excluded, and every ``print`` in the cell is
    after this returns, so a watcher sees nothing for two and a half hours.

    It narrows the scope, not the checks. Every member the caller could select is still asked
    every question it was asked before, including the refusal below - what goes away is asking
    them of members this run cannot reach. One thing does change: an unfinished member under a
    label this run does not sweep no longer refuses the run. That refusal protects a selection,
    and a member that cannot be selected cannot corrupt one; it was an incidental early warning
    about the registry, not a guarantee about the run, and the sweep that does read that label
    still refuses.

    Returns ``None`` and a note where the registry publishes no populations - a fixture or a
    reader's clean clone, where there is no declaration to filter against. ``None`` rather than
    an empty set because these callers pass the result straight to a ``prediction_hashes``
    parameter that already reads ``None`` as unscoped, and because an empty set there would turn
    "nothing to filter by" into "nothing is admissible" - a candidate pool of zero reported as
    an ordinary result.
    """
    from case_studies.research import (
        OfficialPopulation,
        published_population_names_at,
        superseded_members_at,
    )

    # The registry this study reads, which is `study.root` only at canonical tier: a preview
    # keeps `root` pointing at the case directory and writes elsewhere, so asking it returns the
    # canonical populations. A preview then filters its own prediction sets against 866 hashes
    # none of which it holds, and reports "No predictions found" on a registry with 247 of them
    # - the state this function documents as returning None. `declared_population_members` takes
    # the directory for the same reason; this took the study alone. Measured 2026-09-06.
    root = case_dir if case_dir is not None else study.root
    names = sorted(published_population_names_at(root))
    if not names:
        return None, [
            f"{root} publishes no official populations, so no supersession filter is "
            "applied: the candidate pool rests on catalog admissibility alone."
        ]
    published: set[str] = set()
    for name in names:
        published.update(OfficialPopulation.one(study, name=name).members)
    members = frozenset(published - superseded_members_at(root))
    if candidates is not None:
        members = members & frozenset(candidates)

    # A population is written down before its members are fitted, so being in force is not
    # evidence of having finished. Every caller here ranks what comes back and selects from the
    # ranking, and an unfinished member is scored over the folds it managed - a shorter window
    # is an easier window, so the error runs toward the top of the ranking. Refusing is the only
    # safe answer: there is no way to rank around a member without saying the pool changed.
    short = incompletely_registered_predictions(root, members)
    if short:
        named = ", ".join(f"{member}: {why}" for member, why in sorted(short.items())[:5])
        raise RuntimeError(
            f"{len(short)} member(s) of the populations in force are registered but "
            f"unfinished: {named}. Selecting from this pool would compare a partial run "
            "against complete ones."
        )
    uncovered = predictions_without_coverage(study.root, members)
    notes = (
        [
            f"{len(uncovered):,} of {len(members):,} members carry no prediction_coverage row. "
            "They are ranked; the gap is in the registry, not in the run. A member that does "
            "carry one is not thereby shown to cover the cross-section its peers ranked - that "
            "row compares a prediction against the keys its own family's adapter prepared, and "
            "the declared-universe comparison is the one recorded in prediction_admissibility."
        ]
        if uncovered
        else []
    )

    # The registry check above asks whether a completeness row exists. This asks whether the
    # artifact is actually there, across the symbol axis that `ic_n_days` cannot see. A
    # member that scores a fraction of the cross-section is dropped from the pool rather
    # than ranked in it, for the reason the unfinished-member refusal above gives: a
    # narrower sample is an easier one, so the error runs toward the top of the ranking.
    # Dropped rather than refused, because unlike an unfinished run this is a property of
    # the model and the pool is still rankable without it - but never silently, so the note
    # names every member and its shortfall.
    # A run registered at a reduced tier has no denominator here, so it is neither measured
    # nor dropped. Reported, because a member that went unchecked and one that passed are
    # different states and the note is the only place that difference is visible.
    reduced = _reduced_tier_members(root, members)
    if reduced:
        notes.append(
            f"{len(reduced):,} of {len(members):,} members were not measured for "
            f"cross-sectional coverage: {sorted(reduced.values())[0]}. They are ranked."
        )
    from case_studies.utils.coverage import BACKTEST_COVERAGE_MINIMUM

    measured, unevaluable = measure_prediction_cross_sections(
        root, [member for member in members if member not in reduced], case_study=study.case_study
    )
    short = {
        phash: report.summary()
        for phash, report in measured.items()
        if report.accountable_coverage < BACKTEST_COVERAGE_MINIMUM
    }
    short.update(unevaluable)
    if short:
        members = frozenset(members - short.keys())
        listed = "; ".join(
            f"{member} {reason.splitlines()[0]}" for member, reason in sorted(short.items())[:5]
        )
        more = f" (+{len(short) - 5} more)" if len(short) > 5 else ""
        notes.append(
            f"{len(short):,} member(s) were dropped from the candidate pool for covering less "
            f"than the cross-section their feature panels offered them: {listed}{more}"
        )
    notes.extend(
        record_prediction_admissibility(root, admitted=members, short=short, measured=measured)
    )
    if not members:
        raise RuntimeError(
            f"every member of the populations in force at {root} was dropped for incomplete "
            f"cross-sectional coverage ({len(short)} of them). There is nothing left to rank, "
            "and ranking the survivors of a universe filter against each other would not be a "
            f"comparison. Reasons:\n" + "\n".join(sorted(short.values()))
        )
    return members, notes


def record_prediction_admissibility(
    root: Path | str,
    *,
    admitted: Iterable[str],
    short: Mapping[str, str],
    measured: Mapping[str, CrossSectionReport] | None = None,
) -> list[str]:
    """Write down what this measurement found, so the resolver reads it instead of guessing.

    The sweep charges every member against the feature panel it was offered. The carrier
    resolver cannot: `full_coverage_prediction_sql`'s bar counts decision days, and a family
    that scores every day for half the universe ties the day count while ranking a narrower
    cross-section. So the resolver was the looser of the two rules and a prediction the sweep
    refused to backtest could still carry the case study.

    Recomputing the check inside the resolver would pay the sweep's startup cost once per
    strategy-analysis notebook and leave two implementations agreeing by inspection, which is
    the arrangement that produced the divergence. This records the answer where both can read
    it.

    Only measured members are written. A member this sweep did not reach is absent rather than
    admitted, and :func:`selectable_validation_candidates` drops only what is recorded as NOT
    admitted - so a registry nothing has swept keeps exactly the pool it has today, and the
    record can never empty a pool on its own.

    Returns notes, not a refusal. A registry opened read-only is a normal state for a reader's
    clone, and the sweep's own result does not depend on the record being written.
    """
    from datetime import UTC, datetime

    from case_studies.utils.registry.store import REGISTRY_SCHEMA_SQL, _migrate_registry

    admitted = sorted(set(admitted))
    if not admitted and not short:
        return []
    db_path = Path(root) / "run_log" / "registry.db"
    if not db_path.is_file():
        return []
    recorded_at = datetime.now(UTC).isoformat()
    commit = _git_commit_or_none()
    measured = measured or {}

    def counts(member: str) -> tuple[int | None, ...]:
        """The declared denominator beside the delivered numerator, or nulls if unmeasured.

        A member with no report - one whose coverage could not be evaluated at all - stores
        nulls rather than zeros: zero delivered out of zero declared is a measurement, and
        "this was never measured" is not.
        """
        report = measured.get(member)
        if report is None:
            return (None, None, None, None, None)
        return (
            report.expected,
            report.delivered,
            report.achievable,
            report.delivered_achievable,
            report.entities_declared,
        )

    rows = [(member, 1, None, recorded_at, commit, *counts(member)) for member in admitted]
    rows += [
        (member, 0, reason, recorded_at, commit, *counts(member))
        for member, reason in sorted(short.items())
    ]
    try:
        with closing(sqlite3.connect(str(db_path))) as db:
            db.executescript(REGISTRY_SCHEMA_SQL)
            _migrate_registry(db)
            db.executemany(
                "INSERT INTO prediction_admissibility "
                "(prediction_hash, admitted, reason, recorded_at, git_commit, "
                " n_declared, n_delivered, n_offered, n_delivered_offered, n_entities_declared) "
                "VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(prediction_hash) DO UPDATE SET "
                "admitted=excluded.admitted, reason=excluded.reason, "
                "recorded_at=excluded.recorded_at, git_commit=excluded.git_commit, "
                "n_declared=excluded.n_declared, n_delivered=excluded.n_delivered, "
                "n_offered=excluded.n_offered, "
                "n_delivered_offered=excluded.n_delivered_offered, "
                "n_entities_declared=excluded.n_entities_declared",
                rows,
            )
            db.commit()
    except sqlite3.Error as failure:
        return [
            f"the cross-sectional coverage result was not recorded in {db_path}: {failure}. "
            "The pool this run selects from is unaffected; the carrier resolver will fall back "
            "to its own weaker bar for these members."
        ]
    return []


def predictions_the_sweep_refused(case_dir: Path | str) -> dict[str, str]:
    """Members a sweep measured and dropped, with the reason it gave.

    The complement of what :func:`record_prediction_admissibility` writes. Absence is not
    admission: a member no sweep has measured has no row here, and a caller must leave it
    where it is rather than treat the silence as either answer. Returns an empty mapping
    where the registry, or the table, does not exist - which is a fixture, a reader's clean
    clone, or a registry last swept before the table did.
    """
    db_path = Path(case_dir) / "run_log" / "registry.db"
    if not db_path.is_file():
        return {}
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        if not db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_admissibility'"
        ).fetchone():
            return {}
        return {
            row[0]: row[1] or "measured short of the cross-section its feature panels offered"
            for row in db.execute(
                "SELECT prediction_hash, reason FROM prediction_admissibility WHERE admitted = 0"
            )
        }


def _git_commit_or_none() -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _reduced_tier_members(
    case_dir: Path | str,
    members: Iterable[str],
) -> dict[str, str]:
    """Members registered at a reduced tier, whose coverage has no denominator here.

    ``undercovered_prediction_members`` charges a member against the case study's feature
    panel, and a reduced run was not offered that panel. ``Study.activate`` symlinks
    ``features`` and ``labels`` from the canonical case directory into the preview
    workspace, so both read the full universe while the run was handed a fraction of it.
    Measured 2026-09-09 on ``~/ml4t/artifacts/smoke/etfs/.preview/etfs``: 247 members, a
    100-ETF panel, and reductions from ``max_symbols: 8`` down to five names. Charged
    against the panel every one reads short, all 247 are dropped and the pool raises
    "nothing left to rank" - the smoke run this program takes before every stage fails on
    runs that did exactly what they were told.

    **The tier and not the spec.** ``input_data_spec`` carries ``max_symbols`` and
    ``symbols`` for ``linear``, ``gbm`` and ``tabular_dl``, and for ``deep_learning`` and
    ``latent_factors`` it is a different shape entirely - ``{files, input_digest, version}``
    - which declares no universe at all. Reading the reduction out of the spec would
    therefore measure three families and silently skip two, and giving those two a universe
    key would rewrite ``computation``, which is hashed whole, and re-price every run they
    have ever registered. ``execution_tier`` is on the training row, is already part of what
    a reduced run declares about itself, and is true for every family.

    Returned so the caller can say these went unmeasured, and not returned as short: short
    means the run lost rows it was given, and a reduced run was never given them. A row with
    no tier is canonical - that is what the column meant before it was added.
    """
    case_dir = Path(case_dir)
    db_path = case_dir / "run_log" / "registry.db"
    wanted = list(members)
    if not wanted or not db_path.is_file():
        return {}
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        placeholders = ",".join("?" * len(wanted))
        rows = db.execute(
            f"""SELECT p.prediction_hash, t.execution_tier
                FROM prediction_sets p JOIN training_runs t ON t.training_hash = p.training_hash
                WHERE p.prediction_hash IN ({placeholders})""",
            wanted,
        ).fetchall()
    return {
        phash: (
            f"registered at execution tier {tier!r}, which is offered a reduced universe "
            "while the feature panel beside it is the canonical one"
        )
        for phash, tier in rows
        if tier is not None and tier != "canonical"
    }


def _declared_folds(spec_json: str | None) -> tuple[int, ...] | None:
    """The fold indices a training spec says it ran, or ``None`` when it says nothing.

    ``None`` and not an empty tuple: a spec that carries no ``computation.cv.folds`` makes
    no claim about its fold axis, and the caller must then fall back to the configuration
    rather than measure against nothing.
    """
    if not spec_json:
        return None
    try:
        cv = json.loads(spec_json).get("computation", {}).get("cv", {})
    except (TypeError, ValueError):
        return None
    declared = cv.get("folds") if isinstance(cv, dict) else None
    if not isinstance(declared, list):
        return None
    indices = [
        int(entry["fold"])
        for entry in declared
        if isinstance(entry, dict) and isinstance(entry.get("fold"), int)
    ]
    return tuple(sorted(set(indices))) or None


def _persistent_panel_entities(
    spec_json: str | None,
    panel: pl.DataFrame,
    *,
    family: str,
    config: str,
) -> dict[int, list[str]] | None:
    """The entities each fold's panel builder admitted, or ``None`` when all of them are.

    Only the models in ``latent_factors.PERSISTENT_PANEL_MODELS`` build a dense panel, and
    only they refuse an entity before the fit sees it. For everything else the answer is
    "all of them", which is what ``None`` says, and the guard is unchanged.

    The admitted set is recomputed here rather than read from the member, because the member
    records no such declaration - there is no sidecar beside ``predictions.parquet`` saying
    what it was handed. Recomputing is sound only because the rule has one definition:
    ``eligible_persistent_entities`` is the function ``prepare_panel_data`` itself calls, so
    this cannot answer a question the builder would answer differently. It is also cheap -
    a group-by over the panel per fold, against the 120 MB parquet read this function is
    already doing per member - and it runs for no member of any other config.
    """
    # Imported here for the same reason the `coverage` import below is: `coverage` imports
    # from this module, so a top-level import of either closes a cycle.
    from case_studies.utils.coverage import _normalize_time
    from case_studies.utils.persistent_panel import (
        PERSISTENT_PANEL_MODELS,
        eligible_persistent_entities,
    )

    if family != "latent_factors" or config not in PERSISTENT_PANEL_MODELS:
        return None
    if not spec_json:
        return None
    try:
        folds = json.loads(spec_json).get("computation", {}).get("cv", {}).get("folds")
    except (TypeError, ValueError):
        return None
    if not isinstance(folds, list):
        return None

    session = _normalize_time(panel.get_column("session"))
    keys = panel.select(pl.col("entity").cast(pl.String)).with_columns(session.alias("session"))
    admitted: dict[int, list[str]] = {}
    for entry in folds:
        if not isinstance(entry, dict) or not isinstance(entry.get("fold"), int):
            continue
        start, end = entry.get("train_start"), entry.get("train_end")
        if not isinstance(start, str) or not isinstance(end, str):
            # A fold that does not declare its training window cannot be narrowed, and
            # guessing one would charge the member against a denominator nobody declared.
            return None
        window = keys.filter(
            (pl.col("session") >= _parse_declared(start))
            & (pl.col("session") <= _parse_declared(end))
        )
        if window.is_empty():
            return None
        admitted[int(entry["fold"])] = (
            eligible_persistent_entities(window, entity_col="entity", date_col="session")
            .get_column("entity")
            .to_list()
        )
    return admitted or None


def _parse_declared(stamp: str) -> datetime:
    """A spec's declared fold boundary as a naive datetime, matching ``_normalize_time``."""
    parsed = datetime.fromisoformat(stamp)
    return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed


def undercovered_prediction_members(
    root: Path,
    members: Iterable[str],
    *,
    case_study: str,
    minimum: float | None = None,
) -> dict[str, str]:
    """Which in-force members cover too little of the cross-section they were offered.

    The filter over :func:`measure_prediction_cross_sections`, which is where the numbers
    are. Kept as its own function because the threshold is the thing most callers want and
    because the reason strings it returns are what the refusal prints.

    ``full_coverage_prediction_sql`` above asks the same kind of question and asks it
    relatively: keep the rows whose ``ic_n_days`` ties the maximum for their family and
    label. A shortfall that moves every candidate the same way is invisible to it, and a
    shortfall along the *symbol* axis is invisible to it twice over, because
    ``ic_n_days`` counts decision dates. A family that scores every date for half the
    universe ties the maximum and ranks against families that scored all of it.

    This reads the artifacts instead: for each member, the delivered ``(symbol,
    timestamp)`` pairs against the ones its label declares, narrowed to the ones the
    feature panels offered so a family is charged for what it lost and not for what it
    was never given. Returns ``{hash: reason}`` for the members that fall short, empty
    when every member is whole.

    A member whose coverage cannot be evaluated is returned as short rather than passed,
    which is the rule ``coverage.py`` states about itself.
    """
    # Imported here, not at module scope: `coverage` imports `_first_present` and
    # `_is_finite` from this module, so a top-level import closes the cycle.
    from case_studies.utils.coverage import BACKTEST_COVERAGE_MINIMUM

    threshold = BACKTEST_COVERAGE_MINIMUM if minimum is None else minimum
    measured, unevaluable = measure_prediction_cross_sections(root, members, case_study=case_study)
    short = {
        phash: report.summary()
        for phash, report in measured.items()
        if report.accountable_coverage < threshold
    }
    short.update(unevaluable)
    return short


def measure_prediction_cross_sections(
    root: Path,
    members: Iterable[str],
    *,
    case_study: str,
) -> tuple[dict[str, CrossSectionReport], dict[str, str]]:
    """Every member's delivered cross-section against the one its label declares.

    Returns the reports for the members it could measure, and the reasons for the ones it
    could not. Separated from the threshold so the numbers reach
    :func:`record_prediction_admissibility`, which is the only place in the registry that
    holds the declared denominator. ``prediction_coverage.n_expected`` is not it: that column
    is built by the model family's own adapter from its own prepared fold inputs, so it says
    the model produced what it set out to produce. The two are true of the same predictions
    and disagree hard - on ``sp500_equity_option_analytics`` all 140 members this measurement
    rules inadmissible carry a ``prediction_coverage`` row reading ``complete`` with
    ``n_missing = 0``, whose ``n_expected`` is exactly the narrowed numerator here.

    Admitted members are measured too, and their reports are returned: a reader comparing
    126,458 against 248,460 needs the row for a member that passed as much as for one that
    did not.
    """
    # Imported here, not at module scope: `coverage` imports `_first_present` and
    # `_is_finite` from this module, so a top-level import closes the cycle.
    from case_studies.utils.coverage import (
        CoverageError,
        check_prediction_cross_section,
        feature_panel_keys,
    )

    root = Path(root)
    db_path = root / "run_log" / "registry.db"
    wanted = list(members)
    if not wanted or not db_path.is_file():
        return {}, {}

    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        placeholders = ",".join("?" * len(wanted))
        # `training_runs` has no `case_study` column - one registry belongs to one case
        # study, so the id is a property of the directory being read, not of a row in it.
        # Selecting it here raised `sqlite3.OperationalError` on every populated registry,
        # which is every call that had anything to check.
        rows = db.execute(
            f"""SELECT p.prediction_hash, p.split, t.label, t.family, t.config_name, t.spec_json
                FROM prediction_sets p JOIN training_runs t ON t.training_hash = p.training_hash
                WHERE p.prediction_hash IN ({placeholders})""",
            wanted,
        ).fetchall()

    # The feature panel, not each run's own `expected_prediction_keys`. A run declares,
    # before it fits, which keys its key builder says it can score, and it is tempting to
    # charge it against that instead - a sequence model cannot score a window shorter than
    # its lookback, so its declared set is narrower than the panel by construction, and
    # measuring it against the panel looks like charging it for the burn-in.
    #
    # Measured 2026-09-09 on sp500_equity_option_analytics/fwd_ret_10d/validation, against
    # the financial panel, one set per family:
    #
    #     family          accountable   never scored   partially scored
    #     linear             100.0%           1              302
    #     gbm                100.0%           1              302
    #     tabular_dl         100.0%           1              302
    #     latent_factors      91.3%          72              239
    #     deep_learning       65.0%         262               45
    #
    # A burn-in shows up as partially scored, and deep_learning has 45 of those. What it
    # has is 262 of 548 symbols it never scores at all: it ranks a 286-name cross-section
    # while gbm ranks 548, and a Sharpe from one is not comparable with a Sharpe from the
    # other. Charging it against its own declaration would report it whole - the declared
    # set is where the 262 symbols were dropped. It is not a stale artifact either: the
    # runs registered after #857 put the sequence window on the calendar read the same
    # 64.9%.
    #
    # `feature_panel_keys` in `coverage.py` is the one implementation of this denominator.
    # It used to be two: `load_backtest_predictions` measured coverage the same way at load
    # time, and this comment said the two must agree. They could not disagree, because that
    # function had no caller anywhere in the repository - the exclusions it applied read as
    # enforced and reached nothing. It was deleted; a second denominator is worth having only
    # if something runs it.
    # The fold axis comes from the run's own spec, not from the configuration.
    # `declared_sessions` reads the fold windows out of `config/setup.yaml`, which lists
    # every configured fold whatever the run was asked to do, so a run that fitted a subset
    # is charged for the folds it was never asked to produce. Every canonical run today
    # declares the full list and the narrowing is a no-op on them; the shape that makes it
    # fire is `splits[:MAX_FOLDS]`, which appears in three case studies. The symbol axis stays on the panel, so a family that lost names inside the
    # folds it ran is still charged for them.
    panel = feature_panel_keys(root)
    measured: dict[str, CrossSectionReport] = {}
    unevaluable: dict[str, str] = {}
    for phash, split, label, family, config, spec_json in rows:
        path = root / "run_log" / "predictions" / phash / "predictions.parquet"
        if not path.is_file():
            continue

        try:
            report = check_prediction_cross_section(
                pl.read_parquet(path),
                case_study,
                label,
                split=split,
                case_dir=root,
                input_panel=panel,
                # The holdout is one window and `declared_cross_section` labels it
                # `fold=None`, while a holdout spec carries the integer id the CV builder
                # gave it - 8 in etfs, 2 in sp500_options. Filtering on that empties the
                # cross-section and reports a complete holdout member unevaluable, which
                # drops it. The fold axis is a validation question; the holdout has one
                # window and the configuration is the only thing that declares it.
                folds=None if split == "holdout" else _declared_folds(spec_json),
                # The symbol axis, for the one family whose builder narrows it before the
                # fit. `None` for every other member, which leaves the check as it was.
                eligible_entities=(
                    None
                    if split == "holdout"
                    else _persistent_panel_entities(spec_json, panel, family=family, config=config)
                ),
                source=f"{family}/{config}",
            )
        except CoverageError as exc:
            unevaluable[phash] = f"coverage could not be evaluated: {exc}"
            continue
        measured[phash] = report
    return measured, unevaluable


def full_coverage_prediction_sql(
    prediction_set_alias: str = "p",
    training_run_alias: str = "t",
    prediction_metric_alias: str = "pm",
    population_subquery: str | None = None,
) -> str:
    """SQL clause retaining the maximum-coverage rows for each family and label.

    A model family can contain checkpoints evaluated on fewer decision dates than
    its peers even when no fold-level IC is NULL. Comparing or selecting those
    rows against full-coverage checkpoints changes the evaluation sample. The
    eligible surface therefore keeps rows whose ``ic_n_days`` equals the maximum
    for the same ``(split, family, label)``. When ``population_subquery`` is
    supplied, that maximum is computed within the explicitly locked prediction
    population rather than across retired identities.

    The surrounding query must join ``prediction_sets``, ``training_runs``, and
    ``prediction_metrics`` under the supplied aliases. The returned fragment
    begins with ``" AND "``. Any bound parameters required by
    ``population_subquery`` belong to the surrounding query.
    """
    population_clause = ""
    if population_subquery is not None:
        population_clause = f" AND p_full.prediction_hash IN ({population_subquery})"
    return f"""
        AND {prediction_metric_alias}.ic_n_days IS NOT NULL
        AND {prediction_metric_alias}.ic_n_days = (
            SELECT MAX(pm_full.ic_n_days)
            FROM prediction_sets p_full
            JOIN training_runs t_full
              ON p_full.training_hash = t_full.training_hash
            JOIN prediction_metrics pm_full
              ON p_full.prediction_hash = pm_full.prediction_hash
            WHERE p_full.split = {prediction_set_alias}.split
              AND t_full.family = {training_run_alias}.family
              AND t_full.label = {training_run_alias}.label
              {population_clause}
        )
    """


def canonical_coverage_days(
    case_study: str,
    label: str,
    split: str,
    prediction_hash: str,
    case_dir: Path | None = None,
) -> int | None:
    """Count of a prediction set's *scorable* decision dates inside ``canonical_window``.

    ``ic_n_days`` (stored in ``prediction_metrics`` at registration time) counts every
    date in the predictions parquet passed to the metrics function at write time, which
    can include dates outside the *current* ``canonical_window`` when a prediction set
    predates a CV-window change. Comparing that raw stored count across prediction sets
    whose underlying arrays differ only by such out-of-window dates makes
    ``full_coverage_prediction_sql`` exclude sets that cover the modeling window
    identically to their peers.

    This recomputes coverage directly from the prediction parquet, bounded to the
    window, for use by the ``coverage_window="canonical"`` path of
    ``resolve_best_predictions`` / ``resolve_best_backtest_runs``.

    **A date counts only where its cross-section could be scored.** This stands in for
    ``ic_n_days``, which counts the days ``cross_sectional_ic_series`` actually produced
    a coefficient for, and that function nulls a day unless at least ``min_obs``
    entities carry a finite prediction *and* a finite realized return. Counting rows
    present would let a prediction set with a row every day but two usable names on some
    of them read as full coverage while the raw path correctly discounted it, and the
    two counts are compared against each other. ``min_obs`` is 5 at every case-study
    call site.

    Returns ``None`` when the window, the parquet, or the columns the validity rule
    needs are unavailable. Callers must treat that as "cannot evaluate", not as zero
    coverage.
    """
    from case_studies.utils.cv_window import canonical_window
    from utils.paths import get_case_study_dir

    window = canonical_window(case_study, label, split=split)
    if window is None:
        return None
    if case_dir is None:
        case_dir = get_case_study_dir(case_study)
    path = case_dir / "run_log" / "predictions" / prediction_hash / "predictions.parquet"
    if not path.exists():
        return None
    cols = pl.scan_parquet(path).collect_schema().names()
    date_col = "timestamp" if "timestamp" in cols else ("date" if "date" in cols else None)
    if date_col is None:
        return None
    prediction_col = _first_present(cols, _PREDICTION_ALIASES)
    actual_col = _first_present(cols, _ACTUAL_ALIASES)
    if prediction_col is None or actual_col is None:
        return None
    entity_col = _first_present(cols, _ENTITY_ALIASES)

    selected = [date_col, prediction_col, actual_col] + ([entity_col] if entity_col else [])
    frame = pl.scan_parquet(path).select(selected).collect()
    if frame.is_empty():
        return 0

    # Cast to a calendar date BEFORE grouping, never after: on an intraday case
    # study the column is a Datetime and every timestamp within a day is distinct,
    # so grouping first counts one decision date many times and the count stops
    # being comparable to the daily `ic_n_days` it stands in for.
    if frame.schema[date_col] != pl.Date:
        try:
            frame = frame.with_columns(pl.col(date_col).cast(pl.Date))
        except pl.exceptions.PolarsError:
            # A column named `timestamp` that is not a calendar date is one more
            # "cannot evaluate" case, so it degrades to None like the others
            # rather than raising past every caller.
            return None

    lo, hi = window
    scorable = frame.filter(
        pl.col(date_col).is_between(lo, hi)
        & _is_finite(frame.schema[prediction_col], prediction_col)
        & _is_finite(frame.schema[actual_col], actual_col)
    )
    if scorable.is_empty():
        return 0
    # The cross-section is a set of entities, so a duplicated entity on a date does
    # not widen it. Without an entity column a row is the best available proxy.
    breadth = pl.col(entity_col).n_unique() if entity_col else pl.len()
    per_date = scorable.group_by(date_col).agg(
        breadth.alias("_breadth"),
        # A rank correlation is undefined where either side is the same value for
        # every name on the date, so breadth alone would count a date the IC series
        # returns null for. n_unique rather than a variance: Spearman ranks, and a
        # constant column has one rank whatever its spread.
        pl.col(prediction_col).n_unique().alias("_pred_levels"),
        pl.col(actual_col).n_unique().alias("_actual_levels"),
    )
    defined = (
        (per_date["_breadth"] >= IC_MIN_OBS)
        & (per_date["_pred_levels"] > 1)
        & (per_date["_actual_levels"] > 1)
    )
    return int(defined.sum())


def filter_active_model_rows(
    df: pl.DataFrame,
    case_study: str,
    *,
    family_col: str = "family",
) -> pl.DataFrame:
    if df.is_empty() or family_col not in df.columns:
        return df

    excluded = excluded_families(case_study)
    if not excluded:
        return df

    return df.filter(~pl.col(family_col).is_in(sorted(excluded)))


def declared_population_members(
    study,
    case_dir: Path,
    names: dict[str, str],
    *,
    produced: dict[str, int],
) -> tuple[dict[str, set[str]], list[str]]:
    """Resolve each family's declared population, or report that none is declared.

    Three states reach this, and `OfficialPopulation.one` reports two of them in the same
    words, which is why the decision lives here rather than in an exception handler.

    A registry that has published no population is not broken - a fixture, or a reader's clean
    clone. It is answered with a note, and the comparison downstream rests on catalog
    admissibility, which is a weaker claim than a declared population but a statable one. The
    resolver is not asked at all: on a registry whose schema predates the mechanism it raises
    ``sqlite3.OperationalError: no such table``, so tolerating that state means not entering it.

    A registry that has published populations and cannot resolve this name has a broken
    lineage. That refuses when the family has registered rows, because comparing them would
    report a family no declaration covers; where the family produced nothing it is only a note,
    since there is nothing yet to be undeclared.

    A notebook naming its *real* populations will not find them in a CI fixture. The seeded
    registries publish under a ``{cs}-fixture-{family}-validation-v1`` prefix, which they have
    to: ``OfficialPopulation.create`` matches on the member list, so a modelling notebook
    publishing its own newly-fitted hashes under a name the fixture had frozen is refused. Point
    the notebook's population-name parameters at the fixture names in ``tests/overrides.yaml``
    rather than working around it here - the name a notebook declares is the one it means in
    production, and the fixture is the thing that differs.

    Returns the resolved members per family and the notes to print.
    """
    from case_studies.research import OfficialPopulation, published_population_names_at

    published = published_population_names_at(case_dir)
    if not published:
        return {}, [
            f"{case_dir} publishes no official populations, so nothing is checked against a "
            "declaration: every comparison rests on catalog admissibility alone."
        ]

    members: dict[str, set[str]] = {}
    notes: list[str] = []
    for family, name in names.items():
        try:
            members[family] = set(OfficialPopulation.one(study, name=name).members)
        except (ValueError, FileNotFoundError) as error:
            if produced.get(family):
                msg = (
                    f"{family} has {produced[family]} registered prediction sets but its "
                    f"declared population {name} does not resolve ({error}). This registry "
                    f"publishes {len(published)} population name(s), so the declaration is "
                    "missing rather than unused. Comparing them would report a family no "
                    "declaration covers. Republish the population, or name the one in force."
                )
                raise RuntimeError(msg) from error
            notes.append(f"no current official population for {family} ({name}): {error}")
    return members, notes


_STRATEGY_ANALYSIS_TABLES = ("backtest_runs", "cohort_metrics", "backtest_paired_metrics")


def strategy_input_counts(case_dir: Path) -> dict[str, int]:
    """Row counts for the three tables a strategy-analysis notebook reads.

    ``backtest_runs`` is what the backtesting stages register. ``cohort_metrics`` and
    ``backtest_paired_metrics`` are *derived* from those runs, and until recently only
    ``cme_futures/17`` derived them inside its own case study - everywhere else they existed
    solely because ``20_strategy_synthesis/01_aggregate_synthesis.py`` had been run, which makes
    a case study depend upward on the chapter that aggregates it.

    The distinction the caller needs is between "no runs to analyse" and "runs exist but nothing
    has derived from them". The first is a refusal: every figure and gate downstream is computed
    from backtest runs, so with none registered the notebook does not produce a weaker answer,
    it produces an empty one that reads like a finished analysis. The second is work to do, and
    both producers are already case-study-scoped functions.

    A missing registry or a missing table counts as zero, which is the ordinary state of a clean
    clone, and is reported rather than raised so the caller decides what it means.
    """
    import sqlite3

    db_path = Path(case_dir) / "run_log" / "registry.db"
    if not db_path.is_file():
        return dict.fromkeys(_STRATEGY_ANALYSIS_TABLES, 0)
    counts: dict[str, int] = {}
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        present = {
            row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table in _STRATEGY_ANALYSIS_TABLES:
            counts[table] = (
                db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] if table in present else 0
            )
    return counts


_DERIVED_TABLE_REFERENCES: dict[str, tuple[str, ...]] = {
    "cohort_metrics": ("leader_hash",),
    "backtest_paired_metrics": ("challenger_hash", "benchmark_hash"),
}


def derived_tables_off_canonical_universe(case_dir: Path, universe_filter: str | None) -> set[str]:
    """Derived tables holding rows that were selected under a different universe.

    A row count answers "has this been populated", which is not the question a rerun needs.
    ``cohort_metrics`` and ``backtest_paired_metrics`` are written from a *selection*, and a
    table populated by an earlier run that made a different selection is fully populated and
    wrong. Nothing in either table records which selection produced it, so it is recovered
    from what the rows point at: every referenced ``backtest_runs`` row carries its universe in
    ``spec_json``, and a canonical table cannot reference a run outside the canonical universe.

    Only hashes that name a ``backtest_runs`` row are judged. ``backtest_paired_metrics``
    carries no FK on ``benchmark_hash`` and its equal-weight side is a synthetic
    ``side_ew:<cs>:<label>`` identifier that is deliberately not a run; an identifier that
    names no run cannot be evidence of a run outside the universe. Treating an absent hash as
    ``"full"`` instead would report the paired table stale on every run forever.

    ``cohort_metrics`` records only ``leader_hash``, so a cohort whose leader is canonical but
    whose membership was drawn from a wider universe reads as clean here. Its trial counts,
    DSR and PBO are computed over that wider membership, and ``k_variants`` counts the members
    that had usable return series rather than the members selected, so it cannot stand in for
    the missing selection identity. Closing that gap needs the selection persisted on the row.

    Returns the table names to rebuild. An unpinned case study passes ``None`` and gets the
    empty set, because there is no canonical universe for a row to be outside of.
    """
    if universe_filter is None:
        return set()

    from case_studies.utils.backtest_explorer import _parse_spec
    from case_studies.utils.backtest_presets import strategy_view

    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.is_file():
        return set()

    stale: set[str] = set()
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        present = {
            row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "backtest_runs" not in present:
            return set()
        universes = {
            row[0]: (
                strategy_view(_parse_spec(row[1]) or {}).get("signal", {}).get("universe_filter")
                or "full"
            )
            for row in db.execute("SELECT backtest_hash, spec_json FROM backtest_runs")
        }
        for table, columns in _DERIVED_TABLE_REFERENCES.items():
            if table not in present:
                continue
            for column in columns:
                referenced = [
                    row[0]
                    for row in db.execute(f"SELECT {column} FROM {table}")  # noqa: S608
                    if row[0] is not None
                ]
                if any(universes[h] != universe_filter for h in referenced if h in universes):
                    stale.add(table)
    return stale
