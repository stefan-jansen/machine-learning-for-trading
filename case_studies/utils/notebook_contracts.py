from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

import polars as pl

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


_DEGENERATE_SUBQUERY = "SELECT prediction_hash FROM fold_metrics WHERE ic IS NULL"


def degenerate_prediction_sql(prediction_hash_column: str = "p.prediction_hash") -> str:
    """SQL clause excluding prediction sets with any constant-prediction fold.

    When a regularized linear model (LASSO / ElasticNet at high ``alpha_frac``)
    shrinks every coefficient to zero on a fold, that fold's predictions are
    constant and its IC is undefined — stored as NULL in ``fold_metrics.ic``.
    The pooled daily IC is then computed over the surviving folds only, which
    biases it (typically upward) and is not a valid model result. Such
    prediction sets must never be selected for backtesting or any follow-on
    leaderboard.

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
        coverage = {
            row[0]: (row[1], row[2])
            for row in db.execute(
                "SELECT prediction_hash, status, n_folds_expected FROM prediction_coverage"
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
        status, expected = coverage[member]
        actual = folds.get(member, 0)
        artifact = predictions_dir / member / "predictions.parquet"
        if status != "complete":
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
    study, case_dir: Path | None = None
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
            f"{len(uncovered):,} of {len(members):,} members carry no prediction_coverage row, "
            "so their completeness is unevidenced rather than established. They are ranked; "
            "the gap is in the registry, not in the run."
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
    short = undercovered_prediction_members(
        root, [member for member in members if member not in reduced], case_study=study.case_study
    )
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
    if not members:
        raise RuntimeError(
            f"every member of the populations in force at {root} was dropped for incomplete "
            f"cross-sectional coverage ({len(short)} of them). There is nothing left to rank, "
            "and ranking the survivors of a universe filter against each other would not be a "
            f"comparison. Reasons:\n" + "\n".join(sorted(short.values()))
        )
    return members, notes


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


def undercovered_prediction_members(
    root: Path,
    members: Iterable[str],
    *,
    case_study: str,
    minimum: float | None = None,
) -> dict[str, str]:
    """Which in-force members cover too little of the cross-section they were offered.

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
    from case_studies.utils.coverage import (
        BACKTEST_COVERAGE_MINIMUM,
        CoverageError,
        check_prediction_cross_section,
        feature_panel_keys,
    )

    threshold = BACKTEST_COVERAGE_MINIMUM if minimum is None else minimum
    root = Path(root)
    db_path = root / "run_log" / "registry.db"
    wanted = list(members)
    if not wanted or not db_path.is_file():
        return {}

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
    # This is the denominator `load_backtest_predictions` already uses for the same
    # question, and the two must agree or the pool admits members the backtest then
    # refuses, which surfaces as "no rankable validation backtests" three stages later.
    # The fold axis comes from the run's own spec, not from the configuration.
    # `declared_sessions` reads the fold windows out of `config/setup.yaml`, which lists
    # every configured fold whatever the run was asked to do, so a run that fitted a subset
    # is charged for the folds it was never asked to produce. Every canonical run today
    # declares the full list and the narrowing is a no-op on them; the shape that makes it
    # fire is `splits[:MAX_FOLDS]`, which ml4t/agent-workspace#1076 finds in three case
    # studies. The symbol axis stays on the panel, so a family that lost names inside the
    # folds it ran is still charged for them.
    panel = feature_panel_keys(root)
    short: dict[str, str] = {}
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
                source=f"{family}/{config}",
            )
        except CoverageError as exc:
            short[phash] = f"coverage could not be evaluated: {exc}"
            continue
        if report.accountable_coverage < threshold:
            short[phash] = report.summary()
    return short


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
