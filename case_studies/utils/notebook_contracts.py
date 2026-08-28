from __future__ import annotations

import sqlite3
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
