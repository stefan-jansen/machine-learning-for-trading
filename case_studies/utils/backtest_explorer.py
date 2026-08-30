"""Reader-facing API for exploring backtest results from the registry.

Usage::

    from case_studies.utils.backtest_explorer import BacktestExplorer

    explorer = BacktestExplorer("etfs")
    explorer.summary()
    explorer.best(stage="signal", top_n=5)
    explorer.compare_allocators()
    explorer.inspect("backtest_hash_abc")
    explorer.progression("prediction_hash_xyz")
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from case_studies.utils.backtest_presets import cost_view, strategy_view
from case_studies.utils.notebook_contracts import (
    excluded_family_sql,
    filter_active_model_rows,
    full_coverage_prediction_sql,
)
from case_studies.utils.uncertainty import cohort_member_digest

# Sentinel distinguishing "no filter" from "match exit_at_max_days IS NULL".
_UNSET = object()

# `best` bounds its read with a SQL LIMIT, so counting a whole cohort means asking for
# more rows than any cohort holds rather than for no limit at all.
_UNBOUNDED_COHORT = 1_000_000

# Canonical schema for BacktestExplorer.best() output. Used to construct
# schema-stable empty DataFrames so downstream `.select("source", ...)`
# surfaces "(no matching rows)" instead of a cryptic ColumnNotFoundError.
_BEST_SCHEMA: dict[str, pl.DataType] = {
    "backtest_hash": pl.Utf8,
    "prediction_hash": pl.Utf8,
    "source": pl.Utf8,
    "family": pl.Utf8,
    "config_name": pl.Utf8,
    "label": pl.Utf8,
    "signal_method": pl.Utf8,
    "universe_filter": pl.Utf8,
    "exit_at_max_days": pl.Int64,
    "sharpe": pl.Float64,
    "cagr": pl.Float64,
    "max_drawdown": pl.Float64,
    "total_return": pl.Float64,
    "volatility": pl.Float64,
    "ic_mean": pl.Float64,
    "ic_mean_daily": pl.Float64,
    "ic_ci_lo": pl.Float64,
    "ic_ci_hi": pl.Float64,
    "ic_n_days": pl.Float64,
}

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class BacktestDetail:
    """Full detail for a single backtest run."""

    backtest_hash: str
    prediction_hash: str
    stage: str | None
    spec: dict
    metrics: dict[str, float]
    daily_returns_path: Path | None
    trades_path: Path | None
    weights_path: Path | None
    source: str | None = None


# ---------------------------------------------------------------------------
# Spec-string parsing helpers
# ---------------------------------------------------------------------------


def _parse_spec(spec_str: str | None) -> dict | None:
    """Return the parsed JSON spec as a dict, or None if not parseable.

    Failure modes that return None:
      1. ``spec_str is None`` (NULL in the registry)
      2. ``spec_str == ""`` (empty string)
      3. ``spec_str`` is malformed JSON (truncated writes)
      4. The parsed value is not a dict (e.g. ``"42"`` → int, ``"null"``
         → None, ``"[1, 2]"`` → list) — callers that feed this into
         ``strategy_view`` or ``cost_view`` rely on dict-shaped input

    Callers that don't care about distinguishing "not parseable" from
    "empty dict" can write ``_parse_spec(s) or {}`` to get a
    guaranteed-dict. Callers that DO need to distinguish (e.g.
    ``compare_allocators`` which returns "unknown" for unparseable
    rows and "equal_weight" for a successfully-parsed spec missing
    the allocation method) check ``is None`` explicitly.

    Replaces five copies of the same parse-and-default pattern.
    """
    if not spec_str:
        return None
    # Note: only catching JSONDecodeError here, not TypeError. The
    # original sites caught both defensively, but json.loads only raises
    # TypeError on non-string input, which the ``if not spec_str`` guard
    # plus the str|None type annotation make unreachable from well-typed
    # callers. The ``(A, B)`` tuple form is also what ruff 0.15 on a
    # py314 target rewrites to the Python-2-style comma form, which then
    # fails on Python 3.12 CI.
    try:
        value = json.loads(spec_str)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Explorer
# ---------------------------------------------------------------------------


class BacktestExplorer:
    """High-level reader API for querying backtest results.

    All data is read from ``registry.db`` — no JSON files needed.
    """

    def __init__(self, case_study: str, *, case_dir: Path | None = None):
        from utils.paths import get_case_study_dir

        self.case_study = case_study
        self.case_dir = case_dir or get_case_study_dir(case_study)
        self._db_path = self.case_dir / "run_log" / "registry.db"
        if not self._db_path.exists():
            raise FileNotFoundError(f"No registry.db found for '{case_study}' at {self._db_path}")

    # -- helpers --

    def _query(self, sql: str, params: tuple = ()) -> pl.DataFrame:
        db = sqlite3.connect(str(self._db_path))
        db.row_factory = sqlite3.Row
        try:
            rows = db.execute(sql, params).fetchall()
            if not rows:
                return pl.DataFrame()
            return pl.DataFrame([dict(r) for r in rows])
        finally:
            db.close()

    def _backtest_dir(self, b_hash: str) -> Path:
        return self.case_dir / "run_log" / "backtest" / b_hash

    def _filter_active_models(self, df: pl.DataFrame) -> pl.DataFrame:
        return filter_active_model_rows(df, self.case_study)

    # -----------------------------------------------------------------
    # summary: what has been run?
    # -----------------------------------------------------------------

    def summary(self) -> dict[str, int]:
        """Count of backtest runs by stage.

        Returns
        -------
        dict[str, int]
            e.g. {"signal": 3336, "allocation": 329, ...}
        """
        df = self._query(
            "SELECT stage, COUNT(*) AS n FROM backtest_runs GROUP BY stage ORDER BY n DESC"
        )
        if df.is_empty():
            return {}
        return dict(zip(df["stage"].to_list(), df["n"].to_list(), strict=False))

    # -----------------------------------------------------------------
    # best: top backtests at a stage
    # -----------------------------------------------------------------

    def _refuse_if_the_coverage_bar_emptied_it(
        self,
        stage: str,
        label: str | None,
        prediction_hashes: tuple[str, ...] | list[str] | None,
    ) -> None:
        """Raise when backtests exist at this stage but none of them is a complete run.

        `full_coverage_prediction_sql` keeps only rows whose `ic_n_days` equals the maximum
        for their `(split, family, label)`, and that maximum is the population's, not the
        backtested subset's. That is deliberate: the bar is how a *complete* run is
        identified - `reference/CASE_STUDY_PIPELINE.md` section 10, "a result counts only
        when `n_null == 0` and `ic_n_days == max`" - and it exists because an incomplete run
        manufactures a false leader. Computing the maximum over only the rows that happen to
        have backtests would let an incomplete run set its own bar and win whenever no
        complete run was backtested, which is the exact failure the clause was written for.

        So an empty result here is a true statement: no complete run in this population has
        a backtest at this stage. It is not, however, the same statement as "there are no
        backtests", and returning an empty frame says the second. Under a preview reduction
        the two diverge - etfs measured a scoped population of 6 predictions and 1 backtest
        whose maximum coverage belonged to a prediction nothing backtested - and a notebook
        that reads the empty frame as "nothing ran" reports on a comparison that was never
        possible. Refuse instead, and name the gap.
        """
        # Every filter the real query applies EXCEPT the coverage clause, so an empty
        # result there and a non-empty result here isolates the coverage clause as the
        # only thing that removed the rows. Dropping one - the family exclusions, or the
        # zero-trade filter - would report a run excluded for another reason as a coverage
        # failure and send the reader after the wrong thing. `_filter_active_models` runs
        # after this point in `best` and cannot be the cause of an empty query result.
        rows = self._query(
            f"""
            SELECT t.family, t.label, pm.ic_n_days, p.prediction_hash
            FROM backtest_runs b
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            LEFT JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
            LEFT JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
            WHERE b.stage = ?
              AND p.split != 'holdout'
              {excluded_family_sql(self.case_study, "t.family")[0]}
              AND bm.sharpe IS NOT NULL
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
              {" AND t.label = ?" if label else ""}
              {
                " AND p.prediction_hash IN (" + ", ".join("?" for _ in prediction_hashes) + ")"
                if prediction_hashes
                else ""
            }
            """,
            (
                stage,
                *excluded_family_sql(self.case_study, "t.family")[1],
                *([label] if label else []),
                *(prediction_hashes or ()),
            ),
        )
        if rows.is_empty():
            return
        detail = ", ".join(
            f"{family}/{row_label} {hash_} covers {days if days is not None else 'no'} days"
            for family, row_label, days, hash_ in rows.rows()
        )
        raise RuntimeError(
            f"every backtest at stage {stage!r} sits below its family's coverage bar, so the "
            f"complete-run filter admits none of them: {detail}. The bar is the population's "
            f"maximum ic_n_days, which is what makes a run complete - an incomplete run "
            f"compared as complete manufactures a false leader. Backtest a maximum-coverage "
            f"prediction, or say in the notebook that this population supports no comparison."
        )

    def best(
        self,
        stage: str = "signal",
        *,
        top_n: int = 10,
        metric: str = "sharpe",
        label: str | None = None,
        prediction_hashes: tuple[str, ...] | list[str] | None = None,
    ) -> pl.DataFrame:
        """Top-N backtests at a given stage, ranked by ``metric``.

        Returns
        -------
        pl.DataFrame
            Columns: backtest_hash, prediction_hash, source, family,
            config_name, label, signal_method, sharpe, cagr, max_drawdown,
            total_return, volatility, ic_mean
        """
        filter_sql = ""
        filter_params: list[str] = []
        coverage_params: list[str] = []
        coverage_sql = full_coverage_prediction_sql("p", "t", "pm")
        if label:
            filter_sql += " AND t.label = ?"
            filter_params.append(label)
        if prediction_hashes:
            placeholders = ", ".join("?" for _ in prediction_hashes)
            filter_sql += f" AND p.prediction_hash IN ({placeholders})"
            filter_params.extend(prediction_hashes)
            coverage_sql = full_coverage_prediction_sql(
                "p",
                "t",
                "pm",
                population_subquery="SELECT value FROM json_each(?)",
            )
            coverage_params.append(json.dumps(list(prediction_hashes)))

        df = self._query(
            f"""
            SELECT
                b.backtest_hash,
                b.prediction_hash,
                b.spec_json,
                b.stage,
                t.family,
                t.config_name,
                t.label,
                bm.sharpe,
                bm.cagr,
                bm.max_drawdown,
                bm.total_return,
                bm.volatility,
                pm.ic_mean,
                pm.ic_mean_daily,
                pm.ic_ci_lo,
                pm.ic_ci_hi,
                pm.ic_n_days
            FROM backtest_runs b
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            LEFT JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
            LEFT JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
            WHERE b.stage = ?
              AND p.split != 'holdout'
              {excluded_family_sql(self.case_study, "t.family")[0]}
              {coverage_sql}
              AND bm.sharpe IS NOT NULL
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
              {filter_sql}
            ORDER BY bm.sharpe DESC
            LIMIT ?
            """,
            (
                stage,
                *excluded_family_sql(self.case_study, "t.family")[1],
                *coverage_params,
                *filter_params,
                top_n,
            ),
        )
        if df.is_empty():
            self._refuse_if_the_coverage_bar_emptied_it(stage, label, prediction_hashes)
            return pl.DataFrame(schema=_BEST_SCHEMA)
        df = self._filter_active_models(df)
        if df.is_empty():
            return pl.DataFrame(schema=_BEST_SCHEMA)

        # Build source and extract signal_method from spec
        df = df.with_columns(
            (
                pl.col("family") + pl.lit("/") + pl.col("config_name").fill_null(pl.lit("default"))
            ).alias("source"),
        )

        # Extract signal_method, universe_filter, exit_at_max_days from spec_json.
        # The (universe_filter, exit_at_max_days) pair identifies the execution
        # regime in the O'Donovan-Yu (2025) cost-mitigation cascade for
        # sp500_options:
        #   - Rung-1 (naive round-trip): universe_filter="full",  exit_at_max_days=10
        #   - Rung-2 (HTM, full):        universe_filter="full",  exit_at_max_days=None
        #   - Rung-3 (HTM, liquid q20):  universe_filter="liquid", exit_at_max_days=None
        # Both Rung-1 and Rung-2 carry universe_filter="full"; pinning the
        # chapter-wide rank-1 to the HTM baseline therefore needs both fields.
        # Other case studies default to ("full", None) and are unaffected.
        parsed = [_parse_spec(s) or {} for s in df["spec_json"].to_list()]
        methods = [strategy_view(sp).get("signal", {}).get("method", "") for sp in parsed]
        universe_filters = [
            strategy_view(sp).get("signal", {}).get("universe_filter", "full") or "full"
            for sp in parsed
        ]
        exit_at_max_days = [
            strategy_view(sp).get("signal", {}).get("exit_at_max_days") for sp in parsed
        ]
        df = df.with_columns(
            pl.Series("signal_method", methods),
            pl.Series("universe_filter", universe_filters),
            pl.Series("exit_at_max_days", exit_at_max_days, dtype=pl.Int64),
        )

        return df.select(
            "backtest_hash",
            "prediction_hash",
            "source",
            "family",
            "config_name",
            "label",
            "signal_method",
            "universe_filter",
            "exit_at_max_days",
            "sharpe",
            "cagr",
            "max_drawdown",
            "total_return",
            "volatility",
            "ic_mean",
            "ic_mean_daily",
            "ic_ci_lo",
            "ic_ci_hi",
            "ic_n_days",
        )

    # -----------------------------------------------------------------
    # compare_families: model family comparison at a stage
    # -----------------------------------------------------------------

    def compare_families(
        self,
        stage: str = "signal",
        *,
        prediction_hashes: tuple[str, ...] | list[str] | None = None,
        exclude_insolvent: bool = False,
    ) -> pl.DataFrame:
        """Compare model families by backtest Sharpe at a given stage.

        Parameters
        ----------
        prediction_hashes : sequence of str, optional
            Restrict the comparison to these prediction identities. The coverage bar is
            then taken within them, matching ``best`` and ``compare_allocators``: a family
            whose full-coverage rows are outside the population must be compared on its own
            terms rather than dropped against a bar nothing in the population can meet.
            An empty sequence is a population with no members and compares nothing.
        exclude_insolvent : bool, default False
            Compute the Sharpe statistics over the runs that stayed solvent,
            and report how many did not in an added ``insolvent`` column. A run
            whose ``max_drawdown`` reached -100% took its equity to zero or
            past it; there is no capital left to earn a later return, so every
            subsequent period is arithmetic on zero or a negative balance and
            the Sharpe is a number without a meaning. Leaving those in distorts
            a median and can top a maximum.

            The count is reported rather than the runs silently dropped,
            because dropping them alone would rank a family by its survivors: a
            family that went to zero in nine runs out of ten would show the
            Sharpe of the tenth and nothing to say so. ``n`` counts the solvent
            runs the statistics are computed over and ``insolvent`` the rest,
            so a reader can see what the median is conditional on. A family
            with no solvent run has null statistics and sorts last.

            A run with no recorded drawdown is counted under ``unknown`` and
            kept out of the statistics. It cannot be shown to have stayed
            solvent, but neither did it fail: putting it under ``insolvent``
            would report a bankruptcy that was never measured. ``n``,
            ``insolvent`` and ``unknown`` together are every run of the family:
            the query does not drop a run for having no Sharpe when this is on,
            because a bankrupt run can carry a null one and dropping it would take
            a wiped-out family off the table entirely. The Sharpe statistics skip
            any solvent run without a recorded Sharpe, so they can rest on fewer
            than ``n`` runs: these columns count runs, not measurements.

            Off by default, so a caller that has already reported on the full
            population keeps reporting on it, with the same columns as before.

        Returns
        -------
        pl.DataFrame
            Columns: family, n, sharpe_median, sharpe_max, sharpe_q75,
            pct_positive - and ``insolvent``, ``unknown`` when
            ``exclude_insolvent``.
        """
        scope_sql = ""
        scope_params: tuple[str, ...] = ()
        coverage_sql = full_coverage_prediction_sql("p", "t", "pm")
        if prediction_hashes is not None:
            placeholders = ", ".join("?" for _ in prediction_hashes)
            scope_sql = f" AND p.prediction_hash IN ({placeholders})"
            coverage_sql = full_coverage_prediction_sql(
                "p",
                "t",
                "pm",
                population_subquery="SELECT value FROM json_each(?)",
            )
            scope_params = (json.dumps(list(prediction_hashes)), *prediction_hashes)
        # A run that went bankrupt can carry a null Sharpe, and dropping those in SQL would
        # remove it before `insolvent` counts it - so a family that went bankrupt in every
        # run would leave the comparison entirely, which is the survivorship bias this
        # option exists to expose. Under the flag the rows are kept, and the Sharpe
        # aggregates skip the nulls themselves.
        sharpe_sql = "" if exclude_insolvent else "AND bm.sharpe IS NOT NULL"
        df = self._query(
            f"""
            SELECT
                t.family,
                bm.sharpe,
                bm.max_drawdown
            FROM backtest_metrics bm
            JOIN backtest_runs b ON bm.backtest_hash = b.backtest_hash
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
            WHERE b.stage = ?
              AND p.split != 'holdout'
              {excluded_family_sql(self.case_study, "t.family")[0]}
              {coverage_sql}
              {sharpe_sql}
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
              {scope_sql}
            """,
            (stage, *excluded_family_sql(self.case_study, "t.family")[1], *scope_params),
        )
        if df.is_empty():
            return df
        df = self._filter_active_models(df)
        if df.is_empty():
            return df

        if not exclude_insolvent:
            return (
                df.group_by("family")
                .agg(
                    n=pl.len(),
                    sharpe_median=pl.col("sharpe").median(),
                    sharpe_max=pl.col("sharpe").max(),
                    sharpe_q75=pl.col("sharpe").quantile(0.75),
                    pct_positive=((pl.col("sharpe") > 0).sum() / pl.len() * 100),
                )
                .sort("sharpe_median", descending=True)
            )

        # Three outcomes, counted separately, because a run with no recorded drawdown
        # is not a bankruptcy: reporting it under `insolvent` would put a failure on
        # the record that was never measured. Each comparison is null for that run, so
        # the sums skip it and it lands only in `unknown`. n + insolvent + unknown is
        # every run of the family.
        solvent = pl.col("max_drawdown") > -1.0
        # Filtering needs a mask with no nulls; the counts above do not, and take the
        # null-skipping behaviour instead.
        solvent_mask = solvent.fill_null(False)
        return (
            df.group_by("family")
            .agg(
                n=solvent.sum(),
                insolvent=(pl.col("max_drawdown") <= -1.0).sum(),
                unknown=pl.col("max_drawdown").is_null().sum(),
                sharpe_median=pl.col("sharpe").filter(solvent_mask).median(),
                sharpe_max=pl.col("sharpe").filter(solvent_mask).max(),
                sharpe_q75=pl.col("sharpe").filter(solvent_mask).quantile(0.75),
                # `mean` rather than `sum() / len()`: a family whose runs all went
                # insolvent divides by zero, and mean over an empty selection is null,
                # which is what "no solvent run to report a rate over" means.
                pct_positive=(pl.col("sharpe").filter(solvent_mask) > 0).mean() * 100,
            )
            .sort("sharpe_median", descending=True, nulls_last=True)
        )

    # -----------------------------------------------------------------
    # compare_allocators: allocation method comparison
    # -----------------------------------------------------------------

    def compare_allocators(
        self,
        *,
        prediction_hash: str | None = None,
        stages: tuple[str, ...] = ("allocation",),
        label: str | None = None,
        prediction_hashes: tuple[str, ...] | list[str] | None = None,
    ) -> pl.DataFrame:
        """Compare allocation methods from the allocation stage.

        Parameters
        ----------
        prediction_hash : str, optional
            If provided, restrict the comparison to backtests carrying this
            prediction_hash (full or prefix match). Used by Ch20 to align the
            allocator-heatmap pool to the spine rank-1 carrier so Figure 20.7
            and Table 20.6 read off the same prediction.
        stages : tuple of str, default ``("allocation",)``
            Which backtest stages to include. Ch20 Figure 20.14 / Table 20.6
            isolate the allocator layer and read off ``"allocation"`` only; the
            risk overlay (ch19) is a downstream layer covered in §20.7, so
            folding it in here would credit the allocator with the overlay's
            work. Pass ``"risk_overlay"`` explicitly only for cross-stage views.
        label : str, optional
            Restrict the comparison to one target label. Publication notebooks
            should pass their active label so accumulated sibling-label rows do
            not change the reported allocator ranking.
        prediction_hashes : sequence of str, optional
            Restrict the comparison to prediction sets advanced by the current
            funnel. This excludes accumulated rows from earlier sweeps.

        Returns
        -------
        pl.DataFrame
            Columns: allocator, n, avg_sharpe, best_sharpe, avg_max_dd
        """
        if not stages:
            return pl.DataFrame()
        placeholders = ", ".join("?" for _ in stages)
        coverage_params: tuple[str, ...] = ()
        coverage_sql = full_coverage_prediction_sql("p", "t", "pm")
        if prediction_hashes:
            coverage_sql = full_coverage_prediction_sql(
                "p",
                "t",
                "pm",
                population_subquery="SELECT value FROM json_each(?)",
            )
            coverage_params = (json.dumps(list(prediction_hashes)),)
        sql = f"""
            SELECT
                b.spec_json,
                t.family,
                t.config_name,
                bm.sharpe,
                bm.max_drawdown
            FROM backtest_runs b
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
            JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
            WHERE b.stage IN ({placeholders})
              AND p.split != 'holdout'
              {excluded_family_sql(self.case_study, "t.family")[0]}
              {coverage_sql}
              AND bm.sharpe IS NOT NULL
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
        """
        params: tuple = (
            *stages,
            *excluded_family_sql(self.case_study, "t.family")[1],
            *coverage_params,
        )
        if prediction_hash:
            sql += " AND b.prediction_hash LIKE ?"
            params = (*params, prediction_hash + "%")
        if label:
            sql += " AND t.label = ?"
            params = (*params, label)
        if prediction_hashes:
            hash_placeholders = ", ".join("?" for _ in prediction_hashes)
            sql += f" AND p.prediction_hash IN ({hash_placeholders})"
            params = (*params, *prediction_hashes)
        df = self._query(sql, params)
        if df.is_empty():
            return df
        df = self._filter_active_models(df)
        if df.is_empty():
            return df

        # Extract allocator from spec_json. Unparseable spec → "unknown";
        # missing allocation key → "unknown" so risk_overlay rows whose
        # spec carries only the risk overlay (no explicit allocator) are
        # not silently bucketed under equal_weight (Ch20 Figure 20.7 / Table
        # 20.6 pin allocator-method semantics to the spec, not to an engine
        # default).
        def _allocator_from(s: str | None) -> str:
            spec = _parse_spec(s)
            if spec is None:
                return "unknown"
            method = strategy_view(spec).get("allocation", {}).get("method")
            return method if method else "unknown"

        allocators = [_allocator_from(s) for s in df["spec_json"].to_list()]
        df = df.with_columns(pl.Series("allocator", allocators))
        df = df.filter(pl.col("allocator") != "unknown")
        if df.is_empty():
            return df

        return (
            df.group_by("allocator")
            .agg(
                n=pl.len(),
                avg_sharpe=pl.col("sharpe").mean(),
                best_sharpe=pl.col("sharpe").max(),
                avg_max_dd=pl.col("max_drawdown").mean(),
            )
            .sort("avg_sharpe", descending=True)
        )

    # -----------------------------------------------------------------
    # inspect: full detail for one backtest
    # -----------------------------------------------------------------

    def inspect(self, backtest_hash: str) -> BacktestDetail:
        """Load full details for a single backtest run.

        Parameters
        ----------
        backtest_hash : str
            Full or prefix of the backtest hash.

        Returns
        -------
        BacktestDetail
        """
        # Support prefix matching
        df = self._query(
            """
            SELECT
                b.backtest_hash,
                b.prediction_hash,
                b.stage,
                b.spec_json,
                t.family,
                t.config_name
            FROM backtest_runs b
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            WHERE b.backtest_hash LIKE ?
            LIMIT 1
            """,
            (backtest_hash + "%",),
        )
        if df.is_empty():
            raise KeyError(f"No backtest found matching '{backtest_hash}'")

        row = df.row(0, named=True)
        b_hash = row["backtest_hash"]

        # Load metrics (wide format — each column is a metric)
        metrics_df = self._query(
            "SELECT * FROM backtest_metrics WHERE backtest_hash = ?",
            (b_hash,),
        )
        metrics = {}
        if not metrics_df.is_empty():
            row_dict = metrics_df.row(0, named=True)
            metrics = {
                k: v
                for k, v in row_dict.items()
                if k not in ("backtest_hash", "computed_at") and v is not None
            }

        # Parse spec
        spec = {}
        if row["spec_json"]:
            import contextlib

            with contextlib.suppress(json.JSONDecodeError, TypeError):
                spec = json.loads(row["spec_json"])

        # File paths
        bt_dir = self._backtest_dir(b_hash)
        returns_path = bt_dir / "daily_returns.parquet"
        trades_path = bt_dir / "trades.parquet"
        weights_path = bt_dir / "weights.parquet"

        source = None
        if row.get("family"):
            config = row.get("config_name") or "default"
            source = f"{row['family']}/{config}"

        return BacktestDetail(
            backtest_hash=b_hash,
            prediction_hash=row["prediction_hash"],
            stage=row["stage"],
            spec=spec,
            metrics=metrics,
            daily_returns_path=returns_path if returns_path.exists() else None,
            trades_path=trades_path if trades_path.exists() else None,
            weights_path=weights_path if weights_path.exists() else None,
            source=source,
        )

    # -----------------------------------------------------------------
    # progression: Sharpe across stages for a prediction
    # -----------------------------------------------------------------

    def progression(
        self,
        prediction_hash: str,
        *,
        universe_filter: str | None | object = _UNSET,
        exit_at_max_days: int | None | object = _UNSET,
    ) -> pl.DataFrame:
        """Show Sharpe progression across stages for a given prediction.

        Finds the best backtest at each stage for this prediction hash
        and shows how performance changes as allocation, costs, and risk
        overlays are added.

        Parameters
        ----------
        prediction_hash : str
            Prediction set to trace through the pipeline.
        universe_filter : str, None, or _UNSET, optional
            If set to a string, restrict to backtests whose
            ``strategy.signal.universe_filter`` matches (defaulting null
            spec entries to ``"full"`` so case studies without an explicit
            universe_filter still match). If left at ``_UNSET`` (default),
            no filter is applied. Used to scope sp500_options to its full
            vs. liquid execution regime.
        exit_at_max_days : int, None, or _UNSET, optional
            If set to ``None`` explicitly, restrict to backtests whose
            spec has no ``exit_at_max_days`` set (HTM regime). If set to
            an integer, match exactly. If left at ``_UNSET`` (default),
            no filter is applied. Together with ``universe_filter`` this
            pins sp500_options to a specific cascade rung across all
            stages, not just the signal stage.

        Returns
        -------
        pl.DataFrame
            Columns: stage, sharpe, cagr, max_drawdown, backtest_hash
        """
        clauses = [
            "b.prediction_hash = ?",
            "b.stage IS NOT NULL",
            "bm.sharpe IS NOT NULL",
            "(bm.num_trades IS NULL OR bm.num_trades > 0)",
        ]
        params: list[object] = [prediction_hash]
        if universe_filter is not _UNSET:
            clauses.append(
                "COALESCE(json_extract(b.spec_json, '$.strategy.signal.universe_filter'), 'full') = ?"
            )
            params.append(universe_filter)
        if exit_at_max_days is not _UNSET:
            if exit_at_max_days is None:
                clauses.append(
                    "json_extract(b.spec_json, '$.strategy.signal.exit_at_max_days') IS NULL"
                )
            else:
                clauses.append(
                    "json_extract(b.spec_json, '$.strategy.signal.exit_at_max_days') = ?"
                )
                params.append(exit_at_max_days)
        where_sql = " AND ".join(clauses)
        df = self._query(
            f"""
            SELECT
                b.stage,
                b.backtest_hash,
                bm.sharpe,
                bm.cagr,
                bm.max_drawdown
            FROM backtest_runs b
            JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
            WHERE {where_sql}
            ORDER BY bm.sharpe DESC
            """,
            tuple(params),
        )
        if df.is_empty():
            return df

        # Take best Sharpe per stage
        stage_order = {"signal": 0, "allocation": 1, "cost_sensitivity": 2, "risk_overlay": 3}
        best_per_stage = df.sort("sharpe", descending=True).group_by("stage").first()

        # Sort by pipeline order
        return (
            best_per_stage.with_columns(
                pl.col("stage").replace_strict(stage_order, default=99).alias("_order")
            )
            .sort("_order")
            .drop("_order")
        )

    # -----------------------------------------------------------------
    # deflated_sharpe: DSR from registry data
    # -----------------------------------------------------------------

    def deflated_sharpe(
        self,
        stage: str = "signal",
        *,
        top_n: int = 20,
        periods_per_year: int = 252,
        prediction_hashes: tuple[str, ...] | list[str] | None = None,
    ) -> pl.DataFrame:
        """Per-variant Sharpe with selection-bias DSR for family leaders.

        Per-variant PSR (single-strategy probability of skill, no
        multiple-testing correction) is computed on the fly from
        ``daily_returns.parquet``.

        Selection-bias DSR / RAS / Reality Check / PBO come from the
        persisted ``cohort_metrics`` table (cohort_type='family',
        leader_hash=backtest_hash).

        A deflated Sharpe is only meaningful against the trial count of the
        selection actually performed, so when ``prediction_hashes`` scopes
        the ranking these columns are reported only where the persisted
        cohort *is* the scoped cohort. ``k_variants`` is the count the stored
        correction was computed over and ``k_variants_scoped`` is the count
        in hand; where they disagree the selection-bias columns are null and
        both counts are kept, so a reader sees which population the missing
        correction belonged to rather than a leader that appears to need
        none. The counts are built by the same eligibility rule, since the
        scoped cohort is measured with ``best`` itself. Backward-compatible columns
        ``deflated_sharpe``, ``expected_max_sharpe``, ``dsr_pvalue``,
        ``significant`` carry the **effective-rank (ER) DSR** — the
        library maintainer's recommended default. ``dsr_mp`` and
        ``dsr_raw`` are surfaced alongside for sensitivity. Rows that
        are not the family leader for their ``(stage, label, family)``
        have NULL selection-bias columns.

        Returns
        -------
        pl.DataFrame
            Columns: source, sharpe, psr_pvalue, deflated_sharpe,
            expected_max_sharpe, dsr_pvalue, significant, is_best,
            dsr_mp, dsr_mp_pvalue, dsr_raw, dsr_raw_pvalue, k_variants,
            k_variants_scoped, n_trials_effective_er, n_trials_effective_mp, ras_leader,
            ras_pvalue, reality_check_pvalue, pbo, family, label.
        """
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        top = self.best(stage=stage, top_n=top_n, prediction_hashes=prediction_hashes)
        if top.is_empty():
            return pl.DataFrame()

        # The scoped cohort's size per (family, label), measured through `best` so that
        # coverage, excluded families and the tradeless-backtest rule are applied exactly
        # once. `top_n` bounds the rows displayed, not the cohort selected from, so this
        # is a second unbounded read rather than a group-by over `top`.
        scoped_k: dict[tuple[str, str], int] = {}
        scoped_digest: dict[tuple[str, str], str] = {}
        if prediction_hashes:
            cohort = self.best(
                stage=stage, top_n=_UNBOUNDED_COHORT, prediction_hashes=prediction_hashes
            )
            if not cohort.is_empty():
                grouped = cohort.group_by("family", "label").agg(
                    n=pl.len(), members=pl.col("backtest_hash")
                )
                for row in grouped.iter_rows(named=True):
                    key = (row["family"], row["label"])
                    scoped_k[key] = row["n"]
                    scoped_digest[key] = cohort_member_digest(row["members"])

        per_variant_psr: dict[str, float | None] = {}
        for row in top.iter_rows(named=True):
            b_hash = row["backtest_hash"]
            returns_path = self._backtest_dir(b_hash) / "daily_returns.parquet"
            if not returns_path.exists():
                per_variant_psr[b_hash] = None
                continue
            ret_df = pl.read_parquet(returns_path)
            if "daily_return" not in ret_df.columns:
                per_variant_psr[b_hash] = None
                continue
            arr = ret_df["daily_return"].to_numpy()
            if np.std(arr, ddof=1) <= 1e-10:
                per_variant_psr[b_hash] = None
                continue
            try:
                psr = deflated_sharpe_ratio([arr], periods_per_year=periods_per_year)
                per_variant_psr[b_hash] = float(psr.p_value)
            except Exception:  # pragma: no cover
                per_variant_psr[b_hash] = None

        hashes = top["backtest_hash"].to_list()
        placeholders = ",".join("?" for _ in hashes)
        cm = self._query(
            f"""
            SELECT leader_hash, k_variants, member_digest,
                   n_trials_effective_mp, n_trials_effective_er,
                   dsr_raw, dsr_raw_pvalue,
                   dsr_mp,  dsr_mp_pvalue,
                   dsr_er,  dsr_er_pvalue, expected_max_sharpe_er,
                   ras_leader, ras_pvalue,
                   reality_check_pvalue, pbo
            FROM cohort_metrics
            WHERE cohort_type = 'family' AND stage = ?
              AND leader_hash IN ({placeholders})
            """,
            (stage, *hashes),
        )
        cm_by_hash: dict[str, dict] = {}
        if not cm.is_empty():
            for r in cm.iter_rows(named=True):
                cm_by_hash[r["leader_hash"]] = r

        def _round(x, n=4):
            return round(x, n) if x is not None else None

        rows = []
        for r in top.iter_rows(named=True):
            b_hash = r["backtest_hash"]
            cmr = cm_by_hash.get(b_hash)
            is_leader = cmr is not None
            key = (r["family"], r["label"])
            k_scoped = scoped_k.get(key) if prediction_hashes else None
            # Applies only where the stored cohort IS the one that was ranked, established
            # on the members and not on how many there are: swapping a retired member for a
            # live one leaves `k_variants` untouched, and the correction would then be
            # reported against a cohort it was not computed over. `member_digest` is null on
            # rows written before it was persisted; those still fall back to the count, which
            # at least catches the scoping that strictly removes variants, and stop doing so
            # the first time their producer re-runs.
            applies = is_leader and (
                not prediction_hashes
                or (
                    cmr["member_digest"] == scoped_digest.get(key)
                    if cmr["member_digest"]
                    else cmr["k_variants"] == k_scoped
                )
            )
            dsr_er_p = cmr["dsr_er_pvalue"] if applies else None
            rows.append(
                {
                    "source": r["source"],
                    "family": r["family"],
                    "label": r["label"],
                    "sharpe": _round(r["sharpe"]),
                    "psr_pvalue": _round(per_variant_psr.get(b_hash)),
                    "deflated_sharpe": _round(cmr["dsr_er"]) if applies else None,
                    "expected_max_sharpe": _round(cmr["expected_max_sharpe_er"])
                    if applies
                    else None,
                    "dsr_pvalue": _round(dsr_er_p),
                    "significant": (dsr_er_p is not None and dsr_er_p < 0.05) if applies else None,
                    "is_best": is_leader,
                    "dsr_mp": _round(cmr["dsr_mp"]) if applies else None,
                    "dsr_mp_pvalue": _round(cmr["dsr_mp_pvalue"]) if applies else None,
                    "dsr_raw": _round(cmr["dsr_raw"]) if applies else None,
                    "dsr_raw_pvalue": _round(cmr["dsr_raw_pvalue"]) if applies else None,
                    "k_variants": cmr["k_variants"] if is_leader else None,
                    "k_variants_scoped": k_scoped,
                    "n_trials_effective_er": _round(cmr["n_trials_effective_er"], 1)
                    if applies
                    else None,
                    "n_trials_effective_mp": _round(cmr["n_trials_effective_mp"], 1)
                    if applies
                    else None,
                    "ras_leader": _round(cmr["ras_leader"]) if applies else None,
                    "ras_pvalue": _round(cmr["ras_pvalue"]) if applies else None,
                    "reality_check_pvalue": _round(cmr["reality_check_pvalue"])
                    if applies
                    else None,
                    "pbo": _round(cmr["pbo"]) if applies else None,
                }
            )
        return pl.DataFrame(rows).sort("sharpe", descending=True, nulls_last=True)

    # -----------------------------------------------------------------
    # cost_sensitivity: breakeven analysis from registry
    # -----------------------------------------------------------------

    def cost_sensitivity(self, *, prediction_hash: str | None = None) -> pl.DataFrame:
        """Load cost sensitivity results from the cost_sensitivity stage.

        Only the bps (``commission.model='percentage'``) regime is returned;
        per-share rows have ``commission.rate=0`` and ``slippage.rate=0`` so
        their derived ``cost_bps`` is mechanically 0 and would otherwise pile
        up on the bps-axis origin. Notebooks rendering both regimes must
        query the registry directly (see ``etfs/16_costs.py::load_cost_rows``
        for the pattern).

        Parameters
        ----------
        prediction_hash : str, optional
            When provided, restrict to cost rows on this prediction. Case
            studies with a pinned carrier (e.g. nasdaq's cost-feasible
            ensemble) must scope to the carrier so the full-universe
            cost-defeat demonstration rows do not pool into the headline.

        Returns
        -------
        pl.DataFrame
            Columns: cost_bps, sharpe, max_drawdown, allocator
        """
        pred_clause = "" if prediction_hash is None else " AND b.prediction_hash = ?"
        params = () if prediction_hash is None else (prediction_hash,)
        df = self._query(
            f"""
            SELECT
                b.spec_json,
                bm.sharpe,
                bm.max_drawdown
            FROM backtest_runs b
            JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
            WHERE b.stage = 'cost_sensitivity'
              AND bm.sharpe IS NOT NULL
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
              AND json_extract(b.spec_json, '$.backtest_config.commission.model') = 'percentage'
              {pred_clause}
            """,
            params,
        )
        if df.is_empty():
            return df
        df = self._filter_active_models(df)
        if df.is_empty():
            return df

        # Extract cost_bps and allocator from spec
        rows = []
        for spec_str, sharpe, max_dd in zip(
            df["spec_json"].to_list(),
            df["sharpe"].to_list(),
            df["max_drawdown"].to_list(),
            strict=False,
        ):
            spec = _parse_spec(spec_str) or {}
            costs = cost_view(spec)
            cost_bps = costs.get("commission_bps", 0) + costs.get("slippage_bps", 0)
            allocator = strategy_view(spec).get("allocation", {}).get("method", "equal_weight")
            rows.append(
                {
                    "cost_bps": cost_bps,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                    "allocator": allocator,
                }
            )

        return pl.DataFrame(rows).sort("cost_bps")

    # -----------------------------------------------------------------
    # risk_impact: risk overlay comparison from registry
    # -----------------------------------------------------------------

    def risk_impact(
        self,
        *,
        prediction_hash: str | None = None,
        prediction_hashes: tuple[str, ...] | list[str] | None = None,
        parents: Iterable[tuple[str, str, object]] | None = None,
    ) -> pl.DataFrame:
        """Load risk overlay results and compute impact vs the baseline each one modified.

        Parameters
        ----------
        prediction_hash : str, optional
            When provided, restrict to risk-overlay rows on this prediction.
            Case studies with a pinned carrier (e.g. nasdaq's cost-feasible
            ensemble) must scope to the carrier so the full-universe overlay
            demonstration rows do not pool into the headline.
        prediction_hashes : sequence of str, optional
            Restrict to overlays on these predictions. An overlay is a change to
            one allocation, so pooling overlays from predictions the caller did
            not sweep measures rules against strategies it never built.
        parents : sequence of (prediction_hash, allocator, top_k), optional
            Restrict to overlays sitting on these allocation specifications. A
            prediction has one allocation *stage* and several allocation *parents*,
            distinguished by allocator and ``top_k``; scoping on the prediction
            alone admits overlays on combinations the caller did not advance, and
            those can rank among the reported leaders. ``top_k`` is compared as
            written in the spec, so ``None`` matches a spec that carries no
            ``top_k`` rather than matching everything.

        Returns
        -------
        pl.DataFrame
            Columns: risk_name, risk_type, sharpe, max_drawdown, num_trades,
            allocator, prediction_hash, baseline_sharpe, sharpe_delta

        Each overlay's ``baseline_sharpe`` is the no-overlay Sharpe of the
        allocation it was applied to, matched on ``(prediction_hash,
        allocator)``. One registry-wide maximum would measure most overlays
        against a strategy they did not modify, so an overlay could appear to
        destroy Sharpe purely because a different prediction allocated better.
        """
        pred_clause = ""
        pred_params: tuple = ()
        if prediction_hash is not None:
            pred_clause = " AND b.prediction_hash = ?"
            pred_params = (prediction_hash,)
        elif prediction_hashes:
            pred_clause = " AND b.prediction_hash IN (SELECT value FROM json_each(?))"
            pred_params = (json.dumps(list(prediction_hashes)),)
        df = self._query(
            f"""
            SELECT
                b.spec_json,
                b.prediction_hash,
                t.family,
                t.config_name,
                bm.sharpe,
                bm.max_drawdown,
                bm.num_trades
            FROM backtest_runs b
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
            WHERE b.stage = 'risk_overlay'
              {excluded_family_sql(self.case_study, "t.family")[0]}
              AND bm.sharpe IS NOT NULL
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
              {pred_clause}
            """,
            tuple(excluded_family_sql(self.case_study, "t.family")[1]) + pred_params,
        )
        if df.is_empty():
            return df
        df = self._filter_active_models(df)
        if df.is_empty():
            return df

        rows = []
        for spec_str, pred_h, sharpe, max_dd, trades in zip(
            df["spec_json"].to_list(),
            df["prediction_hash"].to_list(),
            df["sharpe"].to_list(),
            df["max_drawdown"].to_list(),
            df["num_trades"].to_list(),
            strict=False,
        ):
            spec = _parse_spec(spec_str) or {}
            risk = strategy_view(spec).get("risk", {})
            risk_name = risk.get("name", "unknown")

            # Determine risk type
            pos_rules = risk.get("position_rules", [])
            port_limits = risk.get("portfolio_limits", [])
            if risk_name == "baseline":
                risk_type = "baseline"
            elif pos_rules:
                risk_type = pos_rules[0].get("type", "unknown")
            elif port_limits:
                risk_type = port_limits[0].get("type", "unknown")
            else:
                risk_type = "unknown"

            rows.append(
                {
                    "risk_name": risk_name,
                    "risk_type": risk_type,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                    "num_trades": trades,
                    "prediction_hash": pred_h,
                    "allocator": strategy_view(spec)
                    .get("allocation", {})
                    .get("method", "equal_weight"),
                    "top_k": strategy_view(spec).get("signal", {}).get("top_k"),
                }
            )

        result = pl.DataFrame(rows)
        if parents is not None:
            wanted = {(str(h), str(a), k) for h, a, k in parents}
            result = result.filter(
                pl.struct("prediction_hash", "allocator", "top_k").map_elements(
                    lambda r: (r["prediction_hash"], r["allocator"], r["top_k"]) in wanted,
                    return_dtype=pl.Boolean,
                )
            )
            if result.is_empty():
                return result

        # Compute baseline: the no-overlay Sharpe the overlays are measured
        # against. Registry-wide (unpinned) this is the best allocation-stage
        # Sharpe — the normal pipeline where overlays sit on an allocator. When
        # the comparison is pinned to a carrier prediction (e.g. nasdaq's
        # signal-stage slot ensemble, which has no allocation stage), the
        # baseline is the carrier's own no-overlay Sharpe over its signal and
        # allocation rows; the registry-wide allocation max would otherwise
        # return an unrelated full-universe strategy.
        baseline_stage_clause = (
            "b.stage = 'allocation'"
            if prediction_hash is None
            else "b.stage IN ('signal', 'allocation')"
        )
        baseline_df = self._query(
            f"""
            SELECT
                bm.sharpe,
                b.prediction_hash,
                b.spec_json,
                t.family,
                t.config_name
            FROM backtest_metrics bm
            JOIN backtest_runs b ON bm.backtest_hash = b.backtest_hash
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            WHERE {baseline_stage_clause}
              {excluded_family_sql(self.case_study, "t.family")[0]}
              {pred_clause}
            """,
            tuple(excluded_family_sql(self.case_study, "t.family")[1]) + pred_params,
        )
        baseline_df = self._filter_active_models(baseline_df)

        if baseline_df.is_empty():
            return result.with_columns(
                pl.lit(None).cast(pl.Float64).alias("baseline_sharpe"),
                pl.lit(None).cast(pl.Float64).alias("sharpe_delta"),
            ).sort("sharpe", descending=True)

        # The parent of an overlay is the no-overlay run of the same prediction under the same
        # allocator at the same concentration. All three are needed: a single registry-wide
        # maximum would charge every overlay for the gap between its own allocation and the best
        # allocation anywhere in the registry, and dropping `top_k` would still leave several
        # distinct parents sharing a key, whose maximum is not the run the overlay modified.
        parents = (
            baseline_df.with_columns(
                pl.col("spec_json")
                .map_elements(
                    lambda js: (
                        strategy_view(_parse_spec(js) or {})
                        .get("allocation", {})
                        .get("method", "equal_weight")
                    ),
                    return_dtype=pl.String,
                )
                .alias("allocator"),
                pl.col("spec_json")
                .map_elements(
                    lambda js: strategy_view(_parse_spec(js) or {}).get("signal", {}).get("top_k"),
                    return_dtype=pl.Int64,
                )
                .alias("top_k"),
            )
            .group_by("prediction_hash", "allocator", "top_k")
            .agg(pl.col("sharpe").max().alias("baseline_sharpe"))
        )
        result = result.join(parents, on=["prediction_hash", "allocator", "top_k"], how="left")
        result = result.with_columns(
            (pl.col("sharpe") - pl.col("baseline_sharpe")).alias("sharpe_delta")
        )
        return result.sort("sharpe", descending=True)

    # -----------------------------------------------------------------
    # fold_performance: per-fold backtest metrics
    # -----------------------------------------------------------------

    def fold_performance(self, backtest_hash: str) -> pl.DataFrame:
        """Per-fold backtest metrics (Sharpe, max_dd, etc.) for one backtest.

        Parameters
        ----------
        backtest_hash : str
            Full or prefix of the backtest hash.

        Returns
        -------
        pl.DataFrame
            Columns: fold_id, sharpe, cagr, max_drawdown, volatility,
            total_return, n_days, ...
        """
        df = self._query(
            """
            SELECT *
            FROM backtest_fold_metrics
            WHERE backtest_hash LIKE ?
            ORDER BY fold_id
            """,
            (backtest_hash + "%",),
        )
        if df.is_empty():
            return df

        # Drop internal columns, already in wide format
        drop_cols = [c for c in ["backtest_hash", "computed_at"] if c in df.columns]
        if drop_cols:
            df = df.drop(drop_cols)
        return df.sort("fold_id")

    # -----------------------------------------------------------------
    # ic_sharpe_scatter: IC vs backtest Sharpe per fold
    # -----------------------------------------------------------------

    def ic_sharpe_scatter(
        self,
        stage: str = "signal",
        *,
        top_n: int = 10,
    ) -> pl.DataFrame:
        """Join prediction IC per fold with backtest Sharpe per fold.

        This enables the empirical fundamental law test: plotting IC
        against realized Sharpe at the fold level to see whether
        better predictions produce better portfolio returns.

        Parameters
        ----------
        stage : str
            Pipeline stage to filter backtests.
        top_n : int
            Number of top backtests (by headline Sharpe) to include.

        Returns
        -------
        pl.DataFrame
            Columns: source, fold_id, ic, sharpe, cagr, max_drawdown
        """
        # Get top backtests at this stage
        top = self.best(stage=stage, top_n=top_n)
        if top.is_empty():
            return pl.DataFrame()

        rows = []
        for row in top.iter_rows(named=True):
            b_hash = row["backtest_hash"]
            p_hash = row["prediction_hash"]
            source = row.get("source", b_hash[:8])

            # Get backtest fold metrics (wide format)
            bt_wide = self._query(
                """
                SELECT *
                FROM backtest_fold_metrics
                WHERE backtest_hash = ?
                """,
                (b_hash,),
            )
            if bt_wide.is_empty():
                continue

            # Drop internal columns
            drop_cols = [c for c in ["backtest_hash", "computed_at"] if c in bt_wide.columns]
            if drop_cols:
                bt_wide = bt_wide.drop(drop_cols)

            # Get prediction fold metrics (IC) — wide format
            pred_folds = self._query(
                """
                SELECT fold_id, ic
                FROM fold_metrics
                WHERE prediction_hash = ?
                """,
                (p_hash,),
            )

            if pred_folds.is_empty():
                continue

            ic_df = pred_folds.select("fold_id", "ic")

            joined = bt_wide.join(ic_df, on="fold_id", how="left")
            joined = joined.with_columns(pl.lit(source).alias("source"))
            rows.append(joined)

        if not rows:
            return pl.DataFrame()

        result = pl.concat(rows, how="diagonal")

        # Select core columns (others available but less critical)
        cols = ["source", "fold_id"]
        for c in ["ic", "sharpe", "cagr", "max_drawdown", "volatility", "n_days"]:
            if c in result.columns:
                cols.append(c)

        return result.select(cols).sort("source", "fold_id")

    # -----------------------------------------------------------------
    # backfill_fold_metrics: compute fold metrics for existing backtests
    # -----------------------------------------------------------------

    def backfill_fold_metrics(
        self,
        stage: str = "signal",
        *,
        label: str,
        limit: int = 0,
    ) -> int:
        """Compute and store fold metrics for existing backtests.

        Finds backtests at the given stage that have daily_returns.parquet
        but no entries in backtest_fold_metrics, then computes and registers
        fold-level performance metrics.

        Parameters
        ----------
        stage : str
            Pipeline stage to backfill.
        label : str
            Required label name for both row selection and fold boundary computation.
        limit : int
            Max backtests to process (0 = all).

        Returns
        -------
        int
            Number of backtests backfilled.
        """
        from case_studies.utils.registry import (
            compute_backtest_fold_metrics,
            register_backtest_fold_metrics,
        )

        # Find backtests without fold metrics
        df = self._query(
            """
            SELECT b.backtest_hash
            FROM backtest_runs b
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            WHERE b.stage = ?
              AND t.label = ?
              AND b.backtest_hash NOT IN (
                  SELECT DISTINCT backtest_hash FROM backtest_fold_metrics
              )
            """,
            (stage, label),
        )
        if df.is_empty():
            return 0

        hashes = df["backtest_hash"].to_list()

        count = 0
        for b_hash in hashes:
            if limit > 0 and count >= limit:
                break

            returns_path = self._backtest_dir(b_hash) / "daily_returns.parquet"
            if not returns_path.exists():
                continue

            daily_ret = pl.read_parquet(returns_path)
            fold_m = compute_backtest_fold_metrics(daily_ret, self.case_study, label=label)
            if fold_m:
                register_backtest_fold_metrics(self.case_study, b_hash, fold_m)
                count += 1

        return count

    # -----------------------------------------------------------------
    # search_context: distribution stats for a stage
    # -----------------------------------------------------------------

    def search_context(
        self,
        stage: str = "signal",
        *,
        prediction_hashes: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, Any]:
        """Distribution statistics for all backtests at a stage.

        Quantifies search risk: how exceptional is the champion relative
        to the full sweep?

        Parameters
        ----------
        prediction_hashes : sequence of str, optional
            Restrict the sweep to these prediction identities. The trial count and the
            distribution the champion is placed in are then the ones the shortlist was
            actually drawn from.

        Returns
        -------
        dict
            Keys: total, median_sharpe, mean_sharpe, p90_sharpe,
            champion_sharpe, champion_source, champion_percentile,
            pct_positive
        """
        scope_sql = ""
        scope_params: tuple[str, ...] = ()
        coverage_sql = full_coverage_prediction_sql("p", "t", "pm")
        if prediction_hashes is not None:
            placeholders = ", ".join("?" for _ in prediction_hashes)
            scope_sql = f" AND p.prediction_hash IN ({placeholders})"
            coverage_sql = full_coverage_prediction_sql(
                "p",
                "t",
                "pm",
                population_subquery="SELECT value FROM json_each(?)",
            )
            scope_params = (json.dumps(list(prediction_hashes)), *prediction_hashes)
        df = self._query(
            f"""
            SELECT
                bm.sharpe,
                t.family || '/' || COALESCE(t.config_name, 'default') AS source
            FROM backtest_metrics bm
            JOIN backtest_runs b ON bm.backtest_hash = b.backtest_hash
            JOIN prediction_sets p ON b.prediction_hash = p.prediction_hash
            JOIN training_runs t ON p.training_hash = t.training_hash
            JOIN prediction_metrics pm ON p.prediction_hash = pm.prediction_hash
            WHERE b.stage = ?
              AND p.split != 'holdout'
              {coverage_sql}
              AND bm.sharpe IS NOT NULL
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
              {scope_sql}
            """,
            (stage, *scope_params),
        )
        if df.is_empty():
            return {}

        sharpes = df["sharpe"].to_numpy()
        best_idx = int(np.argmax(sharpes))

        return {
            "total": len(sharpes),
            "median_sharpe": float(np.median(sharpes)),
            "mean_sharpe": float(np.mean(sharpes)),
            "p90_sharpe": float(np.percentile(sharpes, 90)),
            "champion_sharpe": float(sharpes[best_idx]),
            "champion_source": df["source"][best_idx],
            "champion_percentile": float((sharpes <= sharpes[best_idx]).sum() / len(sharpes) * 100),
            "pct_positive": float((sharpes > 0).sum() / len(sharpes) * 100),
        }

    # -----------------------------------------------------------------
    # champion_lineage: locked path through all stages
    # -----------------------------------------------------------------

    def champion_lineage(self, prediction_hash: str) -> dict[str, dict]:
        """Locked path through signal -> allocation -> cost -> risk.

        For each stage, returns the BEST backtest for this specific
        ``prediction_hash`` with spec annotations (allocator, top_k,
        cost_bps, risk_type).

        Returns
        -------
        dict[str, dict]
            Keyed by stage. Each value has: sharpe, cagr, max_drawdown,
            backtest_hash, plus stage-specific fields.
        """
        df = self._query(
            """
            SELECT
                b.stage,
                b.backtest_hash,
                b.spec_json,
                bm.sharpe,
                bm.cagr,
                bm.max_drawdown,
                bm.volatility,
                bm.total_return
            FROM backtest_runs b
            JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
            WHERE b.prediction_hash = ?
              AND b.stage IS NOT NULL
              AND bm.sharpe IS NOT NULL
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
            ORDER BY bm.sharpe DESC
            """,
            (prediction_hash,),
        )
        if df.is_empty():
            return {}

        result: dict[str, dict] = {}
        stage_order = ["signal", "allocation", "cost_sensitivity", "risk_overlay"]

        for stage_name in stage_order:
            stage_df = df.filter(pl.col("stage") == stage_name)
            if stage_df.is_empty():
                continue

            row = stage_df.row(0, named=True)
            entry: dict[str, Any] = {
                "sharpe": row["sharpe"],
                "cagr": row["cagr"],
                "max_drawdown": row["max_drawdown"],
                "volatility": row["volatility"],
                "total_return": row["total_return"],
                "backtest_hash": row["backtest_hash"],
            }

            # Extract stage-specific annotations from spec_json
            spec = {}
            if row["spec_json"]:
                import contextlib

                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    spec = json.loads(row["spec_json"])
            strategy = strategy_view(spec)

            if stage_name == "signal":
                entry["signal_method"] = strategy.get("signal", {}).get("method", "")
                entry["top_k"] = strategy.get("signal", {}).get("top_k", None)
            elif stage_name == "allocation":
                entry["allocator"] = strategy.get("allocation", {}).get("method", "")
                entry["top_k"] = strategy.get("signal", {}).get("top_k", None)
            elif stage_name == "cost_sensitivity":
                costs = cost_view(spec)
                entry["cost_bps"] = costs.get("commission_bps", 0) + costs.get("slippage_bps", 0)
                entry["allocator"] = strategy.get("allocation", {}).get("method", "")
            elif stage_name == "risk_overlay":
                risk = strategy.get("risk", {})
                entry["risk_name"] = risk.get("name", "")
                pos_rules = risk.get("position_rules", [])
                entry["risk_type"] = pos_rules[0].get("type", "") if pos_rules else ""

            result[stage_name] = entry

        return result

    # -----------------------------------------------------------------
    # concentration_curve: Sharpe vs top_k at allocation stage
    # -----------------------------------------------------------------

    def concentration_curve(self, prediction_hash: str) -> pl.DataFrame:
        """Sharpe vs top_k for a given prediction at allocation stage.

        Shows how portfolio concentration affects performance — typically
        more actionable than allocator comparison alone.

        Returns
        -------
        pl.DataFrame
            Columns: top_k, allocator, sharpe, max_drawdown, cagr
        """
        df = self._query(
            """
            SELECT
                b.spec_json,
                bm.sharpe,
                bm.max_drawdown,
                bm.cagr
            FROM backtest_runs b
            JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
            WHERE b.prediction_hash = ?
              AND b.stage = 'allocation'
              AND bm.sharpe IS NOT NULL
              AND (bm.num_trades IS NULL OR bm.num_trades > 0)
            """,
            (prediction_hash,),
        )
        if df.is_empty():
            return df

        rows = []
        for spec_str, sharpe, max_dd, cagr in zip(
            df["spec_json"].to_list(),
            df["sharpe"].to_list(),
            df["max_drawdown"].to_list(),
            df["cagr"].to_list(),
            strict=False,
        ):
            spec = _parse_spec(spec_str) or {}
            strategy = strategy_view(spec)
            top_k = strategy.get("signal", {}).get("top_k", None)
            allocator = strategy.get("allocation", {}).get("method", "equal_weight")
            rows.append(
                {
                    "top_k": top_k,
                    "allocator": allocator,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                    "cagr": cagr,
                }
            )

        return pl.DataFrame(rows).sort("top_k")

    # -----------------------------------------------------------------
    # repr
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        counts = self.summary()
        total = sum(counts.values())
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return f"BacktestExplorer('{self.case_study}', {total} runs: {parts})"
