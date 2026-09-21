"""A width of 0 asks for every candidate, at the allocation stage as at the signal stage.

`top_n_predictions.signal` is declared `0` in all nine shipped `setup.yaml` files, under the
comment "all signal predictions", and `_STAGE_DEFAULTS["signal"]` is 0 for the same reason. The
four signal-stage notebooks read it that way, each guarding its truncation with
`if TOP_N_PREDICTIONS > 0`.

One stage later the same number meant the opposite. `resolve_best_predictions` bound `top_n`
straight into a SQL `LIMIT`, where 0 returns no rows; `us_equities_panel` and `sp500_options`
passed it to `.head`, which is empty at 0; and `shortlist_signal_configurations` compared the
count it selected against the number asked for, so the only way to ask for every configuration
was to already know how many there were - the one thing the shortlist exists to compute
(`ml4t/agent-workspace#1201`). Measured 2026-09-18 on cme_futures: a run at 999, meaning "all",
raised `signal population has 50 distinct configurations, expected 999`.

What each test holds:

- `top_n_cap` is the single reading of the number. 0 is no cap, a positive width is that cap,
  and a negative one raises rather than passing through - SQLite reads `LIMIT -1` as no limit,
  so a width that arithmetic produced would otherwise select everything by accident.
- both resolvers honour it on both coverage paths, because the raw path binds a SQL parameter
  and the canonical path calls `.head` in Python, and a fix to one leaves the other empty.
- a positive width the population cannot fill still raises. That is not the same request: a
  caller that asked for 20 and got 8 is looking at a degenerate population, and silently
  ranking the 8 would hide it. 999 is indistinguishable from that case, which is why the ask
  for everything needs its own spelling rather than a number large enough to be sure.
"""

from __future__ import annotations

import sqlite3
from unittest import mock

import pytest

from case_studies.cme_futures import research_workflow
from case_studies.research import BacktestResult
from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.registry import resolve_best_backtest_runs, resolve_best_predictions
from case_studies.utils.sweep_config import top_n_cap

# (prediction, family, config, sharpe) - four distinct configurations, ranked by Sharpe.
ROWS = [
    ("best", "gbm", "cfg_best", 4.0),
    ("second", "gbm", "cfg_second", 3.0),
    ("third", "linear", "cfg_third", 2.0),
    ("fourth", "linear", "cfg_fourth", 1.0),
]
EVERY_CONFIG = [row[0] for row in ROWS]


def _build_registry(case_dir) -> None:
    run_log = case_dir / "run_log"
    run_log.mkdir(parents=True)
    with sqlite3.connect(run_log / "registry.db") as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY,
                family TEXT,
                config_name TEXT,
                label TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY,
                training_hash TEXT,
                split TEXT,
                checkpoint_value REAL,
                checkpoint_kind TEXT
            );
            CREATE TABLE prediction_metrics (
                prediction_hash TEXT PRIMARY KEY,
                ic_mean REAL,
                ic_mean_daily REAL,
                ic_ci_lo REAL,
                ic_ci_hi REAL,
                ic_n_days REAL
            );
            CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL, ic_std REAL);
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY,
                prediction_hash TEXT,
                spec_json TEXT,
                stage TEXT
            );
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY,
                sharpe REAL,
                cagr REAL,
                max_drawdown REAL,
                total_return REAL,
                volatility REAL,
                num_trades REAL
            );
            CREATE TABLE backtest_fold_metrics (
                backtest_hash TEXT,
                fold_id INTEGER,
                sharpe REAL
            );
            """
        )
        for prediction_hash, family, config, sharpe in ROWS:
            training_hash = f"train_{prediction_hash}"
            db.execute(
                "INSERT INTO training_runs VALUES (?, ?, ?, 'fwd_ret_5d')",
                (training_hash, family, config),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, 'validation', 0, 'iteration')",
                (prediction_hash, training_hash),
            )
            # Equal coverage, so the full-coverage bar keeps every row and the width is the
            # only thing selecting here.
            db.execute(
                "INSERT INTO prediction_metrics VALUES (?, 0.1, 0.1, 0.0, 0.2, 4.0)",
                (prediction_hash,),
            )
            db.execute(
                """
                INSERT INTO backtest_runs VALUES (
                    ?, ?, '{"allocation":{"method":"score_weighted"}}', 'signal'
                )
                """,
                (f"bt_{prediction_hash}", prediction_hash),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, 0.1, -0.1, 0.2, 0.1, 1)",
                (f"bt_{prediction_hash}", sharpe),
            )


@pytest.fixture
def case_dir(tmp_path):
    path = tmp_path / "case"
    _build_registry(path)
    return path


@pytest.fixture
def canonical_coverage(monkeypatch):
    """Send the canonical coverage path down its Python branch with every row in window."""
    monkeypatch.setattr(
        "case_studies.utils.registry.queries.canonical_coverage_days",
        lambda case_study, label, split, prediction_hash, case_dir: 4,
    )


def _predictions(case_dir, top_n, **kwargs) -> list[str]:
    selected = resolve_best_predictions(
        "test",
        "fwd_ret_5d",
        split="validation",
        stage="signal",
        top_n=top_n,
        case_dir=case_dir,
        **kwargs,
    )
    return selected["prediction_hash"].to_list()


def _backtest_runs(case_dir, top_n, **kwargs) -> list[str]:
    selected = resolve_best_backtest_runs(
        "test",
        "fwd_ret_5d",
        split="validation",
        stage="signal",
        top_n=top_n,
        case_dir=case_dir,
        **kwargs,
    )
    return selected["prediction_hash"].to_list()


def test_zero_is_the_only_spelling_that_means_every_candidate() -> None:
    assert top_n_cap(0) is None
    assert top_n_cap(10) == 10
    assert top_n_cap(1) == 1


def test_a_negative_width_raises_rather_than_meaning_all() -> None:
    # SQLite reads LIMIT -1 as no limit, so passing one through would select everything on
    # the raw path and nothing recognisable on the canonical one.
    with pytest.raises(ValueError, match="0 .every candidate. or positive"):
        top_n_cap(-1)


def test_the_raw_prediction_resolver_returns_everything_at_zero(case_dir) -> None:
    assert _predictions(case_dir, 2) == ["best", "second"]
    assert _predictions(case_dir, 0) == EVERY_CONFIG


def test_the_canonical_prediction_resolver_returns_everything_at_zero(
    case_dir, canonical_coverage
) -> None:
    # A separate branch with its own truncation: the raw path binds a SQL LIMIT, this one
    # calls .head in Python, and fixing either alone leaves the other empty at 0.
    assert _predictions(case_dir, 2, coverage_window="canonical") == ["best", "second"]
    assert _predictions(case_dir, 0, coverage_window="canonical") == EVERY_CONFIG


def test_the_raw_backtest_run_resolver_returns_everything_at_zero(case_dir) -> None:
    assert _backtest_runs(case_dir, 2) == ["best", "second"]
    assert _backtest_runs(case_dir, 0) == EVERY_CONFIG


def test_the_canonical_backtest_run_resolver_returns_everything_at_zero(
    case_dir, canonical_coverage
) -> None:
    assert _backtest_runs(case_dir, 2, coverage_window="canonical") == ["best", "second"]
    assert _backtest_runs(case_dir, 0, coverage_window="canonical") == EVERY_CONFIG


def test_a_width_larger_than_the_population_is_not_an_error_for_the_resolvers(case_dir) -> None:
    # The resolvers cap rather than promise, and always did. Only the two notebook-level
    # checks refused an over-ask, which is what sent a caller to 999 in the first place.
    assert _predictions(case_dir, 999) == EVERY_CONFIG
    assert _backtest_runs(case_dir, 999) == EVERY_CONFIG


def _pool_of(monkeypatch, configs) -> None:
    """Stand a ranked signal pool in front of `shortlist_signal_configurations`.

    The function's own inputs are the two module-level readers, so replacing them leaves the
    selection - which is what the width decides - running for real. Each config appears twice,
    best first, because a duplicate is what the loop's `seen` set exists to drop and a limit
    that counted results rather than configurations would stop halfway through the population.
    """
    results = []
    for family, config_name in configs:
        for _ in range(2):
            result = mock.create_autospec(BacktestResult, instance=True)
            result.lineage.return_value = {
                "training_spec": {"family": family, "config_name": config_name}
            }
            results.append(result)
    monkeypatch.setattr(
        research_workflow, "stage_backtest_results", lambda *args, **kwargs: tuple(results)
    )
    monkeypatch.setattr(research_workflow, "rank_by_validation_sharpe", lambda study, pool: pool)


POPULATION = [("gbm", "a"), ("gbm", "b"), ("linear", "c"), ("linear", "d")]


def _shortlist(limit):
    return research_workflow.shortlist_signal_configurations(
        object(), label="fwd_ret_5d", limit=limit
    )


def test_the_shortlist_takes_every_configuration_at_zero(monkeypatch) -> None:
    _pool_of(monkeypatch, POPULATION)

    assert len(_shortlist(0)) == len(POPULATION)


def test_the_shortlist_still_holds_a_positive_width_to_its_promise(monkeypatch) -> None:
    _pool_of(monkeypatch, POPULATION)

    assert len(_shortlist(2)) == 2
    # Asked for more than the population holds. Indistinguishable from a degenerate
    # population, so it stays a refusal - `0` is how a caller says they meant all of them.
    with pytest.raises(ValueError, match="4 distinct configurations, expected 999"):
        _shortlist(999)


def test_the_shortlist_refuses_an_empty_population_at_zero(monkeypatch) -> None:
    # "All of them" is not a licence to advance nothing: a stage with no registered
    # configurations means the upstream notebook has not run, and every cell below the
    # shortlist would be a no-op on an empty selection that still reports a clean run.
    _pool_of(monkeypatch, [])

    with pytest.raises(ValueError, match="no distinct configurations"):
        _shortlist(0)


def test_the_explorer_returns_every_backtest_at_zero(case_dir) -> None:
    """`BacktestExplorer.best` carried a sentinel built around this defect.

    Counting a whole cohort meant asking for a million rows, "more rows than any cohort
    holds rather than for no limit at all", because 0 truncated to nothing. `deflated_sharpe`
    takes its scoped cohort that way, and a cohort larger than the sentinel would have been
    silently miscounted rather than refused.
    """
    explorer = BacktestExplorer("test", case_dir=case_dir)

    assert explorer.best(stage="signal", top_n=2)["prediction_hash"].to_list() == [
        "best",
        "second",
    ]
    assert explorer.best(stage="signal", top_n=0)["prediction_hash"].to_list() == EVERY_CONFIG
