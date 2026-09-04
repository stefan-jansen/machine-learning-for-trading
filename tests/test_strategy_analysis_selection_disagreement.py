"""Ranking a raw Sharpe column selects a different carrier, and used to say so as silence.

A Sharpe computed over a configuration's own available history is not comparable across
configurations that priced different spans, so ranking the pool's column directly rewards
whichever candidate had the most forgiving window. Measured on cme_futures (992 signal /
120 allocation / 28 risk_overlay backtests, rebuilt 2026-08-30): the raw column answered
latent_factors/sdf on fwd_ret_21d at 1.274, `resolve_solvent_carrier` gbm/leaves_31_mse on
fwd_ret_5d at 1.236 raw and 1.294 over the 1,270 sessions the candidates all price.

`17_holdout_predictions` and `18_holdout_backtest` run the resolver's answer, so a notebook
that selected its own then asked for the holdout replay of a configuration nothing ran, got
None, and printed "not produced yet" while the holdout sat in the registry.
"""

from __future__ import annotations

import pytest

import case_studies.utils.strategy_analysis as sa

CANONICAL = {
    "val_backtest_hash": "619a5cd773b4",
    "holdout_backtest_hash": "aa11bb22cc33",
    "family": "gbm",
    "config_name": "leaves_31_mse",
    "label": "fwd_ret_5d",
}


def _canonical(**overrides):
    def _resolve(case_study, **kwargs):
        return {**CANONICAL, **overrides}

    return _resolve


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A run log holding the hash the caller asks about.

    The diagnosis only fires on a *registered* hash: one the run log has never seen is a
    stale constant, not a carrier chosen from a pool, and `_resolve_holdout_self_backtest`
    already says so more usefully.
    """
    import sqlite3

    case_dir = tmp_path / "cs"
    (case_dir / "run_log").mkdir(parents=True)
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.execute("CREATE TABLE backtest_runs (backtest_hash TEXT PRIMARY KEY)")
    db.executemany(
        "INSERT INTO backtest_runs VALUES (?)",
        [("cbd72d8408d3",), (CANONICAL["val_backtest_hash"],)],
    )
    db.commit()
    db.close()
    monkeypatch.setattr(sa, "get_case_study_dir", lambda cs, **kw: case_dir, raising=False)
    import utils.paths

    monkeypatch.setattr(utils.paths, "get_case_study_dir", lambda cs, **kw: case_dir)
    return case_dir


def test_a_carrier_the_holdout_notebooks_never_ran_is_named_as_such(registry, monkeypatch) -> None:
    monkeypatch.setattr(sa, "resolve_canonical_rank1_lineage", _canonical())
    monkeypatch.setattr(
        sa,
        "_resolve_holdout_self_backtest",
        lambda cs, h: sa.HoldoutSelfBacktest(None, "no holdout backtest is registered"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        sa.resolve_holdout_self_backtest("cme_futures", "cbd72d8408d3")

    message = str(excinfo.value)
    assert "selection disagreement" in message
    assert "cbd72d8408d3" in message and "619a5cd773b4" in message
    assert "gbm/leaves_31_mse on fwd_ret_5d" in message


def test_a_holdout_that_genuinely_has_not_run_still_reports_that(monkeypatch) -> None:
    """The normal state for anyone working the notebooks in order must not raise."""
    monkeypatch.setattr(
        sa, "resolve_canonical_rank1_lineage", _canonical(holdout_backtest_hash=None)
    )
    monkeypatch.setattr(
        sa,
        "_resolve_holdout_self_backtest",
        lambda cs, h: sa.HoldoutSelfBacktest(None, "the holdout has not been evaluated"),
    )

    answer = sa.resolve_holdout_self_backtest("cme_futures", "cbd72d8408d3")

    assert answer.backtest_hash is None
    assert "not been evaluated" in answer.reason


def test_the_canonical_carrier_asking_for_its_own_replay_never_raises(monkeypatch) -> None:
    monkeypatch.setattr(sa, "resolve_canonical_rank1_lineage", _canonical())
    monkeypatch.setattr(
        sa,
        "_resolve_holdout_self_backtest",
        lambda cs, h: sa.HoldoutSelfBacktest(None, "none of them replays that run's strategy"),
    )

    answer = sa.resolve_holdout_self_backtest("cme_futures", CANONICAL["val_backtest_hash"])

    assert answer.backtest_hash is None


def test_a_resolver_that_cannot_answer_leaves_the_original_answer_standing(monkeypatch) -> None:
    """An empty registry must not turn a missing holdout into a resolver error."""

    def _raise(case_study, **kwargs):
        raise RuntimeError("no validation rank-1 candidate")

    monkeypatch.setattr(sa, "resolve_canonical_rank1_lineage", _raise)
    monkeypatch.setattr(
        sa,
        "_resolve_holdout_self_backtest",
        lambda cs, h: sa.HoldoutSelfBacktest(None, "the holdout has not been evaluated"),
    )

    assert sa.resolve_holdout_self_backtest("cme_futures", "cbd72d8408d3").backtest_hash is None


def test_a_found_replay_is_returned_without_consulting_the_resolver(monkeypatch) -> None:
    def _fail(case_study, **kwargs):
        raise AssertionError("the diagnosis ran on a successful lookup")

    monkeypatch.setattr(sa, "resolve_canonical_rank1_lineage", _fail)
    monkeypatch.setattr(
        sa, "_resolve_holdout_self_backtest", lambda cs, h: sa.HoldoutSelfBacktest("aa11bb22cc33")
    )

    assert sa.resolve_holdout_self_backtest("cme_futures", "x").backtest_hash == "aa11bb22cc33"


def test_select_holdout_self_backtest_raises_through_the_same_path(registry, monkeypatch) -> None:
    """The six notebooks that print "not produced yet" call this one, not the resolver."""
    monkeypatch.setattr(sa, "resolve_canonical_rank1_lineage", _canonical())
    monkeypatch.setattr(
        sa,
        "_resolve_holdout_self_backtest",
        lambda cs, h: sa.HoldoutSelfBacktest(None, "no holdout backtest is registered"),
    )

    with pytest.raises(RuntimeError, match="selection disagreement"):
        sa.select_holdout_self_backtest("cme_futures", "cbd72d8408d3")


def test_a_hash_the_run_log_has_never_seen_keeps_its_own_answer(registry, monkeypatch) -> None:
    """A stale hardcoded hash is not a selection, and the specific answer is more useful."""
    monkeypatch.setattr(sa, "resolve_canonical_rank1_lineage", _canonical())
    monkeypatch.setattr(
        sa,
        "_resolve_holdout_self_backtest",
        lambda cs, h: sa.HoldoutSelfBacktest(None, "is not registered in the run log"),
    )

    answer = sa.resolve_holdout_self_backtest("cme_futures", "never_registered")

    assert answer.backtest_hash is None
    assert "not registered" in answer.reason


def test_the_canonical_resolver_does_not_re_enter_the_diagnosis(monkeypatch, tmp_path) -> None:
    """Unmocked, on a registry with no holdout - the shape that used to recurse.

    `resolve_canonical_rank1_lineage` looks the holdout up for its own rank-1, and the
    diagnosis asks that resolver for the canonical carrier. Going through the diagnosing
    wrapper there re-entered it: each call resolved the whole ranking again one level
    deeper until the recursion limit, and the diagnosis swallowed the RecursionError after
    hundreds of registry reads. The resolver takes the raw lookup now - it IS the canonical
    selection, so it can never disagree with itself.
    """
    import sqlite3

    case_dir = tmp_path / "cs"
    (case_dir / "run_log").mkdir(parents=True)
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.executescript(
        """
        CREATE TABLE training_runs (
            training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT,
            spec_json TEXT);
        CREATE TABLE prediction_sets (
            prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT,
            checkpoint_value INTEGER, checkpoint_kind TEXT);
        CREATE TABLE backtest_runs (
            backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT,
            spec_json TEXT);
        CREATE TABLE backtest_metrics (backtest_hash TEXT PRIMARY KEY, sharpe REAL);
        CREATE TABLE fold_metrics (
            prediction_hash TEXT, fold INTEGER, ic REAL, n_days INTEGER);
        CREATE TABLE prediction_metrics (prediction_hash TEXT PRIMARY KEY, ic_mean REAL);
        CREATE TABLE prediction_coverage (prediction_hash TEXT PRIMARY KEY, status TEXT);

        INSERT INTO training_runs VALUES ('t1', 'gbm', 'leaves_31_mse', 'fwd_ret_5d', '{}');
        INSERT INTO prediction_sets VALUES ('p1', 't1', 'validation', 100, 'iteration');
        INSERT INTO backtest_runs VALUES ('b1', 'p1', 'signal', '{"strategy": {"a": 1}}');
        INSERT INTO backtest_metrics VALUES ('b1', 1.2);
        """
    )
    db.commit()
    db.close()
    monkeypatch.setattr(sa, "get_case_study_dir", lambda cs, **kw: case_dir, raising=False)
    import utils.paths

    monkeypatch.setattr(utils.paths, "get_case_study_dir", lambda cs, **kw: case_dir)

    calls = {"n": 0}
    real = sa.resolve_canonical_rank1_lineage

    def _counted(case_study, **kwargs):
        calls["n"] += 1
        assert calls["n"] < 5, "the canonical resolver re-entered the diagnosis"
        return real(case_study, **kwargs)

    monkeypatch.setattr(sa, "resolve_canonical_rank1_lineage", _counted)

    lineage = _counted("cs")

    assert lineage["val_backtest_hash"] == "b1"
    assert lineage["holdout_backtest_hash"] is None
    assert calls["n"] == 1
