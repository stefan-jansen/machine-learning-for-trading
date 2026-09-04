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
