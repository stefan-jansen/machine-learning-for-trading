"""The holdout anchor is never chosen by its own holdout result.

The invariant is the file's original one and it is unchanged. What changed is the mechanism
that enforces it. A research lock used to name the sealed carrier and the run made from it, so
ambiguity among holdout candidates was resolved by reading the lock. PR #685 deleted that layer
- the selection rule is the whole mechanism now - and these tests pin the two things that hold
the invariant up in its absence:

* the caller names the validation carrier, and the checkpoint comes with it; and
* where no carrier is named and more than one candidate survives, the resolver REFUSES.

Refusing is the point. Every way of choosing between surviving candidates - Sharpe, row order,
`backtest_hash` ascending - decides on something the holdout produced, which is the selection
the holdout exists to rule out.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils import paired_metrics

#: A training spec whose own CV declares the holdout fold - what a refit for the holdout looks
#: like, and what `training_run_fitted_for_the_holdout` reads.
HOLDOUT_REFIT_SPEC = json.dumps({"computation": {"cv": {"split": "holdout"}}})

#: A model fitted on the validation folds. It can publish predictions over the holdout window,
#: and it is not a holdout result whatever it scores.
VALIDATION_FITTED_SPEC = json.dumps({"computation": {"cv": {"split": "validation"}}})


def _registry(tmp_path: Path, rows) -> Path:
    """``rows`` are (prediction_hash, training_hash, config_name, backtest_hash, sharpe).

    Optionally a 6th element, the training spec, and a 7th, the checkpoint value. They default
    to a holdout refit at checkpoint 50, which is what every row was before this file had a
    case that needed to vary them.

    There are no `research_locks` or `holdout_evaluations` tables. #685 dropped both from the
    schema, so a fixture that created them would be testing against a registry shape the code
    can no longer produce.
    """
    case_dir = tmp_path / "probe"
    (case_dir / "run_log").mkdir(parents=True)
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.execute(
        "CREATE TABLE prediction_sets (prediction_hash TEXT, training_hash TEXT, split TEXT, "
        "checkpoint_kind TEXT, checkpoint_value INTEGER)"
    )
    db.execute(
        "CREATE TABLE training_runs (training_hash TEXT, family TEXT, config_name TEXT, "
        "label TEXT, spec_json TEXT)"
    )
    db.execute(
        "CREATE TABLE backtest_runs (backtest_hash TEXT, prediction_hash TEXT, stage TEXT, "
        "spec_json TEXT)"
    )
    db.execute("CREATE TABLE backtest_metrics (backtest_hash TEXT, sharpe REAL)")
    seen_predictions: set[str] = set()
    seen_training: set[str] = set()
    for row in rows:
        prediction_hash, training_hash, config_name, backtest_hash, sharpe = row[:5]
        spec = row[5] if len(row) > 5 else HOLDOUT_REFIT_SPEC
        checkpoint = row[6] if len(row) > 6 else 50
        if prediction_hash not in seen_predictions:
            db.execute(
                "INSERT INTO prediction_sets VALUES (?,?,?,?,?)",
                (prediction_hash, training_hash, "holdout", "epoch", checkpoint),
            )
            seen_predictions.add(prediction_hash)
        if training_hash not in seen_training:
            db.execute(
                "INSERT INTO training_runs VALUES (?,?,?,?,?)",
                (training_hash, "gbm", config_name, "fwd_ret_21d", spec),
            )
            seen_training.add(training_hash)
        db.execute(
            "INSERT INTO backtest_runs VALUES (?,?,?,?)",
            (backtest_hash, prediction_hash, "holdout", "{}"),
        )
        db.execute("INSERT INTO backtest_metrics VALUES (?,?)", (backtest_hash, sharpe))
    db.commit()
    db.close()
    return case_dir


@pytest.fixture(autouse=True)
def _clear_cache():
    paired_metrics._retired_prediction_hashes.cache_clear()
    yield
    paired_metrics._retired_prediction_hashes.cache_clear()


def _install(monkeypatch, case_dir: Path) -> None:
    monkeypatch.setattr(paired_metrics, "get_case_study_dir", lambda cs: case_dir)
    monkeypatch.setattr(
        "case_studies.research.population.superseded_members_at",
        lambda _dir, member_kind="prediction": frozenset(),
    )


def _lineage(cs="probe", **kwargs):
    return paired_metrics._holdout_lineage_for(
        cs, "fwd_ret_21d", None, label_restriction=None, rung=None, **kwargs
    )


def test_one_lineage_resolves(monkeypatch, tmp_path) -> None:
    case_dir = _registry(tmp_path, [("p1", "t1", "cfg", "b1", 0.4)])
    _install(monkeypatch, case_dir)

    assert _lineage()["backtest_hash"] == "b1"


def test_no_candidates_is_not_an_error(monkeypatch, tmp_path) -> None:
    case_dir = _registry(tmp_path, [])
    _install(monkeypatch, case_dir)

    assert _lineage() is None


def test_several_trained_models_refuse_rather_than_rank(monkeypatch, tmp_path) -> None:
    """Nothing records which validation carrier each retrain came from, so the higher
    Sharpe is not evidence - it is how a holdout from a retired carrier would win."""
    case_dir = _registry(
        tmp_path,
        [
            ("p1", "t1", "cfg_a", "b1", 0.4),
            ("p2", "t2", "cfg_b", "b2", 9.9),
        ],
    )
    _install(monkeypatch, case_dir)

    with pytest.raises(ValueError, match="rank the holdout on its own result"):
        _lineage()


def test_several_checkpoints_of_one_model_refuse(monkeypatch, tmp_path) -> None:
    """One trained model registers one prediction set per declared checkpoint, and they share
    a strategy spec. Before the lock was deleted this case fell through the multi-model
    refusal - which counted DISTINCT training hashes, and here there is one - and landed on
    `rows[0]` under `ORDER BY b.backtest_hash`. `b1` carries 9.9 and `b2` carries 0.1, so what
    came back was the higher-scoring checkpoint, chosen on nothing but its holdout result.
    """
    case_dir = _registry(
        tmp_path,
        [
            ("p1", "t1", "cfg_a", "b1", 9.9, HOLDOUT_REFIT_SPEC, 50),
            ("p2", "t1", "cfg_a", "b2", 0.1, HOLDOUT_REFIT_SPEC, 60),
        ],
    )
    _install(monkeypatch, case_dir)

    with pytest.raises(ValueError, match="rank the holdout on its own result"):
        _lineage()


def test_sibling_backtests_on_one_prediction_refuse(monkeypatch, tmp_path) -> None:
    """One prediction set can carry several backtests - a replay under a different strategy
    spec, an experimental allocator sharing the holdout prediction. Pinning the prediction
    alone leaves the choice to `backtest_hash` ascending.

    The sibling is named `b0` and carries the higher Sharpe deliberately: with the refusal
    absent, ordering returns `b0` and the test would pass while the resolver was picking on
    the holdout's own result. Naming them the other way round would hide that.
    """
    case_dir = _registry(
        tmp_path,
        [
            ("p1", "t1", "cfg_a", "b0", 9.9),
            ("p1", "t1", "cfg_a", "b9", 0.4),
        ],
    )
    _install(monkeypatch, case_dir)

    with pytest.raises(ValueError, match="rank the holdout on its own result"):
        _lineage()


def test_the_named_carrier_pins_the_checkpoint_over_a_higher_score(monkeypatch, tmp_path) -> None:
    """The replacement for what the lock used to do, and the reason refusing is not a dead end.

    The caller knows which validation run was selected. Naming its prediction set pins the
    configuration AND the checkpoint, so the holdout is resolved from the carrier rather than
    from the scores - here `p2`/`b2` at 0.1, while `b1` sits at 9.9 and is not chosen.
    """
    case_dir = _registry(
        tmp_path,
        [
            ("p1", "t1", "cfg_a", "b1", 9.9, HOLDOUT_REFIT_SPEC, 50),
            ("p2", "t1", "cfg_a", "b2", 0.1, HOLDOUT_REFIT_SPEC, 60),
        ],
    )
    _install(monkeypatch, case_dir)

    assert _lineage(prefer_prediction_hash="p2")["backtest_hash"] == "b2"


def test_a_validation_fitted_run_is_not_a_holdout_however_it_scores(monkeypatch, tmp_path) -> None:
    """The eligibility filter, which the refusal sits behind rather than replaces.

    `t2` is fitted on the validation folds and publishes over the holdout window at 9.9. It is
    dropped before the count, so the one genuine refit resolves instead of the pair refusing -
    which also shows the refusal is not firing on rows that were never candidates.
    """
    case_dir = _registry(
        tmp_path,
        [
            ("p1", "t1", "cfg_a", "b1", 0.4),
            ("p2", "t2", "cfg_b", "b2", 9.9, VALIDATION_FITTED_SPEC),
        ],
    )
    _install(monkeypatch, case_dir)

    assert _lineage()["backtest_hash"] == "b1"


def test_a_retired_prediction_is_not_a_candidate(monkeypatch, tmp_path) -> None:
    """Retirement is passed in, not looked up, and it filters on the prediction hash the row
    actually carries.

    `retired_hashes` is a parameter rather than a call to `_retired_prediction_hashes` inside
    the function, so a test that monkeypatches the helper patches something this code path
    never reaches - it would pass whatever the filter did. Retiring the only candidate leaves
    none, which is `None` and not a refusal: there is nothing ambiguous about an empty set.
    """
    case_dir = _registry(tmp_path, [("p1", "t1", "cfg_a", "b1", 0.4)])
    _install(monkeypatch, case_dir)

    assert _lineage(retired_hashes=frozenset({"p1"})) is None
    assert _lineage(retired_hashes=frozenset({"other"}))["backtest_hash"] == "b1"


def test_the_pinned_carrier_branch_refuses_ambiguity_too(monkeypatch, tmp_path) -> None:
    """Naming the carrier is normally one candidate, and it is not guaranteed to be.

    One prediction set can carry several backtests, and they survive the carrier filter
    together - same configuration, same checkpoint, same strategy. The pinned branch used to
    return the first eligible row in `backtest_hash` order while the unpinned branch below it
    refused the identical ambiguity, so which resolver you asked decided whether the registry
    was ambiguous. The higher Sharpe is on `b0`, which is what hash order returns.
    """
    case_dir = _registry(
        tmp_path,
        [
            ("p1", "t1", "cfg_a", "b0", 9.9),
            ("p1", "t1", "cfg_a", "b9", 0.4),
        ],
    )
    _install(monkeypatch, case_dir)

    with pytest.raises(ValueError, match="rank the holdout on its own result"):
        _lineage(prefer_prediction_hash="p1")


def test_the_pinned_carrier_still_resolves_when_it_is_unambiguous(monkeypatch, tmp_path) -> None:
    """The refusal above must not fire on the ordinary case, which is the one that matters:
    a carrier with one holdout replay resolves, and it is not the higher-scoring sibling
    checkpoint that the unpinned branch would have to refuse over."""
    case_dir = _registry(
        tmp_path,
        [
            ("p1", "t1", "cfg_a", "b1", 9.9, HOLDOUT_REFIT_SPEC, 50),
            ("p2", "t1", "cfg_a", "b2", 0.1, HOLDOUT_REFIT_SPEC, 60),
        ],
    )
    _install(monkeypatch, case_dir)

    assert _lineage(prefer_prediction_hash="p1")["backtest_hash"] == "b1"
    assert _lineage(prefer_prediction_hash="p2")["backtest_hash"] == "b2"
