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


def test_a_named_carrier_with_no_holdout_does_not_borrow_a_siblings(monkeypatch, tmp_path) -> None:
    """The pin is a pin, not a preference.

    Only `p2`/checkpoint 60 has a holdout here. Asking for `p1`'s must answer None: the
    pinned branch used to fall through to the unpinned query when its carrier had no
    eligible row, and with one candidate left in the registry that query returns `b2`
    happily. The reader-facing table would then report checkpoint 60's holdout under the
    name of the checkpoint validation actually selected.

    Asking for `p2` in the same registry still resolves, so the None above is the pin
    refusing to borrow rather than the fixture being empty.
    """
    case_dir = _registry(
        tmp_path,
        [("p2", "t1", "cfg_a", "b2", 9.9, HOLDOUT_REFIT_SPEC, 60)],
    )
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.execute(
        "INSERT INTO prediction_sets VALUES ('p1','t1','holdout','epoch',50)",
    )
    db.commit()
    db.close()
    _install(monkeypatch, case_dir)

    assert _lineage(prefer_prediction_hash="p1") is None
    assert _lineage(prefer_prediction_hash="p2")["backtest_hash"] == "b2"


def test_a_carrier_the_registry_does_not_have_resolves_to_nothing(monkeypatch, tmp_path) -> None:
    """Naming an unregistered prediction set is a question about a carrier that is not
    there. Falling through would answer it with some other case's holdout."""
    case_dir = _registry(tmp_path, [("p1", "t1", "cfg_a", "b1", 0.4)])
    _install(monkeypatch, case_dir)

    assert _lineage(prefer_prediction_hash="not_registered") is None
    assert _lineage(prefer_prediction_hash="p1")["backtest_hash"] == "b1"


# ---------------------------------------------------------------------------
# The Chapter 20 caller.
#
# The resolver is single-sourced - ch20's `_holdout_lineage_for` is a delegate - so the twelve
# tests above cover the RULE for both producers. What they do not reach is the WIRING, and the
# wiring is where this went wrong twice: once passing a spec from one candidate alongside a
# prediction hash from another, and once letting a missing carrier fall through to an unpinned
# query. Both were in the copy, not the original. These run the real Chapter 20 function.
# ---------------------------------------------------------------------------

CH20_SOURCE = (
    Path(__file__).resolve().parents[1] / "20_strategy_synthesis" / "01_aggregate_synthesis.py"
)


def _ch20(name: str, namespace: dict):
    """Lift one Chapter 20 function into ``namespace`` without importing the notebook.

    The module executes a synthesis over nine registries at import, so it cannot be imported
    in a unit test. Re-implementing the function here would test the copy in this file rather
    than the one that ships, which is the exact failure mode being pinned.
    """
    import ast

    tree = ast.parse(CH20_SOURCE.read_text())
    definition = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(  # noqa: S102 - running the shipped producer is the point
        compile(ast.Module(body=[definition], type_ignores=[]), str(CH20_SOURCE), "exec"),
        namespace,
    )
    return namespace[name]


def _ch20_namespace(tmp_path: Path, carrier, calls: list) -> dict:
    """A namespace for `query_holdout_rows` over one case study with a populated registry."""
    case_dir = tmp_path / "probe"
    (case_dir / "run_log").mkdir(parents=True)
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.executescript(
        """
        CREATE TABLE training_runs (training_hash TEXT, family TEXT, config_name TEXT, label TEXT);
        CREATE TABLE prediction_sets (prediction_hash TEXT, training_hash TEXT);
        CREATE TABLE backtest_runs (backtest_hash TEXT, prediction_hash TEXT, stage TEXT);
        CREATE TABLE backtest_metrics (backtest_hash TEXT, sharpe REAL, max_drawdown REAL,
                                       cagr REAL, num_trades INTEGER);
        CREATE TABLE prediction_metrics (prediction_hash TEXT, ic_mean REAL);
        INSERT INTO training_runs VALUES ('train_ho', 'gbm', 'cfg_a', 'fwd_ret_1m');
        INSERT INTO prediction_sets VALUES ('pred_ho', 'train_ho');
        INSERT INTO backtest_runs VALUES ('bt_ho', 'pred_ho', 'holdout');
        INSERT INTO backtest_metrics VALUES ('bt_ho', 1.25, -0.1, 0.2, 40);
        INSERT INTO prediction_metrics VALUES ('pred_ho', 0.03);
        """
    )
    db.commit()
    db.close()

    def _lineage(cs, leader_label, strategy_spec=None, *, prefer_prediction_hash=None):
        calls.append(
            {
                "cs": cs,
                "spec": strategy_spec,
                "prefer_prediction_hash": prefer_prediction_hash,
            }
        )
        return {"backtest_hash": "bt_ho", "label": "fwd_ret_1m"}

    return {
        "ALL_CASE_STUDIES": ["probe"],
        "DISPLAY_NAMES": {"probe": "Probe"},
        "get_case_study_dir": lambda cs: case_dir,
        "_val_rank1_carrier": lambda cs: carrier,
        "_holdout_lineage_for": _lineage,
        "_optional_metric": lambda db_, table, column, alias: f"NULL AS {alias}",
        "sqlite3": sqlite3,
    }


def test_chapter_20_publishes_no_holdout_row_where_nothing_selected_one(tmp_path, capsys):
    """No validation carrier is an answer, and the answer is no row.

    The carrier IS the selection. Falling through to an unpinned query here would publish the
    one eligible holdout the registry happens to hold - a holdout chosen by its own holdout
    result, which is the thing the whole mechanism exists to prevent. The registry in this
    fixture HAS a usable holdout, so a caller that falls through returns a row and fails.
    """
    calls: list = []
    namespace = _ch20_namespace(tmp_path, carrier=None, calls=calls)
    query_holdout_rows = _ch20("query_holdout_rows", namespace)

    assert query_holdout_rows() == []
    assert calls == [], "the resolver must not be reached at all without a carrier"
    assert "no rank-1 validation carrier" in capsys.readouterr().out


def test_chapter_20_pins_the_carriers_own_prediction_hash(tmp_path):
    """The spec and the prediction hash come from the SAME carrier.

    Passing one candidate's spec alongside another's hash asks for a holdout matching neither.
    That was live until the pin became strict, masked by the unpinned fall-through.
    """
    calls: list = []
    carrier = {"spec": {"signal": {"method": "equal_weight_top_k"}}, "prediction_hash": "pred_val"}
    namespace = _ch20_namespace(tmp_path, carrier=carrier, calls=calls)
    query_holdout_rows = _ch20("query_holdout_rows", namespace)

    rows = query_holdout_rows()

    assert len(calls) == 1
    assert calls[0]["prefer_prediction_hash"] == "pred_val"
    assert calls[0]["spec"] is carrier["spec"]
    assert [row["holdout_backtest_hash"] for row in rows] == ["bt_ho"]


def test_chapter_20_reports_a_refusal_rather_than_dropping_the_case_study(tmp_path, capsys):
    """A refusal is per case study and it is printed.

    A case study silently missing from the holdout table looks like unrun work, which is how a
    resolver refusal would be read as a scheduling problem instead of an ambiguity to settle.
    """
    calls: list = []
    carrier = {"spec": {}, "prediction_hash": "pred_val"}
    namespace = _ch20_namespace(tmp_path, carrier=carrier, calls=calls)

    def _refuse(cs, leader_label, strategy_spec=None, *, prefer_prediction_hash=None):
        raise ValueError("two candidates survive")

    namespace["_holdout_lineage_for"] = _refuse
    query_holdout_rows = _ch20("query_holdout_rows", namespace)

    assert query_holdout_rows() == []
    assert "two candidates survive" in capsys.readouterr().out
