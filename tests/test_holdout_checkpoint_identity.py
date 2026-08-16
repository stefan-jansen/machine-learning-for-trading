"""The holdout replay is pinned to the validation set's checkpoint.

``reference/CASE_STUDY_PIPELINE.md`` §5 makes the checkpoint part of the model
configuration and §6 allows the holdout exactly one use, on the selected
configuration. A lookup keyed on ``training_hash`` alone leaves one holdout
candidate per declared checkpoint, all carrying the same strategy spec, and
resolving them by holdout Sharpe reads the holdout to choose among
configurations.

Exposure scales with ``backtest.sweep.checkpoints_per_config``: ``etfs``
advances two checkpoints per configuration, so it registers two
indistinguishable holdout candidates for every carrier.
"""

import json
import sqlite3

import pytest

from case_studies.utils.paired_metrics import _holdout_lineage_for
from case_studies.utils.registry.store import REGISTRY_SCHEMA_SQL
from case_studies.utils.strategy_analysis import select_holdout_self_backtest

TRAINING_HASH = "t_gbm_leaves_7_mae"
STRATEGY = {"signal": {"method": "score_weighted_top_k", "top_k": 10}}
OTHER_STRATEGY = {"signal": {"method": "score_weighted_top_k", "top_k": 20}}


def _spec(strategy: dict) -> str:
    return json.dumps({"strategy": strategy})


def _build_registry(case_dir, *, checkpoints=(200, 400), holdout_sharpes=(0.4, 1.9)):
    """One training run, one prediction set per checkpoint per split.

    The strategy spec is identical across checkpoints, which is what makes the
    candidates indistinguishable without the checkpoint pin. By default the
    second checkpoint is given the better holdout Sharpe, so a lookup ordering on
    Sharpe picks it; a caller can invert `holdout_sharpes` to point Sharpe order
    at the first checkpoint instead, which is how a case makes its carrier win
    neither the `backtest_hash` nor the `sharpe` tiebreak.
    """
    run_log = case_dir / "run_log"
    run_log.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(run_log / "registry.db"))
    db.executescript(REGISTRY_SCHEMA_SQL)
    db.execute(
        "INSERT INTO training_runs (training_hash, family, label, config_name, created_at)"
        " VALUES (?, 'gbm', 'fwd_ret_21d', 'leaves_7_mae', '2026-08-16T00:00:00+00:00')",
        (TRAINING_HASH,),
    )
    for checkpoint, holdout_sharpe in zip(checkpoints, holdout_sharpes, strict=True):
        # A configuration with no checkpoint dimension stores NULL in *both*
        # columns, which is what the linear and GBM holdout rows look like. Binding
        # the kind to the value is what makes the `checkpoint_kind IS ?` half of the
        # guard fail when it is written as `=`.
        checkpoint_kind = "iteration" if checkpoint is not None else None
        for split in ("validation", "holdout"):
            pred = f"p_{split}_{checkpoint}"
            backtest = f"b_{split}_{checkpoint}"
            db.execute(
                "INSERT INTO prediction_sets (prediction_hash, training_hash,"
                " checkpoint_value, checkpoint_kind, split, created_at)"
                " VALUES (?, ?, ?, ?, ?, '2026-08-16T00:00:00+00:00')",
                (pred, TRAINING_HASH, checkpoint, checkpoint_kind, split),
            )
            db.execute(
                "INSERT INTO backtest_runs (backtest_hash, prediction_hash, spec_json,"
                " stage, created_at) VALUES (?, ?, ?, 'signal', '2026-08-16T00:00:00+00:00')",
                (backtest, pred, _spec(STRATEGY)),
            )
            sharpe = holdout_sharpe if split == "holdout" else 1.0
            db.execute(
                "INSERT INTO backtest_metrics (backtest_hash, computed_at, sharpe)"
                " VALUES (?, '2026-08-16T00:00:00+00:00', ?)",
                (backtest, sharpe),
            )
    db.commit()
    db.close()


@pytest.fixture
def case_study(tmp_path, monkeypatch):
    """Redirect both resolvers, which are bound differently.

    ``strategy_analysis`` imports ``get_case_study_dir`` inside the function, so
    patching ``utils.paths`` reaches it; ``paired_metrics`` binds it at module
    import, so it needs its own patch or the test silently reads the real
    registry instead of the fixture.
    """
    case_dir = tmp_path / "etfs"
    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda _: case_dir)
    monkeypatch.setattr("case_studies.utils.paired_metrics.get_case_study_dir", lambda _: case_dir)
    return case_dir


def test_holdout_replay_uses_the_validation_checkpoint_not_the_better_one(case_study):
    """The carrier is checkpoint 200; checkpoint 400 has the higher holdout Sharpe."""
    _build_registry(case_study)

    assert select_holdout_self_backtest("etfs", "b_validation_200") == "b_holdout_200"


def test_each_checkpoint_replays_onto_its_own_holdout(case_study):
    """Both directions, so the pin is not satisfied by always returning the first row."""
    _build_registry(case_study)

    assert select_holdout_self_backtest("etfs", "b_validation_400") == "b_holdout_400"


def test_a_configuration_without_checkpoints_still_matches(case_study):
    """``IS`` is null-safe; ``=`` would drop a linear run storing NULL on both sides."""
    _build_registry(case_study, checkpoints=(None,), holdout_sharpes=(0.7,))

    assert select_holdout_self_backtest("etfs", "b_validation_None") == "b_holdout_None"


def test_a_diverging_strategy_spec_is_not_a_replay(case_study):
    """The existing allocator-variant guard survives the checkpoint pin."""
    _build_registry(case_study, checkpoints=(200,), holdout_sharpes=(0.4,))
    db = sqlite3.connect(str(case_study / "run_log" / "registry.db"))
    db.execute(
        "UPDATE backtest_runs SET spec_json = ? WHERE backtest_hash = 'b_holdout_200'",
        (_spec(OTHER_STRATEGY),),
    )
    db.commit()
    db.close()

    assert select_holdout_self_backtest("etfs", "b_validation_200") is None


def test_both_resolvers_agree_on_one_hash_for_the_same_carrier(case_study):
    """Given the same carrier prediction set, writer and reader return one hash.

    `_holdout_lineage_for` picks the holdout the `val_rank1_self` pair is stored
    against; `select_holdout_self_backtest` is what the strategy-analysis
    notebooks pass to `load_paired_metrics` to find it again. Pinning one side
    only makes them disagree exactly where a training run has holdout backtests
    at more than one checkpoint, and the pair lookup then returns empty: `etfs`
    reports NaN val-to-holdout decay and `us_firm_characteristics` raises.

    The carrier is the checkpoint that wins neither tiebreak, which is what makes
    the case discriminate against both ways of losing the pin. `_holdout_lineage_for`
    reaches its pinned query only inside the `prefer_prediction_hash` branch and
    otherwise falls through to an unpinned `ORDER BY bm.sharpe DESC`. So carrier
    400 with the Sharpes inverted for this case is neither hash-first (200 sorts
    before 400) nor Sharpe-best (200 is given 1.9 here): dropping the checkpoint
    clauses yields `b_holdout_200` by hash order, and dropping the whole
    preference branch yields `b_holdout_200` by Sharpe order. Both fail.

    The default Sharpes are left alone because
    `test_holdout_replay_uses_the_validation_checkpoint_not_the_better_one` needs
    checkpoint 400 to be the better one for its own assertion to mean anything.

    Scope, stated because the name would otherwise overclaim: this covers the two
    resolvers, not the wiring that hands `_holdout_lineage_for` its carrier.
    Mutating that call site does not fail this test. What removes that class of
    error is the signature rather than this assertion - the function takes one
    prediction hash and derives both the training hash and the checkpoint from
    it, so a caller cannot supply half a pin; it can only omit the preference
    entirely and fall back.
    """
    _build_registry(case_study, holdout_sharpes=(1.9, 0.4))

    writer = _holdout_lineage_for(
        "etfs",
        "fwd_ret_21d",
        strategy_spec=None,
        label_restriction=None,
        rung=None,
        prefer_prediction_hash="p_validation_400",
    )
    reader = select_holdout_self_backtest("etfs", "b_validation_400")

    assert writer is not None
    assert writer["backtest_hash"] == reader == "b_holdout_400"


def test_an_ambiguous_pinned_lineage_raises_rather_than_choosing(case_study):
    """Two holdout backtests on one checkpoint and one spec is unresolvable."""
    _build_registry(case_study, checkpoints=(200,), holdout_sharpes=(0.4,))
    db = sqlite3.connect(str(case_study / "run_log" / "registry.db"))
    db.execute(
        "INSERT INTO backtest_runs (backtest_hash, prediction_hash, spec_json, stage, created_at)"
        " VALUES ('b_holdout_200_dup', 'p_holdout_200', ?, 'signal',"
        " '2026-08-16T00:00:00+00:00')",
        (_spec(STRATEGY),),
    )
    db.commit()
    db.close()

    with pytest.raises(ValueError, match="ambiguous"):
        select_holdout_self_backtest("etfs", "b_validation_200")
