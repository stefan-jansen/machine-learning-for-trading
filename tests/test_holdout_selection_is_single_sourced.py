"""Holdout selection has one implementation, so it cannot answer two ways.

The function that decides which configuration receives the case study's single holdout
use existed twice: `20_strategy_synthesis/holdout.py::select_best_models` built its own
pool out of `BacktestExplorer.best` per stage, and
`case_studies/utils/strategy_analysis.py::resolve_canonical_rank1_lineage` built one in
SQL. A comment said to keep them in sync. They were not in sync: the first admitted a
prediction its case study publishes, the second admitted any prediction nobody had
retired, and the two orderings broke an exact Sharpe tie differently.

Measured on the `fx_pairs` production registry 2026-09-07: `deep_learning/tcn` on
`fwd_ret_21d` carries two backtests tied at Sharpe 0.2639142245820113 - `9402978117e9` at
the allocation stage and `56070f34dff1` at the risk overlay - and `select_best_models`
answered the first while the resolver answered the second. Two strategy specifications for
one model, and `select_holdout_self_backtest` matches the specification exactly, so the
disagreement decided which strategy spent the holdout.

The duplicate is gone. `20_strategy_synthesis/holdout.py` was deleted once Chapter 20
stopped generating holdouts, and `resolve_canonical_rank1_lineage` is the only path left,
so what is pinned here is no longer an agreement between two implementations. It is the
set of rules that decides which configuration spends the holdout: the tie-break, the
declared label restriction, publication membership, the common-support re-ranking, the
coverage bar, and the refusal. Each is a property of the surviving path and each of these
tests fails if that path stops applying it.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

import polars as pl
import pytest

import case_studies.utils.uncertainty as uncertainty
from case_studies.utils import strategy_analysis

CASE_STUDY = "fixture_case_study"

_SIGNAL_ONLY = json.dumps({"strategy": {"signal": {"method": "equal_weight_top_k", "top_k": 10}}})
_WITH_ALLOCATION = json.dumps(
    {
        "strategy": {
            "signal": {"method": "equal_weight_top_k", "top_k": 10},
            "allocation": {"method": "mvo_ledoit_wolf", "lookback": 63},
        }
    }
)


def _registry(
    path: Path,
    rows: list[tuple[str, str, str, float, str]],
    *,
    published: list[str] | None = None,
) -> None:
    """A registry both selection paths can rank.

    Each row is ``(backtest_hash, prediction_hash, stage, sharpe, spec_json)``. Every
    prediction hash named gets its own training run, prediction set and metrics row, at
    identical ``ic_n_days`` so the full-coverage bar admits all of them. ``published``, when
    given, is the prediction population the case study declares - the members it stands
    behind - written as one generation under one name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT,
                spec_json TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT,
                checkpoint_value REAL, checkpoint_kind TEXT
            );
            CREATE TABLE prediction_metrics (
                prediction_hash TEXT PRIMARY KEY, ic_mean REAL, ic_mean_daily REAL,
                ic_ci_lo REAL, ic_ci_hi REAL, ic_n_days REAL
            );
            CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL);
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, spec_json TEXT, stage TEXT
            );
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY, sharpe REAL, cagr REAL, max_drawdown REAL,
                total_return REAL, volatility REAL, num_trades REAL
            );
            CREATE TABLE official_populations (
                population_hash TEXT PRIMARY KEY, name TEXT, member_kind TEXT,
                supersedes_hash TEXT
            );
            CREATE TABLE official_population_members (
                population_hash TEXT, member_hash TEXT
            );
            """
        )
        for prediction_hash in sorted({prediction for _, prediction, _, _, _ in rows}):
            training_hash = f"train_{prediction_hash}"
            db.execute(
                "INSERT INTO training_runs VALUES (?, 'gbm', ?, 'fwd_ret_5d', '{}')",
                (training_hash, f"config_{prediction_hash}"),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, 'validation', NULL, NULL)",
                (prediction_hash, training_hash),
            )
            db.execute(
                "INSERT INTO prediction_metrics VALUES (?, 0.02, 0.02, 0.0, 0.04, 250)",
                (prediction_hash,),
            )
            db.execute("INSERT INTO fold_metrics VALUES (?, 0.02)", (prediction_hash,))
        for backtest_hash, prediction_hash, stage, sharpe, spec_json in rows:
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, ?, ?, ?)",
                (backtest_hash, prediction_hash, spec_json, stage),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, 0.1, -0.2, 0.3, 0.1, 100)",
                (backtest_hash, sharpe),
            )
        if published is not None:
            db.execute(
                "INSERT INTO official_populations VALUES "
                "('pop_generation_1', 'fixture-predictions', 'prediction', NULL)"
            )
            db.executemany(
                "INSERT INTO official_population_members VALUES ('pop_generation_1', ?)",
                [(member,) for member in sorted(published)],
            )


@pytest.fixture
def case_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    resolve = lambda case_study, **_: tmp_path / case_study  # noqa: E731
    # `strategy_analysis` imports this inside each function, so patching the source
    # module reaches it.
    monkeypatch.setattr("utils.paths.get_case_study_dir", resolve)
    return tmp_path / CASE_STUDY


def _selected(case_study: str) -> str:
    """The configuration the holdout selection names, as a validation backtest hash."""
    return strategy_analysis.resolve_canonical_rank1_lineage(case_study)["val_backtest_hash"]


def _selected_sharpe(case_study: str) -> float:
    """The Sharpe the selection reports for the configuration it named."""
    return strategy_analysis.resolve_canonical_rank1_lineage(case_study)["val_sharpe"]


def test_an_exact_sharpe_tie_resolves_to_the_specification_the_holdout_replays(
    case_dir: Path,
) -> None:
    """The fx_pairs shape: two specifications, identical Sharpe, one tie-break.

    Under a tie the signal-only specification wins, because that is the one
    `select_holdout_self_backtest` replays, so the selected lineage stays poolable with
    its own holdout. The deleted implementation built its pool stage by stage, knew
    nothing about that rule, and ordered on the stage it happened to visit first - which
    is how `fx_pairs` got two answers for one tie.
    """
    # `aa_allocation_row` sorts first on the final `backtest_hash ASC` key, so the only
    # thing that can put the signal-only row in front of it is the tie-break itself.
    # Naming it `zz_` made the assertion pass whether the rule was applied or not.
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("aa_allocation_row", "pred_b", "allocation", 2.0, _WITH_ALLOCATION),
            ("mm_signal_row", "pred_a", "signal", 2.0, _SIGNAL_ONLY),
        ],
    )

    selected = _selected(CASE_STUDY)
    assert selected == "mm_signal_row", (
        f"the holdout selection named {selected}; under an exact tie the specification "
        "the holdout replays is the one that wins"
    )


def test_a_declared_label_restriction_binds_the_holdout_selection(
    case_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One declaration, and the selection reads it.

    `LABEL_RESTRICTIONS` names the labels eligible to anchor a case study's registered
    strategy - sp500_options' four legacy diagnostic variants went through a path that
    treats a 5d forward return as a daily one, so their Sharpes are not comparable with
    anything. The restriction is patched on `strategy_analysis` here, which is where it is
    declared; when the declaration existed twice, the holdout selector read the copy that
    was not edited, so a restriction the case study declared did not reach the selection
    that spends its holdout.
    """
    monkeypatch.setitem(strategy_analysis.LABEL_RESTRICTIONS, CASE_STUDY, frozenset({"fwd_ret_5d"}))
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("bt_restricted_out", "pred_other_label", "allocation", 9.0, _WITH_ALLOCATION),
            ("bt_eligible", "pred_eligible", "allocation", 1.0, _WITH_ALLOCATION),
        ],
    )
    with sqlite3.connect(str(case_dir / "run_log" / "registry.db")) as db:
        db.execute(
            "UPDATE training_runs SET label = 'fwd_ret_10d' WHERE training_hash = ?",
            ("train_pred_other_label",),
        )

    selected = _selected(CASE_STUDY)
    assert selected == "bt_eligible", (
        f"the holdout selection named {selected}, whose label the case study declared "
        "ineligible to anchor its strategy"
    )


def test_a_prediction_the_case_study_publishes_nothing_about_is_not_selectable(
    case_dir: Path,
) -> None:
    """Membership, not exclusion.

    ``pred_experimental`` outranks everything and no population lists it, so nobody
    retired it: an exclusion set admits it and a membership set does not. That is how an
    experimental fit a case study never published reaches a ranking, and reaching the
    holdout ranking is the expensive version.
    """
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("bt_experimental", "pred_experimental", "allocation", 9.0, _WITH_ALLOCATION),
            ("bt_published", "pred_published", "allocation", 1.0, _WITH_ALLOCATION),
        ],
        published=["pred_published"],
    )

    assert _selected(CASE_STUDY) == "bt_published", (
        "a prediction the case study publishes nothing about was selectable; the pool is "
        "the published population, not everything nobody retired"
    )


# The plain allocator is the better strategy over the whole span and the worse one over the
# stretch both candidates cover: its ten best sessions are exactly the ones the conformal
# allocator sat out. So the two readings disagree, and which candidate comes back says which
# ranking ran.
_SESSIONS = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(40)]
_PLAIN_RETURNS = [0.05] * 10 + [0.02, -0.01] * 15
_CONFORMAL_RETURNS = [0.03, 0.01] * 15
_CONFORMAL_SPEC = json.dumps(
    {
        "strategy": {
            "signal": {"method": "equal_weight_top_k", "top_k": 10},
            "allocation": {
                "method": "conformal_weighted",
                "calibration_version": "walk_forward_v3",
            },
        }
    }
)


def _daily_returns(case_dir: Path, backtest_hash: str, values: list[float]) -> None:
    out = case_dir / "run_log" / "backtest" / backtest_hash
    out.mkdir(parents=True)
    pl.DataFrame(
        {"timestamp": _SESSIONS[len(_SESSIONS) - len(values) :], "returns": values}
    ).write_parquet(out / "daily_returns.parquet")


def test_the_common_support_re_ranking_orders_the_field_the_selection_reads(
    case_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conformal candidate re-orders the field, and the selection must read that order.

    A conformal allocator holds nothing until it is calibrated and books the warm-up as
    returns of exactly zero, so a whole-span Sharpe compares it against a different sample
    from every other candidate. The resolver answers that by re-ranking the field on the
    timestamps every candidate prices. The fixture is built so the two rankings disagree:
    `bt_plain` stores the higher Sharpe and loses over the thirty sessions both price, so
    a selection reading stored Sharpe returns a different candidate and this test fails.

    It changes the order and not only the winner, which is why it belongs to the field: the
    holdout retrain falls back to rank-2 when rank-1's refit degenerates, and a rank-2
    ordered by a criterion the rank-1 was not is not the runner-up of anything.
    """
    monkeypatch.setattr(uncertainty, "periods_per_year_from_setup", lambda cs: 365)
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("bt_plain", "pred_plain", "allocation", 9.0, _WITH_ALLOCATION),
            ("bt_conformal", "pred_conformal", "allocation", 1.0, _CONFORMAL_SPEC),
        ],
    )
    _daily_returns(case_dir, "bt_plain", _PLAIN_RETURNS)
    _daily_returns(case_dir, "bt_conformal", _CONFORMAL_RETURNS)

    selected = _selected(CASE_STUDY)
    assert selected == "bt_conformal", (
        f"the holdout selection named {selected}, which is the stored-Sharpe answer; the "
        "field was re-ranked on common support and the selection read the order from "
        "before that"
    )

    # And the number reported for it has to be the one the selection was made on. The
    # stored Sharpe describes a different sample and is what the whole re-ranking exists
    # to stop the field being compared by.
    assert _selected_sharpe(CASE_STUDY) != 1.0, (
        "the selection reported the stored Sharpe of the row it named, not the "
        "common-support Sharpe it was selected on"
    )


def test_a_retired_prediction_cannot_raise_the_coverage_bar_over_the_live_ones(
    case_dir: Path,
) -> None:
    """The bar is a maximum, so an ineligible row can set it and empty the field.

    Only rows whose `ic_n_days` equals the maximum for their `(split, family, label)` are
    comparable, and are kept. A retired prediction scored over a longer window sets that
    maximum, every live row for the same family and label falls below it, and the query
    returns nothing - so filtering publication in Python afterwards has nothing left to
    filter. The maximum has to be taken inside the published population, which is what
    `full_coverage_prediction_sql`'s `population_subquery` is for.
    """
    _registry(
        case_dir / "run_log" / "registry.db",
        [
            ("bt_retired", "pred_retired", "allocation", 2.0, _WITH_ALLOCATION),
            ("bt_live", "pred_live", "allocation", 1.0, _WITH_ALLOCATION),
        ],
        published=["pred_live"],
    )
    with sqlite3.connect(str(case_dir / "run_log" / "registry.db")) as db:
        # Same family and label as the live row, so it is the same coverage group, and
        # scored over more decision dates, so it is the group's maximum.
        db.execute(
            "UPDATE prediction_metrics SET ic_n_days = 300 WHERE prediction_hash = ?",
            ("pred_retired",),
        )
        db.execute(
            "UPDATE training_runs SET config_name = 'config_pred_live' WHERE training_hash = ?",
            ("train_pred_retired",),
        )

    assert _selected(CASE_STUDY) == "bt_live", (
        "a retired prediction set the coverage maximum for its group and emptied the "
        "field of live rows; the maximum has to be taken inside the published population"
    )


def test_an_empty_field_is_refused_rather_than_answered(case_dir: Path) -> None:
    """Nothing eligible is a refusal, not a rank-1.

    A case study whose validation stages have not run yet has no field to rank. The
    selection raises `NoSelectableCandidates` rather than returning a row, so a caller
    cannot carry an unselected configuration into the holdout. `NoSelectableCandidates`
    subclasses `RuntimeError`; a caller that wants to treat "not run yet" as a normal
    state catches it deliberately, which is what `tests/skip_blockers.py:158` does.

    The registry here is initialised and empty, which is a different cause from the ruin
    refusal `tests/test_common_support_trigger.py` pins.
    """
    _registry(case_dir / "run_log" / "registry.db", [])

    with pytest.raises(strategy_analysis.NoSelectableCandidates):
        strategy_analysis.resolve_canonical_rank1_lineage(CASE_STUDY)
