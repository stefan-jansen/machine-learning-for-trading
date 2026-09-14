"""A prediction set that selection refuses cannot reach a published pair either.

``degenerate_prediction_sql`` states the rule: a LASSO or ElasticNet fit that shrinks every
coefficient to zero on a fold predicts a constant there, so that fold ranks nothing and the
pooled IC computed over it is biased rather than a model result. The rule has two limbs,
because an all-tied fold stores a NULL IC while one constant only to display precision stores
a defined, denormal one.
``selectable_validation_candidates`` applies it, so the published carrier structurally cannot
be one of these.

A chapter-20 pair is the other publication path and applied no such filter. The sweep
backtests the whole declared population rather than a shortlist, so the registry does hold
backtests standing on degenerate sets - measured on us_equities_panel on 2026-09-14, four of
them - and every ranking in ``paired_metrics`` sorts on ``sharpe`` over whatever the registry
holds.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils import paired_metrics

FIXTURE_CASE_STUDY = "fixture_case_study"
_SPEC = json.dumps({"strategy": {"signal": {"method": "equal_weight_top_k", "top_k": 5}}})
_REFIT = json.dumps({"computation": {"cv": {"split": "holdout"}}})


def _registry(case_dir: Path) -> Path:
    db_path = case_dir / "run_log" / "registry.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, config_name TEXT, family TEXT, label TEXT,
                spec_json TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT,
                checkpoint_value TEXT, checkpoint_kind TEXT
            );
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT, spec_json TEXT
            );
            CREATE TABLE backtest_metrics (
                backtest_hash TEXT PRIMARY KEY, sharpe REAL, max_drawdown REAL
            );
            CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL, ic_std REAL);
            """
        )
    return db_path


def _degenerate_fold(degenerate: bool | str) -> tuple[float | None, float]:
    """The second fold's ``(ic, ic_std)``, written the way the rule reads it.

    ``degenerate_prediction_sql`` has two limbs and testing only the first is what let this
    class of defect stay invisible once before: a fold whose predictions all tie stores a NULL
    IC, while a fold constant to display precision but not bit-identical stores a defined,
    denormal one. ``"denormal"`` writes the second so the filter is pinned against both.
    """
    if degenerate == "denormal":
        return 1e-18, 1e-18
    return (None, 0.0) if degenerate else (0.02, 0.3)


def _candidate(case_dir: Path, name: str, *, sharpe: float, degenerate: bool | str = False) -> None:
    """One validation candidate with a holdout sibling the carrier walk can find.

    Both sides carry the same family, config and label, which is what the walk's probe
    matches on. ``degenerate`` writes the fold that makes the validation set inadmissible -
    True for the NULL-IC shape and ``"denormal"`` for the defined-but-vanishing one; the folds
    are otherwise scored.
    """
    with sqlite3.connect(str(case_dir / "run_log" / "registry.db")) as db:
        for split, spec in (("validation", None), ("holdout", _REFIT)):
            training_hash = f"train_{name}_{split}"
            db.execute(
                "INSERT INTO training_runs VALUES (?, ?, 'linear', 'fwd_ret_1m', ?)",
                (training_hash, f"config_{name}", spec),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, ?, NULL, NULL)",
                (f"pred_{name}_{split}", training_hash, split),
            )
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, ?, 'signal', ?)",
                (f"bt_{name}_{split}", f"pred_{name}_{split}", _SPEC),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, -0.2)", (f"bt_{name}_{split}", sharpe)
            )
        db.executemany(
            "INSERT INTO fold_metrics VALUES (?, ?, ?)",
            [
                (f"pred_{name}_validation", 0.03, 0.4),
                (f"pred_{name}_validation", *_degenerate_fold(degenerate)),
            ],
        )


@pytest.fixture
def case_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    case_dir = tmp_path / FIXTURE_CASE_STUDY
    _registry(case_dir)
    monkeypatch.setattr(paired_metrics, "get_case_study_dir", lambda cs, **_: case_dir)
    return case_dir


class _Explorer:
    """Returns the validation rows for the stage, the way ``BacktestExplorer.best`` does."""

    def __init__(self, rows: list[tuple[str, str, float]]) -> None:
        self.rows = rows

    def best(self, *, stage: str, top_n: int, prediction_hashes=None) -> pl.DataFrame:
        if stage != "signal":
            return pl.DataFrame(schema={"backtest_hash": pl.Utf8})
        return pl.DataFrame(
            {
                "backtest_hash": [row[0] for row in self.rows],
                "prediction_hash": [row[1] for row in self.rows],
                "sharpe": [row[2] for row in self.rows],
                "family": ["linear"] * len(self.rows),
                "label": ["fwd_ret_1m"] * len(self.rows),
            }
        )


class TestTheFilterItself:
    def test_a_null_ic_fold_drops_its_prediction_set(self, case_dir: Path) -> None:
        _candidate(case_dir, "deg", sharpe=0.9, degenerate=True)
        _candidate(case_dir, "clean", sharpe=0.5)
        cand = pl.DataFrame({"prediction_hash": ["pred_deg_validation", "pred_clean_validation"]})

        kept = paired_metrics._drop_degenerate_predictions(FIXTURE_CASE_STUDY, cand)

        assert kept["prediction_hash"].to_list() == ["pred_clean_validation"]

    def test_a_fully_scored_field_is_returned_untouched(self, case_dir: Path) -> None:
        """The control: no NULL IC anywhere, so nothing is dropped and no row is reordered."""
        _candidate(case_dir, "a", sharpe=0.9)
        _candidate(case_dir, "b", sharpe=0.5)
        cand = pl.DataFrame({"prediction_hash": ["pred_a_validation", "pred_b_validation"]})

        kept = paired_metrics._drop_degenerate_predictions(FIXTURE_CASE_STUDY, cand)

        assert kept["prediction_hash"].to_list() == ["pred_a_validation", "pred_b_validation"]

    def test_a_frame_without_the_column_is_not_filtered(self, case_dir: Path) -> None:
        """A stage frame that does not carry ``prediction_hash`` cannot be judged on it."""
        _candidate(case_dir, "deg", sharpe=0.9, degenerate=True)
        cand = pl.DataFrame({"backtest_hash": ["bt_deg_validation"]})

        kept = paired_metrics._drop_degenerate_predictions(FIXTURE_CASE_STUDY, cand)

        assert kept["backtest_hash"].to_list() == ["bt_deg_validation"]

    def test_a_registry_with_no_fold_metrics_drops_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty answer must mean "nothing is degenerate", never "everything is"."""
        case_dir = tmp_path / "bare"
        (case_dir / "run_log").mkdir(parents=True)
        with sqlite3.connect(str(case_dir / "run_log" / "registry.db")) as db:
            db.execute("CREATE TABLE prediction_sets (prediction_hash TEXT)")
        monkeypatch.setattr(paired_metrics, "get_case_study_dir", lambda cs, **_: case_dir)
        cand = pl.DataFrame({"prediction_hash": ["anything"]})

        assert paired_metrics._drop_degenerate_predictions("bare", cand)[
            "prediction_hash"
        ].to_list() == ["anything"]

    def test_a_denormal_ic_is_as_degenerate_as_a_null_one(self, case_dir: Path) -> None:
        """The second limb of the rule, which reading only for NULL would miss."""
        _candidate(case_dir, "deg", sharpe=0.9, degenerate="denormal")
        _candidate(case_dir, "clean", sharpe=0.5)
        cand = pl.DataFrame({"prediction_hash": ["pred_deg_validation", "pred_clean_validation"]})

        kept = paired_metrics._drop_degenerate_predictions(FIXTURE_CASE_STUDY, cand)

        assert kept["prediction_hash"].to_list() == ["pred_clean_validation"]

    def test_an_empty_field_is_returned_as_it_came(self, case_dir: Path) -> None:
        _candidate(case_dir, "deg", sharpe=0.9, degenerate=True)
        cand = pl.DataFrame(schema={"prediction_hash": pl.Utf8})

        assert paired_metrics._drop_degenerate_predictions(FIXTURE_CASE_STUDY, cand).is_empty()


class TestTheCarrierWalk:
    """The filter has to be applied where the ranking is, not only be available."""

    def test_a_degenerate_row_does_not_become_the_carrier_by_outranking(
        self, case_dir: Path
    ) -> None:
        """The whole point: it wins on Sharpe and still must not be the pair's carrier."""
        _candidate(case_dir, "deg", sharpe=0.9, degenerate=True)
        _candidate(case_dir, "clean", sharpe=0.5)
        explorer = _Explorer(
            [
                ("bt_deg_validation", "pred_deg_validation", 0.9),
                ("bt_clean_validation", "pred_clean_validation", 0.5),
            ]
        )

        carrier = paired_metrics._val_rank1_carrier(
            FIXTURE_CASE_STUDY, explorer, label_restriction=None, rung=None
        )

        assert carrier is not None, "the clean candidate has a holdout sibling and must be found"
        assert carrier["prediction_hash"] == "pred_clean_validation"

    def test_the_same_row_is_the_carrier_when_its_folds_are_scored(self, case_dir: Path) -> None:
        """The control that makes the test above mean something.

        Same registry, same ranking, one NULL IC removed. Without this the assertion above
        would also pass if the walk simply never reached the top row.
        """
        _candidate(case_dir, "deg", sharpe=0.9)
        _candidate(case_dir, "clean", sharpe=0.5)
        explorer = _Explorer(
            [
                ("bt_deg_validation", "pred_deg_validation", 0.9),
                ("bt_clean_validation", "pred_clean_validation", 0.5),
            ]
        )

        carrier = paired_metrics._val_rank1_carrier(
            FIXTURE_CASE_STUDY, explorer, label_restriction=None, rung=None
        )

        assert carrier is not None
        assert carrier["prediction_hash"] == "pred_deg_validation"
