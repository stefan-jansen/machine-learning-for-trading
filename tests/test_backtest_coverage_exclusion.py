"""A prediction set that covers a fraction of the cross-section is dropped from the pool.

Selection already refuses one class of unrankable result - `degenerate_prediction_sql`
drops a set whose fold shrank every coefficient to zero, because a pooled IC over the
surviving folds is not a model result. A set that scored two thirds of the symbols is the
same kind of thing: it competes against families scored on the whole universe, and the
leaderboard then compares two experiments.

The measurement is charged against what the feature panels offered, so a family that
delivered everything it was handed is never refused for a shortfall it inherited. That
distinction is what the two fixtures below differ in.

These tests used to run against `load_backtest_predictions`, which applied the same rule at
load time and had no caller anywhere in the repository, so they proved a property of code
nothing ran. They now exercise `undercovered_prediction_members`, which
`prediction_members_in_force` calls to build the candidate pool.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.notebook_contracts import undercovered_prediction_members

LABEL = "fwd_ret_1d"
FOLDS = [{"fold": 0, "val_start": dt.date(2020, 1, 6), "val_end": dt.date(2020, 1, 17)}]
SESSIONS = [dt.datetime(2020, 1, d, 16, 0) for d in (6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]
UNIVERSE = ("AAA", "BBB", "CCC")


def _write_predictions(case_dir: Path, phash: str, symbols) -> None:
    out = case_dir / "run_log" / "predictions" / phash
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {"timestamp": ts, "symbol": sym, "fold_id": 0, "y_score": 0.5, "y_true": 0.01}
            for ts in SESSIONS
            for sym in symbols
        ]
    ).write_parquet(out / "predictions.parquet")


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    pl.DataFrame(
        [{"timestamp": ts, "symbol": sym, LABEL: 0.01} for ts in SESSIONS for sym in UNIVERSE]
    ).write_parquet(labels / f"{LABEL}.parquet")

    db_path = tmp_path / "run_log" / "registry.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.executescript(
        """
        CREATE TABLE training_runs (training_hash TEXT, family TEXT, label TEXT,
                                    config_name TEXT, created_at TEXT, spec_json TEXT);
        CREATE TABLE prediction_sets (prediction_hash TEXT, training_hash TEXT, split TEXT);
        CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL, ic_std REAL);
        """
    )
    for phash, family, symbols in (
        ("whole", "gbm", UNIVERSE),
        ("narrow", "deep_learning", ("AAA", "BBB")),
    ):
        db.execute(
            "INSERT INTO training_runs VALUES (?,?,?,?,?,?)",
            (f"t-{phash}", family, LABEL, "default", "2026-01-01", None),
        )
        db.execute(
            "INSERT INTO prediction_sets VALUES (?,?,?)", (phash, f"t-{phash}", "validation")
        )
        _write_predictions(tmp_path, phash, symbols)
    db.commit()
    db.close()
    return tmp_path


@pytest.fixture
def wide_panel(case_dir: Path) -> Path:
    """Features for the whole universe, so a narrow result lost rows it was handed."""
    features = case_dir / "features"
    features.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [{"timestamp": ts, "symbol": sym, "x": 1.0} for ts in SESSIONS for sym in UNIVERSE]
    ).write_parquet(features / "financial.parquet")
    return case_dir


@pytest.fixture(autouse=True)
def declared(monkeypatch):
    import case_studies.utils.cv_window as cv_window

    monkeypatch.setattr(cv_window, "fold_boundaries", lambda cs, label: list(FOLDS))
    monkeypatch.setattr(cv_window, "_holdout_window", lambda cs: None)


def _short(case_dir: Path, minimum: float | None = 0.98) -> dict[str, str]:
    return undercovered_prediction_members(
        case_dir, ["whole", "narrow"], case_study="cs", minimum=minimum
    )


def test_the_narrow_set_is_reported_and_the_whole_one_is_not(wide_panel):
    assert sorted(_short(wide_panel)) == ["narrow"]


def test_the_reason_names_the_entity_that_was_never_scored(wide_panel):
    reason = _short(wide_panel)["narrow"]
    assert "CCC" in reason
    assert "never scored" in reason


def test_a_shortfall_the_feature_panel_caused_withholds_nothing(case_dir):
    """Same two prediction sets, but the panel only ever offered AAA and BBB.

    `deep_learning` is then complete on what it was handed and `gbm`'s third symbol is
    surplus, so a gate charged against the label would refuse the wrong result.
    """
    features = case_dir / "features"
    features.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [{"timestamp": ts, "symbol": sym, "x": 1.0} for ts in SESSIONS for sym in ("AAA", "BBB")]
    ).write_parquet(features / "financial.parquet")

    assert _short(case_dir) == {}


def test_a_set_whose_coverage_cannot_be_evaluated_is_withheld_not_admitted(wide_panel):
    """A check that cannot run must not read as a pass - the module's own rule."""
    bad = wide_panel / "run_log" / "predictions" / "narrow" / "predictions.parquet"
    pl.read_parquet(bad).drop("y_score").write_parquet(bad)

    short = _short(wide_panel)
    assert sorted(short) == ["narrow"]
    assert "could not be evaluated" in short["narrow"]


def test_a_key_only_the_financial_panel_offers_is_still_achievable(case_dir):
    """The denominator is the financial panel, because that is the side the join keeps.

    `load_modeling_dataset` LEFT-joins the model-based panel onto financial
    (`utils/modeling.py:969` and `:987`) and only then inner-joins labels, so a key present
    in `financial` and absent from `model_based` survives into the design matrix carrying
    nulls in the model-based columns. Every family has a missing-value policy for those -
    gbm passes them through, linear medians, the sequence families mean-fill - so the row is
    handed to the model rather than dropped, and a family that never scores it lost a key it
    was given.

    Intersecting the panels instead would make that key unachievable and report the family
    whole. `CCC` here is exactly that key: `deep_learning` never scores it and reads 66.7%.
    """
    features = case_dir / "features"
    features.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [{"timestamp": ts, "symbol": sym, "x": 1.0} for ts in SESSIONS for sym in UNIVERSE]
    ).write_parquet(features / "financial.parquet")
    pl.DataFrame(
        [{"timestamp": ts, "symbol": sym, "z": 1.0} for ts in SESSIONS for sym in ("AAA", "BBB")]
    ).write_parquet(features / "model_based.parquet")

    short = _short(case_dir)
    assert sorted(short) == ["narrow"]
    assert "CCC" in short["narrow"]


def test_a_model_based_panel_narrower_than_financial_does_not_shrink_the_denominator(case_dir):
    """The same fixture read as a measurement rather than as an exclusion.

    Pinning the count separately from the exclusion is what makes the previous test say
    which rule produced it: an intersected denominator is 2 entities and a financial one is
    3, and only the second charges `deep_learning` for `CCC`.
    """
    from case_studies.utils.coverage import feature_panel_keys

    features = case_dir / "features"
    features.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [{"timestamp": ts, "symbol": sym, "x": 1.0} for ts in SESSIONS for sym in UNIVERSE]
    ).write_parquet(features / "financial.parquet")
    pl.DataFrame(
        [{"timestamp": ts, "symbol": sym, "z": 1.0} for ts in SESSIONS for sym in ("AAA", "BBB")]
    ).write_parquet(features / "model_based.parquet")

    panel = feature_panel_keys(case_dir)
    assert panel is not None
    assert sorted(panel.get_column("entity").unique().to_list()) == sorted(UNIVERSE)
