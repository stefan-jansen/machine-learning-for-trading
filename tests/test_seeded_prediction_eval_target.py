"""A seeded classification prediction set carries a continuous eval_actual.

The canonical registries hold one invariant that the CI fixture has to reproduce:
a prediction artifact carries ``eval_actual`` exactly when its training run's label
is declared in ``labels.classification_eval_label`` in that case study's
``setup.yaml``. Measured across all nine, of 5,290 linear artifacts, every artifact
of a declared label has the column and no other artifact does.

It matters because the registry rows in the fixture are real. A classification
prediction set in them carries the ``ic_mean_daily`` production computed, and
production only computes it against the continuous return. An artifact without the
column therefore contradicts its own registry row, and
``11_ml_pipeline/07_case_study_insights`` refuses the whole notebook over it rather
than correlate the score against the binary label, which measures class separation
rather than a ranking against realized returns.

These tests pin where the column's values may come from, which is the part that is
easy to get subtly wrong: a continuous target, never a class column.
"""

import sqlite3

import polars as pl
import pytest

from tests.fixtures import seed_results

# Two case studies, because the seeder has two paths. It rewrites every artifact of
# crypto_perps_funding, so every set there is built on the fabricated grid; elsewhere
# an artifact already on disk survives and supplies the panel the rest are seeded onto.
REWRITTEN_CS = "crypto_perps_funding"
REWRITTEN_CLS_LABEL = "fwd_dir_8h"
CASE_STUDY = "sp500_equity_option_analytics"
CLASSIFICATION_LABEL = "fwd_dir_5d"
REGRESSION_LABEL = "fwd_ret_5d"


def _registry(cs_dir, rows):
    """A minimal registry holding one training run and prediction set per row."""
    db = cs_dir / "run_log" / "registry.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE training_runs (training_hash TEXT, label TEXT, family TEXT)")
    con.execute(
        "CREATE TABLE prediction_sets (prediction_hash TEXT, training_hash TEXT, split TEXT)"
    )
    for p_hash, label in rows:
        con.execute("INSERT INTO training_runs VALUES (?, ?, 'linear')", (f"t_{p_hash}", label))
        con.execute(
            "INSERT INTO prediction_sets VALUES (?, ?, 'validation')", (p_hash, f"t_{p_hash}")
        )
    con.commit()
    con.close()


def _write_artifact(cs_dir, p_hash, frame):
    path = cs_dir / "run_log" / "predictions" / p_hash / "predictions.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def _read(cs_dir, p_hash):
    return pl.read_parquet(cs_dir / "run_log" / "predictions" / p_hash / "predictions.parquet")


def _panel(n_dates=8, n_symbols=4, *, actual, eval_actual=None):
    """A reference artifact shaped like a real one, on real-looking keys."""
    import datetime as dt

    dates = [dt.date(2023, 1, 2) + dt.timedelta(days=i) for i in range(n_dates)]
    rows = {
        "symbol": [f"S{j}" for _ in dates for j in range(n_symbols)],
        "timestamp": [d for d in dates for _ in range(n_symbols)],
        "fold": [i // 4 % 2 for i in range(n_dates) for _ in range(n_symbols)],
        "actual": actual * (n_dates * n_symbols // len(actual)),
        "prediction": [0.001 * i for i in range(n_dates * n_symbols)],
    }
    frame = pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Date))
    if eval_actual is not None:
        frame = frame.with_columns(
            pl.Series("eval_actual", eval_actual * (n_dates * n_symbols // len(eval_actual)))
        )
    return frame


@pytest.fixture
def cs_dir(tmp_path):
    return tmp_path / CASE_STUDY


@pytest.fixture
def rewritten_cs_dir(tmp_path):
    return tmp_path / REWRITTEN_CS


def test_a_fabricated_classification_set_gets_a_continuous_eval_target(rewritten_cs_dir):
    _registry(rewritten_cs_dir, [("aaaa", REWRITTEN_CLS_LABEL)])
    seed_results._backfill_all_prediction_parquets(rewritten_cs_dir, REWRITTEN_CS)

    frame = _read(rewritten_cs_dir, "aaaa")
    assert "eval_actual" in frame.columns
    # Continuous, not a class encoding. A rank correlation against a two-valued column
    # measures how well the score separates the classes, which is an AUC-like quantity
    # and not the ranking against realized returns an IC reports.
    assert frame["eval_actual"].n_unique() > 2


def test_a_reference_artifacts_own_eval_target_survives_seeding(cs_dir):
    # 'keeper' is a real artifact the seeder leaves in place, and supplies the panel
    # every other set of its (split, label) group is seeded onto.
    _registry(cs_dir, [("keeper", CLASSIFICATION_LABEL), ("seeded", CLASSIFICATION_LABEL)])
    _write_artifact(
        cs_dir,
        "keeper",
        _panel(actual=[0.0, 1.0, 1.0, 0.0], eval_actual=[-0.02, 0.03, 0.01, -0.04]),
    )
    seed_results._backfill_all_prediction_parquets(cs_dir, CASE_STUDY)

    seeded = _read(cs_dir, "seeded")
    reference = _read(cs_dir, "keeper")
    assert "eval_actual" in seeded.columns
    # Row-wise on the key, not set containment: a permuted or constant column drawn
    # from the same values would pass containment while pairing every score with the
    # wrong realized return, which is the mistake this test exists to catch.
    joined = seeded.join(
        reference.select("symbol", "timestamp", "eval_actual"),
        on=["symbol", "timestamp"],
        how="inner",
        suffix="_reference",
    )
    assert joined.height == seeded.height
    assert joined["eval_actual"].to_list() == joined["eval_actual_reference"].to_list()


def test_a_class_valued_actual_is_never_reused_as_the_eval_target(cs_dir):
    # The reference is a classification artifact with no eval_actual of its own, so
    # its `actual` is the class label. Aliasing it would put the binary target where
    # the continuous return belongs.
    _registry(cs_dir, [("keeper", CLASSIFICATION_LABEL), ("seeded", CLASSIFICATION_LABEL)])
    _write_artifact(cs_dir, "keeper", _panel(actual=[0.0, 1.0, 1.0, 0.0]))
    seed_results._backfill_all_prediction_parquets(cs_dir, CASE_STUDY)

    seeded = _read(cs_dir, "seeded")
    assert "eval_actual" in seeded.columns
    assert set(seeded["eval_actual"].to_list()) != {0.0, 1.0}
    assert seeded["eval_actual"].n_unique() > 2


def test_every_set_of_one_group_agrees_on_the_realized_return(rewritten_cs_dir):
    _registry(
        rewritten_cs_dir,
        [("one", REWRITTEN_CLS_LABEL), ("two", REWRITTEN_CLS_LABEL)],
    )
    seed_results._backfill_all_prediction_parquets(rewritten_cs_dir, REWRITTEN_CS)

    one, two = _read(rewritten_cs_dir, "one"), _read(rewritten_cs_dir, "two")
    joined = one.join(two, on=["symbol", "timestamp"], how="inner", suffix="_two")
    assert joined.height == one.height
    assert joined["eval_actual"].to_list() == joined["eval_actual_two"].to_list()


def test_a_regression_set_carries_no_eval_target(cs_dir):
    # Production never writes the column for a regression label, even when the panel
    # the set is seeded onto is a classification artifact that has one.
    _registry(cs_dir, [("keeper", REGRESSION_LABEL), ("seeded", REGRESSION_LABEL)])
    _write_artifact(
        cs_dir,
        "keeper",
        _panel(actual=[-0.01, 0.02, 0.03, -0.02], eval_actual=[-0.02, 0.03, 0.01, -0.04]),
    )
    seed_results._backfill_all_prediction_parquets(cs_dir, CASE_STUDY)

    assert "eval_actual" not in _read(cs_dir, "seeded").columns
