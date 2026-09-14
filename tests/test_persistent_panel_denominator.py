"""A model whose panel builder refuses an entity never lost it, and must not be charged.

``latent_factors/pca`` is the only config built by ``prepare_panel_data`` rather than
``prepare_ragged_panel_data``, and that builder admits an entity only if it appears on at
least half the dates of the fold's training window. Measured on ``etfs``/``fwd_ret_5d``
2026-09-14, replaying the rule against the rolling ten-year fold windows in the training
spec predicts pca's missing entity set exactly on 8 of 8 folds, and within the entities it
does carry pca covers 166,372 of 166,372 pairs on every fold. The shortfall is entirely the
entity axis and it is the estimator's declared behaviour, not a defect in the member.

The tests below are the two halves of that: the narrowing must reach pca, and it must not
become a hole. A pca member that loses a session for an entity it *does* carry still fails,
and no other config is narrowed at all.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.coverage import check_prediction_cross_section
from case_studies.utils.latent_factors.panel import (
    PERSISTENT_PANEL_MODELS,
    eligible_persistent_entities,
)
from case_studies.utils.notebook_contracts import _persistent_panel_entities

LABEL = "fwd_ret_1d"
FOLDS = [
    {"fold": 0, "val_start": dt.date(2020, 1, 6), "val_end": dt.date(2020, 1, 10)},
    {"fold": 1, "val_start": dt.date(2020, 1, 13), "val_end": dt.date(2020, 1, 17)},
]
SESSIONS = [dt.datetime(2020, 1, d, 16, 0) for d in (6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]
TRAIN = [dt.datetime(2019, 12, d, 16, 0) for d in range(2, 22)]
UNIVERSE = ("AAA", "BBB", "SPARSE")
SPEC = json.dumps(
    {
        "computation": {
            "cv": {
                "folds": [
                    {
                        "fold": f["fold"],
                        "train_start": "2019-12-02T16:00:00",
                        "train_end": "2019-12-21T16:00:00",
                        "val_start": f["val_start"].isoformat(),
                        "val_end": f["val_end"].isoformat(),
                    }
                    for f in FOLDS
                ]
            }
        }
    }
)


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    pl.DataFrame(
        [
            {"timestamp": ts, "symbol": sym, LABEL: 0.01 * i}
            for i, ts in enumerate(SESSIONS)
            for sym in UNIVERSE
        ]
    ).write_parquet(labels / f"{LABEL}.parquet")
    return tmp_path


@pytest.fixture(autouse=True)
def declared(monkeypatch):
    import case_studies.utils.cv_window as cv_window

    monkeypatch.setattr(cv_window, "fold_boundaries", lambda cs, label: list(FOLDS))
    monkeypatch.setattr(cv_window, "_holdout_window", lambda cs: None)


def _panel() -> pl.DataFrame:
    """The whole panel, train and validation, as ``feature_panel_keys`` returns it.

    ``SPARSE`` appears on 3 of the 20 training dates and on every validation session. So the
    panel offers it and the builder refuses it, which is the shape the fix is about: the
    member is offered a name its own builder will not admit, and charging it for the
    difference charges it for a denominator it cannot reach.
    """
    rows = [{"entity": sym, "session": ts} for ts in TRAIN for sym in ("AAA", "BBB")]
    rows += [{"entity": "SPARSE", "session": ts} for ts in TRAIN[:3]]
    rows += [{"entity": sym, "session": ts} for ts in SESSIONS for sym in UNIVERSE]
    return pl.DataFrame(rows)


def _predictions(symbols=UNIVERSE, sessions=SESSIONS) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"timestamp": ts, "symbol": sym, "fold_id": 0 if ts.day <= 10 else 1, "prediction": 0.5}
            for ts in sessions
            for sym in symbols
        ]
    )


def test_the_rule_admits_the_dense_entities_and_refuses_the_sparse_one():
    admitted = eligible_persistent_entities(_panel(), entity_col="entity", date_col="session")
    assert sorted(admitted.get_column("entity").to_list()) == ["AAA", "BBB"]


def test_ten_dates_is_the_floor_not_half_of_a_short_window():
    """``max(int(n * min_coverage), 10)``: on a five-date window half is 2, the floor is 10."""
    short = pl.DataFrame(
        [{"entity": "AAA", "session": ts} for ts in TRAIN[:5]]
        + [{"entity": "BBB", "session": TRAIN[0]}]
    )
    admitted = eligible_persistent_entities(short, entity_col="entity", date_col="session")
    assert admitted.is_empty()


def test_only_the_persistent_panel_models_are_narrowed():
    """The negative selftest: the exemption must not reach pca's four siblings."""
    assert set(PERSISTENT_PANEL_MODELS) == {"pca"}
    for config in ("ipca", "cae", "sae", "sdf"):
        assert (
            _persistent_panel_entities(SPEC, _panel(), family="latent_factors", config=config)
            is None
        )
    assert _persistent_panel_entities(SPEC, _panel(), family="gbm", config="pca") is None
    assert _persistent_panel_entities(SPEC, _panel(), family="latent_factors", config="pca") == {
        0: ["AAA", "BBB"],
        1: ["AAA", "BBB"],
    }


def test_a_spec_without_a_training_window_narrows_nothing():
    """Guessing a window would charge the member against a denominator nobody declared."""
    spec = json.dumps({"computation": {"cv": {"folds": [{"fold": 0, "val_start": "2020-01-06"}]}}})
    assert _persistent_panel_entities(spec, _panel(), family="latent_factors", config="pca") is None


def test_a_member_missing_only_the_refused_entity_is_whole(case_dir):
    """Charged against the full panel this reads 2/3; against what it was handed, 1.0."""
    delivered = _predictions(symbols=("AAA", "BBB"))
    admitted = _persistent_panel_entities(SPEC, _panel(), family="latent_factors", config="pca")

    unnarrowed = check_prediction_cross_section(delivered, "cs", LABEL, case_dir=case_dir)
    assert unnarrowed.coverage == pytest.approx(2 / 3)
    assert unnarrowed.never_scored == ("SPARSE",)

    report = check_prediction_cross_section(
        delivered, "cs", LABEL, case_dir=case_dir, eligible_entities=admitted
    )
    assert report.complete
    assert report.coverage == 1.0
    assert report.never_scored == ()


def test_a_session_lost_inside_an_admitted_entity_still_fails(case_dir):
    """The case that decides whether this is a narrowing or a hole."""
    delivered = _predictions(symbols=("AAA", "BBB"), sessions=SESSIONS[:-2])
    admitted = _persistent_panel_entities(SPEC, _panel(), family="latent_factors", config="pca")

    report = check_prediction_cross_section(
        delivered, "cs", LABEL, case_dir=case_dir, eligible_entities=admitted
    )
    assert not report.complete
    assert report.coverage < 1.0
    assert report.partially_scored == ("AAA", "BBB")


def test_an_entity_the_builder_admitted_and_the_member_dropped_still_fails(case_dir):
    """Narrowing to the admitted set is not narrowing to what the member happens to carry."""
    delivered = _predictions(symbols=("AAA",))
    admitted = _persistent_panel_entities(SPEC, _panel(), family="latent_factors", config="pca")

    report = check_prediction_cross_section(
        delivered, "cs", LABEL, case_dir=case_dir, eligible_entities=admitted
    )
    assert not report.complete
    assert report.never_scored == ("BBB",)


def test_a_narrowed_call_neither_reads_nor_writes_the_reachable_memo(case_dir):
    """The failure that made the first version of this fix silently do nothing.

    ``_REACHABLE_CACHE`` is keyed on the label artifact, case study, label, split, folds and
    ``id(input_panel)`` - not on ``eligible_entities``. A narrowed call that shared the memo
    would read an entry built on the full entity axis and report the member against exactly
    the denominator the narrowing had just removed, which is silent: the number is wrong and
    nothing raises. Measured on ``etfs``/``fwd_ret_5d``, that returned pca at 83.0% with the
    narrowing in place and correct.
    """
    import case_studies.utils.coverage as cov

    panel = _panel()
    delivered = _predictions(symbols=("AAA", "BBB"))
    admitted = _persistent_panel_entities(SPEC, _panel(), family="latent_factors", config="pca")
    cov._clear_cross_section_cache()

    # Unnarrowed first, so a shared key would be populated and waiting.
    wide = check_prediction_cross_section(
        delivered, "cs", LABEL, case_dir=case_dir, input_panel=panel
    )
    assert cov._REACHABLE_CACHE, "the unnarrowed call should memoize"
    before = len(cov._REACHABLE_CACHE)

    narrow = check_prediction_cross_section(
        delivered, "cs", LABEL, case_dir=case_dir, input_panel=panel, eligible_entities=admitted
    )
    assert len(cov._REACHABLE_CACHE) == before, "a narrowed call must leave no entry"
    assert narrow.accountable_coverage > wide.accountable_coverage
    assert narrow.accountable_coverage == 1.0
