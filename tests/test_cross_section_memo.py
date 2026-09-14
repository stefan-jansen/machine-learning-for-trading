"""The cross-section memos must return what recomputing returns, and must not cross labels.

`undercovered_prediction_members` asks for the same declared cross-section once per member
it checks, and narrows it against the same feature panel each time. Neither answer depends
on the member. On `nasdaq100_microstructure` that was 784 members against a 9.6M-row
declaration: 5.9 s to rebuild it and 2.3 s to re-narrow it, per member, against 0.14 s to
read the member's own predictions. Caching both cut the walk from 11.0 s a member to under
4 s, which is most of the 8,608 s that function spent before a sweep could start.

A memo is only worth having if a hit is indistinguishable from a miss, and the failure it
invites is a hit on a key that does not determine the answer. So these tests compare a
cached answer against a recomputed one, and check that two labels sharing one panel object
do not answer for each other. `test_cross_section_coverage.py` builds a fresh `tmp_path` per
test, so every key there is cold and none of this is exercised by it.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils import coverage as cov
from case_studies.utils.coverage import (
    _clear_cross_section_cache,
    check_prediction_cross_section,
    declared_cross_section,
)

LABEL = "fwd_ret_1d"
OTHER = "fwd_ret_5d"
FOLDS = [
    {"fold": 0, "val_start": dt.date(2020, 1, 6), "val_end": dt.date(2020, 1, 10)},
    {"fold": 1, "val_start": dt.date(2020, 1, 13), "val_end": dt.date(2020, 1, 17)},
]
SESSIONS = [dt.datetime(2020, 1, d, 16, 0) for d in (6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]
UNIVERSE = ("AAA", "BBB", "CCC")

KEYS = ["fold", "entity", "session"]


@pytest.fixture(autouse=True)
def clean_memo():
    _clear_cross_section_cache()
    yield
    _clear_cross_section_cache()


@pytest.fixture(autouse=True)
def declared(monkeypatch):
    import case_studies.utils.cv_window as cv_window

    monkeypatch.setattr(cv_window, "fold_boundaries", lambda cs, label: list(FOLDS))
    monkeypatch.setattr(cv_window, "_holdout_window", lambda cs: None)


def _write_label(case_dir: Path, label: str, universe=UNIVERSE) -> None:
    labels = case_dir / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {"timestamp": ts, "symbol": sym, label: 0.01 * i}
            for i, ts in enumerate(SESSIONS)
            for sym in universe
        ]
    ).write_parquet(labels / f"{label}.parquet")


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    _write_label(tmp_path, LABEL)
    return tmp_path


def _sorted(frame: pl.DataFrame) -> pl.DataFrame:
    # The declaration is built by a `unique()` over a lazy scan, whose row order is not
    # stable between calls. Two answers are the same answer when they hold the same rows,
    # so comparing frames directly would fail on an ordering that never meant anything.
    return frame.sort(KEYS)


def test_a_cached_cross_section_holds_what_recomputing_holds(case_dir):
    first = declared_cross_section("cs", LABEL, case_dir=case_dir)
    cached = declared_cross_section("cs", LABEL, case_dir=case_dir)
    assert cached is first, "second call did not hit the memo"

    _clear_cross_section_cache()
    recomputed = declared_cross_section("cs", LABEL, case_dir=case_dir)
    assert recomputed is not first
    assert _sorted(cached).equals(_sorted(recomputed))


def test_two_labels_on_one_case_dir_do_not_answer_for_each_other(case_dir):
    # BBB carries no `fwd_ret_5d`, so the two declarations differ in their entity set. A key
    # that dropped the label would hand the first answer back for the second and report the
    # missing symbol as covered.
    _write_label(case_dir, OTHER, universe=("AAA", "CCC"))
    first = declared_cross_section("cs", LABEL, case_dir=case_dir)
    other = declared_cross_section("cs", OTHER, case_dir=case_dir)
    assert set(first["entity"].unique()) == set(UNIVERSE)
    assert set(other["entity"].unique()) == {"AAA", "CCC"}


def test_a_rewritten_label_artifact_is_picked_up_after_a_clear(case_dir):
    before = declared_cross_section("cs", LABEL, case_dir=case_dir)
    assert set(before["entity"].unique()) == set(UNIVERSE)
    _write_label(case_dir, LABEL, universe=("AAA",))
    _clear_cross_section_cache()
    after = declared_cross_section("cs", LABEL, case_dir=case_dir)
    assert set(after["entity"].unique()) == {"AAA"}


def _predictions(symbols=UNIVERSE) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"timestamp": ts, "symbol": sym, "fold_id": 0 if ts.day <= 10 else 1, "prediction": 0.5}
            for ts in SESSIONS
            for sym in symbols
        ]
    )


def _panel(symbols=UNIVERSE) -> pl.DataFrame:
    return pl.DataFrame([{"entity": sym, "session": ts} for ts in SESSIONS for sym in symbols])


def test_the_panel_memo_reports_what_an_uncached_call_reports(case_dir):
    # The narrowed cross-section is keyed on the panel object, so a second member checked
    # against the same panel must get the same `achievable` as the first - and the same as a
    # cold call. Two members differing only in what they delivered is the shape the walk has.
    panel = _panel()
    whole = check_prediction_cross_section(
        _predictions(), "cs", LABEL, case_dir=case_dir, input_panel=panel
    )
    partial = check_prediction_cross_section(
        _predictions(symbols=("AAA", "BBB")), "cs", LABEL, case_dir=case_dir, input_panel=panel
    )
    _clear_cross_section_cache()
    cold = check_prediction_cross_section(
        _predictions(symbols=("AAA", "BBB")), "cs", LABEL, case_dir=case_dir, input_panel=panel
    )
    assert partial.accountable_coverage == cold.accountable_coverage
    # And the memo did not make the second member read as the first: two thirds of the
    # universe delivered against a panel offering all three.
    assert whole.accountable_coverage == pytest.approx(1.0)
    assert partial.accountable_coverage == pytest.approx(2 / 3)


def test_a_different_panel_is_not_served_the_first_panels_answer(case_dir):
    # Same case study, same label, same folds, different panel. A wider panel makes more of
    # the declaration reachable, so `achievable` moves and an entry reused across the two
    # would show up here.
    narrow = _panel(symbols=("AAA",))
    wide = _panel()
    on_narrow = check_prediction_cross_section(
        _predictions(), "cs", LABEL, case_dir=case_dir, input_panel=narrow
    )
    on_wide = check_prediction_cross_section(
        _predictions(), "cs", LABEL, case_dir=case_dir, input_panel=wide
    )
    assert on_narrow.achievable == len(SESSIONS)
    assert on_wide.achievable == len(SESSIONS) * len(UNIVERSE)


def test_an_entry_built_from_another_panel_is_not_used(case_dir):
    """The guard, tested apart from the key that usually makes it unnecessary.

    Two things stop one panel's keys being handed to another: `id(input_panel)` is the key,
    and there is an `is` check on the stored frame. Either alone is enough, so removing either
    one leaves the two tests above green and the pair is not coverage of either. The `is`
    check is the one that has to hold on its own, because an id is unique only among LIVE
    objects: a panel that is collected frees its id for the next allocation, and the key
    would then match a panel the entry was never built from.

    That collision cannot be provoked on demand, so the entry is planted instead - under the
    key this call computes, holding a different panel and a deliberately wrong key set.
    Serving it would report one symbol reachable where three are.
    """
    wide = _panel()
    narrow = _panel(symbols=("AAA",))
    first = check_prediction_cross_section(
        _predictions(), "cs", LABEL, case_dir=case_dir, input_panel=wide
    )
    assert first.achievable == len(SESSIONS) * len(UNIVERSE)

    (key,) = list(cov._PANEL_KEYS_CACHE)
    wrong = cov._PANEL_KEYS_CACHE[key][1].filter(pl.col("entity") == "AAA")
    cov._PANEL_KEYS_CACHE[key] = (narrow, wrong)

    again = check_prediction_cross_section(
        _predictions(), "cs", LABEL, case_dir=case_dir, input_panel=wide
    )
    assert again.achievable == first.achievable, (
        "the panel keys were taken from an entry built against a different panel"
    )


def test_the_memo_is_not_load_bearing_for_correctness(case_dir, monkeypatch):
    # The negative control: with both memos disabled the same call must give the same
    # answer. If it does not, the cache is covering a difference rather than a repetition.
    panel = _panel()
    cached = check_prediction_cross_section(
        _predictions(symbols=("AAA", "BBB")), "cs", LABEL, case_dir=case_dir, input_panel=panel
    )

    class _NeverStores(dict):
        def __setitem__(self, key, value):  # noqa: D105
            return None

    monkeypatch.setattr(cov, "_CROSS_SECTION_CACHE", _NeverStores())
    monkeypatch.setattr(cov, "_PANEL_KEYS_CACHE", _NeverStores())
    uncached = check_prediction_cross_section(
        _predictions(symbols=("AAA", "BBB")), "cs", LABEL, case_dir=case_dir, input_panel=panel
    )
    assert cached.accountable_coverage == uncached.accountable_coverage
    assert cached.achievable == uncached.achievable
    assert cached.per_fold == uncached.per_fold


def test_one_panel_across_two_decision_axes_does_not_reuse_the_first(case_dir):
    """A narrowed decision axis narrows `want`, so the narrowing built on one is wrong for
    the other. Bypassing only the declaration memo does not cover this: the reachable memo
    sits after the narrowing and would answer from the entry the full-axis call left.

    The first call restricts the axis to fold 0's sessions, and the second asks about the
    full axis with predictions that cover only those sessions. A reused entry reports the
    unscored half as unreachable, which reads as complete coverage.
    """
    first_week = [ts for ts in SESSIONS if ts.day <= 10]
    panel = _panel()
    restricted = check_prediction_cross_section(
        _predictions(),
        "cs",
        LABEL,
        case_dir=case_dir,
        input_panel=panel,
        decision_axis=pl.Series("session", first_week),
    )
    assert restricted.achievable == len(first_week) * len(UNIVERSE)

    full = check_prediction_cross_section(
        _predictions(),
        "cs",
        LABEL,
        case_dir=case_dir,
        input_panel=panel,
    )
    assert full.achievable == len(SESSIONS) * len(UNIVERSE), (
        "the full axis was served the restricted axis's narrowing"
    )


def test_two_registries_sharing_a_case_study_and_label_do_not_share_an_entry(tmp_path):
    """`case_study` does not identify a registry. A preview writes its own directory and a
    reader's clone is another, and both answer to the same case study and label while
    holding different label artifacts. The key carries the artifact path for that reason.
    """
    wide_dir = tmp_path / "wide"
    narrow_dir = tmp_path / "narrow"
    _write_label(wide_dir, LABEL, universe=UNIVERSE)
    _write_label(narrow_dir, LABEL, universe=("AAA",))
    panel = _panel()

    on_wide = check_prediction_cross_section(
        _predictions(), "cs", LABEL, case_dir=wide_dir, input_panel=panel
    )
    on_narrow = check_prediction_cross_section(
        _predictions(), "cs", LABEL, case_dir=narrow_dir, input_panel=panel
    )
    assert on_wide.achievable == len(SESSIONS) * len(UNIVERSE)
    assert on_narrow.achievable == len(SESSIONS), (
        "the second registry was served the first registry's declaration"
    )
