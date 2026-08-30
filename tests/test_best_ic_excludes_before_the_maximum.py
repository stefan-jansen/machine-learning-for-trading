"""An excluded prediction must not set the coverage bar, or delete its family.

`load_model_ic` keeps only the rows whose `ic_n_days` equals the maximum for their
`(split, family, label)`, and `load_best_ic_per_family` then reduces each family to its
highest-IC row. A caller that has to drop rows - retirement is the usual reason, because the
metrics catalog carries no lineage - cannot do it at either point afterwards:

* Dropping after the coverage bar leaves the excluded row setting the maximum. If the retired
  generation happens to hold the highest coverage, every live row of that family is already
  below the bar and the family is empty before the exclusion is even applied.
* Dropping after the per-family reduction has already discarded every runner-up, so excluding
  a family's leader removes the family rather than falling back to its best remaining run.

Both spellings look correct while nothing is excluded, and both fail in the same direction: a
caller that looks its families up by name raises `KeyError` on a family that is simply absent.
These build a registry on disk rather than patching the loader, because the first of the two is
a property of the SQL and a patched loader cannot exercise it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.utils import analytics

_SCHEMA = """
CREATE TABLE training_runs (
    training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT
);
CREATE TABLE prediction_sets (
    prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT, checkpoint_value INTEGER
);
CREATE TABLE prediction_metrics (
    prediction_hash TEXT PRIMARY KEY, ic_mean REAL, ic_mean_daily REAL, ic_std REAL,
    ic_n_days INTEGER
);
CREATE TABLE fold_metrics (prediction_hash TEXT, fold INTEGER, ic REAL);
"""

# (prediction_hash, family, config_name, ic, ic_n_days)
_ROWS = [
    ("lin_hi", "linear", "ridge_hi", 0.05, 1000),
    ("lin_lo", "linear", "ridge_lo", 0.04, 1000),
    ("gbm_hi", "gbm", "leaves_hi", 0.08, 1000),
    ("gbm_lo", "gbm", "leaves_lo", 0.06, 1000),
]


def _write(db_path: Path, rows: list[tuple]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        db.executescript(_SCHEMA)
        for phash, family, config, ic, n_days in rows:
            db.execute(
                "INSERT INTO training_runs VALUES (?, ?, ?, ?)",
                (f"t_{phash}", family, config, "fwd_ret_21d"),
            )
            db.execute(
                "INSERT INTO prediction_sets VALUES (?, ?, 'validation', NULL)",
                (phash, f"t_{phash}"),
            )
            db.execute(
                "INSERT INTO prediction_metrics VALUES (?, ?, ?, 0.1, ?)",
                (phash, ic, ic, n_days),
            )
            db.execute("INSERT INTO fold_metrics VALUES (?, 0, ?)", (phash, ic))


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build an etfs registry from ``rows`` and point the loaders at it."""

    def build(rows: list[tuple] = _ROWS) -> None:
        _write(tmp_path / "etfs" / "run_log" / "registry.db", rows)
        monkeypatch.setattr(
            analytics, "registry_path", lambda cs: tmp_path / cs / "run_log" / "registry.db"
        )
        monkeypatch.setitem(analytics.PRIMARY_LABELS, "etfs", "fwd_ret_21d")

    return build


def _leaders(**kwargs) -> dict[str, str]:
    result = analytics.load_best_ic_per_family(["linear", "gbm"], case_studies=["etfs"], **kwargs)
    if result.is_empty():
        return {}
    return dict(zip(result["family"], result["config_name"], strict=True))


def test_without_an_exclusion_each_family_reports_its_leader(registry) -> None:
    registry()
    assert _leaders() == {"linear": "ridge_hi", "gbm": "leaves_hi"}


def test_excluding_a_leader_falls_back_to_the_family_runner_up(registry) -> None:
    registry()
    # Filtering the returned frame instead would have left neither family standing.
    assert _leaders(exclude_prediction_hashes=["lin_hi", "gbm_hi"]) == {
        "linear": "ridge_lo",
        "gbm": "leaves_lo",
    }


def test_an_excluded_row_does_not_set_the_coverage_bar(registry) -> None:
    """The case a post-query filter cannot reach.

    `lin_hi` is scored over more days than the rest of its family, so it alone clears the
    coverage bar. Excluding it has to lower the bar to what the surviving linear rows reach;
    an exclusion applied to the query's result would instead return nothing for `linear`,
    because the bar had already removed `lin_lo`.
    """
    registry(
        [
            ("lin_hi", "linear", "ridge_hi", 0.05, 1200),
            ("lin_lo", "linear", "ridge_lo", 0.04, 1000),
            ("gbm_hi", "gbm", "leaves_hi", 0.08, 1000),
        ]
    )
    assert _leaders() == {"linear": "ridge_hi", "gbm": "leaves_hi"}
    assert _leaders(exclude_prediction_hashes=["lin_hi"]) == {
        "linear": "ridge_lo",
        "gbm": "leaves_hi",
    }


def test_the_bar_is_recomputed_per_family_not_globally(registry) -> None:
    """Excluding one family's high-coverage row must not move another family's bar."""
    registry(
        [
            ("lin_hi", "linear", "ridge_hi", 0.05, 1200),
            ("lin_lo", "linear", "ridge_lo", 0.04, 1000),
            ("gbm_hi", "gbm", "leaves_hi", 0.08, 1000),
            ("gbm_lo", "gbm", "leaves_lo", 0.06, 900),
        ]
    )
    result = _leaders(exclude_prediction_hashes=["lin_hi"])
    # gbm_lo is still below gbm's own bar of 1000 and must stay excluded.
    assert result == {"linear": "ridge_lo", "gbm": "leaves_hi"}


def test_excluding_every_row_of_one_family_drops_only_that_family(registry) -> None:
    registry()
    assert _leaders(exclude_prediction_hashes=["lin_hi", "lin_lo"]) == {"gbm": "leaves_hi"}


def test_excluding_everything_returns_an_empty_frame(registry) -> None:
    registry()
    assert _leaders(exclude_prediction_hashes=["lin_hi", "lin_lo", "gbm_hi", "gbm_lo"]) == {}


def test_an_empty_exclusion_is_not_read_as_no_filter(registry) -> None:
    # `[]` is falsy, so an implementation branching on truthiness rather than on `is None`
    # would skip the filter here for the right answer by the wrong route. Pinned so that a
    # later change to a non-empty default cannot pass silently.
    registry()
    assert _leaders(exclude_prediction_hashes=[]) == {"linear": "ridge_hi", "gbm": "leaves_hi"}
