"""Which candidates the rank-1 resolver aligns, and which it is allowed to drop.

A conformal allocator cannot size anything until it has residuals, so it holds nothing
over its warm-up and books that stretch as returns of exactly zero. It is then ranked
against allocators measured over the full span, and the comparison is between two
different samples. `resolve_canonical_rank1_lineage` answers that by re-ranking every
candidate on their exact timestamp intersection.

The trigger for that used to read the calibration version, and decided two things with it.
These tests pin both halves: the alignment has to fire for a field whose only conformal
candidate is the current version, and a mixed field must not lose its current-version
candidates to the older one's presence.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

import polars as pl
import pytest

import case_studies.utils.uncertainty as uncertainty
import utils.paths as paths
from case_studies.utils.strategy_analysis import resolve_canonical_rank1_lineage

CASE_STUDY = "fixture_conformal"
SESSIONS = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(40)]

# The plain allocator is the better strategy over the whole span and the worse one over the
# stretch both candidates cover: its ten best sessions are exactly the ones the conformal
# allocator sat out. So the two readings disagree, and which candidate comes back says which
# ranking ran. Over the shared thirty, plain averages 0.005 with a 0.015 spread and conformal
# 0.02 with a 0.01 spread.
PLAIN_RETURNS = [0.05] * 10 + [0.02, -0.01] * 15
CONFORMAL_RETURNS = [0.03, 0.01] * 15


def _spec(method: str | None, version: str | None = None) -> str:
    if method is None:
        return json.dumps({"strategy": {"signal": {"method": "equal_weight_top_k"}}})
    allocation: dict = {"method": method}
    if version is not None:
        allocation["calibration_version"] = version
    return json.dumps(
        {
            "strategy": {"signal": {"method": "equal_weight_top_k"}},
            "allocation": allocation,
            "strategy_allocation": None,
        }
        | {"strategy": {"signal": {"method": "equal_weight_top_k"}, "allocation": allocation}}
    )


def _returns(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": SESSIONS[len(SESSIONS) - len(values) :],
            "returns": values,
        }
    )


@pytest.fixture
def case_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / CASE_STUDY
    (root / "run_log").mkdir(parents=True)
    monkeypatch.setattr(paths, "get_case_study_dir", lambda cs, **kw: root)
    monkeypatch.setattr(uncertainty, "periods_per_year_from_setup", lambda cs: 365)
    with sqlite3.connect(root / "run_log" / "registry.db") as db:
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
            CREATE TABLE backtest_runs (
                backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT, spec_json TEXT
            );
            CREATE TABLE backtest_metrics (backtest_hash TEXT PRIMARY KEY, sharpe REAL);
            CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL);
            -- The spec is what separates a refit from a validation-fitted model scored on
            -- a later window, so the holdout lineage resolver reads it. This fixture has no
            -- holdout at all: 'train' declares the validation CV, and its presence is what
            -- makes the resolver's answer here "no holdout" rather than a missing column.
            INSERT INTO training_runs VALUES (
                'train', 'gbm', 'cfg', 'fwd_ret_1d',
                '{"computation": {"cv": {"split": "validation"}}}'
            );
            INSERT INTO prediction_sets VALUES ('pred', 'train', 'validation', NULL, NULL);
            INSERT INTO fold_metrics VALUES ('pred', 0.02);
            """
        )
    return root


def _add(root: Path, backtest_hash: str, spec: str, sharpe: float, values: list[float]) -> None:
    with sqlite3.connect(root / "run_log" / "registry.db") as db:
        db.execute(
            "INSERT INTO backtest_runs VALUES (?, 'pred', 'allocation', ?)",
            (backtest_hash, spec),
        )
        db.execute("INSERT INTO backtest_metrics VALUES (?, ?)", (backtest_hash, sharpe))
    out = root / "run_log" / "backtest" / backtest_hash
    out.mkdir(parents=True)
    _returns(values).write_parquet(out / "daily_returns.parquet")


def test_a_current_version_conformal_field_is_still_aligned(case_dir: Path) -> None:
    """No v2 candidate anywhere, and the alignment still has to fire.

    v3 shortens the abstention to a warm-up rather than removing it, so its return series
    is still shorter than the field's and the whole-span Sharpe still compares two samples.
    """
    _add(case_dir, "bt_plain", _spec(None), sharpe=9.0, values=PLAIN_RETURNS)
    _add(
        case_dir,
        "bt_conformal_v3",
        _spec("conformal_weighted", "walk_forward_v3"),
        sharpe=1.0,
        values=CONFORMAL_RETURNS,
    )

    lineage = resolve_canonical_rank1_lineage(CASE_STUDY)

    # Ranked on the 30 sessions both cover, the conformal result wins; ranked on the stored
    # whole-span Sharpe, the plain one does. The alignment is what makes it the former.
    assert lineage["val_backtest_hash"] == "bt_conformal_v3"
    assert lineage["comparison_n_periods"] == 30


def test_a_current_version_candidate_survives_an_older_one_in_the_same_field(
    case_dir: Path,
) -> None:
    """The older version's presence must not remove the newer version's candidates."""
    _add(case_dir, "bt_plain", _spec(None), sharpe=0.5, values=PLAIN_RETURNS)
    _add(
        case_dir,
        "bt_conformal_v2",
        _spec("conformal_weighted", "walk_forward_v2"),
        sharpe=0.6,
        values=CONFORMAL_RETURNS,
    )
    _add(
        case_dir,
        "bt_conformal_v3",
        _spec("conformal_weighted", "walk_forward_v3"),
        sharpe=9.0,
        values=[v * 2 for v in CONFORMAL_RETURNS],
    )

    lineage = resolve_canonical_rank1_lineage(CASE_STUDY)

    assert lineage["val_backtest_hash"] == "bt_conformal_v3"
