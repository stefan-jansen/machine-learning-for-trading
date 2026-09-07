"""An allocator's name and the Sharpe printed beside it come from one row.

`BacktestExplorer.compare_allocators` ranks allocators by their MEAN Sharpe across the
configurations each was run on. A synthesis page that took the name from that ranking and
the number from the single highest-Sharpe allocation row was pairing two different answers:
the sweeps vary concentration per prediction and per allocator, so the allocator with the
best average and the allocator on the best individual row are routinely different.

Measured on the production registries 2026-09-07, on each case study's leading
allocation-stage row: `etfs` printed `risk_parity` beside a Sharpe of 0.8769 that `hrp`
produced, `sp500_equity_option_analytics` printed `mvo_ledoit_wolf` beside 2.0408, also
`hrp`. Four of six agreed, which is why it read as correct.

`allocation_method_of` reads the allocator off the row itself, so there is nothing to
disagree with.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.strategy_analysis import allocation_method_of

CASE_STUDY = "fixture_case_study"

# `hrp` books the single best row and a poor second; `risk_parity` books two middling ones
# and the better average. This is the production shape rather than a contrived one - it is
# what a concentration sweep does when one allocator is sensitive to top_k and another is not.
_ROWS = [
    ("bt_hrp_best", "hrp", 2.10),
    ("bt_hrp_poor", "hrp", 0.10),
    ("bt_rp_one", "risk_parity", 1.30),
    ("bt_rp_two", "risk_parity", 1.20),
    ("bt_baseline", None, 0.50),
]


@pytest.fixture
def case_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    case_dir = tmp_path / CASE_STUDY
    (case_dir / "run_log").mkdir(parents=True)
    monkeypatch.setattr(
        "utils.paths.get_case_study_dir", lambda case_study, **_: tmp_path / case_study
    )
    with sqlite3.connect(str(case_dir / "run_log" / "registry.db")) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT,
                checkpoint_value REAL
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
            INSERT INTO training_runs VALUES ('train', 'gbm', 'cfg', 'fwd_ret_5d');
            INSERT INTO prediction_sets VALUES ('pred', 'train', 'validation', 0);
            INSERT INTO prediction_metrics VALUES ('pred', 0.02, 0.02, 0.0, 0.04, 250);
            INSERT INTO fold_metrics VALUES ('pred', 0.02);
            """
        )
        for backtest_hash, allocator, sharpe in _ROWS:
            strategy: dict = {"signal": {"method": "equal_weight_top_k", "top_k": 10}}
            if allocator is not None:
                strategy["allocation"] = {"method": allocator}
            # The canonical shape `is_backtest_spec` recognises - version 2 with both
            # `strategy` and `backtest_config` - because `strategy_view` reads the whole
            # spec as the strategy block for anything else, and both readers go through it.
            db.execute(
                "INSERT INTO backtest_runs VALUES (?, 'pred', ?, 'allocation')",
                (
                    backtest_hash,
                    json.dumps({"version": 2, "strategy": strategy, "backtest_config": {}}),
                ),
            )
            db.execute(
                "INSERT INTO backtest_metrics VALUES (?, ?, 0.1, -0.2, 0.3, 0.1, 100)",
                (backtest_hash, sharpe),
            )
    return case_dir


def test_the_allocator_reported_is_the_one_on_the_highest_sharpe_row(case_dir: Path) -> None:
    """The pair the page prints, and the two answers that used to supply it.

    `compare_allocators` puts `risk_parity` first because its two rows average better than
    `hrp`'s two. The Sharpe the page prints comes from `bt_hrp_best`. Both statements are
    true and neither is a defect; pairing them is.
    """
    explorer = BacktestExplorer(CASE_STUDY, case_dir=case_dir)
    leading = explorer.best(stage="allocation", top_n=10).head(1)
    assert leading["backtest_hash"][0] == "bt_hrp_best"

    mean_ranked = explorer.compare_allocators(stages=("allocation",))["allocator"][0]
    assert mean_ranked == "risk_parity", (
        "the fixture is meant to make the two rankings disagree; if compare_allocators now "
        f"answers {mean_ranked!r} the case no longer exercises the defect"
    )

    assert allocation_method_of(CASE_STUDY, leading["backtest_hash"][0]) == "hrp"


def test_an_allocation_row_with_no_allocation_block_is_named_equal_weight(
    case_dir: Path,
) -> None:
    """Equal weight is an allocation decision, so it is named rather than left blank.

    ``reference/CASE_STUDY_PIPELINE.md`` section 12: the baseline "acts as if we didn't
    allocate" when it did. A blank here would read as missing data.
    """
    assert allocation_method_of(CASE_STUDY, "bt_baseline") == "equal_weight"


@pytest.mark.parametrize("backtest_hash", [None, "", "a_hash_the_registry_has_never_seen"])
def test_an_absent_row_answers_with_nothing_rather_than_raising(
    case_dir: Path, backtest_hash: str | None
) -> None:
    """A case study with no allocation stage has no allocator, which is not an error."""
    assert allocation_method_of(CASE_STUDY, backtest_hash) == ""
