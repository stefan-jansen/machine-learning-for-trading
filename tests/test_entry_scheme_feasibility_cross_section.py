"""The entry-scheme guard measures against the set the signal ranks.

ml4t/agent-workspace#1003. ``get_entry_schemes_for`` drops any declared ``top_k``
at or above the traded universe, because holding k of k is the equal-weight
benchmark rather than a ranked selection. That rule is right; it was applied to
the price panel, which is not the set the ranking runs over. When the two differ
the check passes and every backtest in the sweep records ``num_trades = 0`` -
measured in ml4t/agent-workspace#989 as twelve backtests with a Sharpe and no
trades, and four notebooks failing several stages downstream of the cause.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl
import pytest
import yaml

from case_studies.utils.sweep_config import get_entry_schemes_for, get_top_k_values_for

SETUP = {
    "backtest": {
        "sweep": {
            "top_k_grid": {"fwd_ret_5d": [3, 5, 10]},
        }
    }
}


def _case_study(tmp_path: Path, monkeypatch, *, n_prediction_symbols: int) -> str:
    """A case study whose price panel is 24 wide and whose predictions are not."""
    case_study = "fixture_cs"
    case_dir = tmp_path / case_study
    (case_dir / "config").mkdir(parents=True)
    (case_dir / "config" / "setup.yaml").write_text(yaml.safe_dump(SETUP))

    run_log = case_dir / "run_log"
    (run_log / "predictions" / "pred1").mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [1] * n_prediction_symbols,
            "symbol": [f"S{i}" for i in range(n_prediction_symbols)],
            "y_score": [0.0] * n_prediction_symbols,
        }
    ).write_parquet(run_log / "predictions" / "pred1" / "predictions.parquet")

    with sqlite3.connect(run_log / "registry.db") as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT
            );
            """
        )
        db.execute("INSERT INTO training_runs VALUES ('t1', 'gbm', 'default', 'fwd_ret_5d')")
        db.execute("INSERT INTO prediction_sets VALUES ('pred1', 't1', 'validation')")

    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    return case_study


def test_the_ranked_width_is_read_from_the_predictions(tmp_path, monkeypatch) -> None:
    from case_studies.utils.sweep_config import ranked_cross_section_width

    case_study = _case_study(tmp_path, monkeypatch, n_prediction_symbols=6)
    assert ranked_cross_section_width(case_study, "fwd_ret_5d") == 6


def test_a_concentration_wider_than_the_cross_section_is_dropped(tmp_path, monkeypatch) -> None:
    """The defect: k=10 survived because the price panel was 24 wide."""
    case_study = _case_study(tmp_path, monkeypatch, n_prediction_symbols=6)
    schemes = get_entry_schemes_for(case_study, "fwd_ret_5d", n_assets=24, long_short=False)
    assert [s["top_k"] for s in schemes] == [3, 5]


def test_the_guard_fires_when_the_cross_section_leaves_no_scheme(tmp_path, monkeypatch) -> None:
    """The failure #989 recorded: every backtest runs, none of them trades."""
    case_study = _case_study(tmp_path, monkeypatch, n_prediction_symbols=3)
    with pytest.raises(ValueError) as excinfo:
        get_entry_schemes_for(case_study, "fwd_ret_5d", n_assets=24, long_short=False)
    message = str(excinfo.value)
    # Both numbers, so the message cannot read the same wrong one the check did.
    assert "3 symbols" in message
    assert "24" in message
    assert "num_trades" in message


def test_the_price_panel_still_bounds_the_grid(tmp_path, monkeypatch) -> None:
    """The control: a narrow panel and a wide cross-section keeps the old rule."""
    case_study = _case_study(tmp_path, monkeypatch, n_prediction_symbols=100)
    schemes = get_entry_schemes_for(case_study, "fwd_ret_5d", n_assets=6, long_short=False)
    assert [s["top_k"] for s in schemes] == [3, 5]


def test_a_case_study_with_no_registered_predictions_keeps_the_panel(tmp_path, monkeypatch) -> None:
    """Nothing to read yet, so the caller's number stands and nothing is dropped."""
    case_study = "fixture_cs"
    case_dir = tmp_path / case_study
    (case_dir / "config").mkdir(parents=True)
    (case_dir / "config" / "setup.yaml").write_text(yaml.safe_dump(SETUP))
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    from case_studies.utils.sweep_config import ranked_cross_section_width

    assert ranked_cross_section_width(case_study, "fwd_ret_5d") is None
    schemes = get_entry_schemes_for(case_study, "fwd_ret_5d", n_assets=24, long_short=False)
    assert [s["top_k"] for s in schemes] == [3, 5, 10]


def test_the_plumbing_test_grid_filters_against_the_same_width(tmp_path, monkeypatch) -> None:
    """`get_top_k_values_for` is the other half of the same grid and must agree."""
    case_study = _case_study(tmp_path, monkeypatch, n_prediction_symbols=6)
    assert get_top_k_values_for(case_study, "fwd_ret_5d", 24, long_short=False) == [3, 5]


def test_both_halves_of_the_grid_report_both_numbers(tmp_path, monkeypatch) -> None:
    case_study = _case_study(tmp_path, monkeypatch, n_prediction_symbols=3)
    with pytest.raises(ValueError) as excinfo:
        get_top_k_values_for(case_study, "fwd_ret_5d", 24, long_short=False)
    assert "price panel 24" in str(excinfo.value)
    assert "ranked cross-section 3" in str(excinfo.value)


def test_disagreeing_prediction_sets_leave_the_callers_number_standing(
    tmp_path, monkeypatch
) -> None:
    """The resolver samples the registry; the caller sweeps a chosen population.

    Those are different questions. A width taken from an unrelated set could
    remove a concentration that is feasible for the sets actually being swept, so
    the resolver only speaks when its sample is unanimous.
    """
    from case_studies.utils.sweep_config import ranked_cross_section_width

    case_study = _case_study(tmp_path, monkeypatch, n_prediction_symbols=6)
    run_log = tmp_path / case_study / "run_log"
    (run_log / "predictions" / "pred2").mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [1] * 40,
            "symbol": [f"S{i}" for i in range(40)],
            "y_score": [0.0] * 40,
        }
    ).write_parquet(run_log / "predictions" / "pred2" / "predictions.parquet")
    with sqlite3.connect(run_log / "registry.db") as db:
        db.execute("INSERT INTO training_runs VALUES ('t2', 'gbm', 'other', 'fwd_ret_5d')")
        db.execute("INSERT INTO prediction_sets VALUES ('pred2', 't2', 'validation')")

    with pytest.warns(UserWarning, match="disagree on the ranked cross-section"):
        assert ranked_cross_section_width(case_study, "fwd_ret_5d") is None
    with pytest.warns(UserWarning):
        schemes = get_entry_schemes_for(case_study, "fwd_ret_5d", n_assets=24, long_short=False)
    assert [s["top_k"] for s in schemes] == [3, 5, 10]


def test_a_caller_that_knows_its_population_is_believed(tmp_path, monkeypatch) -> None:
    """`ranked_width` skips the registry sample entirely."""
    case_study = _case_study(tmp_path, monkeypatch, n_prediction_symbols=100)
    schemes = get_entry_schemes_for(
        case_study, "fwd_ret_5d", n_assets=24, long_short=False, ranked_width=6
    )
    assert [s["top_k"] for s in schemes] == [3, 5]
