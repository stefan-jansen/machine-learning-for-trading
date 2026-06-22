from pathlib import Path

import pytest

from tests.pm_helpers import (
    RECORD_REPLAY,
    RECORD_REWRITE,
    TIER_ON_DEMAND,
    TIER_PER_COMMIT,
    TIER_WEEKLY,
    collect_chapter_notebooks,
    current_test_tier,
    get_record_mode,
    get_reruns,
    get_tier,
)


def test_collect_chapter_notebooks_keeps_real_notebooks_and_skips_helpers() -> None:
    notebooks = {path.as_posix() for path in collect_chapter_notebooks(Path("."), range(1, 28))}

    assert "06_strategy_definition/02_case_study_overview.py" in notebooks
    assert "08_financial_features/case_study_feature_summary.py" in notebooks
    assert "11_ml_pipeline/08_ml_backtest_intro.py" in notebooks
    assert "16_strategy_simulation/01_backtest_first_principles.py" in notebooks
    assert "21_rl_execution_hedging/07_backtest_with_impact.py" in notebooks

    assert "03_market_microstructure/filter_itch_symbol.py" not in notebooks
    assert "03_market_microstructure/lob_utils.py" not in notebooks
    assert "03_market_microstructure/lob_generator.py" not in notebooks
    assert "13_dl_time_series/dl_utils.py" not in notebooks


# ---------------------------------------------------------------------------
# Tier / reruns / record_mode helpers
# ---------------------------------------------------------------------------


def test_get_tier_defaults_to_per_commit() -> None:
    assert get_tier({}) == TIER_PER_COMMIT
    assert get_tier({"tier": None}) == TIER_PER_COMMIT


def test_get_tier_accepts_valid_values() -> None:
    assert get_tier({"tier": "per_commit"}) == TIER_PER_COMMIT
    assert get_tier({"tier": "weekly"}) == TIER_WEEKLY
    assert get_tier({"tier": "on_demand"}) == TIER_ON_DEMAND


def test_get_tier_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid tier"):
        get_tier({"tier": "nightly"})


def test_current_test_tier_defaults_to_per_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ML4T_TEST_TIER", raising=False)
    assert current_test_tier() == TIER_PER_COMMIT


def test_current_test_tier_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML4T_TEST_TIER", "weekly")
    assert current_test_tier() == TIER_WEEKLY
    monkeypatch.setenv("ML4T_TEST_TIER", "on_demand")
    assert current_test_tier() == TIER_ON_DEMAND


def test_current_test_tier_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML4T_TEST_TIER", "bogus")
    with pytest.raises(ValueError, match="Invalid ML4T_TEST_TIER"):
        current_test_tier()


def test_get_reruns_default_zero() -> None:
    assert get_reruns({}) == 0


def test_get_reruns_returns_int() -> None:
    assert get_reruns({"reruns": 3}) == 3


def test_get_reruns_rejects_negative_or_nonint() -> None:
    with pytest.raises(ValueError):
        get_reruns({"reruns": -1})
    with pytest.raises(ValueError):
        get_reruns({"reruns": "2"})


def test_get_record_mode_defaults_to_replay() -> None:
    assert get_record_mode({}) == RECORD_REPLAY


def test_get_record_mode_accepts_rewrite() -> None:
    assert get_record_mode({"record_mode": "rewrite"}) == RECORD_REWRITE


def test_get_record_mode_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid record_mode"):
        get_record_mode({"record_mode": "none"})
