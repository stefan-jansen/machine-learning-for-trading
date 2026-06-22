"""Tests for ``case_studies.utils.backtest_loaders.warmup_periods_for`` +
``_calendar_days_per_period`` — the helpers that replaced the hardcoded
``warmup_periods=126`` constant duplicated across 16 call-sites in 5 CSes.

These tests close P2.8 of the roborev cleanup (review #2510 / #2511).
"""

from __future__ import annotations

import yaml

from case_studies.utils.backtest_loaders import (
    _calendar_days_per_period,
    _load_case_setup_yaml,
    warmup_periods_for,
)

# The expected per-CS warmup is the max over ``execution.allocator_lookback``
# and any per-sweep allocator ``vol_window`` / ``lookback`` overrides. These
# expectations are anchored on the current setup.yaml values; if a CS
# tunes its allocator lookbacks, update the expected value here.
_EXPECTED_WARMUP: dict[str, int] = {
    "etfs": 63,
    "crypto_perps_funding": 240,
    "nasdaq100_microstructure": 520,
    "us_equities_panel": 126,  # mvo_ledoit_wolf lookback=126 > allocator_lookback=63
    "us_firm_characteristics": 12,
    "fx_pairs": 63,
    "cme_futures": 63,
    "sp500_options": 63,
    "sp500_equity_option_analytics": 126,  # mvo_ledoit_wolf lookback=126
}


def test_warmup_periods_for_matches_setup_yaml() -> None:
    for cs, expected in _EXPECTED_WARMUP.items():
        actual = warmup_periods_for(cs)
        assert actual == expected, (
            f"warmup_periods_for({cs}) = {actual}, expected {expected} "
            f"(max of execution.allocator_lookback + sweep allocator overrides)"
        )


def test_warmup_periods_for_unknown_returns_zero(tmp_path) -> None:
    # No setup.yaml → defaults to 0 (the unbounded fallback inside
    # load_backtest_prices_for then skips the prefix-day computation).
    assert warmup_periods_for("__nonexistent_cs__") == 0


def test_warmup_periods_for_picks_max_over_overrides(tmp_path, monkeypatch) -> None:
    """When a per-allocator override exceeds allocator_lookback, the helper
    must surface the override rather than the CS-level default."""
    fake_setup = {
        "execution": {"allocator_lookback": 50},
        "backtest": {
            "sweep": {
                "allocators": [
                    {"name": "equal_weight"},
                    {"name": "iv", "vol_window": 200},
                    {"name": "mvo_lw", "lookback": 100},
                ]
            }
        },
    }
    cs_dir = tmp_path / "fake_cs" / "config"
    cs_dir.mkdir(parents=True)
    (cs_dir / "setup.yaml").write_text(yaml.safe_dump(fake_setup))

    # Drop the cache so the synthetic CS gets a fresh read.
    _load_case_setup_yaml.cache_clear()

    from case_studies.utils.backtest_loaders import warmup_periods_for as wpf
    from utils.paths import get_case_study_dir as orig_get_dir

    def fake_get_dir(cs: str):
        if cs == "fake_cs":
            return tmp_path / "fake_cs"
        return orig_get_dir(cs)

    monkeypatch.setattr("case_studies.utils.backtest_loaders.get_case_study_dir", fake_get_dir)
    _load_case_setup_yaml.cache_clear()
    assert wpf("fake_cs") == 200


def test_calendar_days_per_period_cadence_aware() -> None:
    # Daily NYSE cadence: 1.5× (weekend + holiday allowance)
    assert abs(_calendar_days_per_period("fx_pairs") - 1.5) < 1e-9
    assert abs(_calendar_days_per_period("us_equities_panel") - 1.5) < 1e-9
    # Weekly cadence: 7 calendar days per bar
    assert abs(_calendar_days_per_period("cme_futures") - 7.0) < 1e-9
    assert abs(_calendar_days_per_period("sp500_equity_option_analytics") - 7.0) < 1e-9
    # 8-hour cadence: ~0.333 day per bar (3 bars / 24h day)
    assert abs(_calendar_days_per_period("crypto_perps_funding") - 1.0 / 3.0) < 1e-9
    # 15-minute cadence: ~0.054 day per bar (1/26 trading day × 1.4 calendar buffer)
    assert _calendar_days_per_period("nasdaq100_microstructure") < 0.1
    # Monthly cadence: ~31 calendar days per bar
    assert abs(_calendar_days_per_period("us_firm_characteristics") - 31.0) < 1e-9


def test_calendar_days_per_period_default_for_unknown_cs() -> None:
    # Falls back to the daily 1.5× heuristic when no setup.yaml is present
    # or the cadence token isn't in the lookup table.
    assert _calendar_days_per_period("__nonexistent_cs__") == 1.5
