"""Tests for utils/cv_splits.py — walk-forward split generation.

Pins the invariants that every Ch11+ pipeline depends on:

- Pure duration/calendar normalization (regex-based, hermetic).
- load_evaluation_config reads setup.yaml's ``evaluation`` block and merges
  the market_data semantics calendar.
- generate_cv_splits produces n_splits folds with the correct chronology,
  backward walk-forward direction, embargo gap (label_buffer), and respects
  the holdout_start boundary.
- make_walk_forward_config returns int label_horizon for calendar-aware
  case studies (trading days) and Timedelta for 24/7 crypto.

Uses the real etfs and crypto_perps_funding setup.yaml files as ground
truth so the tests double as regression guards on those configs — if the
n_splits / train_size / val_size values are reordered, these tests will
flag it before a sweep wastes GPU time.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
import yaml

from utils.cv_splits import (
    _map_calendar_id,
    _normalize_duration,
    _normalize_label_buffer,
    generate_cv_splits,
    load_evaluation_config,
    make_walk_forward_config,
    make_wf_config,
)

# -----------------------------------------------------------------------------
# Pure: _map_calendar_id
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup_name, expected",
    [
        (None, None),
        ("NYSE", "NYSE"),
        ("CME", "CME_Equity"),
        ("FX", "CME_FX"),
        ("crypto", None),  # 24/7 → disable calendar-aware splitting
        ("LSE", "LSE"),  # unknown → pass through
    ],
)
def test_map_calendar_id(setup_name, expected) -> None:
    assert _map_calendar_id(setup_name) == expected


# -----------------------------------------------------------------------------
# Pure: _normalize_duration (ISO 8601 stripping + unit aliasing)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, normalized",
    [
        ("P5Y", "5Y"),
        ("P1Y", "1Y"),
        ("1Y", "1Y"),
        ("PT8H", "8h"),
        ("8H", "8h"),  # H → h for pd.Timedelta compatibility
        ("21D", "21D"),
        ("15T", "15min"),  # T is a legacy pandas minute alias
    ],
)
def test_normalize_duration(raw, normalized) -> None:
    assert _normalize_duration(raw) == normalized


# -----------------------------------------------------------------------------
# Pure: _normalize_label_buffer (inherits normalization + M → days)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, normalized",
    [
        ("21D", "21D"),
        ("PT8H", "8h"),
        ("1M", "30D"),  # month → 30 days (pd.Timedelta rejects raw M)
        ("3M", "90D"),
        ("P6M", "180D"),
    ],
)
def test_normalize_label_buffer(raw, normalized) -> None:
    assert _normalize_label_buffer(raw) == normalized


# -----------------------------------------------------------------------------
# load_evaluation_config
# -----------------------------------------------------------------------------


def test_load_evaluation_config_etfs_keys_and_values() -> None:
    """etfs is NYSE / 10Y train / 1Y val / 8 splits / backward (ground truth)."""
    cfg = load_evaluation_config("etfs")
    assert cfg["n_splits"] == 8
    assert cfg["train_size"] == "10Y"
    assert cfg["val_size"] == "1Y"
    assert cfg["holdout_start"] == "2024-01-01"
    assert cfg["holdout_end"] == "2025-12-31"
    assert cfg["calendar"] == "NYSE"


def test_load_evaluation_config_crypto_keeps_24_7_calendar() -> None:
    """crypto sets calendar: crypto (24/7); preserved in the returned config."""
    cfg = load_evaluation_config("crypto_perps_funding")
    assert cfg["calendar"] == "crypto"


def test_load_evaluation_config_raises_on_missing_section(tmp_path, monkeypatch) -> None:
    """A setup.yaml without an ``evaluation`` section raises KeyError.

    We spoof the case-study dir via ML4T_OUTPUT_DIR. The fallback path
    (re-read from source) won't find the fake id either, so the outer
    check raises.
    """
    cs_id = "_cv_splits_test_missing_evaluation"
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))

    cfg_dir = tmp_path / cs_id / "config"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "setup.yaml").write_text(yaml.safe_dump({"labels": {"primary": "x"}}))

    with pytest.raises(KeyError, match="evaluation"):
        load_evaluation_config(cs_id)


# -----------------------------------------------------------------------------
# generate_cv_splits — uses real etfs config (NYSE, 10Y/1Y, 8 splits, backward)
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def etfs_daily_frame() -> pl.DataFrame:
    """~24 years of business days — enough for 8 backward folds of 10+1 years."""
    ts = pd.date_range("1999-01-01", "2023-12-31", freq="B")
    return pl.DataFrame({"timestamp": pl.Series(ts)})


@pytest.fixture(scope="module")
def etfs_splits(etfs_daily_frame) -> list[dict]:
    return generate_cv_splits(etfs_daily_frame, case_study_id="etfs", label_buffer="21D")


def test_generate_cv_splits_etfs_returns_n_splits_folds(etfs_splits) -> None:
    assert len(etfs_splits) == 8


def test_generate_cv_splits_etfs_fold_ids_are_0_through_n_minus_1(etfs_splits) -> None:
    assert [s["fold"] for s in etfs_splits] == list(range(len(etfs_splits)))


def test_generate_cv_splits_etfs_folds_have_required_keys(etfs_splits) -> None:
    required = {"fold", "train_start", "train_end", "val_start", "val_end"}
    for s in etfs_splits:
        assert required <= set(s)


def test_generate_cv_splits_etfs_intra_fold_chronology(etfs_splits) -> None:
    """Within each fold: train_start ≤ train_end < val_start ≤ val_end."""
    for s in etfs_splits:
        assert s["train_start"] <= s["train_end"]
        assert s["train_end"] < s["val_start"]
        assert s["val_start"] <= s["val_end"]


def test_generate_cv_splits_etfs_backward_walk_forward(etfs_splits) -> None:
    """fold_direction=backward → fold 0 is the most recent, folds step back."""
    for i in range(len(etfs_splits) - 1):
        assert etfs_splits[i]["val_start"] > etfs_splits[i + 1]["val_start"]


def test_generate_cv_splits_etfs_embargo_respects_label_buffer(etfs_splits) -> None:
    """The gap between train_end and val_start covers the 21-trading-day label
    horizon. On NYSE that is roughly 29-32 calendar days; allow a generous
    lower bound to avoid flaking on holiday spacing.
    """
    for s in etfs_splits:
        gap = s["val_start"] - s["train_end"]
        assert gap >= pd.Timedelta(days=21), s  # at minimum 21 calendar days


def test_generate_cv_splits_etfs_val_before_holdout(etfs_splits) -> None:
    """All validation windows end strictly before the holdout_start (2024-01-01)."""
    holdout_start = pd.Timestamp("2024-01-01")
    for s in etfs_splits:
        assert s["val_end"] < holdout_start, s


def test_generate_cv_splits_etfs_train_size_10y(etfs_splits) -> None:
    """10Y train_size — span should be ~10 years (±2 months for calendar alignment)."""
    for s in etfs_splits:
        span = s["train_end"] - s["train_start"]
        assert pd.Timedelta(days=365 * 10 - 60) <= span <= pd.Timedelta(days=365 * 10 + 60), s


def test_generate_cv_splits_etfs_val_size_1y(etfs_splits) -> None:
    """1Y val_size — span should be ~1 year."""
    for s in etfs_splits:
        span = s["val_end"] - s["val_start"]
        assert pd.Timedelta(days=330) <= span <= pd.Timedelta(days=380), s


# -----------------------------------------------------------------------------
# generate_cv_splits — crypto (24/7, calendar=None after mapping)
# -----------------------------------------------------------------------------


def test_generate_cv_splits_crypto_respects_8h_buffer_and_no_calendar() -> None:
    ts = pd.date_range("2019-01-01", "2023-12-31", freq="8h")
    df = pl.DataFrame({"timestamp": pl.Series(ts)})
    splits = generate_cv_splits(df, case_study_id="crypto_perps_funding", label_buffer="8H")
    assert len(splits) == 2
    for s in splits:
        # 8h buffer means val_start ≥ train_end + 8h (may be slightly larger
        # because step is in 8-hour bars).
        gap = s["val_start"] - s["train_end"]
        assert gap >= pd.Timedelta(hours=8), s


# -----------------------------------------------------------------------------
# generate_cv_splits — input DataFrame flavors
# -----------------------------------------------------------------------------


def test_generate_cv_splits_accepts_pandas_dataframe() -> None:
    """Both pl.DataFrame and pd.DataFrame inputs produce identical splits."""
    ts = pd.date_range("1999-01-01", "2023-12-31", freq="B")
    pdf = pd.DataFrame({"timestamp": ts})
    pldf = pl.DataFrame({"timestamp": pl.Series(ts)})

    pd_splits = generate_cv_splits(pdf, case_study_id="etfs", label_buffer="21D")
    pl_splits = generate_cv_splits(pldf, case_study_id="etfs", label_buffer="21D")
    assert pd_splits == pl_splits


# -----------------------------------------------------------------------------
# generate_cv_splits — legacy cv_config dict path
# -----------------------------------------------------------------------------


def test_generate_cv_splits_cv_config_passthrough_of_precomputed_splits() -> None:
    """If cv_config already carries a ``splits`` list, return it unchanged."""
    precomputed = [
        {
            "fold": 0,
            "train_start": "2020-01-01",
            "train_end": "2022-12-31",
            "val_start": "2023-01-01",
            "val_end": "2023-12-31",
        }
    ]
    df = pl.DataFrame({"timestamp": pl.Series(pd.date_range("2020", "2023", freq="D"))})
    out = generate_cv_splits(df, cv_config={"splits": precomputed})
    assert out is precomputed or out == precomputed


def test_generate_cv_splits_cv_config_accepts_legacy_alias_keys() -> None:
    """Legacy keys test_size / test_start / test_end must be accepted.

    Old pipeline persisted cv_config.json with these aliases; the loader
    must still accept them so archived runs replay correctly.
    """
    cv = {
        "n_splits": 2,
        "train_size": "5Y",
        "test_size": "1Y",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31",
        "calendar": "NYSE",
    }
    ts = pd.date_range("2010-01-01", "2023-12-31", freq="B")
    df = pl.DataFrame({"timestamp": pl.Series(ts)})
    splits = generate_cv_splits(df, cv_config=cv, label_buffer="5D")
    assert len(splits) == 2
    for s in splits:
        assert s["train_end"] < s["val_start"]


def test_generate_cv_splits_cv_config_with_val_size_key_also_works() -> None:
    """Newer pipelines persist val_size / holdout_start — also supported."""
    cv = {
        "n_splits": 2,
        "train_size": "5Y",
        "val_size": "1Y",
        "holdout_start": "2023-01-01",
        "holdout_end": "2023-12-31",
        "calendar": "NYSE",
    }
    ts = pd.date_range("2010-01-01", "2023-12-31", freq="B")
    df = pl.DataFrame({"timestamp": pl.Series(ts)})
    splits = generate_cv_splits(df, cv_config=cv, label_buffer="5D")
    assert len(splits) == 2


# -----------------------------------------------------------------------------
# generate_cv_splits — error paths
# -----------------------------------------------------------------------------


def test_generate_cv_splits_raises_without_any_config_source() -> None:
    df = pl.DataFrame({"timestamp": pl.Series(pd.date_range("2020", "2023", freq="D"))})
    with pytest.raises(ValueError, match="case_study_id"):
        generate_cv_splits(df)


def test_generate_cv_splits_raises_on_empty_dataset() -> None:
    df = pl.DataFrame({"timestamp": pl.Series([], dtype=pl.Datetime)})
    with pytest.raises(ValueError, match="No timestamps"):
        generate_cv_splits(df, case_study_id="etfs", label_buffer="21D")


# -----------------------------------------------------------------------------
# make_walk_forward_config
# -----------------------------------------------------------------------------


def test_make_walk_forward_config_nyse_label_horizon_is_int_trading_days() -> None:
    """NYSE case study with a D-unit buffer passes label_horizon as int so
    the library counts trading days instead of calendar days.
    """
    cfg = make_walk_forward_config("etfs", label_horizon="21D")
    assert isinstance(cfg.label_horizon, int)
    assert cfg.label_horizon == 21
    assert cfg.calendar_id == "NYSE"
    assert cfg.n_splits == 8
    assert cfg.train_size == "10Y"
    assert cfg.test_size == "1Y"  # val_size → test_size alias
    assert cfg.fold_direction == "backward"


def test_make_walk_forward_config_crypto_label_horizon_is_timedelta() -> None:
    """24/7 crypto: calendar_id=None → horizon stays as string/Timedelta."""
    cfg = make_walk_forward_config("crypto_perps_funding", label_horizon="8H")
    assert cfg.calendar_id is None
    # Library may coerce to Timedelta; never an int for calendar-less case studies.
    assert not isinstance(cfg.label_horizon, int)


def test_make_walk_forward_config_holdout_dates_round_trip() -> None:
    """holdout_start / holdout_end from setup.yaml flow through to test_start / test_end."""
    cfg = make_walk_forward_config("etfs", label_horizon="21D")
    # Library stores as date objects
    assert str(cfg.test_start) == "2024-01-01"
    assert str(cfg.test_end) == "2025-12-31"


def test_make_wf_config_is_alias_of_make_walk_forward_config() -> None:
    """Backward-compat alias should delegate with identical output."""
    a = make_walk_forward_config("etfs", label_horizon="21D")
    b = make_wf_config("etfs", label_horizon="21D")
    assert a.model_dump() == b.model_dump()
