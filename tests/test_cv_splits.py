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

from pathlib import Path

import pandas as pd
import polars as pl
import pytest
import yaml

from utils.cv_splits import (
    _assert_newest_first,
    _map_calendar_id,
    _normalize_duration,
    _normalize_label_buffer,
    earliest_train_start,
    generate_cv_splits,
    load_evaluation_config,
    make_walk_forward_config,
    make_wf_config,
    most_recent_split,
)
from utils.modeling import validate_temporal_fold_coverage, validate_temporal_split_geometry

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
        ("P5Y", "5YE"),
        ("P1Y", "1YE"),
        ("1Y", "1YE"),
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
    """~24 years of business days, including dates inside the sealed holdout."""
    ts = pd.date_range("1999-01-01", "2024-01-31", freq="B")
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


def test_generate_cv_splits_etfs_val_before_holdout(etfs_splits, etfs_daily_frame) -> None:
    """Every 21-session validation label ends before the holdout."""
    holdout_start = pd.Timestamp("2024-01-01")
    timestamps = etfs_daily_frame.select("timestamp").to_series().to_pandas()
    holdout_pos = int(pd.DatetimeIndex(timestamps).searchsorted(holdout_start, side="left"))
    for s in etfs_splits:
        val_end_pos = int(pd.DatetimeIndex(timestamps).searchsorted(s["val_end"], side="left"))
        assert val_end_pos + 21 < holdout_pos, s


def test_generate_cv_splits_etfs_label_outcome_ends_before_holdout(
    etfs_daily_frame, etfs_splits
) -> None:
    """The last validation decision's 21-session outcome must remain pre-holdout."""
    dates = etfs_daily_frame["timestamp"].to_list()
    date_index = {timestamp: index for index, timestamp in enumerate(dates)}
    holdout_start = pd.Timestamp("2024-01-01")

    for split in etfs_splits:
        exit_timestamp = dates[date_index[split["val_end"]] + 21]
        assert exit_timestamp < holdout_start, split


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
        assert s["val_end"] + pd.Timedelta(hours=8) < pd.Timestamp("2024-01-01"), s


def test_generate_cv_splits_crypto_purges_variant_endpoint_at_holdout() -> None:
    ts = pd.date_range("2019-01-01", "2023-12-31 16:00", freq="8h")
    df = pl.DataFrame({"timestamp": pl.Series(ts)})

    splits = generate_cv_splits(
        df,
        case_study_id="crypto_perps_funding",
        label_buffer="24H",
    )

    assert splits[0]["val_end"] == pd.Timestamp("2023-12-30 16:00")
    assert splits[0]["val_end"] + pd.Timedelta(hours=24) < pd.Timestamp("2024-01-01")


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
# Fold-specific temporal artifact alignment
# -----------------------------------------------------------------------------


@pytest.fixture
def backward_temporal_fixture() -> tuple[pl.DataFrame, pl.DataFrame, list[dict]]:
    dates = pd.date_range("2017-01-02", "2020-12-31", freq="B")
    dataset = pl.DataFrame({"timestamp": dates, "symbol": ["A"] * len(dates)})
    splits = [
        {
            "fold": 0,
            "train_start": pd.Timestamp("2018-01-01"),
            "train_end": pd.Timestamp("2019-12-31"),
            "val_start": pd.Timestamp("2020-01-01"),
            "val_end": pd.Timestamp("2020-12-31"),
        },
        {
            "fold": 1,
            "train_start": pd.Timestamp("2017-01-01"),
            "train_end": pd.Timestamp("2018-12-31"),
            "val_start": pd.Timestamp("2019-01-01"),
            "val_end": pd.Timestamp("2019-12-31"),
        },
    ]
    forward_numbered = pl.concat(
        [
            pl.DataFrame(
                {
                    "timestamp": pd.date_range("2017-01-02", "2019-12-31", freq="B"),
                    "fold": 0,
                }
            ),
            pl.DataFrame(
                {
                    "timestamp": pd.date_range("2018-01-01", "2020-12-31", freq="B"),
                    "fold": 1,
                }
            ),
        ]
    ).with_row_index("value")
    return dataset, forward_numbered, splits


def test_temporal_fold_validation_rejects_forward_numbering(backward_temporal_fixture) -> None:
    dataset, temporal, splits = backward_temporal_fixture

    with pytest.raises(ValueError, match=r"fold 0 validation.*0/.*0\.0%"):
        validate_temporal_fold_coverage(dataset, temporal, splits, date_col="timestamp")


def test_temporal_fold_metadata_remap_restores_coverage(backward_temporal_fixture) -> None:
    dataset, temporal, splits = backward_temporal_fixture
    values_before = temporal["value"].sort().to_list()
    remapped = temporal.with_columns((1 - pl.col("fold")).alias("fold"))

    validate_temporal_fold_coverage(dataset, remapped, splits, date_col="timestamp")

    assert remapped["value"].sort().to_list() == values_before


def test_custom_cv_cannot_reuse_temporal_features_from_different_geometry() -> None:
    canonical = [
        {
            "fold": 0,
            "train_start": "2018-01-01",
            "train_end": "2019-12-31",
            "val_start": "2020-01-01",
            "val_end": "2020-12-31",
        }
    ]
    requested = [{**canonical[0], "val_start": "2019-07-01"}]
    temporal = pl.DataFrame({"fold": [0], "timestamp": [pd.Timestamp("2020-01-01")]})

    with pytest.raises(ValueError, match=r"fold 0 differs in \['val_start'\]"):
        validate_temporal_split_geometry(requested, canonical, temporal)


def test_custom_cv_can_select_exact_fitted_temporal_fold_geometry() -> None:
    canonical = [
        {
            "fold": fold,
            "train_start": f"{2018 + fold}-01-01",
            "train_end": f"{2019 + fold}-12-31",
            "val_start": f"{2020 + fold}-01-01",
            "val_end": f"{2020 + fold}-12-31",
        }
        for fold in (0, 1)
    ]
    temporal = pl.DataFrame(
        {"fold": [0, 1], "timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")]}
    )

    validate_temporal_split_geometry([canonical[1]], canonical, temporal)


@pytest.fixture
def warmup_temporal_fixture() -> tuple[pl.DataFrame, list[dict]]:
    """One fold whose artifact can be trimmed to simulate a burn-in prefix."""
    dates = pd.date_range("2018-01-01", "2020-12-31", freq="B")
    dataset = pl.DataFrame({"timestamp": dates, "symbol": ["A"] * len(dates)})
    splits = [
        {
            "fold": 0,
            "train_start": pd.Timestamp("2018-01-01"),
            "train_end": pd.Timestamp("2019-12-31"),
            "val_start": pd.Timestamp("2020-01-01"),
            "val_end": pd.Timestamp("2020-12-31"),
        }
    ]
    return dataset, splits


def _temporal_from(dates: pd.DatetimeIndex) -> pl.DataFrame:
    return pl.DataFrame({"timestamp": dates, "fold": 0})


def test_temporal_warmup_prefix_within_bound_is_excused(warmup_temporal_fixture) -> None:
    dataset, splits = warmup_temporal_fixture
    dates = pd.DatetimeIndex(dataset["timestamp"].to_pandas())
    train = dates[dates <= pd.Timestamp("2019-12-31")]
    # 8% of the train window unavailable at its start, as a rolling warm-up is.
    trimmed = dates[dates >= train[int(len(train) * 0.08)]]

    validate_temporal_fold_coverage(dataset, _temporal_from(trimmed), splits, date_col="timestamp")


def test_temporal_warmup_prefix_beyond_bound_still_fails(warmup_temporal_fixture) -> None:
    dataset, splits = warmup_temporal_fixture
    dates = pd.DatetimeIndex(dataset["timestamp"].to_pandas())
    train = dates[dates <= pd.Timestamp("2019-12-31")]
    # A shifted fold looks like this: a leading gap over half the train window.
    trimmed = dates[dates >= train[int(len(train) * 0.5)]]

    with pytest.raises(ValueError, match=r"fold 0 train: temporal date coverage"):
        validate_temporal_fold_coverage(
            dataset, _temporal_from(trimmed), splits, date_col="timestamp"
        )


def test_temporal_warmup_allowance_does_not_apply_to_validation(warmup_temporal_fixture) -> None:
    dataset, splits = warmup_temporal_fixture
    dates = pd.DatetimeIndex(dataset["timestamp"].to_pandas())
    val = dates[dates >= pd.Timestamp("2020-01-01")]
    # 8% at the start of the window: excused on train, fatal on validation.
    keep = dates[(dates < pd.Timestamp("2020-01-01")) | (dates >= val[int(len(val) * 0.08)])]

    with pytest.raises(ValueError, match=r"fold 0 validation: temporal date coverage"):
        validate_temporal_fold_coverage(dataset, _temporal_from(keep), splits, date_col="timestamp")


def test_temporal_interior_gap_is_not_excused(warmup_temporal_fixture) -> None:
    dataset, splits = warmup_temporal_fixture
    dates = pd.DatetimeIndex(dataset["timestamp"].to_pandas())
    train = dates[dates <= pd.Timestamp("2019-12-31")]
    gap = train[int(len(train) * 0.3) : int(len(train) * 0.5)]
    keep = dates[~dates.isin(gap)]

    with pytest.raises(ValueError, match=r"fold 0 train: temporal date coverage"):
        validate_temporal_fold_coverage(dataset, _temporal_from(keep), splits, date_col="timestamp")


def test_sp500_options_temporal_producer_uses_canonical_split_ids() -> None:
    source = Path("case_studies/sp500_options/04_model_based_features.py").read_text()

    assert "generate_cv_splits(" in source
    assert 'fold_idx = fold["fold"]' in source
    assert "first_test_year" not in source


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
    assert cfg.train_size == "10YE"
    assert cfg.test_size == "1YE"  # val_size → test_size alias
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


# -----------------------------------------------------------------------------
# Fold ordering, and the accessors that do not depend on it
# -----------------------------------------------------------------------------


def test_generate_cv_splits_returns_folds_newest_first(etfs_splits) -> None:
    """Fold 0 validates most recently. Roughly forty call sites read it that way."""
    val_starts = [s["val_start"] for s in etfs_splits]
    assert val_starts == sorted(val_starts, reverse=True)
    assert etfs_splits[0]["val_end"] > etfs_splits[-1]["val_end"]


def test_fold_0_carries_the_latest_train_start_not_the_earliest(etfs_splits) -> None:
    """The shape behind the measured defect: indexing for "everything available"."""
    assert etfs_splits[0]["train_start"] > etfs_splits[-1]["train_start"]
    assert etfs_splits[0]["train_start"] != earliest_train_start(etfs_splits)


def test_an_ascending_fold_list_is_refused_rather_than_returned() -> None:
    """A library change to fold_direction must fail here, not at forty call sites."""
    ascending = [
        {"fold": 0, "val_start": pd.Timestamp("2020-01-01"), "val_end": pd.Timestamp("2020-12-31")},
        {"fold": 1, "val_start": pd.Timestamp("2021-01-01"), "val_end": pd.Timestamp("2021-12-31")},
    ]
    with pytest.raises(RuntimeError, match="not ordered newest first"):
        _assert_newest_first(ascending)

    # Reversing the list alone leaves fold 0 on the oldest window. Every join is
    # by id, so the ids have to move with the positions.
    with pytest.raises(RuntimeError, match="fold ids"):
        _assert_newest_first(list(reversed(ascending)))

    _assert_newest_first([{**split, "fold": i} for i, split in enumerate(reversed(ascending))])


def test_a_precomputed_split_set_is_held_to_the_same_order() -> None:
    """A caller cannot tell which path produced its list, so both owe the contract.

    The two committed configs disagree with each other:
    us_firm_characteristics/config/cv_config.json runs newest first, and
    fx_pairs/config/cv_config.json runs oldest first - fold 0 validates from
    2015-10-28 against fold 7 at 2022-12-15 - while fx_pairs/04_model_based_features
    tags its artifact through generate_cv_splits. Fold 0 then means the earliest
    window on one side of the join and the latest on the other.
    """
    df = pl.DataFrame({"timestamp": pd.date_range("2010-01-01", "2020-01-01", freq="B")})
    ascending = {
        "splits": [
            {"fold": 0, "val_start": "2015-10-28", "val_end": "2016-10-28"},
            {"fold": 1, "val_start": "2016-11-15", "val_end": "2017-11-15"},
        ]
    }
    with pytest.raises(RuntimeError, match="not ordered newest first"):
        generate_cv_splits(df, cv_config=ascending)

    # Reversing the list is not the fix: fold 0 still names the oldest window and
    # every downstream join is by id.
    reversed_only = {"splits": list(reversed(ascending["splits"]))}
    with pytest.raises(RuntimeError, match="fold ids"):
        generate_cv_splits(df, cv_config=reversed_only)

    renumbered = {
        "splits": [{**split, "fold": i} for i, split in enumerate(reversed(ascending["splits"]))]
    }
    assert [s["fold"] for s in generate_cv_splits(df, cv_config=renumbered)] == [0, 1]


def test_fx_materialized_folds_match_the_canonical_label_clock() -> None:
    import json

    from utils import CASE_STUDIES_DIR
    from utils.artifact_specs import load_label_spec, resolve_storage_path
    from utils.modeling import resolve_label_buffer, resolve_label_horizon

    case_study = "fx_pairs"
    label = "fwd_ret_1d"
    source_case_dir = CASE_STUDIES_DIR / case_study
    setup = yaml.safe_load((source_case_dir / "config" / "setup.yaml").read_text())
    label_path = resolve_storage_path(
        case_study,
        load_label_spec(case_study, label),
        f"labels/{label}.parquet",
    )
    if not label_path.exists():
        pytest.skip("Production FX label artifact is not available")
    labels = pl.read_parquet(label_path)
    canonical = generate_cv_splits(
        labels,
        case_study_id=case_study,
        label_buffer=resolve_label_buffer(case_study, label, setup),
        outcome_horizon=resolve_label_horizon(case_study, label, setup),
    )
    materialized = generate_cv_splits(
        labels,
        cv_config=json.loads((source_case_dir / "config" / "cv_config.json").read_text()),
        label_buffer=resolve_label_buffer(case_study, label, setup),
    )

    boundary_keys = ("fold", "train_start", "train_end", "val_start", "val_end")

    def normalized(splits):
        return [
            {
                key: split[key] if key == "fold" else pd.Timestamp(split[key])
                for key in boundary_keys
            }
            for split in splits
        ]

    assert normalized(materialized) == normalized(canonical)


def test_the_order_check_reads_a_stored_config_spelling() -> None:
    """A legacy config writes test_start where the generated path writes val_start."""
    _assert_newest_first(
        [
            {"fold": 0, "test_start": pd.Timestamp("2020-01-01")},
            {"fold": 1, "test_start": pd.Timestamp("2019-01-01")},
        ]
    )
    with pytest.raises(RuntimeError):
        _assert_newest_first(
            [
                {"fold": 0, "test_start": pd.Timestamp("2019-01-01")},
                {"fold": 1, "test_start": pd.Timestamp("2020-01-01")},
            ]
        )


def test_most_recent_split_reads_the_boundaries_not_the_position() -> None:
    """Same folds, three orders, one answer - unlike splits[0] and splits[-1]."""
    folds = [
        {
            "fold": 0,
            "val_end": pd.Timestamp("2023-11-29"),
            "train_start": pd.Timestamp("2013-01-17"),
        },
        {
            "fold": 1,
            "val_end": pd.Timestamp("2022-12-28"),
            "train_start": pd.Timestamp("2012-01-18"),
        },
        {
            "fold": 2,
            "val_end": pd.Timestamp("2016-12-23"),
            "train_start": pd.Timestamp("2006-01-13"),
        },
    ]
    for ordering in (folds, list(reversed(folds)), [folds[1], folds[2], folds[0]]):
        assert most_recent_split(ordering)["fold"] == 0
        assert earliest_train_start(ordering) == pd.Timestamp("2006-01-13")


def test_the_accessors_refuse_an_empty_fold_set() -> None:
    with pytest.raises(ValueError, match="No splits"):
        most_recent_split([])
    with pytest.raises(ValueError, match="No splits"):
        earliest_train_start([])
