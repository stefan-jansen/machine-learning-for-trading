from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils import conformal
from case_studies.utils.allocation import compute_conformal_weights
from case_studies.utils.registry.specs import backtest_hash_from_parts
from case_studies.utils.strategy_analysis import rank_returns_on_common_support


def _panel_rows(n_steps: int = 12, fold_break: int = 8) -> list[dict[str, object]]:
    """A two-symbol panel on a regular grid whose residual equals the step it sits on.

    Symbol A is present at every step with ``|y_true - y_score| = step``, so the largest
    residual a calibration pool contains names the newest step that entered it - which is
    what makes the embargo checkable exactly rather than statistically. Symbol B appears
    every other step, so its own count lags the pooled one and the pooled fallback is
    exercised without a second fixture.
    """
    rows: list[dict[str, object]] = []
    for step in range(n_steps):
        fold = 1 if step < fold_break else 2
        rows.append(
            {
                "timestamp": datetime(2020, 1, 1) + timedelta(days=step),
                "symbol": "A",
                "y_true": float(step),
                "y_score": 0.0,
                "fold_id": fold,
            }
        )
        if step % 2 == 0:
            rows.append(
                {
                    "timestamp": datetime(2020, 1, 1) + timedelta(days=step),
                    "symbol": "B",
                    "y_true": 3.0,
                    "y_score": 0.0,
                    "fold_id": fold,
                }
            )
    return rows


def _write_predictions(root: Path, rows: list[dict[str, object]]) -> None:
    pred_dir = root / "run_log" / "predictions" / "candidate"
    pred_dir.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(pred_dir / "predictions.parquet")


def test_the_earliest_fold_calibrates_on_its_own_elapsed_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The earliest fold trades after a warm-up instead of sitting out entirely.

    Under the prior-fold-only rule it emitted no widths at all, the allocator dropped its
    rebalance dates, and the backtest booked a flat span that still counted as observed
    periods - which is how a strategy that sat out a losing fold outscored every strategy
    that traded it.
    """
    case_dir = tmp_path / "case_studies" / "demo"
    _write_predictions(case_dir, _panel_rows())
    monkeypatch.setattr(conformal, "get_case_study_dir", lambda _: case_dir)

    widths = conformal.compute_conformal_widths(
        "demo", "candidate", min_calibration_n=3, embargo_steps=2, alpha=0.0, write=False
    )

    earliest = widths.filter(pl.col("fold_id") == 1)
    assert not earliest.is_empty(), "the earliest fold must calibrate on its own history"
    # Three eligible residuals plus a two-step embargo. The pooled pool reaches three first,
    # at grid step one, so the earliest sized decision is grid step three.
    assert earliest["timestamp"].min() == datetime(2020, 1, 4)
    assert widths.filter(pl.col("timestamp") < datetime(2020, 1, 4)).is_empty()
    assert set(widths["calibration_scope"].unique()) == {"symbol", "pooled"}
    assert widths["calibration_version"].unique().to_list() == ["walk_forward_v3"]


def test_no_width_reads_a_residual_inside_its_own_label_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The embargo holds at every step, not only at the validation/holdout boundary.

    A residual at ``t'`` is the return realized over ``(t', t'+h]``, so calibrating fold k on
    all of fold k-1 with no embargo carried fold-k prices into the widths that sized fold-k
    positions. The fixture's residual equals its step, so at alpha zero the width is twice
    the newest step the pool was allowed to see - an exact statement about what was read.
    """
    case_dir = tmp_path / "case_studies" / "demo"
    _write_predictions(case_dir, _panel_rows())
    monkeypatch.setattr(conformal, "get_case_study_dir", lambda _: case_dir)

    for embargo in (1, 2, 3):
        widths = conformal.compute_conformal_widths(
            "demo",
            "candidate",
            min_calibration_n=3,
            embargo_steps=embargo,
            alpha=0.0,
            write=False,
        )
        own = widths.filter(
            (pl.col("symbol") == "A") & (pl.col("calibration_scope") == "symbol")
        ).with_columns(step=(pl.col("timestamp") - datetime(2020, 1, 1)).dt.total_days())
        assert not own.is_empty()
        assert own.filter(pl.col("width") != 2.0 * (pl.col("step") - embargo)).is_empty(), (
            f"a width at embargo={embargo} read a residual newer than its horizon allows"
        )
        assert own.filter(pl.col("calibration_n") != pl.col("step") - embargo + 1).is_empty()


def test_conformal_widths_require_the_label_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case_studies" / "demo"
    _write_predictions(case_dir, _panel_rows())
    monkeypatch.setattr(conformal, "get_case_study_dir", lambda _: case_dir)

    with pytest.raises(ValueError, match="needs the label horizon"):
        conformal.compute_conformal_widths("demo", "candidate", write=False)

    with pytest.raises(ValueError, match="not yet realized"):
        conformal.compute_conformal_widths(
            "demo", "candidate", embargo_steps=0, min_calibration_n=3, write=False
        )


def test_conformal_allocation_rejects_missing_selected_widths() -> None:
    predictions = pl.DataFrame(
        {
            "timestamp": [datetime(2020, 2, 1), datetime(2020, 2, 1)],
            "symbol": ["A", "B"],
            "y_score": [2.0, 1.0],
        }
    )
    incomplete_widths = pl.DataFrame(
        {
            "timestamp": [datetime(2020, 2, 1)],
            "symbol": ["A"],
            "width": [1.0],
        }
    )

    with pytest.raises(ValueError, match="missing widths for selected assets"):
        compute_conformal_weights(predictions, incomplete_widths, top_k=2)


def test_conformal_weight_floor_never_uses_future_timestamps() -> None:
    """Appending future widths must not change already emitted allocation weights."""
    early = datetime(2024, 1, 1)
    future = datetime(2024, 2, 1)
    symbols = [f"S{index:03d}" for index in range(100)]
    predictions = pl.DataFrame(
        {
            "timestamp": [early] * 100 + [future] * 100,
            "symbol": symbols * 2,
            "y_score": [float(100 - index) for index in range(100)] * 2,
        }
    )
    widths = pl.DataFrame(
        {
            "timestamp": [early] * 100 + [future] * 100,
            "symbol": symbols * 2,
            "width": [1e-9] + [1.0] * 99 + [1e-12] * 100,
        }
    )

    prefix = compute_conformal_weights(
        predictions.filter(pl.col("timestamp") == early),
        widths.filter(pl.col("timestamp") == early),
        top_k=100,
    ).sort("timestamp", "symbol")
    full = (
        compute_conformal_weights(predictions, widths, top_k=100)
        .filter(pl.col("timestamp") == early)
        .sort("timestamp", "symbol")
    )

    max_difference = (
        prefix.join(full, on=["timestamp", "symbol"], suffix="_full")
        .select((pl.col("weight") - pl.col("weight_full")).abs().max())
        .item()
    )
    assert max_difference < 1e-12


def test_calibration_contract_changes_backtest_identity() -> None:
    legacy = {
        "strategy": {
            "signal": {"method": "equal_weight_top_k", "top_k": 10},
            "allocation": {"method": "conformal_weighted", "top_k": 10},
        }
    }
    corrected = conformal.ensure_conformal_calibration_identity(legacy)

    assert corrected["strategy"]["allocation"]["calibration_version"] == "walk_forward_v3"
    assert backtest_hash_from_parts("pred", legacy) != backtest_hash_from_parts("pred", corrected)


def test_legacy_width_artifact_must_be_preserved_before_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case_studies" / "demo"
    _write_predictions(case_dir, _panel_rows())
    legacy_path = case_dir / "run_log" / "predictions" / "candidate" / "conformal_widths.parquet"
    pl.DataFrame(
        {
            "timestamp": [datetime(2020, 1, 1)],
            "symbol": ["A"],
            "fold_id": [0],
            "width": [2.0],
            "alpha": [0.2],
            "calibration_n": [2],
        }
    ).write_parquet(legacy_path)
    monkeypatch.setattr(conformal, "get_case_study_dir", lambda _: case_dir)

    with pytest.raises(ValueError, match="preserve it in the pre-fix snapshot"):
        conformal.compute_conformal_widths(
            "demo", "candidate", min_calibration_n=3, embargo_steps=2, write=True
        )


def test_a_superseded_calibration_version_is_refused_rather_than_mixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Widths from the retired contract must not sit beside widths from the current one.

    The artifact carries no per-row provenance beyond its version, so a file holding both
    would size some decisions on prior-fold-only widths and some on embargoed expanding
    ones with nothing to tell them apart.
    """
    case_dir = tmp_path / "case_studies" / "demo"
    _write_predictions(case_dir, _panel_rows())
    superseded = case_dir / "run_log" / "predictions" / "candidate" / "conformal_widths.parquet"
    pl.DataFrame(
        {
            "timestamp": [datetime(2020, 1, 1)],
            "symbol": ["A"],
            "fold_id": [0],
            "width": [2.0],
            "alpha": [0.2],
            "calibration_n": [2],
            "calibration_scope": ["symbol"],
            "calibration_version": ["walk_forward_v2"],
        }
    ).write_parquet(superseded)
    monkeypatch.setattr(conformal, "get_case_study_dir", lambda _: case_dir)

    with pytest.raises(ValueError, match="Refusing to mix conformal calibration versions"):
        conformal.compute_conformal_widths(
            "demo", "candidate", min_calibration_n=3, embargo_steps=2, write=True
        )


def test_locked_width_retry_rejects_conflict_without_replacing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "conformal_widths.parquet"
    original = pl.DataFrame(
        {
            "timestamp": [datetime(2020, 1, 1)],
            "symbol": ["A"],
            "fold_id": [-1],
            "width": [2.0],
            "alpha": [0.2],
            "calibration_version": [conformal.CALIBRATION_VERSION],
        }
    )
    original.write_parquet(path)
    conflicting = original.with_columns(pl.lit(3.0).alias("width"))

    with pytest.raises(ValueError, match="locked conformal artifact conflicts"):
        conformal._write_widths(path, conflicting, 0.2, immutable=True)

    assert pl.read_parquet(path).equals(original)


def test_common_support_ranking_uses_identical_timestamps() -> None:
    full = pl.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
            ],
            "daily_return": [0.50, 0.01, -0.01, 0.01],
        }
    )
    strict = pl.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
            ],
            "daily_return": [0.02, 0.00, 0.01],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))

    ranked = rank_returns_on_common_support({"full": full, "strict": strict}, periods_per_year=252)

    assert ranked["n_periods"].unique().to_list() == [3]
    assert ranked["start"].unique().to_list() == [datetime(2020, 1, 2)]
    assert ranked["end"].unique().to_list() == [datetime(2020, 1, 4)]


def test_conformal_allocation_accepts_tz_aware_widths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Widths keep the predictions parquet's time zone; weights are always tz-naive.

    ``normalize_prediction_columns`` strips the time zone from predictions before
    signal weights are built, while ``compute_conformal_widths`` reads
    predictions.parquet directly and preserves it. Case studies that store UTC-aware
    timestamps (crypto_perps_funding) therefore reach ``_apply_allocation`` with a
    tz-aware widths frame and a tz-naive weights frame, which raised
    ``SchemaError: datatypes of join keys don't match`` on the common-support join.
    """
    import case_studies.utils.backtest_loaders as loaders
    from case_studies.utils.backtest_runner import _apply_allocation

    stamps = [datetime(2024, 1, 1, 8), datetime(2024, 1, 1, 16)]
    naive = pl.DataFrame(
        {
            "timestamp": stamps * 2,
            "symbol": ["BTC", "BTC", "ETH", "ETH"],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
    weights = naive.with_columns(weight=pl.lit(0.5))
    predictions = naive.with_columns(y_score=pl.Series([2.0, 2.0, 1.0, 1.0]))
    widths = naive.with_columns(
        width=pl.lit(1.0),
        timestamp=pl.col("timestamp").dt.replace_time_zone("UTC"),
    )
    assert widths["timestamp"].dtype.time_zone == "UTC"

    monkeypatch.setattr(loaders, "get_rebalance_step", lambda *_args: 1)

    result = _apply_allocation(
        weights,
        predictions,
        naive.with_columns(close=pl.lit(100.0)),
        {"method": "conformal_weighted", "top_k": 2},
        label="fwd_ret_24h",
        case_study="crypto_perps_funding",
        prediction_hash="unused_widths_are_passed_in",
        conformal_widths=widths,
    )

    assert result.height == 4
    assert result["timestamp"].dtype == weights["timestamp"].dtype


def test_a_missing_prediction_still_names_the_directory_it_was_expected_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace" / "demo"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(conformal, "get_case_study_dir", lambda _: workspace)

    with pytest.raises(FileNotFoundError, match=str(workspace)):
        conformal.compute_conformal_widths("demo", "candidate", write=False)


def test_widths_compute_on_a_panel_keyed_by_integer_identifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entity column that is not a string must still calibrate.

    `join_asof` matches its `by` columns by dtype. The per-entity calibration table builds
    its key from a Python value, so an unannotated literal infers Int32 and a panel keyed on
    UInt32 identifiers - permnos, on us_firm_characteristics - raises rather than sizing
    anything. Every conformal cell of that case study's allocation stage failed on it.
    """
    rows = [{**row, "symbol": 1001 if row["symbol"] == "A" else 1002} for row in _panel_rows()]
    case_dir = tmp_path / "case_studies" / "demo"
    pred_dir = case_dir / "run_log" / "predictions" / "candidate"
    pred_dir.mkdir(parents=True)
    pl.DataFrame(rows).with_columns(pl.col("symbol").cast(pl.UInt32)).write_parquet(
        pred_dir / "predictions.parquet"
    )
    monkeypatch.setattr(conformal, "get_case_study_dir", lambda _: case_dir)

    widths = conformal.compute_conformal_widths(
        "demo", "candidate", min_calibration_n=3, embargo_steps=2, alpha=0.0, write=False
    )

    assert not widths.is_empty()
    assert widths.schema["symbol"] == pl.UInt32
    assert set(widths["symbol"].unique().to_list()) == {1001, 1002}


def test_the_holdout_embargo_is_part_of_the_backtest_identity() -> None:
    """Two embargoes are two calibrations, so they must not share a backtest hash.

    The widths are an input to the backtest and the embargo decides them, but the widths
    live in an artifact beside the prediction set and nothing in the strategy specification
    named them. Changing the embargo therefore left the hash where it was, and the registry
    refused to overwrite the registered run rather than accepting either result - which is
    how the state announced itself, and it announced a conflict rather than a number.
    """
    spec = {
        "version": 2,
        "strategy": {"allocation": {"method": "conformal_weighted", "min_calibration_n": 30}},
        "backtest_config": {"cash": {"initial": 1_000_000.0}},
    }
    zero = conformal.ensure_conformal_calibration_identity(spec, holdout_embargo_steps=0)
    one = conformal.ensure_conformal_calibration_identity(spec, holdout_embargo_steps=1)

    assert zero["backtest_config"]["calibration"]["holdout_embargo_steps"] == 0
    assert one["backtest_config"]["calibration"]["holdout_embargo_steps"] == 1
    assert zero != one

    # It sits outside `strategy` because that block is what a holdout replay is matched to
    # its validation carrier by. The two run the same strategy; the embargo is a property
    # of calibrating across the boundary between them.
    assert zero["strategy"] == one["strategy"]

    # A validation run records nothing: the embargo has no meaning within validation, and
    # writing it there would rehash every registered conformal backtest for no difference.
    assert (
        "calibration"
        not in conformal.ensure_conformal_calibration_identity(spec)["backtest_config"]
    )
