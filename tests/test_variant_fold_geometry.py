"""A variant label's fold must validate after the primary's fold stops training.

A stage-04 artifact carries one fold set, built on the primary label's geometry.
A model trained on a variant label reads that artifact by ``fold`` id, so the values
it gets for fold F were fit on data through the *primary* label's ``train_end``. The
variant's validation window has to open after that.

Measured across the eight stage-04 case studies on 2026-08-09: 20 variant labels,
zero violations. Nothing in the code would have noticed if there had been one - five
of the eight assert nothing, and fx_pairs' assertion compares the outer span, which
never reads ``val_start``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import case_studies.utils.cv_window as cv_window
from case_studies.research.cv import require_fold_scoped_temporal_compatibility
from case_studies.utils.cv_window import assert_variant_folds_are_out_of_sample


def _splits(spec: list[tuple[str, str, str, str]]) -> list[dict]:
    return [
        {
            "fold": i,
            "train_start": pd.Timestamp(ts),
            "train_end": pd.Timestamp(te),
            "val_start": pd.Timestamp(vs),
            "val_end": pd.Timestamp(ve),
        }
        for i, (ts, te, vs, ve) in enumerate(spec)
    ]


@pytest.fixture
def geometries(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict]]:
    """One fold set per label, returned by the derivation the producers also use."""
    store: dict[str, list[dict]] = {}
    monkeypatch.setattr(cv_window, "_derive_modeling_splits", lambda _cs, label: store.get(label))
    monkeypatch.setattr(
        cv_window,
        "_load_setup_yaml",
        lambda _cs: {"labels": {"primary": "primary", "variants": ["variant"]}},
    )
    return store


def test_a_variant_validating_before_the_fit_ends_is_refused(geometries) -> None:
    """The leak: fold 0's features were fit through the 20th, the variant scores the 15th."""
    geometries["primary"] = _splits([("2020-01-01", "2020-06-20", "2020-06-25", "2020-12-31")])
    geometries["variant"] = _splits([("2020-01-01", "2020-06-10", "2020-06-15", "2020-12-31")])

    with pytest.raises(AssertionError, match="reach into a variant label's validation span"):
        assert_variant_folds_are_out_of_sample("cs", "primary")


def test_the_outer_span_holding_is_not_enough(geometries) -> None:
    """fx_pairs' assertion passes on this geometry. That is the point of the issue.

    train_start >= primary's and val_end <= primary's both hold, and the variant is
    still scored on sessions the fold's features were fit on.
    """
    geometries["primary"] = _splits([("2020-01-01", "2020-06-20", "2020-06-25", "2020-12-31")])
    geometries["variant"] = _splits([("2020-02-01", "2020-06-10", "2020-06-15", "2020-12-01")])

    p, v = geometries["primary"][0], geometries["variant"][0]
    assert v["train_start"] >= p["train_start"] and v["val_end"] <= p["val_end"]

    with pytest.raises(AssertionError):
        assert_variant_folds_are_out_of_sample("cs", "primary")


def test_a_clean_geometry_returns_the_gap_per_fold(geometries) -> None:
    geometries["primary"] = _splits(
        [
            ("2020-01-01", "2020-06-20", "2020-06-25", "2020-12-31"),
            ("2019-01-01", "2019-06-20", "2019-06-25", "2019-12-31"),
        ]
    )
    geometries["variant"] = _splits(
        [
            ("2020-01-01", "2020-06-24", "2020-06-30", "2020-12-31"),
            ("2019-01-01", "2019-06-24", "2019-06-30", "2019-12-31"),
        ]
    )

    rows = assert_variant_folds_are_out_of_sample("cs", "primary")
    assert [r["fold"] for r in rows] == [0, 1]
    assert rows[0]["gap"] == pd.Timedelta(days=10)


def test_the_comparison_is_at_full_resolution_not_by_date(geometries) -> None:
    """nasdaq100_microstructure fold 0, which reads as a violation at date granularity.

    The bars are minutes: the fit closes 15:22 and the variant's validation opens
    15:38, both on 2020-12-29. Truncating to a date reports a leak that is not there.
    """
    geometries["primary"] = _splits(
        [("2020-01-02 09:32", "2020-12-29 15:22", "2020-12-29 15:40", "2021-12-31 15:58")]
    )
    geometries["variant"] = _splits(
        [("2020-01-02 09:32", "2020-12-29 15:20", "2020-12-29 15:38", "2021-12-31 15:58")]
    )

    rows = assert_variant_folds_are_out_of_sample("cs", "primary")
    assert rows[0]["gap"] == pd.Timedelta(minutes=16)

    truncated = [
        {
            k: (pd.Timestamp(v).normalize() if isinstance(v, pd.Timestamp) else v)
            for k, v in s.items()
        }
        for s in geometries["variant"]
    ]
    geometries["variant"] = truncated
    geometries["primary"] = [
        {
            k: (pd.Timestamp(v).normalize() if isinstance(v, pd.Timestamp) else v)
            for k, v in s.items()
        }
        for s in geometries["primary"]
    ]
    with pytest.raises(AssertionError):
        assert_variant_folds_are_out_of_sample("cs", "primary")


def test_a_variant_with_no_label_parquet_is_skipped(geometries) -> None:
    """The producers skip it too, so the check must not invent a failure."""
    geometries["primary"] = _splits([("2020-01-01", "2020-06-20", "2020-06-25", "2020-12-31")])
    assert assert_variant_folds_are_out_of_sample("cs", "primary") == []


def test_a_variant_fold_with_no_counterpart_is_a_violation(geometries) -> None:
    """A fold id the artifact does not carry is read as missing, not as safe."""
    geometries["primary"] = _splits([("2020-01-01", "2020-06-20", "2020-06-25", "2020-12-31")])
    geometries["variant"] = _splits(
        [
            ("2020-01-01", "2020-06-24", "2020-06-30", "2020-12-31"),
            ("2019-01-01", "2019-06-24", "2019-06-30", "2019-12-31"),
        ]
    )
    with pytest.raises(AssertionError, match="no counterpart"):
        assert_variant_folds_are_out_of_sample("cs", "primary")


def test_a_missing_primary_geometry_raises_rather_than_passing(geometries) -> None:
    with pytest.raises(ValueError, match="No folds derivable"):
        assert_variant_folds_are_out_of_sample("cs", "primary")


def _tiny_case_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, splits: list[dict]):
    """A case study with one primary label, one variant, and a fold-tagged artifact.

    Small enough to be read at a glance and complete enough that
    ``load_modeling_dataset`` runs the same path a model notebook does.
    """
    import polars as pl

    import utils.modeling as modeling

    case_dir = tmp_path / "cs"
    (case_dir / "config").mkdir(parents=True)
    (case_dir / "features").mkdir()
    (case_dir / "labels").mkdir()
    (case_dir / "config" / "setup.yaml").write_text(
        "labels:\n  primary: primary\n  buffer: 1D\n  variants: [variant]\n"
        "  variant_buffers:\n    variant: 1D\n"
    )

    days = pl.datetime_range(
        datetime(2020, 1, 1), datetime(2020, 12, 31), "1d", eager=True
    ).dt.date()
    frame = {"timestamp": days, "symbol": ["AAA"] * len(days)}
    pl.DataFrame({**frame, "primary": [0.1] * len(days)}).write_parquet(
        case_dir / "labels" / "primary.parquet"
    )
    pl.DataFrame({**frame, "variant": [0.1] * len(days)}).write_parquet(
        case_dir / "labels" / "variant.parquet"
    )
    pl.DataFrame({**frame, "feature": [1.0] * len(days)}).write_parquet(
        case_dir / "features" / "financial.parquet"
    )
    pl.DataFrame({**frame, "fold": [0] * len(days), "fitted": [2.0] * len(days)}).write_parquet(
        case_dir / "features" / "model_based.parquet"
    )

    monkeypatch.setattr(modeling, "get_case_study_dir", lambda _case_id: case_dir)
    monkeypatch.setattr(modeling, "load_feature_spec", lambda *_args: {})
    monkeypatch.setattr(modeling, "load_label_spec", lambda *_args: {})
    monkeypatch.setattr(
        modeling,
        "resolve_storage_path",
        lambda _case_id, _spec, fallback: case_dir / fallback,
    )
    monkeypatch.setattr(modeling, "resolve_label_buffer", lambda *_args: "1D")
    monkeypatch.setattr(modeling, "resolve_label_horizon", lambda *_args: "1D")
    monkeypatch.setattr(modeling, "generate_cv_splits", lambda *_a, **_k: splits)
    monkeypatch.setattr(modeling, "make_wf_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cv_window,
        "_load_setup_yaml",
        lambda _cs: {"labels": {"primary": "primary", "variants": ["variant"]}},
    )
    monkeypatch.setattr(
        cv_window,
        "temporal_artifact_fold_boundaries",
        lambda case_study, primary_label, _artifact_path: cv_window.modeling_fold_boundaries(
            case_study, primary_label
        ),
    )
    return modeling


# The whole fold: the artifact's fold 0 covers it, so coverage has nothing to say.
_WHOLE_YEAR = _splits([("2020-01-01", "2020-06-20", "2020-06-25", "2020-12-31")])


def test_loading_a_variant_whose_validation_opens_inside_the_fit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard had no production call site, so nothing ran it on a real load.

    ``validate_temporal_fold_coverage`` passes here - the artifact covers every date
    in the window - which is why coverage was never going to catch this.
    """
    modeling = _tiny_case_study(tmp_path, monkeypatch, _WHOLE_YEAR)
    store = {
        "primary": _splits([("2020-01-01", "2020-06-20", "2020-06-25", "2020-12-31")]),
        "variant": _splits([("2020-01-01", "2020-06-10", "2020-06-15", "2020-12-31")]),
    }
    monkeypatch.setattr(cv_window, "_derive_modeling_splits", lambda _cs, lab: store.get(lab))

    with pytest.raises(AssertionError, match="reach into a variant label's validation span"):
        modeling.load_modeling_dataset("cs", "variant")


def test_loading_a_variant_whose_validation_opens_after_the_fit_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary_splits = _splits([("2020-01-01", "2020-06-20", "2020-06-25", "2020-12-31")])
    variant_splits = _splits([("2020-01-01", "2020-06-24", "2020-06-30", "2020-12-31")])
    modeling = _tiny_case_study(tmp_path, monkeypatch, variant_splits)
    store = {
        "primary": primary_splits,
        "variant": variant_splits,
    }
    monkeypatch.setattr(cv_window, "_derive_modeling_splits", lambda _cs, lab: store.get(lab))

    mds = modeling.load_modeling_dataset("cs", "variant")
    assert mds.label_col == "variant"
    # The check hangs off the fold-tagged artifact, so a fixture that loaded no
    # artifact would pass this file vacuously.
    assert mds.temporal_by_fold is not None
    assert mds.splits == variant_splits
    assert mds.temporal_artifact_splits == cv_window.modeling_fold_boundaries("cs", "primary")
    requested = [
        {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in primary_splits[0].items()
        }
    ]
    require_fold_scoped_temporal_compatibility(requested, mds.temporal_artifact_splits)


def test_loading_preserves_intraday_temporal_artifact_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modeling = _tiny_case_study(tmp_path, monkeypatch, _WHOLE_YEAR)
    artifact_splits = _splits(
        [
            (
                "2020-01-02 09:32",
                "2020-06-20 15:22",
                "2020-06-25 09:31",
                "2020-12-31 15:58",
            )
        ]
    )
    monkeypatch.setattr(
        cv_window,
        "_derive_modeling_splits",
        lambda _cs, label: artifact_splits if label == "primary" else None,
    )

    mds = modeling.load_modeling_dataset("cs", "primary")
    requested = [
        {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in artifact_splits[0].items()
        }
    ]

    assert mds.temporal_artifact_splits == artifact_splits
    require_fold_scoped_temporal_compatibility(requested, mds.temporal_artifact_splits)
    changed = [{**requested[0], "train_end": "2020-06-20T15:23:00"}]
    with pytest.raises(ValueError, match="incompatible with fold-scoped temporal features"):
        require_fold_scoped_temporal_compatibility(changed, mds.temporal_artifact_splits)


def test_loading_the_primary_label_does_not_check_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The artifact is built on the primary's own geometry, so there is nothing to check.

    The variant geometry left in the store leaks, and loading the primary still
    succeeds: the check is scoped to the label being loaded rather than sweeping
    every configured label on every load.
    """
    modeling = _tiny_case_study(tmp_path, monkeypatch, _WHOLE_YEAR)
    store = {
        "primary": _splits([("2020-01-01", "2020-06-20", "2020-06-25", "2020-12-31")]),
        "variant": _splits([("2020-01-01", "2020-06-10", "2020-06-15", "2020-12-31")]),
    }
    monkeypatch.setattr(cv_window, "_derive_modeling_splits", lambda _cs, lab: store.get(lab))

    assert modeling.load_modeling_dataset("cs", "primary").label_col == "primary"


@pytest.mark.parametrize(
    ("case_study", "timeline_source", "include_outcome_horizon"),
    [
        ("cme_futures", "primary_label", False),
        ("crypto_perps_funding", "primary_label", True),
        ("etfs", "primary_label", True),
        ("fx_pairs", "primary_label", False),
        ("nasdaq100_microstructure", "primary_label", True),
        ("sp500_equity_option_analytics", "primary_label", True),
        ("sp500_options", "financial", False),
        ("us_equities_panel", "primary_label", True),
    ],
)
def test_legacy_temporal_fold_routes_match_locked_stage04_producers(
    case_study: str,
    timeline_source: str,
    include_outcome_horizon: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polars as pl

    import utils.artifact_specs as artifact_specs
    import utils.cv_splits as cv_splits

    case_dir = tmp_path / case_study
    (case_dir / "features").mkdir(parents=True)
    (case_dir / "labels").mkdir()
    pl.DataFrame({"timestamp": [datetime(2020, 1, 2)], "feature": [1.0]}).write_parquet(
        case_dir / "features" / "financial.parquet"
    )
    pl.DataFrame({"timestamp": [datetime(2020, 1, 1)], "primary": [0.1]}).write_parquet(
        case_dir / "labels" / "primary.parquet"
    )
    monkeypatch.setattr(
        cv_window,
        "_load_setup_yaml",
        lambda _case_study: {"labels": {"primary": "primary", "buffer": "5D"}},
    )
    monkeypatch.setattr(artifact_specs, "load_feature_spec", lambda *_args: {})
    monkeypatch.setattr(artifact_specs, "load_label_spec", lambda *_args: {})
    monkeypatch.setattr(artifact_specs, "resolve_label_buffer", lambda *_args: "5D")
    monkeypatch.setattr(artifact_specs, "resolve_label_horizon", lambda *_args: "2D")
    captured = {}
    producer_folds = _splits(
        [
            (
                "2020-01-02 09:32",
                "2020-06-20 15:22",
                "2020-06-25 09:31",
                "2020-12-31 15:58",
            )
        ]
    )

    def fake_generate(timeline, **kwargs):
        captured["timestamps"] = timeline.get_column("timestamp").to_list()
        captured["kwargs"] = kwargs
        return producer_folds

    monkeypatch.setattr(cv_splits, "generate_cv_splits", fake_generate)

    resolved = cv_window.temporal_artifact_fold_boundaries(
        case_study,
        "primary",
        case_dir / "features" / "model_based.parquet",
    )

    expected_timestamp = (
        datetime(2020, 1, 2) if timeline_source == "financial" else datetime(2020, 1, 1)
    )
    assert captured["timestamps"] == [expected_timestamp]
    assert ("outcome_horizon" in captured["kwargs"]) is include_outcome_horizon
    assert resolved == producer_folds


def test_temporal_artifact_sidecar_fold_geometry_precedes_legacy_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from case_studies.utils.artifact_digest import sidecar_path

    artifact = tmp_path / "model_based.parquet"
    artifact.write_bytes(b"artifact")
    folds = [
        {
            "fold": 0,
            "train_start": "2020-01-02T09:32:00",
            "train_end": "2020-06-20T15:22:00",
            "val_start": "2020-06-25T09:31:00",
            "val_end": "2020-12-31T15:58:00",
        }
    ]
    sidecar_path(artifact).write_text(json.dumps({"fold_geometry": folds}))
    monkeypatch.setitem(cv_window._LEGACY_TEMPORAL_FOLD_ROUTES, "unknown", None)

    assert cv_window.temporal_artifact_fold_boundaries("unknown", "primary", artifact) == folds


def _available_corpus_artifact(case_study: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        repo_root / "case_studies" / case_study / "features" / "model_based.parquet",
        repo_root.parent
        / "code"
        / "case_studies"
        / case_study
        / "features"
        / "model_based.parquet",
    )
    artifact = next((candidate for candidate in candidates if candidate.is_file()), None)
    if artifact is None:
        pytest.skip(f"canonical {case_study} model-based artifact is unavailable")
    return artifact


def _comparable_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


@pytest.mark.parametrize(
    "case_study",
    [
        "cme_futures",
        "crypto_perps_funding",
        "etfs",
        "fx_pairs",
        "nasdaq100_microstructure",
        "sp500_equity_option_analytics",
        "sp500_options",
        "us_equities_panel",
    ],
)
def test_corpus_temporal_fold_geometry_covers_resolved_validation_windows(
    case_study: str,
) -> None:
    import polars as pl

    artifact = _available_corpus_artifact(case_study)
    primary_label = cv_window.configured_labels(case_study)[0]
    resolved = cv_window.temporal_artifact_fold_boundaries(
        case_study,
        primary_label,
        artifact,
    )
    observed = (
        pl.scan_parquet(artifact)
        .select("fold", "timestamp")
        .unique()
        .collect()
        .partition_by("fold", as_dict=True)
    )

    for split in resolved:
        fold_rows = observed.get((split["fold"],))
        assert fold_rows is not None, f"{case_study} has no artifact rows for fold {split['fold']}"
        timestamps = [_comparable_timestamp(value) for value in fold_rows["timestamp"]]
        val_start = _comparable_timestamp(split["val_start"])
        val_end = _comparable_timestamp(split["val_end"])
        assert min(timestamps) <= val_start <= val_end <= max(timestamps)
        assert any(val_start <= timestamp <= val_end for timestamp in timestamps)


def test_sp500_options_corpus_uses_financial_timeline_geometry() -> None:
    import polars as pl
    import yaml

    from utils.artifact_specs import resolve_label_buffer, resolve_label_horizon
    from utils.cv_splits import generate_cv_splits

    case_study = "sp500_options"
    primary_label = "ret_to_expiry"
    artifact = _available_corpus_artifact(case_study)
    case_dir = artifact.parent.parent
    repo_root = Path(__file__).resolve().parents[1]
    setup = yaml.safe_load(
        (repo_root / "case_studies" / case_study / "config" / "setup.yaml").read_text()
    )
    resolved = cv_window.temporal_artifact_fold_boundaries(
        case_study,
        primary_label,
        artifact,
    )
    label_timeline = (
        pl.scan_parquet(case_dir / "labels" / f"{primary_label}.parquet")
        .select("timestamp")
        .unique()
        .collect()
    )
    label_derived = generate_cv_splits(
        label_timeline,
        case_study_id=case_study,
        label_buffer=resolve_label_buffer(case_study, primary_label, setup),
        outcome_horizon=resolve_label_horizon(case_study, primary_label, setup),
        date_col="timestamp",
    )
    observed_starts = (
        pl.scan_parquet(artifact)
        .group_by("fold")
        .agg(pl.col("timestamp").min().alias("timestamp"))
        .collect()
    )

    assert any(
        _comparable_timestamp(actual["train_start"])
        != _comparable_timestamp(from_label["train_start"])
        for actual, from_label in zip(resolved, label_derived, strict=True)
    )
    for split in resolved:
        observed_start = observed_starts.filter(pl.col("fold") == split["fold"]).item(
            0, "timestamp"
        )
        assert _comparable_timestamp(split["train_start"]) == _comparable_timestamp(observed_start)
