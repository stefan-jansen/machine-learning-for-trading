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
