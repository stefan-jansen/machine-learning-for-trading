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
