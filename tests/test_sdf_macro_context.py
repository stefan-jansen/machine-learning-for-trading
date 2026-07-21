"""Contracts for point-in-time SDF macro conditioning."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from case_studies.utils.latent_factors.macro_context import load_configured_macro_context
from case_studies.utils.latent_factors.panel import align_macro_to_dates

EXPECTED_ETF_SERIES = [
    "dgs1",
    "dgs2",
    "dgs3",
    "dgs5",
    "dgs7",
    "dgs10",
    "dgs20",
    "dgs30",
    "vixcls",
    "YIELD_CURVE_SLOPE",
    "YIELD_CURVE_5_10",
]


def _config() -> dict:
    return {
        "source": "alfred_initial_release",
        "policy": "alfred_initial_release_close_lagged",
        "version": "v1",
        "series": ["dgs1", "vixcls"],
        "availability_lag_days": 1,
        "alignment": "backward_asof",
    }


def test_etf_sdf_uses_exact_safe_market_state_series() -> None:
    setup = yaml.safe_load(Path("case_studies/etfs/config/setup.yaml").read_text())
    macro_config = setup["modeling"]["latent_factors"]["macro_context"]

    assert macro_config["series"] == EXPECTED_ETF_SERIES
    assert macro_config["source"] == "alfred_initial_release"
    assert macro_config["availability_lag_days"] == 1
    assert macro_config["alignment"] == "backward_asof"


def test_configured_macro_context_lags_and_hashes_selected_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = pl.DataFrame(
        {
            "timestamp": [date(2024, 1, 2), date(2024, 1, 3)],
            "dgs1": [4.1, 4.2],
            "vixcls": [13.0, 14.0],
            "revised_series": [1.0, 99.0],
        }
    )

    def fake_load_macro(*, series: list[str]) -> pl.DataFrame:
        return raw.select(["timestamp", *series])

    monkeypatch.setattr(
        "case_studies.utils.latent_factors.macro_context.load_macro_initial_release",
        fake_load_macro,
    )
    panel, identity = load_configured_macro_context(_config())

    assert panel.columns == ["timestamp", "dgs1", "vixcls"]
    assert panel["timestamp"].to_list() == [date(2024, 1, 3), date(2024, 1, 4)]
    assert identity["series"] == ["dgs1", "vixcls"]
    assert identity["coverage_start"] == "2024-01-03"
    assert identity["input_digest"].startswith("sha256:")

    changed = raw.with_columns(pl.col("revised_series") * 10)
    monkeypatch.setattr(
        "case_studies.utils.latent_factors.macro_context.load_macro_initial_release",
        lambda *, series: changed.select(["timestamp", *series]),
    )
    _, unchanged_identity = load_configured_macro_context(_config())
    assert unchanged_identity == identity

    changed_selected = raw.with_columns(pl.col("dgs1") + 0.01)
    monkeypatch.setattr(
        "case_studies.utils.latent_factors.macro_context.load_macro_initial_release",
        lambda *, series: changed_selected.select(["timestamp", *series]),
    )
    _, changed_identity = load_configured_macro_context(_config())
    assert changed_identity["input_digest"] != identity["input_digest"]


def test_backward_alignment_keeps_context_non_null_without_future_fill() -> None:
    macro = pl.DataFrame(
        {
            "timestamp": [date(2024, 1, 3), date(2024, 1, 5)],
            "dgs1": [4.1, 4.2],
            "vixcls": [13.0, 14.0],
        }
    )
    values, names = align_macro_to_dates(
        macro,
        np.array(["2024-01-03", "2024-01-04", "2024-01-05"], dtype="datetime64[ns]"),
    )

    assert names == ["dgs1", "vixcls"]
    assert np.isfinite(values).all()
    assert values[:, 0].tolist() == pytest.approx([4.1, 4.1, 4.2])

    with pytest.raises(ValueError, match="unavailable on or before"):
        align_macro_to_dates(macro, [date(2024, 1, 2), date(2024, 1, 3)])


def test_macro_context_enters_hash_and_cache_matching() -> None:
    from case_studies.utils.latent_factors.cv import (
        _apply_latent_factor_runtime_spec,
        _macro_context_matches,
    )
    from case_studies.utils.registry import training_hash_from_spec

    base = {
        "family": "latent_factors",
        "config_name": "sdf",
        "label": "fwd_ret_21d",
        "seed": 42,
        "params": {"n_factors": 5},
    }
    identity = {**_config(), "input_digest": "sha256:abc"}
    resolved = _apply_latent_factor_runtime_spec(
        spec=base,
        n_factors=5,
        n_epochs=50,
        model_kwargs={},
        fold_extras=[],
        feature_names=["value"],
        splits=[],
        task_type="regression",
        class_values=None,
        eval_label_col=None,
        input_digest="input-a",
        macro_digest=identity["input_digest"],
        runtime_spec={"device": "cpu", "seed": 42},
        macro_context_spec=identity,
    )

    assert resolved["macro_context"] == identity
    assert training_hash_from_spec(resolved) != training_hash_from_spec(base)
    assert _macro_context_matches(resolved, identity)
    assert not _macro_context_matches(base, identity)
    assert not _macro_context_matches(
        resolved,
        {**identity, "availability_lag_days": 2},
    )
