"""A ``columns=`` projection narrows the load and must not narrow the recorded identity.

``load_modeling_dataset`` derives ``feature_names`` from the joined frame, and that list is
hashed into every registered run twice: directly as ``computation.feature_names`` and again
inside ``computation.input_data_spec``, both of which ``training_hash_from_spec`` covers. So
a caller that asks for seven of seventy-four columns to save memory would, without this,
re-key every run its case study has registered - for a change that reads the same artifacts,
retains the same values in the columns it keeps, and computes the same estimate.

``build_modeling_input_lineage`` already states the rule for the same event in the other
direction, about why ``feature_dtype`` is written only when it is not the default: a key that
moves the fingerprint of a case study whose declaration did not change invalidates every run
registered against it.

The control is the last test. Without it, the fingerprint comparison would pass just as well
against an implementation that projects nothing at all, which is the outcome the parameter
exists to avoid.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
import yaml

from tests.test_column_projection_happens_in_the_scan import (  # noqa: F401
    CASE,
    FEATURES,
    TEMPORAL,
    seeded_case_study,
)

PROJECTION = ["beta", "latent_2"]


def _lineage(mds) -> dict:  # type: ignore[no-untyped-def]
    return mds.input_lineage


def test_a_projected_load_records_the_panels_feature_list_not_its_own(
    seeded_case_study: Path,  # noqa: F811
) -> None:
    from utils.modeling import load_modeling_dataset

    full = load_modeling_dataset(CASE, "fwd_ret_1d")
    projected = load_modeling_dataset(CASE, "fwd_ret_1d", columns=PROJECTION)

    # Order, not just membership: the payload is json.dumps(..., sort_keys=True), which sorts
    # dict keys and leaves list order alone, so a reordering moves the fingerprint too.
    assert projected.panel_feature_names == full.feature_names


def test_a_projection_does_not_move_the_input_lineage_fingerprint(
    seeded_case_study: Path,  # noqa: F811
) -> None:
    from utils.modeling import load_modeling_dataset

    full = load_modeling_dataset(CASE, "fwd_ret_1d")
    projected = load_modeling_dataset(CASE, "fwd_ret_1d", columns=PROJECTION)

    assert _lineage(projected)["feature_names"] == _lineage(full)["feature_names"]
    assert _lineage(projected)["fingerprint"] == _lineage(full)["fingerprint"]


def test_feature_names_still_describes_the_frame_the_caller_was_handed(
    seeded_case_study: Path,  # noqa: F811
) -> None:
    """The two lists are different things, and the projected load is where that shows."""
    from utils.modeling import load_modeling_dataset

    projected = load_modeling_dataset(CASE, "fwd_ret_1d", columns=PROJECTION)

    assert set(projected.feature_names) == set(PROJECTION)
    assert set(projected.panel_feature_names) == set(FEATURES) | set(TEMPORAL)
    assert set(projected.feature_names) < set(projected.panel_feature_names)


def test_an_unprojected_load_records_exactly_what_it_always_did(
    seeded_case_study: Path,  # noqa: F811
) -> None:
    """The new field must be inert when nothing is projected, or it moves every fingerprint."""
    from utils.modeling import load_modeling_dataset

    full = load_modeling_dataset(CASE, "fwd_ret_1d")

    assert full.panel_feature_names == full.feature_names
    assert _lineage(full)["feature_names"] == full.feature_names


def test_the_fingerprint_comparison_can_fail(seeded_case_study: Path) -> None:  # noqa: F811
    """Control: the lineage fingerprint does move when the recorded feature list moves.

    Without this, the equality above is satisfied by any implementation whose fingerprint is
    insensitive to ``feature_names``, including one that never projects.
    """
    from utils.modeling import load_modeling_dataset

    full = load_modeling_dataset(CASE, "fwd_ret_1d")
    projected = load_modeling_dataset(CASE, "fwd_ret_1d", columns=PROJECTION)

    # What the projected load WOULD have registered if the projection reached the identity.
    projected.panel_feature_names = list(projected.feature_names)
    projected._input_lineage = None  # noqa: SLF001 - re-derive under the narrowed list

    assert projected.feature_names != full.feature_names
    assert _lineage(projected)["fingerprint"] != _lineage(full)["fingerprint"]


COLLIDING = "shared_name"


def _seed_colliding(tmp_path: Path) -> None:
    case_dir = tmp_path / CASE
    (case_dir / "config").mkdir(parents=True, exist_ok=True)
    (case_dir / "features").mkdir(exist_ok=True)
    (case_dir / "labels").mkdir(exist_ok=True)
    (case_dir / "config" / "setup.yaml").write_text(
        yaml.safe_dump(
            {
                "strategy_id": CASE,
                "labels": {"primary": "fwd_ret_1d", "buffer": "1D"},
                "evaluation": {
                    "n_splits": 2,
                    "train_size": "1Y",
                    "val_size": "6M",
                    "calendar": "NYSE",
                    "periods_per_year": 252,
                },
            }
        )
    )
    days = pl.date_range(date(2018, 1, 1), date(2021, 6, 30), interval="1d", eager=True)
    rows: dict[str, list] = {"timestamp": [], "symbol": []}
    for symbol in ("AAA", "BBB", "CCC"):
        rows["timestamp"].extend(days.to_list())
        rows["symbol"].extend([symbol] * len(days))
    frame = pl.DataFrame(rows)
    frame.with_columns([pl.lit(1.0).alias("beta"), pl.lit(2.0).alias(COLLIDING)]).write_parquet(
        case_dir / "features" / "financial.parquet"
    )
    frame.with_columns([pl.lit(3.0).alias("latent_2"), pl.lit(4.0).alias(COLLIDING)]).write_parquet(
        case_dir / "features" / "model_based.parquet"
    )
    frame.with_columns(pl.lit(0.01).alias("fwd_ret_1d")).write_parquet(
        case_dir / "labels" / "fwd_ret_1d.parquet"
    )


def test_a_name_carried_by_both_artifacts_is_refused_rather_than_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polars suffixes the second one ``_t``, and its position is not recoverable here.

    No case study has such a name - checked across all eight on 2026-09-11 - so a projected
    load refuses rather than registering an identity that differs from the unprojected load's.
    An unprojected load is unaffected, because it reads ``feature_names`` itself.
    """
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    _seed_colliding(tmp_path)

    import utils.modeling as modeling
    from case_studies.utils import cv_window

    monkeypatch.setattr(modeling, "load_feature_spec", lambda *_args: {})
    monkeypatch.setattr(modeling, "load_label_spec", lambda *_args: {})
    monkeypatch.setattr(
        modeling,
        "resolve_storage_path",
        lambda _case_id, _spec, fallback: tmp_path / CASE / fallback,
    )
    cv_window._fold_splits.cache_clear()
    cv_window._load_setup_yaml.cache_clear()
    try:
        modeling.load_modeling_dataset(CASE, "fwd_ret_1d")  # unprojected: still fine

        with pytest.raises(ValueError, match=f"both carry.*{COLLIDING}"):
            modeling.load_modeling_dataset(CASE, "fwd_ret_1d", columns=["beta"])
    finally:
        cv_window._fold_splits.cache_clear()
        cv_window._load_setup_yaml.cache_clear()
