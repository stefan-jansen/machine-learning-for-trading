"""Relabeling stored fold ids onto the chronological numbering, without refitting."""

from __future__ import annotations

import copy
import json
from pathlib import PurePosixPath

import polars as pl
import pytest

from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.registry.fold_renumbering import (
    FoldRenumberRefusal,
    cohort_member_digest,
    derive_fold_permutation,
    plan_fold_renumbering,
    remap_json_document,
    remap_relative_path,
    remap_training_spec,
    spec_fold_reference_paths,
)
from case_studies.utils.registry.specs import training_hash_from_spec

NEWEST_FIRST = [
    {
        "fold": 0,
        "train_start": "2005-03-31 00:00:00",
        "train_end": "2014-12-31 00:00:00",
        "val_start": "2015-02-27 00:00:00",
        "val_end": "2015-12-31 00:00:00",
    },
    {
        "fold": 1,
        "train_start": "2004-04-30 00:00:00",
        "train_end": "2014-02-28 00:00:00",
        "val_start": "2014-03-31 00:00:00",
        "val_end": "2015-01-30 00:00:00",
    },
    {
        "fold": 2,
        "train_start": "2003-05-30 00:00:00",
        "train_end": "2013-03-28 00:00:00",
        "val_start": "2013-04-30 00:00:00",
        "val_end": "2014-02-28 00:00:00",
    },
]


def _eligible_keys(folds: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [f"S{index}" for index, _ in enumerate(folds)],
            "timestamp": [fold["val_start"][:10] for fold in folds],
            "fold": [fold["fold"] for fold in folds],
        }
    ).with_columns(pl.col("timestamp").str.to_date())


def _spec(folds: list[dict]) -> dict:
    keys = _eligible_keys(folds)
    return {
        "identity_version": 3,
        "resolved_spec_schema": "ml4t.resolved-spec/v1",
        "family": "linear",
        "label": "fwd_ret_1m",
        "seed": 42,
        "execution_tier": "canonical",
        "config_name": "ridge",
        "computation": {
            "cv": {
                "folds": copy.deepcopy(folds),
                "identity": value_digest(pl.DataFrame(folds)),
                "request": {"source": "case_study_default"},
            },
            "expected_prediction_keys": {
                "digest": value_digest(keys, ("symbol", "timestamp", "fold")),
                "n_rows": keys.height,
                "n_folds": len(folds),
            },
            "model": {
                "class": "Ridge",
                "effective_params_by_fold": {
                    str(fold["fold"]): {"alpha": float(fold["fold"])} for fold in folds
                },
            },
            "numerics": {"precision": "float64", "seed": 42},
        },
    }


def test_permutation_reverses_a_newest_first_fold_list() -> None:
    assert derive_fold_permutation(NEWEST_FIRST) == {0: 2, 1: 1, 2: 0}


def test_permutation_refuses_a_fold_id_set_with_a_gap() -> None:
    # A preview that ran folds 0 and 2 cannot be ranked into contiguous ids: the window it
    # skipped still exists, so position in the surviving list is not the new id.
    partial = [NEWEST_FIRST[0], {**NEWEST_FIRST[2], "fold": 2}]
    with pytest.raises(FoldRenumberRefusal, match="not 0..n-1"):
        derive_fold_permutation(partial)


def test_permutation_refuses_two_folds_sharing_a_window() -> None:
    duplicated = [NEWEST_FIRST[0], {**NEWEST_FIRST[0], "fold": 1}]
    with pytest.raises(FoldRenumberRefusal, match="share the window"):
        derive_fold_permutation(duplicated)


def test_relative_path_keeps_the_padding_each_producer_wrote() -> None:
    permutation = {0: 9, 8: 1, 9: 0}
    assert remap_relative_path(PurePosixPath("models/fold_0.joblib"), permutation) == PurePosixPath(
        "models/fold_9.joblib"
    )
    # `fold_08` and `fold_3` both occur inside one registry, so normalizing the width here
    # would leave a reader globbing for the shape its producer wrote.
    assert remap_relative_path(
        PurePosixPath("models/cfg/fold_08/epoch_0050.pt"), permutation
    ) == PurePosixPath("models/cfg/fold_01/epoch_0050.pt")


def test_relative_path_renames_the_candidate_directory_named_by_the_training_hash() -> None:
    assert remap_relative_path(
        PurePosixPath("models/aaaaaaaaaaaa/fold_0/epoch_0050.pt"),
        {0: 1},
        {"aaaaaaaaaaaa": "bbbbbbbbbbbb"},
    ) == PurePosixPath("models/bbbbbbbbbbbb/fold_1/epoch_0050.pt")


def test_json_remap_moves_fold_ids_and_leaves_epoch_keys_alone() -> None:
    document = {
        "fold_id": 0,
        "checkpoint_metrics": {"0": {"ic": 0.1}, "25": {"ic": 0.2}},
        "files": {"fold_0.joblib": "digest-a", "fold_1.joblib": "digest-b"},
        "nested": [{"fold": 1, "config": "aaaaaaaaaaaa"}],
    }
    remapped = remap_json_document(
        document, {0: 1, 1: 0}, identity_renames={"aaaaaaaaaaaa": "bbbbbbbbbbbb"}
    )
    assert remapped["fold_id"] == 1
    # Keyed by checkpoint, not by fold: a rule that moved every small-integer key would
    # silently rewrite these.
    assert remapped["checkpoint_metrics"] == document["checkpoint_metrics"]
    assert remapped["files"] == {"fold_1.joblib": "digest-a", "fold_0.joblib": "digest-b"}
    assert remapped["nested"] == [{"fold": 0, "config": "bbbbbbbbbbbb"}]


def test_json_remap_refuses_a_fold_id_outside_the_permutation() -> None:
    with pytest.raises(FoldRenumberRefusal, match="outside the permutation"):
        remap_json_document({"fold": 7}, {0: 1, 1: 0})


def test_spec_scan_finds_the_fold_keyed_dict_a_name_search_misses() -> None:
    spec = _spec(NEWEST_FIRST)
    spec["computation"]["task"] = {
        "imbalance": {"effective_class_weights_by_fold": {"0": [1.0], "1": [1.0], "2": [1.0]}}
    }
    paths = spec_fold_reference_paths(spec, [0, 1, 2])
    # Neither of these carries "fold" in a value; both are found by the key-set rule.
    assert "$.computation.task.imbalance.effective_class_weights_by_fold" in paths
    assert "$.computation.model.effective_params_by_fold" in paths


def test_remap_moves_the_identity_and_every_value_derived_from_the_folds() -> None:
    spec = _spec(NEWEST_FIRST)
    permutation = derive_fold_permutation(NEWEST_FIRST)
    target = remap_training_spec(spec, permutation, eligible_keys=_eligible_keys(NEWEST_FIRST))
    computation = target["computation"]
    assert [fold["fold"] for fold in computation["cv"]["folds"]] == [0, 1, 2]
    # Fold 0 is now the earliest window rather than the most recent one.
    assert computation["cv"]["folds"][0]["val_end"] == "2014-02-28 00:00:00"
    assert computation["cv"]["identity"] == value_digest(pl.DataFrame(computation["cv"]["folds"]))
    assert computation["cv"]["identity"] != spec["computation"]["cv"]["identity"]
    assert (
        computation["expected_prediction_keys"]["digest"]
        != spec["computation"]["expected_prediction_keys"]["digest"]
    )
    # The parameters fitted on the 2013 window are still the parameters of that window.
    assert computation["model"]["effective_params_by_fold"]["0"] == {"alpha": 2.0}
    assert training_hash_from_spec(target) != training_hash_from_spec(spec)


def test_remap_refuses_a_spec_whose_stored_cv_identity_does_not_reproduce() -> None:
    spec = _spec(NEWEST_FIRST)
    spec["computation"]["cv"]["identity"] = "0" * 16
    with pytest.raises(FoldRenumberRefusal, match="cv.identity does not reproduce"):
        remap_training_spec(
            spec, derive_fold_permutation(NEWEST_FIRST), eligible_keys=_eligible_keys(NEWEST_FIRST)
        )


def test_remap_refuses_a_prediction_frame_that_is_not_the_one_the_digest_covers() -> None:
    spec = _spec(NEWEST_FIRST)
    wrong = _eligible_keys(NEWEST_FIRST).with_columns(pl.lit("Z").alias("symbol"))
    with pytest.raises(FoldRenumberRefusal, match="does not reproduce from the prediction frame"):
        remap_training_spec(spec, derive_fold_permutation(NEWEST_FIRST), eligible_keys=wrong)


def test_remap_refuses_when_anything_outside_the_fold_fields_moved() -> None:
    spec = _spec(NEWEST_FIRST)
    permutation = derive_fold_permutation(NEWEST_FIRST)

    original = remap_training_spec.__globals__["_strip_fold_bearing"]

    def _drop_seed(candidate):
        reduced = original(candidate)
        # Stand in for a spec whose non-fold content differs between source and target: the
        # comparison has to fail on it rather than migrate a different computation.
        reduced["computation"]["numerics"]["seed"] = id(candidate) % 7
        return reduced

    remap_training_spec.__globals__["_strip_fold_bearing"] = _drop_seed
    try:
        with pytest.raises(FoldRenumberRefusal, match="differs outside its fold-bearing fields"):
            remap_training_spec(spec, permutation, eligible_keys=_eligible_keys(NEWEST_FIRST))
    finally:
        remap_training_spec.__globals__["_strip_fold_bearing"] = original


def test_cohort_member_digest_matches_the_one_the_registry_stores() -> None:
    # Duplicated rather than imported, so this is what stops the copy drifting.
    from case_studies.utils.uncertainty import cohort_member_digest as stored

    members = ["cc", "aa", "bb", "aa"]
    assert cohort_member_digest(members) == stored(members)


def test_plan_refuses_a_registry_whose_stored_hash_does_not_reproduce(tmp_path) -> None:
    import sqlite3

    from case_studies.utils.registry.store import _open_registry

    case_dir = tmp_path / "case"
    (case_dir / "run_log").mkdir(parents=True)
    _open_registry(case_dir).close()
    spec = _spec(NEWEST_FIRST)
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute(
            "INSERT INTO training_runs (training_hash, family, label, config_name, spec_json, "
            "created_at, identity_version, execution_tier) VALUES (?,?,?,?,?,?,?,?)",
            (
                "deadbeefdead",
                spec["family"],
                spec["label"],
                spec["config_name"],
                json.dumps(spec),
                "2026-01-01T00:00:00+00:00",
                3,
                "canonical",
            ),
        )
    plan = plan_fold_renumbering(case_dir)
    assert plan.remaps == []
    assert any("does not reproduce" in refusal for refusal in plan.refusals)
