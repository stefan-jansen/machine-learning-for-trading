"""Fold ids come from the candidates, because no spec shape declares them.

``select_rank1`` only ranks candidates that cover every declared fold, which is right: a
configuration scored on fewer folds saw an easier sample and must not rank against a
complete one. What was wrong is where the fold *ids* came from. Every caller passed
``range(n_folds)``, and no live spec shape carries a fold-id list - identity v3 nests a
count under ``computation.expected_prediction_keys``, the v2 shape carries ``n_folds`` at
the top level, and neither says which folds were used.

``us_equities_panel/deep_learning/fwd_ret_5d`` declares four folds and all 30 of its
prediction sets report folds 0, 5, 10 and 15, a stride-5 sample of a wider geometry. Every
candidate failed the identity comparison, the eligible list came back empty, and
``13_dl_time_series/12_case_study_insights`` stopped executing with "no finite candidate
covers every declared fold" - on a notebook that had executed cleanly before that case
study had any deep-learning rows to select from.

The gate keeps its job here: the ids are read off the candidates that reach the declared
count, so a short candidate is still refused and two disagreeing full-length geometries
are refused as incomparable rather than resolved by majority.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.insight_chapter import (
    RegistrySelectionError,
    collect_rank1_per_cs,
    resolve_expected_fold_ids,
)
from case_studies.utils.registry import (
    register_fold_metrics,
    register_prediction_set,
    register_training_run,
)

LABEL = "fwd_ret_5d"
STRIDE_FIVE = (0, 5, 10, 15)


def _register(case_dir: Path, config_name: str, daily_ic: float, fold_ids: tuple[int, ...]) -> str:
    """One candidate reporting exactly `fold_ids`, with a spec declaring four folds."""
    training_hash = register_training_run(
        "test",
        {
            "family": "deep_learning",
            "label": LABEL,
            "config_name": config_name,
            "params": {},
            "seed": 42,
            "n_folds": 4,
        },
        case_dir=case_dir,
    )
    prediction_hash = register_prediction_set(
        "test",
        training_hash,
        checkpoint_value=5,
        checkpoint_kind="epoch",
        split="validation",
        metrics={
            "ic_mean": daily_ic,
            "ic_mean_daily": daily_ic,
            "ic_std": 0.01,
            "ic_n_days": 100,
            "ic_se_hac": 0.02,
            "ic_ci_lo": daily_ic - 0.04,
            "ic_ci_hi": daily_ic + 0.04,
            "ic_t_hac": daily_ic / 0.02,
            "ic_p_hac": 0.01,
            "ic_hac_lag": 1,
        },
        case_dir=case_dir,
    )
    register_fold_metrics(
        "test",
        prediction_hash,
        {fold_id: {"ic": daily_ic} for fold_id in fold_ids},
        case_dir=case_dir,
    )
    return prediction_hash


def _publish(case_dir: Path, members: list[str]) -> None:
    """One live population, so the supersession filter keeps every member."""
    snapshot = {
        "schema_version": 1,
        "name": "test-deep_learning-validation-v1",
        "member_kind": "prediction",
        "members": members,
        "supersedes": None,
    }
    with sqlite3.connect(case_dir / "run_log" / "registry.db") as db:
        db.execute(
            "INSERT INTO official_populations "
            "(population_hash, name, member_kind, snapshot_json, supersedes_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "pop000000000a",
                snapshot["name"],
                "prediction",
                json.dumps(snapshot),
                None,
                "2026-01-01T00:00:00+00:00",
            ),
        )


@pytest.fixture
def selects_from(monkeypatch):
    """`collect_rank1_per_cs(["test"], "deep_learning")` against one temp registry."""

    def _build(case_dir: Path, candidates: list[tuple[str, float, tuple[int, ...]]]):
        members = [
            _register(case_dir, config_name, daily_ic, fold_ids)
            for config_name, daily_ic, fold_ids in candidates
        ]
        _publish(case_dir, members)
        monkeypatch.setattr(
            "case_studies.utils.insight_chapter.get_case_study_dir", lambda _: case_dir
        )
        monkeypatch.setattr(
            "case_studies.utils.insight_chapter._resolve_label", lambda *_a, **_k: LABEL
        )
        return members

    return _build


def test_a_stride_five_geometry_selects_instead_of_raising(tmp_path, selects_from) -> None:
    """The us_equities_panel case: four declared folds numbered 0, 5, 10, 15."""
    _, winner = selects_from(
        tmp_path, [("nlinear", 0.02, STRIDE_FIVE), ("lstm_h64", 0.05, STRIDE_FIVE)]
    )

    selected = collect_rank1_per_cs(["test"], "deep_learning")

    assert selected.height == 1
    assert selected["prediction_hash"].to_list() == [winner]
    assert selected["config_name"].to_list() == ["lstm_h64"]


def test_a_short_candidate_is_still_refused(tmp_path, selects_from) -> None:
    """Scoring three of four folds is an easier sample, and it wins on IC if admitted."""
    complete, _short = selects_from(
        tmp_path, [("nlinear", 0.02, STRIDE_FIVE), ("lstm_h64", 0.40, (0, 5, 10))]
    )

    selected = collect_rank1_per_cs(["test"], "deep_learning")

    assert selected["prediction_hash"].to_list() == [complete]
    assert selected["ic_mean_daily"].to_list() == [0.02]


def test_two_full_length_geometries_are_refused_as_incomparable(tmp_path, selects_from) -> None:
    """Four folds each, different four. Neither defines the standard for the other."""
    selects_from(tmp_path, [("nlinear", 0.02, STRIDE_FIVE), ("lstm_h64", 0.05, (0, 1, 2, 3))])

    with pytest.raises(RegistrySelectionError) as raised:
        collect_rank1_per_cs(["test"], "deep_learning")

    message = str(raised.value)
    assert "disagree on which 4 folds" in message
    assert "(0, 1, 2, 3)" in message
    assert "(0, 5, 10, 15)" in message


def test_a_contiguous_geometry_is_unchanged(tmp_path, selects_from) -> None:
    _, winner = selects_from(
        tmp_path, [("nlinear", 0.02, (0, 1, 2, 3)), ("lstm_h64", 0.05, (0, 1, 2, 3))]
    )

    selected = collect_rank1_per_cs(["test"], "deep_learning")

    assert selected["prediction_hash"].to_list() == [winner]


def test_no_candidate_reaching_the_declared_count_is_named_as_such(tmp_path, selects_from) -> None:
    """Every candidate short: that is the failure the gate exists for, and it says so."""
    selects_from(tmp_path, [("nlinear", 0.02, (0, 5)), ("lstm_h64", 0.05, (0, 5, 10))])

    with pytest.raises(RegistrySelectionError, match="no candidate reports all 4 declared folds"):
        collect_rank1_per_cs(["test"], "deep_learning")


def test_the_resolver_refuses_an_undeclared_fold_count() -> None:
    import polars as pl

    folds = pl.DataFrame({"prediction_hash": ["a", "a"], "fold_id": [0, 5]})
    with pytest.raises(RegistrySelectionError, match="n_folds is not declared"):
        resolve_expected_fold_ids(folds, 0)
