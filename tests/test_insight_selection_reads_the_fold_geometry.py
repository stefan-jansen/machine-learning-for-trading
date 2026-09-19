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

import polars as pl
import pytest

from case_studies.utils.insight_chapter import (
    IncomparableFoldGeometryError,
    RegistrySelectionError,
    collect_checkpoint_fold_trajectories,
    collect_grid_per_cs,
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
    folds = pl.DataFrame({"prediction_hash": ["a", "a"], "fold_id": [0, 5]})
    with pytest.raises(RegistrySelectionError, match="n_folds is not declared"):
        resolve_expected_fold_ids(folds, 0)


def test_the_grid_refuses_a_disagreement_rather_than_shortening_itself(
    tmp_path, selects_from
) -> None:
    """The whole-grid collector skips a case study it cannot rank, which hides the reason.

    Nothing reaching the declared count is a case study without a complete candidate and
    is a reason to leave it out. Two disagreeing full-length geometries are a registry
    that cannot be ranked, and a table that silently loses a case study over that says
    nothing a reader could act on.
    """
    selects_from(tmp_path, [("nlinear", 0.02, STRIDE_FIVE), ("lstm_h64", 0.05, (0, 1, 2, 3))])

    with pytest.raises(IncomparableFoldGeometryError) as raised:
        collect_grid_per_cs(["test"], "deep_learning")

    # The grid loops over nine case studies, so the error has to say which one.
    assert str(raised.value).startswith(f"test/deep_learning/{LABEL}: ")
    assert "disagree on which 4 folds" in str(raised.value)


def test_the_grid_still_skips_a_case_study_with_no_complete_candidate(
    tmp_path, selects_from
) -> None:
    selects_from(tmp_path, [("nlinear", 0.02, (0, 5)), ("lstm_h64", 0.05, (0, 5, 10))])

    assert collect_grid_per_cs(["test"], "deep_learning").is_empty()


def test_a_checkpoint_geometry_failure_names_the_case_study_and_run(tmp_path, monkeypatch) -> None:
    """`collect_checkpoint_fold_trajectories` loops over nine case studies."""
    training_hash = register_training_run(
        "test",
        {
            "family": "deep_learning",
            "label": LABEL,
            "config_name": "nlinear",
            "params": {},
            "seed": 42,
            "n_folds": 4,
        },
        case_dir=tmp_path,
    )
    for checkpoint, fold_ids in ((5, STRIDE_FIVE), (10, (0, 1, 2, 3))):
        prediction_hash = register_prediction_set(
            "test",
            training_hash,
            checkpoint_value=checkpoint,
            checkpoint_kind="epoch",
            split="validation",
            metrics={"ic_mean": 0.02, "ic_mean_daily": 0.02, "ic_std": 0.01},
            case_dir=tmp_path,
        )
        register_fold_metrics(
            "test",
            prediction_hash,
            {fold_id: {"ic": 0.02} for fold_id in fold_ids},
            case_dir=tmp_path,
        )
    monkeypatch.setattr("case_studies.utils.insight_chapter.get_case_study_dir", lambda _: tmp_path)
    rank1 = pl.DataFrame(
        [
            {
                "case_study": "test",
                "short_name": "Test",
                "family": "deep_learning",
                "config_name": "nlinear",
                "label": LABEL,
                "training_hash": training_hash,
                "spec_json": json.dumps({"n_folds": 4}),
            }
        ]
    )

    with pytest.raises(RegistrySelectionError, match=f"test/{training_hash}: "):
        collect_checkpoint_fold_trajectories(rank1)


def _grid(monkeypatch, fold_ids):
    """Make the case study's canonical modelling grid `fold_ids`, or unavailable."""

    def _boundaries(_case_study, _label):
        if fold_ids is None:
            raise ValueError("fold boundary carries a time of day")
        return [{"fold": fold_id} for fold_id in fold_ids]

    monkeypatch.setattr("case_studies.utils.insight_chapter.modeling_fold_boundaries", _boundaries)


def test_a_subsampled_grid_is_selected_and_says_so(tmp_path, selects_from, monkeypatch) -> None:
    """The one live case: 12_dl_weekly scores 4 of us_equities_panel's 16 modelling folds.

    Comparability within the group is fine - all 30 candidates scored the same four - so
    the selection stands. What must not happen is the row passing as complete: the ids
    are canonical and the run declared the size of the subsample it chose, so a count
    comparison is self-referential and only the grid can answer it.
    """
    _grid(monkeypatch, range(16))
    selects_from(tmp_path, [("nlinear", 0.02, STRIDE_FIVE), ("lstm_h64", 0.05, STRIDE_FIVE)])

    row = collect_rank1_per_cs(["test"], "deep_learning").row(0, named=True)

    assert row["n_folds_scored"] == 4
    assert row["n_folds_canonical"] == 16
    assert row["covers_fold_grid"] is False


def test_a_full_grid_is_marked_covered(tmp_path, selects_from, monkeypatch) -> None:
    _grid(monkeypatch, (0, 5, 10, 15))
    selects_from(tmp_path, [("nlinear", 0.02, STRIDE_FIVE)])

    row = collect_rank1_per_cs(["test"], "deep_learning").row(0, named=True)

    assert row["n_folds_scored"] == 4
    assert row["n_folds_canonical"] == 4
    assert row["covers_fold_grid"] is True


def test_an_underivable_grid_is_null_rather_than_false(tmp_path, selects_from, monkeypatch) -> None:
    """The intraday case studies, whose fold boundaries carry a time of day.

    "Not measured" and "measured and short" are different answers and a consumer has to
    be able to tell them apart, so the columns are null rather than False.
    """
    _grid(monkeypatch, None)
    selects_from(tmp_path, [("nlinear", 0.02, STRIDE_FIVE)])

    row = collect_rank1_per_cs(["test"], "deep_learning").row(0, named=True)

    assert row["n_folds_scored"] == 4
    assert row["n_folds_canonical"] is None
    assert row["covers_fold_grid"] is None
