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

from case_studies.utils.cv_window import IntradayFoldBoundaryError
from case_studies.utils.insight_chapter import (
    IncomparableFoldGeometryError,
    RegistrySelectionError,
    canonical_fold_ids,
    collect_checkpoint_fold_trajectories,
    collect_grid_per_cs,
    collect_multi_label_per_cs,
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


def _register_without_metrics(case_dir: Path, config_name: str, fold_ids: tuple[int, ...]) -> str:
    """A prediction set with fold metrics and no `prediction_metrics` row.

    `register_prediction_set` takes `metrics=None`, and the two tables are written by
    separate calls, so this is a state the registry reaches on its own - an interrupted
    run that got as far as its folds, or a set registered for its predictions alone.
    """
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
        case_dir=case_dir,
    )
    register_fold_metrics(
        "test",
        prediction_hash,
        {fold_id: {"ic": 0.05} for fold_id in fold_ids},
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


@pytest.fixture(autouse=True)
def _clear_canonical_fold_cache():
    """`canonical_fold_ids` is lru_cached on (case study, label).

    That key is stable for the life of a run and is not stable across tests: every test
    here declares a different grid for the same "test"/fwd_ret_5d pair, and the ones that
    never stub it cache a None. Cleared on the way in and on the way out, so the leak does
    not reach `test_insight_selection_reads_the_live_generation.py`, which uses the same
    pair and runs after this file.
    """
    canonical_fold_ids.cache_clear()
    yield
    canonical_fold_ids.cache_clear()


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
            raise IntradayFoldBoundaryError("fold boundary carries a time of day")
        return [{"fold": fold_id} for fold_id in fold_ids]

    monkeypatch.setattr("case_studies.utils.insight_chapter.modeling_fold_boundaries", _boundaries)


def test_a_subsampled_grid_is_selected_and_says_so(tmp_path, selects_from, monkeypatch) -> None:
    """The one live case: 12_dl_weekly scores 4 of us_equities_panel's 16 modelling folds.

    Comparability within the group is fine - all 30 candidates scored the same four - so
    the selection stands. What must not happen is the row passing as complete: the ids
    are canonical and the run declared the size of the subsample it chose, so a count
    comparison is self-referential and only the grid can answer it.

    Asserted on all three collectors. The consumer the columns exist for -
    `13_dl_time_series/12_case_study_insights`'s horizon census - reads them off
    `collect_multi_label_per_cs`, so pinning only `collect_rank1_per_cs` would leave the
    suite green while the notebook lost its exclusion.
    """
    _grid(monkeypatch, range(16))
    selects_from(tmp_path, [("nlinear", 0.02, STRIDE_FIVE), ("lstm_h64", 0.05, STRIDE_FIVE)])

    collected = {
        "rank1": collect_rank1_per_cs(["test"], "deep_learning"),
        "multi_label": collect_multi_label_per_cs(["test"], "deep_learning", [LABEL]),
        "grid": collect_grid_per_cs(["test"], "deep_learning"),
    }
    for name, frame in collected.items():
        assert not frame.is_empty(), name
        for row in frame.iter_rows(named=True):
            assert row["n_folds_scored"] == 4, name
            assert row["n_folds_canonical"] == 16, name
            assert row["covers_fold_grid"] is False, name


def test_a_full_grid_is_marked_covered(tmp_path, selects_from, monkeypatch) -> None:
    _grid(monkeypatch, (0, 5, 10, 15))
    selects_from(tmp_path, [("nlinear", 0.02, STRIDE_FIVE)])

    row = collect_rank1_per_cs(["test"], "deep_learning").row(0, named=True)

    assert row["n_folds_scored"] == 4
    assert row["n_folds_canonical"] == 4
    assert row["covers_fold_grid"] is True


def test_a_missing_label_surface_is_null_rather_than_complete(
    tmp_path, selects_from, monkeypatch
) -> None:
    """The reader bundle: `run_log/` on disk and no `labels/`.

    `case_studies/*/labels` is gitignored and the release bundle ships the registry alone,
    so `modeling_fold_boundaries` returns None outright rather than refusing a boundary.
    That is the second source of an underivable grid, and it has to read as "not measured"
    exactly like the intraday one - a reader whose cells all came back null must not see a
    census claiming every cell covered its grid.
    """
    monkeypatch.setattr(
        "case_studies.utils.insight_chapter.modeling_fold_boundaries",
        lambda _case_study, _label: None,
    )
    selects_from(tmp_path, [("nlinear", 0.02, STRIDE_FIVE)])

    row = collect_rank1_per_cs(["test"], "deep_learning").row(0, named=True)

    assert row["n_folds_scored"] == 4
    assert row["n_folds_canonical"] is None
    assert row["covers_fold_grid"] is None


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


def test_a_prediction_set_with_no_metrics_row_does_not_set_the_geometry(
    tmp_path, selects_from
) -> None:
    """The geometry standard comes from candidates that could be selected.

    `_raw_primary_candidates` builds its metrics frame through a join on
    `prediction_metrics` and its folds frame without one, so a set with fold rows and no
    metrics row reaches the resolver while being unrankable. Here it reports a
    full-length geometry that disagrees with the two real candidates', which before the
    restriction raised `IncomparableFoldGeometryError` and stopped the whole chapter over
    a row `select_rank1` could never have returned.

    Asserted on all three collectors. The first version of this fix restricted the frame
    inside `collect_rank1_per_cs` and left `collect_grid_per_cs` and
    `collect_multi_label_per_cs` resolving a geometry from the unrestricted one, so the
    selection succeeded and the chapter died one cell later at the horizon census -
    which reads as a problem with the horizon labels rather than with a stray
    registration. The restriction belongs in the shared builder for that reason.
    """
    selects_from(tmp_path, [("nlinear", 0.05, STRIDE_FIVE), ("lstm", 0.02, STRIDE_FIVE)])
    _register_without_metrics(tmp_path, "interrupted", (1, 6, 11, 16))

    collected = {
        "rank1": collect_rank1_per_cs(["test"], "deep_learning"),
        "multi_label": collect_multi_label_per_cs(["test"], "deep_learning", [LABEL]),
        "grid": collect_grid_per_cs(["test"], "deep_learning"),
    }
    # Reaching this line at all is most of the assertion: without the restriction each
    # of the three raises IncomparableFoldGeometryError on the stray geometry.
    for name, frame in collected.items():
        assert not frame.is_empty(), name
        assert "interrupted" not in frame["config_name"].to_list(), name
        assert set(frame["n_folds_scored"].to_list()) == {4}, name
    # rank1 and the horizon census return the selected row; the grid returns every
    # candidate, which is why the config assertions differ.
    assert collected["rank1"]["config_name"].to_list() == ["nlinear"]
    assert collected["multi_label"]["config_name"].to_list() == ["nlinear"]
    assert sorted(collected["grid"]["config_name"].to_list()) == ["lstm", "nlinear"]


def test_a_metric_less_checkpoint_does_not_set_the_trajectory_geometry(
    tmp_path, monkeypatch
) -> None:
    """The fourth resolver, and it queries the registry itself rather than taking the
    frame `_raw_primary_candidates` restricts.

    `collect_checkpoint_fold_trajectories` runs immediately after rank-one selection in
    `13_dl_time_series/12_case_study_insights`, so an unguarded geometry here stops the
    chapter one cell past the one the restriction fixed, saying the fold geometries
    disagree rather than that a registration is stray. Here checkpoint 10 has fold rows
    and no metrics row and reports a geometry the two real checkpoints do not share.
    """
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
    for checkpoint in (5, 15):
        prediction_hash = register_prediction_set(
            "test",
            training_hash,
            checkpoint_value=checkpoint,
            checkpoint_kind="epoch",
            split="validation",
            metrics={"ic_mean": 0.02, "ic_mean_daily": 0.02, "ic_n_days": 100},
            case_dir=tmp_path,
        )
        register_fold_metrics(
            "test",
            prediction_hash,
            {fold_id: {"ic": 0.02} for fold_id in STRIDE_FIVE},
            case_dir=tmp_path,
        )
    ghost = register_prediction_set(
        "test",
        training_hash,
        checkpoint_value=10,
        checkpoint_kind="epoch",
        split="validation",
        case_dir=tmp_path,
    )
    register_fold_metrics(
        "test", ghost, {fold_id: {"ic": 0.9} for fold_id in (1, 6, 11, 16)}, case_dir=tmp_path
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

    trajectories = collect_checkpoint_fold_trajectories(rank1)

    assert sorted(trajectories["checkpoint_value"].unique().to_list()) == [5, 15]
