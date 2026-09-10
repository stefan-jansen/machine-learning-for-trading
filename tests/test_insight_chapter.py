"""Behavioral tests for cross-case-study insight diagnostics."""

from __future__ import annotations

import json

import polars as pl
import pytest

from case_studies.utils import insight_chapter
from case_studies.utils.conformal import (
    holdout_conformal_embargo_steps,
    walk_forward_conformal_coverage,
)


def test_compare_ic_uses_only_shared_intraday_timestamps() -> None:
    left = pl.DataFrame(
        {
            "date": ["2026-01-02 09:30", "2026-01-02 10:00", "2026-01-03 09:30"],
            "ic": [0.9, 0.2, 0.4],
        }
    ).with_columns(pl.col("date").str.to_datetime().cast(pl.Datetime("ms")))
    right = pl.DataFrame(
        {
            "date": ["2026-01-02 10:00", "2026-01-03 09:30", "2026-01-03 10:00"],
            "ic": [0.1, 0.3, -0.9],
        }
    ).with_columns(pl.col("date").str.to_datetime().cast(pl.Datetime("us")))

    result = insight_chapter.compare_ic_on_shared_timestamps(left, right)

    assert result == {
        "left_ic": pytest.approx(0.3),
        "right_ic": pytest.approx(0.2),
        "n_timestamps": 2,
    }


def _write_prediction_panel(
    prediction_dir, residuals: dict[str, list[float]], *, fold_ids: list[int] | None = None
):
    """One row per (day, symbol), with `y_true - y_score` exactly as `residuals` says."""
    lengths = {len(values) for values in residuals.values()}
    assert len(lengths) == 1
    steps = lengths.pop()
    days = [f"2020-{1 + step // 28:02d}-{1 + step % 28:02d}" for step in range(steps)]
    scores = [float(step) / steps for step in range(steps)]
    folds = fold_ids or [1 if step < steps // 2 else 0 for step in range(steps)]
    pl.DataFrame(
        {
            "timestamp": [day for day in days for _ in residuals],
            "symbol": [symbol for _ in days for symbol in residuals],
            "y_true": [
                scores[step] + residuals[symbol][step]
                for step in range(steps)
                for symbol in residuals
            ],
            "y_score": [scores[step] for step in range(steps) for _ in residuals],
            "fold_id": [folds[step] for step in range(steps) for _ in residuals],
        }
    ).write_parquet(prediction_dir / "predictions.parquet")


def _selected(prediction_hash: str, spec: dict, *, label: str = "fwd_ret_5d") -> dict:
    return {
        "case_study": "probe",
        "family": "gbm",
        "config_name": "probe-config",
        "label": label,
        "prediction_hash": prediction_hash,
        "spec_json": json.dumps(spec),
    }


_TWO_FOLD_SPEC = {"computation": {"expected_prediction_keys": {"n_folds": 2}}}


def test_selected_prediction_conformal_coverage_measures_the_sizing_widths(
    tmp_path, monkeypatch
) -> None:
    """The chapter reports the estimator `conformal_weighted` allocates with.

    Asserted against `walk_forward_conformal_coverage` on the same artifact, because what this
    pins is that the two are one measurement: the chapter used to run a second estimator -
    pooled across symbols, fixed on the earliest fold, unembargoed - and print its coverage as
    the strategy's.
    """
    case_dir = tmp_path / "case_studies" / "probe"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-a"
    prediction_dir.mkdir(parents=True)
    _write_prediction_panel(prediction_dir, {"CALM": [0.1] * 80, "WILD": [10.0] * 80})
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    result = insight_chapter.conformal_coverage_for_selected_prediction(
        _selected("prediction-a", _TWO_FOLD_SPEC), levels=(0.80,), embargo_steps=1
    )
    expected = walk_forward_conformal_coverage(
        pl.read_parquet(prediction_dir / "predictions.parquet"), levels=(0.80,), embargo_steps=1
    )

    assert result.height == 1
    assert result.row(0, named=True) == {
        "case_study": "probe",
        "family": "gbm",
        "config_name": "probe-config",
        "prediction_hash": "prediction-a",
        **expected[0],
    }


def test_selected_prediction_conformal_coverage_defaults_to_the_reviewed_horizon(
    tmp_path, monkeypatch
) -> None:
    """The row's own label decides the embargo, so the figure and the widths cannot disagree
    about how far a residual reaches. `label` is required of the selected row for that reason.
    """
    case_dir = tmp_path / "case_studies" / "etfs"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-a"
    prediction_dir.mkdir(parents=True)
    _write_prediction_panel(prediction_dir, {"CALM": [0.1] * 80, "WILD": [10.0] * 80})
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    row = _selected("prediction-a", _TWO_FOLD_SPEC)
    row["case_study"] = "etfs"
    defaulted = insight_chapter.conformal_coverage_for_selected_prediction(row, levels=(0.80,))
    explicit = insight_chapter.conformal_coverage_for_selected_prediction(
        row, levels=(0.80,), embargo_steps=holdout_conformal_embargo_steps("etfs", "fwd_ret_5d")
    )
    assert defaulted.equals(explicit)

    # A row with no `label` falls back to the training spec, which names the same one.
    # `us_equities_panel/15_model_analysis` builds exactly that row: it attaches the label to
    # the frame this returns rather than to the dict it passes in.
    unlabelled = {key: value for key, value in row.items() if key != "label"}
    unlabelled["spec_json"] = json.dumps({**_TWO_FOLD_SPEC, "label": "fwd_ret_5d"})
    assert insight_chapter.conformal_coverage_for_selected_prediction(
        unlabelled, levels=(0.80,)
    ).equals(explicit)

    with pytest.raises(insight_chapter.RegistrySelectionError, match="names a label"):
        insight_chapter.conformal_coverage_for_selected_prediction(
            unlabelled | {"spec_json": json.dumps(_TWO_FOLD_SPEC)}, levels=(0.80,)
        )


def test_selected_prediction_conformal_coverage_rejects_all_null_declared_fold(
    tmp_path, monkeypatch
) -> None:
    case_dir = tmp_path / "case_studies" / "probe"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-a"
    prediction_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": ["2019-01-02"] * 40 + ["2020-01-02"] * 40,
            "symbol": ["AAA"] * 80,
            "y_true": [0.1] * 40 + [None] * 40,
            "y_score": [0.0] * 40 + [None] * 40,
            "fold_id": [0] * 40 + [1] * 40,
        }
    ).write_parquet(prediction_dir / "predictions.parquet")
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    with pytest.raises(insight_chapter.RegistrySelectionError, match=r"observed \[0\]"):
        insight_chapter.conformal_coverage_for_selected_prediction(
            _selected("prediction-a", _TWO_FOLD_SPEC), levels=(0.80,), embargo_steps=1
        )


def test_selected_prediction_conformal_coverage_rejects_non_finite_rows(
    tmp_path, monkeypatch
) -> None:
    case_dir = tmp_path / "case_studies" / "probe"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-a"
    prediction_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": ["2019-01-02"] * 40 + ["2020-01-02"] * 40,
            "symbol": ["AAA"] * 80,
            "y_true": [0.1] * 80,
            "y_score": [0.0] * 79 + [float("inf")],
            "fold_id": [0] * 40 + [1] * 40,
        }
    ).write_parquet(prediction_dir / "predictions.parquet")
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    with pytest.raises(insight_chapter.RegistrySelectionError, match="non-finite y_score"):
        insight_chapter.conformal_coverage_for_selected_prediction(
            _selected("prediction-a", _TWO_FOLD_SPEC), levels=(0.80,), embargo_steps=1
        )


def test_selected_prediction_conformal_coverage_reads_the_legacy_spec_shape(
    tmp_path, monkeypatch
) -> None:
    """Two spec shapes are live, and the older one is still written.

    `build_training_spec` puts `n_folds` at the top level and emits no `computation`
    key at all; `run_dl_cv` still uses it and LEGACY_IDENTITY_VERSION is still
    supported. Reading only the identity-v3 location answered 0 for every such row, so
    a row declaring five folds raised "requires at least two declared folds" and
    12_case_study_insights, which tolerates only "fewer than 30 rows", aborted the
    chapter rather than degrading.
    """
    case_dir = tmp_path / "case_studies" / "probe"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-legacy"
    prediction_dir.mkdir(parents=True)
    _write_prediction_panel(prediction_dir, {"CALM": [0.1] * 80, "WILD": [10.0] * 80})
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    legacy = insight_chapter.conformal_coverage_for_selected_prediction(
        _selected("prediction-legacy", {"family": "deep_learning", "n_folds": 2}),
        levels=(0.80,),
        embargo_steps=1,
    )

    assert legacy["nominal_level"].to_list() == [0.80]


def test_selected_prediction_conformal_coverage_still_rejects_a_single_fold(
    tmp_path, monkeypatch
) -> None:
    """The fallback must not turn the real one-fold refusal into a pass."""
    case_dir = tmp_path / "case_studies" / "probe"
    (case_dir / "run_log" / "predictions" / "prediction-one").mkdir(parents=True)
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    with pytest.raises(insight_chapter.RegistrySelectionError, match="at least two declared folds"):
        insight_chapter.conformal_coverage_for_selected_prediction(
            _selected("prediction-one", {"family": "deep_learning", "n_folds": 1}),
            levels=(0.80,),
            embargo_steps=1,
        )


def _write_booster(booster_dir, *, fold: int) -> None:
    """A real two-feature LightGBM booster, so the loader parses what the pipeline writes."""
    import lightgbm as lgb
    import numpy as np

    booster_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(fold)
    x = rng.normal(size=(200, 2))
    y = 3.0 * x[:, 0] + 0.1 * x[:, 1] + rng.normal(scale=0.01, size=200)
    model = lgb.LGBMRegressor(n_estimators=5, num_leaves=4, min_child_samples=5, verbose=-1)
    model.fit(x, y, feature_name=["strong", "weak"])
    model.booster_.save_model(str(booster_dir / f"fold_{fold}.txt"))


def test_gbm_feature_importance_reads_the_boosters_the_training_stage_writes(
    tmp_path, monkeypatch
) -> None:
    """Boosters live under the run's own `models` directory.

    The loader used to look only beside it and one level up, so every case study
    returned an empty frame and the sections built on it published nothing while
    every check still passed.
    """
    case_dir = tmp_path / "case_studies" / "probe"
    training_dir = case_dir / "run_log" / "training" / "hash-a"
    _write_booster(training_dir / "models" / "boosters", fold=0)
    _write_booster(training_dir / "models" / "boosters", fold=1)
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    result = insight_chapter.load_gbm_feature_importance("probe", "hash-a", "probe-config", top_n=2)

    assert sorted(result["fold_id"].unique().to_list()) == [0, 1]
    ordered = (
        result.group_by("feature")
        .agg(pl.col("importance_norm").mean())
        .sort("importance_norm", descending=True)["feature"]
        .to_list()
    )
    assert ordered == ["strong", "weak"]


def test_gbm_feature_importance_still_reads_the_older_layouts(tmp_path, monkeypatch) -> None:
    """Run logs written before boosters moved keep working."""
    case_dir = tmp_path / "case_studies" / "probe"
    _write_booster(case_dir / "run_log" / "training" / "hash-b" / "boosters", fold=0)
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    result = insight_chapter.load_gbm_feature_importance("probe", "hash-b", "probe-config", top_n=2)

    assert result.height > 0


def test_gbm_feature_importance_measures_only_the_selected_checkpoint(
    tmp_path, monkeypatch
) -> None:
    """Importance belongs to the model the selection chose, not to every saved round.

    Training saves all boosting rounds; a configuration is selected at one checkpoint
    along that trajectory. The booster here is fitted so that the first rounds split on
    `early` and later rounds split on `late`, which makes the two readings disagree.
    """
    import lightgbm as lgb
    import numpy as np

    rng = np.random.default_rng(0)
    early = rng.normal(size=400)
    late = rng.normal(size=400)
    # `early` alone explains the signal; `late` only explains what is left after the
    # first rounds have fitted it, so it enters the booster late.
    y = 5.0 * early + 0.05 * late
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.5, num_leaves=4, min_child_samples=5, verbose=-1
    )
    model.fit(np.column_stack([early, late]), y, feature_name=["early", "late"])
    booster_dir = tmp_path / "case_studies" / "probe" / "run_log" / "training" / "h" / "models"
    booster_dir = booster_dir / "boosters"
    booster_dir.mkdir(parents=True)
    model.booster_.save_model(str(booster_dir / "fold_0.txt"))
    monkeypatch.setattr(
        insight_chapter, "get_case_study_dir", lambda _cs: tmp_path / "case_studies" / "probe"
    )

    def gain(num_iteration):
        frame = insight_chapter.load_gbm_feature_importance(
            "probe", "h", "probe-config", top_n=2, num_iteration=num_iteration
        )
        return dict(frame.group_by("feature").agg(pl.col("importance").mean()).iter_rows())

    first_three = gain(3)
    everything = gain(None)

    assert first_three["late"] < everything["late"]
    assert insight_chapter.load_gbm_feature_importance(
        "probe", "h", "probe-config", top_n=2, num_iteration=10_000
    ).equals(insight_chapter.load_gbm_feature_importance("probe", "h", "probe-config", top_n=2))
