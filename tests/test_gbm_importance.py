"""The rule that turns LightGBM gain importances into a charted top-N list.

`case_studies/utils/gbm_importance.top_features_by_gain` is what four case-study
`model_analysis` notebooks and `12_gradient_boosting/12_case_study_insights` cut their
feature figures with. Its two guarantees - a never-split feature is not charted, and
the cut does not move between loads of the same boosters - are the ones a plain
`sort(...).head(top_n)` did not give, and both were reached by a rendered figure
before they were reached by a test.

This file imports the rule alone, which is polars and nothing else, so it runs in the
required `test-unit` job. Reaching it through `insight_chapter` or `model_analysis`
would drag torch or lightgbm in at module scope and the file would have to be
quarantined into `test-unit-image`, which is where the rule's coverage used to live.
"""

import polars as pl

from case_studies.utils.gbm_importance import top_features_by_gain


def test_a_never_split_feature_is_not_charted_whatever_top_n_is() -> None:
    """Gain is zero exactly when no tree split on the feature, so it has no rank.

    Chapter 12's rank-shift cell reports `linear_rank - gbm_rank` per common feature,
    and the old cut handed it features the booster never used: 25 of the 71 features on
    the selected ETFs booster carry zero gain and the top-50 cut landed inside that
    block.
    """
    importance = pl.DataFrame(
        {
            "feature": ["alpha", "beta", "gamma", "delta", "epsilon"],
            "importance_norm": [1.0, 0.5, 0.0, 0.0, 0.0],
        }
    )

    assert top_features_by_gain(importance, top_n=4) == ["alpha", "beta"]


def test_the_answer_does_not_depend_on_the_row_order_it_arrived_in() -> None:
    """Polars' sort is not stable, so every tied block was cut at an arbitrary point.

    Three loads of one booster in one process returned feature sets differing by six
    on ETFs and four on sp500_options.
    """
    importance = pl.DataFrame(
        {
            "feature": ["alpha", "beta", "gamma", "delta", "epsilon"],
            "importance_norm": [1.0, 0.5, 0.0, 0.0, 0.0],
        }
    )
    expected = top_features_by_gain(importance, top_n=4)

    for ordering in (["feature"], ["importance_norm"], ["importance_norm", "feature"]):
        for descending in (True, False):
            shuffled = importance.sort(ordering, descending=descending)
            assert top_features_by_gain(shuffled, top_n=4) == expected


def test_a_nonzero_tie_breaks_on_the_feature_name() -> None:
    importance = pl.DataFrame(
        {
            "feature": ["zulu", "alpha", "mike"],
            "importance_norm": [0.4, 0.4, 0.9],
        }
    )

    assert top_features_by_gain(importance, top_n=2) == ["mike", "alpha"]


def test_a_tie_block_straddling_the_cut_is_resolved_the_same_way_every_time() -> None:
    """The case the published figures were actually exposed to.

    `top_n` lands inside a block of equal gains, so which members are charted is
    decided entirely by the tie-break. Without one the cut is a property of the run.
    """
    importance = pl.DataFrame(
        {
            "feature": ["echo", "alpha", "delta", "bravo", "charlie"],
            "importance_norm": [0.2, 0.2, 0.2, 0.2, 0.2],
        }
    )

    assert top_features_by_gain(importance, top_n=2) == ["alpha", "bravo"]
    assert top_features_by_gain(importance.reverse(), top_n=2) == ["alpha", "bravo"]


def test_the_mean_is_taken_across_every_row_a_feature_has() -> None:
    """Which is folds for one configuration, and (configuration, fold) when pooled.

    `model_analysis.load_gbm_feature_importance` pools every gbm configuration in the
    case study while `insight_chapter`'s reads one, and the rule serves both because it
    averages whatever rows it is handed. A feature one booster never split on survives
    when another did, which is why the zero block nearly vanishes under pooling: one
    feature of 88 on nasdaq100_microstructure, none on the other three.
    """
    importance = pl.DataFrame(
        {
            "feature": ["alpha", "alpha", "beta", "beta", "gamma", "gamma"],
            "importance_norm": [1.0, 0.0, 0.4, 0.4, 0.0, 0.0],
        }
    )

    # alpha averages 0.5 and beta 0.4, so alpha leads; gamma averages zero in every row
    # it has and is dropped rather than ranked last.
    assert top_features_by_gain(importance, top_n=3) == ["alpha", "beta"]


def test_asking_for_more_than_there_are_returns_what_there_is() -> None:
    importance = pl.DataFrame({"feature": ["alpha", "beta"], "importance_norm": [0.3, 0.1]})

    assert top_features_by_gain(importance, top_n=50) == ["alpha", "beta"]


def test_a_frame_where_nothing_was_split_on_charts_nothing() -> None:
    """Reachable from a booster that made no split at all, and the caller's
    `is_in([])` then renders an empty figure rather than an arbitrary one."""
    importance = pl.DataFrame({"feature": ["alpha", "beta"], "importance_norm": [0.0, 0.0]})

    assert top_features_by_gain(importance, top_n=5) == []
