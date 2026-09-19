"""How a set of LightGBM gain importances becomes a ranked top-N feature list.

Polars only, so it imports nothing that pulls in lightgbm or torch. That is the
point rather than tidiness, and it is the same reason
`case_studies/utils/booster_paths.py` is standard library only: the two modules
that rank importances - `case_studies/utils/insight_chapter.py` and
`case_studies/utils/model_analysis.py` - import torch and lightgbm at module
scope for load-order races that have nothing to do with sorting a frame, and a
test of the ranking rule that had to reach through either of them failed at
collection on `test-unit`, which installs neither.
"""

from __future__ import annotations

import polars as pl


def top_features_by_gain(importance: pl.DataFrame, top_n: int) -> list[str]:
    """The `top_n` features the boosters actually split on, most important first.

    `importance` needs a `feature` column and an `importance_norm` column, one row
    per (feature, booster) the caller wants averaged. Both callers satisfy that and
    they mean different things by it, which is fine because the rule does not care:
    `insight_chapter.load_gbm_feature_importance` passes one selected configuration's
    folds, so the mean is over folds; `model_analysis.load_gbm_feature_importance`
    passes every gbm configuration in the case study, so the mean is over
    (configuration, fold). **Pooling changes how often the defects below bite**, and
    it does not change what the right answer is: a feature no booster in the pool
    split on still has no rank, and ties still have to break somewhere fixed.

    Two things this does that a plain ``sort(...).head(top_n)`` did not.

    **A feature whose mean gain is not a positive finite number is left out, whatever
    `top_n` is.** Gain is zero exactly when no tree split on the feature, so it has no
    rank to report and a consumer reads one: `12_gradient_boosting/12_case_study_insights` reports
    `linear_rank - gbm_rank` per common feature. On the single selected ETFs booster,
    25 of 71 features carry zero gain and the top-50 cut landed inside that block, so
    four never-split features were reported with a rank shift. Pooled over every gbm
    configuration the zero block nearly vanishes - measured 2026-09-19, one feature of
    88 on `nasdaq100_microstructure` and none at all on the other three case studies
    `model_analysis` serves - because a feature one booster ignored another split on.

    **The order is fully determined.** Polars' sort is not stable and every zero-gain
    feature ties, so repeated loads of the same booster returned different 50s: the
    symmetric difference was four features on sp500_options and six on ETFs across
    three loads in one process. Ties break on the feature name, so the cut is a
    property of the boosters rather than of the run.

    **`is_finite` is not belt and braces and the filter is wrong without it.** A caller
    normalises by each fold's own maximum gain, so a fold whose booster made no split at
    all divides zero by zero and hands this function `NaN` for every feature of that
    fold. `mean()` propagates it, polars answers `True` to `NaN > 0`, and it sorts NaN
    above every float - so a bare `> 0` filter keeps exactly the features this docstring
    promises to drop and ranks them first, and the tie-break then makes that alphabetical
    and reproducible. The caller should not hand over a dead fold, and
    `model_analysis.load_gbm_feature_importance` now drops one before normalising while
    `insight_chapter`'s refuses the whole run; this is what holds if a third caller does
    neither. Measured 2026-09-19: no (configuration, fold) in the four case studies the
    pooled loader serves has a zero maximum, across 331 groups, so nothing published today
    reaches it.
    """
    ranked = (
        importance.group_by("feature")
        .agg(pl.col("importance_norm").mean().alias("mean_importance"))
        .filter(pl.col("mean_importance").is_finite() & (pl.col("mean_importance") > 0))
        .sort(["mean_importance", "feature"], descending=[True, False])
    )
    return ranked.head(top_n)["feature"].to_list()
