"""Tests for case_studies/utils/registry/metrics.py — prediction metric aggregation.

The critical contract this pins is the *classification IC rule*: when
``task_type='classification'``, IC is computed against the continuous
return column named by ``eval_col``, never against the binary label.
Computing IC against the binary label degenerates to ``2·(AUC − 0.5)``
and is not a valid Spearman rank correlation against returns — the April
classification IC backfill was needed precisely because this was wrong.

These tests lock in:

- Regression path: IC is cross-sectional rank correlation of y_score
  vs y_true (continuous); RMSE / MAE are computed on valid pairs.
- Classification path: IC uses ``eval_col`` (continuous return), and
  AUC / log_loss / accuracy use the binary ``y_true_col``.
- Classification IC on y_score + y_ret equals the regression IC on the
  same y_score + y_ret — i.e., the classification branch does not
  silently fall back to using the binary label for IC.
- Missing ``eval_col`` (or a column that isn't on the DataFrame) raises
  loudly rather than silently collapsing to AUC-disguised-as-IC.
- Headline aggregation: ``ic_mean`` = mean across folds, ``ic_t`` =
  Newey-West-free pooled t, ``pct_positive`` = fraction of folds with
  IC > 0, ``n_folds`` = count, ``task_type`` = 'classification' for classification.

All fixtures are hermetic — no real data, no setup.yaml.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from case_studies.utils.registry.metrics import compute_prediction_fold_metrics

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def regression_predictions() -> pl.DataFrame:
    """2 folds × 10 dates × 10 entities with y_score ≈ y_true (high IC)."""
    rng = np.random.default_rng(0)
    rows = []
    for fold in (0, 1):
        for d in range(10):
            for e in range(10):
                y_true = float(rng.normal())
                y_score = 0.8 * y_true + 0.2 * float(rng.normal())
                rows.append(
                    {
                        "timestamp": f"2024-{fold + 1:02d}-{d + 1:02d}",
                        "symbol": f"S{e}",
                        "fold_id": fold,
                        "y_true": y_true,
                        "y_score": y_score,
                    }
                )
    return pl.DataFrame(rows).with_columns(pl.col("timestamp").str.to_date())


@pytest.fixture(scope="module")
def classification_predictions(regression_predictions) -> pl.DataFrame:
    """Classification variant: y_true is the sign of the continuous return.

    - ``y_ret`` preserves the continuous return (the eval_col target)
    - ``y_true`` is the binary label (1 if return > 0)
    - ``y_score`` is a probability-style score from a logistic squash of
      the original continuous score, so it is still monotone in the
      continuous return
    """
    return regression_predictions.rename({"y_score": "y_score_cont"}).with_columns(
        y_ret=pl.col("y_true"),
        y_true=pl.when(pl.col("y_true") > 0).then(1).otherwise(0).cast(pl.Int8),
        y_score=1.0 / (1.0 + (-pl.col("y_score_cont")).exp()),
    )


# -----------------------------------------------------------------------------
# Regression path
# -----------------------------------------------------------------------------


def test_regression_computes_rmse_mae_and_ic(regression_predictions) -> None:
    headline, folds = compute_prediction_fold_metrics(
        regression_predictions, task_type="regression"
    )
    assert set(folds.keys()) == {0, 1}
    for fm in folds.values():
        assert "rmse" in fm and "mae" in fm
        assert "ic" in fm
        # y_score ≈ 0.8 * y_true so IC should be high
        assert fm["ic"] > 0.5
        # RMSE / MAE on standard normals with 0.2σ noise should be tiny-ish
        assert fm["rmse"] >= 0
        assert fm["mae"] >= 0


def test_regression_headline_task_type_is_regression(regression_predictions) -> None:
    headline, _ = compute_prediction_fold_metrics(regression_predictions, task_type="regression")
    assert headline["task_type"] == "regression"


def test_regression_headline_ic_mean_equals_fold_ic_mean(regression_predictions) -> None:
    headline, folds = compute_prediction_fold_metrics(
        regression_predictions, task_type="regression"
    )
    expected = float(np.mean([fm["ic"] for fm in folds.values()]))
    assert math.isclose(headline["ic_mean"], expected, rel_tol=1e-12)


def test_regression_pct_positive_matches_fraction_of_positive_ic_folds(
    regression_predictions,
) -> None:
    headline, folds = compute_prediction_fold_metrics(
        regression_predictions, task_type="regression"
    )
    expected = float(np.mean([fm["ic"] > 0 for fm in folds.values()]))
    assert headline["pct_positive"] == expected


def test_regression_headline_ic_t_is_mean_over_stderr(regression_predictions) -> None:
    headline, folds = compute_prediction_fold_metrics(
        regression_predictions, task_type="regression"
    )
    fold_ics = np.array([fm["ic"] for fm in folds.values()])
    expected_t = float(np.mean(fold_ics) / (np.std(fold_ics) / np.sqrt(len(fold_ics))))
    assert math.isclose(headline["ic_t"], expected_t, rel_tol=1e-12)


def test_regression_n_folds_matches_unique_fold_ids(regression_predictions) -> None:
    headline, _ = compute_prediction_fold_metrics(regression_predictions, task_type="regression")
    assert headline["n_folds"] == 2


# -----------------------------------------------------------------------------
# Classification path — the load-bearing contract
# -----------------------------------------------------------------------------


def test_classification_without_eval_col_raises_value_error(classification_predictions) -> None:
    """The defensive check that saved us from re-introducing the IC-on-binary bug."""
    with pytest.raises(ValueError, match="eval_col"):
        compute_prediction_fold_metrics(
            classification_predictions, task_type="classification", class_values=[0, 1]
        )


def test_classification_missing_eval_col_raises_key_error(classification_predictions) -> None:
    with pytest.raises(KeyError, match="does_not_exist"):
        compute_prediction_fold_metrics(
            classification_predictions,
            task_type="classification",
            eval_col="does_not_exist",
            class_values=[0, 1],
        )


def test_classification_ic_is_computed_vs_continuous_return(
    regression_predictions, classification_predictions
) -> None:
    """The classification IC on (y_score_cont, y_ret) must equal the regression
    IC on the same (y_score, y_true) — proving classification did not silently
    fall back to IC-vs-binary.

    We compare against a regression run on the ORIGINAL continuous pair, to
    establish what the IC should be. Then we compare the classification IC
    on (y_score_cont, y_ret) against that reference. Classification's IC
    uses the CONTINUOUS score column via ``y_score_col`` override.
    """
    # Reference: regression on the continuous ground truth
    ref_headline, _ = compute_prediction_fold_metrics(
        regression_predictions, task_type="regression"
    )

    # Classification run — pass the continuous score column as ``y_score_col``
    # and point ``eval_col`` at the continuous return. That pairing should
    # reproduce the regression IC exactly.
    cls_headline, _ = compute_prediction_fold_metrics(
        classification_predictions,
        task_type="classification",
        y_score_col="y_score_cont",
        eval_col="y_ret",
        class_values=[0, 1],
    )

    assert math.isclose(cls_headline["ic_mean"], ref_headline["ic_mean"], rel_tol=1e-12)
    assert math.isclose(cls_headline["ic_std"], ref_headline["ic_std"], rel_tol=1e-12)


def test_classification_ic_differs_from_ic_on_binary_label(classification_predictions) -> None:
    """Sanity: IC on continuous return is materially different from what you'd
    get if you wrongly computed IC on the binary label.

    We simulate the wrong behavior by aliasing y_true as eval_col and check
    that IC differs from the correct run.
    """
    correct, _ = compute_prediction_fold_metrics(
        classification_predictions,
        task_type="classification",
        y_score_col="y_score_cont",
        eval_col="y_ret",
        class_values=[0, 1],
    )

    # Build a frame where eval_col points at the BINARY label (simulating the bug).
    wrong_df = classification_predictions.with_columns(y_ret_bin=pl.col("y_true"))
    wrong, _ = compute_prediction_fold_metrics(
        wrong_df,
        task_type="classification",
        y_score_col="y_score_cont",
        eval_col="y_ret_bin",
        class_values=[0, 1],
    )

    # The two IC values MUST differ materially — if they matched, IC-on-binary
    # would be indistinguishable from IC-on-continuous, defeating the rule.
    assert abs(correct["ic_mean"] - wrong["ic_mean"]) > 0.05


def test_classification_adds_auc_accuracy_logloss_to_headline(
    classification_predictions,
) -> None:
    headline, _ = compute_prediction_fold_metrics(
        classification_predictions,
        task_type="classification",
        y_score_col="y_score_cont",
        eval_col="y_ret",
        class_values=[0, 1],
    )
    for m in ("auc_roc", "auc_pr", "log_loss", "brier_score", "accuracy", "balanced_accuracy"):
        assert m in headline, f"missing classification metric {m!r} in headline"


def test_classification_headline_task_type_is_one(classification_predictions) -> None:
    headline, _ = compute_prediction_fold_metrics(
        classification_predictions,
        task_type="classification",
        y_score_col="y_score_cont",
        eval_col="y_ret",
        class_values=[0, 1],
    )
    assert headline["task_type"] == "classification"


def test_classification_auc_is_computed_on_binary_label(classification_predictions) -> None:
    """AUC / accuracy / log_loss go against the binary y_true, not the continuous
    eval_col. A classification score that is perfectly monotone in the binary
    label should yield AUC=1.0 (well above the chance baseline of 0.5).
    """
    headline, _ = compute_prediction_fold_metrics(
        classification_predictions,
        task_type="classification",
        y_score_col="y_score_cont",
        eval_col="y_ret",
        class_values=[0, 1],
    )
    # Strongly monotone score ⇒ high AUC (not 0.5)
    assert headline["auc_roc"] > 0.9


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------


def test_single_fold_returns_zero_ic_std_and_zero_t(regression_predictions) -> None:
    """With only one fold, cross-fold stddev is undefined; the function reports 0."""
    fold0_only = regression_predictions.filter(pl.col("fold_id") == 0)
    headline, _ = compute_prediction_fold_metrics(fold0_only, task_type="regression")
    assert headline["n_folds"] == 1
    assert headline["ic_std"] == 0.0
    assert headline["ic_t"] == 0.0


def test_accepts_pandas_dataframe(regression_predictions) -> None:
    """pd.DataFrame input is converted to polars internally."""
    import pandas as pd

    pdf = regression_predictions.to_pandas()
    assert isinstance(pdf, pd.DataFrame)
    headline, folds = compute_prediction_fold_metrics(pdf, task_type="regression")
    assert set(folds.keys()) == {0, 1}
    assert "ic_mean" in headline


def test_deterministic_across_calls(regression_predictions) -> None:
    """Repeated calls produce numerically equivalent output.

    BLAS threading can introduce <1e-13 jitter in rank-correlation summations,
    so we use approximate equality rather than bit-exact.
    """
    a_head, a_folds = compute_prediction_fold_metrics(
        regression_predictions, task_type="regression"
    )
    b_head, b_folds = compute_prediction_fold_metrics(
        regression_predictions, task_type="regression"
    )
    for key in a_head:
        assert a_head[key] == pytest.approx(b_head[key], abs=1e-10), key
    for fold_id in a_folds:
        for key in a_folds[fold_id]:
            assert a_folds[fold_id][key] == pytest.approx(b_folds[fold_id][key], abs=1e-10), (
                f"fold {fold_id} / {key}"
            )
