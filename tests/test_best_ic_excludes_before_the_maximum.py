"""Excluding a prediction from the family leaderboard must not delete the family.

`load_best_ic_per_family` reduces each family to its highest-IC row. A caller that has to
drop rows - retirement is the usual reason, because the metrics catalog carries no lineage -
can only do that correctly *before* the reduction. Filtering the returned frame is too late:
every runner-up has already been discarded, so excluding a family's leader removes the family
from the result instead of falling back to its best remaining row.

That distinction is invisible in the common case, where nothing is excluded and both spellings
agree. It shows up when the excluded row happens to be the leader, and the caller that looks
its baselines up by name then raises rather than comparing against the live runner-up.
"""

from __future__ import annotations

import polars as pl
import pytest

from case_studies.utils import analytics


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> pl.DataFrame:
    """Two families, two configurations each, ranked within family by IC."""
    frame = pl.DataFrame(
        {
            "case_study": ["etfs"] * 4,
            "family": ["linear", "linear", "gbm", "gbm"],
            "config_name": ["ridge_hi", "ridge_lo", "leaves_hi", "leaves_lo"],
            "label": ["fwd_ret_21d"] * 4,
            "ic_mean": [0.05, 0.04, 0.08, 0.06],
            "ic_n_days": [1995.0] * 4,
            "prediction_hash": ["lin_hi", "lin_lo", "gbm_hi", "gbm_lo"],
        }
    )
    monkeypatch.setattr(analytics, "load_model_ic", lambda *a, **k: frame)
    return frame


def _rows(result: pl.DataFrame) -> dict[str, str]:
    return dict(zip(result["family"], result["config_name"], strict=True))


def test_without_an_exclusion_each_family_reports_its_leader(catalog: pl.DataFrame) -> None:
    result = analytics.load_best_ic_per_family(["linear", "gbm"], case_studies=["etfs"])
    assert _rows(result) == {"linear": "ridge_hi", "gbm": "leaves_hi"}


def test_excluding_a_leader_falls_back_to_the_family_runner_up(catalog: pl.DataFrame) -> None:
    result = analytics.load_best_ic_per_family(
        ["linear", "gbm"],
        case_studies=["etfs"],
        exclude_prediction_hashes=["lin_hi", "gbm_hi"],
    )
    # Both families survive. Filtering the returned frame instead would have left neither.
    assert _rows(result) == {"linear": "ridge_lo", "gbm": "leaves_lo"}


def test_excluding_a_runner_up_leaves_the_leader_standing(catalog: pl.DataFrame) -> None:
    result = analytics.load_best_ic_per_family(
        ["linear", "gbm"], case_studies=["etfs"], exclude_prediction_hashes=["lin_lo"]
    )
    assert _rows(result) == {"linear": "ridge_hi", "gbm": "leaves_hi"}


def test_excluding_every_row_of_one_family_drops_only_that_family(
    catalog: pl.DataFrame,
) -> None:
    result = analytics.load_best_ic_per_family(
        ["linear", "gbm"], case_studies=["etfs"], exclude_prediction_hashes=["lin_hi", "lin_lo"]
    )
    assert _rows(result) == {"gbm": "leaves_hi"}


def test_excluding_everything_returns_an_empty_frame(catalog: pl.DataFrame) -> None:
    result = analytics.load_best_ic_per_family(
        ["linear", "gbm"],
        case_studies=["etfs"],
        exclude_prediction_hashes=["lin_hi", "lin_lo", "gbm_hi", "gbm_lo"],
    )
    assert result.is_empty()


def test_an_empty_exclusion_is_not_read_as_no_filter(catalog: pl.DataFrame) -> None:
    # `[]` is falsy; the parameter is checked against None so that an empty exclusion means
    # "exclude nothing" rather than reaching a branch that skips the filter for the wrong reason.
    result = analytics.load_best_ic_per_family(
        ["linear", "gbm"], case_studies=["etfs"], exclude_prediction_hashes=[]
    )
    assert _rows(result) == {"linear": "ridge_hi", "gbm": "leaves_hi"}
