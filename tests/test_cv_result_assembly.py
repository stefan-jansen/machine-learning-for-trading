"""Behavioral tests for shared cross-validation result assembly."""

import polars as pl
import pytest

from case_studies.utils.cv_results import assemble_cv_result


def test_result_assembly_rejects_empty_prediction_set() -> None:
    curves = pl.DataFrame([{"config": "lstm", "epoch": 10, "ic_mean": 0.03, "ic_n_days": 3}])

    with pytest.raises(ValueError, match="without prediction rows"):
        assemble_cv_result(
            curves,
            pl.DataFrame(),
            date_col="timestamp",
            entity_col="symbol",
        )
