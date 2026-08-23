from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from case_studies.us_firm_characteristics.research_workflow import (
    causal_estimand_labels,
    model_request_catalog,
    model_requests,
    open_study,
)


def test_open_study_writes_canonical_results_to_output_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "production"
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(output_root))

    study = open_study(
        execution_tier="canonical",
        workspace=tmp_path / "fallback",
    )

    assert study.root == output_root / "us_firm_characteristics"
    assert study.output_root == output_root
    assert study.root != study.release_case_root
    assert (study.root / "labels").is_dir()
    assert (study.root / "features").is_dir()
    assert (study.root / "run_log" / "registry.db").is_file()


def test_ipca_requests_share_the_case_runtime_contract() -> None:
    calls: list[dict[str, Any]] = []

    class RecordingStudy:
        def model(self, **kwargs):
            calls.append(kwargs)
            return kwargs

    catalog = pl.DataFrame(
        [{"family": "latent_factors", "label": "fwd_ret_1m", "config_name": "ipca"}]
    )

    requests = model_requests(
        RecordingStudy(),
        catalog,
        execution_tier="preview",
        preview_reductions={"folds": [0]},
    )

    assert requests == (calls[0],)
    assert calls[0]["overrides"] == {"device": "cpu", "fold_workers": 4}
    assert calls[0]["preview_reductions"] == {"folds": [0]}


class _StubLabels:
    """Answer the one question the causal filter asks of a study."""

    def __init__(self, tasks: dict[str, str]) -> None:
        self._tasks = tasks

    def get(self, name: str):
        task = self._tasks[name]
        return SimpleNamespace(definition=SimpleNamespace(task_type=task))


class _StubStudy:
    def __init__(self, tasks: dict[str, str]) -> None:
        self.labels = _StubLabels(tasks)


def test_causal_estimands_keep_continuous_labels_and_name_what_they_drop(capsys) -> None:
    study = _StubStudy(
        {
            "fwd_ret_1m": "regression",
            "fwd_ret_1m_win": "regression",
            "fwd_class_1m": "classification",
        }
    )

    kept = causal_estimand_labels(study, ("fwd_ret_1m", "fwd_ret_1m_win", "fwd_class_1m"))

    # The DML nuisance and outcome models are regressors, so a class target would resolve
    # to a number that means nothing. Dropping it silently is the failure this guards.
    assert kept == ("fwd_ret_1m", "fwd_ret_1m_win")
    assert "fwd_class_1m" in capsys.readouterr().out


def test_causal_estimands_raise_rather_than_return_an_empty_set() -> None:
    study = _StubStudy({"fwd_class_1m": "classification"})

    with pytest.raises(ValueError, match="no continuous label"):
        causal_estimand_labels(study, ("fwd_class_1m",))


def test_an_empty_label_selection_asks_for_nothing_not_for_everything() -> None:
    # `config_names` has always read an empty selection as "nothing" via `is not None`,
    # while `labels` read it as "everything" through `labels or declared_labels()`. The two
    # sat one line apart in the same signature. Asking for no labels now raises instead of
    # silently returning the full published population.
    with pytest.raises(ValueError, match="no declared requests"):
        model_request_catalog("linear", labels=[])

    assert model_request_catalog("linear", labels=None).height > 0
