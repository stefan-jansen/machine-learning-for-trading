from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from case_studies.us_firm_characteristics.research_workflow import model_requests, open_study


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
