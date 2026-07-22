"""Gate: feature-selection notebooks must scope development analysis pre-holdout.

Feature selection, robustness sweeps, and feature evaluation are development
decisions; the sealed holdout (``case_studies/etfs/config/setup.yaml`` ->
``evaluation.holdout_start``) must not inform them (the rule taught in
``06_strategy_definition/02_cv_foundations``). Regenerating the selection
artifacts requires the full data downloads, so this test statically checks the
notebook sources instead: each must read the holdout boundary from
``setup.yaml`` and apply the holdout filter *before* its first IC computation.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
SETUP_YAML = REPO_ROOT / "case_studies" / "etfs" / "config" / "setup.yaml"

# (notebook source, identifier marking its first IC computation). The filter
# must be applied earlier in the file than this marker so no IC ranking,
# BH-FDR discovery, stability selection, or importance ranking ever sees
# holdout rows.
HOLDOUT_SCOPED_NOTEBOOKS = [
    ("08_financial_features/05_feature_selection.py", "ic_by_date"),
    ("08_financial_features/06_robustness_sensitivity.py", "def compute_momentum_ic_series"),
    ("case_studies/etfs/03_financial_features.py", "ic_matrix = np.full"),
    ("case_studies/etfs/02_labels.py", "def compute_ic_by_date"),
    ("case_studies/etfs/05_evaluation.py", "ic_series_data = {feat"),
]


def load_holdout_start() -> str:
    setup = yaml.safe_load(SETUP_YAML.read_text())
    return setup["evaluation"]["holdout_start"]


def test_setup_yaml_declares_sealed_holdout() -> None:
    """The holdout window is declared, ISO-formatted, and non-empty."""
    setup = yaml.safe_load(SETUP_YAML.read_text())
    holdout_start = date.fromisoformat(setup["evaluation"]["holdout_start"])
    holdout_end = date.fromisoformat(setup["evaluation"]["holdout_end"])
    assert holdout_start < holdout_end


@pytest.mark.parametrize(
    ("rel_path", "first_ic_marker"),
    HOLDOUT_SCOPED_NOTEBOOKS,
    ids=[rel_path for rel_path, _ in HOLDOUT_SCOPED_NOTEBOOKS],
)
def test_holdout_filter_precedes_first_ic_computation(rel_path: str, first_ic_marker: str) -> None:
    source = (REPO_ROOT / rel_path).read_text()

    # The boundary must come from setup.yaml, not a hardcoded literal that can
    # silently drift from the case-study config.
    assert "setup.yaml" in source, f"{rel_path}: holdout boundary must be read from setup.yaml"
    assert "holdout_start" in source, (
        f"{rel_path}: must read evaluation.holdout_start from setup.yaml"
    )

    ic_pos = source.find(first_ic_marker)
    assert ic_pos != -1, (
        f"{rel_path}: first-IC marker {first_ic_marker!r} not found — "
        "update HOLDOUT_SCOPED_NOTEBOOKS if the notebook was restructured"
    )

    filter_pos = source.upper().find("HOLDOUT_START")
    assert 0 <= filter_pos < ic_pos, (
        f"{rel_path}: the holdout filter must be applied before the first IC "
        "computation so selection decisions never see the sealed holdout"
    )


def test_selected_features_artifact_stops_before_holdout() -> None:
    """If the Ch8 selection artifact has been regenerated locally, its feature
    panel must not extend into the sealed holdout."""
    pl = pytest.importorskip("polars")

    # Production location of get_output_dir(8, "feature_selection") from
    # 05_feature_selection.py: {chapter_dir}/output/{strategy_id}/.
    artifact = (
        REPO_ROOT
        / "08_financial_features"
        / "output"
        / "feature_selection"
        / "features_selected.parquet"
    )
    if not artifact.exists():
        pytest.skip("selection artifact not generated locally (requires full data)")

    holdout_start = date.fromisoformat(load_holdout_start())
    max_ts = pl.scan_parquet(artifact).select(pl.col("timestamp").max()).collect().item()
    assert max_ts is not None
    max_date = max_ts.date() if hasattr(max_ts, "date") else max_ts
    assert max_date < holdout_start, (
        f"features_selected.parquet extends to {max_date}, at/after the sealed "
        f"holdout start {holdout_start} — feature selection saw holdout data"
    )


def test_crypto_dml_seals_the_label_endpoint_and_hashes_current_inputs() -> None:
    """The DML sample and cache identity must bind the corrected 8-hour lineage."""
    source = (REPO_ROOT / "case_studies" / "crypto_perps_funding" / "11_causal_dml.py").read_text()

    assert 'ESTIMATOR_CONTRACT = "panel_time_v3"' in source
    assert "modeling_input_fingerprint(" in source
    assert '"input_fingerprint": INPUT_FINGERPRINT' in source
    assert "holdout_cutoff - label_horizon" in source
    assert "max() + LABEL_HORIZON >= HOLDOUT_CUTOFF" in source


def test_crypto_gbm_requires_cuda_and_hashes_current_inputs() -> None:
    """The corrected GBM grid must be GPU-only and content-addressed."""
    source = (REPO_ROOT / "case_studies" / "crypto_perps_funding" / "07_gbm.py").read_text()

    assert 'TRAIN_DEVICE = "cuda"' in source
    assert "modeling_input_fingerprint(" in source
    assert '"device": TRAIN_DEVICE' in source
    assert '"input_fingerprint": INPUT_FINGERPRINT' in source
    assert source.count("extra_params=IDENTITY_PARAMS") == 2
    assert "CPU fallback" not in source


@pytest.mark.parametrize("notebook", ["06_linear.py", "07_gbm.py"])
def test_crypto_model_guard_uses_active_24h_label_buffer(notebook: str) -> None:
    """Crypto model guards must purge the endpoint for the selected label horizon."""
    source = (REPO_ROOT / "case_studies" / "crypto_perps_funding" / notebook).read_text()
    setup = yaml.safe_load(
        (REPO_ROOT / "case_studies/crypto_perps_funding/config/setup.yaml").read_text()
    )

    assert setup["labels"]["variant_buffers"]["fwd_ret_24h"] == "24H"
    assert "pd.Timedelta(mds.label_buffer)" in source
    assert 'split["val_end"] + label_horizon < holdout_start' in source
    assert 'split["val_end"] + timedelta(hours=8)' not in source


def test_crypto_tabm_requires_cuda_and_hashes_current_inputs() -> None:
    """The corrected TabM grid must be GPU-only and content-addressed."""
    source = (REPO_ROOT / "case_studies" / "crypto_perps_funding" / "08_tabular_dl.py").read_text()

    assert 'TRAIN_DEVICE = "cuda"' in source
    assert "modeling_input_fingerprint(" in source
    assert '"device": TRAIN_DEVICE' in source
    assert '"input_fingerprint": INPUT_FINGERPRINT' in source
    assert source.count("extra_params=IDENTITY_PARAMS") == 2
    assert "identity_params=IDENTITY_PARAMS" in source
    assert "current CPU" not in source


def test_crypto_lstm_requires_cuda_and_hashes_current_inputs() -> None:
    """The corrected LSTM run must be GPU-only and content-addressed."""
    source = (REPO_ROOT / "case_studies" / "crypto_perps_funding" / "09_dl_lstm.py").read_text()

    assert 'TRAIN_DEVICE = "cuda"' in source
    assert "modeling_input_fingerprint(" in source
    assert '"device": TRAIN_DEVICE' in source
    assert '"input_fingerprint": INPUT_FINGERPRINT' in source
    assert "extra_params=IDENTITY_PARAMS" in source
    assert "identity_params=IDENTITY_PARAMS" in source
    assert "current CPU" not in source


def test_crypto_tcn_requires_cuda_and_hashes_current_inputs() -> None:
    """The corrected TCN run must be GPU-only and content-addressed."""
    source = (REPO_ROOT / "case_studies" / "crypto_perps_funding" / "10_dl_tcn.py").read_text()

    assert 'TRAIN_DEVICE = "cuda"' in source
    assert "modeling_input_fingerprint(" in source
    assert '"device": TRAIN_DEVICE' in source
    assert '"input_fingerprint": INPUT_FINGERPRINT' in source
    assert source.count("extra_params=IDENTITY_PARAMS") == 2
    assert "identity_params=IDENTITY_PARAMS" in source
    assert "current CPU" not in source
