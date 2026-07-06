"""Pytest fixtures for ML4T test suite.

Two modes of operation:
1. CI (GHA): ML4T_DATA_PATH points to pre-subsampled real data (from private repo).
   populated_data_dir just returns that path — no synthetic data needed.
2. Local dev: ML4T_DATA_PATH points to full production data or test data.
"""

import json
import os
import shutil
import time
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent

# Case study IDs whose config/setup.yaml should be seeded into test output dirs
CASE_STUDY_IDS = [
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "us_firm_characteristics",
    "fx_pairs",
    "cme_futures",
    "sp500_options",
    "us_equities_panel",
]


@pytest.fixture(scope="session", autouse=True)
def ci_env_setup():
    """Create .env file if running in CI (where ML4T_DATA_PATH is set externally).

    utils/config.py requires a .env file to exist.
    In CI, environment variables are set by the workflow, but the .env
    file still needs to exist to avoid FileNotFoundError on import.
    """
    env_file = REPO_ROOT / ".env"
    created = False

    if not env_file.exists():
        # Create minimal .env for CI
        env_file.write_text(
            f"ML4T_PATH={REPO_ROOT}\n"
            f"ML4T_DATA_PATH={os.environ.get('ML4T_DATA_PATH', REPO_ROOT / 'data')}\n"
        )
        created = True

    yield

    # Clean up CI-created .env (don't leave artifacts)
    if created and env_file.exists():
        env_file.unlink()


def _resolve_data_path() -> Path | None:
    """Find ML4T_DATA_PATH from env var, .env file, or default location.

    pytest-xdist workers may not inherit env vars set by the parent process,
    so we also check the .env file and well-known test-data locations.
    """
    # 1. Explicit env var (works in single-process pytest and CI)
    env_path = os.environ.get("ML4T_DATA_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists() and any(p.iterdir()):
            return p

    # 2. Read from .env file (works in xdist workers)
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ML4T_DATA_PATH") and "=" in line:
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val and not val.startswith("#"):
                    p = Path(val).expanduser().resolve()
                    if p.exists() and any(p.iterdir()):
                        return p

    # 3. Well-known test-data repo location
    test_data = Path.home() / "ml4t" / "test-data" / "data"
    if test_data.exists() and (test_data / "etfs").exists():
        return test_data

    # 4. Default: repo's own data/ directory
    repo_data = REPO_ROOT / "data"
    if repo_data.exists() and any(repo_data.glob("*/*.parquet")):
        return repo_data

    return None


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Return the data directory for tests.

    Resolves ML4T_DATA_PATH from env var, .env file, well-known test-data
    repo location, or repo's data/ directory. Works with pytest-xdist.
    """
    resolved = _resolve_data_path()
    if resolved:
        os.environ["ML4T_DATA_PATH"] = str(resolved)
        return resolved

    # Fallback: create temp directory
    data_dir = tmp_path_factory.mktemp("ml4t_data")
    os.environ["ML4T_DATA_PATH"] = str(data_dir)
    return data_dir


@pytest.fixture(scope="session")
def populated_data_dir(test_data_dir):
    """Return a data directory populated with test data.

    If ML4T_DATA_PATH points to pre-populated data (e.g., from GHA checkout
    of ml4t/third-edition-test-data), returns it directly.
    """
    if (test_data_dir / "etfs" / "market" / "etf_universe.parquet").exists():
        return test_data_dir

    pytest.skip("No test data available. Set ML4T_DATA_PATH or run in CI.")


@pytest.fixture(scope="session")
def intermediates_dir(test_data_dir):
    """Return directory with pre-computed pipeline intermediates.

    When running downstream chapters (Ch11+), they need labels/features
    from pipeline stages. These are pre-computed and stored in test-data repo.
    """
    idir = test_data_dir.parent / "intermediates"
    if idir.exists() and any(idir.iterdir()):
        return idir
    return None


@pytest.fixture(scope="session")
def seeded_output_dir(tmp_path_factory):
    """Session-scoped output dir seeded with case study config files.

    Chapter notebooks that read case study setup.yaml (via get_case_study_dir())
    need these configs to exist even when ML4T_OUTPUT_DIR redirects writes to
    a temp directory. This fixture copies the real config files into the test
    output dir so notebooks can find them.

    With pytest-xdist, each worker gets its own subdirectory to avoid races
    on shutil.rmtree/copytree when multiple workers seed simultaneously.
    """
    base_dir = os.environ.get("ML4T_OUTPUT_DIR")
    if base_dir:
        # With xdist, append worker id to avoid races
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
        if worker_id:
            output_dir = Path(base_dir) / f"worker_{worker_id}"
        else:
            output_dir = Path(base_dir)
    else:
        output_dir = tmp_path_factory.mktemp("ml4t_output")

    # Set the env var so notebooks see this worker's output dir
    os.environ["ML4T_OUTPUT_DIR"] = str(output_dir)

    cs_root = REPO_ROOT / "case_studies"

    # Copy per-case-study config files (setup.yaml, training menus, backtest presets, etc.)
    for cs_id in CASE_STUDY_IDS:
        src_config_dir = cs_root / cs_id / "config"
        if not src_config_dir.exists():
            continue
        dst_config_dir = output_dir / cs_id / "config"
        if dst_config_dir.exists():
            shutil.rmtree(dst_config_dir)
        shutil.copytree(src_config_dir, dst_config_dir)
        _trim_label_configs(dst_config_dir)

    # Copy global model presets (case_studies/config/) so load_configs() can find them.
    # load_configs() resolves presets at {case_dir.parent}/config/{model_type}/*.yaml
    # We copy (not symlink) so we can patch presets for fast testing.
    global_config_src = cs_root / "config"
    global_config_dst = output_dir / "config"
    if global_config_src.exists():
        if global_config_dst.exists():
            shutil.rmtree(global_config_dst)
        shutil.copytree(global_config_src, global_config_dst)
        _patch_presets_for_testing(global_config_dst)

    # Copy pipeline intermediates (features, labels, run_log) from test-data repo.
    # These are pre-computed so downstream notebooks (Ch11+) can run without
    # executing the full pipeline first.
    # Look for intermediates next to data (test-data repo layout) or at well-known path.
    data_path = _resolve_data_path()
    intermediates_root = None
    if data_path:
        candidate = Path(data_path).parent / "intermediates"
        if candidate.exists():
            intermediates_root = candidate
    if intermediates_root is None:
        # Well-known test-data repo location
        candidate = Path.home() / "ml4t" / "test-data" / "intermediates"
        if candidate.exists():
            intermediates_root = candidate
    if intermediates_root and intermediates_root.exists():
        for cs_id in CASE_STUDY_IDS:
            src = intermediates_root / cs_id
            if not src.exists():
                continue
            dst = output_dir / cs_id
            # Copy features, labels, evaluation, run_log, results, benchmark —
            # anything that downstream notebooks look for in get_case_study_dir()
            for subdir in ["features", "labels", "evaluation", "run_log", "results", "benchmark"]:
                src_sub = src / subdir
                dst_sub = dst / subdir
                if src_sub.exists() and not dst_sub.exists():
                    shutil.copytree(src_sub, dst_sub)
            # Copy top-level intermediate files (e.g. etfs/eligibility.csv,
            # protocol.yaml, baseline_checkpoint.yaml) that sit directly in
            # intermediates/{cs_id}/ rather than in a subdir. Downstream
            # notebooks (etfs 02_labels, 03_financial_features) read these via
            # get_case_study_dir(); without this they fail with FileNotFoundError.
            for item in src.iterdir():
                if item.is_file():
                    dst_file = dst / item.name
                    if not dst_file.exists():
                        dst.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dst_file)
            # Schema reconciliation: test-data predictions parquets were
            # generated with an older column convention (y_score / y_true /
            # fold_id). Production registry uses (prediction / actual / fold).
            # Rename in place so notebooks reading via get_case_study_dir()
            # see the canonical names without per-notebook compat shims.
            preds_root = dst / "run_log" / "predictions"
            if preds_root.exists():
                _migrate_predictions_schema(preds_root)

    # Copy non-case-study intermediates (chapter-scoped fixtures).
    # These are intermediates that downstream teaching notebooks need but aren't
    # part of the per-case-study pipeline (e.g., Ch16 signal comparison, Ch20 synthesis).
    if intermediates_root and intermediates_root.exists():
        for extra_id in ["ch16_signal_method_comparison", "ch20_synthesis"]:
            src = intermediates_root / extra_id
            if not src.exists():
                continue
            dst = output_dir / extra_id
            if not dst.exists():
                shutil.copytree(src, dst)

    # Seed minimal results fixtures so downstream notebooks (latent factors, DL,
    # backtest) can find baseline results without depending on upstream execution.
    # These fill gaps where intermediates don't provide enough (e.g., Ch25 demo
    # predictions, Ch15 causal JSON, synthetic registry entries).
    from tests.fixtures.seed_results import seed_results

    seed_results(output_dir, CASE_STUDY_IDS)

    # Symlink AQR factor data so AQRFactorProvider finds it at ~/ml4t/data/aqr_factors
    aqr_src = data_path.parent / "data" / "factors" / "aqr" if data_path else None
    if aqr_src is None:
        aqr_src = Path.home() / "ml4t" / "test-data" / "data" / "factors" / "aqr"
    aqr_dst = Path.home() / "ml4t" / "data" / "aqr_factors"
    if aqr_src.exists() and not aqr_dst.exists():
        aqr_dst.parent.mkdir(parents=True, exist_ok=True)
        aqr_dst.symlink_to(aqr_src)

    return output_dir


# ---------------------------------------------------------------------------
# Preset patching — reduce workload for CI/test runs
# ---------------------------------------------------------------------------

# Per-model-type overrides applied to copied preset YAMLs.
# Goal: minimal workload that still exercises the training loop + registry.
_TEST_PRESET_PATCHES: dict[str, dict] = {
    "lgb": {"max_iterations": 2, "checkpoint_interval": 1},
    # DL families: 2 epochs, checkpoint every epoch
    "lstm": {"n_epochs": 2, "checkpoint_interval": 1},
    "tsmixer": {"n_epochs": 2, "checkpoint_interval": 1},
    "tcn": {"n_epochs": 2, "checkpoint_interval": 1},
    "nlinear": {"n_epochs": 2, "checkpoint_interval": 1},
    "patchtst": {"n_epochs": 2, "checkpoint_interval": 1},
    # TabDL: 2 epochs
    "tabm": {"n_epochs": 2, "checkpoint_interval": 1},
    # Latent factors: 2 epochs
    "cae": {"n_epochs": 2, "checkpoint_interval": 1},
    "sdf": {"n_epochs": 2, "checkpoint_interval": 1},
    "sae": {"n_epochs": 2, "checkpoint_interval": 1},
    "ipca": {"n_epochs": 2, "checkpoint_interval": 1},
}


_PREDICTION_COL_RENAMES = {
    "y_score": "prediction",
    "y_true": "actual",
    "fold_id": "fold",
}


def _migrate_predictions_schema(preds_root: Path) -> None:
    """Rename test-data prediction columns to canonical production schema.

    Test-data parquets were generated with an older convention
    (y_score / y_true / fold_id). Production registry uses
    (prediction / actual / fold). Walking the seeded predictions tree once
    avoids per-notebook compat shims while keeping test-data immutable on
    its own repo schedule.
    """
    import polars as pl

    for parquet in preds_root.rglob("predictions.parquet"):
        cols = pl.read_parquet(parquet, n_rows=0).columns
        renames = {old: new for old, new in _PREDICTION_COL_RENAMES.items() if old in cols}
        if not renames:
            continue
        df = pl.read_parquet(parquet).rename(renames)
        df.write_parquet(parquet)


def _patch_presets_for_testing(config_dir: Path) -> None:
    """Patch copied preset YAMLs with reduced-workload values for testing."""
    for model_type, overrides in _TEST_PRESET_PATCHES.items():
        model_dir = config_dir / model_type
        if not model_dir.exists():
            continue
        for preset_path in model_dir.glob("*.yaml"):
            preset = yaml.safe_load(preset_path.read_text())
            if preset is None:
                continue
            preset.update(overrides)
            with open(preset_path, "w") as f:
                yaml.dump(preset, f, default_flow_style=False)


# Max configs per family in label config files (keep tests fast but comprehensive).
# Only applied to families with homogeneous sweep configs (linear, gbm).
# DL/TabDL/latent/causal families are NOT trimmed because each config often
# maps to a dedicated notebook (e.g., 09_dl_lstm, 10_dl_tsmixer).
_MAX_CONFIGS_PER_FAMILY = 2
_TRIM_FAMILIES = {"linear", "gbm"}


def _trim_label_configs(cs_config_dir: Path) -> None:
    """Trim training menu YAMLs to at most _MAX_CONFIGS_PER_FAMILY for sweep families."""
    training_dir = cs_config_dir / "training"
    label_root = training_dir if training_dir.exists() else cs_config_dir
    for label_yaml in label_root.glob("fwd_*.yaml"):
        data = yaml.safe_load(label_yaml.read_text())
        if data is None or not isinstance(data, dict):
            continue
        trimmed = False
        for family, configs in data.items():
            if (
                family in _TRIM_FAMILIES
                and isinstance(configs, list)
                and len(configs) > _MAX_CONFIGS_PER_FAMILY
            ):
                data[family] = configs[:_MAX_CONFIGS_PER_FAMILY]
                trimmed = True
        if trimmed:
            with open(label_yaml, "w") as f:
                yaml.dump(data, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# GPU marker — apply `@pytest.mark.gpu` at collection time based on overrides.
# Usage: pytest -m gpu  (GPU only) | pytest -m "not gpu" (CPU only)
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: notebook requires GPU (from overrides.yaml gpu: true)")
    config.addinivalue_line(
        "markers",
        "long_running: notebook takes >10min even with reduced params (from overrides.yaml long_running: true)",
    )
    config.addinivalue_line(
        "markers",
        "weekly: notebook tier=weekly — runs only in scheduled weekly-external workflow. "
        "To execute locally, set ML4T_TEST_TIER=weekly alongside `pytest -m weekly`; "
        "without the env var, matching items are collected but skipped.",
    )
    config.addinivalue_line(
        "markers",
        "on_demand: notebook tier=on_demand — runs only on manual dispatch (e.g., GPU Tier 3). "
        "To execute locally, set ML4T_TEST_TIER=on_demand alongside `pytest -m on_demand`; "
        "without the env var, matching items are collected but skipped.",
    )
    config.addinivalue_line(
        "markers",
        "drift: live external drift smoke (Phase 3) — one tiny real pull per free "
        "data source. Network-bound; runs only in the scheduled weekly-external "
        "workflow's `drift` job, never per-PR.",
    )


def pytest_collection_modifyitems(items):
    """Add markers to test items based on overrides.yaml flags."""
    from tests.pm_helpers import (
        TIER_ON_DEMAND,
        TIER_WEEKLY,
        get_overrides,
        get_reruns,
        get_tier,
    )

    # pytest-rerunfailures provides @pytest.mark.flaky(reruns=N). Detect once
    # so per-NB reruns kick in automatically when the dep lands in Step 2.
    try:
        import pytest_rerunfailures  # noqa: F401

        has_rerunfailures = True
    except ImportError:
        has_rerunfailures = False

    for item in items:
        if hasattr(item, "callspec") and "notebook_path" in item.callspec.params:
            nb_path = item.callspec.params["notebook_path"]
            rel = (
                nb_path.relative_to(REPO_ROOT).with_suffix("")
                if hasattr(nb_path, "relative_to")
                else nb_path
            )
            overrides = get_overrides(str(rel)) or {}
            if overrides.get("gpu"):
                item.add_marker(pytest.mark.gpu)
            if overrides.get("long_running"):
                item.add_marker(pytest.mark.long_running)

            tier = get_tier(overrides)
            if tier == TIER_WEEKLY:
                item.add_marker(pytest.mark.weekly)
            elif tier == TIER_ON_DEMAND:
                item.add_marker(pytest.mark.on_demand)

            reruns = get_reruns(overrides)
            if reruns > 0 and has_rerunfailures:
                item.add_marker(pytest.mark.flaky(reruns=reruns, reruns_delay=30))


# ---------------------------------------------------------------------------
# Incremental result saving — write JSONL after each test so results survive
# process kills. Results file: /tmp/ml4t-test-results.jsonl
# ---------------------------------------------------------------------------

_RESULTS_PATH = Path(os.environ.get("ML4T_RESULTS_FILE", "/tmp/ml4t-test-results.jsonl"))
_test_start_times: dict[str, float] = {}


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Record test start time."""
    _test_start_times[item.nodeid] = time.time()


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    """Write each test result to JSONL as it completes."""
    if report.when != "call" and not (report.when == "setup" and report.skipped):
        return

    start = _test_start_times.pop(report.nodeid, 0)
    duration = report.duration if hasattr(report, "duration") else 0

    outcome = report.outcome  # "passed", "failed", or "skipped"

    record = {
        "nodeid": report.nodeid,
        "outcome": outcome,
        "duration_s": round(duration, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if outcome == "failed" and report.longreprtext:
        record["error"] = report.longreprtext[:500]

    with open(_RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


@pytest.fixture
def clean_env():
    """Fixture that provides a clean environment and restores it after."""
    saved_env = os.environ.copy()
    yield os.environ
    os.environ.clear()
    os.environ.update(saved_env)
