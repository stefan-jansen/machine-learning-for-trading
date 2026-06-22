"""Test model notebooks produce correct registry entries in isolation.

Runs each case study model notebook (stage >= 06) with minimal parameters
in an isolated environment. Production data is read via symlinks; all
writes (registry.db, predictions, results JSON) go to a temp directory.

The production registry is NEVER opened or touched.

Design:
    1. Session fixture creates temp dir with symlinked read-only data
    2. Each notebook runs via Papermill with aggressive param reduction
    3. ML4T_OUTPUT_DIR redirects all get_case_study_dir() writes to temp
    4. After each run, query the test registry.db for expected entries

The goal is code-path coverage, not model quality. Params are set to the
absolute minimum that still exercises the training→register→predict loop:
MAX_SYMBOLS=3, MAX_FOLDS=2, N_EPOCHS=2, NUM_BOOST_ROUND=20.

Usage:
    # All model notebooks (~15-20 min)
    uv run pytest tests/test_model_registry.py -v

    # Specific case study
    uv run pytest tests/test_model_registry.py -v -k "crypto_perps_funding"

    # Specific model family across all case studies
    uv run pytest tests/test_model_registry.py -v -k "06_linear"

    # Single notebook
    uv run pytest tests/test_model_registry.py -v -k "etfs and 06_linear"

    # Dry run — see what would be tested
    uv run pytest tests/test_model_registry.py --collect-only
"""

import re
import sqlite3
from pathlib import Path

import pytest

from tests.pm_helpers import get_overrides, run_notebook

REPO_ROOT = Path(__file__).parent.parent
PROD_CS_DIR = REPO_ROOT / "case_studies"

# Ordered smallest-to-largest for faster feedback
CASE_STUDIES = [
    "crypto_perps_funding",
    "fx_pairs",
    "cme_futures",
    "etfs",
    "sp500_options",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "us_firm_characteristics",
    "us_equities_panel",
]

# Directories containing production pipeline artifacts (read-only).
# Symlinked into the test output directory so model notebooks can read them.
# Everything else (run_log/, results/, models/) is created fresh for writes.
_READ_ONLY_DIRS = {"config", "features", "labels"}

# Minimum stage number for model notebooks
_MODEL_STAGE_MIN = 6

# Suffixes that are NOT model notebooks (backtest, strategy, diagnostics).
# These depend on upstream predictions and should be tested separately.
_EXCLUDED_SUFFIXES = frozenset(
    {
        "backtest",
        "backtest_sweep",
        "backtest_analysis",
        "portfolio_management",
        "costs",
        "risk_management",
        "model_analysis",
        "strategy_analysis",
        "synthesis",
        "ic_diagnostic",
        "prediction_ingestion",
    }
)

# Latent factor models need more symbols than other families because factor
# extraction requires a cross-section wide enough for the covariance matrix.
_LATENT_FACTOR_SUFFIXES = frozenset(
    {
        "latent_factors",
        "pca",
        "ipca",
        "sdf",
        "cae",
        "sae",
        "term_structure_pca",
    }
)
_LATENT_FACTOR_OVERRIDES = {
    "MAX_SYMBOLS": 10,
    "N_FACTORS": 3,
}

# Case studies with sparse data (monthly frequency) need more symbols
# to have enough observations for CV splits.
_SPARSE_DATA_CASE_STUDIES = frozenset({"us_firm_characteristics"})
_SPARSE_DATA_OVERRIDES = {"MAX_SYMBOLS": 20}

# Minimal parameters for code-path coverage. Applied LAST so they
# override anything from overrides.yaml — we want the absolute minimum
# that still exercises the full train→register→predict loop.
_QUICK_PARAMS = {
    "MAX_SYMBOLS": 3,
    "MAX_FOLDS": 2,
    "N_EPOCHS": 2,
    "NUM_BOOST_ROUND": 20,
    "BATCH_SIZE": 64,
    "LOOKBACK": 24,  # PatchTST needs lookback + stride >= patch_len (≥8); 24 leaves margin
    "MAX_SAMPLES": 1000,
    "CV_FOLDS": 2,
    "N_PLACEBO": 3,
    "N_FACTORS": 2,
    "FORCE_RETRAIN": True,
}

# Model suffixes known to use register=True (training_runs + prediction_sets).
# Matched against the suffix after the NN_ prefix, since notebook numbers
# vary across case studies (e.g. causal_dml is 10, 11, 12, or 13 depending
# on the case study).
# Built from: grep -l "register=True" case_studies/*/[0-9][0-9]_*.py
_REGISTERING_SUFFIXES = frozenset(
    {
        "linear",
        "gbm",
        "tabular_dl",
        "dl_lstm",
        "dl_patchtst",
        "dl_tsmixer",
        "dl_nlinear",
        "dl_tcn",
        # NOTE: causal_dml notebooks register to ``causal_runs`` (DML effect
        # estimates), not ``training_runs`` — so they are intentionally NOT in
        # this set. Likewise ``NN_latent_factors`` is a thin index notebook that
        # only displays the best already-registered factor IC; the factor models
        # themselves register under their own sub-stems (pca/ipca/sdf/cae/sae,
        # listed below). Both still execute; only the training-run-registration
        # assertion is skipped for them.
        "ipca",
        "pca",
        "sdf",
        "cae",
        "sae",
    }
)

# DL notebooks use entry_point = "dl_{model}" (e.g. "dl_lstm") instead of
# the full filename stem (e.g. "09_dl_lstm"). Map stage stems to actual
# entry_point values for these notebooks.
_DL_RE = re.compile(r"^\d{2}_(dl_.+)$")


def _expected_entry_point(stage: str) -> str:
    """Return the entry_point value the notebook will use in the registry."""
    m = _DL_RE.match(stage)
    if m:
        return m.group(1)  # "09_dl_lstm" → "dl_lstm"
    return stage  # "06_linear" → "06_linear"


_STAGE_RE = re.compile(r"^(\d{2})_")


# ---------------------------------------------------------------------------
# Test collection
# ---------------------------------------------------------------------------


def _collect_model_notebooks() -> list[tuple[str, str, Path]]:
    """Discover all model notebooks (stage >= 06) across case studies.

    Returns (case_study, stage_stem, notebook_path) tuples sorted by
    case study order then filename within each case study.
    """
    tests = []
    for cs in CASE_STUDIES:
        cs_dir = PROD_CS_DIR / cs
        if not cs_dir.exists():
            continue
        for notebook in sorted(cs_dir.glob("[0-9][0-9]_*.py")):
            if notebook.name.startswith("_"):
                continue
            match = _STAGE_RE.match(notebook.name)
            if not match:
                continue
            stage_num = int(match.group(1))
            if stage_num < _MODEL_STAGE_MIN:
                continue
            # Skip non-model notebooks (backtest, strategy, diagnostics)
            suffix = notebook.stem[len(match.group(0)) :]
            if suffix in _EXCLUDED_SUFFIXES:
                continue
            tests.append((cs, notebook.stem, notebook))
    return tests


MODEL_TESTS = _collect_model_notebooks()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def isolated_model_output(tmp_path_factory):
    """Create an isolated output directory with symlinked production data.

    For each case study, symlinks read-only directories (config/, features/,
    labels/) from the production case study directory so that model notebooks
    can load upstream artifacts. Write-target directories (run_log/, results/,
    models/) are NOT symlinked — they are created fresh by the notebooks.

    Returns the temp root directory (passed as output_dir to run_notebook,
    which sets ML4T_OUTPUT_DIR).
    """
    import shutil

    test_root = tmp_path_factory.mktemp("model_registry_test")

    for cs in CASE_STUDIES:
        prod_cs = PROD_CS_DIR / cs
        if not prod_cs.exists():
            continue

        test_cs = test_root / cs
        test_cs.mkdir()

        for subdir in _READ_ONLY_DIRS:
            src = prod_cs / subdir
            if src.exists():
                (test_cs / subdir).symlink_to(src.resolve())

    # Seed the global preset library (case_studies/config/{model_type}/*.yaml).
    # load_configs() resolves presets at {case_dir.parent}/config/, which maps
    # to test_root/config/ when ML4T_OUTPUT_DIR is set. Without this, every
    # notebook that loads GBM/DL/TabDL/latent/causal presets fails.
    global_config_src = PROD_CS_DIR / "config"
    global_config_dst = test_root / "config"
    if global_config_src.exists():
        shutil.copytree(global_config_src, global_config_dst)
        # Patch presets for minimal runtime (2 epochs, etc.)
        from tests.conftest import _patch_presets_for_testing

        _patch_presets_for_testing(global_config_dst)

    return test_root


LOG_PATH = Path("/tmp/model_registry_test.log")


@pytest.fixture(scope="session", autouse=True)
def _init_log():
    """Initialize the progress log and route Papermill cell output to it."""
    import logging
    import time

    with open(LOG_PATH, "w") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] === Model Registry Test Suite ===\n")
        f.write(f"[{time.strftime('%H:%M:%S')}] {len(MODEL_TESTS)} tests collected\n")
        f.flush()

    # Route Papermill's cell-level progress + notebook print() output to log file.
    # Papermill uses "papermill" logger (not "papermill.execute") for cell markers
    # and captured output. We add a file handler so it goes to our log regardless
    # of pytest's log level.
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    for logger_name in ("papermill", "papermill.execute"):
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't pollute pytest captured output


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _query_registry(db_path: Path, table: str, where: str = "") -> list[dict]:
    """Query a registry table and return rows as dicts."""
    if not db_path.exists():
        return []
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    try:
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return [dict(r) for r in db.execute(sql).fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        db.close()


def _registry_summary(db_path: Path) -> dict:
    """Return a summary of registry contents for reporting."""
    return {
        "training_runs": len(_query_registry(db_path, "training_runs")),
        "prediction_sets": len(_query_registry(db_path, "prediction_sets")),
        "prediction_metrics": len(_query_registry(db_path, "prediction_metrics")),
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_study,stage,notebook_path",
    MODEL_TESTS,
    ids=[f"{cs}::{stage}" for cs, stage, _ in MODEL_TESTS],
)
def test_model_notebook(case_study, stage, notebook_path, isolated_model_output):
    """Run a model notebook in isolation and verify registry output.

    Steps:
    1. Load per-notebook overrides (timeout, parameters, skip/gpu flags)
    2. Merge with default reduced parameters (MAX_SYMBOLS=15, MAX_FOLDS=2)
    3. Execute via Papermill with ML4T_OUTPUT_DIR → isolated temp dir
    4. Assert successful completion
    5. For notebooks with register=True, assert registry entries exist
    """
    # --- Skip / override handling ---
    rel_path = notebook_path.relative_to(REPO_ROOT).with_suffix("")
    overrides = get_overrides(str(rel_path))

    if overrides.get("skip"):
        pytest.skip(overrides.get("skip_reason", "marked skip in overrides"))

    if overrides.get("gpu"):
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("GPU required but not available")
        except ImportError:
            pytest.skip("torch not installed")

    # --- Parameters ---
    # Start with overrides.yaml, then apply ALL quick-test params on top.
    # Quick params win — we want minimal runtime, not overrides.yaml scale.
    # Papermill warns (but doesn't error) about unknown parameters, so it's
    # safe to inject all of them even if the notebook doesn't use them all.
    override_params = overrides.get("parameters", {})
    parameters = {**override_params, **_QUICK_PARAMS}

    # Latent factor models need a wider cross-section for factor extraction
    stage_match_p = _STAGE_RE.match(stage)
    suffix_p = stage[len(stage_match_p.group(0)) :] if stage_match_p else stage
    if suffix_p in _LATENT_FACTOR_SUFFIXES:
        parameters.update(_LATENT_FACTOR_OVERRIDES)

    # Sparse-data case studies (monthly frequency) need more symbols
    if case_study in _SPARSE_DATA_CASE_STUDIES:
        parameters.update(_SPARSE_DATA_OVERRIDES)

    default_timeout = 600 if suffix_p in _LATENT_FACTOR_SUFFIXES else 300
    timeout = overrides.get("timeout", default_timeout)

    # --- Snapshot registry state before run ---
    registry_db = isolated_model_output / case_study / "run_log" / "registry.db"
    before = _registry_summary(registry_db)

    # --- Execute ---
    result = run_notebook(
        py_path=notebook_path,
        parameters=parameters,
        timeout=timeout,
        output_dir=isolated_model_output,
        log_path=LOG_PATH,
    )

    assert result["status"] == "ok", (
        f"\n{'=' * 70}\nFAILED: {case_study}::{stage}\n{'=' * 70}\n{result['error']}\n{'=' * 70}"
    )

    # --- Registry assertions (for notebooks that register) ---
    after = _registry_summary(registry_db)

    # Check if this notebook is expected to register (match on suffix)
    stage_match = _STAGE_RE.match(stage)
    suffix = stage[len(stage_match.group(0)) :] if stage_match else stage
    expects_registration = suffix in _REGISTERING_SUFFIXES

    if expects_registration:
        new_training = after["training_runs"] - before["training_runs"]
        new_predictions = after["prediction_sets"] - before["prediction_sets"]

        # Check for new entries OR updated entries (upserts).
        # Some notebooks (e.g. 12_pca) re-register configs that were
        # already created by an earlier notebook (11_latent_factors),
        # resulting in upserts with 0 net new rows but updated entry_points.
        expected_ep = _expected_entry_point(stage)
        runs = _query_registry(
            registry_db,
            "training_runs",
            f"entry_point = '{expected_ep}'",
        )

        if new_training > 0:
            assert new_predictions > 0, (
                f"{case_study}::{stage} created {new_training} training_runs "
                f"but 0 new prediction_sets"
            )
            print(
                f"\n  Registry OK: +{new_training} training_runs, "
                f"+{new_predictions} prediction_sets"
            )
        elif len(runs) > 0:
            print(
                f"\n  Registry OK: {len(runs)} training_runs with "
                f"entry_point='{expected_ep}' (upserted, no net new rows)"
            )
        else:
            # Neither new entries nor matching entry_points — real failure
            msg = (
                f"{case_study}::{stage} has register=True but created "
                f"0 new training_runs and found 0 with "
                f"entry_point='{expected_ep}' (total: {after['training_runs']})"
            )
            raise AssertionError(msg)
    else:
        # Non-registering notebook — just report what happened
        new_training = after["training_runs"] - before["training_runs"]
        if new_training > 0:
            print(
                f"\n  Note: {stage} created {new_training} training_runs "
                f"(not in _REGISTERING_NOTEBOOKS set — consider adding)"
            )
        else:
            print("\n  OK (no registry writes expected)")
