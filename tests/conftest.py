"""Pytest fixtures for ML4T test suite.

Two modes of operation:
1. CI (GHA): ML4T_DATA_PATH points to pre-subsampled real data (from private repo).
   populated_data_dir just returns that path — no synthetic data needed.
2. Local dev: ML4T_DATA_PATH points to full production data or test data.
"""

import json
import os
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from tests.preset_patches import _patch_presets_for_testing, _trim_label_configs

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


# Set by ``seeded_output_dir`` so the per-test restore below knows the value the
# session deliberately installed, rather than reverting it.
_SESSION_OUTPUT_DIR: str | None = None


@pytest.fixture(autouse=True)
def restore_output_root():
    """No test leaves ML4T_OUTPUT_DIR pointing at its own tmp_path.

    ``Study.activate()`` sets ML4T_OUTPUT_DIR for the whole process and caches the
    root it installed, which is what a notebook wants and what makes the test corpus
    order-dependent: the last test to activate leaves every later test resolving
    ``config/setup.yaml`` under a tmp_path pytest has since deleted.

    Measured on 2026-08-24: ``tests/test_sweep_config_seam.py`` passes 22 on its own
    and fails 19 when it runs after ``tests/test_sp500_options_research.py``, on
    ``FileNotFoundError: .../test_official_population_is_sn0/workspace/
    us_firm_characteristics/config/setup.yaml``. Six test files already carry a
    module-level ``os.environ.pop("ML4T_OUTPUT_DIR", None)`` against this; each one
    protects only the file that remembers to write it, and only against a leak from
    a file that ran earlier.

    The module-level cache has to be cleared with the variable. ``activate()`` skips
    the write when the root it is asked for equals the one it last installed, so
    restoring the environment alone leaves the next activation of the same root
    silently doing nothing.
    """
    before = os.environ.get("ML4T_OUTPUT_DIR")
    yield
    target = _SESSION_OUTPUT_DIR if _SESSION_OUTPUT_DIR is not None else before
    if os.environ.get("ML4T_OUTPUT_DIR") == target:
        return
    if target is None:
        os.environ.pop("ML4T_OUTPUT_DIR", None)
    else:
        os.environ["ML4T_OUTPUT_DIR"] = target
    # Only if the workspace module was imported: touching it otherwise would pull the
    # research package into every test in the corpus.
    workspace = sys.modules.get("case_studies.research.workspace")
    if workspace is not None:
        workspace._ACTIVE_OUTPUT_ROOT = None
        workspace._clear_root_sensitive_caches()


def generated_env_contents(repo_root: Path, environ: Mapping[str, str]) -> str:
    """What a generated ``.env`` should contain, given the environment.

    Carries only a data path someone actually chose. ``sitecustomize.py`` sets
    ML4T_DATA_PATH to ``<repo>/data`` when nothing else did, and marks it as a
    default; writing that into ``.env`` would promote the default to an explicit
    setting that step 2 of ``_resolve_data_path()`` returns — shadowing the
    populated test-data checkout and silently skipping every data-dependent
    notebook test on a clean clone.
    """
    lines = [f"ML4T_PATH={repo_root}\n"]
    data_path = environ.get("ML4T_DATA_PATH")
    if data_path and not environ.get("ML4T_DATA_PATH_IS_DEFAULT"):
        lines.append(f"ML4T_DATA_PATH={data_path}\n")
    return "".join(lines)


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
        env_file.write_text(generated_env_contents(REPO_ROOT, os.environ))
        created = True

    yield

    # Clean up CI-created .env (don't leave artifacts)
    if created and env_file.exists():
        env_file.unlink()


def _holds_datasets(path: Path) -> bool:
    """True where a data directory holds parquet at either documented depth.

    The same test ``sitecustomize._has_datasets`` applies, and for the same
    reason: ``data/`` is tracked for its ``config.yaml`` and ``download.py``
    while every parquet under it is gitignored, so existing and being non-empty
    say nothing about whether there is data to read.
    """
    return any(next(path.glob(p), None) for p in ("*/*.parquet", "*/*/*.parquet"))


def _resolve_data_path() -> Path | None:
    """Find ML4T_DATA_PATH from env var, .env file, or default location.

    pytest-xdist workers may not inherit env vars set by the parent process,
    so we also check the .env file and well-known test-data locations.
    """
    # 1. Explicit env var (works in single-process pytest and CI).
    #    sitecustomize.py sets ML4T_DATA_PATH to <repo>/data when nothing else
    #    did, and marks it. That default must not win here: the tracked data/
    #    tree is never empty, so taking it would shadow the populated test-data
    #    checkout below and silently skip every data-dependent notebook test.
    #    Step 4 applies the real test — does it hold parquet — to that path.
    env_path = os.environ.get("ML4T_DATA_PATH")
    if env_path and not os.environ.get("ML4T_DATA_PATH_IS_DEFAULT"):
        p = Path(env_path).expanduser().resolve()
        if p.exists() and any(p.iterdir()):
            return p

    # 2. Read from .env file (works in xdist workers).
    #    Non-empty is not the test here either, for the same reason it is not in
    #    step 1: a .env naming the repo's own data/ passes it on the strength of
    #    the tracked skeleton, which reintroduces exactly what the marker above
    #    and the glob in step 4 both exist to block. CI writes such a .env by
    #    hand, so this branch is the one that fires there.
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ML4T_DATA_PATH") and "=" in line:
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val and not val.startswith("#"):
                    p = Path(val).expanduser().resolve()
                    if _holds_datasets(p):
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


def _resolve_intermediates_root() -> Path | None:
    """Where the pre-computed pipeline intermediates are, or ``None``.

    Two candidates, in order, because the two places this runs disagree on both:
    beside the resolved data path (the test-data repo layout, which is what CI has -
    the checkout lands under the workspace), then the well-known workstation path.
    Taking only the first misses a workstation whose ``ML4T_DATA_PATH`` points at the
    canonical Dropbox data root, whose parent holds no intermediates; taking only the
    second misses every CI runner. The seeding fixture below has always tried both;
    ``intermediates_dir`` tried only the first, which is why the tests that consume it
    skipped everywhere.
    """
    data_path = _resolve_data_path()
    candidates = []
    if data_path:
        candidates.append(Path(data_path).parent / "intermediates")
    candidates.append(Path.home() / "ml4t" / "test-data" / "intermediates")
    for candidate in candidates:
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    return None


@pytest.fixture(scope="session")
def intermediates_dir(test_data_dir):
    """Return directory with pre-computed pipeline intermediates.

    When running downstream chapters (Ch11+), they need labels/features
    from pipeline stages. These are pre-computed and stored in test-data repo.

    ``test_data_dir`` is requested for its side effect - it puts the resolved data
    path in ``ML4T_DATA_PATH`` for the rest of the session - not for a skip, which it
    does not do (``populated_data_dir`` is the fixture that skips). The path itself
    comes from ``_resolve_intermediates_root``, which the seeding fixture uses too.
    """
    return _resolve_intermediates_root()


SEEDED_SUBDIRS = ("features", "labels", "evaluation", "run_log", "results", "benchmark")


def seed_case_study_intermediates(src: Path, dst: Path) -> None:
    """Make ``dst`` hold the fixture's intermediates for one case study, and only those.

    Replace, never fill a gap. CI checks the fixture out into a clean container every
    time; a lane re-running into an existing ``ML4T_OUTPUT_DIR`` used to keep whatever
    the previous run had left, which for ``run_log/registry.db`` means every training
    run, prediction set and backtest that run registered. The notebooks then resolved
    identities neither the fixture nor production carries, so the same code produced a
    different failure list locally than it did in CI.

    A destination subdir is removed whether or not the fixture still ships a source for
    it. One the fixture has stopped shipping is precisely the leftover nothing would
    ever overwrite.

    ``src`` also holds top-level files (``etfs/eligibility.csv``, ``protocol.yaml``,
    ``baseline_checkpoint.yaml``) that downstream notebooks read through
    ``get_case_study_dir()``, so they are copied too.
    """
    for subdir in SEEDED_SUBDIRS:
        src_sub = src / subdir
        dst_sub = dst / subdir
        if dst_sub.exists():
            shutil.rmtree(dst_sub)
        if src_sub.exists():
            shutil.copytree(src_sub, dst_sub)

    for item in src.iterdir():
        if item.is_file():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst / item.name)


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
    global _SESSION_OUTPUT_DIR
    os.environ["ML4T_OUTPUT_DIR"] = str(output_dir)
    _SESSION_OUTPUT_DIR = str(output_dir)

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
    intermediates_root = _resolve_intermediates_root()
    if intermediates_root and intermediates_root.exists():
        for cs_id in CASE_STUDY_IDS:
            src = intermediates_root / cs_id
            if not src.exists():
                continue
            dst = output_dir / cs_id
            seed_case_study_intermediates(src, dst)
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
            if dst.exists():
                shutil.rmtree(dst)
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


# _patch_presets_for_testing and _trim_label_configs (imported above), and the
# _TEST_PRESET_PATCHES table the first reads, live in tests/preset_patches.py,
# which tests/generate_intermediates.py also imports - that script runs
# standalone without pytest installed, so they can't live in this module.
# _trim_label_configs was duplicated here until 2026-07-30; the copy in
# generate_intermediates.py globbed the wrong directory and trimmed nothing.

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


@pytest.fixture(autouse=True)
def _restore_output_root():
    """Put back the process-global output root a test may have redirected.

    ``Study.open``/``Study.activate`` set ``ML4T_OUTPUT_DIR`` and
    ``workspace._ACTIVE_OUTPUT_ROOT``, and a test that leaves either pointing into its own
    tmp_path makes ``get_case_study_dir`` resolve there for every later test in the worker.

    Restored rather than deleted: ``seeded_output_dir`` is session-scoped and writes the
    per-worker path exactly once, so a teardown that removes the variable removes it for the
    rest of the worker, and every later module resolving through ``get_case_study_dir`` then
    reads the committed ``case_studies/`` tree instead of the seeded output dir. This lives here
    rather than in each module because thirteen copies of it drifted apart once already.
    """
    previous = os.environ.get("ML4T_OUTPUT_DIR")
    yield
    if previous is None:
        os.environ.pop("ML4T_OUTPUT_DIR", None)
    else:
        os.environ["ML4T_OUTPUT_DIR"] = previous
    try:
        from case_studies.research import workspace
    except ImportError:
        return
    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


@pytest.fixture
def clean_env():
    """Fixture that provides a clean environment and restores it after."""
    saved_env = os.environ.copy()
    yield os.environ
    os.environ.clear()
    os.environ.update(saved_env)
