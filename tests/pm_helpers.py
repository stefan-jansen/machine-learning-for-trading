"""Papermill-based notebook execution helpers.

Provides:
- run_notebook(): Execute a .py notebook via Papermill with parameter injection
- get_overrides(): Load per-notebook overrides from tests/overrides.yaml
- collect_chapter_notebooks(): Discover notebooks in chapter directories
- get_tier() / current_test_tier(): Test-tier routing (per-commit / weekly / on-demand)
- get_record_mode(): VCR cassette mode (consumed by Step 5)

NOTE: Notebooks live directly in chapter directories
(e.g., 05_synthetic_data/01_timegan.py), NOT in code/ subdirs.

overrides.yaml schema (per-notebook, all optional):
    timeout: int (seconds, default 300)
    parameters: dict (papermill -p overrides)
    skip: bool — hard skip in uv-native run (Docker tests ignore)
    skip_reason: str
    requires_import: str | list[str]
    gpu: bool
    long_running: bool
    docker_env: str — informational (e.g., "benchmark")
    tier: "per_commit" | "weekly" | "on_demand" — default "per_commit"
        Per-commit runs the Tests workflow on every PR/push.
        Weekly runs the weekly-external scheduled workflow (Step 2).
        On_demand runs only on manual dispatch (GPU-only NBs).
    reruns: int — flaky-retry count (consumed once pytest-rerunfailures lands,
        Step 2). Default 0.
    record_mode: "replay" | "rewrite" — VCR cassette mode (consumed by Step 5).
        Default "replay".
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
OVERRIDES_PATH = REPO_ROOT / "tests" / "overrides.yaml"

# Cache loaded overrides
_overrides_cache: dict | None = None

# ---------------------------------------------------------------------------
# Test tier — controls when a notebook runs in CI.
# Per-commit (default): every PR / push triggers the Tests workflow.
# Weekly: only the scheduled weekly-external workflow (Mon 06:00 UTC).
# On-demand: only manual workflow_dispatch (e.g., GPU-only Tier 3).
# ---------------------------------------------------------------------------
TIER_PER_COMMIT = "per_commit"
TIER_WEEKLY = "weekly"
TIER_ON_DEMAND = "on_demand"
VALID_TIERS = frozenset({TIER_PER_COMMIT, TIER_WEEKLY, TIER_ON_DEMAND})

# VCR cassette modes (Step 5: pytest-recording).
RECORD_REPLAY = "replay"
RECORD_REWRITE = "rewrite"
VALID_RECORD_MODES = frozenset({RECORD_REPLAY, RECORD_REWRITE})


def get_tier(overrides: dict) -> str:
    """Return the test tier declared by overrides (default: per_commit)."""
    tier = overrides.get("tier") or TIER_PER_COMMIT
    if tier not in VALID_TIERS:
        raise ValueError(
            f"Invalid tier {tier!r} in overrides — must be one of {sorted(VALID_TIERS)}"
        )
    return tier


def current_test_tier() -> str:
    """Return the tier the current pytest run is targeting.

    Read from ML4T_TEST_TIER env var; defaults to per_commit so existing
    workflows that don't set it keep their current behavior (only NBs
    without a tier key — i.e., tier=per_commit — execute).
    """
    tier = os.environ.get("ML4T_TEST_TIER") or TIER_PER_COMMIT
    if tier not in VALID_TIERS:
        raise ValueError(f"Invalid ML4T_TEST_TIER={tier!r} — must be one of {sorted(VALID_TIERS)}")
    return tier


def get_reruns(overrides: dict) -> int:
    """Return per-notebook flaky retry count (default 0).

    Consumed by Step 2 (pytest-rerunfailures dep + collection hook adds
    @pytest.mark.flaky(reruns=N) when N > 0). Until that lands the value
    is parsed but no retries happen.
    """
    val = overrides.get("reruns", 0)
    if not isinstance(val, int) or val < 0:
        raise ValueError(f"Invalid reruns={val!r} — must be non-negative int")
    return val


def get_record_mode(overrides: dict) -> str:
    """Return VCR cassette mode for the notebook (default: replay)."""
    mode = overrides.get("record_mode") or RECORD_REPLAY
    if mode not in VALID_RECORD_MODES:
        raise ValueError(
            f"Invalid record_mode {mode!r} — must be one of {sorted(VALID_RECORD_MODES)}"
        )
    return mode


def get_overrides(notebook_key: str) -> dict:
    """Get parameter overrides for a notebook from tests/overrides.yaml.

    Args:
        notebook_key: Notebook path relative to repo root, no extension.
                      e.g., "05_synthetic_data/02_tailgan_tail_risk"

    Returns:
        Dict with optional keys: timeout, gpu, parameters
    """
    global _overrides_cache
    if _overrides_cache is None:
        if OVERRIDES_PATH.exists():
            with open(OVERRIDES_PATH) as f:
                _overrides_cache = yaml.safe_load(f) or {}
        else:
            _overrides_cache = {}

    return _overrides_cache.get(notebook_key) or {}


def sync_notebook(py_path: Path) -> Path:
    """Sync a .py notebook to a temporary .ipynb via Jupytext.

    Writes to a temp file so the real .ipynb (which may contain pre-executed
    outputs) is never overwritten.

    Args:
        py_path: Path to the .py source file

    Returns:
        Path to a temporary .ipynb file (caller must clean up)
    """
    # Write to a temp file — never touch the real .ipynb
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".ipynb", prefix=f"_pm_{py_path.stem}_")
    os.close(tmp_fd)
    tmp_ipynb = Path(tmp_path_str)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupytext",
            "--to",
            "notebook",
            "--set-kernel",
            "python3",
            "--output",
            str(tmp_ipynb),
            str(py_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        tmp_ipynb.unlink(missing_ok=True)
        raise RuntimeError(f"Jupytext sync failed for {py_path}: {result.stderr}")

    if not tmp_ipynb.exists():
        raise FileNotFoundError(f"Expected temp .ipynb not found after sync: {tmp_ipynb}")

    return tmp_ipynb


def run_notebook(
    py_path: Path,
    parameters: dict | None = None,
    timeout: int = 300,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
    log_path: Path | None = None,
    cwd: Path | None = None,
) -> dict:
    """Execute a notebook via Papermill with parameter injection.

    This is the core test helper. It:
    1. Syncs .py -> .ipynb via Jupytext
    2. Executes via Papermill with parameter overrides
    3. Logs per-cell progress to log_path (if provided)
    4. Returns status and error info

    Args:
        py_path: Path to the .py notebook source
        parameters: Dict of parameters to inject (overrides defaults in parameters cell)
        timeout: Per-cell timeout in seconds
        output_dir: Directory for ML4T_OUTPUT_DIR (redirects saves to temp)
        data_dir: Directory for ML4T_DATA_PATH (test data location)
        extra_env: Additional environment variables for notebook execution
        log_path: Path to progress log file (appended to)

    Returns:
        Dict with keys: status ("ok" or "error"), error (str if failed),
        duration_s (float), n_cells (int)
    """
    import time

    import papermill as pm

    start = time.time()
    nb_name = py_path.stem

    def _log(msg: str) -> None:
        if log_path:
            with open(log_path, "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                f.flush()

    _log(f"START {nb_name} (timeout={timeout}s)")

    # Sync to a temp .ipynb (never overwrites the real .ipynb)
    tmp_ipynb: Path | None = None
    try:
        tmp_ipynb = sync_notebook(py_path)
    except (RuntimeError, FileNotFoundError) as e:
        _log(f"SYNC_FAIL {nb_name}: {e}")
        return {
            "status": "error",
            "error": f"Jupytext sync failed: {e}",
            "duration_s": time.time() - start,
            "n_cells": 0,
        }

    ipynb_path = tmp_ipynb

    # Executed notebook output path
    executed_path = py_path.parent / f"_executed_{py_path.stem}.ipynb"

    # Setup environment - Papermill's execute_notebook inherits os.environ,
    # so we temporarily set env vars and restore them after execution.
    saved_env = {}
    env_vars = {
        "MPLBACKEND": "Agg",
        "PLOTLY_RENDERER": "json",
        "DISABLE_HPO": "1",
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        env_vars["ML4T_OUTPUT_DIR"] = str(output_dir)
    if data_dir:
        env_vars["ML4T_DATA_PATH"] = str(data_dir)
    if extra_env:
        env_vars.update({key: str(value) for key, value in extra_env.items()})

    # PYTHONPATH includes repo root for utils imports + notebook dir for sibling imports
    existing = os.environ.get("PYTHONPATH", "")
    nb_dir = str(py_path.parent.resolve())
    env_vars["PYTHONPATH"] = (
        f"{REPO_ROOT}:{nb_dir}:{existing}" if existing else f"{REPO_ROOT}:{nb_dir}"
    )

    # Ensure torch's bundled CUDA libraries are found before system ones.
    # The system libcudart.so.12 may be outdated and missing symbols like
    # cudaGetDriverEntryPointByVersion that torch's bundled version provides.
    try:
        import torch

        torch_lib = str(Path(torch.__file__).parent / "lib")
        nvidia_libs = list((Path(torch.__file__).parent.parent / "nvidia").glob("*/lib"))
        cuda_paths = [torch_lib] + [str(p) for p in nvidia_libs]
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        env_vars["LD_LIBRARY_PATH"] = ":".join(cuda_paths + [existing_ld])
    except ImportError:
        pass

    remove_vars = ["TEST", "QUICK_TEST"]
    if extra_env:
        for key in extra_env:
            if key in remove_vars:
                remove_vars.remove(key)

    # Apply environment changes
    for key, value in env_vars.items():
        saved_env[key] = os.environ.get(key)
        os.environ[key] = value
    for key in remove_vars:
        saved_env[key] = os.environ.pop(key, None)

    # Cell-level progress — always log to /tmp/ml4t-pm-{name}.log for visibility.
    # Since request_save_on_cell_execute=True, the executed notebook is updated
    # after each cell. Monitor it with: watch -n5 'python -c "import json; ..."'
    progress_log = Path(f"/tmp/ml4t-pm-{nb_name}.log")
    n_cells = 0
    try:
        with open(progress_log, "w") as pf:
            pf.write(f"START {nb_name} timeout={timeout}s params={parameters}\n")

        pm.execute_notebook(
            str(ipynb_path),
            str(executed_path),
            parameters=parameters or {},
            cwd=str(cwd or REPO_ROOT),
            kernel_name="python3",
            execution_timeout=timeout,
            request_save_on_cell_execute=True,
            progress_bar=False,
            log_output=True,
        )
        # Count cells in executed notebook
        try:
            import nbformat

            nb = nbformat.read(str(executed_path), as_version=4)
            n_cells = len([c for c in nb.cells if c.cell_type == "code"])
        except Exception:
            pass

        elapsed = time.time() - start
        msg = f"OK    {nb_name} ({elapsed:.1f}s, {n_cells} cells)"
        _log(msg)
        with open(progress_log, "a") as pf:
            pf.write(f"{msg}\n")
        return {"status": "ok", "error": None, "duration_s": elapsed, "n_cells": n_cells}
    except pm.PapermillExecutionError as e:
        elapsed = time.time() - start
        msg = f"FAIL  {nb_name} cell {e.cell_index} ({e.ename}): {e.evalue} ({elapsed:.1f}s)"
        _log(msg)
        with open(progress_log, "a") as pf:
            pf.write(f"{msg}\n")
        return {
            "status": "error",
            "error": f"Cell {e.cell_index} ({e.ename}): {e.evalue}",
            "duration_s": elapsed,
            "n_cells": e.cell_index,
        }
    except Exception as e:
        elapsed = time.time() - start
        _log(f"FAIL  {nb_name}: {e} ({elapsed:.1f}s)")
        return {"status": "error", "error": str(e), "duration_s": elapsed, "n_cells": 0}
    finally:
        # Restore environment
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # Clean up temp input notebook
        if tmp_ipynb is not None and tmp_ipynb.exists():
            tmp_ipynb.unlink()
        # Clean up executed notebook
        if executed_path.exists():
            executed_path.unlink()


def collect_chapter_notebooks(repo_root: Path, chapter_range: range) -> list[Path]:
    """Collect all teaching notebooks from chapter directories.

    NOTE: Review repo has flat layout — notebooks live directly in
    chapter directories (e.g., 05_synthetic_data/01_timegan.py),
    NOT in code/ subdirectories.

    Args:
        repo_root: Repository root path
        chapter_range: Range of chapter numbers to include

    Returns:
        Sorted list of .py notebook paths
    """
    notebooks = []
    for chapter_dir in sorted(repo_root.glob("[0-9][0-9]_*")):
        if not chapter_dir.is_dir():
            continue

        # Extract chapter number from directory name
        try:
            ch_num = int(chapter_dir.name[:2])
        except ValueError:
            continue

        if ch_num not in chapter_range:
            continue

        # Review repo: notebooks are directly in the chapter directory
        for notebook in sorted(chapter_dir.glob("*.py")):
            # Skip non-notebook files — use startswith to avoid false positives
            # (e.g., "test_" must not match "backtest_")
            if any(
                notebook.name.startswith(prefix)
                for prefix in [
                    "test_",
                    "conftest",
                    "extract_book_figures",
                    "export_figures",
                    "batch_",
                ]
            ) or any(x in notebook.name for x in ["__pycache__", "__init__"]):
                continue

            # Skip archived/draft/reserved directories
            if any(
                x in str(notebook)
                for x in ["_archive", "archived", "drafts", "inventory", "_reserved"]
            ):
                continue

            # Skip helper/utility files (start with _)
            if notebook.name.startswith("_"):
                continue

            if not notebook.name[0].isdigit() and not notebook.with_suffix(".ipynb").exists():
                continue

            notebooks.append(notebook)

    return notebooks
