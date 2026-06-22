from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _sync_if_needed(py_path: Path, sync_policy: str) -> Path:
    ipynb_path = py_path.with_suffix(".ipynb")
    should_sync = sync_policy == "always" or (sync_policy == "missing" and not ipynb_path.exists())
    if should_sync:
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
                str(ipynb_path),
                str(py_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(
                py_path.parent.parent
                if py_path.parent.name.startswith("case_studies")
                else py_path.parent
            ),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Jupytext sync failed for {py_path}: {result.stderr}")
    if not ipynb_path.exists():
        raise FileNotFoundError(f"Expected .ipynb not found for {py_path}")
    return ipynb_path


def _run_full_notebook(
    py_path: Path,
    timeout: int,
    output_dir: Path | None,
    data_dir: Path | None,
    extra_env: dict[str, str],
    sync_policy: str,
) -> dict:
    import papermill as pm

    start = time.perf_counter()
    ipynb_path = _sync_if_needed(py_path, sync_policy)
    tmp_out = Path(tempfile.gettempdir()) / f"ml4t-full-{os.getpid()}-{py_path.stem}.ipynb"

    saved_env: dict[str, str | None] = {}
    rc_dir = Path(tempfile.mkdtemp(prefix="ml4t-mplrc-"))
    rc_file = rc_dir / "matplotlibrc"
    rc_file.write_text("figure.constrained_layout.use: False\n", encoding="utf-8")
    env_vars = {
        "MPLBACKEND": "Agg",
        "PLOTLY_RENDERER": "json",
        "PYTHONUNBUFFERED": "1",
        "MATPLOTLIBRC": str(rc_file),
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        env_vars["ML4T_OUTPUT_DIR"] = str(output_dir)
    if data_dir:
        env_vars["ML4T_DATA_PATH"] = str(data_dir)
    env_vars.update({key: str(value) for key, value in extra_env.items()})

    existing = os.environ.get("PYTHONPATH", "")
    repo_root = py_path.parents[2] if "case_studies" in py_path.parts else py_path.parent.parent
    nb_dir = str(py_path.parent.resolve())
    env_vars["PYTHONPATH"] = (
        f"{repo_root}:{nb_dir}:{existing}" if existing else f"{repo_root}:{nb_dir}"
    )

    try:
        import torch

        torch_lib = str(Path(torch.__file__).parent / "lib")
        nvidia_libs = list((Path(torch.__file__).parent.parent / "nvidia").glob("*/lib"))
        cuda_paths = [torch_lib] + [str(p) for p in nvidia_libs]
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        env_vars["LD_LIBRARY_PATH"] = ":".join(cuda_paths + [existing_ld])
    except ImportError:
        pass

    for key, value in env_vars.items():
        saved_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        pm.execute_notebook(
            str(ipynb_path),
            str(tmp_out),
            parameters={},
            cwd=str(repo_root),
            kernel_name="python3",
            execution_timeout=timeout,
            request_save_on_cell_execute=True,
            progress_bar=False,
            log_output=True,
        )
        shutil.copy2(tmp_out, ipynb_path)
        return {"status": "ok", "error": None, "runtime_seconds": time.perf_counter() - start}
    except pm.PapermillExecutionError as exc:
        return {
            "status": "error",
            "error": f"Cell {exc.cell_index} ({exc.ename}): {exc.evalue}",
            "runtime_seconds": time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "runtime_seconds": time.perf_counter() - start,
        }
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tmp_out.unlink(missing_ok=True)
        shutil.rmtree(rc_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--parameters-json", default="{}")
    parser.add_argument("--env-json", default="{}")
    parser.add_argument("--execution-mode", choices=["reduced", "full"], default="reduced")
    parser.add_argument("--sync-policy", choices=["always", "missing", "never"], default="always")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    result_file = Path(args.result_file).resolve()
    params = json.loads(args.parameters_json)
    extra_env = json.loads(args.env_json)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else None

    started = time.perf_counter()
    if args.execution_mode == "full":
        result = _run_full_notebook(
            py_path=path,
            timeout=args.timeout,
            output_dir=output_dir,
            data_dir=data_dir,
            extra_env=extra_env,
            sync_policy=args.sync_policy,
        )
    else:
        from tests.pm_helpers import run_notebook

        result = run_notebook(
            py_path=path,
            parameters=params,
            timeout=args.timeout,
            output_dir=output_dir,
            data_dir=data_dir,
            extra_env=extra_env,
        )
    elapsed = time.perf_counter() - started

    payload = {
        "status": result.get("status", "error"),
        "error": result.get("error"),
        "runtime_seconds": elapsed,
    }
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
