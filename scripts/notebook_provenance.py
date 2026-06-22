"""Provenance stamp + sync gate for paired ``.py``/``.ipynb`` notebooks.

A committed ``.ipynb`` should be the *current* ``.py`` executed in a real
environment with production parameters — not a stale render, not a TEST-mode run,
not a run in an environment missing a dependency (e.g. CUDA-LightGBM). This module
stamps that fact into the notebook and provides a gate that rejects violations, so
"edited the ``.py``, ran TEST or the wrong env, committed a stale ``.ipynb``" is
caught mechanically instead of by review.

The stamp lives in ``nb.metadata["ml4t_provenance"]``::

    source_py_blob : git blob hash of the paired .py at execution time
    executed_at    : ISO-8601 timestamp
    executor       : environment label (e.g. "ml4t-gpu", "local-uv")
    production     : bool — True iff no papermill parameter overrides were injected
    parameters     : the papermill parameter overrides ({} for a production run)
    notes          : optional free text (e.g. "GPU libs: xgboost,lightgbm,catboost")

Gate (``check``): for every tracked ``.ipynb`` that HAS a stamp,

* ``source_py_blob`` must equal ``git hash-object`` of the current paired ``.py``
  (else the ``.py`` changed since the notebook was executed — STALE), and
* ``production`` must be True (else a TEST-mode run was committed).

Notebooks WITHOUT a stamp are reported as "unverified" but do not fail unless
``--strict`` is passed. This is deliberate: adoption is gradual — stamp notebooks
as they are re-run through the canonical path, and the gate enforces only where
provenance exists. Flip to ``--strict`` once the backfill is complete.

Usage::

    uv run python scripts/notebook_provenance.py stamp <nb.ipynb> --executor ml4t-gpu
    uv run python scripts/notebook_provenance.py stamp <nb.ipynb> --executor ml4t-gpu --notes "..."
    uv run python scripts/notebook_provenance.py check          # gate (stamped-only)
    uv run python scripts/notebook_provenance.py check --strict  # also fail on unverified
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_PARTS = {"_reference", ".venv", ".git", ".ipynb_checkpoints"}
STAMP_KEY = "ml4t_provenance"


def iter_notebooks() -> list[Path]:
    out = []
    for p in REPO_ROOT.rglob("*.ipynb"):
        if SKIP_PARTS & set(p.parts):
            continue
        if p.name.startswith("_executed_") or p.name.startswith("_lock_"):
            continue
        out.append(p)
    return sorted(out)


def paired_py(nb_path: Path) -> Path | None:
    """The .py jupytext-paired to this notebook (same dir + stem). None if absent."""
    cand = nb_path.with_suffix(".py")
    return cand if cand.exists() else None


def git_blob(path: Path) -> str:
    """git blob SHA-1 of the file's current content (working tree)."""
    return subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def stamp_notebook(nb_path: Path, executor: str, notes: str | None = None) -> dict:
    py = paired_py(nb_path)
    if py is None:
        raise SystemExit(f"no paired .py for {nb_path.relative_to(REPO_ROOT)} — cannot stamp")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    params = nb.get("metadata", {}).get("papermill", {}).get("parameters", {}) or {}
    stamp = {
        "source_py_blob": git_blob(py),
        "executed_at": datetime.now(UTC).isoformat(),
        "executor": executor,
        "production": not params,
        "parameters": params,
    }
    if notes:
        stamp["notes"] = notes
    nb.setdefault("metadata", {})[STAMP_KEY] = stamp
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return stamp


def check_all(strict: bool = False) -> tuple[list[str], list[str], list[str]]:
    """Return (stale, testmode, unverified) lists of repo-relative offenders."""
    stale: list[str] = []
    testmode: list[str] = []
    unverified: list[str] = []
    for nb_path in iter_notebooks():
        rel = str(nb_path.relative_to(REPO_ROOT))
        py = paired_py(nb_path)
        if py is None:
            continue  # un-paired notebooks have no .py to drift from
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        stamp = nb.get("metadata", {}).get(STAMP_KEY)
        if not stamp:
            unverified.append(rel)
            continue
        if stamp.get("source_py_blob") != git_blob(py):
            stale.append(rel)
        if not stamp.get("production", False):
            testmode.append(f"{rel} (params={stamp.get('parameters')})")
    return stale, testmode, unverified


def _cmd_stamp(args: argparse.Namespace) -> int:
    s = stamp_notebook(Path(args.notebook).resolve(), args.executor, args.notes)
    print(
        f"stamped {args.notebook}: source_py_blob={s['source_py_blob'][:12]} "
        f"executor={s['executor']} production={s['production']}"
    )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    stale, testmode, unverified = check_all(strict=args.strict)
    fail = bool(stale or testmode) or (args.strict and bool(unverified))
    if stale:
        print(
            "STALE (paired .py changed since the notebook was executed — re-run in the canonical env):"
        )
        for r in stale:
            print(f"  {r}")
    if testmode:
        print(
            "TEST-MODE (committed a run with papermill parameter overrides — must be production):"
        )
        for r in testmode:
            print(f"  {r}")
    if unverified:
        verb = "UNVERIFIED (no provenance stamp"
        verb += (
            " — FAILING under --strict):" if args.strict else " — advisory, backfill over time):"
        )
        print(verb)
        for r in unverified:
            print(f"  {r}")
    if not fail:
        print(
            f"notebook sync OK: {len(stale)} stale, {len(testmode)} test-mode, "
            f"{len(unverified)} unverified (advisory)"
        )
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("stamp", help="write a provenance stamp into a notebook")
    sp.add_argument("notebook")
    sp.add_argument("--executor", required=True, help="environment label, e.g. ml4t-gpu / local-uv")
    sp.add_argument("--notes", default=None)
    sp.set_defaults(func=_cmd_stamp)

    cp = sub.add_parser("check", help="gate: fail on stale or test-mode stamped notebooks")
    cp.add_argument("--strict", action="store_true", help="also fail on unstamped notebooks")
    cp.set_defaults(func=_cmd_check)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
