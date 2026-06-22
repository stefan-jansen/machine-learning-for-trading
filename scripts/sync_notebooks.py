"""Sync paired Jupyter notebooks (.ipynb) from their Jupytext .py sources.

The repository stores notebooks in Jupytext percent format: the `.py` file is
the source of truth, the `.ipynb` is a generated artifact. If a reviewer opens
an `.ipynb` and finds it out of date relative to the `.py`, they can run this
script to regenerate every `.ipynb` from its paired `.py`.

Usage (from the repo root):
    uv run python scripts/sync_notebooks.py            # forward sync (.py → .ipynb)
    uv run python scripts/sync_notebooks.py --check    # report drift only
    uv run python scripts/sync_notebooks.py --safe-only  # sync only doc-only drift
    uv run python scripts/sync_notebooks.py 19_risk_management   # one directory

Drift detection is content-based — `.py` and `.ipynb` code cells are compared
directly rather than relying on filesystem mtimes (which are non-deterministic
after `git clone` / `git checkout`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_PARTS = {".venv", ".git", "__pycache__", "archive", "_archive", "node_modules"}


def _is_jupytext_paired_py(py: Path) -> bool:
    """Heuristic: a Jupytext percent `.py` has either a Jupytext header block
    (`# ---` ... `jupytext:` ... `# ---`) or at least one `# %%` cell marker.
    """
    try:
        with py.open("r", encoding="utf-8") as f:
            head = f.read(4096)
    except (OSError, UnicodeDecodeError):
        return False
    return "# %%" in head or "jupytext:" in head


def find_paired(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_PARTS]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            py = Path(dirpath) / name
            ipynb = py.with_suffix(".ipynb")
            if not ipynb.exists():
                continue
            if not _is_jupytext_paired_py(py):
                continue
            pairs.append((py, ipynb))
    return pairs


def ipynb_code_cells(ipynb: Path) -> list[str]:
    """Code-cell sources from the .ipynb, skipping Papermill-injected cells.

    Papermill prepends an `injected-parameters` cell (tag stored in cell
    metadata) when run with `-p`. It's a runtime artifact, not authorial
    content, so we exclude it before comparing against the .py.
    """
    nb = json.loads(ipynb.read_text(encoding="utf-8"))
    cells: list[str] = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        tags = (c.get("metadata") or {}).get("tags") or []
        if "injected-parameters" in tags:
            continue
        cells.append("".join(c.get("source", [])))
    return cells


def py_code_cells(py: Path) -> list[str]:
    """Extract code cells from a Jupytext percent .py file.

    Cells are delimited by `# %%` markers. A cell counts as "code" unless its
    marker line includes `[markdown]` or `[raw]`.
    """
    cells: list[str] = []
    current: list[str] = []
    in_code = False
    for raw in py.read_text(encoding="utf-8").splitlines():
        if raw.startswith("# %%"):
            if in_code and current:
                cells.append("\n".join(current).strip())
            current = []
            in_code = "[markdown]" not in raw and "[raw]" not in raw
            continue
        if in_code:
            current.append(raw)
    if in_code and current:
        cells.append("\n".join(current).strip())
    return [c for c in cells if c]


def code_drift(py: Path, ipynb: Path) -> bool:
    """True if the .py's code cells don't match the .ipynb's embedded code cells.

    Empty cells are ignored on both sides — papermill injects an empty
    "papermill-error-cell" placeholder when execution stops mid-notebook, and
    we don't want to flag that as authorial drift.
    """
    py_cells = [c.strip() for c in py_code_cells(py) if c.strip()]
    nb_cells = [c.strip() for c in ipynb_code_cells(ipynb) if c.strip()]
    return py_cells != nb_cells


def has_outputs(ipynb: Path) -> bool:
    nb = json.loads(ipynb.read_text(encoding="utf-8"))
    return any(c.get("cell_type") == "code" and c.get("outputs") for c in nb.get("cells", []))


@dataclass
class Drift:
    py: Path
    ipynb: Path
    direction: str  # "py_newer" | "ipynb_newer"
    code_diff: bool
    outputs: bool

    @property
    def category(self) -> str:
        if not self.code_diff:
            return "doc_only"
        if self.outputs:
            return "code_drift_with_outputs"
        return "code_drift_no_outputs"


def classify(py: Path, ipynb: Path) -> Drift | None:
    """Content-based classification.

    Drift is determined entirely by source-cell comparison; mtimes are not
    trusted (fresh `git checkout` writes both files at the same time in
    non-deterministic order, and papermill execution bumps `.ipynb` mtime
    without authorial change). When code matches, return None regardless of
    mtime ordering. When code differs, mtime becomes a hint for direction.
    """
    cd = code_drift(py, ipynb)
    if not cd:
        return None

    py_mt = py.stat().st_mtime
    nb_mt = ipynb.stat().st_mtime
    direction = "ipynb_newer" if nb_mt > py_mt else "py_newer"

    return Drift(
        py=py,
        ipynb=ipynb,
        direction=direction,
        code_diff=cd,
        outputs=has_outputs(ipynb),
    )


def sync_one(py: Path) -> tuple[bool, str]:
    # --update preserves existing cell outputs when cell structure matches;
    # falls back to a full regenerate when it can't.
    result = subprocess.run(
        ["uv", "run", "jupytext", "--update", "--to", "ipynb", str(py)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()

    # Jupytext sometimes considers the .ipynb already in sync and leaves the
    # file untouched. Bump the .ipynb mtime to match the .py so later mtime
    # checks recognize them as in sync.
    ipynb = py.with_suffix(".ipynb")
    if ipynb.exists() and ipynb.stat().st_mtime < py.stat().st_mtime:
        py_mt = py.stat().st_mtime
        os.utime(ipynb, (py_mt, py_mt))
    return True, (result.stderr or result.stdout).strip()


def print_group(label: str, drifts: list[Drift]) -> None:
    if not drifts:
        return
    print(f"\n  {label} ({len(drifts)}):")
    for d in drifts:
        arrow = "←" if d.direction == "ipynb_newer" else "→"
        print(f"    {d.py.relative_to(REPO_ROOT)}  [.py {arrow} .ipynb]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=str(REPO_ROOT),
        help="Directory (or single .py / .ipynb file) to scan. Default: repo root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift but do not regenerate.",
    )
    parser.add_argument(
        "--safe-only",
        action="store_true",
        help="Sync only doc-only drift (preserves outputs); skip code-drift entries.",
    )
    args = parser.parse_args()

    target = Path(args.path).resolve()
    if not target.exists():
        print(f"error: {target} does not exist", file=sys.stderr)
        return 2

    if target.is_file():
        py = target.with_suffix(".py")
        ipynb = target.with_suffix(".ipynb")
        if not (py.exists() and ipynb.exists()):
            print(f"error: {target} has no paired .py/.ipynb sibling.", file=sys.stderr)
            return 2
        if not _is_jupytext_paired_py(py):
            print(
                f"error: {py} is not a Jupytext-paired percent script.",
                file=sys.stderr,
            )
            return 2
        pairs = [(py, ipynb)]
    else:
        pairs = find_paired(target)

    if not pairs:
        print(f"No paired .py/.ipynb pairs found under {target}.")
        return 0

    drifts = [d for d in (classify(py, ipynb) for py, ipynb in pairs) if d is not None]
    py_newer = [d for d in drifts if d.direction == "py_newer"]
    ipynb_newer_drift = [d for d in drifts if d.direction == "ipynb_newer"]
    print(
        f"Found {len(pairs)} paired notebook(s); {len(drifts)} with drift "
        f"({len(py_newer)} .py-newer, {len(ipynb_newer_drift)} .ipynb-newer)."
    )

    code_drift_with_outputs = [d for d in py_newer if d.category == "code_drift_with_outputs"]
    code_drift_no_outputs = [d for d in py_newer if d.category == "code_drift_no_outputs"]
    doc_only = [d for d in py_newer if d.category == "doc_only"]

    if args.check:
        print_group("CODE-DRIFT, outputs likely stale (rerun needed)", code_drift_with_outputs)
        print_group("CODE-DRIFT, no outputs (safe to sync forward)", code_drift_no_outputs)
        print_group("DOC-ONLY (safe to sync forward)", doc_only)
        print_group(
            "REVERSE drift (.ipynb newer than .py — manual review!)",
            ipynb_newer_drift,
        )
        return 1 if drifts else 0

    # Sync mode. By default, sync forward (.py → .ipynb) for all py_newer entries.
    # Refuse to overwrite when .ipynb is newer than .py (would lose work).
    if ipynb_newer_drift:
        print("\nWARNING: .ipynb is newer than .py for the following files:")
        for d in ipynb_newer_drift:
            print(f"  {d.py.relative_to(REPO_ROOT)}")
        print(
            "Sync forward would overwrite these. Resolve manually "
            "(`jupytext --to py <file>.ipynb` to pull notebook edits into .py "
            "after reviewing the diff).",
        )

    if args.safe_only:
        # Sync only forward-direction drift that won't invalidate committed
        # outputs: doc-only (outputs still match code) and code-drift-no-outputs
        # (nothing to invalidate). Skip code-drift-with-outputs and reverse drift.
        targets = [d for d in py_newer if d.category in ("doc_only", "code_drift_no_outputs")]
        skipped = len(py_newer) - len(targets)
        if skipped:
            print(f"--safe-only: skipping {skipped} code-drift-with-outputs file(s).")
    else:
        targets = py_newer

    if not targets:
        print("Nothing to sync.")
        return 0 if not ipynb_newer_drift else 1

    failures = 0
    for d in targets:
        ok, msg = sync_one(d.py)
        status = "OK   " if ok else "FAIL "
        print(f"  {status} {d.py.relative_to(REPO_ROOT)}")
        if not ok:
            failures += 1
            print(f"        {msg}")

    if failures:
        print(f"\n{failures} file(s) failed to sync.", file=sys.stderr)
        return 1
    print(f"\nSynced {len(targets)} notebook(s).")
    return 1 if ipynb_newer_drift else 0


if __name__ == "__main__":
    sys.exit(main())
