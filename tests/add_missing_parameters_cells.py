"""Add missing Papermill parameters cells to notebooks.

Notebooks that already have a `# %% tags=["parameters"]` cell are skipped.
For notebooks without one, this inserts an empty parameters cell after the
first code cell (or after imports if detectable).

Usage:
    uv run python tests/add_missing_parameters_cells.py [--dry-run]
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DRY_RUN = "--dry-run" in sys.argv

# Parameters cell template
PARAMS_CELL = """
# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
"""


def find_notebooks() -> list[Path]:
    """Find all Jupytext notebooks missing a parameters cell."""
    notebooks = []
    for d in sorted(REPO_ROOT.glob("[0-9]*_*")):
        notebooks.extend(sorted(d.glob("**/*.py")))
    for d in sorted((REPO_ROOT / "case_studies").glob("*")):
        if d.is_dir() and d.name not in ("utils", "__pycache__"):
            notebooks.extend(sorted(d.glob("**/*.py")))

    missing = []
    for nb in notebooks:
        content = nb.read_text()
        if "# %%" not in content:
            continue
        if 'tags=["parameters"]' in content or "tags=['parameters']" in content:
            continue
        missing.append(nb)
    return missing


def add_parameters_cell(nb_path: Path) -> bool:
    """Insert a parameters cell after the first code cell.

    Strategy: find the first `# %%` code cell (not markdown, not the header)
    and insert the parameters cell after it. If the first code cell has imports,
    the parameters cell goes after the import block.
    """
    content = nb_path.read_text()
    lines = content.split("\n")

    # Find insertion point: after the first code cell
    # Skip: header (---/jupyter metadata), markdown cells
    in_header = False
    found_first_code = False
    insert_after = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track Jupytext header
        if i == 0 and stripped == "# ---":
            in_header = True
            continue
        if in_header:
            if stripped == "# ---":
                in_header = False
            continue

        # Found a code cell
        if stripped == "# %%" and not found_first_code:
            found_first_code = True
            # Look ahead to find end of this cell (next # %% or end)
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith("# %%"):
                    insert_after = j
                    break
            else:
                insert_after = len(lines)
            break

    if insert_after is None:
        return False

    # Insert the parameters cell
    new_lines = (
        lines[:insert_after] + PARAMS_CELL.rstrip().split("\n") + [""] + lines[insert_after:]
    )
    new_content = "\n".join(new_lines)

    if not DRY_RUN:
        nb_path.write_text(new_content)
    return True


def main():
    missing = find_notebooks()
    print(f"Found {len(missing)} notebooks missing parameters cell")
    if DRY_RUN:
        print("(dry run — no changes)")

    modified = 0
    for nb in missing:
        rel = nb.relative_to(REPO_ROOT)
        ok = add_parameters_cell(nb)
        status = "ADDED" if ok else "SKIPPED"
        if ok:
            modified += 1
        print(f"  {status}: {rel}")

    print(f"\nModified: {modified}/{len(missing)}")


if __name__ == "__main__":
    main()
