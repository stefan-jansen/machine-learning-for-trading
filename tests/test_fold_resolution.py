"""The fold dimension of ``model_based.parquet`` must be resolved before the eval join.

``04_model_based_features`` fits its features per walk-forward fold, so the artifact is
keyed ``(timestamp, symbol, fold)``. An evaluation notebook that joins it onto the
financial panel on ``(timestamp, symbol)`` alone multiplies every row by the number of
folds covering it, which inflates every downstream statistic and can exhaust memory.

Independent checks, because each alone is passable while the bug is present:

* :func:`test_fold_resolution_yields_unique_join_keys` proves on the artifact itself
  that restricting each fold to its own validation window leaves one row per join key.
  It is the data-level statement of the invariant and needs the built artifacts.
* :func:`test_evaluation_excludes_fold_from_feature_columns` proves from the source
  that no evaluation notebook treats ``fold`` as a feature, and that none resolves the
  fold dimension by de-duplicating, which keeps an arbitrary fold rather than the one
  the row is out of sample in.
* :func:`test_validation_windows_also_narrow_the_evaluation_panel` proves that a
  notebook which resolves the fold dimension by validation window also screens on
  those windows. Resolving without narrowing is a second wrong answer: a Chapter 9
  feature then exists on the fraction of the span its windows reach, and the coverage
  gate removes it for a property of its design. It removed all twenty model-based
  features from ``nasdaq100_microstructure`` on 2026-08-06.
"""

from __future__ import annotations

import ast
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_STUDIES = REPO_ROOT / "case_studies"
JOIN_KEY_CANDIDATES = ("timestamp", "symbol", "product", "position")


def _evaluation_notebooks() -> list[Path]:
    """Every stage-05 evaluation notebook, however its case study numbers the stage."""
    found = sorted(CASE_STUDIES.glob("*/05_evaluation.py"))
    found += sorted(CASE_STUDIES.glob("*/04_evaluation.py"))
    return found


def _model_based_artifacts() -> list[Path]:
    return sorted(CASE_STUDIES.glob("*/features/model_based.parquet"))


@pytest.mark.parametrize("artifact", _model_based_artifacts(), ids=lambda p: p.parent.parent.name)
def test_fold_resolution_yields_unique_join_keys(artifact: Path) -> None:
    """Restricting each fold to its validation window leaves one row per join key."""
    from utils.artifact_specs import load_setup_config, resolve_label_buffer
    from utils.cv_splits import generate_cv_splits

    case_study = artifact.parent.parent.name
    schema = pl.read_parquet_schema(artifact)
    if "fold" not in schema:
        pytest.skip(f"{case_study}: model_based.parquet is not fold-keyed")

    join_cols = [c for c in JOIN_KEY_CANDIDATES if c in schema]
    date_col = "timestamp"

    setup = load_setup_config(case_study)
    primary = setup["labels"]["primary"]
    label_path = artifact.parent.parent / "labels" / f"{primary}.parquet"
    if not label_path.exists():
        pytest.skip(f"{case_study}: no label artifact at {label_path.name}")

    splits = generate_cv_splits(
        pl.scan_parquet(label_path).select(date_col).unique().sort(date_col).collect(),
        case_study_id=case_study,
        label_buffer=resolve_label_buffer(case_study, primary, setup),
    )
    windows = {int(s["fold"]): (s["val_start"], s["val_end"]) for s in splits}
    ts_dtype = schema[date_col]

    resolved = (
        pl.scan_parquet(artifact)
        .filter(pl.col("fold").is_in(list(windows)))
        .filter(
            pl.col("fold")
            .replace_strict({f: lo for f, (lo, _) in windows.items()}, default=None)
            .cast(ts_dtype)
            <= pl.col(date_col)
        )
        .filter(
            pl.col(date_col)
            <= pl.col("fold")
            .replace_strict({f: hi for f, (_, hi) in windows.items()}, default=None)
            .cast(ts_dtype)
        )
        .drop("fold")
    )
    n_rows = resolved.select(pl.len()).collect().item()
    n_keys = resolved.select(join_cols).unique().select(pl.len()).collect().item()

    assert n_rows == n_keys, (
        f"{case_study}: fold resolution leaves {n_rows - n_keys:,} duplicate "
        f"{join_cols} rows, so validation windows overlap and a fitted feature would "
        f"take two values on one date"
    )


def _feature_list_assignments(tree: ast.Module) -> list[tuple[str, ast.ListComp]]:
    """Assignments of the form ``<name>_cols = [c for c in <frame>.columns if ...]``."""
    out: list[tuple[str, ast.ListComp]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.ListComp):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.endswith("_cols"):
                out.append((target.id, node.value))
    return out


def _iterates_fold_keyed_frame(comp: ast.ListComp, source: str) -> bool:
    """True when the comprehension enumerates the columns of the model-based artifact.

    ``financial.parquet`` has no ``fold`` column, so only lists derived from the
    fold-keyed frame are in scope for this clause.
    """
    for gen in comp.generators:
        segment = ast.get_source_segment(source, gen.iter) or ""
        if "temporal" in segment or "model_based" in segment:
            return True
    return False


def _name_definitions(source: str, tree: ast.Module) -> dict[str, str]:
    """Source text of every module-level assignment, keyed by the name assigned."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out.setdefault(target.id, ast.get_source_segment(source, node.value) or "")
    return out


@pytest.mark.parametrize("notebook", _evaluation_notebooks(), ids=lambda p: p.parent.name)
def test_evaluation_excludes_fold_from_feature_columns(notebook: Path) -> None:
    """``fold`` is a key of the artifact, so it may never enter a feature-column list."""
    source = notebook.read_text()
    if "model_based.parquet" not in source:
        pytest.skip(f"{notebook.parent.name}: does not read model_based.parquet")

    tree = ast.parse(source)
    definitions = _name_definitions(source, tree)

    offenders = []
    for name, comp in _feature_list_assignments(tree):
        if not _iterates_fold_keyed_frame(comp, source):
            continue
        # The guard, the frame it enumerates, and one level of indirection through any
        # name either of them references. A case study may exclude the key through a
        # constant (`NON_FEATURE_COLS`), through an already-filtered list
        # (`temporal_feature_cols`), or under its own name (`validation_fold`).
        considered = [ast.get_source_segment(source, gen.iter) or "" for gen in comp.generators]
        considered += [
            ast.get_source_segment(source, cond) or ""
            for gen in comp.generators
            for cond in gen.ifs
        ]
        for referenced in list(considered):
            for node in ast.walk(ast.parse(referenced.strip() or "None", mode="eval")):
                if isinstance(node, ast.Name):
                    considered.append(definitions.get(node.id, ""))
        if "fold" not in "".join(considered):
            offenders.append(name)

    # A list built after the column has already been dropped is safe by construction.
    if offenders and 'drop("fold")' in source:
        offenders = [n for n in offenders if source.index(f"{n} =") < source.index('drop("fold")')]

    assert not offenders, (
        f"{notebook.parent.name}/{notebook.stem}: {offenders} may contain 'fold', which "
        f"is a key of model_based.parquet and not a feature. Exclude it explicitly, as "
        f"in `[c for c in frame.columns if c not in (*JOIN_COLS, 'fold')]`."
    )


def _expands_one_level(expr: str, definitions: dict[str, str]) -> str:
    """``expr`` plus the source of every module-level name it references."""
    parts = [expr]
    try:
        tree = ast.parse(expr.strip() or "None", mode="eval")
    except SyntaxError:
        return expr
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            parts.append(definitions.get(node.id, ""))
    return "".join(parts)


def _window_filtered_frames(source: str, tree: ast.Module, definitions: dict[str, str]) -> set[str]:
    """Frames whose rows are restricted to the configured validation windows."""
    filtered: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        for call in ast.walk(node.value):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr != "filter":
                continue
            expanded = "".join(
                _expands_one_level(ast.get_source_segment(source, arg) or "", definitions)
                for arg in call.args
            )
            if "val_windows" in expanded:
                filtered.update(targets)
    return filtered


@pytest.mark.parametrize("notebook", _evaluation_notebooks(), ids=lambda p: p.parent.name)
def test_validation_windows_also_narrow_the_evaluation_panel(notebook: Path) -> None:
    """Resolving the fold by validation window obliges screening on those windows.

    A Chapter 9 feature exists only inside the window of the fold that fitted it. Left
    on the full pre-holdout span it is null everywhere else, its coverage is the share
    of the span the windows reach, and the correctness gate rejects it for a property
    of its design. Narrowing the panel to the union of the windows also puts the
    Chapter 8 features on the same rows, so the two ICs are comparable.
    """
    source = notebook.read_text()
    if "val_windows" not in source:
        pytest.skip(f"{notebook.parent.name}: does not resolve folds by validation window")

    tree = ast.parse(source)
    definitions = _name_definitions(source, tree)
    window_filtered = _window_filtered_frames(source, tree, definitions)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not any("panel" in t for t in targets):
            continue
        for call in ast.walk(node.value):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr != "filter":
                continue
            for arg in call.args:
                segment = ast.get_source_segment(source, arg) or ""
                if "val_windows" in _expands_one_level(segment, definitions):
                    return

        # An inner join to an already window-filtered temporal frame narrows the
        # evaluation panel by construction. This is the form used when every
        # financial-feature row must also have one out-of-sample temporal estimate.
        for call in ast.walk(node.value):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr != "join" or not call.args:
                continue
            how = next((kw.value for kw in call.keywords if kw.arg == "how"), None)
            if not isinstance(how, ast.Constant) or how.value != "inner":
                continue
            joined = ast.get_source_segment(source, call.args[0]) or ""
            if any(name in joined for name in window_filtered):
                return

    raise AssertionError(
        f"{notebook.parent.name}/{notebook.stem}: resolves the fold dimension with "
        f"`val_windows` but never filters the evaluation panel on them. Every "
        f"model-based feature is then null outside its own fold's validation window "
        f"and fails the coverage gate for a property of its design - this removed all "
        f"twenty of them from nasdaq100_microstructure. Narrow the panel to the union "
        f"of the windows before screening, as `eval_panel.filter(IN_VALIDATION)`."
    )


@pytest.mark.parametrize("notebook", _evaluation_notebooks(), ids=lambda p: p.parent.name)
def test_evaluation_does_not_resolve_folds_by_deduplication(notebook: Path) -> None:
    """De-duplicating keeps an arbitrary fold, not the one the row is out of sample in."""
    source = notebook.read_text()
    if "model_based.parquet" not in source:
        pytest.skip(f"{notebook.parent.name}: does not read model_based.parquet")

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "unique":
            continue
        kwargs = {kw.arg for kw in node.keywords}
        if "subset" in kwargs and "keep" in kwargs:
            raise AssertionError(
                f"{notebook.parent.name}/{notebook.stem}: resolves duplicate join keys "
                f"with .unique(subset=..., keep=...), which keeps whichever fold sorts "
                f"last rather than the fold the row is out of sample in. Restrict each "
                f"fold to its own validation window instead."
            )
