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
    production     : bool — True iff overrides preserve the full production surface
    parameters     : the papermill parameter overrides, as declared by the executor
    notes          : optional free text (e.g. "GPU libs: xgboost,lightgbm,catboost")

Where ``parameters`` comes from, and why it is not read out of the notebook
----------------------------------------------------------------------------

``metadata.papermill.parameters`` cannot answer "was *this* run parameterized".
Two mechanisms make it a fossil, both reproducible:

1. Papermill does not clear it. Re-executing with no overrides leaves the previous
   run's dict in place next to a freshly written ``start_time``, so the timestamps
   describe the new run and the parameters describe an older one.
2. ``jupytext --sync`` rebuilds the cell list from the ``.py`` and so removes the
   ``injected-parameters`` cell, but notebook-level ``metadata`` survives
   untouched. The marker that belongs to the run is deleted and the stale one is
   kept.

So the executor states the parameters and the stamp records that statement:
``--production`` for an unparameterized run, or ``--parameters '<json>'``. One of
the two is required; this tool never infers.

Where the notebook *does* carry evidence of its own execution — papermill's
``injected-parameters`` cell, which lives in the cell list and is rewritten by
every parameterized run — ``stamp`` cross-checks the declaration against it and
refuses to write a stamp that contradicts it. Stamping also rewrites
``metadata.papermill.parameters`` to the declared set, so the fossil cannot
outlive the stamp and disagree with it later.

Gate (``check``): for every tracked ``.ipynb`` that HAS a stamp,

* ``source_py_blob`` must equal ``git hash-object`` of the current paired ``.py``
  (else the ``.py`` changed since the notebook was executed — STALE),
* ``production`` must be True (else a TEST-mode run was committed), and
* the stamp must not contradict a committed ``injected-parameters`` cell (else the
  notebook was re-executed with overrides after it was stamped).

Notebooks WITHOUT a stamp are reported as "unverified" but do not fail unless
``--strict`` is passed. This is deliberate: adoption is gradual — stamp notebooks
as they are re-run through the canonical path, and the gate enforces only where
provenance exists. Flip to ``--strict`` once the backfill is complete.

Usage::

    uv run python .github/scripts/notebook_provenance.py stamp <nb.ipynb> --executor ml4t-gpu --production
    uv run python .github/scripts/notebook_provenance.py stamp <nb.ipynb> --executor ml4t-gpu --parameters '{"MAX_SYMBOLS": 5}'
    uv run python .github/scripts/notebook_provenance.py stamp <nb.ipynb> --executor ml4t-gpu --production --notes "..."
    uv run python .github/scripts/notebook_provenance.py check          # gate (stamped-only)
    uv run python .github/scripts/notebook_provenance.py check --strict  # also fail on unverified
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from datetime import UTC, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_PARTS = {"_reference", ".venv", ".git", ".ipynb_checkpoints"}
STAMP_KEY = "ml4t_provenance"
INJECTED_TAG = "injected-parameters"
PRODUCTION_SAFE_PARAMETERS = {
    "FORCE_REBACKTEST": True,
    "FORCE_RETRAIN": True,
    "USE_CACHE": False,
}


def _coerce_bool(value: object) -> bool | None:
    """Return a boolean for Papermill's bool-like CLI values.

    The same override reaches this module as ``True`` through a YAML parameter file,
    ``"true"`` through ``papermill -p`` (which stringifies everything), and ``1``
    through a JSON declaration. All three name the same run, so ``1`` and ``"1"``
    have to coerce alike — otherwise one form is read as a boolean and the other as
    the string ``"1"``, and comparing the two reports a contradiction that is not
    there.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None


def production_parameters(parameters: dict[str, object]) -> bool:
    """Whether overrides preserve the full production execution surface."""
    return all(
        name in PRODUCTION_SAFE_PARAMETERS
        and _coerce_bool(value) is PRODUCTION_SAFE_PARAMETERS[name]
        for name, value in parameters.items()
    )


def _normalize_value(value: object) -> tuple[str, object]:
    """Tagged scalar form: ``("bool", …)``, ``("num", …)`` or ``("str", …)``.

    The tag is what keeps ``True`` distinct from the number 1. Python treats them as
    equal — ``bool`` subclasses ``int`` — so an untagged form would let a numeric
    override match a boolean one.

    Numbers compare as ``Decimal`` rather than ``float``, so two integers that differ
    above 2**53 stay different. Floats route through ``str`` first, so the literal
    ``0.1`` matches the string ``"0.1"`` instead of its binary expansion. NaN and the
    infinities compare as lowercased text, because a Decimal NaN does not equal
    itself and every form of a non-finite value has to reach the same normal form.
    """
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float, str)):
        text = str(value).strip()
        try:
            number = Decimal(value if isinstance(value, int) else text)
        except (InvalidOperation, ValueError):
            return ("str", text)  # not a number at all
        if number.is_finite():
            return ("num", number)
        # Decimal's own spelling is the normal form that makes `inf`, `Infinity`
        # and `float("inf")` agree, and NaN compare equal to itself at all.
        return ("str", str(number).lower())
    return ("str", str(value))


def _normalize_parameters(parameters: dict[str, object]) -> dict[str, object]:
    """Comparable form of a parameter set.

    Papermill's ``-p`` stringifies every value, so one override arrives as ``True``
    through a YAML file and ``"true"`` through the CLI. Normalizing lets the two
    compare equal, so a declaration is not reported as contradicting the injected
    cell that recorded the same run.

    Only the parameters in ``PRODUCTION_SAFE_PARAMETERS`` are read as booleans.
    Every other name keeps its type, so ``MAX_SYMBOLS = 1`` still contradicts a
    declared ``{"MAX_SYMBOLS": true}`` while matching ``{"MAX_SYMBOLS": "1"}``.
    """
    out: dict[str, object] = {}
    for name, value in parameters.items():
        if name in PRODUCTION_SAFE_PARAMETERS:
            coerced = _coerce_bool(value)
            if coerced is not None:
                out[name] = ("bool", coerced)
                continue
        out[name] = _normalize_value(value)
    return out


def injected_parameters(nb: dict) -> dict[str, object] | None:
    """Parameters papermill injected into *this* execution, or None if it did not.

    Papermill writes an ``injected-parameters``-tagged cell holding one literal
    assignment per override. Unlike ``metadata.papermill.parameters`` this lives in
    the cell list, so it cannot outlive the run it belongs to: a parameterized run
    rewrites it, and ``jupytext --sync`` drops it with every other cell that is not
    in the paired ``.py``. A cell that is merely left over from an earlier run was
    still executed by the later one — its values were in force either way — so it
    is evidence about the committed outputs, which the metadata block is not.

    Returns None when there is no such cell, which is the ordinary shape of an
    unparameterized run *and* of any notebook synced since. Absence is therefore
    not proof of a production run; that is why the executor must declare.
    """
    cells = [
        c for c in nb.get("cells", []) if INJECTED_TAG in (c.get("metadata", {}).get("tags") or [])
    ]
    if not cells:
        return None
    source = cells[-1].get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    params: dict[str, object] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return params
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            params[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            params[target.id] = _non_literal_value(node.value)
    return params


def _non_literal_value(node: ast.expr) -> object:
    """Value for an injected assignment that ``literal_eval`` will not take.

    Papermill spells a non-finite float as a call, ``float('nan')``, so that form
    has to come back as the number to compare against a declared ``float("nan")``.
    Anything else compares as its own source text.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"float", "int"}
        and len(node.args) == 1
        and not node.keywords
    ):
        constructor = float if node.func.id == "float" else int
        try:
            return constructor(ast.literal_eval(node.args[0]))
        except (ValueError, TypeError):
            pass
    return ast.unparse(node)


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


def contradicts_injected_cell(nb: dict, parameters: dict[str, object]) -> str | None:
    """Why ``parameters`` disagrees with the notebook's injected cell, or None.

    Only meaningful when the notebook still carries an ``injected-parameters``
    cell. No cell means no evidence either way, not agreement.
    """
    injected = injected_parameters(nb)
    if injected is None:
        return None
    declared_n = _normalize_parameters(parameters)
    injected_n = _normalize_parameters(injected)
    if declared_n == injected_n:
        return None
    return (
        f"declared parameters {parameters!r} do not match the injected-parameters "
        f"cell this notebook was executed with, {injected!r}"
    )


def stamp_notebook(
    nb_path: Path,
    executor: str,
    notes: str | None = None,
    *,
    parameters: dict[str, object],
) -> dict:
    """Record how this notebook was executed.

    ``parameters`` is the executor's statement of the overrides the run used —
    ``{}`` for a production run. It is never read back out of the notebook: see the
    module docstring for why ``metadata.papermill.parameters`` cannot answer that
    question. Stamping overwrites that metadata with the declared set, so the
    fossil cannot survive to contradict the stamp.
    """
    py = paired_py(nb_path)
    if py is None:
        raise SystemExit(f"no paired .py for {nb_path.relative_to(REPO_ROOT)} — cannot stamp")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    conflict = contradicts_injected_cell(nb, parameters)
    if conflict:
        raise SystemExit(f"refusing to stamp {nb_path.relative_to(REPO_ROOT)}: {conflict}")
    stamp = {
        "source_py_blob": git_blob(py),
        "executed_at": datetime.now(UTC).isoformat(),
        "executor": executor,
        "production": production_parameters(parameters),
        "parameters": parameters,
    }
    if notes:
        stamp["notes"] = notes
    metadata = nb.setdefault("metadata", {})
    metadata[STAMP_KEY] = stamp
    metadata.setdefault("papermill", {})["parameters"] = parameters
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return stamp


def check_all(strict: bool = False) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (stale, testmode, contradicted, unverified) repo-relative offenders."""
    stale: list[str] = []
    testmode: list[str] = []
    contradicted: list[str] = []
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
        conflict = contradicts_injected_cell(nb, stamp.get("parameters") or {})
        if conflict:
            contradicted.append(f"{rel} ({conflict})")
    return stale, testmode, contradicted, unverified


def _cmd_stamp(args: argparse.Namespace) -> int:
    if args.production:
        parameters: dict[str, object] = {}
    else:
        try:
            parameters = json.loads(args.parameters)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--parameters is not valid JSON: {exc}") from exc
        if not isinstance(parameters, dict):
            raise SystemExit("--parameters must be a JSON object of override name to value")
    s = stamp_notebook(
        Path(args.notebook).resolve(), args.executor, args.notes, parameters=parameters
    )
    print(
        f"stamped {args.notebook}: source_py_blob={s['source_py_blob'][:12]} "
        f"executor={s['executor']} production={s['production']}"
    )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    stale, testmode, contradicted, unverified = check_all(strict=args.strict)
    fail = bool(stale or testmode or contradicted) or (args.strict and bool(unverified))
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
    if contradicted:
        print(
            "CONTRADICTED (the stamp disagrees with the injected-parameters cell in the "
            "committed notebook — re-stamp the run that produced these outputs):"
        )
        for r in contradicted:
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
            f"{len(contradicted)} contradicted, {len(unverified)} unverified (advisory)"
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
    # The executor states the parameters; the tool does not infer them from
    # notebook metadata written by a different process. See the module docstring.
    params = sp.add_mutually_exclusive_group(required=True)
    params.add_argument(
        "--production", action="store_true", help="this run used no parameter overrides"
    )
    params.add_argument(
        "--parameters",
        help="JSON object of the overrides this run used, e.g. '{\"MAX_SYMBOLS\": 5}'",
    )
    sp.set_defaults(func=_cmd_stamp)

    cp = sub.add_parser("check", help="gate: fail on stale or test-mode stamped notebooks")
    cp.add_argument("--strict", action="store_true", help="also fail on unstamped notebooks")
    cp.set_defaults(func=_cmd_check)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
