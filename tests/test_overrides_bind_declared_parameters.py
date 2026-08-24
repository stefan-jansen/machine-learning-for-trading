"""Every parameter `tests/overrides.yaml` binds must be one its notebook reads.

Papermill injects any name it is given. A name the notebook does not declare becomes an
unused variable in the injected cell and the run proceeds at full scale, so a reduction
that was never applied is indistinguishable from one that was - the CI run passes, takes
production time, and reports nothing.

Measured on `cs6/crypto_perps_funding`: `14_portfolio_management`, `15_costs` and
`16_risk_management` bound `MAX_SYMBOLS`, and `16` also bound `MAX_RISK_VARIANTS`, after a
rewrite removed all four declarations. On `cs6/cme_futures` at 166a0791 the same rewrite
left ten such bindings across four notebooks, including `TOP_K`, `TOP_N_PREDICTIONS` and
`TOP_N_COMBOS`.

This test was shown able to fail before it was trusted: against `483b4c94~1` the same
check reported crypto's four bindings, and zero against the commit that fixed them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERRIDES = REPO_ROOT / "tests" / "overrides.yaml"

# `research_preview_parameters` pops these and folds them into PREVIEW_REDUCTIONS rather than
# passing them through, so a notebook declaring PREVIEW_REDUCTIONS reads them by that route and
# must not be flagged. This is why the same name can be exempt on a model notebook and a finding
# on a backtest notebook: the model notebooks take the preview path and the backtest notebooks
# take the verbatim one. It is the run path that differs, not the rule.
_TRANSLATED = frozenset({"MAX_FOLDS", "MAX_SYMBOLS", "TRAIN_SAMPLE_FRAC"})
_PARAMS_CELL = re.compile(r'^# %% tags=\["parameters"\]\n(.*?)(?=^# %%)', re.S | re.M)


def _declared_and_body(source: str) -> tuple[set[str], str]:
    match = _PARAMS_CELL.search(source)
    if match is None:
        return set(), source
    names = set(re.findall(r"^([A-Z][A-Z0-9_]*)\s*[:=]", match.group(1), re.M))
    return names, source[match.end() :]


def _entries() -> list[tuple[str, dict]]:
    loaded = yaml.safe_load(OVERRIDES.read_text(encoding="utf-8")) or {}
    return [
        (key, cfg)
        for key, cfg in sorted(loaded.items())
        if isinstance(cfg, dict) and cfg.get("parameters")
    ]


@pytest.mark.parametrize("key,cfg", _entries(), ids=lambda v: v if isinstance(v, str) else "")
def test_override_parameters_are_read_by_their_notebook(key: str, cfg: dict) -> None:
    py = REPO_ROOT / f"{key}.py"
    if not py.exists():
        pytest.skip(f"{key}.py is not in this tree")
    declared, body = _declared_and_body(py.read_text(encoding="utf-8"))
    inert = []
    for name in sorted(cfg["parameters"]):
        if name == "PREVIEW_REDUCTIONS":
            continue
        if name in declared:
            if not re.search(rf"\b{name}\b", body):
                inert.append(f"{name}: declared in the parameters cell and never read")
            continue
        if name in _TRANSLATED and "PREVIEW_REDUCTIONS" in declared:
            continue
        inert.append(f"{name}: not declared by the notebook")
    assert not inert, (
        f"{key} binds parameters it does not read, so the reduction is not applied and the "
        f"run proceeds at full scale:\n  " + "\n  ".join(inert)
    )
