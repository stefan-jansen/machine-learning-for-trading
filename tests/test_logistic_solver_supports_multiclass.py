"""No shared logistic preset may pin `liblinear`, because these labels are multiclass.

Four case studies fit the `logistic_l1_*` presets, and their direction labels take three
values (-1, 0, 1) rather than two. scikit-learn 1.8 makes multiclass `liblinear` a hard
error, and #740 already moved this repo's floor to 1.7, so the deprecation is one release
away from breaking every one of them.

Speed is the second reason and it is the one that was actually costing time. `liblinear` is
single-threaded coordinate descent scaling about N^1.4 on this data. Measured on
nasdaq100_microstructure's `fwd_dir_15m` panel, L1 at C=100: 144.4 s at 400k rows and 716.9 s
at 1.2M, against 11.0 s and 44.8 s for `saga` at `tol=1e-2`. That extrapolates to roughly
eight hours per configuration at the full 16.9M rows against about twenty minutes - and
`06_linear` did run 7h23m before being killed with exactly these two configurations
unfinished.

`OneVsRestClassifier(liblinear)` is the fix this test is written to refuse. It silences the
deprecation while preserving the objective exactly, because liblinear multiclass already is
one-vs-rest - so it keeps the eight hours.

Refs ml4t/agent-workspace#1040.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PRESETS = Path(__file__).resolve().parents[1] / "case_studies" / "config" / "logistic"
# `saga` needs a looser tolerance than coordinate descent to converge here: at 1e-3 it hits
# the iteration cap at both sizes measured; at 1e-2 it converges in 22 to 29 iterations,
# which is the count liblinear needs at 1e-4.
REQUIRED_TOL = 0.01


def _presets() -> list[Path]:
    return sorted(PRESETS.glob("logistic_l1_*.yaml"))


def test_the_l1_presets_are_present() -> None:
    """A guard over an empty glob passes without checking anything."""
    assert len(_presets()) >= 6, f"expected the six shared L1 presets, found {_presets()}"


@pytest.mark.parametrize("preset", _presets(), ids=lambda p: p.stem)
def test_no_l1_preset_pins_liblinear(preset: Path) -> None:
    params = yaml.safe_load(preset.read_text())["params"]
    assert params["solver"] != "liblinear", (
        f"{preset.name} pins liblinear, which scikit-learn 1.8 refuses for the three-class "
        "direction labels these presets are fitted on, and which does not finish on a "
        "16.9M-row panel. Use saga."
    )
    assert params["solver"] == "saga", (
        f"{preset.name} uses {params['solver']!r}. saga is the solver scikit-learn "
        "recommends for large-n L1 and the only one measured to converge here."
    )


@pytest.mark.parametrize("preset", _presets(), ids=lambda p: p.stem)
def test_each_l1_preset_sets_the_tolerance_saga_needs(preset: Path) -> None:
    """Left at the 1e-4 default, saga does not converge within any iteration cap tried."""
    params = yaml.safe_load(preset.read_text())["params"]
    assert params.get("tol") == REQUIRED_TOL, (
        f"{preset.name} does not set tol={REQUIRED_TOL}. At the scikit-learn default of "
        "1e-4, and at 1e-3, saga hit the iteration cap without converging at both 400k and "
        "1.2M rows."
    )
    assert params.get("max_iter", 0) >= 200, (
        f"{preset.name} caps iterations below the 22-29 saga needed plus headroom"
    )
