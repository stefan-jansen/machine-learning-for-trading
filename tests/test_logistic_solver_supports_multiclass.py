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
# The tolerance is split by where the penalty binds, and both halves are measured.
#
# Weakly penalised (C >= 1): saga needs a LOOSER tolerance than coordinate descent to
# converge at all. At 1e-4 and 1e-3 it hits the iteration cap at every size tried; at 1e-2 it
# converges in 22 to 29 iterations, the count liblinear needs at 1e-4.
#
# Strongly penalised (C <= 0.1): 1e-2 converges but leaves coefficients stranded NEAR zero
# rather than AT zero, and `coef_ != 0` counts those as live. At C=0.01 that is 24 exact
# zeros against 52 below 1e-8, where liblinear has 40 of each. At 1e-3 the two counts agree
# and saga is more sparse than liblinear at every C measured. Sparsity is the entire point of
# the small-C end of this sweep, so it gets the tighter tolerance.
LOOSE_TOL = 0.01
TIGHT_TOL = 0.001
# Where the penalty starts binding hard enough for the stranding to matter.
TIGHT_TOL_MAX_C = 0.1


def _c_of(preset: Path) -> float:
    return float(yaml.safe_load(preset.read_text())["params"]["C"])


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
    expected = TIGHT_TOL if _c_of(preset) <= TIGHT_TOL_MAX_C else LOOSE_TOL
    assert params.get("tol") == expected, (
        f"{preset.name} sets tol={params.get('tol')}, expected {expected}. At C<=0.1 the "
        "penalty binds and 1e-2 strands coefficients near zero rather than at zero, which "
        "blunts the sparsity the sweep exists to show; at C>=1 saga does not converge at "
        "any tolerance tighter than 1e-2."
    )
    assert params.get("max_iter", 0) >= 200, (
        f"{preset.name} caps iterations below the 22-29 saga needed plus headroom"
    )
