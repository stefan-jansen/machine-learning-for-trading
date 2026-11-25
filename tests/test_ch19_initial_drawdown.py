import ast
from pathlib import Path

import numpy as np


# Execute only the notebook function, not the top-level data and plotting cells.
def _load_function(name):
    path = Path(__file__).resolve().parents[1] / "19_risk_management/01_var_cvar.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_initial_wealth_is_a_drawdown_peak():
    drawdown_path = _load_function("drawdown_path")
    wealth, peak, dd = drawdown_path(np.array([-0.1, 0.05]))
    np.testing.assert_allclose(wealth, [0.9, 0.945])
    np.testing.assert_allclose(peak, [1.0, 1.0])
    np.testing.assert_allclose(dd, [-0.1, -0.055])


def test_drawdown_after_new_peak():
    drawdown_path = _load_function("drawdown_path")
    _, peak, dd = drawdown_path(np.array([0.1, -0.1]))
    np.testing.assert_allclose(peak, [1.1, 1.1])
    np.testing.assert_allclose(dd, [0.0, -0.1])
