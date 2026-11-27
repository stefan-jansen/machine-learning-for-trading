import ast
from pathlib import Path

import numpy as np
import pytest
from scipy import stats


# Execute only the notebook function, not the top-level data and plotting cells.
def _load_function(name, namespace=None):
    path = Path(__file__).resolve().parents[1] / "19_risk_management/01_var_cvar.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"np": np, "stats": stats, "N_MC_SIMULATIONS": 10_000, "SEED": 42}
    ns.update(namespace or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


def test_a_multi_period_loss_cannot_exceed_the_capital_at_risk():
    """Summing simple returns reports a >100% loss on a long-only wealth path.

    Compounding bounds the multi-period return below at -100% by construction, because
    prod(1 + r) >= 0 for every r > -1. Summing has no such bound, so over a long horizon
    with a wide return distribution it reports a loss larger than the capital committed.
    """
    monte_carlo_var = _load_function("monte_carlo_var")
    # A wide, near-Gaussian series: the fitted t has a high df, so no single draw reaches
    # the -100% support bound and the added guard does not fire. The sum of 100 such draws
    # does reach past -100% in its 5% tail.
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.12, size=4_000)
    var = monte_carlo_var(returns, confidence=0.95, n_simulations=10_000, horizon=100)
    assert var < 1.0, f"a 100-day VaR of {var:.4f} is a loss above 100% of capital"


def test_the_support_guard_leaves_an_ordinary_daily_series_alone():
    """The guard must reject only draws past the floor, not the notebook's own call.

    A guard applied to the cumulative return instead of the draws, or written with the
    wrong comparison, would raise on the one-day call that produces figure 19.2. The
    series here is the real thing's order of magnitude: daily SPY-like returns, whose
    fitted Student-t has heavy tails but no mass near -100% over a single day.
    """
    monte_carlo_var = _load_function("monte_carlo_var")
    rng = np.random.default_rng(0)
    returns = rng.standard_t(5, size=2_000) * 0.01
    for confidence in (0.90, 0.95, 0.99):
        var = monte_carlo_var(returns, confidence=confidence, horizon=1)
        # A one-day VaR on a ~1% daily vol series: basis points, not a wipeout.
        assert 0.001 < var < 0.10, f"{confidence:.0%} one-day VaR of {var:.4f} is implausible"


def test_a_return_at_or_below_minus_one_is_refused():
    """A fitted Student-t can draw below -100%, which compounds into a negative wealth."""
    monte_carlo_var = _load_function("monte_carlo_var")
    # A heavy-tailed, wide series makes the fitted t sample past the -100% support bound.
    returns = np.array([-0.95, 0.95] * 100 + [-3.0, 3.0] * 10)
    with pytest.raises(ValueError, match="at or below -100%"):
        monte_carlo_var(returns, confidence=0.99, n_simulations=200_000, horizon=5)
