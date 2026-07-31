# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Portfolio Optimization Library Comparison
#
# **Docker image**: `ml4t`
#
# This notebook compares PyPortfolioOpt, Riskfolio-Lib, and skfolio on one training
# panel, then evaluates their frozen allocations on later observations.
#
# **Learning Objectives**:
# - Fit comparable mean-risk allocators through three library APIs
# - Keep walk-forward model assessment inside the training window
# - Compare frozen test-period risk, return, and concentration
# - Reconcile a vectorized allocation with execution-aware daily targets
#
# **Book Reference**: Chapter 17, §17.7 (Comparing Allocator Performance)
#
# **Prerequisites**: `02_mean_variance_optimization`, `03_robust_optimization`

# %% [markdown]
# ## Library Overview
#
# | Library | Role in this notebook | Interface used |
# |---------|-----------------------|----------------|
# | **PyPortfolioOpt** | Classical, tail-risk, HRP, and penalized allocations | Optimizer objects |
# | **Riskfolio-Lib** | Multiple risk measures and risk parity | Portfolio object |
# | **skfolio** | Mean-risk, HRP, and walk-forward assessment | sklearn-style estimators |
#
# The comparison concerns the versions pinned by the `ml4t` image. Package breadth and
# release cadence can change independently of the methods demonstrated here.

# %% [markdown]
# ## Imports & Settings

# %% [markdown]
# ### cvxpy Compatibility Check
#
# Riskfolio-Lib relies on a small set of cvxpy internals. Surface a clear error early
# if the installed cvxpy is incompatible, rather than failing deep inside an optimizer.

# %%
"""Compare three portfolio libraries with train-only fitting and later evaluation."""

import warnings
from contextlib import contextmanager
from unittest.mock import patch

import cvxpy as cp
import cvxpy.reductions.matrix_stuffing as cvxpy_matrix_stuffing
from cvxpy.problems.problem import Problem as _CvxpyProblem

missing_cvxpy_symbols = [
    name
    for name in ("extract_lower_bounds", "extract_upper_bounds")
    if not hasattr(cvxpy_matrix_stuffing, name)
]
if not hasattr(_CvxpyProblem, "_supports_cpp"):
    missing_cvxpy_symbols.append("Problem._supports_cpp")
if missing_cvxpy_symbols:
    raise RuntimeError(
        "Incompatible cvxpy runtime for the Riskfolio-Lib comparison notebook. "
        "Install the supported cvxpy stack instead of patching site-packages at runtime. "
        f"Missing symbols: {missing_cvxpy_symbols}"
    )

# %% [markdown]
# ### Third-Party Imports

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import riskfolio as rp
from IPython.display import Markdown, display

# Portfolio optimization libraries
# %% [markdown]
# ### ml4t and Project Imports
# %%
# ml4t libraries for diagnostics and execution-aware backtesting
from ml4t.backtest import (
    BacktestConfig,
    CommissionType,
    DataFeed,
    Engine,
    ExecutionMode,
    Strategy,
)
from ml4t.backtest.config import SlippageType
from ml4t.backtest.execution.rebalancer import RebalanceConfig, TargetWeightExecutor
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from plotly.subplots import make_subplots
from pypfopt import (
    EfficientCVaR,
    EfficientFrontier,
    HRPOpt,
    objective_functions,
    risk_models,
)
from scipy.optimize import linprog, minimize
from skfolio import RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.distance import PearsonDistance
from skfolio.model_selection import WalkForward
from skfolio.moments import EmpiricalCovariance, EmpiricalMu
from skfolio.optimization import HierarchicalRiskParity, MeanRisk, ObjectiveFunction
from skfolio.prior import EmpiricalPrior
from sklearn.base import clone

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, ml4t_palette

# %% tags=["parameters"]
# Production defaults - Papermill overrides for CI testing
MAX_SYMBOLS = 0  # 0 = all
SEED = 42
TRAIN_END = "2021-12-31"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.04
CVAR_CONFIDENCE = 0.95
FRONTIER_POINTS = 50
COMMISSION_RATE = 0.0005
SLIPPAGE_RATE = 0.0005
TRANSACTION_COST_PENALTY = 0.01
L2_GAMMA = 0.5
ACTIVE_WEIGHT_THRESHOLD = 0.001
WEIGHT_TOLERANCE = 1e-5
COMMON_WEIGHT_TOLERANCE = 5e-4
COMMON_OBJECTIVE_TOLERANCE = 1e-7
MAX_SHARPE_EXCESS_TOLERANCE = 1e-12
INFEASIBLE_MAX_SHARPE_POLICY = "cash"

# All common moments are arithmetic daily estimates. Dividing the annual hurdle by
# the annualization factor preserves exactly the same linear excess-return objective.
RISK_FREE_RATE_DAILY = RISK_FREE_RATE / TRADING_DAYS

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## Load Data
#
# Load ETF price data from canonical dataset for portfolio optimization.

# %% [markdown]
# ### Universe and Date Range

# %%
# Fixed teaching universe diversified across asset classes
_FULL_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",  # Equities
    "AGG",
    "TLT",
    "LQD",  # Fixed Income
    "GLD",
    "VNQ",
    "DBC",  # Alternatives
]
# Configuration
SYMBOLS = _FULL_UNIVERSE[:MAX_SYMBOLS] if MAX_SYMBOLS else _FULL_UNIVERSE
START_DATE = "2018-01-01"
END_DATE = "2024-12-01"

# %% [markdown]
# ### Load and Filter ETF Panel

# %%
etf_data = load_etfs(symbols=SYMBOLS, start_date=START_DATE, end_date=END_DATE).sort(
    ["timestamp", "symbol"]
)
print(f"Loaded {etf_data.height:,} rows for {len(SYMBOLS)} fixed teaching ETFs")

# %% [markdown]
# ### Pivot to Wide Returns Matrix

# %%
# Prepare the canonical panel in Polars before crossing into pandas-native optimizer APIs.
prices_wide = (
    etf_data.select(["timestamp", "symbol", "close"])
    .pivot(on="symbol", index="timestamp", values="close")
    .sort("timestamp")
    .fill_null(strategy="forward")
    .drop_nulls()
)
prices = prices_wide.to_pandas().set_index("timestamp")

returns = prices.pct_change().dropna()
tickers = prices.columns.tolist()
num_stocks = len(tickers)
train_prices = prices.loc[:TRAIN_END]
train_returns = returns.loc[:TRAIN_END]
test_returns = returns.loc[returns.index > TRAIN_END]

if train_returns.empty or test_returns.empty:
    raise RuntimeError("The declared training and test windows must both contain returns.")

print(
    f"Training: {train_returns.index.min().date()} to {train_returns.index.max().date()} "
    f"({len(train_returns):,} returns)"
)
print(
    f"Test: {test_returns.index.min().date()} to {test_returns.index.max().date()} "
    f"({len(test_returns):,} returns)"
)

# %% [markdown]
# The fixed list is a teaching universe, not a point-in-time index reconstruction. Every
# allocator sees the same training rows, arithmetic sample moments, and economic hurdle.

# %%
display(
    Markdown(
        f"All Max-Sharpe optimizers use an annual risk-free hurdle of "
        f"**{RISK_FREE_RATE:.1%}**. The daily APIs receive "
        f"**{RISK_FREE_RATE_DAILY:.6%}** under the same arithmetic annualization contract."
    )
)

# %% [markdown]
# ## Part 1: PyPortfolioOpt
#
# PyPortfolioOpt is the most accessible library with good defaults.

# %% [markdown]
# Every library returns weights in a different container. Aligning them by symbol before
# evaluation prevents silent column-order errors and makes the allocation contract explicit.


# %%
def align_weights(weights, name: str) -> pd.Series:
    """Align a library weight result to the canonical symbol order and validate it."""
    if isinstance(weights, pd.Series):
        aligned = weights.reindex(tickers).fillna(0.0).astype(float)
    elif isinstance(weights, dict):
        aligned = pd.Series(weights, dtype=float).reindex(tickers).fillna(0.0)
    else:
        aligned = pd.Series(np.asarray(weights, dtype=float), index=tickers)

    if not np.isfinite(aligned).all():
        raise RuntimeError(f"{name} returned non-finite weights.")
    if (aligned < -WEIGHT_TOLERANCE).any():
        raise RuntimeError(f"{name} violates the declared long-only bounds.")
    if not np.isclose(aligned.sum(), 1.0, atol=WEIGHT_TOLERANCE):
        raise RuntimeError(f"{name} weights sum to {aligned.sum():.8f}, not one.")
    return aligned


# %% [markdown]
# Solver wrappers keep diagnostics visible. Only the two pinned-library warnings named below are
# scoped to their responsible calls; exposed optimizer problems must report exact optimal status.


# %%
def assert_optimal_status(problem: cp.Problem, name: str) -> None:
    """Require an exact optimal status from an exposed cvxpy problem."""
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"{name} solver status is {problem.status!r}, not {cp.OPTIMAL!r}.")


CVXPY_STAR_WARNING_PATTERN = (
    r"(?s)\A\s*This use of ``\*`` has resulted in matrix multiplication\.\n"
    r"Using ``\*`` for matrix multiplication has been deprecated since CVXPY 1\.1\.\n"
    r"    Use ``\*`` for matrix-scalar and vector-scalar multiplication\.\n"
    r"    Use ``@`` for matrix-matrix and matrix-vector multiplication\.\n"
    r"    Use ``multiply`` for elementwise multiplication\.\n"
    r"This code path has been hit [0-9]+ times so far\.\s*\Z"
)
PPO_MAX_SHARPE_WARNING = (
    "max_sharpe transforms the optimization problem so additional objectives may not work "
    "as expected."
)
PPO_MAX_SHARPE_WARNING_PATTERN = (
    r"\Amax_sharpe transforms the optimization problem so additional objectives may not work "
    r"as expected\.\Z"
)


# %%
@contextmanager
def suppress_riskfolio_cvxpy_star_warning():
    """Suppress only Riskfolio's pinned cvxpy star-multiplication warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=CVXPY_STAR_WARNING_PATTERN,
            category=UserWarning,
            module=r"\Acvxpy\.expressions\.expression\Z",
        )
        yield


@contextmanager
def suppress_ppo_max_sharpe_objective_warning():
    """Suppress only PPO's warning for the intentional regularized Max-Sharpe call."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=PPO_MAX_SHARPE_WARNING_PATTERN,
            category=UserWarning,
            module=r"\Apypfopt\.efficient_frontier\.efficient_frontier\Z",
        )
        yield


# %%
def run_riskfolio(operation, name: str):
    """Run Riskfolio with scoped warnings and observable cvxpy statuses."""
    solver_statuses = []
    original_solve = cp.Problem.solve

    def solve_and_capture(problem, *args, **kwargs):
        value = original_solve(problem, *args, **kwargs)
        solver_statuses.append(problem.status)
        return value

    with (
        suppress_riskfolio_cvxpy_star_warning(),
        patch.object(cp.Problem, "solve", solve_and_capture),
    ):
        result = operation()
    if not solver_statuses:
        raise RuntimeError(f"Riskfolio exposed no solver status for {name}.")
    if any(status != cp.OPTIMAL for status in solver_statuses):
        raise RuntimeError(f"Riskfolio solver statuses for {name}: {solver_statuses}.")
    if result is None or result.empty:
        raise RuntimeError(f"Riskfolio returned no solution for {name}.")
    return result


# %%
# Prove that the scopes reject only the exact pinned warnings. The unrelated RuntimeWarning is
# captured as visible evidence instead of being emitted to the notebook's strict stderr stream.
cvxpy_warning_oracle_message = """
This use of ``*`` has resulted in matrix multiplication.
Using ``*`` for matrix multiplication has been deprecated since CVXPY 1.1.
    Use ``*`` for matrix-scalar and vector-scalar multiplication.
    Use ``@`` for matrix-matrix and matrix-vector multiplication.
    Use ``multiply`` for elementwise multiplication.
This code path has been hit 1 times so far.
"""
unrelated_warning_message = "warning-scope oracle: unrelated warning remains visible"

# %%
with warnings.catch_warnings(record=True) as warning_oracle:
    warnings.simplefilter("always")
    with suppress_riskfolio_cvxpy_star_warning():
        warnings.warn_explicit(
            cvxpy_warning_oracle_message,
            UserWarning,
            filename="cvxpy/expressions/expression.py",
            lineno=830,
            module="cvxpy.expressions.expression",
        )
    with suppress_ppo_max_sharpe_objective_warning():
        warnings.warn_explicit(
            PPO_MAX_SHARPE_WARNING,
            UserWarning,
            filename="pypfopt/efficient_frontier/efficient_frontier.py",
            lineno=259,
            module="pypfopt.efficient_frontier.efficient_frontier",
        )
    warnings.warn(unrelated_warning_message, RuntimeWarning, stacklevel=2)

visible_warning_messages = [str(item.message) for item in warning_oracle]
if cvxpy_warning_oracle_message in visible_warning_messages:
    raise RuntimeError("The exact cvxpy library warning escaped its local scope.")
if PPO_MAX_SHARPE_WARNING in visible_warning_messages:
    raise RuntimeError("The exact PyPortfolioOpt library warning escaped its local scope.")
if visible_warning_messages != [unrelated_warning_message]:
    raise RuntimeError(f"Warning-scope oracle observed unexpected warnings: {warning_oracle!r}.")
print("Warning-scope oracle: 2 exact library warnings suppressed; unrelated warning visible")


# %% [markdown]
# skfolio receives explicit empirical estimators so its moment contract does not depend on defaults.


# %%
def common_empirical_prior() -> EmpiricalPrior:
    """Return skfolio's explicit arithmetic sample-moment estimators."""
    return EmpiricalPrior(
        mu_estimator=EmpiricalMu(),
        covariance_estimator=EmpiricalCovariance(ddof=1, nearest=False),
    )


# %% [markdown]
# ### Expected Returns & Covariance

# %%
# Define the shared estimator once in daily units. PyPortfolioOpt consumes annual
# moments, while Riskfolio and skfolio consume the daily observations directly.
common_mean_daily = train_returns.mean()
common_cov_daily = train_returns.cov(ddof=1)
mu = common_mean_daily * TRADING_DAYS
S = common_cov_daily * TRADING_DAYS

print(f"Expected returns range: [{mu.min():.2%}, {mu.max():.2%}]")
print(f"Covariance matrix shape: {S.shape}")

# %% [markdown]
# A long-only Max-Sharpe risky portfolio requires at least one positive expected excess return.
# When that precondition fails, every library receives the same decision before its API boundary:
# allocate 100% to cash at the declared hurdle and report `cash_precheck` instead of invoking a
# ratio solver. This is an economic policy, not a fallback to a different risky objective.


# %%
def max_sharpe_regime(expected_returns: pd.Series, risk_free_rate: float) -> str:
    """Choose the predeclared risky-optimization or cash regime."""
    best_excess_return = float((expected_returns - risk_free_rate).max())
    if best_excess_return <= MAX_SHARPE_EXCESS_TOLERANCE:
        return INFEASIBLE_MAX_SHARPE_POLICY
    return "optimize"


# Independent nonzero-hurdle microcase: neither risky asset clears the 4% cash rate.
oracle_means = pd.Series(
    [RISK_FREE_RATE_DAILY - 2e-5, RISK_FREE_RATE_DAILY - 1e-5],
    index=["asset_a", "asset_b"],
)
oracle_expected_regime = (
    "cash" if float(np.max(oracle_means.to_numpy() - RISK_FREE_RATE_DAILY)) <= 0 else "optimize"
)
oracle_regimes = {
    library: max_sharpe_regime(oracle_means, RISK_FREE_RATE_DAILY)
    for library in ("PyPortfolioOpt", "Riskfolio", "skfolio")
}
if set(oracle_regimes.values()) != {oracle_expected_regime}:
    raise RuntimeError(f"Library-independent feasibility oracle failed: {oracle_regimes}.")
oracle_feasible_means = pd.Series(
    [RISK_FREE_RATE_DAILY - 1e-5, RISK_FREE_RATE_DAILY + 2e-5],
    index=["asset_a", "asset_b"],
)
if max_sharpe_regime(oracle_feasible_means, RISK_FREE_RATE_DAILY) != "optimize":
    raise RuntimeError("Feasible-window oracle did not reach the Max-Sharpe solver regime.")
oracle_cash_weight = 1.0
oracle_risky_weight = 0.0
oracle_period_return = oracle_cash_weight * RISK_FREE_RATE_DAILY
if oracle_risky_weight != 0 or not np.isclose(oracle_period_return, RISK_FREE_RATE_DAILY):
    raise RuntimeError("Cash policy does not preserve the declared economic hurdle.")
print(
    f"Max-Sharpe regime oracle at {RISK_FREE_RATE:.1%}: "
    f"infeasible={oracle_regimes}, feasible=optimize"
)

# %%
full_training_regime = max_sharpe_regime(common_mean_daily, RISK_FREE_RATE_DAILY)
if full_training_regime != "optimize":
    raise RuntimeError("The full training window requires the predeclared all-cash policy.")
print(f"Full-training Max-Sharpe regime: {full_training_regime}")

# %% [markdown]
# PyPortfolioOpt is easiest to read because it exposes the classical MVO inputs
# directly. That simplicity is valuable for teaching, but it also means the user
# has to be explicit about robustness choices like shrinkage and regularization.

# %% [markdown]
# ### Max Sharpe Portfolio

# %%
ef = EfficientFrontier(mu, S)
weights_sharpe = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
assert_optimal_status(ef._opt, "PPO Max Sharpe")
weights_pypfopt_sharpe = align_weights(weights_sharpe, "PPO: Max Sharpe")

# %% [markdown]
# ### Min Volatility Portfolio

# %%
ef = EfficientFrontier(mu, S)
weights_minvol = ef.min_volatility()
assert_optimal_status(ef._opt, "PPO minimum volatility")
weights_pypfopt_minvol = align_weights(weights_minvol, "PPO: Min Vol")

# %% [markdown]
# ### CVaR Optimization

# %%
cvar = EfficientCVaR(
    mu,
    train_returns,
    beta=CVAR_CONFIDENCE,
    weight_bounds=(0.0, 1.0),
)
weights_cvar = cvar.min_cvar()
assert_optimal_status(cvar._opt, "PPO minimum CVaR")
weights_pypfopt_cvar = align_weights(weights_cvar, "PPO: Min CVaR")

# %% [markdown]
# ### Hierarchical Risk Parity (HRP)

# %%
hrp = HRPOpt(train_returns)
hrp.optimize(linkage_method="ward")
weights_pypfopt_hrp = align_weights(hrp.clean_weights(), "PPO: HRP")

# %% [markdown]
# ### Covariance Shrinkage (Ledoit-Wolf)

# %%
# Shrinkage estimator for more robust covariance
S_shrunk = risk_models.CovarianceShrinkage(train_prices).ledoit_wolf()

ef_shrunk = EfficientFrontier(mu, S_shrunk)
weights_shrunk = ef_shrunk.max_sharpe(risk_free_rate=RISK_FREE_RATE)
assert_optimal_status(ef_shrunk._opt, "PPO shrinkage Max Sharpe")
weights_pypfopt_shrunk = align_weights(weights_shrunk, "PPO: Shrinkage")

# %% [markdown]
# ## Part 2: Riskfolio-Lib
#
# Riskfolio-Lib exposes several risk families through one portfolio object.

# %%
# Create the portfolio object and pass the already-defined sample moments explicitly.
port = rp.Portfolio(returns=train_returns)
port.mu = common_mean_daily.to_frame().T
port.cov = common_cov_daily
port.alpha = 1 - CVAR_CONFIDENCE
port.sht = False
port.budget = 1.0
port.solvers = ["CLARABEL"]

if not np.allclose(port.mu.to_numpy().ravel(), common_mean_daily.to_numpy()):
    raise RuntimeError("Riskfolio expected returns drifted from the common daily estimator.")
if not np.allclose(port.cov.to_numpy(), common_cov_daily.to_numpy()):
    raise RuntimeError("Riskfolio covariance drifted from the common daily estimator.")

# %% [markdown]
# ### Available Risk Measures
#
# The pinned Riskfolio runtime includes multiple risk families. This notebook uses:
# - **Deviation-based**: MV, MAD, MSV, GMD, KT, SKT
# - **Quantile-based**: CVaR, EVaR, RLVaR, WR
# - **Drawdown-based**: MDD, ADD, CDaR, EDaR, RLDaR, UCI
#
# Standard deviation, maximum drawdown, and CDaR appear as ratio objectives. CVaR is
# handled separately as the same minimum-risk task used by the other two libraries.

# %% [markdown]
# ### Max Sharpe with Different Risk Measures

# %%
risk_measures = {
    "MV": "Standard Deviation",
    "MDD": "Max Drawdown",
    "CDaR": "Conditional DaR",
}

riskfolio_weights = {}
for rm, name in risk_measures.items():
    result = run_riskfolio(
        lambda rm=rm: port.optimization(
            model="Classic",
            rm=rm,
            obj="Sharpe",
            rf=RISK_FREE_RATE_DAILY,
            hist=True,
        ),
        f"Max Sharpe with {name}",
    )
    weights = align_weights(result["weights"], f"RF: {name}")
    riskfolio_weights[name] = weights
    n_pos = int((weights > ACTIVE_WEIGHT_THRESHOLD).sum())
    print(f"{name}: {n_pos} positions")

# Match PyPortfolioOpt and skfolio: long-only, fully invested, minimum empirical CVaR.
result_cvar = run_riskfolio(
    lambda: port.optimization(
        model="Classic",
        rm="CVaR",
        obj="MinRisk",
        rf=RISK_FREE_RATE_DAILY,
        hist=True,
    ),
    "minimum CVaR",
)
riskfolio_weights["Minimum CVaR"] = align_weights(result_cvar["weights"], "RF: Min CVaR")
print(
    "Minimum CVaR: "
    f"{int((riskfolio_weights['Minimum CVaR'] > ACTIVE_WEIGHT_THRESHOLD).sum())} positions"
)

# %% [markdown]
# ### Risk Parity

# %%
# Risk parity: equal risk contribution from each asset
weights_rp = run_riskfolio(
    lambda: port.rp_optimization(
        model="Classic",
        rm="MV",
        rf=RISK_FREE_RATE_DAILY,
        b=None,  # Equal risk contribution
        hist=True,
    ),
    "risk parity",
)
riskfolio_weights["Risk Parity"] = align_weights(weights_rp["weights"], "RF: Risk Parity")
n_pos = int((riskfolio_weights["Risk Parity"] > ACTIVE_WEIGHT_THRESHOLD).sum())
print(f"Risk Parity: {n_pos} positions")

# %% [markdown]
# ### Efficient Frontier Comparison

# %%
# Compute efficient frontiers for different risk measures
frontier_mv = run_riskfolio(
    lambda: port.efficient_frontier(
        model="Classic",
        rm="MV",
        points=FRONTIER_POINTS,
        rf=RISK_FREE_RATE_DAILY,
        hist=True,
    ),
    "mean-variance frontier",
)
frontier_cvar = run_riskfolio(
    lambda: port.efficient_frontier(
        model="Classic",
        rm="CVaR",
        points=FRONTIER_POINTS,
        rf=RISK_FREE_RATE_DAILY,
        hist=True,
    ),
    "CVaR frontier",
)


# Convert to plottable format
def frontier_to_df(frontier, mean_returns, covariance, name):
    """Convert riskfolio frontier to DataFrame with risk-return."""
    results = []
    mean_flat = mean_returns.values.flatten()
    for col in frontier.columns:
        w = frontier[col].values
        ret = (w @ mean_flat) * TRADING_DAYS
        vol = np.sqrt(w @ covariance @ w) * np.sqrt(TRADING_DAYS)
        results.append({"return": ret, "volatility": vol, "frontier": name})
    return pd.DataFrame(results)


cov_np = port.cov.values
ef_mv = frontier_to_df(frontier_mv, port.mu, cov_np, "Mean-Variance")
ef_cvar = frontier_to_df(frontier_cvar, port.mu, cov_np, "CVaR")

# %%
# Plot both efficient frontiers
fig = go.Figure()

fig.add_scatter(
    x=ef_mv["volatility"],
    y=ef_mv["return"],
    mode="lines",
    name="Mean-Variance",
    line=dict(color=COLORS["blue"], width=3),
)

fig.add_scatter(
    x=ef_cvar["volatility"],
    y=ef_cvar["return"],
    mode="lines",
    name="CVaR",
    line=dict(color=COLORS["amber"], width=2, dash="dash"),
)

fig.update_layout(
    title="Training frontiers depend on the chosen risk measure",
    xaxis_title="Annualized volatility",
    yaxis_title="Annualized expected return",
    xaxis_tickformat=".0%",
    yaxis_tickformat=".0%",
    height=500,
)
fig.show()

# %% [markdown]
# The two curves are training diagnostics, not test performance. They show which allocations
# each risk definition considers efficient before any later return is observed.

# %% [markdown]
# ## Part 3: skfolio
#
# skfolio is the newest of the three libraries surveyed here; it exposes a sklearn-style
# fit/predict API with built-in cross-validation and hyperparameter tuning.

# %% [markdown]
# ### Mean-Risk Optimization
#
# skfolio uses sklearn-compatible estimators that can be used in pipelines.

# %%
# Max Sharpe with skfolio
model_sharpe = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.VARIANCE,
    prior_estimator=common_empirical_prior(),
    min_weights=0.0,
    max_weights=1.0,
    budget=1.0,
    risk_free_rate=RISK_FREE_RATE_DAILY,
    solver="CLARABEL",
    save_problem=True,
    raise_on_failure=True,
)
model_sharpe.fit(train_returns)
assert_optimal_status(model_sharpe.problem_, "skfolio Max Sharpe")

skfolio_distribution = model_sharpe.prior_estimator_.return_distribution_
if not np.allclose(skfolio_distribution.mu, common_mean_daily.to_numpy()):
    raise RuntimeError("skfolio expected returns drifted from the common daily estimator.")
if not np.allclose(skfolio_distribution.covariance, common_cov_daily.to_numpy()):
    raise RuntimeError("skfolio covariance drifted from the common daily estimator.")

print(f"skfolio Max Sharpe - Fitted {len(model_sharpe.weights_)} assets")
weights_skfolio_sharpe = align_weights(model_sharpe.weights_, "SKF: Max Sharpe")

# %%
# Min Variance with skfolio
model_minvar = MeanRisk(
    objective_function=ObjectiveFunction.MINIMIZE_RISK,
    risk_measure=RiskMeasure.VARIANCE,
    prior_estimator=common_empirical_prior(),
    min_weights=0.0,
    max_weights=1.0,
    budget=1.0,
    solver="CLARABEL",
    save_problem=True,
    raise_on_failure=True,
)
model_minvar.fit(train_returns)
assert_optimal_status(model_minvar.problem_, "skfolio minimum variance")

weights_skfolio_minvar = align_weights(model_minvar.weights_, "SKF: Min Var")

# %%
# CVaR optimization
model_cvar = MeanRisk(
    objective_function=ObjectiveFunction.MINIMIZE_RISK,
    risk_measure=RiskMeasure.CVAR,
    prior_estimator=common_empirical_prior(),
    min_weights=0.0,
    max_weights=1.0,
    budget=1.0,
    cvar_beta=CVAR_CONFIDENCE,
    solver="CLARABEL",
    save_problem=True,
    raise_on_failure=True,
)
model_cvar.fit(train_returns)
assert_optimal_status(model_cvar.problem_, "skfolio minimum CVaR")

weights_skfolio_cvar = align_weights(model_cvar.weights_, "SKF: Min CVaR")

# %% [markdown]
# ### Independent Common-Objective Checks
#
# The libraries use different parameter units and solver wrappers. These independent
# optimizations verify that the boundaries still represent one economic problem.


# %%
def negative_common_sharpe(weights: np.ndarray) -> float:
    """Evaluate the shared daily arithmetic excess-return-to-volatility objective."""
    excess_return = weights @ common_mean_daily.to_numpy() - RISK_FREE_RATE_DAILY
    volatility = np.sqrt(weights @ common_cov_daily.to_numpy() @ weights)
    return -float(excess_return / volatility)


common_sharpe_oracle = minimize(
    negative_common_sharpe,
    np.full(num_stocks, 1 / num_stocks),
    method="SLSQP",
    bounds=[(0.0, 1.0)] * num_stocks,
    constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
    options={"ftol": 1e-13, "maxiter": 2_000},
)
if not common_sharpe_oracle.success:
    raise RuntimeError(f"Independent Max-Sharpe oracle failed: {common_sharpe_oracle.message}")

common_sharpe_weights = {
    "PPO": weights_pypfopt_sharpe,
    "Riskfolio": riskfolio_weights["Standard Deviation"],
    "skfolio": weights_skfolio_sharpe,
}
for library, weights in common_sharpe_weights.items():
    weight_difference = float(np.max(np.abs(weights.to_numpy() - common_sharpe_oracle.x)))
    if weight_difference > COMMON_WEIGHT_TOLERANCE:
        raise RuntimeError(
            f"{library} Max-Sharpe weights differ from the common oracle by "
            f"{weight_difference:.8f}."
        )
    print(f"{library} Max-Sharpe vs independent oracle: {weight_difference:.2e}")

# %% [markdown]
# The matching CVaR task minimizes the historical loss tail at the same 95% confidence,
# with long-only weights that sum to one and no return target or ratio objective.

# %%
# Minimum empirical CVaR is a linear program over weights, the VaR threshold, and tail slacks.
training_scenarios = train_returns.to_numpy()
n_scenarios = len(training_scenarios)
tail_coefficient = 1 / ((1 - CVAR_CONFIDENCE) * n_scenarios)
cvar_objective = np.r_[
    np.zeros(num_stocks),
    1.0,
    np.full(n_scenarios, tail_coefficient),
]
cvar_inequality = np.hstack(
    [
        -training_scenarios,
        -np.ones((n_scenarios, 1)),
        -np.eye(n_scenarios),
    ]
)
cvar_oracle = linprog(
    cvar_objective,
    A_ub=cvar_inequality,
    b_ub=np.zeros(n_scenarios),
    A_eq=np.r_[np.ones(num_stocks), np.zeros(1 + n_scenarios)][None, :],
    b_eq=np.array([1.0]),
    bounds=[(0.0, 1.0)] * num_stocks + [(None, None)] + [(0.0, None)] * n_scenarios,
    method="highs",
)
if not cvar_oracle.success:
    raise RuntimeError(f"Independent minimum-CVaR oracle failed: {cvar_oracle.message}")


# %%
def empirical_cvar(weights: pd.Series) -> float:
    """Evaluate the same historical-loss CVaR minimized by the independent LP."""
    losses = -(training_scenarios @ weights.to_numpy())
    threshold = np.quantile(losses, CVAR_CONFIDENCE, method="lower")
    return float(threshold + tail_coefficient * np.maximum(losses - threshold, 0).sum())


common_cvar_weights = {
    "PPO": weights_pypfopt_cvar,
    "Riskfolio": riskfolio_weights["Minimum CVaR"],
    "skfolio": weights_skfolio_cvar,
}
for library, weights in common_cvar_weights.items():
    objective_gap = empirical_cvar(weights) - cvar_oracle.fun
    if objective_gap > COMMON_OBJECTIVE_TOLERANCE:
        raise RuntimeError(
            f"{library} minimum-CVaR objective exceeds the common oracle by {objective_gap:.8e}."
        )
    print(f"{library} minimum-CVaR objective gap: {objective_gap:.2e}")

# %% [markdown]
# ### Hierarchical Risk Parity

# %%
# HRP with skfolio
model_hrp = HierarchicalRiskParity(
    risk_measure=RiskMeasure.VARIANCE,
    distance_estimator=PearsonDistance(),
    hierarchical_clustering_estimator=HierarchicalClustering(linkage_method=LinkageMethod.WARD),
)
model_hrp.fit(train_returns)

weights_skfolio_hrp = align_weights(model_hrp.weights_, "SKF: HRP")

# %% [markdown]
# ### sklearn Integration: Walk-Forward Cross-Validation
#
# One of skfolio's key advantages is native sklearn compatibility,
# including built-in walk-forward and combinatorial purged cross-validation.
# Here we apply the same Max-Sharpe feasibility policy before every fold fit.

# %%
# Walk-forward cross-validation: one trading year for fitting, then one quarter for validation.
CV_TRAIN_SIZE = 252
CV_TEST_SIZE = 63
cv = WalkForward(train_size=CV_TRAIN_SIZE, test_size=CV_TEST_SIZE)
n_splits = cv.get_n_splits(train_returns)
print(
    f"Training-only walk-forward CV: {n_splits} splits "
    f"({CV_TRAIN_SIZE}d fit / {CV_TEST_SIZE}d validation)"
)
fold_records = []

# %% [markdown]
# Feasible folds invoke the unchanged 4% Max-Sharpe estimator and require an exact solver status.
# Infeasible folds hold cash at the same daily hurdle without calling a ratio solver.

# %%
for fold, (train_indices, test_indices) in enumerate(cv.split(train_returns)):
    fold_train = train_returns.iloc[train_indices]
    fold_test = train_returns.iloc[test_indices]
    regime = max_sharpe_regime(fold_train.mean(), RISK_FREE_RATE_DAILY)

    if regime == "cash":
        fold_returns = np.full(len(fold_test), RISK_FREE_RATE_DAILY)
        solver_status = "cash_precheck"
        active_positions = 0
    else:
        fold_model = clone(model_sharpe).fit(fold_train)
        assert_optimal_status(fold_model.problem_, f"skfolio fold {fold} Max Sharpe")
        fold_returns = fold_test.to_numpy() @ fold_model.weights_
        solver_status = fold_model.problem_.status
        active_positions = int((np.abs(fold_model.weights_) > ACTIVE_WEIGHT_THRESHOLD).sum())

    fold_volatility = float(np.std(fold_returns, ddof=1))
    fold_sharpe = (
        None
        if fold_volatility <= np.finfo(float).eps
        else float(
            np.mean(fold_returns - RISK_FREE_RATE_DAILY) / fold_volatility * np.sqrt(TRADING_DAYS)
        )
    )
    fold_records.append(
        {
            "fold": fold,
            "regime": regime,
            "solver_status": solver_status,
            "active_positions": active_positions,
            "annual_return": float(np.mean(fold_returns) * TRADING_DAYS),
            "annual_sharpe": fold_sharpe,
        }
    )

# %%
fold_summary = pl.DataFrame(fold_records)
cash_folds = fold_summary.filter(pl.col("regime") == "cash").height
if cash_folds == 0:
    raise RuntimeError("The walk-forward oracle did not exercise the predeclared cash policy.")
print(f"\nWalk-forward regimes: {n_splits - cash_folds} optimized, {cash_folds} cash")
fold_summary

# %% [markdown]
# This demonstration stays inside the training window. Cash rows are explicit feasibility
# decisions at the same 4% hurdle, while optimized rows expose the solver status and breadth.

# %%
portfolios_skf = {
    "Max Sharpe": weights_skfolio_sharpe,
    "Min Variance": weights_skfolio_minvar,
    "Min CVaR": weights_skfolio_cvar,
    "HRP": weights_skfolio_hrp,
}

skfolio_summary = pl.DataFrame(
    [
        {
            "portfolio": name,
            "positions": int((weights.abs() > ACTIVE_WEIGHT_THRESHOLD).sum()),
            "max_weight": float(weights.max()),
        }
        for name, weights in portfolios_skf.items()
    ]
)
skfolio_summary

# %% [markdown]
# These summary lines are a quick implementation check: if one model keeps producing
# extremely concentrated portfolios, it may be using the same objective as the others
# but with very different practical behavior.

# %% [markdown]
# ## Part 4: Comprehensive Comparison
#
# Let's compare all optimized portfolios using ml4t-diagnostic.

# %%
# Collect all portfolio weights
all_portfolios = {
    # PyPortfolioOpt
    "PPO: Max Sharpe": weights_pypfopt_sharpe,
    "PPO: Min Vol": weights_pypfopt_minvol,
    "PPO: Min CVaR": weights_pypfopt_cvar,
    "PPO: HRP": weights_pypfopt_hrp,
    "PPO: Shrinkage": weights_pypfopt_shrunk,
    # Riskfolio-Lib
    "RF: Std Dev": riskfolio_weights["Standard Deviation"],
    "RF: Min CVaR": riskfolio_weights["Minimum CVaR"],
    "RF: Max DD": riskfolio_weights["Max Drawdown"],
    "RF: Conditional DaR": riskfolio_weights["Conditional DaR"],
    "RF: Risk Parity": riskfolio_weights["Risk Parity"],
    # skfolio
    "SKF: Max Sharpe": weights_skfolio_sharpe,
    "SKF: Min Var": weights_skfolio_minvar,
    "SKF: Min CVaR": weights_skfolio_cvar,
    "SKF: HRP": weights_skfolio_hrp,
    # Benchmark
    "Equal Weight": pd.Series(1 / num_stocks, index=tickers),
}

# %%
# Compute portfolio returns for each strategy
test_returns_np = test_returns.values
test_dates = test_returns.index.tolist()

portfolio_returns = {}
for name, weights in all_portfolios.items():
    portfolio_returns[name] = test_returns_np @ weights.reindex(tickers).values

# %% [markdown]
# ### Evaluate with ml4t-diagnostic

# %%
# Comprehensive evaluation using PortfolioAnalysis
evaluation_results = []

for name, pf_returns in portfolio_returns.items():
    pa = PortfolioAnalysis(
        returns=pl.Series("returns", pf_returns),
        dates=pl.Series("timestamp", test_dates),
        risk_free=RISK_FREE_RATE,
        periods_per_year=TRADING_DAYS,
    )

    metrics = pa.compute_summary_stats()

    evaluation_results.append(
        {
            "portfolio": name,
            "library": name.split(":", maxsplit=1)[0] if ":" in name else "Benchmark",
            "annual_return": metrics.annual_return,
            "annual_volatility": metrics.annual_volatility,
            "sharpe": metrics.sharpe_ratio,
            "sortino": metrics.sortino_ratio,
            "calmar": metrics.calmar_ratio,
            "max_drawdown": metrics.max_drawdown,
            "var_95": metrics.var_95,
            "cvar_95": metrics.cvar_95,
            "win_rate": metrics.win_rate,
        }
    )

eval_df = pl.DataFrame(evaluation_results).sort("sharpe", descending=True)
eval_df

# %% [markdown]
# These metrics describe frozen allocations on later returns. They support comparison of
# implementations, but this single historical test is not a license to select a permanent winner.

# %% [markdown]
# ### Execution-Aware Bridge with ml4t-backtest
#
# Vectorized matrix multiplication is useful for comparing optimizers under identical assumptions.
# To connect this to deployable execution, replay one optimized portfolio through Engine.

# %% [markdown]
# The bridge strategy restores the same frozen target each day. The engine then adds
# next-bar timing, slippage, and commissions without changing the allocation policy.


# %%
class DailyTargetWeightStrategy(Strategy):
    def __init__(self, target_weights: dict[str, float], allow_short: bool):
        self.target_weights = target_weights
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(
                min_trade_value=0.0,
                min_weight_change=0.0,
                allow_fractional=True,
                allow_short=allow_short,
            )
        )

    def on_data(self, timestamp, data, context, broker):
        targets = {asset: weight for asset, weight in self.target_weights.items() if asset in data}
        if targets:
            self.executor.execute(targets, data, broker)


# %%
# Build engine inputs from the selected library portfolio and the price panel.
bridge_name = "PPO: Max Sharpe"
engine_target_weights = {
    ticker: float(weight)
    for ticker, weight in all_portfolios[bridge_name].items()
    if abs(float(weight)) > 1e-8
}
allow_short_engine = any(weight < 0 for weight in engine_target_weights.values())

test_prices_long = (
    etf_data.filter(pl.col("timestamp") > pl.lit(TRAIN_END).str.to_date())
    .select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    .drop_nulls()
    .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    .sort(["timestamp", "symbol"])
)

# %%
# Run the execution-aware simulation and collect daily returns.
engine = Engine(
    feed=DataFeed(prices_df=test_prices_long),
    strategy=DailyTargetWeightStrategy(engine_target_weights, allow_short=allow_short_engine),
    config=BacktestConfig(
        initial_cash=100_000.0,
        execution_mode=ExecutionMode.NEXT_BAR,
        commission_type=CommissionType.PERCENTAGE,
        commission_rate=COMMISSION_RATE,
        slippage_type=SlippageType.PERCENTAGE,
        slippage_rate=SLIPPAGE_RATE,
        allow_short_selling=allow_short_engine,
    ),
)

engine_daily = (
    engine.run()
    .to_daily_pnl()
    .select(
        pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
        pl.col("return_pct").alias("engine_return"),
    )
)
# NEXT_BAR cannot hold the target during the first test return. Exclude that warm-up
# observation from both paths, then require an identical one-to-one scored date set.
warmup_timestamp = pl.Series("timestamp", [test_dates[0]]).cast(pl.Datetime("us")).item()
vectorized_daily = pl.DataFrame(
    {
        "timestamp": pl.Series(test_dates[1:]).cast(pl.Datetime("us")),
        "vectorized_return": portfolio_returns[bridge_name][1:],
    }
)
engine_scored = engine_daily.filter(pl.col("timestamp") > warmup_timestamp).sort("timestamp")

if engine_scored["timestamp"].to_list() != vectorized_daily["timestamp"].to_list():
    raise RuntimeError("Engine and vectorized bridge do not contain identical scored bars.")

# %%
# Compare vectorized and engine results on the asserted common date set.
bridge = (
    vectorized_daily.join(engine_scored, on="timestamp", how="inner", validate="1:1")
    .drop_nulls(["vectorized_return", "engine_return"])
    .sort("timestamp")
)
if bridge.height != len(test_dates) - 1:
    raise RuntimeError("Execution bridge lost rows after the matched-bar assertion.")

vec_pa = PortfolioAnalysis(
    returns=bridge["vectorized_return"],
    dates=bridge["timestamp"],
    risk_free=RISK_FREE_RATE,
    periods_per_year=TRADING_DAYS,
)
eng_pa = PortfolioAnalysis(
    returns=bridge["engine_return"],
    dates=bridge["timestamp"],
    risk_free=RISK_FREE_RATE,
    periods_per_year=TRADING_DAYS,
)
vec_stats = vec_pa.compute_summary_stats()
eng_stats = eng_pa.compute_summary_stats()

print(f"Execution bridge ({bridge_name}):")
print(
    f"  Matched bars={bridge.height}, "
    f"window={bridge['timestamp'].min().date()} to {bridge['timestamp'].max().date()}"
)
print(
    f"  Vectorized Sharpe={vec_stats.sharpe_ratio:.3f}, Engine Sharpe={eng_stats.sharpe_ratio:.3f}"
)
print(f"  Vectorized MaxDD={vec_stats.max_drawdown:.2%}, Engine MaxDD={eng_stats.max_drawdown:.2%}")

# %% [markdown]
# The first test return is an explicit NEXT_BAR warm-up and is absent from both scored paths.
# Every reported bridge observation therefore has prior target exposure in the vectorized and
# Engine paths; any remaining gap reflects fills and declared costs on identical bars.

# %% [markdown]
# ### Visualization: Portfolio Comparison
#
# The growth chart focuses on the comparable Max-Sharpe implementations and an equal-weight
# benchmark. Showing four lines preserves the cross-library comparison without a thirteen-line
# legend obscuring the evidence.

# %%
growth_methods = ["PPO: Max Sharpe", "RF: Std Dev", "SKF: Max Sharpe", "Equal Weight"]
growth_colors = {
    "PPO: Max Sharpe": COLORS["blue"],
    "RF: Std Dev": COLORS["amber"],
    "SKF: Max Sharpe": COLORS["copper"],
    "Equal Weight": COLORS["neutral"],
}
cumulative_growth = {name: np.cumprod(1 + portfolio_returns[name]) for name in growth_methods}
growth_leader = max(cumulative_growth, key=lambda name: cumulative_growth[name][-1])

fig = go.Figure()
for name in growth_methods:
    fig.add_scatter(
        x=test_dates,
        y=cumulative_growth[name],
        mode="lines",
        name=name,
        line=dict(
            color=growth_colors[name],
            width=3 if name == growth_leader else 2,
            dash="dash" if name == "Equal Weight" else "solid",
        ),
    )

fig.update_layout(
    title=f"{growth_leader} leads growth across comparable frozen allocations",
    xaxis_title="Test timestamp",
    yaxis_title="Growth of $1 (multiple)",
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig.show()

# %% [markdown]
# The risk-return map retains every configuration but uses color only for library identity.
# Hover labels carry the optimizer name, avoiding a thirteen-color legend.

# %%
eval_pd = eval_df.to_pandas()
library_order = ["PPO", "RF", "SKF", "Benchmark"]
library_colors = dict(zip(library_order, ml4t_palette(4, categorical=True), strict=True))
test_leader = str(eval_df.row(0, named=True)["portfolio"])

fig = go.Figure()
for library in library_order:
    subset = eval_pd.loc[eval_pd["library"] == library]
    fig.add_scatter(
        x=subset["annual_volatility"],
        y=subset["annual_return"],
        mode="markers",
        name=library,
        text=subset["portfolio"],
        customdata=subset[["sharpe", "max_drawdown"]],
        marker=dict(color=library_colors[library], size=11, line=dict(width=1)),
        hovertemplate=(
            "%{text}<br>Annual return=%{y:.1%}<br>Annual volatility=%{x:.1%}"
            "<br>Sharpe=%{customdata[0]:.2f}<br>Max drawdown=%{customdata[1]:.1%}<extra></extra>"
        ),
    )

fig.update_layout(
    title=f"{test_leader} has the highest Sharpe on the frozen test window",
    xaxis_title="Annualized volatility",
    yaxis_title="Annualized return",
    xaxis_tickformat=".0%",
    yaxis_tickformat=".0%",
    height=500,
)
fig.show()

# %% [markdown]
# Overlapping points reveal when API choice matters less than objective choice. The chart reports
# a historical test, while the training-only walk-forward exercise provides the stability context.

# %% [markdown]
# A rank heatmap compares unlike metrics without pretending their raw scales are commensurate.
# Higher ranks are better for every displayed column, including less-negative loss measures.

# %%
metrics_cols = ["sharpe", "sortino", "calmar", "max_drawdown", "var_95"]
metric_labels = ["Sharpe", "Sortino", "Calmar", "Max drawdown", "VaR 95%"]
heatmap_data = eval_pd.set_index("portfolio")[metrics_cols]
ranked = heatmap_data.rank(axis=0)
consistency_leader = str(ranked.mean(axis=1).idxmax())

fig = go.Figure(
    data=go.Heatmap(
        z=ranked.values,
        x=metric_labels,
        y=ranked.index,
        text=np.rint(ranked.values).astype(int),
        texttemplate="%{text}",
        colorscale=[
            [0, COLORS["bg_light"]],
            [0.5, COLORS["blue_light"]],
            [1, COLORS["blue"]],
        ],
        zmin=1,
        zmax=len(ranked),
        colorbar=dict(title="Rank<br>(higher is better)"),
        hovertemplate="%{y}<br>%{x}: rank %{z:.0f}<extra></extra>",
    )
)
fig.update_layout(
    title=f"{consistency_leader} ranks most consistently across test metrics",
    xaxis_title="Test metric",
    height=620,
    margin=dict(l=150, r=80, t=90, b=60),
)
fig.show()

# %% [markdown]
# Consistency across metrics is more informative than winning one column, but it remains a
# diagnostic of this test period rather than a second selection stage.

# %% [markdown]
# ### Weight Distribution Comparison

# %%
concentration_stats = []

for name, weights in all_portfolios.items():
    values = weights.reindex(tickers).to_numpy()
    n_positions = int((np.abs(values) > ACTIVE_WEIGHT_THRESHOLD).sum())
    max_weight = float(np.max(values))
    top5_weight = float(np.sort(values)[-5:].sum())
    hhi = float((values**2).sum())

    concentration_stats.append(
        {
            "portfolio": name,
            "positions": n_positions,
            "max_weight": max_weight,
            "top5_weight": top5_weight,
            "hhi": hhi,
        }
    )

conc_df = pl.DataFrame(concentration_stats).sort("hhi", descending=True)
conc_df

# %% [markdown]
# HHI turns visual concentration into a comparable statistic. A value near the equal-weight
# reference indicates broad diversification; larger values expose greater single-name dependence.

# %% [markdown]
# ### Concentration by Frozen Allocation

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    horizontal_spacing=0.08,
    subplot_titles=["Active positions", "Herfindahl-Hirschman index"],
)
portfolios = conc_df["portfolio"].to_list()
lowest_allocator = (
    conc_df.filter(pl.col("portfolio") != "Equal Weight")
    .sort("hhi")
    .row(0, named=True)["portfolio"]
)
equal_weight_hhi = 1 / num_stocks

# %% [markdown]
# Horizontal bars keep all portfolio labels readable. The second panel adds the equal-weight
# HHI as a reference rather than treating the benchmark as another optimized method.

# %%
fig.add_bar(
    x=conc_df["positions"].to_list(),
    y=portfolios,
    orientation="h",
    name="Positions",
    marker_color=COLORS["blue"],
    row=1,
    col=1,
)

fig.add_bar(
    x=conc_df["hhi"].to_list(),
    y=portfolios,
    orientation="h",
    name="HHI",
    marker_color=COLORS["amber"],
    row=1,
    col=2,
)

fig.add_vline(
    x=equal_weight_hhi,
    line_dash="dash",
    line_color=COLORS["neutral"],
    annotation_text=f"EW reference {equal_weight_hhi:.3f}",
    annotation_position="bottom right",
    row=1,
    col=2,
)

fig.update_layout(
    title=f"{lowest_allocator} is the least concentrated optimized allocation",
    height=600,
    showlegend=False,
    margin=dict(l=150, r=40, t=100, b=60),
)
fig.update_xaxes(title_text="Count", row=1, col=1, rangemode="tozero")
fig.update_xaxes(title_text="HHI (0 to 1)", row=1, col=2, rangemode="tozero")
fig.show()

# %% [markdown]
# ## Part 5: Practical Considerations
#
# PyPortfolioOpt exposes objective penalties directly. Comparing two matched pairs shows how a
# turnover penalty changes trading distance and how L2 regularization changes Max-Sharpe breadth.
# The library warns that this deliberate objective combination uses its transformed formulation;
# only that exact warning is scoped to the regularized call below.

# %%
ef = EfficientFrontier(mu, S)
initial_weights = np.full(num_stocks, 1 / num_stocks)
ef.add_objective(
    objective_functions.transaction_cost,
    w_prev=initial_weights,
    k=TRANSACTION_COST_PENALTY,
)
weights_with_cost = align_weights(ef.min_volatility(), "PPO: Min Vol with turnover penalty")
assert_optimal_status(ef._opt, "PPO minimum volatility with turnover penalty")

ef_no_cost = EfficientFrontier(mu, S)
weights_no_cost = align_weights(ef_no_cost.min_volatility(), "PPO: Min Vol without penalty")
assert_optimal_status(ef_no_cost._opt, "PPO minimum volatility without penalty")

turnover_with = np.abs(weights_with_cost.values - initial_weights).sum()
turnover_without = np.abs(weights_no_cost.values - initial_weights).sum()
turnover_reduction = 1 - turnover_with / turnover_without

ef_sharpe_unregularized = EfficientFrontier(mu, S)
weights_sharpe_unregularized = align_weights(
    ef_sharpe_unregularized.max_sharpe(risk_free_rate=RISK_FREE_RATE),
    "PPO: Unregularized Max Sharpe",
)
assert_optimal_status(ef_sharpe_unregularized._opt, "PPO unregularized Max Sharpe")

ef_reg = EfficientFrontier(mu, S)
ef_reg.add_objective(objective_functions.L2_reg, gamma=L2_GAMMA)
with suppress_ppo_max_sharpe_objective_warning():
    weights_regularized = align_weights(
        ef_reg.max_sharpe(risk_free_rate=RISK_FREE_RATE),
        "PPO: Regularized Max Sharpe",
    )
assert_optimal_status(ef_reg._opt, "PPO regularized Max Sharpe")
unregularized_positions = int((weights_sharpe_unregularized > ACTIVE_WEIGHT_THRESHOLD).sum())
regularized_positions = int((weights_regularized > ACTIVE_WEIGHT_THRESHOLD).sum())

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Turnover from equal weight", "Active Max-Sharpe positions"],
)
fig.add_bar(
    x=["No penalty", "Turnover penalty"],
    y=[turnover_without, turnover_with],
    marker_color=[COLORS["neutral"], COLORS["blue"]],
    showlegend=False,
    row=1,
    col=1,
)
fig.add_bar(
    x=["Unregularized", "L2 regularized"],
    y=[unregularized_positions, regularized_positions],
    marker_color=[COLORS["neutral"], COLORS["amber"]],
    showlegend=False,
    row=1,
    col=2,
)
fig.update_layout(
    title=f"The turnover penalty cuts trading distance by {turnover_reduction:.0%}",
    height=430,
)
fig.update_yaxes(title_text="One-way turnover", tickformat=".0%", rangemode="tozero", row=1, col=1)
fig.update_yaxes(
    title_text=f"Positions above {ACTIVE_WEIGHT_THRESHOLD:.1%}",
    rangemode="tozero",
    row=1,
    col=2,
)
fig.show()

# %% [markdown]
# ## API Ergonomics Comparison
#
# Using all three libraries on the same training and test windows reveals different workflow
# strengths without treating a one-period ranking as permanent:
#
# **PyPortfolioOpt** has the most intuitive API for standard tasks. Creating an
# `EfficientFrontier`, calling `max_sharpe()`, and inspecting `portfolio_performance()`
# requires minimal boilerplate. Its built-in objective penalties make turnover and
# regularization experiments explicit.
#
# **Riskfolio-Lib** exposes multiple risk families through a single `Portfolio` object.
# The `optimization()` method
# accepts string codes for risk measures (`"MV"`, `"CVaR"`, `"MDD"`), making it easy
# to sweep across objectives programmatically and compare training frontiers.
#
# **skfolio** stands out for ML integration. Models are sklearn estimators with `fit()` /
# `predict()` semantics, meaning they slot into `Pipeline`, `GridSearchCV`, and
# walk-forward cross-validation without adapters. This is a decisive advantage when
# portfolio construction is one stage in a larger ML workflow.
#
# | Criterion | PyPortfolioOpt | Riskfolio-Lib | skfolio |
# |-----------|----------------|---------------|---------|
# | Optimizer-object workflow | Native | Portfolio object | Estimator object |
# | Multiple risk families | Selected classes | Unified interface | Selected estimators |
# | ML pipeline integration | Manual | Manual | Native sklearn style |
# | Cross-validation | Manual | Manual | Built-in (WalkForward, CPCV) |
# | Objective penalties | Native | Via model settings | Via constraints/settings |

# %% [markdown]
# ## Key Takeaways

# %%
ppo_test_sharpe = float(eval_df.filter(pl.col("portfolio") == "PPO: Max Sharpe")["sharpe"].item())
skfolio_test_sharpe = float(
    eval_df.filter(pl.col("portfolio") == "SKF: Max Sharpe")["sharpe"].item()
)
max_sharpe_test_gap = abs(ppo_test_sharpe - skfolio_test_sharpe)
display(
    Markdown(
        "\n".join(
            [
                f"- **Matched Max-Sharpe implementations agree here**: PyPortfolioOpt records "
                f"{ppo_test_sharpe:.6f} and skfolio {skfolio_test_sharpe:.6f} on the frozen test, "
                f"an absolute gap of {max_sharpe_test_gap:.6f} under the common contract.",
                f"- **Execution changes the realized path**: the matched daily-target bridge "
                f"moves Sharpe from {vec_stats.sharpe_ratio:.3f} vectorized to "
                f"{eng_stats.sharpe_ratio:.3f} with {COMMISSION_RATE * 1e4:.0f} bp commission "
                f"and {SLIPPAGE_RATE * 1e4:.0f} bp slippage.",
                f"- **Turnover belongs in the objective**: the declared penalty reduces "
                f"one-way trading distance from {turnover_without:.1%} to {turnover_with:.1%}.",
                f"- **Regularization changes breadth**: an L2 penalty of {L2_GAMMA:.1f} moves "
                f"the active Max-Sharpe allocation from {unregularized_positions} to "
                f"{regularized_positions} positions.",
                "- **Workflow fit remains the durable distinction**: optimizer objects, broad "
                "risk interfaces, and sklearn-style validation solve different research needs.",
            ]
        )
    )
)

# %% [markdown]
# **Next**: [`09_allocator_comparison`](09_allocator_comparison.ipynb) extends the comparison
# with explicit estimation-risk controls.
#
# **Book**: Chapter 17, §17.7 develops the controlled allocator comparison framework.
