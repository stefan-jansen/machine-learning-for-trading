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
# # Portfolio Allocator Comparison
#
# **Docker image**: `ml4t`
#
# This capstone notebook compares allocation methods on equal footing, using identical
# data and backtest protocols to isolate the impact of the allocation choice.
# Equal-weight and inverse-volatility baselines run against MVO and HRP on the same signals.
#
# **Learning Objectives**:
# - Run controlled comparisons of allocation methods (EW, IV, MVO, HRP)
# - Interpret walk-forward backtest results with regime-sliced diagnostics
# - Evaluate methods on Sharpe, drawdown, turnover, and stability
# - Understand when allocation choice matters vs when signal dominates
#
# **Book Reference**: Chapter 17, §17.4 (Baseline allocators) and §17.7 (Comparing allocator performance)
#
# **Prerequisites**: `02_mean_variance_optimization`, `06_hierarchical_risk_parity`
# %% [markdown]
# ## 1. Setup and Imports

# %%
"""Compare MVO, HRP, inverse volatility, and equal weight on ETF data."""

# %%
from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from IPython.display import Markdown, display
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
from ml4t.diagnostic.visualization import create_portfolio_dashboard
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from case_studies.utils.backtest_loaders import compute_allocator_metrics
from data import load_etfs
from utils.paths import get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, ml4t_diverging, ml4t_palette

# %% tags=["parameters"]
# Production defaults; Papermill overrides these values for CI testing
MAX_SYMBOLS = 0  # 0 = all
START_DATE = "2016-01-01"
END_DATE = "2024-01-01"
SELECTION_END = "2021-12-31"
HOLDOUT_START = "2022-01-01"
LABEL_HORIZON = 5
ALLOCATION_WINDOW = 252
TURNOVER_INCLUDE_INITIAL = True
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## 2. Data Acquisition
#
# We use a diversified ETF universe spanning equities, fixed income, alternatives, and sectors.

# %%
# Full universe using diversified ETFs
US_EQUITIES = ["SPY", "QQQ", "IWM", "VTV", "VUG", "MDY", "IWF", "IWD"]
INTERNATIONAL = ["EFA", "EEM", "VEA", "VWO", "IEFA", "IEMG"]
FIXED_INCOME = ["AGG", "TLT", "LQD", "HYG", "TIP", "SHY", "BND", "IEF", "EMB", "MBB"]
ALTERNATIVES = ["GLD", "SLV", "VNQ", "DBC", "IAU"]
SECTORS = ["XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE"]
_FULL_UNIVERSE = [*US_EQUITIES, *INTERNATIONAL, *FIXED_INCOME, *ALTERNATIVES, *SECTORS]

# %%
# Configuration
UNIVERSE = _FULL_UNIVERSE[:MAX_SYMBOLS] if MAX_SYMBOLS else _FULL_UNIVERSE
BACKTEST_START = "2018-01-01"

print(f"Loading data for {len(UNIVERSE)} ETFs...")


# %% [markdown]
# ### Load ETF Price Panel
#
# This helper enforces a consistent data source and panel shape across all allocation methods.


# %%
def fetch_etf_data(symbols: list[str], start: str, end: str) -> tuple[pd.DataFrame, pl.DataFrame]:
    """Load a modeling panel and canonical OHLCV bars from the ETF dataset."""
    etf_data = load_etfs()
    etf_filtered = etf_data.filter(
        (pl.col("symbol").is_in(symbols))
        & (pl.col("timestamp") >= pl.lit(start).str.to_date())
        & (pl.col("timestamp") <= pl.lit(end).str.to_date())
    )

    bars = (
        etf_filtered.select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        .drop_nulls()
        .sort(["timestamp", "symbol"])
    )
    close_prices = (
        etf_filtered.select(["timestamp", "symbol", "close"])
        .pivot(on="symbol", index="timestamp", values="close")
        .sort("timestamp")
        .to_pandas()
        .set_index("timestamp")
        .ffill()
        .dropna()
    )
    return close_prices, bars


close_prices, etf_bars = fetch_etf_data(UNIVERSE, START_DATE, END_DATE)
returns = close_prices.pct_change().dropna()
print(f"Loaded {len(returns):,} days for {close_prices.shape[1]} ETFs")

loaded_symbols = close_prices.columns.tolist()
missing_symbols = sorted(set(UNIVERSE) - set(loaded_symbols))
coverage = (
    etf_bars.group_by("symbol")
    .agg(
        pl.len().alias("observations"),
        pl.col("timestamp").min().alias("first_timestamp"),
        pl.col("timestamp").max().alias("last_timestamp"),
    )
    .sort("symbol")
)
print(f"Requested symbols: {len(UNIVERSE)}; complete-panel symbols: {len(loaded_symbols)}")
print(f"Missing from the complete panel: {missing_symbols or 'none'}")
coverage

# %% [markdown]
# **Universe limitation**: This is a fixed current teaching list applied historically, not a
# point-in-time ETF membership file. The coverage table makes missing symbols and inception dates
# visible, but the comparison remains subject to survivorship and availability bias.


# %%
def select_feature_config(n_observations: int) -> dict[str, list[int] | int]:
    """Scale feature windows to the available history."""
    if n_observations >= 252:
        return {"momentum": [21, 63, 126], "moving_average": [21, 63], "volatility": 21}
    if n_observations >= 126:
        return {"momentum": [10, 21, 42], "moving_average": [10, 21], "volatility": 10}
    return {"momentum": [5, 10, 21], "moving_average": [5, 10], "volatility": 5}


FEATURE_CONFIG = select_feature_config(len(close_prices))
FEATURE_SUFFIXES = [
    *(f"_mom_{window}d" for window in FEATURE_CONFIG["momentum"]),
    *(f"_ma_dist_{window}d" for window in FEATURE_CONFIG["moving_average"]),
    f"_vol_{FEATURE_CONFIG['volatility']}d",
]
print(f"Feature windows: {FEATURE_CONFIG}")

# %% [markdown]
# ## 3. Generate a Common ETF Signal
#
# We use Ridge regression to predict forward returns and select the top/bottom ETFs.


# %%
def compute_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute predictive features."""
    feature_columns = {}
    momentum_windows = FEATURE_CONFIG["momentum"]
    moving_average_windows = FEATURE_CONFIG["moving_average"]
    volatility_window = FEATURE_CONFIG["volatility"]

    for symbol in prices_df.columns:
        close = prices_df[symbol]

        # Momentum
        for window in momentum_windows:
            feature_columns[f"{symbol}_mom_{window}d"] = close.pct_change(window)

        # Mean reversion
        for window in moving_average_windows:
            feature_columns[f"{symbol}_ma_dist_{window}d"] = (
                close / close.rolling(window).mean() - 1
            )

        # Volatility
        feature_columns[f"{symbol}_vol_{volatility_window}d"] = (
            close.pct_change().rolling(volatility_window).std()
        )

    return pd.DataFrame(feature_columns, index=prices_df.index).dropna()


# %% [markdown]
# ### Ridge Signal Generation
#
# Signals are re-estimated on each rebalance date using rolling cross-sectional samples. The target
# is the cumulative return from close `t` through close `t+5`. A row enters a fit only after its
# label end date has been observed, so the five-observation horizon is purged at every rebalance.

# %% [markdown]
# #### Feature Row Helper


# %%
def get_feature_row(features_df, symbol, date):
    """Fetch one symbol's feature vector for a given date."""
    feature_cols = [f"{symbol}{suffix}" for suffix in FEATURE_SUFFIXES]
    if not all(col in features_df.columns for col in feature_cols):
        return None
    row = features_df.loc[date, feature_cols].values
    return row if len(row) == len(FEATURE_SUFFIXES) else None


# %% [markdown]
# #### Rolling Sample Builder


# %%
def build_training_samples(
    features_df,
    forward_returns,
    symbols,
    dates,
    train_start,
    train_end,
    horizon,
    label_cutoff,
):
    """Build cross-sectional rolling samples for one rebalance step."""
    x_train, y_train = [], []
    for t in range(train_start, train_end):
        target_date = dates[t + horizon] if t + horizon < len(dates) else None
        if target_date is None or target_date > label_cutoff:
            continue
        for symbol in symbols:
            features = get_feature_row(features_df, symbol, dates[t])
            if features is None:
                continue
            target = (
                forward_returns.loc[dates[t], symbol]
                if symbol in forward_returns.columns
                else np.nan
            )
            if not np.isnan(features).any() and not np.isnan(target):
                x_train.append(features)
                y_train.append(target)
    return np.array(x_train), np.array(y_train)


# %% [markdown]
# #### Cross-Section Predictor


# %%
def predict_cross_section(features_df, symbols, signal_date, scaler, model):
    """Generate per-symbol predictions for one signal date."""
    predictions = {}
    for symbol in symbols:
        features = get_feature_row(features_df, symbol, signal_date)
        if features is None or np.isnan(features).any():
            continue
        features_scaled = scaler.transform(features.reshape(1, -1))
        predictions[symbol] = model.predict(features_scaled)[0]
    return predictions


# %% [markdown]
# #### Ridge Fit Helper


# %%
def fit_ridge_model(x_train, y_train):
    """Fit one cross-sectional Ridge model and return scaler + model."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(x_train_scaled, y_train)
    return scaler, model


# %% [markdown]
# #### Signal Row Assembler


# %%
def append_signal_rows(
    all_signals, symbols, signal_date, predictions, long_stocks, short_stocks, top_n, bottom_n
):
    """Append one rebalance date's signal rows."""
    for symbol in symbols:
        weight = 0.0
        if symbol in long_stocks:
            weight = 1.0 / top_n
        elif symbol in short_stocks:
            weight = -1.0 / bottom_n
        all_signals.append(
            {
                "timestamp": signal_date,
                "symbol": symbol,
                "prediction": predictions.get(symbol, np.nan),
                "ml_signal": weight,
            }
        )


# %% [markdown]
# #### Rebalance Step Helper
#
# Process one rebalance date: train model, predict, select long/short, and record signals.


# %%
def _process_rebalance_step(
    features_df,
    forward_returns,
    symbols,
    dates,
    i,
    lookback,
    horizon,
    top_n,
    bottom_n,
    all_signals,
):
    """Train, predict, and record signals for one rebalance date."""
    train_start = max(0, i - lookback)
    x_train, y_train = build_training_samples(
        features_df=features_df,
        forward_returns=forward_returns,
        symbols=symbols,
        dates=dates,
        train_start=train_start,
        train_end=i,
        horizon=horizon,
        label_cutoff=dates[i],
    )
    if len(x_train) < 100:
        return
    scaler, model = fit_ridge_model(x_train, y_train)
    predictions = predict_cross_section(features_df, symbols, dates[i], scaler, model)
    effective_n = min(top_n, bottom_n, max(1, len(predictions) // 2))
    if effective_n == 0:
        return
    pred_series = pd.Series(predictions).sort_values(ascending=False)
    append_signal_rows(
        all_signals,
        symbols,
        dates[i],
        predictions,
        pred_series.head(effective_n).index.tolist(),
        pred_series.tail(effective_n).index.tolist(),
        effective_n,
        effective_n,
    )


# %% [markdown]
# #### Walk-Forward Signal Loop


# %%
def generate_ml_signals(
    prices: pd.DataFrame,
    lookback: int = 252,
    horizon: int = LABEL_HORIZON,
    top_n: int = 10,
    bottom_n: int = 10,
    rebalance_freq: int = 21,
) -> pd.DataFrame:
    """Generate long/short signal weights using walk-forward Ridge regression."""
    forward_returns = prices.pct_change(horizon).shift(-horizon)
    symbols = prices.columns.tolist()
    features_df = compute_features(prices)
    all_signals = []
    dates = features_df.index.tolist()

    for i in range(lookback, len(dates) - horizon, rebalance_freq):
        if dates[i] < pd.Timestamp(BACKTEST_START):
            continue
        _process_rebalance_step(
            features_df,
            forward_returns,
            symbols,
            dates,
            i,
            lookback,
            horizon,
            top_n,
            bottom_n,
            all_signals,
        )
    return pd.DataFrame(all_signals)


# %%
print("Generating ML signals (this may take a minute)...")
if len(close_prices) >= 504:
    LOOKBACK, REBALANCE_FREQ, TOP_N, BOTTOM_N = 252, 21, 10, 10
elif len(close_prices) >= 180:
    LOOKBACK, REBALANCE_FREQ, TOP_N, BOTTOM_N = 126, 10, 5, 5
else:
    LOOKBACK, REBALANCE_FREQ, TOP_N, BOTTOM_N = 42, 5, 3, 3

max_side = max(1, close_prices.shape[1] // 4)
TOP_N = min(TOP_N, max_side)
BOTTOM_N = min(BOTTOM_N, max_side)

print(
    "Signal config:",
    {
        "lookback": LOOKBACK,
        "rebalance_freq": REBALANCE_FREQ,
        "top_n": TOP_N,
        "bottom_n": BOTTOM_N,
    },
)

ml_signals = generate_ml_signals(
    close_prices, lookback=LOOKBACK, top_n=TOP_N, bottom_n=BOTTOM_N, rebalance_freq=REBALANCE_FREQ
)
print(f"Generated {len(ml_signals):,} signal records")

if len(ml_signals) == 0:
    raise ValueError(
        "No ML signals generated from the available ETF panel.\n"
        f"Observations: {len(close_prices)}, assets: {close_prices.shape[1]}, "
        f"feature windows: {FEATURE_CONFIG}, lookback: {LOOKBACK}.\n"
        "This notebook requires enough history to estimate features and fit the walk-forward "
        "cross-sectional ridge model."
    )

# Pivot to wide format
if len(ml_signals) > 0:
    signal_dates = ml_signals["timestamp"].unique()
    print(f"Signal dates: {len(signal_dates)}")
else:
    signal_dates = []
    print("[TEST] No signal dates available")

# %% [markdown]
# ## 4. Portfolio Allocation Methods


# %% [markdown]
# ### Equal Weight


# %%
def equal_weight_allocation(selected: list[str], n_assets: int) -> dict[str, float]:
    """Equal weight among selected assets."""
    weight = 1.0 / len(selected) if selected else 0.0
    return dict.fromkeys(selected, weight)


# %% [markdown]
# ### Inverse Volatility


# %%
def inverse_vol_allocation(
    returns: pd.DataFrame,
    selected: list[str],
    window: int = 63,
) -> dict[str, float]:
    """Inverse volatility weighting."""
    vols = returns[selected].iloc[-window:].std()
    inv_vols = 1 / vols
    inv_vols = inv_vols.replace([np.inf, -np.inf], 0).fillna(0)
    total = inv_vols.sum()
    if total > 0:
        weights = inv_vols / total
    else:
        weights = pd.Series(1 / len(selected), index=selected)
    return weights.to_dict()


# %% [markdown]
# ### Minimum-Variance (Ledoit-Wolf)


# %%
def mvo_lw_allocation(
    returns: pd.DataFrame,
    selected: list[str],
    window: int = 252,
) -> dict[str, float]:
    """Minimum variance with Ledoit-Wolf shrinkage."""
    subset_returns = returns[selected].iloc[-window:].dropna(axis=1, how="any")

    if len(subset_returns.columns) < 2:
        return equal_weight_allocation(selected, len(selected))

    # Fit Ledoit-Wolf
    lw = LedoitWolf().fit(subset_returns)
    cov_shrunk = lw.covariance_

    n_assets = len(subset_returns.columns)
    initial_weights = np.full(n_assets, 1.0 / n_assets)
    result = minimize(
        fun=lambda weights: float(weights @ cov_shrunk @ weights),
        x0=initial_weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}],
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    if not result.success:
        raise RuntimeError(f"Long-only minimum-variance solve failed: {result.message}")

    weights = np.asarray(result.x, dtype=float)
    if (
        not np.isfinite(weights).all()
        or abs(weights.sum() - 1.0) > 1e-8
        or weights.min() < -1e-8
        or weights.max() > 1.0 + 1e-8
    ):
        raise RuntimeError("Long-only minimum-variance solve returned infeasible weights")
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum()
    return dict(zip(subset_returns.columns, weights, strict=False))


# %% [markdown]
# ### Hierarchical Risk Parity (HRP)
#
# HRP uses hierarchical clustering on the correlation matrix to group similar assets,
# then allocates via recursive bisection. We split the implementation into a cluster
# variance helper and the main allocation function.


# %%
def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
    """Inverse-variance-weighted portfolio variance for a cluster of assets."""
    if len(indices) == 1:
        return cov[indices[0], indices[0]]
    c = cov[np.ix_(indices, indices)]
    ivp = 1 / np.diag(c)
    ivp /= ivp.sum()
    return np.dot(ivp, np.dot(c, ivp))


# %% [markdown]
# #### Recursive Bisection Helper


# %%
def _recursive_bisect(cov: np.ndarray, sorted_idx: list[int]) -> np.ndarray:
    """Allocate weights via recursive bisection on clustered covariance matrix."""
    n = len(sorted_idx)
    weights = np.ones(n)
    clusters = [sorted_idx]
    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            left_var = _cluster_variance(cov, left)
            right_var = _cluster_variance(cov, right)
            alpha = 1 - left_var / (left_var + right_var) if (left_var + right_var) > 0 else 0.5
            for i in left:
                weights[sorted_idx.index(i)] *= alpha
            for i in right:
                weights[sorted_idx.index(i)] *= 1 - alpha
            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)
        clusters = new_clusters
    final_weights = np.zeros(n)
    for i, orig_idx in enumerate(sorted_idx):
        final_weights[orig_idx] = weights[i]
    return final_weights / final_weights.sum()


# %% [markdown]
# #### HRP Allocation Function


# %%
def hrp_allocation(
    returns: pd.DataFrame,
    selected: list[str],
    window: int = 252,
) -> dict[str, float]:
    """Hierarchical Risk Parity allocation."""
    subset_returns = returns[selected].iloc[-window:].dropna(axis=1, how="any")
    if len(subset_returns.columns) < 2:
        return equal_weight_allocation(selected, len(selected))

    # Correlation distance and clustering
    corr = subset_returns.corr().values
    dist = np.sqrt(0.5 * (1 - corr))
    link = linkage(squareform(dist, checks=False), method="ward")
    sorted_idx = list(leaves_list(link))

    # Recursive bisection
    cov = subset_returns.cov().values
    final_weights = _recursive_bisect(cov, sorted_idx)
    return dict(zip(subset_returns.columns, final_weights, strict=False))


# %% [markdown]
# ## 5. Backtest Engine


# %% [markdown]
# ### Backtest Result Container
#
# Each backtest returns a plain dict with returns, dates, cumulative return,
# turnover, positions, and a `metrics` sub-dict. Annualization assumes 252
# trading days per year.


# %% [markdown]
# ### Walk-Forward Backtest Helpers
#
# Each method receives the same signal schedule and return panel so performance differences
# can be attributed to allocation, not data leakage or execution timing.


# %%
def _compute_signal_weights(
    signals_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    signal_date,
    allocation_fn: Callable,
    window: int,
) -> dict[str, float]:
    """Compute combined long/short weights for one signal date."""
    date_signals = signals_df[signals_df["timestamp"] == signal_date]
    long_stocks = date_signals[date_signals["ml_signal"] > 0]["symbol"].tolist()
    short_stocks = date_signals[date_signals["ml_signal"] < 0]["symbol"].tolist()
    available_returns = returns_df.loc[:signal_date]

    long_weights = allocation_fn(available_returns, long_stocks, window) if long_stocks else {}
    if short_stocks:
        short_weights = allocation_fn(available_returns, short_stocks, window)
        short_weights = {k: -v for k, v in short_weights.items()}
    else:
        short_weights = {}
    return {**long_weights, **short_weights}


# %% [markdown]
# #### Result Packaging Helper
#
# Convert walk-forward positions into the summary metrics used throughout the controlled
# comparison. Turnover comes from dense transitions over the union of current and prior symbols,
# so complete exits remain visible.


# %%
def compute_rebalance_turnover(
    weights: dict[str, float], previous_weights: dict[str, float]
) -> float:
    """Compute half-turnover across the union of current and prior holdings."""
    symbols = set(weights) | set(previous_weights)
    return float(
        0.5
        * sum(
            abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0)) for symbol in symbols
        )
    )


def average_rebalance_turnover(
    turnover_list: list[float], include_initial: bool = TURNOVER_INCLUDE_INITIAL
) -> float:
    """Average per-rebalance half-turnover, optionally excluding the initial allocation."""
    values = np.asarray(turnover_list, dtype=float)
    if not include_initial:
        values = values[1:]
    return float(values.mean()) if values.size else 0.0


def _build_backtest_result(
    dates_list: list[pd.Timestamp],
    portfolio_returns: list[float],
    turnover_list: list[float],
    positions_list: list[dict[str, float]],
) -> dict:
    """Package returns, positions, and metrics into a dict."""
    returns_arr = np.asarray(portfolio_returns, dtype=float)
    metrics = compute_allocator_metrics(
        pl.Series("returns", returns_arr),
        ann_factor=np.sqrt(252),
    )
    metrics["avg_turnover"] = average_rebalance_turnover(turnover_list)
    annual_vol = float(np.std(returns_arr, ddof=1) * np.sqrt(252)) if len(returns_arr) > 1 else 0.0
    return {
        "dates": dates_list,
        "returns": returns_arr,
        "cumulative_return": np.cumprod(1 + returns_arr),
        "turnover": np.asarray(turnover_list, dtype=float),
        "positions": positions_list,
        "metrics": metrics,
        "annual_vol": annual_vol,
    }


# %% [markdown]
# #### Backtest Loop


# %%
def run_backtest(
    returns_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    allocation_fn: Callable,
    allocation_name: str,
    window: int = ALLOCATION_WINDOW,
) -> dict:
    """Run walk-forward backtest for one allocation method."""
    signal_dates = sorted(signals_df["timestamp"].unique())
    portfolio_returns, turnover_list, positions_list, dates_list = [], [], [], []
    prev_weights: dict[str, float] = {}

    terminal_date = returns_df.index.max()
    for i, signal_date in enumerate(signal_dates):
        weights = _compute_signal_weights(
            signals_df, returns_df, signal_date, allocation_fn, window
        )
        next_signal_date = signal_dates[i + 1] if i + 1 < len(signal_dates) else terminal_date
        date_mask = (returns_df.index > signal_date) & (returns_df.index <= next_signal_date)
        period_dates = returns_df.index[date_mask]
        for date in period_dates:
            daily_return = sum(
                weights.get(s, 0) * returns_df.loc[date, s]
                for s in returns_df.columns
                if s in weights and not np.isnan(returns_df.loc[date, s])
            )
            portfolio_returns.append(daily_return)
            dates_list.append(date)
            positions_list.append(weights.copy())

        turnover_list.append(compute_rebalance_turnover(weights, prev_weights))
        prev_weights = weights

    return _build_backtest_result(
        dates_list,
        portfolio_returns,
        turnover_list,
        positions_list,
    )


# %% [markdown]
# ## 6. Run the Allocator Comparison
#
# The selection region ends before 2022. It chooses one allocator exactly once. The later holdout
# remains untouched by that choice; all four holdout rows are descriptive diagnostics, and their
# ordering does not feed another selection decision.


# %%
def slice_signals(
    signals_df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Select signal rows inside an inclusive chronological interval."""
    timestamps = pd.to_datetime(signals_df["timestamp"])
    mask = pd.Series(True, index=signals_df.index)
    if start is not None:
        mask &= timestamps >= pd.Timestamp(start)
    if end is not None:
        mask &= timestamps <= pd.Timestamp(end)
    return signals_df.loc[mask].copy()


# %%
# Define allocation methods
allocation_methods = {
    "Equal Weight": lambda ret, sel, win: equal_weight_allocation(sel, len(sel)),
    "Inverse Volatility": inverse_vol_allocation,
    "MVO (Ledoit-Wolf)": mvo_lw_allocation,
    "HRP": hrp_allocation,
}

# Freeze the chronological boundary before evaluating either region.
selection_signals = slice_signals(ml_signals, end=SELECTION_END)
holdout_signals = slice_signals(ml_signals, start=HOLDOUT_START)
if selection_signals.empty or holdout_signals.empty:
    raise ValueError("Selection and holdout regions must both contain signal dates")

print("Running the pre-holdout selection comparison...")
selection_results = {}
for name, fn in allocation_methods.items():
    print(f"  {name}...")
    selection_results[name] = run_backtest(returns, selection_signals, fn, name)

print("Running the sealed holdout diagnostics...")
results = {}
for name, fn in allocation_methods.items():
    print(f"  {name}...")
    results[name] = run_backtest(returns, holdout_signals, fn, name)

print("Done!")

portfolio_analyses = {
    name: PortfolioAnalysis(returns=result["returns"], dates=result["dates"], periods_per_year=252)
    for name, result in results.items()
}

# %% [markdown]
# ## 7. Performance Comparison


# %%
def annualized_sharpe_se(sharpe: float, n_observations: int, periods_per_year: int = 252) -> float:
    """Approximate IID standard error of an annualized Sharpe ratio."""
    if n_observations <= 0:
        return np.nan
    return float(np.sqrt((periods_per_year + 0.5 * sharpe**2) / n_observations))


def build_comparison_table(period_results: dict[str, dict]) -> pd.DataFrame:
    """Summarize one frozen evaluation region without selecting on it."""
    rows = []
    for name, result in period_results.items():
        metrics = result["metrics"]
        sharpe = float(metrics.get("sharpe", 0.0))
        rows.append(
            {
                "Method": name,
                "Total Return": metrics.get("total_return", 0.0),
                "Annual Return": metrics.get("annual_return", 0.0),
                "Annual Vol": result["annual_vol"],
                "Sharpe Ratio": sharpe,
                "Sharpe SE (IID)": annualized_sharpe_se(sharpe, len(result["returns"])),
                "Max Drawdown": metrics.get("max_drawdown", 0.0),
                "Calmar Ratio": metrics.get("calmar", 0.0),
                "Avg Turnover": metrics.get("avg_turnover", 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("Sharpe Ratio", ascending=False)


selection_comparison_df = build_comparison_table(selection_results)
selected_method = str(selection_comparison_df.iloc[0]["Method"])
comparison_df = build_comparison_table(results)
comparison_df.insert(1, "Selected pre-holdout", comparison_df["Method"] == selected_method)

comparison_df

# %% [markdown]
# **Reading the table**: The marker identifies the allocator chosen on pre-2022 data. The holdout
# ordering is shown to diagnose generalization, not to select a replacement. Avg Turnover is the
# mean per-rebalance half-turnover, $0.5\sum_i|w_{i,t}-w_{i,t-1}|$, over the union of current and
# prior ETFs. It includes the initial allocation from cash; a switches-only convention would omit
# the first observation. These four rows remain gross of explicit commission and slippage.

# %%
# Same numbers, formatted for readability
comparison_df.style.format(
    {
        "Total Return": "{:.1%}",
        "Annual Return": "{:.1%}",
        "Annual Vol": "{:.1%}",
        "Sharpe Ratio": "{:.3f}",
        "Sharpe SE (IID)": "{:.3f}",
        "Max Drawdown": "{:.1%}",
        "Calmar Ratio": "{:.3f}",
        "Avg Turnover": "{:.1%}",
    }
).hide(axis="index")

# %% [markdown]
# **Trading implication**: If a method only marginally improves Sharpe but materially increases
# turnover, the live edge is likely negative after costs.

# %% [markdown]
# ### Practitioner Interpretation

# %%
selected_holdout_row = comparison_df.loc[comparison_df["Method"] == selected_method].iloc[0]
selected_ci_half_width = 1.96 * float(selected_holdout_row["Sharpe SE (IID)"])
display(
    Markdown(
        f"The pre-holdout choice is **{selected_method}**. Its holdout Sharpe is "
        f"**{selected_holdout_row['Sharpe Ratio']:.3f}**, with an approximate IID 95% "
        f"half-width of **{selected_ci_half_width:.3f}**. The interval is a scale check, not a "
        "multiple-comparison adjustment, so drawdown and turnover remain essential diagnostics."
    )
)

# %% [markdown]
# ## 8. Equity Curves

# %%
fig = go.Figure()

palette = ml4t_palette(4, categorical=True)
colors = dict(zip(allocation_methods, palette, strict=True))
holdout_leader = str(comparison_df.iloc[0]["Method"])

for name, result in results.items():
    fig.add_trace(
        go.Scatter(
            x=result["dates"],
            y=result["cumulative_return"],
            mode="lines",
            name=name,
            line=dict(color=colors[name], width=2 if name == selected_method else 1.5),
        )
    )

fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS["neutral"])

fig.update_layout(
    title=f"{holdout_leader} leads the descriptive holdout paths",
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    height=550,
    legend=dict(orientation="h", x=0, y=1.08),
)
fig.show()

# %% [markdown]
# **Finding**: Sustained curve separation, not short-lived spikes, is the evidence that one
# allocator is adding value beyond noise in the shared signal.

# %% [markdown]
# ## 9. Drawdown Analysis


# %%
fig = go.Figure()

for name, result in results.items():
    dd = portfolio_analyses[name].compute_drawdown_analysis()
    fig.add_trace(
        go.Scatter(
            x=result["dates"],
            y=dd.underwater_curve.to_numpy(),
            mode="lines",
            name=name,
            line=dict(color=colors[name], width=2 if name == selected_method else 1.5),
        )
    )

fig.update_layout(
    title="Allocator choice changes the depth and timing of holdout losses",
    xaxis_title="Date",
    yaxis_title="Drawdown",
    height=400,
    yaxis_tickformat=".0%",
)
fig.show()

# %% [markdown]
# **Trading implication**: Lower and shallower drawdowns can dominate small Sharpe differences
# when capital constraints are driven by investor risk tolerance.

# %% [markdown]
# ## 10. Rolling Sharpe Ratio


# %%
fig = go.Figure()

for name, result in results.items():
    rolling = portfolio_analyses[name].compute_rolling_metrics(
        windows=[252],
        metrics=["sharpe"],
    )
    rs_raw = np.asarray(rolling.sharpe[252].to_list(), dtype=float)
    rs = np.concatenate(([np.nan], rs_raw[:-1])) if rs_raw.size else rs_raw
    fig.add_trace(
        go.Scatter(
            x=result["dates"],
            y=rs,
            mode="lines",
            name=name,
            line=dict(color=colors[name], width=2 if name == selected_method else 1.5),
        )
    )

fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
fig.add_hline(
    y=1,
    line_dash="dot",
    line_color=COLORS["positive"],
    annotation_text="SR=1",
    annotation_position="bottom right",
)

fig.update_layout(
    title="Rolling Sharpe reveals unstable allocator rankings",
    xaxis_title="Date",
    yaxis_title="Sharpe Ratio",
    height=400,
)
fig.show()

# %% [markdown]
# **Finding**: Rolling Sharpe is a persistence diagnostic. It does not by itself define a regime
# classifier or authorize switching allocators after observing the holdout.

# %% [markdown]
# ### Predeclared Volatility-Regime Diagnostic
#
# The regime threshold is the median trailing 63-day SPY volatility estimated only through the
# selection region. Applying that frozen threshold to holdout dates describes performance across
# lower- and higher-volatility observations without fitting a switching rule on the holdout.


# %%
spy_trailing_vol = returns["SPY"].rolling(63).std() * np.sqrt(252)
regime_threshold = float(spy_trailing_vol.loc[:SELECTION_END].dropna().median())
regime_rows = []
for name, result in results.items():
    result_returns = pd.Series(result["returns"], index=pd.to_datetime(result["dates"]))
    aligned_vol = spy_trailing_vol.reindex(result_returns.index)
    for regime, mask in {
        "Lower volatility": aligned_vol <= regime_threshold,
        "Higher volatility": aligned_vol > regime_threshold,
    }.items():
        sliced = result_returns.loc[mask.fillna(False)]
        sliced_sharpe = (
            float(sliced.mean() / sliced.std(ddof=1) * np.sqrt(252))
            if len(sliced) > 1 and sliced.std(ddof=1) > 0
            else np.nan
        )
        regime_rows.append(
            {
                "Method": name,
                "Regime": regime,
                "Observations": len(sliced),
                "Annualized Sharpe": sliced_sharpe,
            }
        )

regime_comparison = pl.DataFrame(regime_rows).sort(["Regime", "Annualized Sharpe"], descending=True)
display(
    Markdown(
        f"The frozen annualized-volatility threshold is **{regime_threshold:.1%}**. "
        "These slices are descriptive; no allocator is reselected from them."
    )
)
regime_comparison

# %% [markdown]
# For $T$ daily observations and $P=252$ periods per year, the standard error of an annualized
# Sharpe ratio under the i.i.d. approximation is:
#
# $$SE(\hat{SR}_{ann}) \approx \sqrt{\frac{P + \frac{\hat{SR}_{ann}^2}{2}}{T}}.$$
#
# Serial dependence and allocator shopping require stronger inference than this IID scale check.

# %% [markdown]
# ## 11. Relative Performance

# %%
# Compute relative performance vs Equal Weight baseline
baseline_cum = results["Equal Weight"]["cumulative_return"]

fig = go.Figure()

for name, result in results.items():
    if name == "Equal Weight":
        continue

    # Align lengths
    min_len = min(len(result["cumulative_return"]), len(baseline_cum))
    relative = result["cumulative_return"][:min_len] / baseline_cum[:min_len]

    fig.add_trace(
        go.Scatter(
            x=result["dates"][:min_len],
            y=relative,
            mode="lines",
            name=f"{name} / EW",
            line=dict(color=colors[name], width=2),
        )
    )

fig.add_hline(
    y=1.0,
    line_dash="dash",
    line_color=COLORS["neutral"],
    annotation_text="Equal-weight parity",
    annotation_position="bottom right",
)

fig.update_layout(
    title="Optimized holdout paths drift against equal weight",
    xaxis_title="Date",
    yaxis_title="Relative Return",
    height=400,
)
fig.show()

# %% [markdown]
# **Trading implication**: Relative-performance drift below parity suggests keeping equal-weight
# as the default fallback when optimization confidence degrades.

# %% [markdown]
# ## 12. Reading the Comparison Table
#
# The pre-holdout region makes the only selection decision. The holdout table then answers whether
# that frozen choice generalizes. Its descriptive leader is reported for diagnosis but is never fed
# back into the tear sheet or engine.

# %%
descriptive_leader = comparison_df.iloc[0]
selected_row = comparison_df.loc[comparison_df["Method"] == selected_method].iloc[0]
equal_row = comparison_df.loc[comparison_df["Method"] == "Equal Weight"].iloc[0]
selected_gap_vs_ew = float(selected_row["Sharpe Ratio"] - equal_row["Sharpe Ratio"])

print("\n" + "=" * 60)
print("ALLOCATOR COMPARISON: FROZEN SELECTION READING")
print("=" * 60)
print(f"\n  Selected before holdout: {selected_method}")
print(f"  Holdout Sharpe:          {selected_row['Sharpe Ratio']:.3f}")
print(f"  IID Sharpe SE:           {selected_row['Sharpe SE (IID)']:.3f}")
print(f"  Gap vs equal weight:     {selected_gap_vs_ew:+.3f}")
print(f"  Descriptive leader:      {descriptive_leader['Method']}")
print("  The descriptive leader is not reselected.")
print("\n" + "=" * 60)

# %% [markdown]
# ## Conclusion
#
# Under identical ML signals, the sealed holdout shows whether a preselected allocator generalizes.
# Gross return, drawdown, turnover, and uncertainty must be read together.

# %% [markdown]
# ## Summary
#
# The controlled comparison separates selection from evaluation. The four-way holdout table is
# gross of explicit costs; only the frozen pre-holdout choice enters the execution-aware replay.

# %% [markdown]
# ## Key Takeaways
#
# ### Allocator Comparison Results:
#
# 1. **Same signals, different outcomes**: The allocation method significantly impacts
#    final performance, even with identical trading signals.
#
# 2. **Complexity tradeoffs**: HRP and MVO-LW estimate covariance structure; the
#    estimation error competes with the signal advantage over equal-weight and inverse
#    volatility, which is why the ordering across allocators depends on the universe
#    and sample.
#
# 3. **Turnover matters**: Higher-turnover methods incur more transaction costs in live trading.
#
# 4. **Regime diagnostics are not a switching rule**: The frozen volatility threshold reveals
#    conditional performance without reselecting an allocator on the holdout.
#
# ### Practical Recommendations:
#
# - **Start simple**: Equal weight or inverse volatility are strong baselines
# - **Test multiple methods**: No single allocation dominates all market conditions
# - **Consider turnover**: Account for transaction costs when comparing methods
# - **Diagnose regimes**: Predeclare a rule before using regimes to change allocations
#
# ### What's Next (Chapter 19):
#
# Historical backtests measure realized risk and return but do not characterize forward
# risk under regime change. Chapter 19 covers **forward-looking risk analysis**: VaR,
# CVaR, stress testing, and factor attribution.

# %%
# Final descriptive holdout table, sorted for readability but not used for selection
final_summary = (
    comparison_df[
        ["Method", "Selected pre-holdout", "Sharpe Ratio", "Annual Return", "Max Drawdown"]
    ]
    .reset_index(drop=True)
    .rename_axis("Rank")
)
final_summary.index = final_summary.index + 1
final_summary.style.format(
    {"Sharpe Ratio": "{:.3f}", "Annual Return": "{:.1%}", "Max Drawdown": "{:.1%}"}
)

# %% [markdown]
# ## 13. Holdout Tear Sheet for the Preselected Allocator - `ml4t-diagnostic`
#
# `ml4t-diagnostic` packages the metrics + plots above into a reusable tear
# sheet. We build it for the allocator selected on data through 2021, with SPY
# as the benchmark so the alpha, beta, and information-ratio readouts have meaning.
#
# The same `PortfolioTearSheet` object supports two delivery modes:
#
# - **Inline** - `tear_sheet.show()` renders the metrics block plus each Plotly
#   figure as standard cell outputs, useful for interactive analysis.
# - **HTML** - `tear_sheet.save_html(path)` writes a self-contained file for
#   sharing or archival; the same content, packaged for distribution.

# %%
selected_result = results[selected_method]

# Align SPY benchmark to the selected allocator's holdout date index
spy_aligned = (
    pd.Series(returns["SPY"].values, index=returns.index)
    .reindex(pd.to_datetime(selected_result["dates"]))
    .dropna()
)
common_dates = pd.Index(selected_result["dates"]).intersection(spy_aligned.index)
selected_returns_aligned = pd.Series(
    selected_result["returns"], index=selected_result["dates"]
).loc[common_dates]
spy_aligned = spy_aligned.loc[common_dates]

selected_analysis = PortfolioAnalysis(
    returns=selected_returns_aligned.to_numpy(),
    benchmark=spy_aligned.to_numpy(),
    dates=common_dates,
    risk_free=0.0,
    periods_per_year=252,
)


def style_diagnostic_figures(figures: dict[str, go.Figure]) -> None:
    """Apply ML4T styling and readable legends to dashboard figures in place."""
    diverging = ml4t_diverging()
    legend_figures = {
        "Cumulative Returns",
        "Rolling Sharpe Ratio",
        "Rolling Volatility",
        "Annual Returns",
    }
    for figure_name, diagnostic_figure in figures.items():
        show_legend = figure_name in legend_figures
        bottom_margin = 90 if figure_name == "Rolling Volatility" else 65
        diagnostic_figure.update_layout(
            template="ml4t",
            paper_bgcolor=COLORS["bg_light"],
            plot_bgcolor=COLORS["bg_light"],
            font=dict(color=COLORS["neutral"]),
            margin=dict(l=70, r=180 if show_legend else 90, t=80, b=bottom_margin),
            showlegend=show_legend,
            legend=dict(x=1.01, xanchor="left", y=1.0, yanchor="top"),
        )
        if figure_name == "Rolling Volatility":
            diagnostic_figure.update_xaxes(title_text="Date", automargin=True)
        for trace in diagnostic_figure.data:
            if show_legend:
                trace.update(showlegend=True)
            if trace.type == "heatmap":
                trace.update(
                    colorscale=[
                        [0.0, diverging[0]],
                        [0.5, diverging[1]],
                        [1.0, diverging[2]],
                    ],
                    zmid=0,
                )


selected_tear_sheet = create_portfolio_dashboard(selected_analysis)
style_diagnostic_figures(selected_tear_sheet.figures)

print(f"Holdout tear sheet generated for preselected allocator: {selected_method}")

# Inline display: metrics summary + each constituent figure as a separate cell.
selected_tear_sheet.show()

# %%
# HTML delivery: same content packaged as a self-contained file.
output_dir = get_output_dir(17, "allocator_comparison")
output_dir.mkdir(parents=True, exist_ok=True)
selected_html_path = (
    output_dir
    / f"allocator_comparison_{selected_method.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}_tear_sheet.html"
)
selected_tear_sheet.save_html(selected_html_path, include_plotlyjs="cdn")
print(f"Holdout tear sheet saved under the ML4T output directory: {selected_html_path.name}")
print(f"  Figures embedded: {list(selected_tear_sheet.figures.keys())}")

# %% [markdown]
# ## 14. Execution-Aware Holdout Replay with ml4t-backtest
#
# The gross weight-based comparison isolates allocation effects. This replay sends the allocator
# chosen before the holdout through actual canonical OHLCV bars with NEXT_BAR execution, commission,
# and slippage. It does not use the descriptive holdout ranking to change the selected method.
#
# The replay is an execution diagnostic rather than a claim that all four gross rankings survive
# costs.

# %% [markdown]
# ### Strategy Wrapper for ml4t-backtest
#
# This class adapts the preselected allocator to the engine API.


# %%
def compute_target_weights(
    date_signals: pd.DataFrame,
    available_returns: pd.DataFrame,
    allocation_fn: Callable,
    allocation_name: str,
    allocation_window: int,
) -> dict[str, float]:
    """Build long/short target weights for one rebalance date."""
    target_weights: dict[str, float] = {}
    long_stocks = date_signals[date_signals["ml_signal"] > 0]["symbol"].tolist()
    short_stocks = date_signals[date_signals["ml_signal"] < 0]["symbol"].tolist()

    if long_stocks:
        if allocation_name == "Equal Weight":
            long_weights = equal_weight_allocation(long_stocks, len(long_stocks))
        else:
            long_weights = allocation_fn(available_returns, long_stocks, allocation_window)
        target_weights.update(long_weights)

    if short_stocks:
        if allocation_name == "Equal Weight":
            short_weights = equal_weight_allocation(short_stocks, len(short_stocks))
        else:
            short_weights = allocation_fn(available_returns, short_stocks, allocation_window)
        target_weights.update({k: -v for k, v in short_weights.items()})
    return target_weights


# %% [markdown]
# #### Engine Strategy Class


# %%
class AllocatorComparisonStrategy(Strategy):
    """Apply comparison allocations through ml4t-backtest execution."""

    def __init__(
        self,
        assets,
        signals_df,
        allocation_fn,
        allocation_name,
        returns_df,
        allocation_window,
    ):
        self.signals_df = signals_df
        self.allocation_fn = allocation_fn
        self.allocation_name = allocation_name
        self.returns_df = returns_df
        self.allocation_window = allocation_window
        self.signal_dates = set(signals_df["timestamp"].unique())
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(
                min_trade_value=100.0,
                min_weight_change=0.01,
                allow_fractional=False,
                allow_short=True,
            )
        )

    def on_data(self, timestamp, data, context, broker):
        current_date = timestamp.date() if hasattr(timestamp, "timestamp") else timestamp
        if current_date not in self.signal_dates:
            return
        date_signals = self.signals_df[self.signals_df["timestamp"] == current_date]
        if date_signals.empty:
            return
        target_weights = compute_target_weights(
            date_signals,
            self.returns_df.loc[:current_date],
            self.allocation_fn,
            self.allocation_name,
            self.allocation_window,
        )
        if not target_weights:
            return
        orders = self.executor.execute(target_weights, data, broker)
        if orders:
            print(f"[{current_date}] {self.allocation_name}: {len(orders)} orders")


# %%
# Prepare canonical OHLCV data for ml4t-backtest
print("\n" + "=" * 70)
print("ML4T-BACKTEST HOLDOUT EXECUTION DIAGNOSTIC")
print("=" * 70)

first_engine_signal = pd.Timestamp(holdout_signals["timestamp"].min()).date()
prices_long = (
    etf_bars.filter(
        (pl.col("symbol").is_in(loaded_symbols))
        & (pl.col("timestamp") >= pl.lit(first_engine_signal))
    )
    .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    .sort(["timestamp", "symbol"])
)
vector_terminal_date = pd.Timestamp(returns.index.max()).date()
engine_terminal_date = pd.Timestamp(prices_long["timestamp"].max()).date()
if engine_terminal_date != vector_terminal_date:
    raise ValueError(
        f"Matched-horizon check failed: vector={vector_terminal_date}, engine={engine_terminal_date}"
    )
if sorted(prices_long["symbol"].unique().to_list()) != sorted(loaded_symbols):
    raise ValueError("Matched-universe check failed between vector and engine paths")
print(
    "Matched horizon and universe: "
    f"final_signal={pd.Timestamp(holdout_signals['timestamp'].max()).date()}, "
    f"terminal={vector_terminal_date}, assets={len(loaded_symbols)}"
)

bar_integrity = prices_long.select(
    (pl.col("open") == pl.col("close")).all().alias("all_open_equals_close"),
    (pl.col("high") == pl.col("close")).all().alias("all_high_equals_close"),
    (pl.col("low") == pl.col("close")).all().alias("all_low_equals_close"),
    pl.col("volume").n_unique().alias("unique_volume_values"),
).row(0, named=True)
if (
    bar_integrity["all_open_equals_close"]
    or bar_integrity["all_high_equals_close"]
    or bar_integrity["all_low_equals_close"]
    or bar_integrity["unique_volume_values"] <= 1
):
    raise ValueError(f"Engine feed failed canonical OHLCV integrity checks: {bar_integrity}")
print(f"Canonical OHLCV integrity: {bar_integrity}")

# %%
# Create DataFeed
feed = DataFeed(prices_df=prices_long)

# Run the execution-aware backtest with the allocator frozen before the holdout
selected_fn = allocation_methods[selected_method]

print(f"\nRunning preselected {selected_method} through ml4t-backtest...")

# %%
# Convert signal dates to proper format
signals_with_dates = holdout_signals.copy()
signals_with_dates["timestamp"] = signals_with_dates["timestamp"].apply(
    lambda x: pd.Timestamp(x).date()
)

strategy = AllocatorComparisonStrategy(
    assets=loaded_symbols,
    signals_df=signals_with_dates,
    allocation_fn=selected_fn,
    allocation_name=selected_method,
    returns_df=returns,
    allocation_window=ALLOCATION_WINDOW,
)

engine = Engine(
    feed=feed,
    strategy=strategy,
    config=BacktestConfig(
        initial_cash=100_000,
        execution_mode=ExecutionMode.NEXT_BAR,
        commission_type=CommissionType.PERCENTAGE,
        commission_rate=0.001,
        slippage_type=SlippageType.PERCENTAGE,
        slippage_rate=0.0005,
        allow_short_selling=True,
    ),
)

ml4t_results = engine.run()

# %%
print("\n" + "=" * 70)
print(f"ML4T-BACKTEST HOLDOUT RESULTS ({selected_method})")
print("=" * 70)
print(f"Final Value:     ${ml4t_results['final_value']:,.2f}")
print(f"Total Return:    {ml4t_results['total_return_pct']:.2f}%")
print(f"Sharpe Ratio:    {ml4t_results['sharpe']:.3f}")
print(f"Max Drawdown:    {ml4t_results['max_drawdown_pct']:.2f}%")
print(f"Order Fills:     {len(ml4t_results['fills'])}")
print(f"Round-Trips:     {ml4t_results['num_trades']}")
print(f"Total Commission: ${ml4t_results['total_commission']:,.2f}")
print(f"Total Slippage:   ${ml4t_results['total_slippage']:,.2f}")

# Compare with weight-based simulation
weight_result = results[selected_method]
print("\nComparison with Weight-Based Simulation:")
weight_total_return = float(weight_result["metrics"].get("total_return", 0.0))
print(f"  Weight-Based Return: {weight_total_return * 100:.2f}%")
print(f"  ml4t-backtest Return: {ml4t_results['total_return_pct']:.2f}%")
print(f"  Difference: {ml4t_results['total_return_pct'] - weight_total_return * 100:.2f}%")
print(
    "\nThe execution replay reports the combined effect of its configured costs and fill schedule;"
)
print("this notebook does not attribute the return difference to individual mechanisms.")

# %% [markdown]
# **Finding**: The execution replay is a diagnostic for the preselected allocator. Because the
# engine applies its configured costs and fill schedule jointly, this notebook does not claim an
# independent commission, slippage, or fill-timing attribution, nor does it rank all allocators
# under execution costs.

# %% [markdown]
# ### When to Use Each Approach
#
# | Approach | Use Case |
# |----------|----------|
# | **Weight-Based Simulation** | Method selection, quick comparison, isolating allocation effects |
# | **ml4t-backtest** | Execution cost modeling, fill diagnostics, strategy validation |
#
# For the controlled comparison, weight-based simulation is appropriate because:
# - All methods use identical execution assumptions
# - We're isolating the allocation method's impact
# - It's faster for comparing multiple methods
#
# For execution-aware validation, use ml4t-backtest because:
# - Bar-based execution modeling on actual OHLCV
# - Commission and slippage impact
# - Position tracking and risk management
# - A path toward live-system integration

# %% [markdown]
# ## Key Takeaways
#
# Simple allocators remain competitive because they avoid the estimation error
# that destabilizes more expressive optimizers. The execution-aware bridge reports
# the preselected allocator under one combined cost and fill schedule; it does not
# isolate individual mechanisms.
#
# **Next**: Ch20's [`05_portfolio_allocation`](../20_strategy_synthesis/05_portfolio_allocation.ipynb)
# tests whether the same ranking holds across the full case-study set.
