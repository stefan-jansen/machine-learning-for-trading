"""Factor attribution for case study strategy analysis.

Runs Fama-French + Momentum regressions on strategy daily returns,
computes rolling exposures, placebo benchmarks, and bootstrap CIs.

Usage::

    from case_studies.utils.factor_attribution import (
        load_factor_data,
        run_factor_regression,
        compute_rolling_exposures,
        run_placebo_benchmark,
        compute_bootstrap_ci,
        format_attribution_summary,
        plot_rolling_exposures,
        plot_attribution_waterfall,
    )
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from data.factors.loader import load_ff_factors

# ---------------------------------------------------------------------------
# Factor data loading
# ---------------------------------------------------------------------------


def load_factor_data(
    start: str | None = None,
    end: str | None = None,
    model: Literal["ff5_mom", "ff3", "ff5"] = "ff5_mom",
) -> pd.DataFrame:
    """Load and merge Fama-French factors into a single daily DataFrame.

    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        model: Factor model specification

    Returns:
        pandas DataFrame indexed by date with factor columns + RF
    """
    if model in ("ff5", "ff5_mom"):
        ff = load_ff_factors(dataset="ff5", frequency="daily", start_date=start, end_date=end)
    else:
        ff = load_ff_factors(dataset="ff3", frequency="daily", start_date=start, end_date=end)

    # Normalize timestamp to date (join in polars, convert to pandas at boundary)
    ff = ff.with_columns(pl.col("timestamp").cast(pl.Date).alias("date")).drop("timestamp")

    if model == "ff5_mom":
        mom = load_ff_factors(dataset="mom", frequency="daily", start_date=start, end_date=end)
        mom = mom.with_columns(pl.col("timestamp").cast(pl.Date).alias("date")).drop("timestamp")
        ff = ff.join(mom, on="date", how="inner")

    # Convert to pandas at boundary (downstream OLS requires pandas)
    ff_pd = ff.to_pandas().set_index("date")
    ff_pd.index = pd.to_datetime(ff_pd.index)
    return ff_pd


def _factor_columns(model: str) -> list[str]:
    """Return the factor column names for a given model specification."""
    if model == "ff5_mom":
        return ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    elif model == "ff5":
        return ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    else:  # ff3
        return ["Mkt-RF", "SMB", "HML"]


# ---------------------------------------------------------------------------
# Core regression
# ---------------------------------------------------------------------------


def _detect_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer annualization factor from return series frequency."""
    if len(index) < 2:
        return 252
    diffs = pd.Series(index).diff().dropna().dt.days
    median_gap = float(diffs.median())
    if median_gap <= 2:
        return 252  # daily (1-2 day gaps = business days)
    elif median_gap <= 8:
        return 52  # weekly
    elif median_gap <= 18:
        return 26  # biweekly
    elif median_gap <= 45:
        return 12  # monthly (28-33 day gaps)
    elif median_gap <= 100:
        return 4  # quarterly
    return 1  # annual


def _aggregate_factors_to_frequency(
    factors: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Aggregate daily factor returns to match a lower-frequency return series.

    For each target date, sums daily factor returns from the previous target
    date (exclusive) to the current date (inclusive). This produces
    period-matched factor returns suitable for regression against periodic
    strategy returns (e.g., monthly strategy returns vs monthly factor returns).
    """
    factor_cols = [c for c in factors.columns if c != "RF"]
    target_sorted = sorted(target_dates)
    rows = []
    for i, end_date in enumerate(target_sorted):
        start_date = target_sorted[i - 1] if i > 0 else factors.index[0] - pd.Timedelta(days=1)
        mask = (factors.index > start_date) & (factors.index <= end_date)
        window = factors.loc[mask]
        if len(window) == 0:
            continue
        row = {"date": end_date}
        for col in factor_cols:
            # Compound factor returns over the period
            row[col] = float((1 + window[col]).prod() - 1)
        # RF: sum of daily rates
        row["RF"] = float(window["RF"].sum())
        rows.append(row)
    if not rows:
        cols = [c for c in factor_cols if c in factors.columns] + ["RF"]
        return pd.DataFrame(columns=cols).rename_axis("date")
    return pd.DataFrame(rows).set_index("date")


def run_factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: Literal["ff5_mom", "ff3", "ff5"] = "ff5_mom",
    hac_lags: int = 5,
    dollar_neutral: bool = True,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Run factor regression with HAC (Newey-West) standard errors.

    Automatically detects return frequency and aggregates daily factor
    returns to match. For daily strategies, factors are used as-is. For
    weekly/monthly strategies, daily factors are compounded to the matching
    period.

    For dollar-neutral strategies, uses raw returns as LHS (not excess).
    For long-only strategies, uses excess returns (return - RF).

    Args:
        returns: Strategy returns (indexed by date, any frequency)
        factors: Daily factor DataFrame from load_factor_data()
        model: Factor specification
        hac_lags: Newey-West bandwidth
        dollar_neutral: If True, use raw returns (standard for zero-investment)
        periods_per_year: Annualization factor (auto-detected if None)

    Returns:
        Dict with alpha, betas, t-stats, R², residual Sharpe, etc.
    """
    factor_cols = _factor_columns(model)
    available_cols = [c for c in factor_cols if c in factors.columns]

    # Detect frequency
    ppy = periods_per_year or _detect_periods_per_year(returns.index)

    # Aggregate factors if strategy is lower than daily frequency
    if ppy < 200:  # Not daily — need to aggregate
        f_agg = _aggregate_factors_to_frequency(factors, returns.index)
        common = returns.index.intersection(f_agg.index)
        if len(common) < 10:
            raise ValueError(f"Only {len(common)} overlapping periods — need at least 10")
        y = returns.loc[common]
        f = f_agg.loc[common]
    else:
        common = returns.index.intersection(factors.index)
        if len(common) < 30:
            raise ValueError(f"Only {len(common)} overlapping dates — need at least 30")
        y = returns.loc[common]
        f = factors.loc[common]

    # LHS: raw returns for dollar-neutral, excess for long-only
    if not dollar_neutral:
        y = y - f["RF"]

    X = sm.add_constant(f[available_cols])

    # OLS with Newey-West HAC standard errors
    ols = sm.OLS(y.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    # Extract results
    col_names = ["const"] + available_cols
    params = dict(zip(col_names, ols.params, strict=False))
    tvalues = dict(zip(col_names, ols.tvalues, strict=False))
    pvalues = dict(zip(col_names, ols.pvalues, strict=False))

    # Annualize using correct frequency
    alpha_per_period = params["const"]
    alpha_annualized = alpha_per_period * ppy

    # Residual Sharpe = alpha / residual_vol (annualized)
    resid = ols.resid
    resid_vol_period = float(resid.std())
    resid_sharpe = (
        float(alpha_per_period / resid_vol_period * np.sqrt(ppy)) if resid_vol_period > 0 else 0.0
    )

    # Strategy Sharpe for comparison
    strategy_sharpe = float(y.mean() / y.std() * np.sqrt(ppy)) if y.std() > 0 else 0.0

    return {
        "model": model,
        "n_obs": len(common),
        "periods_per_year": ppy,
        "alpha_per_period": alpha_per_period,
        "alpha_annualized": alpha_annualized,
        "alpha_t_stat": tvalues["const"],
        "alpha_p_value": pvalues["const"],
        "alpha_significant": pvalues["const"] < 0.05,
        "betas": {k: params[k] for k in available_cols},
        "t_stats": {k: tvalues[k] for k in available_cols},
        "p_values": {k: pvalues[k] for k in available_cols},
        "r_squared": ols.rsquared,
        "adj_r_squared": ols.rsquared_adj,
        "residual_sharpe": resid_sharpe,
        "strategy_sharpe": strategy_sharpe,
        "residual_vol_annual": float(resid_vol_period * np.sqrt(ppy)),
        "dollar_neutral": dollar_neutral,
        "hac_lags": hac_lags,
        "factor_columns": available_cols,
    }


# ---------------------------------------------------------------------------
# Rolling exposures
# ---------------------------------------------------------------------------


def compute_rolling_exposures(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: Literal["ff5_mom", "ff3", "ff5"] = "ff5_mom",
    window: int | None = None,
    dollar_neutral: bool = True,
    periods_per_year: int | None = None,
) -> pd.DataFrame:
    """Compute rolling factor betas over a sliding window.

    Args:
        returns: Strategy returns (any frequency)
        factors: Daily factor DataFrame (aggregated internally if needed)
        model: Factor specification
        window: Rolling window in periods (default: auto — 63 for daily,
            12 for monthly, 26 for weekly)
        dollar_neutral: If True, use raw returns as LHS
        periods_per_year: Annualization factor (auto-detected if None)

    Returns:
        DataFrame with rolling betas indexed by date
    """
    factor_cols = _factor_columns(model)
    available_cols = [c for c in factor_cols if c in factors.columns]
    ppy = periods_per_year or _detect_periods_per_year(returns.index)

    # Aggregate factors if needed
    if ppy < 200:
        f_matched = _aggregate_factors_to_frequency(factors, returns.index)
        common = returns.index.intersection(f_matched.index)
    else:
        f_matched = factors
        common = returns.index.intersection(factors.index)

    y_all = returns.loc[common]
    f_all = f_matched.loc[common]

    if not dollar_neutral:
        y_all = y_all - f_all["RF"]

    # Default window: ~1 year of observations
    if window is None:
        window = min(max(ppy, 12), len(common) // 3)

    rows = []
    for i in range(window, len(common)):
        y_win = y_all.iloc[i - window : i].values
        X_win = sm.add_constant(f_all[available_cols].iloc[i - window : i].values)
        try:
            result = sm.OLS(y_win, X_win).fit()
            row = {"date": common[i], "alpha_ann": result.params[0] * ppy}
            for j, col in enumerate(available_cols):
                row[col] = result.params[j + 1]
            rows.append(row)
        except (np.linalg.LinAlgError, ValueError) as exc:
            warnings.warn(
                f"Rolling exposure OLS failed at window ending {common[i]}: {exc}",
                stacklevel=2,
            )
            continue

    if not rows:
        return pd.DataFrame(columns=["alpha_ann"] + available_cols).rename_axis("date")
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Placebo benchmark
# ---------------------------------------------------------------------------


def run_placebo_benchmark(
    daily_returns_wide: pd.DataFrame,
    factors: pd.DataFrame,
    n_sims: int = 500,
    top_k: int = 20,
    model: Literal["ff5_mom", "ff3", "ff5"] = "ff5_mom",
    dollar_neutral: bool = True,
    seed: int = 42,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Generate random portfolios from the same universe for placebo comparison.

    Constructs n_sims random portfolios and runs factor regressions on each.
    Returns the distribution of factor loadings to determine how much of the
    strategy's exposure is explained by the universe composition.

    When dollar_neutral=True (default), constructs long-short portfolios
    (long top_k, short top_k). When False, constructs long-only portfolios
    (random top_k equal-weight) — appropriate for long-only strategies.

    Args:
        daily_returns_wide: DataFrame with columns = symbols, index = dates,
            values = daily returns
        factors: Factor DataFrame
        n_sims: Number of random portfolios
        top_k: Number of stocks per leg (long-short) or total (long-only)
        model: Factor specification
        dollar_neutral: If True, long-short placebos; if False, long-only
        seed: Random seed
        periods_per_year: Annualization factor (auto-detected if None)

    Returns:
        Dict with distributions of betas, alphas, and R² across placebos
    """
    rng = np.random.default_rng(seed)
    factor_cols = _factor_columns(model)
    available_cols = [c for c in factor_cols if c in factors.columns]

    # Align
    common_dates = daily_returns_wide.index.intersection(factors.index)
    rets = daily_returns_wide.loc[common_dates].dropna(axis=1, how="all")
    f = factors.loc[common_dates]
    ppy = periods_per_year or _detect_periods_per_year(rets.index)
    symbols = rets.columns.tolist()
    n_symbols = len(symbols)

    n_select = 2 * top_k if dollar_neutral else top_k
    if n_symbols < n_select:
        top_k = max(1, n_symbols // 4)
        n_select = 2 * top_k if dollar_neutral else top_k

    placebo_results = []
    for _ in range(n_sims):
        selected = rng.choice(n_symbols, size=n_select, replace=False)

        if dollar_neutral:
            # Long-short: long top_k, short top_k
            long_ret = rets.iloc[:, selected[:top_k]].mean(axis=1)
            short_ret = rets.iloc[:, selected[top_k:]].mean(axis=1)
            port_ret = long_ret - short_ret
        else:
            # Long-only: equal-weight top_k
            port_ret = rets.iloc[:, selected].mean(axis=1)

        # Quick regression (no HAC for speed)
        y = port_ret.values
        X = sm.add_constant(f[available_cols].values)
        try:
            result = sm.OLS(y, X).fit()
            row = {"alpha_ann": result.params[0] * ppy, "r_squared": result.rsquared}
            for j, col in enumerate(available_cols):
                row[col] = result.params[j + 1]
            placebo_results.append(row)
        except (np.linalg.LinAlgError, ValueError) as exc:
            warnings.warn(f"Placebo sim {len(placebo_results)} OLS failed: {exc}", stacklevel=2)
            continue

    if not placebo_results:
        return {"n_sims": 0}

    pdf = pd.DataFrame(placebo_results)
    summary: dict[str, Any] = {"n_sims": len(pdf)}
    for col in available_cols:
        summary[f"{col}_mean"] = float(pdf[col].mean())
        summary[f"{col}_std"] = float(pdf[col].std())
        summary[f"{col}_p5"] = float(pdf[col].quantile(0.05))
        summary[f"{col}_p95"] = float(pdf[col].quantile(0.95))
    summary["alpha_ann_mean"] = float(pdf["alpha_ann"].mean())
    summary["alpha_ann_std"] = float(pdf["alpha_ann"].std())
    summary["r_squared_mean"] = float(pdf["r_squared"].mean())
    summary["_raw"] = pdf  # Keep raw for plotting

    return summary


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------


def compute_bootstrap_ci(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: Literal["ff5_mom", "ff3", "ff5"] = "ff5_mom",
    n_boot: int = 1000,
    block_size: int | None = None,
    dollar_neutral: bool = True,
    confidence: float = 0.95,
    seed: int = 42,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Block bootstrap confidence intervals for alpha and betas.

    Uses moving-block bootstrap with the specified block size to preserve
    serial dependence in residuals. Automatically handles non-daily
    return frequencies.

    Args:
        returns: Strategy returns (any frequency)
        factors: Daily factor DataFrame (aggregated internally if needed)
        model: Factor specification
        n_boot: Number of bootstrap replications
        block_size: Block size in periods (default: auto — 20 for daily,
            3 for monthly, 8 for weekly)
        dollar_neutral: If True, raw returns as LHS
        confidence: Confidence level (default 0.95)
        seed: Random seed
        periods_per_year: Annualization factor (auto-detected if None)

    Returns:
        Dict with point estimates and CI bounds for alpha and betas
    """
    rng = np.random.default_rng(seed)
    factor_cols = _factor_columns(model)
    available_cols = [c for c in factor_cols if c in factors.columns]
    ppy = periods_per_year or _detect_periods_per_year(returns.index)

    # Aggregate factors if needed
    if ppy < 200:
        f_matched = _aggregate_factors_to_frequency(factors, returns.index)
        common = returns.index.intersection(f_matched.index)
    else:
        f_matched = factors
        common = returns.index.intersection(factors.index)

    y = returns.loc[common]
    f = f_matched.loc[common]

    if not dollar_neutral:
        y = y - f["RF"]

    y_arr = y.values
    X_arr = sm.add_constant(f[available_cols].values)
    T = len(y_arr)

    # Default block size: ~1 month of observations
    if block_size is None:
        block_size = max(2, min(ppy // 12, T // 4))

    if block_size >= T:
        return {"n_boot": 0}

    n_blocks = int(np.ceil(T / block_size))

    boot_params = []
    for _ in range(n_boot):
        block_starts = rng.integers(0, T - block_size + 1, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in block_starts])[:T]

        y_boot = y_arr[indices]
        X_boot = X_arr[indices]

        try:
            result = sm.OLS(y_boot, X_boot).fit()
            boot_params.append(result.params)
        except (np.linalg.LinAlgError, ValueError) as exc:
            warnings.warn(f"Bootstrap OLS replication failed: {exc}", stacklevel=2)
            continue

    if not boot_params:
        return {"n_boot": 0}

    params_arr = np.array(boot_params)
    col_names = ["alpha"] + available_cols
    alpha_level = (1 - confidence) / 2

    ci: dict[str, Any] = {"n_boot": len(params_arr), "confidence": confidence}
    for j, name in enumerate(col_names):
        vals = params_arr[:, j]
        if name == "alpha":
            vals_display = vals * ppy  # Annualize with correct frequency
            ci[f"{name}_ann_mean"] = float(vals_display.mean())
            ci[f"{name}_ann_lo"] = float(np.quantile(vals_display, alpha_level))
            ci[f"{name}_ann_hi"] = float(np.quantile(vals_display, 1 - alpha_level))
        else:
            ci[f"{name}_mean"] = float(vals.mean())
            ci[f"{name}_lo"] = float(np.quantile(vals, alpha_level))
            ci[f"{name}_hi"] = float(np.quantile(vals, 1 - alpha_level))

    return ci


# ---------------------------------------------------------------------------
# Assessment integration
# ---------------------------------------------------------------------------


def format_attribution_summary(
    regression: dict[str, Any],
    bootstrap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Format factor attribution results for strategy_assessment.json.

    Returns a dict suitable for embedding in the assessment JSON under the
    ``factor_attribution`` key.
    """
    summary: dict[str, Any] = {
        "model": regression["model"],
        "n_obs": regression["n_obs"],
        "alpha_annualized": round(regression["alpha_annualized"], 4),
        "alpha_t_stat": round(regression["alpha_t_stat"], 2),
        "alpha_p_value": round(regression["alpha_p_value"], 4),
        "alpha_significant": regression["alpha_significant"],
        "r_squared": round(regression["r_squared"], 3),
        "residual_sharpe": round(regression["residual_sharpe"], 2),
        "strategy_sharpe": round(regression["strategy_sharpe"], 2),
        "betas": {k: round(v, 4) for k, v in regression["betas"].items()},
        "significant_factors": [k for k, v in regression["p_values"].items() if v < 0.05],
    }

    # Classify the attribution result
    abs_residual = abs(regression["residual_sharpe"])
    if regression["alpha_significant"] and abs_residual > 0.3:
        summary["classification"] = "alpha-driven"
    elif abs_residual < 0.1:
        summary["classification"] = "exposure-dominated"
    else:
        summary["classification"] = "mixed"

    if bootstrap and bootstrap.get("n_boot", 0) > 0:
        summary["bootstrap"] = {
            "alpha_ann_ci": [
                round(bootstrap["alpha_ann_lo"], 4),
                round(bootstrap["alpha_ann_hi"], 4),
            ],
            "confidence": bootstrap["confidence"],
            "n_boot": bootstrap["n_boot"],
        }

    return summary


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_rolling_exposures(
    rolling: pd.DataFrame,
    title: str = "Rolling Factor Exposures",
) -> plt.Figure:
    """Plot rolling factor betas in a 2×3 grid.

    Args:
        rolling: DataFrame from compute_rolling_exposures()
        title: Figure title

    Returns:
        matplotlib Figure
    """
    # Determine factor columns (exclude alpha_ann and date index)
    factor_cols = [c for c in rolling.columns if c != "alpha_ann"]
    n_factors = len(factor_cols) + 1  # +1 for alpha
    ncols = 3
    nrows = int(np.ceil(n_factors / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    # Plot alpha first
    ax = axes.flat[0]
    ax.plot(rolling.index, rolling["alpha_ann"], linewidth=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Alpha (annualized)")
    ax.set_ylabel("Alpha")

    for i, col in enumerate(factor_cols):
        ax = axes.flat[i + 1]
        ax.plot(rolling.index, rolling[col], linewidth=0.8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(col)
        ax.set_ylabel("Beta")

    # Hide unused subplots
    for j in range(n_factors, nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


def plot_attribution_waterfall(
    regression: dict[str, Any],
    title: str = "Factor Attribution",
) -> plt.Figure:
    """Bar chart showing approximate factor contributions to strategy Sharpe.

    Decomposes strategy Sharpe into factor-explained and residual components.
    Contributions are proportional to |beta|, not to beta × factor_Sharpe,
    so the bar heights are an approximate visual aid rather than an exact
    return decomposition.
    """
    betas = regression["betas"]
    strategy_sr = regression["strategy_sharpe"]
    residual_sr = regression["residual_sharpe"]
    factor_sr = strategy_sr - residual_sr

    labels = list(betas.keys()) + ["Residual"]
    # Approximate factor contribution as beta × factor Sharpe (proportional)
    # For visualization, just show betas scaled to sum to factor_sr
    beta_vals = np.array(list(betas.values()))
    abs_sum = np.abs(beta_vals).sum()
    if abs_sum > 0:
        contributions = beta_vals / abs_sum * factor_sr
    else:
        contributions = np.zeros_like(beta_vals)
    values = list(contributions) + [residual_sr]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = ["#4A90D9" if v >= 0 else "#D94A4A" for v in values]
    colors[-1] = "#7B7B7B"  # Gray for residual

    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(
        strategy_sr,
        color="gray",
        linestyle="--",
        linewidth=0.5,
        label=f"Strategy Sharpe = {strategy_sr:.2f}",
    )

    ax.set_ylabel("Sharpe Contribution")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)

    # Add value labels
    for i, (label, val) in enumerate(zip(labels, values, strict=False)):
        ax.text(i, val + (0.02 if val >= 0 else -0.04), f"{val:+.2f}", ha="center", fontsize=9)

    return fig
