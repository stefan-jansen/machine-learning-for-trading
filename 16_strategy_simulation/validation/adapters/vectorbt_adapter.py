"""VectorBT OSS adapter for backtesting validation.

Takes pre-computed target weights and runs them through VectorBT's
Portfolio.from_orders with size_type='targetpercent'.

VectorBT fills at close price (same-bar execution), matching ml4t-backtest's
SAME_BAR mode.
"""

import pandas as pd
import polars as pl

from validation._library_validation import load_validation_helper
from validation.result import BacktestResult


def run_vectorbt(
    prices: pl.DataFrame,
    weights: pd.DataFrame,
    initial_cash: float = 1_000_000,
    commission_rate: float = 0.001,
) -> BacktestResult:
    """Run VectorBT OSS with pre-computed target weights.

    VectorBT executes at close price (same-bar), matching ml4t SAME_BAR mode.

    Args:
        prices: Polars DataFrame with [timestamp, asset, open, high, low, close, volume]
        weights: Pandas DataFrame with DatetimeIndex, asset columns, weight values
        initial_cash: Starting capital
        commission_rate: Per-trade commission as fraction

    Returns:
        BacktestResult with equity curve, trades, and metrics
    """
    vectorbt_runner = load_validation_helper("vectorbt_runner")
    vbt = vectorbt_runner.load_vectorbt_package("vectorbt")

    close_wide = vectorbt_runner.prices_to_wide(prices)
    close_aligned, weights_aligned = vectorbt_runner.align_weights_to_prices(weights, close_wide)
    portfolio = vectorbt_runner.run_vectorbt_orders(
        vbt=vbt,
        close=close_aligned,
        size=weights_aligned,
        size_type="targetpercent",
        init_cash=initial_cash,
        fees=commission_rate,
        group_by=True,
    )

    equity = vectorbt_runner.get_equity_curve(portfolio)
    trades_df = vectorbt_runner.extract_order_log(portfolio)
    metrics = dict(vectorbt_runner.summarize_order_run(equity, trades_df, initial_cash))

    return BacktestResult(
        framework="vectorbt",
        equity_curve=equity,
        trades=trades_df,
        positions=pd.DataFrame(),
        metrics=metrics,
    )
