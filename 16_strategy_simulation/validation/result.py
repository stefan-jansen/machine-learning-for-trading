"""Standardized backtest result format shared by all framework adapters."""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    """Standardized result from any backtesting framework."""

    framework: str
    equity_curve: pd.Series  # DatetimeIndex -> portfolio value
    trades: pd.DataFrame  # (timestamp, asset, side, quantity, price, commission)
    positions: pd.DataFrame  # (timestamp, asset, quantity, value) or empty
    metrics: dict = field(default_factory=dict)

    @property
    def final_value(self) -> float:
        return float(self.equity_curve.iloc[-1])

    @property
    def total_return(self) -> float:
        return float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1)

    @property
    def n_trades(self) -> int:
        return len(self.trades)
