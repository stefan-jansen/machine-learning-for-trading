"""ml4t-backtest adapter for backtesting validation.

Takes pre-computed target weights and runs them through ml4t-backtest's Engine
with TargetWeightExecutor, returning a standardized BacktestResult.

Supports multiple modes to match different framework behaviors:
- "backtrader": Integer shares, next-bar open fills, commission headroom (matches BT exactly)
- "vectorbt": Fractional shares, same-bar close fills (matches VBT)
- "lean": Verified LEAN execution/account semantics with case-study cost assumptions
- "default": Library defaults (next-bar open, fractional shares)
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
from ml4t.backtest import (
    BacktestConfig,
    DataFeed,
    Engine,
    Strategy,
)
from ml4t.backtest.config import (
    CommissionType,
    ExecutionMode,
    ExecutionPrice,
    FillOrdering,
    ShareType,
    SlippageType,
)
from ml4t.backtest.execution.rebalancer import RebalanceConfig, TargetWeightExecutor
from ml4t.backtest.types import OrderSide

from validation._library_validation import load_validation_helper
from validation.result import BacktestResult

# Match Backtrader adapter's commission headroom (avoids margin rejection)
COMMISSION_HEADROOM = 0.998


def _forward_fill_prices(prices: pl.DataFrame) -> tuple[pl.DataFrame, set[tuple]]:
    """Forward-fill missing bars so held positions are valued at last known close.

    Some assets (e.g., CME ag/livestock) don't trade on certain holidays while
    others do. Without synthetic bars, the engine falls back to entry_price for
    position valuation on those dates, creating artificial equity swings.

    VBT handles this internally via its wide-format matrix (NaN → ffill).
    We replicate by inserting forward-filled bars for missing (timestamp, asset) pairs.

    Returns:
        (filled_prices, synthetic_bars) where synthetic_bars is a set of
        (timestamp, asset) tuples that were forward-filled. The strategy should
        NOT trade on synthetic bars (VBT skips NaN-close assets).
    """
    all_timestamps = prices.select("timestamp").unique().sort("timestamp")
    all_assets = prices.select("symbol").unique().sort("symbol")

    # Track original (timestamp, asset) pairs
    original = set(zip(prices["timestamp"].to_list(), prices["symbol"].to_list(), strict=False))

    # Complete grid of all (timestamp, asset) pairs
    grid = all_timestamps.join(all_assets, how="cross")

    # Join existing prices onto the grid
    filled = grid.join(prices, on=["timestamp", "symbol"], how="left")

    # Forward-fill per asset (sort by timestamp first)
    filled = filled.sort(["symbol", "timestamp"]).with_columns(
        pl.col("open").forward_fill().over("symbol"),
        pl.col("high").forward_fill().over("symbol"),
        pl.col("low").forward_fill().over("symbol"),
        pl.col("close").forward_fill().over("symbol"),
        pl.col("volume").forward_fill().over("symbol"),
    )

    # Drop rows that are still null (before first bar for that asset)
    filled = filled.drop_nulls("close")

    # Identify synthetic bars
    synthetic = (
        set(zip(filled["timestamp"].to_list(), filled["symbol"].to_list(), strict=False)) - original
    )

    return filled, synthetic


def _filter_late_assets(
    prices: pl.DataFrame, weights: pd.DataFrame
) -> tuple[pl.DataFrame, pd.DataFrame]:
    """Remove assets whose price data starts after the first weight date.

    Backtrader's Cerebro aligns all data feeds to the LATEST start date.
    Including a late-starting asset (e.g., XLC starting 2018-06-19) would delay
    the entire simulation. We filter here so both engines use the same universe.
    """
    weight_start = pd.Timestamp(weights.index[0])
    asset_starts = prices.group_by("symbol").agg(pl.col("timestamp").min().alias("start"))
    late = asset_starts.filter(pl.col("start") > pl.lit(weight_start)).select("symbol")
    late_set = set(late["symbol"].to_list())

    if late_set:
        prices = prices.filter(~pl.col("symbol").is_in(late_set))
        weights = weights[[c for c in weights.columns if c not in late_set]]

    return prices, weights


def _align_prices_to_weight_window(
    prices: pl.DataFrame, weights: pd.DataFrame
) -> tuple[pl.DataFrame, pd.DataFrame]:
    """Restrict the simulation to the executable weight window.

    Pre-computed case-study weights are meant to be traded only over their own
    date range. Leaving earlier or later price bars in the feed creates flat
    warm-up periods before the first rebalance and, more importantly, allows
    next-bar fills to occur after the final target date. LEAN stops at the last
    target date, so the ml4t parity path must do the same.
    """
    if weights.empty:
        return prices, weights

    start = pd.Timestamp(weights.index.min()).to_pydatetime()
    end = pd.Timestamp(weights.index.max()).to_pydatetime()
    prices = prices.filter(
        (pl.col("timestamp") >= pl.lit(start)) & (pl.col("timestamp") <= pl.lit(end))
    )
    weights = weights.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    return prices, weights


def _quantize_prices_for_lean(prices: pl.DataFrame) -> pl.DataFrame:
    """Match LEAN daily data precision.

    The LEAN daily zip format stores OHLC values as 4-decimal fixed-point
    integers. Target sizing near integer-share thresholds can change by one
    share if ml4t uses the original higher-precision parquet prices while LEAN
    uses the quantized values.
    """
    return prices.with_columns(
        pl.col("open").round(4),
        pl.col("high").round(4),
        pl.col("low").round(4),
        pl.col("close").round(4),
    )


def _install_synthetic_execution_guard(broker, synthetic_bars: set[tuple]) -> None:
    """Prevent queued orders from filling on synthetic forward-filled bars.

    LEAN's fill-forward behavior makes stale prices available for sizing and
    valuation on sparse-panel dates, but market orders still wait for the next
    real bar. Forward-filling the ml4t feed without a guard would let pending
    next-bar orders execute against synthetic opens, which is too aggressive.
    """
    if not synthetic_bars:
        return

    synthetic_by_ts: dict[datetime, set[str]] = {}
    for ts, asset in synthetic_bars:
        synthetic_by_ts.setdefault(ts, set()).add(asset)

    original_process_orders = broker._process_orders
    cache_names = (
        "_current_prices",
        "_current_opens",
        "_current_highs",
        "_current_lows",
        "_current_closes",
        "_current_volumes",
        "_current_bids",
        "_current_asks",
        "_current_mids",
        "_current_bid_sizes",
        "_current_ask_sizes",
        "_current_signals",
    )

    def guarded_process_orders(*, use_open: bool = False):
        current_time = broker._current_time
        synthetic_assets = synthetic_by_ts.get(current_time, set()) if current_time else set()
        if not synthetic_assets:
            return original_process_orders(use_open=use_open)

        removed: dict[str, dict[str, object]] = {}
        for name in cache_names:
            cache = getattr(broker, name)
            removed[name] = {
                asset: cache.pop(asset) for asset in synthetic_assets if asset in cache
            }

        try:
            return original_process_orders(use_open=use_open)
        finally:
            for name, values in removed.items():
                getattr(broker, name).update(values)

    broker._process_orders = guarded_process_orders


class PrecomputedWeightStrategy(Strategy):
    """Strategy that reads pre-computed target weights per timestamp."""

    def __init__(
        self,
        weights: dict[datetime, dict[str, float]],
        executor: TargetWeightExecutor,
    ) -> None:
        self.executor = executor
        self._weights = weights

    def on_data(self, timestamp, data, context, broker) -> None:
        if timestamp in self._weights:
            self.executor.execute(self._weights[timestamp], data, broker)


class VBTSequentialStrategy(Strategy):
    """Strategy matching VectorBT's sequential order processing.

    VBT with auto_call_seq=False (default), cash_sharing=True, update_value=False:
    1. Compute portfolio value ONCE (pre-trade snapshot, "frozen value")
    2. Process assets in COLUMN ORDER (alphabetical from the pivot)
    3. For each asset: target_shares = (weight * frozen_value) / price
    4. delta = target_shares - current_position (position updated by prior fills)
    5. Fill immediately via broker._process_orders() — cash and positions update
       sequentially, but frozen_value stays constant for all target computations.
    6. Cash-capped: broker rejects/partial-fills when insufficient cash (reject=True).
    """

    def __init__(
        self,
        weights: dict[datetime, dict[str, float]],
        column_order: list[str],
        synthetic_bars: set[tuple] | None = None,
    ) -> None:
        self._weights = weights
        self._column_order = column_order
        self._synthetic_bars = synthetic_bars or set()

    def on_data(self, timestamp, data, context, broker) -> None:
        if timestamp not in self._weights:
            return

        target_w = self._weights[timestamp]
        frozen_value = broker.get_account_value()

        for asset in self._column_order:
            weight = target_w.get(asset, 0.0)
            price = data.get(asset, {}).get("close")
            if not price or price <= 0:
                continue

            # Skip synthetic (forward-filled) bars — VBT skips NaN-close assets
            if (timestamp, asset) in self._synthetic_bars:
                continue

            position = broker.get_position(asset)
            current_shares = position.quantity if position else 0.0

            target_shares = (weight * frozen_value) / price
            delta = target_shares - current_shares

            # Match VBT's min_size filter (default 1e-8)
            if abs(delta) < 1e-8:
                continue

            if delta > 0:
                broker.submit_order(asset, delta, OrderSide.BUY)
            else:
                broker.submit_order(asset, abs(delta), OrderSide.SELL)

            broker._process_orders()


class SequentialWeightStrategy(Strategy):
    """Strategy matching Backtrader's order_target_percent with COC.

    Processes each asset sequentially in sorted order. After each order,
    calls broker._process_orders() to fill immediately, so the next asset's
    target is computed from the UPDATED portfolio value. This matches
    Backtrader's cheat-on-close behavior where each order_target_percent
    call fills before the next call in the loop.

    Uses BT's exact share computation: target_shares = int(target_value / price),
    then delta = target_shares - current_shares. This avoids rounding differences
    versus computing delta_value first then dividing.
    """

    def __init__(
        self,
        weights: dict[datetime, dict[str, float]],
        min_trade_pct: float = 0.0,
    ) -> None:
        self._weights = weights
        self._min_trade_pct = min_trade_pct

    def on_data(self, timestamp, data, context, broker) -> None:
        if timestamp not in self._weights:
            return

        target_w = self._weights[timestamp]

        for asset in sorted(target_w.keys()):
            target_pct = target_w[asset]
            price = data.get(asset, {}).get("close")
            if not price or price <= 0:
                continue

            # Match BT's order_target_percent exactly:
            # 1. target_value = getvalue() * target_pct
            # 2. current_value = getvalue(datas=[data])  (= position.size * close)
            # 3. if target > current: size = int((target - current) // price)
            # 4. if target < current: size = int((current - target) // price)
            equity = broker.get_account_value()
            target_value = equity * target_pct

            position = broker.get_position(asset)
            current_shares = position.quantity if position else 0
            current_value = current_shares * price

            if target_value > current_value:
                delta_value = target_value - current_value
                size = int(delta_value // price)
                if size == 0:
                    continue
                broker.submit_order(asset, size, OrderSide.BUY)
            elif target_value < current_value:
                delta_value = current_value - target_value
                size = int(delta_value // price)
                if size == 0:
                    continue
                broker.submit_order(asset, size, OrderSide.SELL)
            else:
                continue

            # Process immediately — matches BT's COC sequential fill behavior.
            broker._process_orders()


class LeanWeightStrategy(Strategy):
    """Strategy matching the LEAN case-study target-weight protocol.

    LEAN sizes from a frozen pre-trade account value on each rebalance date,
    rounds to integer shares, and submits one target order per active asset.
    Orders fill later under the configured engine semantics, so sizing should
    not update during the loop.
    """

    def __init__(self, weights: dict[datetime, dict[str, float]]) -> None:
        self._weights = weights

    @staticmethod
    def _target_quantity(target_weight: float, equity: float, price: float) -> int:
        raw_qty = (target_weight * equity) / price
        if raw_qty >= 0:
            return int(math.floor(raw_qty + 1e-12))
        return -int(math.floor(abs(raw_qty) + 1e-12))

    def on_data(self, timestamp, data, context, broker) -> None:
        if timestamp not in self._weights:
            return

        target_w = self._weights[timestamp]
        frozen_value = broker.get_account_value()

        active_assets = set(target_w.keys())
        active_assets.update(broker.positions.keys())

        for asset in sorted(active_assets):
            price = data.get(asset, {}).get("close")
            if not price or price <= 0:
                continue

            target_qty = self._target_quantity(target_w.get(asset, 0.0), frozen_value, price)
            position = broker.get_position(asset)
            current_qty = int(position.quantity) if position else 0
            delta = target_qty - current_qty
            if delta == 0:
                continue

            if delta > 0:
                broker.submit_order(asset, delta, OrderSide.BUY)
            else:
                broker.submit_order(asset, abs(delta), OrderSide.SELL)


def _prepare_weights(
    weights: pd.DataFrame,
    mode: str = "default",
    commission_headroom: float = COMMISSION_HEADROOM,
) -> dict[datetime, dict[str, float]]:
    """Convert weights DataFrame to {datetime: {asset: weight}} dict.

    For backtrader/vectorbt modes: include ALL assets (even zero-weight) so the
    strategy can close positions when an asset drops out of the target portfolio.

    For default mode: include only non-zero weights.
    """
    result: dict[datetime, dict[str, float]] = {}
    scale = commission_headroom if mode == "backtrader" else 1.0
    all_cols = sorted(weights.columns)

    for ts in weights.index:
        ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        row = weights.loc[ts]

        if mode in ("backtrader", "vectorbt"):
            # Include ALL assets (zero-weight = sell/close target).
            w = {col: float(row[col]) * scale for col in all_cols}
            result[ts_dt] = w
        else:
            # Only non-zero weights; positions not in target are closed separately
            w = {col: float(row[col]) * scale for col in all_cols if row[col] > 0}
            if any(v > 0 for v in w.values()):
                result[ts_dt] = w

    return result


def run_ml4t(
    prices: pl.DataFrame,
    weights: pd.DataFrame,
    initial_cash: float = 1_000_000,
    commission_rate: float = 0.001,
    mode: str = "default",
    same_bar: bool = False,
) -> BacktestResult:
    """Run ml4t-backtest with pre-computed target weights.

    Args:
        prices: Polars DataFrame with [timestamp, asset, open, high, low, close, volume]
        weights: Pandas DataFrame with DatetimeIndex, asset columns, weight values
        initial_cash: Starting capital
        commission_rate: Per-trade commission as fraction (e.g., 0.001 = 10 bps)
        mode: Configuration mode:
            - "backtrader": Match Backtrader (integer shares, next-bar open, headroom)
            - "vectorbt": Match VectorBT (fractional, same-bar close)
            - "lean": Reuse the verified LEAN execution/account profile
            - "default": Library defaults
        same_bar: If True, use same-bar execution (overrides preset's fill timing).
            For backtrader mode with same_bar=True, matches BT's cheat-on-close.

    Returns:
        BacktestResult with equity curve, trades, and metrics
    """
    if mode == "backtrader":
        # Filter late-starting assets (BT aligns all feeds to latest start)
        prices, weights = _filter_late_assets(prices, weights)
        return _run_backtrader_mode(prices, weights, initial_cash, commission_rate, same_bar)
    elif mode == "vectorbt":
        # VBT does NOT filter late assets — it uses all columns in the matrix
        return _run_vectorbt_mode(prices, weights, initial_cash, commission_rate)
    elif mode == "lean":
        prices, weights = _align_prices_to_weight_window(prices, weights)
        prices = _quantize_prices_for_lean(prices)
        return _run_lean_mode(prices, weights, initial_cash, commission_rate)
    else:
        prices, weights = _filter_late_assets(prices, weights)
        return _run_default_mode(prices, weights, initial_cash, commission_rate, same_bar)


def _run_backtrader_mode(
    prices: pl.DataFrame,
    weights: pd.DataFrame,
    initial_cash: float,
    commission_rate: float,
    same_bar: bool,
) -> BacktestResult:
    """Run with Backtrader-compatible configuration.

    For same_bar=True (cheat-on-close): uses SequentialWeightStrategy that
    processes each asset one at a time with immediate fills, matching BT's
    COC behavior where each order_target_percent fills before the next.

    For same_bar=False (next-bar): uses TargetWeightExecutor with the
    backtrader preset. Orders queue and fill at next bar's open.
    """
    config = BacktestConfig.from_preset("backtrader")
    config.initial_cash = initial_cash
    config.commission_rate = commission_rate
    # BT default is zero slippage; preset has 0.1% — override to match BT
    config.slippage_type = SlippageType.NONE
    config.slippage_rate = 0.0

    # Scale weights by commission headroom and sort keys (matching BT adapter)
    weight_dict = _prepare_weights(weights, mode="backtrader")

    if same_bar:
        # Match Backtrader's cheat-on-close: sequential per-asset processing.
        # Each order fills immediately, updating portfolio value for the next asset.
        config.execution_price = ExecutionPrice.CLOSE
        config.execution_mode = ExecutionMode.SAME_BAR

        strategy = SequentialWeightStrategy(weight_dict)
    else:
        # Next-bar mode: use TargetWeightExecutor (batch processing).
        # Integer shares, zero thresholds — BT relies on integer rounding
        # (shares==0 → no order) to filter tiny trades.
        rebalance_config = RebalanceConfig(
            allow_fractional=False,
            min_trade_value=0.0,
            min_weight_change=0.0,
        )
        executor = TargetWeightExecutor(config=rebalance_config)
        strategy = PrecomputedWeightStrategy(weight_dict, executor)

    feed = DataFeed(prices_df=prices)
    engine = Engine.from_config(feed, strategy, config)
    result = engine.run()

    return _convert_result(result, "ml4t", initial_cash)


def _run_vectorbt_mode(
    prices: pl.DataFrame,
    weights: pd.DataFrame,
    initial_cash: float,
    commission_rate: float,
) -> BacktestResult:
    """Run with VectorBT-compatible configuration.

    Matches VBT's exact execution model:
    - Fractional shares, same-bar close fills
    - Sequential processing in column order (VBT's default auto_call_seq=False)
    - Frozen pre-trade portfolio value for all target computations
    - Cash-capped partial fills (reject_on_insufficient_cash=True)

    VBT processes assets in column order (alphabetical from pivot), computing
    target_shares = (weight * frozen_value) / price for each. Cash and positions
    update after each fill, but the value used for sizing stays constant.
    """
    # Forward-fill missing bars so position valuation matches VBT's ffill behavior.
    # Track which bars are synthetic so the strategy skips them (VBT skips NaN close).
    # NOTE: This works around an ml4t-backtest bug where broker._update_time() replaces
    # _current_prices instead of merging, causing get_account_value() to fall back to
    # entry_price for assets without a bar on the current timestamp.
    prices, synthetic_bars = _forward_fill_prices(prices)

    config = BacktestConfig.from_preset("vectorbt")
    config.initial_cash = initial_cash
    config.commission_type = CommissionType.PERCENTAGE
    config.commission_rate = commission_rate
    # Sequential processing: one order at a time via broker._process_orders()
    config.fill_ordering = FillOrdering.FIFO
    # Cash-capped: VBT partial-fills when cash is insufficient, never over-allocates
    config.reject_on_insufficient_cash = True
    config.partial_fills_allowed = True

    # Determine VBT's column order (from the close price pivot)
    vectorbt_runner = load_validation_helper("vectorbt_runner")
    close_wide = vectorbt_runner.prices_to_wide(prices)
    column_order = list(close_wide.columns)

    # Include ALL assets (zero-weight = close position)
    weight_dict = _prepare_weights(weights, mode="vectorbt")

    strategy = VBTSequentialStrategy(weight_dict, column_order, synthetic_bars)
    feed = DataFeed(prices_df=prices)

    engine = Engine.from_config(feed, strategy, config)
    result = engine.run()

    return _convert_result(result, "ml4t", initial_cash)


def _run_default_mode(
    prices: pl.DataFrame,
    weights: pd.DataFrame,
    initial_cash: float,
    commission_rate: float,
    same_bar: bool,
) -> BacktestResult:
    """Run with library defaults (backward compatible)."""
    config = BacktestConfig(
        initial_cash=initial_cash,
        commission_type=CommissionType.PERCENTAGE,
        commission_rate=commission_rate,
        slippage_type=SlippageType.NONE,
        slippage_rate=0.0,
        execution_mode=ExecutionMode.SAME_BAR if same_bar else ExecutionMode.NEXT_BAR,
    )

    rebalance_config = RebalanceConfig(
        allow_fractional=True,
        min_trade_value=1.0,
        min_weight_change=0.0001,
    )

    weight_dict = _prepare_weights(weights, mode="default")

    executor = TargetWeightExecutor(config=rebalance_config)
    strategy = PrecomputedWeightStrategy(weight_dict, executor)
    feed = DataFeed(prices_df=prices)

    engine = Engine.from_config(feed, strategy, config)
    result = engine.run()

    return _convert_result(result, "ml4t", initial_cash)


def _run_lean_mode(
    prices: pl.DataFrame,
    weights: pd.DataFrame,
    initial_cash: float,
    commission_rate: float,
) -> BacktestResult:
    """Run with the verified LEAN-compatible execution/account profile."""
    prices, synthetic_bars = _forward_fill_prices(prices)

    config = BacktestConfig.from_preset("lean")
    config.initial_cash = initial_cash
    config.commission_type = CommissionType.PERCENTAGE
    config.commission_rate = commission_rate
    config.slippage_type = SlippageType.NONE
    config.slippage_rate = 0.0

    weight_dict = _prepare_weights(weights, mode="vectorbt")
    strategy = LeanWeightStrategy(weight_dict)
    feed = DataFeed(prices_df=prices)

    engine = Engine.from_config(feed, strategy, config)
    _install_synthetic_execution_guard(engine.broker, synthetic_bars)
    result = engine.run()

    return _convert_result(result, "ml4t-lean", initial_cash)


def _convert_result(result, framework: str, initial_cash: float) -> BacktestResult:
    """Convert ml4t-backtest result to standardized BacktestResult.

    Uses fills (individual orders) rather than trades (round-trips) for the
    trade list. This matches Backtrader's convention where each fill is a
    separate entry in the trade log.
    """
    # Convert equity curve to pandas Series
    eq_dates, eq_values = (
        zip(*result.equity_curve, strict=False) if result.equity_curve else ([], [])
    )
    equity_series = pd.Series(eq_values, index=pd.DatetimeIndex(eq_dates), name="portfolio_value")

    # Use fills (individual orders) for apples-to-apples comparison with BT/VBT.
    # BT records each buy/sell as a separate trade; ml4t's "trades" are round-trips.
    # Filter sub-1e-8 fills: VBT's min_size=1e-8 rejects these internally, but ml4t's
    # broker records them as floating-point dust from sequential processing. They have
    # zero economic impact (total value < $0.01 across hundreds of thousands of fills).
    trades_data = []
    for f in result.fills:
        qty = abs(f.quantity)
        if qty < 1e-8:
            continue
        trades_data.append(
            {
                "timestamp": f.timestamp,
                "symbol": f.asset,
                "side": f.side.name.lower(),
                "quantity": qty,
                "price": f.price,
                "commission": f.commission,
            }
        )
    trades_df = (
        pd.DataFrame(trades_data)
        if trades_data
        else pd.DataFrame(
            columns=["timestamp", "symbol", "side", "quantity", "price", "commission"]
        )
    )

    return BacktestResult(
        framework=framework,
        equity_curve=equity_series,
        trades=trades_df,
        positions=pd.DataFrame(),
        metrics=result.metrics,
    )
