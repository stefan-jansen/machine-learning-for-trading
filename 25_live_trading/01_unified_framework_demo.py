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
# # Unified Framework Demo: Same Strategy, Backtest to Live
#
# **Docker image**: `ml4t`
#
# **Section Reference**: 25.1 (Unified Framework Advantage)
#
# **Implementation Skills**:
# - ml4t.backtest: Strategy base class, Engine, DataFeed
# - ml4t.live: LiveEngine, SafeBroker, LiveRiskConfig, VirtualPortfolio
# - Core value proposition: zero code changes from backtest to live
#
# **Key Learning**:
# This notebook demonstrates the fundamental value of the unified framework approach:
# the **same Strategy class** produces **identical signals** whether running in backtest
# mode or live mode. No code changes required.
#
# **Why This Matters**:
# - Eliminates "two pipelines" divergence bugs
# - Confident deployment: what you backtest is what you trade
# - Faster iteration: test ideas in backtest, deploy same code live
#
# **Structure**:
# 1. Define a simple momentum strategy (ONE implementation)
# 2. Run in backtest mode (ml4t.backtest.Engine)
# 3. Run in live mode with historical replay (ml4t.live.LiveEngine)
# 4. Compare signals: prove they match perfectly
#
# **Learning Objectives**
# - Verify that a single strategy class can run without code changes in both engines.
# - Compare backtest and live-style signals on a shared historical data set.
# - Interpret any mismatch as a technical-divergence problem rather than as a market problem.
#
# **Prerequisites**
# - Familiarity with the `Strategy` interface and the distinction between backtest and live engines.
# - Review Chapter 25.1, which motivates why technical parity matters more than demo complexity here.

# %% [markdown]
# ## Setup
#
# The setup keeps the demo focused on parity. We are not trying to optimize a strategy here; we are trying to
# prove that the same implementation survives the engine swap intact.

# %%
"""Verify backtest-to-live signal parity with a single strategy class."""

import warnings

warnings.filterwarnings("ignore")

# %%
import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl
from async_utils import run_async

# ml4t.backtest imports
from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, Strategy
from ml4t.backtest.types import Order, OrderSide, OrderStatus, OrderType, Position

# ml4t.live imports
from ml4t.live import (
    LiveEngine,
    VirtualPortfolio,
)

from data import load_etfs

# Configure logging for live mode
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise for demo
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# %% [markdown]
# ## 1. Configuration
#
# The configuration defines the exact parity experiment: same symbols, same dates, and same moving-average
# parameters in both engines. If those inputs drift, signal comparison becomes meaningless.

# %% tags=["parameters"]
MAX_SYMBOLS = 0
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"

# %%
ALL_SYMBOLS = ["SPY", "QQQ", "IWM"]
SYMBOLS = ALL_SYMBOLS[:MAX_SYMBOLS] if MAX_SYMBOLS > 0 else ALL_SYMBOLS.copy()
INITIAL_CASH = 100_000
FAST_MA = 10
SLOW_MA = 30

print(f"Symbols: {SYMBOLS}")
print(f"Period: {START_DATE} to {END_DATE}")

# %% [markdown]
# **Finding:** The configuration printout defines the exact parity test: one symbol set, one date range, and
# one pair of moving-average parameters. Changing any of those between modes would invalidate the comparison.

# %% [markdown]
# ## 2. Data Acquisition
#
# Download historical data for our demo. Both backtest and live (replay) will use the same data.
#
# The shared data set is the notebook's most important control variable. If both engines do not consume the
# same bars, a later signal mismatch tells us nothing useful about the framework itself.

# %%
print("Loading ETF data from canonical source...")
etf_data = load_etfs()

# Filter for our symbols and date range
etf_filtered = etf_data.filter(
    (pl.col("symbol").is_in(SYMBOLS))
    & (pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    & (pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
).sort("timestamp")

# Convert to yfinance-like MultiIndex format for compatibility with rest of notebook
# Pivot each column separately and combine
ohlcv_cols = ["open", "high", "low", "close", "volume"]
raw_data_dict = {}
for col in ohlcv_cols:
    pivot = (
        etf_filtered.select(["timestamp", "symbol", col])
        .pivot(on="symbol", index="timestamp", values=col)
        .sort("timestamp")
        .to_pandas()
        .set_index("timestamp")
    )
    raw_data_dict[col.title()] = pivot

# Create MultiIndex columns similar to yfinance output
raw_data = pd.concat(raw_data_dict, axis=1)
raw_data = raw_data.ffill().dropna()
raw_data.index = pd.to_datetime(raw_data.index, utc=True)

# Update SYMBOLS to only include available symbols
available_symbols = [s for s in SYMBOLS if s in raw_data["Close"].columns]
SYMBOLS = available_symbols

print(f"Loaded {len(raw_data):,} daily bars for {len(SYMBOLS)} symbols")

# Store close prices for strategy signals (used later for comparison)
close_prices = raw_data["Close"].ffill()

# %% [markdown]
# **Finding:** The data summary above confirms that both engines will see the same cleaned close-price history.
# That shared tape is what turns the rest of the notebook into a genuine technical-parity test.

# %% [markdown]
# ## 3. The Strategy: ONE Implementation
#
# This is the **key insight**: we define the strategy ONCE. It works in both
# ml4t.backtest.Engine AND ml4t.live.LiveEngine without modification.
#
# **Strategy Logic**: Simple Dual Moving Average Crossover
# - Calculate 10-day and 30-day SMAs
# - Buy when fast > slow (bullish crossover)
# - Sell when fast < slow (bearish crossover)
# - Track all signals for comparison
#
# The point of the strategy is not its alpha. It is intentionally plain so any difference between backtest and
# live outputs is easy to trace to the framework rather than to model complexity.


# %%
class DualMAStrategy(Strategy):
    """Simple dual moving average crossover strategy.

    This strategy works IDENTICALLY in backtest and live modes.
    The on_data() method signature is the same for both engines.

    Attributes:
        fast_period: Fast MA lookback (default: 10)
        slow_period: Slow MA lookback (default: 30)
        signal_log: Records all signals for verification
    """

    def __init__(self, symbol: str, fast_period: int = 10, slow_period: int = 30):
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period

        # Price history for MA calculation
        self.prices: list[float] = []

        # Signal log for comparison
        self.signal_log: list[dict] = []

    def on_start(self, broker):
        """Called when engine starts."""
        self.prices = []
        self.signal_log = []

    def on_data(self, timestamp: datetime, data: dict, context: dict, broker):
        """Process each bar and generate signals.

        CRITICAL: This method signature is IDENTICAL for backtest and live!

        Args:
            timestamp: Bar timestamp
            data: {symbol: {'open', 'high', 'low', 'close', 'volume'}}
            context: Additional metadata
            broker: Broker interface (sync in both modes)
        """
        # Get bar for our symbol
        bar = data.get(self.symbol)
        if not bar:
            return

        close = bar["close"]
        self.prices.append(close)

        # Need enough history for slow MA
        if len(self.prices) < self.slow_period:
            return

        # Calculate moving averages
        fast_ma = sum(self.prices[-self.fast_period :]) / self.fast_period
        slow_ma = sum(self.prices[-self.slow_period :]) / self.slow_period

        # Get current position
        position = broker.get_position(self.symbol)
        has_position = position is not None and position.quantity > 0

        # Generate signal
        signal = None
        if fast_ma > slow_ma and not has_position:
            signal = "BUY"
            broker.submit_order(self.symbol, 100, side=OrderSide.BUY)

        elif fast_ma < slow_ma and has_position:
            signal = "SELL"
            broker.submit_order(self.symbol, 100, side=OrderSide.SELL)

        # Log signal for comparison
        if signal:
            self.signal_log.append(
                {
                    "timestamp": timestamp,
                    "symbol": self.symbol,
                    "signal": signal,
                    "fast_ma": round(fast_ma, 2),
                    "slow_ma": round(slow_ma, 2),
                    "price": round(close, 2),
                }
            )

    def on_end(self, broker):
        """Called when engine stops."""
        pass


# %% [markdown]
# ## 4. Backtest Mode: ml4t.backtest.Engine
#
# First, run the strategy in backtest mode using `ml4t.backtest.Engine`. This creates the reference output
# that the live-style replay must match if the unified-framework claim is actually true.

# %%
# Prepare data for backtest engine (long format)
data_records = []
for date in raw_data.index:
    for symbol in SYMBOLS:
        if pd.notna(raw_data["Close"].loc[date, symbol]):
            data_records.append(
                {
                    "timestamp": date.to_pydatetime(),
                    "symbol": symbol,
                    "open": float(raw_data["Open"].loc[date, symbol]),
                    "high": float(raw_data["High"].loc[date, symbol]),
                    "low": float(raw_data["Low"].loc[date, symbol]),
                    "close": float(raw_data["Close"].loc[date, symbol]),
                    "volume": float(raw_data["Volume"].loc[date, symbol]),
                }
            )

prices_df = pl.DataFrame(data_records)
print(f"Prepared {len(prices_df):,} price records for backtest")

# %% [markdown]
# **Finding:** The prepared price-record count confirms that the backtest engine is consuming the same long
# format tape the rest of the notebook expects.
#
# **Trading implication:** Technical parity starts with data-shape parity; if the bar stream changes across
# engines, later signal mismatches are not diagnostically useful.
#
# %%
# Create backtest components
feed_backtest = DataFeed(prices_df=prices_df)
strategy_backtest = DualMAStrategy(symbol="SPY", fast_period=FAST_MA, slow_period=SLOW_MA)

engine_backtest = Engine(
    feed=feed_backtest,
    strategy=strategy_backtest,
    config=BacktestConfig(
        initial_cash=INITIAL_CASH,
        # The replay broker also fills at the finalized current close. SAME_BAR
        # therefore isolates engine/strategy parity. This is not a performance
        # estimate and makes no claim about executable close prices.
        execution_mode=ExecutionMode.SAME_BAR,
        commission_rate=0.0005,
    ),
)

# Run backtest
results = engine_backtest.run()

print(f"Final value:       ${results['final_value']:,.2f}")
print(f"Total return:      {results['total_return_pct']:.2f}%")
print(f"Total trades:      {results['num_trades']}")
print(f"Signals generated: {len(strategy_backtest.signal_log)}")

# Store backtest signals for comparison
backtest_signals = strategy_backtest.signal_log.copy()

# %% [markdown]
# **Finding:** The backtest run establishes the reference signal tape on the shared ETF history.
#
# **Trading implication:** If live mode later disagrees, the notebook can attribute the problem to engine or
# wrapper behavior rather than to different market data.
#
# %% [markdown]
# ## 5. Live Mode Infrastructure
#
# To run the same strategy in live mode, we need:
# 1. A **simulated broker** (implements AsyncBrokerProtocol)
# 2. A **historical replay feed** (implements DataFeedProtocol)
#
# These components enable us to demo live mode without requiring
# a real broker connection or market hours.


# %%
class SimulatedBroker:
    """Simulated broker for demo purposes.

    Implements AsyncBrokerProtocol to work with LiveEngine.
    Uses VirtualPortfolio for realistic position tracking.
    """

    def __init__(self, initial_cash: float = 100_000.0):
        self._portfolio = VirtualPortfolio(initial_cash=initial_cash)
        self._connected = False
        self._pending_orders: list[Order] = []
        self._order_count = 0

        # Current prices for market orders
        self._current_prices: dict[str, float] = {}
        self._current_timestamp: datetime | None = None

    async def connect(self) -> None:
        """Connect (no-op for simulation)."""
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect (no-op for simulation)."""
        self._connected = False

    async def is_connected_async(self) -> bool:
        return self._connected

    @property
    def positions(self) -> dict[str, Position]:
        return self._portfolio.positions

    @property
    def pending_orders(self) -> list[Order]:
        return self._pending_orders

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_position(self, asset: str) -> Position | None:
        return self._portfolio.positions.get(asset)

    async def get_positions_async(self) -> dict[str, Position]:
        return self._portfolio.positions

    async def get_pending_orders_async(self) -> list[Order]:
        return self._pending_orders

    async def get_position_async(self, asset: str) -> Position | None:
        return self.get_position(asset)

    async def get_account_value_async(self) -> float:
        return self._portfolio.account_value

    async def get_cash_async(self) -> float:
        return self._portfolio.cash

    def update_price(self, asset: str, price: float, timestamp: datetime) -> None:
        """Update current price for market orders."""
        self._current_prices[asset] = price
        self._current_timestamp = timestamp
        self._portfolio.update_prices({asset: price})

    async def submit_order_async(
        self,
        asset: str,
        quantity: int,
        side: OrderSide | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        **kwargs,
    ) -> Order:
        """Submit and immediately fill order (simulation)."""
        if side is None:
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
            quantity = abs(quantity)

        # Get fill price
        price = limit_price or self._current_prices.get(asset, 100.0)

        self._order_count += 1
        fill_timestamp = self._current_timestamp or datetime.min

        # Create filled order
        order = Order(
            asset=asset,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            order_id=f"SIM-{self._order_count:04d}",
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=price,
            filled_at=fill_timestamp,
        )

        # Update portfolio
        self._portfolio.process_fill(order)

        return order

    async def cancel_order_async(self, order_id: str) -> bool:
        return False

    async def close_position_async(self, asset: str) -> Order | None:
        pos = self.get_position(asset)
        if pos and pos.quantity != 0:
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            return await self.submit_order_async(asset, abs(pos.quantity), side)
        return None


# %% [markdown]
# ### Historical Replay Feed
#
# The historical replay feed is the notebook's stand-in for a real streaming source. Its job is to preserve
# live-engine semantics while holding the market data constant.


# %%
class HistoricalReplayFeed:
    """Replays historical data as a live feed.

    Implements DataFeedProtocol for use with LiveEngine.
    Enables live mode demos without requiring market hours.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        symbols: list[str],
        broker: SimulatedBroker | None = None,
    ):
        """Initialize replay feed.

        Args:
            data: DataFrame with OHLCV data (MultiIndex columns: metric, symbol)
            symbols: List of symbols to include
            broker: Optional broker to update prices
        """
        self._data = data
        self._symbols = symbols
        self._broker = broker
        self._running = False
        self._index = 0
        self._dates = list(data.index)
        self._stats = {"bars_emitted": 0}

    async def start(self) -> None:
        """Start the feed."""
        self._running = True
        self._index = 0

    def stop(self) -> None:
        """Stop the feed."""
        self._running = False

    @property
    def stats(self) -> dict[str, Any]:
        return self._stats

    def __aiter__(
        self,
    ) -> AsyncIterator[tuple[datetime, dict[str, dict[str, Any]], dict[str, Any]]]:
        return self

    async def __anext__(
        self,
    ) -> tuple[datetime, dict[str, dict[str, Any]], dict[str, Any]]:
        """Get next bar."""
        if not self._running or self._index >= len(self._dates):
            raise StopAsyncIteration

        date = self._dates[self._index]
        self._index += 1

        # Build bar data
        data: dict[str, dict[str, Any]] = {}
        for symbol in self._symbols:
            try:
                bar = {
                    "open": float(self._data["Open"].loc[date, symbol]),
                    "high": float(self._data["High"].loc[date, symbol]),
                    "low": float(self._data["Low"].loc[date, symbol]),
                    "close": float(self._data["Close"].loc[date, symbol]),
                    "volume": float(self._data["Volume"].loc[date, symbol]),
                }
                data[symbol] = bar

                # Update broker prices for fills
                if self._broker:
                    timestamp = date.to_pydatetime() if hasattr(date, "to_pydatetime") else date
                    self._broker.update_price(symbol, bar["close"], timestamp)
            except (KeyError, ValueError):
                pass

        self._stats["bars_emitted"] += 1

        # Small delay to simulate real-time (optional, can be 0)
        await asyncio.sleep(0)

        timestamp = date.to_pydatetime() if hasattr(date, "to_pydatetime") else date
        return timestamp, data, {}


# %% [markdown]
# ## 6. Live Mode: ml4t.live.LiveEngine
#
# Now run the **same strategy** using `ml4t.live.LiveEngine` with historical replay. The point is not to
# simulate latency perfectly, but to test whether the engine swap changes the trading logic.


# %%
async def run_live_mode():
    """Run strategy in live mode with historical replay."""
    # Create simulated broker
    # NOTE: In production, wrap with SafeBroker for risk controls.
    # Here we use the raw broker to demonstrate pure signal parity.
    broker = SimulatedBroker(initial_cash=INITIAL_CASH)

    # Create historical replay feed
    feed = HistoricalReplayFeed(data=raw_data, symbols=SYMBOLS, broker=broker)

    # Create strategy (SAME CLASS as backtest!)
    strategy_live = DualMAStrategy(symbol="SPY", fast_period=FAST_MA, slow_period=SLOW_MA)

    # Create LiveEngine
    engine = LiveEngine(
        strategy=strategy_live,
        broker=broker,
        feed=feed,
    )

    # Connect and run
    await engine.connect()

    try:
        await engine.run()
    finally:
        await engine.stop()

    print(f"Bars processed:    {engine.stats['bar_count']}")
    print(f"Signals generated: {len(strategy_live.signal_log)}")

    return strategy_live.signal_log


# Run async live mode
live_signals = run_async(run_live_mode())

# %% [markdown]
# **Finding:** The live replay consumes the same historical bars through `LiveEngine`, proving that the engine
# swap does not require a second strategy implementation.
#
# **Trading implication:** Deployment confidence improves when the only moving part is the engine and not the
# strategy logic itself.
#
# %% [markdown]
# ## 7. Signal Comparison: Proof of Parity
#
# This is the notebook's decisive comparison: do backtest and live mode produce **identical** signals on the
# same tape? Any mismatch here is a framework problem until proven otherwise.

# %%
print(f"Backtest signals: {len(backtest_signals)}")
print(f"Live signals:     {len(live_signals)}")

# Falsifiability gate: a different count of signals is itself a parity
# failure. The pairwise comparison only makes sense after the counts
# match, so raise immediately if they don't.
assert len(backtest_signals) == len(live_signals), (
    f"Signal count mismatch: backtest={len(backtest_signals)} "
    f"vs live={len(live_signals)} - the engines disagree on the "
    "signal tape, not just on signal values."
)

# %% [markdown]
# Build a single comparison frame keyed by signal index with the
# `engine`, `timestamp`, `symbol`, `side`, `price`, `fast_ma`, `slow_ma`,
# and a `match` column that is True only when every field agrees within
# the tolerance. The frame is the audit surface a mismatch would land on.


# %%
def _rows_for(engine: str, signals: list[dict]) -> list[dict]:
    return [
        {
            "engine": engine,
            "i": i,
            "timestamp": sig["timestamp"],
            "symbol": "SPY",
            "side": sig["signal"],
            "price": sig["price"],
            "fast_ma": sig["fast_ma"],
            "slow_ma": sig["slow_ma"],
        }
        for i, sig in enumerate(signals)
    ]


comparison = pl.DataFrame(
    _rows_for("backtest", backtest_signals) + _rows_for("live", live_signals)
).sort(["i", "engine"])

# Pivot to one row per signal index and compare every recorded field.
pairs = (
    comparison.pivot(
        values=["timestamp", "symbol", "side", "price", "fast_ma", "slow_ma"],
        index="i",
        on="engine",
    )
    .with_columns(
        match=(
            (pl.col("timestamp_backtest") == pl.col("timestamp_live"))
            & (pl.col("symbol_backtest") == pl.col("symbol_live"))
            & (pl.col("side_backtest") == pl.col("side_live"))
            & ((pl.col("price_backtest") - pl.col("price_live")).abs() < 0.01)
            & ((pl.col("fast_ma_backtest") - pl.col("fast_ma_live")).abs() < 0.01)
            & ((pl.col("slow_ma_backtest") - pl.col("slow_ma_live")).abs() < 0.01)
        ),
    )
    .sort("i")
)

matches = int(pairs["match"].sum())
print(f"Matching signals: {matches}/{len(backtest_signals)}")
if matches != len(backtest_signals):
    print("\nFirst three mismatches:")
    print(pairs.filter(~pl.col("match")).head(3))

# Side-by-side comparison frame for the reader (engine-tagged rows).
comparison.head(10)

# %% [markdown]
# **Finding:** The parity check reduces the framework claim to a falsifiable test: either the signals match
# or they do not.
#
# **Trading implication:** Unified frameworks are only valuable if parity is demonstrated, not assumed from
# shared class names or similar APIs.
#
# %% [markdown]
# ## 8. Signal Log Details
#
# The detailed signal logs give readers a human-readable audit trail after the high-level parity check. The
# exact timestamps and moving averages are what make discrepancies debuggable.


# %%
def _signals_to_frame(signals: list[dict]) -> pl.DataFrame:
    """Render the signal log as a polars frame for side-by-side display."""
    rows = []
    for sig in signals:
        ts = sig["timestamp"]
        ts_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
        rows.append(
            {
                "date": ts_str,
                "signal": sig["signal"],
                "price": round(sig["price"], 2),
                "fast_ma": round(sig["fast_ma"], 2),
                "slow_ma": round(sig["slow_ma"], 2),
            }
        )
    return pl.DataFrame(rows)


# %%
backtest_log = _signals_to_frame(backtest_signals).head(5)
backtest_log

# %%
live_log = _signals_to_frame(live_signals).head(5)
live_log

# %% [markdown]
# **Finding:** The side-by-side signal logs make any mismatch inspectable at the timestamp level.
#
# **Trading implication:** Detailed logs are what let a team fix parity breaks quickly instead of arguing
# from aggregate summaries after a deployment regression.
#
# %% [markdown]
# ## Summary
#
# This notebook demonstrated the core value proposition of the unified framework:
#
# | Mode | Engine | Strategy | Result |
# |------|--------|----------|--------|
# | Backtest | ml4t.backtest.Engine | DualMAStrategy | [OK] Signals logged |
# | Live | ml4t.live.LiveEngine | **Same** DualMAStrategy | [OK] Identical signals |
#
# **Key Takeaways**:
#
# 1. **One Strategy Class**: The `DualMAStrategy` class is used unchanged in both modes
# 2. **Same `on_data()` Signature**: `timestamp`, `data`, `context`, `broker` - identical
# 3. **Signal Parity on This Tape**: Backtest and live signals matched exactly on the
#    output-derived event sequence printed above. The demo does not cover execution latency,
#    partial fills, or slippage - those are tested in NB07 (state
#    machine), NB08 (full-pipeline parity), and NB12 (basket rebalance).
# 4. **Scope of the Claim**: This notebook demonstrates that engine swap alone does not
#    change strategy logic on the same historical daily ETF bars. The same-bar fills are a
#    parity oracle, not a tradable performance estimate; broker-side divergence is
#    out of scope here.
#
# **Next Steps**:
# - `03_ib_paper_trading_demo.py`: Connect to Interactive Brokers for real paper trading
# - `10_safety_risk_demo.py`: Explore SafeBroker's 8 layers of protection
# - `08_pipeline_verification.py`: Systematic verification methodology

# %%
parity = "perfect" if matches == len(backtest_signals) == len(live_signals) else "partial"
print(f"Backtest final value: ${results['final_value']:,.2f}")
print(f"Signal parity:        {parity} ({matches}/{len(backtest_signals)})")

# %% [markdown]
# **Next**: Use `08_pipeline_verification.py` to formalize the same comparison as a regression test, then add
# `SafeBroker` controls to the live path before moving toward paper trading.
