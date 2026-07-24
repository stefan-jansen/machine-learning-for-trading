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
# # Alpaca Paper Trading Demo
#
# **Docker image**: `ml4t`
#
# **Chapter**: 25 - Live Trading Systems
# **Section**: 25.3 (Alpaca Integration)
# **Learning Outcome**: LO2 - Connect ml4t-backtest strategies to live brokers
#
# This notebook demonstrates:
# 1. Connecting to Alpaca paper trading account
# 2. Querying account information and positions
# 3. Real-time data feed with bar/quote/trade streaming
# 4. Safe order submission in shadow mode
# 5. Strategy execution with ETF momentum signals
#
# **Learning Objectives**
# - Verify the environment, SDK, and account state before connecting a strategy to a live broker.
# - See how the same backtest strategy is wrapped with shadow-mode risk controls for production use.
# - Compare the optional live-feed wiring with the default offline simulation path.
#
# **Prerequisites**:
# - Alpaca account with API keys (paper trading enabled)
# - Environment variables: ALPACA_API_KEY, ALPACA_SECRET_KEY
# - Familiarity with the ETF case study strategy used earlier in the book
#
# **Why Alpaca alongside IB**: see §25.3 for the broker-comparison narrative.
#
# **Data Contract**:
# - **Input**: Deterministic simulated bars by default; Alpaca bars after explicit opt-in
# - **Output**: Virtual portfolio state, signals, order logs

# %%
"""Connect ml4t strategies to Alpaca with shadow-mode risk controls."""

import asyncio
import logging
import os
import warnings
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from async_utils import run_async
from ml4t.backtest import OrderSide, Strategy

from utils.paths import display_path, get_output_dir
from utils.reproducibility import set_global_seeds

# alpaca-py is an optional broker SDK; the simulated path runs without it. The
# try/except is the only such optional import in the notebook.
HAS_ALPACA_SDK = False
try:
    import alpaca  # noqa: F401
    from alpaca.trading.client import TradingClient
    from ml4t.live import AlpacaBroker, AlpacaDataFeed, LiveEngine, LiveRiskConfig
    from ml4t.live.safety import SafeBroker

    HAS_ALPACA_SDK = True
except ImportError:
    pass

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("alpaca").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

if HAS_ALPACA_SDK:
    print("[OK] ml4t.live Alpaca components imported")
else:
    print("Alpaca SDK not installed (uv add alpaca-py); running simulation only")

# %% tags=["parameters"]
DEMO_DURATION_SECONDS = 60
MAX_SYMBOLS = 0
SIMULATION_STEPS = 10
LIVE_FEED = 0  # explicit opt-in; default execution is offline and paper-safe
SEED = 42

# %%
# The Alpaca WebSocket loop is incompatible with papermill's nest_asyncio:
# `asyncio.wait_for` cannot reliably cancel the inner streaming task, so a
# headless run hangs past DEMO_DURATION_SECONDS. Detect papermill via its
# injected env var and fall back to the simulated path. Interactive Jupyter is
# unaffected.
if os.environ.get("ML4T_HEADLESS_PAPERMILL") == "1":
    LIVE_FEED = 0
    print("Headless papermill detected: LIVE_FEED disabled, simulated path will run")

set_global_seeds(SEED)

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
PAPER_TRADING = True

ALL_SYMBOLS = ["SPY", "QQQ", "IWM"]
SYMBOLS = ALL_SYMBOLS[:MAX_SYMBOLS] if MAX_SYMBOLS > 0 else ALL_SYMBOLS.copy()

# %% [markdown]
# ## 1. Credential Verification
#
# Alpaca requires API keys for authentication. For paper trading:
# - Sign up at https://alpaca.markets
# - Generate API keys in the dashboard
# - Set environment variables (never hardcode!)


# %%
def verify_credentials():
    """Check that API credentials and SDK are available."""
    if not HAS_ALPACA_SDK:
        print("\n" + "=" * 60)
        print("ALPACA SDK NOT INSTALLED")
        print("=" * 60)
        print("\nTo connect to Alpaca, install the SDK:")
        print("   uv add alpaca-py")
        print("\nRunning in DEMO MODE with simulated broker...")
        return False

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("\n" + "=" * 60)
        print("ALPACA CREDENTIALS NOT FOUND")
        print("=" * 60)
        print("\nTo run this notebook with real Alpaca connection:")
        print("1. Create an Alpaca account at https://alpaca.markets")
        print("2. Generate API keys in the dashboard")
        print("3. Set environment variables:")
        print("   export ALPACA_API_KEY='PKXXXXXXXX'")
        print("   export ALPACA_SECRET_KEY='xxxxxxxxxx'")
        print("\nRunning in DEMO MODE with simulated broker...")
        return False

    print("\n" + "=" * 60)
    print("ALPACA CREDENTIALS VERIFIED")
    print("=" * 60)
    print(f"Paper Trading: {'YES' if PAPER_TRADING else 'NO (LIVE!)'}")
    return True


HAS_CREDENTIALS = verify_credentials()

# %% [markdown]
# **Finding**: The credential gate explicitly distinguishes missing SDKs from missing secrets, which makes
# the execution mode observable before the notebook touches a broker connection.
#
# **Trading implication**: Live notebooks should never hide whether they are authenticated, shadowing, or
# fully simulated because that status determines the operational risk of every downstream action.
#
# %% [markdown]
# ## 2. Connect to Alpaca
#
# The AlpacaBroker class handles:
# - REST API for orders and account info
# - WebSocket for real-time order updates
# - Automatic position/order synchronization


# %%
def get_alpaca_account_snapshot():
    """Fetch an account snapshot without starting the streaming broker session."""
    if not HAS_CREDENTIALS:
        return None, None, None, None

    print("\n" + "=" * 60)
    print("CONNECTING TO ALPACA")
    print("=" * 60)

    trading_client = TradingClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=PAPER_TRADING,
    )
    account = trading_client.get_account()
    raw_positions = trading_client.get_all_positions()

    broker = AlpacaBroker(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=PAPER_TRADING,
    )

    print("\nConnected to Alpaca")
    print(f"   Paper Trading: {'YES' if PAPER_TRADING else 'NO'}")

    nlv = float(account.equity)
    cash = float(account.cash)

    print("\nACCOUNT READINESS")
    print("   Account values received: [OK]")

    # Get positions
    positions = {
        position.symbol: {
            "quantity": float(position.qty),
            "entry_price": float(position.avg_entry_price),
            "current_price": float(position.current_price or position.avg_entry_price),
        }
        for position in raw_positions
    }
    print(f"   Open positions received: {len(positions)}")

    return broker, nlv, cash, positions


# Run connection
if LIVE_FEED and HAS_CREDENTIALS and HAS_ALPACA_SDK:
    try:
        broker, nlv, cash, positions = get_alpaca_account_snapshot()
    except Exception as e:
        print(f"\nConnection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Are your API keys correct?")
        print("2. Is your account enabled for paper trading?")
        print("3. Check https://status.alpaca.markets for outages")
        raise RuntimeError("Alpaca paper session unavailable") from e
else:
    broker = None
    print("Offline mode selected; no Alpaca account request was made.")

# %% [markdown]
# **Finding**: The account summary confirms that the broker adapter exposes the same cash, equity, and
# position state the strategy will rely on later in the session.
#
# **Trading implication**: A live strategy should always prove that its broker snapshot is sane before it
# starts listening to market data; otherwise even correct signals can be routed with stale inventory.
#
# %% [markdown]
# ## 3. ETF Momentum Strategy
#
# This strategy is **identical** to what we use in backtesting.
# Uses the ETF case study with SPY, QQQ, IWM.


# %%
class ETFMomentumStrategy(Strategy):
    """ETF momentum strategy for live trading.

    Tracks 5-day momentum across ETFs and generates signals
    when momentum crosses thresholds.

    This is the SAME code used in backtest - zero changes for live!
    """

    def __init__(self, lookback: int = 5, threshold: float = 0.02, position_size: int = 10):
        self.lookback = lookback
        self.threshold = threshold
        self.position_size = position_size
        self.prices: dict[str, list[float]] = {}
        self.signals: list[dict] = []

    def on_start(self, broker):
        """Called when engine starts."""
        logger.info(f"Strategy started: ETFMomentum(lookback={self.lookback})")
        for symbol in SYMBOLS:
            self.prices[symbol] = []

    def on_data(self, timestamp: datetime, data: dict, context: dict, broker):
        """Called for each bar.

        Args:
            timestamp: Bar timestamp
            data: {symbol: {'open', 'high', 'low', 'close', 'volume'}}
            context: Additional metadata (vwap, trade_count, etc.)
            broker: Broker instance (sync interface)
        """
        for symbol, bar in data.items():
            if symbol not in self.prices:
                self.prices[symbol] = []

            close = bar["close"]
            self.prices[symbol].append(close)

            # Calculate momentum
            if len(self.prices[symbol]) > self.lookback:
                old_price = self.prices[symbol][-self.lookback - 1]
                momentum = (close - old_price) / old_price

                # Get current position
                position = broker.get_position(symbol)
                has_position = position is not None and position.quantity > 0

                # Generate signals
                if momentum > self.threshold and not has_position:
                    signal = {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "action": "BUY",
                        "momentum": momentum,
                        "price": close,
                    }
                    self.signals.append(signal)
                    logger.info(f"BUY {symbol}: Momentum {momentum:.2%} > {self.threshold:.2%}")
                    broker.submit_order(symbol, self.position_size, side=OrderSide.BUY)

                elif momentum < -self.threshold and has_position:
                    signal = {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "action": "SELL",
                        "momentum": momentum,
                        "price": close,
                    }
                    self.signals.append(signal)
                    logger.info(f"SELL {symbol}: Momentum {momentum:.2%} < -{self.threshold:.2%}")
                    broker.submit_order(symbol, self.position_size, side=OrderSide.SELL)

    def on_end(self, broker):
        """Called when engine stops."""
        logger.info(f"Strategy ended. Signals generated: {len(self.signals)}")


# %% [markdown]
# ### Structured Signal and Order Log
#
# Both the live and simulated paths accumulate signals and order intentions into a Polars frame so reviewers
# can inspect what the strategy decided at every step, in the same shape, regardless of whether the broker is
# Alpaca or the in-notebook mock.


# %%
def signals_to_frame(signals: list[dict]) -> pl.DataFrame:
    """Render the strategy's signal accumulator as a Polars frame for display."""
    if not signals:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "symbol": pl.String,
                "action": pl.String,
                "momentum": pl.Float64,
                "price": pl.Float64,
            }
        )
    return pl.DataFrame(signals)


# %% [markdown]
# ## 4. Safe Broker Configuration
#
# Before going live, we wrap the broker with `SafeBroker` which provides:
# - Shadow mode (virtual orders only)
# - Position limits
# - Order rate limiting
# - Kill switch


# %%
def create_safe_broker(underlying_broker):
    """Create SafeBroker with risk configuration."""
    risk_state_path = get_output_dir(25, "alpaca_paper_demo") / "risk_state.json"
    risk_config = LiveRiskConfig(
        shadow_mode=True,  # CRITICAL: Virtual orders only!
        max_position_value=50_000.0,
        max_order_value=10_000.0,
        max_orders_per_minute=10,
        dedup_window_seconds=0.0,  # Disable for demo
        state_file=str(risk_state_path),
    )

    safe_broker = SafeBroker(underlying_broker, risk_config)

    print("\n" + "=" * 60)
    print("RISK CONFIGURATION (SHADOW MODE)")
    print("=" * 60)
    print("   Shadow Mode: ENABLED (no real orders)")
    print(f"   Max Position Value: ${risk_config.max_position_value:,.0f}")
    print(f"   Max Order Value: ${risk_config.max_order_value:,.0f}")
    print(f"   Rate Limit: {risk_config.max_orders_per_minute}/minute")
    print(f"   Risk State: {display_path(risk_state_path)}")

    return safe_broker, risk_config


# %% [markdown]
# **Finding**: The risk-configuration printout makes shadow mode and exposure limits visible before the live
# feed starts emitting data.
#
# **Trading implication**: Broker wrappers should surface their active limits explicitly because a live
# rollout is only as safe as the controls that are actually enabled at runtime.
#
# %% [markdown]
# ## 5. Real-Time Data Feed
#
# AlpacaDataFeed subscribes to real-time market data:
# - **bars**: OHLCV aggregates (1-minute default)
# - **quotes**: Bid/ask with sizes
# - **trades**: Individual trades
#
# Data sources:
# - **IEX**: Free, 15-min delayed for some symbols
# - **SIP**: Premium, real-time from all exchanges


# %% [markdown]
# ### Simulated Path: Flat Dict Portfolio
#
# When credentials or the SDK are missing, the demo runs against a tiny in-notebook broker. The portfolio is a
# flat dict (`{"cash": float, "positions": {symbol: {qty, entry_price}}}`) rather than nested dataclasses so
# the simulated state can be read directly into a Polars frame for display alongside the live path.


# %%
class MockBroker:
    """Minimal sync broker for the no-credential simulation path.

    The portfolio is a flat dict to keep the simulation state inspectable; it is
    intentionally not a SafeBroker wrapper because shadow-mode requires fresh
    quotes that the simulation does not produce.
    """

    REF_PRICES = {"SPY": 600.0, "QQQ": 520.0, "IWM": 225.0}

    def __init__(self, initial_cash: float = 100_000.0):
        self.portfolio = {"cash": initial_cash, "positions": {}}
        self.order_log: list[dict] = []
        self.current_prices = dict(self.REF_PRICES)
        self.current_timestamp = datetime(2025, 1, 2, tzinfo=UTC)

    def update_market(self, timestamp: datetime, prices: dict[str, float]) -> None:
        """Update the simulated market used for subsequent fills."""
        self.current_timestamp = timestamp
        self.current_prices.update(prices)

    def get_position(self, symbol: str):
        pos = self.portfolio["positions"].get(symbol)
        if pos is None:
            return None

        class _PosView:
            quantity = pos["quantity"]

        return _PosView()

    def submit_order(self, asset: str, quantity: int, side=None, **kwargs) -> dict:
        price = self.current_prices[asset]
        if isinstance(side, OrderSide):
            side_name = side.value.upper()
        elif side is None:
            side_name = "BUY"
        else:
            side_name = str(getattr(side, "value", side)).upper()
        status = "rejected"
        if side is None or side == OrderSide.BUY:
            cost = quantity * price
            if cost <= self.portfolio["cash"]:
                self.portfolio["cash"] -= cost
                pos = self.portfolio["positions"].get(asset)
                if pos is None:
                    self.portfolio["positions"][asset] = {
                        "quantity": quantity,
                        "entry_price": price,
                    }
                else:
                    total_qty = pos["quantity"] + quantity
                    avg = (pos["quantity"] * pos["entry_price"] + quantity * price) / total_qty
                    self.portfolio["positions"][asset] = {
                        "quantity": total_qty,
                        "entry_price": avg,
                    }
                status = "filled"
        elif side == OrderSide.SELL:
            pos = self.portfolio["positions"].get(asset)
            if pos is not None and pos["quantity"] >= quantity:
                self.portfolio["cash"] += quantity * price
                remaining = pos["quantity"] - quantity
                if remaining == 0:
                    del self.portfolio["positions"][asset]
                else:
                    self.portfolio["positions"][asset] = {
                        "quantity": remaining,
                        "entry_price": pos["entry_price"],
                    }
                status = "filled"
        else:
            status = "unsupported"
        order = {
            "order_id": f"SIM-{len(self.order_log) + 1}",
            "timestamp": self.current_timestamp,
            "symbol": asset,
            "side": side_name,
            "quantity": quantity,
            "price": price,
            "status": status,
        }
        self.order_log.append(order)
        return order


# %% [markdown]
# ### Live Path: Engine Wiring
#
# The live path constructs `AlpacaDataFeed > SafeBroker > LiveEngine`. Each helper is small enough to inspect
# at a glance; `run_engine_for_duration` is the only async piece, and `display_engine_results` is purely
# read-only post-processing.


# %%
def create_alpaca_engine(strategy):
    """Build the AlpacaDataFeed, SafeBroker, and LiveEngine wiring."""
    safe_broker, _ = create_safe_broker(broker)
    feed = AlpacaDataFeed(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        symbols=SYMBOLS,
        data_type="bars",
        feed="iex",
    )
    engine = LiveEngine(strategy=strategy, broker=safe_broker, feed=feed)
    # Alpaca SDK retries aggressively under nest_asyncio; quiet the retry logs.
    for name in [
        "alpaca",
        "alpaca.data",
        "alpaca.data.live",
        "alpaca.data.live.websocket",
        "alpaca.trading.stream",
        "websockets",
    ]:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    return engine, safe_broker, feed


# %%
async def run_engine_for_duration(engine, duration_s: int):
    """Connect the engine and let it stream for at most `duration_s` seconds."""
    await asyncio.wait_for(engine.connect(), timeout=10)
    try:
        await asyncio.wait_for(engine.run(), timeout=duration_s)
    except TimeoutError:
        print(f"Demo duration ({duration_s}s) reached")


# %%
def display_engine_results(strategy, safe_broker, feed, engine):
    """Print engine stats and render the strategy's signal log as a Polars frame."""
    print("Engine stats:", {k: engine.stats[k] for k in list(engine.stats)[:6]})
    print("Feed stats: ", {k: feed.stats[k] for k in list(feed.stats)[:6]})

    vp = safe_broker._virtual_portfolio
    print(f"Virtual Portfolio cash: ${vp.cash:,.2f}")
    for symbol, pos in vp.positions.items():
        value = pos.quantity * (pos.current_price or pos.entry_price)
        print(f"   {symbol}: {pos.quantity} shares @ ${pos.entry_price:.2f} = ${value:,.2f}")

    print(f"\nSignals: {len(strategy.signals)}")
    return signals_to_frame(strategy.signals)


# %%
async def run_live_demo_with_feed():
    """Run the explicitly selected Alpaca or offline simulation path."""
    if not LIVE_FEED:
        print(f"LIVE_FEED={LIVE_FEED}: running the offline simulation")
        return await run_simulated_demo()
    if not HAS_CREDENTIALS or broker is None:
        raise RuntimeError("LIVE_FEED requires the Alpaca SDK and paper credentials")

    print("LIVE TRADING DEMO (Shadow Mode)")
    strategy = ETFMomentumStrategy(lookback=5, threshold=0.02, position_size=10)
    engine, safe_broker, feed = create_alpaca_engine(strategy)
    print(f"Starting live engine for {DEMO_DURATION_SECONDS}s; watching {', '.join(SYMBOLS)}")

    try:
        await run_engine_for_duration(engine, DEMO_DURATION_SECONDS)
    finally:
        feed.stop()
    return display_engine_results(strategy, safe_broker, feed, engine), pl.DataFrame()


# %%
async def run_simulated_demo() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the strategy against the flat-dict MockBroker; return the signal log."""
    print("SIMULATED DEMO (no live Alpaca feed)")
    strategy = ETFMomentumStrategy(lookback=3, threshold=0.01, position_size=10)
    mock_broker = MockBroker()
    strategy.on_start(mock_broker)

    set_global_seeds(SEED)
    base_prices = dict(MockBroker.REF_PRICES)
    start = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    for step in range(SIMULATION_STEPS):
        timestamp = start + timedelta(minutes=step)
        data = {}
        for symbol in SYMBOLS:
            base_prices[symbol] *= 1 + np.random.normal(0.001, 0.01)
            data[symbol] = {
                "open": base_prices[symbol] * 0.999,
                "high": base_prices[symbol] * 1.002,
                "low": base_prices[symbol] * 0.998,
                "close": base_prices[symbol],
                "volume": int(np.random.randint(100000, 1000000)),
            }
        mock_broker.update_market(timestamp, {symbol: bar["close"] for symbol, bar in data.items()})
        strategy.on_data(timestamp, data, {}, mock_broker)
    strategy.on_end(mock_broker)

    print(f"Simulated cash: ${mock_broker.portfolio['cash']:,.2f}")
    for symbol, pos in mock_broker.portfolio["positions"].items():
        value = pos["quantity"] * pos["entry_price"]
        print(f"   {symbol}: {pos['quantity']} shares @ ${pos['entry_price']:.2f} = ${value:,.2f}")
    print(f"Signals: {len(strategy.signals)}; Orders: {len(mock_broker.order_log)}")

    signal_frame = signals_to_frame(strategy.signals)
    order_frame = (
        pl.DataFrame(mock_broker.order_log)
        if mock_broker.order_log
        else pl.DataFrame(
            schema={
                "order_id": pl.String,
                "timestamp": pl.Datetime,
                "symbol": pl.String,
                "side": pl.String,
                "quantity": pl.Int64,
                "price": pl.Float64,
                "status": pl.String,
            }
        )
    )
    return signal_frame, order_frame


# %%
# Run the demo
demo_signal_log, demo_order_log = run_async(run_live_demo_with_feed())
demo_signal_log

# %%
demo_order_log

# %% [markdown]
# **Finding**: The selected execution mode is explicit. The offline path exercises the same strategy interface
# with an inspectable broker, while the opt-in live path adds `SafeBroker` and Alpaca transport controls.
#
# **Trading implication**: Keeping the live and simulated paths structurally aligned makes it easier to
# detect true broker-side issues instead of debugging differences introduced by the demo environment.
#
# %% [markdown]
# ## 6. Order Type Demonstrations
#
# Alpaca supports various order types:
# - **MARKET**: Execute immediately at best available price
# - **LIMIT**: Execute at specified price or better
# - **STOP**: Trigger market order when price reaches stop
# - **STOP_LIMIT**: Trigger limit order when price reaches stop


# %%
async def demonstrate_order_types():
    """Demonstrate different order types (shadow mode)."""
    if not HAS_CREDENTIALS or broker is None:
        print("\nSkipping order demo - no credentials")
        return
    if not LIVE_FEED:
        # SafeBroker requires fresh market data for staleness checks; without the
        # live WebSocket feed there are no recent quotes, so submit_order_async
        # raises RiskLimitError. Skip the demo under headless papermill.
        print(f"\nLIVE_FEED={LIVE_FEED}: skipping order-type demo (needs live market data)")
        return

    print("\n" + "=" * 60)
    print("ORDER TYPE DEMONSTRATIONS (Shadow Mode)")
    print("=" * 60)

    safe_broker, _ = create_safe_broker(broker)

    from ml4t.backtest.types import OrderType

    # Market order
    print("\n1. MARKET ORDER")
    order = await safe_broker.submit_order_async("SPY", 10, side=OrderSide.BUY)
    print(f"   Order ID: {order.order_id}")
    print("   Type: MARKET")
    print(f"   Status: {order.status.value}")

    # Limit order
    print("\n2. LIMIT ORDER")
    order = await safe_broker.submit_order_async(
        "QQQ", 5, side=OrderSide.BUY, order_type=OrderType.LIMIT, limit_price=500.00
    )
    print(f"   Order ID: {order.order_id}")
    print("   Type: LIMIT @ $500.00")
    print(f"   Status: {order.status.value}")

    # Stop order
    print("\n3. STOP ORDER")
    order = await safe_broker.submit_order_async(
        "IWM", 10, side=OrderSide.SELL, order_type=OrderType.STOP, stop_price=220.00
    )
    print(f"   Order ID: {order.order_id}")
    print("   Type: STOP @ $220.00")
    print(f"   Status: {order.status.value}")

    # Show virtual portfolio
    vp = safe_broker._virtual_portfolio
    print("\nVirtual Portfolio After Orders:")
    print(f"   Cash: ${vp.cash:,.2f}")
    for symbol, pos in vp.positions.items():
        print(f"   {symbol}: {pos.quantity} shares")


run_async(demonstrate_order_types())

# %% [markdown]
# **Finding**: The optional live path defines shadow-mode examples for each order type. The default offline
# run skips those submissions explicitly because it has no Alpaca client, live feed, or `SafeBroker`.
#
# **Trading implication**: After credentials and a live paper feed are available, shadow-mode submission is
# the intermediate stage that tests routing, validation, and guardrails without creating exposure.
#
# %% [markdown]
# ## 7. Clean Shutdown
#
# The session should end with an explicit disconnect so the next run starts from a known broker state instead
# of inheriting stale subscriptions or session assumptions.


# %%
# Disconnect from Alpaca
if broker is not None:
    run_async(broker.disconnect())
    print("\nDisconnected from Alpaca")

# %% [markdown]
# ## Summary
#
# This notebook defines the complete optional Alpaca integration workflow and executes the deterministic
# offline strategy/broker path by default. With `LIVE_FEED=True`, credentials, and the SDK available, it can
# additionally exercise:
#
# 1. **Authentication**: API key/secret via environment variables
# 2. **Connection**: Paper trading account access
# 3. **Account Info**: Query equity, cash, positions
# 4. **Real-Time Feed**: Subscribe to bars/quotes/trades
# 5. **Safe Trading**: Use SafeBroker in shadow mode
# 6. **Order Types**: Market, limit, stop orders
#
# ### Alpaca vs IB Comparison
#
# | Feature | Alpaca | Interactive Brokers |
# |---------|--------|---------------------|
# | Minimum Balance | None | None (but higher margin reqs) |
# | Commissions | Free | $0-1 per trade |
# | Real-time Data | Free (IEX) | Paid subscription |
# | API Complexity | Simple REST | Complex socket protocol |
# | Crypto | Yes (24/7) | Limited |
# | Paper Trading | Yes | Yes |
#
# ### Next Steps
#
# 1. Run in shadow mode for 1-2 weeks
# 2. Verify signals match backtest expectations
# 3. Enable paper trading (`shadow_mode=False`)
# 4. Monitor for 2-4 weeks on paper
# 5. Gradually transition to live with small positions
#
# ### Crypto Trading
#
# See `05_alpaca_crypto_live_demo.py` for 24/7 crypto trading demonstration.

# %%
print("\n" + "=" * 60)
print("ALPACA PAPER TRADING DEMO COMPLETE")
print("=" * 60)
print(f"Symbols: {', '.join(SYMBOLS)}")
print(f"Paper Trading: {'YES' if PAPER_TRADING else 'NO'}")
shadow_mode_state = "ENABLED" if LIVE_FEED else "NOT ACTIVE (offline simulation)"
print(f"Shadow Mode: {shadow_mode_state}")
execution_mode = "Alpaca shadow feed" if LIVE_FEED else "offline simulation"
print(f"Execution Mode: {execution_mode}")
print("The same ETFMomentumStrategy interface drives the selected execution path.")

# %% [markdown]
# ## Key Takeaways
#
# **Finding**: Alpaca provides a compact optional live-trading stack, while the default run demonstrates
# the shared strategy interface and offline broker without claiming that live controls executed.
#
# **Trading implication**: The portability of the strategy object matters more than the specific broker API.
# If strategy logic survives the move from backtest to shadow mode unchanged, the remaining work is mostly
# about operational controls.
#
# **Next**: Compare this flow with `03_ib_paper_trading_demo.py` and then use
# `05_alpaca_crypto_live_demo.py` for the 24/7 crypto variant of the same deployment pattern.
