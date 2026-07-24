# ---
# jupyter:
#   jupytext:
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
# # Interactive Brokers Paper Trading Demo
#
# **Chapter**: 25 - Live Trading Systems
# **Section**: 25.2 (Interactive Brokers Integration)
# **Learning Outcome**: LO2 - Connect ml4t-backtest strategies to live brokers
#
# **Docker image**: `ml4t` (requires IB TWS/Gateway running on host port 7497)
#
# This notebook demonstrates:
# 1. Connecting to IB TWS paper trading account
# 2. Querying account information and positions
# 3. Requesting historical data for strategy warm-up
# 4. Real-time data feed with tick aggregation
# 5. Safe order submission in shadow mode
#
# **Learning Objectives**:
# - Validate IB connectivity, account state, and warm-up data before a live engine starts.
# - Reuse the same backtest strategy inside a shadow-mode live workflow.
# - Exercise the loud-fail setup checklist: the notebook does not synthesize live behaviour from history when TWS/Gateway is unreachable or the market is closed.
#
# **Prerequisites**:
# - IB TWS or Gateway running with API enabled
# - Paper trading port: 7497 (TWS) or 4002 (Gateway)
# - `IB_ACCOUNT` environment variable set to your IB paper account ID (e.g. `DU1234567`); the notebook passes `None` to `IBBroker` if unset, which lets the broker pick the default account on the session.
# - Familiarity with the ETF momentum example used throughout the live-trading chapter

# %%
"""Connect ml4t strategies to IB with shadow-mode risk controls."""

import os

# %% tags=["parameters"]
# Production defaults. Papermill may inject overrides for CI.
IB_HOST = "127.0.0.1"
IB_PORT = 7497  # Paper trading port
CLIENT_ID = 10  # Use unique ID per notebook
ACCOUNT = os.environ.get(
    "IB_ACCOUNT"
)  # set IB_ACCOUNT=DU... before running; None picks the session default

# IB market-data type: None leaves TWS at its configured default (use this if
# the paper account has live Level 1 subscriptions). Paper accounts without
# market-data subscriptions reject MARKET orders with "No market data
# available..."; set this to 3 (delayed) to fall back to delayed quotes.
# 1=real-time, 2=frozen, 3=delayed, 4=delayed-frozen.
MARKET_DATA_TYPE: int | None = 3

# Test settings
SYMBOLS = ["SPY", "QQQ", "IWM"]  # ETFs to monitor
WARMUP_DAYS = 10  # Must provide at least lookback + 1 daily closes
LIVE_DURATION_SECONDS = 75

# %%
import asyncio
import logging
import warnings
from datetime import datetime

os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")

from async_utils import run_async

from utils.paths import display_path, get_output_dir

# %%
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce noise from ib_async
logging.getLogger("ib_async").setLevel(logging.WARNING)
logging.getLogger("ml4t.live.brokers.ib").setLevel(logging.WARNING)

# %%
# Import ml4t.live components
from ml4t.backtest import OrderSide, Strategy
from ml4t.live import LiveEngine, LiveRiskConfig
from ml4t.live.brokers.ib import IBBroker
from ml4t.live.feeds import BarAggregator, IBDataFeed
from ml4t.live.safety import SafeBroker

print("[OK] ml4t.live components imported successfully")

# %% [markdown]
# **Finding**: The import check confirms that the live-engine, broker, and feed abstractions are available
# before any Interactive Brokers connection is attempted. That separates environment setup failures from
# broker-session failures.
#
# **Trading implication**: Live deployment debugging is faster when infrastructure imports and network
# connectivity are validated as distinct gates instead of being collapsed into one opaque error.
#
# %% [markdown]
# ## 1. Connect to Interactive Brokers
#
# TWS/Gateway must be running with:
# - API access enabled (Edit > Global Configuration > API > Settings)
# - Socket port set (default: 7497 for paper)
# - "Enable ActiveX and Socket Clients" checked

# %%
# Create IBBroker instance
broker = IBBroker(
    host=IB_HOST,
    port=IB_PORT,
    client_id=CLIENT_ID,
    account=ACCOUNT,
    market_data_type=MARKET_DATA_TYPE,
)

print(f"IBBroker configured for {IB_HOST}:{IB_PORT}")
print(f"Account selection: {'configured paper account' if ACCOUNT else 'session default'}")
print(f"Client ID: {CLIENT_ID}")

# %% [markdown]
# **Finding**: The broker configuration printout turns host, port, account, and client ID into visible
# runtime state rather than hidden constants. That is the minimum context needed before opening a session.
#
# **Trading implication**: Many live-trading failures come from account or session mismatches, so notebooks
# should surface connection parameters before they attempt authentication.
#


# %%
async def connect_to_ib():
    """Connect to IB and show account summary."""
    print("\n" + "=" * 60)
    print("CONNECTING TO INTERACTIVE BROKERS")
    print("=" * 60)

    await broker.connect()

    account = str(broker._account or "")
    if not account.startswith("DU"):
        await broker.disconnect()
        raise RuntimeError("Refusing to continue: connected IB account is not a paper account")

    print("\n[OK] Connected to IB")
    print("   Paper account identity: [OK]")
    # Get account values
    nlv = await broker.get_account_value_async()
    cash = await broker.get_cash_async()

    print("\nACCOUNT READINESS")
    print("   Account values received: [OK]")

    # Get positions
    positions = await broker.get_positions_async()
    print(f"   Open positions received: {len(positions)}")

    return nlv, cash, positions


# %%
# Run connection
try:
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"nest_asyncio")
    nlv, cash, positions = run_async(connect_to_ib())
except Exception as exc:
    print()
    print("=" * 60)
    print("ERROR: IB paper session unreachable")
    print("=" * 60)
    print(f"Could not connect to {IB_HOST}:{IB_PORT}")
    print(f"Underlying error: {type(exc).__name__}: {exc}")
    print()
    print("Setup checklist:")
    print("  1. Start TWS or IB Gateway and log into a paper account.")
    print("  2. Edit -> Global Configuration -> API -> Settings:")
    print("     - 'Enable ActiveX and Socket Clients' must be checked.")
    print(f"     - Socket port must be {IB_PORT} (TWS paper=7497, Gateway paper=4002).")
    print("     - 127.0.0.1 must be in 'Trusted IPs', or 'Read-Only API' unchecked.")
    print("  3. Confirm no other client is using this client_id.")
    print()
    print("Re-run this notebook once TWS is reachable.")
    raise RuntimeError("IB paper session unreachable") from exc

# %% [markdown]
# **Finding**: The connection block fails loudly with an actionable checklist when TWS is unreachable
# rather than silently substituting placeholder data. That prevents the reader from confusing transport
# availability with strategy correctness.
#
# **Trading implication**: Broker health checks belong ahead of any signal loop because a disconnected
# broker is an operational state, not just another data-point in the strategy.
#
# %% [markdown]
# ## 2. Request Historical Data
#
# Before running a strategy live, we need historical data to:
# - Calculate initial indicator values (moving averages, etc.)
# - Establish baseline for position sizing
# - Verify data quality
#
# IB provides historical bars via `reqHistoricalData`.

# %%
from ib_async import Stock


async def get_historical_data(symbol: str, days: int = 5) -> list[dict]:
    """Request historical daily bars from IB and return OHLCV records."""
    contract = Stock(symbol, "SMART", "USD")
    qualified = await broker.ib.qualifyContractsAsync(contract)
    if not qualified:
        raise RuntimeError(f"IB could not qualify the {symbol} contract")

    # `timeout=0` disables ib_async's internal `asyncio.wait_for` wrapper,
    # which fails under nest_asyncio on Python 3.14.
    bars = await broker.ib.reqHistoricalDataAsync(
        contract,
        endDateTime="",
        durationStr=f"{days} D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        timeout=0,
    )

    result = []
    for bar in bars:
        result.append(
            {
                "timestamp": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )

    if not result:
        raise RuntimeError(f"IB returned no warm-up bars for {symbol}")
    return result


# %%
# Request historical data for our symbols
print("\n" + "=" * 60)
print("HISTORICAL DATA (Warm-up)")
print("=" * 60)

historical_data = {}
try:
    for symbol in SYMBOLS:
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"nest_asyncio")
        bars = run_async(get_historical_data(symbol, WARMUP_DAYS))
        if len(bars) < 6:
            raise RuntimeError(f"{symbol} returned {len(bars)} bars; at least 6 are required")
        historical_data[symbol] = bars
        latest = bars[-1]
        print(f"\n{symbol}: {len(bars)} days of data")
        print(
            f"   Latest: {latest['timestamp']} - Close: ${latest['close']:.2f}, Volume: {latest['volume']:,}"
        )
except Exception:
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"nest_asyncio")
    run_async(broker.disconnect())
    raise

assert set(historical_data) == set(SYMBOLS)

# %% [markdown]
# **Finding**: The warm-up section shows the exact inputs used to initialize indicators before the engine is
# allowed to trade. That is the bridge between research assumptions and live-state initialization.
#
# **Trading implication**: Live systems need a deterministic warm-up path; otherwise positions can be sized
# from incomplete indicators during the first minutes after reconnect or market open.
#
# %% [markdown]
# ## 3. Strategy Definition
#
# This strategy is **identical** to what we use in backtesting.
# The `on_data` signature works with both `ml4t.backtest.Engine` and `ml4t.live.LiveEngine`.


# %%
class MomentumStrategy(Strategy):
    """Five-day ETF momentum strategy shared by backtest and live engines."""

    def __init__(self, lookback: int = 5, threshold: float = 0.02):
        self.lookback = lookback
        self.threshold = threshold
        self.prices: dict[str, list[float]] = {}
        self.signals: list[dict] = []

    def on_start(self, broker):
        """Called when engine starts."""
        logger.info(f"Strategy started: Momentum({self.lookback}, {self.threshold})")
        for symbol, bars in historical_data.items():
            self.prices[symbol] = [bar["close"] for bar in bars]
            logger.info(f"  {symbol}: Loaded {len(self.prices[symbol])} historical prices")

    def on_data(self, timestamp: datetime, data: dict, context: dict, broker):
        """Update trailing momentum and route threshold crossings."""
        for symbol, bar in data.items():
            prices = self.prices.setdefault(symbol, [])
            close = bar["close"]
            prices.append(close)
            if len(prices) <= self.lookback:
                continue
            momentum = (close - prices[-self.lookback - 1]) / prices[-self.lookback - 1]
            position = broker.get_position(symbol)
            has_position = position is not None and position.quantity > 0
            side = None
            if momentum > self.threshold and not has_position:
                side = OrderSide.BUY
            elif momentum < -self.threshold and has_position:
                side = OrderSide.SELL
            if side is None:
                continue
            action = side.value.upper()
            self.signals.append(
                {"timestamp": timestamp, "symbol": symbol, "action": action, "momentum": momentum}
            )
            logger.info(f"{action} {symbol}: momentum {momentum:.2%}")
            broker.submit_order(symbol, 100, side=side)

    def on_end(self, broker):
        """Called when engine stops."""
        logger.info(f"Strategy ended. Signals generated: {len(self.signals)}")


# %% [markdown]
# ## 4. Safe Broker Configuration
#
# Before going live, we wrap the broker with `SafeBroker` which provides:
# - Shadow mode (virtual orders routed through `VirtualPortfolio`, never to IB)
# - Position and order value caps, rate limiting, kill switch
# - Persisted `RiskState` (daily-loss counter survives restarts)
# - Startup reconciliation: `safe_broker.connect()` diffs the persisted snapshot from the previous run
#   against the broker's current positions and pending orders


# %%
RISK_STATE_PATH = get_output_dir(25, "ib_paper_demo") / "risk_state.json"
risk_config = LiveRiskConfig(
    shadow_mode=True,  # CRITICAL: Virtual orders only!
    max_position_value=50_000.0,
    max_order_value=10_000.0,
    max_orders_per_minute=5,
    max_daily_loss=2_500.0,
    max_data_staleness_seconds=60,
    dedup_window_seconds=0.0,  # Disable for demo
    state_file=str(RISK_STATE_PATH),
)

safe_broker = SafeBroker(broker, risk_config)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"nest_asyncio")
run_async(safe_broker.connect())

print("\n" + "=" * 60)
print("RISK CONFIGURATION (SHADOW MODE)")
print("=" * 60)
print("   Shadow Mode: [OK] ENABLED (no real orders)")
print(f"   Max Position Value: ${risk_config.max_position_value:,.0f}")
print(f"   Max Order Value: ${risk_config.max_order_value:,.0f}")
print(f"   Max Daily Loss: ${risk_config.max_daily_loss:,.0f}")
print(f"   Max Data Staleness: {risk_config.max_data_staleness_seconds}s")
print(f"   Rate Limit: {risk_config.max_orders_per_minute}/minute")
print(f"   Risk State: {display_path(RISK_STATE_PATH)}")
report = safe_broker.reconciliation_report
print(f"   Startup Reconciliation: {'clean' if report and report['clean'] else 'review report'}")

# %% [markdown]
# ## 5. Real-Time Data Feed
#
# `IBDataFeed` subscribes to live tick data from IB; `BarAggregator` rolls those
# ticks into minute bars for the strategy. A live notebook is only meaningful
# while the market is open. If the run starts after the close, the
# notebook fails loudly rather than substituting historical bars (a backtest
# wearing a live disguise).


# %%
def _is_rth_now(zone: str = "America/New_York") -> bool:
    """Return True iff wall-clock now is inside US-equity regular trading hours."""
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo(zone))
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t


# %% [markdown]
# The feed, aggregator, and engine are assembled in one helper so the bounded
# run below focuses only on lifecycle and cleanup.


# %%
def build_live_stack() -> tuple[BarAggregator, LiveEngine]:
    """Build the IB feed, minute aggregator, and shadow execution engine."""
    ib_feed = IBDataFeed(ib=broker.ib, symbols=SYMBOLS, tick_throttle_ms=1000)
    feed = BarAggregator(source_feed=ib_feed, bar_size_minutes=1, assets=SYMBOLS)
    strategy = MomentumStrategy(lookback=5, threshold=0.02)
    engine = LiveEngine(strategy=strategy, broker=safe_broker, feed=feed)
    return feed, engine


# %% [markdown]
# A fixed-duration RTH run proves connectivity without leaving an unattended
# strategy loop behind.


# %%
async def run_live_demo(duration_seconds: int = 30):
    """Run strategy with live IB data for specified duration."""
    print("\n" + "=" * 60)
    print("LIVE TRADING DEMO (Shadow Mode)")
    print("=" * 60)

    if not _is_rth_now():
        print()
        print("ERROR: Market is closed. This notebook requires an open RTH session.")
        print("US equity RTH: 09:30-16:00 America/New_York, Mon-Fri.")
        print("Re-run during RTH; the notebook will not synthesize live behaviour from history.")
        raise RuntimeError("IB live-feed gate requires an open US-equity RTH session")

    feed, engine = build_live_stack()
    print(f"\nStarting live engine for {duration_seconds} seconds...")
    print("   Watching: " + ", ".join(SYMBOLS))

    engine_task: asyncio.Task | None = None
    try:
        await engine.connect()

        # Run for a fixed duration. `asyncio.wait_for` is unreliable under
        # nest_asyncio in notebook kernels, so we drive the timeout manually.
        engine_task = asyncio.create_task(engine.run())
        await asyncio.sleep(duration_seconds)
        print(f"\nDemo duration ({duration_seconds}s) reached")
    finally:
        if engine_task is not None and not engine_task.done():
            engine_task.cancel()
            try:
                await engine_task
            except asyncio.CancelledError:
                pass
        feed.stop()
        print("\nEngine Statistics:")
        for key, value in engine.stats.items():
            print(f"   {key}: {value}")


# The live workflow runs after the shadow-order routine is defined so one
# outer `finally` block can always release the IB session.

# %% [markdown]
# **Finding**: The notebook keeps the same strategy and risk configuration regardless of whether the live
# feed is active. Only the transport layer changes.
#
# **Trading implication**: Separating market connectivity from strategy logic is what makes a live stack
# testable; when the feed changes, the trading rules should not.
#
# %% [markdown]
# ## 6. Order Submission Demo
#
# Let's demonstrate order submission in shadow mode.
# Orders are tracked virtually but never sent to IB.


# %%
async def fetch_ib_snapshot(symbol: str) -> float | None:
    """Fetch one delayed top-of-book snapshot from IB for `symbol` and return mid price."""
    contract = Stock(symbol, "SMART", "USD")
    qualified = await broker.ib.qualifyContractsAsync(contract)
    if not qualified:
        return None
    ticker = broker.ib.reqMktData(qualified[0], "", snapshot=True, regulatorySnapshot=False)
    deadline = datetime.now().timestamp() + 5.0
    while datetime.now().timestamp() < deadline:
        bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else None
        ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else None
        last = float(ticker.last) if ticker.last and ticker.last > 0 else None
        close = float(ticker.close) if ticker.close and ticker.close > 0 else None
        if bid and ask:
            return (bid + ask) / 2
        if last:
            return last
        if close:
            return close
        await asyncio.sleep(0.2)
    return None


# %%
async def demonstrate_order_submission():
    """Show order submission in shadow mode using live IB snapshot quotes."""
    print("\n" + "=" * 60)
    print("ORDER SUBMISSION DEMO (Shadow Mode)")
    print("=" * 60)

    # Pull a delayed snapshot quote per symbol from IB and seed SafeBroker's
    # staleness cache with it. This is the operational discipline the chapter
    # is teaching: every order is preceded by a real, current observation -
    # never a research-time price, never an in-memory mock. Delayed data is
    # acceptable for paper-account demonstrations; live subscriptions are an
    # account-level concern, not a notebook-level one.
    print("\nFetching delayed snapshot quotes from IB...")
    snapshot_prices = {}
    for symbol in SYMBOLS:
        price = await fetch_ib_snapshot(symbol)
        if price is None:
            raise RuntimeError(f"No IB snapshot quote received for {symbol} within 5 seconds")
        safe_broker.record_market_snapshot(symbol, price)
        snapshot_prices[symbol] = price
        print(f"   {symbol}: ${price:,.2f}")
    assert set(snapshot_prices) == set(SYMBOLS)

    # Order quantities are sized to fit max_order_value=$10,000 at current
    # SPY/QQQ levels so the demo shows a successful virtual fill rather
    # than the cap rejection. SafeBroker still applies every other layer
    # (staleness, daily loss, kill switch) on each leg.
    orders = [("SPY", 10), ("QQQ", 12)]

    for symbol, qty in orders:
        print(f"\nSubmitting virtual BUY order: {qty} shares {symbol}")
        order = await safe_broker.submit_order_async(symbol, qty, side=OrderSide.BUY)
        print(f"   Order ID: {order.order_id}")
        print(f"   Status: {order.status.value}")
        print(f"   Side: {order.side.value}")
        print(f"   Quantity: {order.quantity}")

    vp = safe_broker._virtual_portfolio
    print("\nFinal Virtual Portfolio:")
    print(f"   Cash: ${vp.cash:,.2f}")
    for symbol, pos in vp.positions.items():
        value = pos.quantity * (pos.current_price or pos.entry_price)
        print(f"   {symbol}: {pos.quantity} shares @ ${pos.entry_price:.2f} = ${value:,.2f}")


# %% [markdown]
# One outer cleanup boundary covers both the live-feed demonstration and the
# virtual-order demonstration.


# %%
async def run_shadow_workflow() -> None:
    """Run the live-feed and shadow-order gates with guaranteed cleanup."""
    try:
        await run_live_demo(duration_seconds=LIVE_DURATION_SECONDS)
        await demonstrate_order_submission()
    finally:
        await safe_broker.disconnect()


warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"nest_asyncio")
run_async(run_shadow_workflow())

# %% [markdown]
# **Finding**: The order demo surfaces the shadow portfolio after each submission, so the reader can verify
# how intent becomes inventory without exposing capital.
#
# **Trading implication**: Shadow-mode order routing is where many production bugs first appear because the
# strategy, broker adapter, and risk guard all interact at once.
#
# %% [markdown]
# ## 7. Clean Shutdown
#
# Always disconnect cleanly to avoid connection issues on the next run. A disciplined shutdown sequence is
# part of live reliability because stale sessions and dangling subscriptions are operational bugs too.

# %%
print("\n[OK] Shadow workflow completed and disconnected from IB")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated the complete IB integration workflow:
#
# 1. **Connection**: Connect to TWS/Gateway paper trading
# 2. **Account Info**: Query NLV, cash, positions
# 3. **Historical Data**: Request bars for strategy warm-up
# 4. **Real-Time Feed**: Subscribe to tick data, aggregate to bars
# 5. **Safe Trading**: Use SafeBroker in shadow mode
# 6. **Order Submission**: Virtual orders tracked in VirtualPortfolio
#
# ### Key Takeaways
#
# - **Same Strategy class** works in backtest and live
# - **Shadow mode first** - always test before real trading
# - **SafeBroker** provides 8 layers of protection
# - **IBBroker** handles all IB-specific details
#
# ### Next Steps
#
# 1. Run in shadow mode for 1-2 weeks
# 2. Verify signals match backtest expectations
# 3. Enable paper trading (`shadow_mode=False`)
# 4. Monitor for 2-4 weeks on paper
# 5. Gradually transition to live with small positions

# %%
print("\n" + "=" * 60)
print("DEMO COMPLETE")
print("=" * 60)
print("Paper account readiness: verified")
print(f"Symbols Monitored: {', '.join(SYMBOLS)}")
print("Shadow Mode: ENABLED [OK]")
print("\nThe same MomentumStrategy that runs in backtest")
print("completed the IB shadow workflow without sending an order to the venue.")

# %% [markdown]
# ## Key Takeaways
#
# **Finding**: IB adds more connectivity and market-structure complexity than Alpaca, but the notebook still
# demonstrates that the execution architecture can remain strategy-agnostic.
#
# **Trading implication**: Once broker connectivity, warm-up, and shadow-mode controls are standardized, a
# strategy can move between brokers with much less implementation risk.
#
# **Next**: Compare the simpler REST-style path in `04_alpaca_paper_trading_demo.py`, then move to
# `08_pipeline_verification.py` to check that research outputs and live inputs still match.
