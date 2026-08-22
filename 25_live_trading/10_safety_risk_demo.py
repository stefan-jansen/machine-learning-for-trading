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
# # SafeBroker Risk Controls Demo
#
# **Docker image**: `ml4t`
#
# **Section Reference**: 25.7 (Operational Readiness)
#
# **Implementation Skills**:
# - ml4t.live.safety: SafeBroker, LiveRiskConfig, RiskState, RiskLimitError
# - ml4t.live.safety: VirtualPortfolio for shadow mode position tracking
# - Kill switch, position limits, order limits, drawdown monitoring
#
# **Key Learning**:
# This notebook drives six of SafeBroker's risk controls into their
# failure modes against a synthetic broker:
#
# 1. **Order Size Limits** — Max shares and max value per order
# 2. **Position Limits** — Max value, shares, and total exposure
# 3. **Rate Limiting** — Max orders per minute
# 4. **Asset Restrictions** — Allowed/blocked asset lists
# 5. **Kill Switch** — Emergency halt that persists across restarts
# 6. **Shadow Mode** — VirtualPortfolio tracks fills without touching the broker
#
# Three further controls - duplicate-order filtering, price-deviation checks,
# and daily-loss monitoring - are available through `LiveRiskConfig` but are
# not all re-demonstrated here. NB13 directly exercises the daily-loss
# kill-switch trip and stale-data rejection.
#
# **Why This Matters**:
# Live trading can lose real money. These safeguards provide defense-in-depth
# against bugs, API errors, fat-finger mistakes, and runaway strategies.
#
# **Learning Objectives**
# - See how each protection layer fails closed when the strategy asks for something unsafe.
# - Distinguish order-level checks from portfolio-level checks and persistent kill switches.
# - Read the risk-control output as an operational checklist, not as a library feature tour.
#
# **Prerequisites**
# - Review Chapter 25.7 on operational readiness and the broker wrappers introduced earlier in the chapter.
# - Familiarity with shadow mode and why paper trading alone is not enough for release gating.

# %% [markdown]
# ## Setup
#
# The setup is intentionally minimal because the point of the notebook is to expose the risk engine's
# decisions. A lightweight mock broker is enough to make the failure modes and guardrails visible.

# %%
"""SafeBroker Risk Controls Demo: exercise six controls plus shadow accounting."""

import logging
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from async_utils import run_async
from ml4t.backtest.types import Order, OrderSide, OrderStatus, OrderType, Position
from ml4t.live import LiveRiskConfig, RiskLimitError, SafeBroker, VirtualPortfolio
from ml4t.live.protocols import ExecutionCapability

from utils.paths import get_output_dir

print("[OK] ml4t.live risk controls imported")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
RATE_LIMIT_PER_MINUTE = 3
STATE_DIR = get_output_dir(25, "safety_risk_demo") / "temporary_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
MANAGED_STATE_PATHS: list[Path] = []


# %% [markdown]
# Each risk-control scenario receives a unique state path so persistence is testable without shared files.


# %%
def _temporary_state_path(prefix: str = "nb10_state_") -> str:
    """Return one isolated state path and track its state and journal files."""
    state_path = STATE_DIR / f"{prefix}{len(MANAGED_STATE_PATHS) // 2 + 1:02d}.json"
    journal_path = state_path.with_name(f"{state_path.stem}-journal{state_path.suffix}l")
    for path in (state_path, journal_path):
        path.unlink(missing_ok=True)
        MANAGED_STATE_PATHS.append(path)
    return str(state_path)


# %% [markdown]
# ## 1. Mock Broker for Testing
#
# The query surface separates broker state from fill mechanics. Every blocked order can therefore be
# attributed to the risk layer rather than connectivity or market-data noise.


# %%
class MockBrokerQueries:
    """Read-only and connection methods required by `SafeBroker`."""

    async def connect(self) -> None:
        self._connected = True
        logger.info("MockBroker: Connected")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("MockBroker: Disconnected")

    async def is_connected_async(self) -> bool:
        return self._connected

    @property
    def execution_capabilities(self):
        return frozenset({ExecutionCapability.LIMIT})

    def assert_paper_trading(self) -> None:
        return None

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions.copy()

    @property
    def pending_orders(self) -> list[Order]:
        return self._pending_orders.copy()

    def get_position(self, asset: str) -> Position | None:
        return self._positions.get(asset)

    async def get_positions_async(self) -> dict[str, Position]:
        return self._positions.copy()

    async def get_pending_orders_async(self) -> list[Order]:
        return self._pending_orders.copy()

    async def get_position_async(self, asset: str) -> Position | None:
        return self._positions.get(asset)

    async def get_account_value_async(self) -> float:
        position_value = sum(
            abs(p.quantity) * (p.current_price or p.entry_price) for p in self._positions.values()
        )
        return self._cash + position_value

    async def get_cash_async(self) -> float:
        return self._cash


# %% [markdown]
# Order construction is pure apart from the broker-local sequence number.


# %%
def make_filled_order(
    broker: Any,
    asset: str,
    quantity: int,
    side: OrderSide,
    order_type: OrderType,
    limit_price: float | None,
    stop_price: float | None,
) -> Order:
    """Construct one immediately filled synthetic order."""
    broker._order_counter += 1
    price = limit_price or 100.0
    return Order(
        asset=asset,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        order_id=f"MOCK-{broker._order_counter}",
        status=OrderStatus.FILLED,
        filled_quantity=quantity,
        filled_price=price,
        filled_at=datetime.now(UTC),
    )


# %% [markdown]
# Fill accounting updates position quantity and cash from the same signed transaction.


# %%
def apply_mock_fill(broker: Any, order: Order) -> None:
    """Apply one filled order to synthetic broker state."""
    asset, quantity, side = order.asset, order.quantity, order.side
    price = float(order.filled_price)
    signed_qty = quantity if side == OrderSide.BUY else -quantity
    current = broker._positions.get(asset)
    if current:
        new_qty = current.quantity + signed_qty
        if new_qty == 0:
            del broker._positions[asset]
        else:
            current.quantity = new_qty
    else:
        broker._positions[asset] = Position(
            asset=asset,
            quantity=signed_qty,
            entry_price=price,
            entry_time=datetime.now(UTC),
            current_price=price,
        )
    transaction = quantity * price
    broker._cash += transaction if side == OrderSide.SELL else -transaction


# %% [markdown]
# The mutable broker adds immediate-fill submission to the query surface.


# %%
class MockBroker(MockBrokerQueries):
    """Synthetic broker for exercising `SafeBroker` controls."""

    def __init__(self, initial_cash: float = 100_000.0):
        self._cash = initial_cash
        self._positions: dict[str, Position] = {}
        self._pending_orders: list[Order] = []
        self._connected = False
        self._order_counter = 0

    async def submit_order_async(
        self,
        asset: str,
        quantity: int,
        side: OrderSide | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        **_: Any,
    ) -> Order:
        """Fill one order immediately and update synthetic account state."""
        if side is None:
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
            quantity = abs(quantity)
        order = make_filled_order(self, asset, quantity, side, order_type, limit_price, stop_price)
        apply_mock_fill(self, order)
        logger.info("MockBroker: %s %s %s @ $%.2f", side.value, quantity, asset, order.filled_price)
        return order

    async def cancel_order_async(self, order_id: str) -> bool:
        return False

    async def close_position_async(self, asset: str) -> Order | None:
        pos = self._positions.get(asset)
        if pos and pos.quantity != 0:
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            return await self.submit_order_async(asset, abs(pos.quantity), side)
        return None


# %% [markdown]
# ## 2. Demo Helper Function
#
# Helper to run async code and catch RiskLimitError for demo purposes.
#
# The helper prints the outcome of each attempted order so the notebook reads like an operator console: either
# the request is allowed, or the control layer explains why it is blocked.


# %%
def run_async_demo(awaitable):
    """Run one demo awaitable while suppressing only nest_asyncio deprecations."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"nest_asyncio(?:\..*)?",
        )
        return run_async(awaitable)


# %% [markdown]
# The outcome helper turns every expected allow or block decision into an executable assertion.


# %%
def run_demo(coro, expect_error: bool = False):
    """Run an async demo and fail the notebook on an unexpected outcome."""
    try:
        result = run_async_demo(coro)
    except RiskLimitError as e:
        if not expect_error:
            raise AssertionError(f"Unexpected risk block: {e}") from e
        print(f"   [OK] Blocked as expected: {e}")
        return None
    if expect_error:
        raise AssertionError("Expected RiskLimitError, but the order was accepted")
    print("   [OK] Order accepted by control layer")
    return result


# %% [markdown]
# **Finding**: The helper standardizes how each risk-control demo reports allowed versus blocked behavior, so
# the notebook reads like a repeatable release checklist rather than a pile of ad hoc exceptions.
#
# **Trading implication**: Production risk gates should produce consistent operator-facing outcomes because
# inconsistent error reporting makes real incidents harder to diagnose under time pressure.
#
# %% [markdown]
# ## 3. Demonstration: Order Size Limits
#
# SafeBroker enforces maximum order size (shares and value).
#
# This first demo answers the simplest production question: can one buggy order wipe out the session before
# anything else has a chance to react?

# %%
print("\n" + "=" * 70)
print("DEMO 1: Order Size Limits")
print("=" * 70)

# Use temp file for state to avoid conflicts
state_file = _temporary_state_path()

broker = MockBroker()
run_async_demo(broker.connect())

config = LiveRiskConfig(
    execution_mode="paper",
    max_order_shares=50,  # Max 50 shares per order
    max_order_value=5_000.0,  # Max $5,000 per order
    state_file=state_file,
)
safe_broker = SafeBroker(broker, config)
# Prime the staleness guard with reference quotes for every asset this demo touches.
safe_broker.record_market_snapshot("AAPL", 100.0)
safe_broker.record_market_snapshot("TSLA", 600.0)

print("\n1. Order within limits (10 shares @ $100 = $1,000):")
run_demo(safe_broker.submit_order_async("AAPL", 10, OrderSide.BUY))

print("\n2. Order exceeds share limit (100 shares, limit is 50):")
run_demo(
    safe_broker.submit_order_async("AAPL", 100, OrderSide.BUY),
    expect_error=True,
)

print("\n3. Order exceeds value limit (10 shares @ $600 = $6,000):")
# First set a price by creating a position
broker._positions["TSLA"] = Position(
    asset="TSLA",
    quantity=1,
    entry_price=600.0,
    entry_time=datetime.now(UTC),
    current_price=600.0,
)
run_demo(
    safe_broker.submit_order_async(
        "TSLA", 10, OrderSide.BUY, order_type=OrderType.LIMIT, limit_price=600.0
    ),
    expect_error=True,
)

# %% [markdown]
# **Finding**: Order-size controls stop both oversized share counts and oversized notionals before the broker
# sees them.
#
# **Trading implication**: The first line of defense in live trading is to make a single bad instruction too
# small to become catastrophic.
#
# %% [markdown]
# ## 4. Demonstration: Position Limits
#
# SafeBroker enforces maximum position size and total exposure. This answers the portfolio-level question of
# whether a sequence of valid orders can still push the account into an unsafe aggregate state.

# %%
print("\n" + "=" * 70)
print("DEMO 2: Position Limits")
print("=" * 70)

# Isolate the share cap by setting the dollar limits well above the attempted position.
state_file = _temporary_state_path()
broker = MockBroker()
run_async_demo(broker.connect())

config = LiveRiskConfig(
    execution_mode="paper",
    max_position_value=100_000.0,
    max_position_shares=100,
    max_total_exposure=200_000.0,
    max_order_value=20_000.0,
    max_order_shares=200,
    state_file=state_file,
)
safe_broker = SafeBroker(broker, config)
safe_broker.record_market_snapshot("AAPL", 100.0)

print("\n1. Share cap: start with 50 shares @ $100:")
run_demo(safe_broker.submit_order_async("AAPL", 50, OrderSide.BUY))
print("\n2. Share cap: adding 60 would reach 110 shares:")
run_demo(
    safe_broker.submit_order_async("AAPL", 60, OrderSide.BUY),
    expect_error=True,
)

# %% [markdown]
# A separate broker isolates the per-position dollar cap from the share and total-exposure gates.

# %%
state_file = _temporary_state_path()
broker = MockBroker()
run_async_demo(broker.connect())
config = LiveRiskConfig(
    execution_mode="paper",
    max_position_value=10_000.0,
    max_position_shares=200,
    max_total_exposure=200_000.0,
    max_order_value=20_000.0,
    max_order_shares=200,
    state_file=state_file,
)
safe_broker = SafeBroker(broker, config)
safe_broker.record_market_snapshot("MSFT", 150.0)

print("\n3. Position-value cap: 100 MSFT @ $150 would be $15,000:")
run_demo(
    safe_broker.submit_order_async(
        "MSFT", 100, OrderSide.BUY, order_type=OrderType.LIMIT, limit_price=150.0
    ),
    expect_error=True,
)

# %% [markdown]
# A third broker builds two valid positions before a new order crosses only the total-exposure cap.

# %%
state_file = _temporary_state_path()
broker = MockBroker()
run_async_demo(broker.connect())
config = LiveRiskConfig(
    execution_mode="paper",
    max_position_value=20_000.0,
    max_position_shares=200,
    max_total_exposure=15_000.0,
    max_order_value=20_000.0,
    max_order_shares=200,
    state_file=state_file,
)
safe_broker = SafeBroker(broker, config)
for symbol in ("AAPL", "MSFT", "GOOGL"):
    safe_broker.record_market_snapshot(symbol, 100.0)

print("\n4. Total exposure: build valid $5,000 and $8,000 positions:")
run_demo(safe_broker.submit_order_async("AAPL", 50, OrderSide.BUY))
run_demo(safe_broker.submit_order_async("MSFT", 80, OrderSide.BUY))
print("\n5. Total exposure: another $3,000 would raise exposure to $16,000:")
run_demo(
    safe_broker.submit_order_async("GOOGL", 30, OrderSide.BUY),
    expect_error=True,
)

# %% [markdown]
# **Finding**: Independent scenarios trigger the share, per-position value, and total-exposure gates without
# an earlier control masking the intended rejection.
#
# **Trading implication**: Portfolio-level controls are necessary because many failures arrive as a sequence
# of reasonable-looking orders that add up to unreasonable exposure.
#
# %% [markdown]
# ## 5. Demonstration: Rate Limiting
#
# SafeBroker limits the number of orders per minute. Rate limiting matters because repeated small errors can
# be just as destructive as one oversized order when the loop is running unattended.

# %%
print("\n" + "=" * 70)
print("DEMO 3: Rate Limiting")
print("=" * 70)

state_file = _temporary_state_path()
broker = MockBroker()
run_async_demo(broker.connect())

config = LiveRiskConfig(
    execution_mode="paper",
    max_orders_per_minute=RATE_LIMIT_PER_MINUTE,
    max_order_value=50_000.0,
    max_position_value=100_000.0,
    dedup_window_seconds=0.0,  # Disable dedup for this demo
    state_file=state_file,
)
safe_broker = SafeBroker(broker, config)

print("\nSubmitting 5 orders rapidly (limit is 3/minute):")
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  # Different symbols to avoid dedup
for sym in symbols:
    safe_broker.record_market_snapshot(sym, 100.0)
for i in range(5):
    print(f"\nOrder {i + 1} ({symbols[i]}):")
    run_demo(
        safe_broker.submit_order_async(symbols[i], 10, OrderSide.BUY),
        expect_error=(i >= 3),  # Expect error after 3rd order
    )

# %% [markdown]
# **Finding**: The rate-limit demo shows that SafeBroker can reject a burst even when every order is otherwise
# valid.
#
# **Trading implication**: Throughput limits protect the account from runaway loops, duplicate signal storms,
# and broker bans caused by overly chatty execution code.
#
# %% [markdown]
# ## 6. Demonstration: Asset Restrictions
#
# SafeBroker can restrict trading to allowed assets or block specific assets. This is how a live system keeps
# a strategy inside its approved mandate even if symbol selection logic goes wrong.

# %%
print("\n" + "=" * 70)
print("DEMO 4: Asset Restrictions")
print("=" * 70)

state_file = _temporary_state_path()
broker = MockBroker()
run_async_demo(broker.connect())

# Only allow specific ETFs
config = LiveRiskConfig(
    execution_mode="paper",
    allowed_assets={"SPY", "QQQ", "IWM"},  # Whitelist
    max_order_value=50_000.0,
    max_position_value=100_000.0,
    state_file=state_file,
)
safe_broker = SafeBroker(broker, config)
# Only prime SPY: AAPL is blocked at the asset-restriction check before staleness ever runs.
safe_broker.record_market_snapshot("SPY", 100.0)

print("\n1. Trading SPY (allowed):")
run_demo(safe_broker.submit_order_async("SPY", 10, OrderSide.BUY))

print("\n2. Trading AAPL (not in allowed list):")
run_demo(
    safe_broker.submit_order_async("AAPL", 10, OrderSide.BUY),
    expect_error=True,
)

# %% [markdown]
# Now exercise the blocklist path: trades to non-blocked assets are
# accepted, trades to any listed symbol are rejected before they reach
# the broker.

# %%
print("\n--- Testing Blocked Assets ---")
state_file = _temporary_state_path()
broker = MockBroker()
run_async_demo(broker.connect())

config = LiveRiskConfig(
    execution_mode="paper",
    blocked_assets={"TSLA", "GME", "AMC"},  # Blacklist volatile stocks
    max_order_value=50_000.0,
    max_position_value=100_000.0,
    state_file=state_file,
)
safe_broker = SafeBroker(broker, config)
# Only prime AAPL: GME is blocked at the asset-restriction check before staleness runs.
safe_broker.record_market_snapshot("AAPL", 100.0)

print("\n3. Trading AAPL (not blocked):")
run_demo(safe_broker.submit_order_async("AAPL", 10, OrderSide.BUY))

print("\n4. Trading GME (blocked):")
run_demo(
    safe_broker.submit_order_async("GME", 10, OrderSide.BUY),
    expect_error=True,
)

# %% [markdown]
# **Finding**: Asset allowlists and blocklists create a hard boundary around what the strategy is permitted to
# touch.
#
# **Trading implication**: Universe control is an operational safeguard, not just a research convenience,
# because it prevents accidental routing into unsupported or explicitly banned instruments.
#
# %% [markdown]
# ## 7. Demonstration: Kill Switch
#
# The kill switch is an emergency halt that persists across restarts. It exists for the scenarios where the
# safest action is to stop every new order until a human explicitly clears the system.

# %%
print("\n" + "=" * 70)
print("DEMO 5: Kill Switch")
print("=" * 70)

state_file = _temporary_state_path()
broker = MockBroker()
run_async_demo(broker.connect())

config = LiveRiskConfig(
    execution_mode="paper",
    max_order_value=50_000.0,
    max_position_value=100_000.0,
    dedup_window_seconds=0.0,  # Disable for demo
    state_file=state_file,
)
safe_broker = SafeBroker(broker, config)
safe_broker.record_market_snapshot("AAPL", 100.0)

print("\n1. Normal trading before kill switch:")
run_demo(safe_broker.submit_order_async("AAPL", 10, OrderSide.BUY))

print("\n2. Activating kill switch (manual emergency halt):")
safe_broker.enable_kill_switch("Manual test - simulating emergency")
print("   Kill switch activated!")

print("\n3. Attempting to trade with kill switch active:")
run_demo(
    safe_broker.submit_order_async("AAPL", 10, OrderSide.BUY),
    expect_error=True,
)

# %% [markdown]
# Reconstruct `SafeBroker` from the same state file to confirm the kill
# switch survives — the in-memory toggle is irrelevant if the latch
# disappears on restart.

# %%
print("\n4. Checking state persistence...")
safe_broker.close_persistence()
new_safe_broker = SafeBroker(MockBroker(), config)
print(f"   Kill switch still active: {new_safe_broker._state.kill_switch_activated}")
print(f"   Reason: {new_safe_broker._state.kill_switch_reason}")
assert new_safe_broker._state.kill_switch_activated

print("\n5. Disabling kill switch (manual recovery):")
new_safe_broker.disable_kill_switch()
print("   Kill switch disabled!")

print("\n6. Trading after recovery:")
new_safe_broker.record_market_snapshot("AAPL", 100.0)
_ = run_demo(new_safe_broker.submit_order_async("AAPL", 10, OrderSide.BUY))
new_safe_broker.close_persistence()

# %% [markdown]
# **Finding**: The kill switch persists across SafeBroker instances, so an emergency halt survives process
# restarts instead of disappearing with the notebook kernel.
#
# **Trading implication**: Manual intervention has to outlive the current process, otherwise a restart can
# unintentionally reactivate a strategy that was halted for a real risk event.
#
# %% [markdown]
# ## 8. Demonstration: Shadow Mode
#
# Shadow mode logs orders but doesn't execute them. Uses VirtualPortfolio
# for realistic position tracking to prevent infinite buy loops.

# %%
print("\n" + "=" * 70)
print("DEMO 6: Shadow Mode (No Broker Submission)")
print("=" * 70)

state_file = _temporary_state_path()
broker = MockBroker()
run_async_demo(broker.connect())

config = LiveRiskConfig(
    shadow_mode=True,  # Enable shadow mode
    max_order_value=50_000.0,
    max_position_value=100_000.0,
    state_file=state_file,
)
safe_broker = SafeBroker(broker, config)
safe_broker.record_market_snapshot("AAPL", 100.0)

print("\n1. Submitting order in shadow mode:")
order = run_demo(safe_broker.submit_order_async("AAPL", 100, OrderSide.BUY))

print("\n2. Checking virtual position (shadows real broker):")
pos = safe_broker.get_position("AAPL")
if pos:
    print(f"   Virtual position: {pos.quantity} shares @ ${pos.entry_price:.2f}")
else:
    raise AssertionError("Shadow order did not update the virtual position")

print("\n3. Checking real broker position (should be empty):")
real_pos = broker.get_position("AAPL")
if real_pos:
    raise AssertionError(f"Shadow order reached the real broker: {real_pos.quantity} shares")
else:
    print("   Real broker has NO position (correct - shadow mode)")

print("\n4. Virtual portfolio account value:")
value = run_async_demo(safe_broker.get_account_value_async())
print(f"   Virtual account value: ${value:,.2f}")

# %% [markdown]
# **Finding**: Shadow mode updates the virtual portfolio while leaving the underlying broker flat. That makes
# the execution path observable without changing real inventory.
#
# **Trading implication**: Shadow mode is the safest way to validate end-to-end routing logic before paper or
# live trading because it exercises the controls without external side effects.
#
# %% [markdown]
# ## 9. Demonstration: VirtualPortfolio Details
#
# VirtualPortfolio handles position tracking for shadow mode,
# including weighted average cost basis and position flipping.

# %%
print("\n" + "=" * 70)
print("DEMO 7: VirtualPortfolio Position Tracking")
print("=" * 70)

portfolio = VirtualPortfolio(initial_cash=100_000.0)

print("\n1. Initial state:")
print(f"   Cash: ${portfolio.cash:,.2f}")
print(f"   Positions: {len(portfolio.positions)}")

# Simulate buy order
buy_order = Order(
    asset="AAPL",
    side=OrderSide.BUY,
    quantity=100,
    filled_quantity=100,
    filled_price=150.0,
    status=OrderStatus.FILLED,
)
portfolio.process_fill(buy_order)

print("\n2. After buying 100 AAPL @ $150:")
pos = portfolio.positions.get("AAPL")
print(f"   Position: {pos.quantity} shares @ ${pos.entry_price:.2f}")
print(f"   Cash: ${portfolio.cash:,.2f}")
print(f"   Account value: ${portfolio.account_value:,.2f}")
assert pos.quantity == 100
assert pos.entry_price == 150.0
assert portfolio.account_value == 100_000.0

# %%
# Add to position at higher price
buy_order2 = Order(
    asset="AAPL",
    side=OrderSide.BUY,
    quantity=100,
    filled_quantity=100,
    filled_price=160.0,
    status=OrderStatus.FILLED,
)
portfolio.process_fill(buy_order2)

print("\n3. After buying 100 more AAPL @ $160 (weighted avg cost):")
pos = portfolio.positions.get("AAPL")
print(f"   Position: {pos.quantity} shares @ ${pos.entry_price:.2f}")
print(f"   Cash: ${portfolio.cash:,.2f}")
assert pos.quantity == 200
assert pos.entry_price == 155.0

# Partial sell
sell_order = Order(
    asset="AAPL",
    side=OrderSide.SELL,
    quantity=50,
    filled_quantity=50,
    filled_price=170.0,
    status=OrderStatus.FILLED,
)
portfolio.process_fill(sell_order)

print("\n4. After selling 50 AAPL @ $170:")
pos = portfolio.positions.get("AAPL")
print(f"   Position: {pos.quantity} shares @ ${pos.entry_price:.2f}")
print(f"   Cash: ${portfolio.cash:,.2f}")
print(f"   Account value: ${portfolio.account_value:,.2f}")
assert pos.quantity == 150
assert pos.entry_price == 155.0
assert portfolio.account_value == 103_000.0

# %%
for managed_path in MANAGED_STATE_PATHS:
    for candidate in (
        managed_path,
        managed_path.with_name(f"{managed_path.name}.lock"),
        managed_path.with_name(f"{managed_path.name}.head"),
    ):
        candidate.unlink(missing_ok=True)

unexpected_state_files = list(STATE_DIR.iterdir())
assert not unexpected_state_files, f"Temporary state residue: {unexpected_state_files}"
STATE_DIR.rmdir()
print(f"[OK] Cleaned {len(MANAGED_STATE_PATHS)} managed state and journal paths")

# %% [markdown]
# **Finding**: The virtual portfolio example makes cost-basis and partial-exit bookkeeping explicit instead of
# treating them as hidden implementation details.
#
# **Trading implication**: Paper and shadow environments need realistic position accounting or they will mask
# the exact state-management bugs that later damage live trading.
#
# %% [markdown]
# ## Summary: Controls Exercised Above
#
# | Control | What this notebook showed |
# |---------|---------------------------|
# | **Order Size Limits** | Max shares and value per order rejected before the broker sees them |
# | **Position Limits** | Per-position value + share caps blocked unsafe accumulation |
# | **Rate Limiting** | Burst of valid orders blocked at the per-minute cap |
# | **Asset Restrictions** | Allowed / blocked asset lists held the universe boundary |
# | **Kill Switch** | Manual halt, latch persisted across `SafeBroker` reconstruction |
# | **Shadow Mode** | `VirtualPortfolio` tracked fills while the underlying broker stayed flat |
#
# **Controls covered elsewhere**:
# - **Duplicate Order Filter** and **Price Deviation** are configured through
#   `LiveRiskConfig.dedup_window_seconds` and `LiveRiskConfig.max_price_deviation_pct`;
#   this notebook does not present them as executed results.
# - **Daily-Loss Drawdown Monitor** + **Stale-Data Rejection** are demonstrated in
#   `13_runtime_safety_showcase`, which drives the same `SafeBroker` instance
#   through both failure modes with full kill-switch latch.
#
# **Additional features the demos rely on**:
# - **State Persistence**: kill switch and daily counters survive restarts.
# - **Atomic Writes**: the state file uses atomic JSON writes to prevent corruption.
#
# **Best Practices**:
# 1. Always start with `shadow_mode=True`.
# 2. Graduate to paper trading.
# 3. Use conservative limits when going live.
# 4. Monitor the state file for kill-switch activations.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Defense-in-depth**: SafeBroker layers independent controls so that no
#    single misconfiguration can produce an unchecked order. Six of those
#    controls are exercised above; the remaining duplicate-order, fat-finger,
#    and drawdown gates are wired through the same `LiveRiskConfig` surface
#    without being claimed as outputs of this notebook.
# 2. **Configurable via LiveRiskConfig**: Every limit—order size, position
#    exposure, rate caps, drawdown thresholds—is a parameter, not hard-coded
#    logic.
# 3. **Kill switch persists across restarts**: Emergency halts survive process
#    restarts and must be manually cleared, preventing accidental
#    reactivation of a halted strategy.
# 4. **Shadow mode with VirtualPortfolio**: Orders are logged and tracked with
#    realistic cost-basis accounting without touching the broker, making it
#    the safest first deployment step.
#
# **Next**: Combine these controls with the parity checks in `pipeline_verification` and then keep the
# same SafeBroker configuration when moving from shadow mode to paper trading.
