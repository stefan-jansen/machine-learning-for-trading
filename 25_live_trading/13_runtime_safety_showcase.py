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
# # Runtime Safety Showcase: Stale Data, Kill Switch, Reconciliation, Health
#
# **Docker image**: `ml4t`
#
# **Section Reference**: 25.7 (Operational Readiness)
#
# **Implementation Skills**:
# - `ml4t.live.safety`: enforced `max_data_staleness_seconds` and `max_daily_loss`
# - `ml4t.live.safety`: persisted `RiskState` and `SafeBroker.connect()` startup reconciliation
# - `ml4t.live.engine`: `LiveEngine.runtime_status()` and the `ok` / `waiting_for_data` /
#   `feed_silent` / `idle_market_closed` / `broker_disconnected` / `stopped` health states
# - `ml4t-live` CLI as an out-of-process operator surface
#
# **Why This Notebook Exists**
#
# `10_safety_risk_demo` walks through the configurable risk surface — the knobs the strategy
# author exposes to the operator. This notebook exercises the runtime trust contract: what
# happens *under failure* in the kinds of scenarios a live deployment must survive. A pre-flight
# checklist is only useful if the controls actually fire when the conditions they protect
# against arise. Here each control is driven into the failure mode it was designed for.
#
# **Learning Objectives**
# - Watch `SafeBroker` reject orders against a stale `MarketSnapshot` and against a daily-loss
#   breach, and confirm the kill switch latches across `SafeBroker` reconstruction.
# - Inspect a non-clean `reconciliation_report` from a deliberately divergent persisted state
#   file, then resolve it and reconnect to a clean report.
# - Read `LiveEngine.runtime_status()` and observe the engine transition through the health
#   states named in §25.7.
# - Use the `ml4t-live` CLI to inspect the persisted state file out of process.
#
# **Prerequisites**
# - `ml4t-live >= 0.1.0` installed (`uv sync` from repo root).
# - Read §25.7 for the operational framing; `10_safety_risk_demo` for the configurable surface.
# - No broker credentials, no exchange access — every demo runs against a synthetic broker.

# %% [markdown]
# ## Setup
#
# Imports, logging, and a minimal asynchronous broker that satisfies the protocol `SafeBroker`
# expects. The synthetic broker is deliberately stateful so each demo can shape its position,
# order, and account-value snapshots without involving a live venue.

# %%
"""Runtime Safety Showcase — stale-data, daily-loss, reconciliation, and engine health states."""

import asyncio
import json
import logging
import os
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from ml4t.backtest.strategy import Strategy
from ml4t.backtest.types import Order, OrderSide, OrderStatus, OrderType, Position
from ml4t.live import LiveRiskConfig, RiskLimitError, RiskState, SafeBroker
from ml4t.live.engine import LiveEngine

from utils.paths import get_output_dir

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_demo(awaitable):
    """Run an async demo outside the notebook kernel's active event loop."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(awaitable)).result()


# %% tags=["parameters"]
# Production defaults — Papermill may inject overrides
STALE_DATA_WAIT_SECONDS = 1.5  # how long to sleep before retrying with stale data
KILL_SWITCH_LOSS_USD = 1_000.0  # synthetic equity drawdown driving the trip
HEALTH_OBSERVATION_SECONDS = 6  # total wall time spent observing engine health states
STATE_DIR = get_output_dir(25, "runtime_safety_showcase") / "temporary_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### A Synthetic Broker
#
# The query surface exposes only the state `SafeBroker` needs. Order submission follows in a
# separate cell so readers can distinguish broker observation from broker mutation.


# %%
class DemoBrokerQueries:
    """Connection and read-only methods required by `SafeBroker`."""

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def is_connected_async(self) -> bool:
        return self._connected

    @property
    def execution_capabilities(self):
        return frozenset()

    def assert_paper_trading(self) -> None:
        return None

    def get_position(self, asset: str) -> Position | None:
        return self.positions.get(asset)

    async def get_position_async(self, asset: str) -> Position | None:
        return self.positions.get(asset)

    async def get_positions_async(self) -> dict[str, Position]:
        return dict(self.positions)

    async def get_pending_orders_async(self) -> list[Order]:
        return list(self.pending_orders)

    async def get_account_value_async(self) -> float:
        return self.account_value

    async def get_cash_async(self) -> float:
        return self.account_value


# %% [markdown]
# The mutable broker appends synthetic orders locally and never opens an external connection.


# %%
class DemoBroker(DemoBrokerQueries):
    def __init__(
        self,
        *,
        positions: dict[str, Position] | None = None,
        pending_orders: list[Order] | None = None,
        account_value: float = 100_000.0,
    ) -> None:
        self._connected = False
        self.positions = dict(positions or {})
        self.pending_orders = list(pending_orders or [])
        self.account_value = float(account_value)
        self._order_counter = 0

    async def submit_order_async(
        self,
        asset: str,
        quantity: float,
        side: OrderSide | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        **_: Any,
    ) -> Order:
        if side is None:
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
            quantity = abs(quantity)
        self._order_counter += 1
        order = Order(
            asset=asset,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            order_id=f"DEMO-{self._order_counter:04d}",
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )
        self.pending_orders.append(order)
        return order

    async def cancel_order_async(self, order_id: str) -> bool:
        return False

    async def close_position_async(self, asset: str) -> Order | None:
        return None


# %% [markdown]
# ## 1. Stale-Data Rejection
#
# `SafeBroker` keeps a `MarketSnapshot` per asset whenever the engine receives bars. Every
# order intent runs through `_check_data_staleness`, which compares the snapshot age to
# `LiveRiskConfig.max_data_staleness_seconds`. Orders against a snapshot older than that
# threshold raise `RiskLimitError` — the freshest known price is no longer trustworthy enough
# to size against.
#
# The demo below sets a 1-second staleness threshold, accepts one order against a fresh
# snapshot, then waits past the threshold and submits another order. The second order must
# fail. No retry, no fallback to a stale price — the strategy must wait for fresh data or
# stand down.


# %%
def _temp_state_path(prefix: str) -> Path:
    """Return a non-existent path inside the isolated notebook output directory."""
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".json", dir=STATE_DIR)
    os.close(fd)
    p = Path(name)
    p.unlink(missing_ok=True)
    return p


def _cleanup_state_path(path: Path) -> None:
    """Remove a temporary risk-state file and its default sibling journal."""
    journal = path.with_name(f"{path.stem}-journal{path.suffix or '.json'}l")
    for candidate in (
        path,
        path.with_name(f"{path.name}.lock"),
        journal,
        journal.with_name(f"{journal.name}.lock"),
        journal.with_name(f"{journal.name}.head"),
    ):
        candidate.unlink(missing_ok=True)


stale_state = _temp_state_path("nb13_stale_")
stale_broker = DemoBroker()
stale_safe = SafeBroker(
    stale_broker,
    LiveRiskConfig(
        execution_mode="paper",
        max_order_value=10_000.0,
        max_data_staleness_seconds=1.0,
        state_file=str(stale_state),
    ),
)
run_demo(stale_safe.connect())

# Fresh snapshot — order accepted
stale_safe._record_market_data(datetime.now(UTC), {"DEMO": {"close": 100.0}}, {})
fresh_order = run_demo(stale_safe.submit_order_async("DEMO", 10))
print(f"fresh_data_order: accepted ({fresh_order.order_type.value} {fresh_order.quantity} DEMO)")

# Wait past staleness threshold
time.sleep(STALE_DATA_WAIT_SECONDS)

try:
    run_demo(stale_safe.submit_order_async("DEMO", 1))
except RiskLimitError as exc:
    print(f"stale_data_block: {exc}")
else:
    raise AssertionError("Stale market data did not block the second order")

run_demo(stale_safe.disconnect())
_cleanup_state_path(stale_state)

# %% [markdown]
# **Finding**: the second order never reaches the broker adapter. The `RiskLimitError` carries
# the measured age and the configured threshold, which gives the operator exactly the diagnostic
# needed to decide whether to widen the staleness window, investigate the feed, or halt.
#
# **Trading implication**: a feed that goes silent during a venue outage is a common live
# failure mode, and trading on the last known price during the outage is exactly how a
# stop-loss strategy ends up turning into a momentum-against strategy. The staleness check is
# the runtime-trust counterpart to the pre-flight requirement that "prices are current rather
# than stale" stated in §25.7.

# %% [markdown]
# ## 2. Auto Kill-Switch Trip on Daily-Loss Breach
#
# `_check_daily_loss` runs on every order submission. It reads the broker's account value,
# anchors `session_start_equity` on first call, and computes intraday loss as
# $\max(0, \text{session\_start\_equity} - \text{current\_equity})$. When the loss exceeds
# `LiveRiskConfig.max_daily_loss`, the kill switch latches and the order is rejected.
#
# The demo simulates an equity drawdown by mutating the broker's `account_value` mid-session.
# The first order anchors the session start; the equity drops; the next order trips the
# kill switch. The persisted `RiskState` records the activation reason so a same-day restart
# resumes with the latch intact.

# %%
killswitch_state = _temp_state_path("nb13_killswitch_")
killswitch_broker = DemoBroker(account_value=100_000.0)
killswitch_safe = SafeBroker(
    killswitch_broker,
    LiveRiskConfig(
        execution_mode="paper",
        max_order_value=10_000.0,
        max_daily_loss=KILL_SWITCH_LOSS_USD,
        max_data_staleness_seconds=60.0,
        state_file=str(killswitch_state),
    ),
)
run_demo(killswitch_safe.connect())
killswitch_safe._record_market_data(datetime.now(UTC), {"DEMO": {"close": 100.0}}, {})

# Anchor session_start_equity at $100k
run_demo(killswitch_safe.submit_order_async("DEMO", 1))
print(f"pre_drawdown_state: kill_switch={killswitch_safe._state.kill_switch_activated}")

# Simulated equity drop
killswitch_broker.account_value = 100_000.0 - (KILL_SWITCH_LOSS_USD + 250.0)
killswitch_safe._record_market_data(datetime.now(UTC), {"DEMO": {"close": 99.0}}, {})

try:
    run_demo(killswitch_safe.submit_order_async("DEMO", 1))
except RiskLimitError as exc:
    print(f"daily_loss_block: {exc}")
else:
    raise AssertionError("Daily-loss breach did not block the order")

print(
    "post_breach_state: "
    f"kill_switch={killswitch_safe._state.kill_switch_activated} "
    f"daily_loss=${killswitch_safe._state.daily_loss:,.0f} "
    f"reason={killswitch_safe._state.kill_switch_reason!r}"
)
run_demo(killswitch_safe.disconnect())

# %% [markdown]
# ### 2a. Latch Survives `SafeBroker` Reconstruction
#
# A real engine restart re-instantiates `SafeBroker` from its state file. The latch and the
# activation reason persist; the next order is rejected before it reaches any risk check
# downstream of the kill switch.

# %%
killswitch_safe2 = SafeBroker(
    DemoBroker(account_value=100_000.0),
    LiveRiskConfig(
        execution_mode="paper",
        max_order_value=10_000.0,
        max_daily_loss=KILL_SWITCH_LOSS_USD,
        max_data_staleness_seconds=60.0,
        state_file=str(killswitch_state),
    ),
)
print(
    "after_restart_state: "
    f"kill_switch={killswitch_safe2._state.kill_switch_activated} "
    f"reason={killswitch_safe2._state.kill_switch_reason!r}"
)
run_demo(killswitch_safe2.connect())
killswitch_safe2._record_market_data(datetime.now(UTC), {"DEMO": {"close": 100.0}}, {})
try:
    run_demo(killswitch_safe2.submit_order_async("DEMO", 1))
except RiskLimitError as exc:
    print(f"post_restart_block: {exc}")
else:
    raise AssertionError("Persisted kill switch did not block the restarted broker")
run_demo(killswitch_safe2.disconnect())

# Reset for downstream demos
_cleanup_state_path(killswitch_state)

# %% [markdown]
# **Finding**: the kill-switch behaviour is the runtime-trust counterpart to the operational
# requirement that "the kill switch should remain independent of the component it controls"
# from §25.7. Because the latch lives in a state file, an engine crash and reconnect does not
# silently re-enable trading; the operator must explicitly clear the latch before the engine
# resumes. The Friday-after-hours pile-up that motivated NB12's startup-reconciliation pattern
# is an analogous failure mode in the position dimension; this is the same pattern in the
# P&L dimension.
#
# **Trading implication**: latching the kill switch in persistent state is what makes the
# control useful for emergencies. A non-persistent kill switch is just a flag in process
# memory, and the worst time to discover that is during the recovery from a disorderly day.

# %% [markdown]
# ## 3. Reconciliation Report on a Divergent State File
#
# `SafeBroker.connect()` diffs the persisted snapshot from the previous session against the
# broker's authoritative current state. The demo below writes a state file claiming the
# strategy ended the previous session long 10 AAPL with a working LIMIT order, then connects
# to a broker that knows about neither. The reconciliation report lists every divergence; a
# production launcher refuses to start a new trading cycle until the report is clean or the
# operator has explicitly cleared the persisted state.

# %%
recon_state = _temp_state_path("nb13_recon_")
divergent = RiskState(
    date=datetime.now(UTC).date().isoformat(),
    persisted_positions={"AAPL": 10.0},
    persisted_pending_orders=[
        {
            "asset": "AAPL",
            "side": "buy",
            "quantity": 10.0,
            "order_type": "limit",
            "limit_price": 150.0,
        }
    ],
)
recon_state.write_text(json.dumps(divergent.to_dict(), indent=2))
recon_state.chmod(0o600)

recon_broker = DemoBroker(
    positions={
        "MSFT": Position(
            asset="MSFT",
            quantity=5,
            entry_price=410.0,
            entry_time=datetime.now(UTC),
        )
    }
)
recon_safe = SafeBroker(
    recon_broker,
    LiveRiskConfig(execution_mode="paper", state_file=str(recon_state)),
)
run_demo(recon_safe.connect())

report = recon_safe.reconciliation_report
print(f"clean: {report['clean']}")
print(f"missing_positions:    {report['missing_positions']}")
print(f"unexpected_positions: {report['unexpected_positions']}")
print(f"quantity_mismatches:  {report['quantity_mismatches']}")
print(f"missing_pending_orders:    {len(report['missing_pending_orders'])} order(s)")
print(f"unexpected_pending_orders: {len(report['unexpected_pending_orders'])} order(s)")
assert report["clean"] is False
assert report["missing_positions"] == {"AAPL": 10.0}
assert report["unexpected_positions"] == {"MSFT": 5.0}
assert len(report["missing_pending_orders"]) == 1

# %% [markdown]
# **Finding**: the report names exactly which positions and orders disagree, in which
# direction. `missing_positions` are positions the persisted state expected and the broker no
# longer reports — typically a manual flatten, an after-hours fill, or an overnight
# corporate action. `unexpected_positions` are positions present at the broker that the
# persisted state did not know about — typically a fill that landed after the last persist,
# or a position created out of band.
#
# Reading the report is half the task; resolving it is the other half. The operator either
# investigates the divergence in the broker GUI and re-runs once it's understood, or — when
# the divergence is known to be benign — clears the persisted state and reconnects to a
# clean baseline.

# %%
run_demo(recon_safe.disconnect())
_cleanup_state_path(recon_state)

# Clean baseline: persisted positions match the broker exactly
clean_state = _temp_state_path("nb13_recon_clean_")
matched_state = RiskState(
    date=datetime.now(UTC).date().isoformat(),
    persisted_positions={"MSFT": 5.0},
    persisted_pending_orders=[],
)
clean_state.write_text(json.dumps(matched_state.to_dict(), indent=2))
clean_state.chmod(0o600)

recon_broker_clean = DemoBroker(
    positions={
        "MSFT": Position(
            asset="MSFT",
            quantity=5,
            entry_price=410.0,
            entry_time=datetime.now(UTC),
        )
    }
)
recon_safe_clean = SafeBroker(
    recon_broker_clean,
    LiveRiskConfig(execution_mode="paper", state_file=str(clean_state)),
)
run_demo(recon_safe_clean.connect())
clean_report = recon_safe_clean.reconciliation_report
print(f"after_reset clean: {clean_report['clean']}")
assert clean_report["clean"] is True
run_demo(recon_safe_clean.disconnect())
_cleanup_state_path(clean_state)

# %% [markdown]
# **Trading implication**: every production launcher should treat a non-clean reconciliation
# report the same way it treats a failed authentication probe — refuse to launch and wait for
# operator action. NB12 implements that policy; the IB basket loop will not submit if the
# report is not clean.

# %% [markdown]
# ## 4. Engine Health States
#
# `LiveEngine.runtime_status()` returns a dict whose `health` field reduces engine and feed
# state to a small set of operator-readable categories: `stopped`, `waiting_for_data`,
# `ok`, `feed_silent`, `idle_market_closed`, and `broker_disconnected`. The demo below
# constructs a tiny engine over a synthetic feed and a no-op strategy, runs it for a short
# bounded window, halts the feed, and reports the health transitions. No real venue, no
# broker connectivity — only the engine's own bookkeeping.
#
# The toy feed declares no equity symbols, so the engine treats it as a continuous
# market. This removes wall-clock session dependence and makes the health sequence
# reproducible at any hour.


# %%
@dataclass
class _DummyStrategy(Strategy):
    """Strategy that ignores every bar — health-state observation only."""

    bars_seen: int = 0

    def __post_init__(self) -> None:
        try:
            super().__init__()
        except TypeError:
            pass

    def on_data(self, *args: Any, **kwargs: Any) -> None:
        self.bars_seen += 1

    def on_start(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_end(self, *args: Any, **kwargs: Any) -> None:
        return None


# %% [markdown]
# #### Toy market-data feed
#
# A minimal async feed that emits a fixed number of bars then goes silent.
# Pairing it with `_DummyStrategy` lets the LiveEngine's health-state
# transitions surface (`waiting_for_data` → `ok` → `feed_silent`) without
# any real market dependency.


# %%
@dataclass
class _ToyFeed:
    """Synthetic feed that emits one bar then goes silent until stopped."""

    initial_bars: int = 1
    start_delay_seconds: float = 0.2
    bar_interval_seconds: float = 0.1
    silent_after_initial: bool = True
    stopped: bool = field(default=False)

    async def start(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def __aiter__(self):
        return self._stream()

    async def _stream(self):
        await asyncio.sleep(self.start_delay_seconds)
        for _ in range(self.initial_bars):
            if self.stopped:
                return
            yield (datetime.now(UTC), {"DEMO": {"close": 100.0}}, {})
            await asyncio.sleep(self.bar_interval_seconds)
        # Go silent: never yield more bars, just hold the iterator open
        while not self.stopped:
            await asyncio.sleep(0.1)


# %% [markdown]
# Wire the feed, strategy, and demo broker into a `LiveEngine` with a
# short feed-silence timeout so the watchdog flips to `feed_silent`
# quickly once the toy feed stops emitting bars.

# %%
broker_h = DemoBroker()
feed_h = _ToyFeed(initial_bars=2, start_delay_seconds=0.2, bar_interval_seconds=0.05)
strategy_h = _DummyStrategy()
engine = LiveEngine(
    strategy=strategy_h,
    broker=broker_h,
    feed=feed_h,
    feed_silence_seconds=1.0,
    watchdog_poll_seconds=0.25,
)


# %% [markdown]
# `observe_engine` polls `runtime_status()` within a bounded window and records
# each expected health transition when it occurs.


# %%
async def observe_engine() -> list[tuple[float, str]]:
    """Run the engine until each expected health transition occurs."""
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    deadline = started_at + HEALTH_OBSERVATION_SECONDS
    transitions = [(0.0, engine.runtime_status()["health"])]

    async def wait_for_health(expected: str) -> None:
        while loop.time() < deadline:
            health = engine.runtime_status()["health"]
            if health == expected:
                transitions.append((round(loop.time() - started_at, 2), health))
                return
            await asyncio.sleep(0.01)
        observed = [health for _, health in transitions]
        raise AssertionError(f"expected health {expected!r}; observed {observed}")

    await engine.connect()
    run_task = asyncio.create_task(engine.run())
    for expected in ("waiting_for_data", "ok", "feed_silent"):
        await wait_for_health(expected)
    await engine.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    transitions.append((round(loop.time() - started_at, 2), engine.runtime_status()["health"]))
    return transitions


# %%
timeline = run_demo(observe_engine())

# %%
last = None
transitions: list[dict] = []
for t, health in timeline:
    if health != last:
        transitions.append({"t_seconds": round(t, 2), "health": health})
        last = health
health_timeline = pl.DataFrame(transitions)
observed_health = health_timeline["health"].to_list()
assert observed_health == [
    "stopped",
    "waiting_for_data",
    "ok",
    "feed_silent",
    "stopped",
], observed_health
health_timeline

# %% [markdown]
# **Finding**: the printed sequence dedupes consecutive identical health states, so the
# observed output is `stopped → waiting_for_data → ok → feed_silent → stopped`.
# A short initial delay makes the waiting state observable, and the feed's continuous-market
# contract removes equity-session dependence. None of the transitions require a real venue.
#
# **Trading implication**: `runtime_status()` is what a watchdog or supervisor process reads.
# The categories are deliberately narrow because operators need to act on them under stress —
# `feed_silent` triggers a different response than `broker_disconnected` even though both
# look like "data stopped" from inside the strategy loop.

# %%
assert not list(STATE_DIR.iterdir()), list(STATE_DIR.iterdir())
STATE_DIR.rmdir()
assert not STATE_DIR.exists(), STATE_DIR
print("Temporary state artifacts: cleaned")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Stale-data rejection** prevents trading on prices the runtime no longer trusts. The
#    threshold is enforced per asset on every order intent.
# 2. **Daily-loss kill switch** latches in persisted state; reconstruction does not reset it,
#    which is what makes it useful as an emergency control.
# 3. **Startup reconciliation** surfaces every divergence between the persisted snapshot and
#    the broker's authoritative state. Production launchers refuse to start a new cycle
#    against a non-clean report.
# 4. **Engine health states** reduce the runtime to a small operator-facing vocabulary that a
#    supervisor process can consume directly.
#
# **Next**: NB10 covers the configurable risk surface; NB12 shows the same controls in a live
# IB paper basket-rebalance setting; §25.7 ties the abstract pre-flight requirements to the
# enforced runtime behaviours demonstrated here.
