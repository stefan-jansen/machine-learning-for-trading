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
# # Circuit Breakers for Trading Systems
#
# **Chapter 26: MLOps and Governance**
# **Docker image**: `ml4t`
# **Book Reference**: Chapter 26, Section 26.5
# **Prerequisites**: Chapter 25 deployment verification and the Chapter 26 monitoring sections.
#
# **Learning Objectives**:
# - Implement a shared CLOSED/OPEN/HALF_OPEN state machine that several
#   independent breakers can plug into.
# - Combine four representative breakers (drawdown, daily loss, consecutive
#   loss, latency) into a single halt decision and audit log.
# - Drive the demo with real SPY 2020 returns so the market-stress breakers
#   trip on a recognized real-world episode, not synthetic returns.
#
# This notebook demonstrates **four representative circuit breakers** — a
# drawdown breaker on the equity curve, a daily-loss breaker, a consecutive-loss
# breaker, and an infrastructure latency breaker. The chapter §26.5 discussion
# of additional categories (weekly drawdown, position-size, sector, volatility,
# spread, intraday-move) extends the same state-machine pattern; only these
# four are implemented here.

# %%
"""Circuit Breakers for Trading Systems — implement multi-level circuit breakers for production trading systems."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, add_message_title

warnings.filterwarnings("ignore")


# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
N_STEPS = 100
SEED = 42

# %%
set_global_seeds(SEED)


# %% [markdown]
# **Setup**: The notebook uses a compact simulation with deterministic seeds so
# each breaker's state transition stays easy to inspect.

# %% [markdown]
# ## 1. Circuit Breaker States and Core Classes


# %%
class BreakerState(Enum):
    """Circuit breaker state."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Halted - no trading
    HALF_OPEN = auto()  # Testing - limited trading


# %% [markdown]
# Breaker events are the audit trail. Each state change records the reason and
# the threshold that triggered it.


# %%
@dataclass
class BreakerEvent:
    """Event that triggered breaker."""

    timestamp: datetime
    breaker_name: str
    old_state: BreakerState
    new_state: BreakerState
    reason: str
    value: float | None = None
    threshold: float | None = None


# %% [markdown]
# The base circuit breaker implements the shared state machine. Specific
# breakers only need to define their own trip condition.


# %%
def transition_breaker(
    breaker,
    new_state: BreakerState,
    reason: str,
    value: float | None = None,
    threshold: float | None = None,
    event_time: datetime | None = None,
) -> None:
    event_time = event_time or datetime.now()
    event = BreakerEvent(
        timestamp=event_time,
        breaker_name=breaker.name,
        old_state=breaker.state,
        new_state=new_state,
        reason=reason,
        value=value,
        threshold=threshold,
    )
    breaker.history.append(event)
    breaker.state = new_state
    if new_state == BreakerState.OPEN and event.old_state != BreakerState.OPEN:
        breaker.trip_count += 1
        breaker.trip_time = event_time
    # Audit-trail callback fires on every transition (CLOSED↔OPEN, OPEN→HALF_OPEN,
    # HALF_OPEN↔CLOSED/OPEN). on_trip fires only when the transition lands in OPEN
    # so manager-level alerting stays specific to trips.
    if breaker.on_transition:
        breaker.on_transition(event)
    if breaker.on_trip and new_state == BreakerState.OPEN:
        breaker.on_trip(event)


# %% [markdown]
# Keep the transition logic outside the class so each concrete breaker can
# inherit a compact, readable state machine.


# %%
def advance_breaker_state(
    breaker, event_time: datetime | None = None, **kwargs: object
) -> BreakerState:
    event_time = event_time or datetime.now()
    if breaker.state == BreakerState.OPEN:
        if breaker.trip_time and event_time - breaker.trip_time >= breaker.recovery_timeout:
            transition_breaker(
                breaker, BreakerState.HALF_OPEN, "Recovery timeout elapsed", event_time=event_time
            )
        return breaker.state

    should_trip, reason, value, threshold = breaker.check_condition(**kwargs)
    if should_trip:
        if breaker.state == BreakerState.HALF_OPEN:
            transition_breaker(
                breaker,
                BreakerState.OPEN,
                f"Recovery failed: {reason}",
                value,
                threshold,
                event_time,
            )
        else:
            transition_breaker(breaker, BreakerState.OPEN, reason, value, threshold, event_time)
    elif breaker.state == BreakerState.HALF_OPEN:
        transition_breaker(
            breaker, BreakerState.CLOSED, "Recovery successful", event_time=event_time
        )
    return breaker.state


# %% [markdown]
# The abstract base class now delegates the mechanics to helper functions and
# keeps only the public interface shared across breaker types.


# %%
class CircuitBreaker(ABC):
    def __init__(
        self,
        name: str,
        recovery_timeout: timedelta = timedelta(hours=1),
        on_trip: Callable[[BreakerEvent], None] | None = None,
        on_transition: Callable[[BreakerEvent], None] | None = None,
    ):
        self.name = name
        self.recovery_timeout = recovery_timeout
        self.on_trip = on_trip
        self.on_transition = on_transition

        self.state = BreakerState.CLOSED
        self.trip_time: datetime | None = None
        self.trip_count = 0
        self.history: list[BreakerEvent] = []

    @abstractmethod
    def check_condition(
        self, portfolio_value: float | None = None, **kwargs: object
    ) -> tuple[bool, str, float | None, float | None]:
        pass

    def update(self, event_time: datetime | None = None, **kwargs: object) -> BreakerState:
        return advance_breaker_state(self, event_time=event_time, **kwargs)

    def _transition(self, new_state, reason, value=None, threshold=None, event_time=None):
        transition_breaker(self, new_state, reason, value, threshold, event_time)

    def reset(self, event_time: datetime | None = None):
        """Manually reset breaker to CLOSED."""
        self._transition(BreakerState.CLOSED, "Manual reset", event_time=event_time)
        self.trip_time = None

    def is_open(self) -> bool:
        """Check if trading is halted."""
        return self.state == BreakerState.OPEN

    def allows_trading(self) -> bool:
        """Check if trading is allowed."""
        return self.state in [BreakerState.CLOSED, BreakerState.HALF_OPEN]


# %% [markdown]
# **Finding**: The shared `CLOSED -> OPEN -> HALF_OPEN` state machine is what
# makes layered breakers operationally manageable. Once that lifecycle is
# standardized, each risk rule becomes a small condition rather than a separate
# control framework.

# %% [markdown]
# ## 2. Specific Circuit Breaker Implementations
#
# The drawdown breaker trips when the running peak-to-current decline exceeds a
# threshold:
#
# $$DD_t = 1 - \frac{V_t}{\max_{u \le t} V_u}$$
#
# where $V_t$ is portfolio value at step $t$.


# %%
class DrawdownBreaker(CircuitBreaker):
    """Trips when drawdown from the running peak exceeds a threshold."""

    def __init__(
        self,
        name: str,
        max_drawdown: float = 0.10,  # 10%
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.max_drawdown = max_drawdown
        self.peak_value = 0

    def check_condition(
        self, portfolio_value: float | None = None, **kwargs: object
    ) -> tuple[bool, str, float | None, float | None]:
        if portfolio_value is None:
            return False, "", None, None

        # Update peak
        self.peak_value = max(self.peak_value, portfolio_value)

        if self.peak_value == 0:
            return False, "", None, None

        drawdown = (self.peak_value - portfolio_value) / self.peak_value

        if drawdown >= self.max_drawdown:
            return True, f"Drawdown {drawdown:.1%} exceeds limit", drawdown, self.max_drawdown

        return False, "", drawdown, self.max_drawdown


# %% [markdown]
# Daily loss controls and consecutive-loss controls catch different failure
# modes: a sharp intraday shock versus a strategy repeatedly making poor bets.


# %%
class DailyLossBreaker(CircuitBreaker):
    """
    Trips when daily loss exceeds threshold.
    """

    def __init__(
        self,
        name: str,
        max_daily_loss: float = 0.02,  # 2%
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.max_daily_loss = max_daily_loss
        self.start_of_day_value: float | None = None

    def reset_day(self, current_value: float):
        """Call at start of trading day."""
        self.start_of_day_value = current_value

    def check_condition(
        self, portfolio_value: float | None = None, **kwargs: object
    ) -> tuple[bool, str, float | None, float | None]:
        if portfolio_value is None:
            return False, "", None, None

        if self.start_of_day_value is None:
            self.start_of_day_value = portfolio_value
            return False, "", None, None

        daily_pnl = (portfolio_value - self.start_of_day_value) / self.start_of_day_value

        if daily_pnl <= -self.max_daily_loss:
            return (
                True,
                f"Daily loss {abs(daily_pnl):.1%} exceeds limit",
                daily_pnl,
                -self.max_daily_loss,
            )

        return False, "", daily_pnl, -self.max_daily_loss


# %% [markdown]
# Consecutive-loss logic is a lightweight proxy for strategy health when losses
# come from repeated bad predictions rather than one market shock.


# %%
class ConsecutiveLossBreaker(CircuitBreaker):
    """
    Trips after N consecutive losing trades.
    """

    def __init__(
        self,
        name: str,
        max_consecutive: int = 5,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.max_consecutive = max_consecutive
        self.consecutive_losses = 0

    def record_trade(self, pnl: float):
        """Record trade result."""
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def check_condition(
        self, portfolio_value: float | None = None, **kwargs: object
    ) -> tuple[bool, str, float | None, float | None]:
        if self.consecutive_losses >= self.max_consecutive:
            return (
                True,
                f"{self.consecutive_losses} consecutive losses",
                float(self.consecutive_losses),
                float(self.max_consecutive),
            )

        return False, "", float(self.consecutive_losses), float(self.max_consecutive)


# %% [markdown]
# Latency is an infrastructure breaker rather than a market-risk breaker, but it
# belongs in the same control plane because stale data can be just as dangerous
# as bad signals.


# %%
class LatencyBreaker(CircuitBreaker):
    """
    Trips when system latency exceeds threshold.
    """

    def __init__(
        self,
        name: str,
        max_latency_ms: float = 100,  # 100ms
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.max_latency_ms = max_latency_ms
        self.latency_history: list[float] = []

    def record_latency(self, latency_ms: float):
        """Record operation latency."""
        self.latency_history.append(latency_ms)
        # Keep last 100 measurements
        self.latency_history = self.latency_history[-100:]

    def check_condition(
        self, portfolio_value: float | None = None, **kwargs: object
    ) -> tuple[bool, str, float | None, float | None]:
        if not self.latency_history:
            return False, "", None, None

        avg_latency = np.mean(self.latency_history[-10:])  # Last 10

        if avg_latency > self.max_latency_ms:
            return (
                True,
                f"Latency {avg_latency:.1f}ms exceeds limit",
                avg_latency,
                self.max_latency_ms,
            )

        return False, "", avg_latency, self.max_latency_ms


# %% [markdown]
# **Finding**: These breaker types are intentionally simple. In production, the
# value comes from combining orthogonal rules rather than making any single rule
# overly sophisticated.

# %% [markdown]
# ## 3. Breaker Manager: Multi-Level Defense


# %%
def make_trip_callback(
    manager,
    breaker: CircuitBreaker,
) -> Callable[[BreakerEvent], None]:
    """Wrap the breaker's on_trip so manager-level trip alerts also fire."""
    original_callback = breaker.on_trip

    def wrapped_callback(event: BreakerEvent):
        if original_callback:
            original_callback(event)
        if manager.on_any_trip:
            manager.on_any_trip(event)

    return wrapped_callback


# %%
def make_transition_logger(manager) -> Callable[[BreakerEvent], None]:
    """Append every breaker transition to the manager's audit log."""

    def log_transition(event: BreakerEvent):
        manager.event_log.append(event)

    return log_transition


# %% [markdown]
# The manager is mostly a small coordination layer: registration, centralized
# status, and a single event log for all breaker trips.


# %%
class BreakerManager:
    """
    Manages multiple circuit breakers with hierarchical levels.

    If any breaker trips, trading is halted.
    """

    def __init__(self, on_any_trip: Callable[[BreakerEvent], None] | None = None):
        self.breakers: dict[str, CircuitBreaker] = {}
        self.on_any_trip = on_any_trip
        self.event_log: list[BreakerEvent] = []

    def add_breaker(self, breaker: CircuitBreaker):
        breaker.on_trip = make_trip_callback(self, breaker)
        breaker.on_transition = make_transition_logger(self)
        self.breakers[breaker.name] = breaker

    def check_all(self, **kwargs) -> bool:
        for breaker in self.breakers.values():
            breaker.update(**kwargs)
        return all(b.allows_trading() for b in self.breakers.values())

    def get_status(self) -> dict[str, dict]:
        """Get status of all breakers."""
        return {
            name: {
                "state": breaker.state.name,
                "trip_count": breaker.trip_count,
                "allows_trading": breaker.allows_trading(),
            }
            for name, breaker in self.breakers.items()
        }

    def reset_all(self, event_time: datetime | None = None):
        """Reset all breakers."""
        for breaker in self.breakers.values():
            breaker.reset(event_time=event_time)


# %% [markdown]
# The manager centralizes event logging and gives the trading engine one answer
# to the question that matters operationally: is trading still allowed?


# %%
announced_breakers: set[str] = set()


def alert_handler(event: BreakerEvent):
    """Emit the first trip per breaker; full detail lives in the event log."""
    if event.breaker_name in announced_breakers:
        return
    announced_breakers.add(event.breaker_name)
    detail = f" ({event.value:.4f} vs {event.threshold:.4f})" if event.value is not None else ""
    print(f"ALERT {event.breaker_name}: {event.reason}{detail}")


# %% [markdown]
# Configure one breaker for each risk layer so the later simulation can show how
# independent protections interact.

# %%
manager = BreakerManager(on_any_trip=alert_handler)

# Add breakers at different levels
manager.add_breaker(
    DrawdownBreaker(
        name="drawdown_10pct",
        max_drawdown=0.10,
        recovery_timeout=timedelta(hours=4),
    )
)

manager.add_breaker(
    DailyLossBreaker(
        name="daily_loss_2pct",
        max_daily_loss=0.02,
        recovery_timeout=timedelta(hours=1),
    )
)

manager.add_breaker(
    ConsecutiveLossBreaker(
        name="consecutive_5",
        max_consecutive=5,
        recovery_timeout=timedelta(minutes=30),
    )
)

manager.add_breaker(
    LatencyBreaker(
        name="latency_100ms",
        max_latency_ms=100,
        recovery_timeout=timedelta(minutes=5),
    )
)

print(f"Breaker Manager configured with {len(manager.breakers)} breakers.")


# %% [markdown]
# **Finding**: The manager setup makes the defense-in-depth design explicit.
# Each breaker guards a different failure mode, but all of them feed one halt
# decision and one event log.

# %% [markdown]
# ## 4. Simulation: Breaker Behavior

# %%
# Re-seed deterministically before the latency draws; market returns are real
# SPY data so the SEED only governs the synthetic latency stream below.
set_global_seeds(SEED)

initial_value = 100_000
portfolio_values = [initial_value]
trading_allowed = []
breaker_states = {name: [] for name in manager.breakers.keys()}


# %% [markdown]
# We drive the demo from real SPY daily returns over the first 100 trading days
# of 2020. That window includes the February–March COVID crash with multiple
# −2% (and worse) sessions, so the **market-stress** breakers (daily loss,
# consecutive losses) trip on a real episode rather than fabricated returns.
# Latency is an infrastructure event, not a market event — there is no real
# system-latency dataset in scope here — so the **latency stream remains
# synthetic** and exists solely to demonstrate the infrastructure breaker
# transitioning independently of the market path. The SPY series continues as
# a monitored counterfactual after a halt; it is not executed portfolio P&L.

# %%
covid_returns = (
    load_etfs()
    .filter(
        (pl.col("symbol") == "SPY") & (pl.col("timestamp") >= pl.lit("2020-01-02").str.to_date())
    )
    .sort("timestamp")
    .with_columns(pl.col("close").pct_change().alias("ret"))
    .drop_nulls("ret")
    .head(N_STEPS)
    .select("timestamp", "ret")
)
if covid_returns.height < N_STEPS:
    raise ValueError(f"SPY 2020 window returned {covid_returns.height} returns; need {N_STEPS}")

# %%
simulation_dates = covid_returns["timestamp"].to_list()
manager.reset_all(event_time=pd.Timestamp(simulation_dates[0]).to_pydatetime())
manager.event_log.clear()  # discard CLOSED→CLOSED reset transitions for a clean audit trail
daily_loss_breaker = cast(DailyLossBreaker, manager.breakers["daily_loss_2pct"])
consecutive_loss_breaker = cast(ConsecutiveLossBreaker, manager.breakers["consecutive_5"])
latency_breaker = cast(LatencyBreaker, manager.breakers["latency_100ms"])

n_steps = N_STEPS
real_returns = covid_returns["ret"].to_numpy()
for i in range(n_steps):
    event_time = pd.Timestamp(simulation_dates[i]).to_pydatetime()
    # Each iteration represents one trading day, so reset the daily-loss
    # baseline to the start-of-day value BEFORE applying the day's return.
    # Without this reset_day call the daily-loss breaker would measure
    # cumulative loss from inception (i.e. duplicate the drawdown breaker).
    start_of_day_value = portfolio_values[-1]
    daily_loss_breaker.reset_day(start_of_day_value)

    returns = float(real_returns[i])
    new_value = start_of_day_value * (1 + returns)
    portfolio_values.append(new_value)

    # Record trade P&L from the realized SPY change-on-equity
    trade_pnl = new_value - start_of_day_value
    consecutive_loss_breaker.record_trade(trade_pnl)

    # Latency stays synthetic: the infrastructure breaker needs a stressed
    # regime in the last 20 steps to exhibit its state transitions, and the
    # repo has no real system-latency series at this scope. The stressed
    # mean is set so the rolling-10 average crosses the 100 ms breaker
    # threshold inside the final 20 steps.
    latency = np.random.exponential(20) if i < 80 else np.random.exponential(150)
    latency_breaker.record_latency(latency)

    # Check all breakers
    can_trade = manager.check_all(portfolio_value=new_value, event_time=event_time)
    trading_allowed.append(can_trade)

    # Record states
    for name, breaker in manager.breakers.items():
        breaker_states[name].append(breaker.state.value)


# %%
print(f"\nSimulation complete: {n_steps} steps")
print(f"Final portfolio value: ${portfolio_values[-1]:,.2f}")
print(f"Trading halted {sum(not x for x in trading_allowed)} times")


# %% [markdown]
# **Finding**: The simulation separates market stress from infrastructure stress.
# That matters because the operational response is different even when both
# cases end with trading halted.

# %%
state_colors = {
    BreakerState.CLOSED.value: COLORS["positive"],
    BreakerState.OPEN.value: COLORS["negative"],
    BreakerState.HALF_OPEN.value: COLORS["amber"],
}

fig, axes = plt.subplots(3, 1, figsize=FIGSIZE["grid_3x2"], constrained_layout=True)

ax1 = axes[0]
ax1.plot(simulation_dates, portfolio_values[1:], color=COLORS["blue"], linewidth=1.5)
y_min = min(portfolio_values) * 0.985
y_max = max(portfolio_values) * 1.005
ax1.set_ylim(y_min, y_max)
ax1.fill_between(
    simulation_dates,
    portfolio_values[1:],
    y_min,
    where=[not allowed for allowed in trading_allowed],
    color=COLORS["negative"],
    alpha=0.18,
    label="Trading halted",
)
ax1.axhline(initial_value, color=COLORS["neutral"], linestyle="--", alpha=0.5)
ax1.axhline(initial_value * 0.98, color=COLORS["amber"], linestyle="--", label="Daily loss limit")
ax1.axhline(initial_value * 0.90, color=COLORS["negative"], linestyle="--", label="Drawdown limit")
ax1.set_ylabel("Counterfactual SPY value ($)")
add_message_title(ax1, "Independent breakers halt trading during the 2020 drawdown")
ax1.legend(loc="lower left")

ax2 = axes[1]
for y_pos, (name, states) in enumerate(breaker_states.items()):
    for i, s in enumerate(states):
        ax2.barh(
            y_pos,
            1,
            left=mdates.date2num(simulation_dates[i]),
            color=state_colors[s],
            height=0.8,
        )
ax2.set_yticks(range(len(breaker_states)))
ax2.set_yticklabels(list(breaker_states.keys()), fontsize=9)
ax2.set_xlim(simulation_dates[0], simulation_dates[-1])
ax2.set_ylim(-0.5, len(breaker_states) - 0.5)
ax2.set_xlabel("Monitoring date")
add_message_title(
    ax2,
    "Recovery probes expose repeated failures before breakers close",
    subtitle=(
        f"{BreakerState.CLOSED.name}=green, {BreakerState.OPEN.name}=red, "
        f"{BreakerState.HALF_OPEN.name}=amber"
    ),
)

ax3 = axes[2]
ax3.fill_between(
    simulation_dates,
    [1 if x else 0 for x in trading_allowed],
    step="mid",
    color=COLORS["positive"],
    alpha=0.5,
    label="Trading allowed",
)
ax3.fill_between(
    simulation_dates,
    [0 if x else 1 for x in trading_allowed],
    step="mid",
    color=COLORS["negative"],
    alpha=0.5,
    label="Trading halted",
)
ax3.set_xlim(simulation_dates[0], simulation_dates[-1])
ax3.set_ylim(-0.1, 1.1)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(["Halted", "Active"])
ax3.set_xlabel("Monitoring date")
add_message_title(ax3, "The combined control stays fail-closed while any breaker is open")
ax3.legend()

for ax in axes:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

fig.show()


# %% [markdown]
# **Trading implication**: The visual state timeline is the useful diagnostic,
# not just the final halt count. It shows which breaker tripped first and
# whether later breakers were independent confirmations or downstream effects.

# %% [markdown]
# ## 5. Final Status Report
#
# Simulation outcome at a glance:

# %%
final_return = portfolio_values[-1] / initial_value - 1
print(
    f"Simulation: {n_steps} steps | "
    f"initial ${initial_value:,.0f} -> final ${portfolio_values[-1]:,.0f} "
    f"({final_return:+.2%})"
)


# %% [markdown]
# ### Breaker status at end of simulation

# %%
breaker_status_df = pl.DataFrame(
    [
        {
            "breaker": name,
            "state": status["state"],
            "trips": status["trip_count"],
            "status": "Active" if status["allows_trading"] else "HALTED",
        }
        for name, status in manager.get_status().items()
    ]
)
breaker_status_df


# %% [markdown]
# ### Event log (last five transitions)

# %%
event_log_df = pl.DataFrame(
    [
        {
            "timestamp": event.timestamp.date().isoformat(),
            "breaker": event.breaker_name,
            "transition": f"{event.old_state.name} -> {event.new_state.name}",
            "reason": event.reason,
        }
        for event in manager.event_log[-5:]
    ]
)
event_log_df


# %% [markdown]
# **Finding**: The event log is the post-mortem artifact. Without it, operators
# know trading stopped but cannot reconstruct whether the root cause was losses,
# streak behavior, or infrastructure degradation.

# %% [markdown]
# ## Key Takeaways
#
# 1. A single CLOSED → OPEN → HALF_OPEN state machine, shared across breakers, gives the trading engine one halt decision and one audit log — each rule becomes a small `check_condition` override rather than its own framework.
# 2. The four breakers cover orthogonal failure modes (peak-to-trough drawdown, intraday loss, consecutive-loss streak, infrastructure latency); the daily-loss breaker must reset its baseline at the start of each session or it collapses into the drawdown rule.
# 3. The manager-level event log records **every** transition, not just trips — without that complete trail an operator can see trading stopped but cannot reconstruct whether the root cause was market loss, strategy degradation, or infrastructure stress.
#
# **Next**: Continue with [`05_feast_feature_store`](05_feast_feature_store.ipynb)
# to connect these safety controls to the data-governance layer that keeps
# training and serving inputs consistent.
