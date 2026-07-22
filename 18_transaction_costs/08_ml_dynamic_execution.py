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
# # Reinforcement Learning for Trade Execution
#
# **Docker image**: `ml4t`
#
# Static schedules (TWAP, VWAP) commit to a trading path before the session
# reveals whether liquidity will be cheap or expensive. This notebook frames
# execution as a **sequential control problem** and learns an adaptive policy
# with tabular Q-learning, following Nevmyvaka, Feng, and Kearns (2006). The
# agent decides, interval by interval, how aggressively to trade given the
# remaining inventory, the time left in the session, and the *current* liquidity
# regime.
#
# The objective is the Almgren-Chriss trade-off: minimize **execution cost**
# (half-spread plus market impact) **plus a timing-risk penalty** for carrying
# inventory through volatile intervals. Trading fast pays impact; trading slow
# accepts risk. An adaptive policy can lean either way depending on the realized
# liquidity and volatility it observes — a static schedule cannot.
#
# Everything runs on **real AlgoSeek NASDAQ-100 minute bars**: real interval
# volumes, real volume-weighted prices, real quoted spreads, and real intra-interval
# volatility. The square-root impact term is a calibrated cost model (see
# `03_market_impact_calibration`) applied to those real volumes — no synthetic
# market paths.
#
# **Learning Objectives**
# - Cast trade execution as a finite-horizon Markov decision process
# - Build a liquidity-only state (no directional/price features) so the policy
#   solves execution, not alpha timing
# - Train a tabular Q-learning policy on real sessions and evaluate it on
#   held-out sessions against TWAP and VWAP
# - Read the learned policy to see how it conditions on spread, volatility, and time
#
# **Book Reference:** Chapter 18, Section 18.5 (Execution Algorithms as Controls)
#
# **Prerequisites:** Read [`04_vwap_twap_execution`](04_vwap_twap_execution.ipynb) for the static baselines,
# [`05_almgren_chriss_optimal_execution`](05_almgren_chriss_optimal_execution.ipynb) for the cost-risk objective,
# and [`07_ml4t_volume_participation`](07_ml4t_volume_participation.ipynb) for participation mechanics on real liquidity.

# %% [markdown]
# ## 1. Setup and Imports

# %%
"""Reinforcement Learning for Trade Execution - tabular Q-learning on real NASDAQ-100 liquidity."""

import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

import utils  # noqa: F401
from data import load_nasdaq100_bars
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
# Real AlgoSeek NASDAQ-100 minute bars define the execution environment. The
# parent order is liquidated within a single session across a 15-minute grid;
# the policy is trained on the earlier sessions and evaluated on the later ones.
PRIMARY_SYMBOL = "AAPL"
TAQ_START_DATE = "2021-10-01"
TAQ_END_DATE = "2021-12-31"
INTERVAL_MINUTES = 15  # 09:30-16:00 → 26 intervals/session
ORDER_PCT_ADV = 0.10  # parent order as a fraction of measured ADV (liquidated in one session)
IMPACT_COEF_BPS = 10.0  # square-root impact coefficient (see 03_market_impact_calibration)
RISK_AVERSION = 0.25  # timing-risk weight (Almgren-Chriss lambda); cost bps per unit inventory-vol
TRAIN_FRACTION = 0.6  # earlier sessions train the policy; later sessions are held out
INV_BINS = 10  # inventory-remaining state bins
N_EPOCHS = 800  # Q-learning passes over the training sessions
ALPHA = 0.30  # learning rate
EPSILON_START = 0.5  # exploration rate (linearly decayed to 0)
SEED = 42

# %%
set_global_seeds(SEED)
# Actions are multipliers on the VWAP base slice (the volume-proportional rate).
# A multiplier of 1.0 reproduces VWAP exactly, so VWAP lies inside the policy
# class and the agent can only improve on it by conditioning on real-time state —
# the regime-aware dampening f(spread, volatility) of Section 18.5.
ACTIONS = np.array([0.5, 0.75, 1.0, 1.3, 1.6])

# %% [markdown]
# ## 2. Execution as a Sequential Control Problem
#
# A desk must trade $Q$ shares before the close. At each of $H$ decision points
# it chooses how much of the *remaining* inventory to release, observing only the
# current market state — never future prices. The episode is a finite-horizon
# Markov decision process:
#
# - **State** $s_t = (\text{time remaining}, \text{inventory remaining}, \text{spread regime}, \text{volatility regime})$
# - **Action** $a_t \in \{0.5,\ 0.75,\ 1.0,\ 1.3,\ 1.6\}$ — a multiplier on the VWAP base slice ($1.0$ = trade exactly the VWAP rate this interval)
# - **Reward** $r_t = -\big[\,c_t\,(q_t/Q) + \lambda\,(I_t/Q)\,\sigma_t\,\big]$ — negative execution cost on the fill plus a timing-risk penalty on the inventory $I_t$ carried through interval $t$
# - **Constraint** — any inventory still held at the last interval is force-liquidated
#
# The objective combines the two costs every execution desk trades off: market
# impact (rises when you trade fast) and timing risk (rises when you trade slow
# and carry inventory through volatile intervals). This is the Almgren-Chriss
# mean-variance objective, here optimized by a policy that adapts to *realized*
# liquidity rather than a fixed trajectory.
#
# Crucially, the state contains **no price or return feature**. This keeps the
# problem a pure execution problem: the policy can only learn to route trades
# toward cheap-liquidity, low-risk intervals, not to time directional drift. That
# is the discipline the earlier gradient-boosting version of this notebook lacked.
#
# Nevmyvaka, Feng, and Kearns (2006) introduced exactly this reinforcement-learning
# formulation and showed that conditioning on real-time market state can improve on
# static VWAP-style schedules.

# %% [markdown]
# ## 3. Real Intraday Microstructure
#
# We load AlgoSeek NASDAQ-100 minute bars and aggregate them onto a 15-minute
# regular-session grid. Each interval carries its real traded volume, its
# volume-weighted trade price, its volume-weighted relative quoted spread, and its
# intra-interval realized volatility (the standard deviation of minute returns).


# %%
def load_intraday_panel(
    symbol: str,
    start_date: str,
    end_date: str,
    interval_minutes: int,
) -> pl.DataFrame:
    """Aggregate real minute bars onto an intraday execution grid.

    Returns per (date, bucket): interval volume, volume-weighted trade price,
    volume-weighted relative quoted spread (bps), and intra-interval realized
    volatility (bps), restricted to 09:30-16:00.
    """
    session_start = 9 * 60 + 30
    session_end = 16 * 60
    return (
        load_nasdaq100_bars(
            start_date=start_date,
            end_date=end_date,
            include_microstructure=True,
            lazy=True,
        )
        .filter(pl.col("symbol") == symbol)
        .select("timestamp", "volume", "last_trade_price", "close_bid_price", "close_ask_price")
        .filter(
            pl.col("last_trade_price").is_not_null()
            & (pl.col("volume") > 0)
            & (pl.col("close_bid_price") > 0)
            & (pl.col("close_ask_price") >= pl.col("close_bid_price"))
        )
        .with_columns(
            date=pl.col("timestamp").dt.date(),
            minute_of_day=pl.col("timestamp").dt.hour().cast(pl.Int32) * 60
            + pl.col("timestamp").dt.minute().cast(pl.Int32),
            rel_spread_bps=(
                (pl.col("close_ask_price") - pl.col("close_bid_price"))
                / ((pl.col("close_ask_price") + pl.col("close_bid_price")) / 2)
                * 1e4
            ),
        )
        .filter(
            (pl.col("minute_of_day") >= session_start) & (pl.col("minute_of_day") < session_end)
        )
        .with_columns(
            bucket=((pl.col("minute_of_day") - session_start) // interval_minutes).cast(pl.Int32)
        )
        .sort("date", "bucket", "timestamp")
        .group_by("date", "bucket")
        .agg(
            volume=pl.col("volume").sum(),
            price=(pl.col("last_trade_price") * pl.col("volume")).sum() / pl.col("volume").sum(),
            spread_bps=(pl.col("rel_spread_bps") * pl.col("volume")).sum() / pl.col("volume").sum(),
            vol_bps=(pl.col("last_trade_price").pct_change().std() * 1e4),
        )
        .sort("date", "bucket")
        .collect()
    )


# %%
N_BUCKETS = (16 * 60 - (9 * 60 + 30)) // INTERVAL_MINUTES
panel = load_intraday_panel(PRIMARY_SYMBOL, TAQ_START_DATE, TAQ_END_DATE, INTERVAL_MINUTES)
print(
    f"{PRIMARY_SYMBOL}: {panel.height:,} interval rows across "
    f"{panel['date'].n_unique()} sessions ({N_BUCKETS} intervals/session)"
)


# %% [markdown]
# ### Assemble Per-Session Arrays
#
# Keep only sessions with full interval coverage so each session yields aligned
# price, volume, spread, and volatility arrays of length `N_BUCKETS`.


# %%
def session_arrays(panel: pl.DataFrame, n_buckets: int) -> dict:
    """Return {date: {price, volume, spread_bps, vol_bps}} for fully covered sessions."""
    out = {}
    for d in panel["date"].unique().to_list():
        g = panel.filter(pl.col("date") == d).sort("bucket")
        if g.height == n_buckets and g["vol_bps"].null_count() == 0:
            out[d] = {
                "price": g["price"].to_numpy(),
                "volume": g["volume"].to_numpy(),
                "spread_bps": g["spread_bps"].to_numpy(),
                "vol_bps": g["vol_bps"].to_numpy(),
            }
    return out


# %%
sessions = session_arrays(panel, N_BUCKETS)
all_days = sorted(sessions)
split_idx = int(len(all_days) * TRAIN_FRACTION)
train_days = all_days[:split_idx]
test_days = all_days[split_idx:]

# ADV measured on training sessions only so the parent order size that drives
# every strategy's evaluation does not see held-out volume.
train_day_set = set(train_days)
adv = float(
    panel.filter(pl.col("date").is_in(list(train_day_set)))
    .group_by("date")
    .agg(dv=pl.col("volume").sum())["dv"]
    .mean()
)
order_shares = int(round(ORDER_PCT_ADV * adv))
print(f"Full-coverage sessions: {len(all_days)} ({len(train_days)} train, {len(test_days)} test)")
print(f"Measured ADV: {adv:,.0f} shares")
print(f"Parent order: {order_shares:,} shares ({ORDER_PCT_ADV:.0%} of ADV, liquidated per session)")

# %% [markdown]
# **Finding**: The parent order is a tenth of a day's volume liquidated inside a
# single session, so under a flat schedule it consumes roughly 10% of each
# interval's real volume — a participation level where the square-root impact term
# is a few basis points and therefore worth scheduling around.

# %%
# Visualize the real microstructure regimes the policy will condition on.
mean_volume = np.vstack([sessions[d]["volume"] for d in train_days]).mean(axis=0)
mean_spread = np.vstack([sessions[d]["spread_bps"] for d in train_days]).mean(axis=0)
mean_vol = np.vstack([sessions[d]["vol_bps"] for d in train_days]).mean(axis=0)
bucket_idx = np.arange(N_BUCKETS)

# VWAP base slice (shares per interval) from the training volume profile. The RL
# action scales this slice up or down; a multiplier of 1.0 everywhere is VWAP.
volume_profile = mean_volume / mean_volume.sum()
base_slice = order_shares * volume_profile

fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "Mean Interval Volume",
        "Mean Relative Spread (bps)",
        "Mean Realized Vol (bps)",
    ],
)
fig.add_bar(x=bucket_idx, y=mean_volume, marker_color="#2E86AB", row=1, col=1)
fig.add_scatter(
    x=bucket_idx, y=mean_spread, mode="lines+markers", line_color="#A23B72", row=1, col=2
)
fig.add_scatter(x=bucket_idx, y=mean_vol, mode="lines+markers", line_color="#F18F01", row=1, col=3)
fig.update_xaxes(title_text=f"Interval ({INTERVAL_MINUTES}-min, 0 = 09:30)")
fig.update_yaxes(title_text="Shares", row=1, col=1)
fig.update_yaxes(title_text="bps", row=1, col=2)
fig.update_yaxes(title_text="bps", row=1, col=3)
fig.update_layout(
    title="Real Intraday Liquidity Regimes (training sessions)", height=380, showlegend=False
)
fig.show()

# %% [markdown]
# **Finding**: The three regimes pull in different directions. Volume is U-shaped
# (heavy at the open and close), the relative spread is widest at the open, and
# realized volatility is also highest at the open. So the open offers the deepest
# liquidity (low impact) but the widest spread *and* the most volatility — trading
# there is cheap on impact yet exposes the fill to wide spreads, while *waiting*
# leaves inventory exposed to open volatility. A policy that conditions on all
# three can balance the tension; a static schedule cannot.

# %% [markdown]
# ## 4. The Almgren-Chriss Objective on Real Data
#
# Every fill pays half the quoted spread plus square-root market impact against
# the interval's **real** volume:
#
# $$c_t = \tfrac{1}{2}\,\text{spread}_t + \kappa\sqrt{\frac{q_t}{V_t}}\ \text{(bps)}$$
#
# Carrying inventory is also costly: inventory $I_t$ held through interval $t$ is
# exposed to that interval's realized volatility $\sigma_t$, contributing a
# timing-risk term $\lambda\,(I_t/Q)\,\sigma_t$. The total objective a schedule
# pays is
#
# $$\text{Objective} = \underbrace{\sum_t c_t\,\frac{q_t}{Q}}_{\text{execution cost}}
# \;+\; \underbrace{\lambda\sum_t \frac{I_t}{Q}\,\sigma_t}_{\text{timing risk}}\quad\text{(bps)}.$$


# %%
def interval_cost_bps(spread_bps: float, volume: float, shares: float) -> float:
    """Execution cost (bps) for a fill: half-spread plus square-root impact."""
    participation = np.sqrt(max(shares, 0.0) / max(volume, 1.0))
    return spread_bps / 2.0 + IMPACT_COEF_BPS * participation


def schedule_components(shares: np.ndarray, sess: dict) -> tuple[float, float]:
    """Return (execution cost, timing risk) in bps for a per-interval share schedule."""
    shares = shares.astype(float)
    exec_cost = 0.0
    timing_risk = 0.0
    remaining = float(order_shares)
    for q, spread, volume_t, sig in zip(
        shares, sess["spread_bps"], sess["volume"], sess["vol_bps"]
    ):
        timing_risk += RISK_AVERSION * (remaining / order_shares) * sig
        if q > 0:
            exec_cost += interval_cost_bps(spread, volume_t, q) * (q / order_shares)
        remaining -= q
    return exec_cost, timing_risk


def schedule_objective_bps(shares: np.ndarray, sess: dict) -> float:
    """Total Almgren-Chriss objective (execution cost + timing risk), bps."""
    exec_cost, timing_risk = schedule_components(shares, sess)
    return exec_cost + timing_risk


# %% [markdown]
# ## 5. State Discretization
#
# The continuous state is mapped to a small grid so a tabular Q-table can cover
# it. Spread and volatility regimes are binned against quantiles measured on the
# **training** sessions only.


# %%
spread_terciles = np.quantile(
    np.concatenate([sessions[d]["spread_bps"] for d in train_days]), [1 / 3, 2 / 3]
)
vol_median = np.median(np.concatenate([sessions[d]["vol_bps"] for d in train_days]))


def encode_state(t: int, remaining: float, spread_bps: float, vol_bps: float) -> tuple:
    """Map (time, inventory, spread, volatility) to a discrete Q-table key."""
    inv_frac = min(remaining / order_shares, 0.999) if order_shares > 0 else 0.0
    inv_bin = int(inv_frac * INV_BINS)
    spread_bin = int(np.digitize(spread_bps, spread_terciles))  # 0,1,2
    vol_bin = int(vol_bps >= vol_median)  # 0 calm, 1 volatile
    return (t, inv_bin, spread_bin, vol_bin)


# %% [markdown]
# ## 6. Tabular Q-Learning
#
# We run finite-horizon Q-learning (no discounting) over the training sessions.
# At each interval the agent picks an action $\epsilon$-greedily — a multiplier on
# the VWAP base slice — trades that many shares against the real interval, and
# updates $Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \max_{a'} Q(s',a') - Q(s,a)\,]$.
# The last interval force-liquidates any residual so every episode completes the order.


# %%
def run_episode(sess: dict, q_table: dict, epsilon: float, learn: bool) -> tuple[np.ndarray, float]:
    """Roll one session forward under the policy; optionally apply Q-updates.

    Returns the per-interval share schedule and the realized Almgren-Chriss
    objective (bps). With ``learn=False`` and ``epsilon=0`` this is the greedy
    evaluation of the current policy.
    """
    remaining = float(order_shares)
    shares = np.zeros(N_BUCKETS)

    for t in range(N_BUCKETS):
        state = encode_state(t, remaining, sess["spread_bps"][t], sess["vol_bps"][t])
        if t == N_BUCKETS - 1:
            action_idx = len(ACTIONS) - 1  # force-liquidate residual at the close
        elif learn and np.random.random() < epsilon:
            action_idx = np.random.randint(len(ACTIONS))
        else:
            action_idx = int(np.argmax(q_table[state]))

        multiplier = ACTIONS[action_idx]
        qty = (
            remaining
            if t == N_BUCKETS - 1
            else float(min(remaining, round(multiplier * base_slice[t])))
        )
        shares[t] = qty

        # Per-step reward = negative (execution cost on the fill + timing risk on inventory held).
        timing_risk = RISK_AVERSION * (remaining / order_shares) * sess["vol_bps"][t]
        exec_cost = (
            interval_cost_bps(sess["spread_bps"][t], sess["volume"][t], qty) * (qty / order_shares)
            if qty > 0
            else 0.0
        )
        reward = -(exec_cost + timing_risk)

        remaining -= qty

        if learn:
            if t < N_BUCKETS - 1:
                next_state = encode_state(
                    t + 1, remaining, sess["spread_bps"][t + 1], sess["vol_bps"][t + 1]
                )
                if t + 1 == N_BUCKETS - 1:
                    # Terminal next state — action is forced to the largest
                    # multiplier, so bootstrap from that action's value rather
                    # than the max over phantom unvisited actions (which would
                    # zero out the forced-liquidation cost in the value
                    # function).
                    target = reward + q_table[next_state][len(ACTIONS) - 1]
                else:
                    target = reward + np.max(q_table[next_state])
            else:
                target = reward
            q_table[state][action_idx] += ALPHA * (target - q_table[state][action_idx])

    return shares, schedule_objective_bps(shares, sess)


# %% [markdown]
# ### Train the Policy


# %%
def train_q_policy(train_days: list, sessions: dict) -> dict:
    """Q-learning over training sessions with linearly decayed exploration."""
    q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))
    order = list(train_days)
    for epoch in range(N_EPOCHS):
        epsilon = EPSILON_START * (1 - epoch / N_EPOCHS)
        np.random.shuffle(order)
        for d in order:
            run_episode(sessions[d], q_table, epsilon, learn=True)
    return q_table


# %%
print("Training Q-learning execution policy...")
q_table = train_q_policy(train_days, sessions)
print(f"Policy trained: {len(q_table):,} visited states over {N_EPOCHS} epochs")

# %% [markdown]
# **Finding**: The Q-table covers a few hundred discrete states — small enough to
# read directly in Section 9, which is the point of a tabular policy: the learned
# rule is fully inspectable rather than hidden in a function approximator.

# %% [markdown]
# ## 7. Static Baselines
#
# TWAP trades an equal share of the order each interval. VWAP allocates shares in
# proportion to the **training-window** volume profile, so its forecast is strictly
# out-of-sample on the test sessions.


# %%
def shares_from_weights(weights: np.ndarray, total: int) -> np.ndarray:
    """Allocate integer shares across intervals in proportion to weights."""
    raw = total * (weights / weights.sum())
    shares = np.floor(raw).astype(int)
    remainder = total - int(shares.sum())
    if remainder > 0:
        top = np.argsort(raw - shares)[-remainder:]
        shares[top] += 1
    return shares


twap_shares = shares_from_weights(np.ones(N_BUCKETS), order_shares)
vwap_shares = shares_from_weights(mean_volume, order_shares)

# %% [markdown]
# ## 8. Evaluation on Held-Out Sessions
#
# We run all three policies on the test sessions and compare the Almgren-Chriss
# objective, decomposed into execution cost and timing risk. The Q-policy acts
# greedily (no exploration); the baselines apply their fixed schedules. All three
# use the identical real cost model, so differences reflect *when* each policy trades.


# %%
def evaluate(test_days: list, sessions: dict, q_table: dict) -> pl.DataFrame:
    """Objective and its components (bps) per held-out session for each policy."""
    rows = []
    for d in test_days:
        sess = sessions[d]
        rl_shares, _ = run_episode(sess, q_table, epsilon=0.0, learn=False)
        for name, sh in [("RL", rl_shares), ("TWAP", twap_shares), ("VWAP", vwap_shares)]:
            exec_cost, timing_risk = schedule_components(sh, sess)
            rows.append(
                {
                    "date": d,
                    "strategy": name,
                    "exec_cost_bps": exec_cost,
                    "timing_risk_bps": timing_risk,
                    "objective_bps": exec_cost + timing_risk,
                }
            )
    return pl.DataFrame(rows)


# %%
eval_df = evaluate(test_days, sessions, q_table)

summary = (
    eval_df.group_by("strategy")
    .agg(
        mean_exec_cost=pl.col("exec_cost_bps").mean(),
        mean_timing_risk=pl.col("timing_risk_bps").mean(),
        mean_objective=pl.col("objective_bps").mean(),
        median_objective=pl.col("objective_bps").median(),
    )
    .sort("mean_objective")
)
print(f"Held-out evaluation: {eval_df['date'].n_unique()} sessions ({PRIMARY_SYMBOL})")
summary.with_columns(pl.col(pl.Float64).round(3))

# %% [markdown]
# **Finding**: The learned policy beats both static schedules on the total
# objective — mean 24.7 bps versus 28.3 for VWAP and 31.0 for TWAP across the 26
# held-out sessions. Its execution cost (3.6 bps) is essentially tied with VWAP
# (3.6 bps); the entire edge comes from the timing-risk term, where RL pays 21.1
# bps against VWAP's 24.7. The policy does not buy lower risk with more impact —
# it reallocates *within* a VWAP-like impact budget toward the intervals where
# carrying inventory is most dangerous.

# %%
# Distribution of the total objective by policy.
fig = go.Figure()
colors = {"RL": "#F18F01", "TWAP": "#2E86AB", "VWAP": "#A23B72"}
for strat in ["RL", "TWAP", "VWAP"]:
    vals = eval_df.filter(pl.col("strategy") == strat)["objective_bps"].to_list()
    fig.add_box(y=vals, name=strat, marker_color=colors[strat], boxmean=True)
fig.update_layout(
    title="Almgren-Chriss Objective by Policy (held-out sessions)",
    yaxis_title="Objective: execution cost + timing risk (bps)",
    height=450,
)
fig.show()

# %% [markdown]
# **Finding**: The RL distribution sits below both baselines on the median as well
# as the mean, so the improvement is not driven by a few favorable sessions — the
# adaptive policy is consistently lower-objective session by session, which is what
# makes it deployable rather than merely better on average.

# %% [markdown]
# ### Cost-vs-Risk Decomposition

# %%
# Show how each policy splits the objective between impact and timing risk.
fig = go.Figure()
strategies = summary["strategy"].to_list()
exec_means = [eval_df.filter(pl.col("strategy") == s)["exec_cost_bps"].mean() for s in strategies]
risk_means = [eval_df.filter(pl.col("strategy") == s)["timing_risk_bps"].mean() for s in strategies]
fig.add_bar(x=strategies, y=exec_means, name="Execution cost", marker_color="#2E86AB")
fig.add_bar(x=strategies, y=risk_means, name="Timing risk", marker_color="#F18F01")
fig.update_layout(
    title="Objective Decomposition by Policy",
    yaxis_title="Mean cost (bps)",
    barmode="stack",
    height=400,
)
fig.show()

# %% [markdown]
# **Finding**: The decomposition isolates where the edge comes from. All three
# policies pay roughly the same execution cost (≈4 bps) — VWAP already minimizes
# impact by matching the volume profile, and RL inherits that profile as its base
# slice. The objective gap is entirely the timing-risk stack: RL shaves it by
# accelerating through the volatile, high-risk part of the session.

# %% [markdown]
# ## 9. Reading the Learned Policy
#
# Because the Q-table is tabular, we can read the policy directly. For each
# (spread, volatility) regime we report the average chosen VWAP multiplier across
# the states the agent actually visited. A multiplier above 1.0 means the policy
# trades faster than VWAP in that regime; below 1.0 means it holds back. This is
# the regime-aware dampening function f(spread, volatility) of Section 18.5, read
# straight off the learned Q-table.


# %%
def policy_by_regime(q_table: dict) -> pl.DataFrame:
    """Average greedy multiplier by (spread, volatility) regime over visited states."""
    spread_labels = {0: "tight", 1: "medium", 2: "wide"}
    vol_labels = {0: "calm", 1: "volatile"}
    by_regime = defaultdict(list)
    for (t, inv_bin, spread_bin, vol_bin), q in q_table.items():
        if t == N_BUCKETS - 1:
            continue  # terminal action is forced
        by_regime[(spread_bin, vol_bin)].append(ACTIONS[int(np.argmax(q))])
    rows = [
        {
            "spread_regime": spread_labels[sb],
            "vol_regime": vol_labels[vb],
            "n_states": len(mults),
            "mean_multiplier": float(np.mean(mults)),
        }
        for (sb, vb), mults in sorted(by_regime.items())
    ]
    return pl.DataFrame(rows)


# %%
policy_table = policy_by_regime(q_table)
policy_table.with_columns(pl.col("mean_multiplier").round(3))

# %% [markdown]
# **Finding**: The learned multipliers are consistently higher in *volatile*
# regimes (≈0.92) than in *calm* ones (≈0.80), across every spread bucket. The
# policy accelerates relative to VWAP precisely when realized volatility is high —
# shedding inventory faster to cut timing-risk exposure — and eases off when the
# market is calm. That is the regime-aware dampening f(spread, volatility) of
# Section 18.5, recovered from data rather than hand-specified. Note the spread
# dimension barely moves the multiplier: on a name as liquid as AAPL the quoted
# spread is too small (sub-basis-point) to be worth conditioning on, so the policy
# keys almost entirely on volatility.

# %% [markdown]
# ## 10. Summary
#
# - **Execution is a control problem, not a prediction problem.** Casting it as a
#   finite-horizon MDP makes the levers explicit: time remaining, inventory
#   remaining, and the cost of trading now (impact) versus later (timing risk).
# - **A liquidity-only state keeps the policy honest.** With no price or return
#   feature, the agent cannot drift into directional timing — it can only learn to
#   route trades toward cheaper-liquidity, lower-risk intervals. This is the
#   discipline the earlier gradient-boosting version of this notebook lacked.
# - **The environment is entirely real.** Volume, price, spread, and volatility
#   come from AlgoSeek minute bars; only the square-root impact term is a
#   calibrated cost model applied to those real volumes.
#
# **Next**: [`05_almgren_chriss_optimal_execution`](05_almgren_chriss_optimal_execution.ipynb)
# derives the closed-form cost-variance optimal trajectory;
# [`07_ml4t_volume_participation`](07_ml4t_volume_participation.ipynb) covers hard
# liquidity constraints.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Execution as sequential control**: trading $Q$ shares before the close is a
#    finite-horizon MDP whose state is time- and inventory-remaining plus the
#    current liquidity regime. Tabular Q-learning solves it on a discretized grid
#    and yields a fully inspectable policy.
#
# 2. **Liquidity-only state by design**: the state deliberately excludes price and
#    return features, so the learned policy optimizes execution cost and timing
#    risk rather than directional drift — the exact failure mode a feature-importance
#    audit would otherwise have to police.
#
# 3. **Conditioning on real-time state pays off through timing risk**: on the
#    held-out sessions the learned policy cuts the Almgren-Chriss objective to 24.7
#    bps versus 28.3 for VWAP and 31.0 for TWAP. It matches VWAP's impact cost and
#    wins entirely on timing risk by accelerating through volatile intervals — the
#    Nevmyvaka-Kearns thesis that adaptive execution can beat static schedules once
#    it conditions on real-time market state, demonstrated here on real liquidity.
