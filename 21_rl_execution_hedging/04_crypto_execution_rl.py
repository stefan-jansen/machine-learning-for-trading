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
# # Reinforcement Learning for Crypto Execution
#
# **Chapter 21: Reinforcement Learning for Execution and Hedging**
#
# ## Purpose
#
# The execution notebook before this one trained an agent inside a simulator.
# This one replaces the simulated price path with recorded hourly bars from a
# perpetual-futures venue, which changes what can and cannot be claimed. The
# prices are real, so the agent faces the volatility and the liquidity that
# actually occurred; the fills are still modelled, so nothing here is evidence
# about what an order would have cost.
#
# What recorded data buys is a set of state variables a simulator would have had
# to invent: the perpetual-spot basis, the hours remaining until the next
# funding settlement, and the volume that actually traded in the previous hour.
# The agent can condition on all three, and the diagnostics at the end ask
# whether it does.
#
# ## Learning objectives
#
# After working through this notebook you will be able to:
#
# - Build an hourly execution panel in which every feature a decision uses is
#   drawn from bars that had closed when the decision was made, and say which
#   column would leak if it were not lagged.
# - Explain what the perpetual-spot premium index and the eight-hourly funding
#   settlement are, and why either might change when a trader wants to sell.
# - Split a price history by date so that the training episodes and the reported
#   episodes cannot overlap, and check the split rather than assert it.
# - Compare a learned execution policy with fixed and rule-based schedules on
#   the same windows, using a paired difference and its standard error.
# - Read a forced-liquidation rate correctly: what it counts, what it does not
#   count, and which other column has to be read beside it.
#
# ## Book reference
#
# Section 21.4, *Application I - Optimal trade execution*.
#
# ## Prerequisites
#
# - `02_optimal_execution_ppo`, for implementation shortfall, the pacing action
#   and the Almgren-Chriss reference point.
# - Hourly perpetual-futures bars and the premium index, both read through
#   `data.load_crypto_perps` and `data.load_crypto_premium`.
# - `crypto_execution_env.CryptoExecutionEnv`, the environment that replays
#   those bars.

# %% [markdown]
# ## Setup

# %%
"""RL for crypto execution - a learned hourly schedule on recorded perpetual-futures bars."""

import warnings
from datetime import UTC, datetime

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import Markdown, display
from plotly.subplots import make_subplots

# Deprecation notices from these dependencies repeat on every environment
# construction and report nothing about this run. Convergence, overflow and
# invalid-value warnings stay visible: they report conditions results depend on.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="stable_baselines3")
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

from crypto_execution_env import CryptoExecutionEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import utils  # noqa: F401  - sets the Plotly renderer so figures carry a static PNG
from data import load_crypto_perps, load_crypto_premium
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
SYMBOL = "BTCUSDT"  # the one contract the order is executed in
START_DATE = "2022-01-01"
END_DATE = "2024-12-01"
EVAL_START_DATE = "2024-01-01"  # every reported episode starts on or after this date
TOTAL_SHARES = 100.0  # size of the parent order, in contracts
EXECUTION_HORIZON = 24  # hourly steps available to complete it
TOTAL_TIMESTEPS = 200_000  # environment steps PPO may consume while learning
EVAL_EPISODES = 200  # 24-hour windows every strategy is scored on
EVAL_SEED_BASE = 1000  # first evaluation seed; the rest follow consecutively
RISK_AVERSION = 1e-4  # weight on unsold inventory, per unit of price variance
SCHEDULE_PENALTY = 5e-4  # weight on deviating from the even-pace reference schedule
FUNDING_INTERVAL_HOURS = 8  # settlement cadence on this venue
SEED = 314

# %%
set_global_seeds(SEED)

# %% [markdown]
# ### What the settings decide
#
# The order size and the horizon together set how hard the execution problem is.
# A hundred contracts spread over twenty-four hours is roughly four contracts an
# hour against hourly volume that runs in the thousands, so the participation
# cap in the environment binds only in unusually thin hours. That is a
# deliberate choice: it keeps the exercise about *when* to trade rather than
# about whether the order fits at all, which is the question
# `07_backtest_with_impact` takes up.

# %%
display(
    Markdown(f"""
- **Order**: {TOTAL_SHARES:,.0f} contracts of `{SYMBOL}`, over {EXECUTION_HORIZON} hourly steps,
  or about {TOTAL_SHARES / EXECUTION_HORIZON:.1f} contracts an hour at an even pace.
- **History**: {START_DATE} to {END_DATE}. Episodes used for learning start before
  {EVAL_START_DATE}; every reported episode starts on or after it.
- **Learning budget**: {TOTAL_TIMESTEPS:,} environment steps, about
  {TOTAL_TIMESTEPS / EXECUTION_HORIZON:,.0f} complete executions of experience.
- **Evaluation**: {EVAL_EPISODES} windows on seeds {EVAL_SEED_BASE} to
  {EVAL_SEED_BASE + EVAL_EPISODES - 1}. A seed fixes the window, and every strategy is run
  on every seed, so the comparison is paired window by window.
- **Reward weights**: inventory risk at {RISK_AVERSION:.0e} and schedule deviation at
  {SCHEDULE_PENALTY:.0e}, on a reward otherwise equal to the negative shortfall in basis
  points.
""")
)

# %% [markdown]
# ## 1. Two things a perpetual futures market has that an equity market does not
#
# A **perpetual future** has no expiry date. Nothing forces its price to
# converge to the spot price, so the venue makes the two track each other with a
# periodic cash payment between longs and shorts called **funding**. On this
# venue funding settles every eight hours, at 00:00, 08:00 and 16:00 UTC.
#
# The **premium index** is the venue's measure of how far the perpetual is
# trading from spot, quoted as a fraction of the price. It is what the funding
# payment is computed from: a positive premium means the perpetual is rich, longs
# pay shorts, and a trader who wants to sell is selling into a market that is
# paying to be long.
#
# Both matter to an execution schedule. A seller may prefer to trade while the
# premium is rich, and flow tends to concentrate around a funding settlement as
# positions are adjusted, which changes the liquidity available in those hours.
# Neither is a signal about the direction of the price, and this notebook makes
# no claim that they are; they are state the policy is allowed to condition on.

# %% [markdown]
# ## 2. Build the execution panel
#
# Hourly bars and the premium index arrive on different clocks: the bars are
# hourly, the premium index is published every eight hours. The join carries the
# most recent published premium forward to each hour, which is what a trader
# would have had.

# %%
ohlcv = load_crypto_perps(
    frequency="1h", symbols=[SYMBOL], start_date=START_DATE, end_date=END_DATE
).sort("timestamp")
premium = load_crypto_premium(symbols=[SYMBOL], start_date=START_DATE, end_date=END_DATE).sort(
    "timestamp"
)

print(f"{SYMBOL}: {ohlcv.height:,} hourly bars, {premium.height:,} premium-index observations")

# %%
panel = ohlcv.join_asof(
    premium.select("timestamp", "symbol", "premium_index_close"),
    on="timestamp",
    by="symbol",
    strategy="backward",
    check_sortedness=False,
)

# %% [markdown]
# ### Every feature is lagged to what a decision could have known
#
# A decision taken at the top of hour $t$ can use the opening price of that hour,
# because it is the price being traded at, and the calendar, because a clock
# needs no data. Everything else has to come from bars that had already closed.
# So the return, the rolling volatility, the volume and the premium index are all
# shifted by one hour before they enter the panel.
#
# The volume column carries the most weight of the four. The environment charges
# market impact against a volume figure, and the figure it charges against is the
# previous hour's, because that is the last one a trader deciding at the top of
# the hour has seen. A schedule built on that column is choosing its hours from
# liquidity that had already printed, which is the only version of the choice a
# trader could make.

# %%
panel = (
    panel.with_columns(
        pl.col("close").pct_change().shift(1).alias("return_1h"),
        pl.col("close").pct_change().rolling_std(24).shift(1).alias("volatility_24h"),
        pl.col("volume").shift(1).alias("observed_volume"),
        pl.col("volume").rolling_mean(24).shift(1).alias("avg_volume_24h"),
        pl.col("premium_index_close").shift(1),
        pl.col("timestamp").dt.hour().alias("hour"),
        (
            (FUNDING_INTERVAL_HOURS - (pl.col("timestamp").dt.hour() % FUNDING_INTERVAL_HOURS))
            % FUNDING_INTERVAL_HOURS
        ).alias("hours_to_funding"),
    )
    .drop_nulls()
    .sort("timestamp")
)

# %% [markdown]
# ### Split by date, then check the split
#
# Training episodes are drawn from bars before `EVAL_START_DATE` and reported
# episodes from bars on or after it. The assertion below is the check: a split
# that is only described in prose is a split nobody has verified.

# %%
eval_start_ts = pl.lit(datetime.fromisoformat(EVAL_START_DATE).replace(tzinfo=UTC)).cast(
    panel["timestamp"].dtype
)
train_data = panel.filter(pl.col("timestamp") < eval_start_ts)
evaluation_data = panel.filter(pl.col("timestamp") >= eval_start_ts)
assert train_data["timestamp"].max() < evaluation_data["timestamp"].min()

display(
    Markdown(f"""
Panel: **{panel.height:,}** hourly rows, {panel["timestamp"].min():%Y-%m-%d} to
{panel["timestamp"].max():%Y-%m-%d}.

- Training bars: **{train_data.height:,}**, ending {train_data["timestamp"].max():%Y-%m-%d %H:%M}.
- Evaluation bars: **{evaluation_data.height:,}**, starting
  {evaluation_data["timestamp"].min():%Y-%m-%d %H:%M}. An episode is a
  {EXECUTION_HORIZON}-hour window drawn from these, so there are
  {evaluation_data.height - EXECUTION_HORIZON + 1:,} distinct windows to draw from.
""")
)

# %% [markdown]
# ### The history the agent trades in
#
# Price, premium index and hourly volume over the whole panel, with the
# boundary between the training bars and the evaluation bars marked. The
# evaluation year is not a quiet corner of the sample, which is worth
# establishing before reading any result taken from it.

# %%
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Open price", "Premium index (basis points)", "Hourly volume"],
    vertical_spacing=0.08,
)
fig.add_trace(
    go.Scatter(
        x=panel["timestamp"],
        y=panel["open"],
        line=dict(color=COLORS["blue"], width=1),
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=panel["timestamp"],
        y=panel["premium_index_close"] * 10_000,
        line=dict(color=COLORS["copper"], width=1),
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=panel["timestamp"],
        y=panel["observed_volume"],
        line=dict(color=COLORS["amber"], width=1),
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.add_hline(y=0, line=dict(color=COLORS["neutral"], width=1, dash="dot"), row=2, col=1)
for row in (1, 2, 3):
    fig.add_vline(
        x=evaluation_data["timestamp"].min(),
        line=dict(color=COLORS["neutral"], width=1.5, dash="dash"),
        row=row,
        col=1,
    )
fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
fig.update_yaxes(title_text="Premium (bps)", row=2, col=1)
fig.update_yaxes(title_text="Contracts", row=3, col=1)
fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_layout(
    title=(
        f"{SYMBOL} hourly bars over the panel"
        "<br><sup>Dashed line: the boundary between the training bars and the evaluation "
        "bars</sup>"
    ),
    height=760,
)
show_plotly_with_alt(
    fig,
    "Three stacked panels over the whole hourly panel, with a dashed vertical line at the boundary between the training bars and the evaluation bars: the open price, the premium index in basis points around a dotted zero line, and hourly traded volume.",
)

# %% [markdown]
# ## 3. The environment
#
# `CryptoExecutionEnv` replays a window of the panel. `reset` draws a start hour
# at random from its own generator, so a seed fixes the window; every strategy
# run on the same seed therefore trades the same twenty-four hours.
#
# The observation is seven numbers:
#
# | Feature | What it is |
# |---|---|
# | `inventory_ratio` | fraction of the order still unsold |
# | `time_ratio` | fraction of the horizon still available |
# | `volatility` | 24-hour rolling volatility of returns, through the previous bar |
# | `premium_index` | most recently published perpetual-spot premium |
# | `volume_ratio` | previous hour's volume against its 24-hour average |
# | `hour_of_day` | UTC hour |
# | `hours_to_funding` | hours until the next settlement |
#
# The action is a single number in $[0, 1]$ scaling the even-pace reference
# quantity between half and one and a half times, subject to a cap at a tenth of
# the previous hour's volume. Impact follows a square-root-plus-linear function
# of the participation rate, so the marginal cost of trading more in one hour
# rises but not without limit.
#
# ### What happens when a policy runs out of horizon
#
# There is no exemption from the participation cap on the final hour. Whatever
# the policy has not sold by then is unwound against that same bar, and the
# forced quantity and the policy's own quantity are charged impact on their
# combined participation, so splitting the order between them costs exactly what
# selling it in one trade costs. Each such episode is recorded as a **forced
# liquidation**.

# %% [markdown]
# ## 4. Two schedules to compare against
#
# **TWAP** sells the same quantity every hour and conditions on nothing. The
# **funding-aware rule** starts from the same even pace and speeds up when the
# premium is rich and when a settlement is within two hours, which is the
# hand-written version of the behaviour the agent is being asked to learn.
# Having both makes the comparison specific: the gap to TWAP measures the value
# of conditioning at all, and the gap to the rule measures the value of learning
# the conditioning rather than writing it down.


# %%
def summarize_execution(env: CryptoExecutionEnv) -> dict:
    """Shortfall of a completed episode in basis points, plus its execution history."""
    return {
        "shortfall_bps": env.total_cost / (env.arrival_price * env.total_shares) * 10_000,
        "forced_liquidation": any(h["forced_liquidation"] for h in env.execution_history),
        "history": env.execution_history,
    }


def twap_execution(env: CryptoExecutionEnv, reset_seed: int) -> dict:
    """Sell an equal quantity every hour."""
    env.reset(seed=reset_seed)
    done = False
    while not done:
        target = (
            env.remaining_shares
            if env.step_idx == env.horizon - 1
            else env.total_shares / env.horizon
        )
        _, _, terminated, truncated, _ = env.step(env.target_shares_to_action(target))
        done = terminated or truncated
    return summarize_execution(env)


# %%
PREMIUM_SCALE = 0.001  # premium level at which the pacing adjustment saturates
FUNDING_WINDOW_HOURS = 2  # "near funding" means this many hours or fewer remain
FUNDING_SPEEDUP = 1.25  # pace multiplier applied inside that window


def funding_aware_execution(env: CryptoExecutionEnv, reset_seed: int) -> dict:
    """Sell faster into a rich premium and near a funding settlement."""
    env.reset(seed=reset_seed)
    done = False
    while not done:
        if env.step_idx == env.horizon - 1:
            target = env.remaining_shares
        else:
            market = env.market_state()
            base_rate = env.remaining_shares / max(env.horizon - env.step_idx, 1)
            premium_signal = np.clip(market.premium_index / PREMIUM_SCALE, -1.0, 1.0)
            speedup = FUNDING_SPEEDUP if market.hours_to_funding <= FUNDING_WINDOW_HOURS else 1.0
            target = min(
                base_rate * np.clip(1.0 + 0.5 * premium_signal, 0.5, 1.5) * speedup,
                env.remaining_shares,
            )
        _, _, terminated, truncated, _ = env.step(env.target_shares_to_action(target))
        done = terminated or truncated
    return summarize_execution(env)


# %% [markdown]
# ## 5. Train the PPO agent
#
# The agent learns on the training bars only. Each episode it sees is a
# twenty-four hour window drawn at random from those bars, so over two hundred
# thousand steps it works through roughly eight thousand different windows.


# %%
def make_env(seed: int):
    """Factory returning an execution environment over the training bars."""

    def _init():
        return CryptoExecutionEnv(
            market_data=train_data,
            symbol=SYMBOL,
            total_shares=TOTAL_SHARES,
            horizon=EXECUTION_HORIZON,
            risk_aversion=RISK_AVERSION,
            schedule_penalty=SCHEDULE_PENALTY,
            seed=seed,
        )

    return _init


# The policy is a small MLP; on networks this size the cost of moving a batch to a
# GPU exceeds the cost of the arithmetic, so CPU is the faster device.
model = PPO(
    "MlpPolicy",
    DummyVecEnv([make_env(seed=SEED)]),
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01,
    seed=SEED,
    verbose=0,
    device="cpu",
)

# %%
print(f"Training PPO for {TOTAL_TIMESTEPS:,} steps on {train_data.height:,} hourly bars...")
model.learn(total_timesteps=TOTAL_TIMESTEPS)
print("Training complete")


# %%
def ppo_execution(env: CryptoExecutionEnv, reset_seed: int) -> dict:
    """Execute with the trained policy, acting greedily on its own observation."""
    obs, _ = env.reset(seed=reset_seed)
    done = False
    while not done:
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, _, terminated, truncated, _ = env.step(float(np.asarray(action).reshape(-1)[0]))
        done = terminated or truncated
    return summarize_execution(env)


# %% [markdown]
# ## 6. Evaluate on the held-out year
#
# Every strategy is run on the same list of seeds over the evaluation bars, and
# each gets a fresh environment, so nothing carries between them. A seed fixes
# the twenty-four hour window, which makes the comparison paired: a difference
# between two strategies on the same seed is a difference between the schedules
# rather than between the hours they happened to be given.


# %%
def episode_diagnostics(env: CryptoExecutionEnv, result: dict) -> dict:
    """Cost of one episode, plus where in the horizon its volume sat."""
    history = result["history"]
    quarter_cutoff = max(env.horizon - env.horizon // 4, 0)
    return {
        "shortfall_bps": result["shortfall_bps"],
        "forced_liquidation": result["forced_liquidation"],
        # `shares_sold` is the whole bar and `forced_shares` its involuntary part.
        "last_hour_share": float(history[-1]["shares_sold"]) / env.total_shares,
        "forced_share": sum(float(h["forced_shares"]) for h in history) / env.total_shares,
        "final_quarter_share": sum(
            float(h["shares_sold"]) for h in history if int(h["step"]) >= quarter_cutoff
        )
        / env.total_shares,
        "history": history,
    }


def evaluate_strategy(strategy_name: str, strategy_fn) -> tuple[dict, list[dict]]:
    """Run one strategy over every evaluation seed and collect its per-episode diagnostics."""
    diagnostics, all_paths = [], []
    for i in range(EVAL_EPISODES):
        episode_seed = EVAL_SEED_BASE + i
        env = CryptoExecutionEnv(
            market_data=evaluation_data,
            symbol=SYMBOL,
            total_shares=TOTAL_SHARES,
            horizon=EXECUTION_HORIZON,
            risk_aversion=RISK_AVERSION,
            schedule_penalty=SCHEDULE_PENALTY,
            seed=episode_seed,
        )
        diag = episode_diagnostics(env, strategy_fn(env, reset_seed=episode_seed))
        diagnostics.append(diag)
        all_paths.extend(
            {"strategy": strategy_name, "episode_id": i, "episode_seed": episode_seed, **row}
            for row in diag["history"]
        )

    shortfalls = np.array([d["shortfall_bps"] for d in diagnostics])
    summary = {
        "shortfall_bps": shortfalls,
        "mean_bps": float(shortfalls.mean()),
        "std_bps": float(shortfalls.std()),
        "min_bps": float(shortfalls.min()),
        "max_bps": float(shortfalls.max()),
        "forced_liq_rate": 100 * float(np.mean([d["forced_liquidation"] for d in diagnostics])),
        "last_hour_share_pct": 100 * float(np.mean([d["last_hour_share"] for d in diagnostics])),
        "forced_share_pct": 100 * float(np.mean([d["forced_share"] for d in diagnostics])),
        "final_quarter_share_pct": 100
        * float(np.mean([d["final_quarter_share"] for d in diagnostics])),
    }
    return summary, all_paths


# %%
STRATEGIES = {
    "TWAP": twap_execution,
    "Funding-aware": funding_aware_execution,
    "PPO": ppo_execution,
}
COLOR_BY_STRATEGY = {
    "TWAP": COLORS["blue"],
    "Funding-aware": COLORS["amber"],
    "PPO": COLORS["copper"],
}
REFERENCE_STRATEGY = "TWAP"

results, evaluation_paths = {}, []
for name, fn in STRATEGIES.items():
    results[name], strategy_paths = evaluate_strategy(name, fn)
    evaluation_paths.extend(strategy_paths)
    print(
        f"{name:15s}: {results[name]['mean_bps']:7.2f} bps mean, "
        f"{results[name]['std_bps']:6.2f} bps std, "
        f"forced liquidation in {results[name]['forced_liq_rate']:5.1f}% of episodes"
    )

# %% [markdown]
# ### Cost, and how much of the difference the windows explain
#
# Implementation shortfall on a real price path is dominated by what the price
# did during the window, which is the same for all three strategies. The box
# plot shows how wide that is; the paired differences beside it remove it.

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Per-episode shortfall", f"Paired difference to {REFERENCE_STRATEGY}"),
    horizontal_spacing=0.14,
)
for name, color in COLOR_BY_STRATEGY.items():
    fig.add_trace(
        go.Box(
            y=results[name]["shortfall_bps"],
            name=name,
            marker=dict(color=color),
            boxmean=True,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

paired = {
    name: results[name]["shortfall_bps"] - results[REFERENCE_STRATEGY]["shortfall_bps"]
    for name in STRATEGIES
    if name != REFERENCE_STRATEGY
}
fig.add_trace(
    go.Scatter(
        x=list(paired),
        y=[float(d.mean()) for d in paired.values()],
        error_y=dict(
            type="data",
            array=[float(d.std(ddof=1) / np.sqrt(d.size)) for d in paired.values()],
            color=COLORS["copper"],
            thickness=2,
            width=8,
        ),
        mode="markers",
        marker=dict(color=COLORS["copper"], size=10),
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig.add_hline(y=0, line=dict(color=COLORS["neutral"], width=1, dash="dash"), row=1, col=2)
fig.update_yaxes(title_text="Shortfall (bps of arrival notional)", row=1, col=1)
fig.update_yaxes(title_text=f"Strategy minus {REFERENCE_STRATEGY} (bps)", row=1, col=2)
fig.update_xaxes(title_text="Strategy", row=1, col=1)
fig.update_layout(
    title=(
        "Shortfall per episode, and the paired difference to the fixed schedule"
        "<br><sup>Right: mean per-episode difference with one standard error; negative is "
        "cheaper than the reference</sup>"
    ),
    height=460,
)
show_plotly_with_alt(
    fig,
    "Two panels. Left: a box per strategy over the per-episode implementation shortfall in basis points. Right: the mean paired difference to the fixed schedule for the other two strategies, as markers with one standard error either side, against a dashed line at zero.",
)

# %% [markdown]
# ## 7. The shape of each schedule
#
# Averaging over the evaluation episodes hour by hour shows how each strategy
# paced. The premium panel carries one line rather than three: the seeds are
# shared, so the premium path is the same for every strategy and drawing it
# three times would say nothing.

# %%
evaluation_df = pl.DataFrame(evaluation_paths).with_columns(
    pl.col("shortfall").cum_sum().over(["strategy", "episode_id"]).alias("cum_shortfall"),
    (pl.col("premium_index") * 10_000).alias("premium_bps"),
)
trajectory_stats = (
    evaluation_df.group_by("strategy", "step")
    .agg(
        pl.col("shares_sold").mean().alias("mean_shares_sold"),
        pl.col("remaining").mean().alias("mean_remaining"),
        pl.col("cum_shortfall").mean().alias("mean_cum_shortfall"),
        pl.col("premium_bps").mean().alias("mean_premium_bps"),
    )
    .sort("strategy", "step")
)

# %%
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Execution rate (contracts per hour)",
        "Remaining inventory",
        "Average premium index over the episode windows",
        "Cumulative implementation shortfall",
    ],
    vertical_spacing=0.07,
)
for name, color in COLOR_BY_STRATEGY.items():
    strategy_df = trajectory_stats.filter(pl.col("strategy") == name).sort("step")
    steps = strategy_df["step"].to_list()
    for row, column in ((1, "mean_shares_sold"), (2, "mean_remaining"), (4, "mean_cum_shortfall")):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=strategy_df[column].to_list(),
                name=name,
                line=dict(color=color, width=2),
                legendgroup=name,
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )

reference_df = trajectory_stats.filter(pl.col("strategy") == REFERENCE_STRATEGY).sort("step")
fig.add_trace(
    go.Scatter(
        x=reference_df["step"].to_list(),
        y=reference_df["mean_premium_bps"].to_list(),
        line=dict(color=COLORS["neutral"], width=2, dash="dot"),
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.update_yaxes(title_text="Contracts", row=1, col=1)
fig.update_yaxes(title_text="Contracts unsold", row=2, col=1)
fig.update_yaxes(title_text="Premium (bps)", row=3, col=1)
fig.update_yaxes(title_text="Shortfall (USDT)", row=4, col=1)
fig.update_xaxes(title_text="Hour of the execution window", row=4, col=1)
fig.update_layout(
    title="Execution rate, inventory, premium and cost, averaged over episodes",
    height=900,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
show_plotly_with_alt(
    fig,
    "Four stacked panels averaged over the evaluation episodes and shown by hour of the execution window: contracts executed per hour and contracts still unsold, one line per strategy; the average premium index over the same windows on a single dotted line; and cumulative implementation shortfall per strategy.",
)

# %% [markdown]
# ## 8. Forced liquidation, and how to read it
#
# The flag fires when a policy reaches the final hour still holding more than it
# sells there, either because it paced too slowly or because the schedule and
# participation caps bind. The environment unwinds the remainder against that
# same bar and charges both legs on their combined participation.
#
# The rate counts *episodes that ended with an involuntary trade*. On its own it
# does not say the policy back-loaded, and it says nothing about how much was
# unwound. Read it against the volume columns: a policy can execute less of the
# order late than TWAP does and still leave a residual it fails to clear on the
# last bar, which is a pacing failure of a completely different kind from
# postponement.

# %%
fig = go.Figure()
for label, key in [
    ("Episodes ending in a forced trade", "forced_liq_rate"),
    ("Volume unwound involuntarily", "forced_share_pct"),
    (f"Volume in the final {EXECUTION_HORIZON // 4} hours", "final_quarter_share_pct"),
]:
    fig.add_trace(
        go.Bar(x=list(STRATEGIES), y=[results[name][key] for name in STRATEGIES], name=label)
    )
fig.add_hline(
    y=25.0,
    line=dict(color=COLORS["neutral"], width=1, dash="dot"),
    annotation_text="even pace, final quarter",
)
fig.update_layout(
    title="Forced-liquidation rate beside the volume it actually involved",
    xaxis_title="Strategy",
    yaxis_title="Per cent (of episodes, or of the order)",
    barmode="group",
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
show_plotly_with_alt(
    fig,
    "Grouped bars per strategy giving the percentage of episodes that ended with a forced trade, the percentage of the order unwound involuntarily, and the percentage traded in the final quarter of the horizon, against a dotted line at the even-pace share of a quarter.",
)

# %% [markdown]
# ## 9. Does the agent condition on what it was given?
#
# The state includes the premium index and the hours to the next settlement.
# Pooling every step of every evaluation episode and grouping the execution rate
# by those two variables shows how fast the policy traded in each group. Read it
# as an association and nothing more: the execution rate also depends on the
# inventory left, the horizon remaining and the participation cap, and hours
# grouped by premium state differ in those too, so a gap between the bars is
# consistent with a policy that never reads the premium at all.

# %%
PREMIUM_BAND_BPS = 5.0  # boundary between rich, neutral and cheap premium states

ppo_steps = evaluation_df.filter(pl.col("strategy") == "PPO").with_columns(
    pl.when(pl.col("premium_bps") > PREMIUM_BAND_BPS)
    .then(pl.lit(f"Rich (> {PREMIUM_BAND_BPS:.0f} bps)"))
    .when(pl.col("premium_bps") < -PREMIUM_BAND_BPS)
    .then(pl.lit(f"Cheap (< -{PREMIUM_BAND_BPS:.0f} bps)"))
    .otherwise(pl.lit("Neutral"))
    .alias("premium_state"),
    pl.when(pl.col("hours_to_funding") <= FUNDING_WINDOW_HOURS)
    .then(pl.lit(f"Within {FUNDING_WINDOW_HOURS} hours"))
    .otherwise(pl.lit(f"More than {FUNDING_WINDOW_HOURS} hours"))
    .alias("funding_state"),
)

premium_rates = (
    ppo_steps.group_by("premium_state")
    .agg(pl.col("shares_sold").mean().alias("rate"), pl.len().alias("n"))
    .sort("premium_state")
)
funding_rates = (
    ppo_steps.group_by("funding_state")
    .agg(pl.col("shares_sold").mean().alias("rate"), pl.len().alias("n"))
    .sort("funding_state")
)

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("By premium state", "By funding proximity"),
    horizontal_spacing=0.16,
)
fig.add_trace(
    go.Bar(
        x=premium_rates["premium_state"].to_list(),
        y=premium_rates["rate"].to_list(),
        marker_color=COLORS["blue"],
        text=[f"n={n:,}" for n in premium_rates["n"]],
        textposition="outside",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=funding_rates["funding_state"].to_list(),
        y=funding_rates["rate"].to_list(),
        marker_color=COLORS["amber"],
        text=[f"n={n:,}" for n in funding_rates["n"]],
        textposition="outside",
    ),
    row=1,
    col=2,
)
fig.update_yaxes(title_text="Average contracts per hour", rangemode="tozero", row=1, col=1)
fig.update_yaxes(title_text="Average contracts per hour", rangemode="tozero", row=1, col=2)
fig.update_layout(
    title="Learned execution rate grouped by premium state and funding proximity",
    height=440,
    showlegend=False,
)
show_plotly_with_alt(
    fig,
    "Two bar panels of the learned policy's average execution rate in contracts per hour, the left grouped by premium state and the right by proximity to a funding settlement, each bar labelled with the number of pooled steps behind it.",
)

# %% [markdown]
# ## 10. Summary

# %%
pl.DataFrame(
    [
        {
            "Strategy": name,
            "Mean IS (bps)": round(r["mean_bps"], 2),
            "Std (bps)": round(r["std_bps"], 2),
            "Min (bps)": round(r["min_bps"], 2),
            "Max (bps)": round(r["max_bps"], 2),
            "Forced liq (%)": round(r["forced_liq_rate"], 1),
            "Last hour (%)": round(r["last_hour_share_pct"], 1),
        }
        for name, r in results.items()
    ]
)

# %% [markdown]
# ## 11. Key takeaways

# %%


def comparison_row(reference: str, other: str) -> str:
    """One line comparing two arms, with the paired and unpaired standard errors side by side.

    Pairing changes the variance of a difference by minus twice the covariance, so it narrows
    the interval only when the two arms move together. Reporting both, with the correlation,
    lets the reader see which of the two applies rather than take it on trust.
    """
    a = results[other]["shortfall_bps"]
    b = results[reference]["shortfall_bps"]
    difference = a - b
    n = difference.size
    paired_se = float(difference.std(ddof=1) / np.sqrt(n))
    unpaired_se = float(np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n))
    correlation = float(np.corrcoef(a, b)[0, 1])
    return (
        f"- **{other}** minus **{reference}**: {difference.mean():+.2f} bps per episode. "
        f"Paired standard error {paired_se:.2f}, treating the arms as independent "
        f"{unpaired_se:.2f}; the two series correlate {correlation:+.2f}."
    )


paired_lines = "\n".join(comparison_row(REFERENCE_STRATEGY, name) for name in paired)
profile_lines = "\n".join(
    f"- **{name}**: {results[name]['mean_bps']:.1f} bps mean shortfall, "
    f"{results[name]['final_quarter_share_pct']:.1f}% of the order in the final quarter, "
    f"forced liquidation in {results[name]['forced_liq_rate']:.1f}% of episodes involving "
    f"{results[name]['forced_share_pct']:.1f}% of the order."
    for name in STRATEGIES
)
premium_line = ", ".join(
    f"{row['premium_state']} {row['rate']:.2f}" for row in premium_rates.iter_rows(named=True)
)
funding_line = ", ".join(
    f"{row['funding_state']} {row['rate']:.2f}" for row in funding_rates.iter_rows(named=True)
)

display(
    Markdown(f"""
{profile_lines}

**Match the windows, and say what matching does and does not remove.** Running every strategy
on the same seeds means the difference between two of them is not a difference between the
hours they were given. It is not a difference net of the price either: the schedules sell
different quantities in different hours, so the difference still contains price movement
weighted by those quantity differences, along with the modelled impact. What the paired
difference measures is one schedule against another on matched windows, timing included.

{paired_lines}

Size each difference with its paired standard error, which carries the covariance between the
arms whichever way it points. The independent-samples figure beside it shows how much the
matching changed the precision.

**The grouped rates are associations, and only associations.** In contracts per hour: by
premium state, {premium_line}; by funding proximity, {funding_line}. A difference between the
groups does not show the policy reading either feature. `shares_sold` is also driven by how
much inventory is left, how much of the horizon remains, the participation cap and any forced
remainder, and all four can differ across these groups in a policy that ignores the premium and
the funding clock entirely - the premium is persistent, so hours grouped by it are not
otherwise alike. Establishing that a feature is used takes a sensitivity check that varies it
while holding the rest of the state fixed, or a policy retrained without it.

**A forced-liquidation rate and a forced volume are different measurements.** A policy can
end most of its episodes with an involuntary trade while unwinding a very small part of the
order that way, which is a pacing habit rather than a failure to execute. Reporting the rate
alone would make the two indistinguishable.

### Known limitations

- The prices are recorded but the fills are not. Every execution here happens at the modelled
  impact price against the previous hour's volume, with no queue, no partial fill, no
  rejection and no other participant reacting to the order.
- Episodes are drawn at random from the evaluation bars and overlap heavily, so the
  {EVAL_EPISODES} windows are not {EVAL_EPISODES} independent observations. The standard
  errors above understate the true uncertainty by an amount this notebook does not estimate.
- The premium index and the funding clock are state, not signals. Nothing here establishes
  that either predicts the price or the cost of trading; the conditional rates in section 9
  describe the policy, not the market.
- One training run at one seed. The sign of the paired differences is a property of this run.

**Next**: `05_deep_hedging_pfhedge` moves from executing a decided trade to choosing a hedge,
where the action is continuous and the objective is a tail-risk measure rather than a mean.
""")
)
