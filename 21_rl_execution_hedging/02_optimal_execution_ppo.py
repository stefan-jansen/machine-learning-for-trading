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
# # Optimal Execution with PPO
#
# **Chapter 21: Reinforcement Learning for Execution and Hedging**
#
# ## Purpose
#
# A trader has to sell a fixed quantity within a fixed number of steps. Selling
# it all at once moves the price against the order; spreading it out leaves the
# unsold remainder exposed to whatever the price does next. A schedule that
# resolves that tension is what an execution algorithm is. This notebook trains
# a PPO agent to choose the schedule step by step, and compares it against two
# fixed schedules on the same simulated market paths.
#
# The comparison is set up so that it can be read honestly. Both benchmarks are
# run on byte-identical price paths, the analytical benchmark is given an
# information advantage that is stated rather than hidden, and the diagnostics
# report not only what each schedule cost but where in the horizon it chose to
# trade.
#
# ## Learning objectives
#
# After working through this notebook you will be able to:
#
# - Measure the cost of an execution schedule as implementation shortfall in
#   basis points of the arrival notional, and say which part of that cost comes
#   from the spread, from impact, and from waiting.
# - Build the discrete Almgren-Chriss schedule from a risk-aversion parameter
#   and market statistics, and explain why it needs statistics a live trader
#   would not have.
# - Train a PPO agent on a continuous pacing action and evaluate it against
#   fixed schedules on paired episodes, so that a difference between two
#   strategies is not a difference between two market paths.
# - Read where in the horizon a schedule concentrates its trading, and say why a
#   low average cost achieved at the end of the horizon is a bet on liquidity
#   rather than a saving.
# - Separate the parts of a simulator that were estimated from market data from
#   the parts that were assumed, and state what each one licenses you to claim.
#
# ## Book reference
#
# Section 21.4, *Application I: Optimal Trade Execution*.
#
# ## Prerequisites
#
# - Chapter 18 on transaction costs, for implementation shortfall and market
#   impact.
# - `rl_calibration.CryptoMarketCalibrator`, which fits the simulator's
#   volatility, spread and liquidity parameters to hourly `BTCUSDT` bars read
#   through `data.load_crypto_perps`.
# - `rl_environments.ExecutionEnv`, the Gymnasium environment all three
#   strategies are run in.

# %% [markdown]
# ## Setup

# %%
"""Optimal execution with PPO - a learned liquidation schedule against TWAP and Almgren-Chriss."""

import warnings

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

from rl_calibration import CryptoMarketCalibrator
from rl_environments import ExecutionEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import utils  # noqa: F401  - sets the Plotly renderer so figures carry a static PNG
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

# %% tags=["parameters"]
CALIBRATION_SYMBOL = "BTCUSDT"  # symbol the simulator's market parameters are fitted to
TOTAL_SHARES = 10_000  # size of the parent order every strategy must liquidate
EXECUTION_HORIZON = 60  # steps available to complete it
TOTAL_TIMESTEPS = 100_000  # environment steps PPO may consume while learning
EVAL_EPISODES = 20  # market paths every strategy is scored on
EVAL_SEED_BASE = 1000  # first evaluation seed; the rest follow consecutively
RISK_AVERSION = 1e-4  # weight on holding unsold inventory, per unit of price variance
SCHEDULE_PENALTY = 5e-5  # weight on deviating from the even-pace reference schedule
SEED = 314

# %%
set_global_seeds(SEED)

# %% [markdown]
# ### What the settings decide
#
# `RISK_AVERSION` and `SCHEDULE_PENALTY` are the two that shape the learned
# policy, and they pull in opposite directions. Risk aversion charges the agent
# for inventory it still holds, so raising it makes the agent sell earlier and
# accept more impact. The schedule penalty charges it for departing from an even
# pace, so raising it pulls the agent back towards TWAP whatever the market is
# doing. Both are per-step terms in the reward, and both are zero by default in
# `ExecutionEnv`; the values below are what this notebook chooses.

# %%
display(
    Markdown(f"""
- **Parent order**: {TOTAL_SHARES:,} shares, to be fully liquidated within
  {EXECUTION_HORIZON} steps. The final step sells whatever is left, so no strategy
  can finish holding inventory.
- **Learning budget**: {TOTAL_TIMESTEPS:,} environment steps, about
  {TOTAL_TIMESTEPS / EXECUTION_HORIZON:,.0f} complete liquidations of experience.
- **Evaluation**: {EVAL_EPISODES} episodes on seeds {EVAL_SEED_BASE} to
  {EVAL_SEED_BASE + EVAL_EPISODES - 1}. Every strategy is run on every seed, so the
  comparison is paired path by path.
- **Reward weights**: inventory risk at {RISK_AVERSION:.0e} and schedule deviation at
  {SCHEDULE_PENALTY:.0e}, both applied to a reward otherwise equal to the negative
  shortfall of the step.
""")
)

# %% [markdown]
# ## 1. What the simulator was fitted to, and what it assumes
#
# The agent needs a market that reacts to its own trading, which no recorded
# price series does. The simulator therefore generates paths, and
# `CryptoMarketCalibrator` sets its parameters from hourly perpetual-futures
# bars so those paths are not arbitrary.
#
# Not every parameter is estimated, and the difference matters for what the
# results can be used to claim. Two are fitted directly, two are proxies whose
# limitations are worth knowing, and two are assumptions the calibration falls
# back to.

# %%
calibrator = CryptoMarketCalibrator(CALIBRATION_SYMBOL)
source_bars = calibrator.load_data()
cal_params = calibrator.get_execution_env_params()

# %% [markdown]
# ### The parameters the environment receives

# %%
calibration_summary = pl.DataFrame(
    [
        {
            "parameter": "garch_alpha",
            "value": cal_params.garch.alpha,
            "source": "fitted: GARCH(1,1) on hourly log returns",
        },
        {
            "parameter": "garch_beta",
            "value": cal_params.garch.beta,
            "source": "fitted: GARCH(1,1) on hourly log returns",
        },
        {
            "parameter": "garch_uncond_vol",
            "value": cal_params.garch.unconditional_vol,
            "source": "fitted: sample standard deviation of hourly returns",
        },
        {
            "parameter": "p_stay_normal",
            "value": cal_params.regimes.transition_matrix[0, 0],
            "source": "fitted: transitions of a rolling-volatility quartile split",
        },
        {
            "parameter": "p_stay_stressed",
            "value": cal_params.regimes.transition_matrix[1, 1],
            "source": "fitted: transitions of a rolling-volatility quartile split",
        },
        {
            "parameter": "spread_normal",
            "value": cal_params.spread_normal,
            "source": "proxy: median hourly high-low range, not a quoted spread",
        },
        {
            "parameter": "spread_stressed",
            "value": cal_params.spread_stressed,
            "source": "proxy: median hourly high-low range, not a quoted spread",
        },
        {
            "parameter": "depth_normal",
            "value": cal_params.depth_normal,
            "source": "proxy: median hourly traded volume, not resting depth",
        },
        {
            "parameter": "depth_stressed",
            "value": cal_params.depth_stressed,
            "source": "proxy: median hourly traded volume, not resting depth",
        },
        {
            "parameter": "permanent_impact",
            "value": cal_params.permanent_impact,
            "source": "assumed: Amihud estimate falls below the floor, so the floor is used",
        },
        {
            "parameter": "temporary_impact",
            "value": cal_params.temporary_impact,
            "source": "assumed: fixed fraction of the permanent coefficient",
        },
    ]
)
calibration_summary

# %% [markdown]
# ### Three things to notice before reading any result
#
# **The spread is an upper bound, not a spread.** `estimate_spread_depth` uses
# the median hourly high-low range as a stand-in for the bid-ask spread, because
# the hourly bars carry no quotes. The range over an hour is far wider than the
# quoted spread on a market as liquid as BTC perpetuals, so every strategy here
# pays a trading cost much larger than a real execution of this size would.
# Every comparison below is conditional on that spread process: the spread
# varies from step to step and across regimes, and the three schedules trade
# different quantities at different steps, so their volume-weighted spread costs
# are not the same number and a different spread process could order them
# differently.
#
# **"Depth" is traded volume.** The same function uses median hourly volume as a
# stand-in for the resting size on the book. That is why the stressed value is
# the larger of the two: volume rises when markets are stressed, while resting
# depth falls. The simulator therefore makes a stressed regime a wider, deeper
# market rather than a wider, thinner one, and the participation cap is looser
# there rather than tighter.
#
# **The impact coefficients are assumptions.** `estimate_impact` computes the
# Amihud illiquidity ratio - the average absolute return per dollar traded - and
# rescales it into an impact coefficient, then clamps the result into a range.
# For a market this liquid the rescaled estimate falls well below the lower
# clamp, so the value the environment receives is the clamp itself. Nothing in
# the impact model below was measured.

# %% [markdown]
# ### What the calibration read

# %%
display(
    Markdown(f"""
`{CALIBRATION_SYMBOL}` hourly bars: **{source_bars.height:,}** observations from
{source_bars["timestamp"].min():%Y-%m-%d} to {source_bars["timestamp"].max():%Y-%m-%d}.
The regime split assigns the highest quartile of 24-hour rolling volatility to the
stressed state and the rest to the normal state.
""")
)

# %% [markdown]
# ## 2. The execution problem
#
# **Implementation shortfall** is the cost of an execution measured against the
# price that was on the screen when the decision was made. That reference price
# is the **arrival price**. If the arrival price is $P_0$ and share $i$ of the
# order is filled at $P_i$, the shortfall of a sale is
#
# $$\text{IS} = \sum_i (P_0 - P_i)$$
#
# and it is reported here in **basis points** (hundredths of a percent) of the
# arrival notional $P_0 Q$, so that orders of different sizes are comparable.
#
# Three things push $P_i$ below $P_0$ on a sale. The **spread**: a seller
# crosses to the bid, giving up half the quoted spread on every share.
# **Temporary impact**: pushing a large quantity through the book in one step
# walks down the available bids, and the size of that effect grows with the
# **participation rate**, the fraction of the available liquidity the order
# consumes. **Permanent impact**: the market reads the flow as information and
# the price does not fully recover, so every share sold makes the shares after
# it cheaper.
#
# Against those three, waiting has its own cost. Inventory not yet sold is
# exposed to the price moving on its own, and that exposure is what the
# risk-aversion term in the reward charges for.

# %% [markdown]
# ### The environment
#
# `ExecutionEnv` is a Gymnasium environment holding one liquidation. Its
# observation is six numbers - the fraction of the order still unsold, the
# fraction of the horizon still available, the current spread, depth, volatility
# and regime - and its action is a single number in $[0, 1]$ that sets the pace
# for the step: zero trades at half the even rate, one trades at one and a half
# times it, subject to a cap at a fixed share of the available depth.
#
# The action is a pace multiplier rather than a quantity so that no policy can
# dump the whole order in one step, which would make the comparison a
# comparison of two very different problems. The one exception is the final
# step, which sells the entire remainder and ignores the participation cap:
# the order must complete, so a policy that postpones pays for the postponement
# there. That is what makes postponement measurable rather than free, and
# section 7 measures it.

# %% [markdown]
# ### One market path the agent has to trade
#
# Before any strategy runs, this is what a single episode looks like: the mid
# price the order is sold into, and the liquidity available at each step, with
# the stressed steps shaded.

# %%
sample_env = ExecutionEnv(
    total_shares=TOTAL_SHARES,
    horizon=EXECUTION_HORIZON,
    cal_params=cal_params,
    seed=EVAL_SEED_BASE,
)
sample_env.reset(seed=EVAL_SEED_BASE)
sample_path = pl.DataFrame(
    [
        {
            "step": t,
            "mid_price": s.mid_price,
            "spread_bps": s.spread * 10_000,
            "depth": s.depth,
            "regime": s.regime,
        }
        for t, s in enumerate(sample_env.market_path)
    ]
)

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Mid price", "Available depth and half-spread"],
    specs=[[{}], [{"secondary_y": True}]],
    vertical_spacing=0.12,
)
fig.add_trace(
    go.Scatter(
        x=sample_path["step"],
        y=sample_path["mid_price"],
        name="Mid price",
        line=dict(color=COLORS["blue"], width=2),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=sample_path["step"],
        y=sample_path["depth"],
        name="Depth",
        line=dict(color=COLORS["blue"], width=2),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=sample_path["step"],
        y=sample_path["spread_bps"] / 2,
        name="Half-spread (bps)",
        line=dict(color=COLORS["copper"], width=2, dash="dash"),
    ),
    row=2,
    col=1,
    secondary_y=True,
)

# The rectangles go on after the traces and below them: Plotly's default
# `exclude_empty_subplots=True` drops a shape added to a subplot that holds no
# trace yet, which silently removed the shading this figure's title promises.
for stressed_step in sample_path.filter(pl.col("regime") == 1)["step"]:
    for panel in (1, 2):
        fig.add_vrect(
            x0=stressed_step - 0.5,
            x1=stressed_step + 0.5,
            fillcolor=COLORS["amber"],
            opacity=0.15,
            line_width=0,
            layer="below",
            row=panel,
            col=1,
        )
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Depth (shares)", row=2, col=1)
fig.update_yaxes(title_text="Half-spread (bps)", row=2, col=1, secondary_y=True)
fig.update_xaxes(title_text="Step", row=2, col=1)
fig.update_layout(
    title="One episode of the simulated market, stressed steps shaded",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.show()

# %% [markdown]
# ## 3. Two fixed schedules to compare against
#
# A learned policy is only interesting against something. The two benchmarks
# below bracket the range of what a schedule can know: TWAP knows nothing about
# the market, and the Almgren-Chriss schedule used here knows more than a live
# trader could.

# %% [markdown]
# ### TWAP
#
# **TWAP** - time-weighted average price - sells the same quantity at every
# step. It is the schedule that minimises impact for a given horizon when
# nothing is known about how liquidity or volatility will vary, and it is the
# benchmark almost every execution desk reports against.


# %%
def summarize_execution(env: ExecutionEnv) -> dict:
    """Shortfall of a completed episode, in dollars and in basis points of arrival notional."""
    return {
        "total_shortfall": env.total_cost,
        "shortfall_bps": env.total_cost / (env.arrival_price * env.total_shares) * 10_000,
        "history": env.execution_history,
    }


def twap_execution(env: ExecutionEnv, reset_seed: int | None = None) -> dict:
    """Sell an equal quantity at every step."""
    env.reset(seed=reset_seed)
    shares_per_step = env.total_shares / env.horizon
    done = False

    while not done:
        if env.step_idx == env.horizon - 1:
            target_shares = env.remaining_shares
        else:
            target_shares = min(int(np.ceil(shares_per_step)), env.remaining_shares)
        _, _, terminated, truncated, _ = env.step(env.target_shares_to_action(target_shares))
        done = terminated or truncated

    return summarize_execution(env)


# %% [markdown]
# ### The Almgren-Chriss schedule
#
# Almgren and Chriss derive the schedule that minimises the sum of expected
# impact cost and a risk-aversion-weighted variance of the shortfall. Under
# linear impact the solution is a static schedule whose remaining inventory
# decays as a hyperbolic sine,
#
# $$q_t = Q \, \frac{\sinh\!\big(\kappa (T - t)\big)}{\sinh(\kappa T)},
# \qquad \kappa^2 = \frac{\lambda \sigma^2}{\eta}$$
#
# where $Q$ is the order size, $T$ the horizon, $\lambda$ the risk aversion,
# $\sigma$ the price volatility and $\eta$ the temporary-impact coefficient.
# Raising $\lambda$ or $\sigma$ raises $\kappa$ and front-loads the schedule;
# as $\kappa \to 0$ the schedule flattens into TWAP.
#
# **This benchmark is an oracle.** Computing $\kappa$ needs $\sigma$ and $\eta$,
# and the implementation below takes them from the average volatility and depth
# **over the whole episode**, which is information no trader has at the start.
# Neither TWAP nor PPO receives it. The schedule is therefore a reference point
# for how much a perfectly informed static plan can do, not a competitor an
# online policy can be said to have improved on.


# %%
def discrete_almgren_chriss_schedule(
    total_shares: int,
    horizon: int,
    sigma_price: float,
    eta: float,
    gamma: float,
    risk_aversion: float,
) -> np.ndarray:
    """Per-step share quantities of the discrete Almgren-Chriss trajectory."""
    if horizon <= 0:
        return np.array([], dtype=int)

    effective_eta = max(eta - 0.5 * gamma, 1e-8)
    kappa_sq = max(risk_aversion, 0.0) * sigma_price**2 / effective_eta

    if kappa_sq < 1e-10:
        remaining_path = np.linspace(total_shares, 0.0, horizon + 1)
    else:
        kappa = np.arccosh(1.0 + 0.5 * kappa_sq)
        if not np.isfinite(kappa) or abs(np.sinh(kappa * horizon)) < 1e-10:
            remaining_path = np.linspace(total_shares, 0.0, horizon + 1)
        else:
            time_grid = np.arange(horizon + 1)
            remaining_path = (
                total_shares * np.sinh(kappa * (horizon - time_grid)) / np.sinh(kappa * horizon)
            )

    # Round to whole shares, then hand the rounding remainder to the steps with
    # the largest fractional part so the schedule still sums to the order size.
    float_trades = np.maximum(remaining_path[:-1] - remaining_path[1:], 0.0)
    trades = np.floor(float_trades).astype(int)
    remainder = total_shares - int(trades.sum())
    if remainder > 0:
        order = np.argsort(-(float_trades - trades))
        trades[order[:remainder]] += 1

    return trades


# %%
def almgren_chriss_execution(env: ExecutionEnv, reset_seed: int | None = None) -> dict:
    """Follow the static Almgren-Chriss schedule computed from whole-episode statistics."""
    env.reset(seed=reset_seed)

    sigma_price = env.arrival_price * np.mean([state.volatility for state in env.market_path])
    avg_depth = np.mean([state.depth for state in env.market_path])
    eta = env.arrival_price * env.temporary_impact / max(avg_depth, 1.0)
    gamma = env.arrival_price * env.permanent_impact / max(env.total_shares, 1)
    schedule = discrete_almgren_chriss_schedule(
        total_shares=env.total_shares,
        horizon=env.horizon,
        sigma_price=sigma_price,
        eta=eta,
        gamma=gamma,
        risk_aversion=env.risk_aversion,
    )

    done = False
    while not done:
        if env.step_idx == env.horizon - 1:
            planned_shares = env.remaining_shares
        else:
            planned_shares = min(
                int(schedule[min(env.step_idx, len(schedule) - 1)]), env.remaining_shares
            )
        _, _, terminated, truncated, _ = env.step(env.target_shares_to_action(planned_shares))
        done = terminated or truncated

    return summarize_execution(env)


# %% [markdown]
# ## 4. Train the PPO agent
#
# PPO holds an explicit policy over the pacing action and improves it with a
# policy-gradient step, clipping any update that would move the action
# distribution far from the one that collected the data. The entropy bonus
# `ent_coef` keeps the policy from collapsing onto a single pace early, which on
# this environment is the usual failure: an agent that discovers TWAP quickly
# has little incentive to explore away from it.


# %%
def make_env(seed: int):
    """Factory returning a calibrated execution environment with this notebook's reward weights."""

    def _init():
        return ExecutionEnv(
            total_shares=TOTAL_SHARES,
            horizon=EXECUTION_HORIZON,
            cal_params=cal_params,
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
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01,
    device="cpu",
    seed=SEED,
    verbose=0,
)

# %%
print(f"Training PPO for {TOTAL_TIMESTEPS:,} steps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS)
print("Training complete")


# %%
def ppo_execution(env: ExecutionEnv, reset_seed: int | None = None) -> dict:
    """Execute with the trained PPO policy, acting greedily on its own observation."""
    obs, _ = env.reset(seed=reset_seed)
    done = False

    while not done:
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action[0])
        done = terminated or truncated

    return summarize_execution(env)


# %% [markdown]
# ## 5. Evaluate the three strategies on the same paths
#
# Every strategy is run on the same list of episode seeds, and a fresh
# environment is constructed for each so nothing carries over between them.
# Because a seed fixes the whole market path, a difference between two
# strategies on the same seed is a difference between the schedules and not
# between the markets they happened to face.


# %%
def episode_diagnostics(env: ExecutionEnv, result: dict) -> dict:
    """Cost of one episode, plus where in the horizon the schedule put its volume."""
    history = result["history"]
    final_quarter_cutoff = max(env.horizon - env.horizon // 4, 0)
    final_quarter_volume = sum(
        float(h["shares_sold"]) for h in history if int(h["step"]) >= final_quarter_cutoff
    )
    return {
        "shortfall_bps": result["shortfall_bps"],
        "last_step_share": float(history[-1]["shares_sold"]) / env.total_shares,
        "final_quarter_share": final_quarter_volume / env.total_shares,
        "history": history,
    }


def evaluate_strategy(strategy_name: str, strategy_fn) -> tuple[dict, list[dict]]:
    """Run one strategy over every evaluation seed and collect its per-episode diagnostics."""
    diagnostics, all_paths = [], []
    for i in range(EVAL_EPISODES):
        episode_seed = EVAL_SEED_BASE + i
        env = ExecutionEnv(
            total_shares=TOTAL_SHARES,
            horizon=EXECUTION_HORIZON,
            cal_params=cal_params,
            risk_aversion=RISK_AVERSION,
            schedule_penalty=SCHEDULE_PENALTY,
            seed=episode_seed,
        )
        diag = episode_diagnostics(env, strategy_fn(env, reset_seed=episode_seed))
        diagnostics.append(diag)
        for h in diag["history"]:
            all_paths.append(
                {"strategy": strategy_name, "episode_id": i, "episode_seed": episode_seed, **h}
            )

    shortfalls = np.array([d["shortfall_bps"] for d in diagnostics])
    summary = {
        "shortfall_bps": shortfalls,
        "mean_bps": float(shortfalls.mean()),
        "std_bps": float(shortfalls.std()),
        "min_bps": float(shortfalls.min()),
        "max_bps": float(shortfalls.max()),
        "last_step_share_pct": 100 * float(np.mean([d["last_step_share"] for d in diagnostics])),
        "final_quarter_share_pct": 100
        * float(np.mean([d["final_quarter_share"] for d in diagnostics])),
    }
    return summary, all_paths


# %%
STRATEGIES = {
    "TWAP": twap_execution,
    "Almgren-Chriss": almgren_chriss_execution,
    "PPO": ppo_execution,
}
COLOR_BY_STRATEGY = {
    "TWAP": COLORS["blue"],
    "Almgren-Chriss": COLORS["amber"],
    "PPO": COLORS["copper"],
}

results, evaluation_paths = {}, []
for name, fn in STRATEGIES.items():
    results[name], strategy_paths = evaluate_strategy(name, fn)
    evaluation_paths.extend(strategy_paths)
    print(
        f"{name:16s}: {results[name]['mean_bps']:7.2f} bps mean, "
        f"{results[name]['std_bps']:6.2f} bps std across {EVAL_EPISODES} episodes"
    )

# %% [markdown]
# ## 6. The schedules side by side
#
# Averaging over the evaluation episodes step by step shows the shape of each
# schedule. The band on the top panel is the tenth to ninetieth percentile of
# the per-episode execution rate, which is where a state-responsive policy
# separates from a static one: a fixed schedule trades the same quantity
# whatever the market does, so its band is narrow by construction.

# %%
evaluation_df = pl.DataFrame(evaluation_paths).with_columns(
    pl.col("shortfall").cum_sum().over(["strategy", "episode_id"]).alias("cum_shortfall")
)
trajectory_stats = (
    evaluation_df.group_by(["strategy", "step"])
    .agg(
        pl.col("shares_sold").mean().alias("mean_shares_sold"),
        pl.col("remaining").mean().alias("mean_remaining"),
        pl.col("cum_shortfall").mean().alias("mean_cum_shortfall"),
        pl.col("shares_sold").quantile(0.1).alias("p10_shares_sold"),
        pl.col("shares_sold").quantile(0.9).alias("p90_shares_sold"),
    )
    .sort(["strategy", "step"])
)


# %%
def add_execution_rate_band(fig, strategy_df: pl.DataFrame, name: str, color: str) -> None:
    """Shade the tenth-to-ninetieth-percentile execution rate for one strategy."""
    steps = strategy_df["step"].to_list()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=strategy_df["p90_shares_sold"].to_list(),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=name,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=strategy_df["p10_shares_sold"].to_list(),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=color,
            opacity=0.12,
            showlegend=False,
            hoverinfo="skip",
            legendgroup=name,
        ),
        row=1,
        col=1,
    )


# %%
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Execution rate (shares per step)",
        "Remaining inventory",
        "Cumulative implementation shortfall",
    ],
    vertical_spacing=0.10,
)
for name, color in COLOR_BY_STRATEGY.items():
    strategy_df = trajectory_stats.filter(pl.col("strategy") == name).sort("step")
    steps = strategy_df["step"].to_list()
    add_execution_rate_band(fig, strategy_df, name, color)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=strategy_df["mean_shares_sold"].to_list(),
            name=name,
            line=dict(color=color, width=2),
            legendgroup=name,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=strategy_df["mean_remaining"].to_list(),
            name=name,
            line=dict(color=color, width=2),
            legendgroup=name,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=strategy_df["mean_cum_shortfall"].to_list(),
            name=name,
            line=dict(color=color, width=2),
            legendgroup=name,
            showlegend=False,
        ),
        row=3,
        col=1,
    )

fig.update_yaxes(title_text="Shares per step", row=1, col=1)
fig.update_yaxes(title_text="Shares unsold", row=2, col=1)
fig.update_yaxes(title_text="Shortfall ($)", row=3, col=1)
fig.update_xaxes(title_text="Step", row=3, col=1)
fig.update_layout(
    title=(
        "Execution rate, remaining inventory and cost, averaged over episodes"
        "<br><sup>Lines are means over the evaluation episodes; the band on the top panel is the "
        "tenth to ninetieth percentile of the execution rate</sup>"
    ),
    height=760,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.show()

# %% [markdown]
# ## 7. Cost, and where in the horizon it was incurred
#
# The two figures below are the ones the conclusions rest on. The first is the
# distribution of per-episode cost, which decides whether a difference between
# two averages is worth reading at all. The second is where each schedule put
# its volume, which decides whether a low average cost was earned or borrowed
# from the end of the horizon.

# %%
fig = go.Figure()
for name, color in COLOR_BY_STRATEGY.items():
    fig.add_trace(
        go.Box(
            y=results[name]["shortfall_bps"],
            name=name,
            marker=dict(color=color),
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
        )
    )
fig.update_layout(
    title="Per-episode implementation shortfall, one point per evaluation episode",
    yaxis_title="Implementation shortfall (bps of arrival notional)",
    xaxis_title="Strategy",
    showlegend=False,
    height=460,
)
fig.show()

# %% [markdown]
# ### Where the volume sits
#
# A schedule that leaves inventory for the end must trade it at whatever the
# book charges then, and the final step ignores the participation cap. Two
# shares measure the exposure: the fraction of the order traded in the final
# quarter of the horizon, and the fraction traded on the last step alone. An
# even schedule puts a quarter of the order in the final quarter and one
# sixtieth of it on the last step, which is the reference to read these
# against.

# %%
fig = go.Figure()
for label, key in [
    (f"Final quarter (last {EXECUTION_HORIZON // 4} steps)", "final_quarter_share_pct"),
    ("Last step alone", "last_step_share_pct"),
]:
    fig.add_trace(
        go.Bar(
            x=list(STRATEGIES),
            y=[results[name][key] for name in STRATEGIES],
            name=label,
        )
    )
fig.add_hline(
    y=25.0,
    line=dict(color=COLORS["neutral"], width=1, dash="dot"),
    annotation_text="even pace, final quarter",
)
fig.update_layout(
    title="Share of the order traded late in the horizon, by strategy",
    xaxis_title="Strategy",
    yaxis_title="Share of the parent order (%)",
    barmode="group",
    height=440,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.show()

# %% [markdown]
# ## 8. Summary

# %%
pl.DataFrame(
    [
        {
            "Strategy": name,
            "Mean IS (bps)": round(r["mean_bps"], 2),
            "Std (bps)": round(r["std_bps"], 2),
            "Min (bps)": round(r["min_bps"], 2),
            "Max (bps)": round(r["max_bps"], 2),
            "Final quarter (%)": round(r["final_quarter_share_pct"], 1),
            "Last step (%)": round(r["last_step_share_pct"], 1),
        }
        for name, r in results.items()
    ]
)

# %% [markdown]
# ## 9. Key takeaways

# %%
even_final_quarter = 100 * (EXECUTION_HORIZON // 4) / EXECUTION_HORIZON
lowest_mean = min(results, key=lambda name: results[name]["mean_bps"])
REFERENCE_STRATEGY = "TWAP"
shortfall_by_strategy = {name: r["shortfall_bps"] for name, r in results.items()}


def comparison_row(reference: str, other: str, values: dict, unit: str) -> str:
    """One line comparing two arms, with the paired and unpaired standard errors side by side.

    Pairing changes the variance of a difference by minus twice the covariance, so it narrows
    the interval only when the two arms move together. Reporting both, with the correlation,
    lets the reader see which of the two applies rather than take it on trust.
    """
    a, b = values[reference], values[other]
    difference = a - b
    n = difference.size
    paired_se = float(difference.std(ddof=1) / np.sqrt(n))
    unpaired_se = float(np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n))
    correlation = float(np.corrcoef(a, b)[0, 1])
    return (
        f"- **{other}** minus **{reference}**: {-difference.mean():+.2f} {unit} per episode. "
        f"Paired standard error {paired_se:.2f}, treating the arms as independent "
        f"{unpaired_se:.2f}; the two series correlate {correlation:+.2f}."
    )


paired_lines = "\n".join(
    comparison_row(REFERENCE_STRATEGY, name, shortfall_by_strategy, "bps")
    for name in STRATEGIES
    if name != REFERENCE_STRATEGY
)
profile_lines = "\n".join(
    f"- **{name}**: {results[name]['mean_bps']:.1f} bps mean shortfall, standard deviation "
    f"{results[name]['std_bps']:.1f} bps, {results[name]['final_quarter_share_pct']:.1f}% of the "
    f"order in the final quarter and {results[name]['last_step_share_pct']:.1f}% on the last step."
    for name in STRATEGIES
)

display(
    Markdown(f"""
{profile_lines}

**Size the difference, and say which standard error you sized it with.** Every strategy
traded the same market paths, so the difference to the fixed schedule can be taken episode by
episode:

{paired_lines}

Size each difference with its paired standard error: the episodes are matched, so that figure
already carries the covariance between the two arms. The independent-samples figure beside it
shows what the matching bought, which is not guaranteed to be positive - matching narrows the
interval when the arms move together and widens it when they move apart.
**{lowest_mean}** records the lowest mean over {EVAL_EPISODES} episodes, which on this evidence
is not an ordering to carry out of the notebook.

**A schedule's cost and its shape are separate facts.** An even schedule puts
{even_final_quarter:.0f}% of the order in the final quarter. A strategy that puts substantially
more there has moved cost out of the part of the horizon where it could choose its moment and
into the step where it has no choice at all, and the average shortfall alone will not show you
that it did.

**The information a benchmark has is part of the benchmark.** The Almgren-Chriss schedule here
reads the average volatility and depth of the whole episode before its first trade. Any
comparison against it is a comparison against a plan that knew the future, which is worth making
and is not a statement about which method a desk should run.

### Known limitations

- The trading cost is driven by a spread proxy taken from the hourly high-low range, and by
  impact coefficients that are the calibration's clamp rather than an estimate. The absolute
  cost figures are therefore not a forecast of what this order would cost; only the comparison
  between schedules facing the same costs is meaningful.
- Fills are assumed at the modelled price with no queue, no rejection and no latency. There is
  no other participant reacting to the order, so the impact model is the only feedback the market
  has.
- One PPO training run is one draw from a wide distribution over seeds. Distinguishing the
  method from the seed would need repeated training runs.
- The regime process makes stressed steps deeper as well as more volatile, because the depth
  parameter is fitted to traded volume. A simulator built to stress-test execution would need a
  liquidity series that thins when volatility rises.

**Next**: `03_market_making_ppo` moves from liquidating a known order to quoting both sides,
where the agent chooses its inventory instead of being handed it.
""")
)
