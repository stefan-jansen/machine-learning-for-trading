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
# # RL Algorithm Comparison: DQN, PPO and A2C on One Trading Environment
#
# **Chapter 21: Reinforcement Learning for Execution and Hedging**
#
# ## Purpose
#
# Three reinforcement-learning algorithms are trained on the same trading
# environment with the same interaction budget, then evaluated on the same
# episodes. The point of running them side by side is not to rank them. It is
# to show that a single average reward says almost nothing about what a policy
# actually does, and that a second diagnostic - which positions the policy
# chooses, and how often it changes them - separates policies that the reward
# number leaves indistinguishable.
#
# ## Learning objectives
#
# After working through this notebook you will be able to:
#
# - Fit a GARCH(1,1) volatility model to hourly cryptocurrency returns and use
#   its coefficients to drive a simulated return series that keeps the
#   clustering of the source data.
# - Write a Gymnasium trading environment whose reward pays the position chosen
#   one step earlier, so that an action can never earn the return that was
#   visible when it was taken.
# - Train a value-based agent (DQN) and two actor-critic agents (PPO, A2C) with
#   Stable-Baselines3 on identical inputs, and evaluate them on a fixed list of
#   episode seeds so all three face the same price paths.
# - Read a policy's behaviour off the distribution of positions it chooses,
#   and say why an average reward cannot tell you whether a policy is trading
#   or standing still.
#
# ## Book reference
#
# Section 21.3, *Core algorithms - From DQN to actor-critic*.
#
# ## Prerequisites
#
# - `stable-baselines3` and `gymnasium`, both in the repository environment.
# - Hourly perpetual-futures bars for `BTCUSDT`, read through
#   `data.load_crypto_perps`, which the GARCH calibration in section 1 fits.
#   The environment itself is simulated; only the volatility model is estimated
#   from market data.

# %% [markdown]
# ## Setup

# %%
"""RL algorithm comparison - DQN, PPO and A2C on one GARCH-calibrated trading environment."""

import warnings

import gymnasium as gym
import numpy as np
import plotly.graph_objects as go
import polars as pl
from gymnasium import spaces
from IPython.display import Markdown, display
from plotly.subplots import make_subplots

# Deprecation notices from these two dependencies repeat on every environment
# construction and say nothing about this run. Convergence, overflow and
# invalid-value warnings stay visible: they report a condition results depend on.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="stable_baselines3")
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

from rl_calibration import CryptoMarketCalibrator
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

import utils  # noqa: F401  - sets the Plotly renderer so figures carry a static PNG
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
CALIBRATION_SYMBOL = "BTCUSDT"  # perpetual-futures symbol the GARCH model is fitted to
STEPS_PER_EPISODE = 252  # one step is one hour, so an episode is ~10.5 days of trading
TOTAL_TIMESTEPS = 50_000  # environment steps each algorithm may consume while learning
TRADING_COST_BPS = 10  # cost of moving the position by one unit, in basis points
EVAL_EPISODES = 10  # episodes each trained policy is scored on
VOL_WINDOW = 20  # steps in the observation's volatility estimate
MOM_WINDOW = 5  # steps in the observation's momentum sum
SEED = 314

# %%
set_global_seeds(SEED)

# %% [markdown]
# ### What the settings decide
#
# Two of these settings do most of the work. `TOTAL_TIMESTEPS` is the number of
# environment steps each algorithm is allowed to consume before it is frozen and
# scored, and it is held equal across the three so that any difference in what
# they learn is a difference in how they learn rather than in how much practice
# they had. `TRADING_COST_BPS` is charged on every unit of position change, so
# it is what makes standing still a defensible policy: an agent that flips
# between long and short every step pays it twice as often as one that holds.

# %%
display(
    Markdown(f"""
- **Episode length**: {STEPS_PER_EPISODE} steps of one hour each, about
  {STEPS_PER_EPISODE / 24:.0f} days of continuous trading.
- **Learning budget**: {TOTAL_TIMESTEPS:,} environment steps per algorithm, or roughly
  {TOTAL_TIMESTEPS / STEPS_PER_EPISODE:.0f} episodes of experience.
- **Trading cost**: {TRADING_COST_BPS} basis points per unit of position change, charged on
  the size of the change, so a flip from short to long costs twice a move from flat to long.
- **Evaluation**: {EVAL_EPISODES} episodes, drawn from seeds none of the three saw while
  learning, and identical across all three.
- **Observation windows**: volatility over {VOL_WINDOW} steps, momentum summed over
  {MOM_WINDOW} steps.
""")
)

# %% [markdown]
# ## 1. The market the agents trade
#
# The agents interact with a simulator rather than with recorded prices,
# because an agent's actions have to change what it sees next and a fixed price
# series cannot respond. What the simulator must not do is invent a market
# whose statistics no real market has. The return process is therefore a
# **GARCH(1,1)** model estimated on real data.
#
# GARCH(1,1) models the variance of a return series as a function of its own
# recent history:
#
# $$\sigma_t^2 = \omega + \alpha \, r_{t-1}^2 + \beta \, \sigma_{t-1}^2$$
#
# The variance at each step is a constant $\omega$, plus a reaction $\alpha$ to
# the size of the last return, plus a persistence term $\beta$ carrying forward
# the previous variance. When $\alpha$ and $\beta$ sum close to one, a large
# return raises the variance and the variance decays only slowly, so large
# moves arrive in runs. That property is called **volatility clustering**, and
# it is the single most robust statistical regularity in speculative price
# series: quiet hours follow quiet hours and violent hours follow violent ones,
# even though the direction of the next move stays unpredictable.

# %% [markdown]
# ### Fit the volatility model
#
# `CryptoMarketCalibrator` loads the hourly bars, takes log returns of the
# close, and fits GARCH(1,1) with the `arch` package. It then imposes one
# constraint the raw fit does not: if $\alpha + \beta$ comes back at or above
# the stationarity ceiling it enforces, both coefficients are scaled down so
# their sum sits exactly on that ceiling. An estimated persistence of one or
# more would mean a variance process with no long-run level, which would make a
# simulated path wander to arbitrary scale instead of reverting to the
# volatility the data shows. The table below prints the persistence that
# survived, so the reader can see whether the ceiling bound.

# %%
calibrator = CryptoMarketCalibrator(CALIBRATION_SYMBOL)
btc_bars = calibrator.load_data()
btc_returns = calibrator.compute_returns()
garch = calibrator.fit_garch()

GARCH_OMEGA, GARCH_ALPHA, GARCH_BETA = garch.omega, garch.alpha, garch.beta
UNCOND_VOL = garch.unconditional_vol

# %% [markdown]
# ### What the calibration read, and what came out

# %%
display(
    Markdown(f"""
`{CALIBRATION_SYMBOL}` hourly bars: **{btc_bars.height:,}** observations from
{btc_bars["timestamp"].min():%Y-%m-%d} to {btc_bars["timestamp"].max():%Y-%m-%d},
giving {len(btc_returns):,} log returns.

| GARCH(1,1) coefficient | Value | What it controls |
|---|---|---|
| $\\alpha$ | {GARCH_ALPHA:.4f} | how strongly the variance reacts to the last return |
| $\\beta$ | {GARCH_BETA:.4f} | how much of the previous variance is carried forward |
| $\\alpha + \\beta$ | {GARCH_ALPHA + GARCH_BETA:.4f} | persistence: how slowly a volatility shock decays |
| $\\omega$ | {GARCH_OMEGA:.2e} | the constant that fixes the long-run variance |

The unconditional hourly volatility implied by these coefficients is
{UNCOND_VOL:.4f}, or {UNCOND_VOL * 100:.2f}% per hour.
""")
)

# %% [markdown]
# ### Simulate a return path
#
# One path is drawn by stepping the variance recursion forward and taking a
# normal draw at each step. The path starts at the long-run variance, so no
# burn-in is needed for the level; only the clustering has to build up.


# %%
def garch_return_path(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate one return path from the calibrated GARCH(1,1) coefficients."""
    returns = np.zeros(n_steps)
    variance = UNCOND_VOL**2
    for t in range(n_steps):
        returns[t] = rng.normal(0, np.sqrt(variance))
        variance = GARCH_OMEGA + GARCH_ALPHA * returns[t] ** 2 + GARCH_BETA * variance
        variance = max(variance, 1e-8)
    return returns


# %% [markdown]
# ### Does the simulator keep the clustering?
#
# Volatility clustering shows up as autocorrelation in *squared* returns: if
# the size of a move predicts the size of the next one, then $r_t^2$ and
# $r_{t-k}^2$ are correlated even though $r_t$ and $r_{t-k}$ are not. The
# figure below puts the source data next to a simulated path of the same
# length. The left panel shows the realised hourly volatility of the market
# data over the sample, which is what "clustering" looks like as a time series;
# the right panel measures it, on the market data and on the simulator.


# %%
def squared_return_acf(returns: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation of squared returns at lags 1..max_lag."""
    squared = returns**2 - np.mean(returns**2)
    denom = np.dot(squared, squared)
    return np.array([np.dot(squared[k:], squared[:-k]) / denom for k in range(1, max_lag + 1)])


MAX_LAG = 48
sim_returns = garch_return_path(len(btc_returns), np.random.default_rng(SEED))

realised_vol = (
    pl.DataFrame({"timestamp": btc_bars["timestamp"][1:], "ret": btc_returns})
    .with_columns(pl.col("ret").rolling_std(window_size=24).alias("vol"))
    .drop_nulls()
)

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        f"Realised {CALIBRATION_SYMBOL} volatility (24-hour rolling)",
        "Autocorrelation of squared returns",
    ],
)
fig.add_trace(
    go.Scatter(
        x=realised_vol["timestamp"],
        y=realised_vol["vol"] * 100,
        line=dict(color=COLORS["blue"], width=1),
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=list(range(1, MAX_LAG + 1)),
        y=squared_return_acf(btc_returns, MAX_LAG),
        name=f"{CALIBRATION_SYMBOL} hourly",
        line=dict(color=COLORS["blue"], width=2),
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=list(range(1, MAX_LAG + 1)),
        y=squared_return_acf(sim_returns, MAX_LAG),
        name="GARCH simulator",
        line=dict(color=COLORS["amber"], width=2, dash="dash"),
    ),
    row=1,
    col=2,
)
fig.add_hline(y=0, line=dict(color=COLORS["neutral"], width=1, dash="dot"), row=1, col=2)
fig.update_yaxes(title_text="Hourly volatility (%)", row=1, col=1)
fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_yaxes(title_text="Autocorrelation", row=1, col=2)
fig.update_xaxes(title_text="Lag (hours)", row=1, col=2)
fig.update_layout(
    title="Volatility clusters in the market data, and the simulator keeps it",
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
show_plotly_with_alt(
    fig,
    "Two panels. Left: a 24-hour rolling standard deviation of hourly BTCUSDT returns over the sample, showing long quiet stretches broken by clusters of high volatility. Right: the autocorrelation of squared returns at lags of one to forty-eight hours for the market data and for one simulated GARCH path, both staying well above zero across the whole range.",
)

# %% [markdown]
# The autocorrelation of squared returns stays positive for tens of hours in
# both series, which is the property the agents need their market to have. It
# is also the only property this simulator has: given the current variance, the
# next return is drawn around zero, so its expected value is zero whatever the
# volatility state is. With a nonnegative trading cost on top, no policy in this
# environment has a positive expected reward. A policy that ends an evaluation
# above zero got there through sampling variation; what the volatility state can
# genuinely change is the size of the exposure a policy takes and the cost it
# pays for changing it, not the sign of what it earns.

# %% [markdown]
# ## 2. The trading environment
#
# The environment is a Gymnasium `Env` with three discrete actions - short
# (-1), flat (0) and long (+1) - and a five-element observation. Two design
# decisions in it matter more than the rest.
#
# **The reward pays the position that was already held.** At step $t$ the agent
# sees the return $r_t$ that has just happened. The reward it collects is
# $p_{t-1} r_t$, the return earned by the position chosen at the *previous*
# step, minus the cost of any change. The action taken now is installed for
# $t+1$. Without this ordering the agent could act on a return it has already
# observed, which is the look-ahead that makes a simulated trading result
# meaningless.
#
# **The observation is scaled to comparable magnitudes.** Hourly returns are a
# small fraction of a percent, and a neural network with default weight
# initialisation handles inputs of order one far better than inputs three orders
# of magnitude smaller. The volatility and momentum features are therefore
# multiplied by ten before they enter the observation, which changes nothing
# about their information content.

# %% [markdown]
# ### The observation


# %%
def build_observation(returns: np.ndarray, idx: int, position: int) -> np.ndarray:
    """Build the five-element observation for step `idx`, using no return after `idx`."""
    ret = returns[idx]
    ret_lag = returns[idx - 1] if idx > 0 else 0.0
    vol = np.std(returns[max(0, idx + 1 - VOL_WINDOW) : idx + 1])
    mom = np.sum(returns[max(0, idx + 1 - MOM_WINDOW) : idx + 1])
    return np.array([ret, ret_lag, vol * 10, mom * 10, position], dtype=np.float32)


# %% [markdown]
# The five elements are the current return, the previous return, the standard
# deviation of the last `VOL_WINDOW` returns, the sum of the last `MOM_WINDOW`
# returns, and the position currently held. Every window ends at `idx`, so
# nothing after the current step enters the observation.

# %% [markdown]
# ### The environment


# %%
class SimpleTradingEnv(gym.Env):
    """Three-action trading environment over a simulated GARCH return path."""

    def __init__(
        self,
        n_steps: int = STEPS_PER_EPISODE,
        trading_cost_bps: float = TRADING_COST_BPS,
        seed: int | None = None,
    ):
        super().__init__()
        self.n_steps, self.trading_cost = n_steps, trading_cost_bps / 10_000
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.returns = garch_return_path(self.n_steps, self.rng)
        self.step_idx, self.position, self.nav = 1, 0, 1.0
        return build_observation(self.returns, self.step_idx, self.position), {}

    def step(self, action: int):
        reward_position = self.position  # chosen last step; this is what earns r_t
        new_position = action - 1
        trade = abs(new_position - reward_position)
        strategy_return = reward_position * self.returns[self.step_idx] - trade * self.trading_cost

        self.position = new_position
        self.nav *= 1 + strategy_return
        self.step_idx += 1

        terminated = self.step_idx >= self.n_steps - 1
        info = dict(
            nav=self.nav,
            reward_position=reward_position,
            next_position=new_position,
            strategy_return=strategy_return,
            trade=trade,
        )
        obs = build_observation(self.returns, self.step_idx, self.position)
        return obs, strategy_return * 100, terminated, False, info


# %% [markdown]
# The reward is the step return in percent rather than as a fraction. The scale
# of a reward changes the effective size of every gradient step, and a raw
# hourly return is small enough that the default learning rates in
# Stable-Baselines3 make almost no progress within the budget set here.

# %% [markdown]
# ## 3. Train the three algorithms
#
# The three algorithms differ in where their learning signal comes from.
#
# **DQN** is *value-based* and *off-policy*. It learns an action-value function
# $Q(s, a)$, the expected discounted reward from taking action $a$ in state $s$
# and behaving greedily afterwards, and acts by taking the highest-valued
# action. Because $Q$ can be updated from any past transition, DQN stores its
# experience in a replay buffer and re-uses it, which is what *off-policy*
# means and why it needs fewer fresh environment steps per unit of progress.
#
# **PPO** is an *actor-critic* method and *on-policy*. It holds an explicit
# policy, updates it directly with a policy-gradient step, and uses a learned
# value function only to reduce the variance of that gradient. On-policy means
# each batch of experience is collected under the current policy and discarded
# after use. PPO's contribution is the clipped objective: an update that would
# move the action probabilities more than `clip_range` away from the policy
# that collected the data is truncated, which is what keeps it stable.
#
# **A2C** is the same actor-critic idea without the clipping, updating from
# very short rollouts. It is the cheaper and less stable member of the family,
# and it is here because the action space is discrete: SAC, the usual
# actor-critic reference point, requires continuous actions.
#
# All three are given the same environment, the same seed and the same budget.
# Each gets its own environment instance so one algorithm's training cannot
# advance the random state the next one starts from.


# %%
def make_env(seed: int):
    """Factory returning a fresh environment seeded for reproducible paths."""

    def _init():
        return SimpleTradingEnv(
            n_steps=STEPS_PER_EPISODE, trading_cost_bps=TRADING_COST_BPS, seed=seed
        )

    return _init


# The MLP policies here have a few thousand parameters. On networks this small the
# cost of moving a batch to a GPU exceeds the cost of the arithmetic, so CPU is faster.
model_dqn = DQN(
    "MlpPolicy",
    DummyVecEnv([make_env(seed=SEED)]),
    learning_rate=1e-4,
    buffer_size=10_000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    device="cpu",
    seed=SEED,
    verbose=0,
)
model_ppo = PPO(
    "MlpPolicy",
    DummyVecEnv([make_env(seed=SEED)]),
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    device="cpu",
    seed=SEED,
    verbose=0,
)
model_a2c = A2C(
    "MlpPolicy",
    DummyVecEnv([make_env(seed=SEED)]),
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    device="cpu",
    seed=SEED,
    verbose=0,
)

# %%
models = {"DQN": model_dqn, "PPO": model_ppo, "A2C": model_a2c}
for name, model in models.items():
    print(f"Training {name} for {TOTAL_TIMESTEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
print("Training complete")

# %% [markdown]
# ## 4. Evaluate on held-out episodes
#
# Each policy is scored on the same explicit list of episode seeds, so all
# three face byte-identical price paths and the comparison is paired. A fresh
# `DummyVecEnv` is built for every seed, which stops one policy's evaluation
# from advancing the random state the next policy would start from.

# %%
eval_seeds = list(range(123, 123 + EVAL_EPISODES))
episode_rewards = {}

for name, model in models.items():
    rewards = []
    for seed in eval_seeds:
        eval_env = DummyVecEnv([make_env(seed=seed)])
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True)
        rewards.append(float(mean_reward))
    episode_rewards[name] = np.array(rewards)
    print(
        f"{name}: {np.mean(rewards):+.2f} mean, {np.std(rewards):.2f} std over {EVAL_EPISODES} episodes"
    )

# %% [markdown]
# ### Reward against episode-to-episode spread
#
# The bar heights are the averages just printed and the whiskers are one
# standard deviation of the ten episode rewards. The spread is what decides
# whether a difference between two bars is worth reading.

# %%
fig = go.Figure(
    go.Bar(
        x=list(episode_rewards),
        y=[float(np.mean(r)) for r in episode_rewards.values()],
        error_y=dict(
            type="data",
            array=[float(np.std(r)) for r in episode_rewards.values()],
            color=COLORS["neutral"],
        ),
        marker=dict(color=COLORS["blue"]),
    )
)
fig.update_layout(
    title="Mean reward across held-out episodes, with one standard deviation",
    xaxis_title="Algorithm",
    yaxis_title="Episode reward",
    height=400,
)
show_plotly_with_alt(
    fig,
    "A bar per algorithm giving its mean reward over the held-out evaluation episodes, with a whisker of one standard deviation of the per-episode rewards. The whiskers are long relative to the differences between the bar heights.",
)

# %% [markdown]
# ## 5. What the policies actually do
#
# A mean reward is one number per policy. To see the behaviour behind it, each
# policy is run once more on a single shared episode and its whole trajectory
# is recorded: the position that earned each step's return, the position it
# chose for the next step, the reward, and the net asset value (**NAV**, the
# value of one unit of capital compounded by the step returns).


# %%
def run_episode(model, env) -> dict:
    """Run one deterministic episode and collect the full trajectory."""
    obs, done = env.reset(), False
    reward_positions, chosen_positions, rewards, navs, total_trades = [], [], [], [1.0], 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        reward_positions.append(info[0]["reward_position"])
        chosen_positions.append(info[0]["next_position"])
        rewards.append(float(reward[0]))
        navs.append(info[0]["nav"])
        total_trades += info[0]["trade"]

    return {
        "reward_positions": np.array(reward_positions),
        "chosen_positions": np.array(chosen_positions),
        "rewards": np.array(rewards),
        "navs": np.array(navs),
        "final_nav": navs[-1],
        "total_reward": float(np.sum(rewards)),
        "total_trades": int(total_trades),
    }


DIAGNOSTIC_SEED = 456
trajectories = {
    name: run_episode(model, DummyVecEnv([make_env(seed=DIAGNOSTIC_SEED)]))
    for name, model in models.items()
}

# %% [markdown]
# ### The trajectories side by side
#
# All three panels are drawn from the same episode, so the return series is
# identical and every difference between the lines is a difference in policy.

# %%
styles = {
    "DQN": {"color": COLORS["blue"], "dash": "solid", "pattern": "/"},
    "PPO": {"color": COLORS["amber"], "dash": "dash", "pattern": "x"},
    "A2C": {"color": COLORS["neutral"], "dash": "dot", "pattern": "."},
}

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Net asset value", "Position earning the step return", "Cumulative reward"],
    vertical_spacing=0.08,
)
for name, traj in trajectories.items():
    style = dict(color=styles[name]["color"], dash=styles[name]["dash"], width=2)
    fig.add_trace(
        go.Scatter(y=traj["navs"], name=name, line=style, legendgroup=name),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=traj["reward_positions"], line=style, legendgroup=name, showlegend=False),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=np.cumsum(traj["rewards"]), line=style, legendgroup=name, showlegend=False),
        row=3,
        col=1,
    )

fig.add_hline(y=1.0, line=dict(color=COLORS["neutral"], width=1, dash="dot"), row=1, col=1)
fig.update_yaxes(title_text="NAV (start = 1.0)", row=1, col=1)
fig.update_yaxes(title_text="Position", tickvals=[-1, 0, 1], row=2, col=1)
fig.update_yaxes(title_text="Cumulative reward", row=3, col=1)
fig.update_xaxes(title_text="Step (hours)", row=3, col=1)
fig.update_layout(
    title="NAV, position and cumulative reward on one shared episode",
    height=800,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
show_plotly_with_alt(
    fig,
    "Three stacked panels sharing a step axis, one line per algorithm on the same episode: net asset value starting at one, the position earning each step's return between minus one and plus one, and cumulative reward.",
)

# %% [markdown]
# ### The distribution of chosen positions
#
# The middle panel above is hard to read once two policies sit on the same
# line. Counting how many steps each policy spends in each position says the
# same thing without the overplotting, and it is the diagnostic that separates
# a policy that is trading from one that has settled on a single position and
# stopped.

# %%
fig = go.Figure()
for name, traj in trajectories.items():
    chosen = traj["chosen_positions"]
    fig.add_trace(
        go.Bar(
            x=["Short (-1)", "Flat (0)", "Long (+1)"],
            y=[int(np.sum(chosen == v)) for v in (-1, 0, 1)],
            name=name,
            marker=dict(
                color=styles[name]["color"],
                pattern=dict(shape=styles[name]["pattern"]),
                line=dict(color=COLORS["neutral"], width=0.5),
            ),
        )
    )
fig.update_layout(
    title="Steps spent in each position over one episode, by algorithm",
    xaxis_title="Chosen position",
    yaxis_title="Steps",
    barmode="group",
    height=400,
)
show_plotly_with_alt(
    fig,
    "Grouped bars counting how many steps of one episode each algorithm spent short, flat and long.",
)

# %% [markdown]
# ### Trajectory summary
#
# `Turnover` counts units of position change over the episode: a policy that
# never moves scores zero, and one that flips between long and short every step
# scores twice the episode length.

# %%
pl.DataFrame(
    [
        {
            "Algorithm": name,
            "Final NAV": round(traj["final_nav"], 4),
            "Total reward": round(traj["total_reward"], 2),
            "Distinct positions used": int(np.unique(traj["chosen_positions"]).size),
            "Turnover": traj["total_trades"],
        }
        for name, traj in trajectories.items()
    ]
)

# %% [markdown]
# ## 6. Key takeaways
#
# The three algorithms were scored on the same list of seeds, so each pair of
# reward series is matched episode by episode and the difference can be taken
# within an episode. The paired standard error that comes out of that difference
# is the correct one to judge it by, because it carries the covariance between
# the two arms whatever its sign. What it does *not* guarantee is a narrower
# interval: pairing changes the variance of a difference by
# $-2\,\mathrm{cov}(a, b)$, so matching helps when the arms move together and
# hurts when they move against each other, and two policies holding opposite
# positions on the same price path move against each other. The
# independent-samples standard error is therefore reported alongside, not to be
# used for inference but to make visible how much the matching changed.

# %%
switch_counts = {
    name: int(np.count_nonzero(np.diff(traj["chosen_positions"])))
    for name, traj in trajectories.items()
}
distinct_positions = {
    name: int(np.unique(traj["chosen_positions"]).size) for name, traj in trajectories.items()
}
highest_mean = max(episode_rewards, key=lambda name: episode_rewards[name].mean())


def comparison_row(reference: str, other: str) -> str:
    """One line comparing two arms, with the paired and unpaired standard errors side by side."""
    a, b = episode_rewards[reference], episode_rewards[other]
    difference = a - b
    n = difference.size
    paired_se = float(difference.std(ddof=1) / np.sqrt(n))
    unpaired_se = float(np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n))
    correlation = float(np.corrcoef(a, b)[0, 1])
    return (
        f"- **{reference}** minus **{other}**: {difference.mean():+.2f} per episode. "
        f"Paired standard error {paired_se:.2f}, treating the arms as independent {unpaired_se:.2f}; "
        f"the two reward series correlate {correlation:+.2f}."
    )


paired_lines = "\n".join(
    comparison_row(highest_mean, name) for name in episode_rewards if name != highest_mean
)
behaviour_lines = "\n".join(
    f"- **{name}** used {distinct_positions[name]} of the three positions and changed position "
    f"{switch_counts[name]} times over the diagnostic episode."
    for name in models
)

display(
    Markdown(f"""
**Size a difference before reading it.** **{highest_mean}** reaches the highest mean reward
over the {EVAL_EPISODES} held-out episodes. Every algorithm was scored on the same seeds, so
the difference to each of the others can be taken episode by episode:

{paired_lines}

The paired standard error is the one to size each difference with: the episodes are matched,
so it already accounts for the covariance between the arms whichever way that covariance
points. The independent-samples figure is beside it to show what the matching did. Where the
correlation is negative - two policies sitting on opposite sides of the same price path -
matching widens the interval rather than narrowing it, which is a fact about these policies
worth seeing rather than a reason to use the other number. None of these means is a claim
about which algorithm learned a better policy. What the mean reward cannot tell you at all is
what each policy does:

{behaviour_lines}

**The diagnostic that separates them is the position distribution.** A policy that settles
on one position has stopped responding to its observation, and it will keep collecting a
respectable average reward for as long as that position happens to suit the market. Plotting
which actions a policy actually takes is the cheapest check that catches it, and it applies
to any RL agent whose action space is small enough to enumerate.

**Match the algorithm to the action space before anything else.** DQN needs discrete
actions because it maximises over them; the actor-critic methods do not, which is why the
execution and hedging notebooks that follow use PPO on continuous action spaces.

### Known limitations

- The simulator has no predictable drift, so there is no policy that could earn a
  consistent directional profit. What is being compared is how the three algorithms behave
  in a market that offers no edge, not how well they find one.
- A single seed per algorithm is one draw from a training distribution that is known to be
  wide for all three methods. Distinguishing the algorithms rather than the seeds would need
  repeated training runs, which is beyond this notebook's budget.
- Fills are assumed at the simulated return with a fixed linear cost, so there is no
  market impact, no partial fill and no latency. Section 21.8 and
  `07_backtest_with_impact` take up what changes when those assumptions are dropped.

**Next**: `02_optimal_execution_ppo` applies PPO to a continuous action space, where the
agent chooses how much of a parent order to trade at each step.
""")
)
