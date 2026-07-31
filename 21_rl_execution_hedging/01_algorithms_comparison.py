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
# # RL Algorithm Comparison: DQN vs PPO vs A2C
#
# **Execution environment**: local `uv run` on CPU. These small
# Stable-Baselines3 MLPs train faster on CPU than on GPU.
#
# This notebook compares value-based (DQN), on-policy actor-critic (PPO),
# and actor-critic baseline (A2C) algorithms on a simple trading environment.
#
# **Learning Objectives**:
# - Understand how different RL paradigms approach trading problems
# - Compare sample efficiency, stability, and final performance across algorithms
# - Analyze algorithm sensitivity to reward function design
#
# **Book Reference**: Chapter 21, Section 21.3 (Core Algorithms)
#
# **Prerequisites**: Stable-Baselines3 and Gymnasium installed.

# %%
"""RL Algorithm Comparison: DQN vs PPO vs A2C - compare value-based and actor-critic algorithms on a simple trading environment."""

# Core imports
import warnings

import numpy as np
import polars as pl
from IPython.display import Markdown, display

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

# RL frameworks
import gymnasium as gym
import plotly.graph_objects as go
from gymnasium import spaces
from plotly.subplots import make_subplots

# Calibration from real data
from rl_calibration import CryptoMarketCalibrator

# Stable-baselines3 algorithms
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

import utils  # noqa: F401
from utils.paths import get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

# %%
OUTPUT_DIR = get_output_dir(21, "algorithms_comparison")

# %% tags=["parameters"]
TIMESTEPS_PER_EPISODE = 252
TOTAL_TIMESTEPS = 50_000
EVAL_EPISODES = 10
SEED = 314  # Reproducible training across re-runs
EXPORT_RESULTS = False

# %%
set_global_seeds(SEED)

# %%
# Configuration
config = {
    "timesteps_per_episode": TIMESTEPS_PER_EPISODE,
    "total_timesteps": TOTAL_TIMESTEPS,
    "eval_episodes": EVAL_EPISODES,
}

print(f"Configuration: {config}")


# %%
# Calibrate from real crypto data
print("\nCalibrating GARCH from real crypto data...")
calibrator = CryptoMarketCalibrator("BTCUSDT")
cal_params = calibrator.get_execution_env_params()

# Extract GARCH parameters
GARCH_ALPHA = cal_params.garch.alpha
GARCH_BETA = cal_params.garch.beta
GARCH_OMEGA = cal_params.garch.omega
UNCOND_VOL = cal_params.garch.unconditional_vol

print(f"GARCH alpha: {GARCH_ALPHA:.4f}")
print(f"GARCH beta:  {GARCH_BETA:.4f}")
print(f"Uncond vol:  {UNCOND_VOL:.4f} ({UNCOND_VOL * 100:.2f}%)")

# %% [markdown]
# ## 1. Simple Trading Environment
#
# We create a minimal trading environment to compare algorithm behaviors.
# The environment provides:
# - State: Recent returns and technical features
# - Actions: Short (-1), Hold (0), Long (+1)
# - Reward: Position return minus transaction costs

# %% [markdown]
# ### Return-generating process
#
# We draw returns from a GARCH(1,1) process calibrated on real BTC hourly
# data so the agent faces realistic volatility clustering.


# %%
def garch_return_path(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate one return path from the calibrated GARCH(1,1)."""
    returns = np.zeros(n_steps)
    variance = UNCOND_VOL**2
    for t in range(n_steps):
        returns[t] = rng.normal(0, np.sqrt(variance))
        variance = GARCH_OMEGA + GARCH_ALPHA * returns[t] ** 2 + GARCH_BETA * variance
        variance = max(variance, 1e-8)
    return returns


# %% [markdown]
# ### Observation construction
#
# The observation stacks two recent returns, a 20-step volatility estimate,
# a 5-step momentum sum, and the current position.


# %%
def build_observation(returns: np.ndarray, idx: int, position: int) -> np.ndarray:
    """Build the 5D observation vector for the current step."""
    ret = returns[idx]
    ret_lag = returns[idx - 1] if idx > 0 else 0.0
    lookback = min(20, idx)
    vol = np.std(returns[max(0, idx - lookback) : idx + 1])
    mom = np.sum(returns[max(0, idx - 5) : idx + 1])
    return np.array([ret, ret_lag, vol * 10, mom * 10, position], dtype=np.float32)


# %% [markdown]
# ### Environment class
#
# A thin Gymnasium wrapper around the helpers above. At step $t$, the position
# chosen at $t-1$ earns return $r_t$; the new {short, hold, long} action is
# installed for $t+1$. This ordering lets the agent observe $r_t$ without
# earning that same return with an action chosen after seeing it.


# %%
class SimpleTradingEnv(gym.Env):
    def __init__(self, n_steps: int = 252, trading_cost_bps: float = 10, seed: int | None = None):
        super().__init__()
        self.n_steps, self.trading_cost = n_steps, trading_cost_bps / 10_000
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.returns = garch_return_path(self.n_steps, self.rng)
        self.step_idx, self.position, self.nav = 1, 0, 1.0
        return build_observation(self.returns, self.step_idx, self.position), {}

    def step(self, action: int):
        reward_position = self.position
        new_position = action - 1
        trade = abs(new_position - reward_position)
        cost = trade * self.trading_cost
        market_return = self.returns[self.step_idx]
        strategy_return = reward_position * market_return - cost
        self.position = new_position
        self.nav *= 1 + strategy_return
        self.step_idx += 1
        reward, terminated = strategy_return * 100, self.step_idx >= self.n_steps - 1
        obs = build_observation(self.returns, self.step_idx, self.position)
        info = dict(
            nav=self.nav,
            reward_position=reward_position,
            next_position=new_position,
            market_return=market_return,
            strategy_return=strategy_return,
            trade=trade,
        )
        return obs, reward, terminated, False, info


# %% [markdown]
# ## 2. Train Algorithms
#
# We train three algorithms on the same environment:
# - **DQN**: Value-based, learns Q(s,a) function
# - **PPO**: On-policy actor-critic, stable policy updates
# - **A2C**: Actor-critic baseline (SAC requires continuous actions)
#
# Note: We use A2C instead of SAC because our action space is discrete.
# SAC is designed for continuous action spaces.


# %%
def make_env(seed: int = 42):
    """Factory function for environment creation."""

    def _init():
        env = SimpleTradingEnv(
            n_steps=config["timesteps_per_episode"],
            trading_cost_bps=10,
            seed=seed,
        )
        return env

    return _init


# Create vectorized environments for each algorithm
env_dqn = DummyVecEnv([make_env(seed=SEED)])
env_ppo = DummyVecEnv([make_env(seed=SEED)])
env_a2c = DummyVecEnv([make_env(seed=SEED)])

print("Environments created successfully")

# %%
# Train DQN
print("Training DQN...")
model_dqn = DQN(
    "MlpPolicy",
    env_dqn,
    learning_rate=1e-4,
    buffer_size=10_000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    device="cpu",  # Small MLP - CPU is faster than GPU for SB3
    seed=SEED,  # Reproducible training
    verbose=0,
)

_ = model_dqn.learn(total_timesteps=config["total_timesteps"])

# %%
# Train PPO
print("Training PPO...")
model_ppo = PPO(
    "MlpPolicy",
    env_ppo,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    device="cpu",  # Small MLP - CPU is faster than GPU for SB3
    seed=SEED,  # Reproducible training
    verbose=0,
)

_ = model_ppo.learn(total_timesteps=config["total_timesteps"])

# %%
# Train A2C
print("Training A2C...")
model_a2c = A2C(
    "MlpPolicy",
    env_a2c,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    device="cpu",  # Small MLP - CPU is faster than GPU for SB3
    seed=SEED,  # Reproducible training
    verbose=0,
)

_ = model_a2c.learn(total_timesteps=config["total_timesteps"])

# %% [markdown]
# ## 3. Evaluate Policies
#
# Each algorithm is evaluated on the same explicit list of episode seeds so
# DQN, PPO, and A2C see byte-identical environment dynamics; a fresh
# `DummyVecEnv` per seed prevents one algorithm's evaluation from advancing
# the environment RNG for the next.

# %%
# Evaluate all models on identical per-seed episode paths
eval_seeds = list(range(123, 123 + config["eval_episodes"]))
results = {}

for name, model in [("DQN", model_dqn), ("PPO", model_ppo), ("A2C", model_a2c)]:
    rewards = []
    for seed in eval_seeds:
        eval_env = DummyVecEnv([make_env(seed=seed)])
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True)
        rewards.append(mean_reward)
    results[name] = {"mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards))}
    print(
        f"{name}: Mean Reward = {results[name]['mean_reward']:.2f} ± {results[name]['std_reward']:.2f}"
    )

# %% [markdown]
# ### Episode Trajectory Collection
#
# Run a single deterministic episode to collect reward-bearing positions,
# newly selected actions, rewards, and NAV for each algorithm. Keeping both
# position series makes the one-period action-to-reward timing explicit.


# %%
def run_episode(model, env, deterministic: bool = True):
    """Run a single episode and collect trajectory data."""
    obs = env.reset()
    done = False

    reward_positions = []
    action_positions = []
    rewards = []
    navs = [1.0]
    total_trades = 0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)

        reward_positions.append(info[0]["reward_position"])
        action_positions.append(info[0]["next_position"])
        rewards.append(reward[0])
        navs.append(info[0]["nav"])
        total_trades += info[0]["trade"]

    return {
        "reward_positions": np.array(reward_positions),
        "action_positions": np.array(action_positions),
        "rewards": np.array(rewards),
        "navs": np.array(navs),
        "final_nav": navs[-1],
        "total_reward": sum(rewards),
        "total_trades": total_trades,
    }


# Run episodes for each model
trajectories = {}
for name, model in [("DQN", model_dqn), ("PPO", model_ppo), ("A2C", model_a2c)]:
    eval_env = DummyVecEnv([make_env(seed=456)])
    trajectories[name] = run_episode(model, eval_env)

# %%
if EXPORT_RESULTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for algo, traj in trajectories.items():
        navs = traj["navs"][1:]  # align to per-step length
        cum_reward = np.cumsum(traj["rewards"])
        for step, (reward_pos, action, rew, nav, cr) in enumerate(
            zip(
                traj["reward_positions"],
                traj["action_positions"],
                traj["rewards"],
                navs,
                cum_reward,
            )
        ):
            rows.append(
                {
                    "algo": algo,
                    "step": step,
                    "reward_position": int(reward_pos),
                    "next_position": int(action),
                    "reward": float(rew),
                    "nav": float(nav),
                    "cumulative_reward": float(cr),
                }
            )
    pl.DataFrame(rows).write_parquet(OUTPUT_DIR / "algorithm_trajectories.parquet")
    print(f"Trajectories saved to {OUTPUT_DIR / 'algorithm_trajectories.parquet'}")

# %% [markdown]
# ## 4. Visualize Algorithm Behaviors


# %%
# Create comparison visualization
fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=["Signed Log NAV", "Reward-Bearing Position", "Cumulative Reward"],
    vertical_spacing=0.1,
)

styles = {
    "DQN": {"color": COLORS["blue"], "dash": "solid", "symbol": "circle", "pattern": "/"},
    "PPO": {"color": COLORS["amber"], "dash": "dash", "symbol": "square", "pattern": "x"},
    "A2C": {"color": COLORS["neutral"], "dash": "dot", "symbol": "diamond", "pattern": "."},
}

switch_counts = {
    name: int(np.count_nonzero(np.diff(traj["action_positions"])))
    for name, traj in trajectories.items()
}
most_active = max(switch_counts, key=switch_counts.get)

# %%
for name, traj in trajectories.items():
    nav_display = np.sign(traj["navs"]) * np.log10(1 + np.abs(traj["navs"]))
    fig.add_trace(
        go.Scatter(
            y=nav_display,
            name=f"{name} NAV",
            line=dict(color=styles[name]["color"], dash=styles[name]["dash"], width=2),
            marker=dict(color=styles[name]["color"], symbol=styles[name]["symbol"], size=6),
            legendgroup=name,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=traj["reward_positions"],
            name=f"{name} Reward Position",
            line=dict(color=styles[name]["color"], dash=styles[name]["dash"], width=2),
            legendgroup=name,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=np.cumsum(traj["rewards"]),
            name=f"{name} Cum. Reward",
            line=dict(color=styles[name]["color"], dash=styles[name]["dash"], width=2),
            legendgroup=name,
            showlegend=False,
        ),
        row=3,
        col=1,
    )

# %%
# Configure layout and display
fig.update_layout(
    title="Only one of the three algorithms keeps switching position",
    height=800,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fig.update_yaxes(title_text="Signed log10(1 + |NAV|)", row=1, col=1)
fig.update_yaxes(title_text="Position Earning Return", row=2, col=1)
fig.update_yaxes(title_text="Cumulative Reward", row=3, col=1)
fig.update_xaxes(title_text="Trading Day", row=3, col=1)

fig.show()

# %% [markdown]
# ## 5. Action Distribution Analysis


# %%
# Analyze position distributions
fig = go.Figure()

for name, traj in trajectories.items():
    action_positions = traj["action_positions"]
    pos_counts = {v: int(np.sum(action_positions == v)) for v in [-1, 0, 1]}  # numpy counting

    fig.add_trace(
        go.Bar(
            x=["Short (-1)", "Hold (0)", "Long (+1)"],
            y=[
                pos_counts.get(-1, 0),
                pos_counts.get(0, 0),
                pos_counts.get(1, 0),
            ],
            name=name,
            marker=dict(
                color=styles[name]["color"],
                pattern=dict(shape=styles[name]["pattern"]),
                line=dict(color=COLORS["neutral"], width=0.5),
            ),
        )
    )

fig.update_layout(
    title="Two of the three algorithms collapse onto a single position",
    xaxis_title="Chosen Position",
    yaxis_title="Count (days)",
    barmode="group",
    height=400,
)

fig.show()

# %% [markdown]
# ## 6. Summary Statistics


# %%
# Summary table
summary_data = []
for name, traj in trajectories.items():
    action_positions = traj["action_positions"]
    summary_data.append(
        {
            "Algorithm": name,
            "Final NAV": f"{traj['final_nav']:.4f}",
            "Total Reward": f"{traj['total_reward']:.2f}",
            "% Long": f"{(action_positions == 1).mean() * 100:.1f}%",
            "% Hold": f"{(action_positions == 0).mean() * 100:.1f}%",
            "% Short": f"{(action_positions == -1).mean() * 100:.1f}%",
            "Turnover": f"{traj['total_trades']:.0f}",
        }
    )

summary_df = pl.DataFrame(summary_data)
summary_df

# %%
action_counts = {
    name: int(np.unique(traj["action_positions"]).size) for name, traj in trajectories.items()
}
best_reward = max(results, key=lambda name: results[name]["mean_reward"])
policy_lines = "\n".join(
    f"- **{name}** uses {action_counts[name]} action(s) and switches position "
    f"{switch_counts[name]} time(s) on the common diagnostic path."
    for name in ["DQN", "PPO", "A2C"]
)
display(
    Markdown(
        f"""
## Key Takeaways

With the same {config["total_timesteps"]:,}-timestep budget and held-out evaluation seeds, the
algorithms learn materially different policies:

{policy_lines}

**{best_reward}** has the highest mean evaluation reward in this run
({results[best_reward]["mean_reward"]:.2f}), while **{most_active}** has the most position
switches. That disagreement is the teaching result: a scalar reward does not reveal whether a
policy is active, diversified across actions, or collapsed to one position. The action-distribution
diagnostic makes that behavior visible.

**Next**: See `optimal_execution_ppo` for a continuous-action PPO application to optimal execution.
"""
    )
)
