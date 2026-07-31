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
# **Execution environment**: local `uv run` on CPU. The small PPO MLP has no
# meaningful GPU-heavy path.
#
# This notebook implements a PPO-based agent for optimal trade execution,
# demonstrating how RL learns to minimize implementation shortfall by
# adapting execution to market conditions.
#
# **Key Feature**: The simulator anchors volatility, spread, depth, and impact
# parameters to real crypto data while remaining a stylized execution model.
#
# **Learning Outcomes**:
# - LO2: Implement a PPO-based optimal execution agent
# - LO6: Analyze sensitivity to reward function and simulation realism
#
# **Book Reference**: Chapter 21, Section 21.4.
#
# **Prerequisites**: `calibration` and Chapter 18 on implementation shortfall.
#
# See §21.4 for the discussion of benchmark interpretation caveats.

# %%
"""Optimal Execution with PPO - learn to minimize implementation shortfall by adapting execution to market conditions."""

# Core imports
import warnings

import numpy as np
import polars as pl
from IPython.display import Markdown, display

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Calibration and shared environment
from rl_calibration import CryptoMarketCalibrator
from rl_environments import ExecutionEnv

# Stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ML4T configuration
from utils.paths import get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

# %%
OUTPUT_DIR = get_output_dir(21, "optimal_execution_ppo")

# %% tags=["parameters"]
TOTAL_SHARES = 10_000
EXECUTION_HORIZON = 60
TOTAL_TIMESTEPS = 100_000
EVAL_EPISODES = 20
RISK_AVERSION = 1e-4
SCHEDULE_PENALTY = 5e-5
EXPORT_RESULTS = False
SEED = 314  # Reproducible training across re-runs

# %%
set_global_seeds(SEED)

# %%
# Configuration
config = {
    "total_shares": TOTAL_SHARES,
    "execution_horizon": EXECUTION_HORIZON,
    "total_timesteps": TOTAL_TIMESTEPS,
    "eval_episodes": EVAL_EPISODES,
    "risk_aversion": RISK_AVERSION,
    "schedule_penalty": SCHEDULE_PENALTY,
    "export_results": EXPORT_RESULTS,
}

print(f"Configuration: {config}")


# %% [markdown]
# ## 0. Calibrate Simulation from Real Data
#
# We fit simulation parameters from actual crypto market data. These estimates
# anchor the stylized execution simulator to observed market dynamics:
# - GARCH(1,1) volatility clustering
# - Regime transition probabilities
# - Spread and depth distributions
# - Heuristic market-impact coefficients

# %%
# Calibrate from real BTC hourly data
print("Calibrating simulation from real crypto data...")
calibrator = CryptoMarketCalibrator("BTCUSDT")
cal_params = calibrator.get_execution_env_params()

calibration_summary = pl.DataFrame(
    {
        "parameter": [
            "garch_alpha",
            "garch_beta",
            "garch_uncond_vol",
            "regime_p_stay_normal",
            "regime_p_stay_stressed",
            "spread_normal",
            "spread_stressed",
            "permanent_impact",
            "temporary_impact",
            "risk_aversion",
        ],
        "value": [
            cal_params.garch.alpha,
            cal_params.garch.beta,
            cal_params.garch.unconditional_vol,
            cal_params.regimes.transition_matrix[0, 0],
            cal_params.regimes.transition_matrix[1, 1],
            cal_params.spread_normal,
            cal_params.spread_stressed,
            cal_params.permanent_impact,
            cal_params.temporary_impact,
            config["risk_aversion"],
        ],
    }
)
calibration_summary

# %% [markdown]
# ## 1. Execution Environment
#
# We use `ExecutionEnv` from the shared environments module. It models:
# - Linear temporary and permanent impact
# - Time-varying liquidity with regime switching
# - A step objective that combines implementation shortfall and inventory risk
#
# The result is an Almgren-Chriss-style liquidation problem in a stylized,
# path-dependent simulator. The calibration is heuristic rather than exact.
#
# The horizon step sells whatever inventory is left and ignores the
# participation cap while doing so, so every policy finishes the order on time
# and pays for whatever it postponed. The cost of delay therefore shows up in
# the shortfall and in how concentrated the schedule is at the end, which is
# what the diagnostics in Section 6 measure.

# %% [markdown]
# ## 2. Benchmark Strategies
#
# We implement TWAP and an Almgren-Chriss-style benchmark for comparison.
#
# **Oracle note on the A-C benchmark**: The Almgren-Chriss schedule uses full
# knowledge of the future market path (average volatility and depth across
# all time steps) to calibrate its risk-cost trade-off parameter $\kappa$.
# Neither TWAP nor PPO have access to this information. This makes the A-C
# benchmark an *oracle comparator* with an information advantage,
# rather than a fair online competitor. The comparison remains useful for
# showing how that information changes the liquidation schedule.


# %%
def twap_execution(env: ExecutionEnv, reset_seed: int | None = None) -> dict:
    """
    Time-Weighted Average Price: sell equal amounts each period.
    """
    obs, _ = env.reset(seed=reset_seed)
    done = False

    shares_per_step = env.total_shares / env.horizon

    while not done:
        if env.step_idx == env.horizon - 1:
            target_shares = env.remaining_shares
        else:
            target_shares = min(int(np.ceil(shares_per_step)), env.remaining_shares)
        action = env.target_shares_to_action(target_shares)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return summarize_execution(env)


# %% [markdown]
# ### Execution Summary
#
# Extracts shortfall, in dollars and in basis points of arrival notional, from a
# completed environment for comparison across strategies.


# %%
def summarize_execution(env: ExecutionEnv) -> dict:
    return {
        "total_shortfall": env.total_cost,
        "shortfall_bps": env.total_cost / (env.arrival_price * env.total_shares) * 10_000,
        "history": env.execution_history,
    }


# %% [markdown]
# ### Almgren-Chriss Schedule
#
# The discrete Almgren-Chriss model computes an optimal static liquidation
# schedule that trades off execution urgency (risk aversion) against market
# impact. It uses full-path statistics (mean volatility and depth), making
# it an **oracle comparator** - neither TWAP nor PPO have this information.


# %%
def discrete_almgren_chriss_schedule(
    total_shares: int,
    horizon: int,
    sigma_price: float,
    eta: float,
    gamma: float,
    risk_aversion: float,
) -> np.ndarray:
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

    float_trades = np.maximum(remaining_path[:-1] - remaining_path[1:], 0.0)
    trades = np.floor(float_trades).astype(int)
    remainder = total_shares - int(trades.sum())
    if remainder > 0:
        order = np.argsort(-(float_trades - trades))
        trades[order[:remainder]] += 1

    return trades


# %% [markdown]
# ### Run Almgren-Chriss Execution
#
# Executes the static A-C schedule in the environment, forcing any
# remaining inventory into the final step.


# %%
def almgren_chriss_execution(env: ExecutionEnv, reset_seed: int | None = None) -> dict:
    """
    Discrete Almgren-Chriss-style cost-risk liquidation schedule.
    """
    obs, _ = env.reset(seed=reset_seed)
    done = False

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

    while not done:
        if env.step_idx == env.horizon - 1:
            planned_shares = env.remaining_shares
        else:
            planned_shares = min(
                int(schedule[min(env.step_idx, len(schedule) - 1)]), env.remaining_shares
            )
        action = env.target_shares_to_action(planned_shares)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return summarize_execution(env)


# %% [markdown]
# ## 3. Train PPO Agent


# %%
def make_env(seed: int = 42):
    def _init():
        return ExecutionEnv(
            total_shares=config["total_shares"],
            horizon=config["execution_horizon"],
            cal_params=cal_params,  # Use calibrated parameters
            risk_aversion=config["risk_aversion"],
            schedule_penalty=config["schedule_penalty"],
            seed=seed,
        )

    return _init


# Create environment with calibrated parameters
env = DummyVecEnv([make_env(seed=SEED)])
print("Environment created with calibrated market dynamics")

# %%
print("Training PPO execution agent...")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01,
    device="cpu",  # Small MLP - CPU is faster than GPU for SB3
    seed=SEED,  # Reproducible training
    verbose=0,
)

_ = model.learn(total_timesteps=config["total_timesteps"])

# %% [markdown]
# ## 4. Evaluate Strategies


# %% [markdown]
# ### Per-episode diagnostics


# %%
def _collect_execution_diagnostics(env: "ExecutionEnv", result: dict) -> dict:
    """Extract how much of the order each policy leaves for the end of the horizon."""
    history = result["history"]
    last_step_share = float(history[-1]["shares_sold"]) / env.total_shares
    final_quarter_cutoff = max(env.horizon - env.horizon // 4, 0)
    final_quarter_volume = sum(
        float(h["shares_sold"]) for h in history if int(h["step"]) >= final_quarter_cutoff
    )
    return {
        "shortfall_bps": result["shortfall_bps"],
        "last_step_share": last_step_share,
        "final_quarter_share": final_quarter_volume / env.total_shares,
        "history": history,
    }


# %% [markdown]
# ### Aggregate across evaluation episodes


# %%
def evaluate_strategy(strategy_name: str, strategy_fn, n_episodes: int = 20, seed_base: int = 1000):
    """Evaluate a strategy over multiple episodes with calibrated simulation."""
    diagnostics = []
    all_paths = []
    for i in range(n_episodes):
        episode_seed = seed_base + i
        env = ExecutionEnv(
            total_shares=config["total_shares"],
            horizon=config["execution_horizon"],
            cal_params=cal_params,
            risk_aversion=config["risk_aversion"],
            schedule_penalty=config["schedule_penalty"],
            seed=episode_seed,
        )
        diag = _collect_execution_diagnostics(env, strategy_fn(env, reset_seed=episode_seed))
        diagnostics.append(diag)
        for h in diag["history"]:
            all_paths.append(
                {"strategy": strategy_name, "episode_id": i, "episode_seed": episode_seed, **h}
            )

    shortfalls = np.array([d["shortfall_bps"] for d in diagnostics])
    summary = {
        "mean_bps": shortfalls.mean(),
        "std_bps": shortfalls.std(),
        "min_bps": shortfalls.min(),
        "max_bps": shortfalls.max(),
        "avg_last_step_share_pct": 100
        * float(np.mean([d["last_step_share"] for d in diagnostics])),
        "avg_final_quarter_share_pct": 100
        * float(np.mean([d["final_quarter_share"] for d in diagnostics])),
    }
    return summary, all_paths


# %% [markdown]
# ### PPO Execution Policy
#
# Wraps the trained PPO model in the same episode interface used by
# the TWAP and Almgren-Chriss baselines for fair comparison.


# %%
def ppo_execution(env: ExecutionEnv, reset_seed: int | None = None) -> dict:
    """Execute using trained PPO agent."""
    obs, _ = env.reset(seed=reset_seed)
    done = False

    while not done:
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])
        done = terminated or truncated

    return summarize_execution(env)


# %%
print("\nEvaluating strategies...")

results = {}
evaluation_paths = []
for name, fn in [
    ("TWAP", twap_execution),
    ("Almgren-Chriss", almgren_chriss_execution),
    ("PPO", ppo_execution),
]:
    results[name], strategy_paths = evaluate_strategy(
        name,
        fn,
        n_episodes=config["eval_episodes"],
    )
    evaluation_paths.extend(strategy_paths)
    print(
        f"{name:16s}: {results[name]['mean_bps']:6.2f} ± {results[name]['std_bps']:5.2f} bps"
        f" | last step share: {results[name]['avg_last_step_share_pct']:5.1f}%"
    )

# %% [markdown]
# ## 5. Visualize Execution Trajectories

# %%
evaluation_df = pl.DataFrame(evaluation_paths).with_columns(
    pl.col("shortfall").cum_sum().over(["strategy", "episode_id"]).alias("cum_shortfall")
)
trajectory_stats = (
    evaluation_df.group_by(["strategy", "step"])
    .agg(
        [
            pl.col("shares_sold").mean().alias("mean_shares_sold"),
            pl.col("remaining").mean().alias("mean_remaining"),
            pl.col("cum_shortfall").mean().alias("mean_cum_shortfall"),
            pl.col("shares_sold").quantile(0.1).alias("p10_shares_sold"),
            pl.col("shares_sold").quantile(0.9).alias("p90_shares_sold"),
            pl.col("remaining").quantile(0.1).alias("p10_remaining"),
            pl.col("remaining").quantile(0.9).alias("p90_remaining"),
            pl.col("cum_shortfall").quantile(0.1).alias("p10_cum_shortfall"),
            pl.col("cum_shortfall").quantile(0.9).alias("p90_cum_shortfall"),
        ]
    )
    .sort(["strategy", "step"])
)

# %%
# Create visualization
fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=[
        "Average execution rate (shares/step)",
        "Average remaining inventory",
        "Average cumulative implementation shortfall",
    ],
    vertical_spacing=0.12,
)

colors = {
    "TWAP": COLORS["blue"],
    "Almgren-Chriss": COLORS["amber"],
    "PPO": COLORS["positive"],
}

# %% [markdown]
# The execution-rate band shows the central 80% of episode paths without
# duplicating its trace construction inside the strategy loop.


# %%
def add_execution_rate_band(fig, strategy_df: pl.DataFrame, name: str, color: str) -> None:
    """Add the 10th-to-90th-percentile execution-rate ribbon."""
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
for name in colors:
    strategy_df = trajectory_stats.filter(pl.col("strategy") == name).sort("step")
    if strategy_df.is_empty():
        continue
    steps = strategy_df["step"].to_list()
    shares = strategy_df["mean_shares_sold"].to_list()
    remaining = strategy_df["mean_remaining"].to_list()

    add_execution_rate_band(fig, strategy_df, name, colors[name])

    # Execution rate
    fig.add_trace(
        go.Scatter(x=steps, y=shares, name=name, line=dict(color=colors[name]), legendgroup=name),
        row=1,
        col=1,
    )

    # Remaining inventory
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=remaining,
            name=name,
            line=dict(color=colors[name]),
            legendgroup=name,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Cumulative shortfall
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=strategy_df["mean_cum_shortfall"].to_list(),
            name=name,
            line=dict(color=colors[name]),
            legendgroup=name,
            showlegend=False,
        ),
        row=3,
        col=1,
    )

# %%
# Configure layout and display
fig.update_layout(
    title=(
        "PPO clears inventory earlier than the Almgren-Chriss schedule"
        "<br><sup>Lines are episode means; shaded bands show the 10th to 90th percentile execution rate</sup>"
    ),
    height=700,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fig.update_yaxes(title_text="Shares per step", row=1, col=1)
fig.update_yaxes(title_text="Remaining", row=2, col=1)
fig.update_yaxes(title_text="Shortfall ($)", row=3, col=1)
fig.update_xaxes(title_text="Time Step", row=3, col=1)
fig.show()

# %% [markdown]
# ## 6. Failure-Mode Diagnostic
#
# The failure mode this environment can express is postponement: a policy that
# leaves inventory for the last few steps must trade it there whatever the depth
# is. Two shares measure it - the volume traded in the final quarter of the
# horizon, and the volume traded on the horizon step alone.

# %%
print("\n" + "=" * 60)
print("EXECUTION SCHEDULE CONCENTRATION")
print("=" * 60)
for name in ["TWAP", "Almgren-Chriss", "PPO"]:
    print(
        f"{name:16s}: final-quarter volume {results[name]['avg_final_quarter_share_pct']:5.1f}%"
        f" | last-step volume {results[name]['avg_last_step_share_pct']:5.1f}%"
    )
print("\nA schedule concentrated at the end pays whatever the book charges at that")
print("moment, so a low mean shortfall built that way is a bet on terminal liquidity.")

# %% [markdown]
# ## 7. Summary

# %%
summary_data = []
for name, r in results.items():
    summary_data.append(
        {
            "Strategy": name,
            "Mean IS (bps)": f"{r['mean_bps']:.2f}",
            "Std (bps)": f"{r['std_bps']:.2f}",
            "Best (bps)": f"{r['min_bps']:.2f}",
            "Worst (bps)": f"{r['max_bps']:.2f}",
            "Final Quarter %": f"{r['avg_final_quarter_share_pct']:.1f}",
            "Last Step %": f"{r['avg_last_step_share_pct']:.1f}",
        }
    )

summary_df = pl.DataFrame(summary_data)
summary_df

# %% [markdown]
# Illustrative mean implementation shortfall in this run. The figure
# summarizes average evaluation behavior rather than a single path.
# Treat this notebook as a compact execution benchmark, not a locked ranking.

# %%
if config["export_results"]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    evaluation_df.write_parquet(OUTPUT_DIR / "execution_results.parquet")
    print(f"\nResults saved to {OUTPUT_DIR / 'execution_results.parquet'}")

    summary_export = pl.DataFrame(
        [
            {
                "strategy": name,
                "mean_bps": r["mean_bps"],
                "std_bps": r["std_bps"],
                "min_bps": r["min_bps"],
                "max_bps": r["max_bps"],
                "avg_last_step_share_pct": r["avg_last_step_share_pct"],
                "avg_final_quarter_share_pct": r["avg_final_quarter_share_pct"],
            }
            for name, r in results.items()
        ]
    )
    summary_export.write_parquet(OUTPUT_DIR / "execution_summary.parquet")
    print(f"Summary saved to {OUTPUT_DIR / 'execution_summary.parquet'}")

# %%
best_mean = min(results, key=lambda name: results[name]["mean_bps"])
profile_text = "\n".join(
    f"- **{name}**: mean shortfall {results[name]['mean_bps']:.1f} bps, standard deviation "
    f"{results[name]['std_bps']:.1f} bps, final-quarter share "
    f"{results[name]['avg_final_quarter_share_pct']:.1f}%, and last-step share "
    f"{results[name]['avg_last_step_share_pct']:.1f}%."
    for name in ["TWAP", "Almgren-Chriss", "PPO"]
)
display(
    Markdown(
        f"""
## Key Takeaways

{profile_text}

**{best_mean}** has the lowest mean implementation shortfall in this run, but the between-strategy
mean differences are small relative to episode dispersion, and the three schedules leave very
different amounts of the order for the end of the horizon. The profile diagnostics therefore matter
more than a locked ranking.

Almgren-Chriss is an oracle comparator here because it uses full-path volatility and depth. PPO and
TWAP do not receive that future information. This notebook demonstrates a state-responsive learned
schedule in a calibrated simulator, not a general claim that PPO beats the analytical benchmark.

See Section 21.4 for the optimal-execution discussion.
"""
    )
)
