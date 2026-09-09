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
# # Market Making with PPO
#
# **Chapter 21: Reinforcement Learning for Execution and Hedging**
#
# ## Purpose
#
# A market maker quotes a price to buy and a price to sell at the same time,
# and earns the difference when both sides trade. The difficulty is that the
# two sides do not fill evenly: a run of buyers leaves the maker short, a run of
# sellers leaves it long, and the resulting position is exposed to the next
# price move. Every quoting decision therefore trades expected spread income
# against the inventory it is likely to accumulate.
#
# This notebook trains a PPO agent to make that trade-off by choosing, at each
# step, how far to shift its quotes away from the mid price and how wide to make
# them. It is compared against three fixed rules that already encode the
# textbook inventory response. The environment applies that response to every
# quote it prices, the agent's and the rules' alike, so what the comparison
# isolates is narrow and answerable: what does a learned skew, and a spread
# width that can change during the episode, add to an inventory response that is
# already programmed in?
#
# ## Learning objectives
#
# After working through this notebook you will be able to:
#
# - Write a market-making environment in which fills arrive with a probability
#   that falls as a quote sits further from the mid price, and explain how that
#   single assumption creates the spread-versus-fill-rate trade-off.
# - Compute a reservation price from an inventory position and a volatility
#   estimate, and say which direction it moves a maker's quotes and why.
# - Train a discrete-action PPO agent over a grid of quote skews and spread
#   widths, with observation and reward normalisation, and explain what the
#   normalisation is protecting the optimiser from.
# - Compare a learned policy against fixed rules on paired episodes, size the
#   difference in standard errors, and say why an unpaired comparison of the
#   same two policies would have the wrong standard error.
# - Read a policy's inventory response off a chart of quote offset against the
#   inventory held when the quote was posted.
#
# ## Book reference
#
# Section 21.5, *Application II - Market making*.
#
# ## Prerequisites
#
# - Chapter 18 on transaction costs, for the spread as a cost of immediacy.
# - `market_making_env`, which holds the environment, the quoting geometry and
#   the fill model.
# - `rl_calibration.CryptoMarketCalibrator`, which fits the GARCH(1,1)
#   volatility process to hourly `BTCUSDT` bars.

# %% [markdown]
# ## Setup

# %%
"""Market making with PPO - a learned quoting policy against reservation-price rules."""

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

from market_making_env import MarketMakingDynamics, MarketMakingEnv
from rl_calibration import CryptoMarketCalibrator
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import utils  # noqa: F401  - sets the Plotly renderer so figures carry a static PNG
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
CALIBRATION_SYMBOL = "BTCUSDT"  # symbol the volatility process is fitted to
EPISODE_LENGTH = 500  # quoting steps in one episode
TOTAL_TIMESTEPS = 300_000  # environment steps PPO may consume while learning
EVAL_EPISODES = 240  # episodes every strategy is scored on
INVENTORY_LIMIT = 100  # position at which the maker stops quoting the side that grows it
LAMBDA_INVENTORY = 0.001  # weight on the squared normalised inventory penalty
LEARNING_RATE = 3e-4
EVAL_SEED_BASE = 1000  # first evaluation seed; the rest follow consecutively
VIZ_SEED = 999  # the single episode drawn in the behaviour figures
SEED = 314

# %%
set_global_seeds(SEED)

# %% [markdown]
# ### The quoting grid
#
# The agent's action is one of nine combinations: three **skew** levels and
# three **spread multipliers**. Skew shifts the centre of the two quotes away
# from the reservation price - negative shifts both quotes down, which makes the
# bid less likely to fill and the ask more likely, so it sheds a long position.
# The spread multiplier widens or narrows both quotes around that centre, which
# trades fill rate against income per fill.

# %%
SKEW_LEVELS = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
SPREAD_MULTIPLIERS = np.array([0.8, 1.0, 1.25], dtype=np.float32)
RL_LABEL = "PPO"

display(
    Markdown(f"""
- **Episode**: {EPISODE_LENGTH} quoting steps. Whatever inventory is left at the end is
  liquidated at a half-spread cost, so a strategy cannot hide a position by holding it.
- **Inventory limit**: {INVENTORY_LIMIT} units. Beyond it the side of the book that would grow
  the position stops quoting, which caps the risk but also caps the income.
- **Inventory penalty**: {LAMBDA_INVENTORY} per unit of squared normalised inventory, charged
  every step. At the limit that is {LAMBDA_INVENTORY} of the price per step; at half the limit
  it is a quarter of that.
- **Skew levels**: {", ".join(f"{value:g}" for value in SKEW_LEVELS)}, in units of a quarter of
  one standard deviation of the price.
- **Spread multipliers**: {", ".join(f"{value:g}" for value in SPREAD_MULTIPLIERS)}, applied to
  a half-spread that already widens with volatility.
- **Learning budget**: {TOTAL_TIMESTEPS:,} steps, about
  {TOTAL_TIMESTEPS / EPISODE_LENGTH:,.0f} episodes.
- **Evaluation**: {EVAL_EPISODES} episodes on seeds {EVAL_SEED_BASE} to
  {EVAL_SEED_BASE + EVAL_EPISODES - 1}, run by every strategy.
""")
)

# %% [markdown]
# ## 1. The volatility process
#
# The maker quotes into a simulated price path rather than a recorded one,
# because its own fills have to depend on where it quotes. The path's volatility
# follows a **GARCH(1,1)** process whose coefficients are fitted to hourly
# perpetual-futures returns, so the simulated market has the clustering a real
# one does: a large move raises the variance of the next move, and the variance
# decays slowly.
#
# $$\sigma_t^2 = \omega + \alpha \, \varepsilon_{t-1}^2 + \beta \, \sigma_{t-1}^2$$
#
# Volatility matters here for two separate reasons. It sets how far the price
# can move against an accumulated position, and it enters the quoting geometry
# directly: the half-spread widens with volatility, so the maker charges more
# for immediacy exactly when immediacy is worth more.

# %%
calibrator = CryptoMarketCalibrator(CALIBRATION_SYMBOL)
source_bars = calibrator.load_data()
garch = calibrator.fit_garch()

if not all(np.isfinite([garch.alpha, garch.beta, garch.omega, garch.unconditional_vol])):
    raise ValueError("Calibrated GARCH parameters contain NaN or Inf")

MM_DYNAMICS = MarketMakingDynamics(
    garch_omega=garch.omega,
    garch_alpha=garch.alpha,
    garch_beta=garch.beta,
    unconditional_vol=garch.unconditional_vol,
    skew_levels=tuple(float(value) for value in SKEW_LEVELS),
    spread_multipliers=tuple(float(value) for value in SPREAD_MULTIPLIERS),
)

# %%
display(
    Markdown(f"""
Fitted on **{source_bars.height:,}** hourly `{CALIBRATION_SYMBOL}` bars,
{source_bars["timestamp"].min():%Y-%m-%d} to {source_bars["timestamp"].max():%Y-%m-%d}.

| Coefficient | Value | What it controls |
|---|---|---|
| $\\alpha$ | {garch.alpha:.4f} | reaction of the variance to the last return |
| $\\beta$ | {garch.beta:.4f} | share of the previous variance carried forward |
| $\\omega$ | {garch.omega:.2e} | constant fixing the long-run variance |
| unconditional volatility | {garch.unconditional_vol:.4f} | per-step standard deviation |
""")
)

# %% [markdown]
# ## 2. The market-making environment
#
# ### What the maker sees and does
#
# The observation is six numbers: the inventory as a fraction of its limit, the
# last price change, the current volatility, the **order imbalance**, the
# fraction of the episode remaining, and the width of the maker's own current
# quotes. Order imbalance is a mean-reverting series in $[-1, 1]$ standing in
# for the pressure of arriving orders: it pushes the price slightly and it
# raises the fill probability on the side that the flow is hitting.
#
# ### The reservation price
#
# A maker holding a long position wants to be sold to less and bought from more.
# The standard way to express that is the **reservation price**: the mid price
# shifted against the inventory,
#
# $$p^r_t = p_t \left(1 - \frac{q_t}{q_{\max}} \, \sigma_t \right)$$
#
# where $q_t$ is the inventory and $q_{\max}$ its limit. A long position pulls
# both quotes down, so the ask is closer to the market and fills sooner while
# the bid is further away and fills later. The three fixed baselines in this
# notebook quote symmetrically around exactly this price, at three different
# widths, and differ from the agent only in that they never add skew of their
# own.
#
# ### How a quote fills
#
# A limit order sitting a distance $d$ from the mid fills in a step with
# probability
#
# $$P(\text{fill}) = 1 - \exp\!\left(-\lambda \, e^{-\kappa d / s} \, m\right)$$
#
# where $s$ is the base spread, $\lambda$ the arrival rate, $\kappa$ the
# sensitivity to distance and $m$ an imbalance factor above one on the side the
# flow is hitting. This is the whole of the trade-off: quoting tighter fills
# more often and earns less per fill, and quoting wider does the reverse. The
# multiplier $m$ is where **adverse selection** enters - the maker is more
# likely to be filled on the side the market is moving away from.
#
# ### The reward
#
# $$r_t = \Delta W_t - \lambda_q \left(\frac{q_t}{q_{\max}}\right)^2 p_{t+1}$$
#
# where $W_t$ is **marked wealth**: cash plus inventory valued at the next
# price. The penalty term charges for holding a position at all, which is what
# stops a policy from treating an accumulating inventory as free optionality.
# At the end of the episode the remaining inventory is liquidated at a
# half-spread cost and that cost enters the final step's reward.

# %% [markdown]
# ### One episode of the market the maker quotes into
#
# Before any policy runs, this is the path a single seed produces: the mid
# price, the conditional volatility that widens the quotes, and the order
# imbalance that tilts which side fills.

# %%
sample_env = MarketMakingEnv(
    episode_length=EPISODE_LENGTH,
    inventory_limit=INVENTORY_LIMIT,
    lambda_inventory=LAMBDA_INVENTORY,
    dynamics=MM_DYNAMICS,
    seed=VIZ_SEED,
)
sample_env.reset(seed=VIZ_SEED)

# %%
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Mid price", "Conditional volatility", "Order imbalance"],
    vertical_spacing=0.08,
)
steps = list(range(EPISODE_LENGTH))
fig.add_trace(
    go.Scatter(
        x=steps,
        y=sample_env.prices[:EPISODE_LENGTH],
        line=dict(color=COLORS["blue"], width=2),
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=steps,
        y=sample_env.volatilities * 100,
        line=dict(color=COLORS["copper"], width=2),
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=steps,
        y=sample_env.imbalances,
        line=dict(color=COLORS["amber"], width=2),
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.add_hline(y=0, line=dict(color=COLORS["neutral"], width=1, dash="dot"), row=3, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
fig.update_yaxes(title_text="Imbalance", row=3, col=1)
fig.update_xaxes(title_text="Step", row=3, col=1)
fig.update_layout(
    title="One simulated episode: price, volatility and order imbalance",
    height=700,
)
show_plotly_with_alt(
    fig,
    "Three stacked panels over one 500-step episode: the simulated mid price, the conditional volatility driving it, and the mean-reverting order imbalance oscillating around zero.",
)

# %% [markdown]
# ## 3. Train the PPO agent
#
# ### Why the observations and rewards are normalised
#
# The six observation components have very different scales, and the per-step
# reward is dominated by a handful of large moves in inventory value. Both are
# hard on a policy-gradient optimiser: an unnormalised observation makes the
# first layer's gradients wildly uneven across inputs, and a heavy-tailed reward
# makes the advantage estimates jump. `VecNormalize` maintains a running mean
# and variance for both and clips the standardised values, so the policy sees a
# stationary input distribution. The running statistics are part of the trained
# policy: an evaluation that does not apply them is feeding the network inputs
# from a distribution it never saw.


# %%
def make_env(seed: int):
    """Factory returning a market-making environment with this notebook's settings."""

    def _init():
        return MarketMakingEnv(
            episode_length=EPISODE_LENGTH,
            inventory_limit=INVENTORY_LIMIT,
            lambda_inventory=LAMBDA_INVENTORY,
            dynamics=MM_DYNAMICS,
            seed=seed,
        )

    return _init


train_env = VecNormalize(
    DummyVecEnv([make_env(seed=SEED)]),
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99,
)

# %% [markdown]
# ### The policy
#
# Actor and critic each use a two-layer network of 64 hidden units, which is
# ample for a six-element observation and nine actions. The entropy bonus keeps
# the policy from committing to one of the nine actions before it has tried the
# others; on this environment the tempting early commitment is the widest
# spread, which is nearly riskless and nearly income-free.

# %%
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=LEARNING_RATE,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=SEED,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), ortho_init=True),
    device="cpu",  # a two-layer 64-unit MLP runs faster on CPU than on GPU under SB3
    verbose=0,
)

# %%
print(f"Training PPO for {TOTAL_TIMESTEPS:,} steps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS)
print("Training complete")

# %% [markdown]
# ## 4. Evaluate against the reservation-price rules
#
# The three baselines quote symmetrically around the reservation price at the
# three available widths. They already implement the inventory response, so a
# difference between them and the agent is a difference in what the agent added
# on top: a skew chosen per step from the observation, and a spread width that
# can change during the episode.
#
# Every strategy is run on the same seed in each episode, so the four arms face
# the same price path, the same volatility and the same imbalance series. The
# comparison below is therefore paired, and the seed is passed into the episode
# runner rather than set on the environment beforehand, so that the pairing is a
# property of the call rather than of the order in which resets happen.


# %%
def encode_action(skew_idx: int, spread_idx: int) -> int:
    """Map a (skew, spread) index pair to the environment's discrete action index."""
    return skew_idx * len(SPREAD_MULTIPLIERS) + spread_idx


def run_episode(
    env_instance: MarketMakingEnv,
    seed: int,
    model: PPO | None = None,
    fixed_action: int | None = None,
    vec_normalize: VecNormalize | None = None,
) -> dict:
    """Run one seeded episode, driven either by the trained policy or by a fixed action."""
    obs, _ = env_instance.reset(seed=seed)
    done = False

    while not done:
        if model is not None:
            observation = np.array([obs])
            if vec_normalize is not None:
                observation = vec_normalize.normalize_obs(observation)
            action, _ = model.predict(observation, deterministic=True)
            action = action[0]
        else:
            action = fixed_action

        obs, _, terminated, truncated, info = env_instance.step(action)
        done = terminated or truncated

    return {
        "final_wealth": float(info["wealth"]),
        "terminal_inventory": int(info["terminal_inventory"]),
        "n_trades": env_instance.n_trades,
        "history": env_instance.history,
    }


# %% [markdown]
# The normalisation statistics are frozen before evaluation, so every episode is
# scored against the same input distribution the policy was trained on. Reward
# normalisation is switched off at the same time, because the reported wealth
# has to be in the environment's own units.

# %%
train_env.training = False
train_env.norm_reward = False

BASELINE_ACTIONS = {
    "Reservation + Tight Spread": encode_action(skew_idx=1, spread_idx=0),
    "Reservation + Base Spread": encode_action(skew_idx=1, spread_idx=1),
    "Reservation + Wide Spread": encode_action(skew_idx=1, spread_idx=2),
}
REFERENCE_BASELINE = "Reservation + Base Spread"

results = {RL_LABEL: []} | {name: [] for name in BASELINE_ACTIONS}

for i in range(EVAL_EPISODES):
    episode_seed = EVAL_SEED_BASE + i
    test_env = MarketMakingEnv(
        episode_length=EPISODE_LENGTH,
        inventory_limit=INVENTORY_LIMIT,
        lambda_inventory=LAMBDA_INVENTORY,
        dynamics=MM_DYNAMICS,
        seed=episode_seed,
    )
    results[RL_LABEL].append(
        run_episode(test_env, seed=episode_seed, model=model, vec_normalize=train_env)
    )
    for name, action in BASELINE_ACTIONS.items():
        results[name].append(run_episode(test_env, seed=episode_seed, fixed_action=action))

print(f"Evaluated {len(results)} strategies over {EVAL_EPISODES} paired episodes")

# %% [markdown]
# ## 5. How each strategy reached its result
#
# Terminal inventory and trade count describe the route rather than the
# destination, so they stay as a table; the wealth comparison itself is a figure
# below.

# %%
summary_records = [
    {
        "strategy": name,
        "mean_final_wealth": float(np.mean([r["final_wealth"] for r in runs])),
        "std_final_wealth": float(np.std([r["final_wealth"] for r in runs])),
        "avg_abs_terminal_inventory": float(np.mean([abs(r["terminal_inventory"]) for r in runs])),
        "avg_trades": float(np.mean([r["n_trades"] for r in runs])),
    }
    for name, runs in results.items()
]

pl.DataFrame(
    [
        {
            "Strategy": row["strategy"],
            "Avg |terminal inventory|": round(row["avg_abs_terminal_inventory"], 1),
            "Avg trades per episode": round(row["avg_trades"]),
        }
        for row in summary_records
    ]
)

# %% [markdown]
# ### Sizing the comparison
#
# Mean wealth per episode is noisy, so the standard error goes next to it and
# the gap between strategies is measured in standard errors. Without that scale
# the comparison reads as a ranking when it may be a coin flip.
#
# Every strategy trades the same price path within an episode, so the samples
# are paired and the gap is taken per episode before averaging. Treating the two
# means as independent samples would use the wrong standard error, because the
# shared path is common to both arms and cancels in the difference.
#
# The reference baseline is fixed in advance. Picking whichever baseline happens
# to look best in this run and then testing against it would make the interval a
# selection artefact, so the paired gap to all three is reported and the reader
# sees the spread rather than one chosen number.


# %%
def paired_gap(baseline_name: str) -> tuple[float, float, float]:
    """Mean per-episode wealth gap to one baseline, its standard error, and the arm correlation."""
    learned = np.array([run["final_wealth"] for run in results[RL_LABEL]])
    baseline = np.array([run["final_wealth"] for run in results[baseline_name]])
    differences = learned - baseline
    standard_error = float(differences.std(ddof=1) / np.sqrt(differences.size))
    return (
        float(differences.mean()),
        standard_error,
        float(np.corrcoef(learned, baseline)[0, 1]),
    )


RESOLUTION_SIGMA = 3.0  # standard errors a gap must clear before it is called an ordering

ppo_summary = next(row for row in summary_records if row["strategy"] == RL_LABEL)
baseline_summary = [row for row in summary_records if row["strategy"] != RL_LABEL]
all_gaps = {name: paired_gap(name) for name in BASELINE_ACTIONS}
gap, gap_se, gap_corr = all_gaps[REFERENCE_BASELINE]
gap_sigma = abs(gap) / gap_se
direction = "above" if gap > 0 else "below"

# %% [markdown]
# ### The comparison as a figure
#
# The left panel gives the whole distribution of episode wealth per strategy, so
# the spread is visible next to the mean rather than hidden behind it. The right
# panel gives the paired gap to each baseline with one standard error either
# side, which sets the scale of the evaluation noise. Read the markers against
# that scale rather than against the zero line: the threshold fixed above
# resolves an ordering only at three standard errors, so a gap of one or two is
# a gap this episode count cannot call.


# %%
def plot_wealth_comparison(results: dict, all_gaps: dict) -> go.Figure:
    """Episode-wealth distributions beside the paired gap to each baseline."""

    def short(name: str) -> str:
        return name.replace("Reservation + ", "").replace(" Spread", "")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Episode wealth by strategy", "Paired gap to each baseline"),
        horizontal_spacing=0.12,
    )
    for name, runs in results.items():
        focal = name == RL_LABEL
        fig.add_trace(
            go.Box(
                y=[run["final_wealth"] for run in runs],
                name=short(name),
                boxmean=True,
                fillcolor=COLORS["blue"] if focal else COLORS["silver_muted"],
                line=dict(color=COLORS["blue"] if focal else COLORS["neutral"], width=1),
                marker=dict(color=COLORS["neutral"], size=3),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    names = list(all_gaps)
    fig.add_trace(
        go.Scatter(
            x=[short(name) for name in names],
            y=[all_gaps[name][0] for name in names],
            error_y=dict(
                type="data",
                array=[all_gaps[name][1] for name in names],
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
    # Keep the zero line inside the panel whichever way the gaps point.
    low = min([0.0] + [all_gaps[name][0] - all_gaps[name][1] for name in names])
    high = max([0.0] + [all_gaps[name][0] + all_gaps[name][1] for name in names])
    pad = 0.15 * (high - low)
    fig.update_yaxes(title_text="Liquidated wealth (USD)", row=1, col=1)
    fig.update_yaxes(
        title_text="Learned minus baseline (USD)", range=[low - pad, high + pad], row=1, col=2
    )
    fig.update_xaxes(title_text="Strategy", row=1, col=1)
    fig.update_xaxes(title_text="Baseline rule", row=1, col=2)
    fig.update_layout(
        title=(
            "Episode wealth and the paired gap to each fixed rule"
            "<br><sup>Box: median and quartiles over evaluation episodes, dashed line at the "
            "mean. Right: paired gap with one standard error</sup>"
        ),
        height=440,
    )
    return fig


# %%
show_plotly_with_alt(
    plot_wealth_comparison(results, all_gaps),
    "Two panels. Left: a box per strategy over the liquidated wealth of each evaluation episode, the learned policy filled darker than the three fixed rules. Right: the mean paired wealth gap to each fixed rule as a marker with one standard error either side, against a dashed line at zero.",
)

# %%
gap_range = "; ".join(
    f"{name} {value:+.1f} USD ({value / standard_error:+.1f} SE)"
    for name, (value, standard_error, _) in all_gaps.items()
)
resolution = (
    "outside evaluation noise at the threshold set above"
    if gap_sigma >= RESOLUTION_SIGMA
    else "inside evaluation noise, so the ordering is not resolved at this episode count"
)
display(
    Markdown(f"""
Over {EVAL_EPISODES} evaluation episodes the learned policy earns mean liquidated wealth of
{ppo_summary["mean_final_wealth"]:.1f} USD, against
{min(row["mean_final_wealth"] for row in baseline_summary):.1f} to
{max(row["mean_final_wealth"] for row in baseline_summary):.1f} USD for the reservation-price
rules. Its wealth standard deviation is {ppo_summary["std_final_wealth"]:.1f} USD, against
{min(row["std_final_wealth"] for row in baseline_summary):.1f} to
{max(row["std_final_wealth"] for row in baseline_summary):.1f} USD for the rules.

Against the prespecified reference rule ("{REFERENCE_BASELINE}") the learned policy sits
{abs(gap):.1f} USD {direction} it, a paired gap of {gap_sigma:.1f} standard errors -
{resolution}. Episode wealth correlates {gap_corr:.2f} across the two arms, which is what makes
the pairing worth doing. The paired gap to each rule is: {gap_range}.

The sign of that gap is a property of this trained policy at seed {SEED}, not of PPO in general.
""")
)

# %% [markdown]
# ## 6. What the learned policy does
#
# One episode, run under the trained policy, with the inventory it accumulates,
# the offset it quotes at, and the wealth that results on one time axis.

# %%
viz_env = MarketMakingEnv(
    episode_length=EPISODE_LENGTH,
    inventory_limit=INVENTORY_LIMIT,
    lambda_inventory=LAMBDA_INVENTORY,
    dynamics=MM_DYNAMICS,
    seed=VIZ_SEED,
)
viz_history = run_episode(viz_env, seed=VIZ_SEED, model=model, vec_normalize=train_env)["history"]

# %%
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Inventory", "Quote centre offset from mid", "Marked wealth"],
    vertical_spacing=0.08,
)
viz_steps = [h["step"] for h in viz_history]
fig.add_trace(
    go.Scatter(
        x=viz_steps,
        y=[h["inventory"] for h in viz_history],
        line=dict(color=COLORS["blue"], width=2),
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=viz_steps,
        y=[h["quote_offset_bps"] for h in viz_history],
        line=dict(color=COLORS["amber"], width=2),
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=viz_steps,
        y=[h["wealth"] for h in viz_history],
        line=dict(color=COLORS["copper"], width=2),
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.add_hline(y=0, line=dict(color=COLORS["neutral"], width=1, dash="dot"), row=1, col=1)
fig.add_hline(y=0, line=dict(color=COLORS["neutral"], width=1, dash="dot"), row=2, col=1)
fig.update_yaxes(title_text="Units held", row=1, col=1)
fig.update_yaxes(title_text="Offset (bps)", row=2, col=1)
fig.update_yaxes(title_text="Wealth (USD)", row=3, col=1)
fig.update_xaxes(title_text="Step", row=3, col=1)
fig.update_layout(
    title="Inventory, quote offset and wealth over one episode",
    height=720,
)
show_plotly_with_alt(
    fig,
    "Three stacked panels over one episode under the learned policy: units of inventory held, the quote centre's offset from the mid in basis points, and marked wealth.",
)

# %% [markdown]
# ## 7. Which part of the inventory response is the policy's
#
# `compute_quotes` centres every quote on the reservation price, for the agent
# and for the three fixed rules alike, so a quote centre that moves against
# inventory is the environment doing what the reservation price says and not
# evidence of anything learned. The agent's own contribution is the skew it adds
# on top,
#
# $$\text{quote centre} - \text{reservation price}
#   = a_{\text{skew}} \cdot \tfrac{1}{4}\, \sigma_t \, p_t$$
#
# which is zero whenever it picks the middle skew level. Both are plotted below:
# the total offset, which is what the market sees, and the skew component, which
# is the part a fixed rule would leave flat at zero.
#
# Either is read against the inventory held **when the quote was posted**, not
# the position on the same row after the fills: the latter is that inventory
# plus whatever the quote went on to trade, so plotting against it would build
# part of the answer into the x-axis.

# %%
policy_df = pl.DataFrame(
    [
        {
            "inventory": h["quote_inventory"],
            "quote_offset_bps": h["quote_offset_bps"],
            # The skew the agent chose, in the same units: the quote centre
            # measured from the reservation price the environment supplied.
            "skew_bps": (h["quote_center"] - h["reservation_price"])
            / max(h["mid_price"], 1e-6)
            * 10_000,
        }
        for run in results[RL_LABEL]
        for h in run["history"]
    ]
)

inv_min = int(policy_df["inventory"].min())
inv_max = int(policy_df["inventory"].max())
bin_edges = (
    np.array([inv_min - 1, inv_max + 1], dtype=float)
    if inv_min == inv_max
    else np.linspace(inv_min, inv_max, 9)
)
bin_ids = np.digitize(policy_df["inventory"].to_numpy(), bin_edges[1:-1], right=False)

# Inventory is whole units and each bin is [edge, next_edge), so the label names
# the integers that bin actually holds.
n_bins = len(bin_edges) - 1
bin_labels = [
    f"{max(int(np.ceil(bin_edges[i])), inv_min)} to "
    f"{inv_max if i == n_bins - 1 else int(np.ceil(bin_edges[i + 1])) - 1}"
    for i in range(n_bins)
]

offsets = policy_df["quote_offset_bps"].to_numpy()
skews = policy_df["skew_bps"].to_numpy()
grouped_df = pl.DataFrame(
    [
        {
            "inv_bin": label,
            "quote_offset_bps": float(offsets[bin_ids == idx].mean()),
            "skew_bps": float(skews[bin_ids == idx].mean()),
        }
        for idx, label in enumerate(bin_labels)
        if np.any(bin_ids == idx)
    ]
)

# %%
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=grouped_df["inv_bin"].to_list(),
        y=grouped_df["quote_offset_bps"].to_list(),
        name="Total offset from the mid",
        marker_color=COLORS["blue"],
    )
)
fig.add_trace(
    go.Bar(
        x=grouped_df["inv_bin"].to_list(),
        y=grouped_df["skew_bps"].to_list(),
        name="Skew the agent added",
        marker_color=COLORS["amber"],
    )
)
fig.add_hline(y=0, line=dict(color=COLORS["neutral"], width=1, dash="dot"))
fig.update_layout(
    barmode="group",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    title=(
        "Quote offset by inventory, split into the reservation shift and the skew"
        "<br><sup>Pooled across evaluation episodes. The environment centres every quote on "
        "the reservation price; only the second series is the agent's choice</sup>"
    ),
    xaxis_title="Inventory bucket (units)",
    yaxis_title="Average quote centre offset (bps)",
    height=420,
)
show_plotly_with_alt(
    fig,
    "Grouped bars per inventory bucket, pooled over the evaluation episodes. The first series is the total quote-centre offset from the mid in basis points for quotes posted while holding that inventory; the second is the skew the agent added on top of the environment's reservation price. A dotted line marks zero.",
)

# %% [markdown]
# ## 8. Key takeaways

# %%
same_sign = sum(1 for value, _, _ in all_gaps.values() if (value > 0) == (gap > 0))
agreement = (
    f"all {len(all_gaps)}" if same_sign == len(all_gaps) else f"{same_sign} of the {len(all_gaps)}"
)
display(
    Markdown(f"""
**Separate what the environment does from what the policy chose.** Every quote in this
simulator, the agent's and the rules', is centred on the reservation price, so a quote centre
that moves against inventory is the environment's arithmetic rather than a learned response.
The second series in section 7 is the part the agent chose: a fixed rule would leave it at
zero at every inventory level, so read it for how far the learned policy departs from one and
whether that departure varies with the position it is holding. The learned policy averages
{ppo_summary["avg_trades"]:.0f} trades an episode against
{min(row["avg_trades"] for row in baseline_summary):.0f} to
{max(row["avg_trades"] for row in baseline_summary):.0f} for the fixed rules, so it is quoting
at a different point on the fill-rate curve rather than simply quoting less.

**Reproducing a mechanism is not the same as improving on the rule that encodes it.** On mean
liquidated wealth the learned policy sits {gap_sigma:.1f} standard errors {direction} the
reference rule over {EVAL_EPISODES} paired episodes, and {direction} {agreement} of the rules.
The threshold this notebook fixed in advance is {RESOLUTION_SIGMA:.0f} standard errors, and it
was fixed in advance for a reason: with three baselines available, reading the ordering off
whichever comparison came out largest would be a selection artefact.

**Pair what you can pair, then check that pairing helped.** Every strategy here trades the
same price path in the same episode, so the difference can be taken per episode. That narrows
the standard error only when the two arms move together: the correlation between the learned
policy and the reference rule is reported above with the gap, and it is what decides whether
the paired interval is tighter than an unpaired one or wider.

### Known limitations

- The fill model has no queue, no order size and no cancellation. A quote either fills one unit
  or does not, with a probability that depends only on its distance from the mid and the
  imbalance, so nothing here captures queue position, which is most of the difficulty in real
  market making.
- Adverse selection is present but crude. The order imbalance both raises the fill
  probability on one side and pushes the next return in the same direction, so the maker is
  systematically filled just before a move against the position it has taken. What is missing
  is any informed participant: the imbalance is an exogenous mean-reverting series that knows
  nothing, so no counterparty here trades because it has worked something out.
- One PPO training run at one seed is one draw from a wide distribution. The sign of the gap
  above is a property of this run.
- The price process has no reaction to the maker's own quotes, so the maker is a price taker in
  a market it is nominally making.

**Next**: `04_crypto_execution_rl` returns to execution, on recorded perpetual-futures bars
rather than a simulated path.
""")
)
