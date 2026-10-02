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
# # Bull vs Bear Debate
#
# **Docker image**: `ml4t`
#
# A panel that disagrees is more useful than a panel that agrees, and averaging it throws the
# disagreement away. **Adversarial debate** is one way to spend it instead: assign one model
# the strongest case for yes and another the strongest case for no, show each the other's
# argument, and watch what happens to the distance between them over a few rounds.
#
# The probability gap and transcript show how the two positions change.
# Convergence can reflect new evidence, persuasion or shared error. Inspect
# the arguments and sources; the gap alone cannot establish accuracy or
# whether debate improves on the mean.
#
# **Learning Objectives**:
# - Run a multi-round debate in which each side argues against the other's previous position
# - Stop the debate when the two sides come within a stated distance of each other, rather than
#   after a fixed number of rounds
# - Read the gap trajectory to tell a debate that moved something from one that did not
# - Fold a debate midpoint back into an aggregate under a stated weight, and say what that
#   weight is a claim about
# - Inspect disagreement, probability shifts and call cost, then specify the resolved
#   outcomes and matched comparisons needed to judge forecasting value
#
# **Book Reference**: Chapter 24, Section 24.7 (Multi-Agent Forecasting Systems -
# Debate Pattern)
#
# **Prerequisites**: [`04_research_agent`](04_research_agent.ipynb) (the agent),
# [`05_aggregation_math`](05_aggregation_math.ipynb) (aggregation),
# [`06_multi_agent_research`](06_multi_agent_research.ipynb) (running a panel).
#
# **The question, and why it is not the one used in 06.** This notebook forecasts
# `CHAPTER_CONTESTED_QUESTION`, *"Will the Federal Reserve hike rates in 2026?"*, where credible
# evidence points both ways and the research agents consequently spread out.
# [`06_multi_agent_research`](06_multi_agent_research.ipynb) used `CHAPTER_CLEAR_QUESTION`, where
# the evidence points one way and the agents landed close together. The agents are identical
# across the two notebooks; the question is what differs, and a panel that agrees gives debate
# nothing to work on.
#
# As in 06, the numbers come from one live capture replayed by default, so the table, the
# transcript and the figure are the same on every machine; the setup cell reports which
# provider made it and when. `RUN_LIVE = True` with `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`
# debates a current question instead and will not reproduce these values.

# %%
import sys

from utils.paths import get_chapter_dir

sys.path.insert(0, str(get_chapter_dir(24)))

"""Bull vs Bear Debate - adversarial stress-testing of forecasts."""


import matplotlib.pyplot as plt
import polars as pl
from agent_fixtures import get_chapter_contested_question
from agent_observability import (
    TRACES_DIR,
    RunTrace,
    merge_calls,
    replay_llm_calls,
    show_agents,
    show_debate_transcript,
    trace_llm,
)
from agent_pipeline import neyman_extremize, validate_probabilities
from agent_providers import ChatMessage, TokenUsage, create_llm_client
from agent_research import ResearchAgent, format_agent_summary, parse_json
from agent_schemas import AgentForecastArtifact, DebateArtifact, DebateRound
from agent_tools import create_search_client

from utils.style import show_with_alt

# %% tags=["parameters"]
# RUN_LIVE=False (the default) replays the pinned 2026-06-09 trace named below:
# the notebook reloads that saved run and makes no API calls. Set RUN_LIVE=True
# (with API keys) to debate a current question live; that path produces different
# numbers.
RUN_LIVE = False
PINNED_TRACE = "07_adversarial_debate_20260609T141631Z_cf1ad76cf379.json"

# Empty string auto-detects a provider (see NB06's note); the captured run used
# claude-sonnet-4. LLM_PROVIDER="mock" is a CI smoke-test only (live path).
LLM_PROVIDER = ""
N_AGENTS = 3
DEBATE_ROUNDS = 3
CONSENSUS_THRESHOLD = 0.05
MAX_STEPS = 5

# %% [markdown]
# ## Debate Prompts: Bull
#
# The bull argues for a higher probability of YES. In round 1, `{bear_section}`
# is empty and the bull argues without seeing the bear's position. In
# subsequent rounds, the placeholder carries the bear's previous argument
# and the bull must directly address it.

# %%
BULL_PROMPT_TEMPLATE = """\
You are the BULL debater in a structured forecasting debate.

Your role is to argue for a HIGHER probability of YES for the question below.
You must present the strongest possible case for YES, backed by evidence.

QUESTION:
{question}

AGENT SUMMARIES:
{agent_summaries}

CURRENT AGGREGATE PROBABILITY: {aggregate_p_yes}

{bear_section}

Output JSON only:
{{"argument": "Your strongest case for a higher probability of YES", "p_yes": 0.XX, "key_evidence": ["evidence point 1", "evidence point 2", "evidence point 3"]}}"""


# %% [markdown]
# ## Debate Prompts: Bear
#
# The bear argues for a lower probability of YES. The bull's argument and
# probability are inserted into the template every round, so the bear is
# always responding to the most recent bull position.

# %%
BEAR_PROMPT_TEMPLATE = """\
You are the BEAR debater in a structured forecasting debate.

Your role is to argue for a LOWER probability of YES for the question below.
You must present the strongest possible case for NO (or lower probability), backed by evidence.

QUESTION:
{question}

AGENT SUMMARIES:
{agent_summaries}

CURRENT AGGREGATE PROBABILITY: {aggregate_p_yes}

BULL'S ARGUMENT:
{bull_argument}
Bull's probability: {bull_probability}

You must directly address the Bull's points and explain why the probability should be lower.

Output JSON only:
{{"argument": "Your strongest case for a lower probability of YES", "p_yes": 0.XX, "key_evidence": ["evidence point 1", "evidence point 2", "evidence point 3"]}}"""

# %% [markdown]
# ## Single-round driver
#
# Pulling one round's bull→bear→consensus-check into its own function keeps
# the DebateAgent class small. The driver issues two LLM calls, parses
# both, and returns a `DebateRound` plus the round's token usage.


# %%
def _run_debate_round(
    llm,
    round_num: int,
    question: str,
    agent_summaries: str,
    aggregate_p_yes: float,
    prev_bear_argument: str | None,
    prev_bear_probability: float | None,
    consensus_threshold: float,
    artifact: DebateArtifact,
) -> tuple[DebateRound, TokenUsage]:
    """Run one bull→bear round. Returns (DebateRound, round_token_usage)."""
    round_tokens = TokenUsage()

    bear_section = ""
    if prev_bear_argument is not None:
        bear_section = (
            f"BEAR'S PREVIOUS ARGUMENT:\n{prev_bear_argument}\n"
            f"Bear's probability: {prev_bear_probability:.4f}\n\n"
            "You must directly address the Bear's points and explain "
            "why the probability should be higher."
        )

    bull_prompt = BULL_PROMPT_TEMPLATE.format(
        question=question,
        agent_summaries=agent_summaries,
        aggregate_p_yes=f"{aggregate_p_yes:.4f}",
        bear_section=bear_section,
    )
    bull_raw, bull_tokens = llm.complete_with_usage(
        [ChatMessage(role="user", content=bull_prompt)], json_mode=True
    )
    round_tokens = round_tokens + bull_tokens
    artifact.token_usage = artifact.token_usage + bull_tokens
    bull_parsed = parse_json(bull_raw)
    bull_argument = bull_parsed.get("argument", "")
    if isinstance(bull_parsed["p_yes"], bool):
        raise ValueError("Probability must be a number, not a boolean")
    bull_p = float(bull_parsed["p_yes"])
    validate_probabilities([bull_p])
    bull_evidence = [str(e) for e in bull_parsed.get("key_evidence", [])]

    bear_prompt = BEAR_PROMPT_TEMPLATE.format(
        question=question,
        agent_summaries=agent_summaries,
        aggregate_p_yes=f"{aggregate_p_yes:.4f}",
        bull_argument=bull_argument,
        bull_probability=f"{bull_p:.4f}",
    )
    bear_raw, bear_tokens = llm.complete_with_usage(
        [ChatMessage(role="user", content=bear_prompt)], json_mode=True
    )
    round_tokens = round_tokens + bear_tokens
    artifact.token_usage = artifact.token_usage + bear_tokens
    bear_parsed = parse_json(bear_raw)
    bear_argument = bear_parsed.get("argument", "")
    if isinstance(bear_parsed["p_yes"], bool):
        raise ValueError("Probability must be a number, not a boolean")
    bear_probability = float(bear_parsed["p_yes"])
    validate_probabilities([bear_probability])
    bear_evidence = [str(e) for e in bear_parsed.get("key_evidence", [])]

    consensus = abs(bull_p - bear_probability) < consensus_threshold
    debate_round = DebateRound(
        round_number=round_num,
        bull_argument=bull_argument,
        bull_probability=bull_p,
        bear_argument=bear_argument,
        bear_probability=bear_probability,
        consensus_reached=consensus,
        bull_key_evidence=bull_evidence,
        bear_key_evidence=bear_evidence,
    )
    return debate_round, round_tokens


# %% [markdown]
# ## The DebateAgent Class
#
# Multi-round bull/bear debate. The class owns the LLM, the round budget,
# and the consensus threshold; `run()` walks the rounds via the driver
# above and assembles a `DebateArtifact` with the full transcript.


# %%
class DebateAgent:
    """Structured adversarial debate between bull and bear positions."""

    def __init__(
        self,
        llm,
        max_rounds: int = 3,
        consensus_threshold: float = 0.05,
    ) -> None:
        self.llm = llm
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.token_usage = TokenUsage()

    def run(
        self,
        question: str,
        agent_summaries: str,
        aggregate_p_yes: float,
    ) -> DebateArtifact:
        """Run the debate. Returns a DebateArtifact with full transcript."""
        self.token_usage = TokenUsage()
        self.artifact = DebateArtifact()
        rounds = self.artifact.rounds
        prev_bear_argument: str | None = None
        prev_bear_probability: float | None = None

        for round_num in range(1, self.max_rounds + 1):
            debate_round, round_tokens = _run_debate_round(
                self.llm,
                round_num,
                question,
                agent_summaries,
                aggregate_p_yes,
                prev_bear_argument,
                prev_bear_probability,
                self.consensus_threshold,
                self.artifact,
            )
            self.token_usage = self.token_usage + round_tokens
            rounds.append(debate_round)
            if debate_round.consensus_reached:
                break
            prev_bear_argument = debate_round.bear_argument
            prev_bear_probability = debate_round.bear_probability

        final_bull = rounds[-1].bull_probability if rounds else None
        final_bear = rounds[-1].bear_probability if rounds else None
        consensus_reached = rounds[-1].consensus_reached if rounds else False

        return DebateArtifact(
            rounds=rounds,
            bull_final_probability=final_bull,
            bear_final_probability=final_bear,
            consensus_reached=consensus_reached,
            early_termination=consensus_reached and len(rounds) < self.max_rounds,
            token_usage=self.token_usage,
        )


# %% [markdown]
# ## Setup: Run Research Agents
#
# We first run the research agents from NB06 to establish baseline probability
# estimates that the debate will stress-test - this time on the pinned contested
# question, where the agents are expected to disagree.

# %%
artifacts: list[AgentForecastArtifact] = []

if RUN_LIVE:
    llm = create_llm_client(LLM_PROVIDER)
    search = create_search_client()
    question = get_chapter_contested_question()
    provider_name = llm.model_name

    # Run N agents, each under its own tracer so the full conversation - every
    # prompt sent and every raw response - is captured and attributed per agent.
    agent_tracers = []
    for i in range(N_AGENTS):
        tracer = trace_llm(llm, label=f"agent_{i}")
        agent = ResearchAgent(llm=tracer, search=search, agent_id=f"agent_{i}", max_steps=MAX_STEPS)
        artifacts.append(agent.run(question, market_price=None))
        agent_tracers.append(tracer)
else:
    # Replay: reload the pinned trace and rehydrate the agent panel and debate.
    pinned_run = RunTrace.load(TRACES_DIR / PINNED_TRACE)
    question = pinned_run.question_obj()
    provider_name = pinned_run.provider
    artifacts = pinned_run.agent_artifacts()

artifacts.sort(key=lambda a: a.agent_id)
forecast_artifacts = [a for a in artifacts if a.p_yes is not None]
if not forecast_artifacts:
    run = RunTrace.capture(
        notebook="07_adversarial_debate",
        provider=provider_name,
        question=question,
        agents=artifacts,
        status="abstained",
        llm_calls=merge_calls(*agent_tracers),
    )
    raise RuntimeError(f"All researchers abstained. Diagnostic trace: {run.save()}")
probs = [a.p_yes for a in forecast_artifacts]
aggregate = neyman_extremize(probs, base=0.5, correlation=0.3)

print(f"Mode:         {'LIVE' if RUN_LIVE else 'REPLAY (pinned 2026-06-09 trace)'}")
print(f"Provider:     {provider_name}")
print(f"Question:     {question.question}")
print(f"Market p_yes: {question.current_market_price}\n")
print("Pre-debate agent estimates:")
for a in artifacts:
    print(f"  {a.agent_id}: p_yes={a.p_yes}, confidence={a.confidence:.2f}")
print(f"\nAggregate (Neyman ρ=0.3): {aggregate.extremized_probability:.2f}")

# %% [markdown]
# ### Pre-debate agent timelines
#
# Before the debate stress-tests them, here is the full captured run for each
# research agent - every query, the documents retrieved, and the untruncated
# rationale - rendered by the same `show_agents` observability helper used in
# NB06. On this contested question the agents enter the debate already
# disagreeing; the timelines show *which* evidence pulled each one toward its
# starting probability.

# %%
print(show_agents(artifacts))

# %% [markdown]
# ## Running the Debate
#
# The `DebateAgent` makes real LLM calls for each round. The bull and bear
# prompts include the agent summaries so both sides argue from the same
# evidence base.

# %%
agg_p = aggregate.extremized_probability or aggregate.raw_probability

if RUN_LIVE:
    agent_summaries = "\n\n---\n\n".join(format_agent_summary(a) for a in forecast_artifacts)
    debate_tracer = trace_llm(llm, label="debate")
    debate = DebateAgent(
        llm=debate_tracer,
        max_rounds=DEBATE_ROUNDS,
        consensus_threshold=CONSENSUS_THRESHOLD,
    )
    try:
        result = debate.run(
            question=question.question,
            agent_summaries=agent_summaries,
            aggregate_p_yes=agg_p,
        )
    except Exception as exc:
        failed_run = RunTrace.capture(
            notebook="07_adversarial_debate",
            provider=provider_name,
            question=question,
            agents=artifacts,
            aggregation=aggregate,
            debate=debate.artifact,
            status="failed",
            failure_reason=str(exc),
            execution_mode="live",
            llm_calls=merge_calls(*agent_tracers, debate_tracer),
        )
        failed_path = failed_run.save()
        raise RuntimeError(f"Debate failed: {exc}. Inspect {failed_path}") from exc
    debate_tokens = debate.token_usage.total_tokens
else:
    # Replay: the saved debate transcript drives every cell below.
    result = pinned_run.debate_obj()
    debate_tokens = result.token_usage.total_tokens

print(f"Debate completed: {len(result.rounds)} rounds")
print(f"Consensus reached: {result.consensus_reached}")
print(f"Bull final: {result.bull_final_probability:.2f}")
print(f"Bear final: {result.bear_final_probability:.2f}")
print(f"Debate tokens: {debate_tokens:,}")

# %% [markdown]
# ## Round-by-Round Analysis

# %%
rounds_df = pl.DataFrame(
    [
        {
            "round": r.round_number,
            "bull": round(r.bull_probability, 3),
            "bear": round(r.bear_probability, 3),
            "gap": round(abs(r.bull_probability - r.bear_probability), 3),
            "consensus": r.consensus_reached,
        }
        for r in result.rounds
    ]
)
rounds_df

# %% [markdown]
# The table above is the *shape* of the debate; the transcript below is its
# *substance*. `show_debate_transcript` prints each round in full - both sides'
# complete arguments and the key evidence they cited, with nothing truncated -
# so the reader can see not just that the gap stayed open but the reasoning each
# side used to hold its ground. This is the debate counterpart to NB06's
# per-agent timelines: the same auditing discipline applied to the adversarial
# stage.

# %%
print(show_debate_transcript(result))

# %% [markdown]
# ## Visualizing Probability Trajectory
#
# This figure tracks the bull-bear gap across rounds: whether it narrows as each
# side takes on the other's strongest arguments, or holds when the disagreement
# is genuine.

# %%
if len(result.rounds) >= 2:
    rounds_x = [r.round_number for r in result.rounds]
    bull_probs = [r.bull_probability for r in result.rounds]
    bear_probs = [r.bear_probability for r in result.rounds]
    midpoints = [(b + r) / 2 for b, r in zip(bull_probs, bear_probs, strict=False)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds_x, bull_probs, "g^-", markersize=10, label="Bull", linewidth=2)
    ax.plot(rounds_x, bear_probs, "rv-", markersize=10, label="Bear", linewidth=2)
    ax.plot(rounds_x, midpoints, "ko--", markersize=6, label="Midpoint", alpha=0.6)

    ax.fill_between(rounds_x, bull_probs, bear_probs, alpha=0.15, color="gray")

    ax.axhline(
        agg_p,
        color="blue",
        linestyle=":",
        alpha=0.5,
        label=f"Pre-debate aggregate ({agg_p:.2f})",
    )

    ax.set_xlabel("Debate Round")
    ax.set_ylabel("Probability")
    ax.set_title("Bull-Bear Probability Trajectory")
    ax.legend(loc="best")
    y_min = min(bear_probs + bull_probs + [agg_p])
    y_max = max(bear_probs + bull_probs + [agg_p])
    pad = max(0.05, (y_max - y_min) * 0.15)
    ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
    show_with_alt(
        fig,
        "Bull and bear probabilities, their midpoint and the initial aggregate are plotted across the retained debate rounds. Agreement does not establish forecasting accuracy.",
    )

# %% [markdown]
# **Interpretation**: The plot tracks the bull-bear gap across rounds.
# Agreement shows convergence within this discussion; it does not establish
# accuracy, evidence quality, or calibrated confidence. The captured run
# keeps a wide gap across all three rounds, so no consensus update was reached.

# %% [markdown]
# ## Post-Debate Probability
#
# The following 70/30 blend is an illustrative aggregation choice applied to this
# saved debate. It is not the AIA paper's rule and is not used by the repaired
# capstone, which uses a debate consensus when one is reached.

# %%
pre_debate = agg_p
debate_midpoint = (result.bull_final_probability + result.bear_final_probability) / 2

blended = 0.7 * pre_debate + 0.3 * debate_midpoint

print(f"Pre-debate aggregate:  {pre_debate:.2f}")
print(f"Debate midpoint:       {debate_midpoint:.2f}")
print(f"Blended (70/30):       {blended:.2f}")
print(f"Shift from debate:     {blended - pre_debate:+.2f}")

if result.consensus_reached:
    print("\nDebate reached consensus; accuracy remains untested.")
else:
    print("\nNo consensus was reached in this run.")

# %% [markdown]
# ## Probability changes and debate cost
#
# The cell below reports pre-debate disagreement and the shift under our
# chosen blend. These are descriptive diagnostics. Accuracy and incremental
# value require resolved outcomes and comparable runs with and without debate.

# %%
agent_range = max(probs) - min(probs)
debate_shift = abs(blended - pre_debate)

print(f"Agent disagreement: {agent_range:.2f} ({agent_range:.0%})")
print(f"Debate probability shift: {debate_shift:.2f} ({debate_shift:.0%})")

if agent_range > 0.15 and debate_shift > 0.03:
    print("→ High disagreement and a probability shift; accuracy is untested")
elif agent_range < 0.05:
    print("→ Agents already agreed; examine whether debate adds evidence")
else:
    print("→ A small probability shift under this blend")

# %% [markdown]
# ## Persisting the Full Run Trace
#
# The same auditing discipline as NB06, now covering both stages. `RunTrace`
# bundles the question, the research-agent artifacts, the complete debate
# transcript, and the raw model conversation for every research and debate call
# - captured by the per-agent and debate `TracingLLMClient`s - into one JSON
# record under `forecast_traces/`. Reload it to replay exactly what each
# debater was shown and how it responded, round by round.

# %%
if RUN_LIVE:
    llm_calls = merge_calls(*agent_tracers, debate_tracer)
    run = RunTrace.capture(
        notebook="07_adversarial_debate",
        provider=provider_name,
        question=question,
        params={
            "n_agents": N_AGENTS,
            "debate_rounds": DEBATE_ROUNDS,
            "consensus_threshold": CONSENSUS_THRESHOLD,
            "max_steps": MAX_STEPS,
        },
        agents=artifacts,
        aggregation=aggregate,
        debate=result,
        final_probability=blended,
        notes="Bull/bear debate on the pinned contested question.",
        llm_calls=llm_calls,
    )
    trace_path = run.save()
    print(
        f"Saved {len(run.llm_calls)} model calls "
        f"({run.total_tokens():,} tokens) → {trace_path.relative_to(trace_path.parents[1])}"
    )
else:
    # Replay: report the pinned trace we loaded rather than writing a new file.
    llm_calls = pinned_run.call_log()
    run = pinned_run
    trace_path = TRACES_DIR / PINNED_TRACE
    print(
        f"Replayed {len(run.llm_calls)} model calls "
        f"({run.total_tokens():,} tokens) from {trace_path.name}"
    )

# %% [markdown]
# ## Replaying the Debate Calls
#
# The raw audit view for the debate: every bull and bear prompt - including the
# opposing side's previous argument that gets fed back in each round - next to
# the untruncated JSON each debater returned. The transcript and trajectory
# figure above are both derived from exactly these responses.

# %%
debate_calls = [c for c in llm_calls if c.label == "debate"]
print(replay_llm_calls(debate_calls, content_chars=700))

# %% [markdown]
# ## Key Takeaways
#
# 1. **Gap trajectory** describes agreement. Inspect the cited arguments to
#    determine whether new evidence was used; narrowing alone does not establish
#    learning, and a flat gap alone does not establish that the discussion was useless.
# 2. **Midpoint** is the arithmetic mean of the final bull and bear probabilities.
#    Their opposing prompts may produce different estimates from the initial
#    research ensemble; the averaging operation is the same.
# 3. **Consensus stopping** avoids remaining calls when the declared threshold is
#    reached. It may also omit useful later discussion; the threshold is a design choice.
# 4. **Forecasting value** requires resolved outcomes and matched runs with and
#    without debate. This trace demonstrates the mechanism, not an accuracy gain.
#
# **Next**: [`forecasting_pipeline`](08_forecasting_pipeline.ipynb) - wire everything together into the full
# agent → aggregation → debate → supervisor pipeline.
#
# **Book**: Section 24.7 discusses the debate pattern in the context of Bridgewater's
# AIA system and prediction market design.
