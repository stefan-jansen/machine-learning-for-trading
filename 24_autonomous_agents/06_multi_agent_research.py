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
# # Multi-Agent Research
#
# **Docker image**: `ml4t`
#
# This notebook runs **N research agents** on one question and compares their
# probability forecasts. The pinned replay is a descriptive result from one
# dated run, not an experiment that isolates temperature, question
# contestability, or role design. Every saved model prompt included the market
# probability, so proximity to that value may reflect anchoring rather than
# independent agreement. NB07 changes both the question and the agent roles; it
# illustrates a different workflow but does not provide a controlled comparison.
#
# **A dated, point-in-time capture, replayed by default.** The numbers below come
# from one live run (provider `claude-sonnet-4`, Tavily web search) captured on
# 2026-06-09 and saved to `forecast_traces/`. By default this notebook *replays*
# that pinned trace (`RUN_LIVE = False`): every cell renders from the saved run,
# makes no API calls, and reproduces the figures the chapter discusses no matter
# when you run it. Prediction-market forecasts are not reproducible live - the
# market price and the web evidence both move, and once the market resolves you
# should swap in a current question of the same kind - so the published run is
# pinned rather than regenerated. To forecast a *current* question yourself, set
# `RUN_LIVE = True` with `ANTHROPIC_API_KEY` (or `OPENROUTER_API_KEY` with
# `OPENROUTER_MODEL=deepseek/deepseek-v4-pro` to drive an open model through
# OpenRouter - open models tend to search more, so raise `MAX_STEPS` if an agent
# runs out of steps before forecasting) and `TAVILY_API_KEY`; the live numbers
# will differ from those shown here. The `mock` provider is a deterministic CI
# smoke-test only - it returns a constant forecast and does **not** reproduce
# this run.
#
# **Learning Objectives**:
# - Run multiple agents in parallel via `ThreadPoolExecutor`
# - Audit a dated replay from question through evidence to probability
# - Compare simple-mean and Neyman aggregation under explicit assumptions
# - Analyze how the unestimated correlation parameter $\rho$ moves the aggregate
# - Distinguish a descriptive multi-agent result from a controlled experiment
#
# **Book Reference**: Chapter 24, Section 24.7 (Multi-Agent Forecasting Systems -
# Agent Ensemble)
#
# **Prerequisites**: NB04 (research agent), NB05 (aggregation math).

# %%
"""Multi-Agent Research - parallel agents on one shared question."""

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import polars as pl
from agent_fixtures import get_chapter_clear_question
from agent_observability import (
    TRACES_DIR,
    RunTrace,
    merge_calls,
    replay_llm_calls,
    show_agents,
    trace_llm,
)
from agent_pipeline import neyman_extremize, neyman_extremize_weighted
from agent_providers import TokenUsage, create_llm_client
from agent_research import ResearchAgent, format_agent_summary
from agent_schemas import AgentForecastArtifact
from agent_tools import create_search_client
from IPython.display import Markdown, display

from utils.style import COLORS, add_message_title, format_pct_axis

# %% tags=["parameters"]
# RUN_LIVE=False (the default) replays the pinned trace named in PINNED_TRACE:
# the notebook reloads that saved run and makes no API calls, so the outputs are
# stable and match the chapter. Set RUN_LIVE=True to forecast a current question
# live; that path needs API keys and produces different numbers.
RUN_LIVE = False
PINNED_TRACE = "06_multi_agent_research_20260609T141413Z_9fc4a2655471.json"

# Live-run settings (ignored when RUN_LIVE=False). Empty string auto-detects a
# provider from the keys in your environment (ANTHROPIC_API_KEY → OPENAI_API_KEY
# → GOOGLE_API_KEY → OPENROUTER_API_KEY → local Ollama → mock). The captured run
# used claude-sonnet-4. Set LLM_PROVIDER="openrouter" with
# OPENROUTER_MODEL="deepseek/deepseek-v4-pro" to drive an open model instead
# (raise MAX_STEPS for open models, which tend to issue more searches before
# forecasting). LLM_PROVIDER="mock" is a deterministic CI smoke-test only and
# does not reproduce the captured forecasts.
LLM_PROVIDER = ""
N_AGENTS = 3
MAX_STEPS = 5
MAX_SEARCH_RESULTS = 5

# %% [markdown]
# ## Setup
#
# All agents share the same LLM client and search provider. Each gets a unique
# `agent_id` but the same prompt and capabilities. The question is the pinned
# `CHAPTER_CLEAR_QUESTION` from `agent_fixtures.py` (*"Will the US enter a
# recession by the end of 2026?"*), a macro forecast with a prediction-market
# probability captured on 2026-06-09. The market value appears in every saved
# model prompt. The replay therefore documents market-aware forecasts;
# it does not compare agents with an independent market benchmark.

# %%
if RUN_LIVE:
    llm = create_llm_client(LLM_PROVIDER)
    search = create_search_client(LLM_PROVIDER)
    question = get_chapter_clear_question()
    provider_name = llm.model_name
    n_agents = N_AGENTS
else:
    # Replay: load the pinned trace and rebuild the question it forecast.
    pinned_run = RunTrace.load(TRACES_DIR / PINNED_TRACE)
    question = pinned_run.question_obj()
    provider_name = pinned_run.provider
    n_agents = len(pinned_run.agents)

print(f"Mode:         {'LIVE' if RUN_LIVE else 'REPLAY (pinned 2026-06-09 trace)'}")
print(f"Provider:     {provider_name}")
print(f"Question:     {question.question}")
print(f"Market p_yes: {question.current_market_price}")
print(f"Agents:       {n_agents}")

# %% [markdown]
# ## Running Agents in Parallel
#
# `ThreadPoolExecutor` runs agents concurrently. Each agent independently
# decides what to search and when to forecast.


# %%
def run_agent(agent_id: str):
    """Run a single research agent under its own tracer.

    The pool shares one `llm` and one `search` client across worker threads,
    so thread safety here depends on those providers being safe under
    concurrent calls. Anthropic and Tavily clients are; if you swap in a
    provider that is not, construct the client inside this function.

    Each agent wraps the shared client in its own `TracingLLMClient` labeled
    with the `agent_id`. That records every prompt the agent sent and every
    raw response it received - the full conversation, not just the parsed
    forecast - and a per-agent tracer keeps the capture thread-safe and
    correctly attributed. We merge the call logs afterward.
    """
    tracer = trace_llm(llm, label=agent_id)
    agent = ResearchAgent(
        llm=tracer,
        search=search,
        agent_id=agent_id,
        max_steps=MAX_STEPS,
        max_search_results=MAX_SEARCH_RESULTS,
    )
    return agent.run(question, market_price=question.current_market_price), tracer


# %% [markdown]
# The live path launches one task per agent. The publication path instead
# rehydrates the three saved artifacts and their raw model conversations, then
# sorts them into a stable display order.

# %%
artifacts: list[AgentForecastArtifact] = []

if RUN_LIVE:
    tracers = []
    with ThreadPoolExecutor(max_workers=N_AGENTS) as pool:
        futures = {pool.submit(run_agent, f"agent_{i}"): i for i in range(N_AGENTS)}
        for future in as_completed(futures):
            artifact, tracer = future.result()
            artifacts.append(artifact)
            tracers.append(tracer)
    llm_calls = merge_calls(*tracers)
else:
    # Replay: rehydrate the saved artifacts and the raw model conversation. The
    # display cells below cannot tell these apart from a live run's outputs.
    artifacts = pinned_run.agent_artifacts()
    llm_calls = pinned_run.call_log()

# Sort by agent_id for consistent display
artifacts.sort(key=lambda a: a.agent_id)

# The publication replay is a fixed evidence record. These assertions fail closed
# if the named trace or its expected three-agent result changes.
if not RUN_LIVE:
    assert [a.p_yes for a in artifacts] == [0.12, 0.22, 0.22]
    assert len(llm_calls) == 12

# %% [markdown]
# ## Agent Results
#
# Each agent produces a probability, confidence, sentiment, and evidence
# trail. The Polars DataFrame puts these in sortable columns next to the
# prediction market's own implied probability. Because that market value was
# included in every prompt, the comparison is context rather than an independent
# benchmark. Readers can compare every
# agent's forecast against the market in a single readout.

# %%
total_tokens = TokenUsage()
for a in artifacts:
    total_tokens = total_tokens + a.token_usage

panel_df = pl.DataFrame(
    [
        {
            "agent_id": a.agent_id,
            "p_yes": round(a.p_yes, 3),
            "confidence": round(a.confidence, 3),
            "sentiment": a.sentiment.value,
            "queries": a.search_queries_made,
            "sources": a.sources_consulted,
        }
        for a in artifacts
    ]
)
print(f"Market p_yes (Polymarket):  {question.current_market_price:.2f}")
print(f"Total agent tokens:         {total_tokens.total_tokens:,}\n")
panel_df

# %% [markdown]
# The bar chart makes the observed dispersion visible. The market line is
# labeled as prompt context because it was available to every agent.

# %%
probabilities = panel_df["p_yes"].to_list()
probability_range = max(probabilities) - min(probabilities)

fig, ax = plt.subplots()
bars = ax.bar(
    panel_df["agent_id"].to_list(),
    probabilities,
    color=COLORS["blue"],
    width=0.6,
)
ax.axhline(
    question.current_market_price,
    color=COLORS["amber"],
    linestyle="--",
    linewidth=1.5,
    label=f"Market context ({question.current_market_price:.1%})",
)
ax.bar_label(bars, labels=[f"{value:.0%}" for value in probabilities], padding=3)
ax.set_xlabel("Research Agent")
ax.set_ylabel("Probability of Recession")
ax.set_ylim(0, max(probabilities) + 0.08)
format_pct_axis(ax)
add_message_title(
    ax,
    f"Pinned agents differ by {probability_range:.0%}",
    subtitle="Three market-aware forecasts from the 2026-06-09 replay",
)
ax.legend(loc="upper left")
fig.tight_layout()
fig.show()
plt.show()

# %% [markdown]
# ## Agent Reasoning
#
# The panel above shows only the final probability and confidence; the *why*
# behind each forecast is invisible. `show_agents` from `agent_observability`
# renders the full captured timeline for each agent - every search query, the
# documents it retrieved (title, date, URL, and a snippet of the body), and the
# untruncated rationale, key findings, and uncertainties - so the reader can
# trace question → evidence → interpretation → probability step by step. Each
# `artifact` carries this in its `traces`, the same `AgentTrace` records built
# in NB04; the renderer just lays them out in chronological order. Reading the
# timelines side by side is how you tell apart the two cases that matter: agents
# that reach a similar number by genuinely different search paths (expected),
# versus agents pulling identical evidence and reasoning in lockstep (a sign the
# stochasticity has been switched off: investigate the temperature setting or a
# bug).

# %%
print(show_agents(artifacts))

# %%
probability_text = ", ".join(f"{value:.2f}" for value in probabilities)
display(
    Markdown(
        f"**Interpretation**: The pinned replay has probabilities {probability_text}, "
        f"for a range of {probability_range:.2f}. Those numbers describe this run only. "
        "They do not identify why the agents differed because temperature, retrieval paths, "
        "prompt wording, and the question itself were not varied independently. The shared "
        "market context also creates a direct anchoring channel. NB07 and NB08 demonstrate "
        "alternative workflows, but their different questions and role setups prevent a "
        "causal comparison."
    )
)

# %% [markdown]
# ## Aggregation: Comparing Methods
#
# Three aggregation approaches applied to the same agent outputs:
#
# | Method | What it does |
# |--------|-------------|
# | Simple mean | Ignores correlation structure |
# | Neyman | Accounts for correlation, pushes away from base rate |
# | Weighted Neyman | Also incorporates per-agent confidence |

# %%
probs = [a.p_yes for a in artifacts]
weights = [a.confidence for a in artifacts]

simple_mean = sum(probs) / len(probs)
result_neyman = neyman_extremize(probs, base=0.5, correlation=0.3)
result_weighted = neyman_extremize_weighted(probs, weights, base=0.5, correlation=0.3)

agg_df = pl.DataFrame(
    [
        {
            "method": "Simple mean",
            "aggregate": round(simple_mean, 3),
            "d": None,
            "n_eff": None,
        },
        {
            "method": "Neyman (ρ=0.3)",
            "aggregate": round(result_neyman.extremized_probability, 3),
            "d": round(result_neyman.extremization_factor, 3),
            "n_eff": round(result_neyman.effective_n, 2),
        },
        {
            "method": "Weighted Neyman",
            "aggregate": round(result_weighted.extremized_probability, 3),
            "d": round(result_weighted.extremization_factor, 3),
            "n_eff": round(result_weighted.effective_n, 2),
        },
    ]
)
agg_df

# %%
display(
    Markdown(
        f"**Finding**: At the illustrative setting $\\rho=0.3$, the simple mean is "
        f"{simple_mean:.3f}, unweighted Neyman is "
        f"{result_neyman.extremized_probability:.3f}, and confidence-weighted Neyman is "
        f"{result_weighted.extremized_probability:.3f}. The shift is conditional on the "
        "correlation assumption, while the difference between the two Neyman estimates comes "
        "from heuristic confidence weights."
    )
)

# %% [markdown]
# ## Sensitivity: Correlation Assumption
#
# The correlation parameter $\rho$ is the most important assumption in Neyman
# extremization. Here we sweep it to show the impact.

# %%
correlation_rows = []
for rho in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    r = neyman_extremize(probs, base=0.5, correlation=rho)
    rw = neyman_extremize_weighted(probs, weights, base=0.5, correlation=rho)
    correlation_rows.append(
        {
            "correlation": rho,
            "neyman": round(r.extremized_probability, 3),
            "weighted_neyman": round(rw.extremized_probability, 3),
            "shift_from_mean": round((r.extremized_probability or 0) - simple_mean, 3),
        }
    )
correlation_df = pl.DataFrame(correlation_rows)
correlation_df

# %% [markdown]
# The sensitivity chart treats $\rho$ as an assumption, not an estimate. The
# simple mean supplies the no-extremization baseline; the two Neyman curves show
# how strongly the aggregate can move when the assumed dependence changes.

# %%
rho_values = correlation_df["correlation"].to_list()

fig, ax = plt.subplots()
ax.plot(
    rho_values,
    correlation_df["neyman"].to_list(),
    "o-",
    color=COLORS["blue"],
    linewidth=2,
    label="Neyman",
)
ax.plot(
    rho_values,
    correlation_df["weighted_neyman"].to_list(),
    "s--",
    color=COLORS["copper"],
    linewidth=1.5,
    label="Confidence-weighted Neyman",
)
ax.axhline(
    simple_mean,
    color=COLORS["neutral"],
    linestyle=":",
    linewidth=1.5,
    label=f"Simple mean ({simple_mean:.1%})",
)
ax.set_xlabel(r"Assumed Pairwise Correlation ($\rho$)")
ax.set_ylabel("Aggregate Probability")
ax.set_xlim(0, 0.9)
ax.set_ylim(0, 0.22)
format_pct_axis(ax)
add_message_title(
    ax,
    "Correlation choice drives extremization",
    subtitle="Conditional aggregates from the same three pinned forecasts",
)
ax.legend(loc="lower right")
fig.tight_layout()
fig.show()
plt.show()

# %% [markdown]
# **Finding**: The aggregation conclusion is conditional on $\rho$. At low
# assumed correlation, agreement is amplified; at high assumed correlation, the
# aggregate approaches the simple mean. The replay does not estimate $\rho$, and
# the confidence fields are heuristic metadata rather than validated skill
# weights. The simple mean is therefore the transparent baseline. Neyman values
# are useful as a sensitivity analysis, not as uniquely supported forecasts.

# %% [markdown]
# ## Agent Summaries (Downstream Format)
#
# The supervisor (NB08) and debate (NB07) stages receive agent outputs in this
# summary format.

# %%
all_summaries = "\n\n---\n\n".join(format_agent_summary(a) for a in artifacts)
print(all_summaries)

# %% [markdown]
# ## Search Execution Audit
#
# The per-agent reasoning above shows each query and the titles it returned.
# This cell rolls the same `traces` into one sortable table - every search,
# which agent issued it, and how many results came back - so the reader can
# scan the whole panel's evidence-gathering at a glance. Overlapping query sets
# across agents document shared retrieval context, but do not by themselves
# explain the probability range.

# %%
audit_df = pl.DataFrame(
    [
        {
            "agent_id": a.agent_id,
            "step": t.step,
            "query": t.query,
            "results": len(t.results),
        }
        for a in artifacts
        for t in a.traces
        if t.action == "search"
    ]
)
audit_df

# %% [markdown]
# Publication dates are required to verify whether a result was available at the
# forecast timestamp. The saved search schema contains that field, but every
# result in this replay leaves it empty. The trace remains useful for reproducing
# what the agents saw; it cannot establish the source vintage independently.

# %%
search_results = [
    result for artifact in artifacts for trace in artifact.traces for result in trace.results
]
dated_sources = sum(result.published is not None for result in search_results)
market_mentions = sum(f"{question.current_market_price:.3f}" in str(call) for call in llm_calls)

display(
    Markdown(
        f"**Replay audit**: {market_mentions} of {len(llm_calls)} saved model calls include "
        f"the market probability; {dated_sources} of {len(search_results)} retrieved results "
        "record a publication date."
    )
)

# %% [markdown]
# ## Persisting the Full Run Trace
#
# Auditing an agent means being able to reconstruct exactly what it saw and
# said. `RunTrace.capture` bundles the question, the parameters, every agent's
# structured artifact, the aggregation result, and the complete raw model
# conversation (`llm_calls` - every prompt sent and every response received,
# captured by the per-agent `TracingLLMClient`s above) into one object, and
# `save()` writes it to `forecast_traces/` as JSON. That saved file is exactly
# what the replay path above reloads: it is the durable record of this
# point-in-time run, which a reviewer can reopen long after the live market and
# web evidence have moved on. A live run (`RUN_LIVE = True`) writes a fresh trace
# here; the default replay run reports the pinned trace it loaded instead of
# overwriting it.

# %%
if RUN_LIVE:
    run = RunTrace.capture(
        notebook="06_multi_agent_research",
        provider=provider_name,
        question=question,
        params={
            "n_agents": N_AGENTS,
            "max_steps": MAX_STEPS,
            "max_search_results": MAX_SEARCH_RESULTS,
        },
        agents=artifacts,
        aggregation=result_neyman,
        final_probability=result_neyman.extremized_probability,
        notes="Parallel research agents on one pinned, market-aware question.",
        llm_calls=llm_calls,
    )
    trace_path = run.save()
    print(
        f"Saved {len(run.llm_calls)} model calls "
        f"({run.total_tokens():,} tokens) → {trace_path.relative_to(trace_path.parents[1])}"
    )
else:
    # Replay: report the pinned trace we loaded rather than writing a new file.
    run = pinned_run
    trace_path = TRACES_DIR / PINNED_TRACE
    print(
        f"Replayed {len(run.llm_calls)} model calls "
        f"({run.total_tokens():,} tokens) from {trace_path.name}"
    )

# %% [markdown]
# ## Replaying the Raw Conversation
#
# The timeline view earlier is the *parsed* trace; this is the *raw* one. For
# the first agent, `replay_llm_calls` prints the exact messages the model
# received - system prompt, question, and each tool result fed back in - next
# to the untruncated JSON it returned at every step. This is the audit ground
# truth: the parsed forecast, the search queries, and the rationale all derive
# from these responses, and nothing here is summarized away. Pass
# `content_chars=None` to dump the complete payloads.

# %%
agent_0_calls = [c for c in llm_calls if c.label == "agent_0"]
print(replay_llm_calls(agent_0_calls, content_chars=600))

# %%
display(
    Markdown(
        f"""## Key Takeaways

1. **The pinned replay produces probabilities of {probability_text}, a range of
   {probability_range:.2f}.** This is a descriptive result from one market-aware
   run. It does not isolate temperature, retrieval, roles, or question
   contestability.
2. **Weighted Neyman tracks unweighted Neyman closely**
   ({result_weighted.extremized_probability:.3f} versus
   {result_neyman.extremized_probability:.3f}). The confidence weights are
   heuristic metadata, not validated skill weights.
3. **The aggregate's distance from the mean is driven by the assumed correlation
   $\\rho$.** The replay does not estimate $\\rho$, so the simple mean is the
   transparent baseline.
4. **The audit trail is incomplete for source vintage.** Only {dated_sources} of
   {len(search_results)} retrieved results record publication dates.
5. **Agent summaries** are the structured input for downstream debate (NB07) and
   supervisor reconciliation (NB08).

**Next**: [`07_adversarial_debate`](07_adversarial_debate.ipynb) - bull versus bear
debate on a different question with explicit bull and bear roles.

**Book**: Section 24.7 presents multi-agent forecasting systems. This notebook
supplies one dated replay and an assumption-sensitive aggregation audit."""
    )
)
