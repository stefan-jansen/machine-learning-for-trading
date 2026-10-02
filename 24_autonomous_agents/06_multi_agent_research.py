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
# One research agent gives one answer with no way to tell a considered probability from a
# lucky search. Running several is the cheapest available check, and it only works if they
# disagree: three agents returning the same number are one agent charged for three times.
#
# This notebook runs three copies of the agent from
# [`04_research_agent`](04_research_agent.ipynb) on one question, in parallel, and looks at
# what came back. Nothing distinguishes the three except their identifiers, which is the point:
# no roles, no settings, nothing assigned to make them differ. Two things can still make them
# differ anyway - the model samples its replies, and each agent's searches return different
# documents - and this run separates neither.
#
# **What this run establishes, and what it does not.** The numbers below are one live capture
# (`claude-sonnet-4`, Tavily search) taken on 2026-06-09 and replayed by default, so the
# figures are the same whenever the notebook runs. One capture on one question is a
# demonstration that a panel of identical agents is not degenerate. It is not an experiment:
# nothing here varies temperature, retrieval, or question wording independently, so the spread
# cannot be attributed to any of them. And the market's own probability was in every prompt,
# which rules out reading the agents' closeness to the market as skill.
#
# **Learning Objectives**:
# - Run several agents concurrently and keep each one's conversation separately attributable
# - Read a panel's timelines to tell genuine disagreement from one agent run three times
# - Exclude agents that never produced a forecast before aggregating
# - Show how far the assumed correlation moves an aggregate, relative to how far the agents
#   are apart
# - Say what a single dated capture of a multi-agent run does and does not establish
#
# **Book Reference**: Chapter 24, Section 24.7 (Multi-Agent Forecasting Systems -
# Agent Ensemble)
#
# **Prerequisites**: [`04_research_agent`](04_research_agent.ipynb) (the agent),
# [`05_aggregation_math`](05_aggregation_math.ipynb) (the aggregation rules).

# %%
import sys

from utils.paths import get_chapter_dir

sys.path.insert(0, str(get_chapter_dir(24)))

"""Multi-Agent Research — parallel agents on one shared question."""

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

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

# %% tags=["parameters"]
# RUN_LIVE=False (the default) replays the pinned trace named in PINNED_TRACE:
# the notebook reloads that saved run and makes no API calls, so the outputs are
# stable and match the chapter. Set RUN_LIVE=True to forecast a current question
# live; that path needs API keys and produces different numbers.
RUN_LIVE = False
PINNED_TRACE = "06_multi_agent_research_20260609T141413Z_9fc4a2655471.json"

# Live-run settings (ignored when RUN_LIVE=False). Blank reads the provider from
# your .env (LLM_PROVIDER + LLM_API_KEY); the captured run used claude-sonnet-4.
# Set LLM_PROVIDER="openrouter" with LLM_MODEL="deepseek/deepseek-chat-v3.1" to
# drive an open model instead (raise MAX_STEPS for open models, which tend to
# issue more searches before forecasting). LLM_PROVIDER="mock" is a deterministic
# CI smoke-test only and does not reproduce the captured forecasts.
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
# recession by the end of 2026?"*), a macro forecast the Polymarket market priced
# at `p_yes ≈ 0.18` on the 2026-06-09 capture date. Professional growth forecasts
# and recession-odds trackers are broadly and consistently reported, so the agents
# share a strong evidence base, and any spread from sampling alone has to show up
# here. NB07 switches to the pinned `CHAPTER_CONTESTED_QUESTION` (*"Will the Fed
# hike rates in 2026?"*, where credible evidence cuts both ways) and gets a wider
# spread, so watch which question each notebook forecasts.

# %%
if RUN_LIVE:
    llm = create_llm_client(LLM_PROVIDER)
    search = create_search_client()
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
    raw response it received — the full conversation, not just the parsed
    forecast — and a per-agent tracer keeps the capture thread-safe and
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
    return agent.run(question, market_price=None), tracer


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

# %% [markdown]
# ## Agent Results
#
# Each agent produces a probability, confidence, sentiment, and evidence
# trail. The Polars DataFrame puts these in sortable columns next to the
# Polymarket market's own implied probability — readers can compare every
# agent's forecast against the market in a single readout.

# %%
total_tokens = TokenUsage()
for a in artifacts:
    total_tokens = total_tokens + a.token_usage

panel_df = pl.DataFrame(
    [
        {
            "agent_id": a.agent_id,
            "status": a.status,
            "p_yes": round(a.p_yes, 3) if a.p_yes is not None else None,
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
# ## Agent Reasoning
#
# The panel above shows only the final probability and confidence; the *why*
# behind each forecast is invisible. `show_agents` from `agent_observability`
# renders the full captured timeline for each agent — every search query, the
# documents it retrieved (title, date, URL, and a snippet of the body), and the
# untruncated rationale, key findings, and uncertainties — so the reader can
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

# %% [markdown]
# **Interpretation**: On this one-directional question, the three agents land
# close together, at $p_{\text{yes}}$ of 0.12, 0.22, and 0.22. All three are
# bearish, at confidences from 0.56 to 0.76, and the captured search traces above
# draw on the same mainstream sources (professional-forecaster surveys, NBER
# business-cycle dating, recession-odds trackers). The agents reached those
# probabilities by different search paths, but the evidence pointed one way, so
# the forecasts cluster. That small spread is the expected outcome here, and it
# is negligible next to the much larger disagreement the *contested* question
# provokes in NB07. The result to flag would be the opposite: identical searches
# and identical reasoning across agents, which would mean the stochasticity has
# been switched off. Where a question is genuinely contested, the disagreement to
# aggregate comes from designed-in structure (distinct roles, adversarial
# framing), which is the motivation for NB07 and NB08.

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
forecast_artifacts = [a for a in artifacts if a.p_yes is not None]
if not forecast_artifacts:
    run = RunTrace.capture(
        notebook="06_multi_agent_research",
        provider=provider_name,
        question=question,
        agents=artifacts,
        llm_calls=llm_calls,
        status="abstained",
    )
    raise RuntimeError(f"All researchers abstained. Diagnostic trace: {run.save()}")
probs = [a.p_yes for a in forecast_artifacts]
weights = [a.confidence for a in forecast_artifacts]

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

# %% [markdown]
# **Finding**: Because the agents nearly agree and carry similar confidence,
# weighted Neyman is virtually indistinguishable from unweighted Neyman — both
# extremize the ~0.19 consensus away from the 0.5 base rate (to 0.071 and 0.063
# respectively) at $\rho=0.3$. Weighting only changes the aggregate when
# per-agent probabilities or confidences differ materially, which they barely do
# here. The shift from the simple mean (0.187) to the Neyman aggregate (0.071) is
# structural, driven by the correlation assumption rather than by any real
# cross-agent disagreement.

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
# The correlation values are illustrative assumptions, not measured dependence
# among these models. Confidence weights are model-reported or heuristic and are
# not calibrated reliability estimates. Use the sensitivity sweep to see how those
# assumptions change the arithmetic, not to establish a performance advantage.

# %% [markdown]
# ## Agent Summaries (Downstream Format)
#
# The supervisor (NB08) and debate (NB07) stages receive agent outputs in this
# summary format.

# %%
all_summaries = "\n\n---\n\n".join(format_agent_summary(a) for a in forecast_artifacts)
print(all_summaries)

# %% [markdown]
# ## Search Execution Audit
#
# The per-agent reasoning above shows each query and the titles it returned.
# This cell rolls the same `traces` into one sortable table — every search,
# which agent issued it, and how many results came back — so the reader can
# scan the whole panel's evidence-gathering at a glance. Overlapping query sets
# across agents explain the clustered forecasts: when the agents retrieve the
# same evidence base, they reach similar answers.

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
# ## Persisting the Full Run Trace
#
# Auditing an agent means being able to reconstruct exactly what it saw and
# said. `RunTrace.capture` bundles the question, the parameters, every agent's
# structured artifact, the aggregation result, and the complete raw model
# conversation (`llm_calls` — every prompt sent and every response received,
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
        notes="Parallel research agents on the pinned one-directional question.",
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
# received — system prompt, question, and each tool result fed back in — next
# to the untruncated JSON it returned at every step. This is the audit ground
# truth: the parsed forecast, the search queries, and the rationale all derive
# from these responses, and nothing here is summarized away. Pass
# `content_chars=None` to dump the complete payloads.

# %%
agent_0_calls = [c for c in llm_calls if c.label == "agent_0"]
print(replay_llm_calls(agent_0_calls, content_chars=600))

# %% [markdown]
# ## Key Takeaways
#
# 1. **Forecast spread depends on the question, not the temperature.** On this
#    one-directional question, the three agents landed in a tight cluster
#    (0.12–0.22), all bearish and drawing on the same sources, a spread far
#    smaller than the disagreement NB07 finds on a contested question. Parallel
#    agents are a useful baseline, not a diversity mechanism in themselves.
# 2. **Weighted Neyman tracks unweighted Neyman closely** (0.063 vs 0.071):
#    weighting moves the aggregate only when per-agent confidences differ enough
#    to matter, and here they are close.
# 3. **The aggregate's distance from the mean is driven by the correlation
#    assumption $\rho$**, not by cross-agent disagreement: at $\rho=0$ the Neyman
#    aggregate drops to 0.01; at $\rho=0.9$ it barely leaves the simple mean
#    (0.176 vs 0.187). For LLM agents using overlapping sources,
#    $\rho \in [0.3, 0.5]$ remains the reasonable default.
# 4. **ThreadPoolExecutor** still buys real wall-clock speedup, but the saved
#    minutes have to be weighed against the duplicated cost.
# 5. **Agent summaries** are the structured input for downstream debate (NB07)
#    and supervisor reconciliation (NB08), where a contested question supplies the
#    disagreement those stages reconcile.
#
# **Next**: [`adversarial_debate`](07_adversarial_debate.ipynb) — bull vs bear
# debate on a contested question, where credible evidence supports both sides.
#
# **Book**: Section 24.7 uses this run to show that forecast spread reflects the
# question's evidence: when it points one way, parallel agents agree, and the
# debate and supervisor stages earn their cost on genuinely contested questions.
