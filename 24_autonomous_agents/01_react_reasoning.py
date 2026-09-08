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
# # LLM Providers and the ReAct Loop
#
# **Docker image**: `ml4t`
#
# A **forecasting agent** answers a question by choosing what to look up, reading what comes
# back, and then committing to a number. That is a different shape of program from the models
# in earlier chapters, which are handed a prepared feature matrix and produce a prediction from
# it. This notebook builds the smallest useful version: a loop that alternates between
# reasoning about what it still needs and acting to get it, backed by a language model that can
# be swapped without touching the loop.
#
# **Learning Objectives**:
# - Call any of five language-model backends through one two-method interface, so the same
#   agent code runs against a free local model or a commercial API
# - Write a loop that alternates reasoning and tool calls (the **ReAct** pattern) and stops
#   either on a forecast or on a step budget you set
# - Validate a model's reply against a declared action schema before acting on it, and return
#   a malformed reply to the model for correction
# - Read a saved run record to see which searches were issued, what came back, and what the
#   loop did with each result
#
# **Book Reference**: Chapter 24, Sections 24.1-24.2 (From Prediction Functions to Agentic
# Workflows; Cognitive Architectures)
#
# **Prerequisites**: None. This is the first notebook in the Chapter 24 workshop.

# %%
"""LLM Providers and the ReAct Loop - multi-provider agent reasoning."""

import json
import math
from datetime import date

from agent_fixtures import get_live_question
from agent_observability import TRACES_DIR, RunTrace, trace_llm
from agent_providers import ChatMessage, MockLLMClient, TokenUsage, create_llm_client
from agent_schemas import AgentForecastArtifact, AgentTrace, ForecastQuestion
from agent_tools import ToolExecutor, create_search_client, format_search_results
from IPython.display import Markdown, display

# %% [markdown]
# ## Two ways to run this notebook
#
# `RUN_LIVE` decides where the agent's evidence comes from. Left at `False`, the notebook
# reloads a **run record**: a JSON file holding the question, every prompt sent to the model,
# every reply, and every search result, captured on 2026-06-15 from a Claude Sonnet model and
# the Tavily search API. Nothing is called over the network and the printed session is
# identical on every machine. Set it to `True`, with `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`
# in the environment, and the notebook fetches an open prediction-market question and runs the
# agent against today's web; that path costs money and produces different output each time.
#
# `MAX_STEPS` is the loop's budget: the agent may take at most this many turns, each of which
# is one search or one forecast. Five is enough for a handful of searches and a decision, and
# small enough that a model that never commits is stopped rather than left running.
#
# `LLM_PROVIDER` is empty so the client factory picks the first provider whose API key is set.
# It only takes effect on the live path.

# %% tags=["parameters"]
RUN_LIVE = False
PINNED_TRACE = "01_react_reasoning_20260615T191047Z_04eb6e7c603d.json"
LLM_PROVIDER = ""
MAX_STEPS = 5

# %% [markdown]
# **Optional dependencies** (for real LLM and web search; the chapter runs in
# deterministic mock mode without them):
#
# ```bash
# # Claude (recommended)
# uv pip install anthropic httpx
# # OpenAI (alternative)
# uv pip install openai httpx
# # Ollama (free, local). Install from https://ollama.com, then:
# #   ollama pull qwen2.5:32b
# ```
#
# Then set the relevant API keys in `.env`:
#
# - `ANTHROPIC_API_KEY` (https://console.anthropic.com)
# - `OPENAI_API_KEY` (https://platform.openai.com)
# - `GOOGLE_API_KEY` (https://aistudio.google.com)
# - `OPENROUTER_API_KEY` (https://openrouter.ai), one key for any model
#   (set `OPENROUTER_MODEL`, e.g. `anthropic/claude-sonnet-4`)
# - `TAVILY_API_KEY` (https://tavily.com, web search)
#
# `create_llm_client()` auto-selects a provider in the order Anthropic →
# OpenAI → Google → OpenRouter → local Ollama. Without any API key, all
# notebooks fall back to deterministic mock mode.

# %% [markdown]
# ## The LLM Provider Protocol
#
# Every notebook in this chapter reaches its language model through the same `LLMClient`
# **protocol**: a pair of method signatures that any provider class must offer, with no shared
# base class and no vendor SDK visible to the caller.
#
# - `complete(messages) → str` returns the reply text
# - `complete_with_usage(messages) → (str, TokenUsage)` returns the text and the prompt and
#   completion token counts, which is what makes a run's cost measurable
#
# Two methods is a deliberate floor. Everything a provider offers beyond them, such as
# streaming, native tool calling or thinking budgets, varies across vendors, and an agent that
# reaches for any of it is an agent that can no longer be moved.
#
# The cell below sends one arithmetic question. Its answer is worthless; what it establishes is
# that a client was constructed, a key was accepted, and a reply came back, before the loop
# starts spending money on real prompts. On the replay path there is no client, so the reply
# comes out of the saved record.

# %%
if RUN_LIVE:
    llm = create_llm_client(LLM_PROVIDER)
    provider_name = llm.model_name
    response = llm.complete(
        [ChatMessage(role="user", content="What is 2 + 2? Reply with just the number.")]
    )
else:
    # Replay: reload the pinned trace; the provider name and warm-up response
    # come from the saved run, and no live client is created.
    pinned_run = RunTrace.load(TRACES_DIR / PINNED_TRACE)
    assert pinned_run.notebook == "01_react_reasoning"
    assert len(pinned_run.agents) == 1
    llm = None
    provider_name = pinned_run.provider
    response = pinned_run.params.get("warmup_response", "")

print(f"Mode:     {'LIVE' if RUN_LIVE else 'REPLAY (pinned trace)'}")
print(f"Provider: {provider_name}")
print(f"Response: {response[:200]}")

# %% [markdown]
# ## The Forecasting Question
#
# The agent is pointed at a **prediction market** question: a contract that pays out if a
# stated event happens by a stated date, so its price is the market's probability that it will.
# Polymarket publishes these openly, and an unresolved one is useful here for a reason that
# matters when evaluating any language model on forecasting. A question whose answer is already
# in the model's training data tests recall, not forecasting. An open question does not exist in
# the training data at all, so the only way to say anything about it is to go and look.
#
# On the replay path the question comes out of the saved record. On the live path
# `get_live_question` calls the Polymarket API and falls back to a fixed demonstration question
# when that call fails, so the notebook still runs with no network.

# %%
question = get_live_question() if RUN_LIVE else pinned_run.question_obj()
print(f"Question: {question.question}")
if question.resolution_date:
    print(f"Resolves: {question.resolution_date}")
if question.current_market_price is not None:
    print(f"Market:   {question.current_market_price:.0%}")
if question.resolved_outcome is not None:
    print(f"Outcome:  {'YES' if question.resolved_outcome == 1.0 else 'NO'}")
else:
    print("Status:   OPEN (unresolved)")

# %% [markdown]
# ## Building the ReAct Agent
#
# The ReAct pattern interleaves **reasoning** (thinking about what to do) with **action**
# (calling tools) in a loop. The agent here has exactly two actions:
#
# - `{"action": "search", "query": "..."}`: search the web for evidence
# - `{"action": "forecast", "p_yes": 0.XX, "rationale": "..."}`: produce a probability
#
# The prompts appear in full below rather than hidden behind a helper, because what the model
# is told is as much a part of this program as the loop that calls it. The same two-action
# schema is used by the AIA Forecaster (Alur et al., 2025), the system the rest of this chapter
# reconstructs.

# %%
SYSTEM_PROMPT = """\
You are a forecasting agent in a multi-agent forecasting system.

Your job:
1) Gather evidence by issuing web/news search queries when needed.
2) Then produce a binary probability forecast for the question.

You must follow the action schema exactly and output valid JSON only.
You must not browse prediction market prices unless they are explicitly provided."""

# %% [markdown]
# ### Step prompt
#
# The step prompt provides the question, optional market context, and the action
# schema. The agent must output exactly one JSON action per step.


# %%
def build_step_prompt(q: ForecastQuestion, market_price: float | None = None) -> str:
    """Format the step prompt with question context."""
    prompt = f"QUESTION:\n{q.question}\n\n"
    if q.description:
        prompt += f"MARKET CONTEXT:\n{q.description}\n\n"
    if market_price is not None:
        prompt += f"MARKET IMPLIED PROBABILITY (p_yes):\n{market_price}\n\n"
    prompt += (
        "NEXT ACTION SCHEMA (output JSON only):\n"
        'If you need more info:\n{"action":"search","query":"..."}\n'
        "If you are ready to forecast:\n"
        '{"action":"forecast","p_yes":0.XX,"rationale":"short explanation '
        'grounded in evidence and base rates"}\n\n'
        "Pick exactly one action."
    )
    return prompt


# %% [markdown]
# ### Session setup
#
# A session opens with three pieces of state. The **cutoff date** is the last date the search
# tool may return documents from, which is what keeps an agent forecasting a past question from
# reading the answer. The tool executor holds the search client and is the only route the loop
# has to the outside world. The message list starts with the system prompt and the first step
# prompt, and grows by two entries per turn: what the model said, and what the tool returned.


# %%
def _init_react_session(
    question: ForecastQuestion,
    search_client,
    market_price: float | None,
) -> tuple[date | None, ToolExecutor, list[ChatMessage]]:
    """Build the cutoff date, tool executor, and initial messages for a ReAct run."""
    cutoff = date.fromisoformat(question.cutoff_date) if question.cutoff_date else None
    executor = ToolExecutor(search=search_client)
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=build_step_prompt(question, market_price)),
    ]
    return cutoff, executor, messages


# %% [markdown]
# ### Invalid-action retry
#
# Malformed JSON, schema-invalid values, and unknown actions are recorded and
# returned to the model for correction. Repeated failures reach the step-budget
# sentinel; a production decision layer should retry or abstain rather than
# treat it as a forecast.


# %%
def _record_invalid_action(
    action_type: str,
    response: str,
    messages: list[ChatMessage],
    traces: list[AgentTrace],
    step: int,
) -> None:
    """Record an invalid action and request a corrected JSON response."""
    traces.append(AgentTrace(step=step, action=action_type, llm_raw=response))
    messages.append(ChatMessage(role="assistant", content=response))
    messages.append(
        ChatMessage(role="tool", content="Error: output one valid search or forecast JSON action.")
    )
    print(f"  Step {step}: {action_type}; retrying")


# %% [markdown]
# ### Action validation
#
# Provider responses are untrusted input. Parse and validate the two documented
# schemas before the loop executes a search or accepts a forecast.


# %%
def _parse_action(response: str) -> tuple[str, dict[str, object] | None]:
    """Return a validated action type and object, or a bounded-retry failure."""
    try:
        action = json.loads(response)
    except (json.JSONDecodeError, ValueError):
        return "parse_failure", None
    if not isinstance(action, dict):
        return "schema_failure", None

    action_type = action.get("action")
    if not isinstance(action_type, str):
        return "schema_failure", None
    if action_type == "search":
        query = action.get("query")
        if not isinstance(query, str) or not query.strip():
            return "schema_failure", None
        action["query"] = query.strip()
    elif action_type == "forecast":
        p_yes = action.get("p_yes")
        rationale = action.get("rationale")
        if isinstance(p_yes, bool) or not isinstance(p_yes, (int, float)):
            return "schema_failure", None
        try:
            normalized_p_yes = float(p_yes)
        except (TypeError, ValueError, OverflowError):
            return "schema_failure", None
        if not math.isfinite(normalized_p_yes) or not isinstance(rationale, str):
            return "schema_failure", None
        action["p_yes"] = normalized_p_yes
    return action_type, action


# %% [markdown]
# ### The ReAct loop
#
# Each iteration asks the LLM to search or forecast, executes valid actions, and
# feeds observations back. The loop terminates on a valid forecast or the step
# limit. Forecast probabilities are clamped to the unit interval.


# %%
def run_react_agent(
    llm,
    search_client,
    question: ForecastQuestion,
    max_steps: int = 5,
    market_price: float | None = None,
) -> tuple[float, str, list[AgentTrace], TokenUsage]:
    """Run a ReAct forecasting agent. Returns (p_yes, rationale, traces, token_usage)."""
    cutoff, executor, messages = _init_react_session(question, search_client, market_price)
    traces: list[AgentTrace] = []
    total_tokens = TokenUsage()

    for step in range(1, max_steps + 1):
        response, usage = llm.complete_with_usage(messages, json_mode=True)
        total_tokens = total_tokens + usage

        action_type, action = _parse_action(response)
        if action is None:
            _record_invalid_action(action_type, response, messages, traces, step)
            continue

        if action_type == "search":
            query = str(action["query"])
            results = executor.execute_search(query, max_results=5, cutoff_date=cutoff)
            traces.append(AgentTrace(step=step, action="search", query=query, results=results))
            messages.append(ChatMessage(role="assistant", content=response))
            messages.append(ChatMessage(role="tool", content=format_search_results(results)))
            print(f'  Step {step}: search("{query}") → {len(results)} results')

        elif action_type == "forecast":
            raw_p_yes = float(action["p_yes"])
            rationale = str(action["rationale"])
            p_yes = max(0.0, min(1.0, float(raw_p_yes)))
            traces.append(AgentTrace(step=step, action="forecast", llm_raw=response))
            print(f"  Step {step}: forecast → p_yes={p_yes:.2f}")
            return p_yes, rationale, traces, total_tokens
        else:
            _record_invalid_action(action_type, response, messages, traces, step)

    return 0.5, "Max steps reached", traces, total_tokens


# %% [markdown]
# ## Running the Agent
#
# The agent chooses each turn: search again, or commit to a probability. What follows is the
# recorded session, one line per turn.
#
# The recorded run does not reach a forecast. The model kept searching for evidence that the
# 2026 rate path had already been settled, found reporting that argued both ways, and used all
# five turns doing so. That is the ordinary failure of an agent under a budget, and it is why
# `run_react_agent` returns an explicit no-answer value instead of a probability: a caller that
# reads that value as a forecast would record a confident coin-flip where the agent said nothing.
# The mock run at the end of the notebook takes the other branch and commits.

# %%
if RUN_LIVE:
    search_client = create_search_client(LLM_PROVIDER)
    search_name = type(search_client).__name__
    # Wrap the client so every prompt/response is captured for the run trace.
    tracer = trace_llm(llm, label="react_agent")
    print(f"Search: {search_name}\n")
    print(f"Question: {question.question}\n")
    p_yes, rationale, traces, tokens = run_react_agent(
        tracer, search_client, question, max_steps=MAX_STEPS
    )
else:
    # Replay: rehydrate the pinned agent and reconstruct the step-by-step log
    # that run_react_agent prints during a live run.
    artifact = pinned_run.agent_artifacts()[0]
    p_yes, rationale, traces, tokens = (
        artifact.p_yes,
        artifact.rationale,
        artifact.traces,
        artifact.token_usage,
    )
    search_name = pinned_run.params.get("search_client", "replay (pinned trace)")
    print(f"Search: {search_name}\n")
    print(f"Question: {question.question}\n")
    for t in traces:
        if t.action == "search":
            print(f'  Step {t.step}: search("{t.query}") → {len(t.results)} results')
        elif t.action == "forecast":
            print(f"  Step {t.step}: forecast → p_yes={p_yes:.2f}")

# %%
committed_forecast = any(t.action == "forecast" for t in traces)
if committed_forecast:
    print("--- Forecast ---")
    print(f"p(YES) = {p_yes:.2f}")
    print(f"Rationale: {rationale[:300]}")
else:
    print("--- No forecast: step budget exhausted ---")
    print(f"Loop returned p(YES) = {p_yes:.2f} as its no-answer value")
    print(f"Reason: {rationale[:300]}")
print(f"Tokens: {tokens.total_tokens:,}")

# %% [markdown]
# ## Inspecting the Execution Trace
#
# Every step is captured as an `AgentTrace`. This is critical for **auditability**:
# in production, you need to know which queries were issued, what results came back,
# and how the LLM arrived at its forecast.

# %%
print(f"Total steps: {len(traces)}")
print(f"Token usage: {tokens.total_tokens:,} tokens\n")

for t in traces:
    if t.action == "search":
        print(f'Step {t.step} [SEARCH] query="{t.query}"')
        for r in t.results[:3]:
            print(f"  → {r.title[:70]}")
    elif t.action == "forecast":
        data = json.loads(t.llm_raw) if t.llm_raw else {}
        print(f"Step {t.step} [FORECAST] p_yes={data.get('p_yes', '?')}")
    print()

# %% [markdown]
# ## Persisting the Run
#
# A live run depends on three things that will not exist tomorrow: an open market question, the
# documents a search API returned that day, and a specific model version. Writing the question,
# the agent's steps, and the full prompt-and-reply conversation into one JSON file under
# `forecast_traces/` fixes all three. That file is what `RUN_LIVE = False` reads, which is why
# the printed session above is the same for every reader regardless of which provider or search
# backend they have configured, and why a run can be re-examined months later.

# %%
if RUN_LIVE:
    artifact = AgentForecastArtifact(
        agent_id="react_agent",
        p_yes=p_yes,
        rationale=rationale,
        traces=traces,
        token_usage=tokens,
        search_queries_made=sum(1 for t in traces if t.action == "search"),
        sources_consulted=sum(len(t.results) for t in traces),
    )
    run = RunTrace.capture(
        notebook="01_react_reasoning",
        provider=provider_name,
        question=question,
        params={
            "max_steps": MAX_STEPS,
            "warmup_response": response,
            "search_client": search_name,
        },
        agents=[artifact],
        llm_calls=tracer.calls,
        notes="ReAct agent on a live prediction-market question.",
    )
    trace_path = run.save()
    print(
        f"Saved {len(run.llm_calls)} model calls ({run.total_tokens():,} tokens) → {trace_path.name}"
    )
else:
    print(
        f"Replayed {len(pinned_run.llm_calls)} model calls "
        f"({pinned_run.total_tokens():,} tokens) from {PINNED_TRACE}"
    )

# %% [markdown]
# ### What the run record shows
#
# One limit on this evidence is worth naming before reading the summary. The saved search
# results carry no publication dates, so nothing in the record establishes that a document the
# agent read was available before the question opened. Evaluating a forecasting agent against
# resolved history needs search results that can be filtered by publication date and a policy
# for the ones that carry no date at all, which is what
# [`02_tool_contracts`](02_tool_contracts.ipynb) builds.

# %%
active_run = run if RUN_LIVE else pinned_run
search_steps = [trace for trace in traces if trace.action == "search"]
committed = any(trace.action == "forecast" for trace in traces)
n_searches = len(search_steps)
n_results = sum(len(trace.results) for trace in search_steps)
n_market_price_prompts = sum(
    1
    for call in active_run.llm_calls
    for message in call.get("messages", [])
    if "MARKET IMPLIED PROBABILITY" in message.get("content", "")
)
outcome = (
    f"""committed to $p_{{\\text{{yes}}}}={p_yes:.2f}$, reasoning: *{rationale}*."""
    if committed
    else f"""spent the whole {MAX_STEPS}-step budget on searches without ever emitting a """
    f"""forecast action, so the loop returned its no-answer value of """
    f"""$p_{{\\text{{yes}}}}={p_yes:.2f}$ and the note *{rationale}*. That value is the """
    """absence of a forecast, not a 50/50 judgement, and a caller must branch on it."""
)
display(
    Markdown(
        f"""**What this run did**: {n_searches} searches returning {n_results} documents. """
        f"""The agent then {outcome} """
        f"""Its prompts carried the market-implied probability {n_market_price_prompts} """
        """times, so the market quote printed near the top of this notebook was withheld """
        """from the agent and is a reference for the reader only."""
    )
)

# %% [markdown]
# ## Swapping the Backend
#
# Nothing in `run_react_agent` names a provider. It calls `complete_with_usage` on whatever
# object it was handed, so any class satisfying the two-method protocol can be substituted:
# `MockLLMClient` here, a local Ollama model, or a commercial API. The cell below runs the same
# loop against the mock client, whose replies are canned, and gives it three turns rather than
# five because the mock searches once and then commits.
#
# On the live path, `create_llm_client("")` picks the first provider whose key is present, and
# reading `LLM_PROVIDER` from the environment overrides that. So a continuous-integration run
# forces deterministic replies without editing the notebook:
#
# ```bash
# LLM_PROVIDER=mock uv run python 24_autonomous_agents/01_react_reasoning.py
# ```
#
# That variable only takes effect when `RUN_LIVE = True`; on the replay path no client is
# constructed at all.

# %%
mock_llm = MockLLMClient()
mock_search = create_search_client("mock")

p_mock, _, mock_traces, mock_tokens = run_react_agent(mock_llm, mock_search, question, max_steps=3)

print(f"Mock provider: {mock_llm.model_name}")
print(f"Mock steps: {len(mock_traces)}")
print(f"Mock tokens: {mock_tokens.total_tokens:,}")
print(f"Mock p(YES): {p_mock:.2f}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **One interface, any backend.** An agent that calls `complete_with_usage` and nothing else
#    can be moved between a free local model and a commercial API without an edit. Write the
#    loop against the protocol, not against a vendor's SDK.
# 2. **A small action space is what makes a loop auditable.** With two actions, every turn is
#    either a query you can re-issue or a probability you can score. Adding actions adds ways
#    for a run to go wrong that the record cannot explain.
# 3. **Model output is untrusted input.** Parse it against the declared schema, and hand a
#    malformed reply back for correction rather than letting it reach a tool.
# 4. **A budget needs a distinguishable exhaustion value.** A loop that returns a probability on
#    both success and failure gives the caller no way to tell a judgement from a timeout. Return
#    a value the caller must branch on, and say so in the type or the record.
# 5. **Record the conversation, not just the answer.** Prompts, replies and tool results in one
#    file are what make a run reproducible after the market has closed and the model has been
#    retired.
#
# **Known limitations of what is built here.** The loop keeps every message in the context
# window, so a long session eventually exceeds it and there is no summarisation or eviction. The
# search results carry no publication dates, so a run cannot be shown to be free of hindsight.
# There is one agent and one sample, so the probability has no dispersion around it. And nothing
# scores the forecast: an agent is only as good as the record of how its past probabilities
# resolved, which needs resolved questions and a proper scoring rule.
#
# **Next**: [`02_tool_contracts`](02_tool_contracts.ipynb) gives the search tool a typed
# contract, a publication-date filter, and a source policy.
