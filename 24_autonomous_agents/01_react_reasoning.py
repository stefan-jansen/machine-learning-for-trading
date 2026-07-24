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
# This notebook introduces the **LLM provider abstraction** and the **ReAct** (Reason + Act)
# reasoning pattern, the foundational building block for the AIA-inspired multi-agent
# forecasting system built throughout this chapter.
#
# **Learning Objectives**:
# - Understand the `LLMClient` protocol that decouples agent logic from LLM providers
# - Build a ReAct agent that searches the web and produces probability forecasts
# - Run the same agent with mock, local (Ollama), or commercial (Claude/GPT) backends
# - Capture `AgentTrace` steps and inspect the full action trace
#
# **Book Reference**: Chapter 24, Sections 24.1–24.2 (From Prediction Functions to Agentic
# Workflows; Cognitive Architectures)
#
# **Prerequisites**: None. This is the first notebook in the Chapter 24 workshop.

# %%
"""LLM Providers and the ReAct Loop - multi-provider agent reasoning."""

import json
import math
import warnings
from datetime import date

warnings.filterwarnings("ignore")

from agent_fixtures import get_live_question
from agent_observability import TRACES_DIR, RunTrace, trace_llm
from agent_providers import ChatMessage, MockLLMClient, TokenUsage, create_llm_client
from agent_schemas import AgentForecastArtifact, AgentTrace, ForecastQuestion
from agent_tools import ToolExecutor, create_search_client, format_search_results
from IPython.display import Markdown, display

# %% tags=["parameters"]
# RUN_LIVE=False (the default) replays a pinned claude-sonnet + Tavily run: the
# notebook reloads the saved trace named below, makes no API calls, and
# reproduces the captured ReAct session, so the outputs are stable and match the
# narrative. Set RUN_LIVE=True (with ANTHROPIC_API_KEY and TAVILY_API_KEY) to
# fetch a fresh live question and run the agent against it; that path makes real
# calls and is not reproducible.
RUN_LIVE = False
PINNED_TRACE = "01_react_reasoning_20260615T191047Z_04eb6e7c603d.json"

LLM_PROVIDER = ""  # empty = auto-detect; "mock" for CI (live path only)
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
# Every notebook in this chapter uses the same `LLMClient` protocol. Agent logic should
# be **provider-agnostic**: the same ReAct loop works whether
# backed by a \$0 mock, a local Ollama model, or a commercial API.
#
# The protocol defines two methods:
#
# - `complete(messages) → str`: just the text
# - `complete_with_usage(messages) → (str, TokenUsage)`: text plus token counts
#
# Provider selection is automatic: set `LLM_PROVIDER=mock` for deterministic testing,
# or let the factory auto-detect from available API keys.

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
# We fetch a **live prediction market question** from Polymarket, an open
# question that the LLM cannot answer from training data. This ensures the
# agent must search for current information and reason about genuine uncertainty.
#
# If the Polymarket API is unavailable (offline, CI), we fall back to a static
# demo question.

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
# (calling tools) in a loop. Our agent has exactly two actions:
#
# - `{"action": "search", "query": "..."}`: search the web for evidence
# - `{"action": "forecast", "p_yes": 0.XX, "rationale": "..."}`: produce a probability
#
# The system prompt and step prompt are shown inline so readers can see exactly how
# the LLM is instructed. This is the same prompt structure used by the AIA Forecaster.

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
# Each ReAct session starts with three pieces of state: the cutoff date (used
# to keep the search engine away from post-question evidence), a tool
# executor wrapping the search client, and the initial system + user
# messages. Pulling this out of the loop keeps the loop function focused on
# control flow.


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
# The agent decides when to search and when to forecast, based on the question
# context. In mock mode the search returns deterministic results; with a real LLM
# and Tavily, the reasoning adapts to actual web content.

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
print("--- Forecast ---")
print(f"p(YES) = {p_yes:.2f}")
print(f"Rationale: {rationale[:300]}")
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
# A live run is a point-in-time capture of a live prediction-market question, the
# Tavily documents available that day, and the model's reasoning over them.
# Bundling the question, the agent's traces, and the raw model conversation into
# one JSON record under `forecast_traces/` makes the session auditable and, more
# importantly, *replayable*: the default `RUN_LIVE = False` path above reloads
# this trace and reproduces the run with no API calls, so the chapter is stable
# regardless of which provider or search backend a reader has configured.

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
# The next cell derives the interpretation from the replayed trace. The returned
# probability is a failure sentinel when the loop exhausts its step budget, not a
# substantive forecast. The saved search results do not include publication dates,
# so this live capture should not be treated as a historical point-in-time backtest.

# %%
active_run = run if RUN_LIVE else pinned_run
search_steps = [trace for trace in traces if trace.action == "search"]
result_counts = [len(trace.results) for trace in search_steps]
market_context_calls = [
    message
    for call in active_run.llm_calls
    for message in call.get("messages", [])
    if "MARKET IMPLIED PROBABILITY" in message.get("content", "")
]
display(
    Markdown(
        f"""**Interpretation**: The replay records {len(search_steps)} search actions and """
        f"""{sum(result_counts)} returned results before the loop reached its """
        f"""{MAX_STEPS}-step budget. It therefore returns the neutral sentinel """
        f"""$p_{{\\text{{yes}}}}={p_yes:.2f}$ with the rationale *{rationale}*. """
        f"""The saved prompts contain {len(market_context_calls)} market-price fields, so the """
        """market quote shown above is an ex-post reference rather than an input to the agent."""
    )
)

# %% [markdown]
# ## Provider Swapping
#
# The same `run_react_agent` function works with any provider. To switch:
#
# ```bash
# # Mock (deterministic, no API calls)
# LLM_PROVIDER=mock uv run python 24_autonomous_agents/01_react_reasoning.py
#
# # Local Ollama
# LLM_PROVIDER=ollama uv run python 24_autonomous_agents/01_react_reasoning.py
#
# # Anthropic Claude
# LLM_PROVIDER=anthropic uv run python 24_autonomous_agents/01_react_reasoning.py
# ```

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
# 1. **Provider abstraction**: The `LLMClient` protocol decouples agent logic from
#    providers, so the same code works with mock, Ollama, or commercial APIs
# 2. **Two actions only**: `search` and `forecast` keep the action space minimal;
#    the agent's job is to gather evidence and produce a probability forecast
# 3. **Structured traces**: `AgentTrace` captures every search query and its results,
#    separate from the LLM's context window
# 4. **Mock-first development**: Build and test with deterministic mocks, then swap
#    in real LLMs for evaluation
#
# **Next**: [`02_tool_contracts`](02_tool_contracts.ipynb), which covers the SearchClient protocol, Tavily integration,
# and domain policy enforcement.
