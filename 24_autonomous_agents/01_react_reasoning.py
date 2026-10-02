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
import sys

from utils.paths import get_chapter_dir

sys.path.insert(0, str(get_chapter_dir(24)))

"""LLM Providers and the ReAct Loop — multi-provider agent reasoning."""

import json
import warnings
from datetime import date

warnings.filterwarnings("ignore")

from agent_fixtures import get_live_question
from agent_observability import TRACES_DIR, RunTrace, trace_llm
from agent_pipeline import validate_probabilities
from agent_providers import ChatMessage, MockLLMClient, TokenUsage, create_llm_client
from agent_schemas import AgentForecastArtifact, AgentTrace, ForecastQuestion
from agent_tools import ToolExecutor, create_search_client, format_search_results

# %% tags=["parameters"]
# RUN_LIVE=False (the default) replays a pinned claude-sonnet + Tavily run: the
# notebook reloads the saved trace named below, makes no API calls, and
# reproduces the captured ReAct session, so the outputs are stable and match the
# narrative. Set RUN_LIVE=True (with LLM_PROVIDER + LLM_API_KEY + TAVILY_API_KEY in .env) to
# fetch a fresh live question and run the agent against it; that path makes real
# calls and is not reproducible.
RUN_LIVE = False
PINNED_TRACE = "01_react_reasoning_20260615T191047Z_04eb6e7c603d.json"

LLM_PROVIDER = ""  # Blank reads .env; "mock" selects a synthetic fixture
MAX_STEPS = 6

# %% [markdown]
# **Optional dependencies** (for a real LLM and web search — the chapter runs in
# offline replay mode without them):
#
# ```bash
# uv pip install anthropic openai httpx        # cloud providers
# # Ollama (free, local) — install from https://ollama.com, then:
# #   ollama serve && ollama pull qwen3:8b
# ```
#
# Then set your provider in the repo-root `.env` (copy it first:
# `cp .env.example .env`). You set three things — **never a base URL**:
#
# - `LLM_PROVIDER` — `deepseek` | `openrouter` | `openai` | `anthropic` | `google` | `ollama` | `mock`
# - `LLM_API_KEY` — your key for that provider
# - `LLM_MODEL` — *optional*; blank uses the provider default
# - `TAVILY_API_KEY` — web search ([tavily.com](https://tavily.com))
#
# Live mode auto-loads `.env`. Missing live credentials raise an explicit error;
# select `mock` only for a labeled synthetic test fixture. Run `uv run python check_env.py` to confirm which
# model will be used.

# %% [markdown]
# ## The LLM Provider Protocol
#
# Every notebook in this chapter uses the same `LLMClient` protocol. The key insight is
# that agent logic should be **provider-agnostic** — the same ReAct loop works whether
# backed by a \$0 mock, a local Ollama model, or a commercial API.
#
# The protocol defines two methods:
#
# - `complete(messages) → str` — just the text
# - `complete_with_usage(messages) → (str, TokenUsage)` — text plus token counts
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
    llm = None
    provider_name = pinned_run.provider
    response = pinned_run.params.get("warmup_response", "")

print(f"Mode:     {'LIVE' if RUN_LIVE else 'REPLAY (pinned trace)'}")
print(f"Provider: {provider_name}")
print(f"Response: {response[:200]}")

# %% [markdown]
# ## The Forecasting Question
#
# We fetch a **live prediction market question** from Polymarket — an open
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
# - `{"action": "search", "query": "..."}` — search the web for evidence
# - `{"action": "forecast", "p_yes": 0.XX, "rationale": "..."}` — produce a probability
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
SYSTEM_PROMPT += (
    "\nCite a retained source URL or title in the rationale. Abstain if evidence is insufficient."
)

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
        "SEARCH BUDGET: search only when a specific fact is missing; do not "
        "reformulate a query you already ran. Once you have a base rate and the "
        "current signal, forecast.\n\n"
        "NEXT ACTION SCHEMA (output JSON only):\n"
        'If you need more info:\n{"action":"search","query":"..."}\n'
        "If you are ready to forecast:\n"
        '{"action":"forecast","p_yes":0.XX,"rationale":"short explanation '
        'grounded in evidence and base rates"}\n\n'
        'If evidence is insufficient: {"action":"abstain","rationale":"why"}\n'
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
# ### The ReAct loop
#
# Each iteration: the LLM decides what to do (search or forecast), we
# execute it, and feed the observation back. The loop terminates when the
# agent issues a `forecast` action or hits the step limit. The `p_yes`
# value must be a finite number in [0, 1], with a rationale identifying
# retained evidence. Invalid or unsupported output is rejected. The budget
# can end with abstention rather than a probability.


# %%
def run_react_agent(
    llm,
    search_client,
    question: ForecastQuestion,
    max_steps: int = 5,
    market_price: float | None = None,
) -> tuple[float | None, str, list[AgentTrace], TokenUsage]:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    cutoff, executor, messages = _init_react_session(question, search_client, market_price)
    traces, sources = [], []
    total_tokens = TokenUsage()
    reason = "Budget exhausted without a supported forecast"
    for step in range(max_steps + 1):
        synthesis_only = step == max_steps
        if synthesis_only:
            if not sources:
                break
            messages.append(
                ChatMessage(
                    role="user",
                    content="No more searches. Cite retained evidence and forecast, or abstain.",
                )
            )
        try:
            raw, usage = llm.complete_with_usage(messages, json_mode=True)
        except Exception as exc:
            reason = f"Model call failed ({type(exc).__name__})"
            traces.append(AgentTrace(step=step, action="provider_error", llm_raw=reason))
            break
        total_tokens += usage
        messages.append(ChatMessage(role="assistant", content=raw))
        try:
            action = json.loads(raw)
            if not isinstance(action, dict):
                raise ValueError("Action must be an object")
        except (ValueError, TypeError):
            action = {}
        kind = action.get("action", "invalid")
        if kind == "search" and not synthesis_only:
            query = action.get("query", "")
            results = executor.execute_search(query, max_results=5, cutoff_date=cutoff)
            sources.extend(results)
            traces.append(
                AgentTrace(step=step, action="search", query=query, results=results, llm_raw=raw)
            )
            messages.append(ChatMessage(role="tool", content=format_search_results(results)))
            print(f"Step {step}: search -> {len(results)} eligible results")
            continue
        traces.append(AgentTrace(step=step, action=str(kind), llm_raw=raw))
        if kind == "abstain":
            reason = str(action.get("rationale") or "Insufficient evidence")
            break
        try:
            if kind != "forecast":
                raise ValueError("Invalid action or exhausted search budget")
            if isinstance(action["p_yes"], bool):
                raise ValueError("Probability must be a number, not a boolean")
            probability = float(action["p_yes"])
            validate_probabilities([probability])
            rationale = action.get("rationale")
            if not isinstance(rationale, str) or not any(
                (s.url and s.url in rationale) or (s.title and s.title in rationale)
                for s in sources
            ):
                raise ValueError("Forecast needs a rationale identifying retained evidence")
        except (KeyError, TypeError, ValueError) as exc:
            reason = str(exc)
            messages.append(ChatMessage(role="tool", content=reason))
            continue
        return probability, rationale, traces, total_tokens
    return None, reason, traces, total_tokens


# %% [markdown]
# ## Running the Agent
#
# The agent decides when to search and when to forecast, based on the question
# context. In mock mode the search returns deterministic results; with a real LLM
# and Tavily, the reasoning adapts to actual web content.

# %%
if RUN_LIVE:
    search_client = create_search_client()
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
    if not any(t.action == "forecast" for t in traces):
        p_yes = None
        rationale = "No forecast action in this capture; the saved 0.5 is a legacy sentinel"
    search_name = pinned_run.params.get("search_client", "replay (pinned trace)")
    print(f"Search: {search_name}\n")
    print(f"Question: {question.question}\n")
    for t in traces:
        if t.action == "search":
            print(f'  Step {t.step}: search("{t.query}") → {len(t.results)} results')
        elif t.action == "forecast":
            print(f"  Step {t.step}: forecast → p_yes={p_yes:.2f}")

# %%
print("--- Forecast ---" if p_yes is not None else "--- Abstention ---")
print(f"p(YES) = {p_yes}")
print(f"Rationale: {rationale}")
print(f"Tokens: {tokens.total_tokens:,}")

# %% [markdown]
# ## Inspecting the Execution Trace
#
# Every step is captured as an `AgentTrace`. This is critical for **auditability** —
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
# A live run is a point-in-time capture — a live prediction-market question, the
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
        status="accepted" if p_yes is not None else "abstained",
        failure_reason=rationale if p_yes is None else None,
        execution_mode="synthetic" if "mock" in llm.model_name else "live",
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
# The June capture retains actual searches and model messages. It predates the new
# acceptance check; inspect its rationale and sources rather than assuming a
# validated status. The current loop rejects malformed or unsupported probabilities
# and permits one final synthesis turn only when evidence was retained. It can
# abstain. A citation makes support inspectable, not automatically correct.
#
# The mock example below is an explicitly synthetic flow test, not a model evaluation.

# %% [markdown]
# ## Provider Swapping
#
# The same `run_react_agent` works with any provider — agent logic is
# provider-agnostic. To switch, edit the repo-root **`.env`** (the notebook
# auto-loads it) and change one line:
#
# ```bash
# LLM_PROVIDER=openrouter     # deepseek | openrouter | openai | anthropic | google | ollama | mock
# LLM_API_KEY=...             # your key for that provider
# LLM_MODEL=z-ai/glm-4.6      # optional; blank = the provider default
# ```
#
# The base URL for each provider is predefined — you never set it. Then set
# `RUN_LIVE = True` above and re-run the cells. Tip: `uv run python check_env.py`
# prints the exact model that will run.

# %%
mock_llm = MockLLMClient()
mock_search = create_search_client("mock")

p_mock, _, mock_traces, mock_tokens = run_react_agent(mock_llm, mock_search, question, max_steps=3)

print(f"Mock provider: {mock_llm.model_name}")
print(f"Mock steps: {len(mock_traces)}")
print(f"Mock tokens: {mock_tokens.total_tokens:,}")
print(f"Synthetic p(YES): {p_mock}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Provider abstraction**: The `LLMClient` protocol decouples agent logic from
#    providers — same code works with mock, Ollama, or commercial APIs
# 2. **Explicit outcomes**: `search`, `forecast` and `abstain` keep the action space minimal —
#    the agent's job is to gather evidence and produce a supported probability; calibration is evaluated separately
# 3. **Structured traces**: `AgentTrace` captures every search query and its results,
#    separate from the LLM's context window
# 4. **Mock-first development**: Build and test with deterministic mocks, then swap
#    in real LLMs for evaluation
#
# **Next**: [`tool_contracts`](02_tool_contracts.ipynb) — the SearchClient protocol, Tavily integration,
# and domain policy enforcement.
