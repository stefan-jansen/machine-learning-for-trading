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

import sys

from utils.paths import get_chapter_dir

sys.path.insert(0, str(get_chapter_dir(24)))

import json
from datetime import date

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

# %% [markdown]
# **Optional dependencies** (for a real LLM and web search - the chapter runs in
# offline replay mode without them):
#
# ```bash
# uv pip install anthropic openai httpx        # cloud providers
# # Ollama (free, local) - install from https://ollama.com, then:
# #   ollama serve && ollama pull qwen3:8b
# ```
#
# Then set your provider in the repo-root `.env` (copy it first:
# `cp .env.example .env`). You set three things - **never a base URL**:
#
# - `LLM_PROVIDER` - `deepseek` | `openrouter` | `openai` | `anthropic` | `google` | `ollama` | `mock`
# - `LLM_API_KEY` - your key for that provider
# - `LLM_MODEL` - *optional*; blank uses the provider default
# - `TAVILY_API_KEY` - web search ([tavily.com](https://tavily.com))
#
# Live mode auto-loads `.env`. Missing live credentials raise an explicit error;
# select `mock` only for a labeled synthetic test fixture. Run `uv run python check_env.py` to confirm which
# model will be used.

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
# The agent chooses each turn: search again, or commit to a probability. What follows is the
# recorded session, one line per turn.
#
# On the default replay path the run does not reach a forecast. The model kept searching for
# evidence that the 2026 rate path had already been settled, found reporting that argued both
# ways, and used every turn doing so. That is the ordinary failure of an agent under a budget,
# and it is why `run_react_agent` returns an explicit no-answer value instead of a probability:
# a caller that reads that value as a forecast would record a confident coin-flip where the
# agent said nothing. The mock run at the end of the notebook takes the other branch and
# commits.

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

# %% [markdown]
# The June capture retains actual searches and model messages. It predates the new
# acceptance check; inspect its rationale and sources rather than assuming a
# validated status. The current loop rejects malformed or unsupported probabilities
# and permits one final synthesis turn only when evidence was retained. It can
# abstain. A citation makes support inspectable, not automatically correct.
#
# The mock example below is an explicitly synthetic flow test, not a model evaluation.

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
