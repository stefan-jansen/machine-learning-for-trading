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
# # The Research Agent
#
# **Docker image**: `ml4t`
#
# This notebook is the **core project** of Chapter 24's first half. It combines the
# provider abstraction (NB01), search tools (NB02), and state/quality gates (NB03)
# into a complete `ResearchAgent` that produces probability forecasts with
# structured metadata.
#
# The shared `agent_research.py` module carries the same validated parser and
# control-flow contract for NB06-NB08. This notebook exposes that contract step
# by step for inspection.
#
# **Learning Objectives**:
# - Build a complete ReAct-based research agent with structured output
# - Extract heuristic metadata: confidence, sentiment, key findings, evidence quality
# - Track token usage for cost analysis
# - Produce an `AgentForecastArtifact` with full provenance
#
# **Book Reference**: Chapter 24, Section 24.6 (Core Project: The Research Agent)
#
# **Prerequisites**: NB01 (providers), NB02 (tools), NB03 (state/gates).

# %%
"""The Research Agent: ReAct loop with structured output extraction."""

import re
import warnings
from dataclasses import dataclass, field
from datetime import date

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import polars as pl
from agent_fixtures import get_chapter_contested_question
from agent_observability import (
    TRACES_DIR,
    RunTrace,
    merge_calls,
    show_agent_timeline,
    trace_llm,
)
from agent_providers import ChatMessage, LLMClient, TokenUsage, create_llm_client
from agent_research import (
    extract_confidence as _shared_extract_confidence,
)
from agent_research import (
    parse_json as _shared_parse_json,
)
from agent_research import (
    validate_action as _shared_validate_action,
)
from agent_schemas import (
    AgentForecastArtifact,
    AgentTrace,
    EvidenceQuality,
    ForecastQuestion,
    Sentiment,
)
from agent_tools import (
    SearchClient,
    ToolExecutor,
    create_search_client,
    format_search_results,
)

from utils.style import COLORS, add_message_title, format_pct_axis

# %% tags=["parameters"]
# RUN_LIVE=False (the default) replays the pinned 2026-06-09 trace named below:
# the notebook reloads that saved run and makes no API calls, so the agent
# outputs are stable and match the chapter. Set RUN_LIVE=True (with API keys) to
# forecast a current question live; that path will not match the pinned capture.
RUN_LIVE = False
PINNED_TRACE = "04_research_agent_20260609T141730Z_b694ab4d0453.json"

LLM_PROVIDER = ""  # empty = auto-detect; "mock" for CI (live path only)
MAX_STEPS = 5
MAX_SEARCH_RESULTS = 5

# %% [markdown]
# ## Prompt Templates
#
# The system prompt and step prompt define the agent's behavior. They are shown
# inline so readers can see exactly how the LLM is instructed. These are the same
# prompts used by the chapter's forecasting implementation.

# %%
AGENT_SYSTEM_PROMPT = """\
You are a forecasting agent in a multi-agent forecasting system.

Your job:
1) Gather evidence by issuing web/news search queries when needed.
2) Then produce a binary probability forecast for the question.

You must follow the action schema exactly and output valid JSON only.
You must not browse prediction market prices unless they are explicitly provided."""

# %% [markdown]
# ### Step prompt
#
# Provides the question, optional market context, and the action schema.
# The agent must output exactly one JSON action per step.


# %%
def build_step_prompt(
    question: ForecastQuestion,
    market_price: float | None = None,
) -> str:
    """Format the step prompt with question context."""
    prompt = f"QUESTION:\n{question.question}\n\n"
    if question.description:
        prompt += f"MARKET CONTEXT:\n{question.description}\n\n"
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
# ## JSON Parsing
#
# LLMs sometimes wrap JSON in markdown code blocks or add trailing text.
# The shared parser accepts exactly one JSON object, including fenced output,
# and converts decoder failures or non-object JSON into an explicit parse-failure
# action. Repeated failures reach the step-budget sentinel.


# %%
def parse_json(raw: str) -> dict:
    """Delegate JSON-object parsing to the shared research-agent contract."""
    return _shared_parse_json(raw)


# %% [markdown]
# ### Action validation
#
# Search requires a nonempty string query. Forecast requires finite numeric
# `p_yes`, string rationale, and, when present, finite numeric confidence.
# Probabilities and confidence are normalized to $[0, 1]$ before execution.


# %%
def validate_action(action: dict) -> tuple[str, dict | None]:
    """Delegate schema validation to the shared research-agent contract."""
    return _shared_validate_action(action)


# %% [markdown]
# ## Rich Output Extraction
#
# After the agent produces a forecast, we extract additional metadata from the
# raw LLM output. Confidence, sentiment, and evidence quality are transparent
# heuristics, not calibrated estimates.

# %% [markdown]
# ### Confidence extraction
#
# If the forecast JSON includes a `confidence` field, we use it directly (clamped
# to $[0, 1]$). Otherwise, we compute an extremity heuristic:
# $\text{confidence} = 2 \cdot |p_{\text{yes}} - 0.5|$.


# %%
def extract_confidence(action: dict) -> float:
    """Delegate bounded confidence extraction to the shared contract."""
    return _shared_extract_confidence(action)


# %% [markdown]
# ### Sentiment extraction
#
# Maps $p_{\text{yes}}$ mechanically to a five-level sentiment scale.


# %%
def extract_sentiment(p_yes: float) -> Sentiment:
    """Infer sentiment from probability."""
    if p_yes > 0.8:
        return Sentiment.STRONGLY_BULLISH
    if p_yes > 0.6:
        return Sentiment.BULLISH
    if p_yes > 0.4:
        return Sentiment.NEUTRAL
    if p_yes > 0.2:
        return Sentiment.BEARISH
    return Sentiment.STRONGLY_BEARISH


# %% [markdown]
# ### Key findings and uncertainties
#
# Parses bullet points and numbered items from the rationale text, and identifies
# sentences containing uncertainty language.


# %%
def extract_key_findings(rationale: str) -> list[str]:
    """Extract bullet points and numbered items from rationale."""
    findings = []
    for line in rationale.split("\n"):
        line = line.strip()
        if re.match(r"^[-•*]\s+", line):
            findings.append(re.sub(r"^[-•*]\s+", "", line).strip())
        elif re.match(r"^\d+[.)]\s+", line):
            findings.append(re.sub(r"^\d+[.)]\s+", "", line).strip())
    return findings[:10]


# %% [markdown]
# ### Uncertainty extraction
#
# Identifies sentences containing uncertainty language in the rationale.


# %%
def extract_uncertainties(rationale: str) -> list[str]:
    """Identify sentences mentioning uncertainty."""
    uncertainty_words = {
        "uncertain",
        "unclear",
        "unknown",
        "risk",
        "caveat",
        "however",
        "but",
        "although",
    }
    sentences = re.split(r"[.!?]+", rationale)
    return [
        s.strip() for s in sentences if s.strip() and any(w in s.lower() for w in uncertainty_words)
    ][:5]


# %% [markdown]
# ### Evidence quality assessment
#
# This volume heuristic uses only query and result counts. It does not validate
# source credibility, independence, or point-in-time availability.


# %%
def assess_evidence_quality(sources_consulted: int, queries_made: int) -> EvidenceQuality:
    """Classify evidence volume from query and result counts."""
    if sources_consulted >= 10 and queries_made >= 3:
        return EvidenceQuality.HIGH
    if sources_consulted >= 5 or queries_made >= 2:
        return EvidenceQuality.MEDIUM
    return EvidenceQuality.LOW


# %% [markdown]
# ## ReAct step handlers
#
# The ReAct loop processes one of three action types per step: a search,
# a forecast, or an unrecognised action. Extracting each branch as a small
# free function keeps the agent class focused on iteration and assembly.


# %%
def _handle_search_step(
    executor,
    action: dict,
    cutoff,
    max_search_results: int,
    messages: list[ChatMessage],
    traces: list[AgentTrace],
    step: int,
    response: str,
) -> tuple[int, int]:
    """Execute a search action, append to messages + traces. Returns (queries_inc, sources_inc)."""
    query = action.get("query", "")
    results = executor.execute_search(query, max_results=max_search_results, cutoff_date=cutoff)
    traces.append(
        AgentTrace(step=step, action="search", query=query, results=results, llm_raw=response)
    )
    messages.append(ChatMessage(role="assistant", content=response))
    messages.append(ChatMessage(role="tool", content=format_search_results(results)))
    return 1, len(results)


# %% [markdown]
# ### Forecast handler


# %%
def _handle_forecast_step(
    action: dict,
    traces: list[AgentTrace],
    step: int,
    response: str,
) -> tuple[float, str, dict]:
    """Record a forecast action. Returns (p_yes, rationale, raw_action)."""
    p_yes = action["p_yes"]
    rationale = action["rationale"]
    traces.append(AgentTrace(step=step, action="forecast", llm_raw=response))
    return p_yes, rationale, action


# %% [markdown]
# ### Unknown-action handler


# %%
def _handle_unknown_step(
    action_type: str,
    response: str,
    messages: list[ChatMessage],
    traces: list[AgentTrace],
    step: int,
) -> None:
    """Record an unrecognized action and feed an error back to the LLM."""
    traces.append(AgentTrace(step=step, action=action_type, llm_raw=response))
    messages.append(ChatMessage(role="assistant", content=response))
    messages.append(
        ChatMessage(
            role="tool",
            content=f"Error: unrecognized action '{action_type}'. Use 'search' or 'forecast'.",
        )
    )


# %% [markdown]
# ## Loop State and Assembly
#
# Explicit loop state keeps iteration separate from artifact construction.


# %%
@dataclass
class _LoopResult:
    """Mutable state accumulated by one ReAct loop."""

    p_yes: float = 0.5
    rationale: str = ""
    raw_action: dict = field(default_factory=dict)
    traces: list[AgentTrace] = field(default_factory=list)
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    queries_made: int = 0
    sources_consulted: int = 0


# %% [markdown]
# ### Artifact assembly


# %%
def _assemble_artifact(agent, result: _LoopResult) -> AgentForecastArtifact:
    """Convert loop state into the structured forecast artifact."""
    return AgentForecastArtifact(
        agent_id=agent.agent_id,
        p_yes=result.p_yes,
        rationale=result.rationale,
        traces=result.traces,
        confidence=extract_confidence(result.raw_action),
        sentiment=extract_sentiment(result.p_yes),
        key_findings=extract_key_findings(result.rationale),
        evidence_quality=assess_evidence_quality(
            result.sources_consulted,
            result.queries_made,
        ),
        uncertainties=extract_uncertainties(result.rationale),
        token_usage=result.total_tokens,
        search_queries_made=result.queries_made,
        sources_consulted=result.sources_consulted,
    )


# %% [markdown]
# ### ReAct loop


# %%
def _run_research_loop(
    agent,
    question: ForecastQuestion,
    market_price: float | None,
) -> AgentForecastArtifact:
    cutoff = date.fromisoformat(question.cutoff_date) if question.cutoff_date else None
    messages = [
        ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT),
        ChatMessage(role="user", content=build_step_prompt(question, market_price)),
    ]
    result = _LoopResult()

    for step in range(1, agent.max_steps + 1):
        response, usage = agent.llm.complete_with_usage(messages, json_mode=True)
        result.total_tokens = result.total_tokens + usage
        parsed = parse_json(response)
        action_type, action = validate_action(parsed)

        if action_type == "search" and action is not None:
            dq, ds = _handle_search_step(
                agent.executor,
                action,
                cutoff,
                agent.max_search_results,
                messages,
                result.traces,
                step,
                response,
            )
            result.queries_made += dq
            result.sources_consulted += ds
        elif action_type == "forecast" and action is not None:
            result.p_yes, result.rationale, result.raw_action = _handle_forecast_step(
                action, result.traces, step, response
            )
            break
        else:
            _handle_unknown_step(action_type, response, messages, result.traces, step)
    else:
        result.rationale = "Max steps reached without forecast"
        result.traces.append(AgentTrace(step=agent.max_steps, action="forced_default"))

    return _assemble_artifact(agent, result)


# %% [markdown]
# ## The ResearchAgent Class
#
# The class owns the LLM, search executor, and iteration budget. Its `run()` method
# delegates to the explicit loop above. `agent_research.py` mirrors this behavior
# for downstream imports.


# %%
class ResearchAgent:
    """ReAct-based research agent producing probability forecasts.

    Implements the same loop as the AIA Forecaster's research agent:
    search for evidence, then forecast with rich metadata.
    """

    def __init__(
        self,
        llm: LLMClient,
        search: SearchClient | None = None,
        agent_id: str = "agent_0",
        max_steps: int = 5,
        max_search_results: int = 5,
    ) -> None:
        self.llm = llm
        self.executor = ToolExecutor(search=search)
        self.agent_id = agent_id
        self.max_steps = max_steps
        self.max_search_results = max_search_results

    def run(
        self,
        question: ForecastQuestion,
        market_price: float | None = None,
    ) -> AgentForecastArtifact:
        """Run the agent on a question and return its forecast artifact."""
        return _run_research_loop(self, question, market_price)


# %% [markdown]
# ## Running the Research Agent
#
# We run the agent on the pinned `CHAPTER_CONTESTED_QUESTION` from
# `agent_fixtures.py` (*"Will the Federal Reserve hike rates in 2026?"*, where
# the saved rationales point in opposite directions) and inspect the artifact.
# The whole forecasting arc forecasts this same pinned question (NB07 debates it,
# NB08 runs it through the full pipeline, NB10 ports it across frameworks); NB06
# uses the companion `CHAPTER_CLEAR_QUESTION`, a one-directional question on which
# the agents instead agree. The numbers are a timestamped live capture
# (provider `claude-sonnet-4`, Tavily search, 2026-06-09). By default the notebook
# *replays* that pinned run (`RUN_LIVE = False`): it reloads the saved artifacts
# and raw conversation and makes no API calls, so the outputs are stable. Set
# `RUN_LIVE = True` (with `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`) to forecast a
# current question live, which will not reproduce the pinned values.

# %%
if RUN_LIVE:
    llm = create_llm_client(LLM_PROVIDER)
    search = create_search_client(LLM_PROVIDER)

    # Wrap the client in a TracingLLMClient so every prompt and raw response is
    # captured for the run trace we persist at the end of the notebook.
    tracer_0 = trace_llm(llm, label="agent_0")
    agent = ResearchAgent(
        llm=tracer_0,
        search=search,
        agent_id="agent_0",
        max_steps=MAX_STEPS,
        max_search_results=MAX_SEARCH_RESULTS,
    )

    question = get_chapter_contested_question()
    provider_name = llm.model_name
    search_name = type(search).__name__
    artifact = agent.run(question)
else:
    # Replay: reload the pinned trace and rehydrate its two saved agents. The
    # first is the single-agent forecast inspected here; the second is the
    # preview pair shown later. Display cells consume these unchanged.
    pinned_run = RunTrace.load(TRACES_DIR / PINNED_TRACE)
    question = pinned_run.question_obj()
    provider_name = pinned_run.provider
    search_name = "replay (pinned trace)"
    replayed_artifacts = pinned_run.agent_artifacts()
    artifact = replayed_artifacts[0]

print(f"Mode: {'LIVE' if RUN_LIVE else 'REPLAY (pinned 2026-06-09 trace)'}")
print(f"Provider: {provider_name}")
print(f"Search: {search_name}")
print(f"Question: {question.question}\n")

# %% [markdown]
# ### Pinned trace contract
#
# The default path asserts the exact saved-run shape and its provenance limitations.

# %%
if not RUN_LIVE:
    assert pinned_run.notebook == "04_research_agent"
    assert len(replayed_artifacts) == 2
    replay_results = [
        result
        for saved_artifact in replayed_artifacts
        for trace in saved_artifact.traces
        for result in trace.results
    ]
    assert len(replay_results) == 40
    assert all(result.published is None for result in replay_results)
    assert not any("MARKET IMPLIED PROBABILITY" in str(call) for call in pinned_run.llm_calls)

# %% [markdown]
# The replay is a saved live capture, not a historical backtest. Its 40 search
# results have no publication dates, and the saved prompts contain no market-price
# field. The trace can reproduce the recorded model interaction but cannot establish
# point-in-time source availability.

# %% [markdown]
# ## Inspecting the Forecast Artifact
#
# The `AgentForecastArtifact` records the probability, rationale, evidence trail,
# and configured metadata fields.

# %%
print("=== Forecast ===")
print(f"Agent:       {artifact.agent_id}")
print(f"p(YES):      {artifact.p_yes:.2f}")
print(f"Confidence heuristic: {artifact.confidence:.2f}")
print(f"Sentiment heuristic:  {artifact.sentiment.value}")
print(f"Evidence volume class: {artifact.evidence_quality.value}")
print(f"Queries:     {artifact.search_queries_made}")
print(f"Sources:     {artifact.sources_consulted}")
print(f"Tokens:      {artifact.token_usage.total_tokens:,}")

# %%
print(f"\nRationale:\n{artifact.rationale[:400]}")

# %%
if artifact.key_findings:
    print("\nKey Findings:")
    for f in artifact.key_findings:
        print(f"  • {f}")

if artifact.uncertainties:
    print("\nUncertainties:")
    for u in artifact.uncertainties:
        print(f"  • {u}")

# %% [markdown]
# ## Execution Trace
#
# Every search query and its results are captured in the trace. `show_agent_timeline`
# from `agent_observability` renders the whole run in order: each query, the
# documents it returned (title, date, URL, and a snippet), and the forecast with
# the untruncated rationale. This is the per-agent observability view reused
# across NB06-NB08.

# %%
print(show_agent_timeline(artifact))

# %% [markdown]
# ## Tool Execution Audit
#
# The executor's independent log captures timing and provenance for every
# search call. Rendering it as a Polars DataFrame puts the query / status /
# duration in three sortable columns rather than a hand-aligned string
# table: the same audit data, in a form that downstream analysis code can
# read without parsing. (A live run also records per-call `duration_ms`; the
# replay path reconstructs the same query/status/result-count audit from the
# saved traces, since wall-clock timing is not part of the persisted record.)

# %%
if RUN_LIVE:
    audit_df = (
        pl.DataFrame(
            [
                {
                    "query": entry.args.get("query", "?"),
                    "status": entry.status,
                    "duration_ms": round(entry.duration_ms, 1),
                }
                for entry in agent.executor.execution_log
            ]
        )
        if agent.executor.execution_log
        else pl.DataFrame({"query": [], "status": [], "duration_ms": []})
    )
else:
    # Replay: reconstruct the search audit from the saved agent traces.
    search_steps = [t for t in artifact.traces if t.action == "search"]
    audit_df = (
        pl.DataFrame(
            [
                {
                    "query": t.query or "?",
                    "status": "ok",
                    "results": len(t.results),
                }
                for t in search_steps
            ]
        )
        if search_steps
        else pl.DataFrame({"query": [], "status": [], "results": []})
    )
audit_df

# %% [markdown]
# ## Agent Summary Format
#
# When multiple agents run in parallel (NB06), their outputs are summarized
# for the supervisor and debate stages. This is the format used downstream.


# %%
def format_agent_summary(a: AgentForecastArtifact) -> str:
    """Format an agent artifact as a summary for supervisor/debate prompts."""
    lines = [
        f"Agent: {a.agent_id}",
        f"Probability (p_yes): {a.p_yes:.2f}",
        f"Confidence: {a.confidence:.2f}",
        f"Rationale: {a.rationale[:200]}",
    ]
    if a.key_findings:
        lines.append("Key findings:")
        for f in a.key_findings[:3]:
            lines.append(f"  - {f}")
    return "\n".join(lines)


# %%
print(format_agent_summary(artifact))

# %% [markdown]
# ## Running Multiple Agents
#
# A preview of the multi-agent notebooks: the same agent class with different
# IDs produced different forecasts in this pinned capture. NB06 uses the same
# class on a different question and records a narrower spread. Comparing two
# traces shows the observed contrast, but it does not identify whether question
# framing, retrieval, or model sampling caused it.

# %%
if RUN_LIVE:
    tracer_1 = trace_llm(llm, label="agent_1")
    agent_b = ResearchAgent(llm=tracer_1, search=search, agent_id="agent_1", max_steps=MAX_STEPS)
    artifact_b = agent_b.run(question)
else:
    # Replay: the pinned trace's second saved agent is this preview run.
    artifact_b = replayed_artifacts[1]

print(f"Agent 0: p_yes={artifact.p_yes:.2f}, confidence={artifact.confidence:.2f}")
print(f"Agent 1: p_yes={artifact_b.p_yes:.2f}, confidence={artifact_b.confidence:.2f}")
probability_gap = abs(artifact.p_yes - artifact_b.p_yes)
print(f"\nDifference: {probability_gap:.2f}")

# %% [markdown]
# The shared probability scale makes the observed disagreement immediately visible.
# These are two saved model outputs, not estimates with statistical error bars.

# %%
agent_labels = [artifact.agent_id, artifact_b.agent_id]
agent_probabilities = [artifact.p_yes, artifact_b.p_yes]

fig, ax = plt.subplots()
bars = ax.bar(
    agent_labels,
    agent_probabilities,
    color=[COLORS["blue"], COLORS["copper"]],
    width=0.58,
)
ax.bar_label(bars, labels=[f"{value:.0%}" for value in agent_probabilities], padding=3)
ax.set_xlabel("Research Agent")
ax.set_ylabel("Probability of a 2026 Fed Rate Hike")
ax.set_ylim(0, max(agent_probabilities) + 0.10)
format_pct_axis(ax)
add_message_title(
    ax,
    f"Pinned agents disagree by {probability_gap:.0%}",
    subtitle="Two 2026-06-09 forecasts; search-result publication dates unavailable",
)
fig.tight_layout()
fig.show()
plt.show()

# %% [markdown]
# **Interpretation**: The pinned agents differ materially. Their rationales cite
# conflicting policy-rate levels, and none of the 40 saved search results has a
# publication date. The trace therefore demonstrates observable disagreement and
# the need for provenance checks; it does not identify the cause of the gap.

# %% [markdown]
# ## Persisting the Run Trace
#
# The artifact holds the structured forecast; the `TracingLLMClient`s wrapped
# around each agent hold the raw conversation. `RunTrace.capture` bundles both:
# the question, both agents' artifacts, and every prompt/response, into one JSON
# record under `forecast_traces/`, the same auditable format the multi-agent
# notebooks (NB06-NB08) write. Reload it with `RunTrace.load` to inspect the saved
# inputs and outputs, which is what the default `RUN_LIVE = False` path
# does above. A live run writes a fresh trace here; the default replay run reports
# the pinned trace it loaded rather than overwriting it.

# %%
if RUN_LIVE:
    run = RunTrace.capture(
        notebook="04_research_agent",
        provider=provider_name,
        question=question,
        params={"max_steps": MAX_STEPS, "max_search_results": MAX_SEARCH_RESULTS},
        agents=[artifact, artifact_b],
        notes="Single research agent plus a two-agent preview on the contested question.",
        llm_calls=merge_calls(tracer_0, tracer_1),
    )
    trace_path = run.save()
    print(
        f"Saved {len(run.llm_calls)} model calls "
        f"({run.total_tokens():,} tokens) -> {trace_path.relative_to(trace_path.parents[1])}"
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
# ## Key Takeaways
#
# 1. **ResearchAgent** combines the ReAct loop with structured output extraction;
#    heuristic confidence, sentiment, evidence quality, and uncertainty fields
#    accompany the forecast
# 2. **Structured artifacts**: `AgentForecastArtifact` records the configured run fields:
#    probability, reasoning, traces, and token usage
# 3. **Bounded parsing**: invalid JSON becomes an explicit failure action; production
#    systems still need retry and abstention policy
# 4. **Reusable**: This class is the building block for multi-agent (NB06),
#    debate (NB07), and full pipeline (NB08) notebooks
# 5. **Token tracking**: Every LLM call is metered for cost analysis
#
# **Next**: [`05_aggregation_math`](05_aggregation_math.ipynb), the mathematical
# foundation for combining probability estimates.
#
# **Book**: Section 24.6 discusses agent design patterns, including the trade-off
# between agent complexity and forecast calibration.
